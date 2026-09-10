// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Included after level_zero_loader.cpp defines native owners and its bounded
// completion policy. SYCL and oneMKL headers must be at translation-unit scope.
#if defined(THEROCK_MODULE_ENABLE_SGEMM) && THEROCK_MODULE_ENABLE_SGEMM
#if !defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO) ||                            \
    !defined(SYCL_EXT_ONEAPI_QUEUE_EMPTY) || !defined(SYCL_EXT_ONEAPI_PROD)
#error                                                                         \
    "Intel SGEMM requires Level Zero interoperability and bounded queue-query extensions"
#endif

namespace {

class ServiceOneMkl {
  static constexpr auto kSyclBackend = sycl::backend::ext_oneapi_level_zero;
  static constexpr const char *kProvider = "onemkl";

  struct AsyncFailure {
    void capture(const sycl::exception_list &errors) noexcept {
      try {
        std::lock_guard<std::mutex> lock(mutex);
        if (!first && errors.begin() != errors.end()) {
          first = *errors.begin();
        }
      } catch (...) {
        // Failure to preserve an asynchronous error cannot be reported as a
        // successful operation, nor can we assume queue completion here.
        std::fputs("FAIL backend=intel provider=onemkl async-error-capture "
                   "completion=unknown\n",
                   stderr);
        std::fflush(stderr);
        std::_Exit(EXIT_FAILURE);
      }
    }

    void rethrow() {
      std::exception_ptr error;
      {
        std::lock_guard<std::mutex> lock(mutex);
        error = first;
      }
      if (error) {
        std::rethrow_exception(error);
      }
    }

    std::mutex mutex;
    std::exception_ptr first;
  };

public:
  ServiceOneMkl(ze_driver_handle_t driver, ze_device_handle_t device,
                ze_context_handle_t context)
      : native_driver_(driver), native_device_(device),
        native_context_(context),
        async_failure_(std::make_shared<AsyncFailure>()) {}

  ~ServiceOneMkl() {
    drain();
    if (queue_) {
      try {
        check_async();
      } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL cleanup=oneMKL provider=onemkl error=%s\n",
                     error.what());
        cleanup_failed = true;
      } catch (...) {
        std::fputs("FAIL cleanup=oneMKL provider=onemkl unknown async error\n",
                   stderr);
        cleanup_failed = true;
      }
    }
  }

  ServiceOneMkl(const ServiceOneMkl &) = delete;
  ServiceOneMkl &operator=(const ServiceOneMkl &) = delete;

  std::string info() {
    initialize();
    using therock::module_validation::json_string;
    return "{\"schema_version\":1,\"kind\":\"blas-provider\","
           "\"scope\":\"loaded-provider\",\"vendor\":\"intel\","
           "\"provider\":\"onemkl\",\"library_version\":" +
           json_string(library_version_) +
           ",\"abi\":" + json_string(therock::module_contract::kSgemmAbi) +
           ",\"version\":" +
           std::to_string(therock::module_contract::kSgemmVersion) +
           ",\"contract_sha256\":" +
           json_string(therock::module_contract::kSgemmContractSha256) +
           ",\"capabilities\":[\"blas-provider-onemkl-v1\","
           "\"blas-sgemm-f32-nn-v1\"]}";
  }

  void sgemm(const void *a_base, const void *b_base, void *c_base,
             const therock::module_service::SgemmRequest &request) {
    if (!initialized_) {
      throw std::runtime_error("SGEMM provider has not been negotiated");
    }
    check_async();
    // Validate the native base allocations in the borrowed context before
    // forming matrix views. Raw Level Zero allocations remain owned by the
    // service; SYCL must recognize their allocation kind and exact device.
    require_device_allocation(a_base);
    require_device_allocation(b_base);
    require_device_allocation(c_base);
    const auto *a = static_cast<const float *>(a_base) + request.a_offset;
    const auto *b = static_cast<const float *>(b_base) + request.b_offset;
    auto *c = static_cast<float *>(c_base) + request.c_offset;
    try {
      // Explicit standard mode overrides environment settings that would allow
      // alternate reduced-precision implementations. Scalar values are passed
      // by value; native buffers stay alive through the completion check.
      oneapi::mkl::blas::column_major::gemm(
          *queue_, oneapi::mkl::transpose::nontrans,
          oneapi::mkl::transpose::nontrans, static_cast<int64_t>(request.m),
          static_cast<int64_t>(request.n), static_cast<int64_t>(request.k),
          request.alpha, a, static_cast<int64_t>(request.lda), b,
          static_cast<int64_t>(request.ldb), request.beta, c,
          static_cast<int64_t>(request.ldc),
          oneapi::mkl::blas::compute_mode::standard);
      drain();
      check_async();
    } catch (...) {
      // A synchronous library exception may follow partial submission and may
      // not return an event. Query the entire queue before unwinding buffers.
      drain();
      throw;
    }
  }

  void drain() noexcept {
    if (!queue_) {
      return;
    }
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::nanoseconds(kWaitTimeoutNs);
    try {
      // This is a nonblocking progress hint for runtimes that batch commands.
      // Queue emptiness covers host tasks and partial library submissions too;
      // it does not rely on a GEMM event having been returned successfully.
      queue_->ext_oneapi_prod();
      while (!queue_->ext_oneapi_empty()) {
        if (std::chrono::steady_clock::now() >= deadline) {
          fail_unknown_completion("queue deadline exceeded");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    } catch (const std::exception &error) {
      fail_unknown_completion(error.what());
    } catch (...) {
      fail_unknown_completion("queue completion query failed");
    }
  }

private:
  [[noreturn]] static void
  fail_unknown_completion(const char *reason) noexcept {
    std::fprintf(stderr,
                 "FAIL backend=intel provider=onemkl timeout_ns=%llu "
                 "completion=unknown error=%s; terminating without freeing "
                 "in-flight resources\n",
                 static_cast<unsigned long long>(kWaitTimeoutNs), reason);
    std::fflush(stderr);
    std::_Exit(EXIT_FAILURE);
  }

  void check_async() {
    queue_->throw_asynchronous();
    async_failure_->rethrow();
  }

  void require_device_allocation(const void *base) {
    if (!base ||
        sycl::get_pointer_type(base, *context_) != sycl::usm::alloc::device ||
        sycl::get_pointer_device(base, *context_) != *device_) {
      throw std::runtime_error(
          "oneMKL requires a native device allocation on the selected context "
          "and device");
    }
  }

  void initialize() {
    if (initialized_) {
      return;
    }
    if (initialization_attempted_) {
      throw std::runtime_error("oneMKL initialization previously failed");
    }
    initialization_attempted_ = true;
    static_assert(therock::module_contract::kSgemmEnabled,
                  "oneMKL requires an enabled generated contract");
    if (std::string(therock::module_contract::kSgemmProvider) != kProvider) {
      throw std::runtime_error(
          "Generated SGEMM provider differs from native backend");
    }
    const auto platform = sycl::make_platform<kSyclBackend>(native_driver_);
    device_ = std::make_unique<sycl::device>(
        sycl::make_device<kSyclBackend>(native_device_));
    if (device_->get_platform() != platform || !device_->is_gpu() ||
        sycl::get_native<kSyclBackend>(*device_) != native_device_ ||
        !device_->has(sycl::aspect::usm_device_allocations)) {
      throw std::runtime_error(
          "oneMKL SYCL device differs from the selected native GPU");
    }
    // Shared error state outlives every SYCL object and callback, including
    // partial setup. Construct each wrapper in an already-owned object so later
    // failures cannot bypass cleanup. Native context ownership never transfers.
    const sycl::async_handler handler =
        [failure = async_failure_](sycl::exception_list errors) {
          failure->capture(errors);
        };
    context_ = std::make_unique<sycl::context>(sycl::make_context<kSyclBackend>(
        {native_context_,
         {*device_},
         sycl::ext::oneapi::level_zero::ownership::keep},
        handler));
    if (sycl::get_native<kSyclBackend>(*context_) != native_context_) {
      throw std::runtime_error(
          "oneMKL SYCL context differs from the native worker context");
    }
    queue_ = std::make_unique<sycl::queue>(
        *context_, *device_, handler,
        sycl::property_list{sycl::property::queue::in_order{}});
    if (queue_->get_backend() != kSyclBackend ||
        sycl::get_native<kSyclBackend>(queue_->get_context()) !=
            native_context_ ||
        sycl::get_native<kSyclBackend>(queue_->get_device()) !=
            native_device_) {
      throw std::runtime_error(
          "oneMKL SYCL queue differs from the native worker context/device");
    }
    MKLVersion version{};
    mkl_get_version(&version);
    if (version.MajorVersion <= 0 || version.UpdateVersion < 0 ||
        version.PatchVersion < 0) {
      throw std::runtime_error("oneMKL reports an invalid runtime version");
    }
    library_version_ = std::to_string(version.MajorVersion) + "." +
                       std::to_string(version.UpdateVersion) + "." +
                       std::to_string(version.PatchVersion);
    check_async();
    initialized_ = true;
  }

  ze_driver_handle_t native_driver_;
  ze_device_handle_t native_device_;
  ze_context_handle_t native_context_;
  std::shared_ptr<AsyncFailure> async_failure_;
  std::unique_ptr<sycl::device> device_;
  std::unique_ptr<sycl::context> context_;
  std::unique_ptr<sycl::queue> queue_;
  std::string library_version_;
  bool initialization_attempted_ = false;
  bool initialized_ = false;
};

} // namespace
#endif
