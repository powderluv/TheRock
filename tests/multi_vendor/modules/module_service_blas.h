// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Included after loader.cpp defines its context, stream, and completion owners.
// BLAS headers are included at translation-unit scope by loader.cpp. The linked
// provider is a runtime dependency even though its handle is initialized
// lazily.
#if defined(THEROCK_MODULE_ENABLE_SGEMM) && THEROCK_MODULE_ENABLE_SGEMM
namespace {

class ServiceBlas {
#if defined(THEROCK_MODULE_NVIDIA)
  using Handle = cublasHandle_t;
  using Status = cublasStatus_t;
  static constexpr Status kBlasSuccess = CUBLAS_STATUS_SUCCESS;
  static constexpr const char *kProvider = "cublas";
#else
  using Handle = rocblas_handle;
  using Status = rocblas_status;
  static constexpr Status kBlasSuccess = rocblas_status_success;
  static constexpr const char *kProvider = "rocblas";
#endif

public:
  explicit ServiceBlas(StreamHandle stream) : stream_(stream) {}

  ~ServiceBlas() {
    if (!handle_) {
      return;
    }
    // Also covers successful library creation followed by failed configuration
    // or submission. Unknown completion exits before provider workspace is
    // freed.
    drain();
#if defined(THEROCK_MODULE_NVIDIA)
    const Status status = cublasDestroy(handle_);
    constexpr const char *operation = "cublasDestroy";
#else
    const Status status = rocblas_destroy_handle(handle_);
    constexpr const char *operation = "rocblas_destroy_handle";
#endif
    if (status != kBlasSuccess) {
      std::fprintf(stderr, "FAIL cleanup=%s provider=%s status=%d\n", operation,
                   kProvider, static_cast<int>(status));
      cleanup_failed = true;
    }
  }

  ServiceBlas(const ServiceBlas &) = delete;
  ServiceBlas &operator=(const ServiceBlas &) = delete;

  std::string info() {
    initialize();
    using therock::module_validation::json_string;
    const std::string provider_capability =
        std::string("blas-provider-") + kProvider + "-v1";
    return "{\"schema_version\":1,\"kind\":\"blas-provider\","
           "\"scope\":\"loaded-provider\",\"vendor\":" +
           json_string(kBackend) + ",\"provider\":" + json_string(kProvider) +
           ",\"library_version\":" + json_string(library_version_) +
           ",\"abi\":" + json_string(therock::module_contract::kSgemmAbi) +
           ",\"version\":" +
           std::to_string(therock::module_contract::kSgemmVersion) +
           ",\"contract_sha256\":" +
           json_string(therock::module_contract::kSgemmContractSha256) +
           ",\"capabilities\":[" + json_string(provider_capability) +
           ",\"blas-sgemm-f32-nn-v1\"]}";
  }

  void sgemm(const float *a, const float *b, float *c,
             const therock::module_service::SgemmRequest &request) {
    if (!initialized_) {
      throw std::runtime_error("SGEMM provider has not been negotiated");
    }
    // The protocol checks dimensions, strides, offsets, capacities, and aliases
    // before reaching this binding. Host pointer mode consumes alpha/beta
    // during the call; the matrices and library-owned workspace remain alive
    // until the stream completes. A successful return may acknowledge queued
    // computation.
#if defined(THEROCK_MODULE_NVIDIA)
    check(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                      static_cast<int>(request.m), static_cast<int>(request.n),
                      static_cast<int>(request.k), &request.alpha, a,
                      static_cast<int>(request.lda), b,
                      static_cast<int>(request.ldb), &request.beta, c,
                      static_cast<int>(request.ldc)),
          "cublasSgemm");
#else
    check(rocblas_sgemm(handle_, rocblas_operation_none, rocblas_operation_none,
                        static_cast<rocblas_int>(request.m),
                        static_cast<rocblas_int>(request.n),
                        static_cast<rocblas_int>(request.k), &request.alpha, a,
                        static_cast<rocblas_int>(request.lda), b,
                        static_cast<rocblas_int>(request.ldb), &request.beta, c,
                        static_cast<rocblas_int>(request.ldc)),
          "rocblas_sgemm");
#endif
  }

private:
  static void check(Status status, const char *operation) {
    if (status != kBlasSuccess) {
      throw std::runtime_error(
          std::string(operation) + " provider=" + kProvider +
          " status=" + std::to_string(static_cast<int>(status)));
    }
  }

  void drain() noexcept {
#if defined(THEROCK_MODULE_NVIDIA)
    wait_for_completion([this] { return cuStreamQuery(stream_); },
                        "cuStreamQuery before cublasDestroy");
#else
    wait_for_completion([this] { return hipStreamQuery(stream_); },
                        "hipStreamQuery before rocblas_destroy_handle");
#endif
  }

  void initialize() {
    if (initialized_) {
      return;
    }
    if (handle_) {
      throw std::runtime_error(
          "SGEMM provider initialization previously failed");
    }
    static_assert(therock::module_contract::kSgemmEnabled,
                  "BLAS support requires an enabled generated contract");
    if (std::string(therock::module_contract::kSgemmProvider) != kProvider) {
      throw std::runtime_error(
          "Generated SGEMM provider differs from native backend");
    }
    // Store ownership immediately after create succeeds. Configuration failures
    // leave the handle in this already-constructed owner, so State can drain
    // all work before releasing buffers and then this provider during fatal
    // cleanup.
#if defined(THEROCK_MODULE_NVIDIA)
    Handle created{};
    check(cublasCreate(&created), "cublasCreate");
    handle_ = created;
    if (!handle_) {
      throw std::runtime_error("cublasCreate returned an empty handle");
    }
    check(cublasSetStream(handle_, stream_), "cublasSetStream");
    check(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST),
          "cublasSetPointerMode");
    check(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH),
          "cublasSetMathMode");
    check(cublasSetAtomicsMode(handle_, CUBLAS_ATOMICS_NOT_ALLOWED),
          "cublasSetAtomicsMode");
    int version = 0;
    check(cublasGetVersion(handle_, &version), "cublasGetVersion");
    if (version <= 0) {
      throw std::runtime_error("cuBLAS reports an invalid runtime version");
    }
    library_version_ = std::to_string(version);
#else
    Handle created{};
    check(rocblas_create_handle(&created), "rocblas_create_handle");
    handle_ = created;
    if (!handle_) {
      throw std::runtime_error(
          "rocblas_create_handle returned an empty handle");
    }
    check(rocblas_set_stream(handle_, stream_), "rocblas_set_stream");
    check(rocblas_set_pointer_mode(handle_, rocblas_pointer_mode_host),
          "rocblas_set_pointer_mode");
    check(rocblas_set_math_mode(handle_, rocblas_default_math),
          "rocblas_set_math_mode");
    check(rocblas_set_atomics_mode(handle_, rocblas_atomics_not_allowed),
          "rocblas_set_atomics_mode");
    size_t version_size = 0;
    check(rocblas_get_version_string_size(&version_size),
          "rocblas_get_version_string_size");
    if (version_size == 0 || version_size > 4096) {
      throw std::runtime_error(
          "rocBLAS reports an invalid runtime version size");
    }
    std::vector<char> version(version_size, '\0');
    check(rocblas_get_version_string(version.data(), version.size()),
          "rocblas_get_version_string");
    const auto end = std::find(version.begin(), version.end(), '\0');
    if (end == version.begin() || end == version.end()) {
      throw std::runtime_error(
          "rocBLAS reports an invalid runtime version string");
    }
    library_version_.assign(version.begin(), end);
#endif
    initialized_ = true;
  }

  StreamHandle stream_;
  Handle handle_{};
  std::string library_version_;
  bool initialized_ = false;
};

} // namespace
#endif
