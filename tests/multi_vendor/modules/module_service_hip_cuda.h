// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Included after loader.cpp's native owners and validation helpers. The service
// keeps their context, allocation, module, and bounded-completion policies
// while exposing persistent handles through the framed worker protocol only.
namespace {

class ServiceBackend {
  using NativeBuffer = ::Buffer;
  using NativeModule = ::Module;
  using NativeHostBuffer = ::HostBuffer;

public:
  struct Buffer {
    explicit Buffer(uint32_t requested_capacity)
        : storage(static_cast<size_t>(requested_capacity) * sizeof(float)),
          capacity(requested_capacity) {}
    NativeBuffer storage;
    uint32_t capacity;
  };

  struct Module {
    Module(const std::string &symbol, const std::string &path)
        : storage(path), function(storage.get_function(symbol)) {}
    NativeModule storage;
    FunctionHandle function;
  };

private:
  // Stream destruction drains before releasing the selected device scope.
  Device device_;
  Stream stream_;
#if defined(THEROCK_MODULE_ENABLE_SGEMM) && THEROCK_MODULE_ENABLE_SGEMM
  // This owner is released before the stream/context; State drains before it
  // releases any caller buffers, modules, or library workspace.
  ServiceBlas blas_{stream_.value};
#endif

  static Options device_options(const therock::module_service::Config &config) {
    if (config.device_id != 0) {
      throw std::runtime_error(
          "Device ID selection is only supported for Intel");
    }
    if (config.device < 0) {
      throw std::runtime_error("Service device index must be nonnegative");
    }
    therock::module_validation::validate_device_uuid(config.device_uuid);
    Options options;
    options.device = config.device;
    options.expected_arch = config.expected_arch;
    options.expected_device_uuid = config.device_uuid;
    return options;
  }

  static void validate_range(const Buffer &buffer, uint32_t offset,
                             size_t count) {
    if (offset > buffer.capacity || count > buffer.capacity - offset) {
      throw std::runtime_error("Service transfer exceeds buffer capacity");
    }
  }

  static DevicePointer pointer_at(Buffer &buffer, uint32_t offset) {
    const size_t bytes = static_cast<size_t>(offset) * sizeof(float);
#if defined(THEROCK_MODULE_NVIDIA)
    return buffer.storage.value + bytes;
#else
    return static_cast<unsigned char *>(buffer.storage.value) + bytes;
#endif
  }

  // Declared after each pinned staging allocation, so even a failed async API
  // call cannot free staging while a partially accepted copy may reference it.
  class StagingCompletion {
  public:
    explicit StagingCompletion(ServiceBackend &backend) : backend_(backend) {}
    ~StagingCompletion() {
      if (pending_) {
        backend_.synchronize();
      }
    }
    void finish() noexcept {
      backend_.synchronize();
      pending_ = false;
    }
    StagingCompletion(const StagingCompletion &) = delete;
    StagingCompletion &operator=(const StagingCompletion &) = delete;

  private:
    ServiceBackend &backend_;
    bool pending_ = true;
  };

public:
  explicit ServiceBackend(const therock::module_service::Config &config)
      : device_(device_options(config)) {}

  static bool supports_format(const std::string &format) {
#if defined(THEROCK_MODULE_NVIDIA)
    return format == "cubin" || format == "ptx";
#else
    return format == "hsaco";
#endif
  }

  std::unique_ptr<Buffer> allocate(uint32_t capacity) {
    if (capacity == 0 || capacity > therock::module_service::kMaxCapacity) {
      throw std::runtime_error("Service allocation capacity is out of range");
    }
    return std::make_unique<Buffer>(capacity);
  }

  std::unique_ptr<Module> load(const std::string &format,
                               const std::string &symbol,
                               const std::string &path) {
    if (!supports_format(format)) {
      throw std::runtime_error("Unsupported service payload format " + format);
    }
    if (symbol != "therock_module_saxpy" && symbol != "therock_module_relu") {
      throw std::runtime_error("Unsupported service kernel symbol " + symbol);
    }
    if (inspect_payload(path) != format) {
      throw std::runtime_error(
          "Service payload format does not match its declaration");
    }
    auto module = std::make_unique<Module>(symbol, path);
    // The worker may already own allocations. Check kernel geometry/shared
    // memory now, without reserving an unrelated fixture's future buffers.
    // Actual allocation requests are bounded and checked by the native API.
    device_.validate_kernel(module->function, 0);
    return module;
  }

  void write(Buffer &buffer, uint32_t offset,
             const std::vector<float> &values) {
    validate_range(buffer, offset, values.size());
    if (values.empty()) {
      synchronize();
      return;
    }
    NativeHostBuffer staging(values.size());
    StagingCompletion completion(*this);
    std::copy(values.begin(), values.end(), staging.begin());
    const DevicePointer destination = pointer_at(buffer, offset);
    const size_t bytes = values.size() * sizeof(float);
#if defined(THEROCK_MODULE_NVIDIA)
    GPU_CHECK(
        cuMemcpyHtoDAsync(destination, staging.data(), bytes, stream_.value));
#else
    GPU_CHECK(hipMemcpyAsync(destination, staging.data(), bytes,
                             hipMemcpyHostToDevice, stream_.value));
#endif
    completion.finish();
  }

  std::vector<float> read(Buffer &buffer, uint32_t offset, uint32_t count) {
    validate_range(buffer, offset, count);
    if (count == 0) {
      synchronize();
      return {};
    }
    NativeHostBuffer staging(count);
    StagingCompletion completion(*this);
    const DevicePointer source = pointer_at(buffer, offset);
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);
#if defined(THEROCK_MODULE_NVIDIA)
    GPU_CHECK(cuMemcpyDtoHAsync(staging.data(), source, bytes, stream_.value));
#else
    GPU_CHECK(hipMemcpyAsync(staging.data(), source, bytes,
                             hipMemcpyDeviceToHost, stream_.value));
#endif
    completion.finish();
    return std::vector<float>(staging.begin(), staging.end());
  }

  void launch(Module &module, Buffer &x, Buffer &y, Buffer &output, float alpha,
              uint32_t count) {
    if (count == 0 || count > therock::module_service::kMaxCount) {
      throw std::runtime_error("Service launch count is out of range");
    }
    validate_range(x, 0, count);
    validate_range(y, 0, count);
    validate_range(output, 0, count);
    if (&output == &x || &output == &y) {
      throw std::runtime_error("Service launch output must not alias an input");
    }
    void *arguments[] = {&x.storage.value, &y.storage.value,
                         &output.storage.value, &alpha, &count};
    const unsigned grid_size = (count + kGroupSize - 1) / kGroupSize;
    // Native APIs copy the argument values before returning. Buffer/module
    // handles remain server-owned until a drain precedes free/unload/close.
#if defined(THEROCK_MODULE_NVIDIA)
    GPU_CHECK(cuLaunchKernel(module.function, grid_size, 1, 1, kGroupSize, 1, 1,
                             0, stream_.value, arguments, nullptr));
#else
    GPU_CHECK(hipModuleLaunchKernel(module.function, grid_size, 1, 1,
                                    kGroupSize, 1, 1, 0, stream_.value,
                                    arguments, nullptr));
#endif
  }

#if defined(THEROCK_MODULE_ENABLE_SGEMM) && THEROCK_MODULE_ENABLE_SGEMM
  std::string sgemm_info() { return blas_.info(); }

  void sgemm(Buffer &a, Buffer &b, Buffer &c,
             const therock::module_service::SgemmRequest &request) {
    // All matrix spans and casts have already been checked by the common
    // protocol. Convert connection-owned offsets to native addresses only here.
#if defined(THEROCK_MODULE_NVIDIA)
    const auto a_pointer = reinterpret_cast<const float *>(
        static_cast<uintptr_t>(pointer_at(a, request.a_offset)));
    const auto b_pointer = reinterpret_cast<const float *>(
        static_cast<uintptr_t>(pointer_at(b, request.b_offset)));
    auto c_pointer = reinterpret_cast<float *>(
        static_cast<uintptr_t>(pointer_at(c, request.c_offset)));
#else
    const auto a_pointer =
        static_cast<const float *>(pointer_at(a, request.a_offset));
    const auto b_pointer =
        static_cast<const float *>(pointer_at(b, request.b_offset));
    auto c_pointer = static_cast<float *>(pointer_at(c, request.c_offset));
#endif
    blas_.sgemm(a_pointer, b_pointer, c_pointer, request);
  }
#endif

  void synchronize() noexcept {
#if defined(THEROCK_MODULE_NVIDIA)
    wait_for_completion([this] { return cuStreamQuery(stream_.value); },
                        "cuStreamQuery");
#else
    wait_for_completion([this] { return hipStreamQuery(stream_.value); },
                        "hipStreamQuery");
#endif
  }

  static bool cleanup_ok() { return !cleanup_failed; }
};

} // namespace
