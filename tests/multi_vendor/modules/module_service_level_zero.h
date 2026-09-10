// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "module_service_onemkl.h"

// Included by level_zero_loader.cpp after its anonymous namespace. Reuse its
// native owners and validation helpers while keeping the protocol independent
// of Level Zero types. These aliases avoid the service's nested handle names.
namespace {

using ServiceNativeAllocation = Allocation;
using ServiceNativeModule = Module;
using ServiceNativeKernel = Kernel;

class ServiceBackend {
public:
  struct Buffer {
    explicit Buffer(ze_context_handle_t context, uint32_t size)
        : allocation(context), capacity(size) {}
    ServiceNativeAllocation allocation;
    uint32_t capacity;
  };

  struct Module {
    ServiceNativeModule module{"zeModuleDestroy"};
    ServiceNativeKernel kernel{"zeKernelDestroy"};
  };

  explicit ServiceBackend(const therock::module_service::Config &config)
      : context_("zeContextDestroy"), queue_("zeCommandQueueDestroy") {
    if (config.expected_arch != "spirv" || config.device_id == 0 ||
        config.device_id > 0xffff || config.device < 0) {
      throw std::runtime_error(
          "Intel service requires spirv, a nonzero 16-bit device ID, and "
          "a nonnegative device index");
    }
    therock::module_validation::validate_device_uuid(config.device_uuid);
    Options options;
    options.device = static_cast<uint32_t>(config.device);
    options.expected_device_id = config.device_id;
    options.check_device_id = true;
    options.expected_device_uuid = config.device_uuid;
    selected_ = select_device(options);

    compute_.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
    ZE_CHECK(zeDeviceGetComputeProperties(selected_.device, &compute_));
    constexpr auto max_count = therock::module_service::kMaxCount;
    constexpr size_t max_bytes =
        therock::module_service::kMaxCapacity * sizeof(float);
    if (compute_.maxTotalGroupSize < kGroupSize ||
        compute_.maxGroupSizeX < kGroupSize || compute_.maxGroupSizeY < 1 ||
        compute_.maxGroupSizeZ < 1 ||
        compute_.maxGroupCountX < (max_count + kGroupSize - 1) / kGroupSize ||
        compute_.maxGroupCountY < 1 || compute_.maxGroupCountZ < 1 ||
        selected_.properties.maxMemAllocSize < max_bytes) {
      throw std::runtime_error(
          "Device does not support the module service geometry or allocation");
    }
    module_properties_.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
    ZE_CHECK(
        zeDeviceGetModuleProperties(selected_.device, &module_properties_));
    queue_ordinal_ = select_queue_group(selected_.device);
    memory_ordinal_ = select_memory(selected_.device, max_bytes);
    ze_context_desc_t context_desc{};
    context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    ZE_CHECK(zeContextCreate(selected_.driver, &context_desc, &context_.value));
    ze_command_queue_desc_t queue_desc{};
    queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    queue_desc.ordinal = queue_ordinal_;
    queue_desc.index = 0;
    queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ZE_CHECK(zeCommandQueueCreate(context_.value, selected_.device, &queue_desc,
                                  &queue_.value));
  }

  ~ServiceBackend() { synchronize(); }
  ServiceBackend(const ServiceBackend &) = delete;
  ServiceBackend &operator=(const ServiceBackend &) = delete;

  static bool supports_format(const std::string &format) {
    return format == "spirv";
  }

  std::unique_ptr<Buffer> allocate(uint32_t capacity) {
    if (capacity == 0 || capacity > therock::module_service::kMaxCapacity) {
      throw std::runtime_error("Invalid module service buffer capacity");
    }
    auto buffer = std::make_unique<Buffer>(context_.value, capacity);
    buffer->allocation.allocate_device(selected_.device, memory_ordinal_,
                                       capacity * sizeof(float));
    return buffer;
  }

  std::unique_ptr<Module> load(const std::string &format,
                               const std::string &symbol,
                               const std::string &path) {
    if (!supports_format(format)) {
      throw std::runtime_error("Intel module service requires SPIR-V");
    }
    if (symbol != "therock_module_saxpy" && symbol != "therock_module_relu") {
      throw std::runtime_error("Unsupported module service symbol");
    }
    const auto words = read_spirv(path);
    const uint32_t version =
        ZE_MAKE_VERSION((words[1] >> 16) & 0xff, (words[1] >> 8) & 0xff);
    if (module_properties_.spirvVersionSupported < version) {
      throw std::runtime_error(
          "Device does not support the module SPIR-V version");
    }
    auto module = std::make_unique<Module>();
    {
      BuildLog log("zeModuleBuildLogDestroy");
      ze_module_desc_t desc{};
      desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
      desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
      desc.inputSize = words.size() * sizeof(uint32_t);
      desc.pInputModule = reinterpret_cast<const uint8_t *>(words.data());
      const ze_result_t result =
          zeModuleCreate(context_.value, selected_.device, &desc,
                         &module->module.value, &log.value);
      print_build_log(log.value);
      ZE_CHECK(result);
    }
    ze_kernel_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    desc.pKernelName = symbol.c_str();
    ZE_CHECK(
        zeKernelCreate(module->module.value, &desc, &module->kernel.value));
    ze_kernel_properties_t properties{};
    properties.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
    ZE_CHECK(zeKernelGetProperties(module->kernel.value, &properties));
    if (properties.numKernelArgs != 5 ||
        (properties.requiredGroupSizeX &&
         properties.requiredGroupSizeX != kGroupSize) ||
        (properties.requiredGroupSizeY && properties.requiredGroupSizeY != 1) ||
        (properties.requiredGroupSizeZ && properties.requiredGroupSizeZ != 1) ||
        properties.localMemSize > compute_.maxSharedLocalMemory) {
      throw std::runtime_error(
          "Kernel does not match the module service ABI or group geometry");
    }
    ZE_CHECK(zeKernelSetGroupSize(module->kernel.value, kGroupSize, 1, 1));
    return module;
  }

  void write(Buffer &buffer, uint32_t offset,
             const std::vector<float> &values) {
    require_range(buffer, offset, values.size());
    if (values.empty()) {
      return;
    }
    const size_t bytes = values.size() * sizeof(float);
    ServiceNativeAllocation staging(context_.value);
    staging.allocate_host(bytes);
    std::memcpy(staging.value, values.data(), bytes);
    void *destination = static_cast<float *>(buffer.allocation.value) + offset;
    execute([&](ze_command_list_handle_t list) {
      ZE_CHECK(zeCommandListAppendMemoryCopy(list, destination, staging.value,
                                             bytes, nullptr, 0, nullptr));
    });
  }

  std::vector<float> read(Buffer &buffer, uint32_t offset, uint32_t count) {
    require_range(buffer, offset, count);
    if (count == 0) {
      return {};
    }
    const size_t bytes = count * sizeof(float);
    ServiceNativeAllocation staging(context_.value);
    staging.allocate_host(bytes);
    const void *source =
        static_cast<const float *>(buffer.allocation.value) + offset;
    execute([&](ze_command_list_handle_t list) {
      ZE_CHECK(zeCommandListAppendMemoryCopy(list, staging.value, source, bytes,
                                             nullptr, 0, nullptr));
    });
    std::vector<float> result(count);
    std::memcpy(result.data(), staging.value, bytes);
    return result;
  }

  void launch(Module &module, Buffer &x, Buffer &y, Buffer &output, float alpha,
              uint32_t count) {
    if (count == 0 || count > therock::module_service::kMaxCount ||
        &output == &x || &output == &y) {
      throw std::runtime_error("Invalid module service launch range or alias");
    }
    require_range(x, 0, count);
    require_range(y, 0, count);
    require_range(output, 0, count);
    const auto kernel = module.kernel.value;
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(void *),
                                      &x.allocation.value));
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 1, sizeof(void *),
                                      &y.allocation.value));
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 2, sizeof(void *),
                                      &output.allocation.value));
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 3, sizeof(alpha), &alpha));
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 4, sizeof(count), &count));
    const ze_group_count_t groups{(count + kGroupSize - 1) / kGroupSize, 1, 1};
    execute([&](ze_command_list_handle_t list) {
      ZE_CHECK(zeCommandListAppendLaunchKernel(list, kernel, &groups, nullptr,
                                               0, nullptr));
    });
  }

#if defined(THEROCK_MODULE_ENABLE_SGEMM) && THEROCK_MODULE_ENABLE_SGEMM
  std::string sgemm_info() {
    if (!blas_) {
      blas_ = std::make_unique<ServiceOneMkl>(selected_.driver,
                                              selected_.device, context_.value);
    }
    return blas_->info();
  }

  void sgemm(Buffer &a, Buffer &b, Buffer &c,
             const therock::module_service::SgemmRequest &request) {
    if (!blas_) {
      throw std::runtime_error("SGEMM provider has not been negotiated");
    }
    // Native module/copy operations already complete before acknowledgment.
    // The explicit handoff also covers future queued native operations. The
    // provider drains its separate SYCL queue before returning, preserving one
    // logical ordered worker stream without importing a regular Level Zero
    // command queue into the Xe2 SYCL adapter.
    synchronize();
    blas_->sgemm(a.allocation.value, b.allocation.value, c.allocation.value,
                 request);
  }
#endif

  void synchronize() noexcept {
#if defined(THEROCK_MODULE_ENABLE_SGEMM) && THEROCK_MODULE_ENABLE_SGEMM
    if (blas_) {
      // State calls this before freeing buffers or modules, including when a
      // provider call threw after only partially submitting its operation.
      blas_->drain();
    }
#endif
    if (!queue_.value) {
      return;
    }
    const ze_result_t result =
        zeCommandQueueSynchronize(queue_.value, kWaitTimeoutNs);
    if (result != ZE_RESULT_SUCCESS) {
      std::fprintf(
          stderr,
          "FAIL backend=intel service queue result=0x%08x completion=unknown; "
          "terminating without freeing in-flight resources\n",
          static_cast<unsigned>(result));
      std::fflush(nullptr);
      std::_Exit(EXIT_FAILURE);
    }
  }

  static bool cleanup_ok() { return !cleanup_failed; }

private:
  static void require_range(const Buffer &buffer, uint32_t offset,
                            size_t count) {
    if (offset > buffer.capacity || count > buffer.capacity - offset) {
      throw std::runtime_error("Module service buffer range is out of bounds");
    }
  }

  template <typename Record> void execute(Record record) {
    CommandList list("zeCommandListDestroy");
    ze_command_list_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    desc.commandQueueGroupOrdinal = queue_ordinal_;
    ZE_CHECK(zeCommandListCreate(context_.value, selected_.device, &desc,
                                 &list.value));
    // Staging memory belongs to the caller. Submission unwinds before this
    // list and before that memory on every partial-submission failure.
    Submission submission(queue_.value);
    record(list.value);
    // Publish operation writes before a later request consumes this buffer.
    // The barrier covers all earlier commands and global device memory.
    ZE_CHECK(zeCommandListAppendBarrier(list.value, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListClose(list.value));
    submission.execute(list.value);
    submission.synchronize();
    // This bounded backend deliberately completes each operation before ACK.
    // The persistent service guarantees ordering, not asynchronous overlap.
  }

  Candidate selected_{};
  ze_device_compute_properties_t compute_{};
  ze_device_module_properties_t module_properties_{};
  uint32_t queue_ordinal_ = 0;
  uint32_t memory_ordinal_ = 0;
  Context context_;
  Queue queue_;
#if defined(THEROCK_MODULE_ENABLE_SGEMM) && THEROCK_MODULE_ENABLE_SGEMM
  // Destroy the SYCL queue/context wrappers before the native queue/context.
  std::unique_ptr<ServiceOneMkl> blas_;
#endif
};

} // namespace
