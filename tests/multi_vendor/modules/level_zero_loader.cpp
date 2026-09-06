// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <level_zero/ze_api.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kIntelVendor = 0x8086;
constexpr uint32_t kGroupSize = 128;
constexpr uint32_t kGuardCount = 17;
constexpr float kGuardValue = -12345.5f;
constexpr uint32_t kSizes[] = {1, 127, 128, 129, 4099, 65539};
constexpr uint64_t kWaitTimeoutNs = 30ULL * 1000 * 1000 * 1000;
constexpr size_t kMaxPayloadBytes = 64ULL * 1024 * 1024;
static_assert(sizeof(void *) == 8, "The module ABI requires 64-bit pointers");
bool cleanup_failed = false;

std::string error_string(ze_result_t result) {
  const char *name = "Level Zero error";
  switch (result) {
  case ZE_RESULT_SUCCESS:
    name = "ZE_RESULT_SUCCESS";
    break;
  case ZE_RESULT_NOT_READY:
    name = "ZE_RESULT_NOT_READY";
    break;
  case ZE_RESULT_ERROR_UNINITIALIZED:
    name = "ZE_RESULT_ERROR_UNINITIALIZED";
    break;
  case ZE_RESULT_ERROR_DEVICE_LOST:
    name = "ZE_RESULT_ERROR_DEVICE_LOST";
    break;
  case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
    name = "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
    break;
  case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
    name = "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
    break;
  case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
    name = "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
    break;
  case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
    name = "ZE_RESULT_ERROR_MODULE_LINK_FAILURE";
    break;
  case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
    name = "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
    break;
  case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
    name = "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
    break;
  case ZE_RESULT_ERROR_INVALID_ARGUMENT:
    name = "ZE_RESULT_ERROR_INVALID_ARGUMENT";
    break;
  case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
    name = "ZE_RESULT_ERROR_INVALID_KERNEL_NAME";
    break;
  case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
    name = "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY";
    break;
  default:
    break;
  }
  char number[16]{};
  std::snprintf(number, sizeof(number), "0x%08x",
                static_cast<unsigned>(result));
  return std::string(name) + " (" + number + ")";
}

void check(ze_result_t result, const char *operation, int line) {
  if (result != ZE_RESULT_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " at line " +
                             std::to_string(line) + ": " +
                             error_string(result));
  }
}
#define ZE_CHECK(operation) check((operation), #operation, __LINE__)

void check_cleanup(ze_result_t result, const char *operation) noexcept {
  if (result != ZE_RESULT_SUCCESS) {
    std::fprintf(stderr, "FAIL cleanup=%s result=0x%08x\n", operation,
                 static_cast<unsigned>(result));
    cleanup_failed = true;
  }
}

// API calls initialize already-constructed owners, so partial setup failures
// destroy every previously created object in reverse dependency order.
template <typename T, auto Destroy> class Handle {
public:
  explicit Handle(const char *destroy_name) : destroy_name_(destroy_name) {}
  ~Handle() {
    if (value) {
      check_cleanup(Destroy(value), destroy_name_);
    }
  }
  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;
  T value = nullptr;

private:
  const char *destroy_name_;
};

using Context = Handle<ze_context_handle_t, zeContextDestroy>;
using Module = Handle<ze_module_handle_t, zeModuleDestroy>;
using BuildLog = Handle<ze_module_build_log_handle_t, zeModuleBuildLogDestroy>;
using Kernel = Handle<ze_kernel_handle_t, zeKernelDestroy>;
using Queue = Handle<ze_command_queue_handle_t, zeCommandQueueDestroy>;
using CommandList = Handle<ze_command_list_handle_t, zeCommandListDestroy>;

class Allocation {
public:
  explicit Allocation(ze_context_handle_t context) : context_(context) {}
  ~Allocation() {
    if (value) {
      check_cleanup(zeMemFree(context_, value), "zeMemFree");
    }
  }
  void allocate_host(size_t bytes) {
    ze_host_mem_alloc_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    ZE_CHECK(zeMemAllocHost(context_, &desc, bytes, alignof(float), &value));
  }
  void allocate_device(ze_device_handle_t device, uint32_t ordinal,
                       size_t bytes) {
    ze_device_mem_alloc_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    desc.ordinal = ordinal;
    ZE_CHECK(zeMemAllocDevice(context_, &desc, bytes, alignof(float), device,
                              &value));
  }
  Allocation(const Allocation &) = delete;
  Allocation &operator=(const Allocation &) = delete;
  void *value = nullptr;

private:
  ze_context_handle_t context_;
};

// Declared after all resources referenced by submitted commands. On an ordinary
// exception it completes the queue before those resources unwind. Core Level
// Zero has no cancellation operation: unknown completion cannot safely proceed
// to zeMemFree/list/module/context destruction, nor to an infinite wait.
class Submission {
public:
  explicit Submission(ze_command_queue_handle_t queue) : queue_(queue) {}
  ~Submission() {
    if (pending_) {
      synchronize();
    }
  }
  void execute(ze_command_list_handle_t list) {
    pending_ = true;
    ZE_CHECK(zeCommandQueueExecuteCommandLists(queue_, 1, &list, nullptr));
  }
  void synchronize() noexcept {
    if (!pending_) {
      return;
    }
    const ze_result_t result =
        zeCommandQueueSynchronize(queue_, kWaitTimeoutNs);
    if (result != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr,
                   "FAIL backend=intel operation=zeCommandQueueSynchronize "
                   "result=0x%08x "
                   "timeout_ns=%llu completion=unknown; terminating without "
                   "freeing in-flight resources\n",
                   static_cast<unsigned>(result),
                   static_cast<unsigned long long>(kWaitTimeoutNs));
      std::fflush(nullptr);
      std::_Exit(EXIT_FAILURE);
    }
    pending_ = false;
  }
  Submission(const Submission &) = delete;
  Submission &operator=(const Submission &) = delete;

private:
  ze_command_queue_handle_t queue_;
  bool pending_ = false;
};

std::string lower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
}

struct Options {
  std::string payload;
  std::string symbol = "therock_module_saxpy";
  std::string expected_name;
  uint32_t expected_device_id = 0;
  bool check_device_id = false;
  uint32_t device = 0;
};

uint32_t parse_uint(const std::string &text, int base,
                    const std::string &option) {
  if (text.empty() || text[0] == '-' || text[0] == '+' ||
      std::isspace(static_cast<unsigned char>(text[0]))) {
    throw std::runtime_error(option + " requires an unsigned integer");
  }
  size_t consumed = 0;
  const unsigned long long value = std::stoull(text, &consumed, base);
  if (consumed != text.size() || value > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(option + " requires a 32-bit unsigned integer");
  }
  return static_cast<uint32_t>(value);
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      std::puts(
          "--payload PATH --symbol therock_module_saxpy|therock_module_relu "
          "--device INDEX --expect-arch spirv --expect-device-id 0xe223 "
          "--expect-device-name SUBSTRING");
      std::exit(0);
    }
    if (++i == argc) {
      throw std::runtime_error("Missing value for " + arg);
    }
    const std::string value = argv[i];
    if (arg == "--payload") {
      options.payload = value;
    } else if (arg == "--symbol") {
      options.symbol = value;
    } else if (arg == "--device") {
      options.device = parse_uint(value, 10, arg);
    } else if (arg == "--expect-device-id") {
      const int base =
          value.rfind("0x", 0) == 0 || value.rfind("0X", 0) == 0 ? 16 : 10;
      options.expected_device_id = parse_uint(value, base, arg);
      if (options.expected_device_id == 0 ||
          options.expected_device_id > 0xffff) {
        throw std::runtime_error(
            "--expect-device-id requires a nonzero 16-bit PCI ID");
      }
      options.check_device_id = true;
    } else if (arg == "--expect-device-name") {
      if (value.empty()) {
        throw std::runtime_error("--expect-device-name cannot be empty");
      }
      options.expected_name = lower(value);
    } else if (arg == "--expect-arch") {
      if (value != "spirv") {
        throw std::runtime_error(
            "--expect-arch accepts spirv as a payload format only; "
            "use --expect-device-id or --expect-device-name for Intel hardware "
            "identity");
      }
    } else {
      throw std::runtime_error("Unknown argument " + arg);
    }
  }
  if (options.payload.empty()) {
    throw std::runtime_error("--payload is required");
  }
  if (options.symbol != "therock_module_saxpy" &&
      options.symbol != "therock_module_relu") {
    throw std::runtime_error("No CPU reference is defined for symbol " +
                             options.symbol);
  }
  return options;
}

std::vector<uint32_t> read_spirv(const std::string &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("Cannot read payload " + path);
  }
  const std::streamoff bytes = input.tellg();
  if (bytes < 24 || bytes % 4 != 0 ||
      static_cast<uint64_t>(bytes) > kMaxPayloadBytes) {
    throw std::runtime_error("Malformed SPIR-V size (requires whole words, "
                             "header and instructions, <=64 MiB)");
  }
  input.seekg(0);
  std::vector<uint32_t> words(static_cast<size_t>(bytes) / sizeof(uint32_t));
  if (!input.read(reinterpret_cast<char *>(words.data()), bytes)) {
    throw std::runtime_error("Could not read complete SPIR-V payload");
  }
  // Reject other payload types and swapped-endian files before initializing a
  // GPU driver. spirv-val at build time and zeModuleCreate validate semantics.
  if (words[0] != 0x07230203 || words[3] == 0 || words[4] != 0) {
    throw std::runtime_error("Malformed SPIR-V header or wrong payload format");
  }
  const uint32_t version = words[1];
  if ((version & 0xff0000ff) != 0 || ((version >> 16) & 0xff) != 1 ||
      ((version >> 8) & 0xff) > 6) {
    throw std::runtime_error("Unsupported or malformed SPIR-V version");
  }
  for (size_t i = 5; i < words.size();) {
    const uint32_t word_count = words[i] >> 16;
    if (word_count == 0 || word_count > words.size() - i) {
      throw std::runtime_error("Malformed SPIR-V instruction length at word " +
                               std::to_string(i));
    }
    i += word_count;
  }
  return words;
}

struct Candidate {
  ze_driver_handle_t driver = nullptr;
  ze_device_handle_t device = nullptr;
  ze_driver_properties_t driver_properties{};
  ze_device_properties_t properties{};
};

Candidate select_device(const Options &options) {
  ZE_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  uint32_t driver_count = 0;
  ZE_CHECK(zeDriverGet(&driver_count, nullptr));
  if (driver_count == 0) {
    throw std::runtime_error("Level Zero found no GPU drivers");
  }
  std::vector<ze_driver_handle_t> drivers(driver_count);
  ZE_CHECK(zeDriverGet(&driver_count, drivers.data()));
  if (driver_count > drivers.size()) {
    throw std::runtime_error("GPU driver enumeration changed during discovery");
  }
  drivers.resize(driver_count);
  std::vector<Candidate> candidates;
  for (ze_driver_handle_t driver : drivers) {
    ze_driver_properties_t driver_properties{};
    driver_properties.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
    ZE_CHECK(zeDriverGetProperties(driver, &driver_properties));
    uint32_t device_count = 0;
    ZE_CHECK(zeDeviceGet(driver, &device_count, nullptr));
    if (device_count == 0) {
      continue;
    }
    std::vector<ze_device_handle_t> devices(device_count);
    ZE_CHECK(zeDeviceGet(driver, &device_count, devices.data()));
    if (device_count > devices.size()) {
      throw std::runtime_error(
          "Root device enumeration changed during discovery");
    }
    devices.resize(device_count);
    for (ze_device_handle_t device : devices) {
      ze_device_properties_t properties{};
      properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      ZE_CHECK(zeDeviceGetProperties(device, &properties));
      if (properties.type == ZE_DEVICE_TYPE_GPU &&
          properties.vendorId == kIntelVendor &&
          !(properties.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE)) {
        candidates.push_back({driver, device, driver_properties, properties});
      }
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &a, const Candidate &b) {
              const int drivers_order = std::memcmp(a.driver_properties.uuid.id,
                                                    b.driver_properties.uuid.id,
                                                    ZE_MAX_DRIVER_UUID_SIZE);
              if (drivers_order != 0) {
                return drivers_order < 0;
              }
              return std::memcmp(a.properties.uuid.id, b.properties.uuid.id,
                                 ZE_MAX_DEVICE_UUID_SIZE) < 0;
            });
  // Duplicate identities cannot provide a deterministic root-device index.
  for (size_t i = 1; i < candidates.size(); ++i) {
    if (std::memcmp(candidates[i - 1].driver_properties.uuid.id,
                    candidates[i].driver_properties.uuid.id,
                    ZE_MAX_DRIVER_UUID_SIZE) == 0 &&
        std::memcmp(candidates[i - 1].properties.uuid.id,
                    candidates[i].properties.uuid.id,
                    ZE_MAX_DEVICE_UUID_SIZE) == 0) {
      throw std::runtime_error("Ambiguous duplicate Intel root device UUIDs");
    }
  }
  if (options.device >= candidates.size()) {
    throw std::runtime_error(
        "Requested device " + std::to_string(options.device) +
        " but Level Zero found " + std::to_string(candidates.size()) +
        " Intel GPU root devices");
  }
  const Candidate selected = candidates[options.device];
  ze_api_version_t api_version{};
  ZE_CHECK(zeDriverGetApiVersion(selected.driver, &api_version));
  if (api_version < ZE_API_VERSION_1_0) {
    throw std::runtime_error("Level Zero driver does not support core API 1.0");
  }
  const std::string name(selected.properties.name,
                         strnlen(selected.properties.name, ZE_MAX_DEVICE_NAME));
  std::printf(
      "DEVICE backend=intel runtime=level-zero index=%u count=%zu name=\"%s\" "
      "vendor_id=0x%04x device_id=0x%04x root=true hardware_arch=unverified "
      "driver_version=%u api_version=%u.%u\n",
      options.device, candidates.size(), name.c_str(),
      selected.properties.vendorId, selected.properties.deviceId,
      selected.driver_properties.driverVersion, ZE_MAJOR_VERSION(api_version),
      ZE_MINOR_VERSION(api_version));
  if (options.check_device_id &&
      selected.properties.deviceId != options.expected_device_id) {
    char detail[128]{};
    std::snprintf(detail, sizeof(detail),
                  "Expected device ID 0x%04x but got 0x%04x",
                  options.expected_device_id, selected.properties.deviceId);
    throw std::runtime_error(detail);
  }
  if (!options.expected_name.empty() &&
      lower(name).find(options.expected_name) == std::string::npos) {
    throw std::runtime_error("Device name does not contain " +
                             options.expected_name);
  }
  return selected;
}

uint32_t select_queue_group(ze_device_handle_t device) {
  uint32_t count = 0;
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(device, &count, nullptr));
  if (count == 0) {
    throw std::runtime_error("Device reports no command queue groups");
  }
  std::vector<ze_command_queue_group_properties_t> groups(count);
  for (auto &group : groups) {
    group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  }
  ZE_CHECK(
      zeDeviceGetCommandQueueGroupProperties(device, &count, groups.data()));
  if (count > groups.size()) {
    throw std::runtime_error(
        "Command queue group enumeration changed during discovery");
  }
  for (uint32_t i = 0; i < count; ++i) {
    constexpr auto required = ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE |
                              ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY;
    if ((groups[i].flags & required) == required && groups[i].numQueues > 0) {
      return i;
    }
  }
  throw std::runtime_error(
      "Device lacks a command queue group supporting compute and copies");
}

uint32_t select_memory(ze_device_handle_t device, size_t required_bytes) {
  uint32_t count = 0;
  ZE_CHECK(zeDeviceGetMemoryProperties(device, &count, nullptr));
  if (count == 0) {
    throw std::runtime_error("Device reports no local memory heaps");
  }
  std::vector<ze_device_memory_properties_t> heaps(count);
  for (auto &heap : heaps) {
    heap.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
  }
  ZE_CHECK(zeDeviceGetMemoryProperties(device, &count, heaps.data()));
  if (count > heaps.size()) {
    throw std::runtime_error(
        "Device memory enumeration changed during discovery");
  }
  for (uint32_t i = 0; i < count; ++i) {
    if (heaps[i].totalSize >= required_bytes) {
      return i;
    }
  }
  throw std::runtime_error(
      "Device lacks enough local memory for the validation fixture");
}

void print_build_log(ze_module_build_log_handle_t log) {
  if (!log) {
    return;
  }
  size_t bytes = 0;
  ZE_CHECK(zeModuleBuildLogGetString(log, &bytes, nullptr));
  if (bytes == 0) {
    return;
  }
  std::vector<char> message(bytes, '\0');
  ZE_CHECK(zeModuleBuildLogGetString(log, &bytes, message.data()));
  if (bytes > message.size()) {
    throw std::runtime_error("Level Zero build log changed during retrieval");
  }
  const auto end = std::find(message.begin(), message.end(), '\0');
  const std::string text(message.begin(), end);
  if (!text.empty()) {
    std::fprintf(stderr, "MODULE_BUILD_LOG %s\n", text.c_str());
  }
}

void run(const Options &options) {
  const std::vector<uint32_t> words = read_spirv(options.payload);
  const Candidate selected = select_device(options);
  ze_device_compute_properties_t compute{};
  compute.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
  ZE_CHECK(zeDeviceGetComputeProperties(selected.device, &compute));
  if (compute.maxTotalGroupSize < kGroupSize ||
      compute.maxGroupSizeX < kGroupSize || compute.maxGroupSizeY < 1 ||
      compute.maxGroupSizeZ < 1 ||
      compute.maxGroupCountX < (65539 + kGroupSize - 1) / kGroupSize ||
      compute.maxGroupCountY < 1 || compute.maxGroupCountZ < 1) {
    throw std::runtime_error(
        "Device does not support the validation launch geometry");
  }
  const size_t max_bytes = (65539 + kGuardCount) * sizeof(float);
  if (selected.properties.maxMemAllocSize < max_bytes) {
    throw std::runtime_error("Device maximum allocation size is too small");
  }
  ze_device_module_properties_t module_properties{};
  module_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
  ZE_CHECK(zeDeviceGetModuleProperties(selected.device, &module_properties));
  const uint32_t requested_version =
      ZE_MAKE_VERSION((words[1] >> 16) & 0xff, (words[1] >> 8) & 0xff);
  if (module_properties.spirvVersionSupported < requested_version) {
    throw std::runtime_error(
        "Device does not support the payload SPIR-V version");
  }
  const uint32_t queue_ordinal = select_queue_group(selected.device);
  const uint32_t memory_ordinal = select_memory(selected.device, 3 * max_bytes);
  Context context("zeContextDestroy");
  ze_context_desc_t context_desc{};
  context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ZE_CHECK(zeContextCreate(selected.driver, &context_desc, &context.value));
  Module module("zeModuleDestroy");
  {
    BuildLog log("zeModuleBuildLogDestroy");
    ze_module_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    desc.inputSize = words.size() * sizeof(uint32_t);
    desc.pInputModule = reinterpret_cast<const uint8_t *>(words.data());
    const ze_result_t result = zeModuleCreate(context.value, selected.device,
                                              &desc, &module.value, &log.value);
    if (result != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "MODULE_BUILD_FAIL result=0x%08x\n",
                   static_cast<unsigned>(result));
    }
    print_build_log(log.value);
    ZE_CHECK(result);
  }
  Kernel kernel("zeKernelDestroy");
  ze_kernel_desc_t kernel_desc{};
  kernel_desc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  kernel_desc.pKernelName = options.symbol.c_str();
  ZE_CHECK(zeKernelCreate(module.value, &kernel_desc, &kernel.value));
  ze_kernel_properties_t properties{};
  properties.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  ZE_CHECK(zeKernelGetProperties(kernel.value, &properties));
  if (properties.numKernelArgs != 5 ||
      (properties.requiredGroupSizeX &&
       properties.requiredGroupSizeX != kGroupSize) ||
      (properties.requiredGroupSizeY && properties.requiredGroupSizeY != 1) ||
      (properties.requiredGroupSizeZ && properties.requiredGroupSizeZ != 1)) {
    throw std::runtime_error("Kernel does not match the five-argument "
                             "validation ABI or group geometry");
  }
  ZE_CHECK(zeKernelSetGroupSize(kernel.value, kGroupSize, 1, 1));
  Queue queue("zeCommandQueueDestroy");
  ze_command_queue_desc_t queue_desc{};
  queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  queue_desc.ordinal = queue_ordinal;
  queue_desc.index = 0;
  queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  ZE_CHECK(zeCommandQueueCreate(context.value, selected.device, &queue_desc,
                                &queue.value));
  std::printf("MODULE backend=intel runtime=level-zero format=spirv symbol=%s "
              "payload=\"%s\" "
              "queue_ordinal=%u memory_ordinal=%u\n",
              options.symbol.c_str(), options.payload.c_str(), queue_ordinal,
              memory_ordinal);
  for (uint32_t count : kSizes) {
    const size_t bytes = count * sizeof(float);
    const size_t output_bytes = (count + kGuardCount) * sizeof(float);
    Allocation host_x(context.value), host_y(context.value),
        host_output(context.value);
    host_x.allocate_host(bytes);
    host_y.allocate_host(bytes);
    host_output.allocate_host(output_bytes);
    Allocation device_x(context.value), device_y(context.value),
        device_output(context.value);
    device_x.allocate_device(selected.device, memory_ordinal, bytes);
    device_y.allocate_device(selected.device, memory_ordinal, bytes);
    device_output.allocate_device(selected.device, memory_ordinal,
                                  output_bytes);
    auto *x = static_cast<float *>(host_x.value);
    auto *y = static_cast<float *>(host_y.value);
    auto *actual = static_cast<float *>(host_output.value);
    std::vector<float> expected(count);
    CommandList list("zeCommandListDestroy");
    ze_command_list_desc_t list_desc{};
    list_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    list_desc.commandQueueGroupOrdinal = queue_ordinal;
    ZE_CHECK(zeCommandListCreate(context.value, selected.device, &list_desc,
                                 &list.value));
    Submission submission(queue.value);
    for (uint32_t round = 0; round < 3; ++round) {
      const float alpha = 1.75f;
      for (uint32_t i = 0; i < count; ++i) {
        x[i] = static_cast<float>(static_cast<int>((i * 17 + round * 13) % 97) -
                                  48) /
               16.0f;
        y[i] = static_cast<float>(static_cast<int>((i * 11 + round * 7) % 89) -
                                  44) /
               32.0f;
        expected[i] = alpha * x[i] + y[i];
        if (options.symbol == "therock_module_relu") {
          expected[i] = std::max(0.0f, expected[i] - 0.125f);
        }
      }
      std::fill_n(actual, count + kGuardCount, kGuardValue);
      ZE_CHECK(zeKernelSetArgumentValue(kernel.value, 0, sizeof(void *),
                                        &device_x.value));
      ZE_CHECK(zeKernelSetArgumentValue(kernel.value, 1, sizeof(void *),
                                        &device_y.value));
      ZE_CHECK(zeKernelSetArgumentValue(kernel.value, 2, sizeof(void *),
                                        &device_output.value));
      ZE_CHECK(
          zeKernelSetArgumentValue(kernel.value, 3, sizeof(alpha), &alpha));
      ZE_CHECK(
          zeKernelSetArgumentValue(kernel.value, 4, sizeof(count), &count));
      ZE_CHECK(zeCommandListAppendMemoryCopy(list.value, device_x.value, x,
                                             bytes, nullptr, 0, nullptr));
      ZE_CHECK(zeCommandListAppendMemoryCopy(list.value, device_y.value, y,
                                             bytes, nullptr, 0, nullptr));
      ZE_CHECK(zeCommandListAppendMemoryCopy(list.value, device_output.value,
                                             actual, output_bytes, nullptr, 0,
                                             nullptr));
      ZE_CHECK(zeCommandListAppendBarrier(list.value, nullptr, 0, nullptr));
      const ze_group_count_t groups{(count + kGroupSize - 1) / kGroupSize, 1,
                                    1};
      ZE_CHECK(zeCommandListAppendLaunchKernel(list.value, kernel.value,
                                               &groups, nullptr, 0, nullptr));
      ZE_CHECK(zeCommandListAppendBarrier(list.value, nullptr, 0, nullptr));
      ZE_CHECK(zeCommandListAppendMemoryCopy(list.value, actual,
                                             device_output.value, output_bytes,
                                             nullptr, 0, nullptr));
      ZE_CHECK(zeCommandListClose(list.value));
      submission.execute(list.value);
      submission.synchronize();
      double max_error = 0.0;
      for (uint32_t i = 0; i < count; ++i) {
        const double error =
            std::abs(static_cast<double>(actual[i]) - expected[i]);
        // Bounded dyadic inputs and a few FP32 rounding steps permit FMA
        // contraction without hiding omitted kernels, bad tails or stale data.
        const double tolerance = 1.0e-6 + 2.0e-6 * std::abs(expected[i]);
        if (!std::isfinite(actual[i]) || error > tolerance) {
          throw std::runtime_error(
              "Arithmetic mismatch index=" + std::to_string(i) +
              " expected=" + std::to_string(expected[i]) +
              " actual=" + std::to_string(actual[i]));
        }
        max_error = std::max(max_error, error);
      }
      for (uint32_t i = count; i < count + kGuardCount; ++i) {
        if (actual[i] != kGuardValue) {
          throw std::runtime_error("Output guard overwritten index=" +
                                   std::to_string(i));
        }
      }
      std::printf("CHECK n=%u round=%u max_abs_error=%.9g guard=pass\n", count,
                  round, max_error);
      ZE_CHECK(zeCommandListReset(list.value));
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  try {
    const Options options = parse_options(argc, argv);
    run(options);
    if (cleanup_failed) {
      throw std::runtime_error("Level Zero resource cleanup failed");
    }
    std::printf("PASS backend=intel runtime=level-zero symbol=%s\n",
                options.symbol.c_str());
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL backend=intel error=%s\n", error.what());
    return 1;
  }
}
