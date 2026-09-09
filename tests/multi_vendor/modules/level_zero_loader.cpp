// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <level_zero/ze_api.h>

#include "device_inventory.h"
#include "module_service_protocol.h"
#include "module_session_options.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr uint32_t kIntelVendor = 0x8086;
constexpr uint32_t kGroupSize = therock::module_contract::kGroupSize;
constexpr uint32_t kGuardCount = 17;
constexpr float kGuardValue = -12345.5f;
constexpr uint32_t kSizes[] = {1, 127, 128, 129, 4099, 65539};
constexpr uint64_t kWaitTimeoutNs = 30ULL * 1000 * 1000 * 1000;
constexpr size_t kMaxPayloadBytes = 64ULL * 1024 * 1024;
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
using EventPool = Handle<ze_event_pool_handle_t, zeEventPoolDestroy>;
using Event = Handle<ze_event_handle_t, zeEventDestroy>;

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
  std::string expected_device_uuid;
  uint32_t expected_device_id = 0;
  bool check_device_id = false;
  uint32_t device = 0;
  therock::module_validation::ContractOptions contract;
  std::vector<therock::module_validation::ModuleRequest> modules;
  bool batch = false;
  bool pipeline = false;
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
  std::unordered_set<std::string> seen;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      std::puts(
          "--payload PATH --symbol therock_module_saxpy|therock_module_relu "
          "--device INDEX --expect-arch spirv --expect-device-id 0xe223 "
          "--expect-device-name SUBSTRING [--expect-device-uuid UUID] "
          "--launch-abi ABI --launch-abi-version VERSION "
          "--launch-contract-sha256 SHA256 --payload-format spirv\n"
          "--module FORMAT SYMBOL PATH (repeatable; replaces "
          "payload/symbol/format)\n"
          "--pipeline (compose --module stages through device scratch "
          "buffers)\n"
          "--describe-contract (exclusive, no payload or GPU access)\n"
          "--list-devices (exclusive, runtime properties only)");
      std::exit(0);
    }
    if (arg == "--module") {
      options.modules.push_back(
          therock::module_validation::parse_module_request(i, argc, argv));
      continue;
    }
    if (!seen.insert(arg).second) {
      throw std::runtime_error("Duplicate argument " + arg);
    }
    if (arg == "--pipeline") {
      options.pipeline = true;
      continue;
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
    } else if (arg == "--expect-device-uuid") {
      therock::module_validation::validate_device_uuid(value);
      options.expected_device_uuid = value;
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
    } else if (!options.contract.consume(arg, value)) {
      throw std::runtime_error("Unknown argument " + arg);
    }
  }
  options.batch = !options.modules.empty();
  options.modules = therock::module_validation::finalize_module_requests(
      options.contract, options.payload, options.symbol, options.modules, seen,
      {"spirv"}, options.pipeline);
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

std::vector<Candidate> enumerate_devices() {
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
  // UUID binding uses the device UUID alone, so duplicates are ambiguous even
  // when separate driver handles report them. Also reject unavailable UUIDs.
  std::unordered_set<std::string> device_uuids;
  for (const auto &candidate : candidates) {
    const std::string uuid = therock::module_validation::device_uuid_hex(
        candidate.properties.uuid.id);
    if (!device_uuids.insert(uuid).second) {
      throw std::runtime_error("Duplicate Intel root device UUID: " + uuid);
    }
  }
  return candidates;
}

Candidate select_device(const Options &options) {
  const std::vector<Candidate> candidates = enumerate_devices();
  if (options.device >= candidates.size()) {
    throw std::runtime_error(
        "Requested device " + std::to_string(options.device) +
        " but Level Zero found " + std::to_string(candidates.size()) +
        " Intel GPU root devices");
  }
  const Candidate selected = candidates[options.device];
  const std::string device_uuid =
      therock::module_validation::device_uuid_hex(selected.properties.uuid.id);
  therock::module_validation::require_device_uuid(options.expected_device_uuid,
                                                  device_uuid);
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
      "driver_version=%u api_version=%u.%u device_uuid=%s\n",
      options.device, candidates.size(), name.c_str(),
      selected.properties.vendorId, selected.properties.deviceId,
      selected.driver_properties.driverVersion, ZE_MAJOR_VERSION(api_version),
      ZE_MINOR_VERSION(api_version), device_uuid.c_str());
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

std::vector<ze_device_memory_properties_t>
enumerate_memory(ze_device_handle_t device) {
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
  heaps.resize(count);
  return heaps;
}

uint32_t select_memory(ze_device_handle_t device, size_t required_bytes) {
  const auto heaps = enumerate_memory(device);
  for (uint32_t i = 0; i < heaps.size(); ++i) {
    if (heaps[i].totalSize >= required_bytes) {
      return i;
    }
  }
  throw std::runtime_error(
      "Device lacks enough local memory for the validation fixture");
}

void list_devices() {
  using namespace therock::module_validation;
  // The same filtered and UUID-sorted root list defines launch --device.
  const auto candidates = enumerate_devices();
  std::vector<DeviceInventoryEntry> devices;
  for (size_t index = 0; index < candidates.size(); ++index) {
    const auto &candidate = candidates[index];
    DeviceInventoryEntry entry;
    entry.index = index;
    entry.name = device_name(candidate.properties.name);
    entry.device_uuid = device_uuid_hex(candidate.properties.uuid.id);
    // Core Level Zero does not expose the catalog architecture identity.
    // Leave architecture unknown and select Intel hardware by PCI identity.
    entry.vendor_id = candidate.properties.vendorId;
    entry.device_id = candidate.properties.deviceId;
    ze_device_compute_properties_t compute{};
    compute.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
    ZE_CHECK(zeDeviceGetComputeProperties(candidate.device, &compute));
    entry.max_threads_per_block =
        positive_device_limit(compute.maxTotalGroupSize);
    entry.max_block_dimensions = {positive_device_limit(compute.maxGroupSizeX),
                                  positive_device_limit(compute.maxGroupSizeY),
                                  positive_device_limit(compute.maxGroupSizeZ)};
    entry.max_grid_dimensions = {positive_device_limit(compute.maxGroupCountX),
                                 positive_device_limit(compute.maxGroupCountY),
                                 positive_device_limit(compute.maxGroupCountZ)};
    // Sum the runtime's device-visible heap capacities. This is neither free
    // memory nor maxMemAllocSize, and does not promise one contiguous heap.
    for (const auto &heap : enumerate_memory(candidate.device)) {
      if (heap.totalSize >
          std::numeric_limits<uint64_t>::max() - entry.total_memory_bytes) {
        throw std::runtime_error("Device memory capacity overflows uint64");
      }
      entry.total_memory_bytes += heap.totalSize;
    }
    devices.push_back(entry);
  }
  const std::string result =
      device_inventory_json("intel", "level-zero", devices);
  std::puts(result.c_str());
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

// A kernel is destroyed before the module that owns its code. Stable heap
// owners avoid transferring native handles when the session vector grows.
struct LoadedModule {
  Module module{"zeModuleDestroy"};
  Kernel kernel{"zeKernelDestroy"};
};

void run(const Options &options) {
  // Read and inspect every request before any device discovery or
  // initialization. Keep these exact bytes until all module creation and
  // execution is finished.
  std::vector<std::vector<uint32_t>> payloads;
  payloads.reserve(options.modules.size());
  for (const auto &request : options.modules) {
    payloads.push_back(read_spirv(request.payload));
    if (request.payload_format != "spirv") {
      throw std::runtime_error("Declared module format does not match SPIR-V");
    }
  }
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
  for (const auto &words : payloads) {
    const uint32_t requested_version =
        ZE_MAKE_VERSION((words[1] >> 16) & 0xff, (words[1] >> 8) & 0xff);
    if (module_properties.spirvVersionSupported < requested_version) {
      throw std::runtime_error(
          "Device does not support a payload SPIR-V version");
    }
  }
  const uint32_t queue_ordinal = select_queue_group(selected.device);
  const uint32_t memory_ordinal =
      select_memory(selected.device, (options.pipeline ? 4 : 3) * max_bytes);
  Context context("zeContextDestroy");
  ze_context_desc_t context_desc{};
  context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ZE_CHECK(zeContextCreate(selected.driver, &context_desc, &context.value));
  // Resolve every module and kernel, including argument count and geometry,
  // before allocating or submitting any validation work.
  std::vector<std::unique_ptr<LoadedModule>> modules;
  modules.reserve(options.modules.size());
  for (size_t module_index = 0; module_index < options.modules.size();
       ++module_index) {
    const auto &request = options.modules[module_index];
    const auto &words = payloads[module_index];
    modules.push_back(std::make_unique<LoadedModule>());
    auto &loaded = *modules.back();
    {
      BuildLog log("zeModuleBuildLogDestroy");
      ze_module_desc_t desc{};
      desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
      desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
      desc.inputSize = words.size() * sizeof(uint32_t);
      desc.pInputModule = reinterpret_cast<const uint8_t *>(words.data());
      const ze_result_t result =
          zeModuleCreate(context.value, selected.device, &desc,
                         &loaded.module.value, &log.value);
      if (result != ZE_RESULT_SUCCESS) {
        std::fprintf(stderr,
                     "MODULE_BUILD_FAIL module_index=%zu result=0x%08x\n",
                     module_index, static_cast<unsigned>(result));
      }
      print_build_log(log.value);
      ZE_CHECK(result);
    }
    ze_kernel_desc_t kernel_desc{};
    kernel_desc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kernel_desc.pKernelName = request.symbol.c_str();
    ZE_CHECK(zeKernelCreate(loaded.module.value, &kernel_desc,
                            &loaded.kernel.value));
    ze_kernel_properties_t properties{};
    properties.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
    ZE_CHECK(zeKernelGetProperties(loaded.kernel.value, &properties));
    if (properties.numKernelArgs != 5 ||
        (properties.requiredGroupSizeX &&
         properties.requiredGroupSizeX != kGroupSize) ||
        (properties.requiredGroupSizeY && properties.requiredGroupSizeY != 1) ||
        (properties.requiredGroupSizeZ && properties.requiredGroupSizeZ != 1)) {
      throw std::runtime_error("Kernel does not match the five-argument "
                               "validation ABI or group geometry");
    }
    ZE_CHECK(zeKernelSetGroupSize(loaded.kernel.value, kGroupSize, 1, 1));
    std::printf(
        "MODULE backend=intel runtime=level-zero format=spirv symbol=%s "
        "payload=\"%s\" module_index=%zu "
        "queue_ordinal=%u memory_ordinal=%u\n",
        request.symbol.c_str(), request.payload.c_str(), module_index,
        queue_ordinal, memory_ordinal);
  }
  Queue upload_queue("zeCommandQueueDestroy"),
      compute_queue("zeCommandQueueDestroy"),
      readback_queue("zeCommandQueueDestroy");
  ze_command_queue_desc_t queue_desc{};
  queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  queue_desc.ordinal = queue_ordinal;
  queue_desc.index = 0;
  queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  // Independent logical queues may use the same compute/copy engine. Event
  // dependencies, not host synchronization or engine concurrency, order them.
  ZE_CHECK(zeCommandQueueCreate(context.value, selected.device, &queue_desc,
                                &upload_queue.value));
  ZE_CHECK(zeCommandQueueCreate(context.value, selected.device, &queue_desc,
                                &compute_queue.value));
  ZE_CHECK(zeCommandQueueCreate(context.value, selected.device, &queue_desc,
                                &readback_queue.value));
  EventPool event_pool("zeEventPoolDestroy");
  ze_event_pool_desc_t pool_desc{};
  pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
  pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  pool_desc.count = 3;
  ze_device_handle_t event_device = selected.device;
  ZE_CHECK(zeEventPoolCreate(context.value, &pool_desc, 1, &event_device,
                             &event_pool.value));
  Event uploaded("zeEventDestroy"), computed("zeEventDestroy"),
      readback_done("zeEventDestroy");
  const auto create_event = [&](Event &event, uint32_t index,
                                ze_event_scope_flags_t scope) {
    ze_event_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    desc.index = index;
    desc.signal = scope;
    desc.wait = scope;
    ZE_CHECK(zeEventCreate(event_pool.value, &desc, &event.value));
  };
  // ze_event_desc_t signal scopes flush before signaling; wait scopes
  // invalidate after observing the signal. DEVICE covers the next device queue,
  // while HOST also makes readback bytes visible to the CPU waiting on the
  // final event.
  create_event(uploaded, 0, ZE_EVENT_SCOPE_FLAG_DEVICE);
  create_event(computed, 1, ZE_EVENT_SCOPE_FLAG_DEVICE);
  create_event(readback_done, 2, ZE_EVENT_SCOPE_FLAG_HOST);
  // One maximum-sized set is reused by every size, round, and module.
  Allocation host_x(context.value), host_y(context.value),
      host_output(context.value), host_scratch_b(context.value);
  host_x.allocate_host(max_bytes);
  host_y.allocate_host(max_bytes);
  host_output.allocate_host(max_bytes);
  Allocation device_x(context.value), device_y(context.value),
      device_output(context.value), device_scratch_b(context.value);
  device_x.allocate_device(selected.device, memory_ordinal, max_bytes);
  device_y.allocate_device(selected.device, memory_ordinal, max_bytes);
  device_output.allocate_device(selected.device, memory_ordinal, max_bytes);
  if (options.pipeline) {
    host_scratch_b.allocate_host(max_bytes);
    device_scratch_b.allocate_device(selected.device, memory_ordinal,
                                     max_bytes);
  }
  auto *x = static_cast<float *>(host_x.value);
  auto *y = static_cast<float *>(host_y.value);
  auto *actual = static_cast<float *>(host_output.value);
  std::vector<float> expected(65539);
  CommandList upload_list("zeCommandListDestroy"),
      compute_list("zeCommandListDestroy"),
      readback_list("zeCommandListDestroy");
  ze_command_list_desc_t list_desc{};
  list_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
  list_desc.commandQueueGroupOrdinal = queue_ordinal;
  ZE_CHECK(zeCommandListCreate(context.value, selected.device, &list_desc,
                               &upload_list.value));
  ZE_CHECK(zeCommandListCreate(context.value, selected.device, &list_desc,
                               &compute_list.value));
  ZE_CHECK(zeCommandListCreate(context.value, selected.device, &list_desc,
                               &readback_list.value));
  // These owners unwind before lists, allocations, events, queues, and the
  // modules. A failed submission may still be pending, so each queue is
  // tracked.
  Submission upload_submission(upload_queue.value),
      compute_submission(compute_queue.value),
      readback_submission(readback_queue.value);
  if (options.pipeline) {
    std::printf("SESSION backend=intel mode=device-module-pipeline modules=%zu "
                "contexts=1 buffers=4 queues=3 events=3\n",
                options.modules.size());
    const size_t capacity = max_bytes / sizeof(float);
    const std::array<void *, 2> scratch{device_output.value,
                                        device_scratch_b.value};
    const std::array<float *, 2> host_scratch{
        actual, static_cast<float *>(host_scratch_b.value)};
    std::array<std::vector<float>, 2> references{std::vector<float>(capacity),
                                                 std::vector<float>(capacity)};
    std::array<std::vector<double>, 2> error_bounds{
        std::vector<double>(capacity), std::vector<double>(capacity)};
    for (uint32_t count : kSizes) {
      const size_t bytes = count * sizeof(float);
      for (uint32_t round = 0; round < 3; ++round) {
        const float alpha = 1.75f;
        for (uint32_t i = 0; i < count; ++i) {
          x[i] = static_cast<float>(
                     static_cast<int>((i * 17 + round * 13) % 97) - 48) /
                 16.0f;
          y[i] = static_cast<float>(
                     static_cast<int>((i * 11 + round * 7) % 89) - 44) /
                 32.0f;
        }
        for (size_t buffer = 0; buffer < scratch.size(); ++buffer) {
          std::fill_n(host_scratch[buffer], capacity, kGuardValue);
          std::fill(references[buffer].begin(), references[buffer].end(),
                    kGuardValue);
          std::fill(error_bounds[buffer].begin(), error_bounds[buffer].end(),
                    0.0);
        }
        ZE_CHECK(zeCommandListAppendMemoryCopy(
            upload_list.value, device_x.value, x, bytes, nullptr, 0, nullptr));
        ZE_CHECK(zeCommandListAppendMemoryCopy(
            upload_list.value, device_y.value, y, bytes, nullptr, 0, nullptr));
        for (size_t buffer = 0; buffer < scratch.size(); ++buffer) {
          ZE_CHECK(zeCommandListAppendMemoryCopy(
              upload_list.value, scratch[buffer], host_scratch[buffer],
              max_bytes, nullptr, 0, nullptr));
        }
        ZE_CHECK(zeCommandListAppendBarrier(upload_list.value, uploaded.value,
                                            0, nullptr));
        const ze_group_count_t groups{(count + kGroupSize - 1) / kGroupSize, 1,
                                      1};
        for (size_t stage = 0; stage < options.modules.size(); ++stage) {
          const size_t output_index = stage % 2;
          const size_t input_index = 1 - output_index;
          void *input = stage == 0 ? device_x.value : scratch[input_index];
          void *output = scratch[output_index];
          const float *reference_input =
              stage == 0 ? x : references[input_index].data();
          const bool relu =
              options.modules[stage].symbol == "therock_module_relu";
          for (uint32_t i = 0; i < count; ++i) {
            const double previous_error =
                stage == 0 ? 0.0 : error_bounds[input_index][i];
            error_bounds[output_index][i] =
                therock::module_validation::pipeline_error_bound(
                    reference_input[i], y[i], alpha, relu, previous_error);
            // Store each operation as f32, preserving a sequential reference
            // even when the device compiler contracts multiply and addition.
            volatile float product = alpha * reference_input[i];
            volatile float sum = product + y[i];
            float result = sum;
            if (relu) {
              volatile float shifted = result - 0.125f;
              result = std::max(0.0f, static_cast<float>(shifted));
            }
            references[output_index][i] = result;
          }
          if (stage != 0) {
            // Level Zero defines this as both an execution and global-memory
            // barrier: finish the previous stage and expose its scratch writes
            // before the next stage reads them on the compute queue.
            ZE_CHECK(zeCommandListAppendBarrier(compute_list.value, nullptr, 0,
                                                nullptr));
          }
          const ze_kernel_handle_t kernel = modules[stage]->kernel.value;
          ZE_CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(input), &input));
          ZE_CHECK(zeKernelSetArgumentValue(kernel, 1, sizeof(void *),
                                            &device_y.value));
          ZE_CHECK(
              zeKernelSetArgumentValue(kernel, 2, sizeof(output), &output));
          ZE_CHECK(zeKernelSetArgumentValue(kernel, 3, sizeof(alpha), &alpha));
          ZE_CHECK(zeKernelSetArgumentValue(kernel, 4, sizeof(count), &count));
          ZE_CHECK(zeCommandListAppendLaunchKernel(
              compute_list.value, kernel, &groups,
              stage + 1 == options.modules.size() ? computed.value : nullptr,
              stage == 0 ? 1 : 0, stage == 0 ? &uploaded.value : nullptr));
        }
        // Neither scratch buffer is read back between stages. Both copies wait
        // for the final compute event, and a zero-wait barrier then completes
        // both copies before publishing the host-visible completion event.
        for (size_t buffer = 0; buffer < scratch.size(); ++buffer) {
          ZE_CHECK(zeCommandListAppendMemoryCopy(
              readback_list.value, host_scratch[buffer], scratch[buffer],
              max_bytes, nullptr, 1, &computed.value));
        }
        ZE_CHECK(zeCommandListAppendBarrier(readback_list.value,
                                            readback_done.value, 0, nullptr));
        ZE_CHECK(zeCommandListClose(upload_list.value));
        ZE_CHECK(zeCommandListClose(compute_list.value));
        ZE_CHECK(zeCommandListClose(readback_list.value));
        upload_submission.execute(upload_list.value);
        compute_submission.execute(compute_list.value);
        readback_submission.execute(readback_list.value);
        ZE_CHECK(zeEventHostSynchronize(readback_done.value, kWaitTimeoutNs));
        readback_submission.synchronize();
        compute_submission.synchronize();
        upload_submission.synchronize();
        std::printf(
            "SYNC backend=intel mode=cross-queue-events queues=3 events=3 "
            "n=%u round=%u stages=%zu\n",
            count, round, options.modules.size());
        double max_error = 0.0;
        for (size_t buffer = 0; buffer < scratch.size(); ++buffer) {
          const bool written = buffer < options.modules.size();
          for (size_t i = 0; i < capacity; ++i) {
            const float value = host_scratch[buffer][i];
            if (i >= count || !written) {
              if (value != kGuardValue) {
                throw std::runtime_error(
                    "Pipeline output guard overwritten buffer=" +
                    std::to_string(buffer) + " index=" + std::to_string(i));
              }
              continue;
            }
            const double error =
                std::abs(static_cast<double>(value) - references[buffer][i]);
            const double tolerance = std::max(1.0e-6, error_bounds[buffer][i]);
            if (!std::isfinite(value) || error > tolerance) {
              throw std::runtime_error(
                  "Pipeline arithmetic mismatch buffer=" +
                  std::to_string(buffer) + " index=" + std::to_string(i) +
                  " expected=" + std::to_string(references[buffer][i]) +
                  " actual=" + std::to_string(value) +
                  " tolerance=" + std::to_string(tolerance));
            }
            max_error = std::max(max_error, error);
          }
        }
        std::printf(
            "PIPELINE_CHECK n=%u round=%u stages=%zu max_abs_error=%.9g "
            "guard=pass\n",
            count, round, options.modules.size(), max_error);
        ZE_CHECK(zeCommandListReset(upload_list.value));
        ZE_CHECK(zeCommandListReset(compute_list.value));
        ZE_CHECK(zeCommandListReset(readback_list.value));
        ZE_CHECK(zeEventHostReset(uploaded.value));
        ZE_CHECK(zeEventHostReset(computed.value));
        ZE_CHECK(zeEventHostReset(readback_done.value));
      }
    }
    return;
  }
  std::printf("SESSION backend=intel modules=%zu contexts=1 buffers=3 "
              "queues=3 events=3\n",
              options.modules.size());
  for (uint32_t count : kSizes) {
    const size_t bytes = count * sizeof(float);
    const size_t output_bytes = (count + kGuardCount) * sizeof(float);
    for (uint32_t round = 0; round < 3; ++round) {
      // Alternate requests on the same allocations so stale module state or
      // missing argument rebinding cannot hide behind independent fixtures.
      for (size_t module_index = 0; module_index < options.modules.size();
           ++module_index) {
        const auto &request = options.modules[module_index];
        const ze_kernel_handle_t kernel = modules[module_index]->kernel.value;
        const float alpha = 1.75f;
        for (uint32_t i = 0; i < count; ++i) {
          x[i] = static_cast<float>(
                     static_cast<int>((i * 17 + round * 13) % 97) - 48) /
                 16.0f;
          y[i] = static_cast<float>(
                     static_cast<int>((i * 11 + round * 7) % 89) - 44) /
                 32.0f;
          expected[i] = alpha * x[i] + y[i];
          if (request.symbol == "therock_module_relu") {
            expected[i] = std::max(0.0f, expected[i] - 0.125f);
          }
        }
        std::fill_n(actual, count + kGuardCount, kGuardValue);
        ZE_CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(void *),
                                          &device_x.value));
        ZE_CHECK(zeKernelSetArgumentValue(kernel, 1, sizeof(void *),
                                          &device_y.value));
        ZE_CHECK(zeKernelSetArgumentValue(kernel, 2, sizeof(void *),
                                          &device_output.value));
        ZE_CHECK(zeKernelSetArgumentValue(kernel, 3, sizeof(alpha), &alpha));
        ZE_CHECK(zeKernelSetArgumentValue(kernel, 4, sizeof(count), &count));
        ZE_CHECK(zeCommandListAppendMemoryCopy(
            upload_list.value, device_x.value, x, bytes, nullptr, 0, nullptr));
        ZE_CHECK(zeCommandListAppendMemoryCopy(
            upload_list.value, device_y.value, y, bytes, nullptr, 0, nullptr));
        ZE_CHECK(zeCommandListAppendMemoryCopy(
            upload_list.value, device_output.value, actual, output_bytes,
            nullptr, 0, nullptr));
        // With zero waits, this barrier completes all earlier upload commands
        // before publishing the event consumed by the compute queue.
        ZE_CHECK(zeCommandListAppendBarrier(upload_list.value, uploaded.value,
                                            0, nullptr));
        const ze_group_count_t groups{(count + kGroupSize - 1) / kGroupSize, 1,
                                      1};
        ZE_CHECK(zeCommandListAppendLaunchKernel(compute_list.value, kernel,
                                                 &groups, computed.value, 1,
                                                 &uploaded.value));
        ZE_CHECK(zeCommandListAppendMemoryCopy(
            readback_list.value, actual, device_output.value, output_bytes,
            readback_done.value, 1, &computed.value));
        ZE_CHECK(zeCommandListClose(upload_list.value));
        ZE_CHECK(zeCommandListClose(compute_list.value));
        ZE_CHECK(zeCommandListClose(readback_list.value));
        // Submit producers first. If any execute call fails, no submitted
        // consumer can be waiting on a producer that was never submitted.
        upload_submission.execute(upload_list.value);
        compute_submission.execute(compute_list.value);
        readback_submission.execute(readback_list.value);
        ZE_CHECK(zeEventHostSynchronize(readback_done.value, kWaitTimeoutNs));
        // An event can become visible before its list's final bookkeeping ends.
        // Confirm every queue tail before resetting lists/events or freeing
        // memory. These host queue waits occur only after all three pipeline
        // submissions.
        readback_submission.synchronize();
        compute_submission.synchronize();
        upload_submission.synchronize();
        std::printf(
            "SYNC backend=intel mode=cross-queue-events queues=3 events=3 "
            "n=%u round=%u module_index=%zu symbol=%s\n",
            count, round, module_index, request.symbol.c_str());
        double max_error = 0.0;
        for (uint32_t i = 0; i < count; ++i) {
          const double error =
              std::abs(static_cast<double>(actual[i]) - expected[i]);
          // Bounded dyadic inputs and a few FP32 rounding steps permit FMA
          // contraction without hiding omitted kernels, bad tails or stale
          // data.
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
        std::printf("CHECK n=%u round=%u max_abs_error=%.9g guard=pass "
                    "module_index=%zu symbol=%s\n",
                    count, round, max_error, module_index,
                    request.symbol.c_str());
        ZE_CHECK(zeCommandListReset(upload_list.value));
        ZE_CHECK(zeCommandListReset(compute_list.value));
        ZE_CHECK(zeCommandListReset(readback_list.value));
        ZE_CHECK(zeEventHostReset(uploaded.value));
        ZE_CHECK(zeEventHostReset(computed.value));
        ZE_CHECK(zeEventHostReset(readback_done.value));
      }
    }
  }
}

} // namespace

#include "module_service_level_zero.h"

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  try {
    if (therock::module_service::requested(argc, argv)) {
      return therock::module_service::serve<ServiceBackend>();
    }
    if (therock::module_validation::describe_contract(argc, argv)) {
      return 0;
    }
    if (therock::module_validation::list_devices_requested(argc, argv)) {
      list_devices();
      return 0;
    }
    const Options options = parse_options(argc, argv);
    run(options);
    if (cleanup_failed) {
      throw std::runtime_error("Level Zero resource cleanup failed");
    }
    if (options.pipeline) {
      std::printf("PASS backend=intel runtime=level-zero "
                  "mode=device-module-pipeline stages=%zu\n",
                  options.modules.size());
    } else if (options.batch) {
      std::printf("PASS backend=intel runtime=level-zero modules=%zu\n",
                  options.modules.size());
    } else {
      std::printf("PASS backend=intel runtime=level-zero symbol=%s\n",
                  options.symbol.c_str());
    }
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL backend=intel error=%s\n", error.what());
    return 1;
  }
}
