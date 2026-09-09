// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#if defined(THEROCK_MODULE_NVIDIA)
#include <cuda.h>
#else
#include <hip/hip_runtime_api.h>
#endif

#include "device_inventory.h"
#include "module_service_protocol.h"
#include "module_session_options.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kGroupSize = therock::module_contract::kGroupSize;
constexpr uint32_t kSizes[] = {1, 127, 128, 129, 4099, 65539};
constexpr uint32_t kGuardCount = 17;
constexpr uint32_t kMaxCount = 65539;
constexpr uint32_t kMaxGridSize = (kMaxCount + kGroupSize - 1) / kGroupSize;
constexpr size_t kMaxAllocationBytes =
    (kMaxCount + kGuardCount) * sizeof(float);
constexpr size_t kRequiredDeviceBytes =
    (3 * kMaxCount + kGuardCount) * sizeof(float);
constexpr size_t kPipelineRequiredDeviceBytes =
    2 * kMaxCount * sizeof(float) + 2 * kMaxAllocationBytes;
static_assert(kMaxGridSize == 513,
              "Update geometry checks for changed fixtures");

#if defined(THEROCK_MODULE_NVIDIA)
constexpr const char *kBackend = "nvidia";
using Error = CUresult;
using DevicePointer = CUdeviceptr;
using ModuleHandle = CUmodule;
using FunctionHandle = CUfunction;
using StreamHandle = CUstream;
using EventHandle = CUevent;
constexpr Error kSuccess = CUDA_SUCCESS;
constexpr Error kNotReady = CUDA_ERROR_NOT_READY;
std::string error_string(Error error) {
  const char *name = nullptr;
  const char *description = nullptr;
  cuGetErrorName(error, &name);
  cuGetErrorString(error, &description);
  return std::string(name ? name : "unknown CUDA error") + ": " +
         (description ? description : "no description");
}
#else
constexpr const char *kBackend = "amd";
using Error = hipError_t;
using DevicePointer = void *;
using ModuleHandle = hipModule_t;
using FunctionHandle = hipFunction_t;
using StreamHandle = hipStream_t;
using EventHandle = hipEvent_t;
constexpr Error kSuccess = hipSuccess;
constexpr Error kNotReady = hipErrorNotReady;
std::string error_string(Error error) { return hipGetErrorString(error); }
#endif

static_assert(sizeof(DevicePointer) * CHAR_BIT ==
                  therock::module_contract::kPointerBits,
              "Native device pointers must match the module contract");
static_assert(alignof(DevicePointer) == 8,
              "Native device pointers require eight-byte alignment");
bool cleanup_failed = false;
void check(Error error, const char *operation, int line) {
  if (error != kSuccess) {
    throw std::runtime_error(std::string(operation) + " at line " +
                             std::to_string(line) + ": " + error_string(error));
  }
}
#define GPU_CHECK(operation) check((operation), #operation, __LINE__)

void check_cleanup(Error error, const char *operation) {
  if (error != kSuccess) {
    std::fprintf(stderr, "FAIL cleanup=%s error=%s\n", operation,
                 error_string(error).c_str());
    cleanup_failed = true;
  }
}

struct Options {
  std::string payload;
  std::string symbol = "therock_module_saxpy";
  std::string expected_arch;
  std::string expected_device_uuid;
  int device = 0;
  therock::module_validation::ContractOptions contract;
  std::vector<therock::module_validation::ModuleRequest> modules;
  bool batch = false;
  bool pipeline = false;
};

Options parse_options(int argc, char **argv) {
  Options options;
  std::unordered_set<std::string> seen;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      std::puts(
          "--payload PATH --symbol therock_module_saxpy|therock_module_relu "
          "--device INDEX --expect-arch gfx1201|sm_120 "
          "[--expect-device-uuid UUID] "
          "--launch-abi ABI --launch-abi-version VERSION "
          "--launch-contract-sha256 SHA256 --payload-format hsaco|cubin|ptx\n"
          "--module FORMAT SYMBOL PATH (repeatable, at most 32; replaces "
          "--payload, --symbol, and --payload-format)\n"
          "--pipeline (with --module; device-resident stages, repeats "
          "allowed)\n"
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
    } else if (arg == "--expect-device-uuid") {
      therock::module_validation::validate_device_uuid(value);
      options.expected_device_uuid = value;
    } else if (arg == "--expect-arch") {
      options.expected_arch = value;
    } else if (arg == "--device") {
      size_t consumed = 0;
      options.device = std::stoi(value, &consumed);
      if (consumed != value.size() || options.device < 0) {
        throw std::runtime_error("--device must be a nonnegative integer");
      }
    } else if (!options.contract.consume(arg, value)) {
      throw std::runtime_error("Unknown argument " + arg);
    }
  }
  options.batch = !options.modules.empty();
#if defined(THEROCK_MODULE_NVIDIA)
  options.modules = therock::module_validation::finalize_module_requests(
      options.contract, options.payload, options.symbol, options.modules, seen,
      {"cubin", "ptx"}, options.pipeline);
#else
  options.modules = therock::module_validation::finalize_module_requests(
      options.contract, options.payload, options.symbol, options.modules, seen,
      {"hsaco"}, options.pipeline);
#endif
  return options;
}

std::string inspect_payload(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Cannot read payload " + path);
  }
  std::array<char, 4096> header{};
  input.read(header.data(), header.size());
  const size_t size = static_cast<size_t>(input.gcount());
  if (size >= 20 && static_cast<unsigned char>(header[0]) == 0x7f &&
      header[1] == 'E' && header[2] == 'L' && header[3] == 'F') {
    if (header[4] != 2 || header[5] != 1) {
      throw std::runtime_error(
          "Expected a 64-bit little-endian GPU ELF payload");
    }
    const unsigned machine = static_cast<unsigned char>(header[18]) |
                             (static_cast<unsigned char>(header[19]) << 8);
#if defined(THEROCK_MODULE_NVIDIA)
    constexpr unsigned expected_machine = 190; // EM_CUDA
    const std::string format = "cubin";
#else
    constexpr unsigned expected_machine = 224; // EM_AMDGPU
    const std::string format = "hsaco";
#endif
    if (machine != expected_machine) {
      throw std::runtime_error(
          "Wrong payload ELF machine=" + std::to_string(machine) +
          " expected=" + std::to_string(expected_machine));
    }
    return format;
  }
#if defined(THEROCK_MODULE_NVIDIA)
  const std::string text(header.data(), size);
  if (text.find('\0') == std::string::npos &&
      text.find(".version") != std::string::npos &&
      text.find(".target") != std::string::npos &&
      text.find(".address_size 64") != std::string::npos) {
    // CUDA performs the full PTX parse and JIT. This is format identification,
    // never a request to fall back from a rejected native binary to another
    // one.
    return "ptx";
  }
#endif
  throw std::runtime_error("Unrecognized native payload format for " +
                           std::string(kBackend));
}

#if defined(THEROCK_MODULE_NVIDIA)
std::string query_device_uuid(CUdevice device) {
  CUuuid uuid{};
  // CUDA maps this API to v2, which distinguishes MIG compute instances.
  GPU_CHECK(cuDeviceGetUuid(&uuid, device));
#else
std::string query_device_uuid(hipDevice_t device) {
  hipUUID uuid{};
  GPU_CHECK(hipDeviceGetUuid(&uuid, device));
#endif
  return therock::module_validation::device_uuid_hex(uuid.bytes);
}

void list_devices() {
  using namespace therock::module_validation;
  int count = 0;
#if defined(THEROCK_MODULE_NVIDIA)
  GPU_CHECK(cuInit(0));
  GPU_CHECK(cuDeviceGetCount(&count));
#else
  GPU_CHECK(hipGetDeviceCount(&count));
#endif
  if (count < 0) {
    throw std::runtime_error("Native runtime reports a negative device count");
  }
  std::vector<DeviceInventoryEntry> devices;
  for (int index = 0; index < count; ++index) {
    DeviceInventoryEntry entry;
    entry.index = static_cast<uint64_t>(index);
#if defined(THEROCK_MODULE_NVIDIA)
    CUdevice device{};
    GPU_CHECK(cuDeviceGet(&device, index));
    entry.device_uuid = query_device_uuid(device);
    const auto attribute = [device](CUdevice_attribute key) {
      int value = 0;
      GPU_CHECK(cuDeviceGetAttribute(&value, key, device));
      return value;
    };
    char name[256]{};
    GPU_CHECK(cuDeviceGetName(name, sizeof(name), device));
    entry.name = device_name(name);
    const int major = attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    const int minor = attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    if (major < 1 || minor < 0 || minor > 9) {
      throw std::runtime_error(
          "NVIDIA runtime reports an invalid architecture");
    }
    entry.architecture = "sm_" + std::to_string(major) + std::to_string(minor);
    entry.vendor_id = 0x10de;
    entry.max_threads_per_block = positive_device_limit(
        attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK));
    entry.max_block_dimensions = {
        positive_device_limit(attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)),
        positive_device_limit(attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)),
        positive_device_limit(attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z))};
    entry.max_grid_dimensions = {
        positive_device_limit(attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)),
        positive_device_limit(attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)),
        positive_device_limit(attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z))};
    size_t total_memory = 0;
    GPU_CHECK(cuDeviceTotalMem(&total_memory, device));
    entry.total_memory_bytes = total_memory;
#else
    entry.device_uuid = query_device_uuid(index);
    hipDeviceProp_t properties{};
    GPU_CHECK(hipGetDeviceProperties(&properties, index));
    entry.name = device_name(properties.name);
    std::string architecture = device_name(properties.gcnArchName);
    architecture = architecture.substr(0, architecture.find(':'));
    if (architecture.rfind("gfx", 0) != 0) {
      throw std::runtime_error(
          "AMD native runtime did not report a gfx device");
    }
    entry.architecture = architecture;
    entry.vendor_id = 0x1002;
    entry.max_threads_per_block =
        positive_device_limit(properties.maxThreadsPerBlock);
    entry.max_block_dimensions = {
        positive_device_limit(properties.maxThreadsDim[0]),
        positive_device_limit(properties.maxThreadsDim[1]),
        positive_device_limit(properties.maxThreadsDim[2])};
    entry.max_grid_dimensions = {
        positive_device_limit(properties.maxGridSize[0]),
        positive_device_limit(properties.maxGridSize[1]),
        positive_device_limit(properties.maxGridSize[2])};
    entry.total_memory_bytes = properties.totalGlobalMem;
#endif
    devices.push_back(entry);
  }
#if defined(THEROCK_MODULE_NVIDIA)
  const std::string result = device_inventory_json("nvidia", "cuda", devices);
#else
  const std::string result = device_inventory_json("amd", "hip", devices);
#endif
  // Publish only after the full enumeration and all property queries succeed.
  std::puts(result.c_str());
}

void validate_device_geometry(int max_threads, const std::array<int, 3> &block,
                              const std::array<int, 3> &grid,
                              size_t total_memory) {
  if (max_threads < static_cast<int>(kGroupSize) ||
      block[0] < static_cast<int>(kGroupSize) || block[1] < 1 || block[2] < 1 ||
      grid[0] < static_cast<int>(kMaxGridSize) || grid[1] < 1 || grid[2] < 1) {
    throw std::runtime_error(
        "Device does not support the validation launch geometry");
  }
  if (total_memory < kRequiredDeviceBytes) {
    throw std::runtime_error(
        "Device memory is too small for the validation fixture");
  }
  std::printf("CAPABILITY backend=%s device_max_threads=%d max_block_x=%d "
              "max_grid_x=%d total_memory_bytes=%zu required_grid_x=%u\n",
              kBackend, max_threads, block[0], grid[0], total_memory,
              kMaxGridSize);
}

class Device {
public:
  explicit Device(const Options &options) {
    int count = 0;
    std::string architecture;
    std::string name;
    std::string device_uuid;
#if defined(THEROCK_MODULE_NVIDIA)
    GPU_CHECK(cuInit(0));
    GPU_CHECK(cuDeviceGetCount(&count));
#else
    GPU_CHECK(hipGetDeviceCount(&count));
#endif
    if (options.device >= count) {
      throw std::runtime_error(
          "Requested device " + std::to_string(options.device) +
          " but native backend found " + std::to_string(count));
    }
#if defined(THEROCK_MODULE_NVIDIA)
    GPU_CHECK(cuDeviceGet(&device_, options.device));
    device_uuid = query_device_uuid(device_);
    therock::module_validation::require_device_uuid(
        options.expected_device_uuid, device_uuid);
    int major = 0, minor = 0;
    GPU_CHECK(cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_));
    GPU_CHECK(cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_));
    char device_name[256]{};
    GPU_CHECK(cuDeviceGetName(device_name, sizeof(device_name), device_));
    name = device_name;
    architecture = "sm_" + std::to_string(major) + std::to_string(minor);
    const auto attribute = [this](CUdevice_attribute key) {
      int value = 0;
      GPU_CHECK(cuDeviceGetAttribute(&value, key, device_));
      return value;
    };
    size_t total_memory = 0;
    GPU_CHECK(cuDeviceTotalMem(&total_memory, device_));
    const int shared_memory =
        attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
    if (shared_memory < 0) {
      throw std::runtime_error("Device reports an invalid shared memory limit");
    }
    shared_memory_per_block_ = static_cast<size_t>(shared_memory);
    validate_device_geometry(
        attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
        {attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
         attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
         attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)},
        {attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X),
         attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y),
         attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)},
        total_memory);
#else
    device_uuid = query_device_uuid(options.device);
    therock::module_validation::require_device_uuid(
        options.expected_device_uuid, device_uuid);
    GPU_CHECK(hipSetDevice(options.device));
    hipDeviceProp_t props{};
    GPU_CHECK(hipGetDeviceProperties(&props, options.device));
    name = props.name;
    shared_memory_per_block_ = props.sharedMemPerBlock;
    validate_device_geometry(
        props.maxThreadsPerBlock,
        {props.maxThreadsDim[0], props.maxThreadsDim[1],
         props.maxThreadsDim[2]},
        {props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]},
        props.totalGlobalMem);
    architecture = props.gcnArchName;
    architecture = architecture.substr(0, architecture.find(':'));
    if (architecture.rfind("gfx", 0) != 0) {
      throw std::runtime_error(
          "AMD native runtime did not report a gfx device");
    }
#endif
    std::printf("DEVICE backend=%s index=%d count=%d name=\"%s\" arch=%s "
                "device_uuid=%s\n",
                kBackend, options.device, count, name.c_str(),
                architecture.c_str(), device_uuid.c_str());
    if (!options.expected_arch.empty() &&
        options.expected_arch != architecture) {
      throw std::runtime_error("Expected architecture " +
                               options.expected_arch + " but got " +
                               architecture);
    }
#if defined(THEROCK_MODULE_NVIDIA)
    GPU_CHECK(cuDevicePrimaryCtxRetain(&context_, device_));
    const Error status = cuCtxSetCurrent(context_);
    if (status != kSuccess) {
      check_cleanup(cuDevicePrimaryCtxRelease(device_),
                    "cuDevicePrimaryCtxRelease");
      GPU_CHECK(status);
    }
#endif
  }
  ~Device() {
#if defined(THEROCK_MODULE_NVIDIA)
    check_cleanup(cuCtxSetCurrent(nullptr), "cuCtxSetCurrent");
    check_cleanup(cuDevicePrimaryCtxRelease(device_),
                  "cuDevicePrimaryCtxRelease");
#endif
  }
  void validate_kernel(FunctionHandle function,
                       size_t required_device_bytes) const {
    int max_threads = 0, static_shared_bytes = 0;
    size_t free_memory = 0, total_memory = 0;
#if defined(THEROCK_MODULE_NVIDIA)
    GPU_CHECK(cuFuncGetAttribute(
        &max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function));
    GPU_CHECK(cuFuncGetAttribute(
        &static_shared_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function));
    GPU_CHECK(cuMemGetInfo(&free_memory, &total_memory));
#else
    GPU_CHECK(hipFuncGetAttribute(
        &max_threads, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function));
    GPU_CHECK(hipFuncGetAttribute(
        &static_shared_bytes, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function));
    GPU_CHECK(hipMemGetInfo(&free_memory, &total_memory));
#endif
    if (max_threads < static_cast<int>(kGroupSize) || static_shared_bytes < 0 ||
        static_cast<size_t>(static_shared_bytes) > shared_memory_per_block_) {
      throw std::runtime_error(
          "Kernel does not support the validation launch geometry");
    }
    if (free_memory < required_device_bytes ||
        total_memory < required_device_bytes) {
      throw std::runtime_error(
          "Insufficient available device memory for validation allocations");
    }
    std::printf("CAPABILITY backend=%s kernel_max_threads=%d "
                "static_shared_bytes=%d free_memory_bytes=%zu "
                "required_memory_bytes=%zu largest_allocation_bytes=%zu\n",
                kBackend, max_threads, static_shared_bytes, free_memory,
                required_device_bytes, kMaxAllocationBytes);
  }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;

private:
  size_t shared_memory_per_block_ = 0;
#if defined(THEROCK_MODULE_NVIDIA)
  CUdevice device_{};
  CUcontext context_{};
#endif
};

class Module {
public:
  explicit Module(const std::string &path) {
#if defined(THEROCK_MODULE_NVIDIA)
    GPU_CHECK(cuModuleLoad(&value, path.c_str()));
#else
    GPU_CHECK(hipModuleLoad(&value, path.c_str()));
#endif
  }
  ~Module() {
#if defined(THEROCK_MODULE_NVIDIA)
    check_cleanup(cuModuleUnload(value), "cuModuleUnload");
#else
    check_cleanup(hipModuleUnload(value), "hipModuleUnload");
#endif
  }
  FunctionHandle get_function(const std::string &symbol) {
    FunctionHandle function{};
#if defined(THEROCK_MODULE_NVIDIA)
    GPU_CHECK(cuModuleGetFunction(&function, value, symbol.c_str()));
#else
    GPU_CHECK(hipModuleGetFunction(&function, value, symbol.c_str()));
#endif
    return function;
  }
  Module(const Module &) = delete;
  Module &operator=(const Module &) = delete;
  ModuleHandle value{};
};

// An asynchronous failure cannot establish whether queued commands still own
// these resources. Bound both ordinary host waits and exception-path draining;
// never continue unwinding and freeing buffers/modules with unknown completion.
constexpr auto kCompletionTimeout = std::chrono::seconds(30);
constexpr auto kCompletionPollInterval = std::chrono::milliseconds(1);

[[noreturn]] void unknown_completion(const char *operation, Error status,
                                     const char *reason) noexcept {
  std::fprintf(
      stderr,
      "FAIL backend=%s operation=%s status=0x%08x reason=%s timeout_ms=30000 "
      "completion=unknown; terminating without freeing in-flight resources\n",
      kBackend, operation, static_cast<unsigned>(status), reason);
  std::fflush(nullptr);
  std::_Exit(EXIT_FAILURE);
}

template <typename Query>
void wait_for_completion(Query query, const char *operation) noexcept {
  const auto deadline = std::chrono::steady_clock::now() + kCompletionTimeout;
  while (true) {
    const Error status = query();
    if (status == kSuccess) {
      return;
    }
    if (status != kNotReady) {
      unknown_completion(operation, status, "query-error");
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      unknown_completion(operation, status, "timeout");
    }
    std::this_thread::sleep_for(kCompletionPollInterval);
  }
}

class Event {
public:
  Event() {
#if defined(THEROCK_MODULE_NVIDIA)
    GPU_CHECK(cuEventCreate(&value, CU_EVENT_DISABLE_TIMING));
#else
    GPU_CHECK(hipEventCreateWithFlags(&value, hipEventDisableTiming));
#endif
  }
  ~Event() {
#if defined(THEROCK_MODULE_NVIDIA)
    check_cleanup(cuEventDestroy(value), "cuEventDestroy");
#else
    check_cleanup(hipEventDestroy(value), "hipEventDestroy");
#endif
  }
  void record(StreamHandle stream) {
#if defined(THEROCK_MODULE_NVIDIA)
    GPU_CHECK(cuEventRecord(value, stream));
#else
    GPU_CHECK(hipEventRecord(value, stream));
#endif
  }
  void wait(StreamHandle stream) {
#if defined(THEROCK_MODULE_NVIDIA)
    GPU_CHECK(cuStreamWaitEvent(stream, value, 0));
#else
    GPU_CHECK(hipStreamWaitEvent(stream, value, 0));
#endif
  }
  void wait_host() const noexcept {
#if defined(THEROCK_MODULE_NVIDIA)
    wait_for_completion([this] { return cuEventQuery(value); }, "cuEventQuery");
#else
    wait_for_completion([this] { return hipEventQuery(value); },
                        "hipEventQuery");
#endif
  }
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
  EventHandle value{};
};

class Stream {
public:
  Stream() {
#if defined(THEROCK_MODULE_NVIDIA)
    GPU_CHECK(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING));
#else
    GPU_CHECK(hipStreamCreateWithFlags(&value, hipStreamNonBlocking));
#endif
  }
  ~Stream() {
    // Every queue must complete before the enclosing fixture releases any
    // events or allocations. This also covers partially submitted pipelines.
#if defined(THEROCK_MODULE_NVIDIA)
    wait_for_completion([this] { return cuStreamQuery(value); },
                        "cuStreamQuery");
    check_cleanup(cuStreamDestroy(value), "cuStreamDestroy");
#else
    wait_for_completion([this] { return hipStreamQuery(value); },
                        "hipStreamQuery");
    check_cleanup(hipStreamDestroy(value), "hipStreamDestroy");
#endif
  }
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
  StreamHandle value{};
};

class Buffer {
public:
  explicit Buffer(size_t bytes) {
#if defined(THEROCK_MODULE_NVIDIA)
    GPU_CHECK(cuMemAlloc(&value, bytes));
#else
    GPU_CHECK(hipMalloc(&value, bytes));
#endif
  }
  ~Buffer() {
#if defined(THEROCK_MODULE_NVIDIA)
    check_cleanup(cuMemFree(value), "cuMemFree");
#else
    check_cleanup(hipFree(value), "hipFree");
#endif
  }
  void from_host(const float *input, size_t count, StreamHandle stream) {
#if defined(THEROCK_MODULE_NVIDIA)
    GPU_CHECK(cuMemcpyHtoDAsync(value, input, count * sizeof(float), stream));
#else
    GPU_CHECK(hipMemcpyAsync(value, input, count * sizeof(float),
                             hipMemcpyHostToDevice, stream));
#endif
  }
  void to_host(float *output, size_t count, StreamHandle stream) {
#if defined(THEROCK_MODULE_NVIDIA)
    GPU_CHECK(cuMemcpyDtoHAsync(output, value, count * sizeof(float), stream));
#else
    GPU_CHECK(hipMemcpyAsync(output, value, count * sizeof(float),
                             hipMemcpyDeviceToHost, stream));
#endif
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  DevicePointer value{};
};

class HostBuffer {
public:
  explicit HostBuffer(size_t count) : count_(count) {
#if defined(THEROCK_MODULE_NVIDIA)
    GPU_CHECK(cuMemAllocHost(reinterpret_cast<void **>(&data_),
                             count * sizeof(float)));
#else
    GPU_CHECK(hipHostMalloc(reinterpret_cast<void **>(&data_),
                            count * sizeof(float)));
#endif
  }
  ~HostBuffer() {
#if defined(THEROCK_MODULE_NVIDIA)
    check_cleanup(cuMemFreeHost(data_), "cuMemFreeHost");
#else
    check_cleanup(hipHostFree(data_), "hipHostFree");
#endif
  }
  HostBuffer(const HostBuffer &) = delete;
  HostBuffer &operator=(const HostBuffer &) = delete;
  float *data() { return data_; }
  float *begin() { return data_; }
  float *end() { return data_ + count_; }
  float &operator[](size_t i) { return data_[i]; }
  size_t size() const { return count_; }

private:
  float *data_ = nullptr;
  size_t count_;
};

struct LoadedModule {
  std::unique_ptr<Module> module;
  FunctionHandle function{};
};

void run_pipeline(const Options &options,
                  const std::vector<LoadedModule> &modules) {
  // Scratch outputs alternate, so no stage overwrites its input. All four
  // allocations survive the entire session, including both retained outputs.
  Buffer x(kMaxCount * sizeof(float)), y(kMaxCount * sizeof(float));
  Buffer scratch_a(kMaxAllocationBytes), scratch_b(kMaxAllocationBytes);
  HostBuffer host_x(kMaxCount), host_y(kMaxCount);
  HostBuffer actual_a(kMaxCount + kGuardCount),
      actual_b(kMaxCount + kGuardCount);
  const std::array<Buffer *, 2> scratch{&scratch_a, &scratch_b};
  const std::array<HostBuffer *, 2> actual{&actual_a, &actual_b};
  std::array<std::vector<float>, 2> expected{std::vector<float>(kMaxCount),
                                             std::vector<float>(kMaxCount)};
  std::array<std::vector<double>, 2> error_bounds{
      std::vector<double>(kMaxCount), std::vector<double>(kMaxCount)};
  // Queue owners unwind first, then events and allocations. Loaded modules
  // remain owned by the caller until this complete pipeline scope has drained.
  Event uploaded, computed, readback_done;
  Stream upload, compute, readback;
  std::printf("SESSION backend=%s mode=device-module-pipeline modules=%zu "
              "contexts=1 buffers=4 queues=3 events=3\n",
              kBackend, modules.size());
  constexpr float guard_value = -12345.5f;
  for (uint32_t count : kSizes) {
    for (int round = 0; round < 3; ++round) {
      float alpha = 1.75f;
      for (unsigned i = 0; i < count; ++i) {
        host_x[i] = static_cast<float>(
                        static_cast<int>((i * 17 + round * 13) % 97) - 48) /
                    16.0f;
        host_y[i] = static_cast<float>(
                        static_cast<int>((i * 11 + round * 7) % 89) - 44) /
                    32.0f;
      }
      for (HostBuffer *buffer : actual) {
        std::fill(buffer->begin(), buffer->end(), guard_value);
      }
      x.from_host(host_x.data(), count, upload.value);
      y.from_host(host_y.data(), count, upload.value);
      scratch_a.from_host(actual_a.data(), actual_a.size(), upload.value);
      scratch_b.from_host(actual_b.data(), actual_b.size(), upload.value);
      uploaded.record(upload.value);
      uploaded.wait(compute.value);
      std::array<bool, 2> written{};
      for (size_t stage = 0; stage < modules.size(); ++stage) {
        const size_t output_index = stage % 2;
        const size_t input_index = (stage + 1) % 2;
        const auto &request = options.modules[stage];
        const bool relu = request.symbol == "therock_module_relu";
        // Stored float intermediates model each separately executed stage.
        // Keep both scratch references: one can contain a penultimate result,
        // or remain entirely unwritten for a one-stage pipeline.
        for (unsigned i = 0; i < count; ++i) {
          const float input = stage == 0 ? host_x[i] : expected[input_index][i];
          const double previous_error =
              stage == 0 ? 0.0 : error_bounds[input_index][i];
          float result = alpha * input + host_y[i];
          if (relu) {
            result = std::max(0.0f, result - 0.125f);
          }
          expected[output_index][i] = result;
          error_bounds[output_index][i] =
              therock::module_validation::pipeline_error_bound(
                  input, host_y[i], alpha, relu, previous_error);
        }
        written[output_index] = true;
        DevicePointer input =
            stage == 0 ? x.value : scratch[input_index]->value;
        DevicePointer output = scratch[output_index]->value;
        void *arguments[] = {&input, &y.value, &output, &alpha, &count};
        const unsigned grid_size = (count + kGroupSize - 1) / kGroupSize;
        const FunctionHandle function = modules[stage].function;
        // Same-stream ordering provides the dependencies between stages. There
        // is no intermediate host wait, upload, or device-to-host transfer.
#if defined(THEROCK_MODULE_NVIDIA)
        GPU_CHECK(cuLaunchKernel(function, grid_size, 1, 1, kGroupSize, 1, 1, 0,
                                 compute.value, arguments, nullptr));
#else
        GPU_CHECK(hipModuleLaunchKernel(function, grid_size, 1, 1, kGroupSize,
                                        1, 1, 0, compute.value, arguments,
                                        nullptr));
#endif
      }
      computed.record(compute.value);
      computed.wait(readback.value);
      scratch_a.to_host(actual_a.data(), actual_a.size(), readback.value);
      scratch_b.to_host(actual_b.data(), actual_b.size(), readback.value);
      readback_done.record(readback.value);
      readback_done.wait_host();
      std::printf("SYNC backend=%s mode=cross-queue-events queues=3 events=3\n",
                  kBackend);
      double max_error = 0.0;
      for (size_t buffer_index = 0; buffer_index < scratch.size();
           ++buffer_index) {
        HostBuffer &observed = *actual[buffer_index];
        const size_t active_count = written[buffer_index] ? count : 0;
        for (size_t i = 0; i < active_count; ++i) {
          const float value = actual[buffer_index]->data()[i];
          const float reference = expected[buffer_index][i];
          const double error = std::abs(static_cast<double>(value) - reference);
          const double tolerance =
              std::max(1.0e-6, error_bounds[buffer_index][i]);
          if (!std::isfinite(value) || !std::isfinite(reference) ||
              !std::isfinite(tolerance) || error > tolerance) {
            throw std::runtime_error("Pipeline arithmetic mismatch scratch=" +
                                     std::to_string(buffer_index) +
                                     " index=" + std::to_string(i) +
                                     " expected=" + std::to_string(reference) +
                                     " actual=" + std::to_string(value) +
                                     " tolerance=" + std::to_string(tolerance));
          }
          max_error = std::max(max_error, error);
        }
        for (size_t i = active_count; i < observed.size(); ++i) {
          if (actual[buffer_index]->data()[i] != guard_value) {
            throw std::runtime_error(
                "Pipeline output guard overwritten scratch=" +
                std::to_string(buffer_index) + " index=" + std::to_string(i));
          }
        }
      }
      std::printf("PIPELINE_CHECK n=%u round=%d stages=%zu max_abs_error=%.9g "
                  "guard=pass\n",
                  count, round, modules.size(), max_error);
    }
  }
}

void run(const Options &options) {
  // Reject every malformed or wrongly declared payload before initializing a
  // device, including failures in later requests of a batch.
  for (const auto &request : options.modules) {
    const std::string format = inspect_payload(request.payload);
    auto request_contract = options.contract;
    request_contract.payload_format = request.payload_format;
    request_contract.validate_inspected_format(format);
  }
  Device device(options);
  std::vector<LoadedModule> modules;
  modules.reserve(options.modules.size());
  for (const auto &request : options.modules) {
    auto module = std::make_unique<Module>(request.payload);
    const FunctionHandle function = module->get_function(request.symbol);
    device.validate_kernel(function, options.pipeline
                                         ? kPipelineRequiredDeviceBytes
                                         : kRequiredDeviceBytes);
    modules.push_back({std::move(module), function});
    std::printf("MODULE backend=%s format=%s symbol=%s payload=\"%s\"\n",
                kBackend, request.payload_format.c_str(),
                request.symbol.c_str(), request.payload.c_str());
  }
  // All modules, symbols, and kernel limits are accepted before allocating or
  // submitting fixture work. Keep them alive until every queue has drained.
  if (options.pipeline) {
    run_pipeline(options, modules);
    return;
  }
  Buffer x(kMaxCount * sizeof(float)), y(kMaxCount * sizeof(float));
  Buffer output(kMaxAllocationBytes);
  HostBuffer host_x(kMaxCount), host_y(kMaxCount);
  HostBuffer actual(kMaxCount + kGuardCount);
  std::vector<float> expected(kMaxCount);
  // Reverse destruction drains all queues before destroying their events,
  // then releases host/device buffers and finally the complete module set.
  // This order also covers a partially submitted later module in the session.
  Event uploaded, computed, readback_done;
  Stream upload, compute, readback;
  std::printf("SESSION backend=%s modules=%zu contexts=1 buffers=3 queues=3 "
              "events=3\n",
              kBackend, modules.size());
  constexpr float guard_value = -12345.5f;
  for (uint32_t count : kSizes) {
    for (int round = 0; round < 3; ++round) {
      // Alternate modules within each size/round, exercising shared allocation
      // and event reuse across distinct simultaneously loaded code objects.
      for (size_t module_index = 0; module_index < modules.size();
           ++module_index) {
        const auto &request = options.modules[module_index];
        const FunctionHandle function = modules[module_index].function;
        float alpha = 1.75f;
        for (unsigned i = 0; i < count; ++i) {
          host_x[i] = static_cast<float>(
                          static_cast<int>((i * 17 + round * 13) % 97) - 48) /
                      16.0f;
          host_y[i] = static_cast<float>(
                          static_cast<int>((i * 11 + round * 7) % 89) - 44) /
                      32.0f;
          expected[i] = alpha * host_x[i] + host_y[i];
          if (request.symbol == "therock_module_relu") {
            expected[i] = std::max(0.0f, expected[i] - 0.125f);
          }
        }
        // Guard the complete reusable output allocation, including the inactive
        // tail for smaller inputs and any results left by a previous module.
        std::fill(actual.begin(), actual.end(), guard_value);
        x.from_host(host_x.data(), count, upload.value);
        y.from_host(host_y.data(), count, upload.value);
        output.from_host(actual.data(), actual.size(), upload.value);
        uploaded.record(upload.value);
        uploaded.wait(compute.value);
        void *arguments[] = {&x.value, &y.value, &output.value, &alpha, &count};
        const unsigned grid_size = (count + kGroupSize - 1) / kGroupSize;
#if defined(THEROCK_MODULE_NVIDIA)
        GPU_CHECK(cuLaunchKernel(function, grid_size, 1, 1, kGroupSize, 1, 1, 0,
                                 compute.value, arguments, nullptr));
#else
        GPU_CHECK(hipModuleLaunchKernel(function, grid_size, 1, 1, kGroupSize,
                                        1, 1, 0, compute.value, arguments,
                                        nullptr));
#endif
        computed.record(compute.value);
        computed.wait(readback.value);
        output.to_host(actual.data(), actual.size(), readback.value);
        readback_done.record(readback.value);
        readback_done.wait_host();
        std::printf(
            "SYNC backend=%s mode=cross-queue-events queues=3 events=3\n",
            kBackend);
        double max_error = 0.0;
        for (unsigned i = 0; i < count; ++i) {
          const double error =
              std::abs(static_cast<double>(actual[i]) - expected[i]);
          // A few FP32 rounding steps allow optional FMA contraction. The
          // inputs are bounded dyadic fractions, so failures cannot hide in
          // loose error.
          const double tolerance = 1.0e-6 + 2.0e-6 * std::abs(expected[i]);
          if (!std::isfinite(actual[i]) || error > tolerance) {
            throw std::runtime_error(
                "Arithmetic mismatch module_index=" +
                std::to_string(module_index) + " symbol=" + request.symbol +
                " index=" + std::to_string(i) +
                " expected=" + std::to_string(expected[i]) +
                " actual=" + std::to_string(actual[i]));
          }
          max_error = std::max(max_error, error);
        }
        for (unsigned i = count; i < actual.size(); ++i) {
          if (actual[i] != guard_value) {
            throw std::runtime_error("Output guard overwritten module_index=" +
                                     std::to_string(module_index) +
                                     " symbol=" + request.symbol +
                                     " index=" + std::to_string(i));
          }
        }
        std::printf("CHECK n=%u round=%d max_abs_error=%.9g guard=pass "
                    "module_index=%zu symbol=%s\n",
                    count, round, max_error, module_index,
                    request.symbol.c_str());
      }
    }
  }
}

} // namespace

#include "module_service_hip_cuda.h"

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  try {
    if (therock::module_validation::describe_contract(argc, argv)) {
      return 0;
    }
    if (therock::module_validation::list_devices_requested(argc, argv)) {
      list_devices();
      return 0;
    }
    if (therock::module_service::requested(argc, argv)) {
      return therock::module_service::serve<ServiceBackend>();
    }
    const Options options = parse_options(argc, argv);
    run(options);
    if (cleanup_failed) {
      throw std::runtime_error("Native API resource cleanup failed");
    }
    if (options.pipeline) {
      std::printf("PASS backend=%s pipeline stages=%zu\n", kBackend,
                  options.modules.size());
    } else if (options.batch) {
      std::printf("PASS backend=%s modules=%zu\n", kBackend,
                  options.modules.size());
    } else {
      std::printf("PASS backend=%s symbol=%s\n", kBackend,
                  options.symbol.c_str());
    }
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL backend=%s error=%s\n", kBackend, error.what());
    return 1;
  }
}
