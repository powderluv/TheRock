// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#if defined(THEROCK_MODULE_NVIDIA)
#include <cuda.h>
#else
#include <hip/hip_runtime_api.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#if defined(THEROCK_MODULE_NVIDIA)
constexpr const char *kBackend = "nvidia";
using Error = CUresult;
using DevicePointer = CUdeviceptr;
using ModuleHandle = CUmodule;
using FunctionHandle = CUfunction;
using StreamHandle = CUstream;
constexpr Error kSuccess = CUDA_SUCCESS;
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
constexpr Error kSuccess = hipSuccess;
std::string error_string(Error error) { return hipGetErrorString(error); }
#endif

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
  int device = 0;
};

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      std::puts(
          "--payload PATH --symbol therock_module_saxpy|therock_module_relu "
          "--device INDEX --expect-arch gfx1201|sm_120");
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
    } else if (arg == "--expect-arch") {
      options.expected_arch = value;
    } else if (arg == "--device") {
      size_t consumed = 0;
      options.device = std::stoi(value, &consumed);
      if (consumed != value.size() || options.device < 0) {
        throw std::runtime_error("--device must be a nonnegative integer");
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

class Device {
public:
  explicit Device(const Options &options) {
    int count = 0;
    std::string architecture;
    std::string name;
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
    int major = 0, minor = 0;
    GPU_CHECK(cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_));
    GPU_CHECK(cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_));
    char device_name[256]{};
    GPU_CHECK(cuDeviceGetName(device_name, sizeof(device_name), device_));
    name = device_name;
    architecture = "sm_" + std::to_string(major) + std::to_string(minor);
#else
    GPU_CHECK(hipSetDevice(options.device));
    hipDeviceProp_t props{};
    GPU_CHECK(hipGetDeviceProperties(&props, options.device));
    name = props.name;
    architecture = props.gcnArchName;
    architecture = architecture.substr(0, architecture.find(':'));
    if (architecture.rfind("gfx", 0) != 0) {
      throw std::runtime_error(
          "AMD native runtime did not report a gfx device");
    }
#endif
    std::printf("DEVICE backend=%s index=%d count=%d name=\"%s\" arch=%s\n",
                kBackend, options.device, count, name.c_str(),
                architecture.c_str());
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
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;

private:
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
#if defined(THEROCK_MODULE_NVIDIA)
    check_cleanup(cuStreamSynchronize(value), "cuStreamSynchronize");
    check_cleanup(cuStreamDestroy(value), "cuStreamDestroy");
#else
    check_cleanup(hipStreamSynchronize(value), "hipStreamSynchronize");
    check_cleanup(hipStreamDestroy(value), "hipStreamDestroy");
#endif
  }
  void synchronize() {
#if defined(THEROCK_MODULE_NVIDIA)
    GPU_CHECK(cuStreamSynchronize(value));
#else
    GPU_CHECK(hipStreamSynchronize(value));
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

void run(const Options &options) {
  const std::string format = inspect_payload(options.payload);
  Device device(options);
  Module module(options.payload);
  const FunctionHandle function = module.get_function(options.symbol);
  std::printf("MODULE backend=%s format=%s symbol=%s payload=\"%s\"\n",
              kBackend, format.c_str(), options.symbol.c_str(),
              options.payload.c_str());
  constexpr unsigned sizes[] = {1, 127, 128, 129, 4099, 65539};
  constexpr unsigned guard_count = 17;
  constexpr float guard_value = -12345.5f;
  constexpr unsigned block_size = 128;
  for (unsigned count : sizes) {
    Buffer x(count * sizeof(float)), y(count * sizeof(float));
    Buffer output((count + guard_count) * sizeof(float));
    HostBuffer host_x(count), host_y(count), actual(count + guard_count);
    std::vector<float> expected(count);
    // Stream destruction synchronizes before the referenced allocations are
    // released, including when an API call throws on a failure path.
    Stream stream;
    for (int round = 0; round < 3; ++round) {
      float alpha = 1.75f;
      for (unsigned i = 0; i < count; ++i) {
        host_x[i] = static_cast<float>(
                        static_cast<int>((i * 17 + round * 13) % 97) - 48) /
                    16.0f;
        host_y[i] = static_cast<float>(
                        static_cast<int>((i * 11 + round * 7) % 89) - 44) /
                    32.0f;
        expected[i] = alpha * host_x[i] + host_y[i];
        if (options.symbol == "therock_module_relu") {
          expected[i] = std::max(0.0f, expected[i] - 0.125f);
        }
      }
      std::fill(actual.begin(), actual.end(), guard_value);
      x.from_host(host_x.data(), count, stream.value);
      y.from_host(host_y.data(), count, stream.value);
      output.from_host(actual.data(), actual.size(), stream.value);
      void *arguments[] = {&x.value, &y.value, &output.value, &alpha, &count};
      const unsigned grid_size = (count + block_size - 1) / block_size;
#if defined(THEROCK_MODULE_NVIDIA)
      GPU_CHECK(cuLaunchKernel(function, grid_size, 1, 1, block_size, 1, 1, 0,
                               stream.value, arguments, nullptr));
#else
      GPU_CHECK(hipModuleLaunchKernel(function, grid_size, 1, 1, block_size, 1,
                                      1, 0, stream.value, arguments, nullptr));
#endif
      output.to_host(actual.data(), actual.size(), stream.value);
      stream.synchronize();
      double max_error = 0.0;
      for (unsigned i = 0; i < count; ++i) {
        const double error =
            std::abs(static_cast<double>(actual[i]) - expected[i]);
        // A few FP32 rounding steps allow optional FMA contraction. The inputs
        // are bounded dyadic fractions, so failures cannot hide in loose error.
        const double tolerance = 1.0e-6 + 2.0e-6 * std::abs(expected[i]);
        if (!std::isfinite(actual[i]) || error > tolerance) {
          throw std::runtime_error(
              "Arithmetic mismatch index=" + std::to_string(i) +
              " expected=" + std::to_string(expected[i]) +
              " actual=" + std::to_string(actual[i]));
        }
        max_error = std::max(max_error, error);
      }
      for (unsigned i = count; i < actual.size(); ++i) {
        if (actual[i] != guard_value) {
          throw std::runtime_error("Output guard overwritten index=" +
                                   std::to_string(i));
        }
      }
      std::printf("CHECK n=%u round=%d max_abs_error=%.9g guard=pass\n", count,
                  round, max_error);
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
      throw std::runtime_error("Native API resource cleanup failed");
    }
    std::printf("PASS backend=%s symbol=%s\n", kBackend,
                options.symbol.c_str());
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL backend=%s error=%s\n", kBackend, error.what());
    return 1;
  }
}
