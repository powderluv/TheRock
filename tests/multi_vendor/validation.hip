// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr unsigned kBlockSize = 128;
constexpr size_t kGuardCount = 17;
constexpr float kGuardValue = -12345.5f;
constexpr size_t kSizes[] = {1, 127, 128, 129, 4099, 65539};
bool cleanup_failed = false;

void check(hipError_t error, const char *operation, int line) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + " at line " +
                             std::to_string(line) + ": " +
                             hipGetErrorString(error));
  }
}

#define HIP_CHECK(operation) check((operation), #operation, __LINE__)

void check_cleanup(hipError_t error, const char *operation) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "FAIL cleanup=%s error=%s\n", operation,
                 hipGetErrorString(error));
    cleanup_failed = true;
  }
}

class Buffer {
public:
  explicit Buffer(size_t count, bool pinned_host = false)
      : pinned_host_(pinned_host) {
    if (pinned_host_) {
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&data),
                              count * sizeof(float)));
    } else {
      HIP_CHECK(
          hipMalloc(reinterpret_cast<void **>(&data), count * sizeof(float)));
    }
  }
  ~Buffer() {
    check_cleanup(pinned_host_ ? hipHostFree(data) : hipFree(data),
                  pinned_host_ ? "hipHostFree" : "hipFree");
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  float *data = nullptr;

private:
  bool pinned_host_;
};

class Stream {
public:
  Stream() {
    HIP_CHECK(hipStreamCreateWithFlags(&value, hipStreamNonBlocking));
  }
  ~Stream() { check_cleanup(hipStreamDestroy(value), "hipStreamDestroy"); }
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
  hipStream_t value = nullptr;
};

class Event {
public:
  Event() { HIP_CHECK(hipEventCreateWithFlags(&value, hipEventDisableTiming)); }
  ~Event() { check_cleanup(hipEventDestroy(value), "hipEventDestroy"); }
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
  hipEvent_t value = nullptr;
};

__global__ void saxpy(const float *x, const float *y, float *out, float alpha,
                      size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    out[i] = alpha * x[i] + y[i];
  }
}

__global__ void activation(float *values, size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float value = values[i] - 0.125f;
    values[i] = value > 0.0f ? value : 0.0f;
  }
}

__global__ void reduce_blocks(const float *input, float *output, size_t count) {
  __shared__ float scratch[kBlockSize];
  const unsigned lane = threadIdx.x;
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + lane;
  scratch[lane] = i < count ? input[i] : 0.0f;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (lane < stride) {
      scratch[lane] += scratch[lane + stride];
    }
    // All threads participate, including the padded tail of the final block.
    __syncthreads();
  }
  if (lane == 0) {
    output[blockIdx.x] = scratch[0];
  }
}

unsigned blocks_for(size_t count) {
  return static_cast<unsigned>((count + kBlockSize - 1) / kBlockSize);
}

void fill_inputs(float *x, float *y, size_t count, int seed) {
  for (size_t i = 0; i < count; ++i) {
    x[i] =
        static_cast<float>(static_cast<int>((i * 17 + seed * 13) % 97) - 48) /
        16.0f;
    y[i] = static_cast<float>(static_cast<int>((i * 11 + seed * 7) % 89) - 44) /
           32.0f;
  }
}

void reference_step(const float *x, float *y, size_t count, float alpha) {
  for (size_t i = 0; i < count; ++i) {
    y[i] = std::max(0.0f, alpha * x[i] + y[i] - 0.125f);
  }
}

double compare(const float *actual, const float *expected, size_t count,
               const std::string &context) {
  double max_error = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double error = std::abs(static_cast<double>(actual[i]) - expected[i]);
    // FP32 SAXPY may contract to FMA. These tolerances allow a few rounding
    // steps while still rejecting omitted kernels, stale copies, or bad tails.
    const double tolerance = 1.0e-6 + 2.0e-6 * std::abs(expected[i]);
    if (!std::isfinite(actual[i]) || error > tolerance) {
      throw std::runtime_error(context + " index=" + std::to_string(i) +
                               " expected=" + std::to_string(expected[i]) +
                               " actual=" + std::to_string(actual[i]));
    }
    max_error = std::max(max_error, error);
  }
  return max_error;
}

void run_core() {
  for (size_t count : kSizes) {
    for (int round = 0; round < 3; ++round) {
      const size_t bytes = count * sizeof(float);
      std::vector<float> x(count), y(count), expected(count);
      std::vector<float> actual(count + kGuardCount, kGuardValue);
      fill_inputs(x.data(), y.data(), count, round);
      expected = y;
      reference_step(x.data(), expected.data(), count, 1.75f);
      Buffer dx(count), dy(count), output(actual.size());
      HIP_CHECK(hipMemcpy(dx.data, x.data(), bytes, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dy.data, y.data(), bytes, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(output.data, actual.data(),
                          actual.size() * sizeof(float),
                          hipMemcpyHostToDevice));
      hipLaunchKernelGGL(saxpy, dim3(blocks_for(count)), dim3(kBlockSize), 0, 0,
                         dx.data, dy.data, output.data, 1.75f, count);
      HIP_CHECK(hipGetLastError());
      hipLaunchKernelGGL(activation, dim3(blocks_for(count)), dim3(kBlockSize),
                         0, 0, output.data, count);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(actual.data(), output.data,
                          actual.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
      const double error =
          compare(actual.data(), expected.data(), count, "core");
      for (size_t i = count; i < actual.size(); ++i) {
        if (actual[i] != kGuardValue) {
          throw std::runtime_error("core output guard overwritten at index=" +
                                   std::to_string(i));
        }
      }
      std::printf(
          "CHECK case=core n=%zu round=%d max_abs_error=%.9g guard=pass\n",
          count, round, error);
    }
  }
}

void run_streams() {
  constexpr size_t count = 65539;
  constexpr size_t bytes = count * sizeof(float);
  constexpr int launches = 11;
  Stream transfer, compute;
  Event inputs_ready, compute_done, readback_done;
  for (int round = 0; round < 3; ++round) {
    Buffer x(count, true), y(count, true), actual(count, true);
    Buffer dx(count), dy(count);
    fill_inputs(x.data, y.data, count, round + 19);
    std::fill_n(actual.data, count, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> expected(y.data, y.data + count);
    for (int launch = 0; launch < launches; ++launch) {
      reference_step(x.data, expected.data(), count, 0.25f);
    }
    HIP_CHECK(hipMemcpyAsync(dx.data, x.data, bytes, hipMemcpyHostToDevice,
                             transfer.value));
    HIP_CHECK(hipMemcpyAsync(dy.data, y.data, bytes, hipMemcpyHostToDevice,
                             transfer.value));
    HIP_CHECK(hipEventRecord(inputs_ready.value, transfer.value));
    HIP_CHECK(hipStreamWaitEvent(compute.value, inputs_ready.value, 0));
    for (int launch = 0; launch < launches; ++launch) {
      hipLaunchKernelGGL(saxpy, dim3(blocks_for(count)), dim3(kBlockSize), 0,
                         compute.value, dx.data, dy.data, dy.data, 0.25f,
                         count);
      HIP_CHECK(hipGetLastError());
      hipLaunchKernelGGL(activation, dim3(blocks_for(count)), dim3(kBlockSize),
                         0, compute.value, dy.data, count);
      HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipEventRecord(compute_done.value, compute.value));
    HIP_CHECK(hipStreamWaitEvent(transfer.value, compute_done.value, 0));
    HIP_CHECK(hipMemcpyAsync(actual.data, dy.data, bytes, hipMemcpyDeviceToHost,
                             transfer.value));
    HIP_CHECK(hipEventRecord(readback_done.value, transfer.value));
    // No device-wide synchronization may supply the tested cross-stream
    // ordering.
    HIP_CHECK(hipEventSynchronize(readback_done.value));
    HIP_CHECK(hipEventQuery(inputs_ready.value));
    HIP_CHECK(hipEventQuery(compute_done.value));
    const double error =
        compare(actual.data, expected.data(), count, "streams");
    HIP_CHECK(hipStreamSynchronize(transfer.value));
    HIP_CHECK(hipStreamSynchronize(compute.value));
    std::printf(
        "CHECK case=streams n=%zu round=%d launches=%d max_abs_error=%.9g\n",
        count, round, launches * 2, error);
    // All asynchronous uses complete before host/device buffers are released.
  }
}

void run_reduction() {
  for (size_t count : kSizes) {
    std::vector<float> input(count), unused(count);
    fill_inputs(input.data(), unused.data(), count, 47);
    // Dyadic inputs make this bounded sum exactly representable in FP32, so
    // reduction bugs cannot hide behind a large error allowance or
    // cancellation.
    double reference_sum = 0.0;
    for (float value : input) {
      reference_sum += value;
    }
    Buffer first(count + kGuardCount), second(count + kGuardCount);
    HIP_CHECK(hipMemcpy(first.data, input.data(), count * sizeof(float),
                        hipMemcpyHostToDevice));
    float *current = first.data;
    float *next = second.data;
    size_t remaining = count;
    unsigned stages = 0;
    do {
      const unsigned blocks = blocks_for(remaining);
      std::vector<float> partials(blocks + kGuardCount, kGuardValue);
      HIP_CHECK(hipMemcpy(next, partials.data(),
                          partials.size() * sizeof(float),
                          hipMemcpyHostToDevice));
      hipLaunchKernelGGL(reduce_blocks, dim3(blocks), dim3(kBlockSize), 0, 0,
                         current, next, remaining);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(partials.data(), next,
                          partials.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
      for (size_t block = 0; block < blocks; ++block) {
        double expected = 0.0;
        const size_t end = std::min((block + 1) * kBlockSize, remaining);
        for (size_t i = block * kBlockSize; i < end; ++i) {
          expected += input[i];
        }
        if (!std::isfinite(partials[block]) || partials[block] != expected) {
          throw std::runtime_error(
              "reduction stage=" + std::to_string(stages) + " block=" +
              std::to_string(block) + " expected=" + std::to_string(expected) +
              " actual=" + std::to_string(partials[block]));
        }
      }
      for (size_t i = blocks; i < partials.size(); ++i) {
        if (partials[i] != kGuardValue) {
          throw std::runtime_error("reduction output guard overwritten");
        }
      }
      input.assign(partials.begin(), partials.begin() + blocks);
      remaining = blocks;
      std::swap(current, next);
      ++stages;
    } while (remaining > 1);
    if (input[0] != reference_sum) {
      throw std::runtime_error("reduction final sum mismatch");
    }
    std::printf("CHECK case=reduction n=%zu stages=%u sum=%.9g reference=%.9g "
                "guard=pass\n",
                count, stages, input[0], reference_sum);
  }
}

std::string lower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

struct Options {
  std::string test_case = "all";
  std::string expected_vendor;
  std::string expected_arch;
  std::string expected_device_name;
  int device = 0;
};

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      std::puts("--case all|core|streams|reduction --device INDEX "
                "--expect-vendor amd|nvidia|intel --expect-arch ISA "
                "--expect-device-name SUBSTRING");
      std::exit(0);
    }
    if (++i == argc) {
      throw std::runtime_error("Missing value for " + arg);
    }
    const std::string value = argv[i];
    if (arg == "--case") {
      options.test_case = value;
    } else if (arg == "--device") {
      size_t consumed = 0;
      options.device = std::stoi(value, &consumed);
      if (consumed != value.size() || options.device < 0) {
        throw std::runtime_error("--device must be a nonnegative integer");
      }
    } else if (arg == "--expect-vendor") {
      options.expected_vendor = lower(value);
    } else if (arg == "--expect-arch") {
      options.expected_arch = lower(value);
    } else if (arg == "--expect-device-name") {
      options.expected_device_name = lower(value);
    } else {
      throw std::runtime_error("Unknown argument " + arg);
    }
  }
  if (options.test_case != "all" && options.test_case != "core" &&
      options.test_case != "streams" && options.test_case != "reduction") {
    throw std::runtime_error("Unknown test case " + options.test_case);
  }
  return options;
}

void select_device(const Options &options) {
#if defined(THEROCK_MULTI_VENDOR_INTEL)
  const char *backend = std::getenv("CHIP_BE");
  const char *device_type = std::getenv("CHIP_DEVICE_TYPE");
  if (!backend || std::string(backend) != "level0" || !device_type ||
      std::string(device_type) != "gpu") {
    throw std::runtime_error("Intel validation requires CHIP_BE=level0 and "
                             "CHIP_DEVICE_TYPE=gpu to exclude CPU fallback");
  }
#endif
  int count = 0;
  HIP_CHECK(hipGetDeviceCount(&count));
  if (options.device >= count) {
    throw std::runtime_error("Requested device " +
                             std::to_string(options.device) +
                             " but HIP found " + std::to_string(count));
  }
  HIP_CHECK(hipSetDevice(options.device));
  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, options.device));
  const std::string name = lower(props.name);
  std::string vendor;
  std::string architecture;
#if defined(THEROCK_MULTI_VENDOR_NVIDIA)
  vendor = "nvidia"; // CUDA runtime enumerates NVIDIA GPUs only.
  architecture =
      "sm_" + std::to_string(props.major) + std::to_string(props.minor);
#elif defined(THEROCK_MULTI_VENDOR_AMD)
  architecture = lower(props.gcnArchName);
  architecture = architecture.substr(0, architecture.find(':'));
  if (architecture.rfind("gfx", 0) != 0) {
    throw std::runtime_error("AMD runtime did not report a gfx device ISA");
  }
  vendor = "amd";
#elif defined(THEROCK_MULTI_VENDOR_INTEL)
  if (name.find("intel") == std::string::npos) {
    throw std::runtime_error("Intel GPU validation received device " +
                             std::string(props.name));
  }
  vendor = "intel";
  // chipStar does not report Intel hardware ISA through gcnArchName. SPIR-V
  // identifies the payload; use expected_device_name to qualify a B70 model.
  architecture = "spirv";
#else
#error "Select exactly one THEROCK_MULTI_VENDOR backend"
#endif
  int runtime_version = 0;
  HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
  std::printf(
      "DEVICE backend=%s vendor=%s index=%d count=%d name=\"%s\" arch=%s "
      "memory_bytes=%zu multiprocessors=%d runtime_version=%d\n",
      THEROCK_MULTI_VENDOR_BACKEND_NAME, vendor.c_str(), options.device, count,
      props.name, architecture.c_str(), props.totalGlobalMem,
      props.multiProcessorCount, runtime_version);
  if (!options.expected_vendor.empty() && options.expected_vendor != vendor) {
    throw std::runtime_error("Expected vendor " + options.expected_vendor +
                             " but got " + vendor);
  }
  if (!options.expected_arch.empty() && options.expected_arch != architecture) {
    throw std::runtime_error("Expected architecture " + options.expected_arch +
                             " but got " + architecture);
  }
  if (!options.expected_device_name.empty() &&
      name.find(options.expected_device_name) == std::string::npos) {
    throw std::runtime_error("Device name does not contain " +
                             options.expected_device_name);
  }
  if (props.maxThreadsPerBlock < static_cast<int>(kBlockSize) ||
      props.sharedMemPerBlock < kBlockSize * sizeof(float) ||
      props.multiProcessorCount <= 0 || props.totalGlobalMem == 0) {
    throw std::runtime_error(
        "Device does not satisfy the GPU validation profile");
  }
}

} // namespace

int main(int argc, char **argv) {
  // Preserve the last successful case/operation in logs if a driver hangs.
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  try {
    const Options options = parse_options(argc, argv);
    select_device(options);
    if (options.test_case == "all" || options.test_case == "core") {
      run_core();
    }
    if (options.test_case == "all" || options.test_case == "streams") {
      run_streams();
    }
    if (options.test_case == "all" || options.test_case == "reduction") {
      run_reduction();
    }
    if (cleanup_failed) {
      throw std::runtime_error("HIP resource cleanup failed");
    }
    std::printf("PASS backend=%s case=%s\n", THEROCK_MULTI_VENDOR_BACKEND_NAME,
                options.test_case.c_str());
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL backend=%s error=%s\n",
                 THEROCK_MULTI_VENDOR_BACKEND_NAME, error.what());
    return 1;
  }
}
