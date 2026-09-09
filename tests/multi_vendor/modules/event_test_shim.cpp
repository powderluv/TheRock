// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CPU scheduler for the native event pipeline tests. No function forwards to a
// GPU runtime. Build separately against each SDK to check its real C
// signatures.
#if defined(THEROCK_EVENT_TEST_CUDA)
#include <cuda.h>
using Result = CUresult;
using QueueHandle = CUstream;
using EventHandle = CUevent;
constexpr Result success = CUDA_SUCCESS;
constexpr Result failure = CUDA_ERROR_UNKNOWN;
constexpr Result pending = CUDA_ERROR_NOT_READY;
#else
#include <hip/hip_runtime_api.h>
using Result = hipError_t;
using QueueHandle = hipStream_t;
using EventHandle = hipEvent_t;
constexpr Result success = hipSuccess;
constexpr Result failure = hipErrorUnknown;
constexpr Result pending = hipErrorNotReady;
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {
struct Task {
  std::vector<Task *> dependencies;
  std::function<void()> action;
  std::string kind;
  size_t queue;
  bool done = false;
};
struct Queue {
  size_t id;
  Task *tail = nullptr;
};
struct Event {
  Task *task = nullptr;
  bool queried = false;
};
struct Module;
struct Function {
  Module *module;
  bool relu;
};
struct Module {
  size_t id;
  bool loaded = true;
  std::vector<std::unique_ptr<Function>> functions;
};
std::vector<std::unique_ptr<Module>> modules;
std::vector<std::unique_ptr<Task>> tasks;
size_t module_load_attempts = 0;
size_t kernel_count = 0;
size_t next_queue = 0;
std::vector<void *> device_allocations;
size_t round_uploads = 0, round_launches = 0, round_readbacks = 0;
size_t pipeline_compute_queue = 0;
float *pipeline_x = nullptr, *pipeline_y = nullptr;
float *scratch_a = nullptr, *scratch_b = nullptr, *previous_output = nullptr;
unsigned pipeline_count = 0;
std::set<const void *> pipeline_readback_sources;
size_t expected_module_count();
bool pipeline_mode() {
  const char *value = std::getenv("THEROCK_EVENT_TEST_PIPELINE");
  return value && std::strcmp(value, "1") == 0;
}
bool injected = false;
const char *mode() {
  const char *value = std::getenv("THEROCK_EVENT_TEST_FAILURE");
  return value ? value : "none";
}
void trace(const char *operation) {
  std::fprintf(stderr, "CPU_EVENT_SHIM %s\n", operation);
}
[[noreturn]] void violation(const char *message) {
  std::fprintf(stderr, "CPU_EVENT_SHIM VIOLATION %s\n", message);
  std::fflush(stderr);
  std::_Exit(93);
}
bool inject(const char *operation) {
  if (!injected && std::strcmp(mode(), operation) == 0) {
    injected = true;
    std::fprintf(stderr, "CPU_EVENT_SHIM INJECT %s\n", operation);
    return true;
  }
  return false;
}
bool work_pending() {
  return std::any_of(tasks.begin(), tasks.end(),
                     [](const auto &task) { return !task->done; });
}
void require_complete() {
  if (work_pending())
    violation("resource released while submitted work is pending");
}
void complete(Task *task) {
  if (!task || task->done)
    return;
  for (Task *dependency : task->dependencies)
    complete(dependency);
  task->action();
  task->done = true;
}
Task *submit(Queue *queue, const char *kind, std::function<void()> action,
             Task *dependency = nullptr) {
  auto task = std::make_unique<Task>();
  task->kind = kind;
  task->queue = queue->id;
  task->action = std::move(action);
  if (queue->tail)
    task->dependencies.push_back(queue->tail);
  if (dependency)
    task->dependencies.push_back(dependency);
  queue->tail = task.get();
  tasks.push_back(std::move(task));
  return queue->tail;
}
Queue *queue(QueueHandle value) { return reinterpret_cast<Queue *>(value); }
Event *event(EventHandle value) { return reinterpret_cast<Event *>(value); }
Result create_queue(QueueHandle *out) {
  *out = reinterpret_cast<QueueHandle>(new Queue{next_queue++});
  trace("queue-create");
  return success;
}
Result query_queue(QueueHandle value) {
  trace("queue-query");
  if (work_pending() && !injected)
    violation("host waited on a queue before final readback event completion");
  if (injected && std::strcmp(mode(), "drain") == 0 && queue(value)->tail &&
      !queue(value)->tail->done)
    return failure;
  complete(queue(value)->tail);
  return success;
}
Result destroy_queue(QueueHandle value) {
  if (queue(value)->tail && !queue(value)->tail->done)
    violation("pending queue destroyed");
  delete queue(value);
  trace("queue-destroy");
  return success;
}
Result create_event(EventHandle *out) {
  *out = reinterpret_cast<EventHandle>(new Event);
  trace("event-create");
  return success;
}
Result record_event(EventHandle value, QueueHandle stream) {
  if (inject("record"))
    return failure;
  event(value)->task = queue(stream)->tail;
  event(value)->queried = false;
  if (!event(value)->task)
    violation("empty event recorded");
  trace("event-record");
  return success;
}
Result wait_event(QueueHandle stream, EventHandle value) {
  if (!event(value)->task)
    violation("wait for unrecorded event");
  if (event(value)->task->queue == queue(stream)->id)
    violation("event dependency did not cross queues");
  submit(queue(stream), "wait", [] {}, event(value)->task);
  trace("queue-wait-event");
  return success;
}
void collect(Task *task, std::set<Task *> &seen) {
  if (!task || !seen.insert(task).second)
    return;
  for (Task *dependency : task->dependencies)
    collect(dependency, seen);
}
Result query_event(EventHandle value) {
  trace("event-query");
  if (inject("query"))
    return failure;
  Event *selected = event(value);
  if (!selected->task || selected->task->kind != "readback")
    violation("host waited before submitting final readback");
  if (!selected->queried) {
    selected->queried = true;
    return pending;
  }
  // Require one connected graph with upload, kernel, and readback on distinct
  // queues. An absent wait edge leaves its predecessor outside this closure.
  std::set<Task *> closure;
  collect(selected->task, closure);
  std::set<size_t> uploads, kernels, readbacks;
  size_t upload_count = 0, waits = 0, kernel_tasks = 0, readback_count = 0;
  for (Task *task : closure) {
    if (task->done)
      continue;
    if (task->kind == "upload") {
      uploads.insert(task->queue);
      ++upload_count;
    } else if (task->kind == "kernel") {
      kernels.insert(task->queue);
      ++kernel_tasks;
    } else if (task->kind == "readback") {
      readbacks.insert(task->queue);
      ++readback_count;
    } else if (task->kind == "wait") {
      ++waits;
    }
  }
  if (uploads.size() != 1 || kernels.size() != 1 || readbacks.size() != 1 ||
      upload_count != (pipeline_mode() ? 4 : 3) ||
      kernel_tasks != (pipeline_mode() ? expected_module_count() : 1) ||
      readback_count != (pipeline_mode() ? 2 : 1) || waits != 2 ||
      uploads == kernels || uploads == readbacks || kernels == readbacks)
    violation("final event lacks the complete three-queue dependency graph");
  complete(selected->task);
  require_complete();
  trace("graph-complete");
  round_uploads = round_launches = round_readbacks = 0;
  pipeline_readback_sources.clear();
  return success;
}
Result destroy_event(EventHandle value) {
  require_complete();
  delete event(value);
  trace("event-destroy");
  return success;
}
size_t expected_module_count() {
  const char *value = std::getenv("THEROCK_EVENT_TEST_MODULES");
  return value ? std::strtoul(value, nullptr, 10) : 1;
}
void require_all_modules() {
  if (modules.size() != expected_module_count() ||
      std::any_of(modules.begin(), modules.end(), [](const auto &module) {
        return !module->loaded || module->functions.empty();
      }))
    violation("not all requested modules/functions are loaded and retained");
}
Module *load_module() {
  ++module_load_attempts;
  if (kernel_count)
    violation("module reloaded after session execution began");
  if (module_load_attempts == 2 && inject("second-module"))
    return nullptr;
  auto value = std::make_unique<Module>();
  value->id = modules.size();
  Module *handle = value.get();
  modules.push_back(std::move(value));
  std::fprintf(stderr, "CPU_EVENT_SHIM module-load id=%zu\n", handle->id);
  return handle;
}
Result unload_module(Module *value) {
  require_complete();
  if (!value || !value->loaded)
    violation("unloading a missing or already unloaded module");
  value->loaded = false;
  trace("module-unload");
  return success;
}
Function *get_function(Module *module, const char *name) {
  if (!module || !module->loaded)
    violation("looking up a function in an unloaded module");
  if (std::strcmp(name, "therock_module_saxpy") != 0 &&
      std::strcmp(name, "therock_module_relu") != 0)
    violation("unknown fixture symbol reached module lookup");
  auto value = std::make_unique<Function>();
  value->module = module;
  value->relu = std::strcmp(name, "therock_module_relu") == 0;
  Function *handle = value.get();
  module->functions.push_back(std::move(value));
  std::fprintf(stderr, "CPU_EVENT_SHIM function-load module=%zu symbol=%s\n",
               module->id, name);
  return handle;
}
Result allocate(void **out, size_t bytes, bool device) {
  require_all_modules();
  *out = std::malloc(bytes);
  if (!*out)
    return failure;
  std::memset(*out, 0x5a, bytes);
  if (device)
    device_allocations.push_back(*out);
  std::fprintf(stderr, "CPU_EVENT_SHIM %s-allocate bytes=%zu\n",
               device ? "device" : "host", bytes);
  return success;
}
Result release(void *value) {
  require_complete();
  std::free(value);
  trace("free");
  return success;
}
Result copy(void *to, const void *from, size_t bytes, QueueHandle stream,
            bool upload) {
  if (pipeline_mode()) {
    if (upload) {
      if (round_launches || round_readbacks || ++round_uploads > 4)
        violation("pipeline upload occurred after computation or exceeded four "
                  "copies");
    } else {
      if (round_launches != expected_module_count())
        violation(
            "intermediate readback before all pipeline stages were submitted");
      if ((from != scratch_a && from != scratch_b) ||
          !pipeline_readback_sources.insert(from).second ||
          ++round_readbacks > 2)
        violation("pipeline must read back both distinct scratch buffers");
      if (bytes < (pipeline_count + 17) * sizeof(float))
        violation("pipeline readback omitted its output guards");
    }
  }
  submit(queue(stream), upload ? "upload" : "readback",
         [=] { std::memcpy(to, from, bytes); });
  trace(upload ? "upload" : "readback");
  return success;
}
Result launch(Function *function, QueueHandle stream, void **args) {
  require_all_modules();
  if (!function || !function->module->loaded)
    violation("launch references an unloaded module");
  if (inject("launch") || inject("drain") ||
      (kernel_count == 1 && inject("later-launch")))
    return failure;
  float *x = nullptr, *y = nullptr, *out = nullptr;
  std::memcpy(&x, args[0], sizeof(x));
  std::memcpy(&y, args[1], sizeof(y));
  std::memcpy(&out, args[2], sizeof(out));
  const float alpha = *reinterpret_cast<float *>(args[3]);
  const unsigned count = *reinterpret_cast<unsigned *>(args[4]);
  if (pipeline_mode()) {
    if (round_uploads != 4 || round_readbacks ||
        round_launches >= expected_module_count())
      violation(
          "pipeline stage submitted outside one upload-compute-readback graph");
    if (device_allocations.size() != 4)
      violation("pipeline must reuse exactly four device allocations");
    if (round_launches == 0) {
      if (!pipeline_x) {
        if (std::find(device_allocations.begin(), device_allocations.end(),
                      x) == device_allocations.end() ||
            std::find(device_allocations.begin(), device_allocations.end(),
                      y) == device_allocations.end() ||
            std::find(device_allocations.begin(), device_allocations.end(),
                      out) == device_allocations.end())
          violation(
              "pipeline arguments do not reference allocated device storage");
        if (x == y || x == out || y == out)
          violation("pipeline inputs and scratch storage alias");
        pipeline_x = x;
        pipeline_y = y;
        scratch_a = out;
        for (void *allocation : device_allocations)
          if (allocation != x && allocation != y && allocation != out)
            scratch_b = static_cast<float *>(allocation);
      }
      pipeline_count = count;
      pipeline_compute_queue = queue(stream)->id;
      if (x != pipeline_x || y != pipeline_y || out != scratch_a)
        violation(
            "pipeline first stage did not reuse input and scratch allocations");
    } else if (x != previous_output || y != pipeline_y ||
               out != (round_launches % 2 ? scratch_b : scratch_a) ||
               count != pipeline_count ||
               queue(stream)->id != pipeline_compute_queue) {
      violation("pipeline stage did not consume the previous scratch on the "
                "compute queue");
    }
    previous_output = out;
    ++round_launches;
    trace("pipeline-bindings-pass");
  }
  const bool relu = function->relu;
  ++kernel_count;
  std::fprintf(stderr,
               "CPU_EVENT_SHIM kernel-submit module=%zu symbol=%s count=%u\n",
               function->module->id, relu ? "relu" : "saxpy", count);
  submit(queue(stream), "kernel", [=] {
    if (!function->module->loaded)
      violation("module released before queued kernel completed");
    for (unsigned i = 0; i < count; ++i) {
      float value = alpha * x[i] + y[i];
      out[i] = relu ? std::max(0.0f, value - 0.125f) : value;
    }
  });
  trace("kernel");
  return success;
}
void uuid(void *out) { std::memset(out, 0x12, 16); }
constexpr size_t memory_bytes = size_t{16} << 30;
} // namespace

extern "C" {
#if defined(THEROCK_EVENT_TEST_CUDA)
CUresult cuInit(unsigned) {
  trace("device-init");
  return success;
}
CUresult cuDeviceGetCount(int *count) {
  *count = 1;
  return success;
}
CUresult cuDeviceGet(CUdevice *out, int) {
  *out = 0;
  return success;
}
CUresult cuDeviceGetUuid(CUuuid *out, CUdevice) {
  uuid(out);
  return success;
}
CUresult cuDeviceGetName(char *out, int size, CUdevice) {
  std::snprintf(out, size, "CPU event pipeline fixture");
  return success;
}
CUresult cuDeviceGetAttribute(int *out, CUdevice_attribute attribute,
                              CUdevice) {
  if (attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
    *out = 12;
  else if (attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
    *out = 0;
  else if (attribute == CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    *out = 65536;
  else
    *out = 1024;
  return success;
}
CUresult cuDeviceTotalMem(size_t *out, CUdevice) {
  *out = memory_bytes;
  return success;
}
CUresult cuDevicePrimaryCtxRetain(CUcontext *out, CUdevice) {
  trace("context-create");
  *out = reinterpret_cast<CUcontext>(1);
  return success;
}
CUresult cuDevicePrimaryCtxRelease(CUdevice) {
  require_complete();
  trace("context-release");
  return success;
}
CUresult cuCtxSetCurrent(CUcontext) { return success; }
CUresult cuGetErrorName(CUresult, const char **out) {
  *out = "CPU injected error";
  return success;
}
CUresult cuGetErrorString(CUresult, const char **out) {
  *out = "CPU event scheduler";
  return success;
}
CUresult cuModuleLoad(CUmodule *out, const char *) {
  *out = reinterpret_cast<CUmodule>(load_module());
  return *out ? success : failure;
}
CUresult cuModuleUnload(CUmodule value) {
  return unload_module(reinterpret_cast<Module *>(value));
}
CUresult cuModuleGetFunction(CUfunction *out, CUmodule module,
                             const char *name) {
  *out = reinterpret_cast<CUfunction>(
      get_function(reinterpret_cast<Module *>(module), name));
  return success;
}
CUresult cuFuncGetAttribute(int *out, CUfunction_attribute attribute,
                            CUfunction) {
  *out = attribute == CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK ? 1024 : 0;
  return success;
}
CUresult cuMemGetInfo(size_t *free, size_t *total) {
  *free = *total = memory_bytes;
  return success;
}
CUresult cuMemAlloc(CUdeviceptr *out, size_t size) {
  void *allocation = nullptr;
  const Result result = allocate(&allocation, size, true);
  *out = reinterpret_cast<CUdeviceptr>(allocation);
  return result;
}
CUresult cuMemAllocHost(void **out, size_t size) {
  return allocate(out, size, false);
}
CUresult cuMemFree(CUdeviceptr value) {
  return release(reinterpret_cast<void *>(value));
}
CUresult cuMemFreeHost(void *value) { return release(value); }
CUresult cuStreamCreate(CUstream *out, unsigned) { return create_queue(out); }
CUresult cuStreamQuery(CUstream value) { return query_queue(value); }
CUresult cuStreamSynchronize(CUstream) {
  violation("unexpected blocking stream synchronization");
}
CUresult cuStreamDestroy(CUstream value) { return destroy_queue(value); }
CUresult cuEventCreate(CUevent *out, unsigned) { return create_event(out); }
CUresult cuEventRecord(CUevent value, CUstream stream) {
  return record_event(value, stream);
}
CUresult cuEventQuery(CUevent value) { return query_event(value); }
CUresult cuEventDestroy(CUevent value) { return destroy_event(value); }
CUresult cuStreamWaitEvent(CUstream stream, CUevent value, unsigned) {
  return wait_event(stream, value);
}
CUresult cuMemcpyHtoDAsync(CUdeviceptr to, const void *from, size_t size,
                           CUstream stream) {
  return copy(reinterpret_cast<void *>(to), from, size, stream, true);
}
CUresult cuMemcpyDtoHAsync(void *to, CUdeviceptr from, size_t size,
                           CUstream stream) {
  return copy(to, reinterpret_cast<void *>(from), size, stream, false);
}
CUresult cuLaunchKernel(CUfunction function, unsigned, unsigned, unsigned,
                        unsigned, unsigned, unsigned, unsigned, CUstream stream,
                        void **args, void **) {
  return launch(reinterpret_cast<Function *>(function), stream, args);
}
#else
hipError_t hipInit(unsigned) { return success; }
hipError_t hipGetDeviceCount(int *out) {
  trace("device-init");
  *out = 1;
  return success;
}
hipError_t hipDeviceGetUuid(hipUUID *out, hipDevice_t) {
  uuid(out);
  return success;
}
hipError_t hipSetDevice(int) {
  trace("device-select");
  return success;
}
hipError_t hipGetDeviceProperties(hipDeviceProp_t *out, int) {
  std::memset(out, 0, sizeof(*out));
  std::strcpy(out->name, "CPU event pipeline fixture");
  std::strcpy(out->gcnArchName, "gfx1201");
  out->maxThreadsPerBlock = 1024;
  std::fill_n(out->maxThreadsDim, 3, 1024);
  std::fill_n(out->maxGridSize, 3, 1024);
  out->totalGlobalMem = memory_bytes;
  out->sharedMemPerBlock = 65536;
  return success;
}
const char *hipGetErrorString(hipError_t) {
  return "CPU injected event scheduler error";
}
hipError_t hipModuleLoad(hipModule_t *out, const char *) {
  *out = reinterpret_cast<hipModule_t>(load_module());
  return *out ? success : failure;
}
hipError_t hipModuleUnload(hipModule_t value) {
  return unload_module(reinterpret_cast<Module *>(value));
}
hipError_t hipModuleGetFunction(hipFunction_t *out, hipModule_t module,
                                const char *name) {
  *out = reinterpret_cast<hipFunction_t>(
      get_function(reinterpret_cast<Module *>(module), name));
  return success;
}
hipError_t hipFuncGetAttribute(int *out, hipFunction_attribute attribute,
                               hipFunction_t) {
  *out = attribute == HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK ? 1024 : 0;
  return success;
}
hipError_t hipMemGetInfo(size_t *free, size_t *total) {
  *free = *total = memory_bytes;
  return success;
}
hipError_t hipMalloc(void **out, size_t size) {
  return allocate(out, size, true);
}
hipError_t hipHostMalloc(void **out, size_t size, unsigned) {
  return allocate(out, size, false);
}
hipError_t hipFree(void *value) { return release(value); }
hipError_t hipHostFree(void *value) { return release(value); }
hipError_t hipStreamCreateWithFlags(hipStream_t *out, unsigned) {
  return create_queue(out);
}
hipError_t hipStreamQuery(hipStream_t value) { return query_queue(value); }
hipError_t hipStreamSynchronize(hipStream_t) {
  violation("unexpected blocking stream synchronization");
}
hipError_t hipStreamDestroy(hipStream_t value) { return destroy_queue(value); }
hipError_t hipEventCreateWithFlags(hipEvent_t *out, unsigned) {
  return create_event(out);
}
hipError_t hipEventRecord(hipEvent_t value, hipStream_t stream) {
  return record_event(value, stream);
}
hipError_t hipEventQuery(hipEvent_t value) { return query_event(value); }
hipError_t hipEventDestroy(hipEvent_t value) { return destroy_event(value); }
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t value, unsigned) {
  return wait_event(stream, value);
}
hipError_t hipMemcpyAsync(void *to, const void *from, size_t size,
                          hipMemcpyKind kind, hipStream_t stream) {
  if (kind != hipMemcpyHostToDevice && kind != hipMemcpyDeviceToHost)
    violation("unexpected memcpy direction");
  return copy(to, from, size, stream, kind == hipMemcpyHostToDevice);
}
hipError_t hipModuleLaunchKernel(hipFunction_t function, unsigned, unsigned,
                                 unsigned, unsigned, unsigned, unsigned,
                                 unsigned, hipStream_t stream, void **args,
                                 void **) {
  return launch(reinterpret_cast<Function *>(function), stream, args);
}
#endif
}
