// hsa_aql_dispatch_test.cpp -- standalone HSA AQL kernel dispatch test.
//
// Uses ONLY the stock HSA runtime API to dispatch fill_kernel_raw.co on the
// gfx1201 GPU agent (WindowsGpuAgent / WindowsAqlQueue) and verify it wrote
// out[0..63] == 0xDEADBEEF.
//
// Memory model (per WindowsLiteDriver investigation):
//   - kernarg buffer: HOST pointer from the CPU agent KERNARG region. The GPU
//     agent exposes NO kernarg-flagged region (FB + LDS only, both
//     kernarg=false). WindowsAqlQueue::SubmitKernel memcpy()s the host kernarg
//     bytes into a VRAM staging buffer, qword-translates any registered-VRAM
//     pointers, and dispatches with the staged GPU addr.
//   - output buffer: GPU FB region (the only HSA_REGION_SEGMENT_GLOBAL /
//     COARSE_GRAINED region on the GPU agent). Its CPU pointer is a registered
//     VRAM pointer, so when it appears in a kernarg qword the queue translates
//     it to the FB GPU address. A plain host buffer would NOT be translated and
//     the kernel would fault.
//   - We do NOT hand-fill COV5 hidden args: the real HSA packet processor
//     populates them from grid_size/workgroup_size. We write only the 12
//     explicit kernarg bytes (out ptr @0, val @8) but allocate the full
//     kernarg_segment_size (272) the symbol reports.
//   - completion_signal: SubmitKernel StoreRelease(0). Init to 1, wait for ==0.
//     Dispatch is synchronous inside the doorbell store; signal is already 0
//     when the wait returns.
//
// Output: HSA_AQL_DISPATCH PASS/FAIL bad=N/64

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <utility>
#include <vector>

#include "hsa.h"

#define FILL_VAL   0xDEADBEEFu
#define FILL_N     64u          // out[0..63]
#define BLOCK_X    64u          // threads per workgroup (wave32 x2)
#define CO_GUEST_PATH "Z:\\winfw\\fill_kernel_raw.co"
#define KERNEL_SYMBOL "fill_kernel.kd"   // COV5: hsa_executable_get_symbol_by_name wants the .kd symbol

#define CHECK(expr)                                                          \
  do {                                                                       \
    hsa_status_t _s = (expr);                                                \
    if (_s != HSA_STATUS_SUCCESS) {                                          \
      const char* _m = nullptr;                                              \
      hsa_status_string(_s, &_m);                                            \
      std::printf("FAIL: %s -> %d (%s)\n", #expr, (int)_s,                  \
                  _m ? _m : "?");                                           \
      std::fflush(stdout);                                                   \
      return 2;                                                              \
    }                                                                        \
  } while (0)

// ---- agent discovery -------------------------------------------------------
struct AgentSearch {
  hsa_agent_t gpu{};
  bool have_gpu = false;
  hsa_agent_t cpu{};
  bool have_cpu = false;
};

static hsa_status_t find_agents_cb(hsa_agent_t agent, void* data) {
  auto* s = static_cast<AgentSearch*>(data);
  hsa_device_type_t dt = (hsa_device_type_t)-1;
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dt) != HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  if (dt == HSA_DEVICE_TYPE_GPU && !s->have_gpu) {
    s->gpu = agent;
    s->have_gpu = true;
  } else if (dt == HSA_DEVICE_TYPE_CPU && !s->have_cpu) {
    s->cpu = agent;
    s->have_cpu = true;
  }
  return HSA_STATUS_SUCCESS;
}

// ---- region discovery ------------------------------------------------------
struct RegionSearch {
  hsa_region_t region{};
  bool found = false;
};

// CPU-agent KERNARG region (fine-grained host memory flagged KERNARG).
static hsa_status_t find_kernarg_region_cb(hsa_region_t region, void* data) {
  auto* r = static_cast<RegionSearch*>(data);
  hsa_region_segment_t seg = (hsa_region_segment_t)-1;
  hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &seg);
  if (seg != HSA_REGION_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
  uint32_t flags = 0;
  hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
  if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
    r->region = region;
    r->found = true;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

// GPU-agent FB region: the only global region (COARSE_GRAINED, not KERNARG).
static hsa_status_t find_fb_region_cb(hsa_region_t region, void* data) {
  auto* r = static_cast<RegionSearch*>(data);
  hsa_region_segment_t seg = (hsa_region_segment_t)-1;
  hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &seg);
  if (seg != HSA_REGION_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
  uint32_t flags = 0;
  hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
  // FB is coarse-grained and NOT kernarg-flagged.
  if ((flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) &&
      !(flags & HSA_REGION_GLOBAL_FLAG_KERNARG)) {
    r->region = region;
    r->found = true;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);  // crash-safe unbuffered output

  CHECK(hsa_init());
  std::printf("hsa_init OK\n");

  // 1) agents
  AgentSearch as;
  CHECK(hsa_iterate_agents(find_agents_cb, &as));
  if (!as.have_gpu) {
    std::printf("FAIL: no GPU agent\n");
    std::fflush(stdout);
    return 2;
  }
  if (!as.have_cpu) {
    std::printf("FAIL: no CPU agent (need it for the KERNARG region)\n");
    std::fflush(stdout);
    return 2;
  }
  std::printf("agents: gpu=0x%llx cpu=0x%llx\n",
              (unsigned long long)as.gpu.handle,
              (unsigned long long)as.cpu.handle);

  // 2) regions
  RegionSearch kernarg_rs;
  hsa_agent_iterate_regions(as.cpu, find_kernarg_region_cb, &kernarg_rs);  // INFO_BREAK on found is OK; checked below
  if (!kernarg_rs.found) {
    std::printf("FAIL: no KERNARG region on CPU agent\n");
    std::fflush(stdout);
    return 2;
  }
  RegionSearch fb_rs;
  hsa_agent_iterate_regions(as.gpu, find_fb_region_cb, &fb_rs);  // INFO_BREAK on found is OK; checked below
  if (!fb_rs.found) {
    std::printf("FAIL: no global/coarse FB region on GPU agent\n");
    std::fflush(stdout);
    return 2;
  }
  std::printf("regions: kernarg=0x%llx(cpu) fb=0x%llx(gpu)\n",
              (unsigned long long)kernarg_rs.region.handle,
              (unsigned long long)fb_rs.region.handle);

  // 3) read the code object from the guest path
  FILE* f = std::fopen(CO_GUEST_PATH, "rb");
  if (!f) {
    std::printf("FAIL: cannot open %s\n", CO_GUEST_PATH);
    std::fflush(stdout);
    return 2;
  }
  std::fseek(f, 0, SEEK_END);
  long co_size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (co_size <= 0) {
    std::printf("FAIL: empty code object\n");
    std::fclose(f);
    std::fflush(stdout);
    return 2;
  }
  std::vector<uint8_t> co_bytes((size_t)co_size);
  size_t rd = std::fread(co_bytes.data(), 1, (size_t)co_size, f);
  std::fclose(f);
  if (rd != (size_t)co_size) {
    std::printf("FAIL: short read %zu/%ld\n", rd, co_size);
    std::fflush(stdout);
    return 2;
  }
  std::printf("co loaded: %ld bytes\n", co_size);

  // 4) load -> freeze -> resolve symbol
  hsa_code_object_reader_t cor{};
  CHECK(hsa_code_object_reader_create_from_memory(co_bytes.data(),
                                                  (size_t)co_size, &cor));
  hsa_executable_t exe{};
  CHECK(hsa_executable_create_alt(HSA_PROFILE_FULL,
                                  HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                  nullptr, &exe));
  CHECK(hsa_executable_load_agent_code_object(exe, as.gpu, cor, nullptr,
                                              nullptr));
  CHECK(hsa_executable_freeze(exe, nullptr));

  std::fprintf(stderr, "STEP: freeze ok; iterating agent symbols\n"); std::fflush(stderr);
  // Resolve the kernel by iterating for the HSA_SYMBOL_KIND_KERNEL symbol
  // (robust vs the fill_kernel / fill_kernel.kd name ambiguity).
  hsa_executable_symbol_t sym{};
  bool sym_found = false;
  std::pair<hsa_executable_symbol_t*, bool*> sf{&sym, &sym_found};
  hsa_executable_iterate_agent_symbols(
      exe, as.gpu,
      [](hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t s,
         void* d) -> hsa_status_t {
        auto* p = static_cast<std::pair<hsa_executable_symbol_t*, bool*>*>(d);
        hsa_symbol_kind_t kind = (hsa_symbol_kind_t)-1;
        if (hsa_executable_symbol_get_info(s, HSA_EXECUTABLE_SYMBOL_INFO_TYPE,
                                           &kind) == HSA_STATUS_SUCCESS &&
            kind == HSA_SYMBOL_KIND_KERNEL) {
          *p->first = s;
          *p->second = true;
          return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
      },
      &sf);
  if (!sym_found) {
    std::printf("FAIL: no HSA_SYMBOL_KIND_KERNEL symbol in executable\n");
    std::fflush(stdout);
    return 2;
  }
  std::fprintf(stderr, "STEP: kernel symbol resolved\n"); std::fflush(stderr);

  uint64_t kernel_object = 0;
  uint32_t kernarg_size = 0;
  uint32_t group_size = 0;
  uint32_t private_size = 0;
  CHECK(hsa_executable_symbol_get_info(
      sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object));
  CHECK(hsa_executable_symbol_get_info(
      sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
      &kernarg_size));
  CHECK(hsa_executable_symbol_get_info(
      sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_size));
  CHECK(hsa_executable_symbol_get_info(
      sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
      &private_size));
  std::printf("symbol: ko=0x%llx kernarg_size=%u group=%u private=%u\n",
              (unsigned long long)kernel_object, kernarg_size, group_size,
              private_size);
  if (kernel_object == 0) {
    std::printf("FAIL: kernel_object is 0\n");
    std::fflush(stdout);
    return 2;
  }

  // 5) allocate output (GPU FB, registered-VRAM so it gets translated) and
  //    kernarg (host KERNARG region). Allocate the full kernarg_segment_size
  //    even though we only fill the 12 explicit bytes.
  size_t out_size = FILL_N * sizeof(uint32_t);
  void* out_ptr = nullptr;
  CHECK(hsa_memory_allocate(fb_rs.region, out_size, &out_ptr));
  // Pre-poison so a no-op kernel is detectable.
  std::memset(out_ptr, 0, out_size);
  std::fprintf(stderr, "STEP: output allocated %p\n", out_ptr); std::fflush(stderr);

  size_t kernarg_alloc = kernarg_size ? kernarg_size : 272;  // expect 272
  void* kernarg_ptr = nullptr;
  CHECK(hsa_memory_allocate(kernarg_rs.region, kernarg_alloc, &kernarg_ptr));
  std::memset(kernarg_ptr, 0, kernarg_alloc);
  // Explicit args ONLY: out ptr (u64) @0, val (u32) @8.
  // out_ptr is a registered VRAM CPU pointer; the queue translates it to the
  // FB GPU address during staging.
  std::memcpy(reinterpret_cast<uint8_t*>(kernarg_ptr) + 0, &out_ptr,
              sizeof(void*));
  uint32_t fill_val = FILL_VAL;
  std::memcpy(reinterpret_cast<uint8_t*>(kernarg_ptr) + 8, &fill_val,
              sizeof(uint32_t));
  std::fprintf(stderr, "STEP: kernarg filled %p\n", kernarg_ptr); std::fflush(stderr);

  // 6) queue
  std::fprintf(stderr, "STEP: creating queue\n"); std::fflush(stderr);
  hsa_queue_t* queue = nullptr;
  CHECK(hsa_queue_create(as.gpu, 256, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr,
                         0, 0, &queue));
  std::fprintf(stderr, "STEP: queue created\n"); std::fflush(stderr);
  std::printf("queue: base=%p size=%u doorbell=0x%llx\n", queue->base_address,
              queue->size,
              (unsigned long long)queue->doorbell_signal.handle);

  // 7) completion signal (init 1, dispatch will StoreRelease 0)
  hsa_signal_t completion{};
  CHECK(hsa_signal_create(1, 0, nullptr, &completion));

  // 8) build the AQL packet in the ring
  uint64_t write_index = hsa_queue_add_write_index_relaxed(queue, 1);
  uint64_t mask = (uint64_t)queue->size - 1;
  hsa_kernel_dispatch_packet_t* packets =
      static_cast<hsa_kernel_dispatch_packet_t*>(queue->base_address);
  hsa_kernel_dispatch_packet_t* pkt = &packets[write_index & mask];

  // Zero the packet body but leave the header for last.
  std::memset(reinterpret_cast<uint8_t*>(pkt) + 4, 0, sizeof(*pkt) - 4);
  pkt->setup = 1u << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;  // 1 dim
  pkt->workgroup_size_x = (uint16_t)BLOCK_X;
  pkt->workgroup_size_y = 1;
  pkt->workgroup_size_z = 1;
  pkt->grid_size_x = FILL_N;  // total work-items (thread-dims default mode)
  pkt->grid_size_y = 1;
  pkt->grid_size_z = 1;
  pkt->private_segment_size = private_size;  // expect 0
  pkt->group_segment_size = group_size;      // expect 0
  pkt->kernel_object = kernel_object;
  pkt->kernarg_address = kernarg_ptr;        // HOST pointer
  pkt->completion_signal = completion;

  // header LAST, with system acquire/release fences
  uint16_t header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                    (HSA_FENCE_SCOPE_SYSTEM
                     << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                    (HSA_FENCE_SCOPE_SYSTEM
                     << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
  // header LAST with release ordering (MSVC has no __atomic_store_n): a
  // release fence then a plain store publishes the fully-written packet.
  std::atomic_thread_fence(std::memory_order_release);
  *reinterpret_cast<volatile uint16_t*>(&pkt->header) = header;

  // tell the packet processor the queue advanced, then ring the doorbell.
  hsa_queue_store_write_index_screlease(queue, write_index + 1);
  hsa_signal_store_relaxed(queue->doorbell_signal, (hsa_signal_value_t)write_index);
  std::printf("dispatched write_index=%llu\n",
              (unsigned long long)write_index);

  // 9) wait with a deadline (dispatch is synchronous; this should return fast)
  const uint64_t kTimeout = (uint64_t)1e10;  // ~ a few seconds of HW ticks
  hsa_signal_value_t v = hsa_signal_wait_scacquire(
      completion, HSA_SIGNAL_CONDITION_LT, 1, kTimeout, HSA_WAIT_STATE_BLOCKED);
  std::printf("completion signal value=%lld\n", (long long)v);
  bool timed_out = (v >= 1);

  // VERIFY before shutdown (shutdown is known to crash on this path).
  const uint32_t* out32 = static_cast<const uint32_t*>(out_ptr);
  int bad = 0;
  for (uint32_t i = 0; i < FILL_N; ++i) {
    if (out32[i] != FILL_VAL) ++bad;
  }
  bool pass = (bad == 0) && !timed_out;
  std::printf("out[0]=0x%08x out[63]=0x%08x\n", out32[0], out32[FILL_N - 1]);
  std::printf("HSA_AQL_DISPATCH %s bad=%d/64%s\n", pass ? "PASS" : "FAIL", bad,
              timed_out ? " (TIMEOUT)" : "");
  std::fflush(stdout);

  // best-effort teardown; shutdown may crash -- output is already flushed.
  hsa_signal_destroy(completion);
  hsa_queue_destroy(queue);
  hsa_memory_free(kernarg_ptr);
  hsa_memory_free(out_ptr);
  hsa_executable_destroy(exe);
  hsa_code_object_reader_destroy(cor);
  std::fflush(stdout);
  hsa_shut_down();
  return pass ? 0 : 1;
}
