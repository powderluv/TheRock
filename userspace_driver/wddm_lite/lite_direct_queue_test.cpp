/*
 * lite_direct_queue_test.cpp - Minimal harness: ROCr lite:: direct queue
 * dispatching over the proven wddm_lite gfx1201 driver.
 *
 * This is the lite:: analogue of wddm_lite_test's "nop" path. It validates the
 * ROCr WindowsLiteDriver foundation seam WITHOUT building the full ROCr/
 * hsa-runtime: it links only
 *     amd_lite_direct_queue.cpp  (the SHARED lite:: queue logic)
 *   + wddm_lite.cpp + gpu_init.cpp (the proven driver)
 * and provides a thin DirectQueuePlatform backed by a WddmLite instance --
 * exactly the overrides WindowsLiteDriver implements, minus the core::Driver
 * base class (which would drag in the whole ROCr stack).
 *
 * Flow:
 *   1. WddmLite.open() -> getInfo -> ipDiscovery -> gmcInit.
 *   2. wddmGfxBringUp(): recipeBootload (BOOTLOAD_COMPLETE) + NBIO doorbell
 *      aperture + cqInitGfxForCompute (MEC enable).  [the GPU bring-up]
 *   3. lite::CreateDirectQueue() over this platform (allocates ring/MQD/rptr/
 *      wptr in ONE VRAM buffer via AllocateQueueMemory -> wddmAllocVram, builds
 *      + writes the v12 MQD, activates the HQD via CP_HQD_* MMIO).
 *   4. Build PM4 NOP*4 + RELEASE_MEM(fence,1); lite::SubmitDirectQueue().
 *   5. Poll the fence dword. PASS iff it reaches 1.
 *
 * PASS criteria mirror wddm_lite_test "nop": BOOTLOAD_STATUS bit31 set inside
 * bring-up, MEC enabled (0x3C000000), and the lite:: RELEASE_MEM fence signals.
 */

#include "wddm_lite.h"
#include "core/inc/amd_lite_direct_queue.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace rocr::AMD;

/* ------------------------------------------------------------------------- *
 * PM4 builders (local copies of gpu_init.cpp's static pm4Nop /
 * pm4ReleaseMemFence -- they are file-static there, so we replicate the two
 * tiny ones the NOP+fence path needs). Byte-for-byte identical packets.
 * ------------------------------------------------------------------------- */
static void pm4Pkt3(std::vector<uint32_t> &dw, uint32_t opcode,
                    const uint32_t *payload, uint32_t n) {
  uint32_t header = (3u << 30) | (((n - 1) & 0x3FFF) << 16) | (opcode << 8);
  dw.push_back(header);
  for (uint32_t i = 0; i < n; i++) dw.push_back(payload[i]);
}
static void pm4Nop(std::vector<uint32_t> &dw, uint32_t count) {
  for (uint32_t i = 0; i < count; i++) {
    uint32_t zero = 0;
    pm4Pkt3(dw, 0x10 /*PACKET3_NOP*/, &zero, 1);
  }
}
static void pm4ReleaseMemFence(std::vector<uint32_t> &dw, uint64_t addr,
                               uint64_t value) {
  /* CACHE_FLUSH_AND_INV_TS_EVENT(0x14) | EVENT_INDEX_EOP(5)<<8, full GCR. */
  uint32_t dw0 = (0x14u & 0x3F) | ((5u & 0xF) << 8);
  dw0 |= (1u << 14) | (1u << 15) | (1u << 20) | (1u << 12) | (1u << 13) |
         (1u << 21) | (1u << 22);
  uint32_t dw1 = ((2u & 0x7) << 29) | ((2u & 0x3) << 24); /* 64-bit, int */
  uint32_t payload[7] = {dw0, dw1,
                         (uint32_t)(addr & 0xFFFFFFFF),
                         (uint32_t)((addr >> 32) & 0xFFFFFFFF),
                         (uint32_t)(value & 0xFFFFFFFF),
                         (uint32_t)((value >> 32) & 0xFFFFFFFF), 0};
  pm4Pkt3(dw, 0x49 /*PACKET3_RELEASE_MEM*/, payload, 7);
}

/* ------------------------------------------------------------------------- *
 * WddmLiteDirectPlatform: the lite:: DirectQueuePlatform backed by WddmLite.
 *
 * This is the standalone twin of WindowsLiteDriver's private
 * lite::DirectQueuePlatform overrides. The MMIO byte-offset convention
 *     byte_offset = (base + reg) * 4   (BAR0)
 * matches gpu_init.cpp gcReg/mmhubRead and the Linux transport exactly.
 *
 * Queue memory is the "allocated" path (PreferAllocatedQueueMemory()=true):
 * one wddmAllocVram() per queue, gpu_addr = vramMcBase + offset, so lite::
 * BuildQueueLayoutFromMemory uses cpu_base directly and never calls
 * GpuMemoryCpuPointer/ZeroGpuMemory/WriteGpuMemory32. The doorbell is BAR2 at
 * doorbell_index * 4.
 * ------------------------------------------------------------------------- */
class WddmLiteDirectPlatform : public lite::DirectQueuePlatform {
 public:
  WddmLiteDirectPlatform(WddmLite &gpu, const IpDiscoveryResult &ipd,
                         const WddmComputeContext &ctx)
      : gpu_(gpu), ipd_(ipd), ctx_(ctx) {}

  hsa_status_t EnsureDoorbellAperture() const override {
    return wddmEnsureDoorbellAperture(gpu_, ipd_) ? HSA_STATUS_SUCCESS
                                                  : HSA_STATUS_ERROR;
  }

  hsa_status_t ReadMmio32(uint32_t base, uint32_t reg,
                          uint32_t *value) const override {
    if (value == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    uint32_t off = (base + reg) * 4;
    return gpu_.readReg32(off, value, 0) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
  }

  hsa_status_t WriteMmio32(uint32_t base, uint32_t reg,
                           uint32_t value) const override {
    uint32_t off = (base + reg) * 4;
    return gpu_.writeReg32(off, value, 0) ? HSA_STATUS_SUCCESS
                                          : HSA_STATUS_ERROR;
  }

  hsa_status_t FlushHdp() const override {
    /* HDP flush = write 0 to HDP_MEM_COHERENCY_FLUSH (0x00F7) on NBIF base 2. */
    if (!ctx_.hasNbif) return HSA_STATUS_SUCCESS;
    return gpu_.writeReg32((ctx_.nbifBase2 + 0x00F7) * 4, 0)
               ? HSA_STATUS_SUCCESS
               : HSA_STATUS_ERROR;
  }

  /* Allocated-memory path: one VRAM buffer per queue. */
  bool PreferAllocatedQueueMemory() const override { return true; }

  hsa_status_t AllocateQueueMemory(
      uint64_t size, lite::DirectQueueMemory *memory) const override {
    if (memory == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    *memory = {};
    void *cpu = nullptr;
    uint64_t gpu_addr = 0, handle = 0;
    if (!wddmAllocVram(gpu_, size, &cpu, &gpu_addr, &handle))
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    memory->size = size;
    memory->gpu_addr = gpu_addr;
    memory->cpu = cpu;
    memory->platform_handle = handle;
    return HSA_STATUS_SUCCESS;
  }

  hsa_status_t FreeQueueMemory(lite::DirectQueueMemory *memory) const override {
    /* Bump allocator does not reclaim (matches the scaffold + recipe). */
    if (memory != nullptr) *memory = {};
    return HSA_STATUS_SUCCESS;
  }

  /* cpu_base path is used by the allocated layout, so these are only reached
   * if lite:: ever falls back to the VRAM-window mode. Provide them for
   * completeness mirroring the driver. */
  void *GpuMemoryCpuPointer(uint64_t) const override { return nullptr; }
  hsa_status_t ZeroGpuMemory(uint64_t, uint64_t) const override {
    return HSA_STATUS_ERROR;
  }
  hsa_status_t WriteGpuMemory32(uint64_t, uint32_t) const override {
    return HSA_STATUS_ERROR;
  }

  volatile uint64_t *DoorbellCpuPointer(uint32_t doorbell_index) const override {
    /* BAR2 at doorbell_index * 4 (matches gpu_init.cpp + lite:: Linux). The
     * mapping is cached per index so repeated calls return the same VA. */
    for (auto &m : doorbells_)
      if (m.index == doorbell_index) return m.cpu;
    void *addr = nullptr, *handle = nullptr;
    if (!gpu_.mapBar(2, (uint64_t)doorbell_index * 4, 8, &addr, &handle) ||
        addr == nullptr) {
      std::fprintf(stderr, "  DoorbellCpuPointer: mapBar(2, 0x%X) failed\n",
                   doorbell_index * 4);
      return nullptr;
    }
    doorbells_.push_back({doorbell_index, (volatile uint64_t *)addr});
    return (volatile uint64_t *)addr;
  }

  void SleepUs(uint32_t usec) const override {
    ::Sleep(usec >= 1000 ? usec / 1000 : 1);
  }

 private:
  struct DbMap {
    uint32_t index;
    volatile uint64_t *cpu;
  };
  WddmLite &gpu_;
  const IpDiscoveryResult &ipd_;
  WddmComputeContext ctx_;
  mutable std::vector<DbMap> doorbells_;
};

int main(int argc, char *argv[]) {
  const char *fwDir = "Z:\\winfw";
  if (argc > 1) fwDir = argv[1];

  printf("=== lite_direct_queue_test (ROCr lite:: NOP fence over wddm_lite) "
         "===\n");
  printf("Firmware dir: %s\n", fwDir);

  WddmLite gpu;
  if (!gpu.open()) {
    printf("FATAL: cannot open GPU\n");
    return 1;
  }

  AMDGPU_ESCAPE_GET_INFO_DATA info;
  if (!gpu.getInfo(&info) || info.VendorId != 0x1002) {
    printf("FATAL: GET_INFO failed / not AMD\n");
    return 1;
  }

  IpDiscoveryResult ipd;
  if (!ipDiscovery(gpu, info, ipd) || !ipd.valid) {
    printf("FATAL: IP discovery failed\n");
    return 1;
  }

  GmcState gmc;
  if (!gmcInit(gpu, ipd, gmc) || gmc.vramSize == 0) {
    printf("FATAL: GMC init failed\n");
    return 1;
  }
  printf("vram_mc_base = 0x%llX\n", (unsigned long long)gmc.vramStart);

  /* GPU bring-up: bootload + doorbell aperture + MEC enable. */
  WddmComputeContext ctx;
  if (!wddmGfxBringUp(gpu, ipd, fwDir, gmc.vramStart, ctx)) {
    printf("FATAL: wddmGfxBringUp failed (no BOOTLOAD_COMPLETE / MEC)\n");
    return 1;
  }
  if (!ctx.mecEnabled)
    printf("WARNING: MEC not at 0x3C000000; queue activate may fail\n");

  /* lite:: direct queue. framebuffer_base = vramMcBase: lite:: adds it to the
   * queue offset, and AllocateQueueMemory already returns vramMcBase+offset,
   * so the allocated path is self-consistent. */
  WddmLiteDirectPlatform platform(gpu, ipd, ctx);

  lite::DirectQueueOptions options;
  options.use_mes_queue = false;       /* direct HQD; MES is the next increment */
  options.use_firmware_dequeue = true;
  options.trace = (getenv("LITE_TRACE") != nullptr);
  options.trace_prefix = "lite_direct_queue_test";

  lite::DirectQueueState queue;
  hsa_status_t st = lite::CreateDirectQueue(platform, &queue, /*queue_index=*/0,
                                            ctx.vramMcBase, options);
  if (st != HSA_STATUS_SUCCESS) {
    printf("FAIL: lite::CreateDirectQueue status=%u\n", st);
    return 1;
  }
  printf("lite::CreateDirectQueue OK: qid=%u doorbell=0x%X ring_gpu=0x%llX\n",
         queue.queue_id, queue.doorbell_index,
         (unsigned long long)queue.ring_gpu);

  /* A separate fence buffer (FB-MC addressable + CPU mapped). */
  void *fenceCpu = nullptr;
  uint64_t fenceGpu = 0, fenceHandle = 0;
  if (!wddmAllocVram(gpu, 4096, &fenceCpu, &fenceGpu, &fenceHandle)) {
    printf("FAIL: fence buffer alloc\n");
    return 1;
  }
  volatile uint64_t *fence = (volatile uint64_t *)fenceCpu;
  *fence = 0;

  /* NOP*4 + RELEASE_MEM(fence, 1). */
  std::vector<uint32_t> pm4;
  pm4Nop(pm4, 4);
  pm4ReleaseMemFence(pm4, fenceGpu, 1);

  st = lite::SubmitDirectQueue(platform, queue, pm4.data(), pm4.size(), options);
  if (st != HSA_STATUS_SUCCESS) {
    printf("FAIL: lite::SubmitDirectQueue status=%u\n", st);
    return 1;
  }

  /* Poll the fence (5s). */
  bool ok = false;
  for (int i = 0; i < 5000; i++) {
    if (*fence == 1) { ok = true; break; }
    ::Sleep(1);
  }

  uint32_t rptr = 0;
  lite::ReadDirectQueueRptr(platform, queue, &rptr);

  printf("\nFENCE value=%llu (expected 1) RPTR=0x%X\n",
         (unsigned long long)*fence, rptr);
  printf("LITE NOP+FENCE %s\n", ok ? "PASS" : "FAIL (fence timeout)");

  lite::DestroyDirectQueue(platform, queue, options);
  gpu.close();
  return ok ? 0 : 1;
}
