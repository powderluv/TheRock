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

#include <atomic>
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

/* ------------------------------------------------------------------------- *
 * MES diagnostic dump: read the MES engine + KIQ HQD + compute-queue state via
 * the same MMIO seam the lite:: MES path uses, so a stalled fence can be traced
 * to "engine not started / KIQ not fetching / ADD_QUEUE not servicing". All
 * reads go through the platform overrides (BAR0 (base+reg)*4). GC base_idx1 =
 * 0xA000 (CP_MES_*), base_idx0 = 0x1260 (CP_HQD_* after GRBM_GFX_CNTL select).
 * ------------------------------------------------------------------------- */
namespace {
constexpr uint32_t kGcB0 = 0x1260;
constexpr uint32_t kGcB1 = 0xA000;
constexpr uint32_t regGRBM_GFX_CNTL = 0x0900;   /* base_idx1 */
constexpr uint32_t regCP_MES_CNTL_D = 0x2807;
constexpr uint32_t regCP_MES_GP3_LO_D = 0x2849;
constexpr uint32_t regCP_MES_HEADER_DUMP_D = 0x280D;
constexpr uint32_t regCP_MES_INSTR_PNTR_D = 0x2813;
constexpr uint32_t regRLC_CP_SCHEDULERS_D = 0x098A;
constexpr uint32_t regCP_HQD_ACTIVE_D = 0x1FAB;
constexpr uint32_t regCP_HQD_PQ_RPTR_D = 0x1FB3;
constexpr uint32_t regCP_HQD_PQ_WPTR_LO_D = 0x1FDF;
/* WPTR_POLL workaround signals (ROCR_WINDOWS_MES_WPTR_POLL): the global
 * poll-enable + the per-HQD poll address the MES KIQ pipe's CP polls. */
constexpr uint32_t regCP_PQ_WPTR_POLL_CNTL_D = 0x1E23;
constexpr uint32_t regCP_HQD_PQ_WPTR_POLL_ADDR_D = 0x1FB6;
constexpr uint32_t regCP_HQD_PQ_WPTR_POLL_ADDR_HI_D = 0x1FB7;

/* IH v7.0 register DWORD offsets relative to OSSSYS/IH base (ih_init.py). The
 * IH base is NOT the GC base -- it is ipd.ihBase, carried via WddmIhState. */
constexpr uint32_t regIH_RB_RPTR_D = 0x0081;
constexpr uint32_t regIH_RB_WPTR_D = 0x0082;

void mesSelectHqd(const lite::DirectQueuePlatform &p, uint32_t me, uint32_t pipe,
                  uint32_t hqd) {
  uint32_t v = ((pipe & 0x3) << 0) | ((me & 0x3) << 2) | ((hqd & 0x7) << 8);
  p.WriteMmio32(kGcB1, regGRBM_GFX_CNTL, v);
}
void mesDeselectHqd(const lite::DirectQueuePlatform &p) {
  p.WriteMmio32(kGcB1, regGRBM_GFX_CNTL, 0);
}

void dumpMesDiag(const lite::DirectQueuePlatform &p,
                 const lite::DirectQueueState &queue, const char *phase,
                 const WddmIhState *ih = nullptr) {
  uint32_t mesCntl = 0, version = 0, sched = 0, hdr = 0, ip = 0;
  p.ReadMmio32(kGcB1, regCP_MES_CNTL_D, &mesCntl);
  p.ReadMmio32(kGcB1, regCP_MES_GP3_LO_D, &version);
  p.ReadMmio32(kGcB1, regRLC_CP_SCHEDULERS_D, &sched);
  p.ReadMmio32(kGcB1, regCP_MES_HEADER_DUMP_D, &hdr);
  p.ReadMmio32(kGcB1, regCP_MES_INSTR_PNTR_D, &ip);
  bool p0 = (mesCntl & (1u << 26)) != 0;
  bool p1 = (mesCntl & (1u << 27)) != 0;
  printf("  [MES %s] CP_MES_CNTL=0x%08X (PIPE0_ACTIVE=%d PIPE1_ACTIVE=%d) "
         "GP3_LO(ver)=0x%08X RLC_SCHED=0x%08X HEADER_DUMP=0x%08X "
         "INSTR_PNTR=0x%08X\n",
         phase, mesCntl, p0, p1, version, sched, hdr, ip);

  /* KIQ HQD (me=3 pipe=1 hqd=0): is the KIQ ring active + fetching? */
  mesSelectHqd(p, 3, 1, 0);
  uint32_t kiqActive = 0, kiqRptr = 0, kiqWptr = 0;
  uint32_t kiqPollCntl = 0, kiqPollAddr = 0, kiqPollAddrHi = 0;
  p.ReadMmio32(kGcB0, regCP_HQD_ACTIVE_D, &kiqActive);
  p.ReadMmio32(kGcB0, regCP_HQD_PQ_RPTR_D, &kiqRptr);
  p.ReadMmio32(kGcB0, regCP_HQD_PQ_WPTR_LO_D, &kiqWptr);
  /* WPTR_POLL workaround: with ROCR_WINDOWS_MES_WPTR_POLL set, the lite::
   * MES path programs CP_PQ_WPTR_POLL_CNTL=0x1 here (vs 0 by default) and
   * the POLL_ADDR = the KIQ ring's in-memory wptr (layout.wptr_gpu). */
  p.ReadMmio32(kGcB0, regCP_PQ_WPTR_POLL_CNTL_D, &kiqPollCntl);
  p.ReadMmio32(kGcB0, regCP_HQD_PQ_WPTR_POLL_ADDR_D, &kiqPollAddr);
  p.ReadMmio32(kGcB0, regCP_HQD_PQ_WPTR_POLL_ADDR_HI_D, &kiqPollAddrHi);
  mesDeselectHqd(p);
  printf("  [MES %s] KIQ HQD(me3,pipe1,hqd0) ACTIVE=0x%X RPTR=0x%X WPTR=0x%X\n",
         phase, kiqActive, kiqRptr, kiqWptr);
  printf("  [MES %s] KIQ WPTR_POLL_CNTL=0x%08X (1=enabled workaround) "
         "POLL_ADDR=0x%08X:%08X (== ring in-mem wptr gpu addr)\n",
         phase, kiqPollCntl, kiqPollAddrHi, kiqPollAddr);

  /* The MES-backed compute queue's HQD is NOT MMIO-owned (MES owns it), so on
   * the MES path read the VRAM rptr/wptr the queue tracks instead. */
  uint64_t qWptr = queue.wptr_cpu ? *queue.wptr_cpu : 0;
  uint64_t qRptr = queue.rptr_cpu ? *queue.rptr_cpu : 0;
  printf("  [MES %s] compute queue qid=%u doorbell=0x%X mes_backed=%d "
         "vram_wptr=%llu vram_rptr=%llu\n",
         phase, queue.queue_id, queue.doorbell_index, queue.mes_backed ? 1 : 0,
         (unsigned long long)qWptr, (unsigned long long)qRptr);

  /* IH ring state -- the KEY new signal for the KIQ-doorbell-servicing probe.
   * IH_RB_WPTR (MMIO) is the hardware write pointer; the writeback dword is the
   * GPU's snooped copy of it (MC_SNOOP + WPTR_WRITEBACK_ENABLE). If EITHER
   * advances after the KIQ doorbell is rung, the doorbell -> interrupt -> IH
   * path WORKS and the residual blocker is the MES consuming the ring. If both
   * stay 0, the doorbell is not generating an interrupt (or the ring is
   * misprogrammed / routing disabled). RPTR is the kernel ISR's drain pointer. */
  if (ih != nullptr && ih->configured) {
    uint32_t ihWptrMmio = 0, ihRptrMmio = 0;
    p.ReadMmio32(ih->ihBase, regIH_RB_WPTR_D, &ihWptrMmio);
    p.ReadMmio32(ih->ihBase, regIH_RB_RPTR_D, &ihRptrMmio);
    uint32_t ihWptrWb =
        ih->wptrCpu ? *(volatile uint32_t *)ih->wptrCpu : 0xFFFFFFFFu;
    printf("  [MES %s] IH ring base=0x%04X IH_RB_CNTL=0x%08X msi_enabled=%d "
           "IH_RB_WPTR(mmio)=0x%X wptr_writeback=0x%X IH_RB_RPTR(mmio)=0x%X\n",
           phase, ih->ihBase, ih->rbCntl, ih->msiEnabled ? 1 : 0, ihWptrMmio,
           ihWptrWb, ihRptrMmio);
    /* Dump the first few 32-byte ring entries (8 dwords each). DW[0] decodes as
     * client_id[7:0] / source_id[15:8] / ring_id[23:16] / vmid[27:24]. */
    if (ih->ringCpu != nullptr) {
      const uint32_t *ring = (const uint32_t *)ih->ringCpu;
      for (uint32_t e = 0; e < 4; e++) {
        const uint32_t *ent = ring + e * 8;
        uint32_t dw0 = ent[0];
        printf("    IH entry[%u] DW0=0x%08X (client=0x%02X src=0x%02X "
               "ring=0x%02X vmid=%u) DW[1..3]=0x%08X 0x%08X 0x%08X\n",
               e, dw0, dw0 & 0xFF, (dw0 >> 8) & 0xFF, (dw0 >> 16) & 0xFF,
               (dw0 >> 24) & 0xF, ent[1], ent[2], ent[3]);
      }
    }
  } else {
    printf("  [MES %s] IH ring: not configured (no IH diagnostics)\n", phase);
  }
}
}  // namespace

/* ------------------------------------------------------------------------- *
 * dbprobe: the decisive "does a host doorbell write actually reach the GPU"
 * test for this WDDM/passthrough setup.
 *
 * WHY: The proven direct compute HQD (me=1/pipe0/queue0, doorbell 0x20) advances
 * via lite::SubmitDirectQueue, which writes CP_HQD_PQ_WPTR_LO/HI over MMIO *and*
 * rings the doorbell (amd_lite_direct_queue.cpp ~2322-2347). So the working HQD
 * has never isolated doorbell delivery -- the MMIO wptr poke alone can latch the
 * CP's internal wptr and trigger the fetch. The MES KIQ doorbell (idx 0x18,
 * me=3) does nothing despite every routing register matching amdgpu. dbprobe
 * answers: is the doorbell-DELIVERY path itself broken on this WDDM/passthrough
 * stack (a KMD-side wall that equally kills the MES KIQ), or is the MES stall
 * pipe-specific?
 *
 * HOW (two sub-runs on the SAME proven direct compute HQD):
 *   RUN 1 (mmio baseline): create a direct compute HQD, submit NOP+RELEASE_MEM
 *     via the normal lite::SubmitDirectQueue (MMIO wptr + doorbell). Expect PASS.
 *     This proves the ring/MQD/EOP/fence are correct on THIS queue.
 *   RUN 2 (doorbell-only): re-arm the same queue, then advance it relying on the
 *     DOORBELL ALONE:
 *       - disable WPTR_POLL so the CP cannot autonomously poll the in-memory
 *         wptr: write the global CP_PQ_WPTR_POLL_CNTL=0 AND zero the per-HQD
 *         CP_HQD_PQ_WPTR_POLL_ADDR/_HI (mqd[0x8D/0x8E]) so there is no memory
 *         wptr-poll source. (gfx12 has no CP_HQD_PQ_WPTR_POLL_CNTL.EN; poll is
 *         gated by a valid poll address + the global cntl.)
 *       - write the PM4 to the ring + write the NEW wptr ONLY to the in-memory
 *         wptr dword (queue.wptr_cpu). FlushHdp.
 *       - DO NOT write CP_HQD_PQ_WPTR_LO/HI over MMIO.
 *       - ring the doorbell (queue.doorbell_cpu = new_wptr) and ONLY that.
 *     Read CP_HQD_PQ_WPTR/RPTR (MMIO) before + after the doorbell, poll the fence
 *     ~2s.
 *
 * VERDICT for RUN 2:
 *   "DOORBELL DELIVERS"  iff the fence signals (the CP fetched + executed purely
 *                        from the doorbell ring -- RPTR advances too).
 *   "DOORBELL DOES NOT DELIVER" iff fence stays 0 AND RPTR stays 0 (the doorbell
 *                        write never reached the CP).
 *
 * INTERPRETATION:
 *   mmio:PASS doorbell-only:PASS => doorbells DO deliver; the MES KIQ stall is
 *     MES-pipe-specific (ADD_QUEUE / scheduler servicing), not delivery.
 *   mmio:PASS doorbell-only:FAIL => doorbell DELIVERY is the wall on this
 *     WDDM/passthrough setup (a KMD-side issue). That explains the MES KIQ stall
 *     and means MES needs a kernel-side doorbell fix, not a userspace one.
 * ------------------------------------------------------------------------- */
namespace {
/* CP_HQD_* are base_idx0 (kGcB0=0x1260); GRBM_GFX_CNTL is base_idx1. These
 * mirror amd_lite_direct_queue.cpp's reg constants so the probe pokes exactly
 * the registers the queue uses. */
constexpr uint32_t regCP_HQD_PQ_WPTR_LO_DB = 0x1FDF;
constexpr uint32_t regCP_HQD_PQ_WPTR_HI_DB = 0x1FE0;
constexpr uint32_t regCP_HQD_PQ_RPTR_DB = 0x1FB3;
constexpr uint32_t regCP_HQD_PQ_WPTR_POLL_ADDR_DB = 0x1FB6;
constexpr uint32_t regCP_HQD_PQ_WPTR_POLL_ADDR_HI_DB = 0x1FB7;
constexpr uint32_t regCP_HQD_ACTIVE_DB = 0x1FAB;
constexpr uint32_t regCP_PQ_WPTR_POLL_CNTL_DB = 0x1E23; /* global poll cntl */
constexpr uint32_t kGcB0_DB = 0x1260;
constexpr uint32_t kGcB1_DB = 0xA000;
constexpr uint32_t regGRBM_GFX_CNTL_DB = 0x0900;

/* GRBM_GFX_CNTL selector for me=1/pipe0/queue0 (DirectQueuePipe(0)=0,
 * DirectQueueHqd(0)=0) -- matches lite::SelectHqd's bit layout. */
void dbSelectHqd(const lite::DirectQueuePlatform &p, uint32_t me, uint32_t pipe,
                 uint32_t queue) {
  uint32_t v = ((pipe & 0x3u) << 0) | ((me & 0x3u) << 2) | ((queue & 0x7u) << 8);
  p.WriteMmio32(kGcB1_DB, regGRBM_GFX_CNTL_DB, v);
}
void dbDeselectHqd(const lite::DirectQueuePlatform &p) {
  p.WriteMmio32(kGcB1_DB, regGRBM_GFX_CNTL_DB, 0);
}

struct DbHqdSnapshot {
  uint32_t active;
  uint32_t rptr;
  uint32_t wptr_lo;
  uint32_t wptr_hi;
};
DbHqdSnapshot dbReadHqd(const lite::DirectQueuePlatform &p) {
  DbHqdSnapshot s{};
  dbSelectHqd(p, 1, 0, 0);
  p.ReadMmio32(kGcB0_DB, regCP_HQD_ACTIVE_DB, &s.active);
  p.ReadMmio32(kGcB0_DB, regCP_HQD_PQ_RPTR_DB, &s.rptr);
  p.ReadMmio32(kGcB0_DB, regCP_HQD_PQ_WPTR_LO_DB, &s.wptr_lo);
  p.ReadMmio32(kGcB0_DB, regCP_HQD_PQ_WPTR_HI_DB, &s.wptr_hi);
  dbDeselectHqd(p);
  return s;
}

/* Build the NOP*4 + RELEASE_MEM(fence,1) block (identical to the main path). */
void dbBuildNopFence(std::vector<uint32_t> &pm4, uint64_t fenceGpu) {
  pm4Nop(pm4, 4);
  pm4ReleaseMemFence(pm4, fenceGpu, 1);
}

/* Advance the queue via the DOORBELL ALONE: write ring + in-memory wptr,
 * FlushHdp, ring the doorbell. NO MMIO CP_HQD_PQ_WPTR write. Mirrors the
 * memory-side of SubmitDirectQueue but deliberately omits the MMIO wptr poke. */
bool dbSubmitDoorbellOnly(const lite::DirectQueuePlatform &platform,
                          lite::DirectQueueState &queue, const uint32_t *pm4,
                          size_t dword_count) {
  if (queue.ring_cpu == nullptr || queue.wptr_cpu == nullptr ||
      queue.doorbell_cpu == nullptr)
    return false;
  const uint64_t ring_dw = queue.ring_size_bytes / sizeof(uint32_t);
  uint64_t wptr = queue.wptr;
  uint64_t start = wptr % ring_dw;
  /* The probe block (NOP*4 + RELEASE_MEM = ~13 dwords) fits well inside the
   * 1 KiB-dword ring at wptr 0; no wrap handling needed for the single submit. */
  for (size_t i = 0; i < dword_count; i++)
    queue.ring_cpu[(start + i) % ring_dw] = pm4[i];
  std::atomic_thread_fence(std::memory_order_release);
  const uint64_t new_wptr = wptr + dword_count;
  *queue.wptr_cpu = new_wptr;  /* in-memory wptr ONLY */
  std::atomic_thread_fence(std::memory_order_release);
  if (platform.FlushHdp() != HSA_STATUS_SUCCESS) return false;
  /* Ring the doorbell and ONLY the doorbell. */
  *queue.doorbell_cpu = new_wptr;
  queue.wptr = new_wptr;
  return true;
}

int runDbProbe(const lite::DirectQueuePlatform &platform,
               lite::DirectQueueState &queue,
               const lite::DirectQueueOptions &options, WddmLite &gpu) {
  printf("\n=== dbprobe: doorbell-delivery isolation on the direct compute HQD "
         "===\n");
  printf("HQD: me=1 pipe=0 queue0 doorbell=0x%X (DirectQueueDoorbell(0))\n",
         queue.doorbell_index);

  /* Shared fence buffer (re-zeroed between runs). */
  void *fenceCpu = nullptr;
  uint64_t fenceGpu = 0, fenceHandle = 0;
  if (!wddmAllocVram(gpu, 4096, &fenceCpu, &fenceGpu, &fenceHandle)) {
    printf("FAIL: fence buffer alloc\n");
    return 1;
  }
  volatile uint64_t *fence = (volatile uint64_t *)fenceCpu;

  /* ---------------- RUN 1: MMIO-wptr baseline (proven path) ---------------- */
  *fence = 0;
  std::atomic_thread_fence(std::memory_order_seq_cst);
  platform.FlushHdp();

  std::vector<uint32_t> pm4a;
  dbBuildNopFence(pm4a, fenceGpu);

  DbHqdSnapshot a_pre = dbReadHqd(platform);
  printf("\n[RUN 1 mmio] pre-submit HQD active=0x%X rptr=0x%X wptr=0x%08X:%08X\n",
         a_pre.active, a_pre.rptr, a_pre.wptr_hi, a_pre.wptr_lo);

  hsa_status_t st =
      lite::SubmitDirectQueue(platform, queue, pm4a.data(), pm4a.size(), options);
  if (st != HSA_STATUS_SUCCESS) {
    printf("[RUN 1 mmio] SubmitDirectQueue status=%u\n", st);
  }
  bool mmioOk = false;
  for (int i = 0; i < 2000; i++) {
    if (*fence == 1) { mmioOk = true; break; }
    ::Sleep(1);
  }
  DbHqdSnapshot a_post = dbReadHqd(platform);
  printf("[RUN 1 mmio] post HQD active=0x%X rptr=0x%X wptr=0x%08X:%08X "
         "fence=%llu\n",
         a_post.active, a_post.rptr, a_post.wptr_hi, a_post.wptr_lo,
         (unsigned long long)*fence);
  printf("[RUN 1 mmio] result: %s\n", mmioOk ? "PASS" : "FAIL");

  /* ---------------- RUN 2: doorbell-only (delivery isolation) ----------------
   * Re-arm the SAME queue. The queue is still active; we keep advancing its
   * monotonic wptr (the in-memory wptr + doorbell value both carry the absolute
   * wptr), so the second block lands right after the first. */
  *fence = 0;
  std::atomic_thread_fence(std::memory_order_seq_cst);
  platform.FlushHdp();

  /* Disable WPTR_POLL so a fence signal in RUN 2 can ONLY come from the doorbell:
   *   (a) global CP_PQ_WPTR_POLL_CNTL = 0
   *   (b) per-HQD CP_HQD_PQ_WPTR_POLL_ADDR/_HI = 0 (no memory poll source)
   * After this the CP has no autonomous way to learn the new wptr from memory;
   * only a doorbell write latches it into the HQD's internal wptr. */
  dbSelectHqd(platform, 1, 0, 0);
  uint32_t pollCntlBefore = 0, pollAddrBefore = 0, pollAddrHiBefore = 0;
  platform.ReadMmio32(kGcB0_DB, regCP_PQ_WPTR_POLL_CNTL_DB, &pollCntlBefore);
  platform.ReadMmio32(kGcB0_DB, regCP_HQD_PQ_WPTR_POLL_ADDR_DB, &pollAddrBefore);
  platform.ReadMmio32(kGcB0_DB, regCP_HQD_PQ_WPTR_POLL_ADDR_HI_DB,
                      &pollAddrHiBefore);
  platform.WriteMmio32(kGcB0_DB, regCP_PQ_WPTR_POLL_CNTL_DB, 0);
  platform.WriteMmio32(kGcB0_DB, regCP_HQD_PQ_WPTR_POLL_ADDR_DB, 0);
  platform.WriteMmio32(kGcB0_DB, regCP_HQD_PQ_WPTR_POLL_ADDR_HI_DB, 0);
  uint32_t pollCntlAfter = 0, pollAddrAfter = 0;
  platform.ReadMmio32(kGcB0_DB, regCP_PQ_WPTR_POLL_CNTL_DB, &pollCntlAfter);
  platform.ReadMmio32(kGcB0_DB, regCP_HQD_PQ_WPTR_POLL_ADDR_DB, &pollAddrAfter);
  dbDeselectHqd(platform);
  printf("\n[RUN 2 doorbell-only] WPTR_POLL disabled: CP_PQ_WPTR_POLL_CNTL "
         "0x%08X->0x%08X  CP_HQD_PQ_WPTR_POLL_ADDR 0x%08X(hi 0x%08X)->0x%08X\n",
         pollCntlBefore, pollCntlAfter, pollAddrBefore, pollAddrHiBefore,
         pollAddrAfter);

  std::vector<uint32_t> pm4b;
  dbBuildNopFence(pm4b, fenceGpu);

  DbHqdSnapshot b_pre = dbReadHqd(platform);
  const uint64_t db_value = queue.wptr + pm4b.size();
  printf("[RUN 2 doorbell-only] pre-doorbell HQD active=0x%X rptr=0x%X "
         "wptr=0x%08X:%08X | will write in-mem wptr=%llu, ring doorbell idx=0x%X "
         "value=%llu, NO MMIO wptr write\n",
         b_pre.active, b_pre.rptr, b_pre.wptr_hi, b_pre.wptr_lo,
         (unsigned long long)db_value, queue.doorbell_index,
         (unsigned long long)db_value);

  if (!dbSubmitDoorbellOnly(platform, queue, pm4b.data(), pm4b.size())) {
    printf("[RUN 2 doorbell-only] FAIL: submit setup error\n");
    return 1;
  }

  bool dbOk = false;
  for (int i = 0; i < 2000; i++) {
    if (*fence == 1) { dbOk = true; break; }
    ::Sleep(1);
  }
  DbHqdSnapshot b_post = dbReadHqd(platform);
  printf("[RUN 2 doorbell-only] post-doorbell HQD active=0x%X rptr=0x%X "
         "wptr=0x%08X:%08X fence=%llu\n",
         b_post.active, b_post.rptr, b_post.wptr_hi, b_post.wptr_lo,
         (unsigned long long)*fence);

  const bool rptrAdvanced = b_post.rptr != b_pre.rptr;
  const char *verdict;
  if (dbOk)
    verdict = "DOORBELL DELIVERS";
  else if (!rptrAdvanced && b_post.rptr == 0 && b_pre.rptr == 0)
    verdict = "DOORBELL DOES NOT DELIVER";
  else
    verdict = "INCONCLUSIVE (fence=0 but RPTR moved -- queue/ring fault, not "
              "clean delivery failure)";
  printf("[RUN 2 doorbell-only] VERDICT: %s\n", verdict);

  /* ---------------------------- summary ---------------------------- */
  printf("\n=== dbprobe summary: mmio:%s doorbell-only:%s ===\n",
         mmioOk ? "PASS" : "FAIL", dbOk ? "PASS" : "FAIL");
  if (mmioOk && dbOk) {
    printf("INTERPRETATION: doorbells DO reach the GPU. The MES KIQ stall is "
           "pipe-specific (MES scheduler/ADD_QUEUE servicing), NOT doorbell "
           "delivery.\n");
  } else if (mmioOk && !dbOk) {
    printf("INTERPRETATION: doorbell DELIVERY is the wall on this WDDM/"
           "passthrough setup (KMD-side). This explains the MES KIQ stall; MES "
           "needs a kernel-side doorbell fix, not a userspace one.\n");
  } else {
    printf("INTERPRETATION: the MMIO baseline itself FAILED -- the queue/ring/"
           "fence is not healthy on this HQD, so the doorbell-only result is "
           "not isolating. Fix the baseline first.\n");
  }
  return (mmioOk && dbOk) ? 0 : 1;
}
}  // namespace

/* ------------------------------------------------------------------------- *
 * mesmulti: MES multi-dispatch ceiling test (the #21 ~14-dispatch ceiling).
 *
 * Creates ONE MES-backed compute queue (CreateDirectQueue use_mes_queue=true),
 * then submits N back-to-back NOP*4 + RELEASE_MEM(fence, seq) packets through
 * lite::SubmitDirectQueue (each carries the ROCR_WINDOWS_MES_MMIO_WPTR poke
 * inside SubmitDirectQueue's mes_backed branch), incrementing the fence value
 * each submit and polling each fence to completion before the next submit.
 *
 * RING WRAP: each NOP*4+RELEASE_MEM block is exactly 16 dwords (4 * 2-dword NOP
 * + 8-dword RELEASE_MEM). The MES-backed ring is kDirectComputeRingSize = 0x1000
 * bytes = 1024 dwords, so it wraps every 1024/16 = 64 submits; 1024 submits is
 * 16 full revolutions. We do NOT re-implement wrap handling here: the wrap is
 * handled INSIDE lite::SubmitDirectQueue (it NOP-pads from the current offset to
 * the ring end with a real PM4 TYPE-3 NOP, advances queue.wptr to the boundary,
 * then writes the real block at offset 0 -- the #16 straddle fix). queue.wptr is
 * the monotonic absolute wptr maintained by SubmitDirectQueue across calls, so
 * back-to-back submits land contiguously and wrap correctly with no caller-side
 * bookkeeping. We pass the SAME queue + a single shared fence buffer.
 *
 * Reports per stage: dispatches completed, where (if anywhere) it stalled, and
 * the final rptr/wptr/fence. The max consecutive successes across all stages is
 * the ceiling answer for the #21 blocker.
 * ------------------------------------------------------------------------- */
namespace {
int runMesMulti(const lite::DirectQueuePlatform &platform,
                lite::DirectQueueState &queue,
                const lite::DirectQueueOptions &options, WddmLite &gpu,
                const WddmIhState &ih) {
  printf("\n=== mesmulti: MES multi-dispatch ceiling (NOP+fence x N) ===\n");
  printf("queue qid=%u doorbell=0x%X mes_backed=%d ring=%u bytes (%u dwords)\n",
         queue.queue_id, queue.doorbell_index, queue.mes_backed ? 1 : 0,
         queue.ring_size_bytes, queue.ring_size_bytes / 4u);
  printf("packet = NOP*4 + RELEASE_MEM = 16 dwords; ring wraps every %u "
         "submits (handled inside lite::SubmitDirectQueue)\n",
         (queue.ring_size_bytes / 4u) / 16u);

  /* One shared 64-bit fence dword; each submit writes an incrementing value. */
  void *fenceCpu = nullptr;
  uint64_t fenceGpu = 0, fenceHandle = 0;
  if (!wddmAllocVram(gpu, 4096, &fenceCpu, &fenceGpu, &fenceHandle)) {
    printf("FAIL: fence buffer alloc\n");
    return 1;
  }
  volatile uint64_t *fence = (volatile uint64_t *)fenceCpu;
  *fence = 0;
  std::atomic_thread_fence(std::memory_order_seq_cst);
  platform.FlushHdp();

  const uint32_t stages[] = {16, 64, 256, 1024};
  /* fenceSeq is monotonic across ALL stages (the fence dword is never reset, so
   * RELEASE_MEM writes a strictly increasing value -- no ABA on the poll). */
  uint64_t fenceSeq = 0;
  uint64_t totalDone = 0;
  uint64_t maxConsecutive = 0;
  bool stalled = false;
  uint64_t stallAt = 0;

  for (uint32_t s = 0; s < sizeof(stages) / sizeof(stages[0]) && !stalled; s++) {
    const uint32_t N = stages[s];
    printf("\n--- stage: %u dispatches ---\n", N);
    uint32_t doneThisStage = 0;
    for (uint32_t i = 0; i < N; i++) {
      fenceSeq += 1;
      std::vector<uint32_t> pm4;
      pm4Nop(pm4, 4);
      pm4ReleaseMemFence(pm4, fenceGpu, fenceSeq);

      hsa_status_t st = lite::SubmitDirectQueue(platform, queue, pm4.data(),
                                                pm4.size(), options);
      if (st != HSA_STATUS_SUCCESS) {
        printf("  STALL: SubmitDirectQueue status=%u at dispatch %llu "
               "(stage index %u)\n",
               st, (unsigned long long)fenceSeq, i);
        stalled = true;
        stallAt = fenceSeq - 1;  /* last good one */
        break;
      }
      /* Poll this submit's fence (>= seq tolerates a CP that has already raced
       * ahead to a later value). 2s budget per dispatch. */
      bool ok = false;
      for (int t = 0; t < 2000; t++) {
        if (*fence >= fenceSeq) { ok = true; break; }
        ::Sleep(1);
      }
      if (!ok) {
        uint32_t rptr = 0;
        lite::ReadDirectQueueRptr(platform, queue, &rptr);
        printf("  STALL: fence timeout at dispatch %llu (stage index %u); "
               "fence=%llu expected>=%llu rptr=0x%X wptr=%llu\n",
               (unsigned long long)fenceSeq, i, (unsigned long long)*fence,
               (unsigned long long)fenceSeq, rptr,
               (unsigned long long)queue.wptr);
        stalled = true;
        stallAt = fenceSeq - 1;
        break;
      }
      doneThisStage++;
      totalDone++;
      if (totalDone > maxConsecutive) maxConsecutive = totalDone;
    }
    uint32_t rptr = 0;
    lite::ReadDirectQueueRptr(platform, queue, &rptr);
    printf("  stage %u: dispatches completed=%u/%u  final rptr=0x%X wptr=%llu "
           "fence=%llu\n",
           N, doneThisStage, N, rptr, (unsigned long long)queue.wptr,
           (unsigned long long)*fence);
    if (ih.configured)
      dumpMesDiag(platform, queue, "mesmulti-stage", &ih);
  }

  printf("\n=== mesmulti summary ===\n");
  printf("max consecutive successful dispatches: %llu\n",
         (unsigned long long)maxConsecutive);
  if (stalled)
    printf("STALLED after %llu dispatches (next submit/fence at #%llu failed)\n",
           (unsigned long long)stallAt, (unsigned long long)(stallAt + 1));
  else
    printf("NO STALL: all %llu dispatches (16+64+256+1024) completed -- no "
           "multi-dispatch ceiling on the Windows MES path\n",
           (unsigned long long)totalDone);
  printf("LITE MES MULTI %s\n", stalled ? "FAIL (ceiling hit)"
                                        : "PASS (no ceiling)");
  return stalled ? 1 : 0;
}

/* ------------------------------------------------------------------------- *
 * meskern: a REAL compute kernel dispatched through the MES-backed queue.
 *
 * Reuses the PROVEN recipeKernargDispatch machinery via the wddm_lite helpers
 * (wddmStageKernelGpuvm stages fill_kernel_raw.co + builds the GFXHUB page table
 * + enables GCVM_CONTEXT0; wddmBuildKernelDispatchPm4 builds the dispatch PM4),
 * but routes the DISPATCH through the lite:: MES-backed queue
 * (lite::SubmitDirectQueue with the ROCR_WINDOWS_MES_MMIO_WPTR poke) instead of
 * the direct-MMIO HQD. PASS iff the RELEASE_MEM fence signals AND the GPUVM
 * fault status is 0 AND out[0..63] == 0xDEADBEEF.
 *
 * The GPUVM stage is done BEFORE the queue submit but the queue itself was
 * already created by main (the MES mapped it once). The kernel dispatch is a
 * single PM4 block (~69 dwords) submitted ONCE -- well inside the 1024-dword
 * ring, so no wrap. The fence is the staged kernel's fence buffer.
 * ------------------------------------------------------------------------- */
int runMesKern(const lite::DirectQueuePlatform &platform,
               lite::DirectQueueState &queue,
               const lite::DirectQueueOptions &options, WddmLite &gpu,
               const IpDiscoveryResult &ipd, const WddmComputeContext &ctx,
               const char *fwDir, const WddmIhState &ih) {
  printf("\n=== meskern: REAL kernel (fill_kernel) via the MES-backed queue "
         "===\n");
  printf("queue qid=%u doorbell=0x%X mes_backed=%d\n", queue.queue_id,
         queue.doorbell_index, queue.mes_backed ? 1 : 0);

  /* Stage the kernel GPUVM (code + kernarg + output + page table).
   *
   * skipMecReassert=true: main() already created the MES-backed queue
   * (lite::CreateDirectQueue use_mes_queue=true), so the MES has mapped this
   * compute queue onto a MEC HQD slot. Tell the stage NOT to pulse-reset the
   * MEC pipes (cqInitGfxForCompute) -- that would wipe the MES-mapped HQD and
   * meskern never re-issues ADD_QUEUE. The stage still does the GFXHUB GART
   * re-enable + page-table build + GCVM_CONTEXT0 enable, which is what the
   * real-kernel dispatch actually needs. */
  WddmKernelStage stage;
  if (!wddmStageKernelGpuvm(gpu, ipd, ctx, fwDir, stage,
                            /*skipMecReassert=*/true)) {
    printf("FAIL: wddmStageKernelGpuvm failed\n");
    return 1;
  }

  /* Fence buffer (FB-MC addressable + CPU-mapped), separate from the queue. */
  void *fenceCpu = nullptr;
  uint64_t fenceGpu = 0, fenceHandle = 0;
  if (!wddmAllocVram(gpu, 4096, &fenceCpu, &fenceGpu, &fenceHandle)) {
    printf("FAIL: fence buffer alloc\n");
    return 1;
  }
  volatile uint64_t *fence = (volatile uint64_t *)fenceCpu;
  *fence = 0;
  std::atomic_thread_fence(std::memory_order_seq_cst);
  platform.FlushHdp();

  /* Build the dispatch PM4 (RELEASE_MEM writes value 1 to fenceGpu). */
  std::vector<uint32_t> pm4;
  if (!wddmBuildKernelDispatchPm4(stage, fenceGpu, pm4)) {
    printf("FAIL: wddmBuildKernelDispatchPm4 failed\n");
    return 1;
  }
  printf("  meskern: dispatch PM4 = %zu dwords; submitting via the MES-backed "
         "queue\n", pm4.size());

  hsa_status_t st = lite::SubmitDirectQueue(platform, queue, pm4.data(),
                                            pm4.size(), options);
  if (st != HSA_STATUS_SUCCESS) {
    printf("FAIL: lite::SubmitDirectQueue status=%u\n", st);
    if (ih.configured) dumpMesDiag(platform, queue, "meskern-submit-fail", &ih);
    return 1;
  }

  /* Poll the fence (5s). */
  bool ok = false;
  for (int i = 0; i < 5000; i++) {
    if (*fence == 1) { ok = true; break; }
    ::Sleep(1);
  }

  /* HDP flush so the GPU's output writes are CPU-visible. */
  platform.FlushHdp();

  /* GPUVM fault status (GC base_idx0, regGCVM_L2_PROTECTION_FAULT_STATUS=0x15D0,
   * the same register recipeKernargDispatch reads). */
  uint32_t faultStatus = 0;
  platform.ReadMmio32(stage.gcBase0, 0x15D0, &faultStatus);

  uint32_t rptr = 0;
  lite::ReadDirectQueueRptr(platform, queue, &rptr);

  const volatile uint32_t *res = (const volatile uint32_t *)stage.outCpu;
  uint32_t nbad = 0, firstBad = 0, firstBadVal = 0;
  for (uint32_t i = 0; i < stage.fillN; i++) {
    uint32_t v = res[i];
    if (v != stage.fillVal) {
      if (nbad == 0) { firstBad = i; firstBadVal = v; }
      nbad++;
    }
  }

  if (ih.configured)
    dumpMesDiag(platform, queue, ok ? "meskern-PASS" : "meskern-FAIL", &ih);

  printf("\nFAULT_STATUS=0x%08X FENCE value=%llu (expected 1) RPTR=0x%X\n",
         faultStatus, (unsigned long long)*fence, rptr);
  printf("out[0]=0x%08X out[1]=0x%08X expected=0x%08X bad=%u/%u\n", res[0],
         res[1], stage.fillVal, nbad, stage.fillN);
  if (nbad)
    printf("  first mismatch at index %u: got 0x%08X\n", firstBad, firstBadVal);

  bool pass = ok && (faultStatus == 0) && (nbad == 0);
  printf("LITE MES KERN %s\n",
         pass ? "PASS (fence signaled, no fault, output verified)"
         : (!ok ? "FAIL (fence timeout)"
            : (faultStatus ? "FAIL (GPUVM fault)" : "FAIL (output mismatch)")));
  return pass ? 0 : 1;
}
}  // namespace

int main(int argc, char *argv[]) {
  const char *fwDir = "Z:\\winfw";
  bool mesMode = false;
  bool dbProbeMode = false;
  bool mesMultiMode = false;
  bool mesKernMode = false;
  /* Args (order-independent): "mes" selects the single-NOP MES queue path;
   * "mesmulti" runs the MES multi-dispatch ceiling test (N NOP+fence x
   * 16/64/256/1024); "meskern" dispatches a REAL kernel through the MES-backed
   * queue; "dbprobe" selects the doorbell-delivery isolation probe (direct HQD).
   * Any other positional arg is the firmware dir. Default (no mode arg) = the
   * proven direct HQD path, unchanged. mesmulti/meskern imply the MES bring-up.
   *   lite_direct_queue_test.exe mes      Z:\\winfw
   *   lite_direct_queue_test.exe mesmulti Z:\\winfw
   *   lite_direct_queue_test.exe meskern  Z:\\winfw
   *   lite_direct_queue_test.exe dbprobe  Z:\\winfw  */
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "mes") == 0)
      mesMode = true;
    else if (strcmp(argv[i], "mesmulti") == 0)
      mesMultiMode = true;
    else if (strcmp(argv[i], "meskern") == 0)
      mesKernMode = true;
    else if (strcmp(argv[i], "dbprobe") == 0)
      dbProbeMode = true;
    else
      fwDir = argv[i];
  }
  /* mesmulti/meskern both run over a MES-backed queue, so they share the MES
   * bring-up (wddmStartMes + wddmInitIh + use_mes_queue) with the plain "mes"
   * path. useMes drives the bring-up; the specific mode selects what runs after
   * the queue is created. */
  const bool useMes = mesMode || mesMultiMode || mesKernMode;

  printf("=== lite_direct_queue_test (ROCr lite:: NOP fence over wddm_lite) "
         "===\n");
  printf("Firmware dir: %s\n", fwDir);
  printf("Queue path  : %s\n",
         dbProbeMode  ? "DIRECT HQD (proven) [DBPROBE: doorbell-delivery probe]"
         : mesMultiMode ? "MES (use_mes_queue=TRUE) [MESMULTI: ceiling test]"
         : mesKernMode  ? "MES (use_mes_queue=TRUE) [MESKERN: real kernel]"
         : mesMode      ? "MES (use_mes_queue=TRUE) [DIAGNOSTIC]"
                        : "DIRECT HQD (proven)");

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

  /* MES mode: recipeBootload LOADED the MES firmware but never released the
   * engine (the direct path does not use MES). Start it here BEFORE lite::
   * CreateDirectQueue -- EnsureMesScheduler reads CP_MES_CNTL/GP3 and bails if
   * the engine is not running. wddmStartMes is additive + the direct path
   * never calls it. A false return (pipes not ACTIVE) is reported but we
   * CONTINUE so EnsureMesScheduler's own diagnostics still print. */
  WddmLiteDirectPlatform platform(gpu, ipd, ctx);
  WddmIhState ih;
  memset(&ih, 0, sizeof(ih));
  if (useMes) {
    bool mesUp = wddmStartMes(gpu, ipd, fwDir, ctx);
    printf("wddmStartMes -> %s\n",
           mesUp ? "MES pipes ACTIVE" : "MES NOT active (continuing for diag)");

    /* Increment 2: bring the IH ring up BEFORE the KIQ doorbell is rung (inside
     * lite::CreateDirectQueue -> EnsureMesScheduler), so we can observe whether
     * the doorbell generates an interrupt the IH ring captures. Additive +
     * MES-only; the proven direct path never calls it. A false return is
     * reported but we CONTINUE (dumpMesDiag tolerates !ih.configured). */
    bool ihUp = wddmInitIh(gpu, ipd, ctx, ih);
    printf("wddmInitIh -> %s\n",
           ihUp ? "IH ring live" : "IH ring NOT configured (continuing)");
  }

  /* lite:: direct queue. framebuffer_base = vramMcBase: lite:: adds it to the
   * queue offset, and AllocateQueueMemory already returns vramMcBase+offset,
   * so the allocated path is self-consistent. */
  lite::DirectQueueOptions options;
  options.use_mes_queue = useMes;      /* route through the MES KIQ */
  options.use_firmware_dequeue = true;
  /* In MES modes force verbose tracing so EnsureMesScheduler/SubmitMesApiFrame
   * dump CP_MES_CNTL + KIQ ring + SET_HW_RESOURCES/ADD_QUEUE fence state. */
  options.trace = useMes || dbProbeMode || (getenv("LITE_TRACE") != nullptr);
  options.trace_verbose = useMes || (getenv("LITE_TRACE_VERBOSE") != nullptr);
  options.trace_prefix = "lite_direct_queue_test";

  lite::DirectQueueState queue;
  hsa_status_t st = lite::CreateDirectQueue(platform, &queue, /*queue_index=*/0,
                                            ctx.vramMcBase, options);
  if (st != HSA_STATUS_SUCCESS) {
    printf("FAIL: lite::CreateDirectQueue status=%u\n", st);
    /* In MES modes the failure is usually inside EnsureMesScheduler (KIQ never
     * activates / SET_HW_RESOURCES times out). Dump the MES engine state so the
     * stall is diagnosable even though the queue was never mapped. */
    if (useMes) dumpMesDiag(platform, queue, "create-failed", &ih);
    return 1;
  }
  printf("lite::CreateDirectQueue OK: qid=%u doorbell=0x%X ring_gpu=0x%llX\n",
         queue.queue_id, queue.doorbell_index,
         (unsigned long long)queue.ring_gpu);
  if (useMes) dumpMesDiag(platform, queue, "post-create", &ih);

  /* dbprobe mode: run the doorbell-delivery isolation on this proven direct HQD
   * (MMIO baseline then doorbell-only), then tear down + exit. Never touches the
   * MES path. */
  if (dbProbeMode) {
    int rc = runDbProbe(platform, queue, options, gpu);
    lite::DestroyDirectQueue(platform, queue, options);
    gpu.close();
    return rc;
  }

  /* mesmulti mode: MES multi-dispatch ceiling test (NOP+fence x 16/64/256/1024)
   * over the MES-backed queue, then tear down + exit. */
  if (mesMultiMode) {
    int rc = runMesMulti(platform, queue, options, gpu, ih);
    lite::DestroyDirectQueue(platform, queue, options);
    gpu.close();
    return rc;
  }

  /* meskern mode: dispatch a REAL kernel (fill_kernel) through the MES-backed
   * queue, then tear down + exit. */
  if (mesKernMode) {
    int rc = runMesKern(platform, queue, options, gpu, ipd, ctx, fwDir, ih);
    lite::DestroyDirectQueue(platform, queue, options);
    gpu.close();
    return rc;
  }

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

  if (mesMode) dumpMesDiag(platform, queue, ok ? "post-submit-PASS"
                                               : "post-submit-FAIL", &ih);

  printf("\nFENCE value=%llu (expected 1) RPTR=0x%X\n",
         (unsigned long long)*fence, rptr);
  printf("LITE %s NOP+FENCE %s\n", mesMode ? "MES" : "DIRECT",
         ok ? "PASS" : "FAIL (fence timeout)");

  lite::DestroyDirectQueue(platform, queue, options);
  gpu.close();
  return ok ? 0 : 1;
}
