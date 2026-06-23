/*
 * wddm_lite.h - Userspace interface to WDDM display driver escape channel
 *
 * Wraps D3DKMTEscape for communicating with the amdgpu_wddm.sys driver.
 * Provides register access, BAR mapping, DMA allocation, and compute
 * escape operations.
 */

#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <vector>

/*
 * We don't include d3dkmthk.h directly because it requires full WDK
 * include setup. Instead we define the minimal D3DKMT types we need
 * and load the functions dynamically from gdi32.dll.
 */

typedef UINT D3DKMT_HANDLE;

typedef struct _D3DKMT_ADAPTERINFO {
    D3DKMT_HANDLE hAdapter;
    LUID          AdapterLuid;
    ULONG         NumOfSources;
    BOOL          bPrecisePresentRegionsPreferred;
} D3DKMT_ADAPTERINFO;

typedef struct _D3DKMT_ENUMADAPTERS2 {
    ULONG               NumAdapters;
    D3DKMT_ADAPTERINFO *pAdapters;
} D3DKMT_ENUMADAPTERS2;

typedef struct _D3DKMT_OPENADAPTERFROMLUID {
    LUID          AdapterLuid;
    D3DKMT_HANDLE hAdapter;
} D3DKMT_OPENADAPTERFROMLUID;

/*
 * D3DKMT_CREATEDEVICE on x64:
 *   offset 0: union { hAdapter(UINT); pAdapter(PVOID); } = 8 bytes
 *   offset 8: Flags (UINT) = 4 bytes
 *   offset 12: hDevice (UINT) = 4 bytes
 *   offset 16: pCommandBuffer (PVOID) = 8 bytes
 *   ... more fields follow
 */
typedef struct _D3DKMT_CREATEDEVICE {
    union {
        D3DKMT_HANDLE hAdapter;
        PVOID         pAdapter;
    };
    UINT          Flags;
    D3DKMT_HANDLE hDevice;
    PVOID         pCommandBuffer;
    UINT          CommandBufferSize;
    UINT          _pad1;
    PVOID         pAllocationList;
    UINT          AllocationListSize;
    UINT          _pad2;
    PVOID         pPatchLocationList;
    UINT          PatchLocationListSize;
    UINT          _pad3;
} D3DKMT_CREATEDEVICE;

typedef struct _D3DKMT_DESTROYDEVICE {
    D3DKMT_HANDLE hDevice;
} D3DKMT_DESTROYDEVICE;

typedef struct _D3DKMT_CLOSEADAPTER {
    D3DKMT_HANDLE hAdapter;
} D3DKMT_CLOSEADAPTER;

/* D3DKMT_ESCAPETYPE */
#define D3DKMT_ESCAPE_DRIVERPRIVATE 0

/*
 * D3DKMT_ESCAPE on x64:
 *   offset 0: hAdapter (UINT)
 *   offset 4: hDevice (UINT)
 *   offset 8: Type (D3DKMT_ESCAPETYPE = enum = UINT)
 *   offset 12: Flags (D3DDDI_ESCAPEFLAGS = union of UINT = 4 bytes)
 *   offset 16: pPrivateDriverData (D3DKMT_PTR = PVOID, 8 bytes)
 *   offset 24: PrivateDriverDataSize (UINT)
 *   offset 28: hContext (UINT)
 */
typedef struct _D3DKMT_ESCAPE {
    D3DKMT_HANDLE hAdapter;
    D3DKMT_HANDLE hDevice;
    UINT          Type;
    UINT          Flags;
    PVOID         pPrivateDriverData;
    UINT          PrivateDriverDataSize;
    D3DKMT_HANDLE hContext;
} D3DKMT_ESCAPE;

/* NTSTATUS for userspace */
#ifndef NTSTATUS
typedef LONG NTSTATUS;
#endif

/* Pull in the shared escape structures from the driver header.
 * We redefine them here for userspace since the driver header
 * has kernel-mode dependencies. */

typedef enum _AMDGPU_ESCAPE_CODE {
    AMDGPU_ESCAPE_GET_INFO          = 0x0001,
    AMDGPU_ESCAPE_READ_REG32        = 0x0010,
    AMDGPU_ESCAPE_WRITE_REG32       = 0x0011,
    AMDGPU_ESCAPE_MAP_BAR           = 0x0020,
    AMDGPU_ESCAPE_UNMAP_BAR         = 0x0021,
    AMDGPU_ESCAPE_ALLOC_DMA         = 0x0030,
    AMDGPU_ESCAPE_FREE_DMA          = 0x0031,
    AMDGPU_ESCAPE_MAP_VRAM          = 0x0040,
    AMDGPU_ESCAPE_READ_VRAM         = 0x0041,
    AMDGPU_ESCAPE_REGISTER_EVENT    = 0x0050,
    AMDGPU_ESCAPE_ENABLE_MSI        = 0x0051,
    AMDGPU_ESCAPE_GET_IOMMU_INFO    = 0x0060,

    AMDGPU_ESCAPE_ALLOC_MEMORY      = 0x0100,
    AMDGPU_ESCAPE_FREE_MEMORY       = 0x0101,
    AMDGPU_ESCAPE_MAP_MEMORY        = 0x0102,
    AMDGPU_ESCAPE_UNMAP_MEMORY      = 0x0103,
    AMDGPU_ESCAPE_CREATE_QUEUE      = 0x0110,
    AMDGPU_ESCAPE_DESTROY_QUEUE     = 0x0111,
    AMDGPU_ESCAPE_UPDATE_QUEUE      = 0x0112,
    AMDGPU_ESCAPE_CREATE_EVENT      = 0x0120,
    AMDGPU_ESCAPE_DESTROY_EVENT     = 0x0121,
    AMDGPU_ESCAPE_SET_EVENT         = 0x0122,
    AMDGPU_ESCAPE_RESET_EVENT       = 0x0123,
    AMDGPU_ESCAPE_WAIT_EVENTS       = 0x0124,
    AMDGPU_ESCAPE_GET_PROCESS_APERTURES = 0x0130,
    AMDGPU_ESCAPE_SET_MEMORY_POLICY = 0x0131,
    AMDGPU_ESCAPE_SET_SCRATCH_BACKING = 0x0132,
    AMDGPU_ESCAPE_SET_TRAP_HANDLER  = 0x0133,
    AMDGPU_ESCAPE_GET_CLOCK_COUNTERS = 0x0140,
    AMDGPU_ESCAPE_GET_VERSION       = 0x0150,
} AMDGPU_ESCAPE_CODE;

/* No #pragma pack - must match driver's default packing */

typedef struct _AMDGPU_ESCAPE_HEADER {
    AMDGPU_ESCAPE_CODE  Command;
    NTSTATUS            Status;
    ULONG               Size;
} AMDGPU_ESCAPE_HEADER;

typedef struct _AMDGPU_ESCAPE_GET_INFO_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    USHORT  VendorId;
    USHORT  DeviceId;
    USHORT  SubsystemVendorId;
    USHORT  SubsystemId;
    UCHAR   RevisionId;
    UCHAR   Reserved[3];
    ULONG   NumBars;
    struct {
        LARGE_INTEGER   PhysicalAddress;
        ULONGLONG       Length;
        BOOLEAN         IsMemory;
        BOOLEAN         Is64Bit;
        BOOLEAN         IsPrefetchable;
        UCHAR           Reserved;
    } Bars[6];
    ULONGLONG   VramSizeBytes;
    ULONGLONG   VisibleVramSizeBytes;
    ULONG       MmioBarIndex;
    ULONG       VramBarIndex;
    BOOLEAN     Headless;
    UCHAR       Reserved2[3];
} AMDGPU_ESCAPE_GET_INFO_DATA;

typedef struct _AMDGPU_ESCAPE_REG32_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG   BarIndex;
    ULONG   Offset;
    ULONG   Value;
} AMDGPU_ESCAPE_REG32_DATA;

typedef struct _AMDGPU_ESCAPE_MAP_BAR_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       BarIndex;
    ULONGLONG   Offset;
    ULONGLONG   Length;
    PVOID       MappedAddress;
    PVOID       MappingHandle;
} AMDGPU_ESCAPE_MAP_BAR_DATA;

typedef struct _AMDGPU_ESCAPE_ALLOC_DMA_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   Size;
    PVOID       CpuAddress;
    ULONGLONG   BusAddress;
    PVOID       AllocationHandle;
} AMDGPU_ESCAPE_ALLOC_DMA_DATA;

typedef struct _AMDGPU_ESCAPE_MAP_VRAM_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   Offset;
    ULONGLONG   Length;
    PVOID       MappedAddress;
    PVOID       MappingHandle;
} AMDGPU_ESCAPE_MAP_VRAM_DATA;

typedef struct _AMDGPU_ESCAPE_READ_VRAM_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   Offset;
    ULONGLONG   Length;
    UCHAR       Data[1];    /* Flexible array - actual size = header + Length */
} AMDGPU_ESCAPE_READ_VRAM_DATA;

/* Memory flags */
#define AMDGPU_MEM_TYPE_VRAM        0x0001
#define AMDGPU_MEM_TYPE_GTT         0x0002
#define AMDGPU_MEM_TYPE_SYSTEM      0x0004
#define AMDGPU_MEM_FLAG_USERPTR     0x0010
#define AMDGPU_MEM_FLAG_HOST_ACCESS 0x0020
#define AMDGPU_MEM_FLAG_NONPAGED    0x0040
#define AMDGPU_MEM_FLAG_UNCACHED    0x0400
#define AMDGPU_MEM_FLAG_CONTIGUOUS  0x0800

typedef struct _AMDGPU_ESCAPE_ALLOC_MEMORY_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;
    ULONGLONG   SizeInBytes;
    ULONGLONG   Alignment;
    ULONG       Flags;
    ULONGLONG   VaAddress;
    PVOID       CpuAddress;
    ULONGLONG   GpuAddress;
    ULONGLONG   Handle;
} AMDGPU_ESCAPE_ALLOC_MEMORY_DATA;

typedef struct _AMDGPU_ESCAPE_FREE_MEMORY_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   Handle;
} AMDGPU_ESCAPE_FREE_MEMORY_DATA;

typedef struct _AMDGPU_ESCAPE_CREATE_QUEUE_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;
    ULONG       QueueType;
    ULONG       QueuePercentage;
    LONG        Priority;
    ULONGLONG   QueueAddress;
    ULONGLONG   QueueSizeInBytes;
    ULONGLONG   WritePointerAddress;
    ULONGLONG   ReadPointerAddress;
    ULONGLONG   EopBufferAddress;
    ULONG       EopBufferSize;
    ULONGLONG   ContextSaveAddress;
    ULONG       ContextSaveSize;
    ULONG       SdmaEngineId;
    ULONGLONG   QueueId;
    ULONGLONG   DoorbellOffset;
} AMDGPU_ESCAPE_CREATE_QUEUE_DATA;

typedef struct _AMDGPU_ESCAPE_DESTROY_QUEUE_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   QueueId;
} AMDGPU_ESCAPE_DESTROY_QUEUE_DATA;

#define AMDGPU_EVENT_TYPE_SIGNAL    0
#define AMDGPU_EVENT_TYPE_QUEUE     7
#define AMDGPU_EVENT_TYPE_MEMORY    8

typedef struct _AMDGPU_ESCAPE_CREATE_EVENT_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       EventType;
    ULONG       GpuId;
    BOOLEAN     AutoReset;
    UCHAR       Reserved[3];
    ULONG       EventId;
    ULONGLONG   EventPageAddress;
    ULONG       EventSlotIndex;
} AMDGPU_ESCAPE_CREATE_EVENT_DATA;

typedef struct _AMDGPU_ESCAPE_DESTROY_EVENT_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       EventId;
} AMDGPU_ESCAPE_DESTROY_EVENT_DATA;

typedef struct _AMDGPU_ESCAPE_GET_VERSION_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       KfdMajorVersion;
    ULONG       KfdMinorVersion;
} AMDGPU_ESCAPE_GET_VERSION_DATA;

typedef struct _AMDGPU_ESCAPE_GET_CLOCK_COUNTERS_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;
    ULONGLONG   GpuClockCounter;
    ULONGLONG   CpuClockCounter;
    ULONGLONG   SystemClockCounter;
    ULONGLONG   SystemClockFrequencyHz;
    ULONGLONG   GpuClockFrequencyHz;
} AMDGPU_ESCAPE_GET_CLOCK_COUNTERS_DATA;

typedef struct _AMDGPU_ESCAPE_GET_PROCESS_APERTURES_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;
    ULONGLONG   LdsBase;
    ULONGLONG   LdsLimit;
    ULONGLONG   ScratchBase;
    ULONGLONG   ScratchLimit;
    ULONGLONG   GpuVmBase;
    ULONGLONG   GpuVmLimit;
} AMDGPU_ESCAPE_GET_PROCESS_APERTURES_DATA;

/* ENABLE_MSI: register the IH ring (allocated via ALLOC_DMA) with the kernel
 * driver ISR. IhRingDmaHandle is the ALLOC_DMA handle (the driver treats it as
 * the DmaAllocs[] index). Rptr/WptrRegOffset are MMIO byte offsets into the
 * MMIO BAR (i.e. (ihBase + regIH_RB_RPTR/WPTR) * 4). Mirrors the driver's
 * AMDGPU_ESCAPE_ENABLE_MSI_DATA (wddm_driver/amdgpu_wddm.h) byte-for-byte. */
typedef struct _AMDGPU_ESCAPE_ENABLE_MSI_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    PVOID       IhRingDmaHandle;
    ULONG       IhRingSize;
    ULONG       IhRptrRegOffset;
    ULONG       IhWptrRegOffset;
    BOOLEAN     Enabled;
    UCHAR       Reserved[3];
    ULONG       NumVectors;
} AMDGPU_ESCAPE_ENABLE_MSI_DATA;

/* ======================================================================
 * WddmLite - Userspace GPU access class
 * ====================================================================== */

class WddmLite {
public:
    WddmLite() : m_adapter(0), m_device(0), m_opened(false) {}
    ~WddmLite() { close(); }

    bool open();
    void close();
    bool isOpen() const { return m_opened; }

    /* Escape wrappers */
    bool getInfo(AMDGPU_ESCAPE_GET_INFO_DATA *info);
    bool readReg32(uint32_t offset, uint32_t *value, uint32_t barIndex = 0);
    bool writeReg32(uint32_t offset, uint32_t value, uint32_t barIndex = 0);
    bool mapVram(uint64_t offset, uint64_t length, void **addr, void **handle);
    bool unmapVram(void *addr, void *handle);
    bool mapBar(uint32_t barIndex, uint64_t offset, uint64_t length,
                void **addr, void **handle);
    bool readVram(uint64_t offset, uint64_t length, void *buffer);
    bool allocDma(uint64_t size, void **cpuAddr, uint64_t *busAddr, void **handle);
    bool freeDma(void *handle);

    /* Register an IH ring (from allocDma) with the kernel ISR. ihHandle is the
     * allocDma handle; rptr/wptrByteOffset are MMIO byte offsets. On success
     * *enabled / *numVectors report the driver's MSI state. */
    bool enableMsi(void *ihHandle, uint32_t ihRingSize,
                   uint32_t rptrByteOffset, uint32_t wptrByteOffset,
                   bool *enabled, uint32_t *numVectors);

    /* Compute escapes */
    bool allocMemory(uint64_t size, uint32_t flags, void **cpuAddr,
                     uint64_t *gpuAddr, uint64_t *handle);
    bool freeMemory(uint64_t handle);
    bool createQueue(uint32_t queueType, uint64_t ringAddr, uint64_t ringSize,
                     uint64_t *queueId, uint64_t *doorbellOffset);
    bool destroyQueue(uint64_t queueId);
    bool createEvent(uint32_t eventType, uint32_t *eventId,
                     uint64_t *eventPageAddr, uint32_t *slotIndex);
    bool destroyEvent(uint32_t eventId);
    bool getVersion(uint32_t *major, uint32_t *minor);
    bool getClockCounters(uint64_t *gpuClock, uint64_t *cpuClock);
    bool getProcessApertures(uint64_t *gpuVmBase, uint64_t *gpuVmLimit);

    D3DKMT_HANDLE adapter() const { return m_adapter; }
    D3DKMT_HANDLE device() const { return m_device; }

private:
    bool escape(void *data, uint32_t size);

    D3DKMT_HANDLE m_adapter;
    D3DKMT_HANDLE m_device;
    bool m_opened;
};

/* ======================================================================
 * IP Discovery structures
 * ====================================================================== */

/* Hardware IDs */
#define HWID_MP0    255     /* PSP */
#define HWID_MP1    1       /* SMU */
#define HWID_GC     11      /* Graphics/Compute */
#define HWID_MMHUB  34
#define HWID_SDMA0  42
#define HWID_OSSSYS 40      /* IH */
#define HWID_NBIF   108

#define IP_DISCOVERY_SIGNATURE  0x28211407

struct IpBlock {
    uint16_t hwId;
    uint8_t  instance;
    uint8_t  numBaseAddrs;
    uint8_t  majorVer;
    uint8_t  minorVer;
    uint8_t  revision;
    uint32_t baseAddrs[8];  /* Up to 8 base address registers */
};

struct IpDiscoveryResult {
    IpBlock blocks[64];
    uint32_t numBlocks;

    /* Resolved base addresses */
    uint32_t mmhubBase;
    uint32_t gcBase;        /* GC BASE_IDX=0 */
    uint32_t gcBase1;       /* GC BASE_IDX=1 */
    uint32_t mp0Base;       /* PSP */
    uint32_t mp1Base;       /* SMU */
    uint32_t sdma0Base;
    uint32_t ihBase;        /* OSSSYS/IH */

    bool valid;
};

bool ipDiscovery(WddmLite &gpu, const AMDGPU_ESCAPE_GET_INFO_DATA &info,
                 IpDiscoveryResult &result);

/* ======================================================================
 * GMC Init
 * ====================================================================== */

struct GmcState {
    uint64_t vramStart;     /* FB_LOCATION_BASE << 24 */
    uint64_t vramEnd;       /* FB_LOCATION_TOP << 24 */
    uint64_t vramSize;
    uint64_t gartStart;
    uint64_t gartEnd;
    uint64_t gartSize;
    bool mmhubConfigured;
    bool gfxhubConfigured;
};

bool gmcInit(WddmLite &gpu, const IpDiscoveryResult &ipd, GmcState &gmc);

/* ======================================================================
 * PSP / SMU / Firmware
 * ====================================================================== */

/* PSP firmware types */
#define PSP_FW_TYPE_TOC         4
#define PSP_FW_TYPE_SMU         12  /* SMC/SMU */
#define PSP_FW_TYPE_SDMA0       0
#define PSP_FW_TYPE_PFP         33  /* RS64_PFP */
#define PSP_FW_TYPE_ME          30  /* RS64_ME */
#define PSP_FW_TYPE_MEC         36  /* RS64_MEC */
#define PSP_FW_TYPE_IMU_I       47
#define PSP_FW_TYPE_IMU_D       48
#define PSP_FW_TYPE_RLC_G       7
#define PSP_FW_TYPE_RLC_AUTO    57  /* RLC_AUTOLOAD */

/* PSP ring commands */
#define GFX_CMD_ID_LOAD_IP_FW   0x00006
#define GFX_CMD_ID_AUTOLOAD_RLC 0x00017

/* PSP ring type */
#define PSP_RING_TYPE_KM        2

/* SMU messages */
#define PPSMC_MSG_DisallowGfxOff 0x29

struct PspState {
    /* Ring buffer */
    void    *ringCpuAddr;
    uint64_t ringBusAddr;
    void    *ringDmaHandle;
    uint32_t ringWptr;

    /* Firmware buffer */
    void    *fwBufCpuAddr;
    uint64_t fwBufBusAddr;
    void    *fwBufDmaHandle;

    /* Fence */
    volatile uint32_t *fenceCpuAddr;
    uint64_t fenceBusAddr;
    void    *fenceDmaHandle;
    uint32_t fenceValue;

    bool ringCreated;
    bool sosAlive;
};

bool pspCheckSos(WddmLite &gpu, const IpDiscoveryResult &ipd);
bool pspRingCreate(WddmLite &gpu, const IpDiscoveryResult &ipd, PspState &psp);
bool pspRingDestroy(WddmLite &gpu, const IpDiscoveryResult &ipd, PspState &psp);
bool pspLoadFirmware(WddmLite &gpu, const IpDiscoveryResult &ipd, PspState &psp,
                     uint32_t fwType, const void *fwData, uint32_t fwSize);
bool pspTriggerAutoload(WddmLite &gpu, const IpDiscoveryResult &ipd, PspState &psp);
bool smuDisableGfxOff(WddmLite &gpu, const IpDiscoveryResult &ipd);

/* Firmware file loading */
bool loadFirmwareFile(const char *path, std::vector<uint8_t> &data);

/* ======================================================================
 * macOS-proven gfx1201 cold-boot recipe (LITE_MES_RECIPE=1 equivalent)
 *
 * recipeBootload() faithfully mirrors the Python
 * load_all_firmware_recipe + init_smu(EnableAllSmuFeatures) + BOOTLOAD poll
 * from python/amd_gpu_driver/backends/windows/{psp_init,smu_init}.py.
 *
 * It uses VRAM-backed PSP buffers (alloc_memory equivalent) and the PSP
 * GPCOM command-buffer ABI (use_cmd_buffer=True), which differ from the
 * legacy DMA-based pspLoadFirmware/pspRingCreate path above. The legacy
 * path is left untouched.
 *
 * fwDir: path to the firmware .bin directory as seen by the guest, e.g.
 *        "Z:\\winfw".
 * vramMcBase: GMC VRAM MC base (GmcState.vramStart) used by the SMU driver
 *        table address translation, matching probe_kernel.py's
 *        init_smu(vram_mc_base=gmc.vram_start).
 *
 * Returns true if RLC_RLCS_BOOTLOAD_STATUS bit31 is set (0x8000003f).
 * ====================================================================== */
bool recipeBootload(WddmLite &gpu, const IpDiscoveryResult &ipd,
                    const char *fwDir, uint64_t vramMcBase);

/* ======================================================================
 * Increment 2a: MEC enable + direct compute HQD queue + NOP + RELEASE_MEM
 *
 * recipeNopFence() mirrors the proven Python probe (probe16_vmid0.py +
 * ring_init.py):
 *   1. recipeBootload() -> BOOTLOAD_COMPLETE (PSP autoload).
 *   2. init_gfx_for_compute equivalent: program CP PFP/ME/MEC program
 *      counters from the gc fw ucode_start, RLC/SH_MEM/doorbell-range,
 *      then enable the MEC (CP_MEC_RS64_CNTL -> 0x3C000000).
 *   3. init_compute_queue equivalent: alloc ring/MQD/EOP/rptr/wptr/fence
 *      in VRAM, build the v12 compute MQD, activate the HQD directly via
 *      CP_HQD_* MMIO under grbm_select(me=1,pipe=0,queue=0) at VMID 0.
 *   4. submit_compute_packets: NOP*4 + RELEASE_MEM fence to the ring,
 *      wptr writeback + MMIO wptr write + doorbell (BAR2).
 *   5. wait_fence: PASS iff the RELEASE_MEM fence value appears.
 *
 * fwDir/vramMcBase are passed straight through to recipeBootload(). This
 * is 2a (NOP+fence) only -- no GPUVM page tables, no shader dispatch.
 * ====================================================================== */
bool recipeNopFence(WddmLite &gpu, const IpDiscoveryResult &ipd,
                    const char *fwDir, uint64_t vramMcBase);

/* ======================================================================
 * Increment 2b: single s_endpgm compute dispatch (GPUVM + DISPATCH_DIRECT)
 *
 * recipeDispatch() mirrors the proven Python probe (probe16_vmid0.py +
 * compute_dispatch.py _build_dispatch_packets / test_noop_dispatch):
 *   1. recipeBootload() -> BOOTLOAD_COMPLETE (PSP autoload).
 *   2. init_gfx_for_compute (MEC enable), same as recipeNopFence.
 *   3. gfxhub_gart_enable: re-setup the GFXHUB after AUTOLOAD_RLC (which
 *      resets the GFXHUB contexts), using two DMA buffers for the GART
 *      page table + dummy page (system aperture / fault default).
 *   4. Stage 16 dwords of s_endpgm (0xBF810000) in a 4KB VRAM page.
 *   5. build_compute_gpuvm: build a 4-level GFXHUB page table (PDB2->PDB1->
 *      PDB0->PTB, 4KB VRAM pages, 0-based VRAM-offset entries) mapping the
 *      shader VA 0x200000000000 -> the s_endpgm page, then enable
 *      GCVM_CONTEXT0 = 0x03FFFC07 (ENABLE|depth3|fault-enable) and flush
 *      the GFXHUB TLB. CONTEXT0 is enabled AFTER the queue MEC bring-up and
 *      gfxhub_gart_enable, BEFORE init_compute_queue (probe16 ordering).
 *   6. init_compute_queue (direct-MMIO HQD, VMID 0), same as recipeNopFence.
 *   7. _build_dispatch_packets: ACQUIRE_MEM (gfx12 GCR full invalidate) +
 *      SET_SH_REG COMPUTE_PGM_LO/HI/RSRC1/RSRC2/RSRC3/TMPRING/RESTART/
 *      USER_DATA/RESOURCE_LIMITS/START..NUM_THREAD + DISPATCH_DIRECT
 *      (1,1,1, initiator=0x8045) + CS_PARTIAL_FLUSH + RELEASE_MEM fence.
 *      NO STATIC_THREAD_MGMT override (autoload masks are correct).
 *   8. wait_fence + read GCVM_L2_PROTECTION_FAULT_STATUS (gc base[0]+0x15D0)
 *      + RPTR. PASS iff the fence signals AND fault status == 0.
 *
 * fwDir/vramMcBase are passed straight through to recipeBootload(). This is
 * 2b (single s_endpgm dispatch) only -- no kernargs, multi-wg, or scratch.
 * ====================================================================== */
bool recipeDispatch(WddmLite &gpu, const IpDiscoveryResult &ipd,
                    const char *fwDir, uint64_t vramMcBase);

/* ======================================================================
 * Increment 3a: real compiled kernel + kernargs compute dispatch
 *
 * recipeKernargDispatch() mirrors the proven Python probe (probe_kernel.py +
 * compute_dispatch.py dispatch_elf_kernel):
 *   1. recipeBootload() -> BOOTLOAD_COMPLETE (PSP autoload).
 *   2. init_gfx_for_compute (MEC enable), same as recipeDispatch.
 *   3. gfxhub_gart_enable: re-init the GFXHUB after AUTOLOAD_RLC.
 *   4. Load fill_kernel_raw.co from fwDir; parse the AMDGPU ELF + the 64-byte
 *      KERNEL_DESCRIPTOR (RSRC1/2/3, kernarg_size, entry offset, props);
 *      assemble the loadable image by SECTION VADDR and stage it in VRAM.
 *   5. Allocate a kernarg buffer + an output buffer (FB-MC bump allocator);
 *      fill the kernarg with the output VA (off 0) + the u32 fill value (off 8).
 *   6. buildComputeGpuvmMulti: 4-level GFXHUB page table mapping EVERY code
 *      page + kernarg + output (one shared PTB; all VAs in the same 2MB block
 *      at 0x200000000000), enable GCVM_CONTEXT0 = depth-3, flush the TLB.
 *   7. init_compute_queue (direct-MMIO HQD, VMID 0).
 *   8. _build_dispatch_packets with the KD RSRC1/2/3 and the kernarg base VA
 *      in USER_DATA_<kernarg_sgpr_index> (slot 0 for this kernel); DISPATCH
 *      grid=1 block=64 (single workgroup -> COV5 hidden args irrelevant).
 *   9. wait_fence, HDP flush, read GCVM fault status + the output buffer.
 *      PASS iff the fence signals AND fault status == 0 AND out[0..63] == val.
 *
 * fwDir/vramMcBase are passed straight through to recipeBootload(). fwDir must
 * also contain fill_kernel_raw.co (e.g. copied into Z:\winfw).
 * ====================================================================== */
bool recipeKernargDispatch(WddmLite &gpu, const IpDiscoveryResult &ipd,
                           const char *fwDir, uint64_t vramMcBase);

/* ======================================================================
 * recipeMultiWgDispatch() extends recipeKernargDispatch to a MULTI-WORKGROUP
 * grid (increment 3b). It mirrors python/probe_kernel_mw.py:
 *   1.-7. identical bring-up to recipeKernargDispatch (bootload -> MEC enable
 *         -> GFXHUB re-init -> load fill_kernel_raw.co -> compute HQD).
 *   8. Parse the COV5 hidden-arg offsets from the .co NT_AMDGPU_METADATA note
 *      and fill them in the kernarg buffer: hidden_block_count_x = GRID (the
 *      number of WORKGROUPS), hidden_group_size_x = BLOCK (threads/wg = 64),
 *      remainder = 0, grid_dims = 1, plus the explicit args (out VA, fill).
 *   9. DISPATCH_DIRECT(GRID,1,1) with COMPUTE_NUM_THREAD_X = BLOCK, so the
 *      kernel sees tid = local_id + group_id*get_local_size(0) and every
 *      workgroup writes a distinct slice of the output.
 *  10. The output buffer is GRID*BLOCK*4 bytes spanning MULTIPLE 4KB pages;
 *      buildComputeGpuvmMulti maps EVERY output page (one shared PTB).
 *      PASS iff the fence signals AND fault status == 0 AND all GRID*BLOCK
 *      output dwords == the fill value.
 * ====================================================================== */
bool recipeMultiWgDispatch(WddmLite &gpu, const IpDiscoveryResult &ipd,
                           const char *fwDir, uint64_t vramMcBase);

/* ======================================================================
 * recipeScratchDispatch() runs a REGISTER-SPILLING kernel (increment 3c). It
 * mirrors python/probe_kernel_scratch.py:
 *   1.-7. identical bring-up to recipeMultiWgDispatch (bootload -> MEC enable
 *         -> GFXHUB re-init -> load scratch_kernel.co -> compute HQD).
 *   8. Confirm the kernel spills: KD.private_segment_fixed_size > 0 and
 *      COMPUTE_PGM_RSRC2.SCRATCH_EN (bit 0) == 1.
 *   9. Size the architected flat scratch from psfs (bpt = roundup(psfs,256/
 *      LANES); wave_bytes = bpt*LANES; WAVESIZE = roundup(wave_bytes,256)/256),
 *      allocate the VRAM scratch backing on the FB-MC bump allocator and map it
 *      into the SAME 4-level page table as code/kernarg/output.
 *  10. Program SH_MEM_CONFIG=0xC00C + SH_MEM_BASES=0x00010002 (VMID 0),
 *      COMPUTE_DISPATCH_SCRATCH_BASE = scratch_VA>>8, COMPUTE_TMPRING_SIZE =
 *      WAVES | (WAVESIZE<<12), and RSRC2 (SCRATCH_EN) from the KD; then
 *      DISPATCH_DIRECT(GRID,1,1) + EOP fence.
 *      PASS iff the fence signals AND fault status == 0 AND every GRID*BLOCK
 *      output dword == the spilled sum (val=1 -> 2080).
 * ====================================================================== */
bool recipeScratchDispatch(WddmLite &gpu, const IpDiscoveryResult &ipd,
                           const char *fwDir, uint64_t vramMcBase);

/* ======================================================================
 * ROCr lite:: integration surface (additive; foundation for the ROCr
 * WindowsLiteDriver direct-queue leg).
 *
 * These expose the proven bring-up + bump-allocator primitives that are
 * otherwise static in gpu_init.cpp, so the ROCr WindowsLiteDriver can drive
 * the GPU with the SAME recipe wddm_lite_test uses, then hand the queue
 * ring/MQD/doorbell to lite::CreateDirectQueue / lite::SubmitDirectQueue.
 *
 * The MMIO byte-offset convention is unchanged and matches the Linux
 * transport + gcReg/mmhubRead/pspRead in gpu_init.cpp exactly:
 *     byte_offset = (namespace_base + reg_dword) * 4   (always BAR0)
 * so the ROCr override ReadMmio32(base,reg) maps to
 *     gpu.readReg32((base + reg) * 4, value, 0)
 * and WriteMmio32 likewise. The doorbell is BAR2 at doorbell_index * 4.
 * ====================================================================== */

/* Resolved bring-up context the ROCr driver caches after wddmGfxBringUp().
 * gcBase0/gcBase1/mmhubBase/nbifBase2 are the IP-discovery namespace bases
 * (dword indices); vramMcBase is the GMC FB MC base (GmcState.vramStart),
 * which is ALSO the framebuffer_base lite:: adds to every queue offset. */
struct WddmComputeContext {
    uint32_t gcBase0;       /* ipd.gcBase  (base_idx 0) */
    uint32_t gcBase1;       /* ipd.gcBase1 (base_idx 1, == 0xA000 on gfx1201) */
    uint32_t mmhubBase;     /* ipd.mmhubBase */
    uint32_t nbifBase2;     /* NBIF base_idx 2 (doorbell aperture + HDP flush) */
    bool     hasNbif;
    uint64_t vramMcBase;    /* GMC FB MC base; == lite framebuffer_base */
    bool     mecEnabled;    /* CP_MEC_RS64_CNTL reached 0x3C000000 */
};

/* Full cold-boot + MEC enable + doorbell-aperture enable, exactly as
 * recipeNopFence does before it touches the queue:
 *   recipeBootload() -> BOOTLOAD_COMPLETE (PSP autoload),
 *   init_nbio doorbell-aperture + framebuffer enable,
 *   cqInitGfxForCompute() (CP counters, RLC/SH_MEM/doorbell-range, MEC enable).
 * Resets the VRAM bump allocator to vramMcBase so wddmAllocVram() hands out
 * FB-MC addresses the lite:: layout can use as framebuffer_base + offset.
 * Returns true and fills ctx on success (ctx.mecEnabled set iff
 * CP_MEC_RS64_CNTL == 0x3C000000). fwDir/vramMcBase mirror recipeNopFence. */
bool wddmGfxBringUp(WddmLite &gpu, const IpDiscoveryResult &ipd,
                    const char *fwDir, uint64_t vramMcBase,
                    WddmComputeContext &ctx);

/* Bump-allocate a VRAM buffer over the FB-MC aperture (exposes rcpAllocVram).
 * Returns cpu = CPU-mapped pointer, gpuAddr = vramMcBase + offset (the FB-MC
 * address the CP fetches from AND the lite:: framebuffer_base + queue offset),
 * handle = MAP_VRAM mapping handle. Must be called after wddmGfxBringUp()
 * (which seeds the allocator's vramMcBase). */
bool wddmAllocVram(WddmLite &gpu, uint64_t size, void **cpu,
                   uint64_t *gpuAddr, uint64_t *handle);

/* Re-run only the NBIO doorbell-aperture + framebuffer enable (the lite::
 * DirectQueuePlatform::EnsureDoorbellAperture analog). Safe to call repeatedly.
 * Returns false if the NBIF base could not be resolved. */
bool wddmEnsureDoorbellAperture(WddmLite &gpu, const IpDiscoveryResult &ipd);

/* DIAGNOSTIC (MES-on-Windows increment): start the MES engine after
 * wddmGfxBringUp(). recipeBootload loads the MES firmware (CP_MES/MES_STACK/
 * CP_MES_KIQ/MES_KIQ_STACK via the PSP autoload batch) but NEVER releases the
 * MES engine -- the proven direct-HQD path does not use MES. This mirrors the
 * Linux lite:: / windows ring_init.py _enable_mes_from_ucode sequence:
 *   RLC_CP_SCHEDULERS(KIQ routing) -> CP_MES_CNTL reset+halt -> per-pipe
 *   CP_MES_PRGRM_CNTR_START (entry from gc_<gc>_uni_mes.bin +56, >>2) ->
 *   CP_MES_CNTL release (clear reset/halt, set PIPE0/1_ACTIVE).
 * Because the PSP already loaded the MES ucode (mes_psp_loaded), the IC_BASE/
 * MDBASE VRAM-backdoor staging is intentionally skipped (matches the Python
 * "MES PSP-loaded -- skipping manual IC_BASE staging" branch).
 * Returns true if CP_MES_CNTL reads back with both PIPE0/1_ACTIVE set. Reads
 * CP_MES_HEADER_DUMP/INSTR_PNTR twice to report whether the engine executes.
 * fwDir mirrors wddmGfxBringUp (e.g. "Z:\\winfw"). Additive + safe to skip --
 * the no-arg/direct harness path never calls it. */
bool wddmStartMes(WddmLite &gpu, const IpDiscoveryResult &ipd,
                  const char *fwDir, const WddmComputeContext &ctx);

/* ======================================================================
 * MES-on-Windows increment 2: IH (interrupt handler) ring setup.
 *
 * wddmInitIh() ports python/.../windows/ih_init.py init_ih() into wddm_lite.
 * It allocates the IH ring + a WPTR-writeback dword in SYSTEM memory (allocDma,
 * GPU writes via PCIe bus address), resolves the IH/OSSSYS register base from
 * IP discovery (ipd.ihBase), and programs the IH v7.0 registers so the GPU
 * delivers 32-byte interrupt entries to the ring:
 *   IH_RB_BASE/HI    = ring_bus_addr >> 8 / >> 40
 *   IH_RB_CNTL       = MC_SPACE(bus) | RB_SIZE(log2(size/4)) | ENABLE_INTR
 *                      | WPTR_OVERFLOW_CLEAR/ENABLE | WPTR_WRITEBACK_ENABLE
 *                      | MC_SNOOP | RPTR_REARM   (mirrors ih_init.py _setup_ring)
 *   IH_RB_WPTR_ADDR_LO/HI = wptr writeback bus addr
 *   IH_RB_RPTR/WPTR  = 0
 *   IH_DOORBELL_RPTR = 0 (no doorbell-driven RPTR; ring is WPTR-writeback)
 * Then NBIO interrupt control (INTERRUPT_CNTL2 dummy page + INTERRUPT_CNTL
 * dummy-read-override/non-snoop) and the ENABLE_MSI escape to hand the ring to
 * the kernel ISR. The state lives in WddmIhState (fields filled on success).
 *
 * This is the DIAGNOSTIC MES KIQ-servicing probe: with the IH ring live before
 * the KIQ doorbell is rung, dumpMesDiag can observe whether IH_RB_WPTR advances
 * (doorbell -> interrupt -> IH path works) or stays 0 (no interrupt generated).
 *
 * Additive + only called on the harness "mes" path (after wddmStartMes, before
 * lite::CreateDirectQueue). Returns true if the ring was programmed (ENABLE_MSI
 * success is reported in WddmIhState.msiEnabled but NOT required for success,
 * since the MES may consume the ring on-GPU without the host ISR). */
struct WddmIhState {
    uint32_t ihBase;        /* OSSSYS/IH register base (dword); ipd.ihBase */

    void    *ringCpu;       /* IH ring CPU VA (system memory) */
    uint64_t ringBus;       /* IH ring PCIe bus address */
    void    *ringHandle;    /* allocDma handle */
    uint32_t ringSize;      /* bytes (power of two) */

    void    *wptrCpu;       /* WPTR-writeback CPU VA */
    uint64_t wptrBus;       /* WPTR-writeback bus address */
    void    *wptrHandle;    /* allocDma handle */

    uint32_t rbCntl;        /* programmed IH_RB_CNTL value */
    bool     msiEnabled;    /* ENABLE_MSI escape reported Enabled */
    uint32_t numVectors;    /* ENABLE_MSI vector count */
    bool     configured;    /* ring programmed (registers written) */
};

bool wddmInitIh(WddmLite &gpu, const IpDiscoveryResult &ipd,
                const WddmComputeContext &ctx, WddmIhState &ih);
