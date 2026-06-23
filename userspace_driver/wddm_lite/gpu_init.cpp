/*
 * gpu_init.cpp - GPU initialization: IP discovery, GMC, PSP, SMU
 *
 * Performs hardware bring-up via the WDDM escape channel:
 * 1. IP discovery - Parse IP blocks from VRAM discovery table
 * 2. GMC init - Configure MMHUB/GFXHUB memory controllers
 * 3. PSP GPCOM ring - Create ring, load firmware
 * 4. SMU - Disable GFXOFF
 */

#include "wddm_lite.h"
#include <cstring>
#include <cstdlib>

/* ======================================================================
 * IP Discovery
 * ====================================================================== */

/*
 * IP discovery table lives at VRAM_SIZE - 64KB.
 * Binary format: signature + header + per-die tables with IP blocks.
 */

bool ipDiscovery(WddmLite &gpu, const AMDGPU_ESCAPE_GET_INFO_DATA &info,
                 IpDiscoveryResult &result)
{
    memset(&result, 0, sizeof(result));

    uint64_t vramSize = info.VramSizeBytes;
    if (vramSize == 0) {
        /* Use visible VRAM as fallback */
        vramSize = info.VisibleVramSizeBytes;
    }
    if (vramSize == 0) {
        /* Estimate from BAR size */
        for (uint32_t i = 0; i < info.NumBars; i++) {
            if (info.Bars[i].IsMemory && info.Bars[i].Length > vramSize)
                vramSize = info.Bars[i].Length;
        }
    }

    printf("VRAM size: 0x%llX (%llu MB)\n", vramSize, vramSize / (1024*1024));

    /* Discovery table is at VRAM_SIZE - 64KB.
     * Map the last 64KB + some extra of VRAM. */
    uint64_t discoveryOffset = vramSize - 0x10000;
    uint64_t mapLen = 0x10000;  /* 64KB */

    /* Read VRAM through driver escape (avoids user-mode mapping issues) */
    auto *vramBuf = new uint8_t[mapLen];
    if (!gpu.readVram(discoveryOffset, mapLen, vramBuf)) {
        printf("ERROR: Failed to read VRAM at offset 0x%llX\n", discoveryOffset);
        delete[] vramBuf;
        return false;
    }

    printf("Read VRAM at offset 0x%llX (%llu bytes)\n", discoveryOffset, mapLen);

    const uint8_t *data = vramBuf;

    /* Check for PSP header (256 bytes) */
    uint32_t tableOffset = 0;
    uint32_t sig = *(const uint32_t *)data;
    if (sig != IP_DISCOVERY_SIGNATURE) {
        /* Try after PSP header */
        sig = *(const uint32_t *)(data + 256);
        if (sig == IP_DISCOVERY_SIGNATURE) {
            tableOffset = 256;
        } else {
            printf("ERROR: IP discovery signature not found (got 0x%08X)\n", sig);
            delete[] vramBuf;
            return false;
        }
    }

    printf("IP discovery signature found at offset %u\n", tableOffset);

    /* Binary header: signature + version + checksum + size + table_list[6] */
    const uint8_t *hdr = data + tableOffset;
    uint16_t verMajor = *(const uint16_t *)(hdr + 4);
    uint16_t verMinor = *(const uint16_t *)(hdr + 6);
    printf("IP discovery version: %u.%u\n", verMajor, verMinor);

    /* Table list starts at offset 12, 6 tables x 8 bytes each.
     * Table 0 = IP discovery, offset is relative to binary start. */
    uint16_t ipTableOffset = *(const uint16_t *)(hdr + 12);

    if (ipTableOffset == 0 || ipTableOffset >= mapLen) {
        printf("ERROR: Invalid IP table offset: %u\n", ipTableOffset);
        delete[] vramBuf;
        return false;
    }

    const uint8_t *ipTable = hdr + ipTableOffset;

    /* IP discovery table header (ip_discovery_header):
     *   offset  0: signature  (uint32_t) - should be "IPDS" = 0x53445049
     *   offset  4: version    (uint16_t)
     *   offset  6: size       (uint16_t) - total table size
     *   offset  8: id         (uint32_t)
     *   offset 12: num_dies   (uint16_t)
     *   offset 14: die entries start
     */
    uint32_t ipSig = *(const uint32_t *)(ipTable + 0);
    uint16_t ipVer = *(const uint16_t *)(ipTable + 4);
    uint16_t ipSize = *(const uint16_t *)(ipTable + 6);
    uint32_t ipId = *(const uint32_t *)(ipTable + 8);
    uint16_t numDies = *(const uint16_t *)(ipTable + 12);
    printf("IP table: sig=0x%08X ver=%u size=%u id=0x%X numDies=%u\n",
           ipSig, ipVer, ipSize, ipId, numDies);

    if (numDies == 0 || numDies > 16) {
        printf("ERROR: Invalid numDies: %u\n", numDies);
        delete[] vramBuf;
        return false;
    }

    /* After ip_discovery_header (14 bytes), there's a die_info[] array.
     * Each die_info is 4 bytes: die_id(uint16) + die_offset(uint16).
     * die_offset is relative to the binary header start (hdr). */
    for (uint16_t dieIdx = 0; dieIdx < numDies && dieIdx < 16; dieIdx++) {
        const uint8_t *dieInfo = ipTable + 14 + dieIdx * 4;
        uint16_t dieId = *(const uint16_t *)(dieInfo + 0);
        uint16_t dieOffset = *(const uint16_t *)(dieInfo + 2);

        printf("Die info[%u]: die_id=%u die_offset=0x%04X\n",
               dieIdx, dieId, dieOffset);

        /* Jump to actual die data (offset relative to binary header) */
        const uint8_t *dieData = hdr + dieOffset;
        uint16_t dDieId = *(const uint16_t *)(dieData + 0);
        uint16_t numIps = *(const uint16_t *)(dieData + 2);
        printf("Die %u: %u IP blocks\n", dDieId, numIps);

        if (numIps > 256) {
            printf("WARNING: Suspiciously large numIps: %u, capping at 128\n", numIps);
            numIps = 128;
        }

        /* Parse IP entries starting at dieData + 4 */
        const uint8_t *ipData = dieData + 4;

        for (uint16_t i = 0; i < numIps && result.numBlocks < 64; i++) {
            /* Bounds check */
            if (ipData < data || ipData + 8 > data + mapLen) {
                printf("WARNING: IP data out of bounds at entry %u\n", i);
                break;
            }

            IpBlock &blk = result.blocks[result.numBlocks];
            blk.hwId = *(const uint16_t *)(ipData + 0);
            blk.instance = ipData[2];
            blk.numBaseAddrs = ipData[3];
            blk.majorVer = ipData[4];
            blk.minorVer = ipData[5];
            blk.revision = ipData[6];

            /* Base addresses start at offset 8 */
            uint8_t nBases = blk.numBaseAddrs;
            if (nBases > 8) nBases = 8;
            for (uint8_t j = 0; j < nBases; j++) {
                blk.baseAddrs[j] = *(const uint32_t *)(ipData + 8 + j * 4);
            }

            /* Advance past this IP entry: 8 + numBaseAddrs * 4 bytes */
            ipData += 8 + blk.numBaseAddrs * 4;
            result.numBlocks++;
        }
    }

    printf("\nDiscovered %u IP blocks:\n", result.numBlocks);
    for (uint32_t i = 0; i < result.numBlocks; i++) {
        const IpBlock &b = result.blocks[i];
        printf("  [%2u] HW_ID=%3u inst=%u v%u.%u.%u bases=%u",
               i, b.hwId, b.instance, b.majorVer, b.minorVer, b.revision,
               b.numBaseAddrs);
        if (b.numBaseAddrs > 0)
            printf(" [0x%04X", b.baseAddrs[0]);
        for (uint8_t j = 1; j < b.numBaseAddrs && j < 4; j++)
            printf(", 0x%04X", b.baseAddrs[j]);
        if (b.numBaseAddrs > 0) printf("]");
        printf("\n");
    }

    /* Resolve well-known IP bases */
    for (uint32_t i = 0; i < result.numBlocks; i++) {
        const IpBlock &b = result.blocks[i];
        switch (b.hwId) {
        case HWID_MMHUB:
            if (b.instance == 0 && b.numBaseAddrs > 0)
                result.mmhubBase = b.baseAddrs[0];
            break;
        case HWID_GC:
            if (b.instance == 0) {
                if (b.numBaseAddrs > 0) result.gcBase = b.baseAddrs[0];
                if (b.numBaseAddrs > 1) result.gcBase1 = b.baseAddrs[1];
            }
            break;
        case HWID_MP0:
            if (b.instance == 0 && b.numBaseAddrs > 0)
                result.mp0Base = b.baseAddrs[0];
            break;
        case HWID_MP1:
            if (b.instance == 0 && b.numBaseAddrs > 0)
                result.mp1Base = b.baseAddrs[0];
            break;
        case HWID_SDMA0:
            if (b.instance == 0 && b.numBaseAddrs > 0)
                result.sdma0Base = b.baseAddrs[0];
            break;
        case HWID_OSSSYS:
            if (b.instance == 0 && b.numBaseAddrs > 0)
                result.ihBase = b.baseAddrs[0];
            break;
        }
    }

    printf("\nResolved bases:\n");
    printf("  MMHUB  = 0x%04X\n", result.mmhubBase);
    printf("  GC[0]  = 0x%04X\n", result.gcBase);
    printf("  GC[1]  = 0x%04X\n", result.gcBase1);
    printf("  MP0    = 0x%04X (PSP)\n", result.mp0Base);
    printf("  MP1    = 0x%04X (SMU)\n", result.mp1Base);
    printf("  SDMA0  = 0x%04X\n", result.sdma0Base);
    printf("  IH     = 0x%04X\n", result.ihBase);

    delete[] vramBuf;

    result.valid = (result.mmhubBase != 0 && result.mp0Base != 0);
    return result.valid;
}

/* ======================================================================
 * GMC Init (MMHUB)
 * ====================================================================== */

/* MMHUB v4.1.0 register offsets (DWORD offsets, multiply by 4 for byte offset) */
#define regMMMC_VM_FB_LOCATION_BASE             0x0554
#define regMMMC_VM_FB_LOCATION_TOP              0x0555
#define regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR     0x0559
#define regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR    0x055A
#define regMMMC_VM_MX_L1_TLB_CNTL              0x055B
#define regMMVM_CONTEXT0_CNTL                   0x0564
#define regMMVM_L2_CNTL                         0x04E4

/* Helper: read MMHUB register (base + dword_offset * 4) */
static bool mmhubRead(WddmLite &gpu, const IpDiscoveryResult &ipd,
                      uint32_t dwOffset, uint32_t *value)
{
    uint32_t byteOffset = (ipd.mmhubBase + dwOffset) * 4;
    return gpu.readReg32(byteOffset, value);
}

static bool mmhubWrite(WddmLite &gpu, const IpDiscoveryResult &ipd,
                       uint32_t dwOffset, uint32_t value)
{
    uint32_t byteOffset = (ipd.mmhubBase + dwOffset) * 4;
    return gpu.writeReg32(byteOffset, value);
}

bool gmcInit(WddmLite &gpu, const IpDiscoveryResult &ipd, GmcState &gmc)
{
    memset(&gmc, 0, sizeof(gmc));

    printf("\n=== GMC Init (MMHUB) ===\n");

    /* Read FB location to determine VRAM range.
     * Register defines are DWORD offsets relative to MMHUB base (BASE_IDX=0).
     * mmhubRead adds mmhubBase automatically. */
    uint32_t fbBase, fbTop;
    if (!mmhubRead(gpu, ipd, regMMMC_VM_FB_LOCATION_BASE, &fbBase) ||
        !mmhubRead(gpu, ipd, regMMMC_VM_FB_LOCATION_TOP, &fbTop)) {
        printf("ERROR: Cannot read FB_LOCATION registers\n");
        return false;
    }

    gmc.vramStart = (uint64_t)fbBase << 24;
    gmc.vramEnd = ((uint64_t)fbTop << 24) | 0xFFFFFF;
    gmc.vramSize = gmc.vramEnd - gmc.vramStart + 1;

    printf("FB_LOCATION_BASE = 0x%08X -> VRAM start 0x%llX\n", fbBase, gmc.vramStart);
    printf("FB_LOCATION_TOP  = 0x%08X -> VRAM end   0x%llX\n", fbTop, gmc.vramEnd);
    printf("VRAM size: %llu MB\n", gmc.vramSize / (1024*1024));

    /* GART sits at the end of VRAM range */
    gmc.gartSize = 512ULL * 1024 * 1024;  /* 512MB default */
    gmc.gartEnd = gmc.vramEnd;
    gmc.gartStart = gmc.gartEnd - gmc.gartSize + 1;

    printf("GART range: 0x%llX - 0x%llX (%llu MB)\n",
           gmc.gartStart, gmc.gartEnd, gmc.gartSize / (1024*1024));

    /* Read current MMHUB state */
    uint32_t ctx0Cntl, l1Cntl, l2Cntl;
    mmhubRead(gpu, ipd, regMMVM_CONTEXT0_CNTL, &ctx0Cntl);
    mmhubRead(gpu, ipd, regMMMC_VM_MX_L1_TLB_CNTL, &l1Cntl);
    mmhubRead(gpu, ipd, regMMVM_L2_CNTL, &l2Cntl);

    printf("Current MMHUB state:\n");
    printf("  CONTEXT0_CNTL = 0x%08X\n", ctx0Cntl);
    printf("  L1_TLB_CNTL   = 0x%08X\n", l1Cntl);
    printf("  L2_CNTL       = 0x%08X\n", l2Cntl);

    gmc.mmhubConfigured = (ctx0Cntl != 0);

    printf("MMHUB appears %s\n",
           gmc.mmhubConfigured ? "configured by VBIOS" : "unconfigured");

    return true;
}

/* ======================================================================
 * PSP - Sign of Life check
 * ====================================================================== */

/* PSP v14.0 register offsets (DWORD offsets from MP0 base) */
#define PSP_C2PMSG_64   0x0080  /* Ring command/status */
#define PSP_C2PMSG_67   0x0083  /* Ring write pointer */
#define PSP_C2PMSG_69   0x0085  /* Ring address low */
#define PSP_C2PMSG_70   0x0086  /* Ring address high */
#define PSP_C2PMSG_71   0x0087  /* Ring size */
#define PSP_C2PMSG_81   0x0091  /* SOS sign of life */
#define PSP_C2PMSG_35   0x0063  /* Bootloader mailbox */

static bool pspRead(WddmLite &gpu, const IpDiscoveryResult &ipd,
                    uint32_t dwOffset, uint32_t *value)
{
    uint32_t byteOffset = (ipd.mp0Base + dwOffset) * 4;
    return gpu.readReg32(byteOffset, value);
}

static bool pspWrite(WddmLite &gpu, const IpDiscoveryResult &ipd,
                     uint32_t dwOffset, uint32_t value)
{
    uint32_t byteOffset = (ipd.mp0Base + dwOffset) * 4;
    return gpu.writeReg32(byteOffset, value);
}

bool pspCheckSos(WddmLite &gpu, const IpDiscoveryResult &ipd)
{
    printf("\n=== PSP Sign of Life ===\n");
    printf("MP0 base = 0x%04X\n", ipd.mp0Base);

    /* Read bootloader status */
    uint32_t bootStatus = 0;
    pspRead(gpu, ipd, PSP_C2PMSG_35, &bootStatus);
    printf("C2PMSG_35 (bootloader) = 0x%08X%s\n", bootStatus,
           (bootStatus & 0x80000000) ? " [DONE]" : "");

    /* Read SOS sign of life */
    uint32_t sol = 0;
    pspRead(gpu, ipd, PSP_C2PMSG_81, &sol);
    printf("C2PMSG_81 (SOS sign of life) = 0x%08X\n", sol);

    /* Check ring status */
    uint32_t ringStatus = 0;
    pspRead(gpu, ipd, PSP_C2PMSG_64, &ringStatus);
    printf("C2PMSG_64 (ring status) = 0x%08X\n", ringStatus);

    bool tosReady = (ringStatus & 0x80000000) != 0;
    printf("TOS ready: %s\n", tosReady ? "YES" : "NO");

    if (sol == 0 && ringStatus == 0) {
        printf("WARNING: PSP GPCOM registers read 0\n");
        printf("  Bootloader done but ring not ready.\n");
        printf("  PSP GPCOM may require SMN indirect access on this GPU.\n");
        return false;
    }

    return true;
}

/* ======================================================================
 * PSP GPCOM Ring
 * ====================================================================== */

#define PSP_RING_SIZE   0x10000     /* 64KB */
#define PSP_FW_BUF_SIZE 0x100000   /* 1MB */
#define PSP_FENCE_SIZE  0x1000     /* 4KB */

bool pspRingCreate(WddmLite &gpu, const IpDiscoveryResult &ipd, PspState &psp)
{
    printf("\n=== PSP Ring Create ===\n");
    memset(&psp, 0, sizeof(psp));

    /* Allocate ring buffer via DMA */
    if (!gpu.allocDma(PSP_RING_SIZE, &psp.ringCpuAddr, &psp.ringBusAddr, &psp.ringDmaHandle)) {
        printf("ERROR: Failed to allocate PSP ring buffer\n");
        return false;
    }
    memset(psp.ringCpuAddr, 0, PSP_RING_SIZE);
    printf("Ring buffer: CPU=%p BUS=0x%llX\n", psp.ringCpuAddr, psp.ringBusAddr);

    /* Allocate firmware buffer */
    if (!gpu.allocDma(PSP_FW_BUF_SIZE, &psp.fwBufCpuAddr, &psp.fwBufBusAddr, &psp.fwBufDmaHandle)) {
        printf("ERROR: Failed to allocate firmware buffer\n");
        return false;
    }
    printf("FW buffer:   CPU=%p BUS=0x%llX\n", psp.fwBufCpuAddr, psp.fwBufBusAddr);

    /* Allocate fence buffer */
    void *fenceBuf = nullptr;
    if (!gpu.allocDma(PSP_FENCE_SIZE, &fenceBuf, &psp.fenceBusAddr, &psp.fenceDmaHandle)) {
        printf("ERROR: Failed to allocate fence buffer\n");
        return false;
    }
    psp.fenceCpuAddr = (volatile uint32_t *)fenceBuf;
    *psp.fenceCpuAddr = 0;
    psp.fenceValue = 0;
    printf("Fence:       CPU=%p BUS=0x%llX\n", fenceBuf, psp.fenceBusAddr);

    /* Wait for TOS ready */
    uint32_t status = 0;
    for (int retry = 0; retry < 100; retry++) {
        pspRead(gpu, ipd, PSP_C2PMSG_64, &status);
        if (status & 0x80000000) break;
        Sleep(10);
    }
    if (!(status & 0x80000000)) {
        printf("ERROR: TOS not ready (C2PMSG_64 = 0x%08X)\n", status);
        return false;
    }
    printf("TOS ready (status = 0x%08X)\n", status);

    /* Destroy any existing ring first (cmd = 3 << 16 = 0x30000) */
    printf("Destroying existing ring...\n");
    pspWrite(gpu, ipd, PSP_C2PMSG_64, 0x30000);
    Sleep(50);

    /* Wait for response */
    for (int retry = 0; retry < 100; retry++) {
        pspRead(gpu, ipd, PSP_C2PMSG_64, &status);
        if (status & 0x80000000) break;
        Sleep(10);
    }
    printf("After destroy: C2PMSG_64 = 0x%08X\n", status);

    /* Create new ring (cmd = PSP_RING_TYPE_KM << 16 = 0x20000) */
    uint32_t addrLo = (uint32_t)(psp.ringBusAddr & 0xFFFFFFFF);
    uint32_t addrHi = (uint32_t)(psp.ringBusAddr >> 32);

    pspWrite(gpu, ipd, PSP_C2PMSG_69, addrLo);
    pspWrite(gpu, ipd, PSP_C2PMSG_70, addrHi);
    pspWrite(gpu, ipd, PSP_C2PMSG_71, PSP_RING_SIZE);
    pspWrite(gpu, ipd, PSP_C2PMSG_64, PSP_RING_TYPE_KM << 16);

    Sleep(50);

    /* Wait for response */
    for (int retry = 0; retry < 100; retry++) {
        pspRead(gpu, ipd, PSP_C2PMSG_64, &status);
        if (status & 0x80000000) break;
        Sleep(10);
    }

    uint16_t ringStatus = status & 0xFFFF;
    printf("Ring create response: 0x%08X (status=%u)\n", status, ringStatus);

    if (!(status & 0x80000000) || ringStatus != 0) {
        printf("ERROR: Ring create failed\n");
        return false;
    }

    psp.ringWptr = 0;
    psp.ringCreated = true;
    printf("PSP GPCOM ring created successfully\n");
    return true;
}

bool pspRingDestroy(WddmLite &gpu, const IpDiscoveryResult &ipd, PspState &psp)
{
    if (!psp.ringCreated) return true;

    printf("\n=== PSP Ring Destroy ===\n");
    pspWrite(gpu, ipd, PSP_C2PMSG_64, 0x30000);
    Sleep(50);

    uint32_t status = 0;
    for (int retry = 0; retry < 100; retry++) {
        pspRead(gpu, ipd, PSP_C2PMSG_64, &status);
        if (status & 0x80000000) break;
        Sleep(10);
    }
    printf("Destroy response: 0x%08X\n", status);

    /* Free DMA buffers */
    if (psp.ringDmaHandle) gpu.freeDma(psp.ringDmaHandle);
    if (psp.fwBufDmaHandle) gpu.freeDma(psp.fwBufDmaHandle);
    if (psp.fenceDmaHandle) gpu.freeDma(psp.fenceDmaHandle);

    memset(&psp, 0, sizeof(psp));
    return true;
}

/* ======================================================================
 * PSP Firmware Loading
 * ====================================================================== */

/*
 * PSP ring entry format (16 bytes = 4 DWORDs):
 *   DW[0] = fence_addr_lo
 *   DW[1] = fence_addr_hi
 *   DW[2] = fence_value
 *   DW[3] = cmd_id | (fw_type << 16)
 */

static bool pspRingSubmit(WddmLite &gpu, const IpDiscoveryResult &ipd,
                          PspState &psp, uint32_t cmdId, uint32_t fwType)
{
    uint32_t entryOffset = (psp.ringWptr * 16) % PSP_RING_SIZE;
    uint32_t *entry = (uint32_t *)((uint8_t *)psp.ringCpuAddr + entryOffset);

    psp.fenceValue++;

    entry[0] = (uint32_t)(psp.fenceBusAddr & 0xFFFFFFFF);
    entry[1] = (uint32_t)(psp.fenceBusAddr >> 32);
    entry[2] = psp.fenceValue;
    entry[3] = cmdId | (fwType << 16);

    /* Memory barrier */
    MemoryBarrier();

    psp.ringWptr++;
    pspWrite(gpu, ipd, PSP_C2PMSG_67, psp.ringWptr);

    /* Wait for fence */
    for (int retry = 0; retry < 500; retry++) {
        if (*psp.fenceCpuAddr >= psp.fenceValue)
            return true;
        Sleep(10);
    }

    printf("ERROR: PSP ring submission timed out (fence=%u, expected=%u)\n",
           *psp.fenceCpuAddr, psp.fenceValue);
    return false;
}

bool pspLoadFirmware(WddmLite &gpu, const IpDiscoveryResult &ipd, PspState &psp,
                     uint32_t fwType, const void *fwData, uint32_t fwSize)
{
    if (!psp.ringCreated) {
        printf("ERROR: PSP ring not created\n");
        return false;
    }

    if (fwSize > PSP_FW_BUF_SIZE) {
        printf("ERROR: Firmware too large (%u > %u)\n", fwSize, PSP_FW_BUF_SIZE);
        return false;
    }

    /* Copy firmware data to DMA buffer */
    memcpy(psp.fwBufCpuAddr, fwData, fwSize);
    MemoryBarrier();

    /* Write firmware buffer address to PSP bootloader mailbox
     * Address must be 1MB aligned, shifted >> 20 */
    uint32_t fwAddrShifted = (uint32_t)((psp.fwBufBusAddr >> 20) & 0xFFFFFFFF);
    pspWrite(gpu, ipd, PSP_C2PMSG_35 + 1, fwAddrShifted);

    /* Submit load command via ring */
    printf("  Loading FW type %u (%u bytes) at bus 0x%llX...",
           fwType, fwSize, psp.fwBufBusAddr);

    if (!pspRingSubmit(gpu, ipd, psp, GFX_CMD_ID_LOAD_IP_FW, fwType)) {
        printf(" FAILED\n");

        /* Check C2PMSG_64 for error details */
        uint32_t ringStatus = 0;
        pspRead(gpu, ipd, PSP_C2PMSG_64, &ringStatus);
        printf("  Ring status: 0x%08X\n", ringStatus);
        return false;
    }

    printf(" OK\n");
    return true;
}

bool pspTriggerAutoload(WddmLite &gpu, const IpDiscoveryResult &ipd, PspState &psp)
{
    printf("  Triggering RLC autoload...");
    if (!pspRingSubmit(gpu, ipd, psp, GFX_CMD_ID_AUTOLOAD_RLC, 0)) {
        printf(" FAILED\n");
        return false;
    }
    printf(" OK\n");
    return true;
}

/* ======================================================================
 * SMU Messaging
 * ====================================================================== */

/* SMU v11 mailbox offsets from MP1 base (DWORD offsets) */
#define SMU_MSG_REG     0x0282
#define SMU_PARAM_REG   0x0292
#define SMU_RESP_REG    0x029A

static bool smuRead(WddmLite &gpu, const IpDiscoveryResult &ipd,
                    uint32_t dwOffset, uint32_t *value)
{
    uint32_t byteOffset = (ipd.mp1Base + dwOffset) * 4;
    return gpu.readReg32(byteOffset, value);
}

static bool smuWrite(WddmLite &gpu, const IpDiscoveryResult &ipd,
                     uint32_t dwOffset, uint32_t value)
{
    uint32_t byteOffset = (ipd.mp1Base + dwOffset) * 4;
    return gpu.writeReg32(byteOffset, value);
}

bool smuDisableGfxOff(WddmLite &gpu, const IpDiscoveryResult &ipd)
{
    printf("\n=== SMU: Disable GFXOFF ===\n");

    /* Clear response register */
    smuWrite(gpu, ipd, SMU_RESP_REG, 0);

    /* Write parameter (0 = no param for DisallowGfxOff) */
    smuWrite(gpu, ipd, SMU_PARAM_REG, 0);

    /* Send message */
    smuWrite(gpu, ipd, SMU_MSG_REG, PPSMC_MSG_DisallowGfxOff);

    /* Wait for response */
    uint32_t resp = 0;
    for (int retry = 0; retry < 100; retry++) {
        smuRead(gpu, ipd, SMU_RESP_REG, &resp);
        if (resp != 0) break;
        Sleep(10);
    }

    printf("SMU response: 0x%08X (0 = pending, 1 = success)\n", resp);

    if (resp != 1) {
        printf("WARNING: SMU DisallowGfxOff may have failed\n");
        return false;
    }

    printf("GFXOFF disabled successfully\n");
    return true;
}

/* ======================================================================
 * Firmware file loading utility
 * ====================================================================== */

bool loadFirmwareFile(const char *path, std::vector<uint8_t> &data)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("ERROR: Cannot open firmware file: %s\n", path);
        return false;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0 || size > 16 * 1024 * 1024) {
        printf("ERROR: Invalid firmware size: %ld\n", size);
        fclose(f);
        return false;
    }

    data.resize(size);
    size_t read = fread(data.data(), 1, size, f);
    fclose(f);

    if ((long)read != size) {
        printf("ERROR: Short read: %zu/%ld\n", read, size);
        return false;
    }

    printf("Loaded %s (%ld bytes)\n", path, size);
    return true;
}

/* ======================================================================
 * macOS-proven gfx1201 cold-boot recipe (LITE_MES_RECIPE=1 equivalent)
 *
 * Faithful C++ transcription of:
 *   python/amd_gpu_driver/backends/windows/psp_init.py::load_all_firmware_recipe
 *   python/amd_gpu_driver/backends/windows/smu_init.py::init_smu (+EnableAllSmuFeatures)
 *   probe_kernel.py BOOTLOAD poll: read_reg32((0xA000 + 0x4e7c) * 4)
 *
 * Differences from the legacy pspLoadFirmware path above (all intentional):
 *   1. PSP-visible buffers are VRAM allocations (alloc_memory), not allocDma.
 *      The "bus address" written to PSP is the VRAM MC GPU address.
 *   2. LOAD_IP_FW / LOAD_TOC use the GPCOM command-buffer ABI
 *      (use_cmd_buffer=True): cmd[7..10]=addr_lo/hi/size/type, with a
 *      64-byte ring frame pointing at the cmd buffer + fence.
 *   3. TOC comes from the SOS container (PSP_TOC=4), not gc_*_toc.bin.
 *   4. RS64 ucode offsets read the v2 gfx-header fields at +36 directly;
 *      RLC_G is loaded LAST; AUTOLOAD_RLC uses cmd id 0x21.
 *   5. SMU EnableAllSmuFeatures(param=0) after firmware load is what
 *      completes the autoload.
 * ====================================================================== */

/* PSP GPCOM command-buffer ABI constants (psp_init.py). */
#define RCP_GFX_CMD_RESP_SIZE     1024
#define RCP_GFX_CMD_BUF_VERSION   1
#define RCP_PSP_RING_SIZE         0x10000   /* 64KB */
#define RCP_PSP_RING_FRAME_SIZE   64        /* bytes */
#define RCP_PSP_FENCE_SIZE        4096

/* PSP ring commands (psp_init.py - note AUTOLOAD differs from legacy 0x17). */
#define RCP_GFX_CMD_ID_LOAD_IP_FW   0x00006
#define RCP_GFX_CMD_ID_LOAD_TOC     0x00020
#define RCP_GFX_CMD_ID_AUTOLOAD_RLC 0x00021

/* PSP ring type. */
#define RCP_PSP_RING_TYPE_KM        2

/* Response flags. */
#define RCP_GFX_FLAG_RESPONSE       0x80000000
#define RCP_GFX_CMD_RESPONSE_MASK   0x8000FFFF

/* PSP MP0 C2PMSG DWORD offsets (psp_init.py). */
#define RCP_C2PMSG_64   0x0080   /* Ring command/status */
#define RCP_C2PMSG_67   0x0083   /* Ring write pointer */
#define RCP_C2PMSG_69   0x0085   /* Ring address low */
#define RCP_C2PMSG_70   0x0086   /* Ring address high */
#define RCP_C2PMSG_71   0x0087   /* Ring size */
#define RCP_C2PMSG_81   0x0091   /* SOS sign of life */

/* PSPFWType for the SOS container parse (psp_init.py PSPFWType). */
#define RCP_PSP_FW_TYPE_SOS         1
#define RCP_PSP_FW_TYPE_SYS_DRV     2
#define RCP_PSP_FW_TYPE_KDB         3
#define RCP_PSP_FW_TYPE_TOC         4
#define RCP_PSP_FW_TYPE_SPL         5
#define RCP_PSP_FW_TYPE_SOC_DRV     7
#define RCP_PSP_FW_TYPE_INTF_DRV    8
#define RCP_PSP_FW_TYPE_DBG_DRV     9
#define RCP_PSP_FW_TYPE_RAS_DRV     10
#define RCP_PSP_FW_TYPE_IPKEYMGR_DRV 11

/* MPASP bootloader mailbox registers (regMPASP_SMN_C2PMSG_*, psp_init.py).
 * Same MP0 base + DWORD-offset convention as the legacy pspCheckSos path:
 *   byte_offset = (mp0Base + dw) * 4. C2PMSG_81 is RCP_C2PMSG_81 above. */
#define RCP_C2PMSG_35   0x0063   /* Bootloader command/status (bit31 = ready) */
#define RCP_C2PMSG_36   0x0064   /* Bootloader firmware address (>> 20) */

/* PSP bootloader command IDs (PSP_BL__LOAD_*, psp_init.py). */
#define RCP_PSP_BL_LOAD_KEY_DATABASE   0x80000
#define RCP_PSP_BL_LOAD_TOS_SPL_TABLE  0x10000000
#define RCP_PSP_BL_LOAD_SYSDRV         0x10000
#define RCP_PSP_BL_LOAD_SOCDRV         0xB0000
#define RCP_PSP_BL_LOAD_INTFDRV        0xD0000
#define RCP_PSP_BL_LOAD_HADDRV         0xC0000   /* Debug/HAD driver */
#define RCP_PSP_BL_LOAD_RASDRV         0xE0000
#define RCP_PSP_BL_LOAD_IPKEYMGRDRV    0xF0000
#define RCP_PSP_BL_LOAD_SOSDRV         0x20000

/* Bootloader-ready bit (bit 31 of C2PMSG_35). */
#define RCP_PSP_BL_READY               0x80000000

/* GFXFWType IDs for GFX_CMD_ID_LOAD_IP_FW (psp_init.py GFXFWType). */
#define RCP_FW_RLC_G                       8
#define RCP_FW_SMU                         18
#define RCP_FW_RLC_RESTORE_LIST_GPM_MEM    20
#define RCP_FW_RLC_RESTORE_LIST_SRM_MEM    21
#define RCP_FW_RLC_IRAM                    26
#define RCP_FW_CP_MES                      33
#define RCP_FW_MES_STACK                   34
#define RCP_FW_RLC_DRAM_BOOT               48
#define RCP_FW_IMU_I                       68
#define RCP_FW_IMU_D                       69
#define RCP_FW_SDMA_UCODE_TH0              71
#define RCP_FW_CP_MES_KIQ                  81
#define RCP_FW_MES_KIQ_STACK               82
#define RCP_FW_RS64_PFP                    87
#define RCP_FW_RS64_ME                     88
#define RCP_FW_RS64_MEC                    89
#define RCP_FW_RS64_PFP_P0_STACK           90
#define RCP_FW_RS64_PFP_P1_STACK           91
#define RCP_FW_RS64_ME_P0_STACK            92
#define RCP_FW_RS64_ME_P1_STACK            93
#define RCP_FW_RS64_MEC_P0_STACK           94
#define RCP_FW_RS64_MEC_P1_STACK           95
#define RCP_FW_RS64_MEC_P2_STACK           96
#define RCP_FW_RS64_MEC_P3_STACK           97

/* SMU MP1 C2PMSG DWORD offsets (smu_init.py). */
#define RCP_MP1_C2PMSG_66   0x0282   /* message */
#define RCP_MP1_C2PMSG_82   0x0292   /* parameter / readback */
#define RCP_MP1_C2PMSG_90   0x029A   /* response */

/* SMU14_0_2 message IDs (smu_init.py SMU14_0_2_MESSAGES; MP1 14.0.3 uses it). */
#define RCP_SMU_GET_VERSION              0x02
#define RCP_SMU_SET_DRIVER_DRAM_HIGH     0x0E
#define RCP_SMU_SET_DRIVER_DRAM_LOW      0x0F
#define RCP_SMU_ENABLE_ALL_FEATURES      0x06
#define RCP_SMU_GET_ENABLED_FEATURES_LOW  0x0C
#define RCP_SMU_GET_ENABLED_FEATURES_HIGH 0x0D
#define RCP_SMU_DISALLOW_GFXOFF          0x29
#define RCP_PPSMC_RESULT_OK              0x1

#define RCP_SMU_DRIVER_TABLE_SIZE        0x4000

/* BOOTLOAD status register: read_reg32((0xA000 + 0x4e7c) * 4). */
#define RCP_BOOTLOAD_DWORD   (0xA000 + 0x4e7c)
#define RCP_BOOTLOAD_OK      0x8000003fu

/* Recipe-local PSP state. Uses VRAM (alloc_memory) buffers so the PSP-visible
 * addresses match the proven probe path (NOT the legacy DMA bus addresses). */
struct RecipePspState {
    void    *ringCpu;
    uint64_t ringGpu;       /* VRAM MC address */
    uint64_t ringHandle;

    void    *cmdCpu;
    uint64_t cmdGpu;
    uint64_t cmdHandle;

    volatile uint32_t *fenceCpu;
    uint64_t fenceGpu;
    uint64_t fenceHandle;
    uint32_t fenceValue;

    uint32_t ringWptr;
};

/* Little-endian dword/halfword reads from a firmware blob (struct.unpack_from). */
static uint32_t rcpU32(const std::vector<uint8_t> &d, size_t off)
{
    if (off + 4 > d.size()) return 0;
    return (uint32_t)d[off] | ((uint32_t)d[off + 1] << 8) |
           ((uint32_t)d[off + 2] << 16) | ((uint32_t)d[off + 3] << 24);
}

static uint16_t rcpU16(const std::vector<uint8_t> &d, size_t off)
{
    if (off + 2 > d.size()) return 0;
    return (uint16_t)((uint32_t)d[off] | ((uint32_t)d[off + 1] << 8));
}

/* parse_firmware_header: common_firmware_header (psp_init.py).
 * Only the two fields the recipe needs are surfaced here. */
struct RcpFwHeader {
    uint32_t ucodeSizeBytes;        /* offset 28 */
    uint32_t ucodeArrayOffsetBytes; /* offset 32 */
};

static RcpFwHeader rcpParseHeader(const std::vector<uint8_t> &d)
{
    RcpFwHeader h;
    /* "<IIHHHHIIII" -> size(0) hsize(4) hvMaj(8) hvMin(10) ipMaj(12) ipMin(14)
     * ucodeVer(16) ucodeSize(20)... wait: fields per parse_firmware_header:
     *   [0]size_bytes I @0
     *   [1]header_size_bytes I @4
     *   [2]header_version_major H @8
     *   [3]header_version_minor H @10
     *   [4]ip_version_major H @12
     *   [5]ip_version_minor H @14
     *   [6]ucode_version I @16
     *   [7]ucode_size_bytes I @20
     *   [8]ucode_array_offset_bytes I @24
     *   [9]crc32 I @28
     */
    h.ucodeSizeBytes = rcpU32(d, 20);
    h.ucodeArrayOffsetBytes = rcpU32(d, 24);
    return h;
}

/* _slice equivalent: bounds-checked copy out of a blob. */
static bool rcpSlice(const std::vector<uint8_t> &d, size_t off, size_t size,
                     std::vector<uint8_t> &out)
{
    if (size == 0) { out.clear(); return true; }
    if (off + size > d.size()) {
        printf("  PSP[recipe]: ERROR slice out of range off=0x%zX size=0x%zX blob=0x%zX\n",
               off, size, d.size());
        return false;
    }
    out.assign(d.begin() + off, d.begin() + off + size);
    return true;
}

/* VRAM bump allocator state (mirrors Python backends/windows/device.py
 * _vram_cursor + _VRAM_MC_BASE). recipeBootload initializes both once; the
 * cursor persists across the bootload -> dispatch sequence so that every VRAM
 * allocation gets a unique FB-aperture MC address (g_vramMcBase + offset). */
static uint64_t g_vramCursor = 0;
static uint64_t g_vramMcBase = 0;

/* Allocate a PSP-visible VRAM buffer by bump-allocating over the VRAM BAR and
 * mapping it via MAP_VRAM (mirrors Python alloc_memory). The returned gpuAddr
 * is an FB MC address (g_vramMcBase + offset), which the CP accesses directly
 * and which the page table encodes as offset = gpuAddr - g_vramMcBase. The
 * returned handle is the MAP_VRAM mapping handle. */
static bool rcpAllocVram(WddmLite &gpu, uint64_t size, void **cpu,
                         uint64_t *gpuAddr, uint64_t *handle)
{
    size = (size + 4095) & ~4095ull;
    uint64_t offset = g_vramCursor;
    g_vramCursor += size;

    void *addr = nullptr, *mh = nullptr;
    if (!gpu.mapVram(offset, size, &addr, &mh)) {
        printf("  PSP[recipe]: ERROR VRAM alloc failed (size=0x%llX)\n",
               (unsigned long long)size);
        return false;
    }
    if (addr == nullptr) {
        printf("  PSP[recipe]: ERROR VRAM alloc not CPU mapped\n");
        return false;
    }
    memset(addr, 0, (size_t)size);
    if (cpu) *cpu = addr;
    if (gpuAddr) *gpuAddr = g_vramMcBase + offset;
    if (handle) *handle = (uint64_t)mh;
    return true;
}

/* Allocate a PSP-visible VRAM buffer aligned to `alignment` by rounding the
 * bump-allocator offset up to `alignment` before mapping (mirrors Python
 * alloc_memory + the explicit-alignment _alloc_psp_buffer path). Used for the
 * bootloader fw staging buffer, which is written to C2PMSG_36 as (addr >> 20)
 * and so must be 1MB-aligned. Because g_vramMcBase is 1MB-aligned (0x80..),
 * aligning the offset also aligns the returned MC address. */
static bool rcpAllocVramAligned(WddmLite &gpu, uint64_t size, uint64_t alignment,
                                void **cpu, uint64_t *gpuAddr, uint64_t *handle)
{
    size = (size + 4095) & ~4095ull;
    uint64_t offset = (g_vramCursor + (alignment - 1)) & ~(alignment - 1);
    g_vramCursor = offset + size;

    void *addr = nullptr, *mh = nullptr;
    if (!gpu.mapVram(offset, size, &addr, &mh)) {
        printf("  PSP[recipe]: ERROR aligned VRAM alloc failed (size=0x%llX)\n",
               (unsigned long long)size);
        return false;
    }
    if (addr == nullptr) {
        printf("  PSP[recipe]: ERROR aligned VRAM alloc not CPU mapped\n");
        return false;
    }
    memset(addr, 0, (size_t)size);
    if (cpu) *cpu = addr;
    if (gpuAddr) *gpuAddr = g_vramMcBase + offset;
    if (handle) *handle = (uint64_t)mh;
    return true;
}

/* PSP MP0 register access (DWORD offset from mp0Base). */
static uint32_t rcpPspRead(WddmLite &gpu, const IpDiscoveryResult &ipd, uint32_t dw)
{
    uint32_t v = 0;
    gpu.readReg32((ipd.mp0Base + dw) * 4, &v);
    return v;
}

static void rcpPspWrite(WddmLite &gpu, const IpDiscoveryResult &ipd,
                        uint32_t dw, uint32_t v)
{
    gpu.writeReg32((ipd.mp0Base + dw) * 4, v);
}

/* _wait_for_reg: poll until (val & mask) == expected. */
static bool rcpWaitReg(WddmLite &gpu, const IpDiscoveryResult &ipd, uint32_t dw,
                       uint32_t expected, uint32_t mask, int timeoutMs)
{
    int waited = 0;
    while (waited < timeoutMs) {
        if ((rcpPspRead(gpu, ipd, dw) & mask) == expected)
            return true;
        Sleep(1);
        waited += 1;
    }
    return false;
}

/* create_psp_ring (psp_init.py) using the recipe VRAM ring buffer. */
static bool rcpCreateRing(WddmLite &gpu, const IpDiscoveryResult &ipd,
                          RecipePspState &st)
{
    /* Destroy any stale ring left by an earlier process. */
    if (rcpPspRead(gpu, ipd, RCP_C2PMSG_71) != 0) {
        uint32_t status = rcpPspRead(gpu, ipd, RCP_C2PMSG_64);
        if ((status & RCP_GFX_FLAG_RESPONSE) == 0) {
            printf("  PSP[recipe]: ERROR stale busy ring (C2PMSG_64=0x%08X)\n", status);
            return false;
        }
        rcpPspWrite(gpu, ipd, RCP_C2PMSG_64, 0x00030000);
        Sleep(20);
    }

    if (!rcpWaitReg(gpu, ipd, RCP_C2PMSG_64, RCP_GFX_FLAG_RESPONSE,
                    RCP_GFX_CMD_RESPONSE_MASK, 20000)) {
        printf("  PSP[recipe]: ERROR TOS not ready (C2PMSG_64 timeout)\n");
        return false;
    }

    rcpPspWrite(gpu, ipd, RCP_C2PMSG_69, (uint32_t)(st.ringGpu & 0xFFFFFFFF));
    rcpPspWrite(gpu, ipd, RCP_C2PMSG_70, (uint32_t)(st.ringGpu >> 32));
    rcpPspWrite(gpu, ipd, RCP_C2PMSG_71, RCP_PSP_RING_SIZE);
    rcpPspWrite(gpu, ipd, RCP_C2PMSG_64, RCP_PSP_RING_TYPE_KM << 16);

    Sleep(20);

    if (!rcpWaitReg(gpu, ipd, RCP_C2PMSG_64, RCP_GFX_FLAG_RESPONSE,
                    RCP_GFX_CMD_RESPONSE_MASK, 20000)) {
        printf("  PSP[recipe]: ERROR ring creation failed (C2PMSG_64 timeout)\n");
        return false;
    }

    st.ringWptr = rcpPspRead(gpu, ipd, RCP_C2PMSG_67);
    printf("  PSP[recipe]: GPCOM ring created (wptr=%u)\n", st.ringWptr);
    return true;
}

/* _submit_psp_cmd_buffer (psp_init.py). cmdId selects the cmd[] layout.
 * For LOAD_IP_FW: fwBus/fwSize/fwType are placed at cmd[7..10].
 * For LOAD_TOC:   fwBus/fwSize at cmd[7..9].
 * For AUTOLOAD_RLC: no payload fields.
 * Returns true on PSP status==0; outTmrSize gets cmd[220] when non-null. */
static bool rcpSubmitCmdBuffer(WddmLite &gpu, const IpDiscoveryResult &ipd,
                               RecipePspState &st, uint32_t cmdId,
                               uint32_t fwType, uint64_t fwBus, uint32_t fwSize,
                               uint32_t *outTmrSize)
{
    /* Zero cmd buffer (1024 bytes) and fence (4 bytes). */
    memset(st.cmdCpu, 0, RCP_GFX_CMD_RESP_SIZE);
    volatile uint32_t *cmd = (volatile uint32_t *)st.cmdCpu;
    cmd[0] = RCP_GFX_CMD_RESP_SIZE;
    cmd[1] = RCP_GFX_CMD_BUF_VERSION;
    cmd[2] = cmdId;

    if (cmdId == RCP_GFX_CMD_ID_LOAD_IP_FW) {
        cmd[7] = (uint32_t)(fwBus & 0xFFFFFFFF);
        cmd[8] = (uint32_t)(fwBus >> 32);
        cmd[9] = fwSize;
        cmd[10] = fwType;
    } else if (cmdId == RCP_GFX_CMD_ID_LOAD_TOC) {
        cmd[7] = (uint32_t)(fwBus & 0xFFFFFFFF);
        cmd[8] = (uint32_t)(fwBus >> 32);
        cmd[9] = fwSize;
    }

    st.fenceValue += 1;
    *st.fenceCpu = 0;

    uint32_t ringSizeDw = RCP_PSP_RING_SIZE / 4;
    uint32_t frameSizeDw = RCP_PSP_RING_FRAME_SIZE / 4;   /* 16 */
    uint32_t ringWptr = rcpPspRead(gpu, ipd, RCP_C2PMSG_67) % ringSizeDw;
    uint32_t frameOffset = (ringWptr * 4) % RCP_PSP_RING_SIZE;
    volatile uint32_t *frame =
        (volatile uint32_t *)((uint8_t *)st.ringCpu + frameOffset);
    for (uint32_t i = 0; i < frameSizeDw; i++)
        frame[i] = 0;
    frame[0] = (uint32_t)(st.cmdGpu & 0xFFFFFFFF);
    frame[1] = (uint32_t)(st.cmdGpu >> 32);
    frame[2] = RCP_GFX_CMD_RESP_SIZE;
    frame[3] = (uint32_t)(st.fenceGpu & 0xFFFFFFFF);
    frame[4] = (uint32_t)(st.fenceGpu >> 32);
    frame[5] = st.fenceValue;

    MemoryBarrier();

    st.ringWptr = (ringWptr + frameSizeDw) % ringSizeDw;
    rcpPspWrite(gpu, ipd, RCP_C2PMSG_67, st.ringWptr);

    /* Wait up to 30s for fence == fence_value, then check resp status. */
    int waited = 0;
    const int timeoutMs = 30000;
    while (waited < timeoutMs) {
        if (*st.fenceCpu == st.fenceValue) {
            uint32_t status = cmd[216];   /* psp_gfx_cmd_resp.resp.status */
            if (status != 0) {
                printf("  PSP[recipe]: cmd 0x%X fw_type=%u FAILED status=0x%X\n",
                       cmdId, fwType, status);
                return false;
            }
            if (outTmrSize) *outTmrSize = cmd[220];
            return true;
        }
        Sleep(1);
        waited += 1;
    }
    printf("  PSP[recipe]: cmd 0x%X fw_type=%u TIMEOUT (fence=%u expected=%u)\n",
           cmdId, fwType, *st.fenceCpu, st.fenceValue);
    return false;
}

/* load_ip_firmware (psp_init.py): copy payload to a fresh VRAM stage buffer
 * then LOAD_IP_FW via the cmd buffer. Mirrors the persistent-VRAM-stage path
 * (AMDGPU_LITE_PSP_REUSE_FW_BUFFER unset). Returns false on PSP rejection. */
static bool rcpLoadIpFw(WddmLite &gpu, const IpDiscoveryResult &ipd,
                        RecipePspState &st, uint32_t fwType,
                        const std::vector<uint8_t> &payload)
{
    if (payload.size() > 1024 * 1024) {
        printf("  PSP[recipe]: ERROR fw too large (%zu bytes)\n", payload.size());
        return false;
    }
    void *stageCpu = nullptr;
    uint64_t stageGpu = 0, stageHandle = 0;
    if (!rcpAllocVram(gpu, payload.size(), &stageCpu, &stageGpu, &stageHandle))
        return false;
    memcpy(stageCpu, payload.data(), payload.size());
    MemoryBarrier();
    bool ok = rcpSubmitCmdBuffer(gpu, ipd, st, RCP_GFX_CMD_ID_LOAD_IP_FW,
                                 fwType, stageGpu, (uint32_t)payload.size(), nullptr);
    /* Stage buffers are intentionally retained (matches firmware_memory_handles
     * in the Python path - PSP may reference them through autoload). */
    return ok;
}

/* load_toc_firmware (psp_init.py): TOC via cmd buffer using the shared fw buf.
 * Here we use a dedicated VRAM stage buffer. */
static bool rcpLoadToc(WddmLite &gpu, const IpDiscoveryResult &ipd,
                       RecipePspState &st, const std::vector<uint8_t> &toc)
{
    if (toc.size() > 1024 * 1024) {
        printf("  PSP[recipe]: ERROR TOC too large (%zu bytes)\n", toc.size());
        return false;
    }
    void *stageCpu = nullptr;
    uint64_t stageGpu = 0, stageHandle = 0;
    if (!rcpAllocVram(gpu, toc.size(), &stageCpu, &stageGpu, &stageHandle))
        return false;
    memcpy(stageCpu, toc.data(), toc.size());
    MemoryBarrier();
    uint32_t tmrSize = 0;
    bool ok = rcpSubmitCmdBuffer(gpu, ipd, st, RCP_GFX_CMD_ID_LOAD_TOC,
                                 0, stageGpu, (uint32_t)toc.size(), &tmrSize);
    if (ok)
        printf("  PSP[recipe]: LOAD_TOC OK (%zu bytes, TMR size=0x%X)\n",
               toc.size(), tmrSize);
    return ok;
}

/* _parse_psp_sos_firmware (psp_init.py): walk the v2 descriptor table of a PSP
 * SOS container and slice out the single component whose fw_type matches.
 * Returns true and fills `out` when found; returns false when the type is
 * absent (matches Python's components.get(type) -> None). `out` is cleared
 * on a missing type so callers can treat empty as absent. */
static bool rcpSosComponent(const std::vector<uint8_t> &sos, uint32_t fwType,
                            std::vector<uint8_t> &out)
{
    out.clear();
    RcpFwHeader h = rcpParseHeader(sos);
    uint16_t hverMajor = rcpU16(sos, 8);
    uint32_t headerSize = rcpU32(sos, 4);   /* header_size_bytes @4 (u32) */
    if (hverMajor < 2) {
        printf("  PSP[recipe]: ERROR SOS header v%u (<2) not supported\n", hverMajor);
        return false;
    }
    uint32_t count = rcpU32(sos, 32);
    size_t descOffset = headerSize;
    for (uint32_t i = 0; i < count; i++) {
        size_t base = descOffset + (size_t)i * 16;
        uint32_t fwTypeId = rcpU32(sos, base);
        uint32_t fwOffset = rcpU32(sos, base + 8);
        uint32_t fwSize = rcpU32(sos, base + 12);
        if (fwTypeId == fwType) {
            if (fwSize == 0) return false;   /* absent component */
            return rcpSlice(sos, (size_t)h.ucodeArrayOffsetBytes + fwOffset,
                            fwSize, out);
        }
    }
    return false;
}

/* _toc_from_sos (psp_init.py): pull PSP_TOC (type 4) out of the SOS container. */
static bool rcpTocFromSos(const std::vector<uint8_t> &sos, std::vector<uint8_t> &toc)
{
    if (!rcpSosComponent(sos, RCP_PSP_FW_TYPE_TOC, toc)) {
        printf("  PSP[recipe]: ERROR no PSP_TOC (fw_type 4) in SOS container\n");
        return false;
    }
    return true;
}

/* _bootloader_load_component (psp_init.py): push one SOS component through the
 * MPASP bootloader mailbox. fwBufCpu/fwBufBus is a shared 1MB-aligned VRAM
 * staging buffer. Returns true when the component existed and was loaded,
 * false when the component is absent in the container (caller skips it). */
static bool rcpBootloaderLoadComponent(WddmLite &gpu, const IpDiscoveryResult &ipd,
                                       const std::vector<uint8_t> &sos,
                                       uint32_t fwType, const char *name,
                                       uint32_t command,
                                       void *fwBufCpu, uint64_t fwBufBus)
{
    std::vector<uint8_t> data;
    if (!rcpSosComponent(sos, fwType, data) || data.empty())
        return false;   /* not present in this container */
    if (data.size() > 1024 * 1024) {
        printf("  PSP[recipe]: ERROR SOS component %s too large (%zu bytes)\n",
               name, data.size());
        return false;
    }

    /* Bootloader must be ready (C2PMSG_35 bit31) before staging the component. */
    if (!rcpWaitReg(gpu, ipd, RCP_C2PMSG_35, RCP_PSP_BL_READY, RCP_PSP_BL_READY, 10000)) {
        printf("  PSP[recipe]: ERROR bootloader not ready before %s "
               "(C2PMSG_35=0x%08X)\n", name, rcpPspRead(gpu, ipd, RCP_C2PMSG_35));
        return false;
    }

    memset(fwBufCpu, 0, 1024 * 1024);
    memcpy(fwBufCpu, data.data(), data.size());
    MemoryBarrier();

    rcpPspWrite(gpu, ipd, RCP_C2PMSG_36, (uint32_t)((fwBufBus >> 20) & 0xFFFFFFFF));
    rcpPspWrite(gpu, ipd, RCP_C2PMSG_35, command);

    /* The SOSDRV command does not return the bootloader to a ready state
     * (it hands control to SOS); every other command does. */
    if (command != RCP_PSP_BL_LOAD_SOSDRV &&
        !rcpWaitReg(gpu, ipd, RCP_C2PMSG_35, RCP_PSP_BL_READY, RCP_PSP_BL_READY, 10000)) {
        printf("  PSP[recipe]: ERROR bootloader command for %s timed out "
               "(C2PMSG_35=0x%08X)\n", name, rcpPspRead(gpu, ipd, RCP_C2PMSG_35));
        return false;
    }

    printf("  PSP[recipe]: Bootloader loaded %s (%zu bytes)\n", name, data.size());
    return true;
}

/* _boot_sos_if_needed (psp_init.py): if VBIOS did not auto-start SOS, drive the
 * MPASP bootloader to load KDB/SPL/SYS/SOC/INTF/DBG/RAS/IPKEYMGR drivers and
 * finally SOSDRV, then wait for the SOS sign-of-life (C2PMSG_81) to go nonzero.
 * Returns true when SOS is alive (already-alive fast path or after boot). */
static bool rcpRead(const char *fwDir, const char *name, std::vector<uint8_t> &out);

static bool rcpBootSosIfNeeded(WddmLite &gpu, const IpDiscoveryResult &ipd,
                               const char *fwDir, const char *mp0)
{
    uint32_t sol = rcpPspRead(gpu, ipd, RCP_C2PMSG_81);
    if (sol != 0) {
        printf("  PSP[recipe]: SOS is alive (POST'd by VBIOS) C2PMSG_81=0x%08X\n", sol);
        return true;
    }

    printf("  PSP[recipe]: SOS not started by VBIOS; booting via MPASP bootloader\n");

    char sosName[64];
    snprintf(sosName, sizeof(sosName), "psp_%s_sos.bin", mp0);
    std::vector<uint8_t> sos;
    if (!rcpRead(fwDir, sosName, sos)) {
        printf("  PSP[recipe]: ERROR cannot read %s for bootloader SOS start\n", sosName);
        return false;
    }

    /* Shared 1MB staging buffer (1MB-aligned for the C2PMSG_36 addr>>20 ABI). */
    void *fwBufCpu = nullptr;
    uint64_t fwBufBus = 0, fwBufHandle = 0;
    if (!rcpAllocVramAligned(gpu, 1024 * 1024, 1024 * 1024,
                             &fwBufCpu, &fwBufBus, &fwBufHandle))
        return false;

    struct BlStep { uint32_t fwType; const char *name; uint32_t command; };
    const BlStep sequence[] = {
        { RCP_PSP_FW_TYPE_KDB,          "KDB",          RCP_PSP_BL_LOAD_KEY_DATABASE },
        { RCP_PSP_FW_TYPE_SPL,          "SPL",          RCP_PSP_BL_LOAD_TOS_SPL_TABLE },
        { RCP_PSP_FW_TYPE_SYS_DRV,      "SYS_DRV",      RCP_PSP_BL_LOAD_SYSDRV },
        { RCP_PSP_FW_TYPE_SOC_DRV,      "SOC_DRV",      RCP_PSP_BL_LOAD_SOCDRV },
        { RCP_PSP_FW_TYPE_INTF_DRV,     "INTF_DRV",     RCP_PSP_BL_LOAD_INTFDRV },
        { RCP_PSP_FW_TYPE_DBG_DRV,      "DBG_DRV",      RCP_PSP_BL_LOAD_HADDRV },
        { RCP_PSP_FW_TYPE_RAS_DRV,      "RAS_DRV",      RCP_PSP_BL_LOAD_RASDRV },
        { RCP_PSP_FW_TYPE_IPKEYMGR_DRV, "IPKEYMGR_DRV", RCP_PSP_BL_LOAD_IPKEYMGRDRV },
        { RCP_PSP_FW_TYPE_SOS,          "SOS",          RCP_PSP_BL_LOAD_SOSDRV },
    };

    int loaded = 0;
    for (size_t i = 0; i < sizeof(sequence) / sizeof(sequence[0]); i++) {
        const BlStep &s = sequence[i];
        if (rcpBootloaderLoadComponent(gpu, ipd, sos, s.fwType, s.name, s.command,
                                       fwBufCpu, fwBufBus))
            loaded++;
    }
    if (loaded == 0) {
        printf("  PSP[recipe]: ERROR no SOS components found in %s\n", sosName);
        return false;
    }

    /* Poll the SOS sign-of-life for up to ~5s after the bootloader sequence. */
    for (int i = 0; i < 5000; i++) {
        sol = rcpPspRead(gpu, ipd, RCP_C2PMSG_81);
        if (sol != 0) {
            printf("  PSP[recipe]: SOS started by userspace bootloader path "
                   "(C2PMSG_81=0x%08X)\n", sol);
            return true;
        }
        Sleep(1);
    }

    printf("  PSP[recipe]: ERROR SOS did not start after bootloader load "
           "(C2PMSG_81=0x%08X)\n", rcpPspRead(gpu, ipd, RCP_C2PMSG_81));
    return false;
}

/* _recipe_extract_rs64: gfx_firmware_header_v2_0 -> (ucode, data) @+36/40/44/48. */
static bool rcpExtractRs64(const std::vector<uint8_t> &blob,
                           std::vector<uint8_t> &ucode, std::vector<uint8_t> &data)
{
    uint32_t uSz = rcpU32(blob, 36);
    uint32_t uOff = rcpU32(blob, 40);
    uint32_t dSz = rcpU32(blob, 44);
    uint32_t dOff = rcpU32(blob, 48);
    return rcpSlice(blob, uOff, uSz, ucode) && rcpSlice(blob, dOff, dSz, data);
}

/* _recipe_extract_mes: mes_firmware_header_v1_0.
 * struct.unpack_from("<IIIIII", blob, 32) -> _uver,u_sz,u_off,_dver,d_sz,d_off. */
static bool rcpExtractMes(const std::vector<uint8_t> &blob,
                          std::vector<uint8_t> &ucode, std::vector<uint8_t> &data)
{
    uint32_t uSz = rcpU32(blob, 36);
    uint32_t uOff = rcpU32(blob, 40);
    uint32_t dSz = rcpU32(blob, 48);
    uint32_t dOff = rcpU32(blob, 52);
    return rcpSlice(blob, uOff, uSz, ucode) && rcpSlice(blob, dOff, dSz, data);
}

/* _recipe_extract_imu: iram at uoff (u32@24), dram follows. Sizes @32/@40. */
static bool rcpExtractImu(const std::vector<uint8_t> &blob,
                          std::vector<uint8_t> &iram, std::vector<uint8_t> &dram)
{
    uint32_t uOff = rcpU32(blob, 24);
    uint32_t iramSz = rcpU32(blob, 32);
    uint32_t dramSz = rcpU32(blob, 40);
    return rcpSlice(blob, uOff, iramSz, iram) &&
           rcpSlice(blob, (size_t)uOff + iramSz, dramSz, dram);
}

/* _recipe_extract_sdma: sdma_firmware_header_v3_0, ucode_offset/size @+36/+40. */
static bool rcpExtractSdma(const std::vector<uint8_t> &blob,
                           std::vector<uint8_t> &ucode)
{
    uint32_t uOff = rcpU32(blob, 36);
    uint32_t uSz = rcpU32(blob, 40);
    return rcpSlice(blob, uOff, uSz, ucode);
}

/* _recipe_extract_rlc_subs: RLC main ucode + sub-fw keyed by SOC24 id.
 * Keys: 1=RLC_G_UCODE, 3=SRLG, 4=SRLS, 7=IRAM, 9=DRAM_BOOT. Absolute offsets. */
struct RcpRlcSubs {
    std::vector<uint8_t> rlcG;   /* key 1 */
    std::vector<uint8_t> srlg;   /* key 3 */
    std::vector<uint8_t> srls;   /* key 4 */
    std::vector<uint8_t> iram;   /* key 7 */
    std::vector<uint8_t> dram;   /* key 9 */
};

static bool rcpExtractRlcSubs(const std::vector<uint8_t> &blob, RcpRlcSubs &out)
{
    uint32_t uOff = rcpU32(blob, 24);
    uint32_t uSz = rcpU32(blob, 20);
    uint16_t hverMin = rcpU16(blob, 10);
    if (!rcpSlice(blob, uOff, uSz, out.rlcG)) return false;
    if (hverMin >= 1) {
        uint32_t srlgSz = rcpU32(blob, 132), srlgOff = rcpU32(blob, 136);
        uint32_t srlsSz = rcpU32(blob, 148), srlsOff = rcpU32(blob, 152);
        if (srlgSz && srlgOff && !rcpSlice(blob, srlgOff, srlgSz, out.srlg)) return false;
        if (srlsSz && srlsOff && !rcpSlice(blob, srlsOff, srlsSz, out.srls)) return false;
    }
    if (hverMin >= 2) {
        uint32_t iramSz = rcpU32(blob, 156), iramOff = rcpU32(blob, 160);
        uint32_t dramSz = rcpU32(blob, 164), dramOff = rcpU32(blob, 168);
        if (iramSz && iramOff && !rcpSlice(blob, iramOff, iramSz, out.iram)) return false;
        if (dramSz && dramOff && !rcpSlice(blob, dramOff, dramSz, out.dram)) return false;
    }
    return true;
}

/* Read a firmware file from fwDir into a vector (loadFirmwareFile wrapper). */
static bool rcpRead(const char *fwDir, const char *name, std::vector<uint8_t> &out)
{
    char path[512];
    snprintf(path, sizeof(path), "%s\\%s", fwDir, name);
    return loadFirmwareFile(path, out);
}

/* ---- SMU mailbox (smu_init.py) ------------------------------------------- */

static uint32_t rcpSmuRead(WddmLite &gpu, const IpDiscoveryResult &ipd, uint32_t dw)
{
    uint32_t v = 0;
    gpu.readReg32((ipd.mp1Base + dw) * 4, &v);
    return v;
}

static void rcpSmuWrite(WddmLite &gpu, const IpDiscoveryResult &ipd,
                        uint32_t dw, uint32_t v)
{
    gpu.writeReg32((ipd.mp1Base + dw) * 4, v);
}

/* send_smu_msg (smu_init.py): clear resp, write param, write msg, poll resp.
 * Returns the response code. When readBackArg, *argOut gets C2PMSG_82. */
static uint32_t rcpSmuMsg(WddmLite &gpu, const IpDiscoveryResult &ipd,
                          uint32_t msg, uint32_t param, int timeoutMs,
                          uint32_t *argOut, const char *label)
{
    rcpSmuWrite(gpu, ipd, RCP_MP1_C2PMSG_90, 0);
    rcpSmuWrite(gpu, ipd, RCP_MP1_C2PMSG_82, param);
    rcpSmuWrite(gpu, ipd, RCP_MP1_C2PMSG_66, msg);

    uint32_t resp = 0;
    int waited = 0;
    while (waited < timeoutMs) {
        resp = rcpSmuRead(gpu, ipd, RCP_MP1_C2PMSG_90);
        if (resp != 0) break;
        Sleep(1);
        waited += 1;
    }
    if (resp == 0) {
        printf("  SMU: %s timed out\n", label);
        return 0;
    }
    if (resp != RCP_PPSMC_RESULT_OK) {
        printf("  SMU: %s failed with response 0x%X\n", label, resp);
        return resp;
    }
    if (argOut) *argOut = rcpSmuRead(gpu, ipd, RCP_MP1_C2PMSG_82);
    return resp;
}

/* init_smu (smu_init.py) condensed to the proven probe path for MP1 14.0.3:
 *   GetSmuVersion -> publish driver table -> EnableAllSmuFeatures(0)
 *   -> DisallowGfxOff -> read enabled features. Failures are non-fatal
 *   (matches _try_smu_msg), so we always return after attempting the batch. */
static void rcpInitSmu(WddmLite &gpu, const IpDiscoveryResult &ipd,
                       uint64_t vramMcBase)
{
    printf("\n=== SMU init (recipe) ===\n");

    uint32_t version = 0;
    uint32_t resp = rcpSmuMsg(gpu, ipd, RCP_SMU_GET_VERSION, 0, 1000,
                              &version, "GetSmuVersion");
    if (resp != RCP_PPSMC_RESULT_OK) {
        printf("  SMU: init skipped (GetSmuVersion failed)\n");
        return;
    }
    printf("  SMU: firmware version 0x%08X\n", version);

    /* _allocate_driver_table: publish a small VRAM table. Address translation
     * matches _driver_table_gpu_addr (add vram_mc_base when gpu_addr below it). */
    void *tblCpu = nullptr;
    uint64_t tblGpu = 0, tblHandle = 0;
    if (gpu.allocMemory(RCP_SMU_DRIVER_TABLE_SIZE,
                        AMDGPU_MEM_TYPE_VRAM | AMDGPU_MEM_FLAG_HOST_ACCESS,
                        &tblCpu, &tblGpu, &tblHandle) && tblCpu) {
        memset(tblCpu, 0, RCP_SMU_DRIVER_TABLE_SIZE);
        uint64_t tblMc = tblGpu;
        if (vramMcBase && tblGpu < vramMcBase)
            tblMc = tblGpu + vramMcBase;
        rcpSmuMsg(gpu, ipd, RCP_SMU_SET_DRIVER_DRAM_HIGH,
                  (uint32_t)(tblMc >> 32), 1000, nullptr, "SetDriverDramAddrHigh");
        rcpSmuMsg(gpu, ipd, RCP_SMU_SET_DRIVER_DRAM_LOW,
                  (uint32_t)(tblMc & 0xFFFFFFFF), 1000, nullptr, "SetDriverDramAddrLow");
        printf("  SMU: driver table at MC 0x%012llX\n", (unsigned long long)tblMc);
    } else {
        printf("  SMU: WARNING driver table alloc skipped\n");
    }

    /* _enable_smu_features with AMDGPU_LITE_ENABLE_SMU_FEATURES=1, no allowed
     * mask, not gfx-only -> EnableAllSmuFeatures(param=0). This completes the
     * autoload (see project memory). */
    if (rcpSmuMsg(gpu, ipd, RCP_SMU_ENABLE_ALL_FEATURES, 0, 5000, nullptr,
                  "EnableAllSmuFeatures(param=0x0)") == RCP_PPSMC_RESULT_OK)
        printf("  SMU: EnableAllSmuFeatures OK\n");

    /* disable_gfxoff=True (AMDGPU_LITE_DISALLOW_GFXOFF=1). */
    if (rcpSmuMsg(gpu, ipd, RCP_SMU_DISALLOW_GFXOFF, 0, 2000, nullptr,
                  "DisallowGfxOff") == RCP_PPSMC_RESULT_OK)
        printf("  SMU: DisallowGfxOff OK\n");

    uint32_t lo = 0, hi = 0;
    if (rcpSmuMsg(gpu, ipd, RCP_SMU_GET_ENABLED_FEATURES_LOW, 0, 1000, &lo,
                  "GetEnabledSmuFeaturesLow") == RCP_PPSMC_RESULT_OK &&
        rcpSmuMsg(gpu, ipd, RCP_SMU_GET_ENABLED_FEATURES_HIGH, 0, 1000, &hi,
                  "GetEnabledSmuFeaturesHigh") == RCP_PPSMC_RESULT_OK)
        printf("  SMU: enabled features high=0x%08X low=0x%08X\n", hi, lo);
}

/* ---- Top-level recipe ---------------------------------------------------- */

bool recipeBootload(WddmLite &gpu, const IpDiscoveryResult &ipd,
                    const char *fwDir, uint64_t vramMcBase)
{
    printf("\n=== recipeBootload (LITE_MES_RECIPE) ===\n");
    printf("MP0=0x%04X MP1=0x%04X fwDir=%s vramMcBase=0x%llX\n",
           ipd.mp0Base, ipd.mp1Base, fwDir, (unsigned long long)vramMcBase);

    /* Initialize the VRAM bump allocator once, here at the entry to bootload.
     * The cursor PERSISTS through recipeDispatch/buildComputeGpuvm (only reset
     * here), so every VRAM allocation in the bootload -> dispatch sequence gets
     * a unique FB MC address. Initial cursor mirrors Python device.py
     * _vram_cursor = 32 * 1024 * 1024 (reserve the low 32MB for scratch). */
    g_vramMcBase = vramMcBase;
    g_vramCursor = 32ull * 1024 * 1024;

    /* IP version strings the probe resolves (gc=12_0_1, sdma=12_0_1, mp1=14_0_3).
     * The probe passes GC="12_0_1" and SOS uses mp0 14_0_3. */
    const char *gc = "12_0_1";
    const char *mp1 = "14_0_3";
    const char *mp0 = "14_0_3";

    /* sdma_v = ip_versions.get("sdma", gc): resolved from the SDMA0 IP block
     * (gfx1201 reports 7_0_x). _resolve_ip_versions falls back to gc only if
     * no SDMA0 block exists. SDMA is optional in the recipe, so a missing
     * file is non-fatal. */
    char sdmaVer[32];
    snprintf(sdmaVer, sizeof(sdmaVer), "%s", gc);
    for (uint32_t i = 0; i < ipd.numBlocks; i++) {
        const IpBlock &b = ipd.blocks[i];
        if (b.hwId == HWID_SDMA0 && b.instance == 0) {
            snprintf(sdmaVer, sizeof(sdmaVer), "%u_%u_%u",
                     b.majorVer, b.minorVer, b.revision);
            break;
        }
    }

    /* --- init_psp: allocate VRAM ring/cmd/fence and create the GPCOM ring. --- */
    RecipePspState st;
    memset(&st, 0, sizeof(st));

    /* Start SOS via the MPASP bootloader if VBIOS did not POST it (the GPCOM
     * ring below needs a live SOS). Fast-path returns immediately when the
     * SOS sign-of-life (C2PMSG_81) is already set. Mirrors _boot_sos_if_needed
     * running before the ring/cmd/fence allocs in init_psp (psp_init.py). */
    if (!rcpBootSosIfNeeded(gpu, ipd, fwDir, mp0))
        return false;

    if (!rcpAllocVram(gpu, RCP_PSP_RING_SIZE, &st.ringCpu, &st.ringGpu, &st.ringHandle))
        return false;
    if (!rcpAllocVram(gpu, RCP_GFX_CMD_RESP_SIZE, &st.cmdCpu, &st.cmdGpu, &st.cmdHandle))
        return false;
    {
        void *fc = nullptr;
        if (!rcpAllocVram(gpu, RCP_PSP_FENCE_SIZE, &fc, &st.fenceGpu, &st.fenceHandle))
            return false;
        st.fenceCpu = (volatile uint32_t *)fc;
    }
    printf("  PSP[recipe]: ring MC=0x%llX cmd MC=0x%llX fence MC=0x%llX\n",
           (unsigned long long)st.ringGpu, (unsigned long long)st.cmdGpu,
           (unsigned long long)st.fenceGpu);

    if (!rcpCreateRing(gpu, ipd, st))
        return false;

    /* ===== load_all_firmware_recipe ===== */

    /* 1. LOAD_TOC from the SOS container (NOT gc_*_toc.bin). */
    {
        char sosName[64];
        snprintf(sosName, sizeof(sosName), "psp_%s_sos.bin", mp0);
        std::vector<uint8_t> sos, toc;
        if (!rcpRead(fwDir, sosName, sos)) return false;
        if (!rcpTocFromSos(sos, toc)) return false;
        if (!rcpLoadToc(gpu, ipd, st, toc)) {
            printf("  PSP[recipe]: ERROR LOAD_TOC failed\n");
            return false;
        }
    }

    /* 2. LOAD_IP_FW(SMU) first (tinygrad order). Non-fatal if rejected. */
    {
        char smuName[64];
        snprintf(smuName, sizeof(smuName), "smu_%s.bin", mp1);
        std::vector<uint8_t> smu, payload;
        if (rcpRead(fwDir, smuName, smu)) {
            RcpFwHeader h = rcpParseHeader(smu);
            if (rcpSlice(smu, h.ucodeArrayOffsetBytes, h.ucodeSizeBytes, payload)) {
                if (rcpLoadIpFw(gpu, ipd, st, RCP_FW_SMU, payload))
                    printf("  PSP[recipe]: SMU fw loaded (%zu bytes)\n", payload.size());
                else
                    printf("  PSP[recipe]: WARNING SMU LOAD_IP_FW rejected (non-fatal)\n");
            }
        }
    }

    /* 3. Parse + load the gfx batch, RLC_G LAST. */
    char imuName[64], rlcName[64], pfpName[64], meName[64], mecName[64], uniName[64];
    char sdmaName[64];
    snprintf(imuName, sizeof(imuName), "gc_%s_imu.bin", gc);
    snprintf(rlcName, sizeof(rlcName), "gc_%s_rlc.bin", gc);
    snprintf(pfpName, sizeof(pfpName), "gc_%s_pfp.bin", gc);
    snprintf(meName, sizeof(meName), "gc_%s_me.bin", gc);
    snprintf(mecName, sizeof(mecName), "gc_%s_mec.bin", gc);
    snprintf(uniName, sizeof(uniName), "gc_%s_uni_mes.bin", gc);
    snprintf(sdmaName, sizeof(sdmaName), "sdma_%s.bin", sdmaVer);

    std::vector<uint8_t> imuBlob, rlcBlob, pfpBlob, meBlob, mecBlob, uniBlob, sdmaBlob;
    if (!rcpRead(fwDir, imuName, imuBlob)) return false;
    if (!rcpRead(fwDir, rlcName, rlcBlob)) return false;
    if (!rcpRead(fwDir, pfpName, pfpBlob)) return false;
    if (!rcpRead(fwDir, meName, meBlob)) return false;
    if (!rcpRead(fwDir, mecName, mecBlob)) return false;
    if (!rcpRead(fwDir, uniName, uniBlob)) return false;
    bool haveSdma = rcpRead(fwDir, sdmaName, sdmaBlob);   /* _optional_firmware */

    std::vector<uint8_t> imuI, imuD;
    RcpRlcSubs rlc;
    std::vector<uint8_t> pfpU, pfpD, meU, meD, mecU, mecD, uniU, uniD, sdmaU;
    if (!rcpExtractImu(imuBlob, imuI, imuD)) return false;
    if (!rcpExtractRlcSubs(rlcBlob, rlc)) return false;
    if (!rcpExtractRs64(pfpBlob, pfpU, pfpD)) return false;
    if (!rcpExtractRs64(meBlob, meU, meD)) return false;
    if (!rcpExtractRs64(mecBlob, mecU, mecD)) return false;
    if (!rcpExtractMes(uniBlob, uniU, uniD)) return false;
    if (haveSdma && !rcpExtractSdma(sdmaBlob, sdmaU)) return false;

    /* Build the ordered batch. RLC_G MUST be last. */
    struct BatchItem { const char *label; uint32_t fwType; const std::vector<uint8_t> *payload; };
    std::vector<BatchItem> batch;
    if (haveSdma)
        batch.push_back({"SDMA_TH0", RCP_FW_SDMA_UCODE_TH0, &sdmaU});
    batch.push_back({"RLC_IRAM", RCP_FW_RLC_IRAM, &rlc.iram});
    batch.push_back({"RLC_DRAM_BOOT", RCP_FW_RLC_DRAM_BOOT, &rlc.dram});
    batch.push_back({"RLC_SRLG", RCP_FW_RLC_RESTORE_LIST_GPM_MEM, &rlc.srlg});
    batch.push_back({"RLC_SRLS", RCP_FW_RLC_RESTORE_LIST_SRM_MEM, &rlc.srls});
    batch.push_back({"RS64_PFP", RCP_FW_RS64_PFP, &pfpU});
    batch.push_back({"RS64_PFP_P0", RCP_FW_RS64_PFP_P0_STACK, &pfpD});
    batch.push_back({"RS64_PFP_P1", RCP_FW_RS64_PFP_P1_STACK, &pfpD});
    batch.push_back({"RS64_ME", RCP_FW_RS64_ME, &meU});
    batch.push_back({"RS64_ME_P0", RCP_FW_RS64_ME_P0_STACK, &meD});
    batch.push_back({"RS64_ME_P1", RCP_FW_RS64_ME_P1_STACK, &meD});
    batch.push_back({"RS64_MEC", RCP_FW_RS64_MEC, &mecU});
    batch.push_back({"RS64_MEC_P0", RCP_FW_RS64_MEC_P0_STACK, &mecD});
    batch.push_back({"RS64_MEC_P1", RCP_FW_RS64_MEC_P1_STACK, &mecD});
    batch.push_back({"RS64_MEC_P2", RCP_FW_RS64_MEC_P2_STACK, &mecD});
    batch.push_back({"RS64_MEC_P3", RCP_FW_RS64_MEC_P3_STACK, &mecD});
    batch.push_back({"CP_MES", RCP_FW_CP_MES, &uniU});
    batch.push_back({"MES_STACK", RCP_FW_MES_STACK, &uniD});
    batch.push_back({"CP_MES_KIQ", RCP_FW_CP_MES_KIQ, &uniU});
    batch.push_back({"MES_KIQ_STACK", RCP_FW_MES_KIQ_STACK, &uniD});
    batch.push_back({"IMU_I", RCP_FW_IMU_I, &imuI});
    batch.push_back({"IMU_D", RCP_FW_IMU_D, &imuD});
    batch.push_back({"RLC_G", RCP_FW_RLC_G, &rlc.rlcG});  /* MUST be last */

    int failures = 0;
    bool mesFailed = false;
    for (size_t i = 0; i < batch.size(); i++) {
        const BatchItem &it = batch[i];
        if (it.payload->empty()) {
            printf("  PSP[recipe]: %-16s empty payload, skipped\n", it.label);
            continue;
        }
        if (rcpLoadIpFw(gpu, ipd, st, it.fwType, *it.payload)) {
            printf("  PSP[recipe]: %-16s type=%-3u size=%-7zu OK\n",
                   it.label, it.fwType, it.payload->size());
        } else {
            failures++;
            printf("  PSP[recipe]: %-16s type=%-3u REJECTED\n", it.label, it.fwType);
            if (strncmp(it.label, "CP_MES", 6) == 0)
                mesFailed = true;
        }
    }
    if (failures)
        printf("  PSP[recipe]: %d fw type(s) rejected (non-fatal)\n", failures);
    printf("  PSP[recipe]: mes_psp_loaded=%s\n", mesFailed ? "false" : "true");

    /* 4. AUTOLOAD_RLC: PSP runs the full backdoor autoload internally. */
    if (!rcpSubmitCmdBuffer(gpu, ipd, st, RCP_GFX_CMD_ID_AUTOLOAD_RLC,
                            0, 0, 0, nullptr)) {
        printf("  PSP[recipe]: ERROR AUTOLOAD_RLC failed\n");
        return false;
    }
    printf("  PSP[recipe]: AUTOLOAD_RLC triggered\n");

    /* ===== init_smu(disable_gfxoff=True, vram_mc_base=gmc.vram_start) =====
     * EnableAllSmuFeatures(0) here is what completes the autoload. */
    rcpInitSmu(gpu, ipd, vramMcBase);

    /* ===== BOOTLOAD poll: read_reg32((0xA000 + 0x4e7c) * 4) ===== */
    uint32_t bootStatus = 0;
    for (int i = 0; i < 100; i++) {
        gpu.readReg32(RCP_BOOTLOAD_DWORD * 4, &bootStatus);
        if (bootStatus & 0x80000000) break;
        Sleep(10);
    }
    printf("\nBOOTLOAD_STATUS=0x%08X\n", bootStatus);
    bool pass = (bootStatus == RCP_BOOTLOAD_OK);
    printf("BOOTLOAD %s (expected 0x%08X, bit31=%s)\n",
           pass ? "PASS" : "FAIL", RCP_BOOTLOAD_OK,
           (bootStatus & 0x80000000) ? "set" : "clear");
    return pass;
}

/* ======================================================================
 * Increment 2a: MEC enable + direct compute HQD queue + NOP/RELEASE_MEM
 *
 * Faithful C++ transcription of the proven probe path:
 *   python/amd_gpu_driver/backends/windows/ring_init.py
 *     init_gfx_for_compute (CP PFP/ME/MEC counters from gc fw ucode_start,
 *       RLC/SH_MEM/doorbell-range, _enable_mec), init_compute_queue +
 *       _init_compute_mqd + _activate_compute_queue_mmio,
 *       submit_compute_packets, wait_fence, grbm_select/deselect
 *   python/amd_gpu_driver/commands/pm4.py (PM4PacketBuilder: NOP + RELEASE_MEM)
 *   python/amd_gpu_driver/probe16_vmid0.py + probe_kernel.py
 *     (rs64()/uni_mes ucode_start extraction; VMID 0 stays default)
 *
 * GC register addressing matches _gc_reg/_gc_wreg exactly:
 *   byte_offset = (gc_base[base_idx] + reg) * 4
 *   base_idx 0 -> ipd.gcBase  (gc_base[0])
 *   base_idx 1 -> ipd.gcBase1 (gc_base[1]; == 0xA000 on gfx1201)
 * This is the same scheme recipeBootload uses for the BOOTLOAD read
 * (0xA000 + 0x4e7c == gcBase1 + regRLC_RLCS_BOOTLOAD_STATUS).
 * ====================================================================== */

/* ---- GC 12.0 register DWORD offsets (gc_12_0_0_offset.h, ring_init.py) ---- */

/* GRBM */
#define regGRBM_GFX_CNTL             0x0900   /* base_idx 1 */
#define regGRBM_CNTL                 0x0DA0   /* base_idx 0 */

/* CP/RLC engine bring-up */
#define regCP_PFP_PRGRM_CNTR_START      0x1E44  /* base_idx 0 */
#define regCP_ME_PRGRM_CNTR_START       0x1E45  /* base_idx 0 */
#define regCP_PFP_PRGRM_CNTR_START_HI   0x1E59  /* base_idx 0 */
#define regCP_ME_PRGRM_CNTR_START_HI    0x1E79  /* base_idx 0 */
#define regCP_ME_CNTL                   0x0803  /* base_idx 1 */
#define regCP_MEC_RS64_PRGRM_CNTR_START     0x2900  /* base_idx 1 */
#define regCP_MEC_RS64_CNTL                 0x2904  /* base_idx 1 */
#define regCP_MEC_RS64_PRGRM_CNTR_START_HI  0x2938  /* base_idx 1 */
#define regCP_MEC_DOORBELL_RANGE_LOWER  0x1DFC  /* base_idx 0 */
#define regCP_MEC_DOORBELL_RANGE_UPPER  0x1DFD  /* base_idx 0 */
#define regRLC_CNTL                     0x4C00  /* base_idx 1 */
#define regRLC_SRM_CNTL                 0x4C80  /* base_idx 1 */
#define regRLC_SPM_MC_CNTL              0x0982  /* base_idx 1 */
#define regSH_MEM_BASES                 0x09E3  /* base_idx 1 */
#define regSH_MEM_CONFIG                0x09E4  /* base_idx 1 */
#define regTCP_CNTL                     0x19A2  /* base_idx 1 */

/* CP HQD registers (base_idx 0, direct queue programming) */
#define regCP_MQD_BASE_ADDR                 0x1FA9
#define regCP_MQD_BASE_ADDR_HI              0x1FAA
#define regCP_HQD_ACTIVE                    0x1FAB
#define regCP_HQD_VMID                      0x1FAC
#define regCP_HQD_PERSISTENT_STATE          0x1FAD
#define regCP_HQD_PIPE_PRIORITY             0x1FAE
#define regCP_HQD_QUEUE_PRIORITY            0x1FAF
#define regCP_HQD_QUANTUM                   0x1FB0
#define regCP_HQD_PQ_BASE                   0x1FB1
#define regCP_HQD_PQ_BASE_HI                0x1FB2
#define regCP_HQD_PQ_RPTR                   0x1FB3
#define regCP_HQD_PQ_RPTR_REPORT_ADDR       0x1FB4
#define regCP_HQD_PQ_RPTR_REPORT_ADDR_HI    0x1FB5
#define regCP_HQD_PQ_WPTR_POLL_ADDR         0x1FB6
#define regCP_HQD_PQ_WPTR_POLL_ADDR_HI      0x1FB7
#define regCP_HQD_PQ_DOORBELL_CONTROL       0x1FB8
#define regCP_HQD_PQ_CONTROL                0x1FBA
#define regCP_HQD_IB_CONTROL                0x1FBE
#define regCP_MQD_CONTROL                   0x1FCB
#define regCP_HQD_EOP_BASE_ADDR             0x1FCE
#define regCP_HQD_EOP_BASE_ADDR_HI          0x1FCF
#define regCP_HQD_EOP_CONTROL               0x1FD0
#define regCP_HQD_HQ_STATUS0                0x1FC9
#define regCP_HQD_AQL_CONTROL               0x1FDE
#define regCP_HQD_PQ_WPTR_LO                0x1FDF
#define regCP_HQD_PQ_WPTR_HI                0x1FE0

/* GRBM_GFX_CNTL bit shifts */
#define GRBM_GFX_CNTL__PIPEID__SHIFT   0
#define GRBM_GFX_CNTL__MEID__SHIFT     2
#define GRBM_GFX_CNTL__VMID__SHIFT     4
#define GRBM_GFX_CNTL__QUEUEID__SHIFT  8

/* CP engine control bits */
#define CP_ME_CNTL__PFP_PIPE0_RESET    0x00040000
#define CP_ME_CNTL__ME_PIPE0_RESET     0x00100000
#define CP_ME_CNTL__PFP_HALT           0x04000000
#define CP_ME_CNTL__ME_HALT            0x10000000
#define CP_MEC_RS64_CNTL__MEC_PIPE0_RESET    0x00010000
#define CP_MEC_RS64_CNTL__MEC_PIPE1_RESET    0x00020000
#define CP_MEC_RS64_CNTL__MEC_PIPE2_RESET    0x00040000
#define CP_MEC_RS64_CNTL__MEC_PIPE3_RESET    0x00080000
#define CP_MEC_RS64_CNTL__MEC_PIPE0_ACTIVE   0x04000000
#define CP_MEC_RS64_CNTL__MEC_PIPE1_ACTIVE   0x08000000
#define CP_MEC_RS64_CNTL__MEC_PIPE2_ACTIVE   0x10000000
#define CP_MEC_RS64_CNTL__MEC_PIPE3_ACTIVE   0x20000000
#define CP_MEC_RS64_CNTL__MEC_INVALIDATE_ICACHE 0x00000010
#define CP_MEC_RS64_CNTL__MEC_HALT           0x40000000

/* CP_HQD_PQ_DOORBELL_CONTROL bits */
#define HQD_DOORBELL_OFFSET__SHIFT   2
#define HQD_DOORBELL_EN              (1u << 30)

/* CP_HQD_PQ_CONTROL bits */
#define PQ_CONTROL__QUEUE_SIZE__SHIFT       0
#define PQ_CONTROL__RPTR_BLOCK_SIZE__SHIFT  8
#define PQ_CONTROL__PQ_EMPTY                (1u << 15)
#define PQ_CONTROL__MIN_AVAIL_SIZE__SHIFT   20
#define PQ_CONTROL__NO_UPDATE_RPTR          (1u << 27)
#define PQ_CONTROL__UNORD_DISPATCH          (1u << 28)
#define PQ_CONTROL__PRIV_STATE              (1u << 30)
#define PQ_CONTROL__KMD_QUEUE               (1u << 31)

/* CP_HQD_PERSISTENT_STATE */
#define HQD_PERSISTENT_STATE__PRELOAD_REQ        (1u << 0)
#define HQD_PERSISTENT_STATE__PRELOAD_SIZE__SHIFT 8
#define HQD_PERSISTENT_STATE__PRELOAD_SIZE       0x55
#define HQD_PERSISTENT_STATE_DEFAULT             0x0BE05501u

/* v12_compute_mqd */
#define MQD_HEADER          0xC0310800u
#define MQD_SIZE_BYTES      (256 * 4)

/* Default ring/EOP sizes (match the lite direct queue). */
#define COMPUTE_RING_SIZE   (4 * 1024)
#define EOP_BUFFER_SIZE     (4 * 1024)

/* Doorbell layout (SOC24/Navi). DWORD offsets (slots << 1). */
#define DOORBELL_MEC_RING_START   0x006
#define DOORBELL_MEC_RING_STRIDE  0x2

/* NBIF/NBIO (base_idx 2). nbio_init.py. */
#define RCP_HWID_NBIF                108
#define regRCC_DOORBELL_APER_EN      0x00C0
#define BIF_DOORBELL_APER_EN__BIT    0x00000001
#define regBIF_FB_EN                 0x0100
#define BIF_FB_EN__FB_READ_EN        0x00000001
#define BIF_FB_EN__FB_WRITE_EN       0x00000002
#define regHDP_MEM_COHERENCY_FLUSH   0x00F7

/* PM4 (pm4.py). */
#define PACKET3_NOP                  0x10
#define PACKET3_RELEASE_MEM          0x49
#define EVENT_TYPE_CACHE_FLUSH_AND_INV_TS_EVENT 0x14
#define DATA_SEL_SEND_64BIT          2
#define INT_SEL_SEND_INT_ON_CONFIRM  2
#define RELEASE_MEM_EVENT_INDEX_EOP  5
#define PACKET3_RELEASE_MEM_GCR_GLM_WB   (1u << 12)
#define PACKET3_RELEASE_MEM_GCR_GLM_INV  (1u << 13)
#define PACKET3_RELEASE_MEM_GCR_GLV_INV  (1u << 14)
#define PACKET3_RELEASE_MEM_GCR_GL1_INV  (1u << 15)
#define PACKET3_RELEASE_MEM_GCR_GL2_INV  (1u << 20)
#define PACKET3_RELEASE_MEM_GCR_GL2_WB   (1u << 21)
#define PACKET3_RELEASE_MEM_GCR_SEQ      (1u << 22)

/* Compute queue state (mirror of ComputeQueueConfig, NOP+fence subset). */
struct CqState {
    uint32_t gcBase0;       /* gc_base[0] */
    uint32_t gcBase1;       /* gc_base[1] */

    void    *ringCpu;
    uint64_t ringGpu;
    uint64_t ringHandle;
    uint32_t ringSize;

    void    *mqdCpu;
    uint64_t mqdGpu;
    uint64_t mqdHandle;

    void    *eopCpu;
    uint64_t eopGpu;
    uint64_t eopHandle;

    volatile uint32_t *wptrCpu;
    uint64_t wptrGpu;
    uint64_t wptrHandle;

    volatile uint32_t *rptrCpu;
    uint64_t rptrGpu;
    uint64_t rptrHandle;

    volatile uint64_t *fenceCpu;
    uint64_t fenceGpu;
    uint64_t fenceHandle;

    uint32_t doorbellIndex;     /* DWORD offset */
    volatile uint32_t *doorbellCpu;   /* 64-bit doorbell DWORD via BAR2 */
    void    *doorbellBarHandle;

    /* Queue identity for GRBM_SELECT. */
    uint32_t me;    /* 1 = MEC0 (compute) */
    uint32_t pipe;
    uint32_t queue;

    uint64_t wptr;  /* in DWORDs */

    /* NBIO base for HDP flush + doorbell aperture. */
    uint32_t nbifBase2;
    bool hasNbif;
};

/* integer log2 for power-of-two sizes (math.log2 in ring_init.py). */
static uint32_t cqLog2(uint32_t v)
{
    uint32_t r = 0;
    while (v > 1) { v >>= 1; r++; }
    return r;
}

/* GC register access: byte_offset = (gc_base[base_idx] + reg) * 4. */
static uint32_t gcReg(WddmLite &gpu, const CqState &cq, uint32_t reg, int baseIdx)
{
    uint32_t base = (baseIdx == 0) ? cq.gcBase0 : cq.gcBase1;
    uint32_t v = 0;
    gpu.readReg32((base + reg) * 4, &v);
    return v;
}

static void gcWreg(WddmLite &gpu, const CqState &cq, uint32_t reg,
                   uint32_t val, int baseIdx)
{
    uint32_t base = (baseIdx == 0) ? cq.gcBase0 : cq.gcBase1;
    gpu.writeReg32((base + reg) * 4, val);
}

static void gcWregPair(WddmLite &gpu, const CqState &cq, uint32_t regLo,
                       uint32_t regHi, uint64_t value, int baseIdx)
{
    gcWreg(gpu, cq, regLo, (uint32_t)(value & 0xFFFFFFFF), baseIdx);
    gcWreg(gpu, cq, regHi, (uint32_t)((value >> 32) & 0xFFFFFFFF), baseIdx);
}

/* grbm_select / grbm_deselect (ring_init.py, soc21_grbm_select). */
static void cqGrbmSelect(WddmLite &gpu, const CqState &cq, uint32_t me,
                         uint32_t pipe, uint32_t queue, uint32_t vmid)
{
    uint32_t val = 0;
    val |= (pipe & 0x3) << GRBM_GFX_CNTL__PIPEID__SHIFT;
    val |= (me & 0x3) << GRBM_GFX_CNTL__MEID__SHIFT;
    val |= (vmid & 0xF) << GRBM_GFX_CNTL__VMID__SHIFT;
    val |= (queue & 0x7) << GRBM_GFX_CNTL__QUEUEID__SHIFT;
    /* GRBM_GFX_CNTL is base_idx 1 on GC 12. */
    gpu.writeReg32((cq.gcBase1 + regGRBM_GFX_CNTL) * 4, val);
}

static void cqGrbmDeselect(WddmLite &gpu, const CqState &cq)
{
    cqGrbmSelect(gpu, cq, 0, 0, 0, 0);
}

/* _pulse_reset_bits (ring_init.py). */
static void cqPulseReset(WddmLite &gpu, const CqState &cq, uint32_t reg,
                         uint32_t mask, int baseIdx)
{
    uint32_t v = gcReg(gpu, cq, reg, baseIdx);
    gcWreg(gpu, cq, reg, v | mask, baseIdx);
    v = gcReg(gpu, cq, reg, baseIdx);
    gcWreg(gpu, cq, reg, v & ~mask, baseIdx);
}

/* rs64(name): read gc_<gc>_<name>.bin, struct.unpack_from("<II", blob, off)
 * -> entry = (hi << 32) | lo. RS64 (PFP/ME/MEC) read off=52; MES reads off=56
 * (probe16_vmid0.py / probe_kernel.py). Returns 0 on failure (treated as
 * "missing" by the caller, matching _config_mec_from_ucode). */
static uint64_t cqUcodeStart(const char *fwDir, const char *gc,
                             const char *name, size_t off, bool *ok)
{
    char path[512];
    snprintf(path, sizeof(path), "%s\\gc_%s_%s.bin", fwDir, gc, name);
    std::vector<uint8_t> blob;
    if (!loadFirmwareFile(path, blob)) {
        if (ok) *ok = false;
        return 0;
    }
    uint32_t lo = rcpU32(blob, off);
    uint32_t hi = rcpU32(blob, off + 4);
    if (ok) *ok = true;
    return ((uint64_t)hi << 32) | lo;
}

/* Resolve the NBIF base_idx 2 (nbio_init.py resolve_nbio_bases). Falls back to
 * OSSSYS if NBIF is not enumerated (matches the Python fallback). */
static bool cqResolveNbif(const IpDiscoveryResult &ipd, uint32_t *base2)
{
    bool found = false;
    uint32_t fallback = 0;
    bool haveFallback = false;
    for (uint32_t i = 0; i < ipd.numBlocks; i++) {
        const IpBlock &b = ipd.blocks[i];
        if (b.hwId == RCP_HWID_NBIF && b.instance == 0) {
            if (b.numBaseAddrs > 2) { *base2 = b.baseAddrs[2]; found = true; }
            break;
        }
        if (b.hwId == HWID_OSSSYS && b.instance == 0 && b.numBaseAddrs > 2) {
            fallback = b.baseAddrs[2];
            haveFallback = true;
        }
    }
    if (!found && haveFallback) { *base2 = fallback; found = true; }
    return found;
}

/* hdp_flush (nbio_init.py): write 0 to regHDP_MEM_COHERENCY_FLUSH (NBIF idx 2). */
static void cqHdpFlush(WddmLite &gpu, const CqState &cq)
{
    if (!cq.hasNbif) return;
    gpu.writeReg32((cq.nbifBase2 + regHDP_MEM_COHERENCY_FLUSH) * 4, 0);
}

/* _config_mec_from_ucode (ring_init.py): program CP PFP/ME/MEC program
 * counters from ucode_start, reset+unhalt PFP/ME, reset MEC pipes. */
static void cqConfigMecFromUcode(WddmLite &gpu, const CqState &cq,
                                 uint64_t pfp, uint64_t me, uint64_t mec)
{
    cqGrbmSelect(gpu, cq, 0, 0, 0, 0);
    gcWregPair(gpu, cq, regCP_PFP_PRGRM_CNTR_START,
               regCP_PFP_PRGRM_CNTR_START_HI, pfp >> 2, 0);
    gcWregPair(gpu, cq, regCP_ME_PRGRM_CNTR_START,
               regCP_ME_PRGRM_CNTR_START_HI, me >> 2, 0);
    cqGrbmDeselect(gpu, cq);

    cqPulseReset(gpu, cq, regCP_ME_CNTL,
                 CP_ME_CNTL__PFP_PIPE0_RESET | CP_ME_CNTL__ME_PIPE0_RESET, 1);
    uint32_t val = gcReg(gpu, cq, regCP_ME_CNTL, 1);
    val &= ~(CP_ME_CNTL__PFP_HALT | CP_ME_CNTL__ME_HALT);
    gcWreg(gpu, cq, regCP_ME_CNTL, val, 1);

    for (uint32_t pipe = 0; pipe < 4; pipe++) {
        cqGrbmSelect(gpu, cq, 1, pipe, 0, 0);
        gcWregPair(gpu, cq, regCP_MEC_RS64_PRGRM_CNTR_START,
                   regCP_MEC_RS64_PRGRM_CNTR_START_HI, mec >> 2, 1);
    }
    cqGrbmDeselect(gpu, cq);

    cqPulseReset(gpu, cq, regCP_MEC_RS64_CNTL,
                 CP_MEC_RS64_CNTL__MEC_PIPE0_RESET |
                 CP_MEC_RS64_CNTL__MEC_PIPE1_RESET |
                 CP_MEC_RS64_CNTL__MEC_PIPE2_RESET |
                 CP_MEC_RS64_CNTL__MEC_PIPE3_RESET, 1);
    printf("  GFX: CP PFP/ME/MEC program counters configured\n");
}

/* _enable_mec (ring_init.py): clear reset/halt/icache-inv, set all 4 pipes
 * active. Expected CP_MEC_RS64_CNTL readback = 0x3C000000. */
static void cqEnableMec(WddmLite &gpu, const CqState &cq)
{
    uint32_t val = gcReg(gpu, cq, regCP_MEC_RS64_CNTL, 1);
    val &= ~(CP_MEC_RS64_CNTL__MEC_INVALIDATE_ICACHE |
             CP_MEC_RS64_CNTL__MEC_PIPE0_RESET |
             CP_MEC_RS64_CNTL__MEC_PIPE1_RESET |
             CP_MEC_RS64_CNTL__MEC_PIPE2_RESET |
             CP_MEC_RS64_CNTL__MEC_PIPE3_RESET |
             CP_MEC_RS64_CNTL__MEC_HALT);
    val |= (CP_MEC_RS64_CNTL__MEC_PIPE0_ACTIVE |
            CP_MEC_RS64_CNTL__MEC_PIPE1_ACTIVE |
            CP_MEC_RS64_CNTL__MEC_PIPE2_ACTIVE |
            CP_MEC_RS64_CNTL__MEC_PIPE3_ACTIVE);
    gcWreg(gpu, cq, regCP_MEC_RS64_CNTL, val, 1);
    Sleep(50);
}

/* init_gfx_for_compute (ring_init.py), reduced to the PSP-autoload path used
 * by the probe: _rlc_backdoor_autoload is a no-op (PSP already autoloaded),
 * so this programs CP counters, RLC/SH_MEM/doorbell-range, and enables the MEC.
 * MES enable is NOT needed for the direct-MMIO compute HQD (probe does not use
 * MES for the queue; that is increment 2b/MES work). */
static bool cqInitGfxForCompute(WddmLite &gpu, CqState &cq,
                                uint64_t pfp, uint64_t me, uint64_t mec,
                                bool haveUcode)
{
    if (haveUcode)
        cqConfigMecFromUcode(gpu, cq, pfp, me, mec);
    else
        printf("  GFX: WARNING ucode_start missing; skipping CP program counters\n");

    uint32_t tcp = gcReg(gpu, cq, regTCP_CNTL, 1);
    gcWreg(gpu, cq, regTCP_CNTL, tcp | 0x20000000, 1);
    gcWreg(gpu, cq, regRLC_CNTL, 0x1, 1);
    uint32_t rlcSrm = gcReg(gpu, cq, regRLC_SRM_CNTL, 1);
    gcWreg(gpu, cq, regRLC_SRM_CNTL, rlcSrm | 0x3, 1);
    gcWreg(gpu, cq, regRLC_SPM_MC_CNTL, 0xF, 1);

    uint32_t grbmCntl = gcReg(gpu, cq, regGRBM_CNTL, 0);
    gcWreg(gpu, cq, regGRBM_CNTL, (grbmCntl & ~0xFFF) | 0xFF, 0);

    uint32_t shMemConfig = (3u << 2) | (3u << 14);
    uint32_t shMemBases = (1u << 16) | 2u;
    for (uint32_t vmid = 0; vmid < 16; vmid++) {
        cqGrbmSelect(gpu, cq, 0, 0, 0, vmid);
        gcWreg(gpu, cq, regSH_MEM_CONFIG, shMemConfig, 1);
        gcWreg(gpu, cq, regSH_MEM_BASES, shMemBases, 1);
    }
    cqGrbmDeselect(gpu, cq);

    gcWreg(gpu, cq, regCP_MEC_DOORBELL_RANGE_LOWER, 0, 0);
    gcWreg(gpu, cq, regCP_MEC_DOORBELL_RANGE_UPPER, (0x8A * 2) << 2, 0);

    cqEnableMec(gpu, cq);
    uint32_t mecCntl = gcReg(gpu, cq, regCP_MEC_RS64_CNTL, 1);
    printf("  GFX: MEC enabled (CP_MEC_RS64_CNTL=0x%08X)\n", mecCntl);
    return true;
}

/* _init_compute_mqd (ring_init.py): build the v12 compute MQD image. */
static void cqInitMqd(CqState &cq)
{
    volatile uint32_t *mqd = (volatile uint32_t *)cq.mqdCpu;
    memset(cq.mqdCpu, 0, MQD_SIZE_BYTES);

    mqd[0] = MQD_HEADER;
    mqd[1] = 1;                  /* compute_dispatch_initiator */
    mqd[11] = 1;                 /* compute_pipelinestat_enable */
    mqd[23] = 0xFFFFFFFF;        /* SE0 thread mgmt */
    mqd[24] = 0xFFFFFFFF;        /* SE1 */
    mqd[26] = 0xFFFFFFFF;        /* SE2 */
    mqd[27] = 0xFFFFFFFF;        /* SE3 */
    mqd[32] = 0x00000007;        /* compute_misc_reserved */

    mqd[128] = (uint32_t)(cq.mqdGpu & 0xFFFFFFFC);
    mqd[129] = (uint32_t)((cq.mqdGpu >> 32) & 0xFFFFFFFF);
    mqd[130] = 1;                /* cp_hqd_active */
    mqd[131] = 0;                /* cp_hqd_vmid = 0 */
    mqd[132] = (HQD_PERSISTENT_STATE_DEFAULT & ~(0x3FFu << 8)) |
               (HQD_PERSISTENT_STATE__PRELOAD_SIZE <<
                HQD_PERSISTENT_STATE__PRELOAD_SIZE__SHIFT) |
               HQD_PERSISTENT_STATE__PRELOAD_REQ;
    mqd[133] = 0x2;
    mqd[134] = 0xF;
    mqd[135] = 0x111;

    uint64_t pqBase = cq.ringGpu >> 8;
    mqd[136] = (uint32_t)(pqBase & 0xFFFFFFFF);
    mqd[137] = (uint32_t)((pqBase >> 32) & 0xFFFFFFFF);
    mqd[138] = 0;

    mqd[139] = (uint32_t)(cq.rptrGpu & 0xFFFFFFFC);
    mqd[140] = (uint32_t)((cq.rptrGpu >> 32) & 0xFFFF);
    mqd[141] = (uint32_t)(cq.wptrGpu & 0xFFFFFFF8);
    mqd[142] = (uint32_t)((cq.wptrGpu >> 32) & 0xFFFF);

    uint32_t doorbellCtrl = 0;
    doorbellCtrl |= (cq.doorbellIndex & 0x03FFFFFF) << HQD_DOORBELL_OFFSET__SHIFT;
    doorbellCtrl |= HQD_DOORBELL_EN;
    mqd[143] = doorbellCtrl;

    uint32_t ringSizeLog2 = cqLog2(cq.ringSize / 4) - 1;
    uint32_t pqControl = 0;
    pqControl |= (ringSizeLog2 & 0x3F) << PQ_CONTROL__QUEUE_SIZE__SHIFT;
    pqControl |= (5u & 0x3F) << PQ_CONTROL__RPTR_BLOCK_SIZE__SHIFT;
    pqControl |= PQ_CONTROL__PQ_EMPTY;
    pqControl |= 3u << PQ_CONTROL__MIN_AVAIL_SIZE__SHIFT;
    pqControl |= PQ_CONTROL__NO_UPDATE_RPTR;
    pqControl |= PQ_CONTROL__UNORD_DISPATCH;
    pqControl |= PQ_CONTROL__PRIV_STATE;
    pqControl |= PQ_CONTROL__KMD_QUEUE;
    mqd[145] = pqControl;

    mqd[162] = 1u << 8;          /* cp_mqd_control PRIV_STATE */

    uint64_t eopBase = cq.eopGpu >> 8;
    mqd[165] = (uint32_t)(eopBase & 0xFFFFFFFF);
    mqd[166] = (uint32_t)((eopBase >> 32) & 0xFFFFFFFF);
    mqd[167] = cqLog2(EOP_BUFFER_SIZE / 4) - 1;

    mqd[149] = 3u << 20;         /* cp_hqd_ib_control */
    mqd[160] = 0x20004000;       /* cp_hqd_hq_status0 */
    mqd[181] = 0;                /* cp_hqd_aql_control (non-AQL) */
    mqd[182] = 0;
    mqd[183] = 0;
    mqd[184] = 1u << 15;         /* reserved_184: unmapped doorbell handling */
}

/* _activate_compute_queue_mmio (ring_init.py): program CP_HQD_* directly under
 * grbm_select(me,pipe,queue) and activate. VMID 0 (cp_hqd_vmid &= ~0xF). */
static void cqActivateQueueMmio(WddmLite &gpu, CqState &cq)
{
    cqGrbmSelect(gpu, cq, cq.me, cq.pipe, cq.queue, 0);

    gcWreg(gpu, cq, regCP_HQD_ACTIVE, 0, 0);

    uint32_t vmid = gcReg(gpu, cq, regCP_HQD_VMID, 0);
    gcWreg(gpu, cq, regCP_HQD_VMID, vmid & ~0xFu, 0);

    uint32_t dctl = gcReg(gpu, cq, regCP_HQD_PQ_DOORBELL_CONTROL, 0);
    gcWreg(gpu, cq, regCP_HQD_PQ_DOORBELL_CONTROL, dctl & ~HQD_DOORBELL_EN, 0);

    gcWreg(gpu, cq, regCP_MQD_BASE_ADDR, (uint32_t)(cq.mqdGpu & 0xFFFFFFFC), 0);
    gcWreg(gpu, cq, regCP_MQD_BASE_ADDR_HI,
           (uint32_t)((cq.mqdGpu >> 32) & 0xFFFFFFFF), 0);

    gcWreg(gpu, cq, regCP_MQD_CONTROL, 0, 0);

    uint64_t pqBase = cq.ringGpu >> 8;
    gcWreg(gpu, cq, regCP_HQD_PQ_BASE, (uint32_t)(pqBase & 0xFFFFFFFF), 0);
    gcWreg(gpu, cq, regCP_HQD_PQ_BASE_HI,
           (uint32_t)((pqBase >> 32) & 0xFFFFFFFF), 0);

    gcWreg(gpu, cq, regCP_HQD_PQ_RPTR_REPORT_ADDR,
           (uint32_t)(cq.rptrGpu & 0xFFFFFFFC), 0);
    gcWreg(gpu, cq, regCP_HQD_PQ_RPTR_REPORT_ADDR_HI,
           (uint32_t)((cq.rptrGpu >> 32) & 0xFFFF), 0);

    uint32_t ringSizeLog2 = cqLog2(cq.ringSize / 4) - 1;
    uint32_t pqControl = 0;
    pqControl |= (ringSizeLog2 & 0x3F) << PQ_CONTROL__QUEUE_SIZE__SHIFT;
    pqControl |= (5u & 0x3F) << PQ_CONTROL__RPTR_BLOCK_SIZE__SHIFT;
    pqControl |= PQ_CONTROL__PQ_EMPTY;
    pqControl |= 3u << PQ_CONTROL__MIN_AVAIL_SIZE__SHIFT;
    pqControl |= PQ_CONTROL__NO_UPDATE_RPTR;
    pqControl |= PQ_CONTROL__UNORD_DISPATCH;
    pqControl |= PQ_CONTROL__PRIV_STATE;
    pqControl |= PQ_CONTROL__KMD_QUEUE;
    gcWreg(gpu, cq, regCP_HQD_PQ_CONTROL, pqControl, 0);

    gcWreg(gpu, cq, regCP_HQD_PQ_WPTR_POLL_ADDR,
           (uint32_t)(cq.wptrGpu & 0xFFFFFFF8), 0);
    gcWreg(gpu, cq, regCP_HQD_PQ_WPTR_POLL_ADDR_HI,
           (uint32_t)((cq.wptrGpu >> 32) & 0xFFFF), 0);

    gcWreg(gpu, cq, regCP_HQD_PQ_RPTR, 0, 0);
    gcWreg(gpu, cq, regCP_HQD_PQ_WPTR_LO, 0, 0);
    gcWreg(gpu, cq, regCP_HQD_PQ_WPTR_HI, 0, 0);

    uint32_t doorbellCtrl = 0;
    doorbellCtrl |= (cq.doorbellIndex & 0x03FFFFFF) << HQD_DOORBELL_OFFSET__SHIFT;
    doorbellCtrl |= HQD_DOORBELL_EN;
    gcWreg(gpu, cq, regCP_HQD_PQ_DOORBELL_CONTROL, doorbellCtrl, 0);

    uint32_t persistent = (HQD_PERSISTENT_STATE_DEFAULT & ~(0x3FFu << 8)) |
                          (HQD_PERSISTENT_STATE__PRELOAD_SIZE <<
                           HQD_PERSISTENT_STATE__PRELOAD_SIZE__SHIFT) |
                          HQD_PERSISTENT_STATE__PRELOAD_REQ;
    gcWreg(gpu, cq, regCP_HQD_PERSISTENT_STATE, persistent, 0);
    gcWreg(gpu, cq, regCP_HQD_PIPE_PRIORITY, 0x2, 0);
    gcWreg(gpu, cq, regCP_HQD_QUEUE_PRIORITY, 0xF, 0);
    gcWreg(gpu, cq, regCP_HQD_QUANTUM, 0x111, 0);
    gcWreg(gpu, cq, regCP_HQD_IB_CONTROL, 3u << 20, 0);
    gcWreg(gpu, cq, regCP_HQD_HQ_STATUS0, 0x20004000, 0);
    gcWreg(gpu, cq, regCP_HQD_AQL_CONTROL, 0, 0);

    uint64_t eopBase = cq.eopGpu >> 8;
    gcWreg(gpu, cq, regCP_HQD_EOP_BASE_ADDR, (uint32_t)(eopBase & 0xFFFFFFFF), 0);
    gcWreg(gpu, cq, regCP_HQD_EOP_BASE_ADDR_HI,
           (uint32_t)((eopBase >> 32) & 0xFFFFFFFF), 0);
    gcWreg(gpu, cq, regCP_HQD_EOP_CONTROL, cqLog2(EOP_BUFFER_SIZE / 4) - 1, 0);

    cqHdpFlush(gpu, cq);

    gcWreg(gpu, cq, regCP_HQD_ACTIVE, 1, 0);

    cqGrbmDeselect(gpu, cq);
}

/* init_compute_queue (ring_init.py), direct-MMIO path. Allocates VRAM-backed
 * ring/MQD/EOP/wptr/rptr/fence buffers, maps the doorbell via BAR2, builds the
 * MQD and activates the HQD. */
static bool cqInitComputeQueue(WddmLite &gpu, CqState &cq)
{
    cq.ringSize = COMPUTE_RING_SIZE;
    cq.me = 1;            /* MEC0 */
    cq.pipe = 0;
    cq.queue = 0;
    cq.doorbellIndex = DOORBELL_MEC_RING_START +
                       (cq.pipe * 4 + cq.queue) * DOORBELL_MEC_RING_STRIDE;
    cq.wptr = 0;

    void *cpu = nullptr;
    if (!rcpAllocVram(gpu, cq.ringSize, &cpu, &cq.ringGpu, &cq.ringHandle))
        return false;
    cq.ringCpu = cpu;
    if (!rcpAllocVram(gpu, 4096, &cq.mqdCpu, &cq.mqdGpu, &cq.mqdHandle))
        return false;
    if (!rcpAllocVram(gpu, EOP_BUFFER_SIZE, &cq.eopCpu, &cq.eopGpu, &cq.eopHandle))
        return false;
    if (!rcpAllocVram(gpu, 4096, &cpu, &cq.wptrGpu, &cq.wptrHandle))
        return false;
    cq.wptrCpu = (volatile uint32_t *)cpu;
    if (!rcpAllocVram(gpu, 4096, &cpu, &cq.rptrGpu, &cq.rptrHandle))
        return false;
    cq.rptrCpu = (volatile uint32_t *)cpu;
    if (!rcpAllocVram(gpu, 4096, &cpu, &cq.fenceGpu, &cq.fenceHandle))
        return false;
    cq.fenceCpu = (volatile uint64_t *)cpu;

    /* Doorbell: BAR2 at byte offset doorbell_index * 4 (nbio/ring_init.py:
     * MAP_BAR(2, idx*4, 8)). */
    void *dbAddr = nullptr;
    cq.doorbellCpu = nullptr;
    cq.doorbellBarHandle = nullptr;
    if (gpu.mapBar(2, (uint64_t)cq.doorbellIndex * 4, 8, &dbAddr,
                   &cq.doorbellBarHandle) && dbAddr) {
        cq.doorbellCpu = (volatile uint32_t *)dbAddr;
    } else {
        printf("  Compute: WARNING doorbell BAR2 map failed "
               "(index=0x%X) -- doorbell write will be skipped\n",
               cq.doorbellIndex);
    }

    cqInitMqd(cq);
    cqActivateQueueMmio(gpu, cq);

    printf("  Compute: Queue ME=%u pipe=%u queue=%u activated via direct MMIO\n",
           cq.me, cq.pipe, cq.queue);
    printf("  Compute: Ring MC=0x%012llX size=%uKB\n",
           (unsigned long long)cq.ringGpu, cq.ringSize / 1024);
    printf("  Compute: Doorbell index=0x%X cpu=%p\n",
           cq.doorbellIndex, (void *)cq.doorbellCpu);
    return true;
}

/* submit_compute_packets (ring_init.py): write packets to the ring, advance
 * wptr, wptr-writeback, HDP flush, MMIO wptr write (LITE_NO_MMIO_WPTR=0 on this
 * KMD), then ring the doorbell. */
static void cqSubmitPackets(WddmLite &gpu, CqState &cq,
                            const std::vector<uint32_t> &packets)
{
    uint32_t ringMask = cq.ringSize - 1;             /* byte mask */
    uint32_t byteOffset = (cq.wptr * 4) & ringMask;
    uint32_t bytes = (uint32_t)(packets.size() * 4);

    uint8_t *ring = (uint8_t *)cq.ringCpu;
    uint32_t spaceToEnd = cq.ringSize - byteOffset;
    if (bytes <= spaceToEnd) {
        memcpy(ring + byteOffset, packets.data(), bytes);
    } else {
        memcpy(ring + byteOffset, packets.data(), spaceToEnd);
        memcpy(ring, (const uint8_t *)packets.data() + spaceToEnd,
               bytes - spaceToEnd);
    }

    cq.wptr += (uint32_t)packets.size();

    /* wptr writeback (64-bit). */
    *(volatile uint64_t *)cq.wptrCpu = cq.wptr;

    cqHdpFlush(gpu, cq);

    /* MMIO wptr write under grbm_select (LITE_NO_MMIO_WPTR=0). */
    cqGrbmSelect(gpu, cq, cq.me, cq.pipe, cq.queue, 0);
    gcWreg(gpu, cq, regCP_HQD_PQ_WPTR_LO, cq.wptr & 0xFFFFFFFF, 0);
    gcWreg(gpu, cq, regCP_HQD_PQ_WPTR_HI, (cq.wptr >> 32) & 0xFFFFFFFF, 0);
    cqGrbmDeselect(gpu, cq);

    /* Ring the doorbell (64-bit write to the doorbell BAR). */
    if (cq.doorbellCpu)
        *(volatile uint64_t *)cq.doorbellCpu = cq.wptr;
}

/* wait_fence (ring_init.py): poll the 64-bit fence buffer until >= expected. */
static bool cqWaitFence(CqState &cq, uint64_t expected, int timeoutMs)
{
    int waited = 0;
    while (waited < timeoutMs) {
        if (*cq.fenceCpu >= expected)
            return true;
        Sleep(1);
        waited += 1;
    }
    return false;
}

/* PM4 helpers (pm4.py). Append a Type-3 packet: header + payload. */
static void pm4Pkt3(std::vector<uint32_t> &dw, uint32_t opcode,
                    const uint32_t *payload, uint32_t n)
{
    uint32_t header = (3u << 30) | (((n - 1) & 0x3FFF) << 16) | (opcode << 8);
    dw.push_back(header);
    for (uint32_t i = 0; i < n; i++)
        dw.push_back(payload[i]);
}

static void pm4Nop(std::vector<uint32_t> &dw, uint32_t count)
{
    for (uint32_t i = 0; i < count; i++) {
        uint32_t zero = 0;
        pm4Pkt3(dw, PACKET3_NOP, &zero, 1);
    }
}

/* RELEASE_MEM fence with cache_flush=True, use_gcr=True (pm4.py defaults from
 * test_compute_nop_fence). 7-dword body. */
static void pm4ReleaseMemFence(std::vector<uint32_t> &dw, uint64_t addr,
                               uint64_t value)
{
    uint32_t dw0 = (EVENT_TYPE_CACHE_FLUSH_AND_INV_TS_EVENT & 0x3F) |
                   ((RELEASE_MEM_EVENT_INDEX_EOP & 0xF) << 8);
    dw0 |= PACKET3_RELEASE_MEM_GCR_GLV_INV |
           PACKET3_RELEASE_MEM_GCR_GL1_INV |
           PACKET3_RELEASE_MEM_GCR_GL2_INV |
           PACKET3_RELEASE_MEM_GCR_GLM_WB |
           PACKET3_RELEASE_MEM_GCR_GLM_INV |
           PACKET3_RELEASE_MEM_GCR_GL2_WB |
           PACKET3_RELEASE_MEM_GCR_SEQ;
    uint32_t dw1 = ((DATA_SEL_SEND_64BIT & 0x7) << 29) |
                   ((INT_SEL_SEND_INT_ON_CONFIRM & 0x3) << 24);
    uint32_t payload[7];
    payload[0] = dw0;
    payload[1] = dw1;
    payload[2] = (uint32_t)(addr & 0xFFFFFFFF);
    payload[3] = (uint32_t)((addr >> 32) & 0xFFFFFFFF);
    payload[4] = (uint32_t)(value & 0xFFFFFFFF);
    payload[5] = (uint32_t)((value >> 32) & 0xFFFFFFFF);
    payload[6] = 0;   /* ctxid */
    pm4Pkt3(dw, PACKET3_RELEASE_MEM, payload, 7);
}

/* ---- Top-level: recipeNopFence -------------------------------------------- */

bool recipeNopFence(WddmLite &gpu, const IpDiscoveryResult &ipd,
                    const char *fwDir, uint64_t vramMcBase)
{
    /* 1. PSP cold-boot autoload -> BOOTLOAD_COMPLETE. recipeBootload owns the
     * PSP ring/cmd/fence buffers; those are not needed after BOOTLOAD for the
     * direct-MMIO compute path (the HQD reads its own ring/MQD), so we do not
     * persist its RecipePspState. */
    if (!recipeBootload(gpu, ipd, fwDir, vramMcBase)) {
        printf("  NOP: recipeBootload did not reach BOOTLOAD_COMPLETE\n");
        return false;
    }

    printf("\n=== recipeNopFence (MEC + compute HQD + NOP/RELEASE_MEM) ===\n");

    CqState cq;
    memset(&cq, 0, sizeof(cq));
    cq.gcBase0 = ipd.gcBase;
    cq.gcBase1 = ipd.gcBase1;

    cq.hasNbif = cqResolveNbif(ipd, &cq.nbifBase2);
    if (cq.hasNbif) {
        /* init_nbio (nbio_init.py): enable the doorbell aperture + framebuffer
         * so the doorbell write reaches the CP. */
        uint32_t apEn = 0;
        gpu.readReg32((cq.nbifBase2 + regRCC_DOORBELL_APER_EN) * 4, &apEn);
        gpu.writeReg32((cq.nbifBase2 + regRCC_DOORBELL_APER_EN) * 4,
                       apEn | BIF_DOORBELL_APER_EN__BIT);
        uint32_t fbEn = 0;
        gpu.readReg32((cq.nbifBase2 + regBIF_FB_EN) * 4, &fbEn);
        gpu.writeReg32((cq.nbifBase2 + regBIF_FB_EN) * 4,
                       fbEn | BIF_FB_EN__FB_READ_EN | BIF_FB_EN__FB_WRITE_EN);
        printf("  NBIO: doorbell aperture + framebuffer enabled "
               "(NBIF base[2]=0x%04X)\n", cq.nbifBase2);
    } else {
        printf("  NBIO: WARNING NBIF base[2] not found; HDP flush + doorbell "
               "aperture skipped\n");
    }

    /* ucode_start: rs64() for PFP/ME/MEC (gfx-header entry @52). */
    const char *gc = "12_0_1";
    bool okP = false, okM = false, okC = false;
    uint64_t pfp = cqUcodeStart(fwDir, gc, "pfp", 52, &okP);
    uint64_t me = cqUcodeStart(fwDir, gc, "me", 52, &okM);
    uint64_t mec = cqUcodeStart(fwDir, gc, "mec", 52, &okC);
    bool haveUcode = okP && okM && okC;
    if (haveUcode)
        printf("  GFX: ucode_start PFP=0x%llX ME=0x%llX MEC=0x%llX\n",
               (unsigned long long)pfp, (unsigned long long)me,
               (unsigned long long)mec);

    /* 2. init_gfx_for_compute (MEC enable). */
    if (!cqInitGfxForCompute(gpu, cq, pfp, me, mec, haveUcode))
        return false;
    uint32_t mecCntl = gcReg(gpu, cq, regCP_MEC_RS64_CNTL, 1);

    /* 3. init_compute_queue (direct-MMIO HQD, VMID 0). */
    if (!cqInitComputeQueue(gpu, cq))
        return false;

    /* 4. NOP + RELEASE_MEM fence. */
    uint64_t fenceSeq = 1;
    *cq.fenceCpu = 0;
    std::vector<uint32_t> packets;
    pm4Nop(packets, 4);
    pm4ReleaseMemFence(packets, cq.fenceGpu, fenceSeq);
    cqSubmitPackets(gpu, cq, packets);

    bool ok = cqWaitFence(cq, fenceSeq, 5000);

    /* Read rptr via grbm_select(me=1,pipe=0,queue=0) (probe16_vmid0.py). */
    cqGrbmSelect(gpu, cq, cq.me, cq.pipe, cq.queue, 0);
    uint32_t rptr = gcReg(gpu, cq, regCP_HQD_PQ_RPTR, 0);
    cqGrbmDeselect(gpu, cq);
    uint64_t fenceVal = *cq.fenceCpu;

    printf("\nCP_MEC_RS64_CNTL=0x%08X (expected 0x3C000000)\n", mecCntl);
    printf("FENCE value=%llu (expected %llu)\n",
           (unsigned long long)fenceVal, (unsigned long long)fenceSeq);
    printf("RPTR=0x%X (wptr=%u)\n", rptr, cq.wptr);
    printf("NOP+FENCE %s\n", ok ? "PASS" : "FAIL (fence timeout)");
    return ok;
}

/* ======================================================================
 * Increment 2b: single s_endpgm compute dispatch (GPUVM + DISPATCH_DIRECT)
 *
 * Faithful C++ transcription of the proven probe path:
 *   python/amd_gpu_driver/probe16_vmid0.py (THE authoritative VMID-0 dispatch
 *     that PASSES on this gfx1201): build a 4-level GFXHUB page table in VRAM,
 *     enable GCVM_CONTEXT0 AFTER the autoload, dispatch.
 *   python/amd_gpu_driver/backends/windows/gmc_init.py: gfxhub_gart_enable,
 *     build_compute_gpuvm / _program_compute_gpuvm, _gfxhub_wreg, flush_gpu_tlb.
 *   python/amd_gpu_driver/backends/windows/compute_dispatch.py:
 *     _build_noop_kernel_image (16x s_endpgm = 0xBF810000),
 *     _build_dispatch_packets, test_noop_dispatch (fault-status verify).
 *   python/amd_gpu_driver/commands/pm4.py: acquire_mem (gfx12 GCR 7-dword),
 *     set_sh_reg, dispatch_direct, event_write.
 *
 * GFXHUB register addressing == GC base_idx 0 (the probe reads
 * _gc_reg(dev, gc, C8_CNTL, base_idx=0) and _gfxhub_wreg writes
 * config.gfxhub_base[0] == ipd.gcBase). So all GFXHUB writes here go through
 * gcWreg(..., baseIdx=0) just like CP_HQD_* programming.
 * ====================================================================== */

/* ---- GFXHUB / GCVM register DWORD offsets (gc_12_0_0_offset.h, gmc_init.py).
 * base_idx 0 (GC base[0]). ------------------------------------------------ */

/* GCMC system aperture / TLB */
#define regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB  0x15A8
#define regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB  0x15A9
#define regGCMC_VM_AGP_TOP                           0x1616
#define regGCMC_VM_AGP_BOT                           0x1617
#define regGCMC_VM_AGP_BASE                          0x1618
#define regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR          0x1619
#define regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR         0x161A
#define regGCMC_VM_MX_L1_TLB_CNTL                    0x161B

/* GCVM L2 cache / fault */
#define regGCVM_L2_CNTL                              0x15C4
#define regGCVM_L2_CNTL2                             0x15C5
#define regGCVM_L2_CNTL3                             0x15C6
#define regGCVM_L2_CNTL5                             0x15E3
#define regGCVM_L2_PROTECTION_FAULT_STATUS          0x15D0
#define regGCVM_L2_PROTECTION_FAULT_ADDR_LO32       0x15D2
#define regGCVM_L2_PROTECTION_FAULT_ADDR_HI32       0x15D3
#define regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32 0x15D4
#define regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32 0x15D5
#define regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32  0x15D7
#define regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32  0x15D8
#define regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32 0x15D9
#define regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32 0x15DA
#define regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32     0x15DB
#define regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32     0x15DC

/* GCVM contexts (VMID 0 == CONTEXT0; VMID 1..15 == CONTEXT1..15). */
#define regGCVM_CONTEXT0_CNTL                        0x1624
#define regGCVM_CONTEXT1_CNTL                        0x1625
#define regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32   0x168F
#define regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32   0x1690
#define regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32  0x16AF
#define regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32  0x16B0
#define regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32    0x16CF
#define regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32    0x16D0
#define regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32  0x16B1
#define regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32    0x16D1

/* GCVM invalidation engine 0 (flush_gpu_tlb gfxhub). */
#define regGCVM_INVALIDATE_ENG0_SEM                  0x1635
#define regGCVM_INVALIDATE_ENG0_REQ                  0x1647
#define regGCVM_INVALIDATE_ENG0_ACK                  0x1659
#define regGCVM_INVALIDATE_ENG0_ADDR_RANGE_LO32      0x166B

/* CP debug (disable UTCL1 error halt for GFXHUB). */
#define regCP_DEBUG                                  0x1E1F

#define GFXHUB_CTX_DISTANCE       1
#define GFXHUB_CTX_ADDR_DISTANCE  2
#define GFXHUB_ENG_ADDR_DISTANCE  2

/* L1 TLB control bits (gmc_init.py). */
#define L1_TLB_ENABLE                       (1u << 0)
#define L1_TLB_SYSTEM_ACCESS_MODE_MASK      (0x3u << 3)
#define L1_TLB_ENABLE_ADV_DRIVER_MODEL      (1u << 6)
#define L1_TLB_SYSTEM_APERTURE_UNMAPPED_ACCESS (1u << 5)

#define VM_CONTEXT_ENABLE_CONTEXT           (1u << 0)

/* gfx12 GPUVM PTE/PDE flags (gmc_init.py). */
#define AMDGPU_PTE_VALID        (1ull << 0)
#define AMDGPU_PTE_EXECUTABLE   (1ull << 4)
#define AMDGPU_PTE_READABLE     (1ull << 5)
#define AMDGPU_PTE_WRITEABLE    (1ull << 6)
#define AMDGPU_PTE_IS_PTE       (1ull << 63)
/* leaf PTE flag word = VALID|EXEC|READ|WRITE|IS_PTE (== 0x8000000000000071). */
#define GPUVM_LEAF_FLAGS  (AMDGPU_PTE_IS_PTE | AMDGPU_PTE_WRITEABLE | \
                           AMDGPU_PTE_READABLE | AMDGPU_PTE_EXECUTABLE | \
                           AMDGPU_PTE_VALID)
/* 0-based VRAM offset mask used for PDE/PTE address fields. */
#define GPUVM_ADDR_MASK   0x0000FFFFFFFFF000ull

/* depth-3 CONTEXT0_CNTL: ENABLE | PAGE_TABLE_DEPTH=3 | all fault-enable bits. */
#define GCVM_CONTEXT0_CNTL_DEPTH3  0x03FFFC07u

/* compute dispatch (probe16_vmid0.py / compute_dispatch.py). */
#define COMPUTE_VA_ROOT          0x200000000000ull   /* shader virtual address */
#define NOOP_SHADER_DWORD        0xBF810000u          /* s_endpgm (SOPP op 1) */
#define NOOP_RSRC1               0xE00C0003u
#define NOOP_RSRC2               0x00000080u          /* ENABLE_SGPR_WORKGROUP_ID_X */
#define DISPATCH_INITIATOR_W32   0x8045u              /* SHADER_EN|FORCE_000|ORDER|W32 */

/* SET_SH_REG base + COMPUTE_* SH register addresses (registers.py). */
#define SH_REG_BASE              0x2C00
#define regCOMPUTE_START_X       0x2E04
#define regCOMPUTE_PGM_LO        0x2E0C
#define regCOMPUTE_DISPATCH_SCRATCH_BASE_LO 0x2E10  /* arch flat scratch VA>>8 */
#define regCOMPUTE_DISPATCH_SCRATCH_BASE_HI 0x2E11  /* (HI takes the top 8 bits) */
#define regCOMPUTE_PGM_RSRC1     0x2E12
#define regCOMPUTE_PGM_RSRC2     0x2E13  /* SCRATCH_EN = bit 0 */
#define regCOMPUTE_RESOURCE_LIMITS 0x2E15
#define regCOMPUTE_TMPRING_SIZE  0x2E18
#define regCOMPUTE_RESTART_X     0x2E1B
#define regCOMPUTE_PGM_RSRC3     0x2E28
#define regCOMPUTE_USER_DATA_0   0x2E40

/* PM4 opcodes (pm4.py) extended for dispatch. */
#define PACKET3_DISPATCH_DIRECT  0x15
#define PACKET3_EVENT_WRITE      0x46
#define PACKET3_ACQUIRE_MEM      0x58
#define PACKET3_SET_SH_REG       0x76

/* ACQUIRE_MEM GCR_CNTL full invalidate (gfx12; == 0xC3F1). */
#define ACQUIRE_MEM_GCR_CNTL_FULL_INVALIDATE  0x0000C3F1u

/* EVENT_WRITE: CS_PARTIAL_FLUSH event type / index. */
#define CS_PARTIAL_FLUSH                7
#define EVENT_INDEX_CS_PARTIAL_FLUSH    4

/* ---- PM4 builder extensions (pm4.py) -------------------------------------- */

/* acquire_mem (gfx12 GCR 7-dword form): CP_COHER_CNTL=0, COHER_SIZE lo/hi
 * (full = 0xFFFFFFFFFFFFFFFF), COHER_BASE lo/hi = 0, POLL_INTERVAL = 0,
 * GCR_CNTL = full invalidate. */
static void pm4AcquireMem(std::vector<uint32_t> &dw)
{
    uint32_t payload[7];
    payload[0] = 0;                 /* CP_COHER_CNTL = 0 (gfx10+) */
    payload[1] = 0xFFFFFFFF;        /* COHER_SIZE lo */
    payload[2] = 0xFFFFFFFF;        /* COHER_SIZE hi */
    payload[3] = 0;                 /* COHER_BASE lo */
    payload[4] = 0;                 /* COHER_BASE hi */
    payload[5] = 0;                 /* POLL_INTERVAL */
    payload[6] = ACQUIRE_MEM_GCR_CNTL_FULL_INVALIDATE;  /* GCR_CNTL */
    pm4Pkt3(dw, PACKET3_ACQUIRE_MEM, payload, 7);
}

/* set_sh_reg(offset, *values): offset is reg - SH_REG_BASE; payload is the
 * offset dword followed by the values (matches PM4PacketBuilder.set_sh_reg with
 * a raw reg >= SH_REG_BASE). */
static void pm4SetShReg(std::vector<uint32_t> &dw, uint32_t reg,
                        const uint32_t *values, uint32_t n)
{
    std::vector<uint32_t> payload;
    payload.reserve(n + 1);
    payload.push_back(reg - SH_REG_BASE);
    for (uint32_t i = 0; i < n; i++)
        payload.push_back(values[i]);
    pm4Pkt3(dw, PACKET3_SET_SH_REG, payload.data(), (uint32_t)payload.size());
}

/* dispatch_direct(dim_x, dim_y, dim_z, initiator). */
static void pm4DispatchDirect(std::vector<uint32_t> &dw, uint32_t dimX,
                              uint32_t dimY, uint32_t dimZ, uint32_t initiator)
{
    uint32_t payload[4] = { dimX, dimY, dimZ, initiator };
    pm4Pkt3(dw, PACKET3_DISPATCH_DIRECT, payload, 4);
}

/* event_write(event_type, event_index). */
static void pm4EventWrite(std::vector<uint32_t> &dw, uint32_t eventType,
                          uint32_t eventIndex)
{
    uint32_t dw0 = (eventType & 0x3F) | ((eventIndex & 0xF) << 8);
    pm4Pkt3(dw, PACKET3_EVENT_WRITE, &dw0, 1);
}

/* ---- GFXHUB GART enable (gmc_init.py gfxhub_gart_enable) ------------------- */

/* GMC parameters the GFXHUB bring-up needs that GmcState already holds plus the
 * two DMA buffers the system aperture / fault default address point at. */
struct GfxhubParams {
    uint64_t vramStart;
    uint64_t vramEnd;
    uint64_t gartStart;
    uint64_t gartEnd;
    uint64_t agpStart;
    uint64_t agpEnd;
    uint64_t gartTableBus;   /* DMA bus address of the GART page table */
    uint64_t dummyPageBus;   /* DMA bus address of the fault/dummy page */
};

static void gfxhubGartEnable(WddmLite &gpu, const CqState &cq,
                             const GfxhubParams &p)
{
    /* _gfxhub_init_gart_aperture: VMID0 GART base/start/end. The compute path
     * overwrites CONTEXT0 below with the depth-3 page table, but mirror the
     * probe's full enable so the rest of the GFXHUB state matches amdgpu. */
    uint64_t ptBase = p.gartTableBus | AMDGPU_PTE_VALID;
    gcWreg(gpu, cq, regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32,
           (uint32_t)(ptBase & 0xFFFFFFFF), 0);
    gcWreg(gpu, cq, regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32,
           (uint32_t)((ptBase >> 32) & 0xFFFFFFFF), 0);
    gcWreg(gpu, cq, regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32,
           (uint32_t)((p.gartStart >> 12) & 0xFFFFFFFF), 0);
    gcWreg(gpu, cq, regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32,
           (uint32_t)((p.gartStart >> 44) & 0xFFFFFFFF), 0);
    gcWreg(gpu, cq, regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32,
           (uint32_t)((p.gartEnd >> 12) & 0xFFFFFFFF), 0);
    gcWreg(gpu, cq, regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32,
           (uint32_t)((p.gartEnd >> 44) & 0xFFFFFFFF), 0);

    /* _gfxhub_init_system_aperture. */
    gcWreg(gpu, cq, regGCMC_VM_AGP_BASE, 0, 0);
    gcWreg(gpu, cq, regGCMC_VM_AGP_BOT, (uint32_t)(p.agpStart >> 24), 0);
    gcWreg(gpu, cq, regGCMC_VM_AGP_TOP, (uint32_t)(p.agpEnd >> 24), 0);
    gcWreg(gpu, cq, regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR,
           (uint32_t)(p.vramStart >> 18), 0);
    gcWreg(gpu, cq, regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR,
           (uint32_t)(p.vramEnd >> 18), 0);
    gcWreg(gpu, cq, regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB,
           (uint32_t)(p.dummyPageBus >> 12), 0);
    gcWreg(gpu, cq, regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB,
           (uint32_t)(p.dummyPageBus >> 44), 0);
    gcWreg(gpu, cq, regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32,
           (uint32_t)(p.dummyPageBus >> 12), 0);
    gcWreg(gpu, cq, regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32,
           (uint32_t)(p.dummyPageBus >> 44), 0);

    /* _gfxhub_init_tlb. */
    uint32_t l1 = gcReg(gpu, cq, regGCMC_VM_MX_L1_TLB_CNTL, 0);
    l1 |= L1_TLB_ENABLE;
    l1 = (l1 & ~L1_TLB_SYSTEM_ACCESS_MODE_MASK) | (3u << 3);
    l1 |= L1_TLB_ENABLE_ADV_DRIVER_MODEL;
    l1 &= ~L1_TLB_SYSTEM_APERTURE_UNMAPPED_ACCESS;
    gcWreg(gpu, cq, regGCMC_VM_MX_L1_TLB_CNTL, l1, 0);

    /* _gfxhub_init_cache. */
    uint32_t l2 = gcReg(gpu, cq, regGCVM_L2_CNTL, 0);
    l2 |= (1u << 0);
    l2 |= (1u << 8);
    l2 &= ~(1u << 6);
    gcWreg(gpu, cq, regGCVM_L2_CNTL, l2, 0);
    gcWreg(gpu, cq, regGCVM_L2_CNTL2, (1u << 0) | (1u << 1), 0);
    uint32_t l2c3 = gcReg(gpu, cq, regGCVM_L2_CNTL3, 0);
    l2c3 = (l2c3 & ~0x3F000u) | (9u << 15);
    l2c3 = (l2c3 & ~0x1F00000u) | (6u << 20);
    gcWreg(gpu, cq, regGCVM_L2_CNTL3, l2c3, 0);
    uint32_t l2c5 = gcReg(gpu, cq, regGCVM_L2_CNTL5, 0);
    l2c5 &= ~0x7E0u;
    gcWreg(gpu, cq, regGCVM_L2_CNTL5, l2c5, 0);

    /* _gfxhub_enable_system_domain (VMID 0 base enable; depth set later). */
    gcWreg(gpu, cq, regGCVM_CONTEXT0_CNTL, VM_CONTEXT_ENABLE_CONTEXT, 0);

    /* _gfxhub_disable_identity_aperture. */
    gcWreg(gpu, cq, regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32,
           0xFFFFFFFF, 0);
    gcWreg(gpu, cq, regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32,
           0x0000000F, 0);
    gcWreg(gpu, cq, regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32, 0, 0);
    gcWreg(gpu, cq, regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32, 0, 0);
    gcWreg(gpu, cq, regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32, 0, 0);
    gcWreg(gpu, cq, regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32, 0, 0);

    /* _gfxhub_setup_vmid_config: VMIDs 1..15 (num_level=3, block_size=9). */
    uint64_t maxPfn = (1ull << 36) - 1;
    for (uint32_t vmid = 1; vmid < 16; vmid++) {
        uint32_t val = VM_CONTEXT_ENABLE_CONTEXT;
        val |= (3u & 0x3u) << 1;                       /* num_level = 3 */
        val |= (1u << 7) | (1u << 8) | (1u << 9) | (1u << 10);
        val |= (1u << 11) | (1u << 12) | (1u << 13);
        /* block_size - 9 == 0. */
        uint32_t ctxReg = regGCVM_CONTEXT1_CNTL + (vmid - 1) * GFXHUB_CTX_DISTANCE;
        gcWreg(gpu, cq, ctxReg, val, 0);

        uint32_t startLo = regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32 +
                           (vmid - 1) * GFXHUB_CTX_ADDR_DISTANCE;
        gcWreg(gpu, cq, startLo, 0, 0);
        gcWreg(gpu, cq, startLo + 1, 0, 0);

        uint32_t endLo = regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32 +
                         (vmid - 1) * GFXHUB_CTX_ADDR_DISTANCE;
        gcWreg(gpu, cq, endLo, (uint32_t)(maxPfn & 0xFFFFFFFF), 0);
        gcWreg(gpu, cq, endLo + 1, (uint32_t)((maxPfn >> 32) & 0xFFFFFFFF), 0);
    }

    /* _gfxhub_program_invalidation: engines 0..17. */
    for (uint32_t eng = 0; eng < 18; eng++) {
        uint32_t loReg = regGCVM_INVALIDATE_ENG0_ADDR_RANGE_LO32 +
                         eng * GFXHUB_ENG_ADDR_DISTANCE;
        gcWreg(gpu, cq, loReg, 0xFFFFFFFF, 0);
        gcWreg(gpu, cq, loReg + 1, 0x1F, 0);
    }

    /* Disable CP UTCL1 error halt (CPG_UTCL1_ERROR_HALT_DISABLE). */
    uint32_t cpDbg = gcReg(gpu, cq, regCP_DEBUG, 0);
    cpDbg |= (1u << 15);
    gcWreg(gpu, cq, regCP_DEBUG, cpDbg, 0);

    printf("  GFXHUB: gart_enable done (VMID0 base + TLB/L2 + VMID1-15 + inv)\n");
}

/* flush_gpu_tlb(hub="gfxhub", vmid) (gmc_init.py): engine 0 invalidate. */
static void gfxhubFlushTlb(WddmLite &gpu, const CqState &cq, uint32_t vmid)
{
    /* Acquire semaphore. */
    for (int i = 0; i < 10; i++) {
        uint32_t v = gcReg(gpu, cq, regGCVM_INVALIDATE_ENG0_SEM, 0);
        if (v & 0x1) break;
        gcWreg(gpu, cq, regGCVM_INVALIDATE_ENG0_SEM, 1, 0);
    }
    /* Request: PER_VMID_INVALIDATE_REQ | (vmid << 16). */
    uint32_t req = (1u << 0) | ((vmid & 0xF) << 16);
    gcWreg(gpu, cq, regGCVM_INVALIDATE_ENG0_REQ, req, 0);
    /* Poll for completion. */
    for (int i = 0; i < 100; i++) {
        uint32_t ack = gcReg(gpu, cq, regGCVM_INVALIDATE_ENG0_ACK, 0);
        if (ack & (1u << vmid)) break;
    }
    /* Release semaphore. */
    gcWreg(gpu, cq, regGCVM_INVALIDATE_ENG0_SEM, 0, 0);
}

/* ---- 4-level compute GPUVM page table (build_compute_gpuvm) ---------------- */

struct DispatchGpuvm {
    void    *pdb2Cpu; uint64_t pdb2Gpu; uint64_t pdb2Handle;
    void    *pdb1Cpu; uint64_t pdb1Gpu; uint64_t pdb1Handle;
    void    *pdb0Cpu; uint64_t pdb0Gpu; uint64_t pdb0Handle;
    void    *ptbCpu;  uint64_t ptbGpu;  uint64_t ptbHandle;
};

/* Build the 4-level page table (PDB2->PDB1->PDB0->PTB), map COMPUTE_VA_ROOT ->
 * codeGpu, then enable GCVM_CONTEXT0 = depth-3 and flush the GFXHUB TLB.
 * Entries are 0-BASED VRAM OFFSETS (gpu_addr - vramMcBase), matching amdgpu's
 * CTX page-table base form. Must run AFTER gfxhubGartEnable (which re-inits the
 * hub post-autoload) and BEFORE the queue (probe16 ordering). */
static bool buildComputeGpuvm(WddmLite &gpu, const CqState &cq,
                              uint64_t vramMcBase, uint64_t codeGpu,
                              DispatchGpuvm &pt)
{
    void *cpu = nullptr;
    if (!rcpAllocVram(gpu, 4096, &cpu, &pt.pdb2Gpu, &pt.pdb2Handle)) return false;
    pt.pdb2Cpu = cpu;
    if (!rcpAllocVram(gpu, 4096, &cpu, &pt.pdb1Gpu, &pt.pdb1Handle)) return false;
    pt.pdb1Cpu = cpu;
    if (!rcpAllocVram(gpu, 4096, &cpu, &pt.pdb0Gpu, &pt.pdb0Handle)) return false;
    pt.pdb0Cpu = cpu;
    if (!rcpAllocVram(gpu, 4096, &cpu, &pt.ptbGpu, &pt.ptbHandle)) return false;
    pt.ptbCpu = cpu;
    /* rcpAllocVram already zeroes each page. */

    uint64_t va = COMPUTE_VA_ROOT;
    uint32_t i2 = (uint32_t)((va >> 39) & 0x1FF);
    uint32_t i1 = (uint32_t)((va >> 30) & 0x1FF);
    uint32_t i0 = (uint32_t)((va >> 21) & 0x1FF);
    uint32_t ip = (uint32_t)((va >> 12) & 0x1FF);

    uint64_t pdb1Off = (pt.pdb1Gpu - vramMcBase) & GPUVM_ADDR_MASK;
    uint64_t pdb0Off = (pt.pdb0Gpu - vramMcBase) & GPUVM_ADDR_MASK;
    uint64_t ptbOff  = (pt.ptbGpu  - vramMcBase) & GPUVM_ADDR_MASK;
    uint64_t codeOff = (codeGpu    - vramMcBase) & GPUVM_ADDR_MASK;

    *(volatile uint64_t *)((uint8_t *)pt.pdb2Cpu + i2 * 8) = AMDGPU_PTE_VALID | pdb1Off;
    *(volatile uint64_t *)((uint8_t *)pt.pdb1Cpu + i1 * 8) = AMDGPU_PTE_VALID | pdb0Off;
    *(volatile uint64_t *)((uint8_t *)pt.pdb0Cpu + i0 * 8) = AMDGPU_PTE_VALID | ptbOff;
    *(volatile uint64_t *)((uint8_t *)pt.ptbCpu  + ip * 8) = GPUVM_LEAF_FLAGS | codeOff;

    cqHdpFlush(gpu, cq);

    /* Re-point CONTEXT0 at the depth-3 page table (root = 0-based offset|VALID). */
    uint64_t root = ((pt.pdb2Gpu - vramMcBase) & GPUVM_ADDR_MASK) | AMDGPU_PTE_VALID;
    gcWreg(gpu, cq, regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32,
           (uint32_t)(root & 0xFFFFFFFF), 0);
    gcWreg(gpu, cq, regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32,
           (uint32_t)((root >> 32) & 0xFFFFFFFF), 0);
    gcWreg(gpu, cq, regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32, 0, 0);
    gcWreg(gpu, cq, regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32, 0, 0);
    uint64_t end = 0x7FFFFFFFFFFFull;
    gcWreg(gpu, cq, regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32,
           (uint32_t)((end >> 12) & 0xFFFFFFFF), 0);
    gcWreg(gpu, cq, regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32,
           (uint32_t)((end >> 44) & 0xFFFFFFFF), 0);
    gcWreg(gpu, cq, regGCVM_CONTEXT0_CNTL, GCVM_CONTEXT0_CNTL_DEPTH3, 0);

    cqHdpFlush(gpu, cq);
    gfxhubFlushTlb(gpu, cq, 0);

    printf("  GPUVM: VA=0x%llX idx=[%u,%u,%u,%u] root=0x%llX shaderMC=0x%llX\n",
           (unsigned long long)va, i2, i1, i0, ip,
           (unsigned long long)(root & GPUVM_ADDR_MASK),
           (unsigned long long)codeGpu);
    return true;
}

/* ---- Top-level: recipeDispatch -------------------------------------------- */

bool recipeDispatch(WddmLite &gpu, const IpDiscoveryResult &ipd,
                    const char *fwDir, uint64_t vramMcBase)
{
    /* 1. PSP cold-boot autoload -> BOOTLOAD_COMPLETE. */
    if (!recipeBootload(gpu, ipd, fwDir, vramMcBase)) {
        printf("  DISP: recipeBootload did not reach BOOTLOAD_COMPLETE\n");
        return false;
    }

    printf("\n=== recipeDispatch (GPUVM + s_endpgm DISPATCH_DIRECT) ===\n");

    CqState cq;
    memset(&cq, 0, sizeof(cq));
    cq.gcBase0 = ipd.gcBase;
    cq.gcBase1 = ipd.gcBase1;

    cq.hasNbif = cqResolveNbif(ipd, &cq.nbifBase2);
    if (cq.hasNbif) {
        uint32_t apEn = 0;
        gpu.readReg32((cq.nbifBase2 + regRCC_DOORBELL_APER_EN) * 4, &apEn);
        gpu.writeReg32((cq.nbifBase2 + regRCC_DOORBELL_APER_EN) * 4,
                       apEn | BIF_DOORBELL_APER_EN__BIT);
        uint32_t fbEn = 0;
        gpu.readReg32((cq.nbifBase2 + regBIF_FB_EN) * 4, &fbEn);
        gpu.writeReg32((cq.nbifBase2 + regBIF_FB_EN) * 4,
                       fbEn | BIF_FB_EN__FB_READ_EN | BIF_FB_EN__FB_WRITE_EN);
        printf("  NBIO: doorbell aperture + framebuffer enabled "
               "(NBIF base[2]=0x%04X)\n", cq.nbifBase2);
    } else {
        printf("  NBIO: WARNING NBIF base[2] not found; HDP flush + doorbell "
               "aperture skipped\n");
    }

    /* ucode_start: rs64() for PFP/ME/MEC (gfx-header entry @52). */
    const char *gc = "12_0_1";
    bool okP = false, okM = false, okC = false;
    uint64_t pfp = cqUcodeStart(fwDir, gc, "pfp", 52, &okP);
    uint64_t me = cqUcodeStart(fwDir, gc, "me", 52, &okM);
    uint64_t mec = cqUcodeStart(fwDir, gc, "mec", 52, &okC);
    bool haveUcode = okP && okM && okC;
    if (haveUcode)
        printf("  GFX: ucode_start PFP=0x%llX ME=0x%llX MEC=0x%llX\n",
               (unsigned long long)pfp, (unsigned long long)me,
               (unsigned long long)mec);

    /* 2. init_gfx_for_compute (MEC enable). */
    if (!cqInitGfxForCompute(gpu, cq, pfp, me, mec, haveUcode))
        return false;

    /* 3. gfxhub_gart_enable: re-init the GFXHUB after AUTOLOAD_RLC. Allocate the
     * GART page table + dummy page in DMA (system-memory) so the system aperture
     * / fault-default registers point at valid bus addresses. The depth-3
     * CONTEXT0 below overrides the GART CONTEXT0, so the GART table contents are
     * unused by the compute mapping; only its bus address matters here. */
    void *gartCpu = nullptr, *dummyCpu = nullptr;
    uint64_t gartBus = 0, dummyBus = 0;
    void *gartHandle = nullptr, *dummyHandle = nullptr;
    if (!gpu.allocDma(1 << 20, &gartCpu, &gartBus, &gartHandle)) {
        printf("  DISP: ERROR GART table DMA alloc failed\n");
        return false;
    }
    if (!gpu.allocDma(4096, &dummyCpu, &dummyBus, &dummyHandle)) {
        printf("  DISP: ERROR dummy page DMA alloc failed\n");
        return false;
    }
    memset(gartCpu, 0, 1 << 20);
    memset(dummyCpu, 0, 4096);

    GfxhubParams gp;
    memset(&gp, 0, sizeof(gp));
    /* VRAM range from GMC FB location (mirror gmcInit's derivation, since the
     * NOP/dispatch path does not carry a GmcState). vramMcBase == GmcState.vramStart. */
    uint32_t fbBase = 0, fbTop = 0;
    mmhubRead(gpu, ipd, regMMMC_VM_FB_LOCATION_BASE, &fbBase);
    mmhubRead(gpu, ipd, regMMMC_VM_FB_LOCATION_TOP, &fbTop);
    gp.vramStart = (uint64_t)fbBase << 24;
    gp.vramEnd = ((uint64_t)fbTop << 24) | 0xFFFFFF;
    gp.gartStart = gp.vramEnd + 1;
    gp.gartEnd = gp.gartStart + (512ull * 1024 * 1024) - 1;  /* 512MB GART */
    gp.agpStart = gp.gartEnd + 1;
    gp.agpEnd = gp.agpStart;                                  /* AGP disabled */
    gp.gartTableBus = gartBus;
    gp.dummyPageBus = dummyBus;
    gfxhubGartEnable(gpu, cq, gp);

    /* 4. Stage 16 dwords of s_endpgm in a 4KB VRAM page. */
    void *codeCpu = nullptr;
    uint64_t codeGpu = 0, codeHandle = 0;
    if (!rcpAllocVram(gpu, 4096, &codeCpu, &codeGpu, &codeHandle)) {
        printf("  DISP: ERROR shader VRAM alloc failed\n");
        return false;
    }
    {
        volatile uint32_t *code = (volatile uint32_t *)codeCpu;
        for (int i = 0; i < 16; i++)
            code[i] = NOOP_SHADER_DWORD;
    }

    /* 5. Build the 4-level GPUVM page table + enable CONTEXT0 (depth-3). */
    DispatchGpuvm pt;
    memset(&pt, 0, sizeof(pt));
    if (!buildComputeGpuvm(gpu, cq, vramMcBase, codeGpu, pt))
        return false;

    /* 6. init_compute_queue (direct-MMIO HQD, VMID 0). Queue stays VMID 0. */
    if (!cqInitComputeQueue(gpu, cq))
        return false;

    /* Readback CONTEXT0_CNTL via gc base[0] (probe16 prints this). */
    uint32_t c0 = gcReg(gpu, cq, regGCVM_CONTEXT0_CNTL, 0);
    printf("  GPUVM: GCVM_CONTEXT0_CNTL readback=0x%08X (enable=%u)\n",
           c0, c0 & 1u);

    /* Clear the fault status before dispatch (probe16 zeroes 0x15D0/0x15D1). */
    gpu.writeReg32((cq.gcBase0 + regGCVM_L2_PROTECTION_FAULT_STATUS) * 4, 0);
    gpu.writeReg32((cq.gcBase0 + (regGCVM_L2_PROTECTION_FAULT_STATUS + 1)) * 4, 0);

    /* 7. Build the dispatch PM4 stream (probe16_vmid0.py / _build_dispatch_packets).
     * NO STATIC_THREAD_MGMT override -- the autoload set the correct CU masks. */
    uint64_t pgm = COMPUTE_VA_ROOT >> 8;   /* COMPUTE_PGM = shader VA >> 8 */
    uint64_t fenceSeq = 1;
    *cq.fenceCpu = 0;

    std::vector<uint32_t> packets;
    pm4AcquireMem(packets);

    uint32_t pgmLoHi[2] = { (uint32_t)(pgm & 0xFFFFFFFF),
                            (uint32_t)((pgm >> 32) & 0xFFFFFFFF) };
    pm4SetShReg(packets, regCOMPUTE_PGM_LO, pgmLoHi, 2);     /* PGM_LO, PGM_HI */

    uint32_t rsrc12[2] = { NOOP_RSRC1, NOOP_RSRC2 };
    pm4SetShReg(packets, regCOMPUTE_PGM_RSRC1, rsrc12, 2);   /* RSRC1, RSRC2 */

    uint32_t rsrc3 = 0;
    pm4SetShReg(packets, regCOMPUTE_PGM_RSRC3, &rsrc3, 1);

    uint32_t tmpring = 0;
    pm4SetShReg(packets, regCOMPUTE_TMPRING_SIZE, &tmpring, 1);

    uint32_t restart[3] = { 0, 0, 0 };
    pm4SetShReg(packets, regCOMPUTE_RESTART_X, restart, 3);

    uint32_t userData[2] = { 0, 0 };
    pm4SetShReg(packets, regCOMPUTE_USER_DATA_0, userData, 2);

    uint32_t resLimits = 0;
    pm4SetShReg(packets, regCOMPUTE_RESOURCE_LIMITS, &resLimits, 1);

    /* COMPUTE_START_X..: start xyz, num_thread xyz=(1,1,1), two trailing zeros. */
    uint32_t startBlock[8] = { 0, 0, 0, 1, 1, 1, 0, 0 };
    pm4SetShReg(packets, regCOMPUTE_START_X, startBlock, 8);

    pm4DispatchDirect(packets, 1, 1, 1, DISPATCH_INITIATOR_W32);
    pm4EventWrite(packets, CS_PARTIAL_FLUSH, EVENT_INDEX_CS_PARTIAL_FLUSH);
    pm4ReleaseMemFence(packets, cq.fenceGpu, fenceSeq);

    cqSubmitPackets(gpu, cq, packets);

    /* 8. Wait for the EOP fence, then read fault status + RPTR. */
    bool ok = cqWaitFence(cq, fenceSeq, 5000);

    uint32_t faultStatus = gcReg(gpu, cq, regGCVM_L2_PROTECTION_FAULT_STATUS, 0);
    uint32_t faultAddrLo = gcReg(gpu, cq, regGCVM_L2_PROTECTION_FAULT_ADDR_LO32, 0);
    uint32_t faultAddrHi = gcReg(gpu, cq, regGCVM_L2_PROTECTION_FAULT_ADDR_HI32, 0);
    uint64_t faultVa = (((uint64_t)faultAddrHi << 32) | faultAddrLo) << 12;

    cqGrbmSelect(gpu, cq, cq.me, cq.pipe, cq.queue, 0);
    uint32_t rptr = gcReg(gpu, cq, regCP_HQD_PQ_RPTR, 0);
    cqGrbmDeselect(gpu, cq);
    uint64_t fenceVal = *cq.fenceCpu;

    uint32_t walker = (faultStatus >> 1) & 0x7;
    uint32_t perm = (faultStatus >> 4) & 0xF;

    printf("\nGCVM_CONTEXT0_CNTL=0x%08X (expected 0x%08X)\n",
           c0, GCVM_CONTEXT0_CNTL_DEPTH3);
    printf("FAULT_STATUS=0x%08X [walker=%u perm=0x%X] FAULT_VA=0x%llX\n",
           faultStatus, walker, perm, (unsigned long long)faultVa);
    printf("FENCE value=%llu (expected %llu)\n",
           (unsigned long long)fenceVal, (unsigned long long)fenceSeq);
    printf("RPTR=0x%X (wptr=%u)\n", rptr, cq.wptr);

    bool pass = ok && (faultStatus == 0);
    printf("DISPATCH %s\n", pass ? "PASS (fence signaled, no GPUVM fault)"
           : (ok ? "FAIL (GPUVM fault)" : "FAIL (fence timeout)"));
    return pass;
}

/* ======================================================================
 * Increment 3a: real compiled kernel + kernargs compute dispatch
 *
 * Faithful C++ transcription of the proven probe path:
 *   python/probe_kernel.py (THE probe that PASSED on this gfx1201): load the
 *     fill_kernel_raw.co into VRAM honoring section vaddrs, GPUVM-map code +
 *     kernarg + output (one PTB, VMID 0), pack the kernarg (out VA + u32 val),
 *     DISPATCH_DIRECT wave32 grid=1 block=64, then read out[0..63] == val.
 *   python/amd_gpu_driver/kernel/elf_parser.py: parse_elf (header/sections/
 *     symbols), assemble image by SECTION VADDR.
 *   python/amd_gpu_driver/kernel/descriptor.py: KernelDescriptor (the 64-byte
 *     amdhsa kernel descriptor at the "<name>.kd" symbol).
 *   python/amd_gpu_driver/backends/windows/compute_dispatch.py:
 *     dispatch_elf_kernel / _build_dispatch_packets (kernarg base VA -> the
 *     USER_DATA SGPR at the kernarg_segment_ptr slot index).
 *
 * Kernarg layout source -- KD, NOT msgpack: the probe derives the layout
 * directly from the 64-byte KERNEL_DESCRIPTOR plus the known fill_kernel ABI
 * (off0 = output pointer (8B), off8 = u32 fill value), and it PASSED. The
 * dispatch is a SINGLE workgroup (grid=1,1,1), so tid = local_id and the COV5
 * hidden args (get_local_size()*group_id == 0) are irrelevant; a 64-byte KD is
 * sufficient and no msgpack decoder is needed. This mirrors probe_kernel.py
 * exactly (compute_dispatch.kernel_arg_layout / msgpack is only required for
 * MULTI-workgroup grids, which this increment does not exercise).
 *
 * kernarg_sgpr_index == 0: the KD properties (props=0x408 -> bit3 kernarg-ptr,
 * bit10 wave32; bits 0/1/2 = private-seg-buffer/dispatch-ptr/queue-ptr all 0)
 * mean the kernarg segment pointer is the FIRST preload SGPR, landing in
 * USER_DATA_0 (s[0:1]). The slot computation below reproduces that.
 * ====================================================================== */

/* ---- Minimal AMDGPU ELF parser (elf_parser.py) ---------------------------- */

#define RCP_ELF_SHT_NOBITS  8
#define RCP_ELF_SHF_ALLOC   0x2
/* STT_AMDGPU_HSA_KERNEL == 10; STT_FUNC == 2; STB_GLOBAL == 1 (kernel_symbols). */
#define RCP_ELF_STT_AMDGPU_HSA_KERNEL  10
#define RCP_ELF_STT_FUNC               2
#define RCP_ELF_STB_GLOBAL             1

struct RcpElfSection {
    uint32_t sh_name;
    uint32_t sh_type;
    uint64_t sh_flags;
    uint64_t sh_addr;
    uint64_t sh_offset;
    uint64_t sh_size;
    uint32_t sh_link;
    uint32_t sh_info;
    uint64_t sh_addralign;
    uint64_t sh_entsize;
    char     name[64];
};

struct RcpElfSymbol {
    uint32_t st_name;
    uint8_t  st_info;
    uint8_t  st_other;
    uint16_t st_shndx;
    uint64_t st_value;
    uint64_t st_size;
    char     name[96];
};

struct RcpElf {
    const std::vector<uint8_t> *data;
    uint16_t e_machine;
    uint16_t e_type;
    std::vector<RcpElfSection> sections;
    std::vector<RcpElfSymbol> symbols;
};

/* Little-endian readers over the byte vector (the .co is ELFDATA2LSB; the host
 * is x86-64 LE, but read field-by-field so the parse is endianness-explicit and
 * matches struct.unpack_from in elf_parser.py). */
static uint16_t rcpElfU16(const std::vector<uint8_t> &d, size_t off)
{
    return (uint16_t)(d[off] | ((uint16_t)d[off + 1] << 8));
}
static uint32_t rcpElfU32(const std::vector<uint8_t> &d, size_t off)
{
    return (uint32_t)d[off] | ((uint32_t)d[off + 1] << 8) |
           ((uint32_t)d[off + 2] << 16) | ((uint32_t)d[off + 3] << 24);
}
static uint64_t rcpElfU64(const std::vector<uint8_t> &d, size_t off)
{
    uint64_t lo = rcpElfU32(d, off);
    uint64_t hi = rcpElfU32(d, off + 4);
    return lo | (hi << 32);
}

static void rcpElfCopyStr(const std::vector<uint8_t> &d, size_t strtabOff,
                          size_t strtabSize, uint32_t nameIdx,
                          char *out, size_t outSize)
{
    out[0] = '\0';
    if (nameIdx >= strtabSize)
        return;
    size_t p = strtabOff + nameIdx;
    size_t i = 0;
    while (i + 1 < outSize && p < d.size() && d[p] != 0) {
        out[i++] = (char)d[p++];
    }
    out[i] = '\0';
}

/* Parse a 64-bit AMDGPU ELF (parse_elf). Returns false on a bad/short file. */
static bool rcpElfParse(const std::vector<uint8_t> &data, RcpElf &elf)
{
    if (data.size() < 64) return false;
    if (!(data[0] == 0x7F && data[1] == 'E' && data[2] == 'L' && data[3] == 'F'))
        return false;
    if (data[4] != 2) return false;     /* ELFCLASS64 */

    elf.data = &data;
    elf.e_type = rcpElfU16(data, 16);
    elf.e_machine = rcpElfU16(data, 18);
    uint64_t e_shoff = rcpElfU64(data, 40);
    uint16_t e_shentsize = rcpElfU16(data, 58);
    uint16_t e_shnum = rcpElfU16(data, 60);
    uint16_t e_shstrndx = rcpElfU16(data, 62);

    if (e_shentsize < 64) return false;
    if (e_shoff + (uint64_t)e_shnum * e_shentsize > data.size()) return false;

    elf.sections.clear();
    elf.sections.reserve(e_shnum);
    for (uint16_t i = 0; i < e_shnum; i++) {
        size_t o = (size_t)e_shoff + (size_t)i * e_shentsize;
        RcpElfSection s;
        s.sh_name = rcpElfU32(data, o + 0);
        s.sh_type = rcpElfU32(data, o + 4);
        s.sh_flags = rcpElfU64(data, o + 8);
        s.sh_addr = rcpElfU64(data, o + 16);
        s.sh_offset = rcpElfU64(data, o + 24);
        s.sh_size = rcpElfU64(data, o + 32);
        s.sh_link = rcpElfU32(data, o + 40);
        s.sh_info = rcpElfU32(data, o + 44);
        s.sh_addralign = rcpElfU64(data, o + 48);
        s.sh_entsize = rcpElfU64(data, o + 56);
        s.name[0] = '\0';
        elf.sections.push_back(s);
    }

    /* Resolve section names from shstrtab. */
    if (e_shstrndx < elf.sections.size()) {
        const RcpElfSection &sh = elf.sections[e_shstrndx];
        for (auto &s : elf.sections)
            rcpElfCopyStr(data, (size_t)sh.sh_offset, (size_t)sh.sh_size,
                          s.sh_name, s.name, sizeof(s.name));
    }

    /* Find SYMTAB + its linked STRTAB (parse_elf: SHT_SYMTAB == 2). */
    const RcpElfSection *symtab = nullptr;
    for (const auto &s : elf.sections) {
        if (s.sh_type == 2) { symtab = &s; break; }
    }
    elf.symbols.clear();
    if (symtab) {
        size_t strOff = 0, strSize = 0;
        if (symtab->sh_link < elf.sections.size()) {
            const RcpElfSection &linked = elf.sections[symtab->sh_link];
            strOff = (size_t)linked.sh_offset;
            strSize = (size_t)linked.sh_size;
        }
        const size_t SYM_SIZE = 24;     /* Elf64_Sym */
        size_t nsym = (size_t)symtab->sh_size / SYM_SIZE;
        for (size_t i = 0; i < nsym; i++) {
            size_t o = (size_t)symtab->sh_offset + i * SYM_SIZE;
            if (o + SYM_SIZE > data.size()) break;
            RcpElfSymbol sym;
            sym.st_name = rcpElfU32(data, o + 0);
            sym.st_info = data[o + 4];
            sym.st_other = data[o + 5];
            sym.st_shndx = rcpElfU16(data, o + 6);
            sym.st_value = rcpElfU64(data, o + 8);
            sym.st_size = rcpElfU64(data, o + 16);
            sym.name[0] = '\0';
            if (strSize)
                rcpElfCopyStr(data, strOff, strSize, sym.st_name,
                              sym.name, sizeof(sym.name));
            elf.symbols.push_back(sym);
        }
    }
    return true;
}

/* The 64-byte amdhsa kernel descriptor (descriptor.py KernelDescriptor).
 * Only the fields the dispatch consumes are surfaced. */
struct RcpKernelDescriptor {
    uint32_t group_segment_fixed_size;     /* off 0  */
    uint32_t private_segment_fixed_size;   /* off 4  */
    uint32_t kernarg_size;                 /* off 8  */
    int64_t  kernel_code_entry_byte_offset;/* off 16 */
    uint32_t compute_pgm_rsrc3;            /* off 44 */
    uint32_t compute_pgm_rsrc1;            /* off 48 */
    uint32_t compute_pgm_rsrc2;            /* off 52 */
    uint16_t kernel_code_properties;       /* off 56 */
};

/* Parse the 64-byte KD out of `data` starting at `off` (from_bytes). */
static bool rcpKdFromBytes(const std::vector<uint8_t> &data, size_t off,
                           RcpKernelDescriptor &kd)
{
    if (off + 64 > data.size()) return false;
    kd.group_segment_fixed_size = rcpElfU32(data, off + 0);
    kd.private_segment_fixed_size = rcpElfU32(data, off + 4);
    kd.kernarg_size = rcpElfU32(data, off + 8);
    kd.kernel_code_entry_byte_offset = (int64_t)rcpElfU64(data, off + 16);
    kd.compute_pgm_rsrc3 = rcpElfU32(data, off + 44);
    kd.compute_pgm_rsrc1 = rcpElfU32(data, off + 48);
    kd.compute_pgm_rsrc2 = rcpElfU32(data, off + 52);
    kd.kernel_code_properties = rcpElfU16(data, off + 56);
    return true;
}

/* ---- Multi-region 4-level compute GPUVM page table ------------------------ */

/* One mapped buffer: a GPU VA and the FB-MC address it maps to. */
struct GpuvmRegion {
    uint64_t va;        /* virtual address the kernel uses */
    uint64_t gpuAddr;   /* FB-MC address (g_vramMcBase + offset) of the page */
};

/* Build a 4-level page table (PDB2->PDB1->PDB0->PTB) mapping MULTIPLE 4KB
 * regions, then enable GCVM_CONTEXT0 = depth-3 and flush the GFXHUB TLB.
 *
 * This generalizes buildComputeGpuvm: instead of one leaf it writes one LEAF
 * PTE per region. All regions in this increment lie inside the same 2MB block
 * at COMPUTE_VA_ROOT (0x200000000000) so they share ONE PDB2/PDB1/PDB0/PTB --
 * the assert below enforces that (matching probe_kernel.py, which uses a single
 * PTB for code+kernarg+output). Entries are 0-BASED VRAM OFFSETS
 * (gpuAddr - vramMcBase), the proven 2b math. Must run AFTER gfxhubGartEnable
 * and BEFORE the queue (probe ordering). */
static bool buildComputeGpuvmMulti(WddmLite &gpu, const CqState &cq,
                                   uint64_t vramMcBase,
                                   const GpuvmRegion *regions, uint32_t numRegions,
                                   DispatchGpuvm &pt)
{
    if (numRegions == 0) return false;

    void *cpu = nullptr;
    if (!rcpAllocVram(gpu, 4096, &cpu, &pt.pdb2Gpu, &pt.pdb2Handle)) return false;
    pt.pdb2Cpu = cpu;
    if (!rcpAllocVram(gpu, 4096, &cpu, &pt.pdb1Gpu, &pt.pdb1Handle)) return false;
    pt.pdb1Cpu = cpu;
    if (!rcpAllocVram(gpu, 4096, &cpu, &pt.pdb0Gpu, &pt.pdb0Handle)) return false;
    pt.pdb0Cpu = cpu;
    if (!rcpAllocVram(gpu, 4096, &cpu, &pt.ptbGpu, &pt.ptbHandle)) return false;
    pt.ptbCpu = cpu;
    /* rcpAllocVram already zeroes each page. */

    /* All regions must share the i2/i1/i0 indices (same 2MB block, one PTB). */
    uint64_t base = regions[0].va;
    uint32_t i2 = (uint32_t)((base >> 39) & 0x1FF);
    uint32_t i1 = (uint32_t)((base >> 30) & 0x1FF);
    uint32_t i0 = (uint32_t)((base >> 21) & 0x1FF);

    uint64_t pdb1Off = (pt.pdb1Gpu - vramMcBase) & GPUVM_ADDR_MASK;
    uint64_t pdb0Off = (pt.pdb0Gpu - vramMcBase) & GPUVM_ADDR_MASK;
    uint64_t ptbOff  = (pt.ptbGpu  - vramMcBase) & GPUVM_ADDR_MASK;

    *(volatile uint64_t *)((uint8_t *)pt.pdb2Cpu + i2 * 8) = AMDGPU_PTE_VALID | pdb1Off;
    *(volatile uint64_t *)((uint8_t *)pt.pdb1Cpu + i1 * 8) = AMDGPU_PTE_VALID | pdb0Off;
    *(volatile uint64_t *)((uint8_t *)pt.pdb0Cpu + i0 * 8) = AMDGPU_PTE_VALID | ptbOff;

    for (uint32_t r = 0; r < numRegions; r++) {
        uint64_t va = regions[r].va;
        uint32_t ri2 = (uint32_t)((va >> 39) & 0x1FF);
        uint32_t ri1 = (uint32_t)((va >> 30) & 0x1FF);
        uint32_t ri0 = (uint32_t)((va >> 21) & 0x1FF);
        if (ri2 != i2 || ri1 != i1 || ri0 != i0) {
            printf("  GPUVM(multi): ERROR region %u VA=0x%llX crosses the 2MB "
                   "block of base VA=0x%llX (single-PTB assumption violated)\n",
                   r, (unsigned long long)va, (unsigned long long)base);
            return false;
        }
        uint32_t ip = (uint32_t)((va >> 12) & 0x1FF);
        uint64_t off = (regions[r].gpuAddr - vramMcBase) & GPUVM_ADDR_MASK;
        *(volatile uint64_t *)((uint8_t *)pt.ptbCpu + ip * 8) = GPUVM_LEAF_FLAGS | off;
    }

    cqHdpFlush(gpu, cq);

    /* Re-point CONTEXT0 at the depth-3 page table (root = 0-based offset|VALID). */
    uint64_t root = ((pt.pdb2Gpu - vramMcBase) & GPUVM_ADDR_MASK) | AMDGPU_PTE_VALID;
    gcWreg(gpu, cq, regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32,
           (uint32_t)(root & 0xFFFFFFFF), 0);
    gcWreg(gpu, cq, regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32,
           (uint32_t)((root >> 32) & 0xFFFFFFFF), 0);
    gcWreg(gpu, cq, regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32, 0, 0);
    gcWreg(gpu, cq, regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32, 0, 0);
    uint64_t end = 0x7FFFFFFFFFFFull;
    gcWreg(gpu, cq, regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32,
           (uint32_t)((end >> 12) & 0xFFFFFFFF), 0);
    gcWreg(gpu, cq, regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32,
           (uint32_t)((end >> 44) & 0xFFFFFFFF), 0);
    gcWreg(gpu, cq, regGCVM_CONTEXT0_CNTL, GCVM_CONTEXT0_CNTL_DEPTH3, 0);

    cqHdpFlush(gpu, cq);
    gfxhubFlushTlb(gpu, cq, 0);

    printf("  GPUVM(multi): %u regions, block idx=[%u,%u,%u] root=0x%llX\n",
           numRegions, i2, i1, i0,
           (unsigned long long)(root & GPUVM_ADDR_MASK));
    return true;
}

/* ---- Top-level: recipeKernargDispatch ------------------------------------- */

/* fill_kernel ABI (probe_kernel.py): kernarg[0:8] = output VA, kernarg[8:12] =
 * u32 fill value. Single workgroup grid=1 block=64 -> out[0..N-1] == VAL. */
#define KERN_FILL_VAL   0xDEADBEEFu
#define KERN_FILL_N     64
#define KERN_BLOCK_X    64
#define KERN_CO_FILE    "fill_kernel_raw.co"

bool recipeKernargDispatch(WddmLite &gpu, const IpDiscoveryResult &ipd,
                           const char *fwDir, uint64_t vramMcBase)
{
    /* 1. PSP cold-boot autoload -> BOOTLOAD_COMPLETE. */
    if (!recipeBootload(gpu, ipd, fwDir, vramMcBase)) {
        printf("  KERN: recipeBootload did not reach BOOTLOAD_COMPLETE\n");
        return false;
    }

    printf("\n=== recipeKernargDispatch (real kernel + kernargs) ===\n");

    /* Load + parse the compiled kernel from the firmware dir (same dir the
     * bootload firmware is read from, e.g. Z:\winfw). loadFirmwareFile caps at
     * 16MB, ample for a ~5KB .co. */
    std::vector<uint8_t> co;
    {
        char path[512];
        snprintf(path, sizeof(path), "%s\\%s", fwDir, KERN_CO_FILE);
        if (!loadFirmwareFile(path, co)) {
            printf("  KERN: ERROR cannot read kernel %s\n", path);
            return false;
        }
    }
    RcpElf elf;
    if (!rcpElfParse(co, elf)) {
        printf("  KERN: ERROR failed to parse %s as AMDGPU ELF\n", KERN_CO_FILE);
        return false;
    }

    /* Find the kernel entry symbol and its "<name>.kd" descriptor symbol. */
    const RcpElfSymbol *kSym = nullptr;
    for (const auto &s : elf.symbols) {
        uint8_t type = s.st_info & 0xF;
        uint8_t bind = s.st_info >> 4;
        if (type == RCP_ELF_STT_AMDGPU_HSA_KERNEL ||
            (type == RCP_ELF_STT_FUNC && bind == RCP_ELF_STB_GLOBAL)) {
            kSym = &s;
            break;
        }
    }
    if (!kSym) {
        printf("  KERN: ERROR no kernel symbol in %s\n", KERN_CO_FILE);
        return false;
    }
    char kdName[128];
    snprintf(kdName, sizeof(kdName), "%s.kd", kSym->name);
    const RcpElfSymbol *kdSym = nullptr;
    for (const auto &s : elf.symbols) {
        if (strcmp(s.name, kdName) == 0) { kdSym = &s; break; }
    }
    if (!kdSym) {
        printf("  KERN: ERROR no kernel descriptor %s\n", kdName);
        return false;
    }
    /* The KD lives in whichever ALLOC section its vaddr falls in. */
    const RcpElfSection *kdSec = nullptr;
    for (const auto &s : elf.sections) {
        if (s.sh_size && s.sh_addr <= kdSym->st_value &&
            kdSym->st_value < s.sh_addr + s.sh_size) {
            kdSec = &s;
            break;
        }
    }
    if (!kdSec) {
        printf("  KERN: ERROR kernel descriptor section not found\n");
        return false;
    }
    RcpKernelDescriptor kd;
    size_t kdOff = (size_t)kdSec->sh_offset + (size_t)(kdSym->st_value - kdSec->sh_addr);
    if (!rcpKdFromBytes(co, kdOff, kd)) {
        printf("  KERN: ERROR short kernel descriptor\n");
        return false;
    }
    printf("  KERN: kernel='%s' RSRC1=0x%08X RSRC2=0x%08X RSRC3=0x%08X "
           "kernarg=%u props=0x%X\n",
           kSym->name, kd.compute_pgm_rsrc1, kd.compute_pgm_rsrc2,
           kd.compute_pgm_rsrc3, kd.kernarg_size, kd.kernel_code_properties);

    /* Assemble the loadable image by SECTION VADDR (alloc, non-NOBITS). The
     * image high-water mark is max(sh_addr + sh_size) over ALLOC sections,
     * rounded up to a page. */
    uint64_t imgHi = 0;
    for (const auto &s : elf.sections) {
        if ((s.sh_flags & RCP_ELF_SHF_ALLOC) && s.sh_addr) {
            uint64_t hi = s.sh_addr + s.sh_size;
            if (hi > imgHi) imgHi = hi;
        }
    }
    if (imgHi == 0) {
        printf("  KERN: ERROR no allocatable sections\n");
        return false;
    }
    uint64_t imgBytes = (imgHi + 0xFFF) & ~0xFFFull;
    uint32_t codePages = (uint32_t)(imgBytes / 0x1000);
    std::vector<uint8_t> image((size_t)imgBytes, 0);
    for (const auto &s : elf.sections) {
        if ((s.sh_flags & RCP_ELF_SHF_ALLOC) && s.sh_addr &&
            s.sh_type != RCP_ELF_SHT_NOBITS) {
            if ((size_t)(s.sh_offset + s.sh_size) > co.size()) continue;
            memcpy(&image[(size_t)s.sh_addr], &co[(size_t)s.sh_offset],
                   (size_t)s.sh_size);
        }
    }

    /* --- bring up the GFX/MEC and GFXHUB (same as recipeDispatch) ---------- */
    CqState cq;
    memset(&cq, 0, sizeof(cq));
    cq.gcBase0 = ipd.gcBase;
    cq.gcBase1 = ipd.gcBase1;

    cq.hasNbif = cqResolveNbif(ipd, &cq.nbifBase2);
    if (cq.hasNbif) {
        uint32_t apEn = 0;
        gpu.readReg32((cq.nbifBase2 + regRCC_DOORBELL_APER_EN) * 4, &apEn);
        gpu.writeReg32((cq.nbifBase2 + regRCC_DOORBELL_APER_EN) * 4,
                       apEn | BIF_DOORBELL_APER_EN__BIT);
        uint32_t fbEn = 0;
        gpu.readReg32((cq.nbifBase2 + regBIF_FB_EN) * 4, &fbEn);
        gpu.writeReg32((cq.nbifBase2 + regBIF_FB_EN) * 4,
                       fbEn | BIF_FB_EN__FB_READ_EN | BIF_FB_EN__FB_WRITE_EN);
        printf("  NBIO: doorbell aperture + framebuffer enabled "
               "(NBIF base[2]=0x%04X)\n", cq.nbifBase2);
    } else {
        printf("  NBIO: WARNING NBIF base[2] not found; HDP flush + doorbell "
               "aperture skipped\n");
    }

    const char *gc = "12_0_1";
    bool okP = false, okM = false, okC = false;
    uint64_t pfp = cqUcodeStart(fwDir, gc, "pfp", 52, &okP);
    uint64_t me = cqUcodeStart(fwDir, gc, "me", 52, &okM);
    uint64_t mec = cqUcodeStart(fwDir, gc, "mec", 52, &okC);
    bool haveUcode = okP && okM && okC;
    if (haveUcode)
        printf("  GFX: ucode_start PFP=0x%llX ME=0x%llX MEC=0x%llX\n",
               (unsigned long long)pfp, (unsigned long long)me,
               (unsigned long long)mec);

    if (!cqInitGfxForCompute(gpu, cq, pfp, me, mec, haveUcode))
        return false;

    void *gartCpu = nullptr, *dummyCpu = nullptr;
    uint64_t gartBus = 0, dummyBus = 0;
    void *gartHandle = nullptr, *dummyHandle = nullptr;
    if (!gpu.allocDma(1 << 20, &gartCpu, &gartBus, &gartHandle)) {
        printf("  KERN: ERROR GART table DMA alloc failed\n");
        return false;
    }
    if (!gpu.allocDma(4096, &dummyCpu, &dummyBus, &dummyHandle)) {
        printf("  KERN: ERROR dummy page DMA alloc failed\n");
        return false;
    }
    memset(gartCpu, 0, 1 << 20);
    memset(dummyCpu, 0, 4096);

    GfxhubParams gp;
    memset(&gp, 0, sizeof(gp));
    uint32_t fbBase = 0, fbTop = 0;
    mmhubRead(gpu, ipd, regMMMC_VM_FB_LOCATION_BASE, &fbBase);
    mmhubRead(gpu, ipd, regMMMC_VM_FB_LOCATION_TOP, &fbTop);
    gp.vramStart = (uint64_t)fbBase << 24;
    gp.vramEnd = ((uint64_t)fbTop << 24) | 0xFFFFFF;
    gp.gartStart = gp.vramEnd + 1;
    gp.gartEnd = gp.gartStart + (512ull * 1024 * 1024) - 1;
    gp.agpStart = gp.gartEnd + 1;
    gp.agpEnd = gp.agpStart;
    gp.gartTableBus = gartBus;
    gp.dummyPageBus = dummyBus;
    gfxhubGartEnable(gpu, cq, gp);

    /* --- stage code + kernarg + output in VRAM (FB-MC bump allocator) ------ */
    void *codeCpu = nullptr;
    uint64_t codeGpu = 0, codeHandle = 0;
    if (!rcpAllocVram(gpu, imgBytes, &codeCpu, &codeGpu, &codeHandle)) {
        printf("  KERN: ERROR code VRAM alloc failed\n");
        return false;
    }
    memcpy(codeCpu, image.data(), (size_t)imgBytes);

    void *kaCpu = nullptr;
    uint64_t kaGpu = 0, kaHandle = 0;
    if (!rcpAllocVram(gpu, 4096, &kaCpu, &kaGpu, &kaHandle)) {
        printf("  KERN: ERROR kernarg VRAM alloc failed\n");
        return false;
    }

    void *outCpu = nullptr;
    uint64_t outGpu = 0, outHandle = 0;
    if (!rcpAllocVram(gpu, 4096, &outCpu, &outGpu, &outHandle)) {
        printf("  KERN: ERROR output VRAM alloc failed\n");
        return false;
    }
    /* rcpAllocVram zeroes the output page. */

    /* VA map: code at COMPUTE_VA_ROOT (codePages pages), kernarg + output on
     * the next two pages (one PTB, matching probe_kernel.py). */
    uint64_t codeVa = COMPUTE_VA_ROOT;
    uint64_t kaVa = COMPUTE_VA_ROOT + (uint64_t)codePages * 0x1000;
    uint64_t outVa = COMPUTE_VA_ROOT + (uint64_t)(codePages + 1) * 0x1000;
    /* code entry = code_va + (kd_sym.st_value + entry_byte_offset). */
    uint64_t codeEntryVa = codeVa +
        ((uint64_t)kdSym->st_value + (uint64_t)kd.kernel_code_entry_byte_offset);

    /* Fill the kernarg: [0:8] = output VA (little-endian), [8:12] = u32 val. */
    {
        uint8_t *kb = (uint8_t *)kaCpu;
        for (int b = 0; b < 8; b++)
            kb[0 + b] = (uint8_t)((outVa >> (b * 8)) & 0xFF);
        uint32_t val = KERN_FILL_VAL;
        for (int b = 0; b < 4; b++)
            kb[8 + b] = (uint8_t)((val >> (b * 8)) & 0xFF);
    }

    /* Build the multi-region page table: every code page + kernarg + output. */
    std::vector<GpuvmRegion> regions;
    regions.reserve(codePages + 2);
    for (uint32_t p = 0; p < codePages; p++)
        regions.push_back({ codeVa + (uint64_t)p * 0x1000,
                            codeGpu + (uint64_t)p * 0x1000 });
    regions.push_back({ kaVa, kaGpu });
    regions.push_back({ outVa, outGpu });

    DispatchGpuvm pt;
    memset(&pt, 0, sizeof(pt));
    if (!buildComputeGpuvmMulti(gpu, cq, vramMcBase, regions.data(),
                                (uint32_t)regions.size(), pt))
        return false;

    /* init_compute_queue (direct-MMIO HQD, VMID 0). */
    if (!cqInitComputeQueue(gpu, cq))
        return false;

    uint32_t c0 = gcReg(gpu, cq, regGCVM_CONTEXT0_CNTL, 0);
    printf("  GPUVM: GCVM_CONTEXT0_CNTL readback=0x%08X (enable=%u)\n",
           c0, c0 & 1u);

    /* kernarg_sgpr_index: kernarg-ptr lands after private-seg-buffer (4),
     * dispatch-ptr (2), queue-ptr (2). props=0x408 -> none set -> slot 0. */
    uint32_t slot = 0;
    if (kd.kernel_code_properties & (1u << 0)) slot += 4;   /* private-seg-buffer */
    if (kd.kernel_code_properties & (1u << 1)) slot += 2;   /* dispatch-ptr */
    if (kd.kernel_code_properties & (1u << 2)) slot += 2;   /* queue-ptr */

    printf("  KERN: code_entry=0x%llX ka_va=0x%llX out_va=0x%llX "
           "code_pages=%u sgpr_slot=%u\n",
           (unsigned long long)codeEntryVa, (unsigned long long)kaVa,
           (unsigned long long)outVa, codePages, slot);

    /* Clear the fault status before dispatch. */
    gpu.writeReg32((cq.gcBase0 + regGCVM_L2_PROTECTION_FAULT_STATUS) * 4, 0);
    gpu.writeReg32((cq.gcBase0 + (regGCVM_L2_PROTECTION_FAULT_STATUS + 1)) * 4, 0);

    /* Build the dispatch PM4 (probe_kernel.py / _build_dispatch_packets). KD
     * RSRC1/RSRC2/RSRC3 verbatim -- NO STATIC_THREAD_MGMT override. */
    uint64_t pgm = codeEntryVa >> 8;            /* COMPUTE_PGM = entry VA >> 8 */
    uint64_t fenceSeq = 1;
    *cq.fenceCpu = 0;

    std::vector<uint32_t> packets;
    pm4AcquireMem(packets);

    uint32_t pgmLoHi[2] = { (uint32_t)(pgm & 0xFFFFFFFF),
                            (uint32_t)((pgm >> 32) & 0xFFFFFFFF) };
    pm4SetShReg(packets, regCOMPUTE_PGM_LO, pgmLoHi, 2);

    uint32_t rsrc12[2] = { kd.compute_pgm_rsrc1, kd.compute_pgm_rsrc2 };
    pm4SetShReg(packets, regCOMPUTE_PGM_RSRC1, rsrc12, 2);

    uint32_t rsrc3 = kd.compute_pgm_rsrc3;
    pm4SetShReg(packets, regCOMPUTE_PGM_RSRC3, &rsrc3, 1);

    uint32_t tmpring = 0;
    pm4SetShReg(packets, regCOMPUTE_TMPRING_SIZE, &tmpring, 1);

    uint32_t restart[3] = { 0, 0, 0 };
    pm4SetShReg(packets, regCOMPUTE_RESTART_X, restart, 3);

    /* kernarg base VA -> USER_DATA_<slot> (s[slot:slot+1]). */
    uint32_t userData[2] = { (uint32_t)(kaVa & 0xFFFFFFFF),
                             (uint32_t)((kaVa >> 32) & 0xFFFFFFFF) };
    pm4SetShReg(packets, regCOMPUTE_USER_DATA_0 + slot, userData, 2);

    uint32_t resLimits = 0;
    pm4SetShReg(packets, regCOMPUTE_RESOURCE_LIMITS, &resLimits, 1);

    /* COMPUTE_START_X..: start xyz=0, num_thread xyz=(block,1,1), 2 trailing 0. */
    uint32_t startBlock[8] = { 0, 0, 0, KERN_BLOCK_X, 1, 1, 0, 0 };
    pm4SetShReg(packets, regCOMPUTE_START_X, startBlock, 8);

    pm4DispatchDirect(packets, 1, 1, 1, DISPATCH_INITIATOR_W32);
    pm4EventWrite(packets, CS_PARTIAL_FLUSH, EVENT_INDEX_CS_PARTIAL_FLUSH);
    pm4ReleaseMemFence(packets, cq.fenceGpu, fenceSeq);

    cqSubmitPackets(gpu, cq, packets);

    bool ok = cqWaitFence(cq, fenceSeq, 5000);

    /* HDP flush so the GPU's VRAM writes to the output page are CPU-visible. */
    cqHdpFlush(gpu, cq);

    uint32_t faultStatus = gcReg(gpu, cq, regGCVM_L2_PROTECTION_FAULT_STATUS, 0);
    uint32_t faultAddrLo = gcReg(gpu, cq, regGCVM_L2_PROTECTION_FAULT_ADDR_LO32, 0);
    uint32_t faultAddrHi = gcReg(gpu, cq, regGCVM_L2_PROTECTION_FAULT_ADDR_HI32, 0);
    uint64_t faultVa = (((uint64_t)faultAddrHi << 32) | faultAddrLo) << 12;

    cqGrbmSelect(gpu, cq, cq.me, cq.pipe, cq.queue, 0);
    uint32_t rptr = gcReg(gpu, cq, regCP_HQD_PQ_RPTR, 0);
    cqGrbmDeselect(gpu, cq);
    uint64_t fenceVal = *cq.fenceCpu;

    /* Verify the output buffer via its CPU mapping. */
    const volatile uint32_t *res = (const volatile uint32_t *)outCpu;
    uint32_t nbad = 0;
    uint32_t firstBad = 0;
    uint32_t firstBadVal = 0;
    for (uint32_t i = 0; i < KERN_FILL_N; i++) {
        uint32_t v = res[i];
        if (v != KERN_FILL_VAL) {
            if (nbad == 0) { firstBad = i; firstBadVal = v; }
            nbad++;
        }
    }

    uint32_t walker = (faultStatus >> 1) & 0x7;
    uint32_t perm = (faultStatus >> 4) & 0xF;

    printf("\nFAULT_STATUS=0x%08X [walker=%u perm=0x%X] FAULT_VA=0x%llX\n",
           faultStatus, walker, perm, (unsigned long long)faultVa);
    printf("FENCE value=%llu (expected %llu) RPTR=0x%X\n",
           (unsigned long long)fenceVal, (unsigned long long)fenceSeq, rptr);
    printf("out[0]=0x%08X out[1]=0x%08X expected=0x%08X bad=%u/%u\n",
           res[0], res[1], KERN_FILL_VAL, nbad, KERN_FILL_N);
    if (nbad)
        printf("  first mismatch at index %u: got 0x%08X\n",
               firstBad, firstBadVal);

    bool pass = ok && (faultStatus == 0) && (nbad == 0);
    printf("KERNARG DISPATCH %s\n",
           pass ? "PASS (fence signaled, no fault, output verified)"
           : (!ok ? "FAIL (fence timeout)"
              : (faultStatus ? "FAIL (GPUVM fault)" : "FAIL (output mismatch)")));
    return pass;
}

/* ======================================================================
 * Multi-workgroup compute dispatch (increment 3b) -- COV5 hidden args
 *
 * recipeKernargDispatch dispatched fill_kernel over a SINGLE workgroup, where
 * group_id is always 0 so get_local_size()/get_num_groups() never matter. For
 * a grid > 1 the kernel computes tid = local_id + group_id*get_local_size(0),
 * and get_local_size(0) reads hidden_group_size_x from the COV5 implicit-args
 * block that the compiler places in the kernarg segment. So the dispatch must
 * fill those hidden args (block_count / group_size / remainder / grid_dims) at
 * the metadata-declared offsets, or every workgroup past 0 writes garbage.
 *
 * Mirrors python/probe_kernel_mw.py (the PASSED multi-WG probe) byte-for-byte:
 *   - explicit args:  kernarg[0:8]  = output VA (u64)
 *                     kernarg[8:12] = fill value (u32 0xDEADBEEF)
 *   - hidden args (offsets confirmed against fill_kernel_raw.co's
 *     NT_AMDGPU_METADATA note; see rcpFindHiddenArgs, which parses the note so
 *     the offsets are not hardcoded):
 *       off 16 u32 hidden_block_count_x  = GRID   (# WORKGROUPS, not threads)
 *       off 20 u32 hidden_block_count_y  = 1
 *       off 24 u32 hidden_block_count_z  = 1
 *       off 28 u16 hidden_group_size_x   = BLOCK  (64) <- the one that was missing
 *       off 30 u16 hidden_group_size_y   = 1
 *       off 32 u16 hidden_group_size_z   = 1
 *       off 34 u16 hidden_remainder_x    = 0  (uniform_work_group_size=1)
 *       off 80 u16 hidden_grid_dims      = 1
 *   - DISPATCH_DIRECT(GRID,1,1): DIM_X is the number of WORKGROUPS.
 *   - COMPUTE_NUM_THREAD_X = BLOCK (64): threads PER workgroup, programmed in
 *     COMPUTE_START_X+3 (the startBlock[] num_thread fields), NOT in DIM_X.
 *   => out[0 .. GRID*BLOCK - 1] all == fill.
 *
 * The probe validated the mechanism with GRID=4 (256 dwords = 1 page). This
 * C++ path generalizes to any grid; the "mwg" test uses GRID=64 so the output
 * (4096 dwords = 16384 bytes = 4 pages) genuinely spans MULTIPLE 4KB pages and
 * exercises the per-page GPUVM mapping in buildComputeGpuvmMulti.
 * ====================================================================== */

#define MWG_FILL_VAL    0xDEADBEEFu
#define MWG_BLOCK_X     64u            /* threads per workgroup (proven) */
#define MWG_GRID_X      64u            /* workgroups: 64*64=4096 dw = 4 pages */

/* One COV5 hidden arg located in the kernarg segment: its byte offset + size. */
struct RcpHiddenArg {
    uint32_t offset;
    uint32_t size;
    bool     found;
};

/* The hidden args the multi-WG dispatch must populate (parsed from metadata). */
struct RcpHiddenArgs {
    RcpHiddenArg blockCountX, blockCountY, blockCountZ;
    RcpHiddenArg groupSizeX, groupSizeY, groupSizeZ;
    RcpHiddenArg remainderX, remainderY, remainderZ;
    RcpHiddenArg gridDims;
};

/* ---- Minimal msgpack reader for the NT_AMDGPU_METADATA note --------------- */

/* The AMDGPU metadata note is msgpack (big-endian length/value fields). We walk
 * maps/arrays/strings/ints to reach amdhsa.kernels[].args[] and read each arg's
 * .offset / .size / .value_kind. This is the C++ port of the subset of
 * kernel/metadata.py:_decode that fill_kernel's metadata exercises. */
struct RcpMsgpack {
    const uint8_t *p;
    size_t         n;
    size_t         i;
    bool           err;
};

static uint16_t rcpMpBE16(const uint8_t *d) {
    return (uint16_t)(((uint16_t)d[0] << 8) | d[1]);
}
static uint32_t rcpMpBE32(const uint8_t *d) {
    return ((uint32_t)d[0] << 24) | ((uint32_t)d[1] << 16) |
           ((uint32_t)d[2] << 8) | d[3];
}
static uint64_t rcpMpBE64(const uint8_t *d) {
    return ((uint64_t)rcpMpBE32(d) << 32) | rcpMpBE32(d + 4);
}

enum RcpMpKind { MP_NIL, MP_BOOL, MP_INT, MP_STR, MP_ARRAY, MP_MAP, MP_OTHER };

/* A decoded msgpack value header. For ARRAY/MAP only the count is read here; the
 * caller decodes that many elements/pairs next. For STR the byte span is
 * recorded and the cursor is advanced past it. Scalars are fully consumed. */
struct RcpMpVal {
    RcpMpKind kind;
    uint64_t  uval;        /* MP_INT (and MP_BOOL: 0/1) */
    size_t    strOff;      /* MP_STR: byte offset into the note */
    size_t    strLen;
    uint32_t  count;       /* MP_ARRAY (#elements) / MP_MAP (#pairs) */
};

static bool rcpMpNeed(RcpMsgpack &m, size_t k) {
    if (m.err || m.i + k > m.n) { m.err = true; return false; }
    return true;
}

static RcpMpVal rcpMpHead(RcpMsgpack &m) {
    RcpMpVal v;
    v.kind = MP_OTHER; v.uval = 0; v.strOff = 0; v.strLen = 0; v.count = 0;
    if (!rcpMpNeed(m, 1)) return v;
    uint8_t b = m.p[m.i++];
    if (b < 0x80) { v.kind = MP_INT; v.uval = b; return v; }            /* + fixint */
    if (b >= 0xE0) { v.kind = MP_INT; v.uval = (uint64_t)(int64_t)(int8_t)b; return v; }
    if (b >= 0x80 && b <= 0x8F) { v.kind = MP_MAP; v.count = (uint32_t)(b & 0x0F); return v; }
    if (b >= 0x90 && b <= 0x9F) { v.kind = MP_ARRAY; v.count = (uint32_t)(b & 0x0F); return v; }
    if (b >= 0xA0 && b <= 0xBF) {                                       /* fixstr */
        uint32_t len = (uint32_t)(b & 0x1F);
        if (!rcpMpNeed(m, len)) return v;
        v.kind = MP_STR; v.strOff = m.i; v.strLen = len; m.i += len;
        return v;
    }
    switch (b) {
    case 0xC0:
        v.kind = MP_NIL;
        return v;
    case 0xC2:
        v.kind = MP_BOOL; v.uval = 0;
        return v;
    case 0xC3:
        v.kind = MP_BOOL; v.uval = 1;
        return v;
    case 0xCC:
        if (!rcpMpNeed(m, 1)) return v;
        v.kind = MP_INT; v.uval = m.p[m.i]; m.i += 1;
        return v;
    case 0xCD:
        if (!rcpMpNeed(m, 2)) return v;
        v.kind = MP_INT; v.uval = rcpMpBE16(m.p + m.i); m.i += 2;
        return v;
    case 0xCE:
        if (!rcpMpNeed(m, 4)) return v;
        v.kind = MP_INT; v.uval = rcpMpBE32(m.p + m.i); m.i += 4;
        return v;
    case 0xCF:
        if (!rcpMpNeed(m, 8)) return v;
        v.kind = MP_INT; v.uval = rcpMpBE64(m.p + m.i); m.i += 8;
        return v;
    case 0xD0:
        if (!rcpMpNeed(m, 1)) return v;
        v.kind = MP_INT; v.uval = (uint64_t)(int64_t)(int8_t)m.p[m.i]; m.i += 1;
        return v;
    case 0xD1:
        if (!rcpMpNeed(m, 2)) return v;
        v.kind = MP_INT; v.uval = (uint64_t)(int64_t)(int16_t)rcpMpBE16(m.p + m.i); m.i += 2;
        return v;
    case 0xD2:
        if (!rcpMpNeed(m, 4)) return v;
        v.kind = MP_INT; v.uval = (uint64_t)(int64_t)(int32_t)rcpMpBE32(m.p + m.i); m.i += 4;
        return v;
    case 0xD3:
        if (!rcpMpNeed(m, 8)) return v;
        v.kind = MP_INT; v.uval = rcpMpBE64(m.p + m.i); m.i += 8;
        return v;
    case 0xD9: {                                                        /* str8 */
        if (!rcpMpNeed(m, 1)) return v;
        uint32_t len = m.p[m.i]; m.i += 1;
        if (!rcpMpNeed(m, len)) return v;
        v.kind = MP_STR; v.strOff = m.i; v.strLen = len; m.i += len;
        return v;
    }
    case 0xDA: {                                                        /* str16 */
        if (!rcpMpNeed(m, 2)) return v;
        uint32_t len = rcpMpBE16(m.p + m.i); m.i += 2;
        if (!rcpMpNeed(m, len)) return v;
        v.kind = MP_STR; v.strOff = m.i; v.strLen = len; m.i += len;
        return v;
    }
    case 0xDB: {                                                        /* str32 */
        if (!rcpMpNeed(m, 4)) return v;
        uint32_t len = rcpMpBE32(m.p + m.i); m.i += 4;
        if (!rcpMpNeed(m, len)) return v;
        v.kind = MP_STR; v.strOff = m.i; v.strLen = len; m.i += len;
        return v;
    }
    case 0xDC:
        if (!rcpMpNeed(m, 2)) return v;
        v.kind = MP_ARRAY; v.count = rcpMpBE16(m.p + m.i); m.i += 2;
        return v;
    case 0xDD:
        if (!rcpMpNeed(m, 4)) return v;
        v.kind = MP_ARRAY; v.count = rcpMpBE32(m.p + m.i); m.i += 4;
        return v;
    case 0xDE:
        if (!rcpMpNeed(m, 2)) return v;
        v.kind = MP_MAP; v.count = rcpMpBE16(m.p + m.i); m.i += 2;
        return v;
    case 0xDF:
        if (!rcpMpNeed(m, 4)) return v;
        v.kind = MP_MAP; v.count = rcpMpBE32(m.p + m.i); m.i += 4;
        return v;
    default:
        m.err = true;                  /* unsupported byte (e.g. float/bin) */
        return v;
    }
}

/* Given an already-decoded header v, consume its children (no-op for scalars
 * and strings; recurse for arrays/maps). */
static void rcpMpSkipChildren(RcpMsgpack &m, const RcpMpVal &v);

/* Decode the value at the cursor and fully consume it (header + children). */
static void rcpMpSkipValue(RcpMsgpack &m) {
    RcpMpVal v = rcpMpHead(m);
    rcpMpSkipChildren(m, v);
}

static void rcpMpSkipChildren(RcpMsgpack &m, const RcpMpVal &v) {
    if (m.err) return;
    if (v.kind == MP_ARRAY) {
        for (uint32_t k = 0; k < v.count && !m.err; k++) rcpMpSkipValue(m);
    } else if (v.kind == MP_MAP) {
        for (uint32_t k = 0; k < v.count && !m.err; k++) { rcpMpSkipValue(m); rcpMpSkipValue(m); }
    }
}

/* True if the string value v equals the C string s. */
static bool rcpMpStrEq(const RcpMsgpack &m, const RcpMpVal &v, const char *s) {
    if (v.kind != MP_STR) return false;
    size_t sl = strlen(s);
    if (v.strLen != sl) return false;
    return memcmp(m.p + v.strOff, s, sl) == 0;
}

/* Map a value_kind string -> the RcpHiddenArg field to populate; nullptr for
 * explicit/uninteresting kinds. */
static RcpHiddenArg *rcpHiddenSlot(RcpHiddenArgs &h, const RcpMsgpack &m,
                                   const RcpMpVal &kindStr) {
    if (rcpMpStrEq(m, kindStr, "hidden_block_count_x")) return &h.blockCountX;
    if (rcpMpStrEq(m, kindStr, "hidden_block_count_y")) return &h.blockCountY;
    if (rcpMpStrEq(m, kindStr, "hidden_block_count_z")) return &h.blockCountZ;
    if (rcpMpStrEq(m, kindStr, "hidden_group_size_x")) return &h.groupSizeX;
    if (rcpMpStrEq(m, kindStr, "hidden_group_size_y")) return &h.groupSizeY;
    if (rcpMpStrEq(m, kindStr, "hidden_group_size_z")) return &h.groupSizeZ;
    if (rcpMpStrEq(m, kindStr, "hidden_remainder_x")) return &h.remainderX;
    if (rcpMpStrEq(m, kindStr, "hidden_remainder_y")) return &h.remainderY;
    if (rcpMpStrEq(m, kindStr, "hidden_remainder_z")) return &h.remainderZ;
    if (rcpMpStrEq(m, kindStr, "hidden_grid_dims")) return &h.gridDims;
    return nullptr;
}

/* Decode one arg map ({.offset, .size, .value_kind, ...}); if it is a hidden
 * arg we track, record its offset/size. The cursor is left just past the map. */
static void rcpMpDecodeArg(RcpMsgpack &m, RcpHiddenArgs &h) {
    RcpMpVal arg = rcpMpHead(m);
    if (m.err) return;
    if (arg.kind != MP_MAP) { rcpMpSkipChildren(m, arg); return; }
    uint64_t off = 0, sz = 0;
    bool haveOff = false, haveSz = false, haveKind = false;
    RcpMpVal kindVal; kindVal.kind = MP_OTHER; kindVal.strLen = 0; kindVal.strOff = 0; kindVal.uval = 0; kindVal.count = 0;
    for (uint32_t k = 0; k < arg.count && !m.err; k++) {
        RcpMpVal key = rcpMpHead(m);
        if (m.err) return;
        RcpMpVal val = rcpMpHead(m);
        if (m.err) return;
        if (rcpMpStrEq(m, key, ".offset") && val.kind == MP_INT) {
            off = val.uval; haveOff = true;
        } else if (rcpMpStrEq(m, key, ".size") && val.kind == MP_INT) {
            sz = val.uval; haveSz = true;
        } else if (rcpMpStrEq(m, key, ".value_kind") && val.kind == MP_STR) {
            kindVal = val; haveKind = true;
        } else {
            rcpMpSkipChildren(m, val);   /* arrays/maps need their children eaten */
        }
    }
    if (haveOff && haveSz && haveKind) {
        RcpHiddenArg *slot = rcpHiddenSlot(h, m, kindVal);
        if (slot) { slot->offset = (uint32_t)off; slot->size = (uint32_t)sz; slot->found = true; }
    }
}

/* Walk a kernel map looking for ".args"; decode each arg into h. */
static void rcpMpDecodeKernel(RcpMsgpack &m, RcpHiddenArgs &h) {
    RcpMpVal kmap = rcpMpHead(m);
    if (m.err) return;
    if (kmap.kind != MP_MAP) { rcpMpSkipChildren(m, kmap); return; }
    for (uint32_t k = 0; k < kmap.count && !m.err; k++) {
        RcpMpVal key = rcpMpHead(m);
        if (m.err) return;
        if (rcpMpStrEq(m, key, ".args")) {
            RcpMpVal args = rcpMpHead(m);
            if (m.err) return;
            if (args.kind == MP_ARRAY) {
                for (uint32_t a = 0; a < args.count && !m.err; a++)
                    rcpMpDecodeArg(m, h);
            } else {
                rcpMpSkipChildren(m, args);
            }
        } else {
            rcpMpSkipValue(m);   /* the value for this key */
        }
    }
}

/* Walk the top-level metadata map looking for "amdhsa.kernels" (an array of
 * kernel maps); decode the first kernel's args into h. */
static void rcpMpDecodeRoot(RcpMsgpack &m, RcpHiddenArgs &h) {
    RcpMpVal root = rcpMpHead(m);
    if (m.err) return;
    if (root.kind != MP_MAP) { rcpMpSkipChildren(m, root); return; }
    for (uint32_t k = 0; k < root.count && !m.err; k++) {
        RcpMpVal key = rcpMpHead(m);
        if (m.err) return;
        if (rcpMpStrEq(m, key, "amdhsa.kernels")) {
            RcpMpVal arr = rcpMpHead(m);
            if (m.err) return;
            if (arr.kind == MP_ARRAY && arr.count >= 1) {
                rcpMpDecodeKernel(m, h);          /* first kernel only */
                /* (don't bother decoding the rest; one kernel in this .co) */
            } else {
                rcpMpSkipChildren(m, arr);
            }
            return;
        }
        rcpMpSkipValue(m);   /* the value for this key */
    }
}

/* Find the NT_AMDGPU_METADATA (type 32) note in any SHT_NOTE (7) section and
 * decode the hidden-arg offsets/sizes into h. Returns true iff all ten hidden
 * args we populate were located. Mirrors kernel/metadata.py: parse the note
 * header (namesz/descsz/ntype, 4-byte aligned name+desc), then msgpack-decode
 * the descriptor for amdhsa.kernels[0].args[]. */
static bool rcpFindHiddenArgs(const std::vector<uint8_t> &co, const RcpElf &elf,
                              RcpHiddenArgs &h) {
    memset(&h, 0, sizeof(h));
    const uint32_t SHT_NOTE = 7;
    const uint32_t NT_AMDGPU_METADATA = 32;
    for (const auto &s : elf.sections) {
        if (s.sh_type != SHT_NOTE) continue;
        size_t base = (size_t)s.sh_offset;
        size_t end = base + (size_t)s.sh_size;
        if (end > co.size()) continue;
        size_t j = base;
        while (j + 12 <= end) {
            uint32_t namesz = rcpElfU32(co, j + 0);
            uint32_t descsz = rcpElfU32(co, j + 4);
            uint32_t ntype = rcpElfU32(co, j + 8);
            j += 12;
            size_t nameAdv = (namesz + 3u) & ~3u;
            size_t descAdv = (descsz + 3u) & ~3u;
            if (j + nameAdv + descAdv > end) break;
            size_t nameOff = j;
            size_t descOff = j + nameAdv;
            bool isAmd = (namesz >= 6) && memcmp(&co[nameOff], "AMDGPU", 6) == 0;
            if (ntype == NT_AMDGPU_METADATA && isAmd) {
                RcpMsgpack m;
                m.p = &co[descOff];
                m.n = descsz;
                m.i = 0;
                m.err = false;
                rcpMpDecodeRoot(m, h);
                bool all = h.blockCountX.found && h.blockCountY.found &&
                           h.blockCountZ.found && h.groupSizeX.found &&
                           h.groupSizeY.found && h.groupSizeZ.found &&
                           h.remainderX.found && h.remainderY.found &&
                           h.remainderZ.found && h.gridDims.found;
                return all && !m.err;
            }
            j += nameAdv + descAdv;
        }
    }
    return false;
}

/* Write `value` little-endian, `size` bytes, at kernarg byte offset `off`. */
static void rcpKbStore(uint8_t *kb, uint32_t off, uint64_t value, uint32_t size) {
    for (uint32_t b = 0; b < size; b++)
        kb[off + b] = (uint8_t)((value >> (b * 8)) & 0xFF);
}

/* Fill one hidden arg in the kernarg buffer at its metadata offset/size, with
 * the width the metadata declares (u32 block_count, u16 group_size/remainder/
 * grid_dims). The value is masked to the declared width. */
static void rcpFillHidden(uint8_t *kb, const RcpHiddenArg &a, uint64_t value) {
    if (!a.found || a.size == 0 || a.size > 8) return;
    uint64_t mask = (a.size >= 8) ? ~0ull : ((1ull << (a.size * 8)) - 1);
    rcpKbStore(kb, a.offset, value & mask, a.size);
}

bool recipeMultiWgDispatch(WddmLite &gpu, const IpDiscoveryResult &ipd,
                           const char *fwDir, uint64_t vramMcBase)
{
    /* 1. PSP cold-boot autoload -> BOOTLOAD_COMPLETE. */
    if (!recipeBootload(gpu, ipd, fwDir, vramMcBase)) {
        printf("  MWG: recipeBootload did not reach BOOTLOAD_COMPLETE\n");
        return false;
    }

    printf("\n=== recipeMultiWgDispatch (real kernel, multi-workgroup) ===\n");

    const uint32_t grid = MWG_GRID_X;
    const uint32_t block = MWG_BLOCK_X;
    const uint32_t outN = grid * block;                 /* total output dwords */
    const uint64_t outBytes = (uint64_t)outN * 4;
    const uint32_t outPages = (uint32_t)((outBytes + 0xFFF) / 0x1000);
    printf("  MWG: grid=%u (workgroups) block=%u (threads/wg) N=%u dwords "
           "out=%llu bytes (%u pages)\n",
           grid, block, outN, (unsigned long long)outBytes, outPages);

    /* Load + parse the compiled kernel from fwDir. */
    std::vector<uint8_t> co;
    {
        char path[512];
        snprintf(path, sizeof(path), "%s\\%s", fwDir, KERN_CO_FILE);
        if (!loadFirmwareFile(path, co)) {
            printf("  MWG: ERROR cannot read kernel %s\n", path);
            return false;
        }
    }
    RcpElf elf;
    if (!rcpElfParse(co, elf)) {
        printf("  MWG: ERROR failed to parse %s as AMDGPU ELF\n", KERN_CO_FILE);
        return false;
    }

    /* Kernel entry + descriptor symbols (same lookup as recipeKernargDispatch). */
    const RcpElfSymbol *kSym = nullptr;
    for (const auto &s : elf.symbols) {
        uint8_t type = s.st_info & 0xF;
        uint8_t bind = s.st_info >> 4;
        if (type == RCP_ELF_STT_AMDGPU_HSA_KERNEL ||
            (type == RCP_ELF_STT_FUNC && bind == RCP_ELF_STB_GLOBAL)) {
            kSym = &s;
            break;
        }
    }
    if (!kSym) { printf("  MWG: ERROR no kernel symbol\n"); return false; }
    char kdName[128];
    snprintf(kdName, sizeof(kdName), "%s.kd", kSym->name);
    const RcpElfSymbol *kdSym = nullptr;
    for (const auto &s : elf.symbols)
        if (strcmp(s.name, kdName) == 0) { kdSym = &s; break; }
    if (!kdSym) { printf("  MWG: ERROR no kernel descriptor %s\n", kdName); return false; }
    const RcpElfSection *kdSec = nullptr;
    for (const auto &s : elf.sections)
        if (s.sh_size && s.sh_addr <= kdSym->st_value &&
            kdSym->st_value < s.sh_addr + s.sh_size) { kdSec = &s; break; }
    if (!kdSec) { printf("  MWG: ERROR kernel descriptor section not found\n"); return false; }
    RcpKernelDescriptor kd;
    size_t kdOff = (size_t)kdSec->sh_offset + (size_t)(kdSym->st_value - kdSec->sh_addr);
    if (!rcpKdFromBytes(co, kdOff, kd)) {
        printf("  MWG: ERROR short kernel descriptor\n");
        return false;
    }

    /* Parse the COV5 hidden-arg offsets from the .co metadata note (NOT
     * hardcoded; we cross-checked these against probe_kernel_mw.py). */
    RcpHiddenArgs ha;
    if (!rcpFindHiddenArgs(co, elf, ha)) {
        printf("  MWG: ERROR could not parse all COV5 hidden args from metadata\n");
        return false;
    }
    printf("  MWG: kernel='%s' RSRC1=0x%08X RSRC2=0x%08X RSRC3=0x%08X "
           "kernarg=%u props=0x%X\n",
           kSym->name, kd.compute_pgm_rsrc1, kd.compute_pgm_rsrc2,
           kd.compute_pgm_rsrc3, kd.kernarg_size, kd.kernel_code_properties);
    printf("  MWG: hidden offsets block_count_x=%u(u%u) group_size_x=%u(u%u) "
           "remainder_x=%u(u%u) grid_dims=%u(u%u)\n",
           ha.blockCountX.offset, ha.blockCountX.size * 8,
           ha.groupSizeX.offset, ha.groupSizeX.size * 8,
           ha.remainderX.offset, ha.remainderX.size * 8,
           ha.gridDims.offset, ha.gridDims.size * 8);

    /* Assemble the loadable image by SECTION VADDR. */
    uint64_t imgHi = 0;
    for (const auto &s : elf.sections)
        if ((s.sh_flags & RCP_ELF_SHF_ALLOC) && s.sh_addr) {
            uint64_t hi = s.sh_addr + s.sh_size;
            if (hi > imgHi) imgHi = hi;
        }
    if (imgHi == 0) { printf("  MWG: ERROR no allocatable sections\n"); return false; }
    uint64_t imgBytes = (imgHi + 0xFFF) & ~0xFFFull;
    uint32_t codePages = (uint32_t)(imgBytes / 0x1000);
    std::vector<uint8_t> image((size_t)imgBytes, 0);
    for (const auto &s : elf.sections)
        if ((s.sh_flags & RCP_ELF_SHF_ALLOC) && s.sh_addr &&
            s.sh_type != RCP_ELF_SHT_NOBITS) {
            if ((size_t)(s.sh_offset + s.sh_size) > co.size()) continue;
            memcpy(&image[(size_t)s.sh_addr], &co[(size_t)s.sh_offset],
                   (size_t)s.sh_size);
        }

    /* --- bring up the GFX/MEC and GFXHUB (same as recipeKernargDispatch) --- */
    CqState cq;
    memset(&cq, 0, sizeof(cq));
    cq.gcBase0 = ipd.gcBase;
    cq.gcBase1 = ipd.gcBase1;

    cq.hasNbif = cqResolveNbif(ipd, &cq.nbifBase2);
    if (cq.hasNbif) {
        uint32_t apEn = 0;
        gpu.readReg32((cq.nbifBase2 + regRCC_DOORBELL_APER_EN) * 4, &apEn);
        gpu.writeReg32((cq.nbifBase2 + regRCC_DOORBELL_APER_EN) * 4,
                       apEn | BIF_DOORBELL_APER_EN__BIT);
        uint32_t fbEn = 0;
        gpu.readReg32((cq.nbifBase2 + regBIF_FB_EN) * 4, &fbEn);
        gpu.writeReg32((cq.nbifBase2 + regBIF_FB_EN) * 4,
                       fbEn | BIF_FB_EN__FB_READ_EN | BIF_FB_EN__FB_WRITE_EN);
        printf("  NBIO: doorbell aperture + framebuffer enabled "
               "(NBIF base[2]=0x%04X)\n", cq.nbifBase2);
    } else {
        printf("  NBIO: WARNING NBIF base[2] not found; HDP flush + doorbell "
               "aperture skipped\n");
    }

    const char *gc = "12_0_1";
    bool okP = false, okM = false, okC = false;
    uint64_t pfp = cqUcodeStart(fwDir, gc, "pfp", 52, &okP);
    uint64_t me = cqUcodeStart(fwDir, gc, "me", 52, &okM);
    uint64_t mec = cqUcodeStart(fwDir, gc, "mec", 52, &okC);
    bool haveUcode = okP && okM && okC;
    if (haveUcode)
        printf("  GFX: ucode_start PFP=0x%llX ME=0x%llX MEC=0x%llX\n",
               (unsigned long long)pfp, (unsigned long long)me,
               (unsigned long long)mec);

    if (!cqInitGfxForCompute(gpu, cq, pfp, me, mec, haveUcode))
        return false;

    void *gartCpu = nullptr, *dummyCpu = nullptr;
    uint64_t gartBus = 0, dummyBus = 0;
    void *gartHandle = nullptr, *dummyHandle = nullptr;
    if (!gpu.allocDma(1 << 20, &gartCpu, &gartBus, &gartHandle)) {
        printf("  MWG: ERROR GART table DMA alloc failed\n");
        return false;
    }
    if (!gpu.allocDma(4096, &dummyCpu, &dummyBus, &dummyHandle)) {
        printf("  MWG: ERROR dummy page DMA alloc failed\n");
        return false;
    }
    memset(gartCpu, 0, 1 << 20);
    memset(dummyCpu, 0, 4096);

    GfxhubParams gp;
    memset(&gp, 0, sizeof(gp));
    uint32_t fbBase = 0, fbTop = 0;
    mmhubRead(gpu, ipd, regMMMC_VM_FB_LOCATION_BASE, &fbBase);
    mmhubRead(gpu, ipd, regMMMC_VM_FB_LOCATION_TOP, &fbTop);
    gp.vramStart = (uint64_t)fbBase << 24;
    gp.vramEnd = ((uint64_t)fbTop << 24) | 0xFFFFFF;
    gp.gartStart = gp.vramEnd + 1;
    gp.gartEnd = gp.gartStart + (512ull * 1024 * 1024) - 1;
    gp.agpStart = gp.gartEnd + 1;
    gp.agpEnd = gp.agpStart;
    gp.gartTableBus = gartBus;
    gp.dummyPageBus = dummyBus;
    gfxhubGartEnable(gpu, cq, gp);

    /* --- stage code + kernarg + (multi-page) output in VRAM ---------------- */
    void *codeCpu = nullptr;
    uint64_t codeGpu = 0, codeHandle = 0;
    if (!rcpAllocVram(gpu, imgBytes, &codeCpu, &codeGpu, &codeHandle)) {
        printf("  MWG: ERROR code VRAM alloc failed\n");
        return false;
    }
    memcpy(codeCpu, image.data(), (size_t)imgBytes);

    void *kaCpu = nullptr;
    uint64_t kaGpu = 0, kaHandle = 0;
    if (!rcpAllocVram(gpu, 4096, &kaCpu, &kaGpu, &kaHandle)) {
        printf("  MWG: ERROR kernarg VRAM alloc failed\n");
        return false;
    }

    /* Output buffer sized N*4 bytes, rounded to whole pages. rcpAllocVram
     * rounds the size up to a page and zeroes every page; the returned MC
     * address is contiguous so out page p lives at outGpu + p*4096. */
    uint64_t outAllocBytes = (uint64_t)outPages * 0x1000;
    void *outCpu = nullptr;
    uint64_t outGpu = 0, outHandle = 0;
    if (!rcpAllocVram(gpu, outAllocBytes, &outCpu, &outGpu, &outHandle)) {
        printf("  MWG: ERROR output VRAM alloc failed\n");
        return false;
    }

    /* VA map: code at COMPUTE_VA_ROOT (codePages pages), then kernarg, then the
     * outPages output pages -- all in one 2MB block / one PTB. */
    uint64_t codeVa = COMPUTE_VA_ROOT;
    uint64_t kaVa = COMPUTE_VA_ROOT + (uint64_t)codePages * 0x1000;
    uint64_t outVa = COMPUTE_VA_ROOT + (uint64_t)(codePages + 1) * 0x1000;
    uint64_t codeEntryVa = codeVa +
        ((uint64_t)kdSym->st_value + (uint64_t)kd.kernel_code_entry_byte_offset);

    /* Fill the kernarg: explicit args + the COV5 hidden args at their metadata
     * offsets. block_count = GRID (workgroups), group_size = BLOCK (threads). */
    {
        uint8_t *kb = (uint8_t *)kaCpu;
        rcpKbStore(kb, 0, outVa, 8);              /* [0:8] output VA  (u64) */
        rcpKbStore(kb, 8, MWG_FILL_VAL, 4);       /* [8:12] fill val (u32) */
        /* 1-D dispatch (only x varies), so grid_dims = 1 (matches the probe). */
        uint32_t dims = 1u;
        rcpFillHidden(kb, ha.blockCountX, grid);  /* u32 # workgroups */
        rcpFillHidden(kb, ha.blockCountY, 1);
        rcpFillHidden(kb, ha.blockCountZ, 1);
        rcpFillHidden(kb, ha.groupSizeX, block);  /* u16 threads/wg = 64 */
        rcpFillHidden(kb, ha.groupSizeY, 1);
        rcpFillHidden(kb, ha.groupSizeZ, 1);
        rcpFillHidden(kb, ha.remainderX, 0);      /* uniform_work_group_size=1 */
        rcpFillHidden(kb, ha.remainderY, 0);
        rcpFillHidden(kb, ha.remainderZ, 0);
        rcpFillHidden(kb, ha.gridDims, dims);     /* u16 grid dims = 1 */
    }

    /* Build the multi-region page table: every code page + kernarg + EVERY
     * output page. Without mapping all output pages, workgroups whose global
     * id lands past the first 4KB would GPUVM-fault. */
    std::vector<GpuvmRegion> regions;
    regions.reserve(codePages + 1 + outPages);
    for (uint32_t p = 0; p < codePages; p++)
        regions.push_back({ codeVa + (uint64_t)p * 0x1000,
                            codeGpu + (uint64_t)p * 0x1000 });
    regions.push_back({ kaVa, kaGpu });
    for (uint32_t p = 0; p < outPages; p++)
        regions.push_back({ outVa + (uint64_t)p * 0x1000,
                            outGpu + (uint64_t)p * 0x1000 });

    DispatchGpuvm pt;
    memset(&pt, 0, sizeof(pt));
    if (!buildComputeGpuvmMulti(gpu, cq, vramMcBase, regions.data(),
                                (uint32_t)regions.size(), pt))
        return false;
    printf("  MWG: mapped %u code + 1 kernarg + %u output pages "
           "(out VA 0x%llX..0x%llX)\n",
           codePages, outPages, (unsigned long long)outVa,
           (unsigned long long)(outVa + outBytes - 1));

    if (!cqInitComputeQueue(gpu, cq))
        return false;

    uint32_t c0 = gcReg(gpu, cq, regGCVM_CONTEXT0_CNTL, 0);
    printf("  GPUVM: GCVM_CONTEXT0_CNTL readback=0x%08X (enable=%u)\n",
           c0, c0 & 1u);

    /* kernarg_sgpr_index (same derivation as recipeKernargDispatch). */
    uint32_t slot = 0;
    if (kd.kernel_code_properties & (1u << 0)) slot += 4;
    if (kd.kernel_code_properties & (1u << 1)) slot += 2;
    if (kd.kernel_code_properties & (1u << 2)) slot += 2;

    printf("  MWG: code_entry=0x%llX ka_va=0x%llX out_va=0x%llX "
           "code_pages=%u sgpr_slot=%u\n",
           (unsigned long long)codeEntryVa, (unsigned long long)kaVa,
           (unsigned long long)outVa, codePages, slot);

    gpu.writeReg32((cq.gcBase0 + regGCVM_L2_PROTECTION_FAULT_STATUS) * 4, 0);
    gpu.writeReg32((cq.gcBase0 + (regGCVM_L2_PROTECTION_FAULT_STATUS + 1)) * 4, 0);

    uint64_t pgm = codeEntryVa >> 8;
    uint64_t fenceSeq = 1;
    *cq.fenceCpu = 0;

    std::vector<uint32_t> packets;
    pm4AcquireMem(packets);

    uint32_t pgmLoHi[2] = { (uint32_t)(pgm & 0xFFFFFFFF),
                            (uint32_t)((pgm >> 32) & 0xFFFFFFFF) };
    pm4SetShReg(packets, regCOMPUTE_PGM_LO, pgmLoHi, 2);

    uint32_t rsrc12[2] = { kd.compute_pgm_rsrc1, kd.compute_pgm_rsrc2 };
    pm4SetShReg(packets, regCOMPUTE_PGM_RSRC1, rsrc12, 2);

    uint32_t rsrc3 = kd.compute_pgm_rsrc3;
    pm4SetShReg(packets, regCOMPUTE_PGM_RSRC3, &rsrc3, 1);

    uint32_t tmpring = 0;
    pm4SetShReg(packets, regCOMPUTE_TMPRING_SIZE, &tmpring, 1);

    uint32_t restart[3] = { 0, 0, 0 };
    pm4SetShReg(packets, regCOMPUTE_RESTART_X, restart, 3);

    uint32_t userData[2] = { (uint32_t)(kaVa & 0xFFFFFFFF),
                             (uint32_t)((kaVa >> 32) & 0xFFFFFFFF) };
    pm4SetShReg(packets, regCOMPUTE_USER_DATA_0 + slot, userData, 2);

    uint32_t resLimits = 0;
    pm4SetShReg(packets, regCOMPUTE_RESOURCE_LIMITS, &resLimits, 1);

    /* COMPUTE_START_X..: start xyz=0, NUM_THREAD xyz=(BLOCK,1,1), 2 trailing 0.
     * NUM_THREAD_X is threads PER WORKGROUP -- it is NOT the dispatch DIM_X. */
    uint32_t startBlock[8] = { 0, 0, 0, block, 1, 1, 0, 0 };
    pm4SetShReg(packets, regCOMPUTE_START_X, startBlock, 8);

    /* DISPATCH_DIRECT DIM_X = GRID = number of WORKGROUPS (not threads). */
    pm4DispatchDirect(packets, grid, 1, 1, DISPATCH_INITIATOR_W32);
    pm4EventWrite(packets, CS_PARTIAL_FLUSH, EVENT_INDEX_CS_PARTIAL_FLUSH);
    pm4ReleaseMemFence(packets, cq.fenceGpu, fenceSeq);

    cqSubmitPackets(gpu, cq, packets);

    bool ok = cqWaitFence(cq, fenceSeq, 5000);
    cqHdpFlush(gpu, cq);

    uint32_t faultStatus = gcReg(gpu, cq, regGCVM_L2_PROTECTION_FAULT_STATUS, 0);
    uint32_t faultAddrLo = gcReg(gpu, cq, regGCVM_L2_PROTECTION_FAULT_ADDR_LO32, 0);
    uint32_t faultAddrHi = gcReg(gpu, cq, regGCVM_L2_PROTECTION_FAULT_ADDR_HI32, 0);
    uint64_t faultVa = (((uint64_t)faultAddrHi << 32) | faultAddrLo) << 12;

    cqGrbmSelect(gpu, cq, cq.me, cq.pipe, cq.queue, 0);
    uint32_t rptr = gcReg(gpu, cq, regCP_HQD_PQ_RPTR, 0);
    cqGrbmDeselect(gpu, cq);
    uint64_t fenceVal = *cq.fenceCpu;

    /* Verify ALL N output dwords across every page. */
    const volatile uint32_t *res = (const volatile uint32_t *)outCpu;
    uint32_t nbad = 0, firstBad = 0, firstBadVal = 0;
    for (uint32_t i = 0; i < outN; i++) {
        uint32_t v = res[i];
        if (v != MWG_FILL_VAL) {
            if (nbad == 0) { firstBad = i; firstBadVal = v; }
            nbad++;
        }
    }

    uint32_t walker = (faultStatus >> 1) & 0x7;
    uint32_t perm = (faultStatus >> 4) & 0xF;

    printf("\nFAULT_STATUS=0x%08X [walker=%u perm=0x%X] FAULT_VA=0x%llX\n",
           faultStatus, walker, perm, (unsigned long long)faultVa);
    printf("FENCE value=%llu (expected %llu) RPTR=0x%X\n",
           (unsigned long long)fenceVal, (unsigned long long)fenceSeq, rptr);
    /* Sample a dword from the last page so the log shows cross-page coverage. */
    uint32_t lastIdx = outN - 1;
    printf("out[0]=0x%08X out[%u]=0x%08X out[%u]=0x%08X expected=0x%08X "
           "bad=%u/%u\n",
           res[0], block, res[block], lastIdx, res[lastIdx],
           MWG_FILL_VAL, nbad, outN);
    if (nbad)
        printf("  first mismatch at index %u (dword page %u): got 0x%08X\n",
               firstBad, (firstBad * 4) / 0x1000, firstBadVal);

    bool pass = ok && (faultStatus == 0) && (nbad == 0);
    printf("MULTI-WG DISPATCH %s (grid=%u x block=%u, %u pages)\n",
           pass ? "PASS (fence signaled, no fault, all output verified)"
           : (!ok ? "FAIL (fence timeout)"
              : (faultStatus ? "FAIL (GPUVM fault)" : "FAIL (output mismatch)")),
           grid, block, outPages);
    return pass;
}

/* ======================================================================
 * recipeScratchDispatch (increment 3c): a kernel that SPILLS registers
 * ======================================================================
 * Mirrors python/probe_kernel_scratch.py (the PASSED scratch probe on the
 * Windows DIRECT MEC HQD) byte-for-byte. The previous increments dispatched
 * kernels with no private segment (KD.private_segment_fixed_size == 0). This
 * one runs scratch_kernel.co, which spills registers to the per-wave PRIVATE
 * (scratch) segment, so the architected flat scratch path must be programmed.
 *
 * gfx12 architected flat scratch (proven on a direct MEC HQD): the SPI hands
 * each wave its FLAT_SCRATCH from COMPUTE_DISPATCH_SCRATCH_BASE_LO/HI plus the
 * per-wave COMPUTE_TMPRING_SIZE.WAVESIZE offset. There is NO private_segment
 * buffer V# (KD code-properties bit0/bit5 = 0). Steps added vs the multi-WG
 * fill:
 *   - SH_MEM_BASES = 0x00010002 (PRIVATE_BASE field = 2) +
 *     SH_MEM_CONFIG = 0xC00C (alignment-mode UNALIGNED) for VMID 0, so the
 *     private aperture and sub-dword scratch work. (cqInitGfxForCompute
 *     already writes these identical values for all 16 VMIDs; the probe
 *     re-asserts them for VMID 0 after the queue is up, and so do we.)
 *   - allocate a VRAM scratch backing, GPUVM-map it into the same PTB, set
 *     COMPUTE_DISPATCH_SCRATCH_BASE = scratch_VA >> 8.
 *   - COMPUTE_TMPRING_SIZE = WAVES | (WAVESIZE << 12).
 *   - COMPUTE_PGM_RSRC2 from the KD already has SCRATCH_EN (bit0) = 1.
 *
 * scratch_kernel ABI (probe_kernel_scratch.py): explicit args are the same
 * shape as fill_kernel -- kernarg[0:8] = output VA (u64), kernarg[8:12] =
 * u32 val -- followed by the standard COV5 hidden args (block_count/group_size/
 * grid_dims) at the metadata-declared offsets. Every thread computes
 *   s = sum(val + (i*7 + val) % 64  for i in 0..63)
 * which spills enough live values to force the private segment. val=1 ->
 * s = 2080 (0x820); out[0 .. N-1] all == 2080.
 *
 * Parameters match the probe exactly: GRID=4 workgroups, BLOCK=64 threads.
 * ====================================================================== */

#define SCR_CO_FILE     "scratch_kernel.co"
#define SCR_VAL         1u                 /* by_value arg -> EXPECT 2080 */
#define SCR_BLOCK_X     64u                /* threads per workgroup (probe) */
#define SCR_GRID_X      4u                 /* workgroups (probe) */
#define SCR_LANES       32u                /* wave32 */
#define SCR_WAVES       32u                /* probe over-allocs 4x SE; fits 1 PTB */
/* SH aperture for VMID 0 (probe values; also already set by cqInitGfxForCompute). */
#define SCR_SH_MEM_CONFIG  0x0000C00Cu     /* alignment-mode UNALIGNED */
#define SCR_SH_MEM_BASES   0x00010002u     /* PRIVATE_BASE field = 2 */

bool recipeScratchDispatch(WddmLite &gpu, const IpDiscoveryResult &ipd,
                           const char *fwDir, uint64_t vramMcBase)
{
    /* 1. PSP cold-boot autoload -> BOOTLOAD_COMPLETE. */
    if (!recipeBootload(gpu, ipd, fwDir, vramMcBase)) {
        printf("  SCR: recipeBootload did not reach BOOTLOAD_COMPLETE\n");
        return false;
    }

    printf("\n=== recipeScratchDispatch (register-spilling kernel + scratch) ===\n");

    const uint32_t grid = SCR_GRID_X;
    const uint32_t block = SCR_BLOCK_X;
    const uint32_t outN = grid * block;                 /* total output dwords */
    const uint64_t outBytes = (uint64_t)outN * 4;
    const uint32_t outPages = (uint32_t)((outBytes + 0xFFF) / 0x1000);
    /* EXPECT = sum(val + (i*7 + val) % 64) for i in 0..63, masked to u32. */
    uint32_t expect = 0;
    for (uint32_t i = 0; i < 64; i++)
        expect += SCR_VAL + ((i * 7u + SCR_VAL) % 64u);
    printf("  SCR: grid=%u (workgroups) block=%u (threads/wg) N=%u dwords "
           "out=%llu bytes (%u pages) EXPECT=0x%X\n",
           grid, block, outN, (unsigned long long)outBytes, outPages, expect);

    /* Load + parse the compiled kernel from fwDir. */
    std::vector<uint8_t> co;
    {
        char path[512];
        snprintf(path, sizeof(path), "%s\\%s", fwDir, SCR_CO_FILE);
        if (!loadFirmwareFile(path, co)) {
            printf("  SCR: ERROR cannot read kernel %s\n", path);
            return false;
        }
    }
    RcpElf elf;
    if (!rcpElfParse(co, elf)) {
        printf("  SCR: ERROR failed to parse %s as AMDGPU ELF\n", SCR_CO_FILE);
        return false;
    }

    /* Kernel entry + descriptor symbols (same lookup as recipeMultiWgDispatch). */
    const RcpElfSymbol *kSym = nullptr;
    for (const auto &s : elf.symbols) {
        uint8_t type = s.st_info & 0xF;
        uint8_t bind = s.st_info >> 4;
        if (type == RCP_ELF_STT_AMDGPU_HSA_KERNEL ||
            (type == RCP_ELF_STT_FUNC && bind == RCP_ELF_STB_GLOBAL)) {
            kSym = &s;
            break;
        }
    }
    if (!kSym) { printf("  SCR: ERROR no kernel symbol\n"); return false; }
    char kdName[128];
    snprintf(kdName, sizeof(kdName), "%s.kd", kSym->name);
    const RcpElfSymbol *kdSym = nullptr;
    for (const auto &s : elf.symbols)
        if (strcmp(s.name, kdName) == 0) { kdSym = &s; break; }
    if (!kdSym) { printf("  SCR: ERROR no kernel descriptor %s\n", kdName); return false; }
    const RcpElfSection *kdSec = nullptr;
    for (const auto &s : elf.sections)
        if (s.sh_size && s.sh_addr <= kdSym->st_value &&
            kdSym->st_value < s.sh_addr + s.sh_size) { kdSec = &s; break; }
    if (!kdSec) { printf("  SCR: ERROR kernel descriptor section not found\n"); return false; }
    RcpKernelDescriptor kd;
    size_t kdOff = (size_t)kdSec->sh_offset + (size_t)(kdSym->st_value - kdSec->sh_addr);
    if (!rcpKdFromBytes(co, kdOff, kd)) {
        printf("  SCR: ERROR short kernel descriptor\n");
        return false;
    }

    /* CONFIRM this kernel actually spills: private_segment_fixed_size > 0 and
     * COMPUTE_PGM_RSRC2.SCRATCH_EN (bit 0) == 1. Otherwise the scratch path is
     * not exercised and the test is meaningless. */
    uint32_t psfs = kd.private_segment_fixed_size;
    if (psfs == 0) {
        printf("  SCR: ERROR kernel '%s' has private_segment_fixed_size=0 "
               "(does not spill); not a scratch kernel\n", kSym->name);
        return false;
    }
    if ((kd.compute_pgm_rsrc2 & 0x1u) == 0) {
        printf("  SCR: ERROR RSRC2=0x%08X has SCRATCH_EN(bit0)=0 despite "
               "psfs=%u\n", kd.compute_pgm_rsrc2, psfs);
        return false;
    }

    /* Parse the COV5 hidden-arg offsets from the .co metadata note (NOT
     * hardcoded; the same parser the multi-WG path uses). The scratch kernel
     * has the identical explicit+hidden arg layout as fill_kernel. */
    RcpHiddenArgs ha;
    if (!rcpFindHiddenArgs(co, elf, ha)) {
        printf("  SCR: ERROR could not parse all COV5 hidden args from metadata\n");
        return false;
    }
    printf("  SCR: kernel='%s' psfs=%u RSRC1=0x%08X RSRC2=0x%08X "
           "RSRC3=0x%08X kernarg=%u props=0x%X SCRATCH_EN=%u\n",
           kSym->name, psfs, kd.compute_pgm_rsrc1, kd.compute_pgm_rsrc2,
           kd.compute_pgm_rsrc3, kd.kernarg_size, kd.kernel_code_properties,
           kd.compute_pgm_rsrc2 & 0x1u);

    /* --- Architected flat scratch sizing (probe_kernel_scratch.py) ----------
     * Per-thread scratch is rounded UP to a multiple of (256 / LANES) bytes so
     * that LANES threads consume a whole 256-byte granule:
     *   bpt        = roundup(psfs, 256/LANES)
     *   wave_bytes = bpt * LANES                 (bytes one wave needs)
     *   WAVESIZE   = roundup(wave_bytes, 256) / 256   (in 256-byte units)
     * The backing is wave_bytes * WAVES * 4 (the probe's 4x SE over-allocation),
     * rounded up to whole pages so it maps cleanly into the PTB.
     *   TMPRING    = (WAVES & 0xFFF) | ((WAVESIZE & 0x3FFFF) << 12)
     * Widths: WAVES is the low 12 bits (NUM_WAVES), WAVESIZE the next 18 bits
     * (WAVESIZE field) per COMPUTE_TMPRING_SIZE. */
    const uint32_t lanes = SCR_LANES;
    const uint32_t waves = SCR_WAVES;
    const uint32_t gran = 256u / lanes;                 /* 8 bytes for wave32 */
    uint32_t bpt = (psfs + gran - 1u) & ~(gran - 1u);    /* round psfs up */
    uint32_t waveBytes = bpt * lanes;
    uint32_t waveSize = (waveBytes + 255u) / 256u;       /* 256-byte units */
    uint64_t scratchBytes = ((uint64_t)waveBytes * waves * 4u + 0xFFFull) & ~0xFFFull;
    uint32_t scratchPages = (uint32_t)(scratchBytes / 0x1000);
    uint32_t tmpring = (waves & 0xFFFu) | ((waveSize & 0x3FFFFu) << 12);
    printf("  SCR: psfs=%u gran=%u bpt=%u wave_bytes=%u WAVESIZE=%u WAVES=%u "
           "TMPRING=0x%08X scratch=%lluKB (%u pages)\n",
           psfs, gran, bpt, waveBytes, waveSize, waves, tmpring,
           (unsigned long long)(scratchBytes / 1024), scratchPages);

    /* Assemble the loadable image by SECTION VADDR. */
    uint64_t imgHi = 0;
    for (const auto &s : elf.sections)
        if ((s.sh_flags & RCP_ELF_SHF_ALLOC) && s.sh_addr) {
            uint64_t hi = s.sh_addr + s.sh_size;
            if (hi > imgHi) imgHi = hi;
        }
    if (imgHi == 0) { printf("  SCR: ERROR no allocatable sections\n"); return false; }
    uint64_t imgBytes = (imgHi + 0xFFF) & ~0xFFFull;
    uint32_t codePages = (uint32_t)(imgBytes / 0x1000);
    std::vector<uint8_t> image((size_t)imgBytes, 0);
    for (const auto &s : elf.sections)
        if ((s.sh_flags & RCP_ELF_SHF_ALLOC) && s.sh_addr &&
            s.sh_type != RCP_ELF_SHT_NOBITS) {
            if ((size_t)(s.sh_offset + s.sh_size) > co.size()) continue;
            memcpy(&image[(size_t)s.sh_addr], &co[(size_t)s.sh_offset],
                   (size_t)s.sh_size);
        }

    /* --- bring up the GFX/MEC and GFXHUB (same as recipeMultiWgDispatch) --- */
    CqState cq;
    memset(&cq, 0, sizeof(cq));
    cq.gcBase0 = ipd.gcBase;
    cq.gcBase1 = ipd.gcBase1;

    cq.hasNbif = cqResolveNbif(ipd, &cq.nbifBase2);
    if (cq.hasNbif) {
        uint32_t apEn = 0;
        gpu.readReg32((cq.nbifBase2 + regRCC_DOORBELL_APER_EN) * 4, &apEn);
        gpu.writeReg32((cq.nbifBase2 + regRCC_DOORBELL_APER_EN) * 4,
                       apEn | BIF_DOORBELL_APER_EN__BIT);
        uint32_t fbEn = 0;
        gpu.readReg32((cq.nbifBase2 + regBIF_FB_EN) * 4, &fbEn);
        gpu.writeReg32((cq.nbifBase2 + regBIF_FB_EN) * 4,
                       fbEn | BIF_FB_EN__FB_READ_EN | BIF_FB_EN__FB_WRITE_EN);
        printf("  NBIO: doorbell aperture + framebuffer enabled "
               "(NBIF base[2]=0x%04X)\n", cq.nbifBase2);
    } else {
        printf("  NBIO: WARNING NBIF base[2] not found; HDP flush + doorbell "
               "aperture skipped\n");
    }

    const char *gc = "12_0_1";
    bool okP = false, okM = false, okC = false;
    uint64_t pfp = cqUcodeStart(fwDir, gc, "pfp", 52, &okP);
    uint64_t me = cqUcodeStart(fwDir, gc, "me", 52, &okM);
    uint64_t mec = cqUcodeStart(fwDir, gc, "mec", 52, &okC);
    bool haveUcode = okP && okM && okC;
    if (haveUcode)
        printf("  GFX: ucode_start PFP=0x%llX ME=0x%llX MEC=0x%llX\n",
               (unsigned long long)pfp, (unsigned long long)me,
               (unsigned long long)mec);

    if (!cqInitGfxForCompute(gpu, cq, pfp, me, mec, haveUcode))
        return false;

    void *gartCpu = nullptr, *dummyCpu = nullptr;
    uint64_t gartBus = 0, dummyBus = 0;
    void *gartHandle = nullptr, *dummyHandle = nullptr;
    if (!gpu.allocDma(1 << 20, &gartCpu, &gartBus, &gartHandle)) {
        printf("  SCR: ERROR GART table DMA alloc failed\n");
        return false;
    }
    if (!gpu.allocDma(4096, &dummyCpu, &dummyBus, &dummyHandle)) {
        printf("  SCR: ERROR dummy page DMA alloc failed\n");
        return false;
    }
    memset(gartCpu, 0, 1 << 20);
    memset(dummyCpu, 0, 4096);

    GfxhubParams gp;
    memset(&gp, 0, sizeof(gp));
    uint32_t fbBase = 0, fbTop = 0;
    mmhubRead(gpu, ipd, regMMMC_VM_FB_LOCATION_BASE, &fbBase);
    mmhubRead(gpu, ipd, regMMMC_VM_FB_LOCATION_TOP, &fbTop);
    gp.vramStart = (uint64_t)fbBase << 24;
    gp.vramEnd = ((uint64_t)fbTop << 24) | 0xFFFFFF;
    gp.gartStart = gp.vramEnd + 1;
    gp.gartEnd = gp.gartStart + (512ull * 1024 * 1024) - 1;
    gp.agpStart = gp.gartEnd + 1;
    gp.agpEnd = gp.agpStart;
    gp.gartTableBus = gartBus;
    gp.dummyPageBus = dummyBus;
    gfxhubGartEnable(gpu, cq, gp);

    /* --- stage code + kernarg + output + scratch in VRAM ------------------- */
    void *codeCpu = nullptr;
    uint64_t codeGpu = 0, codeHandle = 0;
    if (!rcpAllocVram(gpu, imgBytes, &codeCpu, &codeGpu, &codeHandle)) {
        printf("  SCR: ERROR code VRAM alloc failed\n");
        return false;
    }
    memcpy(codeCpu, image.data(), (size_t)imgBytes);

    void *kaCpu = nullptr;
    uint64_t kaGpu = 0, kaHandle = 0;
    if (!rcpAllocVram(gpu, 4096, &kaCpu, &kaGpu, &kaHandle)) {
        printf("  SCR: ERROR kernarg VRAM alloc failed\n");
        return false;
    }

    uint64_t outAllocBytes = (uint64_t)outPages * 0x1000;
    void *outCpu = nullptr;
    uint64_t outGpu = 0, outHandle = 0;
    if (!rcpAllocVram(gpu, outAllocBytes, &outCpu, &outGpu, &outHandle)) {
        printf("  SCR: ERROR output VRAM alloc failed\n");
        return false;
    }

    /* Scratch backing on the SAME FB-MC bump allocator. The kernel never CPU-
     * touches it; the SPI writes/reads it as private memory. We map it into
     * GPUVM (architected flat scratch on gfx12 uses the VA aperture: SH_MEM
     * PRIVATE_BASE + SCRATCH_BASE = scratch_VA>>8), so it must live in the same
     * 4-level page table as code/kernarg/output -- it is NOT physically
     * addressed. */
    void *scrCpu = nullptr;
    uint64_t scrGpu = 0, scrHandle = 0;
    if (!rcpAllocVram(gpu, scratchBytes, &scrCpu, &scrGpu, &scrHandle)) {
        printf("  SCR: ERROR scratch VRAM alloc failed (%llu bytes)\n",
               (unsigned long long)scratchBytes);
        return false;
    }
    /* rcpAllocVram zeroes every page of code/kernarg/output/scratch. */

    /* VA map: code (codePages), kernarg, output (outPages), scratch
     * (scratchPages) -- all in one 2MB block / one PTB. */
    uint64_t codeVa = COMPUTE_VA_ROOT;
    uint64_t kaVa = COMPUTE_VA_ROOT + (uint64_t)codePages * 0x1000;
    uint64_t outVa = COMPUTE_VA_ROOT + (uint64_t)(codePages + 1) * 0x1000;
    uint64_t scrVa = COMPUTE_VA_ROOT + (uint64_t)(codePages + 1 + outPages) * 0x1000;
    uint64_t codeEntryVa = codeVa +
        ((uint64_t)kdSym->st_value + (uint64_t)kd.kernel_code_entry_byte_offset);

    /* Fill the kernarg: explicit args + COV5 hidden args at metadata offsets.
     * block_count = GRID (workgroups), group_size = BLOCK (threads). */
    {
        uint8_t *kb = (uint8_t *)kaCpu;
        rcpKbStore(kb, 0, outVa, 8);              /* [0:8] output VA  (u64) */
        rcpKbStore(kb, 8, SCR_VAL, 4);            /* [8:12] by_value  (u32) */
        rcpFillHidden(kb, ha.blockCountX, grid);  /* u32 # workgroups */
        rcpFillHidden(kb, ha.blockCountY, 1);
        rcpFillHidden(kb, ha.blockCountZ, 1);
        rcpFillHidden(kb, ha.groupSizeX, block);  /* u16 threads/wg = 64 */
        rcpFillHidden(kb, ha.groupSizeY, 1);
        rcpFillHidden(kb, ha.groupSizeZ, 1);
        rcpFillHidden(kb, ha.remainderX, 0);      /* uniform_work_group_size=1 */
        rcpFillHidden(kb, ha.remainderY, 0);
        rcpFillHidden(kb, ha.remainderZ, 0);
        rcpFillHidden(kb, ha.gridDims, 1);        /* u16 grid dims = 1 */
    }

    /* Page table: every code page + kernarg + every output page + every
     * scratch page. The scratch pages MUST be mapped or the SPI faults the
     * first spill (GCVM walker fault on the scratch VA). */
    std::vector<GpuvmRegion> regions;
    regions.reserve(codePages + 1 + outPages + scratchPages);
    for (uint32_t p = 0; p < codePages; p++)
        regions.push_back({ codeVa + (uint64_t)p * 0x1000,
                            codeGpu + (uint64_t)p * 0x1000 });
    regions.push_back({ kaVa, kaGpu });
    for (uint32_t p = 0; p < outPages; p++)
        regions.push_back({ outVa + (uint64_t)p * 0x1000,
                            outGpu + (uint64_t)p * 0x1000 });
    for (uint32_t p = 0; p < scratchPages; p++)
        regions.push_back({ scrVa + (uint64_t)p * 0x1000,
                            scrGpu + (uint64_t)p * 0x1000 });

    DispatchGpuvm pt;
    memset(&pt, 0, sizeof(pt));
    if (!buildComputeGpuvmMulti(gpu, cq, vramMcBase, regions.data(),
                                (uint32_t)regions.size(), pt))
        return false;
    printf("  SCR: mapped %u code + 1 kernarg + %u output + %u scratch pages "
           "(scratch VA 0x%llX..0x%llX)\n",
           codePages, outPages, scratchPages, (unsigned long long)scrVa,
           (unsigned long long)(scrVa + scratchBytes - 1));

    if (!cqInitComputeQueue(gpu, cq))
        return false;

    /* SH aperture for VMID 0: PRIVATE_BASE=2 + UNALIGNED, re-asserted under
     * grbm_select AFTER the queue is up (probe ordering). These are the same
     * values cqInitGfxForCompute already programmed for all 16 VMIDs. */
    cqGrbmSelect(gpu, cq, 0, 0, 0, 0);
    gcWreg(gpu, cq, regSH_MEM_CONFIG, SCR_SH_MEM_CONFIG, 1);
    gcWreg(gpu, cq, regSH_MEM_BASES, SCR_SH_MEM_BASES, 1);
    cqGrbmDeselect(gpu, cq);
    printf("  SCR: SH_MEM_CONFIG=0x%08X SH_MEM_BASES=0x%08X (VMID 0)\n",
           SCR_SH_MEM_CONFIG, SCR_SH_MEM_BASES);

    uint32_t c0 = gcReg(gpu, cq, regGCVM_CONTEXT0_CNTL, 0);
    printf("  GPUVM: GCVM_CONTEXT0_CNTL readback=0x%08X (enable=%u)\n",
           c0, c0 & 1u);

    /* kernarg_sgpr_index (same derivation as the other recipes; props 0/1/2). */
    uint32_t slot = 0;
    if (kd.kernel_code_properties & (1u << 0)) slot += 4;
    if (kd.kernel_code_properties & (1u << 1)) slot += 2;
    if (kd.kernel_code_properties & (1u << 2)) slot += 2;

    uint64_t sbase = scrVa >> 8;          /* COMPUTE_DISPATCH_SCRATCH_BASE = VA>>8 */
    printf("  SCR: code_entry=0x%llX ka_va=0x%llX out_va=0x%llX scr_va=0x%llX "
           "SCRATCH_BASE=0x%llX code_pages=%u sgpr_slot=%u\n",
           (unsigned long long)codeEntryVa, (unsigned long long)kaVa,
           (unsigned long long)outVa, (unsigned long long)scrVa,
           (unsigned long long)sbase, codePages, slot);

    gpu.writeReg32((cq.gcBase0 + regGCVM_L2_PROTECTION_FAULT_STATUS) * 4, 0);
    gpu.writeReg32((cq.gcBase0 + (regGCVM_L2_PROTECTION_FAULT_STATUS + 1)) * 4, 0);

    uint64_t pgm = codeEntryVa >> 8;
    uint64_t fenceSeq = 1;
    *cq.fenceCpu = 0;

    std::vector<uint32_t> packets;
    pm4AcquireMem(packets);

    uint32_t pgmLoHi[2] = { (uint32_t)(pgm & 0xFFFFFFFF),
                            (uint32_t)((pgm >> 32) & 0xFFFFFFFF) };
    pm4SetShReg(packets, regCOMPUTE_PGM_LO, pgmLoHi, 2);

    /* COMPUTE_DISPATCH_SCRATCH_BASE_LO/HI = scratch_VA >> 8. The HI register
     * holds the top 8 bits of the >>8 base (a 40-bit MC address). */
    uint32_t sbaseLoHi[2] = { (uint32_t)(sbase & 0xFFFFFFFF),
                              (uint32_t)((sbase >> 32) & 0xFFu) };
    pm4SetShReg(packets, regCOMPUTE_DISPATCH_SCRATCH_BASE_LO, sbaseLoHi, 2);

    /* RSRC1 + RSRC2 straight from the KD; RSRC2 already has SCRATCH_EN (bit0). */
    uint32_t rsrc12[2] = { kd.compute_pgm_rsrc1, kd.compute_pgm_rsrc2 };
    pm4SetShReg(packets, regCOMPUTE_PGM_RSRC1, rsrc12, 2);

    uint32_t rsrc3 = kd.compute_pgm_rsrc3;
    pm4SetShReg(packets, regCOMPUTE_PGM_RSRC3, &rsrc3, 1);

    /* COMPUTE_TMPRING_SIZE = WAVES | (WAVESIZE << 12) -- the per-wave scratch. */
    pm4SetShReg(packets, regCOMPUTE_TMPRING_SIZE, &tmpring, 1);

    uint32_t restart[3] = { 0, 0, 0 };
    pm4SetShReg(packets, regCOMPUTE_RESTART_X, restart, 3);

    uint32_t userData[2] = { (uint32_t)(kaVa & 0xFFFFFFFF),
                             (uint32_t)((kaVa >> 32) & 0xFFFFFFFF) };
    pm4SetShReg(packets, regCOMPUTE_USER_DATA_0 + slot, userData, 2);

    uint32_t resLimits = 0;
    pm4SetShReg(packets, regCOMPUTE_RESOURCE_LIMITS, &resLimits, 1);

    uint32_t startBlock[8] = { 0, 0, 0, block, 1, 1, 0, 0 };
    pm4SetShReg(packets, regCOMPUTE_START_X, startBlock, 8);

    pm4DispatchDirect(packets, grid, 1, 1, DISPATCH_INITIATOR_W32);
    pm4EventWrite(packets, CS_PARTIAL_FLUSH, EVENT_INDEX_CS_PARTIAL_FLUSH);
    pm4ReleaseMemFence(packets, cq.fenceGpu, fenceSeq);

    cqSubmitPackets(gpu, cq, packets);

    bool ok = cqWaitFence(cq, fenceSeq, 5000);
    cqHdpFlush(gpu, cq);

    uint32_t faultStatus = gcReg(gpu, cq, regGCVM_L2_PROTECTION_FAULT_STATUS, 0);
    uint32_t faultAddrLo = gcReg(gpu, cq, regGCVM_L2_PROTECTION_FAULT_ADDR_LO32, 0);
    uint32_t faultAddrHi = gcReg(gpu, cq, regGCVM_L2_PROTECTION_FAULT_ADDR_HI32, 0);
    uint64_t faultVa = (((uint64_t)faultAddrHi << 32) | faultAddrLo) << 12;

    cqGrbmSelect(gpu, cq, cq.me, cq.pipe, cq.queue, 0);
    uint32_t rptr = gcReg(gpu, cq, regCP_HQD_PQ_RPTR, 0);
    cqGrbmDeselect(gpu, cq);
    uint64_t fenceVal = *cq.fenceCpu;

    /* Verify ALL N output dwords == EXPECT (every thread's spilled sum). */
    const volatile uint32_t *res = (const volatile uint32_t *)outCpu;
    uint32_t nbad = 0, firstBad = 0, firstBadVal = 0;
    for (uint32_t i = 0; i < outN; i++) {
        uint32_t v = res[i];
        if (v != expect) {
            if (nbad == 0) { firstBad = i; firstBadVal = v; }
            nbad++;
        }
    }

    uint32_t walker = (faultStatus >> 1) & 0x7;
    uint32_t perm = (faultStatus >> 4) & 0xF;

    printf("\nFAULT_STATUS=0x%08X [walker=%u perm=0x%X] FAULT_VA=0x%llX\n",
           faultStatus, walker, perm, (unsigned long long)faultVa);
    printf("FENCE value=%llu (expected %llu) RPTR=0x%X\n",
           (unsigned long long)fenceVal, (unsigned long long)fenceSeq, rptr);
    uint32_t lastIdx = outN - 1;
    printf("out[0]=0x%08X out[%u]=0x%08X out[%u]=0x%08X expected=0x%08X "
           "bad=%u/%u\n",
           res[0], block, res[block], lastIdx, res[lastIdx], expect, nbad, outN);
    if (nbad)
        printf("  first mismatch at index %u (dword page %u): got 0x%08X\n",
               firstBad, (firstBad * 4) / 0x1000, firstBadVal);

    bool pass = ok && (faultStatus == 0) && (nbad == 0);
    printf("SCRATCH DISPATCH %s (grid=%u x block=%u, psfs=%u, %u scratch pages)\n",
           pass ? "PASS (fence signaled, no fault, all spilled output verified)"
           : (!ok ? "FAIL (fence timeout)"
              : (faultStatus ? "FAIL (GPUVM/CP fault)" : "FAIL (output mismatch)")),
           grid, block, psfs, scratchPages);
    return pass;
}

/* ======================================================================
 * ROCr lite:: integration surface (additive)
 *
 * Thin public wrappers over the proven static bring-up helpers so the ROCr
 * WindowsLiteDriver can run the SAME recipe wddm_lite_test uses, then route
 * its DirectQueuePlatform overrides through this WddmLite instance. No recipe
 * behaviour changes; these only re-expose existing static helpers.
 * ====================================================================== */

/* init_nbio doorbell-aperture + framebuffer enable, factored out of
 * recipeNopFence so both the recipe and the ROCr EnsureDoorbellAperture path
 * use one implementation. Mirrors the lite:: Linux transport's
 * EnsureDoorbellAperture register sequence (RCC_DOORBELL_APER_EN +
 * BIF_FB_EN). */
bool wddmEnsureDoorbellAperture(WddmLite &gpu, const IpDiscoveryResult &ipd)
{
    uint32_t nbifBase2 = 0;
    if (!cqResolveNbif(ipd, &nbifBase2)) {
        printf("  wddmEnsureDoorbellAperture: NBIF base[2] not found\n");
        return false;
    }
    uint32_t apEn = 0;
    gpu.readReg32((nbifBase2 + regRCC_DOORBELL_APER_EN) * 4, &apEn);
    gpu.writeReg32((nbifBase2 + regRCC_DOORBELL_APER_EN) * 4,
                   apEn | BIF_DOORBELL_APER_EN__BIT);
    /* GC doorbell self-ring S2A entries (GDC_S2A0_S2A_DOORBELL_ENTRY_0/3_CTRL).
     * Without these, a host doorbell write reaches the BAR aperture but is NOT
     * forwarded onto the on-die GC doorbell fabric, so the MES never sees the
     * KIQ doorbell. amdgpu programs these in nbif_v6_3_1_gc_doorbell_init and
     * the Linux lite:: transport does too. The direct MEC HQD works without
     * them (it advances via the MMIO wptr/poll), but the MES KIQ only watches
     * the doorbell -> required for MES KIQ servicing. */
    gpu.writeReg32((nbifBase2 + 0x01cb) * 4, (1u << 0) | (3u << 1) | (3u << 28));
    gpu.writeReg32((nbifBase2 + 0x01ce) * 4, (1u << 0) | (6u << 1) | (3u << 28));
    uint32_t fbEn = 0;
    gpu.readReg32((nbifBase2 + regBIF_FB_EN) * 4, &fbEn);
    gpu.writeReg32((nbifBase2 + regBIF_FB_EN) * 4,
                   fbEn | BIF_FB_EN__FB_READ_EN | BIF_FB_EN__FB_WRITE_EN);
    /* Doorbell self-ring GPA aperture -- the ONE doorbell-routing register
     * amdgpu programs (nbif_v6_3_1_enable_doorbell_selfring_aperture) that the
     * lite:: path omits. Without it, a doorbell write lands in the BAR but is
     * never routed back to the CP/MES doorbell monitor, so the MES never sees
     * the KIQ doorbell (the direct MEC HQD does not need it). BASE must be the
     * doorbell BAR address the GPU/NBIF sees -- under VM passthrough that is the
     * guest-physical BAR2 base (info.Bars[2].PhysicalAddress). CNTL = EN|MODE. */
    AMDGPU_ESCAPE_GET_INFO_DATA info = {};
    unsigned long long dbBase = 0;
    if (gpu.getInfo(&info)) dbBase = (unsigned long long)info.Bars[2].PhysicalAddress.QuadPart;
    gpu.writeReg32((nbifBase2 + 0x00f4) * 4, (uint32_t)(dbBase & 0xFFFFFFFFull));
    gpu.writeReg32((nbifBase2 + 0x00f3) * 4, (uint32_t)(dbBase >> 32));
    gpu.writeReg32((nbifBase2 + 0x00f5) * 4, 0x3u);
    uint32_t s2a0 = 0, s2a3 = 0, srC = 0, srL = 0, srH = 0;
    gpu.readReg32((nbifBase2 + 0x01cb) * 4, &s2a0);
    gpu.readReg32((nbifBase2 + 0x01ce) * 4, &s2a3);
    gpu.readReg32((nbifBase2 + 0x00f5) * 4, &srC);
    gpu.readReg32((nbifBase2 + 0x00f4) * 4, &srL);
    gpu.readReg32((nbifBase2 + 0x00f3) * 4, &srH);
    printf("  wddmEnsureDoorbellAperture: NBIF base[2]=0x%04X dbBAR=0x%llX "
           "S2A0=0x%08X S2A3=0x%08X SELFRING cntl=0x%08X base=0x%08X%08X\n",
           nbifBase2, dbBase, s2a0, s2a3, srC, srH, srL);
    return true;
}

bool wddmAllocVram(WddmLite &gpu, uint64_t size, void **cpu,
                   uint64_t *gpuAddr, uint64_t *handle)
{
    /* g_vramMcBase / g_vramCursor are seeded by recipeBootload (invoked from
     * wddmGfxBringUp), so callers must bring up the GPU first. */
    return rcpAllocVram(gpu, size, cpu, gpuAddr, handle);
}

bool wddmGfxBringUp(WddmLite &gpu, const IpDiscoveryResult &ipd,
                    const char *fwDir, uint64_t vramMcBase,
                    WddmComputeContext &ctx)
{
    memset(&ctx, 0, sizeof(ctx));
    ctx.gcBase0 = ipd.gcBase;
    ctx.gcBase1 = ipd.gcBase1;
    ctx.mmhubBase = ipd.mmhubBase;
    ctx.vramMcBase = vramMcBase;

    /* 1. PSP cold-boot autoload -> BOOTLOAD_COMPLETE. Also seeds the VRAM bump
     * allocator (g_vramMcBase = vramMcBase). */
    if (!recipeBootload(gpu, ipd, fwDir, vramMcBase)) {
        printf("  wddmGfxBringUp: recipeBootload did not reach "
               "BOOTLOAD_COMPLETE\n");
        return false;
    }

    /* 2. NBIO doorbell aperture + framebuffer enable. */
    ctx.hasNbif = cqResolveNbif(ipd, &ctx.nbifBase2);
    if (ctx.hasNbif)
        wddmEnsureDoorbellAperture(gpu, ipd);
    else
        printf("  wddmGfxBringUp: WARNING NBIF base[2] not found; doorbell "
               "aperture skipped\n");

    /* 3. init_gfx_for_compute (CP counters, RLC/SH_MEM/doorbell-range, MEC
     * enable), exactly as recipeNopFence. */
    CqState cq;
    memset(&cq, 0, sizeof(cq));
    cq.gcBase0 = ipd.gcBase;
    cq.gcBase1 = ipd.gcBase1;
    cq.nbifBase2 = ctx.nbifBase2;
    cq.hasNbif = ctx.hasNbif;

    const char *gc = "12_0_1";
    bool okP = false, okM = false, okC = false;
    uint64_t pfp = cqUcodeStart(fwDir, gc, "pfp", 52, &okP);
    uint64_t me = cqUcodeStart(fwDir, gc, "me", 52, &okM);
    uint64_t mec = cqUcodeStart(fwDir, gc, "mec", 52, &okC);
    bool haveUcode = okP && okM && okC;

    if (!cqInitGfxForCompute(gpu, cq, pfp, me, mec, haveUcode)) {
        printf("  wddmGfxBringUp: cqInitGfxForCompute failed\n");
        return false;
    }
    uint32_t mecCntl = gcReg(gpu, cq, regCP_MEC_RS64_CNTL, 1);
    ctx.mecEnabled = (mecCntl == 0x3C000000);
    printf("  wddmGfxBringUp: CP_MEC_RS64_CNTL=0x%08X (expected 0x3C000000) "
           "mec=%s\n", mecCntl, ctx.mecEnabled ? "ENABLED" : "NOT-ENABLED");
    return true;
}

/* ======================================================================
 * MES-on-Windows increment 2: IH (interrupt handler) ring setup.
 *
 * Faithful C++ port of python/.../windows/ih_init.py init_ih() (the proven IH
 * v7.0 ring setup for Navi 48 / gfx1201). The IH ring + WPTR-writeback dword
 * live in SYSTEM memory (allocDma -> PCIe bus address); the GPU writes 32-byte
 * interrupt entries there and mirrors the hardware WPTR into the writeback
 * dword (MC_SNOOP + WPTR_WRITEBACK_ENABLE). All IH MMIO uses
 *     byte_offset = (ipd.ihBase + reg_dword) * 4   (BAR0, like mmhubRead/pspRead)
 * ====================================================================== */

/* IH v7.0 register DWORD offsets relative to OSSSYS base (osssys_7_0_0_offset.h,
 * ih_init.py). */
#define regIH_RB_CNTL          0x0080
#define regIH_RB_RPTR          0x0081
#define regIH_RB_WPTR          0x0082
#define regIH_RB_BASE          0x0083
#define regIH_RB_BASE_HI       0x0084
#define regIH_RB_WPTR_ADDR_HI  0x0085
#define regIH_RB_WPTR_ADDR_LO  0x0086
#define regIH_DOORBELL_RPTR    0x0087
#define regIH_CNTL             0x00A8
#define regIH_INT_FLOOD_CNTL   0x00D5
#define regIH_MSI_STORM_CTRL   0x00F1

/* IH_RB_CNTL bit fields (ih_init.py). */
#define IH_RB_CNTL__RB_SIZE__SHIFT       1
#define IH_RB_CNTL__MC_SPACE__SHIFT      4
#define IH_RB_CNTL__ENABLE_INTR          (1u << 0)
#define IH_RB_CNTL__WPTR_OVERFLOW_ENABLE (1u << 8)
#define IH_RB_CNTL__WPTR_WRITEBACK_ENABLE (1u << 12)
#define IH_RB_CNTL__MC_SNOOP             (1u << 14)
#define IH_RB_CNTL__RPTR_REARM           (1u << 15)
#define IH_RB_CNTL__WPTR_OVERFLOW_CLEAR  (1u << 31)

#define IH_MC_SPACE_BUS_ADDR  2   /* PCIe bus address (system memory ring) */

/* NBIO interrupt-control registers (NBIF base_idx 2, nbio_init.py). */
#define regINTERRUPT_CNTL      0x00F1
#define regINTERRUPT_CNTL2     0x00F2
#define IH_DUMMY_RD_OVERRIDE   0x00000001
#define IH_REQ_NONSNOOP_EN     0x00000008

#define IH_RING_SIZE_DEFAULT   (256 * 1024)   /* 256KB, 8192 entries */

/* IH register access: byte_offset = (ihBase + reg) * 4, BAR0. */
static uint32_t ihReg(WddmLite &gpu, const WddmIhState &ih, uint32_t reg)
{
    uint32_t v = 0;
    gpu.readReg32((ih.ihBase + reg) * 4, &v);
    return v;
}

static void ihWreg(WddmLite &gpu, const WddmIhState &ih, uint32_t reg,
                   uint32_t val)
{
    gpu.writeReg32((ih.ihBase + reg) * 4, val);
}

bool wddmInitIh(WddmLite &gpu, const IpDiscoveryResult &ipd,
                const WddmComputeContext &ctx, WddmIhState &ih)
{
    printf("\n=== wddmInitIh (DIAGNOSTIC IH ring for MES KIQ servicing) ===\n");

    memset(&ih, 0, sizeof(ih));
    ih.ihBase = ipd.ihBase;
    ih.ringSize = IH_RING_SIZE_DEFAULT;

    if (ih.ihBase == 0) {
        printf("  wddmInitIh: IH/OSSSYS base is 0 (not enumerated); aborting\n");
        return false;
    }
    printf("  wddmInitIh: IH base = 0x%04X\n", ih.ihBase);

    /* Allocate the IH ring (system memory; GPU writes via PCIe bus addr). */
    if (!gpu.allocDma(ih.ringSize, &ih.ringCpu, &ih.ringBus, &ih.ringHandle)) {
        printf("  wddmInitIh: allocDma(ring %u KB) failed\n",
               ih.ringSize / 1024);
        return false;
    }
    memset(ih.ringCpu, 0, ih.ringSize);

    /* Allocate the WPTR-writeback dword (one page). */
    void *wptrCpu = nullptr;
    if (!gpu.allocDma(4096, &wptrCpu, &ih.wptrBus, &ih.wptrHandle)) {
        printf("  wddmInitIh: allocDma(wptr page) failed\n");
        return false;
    }
    ih.wptrCpu = wptrCpu;
    *(volatile uint32_t *)wptrCpu = 0;

    /* Dummy page for NBIO interrupt coalescing (setup_interrupt_control). */
    void *dummyCpu = nullptr; uint64_t dummyBus = 0; void *dummyHandle = nullptr;
    if (!gpu.allocDma(4096, &dummyCpu, &dummyBus, &dummyHandle)) {
        printf("  wddmInitIh: allocDma(dummy page) failed\n");
        return false;
    }

    printf("  wddmInitIh: ring bus=0x%012llX wptr bus=0x%012llX dummy bus="
           "0x%012llX\n",
           (unsigned long long)ih.ringBus, (unsigned long long)ih.wptrBus,
           (unsigned long long)dummyBus);

    /* Disable interrupts during setup (ih_v7_0_toggle_interrupts(false)). */
    {
        uint32_t v = ihReg(gpu, ih, regIH_RB_CNTL);
        v &= ~IH_RB_CNTL__ENABLE_INTR;
        ihWreg(gpu, ih, regIH_RB_CNTL, v);
    }

    /* NBIO interrupt control (setup_interrupt_control): dummy page + flags.
     * NBIF base_idx 2 (same base wddmEnsureDoorbellAperture uses). */
    if (ctx.hasNbif && ctx.nbifBase2 != 0) {
        gpu.writeReg32((ctx.nbifBase2 + regINTERRUPT_CNTL2) * 4,
                       (uint32_t)(dummyBus >> 8));
        uint32_t icntl = 0;
        gpu.readReg32((ctx.nbifBase2 + regINTERRUPT_CNTL) * 4, &icntl);
        icntl |= IH_DUMMY_RD_OVERRIDE | IH_REQ_NONSNOOP_EN;
        gpu.writeReg32((ctx.nbifBase2 + regINTERRUPT_CNTL) * 4, icntl);
        printf("  wddmInitIh: NBIO INTERRUPT_CNTL=0x%08X (dummy page set)\n",
               icntl);
    } else {
        printf("  wddmInitIh: WARNING no NBIF base; skipping NBIO interrupt "
               "control\n");
    }

    /* Program IH ring registers (_setup_ring). */
    ihWreg(gpu, ih, regIH_RB_BASE, (uint32_t)((ih.ringBus >> 8) & 0xFFFFFFFF));
    ihWreg(gpu, ih, regIH_RB_BASE_HI, (uint32_t)((ih.ringBus >> 40) & 0xFF));

    uint32_t sizeLog2 = cqLog2(ih.ringSize / 4);   /* log2(entries dwords) */
    uint32_t cntl = 0;
    cntl |= IH_MC_SPACE_BUS_ADDR << IH_RB_CNTL__MC_SPACE__SHIFT;
    cntl |= (sizeLog2 & 0x3F) << IH_RB_CNTL__RB_SIZE__SHIFT;
    cntl |= IH_RB_CNTL__WPTR_OVERFLOW_CLEAR;
    cntl |= IH_RB_CNTL__WPTR_OVERFLOW_ENABLE;
    cntl |= IH_RB_CNTL__WPTR_WRITEBACK_ENABLE;
    cntl |= IH_RB_CNTL__MC_SNOOP;
    cntl |= IH_RB_CNTL__RPTR_REARM;
    ihWreg(gpu, ih, regIH_RB_CNTL, cntl);
    ih.rbCntl = cntl;

    ihWreg(gpu, ih, regIH_RB_WPTR_ADDR_LO,
           (uint32_t)(ih.wptrBus & 0xFFFFFFFF));
    ihWreg(gpu, ih, regIH_RB_WPTR_ADDR_HI,
           (uint32_t)((ih.wptrBus >> 32) & 0xFFFF));

    ihWreg(gpu, ih, regIH_RB_RPTR, 0);
    ihWreg(gpu, ih, regIH_RB_WPTR, 0);

    /* No doorbell-driven RPTR: this ring is WPTR-writeback only. Clear any
     * stale IH_DOORBELL_RPTR routing (deliverable's "clear doorbell-rptr").
     * ih_init.py leaves IH_DOORBELL_RPTR untouched (it never enables doorbell
     * mode); writing 0 is a safe no-op-equivalent that disables it. */
    ihWreg(gpu, ih, regIH_DOORBELL_RPTR, 0);

    /* Flood control + MSI storm control (ih_init.py). */
    {
        uint32_t v = ihReg(gpu, ih, regIH_INT_FLOOD_CNTL);
        v |= (1u << 0);   /* FLOOD_CNTL_ENABLE */
        ihWreg(gpu, ih, regIH_INT_FLOOD_CNTL, v);
    }
    ihWreg(gpu, ih, regIH_MSI_STORM_CTRL, 3);

    /* Enable interrupts (ih_v7_0_toggle_interrupts(true)). */
    {
        uint32_t v = ihReg(gpu, ih, regIH_RB_CNTL);
        v |= IH_RB_CNTL__ENABLE_INTR;
        ihWreg(gpu, ih, regIH_RB_CNTL, v);
        ih.rbCntl = v;
    }

    uint32_t rbCntlRb = ihReg(gpu, ih, regIH_RB_CNTL);
    printf("  wddmInitIh: IH_RB_CNTL programmed=0x%08X readback=0x%08X "
           "(size_log2=%u; live amdgpu ref=0x40330121)\n",
           ih.rbCntl, rbCntlRb, sizeLog2);

    /* Register the ring with the kernel ISR (ENABLE_MSI). The driver treats
     * IhRingDmaHandle as the DmaAllocs[] index (== our allocDma handle) and
     * needs the IH RPTR/WPTR MMIO BYTE offsets so its ISR can drain the ring.
     * Not required for the on-GPU probe: even if this fails, the GPU still
     * writes IH_RB_WPTR + the writeback dword, which dumpMesDiag reads. */
    uint32_t rptrByteOff = (ih.ihBase + regIH_RB_RPTR) * 4;
    uint32_t wptrByteOff = (ih.ihBase + regIH_RB_WPTR) * 4;
    bool enabled = false; uint32_t vectors = 0;
    if (gpu.enableMsi(ih.ringHandle, ih.ringSize, rptrByteOff, wptrByteOff,
                      &enabled, &vectors)) {
        ih.msiEnabled = enabled;
        ih.numVectors = vectors;
        printf("  wddmInitIh: ENABLE_MSI ok enabled=%d vectors=%u\n",
               enabled ? 1 : 0, vectors);
    } else {
        printf("  wddmInitIh: ENABLE_MSI escape FAILED (continuing; the GPU "
               "still updates IH_RB_WPTR for the on-GPU probe)\n");
    }

    ih.configured = true;
    printf("  wddmInitIh: IH ring live (size=%u KB)\n", ih.ringSize / 1024);
    return true;
}

/* ======================================================================
 * DIAGNOSTIC (MES-on-Windows): start the MES engine.
 *
 * recipeBootload loads the MES firmware (CP_MES 33 / MES_STACK 34 /
 * CP_MES_KIQ 81 / MES_KIQ_STACK 82) as part of the PSP autoload batch, but the
 * direct-HQD bring-up (cqInitGfxForCompute) stops at the MEC enable and never
 * touches CP_MES_CNTL -- so on the proven path the MES engine is LOADED but NOT
 * RUNNING. This helper releases it, faithfully transcribing
 * ring_init.py::_enable_mes_from_ucode (the same sequence Linux lite:: used to
 * reach CP_MES_CNTL=0x0C000000 with CP_MES_HEADER_DUMP ticking).
 *
 * Because the PSP already loaded the uni_mes ucode (mes_psp_loaded), the
 * IC_BASE/MDBASE VRAM-backdoor staging (_program_mes_ucode_buffers) is skipped,
 * exactly as the Python path does when mes_psp_loaded is set. We still must
 * program CP_MES_PRGRM_CNTR_START with the MES entry point (from the uni_mes fw
 * header) so the released pipe fetches from the right PC.
 * ====================================================================== */
#define regCP_MES_CNTL                 0x2807   /* base_idx 1 */
#define regCP_MES_PRGRM_CNTR_START     0x2800   /* base_idx 1 */
#define regCP_MES_PRGRM_CNTR_START_HI  0x289D   /* base_idx 1 */
#define regCP_MES_HEADER_DUMP          0x280D   /* base_idx 1 */
#define regCP_MES_INSTR_PNTR           0x2813   /* base_idx 1 */
#define regCP_MES_GP3_LO               0x2849   /* base_idx 1 (version) */
#define regRLC_CP_SCHEDULERS           0x098A   /* base_idx 1 */

#define CP_MES_CNTL__MES_INVALIDATE_ICACHE  (1u << 4)
#define CP_MES_CNTL__MES_PIPE0_RESET        (1u << 16)
#define CP_MES_CNTL__MES_PIPE1_RESET        (1u << 17)
#define CP_MES_CNTL__MES_PIPE0_ACTIVE       (1u << 26)
#define CP_MES_CNTL__MES_PIPE1_ACTIVE       (1u << 27)
#define CP_MES_CNTL__MES_HALT               (1u << 30)

#define MES_ME_INDEX   3   /* MEID for MES queues */
#define MES_PIPE_KIQ   1

bool wddmStartMes(WddmLite &gpu, const IpDiscoveryResult &ipd,
                  const char *fwDir, const WddmComputeContext &ctx)
{
    printf("\n=== wddmStartMes (DIAGNOSTIC MES engine release) ===\n");

    CqState cq;
    memset(&cq, 0, sizeof(cq));
    cq.gcBase0 = ctx.gcBase0;
    cq.gcBase1 = ctx.gcBase1;
    cq.nbifBase2 = ctx.nbifBase2;
    cq.hasNbif = ctx.hasNbif;

    /* MES entry point: gc_<gc>_uni_mes.bin, mes_firmware_header_v1_0
     * ucode_start_addr_lo/hi at +56/+60 (matches bringup.py _mes_entry()). */
    const char *gc = "12_0_1";
    bool okMes = false;
    uint64_t mesEntry = cqUcodeStart(fwDir, gc, "uni_mes", 56, &okMes);
    if (!okMes) {
        printf("  wddmStartMes: cannot read MES entry from gc_%s_uni_mes.bin\n",
               gc);
        return false;
    }
    printf("  wddmStartMes: MES entry=0x%016llX (PC=0x%llX)\n",
           (unsigned long long)mesEntry, (unsigned long long)(mesEntry >> 2));

    /* CP_MES_CNTL pre-state (after bootload, before release). */
    uint32_t mesCntlPre = gcReg(gpu, cq, regCP_MES_CNTL, 1);
    uint32_t version = gcReg(gpu, cq, regCP_MES_GP3_LO, 1);
    printf("  wddmStartMes: CP_MES_CNTL(pre)=0x%08X CP_MES_GP3_LO=0x%08X\n",
           mesCntlPre, version);

    /* 1. RLC_CP_SCHEDULERS: route the KIQ (me=3 pipe=KIQ hqd=0) + enable bit. */
    uint32_t schedulers = gcReg(gpu, cq, regRLC_CP_SCHEDULERS, 1);
    schedulers &= 0xFFFFFF00u;
    schedulers |= (MES_ME_INDEX << 5) | (MES_PIPE_KIQ << 3) | 0u | 0x80u;
    gcWreg(gpu, cq, regRLC_CP_SCHEDULERS, schedulers, 1);

    /* 2. CP_MES_CNTL: clear ACTIVE, set INVALIDATE_ICACHE + PIPE0/1_RESET +
     *    HALT (reset+halt both pipes before programming the PC). */
    uint32_t val = gcReg(gpu, cq, regCP_MES_CNTL, 1);
    val &= ~(CP_MES_CNTL__MES_PIPE0_ACTIVE | CP_MES_CNTL__MES_PIPE1_ACTIVE);
    val |= (CP_MES_CNTL__MES_INVALIDATE_ICACHE |
            CP_MES_CNTL__MES_PIPE0_RESET |
            CP_MES_CNTL__MES_PIPE1_RESET |
            CP_MES_CNTL__MES_HALT);
    gcWreg(gpu, cq, regCP_MES_CNTL, val, 1);

    /* 3. Per-pipe program counter (both MES pipe0 and pipe1 share the uni_mes
     *    entry, mirroring bringup.py MES/MES1 = mes_entry). */
    uint32_t activeMask = 0;
    for (uint32_t pipe = 0; pipe < 2; pipe++) {
        cqGrbmSelect(gpu, cq, MES_ME_INDEX, pipe, 0, 0);
        gcWregPair(gpu, cq, regCP_MES_PRGRM_CNTR_START,
                   regCP_MES_PRGRM_CNTR_START_HI, mesEntry >> 2, 1);
        activeMask |= (pipe == 0) ? CP_MES_CNTL__MES_PIPE0_ACTIVE
                                  : CP_MES_CNTL__MES_PIPE1_ACTIVE;
    }
    cqGrbmDeselect(gpu, cq);

    /* 4. CP_MES_CNTL release: clear reset/halt/icache + set PIPE0/1_ACTIVE. */
    val = gcReg(gpu, cq, regCP_MES_CNTL, 1);
    val &= ~(CP_MES_CNTL__MES_INVALIDATE_ICACHE |
             CP_MES_CNTL__MES_PIPE0_RESET |
             CP_MES_CNTL__MES_PIPE1_RESET |
             CP_MES_CNTL__MES_HALT |
             CP_MES_CNTL__MES_PIPE0_ACTIVE |
             CP_MES_CNTL__MES_PIPE1_ACTIVE);
    val |= activeMask;
    gcWreg(gpu, cq, regCP_MES_CNTL, val, 1);
    Sleep(1);

    /* Liveness check: HEADER_DUMP / INSTR_PNTR should change if MES executes. */
    uint32_t hdr0 = gcReg(gpu, cq, regCP_MES_HEADER_DUMP, 1);
    uint32_t ip0  = gcReg(gpu, cq, regCP_MES_INSTR_PNTR, 1);
    Sleep(200);
    uint32_t hdr1 = gcReg(gpu, cq, regCP_MES_HEADER_DUMP, 1);
    uint32_t ip1  = gcReg(gpu, cq, regCP_MES_INSTR_PNTR, 1);
    uint32_t mesCntlPost = gcReg(gpu, cq, regCP_MES_CNTL, 1);

    bool pipesActive =
        (mesCntlPost & (CP_MES_CNTL__MES_PIPE0_ACTIVE |
                        CP_MES_CNTL__MES_PIPE1_ACTIVE)) ==
        (CP_MES_CNTL__MES_PIPE0_ACTIVE | CP_MES_CNTL__MES_PIPE1_ACTIVE);
    bool mesAlive = (hdr0 != hdr1) || (ip1 != 0);

    printf("  wddmStartMes: CP_MES_CNTL(post)=0x%08X (PIPE0/1_ACTIVE=%s)\n",
           mesCntlPost, pipesActive ? "set" : "NOT-set");
    printf("  wddmStartMes: HEADER_DUMP 0x%08X->0x%08X INSTR_PNTR "
           "0x%08X->0x%08X  MES %s\n",
           hdr0, hdr1, ip0, ip1,
           mesAlive ? "RUNNING" : "NOT visibly executing");
    return pipesActive;
}
