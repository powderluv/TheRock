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

/* Allocate a PSP-visible VRAM buffer (mirrors _alloc_psp_buffer alloc_memory
 * branch). Returns CPU mapping + raw VRAM MC GPU address. */
static bool rcpAllocVram(WddmLite &gpu, uint64_t size, void **cpu,
                         uint64_t *gpuAddr, uint64_t *handle)
{
    uint32_t flags = AMDGPU_MEM_TYPE_VRAM | AMDGPU_MEM_FLAG_HOST_ACCESS;
    if (!gpu.allocMemory(size, flags, cpu, gpuAddr, handle)) {
        printf("  PSP[recipe]: ERROR VRAM alloc failed (size=0x%llX)\n",
               (unsigned long long)size);
        return false;
    }
    if (*cpu == nullptr) {
        printf("  PSP[recipe]: ERROR VRAM alloc not CPU mapped\n");
        return false;
    }
    memset(*cpu, 0, (size_t)size);
    return true;
}

/* Allocate a PSP-visible VRAM buffer aligned to `alignment` (mirrors
 * _alloc_psp_buffer with an explicit alignment: over-allocate then advance
 * the CPU pointer + MC address by the same delta). Used for the bootloader
 * fw staging buffer, which is written to C2PMSG_36 as (addr >> 20) and so
 * must be 1MB-aligned. */
static bool rcpAllocVramAligned(WddmLite &gpu, uint64_t size, uint64_t alignment,
                                void **cpu, uint64_t *gpuAddr, uint64_t *handle)
{
    uint64_t allocSize = size + alignment;
    void *rawCpu = nullptr;
    uint64_t rawGpu = 0, rawHandle = 0;
    if (!rcpAllocVram(gpu, allocSize, &rawCpu, &rawGpu, &rawHandle))
        return false;
    uint64_t delta = ((~rawGpu) + 1) & (alignment - 1);   /* (-rawGpu) & (align-1) */
    *cpu = (void *)((uint8_t *)rawCpu + delta);
    *gpuAddr = rawGpu + delta;
    *handle = rawHandle;
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
