/*
 * ddi_query.c - QueryAdapterInfo and engine query DDIs
 *
 * The critical DDI here is QueryAdapterInfo with DXGKQAITYPE_DRIVERCAPS,
 * which tells dxgkrnl we are a compute-only (MCDM) device.
 */

#include "amdgpu_mcdm.h"

/* Record a QueryAdapterInfo type we rejected (diagnostic bitmask). */
static void
RecordUnhandledQai(
    AMDGPU_ADAPTER *pAdapter,
    ULONG           Type
    )
{
    AmdGpuDiag(L"QAI_Unhandled", Type);
    if (pAdapter != NULL && Type < 64) {
        pAdapter->QaiUnhandledMask |= (1LL << Type);
        AmdGpuDiag(L"QAI_UnhandledLo", (ULONG)(pAdapter->QaiUnhandledMask & 0xFFFFFFFF));
        AmdGpuDiag(L"QAI_UnhandledHi", (ULONG)((pAdapter->QaiUnhandledMask >> 32) & 0xFFFFFFFF));
    }
}

/* ======================================================================
 * QueryAdapterInfo — report driver capabilities
 *
 * dxgkrnl queries various adapter properties. The most important are:
 * - DXGKQAITYPE_DRIVERCAPS: declares ComputeOnly=TRUE
 * - DXGKQAITYPE_QUERYSEGMENT4: memory segment layout
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuQueryAdapterInfo(
    IN_CONST_HANDLE                         hAdapter,
    IN_CONST_PDXGKARG_QUERYADAPTERINFO      pQueryAdapterInfo
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)hAdapter;

    if (pQueryAdapterInfo == NULL || pQueryAdapterInfo->pOutputData == NULL)
        return STATUS_INVALID_PARAMETER;

    AmdGpuDiag(L"LastDDI", AMDGPU_DDI_QUERYADAPTERINFO);
    AmdGpuDiag(L"QAI_LastType", (ULONG)pQueryAdapterInfo->Type);
    if ((ULONG)pQueryAdapterInfo->Type < 64) {
        pAdapter->QaiQueriedMask |= (1LL << (ULONG)pQueryAdapterInfo->Type);
        AmdGpuDiag(L"QAI_QueriedLo", (ULONG)(pAdapter->QaiQueriedMask & 0xFFFFFFFF));
        AmdGpuDiag(L"QAI_QueriedHi", (ULONG)((pAdapter->QaiQueriedMask >> 32) & 0xFFFFFFFF));
    }

    switch (pQueryAdapterInfo->Type) {

    case DXGKQAITYPE_DRIVERCAPS:
    {
        DXGK_DRIVERCAPS *pCaps;
        ULONG capsSize = pQueryAdapterInfo->OutputDataSize;

        /*
         * dxgkrnl sizes this buffer for the WDDM version we declared
         * (DXGKDDI_INTERFACE_VERSION_WDDM2_6). These headers default to
         * WDDM3_2, so sizeof(DXGK_DRIVERCAPS) is larger than dxgkrnl's
         * buffer; a "< sizeof" check wrongly returns BUFFER_TOO_SMALL and
         * aborts adapter start (dxgkrnl: StartAdapter_AddAdapterFailed).
         * Accept dxgkrnl's size and zero only that — every field we set
         * lives in the <=2.6 prefix of the struct, so it fits.
         */
        if (capsSize > sizeof(DXGK_DRIVERCAPS))
            capsSize = sizeof(DXGK_DRIVERCAPS);

        pCaps = (DXGK_DRIVERCAPS *)pQueryAdapterInfo->pOutputData;
        RtlZeroMemory(pCaps, capsSize);

        /* Accept DMA from any physical address */
        pCaps->HighestAcceptableAddress.QuadPart = (LONGLONG)-1;

        /* GPU engine topology: one compute node */
        pCaps->GpuEngineTopology.NbAsymetricProcessingNodes = 1;

        /* Scheduling caps: WDDMv2 drivers MUST be MultiEngineAware — dxgkrnl
         * rejects adapter start with STATUS_INVALID_PARAMETER otherwise
         * (confirmed via the DxgKrnl trace message "SchedulingCaps.
         * MultiEngineAware is not set by WDDMv2 driver"). */
        pCaps->SchedulingCaps.MultiEngineAware = 1;

        /* Memory management caps. WDDM2.0+ removed the legacy/physical
         * addressing model, so VirtualAddressingSupported MUST be set
         * alongside the chosen MMU model — IoMmuSupported=1 alone (Value
         * 0x80) is an incomplete model dxgkrnl rejects at adapter start.
         * IoMmu-only (no GpuMmu), matching our DART/IOMMU design; Value=0xA0. */
        pCaps->MemoryManagementCaps.VirtualAddressingSupported = 1;
        pCaps->MemoryManagementCaps.IoMmuSupported = 1;

        /* WDDM version: 2.6 for ComputeOnly support */
        pCaps->WDDMVersion = DXGKDDI_WDDMv2_6;

        /* Per-engine TDR support */
        pCaps->SupportPerEngineTDR = TRUE;

        /* The key bit: this is a compute-only (MCDM) adapter */
        pCaps->MiscCaps.ComputeOnly = 1;

        return STATUS_SUCCESS;
    }

    case DXGKQAITYPE_QUERYSEGMENT4:
    {
        DXGK_QUERYSEGMENTOUT4 *pSegmentOut;

        if (pQueryAdapterInfo->OutputDataSize < sizeof(DXGK_QUERYSEGMENTOUT4))
            return STATUS_BUFFER_TOO_SMALL;

        pSegmentOut = (DXGK_QUERYSEGMENTOUT4 *)pQueryAdapterInfo->pOutputData;

        if (pSegmentOut->pSegmentDescriptor == NULL) {
            /* First call: return count of segments */
            pSegmentOut->NbSegment = 2;  /* VRAM + System */
            pSegmentOut->SegmentDescriptorStride = sizeof(DXGK_SEGMENTDESCRIPTOR4);
            return STATUS_SUCCESS;
        }

        /* Second call: fill in segment descriptors using stride-based indexing */
        if (pSegmentOut->NbSegment >= 1) {
            DXGK_SEGMENTDESCRIPTOR4 *pVram =
                (DXGK_SEGMENTDESCRIPTOR4 *)(pSegmentOut->pSegmentDescriptor);
            RtlZeroMemory(pVram, sizeof(*pVram));

            /* Segment 1: VRAM */
            pVram->Flags.Aperture = 0;
            pVram->Flags.CpuVisible = 1;           /* BAR-mapped VRAM */
            pVram->BaseAddress.QuadPart = 0;
            pVram->Size = pAdapter->VramSize;
            if (pVram->Size == 0)
                pVram->Size = 256 * 1024 * 1024;   /* Default 256MB until real detection */
            pVram->CpuTranslatedAddress.QuadPart = 0; /* Filled in v0.2+ */
            pVram->CommitLimit = pVram->Size;
        }

        if (pSegmentOut->NbSegment >= 2) {
            DXGK_SEGMENTDESCRIPTOR4 *pSys =
                (DXGK_SEGMENTDESCRIPTOR4 *)(pSegmentOut->pSegmentDescriptor +
                    pSegmentOut->SegmentDescriptorStride);
            RtlZeroMemory(pSys, sizeof(*pSys));

            /* Segment 2: System memory (aperture) */
            pSys->Flags.Aperture = 1;
            pSys->Flags.CpuVisible = 1;
            pSys->Size = 1ULL * 1024 * 1024 * 1024;  /* 1GB aperture */
            pSys->CommitLimit = pSys->Size;
        }

        pSegmentOut->PagingBufferSegmentId = 0;     /* System memory */
        pSegmentOut->PagingBufferSize = 4096;        /* Minimal */

        return STATUS_SUCCESS;
    }

    case DXGKQAITYPE_PHYSICALADAPTERCAPS:
    {
        /*
         * Required during adapter start. Like DRIVERCAPS this struct grew
         * across versions (VirtualCopyNodeIndex @ WDDM2_7), so accept
         * dxgkrnl's (smaller, version-appropriate) buffer and fill the
         * leading fields. We expose a single compute execution node.
         */
        DXGK_PHYSICALADAPTERCAPS *pPhys;
        ULONG physSize = pQueryAdapterInfo->OutputDataSize;

        if (physSize < FIELD_OFFSET(DXGK_PHYSICALADAPTERCAPS, DxgkPhysicalAdapterHandle))
            return STATUS_INVALID_PARAMETER;
        if (physSize > sizeof(DXGK_PHYSICALADAPTERCAPS))
            physSize = sizeof(DXGK_PHYSICALADAPTERCAPS);

        pPhys = (DXGK_PHYSICALADAPTERCAPS *)pQueryAdapterInfo->pOutputData;
        RtlZeroMemory(pPhys, physSize);
        pPhys->NumExecutionNodes = 1;   /* one compute engine node */
        pPhys->PagingNodeIndex = 0;
        /* Driver-invented opaque handle identifying this physical adapter
         * (a non-null, unique value; the miniport adapter context). */
        pPhys->DxgkPhysicalAdapterHandle = (HANDLE)hAdapter;
        /* Declare the per-physical-adapter MMU model. This MUST match
         * DRIVERCAPS.MemoryManagementCaps (IoMmu). A zeroed Flags advertises
         * NO addressing model on the adapter, which dxgkrnl rejects with
         * STATUS_INVALID_PARAMETER at adapter start (this is the last cap
         * queried before the rejection). */
        if (physSize > FIELD_OFFSET(DXGK_PHYSICALADAPTERCAPS, Flags))
            pPhys->Flags.IoMmuSupported = 1;
        return STATUS_SUCCESS;
    }

    case DXGKQAITYPE_GPUVERSION:
    case DXGKQAITYPE_ADAPTERPERFDATA:
    case DXGKQAITYPE_ADAPTERPERFDATA_CAPS:
    {
        /*
         * Version / perf-telemetry info caps. Returning NOT_SUPPORTED makes
         * dxgmms2's VIDMM_GLOBAL::ReadPhysicalAdapterConfiguration fall into a
         * fallback path that walks an uninitialized UNICODE_STRING (garbage
         * Length) → RtlAppendUnicodeStringToString → memcpy GP fault (0x7E)
         * during VidMm init. Provide a zeroed, valid response so any embedded
         * strings are empty (Length=0) and dxgmms2 uses our data.
         */
        if (pQueryAdapterInfo->OutputDataSize == 0)
            return STATUS_INVALID_PARAMETER;
        RtlZeroMemory(pQueryAdapterInfo->pOutputData, pQueryAdapterInfo->OutputDataSize);
        return STATUS_SUCCESS;
    }

#if (DXGKDDI_INTERFACE_VERSION >= DXGKDDI_INTERFACE_VERSION_WDDM3_2)
    case DXGKQAITYPE_64BITONLYCAPS:
    {
        /*
         * Newer (WDDM3.x) capability dxgkrnl probes speculatively. We
         * have no special 64-bit-only requirements; answer with a zeroed
         * buffer + SUCCESS rather than NOT_SUPPORTED. (Only compiled when
         * the DDI version is high enough to define this enumerant — at our
         * pinned WDDM2_6 it does not exist and dxgkrnl never queries it.)
         */
        if (pQueryAdapterInfo->OutputDataSize == 0)
            return STATUS_INVALID_PARAMETER;
        RtlZeroMemory(pQueryAdapterInfo->pOutputData, pQueryAdapterInfo->OutputDataSize);
        return STATUS_SUCCESS;
    }
#endif

    case DXGKQAITYPE_WDDMDEVICECAPS:
    {
        /*
         * Mandatory at WDDM 2.6: queried during device init (after
         * AddDevice). Rejecting it aborts adapter bring-up (Code 43)
         * before dxgkrnl ever queries memory segments. The only field
         * is WDDMVersion, which must match DXGK_DRIVERCAPS::WDDMVersion.
         */
        DXGK_WDDMDEVICECAPS *pDevCaps;

        if (pQueryAdapterInfo->OutputDataSize < sizeof(DXGK_WDDMDEVICECAPS))
            return STATUS_BUFFER_TOO_SMALL;

        pDevCaps = (DXGK_WDDMDEVICECAPS *)pQueryAdapterInfo->pOutputData;
        RtlZeroMemory(pDevCaps, sizeof(*pDevCaps));
        pDevCaps->WDDMVersion = DXGKDDI_WDDMv2_6;
        return STATUS_SUCCESS;
    }

    case DXGKQAITYPE_UMDRIVERPRIVATE:
        /* No UMD private data */
        RecordUnhandledQai(pAdapter, (ULONG)pQueryAdapterInfo->Type);
        return STATUS_NOT_SUPPORTED;

    default:
        /* Record the type dxgkrnl asked for that we don't satisfy — this
         * is the prime suspect for a silent post-start (Code 43) failure
         * (e.g. a memory-segment query variant we don't implement). */
        RecordUnhandledQai(pAdapter, (ULONG)pQueryAdapterInfo->Type);
        return STATUS_NOT_SUPPORTED;
    }
}

/* ======================================================================
 * GetNodeMetadata — describe GPU engines
 *
 * We report one compute-only engine node.
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuGetNodeMetadata(
    IN_CONST_HANDLE                     hAdapter,
    UINT                                NodeOrdinalAndAdapterIndex,
    OUT_PDXGKARG_GETNODEMETADATA        pGetNodeMetadata
    )
{
    UINT NodeOrdinal = NodeOrdinalAndAdapterIndex & 0xFFFF;

    UNREFERENCED_PARAMETER(hAdapter);

    if (pGetNodeMetadata == NULL)
        return STATUS_INVALID_PARAMETER;

    RtlZeroMemory(pGetNodeMetadata, sizeof(*pGetNodeMetadata));

    if (NodeOrdinal == 0) {
        pGetNodeMetadata->EngineType = DXGK_ENGINE_TYPE_3D;
        /* For compute-only, 3D engine type is used by convention */
        pGetNodeMetadata->FriendlyName[0] = L'C';
        pGetNodeMetadata->FriendlyName[1] = L'o';
        pGetNodeMetadata->FriendlyName[2] = L'm';
        pGetNodeMetadata->FriendlyName[3] = L'p';
        pGetNodeMetadata->FriendlyName[4] = L'u';
        pGetNodeMetadata->FriendlyName[5] = L't';
        pGetNodeMetadata->FriendlyName[6] = L'e';
        pGetNodeMetadata->FriendlyName[7] = L'\0';
        /* IoMmu-only model — must match DRIVERCAPS / PHYSICALADAPTERCAPS
         * (advertising both here while the adapter declares only IoMmu is
         * inconsistent and would fail after AddAdapter). */
        pGetNodeMetadata->GpuMmuSupported = FALSE;
        pGetNodeMetadata->IoMmuSupported = TRUE;
    } else {
        return STATUS_INVALID_PARAMETER;
    }

    return STATUS_SUCCESS;
}

/* ======================================================================
 * Engine status queries
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuQueryDependentEngineGroup(
    IN_CONST_HANDLE                                 hAdapter,
    INOUT_DXGKARG_QUERYDEPENDENTENGINEGROUP         pQueryDependentEngineGroup
    )
{
    UNREFERENCED_PARAMETER(hAdapter);

    if (pQueryDependentEngineGroup == NULL)
        return STATUS_INVALID_PARAMETER;

    /* Single engine — depends only on itself */
    pQueryDependentEngineGroup->DependentNodeOrdinalMask = 1;

    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuQueryEngineStatus(
    IN_CONST_HANDLE                         hAdapter,
    INOUT_PDXGKARG_QUERYENGINESTATUS        pQueryEngineStatus
    )
{
    UNREFERENCED_PARAMETER(hAdapter);

    if (pQueryEngineStatus == NULL)
        return STATUS_INVALID_PARAMETER;

    /* Report engine as active and not hung */
    pQueryEngineStatus->EngineStatus.Responsive = 1;

    return STATUS_SUCCESS;
}
