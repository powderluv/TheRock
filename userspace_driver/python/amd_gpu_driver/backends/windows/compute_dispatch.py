"""Compute dispatch and GPU bring-up orchestration for Windows.

Ties together all initialization modules to bring up the GPU from cold
state to running compute workloads:

Init sequence:
1. Open device via D3DKMTEscape
2. IP discovery - enumerate IP blocks and base addresses
3. NBIO init - doorbell aperture, framebuffer access
4. GMC init - memory controller, system aperture, GART
5. PSP init - firmware loading (SOS, RLC, MEC, SDMA)
6. IH init - interrupt handler ring
7. Compute ring init - MQD, HQD registers, doorbell
8. Self-test - NOP + RELEASE_MEM fence verification

After bring-up, provides:
- PM4 WRITE_DATA memory test (no shader needed)
- Compute kernel dispatch from ELF code objects (.co files)
- Inline noop shader dispatch (validates the compute pipeline)

Usage:
    ctx = full_gpu_bringup()
    test_write_data(ctx)
    # dispatch_elf_kernel(ctx, "kernel.co", grid=(4,1,1), block=(64,1,1), args=[buf])
    shutdown(ctx)

Reference: Linux amdgpu driver initialization path
"""

from __future__ import annotations

import ctypes
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from amd_gpu_driver.backends.windows.device import WindowsDevice
from amd_gpu_driver.backends.windows.ip_discovery import (
    IPDiscoveryResult,
    parse_ip_discovery,
    read_discovery_table_via_mmio,
)
from amd_gpu_driver.backends.windows.nbio_init import NBIOConfig, init_nbio
from amd_gpu_driver.backends.windows.gmc_init import GMCConfig, init_gmc
from amd_gpu_driver.backends.windows.psp_init import (
    PSPConfig,
    init_psp,
    load_all_firmware,
)
from amd_gpu_driver.backends.windows.ih_init import IHConfig, init_ih
from amd_gpu_driver.backends.windows.ring_init import (
    ComputeQueueConfig,
    init_compute_queue,
    submit_compute_packets,
    read_fence_value,
    wait_fence,
    test_compute_nop_fence,
)
from amd_gpu_driver.commands.pm4 import (
    PM4PacketBuilder,
    SH_REG_BASE,
)
from amd_gpu_driver.gpu.registers import (
    COMPUTE_PGM_LO,
    COMPUTE_PGM_RSRC1,
    COMPUTE_PGM_RSRC2,
    COMPUTE_PGM_RSRC3,
    COMPUTE_RESOURCE_LIMITS,
    COMPUTE_RESTART_X,
    COMPUTE_START_X,
    COMPUTE_TMPRING_SIZE,
    COMPUTE_USER_DATA_0,
    sh_reg_offset,
)


# ============================================================================
# PM4 opcodes not in the existing builder
# ============================================================================

PACKET3_WRITE_DATA = 0x37

# WRITE_DATA dst_sel values
WRITE_DATA_DST_SEL_MEM_MAPPED = 1
WRITE_DATA_DST_SEL_MEM_ASYNC = 5
WRITE_DATA_WR_CONFIRM = 1 << 20
WRITE_DATA_ENGINE_SEL_ME = 0


# ============================================================================
# GPUContext - holds all initialized subsystem configs
# ============================================================================

@dataclass
class GPUContext:
    """Holds all configuration state for an initialized GPU."""
    dev: WindowsDevice
    ip_result: IPDiscoveryResult
    nbio_config: NBIOConfig
    gmc_config: GMCConfig
    psp_config: PSPConfig
    ih_config: IHConfig
    compute_queue: ComputeQueueConfig

    # Additional DMA allocations used during bring-up
    gart_table_dma_handle: int = 0
    dummy_page_dma_handle: int = 0
    _fence_seq: int = 0

    def next_fence_seq(self) -> int:
        """Return the next fence sequence number."""
        self._fence_seq += 1
        return self._fence_seq


# ============================================================================
# PM4 packet helpers
# ============================================================================

def _build_write_data_packet(
    addr: int,
    values: list[int],
) -> bytes:
    """Build a PM4 WRITE_DATA packet to write DWORDs to a memory address.

    WRITE_DATA (opcode 0x37):
      DW0: header
      DW1: [11:8]=DST_SEL(5=mem_async), [20]=WR_CONFIRM, [31:30]=ENGINE_SEL
      DW2: addr_lo (DWORD-aligned)
      DW3: addr_hi
      DW4+: data values

    Reference: si_pm4.h, PACKET3_WRITE_DATA
    """
    n_body = 3 + len(values)  # control + addr_lo + addr_hi + data
    header = (3 << 30) | (((n_body - 1) & 0x3FFF) << 16) | (PACKET3_WRITE_DATA << 8)

    dw1 = (WRITE_DATA_DST_SEL_MEM_ASYNC << 8) | WRITE_DATA_WR_CONFIRM
    dwords = [header, dw1, addr & 0xFFFFFFFF, (addr >> 32) & 0xFFFFFFFF]
    dwords.extend(v & 0xFFFFFFFF for v in values)

    return struct.pack(f"<{len(dwords)}I", *dwords)


def _build_dispatch_packets(
    code_gpu_addr: int,
    kernarg_gpu_addr: int,
    pgm_rsrc1: int,
    pgm_rsrc2: int,
    pgm_rsrc3: int,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    fence_addr: int,
    fence_value: int,
    kernarg_sgpr_index: int = 0,
) -> bytes:
    """Build a complete PM4 command stream for compute dispatch.

    ``kernarg_sgpr_index`` is the USER_DATA SGPR slot that receives the kernarg
    segment pointer (0 when kernarg is the first enabled preload SGPR, as for a
    minimal kernel).

    Sequence:
    1. ACQUIRE_MEM - invalidate caches
    2. SET_SH_REG - program address (COMPUTE_PGM_LO/HI)
    3. SET_SH_REG - program resources (RSRC1, RSRC2)
    4. SET_SH_REG - RSRC3
    5. SET_SH_REG - scratch (TMPRING_SIZE = 0)
    6. SET_SH_REG - restart coordinates
    7. SET_SH_REG - kernarg pointer (USER_DATA_0/1)
    8. SET_SH_REG - resource limits
    9. SET_SH_REG - start coordinates + workgroup dimensions
    10. DISPATCH_DIRECT
    11. RELEASE_MEM - write fence value

    Args:
        code_gpu_addr: GPU address of the kernel code entry point.
        kernarg_gpu_addr: GPU address of kernel arguments.
        pgm_rsrc1: COMPUTE_PGM_RSRC1 register value.
        pgm_rsrc2: COMPUTE_PGM_RSRC2 register value.
        pgm_rsrc3: COMPUTE_PGM_RSRC3 register value.
        grid: Dispatch dimensions (workgroups in x, y, z).
        block: Workgroup dimensions (threads in x, y, z).
        fence_addr: GPU address for completion fence write.
        fence_value: Value to write on completion.

    Returns:
        Serialized PM4 packet bytes.
    """
    from amd_gpu_driver.commands.pm4 import (
        CP_COHER_CNTL_SH_KCACHE_ACTION,
        CP_COHER_CNTL_TCL1_ACTION,
        CS_PARTIAL_FLUSH,
        EVENT_INDEX_CS_PARTIAL_FLUSH,
    )

    pm4 = PM4PacketBuilder()

    # 1. Cache invalidation
    pm4.acquire_mem(
        coher_cntl=CP_COHER_CNTL_SH_KCACHE_ACTION | CP_COHER_CNTL_TCL1_ACTION,
    )

    # 2. Program address (shifted right by 8 bits)
    pgm_addr = code_gpu_addr >> 8
    pm4.set_sh_reg(
        sh_reg_offset(COMPUTE_PGM_LO),
        pgm_addr & 0xFFFFFFFF,
        (pgm_addr >> 32) & 0xFFFFFFFF,
    )

    # 3. Program resources (RSRC1 and RSRC2 are consecutive)
    pm4.set_sh_reg(
        sh_reg_offset(COMPUTE_PGM_RSRC1),
        pgm_rsrc1,
        pgm_rsrc2,
    )

    # 4. RSRC3
    pm4.set_sh_reg(sh_reg_offset(COMPUTE_PGM_RSRC3), pgm_rsrc3)

    # 5. Scratch (none)
    pm4.set_sh_reg(sh_reg_offset(COMPUTE_TMPRING_SIZE), 0)

    # 6. Restart coordinates (zero)
    pm4.set_sh_reg(sh_reg_offset(COMPUTE_RESTART_X), 0, 0, 0)

    # 7. Kernarg pointer (in the USER_DATA slot the kernel descriptor expects)
    pm4.set_sh_reg(
        sh_reg_offset(COMPUTE_USER_DATA_0 + kernarg_sgpr_index),
        kernarg_gpu_addr & 0xFFFFFFFF,
        (kernarg_gpu_addr >> 32) & 0xFFFFFFFF,
    )

    # 8. Resource limits (0 = no restrictions)
    pm4.set_sh_reg(sh_reg_offset(COMPUTE_RESOURCE_LIMITS), 0)

    # 9. Start coordinates + workgroup dimensions
    pm4.set_sh_reg(
        sh_reg_offset(COMPUTE_START_X),
        0, 0, 0,                           # start x, y, z
        block[0], block[1], block[2],       # num_thread x, y, z
        0, 0,                               # trailing zeros
    )

    # 10. Dispatch (gfx12 wave32: COMPUTE_SHADER_EN|FORCE_START_000|ORDER_MODE|
    #     CS_W32_EN = 0x8045; the default 0x5 launches wave64 and never retires)
    pm4.dispatch_direct(grid[0], grid[1], grid[2], initiator=0x8045)

    # 11. CS_PARTIAL_FLUSH (ensures shader completion before fence)
    pm4.event_write(CS_PARTIAL_FLUSH, EVENT_INDEX_CS_PARTIAL_FLUSH)

    # 12. RELEASE_MEM fence
    pm4.release_mem(
        addr=fence_addr,
        value=fence_value,
        cache_flush=True,
    )

    return pm4.build()


# ============================================================================
# Noop shader - minimal compute kernel for pipeline validation
# ============================================================================

# GFX12 s_endpgm = 0xBF810000 (SOPP format: opcode=1)
# This shader does nothing and returns immediately. Used to verify
# the dispatch pipeline works end-to-end (SET_SH_REG -> DISPATCH_DIRECT
# -> CP fetches/executes shader -> RELEASE_MEM writes fence).
_NOOP_SHADER_CODE = struct.pack("<I", 0xBF810000)  # s_endpgm

# Kernel descriptor for the noop shader (64 bytes)
# All zero except:
#   kernel_code_entry_byte_offset = 64 (code starts after descriptor)
#   compute_pgm_rsrc1: float_mode=0xC0, dx10_clamp=1, ieee_mode=1
#   compute_pgm_rsrc2: enable_workgroup_id_x=1
# gfx12 RSRC1: VGPRS=3 (32 wave32), float_mode=0xC0, WGP_MODE|MEM_ORDERED|
# FWD_PROGRESS (bits 29/30/31). The gfx9-style 0x00AC0000 mis-launches on gfx12.
_NOOP_RSRC1 = 0xE00C0003
_NOOP_RSRC2 = (1 << 7)  # enable_sgpr_workgroup_id_x


def _build_noop_kernel_image() -> tuple[bytes, int, int, int]:
    """Build a minimal noop kernel: descriptor + s_endpgm.

    Returns:
        (image_bytes, pgm_rsrc1, pgm_rsrc2, pgm_rsrc3)
        image_bytes contains the 64-byte kernel descriptor followed by code.
        The code entry point is at offset 64 from the start of the image.
    """
    from amd_gpu_driver.kernel.descriptor import KernelDescriptor

    kd = KernelDescriptor(
        group_segment_fixed_size=0,
        private_segment_fixed_size=0,
        kernarg_size=0,
        kernel_code_entry_byte_offset=64,
        compute_pgm_rsrc3=0,
        compute_pgm_rsrc1=_NOOP_RSRC1,
        compute_pgm_rsrc2=_NOOP_RSRC2,
    )
    descriptor_bytes = kd.to_bytes()
    assert len(descriptor_bytes) == 64

    # Pad code to 256-byte alignment (GPU requires code address alignment)
    code = _NOOP_SHADER_CODE
    padding = (256 - (64 + len(code)) % 256) % 256
    image = descriptor_bytes + code + b"\x00" * padding

    return image, _NOOP_RSRC1, _NOOP_RSRC2, 0


# ============================================================================
# GPU bring-up sequence
# ============================================================================

def _read_discovery_via_vram_bar(
    dev: WindowsDevice, vram_size: int, read_size: int = 65536
) -> bytes:
    """Read the IP discovery table from the top of VRAM via the VRAM BAR.

    The table lives at (vram_size - 64KB) in VRAM. read_discovery_table_via_mmio
    feeds that VRAM byte-offset to SMN index/data, which cannot address VRAM
    data (returns 0). When the full VRAM BAR is exposed (ReBAR on; e.g. the
    32GB BAR0 on shark-a), map the region directly and read it. BAR0 = VRAM
    (VramBarIndex=0 per GET_INFO).
    """
    base = vram_size - read_size
    mapped_va, handle = dev.driver.map_bar(0, base, read_size)
    data = ctypes.string_at(mapped_va, read_size)
    try:
        dev.driver.unmap_bar(handle)
    except RuntimeError:
        # unmap_bar currently fails STATUS_INVALID_PARAMETER (the KMD's
        # UNMAP_BAR needs both MappingHandle and MappedAddress, but the Python
        # helper only passes the handle). The read already succeeded; leak the
        # 64KB mapping rather than abort discovery. TODO: fix unmap_bar.
        pass
    return data


def full_gpu_bringup(
    device_index: int = 0,
    fw_dir: str | Path = ".",
) -> GPUContext:
    """Run the full GPU initialization sequence.

    Opens the device, discovers IP blocks, initializes all IP subsystems,
    creates a compute queue, and runs a NOP fence self-test.

    Args:
        device_index: GPU index (0 for first AMD GPU).
        fw_dir: Directory containing firmware .bin files (for PSP).

    Returns:
        GPUContext with all initialized subsystems.

    Raises:
        RuntimeError: If any initialization step fails.
    """
    print("=" * 60)
    print("AMD GPU Bring-up (Windows Userspace Driver)")
    print("=" * 60)

    # --- 1. Open device ---
    print("\n[1/8] Opening device...")
    dev = WindowsDevice()
    dev.open(device_index)
    print(f"  Device: {dev.name}")
    print(f"  VRAM: {dev.vram_size // (1024**3)} GB")

    # --- 2. IP discovery ---
    print("\n[2/8] Running IP discovery...")
    raw_table = _read_discovery_via_vram_bar(dev, dev.vram_size)
    ip_result = parse_ip_discovery(raw_table)
    print(f"  Found {len(ip_result.ip_blocks)} IP blocks")
    for block in ip_result.ip_blocks:
        hw = getattr(block.hw_id, "name", None) or f"hw_id={int(block.hw_id)}"
        print(f"    {hw}: "
              f"v{block.major}.{block.minor}.{block.revision}")

    # --- 3. NBIO init ---
    print("\n[3/8] Initializing NBIO...")
    nbio_config = init_nbio(dev, ip_result)

    # --- 4. GMC init ---
    print("\n[4/8] Initializing GMC...")
    # GMC needs pre-allocated DMA buffers for GART page table and dummy page
    gart_cpu, gart_bus, gart_handle = dev.driver.alloc_dma(
        1024 * 1024)  # 1MB GART table
    dummy_cpu, dummy_bus, dummy_handle = dev.driver.alloc_dma(4096)
    # Zero the GART table and dummy page
    ctypes.memset(gart_cpu, 0, 1024 * 1024)
    ctypes.memset(dummy_cpu, 0, 4096)

    gmc_config = init_gmc(
        dev, ip_result, nbio_config,
        vram_size_bytes=dev.vram_size,
        gart_table_bus_addr=gart_bus,
        dummy_page_bus_addr=dummy_bus,
    )

    # --- 5. PSP init + firmware loading ---
    print("\n[5/8] Initializing PSP (firmware)...")
    psp_config = init_psp(dev, ip_result, fw_dir=fw_dir)
    # Load firmware (SOS should be alive from VBIOS POST)
    try:
        load_all_firmware(dev, psp_config)
    except FileNotFoundError as e:
        print(f"  WARNING: Firmware loading skipped - {e}")
        print("  Continuing with VBIOS-initialized firmware state")
    except RuntimeError as e:
        print(f"  WARNING: Firmware loading failed - {e}")
        print("  Continuing with VBIOS-initialized firmware state")

    # --- 5b. SMU mailbox: SetDriverDramAddr + EnableAllSmuFeatures(0) ---
    # Mirrors Linux _recipe_bringup [recipe 2/5]: after PSP AUTOLOAD_RLC the SMU
    # must publish the driver DRAM addr and enable all features (param=0) so the
    # IMU/RLC backdoor autoload can power up the GFX blocks and finish. WITHOUT
    # this, RLC_RLCS_BOOTLOAD_STATUS bit31 never sets, the MEC stays down, and
    # every compute dispatch fence-times-out. EnableAllSmuFeatures(0)=ALL (not
    # 3=SOC); do NOT DisallowGfxOff (matches macOS recipe).
    import os
    os.environ.setdefault("AMDGPU_LITE_ENABLE_SMU_FEATURES", "1")
    # On Windows the doorbell BAR is NOT effective under VFIO/WDDM passthrough,
    # so the dispatch relies on the MMIO CP_HQD_PQ_WPTR write (LITE_NO_MMIO_WPTR=0)
    # which does NOT wake GFXOFF -> DisallowGfxOff to keep the CP clocked.
    os.environ.setdefault("AMDGPU_LITE_DISALLOW_GFXOFF", "1")
    os.environ.setdefault("LITE_NO_MMIO_WPTR", "0")
    print("\n[5b] SMU mailbox (SetDriverDramAddr + EnableAllSmuFeatures)...")
    smu_config = None
    try:
        from amd_gpu_driver.backends.windows.smu_init import init_smu
        smu_config = init_smu(
            dev, ip_result,
            disable_gfxoff=True,
            vram_mc_base=gmc_config.vram_start,
        )
        print(f"  SMU: MP1[0]=0x{smu_config.mp1_base[0]:x} "
              f"{smu_config.messages.name}")
    except Exception as e:  # noqa: BLE001
        print(f"  SMU step failed (non-fatal): {e}")

    # --- 5c. Poll RLC_RLCS_BOOTLOAD_STATUS bit31 (autoload completion) ---
    # GC BASE_IDX=1 DWORD base = 0xA000 on gfx1201 (validated). bit31 =
    # BOOTLOAD_COMPLETE. RESET_CTRL==0x7F when all 7 GFX blocks released;
    # CORE_CTRL==0x8 when IMU running; RLC_CNTL==0x1 when RLC enabled.
    _GC_B1_DW = 0xA000
    _bl = 0
    _deadline = time.monotonic() + 5.0
    while time.monotonic() < _deadline:
        _bl = dev.read_reg32((_GC_B1_DW + 0x4e7c) * 4)
        if _bl & 0x80000000:
            break
        time.sleep(0.01)
    _reset = dev.read_reg32((_GC_B1_DW + 0x40bc) * 4)
    _core = dev.read_reg32((_GC_B1_DW + 0x40b6) * 4)
    _rlc = dev.read_reg32((_GC_B1_DW + 0x4c00) * 4)
    print(f"  BOOTLOAD_STATUS=0x{_bl:08x} RESET_CTRL=0x{_reset:08x} "
          f"CORE_CTRL=0x{_core:x} RLC_CNTL=0x{_rlc:x}  [want bit31 set]")
    if _bl & 0x80000000:
        print("  PASS: BOOTLOAD_COMPLETE - RLC/IMU autoload succeeded")
    else:
        print("  FAIL: BOOTLOAD_COMPLETE not set within timeout")

    # --- 6. IH init ---
    print("\n[6/8] Initializing IH (interrupts)...")
    ih_config = init_ih(dev, ip_result, nbio_config)

    # --- 6b. GFX/MEC enable for direct compute ---
    # Mirrors Linux _recipe_bringup [recipe 4/5] (_recipe_mes_start +
    # init_gfx_for_compute): set the MEC program counter from the gfx fw headers,
    # program CP_MEC_DOORBELL_RANGE, enable the MEC pipes (clear HALT), enable
    # MES. Without this the MEC never services the queue (RPTR stays 0 -> fence
    # timeout). The LITE_MES_RECIPE PSP path doesn't populate ucode_start, so
    # fill it from the fw headers (PFP/ME/MEC ucode_start @+52; MES uni_mes @+56).
    print("\n[6b] GFX/MEC enable (init_gfx_for_compute)...")
    try:
        import struct as _struct
        from pathlib import Path as _Path
        from amd_gpu_driver.backends.windows.psp_init import (
            _read_firmware as _rf,
        )
        from amd_gpu_driver.backends.windows.ring_init import (
            init_gfx_for_compute,
        )
        _gc = psp_config.ip_versions.get("gc", "12_0_1")

        def _rs64_entry(_name: str) -> int:
            _b = _rf(_Path(fw_dir) / f"gc_{_gc}_{_name}.bin")
            _lo, _hi = _struct.unpack_from("<II", _b, 52)
            return (_hi << 32) | _lo

        _mb = _rf(_Path(fw_dir) / f"gc_{_gc}_uni_mes.bin")
        _mlo, _mhi = _struct.unpack_from("<II", _mb, 56)
        _mes = (_mhi << 32) | _mlo
        if not getattr(psp_config, "ucode_start", None):
            psp_config.ucode_start = {}
        psp_config.ucode_start.update({
            "PFP": _rs64_entry("pfp"), "ME": _rs64_entry("me"),
            "MEC": _rs64_entry("mec"), "MES": _mes, "MES1": _mes,
        })
        psp_config.mes_psp_loaded = True
        init_gfx_for_compute(dev, ip_result, psp_config, smu_config)
    except Exception as e:  # noqa: BLE001
        print(f"  GFX/MEC enable failed (non-fatal): {e}")

    # NOTE: GFXHUB GPUVM (gfxhub_gart_enable + CONTEXT0) is intentionally NOT
    # enabled in the general bring-up. Enabling CONTEXT0 with fault-enable makes
    # VMID-0 memory accesses (e.g. the WRITE_DATA self-test target) get GPUVM-
    # translated and fault. Only the shader dispatch needs translation, so
    # build_compute_gpuvm() sets the hub + page table up just before dispatch.

    # --- 7. Compute ring ---
    print("\n[7/8] Creating compute queue...")
    compute_queue = init_compute_queue(dev, ip_result, nbio_config)

    # --- 8. NOP fence test ---
    print("\n[8/8] Running NOP + fence self-test...")
    if test_compute_nop_fence(compute_queue):
        print("  PASS: NOP + RELEASE_MEM fence completed")
    else:
        print("  FAIL: Fence timeout - GPU may not be processing commands")

    print("\n" + "=" * 60)
    print("GPU bring-up complete!")
    print("=" * 60)

    return GPUContext(
        dev=dev,
        ip_result=ip_result,
        nbio_config=nbio_config,
        gmc_config=gmc_config,
        psp_config=psp_config,
        ih_config=ih_config,
        compute_queue=compute_queue,
        gart_table_dma_handle=gart_handle,
        dummy_page_dma_handle=dummy_handle,
    )


# ============================================================================
# Memory write test (PM4 WRITE_DATA - no shader needed)
# ============================================================================

def test_write_data(ctx: GPUContext, num_dwords: int = 16) -> bool:
    """Verify PM4 WRITE_DATA by writing and reading back values.

    Uses WRITE_DATA to write known patterns to a DMA buffer,
    then reads back via CPU to verify. This tests the compute queue
    packet processing without requiring a compiled shader.

    Args:
        ctx: Initialized GPU context.
        num_dwords: Number of DWORDs to write (default 16 = 64 bytes).

    Returns:
        True if all values match.
    """
    print("\n--- WRITE_DATA memory test ---")
    cq = ctx.compute_queue

    # Allocate a test buffer
    buf_size = max(num_dwords * 4, 4096)
    buf_cpu, buf_bus, buf_handle = ctx.dev.driver.alloc_dma(buf_size)
    ctypes.memset(buf_cpu, 0, buf_size)

    # Write known pattern via PM4 WRITE_DATA
    test_values = [(0xCAFE0000 + i) & 0xFFFFFFFF for i in range(num_dwords)]

    write_pkt = _build_write_data_packet(buf_bus, test_values)

    # Also add a RELEASE_MEM fence to know when the writes are done
    fence_seq = ctx.next_fence_seq()
    ctypes.c_uint64.from_address(cq.fence_cpu_addr).value = 0

    pm4 = PM4PacketBuilder()
    pm4.release_mem(
        addr=cq.fence_bus_addr,
        value=fence_seq,
        cache_flush=True,
    )
    fence_pkt = pm4.build()

    # Submit: WRITE_DATA + RELEASE_MEM
    submit_compute_packets(cq, write_pkt + fence_pkt)

    # Wait for fence
    if not wait_fence(cq, fence_seq, timeout_ms=5000):
        print("  FAIL: Fence timeout after WRITE_DATA")
        ctx.dev.driver.free_dma(buf_handle)
        return False

    # Read back and verify
    result = (ctypes.c_uint32 * num_dwords).from_address(buf_cpu)
    mismatches = []
    for i in range(num_dwords):
        if result[i] != test_values[i]:
            mismatches.append(
                f"  [{i}] expected 0x{test_values[i]:08X}, "
                f"got 0x{result[i]:08X}")

    ctx.dev.driver.free_dma(buf_handle)

    if mismatches:
        print(f"  FAIL: {len(mismatches)}/{num_dwords} mismatches:")
        for m in mismatches[:5]:
            print(m)
        return False

    print(f"  PASS: {num_dwords} DWORDs written and verified")
    return True


# ============================================================================
# Noop shader dispatch test
# ============================================================================

def test_noop_dispatch(ctx: GPUContext) -> bool:
    """Dispatch a noop shader (s_endpgm) to validate the compute pipeline.

    This tests the full dispatch path:
    1. Upload noop kernel (descriptor + s_endpgm) to DMA memory
    2. Build SET_SH_REG + DISPATCH_DIRECT + RELEASE_MEM packets
    3. Submit to compute queue
    4. Wait for fence

    The shader does nothing, but the GPU must successfully fetch it,
    execute it, and signal completion. This validates that the compute
    unit is alive and the register programming is correct.

    Args:
        ctx: Initialized GPU context.

    Returns:
        True if the dispatch completed (fence received).
    """
    print("\n--- Noop shader dispatch test ---")
    cq = ctx.compute_queue

    # Stage s_endpgm in VRAM at the 256-aligned base (no descriptor - a raw HW
    # dispatch programs RSRC1/2 directly). The code MUST be in VRAM (alloc_dma is
    # not GPU-fetchable on this KMD) and reachable via a GPUVM virtual address the
    # compute wave's VMID-0 page table translates.
    from amd_gpu_driver.backends.base import MemoryLocation
    from amd_gpu_driver.backends.windows.gmc_init import build_compute_gpuvm

    code = ctx.dev.alloc_memory(4096, MemoryLocation.VRAM)
    buf = (ctypes.c_uint32 * 16).from_address(code.cpu_addr)
    for i in range(16):
        buf[i] = 0xBF810000  # s_endpgm
    shader_va = build_compute_gpuvm(
        ctx.dev, ctx.gmc_config, ctx.nbio_config, code.gpu_addr)

    fence_seq = ctx.next_fence_seq()
    ctypes.c_uint64.from_address(cq.fence_cpu_addr).value = 0
    packets = _build_dispatch_packets(
        code_gpu_addr=shader_va,          # GPUVM virtual address (not physical)
        kernarg_gpu_addr=0,
        pgm_rsrc1=_NOOP_RSRC1,
        pgm_rsrc2=_NOOP_RSRC2,
        pgm_rsrc3=0,
        grid=(1, 1, 1),
        block=(1, 1, 1),
        fence_addr=cq.fence_bus_addr,
        fence_value=fence_seq,
    )
    submit_compute_packets(cq, packets)
    if not wait_fence(cq, fence_seq, timeout_ms=5000):
        print("  FAIL: Fence timeout after noop dispatch")
        return False
    print("  PASS: Noop shader dispatched and completed")
    return True


# ============================================================================
# ELF kernel dispatch
# ============================================================================

def dispatch_elf_kernel(
    ctx: GPUContext,
    co_path: str | Path,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    args: list,
    *,
    kernel_name: str | None = None,
    timeout_ms: int = 5000,
) -> bool:
    """Dispatch a compute kernel from a compiled ELF code object (.co file).

    Loads the code into VRAM honoring section vaddrs, GPUVM-maps the code +
    kernargs + pointer-arg buffers (VMID 0), packs the kernargs, dispatches via
    PM4 and waits for the completion fence.

    Args:
        ctx: Initialized GPU context.
        co_path: Path to .co / .hsaco file compiled for the target GPU.
        grid: Dispatch dimensions (workgroups in x, y, z).
        block: Workgroup dimensions (threads in x, y, z).
        args: Kernel arguments. Each is either a VRAM ``MemoryHandle`` (a pointer
            argument -- its GPUVM-mapped VA is packed) or an int (a scalar,
            packed verbatim).
        kernel_name: Specific kernel name (None = first kernel found).
        timeout_ms: Fence timeout in milliseconds.

    Returns:
        True if the dispatch completed within the timeout.
    """
    from amd_gpu_driver.kernel.elf_parser import parse_elf_file
    from amd_gpu_driver.kernel.descriptor import KernelDescriptor
    from amd_gpu_driver.kernel.metadata import kernel_arg_layout
    from amd_gpu_driver.backends.base import MemoryLocation, MemoryHandle
    from amd_gpu_driver.backends.windows.gmc_init import map_compute_buffers
    from amd_gpu_driver.backends.windows.nbio_init import hdp_flush

    co_path = Path(co_path)
    print(f"\n--- ELF kernel dispatch: {co_path.name} ---")

    co = parse_elf_file(str(co_path))
    kernels = co.kernel_symbols()
    if not kernels:
        raise RuntimeError(f"No kernel symbols in {co_path}")
    if kernel_name is not None:
        matches = [k for k in kernels if k.name == kernel_name]
        if not matches:
            raise RuntimeError(
                f"Kernel '{kernel_name}' not found. "
                f"Available: {[k.name for k in kernels]}")
        kernel_sym = matches[0]
    else:
        kernel_sym = kernels[0]
    print(f"  Kernel: {kernel_sym.name}")

    # The kernel descriptor is the "<name>.kd" object symbol; read its 64 bytes
    # from whichever allocated section its vaddr falls in.
    kd_syms = [s for s in co.symbols if s.name == kernel_sym.name + ".kd"]
    if not kd_syms:
        raise RuntimeError(f"No kernel descriptor {kernel_sym.name}.kd in ELF")
    kd_sym = kd_syms[0]
    kd_sec = next((x for x in co.sections
                   if x.sh_size and x.sh_addr <= kd_sym.st_value
                   < x.sh_addr + x.sh_size), None)
    if kd_sec is None:
        raise RuntimeError("Kernel descriptor section not found")
    kd = KernelDescriptor.from_bytes(
        co.get_section_data(kd_sec)[kd_sym.st_value - kd_sec.sh_addr:][:64])
    print(f"  RSRC1=0x{kd.compute_pgm_rsrc1:08X} RSRC2=0x{kd.compute_pgm_rsrc2:08X} "
          f"RSRC3=0x{kd.compute_pgm_rsrc3:08X} kernarg={kd.kernarg_size} "
          f"LDS={kd.group_segment_fixed_size}")
    if kd.private_segment_fixed_size:
        raise RuntimeError(
            f"kernel needs {kd.private_segment_fixed_size} bytes of scratch "
            "per work-item (scratch programming not yet implemented)")

    # Assemble the loadable image by SECTION VADDR -- on AMDGPU code objects the
    # section vaddrs differ from their file offsets (e.g. .text loads above the
    # descriptor). Upload it to a VRAM allocation; GPU VA(x) = code_va + x.
    SHF_ALLOC, SHT_NOBITS = 0x2, 8
    alloc_secs = [s for s in co.sections if (s.sh_flags & SHF_ALLOC) and s.sh_addr]
    if not alloc_secs:
        raise RuntimeError("No allocatable sections in ELF")
    img_bytes = (max(s.sh_addr + s.sh_size for s in alloc_secs) + 0xFFF) & ~0xFFF
    image = bytearray(img_bytes)
    for s in alloc_secs:
        if s.sh_type != SHT_NOBITS:
            image[s.sh_addr:s.sh_addr + s.sh_size] = co.get_section_data(s)
    code = ctx.dev.alloc_memory(img_bytes, MemoryLocation.VRAM)
    ctypes.memmove(code.cpu_addr, bytes(image), img_bytes)

    # Kernarg buffer in VRAM. Pointer args are VRAM MemoryHandles (their mapped
    # VA is packed); scalar args are ints (packed verbatim).
    kernarg = ctx.dev.alloc_memory(max(kd.kernarg_size, 4096), MemoryLocation.VRAM)
    ctypes.memset(kernarg.cpu_addr, 0, kernarg.size)
    ptr_handles = [a for a in args if isinstance(a, MemoryHandle)]

    # GPUVM-map code + kernarg + pointer-arg buffers (one PTB, VMID 0).
    vas = map_compute_buffers(ctx.dev, ctx.gmc_config, ctx.nbio_config,
                              [code, kernarg] + ptr_handles)
    code_va, kernarg_va = vas[0], vas[1]
    handle_va = {id(h): v for h, v in zip(ptr_handles, vas[2:])}

    kbuf = (ctypes.c_char * kernarg.size).from_address(kernarg.cpu_addr)
    layout = kernel_arg_layout(co, kernel_sym.name)
    _fmt = {1: "<B", 2: "<H", 4: "<I", 8: "<Q"}
    if layout:
        # Pack explicit args at their metadata offsets (positional match), then
        # fill the COV5 hidden args so get_local_size()/get_num_groups()/the grid
        # are correct -- required for multi-workgroup dispatches.
        explicit = [a for a in layout if not a["value_kind"].startswith("hidden_")]
        if len(args) != len(explicit):
            raise RuntimeError(
                f"{kernel_sym.name} expects {len(explicit)} args, got {len(args)}")
        for user_arg, ma in zip(args, explicit):
            value = (handle_va[id(user_arg)] if isinstance(user_arg, MemoryHandle)
                     else user_arg)
            sz = ma["size"]
            struct.pack_into(_fmt.get(sz, "<Q"), kbuf, ma["offset"],
                             value & ((1 << (sz * 8)) - 1))
        dims = (3 if (grid[2] > 1 or block[2] > 1)
                else 2 if (grid[1] > 1 or block[1] > 1) else 1)
        hidden = {
            "hidden_block_count_x": grid[0], "hidden_block_count_y": grid[1],
            "hidden_block_count_z": grid[2],
            "hidden_group_size_x": block[0], "hidden_group_size_y": block[1],
            "hidden_group_size_z": block[2],
            "hidden_remainder_x": 0, "hidden_remainder_y": 0, "hidden_remainder_z": 0,
            "hidden_grid_dims": dims,
        }
        for ma in layout:
            if ma["value_kind"] in hidden:
                sz = ma["size"]
                struct.pack_into(_fmt.get(sz, "<I"), kbuf, ma["offset"],
                                 hidden[ma["value_kind"]] & ((1 << (sz * 8)) - 1))
    else:
        # No metadata: sequential 8-byte packing (single-workgroup only).
        offset = 0
        for arg in args:
            value = (handle_va[id(arg)] if isinstance(arg, MemoryHandle)
                     else arg & 0xFFFFFFFFFFFFFFFF)
            struct.pack_into("<Q", kbuf, offset, value)
            offset += 8

    code_entry_va = code_va + (kd_sym.st_value + kd.kernel_code_entry_byte_offset)

    # The kernarg segment pointer lands in the USER_DATA slot following any
    # earlier-enabled preload SGPRs (canonical order: private-seg-buffer=4,
    # dispatch-ptr=2, queue-ptr=2, then kernarg-ptr).
    slot = 0
    if kd.enable_sgpr_private_segment_buffer:
        slot += 4
    if kd.enable_sgpr_dispatch_ptr:
        slot += 2
    if kd.enable_sgpr_queue_ptr:
        slot += 2

    cq = ctx.compute_queue
    fence_seq = ctx.next_fence_seq()
    ctypes.c_uint64.from_address(cq.fence_cpu_addr).value = 0
    packets = _build_dispatch_packets(
        code_gpu_addr=code_entry_va,
        kernarg_gpu_addr=kernarg_va,
        pgm_rsrc1=kd.compute_pgm_rsrc1,
        pgm_rsrc2=kd.compute_pgm_rsrc2,
        pgm_rsrc3=kd.compute_pgm_rsrc3,
        grid=grid,
        block=block,
        fence_addr=cq.fence_bus_addr,
        fence_value=fence_seq,
        kernarg_sgpr_index=slot,
    )
    print(f"  code_entry=0x{code_entry_va:X} kernarg=0x{kernarg_va:X} "
          f"sgpr_slot={slot} grid={grid} block={block}")
    submit_compute_packets(cq, packets)
    if not wait_fence(cq, fence_seq, timeout_ms=timeout_ms):
        print(f"  FAIL: Fence timeout after {timeout_ms}ms")
        return False
    hdp_flush(ctx.dev, ctx.nbio_config)  # make GPU's VRAM writes CPU-visible
    print("  PASS: Dispatch completed")
    return True


# ============================================================================
# Fill buffer test (dispatch with result verification)
# ============================================================================

def test_fill_dispatch(
    ctx: GPUContext,
    co_path: str | Path,
    *,
    num_elements: int = 256,
    fill_value: int = 0xDEADBEEF,
    kernel_name: str = "fill_kernel",
) -> bool:
    """Dispatch a real compiled fill kernel and verify the output buffer.

    The kernel reads its output pointer from the kernarg segment and writes
    ``fill_value`` to ``out[tid]`` where ``tid = local_id + group_id*group_size``.
    Output lives in VRAM and is GPUVM-mapped; the kernarg's pointer arg is the
    output's mapped VA. Launches ceil(num_elements/64) workgroups of 64 threads;
    dispatch_elf_kernel populates the COV5 hidden args so get_local_size() across
    workgroups is correct.

    Args:
        ctx: Initialized GPU context.
        co_path: Path to compiled fill kernel .co file.
        num_elements: Number of uint32 elements to fill.
        fill_value: Value to fill with.
        kernel_name: Name of the kernel in the .co file.

    Returns:
        True if all elements match the fill value.
    """
    from amd_gpu_driver.backends.base import MemoryLocation

    print(f"\n--- Fill kernel test ({num_elements} elements) ---")
    block = min(num_elements, 64) or 1
    grid = (num_elements + block - 1) // block
    out = ctx.dev.alloc_memory(max(grid * block * 4, 4096), MemoryLocation.VRAM)
    ctypes.memset(out.cpu_addr, 0, out.size)

    if not dispatch_elf_kernel(
        ctx, co_path,
        grid=(grid, 1, 1),
        block=(block, 1, 1),
        args=[out, fill_value],
        kernel_name=kernel_name,
    ):
        return False

    result = (ctypes.c_uint32 * num_elements).from_address(out.cpu_addr)
    bad = [i for i in range(num_elements) if result[i] != fill_value]
    if bad:
        print(f"  FAIL: {len(bad)}/{num_elements} mismatches "
              f"(first at index {bad[0]}: got 0x{result[bad[0]]:08X})")
        return False

    print(f"  PASS: All {num_elements} elements = 0x{fill_value:08X}")
    return True


# ============================================================================
# DMA buffer allocation helper for external use
# ============================================================================

def alloc_gpu_buffer(ctx: GPUContext, size: int) -> tuple[int, int, int]:
    """Allocate a DMA buffer accessible by both CPU and GPU.

    Returns:
        (cpu_addr, bus_addr, handle) - cpu_addr for CPU access,
        bus_addr for GPU access (as kernarg or output pointer).
    """
    cpu_addr, bus_addr, handle = ctx.dev.driver.alloc_dma(
        max(size, 4096))
    ctypes.memset(cpu_addr, 0, max(size, 4096))
    return cpu_addr, bus_addr, handle


def free_gpu_buffer(ctx: GPUContext, handle: int) -> None:
    """Free a previously allocated DMA buffer."""
    ctx.dev.driver.free_dma(handle)


# ============================================================================
# Shutdown
# ============================================================================

def shutdown(ctx: GPUContext) -> None:
    """Clean shutdown: close device and release resources.

    Note: Does not free individual DMA buffers from tests.
    The kernel driver will reclaim all resources when the device is closed.
    """
    print("\nShutting down GPU...")
    ctx.dev.close()
    print("  Device closed")


# ============================================================================
# Demo entry point
# ============================================================================

def run_demo(
    device_index: int = 0,
    fw_dir: str = ".",
    kernel_co: str | None = None,
) -> None:
    """Run the full GPU bring-up and self-test sequence.

    Args:
        device_index: GPU index.
        fw_dir: Firmware directory.
        kernel_co: Optional path to a compiled fill kernel .co file.
    """
    ctx = full_gpu_bringup(device_index=device_index, fw_dir=fw_dir)

    print("\n" + "=" * 60)
    print("Running self-tests")
    print("=" * 60)

    results: list[tuple[str, bool]] = []

    # Test 1: WRITE_DATA
    results.append(("WRITE_DATA memory test", test_write_data(ctx)))

    # Test 2: Noop dispatch
    results.append(("Noop shader dispatch", test_noop_dispatch(ctx)))

    # Test 3: Fill kernel (if .co file provided)
    if kernel_co is not None:
        results.append((
            "Fill kernel dispatch",
            test_fill_dispatch(ctx, kernel_co),
        ))

    # Summary
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed!")
    else:
        print(f"\n{sum(1 for _, p in results if not p)} test(s) failed")

    shutdown(ctx)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AMD GPU bring-up and compute dispatch demo")
    parser.add_argument(
        "--device", type=int, default=0,
        help="GPU device index (default: 0)")
    parser.add_argument(
        "--fw-dir", type=str, default=".",
        help="Directory containing firmware .bin files")
    parser.add_argument(
        "--kernel", type=str, default=None,
        help="Path to compiled fill kernel .co file (optional)")
    args = parser.parse_args()

    run_demo(
        device_index=args.device,
        fw_dir=args.fw_dir,
        kernel_co=args.kernel,
    )
