"""Parse the NT_AMDGPU_METADATA note (msgpack) of an AMDGPU code object to get
each kernel's argument layout (offset / size / value_kind), including the COV5
hidden arguments (hidden_block_count_*, hidden_group_size_*, etc.).

This is what lets a dispatch populate the implicit args a kernel reads for
get_local_size()/get_num_groups()/etc., so multi-workgroup grids work.
"""
from __future__ import annotations

import struct

from amd_gpu_driver.kernel.elf_parser import AMDGPUCodeObject

SHT_NOTE = 7
NT_AMDGPU_METADATA = 32


def _decode(data: bytes, i: int):
    """Minimal msgpack decoder for the subset used by AMDGPU metadata.

    Returns (value, next_index).
    """
    b = data[i]
    i += 1
    if b < 0x80:  # positive fixint
        return b, i
    if b >= 0xE0:  # negative fixint
        return b - 0x100, i
    if 0x80 <= b <= 0x8F:  # fixmap
        return _decode_map(data, i, b & 0x0F)
    if 0x90 <= b <= 0x9F:  # fixarray
        return _decode_array(data, i, b & 0x0F)
    if 0xA0 <= b <= 0xBF:  # fixstr
        n = b & 0x1F
        return data[i:i + n].decode("utf-8", "replace"), i + n
    if b == 0xC0:  # nil
        return None, i
    if b == 0xC2:  # false
        return False, i
    if b == 0xC3:  # true
        return True, i
    if b == 0xCC:  # uint8
        return data[i], i + 1
    if b == 0xCD:  # uint16
        return struct.unpack_from(">H", data, i)[0], i + 2
    if b == 0xCE:  # uint32
        return struct.unpack_from(">I", data, i)[0], i + 4
    if b == 0xCF:  # uint64
        return struct.unpack_from(">Q", data, i)[0], i + 8
    if b == 0xD0:  # int8
        return struct.unpack_from(">b", data, i)[0], i + 1
    if b == 0xD1:  # int16
        return struct.unpack_from(">h", data, i)[0], i + 2
    if b == 0xD2:  # int32
        return struct.unpack_from(">i", data, i)[0], i + 4
    if b == 0xD3:  # int64
        return struct.unpack_from(">q", data, i)[0], i + 8
    if b == 0xD9:  # str8
        n = data[i]
        i += 1
        return data[i:i + n].decode("utf-8", "replace"), i + n
    if b == 0xDA:  # str16
        n = struct.unpack_from(">H", data, i)[0]
        i += 2
        return data[i:i + n].decode("utf-8", "replace"), i + n
    if b == 0xDB:  # str32
        n = struct.unpack_from(">I", data, i)[0]
        i += 4
        return data[i:i + n].decode("utf-8", "replace"), i + n
    if b == 0xDC:  # array16
        n = struct.unpack_from(">H", data, i)[0]
        return _decode_array(data, i + 2, n)
    if b == 0xDD:  # array32
        n = struct.unpack_from(">I", data, i)[0]
        return _decode_array(data, i + 4, n)
    if b == 0xDE:  # map16
        n = struct.unpack_from(">H", data, i)[0]
        return _decode_map(data, i + 2, n)
    if b == 0xDF:  # map32
        n = struct.unpack_from(">I", data, i)[0]
        return _decode_map(data, i + 4, n)
    raise ValueError(f"unsupported msgpack byte 0x{b:02x} at {i - 1}")


def _decode_array(data: bytes, i: int, n: int):
    out = []
    for _ in range(n):
        v, i = _decode(data, i)
        out.append(v)
    return out, i


def _decode_map(data: bytes, i: int, n: int):
    out = {}
    for _ in range(n):
        k, i = _decode(data, i)
        v, i = _decode(data, i)
        out[k] = v
    return out, i


def parse_amdgpu_metadata(co: AMDGPUCodeObject) -> dict:
    """Return the decoded NT_AMDGPU_METADATA map, or {} if absent."""
    for sh in co.sections:
        if sh.sh_type != SHT_NOTE:
            continue
        note = co.raw_data[sh.sh_offset:sh.sh_offset + sh.sh_size]
        j = 0
        while j + 12 <= len(note):
            namesz, descsz, ntype = struct.unpack_from("<III", note, j)
            j += 12
            name = note[j:j + namesz]
            j += (namesz + 3) & ~3
            desc = note[j:j + descsz]
            j += (descsz + 3) & ~3
            if ntype == NT_AMDGPU_METADATA and name.startswith(b"AMDGPU"):
                md, _ = _decode(desc, 0)
                return md
    return {}


def kernel_arg_layout(co: AMDGPUCodeObject, kernel_name: str) -> list:
    """Return the kernel's argument list: [{offset, size, value_kind}, ...]
    (explicit args first, then the COV5 hidden args). Empty if no metadata."""
    md = parse_amdgpu_metadata(co)
    for k in md.get("amdhsa.kernels", []):
        if k.get(".name") == kernel_name or k.get(".symbol") == kernel_name + ".kd":
            return [
                {"offset": a.get(".offset", 0), "size": a.get(".size", 0),
                 "value_kind": a.get(".value_kind", "")}
                for a in k.get(".args", [])
            ]
    return []
