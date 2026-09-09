# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Strict observed device inventory, separate from compiled adapter contracts.

UUIDs encode the driver's raw 16 bytes and are scoped to an inventory's vendor
and backend. They are not globally unique hardware handles. Launch ordinals are
process-local observations and may change between discovery and execution.
"""

import json
import re
from dataclasses import dataclass
from typing import cast

from .gpu_targets import Backend, GpuTarget, Vendor

_BACKENDS: dict[str, str] = {"amd": "hip", "nvidia": "cuda", "intel": "level-zero"}
_VENDOR_IDS = {"amd": 0x1002, "nvidia": 0x10DE, "intel": 0x8086}


def _object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"Invalid {label} fields")
    return cast(dict[str, object], value)


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate inventory field: {key}")
        result[key] = value
    return result


def _integer(value: object, label: str, minimum: int = 1) -> int:
    if type(value) is not int or not minimum <= value <= (1 << 64) - 1:
        raise ValueError(f"Invalid {label}: expected integer >= {minimum}")
    return cast(int, value)


def _dimensions(value: object, label: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Invalid {label}: expected three dimensions")
    return cast(tuple[int, int, int], tuple(_integer(v, label) for v in value))


def parse_device_uuid(value: object) -> str:
    """Require a nonzero raw driver UUID encoded as 32 lowercase hex digits."""
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[0-9a-f]{32}", value)
        or value == "0" * 32
    ):
        raise ValueError(
            "Device UUID must be 32 lowercase hexadecimal digits and nonzero"
        )
    return value


@dataclass(frozen=True)
class DeviceLimits:
    max_threads_per_block: int
    max_block_dimensions: tuple[int, int, int]
    max_grid_dimensions: tuple[int, int, int]
    total_memory_bytes: int

    def record(self) -> dict[str, object]:
        return {
            "max_threads_per_block": self.max_threads_per_block,
            "max_block_dimensions": list(self.max_block_dimensions),
            "max_grid_dimensions": list(self.max_grid_dimensions),
            "total_memory_bytes": self.total_memory_bytes,
        }


@dataclass(frozen=True)
class ObservedDevice:
    index: int
    name: str
    device_uuid: str
    architecture: str | None
    vendor_id: int
    device_id: int | None
    limits: DeviceLimits

    def __post_init__(self) -> None:
        parse_device_uuid(self.device_uuid)

    def record(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "device_uuid": self.device_uuid,
            "architecture": self.architecture,
            "vendor_id": self.vendor_id,
            "device_id": self.device_id,
            "limits": self.limits.record(),
        }


@dataclass(frozen=True)
class DeviceInventory:
    vendor: Vendor
    backend: Backend
    devices: tuple[ObservedDevice, ...]

    def record(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "kind": "native-device-inventory",
            "scope": "observed-runtime",
            "vendor": self.vendor,
            "backend": self.backend,
            "devices": [device.record() for device in self.devices],
        }


def parse_inventory_json(text: str) -> DeviceInventory:
    try:
        document: object = json.loads(text, object_pairs_hook=_unique_fields)
    except RecursionError as exc:
        raise ValueError("Device inventory JSON is too deeply nested") from exc
    raw = _object(
        document,
        {"schema_version", "kind", "scope", "vendor", "backend", "devices"},
        "device inventory",
    )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 2:
        raise ValueError("Unsupported device inventory schema")
    if raw["kind"] != "native-device-inventory" or raw["scope"] != "observed-runtime":
        raise ValueError("Device inventory must have observed-runtime scope")
    vendor, backend = raw["vendor"], raw["backend"]
    if (
        not isinstance(vendor, str)
        or vendor not in _BACKENDS
        or backend != _BACKENDS[vendor]
    ):
        raise ValueError("Device inventory vendor/backend mismatch")
    values = raw["devices"]
    if not isinstance(values, list):
        raise ValueError("Device inventory devices must be a list")
    devices: list[ObservedDevice] = []
    seen_uuids: set[str] = set()
    for index, value in enumerate(values):
        entry = _object(
            value,
            {
                "index",
                "name",
                "device_uuid",
                "architecture",
                "vendor_id",
                "device_id",
                "limits",
            },
            "observed device",
        )
        if _integer(entry["index"], "device index", 0) != index:
            raise ValueError(
                "Device inventory indices must be consecutive launch ordinals"
            )
        name = entry["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Device name must be a nonempty string")
        device_uuid = parse_device_uuid(entry["device_uuid"])
        if device_uuid in seen_uuids:
            raise ValueError("Duplicate device UUID within one inventory")
        seen_uuids.add(device_uuid)
        architecture = entry["architecture"]
        if vendor == "intel":
            if architecture is not None:
                raise ValueError(
                    "Intel observed architecture must be null; identity uses PCI IDs"
                )
        else:
            pattern = r"gfx[0-9a-f]{3,}" if vendor == "amd" else r"sm_[1-9][0-9]{1,2}"
            if not isinstance(architecture, str) or not re.fullmatch(
                pattern, architecture
            ):
                raise ValueError("Invalid observed hardware architecture")
        vendor_id = _integer(entry["vendor_id"], "vendor ID")
        if vendor_id != _VENDOR_IDS[vendor]:
            raise ValueError("Observed PCI vendor ID differs from inventory vendor")
        device_id = entry["device_id"]
        if device_id is not None:
            device_id = _integer(device_id, "device ID")
            if device_id > 0xFFFF:
                raise ValueError("Device ID must fit in 16 bits")
        elif vendor == "intel":
            raise ValueError("Intel observed device requires a PCI device ID")
        limits = _object(
            entry["limits"],
            {
                "max_threads_per_block",
                "max_block_dimensions",
                "max_grid_dimensions",
                "total_memory_bytes",
            },
            "device limits",
        )
        devices.append(
            ObservedDevice(
                index,
                name,
                device_uuid,
                cast(str | None, architecture),
                vendor_id,
                cast(int | None, device_id),
                DeviceLimits(
                    _integer(
                        limits["max_threads_per_block"], "maximum threads per block"
                    ),
                    _dimensions(
                        limits["max_block_dimensions"], "maximum block dimensions"
                    ),
                    _dimensions(
                        limits["max_grid_dimensions"], "maximum grid dimensions"
                    ),
                    _integer(limits["total_memory_bytes"], "total memory bytes"),
                ),
            )
        )
    return DeviceInventory(cast(Vendor, vendor), cast(Backend, backend), tuple(devices))


def select_device(
    inventory: DeviceInventory,
    target: GpuTarget,
    index: int,
    expected_device_id: int | None = None,
) -> ObservedDevice:
    """Require exact observable identity; this does not qualify kernel execution."""
    if inventory.vendor != target.vendor or inventory.backend != target.backend:
        raise ValueError("Device inventory vendor/backend does not match target")
    if type(index) is not int or index < 0 or index >= len(inventory.devices):
        raise ValueError(
            f"Device index {index} is unavailable for {target.canonical_id}"
        )
    if target.features or (
        target.vendor == "nvidia" and target.processor[-1].isalpha()
    ):
        raise ValueError("Discovery cannot qualify architecture feature suffixes")
    device = inventory.devices[index]
    if target.vendor == "intel":
        if target.processor == "xe2-b70":
            if expected_device_id not in (None, 0xE223):
                raise ValueError("Intel xe2-b70 requires device ID 0xe223")
            expected_device_id = 0xE223
        if type(expected_device_id) is not int or not 0 < expected_device_id <= 0xFFFF:
            raise ValueError("Intel target requires an explicit 16-bit device ID")
        if device.device_id != expected_device_id:
            raise ValueError(
                f"Observed Intel device ID {device.device_id:#x} does not match {expected_device_id:#x}"
            )
    else:
        if expected_device_id is not None:
            raise ValueError("Expected device ID is only supported for Intel")
        if device.architecture != target.processor:
            raise ValueError(
                f"Observed architecture {device.architecture} does not match {target.processor}"
            )
    return device


def select_device_uuid(
    inventory: DeviceInventory,
    target: GpuTarget,
    device_uuid: str,
    expected_device_id: int | None = None,
) -> ObservedDevice:
    """Find one exact inventory-local UUID, retaining all target identity checks."""
    requested = parse_device_uuid(device_uuid)
    matches = [
        index
        for index, device in enumerate(inventory.devices)
        if device.device_uuid == requested
    ]
    if not matches:
        raise ValueError(
            f"Device UUID {requested} is unavailable for {target.canonical_id}"
        )
    if len(matches) != 1:
        # Parsed inventories already reject duplicates. Also fail closed for
        # a caller that constructs a DeviceInventory directly.
        raise ValueError("Ambiguous device UUID within one inventory")
    return select_device(inventory, target, matches[0], expected_device_id)
