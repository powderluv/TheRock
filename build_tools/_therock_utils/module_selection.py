# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exact worker and packed-module selection for application sessions.

Prepared module selections retain bytes extracted from verified archives. Runner
paths remain paths rather than pinned executable inodes; distribution updates
must be serialized with use. Metadata hashes establish integrity, not publisher
identity or hardware qualification.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import re

from .device_inventory import (
    DeviceInventory,
    ObservedDevice,
    parse_device_uuid,
    parse_inventory_json,
    select_device,
    select_device_uuid,
)
from .gpu_targets import GpuTarget, PayloadType, parse_gpu_target
from .module_contract import (
    require_runner_capabilities,
    require_runner_compatibility,
    validate_required_capabilities,
    validation_contract,
)
from .payload_catalog import VerifiedPayload, extract_verified_payload
from .runner_query import describe_runner, query_runner
from .runner_registry import (
    RunnerEntry,
    RunnerRegistry,
    resolve_registry_path,
    verify_runner,
)

REGISTRY_PATH = "share/therock/packs/runners.json"


@dataclass(frozen=True)
class ModuleRequest:
    module: str
    payload_type: PayloadType
    entry_point: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str)
            for value in (self.module, self.payload_type, self.entry_point)
        ):
            raise ValueError(
                "Module requests require string module, format, and entry point"
            )


@dataclass(frozen=True)
class PreparedWorker:
    target: GpuTarget
    entry: RunnerEntry
    runner: Path
    device: ObservedDevice
    required_capabilities: tuple[str, ...]


@dataclass(frozen=True)
class _RegisteredWorker:
    target: GpuTarget
    entry: RunnerEntry
    runner: Path
    required_capabilities: tuple[str, ...]


@dataclass(frozen=True)
class PreparedModules:
    target: GpuTarget
    entry: RunnerEntry
    runner: Path
    device: ObservedDevice
    invocations: tuple[tuple[VerifiedPayload, str], ...]
    required_capabilities: tuple[str, ...]


def hardware_arguments(target: GpuTarget, expected_device_id: int | None) -> list[str]:
    """Build runtime checks without treating a SPIR-V format as hardware identity."""
    if target.vendor == "intel":
        if target.processor == "xe2-b70":
            if expected_device_id not in (None, 0xE223):
                raise ValueError("Intel xe2-b70 requires --expect-device-id 0xe223")
            expected_device_id = 0xE223
        elif expected_device_id is None:
            raise ValueError(
                f"Intel target {target.canonical_id} requires an explicit --expect-device-id"
            )
        if type(expected_device_id) is not int or not 0 < expected_device_id <= 0xFFFF:
            raise ValueError("Intel device ID must be a nonzero 16-bit integer")
        return ["--expect-arch", "spirv", "--expect-device-id", hex(expected_device_id)]
    if expected_device_id is not None:
        raise ValueError("--expect-device-id is only supported for Intel targets")
    expected_arch = (
        re.sub(r"[a-z]$", "", target.processor)
        if target.vendor == "nvidia"
        else target.processor
    )
    return ["--expect-arch", expected_arch]


def query_inventory(runner: Path, entry: RunnerEntry) -> DeviceInventory:
    description = describe_runner(runner)
    if description != entry.description:
        raise ValueError(
            "Compiled runner description differs from registry description"
        )
    inventory = parse_inventory_json(query_runner(runner, "--list-devices"))
    if (
        inventory.vendor != entry.target.vendor
        or inventory.backend != entry.target.backend
    ):
        raise ValueError("Device inventory vendor/backend differs from registry target")
    return inventory


def _register_worker(
    root: Path,
    registry: RunnerRegistry,
    *,
    target_id: str,
    device_index: int | None,
    device_uuid: str | None,
    expected_device_id: int | None,
    required_capabilities: tuple[str, ...],
) -> _RegisteredWorker:
    """Validate selection and capabilities before executing a registered runner."""
    if not isinstance(target_id, str):
        raise ValueError("Expected a canonical GPU target string")
    target = parse_gpu_target(target_id)
    required_capabilities = validate_required_capabilities(required_capabilities)
    if device_index is not None and device_uuid is not None:
        raise ValueError("--device and --device-uuid are mutually exclusive")
    if device_index is not None and (type(device_index) is not int or device_index < 0):
        raise ValueError("Device index must be nonnegative")
    if device_uuid is not None:
        parse_device_uuid(device_uuid)
    hardware_arguments(target, expected_device_id)
    if target.features or (
        target.vendor == "nvidia" and target.processor[-1].isalpha()
    ):
        raise ValueError("Discovery cannot qualify architecture feature suffixes")
    matches = [entry for entry in registry.runners if entry.target == target]
    if len(matches) != 1:
        raise ValueError(f"No exact registered runner for {target.canonical_id}")
    entry = matches[0]
    require_runner_capabilities(entry.description, required_capabilities)
    runner = verify_runner(root, entry)
    return _RegisteredWorker(target, entry, runner, required_capabilities)


def _discover_worker(
    root: Path,
    registered: _RegisteredWorker,
    *,
    device_index: int | None,
    device_uuid: str | None,
    expected_device_id: int | None,
) -> PreparedWorker:
    target, entry, runner = registered.target, registered.entry, registered.runner
    inventory = query_inventory(runner, entry)
    if device_uuid is not None:
        device = select_device_uuid(inventory, target, device_uuid, expected_device_id)
    else:
        device = select_device(
            inventory,
            target,
            device_index if device_index is not None else 0,
            expected_device_id,
        )
    verify_runner(root, entry)
    return PreparedWorker(
        target, entry, runner, device, registered.required_capabilities
    )


def prepare_worker(
    root: Path,
    registry: RunnerRegistry,
    *,
    target_id: str,
    device_index: int | None = None,
    device_uuid: str | None = None,
    expected_device_id: int | None = None,
    required_capabilities: tuple[str, ...] = (),
) -> PreparedWorker:
    """Verify an exact worker and observed device without inspecting any packs.

    Registry metadata is still required and validated. Catalog paths are neither
    resolved nor opened, and no payload snapshots or execution resources exist.
    Callers must reverify runner bytes immediately before starting their worker.
    """
    registered = _register_worker(
        root,
        registry,
        target_id=target_id,
        device_index=device_index,
        device_uuid=device_uuid,
        expected_device_id=expected_device_id,
        required_capabilities=required_capabilities,
    )
    return _discover_worker(
        root,
        registered,
        device_index=device_index,
        device_uuid=device_uuid,
        expected_device_id=expected_device_id,
    )


def prepare_modules(
    root: Path,
    registry: RunnerRegistry,
    *,
    requests: tuple[ModuleRequest, ...],
    target_id: str,
    device_index: int | None = None,
    device_uuid: str | None = None,
    expected_device_id: int | None = None,
    required_capabilities: tuple[str, ...] = (),
) -> PreparedModules:
    """Verify every payload before discovery, preserving ordered repeated stages.

    Callers impose their own duplicate/stage-count policy. This common operation
    accepts at most 32 requests and returns no process-owned execution resources.
    """
    if not isinstance(requests, tuple) or not 1 <= len(requests) <= 32:
        raise ValueError("A module session requires between 1 and 32 requests")
    if not all(isinstance(request, ModuleRequest) for request in requests):
        raise ValueError("Expected typed ModuleRequest values")
    registered = _register_worker(
        root,
        registry,
        target_id=target_id,
        device_index=device_index,
        device_uuid=device_uuid,
        expected_device_id=expected_device_id,
        required_capabilities=required_capabilities,
    )
    target, entry = registered.target, registered.entry
    catalogs = [resolve_registry_path(root, path) for path in registry.catalogs]
    invocations = []
    snapshots: dict[ModuleRequest, VerifiedPayload] = {}
    for request in requests:
        if request not in snapshots:
            snapshots[request] = extract_verified_payload(
                catalogs,
                request.module,
                target.canonical_id,
                request.payload_type,
                entry_point=request.entry_point,
            )
        selected = snapshots[request]
        contract = selected.entry.contract
        if contract is None:
            raise ValueError("Guarded validation requires a schema 2 launch contract")
        if contract != validation_contract():
            raise ValueError("Unsupported validation launch contract")
        require_runner_compatibility(
            entry.description,
            target,
            (request.payload_type,),
            (request.entry_point,),
            contract,
        )
        invocations.append((selected, request.entry_point))
    worker = _discover_worker(
        root,
        registered,
        device_index=device_index,
        device_uuid=device_uuid,
        expected_device_id=expected_device_id,
    )
    device = worker.device
    contract = validation_contract()
    if math.prod(contract.launch.block) > device.limits.max_threads_per_block or any(
        required > available
        for required, available in zip(
            contract.launch.block, device.limits.max_block_dimensions
        )
    ):
        raise ValueError(
            "Observed device limits do not support contract launch geometry"
        )
    return PreparedModules(
        worker.target,
        worker.entry,
        worker.runner,
        worker.device,
        tuple(invocations),
        worker.required_capabilities,
    )
