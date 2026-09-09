# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Logical launch contracts for compiled module adapters.

Argument size/alignment describes each native adapter argument, not offsets in a
packed cross-vendor wire block. Parsing metadata does not establish adapter
support, runtime availability, or hardware qualification.
"""

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, cast

from .gpu_targets import Backend, GpuTarget, PayloadType, Vendor

ArgumentKind = Literal["device-pointer", "scalar"]
ValueType = Literal["f32", "f64", "u32", "u64", "i32", "i64"]
Access = Literal["read-only", "write-only", "read-write", "value"]

_VALUE_SIZES: dict[ValueType, int] = {
    "f32": 4,
    "f64": 8,
    "u32": 4,
    "u64": 8,
    "i32": 4,
    "i64": 8,
}
_BACKENDS: dict[Vendor, Backend] = {
    "amd": "hip",
    "nvidia": "cuda",
    "intel": "level-zero",
}
_FORMATS: dict[Vendor, tuple[PayloadType, ...]] = {
    "amd": ("hsaco",),
    "nvidia": ("cubin", "ptx"),
    "intel": ("spirv",),
}
_CAPABILITIES = (
    "device-allocation",
    "kernel-launch",
    "module-load",
    "ordered-copy",
    "queue-synchronize",
)
# Additional adapter behavior does not change the logical kernel contract.
# Callers can require this capability explicitly when qualifying event ordering.
_ADAPTER_CAPABILITIES = (
    *_CAPABILITIES,
    "cross-queue-events",
    "multi-module-session",
    "device-module-pipeline",
    "persistent-module-service",
)
_ENTRY_POINTS = ("therock_module_relu", "therock_module_saxpy")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"Invalid {label} fields; expected {sorted(fields)}")
    return cast(dict[str, object], value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 0xFFFFFFFF:
        raise ValueError(f"{label} must be an integer in [{minimum}, 4294967295]")
    return cast(int, value)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _token(value: str, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", value
    ):
        raise ValueError(f"Invalid {label}: {value!r}")


def _unique_strings(
    values: tuple[str, ...], label: str, *, symbols: bool = False
) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{label} must be a nonempty tuple")
    for value in values:
        if symbols:
            if not isinstance(value, str) or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_.$]*", value
            ):
                raise ValueError(f"Invalid {label}: {value!r}")
        else:
            _token(value, label)
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {label}")
    return tuple(sorted(values))


def _sha(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a lowercase SHA256 digest")
    return value


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class Argument:
    name: str
    kind: ArgumentKind
    value_type: ValueType
    access: Access
    size_bytes: int
    alignment_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.name
        ):
            raise ValueError(f"Invalid argument name: {self.name!r}")
        if (
            self.kind not in ("device-pointer", "scalar")
            or self.value_type not in _VALUE_SIZES
        ):
            raise ValueError("Unknown argument kind or value type")
        _integer(self.size_bytes, "argument size", minimum=1)
        _integer(self.alignment_bytes, "argument alignment", minimum=1)
        if (
            self.alignment_bytes & (self.alignment_bytes - 1)
            or self.alignment_bytes > self.size_bytes
        ):
            raise ValueError(
                "Argument alignment must be a power of two no larger than its size"
            )
        if self.kind == "scalar":
            if (
                self.access != "value"
                or self.size_bytes != _VALUE_SIZES[self.value_type]
            ):
                raise ValueError(
                    "Scalar arguments require value access and their type's size"
                )
        elif self.access not in (
            "read-only",
            "write-only",
            "read-write",
        ) or self.size_bytes not in (4, 8):
            raise ValueError(
                "Device pointers require device access and a 4- or 8-byte size"
            )

    def record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "value_type": self.value_type,
            "access": self.access,
            "size_bytes": self.size_bytes,
            "alignment_bytes": self.alignment_bytes,
        }


@dataclass(frozen=True)
class Launch:
    block: tuple[int, int, int]
    dynamic_shared_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.block, tuple) or len(self.block) != 3:
            raise ValueError("Launch block must contain exactly three dimensions")
        for dimension in self.block:
            _integer(dimension, "launch block dimension", minimum=1)
        _integer(self.dynamic_shared_bytes, "dynamic shared bytes")

    def record(self) -> dict[str, object]:
        return {
            "block": list(self.block),
            "dynamic_shared_bytes": self.dynamic_shared_bytes,
        }


@dataclass(frozen=True)
class Ownership:
    context: str
    queue: str
    device_allocations: str
    module_lifetime: str

    def __post_init__(self) -> None:
        for owner in (self.context, self.queue, self.device_allocations):
            if owner not in ("runner", "caller"):
                raise ValueError("Ownership must be runner or caller")
        _token(self.module_lifetime, "module lifetime")

    def record(self) -> dict[str, str]:
        return {
            "context": self.context,
            "queue": self.queue,
            "device_allocations": self.device_allocations,
            "module_lifetime": self.module_lifetime,
        }


@dataclass(frozen=True)
class Execution:
    ordering: str
    completion: str

    def __post_init__(self) -> None:
        _token(self.ordering, "execution ordering")
        _token(self.completion, "execution completion")

    def record(self) -> dict[str, str]:
        return {"ordering": self.ordering, "completion": self.completion}


@dataclass(frozen=True)
class ModuleContract:
    abi: str
    version: int
    pointer_bits: int
    arguments: tuple[Argument, ...]
    launch: Launch
    ownership: Ownership
    execution: Execution
    required_capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        _token(self.abi, "contract ABI")
        _integer(self.version, "contract version", minimum=1)
        if type(self.pointer_bits) is not int or self.pointer_bits not in (32, 64):
            raise ValueError("Pointer width must be 32 or 64 bits")
        if (
            not isinstance(self.arguments, tuple)
            or not self.arguments
            or not all(isinstance(argument, Argument) for argument in self.arguments)
        ):
            raise ValueError(
                "Contract arguments must be a nonempty tuple of Argument values"
            )
        if len({argument.name for argument in self.arguments}) != len(self.arguments):
            raise ValueError("Duplicate argument name")
        if any(
            argument.kind == "device-pointer"
            and argument.size_bytes * 8 != self.pointer_bits
            for argument in self.arguments
        ):
            raise ValueError("Device argument size does not match pointer width")
        if (
            not isinstance(self.launch, Launch)
            or not isinstance(self.ownership, Ownership)
            or not isinstance(self.execution, Execution)
        ):
            raise ValueError(
                "Contract launch, ownership, and execution must be typed values"
            )
        object.__setattr__(
            self,
            "required_capabilities",
            _unique_strings(self.required_capabilities, "required capability"),
        )

    def record(self) -> dict[str, object]:
        return {
            "abi": self.abi,
            "version": self.version,
            "pointer_bits": self.pointer_bits,
            "arguments": [argument.record() for argument in self.arguments],
            "launch": self.launch.record(),
            "ownership": self.ownership.record(),
            "execution": self.execution.record(),
            "required_capabilities": list(self.required_capabilities),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical(self.record()).encode("utf-8")).hexdigest()


def parse_contract(value: object) -> ModuleContract:
    raw = _object(
        value,
        {
            "abi",
            "version",
            "pointer_bits",
            "arguments",
            "launch",
            "ownership",
            "execution",
            "required_capabilities",
        },
        "module contract",
    )
    arguments: list[Argument] = []
    for value in _list(raw["arguments"], "arguments"):
        arg = _object(
            value,
            {"name", "kind", "value_type", "access", "size_bytes", "alignment_bytes"},
            "argument",
        )
        arguments.append(
            Argument(
                _string(arg["name"], "argument name"),
                cast(ArgumentKind, _string(arg["kind"], "argument kind")),
                cast(ValueType, _string(arg["value_type"], "value type")),
                cast(Access, _string(arg["access"], "argument access")),
                _integer(arg["size_bytes"], "argument size", minimum=1),
                _integer(arg["alignment_bytes"], "argument alignment", minimum=1),
            )
        )
    launch = _object(raw["launch"], {"block", "dynamic_shared_bytes"}, "launch")
    block = tuple(
        _integer(value, "block dimension", minimum=1)
        for value in _list(launch["block"], "block")
    )
    if len(block) != 3:
        raise ValueError("Launch block must contain exactly three dimensions")
    ownership = _object(
        raw["ownership"],
        {"context", "queue", "device_allocations", "module_lifetime"},
        "ownership",
    )
    execution = _object(raw["execution"], {"ordering", "completion"}, "execution")
    return ModuleContract(
        _string(raw["abi"], "contract ABI"),
        _integer(raw["version"], "contract version", minimum=1),
        _integer(raw["pointer_bits"], "pointer bits", minimum=1),
        tuple(arguments),
        Launch(
            cast(tuple[int, int, int], block),
            _integer(launch["dynamic_shared_bytes"], "dynamic shared bytes"),
        ),
        Ownership(
            *(
                _string(ownership[key], key)
                for key in ("context", "queue", "device_allocations", "module_lifetime")
            )
        ),
        Execution(
            _string(execution["ordering"], "ordering"),
            _string(execution["completion"], "completion"),
        ),
        tuple(
            _string(value, "capability")
            for value in _list(raw["required_capabilities"], "required capabilities")
        ),
    )


def verify_contract(record: object, sha256: str) -> ModuleContract:
    contract = parse_contract(record)
    if contract.sha256 != _sha(sha256):
        raise ValueError("Module contract SHA256 mismatch")
    return contract


def validation_contract() -> ModuleContract:
    return ModuleContract(
        "therock.validation.f32-vector",
        1,
        64,
        (
            Argument("x", "device-pointer", "f32", "read-only", 8, 8),
            Argument("y", "device-pointer", "f32", "read-only", 8, 8),
            Argument("output", "device-pointer", "f32", "write-only", 8, 8),
            Argument("alpha", "scalar", "f32", "value", 4, 4),
            Argument("count", "scalar", "u32", "value", 4, 4),
        ),
        Launch((128, 1, 1), 0),
        Ownership("runner", "runner", "runner", "until-queue-complete"),
        Execution("in-order-copy-launch-copy", "host-synchronized"),
        _CAPABILITIES,
    )


@dataclass(frozen=True)
class RunnerDescription:
    vendor: Vendor
    backend: Backend
    payload_types: tuple[PayloadType, ...]
    entry_points: tuple[str, ...]
    capabilities: tuple[str, ...]
    contract: ModuleContract

    def __post_init__(self) -> None:
        if self.vendor not in _BACKENDS or self.backend != _BACKENDS[self.vendor]:
            raise ValueError("Runner vendor/backend mismatch")
        formats = _unique_strings(self.payload_types, "payload format")
        if not set(formats) <= set(_FORMATS[self.vendor]):
            raise ValueError("Runner payload formats do not match its vendor")
        object.__setattr__(
            self, "payload_types", cast(tuple[PayloadType, ...], formats)
        )
        object.__setattr__(
            self,
            "entry_points",
            _unique_strings(self.entry_points, "entry point", symbols=True),
        )
        object.__setattr__(
            self, "capabilities", _unique_strings(self.capabilities, "capability")
        )
        if not isinstance(self.contract, ModuleContract):
            raise ValueError("Runner contract must be a ModuleContract")
        if not set(self.contract.required_capabilities) <= set(self.capabilities):
            raise ValueError("Runner lacks contract-required capabilities")

    @property
    def contract_sha256(self) -> str:
        return self.contract.sha256

    def record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "module-runner-contract",
            "scope": "compiled-adapter",
            "vendor": self.vendor,
            "backend": self.backend,
            "payload_types": list(self.payload_types),
            "entry_points": list(self.entry_points),
            "capabilities": list(self.capabilities),
            "contract": self.contract.record(),
            "contract_sha256": self.contract_sha256,
        }


def runner_description(vendor: Vendor) -> RunnerDescription:
    if vendor not in _BACKENDS:
        raise ValueError(f"Unknown runner vendor: {vendor!r}")
    return RunnerDescription(
        vendor,
        _BACKENDS[vendor],
        _FORMATS[vendor],
        _ENTRY_POINTS,
        _ADAPTER_CAPABILITIES,
        validation_contract(),
    )


def parse_runner_description(value: object) -> RunnerDescription:
    raw = _object(
        value,
        {
            "schema_version",
            "kind",
            "scope",
            "vendor",
            "backend",
            "payload_types",
            "entry_points",
            "capabilities",
            "contract",
            "contract_sha256",
        },
        "runner description",
    )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("Unsupported runner description schema version")
    if raw["kind"] != "module-runner-contract" or raw["scope"] != "compiled-adapter":
        raise ValueError("Runner description must have compiled-adapter scope")
    return RunnerDescription(
        cast(Vendor, _string(raw["vendor"], "vendor")),
        cast(Backend, _string(raw["backend"], "backend")),
        tuple(
            cast(PayloadType, _string(value, "payload type"))
            for value in _list(raw["payload_types"], "payload types")
        ),
        tuple(
            _string(value, "entry point")
            for value in _list(raw["entry_points"], "entry points")
        ),
        tuple(
            _string(value, "capability")
            for value in _list(raw["capabilities"], "capabilities")
        ),
        verify_contract(
            raw["contract"], _string(raw["contract_sha256"], "contract SHA256")
        ),
    )


def parse_runner_description_json(text: str) -> RunnerDescription:
    return parse_runner_description(
        json.loads(text, object_pairs_hook=_unique_json_object)
    )


def require_runner_compatibility(
    description: RunnerDescription,
    target: GpuTarget,
    payload_types: Iterable[str],
    entry_points: Iterable[str],
    contract: ModuleContract,
) -> None:
    """Require current fixture support; parsing alone permits future contracts."""
    if not isinstance(description, RunnerDescription) or not isinstance(
        target, GpuTarget
    ):
        raise ValueError("Expected a typed runner description and GPU target")
    if contract != validation_contract() or description.contract != contract:
        raise ValueError("Unsupported or mismatched module contract")
    if description.vendor != target.vendor or description.backend != target.backend:
        raise ValueError("Runner vendor/backend does not match selected target")
    formats = tuple(payload_types)
    symbols = tuple(entry_points)
    if (
        not formats
        or not all(isinstance(value, str) for value in formats)
        or not set(formats) <= set(description.payload_types)
    ):
        raise ValueError("Runner does not advertise all requested payload formats")
    if (
        not symbols
        or not all(isinstance(value, str) for value in symbols)
        or not set(symbols) <= set(description.entry_points)
    ):
        raise ValueError("Runner does not advertise all requested entry points")
    if not set(description.entry_points) <= set(_ENTRY_POINTS):
        raise ValueError("Runner advertises unsupported validation entry points")
    if not set(contract.required_capabilities) <= set(
        description.capabilities
    ) or not set(description.capabilities) <= set(_ADAPTER_CAPABILITIES):
        raise ValueError("Runner capabilities are unsupported or insufficient")


def validate_required_capabilities(values: Iterable[str]) -> tuple[str, ...]:
    """Validate explicit adapter requirements separately from the kernel ABI."""
    requested = tuple(values)
    if not all(
        isinstance(value, str) and value in _ADAPTER_CAPABILITIES for value in requested
    ):
        raise ValueError("Unknown required adapter capability")
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate required adapter capability")
    return tuple(sorted(requested))


def require_runner_capabilities(
    description: RunnerDescription, values: Iterable[str]
) -> None:
    requested = validate_required_capabilities(values)
    if not set(requested) <= set(description.capabilities):
        missing = sorted(set(requested) - set(description.capabilities))
        raise ValueError(
            f"Runner lacks required adapter capabilities: {', '.join(missing)}"
        )
