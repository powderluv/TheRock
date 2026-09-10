# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Opt-in FP32 SGEMM operation and loaded-provider metadata.

This contract is separate from the fixed vector module ABI. Compiled capability
advertisement does not establish provider availability or hardware qualification.
Library versions are observed after lazy provider initialization in the worker.
"""

from dataclasses import dataclass
import hashlib
import json
import re
from typing import cast

from .gpu_targets import Vendor

SGEMM_ABI = "therock.blas.f32-sgemm-nn"
SGEMM_VERSION = 1
SGEMM_CAPABILITY = "blas-sgemm-f32-nn-v1"
SGEMM_MAX_DIMENSION = 256
SGEMM_MAX_CAPACITY = 65556
SGEMM_PROVIDER_CAPABILITIES = (
    "blas-provider-cublas-v1",
    "blas-provider-onemkl-v1",
    "blas-provider-rocblas-v1",
)
_PROVIDERS: dict[Vendor, str] = {
    "amd": "rocblas",
    "nvidia": "cublas",
    "intel": "onemkl",
}


def sgemm_provider_for_vendor(vendor: Vendor) -> str:
    if not isinstance(vendor, str) or vendor not in _PROVIDERS:
        raise ValueError(f"SGEMM is unsupported for vendor {vendor!r}")
    return _PROVIDERS[vendor]


def sgemm_capabilities(vendor: Vendor) -> tuple[str, ...]:
    provider = sgemm_provider_for_vendor(vendor)
    return (f"blas-provider-{provider}-v1", SGEMM_CAPABILITY)


def validate_sgemm_capabilities(vendor: Vendor, capabilities: tuple[str, ...]) -> None:
    extension = set(capabilities) & {
        SGEMM_CAPABILITY,
        *SGEMM_PROVIDER_CAPABILITIES,
    }
    if extension and extension != set(sgemm_capabilities(vendor)):
        raise ValueError(
            "SGEMM capability and matching vendor provider must appear together"
        )


def sgemm_contract_record() -> dict[str, object]:
    """Return a fresh record; this does not alter the vector-kernel contract."""
    return {
        "schema_version": 1,
        "kind": "blas-operation-contract",
        "abi": SGEMM_ABI,
        "version": SGEMM_VERSION,
        "operation": "sgemm",
        "dtype": "f32",
        "layout": "column-major",
        "transpose_a": False,
        "transpose_b": False,
        "expression": "C = alpha * A * B + beta * C",
        "dimensions": {"minimum": 1, "maximum": SGEMM_MAX_DIMENSION},
        "maximum_buffer_capacity": SGEMM_MAX_CAPACITY,
        "maximum_leading_dimension": SGEMM_MAX_CAPACITY,
        "offset_unit": "f32-elements",
        "leading_dimensions": {"a": "at-least-m", "b": "at-least-k", "c": "at-least-m"},
        "scalars": {"alpha": "finite-f32", "beta": "finite-f32"},
        "aliasing": {
            "output_input": "forbidden-by-buffer-handle",
            "input_input": "permitted",
        },
        "execution": {
            "queue": "worker-stream",
            "submission": "queued",
            "completion_operations": ["read", "synchronize"],
        },
    }


def sgemm_contract_sha256() -> str:
    canonical = json.dumps(
        sgemm_contract_record(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class SgemmProviderInfo:
    vendor: Vendor
    provider: str
    library_version: str
    abi: str
    version: int
    contract_sha256: str
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.provider != sgemm_provider_for_vendor(self.vendor):
            raise ValueError("SGEMM provider does not match its vendor")
        if (
            not isinstance(self.library_version, str)
            or not self.library_version.strip()
            or len(self.library_version.encode("utf-8")) > 4096
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in self.library_version
            )
        ):
            raise ValueError(
                "SGEMM library version must be nonempty text without control characters"
            )
        if (
            self.abi != SGEMM_ABI
            or type(self.version) is not int
            or self.version != SGEMM_VERSION
            or self.contract_sha256 != sgemm_contract_sha256()
        ):
            raise ValueError("Unsupported SGEMM operation contract")
        if (
            not isinstance(self.capabilities, tuple)
            or not all(isinstance(value, str) for value in self.capabilities)
            or tuple(sorted(self.capabilities)) != sgemm_capabilities(self.vendor)
        ):
            raise ValueError(
                "SGEMM provider capabilities do not match the operation and vendor"
            )
        object.__setattr__(self, "capabilities", tuple(sorted(self.capabilities)))

    def record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "blas-provider",
            "scope": "loaded-provider",
            "vendor": self.vendor,
            "provider": self.provider,
            "library_version": self.library_version,
            "abi": self.abi,
            "version": self.version,
            "contract_sha256": self.contract_sha256,
            "capabilities": list(self.capabilities),
        }


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate SGEMM provider JSON field: {key}")
        result[key] = value
    return result


def parse_sgemm_provider_json(text: str, *, vendor: Vendor) -> SgemmProviderInfo:
    try:
        raw = json.loads(text, object_pairs_hook=_unique_fields)
    except RecursionError as exc:
        raise ValueError("SGEMM provider JSON is nested too deeply") from exc
    fields = {
        "schema_version",
        "kind",
        "scope",
        "vendor",
        "provider",
        "library_version",
        "abi",
        "version",
        "contract_sha256",
        "capabilities",
    }
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError("Invalid SGEMM provider fields")
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["kind"] != "blas-provider"
        or raw["scope"] != "loaded-provider"
        or raw["vendor"] != vendor
    ):
        raise ValueError("SGEMM provider schema, scope, or vendor mismatch")
    for key in ("vendor", "provider", "library_version", "abi", "contract_sha256"):
        if not isinstance(raw[key], str):
            raise ValueError(f"SGEMM provider {key} must be a string")
    if not re.fullmatch(r"[0-9a-f]{64}", raw["contract_sha256"]):
        raise ValueError("Invalid SGEMM contract SHA256")
    if not isinstance(raw["capabilities"], list):
        raise ValueError("SGEMM provider capabilities must be a list")
    return SgemmProviderInfo(
        cast(Vendor, raw["vendor"]),
        raw["provider"],
        raw["library_version"],
        raw["abi"],
        raw["version"],
        raw["contract_sha256"],
        tuple(raw["capabilities"]),
    )
