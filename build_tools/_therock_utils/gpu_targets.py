# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Typed target identities for the experimental multi-vendor build driver.

These descriptors identify requested build targets, not a registry of supported
hardware. Parsing a target does not establish compiler support or hardware
qualification. In particular, an Intel product identifier is not a compiler flag:
the chipStar toolchain selects its SPIR-V target independently.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, cast

Vendor = Literal["amd", "nvidia", "intel"]
Backend = Literal["hip", "cuda", "level-zero"]
FeatureName = Literal["xnack", "sramecc"]
PayloadType = Literal["hsaco", "cubin", "ptx", "spirv"]
HipPlatform = Literal["amd", "nvidia", "spirv"]

_BACKENDS: dict[Vendor, Backend] = {
    "amd": "hip",
    "nvidia": "cuda",
    "intel": "level-zero",
}
_PROCESSOR_PATTERNS: dict[Vendor, re.Pattern[str]] = {
    "amd": re.compile(r"gfx[0-9a-f]{3,}"),
    "nvidia": re.compile(r"sm_[1-9][0-9]{1,2}[a-z]?"),
    "intel": re.compile(r"xe[1-9][0-9]*-[a-z][a-z0-9]*(?:-[a-z0-9]+)*"),
}


@dataclass(frozen=True)
class TargetFeature:
    name: FeatureName
    enabled: bool

    def __post_init__(self) -> None:
        if self.name not in ("xnack", "sramecc"):
            raise ValueError(f"Unknown AMD target feature: {self.name!r}")
        if not isinstance(self.enabled, bool):
            raise ValueError("Target feature enabled must be a bool")

    @property
    def target_id_suffix(self) -> str:
        return self.name + ("+" if self.enabled else "-")


@dataclass(frozen=True)
class GpuTarget:
    vendor: Vendor
    backend: Backend
    processor: str
    features: tuple[TargetFeature, ...] = ()

    def __post_init__(self) -> None:
        if self.vendor not in _BACKENDS:
            raise ValueError(f"Unknown GPU vendor: {self.vendor!r}")
        if self.backend != _BACKENDS[self.vendor]:
            raise ValueError(
                f"Vendor {self.vendor!r} requires backend {_BACKENDS[self.vendor]!r}, "
                f"not {self.backend!r}"
            )
        if not _PROCESSOR_PATTERNS[self.vendor].fullmatch(self.processor):
            raise ValueError(
                f"Invalid {self.vendor} processor identifier: {self.processor!r}"
            )
        if not isinstance(self.features, tuple) or not all(
            isinstance(feature, TargetFeature) for feature in self.features
        ):
            raise ValueError("Target features must be a tuple of TargetFeature values")
        if self.features and self.vendor != "amd":
            raise ValueError("Target feature suffixes are only supported for AMD")
        seen: set[FeatureName] = set()
        for feature in self.features:
            if feature.name in seen:
                raise ValueError(f"Duplicate or conflicting feature: {feature.name}")
            seen.add(feature.name)
        object.__setattr__(
            self,
            "features",
            tuple(sorted(self.features, key=lambda feature: feature.name)),
        )

    @property
    def canonical_id(self) -> str:
        return ":".join(
            (self.vendor, self.backend, self.processor)
            + tuple(feature.target_id_suffix for feature in self.features)
        )

    @property
    def slug(self) -> str:
        """Filesystem-safe, collision-free encoding for the accepted grammar.

        Artifact names reserve underscores as field separators. The accepted
        grammar fixes each vendor's backend and processor prefix, so removing
        NVIDIA's single underscore and spelling out AMD feature signs preserves
        distinct identities. Consumers must store canonical_id as the identity
        rather than parse this slug.
        """
        return "-".join(
            (self.vendor, self.backend, self.processor.replace("_", ""))
            + tuple(
                feature.name + ("-on" if feature.enabled else "-off")
                for feature in self.features
            )
        )

    @property
    def cmake_hip_platform(self) -> HipPlatform:
        if self.vendor == "amd":
            return "amd"
        if self.vendor == "nvidia":
            return "nvidia"
        return "spirv"

    @property
    def cmake_hip_architecture(self) -> str | None:
        if self.vendor == "amd":
            return ":".join(
                (self.processor,)
                + tuple(feature.target_id_suffix for feature in self.features)
            )
        if self.vendor == "nvidia":
            return self.processor.removeprefix("sm_")
        # The toolchain, not an Intel product name, owns the SPIR-V target.
        return None

    @property
    def compiler_target(self) -> str | None:
        if self.vendor == "amd":
            return self.cmake_hip_architecture
        if self.vendor == "nvidia":
            return self.processor
        return None

    @property
    def payload_types(self) -> tuple[PayloadType, ...]:
        """Possible backend formats, not a claim about emitted build contents."""
        if self.vendor == "amd":
            return ("hsaco",)
        if self.vendor == "nvidia":
            return ("cubin", "ptx")
        return ("spirv",)

    @property
    def experimental(self) -> bool:
        """Whether this backend is an experimental extension to TheRock.

        Hardware validation is recorded separately by build/test results. Even
        a non-experimental backend does not qualify every parsed processor.
        """
        return self.vendor != "amd"


def parse_gpu_target(value: str) -> GpuTarget:
    """Parse vendor:backend:processor[:feature+|-...] without guessing a vendor.

    AMD feature order is normalized in the resulting canonical identity. No
    whitespace, shell punctuation, paths, or implicit processor aliases are
    accepted.
    """
    components = value.split(":")
    if len(components) < 3:
        raise ValueError(
            f"Expected vendor:backend:processor[:feature+|-...], got {value!r}"
        )
    vendor, backend, processor, *suffixes = components
    features: list[TargetFeature] = []
    for suffix in suffixes:
        match = re.fullmatch(r"(xnack|sramecc)([+-])", suffix)
        if match is None:
            raise ValueError(f"Invalid target feature suffix: {suffix!r}")
        features.append(
            TargetFeature(cast(FeatureName, match.group(1)), match.group(2) == "+")
        )
    # The dataclass validates these strings before exposing the typed fields.
    return GpuTarget(
        cast(Vendor, vendor), cast(Backend, backend), processor, tuple(features)
    )


def parse_gpu_targets(values: Iterable[str]) -> tuple[GpuTarget, ...]:
    """Parse targets in input order, rejecting duplicate canonical identities."""
    targets: list[GpuTarget] = []
    identities: set[str] = set()
    for value in values:
        target = parse_gpu_target(value)
        if target.canonical_id in identities:
            raise ValueError(f"Duplicate GPU target: {target.canonical_id}")
        identities.add(target.canonical_id)
        targets.append(target)
    return tuple(targets)
