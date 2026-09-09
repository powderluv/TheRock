# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Strict distribution-relative registry of compiled native module runners.

The digest binds an executable to its declared compiled adapter description. It
is an integrity check, not a signature or a claim of hardware qualification.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast

from .gpu_targets import GpuTarget, parse_gpu_target
from .module_contract import (
    RunnerDescription,
    parse_runner_description,
    require_runner_compatibility,
    validation_contract,
)

_ENTRY_POINTS = ("therock_module_saxpy", "therock_module_relu")


def _relative_path(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(character) < 32 for character in value)
        or any(part in ("", ".", "..") for part in value.split("/"))
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).drive
    ):
        raise ValueError(
            f"Expected a normalized relative POSIX registry path: {value!r}"
        )
    return value


def _object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"Invalid {label} fields; expected {sorted(fields)}")
    return cast(dict[str, object], value)


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate runner registry JSON field: {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class RunnerEntry:
    target: GpuTarget
    path: str
    sha256: str
    description: RunnerDescription

    def __post_init__(self) -> None:
        if not isinstance(self.target, GpuTarget):
            raise ValueError("Runner target must be a GpuTarget")
        _relative_path(self.path)
        if not isinstance(self.sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.sha256
        ):
            raise ValueError("Runner SHA256 must be a lowercase SHA256 digest")
        require_runner_compatibility(
            self.description,
            self.target,
            self.target.payload_types,
            _ENTRY_POINTS,
            validation_contract(),
        )

    def record(self) -> dict[str, object]:
        return {
            "target": self.target.canonical_id,
            "path": self.path,
            "sha256": self.sha256,
            "description": self.description.record(),
        }


@dataclass(frozen=True)
class RunnerRegistry:
    runners: tuple[RunnerEntry, ...]
    catalogs: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.runners, tuple)
            or not self.runners
            or not all(isinstance(entry, RunnerEntry) for entry in self.runners)
        ):
            raise ValueError(
                "Registry runners must be a nonempty tuple of RunnerEntry values"
            )
        if not isinstance(self.catalogs, tuple) or not self.catalogs:
            raise ValueError("Registry catalogs must be a nonempty tuple")
        for path in self.catalogs:
            _relative_path(path)
        if len({entry.target.canonical_id for entry in self.runners}) != len(
            self.runners
        ):
            raise ValueError("Duplicate runner target")
        if len({entry.path for entry in self.runners}) != len(self.runners):
            raise ValueError("Duplicate runner path")
        if len(set(self.catalogs)) != len(self.catalogs):
            raise ValueError("Duplicate catalog path")
        object.__setattr__(
            self,
            "runners",
            tuple(sorted(self.runners, key=lambda entry: entry.target.canonical_id)),
        )

    def record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "native-runner-registry",
            "runners": [entry.record() for entry in self.runners],
            "catalogs": list(self.catalogs),
        }


def load_registry(path: Path) -> RunnerRegistry:
    """Load strict registry metadata without resolving or executing its entries."""
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_fields
        )
    except RecursionError as exc:
        raise ValueError("Runner registry JSON is nested too deeply") from exc
    raw = _object(
        document, {"schema_version", "kind", "runners", "catalogs"}, "runner registry"
    )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("Unsupported runner registry schema version")
    if raw["kind"] != "native-runner-registry":
        raise ValueError("Expected a native-runner-registry document")
    if not isinstance(raw["runners"], list) or not isinstance(raw["catalogs"], list):
        raise ValueError("Registry runners and catalogs must be lists")
    entries: list[RunnerEntry] = []
    for record in raw["runners"]:
        entry = _object(
            record, {"target", "path", "sha256", "description"}, "runner entry"
        )
        identity = entry["target"]
        if not isinstance(identity, str):
            raise ValueError("Runner target must be a canonical target string")
        target = parse_gpu_target(identity)
        if target.canonical_id != identity:
            raise ValueError("Runner target must be canonical")
        entries.append(
            RunnerEntry(
                target=target,
                path=_relative_path(entry["path"]),
                sha256=cast(str, entry["sha256"]),
                description=parse_runner_description(entry["description"]),
            )
        )
    return RunnerRegistry(
        tuple(entries), tuple(_relative_path(path) for path in raw["catalogs"])
    )


def resolve_registry_path(root: Path, relative: str) -> Path:
    """Resolve an existing regular file without allowing a symlink to escape root."""
    _relative_path(relative)
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError(f"Distribution root is not a directory: {root}")
    try:
        path = resolved_root.joinpath(*PurePosixPath(relative).parts).resolve(
            strict=True
        )
    except RuntimeError as exc:
        # Python 3.10-3.12 report symlink loops as RuntimeError, later versions
        # use OSError. Keep malformed registry paths actionable for either case.
        raise ValueError(f"Cannot resolve registry path: {relative!r}") from exc
    if not path.is_relative_to(resolved_root):
        raise ValueError(f"Registry path escapes the distribution root: {relative!r}")
    if not path.is_file():
        raise ValueError(f"Registry path is not a regular file: {relative!r}")
    return path


def verify_runner(root: Path, entry: RunnerEntry) -> Path:
    """Resolve and verify runner bytes before the caller queries or executes it."""
    if not isinstance(entry, RunnerEntry):
        raise ValueError("Expected a RunnerEntry")
    path = resolve_registry_path(root, entry.path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry.sha256:
        raise ValueError(
            f"Runner SHA256 mismatch for {entry.target.canonical_id}: {entry.path}"
        )
    return path
