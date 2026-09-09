# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Typed, exact-match catalogs for multi-vendor payloads in real KPAK v1 archives.

A catalog describes packs and logical modules, not runtime compatibility. Each
archive TOC uses "<logical-module>/<payload_type>" as its module key and the
canonical vendor:backend:processor[:features] identity as its exact target key.
This preserves the existing two-dimensional KPAK TOC while allowing multiple
formats of a module for one target. No AMD ISA compatibility matcher is used.

Schema 1 retains legacy entries without contracts. Schema 2 requires a logical
module contract and its digest on every entry, mirrored in archive metadata.
The contract describes native adapter arguments, not a packed vendor-neutral
wire block. Selecting an entry does not negotiate or implement that contract.

SHA256 checks provide integrity relative to the supplied catalog, not publisher
authentication. Catalogs contain no hardware qualification claims.
"""

import hashlib
import json
import os
import re
import struct
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from rocm_kpack.kpack import PackedKernelArchive

from .gpu_targets import PayloadType, parse_gpu_target
from .module_contract import ModuleContract, verify_contract


def _identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", value
    ):
        raise ValueError(f"Invalid {label}: {value!r}")


def _relative_path(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9_./-]+", value)
        or PurePosixPath(value).is_absolute()
        or str(PurePosixPath(value)) != value
        or any(part in (".", "..") for part in value.split("/"))
    ):
        raise ValueError(f"Invalid relative {label}: {value!r}")


def _digest(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid SHA256: {value!r}")


def _entry_fields(
    module: str, target: str, payload_type: PayloadType, entry_points: tuple[str, ...]
) -> None:
    _relative_path(module, "module")
    parsed = parse_gpu_target(target)
    if parsed.canonical_id != target:
        raise ValueError(f"Target must be canonical: {target!r}")
    if payload_type not in parsed.payload_types:
        raise ValueError(f"Payload type {payload_type!r} is not valid for {target!r}")
    if not isinstance(entry_points, tuple) or not entry_points:
        raise ValueError("Entry points must be a nonempty tuple")
    for symbol in entry_points:
        if not isinstance(symbol, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_.$]*", symbol
        ):
            raise ValueError(f"Invalid entry point: {symbol!r}")
    if len(set(entry_points)) != len(entry_points):
        raise ValueError("Duplicate entry points")


@dataclass(frozen=True)
class PayloadInput:
    module: str
    target: str
    payload_type: PayloadType
    entry_points: tuple[str, ...]
    data: bytes
    contract: ModuleContract | None = None

    def __post_init__(self) -> None:
        _entry_fields(self.module, self.target, self.payload_type, self.entry_points)
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("Payload must be nonempty bytes")
        if self.contract is not None and not isinstance(self.contract, ModuleContract):
            raise ValueError("Payload contract must be a ModuleContract")
        object.__setattr__(self, "entry_points", tuple(sorted(self.entry_points)))


@dataclass(frozen=True)
class PayloadEntry:
    module: str
    target: str
    payload_type: PayloadType
    entry_points: tuple[str, ...]
    sha256: str
    contract: ModuleContract | None = None

    def __post_init__(self) -> None:
        _entry_fields(self.module, self.target, self.payload_type, self.entry_points)
        _digest(self.sha256)
        if self.contract is not None and not isinstance(self.contract, ModuleContract):
            raise ValueError("Entry contract must be a ModuleContract")
        object.__setattr__(self, "entry_points", tuple(sorted(self.entry_points)))

    def record(self) -> dict[str, object]:
        result: dict[str, object] = {
            "module": self.module,
            "target": self.target,
            "payload_type": self.payload_type,
            "entry_points": list(self.entry_points),
            "sha256": self.sha256,
        }
        if self.contract is not None:
            result.update(
                contract=self.contract.record(), contract_sha256=self.contract.sha256
            )
        return result

    @property
    def key(self) -> tuple[str, str, PayloadType]:
        return self.module, self.target, self.payload_type

    @property
    def archive_module(self) -> str:
        return f"{self.module}/{self.payload_type}"


@dataclass(frozen=True)
class PayloadPack:
    pack_id: str
    path: str
    sha256: str
    entries: tuple[PayloadEntry, ...]

    def __post_init__(self) -> None:
        _identifier(self.pack_id, "pack ID")
        _relative_path(self.path, "pack path")
        if not self.path.endswith(".kpack"):
            raise ValueError("Pack path must end in .kpack")
        _digest(self.sha256)
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValueError("A pack must have a nonempty tuple of entries")
        if not all(isinstance(entry, PayloadEntry) for entry in self.entries):
            raise ValueError("Pack entries must be PayloadEntry values")
        if len({entry.key for entry in self.entries}) != len(self.entries):
            raise ValueError("Ambiguous duplicate payload entry")
        if len({entry.contract is not None for entry in self.entries}) != 1:
            raise ValueError(
                "Cannot mix entries with and without contracts in one pack"
            )


@dataclass(frozen=True)
class PayloadCatalog:
    packs: tuple[PayloadPack, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in (1, 2):
            raise ValueError(
                f"Unsupported catalog schema version: {self.schema_version!r}"
            )
        if not isinstance(self.packs, tuple) or not self.packs:
            raise ValueError("Catalog must have a nonempty tuple of packs")
        if not all(isinstance(pack, PayloadPack) for pack in self.packs):
            raise ValueError("Catalog packs must be PayloadPack values")
        if len({pack.pack_id for pack in self.packs}) != len(self.packs):
            raise ValueError("Duplicate pack ID")
        if len({pack.path for pack in self.packs}) != len(self.packs):
            raise ValueError("Duplicate pack path")
        keys = [entry.key for pack in self.packs for entry in pack.entries]
        if len(set(keys)) != len(keys):
            raise ValueError("Ambiguous duplicate payload entry")
        for pack in self.packs:
            if any(
                (entry.contract is not None) != (self.schema_version == 2)
                for entry in pack.entries
            ):
                raise ValueError(
                    "Catalog schema 2 requires contracts; schema 1 forbids them"
                )

    def record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "packs": [
                {
                    "pack_id": pack.pack_id,
                    "path": pack.path,
                    "sha256": pack.sha256,
                    "entries": [entry.record() for entry in pack.entries],
                }
                for pack in self.packs
            ],
        }


def _object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"Invalid {label} fields; expected {sorted(fields)}")
    return cast(dict[str, object], value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def load_catalog(path: Path) -> PayloadCatalog:
    """Parse strict schema 1/2 metadata; filesystem integrity is checked at extraction."""
    decoded: object = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object
    )
    raw = _object(decoded, {"schema_version", "packs"}, "catalog")
    if type(raw["schema_version"]) is not int or raw["schema_version"] not in (1, 2):
        raise ValueError(
            f"Unsupported catalog schema version: {raw['schema_version']!r}"
        )
    schema_version = cast(int, raw["schema_version"])
    packs: list[PayloadPack] = []
    for value in _list(raw["packs"], "packs"):
        pack = _object(value, {"pack_id", "path", "sha256", "entries"}, "pack")
        entries: list[PayloadEntry] = []
        for entry_value in _list(pack["entries"], "entries"):
            entry = _object(
                entry_value,
                {"module", "target", "payload_type", "entry_points", "sha256"}
                | ({"contract", "contract_sha256"} if schema_version == 2 else set()),
                "payload entry",
            )
            entries.append(
                PayloadEntry(
                    module=_string(entry["module"], "module"),
                    target=_string(entry["target"], "target"),
                    payload_type=cast(
                        PayloadType, _string(entry["payload_type"], "payload type")
                    ),
                    entry_points=tuple(
                        _string(symbol, "entry point")
                        for symbol in _list(entry["entry_points"], "entry points")
                    ),
                    sha256=_string(entry["sha256"], "payload SHA256"),
                    contract=(
                        verify_contract(
                            entry["contract"],
                            _string(entry["contract_sha256"], "contract SHA256"),
                        )
                        if schema_version == 2
                        else None
                    ),
                )
            )
        packs.append(
            PayloadPack(
                pack_id=_string(pack["pack_id"], "pack ID"),
                path=_string(pack["path"], "pack path"),
                sha256=_string(pack["sha256"], "pack SHA256"),
                entries=tuple(entries),
            )
        )
    return PayloadCatalog(tuple(packs), schema_version=schema_version)


def create_pack(output_dir: Path, pack_id: str, inputs: Iterable[PayloadInput]) -> Path:
    """Create one deterministic .kpack and catalog.json, returning the catalog path.

    The destination files must not exist. Input validation precedes writes; both
    files are staged before publishing, and failed publication rolls back files
    created by this call. Existing packs are never replaced in place.
    """
    _identifier(pack_id, "pack ID")
    selected = tuple(inputs)
    if not selected or not all(isinstance(item, PayloadInput) for item in selected):
        raise ValueError("Provide at least one PayloadInput")
    selected = tuple(
        sorted(selected, key=lambda item: (item.module, item.target, item.payload_type))
    )
    entries = tuple(
        PayloadEntry(
            item.module,
            item.target,
            item.payload_type,
            item.entry_points,
            hashlib.sha256(item.data).hexdigest(),
            item.contract,
        )
        for item in selected
    )
    # Validate duplicate keys before any filesystem writes.
    PayloadPack(pack_id, f"{pack_id}.kpack", "0" * 64, entries)
    archive = PackedKernelArchive(
        group_name=pack_id,
        gfx_arch_family="multi-vendor",
        gfx_arches=sorted({item.target for item in selected}),
    )
    for item, entry in zip(selected, entries):
        archive.add_kernel(
            archive.prepare_kernel(
                entry.archive_module,
                item.target,
                item.data,
                metadata={
                    "entry_points": list(entry.entry_points),
                    **(
                        {
                            "contract": entry.contract.record(),
                            "contract_sha256": entry.contract.sha256,
                        }
                        if entry.contract is not None
                        else {}
                    ),
                },
                payload_type=item.payload_type,
            )
        )
    archive.finalize_archive()

    destinations = (output_dir / f"{pack_id}.kpack", output_dir / "catalog.json")
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError(
            "Pack/catalog output already exists; choose a fresh output directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    with tempfile.TemporaryDirectory(
        prefix=".payload-stage-", dir=output_dir
    ) as staging:
        staged_pack = Path(staging) / destinations[0].name
        staged_catalog = Path(staging) / destinations[1].name
        archive.write(staged_pack)
        catalog = PayloadCatalog(
            (
                PayloadPack(
                    pack_id,
                    staged_pack.name,
                    hashlib.sha256(staged_pack.read_bytes()).hexdigest(),
                    entries,
                ),
            ),
            schema_version=2 if entries[0].contract is not None else 1,
        )
        staged_catalog.write_text(
            json.dumps(catalog.record(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            # Hard links publish complete files without replacing a concurrently
            # created destination. Staging and destination share a filesystem.
            for staged, destination in zip((staged_pack, staged_catalog), destinations):
                os.link(staged, destination)
                published.append(destination)
        except OSError:
            for destination in published:
                destination.unlink()
            raise
    return destinations[1]


def _contained_pack(catalog_path: Path, pack: PayloadPack) -> Path:
    root = catalog_path.parent.resolve(strict=True)
    resolved = (root / pack.path).resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(
            f"Pack path escapes catalog directory or is not a file: {pack.path}"
        )
    return resolved


@dataclass(frozen=True)
class VerifiedPayload:
    entry: PayloadEntry
    data: bytes


def extract_verified_payload(
    catalog_paths: Iterable[Path],
    module: str,
    target: str,
    payload_type: PayloadType,
    *,
    entry_point: str | None = None,
) -> VerifiedPayload:
    """Select a unique exact module/target/format across packs, verifying integrity.

    Every supplied pack hash is checked before selection. The selected archive is
    read from the bytes that were hashed, so a later change to its original path
    cannot replace the payload between verification and extraction.
    """
    canonical = parse_gpu_target(target).canonical_id
    _entry_fields(module, canonical, payload_type, (entry_point or "_selection",))
    requested = module, canonical, payload_type
    matches: list[tuple[PayloadPack, PayloadEntry, bytes]] = []
    pack_ids: set[str] = set()
    found_catalog = False
    for catalog_path in catalog_paths:
        found_catalog = True
        catalog = load_catalog(catalog_path)
        for pack in catalog.packs:
            if pack.pack_id in pack_ids:
                raise ValueError(f"Duplicate pack ID across catalogs: {pack.pack_id}")
            pack_ids.add(pack.pack_id)
            data = _contained_pack(catalog_path, pack).read_bytes()
            if hashlib.sha256(data).hexdigest() != pack.sha256:
                raise ValueError(f"Pack SHA256 mismatch: {pack.pack_id}")
            for entry in pack.entries:
                if entry.key == requested:
                    matches.append((pack, entry, data))
    if not found_catalog:
        raise ValueError("At least one catalog is required")
    if not matches:
        raise ValueError(f"No exact payload match for {requested!r}")
    if len(matches) != 1:
        raise ValueError(f"Ambiguous payload match for {requested!r}")
    pack, selected, data = matches[0]
    if entry_point is not None and entry_point not in selected.entry_points:
        raise ValueError(
            f"Entry point {entry_point!r} is not declared for {requested!r}"
        )
    try:
        with tempfile.TemporaryDirectory(prefix="therock-payload-read-") as temp_dir:
            snapshot = Path(temp_dir) / "verified.kpack"
            snapshot.write_bytes(data)
            archive = PackedKernelArchive.read(snapshot)
            if archive.gfx_arch_family != "multi-vendor":
                raise ValueError("Archive target namespace does not match catalog")
            if archive.group_name != pack.pack_id:
                raise ValueError("Archive group does not match catalog pack ID")
            expected_keys = {
                (entry.archive_module, entry.target) for entry in pack.entries
            }
            actual_keys = {
                (name, arch) for name, arches in archive.toc.items() for arch in arches
            }
            if actual_keys != expected_keys:
                raise ValueError("Archive TOC does not match catalog entries")
            if set(archive.gfx_arches) != {entry.target for entry in pack.entries}:
                raise ValueError("Archive target list does not match catalog")
            for entry in pack.entries:
                toc_entry = archive.toc[entry.archive_module][entry.target]
                if toc_entry["type"] != entry.payload_type:
                    raise ValueError("Archive payload type does not match catalog")
                metadata = toc_entry.get("metadata", {})
                if metadata.get("entry_points") != list(entry.entry_points):
                    raise ValueError("Archive entry points do not match catalog")
                if entry.contract is None:
                    if "contract" in metadata or "contract_sha256" in metadata:
                        raise ValueError(
                            "Schema 1 archive must not contain contract metadata"
                        )
                else:
                    metadata = _object(
                        metadata,
                        {"entry_points", "contract", "contract_sha256"},
                        "archive contract metadata",
                    )
                    archive_contract = verify_contract(
                        metadata["contract"],
                        _string(metadata["contract_sha256"], "archive contract SHA256"),
                    )
                    if archive_contract != entry.contract:
                        raise ValueError("Archive contract does not match catalog")
            payload = archive.get_kernel(selected.archive_module, selected.target)
            if (
                payload is None
                or hashlib.sha256(payload).hexdigest() != selected.sha256
            ):
                raise ValueError("Payload SHA256 mismatch")
            return VerifiedPayload(selected, payload)
    except (
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
        OSError,
        struct.error,
        OverflowError,
    ) as exc:
        raise ValueError(f"Invalid packed payload: {exc}") from exc


def extract_payload(
    catalog_paths: Iterable[Path],
    module: str,
    target: str,
    payload_type: PayloadType,
    *,
    entry_point: str | None = None,
) -> bytes:
    """Return verified payload bytes, preserving the schema-1 extraction API."""
    return extract_verified_payload(
        catalog_paths, module, target, payload_type, entry_point=entry_point
    ).data
