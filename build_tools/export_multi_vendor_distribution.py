# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Export whole target subsets of a built multi-vendor module distribution.

Export requires Linux renameat2(RENAME_NOREPLACE) and a supporting filesystem;
there is deliberately no overwriting publication fallback. The destination's
parent must already exist. Verification and repacking use frozen input bytes,
never run a native worker, and do not resolve the original SDK input paths.

Digests provide integrity relative to supplied metadata, not authentication,
complete dependency closure, hardware qualification, or a transactional snapshot
of a concurrently modified source tree. Original build receipts remain unchanged.
"""

import argparse
import ctypes
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast

from _therock_utils.gpu_targets import PayloadType, parse_gpu_target
from _therock_utils.module_contract import (
    parse_runner_description_json,
    require_runner_compatibility,
    validation_contract,
)
from _therock_utils.payload_catalog import (
    PayloadCatalog,
    PayloadInput,
    PayloadPack,
    create_pack,
    extract_verified_payload,
    load_catalog,
)
from _therock_utils.runner_registry import RunnerRegistry, load_registry

REGISTRY = "share/therock/packs/runners.json"
PROVENANCE = "share/therock/packs/input-provenance.json"
RUNTIME_MANIFEST = "share/therock/python/runtime-manifest.json"
MANIFEST = "share/therock/distribution-manifest.json"
_RUNTIME_PREFIX = "share/therock/python/"
_EXAMPLE = "share/therock/examples/packed_session_client.py"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _decode(data: bytes) -> object:
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_unique_fields)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError(f"Invalid UTF-8 JSON: {exc}") from exc


def _object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"Invalid {label} fields; expected {sorted(fields)}")
    return cast(dict[str, object], value)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _digest(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a lowercase SHA256 digest")
    return value


def _relative(value: object) -> str:
    path = _string(value, "relative path")
    if (
        "\\" in path
        or any(ord(character) < 32 for character in path)
        or PurePosixPath(path).is_absolute()
        or PureWindowsPath(path).drive
        or any(part in ("", ".", "..") for part in path.split("/"))
    ):
        raise ValueError(f"Expected a normalized relative POSIX path: {path!r}")
    return path


def _targets(values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError("Select a nonempty tuple of canonical target IDs")
    for identity in values:
        if not isinstance(identity, str):
            raise ValueError("Target IDs must be strings")
        if parse_gpu_target(identity).canonical_id != identity:
            raise ValueError(f"Target ID is not canonical: {identity!r}")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate selected target")
    return tuple(sorted(values))


def _absolute_directory(path: Path) -> Path:
    # abspath removes benign CLI '..' components without following symlinks.
    absolute = Path(os.path.abspath(path))
    for ancestor in (*reversed(absolute.parents), absolute):
        info = ancestor.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(
                f"Directory path contains a symlink or non-directory: {ancestor}"
            )
    return absolute


def _fingerprint(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


@dataclass(frozen=True)
class _File:
    data: bytes
    mode: int

    def record(self, path: str) -> dict[str, object]:
        return {
            "path": path,
            "sha256": _sha(self.data),
            "size": len(self.data),
            "mode": self.mode,
        }


def _read_file(path: Path) -> _File:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) & ~0o777:
        raise ValueError(f"Expected a regular file without special mode bits: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if _fingerprint(before) != _fingerprint(opened):
            raise ValueError(f"Source changed while opening: {path}")
        data = stream.read()
        after = os.fstat(stream.fileno())
    if (
        _fingerprint(before) != _fingerprint(after)
        or _fingerprint(before) != _fingerprint(path.lstat())
        or len(data) != before.st_size
    ):
        raise ValueError(f"Source changed while reading: {path}")
    return _File(data, stat.S_IMODE(before.st_mode))


def _walk_error(error: OSError) -> None:
    raise error


def _snapshot(root: Path) -> dict[str, _File]:
    _absolute_directory(root)
    result: dict[str, _File] = {}
    for directory, directories, filenames in os.walk(
        root, followlinks=False, onerror=_walk_error
    ):
        for name in directories:
            path = Path(directory) / name
            if not stat.S_ISDIR(path.lstat().st_mode):
                raise ValueError(
                    f"Distribution contains a symlink or special directory: {path}"
                )
        for name in filenames:
            path = Path(directory) / name
            relative = _relative(path.relative_to(root).as_posix())
            result[relative] = _read_file(path)
    return result


def _file(files: dict[str, _File], relative: str) -> _File:
    try:
        return files[_relative(relative)]
    except KeyError as exc:
        raise ValueError(f"Missing distribution file: {relative}") from exc


def _write_files(root: Path, files: dict[str, _File]) -> None:
    for relative, item in sorted(files.items()):
        path = root / _relative(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(item.data)
        path.chmod(item.mode)


def _claim(paths: set[str], relative: str) -> None:
    _relative(relative)
    if relative in paths or any(
        relative.startswith(existing + "/") or existing.startswith(relative + "/")
        for existing in paths
    ):
        raise ValueError(f"Overlapping distribution file paths: {relative}")
    paths.add(relative)


def _self_hashed(raw: dict[str, object]) -> None:
    expected = _digest(raw["content_sha256"])
    body = {key: value for key, value in raw.items() if key != "content_sha256"}
    if _sha(_json_bytes(body)) != expected:
        raise ValueError("Manifest content SHA256 mismatch")


def _file_records(value: object) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    paths: set[str] = set()
    for value in _list(value, "file records"):
        record = _object(value, {"path", "sha256", "size", "mode"}, "file record")
        path = _relative(record["path"])
        _claim(paths, path)
        _digest(record["sha256"])
        _integer(record["size"], "file size")
        if _integer(record["mode"], "file mode") > 0o777:
            raise ValueError("File mode contains special bits")
        result[path] = record
    return result


def _check_records(
    files: dict[str, _File], records: dict[str, dict[str, object]]
) -> None:
    for path, record in records.items():
        if _file(files, path).record(path) != record:
            raise ValueError(f"File hash, size, or mode mismatch: {path}")


def _runtime_files(files: dict[str, _File]) -> set[str]:
    raw = _object(
        _decode(_file(files, RUNTIME_MANIFEST).data),
        {
            "schema_version",
            "kind",
            "coverage",
            "kpack_scope",
            "packaging",
            "sources",
            "outputs",
            "content_sha256",
        },
        "runtime manifest",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["kind"] != "multi-vendor-python-runtime"
        or raw["coverage"] != "selected-runtime-sources"
        or raw["kpack_scope"] != "archive-reader-subset"
    ):
        raise ValueError("Unsupported runtime manifest")
    _self_hashed(raw)
    packaging = _object(
        raw["packaging"], {"helper_sha256", "source_map_sha256"}, "runtime packaging"
    )
    for value in packaging.values():
        _digest(value)
    records = _file_records(raw["outputs"])
    for path, record in records.items():
        if (
            not (path.startswith(_RUNTIME_PREFIX) or path == _EXAMPLE)
            or path == RUNTIME_MANIFEST
            or record["mode"] != 0o644
        ):
            raise ValueError(f"Unexpected runtime output: {path}")
    destinations: set[str] = set()
    sources: set[tuple[str, str]] = set()
    for value in _list(raw["sources"], "runtime sources"):
        source = _object(
            value, {"root", "path", "destination", "sha256"}, "runtime source"
        )
        root = _string(source["root"], "runtime source root")
        if root not in ("tools", "repository", "kpack"):
            raise ValueError("Unsupported runtime source root")
        path = _relative(source["path"])
        destination = _relative(source["destination"])
        if (root, path) in sources or destination in destinations:
            raise ValueError("Duplicate runtime source or destination")
        sources.add((root, path))
        destinations.add(destination)
        if (
            destination not in records
            or _digest(source["sha256"]) != records[destination]["sha256"]
        ):
            raise ValueError("Runtime source/output digest mismatch")
    if destinations != set(records):
        raise ValueError("Runtime source/output inventory mismatch")
    _check_records(files, records)
    expected = set(records) | {RUNTIME_MANIFEST}
    actual = {
        path for path in files if path.startswith(_RUNTIME_PREFIX) or path == _EXAMPLE
    }
    if actual != expected or _file(files, RUNTIME_MANIFEST).mode != 0o644:
        raise ValueError("Runtime inventory or manifest mode mismatch")
    return expected


def _validate_provenance(data: bytes) -> None:
    raw = _object(
        _decode(data),
        {
            "schema_version",
            "kind",
            "coverage",
            "cache_eligible",
            "global_content_sha256",
            "inputs",
        },
        "input provenance",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["kind"] != "multi-vendor-input-provenance"
        or raw["coverage"] != "declared-inputs"
        or raw["cache_eligible"] is not False
    ):
        raise ValueError("Unsupported original input provenance")
    _digest(raw["global_content_sha256"])
    names: set[str] = set()
    identities: list[dict[str, object]] = []
    for value in _list(raw["inputs"], "provenance inputs"):
        item = _object(
            value,
            {
                "name",
                "kind",
                "current_path",
                "resolved_path",
                "content_sha256",
                "counts",
            },
            "provenance input",
        )
        name = _string(item["name"], "input name")
        if (
            name in names
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
            or item["kind"] not in ("file", "tree")
        ):
            raise ValueError("Duplicate input name or invalid input kind")
        names.add(name)
        _string(item["current_path"], "original input path")
        _string(item["resolved_path"], "original resolved input path")
        _digest(item["content_sha256"])
        identities.append(
            {key: item[key] for key in ("name", "kind", "content_sha256")}
        )
        counts = _object(
            item["counts"],
            {"bytes", "directories", "files", "symlinks"},
            "input counts",
        )
        for value in counts.values():
            _integer(value, "input count")
    if [item["name"] for item in identities] != sorted(names):
        raise ValueError("Original input names must be sorted")
    # Match InputLock.global_content_sha256 without accessing its original inputs.
    canonical = json.dumps(
        identities, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    if _sha(canonical) != raw["global_content_sha256"]:
        raise ValueError("Original input provenance aggregate SHA256 mismatch")


@dataclass(frozen=True)
class _Distribution:
    registry: RunnerRegistry
    catalogs: dict[str, PayloadCatalog]
    payloads: dict[tuple[str, str, PayloadType], PayloadInput]
    required_files: set[str]
    source_catalogs: list[dict[str, object]]


def _archive_path(catalog_path: str, pack_path: str) -> str:
    return _relative(
        (PurePosixPath(catalog_path).parent / _relative(pack_path)).as_posix()
    )


def _validate_contents(
    files: dict[str, _File], frozen: Path, selected: tuple[str, ...]
) -> _Distribution:
    registry = load_registry(frozen / REGISTRY)
    runners = {entry.target.canonical_id: entry for entry in registry.runners}
    if not set(selected) <= runners.keys():
        raise ValueError(
            "Unknown selected target; exact registered targets are required"
        )
    paths: set[str] = set()
    for path in (REGISTRY, PROVENANCE, MANIFEST):
        _claim(paths, path)
    required = {REGISTRY, PROVENANCE}
    provenance = _file(files, PROVENANCE).data
    _validate_provenance(provenance)
    for path in _runtime_files(files):
        _claim(paths, path)
        required.add(path)
    for identity, runner in runners.items():
        native_paths = (
            runner.path,
            PurePosixPath(runner.path).with_name("runner-contract.json").as_posix(),
            PurePosixPath(runner.path).with_name("input-build-receipt.json").as_posix(),
        )
        for path in native_paths:
            _claim(paths, path)
        if identity not in selected:
            continue
        executable = _file(files, runner.path)
        if (
            not executable.data
            or not executable.mode & 0o111
            or _sha(executable.data) != runner.sha256
        ):
            raise ValueError(f"Runner hash or executable mode mismatch: {runner.path}")
        if (
            parse_runner_description_json(
                _file(files, native_paths[1]).data.decode("utf-8")
            )
            != runner.description
        ):
            raise ValueError(f"Runner contract differs from registry: {identity}")
        if _file(files, native_paths[2]).data != provenance:
            raise ValueError(
                f"Native input receipt differs from original pack provenance: {identity}"
            )
        required.update(native_paths)
    catalogs: dict[str, PayloadCatalog] = {}
    source_catalogs: list[dict[str, object]] = []
    keys: set[tuple[str, str, PayloadType]] = set()
    pack_ids: set[str] = set()
    module_symbols: dict[str, tuple[str, ...]] = {}
    for relative in sorted(registry.catalogs):
        _claim(paths, relative)
        required.add(relative)
        catalog = load_catalog(frozen / relative)
        if catalog.schema_version != 2:
            raise ValueError("Selective export requires schema 2 payload catalogs")
        catalogs[relative] = catalog
        archives: list[dict[str, object]] = []
        for pack in catalog.packs:
            if pack.pack_id in pack_ids:
                raise ValueError(f"Duplicate pack ID across catalogs: {pack.pack_id}")
            pack_ids.add(pack.pack_id)
            archive_path = _archive_path(relative, pack.path)
            _claim(paths, archive_path)
            required.add(archive_path)
            if _sha(_file(files, archive_path).data) != pack.sha256:
                raise ValueError(f"Pack SHA256 mismatch: {archive_path}")
            archives.append(
                {"pack_id": pack.pack_id, "path": archive_path, "sha256": pack.sha256}
            )
            for entry in pack.entries:
                if entry.key in keys:
                    raise ValueError(f"Ambiguous duplicate payload entry: {entry.key}")
                keys.add(entry.key)
                if (
                    entry.target not in runners
                    or entry.contract != validation_contract()
                ):
                    raise ValueError(
                        "Payload requires an unregistered target or incompatible contract"
                    )
                runner = runners[entry.target]
                require_runner_compatibility(
                    runner.description,
                    runner.target,
                    (entry.payload_type,),
                    entry.entry_points,
                    entry.contract,
                )
                if (
                    module_symbols.setdefault(entry.module, entry.entry_points)
                    != entry.entry_points
                ):
                    raise ValueError(
                        f"Inconsistent entry points for logical module: {entry.module}"
                    )
        source_catalogs.append(
            {
                "path": relative,
                "sha256": _sha(_file(files, relative).data),
                "archives": sorted(archives, key=lambda item: cast(str, item["path"])),
            }
        )
    expected = {
        (module, identity, fmt)
        for module in module_symbols
        for identity in selected
        for fmt in runners[identity].description.payload_types
    }
    if {key for key in keys if key[1] in selected} != expected:
        raise ValueError("Incomplete selected logical-module/target/format coverage")
    payloads: dict[tuple[str, str, PayloadType], PayloadInput] = {}
    catalog_paths = tuple(frozen / relative for relative in catalogs)
    # Validate every referenced archive's contents, including unselected targets.
    # Extract from private copies of exactly the bytes already hashed above.
    for module, identity, fmt in sorted(keys):
        verified = extract_verified_payload(catalog_paths, module, identity, fmt)
        entry = verified.entry
        if identity in selected:
            payloads[entry.key] = PayloadInput(
                module, identity, fmt, entry.entry_points, verified.data, entry.contract
            )
    return _Distribution(registry, catalogs, payloads, required, source_catalogs)


def _source_record(
    files: dict[str, _File], contents: _Distribution
) -> dict[str, object]:
    return {
        "registry": {"path": REGISTRY, "sha256": _sha(_file(files, REGISTRY).data)},
        "catalogs": contents.source_catalogs,
        "input_provenance_sha256": _sha(_file(files, PROVENANCE).data),
        "runtime_manifest_sha256": _sha(_file(files, RUNTIME_MANIFEST).data),
    }


def _validate_source_record(
    value: object, files: dict[str, _File], contents: _Distribution
) -> None:
    source = _object(
        value,
        {"registry", "catalogs", "input_provenance_sha256", "runtime_manifest_sha256"},
        "derivation source",
    )
    registry = _object(source["registry"], {"path", "sha256"}, "source registry")
    if registry["path"] != REGISTRY:
        raise ValueError("Unexpected source registry path")
    _digest(registry["sha256"])
    for key, path in (
        ("input_provenance_sha256", PROVENANCE),
        ("runtime_manifest_sha256", RUNTIME_MANIFEST),
    ):
        if _digest(source[key]) != _sha(_file(files, path).data):
            raise ValueError(f"Copied source digest mismatch: {path}")
    paths = {REGISTRY, PROVENANCE, RUNTIME_MANIFEST, MANIFEST}
    ids: set[str] = set()
    references: dict[str, dict[str, str]] = {}
    for value in _list(source["catalogs"], "source catalogs"):
        catalog = _object(value, {"path", "sha256", "archives"}, "source catalog")
        relative = _relative(catalog["path"])
        _claim(paths, relative)
        _digest(catalog["sha256"])
        packs: dict[str, str] = {}
        for value in _list(catalog["archives"], "source archives"):
            archive = _object(value, {"pack_id", "path", "sha256"}, "source archive")
            pack_id = _string(archive["pack_id"], "source pack ID")
            if (
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", pack_id)
                or pack_id in ids
            ):
                raise ValueError("Invalid or duplicate source pack ID")
            ids.add(pack_id)
            path = _relative(archive["path"])
            if not PurePosixPath(path).is_relative_to(
                PurePosixPath(relative).parent
            ) or not path.endswith(".kpack"):
                raise ValueError("Source archive escapes its catalog directory")
            _claim(paths, path)
            _digest(archive["sha256"])
            packs[pack_id] = path
        references[relative] = packs
    for relative, catalog in contents.catalogs.items():
        if relative not in references:
            raise ValueError("Output catalog has no original source reference")
        for pack in catalog.packs:
            if references[relative].get(pack.pack_id) != _archive_path(
                relative, pack.path
            ):
                raise ValueError(
                    "Output pack has no matching original source reference"
                )


def verify_distribution(dist_root: Path) -> dict[str, object]:
    """Verify without reading the original checkout/SDK or executing its runtime.

    Returned metadata records derivation and integrity, not publisher identity.
    No source paths from the original provenance are accessed.
    """
    root = _absolute_directory(dist_root)
    files = _snapshot(root)
    raw = _object(
        _decode(_file(files, MANIFEST).data),
        {
            "schema_version",
            "kind",
            "cache_eligible",
            "selected_targets",
            "source",
            "files",
            "content_sha256",
        },
        "distribution manifest",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["kind"] != "selective-multi-vendor-distribution"
        or raw["cache_eligible"] is not False
    ):
        raise ValueError("Unsupported selective distribution manifest")
    _self_hashed(raw)
    selected = _targets(
        tuple(cast(list[str], _list(raw["selected_targets"], "selected targets")))
    )
    if list(selected) != raw["selected_targets"]:
        raise ValueError("Selected targets must be sorted")
    records = _file_records(raw["files"])
    if set(files) != set(records) | {MANIFEST} or MANIFEST in records:
        raise ValueError("Distribution file inventory mismatch")
    if files[MANIFEST].mode != 0o644:
        raise ValueError("Distribution manifest mode must be 0644")
    _check_records(files, records)
    with tempfile.TemporaryDirectory(
        prefix="therock-distribution-verify-"
    ) as temporary:
        frozen = Path(temporary)
        _write_files(frozen, files)
        contents = _validate_contents(files, frozen, selected)
    if (
        tuple(entry.target.canonical_id for entry in contents.registry.runners)
        != selected
    ):
        raise ValueError("Manifest targets do not match the exported registry")
    if any(
        entry.target not in selected
        for catalog in contents.catalogs.values()
        for pack in catalog.packs
        for entry in pack.entries
    ):
        raise ValueError("Exported catalog contains an unselected target")
    if contents.required_files != set(records):
        raise ValueError(
            "Distribution contains files outside the selected registry, packs, and runtime"
        )
    _validate_source_record(raw["source"], files, contents)
    if _snapshot(root) != files:
        raise ValueError("Distribution changed during verification")
    return raw


def _rename_library() -> ctypes.CDLL:
    if sys.platform != "linux":
        raise OSError("Selective export requires Linux renameat2(RENAME_NOREPLACE)")
    library = ctypes.CDLL(None, use_errno=True)
    if not hasattr(library, "renameat2"):
        raise OSError(
            "Linux libc does not provide renameat2; no overwriting fallback is supported"
        )
    library.renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    library.renameat2.restype = ctypes.c_int
    return library


def _publish_directory(staged: Path, destination: Path) -> None:
    library = _rename_library()
    if (
        library.renameat2(-100, os.fsencode(staged), -100, os.fsencode(destination), 1)
        != 0
    ):
        error = ctypes.get_errno()
        raise OSError(
            error,
            f"Non-overwriting directory publication failed: {os.strerror(error)}",
            str(destination),
        )


def export_distribution(
    source_dist: Path, output_dir: Path, target_ids: tuple[str, ...]
) -> dict[str, object]:
    """Publish a verified exact target subset into a new Linux directory.

    The complete shared runtime and original declared-input build history are
    retained. All logical modules and runner-advertised formats are mandatory.
    """
    selected = _targets(target_ids)
    _rename_library()
    source = _absolute_directory(source_dist)
    destination = Path(os.path.abspath(output_dir))
    parent = _absolute_directory(destination.parent)
    if (
        destination == source
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError("Source and output distribution paths overlap")
    if os.path.lexists(destination):
        raise FileExistsError(f"Output already exists: {destination}")
    files = _snapshot(source)
    if MANIFEST in files:
        raise ValueError(
            "Re-export is unsupported; use the original built distribution"
        )
    with tempfile.TemporaryDirectory(prefix="therock-distribution-input-") as inputs:
        frozen = Path(inputs)
        _write_files(frozen, files)
        contents = _validate_contents(files, frozen, selected)
        output_files = {
            path: files[path]
            for path in contents.required_files
            if path not in contents.catalogs
            and path != REGISTRY
            and not any(
                path == _archive_path(relative, pack.path)
                for relative, catalog in contents.catalogs.items()
                for pack in catalog.packs
            )
        }
        catalog_paths: list[str] = []
        with tempfile.TemporaryDirectory(
            prefix="therock-distribution-repack-"
        ) as repacks:
            for relative, catalog in contents.catalogs.items():
                packs: list[PayloadPack] = []
                for pack in catalog.packs:
                    payloads = [
                        contents.payloads[entry.key]
                        for entry in pack.entries
                        if entry.target in selected
                    ]
                    if not payloads:
                        continue
                    generated_path = create_pack(
                        Path(repacks) / pack.pack_id, pack.pack_id, payloads
                    )
                    generated = load_catalog(generated_path).packs[0]
                    output_files[_archive_path(relative, pack.path)] = _File(
                        (generated_path.parent / generated.path).read_bytes(), 0o644
                    )
                    packs.append(
                        PayloadPack(
                            pack.pack_id, pack.path, generated.sha256, generated.entries
                        )
                    )
                if packs:
                    output_files[relative] = _File(
                        _json_bytes(
                            PayloadCatalog(tuple(packs), schema_version=2).record()
                        ),
                        0o644,
                    )
                    catalog_paths.append(relative)
        registry = RunnerRegistry(
            tuple(
                entry
                for entry in contents.registry.runners
                if entry.target.canonical_id in selected
            ),
            tuple(catalog_paths),
        )
        output_files[REGISTRY] = _File(_json_bytes(registry.record()), 0o644)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "kind": "selective-multi-vendor-distribution",
            "cache_eligible": False,
            "selected_targets": list(selected),
            "source": _source_record(files, contents),
            "files": [item.record(path) for path, item in sorted(output_files.items())],
        }
        manifest["content_sha256"] = _sha(_json_bytes(manifest))
        output_files[MANIFEST] = _File(_json_bytes(manifest), 0o644)
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}-", dir=parent
        ) as staging:
            staged = Path(staging)
            _write_files(staged, output_files)
            verify_distribution(staged)
            if _snapshot(source) != files:
                raise ValueError("Source distribution changed during export")
            _absolute_directory(parent)
            staged.chmod(0o755)
            _publish_directory(staged, destination)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser(
        "export", help="Export into a fresh directory (Linux renameat2 required)"
    )
    export_parser.add_argument("--source-dist", type=Path, required=True)
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument("--target", action="append", required=True)
    verify_parser = commands.add_parser(
        "verify", help="Verify all exported files without executing a worker"
    )
    verify_parser.add_argument("--dist-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            result = export_distribution(
                args.source_dist, args.output_dir, tuple(args.target)
            )
        else:
            result = verify_distribution(args.dist_root)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
