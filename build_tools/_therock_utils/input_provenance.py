# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Content locks for explicitly declared imported SDK trees and compiler files.

Locations and symlink spelling are observations, not content identity. Symlink
identity uses a declared input name and relative target path. Trees are walked
without following directory symlinks; contained ancestor links are safe because
the target tree is already enumerated. Broken, escaping, and resolution-cyclic
links fail. A file input captures its referent and final-component link chain,
not its dynamic libraries, subprocesses, configuration files, or other implicit
dependencies. These locks do not enable artifact cache reuse.

The optional stat cache is an optimization for a trusted local filesystem. Its
checksum detects accidental corruption, not malicious cache edits. Full
verification bypasses prior cached digests. File and directory stat checks
detect changes observed during scanning; they do not provide a filesystem-wide
transaction or a guarantee against changes after verification.
"""

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

InputKind = Literal["tree", "file"]
EntryKind = Literal["directory", "file", "symlink"]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _name(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", value
    ):
        raise ValueError(f"Invalid imported input name: {value!r}")


def _relative(value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"Invalid relative input entry path: {value!r}")


def _sha(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a lowercase SHA256 digest")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> object:
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"JSON metadata must be a regular file: {path}")
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
    )


def _object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"Invalid {label} fields; expected {sorted(fields)}")
    return cast(dict[str, object], value)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return cast(int, value)


def _schema(value: object) -> None:
    if type(value) is not int or value != 1:
        raise ValueError(f"Unsupported imported input schema version: {value!r}")


@dataclass(frozen=True)
class InputSpec:
    name: str
    kind: InputKind
    path: Path

    def __post_init__(self) -> None:
        _name(self.name)
        if self.kind not in ("tree", "file"):
            raise ValueError(f"Invalid input kind for {self.name}: {self.kind!r}")
        if not isinstance(self.path, Path):
            raise ValueError("Input path must be a Path")


def load_spec(path: Path) -> tuple[InputSpec, ...]:
    raw = _object(_read_json(path), {"schema_version", "kind", "inputs"}, "input spec")
    _schema(raw["schema_version"])
    if raw["kind"] != "multi-vendor-input-spec":
        raise ValueError("Expected kind multi-vendor-input-spec")
    inputs: list[InputSpec] = []
    for value in _list(raw["inputs"], "inputs"):
        item = _object(value, {"name", "kind", "path"}, "input spec entry")
        source_path = Path(_string(item["path"], "input path"))
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        inputs.append(
            InputSpec(
                _string(item["name"], "input name"),
                cast(InputKind, _string(item["kind"], "input kind")),
                source_path,
            )
        )
    return _validate_specs(inputs)


def _validate_specs(inputs: Iterable[InputSpec]) -> tuple[InputSpec, ...]:
    result = tuple(inputs)
    if not result or not all(isinstance(item, InputSpec) for item in result):
        raise ValueError("At least one InputSpec is required")
    if len({item.name for item in result}) != len(result):
        raise ValueError("Duplicate imported input name")
    return tuple(sorted(result, key=lambda item: item.name))


@dataclass(frozen=True)
class Link:
    link_text: str
    target_input: str
    target_path: str

    def identity(self) -> dict[str, str]:
        return {"target_input": self.target_input, "target_path": self.target_path}

    def record(self) -> dict[str, str]:
        return {"link_text": self.link_text, **self.identity()}


@dataclass(frozen=True)
class Entry:
    path: str
    kind: EntryKind
    mode: int
    size: int = 0
    sha256: str = ""
    link: Link | None = None
    links: tuple[Link, ...] = ()

    def record(self, *, identity: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "path": self.path,
            "kind": self.kind,
            "mode": self.mode,
        }
        if self.kind == "file":
            result.update(size=self.size, sha256=self.sha256)
            if self.links:
                result["links"] = [
                    link.identity() if identity else link.record()
                    for link in self.links
                ]
        elif self.kind == "symlink":
            if self.link is None:
                raise ValueError("Missing symlink metadata")
            result["link"] = self.link.identity() if identity else self.link.record()
        return result


@dataclass(frozen=True)
class LockedInput:
    name: str
    kind: InputKind
    entries: tuple[Entry, ...]

    @property
    def content_sha256(self) -> str:
        return _digest(
            {
                "kind": self.kind,
                "entries": [entry.record(identity=True) for entry in self.entries],
            }
        )

    def record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "content_sha256": self.content_sha256,
            "entries": [entry.record() for entry in self.entries],
        }


@dataclass(frozen=True)
class InputLock:
    inputs: tuple[LockedInput, ...]

    @property
    def global_content_sha256(self) -> str:
        return _digest(
            [
                {
                    "name": item.name,
                    "kind": item.kind,
                    "content_sha256": item.content_sha256,
                }
                for item in self.inputs
            ]
        )

    def record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "multi-vendor-input-lock",
            "coverage": "declared-inputs",
            "cache_eligible": False,
            "global_content_sha256": self.global_content_sha256,
            "inputs": [item.record() for item in self.inputs],
        }


def _read_link(value: object) -> Link:
    raw = _object(value, {"link_text", "target_input", "target_path"}, "symlink")
    text = _string(raw["link_text"], "link text")
    name = _string(raw["target_input"], "symlink target input")
    relative = _string(raw["target_path"], "symlink target path")
    _name(name)
    _relative(relative)
    return Link(text, name, relative)


def _read_entry(value: object) -> Entry:
    if not isinstance(value, dict):
        raise ValueError("Input lock entry must be an object")
    kind = value.get("kind")
    fields = {"path", "kind", "mode"}
    if kind == "file":
        fields |= {"size", "sha256"}
        if "links" in value:
            fields.add("links")
    elif kind == "symlink":
        fields.add("link")
    elif kind != "directory":
        raise ValueError(f"Invalid input lock entry kind: {kind!r}")
    raw = _object(value, fields, "input lock entry")
    relative = _string(raw["path"], "entry path")
    _relative(relative)
    mode = _integer(raw["mode"], "entry mode")
    if not 0 <= mode <= 0o7777:
        raise ValueError("Invalid entry permission mode")
    size = _integer(raw["size"], "file size") if kind == "file" else 0
    if size < 0:
        raise ValueError("File size must be nonnegative")
    links = tuple(
        _read_link(link) for link in _list(raw.get("links", []), "file links")
    )
    return Entry(
        relative,
        cast(EntryKind, kind),
        mode,
        size,
        _sha(raw["sha256"]) if kind == "file" else "",
        _read_link(raw["link"]) if kind == "symlink" else None,
        links,
    )


def load_lock(path: Path) -> InputLock:
    raw = _object(
        _read_json(path),
        {
            "schema_version",
            "kind",
            "coverage",
            "cache_eligible",
            "global_content_sha256",
            "inputs",
        },
        "input lock",
    )
    _schema(raw["schema_version"])
    if raw["kind"] != "multi-vendor-input-lock":
        raise ValueError("Expected kind multi-vendor-input-lock")
    if raw["coverage"] != "declared-inputs" or raw["cache_eligible"] is not False:
        raise ValueError(
            "Input locks cover declared inputs only and cannot enable cache reuse"
        )
    inputs: list[LockedInput] = []
    for value in _list(raw["inputs"], "locked inputs"):
        item = _object(
            value, {"name", "kind", "content_sha256", "entries"}, "locked input"
        )
        name = _string(item["name"], "input name")
        _name(name)
        kind = _string(item["kind"], "input kind")
        if kind not in ("tree", "file"):
            raise ValueError(f"Invalid locked input kind: {kind}")
        entries = tuple(
            _read_entry(entry) for entry in _list(item["entries"], "entries")
        )
        paths = [entry.path for entry in entries]
        if not entries or paths != sorted(set(paths)):
            raise ValueError(
                f"Input {name} entries must be nonempty, unique and sorted"
            )
        if entries[0].path != "." or entries[0].kind != (
            "directory" if kind == "tree" else "file"
        ):
            raise ValueError(f"Input {name} has an invalid root entry")
        if kind == "file" and len(entries) != 1:
            raise ValueError(f"File input {name} must have one root entry")
        locked = LockedInput(name, cast(InputKind, kind), entries)
        if locked.content_sha256 != _sha(item["content_sha256"]):
            raise ValueError(f"Input lock content checksum mismatch: {name}")
        inputs.append(locked)
    names = [item.name for item in inputs]
    if not names or names != sorted(set(names)):
        raise ValueError("Locked input names must be nonempty, unique and sorted")
    for item in inputs:
        for entry in item.entries:
            links = entry.links + ((entry.link,) if entry.link is not None else ())
            if any(link.target_input not in names for link in links):
                raise ValueError("Symlink references an undeclared input")
    lock = InputLock(tuple(inputs))
    if lock.global_content_sha256 != _sha(raw["global_content_sha256"]):
        raise ValueError("Input lock global checksum mismatch")
    return lock


def _signature(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mode,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


@dataclass(frozen=True)
class CachedFile:
    signature: tuple[int, ...]
    sha256: str


class _StatCache:
    def __init__(self, path: Path | None, full: bool):
        self.path = path
        self.previous: dict[str, CachedFile] = {}
        self.current: dict[str, CachedFile] = {}
        if path is None or full:
            return
        try:
            raw = _object(
                _read_json(path),
                {"schema_version", "kind", "files", "checksum"},
                "stat cache",
            )
            _schema(raw["schema_version"])
            if raw["kind"] != "multi-vendor-input-stat-cache":
                raise ValueError("Wrong cache kind")
            if _digest(raw["files"]) != _sha(raw["checksum"]):
                raise ValueError("Cache checksum mismatch")
            if not isinstance(raw["files"], dict):
                raise ValueError("Invalid cached files")
            loaded: dict[str, CachedFile] = {}
            for filename, value in raw["files"].items():
                if not isinstance(filename, str):
                    raise ValueError("Invalid cache filename")
                entry = _object(value, {"signature", "sha256"}, "cached file")
                signature = tuple(
                    _integer(part, "cached stat")
                    for part in _list(entry["signature"], "signature")
                )
                if len(signature) != 6:
                    raise ValueError("Invalid stat signature")
                loaded[filename] = CachedFile(signature, _sha(entry["sha256"]))
            self.previous = loaded
        except (OSError, ValueError, TypeError):
            # A broken optimization must never become an input trust decision.
            self.previous = {}

    def file(self, path: Path) -> tuple[str, os.stat_result]:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Expected regular input file: {path}")
        signature = _signature(before)
        key = os.fspath(path)
        cached = self.current.get(key) or self.previous.get(key)
        if cached is not None and cached.signature == signature:
            digest = cached.sha256
        else:
            digest_state = hashlib.sha256()
            with path.open("rb") as file:
                if _signature(os.fstat(file.fileno())) != signature:
                    raise ValueError(f"Input changed before reading: {path}")
                while chunk := file.read(1024 * 1024):
                    digest_state.update(chunk)
                if _signature(os.fstat(file.fileno())) != signature:
                    raise ValueError(f"Input changed during reading: {path}")
            digest = digest_state.hexdigest()
        after = path.stat()
        if _signature(after) != signature:
            raise ValueError(f"Input changed during hashing: {path}")
        self.current[key] = CachedFile(signature, digest)
        return digest, before

    def save(self) -> None:
        if self.path is None:
            return
        files = {
            path: {"signature": list(value.signature), "sha256": value.sha256}
            for path, value in sorted(self.current.items())
        }
        _write_if_changed(
            self.path,
            {
                "schema_version": 1,
                "kind": "multi-vendor-input-stat-cache",
                "files": files,
                "checksum": _digest(files),
            },
        )


@dataclass(frozen=True)
class _Root:
    spec: InputSpec
    current_path: Path
    resolved_path: Path


@dataclass(frozen=True)
class Observation:
    lock: InputLock
    roots: tuple[_Root, ...]

    def report(self) -> dict[str, object]:
        items: list[dict[str, object]] = []
        for root, item in zip(self.roots, self.lock.inputs):
            counts = {
                "files": sum(entry.kind == "file" for entry in item.entries),
                "directories": sum(entry.kind == "directory" for entry in item.entries),
                "symlinks": sum(entry.kind == "symlink" for entry in item.entries)
                + sum(len(entry.links) for entry in item.entries),
                "bytes": sum(
                    entry.size for entry in item.entries if entry.kind == "file"
                ),
            }
            items.append(
                {
                    "name": item.name,
                    "kind": item.kind,
                    "content_sha256": item.content_sha256,
                    "current_path": os.fspath(root.current_path),
                    "resolved_path": os.fspath(root.resolved_path),
                    "counts": counts,
                }
            )
        return {
            "schema_version": 1,
            "kind": "multi-vendor-input-provenance",
            "coverage": "declared-inputs",
            "cache_eligible": False,
            "global_content_sha256": self.lock.global_content_sha256,
            "inputs": items,
        }


def _resolve(path: Path) -> Path:
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            f"Broken, inaccessible, or cyclic input path: {path}: {exc}"
        ) from exc


def _link(path: Path, roots: tuple[_Root, ...]) -> Link:
    before = path.lstat()
    if not stat.S_ISLNK(before.st_mode):
        raise ValueError(f"Input symlink changed during scanning: {path}")
    text = os.readlink(path)
    target = _resolve(path)
    target_info = target.stat()
    if not (stat.S_ISREG(target_info.st_mode) or stat.S_ISDIR(target_info.st_mode)):
        raise ValueError(f"Symlink target is not a regular file or directory: {path}")
    if _signature(before) != _signature(path.lstat()) or text != os.readlink(path):
        raise ValueError(f"Input symlink changed during scanning: {path}")
    # Trees take precedence over file aliases; selection is stable by input name.
    for root in roots:
        if root.spec.kind == "tree" and target.is_relative_to(root.resolved_path):
            return Link(
                text, root.spec.name, target.relative_to(root.resolved_path).as_posix()
            )
    for root in roots:
        if root.spec.kind == "file" and target == root.resolved_path:
            return Link(text, root.spec.name, ".")
    raise ValueError(f"Input symlink escapes all declared roots: {path} -> {text}")


def _file_links(path: Path, roots: tuple[_Root, ...]) -> tuple[Link, ...]:
    # Canonicalize parent aliases but retain symlinks on the declared filename.
    current = _resolve(path.parent) / path.name
    links: list[Link] = []
    seen: set[Path] = set()
    while current.is_symlink():
        if current in seen:
            raise ValueError(f"Cyclic file input symlink: {path}")
        seen.add(current)
        link = _link(current, roots)
        links.append(link)
        next_path = Path(link.link_text)
        if not next_path.is_absolute():
            next_path = current.parent / next_path
        current = _resolve(next_path.parent) / next_path.name
    return tuple(links)


def observe_inputs(
    inputs: Iterable[InputSpec], *, cache_path: Path | None = None, full: bool = False
) -> Observation:
    specs = _validate_specs(inputs)
    _check_output_paths(specs, (("cache", cache_path),))
    roots_list: list[_Root] = []
    for spec in specs:
        current = Path(os.path.abspath(spec.path))
        resolved = _resolve(current)
        info = resolved.stat()
        expected = (
            stat.S_ISDIR(info.st_mode)
            if spec.kind == "tree"
            else stat.S_ISREG(info.st_mode)
        )
        if not expected:
            raise ValueError(
                f"Input {spec.name} is not a regular {spec.kind}: {current}"
            )
        roots_list.append(_Root(spec, current, resolved))
    roots = tuple(roots_list)
    cache = _StatCache(cache_path, full)
    locked_inputs: list[LockedInput] = []
    for root in roots:
        entries: list[Entry] = []
        if root.spec.kind == "file":
            links = _file_links(root.current_path, roots)
            digest, info = cache.file(root.resolved_path)
            if _resolve(root.current_path) != root.resolved_path:
                raise ValueError(
                    f"File input referent changed during scanning: {root.spec.name}"
                )
            entries.append(
                Entry(
                    ".",
                    "file",
                    stat.S_IMODE(info.st_mode),
                    info.st_size,
                    digest,
                    links=links,
                )
            )
        else:
            # os.walk lists a directory before yielding it. Save its stat before
            # descent so changes during that listing cannot escape the checks.
            directory_stats: dict[Path, tuple[int, ...]] = {
                root.resolved_path: _signature(root.resolved_path.lstat())
            }

            def on_error(error: OSError) -> None:
                raise error

            for directory, names, files in os.walk(
                root.resolved_path, topdown=True, followlinks=False, onerror=on_error
            ):
                path = Path(directory)
                info = path.lstat()
                if not stat.S_ISDIR(info.st_mode) or directory_stats.get(
                    path
                ) != _signature(info):
                    raise ValueError(f"Directory changed during scanning: {path}")
                entries.append(
                    Entry(
                        path.relative_to(root.resolved_path).as_posix(),
                        "directory",
                        stat.S_IMODE(info.st_mode),
                    )
                )
                names.sort()
                files.sort()
                for name in [*names, *files]:
                    child = path / name
                    info = child.lstat()
                    relative = child.relative_to(root.resolved_path).as_posix()
                    if stat.S_ISLNK(info.st_mode):
                        entries.append(
                            Entry(
                                relative,
                                "symlink",
                                stat.S_IMODE(info.st_mode),
                                link=_link(child, roots),
                            )
                        )
                    elif stat.S_ISREG(info.st_mode):
                        digest, file_info = cache.file(child)
                        entries.append(
                            Entry(
                                relative,
                                "file",
                                stat.S_IMODE(file_info.st_mode),
                                file_info.st_size,
                                digest,
                            )
                        )
                    elif stat.S_ISDIR(info.st_mode):
                        directory_stats[child] = _signature(info)
                    else:
                        raise ValueError(
                            f"Special input file is not supported: {child}"
                        )
            for directory, expected_stat in directory_stats.items():
                if _signature(directory.lstat()) != expected_stat:
                    raise ValueError(
                        f"Input directory changed during scanning: {directory}"
                    )
            if _resolve(root.current_path) != root.resolved_path:
                raise ValueError(
                    f"Tree input referent changed during scanning: {root.spec.name}"
                )
        locked_inputs.append(
            LockedInput(
                root.spec.name,
                root.spec.kind,
                tuple(sorted(entries, key=lambda entry: entry.path)),
            )
        )
    observation = Observation(InputLock(tuple(locked_inputs)), roots)
    cache.save()
    return observation


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _write_if_changed(path: Path, value: object) -> None:
    data = _json_bytes(value)
    try:
        if path.read_bytes() == data:
            return
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _publish_new(path: Path, lock: InputLock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(_json_bytes(lock.record()))
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _compare(expected: InputLock, observed: InputLock) -> None:
    if expected.global_content_sha256 == observed.global_content_sha256:
        return
    before = {item.name: item for item in expected.inputs}
    after = {item.name: item for item in observed.inputs}
    changes = []
    for name in sorted(set(before) | set(after)):
        old = before[name].content_sha256 if name in before else "<missing>"
        new = after[name].content_sha256 if name in after else "<missing>"
        if old != new:
            changes.append(f"{name}: old={old}, new={new}")
    raise ValueError(
        "Imported input lock mismatch; changed inputs: "
        + "; ".join(changes)
        + ". Use a clean build or review and explicitly select a new snapshot lock. "
        "Existing locks are never replaced automatically."
    )


def _check_output_paths(
    inputs: tuple[InputSpec, ...], outputs: tuple[tuple[str, Path | None], ...]
) -> None:
    resolved_outputs: dict[Path, str] = {}
    for role, path in outputs:
        if path is None:
            continue
        resolved = path.resolve(strict=False)
        if resolved.exists() and not stat.S_ISREG(resolved.stat().st_mode):
            raise ValueError(f"{role} output must be a regular file: {path}")
        if resolved in resolved_outputs:
            raise ValueError(
                f"Output paths overlap: {resolved_outputs[resolved]} and {role}"
            )
        resolved_outputs[resolved] = role
    for spec in inputs:
        root = _resolve(spec.path)
        for path, role in resolved_outputs.items():
            if path == root or (spec.kind == "tree" and path.is_relative_to(root)):
                raise ValueError(
                    f"{role} output must be outside declared input {spec.name}"
                )


def capture_lock(
    inputs: Iterable[InputSpec],
    lock_path: Path,
    *,
    cache_path: Path | None = None,
    report_path: Path | None = None,
    verify_only: bool = False,
    full: bool = False,
) -> InputLock:
    inputs = _validate_specs(inputs)
    _check_output_paths(
        inputs, (("lock", lock_path), ("cache", cache_path), ("report", report_path))
    )
    existing = load_lock(lock_path) if os.path.lexists(lock_path) else None
    if verify_only and existing is None:
        raise ValueError(
            f"Input lock does not exist: {lock_path}; capture or review a snapshot first"
        )
    observation = observe_inputs(inputs, cache_path=cache_path, full=full)
    if existing is None:
        try:
            _publish_new(lock_path, observation.lock)
        except FileExistsError:
            _compare(load_lock(lock_path), observation.lock)
    else:
        _compare(existing, observation.lock)
    if report_path is not None:
        _write_if_changed(report_path, observation.report())
    return observation.lock


def snapshot_lock(
    inputs: Iterable[InputSpec], output_path: Path, *, cache_path: Path | None = None
) -> InputLock:
    if os.path.lexists(output_path):
        raise FileExistsError(
            f"Snapshot output already exists: {output_path}; choose a new lock path"
        )
    inputs = _validate_specs(inputs)
    _check_output_paths(inputs, (("lock", output_path), ("cache", cache_path)))
    observation = observe_inputs(inputs, cache_path=cache_path)
    _publish_new(output_path, observation.lock)
    return observation.lock
