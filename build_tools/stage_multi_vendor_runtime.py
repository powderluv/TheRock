#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Materialize the bounded Python runtime and verify it before child install.

The manifest describes selected source bytes and installed output bytes. It is a
local build-completion receipt, not publisher authentication or a dependency lock.
The archive-reader subset depends on explicitly installed msgpack and zstandard.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile

PYTHON_ROOT = "share/therock/python"
EXAMPLE = "share/therock/examples/packed_session_client.py"
MANIFEST = PYTHON_ROOT + "/runtime-manifest.json"


def _json_bytes(record: object) -> bytes:
    return (json.dumps(record, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ";" in value:
        raise ValueError("paths must be nonempty normalized relative paths")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {value}")
    return value


def _read_regular(path: Path) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"expected a regular file: {path}")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        data = stream.read()
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mode", "st_mtime_ns", "st_ctime_ns")
    fingerprint = lambda item: tuple(getattr(item, name) for name in fields)
    if not (
        fingerprint(before)
        == fingerprint(opened)
        == fingerprint(after)
        == fingerprint(path.lstat())
    ):
        raise ValueError(f"file changed while reading: {path}")
    return data


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Source:
    root: str
    path: str
    destination: str
    resolved: Path


@dataclass(frozen=True)
class Snapshot:
    manifest: bytes
    outputs: dict[str, bytes]


class RuntimeSources:
    def __init__(
        self, tools_dir: Path, kpack_python_dir: Path, config: Path | None = None
    ):
        self.tools_dir = tools_dir.resolve(strict=True)
        self.config = config or self.tools_dir / "multi_vendor_runtime_sources.json"
        self.helper = Path(__file__).resolve()
        roots = {
            "tools": self.tools_dir,
            "repository": self.tools_dir.parent,
            "kpack": kpack_python_dir.resolve(strict=True).parent,
        }
        if any(
            ";" in str(path) for path in (*roots.values(), self.config, self.helper)
        ):
            raise ValueError("source paths cannot contain CMake list separators")
        self.config_bytes = _read_regular(self.config)
        record = json.loads(self.config_bytes, object_pairs_hook=_unique_object)
        if not isinstance(record, dict) or set(record) != {"schema_version", "files"}:
            raise ValueError("invalid runtime source map")
        if type(record["schema_version"]) is not int or record["schema_version"] != 1:
            raise ValueError("unsupported runtime source map schema")
        files = record["files"]
        if not isinstance(files, list) or not files:
            raise ValueError("runtime source map requires files")
        self.sources: list[Source] = []
        destinations = set()
        for item in files:
            if not isinstance(item, dict) or set(item) != {
                "root",
                "path",
                "destination",
            }:
                raise ValueError("invalid runtime source entry")
            role = item["root"]
            if not isinstance(role, str) or role not in roots:
                raise ValueError("unknown runtime source root")
            path = _relative(item["path"])
            destination = _relative(item["destination"])
            if not (
                destination.startswith(PYTHON_ROOT + "/") or destination == EXAMPLE
            ):
                raise ValueError("runtime destination outside managed paths")
            if destination == MANIFEST or destination in destinations:
                raise ValueError("duplicate or reserved runtime destination")
            destinations.add(destination)
            source_path = roots[role] / path
            # A selected file may not silently resolve outside its declared root.
            if not source_path.resolve(strict=True).is_relative_to(roots[role]):
                raise ValueError(f"runtime source escapes declared root: {path}")
            self.sources.append(Source(role, path, destination, source_path))
        self.sources.sort(key=lambda item: item.destination)
        for destination in destinations:
            if any(
                str(parent) in destinations
                for parent in PurePosixPath(destination).parents
            ):
                raise ValueError(
                    "runtime destinations have conflicting file/directory paths"
                )

    def listing(self) -> dict[str, list[str]]:
        return {
            "source_files": sorted(
                {
                    str(self.config),
                    str(self.helper),
                    *(str(source.resolved) for source in self.sources),
                }
            ),
            "output_files": [
                *(source.destination for source in self.sources),
                MANIFEST,
            ],
        }

    def snapshot(self) -> Snapshot:
        if _read_regular(self.config) != self.config_bytes:
            raise ValueError("runtime source map changed; reconfigure before building")
        outputs = {}
        sources = []
        for source in self.sources:
            data = _read_regular(source.resolved)
            outputs[source.destination] = data
            sources.append(
                {
                    "root": source.root,
                    "path": source.path,
                    "destination": source.destination,
                    "sha256": _sha(data),
                }
            )
        body = {
            "schema_version": 1,
            "kind": "multi-vendor-python-runtime",
            "coverage": "selected-runtime-sources",
            "kpack_scope": "archive-reader-subset",
            "sources": sources,
            "packaging": {
                "source_map_sha256": _sha(self.config_bytes),
                "helper_sha256": _sha(_read_regular(self.helper)),
            },
            "outputs": [
                {"path": path, "sha256": _sha(data), "size": len(data), "mode": 0o644}
                for path, data in sorted(outputs.items())
            ],
        }
        body["content_sha256"] = _sha(_json_bytes(body))
        return Snapshot(_json_bytes(body), outputs)


def _safe_output_path(root: Path, relative: str) -> Path:
    path = root / relative
    for candidate in (
        root,
        *(
            root / parent
            for parent in reversed(PurePosixPath(relative).parents)
            if str(parent) != "."
        ),
        path,
    ):
        if candidate.is_symlink():
            raise ValueError(f"runtime output must not be a symlink: {candidate}")
    return path


def _managed_files(root: Path) -> set[str]:
    result = set()
    python_root = _safe_output_path(root, PYTHON_ROOT)
    if python_root.exists():
        if not python_root.is_dir():
            raise ValueError("runtime Python root must be a directory")
        for directory, directories, files in os.walk(python_root, followlinks=False):
            for name in directories:
                if (Path(directory) / name).is_symlink():
                    raise ValueError("runtime output contains a directory symlink")
            for name in files:
                path = Path(directory) / name
                if not stat.S_ISREG(path.lstat().st_mode):
                    raise ValueError(f"runtime output must be a regular file: {path}")
                result.add(path.relative_to(root).as_posix())
    example = _safe_output_path(root, EXAMPLE)
    if example.exists():
        if not stat.S_ISREG(example.lstat().st_mode):
            raise ValueError("runtime example must be a regular file")
        result.add(EXAMPLE)
    return result


def _same(path: Path, data: bytes) -> bool:
    return (
        path.exists()
        and _read_regular(path) == data
        and stat.S_IMODE(path.stat().st_mode) == 0o644
    )


def _write(path: Path, data: bytes) -> None:
    if _same(path, data):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            os.fchmod(stream.fileno(), 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _materialize(root: Path, snapshot: Snapshot) -> None:
    existing = _managed_files(root)
    expected = {*snapshot.outputs, MANIFEST}
    changes = existing != expected
    for relative, data in {**snapshot.outputs, MANIFEST: snapshot.manifest}.items():
        path = _safe_output_path(root, relative)
        if not _same(path, data):
            changes = True
    if not changes:
        return
    # Completion is withdrawn before replacing any managed output and published last.
    manifest = _safe_output_path(root, MANIFEST)
    manifest.unlink(missing_ok=True)
    for relative in sorted(existing - expected):
        (root / relative).unlink()
    for relative, data in snapshot.outputs.items():
        _write(_safe_output_path(root, relative), data)
    _write(manifest, snapshot.manifest)


def verify(sources: RuntimeSources, output_dir: Path) -> Snapshot:
    snapshot = sources.snapshot()
    expected = {*snapshot.outputs, MANIFEST}
    if _managed_files(output_dir) != expected:
        raise ValueError(
            "runtime outputs are missing or unexpected; rebuild before install"
        )
    if not _same(_safe_output_path(output_dir, MANIFEST), snapshot.manifest):
        raise ValueError(
            "runtime sources or completion manifest changed; rebuild before install"
        )
    for relative, data in snapshot.outputs.items():
        if not _same(_safe_output_path(output_dir, relative), data):
            raise ValueError(
                f"runtime output changed: {relative}; rebuild before install"
            )
    return snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "build", "verify", "install"))
    parser.add_argument("--tools-dir", type=Path, required=True)
    parser.add_argument("--kpack-python-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args(argv)
    if args.command != "list" and args.output_dir is None:
        parser.error("--output-dir is required for this command")
    if args.command == "install" and args.destination is None:
        parser.error("--destination is required for install")
    try:
        sources = RuntimeSources(args.tools_dir, args.kpack_python_dir, args.config)
        if args.command == "list":
            print(json.dumps(sources.listing(), sort_keys=True))
        elif args.command == "build":
            snapshot = sources.snapshot()
            # Catch edits during source collection before claiming completed output.
            if snapshot != sources.snapshot():
                raise ValueError("runtime sources changed during build; retry")
            _materialize(args.output_dir.absolute(), snapshot)
        elif args.command == "verify":
            verify(sources, args.output_dir.absolute())
        else:
            snapshot = verify(sources, args.output_dir.absolute())
            _materialize(args.destination.absolute(), snapshot)
        return 0
    except (OSError, ValueError) as exc:
        print(f"runtime packaging error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
