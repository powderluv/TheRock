# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Strict registry metadata and filesystem integrity tests; no GPU required."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "build_tools"))

from _therock_utils.gpu_targets import parse_gpu_target
from _therock_utils.module_contract import (
    parse_contract,
    runner_description,
    validation_contract,
)
from _therock_utils.runner_registry import (
    RunnerEntry,
    RunnerRegistry,
    load_registry,
    resolve_registry_path,
    verify_runner,
)

AMD = "amd:hip:gfx1201"
NVIDIA = "nvidia:cuda:sm_120"
INTEL = "intel:level-zero:xe2-b70"
CATALOG = "share/therock/packs/saxpy/catalog.json"


class RunnerRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dist = self.root / "distribution"
        self.dist.mkdir()
        self.path = self.root / "runners.json"
        self.entry = self.make_entry(AMD)
        self.registry = RunnerRegistry((self.entry,), (CATALOG,))

    def make_entry(self, identity: str) -> RunnerEntry:
        target = parse_gpu_target(identity)
        relative = f"bin/{target.slug}/therock_module_validation"
        data = f"synthetic runner {identity}".encode()
        path = self.dist / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return RunnerEntry(
            target,
            relative,
            hashlib.sha256(data).hexdigest(),
            runner_description(target.vendor),
        )

    def load(self, document: object) -> RunnerRegistry:
        self.path.write_text(json.dumps(document))
        return load_registry(self.path)

    def test_roundtrip_three_vendors_and_sorted_runner_order(self) -> None:
        nvidia = self.make_entry(NVIDIA)
        intel = self.make_entry(INTEL)
        registry = RunnerRegistry(
            (nvidia, intel, self.entry),
            (CATALOG, "share/therock/packs/relu/catalog.json"),
        )
        self.assertEqual(self.load(registry.record()), registry)
        self.assertEqual(
            tuple(entry.target.canonical_id for entry in registry.runners),
            (AMD, INTEL, NVIDIA),
        )
        for entry in registry.runners:
            self.assertEqual(verify_runner(self.dist, entry), self.dist / entry.path)

    def test_invalid_top_level_schema_is_rejected(self) -> None:
        valid = self.registry.record()
        documents = (
            None,
            [],
            {},
            {**valid, "extra": True},
            {**valid, "schema_version": True},
            {**valid, "schema_version": 2},
            {**valid, "schema_version": "1"},
            {**valid, "kind": "payload-catalog"},
            {**valid, "runners": ()},
            {**valid, "runners": {}},
            {**valid, "catalogs": None},
            {**valid, "catalogs": []},
        )
        for document in documents:
            with self.subTest(document=document), self.assertRaises(ValueError):
                self.load(document)

    def test_invalid_runner_fields_targets_and_hashes_are_rejected(self) -> None:
        valid = self.entry.record()
        changes = (
            {"extra": True},
            {"target": True},
            {"target": "gfx1201"},
            {"target": "amd:hip:gfx942:xnack+:sramecc-"},
            {"sha256": "A" * 64},
            {"sha256": "0" * 63},
            {"sha256": 0},
            {"description": None},
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load({**self.registry.record(), "runners": [{**valid, **change}]})
        for missing in valid:
            invalid = dict(valid)
            del invalid[missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.load({**self.registry.record(), "runners": [invalid]})

    def test_invalid_paths_are_rejected_for_runners_and_catalogs(self) -> None:
        paths = (
            None,
            7,
            "",
            "/bin/runner",
            "../runner",
            "bin/../runner",
            "./runner",
            "bin//runner",
            "bin/runner/",
            "bin\\runner",
            "C:/runner",
            "//host/runner",
            "bin/\0runner",
            "bin/\nrunner",
        )
        for path in paths:
            for field in ("runner", "catalog"):
                document = self.registry.record()
                if field == "runner":
                    document["runners"][0]["path"] = path
                else:
                    document["catalogs"] = [path]
                with self.subTest(path=path, field=field), self.assertRaises(
                    ValueError
                ):
                    self.load(document)

    def test_duplicate_identities_and_paths_are_rejected(self) -> None:
        nvidia = self.make_entry(NVIDIA).record()
        documents = (
            {
                **self.registry.record(),
                "runners": [self.entry.record(), self.entry.record()],
            },
            {
                **self.registry.record(),
                "runners": [self.entry.record(), {**nvidia, "path": self.entry.path}],
            },
            {**self.registry.record(), "catalogs": [CATALOG, CATALOG]},
        )
        for document in documents:
            with self.subTest(document=document), self.assertRaisesRegex(
                ValueError, "Duplicate"
            ):
                self.load(document)

    def test_duplicate_json_fields_at_any_depth_are_rejected(self) -> None:
        valid = json.dumps(self.registry.record())
        documents = (
            valid.replace(
                '"schema_version": 1', '"schema_version": 2, "schema_version": 1', 1
            ),
            valid.replace('"target":', '"target": "other", "target":', 1),
            valid.replace('"vendor": "amd"', '"vendor": "nvidia", "vendor": "amd"', 1),
        )
        for document in documents:
            self.path.write_text(document)
            with self.subTest(document=document), self.assertRaisesRegex(
                ValueError, "Duplicate runner registry JSON field"
            ):
                load_registry(self.path)

    def test_incompatible_descriptions_are_rejected(self) -> None:
        description = runner_description("amd").record()
        changed_contract = parse_contract(
            {**validation_contract().record(), "version": 2}
        )
        alternatives = (
            runner_description("nvidia").record(),
            {**description, "scope": "hardware-qualified"},
            {**description, "entry_points": ["therock_module_saxpy"]},
            {
                **description,
                "capabilities": description["capabilities"] + ["future-capability"],
            },
            {**description, "contract_sha256": "0" * 64},
            {
                **description,
                "contract": changed_contract.record(),
                "contract_sha256": changed_contract.sha256,
            },
            {**description, "schema_version": True},
        )
        for alternate in alternatives:
            document = self.registry.record()
            document["runners"][0]["description"] = alternate
            with self.subTest(description=alternate), self.assertRaises(ValueError):
                self.load(document)
        nvidia = self.make_entry(NVIDIA)
        document = RunnerRegistry((nvidia,), (CATALOG,)).record()
        document["runners"][0]["description"]["payload_types"] = ["cubin"]
        with self.assertRaisesRegex(ValueError, "payload formats"):
            self.load(document)

    def test_loading_metadata_does_not_require_installed_files(self) -> None:
        (self.dist / self.entry.path).unlink()
        self.assertEqual(self.load(self.registry.record()), self.registry)

    def test_changed_runner_is_rejected_even_with_same_size(self) -> None:
        path = self.dist / self.entry.path
        data = path.read_bytes()
        path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        with self.assertRaisesRegex(ValueError, "Runner SHA256 mismatch"):
            verify_runner(self.dist, self.entry)

    def test_missing_runner_and_catalog_are_rejected(self) -> None:
        (self.dist / self.entry.path).unlink()
        with self.assertRaises(FileNotFoundError):
            verify_runner(self.dist, self.entry)
        with self.assertRaises(FileNotFoundError):
            resolve_registry_path(self.dist, CATALOG)

    def test_directory_and_nonexistent_root_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            resolve_registry_path(self.dist, "bin")
        with self.assertRaises(FileNotFoundError):
            resolve_registry_path(self.root / "missing", self.entry.path)
        with self.assertRaisesRegex(ValueError, "not a directory"):
            resolve_registry_path(self.dist / self.entry.path, "runner")

    def test_resolver_rejects_unsafe_paths_independent_of_loading(self) -> None:
        for path in ("../runners.json", "/etc/passwd", "bin/../runner", "bin\\runner"):
            with self.subTest(path=path), self.assertRaisesRegex(
                ValueError, "relative POSIX"
            ):
                resolve_registry_path(self.dist, path)

    def test_symlinks_inside_distribution_are_allowed_but_escape_is_rejected(
        self,
    ) -> None:
        runner = self.dist / self.entry.path
        alias = self.dist / "alias"
        alias.symlink_to(runner)
        self.assertEqual(resolve_registry_path(self.dist, "alias"), runner)
        outside = self.root / "outside"
        outside.write_bytes(runner.read_bytes())
        runner.unlink()
        runner.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes"):
            verify_runner(self.dist, self.entry)
        outside_dir = self.root / "outside-dir"
        outside_dir.mkdir()
        (outside_dir / "catalog.json").write_text("{}")
        (self.dist / "escape").symlink_to(outside_dir, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            resolve_registry_path(self.dist, "escape/catalog.json")

    def test_symlink_loop_is_rejected(self) -> None:
        (self.dist / "loop").symlink_to("loop")
        with self.assertRaises((ValueError, OSError)):
            resolve_registry_path(self.dist, "loop")

    def test_typed_constructor_rejects_invalid_shapes(self) -> None:
        for runners, catalogs in (
            ([], (CATALOG,)),
            ((), (CATALOG,)),
            ((None,), (CATALOG,)),
            ((self.entry,), []),
            ((self.entry,), ()),
        ):
            with self.subTest(runners=runners, catalogs=catalogs), self.assertRaises(
                ValueError
            ):
                RunnerRegistry(runners, catalogs)
        with self.assertRaises(ValueError):
            RunnerEntry(AMD, self.entry.path, self.entry.sha256, self.entry.description)
        with self.assertRaises(ValueError):
            verify_runner(self.dist, None)

    def test_excessively_nested_json_has_actionable_error(self) -> None:
        depth = sys.getrecursionlimit() + 100
        self.path.write_text("[" * depth + "0" + "]" * depth)
        with self.assertRaises(ValueError):
            load_registry(self.path)


if __name__ == "__main__":
    unittest.main()
