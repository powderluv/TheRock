# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Imported-input lock contracts exercised through the CLI, without GPU claims."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "multi_vendor_inputs.py"
sys.path.insert(0, str(SCRIPT.parent))
from _therock_utils import input_provenance


class MultiVendorInputsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="therock-input-lock-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sdk = self.root / "SDK with spaces"
        (self.sdk / "include").mkdir(parents=True)
        (self.sdk / "lib" / "empty").mkdir(parents=True)
        self.header = self.sdk / "include" / "fixture.h"
        self.header.write_bytes(b"#define FIXTURE 1\n")
        self.header.chmod(0o644)
        self.library = self.sdk / "lib" / "fixture.so"
        self.library.write_bytes(b"synthetic library bytes\x00\xff")
        self.compiler = self.root / "compiler"
        self.executed = self.root / "compiler-was-executed"
        self.compiler.write_text(f"#!/bin/sh\ntouch '{self.executed}'\nexit 19\n")
        self.compiler.chmod(0o755)
        self.inputs = [("sdk", "tree", self.sdk), ("compiler", "file", self.compiler)]
        self.lock = self.root / "inputs.lock.json"
        self.cache = self.root / "inputs.cache.json"

    def cli(self, operation, *args, inputs=None, spec=None, success=True):
        command = [sys.executable, str(SCRIPT), operation, *map(str, args)]
        if spec is not None:
            command.extend(("--spec", str(spec)))
        else:
            for name, kind, path in self.inputs if inputs is None else inputs:
                command.extend(("--input", name, kind, str(path)))
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        evidence = f"{command!r}\n{result.stdout}\n{result.stderr}"
        if success:
            self.assertEqual(result.returncode, 0, evidence)
        else:
            self.assertNotEqual(result.returncode, 0, evidence)
            self.assertNotIn("Traceback", result.stderr, evidence)
        self.assertFalse(self.executed.exists(), "Input capture executed a compiler")
        return result

    def capture(self, *, path=None, inputs=None, extra=(), success=True):
        path = path or self.lock
        self.cli("capture", "--lock", path, *extra, inputs=inputs, success=success)
        return json.loads(path.read_text()) if success else None

    def verify(self, *, inputs=None, extra=(), success=True):
        return self.cli(
            "verify", "--lock", self.lock, *extra, inputs=inputs, success=success
        )

    @staticmethod
    def identity(lock):
        return lock["global_content_sha256"]

    def replace_header_preserving_size_and_mtime(self):
        before = self.header.stat()
        self.header.write_bytes(b"#define FIXTURE 2\n")
        os.utime(self.header, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(self.header.stat().st_size, before.st_size)
        self.assertEqual(self.header.stat().st_mtime_ns, before.st_mtime_ns)

    def test_capture_and_verify_do_not_execute_tools_or_rewrite_equal_lock(self):
        captured = self.capture(extra=("--cache", self.cache))
        self.assertEqual(captured["schema_version"], 1)
        self.assertEqual(captured["kind"], "multi-vendor-input-lock")
        self.assertEqual(captured["coverage"], "declared-inputs")
        self.assertIs(captured["cache_eligible"], False)
        self.assertEqual([x["name"] for x in captured["inputs"]], ["compiler", "sdk"])
        original = self.lock.read_bytes()
        os.utime(self.lock, ns=(1_000_000_000, 1_000_000_000))
        before = self.lock.stat().st_mtime_ns
        self.capture(extra=("--cache", self.cache))
        self.verify(extra=("--cache", self.cache))
        self.assertEqual(self.lock.read_bytes(), original)
        self.assertEqual(self.lock.stat().st_mtime_ns, before)

    def test_input_order_and_timestamps_do_not_change_identity(self):
        captured = self.capture()
        os.utime(self.header, ns=(9_000_000_000, 9_000_000_000))
        os.utime(self.sdk, ns=(8_000_000_000, 8_000_000_000))
        second = self.capture(
            path=self.root / "second.lock", inputs=list(reversed(self.inputs))
        )
        self.assertEqual(self.identity(captured), self.identity(second))
        self.verify(inputs=list(reversed(self.inputs)))

    def test_identical_relocated_inputs_verify_against_original_lock(self):
        captured = self.capture()
        relocated = self.root / "relocated"
        relocated.mkdir()
        shutil.copytree(self.sdk, relocated / "sdk", symlinks=True)
        shutil.copy2(self.compiler, relocated / "compiler")
        inputs = [
            ("sdk", "tree", relocated / "sdk"),
            ("compiler", "file", relocated / "compiler"),
        ]
        second = self.capture(path=self.root / "relocated.lock", inputs=inputs)
        self.assertEqual(self.identity(captured), self.identity(second))
        self.assertNotIn(str(self.root), self.lock.read_text())
        self.verify(inputs=inputs)

    def test_changed_bytes_with_restored_mtime_fail_verification_and_repin(self):
        self.capture(extra=("--cache", self.cache))
        before = self.lock.read_bytes()
        self.replace_header_preserving_size_and_mtime()
        self.verify(extra=("--cache", self.cache), success=False)
        self.capture(extra=("--cache", self.cache), success=False)
        self.assertEqual(self.lock.read_bytes(), before)
        new = self.capture(path=self.root / "explicit-new.lock")
        self.assertNotEqual(self.identity(json.loads(before)), self.identity(new))

    def test_tree_shape_and_permissions_are_locked(self):
        self.capture()
        baseline = self.lock.read_bytes()
        variants = (
            (
                "added file",
                lambda: (self.sdk / "new.h").write_bytes(b"new"),
                lambda: (self.sdk / "new.h").unlink(),
            ),
            (
                "removed file",
                self.library.unlink,
                lambda: self.library.write_bytes(b"synthetic library bytes\x00\xff"),
            ),
            (
                "added empty directory",
                lambda: (self.sdk / "new-empty").mkdir(),
                lambda: (self.sdk / "new-empty").rmdir(),
            ),
            (
                "removed empty directory",
                lambda: (self.sdk / "lib" / "empty").rmdir(),
                lambda: (self.sdk / "lib" / "empty").mkdir(),
            ),
            (
                "executable mode",
                lambda: self.compiler.chmod(0o644),
                lambda: self.compiler.chmod(0o755),
            ),
            (
                "non-executable permission",
                lambda: self.header.chmod(0o600),
                lambda: self.header.chmod(0o644),
            ),
        )
        for name, mutate, restore in variants:
            with self.subTest(change=name):
                mutate()
                self.verify(success=False)
                self.assertEqual(self.lock.read_bytes(), baseline)
                restore()
                self.verify()

    def test_empty_tree_has_identity_and_detects_first_added_child(self):
        empty = self.root / "empty-sdk"
        empty.mkdir()
        inputs = [("empty", "tree", empty)]
        self.capture(inputs=inputs)
        self.verify(inputs=inputs)
        (empty / "first").mkdir()
        self.verify(inputs=inputs, success=False)

    def test_file_input_content_is_locked_independently_of_sdk_tree(self):
        self.capture()
        self.compiler.write_bytes(self.compiler.read_bytes() + b"# changed\n")
        self.verify(success=False)

    def test_logical_input_names_and_complete_input_set_are_locked(self):
        self.capture()
        self.verify(
            inputs=[("renamed", "tree", self.sdk), self.inputs[1]], success=False
        )
        self.verify(inputs=self.inputs[:1], success=False)
        self.verify(
            inputs=[*self.inputs, ("another", "file", self.header)], success=False
        )

    def test_contained_file_directory_and_ancestor_symlinks_do_not_recurse(self):
        (self.sdk / "header-link").symlink_to("include/fixture.h")
        (self.sdk / "include-link").symlink_to("include", target_is_directory=True)
        (self.sdk / "include" / "ancestor").symlink_to("..", target_is_directory=True)
        self.capture()
        self.verify()
        self.replace_header_preserving_size_and_mtime()
        self.verify(success=False)

    def test_symlink_into_another_declared_root_is_covered(self):
        other = self.root / "other SDK"
        other.mkdir()
        target = other / "external.h"
        target.write_bytes(b"external original")
        (self.sdk / "external").symlink_to(target)
        inputs = [*self.inputs, ("other", "tree", other)]
        self.capture(inputs=inputs)
        self.verify(inputs=inputs)
        target.write_bytes(b"external modified")
        self.verify(inputs=inputs, success=False)

    def test_escaping_broken_and_cyclic_symlinks_reject_without_lock_creation(self):
        outside = self.root / "undeclared.h"
        outside.write_bytes(b"not declared")
        link = self.sdk / "bad-link"
        for name, target in (
            ("escape", outside),
            ("broken", "absent"),
            ("cycle", "bad-link"),
        ):
            with self.subTest(link=name):
                link.symlink_to(target)
                self.capture(success=False)
                self.assertFalse(self.lock.exists())
                link.unlink()

    def test_symlink_retarget_is_detected_even_when_referent_bytes_are_equal(self):
        second = self.sdk / "include" / "other.h"
        second.write_bytes(self.header.read_bytes())
        link = self.sdk / "selected-header"
        link.symlink_to("include/fixture.h")
        self.capture()
        link.unlink()
        link.symlink_to("include/other.h")
        self.verify(success=False)

    def test_file_symlink_chain_tracks_referent_content(self):
        first = self.root / "compiler-link-one"
        second = self.root / "compiler-link-two"
        first.symlink_to(second.name)
        second.symlink_to(self.compiler.name)
        inputs = [("compiler", "file", first)]
        self.capture(inputs=inputs)
        self.verify(inputs=inputs)
        self.compiler.write_bytes(self.compiler.read_bytes() + b"# changed\n")
        self.verify(inputs=inputs, success=False)

    @unittest.skipUnless(
        hasattr(os, "mkfifo") and hasattr(socket, "AF_UNIX"),
        "POSIX special files required",
    )
    def test_fifo_and_unix_socket_reject_without_blocking(self):
        fifo = self.sdk / "fifo"
        os.mkfifo(fifo)
        self.capture(success=False)
        self.assertFalse(self.lock.exists())
        fifo.unlink()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as endpoint:
            endpoint.bind(str(self.sdk / "socket"))
            self.capture(success=False)
            self.assertFalse(self.lock.exists())

    def test_capture_failures_preserve_preexisting_invalid_output(self):
        original = b"{ not a valid lock\n"
        self.lock.write_bytes(original)
        self.capture(success=False)
        self.assertEqual(self.lock.read_bytes(), original)

    def test_snapshot_requires_an_explicit_fresh_destination(self):
        snapshot = self.root / "snapshot.json"
        self.cli("snapshot", "--output", snapshot)
        original = snapshot.read_bytes()
        self.cli("snapshot", "--output", snapshot, success=False)
        self.replace_header_preserving_size_and_mtime()
        self.cli("snapshot", "--output", snapshot, success=False)
        self.assertEqual(snapshot.read_bytes(), original)
        self.cli("snapshot", "--output", self.root / "new-snapshot.json")

    def test_specs_reject_malformed_duplicate_unknown_and_invalid_fields(self):
        valid = {
            "schema_version": 1,
            "kind": "multi-vendor-input-spec",
            "inputs": [
                {"name": n, "kind": k, "path": str(p)} for n, k, p in self.inputs
            ],
        }
        spec = self.root / "spec.json"
        spec.write_text(json.dumps(valid))
        self.cli("capture", "--lock", self.lock, spec=spec)
        original = self.lock.read_bytes()
        malformed = [
            "{",
            "[]",
            '{"schema_version":1,"schema_version":1,"kind":"multi-vendor-input-spec","inputs":[]}',
            json.dumps(valid).replace('"name": "sdk"', '"name": "sdk", "name": "sdk"'),
        ]
        for key, value in (
            ("schema_version", True),
            ("schema_version", 2),
            ("kind", "other"),
            ("unexpected", 1),
            ("inputs", []),
        ):
            changed = copy.deepcopy(valid)
            changed[key] = value
            malformed.append(json.dumps(changed))
        for key, value in (
            ("name", "../escape"),
            ("kind", "directory"),
            ("path", True),
            ("extra", "value"),
        ):
            changed = copy.deepcopy(valid)
            changed["inputs"][0][key] = value
            malformed.append(json.dumps(changed))
        duplicate = copy.deepcopy(valid)
        duplicate["inputs"].append(duplicate["inputs"][0])
        malformed.append(json.dumps(duplicate))
        for index, contents in enumerate(malformed):
            with self.subTest(case=index):
                spec.write_text(contents)
                self.cli("verify", "--lock", self.lock, spec=spec, success=False)
                self.assertEqual(self.lock.read_bytes(), original)

    def test_spec_relative_paths_resolve_against_spec_location(self):
        spec = self.root / "relative-spec.json"
        spec.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "multi-vendor-input-spec",
                    "inputs": [
                        {"name": n, "kind": k, "path": str(p.relative_to(self.root))}
                        for n, k, p in self.inputs
                    ],
                }
            )
        )
        self.cli("capture", "--lock", self.lock, spec=spec)
        self.cli("verify", "--lock", self.lock, spec=spec)
        self.replace_header_preserving_size_and_mtime()
        self.cli("verify", "--lock", self.lock, spec=spec, success=False)

    def test_invalid_or_duplicate_cli_names_and_wrong_path_kinds_reject(self):
        invalid = (
            [("../escape", "tree", self.sdk)],
            [("sdk", "tree", self.sdk), ("sdk", "file", self.compiler)],
            [("sdk", "file", self.sdk)],
            [("compiler", "tree", self.compiler)],
            [("missing", "file", self.root / "missing")],
        )
        for inputs in invalid:
            with self.subTest(inputs=inputs):
                self.capture(inputs=inputs, success=False)
                self.assertFalse(self.lock.exists())

    def test_lock_rejects_schema_digest_duplicate_and_entry_corruption(self):
        valid = self.capture()
        malformed = [
            "{",
            "[]",
            json.dumps(valid).replace(
                '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
            ),
        ]
        for key, value in (
            ("schema_version", True),
            ("schema_version", 2),
            ("kind", "other"),
            ("global_content_sha256", "0" * 63),
            ("global_content_sha256", "G" * 64),
            ("cache_eligible", True),
            ("coverage", "everything"),
            ("extra", 1),
        ):
            changed = copy.deepcopy(valid)
            changed[key] = value
            malformed.append(json.dumps(changed))
        duplicate = copy.deepcopy(valid)
        duplicate["inputs"].append(duplicate["inputs"][0])
        malformed.append(json.dumps(duplicate))
        changed = copy.deepcopy(valid)
        changed["inputs"][0]["content_sha256"] = "f" * 64
        malformed.append(json.dumps(changed))
        for field, value in (("name", "../escape"), ("kind", "unknown")):
            changed = copy.deepcopy(valid)
            changed["inputs"][0][field] = value
            malformed.append(json.dumps(changed))
        for index, contents in enumerate(malformed):
            with self.subTest(case=index):
                self.lock.write_text(contents)
                self.verify(success=False)
                self.assertEqual(self.lock.read_text(), contents)

    def test_malformed_and_checksum_corrupt_cache_cannot_bless_mutated_input(self):
        self.capture(extra=("--cache", self.cache))
        original_lock = self.lock.read_bytes()
        valid_cache = json.loads(self.cache.read_text())
        self.replace_header_preserving_size_and_mtime()
        corrupt = copy.deepcopy(valid_cache)
        corrupt["untrusted-change"] = "must invalidate the checksum"
        for contents in ("{", "[]", json.dumps(corrupt)):
            with self.subTest(cache=contents[:80]):
                self.cache.write_text(contents)
                self.verify(extra=("--cache", self.cache), success=False)
                self.assertEqual(self.lock.read_bytes(), original_lock)

    def test_full_verification_rehashes_and_rejects_restored_timestamp_change(self):
        self.capture(extra=("--cache", self.cache))
        old_cache = self.cache.read_bytes()
        self.replace_header_preserving_size_and_mtime()
        self.cache.write_bytes(old_cache)
        self.verify(extra=("--cache", self.cache, "--full"), success=False)
        self.header.write_bytes(b"#define FIXTURE 1\n")
        self.verify(extra=("--cache", self.cache, "--full"))

    def test_relative_symlink_spelling_and_relocation_preserve_identity(self):
        link = self.sdk / "header-link"
        link.symlink_to("include/fixture.h")
        original = self.capture()
        link.unlink()
        link.symlink_to("./include/fixture.h")
        self.verify()
        relocated = self.root / "relocated-sdk"
        shutil.copytree(self.sdk, relocated, symlinks=True)
        inputs = [("sdk", "tree", relocated), self.inputs[1]]
        second = self.capture(path=self.root / "relocated-link.lock", inputs=inputs)
        self.assertEqual(self.identity(original), self.identity(second))
        self.verify(inputs=inputs)

    def test_lock_rejects_invalid_entry_paths_modes_sizes_digests_and_order(self):
        valid = self.capture()
        variants = []
        for key, value in (
            ("path", "../escape"),
            ("mode", True),
            ("mode", 0o10000),
            ("size", -1),
            ("size", True),
            ("sha256", "F" * 64),
            ("unknown", 0),
        ):
            changed = copy.deepcopy(valid)
            changed["inputs"][0]["entries"][0][key] = value
            variants.append(json.dumps(changed))
        for entries in ([], [valid["inputs"][0]["entries"][0]] * 2):
            changed = copy.deepcopy(valid)
            changed["inputs"][0]["entries"] = entries
            variants.append(json.dumps(changed))
        changed = copy.deepcopy(valid)
        changed["inputs"][1]["entries"].reverse()
        variants.append(json.dumps(changed))
        variants.append(
            json.dumps(valid).replace('"mode": 493', '"mode": 493, "mode": 493')
        )
        for index, contents in enumerate(variants):
            with self.subTest(case=index):
                self.lock.write_text(contents)
                self.verify(success=False)

    def test_full_verification_bypasses_a_checksum_valid_but_untrusted_cache(self):
        self.capture(extra=("--cache", self.cache))
        cached = json.loads(self.cache.read_text())
        for entry in cached["files"].values():
            entry["sha256"] = "f" * 64
        cached["checksum"] = hashlib.sha256(
            json.dumps(
                cached["files"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ).hexdigest()
        self.cache.write_text(json.dumps(cached))
        # The checksum protects accidental corruption, not hostile rewrites.
        # Full verification must ignore even structurally valid cached hashes.
        self.verify(extra=("--cache", self.cache, "--full"))

    def test_metadata_outputs_cannot_alias_lock_or_declared_input_bytes(self):
        self.capture()
        original = self.lock.read_bytes()
        original_header = self.header.read_bytes()
        alias = self.root / "lock-alias"
        alias.symlink_to(self.lock.name)
        for flag, output in (
            ("--cache", self.lock),
            ("--report", self.lock),
            ("--cache", alias),
            ("--report", self.header),
        ):
            with self.subTest(flag=flag, output=output):
                self.verify(extra=(flag, output), success=False)
                self.assertEqual(self.lock.read_bytes(), original)
                self.assertEqual(self.header.read_bytes(), original_header)

    def test_metadata_outputs_cannot_be_created_inside_a_declared_tree(self):
        for operation, flag in (("capture", "--lock"), ("snapshot", "--output")):
            output = self.sdk / f"{operation}.json"
            with self.subTest(operation=operation):
                self.cli(operation, flag, output, success=False)
                self.assertFalse(output.exists())
        self.capture()
        for flag in ("--cache", "--report"):
            output = self.sdk / (flag[2:] + ".json")
            with self.subTest(flag=flag):
                self.verify(extra=(flag, output), success=False)
                self.assertFalse(output.exists())

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX FIFO required")
    def test_nonregular_lock_and_spec_metadata_reject_without_blocking(self):
        os.mkfifo(self.lock)
        self.verify(success=False)
        self.lock.unlink()
        spec = self.root / "fifo-spec"
        os.mkfifo(spec)
        self.cli("capture", "--lock", self.lock, spec=spec, success=False)
        self.assertFalse(self.lock.exists())

    def test_directory_mutation_between_listing_and_yield_rejects_observation(self):
        original_walk = os.walk
        changed = False

        def mutate_after_listing(*args, **kwargs):
            nonlocal changed
            for item in original_walk(*args, **kwargs):
                if not changed:
                    (self.sdk / "added-after-listing").write_text("racing input")
                    changed = True
                yield item

        with mock.patch.object(
            input_provenance.os, "walk", side_effect=mutate_after_listing
        ), self.assertRaisesRegex(ValueError, "changed"):
            input_provenance.observe_inputs(
                (input_provenance.InputSpec("sdk", "tree", self.sdk),)
            )
        self.assertTrue(changed)
        self.assertFalse(self.lock.exists())

    def test_reports_record_current_locations_separately_from_lock_identity(self):
        capture_report = self.root / "capture-report.json"
        captured = self.capture(extra=("--report", capture_report))
        self.assertNotIn(str(self.root), self.lock.read_text())
        self.assertIn(str(self.sdk), capture_report.read_text())
        verify_report = self.root / "verify-report.json"
        self.verify(extra=("--report", verify_report, "--full"))
        self.assertIn(str(self.sdk), verify_report.read_text())
        self.assertIn(self.identity(captured), verify_report.read_text())


if __name__ == "__main__":
    unittest.main()
