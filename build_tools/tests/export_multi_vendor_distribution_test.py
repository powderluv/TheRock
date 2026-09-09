# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU-only selective distribution export: real packs, inert native runners."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPO_ROOT / "build_tools"
KPACK = REPO_ROOT / "rocm-systems/shared/kpack/python"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(KPACK))

from _therock_utils.gpu_targets import parse_gpu_target
from _therock_utils.input_provenance import InputSpec, capture_lock
from _therock_utils.module_contract import runner_description, validation_contract
from _therock_utils.payload_catalog import (
    PayloadInput,
    create_pack,
    extract_verified_payload,
    load_catalog,
)
from _therock_utils.runner_registry import RunnerEntry, RunnerRegistry, load_registry
import export_multi_vendor_distribution as exporter
from export_multi_vendor_distribution import export_distribution, verify_distribution

AMD = "amd:hip:gfx1201"
NVIDIA = "nvidia:cuda:sm_120"
SM90 = "nvidia:cuda:sm_90"
INTEL = "intel:level-zero:xe2-b70"
TARGETS = (AMD, NVIDIA, SM90, INTEL)
REGISTRY = "share/therock/packs/runners.json"
PROVENANCE = "share/therock/packs/input-provenance.json"
RUNTIME_MANIFEST = "share/therock/python/runtime-manifest.json"
MANIFEST = "share/therock/distribution-manifest.json"
SCRIPT = TOOLS / "export_multi_vendor_distribution.py"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _payload(module: str, target: str, payload_type: str) -> bytes:
    return f"CPU archive fixture: {module}/{target}/{payload_type}".encode()


def _tree(root: Path) -> dict[str, tuple[bytes | str, int, int]]:
    result = {}
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            data = os.readlink(path)
        elif stat.S_ISREG(info.st_mode):
            data = path.read_bytes()
        else:
            continue
        result[path.relative_to(root).as_posix()] = (
            data,
            stat.S_IMODE(info.st_mode),
            info.st_mtime_ns,
        )
    return result


class ExportMultiVendorDistributionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="therock-export-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "exported"
        self.native_execution = self.root / "runner-was-executed"
        fixture_tool = self.root / "inert-tool"
        fixture_tool.write_bytes(b"Declared input fixture; never executed")
        second_tool = self.root / "second-inert-tool"
        second_tool.write_bytes(b"Second declared input; never executed")
        capture_lock(
            (
                InputSpec("fixture-tool-z", "file", second_tool),
                InputSpec("fixture-tool", "file", fixture_tool),
            ),
            self.root / "original-inputs.lock.json",
            report_path=self.source / PROVENANCE,
        )
        self.provenance = (self.source / PROVENANCE).read_bytes()
        subprocess.run(
            [
                sys.executable,
                str(TOOLS / "stage_multi_vendor_runtime.py"),
                "build",
                "--tools-dir",
                str(TOOLS),
                "--kpack-python-dir",
                str(KPACK),
                "--output-dir",
                str(self.source),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        runners = []
        for identity in TARGETS:
            target = parse_gpu_target(identity)
            path = self.source / f"bin/{target.slug}/therock_module_validation"
            path.parent.mkdir(parents=True)
            body = (
                "#!/usr/bin/env python3\nfrom pathlib import Path\n"
                f"Path({str(self.native_execution)!r}).write_text('unexpected query')\n"
                "raise SystemExit(99)\n"
            ).encode()
            path.write_bytes(body)
            path.chmod(0o755)
            description = runner_description(target.vendor)
            _write_json(path.with_name("runner-contract.json"), description.record())
            path.with_name("input-build-receipt.json").write_bytes(self.provenance)
            runners.append(
                RunnerEntry(
                    target,
                    path.relative_to(self.source).as_posix(),
                    hashlib.sha256(body).hexdigest(),
                    description,
                )
            )
        catalogs = []
        self.inputs = {}
        for module in ("saxpy", "relu"):
            values = []
            for identity in TARGETS:
                for fmt in parse_gpu_target(identity).payload_types:
                    values.append(
                        PayloadInput(
                            f"validation/{module}",
                            identity,
                            fmt,
                            (f"therock_module_{module}",),
                            _payload(module, identity, fmt),
                            validation_contract(),
                        )
                    )
            self.inputs[module] = values
            path = create_pack(
                self.source / "share/therock/packs" / module,
                f"validation-{module}",
                values,
            )
            catalogs.append(path.relative_to(self.source).as_posix())
        _write_json(
            self.source / REGISTRY,
            RunnerRegistry(tuple(runners), tuple(catalogs)).record(),
        )

    def tearDown(self):
        self.assertFalse(
            self.native_execution.exists(), "Export must not execute native runners"
        )

    def _replace_pack(self, source, module, values):
        directory = source / "share/therock/packs" / module
        shutil.rmtree(directory)
        create_pack(directory, f"validation-{module}", values)

    def _copy_source(self, suffix):
        copied = self.root / ("source-" + suffix)
        shutil.copytree(self.source, copied)
        return copied

    def _assert_rejected(self, source=None, targets=(AMD,), output=None):
        source = source or self.source
        output = output or self.output
        before = _tree(source)
        destination_before = _tree(output) if output.exists() else None
        with self.assertRaises((ValueError, OSError)):
            export_distribution(source, output, targets)
        self.assertEqual(_tree(source), before)
        if destination_before is None:
            self.assertFalse(output.exists())
        else:
            self.assertEqual(_tree(output), destination_before)

    def test_exact_target_subsets_preserve_payloads_contracts_and_original_receipts(
        self,
    ):
        source_before = _tree(self.source)
        for index, selection in enumerate(
            ((AMD,), (NVIDIA,), (AMD, NVIDIA), (INTEL,), (SM90, NVIDIA))
        ):
            with self.subTest(selection=selection):
                output = self.root / f"subset-{index}"
                manifest = export_distribution(self.source, output, selection)
                self.assertEqual(manifest, verify_distribution(output))
                self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o755)
                self.assertEqual(manifest["selected_targets"], sorted(selection))
                self.assertEqual(
                    manifest["kind"], "selective-multi-vendor-distribution"
                )
                self.assertIs(manifest["cache_eligible"], False)
                self.assertNotIn(str(self.source), json.dumps(manifest))
                registry = load_registry(output / REGISTRY)
                self.assertEqual(
                    [entry.target.canonical_id for entry in registry.runners],
                    sorted(selection),
                )
                self.assertEqual(
                    {path.name for path in (output / "bin").iterdir()},
                    {parse_gpu_target(value).slug for value in selection},
                )
                catalogs = [output / relative for relative in registry.catalogs]
                observed = {
                    entry.key
                    for path in catalogs
                    for pack in load_catalog(path).packs
                    for entry in pack.entries
                }
                expected = {
                    (f"validation/{module}", identity, fmt)
                    for module in ("saxpy", "relu")
                    for identity in selection
                    for fmt in parse_gpu_target(identity).payload_types
                }
                self.assertEqual(observed, expected)
                for module, identity, fmt in sorted(expected):
                    selected = extract_verified_payload(catalogs, module, identity, fmt)
                    self.assertEqual(
                        selected.data, _payload(module.split("/")[-1], identity, fmt)
                    )
                    self.assertEqual(selected.entry.contract, validation_contract())
                self.assertEqual((output / PROVENANCE).read_bytes(), self.provenance)
                self.assertEqual(
                    (output / RUNTIME_MANIFEST).read_bytes(),
                    (self.source / RUNTIME_MANIFEST).read_bytes(),
                )
                for entry in registry.runners:
                    for name in (
                        "therock_module_validation",
                        "runner-contract.json",
                        "input-build-receipt.json",
                    ):
                        relative = Path(entry.path).with_name(name)
                        self.assertEqual(
                            (output / relative).read_bytes(),
                            (self.source / relative).read_bytes(),
                        )
                for record in json.loads((self.source / RUNTIME_MANIFEST).read_text())[
                    "outputs"
                ]:
                    self.assertEqual(
                        (output / record["path"]).read_bytes(),
                        (self.source / record["path"]).read_bytes(),
                    )
                self.assertNotIn("qualified", json.dumps(manifest).lower())
        self.assertEqual(_tree(self.source), source_before)

    def test_reordered_targets_and_repeated_exports_have_identical_bytes_and_modes(
        self,
    ):
        first = self.root / "first"
        second = self.root / "second"
        export_distribution(self.source, first, (AMD, NVIDIA))
        export_distribution(self.source, second, (NVIDIA, AMD))
        first_files = {name: value[:2] for name, value in _tree(first).items()}
        second_files = {name: value[:2] for name, value in _tree(second).items()}
        self.assertEqual(first_files, second_files)

    def test_unknown_duplicate_empty_and_noncanonical_selections_are_rejected(self):
        for selection in (
            (),
            (AMD, AMD),
            ("nvidia:cuda:sm_80",),
            ("gfx1201",),
            ("amd:hip:gfx942:xnack+:sramecc-",),
        ):
            with self.subTest(selection=selection):
                self._assert_rejected(targets=selection)

    def test_missing_module_or_advertised_format_coverage_is_rejected(self):
        for label, predicate in (
            ("missing-module", lambda item: item.target != NVIDIA),
            (
                "missing-ptx",
                lambda item: not (item.target == NVIDIA and item.payload_type == "ptx"),
            ),
        ):
            source = self._copy_source(label)
            self._replace_pack(
                source,
                "relu",
                [item for item in self.inputs["relu"] if predicate(item)],
            )
            with self.subTest(label=label):
                self._assert_rejected(source, (NVIDIA,))

    def test_duplicate_across_packs_and_schema_one_payloads_are_rejected(self):
        source = self._copy_source("ambiguous")
        duplicate = create_pack(
            source / "share/therock/packs/duplicate",
            "duplicate",
            [self.inputs["saxpy"][0]],
        )
        registry = json.loads((source / REGISTRY).read_text())
        registry["catalogs"].append(duplicate.relative_to(source).as_posix())
        _write_json(source / REGISTRY, registry)
        self._assert_rejected(source)
        source = self._copy_source("legacy")
        self._replace_pack(
            source,
            "saxpy",
            [
                PayloadInput(
                    item.module,
                    item.target,
                    item.payload_type,
                    item.entry_points,
                    item.data,
                )
                for item in self.inputs["saxpy"]
            ],
        )
        self._assert_rejected(source)

    def test_tampered_runner_runtime_pack_and_receipts_fail_before_publication(self):
        runner_dir = f"bin/{parse_gpu_target(AMD).slug}"
        mutations = {
            "runner": (f"{runner_dir}/therock_module_validation", b"tampered runner"),
            "description": (f"{runner_dir}/runner-contract.json", b"{}"),
            "receipt": (f"{runner_dir}/input-build-receipt.json", b"{}"),
            "provenance": (PROVENANCE, b"{}"),
            "runtime": (
                "share/therock/python/therock_multi_vendor/__init__.py",
                b"# altered",
            ),
            "runtime-manifest": (RUNTIME_MANIFEST, b"{}"),
            "pack": (
                "share/therock/packs/relu/validation-relu.kpack",
                b"tampered pack",
            ),
            "catalog": ("share/therock/packs/relu/catalog.json", b"{}"),
        }
        for label, (relative, value) in mutations.items():
            with self.subTest(label=label):
                source = self._copy_source(label)
                (source / relative).write_bytes(value)
                self._assert_rejected(source)

    def test_unselected_runner_is_not_required_but_all_pack_hashes_are_checked(self):
        unselected = (
            self.source
            / f"bin/{parse_gpu_target(INTEL).slug}/therock_module_validation"
        )
        unselected.write_bytes(b"unavailable unselected runner")
        export_distribution(self.source, self.output, (AMD,))
        self.assertEqual(verify_distribution(self.output)["selected_targets"], [AMD])
        source = self._copy_source("bad-unselected-pack")
        # Move an existing Intel-only variant into a separate unique pack.
        # AMD's module/format coverage remains complete, so only its bad hash
        # makes this irrelevant-to-selection archive reject the export.
        item = next(item for item in self.inputs["saxpy"] if item.target == INTEL)
        self._replace_pack(
            source, "saxpy", [value for value in self.inputs["saxpy"] if value != item]
        )
        catalog = create_pack(source / "share/therock/packs/extra", "extra", [item])
        (catalog.parent / "extra.kpack").write_bytes(b"bad hash")
        registry = json.loads((source / REGISTRY).read_text())
        registry["catalogs"].append(catalog.relative_to(source).as_posix())
        _write_json(source / REGISTRY, registry)
        self._assert_rejected(source, output=self.root / "bad-export")

    def test_existing_destination_and_overlapping_paths_are_not_changed(self):
        self.output.mkdir()
        (self.output / "sentinel").write_text("previous distribution")
        self._assert_rejected()
        self._assert_rejected(output=self.source / "nested-output")
        self._assert_rejected(output=self.source)
        self._assert_rejected(output=self.root)
        alias = self.root / "source-alias"
        alias.symlink_to(self.source, target_is_directory=True)
        self._assert_rejected(output=alias / "nested-output")

    def test_symlink_and_special_source_files_are_rejected_without_following_them(self):
        source = self._copy_source("symlink")
        runner = source / f"bin/{parse_gpu_target(AMD).slug}/therock_module_validation"
        external = self.root / "outside-runner"
        external.write_bytes(runner.read_bytes())
        runner.unlink()
        runner.symlink_to(external)
        self._assert_rejected(source)
        source = self._copy_source("fifo")
        os.mkfifo(source / "unrecognized-special-file")
        # Subprocess timeout also proves the special file is rejected, not read.
        result = self._cli(
            "export",
            "--source-dist",
            str(source),
            "--output-dir",
            str(self.root / "fifo-output"),
            "--target",
            AMD,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.root / "fifo-output").exists())

    def test_publication_race_preserves_new_destination_and_cleans_private_staging(
        self,
    ):
        before_source = _tree(self.source)
        before_names = {path.name for path in self.root.iterdir()}
        publish = exporter._publish_directory

        def concurrent_destination(staged, destination):
            destination.mkdir()
            (destination / "racing-publisher").write_bytes(b"retain this directory")
            publish(staged, destination)

        with mock.patch.object(exporter, "_publish_directory", concurrent_destination):
            with self.assertRaises((ValueError, OSError)):
                export_distribution(self.source, self.output, (AMD,))
        self.assertEqual(
            (self.output / "racing-publisher").read_bytes(), b"retain this directory"
        )
        self.assertEqual(set(_tree(self.output)), {"racing-publisher"})
        self.assertEqual(_tree(self.source), before_source)
        self.assertEqual(
            {path.name for path in self.root.iterdir()},
            before_names | {self.output.name},
        )

    def test_verifier_rejects_added_missing_mutated_and_mode_changed_outputs(self):
        export_distribution(self.source, self.output, (AMD, NVIDIA))
        for label in ("added", "missing", "mutated", "mode", "symlink", "manifest"):
            with self.subTest(label=label):
                output = self.root / ("verify-" + label)
                shutil.copytree(self.output, output)
                path = output / "share/therock/python/requirements.txt"
                if label == "added":
                    (output / "unexpected-output").write_text("not inventoried")
                elif label == "missing":
                    path.unlink()
                elif label == "mutated":
                    path.write_bytes(path.read_bytes() + b"\n# changed\n")
                elif label == "mode":
                    path.chmod(0o755)
                elif label == "symlink":
                    path.unlink()
                    path.symlink_to(self.output / path.relative_to(output))
                else:
                    document = json.loads((output / MANIFEST).read_text())
                    document["selected_targets"] = [INTEL]
                    _write_json(output / MANIFEST, document)
                before = _tree(output)
                with self.assertRaises((ValueError, OSError)):
                    verify_distribution(output)
                self.assertEqual(before, _tree(output))

    def test_self_consistent_manifest_cannot_escape_paths_or_relabel_targets(self):
        export_distribution(self.source, self.output, (AMD,))
        original = json.loads((self.output / MANIFEST).read_text())
        for label in ("escape", "target", "cache-claim"):
            with self.subTest(label=label):
                document = json.loads(json.dumps(original))
                if label == "escape":
                    document["files"][0]["path"] = "../outside"
                elif label == "target":
                    document["selected_targets"] = [NVIDIA]
                else:
                    document["cache_eligible"] = True
                body = {
                    key: value
                    for key, value in document.items()
                    if key != "content_sha256"
                }
                document["content_sha256"] = hashlib.sha256(
                    (json.dumps(body, sort_keys=True, indent=2) + "\n").encode()
                ).hexdigest()
                _write_json(self.output / MANIFEST, document)
                with self.assertRaises((ValueError, OSError)):
                    verify_distribution(self.output)

    def test_exported_client_rejects_dist_root_overrides_before_source_resolution(self):
        wrapper = REPO_ROOT / "tests/multi_vendor/modules/exported_session_client.py"
        before = _tree(self.root)
        for override in (
            ("--dist-root", "/outside"),
            ("--dist-root=/outside",),
            ("--dist", "/outside"),
            ("--dist=/outside",),
        ):
            with self.subTest(override=override):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(wrapper),
                        "--source-dist",
                        str(self.root / "intentionally-missing-source"),
                        "--target",
                        AMD,
                        "--",
                        "--target",
                        AMD,
                        "--format",
                        "hsaco",
                        *override,
                    ],
                    cwd=self.root,
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("fixed", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(_tree(self.root), before)

    def test_walk_permission_errors_are_not_silently_ignored_by_export_or_verify(self):
        export_distribution(self.source, self.output, (AMD,))
        real_walk = os.walk

        def unreadable_subtree(path, *arguments, onerror=None, **keywords):
            if onerror is not None:
                onerror(PermissionError(13, "fixture unreadable subtree", str(path)))
            yield from real_walk(path, *arguments, **keywords)

        destination = self.root / "unreadable-export"
        before_source = _tree(self.source)
        before_output = _tree(self.output)
        for operation in (
            lambda: export_distribution(self.source, destination, (AMD,)),
            lambda: verify_distribution(self.output),
        ):
            with mock.patch.object(exporter.os, "walk", unreadable_subtree):
                with self.assertRaises(OSError):
                    operation()
            self.assertFalse(destination.exists())
            self.assertEqual(_tree(self.source), before_source)
            self.assertEqual(_tree(self.output), before_output)

    def test_consistently_replaced_receipts_still_require_valid_ordered_aggregate(self):
        original = json.loads(self.provenance)
        self.assertEqual(len(original["inputs"]), 2)
        for label in ("wrong-aggregate", "reordered-inputs"):
            with self.subTest(label=label):
                source = self._copy_source(label)
                document = json.loads(json.dumps(original))
                if label == "wrong-aggregate":
                    document["global_content_sha256"] = "0" * 64
                else:
                    document["inputs"].reverse()
                    # Internally hash the wrong ordering, so sorting enforcement
                    # is checked independently of a plain bad aggregate digest.
                    identities = [
                        {key: item[key] for key in ("name", "kind", "content_sha256")}
                        for item in document["inputs"]
                    ]
                    document["global_content_sha256"] = hashlib.sha256(
                        json.dumps(
                            identities, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest()
                paths = [source / PROVENANCE]
                paths.extend(source.glob("bin/*/input-build-receipt.json"))
                for path in paths:
                    _write_json(path, document)
                self.assertEqual(len({path.read_bytes() for path in paths}), 1)
                self._assert_rejected(source)

    def test_reexport_is_explicitly_rejected_without_changing_verified_source(self):
        export_distribution(self.source, self.output, (AMD, NVIDIA))
        verify_distribution(self.output)
        before = _tree(self.output)
        destination = self.root / "reexported"
        with self.assertRaisesRegex(ValueError, "(?i)re-export|original built"):
            export_distribution(self.output, destination, (AMD,))
        self.assertFalse(destination.exists())
        self.assertEqual(_tree(self.output), before)
        verify_distribution(self.output)

    def _cli(self, *arguments):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(KPACK) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        return subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            cwd=self.root,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
        )

    def test_cli_exports_and_verifies_and_reports_invalid_input_without_traceback(self):
        result = self._cli(
            "export",
            "--source-dist",
            str(self.source),
            "--output-dir",
            str(self.output),
            "--target",
            NVIDIA,
            "--target",
            AMD,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self._cli("verify", "--dist-root", str(self.output))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            verify_distribution(self.output)["selected_targets"], [AMD, NVIDIA]
        )
        result = self._cli(
            "export",
            "--source-dist",
            str(self.source),
            "--output-dir",
            str(self.root / "invalid-cli"),
            "--target",
            "nvidia:cuda:sm_80",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.root / "invalid-cli").exists())


if __name__ == "__main__":
    unittest.main()
