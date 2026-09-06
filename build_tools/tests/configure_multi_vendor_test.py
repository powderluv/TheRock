# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.gpu_targets import parse_gpu_target
from configure_multi_vendor import configure_targets


_SCRIPT = Path(__file__).parent.parent / "configure_multi_vendor.py"
_REFERENCE_TARGETS = (
    "amd:hip:gfx942:xnack+:sramecc-",
    "nvidia:cuda:sm_120",
    "intel:level-zero:xe2-b70",
)


class ConfigureMultiVendorTest(unittest.TestCase):
    def test_manifest_round_trips_canonical_target_selection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _, manifest_path = configure_targets(_REFERENCE_TARGETS, Path(temp_dir))
            manifest = json.loads(manifest_path.read_text())

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["kind"], "target-selection")
        self.assertEqual(len(manifest["targets"]), 3)
        for requested, recorded in zip(_REFERENCE_TARGETS, manifest["targets"]):
            with self.subTest(requested=requested):
                target = parse_gpu_target(requested)
                self.assertEqual(parse_gpu_target(recorded["id"]), target)
                self.assertEqual(recorded["slug"], target.slug)
                self.assertEqual(recorded["vendor"], target.vendor)
                self.assertEqual(recorded["backend"], target.backend)
                self.assertEqual(recorded["processor"], target.processor)
                self.assertEqual(recorded["qualification"], "unvalidated")
        amd = manifest["targets"][0]
        self.assertEqual(amd["id"], "amd:hip:gfx942:sramecc-:xnack+")
        self.assertEqual(
            amd["features"],
            [{"name": "sramecc", "enabled": False}, {"name": "xnack", "enabled": True}],
        )
        self.assertEqual(amd["cmake_hip_architecture"], "gfx942:sramecc-:xnack+")
        self.assertEqual(amd["compiler_target"], "gfx942:sramecc-:xnack+")
        self.assertEqual(amd["payload_types"], ["hsaco"])
        nvidia = manifest["targets"][1]
        self.assertEqual(nvidia["cmake_hip_platform"], "nvidia")
        self.assertEqual(nvidia["cmake_hip_architecture"], "120")
        self.assertEqual(nvidia["compiler_target"], "sm_120")
        self.assertEqual(nvidia["payload_types"], ["cubin", "ptx"])

    def test_intel_selection_delegates_compiler_target_and_stays_unvalidated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cmake_path, manifest_path = configure_targets(
                ("intel:level-zero:xe2-b70",), Path(temp_dir)
            )
            cmake = cmake_path.read_text()
            target = json.loads(manifest_path.read_text())["targets"][0]

        self.assertEqual(target["processor"], "xe2-b70")
        self.assertEqual(target["cmake_hip_platform"], "spirv")
        self.assertIsNone(target["cmake_hip_architecture"])
        self.assertIsNone(target["compiler_target"])
        self.assertEqual(target["payload_types"], ["spirv"])
        self.assertTrue(target["experimental"])
        self.assertEqual(target["qualification"], "unvalidated")
        prefix = "THEROCK_MULTI_VENDOR_intel-level-zero-xe2-b70"
        self.assertIn(f'set({prefix}_HIP_ARCHITECTURE "")', cmake)
        self.assertIn(f'set({prefix}_COMPILER_TARGET "")', cmake)
        self.assertNotIn("spirv64", cmake)

    def test_output_bytes_are_deterministic_across_locations_and_feature_order(self):
        alternate = (
            "amd:hip:gfx942:sramecc-:xnack+",
            "nvidia:cuda:sm_120",
            "intel:level-zero:xe2-b70",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            first = configure_targets(_REFERENCE_TARGETS, Path(temp_dir) / "first")
            contents = tuple(path.read_bytes() for path in first)
            second = configure_targets(alternate, Path(temp_dir) / "second")
            self.assertEqual(contents, tuple(path.read_bytes() for path in second))
            repeated = configure_targets(_REFERENCE_TARGETS, Path(temp_dir) / "first")
            self.assertEqual(contents, tuple(path.read_bytes() for path in repeated))
            for content in contents:
                self.assertTrue(content.endswith(b"\n"))
                self.assertNotIn(temp_dir.encode(), content)

    def test_invalid_selection_never_creates_output_directory(self):
        selections = (
            (),
            ("amd:hip:gfx1201", "nvidia:hip:sm_120"),
            ("nvidia:cuda:sm_120", 'amd:hip:gfx942");message(FATAL_ERROR "injected'),
            ("amd:hip:gfx1201", "amd:hip:gfx1201"),
            ("amd:hip:gfx942:xnack+:sramecc-", "amd:hip:gfx942:sramecc-:xnack+"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "new" / "generated"
            for selection in selections:
                with self.subTest(selection=selection), self.assertRaises(ValueError):
                    configure_targets(selection, output_dir)
                self.assertFalse(output_dir.parent.exists())

    def test_invalid_selection_preserves_both_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            paths = configure_targets(_REFERENCE_TARGETS, output_dir)
            before = tuple(path.read_bytes() for path in paths)
            with self.assertRaises(ValueError):
                configure_targets(
                    ("amd:hip:gfx1201", "intel:level-zero:../escape"), output_dir
                )
            self.assertEqual(before, tuple(path.read_bytes() for path in paths))
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                ["gpu_targets.json", "targets.cmake"],
            )

    def test_staging_failure_keeps_existing_outputs_and_cleans_temporary_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            paths = configure_targets(_REFERENCE_TARGETS, output_dir)
            before = tuple(path.read_bytes() for path in paths)
            original = tempfile.NamedTemporaryFile
            calls = 0

            def fail_second_staging_file(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("Simulated staging failure")
                return original(*args, **kwargs)

            with mock.patch(
                "configure_multi_vendor.tempfile.NamedTemporaryFile",
                side_effect=fail_second_staging_file,
            ), self.assertRaisesRegex(OSError, "Simulated staging failure"):
                configure_targets(("amd:hip:gfx1201",), output_dir)
            self.assertEqual(before, tuple(path.read_bytes() for path in paths))
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                ["gpu_targets.json", "targets.cmake"],
            )

    @unittest.skipUnless(shutil.which("cmake"), "cmake is not installed")
    def test_generated_cmake_loads_actual_variables(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "directory with spaces"
            cmake_path, _ = configure_targets(_REFERENCE_TARGETS, output_dir)
            script = Path(temp_dir) / "verify.cmake"
            checks = (
                (
                    "THEROCK_MULTI_VENDOR_TARGET_KEYS",
                    "amd-hip-gfx942-sramecc-off-xnack-on;nvidia-cuda-sm120;intel-level-zero-xe2-b70",
                ),
                (
                    "THEROCK_MULTI_VENDOR_amd-hip-gfx942-sramecc-off-xnack-on_ID",
                    "amd:hip:gfx942:sramecc-:xnack+",
                ),
                (
                    "THEROCK_MULTI_VENDOR_amd-hip-gfx942-sramecc-off-xnack-on_HIP_ARCHITECTURE",
                    "gfx942:sramecc-:xnack+",
                ),
                ("THEROCK_MULTI_VENDOR_nvidia-cuda-sm120_VENDOR", "nvidia"),
                ("THEROCK_MULTI_VENDOR_nvidia-cuda-sm120_BACKEND", "cuda"),
                ("THEROCK_MULTI_VENDOR_nvidia-cuda-sm120_PROCESSOR", "sm_120"),
                ("THEROCK_MULTI_VENDOR_nvidia-cuda-sm120_HIP_PLATFORM", "nvidia"),
                ("THEROCK_MULTI_VENDOR_nvidia-cuda-sm120_HIP_ARCHITECTURE", "120"),
                ("THEROCK_MULTI_VENDOR_nvidia-cuda-sm120_COMPILER_TARGET", "sm_120"),
                ("THEROCK_MULTI_VENDOR_intel-level-zero-xe2-b70_HIP_PLATFORM", "spirv"),
                ("THEROCK_MULTI_VENDOR_intel-level-zero-xe2-b70_HIP_ARCHITECTURE", ""),
                ("THEROCK_MULTI_VENDOR_intel-level-zero-xe2-b70_COMPILER_TARGET", ""),
            )
            lines = [f"include([[{cmake_path.as_posix()}]])"]
            for variable, expected in checks:
                lines.extend(
                    (
                        f"if(NOT DEFINED {variable})",
                        f'  message(FATAL_ERROR "Missing {variable}")',
                        "endif()",
                        f'if(NOT "${{{variable}}}" STREQUAL "{expected}")',
                        f'  message(FATAL_ERROR "Unexpected value for {variable}")',
                        "endif()",
                    )
                )
            script.write_text("\n".join(lines) + "\n")
            result = subprocess.run(
                ["cmake", "-P", str(script)], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cli_succeeds_and_reports_validation_errors_without_tracebacks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "selected"
            command = [
                sys.executable,
                str(_SCRIPT),
                "--output-dir",
                str(output_dir),
                "--targets",
            ]
            success = subprocess.run(
                command + list(_REFERENCE_TARGETS), capture_output=True, text=True
            )
            self.assertEqual(success.returncode, 0, success.stdout + success.stderr)
            before = (output_dir / "gpu_targets.json").read_bytes()
            failure = subprocess.run(
                command + ["amd:cuda:gfx1201"], capture_output=True, text=True
            )
            self.assertEqual(failure.returncode, 2)
            self.assertIn("requires backend", failure.stderr)
            self.assertNotIn("Traceback", failure.stderr)
            self.assertEqual((output_dir / "gpu_targets.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
