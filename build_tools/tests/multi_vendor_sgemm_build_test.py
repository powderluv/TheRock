# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Isolated configure/install checks using a synthetic AMD SDK, without GPU work."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class MultiVendorSgemmBuildTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cmake = shutil.which("cmake")
        if (
            self.cmake is None
            or shutil.which("c++") is None
            or shutil.which("ninja") is None
        ):
            self.skipTest("CMake, Ninja, and a C++ compiler are required")
        temporary = tempfile.TemporaryDirectory(prefix="therock-sgemm-cmake-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.build = self.root / "build"
        self.sdk = self.root / "sdk"
        self.modules = (
            Path(__file__).resolve().parents[2] / "tests/multi_vendor/modules"
        )
        self.put("include/hip/hip_runtime_api.h")
        self.put("lib/libamdhip64.so")
        self.put("include/rocblas/rocblas.h")
        self.put("lib/librocblas.so")

    def put(self, relative: str) -> Path:
        result = self.sdk / relative
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text("synthetic configure-only input\n")
        return result

    def configure(
        self, enabled: bool | None, *extra: str
    ) -> subprocess.CompletedProcess[str]:
        args = [
            self.cmake,
            "-S",
            str(self.modules),
            "-B",
            str(self.build),
            "-G",
            "Ninja",
            "-DTHEROCK_MODULE_BACKEND=amd",
            "-DTHEROCK_MODULE_TARGET=gfx1201",
            f"-DTHEROCK_MODULE_SDK_ROOT={self.sdk}",
            "-DTHEROCK_MODULE_COMPILER=/bin/true",
            "-DTHEROCK_MODULE_INSTALL_SUBDIR=amd-hip-gfx1201",
            "-DBUILD_TESTING=OFF",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]
        if enabled is not None:
            args.append(f"-DTHEROCK_MODULE_ENABLE_SGEMM={'ON' if enabled else 'OFF'}")
        return subprocess.run(
            [*args, *extra], capture_output=True, text=True, timeout=30
        )

    def assert_configured(self, result: subprocess.CompletedProcess[str]) -> None:
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_default_has_no_blas_requirement_and_enabled_uses_selected_sdk(
        self,
    ) -> None:
        library = self.sdk / "lib/librocblas.so"
        library.unlink()
        self.assert_configured(self.configure(None))
        disabled = json.loads((self.build / "runner-contract-config.json").read_text())
        ninja = (self.build / "build.ninja").read_text()
        self.assertNotIn("-DTHEROCK_MODULE_ENABLE_SGEMM=1", ninja)
        self.assertNotIn("librocblas.so", ninja)
        self.assertFalse(any("blas" in value for value in disabled["capabilities"]))
        failed = self.configure(True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("_blas_library", failed.stdout + failed.stderr)
        self.put("lib/librocblas.so")
        self.assert_configured(self.configure(True))
        enabled = json.loads((self.build / "runner-contract-config.json").read_text())
        self.assertEqual(disabled["contract"], enabled["contract"])
        self.assertIn("blas-provider-rocblas-v1", enabled["capabilities"])
        self.assertIn("blas-sgemm-f32-nn-v1", enabled["capabilities"])
        ninja = (self.build / "build.ninja").read_text()
        self.assertIn("-DTHEROCK_MODULE_ENABLE_SGEMM=1", ninja)
        self.assertIn(str(library), ninja)
        self.assertIn(
            str(self.sdk / "lib"), (self.build / "cmake_install.cmake").read_text()
        )
        self.assertNotIn("/stubs", (self.build / "cmake_install.cmake").read_text())

    def test_provider_symlink_cannot_escape_selected_sdk_or_point_to_stubs(
        self,
    ) -> None:
        library = self.sdk / "lib/librocblas.so"
        external = self.root / "outside.so"
        external.write_bytes(b"outside SDK")
        stub = self.put("lib/stubs/librocblas.so")
        for referent in (external, stub):
            with self.subTest(referent=referent):
                library.unlink()
                library.symlink_to(referent)
                result = self.configure(True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("inside the selected SDK", result.stdout + result.stderr)

    def test_explicit_intel_provider_requires_its_sdk_before_contract_generation(
        self,
    ) -> None:
        result = self.configure(
            True, "-DTHEROCK_MODULE_BACKEND=intel", "-DTHEROCK_MODULE_TARGET=xe2-b70"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "Intel SGEMM requires THEROCK_MODULE_ONEAPI_ROOT",
            " ".join((result.stdout + result.stderr).split()),
        )
        self.assertFalse((self.build / "module_contract_data.h").exists())

    def test_option_changes_cannot_relabel_completed_runner_or_touch_prior_stage(
        self,
    ) -> None:
        # Simulate the previous completed-contract marker, then reconfigure only.
        # The actual install script must reject before attempting any output copy
        # or deleting an existing receipt, even without a parent input guard.
        for initial, changed in ((False, True), (True, False)):
            with self.subTest(initial=initial, changed=changed):
                self.assert_configured(self.configure(initial))
                configured = self.build / "runner-contract-config.json"
                completed = self.build / "runner-contract.json"
                completed.write_bytes(configured.read_bytes())
                stage = self.root / "stage" / "bin/amd-hip-gfx1201"
                stage.mkdir(parents=True, exist_ok=True)
                sentinel = stage / "therock_module_validation"
                receipt = stage / "input-build-receipt.json"
                sentinel.write_bytes(b"previous runner")
                receipt.write_bytes(b"previous provenance receipt")
                self.assert_configured(self.configure(changed))
                result = subprocess.run(
                    [
                        self.cmake,
                        "--install",
                        str(self.build),
                        "--prefix",
                        str(self.root / "stage"),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "Stale completed native module contract",
                    result.stdout + result.stderr,
                )
                self.assertEqual(sentinel.read_bytes(), b"previous runner")
                self.assertEqual(receipt.read_bytes(), b"previous provenance receipt")
                self.assertNotEqual(completed.read_bytes(), configured.read_bytes())


if __name__ == "__main__":
    unittest.main()
