# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configure-only Intel SGEMM guards with synthetic SDKs and no GPU execution.

The positive fixture supplies a forced compiler identity and placeholder SDK
files. It checks build wiring, not SYCL compilation or oneMKL availability.
Actual imported compiler and library compatibility require a real SDK build.
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(
    shutil.which("cmake") and shutil.which("ninja") and shutil.which("c++"),
    "CMake, Ninja, and a C++ compiler are required",
)
class IntelSgemmBuildTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="therock-intel-sgemm-cmake-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.build = self.root / "build"
        self.level_zero = self.root / "level-zero"
        self.oneapi = self.root / "oneapi"
        self.mkl = self.oneapi / "mkl/2026.1"
        self.compiler = self.oneapi / "compiler/2026.1"
        for root, relative in (
            (self.level_zero, "include/level_zero/ze_api.h"),
            (self.level_zero, "lib/libze_loader.so"),
            (self.compiler, "include/sycl/sycl.hpp"),
            (self.compiler, "lib/libsycl.so"),
            (self.mkl, "include/oneapi/mkl.hpp"),
            (self.mkl, "include/mkl.h"),
            (self.mkl, "lib/libmkl_sycl_blas.so"),
            (self.mkl, "lib/libmkl_intel_ilp64.so"),
            (self.mkl, "lib/libmkl_sequential.so"),
            (self.mkl, "lib/libmkl_core.so"),
            (self.oneapi, "umf/latest/lib/libumf.so"),
            (self.oneapi, "tcm/latest/lib/libhwloc.so"),
        ):
            self.put(root / relative)
        self.icpx = self.compiler / "bin/icpx"
        self.icpx.parent.mkdir(parents=True)
        self.icpx.write_text(f'#!/bin/sh\nexec "{shutil.which("c++")}" "$@"\n')
        self.icpx.chmod(0o755)
        self.toolchain = self.root / "configure-only-toolchain.cmake"
        self.toolchain.write_text(
            f"set(CMAKE_CXX_COMPILER [[{self.icpx}]])\n"
            "set(CMAKE_CXX_COMPILER_ID IntelLLVM)\n"
            "set(CMAKE_CXX_COMPILER_VERSION 2026.1.0)\n"
            "set(CMAKE_CXX_COMPILER_ID_RUN TRUE)\n"
            "set(CMAKE_CXX_COMPILER_FORCED TRUE)\n"
            "set(CMAKE_CXX_COMPILER_WORKS TRUE)\n"
            "set(CMAKE_CXX_COMPILE_FEATURES cxx_std_17)\n"
            "set(CMAKE_EXECUTABLE_FORMAT ELF)\n"
        )

    def put(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic configure-only input\n")
        return path

    def configure(
        self, enabled: bool, *extra: str, forced_compiler: bool = True
    ) -> subprocess.CompletedProcess[str]:
        args = [
            "cmake",
            "-S",
            str(REPO_ROOT / "tests/multi_vendor/modules"),
            "-B",
            str(self.build),
            "-GNinja",
            "-DTHEROCK_MODULE_BACKEND=intel",
            "-DTHEROCK_MODULE_TARGET=xe2-b70",
            "-DTHEROCK_MODULE_INSTALL_SUBDIR=intel-level-zero-xe2-b70",
            f"-DTHEROCK_MODULE_SDK_ROOT={self.level_zero}",
            "-DTHEROCK_MODULE_COMPILER=/bin/true",
            "-DTHEROCK_MODULE_SPIRV_TRANSLATOR=/bin/true",
            "-DTHEROCK_MODULE_SPIRV_VALIDATOR=/bin/true",
            f"-DTHEROCK_MODULE_ENABLE_SGEMM={'ON' if enabled else 'OFF'}",
            f"-DTHEROCK_MODULE_ONEAPI_ROOT={self.oneapi}",
            f"-DTHEROCK_MODULE_ONEMKL_ROOT={self.mkl}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DBUILD_TESTING=OFF",
        ]
        if forced_compiler:
            args.append(f"-DCMAKE_TOOLCHAIN_FILE={self.toolchain}")
        return subprocess.run(
            [*args, *extra], capture_output=True, text=True, timeout=30
        )

    def assert_success(self, result: subprocess.CompletedProcess[str]) -> None:
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def assert_failure(
        self, result: subprocess.CompletedProcess[str], expected: str
    ) -> None:
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(expected, " ".join((result.stdout + result.stderr).split()))

    def test_default_intel_worker_needs_no_oneapi_sdk_or_sycl_flags(self) -> None:
        self.assert_success(
            self.configure(
                False,
                "-DTHEROCK_MODULE_ONEAPI_ROOT=",
                "-DTHEROCK_MODULE_ONEMKL_ROOT=",
                forced_compiler=False,
            )
        )
        ninja = (self.build / "build.ninja").read_text()
        description = json.loads(
            (self.build / "runner-contract-config.json").read_text()
        )
        self.assertFalse(any("blas" in cap for cap in description["capabilities"]))
        self.assertNotIn("-fsycl", ninja)
        self.assertNotIn("mkl_sycl_blas", ninja)
        self.assertNotIn(str(self.oneapi), ninja)

    def test_missing_sdk_rejected_before_contract_generation(self) -> None:
        self.assert_failure(
            self.configure(
                True, "-DTHEROCK_MODULE_ONEAPI_ROOT=", forced_compiler=False
            ),
            "Intel SGEMM requires THEROCK_MODULE_ONEAPI_ROOT",
        )
        self.assertFalse((self.build / "module_contract_data.h").exists())

    def test_compiler_and_mkl_cannot_resolve_outside_imported_sdk(self) -> None:
        outside = self.root / "outside-mkl"
        outside.mkdir()
        self.assert_failure(
            self.configure(True, f"-DTHEROCK_MODULE_ONEMKL_ROOT={outside}"),
            "must resolve inside THEROCK_MODULE_ONEAPI_ROOT",
        )
        shutil.rmtree(self.build)
        self.assert_failure(
            self.configure(True, forced_compiler=False),
            "must resolve inside THEROCK_MODULE_ONEAPI_ROOT",
        )

    def test_contained_non_sycl_compiler_is_rejected(self) -> None:
        self.assert_failure(
            self.configure(
                True, f"-DCMAKE_CXX_COMPILER={self.icpx}", forced_compiler=False
            ),
            "Intel SGEMM requires an IntelLLVM icpx",
        )
        self.assertFalse((self.build / "module_contract_data.h").exists())

    def test_enabled_worker_uses_separate_sycl_and_spirv_compilers(self) -> None:
        self.assert_success(self.configure(True))
        ninja = (self.build / "build.ninja").read_text()
        rules = (self.build / "CMakeFiles/rules.ninja").read_text()
        description = json.loads(
            (self.build / "runner-contract-config.json").read_text()
        )
        self.assertIn("blas-provider-onemkl-v1", description["capabilities"])
        self.assertIn("-fsycl", ninja)
        self.assertIn(str(self.icpx), rules)
        self.assertIn("/bin/true -target spir64-unknown-unknown", ninja)
        for name in ("mkl_sycl_blas", "mkl_intel_ilp64", "mkl_sequential", "mkl_core"):
            self.assertIn(str(self.mkl / f"lib/lib{name}.so"), ninja)
        install = (self.build / "cmake_install.cmake").read_text()
        self.assertIn(str(self.mkl / "lib"), install)
        self.assertIn(str(self.compiler / "lib"), install)
        self.assertNotIn("-qmkl", ninja)
        self.assertIn("-Wl,--push-state,--no-as-needed", ninja)
        self.assertIn("-Wl,--pop-state", ninja)
        for relative in ("umf/latest/lib/libumf.so", "tcm/latest/lib/libhwloc.so"):
            self.assertIn(str(self.oneapi / relative), ninja)
            self.assertIn(str((self.oneapi / relative).parent), install)

    def test_link_input_symlinks_cannot_escape_sdk_or_point_to_stubs(self) -> None:
        library = self.mkl / "lib/libmkl_sycl_blas.so"
        outside = self.put(self.root / "outside.so")
        stub = self.put(self.mkl / "lib/stubs/libmkl_sycl_blas.so")
        for referent in (outside, stub):
            with self.subTest(referent=referent):
                library.unlink()
                library.symlink_to(referent)
                self.assert_failure(self.configure(True), "outside stub directories")

    def test_missing_selected_library_cannot_fall_back_to_another_prefix(self) -> None:
        (self.mkl / "lib/libmkl_core.so").unlink()
        fallback = self.root / "fallback"
        self.put(fallback / "lib/libmkl_core.so")
        self.assert_failure(
            self.configure(True, f"-DCMAKE_PREFIX_PATH={fallback}"), "_mkl_core"
        )

    def test_ur_runtime_components_are_required_and_cannot_escape_sdk(self) -> None:
        for relative, variable in (
            ("umf/latest/lib/libumf.so", "_umf_library"),
            ("tcm/latest/lib/libhwloc.so", "_hwloc_library"),
        ):
            path = self.oneapi / relative
            with self.subTest(relative=relative):
                path.unlink()
                self.assert_failure(self.configure(True), variable)
                outside = self.put(self.root / path.name)
                path.symlink_to(outside)
                self.assert_failure(self.configure(True), "outside stub directories")
                path.unlink()
                self.put(path)

    def test_intel_option_requires_native_testing_and_intel_selection(self) -> None:
        source = self.root / "parent"
        source.mkdir()
        (source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.25)\n"
            "project(intel_optin_guard LANGUAGES NONE)\n"
            f"set(THEROCK_SOURCE_DIR [[{REPO_ROOT}]])\n"
            f"set(Python3_EXECUTABLE [[{sys.executable}]])\n"
            f"set(Python3_VERSION [[{sys.version_info.major}.{sys.version_info.minor}]])\n"
            f"include([[{REPO_ROOT}/experimental/multi-vendor/CMakeLists.txt]])\n"
        )
        for testing, modules, target in (
            ("OFF", "ON", "intel:level-zero:xe2-b70"),
            ("ON", "OFF", "intel:level-zero:xe2-b70"),
            ("ON", "ON", "amd:hip:gfx1201"),
        ):
            with self.subTest(testing=testing, modules=modules, target=target):
                result = subprocess.run(
                    [
                        "cmake",
                        "-S",
                        str(source),
                        "-B",
                        str(self.root / "parent-build"),
                        "-GNinja",
                        "-DTHEROCK_ENABLE_MULTI_VENDOR_INTEL_SGEMM=ON",
                        f"-DTHEROCK_BUILD_TESTING={testing}",
                        f"-DTHEROCK_ENABLE_MULTI_VENDOR_MODULES={modules}",
                        "-DTHEROCK_ENABLE_MULTI_VENDOR_VALIDATION=OFF",
                        f"-DTHEROCK_MULTI_VENDOR_TARGETS={target}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assert_failure(
                    result, "Intel SGEMM requires THEROCK_BUILD_TESTING"
                )


if __name__ == "__main__":
    unittest.main()
