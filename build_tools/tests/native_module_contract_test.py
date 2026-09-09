# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU-only CLI gates for real compiled native runners; no GPU is initialized.

Set THEROCK_NATIVE_MODULE_BUILD_ROOT to the profile's modules directory. Linux
runs interpose the first driver APIs with an aborting CPU shim, so a regression
cannot silently turn a metadata/negative test into hardware validation.
"""

import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest


class NativeModuleContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get("THEROCK_NATIVE_MODULE_BUILD_ROOT")
        if not root:
            raise unittest.SkipTest(
                "Set THEROCK_NATIVE_MODULE_BUILD_ROOT to test compiled native runners"
            )
        cls.runners = sorted(Path(root).glob("*/build/therock_module_validation"))
        if not cls.runners:
            raise AssertionError(f"No compiled native runners found beneath {root}")
        temporary = tempfile.TemporaryDirectory(prefix="therock-native-contract-")
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.environment = dict(os.environ)
        if sys.platform.startswith("linux"):
            compiler = shutil.which("cc")
            if compiler is None:
                raise AssertionError(
                    "A CPU C compiler is required for the GPU API trap"
                )
            trap_source = cls.root / "gpu_api_trap.c"
            trap_source.write_text(
                """#include <stdio.h>
#include <stdlib.h>
static int forbidden(void) {
  fputs("FORBIDDEN_GPU_API_CALL\\n", stderr);
  fflush(stderr);
  _Exit(91);
}
int cuInit(unsigned flags) { return forbidden(); }
int cuDeviceGetCount(int *count) { return forbidden(); }
int hipInit(unsigned flags) { return forbidden(); }
int hipGetDeviceCount(int *count) { return forbidden(); }
int zeInit(unsigned flags) { return forbidden(); }
"""
            )
            trap = cls.root / "gpu_api_trap.so"
            result = subprocess.run(
                [compiler, "-shared", "-fPIC", str(trap_source), "-o", str(trap)],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode:
                raise AssertionError(result.stdout + result.stderr)
            previous = cls.environment.get("LD_PRELOAD", "")
            cls.environment["LD_PRELOAD"] = str(trap) + (
                ":" + previous if previous else ""
            )
        cls.descriptions = {}
        for runner in cls.runners:
            result = subprocess.run(
                [str(runner), "--describe-contract"],
                env=cls.environment,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode or result.stderr:
                raise AssertionError(f"{runner}: {result.stdout}\n{result.stderr}")
            cls.descriptions[runner] = json.loads(result.stdout)

    def invoke(self, runner, *args, success=False, message=None):
        result = subprocess.run(
            [str(runner), *map(str, args)],
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=5,
        )
        evidence = f"{runner}: {args!r}\n{result.stdout}\n{result.stderr}"
        self.assertNotIn("FORBIDDEN_GPU_API_CALL", result.stderr, evidence)
        self.assertNotEqual(result.returncode, 91, evidence)
        self.assertNotIn("DEVICE ", result.stdout, evidence)
        self.assertNotIn("CAPABILITY ", result.stdout, evidence)
        if success:
            self.assertEqual(result.returncode, 0, evidence)
        else:
            self.assertNotEqual(result.returncode, 0, evidence)
        if message is not None:
            self.assertIn(message, result.stderr, evidence)
        return result

    def contract_args(self, runner, *, payload_format=None):
        description = self.descriptions[runner]
        return [
            "--launch-abi",
            description["contract"]["abi"],
            "--launch-abi-version",
            str(description["contract"]["version"]),
            "--launch-contract-sha256",
            description["contract_sha256"],
            "--payload-format",
            payload_format or description["payload_types"][0],
        ]

    def test_description_reports_compiled_contract_without_hardware_claims(self):
        identities = set()
        for runner, description in self.descriptions.items():
            with self.subTest(runner=runner):
                self.assertEqual(description["schema_version"], 1)
                self.assertEqual(description["kind"], "module-runner-contract")
                self.assertEqual(description["scope"], "compiled-adapter")
                self.assertEqual(
                    description["contract"]["abi"], "therock.validation.f32-vector"
                )
                self.assertEqual(description["contract"]["version"], 1)
                self.assertRegex(description["contract_sha256"], r"^[0-9a-f]{64}$")
                self.assertEqual(
                    set(description["entry_points"]),
                    {"therock_module_saxpy", "therock_module_relu"},
                )
                self.assertNotIn("device", description)
                self.assertNotIn("device_id", description)
                identities.add(description["contract_sha256"])
                result = self.invoke(runner, "--describe-contract", success=True)
                self.assertEqual(json.loads(result.stdout), description)
        self.assertEqual(
            len(identities), 1, "The fixture contract must agree across native adapters"
        )

    def test_description_mode_is_exclusive(self):
        for runner in self.runners:
            for extra in (
                ("--payload", self.root / "missing"),
                ("--device", "0"),
                ("--help",),
                ("--describe-contract",),
            ):
                with self.subTest(runner=runner, extra=extra):
                    self.invoke(
                        runner,
                        "--describe-contract",
                        *extra,
                        message="must be used alone",
                    )

    def test_every_contract_flag_is_required_before_payload_access(self):
        for runner in self.runners:
            flags = self.contract_args(runner)
            for omitted in range(0, len(flags), 2):
                with self.subTest(runner=runner, omitted=flags[omitted]):
                    remaining = flags[:omitted] + flags[omitted + 2 :]
                    self.invoke(
                        runner,
                        "--payload",
                        self.root / "does-not-exist",
                        *remaining,
                        message="Payload execution requires",
                    )

    def test_mismatched_contract_fields_fail_before_payload_access(self):
        for runner in self.runners:
            for flag, replacement, message in (
                ("--launch-abi", "other.abi", "Unsupported --launch-abi"),
                ("--launch-abi-version", "2", "Unsupported --launch-abi-version"),
                ("--launch-abi-version", "1x", "Unsupported --launch-abi-version"),
                (
                    "--launch-contract-sha256",
                    "0" * 64,
                    "Mismatched --launch-contract-sha256",
                ),
                (
                    "--launch-contract-sha256",
                    "not-a-hash",
                    "Mismatched --launch-contract-sha256",
                ),
                ("--payload-format", "unknown", "Unsupported --payload-format"),
            ):
                with self.subTest(runner=runner, flag=flag, value=replacement):
                    flags = self.contract_args(runner)
                    flags[flags.index(flag) + 1] = replacement
                    self.invoke(
                        runner,
                        "--payload",
                        self.root / "does-not-exist",
                        *flags,
                        message=message,
                    )

    def test_duplicate_or_unknown_arguments_are_rejected(self):
        for runner in self.runners:
            flags = self.contract_args(runner)
            for extra, message in (
                (("--launch-abi", "other"), "Duplicate argument"),
                (("--payload", "other"), "Duplicate argument"),
                (("--undeclared-option", "value"), "Unknown argument"),
            ):
                with self.subTest(runner=runner, extra=extra):
                    self.invoke(
                        runner,
                        "--payload",
                        self.root / "does-not-exist",
                        *flags,
                        *extra,
                        message=message,
                    )

    def test_valid_contract_reaches_payload_read_without_initializing_driver(self):
        for runner in self.runners:
            with self.subTest(runner=runner):
                self.invoke(
                    runner,
                    "--payload",
                    self.root / "does-not-exist",
                    *self.contract_args(runner),
                    message="Cannot read payload",
                )

    def test_malformed_payload_fails_before_driver_initialization(self):
        payload = self.root / "malformed"
        payload.write_bytes(b"this is not a device module")
        for runner in self.runners:
            with self.subTest(runner=runner):
                result = self.invoke(
                    runner, "--payload", payload, *self.contract_args(runner)
                )
                self.assertRegex(
                    result.stderr, "Unrecognized native payload|Malformed SPIR-V"
                )

    def test_wrong_backend_or_declared_format_is_rejected_before_driver(self):
        cuda_elf = bytearray(64)
        cuda_elf[:7] = b"\x7fELF\x02\x01\x01"
        cuda_elf[18:20] = struct.pack("<H", 190)
        amd_elf = bytearray(cuda_elf)
        amd_elf[18:20] = struct.pack("<H", 224)
        payload = self.root / "wrong-format"
        for runner, description in self.descriptions.items():
            with self.subTest(runner=runner):
                vendor = description["vendor"]
                payload.write_bytes(cuda_elf if vendor != "nvidia" else amd_elf)
                result = self.invoke(
                    runner, "--payload", payload, *self.contract_args(runner)
                )
                self.assertRegex(
                    result.stderr, "Wrong payload ELF machine|Malformed SPIR-V"
                )
                if vendor == "nvidia":
                    payload.write_bytes(
                        b".version 8.0\n.target sm_90\n.address_size 64\n"
                    )
                    self.invoke(
                        runner,
                        "--payload",
                        payload,
                        *self.contract_args(runner, payload_format="cubin"),
                        message="does not match inspected format ptx",
                    )
                    payload.write_bytes(cuda_elf)
                    self.invoke(
                        runner,
                        "--payload",
                        payload,
                        *self.contract_args(runner, payload_format="ptx"),
                        message="does not match inspected format cubin",
                    )

    def test_help_is_available_without_payload_or_gpu(self):
        for runner in self.runners:
            with self.subTest(runner=runner):
                result = self.invoke(runner, "--help", success=True)
                self.assertIn("--describe-contract", result.stdout)
                self.assertIn("--launch-contract-sha256", result.stdout)


if __name__ == "__main__":
    unittest.main()
