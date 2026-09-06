# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU process-boundary tests using synthetic packs; no GPU qualification is implied."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "build_tools"))
sys.path.insert(0, str(REPO_ROOT / "rocm-systems/shared/kpack/python"))

from _therock_utils.payload_catalog import PayloadInput, create_pack
from validate_multi_vendor_modules import _parse_device_id


_SCRIPT = REPO_ROOT / "build_tools/validate_multi_vendor_modules.py"
_INTEL = "intel:level-zero:xe2-b70"
_MODULE = "validation/saxpy"
_SYMBOL = "therock_module_saxpy"


class DeviceIdParserTest(unittest.TestCase):
    def test_decimal_and_hex_ids_accept_leading_decimal_zeroes(self):
        for value, expected in (
            ("08", 8),
            ("0008", 8),
            ("0xE223", 0xE223),
            ("0Xe223", 0xE223),
            ("57891", 0xE223),
            ("65535", 0xFFFF),
        ):
            with self.subTest(value=value):
                self.assertEqual(_parse_device_id(value), expected)

    def test_invalid_syntax_and_range_are_rejected(self):
        for value in (
            "",
            "0",
            "00",
            "0x0",
            "+1",
            "-1",
            " 8",
            "8 ",
            "8\n",
            " ",
            "0x",
            "0X",
            "0x+1",
            "0x-1",
            "5_7891",
            "0xE_223",
            "65536",
            "0x10000",
            "0b1000",
            "0o10",
        ):
            with self.subTest(value=value), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                _parse_device_id(value)


class ValidateMultiVendorModulesHostBoundaryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.record = self.root / "runner-record.json"
        self.runner = self.root / "cpu-runner-stub"
        self.runner.write_text(
            f"#!{sys.executable}\n"
            + """
# This test process only records arguments and payload bytes. It never calls a GPU API.
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--payload", required=True)
parser.add_argument("--symbol", required=True)
parser.add_argument("--device", required=True)
parser.add_argument("--expect-arch", required=True)
parser.add_argument("--expect-device-id")
args = parser.parse_args()
record = vars(args)
record["payload_bytes"] = Path(args.payload).read_bytes().hex()
Path(os.environ["THEROCK_TEST_RUNNER_RECORD"]).write_text(json.dumps(record))
raise SystemExit(int(os.environ.get("THEROCK_TEST_RUNNER_EXIT", "0")))
"""
        )
        self.runner.chmod(0o755)
        self.pack_count = 0

    def pack(self, target=_INTEL, payload_type="spirv", module=_MODULE):
        self.pack_count += 1
        pack_id = f"host-boundary-{self.pack_count}"
        payload = f"synthetic:{target}:{payload_type}:{module}".encode()
        catalog = create_pack(
            self.root / pack_id,
            pack_id,
            (PayloadInput(module, target, payload_type, (_SYMBOL,), payload),),
        )
        return catalog, payload

    def invoke(
        self,
        catalogs,
        *,
        target=_INTEL,
        payload_type="spirv",
        module=_MODULE,
        symbol=_SYMBOL,
        extra=(),
        exit_code=0,
    ):
        env = os.environ.copy()
        paths = [str(REPO_ROOT / "rocm-systems/shared/kpack/python")]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        env["THEROCK_TEST_RUNNER_RECORD"] = str(self.record)
        env["THEROCK_TEST_RUNNER_EXIT"] = str(exit_code)
        command = [sys.executable, str(_SCRIPT)]
        for catalog in catalogs:
            command.extend(("--catalog", str(catalog)))
        command.extend(
            (
                "--module",
                module,
                "--target",
                target,
                "--format",
                payload_type,
                "--entry-point",
                symbol,
                "--runner",
                str(self.runner),
                *extra,
            )
        )
        return subprocess.run(command, capture_output=True, text=True, env=env)

    def assert_not_invoked(self, result, error):
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn(error, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(self.record.exists())

    def test_b70_exact_pack_reaches_cpu_stub_with_pci_identity(self):
        # First catalog shares the target but not the requested module.
        first, _ = self.pack(module="validation/relu")
        second, payload = self.pack()
        result = self.invoke((first, second), extra=("--device", "2"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record = json.loads(self.record.read_text())
        self.assertEqual(record["expect_arch"], "spirv")
        self.assertEqual(record["expect_device_id"], "0xe223")
        self.assertEqual(record["device"], "2")
        self.assertEqual(record["symbol"], _SYMBOL)
        self.assertEqual(bytes.fromhex(record["payload_bytes"]), payload)
        self.assertFalse(Path(record["payload"]).exists())
        self.assertIn("verified=sha256", result.stdout)

    def test_matching_b70_device_id_is_accepted_and_conflict_rejected(self):
        catalog, _ = self.pack()
        result = self.invoke((catalog,), extra=("--expect-device-id", "0x1234"))
        self.assert_not_invoked(result, "requires --expect-device-id 0xe223")
        result = self.invoke((catalog,), extra=("--expect-device-id", str(0xE223)))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(self.record.read_text())["expect_device_id"], "0xe223"
        )

    def test_other_intel_target_requires_explicit_hardware_identity(self):
        target = "intel:level-zero:xe2-future"
        catalog, _ = self.pack(target=target)
        result = self.invoke((catalog,), target=target)
        self.assert_not_invoked(result, "requires an explicit --expect-device-id")
        for value, expected in (("4660", "0x1234"), ("08", "0x8")):
            with self.subTest(value=value):
                result = self.invoke(
                    (catalog,), target=target, extra=("--expect-device-id", value)
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                record = json.loads(self.record.read_text())
                self.assertEqual(record["expect_arch"], "spirv")
                self.assertEqual(record["expect_device_id"], expected)

    def test_invalid_pci_ids_never_invoke_runner(self):
        catalog, _ = self.pack()
        for value in ("0", "-1", "0x10000", "0xe223junk", "not-a-number"):
            with self.subTest(value=value):
                result = self.invoke((catalog,), extra=("--expect-device-id", value))
                self.assert_not_invoked(result, "Device ID must be")

    def test_amd_and_nvidia_keep_architecture_checks(self):
        for target, payload_type, expected_arch in (
            ("amd:hip:gfx1201", "hsaco", "gfx1201"),
            ("amd:hip:gfx942:sramecc-:xnack+", "hsaco", "gfx942"),
            ("nvidia:cuda:sm_120", "cubin", "sm_120"),
            ("nvidia:cuda:sm_90a", "ptx", "sm_90"),
        ):
            with self.subTest(target=target, payload_type=payload_type):
                self.record.unlink(missing_ok=True)
                catalog, payload = self.pack(target=target, payload_type=payload_type)
                result = self.invoke(
                    (catalog,), target=target, payload_type=payload_type
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                record = json.loads(self.record.read_text())
                self.assertEqual(record["expect_arch"], expected_arch)
                self.assertIsNone(record["expect_device_id"])
                self.assertEqual(bytes.fromhex(record["payload_bytes"]), payload)
                self.record.unlink()
                rejected = self.invoke(
                    (catalog,),
                    target=target,
                    payload_type=payload_type,
                    extra=("--expect-device-id", "0xe223"),
                )
                self.assert_not_invoked(rejected, "only supported for Intel")

    def test_catalog_and_symbol_failures_never_invoke_runner(self):
        catalog, _ = self.pack()
        result = self.invoke((catalog,), symbol="undeclared")
        self.assert_not_invoked(result, "not declared")
        result = self.invoke((catalog,), module="validation/missing")
        self.assert_not_invoked(result, "No exact payload match")
        result = self.invoke((catalog,), payload_type="cubin")
        self.assert_not_invoked(result, "not valid for")
        document = json.loads(catalog.read_text())
        document["packs"][0]["entries"][0]["sha256"] = "0" * 64
        catalog.write_text(json.dumps(document))
        result = self.invoke((catalog,))
        self.assert_not_invoked(result, "Payload SHA256 mismatch")

    def test_pack_hash_and_schema_failures_never_invoke_runner(self):
        catalog, _ = self.pack()
        original = json.loads(catalog.read_text())
        altered = json.loads(catalog.read_text())
        altered["packs"][0]["sha256"] = "0" * 64
        catalog.write_text(json.dumps(altered))
        result = self.invoke((catalog,))
        self.assert_not_invoked(result, "Pack SHA256 mismatch")
        original["schema_version"] = 2
        catalog.write_text(json.dumps(original))
        result = self.invoke((catalog,))
        self.assert_not_invoked(result, "Unsupported catalog schema")

    def test_negative_device_index_never_invokes_runner(self):
        catalog, _ = self.pack()
        result = self.invoke((catalog,), extra=("--device", "-1"))
        self.assert_not_invoked(result, "Device index must be nonnegative")

    def test_native_runner_failure_propagates_and_temporary_payload_is_removed(self):
        catalog, _ = self.pack()
        result = self.invoke((catalog,), exit_code=7)
        self.assertEqual(result.returncode, 7, result.stderr)
        record = json.loads(self.record.read_text())
        self.assertFalse(Path(record["payload"]).exists())


if __name__ == "__main__":
    unittest.main()
