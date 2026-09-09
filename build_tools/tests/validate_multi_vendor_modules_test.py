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
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "build_tools"))
sys.path.insert(0, str(REPO_ROOT / "rocm-systems/shared/kpack/python"))

from _therock_utils.payload_catalog import (
    PayloadInput,
    create_pack,
    extract_verified_payload,
)
from _therock_utils.gpu_targets import parse_gpu_target
from _therock_utils.module_contract import (
    parse_contract,
    runner_description,
    validation_contract,
)
from validate_multi_vendor_modules import (
    _parse_device_id,
    _validate_service_fixture,
    describe_runner,
    execute_payload,
    execute_module_pipeline,
)


_SCRIPT = REPO_ROOT / "build_tools/validate_multi_vendor_modules.py"
_INTEL = "intel:level-zero:xe2-b70"
_MODULE = "validation/saxpy"
_SYMBOL = "therock_module_saxpy"
_UUID = "0123456789abcdef0123456789abcdef"


class ServiceDiagnosticsTest(unittest.TestCase):
    def test_worker_failure_preserves_unknown_completion_diagnostic(self):
        from _therock_utils.module_service import ModuleServiceProtocolError

        error = ModuleServiceProtocolError(
            "Native module service response was truncated",
            "SYNC backend=nvidia operation=stream-query completion=unknown\n",
        )
        with mock.patch(
            "_therock_utils.module_service.NativeModuleSession", side_effect=error
        ), self.assertRaisesRegex(ValueError, "completion=unknown") as caught:
            _validate_service_fixture(
                (),
                (),
                parse_gpu_target("nvidia:cuda:sm_120"),
                Path("unused"),
                0,
                None,
                _UUID,
                runner_description("nvidia"),
            )
        self.assertIn("response was truncated", str(caught.exception))
        self.assertIs(caught.exception.__cause__, error)


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
        self.calls = self.root / "runner-calls.jsonl"
        self.description = self.root / "runner-description.json"
        self.runner = self.root / "cpu-runner-stub"
        self.runner.write_text(
            f"#!{sys.executable}\n"
            + r"""
# This test process only records arguments and payload bytes. It never calls a GPU API.
import argparse
import json
import os
import sys
import time
from pathlib import Path

calls = Path(os.environ["THEROCK_TEST_RUNNER_CALLS"])
with calls.open("a") as file:
    file.write(json.dumps(sys.argv[1:]) + "\n")
if sys.argv[1:] == ["--describe-contract"]:
    mode = os.environ.get("THEROCK_TEST_DESCRIBE_MODE", "ok")
    if mode == "sleep":
        time.sleep(30)
    if mode == "flood":
        sys.stdout.write("x" * (65 * 1024))
        raise SystemExit(0)
    if mode == "invalid-utf8":
        sys.stdout.buffer.write(b"\xff")
        raise SystemExit(0)
    if os.environ.get("THEROCK_TEST_MUTATE_CATALOG"):
        Path(os.environ["THEROCK_TEST_MUTATE_CATALOG"]).write_text("{\"after\":\"extraction\"}")
    sys.stdout.write(Path(os.environ["THEROCK_TEST_RUNNER_DESCRIPTION"]).read_text())
    raise SystemExit(int(os.environ.get("THEROCK_TEST_DESCRIBE_EXIT", "0")))

parser = argparse.ArgumentParser()
parser.add_argument("--launch-abi", required=True)
parser.add_argument("--launch-abi-version", required=True)
parser.add_argument("--launch-contract-sha256", required=True)
parser.add_argument("--payload-format", required=True)
parser.add_argument("--payload", required=True)
parser.add_argument("--symbol", required=True)
parser.add_argument("--device", required=True)
parser.add_argument("--expect-arch", required=True)
parser.add_argument("--expect-device-id")
parser.add_argument("--expect-device-uuid")
args = parser.parse_args()
record = vars(args)
record["payload_bytes"] = Path(args.payload).read_bytes().hex()
Path(os.environ["THEROCK_TEST_RUNNER_RECORD"]).write_text(json.dumps(record))
raise SystemExit(int(os.environ.get("THEROCK_TEST_RUNNER_EXIT", "0")))
"""
        )
        self.runner.chmod(0o755)
        self.pack_count = 0

    def pack(
        self,
        target=_INTEL,
        payload_type="spirv",
        module=_MODULE,
        *,
        contracted=True,
        contract=None,
    ):
        self.pack_count += 1
        pack_id = f"host-boundary-{self.pack_count}"
        payload = f"synthetic:{target}:{payload_type}:{module}".encode()
        catalog = create_pack(
            self.root / pack_id,
            pack_id,
            (
                PayloadInput(
                    module,
                    target,
                    payload_type,
                    (_SYMBOL,),
                    payload,
                    contract=(
                        contract
                        if contract is not None
                        else validation_contract() if contracted else None
                    ),
                ),
            ),
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
        description=None,
        description_text=None,
        describe_mode="ok",
        describe_exit=0,
        mutate_catalog=None,
    ):
        env = os.environ.copy()
        paths = [str(REPO_ROOT / "rocm-systems/shared/kpack/python")]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        self.calls.unlink(missing_ok=True)
        self.description.write_text(
            description_text
            if description_text is not None
            else json.dumps(
                description
                if description is not None
                else runner_description(parse_gpu_target(target).vendor).record()
            )
        )
        env["THEROCK_TEST_RUNNER_CALLS"] = str(self.calls)
        env["THEROCK_TEST_RUNNER_DESCRIPTION"] = str(self.description)
        env["THEROCK_TEST_DESCRIBE_MODE"] = describe_mode
        env["THEROCK_TEST_DESCRIBE_EXIT"] = str(describe_exit)
        if mutate_catalog is not None:
            env["THEROCK_TEST_MUTATE_CATALOG"] = str(mutate_catalog)
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
        return subprocess.run(
            command, capture_output=True, text=True, env=env, timeout=15
        )

    def assert_not_invoked(self, result, error, *, described=False):
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn(error, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(self.record.exists())
        if described:
            self.assertEqual(
                [json.loads(line) for line in self.calls.read_text().splitlines()],
                [["--describe-contract"]],
            )
        else:
            self.assertFalse(self.calls.exists())

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
        self.assertIsNone(record["expect_device_uuid"])
        self.assertEqual(record["symbol"], _SYMBOL)
        self.assertEqual(record["launch_abi"], validation_contract().record()["abi"])
        self.assertEqual(record["launch_abi_version"], "1")
        self.assertEqual(record["launch_contract_sha256"], validation_contract().sha256)
        self.assertEqual(record["payload_format"], "spirv")
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual(calls[0], ["--describe-contract"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(bytes.fromhex(record["payload_bytes"]), payload)
        self.assertFalse(Path(record["payload"]).exists())
        self.assertIn("verified=sha256", result.stdout)
        self.assertIn("abi_version=1", result.stdout)
        self.assertIn(f"contract_sha256={validation_contract().sha256}", result.stdout)

    def test_optional_expected_uuid_is_forwarded_for_each_native_backend(self):
        for target, payload_type in (
            ("amd:hip:gfx1201", "hsaco"),
            ("nvidia:cuda:sm_120", "cubin"),
            (_INTEL, "spirv"),
        ):
            with self.subTest(target=target):
                catalog, _ = self.pack(target=target, payload_type=payload_type)
                result = self.invoke(
                    (catalog,),
                    target=target,
                    payload_type=payload_type,
                    extra=("--expect-device-uuid", _UUID),
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(
                    json.loads(self.record.read_text())["expect_device_uuid"], _UUID
                )

    def test_malformed_expected_uuid_never_invokes_runner(self):
        catalog, _ = self.pack()
        for value in (
            "",
            "0" * 32,
            _UUID.upper(),
            _UUID[:-1],
            "g" * 32,
            " " + _UUID,
            _UUID + "\n",
            "01234567-89ab-cdef-0123-456789abcdef",
        ):
            with self.subTest(value=value):
                result = self.invoke((catalog,), extra=("--expect-device-uuid", value))
                self.assert_not_invoked(result, "UUID")

    def test_direct_execution_rejects_malformed_uuid_before_description_query(self):
        catalog, _ = self.pack()
        selected = extract_verified_payload(
            (catalog,), _MODULE, _INTEL, "spirv", entry_point=_SYMBOL
        )
        for value in ("", "0" * 32, _UUID.upper()):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "UUID"):
                execute_payload(
                    selected,
                    parse_gpu_target(_INTEL),
                    self.runner,
                    0,
                    None,
                    entry_point=_SYMBOL,
                    expected_device_uuid=value,
                )
        self.assertFalse(self.calls.exists())
        self.assertFalse(self.record.exists())

    def test_repeated_pipeline_request_cannot_change_snapshot_before_runner_query(self):
        catalog, _ = self.pack()
        selected = extract_verified_payload(
            (catalog,), _MODULE, _INTEL, "spirv", entry_point=_SYMBOL
        )
        other_catalog = create_pack(
            self.root / "alternate",
            "alternate",
            (
                PayloadInput(
                    _MODULE,
                    _INTEL,
                    "spirv",
                    (_SYMBOL,),
                    b"another verified payload",
                    contract=validation_contract(),
                ),
            ),
        )
        conflicting = extract_verified_payload(
            (other_catalog,), _MODULE, _INTEL, "spirv", entry_point=_SYMBOL
        )
        with self.assertRaisesRegex(ValueError, "same verified snapshot"):
            execute_module_pipeline(
                ((selected, _SYMBOL), (conflicting, _SYMBOL)),
                parse_gpu_target(_INTEL),
                self.runner,
                0,
                None,
            )
        self.assertFalse(self.calls.exists())
        self.assertFalse(self.record.exists())

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
        original["schema_version"] = 3
        catalog.write_text(json.dumps(original))
        result = self.invoke((catalog,))
        self.assert_not_invoked(result, "Unsupported catalog schema")

    def test_legacy_catalog_is_extractable_but_cannot_launch(self):
        catalog, _ = self.pack(contracted=False)
        result = self.invoke((catalog,))
        self.assert_not_invoked(result, "requires a schema 2 launch contract")

    def test_self_consistent_but_unsupported_contract_never_invokes_runner(self):
        original = validation_contract().record()
        variants = []
        for field, value in (
            ("abi", "different-fixture"),
            ("version", 2),
            ("required_capabilities", [*original["required_capabilities"], "fp64"]),
        ):
            variants.append({**original, field: value})
        reordered = json.loads(json.dumps(original))
        reordered["arguments"][0], reordered["arguments"][1] = (
            reordered["arguments"][1],
            reordered["arguments"][0],
        )
        variants.append(reordered)
        scalar = json.loads(json.dumps(original))
        scalar["arguments"][3]["value_type"] = "u32"
        variants.append(scalar)
        narrow = json.loads(json.dumps(original))
        narrow["pointer_bits"] = 32
        for argument in narrow["arguments"][:3]:
            argument["size_bytes"] = argument["alignment_bytes"] = 4
        variants.append(narrow)
        geometry = json.loads(json.dumps(original))
        geometry["launch"]["block"] = [64, 1, 1]
        variants.append(geometry)
        for record in variants:
            with self.subTest(record=record):
                catalog, _ = self.pack(contract=parse_contract(record))
                result = self.invoke((catalog,))
                self.assert_not_invoked(
                    result, "Unsupported validation launch contract"
                )

    def test_bad_later_catalog_is_checked_before_any_runner_invocation(self):
        selected, _ = self.pack()
        other, _ = self.pack(module="validation/relu")
        document = json.loads(other.read_text())
        document["packs"][0]["entries"][0]["contract"]["version"] = True
        other.write_text(json.dumps(document))
        result = self.invoke((selected, other))
        self.assert_not_invoked(result, "version")

    def test_catalog_change_after_description_cannot_replace_verified_snapshot(self):
        catalog, payload = self.pack()
        result = self.invoke((catalog,), mutate_catalog=catalog)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(catalog.read_text()), {"after": "extraction"})
        record = json.loads(self.record.read_text())
        self.assertEqual(bytes.fromhex(record["payload_bytes"]), payload)
        self.assertEqual(record["launch_contract_sha256"], validation_contract().sha256)

    def test_description_mismatches_never_invoke_payload(self):
        catalog, _ = self.pack()
        original = runner_description("intel").record()
        alternate = parse_contract({**validation_contract().record(), "version": 2})
        changes = (
            {
                **original,
                "contract": alternate.record(),
                "contract_sha256": alternate.sha256,
            },
            runner_description("nvidia").record(),
            {**original, "backend": "hip"},
            {**original, "payload_types": []},
            {**original, "entry_points": ["therock_module_relu"]},
            {
                **original,
                "entry_points": [*original["entry_points"], "unsupported_fixture"],
            },
            {**original, "capabilities": []},
            {**original, "capabilities": [*original["capabilities"], "fp64"]},
            {**original, "contract_sha256": "0" * 64},
            {**original, "schema_version": True},
            {**original, "schema_version": 2},
            {**original, "scope": "hardware-qualified"},
            {**original, "unsupported": True},
        )
        for description in changes:
            with self.subTest(description=description):
                result = self.invoke((catalog,), description=description)
                self.assert_not_invoked(result, "", described=True)

    def test_nvidia_runner_missing_requested_format_never_invokes_payload(self):
        target = "nvidia:cuda:sm_120"
        catalog, _ = self.pack(target=target, payload_type="ptx")
        description = runner_description("nvidia").record()
        description["payload_types"] = ["cubin"]
        result = self.invoke(
            (catalog,), target=target, payload_type="ptx", description=description
        )
        self.assert_not_invoked(result, "", described=True)

    def test_non_posix_description_request_fails_before_starting_runner(self):
        with mock.patch(
            "_therock_utils.runner_query.os.name", "nt"
        ), self.assertRaisesRegex(ValueError, "POSIX host"):
            describe_runner(self.runner)
        self.assertFalse(self.calls.exists())

    def test_malformed_description_never_invokes_payload(self):
        catalog, _ = self.pack()
        for text in (
            "",
            "[]",
            "not JSON",
            '{"schema_version":1,"schema_version":1}',
            json.dumps(runner_description("intel").record()) + " trailing",
        ):
            with self.subTest(text=text):
                result = self.invoke((catalog,), description_text=text)
                self.assert_not_invoked(result, "", described=True)
        result = self.invoke((catalog,), describe_mode="invalid-utf8")
        self.assert_not_invoked(result, "UTF-8 JSON", described=True)

    def test_description_json_nesting_is_bounded(self):
        catalog, _ = self.pack()
        result = self.invoke((catalog,), description_text="[" * 10000 + "]" * 10000)
        self.assert_not_invoked(result, "too deeply nested", described=True)

    def test_failed_or_oversized_description_never_invokes_payload(self):
        catalog, _ = self.pack()
        result = self.invoke((catalog,), describe_exit=7)
        self.assert_not_invoked(
            result, "description failed with exit code 7", described=True
        )
        result = self.invoke((catalog,), describe_mode="flood")
        self.assert_not_invoked(result, "output limit", described=True)

    def test_description_has_a_bounded_timeout(self):
        self.description.write_text(json.dumps(runner_description("intel").record()))
        environment = {
            "THEROCK_TEST_RUNNER_CALLS": str(self.calls),
            "THEROCK_TEST_RUNNER_DESCRIPTION": str(self.description),
            "THEROCK_TEST_DESCRIBE_MODE": "sleep",
        }
        with mock.patch.dict(os.environ, environment), self.assertRaisesRegex(
            ValueError, "timed out"
        ):
            describe_runner(self.runner, timeout=0.2)
        self.assertFalse(self.record.exists())

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

    def test_explicit_event_capability_requires_advertised_support(self):
        catalog, _ = self.pack()
        description = runner_description("intel").record()
        description["capabilities"].remove("cross-queue-events")
        result = self.invoke(
            (catalog,),
            description=description,
            extra=("--require-capability", "cross-queue-events"),
        )
        self.assert_not_invoked(
            result, "lacks required adapter capabilities", described=True
        )
        result = self.invoke(
            (catalog,), extra=("--require-capability", "cross-queue-events")
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unknown_capability_never_invokes_runner(self):
        catalog, _ = self.pack()
        result = self.invoke((catalog,), extra=("--require-capability", "unknown"))
        self.assert_not_invoked(result, "Unknown required adapter capability")


if __name__ == "__main__":
    unittest.main()
