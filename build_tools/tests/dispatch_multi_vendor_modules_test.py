# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU process-boundary dispatch tests; synthetic inventories never qualify GPUs."""

import copy
import hashlib
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

from _therock_utils.gpu_targets import parse_gpu_target
from _therock_utils.module_contract import runner_description, validation_contract
from _therock_utils.payload_catalog import PayloadInput, create_pack


_SCRIPT = REPO_ROOT / "build_tools/dispatch_multi_vendor_modules.py"
_AMD = "amd:hip:gfx1201"
_NVIDIA = "nvidia:cuda:sm_120"
_INTEL = "intel:level-zero:xe2-b70"
_MODULE = "validation/relu"
_SYMBOL = "therock_module_relu"
_UUID_A = "0123456789abcdef0123456789abcdef"
_UUID_B = "fedcba9876543210fedcba9876543210"
_UUID_MISSING = "1" * 32

# The runner uses the standard library only and never imports or calls GPU APIs.
_RUNNER_BODY = r"""
import argparse
import json
import os
from pathlib import Path
import sys

root = Path(os.environ["THEROCK_DISPATCH_TEST_LOGS"])
with (root / (CONFIG["tag"] + ".jsonl")).open("a") as file:
    file.write(json.dumps(sys.argv[1:]) + "\n")
if sys.argv[1:] == ["--describe-contract"]:
    for path in CONFIG.get("mutate_catalogs_on_describe", []):
        Path(path).write_text('{"after":"extraction"}')
    print(json.dumps(CONFIG["description"]))
    raise SystemExit(CONFIG.get("describe_exit", 0))
if sys.argv[1:] == ["--list-devices"]:
    if CONFIG.get("mutate_on_list"):
        with Path(__file__).open("a") as file:
            file.write("\n# fixture update during device discovery\n")
    mode = CONFIG.get("list_mode", "ok")
    if mode == "unavailable":
        print("fixture runtime unavailable", file=sys.stderr)
        raise SystemExit(17)
    if mode == "flood":
        print("x" * (65 * 1024))
        raise SystemExit(0)
    if mode == "malformed":
        print('{"schema_version":')
        raise SystemExit(0)
    if mode == "duplicate":
        print(json.dumps(CONFIG["inventory"]).replace(
            '"schema_version": 2', '"schema_version": 2, "schema_version": 2'))
        raise SystemExit(0)
    print(json.dumps(CONFIG["inventory"]))
    raise SystemExit(0)

parser = argparse.ArgumentParser()
for flag in (
    "--launch-abi", "--launch-abi-version", "--launch-contract-sha256",
    "--device", "--expect-arch",
    "--expect-device-uuid",
):
    parser.add_argument(flag, required=True)
parser.add_argument("--expect-device-id")
parser.add_argument("--payload-format")
parser.add_argument("--payload")
parser.add_argument("--symbol")
parser.add_argument("--module", action="append", nargs=3)
parser.add_argument("--pipeline", action="store_true")
args = parser.parse_args()
record = vars(args)
if args.module:
    record["modules"] = [
        {"format": fmt, "symbol": symbol, "path": path,
         "payload_bytes": Path(path).read_bytes().hex()}
        for fmt, symbol, path in args.module
    ]
else:
    record["payload_bytes"] = Path(args.payload).read_bytes().hex()
(root / (CONFIG["tag"] + ".payload.json")).write_text(json.dumps(record))
actual_uuid = CONFIG.get(
    "launch_device_uuid", CONFIG["inventory"]["devices"][int(args.device)]["device_uuid"]
)
if args.expect_device_uuid != actual_uuid:
    print("fixture native device UUID mismatch", file=sys.stderr)
    raise SystemExit(23)
raise SystemExit(CONFIG.get("launch_exit", 0))
"""


def _inventory(target_id, *, devices=None):
    target = parse_gpu_target(target_id)
    vendor_id = {"amd": 0x1002, "nvidia": 0x10DE, "intel": 0x8086}[target.vendor]
    return {
        "schema_version": 2,
        "kind": "native-device-inventory",
        "scope": "observed-runtime",
        "vendor": target.vendor,
        "backend": target.backend,
        "devices": (
            devices
            if devices is not None
            else [
                {
                    "index": 0,
                    "device_uuid": _UUID_A,
                    "name": f"CPU fixture {target.vendor} device",
                    "architecture": (
                        target.processor if target.vendor != "intel" else None
                    ),
                    "vendor_id": vendor_id,
                    "device_id": 0xE223 if target.vendor == "intel" else None,
                    "limits": {
                        "max_threads_per_block": 1024,
                        "max_block_dimensions": [1024, 1024, 64],
                        "max_grid_dimensions": [2147483647, 65535, 65535],
                        "total_memory_bytes": 16 * 1024 * 1024 * 1024,
                    },
                }
            ]
        ),
    }


class DispatchMultiVendorModulesHostBoundaryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dist = self.root / "dist"
        self.logs = self.root / "calls"
        self.logs.mkdir()
        self.configs = {}
        self.registry = {
            "schema_version": 1,
            "kind": "native-runner-registry",
            "runners": [],
            "catalogs": [],
        }
        self.registry_path = self.dist / "share/therock/packs/runners.json"
        self.targets = (_AMD, _NVIDIA, _INTEL)
        for target_id in self.targets:
            self.add_runner(target_id)
        for module in ("saxpy", "relu"):
            inputs = []
            for target_id in self.targets:
                target = parse_gpu_target(target_id)
                for payload_type in target.payload_types:
                    inputs.append(
                        PayloadInput(
                            f"validation/{module}",
                            target_id,
                            payload_type,
                            (f"therock_module_{module}",),
                            self.payload_bytes(module, target_id, payload_type),
                            contract=validation_contract(),
                        )
                    )
            path = create_pack(
                self.dist / "share/therock/packs" / module,
                f"validation-{module}",
                inputs,
            )
            self.registry["catalogs"].append(path.relative_to(self.dist).as_posix())
        self.write_registry()

    @staticmethod
    def payload_bytes(module, target, payload_type):
        return f"CPU fixture bytes:{module}:{target}:{payload_type}".encode()

    def runner_path(self, target):
        return (
            self.dist
            / "bin"
            / parse_gpu_target(target).slug
            / "therock_module_validation"
        )

    def add_runner(self, target_id, **changes):
        target = parse_gpu_target(target_id)
        config = {
            "tag": target.slug,
            "description": runner_description(target.vendor).record(),
            "inventory": _inventory(target_id),
            **changes,
        }
        self.configs[target_id] = config
        path = self.runner_path(target_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_runner(path, config)
        self.registry["runners"].append(
            {
                "target": target_id,
                "path": path.relative_to(self.dist).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "description": runner_description(target.vendor).record(),
            }
        )

    @staticmethod
    def _write_runner(path, config):
        path.write_text(
            f"#!{sys.executable}\n"
            + "import json\n"
            + f"CONFIG = json.loads({json.dumps(config)!r})\n"
            + _RUNNER_BODY
        )
        path.chmod(0o755)

    def change_runner(self, target_id, **changes):
        self.configs[target_id].update(changes)
        path = self.runner_path(target_id)
        self._write_runner(path, self.configs[target_id])
        for entry in self.registry["runners"]:
            if entry["target"] == target_id:
                entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_registry()

    def write_registry(self):
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(self.registry, indent=2))

    def invoke(self, *arguments, success=True):
        for path in self.logs.iterdir():
            path.unlink()
        env = os.environ.copy()
        source_package = str(REPO_ROOT / "rocm-systems/shared/kpack/python")
        env["PYTHONPATH"] = os.pathsep.join(
            [source_package] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        env["THEROCK_DISPATCH_TEST_LOGS"] = str(self.logs)
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--dist-root", str(self.dist), *arguments],
            text=True,
            capture_output=True,
            env=env,
            timeout=20,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("Traceback", result.stderr)
        return result

    def run_module(
        self,
        *,
        target=_NVIDIA,
        payload_type="cubin",
        module=_MODULE,
        symbol=_SYMBOL,
        extra=(),
        success=True,
    ):
        return self.invoke(
            "run",
            "--module",
            module,
            "--target",
            target,
            "--format",
            payload_type,
            "--entry-point",
            symbol,
            *extra,
            success=success,
        )

    def calls(self, target):
        path = self.logs / (parse_gpu_target(target).slug + ".jsonl")
        return (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )

    def payload_record(self, target):
        return json.loads(
            (self.logs / (parse_gpu_target(target).slug + ".payload.json")).read_text()
        )

    def assert_no_payload_process(self):
        for path in self.logs.glob("*.jsonl"):
            for line in path.read_text().splitlines():
                self.assertNotIn("--payload", json.loads(line))
                self.assertNotIn("--module", json.loads(line))
        self.assertEqual(list(self.logs.glob("*.payload.json")), [])

    def assert_no_runner_process(self):
        self.assertEqual(list(self.logs.iterdir()), [])

    def test_list_reports_independent_runtime_unavailability_without_qualification(
        self,
    ):
        self.change_runner(_INTEL, list_mode="unavailable")
        result = self.invoke("list")
        report = json.loads(result.stdout)
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["kind"], "multi-vendor-device-inventory")
        self.assertEqual(set(report), {"schema_version", "kind", "runners"})
        entries = {entry["target"]: entry for entry in report["runners"]}
        self.assertEqual(set(entries), set(self.targets))
        for target in (_AMD, _NVIDIA):
            self.assertEqual(entries[target]["status"], "available")
            self.assertEqual(
                entries[target]["inventory"], self.configs[target]["inventory"]
            )
            self.assertIsNone(entries[target]["error"])
            self.assertEqual(entries[target]["matching_device_indices"], [0])
        self.assertEqual(entries[_INTEL]["status"], "unavailable")
        self.assertEqual(entries[_INTEL]["matching_device_indices"], [])
        self.assertIsNone(entries[_INTEL]["inventory"])
        self.assertIsInstance(entries[_INTEL]["error"], str)
        self.assertTrue(entries[_INTEL]["error"])
        self.assertNotIn("qualification", result.stdout)
        self.assert_no_payload_process()

    def test_empty_device_inventory_is_available_but_cannot_run(self):
        self.change_runner(_NVIDIA, inventory=_inventory(_NVIDIA, devices=[]))
        report = json.loads(self.invoke("list").stdout)
        entry = next(item for item in report["runners"] if item["target"] == _NVIDIA)
        self.assertEqual(entry["status"], "available")
        self.assertEqual(entry["inventory"]["devices"], [])
        self.assertEqual(entry["matching_device_indices"], [])
        self.run_module(success=False)
        self.assert_no_payload_process()

    def test_tampered_other_registry_runner_aborts_list_before_any_query(self):
        path = self.runner_path(_INTEL)
        path.write_text(path.read_text() + "\n# tampered fixture executable\n")
        self.invoke("list", success=False)
        self.assert_no_runner_process()

    def test_tampered_selected_runner_aborts_run_before_any_query(self):
        path = self.runner_path(_NVIDIA)
        path.write_text(path.read_text() + "\n# tampered fixture executable\n")
        self.run_module(success=False)
        self.assert_no_runner_process()

    def test_corrupt_registry_is_rejected_before_any_query(self):
        baseline = copy.deepcopy(self.registry)
        variants = []
        for key, value in (
            ("schema_version", True),
            ("schema_version", 2),
            ("kind", "other"),
            ("extra", 0),
        ):
            changed = copy.deepcopy(baseline)
            changed[key] = value
            variants.append(json.dumps(changed))
        changed = copy.deepcopy(baseline)
        changed["runners"][0]["path"] = "../escaped-runner"
        variants.append(json.dumps(changed))
        changed = copy.deepcopy(baseline)
        changed["runners"].append(changed["runners"][0])
        variants.append(json.dumps(changed))
        variants.append(
            json.dumps(baseline).replace(
                '"schema_version": 1', '"schema_version": 1, "schema_version": 1', 1
            )
        )
        variants.append("{")
        for index, text in enumerate(variants):
            with self.subTest(index=index):
                self.registry_path.write_text(text)
                self.invoke("list", success=False)
                self.assert_no_runner_process()

    def test_runner_symlink_cannot_escape_distribution(self):
        path = self.runner_path(_NVIDIA)
        outside = self.root / "outside-runner"
        path.rename(outside)
        path.symlink_to(outside)
        self.invoke("list", success=False)
        self.assert_no_runner_process()

    def test_exact_nvidia_dispatch_does_not_query_unavailable_intel(self):
        self.change_runner(_INTEL, list_mode="unavailable", describe_exit=19)
        self.run_module()
        self.assertEqual(self.calls(_INTEL), [])
        self.assertEqual(self.calls(_AMD), [])
        calls = self.calls(_NVIDIA)
        self.assertIn(["--describe-contract"], calls)
        self.assertIn(["--list-devices"], calls)
        launches = [call for call in calls if "--payload" in call]
        self.assertEqual(len(launches), 1)
        self.assertEqual(calls[-1], launches[0])
        record = self.payload_record(_NVIDIA)
        contract = validation_contract()
        self.assertEqual(record["launch_abi"], contract.abi)
        self.assertEqual(record["launch_abi_version"], str(contract.version))
        self.assertEqual(record["launch_contract_sha256"], contract.sha256)
        self.assertEqual(record["payload_format"], "cubin")
        self.assertEqual(record["symbol"], _SYMBOL)
        self.assertEqual(record["expect_arch"], "sm_120")
        self.assertEqual(record["device"], "0")
        self.assertEqual(record["expect_device_uuid"], _UUID_A)
        self.assertIsNone(record["expect_device_id"])
        self.assertEqual(
            bytes.fromhex(record["payload_bytes"]),
            self.payload_bytes("relu", _NVIDIA, "cubin"),
        )
        self.assertFalse(Path(record["payload"]).exists())

    def test_amd_and_intel_dispatch_preserve_native_identity_flags(self):
        for target, payload_type, expected_arch, expected_id in (
            (_AMD, "hsaco", "gfx1201", None),
            (_INTEL, "spirv", "spirv", "0xe223"),
        ):
            with self.subTest(target=target):
                self.run_module(target=target, payload_type=payload_type)
                record = self.payload_record(target)
                self.assertEqual(record["expect_arch"], expected_arch)
                self.assertEqual(record["expect_device_id"], expected_id)
                self.assertEqual(record["expect_device_uuid"], _UUID_A)
                self.assertEqual(
                    bytes.fromhex(record["payload_bytes"]),
                    self.payload_bytes("relu", target, payload_type),
                )
                if target == _INTEL:
                    self.assertIsNone(
                        self.configs[target]["inventory"]["devices"][0]["architecture"]
                    )

    def test_selected_ordinal_is_exact_and_never_searches_for_another_matching_device(
        self,
    ):
        inventory = _inventory(_NVIDIA)
        inventory["devices"][0]["architecture"] = "sm_90"
        matching = copy.deepcopy(inventory["devices"][0])
        matching.update(
            index=1,
            device_uuid=_UUID_B,
            architecture="sm_120",
            name="CPU fixture matching second device",
        )
        inventory["devices"].append(matching)
        self.change_runner(_NVIDIA, inventory=inventory)
        self.run_module(success=False)
        self.assert_no_payload_process()
        self.run_module(extra=("--device", "1"))
        self.assertEqual(self.payload_record(_NVIDIA)["device"], "1")
        self.assertEqual(self.payload_record(_NVIDIA)["expect_device_uuid"], _UUID_B)
        for ordinal in ("2", "-1"):
            with self.subTest(ordinal=ordinal):
                self.run_module(extra=("--device", ordinal), success=False)
                self.assert_no_payload_process()

    def test_uuid_selection_tracks_identity_when_ordinals_are_reordered(self):
        inventory = _inventory(_NVIDIA)
        second = copy.deepcopy(inventory["devices"][0])
        second.update(index=1, device_uuid=_UUID_B, name="Second same-architecture GPU")
        inventory["devices"].append(second)
        self.change_runner(_NVIDIA, inventory=inventory)
        self.run_module(extra=("--device-uuid", _UUID_A))
        self.assertEqual(self.payload_record(_NVIDIA)["device"], "0")
        self.assertEqual(self.payload_record(_NVIDIA)["expect_device_uuid"], _UUID_A)

        inventory["devices"].reverse()
        for index, device in enumerate(inventory["devices"]):
            device["index"] = index
        self.change_runner(_NVIDIA, inventory=inventory)
        result = self.run_module(extra=("--device-uuid", _UUID_A))
        self.assertEqual(self.payload_record(_NVIDIA)["device"], "1")
        self.assertEqual(self.payload_record(_NVIDIA)["expect_device_uuid"], _UUID_A)
        dispatch = next(
            line for line in result.stdout.splitlines() if line.startswith("DISPATCH ")
        )
        selected = json.loads(dispatch.removeprefix("DISPATCH "))["device"]
        self.assertEqual(selected["device_uuid"], _UUID_A)
        self.assertEqual(selected["index"], 1)

    def test_missing_uuid_never_falls_back_to_ordinal_zero(self):
        result = self.run_module(extra=("--device-uuid", _UUID_MISSING), success=False)
        self.assertIn("UUID", result.stderr)
        self.assertIn(["--list-devices"], self.calls(_NVIDIA))
        self.assert_no_payload_process()
        self.assertEqual(self.calls(_AMD), [])
        self.assertEqual(self.calls(_INTEL), [])

    def test_uuid_selection_preserves_exact_target_identity(self):
        inventory = _inventory(_NVIDIA)
        inventory["devices"][0]["architecture"] = "sm_90"
        second = copy.deepcopy(inventory["devices"][0])
        second.update(index=1, device_uuid=_UUID_B, architecture="sm_120")
        inventory["devices"].append(second)
        self.change_runner(_NVIDIA, inventory=inventory)
        self.run_module(extra=("--device-uuid", _UUID_A), success=False)
        self.assert_no_payload_process()
        self.run_module(extra=("--device-uuid", _UUID_B))
        self.assertEqual(self.payload_record(_NVIDIA)["device"], "1")
        self.assertEqual(self.payload_record(_NVIDIA)["expect_device_uuid"], _UUID_B)

    def test_malformed_uuid_fails_before_any_runner_query(self):
        for value in (
            "",
            "0" * 32,
            _UUID_A.upper(),
            _UUID_A[:-1],
            _UUID_A + "0",
            "g" * 32,
            " " + _UUID_A,
            _UUID_A + "\n",
            "01234567-89ab-cdef-0123-456789abcdef",
        ):
            with self.subTest(value=value):
                result = self.run_module(extra=("--device-uuid", value), success=False)
                self.assertIn("UUID", result.stderr)
                self.assert_no_runner_process()

    def test_uuid_and_ordinal_options_are_mutually_exclusive(self):
        for extra in (
            ("--device", "0", "--device-uuid", _UUID_A),
            ("--device-uuid", _UUID_A, "--device", "0"),
        ):
            with self.subTest(extra=extra):
                self.run_module(extra=extra, success=False)
                self.assert_no_runner_process()

    def test_old_inventory_without_uuid_is_unavailable_and_cannot_launch(self):
        inventory = _inventory(_NVIDIA)
        inventory["schema_version"] = 1
        del inventory["devices"][0]["device_uuid"]
        self.change_runner(_NVIDIA, inventory=inventory)
        report = json.loads(self.invoke("list").stdout)
        entry = next(item for item in report["runners"] if item["target"] == _NVIDIA)
        self.assertEqual(entry["status"], "unavailable")
        self.assertIsNone(entry["inventory"])
        self.run_module(success=False)
        self.assert_no_payload_process()

    def test_missing_malformed_and_duplicate_inventory_uuid_prevent_launch(self):
        for value in (None, "", "0" * 32, _UUID_A.upper(), "duplicate"):
            with self.subTest(value=value):
                inventory = _inventory(_NVIDIA)
                if value is None:
                    del inventory["devices"][0]["device_uuid"]
                elif value == "duplicate":
                    second = copy.deepcopy(inventory["devices"][0])
                    second["index"] = 1
                    inventory["devices"].append(second)
                else:
                    inventory["devices"][0]["device_uuid"] = value
                self.change_runner(_NVIDIA, inventory=inventory)
                self.run_module(success=False)
                self.assert_no_payload_process()

    def test_native_uuid_mismatch_propagates_without_retrying_another_device(self):
        self.change_runner(_NVIDIA, launch_device_uuid=_UUID_B)
        for extra in ((), ("--device-uuid", _UUID_A)):
            with self.subTest(extra=extra):
                result = self.run_module(extra=extra, success=False)
                self.assertEqual(result.returncode, 23)
                self.assertIn("UUID mismatch", result.stderr)
                self.assertEqual(
                    self.payload_record(_NVIDIA)["expect_device_uuid"], _UUID_A
                )
                self.assertEqual(self.payload_record(_NVIDIA)["device"], "0")
                self.assertEqual(
                    len([call for call in self.calls(_NVIDIA) if "--payload" in call]),
                    1,
                )
                self.assertEqual(self.calls(_AMD), [])
                self.assertEqual(self.calls(_INTEL), [])

    def test_sm90_request_does_not_fall_back_to_sm120_runner_or_payload(self):
        target = "nvidia:cuda:sm_90"
        self.add_runner(target, inventory=_inventory(_NVIDIA))
        path = create_pack(
            self.dist / "share/therock/packs/sm90",
            "sm90-fixture",
            (
                PayloadInput(
                    _MODULE,
                    target,
                    "cubin",
                    (_SYMBOL,),
                    b"CPU sm90 payload",
                    contract=validation_contract(),
                ),
            ),
        )
        self.registry["catalogs"].append(path.relative_to(self.dist).as_posix())
        self.write_registry()
        report = json.loads(self.invoke("list").stdout)
        entry = next(item for item in report["runners"] if item["target"] == target)
        self.assertEqual(entry["status"], "available")
        self.assertEqual(entry["inventory"]["devices"][0]["architecture"], "sm_120")
        self.assertEqual(entry["matching_device_indices"], [])
        native = next(item for item in report["runners"] if item["target"] == _NVIDIA)
        self.assertEqual(native["matching_device_indices"], [0])
        self.run_module(target=target, success=False)
        self.assertEqual(self.calls(_NVIDIA), [])
        self.assert_no_payload_process()

    def test_b70_requires_observed_vendor_and_device_identity_and_rejects_conflict(
        self,
    ):
        for field, value in (("device_id", 0xE220), ("vendor_id", 0x10DE)):
            with self.subTest(field=field):
                inventory = _inventory(_INTEL)
                inventory["devices"][0][field] = value
                self.change_runner(_INTEL, inventory=inventory)
                self.run_module(target=_INTEL, payload_type="spirv", success=False)
                self.assert_no_payload_process()
        self.change_runner(_INTEL, inventory=_inventory(_INTEL))
        self.run_module(
            target=_INTEL,
            payload_type="spirv",
            extra=("--expect-device-id", "0xe220"),
            success=False,
        )
        self.assert_no_payload_process()
        self.run_module(
            target=_INTEL, payload_type="spirv", extra=("--expect-device-id", "57891")
        )
        self.assertEqual(self.payload_record(_INTEL)["expect_device_id"], "0xe223")

    def test_registry_description_vendor_mismatch_fails_before_query(self):
        entry = next(
            entry for entry in self.registry["runners"] if entry["target"] == _NVIDIA
        )
        entry["description"] = runner_description("amd").record()
        self.write_registry()
        self.run_module(success=False)
        self.assert_no_runner_process()

    def test_live_runner_description_mismatch_cannot_launch(self):
        self.change_runner(_NVIDIA, description=runner_description("amd").record())
        self.run_module(success=False)
        self.assert_no_payload_process()

    def test_inventory_limits_must_allow_the_selected_launch_contract(self):
        for change in ("threads", "dimension"):
            with self.subTest(change=change):
                inventory = _inventory(_NVIDIA)
                limits = inventory["devices"][0]["limits"]
                if change == "threads":
                    limits["max_threads_per_block"] = 64
                else:
                    limits["max_block_dimensions"][0] = 64
                self.change_runner(_NVIDIA, inventory=inventory)
                self.run_module(success=False)
                self.assert_no_payload_process()

    def test_invalid_or_oversized_inventory_is_unavailable_in_list_and_fails_run(self):
        for mode in ("malformed", "duplicate", "flood"):
            with self.subTest(mode=mode):
                self.change_runner(_NVIDIA, list_mode=mode)
                report = json.loads(self.invoke("list").stdout)
                entry = next(
                    item for item in report["runners"] if item["target"] == _NVIDIA
                )
                self.assertEqual(entry["status"], "unavailable")
                self.assertIsNone(entry["inventory"])
                self.run_module(success=False)
                self.assert_no_payload_process()

    def test_payload_selection_has_no_module_format_or_symbol_fallback(self):
        for changes in (
            {"module": "validation/missing"},
            {"payload_type": "spirv"},
            {"symbol": "undeclared_symbol"},
            {"target": "nvidia:cuda:sm_80"},
        ):
            with self.subTest(changes=changes):
                self.run_module(**changes, success=False)
                self.assert_no_payload_process()

    def test_runner_update_during_inventory_is_detected_before_launch(self):
        self.change_runner(_NVIDIA, mutate_on_list=True)
        self.run_module(success=False)
        self.assertIn(["--list-devices"], self.calls(_NVIDIA))
        self.assert_no_payload_process()

    def test_explicit_symbol_from_multi_entry_payload_is_preserved(self):
        payload = b"CPU fixture module exposing two contracted entry points"
        path = create_pack(
            self.dist / "share/therock/packs/shared",
            "shared-fixture",
            (
                PayloadInput(
                    "validation/shared",
                    _NVIDIA,
                    "cubin",
                    ("therock_module_saxpy", "therock_module_relu"),
                    payload,
                    contract=validation_contract(),
                ),
            ),
        )
        self.registry["catalogs"].append(path.relative_to(self.dist).as_posix())
        self.write_registry()
        self.run_module(module="validation/shared", symbol="therock_module_relu")
        record = self.payload_record(_NVIDIA)
        self.assertEqual(record["symbol"], "therock_module_relu")
        self.assertEqual(bytes.fromhex(record["payload_bytes"]), payload)

    def test_native_payload_exit_code_is_propagated(self):
        self.change_runner(_NVIDIA, launch_exit=7)
        result = self.run_module(success=False)
        self.assertEqual(result.returncode, 7)
        self.assertTrue(self.payload_record(_NVIDIA))

    def test_explicit_event_capability_is_required_before_driver_query(self):
        description = runner_description("nvidia").record()
        description["capabilities"].remove("cross-queue-events")
        self.change_runner(_NVIDIA, description=description)
        for entry in self.registry["runners"]:
            if entry["target"] == _NVIDIA:
                entry["description"] = description
        self.write_registry()
        result = self.run_module(
            extra=("--require-capability", "cross-queue-events"), success=False
        )
        self.assertIn("lacks required adapter capabilities", result.stderr)
        self.assert_no_runner_process()
        self.run_module()  # The older logical fixture contract remains compatible.

    def test_event_capability_request_succeeds_and_is_logged(self):
        result = self.run_module(extra=("--require-capability", "cross-queue-events"))
        self.assertIn('"required_capabilities": ["cross-queue-events"]', result.stdout)

    def test_unknown_and_duplicate_capability_requests_do_not_query(self):
        for values in [("unknown",), ("cross-queue-events", "cross-queue-events")]:
            with self.subTest(values=values):
                result = self.run_module(
                    extra=tuple(
                        argument
                        for value in values
                        for argument in ("--require-capability", value)
                    ),
                    success=False,
                )
                self.assertIn("capability", result.stderr)
                self.assert_no_runner_process()

    def run_batch(self, *, target=_NVIDIA, requests=None, extra=(), success=True):
        if requests is None:
            fmt = {"amd": "hsaco", "nvidia": "cubin", "intel": "spirv"}[
                parse_gpu_target(target).vendor
            ]
            requests = (
                ("validation/saxpy", fmt, "therock_module_saxpy"),
                ("validation/relu", fmt, "therock_module_relu"),
            )
        arguments = ["run-batch", "--target", target]
        for request in requests:
            arguments.extend(("--module", *request))
        return self.invoke(*arguments, *extra, success=success)

    def test_batch_runs_separate_packs_in_one_process_for_each_backend(self):
        for target in self.targets:
            with self.subTest(target=target):
                result = self.run_batch(target=target)
                calls = self.calls(target)
                self.assertEqual(calls.count(["--list-devices"]), 1)
                launches = [call for call in calls if "--module" in call]
                self.assertEqual(len(launches), 1)
                self.assertNotIn("--payload", launches[0])
                record = self.payload_record(target)
                self.assertEqual(record["expect_device_uuid"], _UUID_A)
                self.assertEqual(
                    record["launch_contract_sha256"], validation_contract().sha256
                )
                self.assertEqual(len(record["modules"]), 2)
                for name, item in zip(("saxpy", "relu"), record["modules"]):
                    self.assertEqual(item["symbol"], f"therock_module_{name}")
                    self.assertEqual(
                        bytes.fromhex(item["payload_bytes"]),
                        self.payload_bytes(name, target, item["format"]),
                    )
                    self.assertFalse(Path(item["path"]).exists())
                dispatch = json.loads(
                    next(
                        line.removeprefix("DISPATCH ")
                        for line in result.stdout.splitlines()
                        if line.startswith("DISPATCH ")
                    )
                )
                self.assertEqual(dispatch["mode"], "multi-module-session")
                self.assertEqual(
                    dispatch["required_capabilities"],
                    ["cross-queue-events", "multi-module-session"],
                )
                self.assertEqual(
                    [item["module"] for item in dispatch["modules"]],
                    ["validation/saxpy", "validation/relu"],
                )
                for other in set(self.targets) - {target}:
                    self.assertEqual(self.calls(other), [])

    def test_batch_mixes_explicit_native_and_jit_formats_without_fallback(self):
        requests = (
            ("validation/saxpy", "cubin", "therock_module_saxpy"),
            ("validation/relu", "ptx", "therock_module_relu"),
        )
        self.run_batch(requests=requests)
        self.assertEqual(
            [item["format"] for item in self.payload_record(_NVIDIA)["modules"]],
            ["cubin", "ptx"],
        )
        self.run_batch(
            requests=(*requests, ("validation/saxpy", "ptx", "therock_module_saxpy"))
        )
        self.assertEqual(len(self.payload_record(_NVIDIA)["modules"]), 3)

    def test_bad_later_batch_request_prevents_all_runner_queries(self):
        first = ("validation/saxpy", "cubin", "therock_module_saxpy")
        for later in (
            ("validation/missing", "cubin", "therock_module_relu"),
            ("validation/relu", "spirv", "therock_module_relu"),
            ("validation/relu", "unknown", "therock_module_relu"),
            ("validation/relu", "ptx", "undeclared"),
            first,
        ):
            with self.subTest(later=later):
                self.run_batch(requests=(first, later), success=False)
                self.assert_no_runner_process()
        requests = tuple(
            (f"validation/item{i}", "cubin", "therock_module_saxpy") for i in range(33)
        )
        result = self.run_batch(requests=requests, success=False)
        self.assertIn("between 1 and 32", result.stderr)
        self.assert_no_runner_process()
        self.run_batch(requests=(), success=False)
        self.assert_no_runner_process()

    def test_corrupt_later_pack_prevents_running_the_valid_first_pack(self):
        catalog_path = self.dist / self.registry["catalogs"][1]
        catalog = json.loads(catalog_path.read_text())
        archive = catalog_path.parent / catalog["packs"][0]["path"]
        archive.write_bytes(archive.read_bytes() + b"tampered")
        result = self.run_batch(success=False)
        self.assertIn("Pack SHA256 mismatch", result.stderr)
        self.assert_no_runner_process()

    def test_batch_requires_session_capability_before_query_and_allows_legacy_run(self):
        description = runner_description("nvidia").record()
        description["capabilities"].remove("multi-module-session")
        self.change_runner(_NVIDIA, description=description)
        for entry in self.registry["runners"]:
            if entry["target"] == _NVIDIA:
                entry["description"] = description
        self.write_registry()
        result = self.run_batch(success=False)
        self.assertIn("multi-module-session", result.stderr)
        self.assert_no_runner_process()
        self.run_module()

    def test_batch_uuid_identity_rechecked_and_failure_removes_every_snapshot(self):
        self.change_runner(_NVIDIA, launch_device_uuid=_UUID_B)
        result = self.run_batch(extra=("--device-uuid", _UUID_A), success=False)
        self.assertEqual(result.returncode, 23)
        for item in self.payload_record(_NVIDIA)["modules"]:
            self.assertFalse(Path(item["path"]).exists())

    def test_batch_native_failure_propagates_with_all_snapshots_removed(self):
        self.change_runner(_NVIDIA, launch_exit=19)
        result = self.run_batch(success=False)
        self.assertEqual(result.returncode, 19)
        for item in self.payload_record(_NVIDIA)["modules"]:
            self.assertFalse(Path(item["path"]).exists())

    def test_all_batch_bytes_survive_catalog_changes_during_description(self):
        paths = [str(self.dist / path) for path in self.registry["catalogs"]]
        self.change_runner(_NVIDIA, mutate_catalogs_on_describe=paths)
        self.run_batch()
        record = self.payload_record(_NVIDIA)
        for name, item in zip(("saxpy", "relu"), record["modules"]):
            self.assertEqual(
                bytes.fromhex(item["payload_bytes"]),
                self.payload_bytes(name, _NVIDIA, "cubin"),
            )
        for path in paths:
            self.assertEqual(
                json.loads(Path(path).read_text()), {"after": "extraction"}
            )

    def run_pipeline(self, *, target=_NVIDIA, requests=None, extra=(), success=True):
        if requests is None:
            fmt = {"amd": "hsaco", "nvidia": "cubin", "intel": "spirv"}[
                parse_gpu_target(target).vendor
            ]
            requests = (
                ("validation/saxpy", fmt, "therock_module_saxpy"),
                ("validation/relu", fmt, "therock_module_relu"),
                ("validation/saxpy", fmt, "therock_module_saxpy"),
            )
        arguments = ["run-pipeline", "--target", target]
        for request in requests:
            arguments.extend(("--module", *request))
        return self.invoke(*arguments, *extra, success=success)

    def test_pipeline_preserves_repeated_stage_order_and_device_binding(self):
        for target in self.targets:
            with self.subTest(target=target):
                result = self.run_pipeline(
                    target=target, extra=("--device-uuid", _UUID_A)
                )
                record = self.payload_record(target)
                self.assertTrue(record["pipeline"])
                self.assertEqual(record["expect_device_uuid"], _UUID_A)
                self.assertEqual(
                    [item["symbol"] for item in record["modules"]],
                    [
                        "therock_module_saxpy",
                        "therock_module_relu",
                        "therock_module_saxpy",
                    ],
                )
                self.assertEqual(
                    record["modules"][0]["payload_bytes"],
                    record["modules"][2]["payload_bytes"],
                )
                self.assertEqual(
                    len([call for call in self.calls(target) if "--pipeline" in call]),
                    1,
                )
                self.assertEqual(self.calls(target).count(["--list-devices"]), 1)
                for item in record["modules"]:
                    self.assertFalse(Path(item["path"]).exists())
                dispatch = json.loads(
                    next(
                        line.removeprefix("DISPATCH ")
                        for line in result.stdout.splitlines()
                        if line.startswith("DISPATCH ")
                    )
                )
                self.assertEqual(dispatch["mode"], "device-module-pipeline")
                self.assertEqual(
                    dispatch["required_capabilities"],
                    [
                        "cross-queue-events",
                        "device-module-pipeline",
                        "multi-module-session",
                    ],
                )
                self.assertEqual(len(dispatch["modules"]), 3)

    def test_pipeline_mixes_explicit_formats_and_rejects_a_bad_later_stage(self):
        requests = (
            ("validation/saxpy", "cubin", "therock_module_saxpy"),
            ("validation/relu", "ptx", "therock_module_relu"),
            ("validation/saxpy", "cubin", "therock_module_saxpy"),
        )
        self.run_pipeline(requests=requests)
        self.assertEqual(
            [item["format"] for item in self.payload_record(_NVIDIA)["modules"]],
            ["cubin", "ptx", "cubin"],
        )
        for last in (
            ("validation/missing", "cubin", "therock_module_saxpy"),
            ("validation/relu", "spirv", "therock_module_relu"),
            ("validation/relu", "ptx", "undeclared"),
        ):
            with self.subTest(last=last):
                self.run_pipeline(requests=(*requests, last), success=False)
                self.assert_no_runner_process()

    def test_session_adapter_cannot_silently_run_a_device_pipeline(self):
        description = runner_description("nvidia").record()
        description["capabilities"].remove("device-module-pipeline")
        self.change_runner(_NVIDIA, description=description)
        for entry in self.registry["runners"]:
            if entry["target"] == _NVIDIA:
                entry["description"] = description
        self.write_registry()
        result = self.run_pipeline(success=False)
        self.assertIn("device-module-pipeline", result.stderr)
        self.assert_no_runner_process()
        self.run_batch()
        self.assertFalse(self.payload_record(_NVIDIA)["pipeline"])

    def test_pipeline_native_failure_propagates_and_cleans_all_stage_files(self):
        self.change_runner(_NVIDIA, launch_exit=29)
        result = self.run_pipeline(success=False)
        self.assertEqual(result.returncode, 29)
        for item in self.payload_record(_NVIDIA)["modules"]:
            self.assertFalse(Path(item["path"]).exists())

    def test_pipeline_stage_count_is_bounded_even_when_stages_repeat(self):
        request = ("validation/saxpy", "cubin", "therock_module_saxpy")
        self.run_pipeline(requests=(request,))
        self.run_pipeline(requests=(request,) * 32)
        self.assertEqual(len(self.payload_record(_NVIDIA)["modules"]), 32)
        result = self.run_pipeline(requests=(request,) * 33, success=False)
        self.assertIn("between 1 and 32", result.stderr)
        self.assert_no_runner_process()
        self.run_pipeline(requests=(), success=False)
        self.assert_no_runner_process()

    def test_service_capability_is_required_before_any_runner_query(self):
        description = runner_description("nvidia").record()
        description["capabilities"].remove("persistent-module-service")
        self.change_runner(_NVIDIA, description=description)
        for entry in self.registry["runners"]:
            if entry["target"] == _NVIDIA:
                entry["description"] = description
        self.write_registry()
        result = self.invoke(
            "run-service",
            "--target",
            _NVIDIA,
            "--module",
            "validation/saxpy",
            "cubin",
            "therock_module_saxpy",
            success=False,
        )
        self.assertIn("persistent-module-service", result.stderr)
        self.assert_no_runner_process()

    def test_service_fixture_rejects_bad_later_selection_and_excess_stages(self):
        first = ("--module", "validation/saxpy", "cubin", "therock_module_saxpy")
        self.invoke(
            "run-service",
            "--target",
            _NVIDIA,
            *first,
            "--module",
            "validation/missing",
            "cubin",
            "therock_module_saxpy",
            success=False,
        )
        self.assert_no_runner_process()
        result = self.invoke(
            "run-service", "--target", _NVIDIA, *(first * 4), success=False
        )
        self.assertIn("one to three stages", result.stderr)
        self.assert_no_runner_process()


if __name__ == "__main__":
    unittest.main()
