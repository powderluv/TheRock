# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.gpu_targets import parse_gpu_target
from _therock_utils.module_contract import (
    parse_contract,
    parse_runner_description,
    parse_runner_description_json,
    require_runner_compatibility,
    runner_description,
    validation_contract,
    verify_contract,
)
from configure_module_contract import configure_contract


class ModuleContractTest(unittest.TestCase):
    def test_fixed_logical_arguments_and_execution_requirements(self):
        contract = validation_contract()
        self.assertEqual(contract.abi, "therock.validation.f32-vector")
        self.assertEqual(contract.version, 1)
        self.assertEqual(contract.pointer_bits, 64)
        self.assertEqual(
            [argument.name for argument in contract.arguments],
            ["x", "y", "output", "alpha", "count"],
        )
        self.assertEqual(
            [argument.size_bytes for argument in contract.arguments], [8, 8, 8, 4, 4]
        )
        self.assertEqual(
            [argument.alignment_bytes for argument in contract.arguments],
            [8, 8, 8, 4, 4],
        )
        self.assertEqual(
            [argument.access for argument in contract.arguments],
            ["read-only", "read-only", "write-only", "value", "value"],
        )
        self.assertEqual(
            [argument.value_type for argument in contract.arguments],
            ["f32", "f32", "f32", "f32", "u32"],
        )
        self.assertEqual(contract.launch.block, (128, 1, 1))
        self.assertEqual(contract.launch.dynamic_shared_bytes, 0)
        self.assertEqual(contract.ownership.device_allocations, "runner")
        self.assertEqual(contract.ownership.module_lifetime, "until-queue-complete")
        self.assertEqual(contract.execution.ordering, "in-order-copy-launch-copy")
        self.assertEqual(contract.execution.completion, "host-synchronized")
        self.assertEqual(
            set(contract.required_capabilities),
            {
                "device-allocation",
                "ordered-copy",
                "module-load",
                "kernel-launch",
                "queue-synchronize",
            },
        )

    def test_roundtrip_canonical_hash_and_argument_order(self):
        contract = validation_contract()
        record = contract.record()
        expected = hashlib.sha256(
            json.dumps(
                record, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode()
        ).hexdigest()
        self.assertEqual(contract.sha256, expected)
        self.assertEqual(verify_contract(record, expected), contract)
        record["required_capabilities"].reverse()
        self.assertEqual(parse_contract(record), contract)
        record["arguments"][0], record["arguments"][1] = (
            record["arguments"][1],
            record["arguments"][0],
        )
        alternate = parse_contract(record)
        self.assertNotEqual(alternate, contract)
        self.assertNotEqual(alternate.sha256, contract.sha256)

    def test_well_formed_future_contract_parses_but_is_not_supported(self):
        target = parse_gpu_target("nvidia:cuda:sm_120")
        base = validation_contract()
        variants = (
            replace(base, abi="future.vector"),
            replace(base, version=2),
            replace(base, launch=replace(base.launch, block=(64, 1, 1))),
            replace(base, execution=replace(base.execution, completion="asynchronous")),
            replace(base, ownership=replace(base.ownership, context="caller")),
        )
        for contract in variants:
            with self.subTest(contract=contract):
                parsed = parse_contract(contract.record())
                self.assertEqual(parsed, contract)
                description = replace(runner_description("nvidia"), contract=contract)
                with self.assertRaisesRegex(ValueError, "contract"):
                    require_runner_compatibility(
                        description,
                        target,
                        ("cubin",),
                        ("therock_module_saxpy",),
                        contract,
                    )

    def test_invalid_nested_contract_values_rejected(self):
        baseline = validation_contract().record()
        variants = []
        for key, value in (
            ("version", True),
            ("version", 0),
            ("pointer_bits", 16),
            ("pointer_bits", True),
            ("abi", "../bad"),
            ("extra", 1),
            ("arguments", []),
            ("required_capabilities", ["module-load", "module-load"]),
        ):
            changed = copy.deepcopy(baseline)
            changed[key] = value
            variants.append(changed)
        for key, value in (
            ("kind", "host-pointer"),
            ("value_type", "unknown"),
            ("size_bytes", True),
            ("size_bytes", 4),
            ("alignment_bytes", 3),
            ("access", "value"),
            ("extra", 1),
        ):
            changed = copy.deepcopy(baseline)
            changed["arguments"][0][key] = value
            variants.append(changed)
        changed = copy.deepcopy(baseline)
        changed["arguments"][1]["name"] = "x"
        variants.append(changed)
        for key, value in (
            ("block", [128, 1]),
            ("block", [128, 1, True]),
            ("block", [0, 1, 1]),
            ("dynamic_shared_bytes", -1),
            ("dynamic_shared_bytes", True),
            ("extra", 1),
        ):
            changed = copy.deepcopy(baseline)
            changed["launch"][key] = value
            variants.append(changed)
        changed = copy.deepcopy(baseline)
        changed["ownership"]["context"] = "implicit"
        variants.append(changed)
        changed = copy.deepcopy(baseline)
        changed["execution"]["completion"] = "host synchronized"
        variants.append(changed)
        for index, value in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                parse_contract(value)
        for digest in ("f" * 64, "A" * 64, "short", True):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                verify_contract(baseline, digest)

    def test_runner_descriptions_are_adapter_scope_and_backend_specific(self):
        for vendor, target, formats in (
            ("amd", "amd:hip:gfx1201", ("hsaco",)),
            ("nvidia", "nvidia:cuda:sm_120", ("cubin", "ptx")),
            ("intel", "intel:level-zero:xe2-b70", ("spirv",)),
        ):
            with self.subTest(vendor=vendor):
                description = runner_description(vendor)
                record = description.record()
                self.assertEqual(record["scope"], "compiled-adapter")
                self.assertNotIn("qualification", record)
                self.assertNotIn("devices", record)
                self.assertEqual(
                    parse_runner_description_json(json.dumps(record)), description
                )
                require_runner_compatibility(
                    description,
                    parse_gpu_target(target),
                    formats,
                    ("therock_module_saxpy", "therock_module_relu"),
                    validation_contract(),
                )

    def test_runner_schema_hash_vendor_formats_and_capability_relationship(self):
        baseline = runner_description("nvidia").record()
        variants = []
        for key, value in (
            ("schema_version", True),
            ("schema_version", 2),
            ("scope", "hardware-qualified"),
            ("kind", "other"),
            ("vendor", "amd"),
            ("backend", "hip"),
            ("payload_types", ["hsaco"]),
            ("payload_types", ["cubin", "cubin"]),
            ("entry_points", ["bad symbol"]),
            ("capabilities", ["module-load"]),
            ("contract_sha256", "0" * 64),
            ("extra", 0),
        ):
            changed = copy.deepcopy(baseline)
            changed[key] = value
            variants.append(changed)
        for index, value in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                parse_runner_description(value)
        text = json.dumps(baseline)
        for field, replacement in (
            ('"schema_version": 1', '"schema_version": 1, "schema_version": 1'),
            ('"version": 1', '"version": 1, "version": 1'),
            ('"size_bytes": 8', '"size_bytes": 8, "size_bytes": 8'),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "Duplicate JSON"
            ):
                parse_runner_description_json(text.replace(field, replacement))
        with self.assertRaises(ValueError):
            parse_runner_description_json("{")

    def test_compatibility_rejects_missing_symbol_format_vendor_and_unknown_operation(
        self,
    ):
        target = parse_gpu_target("nvidia:cuda:sm_120")
        description = runner_description("nvidia")
        for changed, requested_format, requested_symbol in (
            (runner_description("amd"), "cubin", "therock_module_saxpy"),
            (description, "spirv", "therock_module_saxpy"),
            (description, "cubin", "not_declared"),
            (
                replace(
                    description,
                    entry_points=description.entry_points + ("future_kernel",),
                ),
                "cubin",
                "future_kernel",
            ),
            (
                replace(
                    description,
                    capabilities=description.capabilities + ("future-operation",),
                ),
                "cubin",
                "therock_module_saxpy",
            ),
        ):
            with self.subTest(
                description=changed, format=requested_format, symbol=requested_symbol
            ), self.assertRaises(ValueError):
                require_runner_compatibility(
                    changed,
                    target,
                    (requested_format,),
                    (requested_symbol,),
                    validation_contract(),
                )


class ModuleContractGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.header = self.root / "module_contract_data.h"
        self.description = self.root / "runner-contract.json"

    def test_generated_header_compiles_and_reports_identical_json_without_gpu(self):
        # Host-boundary check: this program contains no GPU APIs or SDK linkage.
        configure_contract("intel", self.header, self.description)
        source = self.root / "describe.cpp"
        source.write_text(
            '#include "module_contract_data.h"\n#include <iostream>\nstatic_assert(therock::module_contract::kPointerBits == 64);\nstatic_assert(therock::module_contract::kGroupSize == 128);\nint main() { std::cout << therock::module_contract::kRunnerDescription; }\n'
        )
        binary = self.root / "describe"
        subprocess.run(
            ["c++", "-std=c++17", str(source), "-o", str(binary)],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            [str(binary)], check=True, text=True, capture_output=True
        )
        actual = parse_runner_description_json(result.stdout)
        self.assertEqual(actual.record(), json.loads(self.description.read_text()))
        self.assertEqual(actual, runner_description("intel"))

    def test_noop_generation_preserves_bytes_and_timestamps(self):
        configure_contract("nvidia", self.header, self.description)
        original = [
            (path.read_bytes(), path.stat().st_mtime_ns)
            for path in (self.header, self.description)
        ]
        configure_contract("nvidia", self.header, self.description)
        self.assertEqual(
            original,
            [
                (path.read_bytes(), path.stat().st_mtime_ns)
                for path in (self.header, self.description)
            ],
        )
        configure_contract("amd", self.header, self.description)
        self.assertEqual(
            parse_runner_description_json(self.description.read_text()),
            runner_description("amd"),
        )
        self.assertNotEqual(self.description.read_bytes(), original[1][0])

    def test_invalid_generation_preserves_existing_files_and_cli_fails_cleanly(self):
        configure_contract("amd", self.header, self.description)
        original = self.header.read_bytes(), self.description.read_bytes()
        with self.assertRaises(ValueError):
            configure_contract("unknown", self.header, self.description)
        with self.assertRaises(ValueError):
            configure_contract("amd", self.header, self.header)
        self.assertEqual(
            original, (self.header.read_bytes(), self.description.read_bytes())
        )
        tool = Path(__file__).parent.parent / "configure_module_contract.py"
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            [
                sys.executable,
                str(tool),
                "--vendor",
                "intel",
                "--header",
                str(self.header),
                "--description",
                str(self.description),
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            [
                sys.executable,
                str(tool),
                "--vendor",
                "bad",
                "--header",
                str(self.header),
                "--description",
                str(self.description),
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
