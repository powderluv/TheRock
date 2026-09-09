# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _therock_utils.gpu_targets import parse_gpu_target
from _therock_utils.module_contract import (
    parse_runner_description,
    require_runner_capabilities,
    require_runner_compatibility,
    runner_description,
    validation_contract,
    validate_required_capabilities,
)


class AdapterCapabilitiesTest(unittest.TestCase):
    def test_event_implementation_does_not_change_logical_kernel_contract(self):
        contract = validation_contract()
        self.assertEqual(contract.version, 1)
        self.assertEqual(
            contract.sha256,
            "647e4ce315c7f3777ce40788f2ac62b1a62d1f964d7d0cbd4e8cfcfd7412ae8b",
        )
        self.assertNotIn("cross-queue-events", contract.required_capabilities)
        self.assertNotIn("multi-module-session", contract.required_capabilities)
        self.assertNotIn("device-module-pipeline", contract.required_capabilities)
        self.assertNotIn("persistent-module-service", contract.required_capabilities)
        for vendor in ("amd", "nvidia", "intel"):
            require_runner_capabilities(
                runner_description(vendor),
                (
                    "cross-queue-events",
                    "multi-module-session",
                    "device-module-pipeline",
                ),
            )

    def test_older_adapter_remains_compatible_without_event_requirement(self):
        description = runner_description("nvidia").record()
        description["capabilities"].remove("cross-queue-events")
        adapter = parse_runner_description(description)
        require_runner_compatibility(
            adapter,
            parse_gpu_target("nvidia:cuda:sm_120"),
            ("cubin",),
            ("therock_module_saxpy",),
            validation_contract(),
        )
        require_runner_capabilities(adapter, ())
        with self.assertRaisesRegex(ValueError, "cross-queue-events"):
            require_runner_capabilities(adapter, ("cross-queue-events",))

    def test_requirements_are_strict_and_normalized(self):
        self.assertEqual(
            validate_required_capabilities(("queue-synchronize", "cross-queue-events")),
            ("cross-queue-events", "queue-synchronize"),
        )
        for values in [
            ("unknown",),
            ("cross-queue-events", "cross-queue-events"),
            (True,),
        ]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_required_capabilities(values)


if __name__ == "__main__":
    unittest.main()
