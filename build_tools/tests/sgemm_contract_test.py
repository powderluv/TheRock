# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SGEMM metadata is optional and independent of the original vector ABI."""

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

from _therock_utils.gpu_targets import parse_gpu_target
from _therock_utils.module_contract import (
    require_runner_compatibility,
    runner_description,
    validate_required_capabilities,
    validation_contract,
)
from _therock_utils.sgemm_contract import (
    SGEMM_ABI,
    SGEMM_CAPABILITY,
    SGEMM_VERSION,
    SgemmProviderInfo,
    parse_sgemm_provider_json,
    sgemm_capabilities,
    sgemm_contract_record,
    sgemm_contract_sha256,
    sgemm_provider_for_vendor,
)
from configure_module_contract import configure_contract


BASE_CAPABILITIES = {
    "device-allocation",
    "kernel-launch",
    "module-load",
    "ordered-copy",
    "queue-synchronize",
    "cross-queue-events",
    "multi-module-session",
    "device-module-pipeline",
    "persistent-module-service",
}
VECTOR_SHA = "647e4ce315c7f3777ce40788f2ac62b1a62d1f964d7d0cbd4e8cfcfd7412ae8b"
SGEMM_SHA = "fe1ce4851d9a3d5798e19b5e25ece80e4b4b4bbcdbf9dd350cea8912f3e2a5a0"


def provider(vendor="nvidia"):
    return SgemmProviderInfo(
        vendor,
        sgemm_provider_for_vendor(vendor),
        "13.2.0-observed",
        SGEMM_ABI,
        SGEMM_VERSION,
        sgemm_contract_sha256(),
        sgemm_capabilities(vendor),
    )


class SgemmContractTest(unittest.TestCase):
    def test_default_descriptions_keep_original_schema_capabilities_and_vector_hash(
        self,
    ):
        for vendor in ("amd", "nvidia", "intel"):
            with self.subTest(vendor=vendor):
                description = runner_description(vendor)
                self.assertEqual(
                    description, runner_description(vendor, enable_sgemm=False)
                )
                self.assertEqual(set(description.capabilities), BASE_CAPABILITIES)
                self.assertEqual(description.record()["schema_version"], 1)
                self.assertEqual(description.contract_sha256, VECTOR_SHA)
                self.assertEqual(description.contract, validation_contract())

    def test_enabled_capability_pair_is_vendor_specific_and_contract_stays_fixed(self):
        for vendor, identity in (
            ("amd", "amd:hip:gfx1201"),
            ("nvidia", "nvidia:cuda:sm_120"),
        ):
            with self.subTest(vendor=vendor):
                description = runner_description(vendor, enable_sgemm=True)
                self.assertEqual(
                    set(description.capabilities),
                    BASE_CAPABILITIES | set(sgemm_capabilities(vendor)),
                )
                self.assertEqual(description.contract_sha256, VECTOR_SHA)
                require_runner_compatibility(
                    description,
                    parse_gpu_target(identity),
                    description.payload_types,
                    description.entry_points,
                    validation_contract(),
                )
                self.assertEqual(
                    validate_required_capabilities(sgemm_capabilities(vendor)),
                    sgemm_capabilities(vendor),
                )
        with self.assertRaises(ValueError):
            runner_description("amd", enable_sgemm=1)

    def test_compatibility_rejects_partial_wrong_vendor_or_multiple_provider_pairs(
        self,
    ):
        for vendor, identity in (
            ("amd", "amd:hip:gfx1201"),
            ("nvidia", "nvidia:cuda:sm_120"),
            ("intel", "intel:level-zero:xe2-b70"),
        ):
            description = runner_description(vendor)
            for extension in (
                (SGEMM_CAPABILITY,),
                ("blas-provider-rocblas-v1",),
                ("blas-provider-cublas-v1",),
                (
                    SGEMM_CAPABILITY,
                    "blas-provider-cublas-v1",
                    "blas-provider-rocblas-v1",
                ),
                sgemm_capabilities("nvidia" if vendor != "nvidia" else "amd"),
            ):
                with self.subTest(
                    vendor=vendor, extension=extension
                ), self.assertRaises(ValueError):
                    changed = replace(
                        description, capabilities=description.capabilities + extension
                    )
                    require_runner_compatibility(
                        changed,
                        parse_gpu_target(identity),
                        changed.payload_types,
                        changed.entry_points,
                        validation_contract(),
                    )

    def test_operation_contract_is_fresh_bounded_and_has_independent_stable_digest(
        self,
    ):
        record = sgemm_contract_record()
        self.assertEqual(record["dimensions"], {"minimum": 1, "maximum": 256})
        self.assertEqual(record["maximum_buffer_capacity"], 65556)
        self.assertEqual(record["maximum_leading_dimension"], 65556)
        self.assertEqual(record["layout"], "column-major")
        self.assertIs(record["transpose_a"], False)
        self.assertIs(record["transpose_b"], False)
        canonical = json.dumps(
            record, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), SGEMM_SHA)
        self.assertEqual(sgemm_contract_sha256(), SGEMM_SHA)
        self.assertNotEqual(SGEMM_SHA, VECTOR_SHA)
        record["dimensions"]["maximum"] = 1
        self.assertEqual(sgemm_contract_record()["dimensions"]["maximum"], 256)

    def test_provider_roundtrip_preserves_observed_version(self):
        for vendor in ("amd", "nvidia"):
            info = provider(vendor)
            self.assertEqual(
                parse_sgemm_provider_json(json.dumps(info.record()), vendor=vendor),
                info,
            )
            self.assertEqual(info.library_version, "13.2.0-observed")
            self.assertEqual(info.record()["scope"], "loaded-provider")

    def test_provider_parser_rejects_mismatched_or_malformed_metadata(self):
        baseline = provider().record()
        for key, value in (
            ("schema_version", True),
            ("schema_version", 2),
            ("kind", "other"),
            ("scope", "compiled-adapter"),
            ("vendor", "amd"),
            ("provider", "rocblas"),
            ("library_version", ""),
            ("library_version", "  "),
            ("library_version", "bad\nversion"),
            ("library_version", 130200),
            ("abi", "therock.validation.f32-vector"),
            ("version", True),
            ("version", 2),
            ("contract_sha256", VECTOR_SHA),
            ("contract_sha256", SGEMM_SHA.upper()),
            ("capabilities", [SGEMM_CAPABILITY]),
            ("capabilities", list(sgemm_capabilities("amd"))),
            ("capabilities", list(sgemm_capabilities("nvidia")) + [SGEMM_CAPABILITY]),
            ("extra", "field"),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                changed = copy.deepcopy(baseline)
                changed[key] = value
                parse_sgemm_provider_json(json.dumps(changed), vendor="nvidia")
        text = json.dumps(baseline)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_sgemm_provider_json(
                text.replace(
                    '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
                ),
                vendor="nvidia",
            )
        for text in ("{", "[]", "null"):
            with self.assertRaises(ValueError):
                parse_sgemm_provider_json(text, vendor="nvidia")

    def test_generator_opt_in_is_explicit_and_invalid_input_preserves_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            header, description = root / "contract.h", root / "description.json"
            configure_contract("nvidia", header, description)
            baseline = description.read_bytes()
            self.assertIn("kSgemmEnabled = false;", header.read_text())
            self.assertIn('kSgemmProvider[] = "";', header.read_text())
            self.assertIn(SGEMM_SHA, header.read_text())
            configure_contract("nvidia", header, description, enable_sgemm=False)
            self.assertEqual(description.read_bytes(), baseline)
            result = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "configure_module_contract.py"),
                    "--vendor",
                    "nvidia",
                    "--header",
                    str(header),
                    "--description",
                    str(description),
                    "--enable-sgemm",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("kSgemmEnabled = true;", header.read_text())
            self.assertIn('kSgemmProvider[] = "cublas";', header.read_text())
            self.assertEqual(
                json.loads(description.read_text()),
                runner_description("nvidia", enable_sgemm=True).record(),
            )
            before = (header.read_bytes(), description.read_bytes())
            with self.assertRaises(ValueError):
                configure_contract("intel", header, description, enable_sgemm=1)
            self.assertEqual((header.read_bytes(), description.read_bytes()), before)


if __name__ == "__main__":
    unittest.main()
