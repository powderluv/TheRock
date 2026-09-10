# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Intel oneMKL metadata remains opt-in and independent of hardware availability."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

from _therock_utils.gpu_targets import parse_gpu_target
from _therock_utils.module_contract import (
    parse_runner_description_json,
    require_runner_capabilities,
    require_runner_compatibility,
    runner_description,
    validate_required_capabilities,
    validation_contract,
)
from _therock_utils.sgemm_contract import (
    SGEMM_ABI,
    SGEMM_CAPABILITY,
    SGEMM_PROVIDER_CAPABILITIES,
    SGEMM_VERSION,
    SgemmProviderInfo,
    parse_sgemm_provider_json,
    sgemm_capabilities,
    sgemm_contract_record,
    sgemm_contract_sha256,
    sgemm_provider_for_vendor,
)

INTEL_TARGET = "intel:level-zero:xe2-b70"
INTEL_CAPABILITIES = ("blas-provider-onemkl-v1", SGEMM_CAPABILITY)
VECTOR_SHA = "647e4ce315c7f3777ce40788f2ac62b1a62d1f964d7d0cbd4e8cfcfd7412ae8b"
SGEMM_SHA = "fe1ce4851d9a3d5798e19b5e25ece80e4b4b4bbcdbf9dd350cea8912f3e2a5a0"


def observed_provider() -> SgemmProviderInfo:
    return SgemmProviderInfo(
        vendor="intel",
        provider="onemkl",
        library_version="test-observed-onemkl-version",
        abi=SGEMM_ABI,
        version=SGEMM_VERSION,
        contract_sha256=SGEMM_SHA,
        capabilities=INTEL_CAPABILITIES,
    )


class IntelSgemmContractTest(unittest.TestCase):
    def test_intel_opt_in_adds_only_its_provider_pair_and_preserves_vector_abi(
        self,
    ) -> None:
        default = runner_description("intel")
        enabled = runner_description("intel", enable_sgemm=True)
        self.assertEqual(default, runner_description("intel", enable_sgemm=False))
        self.assertTrue(set(INTEL_CAPABILITIES).isdisjoint(default.capabilities))
        self.assertEqual(
            set(enabled.capabilities) - set(default.capabilities),
            set(INTEL_CAPABILITIES),
        )
        self.assertEqual(
            dict(enabled.record(), capabilities=list(default.capabilities)),
            default.record(),
        )
        self.assertEqual(enabled.backend, "level-zero")
        self.assertEqual(enabled.payload_types, ("spirv",))
        self.assertEqual(enabled.contract_sha256, VECTOR_SHA)
        self.assertEqual(enabled.contract, validation_contract())
        self.assertEqual(sgemm_contract_sha256(), SGEMM_SHA)
        self.assertEqual(sgemm_provider_for_vendor("intel"), "onemkl")
        self.assertEqual(sgemm_capabilities("intel"), INTEL_CAPABILITIES)
        self.assertEqual(
            parse_runner_description_json(json.dumps(enabled.record())), enabled
        )
        require_runner_compatibility(
            enabled,
            parse_gpu_target(INTEL_TARGET),
            ("spirv",),
            enabled.entry_points,
            validation_contract(),
        )
        require_runner_capabilities(enabled, INTEL_CAPABILITIES)
        self.assertEqual(
            validate_required_capabilities(INTEL_CAPABILITIES), INTEL_CAPABILITIES
        )
        with self.assertRaisesRegex(ValueError, "lacks required adapter capabilities"):
            require_runner_capabilities(default, INTEL_CAPABILITIES)

    def test_existing_amd_nvidia_provider_choices_and_descriptions_are_unchanged(
        self,
    ) -> None:
        self.assertEqual(
            set(SGEMM_PROVIDER_CAPABILITIES),
            {
                "blas-provider-rocblas-v1",
                "blas-provider-cublas-v1",
                "blas-provider-onemkl-v1",
            },
        )
        for vendor, provider in (("amd", "rocblas"), ("nvidia", "cublas")):
            with self.subTest(vendor=vendor):
                default = runner_description(vendor)
                enabled = runner_description(vendor, enable_sgemm=True)
                expected = (f"blas-provider-{provider}-v1", SGEMM_CAPABILITY)
                self.assertEqual(sgemm_provider_for_vendor(vendor), provider)
                self.assertEqual(sgemm_capabilities(vendor), expected)
                self.assertEqual(
                    set(enabled.capabilities), set(default.capabilities) | set(expected)
                )
                self.assertNotIn("blas-provider-onemkl-v1", enabled.capabilities)
                self.assertEqual(
                    dict(enabled.record(), capabilities=list(default.capabilities)),
                    default.record(),
                )
                self.assertEqual(enabled.contract_sha256, VECTOR_SHA)

    def test_one_mkl_capabilities_cannot_be_partial_conflicting_or_cross_vendor(
        self,
    ) -> None:
        for vendor, target_id in (
            ("intel", INTEL_TARGET),
            ("amd", "amd:hip:gfx1201"),
            ("nvidia", "nvidia:cuda:sm_120"),
        ):
            default = runner_description(vendor)
            invalid = [
                ("blas-provider-onemkl-v1",),
                (SGEMM_CAPABILITY,),
                (*INTEL_CAPABILITIES, "blas-provider-rocblas-v1"),
                (*INTEL_CAPABILITIES, "blas-provider-cublas-v1"),
                (*INTEL_CAPABILITIES, "blas-provider-onemkl-v1"),
            ]
            if vendor != "intel":
                invalid.append(INTEL_CAPABILITIES)
            else:
                invalid.extend(sgemm_capabilities(other) for other in ("amd", "nvidia"))
            for extension in invalid:
                with self.subTest(
                    vendor=vendor, extension=extension
                ), self.assertRaises(ValueError):
                    changed = replace(
                        default, capabilities=default.capabilities + extension
                    )
                    require_runner_compatibility(
                        changed,
                        parse_gpu_target(target_id),
                        changed.payload_types,
                        changed.entry_points,
                        validation_contract(),
                    )

    def test_loaded_intel_provider_json_preserves_observed_version_and_exact_contract(
        self,
    ) -> None:
        info = observed_provider()
        parsed = parse_sgemm_provider_json(json.dumps(info.record()), vendor="intel")
        self.assertEqual(parsed, info)
        self.assertEqual(parsed.library_version, "test-observed-onemkl-version")
        self.assertEqual(parsed.contract_sha256, SGEMM_SHA)
        self.assertEqual(parsed.record()["scope"], "loaded-provider")
        self.assertEqual(parsed.record()["capabilities"], list(INTEL_CAPABILITIES))
        for other in ("amd", "nvidia"):
            with self.subTest(vendor=other), self.assertRaisesRegex(
                ValueError, "vendor mismatch"
            ):
                parse_sgemm_provider_json(json.dumps(info.record()), vendor=other)

    def test_loaded_intel_provider_rejects_mismatched_identity_and_contract(
        self,
    ) -> None:
        baseline = observed_provider().record()
        for key, value in (
            ("vendor", "amd"),
            ("vendor", "nvidia"),
            ("provider", "rocblas"),
            ("provider", "cublas"),
            ("provider", "oneMKL"),
            ("scope", "compiled-adapter"),
            ("library_version", ""),
            ("library_version", "invalid\nversion"),
            ("version", True),
            ("version", 2),
            ("abi", "therock.validation.f32-vector"),
            ("contract_sha256", VECTOR_SHA),
            ("contract_sha256", SGEMM_SHA.upper()),
            ("capabilities", [SGEMM_CAPABILITY]),
            ("capabilities", ["blas-provider-onemkl-v1"]),
            ("capabilities", list(sgemm_capabilities("amd"))),
            ("capabilities", list(sgemm_capabilities("nvidia"))),
            ("capabilities", [*INTEL_CAPABILITIES, "blas-provider-cublas-v1"]),
            ("capabilities", [*INTEL_CAPABILITIES, SGEMM_CAPABILITY]),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                changed = copy.deepcopy(baseline)
                changed[key] = value
                parse_sgemm_provider_json(json.dumps(changed), vendor="intel")
        duplicate = json.dumps(baseline).replace(
            '"provider": "onemkl"', '"provider": "onemkl", "provider": "onemkl"'
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_sgemm_provider_json(duplicate, vendor="intel")

    def test_cli_generates_intel_onemkl_header_and_restores_default_on_disable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="therock-intel-sgemm-contract-"
        ) as temporary:
            root = Path(temporary)
            header, description = root / "contract.h", root / "description.json"
            command = [
                sys.executable,
                str(TOOLS / "configure_module_contract.py"),
                "--vendor",
                "intel",
                "--header",
                str(header),
                "--description",
                str(description),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            default = (header.read_bytes(), description.read_bytes())
            self.assertIn("kSgemmEnabled = false;", header.read_text())
            self.assertIn('kSgemmProvider[] = "";', header.read_text())
            result = subprocess.run(
                [*command, "--enable-sgemm"], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            generated = header.read_text()
            self.assertIn("kSgemmEnabled = true;", generated)
            self.assertIn('kSgemmProvider[] = "onemkl";', generated)
            self.assertIn(f'kSgemmAbi[] = "{SGEMM_ABI}";', generated)
            self.assertIn(f"kSgemmVersion = {SGEMM_VERSION};", generated)
            self.assertIn(f'kSgemmContractSha256[] = "{SGEMM_SHA}";', generated)
            encoded_contract = re.search(r"kSgemmContractJson\[\] = (.+);", generated)
            self.assertIsNotNone(encoded_contract)
            self.assertEqual(
                json.loads(json.loads(encoded_contract[1])), sgemm_contract_record()
            )
            self.assertEqual(
                json.loads(description.read_text()),
                runner_description("intel", enable_sgemm=True).record(),
            )
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((header.read_bytes(), description.read_bytes()), default)


if __name__ == "__main__":
    unittest.main()
