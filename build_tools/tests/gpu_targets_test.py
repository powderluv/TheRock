# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.gpu_targets import (
    GpuTarget,
    TargetFeature,
    parse_gpu_target,
    parse_gpu_targets,
)


class GpuTargetsTest(unittest.TestCase):
    def test_reference_target_toolchain_mappings(self):
        cases = (
            ("amd:hip:gfx1201", "amd", "gfx1201", "gfx1201", ("hsaco",), False),
            (
                "nvidia:cuda:sm_120",
                "nvidia",
                "120",
                "sm_120",
                ("cubin", "ptx"),
                True,
            ),
            ("intel:level-zero:xe2-b70", "spirv", None, None, ("spirv",), True),
        )
        for identity, platform, architecture, compiler, payloads, experimental in cases:
            with self.subTest(identity=identity):
                target = parse_gpu_target(identity)
                self.assertEqual(target.canonical_id, identity)
                self.assertEqual(target.cmake_hip_platform, platform)
                self.assertEqual(target.cmake_hip_architecture, architecture)
                self.assertEqual(target.compiler_target, compiler)
                self.assertEqual(target.payload_types, payloads)
                self.assertEqual(target.experimental, experimental)

    def test_amd_feature_signs_survive_mapping_and_canonicalization(self):
        target = parse_gpu_target("amd:hip:gfx90a:xnack+:sramecc-")
        self.assertEqual(
            target.features,
            (TargetFeature("sramecc", False), TargetFeature("xnack", True)),
        )
        self.assertEqual(target.canonical_id, "amd:hip:gfx90a:sramecc-:xnack+")
        self.assertEqual(target.cmake_hip_architecture, "gfx90a:sramecc-:xnack+")
        self.assertEqual(target.compiler_target, "gfx90a:sramecc-:xnack+")
        self.assertEqual(target.slug, "amd-hip-gfx90a-sramecc-off-xnack-on")

    def test_feature_order_does_not_change_identity(self):
        first = parse_gpu_target("amd:hip:gfx942:xnack-:sramecc+")
        second = parse_gpu_target("amd:hip:gfx942:sramecc+:xnack-")
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertEqual(first.slug, second.slug)
        self.assertEqual(parse_gpu_target(first.canonical_id), first)

    def test_processor_grammar_is_not_a_hardware_support_registry(self):
        # Syntactically valid future processors can be planned. The toolchain
        # and hardware validation, not parsing, determine whether they work.
        self.assertEqual(parse_gpu_target("amd:hip:gfx9999").processor, "gfx9999")
        self.assertEqual(
            parse_gpu_target("nvidia:cuda:sm_100a").cmake_hip_architecture, "100a"
        )

    def test_slug_is_safe_and_preserves_distinct_feature_states(self):
        targets = parse_gpu_targets(
            (
                "amd:hip:gfx942",
                "amd:hip:gfx942:xnack+",
                "amd:hip:gfx942:xnack-",
                "amd:hip:gfx942:sramecc+",
                "amd:hip:gfx942:sramecc-",
                "amd:hip:gfx942:xnack+:sramecc-",
                "amd:hip:gfx942:xnack-:sramecc+",
                "nvidia:cuda:sm_120",
                "intel:level-zero:xe2-b70",
            )
        )
        self.assertEqual(len({target.slug for target in targets}), len(targets))
        for target in targets:
            with self.subTest(target=target.canonical_id):
                self.assertRegex(target.slug, r"\A[a-z0-9-]+\Z")
                self.assertNotEqual(target.slug, target.canonical_id)

    def test_slugs_fit_existing_artifact_target_field(self):
        from _therock_utils.artifacts import ArtifactName

        for value, slug in (
            ("amd:hip:gfx942:xnack+", "amd-hip-gfx942-xnack-on"),
            ("nvidia:cuda:sm_120", "nvidia-cuda-sm120"),
            ("intel:level-zero:xe2-b70", "intel-level-zero-xe2-b70"),
        ):
            with self.subTest(value=value):
                target = parse_gpu_target(value)
                self.assertEqual(target.slug, slug)
                artifact = ArtifactName.from_path(Path(f"smoke_run_{slug}.tar.xz"))
                self.assertIsNotNone(artifact)
                self.assertEqual(artifact.target_family, slug)

    def test_rejects_invalid_or_unsafe_targets(self):
        invalid = (
            "",
            "gfx1201",
            "amd:hip",
            "amd:hip:",
            "amd:hip:gfx1201:",
            "AMD:hip:gfx1201",
            " amd:hip:gfx1201",
            "amd:hip:gfx1201\n",
            "nvidia:cuda:sm_120 ",
            "amd:hip:../gfx1201",
            "amd:hip:gfx1201/../../outside",
            "amd:hip:gfx1201\\outside",
            "amd:hip:gfx1201;OTHER=ON",
            "amd:hip:gfx1201$(id)",
            "amd:hip:gfx1201`id`",
            "amd:hip:gfx1201\0",
            "amd:hip:gfx1201__xnack-plus",
            "amd:hip:gfxZZZ",
            "nvidia:cuda:sm_0120",
            "nvidia:cuda:sm__120",
            "intel:level-zero:xe2-b70--extra",
            "intel:level-zero:xe2-../b70",
            "unknown:hip:gfx1201",
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_gpu_target(value)

    def test_rejects_mismatched_vendor_backend_and_processor(self):
        for value in (
            "amd:cuda:gfx1201",
            "nvidia:hip:sm_120",
            "intel:cuda:xe2-b70",
            "intel:hip:xe2-b70",
            "amd:hip:sm_120",
            "nvidia:cuda:gfx1201",
            "intel:level-zero:sm_120",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_gpu_target(value)

    def test_rejects_unknown_duplicate_and_conflicting_features(self):
        for value in (
            "amd:hip:gfx942:xnack",
            "amd:hip:gfx942:xnack=on",
            "amd:hip:gfx942:xnack++",
            "amd:hip:gfx942:XNACK+",
            "amd:hip:gfx942:wave64+",
            "amd:hip:gfx942:xnack+:xnack+",
            "amd:hip:gfx942:xnack+:xnack-",
            "amd:hip:gfx942:sramecc-:sramecc+",
            "nvidia:cuda:sm_120:xnack+",
            "intel:level-zero:xe2-b70:sramecc-",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_gpu_target(value)

    def test_direct_construction_keeps_invariants(self):
        with self.assertRaises(ValueError):
            GpuTarget("amd", "cuda", "gfx1201")
        with self.assertRaises(ValueError):
            GpuTarget("amd", "hip", "gfx1201", ("xnack+",))
        with self.assertRaises(ValueError):
            GpuTarget("amd", "hip", "gfx1201", [TargetFeature("xnack", True)])
        with self.assertRaises(ValueError):
            TargetFeature("unsafe", True)
        with self.assertRaises(ValueError):
            TargetFeature("xnack", "false")
        target = GpuTarget("amd", "hip", "gfx1201")
        with self.assertRaises(FrozenInstanceError):
            target.processor = "gfx942"

    def test_target_lists_keep_order_and_reject_duplicate_identities(self):
        values = ("nvidia:cuda:sm_120", "amd:hip:gfx1201", "intel:level-zero:xe2-b70")
        self.assertEqual(
            tuple(target.canonical_id for target in parse_gpu_targets(iter(values))),
            values,
        )
        with self.assertRaisesRegex(ValueError, "Duplicate GPU target"):
            parse_gpu_targets(("amd:hip:gfx1201", "amd:hip:gfx1201"))
        with self.assertRaisesRegex(ValueError, "Duplicate GPU target"):
            parse_gpu_targets(
                (
                    "amd:hip:gfx942:xnack+:sramecc-",
                    "amd:hip:gfx942:sramecc-:xnack+",
                )
            )


if __name__ == "__main__":
    unittest.main()
