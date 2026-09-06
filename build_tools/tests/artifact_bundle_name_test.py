# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise bundle naming through the real CMake artifact/build pipeline."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(
    shutil.which("cmake") and shutil.which("ninja"), "CMake and Ninja required"
)
class ArtifactBundleNameTest(unittest.TestCase):
    def configure(self, artifact_args="", *, split=False):
        context = tempfile.TemporaryDirectory()
        self.addCleanup(context.cleanup)
        source = Path(context.name) / "source"
        build = Path(context.name) / "build"
        source.mkdir()
        (source / "artifact.toml").write_text(
            '[components.test."stage"]\ninclude = ["share/value.txt"]\n'
        )
        # The stage contains an ordinary text file, so these tests require no
        # compiler, GPU SDK, device, or external subproject source checkout.
        (source / "CMakeLists.txt").write_text(
            textwrap.dedent(
                """\
                cmake_minimum_required(VERSION 3.25)
                project(ArtifactBundleNameTest LANGUAGES NONE)
                set(THEROCK_BINARY_DIR "${CMAKE_BINARY_DIR}")
                list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
                include(therock_default_targets)
                include(therock_subproject)
                include(therock_artifacts)
                set(THEROCK_AMDGPU_DIST_BUNDLE_NAME gfx1201)
                set(THEROCK_ARTIFACT_TYPE_sample target-specific)
                file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/stage/share")
                file(WRITE "${CMAKE_BINARY_DIR}/stage/share/value.txt" "payload\n")
                therock_provide_artifact(sample
                  DESCRIPTOR "${CMAKE_CURRENT_SOURCE_DIR}/artifact.toml"
                  DISTRIBUTION validation
                  COMPONENTS test
                """
            )
            + "  "
            + artifact_args
            + "\n)\n"
        )
        result = subprocess.run(
            [
                "cmake",
                "-G",
                "Ninja",
                "-S",
                str(source),
                "-B",
                str(build),
                f"-DTHEROCK_SOURCE_DIR={REPO_ROOT}",
                f"-DPython3_EXECUTABLE={sys.executable}",
                f"-DTHEROCK_FLAG_KPACK_SPLIT_ARTIFACTS={'ON' if split else 'OFF'}",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        return build, result

    def build_artifact(self, artifact_args, expected_bundle):
        build, result = self.configure(artifact_args)
        self.assertEqual(result.returncode, 0, result.stdout)
        result = subprocess.run(
            ["cmake", "--build", str(build), "--target", "artifact-sample"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        artifact_name = f"sample_test_{expected_bundle}"
        artifacts = build / "artifacts"
        self.assertEqual(
            sorted(path.name for path in artifacts.iterdir()),
            [artifact_name, artifact_name + ".fprint"],
        )
        artifact = artifacts / artifact_name
        self.assertTrue((artifact / "artifact_manifest.txt").is_file())
        self.assertEqual((artifact / "stage/share/value.txt").read_text(), "payload\n")
        self.assertEqual(
            (build / "dist/validation/share/value.txt").read_text(), "payload\n"
        )
        return (artifacts / (artifact_name + ".fprint")).read_text()

    def test_explicit_nvidia_bundle(self):
        self.build_artifact("DIST_BUNDLE_NAME nvidia-cuda-sm120", "nvidia-cuda-sm120")

    def test_legacy_amd_bundle_is_preserved(self):
        self.build_artifact("", "gfx1201")

    def test_target_neutral_keeps_generic_bundle(self):
        self.build_artifact("TARGET_NEUTRAL", "generic")

    def test_explicit_bundle_changes_fingerprint(self):
        first = self.build_artifact(
            "DIST_BUNDLE_NAME nvidia-cuda-sm120", "nvidia-cuda-sm120"
        )
        second = self.build_artifact(
            "DIST_BUNDLE_NAME intel-level-zero-xe2", "intel-level-zero-xe2"
        )
        self.assertNotEqual(first, second)

    def test_invalid_filename_bundle_is_rejected(self):
        for bundle in ("../escape", "nvidia/cuda", "sm_120", "-nvidia"):
            with self.subTest(bundle=bundle):
                build, result = self.configure(f'DIST_BUNDLE_NAME "{bundle}"')
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("DIST_BUNDLE_NAME must contain only", result.stdout)
                self.assertFalse((build / "artifacts").exists())

    def test_empty_bundle_is_rejected(self):
        for artifact_args in ('DIST_BUNDLE_NAME ""', "DIST_BUNDLE_NAME"):
            with self.subTest(artifact_args=artifact_args):
                build, result = self.configure(artifact_args)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(
                    "DIST_BUNDLE_NAME requires a nonempty value", result.stdout
                )
                self.assertFalse((build / "artifacts").exists())

    def test_target_neutral_with_override_is_rejected(self):
        _, result = self.configure("TARGET_NEUTRAL DIST_BUNDLE_NAME nvidia-cuda-sm120")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            "DIST_BUNDLE_NAME cannot be combined with TARGET_NEUTRAL", result.stdout
        )

    def test_amd_kpack_split_with_override_is_rejected(self):
        _, result = self.configure("DIST_BUNDLE_NAME nvidia-cuda-sm120", split=True)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            "DIST_BUNDLE_NAME cannot use the AMD-only KPACK_SPLIT_ARTIFACTS",
            result.stdout,
        )


if __name__ == "__main__":
    unittest.main()
