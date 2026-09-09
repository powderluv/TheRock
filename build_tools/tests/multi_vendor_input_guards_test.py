# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU-only producer-ordering tests for locked imported-input build guards."""

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUTS_SCRIPT = REPO_ROOT / "build_tools/multi_vendor_inputs.py"


@unittest.skipUnless(
    shutil.which("cmake") and shutil.which("ninja"), "CMake and Ninja required"
)
class MultiVendorInputGuardsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="therock-input-guards-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.build = self.root / "build"
        self.sdk = self.root / "sdk"
        self.sdk.mkdir()
        self.header = self.sdk / "header.h"
        self.header.write_text("original SDK bytes\n")
        self.lock = self.root / "inputs.lock.json"
        self.run_command(
            sys.executable,
            INPUTS_SCRIPT,
            "capture",
            "--lock",
            self.lock,
            "--input",
            "sdk",
            "tree",
            self.sdk,
        )
        self.write(
            "artifact.toml",
            '[components.test."child/stage"]\ninclude = ["share/value.txt"]\n',
        )
        self.write("child/value.txt", "fixture payload\n")
        self.write(
            "child/CMakeLists.txt",
            """
            cmake_minimum_required(VERSION 3.25)
            project(guarded_child LANGUAGES NONE)
            file(APPEND "${CMAKE_BINARY_DIR}/configure-event.txt" "configured\n")
            add_custom_target(fixture ALL
              COMMAND "${CMAKE_COMMAND}" -E touch "${CMAKE_BINARY_DIR}/build-event.txt"
              COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/value.txt" "${CMAKE_BINARY_DIR}/value.txt"
              VERBATIM)
            install(CODE [[file(TOUCH "stage-event.txt")]])
            install(FILES "${CMAKE_BINARY_DIR}/value.txt" DESTINATION share)
        """,
        )
        self.write(
            "CMakeLists.txt",
            f"""
            cmake_minimum_required(VERSION 3.25)
            project(input_guard_fixture LANGUAGES NONE)
            set(THEROCK_SOURCE_DIR [[{REPO_ROOT}]])
            set(THEROCK_BINARY_DIR "${{CMAKE_BINARY_DIR}}")
            list(APPEND CMAKE_MODULE_PATH "${{THEROCK_SOURCE_DIR}}/cmake")
            set(Python3_EXECUTABLE [[{sys.executable}]])
            set(ROCM_BUILD_FLAGS_STATE_FILE "${{CMAKE_BINARY_DIR}}/rocm_build_flags_state.cmake")
            file(WRITE "${{ROCM_BUILD_FLAGS_STATE_FILE}}" "# no build flags\n")
            include(therock_globals)
            include(therock_sanitizers)
            include(therock_flag_utils)
            include(therock_default_targets)
            include(therock_subproject)
            include(therock_artifacts)
            add_custom_target(input_guard
              COMMAND "${{CMAKE_COMMAND}}" -E echo VERIFY_IMPORTED_INPUTS
              COMMAND "${{Python3_EXECUTABLE}}" [[{INPUTS_SCRIPT}]] verify
                --lock [[{self.lock}]] --input sdk tree [[{self.sdk}]] --full
              VERBATIM)
            therock_cmake_subproject_declare(child
              EXTERNAL_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/child"
              BINARY_DIR "${{CMAKE_CURRENT_BINARY_DIR}}/child"
              BUILD_GUARDS input_guard
            )
            target_sources(child PRIVATE "${{CMAKE_CURRENT_SOURCE_DIR}}/child/value.txt")
            therock_cmake_subproject_activate(child)
            # Prebuilt configure/build stamps otherwise have no consumers, so
            # retain their producers in this focused graph for direct testing.
            add_custom_target(all_child_stamp_rules DEPENDS
              "${{CMAKE_CURRENT_BINARY_DIR}}/child/stamp/configure.stamp"
              "${{CMAKE_CURRENT_BINARY_DIR}}/child/stamp/build.stamp")
            therock_provide_artifact(sample
              DESCRIPTOR "${{CMAKE_CURRENT_SOURCE_DIR}}/artifact.toml"
              DISTRIBUTION validation
              COMPONENTS test
              DIST_BUNDLE_NAME synthetic
              SUBPROJECT_DEPS child
              BUILD_GUARDS input_guard
            )
        """,
        )

    def write(self, relative, contents):
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(contents))

    def run_command(self, *args, success=True):
        result = subprocess.run(
            list(map(str, args)), capture_output=True, text=True, timeout=30
        )
        evidence = f"{args!r}\n{result.stdout}\n{result.stderr}"
        if success:
            self.assertEqual(result.returncode, 0, evidence)
        else:
            self.assertNotEqual(result.returncode, 0, evidence)
        return result.stdout + result.stderr

    def configure(self, *, prebuilt=False):
        if prebuilt:
            stage = self.build / "child/stage/share"
            stage.mkdir(parents=True)
            (stage / "value.txt").write_text("prebuilt payload\n")
            (self.build / "child/stage.prebuilt").touch()
        self.run_command("cmake", "-S", self.source, "-B", self.build, "-GNinja")

    def build_target(self, target, *, success=True):
        output = self.run_command(
            "cmake",
            "--build",
            self.build,
            "--parallel",
            "8",
            "--target",
            target,
            success=success,
        )
        self.assertIn("VERIFY_IMPORTED_INPUTS", output)
        return output

    def snapshot(self):
        # Ignore Ninja bookkeeping and guard diagnostics. Every actual child
        # lifecycle event, stage/dist file, and artifact payload is recorded.
        paths = []
        for relative in (
            "child/stamp",
            "child/stage",
            "child/dist",
            "artifacts",
            "dist",
        ):
            root = self.build / relative
            if root.exists():
                paths.extend(path for path in root.rglob("*") if path.is_file())
        paths.extend((self.build / "child/build").glob("*-event.txt"))
        return {
            str(p.relative_to(self.build)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in paths
        }

    def test_mutation_stops_direct_normal_producers_before_first_child_configure(self):
        self.configure()
        before = self.snapshot()
        self.header.write_text("modified SDK bytes\n")
        targets = (
            "child+configure",
            "child+build",
            "child+stage",
            "child/stamp/build.stamp",
            "artifact-sample",
            "artifacts/sample_test_synthetic/artifact_manifest.txt",
        )
        for target in targets:
            with self.subTest(target=target):
                self.build_target(target, success=False)
                self.assertEqual(self.snapshot(), before)
                self.assertFalse((self.build / "child/build/CMakeCache.txt").exists())

    def test_unchanged_guard_runs_without_invalidating_child_or_artifact_outputs(self):
        self.configure()
        self.build_target("artifact-sample")
        before = self.snapshot()
        self.assertTrue((self.build / "dist/validation/share/value.txt").is_file())
        for target in ("child+build", "child+stage", "artifact-sample", "all"):
            with self.subTest(target=target):
                self.build_target(target)
                self.assertEqual(self.snapshot(), before)

    def test_mutation_stops_dirty_producers_after_a_successful_build(self):
        self.configure()
        self.build_target("artifact-sample")
        self.header.write_text("modified SDK bytes\n")
        producers = (
            ("child+configure", "child/stamp/configure.stamp"),
            ("child+build", "child/stamp/build.stamp"),
            ("child+stage", "child/stamp/stage.stamp"),
            (
                "artifact-sample",
                "artifacts/sample_test_synthetic/artifact_manifest.txt",
            ),
        )
        for target, output in producers:
            with self.subTest(target=target):
                stamp = self.build / output
                original = stamp.read_bytes()
                timestamp = stamp.stat()
                stamp.unlink()
                before = self.snapshot()
                self.build_target(target, success=False)
                self.assertEqual(self.snapshot(), before)
                stamp.write_bytes(original)
                os.utime(stamp, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))

    def test_artifact_guard_protects_population_without_subproject_dependencies(self):
        parent = self.source / "CMakeLists.txt"
        parent.write_text(parent.read_text().replace("SUBPROJECT_DEPS child", ""))
        self.configure(prebuilt=True)
        before = self.snapshot()
        self.header.write_text("modified SDK bytes\n")
        self.build_target("artifact-sample", success=False)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.build / "dist/validation/share/value.txt").exists())
        self.header.write_text("original SDK bytes\n")
        self.build_target("artifact-sample")
        self.assertFalse((self.build / "child/stamp/stage.stamp").exists())
        self.assertEqual(
            (self.build / "dist/validation/share/value.txt").read_text(),
            "prebuilt payload\n",
        )
        self.header.write_text("modified SDK bytes\n")
        (self.build / "artifacts/sample_test_synthetic/artifact_manifest.txt").unlink()
        before = self.snapshot()
        self.build_target(
            "artifacts/sample_test_synthetic/artifact_manifest.txt", success=False
        )
        self.assertEqual(self.snapshot(), before)

    def test_mutation_stops_all_prebuilt_stamp_and_population_producers(self):
        self.configure(prebuilt=True)
        before = self.snapshot()
        self.header.write_text("modified SDK bytes\n")
        targets = (
            "child/stamp/configure.stamp",
            "child/stamp/build.stamp",
            "child+stage",
            "artifact-sample",
            "artifacts/sample_test_synthetic/artifact_manifest.txt",
        )
        for target in targets:
            with self.subTest(target=target):
                self.build_target(target, success=False)
                self.assertEqual(self.snapshot(), before)
        self.header.write_text("original SDK bytes\n")
        self.build_target("artifact-sample")
        self.assertEqual(
            (self.build / "dist/validation/share/value.txt").read_text(),
            "prebuilt payload\n",
        )
        before = self.snapshot()
        self.build_target("artifact-sample")
        self.assertEqual(self.snapshot(), before)


@unittest.skipUnless(
    shutil.which("cmake") and shutil.which("ninja") and shutil.which("cc"),
    "CMake, Ninja, and a CPU C compiler required",
)
class MultiVendorInputReceiptsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="therock-input-receipts-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.build = self.root / "build"
        self.stage = self.root / "stage"
        self.sdk = self.root / "sdk"
        self.sdk.mkdir()
        self.header = self.sdk / "fixture.h"
        self.header.write_text("#define FIXTURE_VALUE 1\n")
        self.report = self.root / "provenance.json"
        self.guard = self.root / "verify-current.cmake"
        self.helper = REPO_ROOT / "tests/multi_vendor/input_guard.cmake"
        self.executed = self.root / "execution.txt"
        self.receipt = self.build / "input-build-receipt.json"
        self.staged_receipt = self.stage / "bin/fixture/input-build-receipt.json"
        self.select_new_lock("a")
        self.program = textwrap.dedent(
            """\
            #include <stdio.h>
            #include "fixture.h"
            int main(int argc, char **argv) {
              if (argc != 2) return 1;
              FILE *output = fopen(argv[1], "a");
              if (!output) return 2;
              if (fprintf(output, "%d\\n", FIXTURE_VALUE) < 0) return 3;
              return fclose(output) == 0 ? 0 : 4;
            }
        """
        )
        (self.source / "fixture.c").write_text(self.program)
        (self.source / "CMakeLists.txt").write_text(
            textwrap.dedent(
                f"""\
            cmake_minimum_required(VERSION 3.25)
            set(THEROCK_MULTI_VENDOR_INPUT_GUARD [[{self.guard}]])
            set(THEROCK_MULTI_VENDOR_INPUT_PROVENANCE [[{self.report}]])
            include([[{self.helper}]])
            project(cpu_receipt_fixture LANGUAGES C)
            include(CTest)
            therock_multi_vendor_initialize_inputs()
            add_executable(fixture fixture.c)
            target_include_directories(fixture PRIVATE [[{self.sdk}]])
            therock_multi_vendor_guard_consumer(fixture)
            add_test(NAME cpu-execution COMMAND fixture [[{self.executed}]])
            therock_multi_vendor_guard_test(cpu-execution)
            therock_multi_vendor_prepare_install(
              RECEIPT "${{CMAKE_CURRENT_BINARY_DIR}}/input-build-receipt.json"
              FILES "bin/fixture/fixture" "bin/fixture/input-build-receipt.json")
            install(TARGETS fixture RUNTIME DESTINATION bin/fixture)
            therock_multi_vendor_record_inputs(input-receipt
              DEPENDS fixture INSTALL_DESTINATION bin/fixture)
        """
            )
        )

    def run_command(self, *args, success=True):
        result = subprocess.run(
            list(map(str, args)), capture_output=True, text=True, timeout=30
        )
        evidence = f"{args!r}\n{result.stdout}\n{result.stderr}"
        if success:
            self.assertEqual(result.returncode, 0, evidence)
        else:
            self.assertNotEqual(result.returncode, 0, evidence)
        return result.stdout + result.stderr

    def select_new_lock(self, label):
        lock = self.root / f"inputs-{label}.lock.json"
        self.run_command(
            sys.executable,
            INPUTS_SCRIPT,
            "capture",
            "--lock",
            lock,
            "--input",
            "sdk",
            "tree",
            self.sdk,
            "--report",
            self.report,
        )
        self.guard.write_text(
            textwrap.dedent(
                f"""\
            execute_process(COMMAND [[{sys.executable}]] [[{INPUTS_SCRIPT}]] verify
              --lock [[{lock}]] --input sdk tree [[{self.sdk}]] --full
              --report [[{self.report}]] COMMAND_ERROR_IS_FATAL ANY)
        """
            )
        )
        return json.loads(self.report.read_text())["global_content_sha256"]

    def configure(self, *, success=True):
        return self.run_command(
            "cmake", "-S", self.source, "-B", self.build, "-GNinja", success=success
        )

    def build_receipt(self, *, success=True):
        return self.run_command(
            "cmake", "--build", self.build, "--target", "input-receipt", success=success
        )

    def run_tests(self, *, success=True):
        return self.run_command(
            "ctest",
            "--test-dir",
            self.build,
            "-R",
            "cpu-execution",
            "--output-on-failure",
            success=success,
        )

    def check_receipts(self, *paths, success=True):
        return self.run_command(
            "cmake",
            f"-DTHEROCK_MULTI_VENDOR_CHECK_PROVENANCE={self.report}",
            "-DTHEROCK_MULTI_VENDOR_CHECK_RECEIPTS=" + ";".join(map(str, paths)),
            "-P",
            self.helper,
            success=success,
        )

    def change_sdk_bytes_with_restored_mtime(self):
        before = self.header.stat()
        self.header.write_text("#define FIXTURE_VALUE 2\n")
        os.utime(self.header, ns=(before.st_atime_ns, before.st_mtime_ns))

    def test_child_configuration_rejects_changed_inputs_before_compiler_detection(self):
        self.change_sdk_bytes_with_restored_mtime()
        self.configure(success=False)
        self.assertFalse(list(self.build.glob("CMakeFiles/*/CMakeCCompiler.cmake")))
        self.assertFalse(self.receipt.exists())
        self.assertFalse(self.executed.exists())

    def test_missing_receipt_blocks_direct_ctest_before_execution(self):
        self.configure()
        self.run_tests(success=False)
        self.assertFalse(self.executed.exists())
        self.assertFalse(self.receipt.exists())
        self.check_receipts(self.receipt, success=False)
        self.run_command(
            "cmake", "--install", self.build, "--prefix", self.stage, success=False
        )
        self.assertFalse(self.stage.exists())

    def test_direct_pack_build_rejects_stale_receipt_in_a_later_target(self):
        selection = self.root / "selection"
        self.run_command(
            sys.executable,
            REPO_ROOT / "build_tools/configure_multi_vendor.py",
            "--output-dir",
            selection,
            "--targets",
            "amd:hip:gfx1201",
            "nvidia:cuda:sm_120",
        )
        module_root = self.root / "native-stages"
        targets = (
            ("amd-hip-gfx1201", ("hsaco",)),
            ("nvidia-cuda-sm120", ("cubin", "ptx")),
        )
        payloads = []
        receipts = []
        for slug, formats in targets:
            stage = module_root / slug / "stage"
            receipt = stage / "bin" / slug / "input-build-receipt.json"
            receipt.parent.mkdir(parents=True)
            shutil.copyfile(self.report, receipt)
            receipts.append(receipt)
            (receipt.parent / "therock_module_validation").write_bytes(
                b"synthetic staged native executable"
            )
            self.run_command(
                sys.executable,
                REPO_ROOT / "build_tools/configure_module_contract.py",
                "--vendor",
                slug.split("-", 1)[0],
                "--header",
                self.root / f"{slug}-contract.h",
                "--description",
                receipt.parent / "runner-contract.json",
            )
            for module in ("saxpy", "relu"):
                for payload_type in formats:
                    path = (
                        stage
                        / "share/therock/modules"
                        / slug
                        / f"{module}.{payload_type}"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(
                        f"synthetic input A:{module}:{payload_type}".encode()
                    )
                    payloads.append(path)
        pack_build = self.root / "pack-build"
        self.run_command(
            "cmake",
            "-S",
            REPO_ROOT / "experimental/multi-vendor/pack",
            "-B",
            pack_build,
            "-GNinja",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DTHEROCK_MODULE_TOOLS_DIR={REPO_ROOT / 'build_tools'}",
            f"-DTHEROCK_MODULE_KPACK_PYTHON_DIR={REPO_ROOT / 'rocm-systems/shared/kpack/python'}",
            f"-DTHEROCK_MODULE_TARGETS_FILE={selection / 'gpu_targets.json'}",
            f"-DTHEROCK_MODULE_BUILD_ROOT={module_root}",
            f"-DTHEROCK_MULTI_VENDOR_INPUT_GUARD={self.guard}",
            f"-DTHEROCK_MULTI_VENDOR_INPUT_PROVENANCE={self.report}",
        )

        def build_packs(success=True):
            return self.run_command(
                "cmake",
                "--build",
                pack_build,
                "--target",
                "module-packs",
                success=success,
            )

        def snapshot():
            return {
                str(p.relative_to(pack_build)): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in (pack_build / "packs").rglob("*")
                if p.is_file()
            }

        build_packs()
        original = snapshot()
        self.assertEqual(len(original), 6)
        pack_stage = self.root / "pack-stage"
        self.run_command("cmake", "--install", pack_build, "--prefix", pack_stage)
        self.change_sdk_bytes_with_restored_mtime()
        new_id = self.select_new_lock("b")
        build_packs(success=False)
        self.assertEqual(snapshot(), original)
        # The checker must examine every target, not just the first stage.
        shutil.copyfile(self.report, receipts[0])
        build_packs(success=False)
        self.assertEqual(snapshot(), original)
        shutil.copyfile(self.report, receipts[1])
        for path in payloads:
            path.write_bytes(path.read_bytes().replace(b"input A", b"input B"))
        build_packs()
        self.assertNotEqual(snapshot(), original)
        self.assertEqual(
            json.loads((pack_build / "packs/input-provenance.json").read_text())[
                "global_content_sha256"
            ],
            new_id,
        )
        staged_files = []
        for source in (pack_build / "packs").rglob("*"):
            if source.is_file():
                destination = (
                    pack_stage
                    / "share/therock/packs"
                    / source.relative_to(pack_build / "packs")
                )
                before = destination.stat()
                os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
                staged_files.append((source, destination))
        self.run_command("cmake", "--install", pack_build, "--prefix", pack_stage)
        for source, destination in staged_files:
            self.assertEqual(destination.read_bytes(), source.read_bytes())

    def test_report_change_rebuilds_output_despite_restored_header_mtime(self):
        self.configure()
        self.build_receipt()
        self.run_tests()
        original = (self.receipt.read_bytes(), self.receipt.stat().st_mtime_ns)
        self.build_receipt()
        self.assertEqual(
            (self.receipt.read_bytes(), self.receipt.stat().st_mtime_ns), original
        )
        source_before = (self.source / "fixture.c").stat().st_mtime_ns
        self.change_sdk_bytes_with_restored_mtime()
        new_id = self.select_new_lock("b")
        self.build_receipt()
        self.assertEqual((self.source / "fixture.c").stat().st_mtime_ns, source_before)
        self.assertEqual(
            json.loads(self.receipt.read_text())["global_content_sha256"], new_id
        )
        self.run_tests()
        self.assertEqual(self.executed.read_text(), "1\n2\n")

    def test_explicit_new_lock_requires_successful_rebuild_and_reinstallation(self):
        self.configure()
        self.build_receipt()
        self.run_tests()
        self.assertEqual(self.executed.read_text(), "1\n")
        self.run_command("cmake", "--install", self.build, "--prefix", self.stage)
        original_receipt = self.receipt.read_bytes()
        original_id = json.loads(original_receipt)["global_content_sha256"]
        self.check_receipts(self.receipt, self.staged_receipt)

        def stage_snapshot():
            return {
                str(p.relative_to(self.stage)): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in self.stage.rglob("*")
                if p.is_file()
            }

        installed_before = stage_snapshot()
        self.change_sdk_bytes_with_restored_mtime()
        self.run_command(
            "cmake", "--install", self.build, "--prefix", self.stage, success=False
        )
        self.assertEqual(stage_snapshot(), installed_before)
        new_id = self.select_new_lock("b")
        self.assertNotEqual(original_id, new_id)
        self.run_command(
            "cmake", "--install", self.build, "--prefix", self.stage, success=False
        )
        self.assertEqual(stage_snapshot(), installed_before)
        self.run_tests(success=False)
        self.check_receipts(self.staged_receipt, success=False)
        self.assertEqual(self.executed.read_text(), "1\n")
        self.assertEqual(self.receipt.read_bytes(), original_receipt)

        # A receipt must never bless the old executable after a failed rebuild.
        (self.source / "fixture.c").write_text(
            "#error intentional CPU fixture build failure\n"
        )
        self.build_receipt(success=False)
        self.assertEqual(self.receipt.read_bytes(), original_receipt)
        self.run_tests(success=False)
        self.assertEqual(self.executed.read_text(), "1\n")

        (self.source / "fixture.c").write_text(self.program)
        self.build_receipt()
        self.assertEqual(
            json.loads(self.receipt.read_text())["global_content_sha256"], new_id
        )
        self.run_tests()
        self.assertEqual(self.executed.read_text(), "1\n2\n")
        self.check_receipts(self.receipt)
        self.check_receipts(self.staged_receipt, success=False)
        # Model same-size updates on a filesystem with coarse timestamps.
        prior = self.staged_receipt.stat()
        self.assertEqual(self.receipt.stat().st_size, prior.st_size)
        os.utime(self.receipt, ns=(prior.st_atime_ns, prior.st_mtime_ns))
        installed_binary = self.stage / "bin/fixture/fixture"
        built_binary = self.build / "fixture"
        binary_prior = installed_binary.stat()
        self.assertEqual(built_binary.stat().st_size, binary_prior.st_size)
        os.utime(built_binary, ns=(binary_prior.st_atime_ns, binary_prior.st_mtime_ns))
        self.run_command("cmake", "--install", self.build, "--prefix", self.stage)
        self.check_receipts(self.receipt, self.staged_receipt)
        self.assertEqual(self.staged_receipt.read_bytes(), self.receipt.read_bytes())
        self.run_command(self.stage / "bin/fixture/fixture", self.executed)
        self.assertEqual(self.executed.read_text(), "1\n2\n2\n")


if __name__ == "__main__":
    unittest.main()
