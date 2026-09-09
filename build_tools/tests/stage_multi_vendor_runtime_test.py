# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU-only packaging and direct child-install boundaries."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1]
REPOSITORY = TOOLS.parent
HELPER = TOOLS / "stage_multi_vendor_runtime.py"
KPACK = REPOSITORY / "rocm-systems/shared/kpack/python"
PYTHON_ROOT = "share/therock/python"
MANIFEST = PYTHON_ROOT + "/runtime-manifest.json"


class RuntimeStagingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tools = self.root / "checkout/build_tools"
        self.tools.mkdir(parents=True)
        self.kpack = self.root / "checkout/kpack/python"
        self.kpack.mkdir(parents=True)
        self.output = self.root / "build"
        self.stage = self.root / "stage"
        self.source = self.tools / "package.py"
        self.source.write_text("VALUE = 1\n")
        self.config = self.tools / "multi_vendor_runtime_sources.json"
        self.config.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "files": [
                        {
                            "root": "tools",
                            "path": "package.py",
                            "destination": PYTHON_ROOT + "/package.py",
                        }
                    ],
                }
            )
        )

    def run_command(self, command, *, success=True):
        result = subprocess.run(
            command,
            cwd=self.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("Traceback", result.stderr)
        return result

    def run_helper(self, command, *, success=True, extra=()):
        return self.run_command(
            [
                sys.executable,
                str(HELPER),
                command,
                "--tools-dir",
                str(self.tools),
                "--kpack-python-dir",
                str(self.kpack),
                "--output-dir",
                str(self.output),
                *extra,
            ],
            success=success,
        )

    def install(self, *, success=True):
        return self.run_helper(
            "install", success=success, extra=("--destination", str(self.stage))
        )

    def tree(self, root):
        return {
            path.relative_to(root).as_posix(): (
                path.read_bytes(),
                path.stat().st_mtime_ns,
                path.stat().st_mode & 0o777,
            )
            for path in root.rglob("*")
            if path.is_file()
        }

    def test_materialization_and_install_preserve_noop_bytes_modes_and_mtimes(self):
        self.run_helper("build")
        self.install()
        built = self.tree(self.output)
        staged = self.tree(self.stage)
        self.run_helper("build")
        self.install()
        self.assertEqual(built, self.tree(self.output))
        self.assertEqual(staged, self.tree(self.stage))
        self.assertTrue(all(item[2] == 0o644 for item in staged.values()))
        manifest = json.loads((self.stage / MANIFEST).read_text())
        self.assertEqual(manifest["kind"], "multi-vendor-python-runtime")
        self.assertEqual(manifest["kpack_scope"], "archive-reader-subset")
        self.assertNotIn(str(self.root), json.dumps(manifest))
        listing = json.loads(self.run_helper("list").stdout)
        self.assertIn(str(HELPER), listing["source_files"])
        self.assertIn(str(self.config), listing["source_files"])
        self.assertIn(MANIFEST, listing["output_files"])

    def test_changed_sources_with_restored_mtime_cannot_install_before_rebuild(self):
        self.run_helper("build")
        self.install()
        before = self.tree(self.stage)
        original = self.source.stat()
        self.source.write_text("VALUE = 2\n")
        os.utime(self.source, ns=(original.st_atime_ns, original.st_mtime_ns))
        self.run_helper("verify", success=False)
        self.install(success=False)
        self.assertEqual(before, self.tree(self.stage))
        self.run_helper("build")
        self.install()
        self.assertEqual(
            (self.stage / PYTHON_ROOT / "package.py").read_text(), "VALUE = 2\n"
        )

    def test_missing_tampered_and_extra_build_outputs_reject_without_stage_mutation(
        self,
    ):
        self.run_helper("build")
        self.install()
        before = self.tree(self.stage)
        output = self.output / PYTHON_ROOT / "package.py"
        for action in ("missing", "tampered", "mode", "extra", "manifest"):
            with self.subTest(action=action):
                self.run_helper("build")
                if action == "missing":
                    output.unlink()
                elif action == "tampered":
                    output.write_text("VALUE = 9\n")
                elif action == "mode":
                    output.chmod(0o755)
                elif action == "extra":
                    (output.parent / "unexpected.py").write_text("extra")
                else:
                    (self.output / MANIFEST).write_text("{}")
                self.install(success=False)
                self.assertEqual(before, self.tree(self.stage))

    def test_symlink_outputs_and_escaping_sources_fail_closed(self):
        self.run_helper("build")
        self.install()
        before = self.tree(self.stage)
        output = self.output / PYTHON_ROOT / "package.py"
        output.unlink()
        output.symlink_to(self.source)
        self.install(success=False)
        self.assertEqual(before, self.tree(self.stage))
        output.unlink()
        external = self.root / "external.py"
        external.write_text("outside")
        self.source.unlink()
        self.source.symlink_to(external)
        self.run_helper("build", success=False)
        self.assertEqual(before, self.tree(self.stage))

    def test_bad_source_maps_rejected_without_materialization(self):
        original = json.loads(self.config.read_text())
        invalid = [
            {"schema_version": True, "files": original["files"]},
            {"schema_version": 1, "files": original["files"] * 2},
            {
                "schema_version": 1,
                "files": [
                    {
                        "root": "tools",
                        "path": "../escape",
                        "destination": PYTHON_ROOT + "/x",
                    }
                ],
            },
            {
                "schema_version": 1,
                "files": [
                    {
                        "root": "tools",
                        "path": "package.py",
                        "destination": "bin/injected",
                    }
                ],
            },
        ]
        for record in invalid:
            with self.subTest(record=record):
                self.config.write_text(json.dumps(record))
                self.run_helper("build", success=False)
                self.assertFalse(self.output.exists())
        self.config.write_text('{"schema_version":1,"schema_version":1,"files":[]}')
        self.run_helper("build", success=False)

    def test_removing_source_map_entry_removes_obsolete_packaged_file(self):
        record = json.loads(self.config.read_text())
        obsolete = self.tools / "obsolete.py"
        obsolete.write_text("obsolete")
        record["files"].append(
            {
                "root": "tools",
                "path": "obsolete.py",
                "destination": PYTHON_ROOT + "/obsolete.py",
            }
        )
        self.config.write_text(json.dumps(record))
        self.run_helper("build")
        self.install()
        record["files"].pop()
        self.config.write_text(json.dumps(record))
        self.install(success=False)
        self.run_helper("build")
        self.install()
        self.assertFalse((self.stage / PYTHON_ROOT / "obsolete.py").exists())

    def test_real_runtime_imports_without_checkout_or_pythonpath(self):
        self.run_command(
            [
                sys.executable,
                str(HELPER),
                "build",
                "--tools-dir",
                str(TOOLS),
                "--kpack-python-dir",
                str(KPACK),
                "--output-dir",
                str(self.output),
            ]
        )
        self.run_command(
            [
                sys.executable,
                str(HELPER),
                "install",
                "--tools-dir",
                str(TOOLS),
                "--kpack-python-dir",
                str(KPACK),
                "--output-dir",
                str(self.output),
                "--destination",
                str(self.stage),
            ]
        )
        python_root = self.stage / PYTHON_ROOT
        code = """
import importlib, pathlib, sys
sys.dont_write_bytecode = True
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import therock_multi_vendor
assert callable(therock_multi_vendor.open_session)
for name in ("_therock_utils.module_selection", "_therock_utils.runner_query",
             "_therock_utils.module_service", "rocm_kpack.kpack", "rocm_kpack.compression"):
    importlib.import_module(name)
for name, module in list(sys.modules.items()):
    if name == "therock_multi_vendor" or name.startswith(("_therock_utils.", "rocm_kpack")):
        assert pathlib.Path(module.__file__).resolve().is_relative_to(root), name
assert "_therock_utils" in sys.modules
assert all(pathlib.Path(p).resolve().is_relative_to(root)
           for p in sys.modules["_therock_utils"].__path__)
assert "validate_multi_vendor_modules" not in sys.modules
assert "rocm_kpack.kpack_transform" not in sys.modules
"""
        self.run_command([sys.executable, "-I", "-c", code, str(python_root)])
        self.assertTrue((python_root / "licenses/TheRock-LICENSE").is_file())
        self.assertTrue((python_root / "licenses/rocm-kpack-LICENSE").is_file())
        requirements = (python_root / "requirements.txt").read_text()
        self.assertIn("msgpack", requirements)
        self.assertIn("zstandard", requirements)
        self.assertTrue(
            (self.stage / "share/therock/examples/packed_session_client.py").is_file()
        )
        self.assertFalse(list(self.stage.rglob("__pycache__")))

    def test_direct_pack_child_install_checks_runtime_before_removing_prior_receipts(
        self,
    ):
        shutil.copy2(HELPER, self.tools / HELPER.name)
        (self.tools / "assemble_multi_vendor_modules.py").write_text(
            "from types import SimpleNamespace\n"
            "def read_targets(path): return [SimpleNamespace(slug='nvidia-cuda-sm120')]\n"
        )
        targets = self.root / "targets.json"
        targets.write_text("{}")
        guard = self.root / "guard.cmake"
        guard.write_text("# No imported SDKs in this CPU-only install fixture.\n")
        provenance = self.root / "provenance.json"
        provenance.write_text('{"fixture":"current"}\n')
        build = self.root / "child-build"
        self.run_command(
            [
                "cmake",
                "-S",
                str(REPOSITORY / "experimental/multi-vendor/pack"),
                "-B",
                str(build),
                "-G",
                "Ninja",
                "-DPython3_EXECUTABLE=" + sys.executable,
                "-DTHEROCK_MODULE_TOOLS_DIR=" + str(self.tools),
                "-DTHEROCK_MODULE_KPACK_PYTHON_DIR=" + str(self.kpack),
                "-DTHEROCK_MODULE_TARGETS_FILE=" + str(targets),
                "-DTHEROCK_MODULE_BUILD_ROOT=" + str(self.root / "native"),
                "-DTHEROCK_MULTI_VENDOR_INPUT_GUARD=" + str(guard),
                "-DTHEROCK_MULTI_VENDOR_INPUT_PROVENANCE=" + str(provenance),
            ]
        )
        self.run_command(["cmake", "--build", str(build), "--target", "module-runtime"])
        packs = build / "packs"
        packs.mkdir()
        (packs / "input-provenance.json").write_bytes(provenance.read_bytes())
        (packs / "runners.json").write_text("fixture registry")
        for module in ("saxpy", "relu"):
            (packs / module).mkdir()
            (packs / module / "catalog.json").write_text("fixture catalog")
            (packs / module / ("validation-" + module + ".kpack")).write_bytes(
                b"fixture pack"
            )
        command = ["cmake", "--install", str(build), "--prefix", str(self.stage)]
        self.run_command(command)
        before = self.tree(self.stage)
        self.assertIn("share/therock/packs/input-provenance.json", before)
        self.source.write_text("VALUE = 2\n")
        self.run_command(command, success=False)
        self.assertEqual(before, self.tree(self.stage))
        # A direct runtime target rebuild repairs current source materialization.
        self.run_command(["cmake", "--build", str(build), "--target", "module-runtime"])
        self.run_command(command)
        self.assertEqual(
            (self.stage / PYTHON_ROOT / "package.py").read_text(), "VALUE = 2\n"
        )
        before = self.tree(self.stage)
        (build / "runtime" / PYTHON_ROOT / "package.py").unlink()
        self.run_command(command, success=False)
        self.assertEqual(before, self.tree(self.stage))


if __name__ == "__main__":
    unittest.main()
