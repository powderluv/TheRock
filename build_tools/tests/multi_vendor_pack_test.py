# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).parent.parent / "multi_vendor_pack.py"
_TARGET = "nvidia:cuda:sm_120"


class MultiVendorPackCliTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def _run(self, *args):
        env = os.environ.copy()
        kpack_python = (
            Path(__file__).parent.parent.parent / "rocm-systems/shared/kpack/python"
        )
        python_paths = [os.fspath(kpack_python)]
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        return subprocess.run(
            [sys.executable, str(_SCRIPT), *map(str, args)],
            capture_output=True,
            text=True,
            env=env,
        )

    def _create(self, name, module="saxpy", payload=b"device bytes"):
        source = self.root / (name + ".payload")
        source.write_bytes(payload)
        output = self.root / name
        result = self._run(
            "create",
            "--output-dir",
            output,
            "--pack-id",
            name,
            "--entry",
            module,
            _TARGET,
            "cubin",
            "launch",
            source,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return output / "catalog.json"

    def test_create_multiple_variants_and_extract_second_pack(self):
        first = self._create("first", module="unrelated")
        second = self._create("second", payload=b"selected payload")
        output = self.root / "extracted.bin"
        result = self._run(
            "extract",
            "--catalog",
            first,
            "--catalog",
            second,
            "--module",
            "saxpy",
            "--target",
            _TARGET,
            "--format",
            "cubin",
            "--entry-point",
            "launch",
            "--output",
            output,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output.read_bytes(), b"selected payload")
        source = self.root / "variants.payload"
        source.write_bytes(b"synthetic variant")
        result = self._run(
            "create",
            "--output-dir",
            self.root / "variants",
            "--pack-id",
            "variants",
            "--entry",
            "saxpy",
            _TARGET,
            "cubin",
            "launch",
            source,
            "--entry",
            "saxpy",
            "amd:hip:gfx1201",
            "hsaco",
            "launch",
            source,
            "--entry",
            "saxpy",
            "intel:level-zero:xe2-b70",
            "spirv",
            "launch",
            source,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        catalog = json.loads((self.root / "variants/catalog.json").read_text())
        self.assertEqual(catalog["schema_version"], 1)
        self.assertTrue(
            all("contract" not in entry for entry in catalog["packs"][0]["entries"])
        )
        self.assertEqual(len(catalog["packs"][0]["entries"]), 3)

    def test_validation_contract_flag_creates_schema_two_and_remains_extractable(self):
        source = self.root / "contract.payload"
        source.write_bytes(b"synthetic contracted payload")
        output = self.root / "contracted"
        result = self._run(
            "create",
            "--validation-contract",
            "--output-dir",
            output,
            "--pack-id",
            "contracted",
            "--entry",
            "validation/saxpy",
            _TARGET,
            "cubin",
            "therock_module_saxpy",
            source,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        catalog = json.loads((output / "catalog.json").read_text())
        self.assertEqual(catalog["schema_version"], 2)
        entry = catalog["packs"][0]["entries"][0]
        self.assertEqual(entry["contract"]["version"], 1)
        self.assertEqual(entry["contract"]["pointer_bits"], 64)
        extracted = self.root / "contracted.bin"
        result = self._run(
            "extract",
            "--catalog",
            output / "catalog.json",
            "--module",
            "validation/saxpy",
            "--target",
            _TARGET,
            "--format",
            "cubin",
            "--entry-point",
            "therock_module_saxpy",
            "--output",
            extracted,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(extracted.read_bytes(), source.read_bytes())

    def test_invalid_create_leaves_output_absent(self):
        source = self.root / "source.payload"
        source.write_bytes(b"synthetic")
        output = self.root / "absent"
        result = self._run(
            "create",
            "--output-dir",
            output,
            "--pack-id",
            "bad",
            "--entry",
            "saxpy",
            _TARGET,
            "hsaco",
            "launch",
            source,
        )
        self.assertEqual(result.returncode, 2)
        self.assertFalse(output.exists())
        self.assertNotIn("Traceback", result.stderr)

    def test_failed_extract_preserves_existing_output(self):
        catalog = self._create("test")
        output = self.root / "result.bin"
        output.write_bytes(b"keep me")
        result = self._run(
            "extract",
            "--catalog",
            catalog,
            "--module",
            "saxpy",
            "--target",
            _TARGET,
            "--format",
            "cubin",
            "--entry-point",
            "undeclared",
            "--output",
            output,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("not declared", result.stderr)
        self.assertEqual(output.read_bytes(), b"keep me")
        self.assertNotIn("Traceback", result.stderr)

    def test_duplicate_match_reports_ambiguity(self):
        first = self._create("first")
        second = self._create("second")
        output = self.root / "absent.bin"
        result = self._run(
            "extract",
            "--catalog",
            first,
            "--catalog",
            second,
            "--module",
            "saxpy",
            "--target",
            _TARGET,
            "--format",
            "cubin",
            "--output",
            output,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Ambiguous", result.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
