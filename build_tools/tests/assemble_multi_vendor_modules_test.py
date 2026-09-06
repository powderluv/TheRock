# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real KPAK assembly tests with synthetic bytes; no GPU qualification is implied."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "build_tools"))
sys.path.insert(0, str(REPO_ROOT / "rocm-systems/shared/kpack/python"))

from rocm_kpack.kpack import PackedKernelArchive
from _therock_utils.gpu_targets import parse_gpu_targets
from _therock_utils.payload_catalog import extract_payload, load_catalog
from assemble_multi_vendor_modules import MODULES, assemble, read_targets
from configure_multi_vendor import configure_targets


AMD = "amd:hip:gfx1201"
AMD_FEATURES = "amd:hip:gfx942:xnack+:sramecc-"
NVIDIA = "nvidia:cuda:sm_120"
NVIDIA_OTHER = "nvidia:cuda:sm_90"
INTEL = "intel:level-zero:xe2-b70"


class AssembleMultiVendorModulesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.build_root = self.root / "modules"
        self.output = self.root / "packs"
        self.expected = {}
        self.select((AMD, NVIDIA))

    def select(self, identities):
        self.targets = parse_gpu_targets(identities)
        _, self.selector = configure_targets(identities, self.root / "selection")
        for target in self.targets:
            for module in MODULES:
                for payload_type in target.payload_types:
                    data = f"synthetic:{module}:{target.canonical_id}:{payload_type}".encode()
                    path = self.payload_path(target, module, payload_type)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    self.expected[(module, target.canonical_id, payload_type)] = data

    def payload_path(self, target, module, payload_type):
        return (
            self.build_root
            / target.slug
            / "stage/share/therock/modules"
            / target.slug
            / f"{module}.{payload_type}"
        )

    def snapshot(self):
        return {
            path.relative_to(self.output).as_posix(): path.read_bytes()
            for path in self.output.rglob("*")
            if path.is_file()
        }

    def assemble(self):
        assemble(self.selector, self.build_root, self.output)

    def test_multiple_targets_roundtrip_with_exact_formats_and_only_packs(self):
        self.select((AMD, AMD_FEATURES, NVIDIA_OTHER, NVIDIA, INTEL))
        self.assemble()
        expected_files = {
            f"{module}/{filename}"
            for module in MODULES
            for filename in ("catalog.json", f"validation-{module}.kpack")
        }
        self.assertEqual(set(self.snapshot()), expected_files)
        catalogs = tuple(self.output / module / "catalog.json" for module in MODULES)
        for module, catalog_path in zip(MODULES, catalogs):
            catalog = load_catalog(catalog_path)
            self.assertEqual(len(catalog.packs), 1)
            pack = catalog.packs[0]
            archive_path = catalog_path.parent / pack.path
            self.assertEqual(archive_path.read_bytes()[:4], b"KPAK")
            archive = PackedKernelArchive.read(archive_path)
            self.assertEqual(
                set(archive.gfx_arches), {t.canonical_id for t in self.targets}
            )
            expected_keys = {
                (f"validation/{module}", target.canonical_id, payload_type)
                for target in self.targets
                for payload_type in target.payload_types
            }
            self.assertEqual({entry.key for entry in pack.entries}, expected_keys)
            for target in self.targets:
                for payload_type in target.payload_types:
                    with self.subTest(
                        module=module, target=target, format=payload_type
                    ):
                        toc = archive.toc[f"validation/{module}/{payload_type}"][
                            target.canonical_id
                        ]
                        self.assertEqual(toc["type"], payload_type)
                        self.assertEqual(
                            extract_payload(
                                catalogs,
                                f"validation/{module}",
                                target.canonical_id,
                                payload_type,
                                entry_point=f"therock_module_{module}",
                            ),
                            self.expected[(module, target.canonical_id, payload_type)],
                        )

    def test_managed_output_can_be_rebuilt_deterministically(self):
        self.assemble()
        original = self.snapshot()
        self.assemble()
        self.assertEqual(self.snapshot(), original)
        # Selection order is not part of pack identity or binary serialization.
        self.select((NVIDIA, AMD))
        self.assemble()
        self.assertEqual(self.snapshot(), original)
        self.assertFalse(list(self.root.glob("module-packs-*")))

    def test_updated_payload_changes_only_its_module_pack(self):
        self.assemble()
        original = self.snapshot()
        target = self.targets[-1]
        updated = b"updated synthetic relu PTX bytes"
        self.payload_path(target, "relu", "ptx").write_bytes(updated)
        self.assemble()
        changed = {
            name for name, data in self.snapshot().items() if data != original[name]
        }
        self.assertEqual(changed, {"relu/catalog.json", "relu/validation-relu.kpack"})
        self.assertEqual(
            extract_payload(
                (self.output / "relu/catalog.json",),
                "validation/relu",
                NVIDIA,
                "ptx",
                entry_point="therock_module_relu",
            ),
            updated,
        )

    def test_missing_later_payload_preserves_previous_complete_output(self):
        self.assemble()
        original = self.snapshot()
        self.payload_path(self.targets[0], "saxpy", "hsaco").write_bytes(
            b"new first payload"
        )
        self.payload_path(self.targets[-1], "relu", "ptx").unlink()
        with self.assertRaises(FileNotFoundError):
            self.assemble()
        self.assertEqual(self.snapshot(), original)
        self.assertFalse(list(self.root.glob("module-packs-*")))

    def test_missing_payload_creates_no_output(self):
        self.payload_path(self.targets[-1], "relu", "ptx").unlink()
        with self.assertRaises(FileNotFoundError):
            self.assemble()
        self.assertFalse(self.output.exists())

    def test_empty_payload_preserves_previous_output(self):
        self.assemble()
        original = self.snapshot()
        self.payload_path(self.targets[-1], "relu", "ptx").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "nonempty bytes"):
            self.assemble()
        self.assertEqual(self.snapshot(), original)

    def test_mixed_intel_selection_includes_staged_spirv_payloads(self):
        self.select((AMD, INTEL))
        self.assertEqual(
            tuple(t.canonical_id for t in read_targets(self.selector)), (AMD, INTEL)
        )
        self.assemble()
        for module in MODULES:
            catalog = load_catalog(self.output / module / "catalog.json")
            self.assertEqual(
                {entry.target for entry in catalog.packs[0].entries}, {AMD, INTEL}
            )

    def test_intel_only_selection_assembles_spirv(self):
        self.select((INTEL,))
        self.assemble()
        for module in MODULES:
            catalog_path = self.output / module / "catalog.json"
            entries = load_catalog(catalog_path).packs[0].entries
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].key, (f"validation/{module}", INTEL, "spirv"))
            self.assertEqual(
                extract_payload(
                    (catalog_path,),
                    f"validation/{module}",
                    INTEL,
                    "spirv",
                    entry_point=f"therock_module_{module}",
                ),
                self.expected[(module, INTEL, "spirv")],
            )

    def test_missing_intel_payload_preserves_previous_complete_output(self):
        self.assemble()
        original = self.snapshot()
        self.select((AMD, NVIDIA, INTEL))
        self.payload_path(self.targets[-1], "relu", "spirv").unlink()
        with self.assertRaises(FileNotFoundError):
            self.assemble()
        self.assertEqual(self.snapshot(), original)
        self.assertFalse(list(self.root.glob("module-packs-*")))

    def test_invalid_selector_documents_leave_existing_output_unchanged(self):
        self.assemble()
        original = self.snapshot()
        valid = {
            "kind": "target-selection",
            "schema_version": 1,
            "targets": [{"id": AMD}],
        }
        invalid = [
            None,
            [],
            {},
            {**valid, "kind": "payload-catalog"},
            {**valid, "schema_version": True},
            {**valid, "schema_version": 2},
            {**valid, "schema_version": "1"},
            {**valid, "targets": None},
            {**valid, "targets": []},
            {**valid, "targets": [AMD]},
            {**valid, "targets": [{"id": 1}]},
            {**valid, "targets": [{"id": "amd:hip:../../escape"}]},
            {**valid, "targets": [{"id": AMD}, {"id": AMD}]},
            {
                **valid,
                "targets": [
                    {"id": AMD_FEATURES},
                    {"id": "amd:hip:gfx942:sramecc-:xnack+"},
                ],
            },
        ]
        for document in invalid:
            with self.subTest(document=document):
                self.selector.write_text(json.dumps(document))
                with self.assertRaises(ValueError):
                    self.assemble()
                self.assertEqual(self.snapshot(), original)

    def test_duplicate_selector_json_fields_are_rejected(self):
        self.assemble()
        original = self.snapshot()
        documents = [
            '{"kind":"other","kind":"target-selection","schema_version":1,"targets":[{"id":"amd:hip:gfx1201"}]}',
            '{"kind":"target-selection","schema_version":2,"schema_version":1,"targets":[{"id":"amd:hip:gfx1201"}]}',
            '{"kind":"target-selection","schema_version":1,"targets":[{"id":"nvidia:cuda:sm_120","id":"amd:hip:gfx1201"}]}',
        ]
        for document in documents:
            with self.subTest(document=document):
                self.selector.write_text(document)
                with self.assertRaisesRegex(
                    ValueError, "Duplicate target-selection field"
                ):
                    self.assemble()
                self.assertEqual(self.snapshot(), original)


if __name__ == "__main__":
    unittest.main()
