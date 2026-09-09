# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real KPAK assembly tests with synthetic bytes; no GPU qualification is implied."""

import hashlib
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
from _therock_utils.runner_registry import load_registry
from _therock_utils.module_contract import (
    parse_contract,
    runner_description,
    validation_contract,
)
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
            description = self.description_path(target)
            description.parent.mkdir(parents=True, exist_ok=True)
            description.write_text(
                json.dumps(runner_description(target.vendor).record())
            )
            description.with_name("therock_module_validation").write_bytes(
                f"synthetic runner for {target.canonical_id}".encode()
            )
            for module in MODULES:
                for payload_type in target.payload_types:
                    data = f"synthetic:{module}:{target.canonical_id}:{payload_type}".encode()
                    path = self.payload_path(target, module, payload_type)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    self.expected[(module, target.canonical_id, payload_type)] = data

    def description_path(self, target):
        return (
            self.build_root
            / target.slug
            / "stage/bin"
            / target.slug
            / "runner-contract.json"
        )

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

    def test_multiple_targets_roundtrip_with_exact_formats_packs_and_registry(self):
        self.select((AMD, AMD_FEATURES, NVIDIA_OTHER, NVIDIA, INTEL))
        self.assemble()
        expected_files = {
            f"{module}/{filename}"
            for module in MODULES
            for filename in ("catalog.json", f"validation-{module}.kpack")
        }
        expected_files.add("runners.json")
        self.assertEqual(set(self.snapshot()), expected_files)
        registry = load_registry(self.output / "runners.json")
        self.assertEqual(
            {entry.target.canonical_id for entry in registry.runners},
            {target.canonical_id for target in self.targets},
        )
        self.assertEqual(
            registry.catalogs,
            tuple(f"share/therock/packs/{module}/catalog.json" for module in MODULES),
        )
        for entry in registry.runners:
            self.assertEqual(
                entry.path, f"bin/{entry.target.slug}/therock_module_validation"
            )
            self.assertEqual(entry.description, runner_description(entry.target.vendor))
            self.assertEqual(
                entry.sha256,
                hashlib.sha256(
                    self.description_path(entry.target)
                    .with_name("therock_module_validation")
                    .read_bytes()
                ).hexdigest(),
            )
        catalogs = tuple(self.output / module / "catalog.json" for module in MODULES)
        for module, catalog_path in zip(MODULES, catalogs):
            catalog = load_catalog(catalog_path)
            self.assertEqual(catalog.schema_version, 2)
            self.assertEqual(len(catalog.packs), 1)
            for entry in catalog.packs[0].entries:
                self.assertEqual(
                    entry.contract.record(), validation_contract().record()
                )
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

    def test_missing_later_runner_preserves_previous_outputs(self):
        self.assemble()
        original = self.snapshot()
        self.description_path(self.targets[-1]).with_name(
            "therock_module_validation"
        ).unlink()
        with self.assertRaises(FileNotFoundError):
            self.assemble()
        self.assertEqual(self.snapshot(), original)

    def test_empty_runner_preserves_previous_outputs(self):
        self.assemble()
        original = self.snapshot()
        self.description_path(self.targets[-1]).with_name(
            "therock_module_validation"
        ).write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Staged runner is empty"):
            self.assemble()
        self.assertEqual(self.snapshot(), original)

    def test_changed_runner_updates_only_registry(self):
        self.assemble()
        original = self.snapshot()
        updated = b"updated synthetic runner"
        self.description_path(self.targets[-1]).with_name(
            "therock_module_validation"
        ).write_bytes(updated)
        self.assemble()
        changed = {
            name for name, data in self.snapshot().items() if data != original[name]
        }
        self.assertEqual(changed, {"runners.json"})
        entry = next(
            entry
            for entry in load_registry(self.output / "runners.json").runners
            if entry.target == self.targets[-1]
        )
        self.assertEqual(entry.sha256, hashlib.sha256(updated).hexdigest())

    def test_missing_later_runner_description_preserves_previous_outputs(self):
        self.assemble()
        original = self.snapshot()
        self.description_path(self.targets[-1]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.assemble()
        self.assertEqual(self.snapshot(), original)

    def test_incompatible_staged_runner_description_cannot_relabel_payloads(self):
        self.assemble()
        original = self.snapshot()
        path = self.description_path(self.targets[-1])
        valid = runner_description(self.targets[-1].vendor).record()
        alternate = parse_contract({**validation_contract().record(), "version": 2})
        changes = (
            {
                **valid,
                "contract": alternate.record(),
                "contract_sha256": alternate.sha256,
            },
            {**valid, "vendor": "amd", "backend": "hip"},
            {**valid, "payload_types": ["cubin"]},
            {**valid, "entry_points": ["therock_module_saxpy"]},
            {**valid, "contract_sha256": "0" * 64},
            {**valid, "scope": "hardware-qualified"},
        )
        for description in changes:
            with self.subTest(description=description):
                path.write_text(json.dumps(description))
                with self.assertRaises(ValueError):
                    self.assemble()
                self.assertEqual(self.snapshot(), original)
        path.write_text(json.dumps(valid))
        self.assemble()
        self.assertEqual(self.snapshot(), original)

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
