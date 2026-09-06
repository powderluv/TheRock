# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
sys.path.insert(
    0,
    os.fspath(Path(__file__).parent.parent.parent / "rocm-systems/shared/kpack/python"),
)

import msgpack
from rocm_kpack.kpack import PackedKernelArchive

from _therock_utils.payload_catalog import (
    PayloadCatalog,
    PayloadInput,
    create_pack,
    extract_payload,
    load_catalog,
)


_NVIDIA = "nvidia:cuda:sm_120"
_AMD = "amd:hip:gfx1201"
_INTEL = "intel:level-zero:xe2-b70"


def _input(module="math/saxpy", target=_NVIDIA, payload_type="cubin", data=b"kernel"):
    return PayloadInput(module, target, payload_type, ("launch",), data)


def _edit_catalog(path, update):
    value = json.loads(path.read_text())
    update(value)
    path.write_text(json.dumps(value))


class PayloadCatalogTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def _create(self, pack_id="pack-a", inputs=None):
        return create_pack(self.root / pack_id, pack_id, inputs or (_input(),))

    def test_real_archive_roundtrip_all_vendor_payload_types(self):
        inputs = (
            _input(target=_AMD, payload_type="hsaco", data=b"amd-object"),
            _input(data=b"cuda-object"),
            _input(payload_type="ptx", data=b"cuda-assembly"),
            _input(target=_INTEL, payload_type="spirv", data=b"intel-bytecode"),
        )
        path = self._create(inputs=inputs)
        catalog = load_catalog(path)
        self.assertIsInstance(catalog, PayloadCatalog)
        self.assertEqual(catalog.schema_version, 1)
        self.assertEqual(len(catalog.packs[0].entries), 4)
        archive_path = path.parent / catalog.packs[0].path
        self.assertEqual(archive_path.read_bytes()[:4], b"KPAK")
        archive = PackedKernelArchive.read(archive_path)
        for item in inputs:
            with self.subTest(target=item.target, payload_type=item.payload_type):
                self.assertEqual(
                    extract_payload(
                        (path,),
                        item.module,
                        item.target,
                        item.payload_type,
                        entry_point="launch",
                    ),
                    item.data,
                )
                toc = archive.toc[item.module + "/" + item.payload_type]
                self.assertIn(item.target, toc)
                self.assertEqual(toc[item.target]["type"], item.payload_type)
        self.assertEqual(set(archive.gfx_arches), {_AMD, _NVIDIA, _INTEL})

    def test_same_target_missing_module_in_first_pack_continues(self):
        first = self._create("pack-first", (_input(module="other"),))
        second = self._create("pack-second", (_input(data=b"wanted"),))
        for catalogs in ((first, second), (second, first)):
            self.assertEqual(
                extract_payload(catalogs, "math/saxpy", _NVIDIA, "cubin"), b"wanted"
            )

    def test_exact_target_and_format_never_use_isa_fallback(self):
        path = self._create()
        for target, payload_type in (
            ("nvidia:cuda:sm_90", "cubin"),
            (_NVIDIA, "ptx"),
            (_AMD, "hsaco"),
        ):
            with self.subTest(target=target, payload_type=payload_type):
                with self.assertRaisesRegex(ValueError, "No exact payload match"):
                    extract_payload((path,), "math/saxpy", target, payload_type)
        amd = self._create(
            "amd", (_input(target="amd:hip:gfx942:xnack+", payload_type="hsaco"),)
        )
        with self.assertRaisesRegex(ValueError, "No exact payload match"):
            extract_payload((amd,), "math/saxpy", "amd:hip:gfx942:xnack-", "hsaco")

    def test_duplicate_matching_payloads_are_ambiguous(self):
        first = self._create("first")
        second = self._create("second")
        with self.assertRaisesRegex(ValueError, "Ambiguous payload"):
            extract_payload((first, second), "math/saxpy", _NVIDIA, "cubin")
        with self.assertRaisesRegex(ValueError, "Duplicate pack ID"):
            extract_payload((first, first), "math/saxpy", _NVIDIA, "cubin")

    def test_invalid_inputs_fail_before_output_creation(self):
        output = self.root / "absent"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            create_pack(output, "test", (_input(), _input()))
        self.assertFalse(output.exists())
        with self.assertRaises(ValueError):
            create_pack(output, "../escape", (_input(),))
        self.assertFalse(output.exists())
        for target, payload_type in (
            (_AMD, "cubin"),
            (_NVIDIA, "hsaco"),
            (_INTEL, "ptx"),
        ):
            with self.subTest(target=target), self.assertRaises(ValueError):
                _input(target=target, payload_type=payload_type)
        with self.assertRaises(ValueError):
            _input(module="../unsafe")
        with self.assertRaises(ValueError):
            _input(data=b"")

    def test_archive_and_catalog_are_deterministic(self):
        first = (_input(module="z"), _input(module="a", payload_type="ptx"))
        a = create_pack(self.root / "one", "deterministic", first)
        b = create_pack(self.root / "two", "deterministic", reversed(first))
        self.assertEqual(a.read_bytes(), b.read_bytes())
        self.assertEqual(
            (a.parent / "deterministic.kpack").read_bytes(),
            (b.parent / "deterministic.kpack").read_bytes(),
        )

    def test_existing_outputs_are_preserved_and_publish_failure_rolls_back(self):
        path = self._create()
        before = {item.name: item.read_bytes() for item in path.parent.iterdir()}
        with self.assertRaises(FileExistsError):
            create_pack(path.parent, "pack-a", (_input(data=b"replacement"),))
        self.assertEqual(
            before, {item.name: item.read_bytes() for item in path.parent.iterdir()}
        )
        original_link = os.link
        calls = 0

        def fail_second_link(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("Publication interrupted")
            return original_link(source, destination)

        with mock.patch(
            "_therock_utils.payload_catalog.os.link", side_effect=fail_second_link
        ):
            with self.assertRaisesRegex(OSError, "Publication interrupted"):
                create_pack(self.root / "new", "new", (_input(),))
        self.assertEqual(list((self.root / "new").iterdir()), [])

    def test_wrong_schema_and_duplicate_json_fields_fail_closed(self):
        path = self._create()
        baseline = path.read_text()
        for version in (2, True, "1", None):
            path.write_text(baseline)
            _edit_catalog(path, lambda value: value.update(schema_version=version))
            with self.subTest(version=version), self.assertRaises(ValueError):
                load_catalog(path)
        path.write_text('{"schema_version":1,"schema_version":1,"packs":[]}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            load_catalog(path)
        path.write_text(baseline)
        _edit_catalog(path, lambda value: value.update(unknown=123))
        with self.assertRaisesRegex(ValueError, "fields"):
            load_catalog(path)

    def test_duplicate_catalog_entries_and_malformed_hashes_rejected(self):
        path = self._create()
        baseline = path.read_text()
        _edit_catalog(
            path,
            lambda value: value["packs"][0]["entries"].append(
                value["packs"][0]["entries"][0].copy()
            ),
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_catalog(path)
        path.write_text(baseline)
        _edit_catalog(
            path, lambda value: value["packs"][0].update(sha256="not-a-digest")
        )
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_catalog(path)

    def test_pack_and_payload_hash_mismatches_fail_closed(self):
        path = self._create()
        baseline = path.read_text()
        _edit_catalog(path, lambda value: value["packs"][0].update(sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "Pack SHA256"):
            extract_payload((path,), "math/saxpy", _NVIDIA, "cubin")
        path.write_text(baseline)
        _edit_catalog(
            path, lambda value: value["packs"][0]["entries"][0].update(sha256="0" * 64)
        )
        with self.assertRaisesRegex(ValueError, "Payload SHA256"):
            extract_payload((path,), "math/saxpy", _NVIDIA, "cubin")

    def test_unselected_pack_integrity_is_also_checked(self):
        first = self._create("first")
        other = self._create("other", (_input(module="unrelated"),))
        _edit_catalog(other, lambda value: value["packs"][0].update(sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "Pack SHA256"):
            extract_payload((first, other), "math/saxpy", _NVIDIA, "cubin")

    def test_path_traversal_absolute_paths_and_symlink_escape_rejected(self):
        path = self._create()
        baseline = path.read_text()
        for unsafe in (
            "../outside.kpack",
            "/tmp/outside.kpack",
            "x/../../outside.kpack",
            "x\\outside.kpack",
        ):
            path.write_text(baseline)
            _edit_catalog(path, lambda value: value["packs"][0].update(path=unsafe))
            with self.subTest(path=unsafe), self.assertRaises(ValueError):
                extract_payload((path,), "math/saxpy", _NVIDIA, "cubin")
        path.write_text(baseline)
        pack_path = path.parent / "pack-a.kpack"
        outside = self.root / "outside.kpack"
        pack_path.rename(outside)
        pack_path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes"):
            extract_payload((path,), "math/saxpy", _NVIDIA, "cubin")

    def test_symbol_allowlist_rejects_undeclared_entry_point(self):
        path = self._create()
        with self.assertRaisesRegex(ValueError, "not declared"):
            extract_payload(
                (path,), "math/saxpy", _NVIDIA, "cubin", entry_point="not_listed"
            )

    def test_truncated_archive_fails_closed_even_with_updated_pack_hash(self):
        path = self._create()
        pack_path = path.parent / load_catalog(path).packs[0].path
        pack_path.write_bytes(b"KPAK")
        _edit_catalog(
            path,
            lambda value: value["packs"][0].update(
                sha256=hashlib.sha256(b"KPAK").hexdigest()
            ),
        )
        with self.assertRaisesRegex(ValueError, "Invalid packed payload"):
            extract_payload((path,), "math/saxpy", _NVIDIA, "cubin")

    def test_archive_type_and_entry_point_metadata_must_match_catalog(self):
        for change, message in (
            (
                lambda toc: toc["toc"]["math/saxpy/cubin"][_NVIDIA].update(
                    type="hsaco"
                ),
                "payload type",
            ),
            (
                lambda toc: toc["toc"]["math/saxpy/cubin"][_NVIDIA]["metadata"].update(
                    entry_points=["other"]
                ),
                "entry points",
            ),
            (lambda toc: toc.update(gfx_arches=["gfx1201"]), "target list"),
        ):
            with self.subTest(message=message):
                path = self._create("mutation-" + message.replace(" ", "-"))
                pack_path = path.parent / load_catalog(path).packs[0].path
                data = pack_path.read_bytes()
                offset = struct.unpack("<4sIQ", data[:16])[2]
                toc = msgpack.unpackb(data[offset:], raw=False)
                change(toc)
                altered = data[:offset] + msgpack.packb(toc, use_bin_type=True)
                pack_path.write_bytes(altered)
                _edit_catalog(
                    path,
                    lambda value: value["packs"][0].update(
                        sha256=hashlib.sha256(altered).hexdigest()
                    ),
                )
                with self.assertRaisesRegex(ValueError, message):
                    extract_payload((path,), "math/saxpy", _NVIDIA, "cubin")


if __name__ == "__main__":
    unittest.main()
