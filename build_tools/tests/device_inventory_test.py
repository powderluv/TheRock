# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import copy
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _therock_utils.device_inventory import (
    DeviceInventory,
    parse_device_uuid,
    parse_inventory_json,
    select_device,
    select_device_uuid,
)
from _therock_utils.gpu_targets import parse_gpu_target


_UUID_A = "00112233445566778899aabbccddeeff"
_UUID_B = "ffeeddccbbaa99887766554433221100"
_UUID_MISSING = "11111111111111111111111111111111"


def inventory_record(vendor="nvidia"):
    return {
        "schema_version": 2,
        "kind": "native-device-inventory",
        "scope": "observed-runtime",
        "vendor": vendor,
        "backend": {"amd": "hip", "nvidia": "cuda", "intel": "level-zero"}[vendor],
        "devices": [
            {
                "index": 0,
                "device_uuid": _UUID_A,
                "name": 'GPU "quoted"\\name\nΩ',
                "architecture": {"amd": "gfx1201", "nvidia": "sm_120", "intel": None}[
                    vendor
                ],
                "vendor_id": {"amd": 0x1002, "nvidia": 0x10DE, "intel": 0x8086}[vendor],
                "device_id": 0xE223 if vendor == "intel" else None,
                "limits": {
                    "max_threads_per_block": 1024,
                    "max_block_dimensions": [1024, 1024, 64],
                    "max_grid_dimensions": [2147483647, 65535, 65535],
                    "total_memory_bytes": 1 << 35,
                },
            }
        ],
    }


class DeviceInventoryTest(unittest.TestCase):
    def parse(self, record):
        return parse_inventory_json(json.dumps(record))

    def test_all_vendors_roundtrip_and_exact_selection(self):
        for vendor, target in [
            ("amd", "amd:hip:gfx1201"),
            ("nvidia", "nvidia:cuda:sm_120"),
            ("intel", "intel:level-zero:xe2-b70"),
        ]:
            with self.subTest(vendor=vendor):
                record = inventory_record(vendor)
                parsed = self.parse(record)
                self.assertEqual(parsed.record(), record)
                self.assertEqual(
                    select_device(parsed, parse_gpu_target(target), 0),
                    parsed.devices[0],
                )
                self.assertEqual(
                    select_device_uuid(parsed, parse_gpu_target(target), _UUID_A),
                    parsed.devices[0],
                )

    def test_empty_inventory_does_not_select_device(self):
        record = inventory_record()
        record["devices"] = []
        parsed = self.parse(record)
        self.assertEqual(parsed.devices, ())
        with self.assertRaisesRegex(ValueError, "unavailable"):
            select_device(parsed, parse_gpu_target("nvidia:cuda:sm_120"), 0)

    def test_unknown_duplicate_and_version_fields_rejected(self):
        record = inventory_record()
        for update in [
            {"schema_version": True},
            {"schema_version": 1},
            {"schema_version": 3},
            {"extra": 1},
            {"scope": "compiled-adapter"},
            {"backend": "hip"},
        ]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.parse({**record, **update})
        text = json.dumps(record).replace(
            '"schema_version": 2', '"schema_version": 2, "schema_version": 2'
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_inventory_json(text)

    def test_malformed_device_fields_rejected(self):
        for update in [
            {"index": True},
            {"index": -1},
            {"index": 1},
            {"name": ""},
            {"architecture": "sm_120a"},
            {"vendor_id": 0x1002},
            {"device_id": True},
            {"device_id": 0x10000},
            {"extra": 1},
        ]:
            with self.subTest(update=update):
                record = inventory_record()
                record["devices"][0].update(update)
                with self.assertRaises(ValueError):
                    self.parse(record)

    def test_limits_are_strict_positive_integers(self):
        for update in [
            {"max_threads_per_block": True},
            {"max_threads_per_block": 0},
            {"max_block_dimensions": [128, 1]},
            {"max_grid_dimensions": [1, 0, 1]},
            {"total_memory_bytes": 1.5},
            {"total_memory_bytes": 1 << 64},
            {"unknown": 1},
        ]:
            with self.subTest(update=update):
                record = inventory_record()
                record["devices"][0]["limits"].update(update)
                with self.assertRaises(ValueError):
                    self.parse(record)

    def test_indices_must_match_launch_order(self):
        record = inventory_record()
        record["devices"].append(copy.deepcopy(record["devices"][0]))
        with self.assertRaisesRegex(ValueError, "consecutive"):
            self.parse(record)
        record["devices"][1]["index"] = 1
        record["devices"][1]["device_uuid"] = _UUID_B
        parsed = self.parse(record)
        self.assertEqual(
            select_device(parsed, parse_gpu_target("nvidia:cuda:sm_120"), 1).index, 1
        )

    def test_architecture_and_vendor_mismatch_fail(self):
        parsed = self.parse(inventory_record())
        for target in ["nvidia:cuda:sm_90", "amd:hip:gfx1201"]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                select_device(parsed, parse_gpu_target(target), 0)

    def test_unobservable_feature_suffixes_fail(self):
        for vendor, target in [
            ("nvidia", "nvidia:cuda:sm_120a"),
            ("amd", "amd:hip:gfx1201:xnack+"),
        ]:
            with self.subTest(target=target), self.assertRaisesRegex(
                ValueError, "feature suffixes"
            ):
                select_device(
                    self.parse(inventory_record(vendor)), parse_gpu_target(target), 0
                )

    def test_device_ordinals_rejected(self):
        for index in [-1, 1, True]:
            with self.subTest(index=index), self.assertRaisesRegex(
                ValueError, "unavailable"
            ):
                select_device(
                    self.parse(inventory_record()),
                    parse_gpu_target("nvidia:cuda:sm_120"),
                    index,
                )

    def test_intel_requires_pci_identity_not_payload_format(self):
        for update in [
            {"architecture": "spirv"},
            {"architecture": "xe2-b70"},
            {"device_id": None},
        ]:
            with self.subTest(update=update):
                record = inventory_record("intel")
                record["devices"][0].update(update)
                with self.assertRaises(ValueError):
                    self.parse(record)
        record = inventory_record("intel")
        record["devices"][0]["device_id"] = 0xE20B
        with self.assertRaisesRegex(ValueError, "does not match"):
            select_device(
                self.parse(record), parse_gpu_target("intel:level-zero:xe2-b70"), 0
            )

    def test_intel_other_products_need_explicit_id(self):
        parsed = self.parse(inventory_record("intel"))
        target = parse_gpu_target("intel:level-zero:xe2-other")
        with self.assertRaisesRegex(ValueError, "explicit"):
            select_device(parsed, target, 0)
        self.assertEqual(select_device(parsed, target, 0, 0xE223), parsed.devices[0])
        with self.assertRaisesRegex(ValueError, "requires device ID"):
            select_device(
                parsed, parse_gpu_target("intel:level-zero:xe2-b70"), 0, 0xE20B
            )

    def test_schema_one_is_rejected_and_schema_two_requires_uuid(self):
        record = inventory_record()
        record["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "schema"):
            self.parse(record)
        del record["devices"][0]["device_uuid"]
        with self.assertRaisesRegex(ValueError, "schema"):
            self.parse(record)
        record["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "fields"):
            self.parse(record)

    def test_uuid_parser_rejects_noncanonical_and_zero_values(self):
        malformed = (
            None,
            True,
            1,
            1.5,
            [],
            {},
            b"00112233445566778899aabbccddeeff",
            "",
            "0" * 32,
            _UUID_A[:-1],
            _UUID_A + "0",
            _UUID_A.upper(),
            "00112233-4455-6677-8899-aabbccddeeff",
            "GPU-" + _UUID_A,
            " " + _UUID_A,
            _UUID_A + "\n",
            "g" * 32,
        )
        parsed = self.parse(inventory_record())
        target = parse_gpu_target("nvidia:cuda:sm_120")
        for value in malformed:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "UUID"):
                    parse_device_uuid(value)
                with self.assertRaisesRegex(ValueError, "UUID"):
                    select_device_uuid(parsed, target, value)
                with self.assertRaisesRegex(ValueError, "UUID"):
                    replace(parsed.devices[0], device_uuid=value)
                if not isinstance(value, bytes):
                    record = inventory_record()
                    record["devices"][0]["device_uuid"] = value
                    with self.assertRaisesRegex(ValueError, "UUID"):
                        self.parse(record)
        self.assertEqual(parse_device_uuid(_UUID_A), _UUID_A)
        self.assertEqual(parse_device_uuid(_UUID_B), _UUID_B)

    def test_duplicate_uuid_and_duplicate_uuid_json_field_rejected(self):
        record = inventory_record()
        second = copy.deepcopy(record["devices"][0])
        second.update(index=1, name="Distinct ordinal with ambiguous UUID")
        record["devices"].append(second)
        with self.assertRaisesRegex(ValueError, "Duplicate device UUID"):
            self.parse(record)
        text = json.dumps(inventory_record()).replace(
            f'"device_uuid": "{_UUID_A}"',
            f'"device_uuid": "{_UUID_A}", "device_uuid": "{_UUID_B}"',
        )
        with self.assertRaisesRegex(ValueError, "Duplicate inventory field"):
            parse_inventory_json(text)

    def test_uuid_selection_survives_ordinal_reordering(self):
        record = inventory_record()
        record["devices"][0]["name"] = "First observed identity"
        second = copy.deepcopy(record["devices"][0])
        second.update(index=1, name="Second observed identity", device_uuid=_UUID_B)
        record["devices"].append(second)
        before = self.parse(record)
        target = parse_gpu_target("nvidia:cuda:sm_120")
        first = select_device_uuid(before, target, _UUID_A)
        record["devices"].reverse()
        for index, device in enumerate(record["devices"]):
            device["index"] = index
        after = self.parse(record)
        moved = select_device_uuid(after, target, _UUID_A)
        self.assertEqual(first.index, 0)
        self.assertEqual(moved.index, 1)
        self.assertEqual(moved.device_uuid, first.device_uuid)
        self.assertEqual(moved.name, first.name)
        self.assertEqual(select_device(after, target, 0).device_uuid, _UUID_B)
        self.assertNotEqual(
            select_device(after, target, 0).device_uuid, first.device_uuid
        )

    def test_uuid_does_not_override_vendor_architecture_or_feature_checks(self):
        parsed = self.parse(inventory_record())
        for target in (
            "amd:hip:gfx1201",
            "intel:level-zero:xe2-b70",
            "nvidia:cuda:sm_90",
            "nvidia:cuda:sm_120a",
        ):
            with self.subTest(target=target), self.assertRaises(ValueError):
                select_device_uuid(parsed, parse_gpu_target(target), _UUID_A)
        # Identical raw bytes in different vendor inventories are allowed, but
        # selection remains scoped to each inventory's vendor and backend.
        amd = self.parse(inventory_record("amd"))
        self.assertEqual(amd.devices[0].device_uuid, parsed.devices[0].device_uuid)
        with self.assertRaisesRegex(ValueError, "vendor/backend"):
            select_device_uuid(amd, parse_gpu_target("nvidia:cuda:sm_120"), _UUID_A)
        with self.assertRaisesRegex(ValueError, "feature suffixes"):
            select_device_uuid(amd, parse_gpu_target("amd:hip:gfx1201:xnack+"), _UUID_A)

    def test_uuid_does_not_override_intel_pci_identity(self):
        record = inventory_record("intel")
        record["devices"][0]["device_id"] = 0xE20B
        parsed = self.parse(record)
        target = parse_gpu_target("intel:level-zero:xe2-b70")
        with self.assertRaisesRegex(ValueError, "does not match"):
            select_device_uuid(parsed, target, _UUID_A)
        with self.assertRaisesRegex(ValueError, "requires device ID"):
            select_device_uuid(parsed, target, _UUID_A, 0xE20B)
        other = parse_gpu_target("intel:level-zero:xe2-other")
        with self.assertRaisesRegex(ValueError, "explicit"):
            select_device_uuid(parsed, other, _UUID_A)
        self.assertEqual(
            select_device_uuid(parsed, other, _UUID_A, 0xE20B), parsed.devices[0]
        )

    def test_missing_uuid_and_manually_ambiguous_inventory_fail_closed(self):
        parsed = self.parse(inventory_record())
        target = parse_gpu_target("nvidia:cuda:sm_120")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            select_device_uuid(parsed, target, _UUID_MISSING)
        empty = self.parse({**inventory_record(), "devices": []})
        with self.assertRaisesRegex(ValueError, "unavailable"):
            select_device_uuid(empty, target, _UUID_A)
        duplicate = DeviceInventory(
            parsed.vendor,
            parsed.backend,
            (parsed.devices[0], replace(parsed.devices[0], index=1)),
        )
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            select_device_uuid(duplicate, target, _UUID_A)

    def test_deep_json_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_inventory_json(
                "[" * (sys.getrecursionlimit() + 100)
                + "]" * (sys.getrecursionlimit() + 100)
            )


if __name__ == "__main__":
    unittest.main()
