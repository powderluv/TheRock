# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU application-boundary tests of real packs, selection, and framed sessions."""

from array import array
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "build_tools"))
sys.path.insert(0, str(REPO_ROOT / "rocm-systems/shared/kpack/python"))

import therock_multi_vendor
from therock_multi_vendor import ModuleRequest, PackedModuleSession, open_session
from _therock_utils.gpu_targets import parse_gpu_target
from _therock_utils.module_contract import runner_description, validation_contract
from _therock_utils.module_service import ModuleServiceError, ModuleServiceRemoteError
from _therock_utils.payload_catalog import PayloadInput, create_pack
from dispatch_multi_vendor_modules_test import _inventory
from module_service_test import _WORKER

_TARGET = "nvidia:cuda:sm_120"
_UUID_A = "0123456789abcdef0123456789abcdef"
_UUID_B = "fedcba9876543210fedcba9876543210"
_SAXPY = ModuleRequest("validation/saxpy", "cubin", "therock_module_saxpy")
_RELU = ModuleRequest("validation/relu", "ptx", "therock_module_relu")

_QUERY = r"""
import json
from pathlib import Path
import sys

if sys.argv[1:] in (["--describe-contract"], ["--list-devices"]):
    with Path(__file__).with_suffix('.log').open('a') as log:
        log.write(json.dumps({'argv': sys.argv[1:]}) + '\n')
    if sys.argv[1:] == ["--describe-contract"]:
        print(json.dumps(CONFIG['description']))
    else:
        for path in CONFIG.get('mutate_catalogs', []):
            Path(path).write_text('{"after":"verified-extraction"}')
        if CONFIG.get('mutate_runner'):
            with Path(__file__).open('a') as runner:
                runner.write('\n# changed during discovery\n')
        print(json.dumps(CONFIG['inventory']))
    sys.exit(0)
CONFIG['description'] = CONFIG.get('hello_description', CONFIG['description'])
"""


class PackedModuleSessionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="therock-packed-client-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dist = self.root / "dist"
        self.target = parse_gpu_target(_TARGET)
        self.runner = self.dist / "bin" / self.target.slug / "therock_module_validation"
        self.runner.parent.mkdir(parents=True)
        self.registry_path = self.dist / "share/therock/packs/runners.json"
        self.registry = {
            "schema_version": 1,
            "kind": "native-runner-registry",
            "runners": [],
            "catalogs": [],
        }
        self.config = {
            "description": runner_description("nvidia").record(),
            "inventory": _inventory(_TARGET),
        }
        for name, fmt in (("saxpy", "cubin"), ("relu", "ptx")):
            path = create_pack(
                self.dist / "share/therock/packs" / name,
                f"validation-{name}",
                (
                    PayloadInput(
                        f"validation/{name}",
                        _TARGET,
                        fmt,
                        (f"therock_module_{name}",),
                        name.encode(),
                        validation_contract(),
                    ),
                ),
            )
            self.registry["catalogs"].append(path.relative_to(self.dist).as_posix())
        self.write_runner()

    def write_runner(self, **changes):
        self.config.update(changes)
        body = _WORKER.replace(
            "config = json.loads(Path(__file__).with_suffix('.json').read_text())",
            "config = CONFIG",
        )
        body = body.replace("buffers = {}", "loaded_paths = []\nbuffers = {}")
        body = body.replace(
            "        modules[next_handle] = symbol",
            "        loaded_paths.append(path)\n        record({'loaded_path': path, 'bytes': Path(path).read_bytes().hex()})\n        modules[next_handle] = symbol",
        )
        body = body.replace(
            "    if opcode == 11:\n",
            "    if opcode == 11:\n        record({'paths_at_close': {path: Path(path).is_file() for path in loaded_paths}})\n",
        )
        self.runner.write_text(
            f"#!{sys.executable}\nCONFIG = {self.config!r}\n" + _QUERY + body
        )
        self.runner.chmod(0o755)
        self.registry["runners"] = [
            {
                "target": _TARGET,
                "path": self.runner.relative_to(self.dist).as_posix(),
                "sha256": hashlib.sha256(self.runner.read_bytes()).hexdigest(),
                "description": runner_description("nvidia").record(),
            }
        ]
        self.write_registry()

    def write_registry(self):
        self.registry_path.write_text(json.dumps(self.registry))

    def records(self):
        log = self.runner.with_suffix(".log")
        return (
            [json.loads(line) for line in log.read_text().splitlines()]
            if log.exists()
            else []
        )

    def opcodes(self):
        return [record["opcode"] for record in self.records() if "opcode" in record]

    def session(self, requests=(_SAXPY, _RELU), **kwargs):
        session = open_session(self.dist, _TARGET, requests, **kwargs)
        self.addCleanup(session.close)
        return session

    def assert_private_cleanup(self):
        paths = [
            Path(record["loaded_path"])
            for record in self.records()
            if "loaded_path" in record
        ]
        self.assertTrue(paths)
        self.assertTrue(
            all(not path.exists() and not path.parent.exists() for path in paths)
        )
        close = [
            record["paths_at_close"]
            for record in self.records()
            if "paths_at_close" in record
        ]
        self.assertEqual(len(close), 1)
        self.assertTrue(all(close[0].values()))

    def test_application_arrays_use_preloaded_modules_and_retained_private_files(self):
        with self.session() as session:
            self.assertIsInstance(session, PackedModuleSession)
            self.assertFalse(hasattr(session, "load"))
            self.assertEqual(session.target, self.target)
            self.assertEqual(session.device.device_uuid, _UUID_A)
            self.assertEqual(
                session.runner_sha256, self.registry["runners"][0]["sha256"]
            )
            saxpy = session.module(_SAXPY)
            self.assertIs(
                saxpy,
                session.module(
                    ModuleRequest(
                        *(_SAXPY.module, _SAXPY.payload_type, _SAXPY.entry_point)
                    )
                ),
            )
            self.assertEqual(self.opcodes(), [1, 2, 5, 5])
            x, y, output = (session.allocate(8) for _ in range(3))
            session.write(x, 0, [1, 2, 3, 4])
            session.write(y, 0, array("f", [0.5] * 4))
            session.write(output, 0, [-77.0] * 8)
            session.launch(saxpy, x, y, output, 1.25, 4)
            self.assertEqual(
                session.read(output, 0, 4), array("f", [1.75, 3.0, 4.25, 5.5])
            )
            self.assertEqual(session.read(output, 4, 4), array("f", [-77] * 4))
            session.synchronize()
            session.release(y)
            self.assertTrue(y.closed)
        self.assertTrue(session.closed)
        self.assertTrue(x.closed)
        self.assertTrue(saxpy.closed)
        self.assert_private_cleanup()
        with self.assertRaises(ModuleServiceError):
            session.module(_SAXPY)

    def test_public_type_and_error_vocabulary_requires_no_private_import(self):
        expected = {
            "ModuleRequest",
            "PackedModuleSession",
            "open_session",
            "Buffer",
            "Module",
            "ModuleServiceError",
            "ModuleServiceProtocolError",
            "ModuleServiceTimeoutError",
            "ModuleServiceRemoteError",
            "SgemmProviderInfo",
        }
        self.assertEqual(set(therock_multi_vendor.__all__), expected)
        self.assertTrue(
            issubclass(
                therock_multi_vendor.ModuleServiceRemoteError,
                therock_multi_vendor.ModuleServiceError,
            )
        )
        with self.session((_SAXPY,)) as session:
            self.assertIsInstance(session.module(_SAXPY), therock_multi_vendor.Module)
            self.assertIsInstance(session.allocate(1), therock_multi_vendor.Buffer)

    def test_requests_are_typed_unique_and_bounded_before_queries(self):
        cases = ((), (_SAXPY,) * 2, (_SAXPY,) * 33, [_SAXPY], ("saxpy",))
        for requests in cases:
            with self.subTest(requests=requests), self.assertRaises(ValueError):
                self.session(requests)
        with self.assertRaises(ValueError):
            ModuleRequest("saxpy", [], "entry")
        self.assertEqual(self.records(), [])

    def test_all_32_unique_module_slots_are_available_to_application_callers(self):
        requests = tuple(
            ModuleRequest(f"validation/item{index}", "cubin", "therock_module_saxpy")
            for index in range(32)
        )
        path = create_pack(
            self.dist / "share/therock/packs/application",
            "application",
            tuple(
                PayloadInput(
                    request.module,
                    _TARGET,
                    request.payload_type,
                    (request.entry_point,),
                    b"saxpy",
                    validation_contract(),
                )
                for request in requests
            ),
        )
        self.registry["catalogs"].append(path.relative_to(self.dist).as_posix())
        self.write_registry()
        with self.session(requests) as session:
            handles = {session.module(request).handle for request in requests}
            self.assertEqual(len(handles), 32)
        self.assertEqual(self.opcodes().count(5), 32)
        self.assert_private_cleanup()
        self.runner.with_suffix(".log").unlink()
        with self.assertRaisesRegex(ValueError, "between 1 and 32"):
            self.session((*requests, _SAXPY))
        self.assertEqual(self.records(), [])

    def test_bad_later_selection_does_not_start_any_runner(self):
        for later in (
            ModuleRequest("validation/missing", "cubin", "therock_module_saxpy"),
            ModuleRequest("validation/relu", "spirv", "therock_module_relu"),
            ModuleRequest("validation/relu", "ptx", "undeclared"),
        ):
            with self.subTest(later=later), self.assertRaises(ValueError):
                self.session((_SAXPY, later))
        self.assertEqual(self.records(), [])

    def test_corrupt_later_pack_prevents_all_queries(self):
        catalog_path = self.dist / self.registry["catalogs"][1]
        catalog = json.loads(catalog_path.read_text())
        archive = catalog_path.parent / catalog["packs"][0]["path"]
        archive.write_bytes(archive.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Pack SHA256 mismatch"):
            self.session()
        self.assertEqual(self.records(), [])

    def test_legacy_or_different_contract_is_rejected_before_queries(self):
        for contract in (None, replace(validation_contract(), version=2)):
            with self.subTest(contract=contract):
                path = create_pack(
                    self.root / ("legacy" if contract is None else "different"),
                    "replacement",
                    (
                        PayloadInput(
                            "validation/relu",
                            _TARGET,
                            "ptx",
                            ("therock_module_relu",),
                            b"relu",
                            contract,
                        ),
                    ),
                )
                # Put the replacement catalog under the same distribution root.
                import shutil

                destination = self.dist / path.parent.name
                shutil.copytree(path.parent, destination)
                self.registry["catalogs"][1] = (
                    (destination / path.name).relative_to(self.dist).as_posix()
                )
                self.write_registry()
                with self.assertRaises(ValueError):
                    self.session()
                self.assertEqual(self.records(), [])

    def test_missing_or_unknown_capability_prevents_all_queries(self):
        for requested in (("unknown",), ("cross-queue-events", "cross-queue-events")):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                self.session(required_capabilities=requested)
        description = runner_description("nvidia").record()
        description["capabilities"].remove("persistent-module-service")
        self.write_runner(description=description)
        self.registry["runners"][0]["description"] = description
        self.write_registry()
        with self.assertRaisesRegex(ValueError, "persistent-module-service"):
            self.session()
        self.assertEqual(self.records(), [])

    def test_changed_runner_hash_before_discovery_or_serve_is_rejected(self):
        self.runner.write_text(self.runner.read_text() + "\n# tampered\n")
        with self.assertRaisesRegex(ValueError, "Runner SHA256 mismatch"):
            self.session()
        self.assertEqual(self.records(), [])
        self.write_runner(mutate_runner=True)
        with self.assertRaisesRegex(ValueError, "Runner SHA256 mismatch"):
            self.session()
        self.assertEqual(self.opcodes(), [])
        self.assertEqual(
            [record["argv"] for record in self.records()],
            [["--describe-contract"], ["--list-devices"]],
        )

    def test_exact_hello_comparison_precedes_open(self):
        description = runner_description("nvidia").record()
        description["capabilities"].remove("device-module-pipeline")
        self.write_runner(hello_description=description)
        with self.assertRaisesRegex(ValueError, "verified runner description"):
            self.session()
        self.assertEqual(self.opcodes(), [1])

    def test_catalog_mutation_after_selection_cannot_replace_loaded_bytes(self):
        self.write_runner(
            mutate_catalogs=[
                str(self.dist / path) for path in self.registry["catalogs"]
            ]
        )
        with self.session():
            loaded = [
                bytes.fromhex(record["bytes"])
                for record in self.records()
                if "loaded_path" in record
            ]
            self.assertEqual(loaded, [b"saxpy", b"relu"])
        self.assert_private_cleanup()

    def test_uuid_selection_resolves_current_ordinal_and_binds_open(self):
        inventory = _inventory(_TARGET)
        first = inventory["devices"][0]
        first["device_uuid"] = _UUID_B
        second = dict(first, index=1, device_uuid=_UUID_A)
        inventory["devices"] = [first, second]
        self.write_runner(inventory=inventory)
        with self.session(device_uuid=_UUID_A) as session:
            self.assertEqual(session.device.index, 1)
        opening = next(record for record in self.records() if record.get("opcode") == 2)
        payload = bytes.fromhex(opening["payload"])
        self.assertEqual(int.from_bytes(payload[:4], "little"), 1)
        self.assertIn(_UUID_A.encode(), payload)

    def test_bad_device_selection_or_geometry_never_opens_service(self):
        for options in (
            {"device_index": True},
            {"device_index": 0, "device_uuid": _UUID_A},
            {"device_uuid": "0" * 32},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.session(**options)
        self.assertEqual(self.records(), [])
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.session(device_uuid=_UUID_B)
        self.assertEqual(self.opcodes(), [])
        inventory = _inventory(_TARGET)
        inventory["devices"][0]["limits"]["max_threads_per_block"] = 64
        self.write_runner(inventory=inventory)
        with self.assertRaisesRegex(ValueError, "geometry"):
            self.session()
        self.assertEqual(self.opcodes(), [])

    def test_partial_load_failure_closes_before_private_files_are_removed(self):
        # Fatal backend responses terminate immediately; use a recoverable
        # argument failure to exercise graceful partial-setup CLOSE ordering.
        self.write_runner(fault="status1", at=5, occurrence=2)
        with self.assertRaises(ModuleServiceRemoteError):
            self.session()
        self.assertEqual(self.opcodes(), [1, 2, 5, 5, 11])
        self.assertNotIn(3, self.opcodes())
        self.assert_private_cleanup()

    def test_fatal_partial_load_failure_removes_private_files(self):
        self.write_runner(fault="status3", at=5, occurrence=2)
        with self.assertRaises(ModuleServiceRemoteError) as caught:
            self.session()
        self.assertFalse(caught.exception.recoverable)
        self.assertEqual(self.opcodes(), [1, 2, 5, 5])
        paths = [
            Path(record["loaded_path"])
            for record in self.records()
            if "loaded_path" in record
        ]
        self.assertEqual(len(paths), 1)
        self.assertFalse(paths[0].parent.exists())

    def test_foreign_or_stale_handles_and_undeclared_module_are_rejected(self):
        with self.session() as first, self.session() as second:
            module = first.module(_SAXPY)
            x, y, output = (first.allocate(4) for _ in range(3))
            alien = second.allocate(4)
            for operation in (
                lambda: first.read(alien, 0, 1),
                lambda: second.release(x),
                lambda: first.launch(second.module(_SAXPY), x, y, output, 1.0, 1),
                lambda: first.launch(module, alien, y, output, 1.0, 1),
                lambda: first.module(
                    ModuleRequest("missing", "cubin", "therock_module_saxpy")
                ),
            ):
                with self.assertRaises(ValueError):
                    operation()
            first.release(x)
            with self.assertRaises(ValueError):
                first.read(x, 0, 1)
            first.close()
            self.assertTrue(first.closed)
            self.assertFalse(second.closed)
            self.assertFalse(alien.closed)
            second.write(alien, 0, [7, 8, 9, 10])
            self.assertEqual(second.read(alien, 0, 4), array("f", [7, 8, 9, 10]))

    def test_fatal_failure_invalidates_facade_handles_and_cleans_private_files(self):
        self.write_runner(fault="status3", at=10)
        session = self.session()
        buffer = session.allocate(1)
        module = session.module(_SAXPY)
        paths = [
            Path(record["loaded_path"])
            for record in self.records()
            if "loaded_path" in record
        ]
        with self.assertRaises(ModuleServiceRemoteError):
            with session:
                session.synchronize()
        self.assertTrue(buffer.closed and module.closed and session.closed)
        self.assertTrue(
            all(not path.exists() and not path.parent.exists() for path in paths)
        )

    def test_close_failure_still_removes_private_files(self):
        self.write_runner(fault="status3", at=11)
        session = self.session()
        with self.assertRaises(ModuleServiceRemoteError):
            session.close()
        self.assertTrue(session.closed)
        paths = [
            Path(record["loaded_path"])
            for record in self.records()
            if "loaded_path" in record
        ]
        self.assertTrue(all(not path.exists() for path in paths))

    def test_last_runner_check_occurs_after_materializing_files(self):
        original = Path.write_bytes
        changed = False

        def write(path, data):
            nonlocal changed
            result = original(path, data)
            if path.name.startswith("payload-") and not changed:
                changed = True
                self.runner.write_text(
                    self.runner.read_text() + "\n# changed while staging\n"
                )
            return result

        with mock.patch.object(Path, "write_bytes", write), self.assertRaisesRegex(
            ValueError, "Runner SHA256 mismatch"
        ):
            self.session()
        self.assertTrue(changed)
        self.assertEqual(self.opcodes(), [])


if __name__ == "__main__":
    unittest.main()
