# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU tests of verified provider-only sessions with independent wire workers."""

from array import array
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "build_tools"))
sys.path.insert(0, str(REPOSITORY / "build_tools/tests"))
sys.path.insert(0, str(REPOSITORY / "rocm-systems/shared/kpack/python"))

import therock_multi_vendor
from _therock_utils import module_service
from _therock_utils.gpu_targets import parse_gpu_target
from _therock_utils.module_contract import runner_description, validation_contract
from _therock_utils.payload_catalog import PayloadInput, create_pack
from dispatch_multi_vendor_modules_test import _inventory
from packed_module_session_test import _QUERY
from sgemm_client_test import _SGEMM_WORKER
from sgemm_contract_test import provider
from therock_multi_vendor import (
    ModuleRequest,
    ModuleServiceError,
    ModuleServiceProtocolError,
    ModuleServiceRemoteError,
    SgemmSession,
    open_session,
    open_sgemm_session,
)

_TARGET = "nvidia:cuda:sm_120"
_UUID_A = "0123456789abcdef0123456789abcdef"
_UUID_B = "fedcba9876543210fedcba9876543210"
_SAXPY = ModuleRequest("validation/saxpy", "cubin", "therock_module_saxpy")


@unittest.skipUnless(os.name == "posix", "Native service transport requires POSIX")
class SgemmSessionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="therock-sgemm-session-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dist = self.root / "dist"
        self.target = parse_gpu_target(_TARGET)
        self.runner = self.dist / "bin" / self.target.slug / "therock_module_validation"
        self.runner.parent.mkdir(parents=True)
        self.catalog = create_pack(
            self.dist / "share/therock/packs/saxpy",
            "validation-saxpy",
            (
                PayloadInput(
                    _SAXPY.module,
                    _TARGET,
                    _SAXPY.payload_type,
                    (_SAXPY.entry_point,),
                    b"saxpy",
                    validation_contract(),
                ),
            ),
        )
        self.registry_path = self.dist / "share/therock/packs/runners.json"
        self.registry = {
            "schema_version": 1,
            "kind": "native-runner-registry",
            "runners": [],
            "catalogs": [self.catalog.relative_to(self.dist).as_posix()],
        }
        self.config = {
            "description": runner_description("nvidia", enable_sgemm=True).record(),
            "inventory": _inventory(_TARGET),
            "provider": provider().record(),
        }
        self.write_runner()
        self.processes = []
        original = module_service.subprocess.Popen

        def spawn(*args, **kwargs):
            process = original(*args, **kwargs)
            self.processes.append(process)
            return process

        patch = mock.patch.object(module_service.subprocess, "Popen", side_effect=spawn)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self.assert_reaped)

    def assert_reaped(self):
        live = [process for process in self.processes if process.poll() is None]
        for process in live:
            process.kill()
            process.wait()
        self.assertEqual(live, [], "Provider session left a worker alive")

    def write_runner(self, *, target_id=_TARGET, **changes):
        self.config.update(changes)
        body = _SGEMM_WORKER.replace(
            "config = json.loads(Path(__file__).with_suffix('.json').read_text())",
            "config = CONFIG",
        )
        self.runner.write_text(
            f"#!{sys.executable}\nCONFIG = {self.config!r}\n" + _QUERY + body
        )
        self.runner.chmod(0o755)
        self.registry["runners"] = [
            {
                "target": target_id,
                "path": self.runner.relative_to(self.dist).as_posix(),
                "sha256": hashlib.sha256(self.runner.read_bytes()).hexdigest(),
                "description": self.config["description"],
            }
        ]
        self.write_registry()

    def write_registry(self):
        self.registry_path.write_text(json.dumps(self.registry))

    def records(self):
        path = self.runner.with_suffix(".log")
        return (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )

    def clear_records(self):
        self.runner.with_suffix(".log").unlink(missing_ok=True)

    def opcodes(self):
        return [record["opcode"] for record in self.records() if "opcode" in record]

    def session(self, target_id=_TARGET, **options):
        session = open_sgemm_session(self.dist, target_id, **options)
        self.addCleanup(session.close)
        return session

    def test_open_verifies_runner_and_negotiates_before_return_without_payloads(self):
        original_open = Path.open

        def open_path(path, *args, **kwargs):
            if path.is_relative_to(self.catalog.parent):
                self.fail("Provider-only opening read a kernel catalog or archive")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", open_path), mock.patch.object(
            therock_multi_vendor.tempfile,
            "TemporaryDirectory",
            side_effect=AssertionError("Provider-only opening materialized payloads"),
        ):
            with self.session() as session:
                self.assertIsInstance(session, SgemmSession)
                self.assertEqual(session.target, self.target)
                self.assertEqual(session.device.device_uuid, _UUID_A)
                self.assertEqual(
                    session.runner_sha256, self.registry["runners"][0]["sha256"]
                )
                self.assertEqual(self.opcodes(), [1, 2, 12])
                observed = session.sgemm_provider()
                self.assertEqual(observed, provider())
                self.assertIs(session.sgemm_provider(), observed)
                self.assertEqual(self.opcodes(), [1, 2, 12])
                for name in ("module", "load", "launch"):
                    self.assertFalse(hasattr(session, name))
        self.assertTrue(session.closed)
        self.assertEqual(self.opcodes(), [1, 2, 12, 11])
        self.assertEqual(
            [record["argv"] for record in self.records() if "argv" in record][:2],
            [["--describe-contract"], ["--list-devices"]],
        )

    def test_missing_and_corrupt_catalogs_do_not_block_provider_but_block_packed_open(
        self,
    ):
        original = self.catalog.read_bytes()
        for content in (None, b"{broken catalog"):
            with self.subTest(content=content):
                if content is None:
                    self.catalog.unlink()
                else:
                    self.catalog.write_bytes(content)
                self.clear_records()
                with self.session():
                    self.assertEqual(self.opcodes(), [1, 2, 12])
                self.clear_records()
                with self.assertRaises((ValueError, OSError)):
                    open_session(self.dist, _TARGET, (_SAXPY,))
                self.assertEqual(self.records(), [])
                self.catalog.write_bytes(original)

    def test_corrupt_pack_bytes_do_not_block_provider_but_block_packed_open(self):
        document = json.loads(self.catalog.read_text())
        archive = self.catalog.parent / document["packs"][0]["path"]
        archive.write_bytes(archive.read_bytes() + b"changed")
        with self.session():
            self.assertNotIn(5, self.opcodes())
        self.clear_records()
        with self.assertRaisesRegex(ValueError, "Pack SHA256 mismatch"):
            open_session(self.dist, _TARGET, (_SAXPY,))
        self.assertEqual(self.records(), [])

    def test_registry_metadata_remains_strict_even_when_catalogs_are_not_read(self):
        for catalog in ("../outside.json", "/tmp/catalog.json", "packs//catalog.json"):
            with self.subTest(catalog=catalog):
                self.registry["catalogs"] = [catalog]
                self.write_registry()
                with self.assertRaises(ValueError):
                    self.session()
                self.assertEqual(self.records(), [])

    def test_sgemm_off_workers_including_intel_fail_before_device_queries(self):
        for vendor, target in (
            ("nvidia", _TARGET),
            ("intel", "intel:level-zero:xe2-b70"),
        ):
            with self.subTest(vendor=vendor):
                self.write_runner(
                    target_id=target,
                    description=runner_description(vendor, enable_sgemm=False).record(),
                )
                with self.assertRaisesRegex(ValueError, "blas-sgemm-f32-nn-v1"):
                    self.session(target)
                self.assertEqual(self.records(), [])

    def test_amd_provider_uses_its_exact_worker_without_a_matching_kernel_pack(self):
        target_id = "amd:hip:gfx1201"
        self.write_runner(
            target_id=target_id,
            description=runner_description("amd", enable_sgemm=True).record(),
            inventory=_inventory(target_id),
            provider=provider("amd").record(),
        )
        with self.session(target_id) as session:
            self.assertEqual(session.target.canonical_id, target_id)
            self.assertEqual(session.sgemm_provider(), provider("amd"))
            self.assertEqual(self.opcodes(), [1, 2, 12])
        self.clear_records()
        with self.assertRaises(ValueError):
            self.session(target_id, required_capabilities=("blas-provider-cublas-v1",))
        self.assertEqual(self.records(), [])

    def test_intel_onemkl_provider_negotiates_without_loading_spirv_payloads(self):
        target_id = "intel:level-zero:xe2-b70"
        self.write_runner(
            target_id=target_id,
            description=runner_description("intel", enable_sgemm=True).record(),
            inventory=_inventory(target_id),
            provider=provider("intel").record(),
        )
        with self.session(target_id) as session:
            self.assertEqual(session.target.canonical_id, target_id)
            self.assertEqual(session.device.device_id, 0xE223)
            self.assertEqual(session.sgemm_provider(), provider("intel"))
            self.assertEqual(session.sgemm_provider().provider, "onemkl")
            self.assertEqual(self.opcodes(), [1, 2, 12])
            a, b, c = (session.allocate(4) for _ in range(3))
            session.write(a, 0, [1, 2, 3, 4])
            session.write(b, 0, [5, 6, 7, 8])
            session.sgemm(a, b, c, m=2, n=2, k=2, lda=2, ldb=2, ldc=2)
            self.assertEqual(session.read(c, 0, 4), array("f", [23, 34, 31, 46]))
            self.assertNotIn(5, self.opcodes())
            self.assertNotIn(9, self.opcodes())
        self.clear_records()
        for capability in ("blas-provider-rocblas-v1", "blas-provider-cublas-v1"):
            with self.subTest(capability=capability), self.assertRaises(ValueError):
                self.session(target_id, required_capabilities=(capability,))
            self.assertEqual(self.records(), [])

    def test_persistent_service_capability_is_mandatory_before_queries(self):
        description = copy.deepcopy(self.config["description"])
        description["capabilities"].remove("persistent-module-service")
        self.write_runner(description=description)
        with self.assertRaisesRegex(ValueError, "persistent-module-service"):
            self.session()
        self.assertEqual(self.records(), [])

    def test_invalid_capabilities_targets_and_selectors_fail_before_queries(self):
        for options in (
            {"required_capabilities": ("unknown",)},
            {"required_capabilities": ("ordered-copy", "ordered-copy")},
            {"device_index": True},
            {"device_index": -1},
            {"device_index": 0, "device_uuid": _UUID_A},
            {"device_uuid": "0" * 32},
            {"expected_device_id": 0xE223},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.session(**options)
        for target in (
            "nvidia:cuda:sm_90",
            "nvidia:cuda:sm_120a",
            "amd:hip:gfx1201:xnack+",
            42,
        ):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.session(target)
        self.assertEqual(self.records(), [])

    def test_provider_open_does_not_require_packed_vector_launch_geometry(self):
        inventory = _inventory(_TARGET)
        inventory["devices"][0]["limits"]["max_threads_per_block"] = 64
        self.write_runner(inventory=inventory)
        with self.session():
            self.assertEqual(self.opcodes(), [1, 2, 12])
        self.clear_records()
        with self.assertRaisesRegex(ValueError, "geometry"):
            open_session(self.dist, _TARGET, (_SAXPY,))
        self.assertEqual(self.opcodes(), [])

    def test_runner_hash_is_checked_before_queries_and_after_discovery(self):
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

    def test_compiled_and_hello_description_must_match_verified_registry(self):
        description = copy.deepcopy(self.config["description"])
        description["capabilities"].remove("device-module-pipeline")
        self.write_runner(hello_description=description)
        with self.assertRaisesRegex(ValueError, "verified runner description"):
            self.session()
        self.assertEqual(self.opcodes(), [1])
        self.assert_reaped()
        self.clear_records()
        self.registry["runners"][0]["description"] = description
        self.write_registry()
        with self.assertRaisesRegex(ValueError, "differs from registry description"):
            self.session()
        self.assertEqual(self.opcodes(), [])
        self.assertEqual(
            [record["argv"] for record in self.records()], [["--describe-contract"]]
        )

    def test_uuid_selection_binds_the_observed_ordinal_in_open(self):
        inventory = _inventory(_TARGET)
        first = inventory["devices"][0]
        first["device_uuid"] = _UUID_B
        inventory["devices"].append(dict(first, index=1, device_uuid=_UUID_A))
        self.write_runner(inventory=inventory)
        with self.session(device_uuid=_UUID_A) as session:
            self.assertEqual(session.device.index, 1)
            self.assertEqual(session.device.device_uuid, _UUID_A)
        opening = next(record for record in self.records() if record.get("opcode") == 2)
        payload = bytes.fromhex(opening["payload"])
        self.assertEqual(struct.unpack_from("<I", payload)[0], 1)
        self.assertIn(_UUID_A.encode(), payload)

    def test_unavailable_uuid_and_native_open_failure_never_return_a_session(self):
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.session(device_uuid=_UUID_B)
        self.assertEqual(self.opcodes(), [])
        self.clear_records()
        self.write_runner(fault="status3", at=2)
        with self.assertRaises(ModuleServiceRemoteError) as caught:
            self.session()
        self.assertFalse(caught.exception.recoverable)
        self.assertEqual(self.opcodes(), [1, 2])
        self.assert_reaped()

    def test_provider_negotiation_failures_reap_before_open_returns(self):
        for status in (1, 3, 5):
            with self.subTest(status=status):
                self.clear_records()
                self.write_runner(fault=f"status{status}", at=12)
                with self.assertRaises(ModuleServiceRemoteError) as caught:
                    self.session()
                self.assertEqual(caught.exception.recoverable, status == 1)
                self.assertEqual(
                    self.opcodes(), [1, 2, 12, 11] if status == 1 else [1, 2, 12]
                )
                self.assert_reaped()

    def test_bad_provider_contract_version_and_malformed_json_reap_before_return(self):
        bad_version = dict(provider().record(), version=2)
        wrong_provider = dict(provider().record(), provider="rocblas")
        for response in (
            "not-json",
            json.dumps(bad_version),
            json.dumps(wrong_provider),
        ):
            with self.subTest(response=response):
                self.clear_records()
                self.write_runner(provider_text=response)
                with self.assertRaises(ModuleServiceProtocolError):
                    self.session()
                self.assertEqual(self.opcodes(), [1, 2, 12])
                self.assert_reaped()

    def test_actual_matrix_arithmetic_buffers_guards_and_reuse_need_no_modules(self):
        with self.session() as session:
            a, b, c = (session.allocate(12) for _ in range(3))
            session.write(a, 0, [-77, 1, 4, 2, 5, 3, 6, -77, -77, -77, -77, -77])
            session.write(b, 0, [-77, 7, 9, 11, 8, 10, 12, -77, -77, -77, -77, -77])
            session.write(c, 0, [-77] * 12)
            options = dict(
                m=2, n=2, k=3, lda=2, ldb=3, ldc=3, a_offset=1, b_offset=1, c_offset=2
            )
            session.sgemm(a, b, c, **options)
            expected = array(
                "f", [-77, -77, 58, 139, -77, 64, 154, -77, -77, -77, -77, -77]
            )
            self.assertEqual(session.read(c, 0, 12), expected)
            session.sgemm(a, b, c, alpha=-1, beta=2, **options)
            session.synchronize()
            self.assertEqual(session.read(c, 0, 12), expected)
            self.assertEqual(session.read(a, 1, 6), array("f", [1, 4, 2, 5, 3, 6]))
            self.assertEqual(session.read(b, 1, 6), array("f", [7, 9, 11, 8, 10, 12]))
            session.release(b)
            self.assertTrue(b.closed)
        self.assertTrue(a.closed and c.closed and session.closed)
        self.assertEqual(self.opcodes().count(12), 1)
        self.assertEqual(self.opcodes().count(13), 2)
        self.assertNotIn(5, self.opcodes())
        self.assertNotIn(9, self.opcodes())

    def test_owner_checks_and_close_leave_another_session_usable(self):
        with self.session() as first, self.session() as second:
            a, b, c = (first.allocate(4) for _ in range(3))
            other = second.allocate(4)
            options = dict(m=2, n=2, k=2, lda=2, ldb=2, ldc=2)
            for operation in (
                lambda: first.sgemm(other, b, c, **options),
                lambda: first.read(other, 0, 1),
                lambda: second.release(a),
                lambda: first.sgemm(a, b, a, **options),
            ):
                with self.assertRaises(ValueError):
                    operation()
            first.release(a)
            with self.assertRaises(ValueError):
                first.sgemm(a, b, c, **options)
            self.assertNotIn(13, self.opcodes())
            first.close()
            first.close()
            self.assertTrue(first.closed and b.closed and c.closed)
            for operation in (
                lambda: first.allocate(1),
                first.sgemm_provider,
                first.synchronize,
            ):
                with self.assertRaises(ModuleServiceError):
                    operation()
            second.write(other, 0, [1, 2, 3, 4])
            self.assertEqual(second.read(other, 0, 4), array("f", [1, 2, 3, 4]))
            self.assertFalse(second.closed or other.closed)

    def test_fatal_sgemm_failure_invalidates_buffers_and_reaps_worker(self):
        self.write_runner(fault="status3", at=13)
        session = self.session()
        buffers = tuple(session.allocate(4) for _ in range(3))
        with self.assertRaises(ModuleServiceRemoteError):
            session.sgemm(*buffers, m=2, n=2, k=2, lda=2, ldb=2, ldc=2)
        self.assertTrue(session.closed)
        self.assertTrue(all(buffer.closed for buffer in buffers))
        self.assert_reaped()


if __name__ == "__main__":
    unittest.main()
