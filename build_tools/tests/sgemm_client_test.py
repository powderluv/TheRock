# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU-only SGEMM client negotiation, matrix framing, and ownership checks."""

from array import array
import copy
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

from _therock_utils import module_service
from _therock_utils.gpu_targets import parse_gpu_target
from _therock_utils.module_contract import runner_description
from _therock_utils.module_service import (
    ModuleServiceProtocolError,
    ModuleServiceRemoteError,
    ModuleServiceTimeoutError,
    NativeModuleSession,
)
from _therock_utils.sgemm_contract import (
    SGEMM_ABI,
    SGEMM_VERSION,
    sgemm_contract_sha256,
)
from module_service_test import _WORKER
from sgemm_contract_test import provider
from therock_multi_vendor import PackedModuleSession, SgemmProviderInfo

_UUID = "0123456789abcdef0123456789abcdef"
# Extend the existing independent wire worker. Its framing/fault machinery does
# not import production helpers; SGEMM math below uses decoded matrix indices.
_SGEMM_CASES = r"""
    elif opcode == 12:
        expected_provider, offset = take_string(payload, 0)
        abi, offset = take_string(payload, offset)
        version, = struct.unpack_from('<I', payload, offset)
        digest, offset = take_string(payload, offset + 4)
        assert offset == len(payload)
        assert expected_provider == config['provider']['provider']
        assert abi == config['provider']['abi']
        assert version == config['provider']['version']
        assert digest == config['provider']['contract_sha256']
        result = string(config.get('provider_text', json.dumps(config['provider'])))
    elif opcode == 13:
        a, b, c, ao, bo, co, m, n, k, lda, ldb, ldc, alpha, beta = struct.unpack('<QQQ9I2f', payload)
        for column in range(n):
            for row in range(m):
                total = sum(buffers[a][ao + inner * lda + row] * buffers[b][bo + column * ldb + inner] for inner in range(k))
                index = co + column * ldc + row
                value = alpha * total + beta * buffers[c][index]
                buffers[c][index] = struct.unpack('<f', struct.pack('<f', value))[0]
"""
_SGEMM_WORKER = _WORKER.replace(
    "    elif opcode not in (10, 11):",
    _SGEMM_CASES + "    elif opcode not in (10, 11):",
)


@unittest.skipUnless(os.name == "posix", "Native service transport requires POSIX")
class SgemmClientTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="therock-sgemm-client-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.counter = 0
        self.sessions = []
        self.addCleanup(self._close_sessions)

    def _close_sessions(self):
        for session in self.sessions:
            session._terminate()
            self.assertIsNotNone(session._process.poll())

    def session(self, *, vendor="nvidia", enabled=True, **config):
        self.counter += 1
        path = self.root / f"worker-{self.counter}.py"
        path.write_text(f"#!{sys.executable}\n" + _SGEMM_WORKER)
        path.chmod(0o755)
        config.setdefault(
            "description", runner_description(vendor, enable_sgemm=enabled).record()
        )
        config.setdefault("provider", provider(vendor).record())
        path.with_suffix(".json").write_text(json.dumps(config))
        target = parse_gpu_target(
            "nvidia:cuda:sm_120" if vendor == "nvidia" else "amd:hip:gfx1201"
        )
        session = NativeModuleSession(path, target, 0, _UUID)
        self.sessions.append(session)
        return session, path

    def requests(self, path):
        return [
            record
            for line in path.with_suffix(".log").read_text().splitlines()
            if "opcode" in (record := json.loads(line))
        ]

    def options(self, **updates):
        values = dict(
            m=2,
            n=3,
            k=2,
            lda=3,
            ldb=4,
            ldc=5,
            a_offset=1,
            b_offset=2,
            c_offset=3,
            alpha=2.0,
            beta=0.5,
        )
        values.update(updates)
        return values

    def buffers(self, session, capacity=64):
        return tuple(session.allocate(capacity) for _ in range(3))

    def test_default_worker_keeps_vector_calls_and_rejects_provider_locally(self):
        session, path = self.session(enabled=False)
        a, b, c = self.buffers(session)
        payload = self.root / "saxpy.cubin"
        payload.write_bytes(b"inert trusted test module")
        kernel = session.load("cubin", "therock_module_saxpy", payload)
        session.write(a, 0, [1, 2])
        session.write(b, 0, [3, 4])
        session.launch(kernel, a, b, c, 2, 2)
        self.assertEqual(session.read(c, 0, 2), array("f", [5, 8]))
        with self.assertRaises(ValueError):
            session.sgemm_provider()
        with self.assertRaises(ValueError):
            session.sgemm(a, b, c, **self.options())
        session.synchronize()
        self.assertFalse(session.closed)
        self.assertNotIn(12, [record["opcode"] for record in self.requests(path)])
        self.assertNotIn(13, [record["opcode"] for record in self.requests(path)])

    def test_provider_negotiates_exact_identity_contract_and_caches_observed_version(
        self,
    ):
        for vendor in ("amd", "nvidia"):
            with self.subTest(vendor=vendor):
                session, path = self.session(vendor=vendor, fragment=True)
                info = session.sgemm_provider()
                self.assertIsInstance(info, SgemmProviderInfo)
                self.assertEqual(info, provider(vendor))
                self.assertIs(session.sgemm_provider(), info)
                request = [
                    record for record in self.requests(path) if record["opcode"] == 12
                ]
                self.assertEqual(len(request), 1)

                def string(value):
                    data = value.encode()
                    return struct.pack("<I", len(data)) + data

                expected = (
                    string(info.provider)
                    + string(SGEMM_ABI)
                    + struct.pack("<I", SGEMM_VERSION)
                    + string(sgemm_contract_sha256())
                )
                self.assertEqual(bytes.fromhex(request[0]["payload"]), expected)

    def test_padded_offset_matrices_preserve_guards_and_use_exact_wire_layout(self):
        session, path = self.session(fragment=True)
        a, b, c = self.buffers(session)
        av, bv, cv = ([-77.0] * 64 for _ in range(3))
        # A=[[1,3],[2,4]], B=[[2,4,6],[3,5,7]], all column-major with gaps.
        av[1:3], av[4:6] = [1, 2], [3, 4]
        bv[2:4], bv[6:8], bv[10:12] = [2, 3], [4, 5], [6, 7]
        for index in (3, 4, 8, 9, 13, 14):
            cv[index] = 8.0
        for buffer, values in ((a, av), (b, bv), (c, cv)):
            session.write(buffer, 0, values)
        session.sgemm(a, b, c, **self.options())
        expected = list(cv)
        for index, value in zip((3, 4, 8, 9, 13, 14), (26, 36, 42, 60, 58, 84)):
            expected[index] = value
        self.assertEqual(session.read(c, 0, 64), array("f", expected))
        self.assertEqual(session.read(a, 0, 64), array("f", av))
        self.assertEqual(session.read(b, 0, 64), array("f", bv))
        session.sgemm(a, b, c, **self.options(alpha=0.0, beta=1.0))
        self.assertEqual(session.read(c, 0, 64), array("f", expected))
        requests = self.requests(path)
        self.assertEqual(sum(item["opcode"] == 12 for item in requests), 1)
        raw = bytes.fromhex(
            next(item for item in requests if item["opcode"] == 13)["payload"]
        )
        self.assertEqual(len(raw), 68)
        self.assertEqual(
            struct.unpack("<QQQ9I2f", raw),
            (a.handle, b.handle, c.handle, 1, 2, 3, 2, 3, 2, 3, 4, 5, 2.0, 0.5),
        )

    def test_input_alias_is_allowed_and_output_alias_rejected_before_negotiation(self):
        session, path = self.session()
        a, b, c = self.buffers(session)
        for arguments in ((a, b, a), (a, b, b)):
            with self.assertRaises(ValueError):
                session.sgemm(*arguments, **self.options())
        self.assertNotIn(12, [item["opcode"] for item in self.requests(path)])
        session.write(a, 0, [1, 2, 3, 4])
        session.sgemm(a, a, c, m=2, n=2, k=2, lda=2, ldb=2, ldc=2)
        self.assertEqual(session.read(c, 0, 4), array("f", [7, 10, 15, 22]))

    def test_invalid_matrices_and_scalars_do_not_initialize_provider(self):
        session, path = self.session()
        a, b, c = self.buffers(session)
        for updates in (
            {"m": 0},
            {"m": True},
            {"m": 257},
            {"n": -1},
            {"n": 2.0},
            {"k": 0},
            {"lda": 1},
            {"ldb": 1},
            {"ldc": 1},
            {"lda": -1},
            {"ldc": True},
            {"a_offset": -1},
            {"b_offset": True},
            {"c_offset": 64},
            {"a_offset": 61},
            {"ldb": 100},
            {"lda": 2**64},
            {"k": 1, "lda": 65557},
            {"n": 1, "ldb": 65557},
            {"n": 1, "ldc": 65557},
            {"alpha": True},
            {"alpha": float("nan")},
            {"alpha": float("inf")},
            {"alpha": 1e100},
            {"alpha": 10**1000},
            {"beta": "1"},
            {"beta": False},
            {"beta": float("nan")},
            {"beta": 1e100},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                session.sgemm(a, b, c, **self.options(**updates))
        self.assertNotIn(12, [item["opcode"] for item in self.requests(path)])
        self.assertNotIn(13, [item["opcode"] for item in self.requests(path)])
        session.synchronize()

    def test_foreign_stale_and_wrong_type_buffers_fail_before_provider_query(self):
        session, path = self.session()
        a, b, c = self.buffers(session)
        other, _ = self.session()
        foreign = other.allocate(64)
        stale = session.allocate(64)
        session.release(stale)
        payload = self.root / "relu.cubin"
        payload.write_bytes(b"inert")
        module = session.load("cubin", "therock_module_relu", payload)
        for invalid in (foreign, stale, module, a.handle, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                session.sgemm(invalid, b, c, **self.options())
        self.assertNotIn(12, [item["opcode"] for item in self.requests(path)])
        self.assertFalse(other.closed)

    def test_malformed_or_mismatched_provider_reply_invalidates_all_handles(self):
        baseline = provider().record()
        variants = []
        for key, value in (
            ("vendor", "amd"),
            ("provider", "rocblas"),
            ("version", 2),
            ("contract_sha256", "0" * 64),
            ("library_version", ""),
            ("capabilities", []),
            ("extra", "bad"),
        ):
            changed = copy.deepcopy(baseline)
            changed[key] = value
            variants.append(json.dumps(changed))
        variants += [
            "{",
            json.dumps(baseline).replace(
                '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
            ),
        ]
        for value in variants:
            with self.subTest(value=value):
                session, path = self.session(provider_text=value)
                buffers = self.buffers(session)
                with self.assertRaises(ModuleServiceProtocolError):
                    session.sgemm(*buffers, **self.options())
                self.assertTrue(session.closed)
                self.assertTrue(all(buffer.closed for buffer in buffers))
                self.assertNotIn(13, [item["opcode"] for item in self.requests(path)])

    def test_recoverable_provider_error_does_not_cache_or_close_worker(self):
        for status in (1, 4):
            with self.subTest(status=status):
                session, path = self.session(fault=f"status{status}", at=12)
                with self.assertRaises(ModuleServiceRemoteError) as caught:
                    session.sgemm_provider()
                self.assertTrue(caught.exception.recoverable)
                self.assertFalse(session.closed)
                session.allocate(1)
                self.assertEqual(session.sgemm_provider(), provider())
                self.assertEqual(
                    sum(item["opcode"] == 12 for item in self.requests(path)), 2
                )

    def test_fatal_provider_or_sgemm_error_closes_worker_and_handles(self):
        for opcode, status in ((12, 5), (13, 3)):
            with self.subTest(opcode=opcode):
                session, _ = self.session(fault=f"status{status}", at=opcode)
                buffers = self.buffers(session)
                with self.assertRaises(ModuleServiceRemoteError) as caught:
                    session.sgemm(*buffers, **self.options())
                self.assertFalse(caught.exception.recoverable)
                self.assertTrue(session.closed)
                self.assertTrue(all(buffer.closed for buffer in buffers))

    def test_provider_and_sgemm_timeouts_reap_worker(self):
        for opcode in (12, 13):
            with self.subTest(opcode=opcode):
                session, _ = self.session(fault="stall", at=opcode)
                buffers = self.buffers(session)
                with mock.patch.object(
                    module_service, "_REQUEST_TIMEOUT_SECONDS", 0.05
                ):
                    with self.assertRaises(ModuleServiceTimeoutError):
                        session.sgemm(*buffers, **self.options())
                self.assertTrue(session.closed)
                self.assertIsNotNone(session._process.poll())

    def test_public_facade_preserves_handle_identity_and_matrix_keywords(self):
        session, _ = self.session()
        facade = PackedModuleSession.__new__(PackedModuleSession)
        facade._native = session
        a, b, c = (facade.allocate(64) for _ in range(3))
        facade.write(a, 1, [2])
        facade.write(b, 2, [3])
        facade.write(c, 3, [4])
        facade.sgemm(
            a,
            b,
            c,
            m=1,
            n=1,
            k=1,
            lda=3,
            ldb=4,
            ldc=5,
            a_offset=1,
            b_offset=2,
            c_offset=3,
            alpha=2,
            beta=0.5,
        )
        self.assertEqual(facade.read(c, 3, 1), array("f", [14]))
        self.assertIs(facade.sgemm_provider(), session.sgemm_provider())


if __name__ == "__main__":
    unittest.main()
