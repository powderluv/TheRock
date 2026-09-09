# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU-only tests using independent fragmented/faulting protocol workers."""

from array import array
import copy
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils import module_service
from _therock_utils.gpu_targets import parse_gpu_target
from _therock_utils.module_contract import runner_description, validation_contract
from _therock_utils.module_service import (
    Buffer,
    ModuleServiceError,
    ModuleServiceProtocolError,
    ModuleServiceRemoteError,
    ModuleServiceTimeoutError,
    NativeModuleSession,
)

_UUID = "0123456789abcdef0123456789abcdef"
_TARGET = parse_gpu_target("nvidia:cuda:sm_120")

# This worker deliberately does not import the client or its framing helpers.
# Requests are decoded with an independent implementation. Faults are emitted
# after a selected request so tests can also check live handle invalidation.
_WORKER = r"""
import json
import os
from pathlib import Path
import struct
import sys
import time

config = json.loads(Path(__file__).with_suffix('.json').read_text())
log = Path(__file__).with_suffix('.log')
def record(data):
    with log.open('a') as stream:
        stream.write(json.dumps(data) + '\n')
record({'argv': sys.argv[1:]})

def exact(size):
    result = bytearray()
    while len(result) < size:
        chunk = os.read(0, size - len(result))
        if not chunk:
            sys.exit(0)
        result.extend(chunk)
    return bytes(result)

def emit(data):
    if config.get('fragment'):
        for size in (1, 2, 3, 5, 7, 11, 17, 29):
            if not data:
                break
            os.write(1, data[:size])
            data = data[size:]
            time.sleep(0.0002)
    while data:
        sent = os.write(1, data)
        data = data[sent:]

def string(value):
    data = value.encode('utf-8')
    return struct.pack('<I', len(data)) + data

def take_string(payload, offset):
    size = struct.unpack_from('<I', payload, offset)[0]
    offset += 4
    return payload[offset:offset + size].decode('utf-8'), offset + size

buffers = {}
modules = {}
next_handle = 1
occurrences = {}
while True:
    header = exact(16)
    magic, version, opcode, request, length = struct.unpack('<4sHHII', header)
    assert magic == b'TRMS' and version == 1 and length <= 1048576
    payload = exact(length)
    record({'opcode': opcode, 'request': request, 'length': length, 'payload': payload[:512].hex()})
    occurrences[opcode] = occurrences.get(opcode, 0) + 1
    fault = config.get('fault') if opcode == config.get('at', 3) and occurrences[opcode] == config.get('occurrence', 1) else None
    if config.get('stderr') and opcode == 3:
        data = b'begin:' + b'x' * 180000 + b':diagnostic-end'
        while data:
            sent = os.write(2, data)
            data = data[sent:]
    if fault == 'stderr_stall':
        os.write(2, b'worker waiting forever')
        time.sleep(20)
    if fault == 'stall':
        time.sleep(20)
    status = 0
    error = b''
    result = b''
    if fault and fault.startswith('status'):
        status = int(fault[6:])
        error = b'request intentionally rejected'
    elif opcode == 1:
        result = string(json.dumps(config['description'], separators=(',', ':')))
    elif opcode == 2:
        pass
    elif opcode == 3:
        capacity, = struct.unpack('<I', payload)
        buffers[next_handle] = [0.0] * capacity
        result = struct.pack('<Q', next_handle)
        next_handle += 1
    elif opcode == 4:
        del buffers[struct.unpack('<Q', payload)[0]]
    elif opcode == 5:
        fmt, offset = take_string(payload, 0)
        symbol, offset = take_string(payload, offset)
        path, offset = take_string(payload, offset)
        assert offset == len(payload) and Path(path).is_file()
        modules[next_handle] = symbol
        result = struct.pack('<Q', next_handle)
        next_handle += 1
    elif opcode == 6:
        del modules[struct.unpack('<Q', payload)[0]]
    elif opcode == 7:
        handle, offset, count = struct.unpack_from('<QII', payload)
        values = struct.unpack_from('<' + 'f' * count, payload, 16)
        assert len(payload) == 16 + count * 4
        buffers[handle][offset:offset + count] = values
    elif opcode == 8:
        handle, offset, count = struct.unpack('<QII', payload)
        result = struct.pack('<' + 'f' * count, *buffers[handle][offset:offset + count])
    elif opcode == 9:
        module, x, y, output, alpha, count = struct.unpack('<QQQQfI', payload)
        for index in range(count):
            value = max(buffers[x][index], 0.0) if modules[module] == 'therock_module_relu' else alpha * buffers[x][index] + buffers[y][index]
            buffers[output][index] = struct.unpack('<f', struct.pack('<f', value))[0]
    elif opcode not in (10, 11):
        raise AssertionError(opcode)
    if fault == 'handlezero':
        result = struct.pack('<Q', 0)
    if fault == 'reusedhandle':
        result = struct.pack('<Q', 1)
    if fault == 'handlesize':
        result = b'bad'
    if fault == 'readsize':
        result = result[:-1]
    if fault == 'nonempty':
        result = b'extra'
    if fault == 'successerror':
        error = b'unexpected'
    if fault == 'errorutf8':
        status, error = 1, b'\xff'
        result = b''
    if fault == 'errorresult':
        status, error = 1, b'bad'
        result = b'unexpected'
    body = struct.pack('<II', status, len(error)) + error + result
    if fault == 'errorlength':
        body = struct.pack('<II', 1, 4097)
    reply = struct.pack('<4sHHII', b'FAIL' if fault == 'magic' else b'TRMS', 2 if fault == 'version' else 1, (opcode | 0x8000) + (1 if fault == 'opcode' else 0), request + (1 if fault == 'id' else 0), 1048577 if fault == 'huge' else (7 if fault == 'tiny' else len(body))) + body
    if fault == 'shortheader':
        emit(reply[:9])
        sys.exit(0)
    if fault == 'shortbody':
        emit(reply[:-1])
        sys.exit(0)
    if fault == 'stdoutnoise':
        reply = b'driver diagnostic\n' + reply
    if fault == 'trailing':
        reply += b'extra'
    emit(reply)
    if fault == 'inputstall':
        time.sleep(20)
    if opcode == 11:
        if fault == 'close_stall':
            time.sleep(20)
        if fault == 'close_extra':
            os.write(1, b'extra')
        sys.exit(7 if fault == 'close_failure' else 0)
"""


@unittest.skipUnless(os.name == "posix", "Native service transport requires POSIX")
class ModuleServiceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="therock-module-service-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.payload = self.root / "module.cubin"
        self.payload.write_bytes(b"trusted test module")
        self.counter = 0
        self.processes = []
        original = subprocess.Popen

        def spawn(*args, **kwargs):
            process = original(*args, **kwargs)
            self.processes.append(process)
            return process

        patch = mock.patch.object(module_service.subprocess, "Popen", side_effect=spawn)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self._check_reaped)

    def _check_reaped(self):
        for process in self.processes:
            if process.poll() is None:
                process.kill()
                process.wait()
                self.fail("Native client left its worker alive")

    def worker(self, **config):
        self.counter += 1
        runner = self.root / f"worker-{self.counter}.py"
        runner.write_text(f"#!{sys.executable}\n" + _WORKER)
        runner.chmod(0o755)
        config.setdefault("description", runner_description("nvidia").record())
        runner.with_suffix(".json").write_text(json.dumps(config))
        return runner

    def session(self, runner=None, target=_TARGET, **kwargs):
        session = NativeModuleSession(
            runner or self.worker(), target, 2, _UUID, **kwargs
        )
        self.addCleanup(session._terminate)
        return session

    def records(self, runner):
        return [
            json.loads(line)
            for line in runner.with_suffix(".log").read_text().splitlines()
        ]

    def requests(self, runner):
        return [record for record in self.records(runner) if "opcode" in record]

    def test_fragmented_session_roundtrip_and_typed_lifetime(self):
        runner = self.worker(fragment=True)
        with self.session(runner) as session:
            self.assertEqual(session.description, runner_description("nvidia"))
            x, y, output = (session.allocate(8) for _ in range(3))
            saxpy = session.load("cubin", "therock_module_saxpy", self.payload)
            relu = session.load("ptx", "therock_module_relu", self.payload)
            self.assertEqual(
                [x.handle, y.handle, output.handle, saxpy.handle, relu.handle],
                [1, 2, 3, 4, 5],
            )
            session.write(x, 0, [-2.0, -1.0, 2.0, 3.0])
            session.write(y, 0, array("f", [1.0] * 4))
            session.write(output, 0, [-77.0] * 8)
            session.launch(saxpy, x, y, output, 2.0, 4)
            session.launch(relu, output, y, x, 0.0, 4)
            session.synchronize()
            self.assertEqual(session.read(x, 0, 4), array("f", [0, 0, 5, 7]))
            self.assertEqual(session.read(output, 4, 4), array("f", [-77] * 4))
            session.release(output)
            session.unload(saxpy)
            self.assertTrue(output.closed)
            self.assertTrue(saxpy.closed)
            with self.assertRaises(ValueError):
                session.read(output, 0, 1)
            session.allocate(1)  # Released numbers may never be recycled.
        self.assertTrue(session.closed)
        self.assertTrue(x.closed)
        self.assertTrue(relu.closed)
        session.close()
        requests = self.requests(runner)
        self.assertEqual(self.records(runner)[0], {"argv": ["--serve"]})
        self.assertEqual(
            [record["request"] for record in requests],
            list(range(1, len(requests) + 1)),
        )
        self.assertEqual([record["opcode"] for record in requests[:2]], [1, 2])
        self.assertEqual(requests[-1]["opcode"], 11)
        contract = validation_contract()
        payload = bytes.fromhex(requests[1]["payload"])
        self.assertEqual(struct.unpack_from("<II", payload), (2, 0))
        offset = 8
        values = []
        for _ in range(3):
            size = struct.unpack_from("<I", payload, offset)[0]
            offset += 4
            values.append(payload[offset : offset + size].decode())
            offset += size
        self.assertEqual(values, ["sm_120", _UUID, contract.abi])
        self.assertEqual(struct.unpack_from("<I", payload, offset)[0], contract.version)
        self.assertEqual(payload[offset + 8 :].decode(), contract.sha256)

    def test_full_capacity_transfers_include_guard_region(self):
        with self.session() as session:
            buffer = session.allocate(65556)
            expected = array("f", (float(index % 17) for index in range(65556)))
            session.write(buffer, 0, expected)
            self.assertEqual(session.read(buffer, 0, 65556), expected)

    def test_intel_open_binds_expected_pci_identity(self):
        runner = self.worker(description=runner_description("intel").record())
        with self.session(runner, parse_gpu_target("intel:level-zero:xe2-b70")):
            pass
        payload = bytes.fromhex(self.requests(runner)[1]["payload"])
        self.assertEqual(struct.unpack_from("<II", payload), (2, 0xE223))
        self.assertEqual(payload[12:17], b"spirv")

    def test_selector_creation_failure_cannot_start_worker(self):
        runner = self.worker()
        with mock.patch.object(
            module_service.selectors, "DefaultSelector", side_effect=OSError("fd limit")
        ):
            with self.assertRaisesRegex(OSError, "fd limit"):
                self.session(runner)
        self.assertEqual(self.processes, [])
        self.assertFalse(runner.with_suffix(".log").exists())

    def test_spawn_failure_closes_acquired_selector(self):
        runner = self.worker()
        selector = module_service.selectors.DefaultSelector()
        self.addCleanup(selector.close)
        with mock.patch.object(
            module_service.selectors, "DefaultSelector", return_value=selector
        ), mock.patch.object(
            module_service.subprocess, "Popen", side_effect=OSError("exec failed")
        ):
            with self.assertRaisesRegex(OSError, "exec failed"):
                self.session(runner)
        self.assertIsNone(selector.get_map())
        self.assertEqual(self.processes, [])

    def test_bad_identity_rejected_before_process_creation(self):
        runner = self.worker()
        cases = [
            {"device": True},
            {"device": -1},
            {"device": 2**32},
            {"expected_device_uuid": ""},
            {"expected_device_uuid": "0" * 32},
            {"expected_device_uuid": _UUID.upper()},
            {"target": parse_gpu_target("nvidia:cuda:sm_90a")},
            {"target": parse_gpu_target("amd:hip:gfx1201:xnack+")},
            {"expected_device_id": 1},
            {
                "target": parse_gpu_target("intel:level-zero:xe2-b70"),
                "expected_device_id": 1,
            },
        ]
        for values in cases:
            with self.subTest(values=values):
                arguments = dict(
                    runner=runner, target=_TARGET, device=0, expected_device_uuid=_UUID
                )
                arguments.update(values)
                with self.assertRaises(ValueError):
                    NativeModuleSession(**arguments)
        self.assertEqual(self.processes, [])

    def test_incompatible_hello_never_opens_device(self):
        baseline = runner_description("nvidia").record()
        wrong_contract = copy.deepcopy(baseline)
        wrong_contract["contract_sha256"] = "a" * 64
        missing_capability = copy.deepcopy(baseline)
        missing_capability["capabilities"].remove("persistent-module-service")
        for description in (
            wrong_contract,
            missing_capability,
            runner_description("amd").record(),
        ):
            with self.subTest(description=description):
                runner = self.worker(description=description)
                with self.assertRaises(ValueError):
                    self.session(runner)
                self.assertEqual(
                    [record["opcode"] for record in self.requests(runner)], [1]
                )
                self.assertIsNotNone(self.processes[-1].poll())

    def test_verified_description_is_compared_before_open(self):
        description = runner_description("nvidia")
        changed = description.record()
        changed["capabilities"].remove("device-module-pipeline")
        runner = self.worker(description=changed)
        with self.assertRaisesRegex(ValueError, "verified runner description"):
            self.session(runner, expected_description=description)
        self.assertEqual([item["opcode"] for item in self.requests(runner)], [1])
        self.assertIsNotNone(self.processes[-1].poll())
        with self.session(expected_description=description):
            pass

    def test_local_handle_and_range_errors_send_no_requests(self):
        runner = self.worker()
        with self.session(runner) as session, self.session() as other:
            x, y, output = (session.allocate(8) for _ in range(3))
            module = session.load("cubin", "therock_module_saxpy", self.payload)
            alien = other.allocate(8)
            before = len(self.requests(runner))
            failures = [
                lambda: session.release(module),
                lambda: session.unload(x),
                lambda: session.read(alien, 0, 1),
                lambda: session.read(Buffer(session._owner, x.handle, 8), 0, 1),
                lambda: session.read(x, 7, 2),
                lambda: session.read(x, 0, 0),
                lambda: session.read(x, True, 1),
                lambda: session.write(x, 0, []),
                lambda: session.write(x, 0, "abc"),
                lambda: session.write(x, 0, [object()]),
                lambda: session.allocate(True),
                lambda: session.allocate(65557),
                lambda: session.load("spirv", "therock_module_saxpy", self.payload),
                lambda: session.load("cubin", "unknown", self.payload),
                lambda: session.launch(module, x, y, x, 1.0, 1),
                lambda: session.launch(module, x, y, y, 1.0, 1),
                lambda: session.launch(module, x, y, output, 1.0, 0),
                lambda: session.launch(module, x, y, output, 1.0, 9),
                lambda: session.launch(module, x, y, output, float("nan"), 1),
                lambda: session.launch(module, x, y, output, float("inf"), 1),
                lambda: session.launch(module, x, y, output, True, 1),
                lambda: session.launch(module, x, y, output, 1e100, 1),
                lambda: session.launch(module, x, y, output, 10**400, 1),
            ]
            for failure in failures:
                with self.subTest(failure=failures.index(failure)):
                    with self.assertRaises(ValueError):
                        failure()
            self.assertEqual(len(self.requests(runner)), before)
            session.synchronize()

    def test_recoverable_errors_keep_connection_and_handles_live(self):
        for status in (1, 4):
            with self.subTest(status=status):
                runner = self.worker(fault=f"status{status}", at=8)
                with self.session(runner) as session:
                    buffer = session.allocate(4)
                    session.write(buffer, 0, [1, 2, 3, 4])
                    with self.assertRaises(ModuleServiceRemoteError) as caught:
                        session.read(buffer, 0, 4)
                    self.assertEqual(caught.exception.status, status)
                    self.assertEqual(caught.exception.opcode, 8)
                    self.assertTrue(caught.exception.recoverable)
                    self.assertFalse(session.closed)
                    self.assertFalse(buffer.closed)
                    self.assertEqual(
                        session.read(buffer, 0, 4), array("f", [1, 2, 3, 4])
                    )

    def test_fatal_worker_errors_invalidate_and_reap(self):
        for status in (2, 3, 5):
            with self.subTest(status=status):
                session = self.session(self.worker(fault=f"status{status}", at=10))
                buffer = session.allocate(4)
                with self.assertRaises(ModuleServiceRemoteError) as caught:
                    session.synchronize()
                self.assertFalse(caught.exception.recoverable)
                self.assertTrue(session.closed)
                self.assertTrue(buffer.closed)
                self.assertIsNotNone(session._process.poll())
                with self.assertRaises(ModuleServiceError):
                    session.allocate(1)

    def test_malformed_response_invalidates_handles(self):
        for fault in (
            "magic",
            "version",
            "opcode",
            "id",
            "huge",
            "tiny",
            "shortheader",
            "shortbody",
            "trailing",
            "stdoutnoise",
            "status6",
            "errorutf8",
            "errorlength",
            "successerror",
            "errorresult",
            "nonempty",
        ):
            with self.subTest(fault=fault):
                session = self.session(self.worker(fault=fault, at=10))
                buffer = session.allocate(4)
                with self.assertRaises(ModuleServiceProtocolError):
                    session.synchronize()
                self.assertTrue(session.closed)
                self.assertTrue(buffer.closed)
                self.assertIsNotNone(session._process.poll())

    def test_zero_reused_and_wrong_sized_handles_are_fatal(self):
        for fault in ("handlezero", "reusedhandle", "handlesize"):
            with self.subTest(fault=fault):
                session = self.session(self.worker(fault=fault, occurrence=2))
                first = session.allocate(1)
                with self.assertRaises(ModuleServiceProtocolError):
                    session.allocate(1)
                self.assertTrue(first.closed)
                self.assertTrue(session.closed)

    def test_wrong_read_size_is_fatal(self):
        session = self.session(self.worker(fault="readsize", at=8))
        buffer = session.allocate(4)
        with self.assertRaises(ModuleServiceProtocolError):
            session.read(buffer, 0, 4)
        self.assertTrue(buffer.closed)

    def test_stderr_is_drained_concurrently_with_bounded_tail(self):
        runner = self.worker(stderr=True, fragment=True)
        with self.session(runner) as session:
            session.allocate(1)
        self.assertEqual(len(session._stderr), 65536)
        self.assertTrue(session.stderr_tail.endswith(":diagnostic-end"))
        self.assertNotIn("begin:", session.stderr_tail)

    def test_response_timeout_kills_worker_and_preserves_diagnostics(self):
        session = self.session(self.worker(fault="stderr_stall", at=10))
        buffer = session.allocate(1)
        start = time.monotonic()
        with mock.patch.object(module_service, "_REQUEST_TIMEOUT_SECONDS", 0.2):
            with self.assertRaises(ModuleServiceTimeoutError) as caught:
                session.synchronize()
        self.assertLess(time.monotonic() - start, 2.0)
        self.assertIn("worker waiting forever", caught.exception.stderr_tail)
        self.assertTrue(buffer.closed)
        self.assertIsNotNone(session._process.poll())

    def test_blocked_request_write_is_bounded(self):
        session = self.session(self.worker(fault="inputstall"))
        buffer = session.allocate(65556)
        with mock.patch.object(module_service, "_REQUEST_TIMEOUT_SECONDS", 0.2):
            with self.assertRaises(ModuleServiceTimeoutError):
                session.write(buffer, 0, array("f", [0]) * 65556)
        self.assertTrue(buffer.closed)
        self.assertIsNotNone(session._process.poll())

    def test_close_requires_worker_exit_and_no_extra_output(self):
        for fault in ("close_stall", "close_extra", "close_failure"):
            with self.subTest(fault=fault):
                session = self.session(self.worker(fault=fault, at=11))
                buffer = session.allocate(1)
                expected = (
                    ModuleServiceTimeoutError
                    if fault == "close_stall"
                    else ModuleServiceProtocolError
                )
                with mock.patch.object(module_service, "_REQUEST_TIMEOUT_SECONDS", 0.2):
                    with self.assertRaises(expected):
                        session.close()
                self.assertTrue(buffer.closed)
                self.assertTrue(session.closed)
                self.assertIsNotNone(session._process.poll())

    def test_body_exception_survives_failed_close(self):
        with self.assertRaisesRegex(ValueError, "body failed"):
            with self.session(self.worker(fault="status3", at=11)):
                raise ValueError("body failed")


if __name__ == "__main__":
    unittest.main()
