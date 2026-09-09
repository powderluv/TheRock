# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU tests of the real native worker framing/server and deferred ownership."""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest


MAGIC = 0x534D5254
Command = tuple[int, bytes]


def u32(value: int) -> bytes:
    return struct.pack("<I", value)


def u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def string(value: str | bytes) -> bytes:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return u32(len(data)) + data


def frame(opcode: int, request_id: int, payload: bytes = b"") -> bytes:
    return (
        struct.pack("<4I", MAGIC, 1 | (opcode << 16), request_id, len(payload))
        + payload
    )


def write(handle: int, offset: int, values: list[float]) -> bytes:
    return (
        u64(handle)
        + u32(offset)
        + u32(len(values))
        + struct.pack(f"<{len(values)}f", *values)
    )


def read(handle: int, offset: int, count: int) -> bytes:
    return u64(handle) + u32(offset) + u32(count)


def launch(
    module: int, x: int, y: int, output: int, alpha: float = 1.75, count: int = 4
) -> bytes:
    return struct.pack("<4QfI", module, x, y, output, alpha, count)


def load(
    symbol: str = "therock_module_saxpy",
    *,
    format: str = "cubin",
    path: str | bytes = "cpu-module",
) -> bytes:
    return string(format) + string(symbol) + string(path)


@dataclass(frozen=True)
class Response:
    opcode: int
    request_id: int
    status: int
    error: str
    payload: bytes


@dataclass(frozen=True)
class Transcript:
    responses: list[Response]
    events: list[str]
    stderr: str
    returncode: int


class ModuleServiceProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        compiler = shutil.which("c++")
        if compiler is None or sys.platform == "win32":
            raise unittest.SkipTest("A POSIX host and C++ compiler are required")
        temporary = tempfile.TemporaryDirectory(prefix="therock-service-protocol-")
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        repository = Path(__file__).resolve().parents[2]
        modules = repository / "tests/multi_vendor/modules"
        description = cls.root / "description.json"
        subprocess.run(
            [
                sys.executable,
                str(repository / "build_tools/configure_module_contract.py"),
                "--vendor",
                "nvidia",
                "--header",
                str(cls.root / "module_contract_data.h"),
                "--description",
                str(description),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        cls.description = json.loads(description.read_text())
        cls.executable = cls.root / "service"
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(cls.root),
                "-I",
                str(modules),
                str(modules / "service_test_backend.cpp"),
                "-o",
                str(cls.executable),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )

    def open_payload(
        self,
        *,
        uuid: str = "1" * 32,
        abi: str | None = None,
        version: int = 1,
        digest: str | None = None,
    ) -> bytes:
        return (
            u32(3)
            + u32(0)
            + string("sm_120")
            + string(uuid)
            + string(self.description["contract"]["abi"] if abi is None else abi)
            + u32(version)
            + string(self.description["contract_sha256"] if digest is None else digest)
        )

    def run_transcript(
        self,
        commands: list[Command],
        *,
        suffix: bytes = b"",
        arguments: tuple[str, ...] = ("--serve",),
    ) -> Transcript:
        data = (
            b"".join(
                frame(opcode, i, payload)
                for i, (opcode, payload) in enumerate(commands, 1)
            )
            + suffix
        )
        result = subprocess.run(
            [str(self.executable), *arguments],
            input=data,
            capture_output=True,
            timeout=10,
        )
        stderr = result.stderr.decode("utf-8")
        self.assertNotEqual(result.returncode, 93, stderr)
        self.assertNotIn("DESTRUCTION_WHILE_PENDING", stderr)
        responses = []
        offset = 0
        while offset < len(result.stdout):
            self.assertGreaterEqual(len(result.stdout) - offset, 16, stderr)
            magic, version_opcode, request_id, size = struct.unpack_from(
                "<4I", result.stdout, offset
            )
            self.assertEqual(magic, MAGIC, stderr)
            self.assertEqual(version_opcode & 0xFFFF, 1, stderr)
            self.assertTrue(version_opcode >> 16 & 0x8000, stderr)
            offset += 16
            body = result.stdout[offset : offset + size]
            self.assertEqual(len(body), size, stderr)
            self.assertGreaterEqual(size, 8, stderr)
            status, error_size = struct.unpack_from("<2I", body)
            self.assertLessEqual(error_size, size - 8, stderr)
            error = body[8 : 8 + error_size].decode("utf-8")
            responses.append(
                Response(
                    (version_opcode >> 16) & 0x7FFF,
                    request_id,
                    status,
                    error,
                    body[8 + error_size :],
                )
            )
            offset += size
        events = [
            line.removeprefix("CPU ")
            for line in stderr.splitlines()
            if line.startswith("CPU ")
        ]
        return Transcript(responses, events, stderr, result.returncode)

    def pending_session(self) -> list[Command]:
        # Shared handle namespace: buffers 1/2/3, then module 4.
        return [
            (1, b""),
            (2, self.open_payload()),
            (3, u32(4)),
            (3, u32(4)),
            (3, u32(4)),
            (5, load()),
            (7, write(1, 0, [1, 2, 3, 4])),
            (7, write(2, 0, [0.5, 0.5, 0.5, 0.5])),
            (9, launch(4, 1, 2, 3)),
        ]

    def assert_drained(
        self, transcript: Transcript, *, buffers: int, modules: int
    ) -> None:
        self.assertEqual(transcript.events.count("DESTROY_BUFFER"), buffers)
        self.assertEqual(transcript.events.count("DESTROY_MODULE"), modules)
        self.assertEqual(transcript.events.count("DESTROY_BACKEND"), 1)
        if "EXECUTE" in transcript.events:
            last_execution = max(
                i for i, event in enumerate(transcript.events) if event == "EXECUTE"
            )
            first_destruction = min(
                i
                for i, event in enumerate(transcript.events)
                if event.startswith("DESTROY_")
            )
            self.assertLess(last_execution, first_destruction)

    def test_hello_is_offline_and_service_mode_is_exclusive(self) -> None:
        result = self.run_transcript([(1, b""), (11, b"")])
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = result.responses[0].payload
        size = struct.unpack_from("<I", payload)[0]
        self.assertEqual(size, len(payload) - 4)
        self.assertEqual(json.loads(payload[4:]), self.description)
        self.assertEqual(result.events, [])
        for arguments in (("--serve", "--help"), ("--serve", "--serve")):
            with self.subTest(arguments=arguments):
                result = self.run_transcript([], arguments=arguments)
                self.assertEqual(result.returncode, 1)
                self.assertIn("must be used alone", result.stderr)
                self.assertEqual(result.responses, [])
                self.assertNotIn("OPEN", result.events)

    def test_bad_framing_is_fatal_before_backend_creation(self) -> None:
        cases = [
            (struct.pack("<4I", 0, 1 | (1 << 16), 1, 0), 2, "magic"),
            (struct.pack("<4I", MAGIC, 2 | (1 << 16), 1, 0), 5, "version"),
            (frame(1, 0), 2, "sequence"),
            (frame(1, 2), 2, "sequence"),
            (frame(0, 1), 2, "opcode"),
            (frame(14, 1), 2, "opcode"),
            (struct.pack("<4I", MAGIC, 1 | (1 << 16), 1, 1024 * 1024 + 1), 2, "1 MiB"),
            (b"TRMSx", 2, "Truncated"),
            (struct.pack("<4I", MAGIC, 1 | (1 << 16), 1, 8) + b"xx", 2, "Truncated"),
        ]
        for data, status, error in cases:
            with self.subTest(error=error, data=data[:16]):
                result = self.run_transcript([], suffix=data)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(len(result.responses), 1)
                self.assertEqual(result.responses[0].status, status)
                self.assertIn(error, result.responses[0].error)
                self.assertNotIn("OPEN", result.events)

    def test_open_validates_state_contract_uuid_and_utf8_before_backend(self) -> None:
        for updates, status in [
            ({"abi": "other"}, 5),
            ({"version": 2}, 5),
            ({"digest": "0" * 64}, 5),
            ({"uuid": "0" * 32}, 1),
            ({"uuid": "A" * 32}, 1),
            ({"uuid": "short"}, 1),
        ]:
            with self.subTest(updates=updates):
                result = self.run_transcript(
                    [(1, b""), (2, self.open_payload(**updates))]
                )
                self.assertEqual(result.responses[-1].status, status)
                self.assertNotIn("OPEN", result.events)
                self.assertEqual(result.returncode, 1 if status == 5 else 0)
        for text in (
            b"\xff",
            b"\xc0\x80",
            b"\xed\xa0\x80",
            b"\xf4\x90\x80\x80",
            b"a\0b",
            b"x" * 4097,
        ):
            with self.subTest(text=text[:8]):
                payload = u32(3) + u32(0) + string(text)
                result = self.run_transcript([(1, b""), (2, payload)])
                self.assertEqual(result.responses[-1].status, 1)
                self.assertNotIn("OPEN", result.events)
        result = self.run_transcript(
            [
                (3, u32(1)),
                (1, b""),
                (1, b""),
                (3, u32(1)),
                (2, self.open_payload()),
                (2, self.open_payload()),
                (11, b""),
            ]
        )
        self.assertEqual(
            [response.status for response in result.responses], [4, 0, 4, 4, 0, 4, 0]
        )
        self.assertEqual(result.events.count("OPEN"), 1)
        self.assertIn("BACKEND_STDOUT", result.events)

    def test_validation_errors_are_recoverable_without_backend_side_effects(
        self,
    ) -> None:
        setup = self.pending_session()[:6]
        invalid = [
            (4, u64(999)),
            (4, u64(4)),
            (6, u64(1)),
            (8, read(4, 0, 1)),
            (3, u32(0)),
            (3, u32(65557)),
            (3, u32(1) + b"x"),
            (5, load(format="spirv")),
            (5, load(symbol="unknown")),
            (5, load(path="")),
            (7, write(1, 4, [1])),
            (7, write(1, 0, [])),
            (7, u64(1) + u32(0) + u32(4) + struct.pack("<f", 1)),
            (7, write(1, 0, [1]) + b"x"),
            (8, read(1, 0xFFFFFFFF, 2)),
            (8, read(1, 0, 1) + b"x"),
            (9, launch(1, 1, 2, 3)),
            (9, launch(4, 1, 2, 1)),
            (9, launch(4, 1, 2, 2)),
            (9, launch(4, 1, 2, 3, alpha=float("nan"))),
            (9, launch(4, 1, 2, 3, alpha=float("inf"))),
            (9, launch(4, 1, 2, 3, count=0)),
            (9, launch(4, 1, 2, 3, count=5)),
            (9, launch(4, 1, 2, 3, count=65540)),
            (9, launch(4, 1, 2, 3) + b"x"),
            (10, b"x"),
        ]
        valid = self.pending_session()[6:] + [
            (8, read(3, 0, 4)),
            (4, u64(1)),
            (4, u64(1)),
            (10, b""),
            (11, b""),
        ]
        result = self.run_transcript(setup + invalid + valid)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(
            all(
                response.status == 1
                for response in result.responses[6 : 6 + len(invalid)]
            )
        )
        self.assertEqual(result.responses[-3].status, 1)
        for event, count in (
            ("ALLOCATE", 3),
            ("LOAD", 1),
            ("WRITE", 2),
            ("LAUNCH", 1),
            ("READ", 1),
            ("EXECUTE", 1),
        ):
            self.assertEqual(result.events.count(event), count, (event, result.stderr))
        output = result.responses[6 + len(invalid) + 3].payload
        self.assertEqual(struct.unpack("<4f", output), (2.25, 4.0, 5.75, 7.5))
        self.assert_drained(result, buffers=3, modules=1)

    def test_persistent_handles_offset_transfers_and_module_composition(self) -> None:
        commands = [(1, b""), (2, self.open_payload())] + [(3, u32(8))] * 4
        commands += [
            (5, load()),
            (5, load("therock_module_relu", format="ptx", path="雪.ptx")),
            (7, write(1, 1, [1, 2, 3, 4])),
            (7, write(2, 0, [0.5] * 8)),
            (9, launch(5, 1, 2, 3, alpha=2.0, count=5)),
            (9, launch(6, 3, 2, 4, alpha=1.0, count=5)),
            (8, read(4, 1, 4)),
            (10, b""),
            (6, u64(5)),
            (4, u64(1)),
            (11, b""),
        ]
        result = self.run_transcript(commands)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(response.status == 0 for response in result.responses))
        self.assertEqual(
            struct.unpack("<4f", result.responses[12].payload),
            (2.875, 4.875, 6.875, 8.875),
        )
        handles = [
            struct.unpack("<Q", result.responses[i].payload)[0] for i in range(2, 8)
        ]
        self.assertEqual(handles, list(range(1, 7)))
        self.assertEqual(result.events.count("OPEN"), 1)
        self.assertEqual(result.events.count("LAUNCH"), 2)
        self.assertEqual(result.events.count("EXECUTE"), 2)
        self.assert_drained(result, buffers=4, modules=2)

    def test_handle_limits_are_enforced_before_allocation_or_load(self) -> None:
        commands = [(1, b""), (2, self.open_payload())] + [(3, u32(1))] * 65
        commands += [(5, load())] * 33 + [(11, b"")]
        result = self.run_transcript(commands)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.responses[66].status, 1)
        self.assertEqual(result.responses[99].status, 1)
        self.assertEqual(result.events.count("ALLOCATE"), 64)
        self.assertEqual(result.events.count("LOAD"), 32)
        self.assert_drained(result, buffers=64, modules=32)

    def test_close_eof_and_fatal_frame_errors_drain_deferred_work(self) -> None:
        for ending in ("close", "eof", "bad-frame", "truncated-frame"):
            with self.subTest(ending=ending):
                commands = self.pending_session()
                suffix = b""
                if ending == "close":
                    commands.append((11, b""))
                elif ending == "bad-frame":
                    suffix = frame(10, 999)
                elif ending == "truncated-frame":
                    suffix = b"TRM"
                result = self.run_transcript(commands, suffix=suffix)
                self.assertEqual(
                    result.returncode, int("frame" in ending), result.stderr
                )
                self.assertEqual(result.events.count("EXECUTE"), 1)
                self.assert_drained(result, buffers=3, modules=1)

    def test_backend_failure_closes_existing_handles_after_drain(self) -> None:
        for failure in ("load", "launch-after-enqueue"):
            with self.subTest(failure=failure):
                if failure == "load":
                    commands = self.pending_session() + [(5, load(path="throw-load"))]
                else:
                    commands = self.pending_session()[:-1] + [
                        (9, launch(4, 1, 2, 3, alpha=13.0))
                    ]
                result = self.run_transcript(commands)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(result.responses[-1].status, 3)
                self.assertIn("Injected CPU", result.responses[-1].error)
                self.assertEqual(result.events.count("EXECUTE"), 1)
                self.assert_drained(result, buffers=3, modules=1)


if __name__ == "__main__":
    unittest.main()
