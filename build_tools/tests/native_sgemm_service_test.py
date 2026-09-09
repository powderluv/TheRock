# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU-only SGEMM tests of the real worker protocol and deferred resource lifetime.

The fixture models queued arithmetic; it does not load or qualify a BLAS provider.
"""

import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import module_service_protocol_test as protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _therock_utils.sgemm_contract import (
    SGEMM_ABI,
    SGEMM_VERSION,
    parse_sgemm_provider_json,
    sgemm_contract_sha256,
)


def negotiate(
    provider: str = "cublas",
    abi: str = SGEMM_ABI,
    version: int = SGEMM_VERSION,
    digest: str | None = None,
) -> bytes:
    return (
        protocol.string(provider)
        + protocol.string(abi)
        + protocol.u32(version)
        + protocol.string(sgemm_contract_sha256() if digest is None else digest)
    )


def sgemm(
    a: int = 1,
    b: int = 2,
    c: int = 3,
    *,
    a_offset: int = 0,
    b_offset: int = 0,
    c_offset: int = 0,
    m: int = 2,
    n: int = 2,
    k: int = 2,
    lda: int = 2,
    ldb: int = 2,
    ldc: int = 2,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> bytes:
    return struct.pack(
        "<3Q9I2f",
        a,
        b,
        c,
        a_offset,
        b_offset,
        c_offset,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        alpha,
        beta,
    )


class NativeSgemmServiceTest(unittest.TestCase):
    # Reuse framing assertions without inheriting and rerunning unrelated tests.
    open_payload = protocol.ModuleServiceProtocolTest.open_payload
    run_transcript = protocol.ModuleServiceProtocolTest.run_transcript

    @classmethod
    def setUpClass(cls) -> None:
        compiler = shutil.which("c++")
        if compiler is None or sys.platform == "win32":
            raise unittest.SkipTest("A POSIX host and C++ compiler are required")
        temporary = tempfile.TemporaryDirectory(prefix="therock-sgemm-protocol-")
        cls.addClassCleanup(temporary.cleanup)
        root = Path(temporary.name)
        repository = Path(__file__).resolve().parents[2]
        modules = repository / "tests/multi_vendor/modules"
        for enabled in (False, True):
            build = root / ("on" if enabled else "off")
            build.mkdir()
            description = build / "description.json"
            subprocess.run(
                [
                    sys.executable,
                    str(repository / "build_tools/configure_module_contract.py"),
                    "--vendor",
                    "nvidia",
                    "--header",
                    str(build / "module_contract_data.h"),
                    "--description",
                    str(description),
                    *(["--enable-sgemm"] if enabled else []),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            executable = build / "service"
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    *(["-DTHEROCK_MODULE_ENABLE_SGEMM=1"] if enabled else []),
                    "-I",
                    str(build),
                    "-I",
                    str(modules),
                    str(modules / "service_test_backend.cpp"),
                    "-o",
                    str(executable),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            if enabled:
                cls.executable = executable
                cls.description = json.loads(description.read_text())
            else:
                cls.disabled_executable = executable
                cls.disabled_description = json.loads(description.read_text())

    def setup_commands(self, capacity: int = 20) -> list[protocol.Command]:
        return [(1, b""), (2, self.open_payload()), (12, negotiate())] + [
            (3, protocol.u32(capacity))
        ] * 3

    def assert_drained(
        self, result: protocol.Transcript, buffers: int, modules: int = 0
    ) -> None:
        self.assertEqual(result.events.count("DESTROY_BUFFER"), buffers, result.stderr)
        self.assertEqual(result.events.count("DESTROY_MODULE"), modules, result.stderr)
        self.assertEqual(result.events.count("DESTROY_BACKEND"), 1, result.stderr)
        executions = [
            i for i, value in enumerate(result.events) if value.startswith("EXECUTE")
        ]
        if executions:
            first_destroy = next(
                i
                for i, value in enumerate(result.events)
                if value.startswith("DESTROY_")
            )
            self.assertLess(max(executions), first_destroy, result.stderr)

    def test_disabled_worker_rejects_provider_without_initializing_it(self) -> None:
        self.executable = self.disabled_executable
        result = self.run_transcript(
            [(1, b""), (2, self.open_payload()), (12, negotiate())]
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.responses[-1].status, 5)
        self.assertNotIn("SGEMM_INFO", result.events)
        self.assertNotIn("SGEMM", result.events)
        self.assertFalse(
            any("blas" in value for value in self.disabled_description["capabilities"])
        )
        self.assertEqual(
            self.disabled_description["contract"], self.description["contract"]
        )
        self.assertEqual(
            self.disabled_description["contract_sha256"],
            self.description["contract_sha256"],
        )
        dynamic = subprocess.run(
            ["ldd", str(self.executable)], capture_output=True, text=True, check=True
        )
        self.assertNotIn("cublas", dynamic.stdout)
        self.assertNotIn("rocblas", dynamic.stdout)

    def test_negotiation_checks_identity_before_provider_and_enforces_state(
        self,
    ) -> None:
        for update in (
            {"provider": "rocblas"},
            {"abi": "other"},
            {"version": 2},
            {"digest": "0" * 64},
        ):
            with self.subTest(update=update):
                result = self.run_transcript(
                    [(1, b""), (2, self.open_payload()), (12, negotiate(**update))]
                )
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(result.responses[-1].status, 5)
                self.assertNotIn("SGEMM_INFO", result.events)
        result = self.run_transcript(
            [
                (1, b""),
                (12, negotiate()),
                (2, self.open_payload()),
                (13, sgemm()),
                (12, negotiate()),
                (12, negotiate()),
                (11, b""),
            ]
        )
        self.assertEqual(
            [item.status for item in result.responses], [0, 4, 0, 4, 0, 4, 0]
        )
        payload = result.responses[4].payload
        self.assertEqual(protocol.u32(len(payload) - 4), payload[:4])
        info = parse_sgemm_provider_json(payload[4:].decode(), vendor="nvidia")
        self.assertEqual(info.library_version, "cpu-fixture")
        self.assertEqual(result.events.count("SGEMM_INFO"), 1)
        self.assertNotIn("SGEMM", result.events)

    def test_invalid_ranges_aliases_scalars_and_frames_do_not_enqueue(self) -> None:
        invalid = [
            sgemm(**update)
            for update in (
                {"a": 999},
                {"c": 1},
                {"c": 2},
                {"m": 0},
                {"n": 257},
                {"k": 0},
                {"lda": 1},
                {"ldb": 1},
                {"ldc": 1},
                {"lda": 65557},
                {"a_offset": 0xFFFFFFFF},
                {"b_offset": 18},
                {"c_offset": 18},
                {
                    "m": 256,
                    "n": 256,
                    "k": 256,
                    "lda": 65556,
                    "ldb": 65556,
                    "ldc": 65556,
                },
                {"alpha": float("nan")},
                {"alpha": float("inf")},
                {"beta": float("nan")},
                {"beta": float("-inf")},
            )
        ]
        invalid += [sgemm()[:-1], sgemm() + b"x"]
        setup = self.setup_commands()
        result = self.run_transcript(
            setup
            + [(13, payload) for payload in invalid]
            + [(13, sgemm(a=1, b=1)), (10, b""), (11, b"")]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(
            all(item.status == 1 for item in result.responses[len(setup) : -3]),
            result.responses,
        )
        self.assertTrue(all(item.status == 0 for item in result.responses[-3:]))
        self.assertEqual(result.events.count("SGEMM"), 1)
        self.assertEqual(result.events.count("EXECUTE_SGEMM"), 1)
        self.assert_drained(result, 3)

    def test_padded_column_major_sgemm_and_kernel_share_one_ordered_queue(self) -> None:
        a = [-99.0] * 20
        b = [-99.0] * 20
        for index, value in zip((1, 2, 4, 5, 7, 8), (1, 2, 3, 4, 5, 6)):
            a[index] = value
        for index, value in zip((2, 3, 4, 6, 7, 8), (1, 0, 1, 2, 1, 0)):
            b[index] = value
        c = [-99.0] * 20
        for index in (3, 4, 7, 8):
            c[index] = 2.0
        commands = (
            self.setup_commands()
            + [(3, protocol.u32(20))] * 2
            + [(5, protocol.load("therock_module_relu"))]
        )
        commands += [
            (7, protocol.write(1, 0, a)),
            (7, protocol.write(2, 0, b)),
            (7, protocol.write(3, 0, c)),
            (7, protocol.write(4, 0, [0.5] * 20)),
            (
                13,
                sgemm(
                    a_offset=1,
                    b_offset=2,
                    c_offset=3,
                    k=3,
                    lda=3,
                    ldb=4,
                    ldc=4,
                    alpha=2,
                    beta=0.5,
                ),
            ),
            (9, protocol.launch(6, 3, 4, 5, alpha=1, count=20)),
            (
                13,
                sgemm(
                    a=5, b=1, c=3, a_offset=3, b_offset=1, c_offset=12, n=1, k=1, ldb=1
                ),
            ),
            (8, protocol.read(3, 0, 20)),
            (11, b""),
        ]
        result = self.run_transcript(commands)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(
            all(item.status == 0 for item in result.responses), result.responses
        )
        expected = c.copy()
        for index, value in zip((3, 4, 7, 8, 12, 13), (13, 17, 11, 17, 13.375, 17.375)):
            expected[index] = value
        self.assertEqual(
            struct.unpack("<20f", result.responses[-2].payload), tuple(expected)
        )
        self.assertEqual(
            [event for event in result.events if event.startswith("EXECUTE")],
            ["EXECUTE_SGEMM", "EXECUTE", "EXECUTE_SGEMM"],
        )
        self.assert_drained(result, 5, 1)

    def test_release_close_eof_and_fatal_errors_drain_sgemm_before_destruction(
        self,
    ) -> None:
        for ending in (
            "release",
            "unload",
            "close",
            "eof",
            "bad-frame",
            "backend-failure",
        ):
            with self.subTest(ending=ending):
                commands = self.setup_commands(4) + [
                    (5, protocol.load()),
                    (13, sgemm(alpha=13 if ending == "backend-failure" else 1)),
                ]
                suffix = b""
                if ending == "release":
                    commands += [(4, protocol.u64(1)), (11, b"")]
                elif ending == "unload":
                    commands += [(6, protocol.u64(4)), (11, b"")]
                elif ending == "close":
                    commands += [(11, b"")]
                elif ending == "bad-frame":
                    suffix = protocol.frame(10, 999)
                result = self.run_transcript(commands, suffix=suffix)
                self.assertEqual(
                    result.returncode,
                    int(ending in ("bad-frame", "backend-failure")),
                    result.stderr,
                )
                if ending == "backend-failure":
                    self.assertEqual(result.responses[-1].status, 3)
                    self.assertIn("after enqueue", result.responses[-1].error)
                self.assertEqual(result.events.count("EXECUTE_SGEMM"), 1)
                self.assert_drained(result, 3, 1)


if __name__ == "__main__":
    unittest.main()
