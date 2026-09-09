# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Offline session gates for real compiled Level Zero runners.

Set THEROCK_NATIVE_MODULE_BUILD_ROOT to the profile's modules directory. Linux
tests compile a zeInit trap against the SDK recorded in each child CMake cache.
The trap exits immediately, so neither negative cases nor the positive boundary
control initialize a driver. Synthetic SPIR-V only exercises structural reading;
these tests do not validate SPIR-V semantics or Intel hardware execution.
Service framing tests likewise keep HELLO and rejected OPEN operations offline.
"""

import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest


def _cache(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if re.match(r"[A-Za-z_][A-Za-z0-9_]*:[^=]+=", line):
            field, value = line.split("=", 1)
            values[field.split(":", 1)[0]] = value
    return values


class NativeLevelZeroSessionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = os.environ.get("THEROCK_NATIVE_MODULE_BUILD_ROOT")
        if not root:
            raise unittest.SkipTest(
                "Set THEROCK_NATIVE_MODULE_BUILD_ROOT for native session tests"
            )
        if not sys.platform.startswith("linux"):
            raise unittest.SkipTest("Level Zero interposition requires Linux")
        compiler = shutil.which("c++")
        if compiler is None:
            raise AssertionError("Level Zero session tests require a CPU C++ compiler")
        temporary = tempfile.TemporaryDirectory(prefix="therock-level-zero-session-")
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        source = cls.root / "init_trap.cpp"
        source.write_text(
            r"""#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstdlib>
extern "C" ze_result_t ZE_APICALL zeInit(ze_init_flags_t) {
  std::fputs("LEVEL_ZERO_STDOUT_DIAGNOSTIC\n", stdout);
  std::fflush(stdout);
  std::fputs("LEVEL_ZERO_SESSION_INIT_TRAP\n", stderr);
  std::fflush(stderr);
  std::_Exit(97);
}
"""
        )
        cls.runners = {}
        traps = {}
        for runner in sorted(Path(root).glob("*/build/therock_module_validation")):
            runner = runner.resolve(strict=True)
            cache = _cache(runner.parent / "CMakeCache.txt")
            if cache["THEROCK_MODULE_BACKEND"] != "intel":
                continue
            sdk = Path(cache["THEROCK_MODULE_SDK_ROOT"])
            if sdk not in traps:
                trap = cls.root / f"init-trap-{len(traps)}.so"
                subprocess.run(
                    [
                        compiler,
                        "-std=c++17",
                        "-shared",
                        "-fPIC",
                        "-Wall",
                        "-Wextra",
                        "-Werror",
                        f"-I{sdk / 'include'}",
                        str(source),
                        "-o",
                        str(trap),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
                traps[sdk] = trap
            environment = dict(os.environ)
            previous = environment.get("LD_PRELOAD", "")
            environment["LD_PRELOAD"] = str(traps[sdk]) + (
                ":" + previous if previous else ""
            )
            result = subprocess.run(
                [str(runner), "--describe-contract"],
                env=environment,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            description = json.loads(result.stdout)
            if (
                description["vendor"] != "intel"
                or description["backend"] != "level-zero"
                or not {
                    "multi-module-session",
                    "device-module-pipeline",
                    "persistent-module-service",
                }
                <= set(description["capabilities"])
            ):
                raise AssertionError(
                    f"Rebuild {runner}: expected an Intel session and pipeline adapter"
                )
            cls.runners[runner] = (environment, description)
        if not cls.runners:
            raise AssertionError(f"No compiled Intel native runners beneath {root}")

    def payload(self, name: str = "structural") -> Path:
        path = self.root / f"{name}.spv"
        # Structurally well-formed SPIR-V 1.0 header plus a one-word instruction.
        # It is deliberately not a semantic device module; zeInit is trapped.
        path.write_bytes(struct.pack("<6I", 0x07230203, 0x10000, 0, 1, 0, 0x10000))
        return path

    def contract_arguments(self, runner: Path) -> list[str]:
        _, description = self.runners[runner]
        return [
            "--launch-abi",
            description["contract"]["abi"],
            "--launch-abi-version",
            str(description["contract"]["version"]),
            "--launch-contract-sha256",
            description["contract_sha256"],
        ]

    def invoke(
        self, runner: Path, arguments: list[str], *, reaches_init: bool = False
    ) -> subprocess.CompletedProcess[str]:
        environment, _ = self.runners[runner]
        result = subprocess.run(
            [str(runner), *arguments],
            env=environment,
            capture_output=True,
            text=True,
            timeout=5,
        )
        evidence = f"{runner} {arguments!r}\n{result.stdout}\n{result.stderr}"
        self.assertEqual(result.returncode, 97 if reaches_init else 1, evidence)
        self.assertEqual(
            "LEVEL_ZERO_SESSION_INIT_TRAP" in result.stderr, reaches_init, evidence
        )
        for marker in ("DEVICE ", "MODULE ", "SESSION ", "CHECK ", "PASS "):
            self.assertNotIn(marker, result.stdout, evidence)
        return result

    def test_every_payload_is_read_before_driver_initialization(self) -> None:
        first = ["--module", "spirv", "therock_module_saxpy", str(self.payload())]
        bad_payload = self.root / "malformed.spv"
        malformed = (
            b"not SPIR-V",
            struct.pack("<6I", 0, 0x10000, 0, 1, 0, 0x10000),
            struct.pack("<6I", 0x07230203, 0x10000, 0, 1, 0, 0x20000),
        )
        for runner in self.runners:
            for data in malformed:
                with self.subTest(runner=runner, payload=data):
                    bad_payload.write_bytes(data)
                    result = self.invoke(
                        runner,
                        self.contract_arguments(runner)
                        + first
                        + [
                            "--module",
                            "spirv",
                            "therock_module_relu",
                            str(bad_payload),
                        ],
                    )
                    self.assertIn("Malformed SPIR-V", result.stderr)
            with self.subTest(runner=runner, payload="missing second"):
                result = self.invoke(
                    runner,
                    self.contract_arguments(runner)
                    + first
                    + [
                        "--module",
                        "spirv",
                        "therock_module_relu",
                        str(self.root / "missing.spv"),
                    ],
                )
                self.assertIn("Cannot read payload", result.stderr)

    def test_invalid_batch_metadata_never_initializes_driver(self) -> None:
        payload = str(self.payload())
        first = ["--module", "spirv", "therock_module_saxpy", payload]
        for runner in self.runners:
            contract = self.contract_arguments(runner)
            cases = {
                "wrong second format": contract
                + first
                + ["--module", "cubin", "therock_module_relu", payload],
                "wrong second symbol": contract
                + first
                + ["--module", "spirv", "unknown_symbol", payload],
                "duplicate request": contract + first + first,
                "missing module value": contract
                + first
                + ["--module", "spirv", "therock_module_relu"],
                "33 unique requests": contract
                + [
                    argument
                    for index in range(33)
                    for argument in (
                        "--module",
                        "spirv",
                        "therock_module_saxpy",
                        str(self.root / f"module-{index}.spv"),
                    )
                ],
            }
            for flag, value in (
                ("--payload", payload),
                ("--symbol", "therock_module_relu"),
                ("--payload-format", "spirv"),
            ):
                cases[f"mixed legacy {flag}"] = contract + first + [flag, value]
            for index in range(0, len(contract), 2):
                cases[f"missing {contract[index]}"] = (
                    contract[:index] + contract[index + 2 :] + first
                )
                changed = contract.copy()
                changed[index + 1] = "invalid"
                cases[f"wrong {contract[index]}"] = changed + first
            for name, arguments in cases.items():
                with self.subTest(runner=runner, case=name):
                    self.invoke(runner, arguments)

    def test_valid_two_request_control_reaches_the_initialization_trap(self) -> None:
        payload = str(self.payload())
        for runner in self.runners:
            with self.subTest(runner=runner):
                self.invoke(
                    runner,
                    self.contract_arguments(runner)
                    + [
                        "--module",
                        "spirv",
                        "therock_module_saxpy",
                        payload,
                        "--module",
                        "spirv",
                        "therock_module_relu",
                        payload,
                    ],
                    reaches_init=True,
                )

    def test_pipeline_accepts_repeated_stages_before_the_initialization_trap(
        self,
    ) -> None:
        payload = str(self.payload())
        saxpy = ["--module", "spirv", "therock_module_saxpy", payload]
        relu = ["--module", "spirv", "therock_module_relu", payload]
        for runner in self.runners:
            for stages in (saxpy, saxpy + relu + saxpy, saxpy * 32):
                with self.subTest(runner=runner, stages=len(stages) // 4):
                    self.invoke(
                        runner,
                        self.contract_arguments(runner) + ["--pipeline"] + stages,
                        reaches_init=True,
                    )

    def test_pipeline_requires_module_mode_and_preserves_request_validation(
        self,
    ) -> None:
        payload = str(self.payload())
        stage = ["--module", "spirv", "therock_module_saxpy", payload]
        for runner in self.runners:
            contract = self.contract_arguments(runner)
            prefix = contract + ["--pipeline"]
            cases = {
                "missing modules": prefix,
                "legacy mode": prefix
                + [
                    "--payload",
                    payload,
                    "--symbol",
                    "therock_module_saxpy",
                    "--payload-format",
                    "spirv",
                ],
                "duplicate pipeline flag": prefix + stage + ["--pipeline"],
                "pipeline takes no value": prefix + ["true"] + stage,
                "33 repeated stages": prefix + stage * 33,
                "missing module value": prefix
                + stage
                + ["--module", "spirv", "therock_module_relu"],
                "wrong later format": prefix
                + stage
                + ["--module", "cubin", "therock_module_relu", payload],
                "missing contract": ["--pipeline"] + stage,
                "missing later payload": prefix
                + stage
                + [
                    "--module",
                    "spirv",
                    "therock_module_relu",
                    str(self.root / "missing.spv"),
                ],
            }
            for flag, value in (
                ("--payload", payload),
                ("--symbol", "therock_module_relu"),
                ("--payload-format", "spirv"),
            ):
                cases[f"mixed legacy {flag}"] = prefix + stage + [flag, value]
            for name, arguments in cases.items():
                with self.subTest(runner=runner, case=name):
                    self.invoke(runner, arguments)

    @staticmethod
    def service_frame(
        opcode: int, request_id: int, payload: bytes = b"", *, version: int = 1
    ) -> bytes:
        return (
            struct.pack(
                "<4I", 0x534D5254, version | (opcode << 16), request_id, len(payload)
            )
            + payload
        )

    @staticmethod
    def service_string(value: str) -> bytes:
        data = value.encode("utf-8")
        return struct.pack("<I", len(data)) + data

    def service_open(
        self,
        runner: Path,
        *,
        device_id: int = 0xE223,
        architecture: str = "spirv",
        uuid: str = "12" * 16,
        version: int | None = None,
    ) -> bytes:
        _, description = self.runners[runner]
        return (
            struct.pack("<II", 0, device_id)
            + self.service_string(architecture)
            + self.service_string(uuid)
            + self.service_string(description["contract"]["abi"])
            + struct.pack(
                "<I",
                description["contract"]["version"] if version is None else version,
            )
            + self.service_string(description["contract_sha256"])
        )

    def invoke_service(
        self, runner: Path, requests: bytes, *, status: int = 0, trapped: bool = False
    ) -> list[tuple[int, int, int, str, bytes]]:
        environment, _ = self.runners[runner]
        result = subprocess.run(
            [str(runner), "--serve"],
            input=requests,
            env=environment,
            capture_output=True,
            timeout=5,
        )
        evidence = f"{runner}\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
        self.assertEqual(result.returncode, status, evidence)
        self.assertEqual(
            b"LEVEL_ZERO_SESSION_INIT_TRAP" in result.stderr, trapped, evidence
        )
        if trapped:
            self.assertIn(b"LEVEL_ZERO_STDOUT_DIAGNOSTIC", result.stderr, evidence)
        # Decode the complete stdout stream. Any leaked native diagnostic text
        # would invalidate the magic, response opcode, frame length, or suffix.
        frames = []
        offset = 0
        while offset < len(result.stdout):
            self.assertGreaterEqual(len(result.stdout) - offset, 16, evidence)
            magic, code, request_id, length = struct.unpack_from(
                "<4I", result.stdout, offset
            )
            self.assertEqual(magic, 0x534D5254, evidence)
            self.assertEqual(code & 0xFFFF, 1, evidence)
            self.assertTrue((code >> 16) & 0x8000, evidence)
            offset += 16
            body = result.stdout[offset : offset + length]
            self.assertEqual(len(body), length, evidence)
            self.assertGreaterEqual(len(body), 8, evidence)
            reply_status, error_length = struct.unpack_from("<II", body)
            self.assertLessEqual(8 + error_length, len(body), evidence)
            error = body[8 : 8 + error_length].decode("utf-8")
            frames.append(
                (
                    (code >> 16) & 0x7FFF,
                    request_id,
                    reply_status,
                    error,
                    body[8 + error_length :],
                )
            )
            offset += length
        return frames

    def test_service_hello_returns_binary_description_without_initialization(
        self,
    ) -> None:
        for runner in self.runners:
            with self.subTest(runner=runner):
                frames = self.invoke_service(
                    runner, self.service_frame(1, 1) + self.service_frame(11, 2)
                )
                self.assertEqual(len(frames), 2)
                self.assertEqual(frames[0][:4], (1, 1, 0, ""))
                body = frames[0][4]
                self.assertGreaterEqual(len(body), 4)
                length = struct.unpack_from("<I", body)[0]
                self.assertEqual(length, len(body) - 4)
                self.assertEqual(json.loads(body[4:]), self.runners[runner][1])
                self.assertEqual(frames[1], (11, 2, 0, "", b""))

    def test_service_invalid_open_and_protocol_fail_before_initialization(
        self,
    ) -> None:
        for runner in self.runners:
            cases = (
                ("zero UUID", {"uuid": "0" * 32}, 1),
                ("uppercase UUID", {"uuid": "AB" * 16}, 1),
                ("short UUID", {"uuid": "12"}, 1),
                ("wrong ABI version", {"version": 2}, 5),
                ("wrong Intel architecture", {"architecture": "sm_120"}, 3),
                ("missing Intel device ID", {"device_id": 0}, 3),
            )
            for name, changes, reply_status in cases:
                with self.subTest(runner=runner, case=name):
                    fatal = reply_status in (3, 5)
                    frames = self.invoke_service(
                        runner,
                        self.service_frame(1, 1)
                        + self.service_frame(2, 2, self.service_open(runner, **changes))
                        + self.service_frame(11, 3),
                        status=1 if fatal else 0,
                    )
                    self.assertEqual(len(frames), 2 if fatal else 3)
                    self.assertEqual(frames[1][:3], (2, 2, reply_status))
                    self.assertTrue(frames[1][3])
                    self.assertEqual(frames[1][4], b"")
                    if not fatal:
                        self.assertEqual(frames[2], (11, 3, 0, "", b""))
            with self.subTest(runner=runner, case="wrong protocol version"):
                frames = self.invoke_service(
                    runner, self.service_frame(1, 1, version=2), status=1
                )
                self.assertEqual(len(frames), 1)
                self.assertEqual(frames[0][:3], (1, 1, 5))

    def test_service_valid_open_reaches_only_the_initialization_trap(self) -> None:
        for runner in self.runners:
            with self.subTest(runner=runner):
                frames = self.invoke_service(
                    runner,
                    self.service_frame(1, 1)
                    + self.service_frame(2, 2, self.service_open(runner)),
                    status=97,
                    trapped=True,
                )
                # HELLO completed; the OPEN control exited inside the trap before
                # any native driver could initialize or publish a successful OPEN.
                self.assertEqual(len(frames), 1)
                self.assertEqual(frames[0][:4], (1, 1, 0, ""))


if __name__ == "__main__":
    unittest.main()
