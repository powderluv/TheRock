# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU-scheduled event/cleanup tests of real compiled HIP and CUDA host runners.

Set THEROCK_NATIVE_MODULE_BUILD_ROOT to a profile's modules directory. This test
requires Linux and SDK headers recorded in the native child CMake caches. It
interposes every imported GPU function, never forwards to a driver, and uses
synthetic payload headers. Intel execution is outside this CPU shim's scope.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
_SHIM_SOURCE = REPO_ROOT / "tests/multi_vendor/modules/event_test_shim.cpp"


def _symbols(path: Path, *, defined: bool) -> set[str]:
    result = subprocess.run(
        ["nm", "-D", "--defined-only" if defined else "--undefined-only", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.split()[-1].split("@")[0] for line in result.stdout.splitlines()}


def _cache(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if re.match(r"[A-Za-z_][A-Za-z0-9_]*:[^=]+=", line):
            field, value = line.split("=", 1)
            result[field.split(":", 1)[0]] = value
    return result


class NativeEventPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get("THEROCK_NATIVE_MODULE_BUILD_ROOT")
        if not root:
            raise unittest.SkipTest(
                "Set THEROCK_NATIVE_MODULE_BUILD_ROOT for native event tests"
            )
        if not sys.platform.startswith("linux"):
            raise unittest.SkipTest("Native event interposition requires Linux")
        compiler = shutil.which("c++")
        if compiler is None or shutil.which("nm") is None:
            raise AssertionError("Native event tests require a CPU C++ compiler and nm")
        temporary = tempfile.TemporaryDirectory(prefix="therock-native-events-")
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.runners = {}
        built_shims = {}
        for runner in sorted(Path(root).glob("*/build/therock_module_validation")):
            cache = _cache(runner.parent / "CMakeCache.txt")
            vendor = cache["THEROCK_MODULE_BACKEND"]
            if vendor not in ("amd", "nvidia"):
                continue
            sdk = Path(cache["THEROCK_MODULE_SDK_ROOT"])
            key = (vendor, sdk)
            if key not in built_shims:
                shim = cls.root / f"shim-{len(built_shims)}.so"
                definition = (
                    "THEROCK_EVENT_TEST_CUDA=1"
                    if vendor == "nvidia"
                    else "__HIP_PLATFORM_AMD__=1"
                )
                subprocess.run(
                    [
                        compiler,
                        "-std=c++17",
                        "-shared",
                        "-fPIC",
                        "-Wall",
                        "-Wextra",
                        "-Werror",
                        f"-D{definition}",
                        f"-I{sdk / 'include'}",
                        str(_SHIM_SOURCE),
                        "-o",
                        str(shim),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                built_shims[key] = shim
            shim = built_shims[key]
            # Refuse to run if a newly imported native API is not interposed.
            # No new runner operation may accidentally escape into a real GPU.
            imported = {
                name
                for name in _symbols(runner, defined=False)
                if name.startswith(("cu", "hip"))
            }
            missing = imported - _symbols(shim, defined=True)
            if missing:
                raise AssertionError(
                    f"Uninterposed GPU APIs in {runner}: {sorted(missing)}"
                )
            environment = dict(os.environ)
            environment["LD_PRELOAD"] = str(shim)
            result = subprocess.run(
                [str(runner), "--describe-contract"],
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            )
            description = json.loads(result.stdout)
            if "cross-queue-events" not in description["capabilities"]:
                raise AssertionError(
                    f"Rebuild {runner}: missing cross-queue-events capability"
                )
            cls.runners[runner] = (vendor, shim, description)
        if not cls.runners:
            raise AssertionError(f"No HIP/CUDA runners beneath {root}")

    def payload(self, vendor: str, name: str = "fixture") -> Path:
        payload = self.root / f"synthetic-{vendor}-{name}.elf"
        header = bytearray(64)
        header[:7] = b"\x7fELF\x02\x01\x01"
        header[18:20] = struct.pack("<H", 190 if vendor == "nvidia" else 224)
        payload.write_bytes(header)
        return payload

    def invoke_arguments(
        self,
        runner: Path,
        arguments: list[str],
        *,
        failure: str = "none",
        modules: int = 1,
        omit_contract_flag: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        _, shim, description = self.runners[runner]
        environment = dict(os.environ)
        environment["LD_PRELOAD"] = str(shim)
        environment["THEROCK_EVENT_TEST_FAILURE"] = failure
        environment["THEROCK_EVENT_TEST_MODULES"] = str(modules)
        environment["THEROCK_EVENT_TEST_PIPELINE"] = (
            "1" if "--pipeline" in arguments else "0"
        )
        contract_arguments = [
            "--launch-abi",
            description["contract"]["abi"],
            "--launch-abi-version",
            str(description["contract"]["version"]),
            "--launch-contract-sha256",
            description["contract_sha256"],
        ]
        if omit_contract_flag is not None:
            index = contract_arguments.index(omit_contract_flag)
            del contract_arguments[index : index + 2]
        result = subprocess.run(
            [
                str(runner),
                *arguments,
                "--device",
                "0",
                "--expect-device-uuid",
                "12" * 16,
                *contract_arguments,
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=5,
        )
        evidence = f"{runner} {failure} {arguments}\n{result.stdout}\n{result.stderr}"
        self.assertNotEqual(result.returncode, 93, evidence)
        self.assertNotIn("VIOLATION", result.stderr, evidence)
        return result

    def invoke(
        self, runner: Path, *, symbol: str = "saxpy", failure: str = "none"
    ) -> subprocess.CompletedProcess[str]:
        vendor, _, _ = self.runners[runner]
        return self.invoke_arguments(
            runner,
            [
                "--payload",
                str(self.payload(vendor)),
                "--symbol",
                f"therock_module_{symbol}",
                "--payload-format",
                "cubin" if vendor == "nvidia" else "hsaco",
            ],
            failure=failure,
        )

    def session_arguments(self, runner: Path) -> list[str]:
        vendor, _, _ = self.runners[runner]
        payload_format = "cubin" if vendor == "nvidia" else "hsaco"
        return [
            "--module",
            payload_format,
            "therock_module_saxpy",
            str(self.payload(vendor, "saxpy")),
            "--module",
            payload_format,
            "therock_module_relu",
            str(self.payload(vendor, "relu")),
        ]

    def pipeline_arguments(self, runner: Path, stages: tuple[str, ...]) -> list[str]:
        vendor, _, _ = self.runners[runner]
        arguments = ["--pipeline"]
        for stage in stages:
            arguments.extend(
                [
                    "--module",
                    "cubin" if vendor == "nvidia" else "hsaco",
                    f"therock_module_{stage}",
                    str(self.payload(vendor, stage)),
                ]
            )
        return arguments

    def test_three_queue_graph_computes_fixture_only_after_final_event_wait(self):
        for runner in self.runners:
            for symbol in ("saxpy", "relu"):
                with self.subTest(runner=runner, symbol=symbol):
                    result = self.invoke(runner, symbol=symbol)
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertIn("PASS backend=", result.stdout)
                    self.assertEqual(result.stdout.count("CHECK n="), 18)
                    self.assertEqual(
                        result.stderr.count("CPU_EVENT_SHIM graph-complete"), 18
                    )
                    self.assertEqual(
                        result.stderr.count("CPU_EVENT_SHIM queue-wait-event"), 36
                    )
                    self.assertEqual(
                        result.stderr.count("CPU_EVENT_SHIM queue-create"), 3
                    )
                    self.assertEqual(
                        result.stderr.count("CPU_EVENT_SHIM queue-destroy"), 3
                    )

    def test_partial_submission_failure_drains_before_releasing_resources(self):
        for runner in self.runners:
            for failure in ("record", "launch"):
                with self.subTest(runner=runner, failure=failure):
                    result = self.invoke(runner, failure=failure)
                    self.assertNotEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertIn(f"CPU_EVENT_SHIM INJECT {failure}", result.stderr)
                    self.assertIn("CPU_EVENT_SHIM free", result.stderr)
                    self.assertIn("CPU_EVENT_SHIM module-unload", result.stderr)
                    self.assertEqual(
                        result.stderr.count("CPU_EVENT_SHIM queue-destroy"), 3
                    )
                    self.assertEqual(
                        result.stderr.count("CPU_EVENT_SHIM event-destroy"), 3
                    )
                    self.assertNotIn("PASS backend=", result.stdout)

    def test_unknown_host_event_completion_does_not_destroy_pending_resources(self):
        for runner in self.runners:
            with self.subTest(runner=runner):
                result = self.invoke(runner, failure="query")
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("CPU_EVENT_SHIM INJECT query", result.stderr)
                self.assertIn("completion=unknown", result.stderr)
                for forbidden in (
                    "free",
                    "event-destroy",
                    "queue-destroy",
                    "module-unload",
                ):
                    self.assertNotIn(f"CPU_EVENT_SHIM {forbidden}", result.stderr)
                self.assertNotIn("PASS backend=", result.stdout)

    def test_unknown_cleanup_completion_does_not_free_pending_resources(self):
        for runner in self.runners:
            with self.subTest(runner=runner):
                result = self.invoke(runner, failure="drain")
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("CPU_EVENT_SHIM INJECT drain", result.stderr)
                self.assertIn("completion=unknown", result.stderr)
                for forbidden in ("free", "event-destroy", "module-unload"):
                    self.assertNotIn(f"CPU_EVENT_SHIM {forbidden}", result.stderr)
                self.assertNotIn("PASS backend=", result.stdout)

    def test_session_retains_modules_and_reuses_one_context_and_resource_set(self):
        for runner, (vendor, _, _) in self.runners.items():
            with self.subTest(runner=runner):
                result = self.invoke_arguments(
                    runner, self.session_arguments(runner), modules=2
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("PASS backend=", result.stdout)
                self.assertEqual(result.stdout.count("CHECK n="), 36)
                for operation, count in (
                    ("device-init", 1),
                    ("module-load id=", 2),
                    ("function-load", 2),
                    ("device-allocate", 3),
                    ("host-allocate", 3),
                    ("queue-create", 3),
                    ("event-create", 3),
                    ("graph-complete", 36),
                    ("queue-destroy", 3),
                    ("event-destroy", 3),
                    ("module-unload", 2),
                    ("free", 6),
                ):
                    self.assertEqual(
                        result.stderr.count(f"CPU_EVENT_SHIM {operation}"),
                        count,
                        result.stderr,
                    )
                if vendor == "nvidia":
                    self.assertEqual(
                        result.stderr.count("CPU_EVENT_SHIM context-create"), 1
                    )
                    self.assertEqual(
                        result.stderr.count("CPU_EVENT_SHIM context-release"), 1
                    )
                else:
                    self.assertEqual(
                        result.stderr.count("CPU_EVENT_SHIM device-select"), 1
                    )
                first_allocation = result.stderr.index("CPU_EVENT_SHIM device-allocate")
                self.assertLess(
                    result.stderr.rindex("CPU_EVENT_SHIM module-load id="),
                    first_allocation,
                )
                self.assertLess(
                    result.stderr.rindex("CPU_EVENT_SHIM function-load"),
                    first_allocation,
                )
                self.assertGreater(
                    result.stderr.index("CPU_EVENT_SHIM module-unload"),
                    result.stderr.rindex("CPU_EVENT_SHIM graph-complete"),
                )
                submissions = re.findall(
                    r"CPU_EVENT_SHIM kernel-submit module=(\d+) symbol=(\w+) count=(\d+)",
                    result.stderr,
                )
                expected = [
                    (str(module), symbol, str(count))
                    for count in (1, 127, 128, 129, 4099, 65539)
                    for _ in range(3)
                    for module, symbol in ((0, "saxpy"), (1, "relu"))
                ]
                self.assertEqual(submissions, expected)
                device_sizes = [
                    int(value)
                    for value in re.findall(
                        r"CPU_EVENT_SHIM device-allocate bytes=(\d+)", result.stderr
                    )
                ]
                self.assertTrue(all(value >= 65539 * 4 for value in device_sizes))
                self.assertGreaterEqual(max(device_sizes), (65539 + 17) * 4)

    def test_second_module_load_failure_cleans_first_without_allocating_or_launching(
        self,
    ):
        for runner in self.runners:
            with self.subTest(runner=runner):
                result = self.invoke_arguments(
                    runner,
                    self.session_arguments(runner),
                    modules=2,
                    failure="second-module",
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("CPU_EVENT_SHIM INJECT second-module", result.stderr)
                self.assertEqual(
                    result.stderr.count("CPU_EVENT_SHIM module-load id="), 1
                )
                self.assertEqual(result.stderr.count("CPU_EVENT_SHIM module-unload"), 1)
                for operation in (
                    "device-allocate",
                    "host-allocate",
                    "kernel-submit",
                    "queue-create",
                ):
                    self.assertNotIn(f"CPU_EVENT_SHIM {operation}", result.stderr)
                self.assertNotIn("PASS backend=", result.stdout)

    def test_late_invalid_batch_input_fails_before_driver_initialization(self):
        for runner, (vendor, _, _) in self.runners.items():
            valid = self.session_arguments(runner)
            bad_format = list(valid)
            bad_format[5] = "spirv"
            bad_symbol = list(valid)
            bad_symbol[6] = "therock_module_unknown"
            missing_payload = list(valid)
            missing_payload[7] = str(self.root / "missing-payload")
            wrong_payload = list(valid)
            wrong_payload[7] = str(
                self.payload("amd" if vendor == "nvidia" else "nvidia", "wrong-backend")
            )
            for arguments in (bad_format, bad_symbol, missing_payload, wrong_payload):
                with self.subTest(runner=runner, arguments=arguments):
                    result = self.invoke_arguments(runner, arguments, modules=2)
                    self.assertNotEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertNotIn("CPU_EVENT_SHIM", result.stderr)
                    self.assertNotIn("PASS backend=", result.stdout)

    def test_mixed_duplicate_or_incomplete_module_options_fail_before_driver(self):
        for runner, (vendor, _, _) in self.runners.items():
            valid = self.session_arguments(runner)
            mixed = [
                valid + ["--payload", str(self.payload(vendor))],
                valid + ["--symbol", "therock_module_saxpy"],
                valid + ["--payload-format", valid[1]],
                valid + ["--payload", ""],
                valid + ["--symbol", ""],
                valid + ["--payload-format", ""],
                valid + valid[:4],
                valid + ["--module"],
                valid + ["--module", valid[1]],
                valid + ["--module", valid[1], "therock_module_relu"],
            ]
            for arguments in mixed:
                with self.subTest(runner=runner, arguments=arguments):
                    result = self.invoke_arguments(runner, arguments, modules=2)
                    self.assertNotEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertNotIn("CPU_EVENT_SHIM", result.stderr)

    def test_batch_requires_each_global_contract_flag_before_driver_access(self):
        for runner in self.runners:
            for flag in (
                "--launch-abi",
                "--launch-abi-version",
                "--launch-contract-sha256",
            ):
                with self.subTest(runner=runner, flag=flag):
                    result = self.invoke_arguments(
                        runner,
                        self.session_arguments(runner),
                        modules=2,
                        omit_contract_flag=flag,
                    )
                    self.assertNotEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertIn("require", result.stderr)
                    self.assertNotIn("CPU_EVENT_SHIM", result.stderr)

    def test_more_than_32_modules_fail_before_payload_or_driver_access(self):
        for runner in self.runners:
            valid = self.session_arguments(runner)
            arguments = []
            for index in range(33):
                arguments.extend(
                    [
                        "--module",
                        valid[1],
                        "therock_module_saxpy",
                        str(self.root / f"missing-{index}"),
                    ]
                )
            with self.subTest(runner=runner):
                result = self.invoke_arguments(runner, arguments, modules=33)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertRegex(result.stderr, r"32|[Tt]oo many")
                self.assertNotIn("CPU_EVENT_SHIM", result.stderr)

    def test_device_resident_pipeline_composes_stages_without_intermediate_readback(
        self,
    ):
        for runner, (vendor, _, _) in self.runners.items():
            for stages in (("saxpy",), ("saxpy", "relu"), ("saxpy", "relu", "saxpy")):
                with self.subTest(runner=runner, stages=stages):
                    result = self.invoke_arguments(
                        runner,
                        self.pipeline_arguments(runner, stages),
                        modules=len(stages),
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertIn("mode=device-module-pipeline", result.stdout)
                    self.assertIn("PASS backend=", result.stdout)
                    self.assertEqual(result.stdout.count("PIPELINE_CHECK n="), 18)
                    for operation, count in (
                        ("device-init", 1),
                        ("module-load id=", len(stages)),
                        ("module-unload", len(stages)),
                        ("function-load", len(stages)),
                        ("device-allocate", 4),
                        ("host-allocate", 4),
                        ("queue-create", 3),
                        ("queue-destroy", 3),
                        ("event-create", 3),
                        ("event-destroy", 3),
                        ("upload", 72),
                        ("readback", 36),
                        ("queue-wait-event", 36),
                        ("event-query", 36),
                        ("graph-complete", 18),
                        ("pipeline-bindings-pass", len(stages) * 18),
                        ("free", 8),
                    ):
                        self.assertEqual(
                            result.stderr.count(f"CPU_EVENT_SHIM {operation}"),
                            count,
                            result.stderr,
                        )
                    context_operation = (
                        "context-create" if vendor == "nvidia" else "device-select"
                    )
                    self.assertEqual(
                        result.stderr.count(f"CPU_EVENT_SHIM {context_operation}"), 1
                    )
                    self.assertGreater(
                        result.stderr.index("CPU_EVENT_SHIM module-unload"),
                        result.stderr.rindex("CPU_EVENT_SHIM graph-complete"),
                    )
                    submissions = re.findall(
                        r"CPU_EVENT_SHIM kernel-submit module=(\d+) symbol=(\w+) count=(\d+)",
                        result.stderr,
                    )
                    expected = [
                        (str(module), stage, str(count))
                        for count in (1, 127, 128, 129, 4099, 65539)
                        for _ in range(3)
                        for module, stage in enumerate(stages)
                    ]
                    self.assertEqual(submissions, expected)

    def test_failure_on_later_pipeline_stage_drains_before_releasing_resources(self):
        for runner in self.runners:
            with self.subTest(runner=runner):
                stages = ("saxpy", "relu", "saxpy")
                result = self.invoke_arguments(
                    runner,
                    self.pipeline_arguments(runner, stages),
                    modules=3,
                    failure="later-launch",
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("CPU_EVENT_SHIM INJECT later-launch", result.stderr)
                self.assertEqual(result.stderr.count("CPU_EVENT_SHIM kernel-submit"), 1)
                self.assertEqual(result.stderr.count("CPU_EVENT_SHIM module-unload"), 3)
                self.assertEqual(result.stderr.count("CPU_EVENT_SHIM free"), 8)
                self.assertEqual(result.stderr.count("CPU_EVENT_SHIM event-destroy"), 3)
                self.assertEqual(result.stderr.count("CPU_EVENT_SHIM queue-destroy"), 3)
                self.assertNotIn("CPU_EVENT_SHIM readback", result.stderr)
                self.assertNotIn("PASS backend=", result.stdout)

    def test_unknown_pipeline_completion_retains_all_pending_stages_and_allocations(
        self,
    ):
        for runner in self.runners:
            with self.subTest(runner=runner):
                stages = ("saxpy", "relu", "saxpy")
                result = self.invoke_arguments(
                    runner,
                    self.pipeline_arguments(runner, stages),
                    modules=3,
                    failure="query",
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("CPU_EVENT_SHIM INJECT query", result.stderr)
                self.assertIn("completion=unknown", result.stderr)
                self.assertEqual(result.stderr.count("CPU_EVENT_SHIM kernel-submit"), 3)
                for operation in (
                    "free",
                    "module-unload",
                    "queue-destroy",
                    "event-destroy",
                ):
                    self.assertNotIn(f"CPU_EVENT_SHIM {operation}", result.stderr)
                self.assertNotIn("PASS backend=", result.stdout)

    def test_pipeline_requires_module_mode_and_a_single_valueless_flag(self):
        for runner, (vendor, _, _) in self.runners.items():
            legacy = [
                "--payload",
                str(self.payload(vendor)),
                "--payload-format",
                "cubin" if vendor == "nvidia" else "hsaco",
            ]
            valid = self.pipeline_arguments(runner, ("saxpy", "relu"))
            for arguments in (
                ["--pipeline"],
                ["--pipeline", *legacy],
                valid + ["--pipeline"],
                valid + ["true"],
            ):
                with self.subTest(runner=runner, arguments=arguments):
                    result = self.invoke_arguments(runner, arguments, modules=2)
                    self.assertNotEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertNotIn("CPU_EVENT_SHIM", result.stderr)

    def test_pipeline_stage_limit_counts_repeated_requests(self):
        for runner in self.runners:
            with self.subTest(runner=runner):
                result = self.invoke_arguments(
                    runner, self.pipeline_arguments(runner, ("saxpy",) * 33), modules=33
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertRegex(result.stderr, r"32|[Tt]oo many")
                self.assertNotIn("CPU_EVENT_SHIM", result.stderr)


if __name__ == "__main__":
    unittest.main()
