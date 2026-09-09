# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU checks for the standalone installed-runtime consumer fixture."""

from array import array
from contextlib import redirect_stderr
from fractions import Fraction
import importlib.util
import io
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tests/multi_vendor/modules/packed_session_client.py"
)
SPEC = importlib.util.spec_from_file_location("packed_session_client_fixture", SCRIPT)
fixture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fixture
SPEC.loader.exec_module(fixture)


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


class CpuSession:
    """Deferred CPU operations retain buffers until read/sync, with owner checks."""

    def __init__(self) -> None:
        self.closed = False
        self.modules = {"saxpy": object(), "relu": object()}
        self.buffers: list[array] = []
        self.pending = []
        self.allocations = 0
        self.launches = 0

    def module(self, request: str) -> object:
        return self.modules[request]

    def allocate(self, capacity: int) -> array:
        result = array("f", [0.0]) * capacity
        self.buffers.append(result)
        self.allocations += 1
        return result

    def check_buffer(self, buffer: array, offset: int, count: int) -> None:
        if self.closed or not any(buffer is value for value in self.buffers):
            raise ValueError("Foreign, released, or closed buffer")
        if offset < 0 or count < 1 or offset + count > len(buffer):
            raise ValueError("Invalid range")

    def write(self, buffer: array, offset: int, values: array) -> None:
        self.check_buffer(buffer, offset, len(values))
        self.synchronize()
        buffer[offset : offset + len(values)] = values

    def read(self, buffer: array, offset: int, count: int) -> array:
        self.check_buffer(buffer, offset, count)
        self.synchronize()
        return buffer[offset : offset + count]

    def launch(
        self,
        module: object,
        x: array,
        y: array,
        output: array,
        alpha: float,
        count: int,
    ) -> None:
        if self.closed or not any(module is value for value in self.modules.values()):
            raise ValueError("Foreign or closed module")
        for buffer in (x, y, output):
            self.check_buffer(buffer, 0, count)
        if output is x or output is y:
            raise ValueError("Aliased output")

        def execute() -> None:
            for index in range(count):
                value = float32(float32(alpha * x[index]) + y[index])
                if module is self.modules["relu"]:
                    value = max(0.0, float32(value - 0.125))
                output[index] = value

        self.pending.append(execute)
        self.launches += 1

    def synchronize(self) -> None:
        if self.closed:
            raise ValueError("Closed worker")
        for operation in self.pending:
            operation()
        self.pending.clear()

    def release(self, buffer: array) -> None:
        self.check_buffer(buffer, 0, 1)
        self.synchronize()
        self.buffers = [value for value in self.buffers if value is not buffer]

    def close(self) -> None:
        self.synchronize()
        self.buffers.clear()
        self.closed = True


class PackedSessionClientTest(unittest.TestCase):
    def test_arguments_and_exact_dyadic_reference(self) -> None:
        args = fixture.parse_args(
            [
                "--dist-root",
                ".",
                "--target",
                "intel:level-zero:xe2-b70",
                "--format",
                "spirv",
                "--device",
                "1",
                "--expect-device-id",
                "0xe223",
            ]
        )
        self.assertEqual(args.device, 1)
        self.assertEqual(args.expect_device_id, 0xE223)
        for extra in (
            ["--peer-target", "amd:hip:gfx1201"],
            ["--format", "mixed"],
            ["--peer-target", "intel:level-zero:xe2-b70", "--peer-format", "spirv"],
        ):
            with self.subTest(extra=extra), redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                fixture.parse_args(
                    [
                        "--dist-root",
                        ".",
                        "--target",
                        "amd:hip:gfx1201",
                        "--format",
                        "hsaco",
                        *extra,
                    ]
                )
        runtime = SimpleNamespace(ModuleRequest=lambda **kwargs: kwargs)
        requests = fixture.requests_for(runtime, "mixed")
        self.assertEqual(
            [request["payload_type"] for request in requests], ["cubin", "ptx"]
        )
        for round_index, alpha in enumerate(fixture.ALPHAS):
            x, y = fixture.inputs(713, round_index)
            exact = [Fraction(value) for value in x]
            for relu in (False, True, False):
                x = fixture.reference(x, y, alpha, relu=relu)
                exact = [
                    Fraction(alpha) * value + Fraction(other)
                    for value, other in zip(exact, y)
                ]
                if relu:
                    exact = [
                        max(Fraction(0), value - Fraction(1, 8)) for value in exact
                    ]
                self.assertEqual([Fraction(value) for value in x], exact)
        with self.assertRaisesRegex(AssertionError, "differ"):
            fixture.check_array(
                array("f", [1.0]), array("f", [2.0]), "corrupted readback"
            )

    def test_single_and_two_live_worker_dataflow(self) -> None:
        single = CpuSession()
        checks = fixture.run_single(fixture.Fixture.create(single, ("saxpy", "relu")))
        self.assertEqual(len(checks), 18)
        self.assertEqual(single.allocations, 4)
        self.assertEqual(single.launches, 54)
        self.assertEqual(single.buffers, [])
        first, peer = CpuSession(), CpuSession()
        result = fixture.run_exchange(
            fixture.Fixture.create(first, ("saxpy", "relu")),
            fixture.Fixture.create(peer, ("saxpy", "relu")),
        )
        self.assertEqual(result["check_count"], 18)
        self.assertEqual(
            result["foreign_handle_rejections"],
            {"buffer": 2, "module": 2, "location": "client"},
        )
        self.assertTrue(first.closed)
        self.assertFalse(peer.closed)
        self.assertEqual((first.allocations, peer.allocations), (4, 4))
        self.assertEqual((first.launches, peer.launches), (36, 19))
        self.assertEqual(first.buffers, [])
        self.assertEqual(peer.buffers, [])
        peer.close()

    def test_isolated_installed_import_origins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dist"
            python_root = root / "share/therock/python"
            python_root.mkdir(parents=True)
            (python_root / "therock_multi_vendor.py").write_text(
                "# fake public runtime\n"
            )
            (python_root / "_therock_utils").mkdir()
            (python_root / "rocm_kpack").mkdir()
            (python_root / "rocm_kpack/__init__.py").write_text("# fake kpack\n")
            command = "import runpy, pathlib, sys; f=runpy.run_path(sys.argv[1]); f['load_runtime'](pathlib.Path(sys.argv[2])); print(sorted(f['runtime_origins'](pathlib.Path(sys.argv[2]))))"
            result = subprocess.run(
                [sys.executable, "-I", "-c", command, str(SCRIPT), str(root)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("_therock_utils", result.stdout)
            self.assertEqual(list(root.rglob("__pycache__")), [])
            outside = Path(temporary) / "outside.py"
            outside.write_text("# source-tree stand-in\n")
            (python_root / "therock_multi_vendor.py").unlink()
            (python_root / "therock_multi_vendor.py").symlink_to(outside)
            result = subprocess.run(
                [sys.executable, "-I", "-c", command, str(SCRIPT), str(root)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not found in the installed runtime", result.stderr)


if __name__ == "__main__":
    unittest.main()
