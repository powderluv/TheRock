# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU validation of the installed SGEMM example's oracle and dataflow."""

from array import array
from fractions import Fraction
import importlib.util
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "therock_multi_vendor/sgemm_example.py"
SPEC = importlib.util.spec_from_file_location("sgemm_example_fixture", SCRIPT)
fixture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fixture
SPEC.loader.exec_module(fixture)


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


class CpuSession:
    """Independent FP32 executor; queued work runs only at host transfer or close."""

    def __init__(self, *, reverse_work: bool = False) -> None:
        self.closed = False
        self.reverse_work = reverse_work
        self.buffers = []
        self.pending = []
        self.trace = []
        self.drains = []
        self.close_pending = None
        self.foreign_rejections = 0
        self.provider = SimpleNamespace(record=lambda: {"provider": "cpu-test"})

    def allocate(self, capacity: int) -> array:
        result = array("f", [0.0]) * capacity
        self.buffers.append(result)
        return result

    def validate(self, *buffers: array) -> None:
        if self.closed:
            raise ValueError("Closed worker")
        if any(not any(b is owned for owned in self.buffers) for b in buffers):
            self.foreign_rejections += 1
            raise ValueError("Foreign buffer")

    def sgemm_provider(self) -> object:
        return self.provider

    def module(self, request: str) -> str:
        if request not in ("saxpy", "relu"):
            raise ValueError("Unknown module")
        return request

    def write(self, buffer: array, offset: int, values: array) -> None:
        self.validate(buffer)
        self.trace.append("write")
        self.synchronize()
        buffer[offset : offset + len(values)] = values

    def read(self, buffer: array, offset: int, count: int) -> array:
        self.validate(buffer)
        self.trace.append("read")
        self.synchronize()
        return buffer[offset : offset + count]

    def launch(
        self, module: str, x: array, y: array, out: array, alpha: float, count: int
    ) -> None:
        self.validate(x, y, out)

        def execute() -> None:
            for index in range(count):
                value = float32(float32(alpha * x[index]) + y[index])
                if module == "relu":
                    value = max(0.0, float32(value - 0.125))
                out[index] = value

        self.trace.append(f"launch:{module}")
        self.pending.append((f"launch:{module}", execute))

    def sgemm(
        self,
        a: array,
        b: array,
        c: array,
        *,
        m: int,
        n: int,
        k: int,
        lda: int,
        ldb: int,
        ldc: int,
        a_offset: int = 0,
        b_offset: int = 0,
        c_offset: int = 0,
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> None:
        self.validate(a, b, c)

        def execute() -> None:
            # Slice logical rows/columns independently of the example's FP64
            # products oracle; round each multiply and accumulation to FP32.
            for row in range(m):
                row_start = a_offset + row
                a_row = a[row_start : row_start + k * lda : lda]
                for column in range(n):
                    column_start = b_offset + column * ldb
                    b_column = b[column_start : column_start + k]
                    total = 0.0
                    for left, right in zip(a_row, b_column, strict=True):
                        total = float32(total + float32(left * right))
                    output = c_offset + row + column * ldc
                    c[output] = float32(
                        float32(alpha * total) + float32(beta * c[output])
                    )

        self.trace.append("sgemm")
        self.pending.append(("sgemm", execute))

    def synchronize(self) -> None:
        if self.pending:
            self.drains.append(tuple(label for label, _ in self.pending))
            work = reversed(self.pending) if self.reverse_work else self.pending
            for _, operation in work:
                operation()
            self.pending.clear()

    def close(self) -> None:
        self.close_pending = len(self.pending)
        self.trace.append("close")
        self.synchronize()
        self.buffers.clear()
        self.closed = True


class SgemmExampleTest(unittest.TestCase):
    def test_column_major_offsets_padding_and_hand_computed_result(self) -> None:
        case = fixture.Case(2, 2, 3, 4, 5, 6, a_offset=2, b_offset=3, c_offset=4)
        a, b, c = (array("f", [fixture.GUARD]) * 30 for _ in range(3))
        # A=[[1,2,3],[4,5,6]], B=[[7,8],[9,10],[11,12]].
        for index, value in zip((2, 3, 6, 7, 10, 11), (1, 4, 2, 5, 3, 6)):
            a[index] = value
        for index, value in zip((3, 4, 5, 8, 9, 10), (7, 9, 11, 8, 10, 12)):
            b[index] = value
        expected, bounds = fixture.reference(case, a, b, c)
        self.assertEqual(set(bounds), {4, 5, 10, 11})
        self.assertEqual([expected[i] for i in (4, 5, 10, 11)], [58, 139, 64, 154])
        self.assertTrue(
            all(value == c[i] for i, value in enumerate(expected) if i not in bounds)
        )
        generated = fixture.matrix(2, 3, 4, 2, 1)
        live = {2, 3, 6, 7, 10, 11}
        self.assertEqual(
            {i for i, value in enumerate(generated) if value != fixture.GUARD}, live
        )
        self.assertEqual(generated[7], float32(10 / 19))
        # Include both scalar operations against an exact rational result.
        scaled = fixture.Case(2, 2, 3, 4, 5, 6, 2, 3, 4, -0.5, 0.75)
        for index in bounds:
            c[index] = 8.0
        expected, _ = fixture.reference(scaled, a, b, c)
        self.assertEqual([expected[i] for i in (4, 5, 10, 11)], [-23, -63.5, -26, -71])

    def test_tolerance_is_cancellation_aware_and_rejects_corruption(self) -> None:
        case = fixture.Case(1, 1, 3, 1, 3, 1, 0, 0, 0)
        a = array("f", [2**24, 1, -(2**24)])
        b = array("f", [1, 1, 1])
        expected, bounds = fixture.reference(case, a, b, array("f", [0]))
        exact = sum(Fraction(x) * Fraction(y) for x, y in zip(a, b))
        self.assertEqual(expected[0], float(exact))
        # A valid serial FP32 accumulation loses the middle term. A relative
        # tolerance based only on the final value would wrongly reject it.
        serial = 0.0
        for left, right in zip(a, b):
            serial = float32(serial + float32(left * right))
        self.assertEqual(serial, 0.0)
        self.assertEqual(fixture.check(array("f", [serial]), expected, bounds), 1.0)
        wanted = array("f", [fixture.GUARD, 1.0, fixture.GUARD, -2.0, fixture.GUARD])
        accepted = array("f", wanted)
        accepted[1] += 2**-20
        limits = {1: 2**-20, 3: 2**-20}
        self.assertEqual(fixture.check(accepted, wanted, limits), 2**-20)
        for value in (1 + 2**-19, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(
                AssertionError, "mismatch"
            ):
                actual = array("f", wanted)
                actual[1] = value
                fixture.check(actual, wanted, limits)
        for index in (0, 2, 4):
            with self.subTest(guard=index), self.assertRaisesRegex(
                AssertionError, "padding or guard"
            ):
                actual = array("f", wanted)
                actual[index] = 0.0
                fixture.check(actual, wanted, limits)
        with self.assertRaisesRegex(AssertionError, "wrong buffer length"):
            fixture.check(wanted[:-1], wanted, limits)

    def test_all_matrix_cases_accept_independent_fp32_execution(self) -> None:
        session = CpuSession()
        result = fixture.Fixture.create(session, ("saxpy", "relu")).run()
        self.assertEqual(len(session.buffers), 4)
        self.assertEqual(len(result["checks"]), 7)
        self.assertEqual(
            [record["m"] for record in result["checks"]], [1, 5, 17, 7, 256, 1, 256]
        )
        self.assertTrue(
            all(
                record["guards"] == "pass" and record["inputs_unchanged"]
                for record in result["checks"]
            )
        )
        self.assertEqual(
            session.drains,
            [("sgemm",)] * 7 + [("launch:saxpy", "sgemm", "launch:relu")],
        )
        self.assertEqual(
            session.trace[-4:], ["launch:saxpy", "sgemm", "launch:relu", "read"]
        )

    def test_pipeline_exact_dyadic_oracle_and_ordering_failure(self) -> None:
        session = CpuSession()
        item = fixture.Fixture.create(session, ("saxpy", "relu"))
        result = item.pipeline()
        values = [Fraction(index % 13 - 6, 8) for index in range(64)]
        exact = [
            max(
                Fraction(0),
                sum(
                    values[row + 8 * inner] * values[inner + 8 * column] / 2
                    for inner in range(8)
                )
                - Fraction(1, 8),
            )
            for column in range(8)
            for row in range(8)
        ]
        self.assertEqual([Fraction(value) for value in item.buffers[3][:64]], exact)
        self.assertEqual(result["arithmetic"], "exact-dyadic")
        self.assertEqual(session.drains, [("launch:saxpy", "sgemm", "launch:relu")])
        with self.assertRaisesRegex(AssertionError, "stream ordering failed"):
            fixture.Fixture.create(
                CpuSession(reverse_work=True), ("saxpy", "relu")
            ).pipeline()

    def test_peer_exchange_rejects_foreign_handles_and_drains_before_close(
        self,
    ) -> None:
        first, peer = CpuSession(), CpuSession()
        result = fixture.exchange(
            fixture.Fixture.create(first, ("saxpy", "relu")),
            fixture.Fixture.create(peer, ("saxpy", "relu")),
        )
        self.assertEqual(
            result["host_transfer_directions"], ["target-to-peer", "peer-to-target"]
        )
        self.assertEqual((first.foreign_rejections, peer.foreign_rejections), (1, 1))
        self.assertTrue(first.closed)
        self.assertEqual(first.close_pending, 1)
        self.assertEqual(first.buffers, [])
        self.assertFalse(peer.closed)
        self.assertEqual(peer.pending, [])
        self.assertEqual(first.drains, [("sgemm",), ("sgemm",)])
        self.assertEqual(peer.drains, [("sgemm",), ("sgemm",)])
        self.assertEqual(peer.trace[-2:], ["sgemm", "read"])
        self.assertEqual(result["peer_after_first_close"], "pass")


if __name__ == "__main__":
    unittest.main()
