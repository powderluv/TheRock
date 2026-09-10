# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate installed SGEMM providers and packed kernels with python -I.

This correctness fixture uses explicit host transfers between independent vendor
workers. Provider libraries and drivers must already be installed on the host.
"""

from array import array
import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType

CAPACITY = 65556
GUARD = -12345.5
EPSILON = 2.0**-24


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def bootstrap(argv: list[str]) -> tuple[ModuleType, object, Path, ModuleType]:
    # Locate the shared installed consumer bootstrap without importing any
    # checkout or ambient TheRock package. Its parser performs CLI validation.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dist-root", type=Path, required=True)
    known, _ = parser.parse_known_args(argv)
    root = known.dist_root.resolve(strict=True)
    path = (root / "share/therock/examples/packed_session_client.py").resolve(
        strict=True
    )
    require(path.is_relative_to(root), "Installed bootstrap escapes distribution")
    spec = importlib.util.spec_from_file_location("_installed_session_example", path)
    require(spec is not None and spec.loader is not None, "Missing installed bootstrap")
    helper = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    args = helper.parse_args(argv)
    return helper, args, root, helper.load_runtime(root)


@dataclass(frozen=True)
class Case:
    m: int
    n: int
    k: int
    lda: int
    ldb: int
    ldc: int
    a_offset: int = 3
    b_offset: int = 5
    c_offset: int = 7
    alpha: float = 1.0
    beta: float = 0.0

    def arguments(self) -> dict[str, int | float]:
        return dict(vars(self))


CASES = (
    Case(1, 1, 1, 1, 1, 1),
    Case(5, 7, 3, 8, 6, 9, alpha=-0.5, beta=0.75),
    Case(17, 11, 33, 19, 35, 21, alpha=1.25, beta=1.0),
    Case(7, 5, 9, 10, 12, 11, alpha=0.0, beta=-0.5),
    Case(256, 1, 256, 256, 256, 256),
    Case(1, 256, 1, 1, 1, 1, alpha=-1.0, beta=1.0),
    Case(256, 256, 1, 256, 1, 256),
)


def matrix(rows: int, columns: int, stride: int, offset: int, seed: int) -> array:
    result = array("f", [GUARD]) * CAPACITY
    for column in range(columns):
        for row in range(rows):
            result[offset + column * stride + row] = (
                ((row * 13 + column * 7 + seed * 11) % 43) - 21
            ) / 19.0
    return result


def reference(
    case: Case, a: array, b: array, c: array
) -> tuple[array, dict[int, float]]:
    expected = array("f", c)
    bounds = {}
    for column in range(case.n):
        for row in range(case.m):
            products = [
                float(a[case.a_offset + inner * case.lda + row])
                * float(b[case.b_offset + column * case.ldb + inner])
                for inner in range(case.k)
            ]
            index = case.c_offset + column * case.ldc + row
            value = case.alpha * math.fsum(products) + case.beta * c[index]
            expected[index] = value
            magnitude = abs(case.alpha) * math.fsum(map(abs, products)) + abs(
                case.beta * c[index]
            )
            # Cancellation-aware FP32 bound, including scalar operations and
            # conversion of the FP64 reference to FP32. No timing is measured.
            gamma = (case.k + 3) * EPSILON / (1 - (case.k + 3) * EPSILON)
            bounds[index] = 8 * gamma * magnitude + 2.0**-126
    return expected, bounds


def check(actual: array, expected: array, bounds: dict[int, float]) -> float:
    require(len(actual) == len(expected), "Read returned wrong buffer length")
    maximum = 0.0
    for index, (value, wanted) in enumerate(zip(actual, expected, strict=True)):
        if index in bounds:
            error = abs(value - wanted)
            require(
                math.isfinite(value) and error <= bounds[index],
                f"SGEMM mismatch at {index}: {value} versus {wanted}; bound {bounds[index]}",
            )
            maximum = max(maximum, error)
        else:
            require(value == wanted, f"SGEMM changed padding or guard at {index}")
    return maximum


@dataclass
class Fixture:
    session: object
    requests: tuple[object, ...]
    buffers: tuple[object, ...]

    @classmethod
    def create(cls, session: object, requests: tuple[object, ...]) -> "Fixture":
        return cls(
            session, requests, tuple(session.allocate(CAPACITY) for _ in range(4))
        )

    def run(self, *, kernel_interop: bool = True) -> dict[str, object]:
        provider = self.session.sgemm_provider()
        require(
            self.session.sgemm_provider() is provider,
            "Provider negotiation was not cached",
        )
        checks = []
        ba, bb, bc, _ = self.buffers
        for case in CASES:
            a = matrix(case.m, case.k, case.lda, case.a_offset, 1)
            b = matrix(case.k, case.n, case.ldb, case.b_offset, 2)
            c = matrix(case.m, case.n, case.ldc, case.c_offset, 3)
            for handle, values in ((ba, a), (bb, b), (bc, c)):
                self.session.write(handle, 0, values)
            expected, bounds = reference(case, a, b, c)
            self.session.sgemm(ba, bb, bc, **case.arguments())
            error = check(self.session.read(bc, 0, CAPACITY), expected, bounds)
            require(self.session.read(ba, 0, CAPACITY) == a, "SGEMM modified A")
            require(self.session.read(bb, 0, CAPACITY) == b, "SGEMM modified B")
            checks.append(
                {
                    **case.arguments(),
                    "maximum_absolute_error": error,
                    "maximum_absolute_bound": max(bounds.values()),
                    "guards": "pass",
                    "inputs_unchanged": True,
                }
            )
        result = {"provider": provider.record(), "checks": checks}
        if kernel_interop:
            result["kernel_interop"] = self.pipeline()
        return result

    def pipeline(self) -> dict[str, object]:
        a, b, c, d = self.buffers
        size = 8
        values = array("f", ((index % 13 - 6) / 8 for index in range(size * size)))
        zeros = array("f", [0]) * (size * size)
        guards = array("f", [GUARD]) * CAPACITY
        for handle in self.buffers:
            self.session.write(handle, 0, guards)
        self.session.write(a, 0, zeros)
        self.session.write(b, 0, values)
        self.session.write(c, 0, zeros)
        # All three operations are enqueued without a host read or synchronize.
        self.session.launch(
            self.session.module(self.requests[0]), b, a, d, 0.5, size * size
        )
        self.session.sgemm(
            d, b, c, m=size, n=size, k=size, lda=size, ldb=size, ldc=size
        )
        self.session.launch(
            self.session.module(self.requests[1]), c, a, d, 1.0, size * size
        )
        expected = array(
            "f",
            (
                max(
                    0.0,
                    sum(
                        0.5 * values[inner * size + row] * values[column * size + inner]
                        for inner in range(size)
                    )
                    - 0.125,
                )
                for column in range(size)
                for row in range(size)
            ),
        )
        actual = self.session.read(d, 0, CAPACITY)
        require(
            actual[: size * size] == expected,
            "SAXPY -> SGEMM -> ReLU stream ordering failed",
        )
        require(
            actual[size * size :] == guards[size * size :],
            "Kernel/SGEMM pipeline changed tail guards",
        )
        return {
            "operations": ["packed-saxpy", "native-sgemm", "packed-relu"],
            "intermediate_host_reads": 0,
            "arithmetic": "exact-dyadic",
            "guards": "pass",
        }


def exchange(first: Fixture, peer: Fixture) -> dict[str, object]:
    a, b, c, _ = first.buffers
    pa, pb, pc, _ = peer.buffers
    arguments = dict(m=8, n=8, k=8, lda=8, ldb=8, ldc=8)
    values = array("f", ((index % 11 - 5) / 8 for index in range(64)))
    identity = array(
        "f", (float(row == column) for column in range(8) for row in range(8))
    )
    for fixture in (first, peer):
        fixture.session.write(fixture.buffers[1], 0, identity)
    first.session.write(a, 0, values)
    first.session.sgemm(a, b, c, **arguments)
    peer.session.write(pa, 0, first.session.read(c, 0, 64))
    peer.session.sgemm(pa, pb, pc, **arguments)
    returned = peer.session.read(pc, 0, 64)
    require(returned == values, "Cross-vendor host exchange failed")
    first.session.write(a, 0, returned)
    for session, foreign, own_b, own_c in (
        (first.session, pa, b, c),
        (peer.session, a, pb, pc),
    ):
        try:
            session.sgemm(foreign, own_b, own_c, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("Foreign SGEMM buffer was accepted")
    first.session.sgemm(a, b, c, **arguments)
    first.session.close()  # Drains queued BLAS work before freeing buffers/handle.
    peer.session.sgemm(pa, pb, pc, **arguments)
    require(
        peer.session.read(pc, 0, 64) == values, "Peer failed after first worker closed"
    )
    return {
        "host_transfer_directions": ["target-to-peer", "peer-to-target"],
        "foreign_handle_rejections": 2,
        "first_close_with_queued_sgemm": True,
        "peer_after_first_close": "pass",
    }


def main(argv: list[str]) -> int:
    helper, args, root, runtime = bootstrap(argv)
    requests = helper.requests_for(runtime, args.format)
    records = []
    details = {}
    with runtime.open_session(
        root,
        args.target,
        requests,
        device_index=args.device,
        expected_device_id=args.expect_device_id,
        required_capabilities=("blas-sgemm-f32-nn-v1",),
    ) as session:
        first = Fixture.create(session, requests)
        records.append({**helper.session_record(session, requests), **first.run()})
        if args.peer_target:
            peer_requests = helper.requests_for(runtime, args.peer_format)
            with runtime.open_session(
                root,
                args.peer_target,
                peer_requests,
                device_index=args.peer_device,
                required_capabilities=("blas-sgemm-f32-nn-v1",),
            ) as peer_session:
                peer = Fixture.create(peer_session, peer_requests)
                records.append(
                    {**helper.session_record(peer_session, peer_requests), **peer.run()}
                )
                details = exchange(first, peer)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "installed-sgemm-client",
                "status": "pass",
                "isolated_python": True,
                "runtime_origins": helper.runtime_origins(root),
                "sessions": records,
                **details,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
