# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Consume installed packs from an isolated Python host, without checkout imports.

Run with python -I and --dist-root. The optional peer is a second simultaneously
live AMD/NVIDIA worker; intermediate arrays move through explicit host reads and
writes. This fixture does not measure performance or imply shared GPU pointers.
"""

from array import array
import argparse
from dataclasses import dataclass
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

SIZES = (1, 127, 128, 129, 4099, 65539)
ALPHAS = (0.0, -0.5, 1.25)
CAPACITY = max(SIZES) + 17
GUARD = -12345.5
TAIL_MARKER = array("f", [2.0, -3.0, 4.0])
PACKAGES = ("therock_multi_vendor", "_therock_utils", "rocm_kpack")
FORMATS = ("hsaco", "cubin", "ptx", "spirv", "mixed")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def integer(text: str) -> int:
    try:
        value = int(text, 16 if text.lower().startswith("0x") else 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected a decimal or hexadecimal integer"
        ) from exc
    if value < 0:
        raise argparse.ArgumentTypeError("Expected a nonnegative integer")
    return value


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-root", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--format", choices=FORMATS, required=True)
    parser.add_argument("--device", type=integer, default=None)
    parser.add_argument("--expect-device-id", type=integer, default=None)
    parser.add_argument("--peer-target")
    parser.add_argument("--peer-format", choices=FORMATS)
    parser.add_argument("--peer-device", type=integer, default=0)
    args = parser.parse_args(argv)
    if bool(args.peer_target) != bool(args.peer_format):
        parser.error("--peer-target and --peer-format must be specified together")
    if args.peer_target and {
        args.target.split(":")[0],
        args.peer_target.split(":")[0],
    } != {"amd", "nvidia"}:
        parser.error("Peer mode requires one AMD target and one NVIDIA target")
    for target, fmt in (
        (args.target, args.format),
        (args.peer_target, args.peer_format),
    ):
        if fmt == "mixed" and not target.startswith("nvidia:"):
            parser.error("--format mixed requires NVIDIA (SAXPY cubin, ReLU PTX)")
    return args


def runtime_origins(dist_root: Path) -> dict[str, object]:
    """Reject source-tree, symlink-escaped, or ambient copies of runtime modules."""
    origins = {}
    for name, module in tuple(sys.modules.items()):
        if not any(
            name == prefix or name.startswith(prefix + ".") for prefix in PACKAGES
        ):
            continue
        filename = getattr(module, "__file__", None)
        directories = [
            Path(directory).resolve(strict=True)
            for directory in getattr(module, "__path__", ())
        ]
        require(
            filename is not None or bool(directories),
            f"Runtime module {name} has no file or namespace origin",
        )
        for directory in directories:
            require(
                directory.is_relative_to(dist_root),
                f"Runtime package {name} has an external search path",
            )
        if filename is not None:
            origin = Path(filename).resolve(strict=True)
            require(
                origin.is_relative_to(dist_root),
                f"Runtime module {name} escaped dist: {origin}",
            )
            origins[name] = str(origin)
        else:
            origins[name] = [str(directory) for directory in directories]
    return dict(sorted(origins.items()))


def load_runtime(dist_root: Path) -> ModuleType:
    require(sys.flags.isolated == 1, "Run this consumer with python -I")
    runtime_origins(dist_root)
    sys.dont_write_bytecode = True
    python_root = (dist_root / "share/therock/python").resolve(strict=True)
    require(
        python_root.is_relative_to(dist_root), "Installed Python directory escapes dist"
    )
    # This is the only path the consumer adds. Imports deliberately follow CLI
    # parsing so no source-tree build_tools path is needed for this example.
    sys.path.insert(0, str(python_root))
    for name in PACKAGES:
        spec = importlib.util.find_spec(name)
        require(spec is not None, f"Missing installed package {name}")
        locations = (
            [spec.origin]
            if spec.origin is not None
            else list(spec.submodule_search_locations or ())
        )
        require(
            bool(locations)
            and all(
                Path(location).resolve(strict=True).is_relative_to(python_root)
                for location in locations
            ),
            f"Package {name} was not found in the installed runtime",
        )
    runtime = importlib.import_module("therock_multi_vendor")
    for name in PACKAGES[1:]:
        importlib.import_module(name)
    runtime_origins(dist_root)
    return runtime


def requests_for(runtime: ModuleType, fmt: str) -> tuple[object, ...]:
    formats = ("cubin", "ptx") if fmt == "mixed" else (fmt, fmt)
    return tuple(
        runtime.ModuleRequest(
            module="validation/" + symbol,
            payload_type=payload_type,
            entry_point="therock_module_" + symbol,
        )
        for symbol, payload_type in zip(("saxpy", "relu"), formats)
    )


def inputs(count: int, round_index: int) -> tuple[array, array]:
    # Bounded dyadic values make all three stages exact in float32, including
    # either fused or unfused multiply/add. This exposes omitted/reordered stages
    # without a tolerance hiding cancellation or host-transfer mistakes.
    x = array(
        "f", (((index * 7 + round_index * 3) % 31 - 15) / 8 for index in range(count))
    )
    y = array(
        "f", (((index * 5 + round_index * 7) % 23 - 11) / 16 for index in range(count))
    )
    return x, y


def reference(x: array, y: array, alpha: float, *, relu: bool = False) -> array:
    values = array("f", (alpha * a + b for a, b in zip(x, y, strict=True)))
    if relu:
        values = array("f", (max(0.0, value - 0.125) for value in values))
    return values


def check_array(actual: array, expected: array, label: str) -> None:
    require(actual == expected, f"{label}: caller-visible float32 values differ")


@dataclass
class Fixture:
    session: object
    saxpy: object
    relu: object
    x: object
    y: object
    a: object
    b: object

    @classmethod
    def create(cls, session: object, requests: tuple[object, ...]) -> "Fixture":
        saxpy, relu = (session.module(request) for request in requests)
        require(
            session.module(requests[0]) is saxpy,
            "Module lookup did not reuse its handle",
        )
        require(saxpy is not relu, "Distinct module requests share a handle")
        return cls(
            session, saxpy, relu, *(session.allocate(CAPACITY) for _ in range(4))
        )

    def prepare(self, x: array, y: array) -> None:
        guards = array("f", [GUARD]) * CAPACITY
        for buffer in (self.x, self.y, self.a, self.b):
            self.session.write(buffer, 0, guards)
        for buffer, values in ((self.x, x), (self.y, y)):
            self.write_input(buffer, values)
            self.session.write(buffer, CAPACITY - len(TAIL_MARKER), TAIL_MARKER)
            check_array(
                self.session.read(
                    buffer, CAPACITY - len(TAIL_MARKER), len(TAIL_MARKER)
                ),
                TAIL_MARKER,
                "nonzero transfer offset",
            )

    def write_input(self, buffer: object, values: array) -> None:
        split = len(values) // 2
        if split:
            self.session.write(buffer, 0, values[:split])
        self.session.write(buffer, split, values[split:])

    def check_output(self, buffer: object, expected: array, label: str) -> array:
        actual = self.session.read(buffer, 0, CAPACITY)
        check_array(actual[: len(expected)], expected, label)
        check_array(
            actual[len(expected) :],
            array("f", [GUARD]) * (CAPACITY - len(expected)),
            label + " full tail guard",
        )
        return actual[: len(expected)]

    def check_inputs(self, x: array, y: array) -> None:
        for buffer, expected in ((self.x, x), (self.y, y)):
            check_array(
                self.session.read(buffer, 0, len(expected)), expected, "unchanged input"
            )
            check_array(
                self.session.read(
                    buffer, CAPACITY - len(TAIL_MARKER), len(TAIL_MARKER)
                ),
                TAIL_MARKER,
                "input tail marker",
            )

    def release(self) -> None:
        self.session.synchronize()
        for buffer in (self.x, self.y, self.a, self.b):
            self.session.release(buffer)


def session_record(session: object, requests: tuple[object, ...]) -> dict[str, object]:
    return {
        "target": session.target.canonical_id,
        "device": session.device.record(),
        "device_uuid": session.device.device_uuid,
        "runner_sha256": session.runner_sha256,
        "modules": [
            {
                "module": request.module,
                "format": request.payload_type,
                "entry_point": request.entry_point,
            }
            for request in requests
        ],
        "module_handles": 2,
        "buffer_handles": 4,
    }


def check_record(count: int, round_index: int, alpha: float) -> dict[str, object]:
    return {
        "n": count,
        "round": round_index,
        "alpha": alpha,
        "stages": 3,
        "arithmetic": "exact",
        "guards": "pass",
        "offset_transfers": "pass",
    }


def run_single(fixture: Fixture) -> list[dict[str, object]]:
    checks = []
    for count in SIZES:
        for round_index, alpha in enumerate(ALPHAS):
            x, y = inputs(count, round_index)
            first = reference(x, y, alpha)
            second = reference(first, y, alpha, relu=True)
            third = reference(second, y, alpha)
            fixture.prepare(x, y)
            session = fixture.session
            session.launch(fixture.saxpy, fixture.x, fixture.y, fixture.a, alpha, count)
            session.launch(fixture.relu, fixture.a, fixture.y, fixture.b, alpha, count)
            session.launch(fixture.saxpy, fixture.b, fixture.y, fixture.a, alpha, count)
            session.synchronize()
            fixture.check_output(fixture.a, third, "single final SAXPY")
            fixture.check_output(fixture.b, second, "single retained ReLU")
            fixture.check_inputs(x, y)
            checks.append(check_record(count, round_index, alpha))
    fixture.release()
    return checks


def reject_foreign_handles(owner: Fixture, peer: Fixture) -> None:
    for label, operation in (
        ("foreign buffer", lambda: peer.session.read(owner.x, 0, 1)),
        (
            "foreign module",
            lambda: peer.session.launch(owner.saxpy, peer.x, peer.y, peer.a, 0.0, 1),
        ),
    ):
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError(label + " was not rejected locally")
    # A remote error is a ModuleServiceError, not ValueError. Both workers must
    # remain usable after these checks; no private transport internals are read.
    owner.session.synchronize()
    peer.session.synchronize()


def run_exchange(first: Fixture, peer: Fixture) -> dict[str, object]:
    reject_foreign_handles(first, peer)
    reject_foreign_handles(peer, first)
    checks = []
    for count in SIZES:
        for round_index, alpha in enumerate(ALPHAS):
            x, y = inputs(count, round_index)
            expected_first = reference(x, y, alpha)
            expected_second = reference(expected_first, y, alpha, relu=True)
            expected_third = reference(expected_second, y, alpha)
            first.prepare(x, y)
            peer.prepare(x, y)
            first.session.launch(first.saxpy, first.x, first.y, first.a, alpha, count)
            outgoing = first.check_output(
                first.a, expected_first, "first-to-peer intermediate"
            )
            peer.write_input(peer.x, outgoing)
            peer.session.launch(peer.relu, peer.x, peer.y, peer.a, alpha, count)
            incoming = peer.check_output(
                peer.a, expected_second, "peer-to-first intermediate"
            )
            first.write_input(first.b, incoming)
            first.session.launch(first.saxpy, first.b, first.y, first.a, alpha, count)
            first.check_output(first.a, expected_third, "host exchange final SAXPY")
            first.check_output(first.b, expected_second, "host exchange retained ReLU")
            peer.check_output(peer.b, array("f"), "unwritten peer scratch")
            first.check_inputs(x, y)
            peer.check_inputs(expected_first, y)
            checks.append(check_record(count, round_index, alpha))
    first.release()
    first.session.close()
    # The peer still owns its original module and buffers after the first worker
    # exits. Run a fresh launch and readback before releasing any peer resources.
    x, y = inputs(129, 2)
    peer.prepare(x, y)
    peer.session.launch(peer.saxpy, peer.x, peer.y, peer.a, ALPHAS[2], len(x))
    peer.check_output(
        peer.a, reference(x, y, ALPHAS[2]), "peer after first worker close"
    )
    peer.check_output(peer.b, array("f"), "peer guard after first worker close")
    peer.release()
    return {
        "checks": checks,
        "check_count": len(checks),
        "both_sessions_open_during_exchange": True,
        "host_transfer_directions": ["target-to-peer", "peer-to-target"],
        "foreign_handle_rejections": {"buffer": 2, "module": 2, "location": "client"},
        "peer_after_first_close": {
            "n": len(x),
            "alpha": ALPHAS[2],
            "arithmetic": "exact",
            "guards": "pass",
        },
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    dist_root = args.dist_root.resolve(strict=True)
    runtime = load_runtime(dist_root)
    requests = requests_for(runtime, args.format)
    with runtime.open_session(
        dist_root,
        args.target,
        requests,
        device_index=args.device,
        expected_device_id=args.expect_device_id,
    ) as session:
        fixture = Fixture.create(session, requests)
        sessions = [session_record(session, requests)]
        if args.peer_target:
            peer_requests = requests_for(runtime, args.peer_format)
            with runtime.open_session(
                dist_root,
                args.peer_target,
                peer_requests,
                device_index=args.peer_device,
            ) as peer_session:
                require(
                    session.device.device_uuid != peer_session.device.device_uuid,
                    "Two-vendor fixture selected the same device UUID",
                )
                peer = Fixture.create(peer_session, peer_requests)
                sessions.append(session_record(peer_session, peer_requests))
                details = run_exchange(fixture, peer)
        else:
            checks = run_single(fixture)
            details = {"checks": checks, "check_count": len(checks)}
    summary = {
        "schema_version": 1,
        "kind": "installed-packed-session-client",
        "status": "pass",
        "mode": (
            "host-mediated-two-worker-exchange" if args.peer_target else "single-worker"
        ),
        "isolated_python": True,
        "runtime_origins": runtime_origins(dist_root),
        "sessions": sessions,
        **details,
    }
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
