# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run installed SGEMM sessions without selecting or loading packed kernels.

Use python -I, --dist-root, and an exact --target. An optional AMD/NVIDIA peer
exchanges matrices through host memory while both workers remain open.
"""

import argparse
from dataclasses import dataclass
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType


def nonnegative_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a nonnegative integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("Expected a nonnegative integer")
    return number


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-root", type=Path, required=True)
    parser.add_argument("--target", required=True)
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--device", type=nonnegative_integer)
    device.add_argument("--device-uuid")
    parser.add_argument("--peer-target")
    peer = parser.add_mutually_exclusive_group()
    peer.add_argument("--peer-device", type=nonnegative_integer)
    peer.add_argument("--peer-device-uuid")
    args = parser.parse_args(argv)
    if not args.peer_target and (
        args.peer_device is not None or args.peer_device_uuid is not None
    ):
        parser.error("Peer device selection requires --peer-target")
    if args.peer_target and {
        args.target.split(":")[0],
        args.peer_target.split(":")[0],
    } != {"amd", "nvidia"}:
        parser.error("Peer mode requires one AMD target and one NVIDIA target")
    return args


@dataclass(frozen=True)
class InstalledRuntime:
    helper: ModuleType
    args: argparse.Namespace
    root: Path
    runtime: ModuleType


def bootstrap(argv: list[str]) -> InstalledRuntime:
    args = parse_args(argv)
    if sys.flags.isolated != 1:
        raise ValueError("Run this consumer with python -I")
    root = args.dist_root.resolve(strict=True)
    path = (root / "share/therock/examples/packed_session_client.py").resolve(
        strict=True
    )
    if not path.is_relative_to(root):
        raise ValueError("Installed bootstrap escapes distribution")
    spec = importlib.util.spec_from_file_location("_installed_provider_bootstrap", path)
    if spec is None or spec.loader is None:
        raise ValueError("Missing installed bootstrap")
    helper = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    return InstalledRuntime(helper, args, root, helper.load_runtime(root))


def load_numerics(root: Path) -> ModuleType:
    name = "therock_multi_vendor.sgemm_example"
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise ValueError("Missing installed SGEMM numerical example")
    if not Path(spec.origin).resolve(strict=True).is_relative_to(root):
        raise ValueError("SGEMM numerical example escaped distribution")
    return importlib.import_module(name)


def session_record(session: object) -> dict[str, object]:
    return {
        "target": session.target.canonical_id,
        "device": session.device.record(),
        "device_uuid": session.device.device_uuid,
        "runner_sha256": session.runner_sha256,
        "modules_loaded": 0,
        "buffer_handles": 4,
    }


def main(argv: list[str]) -> int:
    installed = bootstrap(argv)
    helper, args = installed.helper, installed.args
    root, runtime = installed.root, installed.runtime
    numerical = load_numerics(root)
    helper.runtime_origins(root)
    records = []
    details = {}
    with runtime.open_sgemm_session(
        root, args.target, device_index=args.device, device_uuid=args.device_uuid
    ) as session:
        first = numerical.Fixture.create(session, ())
        records.append({**session_record(session), **first.run(kernel_interop=False)})
        if args.peer_target:
            with runtime.open_sgemm_session(
                root,
                args.peer_target,
                device_index=args.peer_device,
                device_uuid=args.peer_device_uuid,
            ) as peer_session:
                peer = numerical.Fixture.create(peer_session, ())
                records.append(
                    {**session_record(peer_session), **peer.run(kernel_interop=False)}
                )
                details = numerical.exchange(first, peer)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "installed-sgemm-provider-client",
                "status": "pass",
                "isolated_python": True,
                "modules_loaded": 0,
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
