# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate a verified pack payload through the target's native GPU module loader."""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path
from typing import cast

from _therock_utils.gpu_targets import GpuTarget, PayloadType, parse_gpu_target
from _therock_utils.payload_catalog import extract_payload


def _parse_device_id(value: str) -> int:
    if not re.fullmatch(r"(?:[0-9]+|0[xX][0-9a-fA-F]+)", value):
        raise argparse.ArgumentTypeError("Device ID must be decimal or 0x-prefixed hex")
    device_id = int(value, 16 if value.lower().startswith("0x") else 10)
    if not 0 < device_id <= 0xFFFF:
        raise argparse.ArgumentTypeError("Device ID must be a nonzero 16-bit value")
    return device_id


def hardware_arguments(target: GpuTarget, expected_device_id: int | None) -> list[str]:
    """Build runtime checks without treating a SPIR-V format as hardware identity."""
    if target.vendor == "intel":
        if target.processor == "xe2-b70":
            if expected_device_id not in (None, 0xE223):
                raise ValueError("Intel xe2-b70 requires --expect-device-id 0xe223")
            expected_device_id = 0xE223
        elif expected_device_id is None:
            raise ValueError(
                f"Intel target {target.canonical_id} requires an explicit --expect-device-id"
            )
        # The native Level Zero runner separately requires Intel vendor ID 0x8086.
        return ["--expect-arch", "spirv", "--expect-device-id", hex(expected_device_id)]
    if expected_device_id is not None:
        raise ValueError("--expect-device-id is only supported for Intel targets")
    # Runtime properties report CUDA major/minor, not compiler suffix letters.
    expected_arch = (
        re.sub(r"[a-z]$", "", target.processor)
        if target.vendor == "nvidia"
        else target.processor
    )
    return ["--expect-arch", expected_arch]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, action="append", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--format", choices=("hsaco", "cubin", "ptx", "spirv"), required=True
    )
    parser.add_argument("--entry-point", required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--expect-device-id",
        type=_parse_device_id,
        help="Intel PCI device ID; xe2-b70 defaults to 0xe223, other Intel targets require it",
    )
    args = parser.parse_args()
    try:
        target = parse_gpu_target(args.target)
        if args.device < 0:
            raise ValueError("Device index must be nonnegative")
        hardware_checks = hardware_arguments(target, args.expect_device_id)
        data = extract_payload(
            args.catalog,
            args.module,
            target.canonical_id,
            cast(PayloadType, args.format),
            entry_point=args.entry_point,
        )
        # Verify catalog identity and bytes before any GPU process is started.
        with tempfile.TemporaryDirectory(prefix="therock-native-module-") as temporary:
            payload = Path(temporary) / f"payload.{args.format}"
            payload.write_bytes(data)
            print(
                f"PACK module={args.module} target={target.canonical_id} "
                f"format={args.format} bytes={len(data)} verified=sha256",
                flush=True,
            )
            result = subprocess.run(
                [
                    str(args.runner.resolve()),
                    "--payload",
                    str(payload),
                    "--symbol",
                    args.entry_point,
                    "--device",
                    str(args.device),
                    *hardware_checks,
                ],
                check=False,
            )
        return result.returncode if result.returncode >= 0 else 1
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
