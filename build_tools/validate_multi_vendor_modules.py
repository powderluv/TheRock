# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate a verified pack payload through its native GPU module loader on POSIX hosts."""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path
from typing import cast

from _therock_utils.device_inventory import parse_device_uuid
from _therock_utils.gpu_targets import GpuTarget, PayloadType, parse_gpu_target
from _therock_utils.module_contract import (
    RunnerDescription,
    require_runner_compatibility,
    require_runner_capabilities,
    validate_required_capabilities,
    validation_contract,
)
from _therock_utils.payload_catalog import VerifiedPayload, extract_verified_payload
from _therock_utils.runner_query import describe_runner, query_runner
from _therock_utils.module_selection import hardware_arguments


def _parse_device_id(value: str) -> int:
    if not re.fullmatch(r"(?:[0-9]+|0[xX][0-9a-fA-F]+)", value):
        raise argparse.ArgumentTypeError("Device ID must be decimal or 0x-prefixed hex")
    device_id = int(value, 16 if value.lower().startswith("0x") else 10)
    if not 0 < device_id <= 0xFFFF:
        raise argparse.ArgumentTypeError("Device ID must be a nonzero 16-bit value")
    return device_id


def _parse_device_uuid(value: str) -> str:
    try:
        return parse_device_uuid(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def execute_payload(
    selected: VerifiedPayload,
    target: GpuTarget,
    runner: Path,
    device: int,
    expected_device_id: int | None,
    *,
    entry_point: str | None = None,
    expected_device_uuid: str | None = None,
    required_capabilities: tuple[str, ...] = (),
) -> int:
    """Execute an already verified snapshot through the guarded native path."""
    if entry_point is None:
        if len(selected.entry.entry_points) != 1:
            raise ValueError("An explicit entry point is required for this payload")
        entry_point = selected.entry.entry_points[0]
    return _execute_payloads(
        ((selected, entry_point),),
        target,
        runner,
        device,
        expected_device_id,
        expected_device_uuid=expected_device_uuid,
        required_capabilities=required_capabilities,
        session=False,
    )


def session_capabilities(values: tuple[str, ...]) -> tuple[str, ...]:
    """A session always requires resource reuse and event ordering support."""
    requested = validate_required_capabilities(values)
    return tuple(
        sorted(set(requested) | {"cross-queue-events", "multi-module-session"})
    )


def execute_module_session(
    invocations: tuple[tuple[VerifiedPayload, str], ...],
    target: GpuTarget,
    runner: Path,
    device: int,
    expected_device_id: int | None,
    *,
    expected_device_uuid: str | None = None,
    required_capabilities: tuple[str, ...] = (),
) -> int:
    """Launch verified modules together; no earlier item runs if validation fails."""
    return _execute_payloads(
        invocations,
        target,
        runner,
        device,
        expected_device_id,
        expected_device_uuid=expected_device_uuid,
        required_capabilities=session_capabilities(required_capabilities),
        session=True,
    )


def pipeline_capabilities(values: tuple[str, ...]) -> tuple[str, ...]:
    """Device pipelines require session reuse and cross-stage device dataflow."""
    return tuple(sorted(set(session_capabilities(values)) | {"device-module-pipeline"}))


def execute_module_pipeline(
    invocations: tuple[tuple[VerifiedPayload, str], ...],
    target: GpuTarget,
    runner: Path,
    device: int,
    expected_device_id: int | None,
    *,
    expected_device_uuid: str | None = None,
    required_capabilities: tuple[str, ...] = (),
) -> int:
    """Compose selected fixtures entirely on device before final readback."""
    return _execute_payloads(
        invocations,
        target,
        runner,
        device,
        expected_device_id,
        expected_device_uuid=expected_device_uuid,
        required_capabilities=pipeline_capabilities(required_capabilities),
        session=True,
        pipeline=True,
    )


def service_capabilities(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(validate_required_capabilities(values)) | {"persistent-module-service"}
        )
    )


def execute_module_service(
    invocations: tuple[tuple[VerifiedPayload, str], ...],
    target: GpuTarget,
    runner: Path,
    device: int,
    expected_device_id: int | None,
    *,
    expected_device_uuid: str | None = None,
    required_capabilities: tuple[str, ...] = (),
) -> int:
    if expected_device_uuid is None:
        raise ValueError("A module service requires a discovered device UUID")
    return _execute_payloads(
        invocations,
        target,
        runner,
        device,
        expected_device_id,
        expected_device_uuid=expected_device_uuid,
        required_capabilities=service_capabilities(required_capabilities),
        session=True,
        pipeline=True,
        service=True,
    )


def _validate_service_fixture(
    paths: tuple[Path, ...],
    invocations: tuple[tuple[VerifiedPayload, str], ...],
    target: GpuTarget,
    runner: Path,
    device: int,
    device_id: int | None,
    device_uuid: str,
    description: RunnerDescription,
) -> int:
    from array import array
    from _therock_utils.module_service import ModuleServiceError, NativeModuleSession

    capacity = 65539 + 17
    guard = -12345.5
    try:
        with NativeModuleSession(
            runner,
            target,
            device,
            device_uuid,
            device_id,
            expected_description=description,
        ) as session:
            loaded = {}
            stages = []
            for path, (selected, symbol) in zip(paths, invocations):
                key = (selected.entry.module, selected.entry.payload_type, symbol)
                if key not in loaded:
                    loaded[key] = session.load(
                        selected.entry.payload_type, symbol, path
                    )
                stages.append((loaded[key], symbol))
            x, y = session.allocate(capacity), session.allocate(capacity)
            scratch = (session.allocate(capacity), session.allocate(capacity))
            print(
                f"SERVICE target={target.canonical_id} modules={len(loaded)} stages={len(stages)} buffers=4",
                flush=True,
            )
            # Exact dyadic inputs and small stage counts exercise caller-supplied
            # values/alpha, offsets, persistent handles and on-device composition.
            # This validation command is intentionally bounded to three stages.
            for count in (1, 127, 128, 129, 4099, 65539):
                for round, alpha in enumerate((0.0, -0.5, 1.25)):
                    host_x = array(
                        "f",
                        (
                            ((i * 17 + round * 13) % 97 - 48) / 16.0
                            for i in range(count)
                        ),
                    )
                    host_y = array(
                        "f",
                        (((i * 11 + round * 7) % 89 - 44) / 32.0 for i in range(count)),
                    )
                    # Split writes include a nonzero offset whenever possible.
                    for buffer, values in ((x, host_x), (y, host_y)):
                        split = max(1, count // 2)
                        session.write(buffer, 0, values[:split])
                        if split < count:
                            session.write(buffer, split, values[split:])
                    references = [array("f", [guard]) * capacity for _ in range(2)]
                    for buffer in scratch:
                        session.write(buffer, 0, array("f", [guard]) * capacity)
                    for index, (module, symbol) in enumerate(stages):
                        destination = index % 2
                        source = x if index == 0 else scratch[1 - destination]
                        inputs = host_x if index == 0 else references[1 - destination]
                        session.launch(
                            module, source, y, scratch[destination], alpha, count
                        )
                        values = array(
                            "f", (alpha * inputs[i] + host_y[i] for i in range(count))
                        )
                        if symbol == "therock_module_relu":
                            values = array(
                                "f", (max(0.0, value - 0.125) for value in values)
                            )
                        references[destination][:count] = values
                    for index, buffer in enumerate(scratch):
                        actual = session.read(buffer, 0, capacity)
                        if actual != references[index]:
                            raise ValueError(
                                f"Service arithmetic or guard mismatch buffer={index} n={count} round={round}"
                            )
                        # Confirm an offset read has the same transport semantics.
                        if session.read(buffer, count, 17) != array("f", [guard]) * 17:
                            raise ValueError("Service offset guard read mismatch")
                    print(
                        f"SERVICE_CHECK n={count} round={round} alpha={alpha} stages={len(stages)} arithmetic=exact guard=pass",
                        flush=True,
                    )
            session.synchronize()
            for buffer in (x, y, *scratch):
                session.release(buffer)
            for module in loaded.values():
                session.unload(module)
        print(f"PASS service target={target.canonical_id}", flush=True)
    except ModuleServiceError as error:
        diagnostics = error.stderr_tail.strip()
        message = str(error)
        if diagnostics:
            message += f"\nNative worker diagnostics:\n{diagnostics}"
        raise ValueError(message) from error
    return 0


def _execute_payloads(
    invocations: tuple[tuple[VerifiedPayload, str], ...],
    target: GpuTarget,
    runner: Path,
    device: int,
    expected_device_id: int | None,
    *,
    expected_device_uuid: str | None,
    required_capabilities: tuple[str, ...],
    session: bool,
    pipeline: bool = False,
    service: bool = False,
) -> int:
    required_capabilities = validate_required_capabilities(required_capabilities)
    if service and len(invocations) > 3:
        raise ValueError(
            "The caller-data service validation fixture supports at most three stages"
        )
    if not 1 <= len(invocations) <= 32 or (not session and len(invocations) != 1):
        raise ValueError("A module session requires between 1 and 32 requests")
    if type(device) is not int or device < 0:
        raise ValueError("Device index must be nonnegative")
    hardware_checks = hardware_arguments(target, expected_device_id)
    if expected_device_uuid is not None:
        hardware_checks.extend(
            ["--expect-device-uuid", parse_device_uuid(expected_device_uuid)]
        )
    snapshots: dict[tuple[str, str, str], VerifiedPayload] = {}
    for selected, entry_point in invocations:
        if selected.entry.target != target.canonical_id:
            raise ValueError("Selected payload target differs from requested target")
        if entry_point not in selected.entry.entry_points:
            raise ValueError("Entry point is not declared by the selected payload")
        key = (selected.entry.module, selected.entry.payload_type, entry_point)
        if key in snapshots:
            if not pipeline:
                raise ValueError("Duplicate module session request")
            if snapshots[key] != selected:
                raise ValueError(
                    "Repeated pipeline stages must use the same verified snapshot"
                )
        snapshots[key] = selected
        # Every entry and its bytes come from the same verified extraction snapshot.
        if selected.entry.contract is None:
            raise ValueError("Guarded validation requires a schema 2 launch contract")
        if selected.entry.contract != validation_contract():
            raise ValueError("Unsupported validation launch contract")
    runner = runner.resolve(strict=True)
    description = describe_runner(runner)
    contract = validation_contract()
    require_runner_compatibility(
        description,
        target,
        tuple(selected.entry.payload_type for selected, _ in invocations),
        tuple(symbol for _, symbol in invocations),
        contract,
    )
    require_runner_capabilities(description, required_capabilities)
    record = contract.record()
    # Validate the complete batch before any payload process starts. Materialize
    # independent verified snapshots in one private directory held until exit.
    with tempfile.TemporaryDirectory(prefix="therock-native-module-") as temporary:
        arguments = [
            str(runner),
            "--launch-abi",
            str(record["abi"]),
            "--launch-abi-version",
            str(record["version"]),
            "--launch-contract-sha256",
            contract.sha256,
            "--device",
            str(device),
            *hardware_checks,
        ]
        paths = []
        if pipeline:
            arguments.append("--pipeline")
        for index, (selected, entry_point) in enumerate(invocations):
            payload = Path(temporary) / f"payload-{index}.{selected.entry.payload_type}"
            payload.write_bytes(selected.data)
            paths.append(payload)
            print(
                f"PACK module={selected.entry.module} target={target.canonical_id} "
                f"format={selected.entry.payload_type} bytes={len(selected.data)} verified=sha256 "
                f"abi={record['abi']} abi_version={record['version']} "
                f"contract_sha256={contract.sha256}",
                flush=True,
            )
            if session:
                arguments.extend(
                    ["--module", selected.entry.payload_type, entry_point, str(payload)]
                )
            else:
                arguments.extend(
                    [
                        "--payload-format",
                        selected.entry.payload_type,
                        "--payload",
                        str(payload),
                        "--symbol",
                        entry_point,
                    ]
                )
        if service:
            assert expected_device_uuid is not None
            return _validate_service_fixture(
                tuple(paths),
                invocations,
                target,
                runner,
                device,
                expected_device_id,
                expected_device_uuid,
                description,
            )
        result = subprocess.run(arguments, check=False)
    return result.returncode if result.returncode >= 0 else 1


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
        "--require-capability",
        action="append",
        default=[],
        help="Require a compiled adapter capability (repeatable)",
    )
    parser.add_argument(
        "--expect-device-uuid",
        type=_parse_device_uuid,
        help="Require the selected native ordinal to have this driver-reported UUID",
    )
    parser.add_argument(
        "--expect-device-id",
        type=_parse_device_id,
        help="Intel PCI device ID; xe2-b70 defaults to 0xe223, other Intel targets require it",
    )
    args = parser.parse_args()
    try:
        required_capabilities = validate_required_capabilities(args.require_capability)
        target = parse_gpu_target(args.target)
        if args.device < 0:
            raise ValueError("Device index must be nonnegative")
        hardware_arguments(target, args.expect_device_id)
        selected = extract_verified_payload(
            args.catalog,
            args.module,
            target.canonical_id,
            cast(PayloadType, args.format),
            entry_point=args.entry_point,
        )
        return execute_payload(
            selected,
            target,
            args.runner,
            args.device,
            args.expect_device_id,
            entry_point=args.entry_point,
            expected_device_uuid=args.expect_device_uuid,
            required_capabilities=required_capabilities,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
