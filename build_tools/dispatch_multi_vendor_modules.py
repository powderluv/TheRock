# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Discover native devices and dispatch exact packed fixture targets on POSIX hosts."""

import argparse
import json
from pathlib import Path
from typing import cast

from _therock_utils.device_inventory import select_device
from _therock_utils.gpu_targets import PayloadType
from _therock_utils.module_contract import validate_required_capabilities
from _therock_utils.module_selection import (
    REGISTRY_PATH as _REGISTRY_PATH,
    ModuleRequest,
    prepare_modules,
    query_inventory as _query_inventory,
)
from _therock_utils.runner_registry import (
    RunnerRegistry,
    load_registry,
    resolve_registry_path,
    verify_runner,
)
from validate_multi_vendor_modules import (
    _parse_device_id,
    _parse_device_uuid,
    execute_payload,
    execute_module_session,
    execute_module_pipeline,
    execute_module_service,
    service_capabilities,
    pipeline_capabilities,
    session_capabilities,
)


def list_devices(root: Path, registry: RunnerRegistry) -> dict[str, object]:
    # Validate every executable before asking any of them to run. An unavailable
    # installed driver is distinct from a corrupt or incomplete distribution.
    verified = [(entry, verify_runner(root, entry)) for entry in registry.runners]
    results: list[dict[str, object]] = []
    for entry, runner in verified:
        try:
            inventory = _query_inventory(runner, entry)
            matches = []
            for device in inventory.devices:
                try:
                    select_device(inventory, entry.target, device.index)
                except ValueError:
                    continue
                matches.append(device.index)
            results.append(
                {
                    "target": entry.target.canonical_id,
                    "status": "available",
                    "matching_device_indices": matches,
                    "inventory": inventory.record(),
                    "error": None,
                }
            )
        except (ValueError, OSError) as exc:
            results.append(
                {
                    "target": entry.target.canonical_id,
                    "status": "unavailable",
                    "matching_device_indices": [],
                    "inventory": None,
                    "error": str(exc),
                }
            )
    return {
        "schema_version": 2,
        "kind": "multi-vendor-device-inventory",
        "runners": results,
    }


def run_module(
    root: Path,
    registry: RunnerRegistry,
    *,
    module: str,
    target_id: str,
    payload_type: PayloadType,
    entry_point: str,
    device_index: int | None,
    expected_device_id: int | None,
    device_uuid: str | None = None,
    required_capabilities: tuple[str, ...] = (),
) -> int:
    return run_modules(
        root,
        registry,
        requests=(ModuleRequest(module, payload_type, entry_point),),
        target_id=target_id,
        device_index=device_index,
        expected_device_id=expected_device_id,
        device_uuid=device_uuid,
        required_capabilities=required_capabilities,
        session=False,
    )


def run_modules(
    root: Path,
    registry: RunnerRegistry,
    *,
    requests: tuple[ModuleRequest, ...],
    target_id: str,
    device_index: int | None,
    expected_device_id: int | None,
    device_uuid: str | None = None,
    required_capabilities: tuple[str, ...] = (),
    session: bool = True,
    pipeline: bool = False,
    service: bool = False,
) -> int:
    if service and (not session or len(requests) > 3):
        raise ValueError(
            "Service validation requires a session with one to three stages"
        )
    if pipeline and not session:
        raise ValueError("A device pipeline requires module session mode")
    required_capabilities = (
        service_capabilities(required_capabilities)
        if service
        else (
            pipeline_capabilities(required_capabilities)
            if pipeline
            else (
                session_capabilities(required_capabilities)
                if session
                else validate_required_capabilities(required_capabilities)
            )
        )
    )
    if not 1 <= len(requests) <= 32 or (not session and len(requests) != 1):
        raise ValueError("A module session requires between 1 and 32 requests")
    if not (pipeline or service) and len(set(requests)) != len(requests):
        raise ValueError("Duplicate module session request")
    prepared = prepare_modules(
        root,
        registry,
        requests=requests,
        target_id=target_id,
        device_index=device_index,
        device_uuid=device_uuid,
        expected_device_id=expected_device_id,
        required_capabilities=required_capabilities,
    )
    target, entry, runner, device = (
        prepared.target,
        prepared.entry,
        prepared.runner,
        prepared.device,
    )
    invocations = prepared.invocations
    dispatch: dict[str, object] = {
        "target": target.canonical_id,
        "runner_sha256": entry.sha256,
        "required_capabilities": list(required_capabilities),
        "device": device.record(),
    }
    if session:
        dispatch["mode"] = (
            "persistent-module-service"
            if service
            else "device-module-pipeline" if pipeline else "multi-module-session"
        )
        dispatch["modules"] = [
            {
                "module": request.module,
                "format": request.payload_type,
                "entry_point": request.entry_point,
            }
            for request in requests
        ]
    else:
        dispatch.update(module=requests[0].module, format=requests[0].payload_type)
    print("DISPATCH " + json.dumps(dispatch, sort_keys=True), flush=True)
    if service:
        return execute_module_service(
            tuple(invocations),
            target,
            runner,
            device.index,
            expected_device_id,
            expected_device_uuid=device.device_uuid,
            required_capabilities=required_capabilities,
        )
    if pipeline:
        return execute_module_pipeline(
            tuple(invocations),
            target,
            runner,
            device.index,
            expected_device_id,
            expected_device_uuid=device.device_uuid,
            required_capabilities=required_capabilities,
        )
    if session:
        return execute_module_session(
            tuple(invocations),
            target,
            runner,
            device.index,
            expected_device_id,
            expected_device_uuid=device.device_uuid,
            required_capabilities=required_capabilities,
        )
    selected, symbol = invocations[0]
    return execute_payload(
        selected,
        target,
        runner,
        device.index,
        expected_device_id,
        entry_point=symbol,
        expected_device_uuid=device.device_uuid,
        required_capabilities=required_capabilities,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "list",
        help="Report observed devices or driver errors for every registered runner",
    )
    run = commands.add_parser(
        "run", help="Select one exact target and execute a contracted fixture"
    )
    run.add_argument("--module", required=True)
    run.add_argument(
        "--format", choices=("hsaco", "cubin", "ptx", "spirv"), required=True
    )
    run.add_argument("--entry-point", required=True)
    batch = commands.add_parser(
        "run-batch", help="Run exact modules from multiple packs in one native session"
    )
    pipeline = commands.add_parser(
        "run-pipeline",
        help="Feed packed kernel outputs into later stages on the device",
    )
    service = commands.add_parser(
        "run-service",
        help="Validate caller-provided data through persistent module/buffer handles",
    )
    for command in (batch, pipeline, service):
        command.add_argument(
            "--module",
            nargs=3,
            action="append",
            required=True,
            metavar=("MODULE", "FORMAT", "ENTRY_POINT"),
            help=(
                "One to three caller-data fixture stages"
                if command is service
                else "An exact module request (up to 32; stages may repeat in pipelines)"
            ),
        )
    for command in (run, batch, pipeline, service):
        command.add_argument("--target", required=True)
        command.add_argument(
            "--require-capability",
            action="append",
            default=[],
            help="Require a compiled adapter capability (repeatable)",
        )
        device_selection = command.add_mutually_exclusive_group()
        device_selection.add_argument(
            "--device", type=int, help="Process-local ordinal (default: 0)"
        )
        device_selection.add_argument(
            "--device-uuid",
            type=_parse_device_uuid,
            help="Select a driver UUID reported by list, preserving exact target identity",
        )
        command.add_argument("--expect-device-id", type=_parse_device_id)
    args = parser.parse_args()
    try:
        root = args.dist_root.resolve(strict=True)
        registry = load_registry(resolve_registry_path(root, _REGISTRY_PATH))
        if args.command == "list":
            print(json.dumps(list_devices(root, registry), indent=2))
            return 0
        if args.command in ("run-batch", "run-pipeline", "run-service"):
            return run_modules(
                root,
                registry,
                requests=tuple(
                    ModuleRequest(module, cast(PayloadType, fmt), symbol)
                    for module, fmt, symbol in args.module
                ),
                target_id=args.target,
                device_index=args.device,
                device_uuid=args.device_uuid,
                expected_device_id=args.expect_device_id,
                required_capabilities=tuple(args.require_capability),
                pipeline=args.command == "run-pipeline",
                service=args.command == "run-service",
            )
        return run_module(
            root,
            registry,
            module=args.module,
            target_id=args.target,
            payload_type=cast(PayloadType, args.format),
            entry_point=args.entry_point,
            device_index=args.device,
            device_uuid=args.device_uuid,
            expected_device_id=args.expect_device_id,
            required_capabilities=tuple(args.require_capability),
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
