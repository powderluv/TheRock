# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Assemble validation module packs from the multi-vendor profile's staged builds."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from _therock_utils.gpu_targets import GpuTarget, parse_gpu_targets
from _therock_utils.module_contract import (
    parse_runner_description_json,
    require_runner_compatibility,
    validation_contract,
)
from _therock_utils.payload_catalog import PayloadInput, create_pack
from _therock_utils.runner_registry import RunnerEntry, RunnerRegistry

MODULES = ("saxpy", "relu")


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError(f"Duplicate target-selection field: {key!r}")
        fields[key] = value
    return fields


def read_targets(path: Path) -> tuple[GpuTarget, ...]:
    document = json.loads(path.read_text(), object_pairs_hook=_unique_fields)
    if not isinstance(document, dict) or document.get("kind") != "target-selection":
        raise ValueError("Expected a target-selection document")
    if (
        type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
    ):
        raise ValueError("Unsupported target-selection schema")
    entries = document.get("targets")
    if not isinstance(entries, list) or not all(
        isinstance(entry, dict) and isinstance(entry.get("id"), str)
        for entry in entries
    ):
        raise ValueError("Target selection must contain target IDs")
    targets = parse_gpu_targets(entry["id"] for entry in entries)
    if not targets:
        raise ValueError("Native validation module packs require at least one target")
    return targets


def assemble(targets_path: Path, module_build_root: Path, output_dir: Path) -> None:
    targets = read_targets(targets_path)
    # Read all compiler outputs before touching previous build outputs.
    inputs: dict[str, list[PayloadInput]] = {module: [] for module in MODULES}
    contract = validation_contract()
    runners: list[RunnerEntry] = []
    for target in targets:
        description_path = (
            module_build_root
            / target.slug
            / "stage/bin"
            / target.slug
            / "runner-contract.json"
        )
        description = parse_runner_description_json(description_path.read_text())
        require_runner_compatibility(
            description,
            target,
            target.payload_types,
            tuple(f"therock_module_{module}" for module in MODULES),
            contract,
        )
        runner_relative = f"bin/{target.slug}/therock_module_validation"
        runner_bytes = (
            module_build_root / target.slug / "stage" / runner_relative
        ).read_bytes()
        if not runner_bytes:
            raise ValueError(f"Staged runner is empty: {runner_relative}")
        runners.append(
            RunnerEntry(
                target=target,
                path=runner_relative,
                sha256=hashlib.sha256(runner_bytes).hexdigest(),
                description=description,
            )
        )
        payload_dir = (
            module_build_root
            / target.slug
            / "stage/share/therock/modules"
            / target.slug
        )
        for module in MODULES:
            for payload_type in target.payload_types:
                inputs[module].append(
                    PayloadInput(
                        module=f"validation/{module}",
                        target=target.canonical_id,
                        payload_type=payload_type,
                        entry_points=(f"therock_module_{module}",),
                        contract=contract,
                        data=(payload_dir / f"{module}.{payload_type}").read_bytes(),
                    )
                )
    registry = RunnerRegistry(
        tuple(runners),
        tuple(f"share/therock/packs/{module}/catalog.json" for module in MODULES),
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="module-packs-", dir=output_dir.parent
    ) as temp:
        temporary = Path(temp)
        for module in MODULES:
            create_pack(temporary / module, f"validation-{module}", inputs[module])
        (temporary / "runners.json").write_text(
            json.dumps(registry.record(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # This directory is a private build output. Publish complete files only
        # after both packs and the registry are assembled; installation/artifacts
        # follow build success.
        for module in MODULES:
            destination = output_dir / module
            destination.mkdir(parents=True, exist_ok=True)
            for filename in (f"validation-{module}.kpack", "catalog.json"):
                os.replace(temporary / module / filename, destination / filename)
        os.replace(temporary / "runners.json", output_dir / "runners.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-file", type=Path, required=True)
    parser.add_argument("--module-build-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        assemble(args.targets_file, args.module_build_root, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
