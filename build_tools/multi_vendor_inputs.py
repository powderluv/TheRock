# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Capture and verify explicitly declared imported SDK/compiler content locks."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from _therock_utils.input_provenance import (
    InputKind,
    InputSpec,
    capture_lock,
    load_spec,
    snapshot_lock,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "verify", "snapshot"):
        command = commands.add_parser(name)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--spec", type=Path, help="Schema-1 input spec JSON")
        source.add_argument(
            "--input",
            dest="input_entries",
            nargs=3,
            action="append",
            metavar=("NAME", "KIND", "PATH"),
            help="Declare a logical input name, tree/file kind, and path; repeatable",
        )
        command.add_argument(
            "--cache", type=Path, help="Optional local stat/digest cache"
        )
        if name == "snapshot":
            command.add_argument(
                "--output",
                type=Path,
                required=True,
                help="New lock path; existing files are never replaced",
            )
        else:
            command.add_argument("--lock", type=Path, required=True)
            command.add_argument(
                "--report",
                type=Path,
                help="Observed path/content summary; does not enable artifact cache reuse",
            )
        if name == "verify":
            command.add_argument(
                "--full",
                action="store_true",
                help="Hash all file contents without reusing prior stat-cache digests",
            )
    args = parser.parse_args(argv)
    try:
        inputs = (
            load_spec(args.spec)
            if args.spec is not None
            else tuple(
                InputSpec(name, cast(InputKind, kind), Path(path))
                for name, kind, path in args.input_entries
            )
        )
        if args.command == "snapshot":
            lock = snapshot_lock(inputs, args.output, cache_path=args.cache)
        else:
            lock = capture_lock(
                inputs,
                args.lock,
                cache_path=args.cache,
                report_path=args.report,
                verify_only=args.command == "verify",
                full=args.command == "verify" and args.full,
            )
        print(
            f"{args.command}: {len(lock.inputs)} imported inputs; content={lock.global_content_sha256}"
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
