# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Create and extract exact-target device payload packs with integrity catalogs."""

import argparse
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from _therock_utils.gpu_targets import PayloadType, parse_gpu_target
from _therock_utils.module_contract import validation_contract
from _therock_utils.payload_catalog import PayloadInput, create_pack, extract_payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Create one .kpack and catalog.json")
    create.add_argument("--output-dir", type=Path, required=True)
    create.add_argument("--pack-id", required=True)
    create.add_argument(
        "--validation-contract",
        action="store_true",
        help="Declare the fixed validation launch ABI and write schema 2 (default: schema 1)",
    )
    create.add_argument(
        "--entry",
        nargs=5,
        action="append",
        required=True,
        metavar=("MODULE", "TARGET", "FORMAT", "SYMBOL", "FILE"),
    )
    extract = commands.add_parser("extract", help="Extract one verified exact payload")
    extract.add_argument("--catalog", type=Path, action="append", required=True)
    extract.add_argument("--module", required=True)
    extract.add_argument("--target", required=True)
    extract.add_argument(
        "--format", choices=("hsaco", "cubin", "ptx", "spirv"), required=True
    )
    extract.add_argument("--entry-point", help="Require this declared entry point")
    extract.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            inputs = [
                PayloadInput(
                    module=module,
                    target=parse_gpu_target(target).canonical_id,
                    payload_type=cast(PayloadType, payload_type),
                    entry_points=(symbol,),
                    data=Path(filename).read_bytes(),
                    contract=(
                        validation_contract() if args.validation_contract else None
                    ),
                )
                for module, target, payload_type, symbol, filename in args.entry
            ]
            create_pack(args.output_dir, args.pack_id, inputs)
        else:
            data = extract_payload(
                args.catalog,
                args.module,
                args.target,
                cast(PayloadType, args.format),
                entry_point=args.entry_point,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=args.output.parent, delete=False
                ) as file:
                    temporary = Path(file.name)
                    file.write(data)
                os.replace(temporary, args.output)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
