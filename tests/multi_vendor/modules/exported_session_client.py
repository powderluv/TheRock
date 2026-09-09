# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Export an exact target subset and run its installed client outside the checkout.

The export and inventory verifier do not query vendor runtimes. The final isolated
consumer executes on the requested hardware and retains its existing arithmetic,
ownership, cleanup, and installed-import checks.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

REPOSITORY = Path(__file__).resolve().parents[3]
EXPORTER = REPOSITORY / "build_tools/export_multi_vendor_distribution.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dist", type=Path, required=True)
    parser.add_argument("--target", action="append", required=True)
    parser.add_argument("client_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    client_args = args.client_args
    if client_args and client_args[0] == "--":
        client_args = client_args[1:]
    if not client_args:
        parser.error("provide installed-client arguments after --")
    for argument in client_args:
        option = argument.split("=", 1)[0]
        if option.startswith("--") and "--dist-root".startswith(option):
            parser.error("the exported consumer's --dist-root is fixed by this test")
    source = args.source_dist.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="therock-target-export-") as temporary:
        scratch = Path(temporary).resolve()
        if scratch.is_relative_to(REPOSITORY) or scratch.is_relative_to(source):
            raise ValueError("select a temporary directory outside the checkout")
        output = scratch / "exported"
        command = [
            sys.executable,
            "-B",
            str(EXPORTER),
            "export",
            "--source-dist",
            str(source),
            "--output-dir",
            str(output),
        ]
        for target in args.target:
            command.extend(("--target", target))
        subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
        relocated = scratch / "relocated"
        output.rename(relocated)
        work = scratch / "working-directory"
        work.mkdir()
        verify = [
            sys.executable,
            "-B",
            str(EXPORTER),
            "verify",
            "--dist-root",
            str(relocated),
        ]
        subprocess.run(verify, check=True, cwd=work, stdout=subprocess.PIPE, text=True)
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                str(relocated / "share/therock/examples/packed_session_client.py"),
                *client_args,
                "--dist-root",
                str(relocated),
            ],
            check=True,
            cwd=work,
            stdout=subprocess.PIPE,
            text=True,
        )
        consumer = json.loads(result.stdout)
        if consumer.get("status") != "pass" or not consumer.get("runtime_origins"):
            raise ValueError(
                "installed consumer did not report verified runtime origins"
            )
        if {session["target"] for session in consumer["sessions"]} != set(args.target):
            raise ValueError("consumer did not execute exactly the exported targets")
        for paths in consumer["runtime_origins"].values():
            for origin in [paths] if isinstance(paths, str) else paths:
                if not Path(origin).resolve(strict=True).is_relative_to(relocated):
                    raise ValueError("consumer runtime escaped the relocated export")
        # A successful consumer must not change the exported file inventory.
        subprocess.run(verify, check=True, cwd=work, stdout=subprocess.PIPE, text=True)
        print(
            json.dumps(
                {
                    "selected_targets": args.target,
                    "relocated": True,
                    "verified_before_and_after_execution": True,
                    "consumer": consumer,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
