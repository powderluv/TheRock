# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bounded native metadata queries shared by the CLI and application runtime."""

import os
from pathlib import Path
import selectors
import subprocess
import time

from .module_contract import RunnerDescription, parse_runner_description_json

_DESCRIPTION_TIMEOUT_SECONDS = 10.0
_MAX_DESCRIPTION_BYTES = 64 * 1024


def query_runner(
    runner: Path, argument: str, *, timeout: float = _DESCRIPTION_TIMEOUT_SECONDS
) -> str:
    """Read one bounded native query; only --describe-contract is offline."""
    if argument not in ("--describe-contract", "--list-devices"):
        raise ValueError("Unsupported runner query")
    label = "description" if argument == "--describe-contract" else "device inventory"
    if os.name != "posix":
        raise ValueError("Bounded runner queries currently require a POSIX host")
    deadline = time.monotonic() + timeout
    output = bytearray()
    errors = bytearray()
    with selectors.DefaultSelector() as selector:
        with subprocess.Popen(
            [str(runner), argument],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as process:
            assert process.stdout is not None and process.stderr is not None
            try:
                selector.register(process.stdout, selectors.EVENT_READ, output)
                selector.register(process.stderr, selectors.EVENT_READ, errors)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ValueError(f"Runner {label} timed out")
                    for event, _ in selector.select(remaining):
                        chunk = os.read(event.fd, 8192)
                        if not chunk:
                            selector.unregister(event.fileobj)
                            continue
                        event.data.extend(chunk)
                        if len(output) + len(errors) > _MAX_DESCRIPTION_BYTES:
                            raise ValueError(
                                f"Runner {label} exceeds 64 KiB output limit"
                            )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError(f"Runner {label} timed out")
                returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.wait()
                raise ValueError(f"Runner {label} timed out") from exc
            except BaseException:
                if process.poll() is None:
                    process.kill()
                process.wait()
                raise
    if returncode != 0:
        detail = errors.decode("utf-8", errors="replace")[:2048].strip()
        raise ValueError(f"Runner {label} failed with exit code {returncode}: {detail}")
    try:
        text = output.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Runner {label} must be UTF-8 JSON") from exc
    return text


def describe_runner(
    runner: Path, *, timeout: float = _DESCRIPTION_TIMEOUT_SECONDS
) -> RunnerDescription:
    """Read a bounded, offline description without accepting GPU support claims."""
    text = query_runner(runner, "--describe-contract", timeout=timeout)
    try:
        return parse_runner_description_json(text)
    except RecursionError as exc:
        raise ValueError("Runner description JSON is too deeply nested") from exc
