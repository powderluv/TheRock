# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verified packed-module sessions for the experimental FP32 vector contract.

Use ``with open_session(...) as session``. Every module is selected and verified
before device discovery, and loaded before the session is returned. Calls on a
session must be serialized. Integrity is relative to supplied distribution
metadata; publication must be serialized with use. No target or format fallback
is performed. The low-level trusted-path client is intentionally not exposed.
"""

from array import array
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
import tempfile

from _therock_utils.device_inventory import ObservedDevice
from _therock_utils.gpu_targets import GpuTarget
from _therock_utils.module_contract import validate_required_capabilities
from _therock_utils.module_selection import (
    REGISTRY_PATH,
    ModuleRequest,
    prepare_modules,
)
from _therock_utils.module_service import (
    Buffer,
    Module,
    ModuleServiceError,
    ModuleServiceProtocolError,
    ModuleServiceTimeoutError,
    ModuleServiceRemoteError,
    NativeModuleSession as _NativeModuleSession,
)
from _therock_utils.runner_registry import (
    load_registry,
    resolve_registry_path,
    verify_runner,
)

__all__ = [
    "ModuleRequest",
    "PackedModuleSession",
    "open_session",
    "Buffer",
    "Module",
    "ModuleServiceError",
    "ModuleServiceProtocolError",
    "ModuleServiceTimeoutError",
    "ModuleServiceRemoteError",
]


class PackedModuleSession:
    """One verified module set and device with caller-managed, owner-bound buffers.

    Local validation raises ValueError, filesystem errors raise OSError, and
    native/transport failures preserve ModuleServiceError and its stderr_tail.
    Close or fatal failure invalidates every returned module and buffer handle.
    """

    def __init__(
        self,
        dist_root: Path,
        target_id: str,
        requests: tuple[ModuleRequest, ...],
        *,
        device_index: int | None = None,
        device_uuid: str | None = None,
        expected_device_id: int | None = None,
        required_capabilities: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(requests, tuple) or not 1 <= len(requests) <= 32:
            raise ValueError(
                "A packed module session requires between 1 and 32 unique requests"
            )
        if not all(isinstance(request, ModuleRequest) for request in requests):
            raise ValueError("Expected typed ModuleRequest values")
        if len(set(requests)) != len(requests):
            raise ValueError(
                "Duplicate packed module request; reuse its returned module handle"
            )
        capabilities = tuple(
            sorted(
                set(validate_required_capabilities(required_capabilities))
                | {"persistent-module-service"}
            )
        )
        root = dist_root.resolve(strict=True)
        registry = load_registry(resolve_registry_path(root, REGISTRY_PATH))
        prepared = prepare_modules(
            root,
            registry,
            requests=requests,
            target_id=target_id,
            device_index=device_index,
            device_uuid=device_uuid,
            expected_device_id=expected_device_id,
            required_capabilities=capabilities,
        )
        # ExitStack unwinds in reverse order: close/reap the worker before
        # removing any file that a loaded module may still reference.
        with ExitStack() as resources:
            temporary = resources.enter_context(
                tempfile.TemporaryDirectory(prefix="therock-packed-session-")
            )
            paths = []
            for index, (selected, _) in enumerate(prepared.invocations):
                path = (
                    Path(temporary) / f"payload-{index}.{selected.entry.payload_type}"
                )
                path.write_bytes(selected.data)
                paths.append(path)
            runner = verify_runner(root, prepared.entry)
            native = resources.enter_context(
                _NativeModuleSession(
                    runner,
                    prepared.target,
                    prepared.device.index,
                    prepared.device.device_uuid,
                    expected_device_id,
                    expected_description=prepared.entry.description,
                )
            )
            modules = {}
            for request, path in zip(requests, paths):
                modules[request] = native.load(
                    request.payload_type, request.entry_point, path
                )
            self._native = native
            self._modules = modules
            self._prepared = prepared
            self._resources = resources.pop_all()

    @property
    def device(self) -> ObservedDevice:
        return self._prepared.device

    @property
    def target(self) -> GpuTarget:
        return self._prepared.target

    @property
    def runner_sha256(self) -> str:
        return self._prepared.entry.sha256

    @property
    def closed(self) -> bool:
        return self._native.closed

    @property
    def stderr_tail(self) -> str:
        return self._native.stderr_tail

    def module(self, request: ModuleRequest) -> Module:
        if self.closed:
            raise ModuleServiceError(
                "Packed module session is closed", self.stderr_tail
            )
        if not isinstance(request, ModuleRequest) or request not in self._modules:
            raise ValueError("Module was not selected when this session was opened")
        return self._modules[request]

    def allocate(self, capacity: int) -> Buffer:
        return self._native.allocate(capacity)

    def release(self, buffer: Buffer) -> None:
        self._native.release(buffer)

    def write(
        self, buffer: Buffer, offset: int, values: Sequence[float] | array
    ) -> None:
        self._native.write(buffer, offset, values)

    def read(self, buffer: Buffer, offset: int, count: int) -> array:
        return self._native.read(buffer, offset, count)

    def launch(
        self,
        module: Module,
        x: Buffer,
        y: Buffer,
        output: Buffer,
        alpha: float,
        count: int,
    ) -> None:
        self._native.launch(module, x, y, output, alpha, count)

    def synchronize(self) -> None:
        self._native.synchronize()

    def close(self) -> None:
        self._resources.close()

    def __enter__(self) -> "PackedModuleSession":
        self._native.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        self._resources.__exit__(exc_type, exc, traceback)


def open_session(
    dist_root: Path,
    target_id: str,
    requests: tuple[ModuleRequest, ...],
    *,
    device_index: int | None = None,
    device_uuid: str | None = None,
    expected_device_id: int | None = None,
    required_capabilities: tuple[str, ...] = (),
) -> PackedModuleSession:
    """Open one verified worker and preload all unique logical module selections."""
    return PackedModuleSession(
        dist_root,
        target_id,
        requests,
        device_index=device_index,
        device_uuid=device_uuid,
        expected_device_id=expected_device_id,
        required_capabilities=required_capabilities,
    )
