# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verified packed-module and native SGEMM application sessions.

Use ``with open_session(...) as session`` for preloaded packed modules, or
``with open_sgemm_session(...) as session`` for a negotiated BLAS provider without
module selection. Packed modules are verified before device discovery. All
sessions verify the exact registered runner and observed device. Calls on a
session must be serialized. Integrity is relative to supplied distribution
metadata; publication must be serialized with use. No target or format fallback
is performed. The low-level trusted-path client is intentionally not exposed.
"""

from array import array
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
import tempfile
from typing import TypeVar

from _therock_utils.device_inventory import ObservedDevice
from _therock_utils.gpu_targets import GpuTarget
from _therock_utils.module_contract import validate_required_capabilities
from _therock_utils.module_selection import (
    REGISTRY_PATH,
    ModuleRequest,
    PreparedModules,
    PreparedWorker,
    prepare_modules,
    prepare_worker,
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
from _therock_utils.sgemm_contract import SGEMM_CAPABILITY, SgemmProviderInfo
from _therock_utils.runner_registry import (
    load_registry,
    resolve_registry_path,
    verify_runner,
)

__all__ = [
    "ModuleRequest",
    "PackedModuleSession",
    "SgemmSession",
    "open_session",
    "open_sgemm_session",
    "Buffer",
    "Module",
    "ModuleServiceError",
    "ModuleServiceProtocolError",
    "ModuleServiceTimeoutError",
    "ModuleServiceRemoteError",
    "SgemmProviderInfo",
]


_Session = TypeVar("_Session", bound="_WorkerSession")


def _session_capabilities(
    required_capabilities: tuple[str, ...], *, sgemm: bool = False
) -> tuple[str, ...]:
    capabilities = set(validate_required_capabilities(required_capabilities))
    capabilities.add("persistent-module-service")
    if sgemm:
        capabilities.add(SGEMM_CAPABILITY)
    return tuple(sorted(capabilities))


def _start_worker(
    root: Path,
    prepared: PreparedWorker | PreparedModules,
    expected_device_id: int | None,
    resources: ExitStack,
) -> _NativeModuleSession:
    # Discovery and private payload writes may take time. Recheck the executable
    # immediately before --serve; distribution updates must remain serialized.
    runner = verify_runner(root, prepared.entry)
    return resources.enter_context(
        _NativeModuleSession(
            runner,
            prepared.target,
            prepared.device.index,
            prepared.device.device_uuid,
            expected_device_id,
            expected_description=prepared.entry.description,
        )
    )


class _WorkerSession:
    """Buffer, math, and lifetime operations shared by verified session types."""

    def __init__(
        self,
        prepared: PreparedWorker | PreparedModules,
        native: _NativeModuleSession,
        resources: ExitStack,
    ) -> None:
        self._prepared = prepared
        self._native = native
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

    def sgemm_provider(self) -> SgemmProviderInfo:
        return self._native.sgemm_provider()

    def sgemm(
        self,
        a: Buffer,
        b: Buffer,
        c: Buffer,
        *,
        m: int,
        n: int,
        k: int,
        lda: int,
        ldb: int,
        ldc: int,
        a_offset: int = 0,
        b_offset: int = 0,
        c_offset: int = 0,
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> None:
        self._native.sgemm(
            a,
            b,
            c,
            m=m,
            n=n,
            k=k,
            lda=lda,
            ldb=ldb,
            ldc=ldc,
            a_offset=a_offset,
            b_offset=b_offset,
            c_offset=c_offset,
            alpha=alpha,
            beta=beta,
        )

    def synchronize(self) -> None:
        self._native.synchronize()

    def close(self) -> None:
        self._resources.close()

    def __enter__(self: _Session) -> _Session:
        self._native.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        self._resources.__exit__(exc_type, exc, traceback)


class PackedModuleSession(_WorkerSession):
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
        capabilities = _session_capabilities(required_capabilities)
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
            native = _start_worker(root, prepared, expected_device_id, resources)
            modules = {}
            for request, path in zip(requests, paths):
                modules[request] = native.load(
                    request.payload_type, request.entry_point, path
                )
            self._modules = modules
            super().__init__(prepared, native, resources)

    def module(self, request: ModuleRequest) -> Module:
        if self.closed:
            raise ModuleServiceError(
                "Packed module session is closed", self.stderr_tail
            )
        if not isinstance(request, ModuleRequest) or request not in self._modules:
            raise ValueError("Module was not selected when this session was opened")
        return self._modules[request]

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


class SgemmSession(_WorkerSession):
    """One verified device and eagerly negotiated native BLAS provider.

    Opening verifies the registry and runner, discovers the exact device, and
    negotiates the operation contract before returning. It does not resolve
    catalogs, extract payloads, or load modules. The distribution and runtime
    dependency requirements are unchanged; this is not a pack-free export mode.
    Buffers are owner-bound and become invalid after close or fatal failure.
    Errors follow the same vocabulary as PackedModuleSession.
    """

    def __init__(
        self,
        dist_root: Path,
        target_id: str,
        *,
        device_index: int | None = None,
        device_uuid: str | None = None,
        expected_device_id: int | None = None,
        required_capabilities: tuple[str, ...] = (),
    ) -> None:
        capabilities = _session_capabilities(required_capabilities, sgemm=True)
        root = dist_root.resolve(strict=True)
        registry = load_registry(resolve_registry_path(root, REGISTRY_PATH))
        prepared = prepare_worker(
            root,
            registry,
            target_id=target_id,
            device_index=device_index,
            device_uuid=device_uuid,
            expected_device_id=expected_device_id,
            required_capabilities=capabilities,
        )
        with ExitStack() as resources:
            native = _start_worker(root, prepared, expected_device_id, resources)
            native.sgemm_provider()
            super().__init__(prepared, native, resources)


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


def open_sgemm_session(
    dist_root: Path,
    target_id: str,
    *,
    device_index: int | None = None,
    device_uuid: str | None = None,
    expected_device_id: int | None = None,
    required_capabilities: tuple[str, ...] = (),
) -> SgemmSession:
    """Open an exact verified worker and negotiate SGEMM without loading modules."""
    return SgemmSession(
        dist_root,
        target_id,
        device_index=device_index,
        device_uuid=device_uuid,
        expected_device_id=expected_device_id,
        required_capabilities=required_capabilities,
    )
