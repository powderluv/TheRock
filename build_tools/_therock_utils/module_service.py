# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Synchronous client for the experimental process-owned native module service.

Callers must verify the runner and module payloads before constructing/loading.
This explicit low-level client accepts trusted local paths; it does not select
packs or authenticate binaries. Handles belong to one worker connection and
become unusable after release, close, or a fatal protocol/backend failure.
"""

from array import array
from collections.abc import Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
import selectors
import struct
import subprocess
import sys
import time

from .device_inventory import parse_device_uuid
from .gpu_targets import GpuTarget
from .module_contract import (
    RunnerDescription,
    parse_runner_description_json,
    require_runner_compatibility,
    validation_contract,
)

from .sgemm_contract import (
    SGEMM_ABI,
    SGEMM_VERSION,
    SGEMM_CAPABILITY,
    SGEMM_MAX_DIMENSION,
    SGEMM_MAX_CAPACITY,
    SgemmProviderInfo,
    parse_sgemm_provider_json,
    sgemm_contract_sha256,
    sgemm_provider_for_vendor,
)

_HEADER = struct.Struct("<4sHHII")
_MAGIC = b"TRMS"
_VERSION = 1
_MAX_PAYLOAD = 1024 * 1024
_MAX_STRING = 4096
_MAX_STDERR = 64 * 1024
_REQUEST_TIMEOUT_SECONDS = 45.0
_MAX_CAPACITY = 65556
_MAX_COUNT = 65539
_HELLO, _OPEN, _ALLOC, _FREE, _LOAD, _UNLOAD, _WRITE, _READ, _LAUNCH, _SYNC, _CLOSE = (
    range(1, 12)
)
_SGEMM_PROVIDER, _SGEMM = 12, 13


class ModuleServiceError(RuntimeError):
    """Base error with the bounded worker diagnostic tail available separately."""

    def __init__(self, message: str, stderr_tail: str = "") -> None:
        super().__init__(message)
        self.stderr_tail = stderr_tail


class ModuleServiceProtocolError(ModuleServiceError):
    """Malformed or prematurely terminated transport; this worker is unusable."""


class ModuleServiceTimeoutError(ModuleServiceError, TimeoutError):
    """A request exceeded its deadline; this worker has been terminated."""


class ModuleServiceRemoteError(ModuleServiceError):
    def __init__(
        self, status: int, opcode: int, message: str, stderr_tail: str
    ) -> None:
        super().__init__(
            f"Native module service opcode {opcode} failed ({status}): {message}",
            stderr_tail,
        )
        self.status = status
        self.opcode = opcode
        self.recoverable = status in (1, 4)


@dataclass
class _Owner:
    live: bool = True


class _Handle:
    def __init__(self, owner: _Owner, number: int) -> None:
        self._owner = owner
        self._number = number
        self._live = True

    @property
    def handle(self) -> int:
        return self._number

    @property
    def closed(self) -> bool:
        return not self._live or not self._owner.live


class Buffer(_Handle):
    def __init__(self, owner: _Owner, number: int, capacity: int) -> None:
        super().__init__(owner, number)
        self._capacity = capacity

    @property
    def capacity(self) -> int:
        return self._capacity


class Module(_Handle):
    def __init__(
        self, owner: _Owner, number: int, payload_format: str, symbol: str
    ) -> None:
        super().__init__(owner, number)
        self.payload_format = payload_format
        self.symbol = symbol


def _integer(value: object, label: str, *, maximum: int, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _string(value: str) -> bytes:
    if not isinstance(value, str) or "\0" in value:
        raise ValueError("Service strings must be strings without NUL bytes")
    data = value.encode("utf-8")
    if len(data) > _MAX_STRING:
        raise ValueError("Service strings are limited to 4096 UTF-8 bytes")
    return struct.pack("<I", len(data)) + data


def _decode_string(data: bytes) -> str:
    if len(data) < 4:
        raise ValueError("Missing service string length")
    length = struct.unpack_from("<I", data)[0]
    if length > _MAX_STRING or len(data) != length + 4:
        raise ValueError("Invalid service string length")
    return data[4:].decode("utf-8")


def _identity(target: GpuTarget, expected_device_id: int | None) -> tuple[str, int]:
    if not isinstance(target, GpuTarget):
        raise ValueError("Expected a typed GPU target")
    if target.features or (
        target.vendor == "nvidia" and target.processor[-1].isalpha()
    ):
        raise ValueError(
            "Service discovery cannot qualify architecture feature suffixes"
        )
    if target.vendor != "intel":
        if expected_device_id is not None:
            raise ValueError("Expected device ID is only supported for Intel")
        return target.processor, 0
    if target.processor == "xe2-b70":
        if expected_device_id not in (None, 0xE223):
            raise ValueError("Intel xe2-b70 requires device ID 0xe223")
        expected_device_id = 0xE223
    return "spirv", _integer(
        expected_device_id, "Intel device ID", minimum=1, maximum=0xFFFF
    )


class NativeModuleSession:
    """One synchronous worker; requests on a session must be serialized by its caller."""

    def __init__(
        self,
        runner: Path,
        target: GpuTarget,
        device: int,
        expected_device_uuid: str,
        expected_device_id: int | None = None,
        *,
        expected_description: RunnerDescription | None = None,
    ) -> None:
        if os.name != "posix":
            raise ValueError("Native module service transport requires a POSIX host")
        _integer(device, "Device index", maximum=0xFFFFFFFF)
        parse_device_uuid(expected_device_uuid)
        if expected_description is not None and not isinstance(
            expected_description, RunnerDescription
        ):
            raise ValueError("Expected a typed runner description")
        architecture, device_id = _identity(target, expected_device_id)
        runner = runner.resolve(strict=True)
        if not runner.is_file():
            raise ValueError("Native module service runner must be a regular file")
        self._owner = _Owner()
        self._handles: dict[int, Buffer | Module] = {}
        self._last_handle = 0
        self._sgemm_provider: SgemmProviderInfo | None = None
        self._next_request = 1
        self._stderr = bytearray()
        # Acquire the transport resource before starting a worker, so a failed
        # selector allocation cannot strand a process that we cannot service.
        self._selector = selectors.DefaultSelector()
        try:
            self._process = subprocess.Popen(
                [str(runner), "--serve"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except BaseException:
            self._owner.live = False
            self._selector.close()
            raise
        assert (
            self._process.stdin is not None
            and self._process.stdout is not None
            and self._process.stderr is not None
        )
        self._input = self._process.stdin
        self._output = self._process.stdout
        self._errors = self._process.stderr
        try:
            for stream in (self._input, self._output, self._errors):
                os.set_blocking(stream.fileno(), False)
            self._selector.register(self._output, selectors.EVENT_READ, "stdout")
            self._selector.register(self._errors, selectors.EVENT_READ, "stderr")
            description = parse_runner_description_json(
                _decode_string(self._request(_HELLO))
            )
            if expected_description is not None and description != expected_description:
                raise ValueError(
                    "Service HELLO does not match the verified runner description"
                )
            require_runner_compatibility(
                description,
                target,
                description.payload_types,
                description.entry_points,
                validation_contract(),
            )
            if "persistent-module-service" not in description.capabilities:
                raise ValueError("Runner does not advertise persistent-module-service")
            self.description: RunnerDescription = description
            self.target = target
            contract = validation_contract()
            request = (
                struct.pack("<II", device, device_id)
                + _string(architecture)
                + _string(expected_device_uuid)
                + _string(contract.abi)
                + struct.pack("<I", contract.version)
                + _string(contract.sha256)
            )
            self._empty(self._request(_OPEN, request))
        except BaseException:
            self._terminate()
            raise

    @property
    def closed(self) -> bool:
        return not self._owner.live

    @property
    def stderr_tail(self) -> str:
        return self._stderr.decode("utf-8", errors="replace")

    def __enter__(self) -> "NativeModuleSession":
        self._ensure_live()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except ModuleServiceError:
                pass

    def _ensure_live(self) -> None:
        if self.closed:
            raise ModuleServiceError(
                "Native module session is closed", self.stderr_tail
            )
        if self._process.poll() is not None:
            self._terminate()
            raise ModuleServiceProtocolError(
                "Native module worker exited unexpectedly", self.stderr_tail
            )

    def _terminate(self) -> None:
        self._owner.live = False
        for handle in self._handles.values():
            handle._live = False
        self._handles.clear()
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()
        self._selector.close()
        for stream in (self._input, self._output, self._errors):
            stream.close()

    def _drain_stderr(self, chunk: bytes) -> None:
        self._stderr.extend(chunk)
        del self._stderr[:-_MAX_STDERR]

    def _request(
        self, opcode: int, payload: bytes = b"", *, deadline: float | None = None
    ) -> bytes:
        self._ensure_live()
        if len(payload) > _MAX_PAYLOAD or self._next_request > 0xFFFFFFFF:
            self._terminate()
            raise ModuleServiceProtocolError(
                "Native service request exceeds protocol bounds", self.stderr_tail
            )
        request_id = self._next_request
        self._next_request += 1
        frame = (
            _HEADER.pack(_MAGIC, _VERSION, opcode, request_id, len(payload)) + payload
        )
        deadline = (
            deadline
            if deadline is not None
            else time.monotonic() + _REQUEST_TIMEOUT_SECONDS
        )
        sent = 0
        received = bytearray()
        expected = None
        try:
            self._selector.register(self._input, selectors.EVENT_WRITE, "stdin")
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ModuleServiceTimeoutError(
                        "Native module service request timed out", self.stderr_tail
                    )
                for key, _ in self._selector.select(remaining):
                    if key.data == "stdin":
                        try:
                            sent += os.write(key.fd, frame[sent:])
                        except BlockingIOError:
                            continue
                        if sent == len(frame):
                            self._selector.unregister(self._input)
                    else:
                        try:
                            chunk = os.read(key.fd, 8192)
                        except BlockingIOError:
                            continue
                        if key.data == "stderr":
                            if chunk:
                                self._drain_stderr(chunk)
                            else:
                                self._selector.unregister(self._errors)
                            continue
                        if not chunk:
                            raise ModuleServiceProtocolError(
                                "Native module service response was truncated",
                                self.stderr_tail,
                            )
                        received.extend(chunk)
                        if expected is None and len(received) >= _HEADER.size:
                            magic, version, response, identity, length = (
                                _HEADER.unpack_from(received)
                            )
                            if (
                                magic != _MAGIC
                                or version != _VERSION
                                or response != (opcode | 0x8000)
                                or identity != request_id
                            ):
                                raise ModuleServiceProtocolError(
                                    "Invalid native module service response header",
                                    self.stderr_tail,
                                )
                            if length < 8 or length > _MAX_PAYLOAD:
                                raise ModuleServiceProtocolError(
                                    "Invalid native module service response size",
                                    self.stderr_tail,
                                )
                            expected = _HEADER.size + length
                if expected is not None and len(received) >= expected:
                    if len(received) != expected or sent != len(frame):
                        raise ModuleServiceProtocolError(
                            "Unexpected native module service response bytes",
                            self.stderr_tail,
                        )
                    status, error_length = struct.unpack_from(
                        "<II", received, _HEADER.size
                    )
                    if (
                        status > 5
                        or error_length > _MAX_STRING
                        or error_length > len(received) - _HEADER.size - 8
                    ):
                        raise ModuleServiceProtocolError(
                            "Invalid native module service response status",
                            self.stderr_tail,
                        )
                    offset = _HEADER.size + 8
                    error = received[offset : offset + error_length].decode("utf-8")
                    result = bytes(received[offset + error_length :])
                    if (status == 0 and error) or (status != 0 and result):
                        raise ModuleServiceProtocolError(
                            "Invalid native module service error payload",
                            self.stderr_tail,
                        )
                    if status:
                        exception = ModuleServiceRemoteError(
                            status, opcode, error, self.stderr_tail
                        )
                        if not exception.recoverable:
                            self._terminate()
                        raise exception
                    return result
        except ModuleServiceRemoteError:
            raise
        except BaseException as exc:
            self._terminate()
            if isinstance(exc, (ModuleServiceError, KeyboardInterrupt, SystemExit)):
                raise
            raise ModuleServiceProtocolError(
                f"Native module service transport failed: {exc}", self.stderr_tail
            ) from exc

    def _empty(self, result: bytes) -> None:
        if result:
            self._terminate()
            raise ModuleServiceProtocolError(
                "Native service returned unexpected operation data", self.stderr_tail
            )

    def _handle(self, value: _Handle, expected: type[Buffer] | type[Module]) -> int:
        self._ensure_live()
        if (
            type(value) is not expected
            or value._owner is not self._owner
            or value.closed
            or self._handles.get(value.handle) is not value
        ):
            raise ValueError(
                "Handle is released, belongs to another session, or has the wrong type"
            )
        return value.handle

    def _new_handle(self, result: bytes) -> int:
        if len(result) != 8:
            self._terminate()
            raise ModuleServiceProtocolError(
                "Native service returned an invalid handle size", self.stderr_tail
            )
        number = struct.unpack("<Q", result)[0]
        if number <= self._last_handle:
            self._terminate()
            raise ModuleServiceProtocolError(
                "Native service reused or reordered a handle", self.stderr_tail
            )
        self._last_handle = number
        return number

    def allocate(self, capacity: int) -> Buffer:
        _integer(capacity, "Buffer capacity", minimum=1, maximum=_MAX_CAPACITY)
        number = self._new_handle(self._request(_ALLOC, struct.pack("<I", capacity)))
        value = Buffer(self._owner, number, capacity)
        self._handles[number] = value
        return value

    def release(self, buffer: Buffer) -> None:
        number = self._handle(buffer, Buffer)
        self._empty(self._request(_FREE, struct.pack("<Q", number)))
        buffer._live = False
        del self._handles[number]

    def load(self, payload_format: str, symbol: str, path: Path) -> Module:
        self._ensure_live()
        if (
            payload_format not in self.description.payload_types
            or symbol not in self.description.entry_points
        ):
            raise ValueError("Module format or symbol is not advertised by the service")
        path = path.resolve(strict=True)
        if not path.is_file():
            raise ValueError("Module payload must be a regular file")
        payload = _string(payload_format) + _string(symbol) + _string(str(path))
        number = self._new_handle(self._request(_LOAD, payload))
        value = Module(self._owner, number, payload_format, symbol)
        self._handles[number] = value
        return value

    def unload(self, module: Module) -> None:
        number = self._handle(module, Module)
        self._empty(self._request(_UNLOAD, struct.pack("<Q", number)))
        module._live = False
        del self._handles[number]

    def _range(self, buffer: Buffer, offset: int, count: int) -> int:
        number = self._handle(buffer, Buffer)
        _integer(offset, "Buffer offset", maximum=buffer.capacity)
        _integer(count, "Transfer count", minimum=1, maximum=buffer.capacity)
        if offset + count > buffer.capacity:
            raise ValueError("Transfer exceeds buffer capacity")
        return number

    def write(
        self, buffer: Buffer, offset: int, values: Sequence[float] | array
    ) -> None:
        if isinstance(values, (str, bytes, bytearray)) or not isinstance(
            values, (Sequence, array)
        ):
            raise ValueError("Buffer writes require a sequence of float values")
        number = self._range(buffer, offset, len(values))
        try:
            data = array("f", values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Buffer write values must be representable as float32"
            ) from exc
        if data.itemsize != 4:
            raise ValueError("Native service requires four-byte float storage")
        if sys.byteorder != "little":
            data.byteswap()
        self._empty(
            self._request(
                _WRITE, struct.pack("<QII", number, offset, len(data)) + data.tobytes()
            )
        )

    def read(self, buffer: Buffer, offset: int, count: int) -> array:
        number = self._range(buffer, offset, count)
        result = self._request(_READ, struct.pack("<QII", number, offset, count))
        if len(result) != count * 4:
            self._terminate()
            raise ModuleServiceProtocolError(
                "Native service returned an invalid read size", self.stderr_tail
            )
        data = array("f")
        data.frombytes(result)
        if sys.byteorder != "little":
            data.byteswap()
        return data

    def launch(
        self,
        module: Module,
        x: Buffer,
        y: Buffer,
        output: Buffer,
        alpha: float,
        count: int,
    ) -> None:
        module_number = self._handle(module, Module)
        numbers = [self._handle(buffer, Buffer) for buffer in (x, y, output)]
        _integer(count, "Launch count", minimum=1, maximum=_MAX_COUNT)
        if any(count > buffer.capacity for buffer in (x, y, output)):
            raise ValueError("Launch exceeds a buffer capacity")
        if output is x or output is y:
            raise ValueError("Launch output must not alias either input")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise ValueError("Launch alpha must be finite")
        try:
            if not math.isfinite(alpha):
                raise ValueError("Launch alpha must be finite")
            scalar = struct.pack("<f", alpha)
        except (OverflowError, struct.error) as exc:
            raise ValueError("Launch alpha must fit float32") from exc
        self._empty(
            self._request(
                _LAUNCH,
                struct.pack("<QQQQ", module_number, *numbers)
                + scalar
                + struct.pack("<I", count),
            )
        )

    def sgemm_provider(self) -> SgemmProviderInfo:
        """Negotiate and cache the target's optional loaded BLAS provider.

        Default vector sessions do not initialize a BLAS provider. An invalid
        descriptor is a fatal protocol error; absent compiled support is local.
        """
        self._ensure_live()
        if SGEMM_CAPABILITY not in self.description.capabilities:
            raise ValueError("Runner does not advertise " + SGEMM_CAPABILITY)
        if self._sgemm_provider is not None:
            return self._sgemm_provider
        provider = sgemm_provider_for_vendor(self.target.vendor)
        request = (
            _string(provider)
            + _string(SGEMM_ABI)
            + struct.pack("<I", SGEMM_VERSION)
            + _string(sgemm_contract_sha256())
        )
        result = self._request(_SGEMM_PROVIDER, request)
        try:
            info = parse_sgemm_provider_json(
                _decode_string(result), vendor=self.target.vendor
            )
        except ValueError as exc:
            self._terminate()
            raise ModuleServiceProtocolError(
                f"Invalid SGEMM provider response: {exc}", self.stderr_tail
            ) from exc
        self._sgemm_provider = info
        return info

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
        """Queue bounded column-major NN FP32 C = alpha*A*B + beta*C.

        Offsets and leading dimensions count float elements. READ/SYNC supplies
        completion. The output buffer must differ from both input buffers.
        """
        numbers = tuple(self._handle(buffer, Buffer) for buffer in (a, b, c))
        if c is a or c is b:
            raise ValueError("SGEMM output must not alias either input buffer")
        for name, value in (("m", m), ("n", n), ("k", k)):
            _integer(value, "SGEMM " + name, minimum=1, maximum=SGEMM_MAX_DIMENSION)
        for buffer, offset, rows, columns, leading, name in (
            (a, a_offset, m, k, lda, "A"),
            (b, b_offset, k, n, ldb, "B"),
            (c, c_offset, m, n, ldc, "C"),
        ):
            _integer(offset, f"SGEMM {name} offset", maximum=buffer.capacity)
            _integer(
                leading,
                f"SGEMM {name} leading dimension",
                minimum=rows,
                maximum=SGEMM_MAX_CAPACITY,
            )
            span = (columns - 1) * leading + rows
            if offset + span > buffer.capacity:
                raise ValueError(f"SGEMM {name} matrix exceeds buffer capacity")
        scalars = b""
        for name, value in (("alpha", alpha), ("beta", beta)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"SGEMM {name} must be finite float32")
            try:
                if not math.isfinite(value):
                    raise ValueError(f"SGEMM {name} must be finite float32")
                scalars += struct.pack("<f", value)
            except (OverflowError, struct.error) as exc:
                raise ValueError(f"SGEMM {name} must fit float32") from exc
        # All local ownership, dimensions, ranges and scalar checks precede
        # lazy provider initialization, including the first SGEMM request.
        self.sgemm_provider()
        payload = (
            struct.pack(
                "<QQQ9I", *numbers, a_offset, b_offset, c_offset, m, n, k, lda, ldb, ldc
            )
            + scalars
        )
        self._empty(self._request(_SGEMM, payload))

    def synchronize(self) -> None:
        self._empty(self._request(_SYNC))

    def close(self) -> None:
        if self.closed:
            return
        deadline = time.monotonic() + _REQUEST_TIMEOUT_SECONDS
        try:
            self._empty(self._request(_CLOSE, deadline=deadline))
            while self._selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ModuleServiceTimeoutError(
                        "Native module service close timed out", self.stderr_tail
                    )
                for key, _ in self._selector.select(remaining):
                    try:
                        chunk = os.read(key.fd, 8192)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        self._selector.unregister(key.fileobj)
                    elif key.data == "stderr":
                        self._drain_stderr(chunk)
                    else:
                        raise ModuleServiceProtocolError(
                            "Native service emitted data after CLOSE", self.stderr_tail
                        )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModuleServiceTimeoutError(
                    "Native module service close timed out", self.stderr_tail
                )
            try:
                status = self._process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise ModuleServiceTimeoutError(
                    "Native module service close timed out", self.stderr_tail
                ) from exc
            if status != 0:
                raise ModuleServiceProtocolError(
                    f"Native module worker exited with status {status}",
                    self.stderr_tail,
                )
        except ModuleServiceError:
            raise
        except OSError as exc:
            raise ModuleServiceProtocolError(
                f"Native module service close failed: {exc}", self.stderr_tail
            ) from exc
        finally:
            self._terminate()
