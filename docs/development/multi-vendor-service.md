# Persistent native module service

The experimental native runners now accept a persistent connection with caller-owned
request sequencing and worker-owned GPU resources. The Python
[NativeModuleSession client](../../build_tools/_therock_utils/module_service.py)
exposes buffer and module handles, offset transfers, launches, synchronization,
and release. Each connection selects one device and keeps its context and queue
alive until close.

The service implements the existing five-argument
[vector module contract](multi-vendor-module-contracts.md). It is a versioned
process protocol, not a shared-library C ABI, a HIP runtime replacement, or a
general kernel argument interface. Device pointers never cross this boundary.

## Verified service validation

Build the profile distribution, then run from the Shark-a checkout:

```sh
cmake --build build/multi-vendor-contracts --target therock-dist --parallel 8

PYTHONPATH=rocm-systems/shared/kpack/python \
  .venv/bin/python build_tools/dispatch_multi_vendor_modules.py \
  --dist-root build/multi-vendor-contracts/dist/multi-vendor-modules run-service \
  --target nvidia:cuda:sm_120 \
  --module validation/saxpy cubin therock_module_saxpy \
  --module validation/relu ptx therock_module_relu \
  --module validation/saxpy cubin therock_module_saxpy
```

Each repeated `--module` takes `MODULE FORMAT ENTRY_POINT`. This validation command
accepts one to three stages, preserves order, and allows repeated selections.
It automatically requires `persistent-module-service`; it does not require the
separate three-queue fixture capabilities. Use `--device-uuid` to select a listed
UUID or `--device` for a process-local ordinal. Both paths bind the observed UUID
into OPEN. Intel B70 uses vendor ID 0x8086 and device ID 0xe223, with `spirv`
describing the payload format rather than the hardware architecture.

The dispatcher verifies the selected executable, all selected payload snapshots,
contracts, formats, symbols, and compiled capabilities before device discovery.
It then checks observed identity and geometry. The service's HELLO must exactly
match the verified runner description **before OPEN** initializes execution
resources. No architecture, format, backend, or contract fallback is available.

Repeated selections reuse identical verified bytes and one loaded module handle.
Private payload files remain available until the session has closed, including
error cleanup. Separate catalog reads still do not form an atomic release;
distribution publication must be serialized with use.

`run-service` supplies fixture data from Python rather than accepting user arrays
on its command line. It reuses four device buffers across six sizes and three
rounds, varies alpha over 0, -0.5, and 1.25, exercises offset writes and reads,
and composes stages through alternating scratch buffers. Every round checks
both retained scratch results and their full unused tails. One-stage execution
leaves the second scratch buffer entirely as guards. These dyadic inputs and
the three-stage limit keep the tested arithmetic exactly representable in FP32.
The 18 `SERVICE_CHECK` results and final PASS also require successful cleanup.
They are distinct from the native `run-pipeline` fixture and its propagated
rounding bound.

## Verified application sessions

The installed `therock_multi_vendor.open_session` API accepts logical
`ModuleRequest` declarations and owns registry selection, payload verification,
device discovery, and private-file lifetime. Applications can use it from the
staged distribution without checkout paths. Its facade returns only preloaded
module handles; the lower-level trusted-path client below remains the transport
implementation. See the [installed client guide](multi-vendor-client.md) for a
caller-data example and an AMD/NVIDIA consumer using independent workers.

## Python interface and ownership

The direct client accepts trusted local runner and payload paths. It does not
select catalogs or authenticate files. Callers should perform the same registry,
payload, contract, and device checks as the dispatcher, retain their verified
payload files through close, and pass the verified `RunnerDescription` as
`expected_description`.

The following API fragment assumes those checks have produced `runner`, `target`,
`device_index`, `device_uuid`, `expected_description`, and a retained `payload`
path. `expected_device_id` is `None` for AMD/NVIDIA, or the selected Intel PCI ID.
For source-tree Python callers, add `build_tools` to `PYTHONPATH`; catalog
verification also needs `rocm-systems/shared/kpack/python`.

```python
from array import array
from _therock_utils.module_service import NativeModuleSession

with NativeModuleSession(
    runner,
    target,
    device_index,
    device_uuid,
    expected_device_id,
    expected_description=expected_description,
) as session:
    saxpy = session.load("cubin", "therock_module_saxpy", payload)
    x = session.allocate(4)
    y = session.allocate(4)
    output = session.allocate(4)
    session.write(x, 0, array("f", [1.0, 2.0, 3.0, 4.0]))
    session.write(y, 0, array("f", [0.5, 0.5, 0.5, 0.5]))
    session.launch(saxpy, x, y, output, 1.25, 4)
    result = session.read(output, 0, 4)
    assert result == array("f", [1.75, 3.0, 4.25, 5.5])
    session.synchronize()
    for buffer in (x, y, output):
        session.release(buffer)
    session.unload(saxpy)
```

This example uses a verified NVIDIA cubin. Select `hsaco` for AMD or `spirv` for
Intel; NVIDIA also accepts explicitly selected PTX. The only entry points are
`therock_module_saxpy` and `therock_module_relu`. Their argument binding remains
`(x, y, output, alpha, count)`, with three device FP32 pointers, an FP32 scalar,
and a uint32 count. The ReLU variant computes `max(0, alpha*x + y - 0.125)`.

Offsets and capacities count FP32 elements, not bytes. Each connection supports
at most 64 live buffers and 32 live modules. A buffer holds 1–65556 elements;
a launch processes 1–65539 elements, bounded by all three buffers. Alpha must
be finite and representable as FP32. Output cannot alias either input; the
read-only inputs may share a handle.

Buffer and module handles are typed Python objects backed by opaque uint64
connection-local IDs. They are not native pointers or transferable device handles.
Released handles, handles from another connection, and handles of the wrong type
are rejected. IDs are never reused within a connection. Close or a fatal error
invalidates every client handle. Explicit release/unload is useful for long-lived
sessions; the context manager also closes and cleans up remaining resources.

Calls on one `NativeModuleSession` must be serialized by its caller. Its methods
use synchronous request/reply transport, but a successful LAUNCH reply need not
mean GPU completion. AMD/NVIDIA enqueue launches on one persistent nonblocking
stream. READ, SYNC, FREE, UNLOAD, and CLOSE establish completion where needed;
writes complete their pinned staging transfer before returning. Intel uses one
asynchronous queue with a regular command list per operation, and completes each
operation before its reply. Command lists and pinned staging memory remain alive
until bounded queue completion. Neither backend promises overlap or exposes
public event handles.

## Wire protocol and failures

A runner enters service mode with `--serve` alone. HELLO is offline; a successful
OPEN selects one device scope for that connection. The worker duplicates its
protocol output descriptor and redirects existing native stdout diagnostics to
stderr, leaving stdout exclusively for binary responses.

Protocol v1 has a 16-byte little-endian frame header: four-byte magic `TRMS`,
uint16 version, uint16 opcode, uint32 request ID, and uint32 payload length.
Request IDs start at one and increase monotonically. Payloads are bounded to
1 MiB. Strings carry a uint32 byte length, contain valid UTF-8 without embedded
NULs, and are limited to 4096 bytes. Float values are IEEE FP32 encoded little-endian.

| Opcode   | Request fields                                                                                   | Successful result                                |
| -------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------ |
| 1 HELLO  | Empty                                                                                            | Length-prefixed compiled runner description JSON |
| 2 OPEN   | Device ordinal, PCI ID, architecture/format string, UUID, ABI name, ABI version, contract SHA256 | Empty                                            |
| 3 ALLOC  | Capacity                                                                                         | Buffer ID                                        |
| 4 FREE   | Buffer ID                                                                                        | Empty                                            |
| 5 LOAD   | Payload format, entry point, local path                                                          | Module ID                                        |
| 6 UNLOAD | Module ID                                                                                        | Empty                                            |
| 7 WRITE  | Buffer ID, offset, count, FP32 values                                                            | Empty                                            |
| 8 READ   | Buffer ID, offset, count                                                                         | FP32 values                                      |
| 9 LAUNCH | Module ID, x/y/output IDs, alpha, count                                                          | Empty                                            |
| 10 SYNC  | Empty                                                                                            | Empty                                            |
| 11 CLOSE | Empty                                                                                            | Empty, then worker exit                          |

IDs in operation payloads are uint64; counts, offsets, ordinals, PCI IDs, and
the ABI version are uint32. A response repeats the request ID and uses the
request opcode with bit 0x8000 set. Its body begins with uint32 status and a
length-prefixed error string, followed by any successful result. Error text is
sanitized and truncated on a UTF-8 boundary.

Status 1 reports invalid arguments or handles, and status 4 reports invalid
connection state; these are recoverable command errors. Status 2 reports framing
failure, status 3 a backend or cleanup failure, and status 5 a protocol or
contract-version mismatch; these close the worker. Malformed replies, transport
loss, and request timeouts also invalidate the client connection. The Python
exceptions retain a bounded stderr diagnostic tail.

The client enforces a 45-second deadline per request, including graceful close.
Native queue completion is bounded separately at 30 seconds. EOF, fatal errors,
FREE, UNLOAD, and CLOSE drain queued work before releasing referenced objects.
When completion cannot be established, the isolated worker uses `_Exit` instead
of freeing possibly active resources. Client timeout terminates the worker.
These failure policies do not provide cancellation, rollback, or recoverable
driver errors inside an application's process.

## Validation and remaining work

The current native profile passed 21 CTests on Shark-a, including four AMD/NVIDIA
service cases with 18 exact arithmetic/guard checks each. Those four cases are a
subset of the 21, not an additional unique-test total. The combined HIP/native
profile passed 25 CTests; it overlaps the native run. The final focused
regression run passed 310 tests with no skips. These runs and the Intel subset
below are separate evidence, not an additive unique-test total. Previous session
and pipeline counts describe their own checkpoints.

The Intel adapter compiled, and eight offline tests passed with `zeInit`
interposed. They cover existing session/pipeline gates plus binary HELLO/CLOSE,
bad OPEN requests, a valid OPEN initialization-boundary control, and redirection
of native stdout diagnostics away from the protocol channel. No Intel driver
initialized in those tests. B70 and SM90 hardware execution remain deferred.

This increment adds process-scoped buffer/module ownership and protocol error
handling. A shared-library C ABI, public device/context/event handles, general
argument layouts, multi-device workers, cancellation, math providers, and
production consumer qualification remain future work. The logical kernel ABI,
catalog schema 2, registry schema 1, runner-description protocol 1, inventory
schema 2, and KPAK v1 are unchanged; service RPC v1 is a separate protocol.

Rebuild the profile distribution after source or description changes.
[Imported-input locks](multi-vendor-inputs.md), completion receipts, external SDK
and driver dependencies, and restrictions on artifact cache reuse remain in
force. The service does not complete toolchain dependency capture or establish
full ROCm compatibility on other vendors. Startup cost, throughput, and other
performance changes have not been measured.
