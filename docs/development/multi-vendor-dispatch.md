# Native device discovery and exact dispatch

The multi-vendor profile packages a registry of native runners and lets one
command discover devices or execute an exact packed fixture. The caller selects
a module, target, format, and entry point; the registry supplies the runner and
catalog paths. This extends the [launch contract](multi-vendor-module-contracts.md)
without introducing a same-process runtime plugin ABI.

## Build and use

Build the profile as described in the [runbook](multi-vendor-hip.md). Rebuilding
an existing native-module profile adds
`dist/multi-vendor-modules/share/therock/packs/runners.json`. The commands below
use the conventional `build/multi-vendor` build directory; substitute your build
path if different. The Python command runs from this source checkout and still
requires its kpack Python dependency. It is not a standalone installed SDK CLI.

```sh
PYTHONPATH=rocm-systems/shared/kpack/python \
  .venv/bin/python build_tools/dispatch_multi_vendor_modules.py \
  --dist-root build/multi-vendor/dist/multi-vendor-modules list

PYTHONPATH=rocm-systems/shared/kpack/python \
  .venv/bin/python build_tools/dispatch_multi_vendor_modules.py \
  --dist-root build/multi-vendor/dist/multi-vendor-modules run \
  --module validation/relu --target nvidia:cuda:sm_120 \
  --format cubin --entry-point therock_module_relu --device 0

PYTHONPATH=rocm-systems/shared/kpack/python \
  .venv/bin/python build_tools/dispatch_multi_vendor_modules.py \
  --dist-root build/multi-vendor/dist/multi-vendor-modules run \
  --module validation/saxpy --target amd:hip:gfx1201 \
  --format hsaco --entry-point therock_module_saxpy --device 0
```

The profile's packed-module CTests use this dispatcher. The earlier
`validate_multi_vendor_modules.py --runner ...` interface remains available for
explicit runner testing and shares the guarded execution implementation.

## Three distinct kinds of metadata

| Record                         | Source                | Meaning                                                                                             |
| ------------------------------ | --------------------- | --------------------------------------------------------------------------------------------------- |
| Runner registry, schema 1      | Pack assembly         | Exact target, relative executable path, executable SHA256, compiled description, and catalog paths. |
| Runner description, protocol 1 | `--describe-contract` | Compiled adapter support; no driver initialization.                                                 |
| Device inventory, protocol 2   | `--list-devices`      | Devices and limits observed through the installed driver in this process.                           |

Registry and catalog schemas are independent. Catalog schema 2, launch ABI 1,
and KPAK v1 remain unchanged. Assembly validates every staged executable,
description, and payload before publishing its managed files. An executable
change rebuilds its registry digest. A missing staged executable fails the pack
build and preserves previously published outputs.

Registry paths are normalized POSIX paths relative to the distribution root.
Absolute paths, traversal, duplicate targets or paths, and symlinks escaping the
root fail explicitly. Registry SHA256 checks establish integrity relative to
local metadata; they do not authenticate a publisher or bind imported dynamic
libraries. Publication remains per file, so installation must be serialized
with use. Executable identity is checked before queries and again after selected
device discovery; execution is not pinned to a verified inode across concurrent
filesystem replacement.

## Discovery behavior

Each runner accepts `--list-devices` alone. Conflicting flags fail before any
GPU API call. Discovery initializes the driver and queries properties, without
explicitly creating contexts, queues, allocations, modules, or kernels. Driver
initialization may itself maintain internal resources. Device names are escaped
as JSON, and malformed UTF-8 fails explicitly.

The strict native response has `scope: "observed-runtime"`. Each device includes
its launch ordinal, driver-reported device UUID, name, vendor/device IDs where available, architecture where
observable, maximum block/grid dimensions, maximum threads per block, and total
memory capacity. AMD and NVIDIA report base `gfx...` and `sm_...` architectures;
Intel reports `architecture: null` and uses PCI identity. The Intel B70 selector
requires vendor `0x8086` and device `0xe223`; this remains a hardware assumption
until the card is qualified. Other Intel targets require an explicit
`--expect-device-id` when executing.

Intel enumerates the same filtered root GPUs, in the same UUID-sorted order, as
the payload runner. Intel memory capacity sums the device-visible memory heaps
reported by Level Zero. Capacity is neither currently free memory nor a maximum
single allocation. All ordinals are process-local selections. UUID selection resolves the device to
its current ordinal, and execution checks that UUID again before explicitly
creating execution resources.

The aggregate `list` response uses schema 2 and reports each registry entry separately:

- `status: "available"` means its description and inventory queries succeeded;
  zero devices is a valid result.
- `matching_device_indices` contains ordinals whose observable identity matches
  that registry target. It is empty for an SM90 entry observing only an SM120
  card. It does not establish launch capability or arithmetic correctness.
- `status: "unavailable"` includes a diagnostic for a missing driver, query
  failure, or malformed response; other installed vendors are still listed.

All registry executable hashes are verified before any `list` query begins.
A corrupt or missing registered executable therefore fails the command instead
of being reported as a missing driver. Listing does not verify catalog or payload
files; those checks belong to execution. A valid distribution with an unavailable
Intel driver can still return an AMD/NVIDIA inventory successfully. This makes
`list` useful before Intel hardware arrives, without treating it as a hardware
acceptance test. Multiple architecture-specific runners can list the same
physical device; this is a per-runner inventory, not a deduplicated system map.

Queries use the shared POSIX subprocess reader: 10 seconds per query and 64 KiB
of combined stdout/stderr. `--describe-contract` stays offline;
`--list-devices` deliberately accesses the driver. Non-POSIX hosts fail before
starting queries.

## UUID selection and identity binding

Use a device UUID from the native inventory when an ordinal is unsuitable. UUIDs
are represented as 32 lowercase hexadecimal digits in the driver's raw byte
order, with no hyphens. Empty, all-zero, malformed, or duplicate UUIDs within a
runner's inventory fail explicitly. Selection is scoped to the requested
vendor/backend and still requires the exact target architecture or Intel PCI ID.
A UUID never overrides those checks.

```sh
# Replace the value with a device_uuid returned by list for this target.
PYTHONPATH=rocm-systems/shared/kpack/python \
  .venv/bin/python build_tools/dispatch_multi_vendor_modules.py \
  --dist-root build/multi-vendor/dist/multi-vendor-modules run \
  --module validation/relu --target nvidia:cuda:sm_120 \
  --format cubin --entry-point therock_module_relu \
  --device-uuid "$DEVICE_UUID"
```

`--device-uuid` and `--device` are mutually exclusive. With neither option, the
command selects ordinal zero. Both paths pass the discovered UUID to the native
runner. If that ordinal identifies another UUID in the execution process, the
runner fails before explicitly creating its context, queue, allocations, or
module. It does not search for another ordinal or silently switch devices.
The explicit validation wrapper also accepts `--expect-device-uuid`; legacy
explicit-runner callers may omit it to disable the equality guard. A valid
driver UUID is still required by the native paths.

AMD obtains the UUID through `hipDeviceGetUuid`, which the imported ROCm 7.2
headers mark as beta. CUDA uses `cuDeviceGetUuid` mapped by the imported CUDA
headers to its v2 API, including compute-instance identity where applicable.
Intel uses the device UUID from Level Zero properties, distinct from the driver
UUID used in sorting. These are driver-provided identifiers, not a new global
hardware naming standard. Persistence across driver updates, repartitioning, or
hardware changes is not promised. There is no fallback to a fabricated UUID or
PCI address when an implementation cannot supply a valid UUID.

Native and aggregate inventory schemas advance from 1 to 2 for this required
field. The dispatcher rejects schema-1 native inventories; rebuild old native
artifacts to regenerate their executables and registry hashes. Registry schema
1, compiled-description protocol 1, launch ABI 1, and catalog schema 2 stay
unchanged. The fixture's argument layout and resource ownership have not changed.

## Execution checks

`run` first selects the exact registered target and verifies its executable. It
resolves the registry's catalogs and extracts a verified payload snapshot before
any driver discovery. The payload contract, registry description, and actual
compiled description must agree. Only the selected runner is queried; an
unavailable Intel driver does not block a selected NVIDIA or AMD run.

Discovery resolves the requested UUID or ordinal, then checks its vendor, architecture or PCI ID,
and the block dimensions/thread count required by the contract. Architecture
feature suffixes that discovery cannot observe are rejected rather than
qualified by a base architecture match. This restriction belongs to the new
dispatch command; lower-level compiler/explicit-runner tools retain their prior
interfaces.

The shared guarded execution path passes the agreed contract and format to the
native runner, including the discovered `--expect-device-uuid`. The runner
rechecks that identity before explicitly creating execution resources and checks actual device/kernel
limits before launch. Memory availability, kernel-specific constraints, module
loading/JIT, synchronization, and arithmetic still require this execution step.
The `DISPATCH` log records the selected identity; the native `PASS` records the
fixture result. Discovery alone never records a hardware pass.

There is no architecture, cubin/PTX, backend, or ABI fallback. The current
protocol remains separate processes with runner-owned resources. A batch can
now reuse resources across modules inside its native process. The persistent
service adds connection-scoped module/buffer handles and protocol errors.
Public device/context/event handles and shared-library dispatch tables remain
future work. Intel and SM90 hardware qualification remain deferred until
matching devices are available.

## Requiring adapter behavior

`run --require-capability cross-queue-events` requires the native event pipeline
in addition to the unchanged kernel contract. The profile's packed-module CTests
use this requirement. An older adapter missing it fails before driver discovery;
without an explicit requirement, the original compatible adapter remains usable.
See the [event validation guide](multi-vendor-events.md) for queue ordering,
resource lifetime, capability negotiation, and remaining hardware limits.

## Running modules in one session

`run-batch --target TARGET --module MODULE FORMAT ENTRY_POINT` accepts repeated
module requests from the registered catalogs. It verifies all selections before
discovery, binds one device UUID, and executes them in one native process with
shared context and resources. It automatically requires `multi-module-session`
and `cross-queue-events`; the kernel ABI and single-module `run` are unchanged.
See the [session guide](multi-vendor-sessions.md) for mixed cubin/PTX examples,
resource ownership, limits, and failure behavior.

## Composing stages on the device

`run-pipeline --target TARGET --module MODULE FORMAT ENTRY_POINT` preserves
stage order and allows repeated stages. Each stage reads the previous output
through alternating device scratch buffers; readback follows the final stage.
It adds the required `device-module-pipeline` capability to session/event
requirements. See the [pipeline guide](multi-vendor-pipelines.md) for bindings,
numerical checks, synchronization, and validation boundaries.

## Caller data through persistent handles

`run-service --target TARGET --module MODULE FORMAT ENTRY_POINT` validates
one to three composed stages through a persistent native worker. Repeated
selections share verified payload bytes and loaded module handles. The
dispatcher retains those files until close, checks the exact HELLO description
before OPEN, and binds the observed device UUID. The required compiled capability
is `persistent-module-service`.

The Python `NativeModuleSession` API also accepts caller arrays and exposes
allocation, offset copies, launch, synchronization, release, and unload.
Its direct interface assumes trusted, already-verified local artifacts; the
CLI performs registry/catalog verification. See the [service guide](multi-vendor-service.md)
for the 32-module/64-buffer connection limits, wire protocol, and failure policy.

## Reusing verified selection in applications

The installed `therock_multi_vendor` package exposes `ModuleRequest` and
`open_session`. It shares the CLI's exact target, contract, payload, capability,
and device checks while returning preloaded handles for caller-owned data.
The session retains verified files until worker cleanup completes. See the
[client guide](multi-vendor-client.md) for application code and the installed
example that keeps Radeon and NVIDIA workers alive in one Python host.
