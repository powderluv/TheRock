# Installed multi-vendor Python client

The experimental `therock_multi_vendor` package opens persistent native workers
from a TheRock multi-vendor distribution. Applications name logical modules,
exact targets, formats, and entry points. The client performs registry, payload,
contract, and device checks and returns loaded opaque handles for caller data.

This extends the [persistent service](multi-vendor-service.md) with a reusable
verified application interface and an installed Python runtime. It retains the
fixed `therock.validation.f32-vector` ABI, one device per worker, and serialized
calls per session. The parent Python application can retain sessions for several
vendors; each vendor driver and its resources remain in a separate native process.

## Build and use

Build the existing experimental profile distribution:

```sh
cmake --build build/multi-vendor-contracts --target therock-dist --parallel 8
```

The resulting `dist/multi-vendor-modules/share/therock/python` directory contains
the public package and its private support code. The installation also includes
a consumer example under `share/therock/examples/packed_session_client.py`.
Python, `msgpack`, `zstandard`, the vendor user-space runtime, and the installed
GPU driver remain external dependencies. The installed `requirements.txt`
declares the Python dependencies; installation does not fetch them. This staged
Python module tree ships in the existing experimental test artifact. It is not
a published wheel or a complete ROCm runtime. The bundled `rocm_kpack` package
contains the archive-reader subset, not the full binary transformation toolkit.

For an application using a trusted installed distribution, put its
`share/therock/python` directory on Python's import path. This NVIDIA example
opens a packed SAXPY module and sends application-owned data:

```python
from array import array
from pathlib import Path
from therock_multi_vendor import ModuleRequest, open_session

root = Path("/path/to/dist/multi-vendor-modules")
saxpy = ModuleRequest("validation/saxpy", "cubin", "therock_module_saxpy")
with open_session(root, "nvidia:cuda:sm_120", (saxpy,)) as session:
    kernel = session.module(saxpy)
    x = session.allocate(4)
    y = session.allocate(4)
    output = session.allocate(4)
    session.write(x, 0, array("f", [1.0, 2.0, 3.0, 4.0]))
    session.write(y, 0, array("f", [0.5, 0.5, 0.5, 0.5]))
    session.launch(kernel, x, y, output, 1.25, 4)
    assert session.read(output, 0, 4) == array("f", [1.75, 3.0, 4.25, 5.5])
```

Select `amd:hip:gfx1201` with `hsaco` for the Radeon or an exact Intel target
with `spirv`. NVIDIA `cubin` and `ptx` are explicit alternatives; there is no
implicit format or target fallback. `validation/relu` exports
`therock_module_relu`, which computes `max(0, alpha*x + y - 0.125)`.

A session declares one to 32 unique `ModuleRequest` values at opening. Lookup
returns the existing loaded handle; repeated launches do not reload the module.
The public facade does not accept arbitrary payload paths or add modules after
opening. The lower-level `NativeModuleSession` remains available internally for
trusted-path protocol use.

Use `device_uuid` to choose a discovered UUID, or `device_index` for an ordinal;
they are mutually exclusive. Both choices bind the observed UUID into native
OPEN. Intel `expected_device_id` defaults to 0xe223 for `xe2-b70`; other Intel
targets require an explicit device ID. Device UUIDs are driver-reported identity
checks, not a guarantee of persistence across driver or firmware changes.

## Verification and ownership

Opening validates the full declared module set before device discovery. A bad
later pack cannot start an earlier module. The client checks the registered
runner's digest and description, every selected schema-2 payload and contract,
required capabilities, exact target identity, device limits, and selected UUID.
It rechecks the runner around discovery and compares the worker's exact HELLO
description before OPEN. All requested native modules must load before the
application receives a session.

Verified payload bytes are written into a private temporary directory. The
session retains those files until native cleanup completes, including errors
while loading a later module. Filesystem and selection errors remain distinct
from typed service transport/backend errors, whose bounded native diagnostic
tails are preserved.

Buffer and module handles belong to one connection. Passing a handle to another
session fails locally. Releasing a buffer, closing a session, or a fatal worker
failure invalidates the associated handles. Context-manager exit closes the
worker and releases remaining resources. The existing service limits still
apply: 64 live buffers, 65556 FP32 elements per buffer, and 65539 elements per
launch. READ and explicit synchronization establish completion; a successful
AMD/NVIDIA LAUNCH reply may only acknowledge queued work.

Each session owns an independent worker. Closing one session must leave another
usable. Moving an intermediate result between devices requires an explicit READ
into host memory followed by WRITE to the other session. This API does not
provide cross-vendor pointers, peer memory, shared events, or automatic transfers.
It does not promise simultaneous kernel execution.

Hashes provide integrity relative to the supplied metadata. The installed
Python code and catalog/registry publisher are trusted inputs. Separate pack
reads do not make publication atomic, and executable hashes do not pin an
inode through process creation. Keep distribution updates serialized with use.

## Installed consumer validation

The packaged example runs with isolated Python import settings and checks that
the public package, private utilities, and kpack code all come from the installed
runtime tree. It does not add checkout paths to Python's import search path.

```sh
.venv/bin/python -I \
  build/multi-vendor-contracts/dist/multi-vendor-modules/share/therock/examples/packed_session_client.py \
  --dist-root build/multi-vendor-contracts/dist/multi-vendor-modules \
  --target amd:hip:gfx1201 --format hsaco \
  --peer-target nvidia:cuda:sm_120 --peer-format mixed
```

Single-device cases exercise six sizes, three scalars, offset transfers,
SAXPY/ReLU/SAXPY composition, and reused module handles. The paired case retains
both vendors' workers, passes intermediate data through host memory in both
directions, rejects foreign handles, closes one worker, and checks the surviving
session. Exact arithmetic checks use bounded dyadic inputs; they do not establish
a general floating-point error guarantee.

On Shark-a, the focused regression suite passed **345 tests** with no skips.
The native distribution passed **26 CTests** and the combined HIP/native
configuration passed **30 CTests**. Each includes four installed single-worker
cases (AMD HSACO and NVIDIA cubin, PTX, and mixed formats), plus one paired
AMD/NVIDIA case. Every case checks 18 size/scalar combinations; the paired case
also verifies a launch after closing its first worker. These are overlapping
validation profiles, not additive unique-test totals.

The paired consumer also passed from a copied distribution outside the checkout
and an unrelated working directory, using Python `-I`. Its import-origin record
points to the relocated runtime. The temporary copy was removed after validation.
Eight packaging tests cover stable no-op output, changed or missing source/output
rejection, and direct child-install preservation of previous stage contents.

Results are recorded with the final source and artifact snapshot under
`build/multi-vendor/validation-results/installed-client/`. Intel B70 and NVIDIA
SM90 execution remain deferred to matching hardware. Existing Intel offline
checks still pass; they do not qualify execution of this client on the B70.

## Alternatives considered

- **Keep verification only in command-line fixtures.** That leaves applications
  rebuilding selection, device checks, and temporary-file lifetime themselves.
  The shared selection helpers now serve both commands and the application API.
- **Expose the trusted-path client directly as the public API.** Its raw LOAD
  operation is useful at the protocol boundary, but does not enforce logical
  pack selection. The public facade accepts only the preverified module set.
- **Load every vendor driver into one host process.** Existing worker isolation
  keeps native runtime state and unknown-completion termination contained. A
  shared-library C ABI remains a separate design and qualification task.
- **Ship the checkout as the runtime.** A selected installed Python runtime and
  consumer example make its dependencies and relocatability testable. Packaging
  a published wheel, full toolchain closure, and production cache reuse remain
  separate work.

The next interfaces still need agreed semantics: general kernel argument layouts,
public context/queue/event ownership, library providers, and a shared-library ABI.
See the [architecture roadmap](multi-vendor-architecture.md#roadmap-and-acceptance-gates).
