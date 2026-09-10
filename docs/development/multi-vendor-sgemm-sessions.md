# SGEMM sessions without kernel modules

`open_sgemm_session` opens a verified native BLAS worker without selecting,
extracting, or loading a packed kernel. Applications doing only matrix
multiplication no longer need to declare an unrelated SAXPY module. AMD uses
rocBLAS, NVIDIA uses cuBLAS, and opt-in Intel uses oneMKL; the operation contract and native worker are the
same as the [bounded SGEMM implementation](multi-vendor-sgemm.md).

Use `open_session` when an application needs packed kernels and SGEMM together on
the same worker stream. Its one-to-32 module-request policy and verification order
are unchanged. `SgemmSession` exposes buffer allocation, transfer, release, SGEMM,
synchronization, provider information, and context-managed lifetime. It does not
expose module lookup or kernel launch methods.

## Build and run

Build with `THEROCK_ENABLE_MULTI_VENDOR_MODULES=ON` and
`THEROCK_ENABLE_MULTI_VENDOR_SGEMM=ON` using the
[provider build instructions](multi-vendor-sgemm.md#build-and-dependencies).
Default workers do not advertise SGEMM and reject this session request before
device discovery. Intel requires the separate
[oneMKL build option](multi-vendor-intel-sgemm.md); Intel execution is not yet
qualified. Vendor libraries and drivers remain external.

The installed example has no payload-format or module arguments:

```sh
module_dist="$PWD/build/multi-vendor-sgemm/dist/multi-vendor-modules"
.venv/bin/python -I "$module_dist/share/therock/python/therock_multi_vendor/sgemm_provider_example.py" \
  --dist-root "$module_dist" --target amd:hip:gfx1201 \
  --peer-target nvidia:cuda:sm_120
```

Omit `--peer-target` for one device. Use `--device` or `--device-uuid` to select the
primary device, and `--peer-device` or `--peer-device-uuid` for the peer. The two
forms are mutually exclusive for each worker. Omitted selectors use device index
zero. Intel callers can also pass `--expect-device-id 0xe223` to bind the expected
B70 PCI ID. The target must match exactly; SM120 execution does not qualify SM90.

A small application can call the installed public API directly:

```sh
PYTHONPATH="$module_dist/share/therock/python" \
  .venv/bin/python -B - "$module_dist" <<'PYTHON'
from array import array
from pathlib import Path
import sys
from therock_multi_vendor import open_sgemm_session

with open_sgemm_session(Path(sys.argv[1]), "nvidia:cuda:sm_120") as session:
    print(session.sgemm_provider().record())
    a, b, c = session.allocate(6), session.allocate(6), session.allocate(4)
    session.write(a, 0, array("f", [1, 4, 2, 5, 3, 6]))
    session.write(b, 0, array("f", [7, 9, 11, 8, 10, 12]))
    session.write(c, 0, array("f", [0, 0, 0, 0]))
    session.sgemm(a, b, c, m=2, n=2, k=3, lda=2, ldb=3, ldc=2)
    assert session.read(c, 0, 4) == array("f", [58, 139, 64, 154])
PYTHON
```

For Radeon, change only the target to `amd:hip:gfx1201`. Dimensions, leading
dimensions, offsets, finite FP32 scalars, alias restrictions, numerical policy,
and queued completion follow the existing SGEMM contract. Buffers belong to one
session; cross-vendor exchange uses explicit host reads and writes.

## Verification and lifetime

Opening loads the strict runner registry and requires both
`persistent-module-service` and `blas-sgemm-f32-nn-v1`. The client checks the exact
registered target, executable digest, compiled description, and observed device
identity. It rechecks the executable digest before starting the persistent worker,
which checks the UUID again during OPEN. Provider negotiation then verifies the
provider, ABI, version, and contract digest before the session is returned.

This negotiation eagerly creates the BLAS handle. An unsupported capability is a
local selection error; provider setup, protocol, and runtime failures use the
existing `ModuleServiceError` vocabulary and close/reap the worker. Successful
queries reuse the negotiated provider information. Closing or fatal failure
invalidates owned buffers; teardown drains queued work before freeing resources.
Calls on one session must remain serialized.

The registry schema still requires catalog path metadata, but this opening path
does not resolve catalog files or read archives. It creates no temporary payload
directory and sends no LOAD request. This is an execution API distinction:
[current exports](multi-vendor-distributions.md) still contain and verify their
selected kernel packs, runtime files, registry, and receipts. Deleting files from
an export invalidates its inventory even when those files are unnecessary for a
particular SGEMM call. No new distribution schema or packaging-pruning option is
introduced here.

The installed Python dependencies are unchanged. Runner and metadata hashes
provide integrity relative to the supplied distribution, not publisher
authentication. Distribution updates must be serialized with verification and
execution; executable paths are not pinned inodes. The worker still uses the
existing vector-capable transport handshake and device prerequisites internally.
This increment does not introduce a general plugin ABI or a new vendor runtime.

## Validation

On Shark-a, **420 focused regression tests passed without skips**, including
26 new session and installed-example tests. The enabled native profile passed
**35 applicable CTests**; the default combined HIP/native profile passed
**33 CTests**. These suites overlap and are not an additive unique-test count.
The four native worker binaries remained byte-for-byte identical to the initial
SGEMM checkpoint; this increment changes the host API and installed consumers.

A selected AMD/NVIDIA export from a copied distribution passed the paired example
from an unrelated working directory after relocation. Both devices were selected
by their observed UUID, and the export inventory verified before and after use.
The new provider-only consumers and the existing packed-kernel/SGEMM composition
cases both passed on Radeon R9700 and RTX PRO 6000 Blackwell.

The new `sgemm-session` CTest label selects Radeon, NVIDIA, and paired installed
consumers. Each worker checks seven dimension/stride/offset/scalar cases with a
cancellation-aware FP32 bound, unchanged inputs, and padding guards. The paired
case transfers data in both directions, rejects foreign handles, closes one
worker with queued SGEMM work, and verifies the peer remains usable.

```sh
ctest --test-dir build/multi-vendor-sgemm -L sgemm-session \
  -E 'intel|sm90' --output-on-failure
```

The existing packed-kernel/SGEMM interoperation tests remain separate. CPU tests
exercise constructor failure cleanup, rejected identities and capabilities,
missing or corrupt kernel files, ownership, and absence of module-load requests.
Hardware validation results for this increment are recorded separately under
`build/multi-vendor/validation-results/sgemm-sessions/` on Shark-a. Intel and SM90
hardware execution remain unqualified.

## Alternatives considered

- **Accept an empty request list in `open_session`:** would combine two different
  opening contracts in an API whose documented purpose is preloading verified
  modules. A separate constructor makes provider negotiation and the absence of
  kernel methods explicit while sharing selection and lifetime implementation.
- **Keep a dummy module declaration:** works with the preceding checkpoint, but
  forces a library-only application to depend on unrelated payload selection and
  extraction. The new API requires only the worker for execution.
- **Introduce a provider-only registry or export schema:** could remove unused
  kernel assets from delivery, but needs separate package ownership and dependency
  rules. This increment retains the existing delivery format and isolates the
  execution change for validation.
