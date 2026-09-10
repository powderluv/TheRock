# Bounded SGEMM providers

**Status: experimental; validated on Shark-a Radeon and NVIDIA GPUs.** This increment
adds an optional matrix operation to the installed multi-vendor client. AMD
workers call rocBLAS and NVIDIA workers call cuBLAS using their existing device,
buffers, and stream. Intel workers now have a separate, opt-in
[oneMKL provider](multi-vendor-intel-sgemm.md), compiled and linked with Intel
oneAPI; its device execution remains unqualified. Default Intel workers retain
SGEMM-disabled behavior.

The operation contract is `therock.blas.f32-sgemm-nn`, version 1, with SHA256
`fe1ce4851d9a3d5798e19b5e25ece80e4b4b4bbcdbf9dd350cea8912f3e2a5a0`.

The operation is column-major, non-transposed FP32 matrix multiplication:

```text
C = alpha * A * B + beta * C
A: m rows by k columns
B: k rows by n columns
C: m rows by n columns
```

This is a bounded provider interface, not a general BLAS implementation, common
kernel-argument ABI, or full ROCm math stack. The existing
[vector module contract](multi-vendor-module-contracts.md), packed SAXPY/ReLU
modules, and explicit target/format selection remain separate.

## Build and dependencies

Use a new `build/multi-vendor-sgemm` build directory, with the source and Python
prerequisites from the [getting-started guide](multi-vendor-getting-started.md).
The parent option `THEROCK_ENABLE_MULTI_VENDOR_SGEMM` defaults to OFF. Enable it
explicitly for the Radeon and NVIDIA profile:

```sh
cmake -S . -B build/multi-vendor-sgemm -GNinja \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -DTHEROCK_BUILD_PROFILE=multi-vendor-hip \
  '-DTHEROCK_MULTI_VENDOR_TARGETS=amd:hip:gfx1201;nvidia:cuda:sm_120' \
  -DTHEROCK_ENABLE_MULTI_VENDOR_VALIDATION=OFF \
  -DTHEROCK_ENABLE_MULTI_VENDOR_MODULES=ON \
  -DTHEROCK_ENABLE_MULTI_VENDOR_SGEMM=ON \
  -DTHEROCK_MULTI_VENDOR_AMD_ROOT=/opt/rocm \
  -DTHEROCK_MULTI_VENDOR_CUDA_ROOT=/usr/local/cuda
cmake --build build/multi-vendor-sgemm --target therock-dist --parallel 8
```

The parent enables SGEMM only for AMD/NVIDIA native children. The standalone
child option is `THEROCK_MODULE_ENABLE_SGEMM`, also OFF by default. Intel uses
the independent parent option `THEROCK_ENABLE_MULTI_VENDOR_INTEL_SGEMM` and
requires an explicit oneAPI SDK. Other selected targets can remain in the same
parent build with their existing module functionality.
Configured provider headers and libraries must resolve within the selected SDK,
and stub libraries are rejected. Enabled workers retain the selected provider
library directory in their installation RPATH.

AMD needs the selected ROCm installation's rocBLAS headers, shared library, and
runtime dependencies. NVIDIA needs the selected CUDA installation's cuBLAS
headers, shared library, and runtime dependencies. The workers link these
libraries when SGEMM is enabled, so the library dependencies apply even to a
vector-only application using that enabled worker. Only BLAS handle creation is
lazy. Disabling SGEMM excludes BLAS headers and symbols from the worker build.

Python, `msgpack`, `zstandard`, the vendor user-space libraries, and GPU drivers
remain external dependencies. Selective exports retain the compiled worker's
provider capabilities and dependency requirements; they do not bundle BLAS
libraries or convert an enabled worker into a disabled one. Imported SDK content
locks cover declared inputs as described in the
[input guide](multi-vendor-inputs.md), without capturing a complete runtime
library closure or authenticating the provider installation.

## Operation contract and selection

The [SGEMM contract model](../../build_tools/_therock_utils/sgemm_contract.py)
defines ABI `therock.blas.f32-sgemm-nn`, version 1. Its canonical digest is
negotiated independently of the vector module ABI.

| Property           | Contract                                                              |
| ------------------ | --------------------------------------------------------------------- |
| Element type       | IEEE FP32                                                             |
| Layout             | Column-major; neither input is transposed                             |
| Dimensions         | `m`, `n`, and `k` each lie in 1 through 256                           |
| Buffer capacity    | Existing limit of 65556 FP32 elements per buffer                      |
| Leading dimensions | `lda >= m`, `ldb >= k`, `ldc >= m`; each at most 65556                |
| Offsets            | Nonnegative FP32-element offsets from each buffer's start             |
| Scalars            | Alpha and beta must be finite and representable as FP32               |
| Aliasing           | C cannot use either input's buffer handle; A and B may share a handle |
| Submission         | Ordered within the worker; Intel currently completes before reply     |
| Completion         | READ or explicit synchronization establishes completion               |

The worker validates these spans using 64-bit arithmetic before forming pointers
or calling the provider:

```text
a_offset + (k - 1) * lda + m <= A.capacity
b_offset + (n - 1) * ldb + k <= B.capacity
c_offset + (n - 1) * ldc + m <= C.capacity
```

Leading dimensions include inter-column padding. For example, A element at row
`i`, column `j` is stored at `a_offset + i + j * lda`. There is no requirement that
matrix data begin at element zero. Zero dimensions are rejected, including cases
where a BLAS implementation might otherwise return immediately. The C/input
alias restriction applies to the entire handle, even when the proposed views
would be disjoint.

Enabled workers advertise `blas-sgemm-f32-nn-v1` together with exactly their
provider capability: `blas-provider-rocblas-v1` for AMD,
`blas-provider-cublas-v1` for NVIDIA, or `blas-provider-onemkl-v1` for opt-in Intel.
There is no fallback between providers, to a packed reference kernel, or to CPU
execution. Disabled workers advertise neither operation nor provider capability. Applications can require the operation capability when
opening a session to reject an unsupported worker before device discovery.

`session.sgemm_provider()` negotiates the exact provider and operation contract,
initializes the handle lazily, and returns cached `SgemmProviderInfo`. Its
`record()` contains schema version 1, kind `blas-provider`, scope
`loaded-provider`, vendor, provider, actual library version, operation ABI and
version, contract SHA256, and capabilities. The rocBLAS version is its runtime
version string; cuBLAS reports its integer runtime version as a string. These
observations distinguish the loaded library from the compiled capability
advertisement; neither establishes hardware or application qualification.

## Installed application interface

The [installed Python client](multi-vendor-client.md) adds:

```python
session.sgemm_provider()
session.sgemm(
    a,
    b,
    c,
    m=m,
    n=n,
    k=k,
    lda=lda,
    ldb=ldb,
    ldc=ldc,
    a_offset=0,
    b_offset=0,
    c_offset=0,
    alpha=1.0,
    beta=0.0,
)
```

`sgemm` validates local arguments and buffer ownership before lazy provider
negotiation. It returns no new handle. C remains the caller's existing allocation
and contains the updated matrix after completion.

For matrix multiplication without packed kernels, use
[`open_sgemm_session`](multi-vendor-sgemm-sessions.md). It verifies the worker and
negotiates the provider without reading kernel catalogs or loading modules.
`open_session` retains its one-to-32 packed `ModuleRequest` policy for applications
that compose kernels with SGEMM. After building the enabled distribution, run:

```sh
module_dist="$PWD/build/multi-vendor-sgemm/dist/multi-vendor-modules"
PYTHONPATH="$module_dist/share/therock/python" \
  .venv/bin/python -B - "$module_dist" <<'PYTHON'
from array import array
from pathlib import Path
import sys
from therock_multi_vendor import open_sgemm_session

with open_sgemm_session(Path(sys.argv[1]), "nvidia:cuda:sm_120") as session:
    a = session.allocate(6)
    b = session.allocate(6)
    c = session.allocate(4)
    session.write(a, 0, array("f", [1, 4, 2, 5, 3, 6]))
    session.write(b, 0, array("f", [7, 9, 11, 8, 10, 12]))
    session.write(c, 0, array("f", [0, 0, 0, 0]))
    print(session.sgemm_provider().record())
    session.sgemm(a, b, c, m=2, n=2, k=3, lda=2, ldb=3, ldc=2)
    result = session.read(c, 0, 4)
    assert result == array("f", [58, 139, 64, 154])
    print(list(result))
PYTHON
```

For the Radeon, use `amd:hip:gfx1201`. Device-index
and UUID selection remain unchanged. Buffers cannot cross sessions; transferring
a result to another vendor still requires a host READ followed by WRITE.

The example's small integer calculation is exactly representable. General
SGEMM results require a numerical tolerance accounting for dot-product length,
input magnitudes, cancellation, and alpha/beta scaling. Cross-provider bitwise
identity, arbitrary NaN/Inf behavior, reproducibility across library versions,
and performance are outside this qualification.

## Native execution and failures

For AMD/NVIDIA, the BLAS handle is bound to the existing nonblocking stream and host scalar
mode. AMD explicitly selects `rocblas_default_math`; NVIDIA selects
`CUBLAS_PEDANTIC_MATH`. Both disable atomic algorithms through their respective
handle settings. The implementation does not opt into XF32, TF32, or reduced
precision emulation. Algorithm choice and library-owned workspace remain vendor
implementation details. The API and stream/context requirements follow the
[cuBLAS documentation](https://docs.nvidia.com/cuda/archive/13.2.0/cublas/index.html)
and [rocBLAS documentation](https://rocm.docs.amd.com/projects/rocBLAS/en/docs-7.2.0/reference/level-3.html);
the implementation was compiled against the selected local SDK headers.

Intel borrows the native Level Zero context/device into a separate in-order SYCL
queue and explicitly drains work between the two queues. Its allocation checks,
precision policy, and failure handling are described in the
[Intel integration guide](multi-vendor-intel-sgemm.md).

The existing [service RPC](multi-vendor-service.md) frame version remains 1.
Opcode 12 negotiates provider, ABI, ABI version, and contract hash after OPEN;
it returns length-prefixed loaded-provider JSON. Opcode 13 carries A/B/C buffer
IDs, three element offsets, `m/n/k`, `lda/ldb/ldc`, and alpha/beta. It requires
successful negotiation. Existing operations and the vector launch ABI are
unchanged. Older or disabled workers cannot satisfy the new capability request.
Use the matching bundled Python runtime with an enabled worker: older host
clients intentionally reject unfamiliar capabilities in its description.

Calls on one session remain serialized. A successful SGEMM reply can precede
GPU completion. READ, synchronization, and resource-release operations preserve
the existing stream ordering. There are no public events, cancellation, or
concurrent-kernel guarantees.

Provider configuration and submission errors are reported as backend failures;
the worker drains before releasing referenced buffers, modules, and the BLAS
handle. The provider owner also drains before destroying its workspace, then
stream/context cleanup follows. Partial handle setup retains ownership through
failure cleanup. Unknown completion uses the existing bounded queue-wait and
isolated-process termination policy; this does not bound every vendor library
call or provide rollback after a failed operation.

## Validation status and limits

The initial bounded SGEMM checkpoint
(`610e6fd92760b40489c5f9f4948ed37e47c4dde0`) passed **32 applicable CTests** on Shark-a,
including Radeon, NVIDIA, and paired SGEMM consumers. The default combined
HIP/native profile passed **33 CTests** with SGEMM disabled. The focused Python
regression suite passed **394 tests, with no skips**, including 32 new contract,
client, protocol, build, and numerical-fixture tests. These suites overlap and
are not an additive count of unique tests.

Observed providers were rocBLAS `5.2.0.5b515cf1bc` on Radeon AI PRO R9700
(`amd:hip:gfx1201`) and cuBLAS version value `130400` on RTX PRO 6000 Blackwell
(`nvidia:cuda:sm_120`). The four-target enabled build also compiled and linked
the SM90 worker and compiled Intel SPIR-V with SGEMM disabled; neither target
received hardware execution qualification.

An AMD/NVIDIA export from a copied distribution passed the paired SGEMM example
from an unrelated working directory after relocation. Its exact inventory passed
verification before and after execution. The default AMD/NVIDIA/Intel runner
descriptions matched the preceding checkpoint, and their dynamic dependencies
contained no BLAS library. The enabled AMD/NVIDIA workers linked the selected
BLAS library with no stub directory in the runtime search path.

The `sgemm` CTest label selects
single-target consumers and AMD/NVIDIA paired consumers from the installed
runtime. Run the new cases and then the complete configured profile:

```sh
ctest --test-dir build/multi-vendor-sgemm -L sgemm --output-on-failure
ctest --test-dir build/multi-vendor-sgemm --output-on-failure
```

The installed correctness example can also run directly, including after an
[exact-target export](multi-vendor-distributions.md):

```sh
module_dist="$PWD/build/multi-vendor-sgemm/dist/multi-vendor-modules"
.venv/bin/python -I "$module_dist/share/therock/python/therock_multi_vendor/sgemm_example.py" \
  --dist-root "$module_dist" --target amd:hip:gfx1201 --format hsaco \
  --peer-target nvidia:cuda:sm_120 --peer-format mixed
```

For a build that additionally contains unqualified SM90 or Intel targets, apply
`-E 'intel|sm90'` to hardware runs until those devices are available. SGEMM cases
are registered only for enabled providers; selecting an NVIDIA architecture remains an
exact hardware requirement, not a fallback to another card.

Completed checks cover padded
rectangular matrices, offsets, alpha/beta cases, numerical error bounds,
unchanged inputs and guards, repeated operations, composition with existing
packed modules, foreign handles, malformed requests, and cleanup after failure.
The hardware records identify the selected device UUID, provider, loaded library
version, runner hash, and operation contract. Evidence is retained on Shark-a in
`build/multi-vendor/validation-results/sgemm-provider/`, separately from older
checkpoints. Build outputs and machine-local evidence are not included in a clone.

Radeon and NVIDIA validation belongs in the dedicated
`build/multi-vendor-sgemm` tree. Intel remains a compile/offline regression target
with explicit unsupported SGEMM behavior when its separate provider option is OFF. Earlier vector-module, installed-client,
and selective-export results do not qualify this new operation. Intel B70 and
NVIDIA SM90 execution remain unqualified until matching hardware is tested.

Transpose variants, row-major wrappers, batching, other dtypes, larger or empty
matrices, provider switching, general argument layouts, public event handles,
and a shared-library ABI remain future work. This increment establishes one
additional operation and provider boundary while preserving the existing
pack-based module path.

## Alternatives considered

- **HIP/hipBLAS source wrapper:** keeps a familiar source API and may be useful
  for existing applications. It adds another adapter and still needs a supported
  backend library, numerical policy, stream binding, and qualification. The
  existing HIP-on-CUDA baseline remains available; this operation directly binds
  rocBLAS/cuBLAS to the already implemented native worker.
- **Portable kernels in typed packs:** would make math implementations participate
  directly in existing multi-pack/multi-architecture selection and could supply
  an Intel implementation. Kernel tuning, numerical behavior, and performance
  would become TheRock-owned work. The pack layer continues to serve custom
  kernels while vendor libraries supply this initial matrix operation.
- **Generic dynamically loaded provider plugin:** could separate provider delivery
  from the worker binary. A stable plugin ABI, library ownership, dependency
  resolution, and complete runtime manifests need further design. This bounded
  opt-in linked provider provides a testable contract before broadening that ABI.
- **Intel oneMath/oneMKL integration:** the subsequent opt-in
  [oneMKL integration](multi-vendor-intel-sgemm.md) supplies this provider boundary
  for native Level Zero allocations. Hardware interoperation and numerical
  qualification remain pending. A broader oneMath dispatcher remains an option.
