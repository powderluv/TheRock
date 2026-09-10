# Intel oneMKL SGEMM integration

**Status: experimental implementation; Intel device execution is unqualified.**
The native Level Zero worker can optionally execute the existing bounded FP32
SGEMM operation through oneMKL. The same installed Python APIs, worker protocol,
exact target selection, and multi-pack distribution serve all three vendors.
Radeon/NVIDIA regression runs and Intel compile/CPU tests are separate evidence;
neither qualifies the assumed Arc Pro B70.

## Build prerequisites and SDK selection

The independent `THEROCK_ENABLE_MULTI_VENDOR_INTEL_SGEMM` option defaults to OFF.
It requires native modules, build testing, and a selected Intel target. The
existing `THEROCK_ENABLE_MULTI_VENDOR_SGEMM` option continues to control only
rocBLAS/cuBLAS. Either option can be enabled independently. Disabled Intel workers
retain their Level Zero-only dependencies and advertise no SGEMM capability.

Use the source, Python, Level Zero, and SPIR-V prerequisites from the
[getting-started guide](multi-vendor-getting-started.md). Intel math additionally
needs a coherent DPC++/C++ compiler and oneMKL SYCL BLAS installation, including
its oneTBB, OpenMP, Unified Runtime, Unified Memory Framework, and TCM/hwloc
dependencies.
Follow Intel's [Linux installation guide](https://www.intel.com/content/www/us/en/docs/oneapi-toolkit/installation-guide-linux/latest/overview.html)
to configure its signed package repository. The package roots used on Shark-a are:

```sh
sudo apt-get install intel-oneapi-compiler-dpcpp-cpp-2026.1 \
  intel-oneapi-mkl-sycl-blas-2026.1 intel-oneapi-mkl-sycl-include-2026.1
```

On Shark-a, these packages and their Intel dependency closure were downloaded
and extracted locally with `dpkg-deb --extract`; no system installation was
changed. The signed repository index and all 27 package SHA256 values were
verified. The compiler package version is `2026.1.1-325`, oneMKL is
`2026.1.0-236`, and the extracted prefix is:

```text
/home/nod/github/TheRock/build/oneapi-deps/sdk/opt/intel/oneapi
```

Raw extraction does not run package maintainer scripts. Relative `latest` aliases
were created for the single installed version of each component; this also
resolves the compiler's link to `umf/latest`. No SDK symlink escapes the prefix.
The evidence directory retains `sdk-packages.json`, signature verification,
and `layout-aliases.json`. Package selection is an observed SDK checkpoint;
future repository revisions can select different bytes. The content lock below
records the actual tree used by a build.

Configure a new tree, adjusting the Level Zero and oneAPI prefixes if installed
elsewhere:

```sh
oneapi_root="$PWD/build/oneapi-deps/sdk/opt/intel/oneapi"
cmake -S . -B build/multi-vendor-intel-sgemm -GNinja \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -DTHEROCK_BUILD_PROFILE=multi-vendor-hip \
  '-DTHEROCK_MULTI_VENDOR_TARGETS=amd:hip:gfx1201;nvidia:cuda:sm_120;nvidia:cuda:sm_90;intel:level-zero:xe2-b70' \
  -DTHEROCK_ENABLE_MULTI_VENDOR_VALIDATION=OFF \
  -DTHEROCK_ENABLE_MULTI_VENDOR_MODULES=ON \
  -DTHEROCK_ENABLE_MULTI_VENDOR_SGEMM=ON \
  -DTHEROCK_ENABLE_MULTI_VENDOR_INTEL_SGEMM=ON \
  -DTHEROCK_MULTI_VENDOR_AMD_ROOT=/opt/rocm \
  -DTHEROCK_MULTI_VENDOR_CUDA_ROOT=/usr/local/cuda \
  -DTHEROCK_MULTI_VENDOR_LEVEL_ZERO_ROOT="$PWD/build/level-zero-deps/sdk/usr" \
  -DTHEROCK_MULTI_VENDOR_ONEAPI_ROOT="$oneapi_root" \
  -DTHEROCK_MULTI_VENDOR_SYCL_COMPILER="$oneapi_root/compiler/2026.1/bin/icpx" \
  -DTHEROCK_MULTI_VENDOR_ONEMKL_ROOT="$oneapi_root/mkl/2026.1"
cmake --build build/multi-vendor-intel-sgemm --target therock-dist --parallel 8
ctest --test-dir build/multi-vendor-intel-sgemm -E 'intel|sm90' --output-on-failure
```

The last command intentionally selects the GPUs available on Shark-a. Intel and
SM90 test registration does not mean those devices have passed. Use `/opt/intel/oneapi`
for a conventional system installation. When explicit compiler/oneMKL paths are
omitted, discovery uses `compiler/latest/bin/icpx` and `mkl/latest` within the
specified prefix. No global oneMKL lookup or implicit `-qmkl` discovery is used.

Only the Intel math child's C++ compiler changes to `icpx -fsycl`; AMD/NVIDIA
host compilers and the Intel OpenCL-to-SPIR-V module compiler remain separately
selected. The entire oneAPI prefix and the actual SYCL compiler file participate
in [imported-input locks](multi-vendor-inputs.md). Headers, linked libraries, and
runtime directories must remain inside that tree and outside stub directories.
The worker retains explicit dynamic dependencies on the imported UMF and
TCM/hwloc libraries, with their SDK directories in RPATH, so dynamically loaded
Level Zero adapters can resolve that dependency chain without global SDK setup.
The SDK and GPU driver remain external to distributions and selected exports.
Declared-input coverage still does not capture a complete runtime closure or
permit artifact-cache reuse.

## Context, memory, and ordering

The provider advertises `blas-provider-onemkl-v1` together with
`blas-sgemm-f32-nn-v1`. It negotiates the existing
`therock.blas.f32-sgemm-nn` version 1 contract, SHA256
`fe1ce4851d9a3d5798e19b5e25ece80e4b4b4bbcdbf9dd350cea8912f3e2a5a0`.
Provider information reports `onemkl`, vendor `intel`, and the runtime
`mkl_get_version` major/update/patch values. No wire, vector ABI, catalog,
registry, or export schema changes are introduced.

A provider borrows the worker's existing Level Zero device and context through
SYCL interoperability with `ownership::keep`; it owns a separate in-order SYCL
queue. Native handles round-trip through SYCL and must match. Before forming
offset matrix pointers, each allocation must be recognized as device USM in the
borrowed context and associated with the selected device. A mismatch fails;
there is no host-copy or CPU fallback.

Existing Level Zero copies and kernels complete before their replies. The
provider drains the native queue before submitting oneMKL and drains its SYCL
queue before replying. These explicit completion boundaries allow native packed
kernels and oneMKL to reuse device buffers without assuming two physical queues
share implicit ordering. Intel currently offers no overlap between these calls.
The BLAS call explicitly selects `compute_mode::standard`; numerical bounds and
matrix restrictions follow the [common SGEMM contract](multi-vendor-sgemm.md).

The implementation requires Intel's supported SYCL queue-empty and queue-prod
extensions: nonblocking `ext_oneapi_prod()` flushes pending submission, then
`ext_oneapi_empty()` is polled against the existing 30-second completion deadline.
Async errors are captured and reported. Partial submission is drained before an
exception can release referenced buffers. Unknown completion terminates the
isolated worker without freeing potentially active resources. The provider queue
is destroyed before the borrowed native context. This bounds the completion
poll, not every vendor library call; the host's existing request deadline remains
in force.

The interoperability model follows Intel's
[Level Zero backend extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_backend_level_zero.md),
[queue-empty extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_queue_empty.asciidoc),
and [queue-prod extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_prod.asciidoc).
A regular Level Zero command queue cannot be assumed compatible with the
immediate-command-list interface used by the newer SYCL Level Zero adapter;
creating a separate queue avoids that dependency.

## Device acceptance when B70 arrives

Install a compute driver supporting the actual card and verify its identity.
The current target assumption is `intel:level-zero:xe2-b70`, PCI ID `0xe223`.
Keep the exact identity check: update the target configuration deliberately if
the delivered card differs. Run the registered Intel suite, then the two public
consumers directly:

```sh
ctest --test-dir build/multi-vendor-intel-sgemm -L intel --output-on-failure
module_dist="$PWD/build/multi-vendor-intel-sgemm/dist/multi-vendor-modules"
.venv/bin/python -I "$module_dist/share/therock/python/therock_multi_vendor/sgemm_provider_example.py" \
  --dist-root "$module_dist" --target intel:level-zero:xe2-b70 \
  --expect-device-id 0xe223
.venv/bin/python -I "$module_dist/share/therock/python/therock_multi_vendor/sgemm_example.py" \
  --dist-root "$module_dist" --target intel:level-zero:xe2-b70 --format spirv \
  --expect-device-id 0xe223
```

The first consumer opens a verified provider without loading any module. The
second composes packed SAXPY, SGEMM, and ReLU through the same device buffers.
Both check seven matrix/stride/offset/scalar cases, numerical error bounds,
unchanged inputs, and padding guards. Record device UUID, actual provider version,
runner hash, input locks, and driver before promoting any Intel support claim.
Also qualify release/close after failures and repeated operations, then export an
Intel selection and repeat after relocation. CPU shims do not establish native
USM interoperability, driver behavior, or numerical correctness on B70.

## Validation record

On Shark-a, the four-target Intel-enabled distribution compiled and linked with
DPC++ 2026.1.1 and oneMKL 2026.1.0. **444 focused regression tests passed without
skips**, including 24 additional Intel contract, native control-flow, build-guard,
and installed-consumer cases. **35 applicable native CTests** passed on Radeon
R9700 and RTX PRO 6000, and the default combined HIP/native build passed **33
CTests** with SGEMM disabled. These suites overlap.

Generated toolchains and actual compile commands confirmed that only the Intel
math worker used `icpx`; its SPIR-V frontend remained the AMD LLVM compiler.
The installed Intel worker's offline description advertised the expected oneMKL
capability, and its direct dynamic dependencies resolved without SDK environment
variables. A separate loader probe used the installed worker's exact RUNPATH
and retained dependencies to open both Level Zero UR adapters with `RTLD_NOW`,
without invoking device APIs; UMF and hwloc resolved inside the declared SDK.
The rebuilt Intel OFF worker retained its previous description bytes and no
oneAPI library dependencies. An Intel-only export from a copied distribution retained that
capability after relocation and passed inventory verification before and after
the offline check. No Intel discovery, allocation, kernel, or BLAS call was run.

Validation evidence for this increment is retained on Shark-a under
`build/multi-vendor/validation-results/intel-sgemm/`. Build outputs and local SDKs
are ignored and are not included in a clone. Intel hardware and NVIDIA SM90
execution remain unqualified; performance and complete ROCm application
compatibility are outside this increment.

## Alternatives considered

- **HIP wrappers or oneMath dispatch:** may serve applications needing those
  source APIs, but still require a supported Intel backend, queue interoperability,
  numerical policy, and hardware validation. The existing worker boundary permits
  a direct oneMKL provider without changing application transport.
- **Borrow the existing native queue into SYCL:** would avoid the second queue,
  but adapter generations differ in the accepted Level Zero queue handle kind.
  Explicitly ordered owned SYCL execution provides a narrower initial contract.
- **Copy into SYCL-owned buffers:** would avoid native allocation interoperability
  but introduce hidden copies and break direct packed-kernel/library buffer reuse.
  This implementation checks native allocation compatibility and fails explicitly.
- **Ship portable SGEMM kernels in multi-architecture packs:** remains possible
  within the existing catalog design, with tuning and numerical behavior owned
  by that implementation. Vendor BLAS is an independent provider boundary.
