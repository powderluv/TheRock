# Experimental multi-vendor HIP builds

The `multi-vendor-hip` build profile uses TheRock's subproject, artifact, and
consumer graph infrastructure for backend-specific HIP SDK consumers. It is an
opt-in first step toward multi-vendor ROCm builds. The default `rocm` profile
retains the existing AMD build.

## Supported integration paths

| Target example             | Compiler/runtime            | SDK source                                                         | Qualification                                  |
| -------------------------- | --------------------------- | ------------------------------------------------------------------ | ---------------------------------------------- |
| `amd:hip:gfx1201`          | AMD Clang / native HIP      | Installed Linux ROCm SDK                                           | Requires a Radeon test run                     |
| `nvidia:cuda:sm_120`       | NVCC / CUDA runtime         | HIP and HIPNV headers built from the pinned `rocm-systems` sources | Experimental; requires an NVIDIA test run      |
| `intel:level-zero:xe2-b70` | chipStar Clang / Level Zero | Explicit installed chipStar SDK                                    | Experimental; B70 hardware validation deferred |

NVIDIA uses the retained HIP-on-CUDA header adapter as a source portability
baseline. There is no AMD HIP binary ABI compatibility: each backend has its own
compiler invocation, SDK dependencies, executable, and runtime. AMD and Intel
SDKs are imported prerequisites in this increment; the profile does not rebuild
their runtimes or install the CUDA toolkit.

Target identity includes vendor, backend, processor, and optional AMD features.
For example, `amd:hip:gfx942:xnack+:sramecc-` retains the feature signs in compiler
arguments and maps to the artifact-safe key
`amd-hip-gfx942-sramecc-off-xnack-on`. A CUDA `sm_120` maps to CMake HIP architecture
`120`. The Intel B70 label identifies requested hardware; it does not invent a
compiler architecture flag. The HIP consumer delegates Intel SPIR-V compilation to chipStar; the native
module consumer uses a portable SPIR64 frontend and LLVM-to-SPIR-V translator.

## Build and validate

Use Python 3.10 or newer, install TheRock's Python requirements, and fetch sources
as usual. Full source
fetching works; an isolated profile checkout can fetch just its source stage:

```sh
python build_tools/fetch_sources.py --stage multi-vendor-hip
```

The NVIDIA HIP language requires CMake 3.28 or newer. Configure one graph with
both Radeon and NVIDIA targets:

```sh
cmake -S . -B build/multi-vendor -GNinja \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -DTHEROCK_BUILD_PROFILE=multi-vendor-hip \
  '-DTHEROCK_MULTI_VENDOR_TARGETS=amd:hip:gfx1201;nvidia:cuda:sm_120' \
  -DTHEROCK_MULTI_VENDOR_AMD_ROOT=/opt/rocm \
  -DTHEROCK_MULTI_VENDOR_CUDA_ROOT=/usr/local/cuda
cmake --build build/multi-vendor
ctest --test-dir build/multi-vendor --output-on-failure
```

Select only `nvidia:cuda:sm_120` when an AMD SDK is unavailable, or only
`amd:hip:gfx1201` when a CUDA toolkit is unavailable. Each additional selected
target gets a separate child build. This supports multiple backend and processor
outputs in one build graph; it does not merge their device code into one
executable. Compilation and packaging do not require access to a GPU. CTest does,
and missing or mismatched hardware causes a failure.

The validation executable runs real HIP allocation, transfer, arithmetic,
shared-memory reduction, stream, and event operations with CPU reference checks.
See [the validation consumer](../../tests/multi_vendor/README.md) for standalone
use and device selection. Keep GPU test results separate from target selection:
the generated `gpu_targets.json` is explicitly unvalidated configuration metadata.
The aggregate validation artifact installs this manifest under
`share/therock/multi-vendor/`. Validation artifact cache reuse is disabled because
installed SDKs and compilers are external inputs without content identities.
Start a clean build directory after changing an SDK or compiler in place.

For Intel, use CMake 4.3 or newer and an installed chipStar toolchain:

```sh
cmake -S . -B build/intel-hip -GNinja \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -DTHEROCK_BUILD_PROFILE=multi-vendor-hip \
  -DTHEROCK_MULTI_VENDOR_TARGETS=intel:level-zero:xe2-b70 \
  -DTHEROCK_ENABLE_MULTI_VENDOR_MODULES=OFF \
  -DTHEROCK_MULTI_VENDOR_INTEL_ROOT=/path/to/chipstar \
  -DTHEROCK_MULTI_VENDOR_INTEL_COMPILER=/path/to/chipstar-clang/bin/clang++
cmake --build build/intel-hip
```

Intel runtime tests require `CHIP_BE=level0` and `CHIP_DEVICE_TYPE=gpu`, set by
CTest. The B70 target also checks the device name. There is no CPU/OpenCL fallback
that can qualify the card. Until the SDK and hardware tests have run, this is an
integration path, not a validated B70 backend.

## Artifacts and extension boundary

The NVIDIA SDK is a backend-specific, target-neutral `hip-nvidia-sdk` artifact.
It is staged into its own `hip-nvidia` distribution. Validation outputs use
`bin/<target-key>/` so the same executable name from different targets cannot
collide. The `multi-vendor-validation` artifact has an explicit bundle name
derived from the selected target keys, independent of AMD family naming.
TheRock's regular subproject manifest and consumer graph record the build edges.

The artifact API accepts `DIST_BUNDLE_NAME` for non-AMD device bundles while
retaining the existing AMD naming behavior for current callers. Explicit bundle
names reject the AMD-only kpack split pipeline. Configuration fails instead of
feeding CUDA or Intel binaries into AMD code-object surgery.

## Native module packs

The profile also builds a `multi-vendor-modules` artifact by default when testing
is enabled. Disable it with `THEROCK_ENABLE_MULTI_VENDOR_MODULES=OFF`. It compiles
SAXPY and ReLU into separate modules: AMD HSACO, NVIDIA cubin plus PTX, and
Intel SPIR-V. The
NVIDIA host loader calls the CUDA Driver API directly and links `libcuda`, without
the HIP adapter or CUDA runtime library. The AMD loader calls native HIP module
APIs. The Intel runner links directly to the Level Zero loader and uses
`zeModuleCreate`, kernel arguments, command lists, and an asynchronous compute
queue. Each backend has allocation, transfer, launch, and result checks. See [native module validation](../../tests/multi_vendor/modules/README.md).

Two real KPAK v1 archives, one per logical module, each contain the selected
backend/architecture/format variants. A versioned `catalog.json` beside each pack
records canonical targets, payload formats, entry points, and SHA256 hashes. The
archive TOC key is `<logical-module>/<payload-format>` plus the exact canonical
target ID. Existing archive fields retain their legacy names; the default payload
type for existing writers remains `hsaco`.

The experimental catalog selector searches all provided packs for an exact
module/target/format match. It rejects ambiguous matches and verifies pack bytes,
payload bytes, metadata agreement, relative path containment, and the requested
entry point before starting a GPU process. Hashes provide integrity relative to
the catalog, not publisher authentication. There is no implicit architecture or
payload-format fallback. Canonical vendor IDs are never fed through the legacy
AMD ISA compatibility matcher.

Run the packaged path with the same build command above, then:

```sh
ctest --test-dir build/multi-vendor -L packed-module --output-on-failure
```

The distribution contains native runners under `bin/<target-key>/` and catalogs
and archives under `share/therock/packs/{saxpy,relu}/`. Raw compiler outputs remain
in child stages; only packed device payloads enter this distribution. Each ReLU
test supplies the SAXPY catalog first, exercising module lookup beyond the first
same-target pack. For an explicit NVIDIA PTX run:

```sh
module_dist="$PWD/build/multi-vendor/dist/multi-vendor-modules"
PYTHONPATH="$PWD/rocm-systems/shared/kpack/python" \
  .venv/bin/python build_tools/validate_multi_vendor_modules.py \
  --catalog "$module_dist/share/therock/packs/saxpy/catalog.json" \
  --catalog "$module_dist/share/therock/packs/relu/catalog.json" \
  --module validation/relu --target nvidia:cuda:sm_120 --format ptx \
  --entry-point therock_module_relu \
  --runner "$module_dist/bin/nvidia-cuda-sm120/therock_module_validation"
```

`build_tools/multi_vendor_pack.py` provides lower-level `create` and `extract`
commands for these catalogs. Its Python dependency is the fetched `rocm_kpack`
package; the example supplies that source package through `PYTHONPATH`. Creation
requires fresh catalog/pack destinations. The build assembler creates complete
packs privately and replaces its managed output files before stage installation.
This is file-level publication within a build, not a live atomic deployment API.

The existing C++ kpack loader also now checks the requested module and code-object
index before choosing a compatible AMD pack. This preserves its existing ISA,
feature-specificity, and path-order precedence while fixing the case where the
first compatible pack lacks the module. That source change is validated by the
standalone kpack runtime tests; the imported ROCm SDK itself is not rebuilt here.

## Native Intel Level Zero modules

Native Intel modules do not require chipStar or CMake's HIP language support.
They use ordinary C++ for the host and OpenCL C 1.2 for the validation kernels'
five-argument ABI. Clang emits portable `spir64-unknown-unknown` LLVM bitcode;
a matching `amd-llvm-spirv` or `llvm-spirv` translator emits SPIR-V 1.0 with the
Physical64/OpenCL memory model. `spirv-val --target-env opencl1.2` must pass before
a payload can enter a pack. The B70 product label never becomes a compiler ISA
flag. This validates the native module integration; it does not implement an
Intel HIP frontend or a general HIP runtime.

Configure all three native backends, with HIP consumers disabled:

```sh
cmake -S . -B build/multi-vendor-native-all -GNinja \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -DTHEROCK_BUILD_PROFILE=multi-vendor-hip \
  '-DTHEROCK_MULTI_VENDOR_TARGETS=amd:hip:gfx1201;nvidia:cuda:sm_120;intel:level-zero:xe2-b70' \
  -DTHEROCK_ENABLE_MULTI_VENDOR_VALIDATION=OFF \
  -DTHEROCK_MULTI_VENDOR_LEVEL_ZERO_ROOT=/path/to/level-zero-sdk \
  -DTHEROCK_MULTI_VENDOR_SPIRV_COMPILER=/opt/rocm/llvm/bin/clang \
  -DTHEROCK_MULTI_VENDOR_SPIRV_TRANSLATOR=/opt/rocm/llvm/bin/amd-llvm-spirv \
  -DTHEROCK_MULTI_VENDOR_SPIRV_VALIDATOR=/path/to/spirv-val
cmake --build build/multi-vendor-native-all
ctest --test-dir build/multi-vendor-native-all -LE intel --output-on-failure
```

The Level Zero SDK prefix must contain `include/level_zero/ze_api.h` and
`lib[64]/libze_loader.so` or a multiarch library directory. It is distinct from
`THEROCK_MULTI_VENDOR_INTEL_ROOT`, which remains the chipStar HIP SDK. The
SPIR-V frontend defaults to Clang in `THEROCK_MULTI_VENDOR_AMD_ROOT`, the
translator to a matching binary beside that frontend, and the validator to
`spirv-val` in the Level Zero SDK or `PATH`. An explicitly supplied frontend
allows an Intel-only build without an AMD SDK. Clang and the translator must
support the same LLVM bitcode version; compiler or validation errors fail the
build. Imported toolchains require a clean build after in-place changes.

The Level Zero runner selects only Intel GPU root devices, with indices sorted
by driver UUID then device UUID. It checks the Intel vendor ID `0x8086` and, for
`xe2-b70`, device ID `0xe223`, as listed in Intel's
[supported GPU table](https://dgpu-docs.intel.com/overview/supported-hardware/xe-driver-gpus.html).
Other Intel product identities require an explicit
`THEROCK_MULTI_VENDOR_INTEL_DEVICE_ID`. A conflicting B70 device ID is rejected.
Marketing-name checks are unnecessary: supported drivers may report different
names. `--expect-arch spirv` describes the payload format and does not identify
hardware.

When the card and its supported Intel compute driver are installed, run:

```sh
ctest --test-dir build/multi-vendor-native-all -L intel --output-on-failure
```

Those tests extract verified SPIR-V from the same two multi-vendor packs before
launching the native Intel runner. Missing drivers/devices fail; no CPU fallback
or success-by-skip is provided. The loader reports module build logs, uses finite
queue waits, and terminates on unknown completion without freeing resources
that could still be in use.

On Shark-a, both SPIR-V modules compile and pass offline validation using the
installed ROCm LLVM 22 frontend/translator. The C++ loader builds against Ubuntu's
Level Zero loader package 1.16.1 (headers exposing API 1.9), extracted into an
isolated build SDK. The three-vendor artifact is assembled with real compiler
outputs. Host-side extraction and rejection tests pass, and AMD/NVIDIA execution
is checked separately. **Intel execution remains unvalidated until the B70
arrives.** The loader library is not the Intel GPU compute driver; hardware
validation requires a B70-capable runtime, such as the release documented by
Intel for [compute-runtime 26.14.37833.4](https://dgpu-docs.intel.com/overview/release-notes/containers/compute-runtime/26.14.37833.4.html).

This remains an experimental build and validation layer. Full ROCm support still
needs math-library providers, backend capability/version negotiation, a general
launch ABI, and integration of the selected payload with production runtimes.
AMD code objects, CUDA cubin/PTX, and Intel SPIR-V require distinct compiler and
runtime handling; the existing AMD code-object surgery pipeline remains disabled
for these explicitly named bundles.
