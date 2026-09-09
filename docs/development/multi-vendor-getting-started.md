# Run the experimental multi-vendor TheRock build

This branch extends TheRock's build graph, artifacts, and typed module packs to
AMD HIP, NVIDIA CUDA, and Intel Level Zero. It includes the HIP-on-CUDA source
adapter as a baseline and separate native loaders for AMD HSACO, NVIDIA cubin/PTX,
and Intel SPIR-V. The installed Python client can keep Radeon and NVIDIA workers
alive in one application and explicitly transfer results through host memory.

Start with the two-target build below on Shark-a. The four-target build also
compiles Intel B70 and NVIDIA SM90 payloads; execution on those GPUs remains
unvalidated. This is an experimental fixed-vector module interface, not a full
ROCm runtime or a general replacement for HIP/library APIs. See the
[architecture](multi-vendor-architecture.md) for the design, alternatives to the
HIP wrapper, multi-pack/multi-architecture layout, and remaining acceptance gates.

## Tested reference environment

| Component                | Recorded configuration                                                                    |
| ------------------------ | ----------------------------------------------------------------------------------------- |
| Host                     | Shark-a, Ubuntu 24.04, x86-64 Linux                                                       |
| Radeon                   | Radeon AI PRO R9700, 32 GB; `amd:hip:gfx1201`                                             |
| NVIDIA                   | RTX PRO 6000 Blackwell Workstation Edition, 96 GB; `nvidia:cuda:sm_120`                   |
| Intel assumption         | Arc Pro B70; `intel:level-zero:xe2-b70`, expected PCI device ID `0xe223`; hardware absent |
| Additional NVIDIA target | `nvidia:cuda:sm_90`; compiler/pack checks only                                            |
| Host tools               | CMake 3.28.3, Ninja 1.11.1, Python 3.12.3, GCC/G++ 13.3.0                                 |
| AMD SDK                  | `/opt/rocm`, ROCm 7.2.0, AMD LLVM 22.0.0git                                               |
| CUDA SDK                 | `/usr/local/cuda`, CUDA 13.2, NVCC 13.2.78                                                |
| NVIDIA driver            | 595.71.05 at the validation checkpoint                                                    |
| Intel offline tools      | Level Zero 1.16.1 headers/loader; SPIRV-Tools 2025.1; AMD LLVM SPIR64 frontend/translator |

These are observed versions, not a compatibility matrix. On Shark-a the SDKs,
drivers, and host tools already exist. On another Ubuntu 24.04 machine, install
host prerequisites first:

```sh
sudo apt-get update
sudo apt-get install git build-essential cmake ninja-build python3-venv \
  python3-dev pkg-config libmagic1
```

Install a GPU driver and SDK appropriate for each selected backend, following the
[ROCm 7.2.0 installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/)
and [CUDA 13.2 Linux guide](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-installation-guide-linux/contents.html).
The commands below import those installations; they do not install vendor
SDKs or drivers. The NVIDIA path needs CMake 3.28 or newer. GPU tests require
access to the corresponding devices and will fail when hardware is missing or
its identity differs from the configured target. Compilation itself needs no GPU.

## Clone the published branch and fetch pinned sources

Use a new directory, without recursive submodule initialization:

```sh
git clone --depth 1 --branch users/powderluv/multi-vendor-rocm --single-branch \
  https://github.com/powderluv/TheRock.git TheRock-multi-vendor
cd TheRock-multi-vendor

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt \
  -r build_tools/multi_vendor_runtime/requirements.txt
.venv/bin/python build_tools/fetch_sources.py \
  --stage multi-vendor-hip --no-remote --depth 1 --jobs 8

git rev-parse HEAD
git -C rocm-systems rev-parse HEAD
```

The parent branch's `.gitmodules` points `rocm-systems` at the
[powderluv fork](https://github.com/powderluv/rocm-systems/tree/users/powderluv/multi-vendor-rocm).
The child must resolve to **`a5dd53d3be5ced72d3e1f521c2067d341aadece3`**. It includes
typed multi-vendor kpack payloads and module-aware compatible-pack lookup. Record
the printed parent revision with your results; the branch can advance.

Only `rocm-systems` is needed for this source stage. Install Python dependencies
before running the fetch script, which imports them even for help. A missing
DVC configuration for the unselected `rocm-libraries` checkout may produce a
warning; that source tree is not needed for this profile.

Use `--no-remote` to retain the committed gitlink. `--remote` follows `develop`
and loses the required child changes. In an older checkout, synchronize the fork
URL with `git submodule sync -- rocm-systems` before fetching. The fetch script
can reset submodule work; use a fresh checkout for reproduction and preserve
local edits before invoking it on an existing development tree.

## Build and validate Radeon plus NVIDIA

All remaining commands assume the checkout root as the working directory unless
stated otherwise. Quote the semicolon-separated target list as shown:

```sh
cmake -S . -B build/multi-vendor-inputs-hip -GNinja \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -DTHEROCK_BUILD_PROFILE=multi-vendor-hip \
  '-DTHEROCK_MULTI_VENDOR_TARGETS=amd:hip:gfx1201;nvidia:cuda:sm_120' \
  -DTHEROCK_ENABLE_MULTI_VENDOR_VALIDATION=ON \
  -DTHEROCK_ENABLE_MULTI_VENDOR_MODULES=ON \
  -DTHEROCK_MULTI_VENDOR_AMD_ROOT=/opt/rocm \
  -DTHEROCK_MULTI_VENDOR_CUDA_ROOT=/usr/local/cuda
cmake --build build/multi-vendor-inputs-hip --target therock-dist --parallel 8
ctest --test-dir build/multi-vendor-inputs-hip --output-on-failure
```

The recorded combined profile passed **30 CTests**, covering the HIP wrapper
baseline, native loading, and installed consumers. The default `rocm` profile
remains separate. To build for one installed SDK, select only its target and omit
the unused SDK root. Other hardware needs its own supported target and validation;
SM120 results do not qualify SM90 or another GPU.

First configuration captures content locks for selected SDKs and tools and can
hash tens of gigabytes. Builds and tests verify those inputs. If an SDK changes,
use a new build directory or follow the explicit
[reviewed lock-update workflow](multi-vendor-inputs.md#review-an-sdk-update-and-select-a-new-lock),
then rebuild. Reusing stale stages or deleting receipts is not an SDK update.

## Run the installed client

The distribution includes native workers, catalogs, typed packs, a selected Python
runtime, dependency declarations, and an example. It still depends on Python,
`msgpack`, `zstandard`, vendor user-space runtimes, and installed drivers.

```sh
module_dist="$PWD/build/multi-vendor-inputs-hip/dist/multi-vendor-modules"
.venv/bin/python -I "$module_dist/share/therock/examples/packed_session_client.py" \
  --dist-root "$module_dist" \
  --target amd:hip:gfx1201 --format hsaco \
  --peer-target nvidia:cuda:sm_120 --peer-format mixed
```

The paired case keeps both workers open, composes SAXPY/ReLU operations, transfers
intermediate data through host memory in both directions, rejects foreign handles,
and checks that NVIDIA still runs after closing AMD. Each case checks 18
size/scalar combinations. `mixed` exercises NVIDIA cubin and PTX modules together.
For a single GPU, omit the two `--peer-*` arguments; choose AMD `hsaco`, or NVIDIA
`cubin`, `ptx`, or `mixed` with its corresponding exact target.

To list discovered devices using the repository dispatcher:

```sh
PYTHONPATH="$PWD/rocm-systems/shared/kpack/python" \
  .venv/bin/python build_tools/dispatch_multi_vendor_modules.py \
  --dist-root "$module_dist" list
```

An application can use the installed public API directly:

```sh
PYTHONPATH="$module_dist/share/therock/python" \
  .venv/bin/python -B - "$module_dist" <<'PYTHON'
from array import array
from pathlib import Path
import sys
from therock_multi_vendor import ModuleRequest, open_session

root = Path(sys.argv[1])
saxpy = ModuleRequest("validation/saxpy", "cubin", "therock_module_saxpy")
with open_session(root, "nvidia:cuda:sm_120", (saxpy,)) as session:
    kernel = session.module(saxpy)
    x, y, output = (session.allocate(4) for _ in range(3))
    session.write(x, 0, array("f", [1.0, 2.0, 3.0, 4.0]))
    session.write(y, 0, array("f", [0.5, 0.5, 0.5, 0.5]))
    session.launch(kernel, x, y, output, 1.25, 4)
    result = session.read(output, 0, 4)
    assert result == array("f", [1.75, 3.0, 4.25, 5.5])
    print(list(result))
PYTHON
```

For AMD, change the request format to `hsaco` and target to `amd:hip:gfx1201`.
See the [client guide](multi-vendor-client.md) for device UUID selection, ownership,
error behavior, and limits. Each worker binds one vendor/device; buffers cannot
cross sessions. This increment supports the `therock.validation.f32-vector` ABI,
up to 32 preloaded modules, 64 live buffers, 65556 FP32 elements per buffer, and
65539 elements per launch. General kernel arguments, a shared-library C ABI,
cross-vendor peer memory, and math-library providers remain future work.

To deploy the example outside the checkout, copy the entire
`dist/multi-vendor-modules` directory. In the destination's Python environment,
install `share/therock/python/requirements.txt` from that copied directory and
run its example with `-I`. The SDK/runtime and driver dependencies still apply;
the installed example does not import checkout source paths.

## Add Intel native Level Zero and an additional NVIDIA architecture

This path compiles and packages Intel SPIR-V and SM90 payloads while retaining
the validated Radeon/SM120 native paths. It uses the native Level Zero loader;
chipStar is not required. Keep HIP consumer validation **OFF** for this four-target
configuration. Intel HIP through chipStar is a separate, unvalidated integration
path described in the [HIP runbook](multi-vendor-hip.md).

The tested Level Zero headers/loader and SPIR-V validator were extracted locally
from Ubuntu packages, without changing system drivers:

```sh
mkdir -p build/level-zero-deps/downloads build/level-zero-deps/sdk
(
  cd build/level-zero-deps/downloads
  apt-get download libze-dev=1.16.1-1build1 libze1=1.16.1-1build1 \
    'spirv-tools=2025.1~rc1-1~ubuntu0.24.04.2'
)
for level_zero_deb in build/level-zero-deps/downloads/*.deb; do
  dpkg-deb -x "$level_zero_deb" build/level-zero-deps/sdk
done
```

Those exact packages must remain available in the configured apt repositories.
If unavailable, obtain them from an appropriate Ubuntu archive or explicitly
qualify replacement versions and record new input locks. The recorded SHA256s are:

```text
c4afc8a13f4d7d115121f39f271b765a5bbb85ee8332766afa8687f317fc776d  libze-dev_1.16.1-1build1_amd64.deb
1b6f94fa36fe35ff960bcbcc1ef61bbd6f617855bf61d961c8b9e398cb92a2e3  libze1_1.16.1-1build1_amd64.deb
529e396c337d15beb8ed0f03fcbeb2a066e5a818e032e197b6ef2d3328dc7ae9  spirv-tools_2025.1~rc1-1~ubuntu0.24.04.2_amd64.deb
```

The [Level Zero release](https://github.com/oneapi-src/level-zero/releases/tag/v1.16.1)
provides loader/header context; an Intel compute driver is a separate requirement
for eventual GPU execution. Configure using the actual tested SPIR-V tool paths:

```sh
cmake -S . -B build/multi-vendor-contracts -GNinja \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -DTHEROCK_BUILD_PROFILE=multi-vendor-hip \
  '-DTHEROCK_MULTI_VENDOR_TARGETS=amd:hip:gfx1201;nvidia:cuda:sm_120;nvidia:cuda:sm_90;intel:level-zero:xe2-b70' \
  -DTHEROCK_ENABLE_MULTI_VENDOR_VALIDATION=OFF \
  -DTHEROCK_ENABLE_MULTI_VENDOR_MODULES=ON \
  -DTHEROCK_MULTI_VENDOR_AMD_ROOT=/opt/rocm \
  -DTHEROCK_MULTI_VENDOR_CUDA_ROOT=/usr/local/cuda \
  -DTHEROCK_MULTI_VENDOR_LEVEL_ZERO_ROOT="$PWD/build/level-zero-deps/sdk/usr" \
  -DTHEROCK_MULTI_VENDOR_SPIRV_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DTHEROCK_MULTI_VENDOR_SPIRV_TRANSLATOR=/opt/rocm/lib/llvm/bin/amd-llvm-spirv \
  -DTHEROCK_MULTI_VENDOR_SPIRV_VALIDATOR="$PWD/build/level-zero-deps/sdk/usr/bin/spirv-val"
cmake --build build/multi-vendor-contracts --target therock-dist --parallel 8
ctest --test-dir build/multi-vendor-contracts -E 'intel|sm90' --output-on-failure
```

The recorded native profile passed **26 CTests** with that hardware filter. The
build validates both Intel SPIR-V modules offline; SM90 payload architecture checks
also passed. Neither result establishes execution on those absent cards.

After a B70 and its supported compute driver are installed, run the Intel tests
and installed consumer explicitly:

```sh
ctest --test-dir build/multi-vendor-contracts -L intel --output-on-failure
intel_dist="$PWD/build/multi-vendor-contracts/dist/multi-vendor-modules"
.venv/bin/python -I "$intel_dist/share/therock/examples/packed_session_client.py" \
  --dist-root "$intel_dist" --target intel:level-zero:xe2-b70 --format spirv
```

These commands are prospective validation steps. Missing hardware or an incorrect
identity fails explicitly, with no CPU fallback. Record the driver, PCI ID, UUID,
JIT diagnostics, arithmetic, ordering, and cleanup results before qualifying B70.

## Run the focused regression suite

After building the four-target native tree above, install the recorded pytest
version separately. Do not combine `requirements.txt` and
`requirements-test.txt` in this environment: their boto3 constraints conflict.

```sh
.venv/bin/python -m pip install pytest==9.0.3
THEROCK_NATIVE_MODULE_BUILD_ROOT="$PWD/build/multi-vendor-contracts/experimental/multi-vendor/modules" \
  .venv/bin/python -m pytest -q \
    build_tools/tests/adapter_capabilities_test.py \
    build_tools/tests/native_event_pipeline_test.py \
    build_tools/tests/native_level_zero_session_test.py \
    build_tools/tests/module_service_test.py \
    build_tools/tests/module_service_protocol_test.py \
    build_tools/tests/device_inventory_test.py \
    build_tools/tests/native_device_inventory_test.py \
    build_tools/tests/runner_registry_test.py \
    build_tools/tests/dispatch_multi_vendor_modules_test.py \
    build_tools/tests/module_contract_test.py \
    build_tools/tests/native_module_contract_test.py \
    build_tools/tests/multi_vendor_inputs_test.py \
    build_tools/tests/multi_vendor_input_guards_test.py \
    build_tools/tests/gpu_targets_test.py \
    build_tools/tests/configure_multi_vendor_test.py \
    build_tools/tests/payload_catalog_test.py \
    build_tools/tests/multi_vendor_pack_test.py \
    build_tools/tests/assemble_multi_vendor_modules_test.py \
    build_tools/tests/validate_multi_vendor_modules_test.py \
    build_tools/tests/artifact_bundle_name_test.py \
    build_tools/tests/therock_subproject_prebuilt_test.py \
    build_tools/tests/packed_module_session_test.py \
    build_tools/tests/runner_query_test.py \
    build_tools/tests/packed_session_client_test.py \
    build_tools/tests/stage_multi_vendor_runtime_test.py
```

The installed-client checkpoint recorded **345 focused tests passed, no skips**,
plus **26 native** and **30 combined HIP/native CTests**. Those profiles overlap;
they are not an additive unique-test total. The installed paired consumer also
passed from a relocated distribution and unrelated working directory. Earlier
independent child runs recorded 110 C++ kpack tests and 38 Python kpack tests.

An earlier broader Python run had 137 passes and one pre-existing artifact
manifest-drift failure; this work does not claim an entirely green full upstream
suite. Detailed machine-local evidence is under
`build/multi-vendor/validation-results/` on Shark-a. Those build outputs and logs
are not included in a clone. Published source, exact git revisions, input locks,
and newly captured test logs are the reproducibility inputs for another machine.

## Further reading

- [Architecture and alternatives](multi-vendor-architecture.md)
- [HIP baseline and native module build details](multi-vendor-hip.md)
- [Input provenance and SDK updates](multi-vendor-inputs.md)
- [Module launch contracts](multi-vendor-module-contracts.md)
- [Device discovery and dispatch](multi-vendor-dispatch.md)
- [Events](multi-vendor-events.md), [sessions](multi-vendor-sessions.md), and [device pipelines](multi-vendor-pipelines.md)
- [Persistent native service](multi-vendor-service.md) and [installed Python client](multi-vendor-client.md)
