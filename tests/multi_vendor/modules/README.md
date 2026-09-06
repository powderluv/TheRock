# Native module validation

This standalone project builds two independent device modules and loads them
through native vendor APIs. Its ordinary C++ host executables use the CUDA
Driver API on NVIDIA, the HIP runtime module API on AMD, and Level Zero on Intel.
The NVIDIA host does not include HIP headers or link a HIP adapter or the CUDA
runtime library. Intel uses `level_zero_loader.cpp` and the Level Zero loader.

AMD and NVIDIA compile `kernels.cpp`. Intel compiles the equivalent OpenCL C
source `kernels.cl` to SPIR-V. The Intel path tests native module loading and the
five-argument kernel ABI; HIP source portability through chipStar is a separate
consumer and is not a dependency of this project.

## Build inputs

CMake 3.25 or newer is required.

| Variable                          | Meaning                                                                                                                                             |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `THEROCK_MODULE_BACKEND`          | `amd`, `nvidia`, or `intel`.                                                                                                                        |
| `THEROCK_MODULE_TARGET`           | AMD/NVIDIA compiler architecture, such as `gfx1201` or `120`; Intel product identity, such as `xe2-b70`.                                            |
| `THEROCK_MODULE_SDK_ROOT`         | ROCm SDK, CUDA toolkit, or Level Zero SDK containing headers and the loader library.                                                                |
| `THEROCK_MODULE_INSTALL_SUBDIR`   | Target slug, such as `nvidia-cuda-sm120` or `intel-level-zero-xe2-b70`.                                                                             |
| `THEROCK_MODULE_COMPILER`         | Device compiler executable: AMD clang, NVIDIA nvcc, or a clang supporting the SPIR64 target for Intel. AMD/NVIDIA can discover it within their SDK. |
| `THEROCK_MODULE_SPIRV_TRANSLATOR` | Intel: `amd-llvm-spirv` or `llvm-spirv` matching the compiler's LLVM bitcode version.                                                               |
| `THEROCK_MODULE_SPIRV_VALIDATOR`  | Intel: `spirv-val` from SPIRV-Tools.                                                                                                                |
| `THEROCK_MODULE_INTEL_DEVICE_ID`  | Intel hardware check; defaults to `0xe223` for `xe2-b70`. Other Intel target labels require an explicit device ID.                                  |
| `THEROCK_MODULE_DEVICE_INDEX`     | Optional CTest device index, default `0`.                                                                                                           |

For NVIDIA:

```sh
cmake -S tests/multi_vendor/modules -B build/modules-nvidia -G Ninja \
  -DTHEROCK_MODULE_BACKEND=nvidia \
  -DTHEROCK_MODULE_TARGET=120 \
  -DTHEROCK_MODULE_SDK_ROOT=/usr/local/cuda \
  -DTHEROCK_MODULE_INSTALL_SUBDIR=nvidia-cuda-sm120
cmake --build build/modules-nvidia
ctest --test-dir build/modules-nvidia --output-on-failure
```

The AMD equivalent uses `amd`, `gfx1201`, `/opt/rocm`, and
`amd-hip-gfx1201`. AMD clang uses
`--offload-device-only --no-gpu-bundle-output` to emit a raw AMDGPU ELF code
object. NVIDIA nvcc uses `--cubin` and `--ptx` for native and JIT payloads.

For Intel, the following example uses Shark-a's existing ROCm compiler and an
isolated Level Zero SDK and SPIR-V validator under
`/home/nod/github/TheRock/build/level-zero-deps/sdk/usr`. Run it from the repository
root, adjusting the paths for another machine:

```sh
cmake -S tests/multi_vendor/modules -B build/modules-intel -G Ninja \
  -DTHEROCK_MODULE_BACKEND=intel \
  -DTHEROCK_MODULE_TARGET=xe2-b70 \
  -DTHEROCK_MODULE_SDK_ROOT="$PWD/build/level-zero-deps/sdk/usr" \
  -DTHEROCK_MODULE_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DTHEROCK_MODULE_SPIRV_TRANSLATOR=/opt/rocm/lib/llvm/bin/amd-llvm-spirv \
  -DTHEROCK_MODULE_SPIRV_VALIDATOR="$PWD/build/level-zero-deps/sdk/usr/bin/spirv-val" \
  -DTHEROCK_MODULE_INTEL_DEVICE_ID=0xe223 \
  -DTHEROCK_MODULE_INSTALL_SUBDIR=intel-level-zero-xe2-b70
cmake --build build/modules-intel
```

The Intel device compilation steps are:

1. Compile OpenCL C 1.2 with clang for `spir64`, emitting LLVM IR bitcode.
1. Translate that bitcode with a matching `amd-llvm-spirv` or `llvm-spirv`, using
   `--spirv-max-version=1.0`.
1. Validate the resulting binary with `spirv-val --target-env opencl1.2`.

The product label `xe2-b70` never becomes a compiler architecture flag. The
compiler and translator must agree on their LLVM bitcode format; an unrelated
system translator may reject the compiler output. Building and installing the
loader and payloads requires no Intel GPU and does not install system drivers.
The isolated SDK supplies development headers and a userspace loader; hardware
execution still requires an Intel GPU and its compatible runtime/driver.

Intel hardware validation is deferred until the B70 is available. SPIR-V
validation, CPU process-boundary tests, and a successful host build do not qualify
the GPU. Once the card and driver are installed, run:

```sh
ctest --test-dir build/modules-intel --output-on-failure
```

## Payloads and execution

| Backend | Payload files                                        |
| ------- | ---------------------------------------------------- |
| AMD     | `saxpy.hsaco`, `relu.hsaco`                          |
| NVIDIA  | `saxpy.cubin`, `saxpy.ptx`, `relu.cubin`, `relu.ptx` |
| Intel   | `saxpy.spirv`, `relu.spirv`                          |

Payloads install under `share/therock/modules/<slug>/`. Each backend's loader
installs as `bin/<slug>/therock_module_validation`. For example:

```sh
therock_module_validation --payload /path/to/relu.cubin \
  --symbol therock_module_relu --device 0 --expect-arch sm_120
```

For the Intel Level Zero loader:

```sh
therock_module_validation --payload /path/to/relu.spirv \
  --symbol therock_module_relu --device 0 \
  --expect-arch spirv --expect-device-id 0xe223
```

`--expect-arch spirv` identifies the payload contract; it does not identify an
Intel GPU architecture. The Intel loader enumerates only root GPU devices with
vendor ID `0x8086`, sorts them by driver and device UUID, and checks the selected
device's ID against `0xe223` for B70. It prints device name and ID for diagnosis.
The pack-validation Python wrapper supplies this B70 ID automatically, rejects
conflicting overrides, and requires an explicit ID for other Intel target labels.
The low-level loader also accepts an optional case-insensitive
`--expect-device-name` substring; the B70 qualification path uses the exact ID.

The default symbol is `therock_module_saxpy`. Only that symbol and
`therock_module_relu` have CPU references; other names are rejected. Each payload
exports just its corresponding kernel, allowing separate archives with the same
device target to own distinct logical modules. Both kernels have the parameter
ABI `(const float *x, const float *y, float *output, float alpha, unsigned count)`.
SAXPY computes `alpha*x+y`; the activation variant computes
`max(0, alpha*x+y-0.125)`.

Each hardware run checks sizes 1, 127, 128, 129, 4099, and 65539 over three rounds,
including 17 output guard elements. AMD/NVIDIA copies and launches use one
explicit nonblocking stream. Intel uses Level Zero host/device allocations and
an asynchronous command queue with command-list barriers between input copies,
the kernel launch, and readback. Queue completion precedes CPU result checks and
allocation cleanup. Every allocation, copy, module operation, launch,
synchronization, and cleanup operation is checked. The sources do not assume a
warp or wave width.

The AMD/NVIDIA loader identifies raw ELF payloads by their machine type and
recognizes NVIDIA PTX. The Intel loader checks SPIR-V structure and asks
`zeModuleCreate` to build it using `ZE_MODULE_FORMAT_IL_SPIRV`; driver build logs
are reported. It checks the device's supported SPIR-V version, launch geometry,
memory, and kernel argument count before execution. Native APIs validate the
complete payload and entry point. Missing files, wrong formats, missing symbols,
wrong selected devices, and arithmetic errors fail without substituting another
payload or backend.

Runtime NVIDIA architecture checks use compute-capability major/minor, since
CUDA does not report compiler target feature suffixes such as `a` through those
attributes. These suffixes remain in the device compilation target.

Installed AMD and Intel loaders retain runpaths to their imported SDK libraries.
Vendor drivers and SDK runtimes remain external dependencies. Rebuild in a clean
directory when an SDK or compiler installation changes in place.
