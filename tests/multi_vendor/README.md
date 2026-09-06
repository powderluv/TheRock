# Multi-vendor HIP validation

This standalone SDK consumer compiles the same HIP source in a separate CMake
tree for each backend. It validates a bounded common runtime profile, rather than
claiming full ROCm library or HIP API compatibility.

Configure with `THEROCK_MULTI_VENDOR_BACKEND=amd`, `nvidia`, or `intel`, standard
`CMAKE_HIP_COMPILER` / `CMAKE_HIP_ARCHITECTURES`, and the appropriate SDK prefix.
`THEROCK_MULTI_VENDOR_HIP_ROOT` locates installed HIP headers; AMD also requires
`CMAKE_HIP_COMPILER_ROCM_ROOT` when the compiler cannot discover its installed SDK.
NVIDIA requires CMake 3.28 or newer; Intel chipStar requires CMake 4.3 or newer and
`CMAKE_HIP_PLATFORM=spirv`. Use the compiler itself, not a `hipcc` wrapper.

For example, after installing the NVIDIA HIP headers and CUDA:

```sh
cmake -S tests/multi_vendor -B build/validation-nvidia -G Ninja \
  -DTHEROCK_MULTI_VENDOR_BACKEND=nvidia \
  -DCMAKE_HIP_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_HIP_ARCHITECTURES=120 \
  -DTHEROCK_MULTI_VENDOR_HIP_ROOT=/path/to/hip-nvidia \
  -DTHEROCK_MULTI_VENDOR_EXPECT_ARCH=sm_120
cmake --build build/validation-nvidia
ctest --test-dir build/validation-nvidia --output-on-failure
```

The installed executable is `bin/therock_multi_vendor_validation`. Its arguments
are `--case all|core|streams|reduction`, `--device INDEX`,
`--expect-vendor amd|nvidia|intel`, `--expect-arch ISA`, and
`--expect-device-name SUBSTRING`. CTest receives these through the corresponding
`THEROCK_MULTI_VENDOR_DEVICE_INDEX`, `THEROCK_MULTI_VENDOR_EXPECT_VENDOR`,
`THEROCK_MULTI_VENDOR_EXPECT_ARCH`, and
`THEROCK_MULTI_VENDOR_EXPECT_DEVICE_NAME` cache variables. Runtime device names
are matched without case sensitivity; runtime architecture names must match
exactly (`gfx1201`, for example, or `sm_120`).

Intel CTest cases explicitly select `CHIP_BE=level0` and `CHIP_DEVICE_TYPE=gpu`;
manual runs must set these too. chipStar currently does not expose Intel hardware
ISA through HIP device properties. Its reported `spirv` identifies the portable
payload, and does **not** establish that the GPU is Battlemage. Set
`THEROCK_MULTI_VENDOR_EXPECT_DEVICE_NAME=B70` when qualifying that card. Intel
hardware validation is deferred until hardware is available.

Each case is a separate process, runs serially in CTest, and fails on missing
hardware, wrong device, unsupported APIs, arithmetic errors, or cleanup errors.
There are no skip return codes or CPU reference execution paths that count as
GPU success. The CPU computes expected answers only.

| Case        | Checks                                                                                                                                                                                                                  |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `core`      | Allocation, synchronous copies, SAXPY followed by activation, repeated allocation/launch cycles, tail guards at sizes 1, 127, 128, 129, 4099, and 65539.                                                                |
| `streams`   | Pinned host memory, asynchronous copies, two nonblocking streams with event dependencies in both directions, 22 dependent launches per round, event reuse over three rounds, and completed uses before freeing buffers. |
| `reduction` | Shared-memory tree reduction with full-block barriers, padded tail lanes, checks of every partial sum, successive GPU reduction stages, final CPU-reference sum, and output guards.                                     |

The kernels use 128 threads per block and make no wave/subgroup-width assumptions.
Numerical inputs are deterministic binary fractions. Reduction sums are exactly
representable for these bounded inputs; SAXPY checks permit a small FP32 rounding
tolerance for contraction. Logs include runtime device identity, tested sizes,
errors or sums, and a final `PASS` line only after the selected case succeeds.

The experimental parent build profile consumes installed SDKs and compilers. It
therefore disables reusable fingerprints for the validation artifacts until those
external inputs have trustworthy identities. Start a clean build directory when
an SDK or compiler installation changes in place; ordinary source edits trigger
incremental rebuilds through the parent graph.
