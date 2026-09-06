# Native Intel Level Zero module runner

`level_zero_loader.cpp` implements the native Intel host runner using core Level
Zero APIs available since API 1.0. It does not use HIP or chipStar host APIs.
The ordinary `therock_module_validation` command-line contract is preserved:

```sh
therock_module_validation --payload /path/to/saxpy.spirv \
  --symbol therock_module_saxpy --device 0 \
  --expect-arch spirv --expect-device-id 0xe223
```

`--symbol` accepts `therock_module_saxpy` (the default) or `therock_module_relu`.
Each has the same five-argument ABI and CPU reference used by the AMD/NVIDIA
module runners. `--expect-device-name` optionally adds a case-insensitive name
substring check. Product qualification should use the PCI device ID because
some Intel driver versions return generic device names. PCI IDs must be nonzero
16-bit values; decimal is the default, and a `0x` prefix selects hexadecimal.

`--expect-arch spirv` checks only the payload format. It cannot establish Xe2 or
B70 hardware identity; other architecture values are rejected. The B70 profile
passes `--expect-device-id 0xe223`, and the loader always requires PCI vendor ID
`0x8086` and GPU device type. It enumerates root devices through `zeDeviceGet`,
excludes subdevice handles, and orders the resulting list by driver UUID followed
by device UUID. Duplicate identities fail rather than yielding an ambiguous
index. Driver selection and affinity settings can change the visible device
list, so each run prints the actual selected name and PCI identifiers.

Before calling `zeInit`, the loader reads the complete payload and checks size,
SPIR-V magic/version/header fields, and every instruction's word-count boundary.
The bounded validation fixture accepts at most 64 MiB. These checks do not
replace semantic SPIR-V validation: the build uses SPIR-V Tools, and
`zeModuleCreate` validates the module against the actual driver/device. Module
build logs are collected on success and failure and their handles are released.

Execution uses an explicitly asynchronous compute/copy queue. It allocates native
host and device buffers in the selected context, appends host-to-device copies,
a barrier, the kernel launch, a barrier, and device-to-host readback. Each round
waits for completion before inspecting output or resetting the command list.
The fixture checks six boundary sizes over three rounds, with 17 output guard
values and a small documented FP32 tolerance. It checks device capabilities,
local-memory availability, the kernel's five-argument ABI, and launch geometry.

A submission owner waits before any referenced list, kernel, allocation, module,
or context can be destroyed during normal execution or exception unwinding.
Queue synchronization has a finite 30-second timeout. Core Level Zero offers no
command cancellation: if submission completion is unknown after synchronization
fails, the runner prints and flushes a failure diagnostic and terminates the
process without freeing resources still potentially in use. This avoids both
an unbounded wait and unsafe cleanup. All ordinary setup and completed-work
paths use checked RAII cleanup, including partially completed setup.

An absent Intel GPU or unavailable Level Zero driver is an explicit failure,
with no skip result or CPU fallback. The Intel loader can be compiled and its
malformed-input/missing-driver failures verified without a card. GPU arithmetic,
module compilation by the Intel driver, execution ordering, and cleanup still
require validation on the B70 hardware.
