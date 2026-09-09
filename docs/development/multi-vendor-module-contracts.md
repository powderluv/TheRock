# Multi-vendor module launch contracts

The native validation path checks a declared launch contract before executing a
packed kernel. Catalog metadata, archive metadata, and the compiled adapter must
agree. This is the first capability/ABI increment: a logical contract for the
SAXPY and ReLU fixtures, not a general runtime plugin or shared binary argument
block.

The [contract model](../../build_tools/_therock_utils/module_contract.py) supplies
both pack metadata and generated C++ runner descriptions. The
[generator](../../build_tools/configure_module_contract.py) writes
`module_contract_data.h` and a configure-time description for each native child
build. A completed `runner-contract.json` is published only after its executable
and payloads have been built successfully.
They describe compiled adapter support; they do not describe an observed GPU or
establish hardware qualification.

## Independent version boundaries

| Boundary                    | Current version                            | Meaning                                                       |
| --------------------------- | ------------------------------------------ | ------------------------------------------------------------- |
| KPAK container              | 1                                          | Existing typed archive and compression representation.        |
| Catalog schema              | 2                                          | Every entry contains a launch contract and its SHA256.        |
| Launch ABI                  | `therock.validation.f32-vector`, version 1 | The logical fixture arguments, ownership, and ordering rules. |
| Runner description protocol | 1                                          | Strict JSON response to `--describe-contract`.                |

Schema-1 catalogs remain readable and extractable. They do not carry a launch
contract, so the guarded validation wrapper refuses to execute them. Rebuild the
profile to regenerate schema-2 packs; do not add contract fields to an old
catalog by hand. The contract and digest are also stored in each archive TOC
entry and must match the catalog, including when extracting a different entry
from that selected archive. A schema-1 catalog cannot silently hide contract
metadata in its archive.

The generic pack CLI retains schema 1 by default. Its explicit
`create --validation-contract` option declares the known fixture ABI and emits
schema 2. This option is appropriate only for payloads built to that ABI. It
does not inspect arbitrary machine code to prove an argument layout.

## Logical fixture ABI

The ordered arguments are:

| Argument | Meaning                                    | Native argument bytes / alignment |
| -------- | ------------------------------------------ | --------------------------------- |
| `x`      | Read-only device pointer to FP32 elements  | 8 / 8                             |
| `y`      | Read-only device pointer to FP32 elements  | 8 / 8                             |
| `output` | Write-only device pointer to FP32 elements | 8 / 8                             |
| `alpha`  | FP32 value                                 | 4 / 4                             |
| `count`  | Unsigned 32-bit element count              | 4 / 4                             |

The launch uses a `[128, 1, 1]` block and no dynamic shared memory. Each adapter
binds these arguments through its native API. CUDA device addresses, HIP
pointers, and Level Zero pointer arguments are not interchangeable host handles.
The metadata does not define offsets in a common packed argument buffer.

The runner owns its context, queue, and device allocations. Copies, launch, and
readback execute in order; completion is synchronized before host validation and
resource release. Modules remain alive through queue completion. The required
adapter operations are `device-allocation`, `module-load`, `kernel-launch`,
`ordered-copy`, and `queue-synchronize`. The required kernel contract does not include public events, graph capture,
cross-vendor pointers, peer copies, or a general HIP ABI. Compiled adapters now
separately advertise the internal `cross-queue-events` fixture capability; see
the [event validation guide](multi-vendor-events.md).

The native code checks pointer/scalar representation at compilation. At runtime,
AMD and NVIDIA check device and kernel launch limits and available memory. Intel
checks device geometry, allocation limits, SPIR-V support, and kernel argument
count/group requirements through Level Zero. These checks and successful driver
module loading are separate from the compiled capability description.

## Build, package, and execute

Use the normal [multi-vendor runbook](multi-vendor-hip.md). Python is now also a
native-child configure dependency for generating the description. The profile
passes its locked Python executable to each child.

The generated description participates in device-output dependencies. A separate
completed description participates in the build receipt; configure alone cannot
replace it. Direct install compares the completed and configured descriptions
before changing the stage. It installs beside the runner under
`bin/<target-key>/runner-contract.json`. Before packaging, the assembler checks
every selected stage's description against the current contract, vendor,
backend, formats, and entry points. A direct pack build therefore cannot attach
a new contract to a stage built for an older one. No GPU or runner process is
needed during assembly.

The execution wrapper currently runs on POSIX hosts and limits the offline
description query to 10 seconds and 64 KiB of combined output. Other hosts fail
explicitly before starting a runner.

The execution wrapper:

1. Selects the exact module, target, format, and entry point; verifies catalog and
   pack integrity; returns metadata and payload bytes from the same snapshot.
1. Requires the supported launch contract and queries the chosen executable with
   `--describe-contract`.
1. Checks the strict description protocol, vendor/backend, formats, symbols,
   capabilities, full contract, and contract hash.
1. Passes the agreed ABI name, version, hash, and format to the native runner.
   The runner repeats those checks before reading the payload or initializing a
   GPU, then checks the inspected payload format and observed device limits.

There is no fallback to another backend, architecture, format, or ABI. Unknown
required capabilities, incompatible versions, missing fields, duplicate JSON
fields, and changed argument layouts fail explicitly. SHA256 establishes
integrity relative to the metadata, not publisher authentication or proof that
arbitrary code implements the declared behavior.

Inspect a built runner without initializing a GPU:

```sh
build/multi-vendor/dist/multi-vendor-modules/bin/nvidia-cuda-sm120/therock_module_validation \
  --describe-contract
```

This works for the compiled Intel adapter while B70 hardware is absent. The
response has `scope: "compiled-adapter"` and no hardware-pass claim. Direct
payload invocation additionally requires `--launch-abi`,
`--launch-abi-version`, `--launch-contract-sha256`, and `--payload-format`.
Prefer CTest or the pack-validation wrapper, which supplies these automatically.

## Remaining boundaries

The [registry dispatcher](multi-vendor-dispatch.md) now selects these executables
and checks structured device discovery before launch. It binds the observed
device UUID into the execution process independently of the fixture launch ABI. The current launch protocol
uses separate backend executables. The [persistent service](multi-vendor-service.md)
now adds connection-scoped buffer/module handles and request/reply errors while
retaining this fixed kernel contract. It does not provide same-process
heterogeneous dispatch, public device/context/event handles, ABI-stable plugin
tables, or a production capability registry. Argument metadata
is a source/build declaration; the arithmetic tests qualify the fixture kernels
on the devices actually exercised.

Intel hardware execution and SM90 hardware execution remain unqualified until
appropriate devices are tested. The [imported-input lock](multi-vendor-inputs.md)
continues to cover declared SDK/tool inputs only; these new contracts do not
enable production artifact cache reuse or complete the toolchain dependency
closure.
