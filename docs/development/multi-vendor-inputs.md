# Imported inputs for multi-vendor builds

The experimental `multi-vendor-hip` profile locks the contents of its declared
SDKs and tool executables. A changed input stops the build until the operator
restores the expected input or explicitly selects a reviewed replacement lock.
The lock is independent of GPU qualification: it cannot establish that an Intel
device, driver, or kernel has run successfully.

Use the [build runbook](multi-vendor-hip.md) for target selection and SDK setup.
This guide covers the imported-input layer implemented in
[`input_provenance.cmake`](../../experimental/multi-vendor/input_provenance.cmake)
and [`multi_vendor_inputs.py`](../../build_tools/multi_vendor_inputs.py).

## Default capture and verification

During the first configure, the profile captures an immutable build-local lock
once its declared inputs pass scanning. For a build configured with `-B build/multi-vendor`, its files are:

| File under `build/multi-vendor/experimental/multi-vendor/` | Purpose                                                                                 |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `inputs.lock.json`                                         | Expected per-file metadata and content hashes, with a combined `global_content_sha256`. |
| `input-digest-cache.json`                                  | Optional local stat/digest cache used to avoid reading unchanged file contents again.   |
| `input-provenance.json`                                    | Current declared locations, content identities, and file/directory/link/byte counts.    |
| `verify-inputs.cmake`                                      | Generated verification command used by the build and test graph.                        |

Later configurations verify the existing lock. They do not replace it when an
SDK changes. A failure names the changed logical inputs and their old/new
digests. Changing the selected set of SDKs or tools can also require a new lock.

The graph verifies inputs before participating child configure, build, and stage
commands, before artifact population, and through CTest fixtures. Children
configured by the profile receive the same guard for direct child builds and
tests. Standalone consumers configured without that parent guard do not acquire
this contract automatically.

HIP and native validation children also produce `input-build-receipt.json`
after their declared executable and device-payload outputs complete. The receipt
is a copy of the selected provenance report, installed under
`bin/<target-key>/` after those outputs. Child CTest checks the build receipt;
top-level HIP tests also check the stage receipt. Pack assembly checks staged
receipts, and packaged module tests check the distribution's runner receipts
and pack provenance. Missing or stale receipts fail before those consumers can
reuse binaries or payloads from an earlier input selection. Direct child installs
also verify the current inputs and completed build receipt before changing a
stage. Guarded installs replace their known outputs and publish the receipt last,
including when files have equal sizes and timestamps.

Run an explicit check without executing GPU tests:

```sh
cmake --build build/multi-vendor --target multi-vendor-verify-inputs
cmake --build build/multi-vendor --target multi-vendor-verify-inputs-full
ctest --test-dir build/multi-vendor -L provenance --output-on-failure
```

The two build targets can check inputs before compilation. CTest provenance
fixtures additionally require completed build/stage receipts, so run the normal
build first on a newly configured build directory.

Normal verification enumerates the trees every time, detecting added and removed
files. It reuses a file digest only when its resolved path, device, inode, size,
mode, modification time, and change time match the cache entry. Full verification
reads every file again. Neither verification executes the captured compiler.

The cache has a checksum to detect accidental corruption. An invalid cache is
ignored and rebuilt; its checksum is not authentication against deliberate
tampering. Use full verification when independent content rehashing is required.
The initial capture and full checks can read tens of gigabytes of installed SDK
contents. Unchanged locks, reports, and cache files retain their timestamps.

## What is captured

Every profile configuration declares the selected CMake, build tool, Python, C,
and C++ executables. Enabled consumers add their SDK and backend tools:

| Consumer                     | Additional declared inputs                                                                                                                         |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| AMD HIP or native modules    | ROCm SDK tree and selected AMD compiler file.                                                                                                      |
| NVIDIA HIP or native modules | CUDA SDK tree and selected NVCC file. The HIPNV header SDK is built separately from pinned sources.                                                |
| Intel HIP                    | chipStar SDK tree and selected chipStar compiler file.                                                                                             |
| Intel native modules         | Level Zero SDK tree, SPIR-V frontend, matching translator, and validator files. The ROCm SDK tree is also included when it supplies that frontend. |

Intel native SGEMM additionally locks the entire declared
`THEROCK_MULTI_VENDOR_ONEAPI_ROOT` and the selected SYCL compiler executable.
This includes compiler, oneMKL, and sibling runtime components within that prefix;
the selected oneMKL directory and compiler must resolve inside it. See the
[Intel provider build guide](multi-vendor-intel-sgemm.md). External system and
driver dependencies remain outside this declared-input coverage.

A tree lock includes directories, empty directories, permission modes, file
sizes and SHA256 digests, and symlinks. Directory symlinks are recorded without
recursive traversal. Their resolved targets must lie in a declared tree or
match a declared file input; broken, escaping, cyclic-resolution, and special
file targets fail. A contained ancestor directory link is safe to record
because the walk does not follow it.

File inputs capture the final referent bytes and final-component symlink chain.
Symlink identity uses a logical input name and relative target path. Absolute
root locations and symlink spelling do not determine content identity, so
identical relocated inputs can match the same lock. Diagnostic link text is
retained in the lock; current requested and resolved root paths belong in the
provenance report.

The lock schema is version 1, kind `multi-vendor-input-lock`. The report is kind
`multi-vendor-input-provenance`. Both state `coverage: "declared-inputs"` and
`cache_eligible: false`. Reports describe observed inputs rather than proving an
artifact's full build history. The HIP distribution carries
`share/therock/multi-vendor/hip-input-provenance.json`; the module distribution
carries `share/therock/packs/input-provenance.json`.

## Review an SDK update and select a new lock

Keep the SDK stable during verification and builds. File and directory checks
detect changes observed during scanning; they do not create a filesystem-wide
snapshot or prevent a later update.

A clean build directory captures the inputs currently installed there. When
keeping an existing build, create a new snapshot and explicitly select it.
Preserve the old lock for comparison. Do not delete or edit it to make a failed
verification pass.

First derive an input spec from the last successful configure's report. This
preserves the exact logical names and tool selections expected by that build:

```sh
mkdir -p build/input-locks
.venv/bin/python - \
  build/multi-vendor/experimental/multi-vendor/input-provenance.json \
  build/input-locks/inputs-next.json <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
spec = {
    "schema_version": 1,
    "kind": "multi-vendor-input-spec",
    "inputs": [
        {"name": entry["name"], "kind": entry["kind"], "path": entry["current_path"]}
        for entry in report["inputs"]
    ],
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(spec, indent=2) + "\n")
PY
```

A spec entry has the form
`{"name": "cuda-sdk", "kind": "tree", "path": "/usr/local/cuda"}`.
For an in-place update, keep the paths. For a new SDK prefix or compiler, edit
the corresponding paths and supply matching SDK/compiler CMake options when
reconfiguring. Relative spec paths resolve against the spec's directory.
The CMake profile rejects input paths containing semicolons, which are CMake
list separators.

Capture the new contents after the intended update:

```sh
.venv/bin/python build_tools/multi_vendor_inputs.py snapshot \
  --spec build/input-locks/inputs-next.json \
  --output build/input-locks/inputs-next.lock.json
```

This command refuses an existing output. With no `--cache` argument, it reads
all declared file contents. Review the changed input names, entries, modes, and
digests against the old lock, together with the SDK/tool update that caused
them. A matching hash only identifies bytes; it does not establish their origin
or correctness.

After review, explicitly select that already-existing lock:

```sh
cmake -S . -B build/multi-vendor \
  -DTHEROCK_MULTI_VENDOR_INPUT_LOCK="$PWD/build/input-locks/inputs-next.lock.json"
cmake --build build/multi-vendor
cmake --build build/multi-vendor --target multi-vendor-verify-inputs-full
```

Keep the original profile/target configuration and add updated SDK/compiler
options if needed. A nonempty `THEROCK_MULTI_VENDOR_INPUT_LOCK` selects an
external reviewed lock and uses verification only; a missing file is an error.
Relative option paths resolve against TheRock's source directory. The new
provenance file participates in child and compiler-output dependencies, so
selecting new input contents requires consumer work before repackaging.

Use the normal top-level build to rebuild and stage the affected consumers and
produce matching receipts. A direct pack-child build against old stages, or a
CTest run against old binaries, must fail after the input selection changes.
Building only an executable target can leave its receipt unfinished; complete
the normal child or top-level build before testing or staging. Do not copy or
edit receipts to bypass this check.

Run the hardware tests appropriate to the installed cards after rebuilding.
Intel compilation and offline SPIR-V validation remain separate from B70
execution qualification.

## Direct CLI use

The same operations are available without CMake:

```sh
.venv/bin/python build_tools/multi_vendor_inputs.py capture \
  --spec build/input-locks/inputs-next.json \
  --lock build/input-locks/standalone.lock.json
.venv/bin/python build_tools/multi_vendor_inputs.py verify \
  --spec build/input-locks/inputs-next.json \
  --lock build/input-locks/standalone.lock.json \
  --cache build/input-locks/stat-cache.json \
  --report build/input-locks/observed.json --full
```

`capture` creates a missing lock or verifies an existing one. `verify` requires
an existing lock. `snapshot` always requires a new output path. All accept
repeated `--input NAME tree|file PATH` instead of `--spec`; these relative paths
resolve against the process working directory. Input names must be unique.

Keep lock, cache, and report outputs distinct and outside declared inputs.
Resolved output aliases are checked. Malformed schemas, duplicate JSON fields,
invalid hashes, and ambiguous names fail without replacing the existing lock.
New locks are published atomically without overwriting an existing path.

## Remaining provenance and cache boundary

Locks cover only the declared inputs. Receipts record completion through the
local build graph under a selected observation; they are not output signatures
or independently authenticated build attestations. These checks do not
automatically capture
host dynamic libraries, system headers outside an SDK, compiler subprocesses,
Python modules, configuration discovered by tools, environment variables, driver
state, or a complete runtime/toolchain dependency closure. They also do not
authenticate a publisher, grant redistribution rights, or qualify a GPU.

Affected experimental artifact fingerprints remain disabled. Content locking
does not enable reusable artifact cache entries. Imported `.prebuilt` stage
markers are rejected during configuration and guarded build/test operations:
matching today's SDK is insufficient evidence for how an existing stage was
built. A supported prebuilt import path still needs original build provenance.

Pinned provider recipes and complete dependency, option, ABI, and redistribution
contracts remain follow-up work in the
[architecture roadmap](multi-vendor-architecture.md#roadmap-and-acceptance-gates).
