# Selective multi-vendor distributions

One multi-target build can now produce separate installed distributions for exact
target subsets. The exporter copies the selected native runners and shared Python
client, repacks only the selected device payloads, and writes a file inventory for
independent verification. It neither rebuilds device code nor queries GPU drivers.
This is the selective-delivery step following the
[installed client](multi-vendor-client.md).

## Export from one build

Complete `therock-dist` using the [getting-started guide](multi-vendor-getting-started.md).
The native four-target build supplies AMD gfx1201, NVIDIA SM120 and SM90, and Intel
B70 SPIR-V. Run from the checkout root, creating the output parent first:

```sh
source_dist="$PWD/build/multi-vendor-contracts/dist/multi-vendor-modules"
mkdir -p build/target-exports
PYTHONPATH="$PWD/rocm-systems/shared/kpack/python" \
  .venv/bin/python -B build_tools/export_multi_vendor_distribution.py export \
  --source-dist "$source_dist" --output-dir build/target-exports/radeon \
  --target amd:hip:gfx1201
PYTHONPATH="$PWD/rocm-systems/shared/kpack/python" \
  .venv/bin/python -B build_tools/export_multi_vendor_distribution.py export \
  --source-dist "$source_dist" --output-dir build/target-exports/nvidia \
  --target nvidia:cuda:sm_120
PYTHONPATH="$PWD/rocm-systems/shared/kpack/python" \
  .venv/bin/python -B build_tools/export_multi_vendor_distribution.py export \
  --source-dist "$source_dist" --output-dir build/target-exports/paired \
  --target amd:hip:gfx1201 --target nvidia:cuda:sm_120
```

Each destination must be new, with an existing parent. Use a different destination
for a new export; this command does not update, merge, or uninstall an existing
installation. Source and destination trees cannot overlap. Repeated or unknown
target IDs fail. Target selection is exact: selecting SM120 does not also select
SM90. Each selected target retains every registered logical module and its
advertised formats. Per-module and per-format pruning are outside this increment.

For a NVIDIA multi-architecture distribution, select both
`--target nvidia:cuda:sm_120 --target nvidia:cuda:sm_90`. For an Intel-only package,
select `--target intel:level-zero:xe2-b70`. These exports need the corresponding
payloads in the source distribution, but no matching hardware. SM90 and Intel
execution remain unqualified until their GPU tests pass.

The export tool runs from TheRock with its Python dependencies and fetched kpack
writer. The input can be a copy of the original built distribution; its producer checkout,
SDK directories, and devices need not be available to the exporter. Deriving another
export from an already-exported tree is rejected; select the new subset from the
original built distribution instead. Running the
result still requires Python dependencies and the selected vendor's compatible
user-space runtime and GPU driver. Export does not bundle those dependencies or
establish a complete runtime dependency closure.

## Verify and run a relocated export

The manifest is `share/therock/distribution-manifest.json`. Verification checks
its self-digest and exact file inventory, including file bytes, sizes, and modes.
Added, removed, altered, or symlinked files fail. Treat Python bytecode caches as
additional files: use the installed example's `-I` mode, or `python -B` for your
own application if you want the exported inventory to remain unchanged.

```sh
PYTHONPATH="$PWD/rocm-systems/shared/kpack/python" \
  .venv/bin/python -B build_tools/export_multi_vendor_distribution.py verify \
  --dist-root build/target-exports/paired

paired_dist="$PWD/build/target-exports/paired"
.venv/bin/python -I "$paired_dist/share/therock/examples/packed_session_client.py" \
  --dist-root "$paired_dist" --target amd:hip:gfx1201 --format hsaco \
  --peer-target nvidia:cuda:sm_120 --peer-format mixed
```

The whole exported directory can be copied or renamed. Its registry and catalogs
use relative paths. Install `share/therock/python/requirements.txt` in the receiving
Python environment, retain required file modes, and use the installed example or
[public client API](multi-vendor-client.md). AMD uses HSACO; NVIDIA can explicitly
select cubin, PTX, or the example's mixed mode. No absent vendor runner or payload
is needed for the selected application. Selecting an excluded target fails rather
than falling back to another architecture or vendor.

The inventory verifier is a delivery check, not a signature. Applications still
perform the client's runner, payload, contract, and device checks. Keep the
publisher and installed Python code trusted and serialize distribution updates
with export, verification, and execution.

## Selection and provenance

Export validates selected runner bytes against their registry hashes and checks
the corresponding compiled description. It reads and verifies selected payloads
from schema-2 catalogs before repacking them. Missing formats, duplicate module
selections, invalid contracts, and corrupted selected bytes fail. It validates the
installed Python runtime against the source-independent runtime manifest, without
importing that runtime or consulting its original checkout sources.

Selected `input-build-receipt.json` files must match the source pack's
`input-provenance.json`. Those files, runner descriptions, runtime files, and the
runtime manifest are preserved byte-for-byte. Original build provenance can still
name unselected vendors' SDKs: it records the build that produced these artifacts,
not the runtime requirements of the smaller distribution.

A separate schema-1 `selective-multi-vendor-distribution` manifest records the
exact selected targets, source registry/catalog/archive identities, original
provenance and runtime-manifest hashes, and exported file identities. It has
`cache_eligible: false`. Exporting is a derivation of existing outputs and cannot
turn local build receipts into authenticated attestations or complete provenance.

Only verified snapshots enter a temporary sibling directory. Publication requires
Linux `renameat2` with `RENAME_NOREPLACE` and filesystem support; an existing output
is preserved even if it appears after the initial check. Unsupported platforms or
filesystems fail instead of falling back to overwrite behavior. See the
[Linux rename documentation](https://man7.org/linux/man-pages/man2/rename.2.html).
This does not provide crash-durable publication or a filesystem-wide snapshot of
concurrent source updates. Preserve a stable source distribution while exporting.

## Validation

The `selective-export` CTest label exports a single target or AMD/NVIDIA pair,
relocates the directory outside the checkout, verifies it, runs its installed
client from an unrelated working directory, and verifies it again. The client
retains its existing 18 size/scalar cases, ownership checks, and paired-session
survival check.

```sh
ctest --test-dir build/multi-vendor-contracts -L selective-export \
  -E 'intel|sm90' --output-on-failure
ctest --test-dir build/multi-vendor-inputs-hip -L selective-export --output-on-failure
.venv/bin/python -m pytest -q build_tools/tests/export_multi_vendor_distribution_test.py
```

On Shark-a, **17 exporter CPU tests** passed as part of **362 focused regression
tests**, with no skips. The native profile passed **29 CTests** after the hardware
filter; the combined HIP/native profile passed **33 CTests**. Each profile includes
three relocated export consumers: AMD, NVIDIA with mixed cubin/PTX modules, and
an AMD/NVIDIA pair. These suites overlap and are not an additive unique-test total.

Five exports from a copied four-target source passed inventory verification:

| Export                    | Exact targets                | Payload variants |
| ------------------------- | ---------------------------- | ---------------- |
| Radeon                    | `amd:hip:gfx1201`            | 2                |
| NVIDIA                    | `nvidia:cuda:sm_120`         | 4                |
| Paired                    | AMD gfx1201 and NVIDIA SM120 | 6                |
| NVIDIA multi-architecture | NVIDIA SM120 and SM90        | 8                |
| Intel                     | `intel:level-zero:xe2-b70`   | 2                |

The CPU suite includes malformed or inconsistent provenance, corrupt payloads and
runtime files, incomplete selections, extra or inaccessible files, and a competing
publisher creating the destination. Original installed distributions retained
the same file bytes, modes, and modification times after validation and export.

Intel and SM90 CTests are registered for future matching hardware. CPU packaging
checks and offline SPIR-V validation do not qualify their execution. Machine-local
results for this increment belong under
`build/multi-vendor/validation-results/selective-export/` on Shark-a.

## Alternatives Considered

- **One complete distribution for every machine.** It preserves all choices but
  ships unused native runners and payloads. Exact-target exports make the
  selected delivery profile explicit while sharing one producer build.
- **Rebuild separately for each delivery profile.** This remains supported and
  can avoid installing unneeded build SDKs. Export instead derives several
  profiles from already-built bytes and retains their original provenance.
- **Filter only the catalogs.** Unselected payload bytes would remain in the
  archives. Repacking removes them and gives the selected files their own hashes.
- **Rewrite receipts for the selected targets.** That would misrepresent how
  existing outputs were built. A separate derivation manifest records selection
  while preserving the original build observations.

Incremental installation ownership, signed release manifests, production package
formats, provider dependency closure, and general runtime/library ABI work remain
separate steps in the [architecture roadmap](multi-vendor-architecture.md).
