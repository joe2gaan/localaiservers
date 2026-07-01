# vnext-gfx906-rocm72-gguf-hf-repro - GFX906 ROCm7.2 GGUF/HF Stand-Alone Contract Reproduction Package

Release tag: `vnext-gfx906-rocm72-gguf-hf-repro`

## Release Boundary

This release describes the stand-alone vNext reproduction-package release for
the GFX906 / MI50 ROCm7.2 Dense/MoE work.

vNext is not a patch, addendum, or silent replacement for the published
v0.2.0/v0.2.1 release boundaries. It is a separate release package with its own
tag, release assets, portability gate, reproduction instructions, validation
tables, and claim boundary.

The existing v0.2.1 reproduction package remains a historical public
reproduction package for the v0.2 Dense/MoE release line. vNext does not revise
that tag, retarget that release, or depend on that checkout for its claim. The
vNext claim must be reproducible from the vNext release tag and its own public
inputs.

Post-publication note: this live GitHub Release body may include documentation
clarifications made after the tag was cut. The tag checkout remains the
reproducible executable package; use this live release body, the vNext tag
checkout, and the published release assets for the current public reproduction
flow. These wording updates do not retarget the tag and do not change the
Docker image, model assets, or runtime artifacts.

The goal is to package the next reproduction path: model-format detection,
profile-driven overlay selection, format-specific patch isolation, generated
launch artifacts, and the normal benchmark ladder for both HF and GGUF model
packages. Historical v0.2.0/v0.2.1 values remain useful controls and
comparators, but vNext must stand on its own from public release inputs.

The runtime image remains:

```text
joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b
```

Docker Hub manifest digest:

```text
sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e
```

## Runtime Packaging Model

vNext does not publish a separate Docker image with the GGUF/HF release bits
baked into the image. The runtime dependency is the pinned public ROCm7.2 image
shown above.

The reproducible vNext runtime is assembled from these public inputs:

- the pinned Docker image and digest;
- the tagged vNext repository checkout;
- the vNext profile contracts, overlays, patch bundles, launcher scripts, and
  validation scripts from that checkout;
- the GitHub Release GGUF split assets and hashes for GGUF profiles; and
- the pinned public Hugging Face model revisions for HF profiles.

`vnext_repro_launcher.sh` generates a profile-specific Docker launch artifact
that mounts the selected model inputs and applies the selected overlay/patch
bundle at launch time. The generated wrapper starts a normal `vllm serve`
process inside the pinned runtime image.

In short, users get the vNext GGUF/HF bits by cloning the vNext release tag and
staging the public release assets, not by pulling a second vNext-specific Docker
image.

GitHub Releases remain canonical for published release claim boundaries. Docker
Hub remains an evergreen artifact distribution channel and should not be
treated as the latest benchmark announcement.

## Reference Hardware Used For vNext Validation

The vNext validation evidence was recorded on the following local validation
lanes. Host labels are sanitized evidence labels only; they are not public
access endpoints and are not required reproduction targets.

| Field | `.20` validation lane | `.30` validation lane |
| --- | --- | --- |
| System vendor/model | GIGABYTE `G292-Z20-00` | GIGABYTE `G292-Z20-00` |
| System firmware | `R23`, firmware date `2021-09-06` | `R23`, firmware date `2021-09-06` |
| CPU | 1x AMD EPYC 7F32 8-Core Processor | 1x AMD EPYC 7F32 8-Core Processor |
| CPU topology | 8 cores / 16 threads, SMT on, 1 socket | 8 cores / 16 threads, SMT on, 1 socket |
| CPU clocks reported | min `2500 MHz`, max `3700 MHz`, boost enabled | min `2500 MHz`, max `3700 MHz`, boost enabled |
| L3 cache | `128 MiB` | `128 MiB` |
| System memory | `125 GiB` visible | `125 GiB` visible |
| OS | Ubuntu `24.04.2 LTS` | Ubuntu `24.04.2 LTS` |
| Kernel | `6.8.0-52-generic` | `6.8.0-52-generic` |
| ROCm-SMI driver version | `6.8.5` | `6.8.5` |
| Root disk | `447.1G` Crucial `CT480BX500SSD1` SATA SSD | `447.1G` Crucial `CT480BX500SSD1` SATA SSD |
| Local model/runtime NVMe | `1.7T` KIOXIA `KCD6XLUL1T92`; validation-local mount path omitted | `1.7T` KIOXIA `KCD6XLUL1T92`; validation-local mount path omitted |
| GPU count | 8x AMD GFX906 / Vega 20 | 8x AMD GFX906 / Vega 20 |
| GPU PCI device | `1002:66a1`, rev `02` | `1002:66a1`, rev `02` |
| GPU SKU/subsystem | SKU `D1631700`, subsystem `0x0834` | SKU `D1631700`, subsystem `0x0834` |
| GPU VBIOS | `113-D1631700-111` on all 8 GPUs | `113-D1631700-111` on all 8 GPUs |
| GPU VRAM visible | `34342961152` bytes per GPU, all 8 GPUs | `34342961152` bytes per GPU, all 8 GPUs |
| GPU BAR0 visible | `34359738368` bytes per GPU, all 8 GPUs | `34359738368` bytes per GPU, all 8 GPUs |
| GPU BAR2 visible | `2097152` bytes per GPU, all 8 GPUs | `2097152` bytes per GPU, all 8 GPUs |
| GPU PCI bus IDs | `06:00.0`, `09:00.0`, `45:00.0`, `48:00.0`, `89:00.0`, `8c:00.0`, `c5:00.0`, `c8:00.0` | `06:00.0`, `09:00.0`, `45:00.0`, `48:00.0`, `89:00.0`, `8c:00.0`, `c5:00.0`, `c8:00.0` |
| NUMA reporting | GPU NUMA node reports `-1`; local CPU list `0-15` | GPU NUMA node reports `-1`; local CPU list `0-15` |
| BMC/display adapter | ASPEED VGA controller present | ASPEED VGA controller present |
| Fabric/network observed | Mellanox InfiniBand present; additional Mellanox Ethernet present | Mellanox InfiniBand present; additional Mellanox Ethernet present |

Notes:

- The release profiles require full-BAR/P2P-on platform state. The live
  validation query confirmed full 32 GiB BAR0 visibility on all 8 GPUs on both
  validation lanes.
- ROCm-SMI product-name strings may label some devices inconsistently, but the
  memory-total query and sysfs VRAM totals showed `34342961152` bytes visible
  per GPU on all 8 GPUs.
- The InfiniBand/Ethernet devices are validation-site infrastructure and are
  not public reproduction requirements.
- Users should choose their own local SSD/NVMe-backed `LOCAL_MODEL_ROOT`,
  `LOCAL_HF_CACHE`, and `LOCAL_RUNTIME_ROOT` values for reproduction.
- Do not publish per-card unique IDs, GUIDs, MAC addresses, hostnames, private
  addresses, validation-local mount paths, or management endpoints in release
  notes.

## Public Reproducibility Requirement

This release must be reproducible on a user's own qualifying GFX906 host. It
must not depend on LocalAIServers internal hosts, internal mount paths, or
unpublished model files.

Portability is a release gate, not a best-effort note. A clean-room run must
start from only these public inputs:

- a fresh clone of the tagged repository checkout;
- the pinned public Docker image and digest listed in this release;
- public upstream Hugging Face model repositories for HF profiles;
- GitHub Release assets, manifests, and hashes for GGUF profiles;
- local SSD/NVMe storage selected by the user; and
- the user's own GFX906 host platform that satisfies the documented
  full-BAR/P2P-on prerequisites.

The public instructions must not require copying files from `.20`, `.30`, a
validation-host NVMe workspace, an InfiniBand address, a private mount, a
retained model cache, or any LocalAIServers validation-host path. Those
locations may appear only in validation evidence and experiment logs, never as
required reproduction inputs.

Generated launch artifacts may record local preflight paths in
`effective_config.env` because each user generates those artifacts on their own
checkout. The reusable `docker_run.sh` wrapper still takes user-selected
`HOST_MODEL_ROOT`, `HOST_HF_CACHE`, and `HOST_RUNTIME_ROOT` values at run time.
If an HF model directory below `HOST_MODEL_ROOT` is a symlink to another
NVMe-backed directory, the generated wrapper resolves that symlink and mounts
the target at the same `/opt/local-models/...` container path. If that target
is a Hugging Face cache snapshot with symlinked safetensors shards, the wrapper
uses a runtime `model-bind-root` shim under `HOST_RUNTIME_ROOT`, mounts the
resolved snapshot under `/opt/vnext-models/...`, and mounts the matching
snapshot `blobs` directory read-only at the container path required by those
relative shard links. Broken or non-directory model symlinks fail closed before
benchmark launch, and unresolved snapshot shard links also fail closed before
benchmark launch.
Container-internal paths such as
`/usr/share/ollama/kernel_labs/...` are populated inside the container from the
mounted release patch bundle; they are not host validation-workspace
dependencies and do not require copying files from a LocalAIServers machine.

The sanitized host labels in the validation tables, such as `.20` and `.30`,
identify the validation lanes used for the recorded evidence. They are not
deployment targets, are not public access endpoints, and are not required for
reproduction.

The HF profiles are portable when the user stages the public upstream
`Qwen/Qwen3.6-27B` and `Qwen/Qwen3.6-35B-A3B` model packages into a local
SSD/NVMe-backed model directory.

The HF validation path used these pinned upstream revisions:

| Public Hugging Face repo | Pinned revision |
| --- | --- |
| `Qwen/Qwen3.6-27B` | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| `Qwen/Qwen3.6-35B-A3B` | `995ad96eacd98c81ed38be0c5b274b04031597b0` |

Use these revisions for HF staging and GGUF tokenizer resolution. Do not rely
on floating `main` for a reproduction claim.

The GGUF profiles are portable only when the exact validated F16 GGUF inputs
are available through one of these public paths:

- release-hosted split GGUF assets with SHA256 manifests; or
- a locked public conversion recipe that names the upstream model revision,
  converter source revision, conversion command, output filename, and SHA256.

This package publishes the validated GGUF files as split GitHub Release assets.
Each part stays below the GitHub Release per-asset limit, and each model
includes a part manifest plus final assembled-file hash.

| Required artifact | Public source for final release | Final SHA256 |
| --- | --- | --- |
| `qwen36-27b-f16-gguf/Qwen3.6-27B-F16.gguf` | GitHub Release assets listed by `Qwen3.6-27B-F16.gguf.parts.sha256` | `b2347376b9bb7d12cf5f1d31c53ac4c60dd3d4b95068a09351187b328b5e9d89` |
| `qwen36-35b-a3b-f16-gguf/Qwen3.6-35B-A3B-F16.gguf` | GitHub Release assets listed by `Qwen3.6-35B-A3B-F16.gguf.parts.sha256` | `1f2443bb0ff958943d091410c61120c181a0579b3bc85192029aa51d821d141c` |
| `qwen36-27b-text-config-eosfix/` | GitHub Release asset `Qwen3.6-27B-text-config-eosfix.tar.gz` | `9a85f0a18a012ed37f0eb4c42549569234af880c5f6e3ce908e0e45cca835719` |
| Dense tokenizer | Public Hugging Face repo `Qwen/Qwen3.6-27B` pinned with `TOKENIZER_REVISION=6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | n/a |
| MoE tokenizer | Public Hugging Face repo `Qwen/Qwen3.6-35B-A3B` pinned with `TOKENIZER_REVISION=995ad96eacd98c81ed38be0c5b274b04031597b0` | n/a |

The split release assets, part manifests, dense text-config asset, and public
pinned tokenizer paths are the public GGUF reproduction inputs for this package.
If any of these assets are missing from the GitHub Release, the public asset URL
gate must fail before benchmark launch.

A local reconstruction attempt from the pinned public `Qwen/Qwen3.6-27B`
`config.json` did not reproduce the locked Dense text-config hash under the
obvious text-only `Qwen3_5ForCausalLM` variants or under the pinned runtime
image's `Qwen3_5TextConfig` / `Qwen3_5Config` serialization paths. Do not
substitute a regenerated config for the locked archive unless it is
re-benchmarked through the normal warmup -> strict -> fixed-token ladder and
then documented as the vNext stand-alone release input.

Earlier `file://` staged audits, clean-room simulations, and fixture hash
substitutions are useful maintainer checks for the split-asset mechanics, but
they do not satisfy this public reproduction requirement. Public reproduction
requires the GitHub Release assets for `vnext-gfx906-rocm72-gguf-hf-repro` and a
replay without local asset overrides.

Public GGUF repositories that provide BF16 or quantized variants are useful for
source comparison, but they are not interchangeable with the F16 GGUF artifacts
listed above for this benchmark claim.

Published asset inventory and maintainer upload controls:

```sh
split_gguf_for_release() {
  model_file=$1

  rm -f "$model_file".part-* "$model_file.parts.sha256" "$model_file.sha256"
  split -b 1900M -d -a 4 "$model_file" "$model_file.part-"
  sha256sum "$model_file".part-* > "$model_file.parts.sha256"
  sha256sum "$model_file" > "$model_file.sha256"
}

split_gguf_for_release Qwen3.6-27B-F16.gguf
split_gguf_for_release Qwen3.6-35B-A3B-F16.gguf
```

The published release attaches every `*.part-*`, `*.parts.sha256`, and
`*.sha256` file to the GitHub Release. The part manifests list only basename
files matching `<model>.part-*`; they must not include paths, `..`, whitespace
in part filenames, or stale split outputs. The release also attaches
`Qwen3.6-27B-text-config-eosfix.tar.gz`. Reassemble from the published assets
in a clean directory and confirm the final SHA256 values above before treating
a run as a public reproduction attempt.

Published hosted-asset inventory:

- Dense GGUF part files: `28`
- MoE GGUF part files: `36`
- part manifests: `2`
- final assembled-file hash sidecars: `2`
- Dense text-config archive: `1`
- total uploadable files: `69`

A maintainer-side simulated asset bundle with that inventory has been verified
locally by part manifest checks and by streaming the split parts in manifest
order to reproduce the final GGUF hashes above. That was pre-publication
evidence for the asset layout. The live release now exposes the same expected
public asset inventory, and public reproduction should verify the GitHub
Release URLs before replay.

For maintainers preparing equivalent release asset bundles, verify the complete
upload directory:

```sh
cd qwen36-gfx906
./verify_vnext_release_asset_bundle.sh /path/to/vnext-release-assets
./list_vnext_release_asset_uploads.sh /path/to/vnext-release-assets
./upload_vnext_release_assets.sh <release-tag> /path/to/vnext-release-assets
cd ..
```

This verifier checks the expected `69` files, both part manifests, every split
part hash, both final hash sidecars, the Dense text-config archive hash, and
the final assembled Dense/MoE GGUF hashes by streaming parts in manifest order
without creating another full GGUF copy. The upload-list helper then prints the
same `69` files in a stable order for maintainer review before any GitHub
Release upload is attempted.

`upload_vnext_release_assets.sh` is dry-run by default. In dry-run mode, it
reruns the bundle verifier, confirms the `69`-file upload list, and prints the
exact upload plan without changing GitHub. Public users may use the helper to
upload the same asset shape to their own repository by setting
`GH_RELEASE_REPO=owner/repo` and `UPLOAD_VNEXT_RELEASE_ASSETS=1` after creating
their own tag and GitHub Release. Uploading assets to
`joe2gaan/localaiservers` is maintainer-only and additionally requires
`ALLOW_LOCALAISERVERS_RELEASE_UPLOAD=1`. The helper does not create tags,
create releases, edit release notes, delete assets, or use `--clobber`.

Verify that the public GitHub Release URLs exist before running the expensive
full download/replay. For a user-owned
repository, set `RELEASE_ASSET_BASE` to that repository's release download URL:

```sh
cd qwen36-gfx906
RELEASE_ASSET_BASE="https://github.com/owner/repo/releases/download/<release-tag>" \
  ./verify_vnext_release_asset_urls.sh <release-tag>
cd ..
```

For this LocalAIServers-maintained release, use the default
`https://github.com/joe2gaan/localaiservers/releases/download/<release-tag>`
base. This URL verifier checks the exact expected asset names for the tag
without downloading the full multi-GB GGUF parts. The full reproduction claim
still requires `verify_vnext_release_readiness.sh` with
`RUN_SERVING_BENCHMARKS=1` against those public asset URLs.

## Final Public Replay Proof Checklist

vNext public replay is complete only when the stand-alone release can be
reproduced from public inputs without relying on v0.2.0, v0.2.1,
validation-host storage, or maintainer-only `file://` asset mirrors.

For a full public replay, confirm:

- the final stand-alone vNext tag exists;
- a clean checkout of that tag contains this release note, the vNext profile
  directory, and the vNext verification/launcher scripts listed in
  `qwen36-gfx906/profiles/vnext/README.md`;
- the GitHub Release for that tag has all `69` expected GGUF release assets;
- `./verify_vnext_release_asset_urls.sh <release-tag>` passes without
  `RELEASE_ASSET_BASE`;
- no final replay uses `ALLOW_LOCAL_ASSET_PREFLIGHT=1`;
- the replay starts from a fresh checkout of the vNext tag or pinned release
  commit;
- `RUN_SERVING_BENCHMARKS=1` is set;
- `STAGE_HF_PUBLIC_INPUTS=1` is used unless the pinned HF model packages are
  already staged and verified from public inputs;
- the readiness command prints `vNext full release reproduction path
  completed`;
- all five profiles write summary tables under
  `LOCAL_RUNTIME_ROOT/vnext-launch-runs/...`;
- every strict row is `qwen_gate_valid=true`;
- Dense 27B `c1_10000` clears the 65 TPS ai-info gate;
- benchmark TPS lands in the observed validation band for the selected lane;
  exact TPS equality with any single evidence row below is not required because
  strict output length and run-to-run scheduling can vary;
- generated launch artifacts and benchmark summaries are archived or copied
  only after sanitizing local paths and host details; and
- no v0.2.0 or v0.2.1 release claim is updated by the vNext replay.

If any checklist item fails, treat the run as incomplete reproduction evidence,
not full public replay success.

Minimum tree-content check from a fresh checkout:

```sh
test -f qwen36-gfx906/profiles/vnext/README.md
test -f qwen36-gfx906/verify_vnext_release_readiness.sh
test -f qwen36-gfx906/vnext_repro_launcher.sh
test -f qwen36-gfx906/run_vnext_profile_benchmark.sh
test -d qwen36-gfx906/profiles/vnext
test -d qwen36-gfx906/overlays
```

## What This Adds

This vNext package turns the GGUF/HF work into a reproducible contract path:

- binary model-format probing before launch;
- model-family and model-format profile checks;
- isolated HF and GGUF overlays;
- profile-scoped patch target groups;
- pinned patch-bundle manifest hashes;
- generated vLLM argv, Docker wrapper, benchmark environment, and effective
  config;
- runtime vLLM argument-schema validation against the image's own
  `vllm serve --help=all`;
- the normal benchmark ladder:
  `8` warmups -> `c1_128` uncapped strict -> `c1_2000` -> `c1_10000`.

The launcher is intentionally fail-closed. It rejects HF/GGUF format mismatch,
Dense/MoE family mismatch, GGUF environment leakage into HF profiles, patch
target-group mismatch, patch-bundle hash mismatch, TP mismatch, dtype mismatch,
`MAX_MODEL_LEN` mismatch, and P2P-off launches for profiles that require the
full-BAR/P2P-on lane.

This removes the hand-assembled production `EXTRA_VLLM_ARGS` path from the
release workflow. A developer can inspect the profile, generate the launch
artifacts, validate them, run the serviceability checks, and then run the same
benchmark ladder used for promotion.

## Validated Profile Results

The following values are clean contract-run evidence from the vNext launcher
path. Public reproduction of these values requires the GitHub Release assets,
public HF inputs, qualifying full-BAR/P2P-on host state, and the documented
readiness flow with `RUN_SERVING_BENCHMARKS=1`.

| Format | Profile | Host | Strict backend TPS | `c1_2000` backend TPS | `c1_10000` backend TPS | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| GGUF F16 | Dense 27B TP8 full-BAR/P2P-on | `.20` | `69.914` | `70.759` | `66.316` | strict-valid; ai-info 10K gate cleared |
| GGUF F16 | Dense 27B TP8 full-BAR/P2P-on | `.30` | `69.851` | `70.959` | `66.437` | strict-valid; ai-info 10K gate cleared |
| GGUF F16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | `.20` | `119.333` | `120.565` | `113.366` | strict-valid |
| GGUF F16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | `.30` | `119.521` | `120.460` | `113.258` | strict-valid |
| HF FP16 | Dense 27B TP8 full-BAR/P2P-on | `.20` | `70.168` | `71.316` | `66.824` | fresh-cache release-band control |
| HF FP16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | `.20` | `115.105` | `115.933` | `109.095` | strict-valid |
| HF FP16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | `.30` | `114.409` | `115.685` | `108.918` | strict-valid |
| HF FP16 | Qwen3.6 35B-A3B MoE TP8 full-BAR/P2P-on | `.20` | `115.041` | `115.549` | `108.805` | strict-valid |
| HF FP16 | Qwen3.6 35B-A3B MoE TP8 full-BAR/P2P-on | `.30` | `114.698` | `115.532` | `108.673` | strict-valid |

### Public-Staged POSIX-Runner Reproduction Audit

The public-style reproduction path was rerun from maintainer-staged split GGUF
assets, the dense text-config archive, generated launch artifacts, and the
normal benchmark ladder. The MoE GGUF profile used the public
`Qwen/Qwen3.6-35B-A3B` tokenizer repo pinned to
`995ad96eacd98c81ed38be0c5b274b04031597b0`, not a validation-host cache path.
The final repeat used the vNext POSIX-shell benchmark runner directly rather
than delegating to the older Bash-only v0.2 helper.

This audit exercised the intended release shape before the final assets were
published. It remains historical pre-publication evidence; public reproduction
should use the GitHub Release asset URLs.

| Format | Profile | Host | Strict backend TPS | `c1_2000` backend TPS | `c1_10000` backend TPS | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| GGUF F16 | Dense 27B TP8 full-BAR/P2P-on | `.30` | `69.986` | `70.935` | `66.403` | strict-valid; ai-info 10K gate cleared |
| GGUF F16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | `.30` | `120.822` | `121.733` | `114.442` | strict-valid |

The GGUF Dense TP8 `c1_10000` results beat the published Dense v0.2.0 value
of `66.069` backend TPS on both `.20` and `.30`. The GGUF MoE TP4 results are
also strict-valid on both lanes under the native contract path.

### Clean-Room GGUF Package-Path Reproduction Audit

A clean-room checkout on `.30` was created outside the working repository and
run against a user-selected local model root. The GGUF files and Dense
text-config archive were staged through the split-release-asset layout and
verified by SHA256 before launch. This run used `RELEASE_ASSET_BASE=file://...`
only as a maintainer-side simulation of the GitHub Release asset layout; the
published release must use real GitHub Release asset URLs for the same files.

The normal public launch path required only `LOCAL_MODEL_ROOT` to map the
profile-default `/opt/local-models/...` paths back to the host staging
directory. The generated Docker wrappers, runtime argument-schema validation,
P2P-on profile defaults, bundled begin-think benchmark proxy, and normal
benchmark ladder were then run from the clean checkout.

| Format | Profile | Host | Strict backend TPS | `c1_2000` backend TPS | `c1_10000` backend TPS | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| GGUF F16 | Dense 27B TP8 full-BAR/P2P-on | `.30` | `69.580` | `70.837` | `66.309` | strict-valid; ai-info 10K gate cleared |
| GGUF F16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | `.30` | `117.731` | `119.649` | `112.568` | strict-valid |

This is positive maintainer evidence that the profile-driven GGUF flow can
reproduce the expected Dense and MoE bands without depending on a
LocalAIServers validation-host model path or retained cache. At the time, it
was not a public reproduction claim because it used simulated release assets.
The live release now provides the split GGUF assets, part manifests, final
SHA256 files, and Dense text-config archive through GitHub Release URLs.

### Stand-Alone GGUF Release-Package Audit

A later stand-alone vNext audit on `.30` copied the draft release checkout to
NVMe-backed storage, staged GGUF inputs through the split-release-asset layout,
verified part manifests and final file SHA256 values, generated fresh launch
artifacts, validated the runtime image/digest and vLLM argument schema, then
ran the normal benchmark ladder:
`8` warmups -> `c1_128` uncapped strict -> `c1_2000` -> `c1_10000`.

This audit used `RELEASE_ASSET_BASE=file://...` as a maintainer-side simulation
before the GitHub Release assets were published. That is not a public
dependency and is not acceptable for public reproduction. Public replay should
use the published release-asset URLs.

| Format | Profile | Host | Strict backend TPS | `c1_2000` backend TPS | `c1_10000` backend TPS | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| GGUF F16 | Dense 27B TP8 full-BAR/P2P-on | `.30` | `70.070` | `71.129` | `66.530` | strict-valid; ai-info 10K gate cleared |
| GGUF F16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | `.30` | `119.197` | `120.618` | `113.377` | strict-valid |

Startup note: the Dense GGUF path can exceed a short five-minute health wait
during first-run graph compilation. The release benchmark runner waits for the
backend before starting warmups; users should not classify the first
shared-memory wait warning during compile as failure if logs and GPU activity
show progress.

### Public-Staged HF Reproduction Audit

The HF profile path was also rerun from public Hugging Face downloads staged
into a clean local model root, generated vNext launch artifacts, runtime
argument-schema validation, and the normal benchmark ladder. This run did not
use retained validation-host Hugging Face snapshots.

| Format | Profile | Host | Strict backend TPS | `c1_2000` backend TPS | `c1_10000` backend TPS | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| HF FP16 | Dense 27B TP8 full-BAR/P2P-on | `.30` | `69.682` | `71.346` | `66.729` | strict-valid; ai-info 10K gate cleared |
| HF FP16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | `.30` | `111.964` | `115.937` | `108.848` | strict-valid |
| HF FP16 | Qwen3.6 35B-A3B MoE TP8 full-BAR/P2P-on | `.30` | `115.488` | `116.258` | `109.242` | strict-valid |

This confirms that the HF reproduction path can start from public upstream
model repositories and the generated profile artifacts. The exact measured
strict token count may vary by strict prompt output length, so backend TPS and
gate validity are the release metrics.

### Full vNext Release-Readiness Serving Replay

A fresh `.20` readiness replay followed the vNext release-notes flow with
`RUN_SERVING_BENCHMARKS=1` after the HF staging-mode clarification. It
completed serviceability checks, runtime image/path validation, contract matrix
checks, public GGUF asset staging, public HF input validation, launch-artifact
generation, runtime vLLM argument-schema validation, host preflight, and the
serving benchmark ladder for all five profiles.

This replay used a maintainer-only local mirror before the release assets were
published. It validates the package path as historical pre-publication
evidence. Public replay should use the final vNext tag and public GitHub
Release asset URLs.

| Format | Profile | Host | Strict backend TPS | `c1_2000` backend TPS | `c1_10000` backend TPS | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| GGUF F16 | Dense 27B TP8 full-BAR/P2P-on | `.20` | `69.908` | `70.881` | `66.446` | strict-valid; ai-info 10K gate cleared |
| GGUF F16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | `.20` | `118.280` | `120.576` | `113.347` | strict-valid |
| HF FP16 | Dense 27B TP8 full-BAR/P2P-on | `.20` | `70.153` | `71.395` | `66.876` | strict-valid; ai-info 10K gate cleared |
| HF FP16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | `.20` | `113.968` | `116.585` | `108.544` | strict-valid |
| HF FP16 | Qwen3.6 35B-A3B MoE TP8 full-BAR/P2P-on | `.20` | `115.673` | `116.542` | `109.672` | strict-valid |

A follow-up `.30` full release-readiness replay used the same release-note
readiness flow with `RUN_SERVING_BENCHMARKS=1`, verified local simulated
release assets, and already-staged public-input model packages from a fresh
temporary copy of the current vNext package. It provides second-lane
pre-publication evidence. Public replay should use the final stand-alone vNext
tag and public GitHub Release asset URLs.

| Format | Profile | Host | Strict backend TPS | `c1_2000` backend TPS | `c1_10000` backend TPS | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| GGUF F16 | Dense 27B TP8 full-BAR/P2P-on | `.30` | `69.873` | `71.161` | `66.638` | strict-valid; ai-info 10K gate cleared |
| GGUF F16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | `.30` | `119.363` | `119.808` | `112.755` | strict-valid |
| HF FP16 | Dense 27B TP8 full-BAR/P2P-on | `.30` | `70.474` | `71.326` | `66.810` | strict-valid; ai-info 10K gate cleared |
| HF FP16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | `.30` | `115.592` | `116.062` | `109.240` | strict-valid |
| HF FP16 | Qwen3.6 35B-A3B MoE TP8 full-BAR/P2P-on | `.30` | `114.652` | `115.895` | `109.001` | strict-valid |

A second `.30` clean-checkout serving replay from the local proof commit
followed the same release-note flow, again with local simulated release assets
before this stand-alone release was published. It completed all five generated
`vllm serve` profiles and produced the following repeated evidence:

| Format | Profile | Host | Strict backend TPS | `c1_2000` backend TPS | `c1_10000` backend TPS | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| GGUF F16 | Dense 27B TP8 full-BAR/P2P-on | `.30` | `70.097` | `71.081` | `66.543` | strict-valid; ai-info 10K gate cleared |
| GGUF F16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | `.30` | `118.948` | `120.961` | `113.684` | strict-valid |
| HF FP16 | Dense 27B TP8 full-BAR/P2P-on | `.30` | `70.306` | `71.254` | `66.796` | strict-valid; ai-info 10K gate cleared |
| HF FP16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | `.30` | `114.557` | `113.436` | `108.044` | strict-valid |
| HF FP16 | Qwen3.6 35B-A3B MoE TP8 full-BAR/P2P-on | `.30` | `115.509` | `116.704` | `109.754` | strict-valid |

The readiness wrapper ended with:

```text
vNext full release reproduction path completed
```

These values are stand-alone vNext validation evidence. They do not revise
v0.2.0 or v0.2.1.

All vNext profiles preserve:

- `MAX_MODEL_LEN=131072`;
- `dtype=half`;
- full-BAR/P2P-on profile requirements;
- the same ROCm7.2 runtime image and digest shown above.

## Profile And Bundle Locks

The current vNext profile set includes:

- `gguf-dense27b-tp8`
- `gguf-moe35b-tp4`
- `hf-dense27b-tp8`
- `hf-moe35b-tp4`
- `hf-moe35b-tp8`

Current patch-bundle manifest locks:

| Bundle | Manifest SHA256 |
| --- | --- |
| HF release bundle | `d9e1168f67731ab66ee53671f89dfb31fd3c74c8f61e2f0d0b7c4f0704c298be` |
| GGUF Dense minimal bundle | `ce2c6f2d974de34c7abf4c25a67985b0eccc452a97f1a887ca97a8de1687f8d2` |
| GGUF MoE minimal bundle | `794cb8760003cfa7dafeb0f06ee01823e5e8eebb3f9d398d8bc8bd7697143f0b` |

Patch target groups are profile-scoped. For example, HF MoE TP4 uses
`common moe hf`, while GGUF MoE uses `common moe gguf`. This keeps a broad
bundle hash-pinned without accidentally activating GGUF-only loader targets
inside HF launches or dense-only targets inside MoE launches.

## Reproduction Flow

Clone the release checkout:

```sh
git clone --depth 1 --branch vnext-gfx906-rocm72-gguf-hf-repro https://github.com/joe2gaan/localaiservers.git
cd localaiservers
```

Host tool prerequisites:

- POSIX-compatible `/bin/sh`
- `git`
- `docker`
- `make`
- a C compiler for `tools/model-format-probe/`
- `python3`
- `curl`
- `sha256sum`
- `tar`
- `split`
- `cat`
- standard POSIX/core userland utilities used by the scripts, including
  `grep`, `sed`, `awk`, `find`, `sort`, `wc`, `mktemp`, `mkdir`, `rm`, `mv`,
  `cp`, `ln`, `chmod`, `touch`, `date`, `sleep`, `kill`, `wait`, `dd`,
  `dirname`, `basename`, and `tail`
- `hf` from `huggingface_hub` when using HF profiles

`hf` is the Hugging Face Hub CLI. Install it with the official Hugging Face CLI
installer or from the public `huggingface_hub` Python package in a virtual
environment, then verify that the `download` command is available:

```sh
curl -LsSf https://hf.co/cli/install.sh | bash
command -v hf
hf download --help >/dev/null
```

The GGUF split-release-asset path does not require the Hugging Face CLI for the
model weights. It downloads the published GGUF parts with `curl` and verifies
them with `sha256sum`. GGUF tokenizer resolution still uses public Hugging
Face repositories pinned by each profile's `TOKENIZER_REVISION`.

This flow must succeed from a clean checkout plus public assets. It must not
require access to a LocalAIServers validation host, InfiniBand address,
private mount, pre-existing Hugging Face snapshot directory, or unpublished
converted GGUF file.

Maintainer-only local mirrors may be used to preflight the asset layout before
uploading release files, but the published reproduction path must use the
GitHub Release asset URLs produced from the final tag. Any `RELEASE_ASSET_BASE`
override is a test hook, not a public dependency. A `file://`
`RELEASE_ASSET_BASE` requires `ALLOW_LOCAL_ASSET_PREFLIGHT=1` and must never be
used as a published reproduction input.

Choose local SSD/NVMe-backed storage. Replace `/mnt/nvme` with the large local
storage path on your own host:

```sh
export LOCAL_MODEL_ROOT=/mnt/nvme/local-models
export LOCAL_HF_CACHE=/mnt/nvme/hf-cache
export LOCAL_RUNTIME_ROOT=/mnt/nvme/vnext-runtime

mkdir -p "$LOCAL_MODEL_ROOT" "$LOCAL_HF_CACHE" "$LOCAL_RUNTIME_ROOT"
```

Do not use a small root partition for model weights, Hugging Face cache, or
runtime compile/cache directories.

Recommended full release-readiness command:

```sh
cd qwen36-gfx906
RELEASE_TAG=vnext-gfx906-rocm72-gguf-hf-repro
STAGE_HF_PUBLIC_INPUTS=1 \
RUN_SERVING_BENCHMARKS=1 \
./verify_vnext_release_readiness.sh \
  --release-tag "$RELEASE_TAG" \
  --model-root "$LOCAL_MODEL_ROOT" \
  --hf-cache "$LOCAL_HF_CACHE" \
  --runtime-root "$LOCAL_RUNTIME_ROOT"
cd ..
```

Use `STAGE_HF_PUBLIC_INPUTS=1` when `$LOCAL_MODEL_ROOT/qwen36-27b-hf` and
`$LOCAL_MODEL_ROOT/qwen36-35b-a3b-hf` are absent or are writable regular
directories where `hf download` can materialize the pinned public model
packages. If those directories are already staged and verified, especially if
they are symlinks to a local Hugging Face snapshot or another NVMe-backed
directory, omit `STAGE_HF_PUBLIC_INPUTS=1` or set `STAGE_HF_PUBLIC_INPUTS=0`
for reruns. The verifier will still check tokenizer files, the safetensors
index, and every referenced shard before launch.

That command follows the release-note sequence: serviceability, contract
matrix, public GGUF asset staging, public HF staging, launch-artifact
generation, runtime vLLM argument-schema validation, host platform preflight,
and the serving benchmark ladder. It exits nonzero before benchmark launch if
the public assets are missing, if the host does not satisfy the full-BAR/P2P-on
preflight, or if any generated artifact fails validation. If
`RUN_SERVING_BENCHMARKS=1` is not set, the script intentionally refuses to
report full release reproduction success; `ALLOW_PRE_SERVING_ONLY=1` is only a
maintainer dry-run mode for non-serving gates.
Each readiness invocation writes generated launch artifacts under a unique
timestamped directory below `LOCAL_RUNTIME_ROOT/vnext-launch-runs/` so repeated
runs do not need to remove container-owned files from a previous attempt.

Stage the public HF model packages only when using HF profiles:

```sh
HF_DENSE_REVISION=6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
HF_MOE_REVISION=995ad96eacd98c81ed38be0c5b274b04031597b0

HF_HOME="$LOCAL_HF_CACHE" hf download Qwen/Qwen3.6-27B \
  --revision "$HF_DENSE_REVISION" \
  --local-dir "$LOCAL_MODEL_ROOT/qwen36-27b-hf"

HF_HOME="$LOCAL_HF_CACHE" hf download Qwen/Qwen3.6-35B-A3B \
  --revision "$HF_MOE_REVISION" \
  --local-dir "$LOCAL_MODEL_ROOT/qwen36-35b-a3b-hf"
```

Then verify that the staged public HF directories satisfy all HF profile
contracts:

```sh
cd qwen36-gfx906
./verify_vnext_public_hf_inputs.sh "$LOCAL_MODEL_ROOT"
cd ..
```

The HF verifier rejects unsafe model roots such as `/` or `/home`, non-pinned
HF revisions, invalid `STAGE_HF_PUBLIC_INPUTS` values, and, when staging
downloads, unsafe HF cache roots such as `/` or `/home`. It also verifies that
the staged HF packages include tokenizer files, a safetensors index, and every
non-empty shard named by `model.safetensors.index.json` rather than accepting a
config-only directory or unresolved Git LFS pointer files. Symlinked HF model
directories are accepted only when they resolve to real local directories that
satisfy the same shard checks; the generated Docker wrapper then mounts that
resolved directory at the expected container path.

If you want the gate to perform the pinned downloads and then validate the
profiles in one step, use:

```sh
cd qwen36-gfx906
HF_HOME="$LOCAL_HF_CACHE" STAGE_HF_PUBLIC_INPUTS=1 \
  ./verify_vnext_public_hf_inputs.sh "$LOCAL_MODEL_ROOT"
cd ..
```

If `hf` is not installed, install the public Hugging Face Hub client according
to the Hugging Face documentation before running HF profiles, or skip this
block and reproduce only the GGUF profiles from release assets. Recent
`huggingface_hub` releases deprecate the older `huggingface-cli` command; use
`hf download` for the portable reproduction flow.

Stage GGUF model packages only from the public source named in the release
artifact table. Do not substitute a BF16 or quantized GGUF for these F16
benchmark claims unless a separate result is published for that file.

For the default split-release-asset path:

```sh
RELEASE_TAG=vnext-gfx906-rocm72-gguf-hf-repro
cd qwen36-gfx906
./stage_vnext_gguf_assets.sh "$RELEASE_TAG" "$LOCAL_MODEL_ROOT"
cd ..
```

The staging script downloads the dense text-config asset, each model's part
manifest, and every listed GGUF part. It verifies the text-config archive, the
part hashes, and the final assembled F16 GGUF SHA256 values from the release
table. It rejects placeholder release tags, unsafe model roots such as `/` or
`/home`, and unapproved `file://` asset bases before attempting any download.
When the local tar implementation supports it, the text-config archive is
extracted with `--no-same-owner` so reproduction does not depend on restoring
the uid/gid metadata from the archive creator's host. If a final assembled
regular GGUF file already exists under `LOCAL_MODEL_ROOT` and matches the locked
final SHA256, the staging helper reuses it instead of downloading and storing
the split parts again. Symlinked GGUF files are materialized as regular files
because the Docker wrapper mounts only `HOST_MODEL_ROOT` at `/opt/local-models`;
absolute host symlinks outside that tree are not portable release inputs.

Run the public GGUF asset gate against this release tag and a clean local model
root:

```sh
cd qwen36-gfx906
./verify_vnext_public_assets.sh "$RELEASE_TAG" "$LOCAL_MODEL_ROOT"
cd ..
```

This gate uses the same release-asset staging script, then checks that the
staged Dense and MoE GGUF assets satisfy the profile contracts for
`gguf-dense27b-tp8` and `gguf-moe35b-tp4`. It must pass against the final
GitHub Release asset URLs before the GGUF results can be described as public
reproduction claims.

This release uses hosted split GGUF files. A future conversion-based release
would need to replace the release-asset download commands with exact public
conversion commands and expected output hashes.

After staging, a user should be able to verify the public inputs without
knowing any LocalAIServers host path:

```sh
test -f "$LOCAL_MODEL_ROOT/qwen36-27b-f16-gguf/Qwen3.6-27B-F16.gguf"
test -f "$LOCAL_MODEL_ROOT/qwen36-35b-a3b-f16-gguf/Qwen3.6-35B-A3B-F16.gguf"
test -f "$LOCAL_MODEL_ROOT/qwen36-27b-text-config-eosfix/config.json"

sha256sum "$LOCAL_MODEL_ROOT/qwen36-27b-f16-gguf/Qwen3.6-27B-F16.gguf"
sha256sum "$LOCAL_MODEL_ROOT/qwen36-35b-a3b-f16-gguf/Qwen3.6-35B-A3B-F16.gguf"
```

The printed hashes must match the artifact table above before any benchmark
run is considered a reproduction attempt.

Build the model-format probe:

```sh
make -C tools/model-format-probe
```

Inspect the available vNext profiles:

```sh
cd qwen36-gfx906
./vnext_profile_inspect.sh list
./vnext_profile_inspect.sh show gguf-dense27b-tp8
```

Run the read-only serviceability checks:

```sh
./verify_vnext_serviceability.sh
./verify_vnext_contract_matrix.sh
```

For a slower image-path check, opt in to Docker verification:

```sh
CHECK_RUNTIME_IMAGE=1 CHECK_RUNTIME_PATHS=1 ./verify_vnext_serviceability.sh
```

Generate a launch artifact from a selected contract profile. The profile
selects the runtime model path, tensor-parallel degree, overlay, patch bundle,
P2P-on environment, and benchmark defaults. `LOCAL_MODEL_ROOT` is the only
host-specific model path needed for the normal public flow:

```sh
LOCAL_MODEL_ROOT="$LOCAL_MODEL_ROOT" \
./vnext_repro_launcher.sh \
  --profile gguf-dense27b-tp8 \
  --out ./vnext-launch-runs/gguf-dense27b-tp8
```

The launcher maps profile paths under `/opt/local-models` back to
`LOCAL_MODEL_ROOT` for host-side preflight. It verifies the binary model format,
model-family markers, required local config paths, the selected patch-bundle
manifest, P2P-on settings, `MAX_MODEL_LEN=131072`, and `VLLM_DTYPE=half`
before writing a launch artifact.

Use the same shape for each published profile:

```sh
for profile in \
  gguf-dense27b-tp8 \
  gguf-moe35b-tp4 \
  hf-dense27b-tp8 \
  hf-moe35b-tp4 \
  hf-moe35b-tp8
do
  LOCAL_MODEL_ROOT="$LOCAL_MODEL_ROOT" \
  ./vnext_repro_launcher.sh \
    --profile "$profile" \
    --out "./vnext-launch-runs/$profile"
done
```

The profile defaults expect these public staging locations:

| Profile | Public input under `LOCAL_MODEL_ROOT` |
| --- | --- |
| `gguf-dense27b-tp8` | `qwen36-27b-f16-gguf/Qwen3.6-27B-F16.gguf` plus `qwen36-27b-text-config-eosfix/` |
| `gguf-moe35b-tp4` | `qwen36-35b-a3b-f16-gguf/Qwen3.6-35B-A3B-F16.gguf` |
| `hf-dense27b-tp8` | `qwen36-27b-hf/` from `hf download Qwen/Qwen3.6-27B --revision "$HF_DENSE_REVISION"` |
| `hf-moe35b-tp4` | `qwen36-35b-a3b-hf/` from `hf download Qwen/Qwen3.6-35B-A3B --revision "$HF_MOE_REVISION"` |
| `hf-moe35b-tp8` | `qwen36-35b-a3b-hf/` from `hf download Qwen/Qwen3.6-35B-A3B --revision "$HF_MOE_REVISION"` |

Advanced users may still override `MODEL`, `MODEL_PROBE_PATH`, and
`PATCH_BUNDLE_PATH` for nonstandard storage layouts, but the public
reproduction instructions must not require those overrides.

The GGUF tokenizers must use the public Qwen repository ids pinned by each
profile's `TOKENIZER_REVISION`, not an internal Hugging Face snapshot path or
floating `main`.

For example, if an unusual host layout requires explicit paths:

```sh
MODEL=/opt/local-models/qwen36-35b-a3b-hf \
MODEL_PROBE_PATH="$LOCAL_MODEL_ROOT/qwen36-35b-a3b-hf" \
PATCH_BUNDLE_PATH="$(pwd)/overlays/hf/minimal-bundle" \
./vnext_repro_launcher.sh \
  --profile hf-moe35b-tp4 \
  --out ./vnext-launch-runs/hf-moe35b-tp4
```

Verify every generated launch artifact:

```sh
for profile in \
  gguf-dense27b-tp8 \
  gguf-moe35b-tp4 \
  hf-dense27b-tp8 \
  hf-moe35b-tp4 \
  hf-moe35b-tp8
do
  ./verify_vnext_launch_artifact.sh "./vnext-launch-runs/$profile"
done
```

Capture the runtime image's vLLM argument schema with GPU devices exposed:

```sh
mkdir -p ./vnext-launch-runs

docker run --rm \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --security-opt seccomp=unconfined \
  --security-opt label=disable \
  --entrypoint vllm \
  joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b \
  serve --help=all > ./vnext-launch-runs/vllm-serve-help.txt
```

Validate every generated argv against that schema:

```sh
for profile in \
  gguf-dense27b-tp8 \
  gguf-moe35b-tp4 \
  hf-dense27b-tp8 \
  hf-moe35b-tp4 \
  hf-moe35b-tp8
do
  ./verify_vllm_arg_schema.sh \
    "./vnext-launch-runs/$profile" \
    ./vnext-launch-runs/vllm-serve-help.txt
done
```

Run the read-only host platform preflight before starting any benchmark server.
This checks the visible full-BAR/P2P host boundary without patching amdgpu,
flashing firmware, changing kernel settings, starting containers, or running
model workloads. Do not compare a run with the published vNext table if this
preflight fails for the selected profile family:

```sh
for profile in \
  dense27b_tp8_fullbar_p2pon \
  moe35b_tp4_fullbar_p2pon \
  moe35b_tp8_fullbar_p2pon
do
  QWEN36_PROFILE="$profile" ./check_host_platform_prereqs.sh
done
```

Start one generated Docker wrapper at a time after reviewing
`vllm_command.sh`. The published profiles default to port `8001`, so do not
start multiple profiles on the same host port at once. Use a reusable container
name and a separate runtime/cache directory for each profile:

```sh
HOST_MODEL_ROOT="$LOCAL_MODEL_ROOT" \
HOST_HF_CACHE="$LOCAL_HF_CACHE" \
HOST_RUNTIME_ROOT="$LOCAL_RUNTIME_ROOT/gguf-dense27b-tp8" \
VNEXT_CONTAINER_NAME=vnext_repro_current \
./vnext-launch-runs/gguf-dense27b-tp8/docker_run.sh
```

The generated wrapper starts a normal `vllm serve` process. After the server is
ready, normal inference clients should call the OpenAI-compatible vLLM endpoint
directly on the selected port. The begin-think proxy is only part of the Qwen
benchmark validation harness and is not required for ordinary inference. Use
the `served_model_name` recorded in the generated `effective_config.env` when
calling a profile other than the example below:

```sh
curl -fsS http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-27B-F16-GGUF-lmheadunquant",
    "messages": [
      {"role": "user", "content": "Write one concise paragraph about local AI reproducibility."}
    ],
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

Run the normal benchmark ladder:

```sh
./run_vnext_profile_benchmark.sh ./vnext-launch-runs/gguf-dense27b-tp8
```

The benchmark runner uses the profile's generated benchmark environment and
runs the same promotion shape:

```text
8 warmups -> c1_128 uncapped strict -> c1_2000 -> c1_10000
```

To reproduce every profile, repeat the start-and-benchmark step one profile at
a time:

```sh
for profile in \
  gguf-dense27b-tp8 \
  gguf-moe35b-tp4 \
  hf-dense27b-tp8 \
  hf-moe35b-tp4 \
  hf-moe35b-tp8
do
  docker stop vnext_repro_current >/dev/null 2>&1 || true
  docker rm vnext_repro_current >/dev/null 2>&1 || true

  HOST_MODEL_ROOT="$LOCAL_MODEL_ROOT" \
  HOST_HF_CACHE="$LOCAL_HF_CACHE" \
  HOST_RUNTIME_ROOT="$LOCAL_RUNTIME_ROOT/$profile" \
  VNEXT_CONTAINER_NAME=vnext_repro_current \
  "./vnext-launch-runs/$profile/docker_run.sh"

  ./run_vnext_profile_benchmark.sh "./vnext-launch-runs/$profile"

  docker stop vnext_repro_current >/dev/null 2>&1 || true
  docker rm vnext_repro_current >/dev/null 2>&1 || true
done
```

If a profile run fails, stop the reusable container before retrying:

```sh
docker stop vnext_repro_current >/dev/null 2>&1 || true
docker rm vnext_repro_current >/dev/null 2>&1 || true
```

## GGUF Source Notes

The GGUF path reached release-candidate quality only after separating
correctness, loader compatibility, profile selection, and source-path
performance work.

Important promoted findings:

- binary GGUF probing is more reliable than extension-only parsing;
- GGUF tokenizer/config handling must be explicit and must not rely on
  validation-host cache paths;
- GGUF Dense and GGUF MoE need separate overlays;
- GGUF and HF profiles must not share hidden environment state;
- the `lm_head` logits path mattered for Dense GGUF correctness and
  performance;
- MoE semantic correctness required a native Qwen3.6 MoE-aware path rather than
  a generic architecture alias;
- fixed-token throughput and strict thinking-gate validity must remain separate
  measurements;
- backend TPS is the release metric, not screenshots or wall-time-only claims.

MoE top-k8 note: the MoE overlay includes a top-k8 fastpath path for token-sized
decode shapes. Larger graph shapes can be rejected by shape/layout checks. Do
not read this draft as a claim that every MoE graph shape uses the top-k8 path.

## Host Platform Prerequisites

The full-BAR/P2P-on lane still depends on host platform prerequisites outside
the container. The runtime image does not patch the host, flash firmware, or
configure the kernel.

The public v0.2 host-platform record used standardized full-BAR GFX906 VBIOS
revision:

```text
113-D1631700-111
```

The host amdgpu full-BAR/P2P source state is documented separately. Current
public docs lock the recovered source state to the official ROCm
`ROCK-Kernel-Driver` `rocm-7.2.1` source base and include reconstruction notes
for review. The original standalone patch file was not bundled into the
runtime image.

No BIOS/VBIOS binaries are redistributed. This release wording is not a user
instruction to flash cards and does not imply warranty, certification, official
AMD validation, procurement support, resale support, or a hardware
recommendation.

## Validation Status

The passed checks below are supporting validation evidence. Public reproduction
still requires the public GGUF asset gate to pass against the
`vnext-gfx906-rocm72-gguf-hf-repro` GitHub Release asset URLs and a qualifying
full-BAR/P2P host run to reproduce the serving benchmark ladder from those
public inputs.

The vNext path has passed:

- POSIX shell syntax checks for vNext scripts;
- host-tool prerequisite coverage checks against the vNext script command
  surface;
- local serviceability checks;
- release-note lock checks tying artifact hashes, tokenizer revisions, and
  patch-bundle manifests back to scripts/profiles;
- CPU-only contract matrix checks;
- profile-default launcher checks using only `LOCAL_MODEL_ROOT` for host-side
  model staging;
- launch-artifact verification;
- synthetic deploy/profile parity checks for HF MoE TP4/TP8;
- runtime vLLM argument-schema checks on `.20` and `.30`;
- a clean-room release-note preflight that generated and schema-validated
  launch artifacts for all five profiles with synthetic model signatures;
- split-asset staging fixture checks that verify release-manifest assembly and
  reject unsafe GGUF part manifests before download;
- split-asset staging checks that verify release-manifest assembly, final
  hashes, and generated GGUF launch artifacts under fixture or maintainer-local
  asset inputs; public reproduction uses the same GGUF asset gate against the
  published GitHub Release asset URLs;
- a real local split-asset replay using the recovered Dense text-config
  archive, Dense GGUF, and MoE GGUF locks; this passed the public GGUF asset
  gate after the tar ownership portability fix;
- a release-readiness replay against an intentionally unpublished tag that
  stopped at the public GGUF asset gate with a `404`, proving the documented
  sequence fails closed instead of falling back to cached private assets;
- a public HF input gate that checks staged pinned Hugging Face downloads
  against the generated HF launch artifacts;
- explicit HF model and GGUF tokenizer revision pinning;
- clean GGUF Dense TP8 contract runs on `.20` and `.30`;
- clean GGUF MoE TP4 contract runs on `.20` and `.30`;
- a stand-alone GGUF release-package audit on `.30` using split-asset staging,
  final SHA256 verification, generated launch artifacts, and the normal
  benchmark ladder;
- a `.30` full release-readiness replay that ran all five
  vNext profiles through serviceability, contract checks, asset/input gates,
  launch-artifact generation, runtime argument-schema validation, host
  preflight, eight warmups, uncapped strict, `c1_2000`, and `c1_10000`;
- HF MoE TP4 and TP8 reproduction runs on `.20` and `.30`.

Runtime schema note: in this ROCm image, `vllm serve --help=all` still imports
ROCm platform code, so the help capture requires GPU device exposure. A
CPU-only help container fails before printing help.

Benchmark harness note: the vNext benchmark runner uses the bundled
`begin_think_proxy.py` and `run_chat_capture.py` scripts for these Qwen3.6
profiles. The begin-think proxy is a repository-local benchmark harness
component for Qwen thinking-gate validation; it is not an external service and
is not required for normal inference against the generated `vllm serve`
endpoint.

## Boundaries

These are reproducibility and benchmark-methodology artifacts, not general
workload guarantees. The results were measured on the recorded full-BAR/P2P-on
GFX906 lanes and should not be read as a guarantee that every GFX906 system
will reproduce the same numbers.

This package does not provide public compute access, cloud hosting, public SSH,
hardware, procurement support, resale support, fulfillment, discounts,
warranty, certification, official AMD validation, compensation, or professional
services.

## Links

- vNext profile docs:
  [qwen36-gfx906/profiles/vnext/README.md](https://github.com/joe2gaan/localaiservers/blob/vnext-gfx906-rocm72-gguf-hf-repro/qwen36-gfx906/profiles/vnext/README.md)
- GGUF experiment log:
  [docs/gguf-gfx906-experiment-log-20260625.md](https://github.com/joe2gaan/localaiservers/blob/vnext-gfx906-rocm72-gguf-hf-repro/docs/gguf-gfx906-experiment-log-20260625.md)
- GGUF key learnings:
  [docs/gguf-gfx906-key-learnings-20260625.md](https://github.com/joe2gaan/localaiservers/blob/vnext-gfx906-rocm72-gguf-hf-repro/docs/gguf-gfx906-key-learnings-20260625.md)
- GGUF source/kernel inventory:
  [docs/gguf-gfx906-source-kernel-inventory-20260625.md](https://github.com/joe2gaan/localaiservers/blob/vnext-gfx906-rocm72-gguf-hf-repro/docs/gguf-gfx906-source-kernel-inventory-20260625.md)
- Canonical Qwen/GFX906 deployment package:
  [qwen36-gfx906/README.md](https://github.com/joe2gaan/localaiservers/blob/vnext-gfx906-rocm72-gguf-hf-repro/qwen36-gfx906/README.md)
- Host platform prerequisites:
  [docs/gfx906-host-platform-prereqs-v02.md](https://github.com/joe2gaan/localaiservers/blob/vnext-gfx906-rocm72-gguf-hf-repro/docs/gfx906-host-platform-prereqs-v02.md)
- Current GitHub Releases:
  [https://github.com/joe2gaan/localaiservers/releases](https://github.com/joe2gaan/localaiservers/releases)
