# vNext Reproduction Launcher Profiles

This directory contains contract profiles for the vNext reproduction launcher.
The launcher is experimental scaffolding for a future stand-alone vNext release package.
vNext is not a patch, addendum, or silent replacement for the published v0.2.1
reproduction package; v0.2.1 remains a historical control while vNext must be
reproducible from its own release tag, public assets, instructions, validation
tables, and claim boundary.

Flow:

```text
binary model probe
-> profile contract validation
-> overlay/env selection
-> generated vLLM command
-> saved effective config
-> benchmark ladder handoff
```

Build the binary probe:

```sh
make -C tools/model-format-probe
```

Host tool prerequisites for the portable flow are POSIX `/bin/sh`, `git`,
`docker`, `make`, a C compiler, `python3`, `curl`, `sha256sum`, `tar`,
`split`, `cat`, and standard POSIX/core userland utilities used by the scripts
such as `grep`, `sed`, `awk`, `find`, `sort`, `wc`, `mktemp`, `mkdir`, `rm`,
`mv`, `cp`, `ln`, `chmod`, `touch`, `date`, `sleep`, `kill`, `wait`, `dd`,
`dirname`, `basename`, and `tail`. HF profile staging also requires `hf` from
the public Hugging Face Hub client. GGUF release-asset staging does not require
the Hugging Face CLI for model weights, but GGUF tokenizer resolution still
uses the public Qwen repos pinned by each profile's `TOKENIZER_REVISION`.
Recent `huggingface_hub` releases deprecate the older `huggingface-cli`
command; use `hf download --revision ...` for HF staging.

List or inspect the available profile contracts without starting a container:

```sh
cd qwen36-gfx906
./vnext_profile_inspect.sh list
./vnext_profile_inspect.sh show hf-moe35b-tp4
```

The inspector is read-only. It prints the selected model format, model family,
overlay, image tag/digest, TP degree, `MAX_MODEL_LEN`, dtype, P2P requirement,
patch-bundle requirement, manifest hash, and profile-scoped environment
variables so another developer can audit the profile before launch.

Each profile also declares `PATCH_TARGET_GROUPS`. This is a second isolation
layer beyond the bundle hash: the generated container entrypoint only copies
patch files whose target group is allowed by the profile. For example, HF MoE
TP4 allows `common moe hf`, while HF Dense allows `common dense hf`. A broad
bundle can therefore stay hash-pinned without accidentally activating
dense-only persistent-AR/SWiGLU targets inside a MoE launch, or GGUF-only
loader targets inside an HF launch.

Run the read-only serviceability check before publishing or sharing a vNext
runtime path:

```sh
cd qwen36-gfx906
./verify_vnext_serviceability.sh
```

The serviceability check builds only the local model-format probe, verifies
that all expected HF and GGUF profiles are present, confirms that the public
runtime image tag and digest are pinned, checks `P2P_REQUIRED=1` /
`NCCL_P2P_DISABLE=0`, verifies `MAX_MODEL_LEN=131072` and `VLLM_DTYPE=half`,
checks that HF profiles do not leak GGUF-only environment variables, verifies
the pinned patch-bundle manifests, checks that the draft release note's
artifact hashes and manifest locks match the staging script/profile pins, and
scans developer-facing vNext files for private host/path references. It does
not start Docker, patch a host, inspect live GPUs, mutate model caches, or run
benchmarks.

For a slower clean-developer image check, opt in to Docker verification:

```sh
cd qwen36-gfx906
CHECK_RUNTIME_IMAGE=1 CHECK_RUNTIME_PATHS=1 ./verify_vnext_serviceability.sh
```

That mode verifies that the public Docker tag resolves to the pinned manifest
digest and starts a short `/bin/sh` container to confirm the baseline native
runtime paths used by the profiles exist in the public image. It may pull the
large runtime image if it is not already present locally, but it still does not
touch GPUs or run benchmarks.

Run the CPU-only contract matrix when changing launcher/profile logic:

```sh
cd qwen36-gfx906
./verify_vnext_contract_matrix.sh
```

That check builds the C model probe, creates tiny synthetic GGUF and
safetensors model signatures, generates launch artifacts for all supported
profiles, verifies those artifacts, and confirms expected mismatch cases fail
closed. It exercises GGUF-vs-HF format separation, Dense-vs-MoE family
separation, patch-target group isolation, HF GGUF-env leakage rejection, and
patch-bundle hash mismatch rejection without starting Docker or touching GPUs.

When deriving a vNext profile from a known-good `deploy.sh` run, compare the
captured `.deploy.runtime.env` against the profile and generated artifact:

```sh
cd qwen36-gfx906
./verify_vnext_deploy_profile_parity.sh \
  hf-moe35b-tp4 \
  /path/to/.deploy.runtime.env \
  ./vnext-launch-runs/hf-moe35b-tp4-check
```

That check verifies the profile name, TP degree, `MAX_MODEL_LEN`, dtype,
async-scheduling toggle, P2P/NCCL settings, MoE fastpath flag, deploy-style
tuned-config path, language-model-only mapping, and generated argv shape. It is
the intended way to compare vNext HF/MoE contracts against a high-band
`deploy.sh` artifact without manually diffing environment dumps.

Generated Docker launch artifacts also preserve the generic release runtime
shape used by `deploy.sh`: ROCm target-device defaults, `gfx906` arch exports,
`OMP_NUM_THREADS=4`, `TORCH_BLAS_PREFER_HIPBLASLT=0`, and privileged container
mode. Those settings are runtime-shape guardrails, not profile-specific vLLM
arguments, and they keep the image behavior closer to the published deploy
path while the vLLM argv remains profile-generated.
For HF profiles, the generated wrapper also resolves a symlinked model
directory below `HOST_MODEL_ROOT` and mounts the resolved target over the
profile's `/opt/local-models/...` path. That permits developers to keep large
HF snapshots on any local NVMe path without relying on a validation-host
absolute path inside the container. If that target is a Hugging Face cache
snapshot with symlinked safetensors shards, the wrapper uses a runtime
`model-bind-root` shim under `HOST_RUNTIME_ROOT`, mounts the resolved snapshot
under `/opt/vnext-models/...`, and mounts the matching snapshot `blobs` directory read-only at the container path required by those relative shard links.

The image should stay serviceable as a normal developer runtime: the public
image pin, selected overlay, model format, TP degree, P2P requirement, and
patch-bundle hash must be visible from profile metadata before launch. Hidden
lab-only paths, host-specific defaults, or benchmark-only one-off settings are
not acceptable release-profile inputs.

Generated launch artifacts are local outputs, not reusable release assets.
`effective_config.env` records the local model probe path and local patch
bundle path used to create the artifact so the run can be audited later.
`docker_run.sh` still requires the user to supply `HOST_MODEL_ROOT`,
`HOST_HF_CACHE`, and `HOST_RUNTIME_ROOT` when starting the container.
If an HF model directory under `HOST_MODEL_ROOT` is a symlink to another
NVMe-backed directory, the generated Docker wrapper resolves that symlink and
mounts the target at the same `/opt/local-models/...` container path. Broken
or non-directory model symlinks fail closed before benchmark launch. Hugging
Face cache snapshots with symlinked shard files also require a readable
snapshot `blobs` directory. The wrapper creates a runtime `model-bind-root`
shim and mounts resolved snapshots under `/opt/vnext-models/...`; unresolved snapshot shard links fail closed before benchmark launch.
Container-internal paths such as `/usr/share/ollama/kernel_labs/...` are
created inside the runtime container from the mounted patch bundle and are not
host validation-workspace dependencies.

## Portable Reproduction Standard

The vNext package is release-ready only if another developer can reproduce the
path from public inputs:

- a fresh clone of the tagged repository checkout;
- the pinned public runtime image and digest;
- public upstream Hugging Face model repositories for HF profiles;
- published GitHub Release assets and SHA256 manifests for GGUF profiles;
- local SSD/NVMe storage selected by that developer; and
- a qualifying GFX906 host that satisfies the documented full-BAR/P2P-on
  prerequisites.

The reproduction flow must not depend on `.20`, `.30`, validation-host network
addresses, validation-host storage roots, retained Hugging Face snapshots,
private mounts, copied validation-host model files, or any LocalAIServers
internal path. Host labels such as `.20` and `.30` are validation evidence
labels only; they are not deployment targets.

The HF public-input gate verifies that staged HF directories include tokenizer
files, `model.safetensors.index.json`, and every non-empty shard named by that
index. It rejects config-only directories and unresolved Git LFS pointer files;
`hf download` must materialize the real model package before the HF profiles
can be treated as staged. Symlinked HF model directories are accepted only when
they resolve to real local directories that satisfy the same shard checks; the
generated Docker wrapper then mounts that resolved directory at the expected
container path.

`RELEASE_ASSET_BASE` is a maintainer test hook for checking the split-asset
layout before upload. Published user instructions must work against the final
GitHub Release asset URLs for the selected tag. A `file://`
`RELEASE_ASSET_BASE` requires `ALLOW_LOCAL_ASSET_PREFLIGHT=1` and must never be
used as a published reproduction input.
The GGUF staging script assembles split model files only from validated
manifest rows and rejects part names with paths, parent-directory traversal, or
extra fields before download. It also rejects placeholder release tags, unsafe
model roots such as `/` or `/home`, and unapproved `file://` asset bases before
attempting any download.
Before publishing, run the public asset gate against the final release tag and
an empty or intentionally selected model root:

```sh
cd qwen36-gfx906
RELEASE_TAG=<release-tag>
LOCAL_MODEL_ROOT=${LOCAL_MODEL_ROOT:-/mnt/nvme/local-models}
./verify_vnext_public_assets.sh "$RELEASE_TAG" "$LOCAL_MODEL_ROOT"
```

That gate stages the published GGUF assets, verifies the final SHA256 values,
and checks that the staged Dense and MoE GGUF files satisfy their profile
contracts.

For the full release-note path, use the readiness verifier from a clean
checkout after selecting local SSD/NVMe-backed storage:

```sh
cd qwen36-gfx906
RELEASE_TAG=<release-tag>
STAGE_HF_PUBLIC_INPUTS=1 \
RUN_SERVING_BENCHMARKS=1 \
./verify_vnext_release_readiness.sh \
  --release-tag "$RELEASE_TAG" \
  --model-root "$LOCAL_MODEL_ROOT" \
  --hf-cache "$LOCAL_HF_CACHE" \
  --runtime-root "$LOCAL_RUNTIME_ROOT"
```

Use `STAGE_HF_PUBLIC_INPUTS=1` when the HF model directories under
`$LOCAL_MODEL_ROOT` are absent or are writable regular directories for
`hf download`. If the pinned HF packages are already staged and verified,
including symlinked local snapshot directories, rerun with
`STAGE_HF_PUBLIC_INPUTS=0` or omit the variable; the verifier still checks
tokenizer files, the safetensors index, and every referenced shard.

The readiness verifier sequences serviceability, contract matrix, public GGUF
asset staging, public HF staging, launch-artifact generation, runtime vLLM
argument-schema validation, host platform preflight, and the serving benchmark
ladder. It exits nonzero before benchmark launch if the public assets are
missing or if the host fails the full-BAR/P2P-on preflight. Without
`RUN_SERVING_BENCHMARKS=1`, it intentionally refuses to report full
reproduction success unless `ALLOW_PRE_SERVING_ONLY=1` is set for a maintainer
dry run.
Each readiness invocation writes generated launch artifacts under a unique
timestamped directory below `LOCAL_RUNTIME_ROOT/vnext-launch-runs/` so repeated
runs do not need to remove container-owned files from a previous attempt.

The final public vNext replay must be run from the stand-alone vNext tag or
pinned release commit, not from v0.2.0, v0.2.1, a validation-host checkout, or
a maintainer-only `file://` asset mirror. Before calling the package publicly
reproduced, confirm that:

- a clean checkout of the final tag contains this profile directory, the vNext
  release note, and the vNext verification/launcher scripts;
- `./verify_vnext_release_asset_urls.sh <release-tag>` passes against the real
  GitHub Release assets;
- no replay uses `RELEASE_ASSET_BASE=file://...` or
  `ALLOW_LOCAL_ASSET_PREFLIGHT=1`;
- `RUN_SERVING_BENCHMARKS=1` is set for the full readiness command;
- the readiness command prints `vNext full release reproduction path
  completed`;
- all five profiles write benchmark summaries under
  `LOCAL_RUNTIME_ROOT/vnext-launch-runs/...`;
- every strict row is `qwen_gate_valid=true`;
- Dense 27B `c1_10000` clears the 65 TPS ai-info gate; and
- no v0.2.0 or v0.2.1 release claim is revised by this vNext evidence.

Minimum tree-content check from a fresh checkout:

```sh
test -f releases/draft-vnext-gfx906-rocm72-gguf-hf-repro.md
test -f qwen36-gfx906/verify_vnext_release_readiness.sh
test -f qwen36-gfx906/vnext_repro_launcher.sh
test -f qwen36-gfx906/run_vnext_profile_benchmark.sh
test -d qwen36-gfx906/profiles/vnext
test -d qwen36-gfx906/overlays
```

Generate launch commands from profile defaults. The profile selects the
container model path, tensor-parallel degree, overlay, patch bundle, P2P-on
environment, and benchmark defaults. `LOCAL_MODEL_ROOT` is the only
host-specific model path needed for the normal public flow:

```sh
cd qwen36-gfx906
export RELEASE_TAG=<release-tag>
export LOCAL_MODEL_ROOT=/mnt/nvme/local-models
export LOCAL_HF_CACHE=/mnt/nvme/hf-cache
export LOCAL_RUNTIME_ROOT=/mnt/nvme/vnext-runtime

mkdir -p "$LOCAL_MODEL_ROOT" "$LOCAL_HF_CACHE" "$LOCAL_RUNTIME_ROOT"

LOCAL_MODEL_ROOT="$LOCAL_MODEL_ROOT" \
  ./vnext_repro_launcher.sh \
  --profile gguf-dense27b-tp8 \
  --out ./vnext-launch-runs/gguf-dense27b-tp8
```

`/mnt/nvme` is only an example. Use any large local SSD/NVMe-backed path on
your own host. Do not use a small root partition for model weights, Hugging
Face cache, runtime cache, or compile output.

For GGUF profiles, the referenced GGUF files must be staged under
`$LOCAL_MODEL_ROOT` from public release artifacts or a locked public conversion
recipe before launch. The preferred portable path is for the release to attach
split GitHub Release assets for the exact validated F16 GGUF files, publish
part manifests, and verify both the parts and final assembled files by SHA256
before launch. If hosted split assets are not used, the release notes must
provide the exact upstream model revision, converter source revision,
conversion command, output filename, and SHA256 before a GGUF reproduction
package is considered portable.

A clean-room reproduction must work from a fresh checkout, the public Docker
image, public model or release assets, and local SSD/NVMe-backed storage. It
must not require a LocalAIServers validation host, private network address,
private mount, pre-existing local model cache, or unpublished converted GGUF
file.

After staging the public Hugging Face model directories, verify the HF profile
contracts:

```sh
cd qwen36-gfx906
./verify_vnext_public_hf_inputs.sh "$LOCAL_MODEL_ROOT"
```

The HF verifier rejects unsafe model roots such as `/` or `/home`, non-pinned
HF revisions, invalid `STAGE_HF_PUBLIC_INPUTS` values, and, when staging
downloads, unsafe HF cache roots such as `/` or `/home`.

To let the gate run the pinned public downloads before validation:

```sh
cd qwen36-gfx906
HF_HOME="$LOCAL_HF_CACHE" STAGE_HF_PUBLIC_INPUTS=1 \
  ./verify_vnext_public_hf_inputs.sh "$LOCAL_MODEL_ROOT"
```

Do not depend on LocalAIServers validation hosts, internal mount paths, or
local-only converted model files for public reproduction. Host labels such as
`.20` and `.30` identify validation lanes only; they are not deployment
targets.

Public BF16 or quantized GGUF files are not substitutes for the F16 GGUF files
used by the current benchmark evidence unless a separate result is published
for those exact artifacts.

The portable split-asset staging helper is:

```sh
cd qwen36-gfx906
./stage_vnext_gguf_assets.sh "$RELEASE_TAG" "$LOCAL_MODEL_ROOT"
```

It downloads the Dense text-config/tokenizer archive, downloads the GGUF part
manifests, downloads every listed part, verifies the part hashes, assembles the
final F16 GGUF files, and verifies final file SHA256 values before launch. The
GGUF profiles use public Qwen tokenizer repos pinned by `TOKENIZER_REVISION`;
they must not point at validation-host Hugging Face snapshot paths or floating
`main`. When supported by the local tar implementation, the text-config archive
is extracted with `--no-same-owner` so reproduction does not depend on archive
uid/gid metadata. If a final assembled regular GGUF file already exists under
`LOCAL_MODEL_ROOT` and matches the locked final SHA256, the staging helper
reuses it instead of downloading and storing the split parts again. Symlinked
GGUF files are materialized as regular files because the Docker wrapper mounts
only `HOST_MODEL_ROOT` at `/opt/local-models`; absolute host symlinks outside
that tree are not portable release inputs.

When running host-side preflight against container-only config paths, use
`LOCAL_MODEL_ROOT` so paths under `/opt/local-models` are mapped back to the
host staging directory. The launcher verifies the model bytes, config paths,
patch-bundle manifest, P2P-on settings, `MAX_MODEL_LEN=131072`, and
`VLLM_DTYPE=half` before writing an artifact.

Use the same profile-driven shape for each published profile:

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

The profile defaults expect these staging locations:

| Profile | Public input under `LOCAL_MODEL_ROOT` |
| --- | --- |
| `gguf-dense27b-tp8` | `qwen36-27b-f16-gguf/Qwen3.6-27B-F16.gguf` plus `qwen36-27b-text-config-eosfix/` |
| `gguf-moe35b-tp4` | `qwen36-35b-a3b-f16-gguf/Qwen3.6-35B-A3B-F16.gguf` |
| `hf-dense27b-tp8` | `qwen36-27b-hf/` from `hf download Qwen/Qwen3.6-27B --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| `hf-moe35b-tp4` | `qwen36-35b-a3b-hf/` from `hf download Qwen/Qwen3.6-35B-A3B --revision 995ad96eacd98c81ed38be0c5b274b04031597b0` |
| `hf-moe35b-tp8` | `qwen36-35b-a3b-hf/` from `hf download Qwen/Qwen3.6-35B-A3B --revision 995ad96eacd98c81ed38be0c5b274b04031597b0` |

`MODEL`, `MODEL_PROBE_PATH`, and `PATCH_BUNDLE_PATH` remain available for
nonstandard local storage layouts, but the public reproduction path should not
require them.

The launcher writes:

- `effective_config.env`
- `vllm_command.sh`
- `vllm_argv.txt`
- `benchmark_env.sh`
- `preflight.txt`
- `container_entrypoint.sh`
- `docker_run.sh`

`effective_config.env` records the selected runtime image tag/digest, detected
model format, detected model family when available, profile, overlay, patch
target groups, patch bundle path, patch bundle manifest hash, P2P state, and
the benchmark ladder shape.

Verify every generated launch artifact before starting a server:

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

Start one server at a time from the generated Docker wrapper only after
reviewing that profile's `vllm_command.sh`. Keep model and runtime/cache paths
on NVMe-backed storage. `HOST_MODEL_ROOT` is the host directory mounted at
`/opt/local-models` inside the container. `HOST_HF_CACHE` is the host Hugging
Face cache directory. The profile defaults use port `8001`, so do not start
multiple profiles on the same host port at once.

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

Capture the `vllm serve` help text from the exact runtime image, then validate
every generated argv against it. This catches vLLM flag drift before a
benchmark run starts:

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

On this ROCm image, the help capture still imports ROCm platform code, so it
must expose GPU devices even though it does not load a model. A CPU-only help
container fails before printing help. Runtime arg-schema checks passed on
`.20` and `.30` for the clean GGUF Dense, GGUF MoE, HF MoE TP4, and HF MoE TP8
contract artifacts.

Run the read-only host platform preflight before starting benchmark servers.
This script checks the visible full-BAR/P2P host state without patching amdgpu,
flashing firmware, changing kernel settings, starting containers, or running
model workloads. A failing preflight means the host is outside the comparable
release platform boundary for that profile family:

```sh
for profile in \
  dense27b_tp8_fullbar_p2pon \
  moe35b_tp4_fullbar_p2pon \
  moe35b_tp8_fullbar_p2pon
do
  QWEN36_PROFILE="$profile" ./check_host_platform_prereqs.sh
done
```

Then run the normal benchmark ladder. For the full profile set, repeat the
server start and benchmark step one profile at a time:

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

For these Qwen3.6 profiles, the benchmark runner uses the bundled
`begin_think_proxy.py` and `run_chat_capture.py` scripts to validate the
thinking-gate path. That proxy is a local benchmark harness component, not a
dependency on any external proxy service and not a general model-serving
requirement for normal inference against the generated `vllm serve` endpoint.

Patch bundle manifests:

```sh
./hash_vnext_patch_bundle.sh create ./overlays/gguf-moe/minimal-bundle
./hash_vnext_patch_bundle.sh verify ./overlays/gguf-moe/minimal-bundle <manifest-sha256>
```

The profile's `PATCH_BUNDLE_MANIFEST_SHA256` must match the generated
`sha256sum.txt` manifest before a profile is release-ready.
Release-ready manifests must use deterministic relative paths, must exclude
`sha256sum.txt`, and must pass `sha256sum -c sha256sum.txt` from the bundle
root. Scratch manifests with absolute paths or self-inclusion are rejected by
the launcher when `CHECK_REQUIRED_PATHS=1`.

The vNext profile contracts are pinned to the minimal bundle manifest hashes:

- HF Dense TP8 and HF MoE TP4/TP8:
  `d9e1168f67731ab66ee53671f89dfb31fd3c74c8f61e2f0d0b7c4f0704c298be`
- Dense GGUF TP8:
  `ce2c6f2d974de34c7abf4c25a67985b0eccc452a97f1a887ca97a8de1687f8d2`
- MoE GGUF TP4:
  `794cb8760003cfa7dafeb0f06ee01823e5e8eebb3f9d398d8bc8bd7697143f0b`

Development preflight can pass `VNEXT_PATCH_BUNDLE_MANIFEST_SHA256=<sha256>`
or `ALLOW_UNPINNED_PATCH_BUNDLE=1`, but those overrides are not release
reproduction settings.

Guardrails:

- HF/non-GGUF profiles reject GGUF model bytes.
- GGUF profiles reject HF/safetensors model bytes.
- HF directory profiles inspect `config.json` for obvious Dense/MoE family
  mismatches.
- GGUF file profiles inspect bounded header/metadata bytes for obvious
  Dense/MoE family mismatches instead of scanning entire multi-GB files.
- HF/non-GGUF profiles reject leaked `VLLM_QWEN35_GGUF_*`,
  `VLLM_GFX906_GGUF_*`, and `VLLM_GGUF_*` environment variables.
- Dense profiles reject MoE-only fastpath envs.
- MoE profiles reject Dense-only GGUF overlays.
- Generated entrypoints gate patch-copy targets by profile group, so dense-only
  native/Persistent-AR/SWiGLU targets, MoE-only targets, and GGUF-only loader
  targets cannot activate just because a file exists in a broad bundle.
- P2P-required profiles require `NCCL_P2P_DISABLE=0`.
- `MAX_MODEL_LEN` must be `131072`.
- `VLLM_DTYPE` must be `half`.
- GGUF profiles require pinned patch-bundle manifests before they are
  release-ready.
- Runtime image tag and digest are recorded in every generated launch artifact.
- Generated vLLM argv is saved as one argument per line so runtime-help schema
  checks can reject renamed or removed vLLM flags before launch.

Current validation status:

- The CPU-only contract matrix now covers both explicit advanced overrides and
  the public profile-default path where `LOCAL_MODEL_ROOT` is the only
  host-specific model staging input.
- Dense GGUF TP8 package path has reproduced the published Dense 27B band on
  `.20` and `.30` from a clean vNext contract run. Run
  `20260629T003817Z-gguf-dense27b-tp8` produced `.20` strict `69.914`,
  c1_2000 `70.759`, and c1_10000 `66.316`; and `.30` strict `69.851`,
  c1_2000 `70.959`, and c1_10000 `66.437`. Both strict runs stopped normally
  and passed the Qwen gate.
- MoE GGUF TP4 package path has reproduced the high-band MoE TP4 result on
  `.20` and `.30` from a clean vNext contract run. Run
  `20260629T003817Z-gguf-moe35b-tp4` produced `.20` strict `119.333`,
  c1_2000 `120.565`, and c1_10000 `113.366`; and `.30` strict `119.521`,
  c1_2000 `120.460`, and c1_10000 `113.258`. Both strict runs stopped normally
  and passed the Qwen gate.
- HF Dense TP8 package path has reproduced the historical non-GGUF HF Dense
  band on `.20` after restoring the expanded clean HF release bundle.
- HF MoE TP4 package path has reproduced the high-band non-GGUF TP4 lane on
  `.20` and `.30` after profile-gated patch target activation. `.20` produced
  strict `115.105`, c1_2000 `115.933`, and c1_10000 `109.095`; `.30` produced
  strict `114.409`, c1_2000 `115.685`, and c1_10000 `108.918`.
- HF MoE TP8 package path has reproduced the high-band non-GGUF TP8 lane on
  `.20` and `.30` after profile-gated patch target activation. `.20` produced
  strict `115.041`, c1_2000 `115.549`, and c1_10000 `108.805`; `.30` produced
  strict `114.698`, c1_2000 `115.532`, and c1_10000 `108.673`.
- A clean public-staged HF audit on `.30` downloaded `Qwen/Qwen3.6-27B` and
  `Qwen/Qwen3.6-35B-A3B` with `hf download`, generated fresh vNext launch
  artifacts, validated vLLM argv against the runtime image help, and reran the
  normal benchmark ladder. It produced HF Dense TP8 strict `69.682`, c1_2000
  `71.346`, and c1_10000 `66.729`; HF MoE TP4 strict `111.964`, c1_2000
  `115.937`, and c1_10000 `108.848`; and HF MoE TP8 strict `115.488`,
  c1_2000 `116.258`, and c1_10000 `109.242`.
- A clean-room GGUF package-path audit on `.30` copied the current repository
  tree into a fresh checkout directory, staged GGUF inputs through the
  split-release-asset layout, verified final SHA256 values, generated launch
  artifacts using only `LOCAL_MODEL_ROOT` for host model-path mapping, validated
  the runtime vLLM argument schema, and reran the normal benchmark ladder. It
  produced GGUF Dense TP8 strict `69.580`, c1_2000 `70.837`, and c1_10000
  `66.309`; and GGUF MoE TP4 strict `117.731`, c1_2000 `119.649`, and
  c1_10000 `112.568`. This used `RELEASE_ASSET_BASE=file://...` only to
  simulate final GitHub Release assets; published instructions must use real
  release asset URLs.
- A later `.30` full release-readiness replay ran the documented wrapper
  through all five profiles with `RUN_SERVING_BENCHMARKS=1` from a fresh
  temporary copy of the current vNext package. It completed serviceability,
  contract checks, asset/input gates, launch-artifact generation, runtime
  argument-schema validation, host preflight, eight warmups, uncapped strict,
  `c1_2000`, and `c1_10000`. Results were: GGUF Dense TP8 strict `69.873`,
  c1_2000 `71.161`, c1_10000 `66.638`; GGUF MoE TP4 strict `119.363`,
  c1_2000 `119.808`, c1_10000 `112.755`; HF Dense TP8 strict `70.474`,
  c1_2000 `71.326`, c1_10000 `66.810`; HF MoE TP4 strict `115.592`,
  c1_2000 `116.062`, c1_10000 `109.240`; and HF MoE TP8 strict `114.652`,
  c1_2000 `115.895`, c1_10000 `109.001`. This is still release-candidate
  evidence because it used maintainer local simulated release assets; final
  public reproduction requires the same replay from the stand-alone vNext tag
  and public GitHub Release asset URLs.
- A second `.30` clean-checkout serving replay from the local proof commit
  repeated the same documented flow and completed all five generated
  `vllm serve` profiles. Results were: GGUF Dense TP8 strict `70.097`,
  c1_2000 `71.081`, c1_10000 `66.543`; GGUF MoE TP4 strict `118.948`,
  c1_2000 `120.961`, c1_10000 `113.684`; HF Dense TP8 strict `70.306`,
  c1_2000 `71.254`, c1_10000 `66.796`; HF MoE TP4 strict `114.557`,
  c1_2000 `113.436`, c1_10000 `108.044`; and HF MoE TP8 strict `115.509`,
  c1_2000 `116.704`, c1_10000 `109.754`. Exact TPS equality with a single
  evidence row is not required for reproduction; the pass condition is the full
  readiness wrapper completing, strict rows passing the Qwen gate, Dense
  c1_10000 clearing 65 TPS, and each profile staying in the observed validation
  band for the selected lane.
- Intentional model-format and HF/GGUF env-leak mismatches fail closed.

Release-readiness work that remains before public promotion:

- attach the split GGUF assets, part manifests, final SHA256 files, and Dense
  text-config archive to the GitHub Release;
- rerun or verify the exact public package path from a clean checkout against
  the real GitHub Release asset URLs when selecting a release candidate;
- copy only sanitized logs/results into public test reports;
- decide whether GGUF results are a public release claim or still an
  experimental source-work artifact;
- promote public claims only after package-path validation, not scratch
  overlays, is complete.
