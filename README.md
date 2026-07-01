# LocalAIServers

LocalAIServers is a 501(c)(3) public charity providing public education and open-source
infrastructure for locally hosted AI systems.

LocalAIServers preserves affordable AI research infrastructure by maintaining a
controlled air-gapped GFX906 compute site for benchmarking, hardware verification, and
reproducibility work. The cluster is not an interactive-use service. Public benefit is
delivered through published outputs: open-source deployment scripts, reproducible
benchmark reports, QC methods, source-level findings, hardware verification standards,
and educational documentation.

## What This Repository Provides

- Reproducible local AI server configurations.
- GFX906 / ROCm runtime maintenance notes.
- Public benchmark reports.
- Source-level kernel and graph-runtime findings.
- Hardware verification and QC methodology.
- Educational documentation for locally hosted AI systems.

## Published Releases

GitHub Releases are canonical for published release claim boundaries. Docker Hub
remains an evergreen artifact distribution channel and should not be treated as
the latest benchmark announcement. The root deployment package is
[qwen36-gfx906/README.md](qwen36-gfx906/README.md); the latest vNext
profile-specific reproduction package is
[qwen36-gfx906/profiles/vnext/README.md](qwen36-gfx906/profiles/vnext/README.md).

### vNext Stand-Alone GGUF/HF Reproduction Release

`vnext-gfx906-rocm72-gguf-hf-repro` is the current stand-alone release for the
ROCm7.2 GFX906 GGUF/HF reproduction path. It is not a patch, addendum, or
silent replacement for v0.2.0/v0.2.1. It has its own release tag, public GGUF
release assets, HF public-input gates, launcher profiles, validation tables, and
claim boundary.

Use vNext when you want the latest public reproduction package:

```bash
git clone --depth 1 --branch vnext-gfx906-rocm72-gguf-hf-repro https://github.com/joe2gaan/localaiservers.git
cd localaiservers
```

vNext covers five generated `vllm serve` profiles. The release-note
clean-checkout replay evidence includes:

| Release | Format | Profile | Strict backend TPS | `c1_2000` backend TPS | `c1_10000` backend TPS | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| vNext | GGUF F16 | Dense 27B TP8 full-BAR/P2P-on | 70.097 | 71.081 | 66.543 | strict-valid; ai-info 10K gate cleared |
| vNext | GGUF F16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | 118.948 | 120.961 | 113.684 | strict-valid |
| vNext | HF FP16 | Dense 27B TP8 full-BAR/P2P-on | 70.306 | 71.254 | 66.796 | strict-valid; ai-info 10K gate cleared |
| vNext | HF FP16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | 114.557 | 113.436 | 108.044 | strict-valid |
| vNext | HF FP16 | Qwen3.6 35B-A3B MoE TP8 full-BAR/P2P-on | 115.509 | 116.704 | 109.754 | strict-valid |

All vNext profiles preserve `MAX_MODEL_LEN=131072`, `dtype=half`, and
full-BAR/P2P-on requirements. The same ROCm7.2 runtime image and digest are
used:

```text
joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b
sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e
```

Full vNext release notes:
[releases/draft-vnext-gfx906-rocm72-gguf-hf-repro.md](releases/draft-vnext-gfx906-rocm72-gguf-hf-repro.md).

Published vNext GitHub Release:
[https://github.com/joe2gaan/localaiservers/releases/tag/vnext-gfx906-rocm72-gguf-hf-repro](https://github.com/joe2gaan/localaiservers/releases/tag/vnext-gfx906-rocm72-gguf-hf-repro).

### v0.2.1 Reproduction Package

`v0.2.1-gfx906-rocm72-dense-moe-repro` is the easiest checkout for reproducing
the historical v0.2 ROCm7.2 Dense/MoE measurements. It does not change the Docker
image, model package, runtime artifact, benchmark values, or v0.2.0 release
boundary. It packages the corrected public reproduction docs, the v0.2 scorer,
the bundled begin-think proxy, and the 2026-06-22 reproduction report.

Use this tag for reproduction:

```bash
git clone --depth 1 --branch v0.2.1-gfx906-rocm72-dense-moe-repro https://github.com/joe2gaan/localaiservers.git
cd localaiservers/qwen36-gfx906
```

### v0.2.0 Published ROCm7.2 Dense/MoE Artifact

`v0.2.0-gfx906-rocm72-dense-moe` is the ROCm7.2 Dense/MoE GFX906
active-contract benchmark release. Use the v0.2.1 reproduction-package tag
above when you want the simplest source checkout for reproducing the historical
v0.2 results. Use vNext for the latest stand-alone GGUF/HF reproduction path.

| Release | Profile | Strict backend TPS | `c1_2000` backend TPS | `c1_10000` backend TPS | Status |
| --- | --- | ---: | ---: | ---: | --- |
| v0.2.0 | Dense 27B TP8 full-BAR/P2P-on | 69.514 | 70.347 | 66.069 | strict-valid; ai-info 10K gate cleared |
| v0.2.0 | Qwen3.6 35B-A3B MoE TP8 full-BAR/P2P-on | 94.907 | 97.028 | 91.290 | MoE publication bar |
| v0.2.0 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | initial caveat corrected below | 116.146 | 109.283 | release-time fixed-token result; post-v0.2 repeatability passed |

`91.290` is the MoE TP8 `c1_10000` backend TPS.
The MoE TP8 strict backend TPS is `94.907`.
MoE TP4 `109.283` is a capped fixed-token `c1_10000` result.
It is not a strict TPS value.

2026-06-22 public deploy reproduction note: the current public deploy path
reproduced the Dense TP8 10K gate clear, reproduced the corrected MoE TP4
strict/fixed-token band, and reproduced the MoE TP8 strict-valid bar on `.20`.
MoE TP8 fixed-token tiers repeated lower than the release-time high point, so
those fixed-token values should be read as measured release points rather than
repeatability floors. The report is in
[test-reports/qwen36-gfx906-v02-reproduction-20260622/](test-reports/qwen36-gfx906-v02-reproduction-20260622/).

Post-v0.2 validation note: a follow-up MoE TP4 strict repeatability study
passed `6/6` strict repeats across `.20` and `.30` under the native
`moe35b_tp4_fullbar_p2pon` release profile, with strict backend TPS from
`113.196` to `115.995`. This correction found that the earlier TP4 strict
runaway did not reproduce; no code, Docker image, tag, or runtime artifact
changed. The report is in
[test-reports/qwen36-gfx906-moe-tp4-strict-runaway/](test-reports/qwen36-gfx906-moe-tp4-strict-runaway/).

`MAX_MODEL_LEN=131072` is preserved for the v0.2 active contracts. The same
ROCm7.2 image covers the dense and MoE active contracts with model-specific
environment settings and overlays.

Active-contract details:
[docs/rocm72-dense-moe-active-contracts-20260620.md](docs/rocm72-dense-moe-active-contracts-20260620.md).

Full release notes:
[https://github.com/joe2gaan/localaiservers/releases](https://github.com/joe2gaan/localaiservers/releases).

Canonical technical deployment package:
[qwen36-gfx906/README.md](qwen36-gfx906/README.md)

Stable benchmark artifact:
[benchmarks/qwen36-gfx906-mi50-tp4/](benchmarks/qwen36-gfx906-mi50-tp4/)

### v0.1.0 Historical Published Artifact

`v0.1.0-gfx906-qwen36-mi50` is the Qwen3.6 / GFX906 / MI50 TP4 reproducibility
artifact. It remains the historical 90+ TPS sustained 10K backend decode
publication baseline, not the latest ROCm7.2 Dense/MoE or vNext GGUF/HF result.

Canonical reproduction package:
[qwen36-gfx906/README.md](qwen36-gfx906/README.md).

## Controlled Air-Gapped Compute Model

The LocalAIServers GFX906 cluster is controlled research and verification
infrastructure, not an interactive host environment. It is used to validate hardware,
reproduce AI workloads, test source/runtime changes, and publish public outputs. This
allows the public to benefit from the methods, code, benchmark reports, QC standards,
and findings without requiring host access.

See [docs/controlled-air-gapped-compute.md](docs/controlled-air-gapped-compute.md).

## Source-Level GFX906 Maintenance

This repo includes source-level GFX906 kernel/runtime work, not only benchmark scripts.
LocalAIServers documents source-level kernel/runtime adaptation for GFX906-class
systems, including MoE fastpath analysis, dense RowParallel/RCCL collective-boundary
research, graph-runtime integration, rejected-path evidence, promotion methodology, and
technical progress reporting under [docs/](docs/).

Current source-level proof documents:

- [GFX906 source kernel inventory](docs/gfx906-source-kernel-inventory-20260612.md)
- [GFX906 key learnings](docs/gfx906-key-learnings-20260606.md)
- [Technical progress summary](docs/gfx906-technical-progress-summary.md)
- [Experimental methodology](docs/gfx906-experimental-methodology.md)
- [Current research roadmap](docs/gfx906-current-research-roadmap.md)
- [GGUF GFX906 experiment log](docs/gguf-gfx906-experiment-log-20260625.md)
- [GGUF GFX906 key learnings](docs/gguf-gfx906-key-learnings-20260625.md)
- [GGUF GFX906 source/kernel inventory](docs/gguf-gfx906-source-kernel-inventory-20260625.md)

## Public Outputs

Public benefit is delivered through:

- Deployment scripts.
- Docker/runtime details.
- Reproducible benchmark methods.
- QC methodology:
  [docs/qc-methodology.md](docs/qc-methodology.md).
- Hardware verification standards:
  [docs/hardware-verification-standards.md](docs/hardware-verification-standards.md).
- Hardware QC field-check tooling:
  [tools/gfx906-mi50-vram-qc](tools/gfx906-mi50-vram-qc/).
- Source-kernel inventories.
- Experiment and key-learning summaries.
- Public documentation.

## Reproducibility and Promotion Policy

Results are promoted only when they pass optimized serving-path validation, backend
metric checks, and correctness requirements. Diagnostic-only paths and near-tie results
are recorded as source milestones, not serving winners.

See [docs/reproducibility-policy.md](docs/reproducibility-policy.md).

## Roadmap

See [docs/roadmap.md](docs/roadmap.md).

## Funder Proof Index

See [docs/funder-proof-index.md](docs/funder-proof-index.md) for the reviewer-oriented
map of benchmark proof, canonical deployment artifacts, source-level GFX906
preservation work, experimental methodology, QC methods, and the public-output model.

## Public Proof Links

- Latest vNext stand-alone release notes:
  [releases/draft-vnext-gfx906-rocm72-gguf-hf-repro.md](releases/draft-vnext-gfx906-rocm72-gguf-hf-repro.md).
- vNext profile reproduction package:
  [qwen36-gfx906/profiles/vnext/README.md](qwen36-gfx906/profiles/vnext/README.md).
- Canonical technical deployment package:
  [qwen36-gfx906/README.md](qwen36-gfx906/README.md).
- Published GitHub Releases:
  [https://github.com/joe2gaan/localaiservers/releases](https://github.com/joe2gaan/localaiservers/releases).
- Funder proof map:
  [docs/funder-proof-index.md](docs/funder-proof-index.md).
- ROCm7.2 Dense/MoE active-contract notes:
  [docs/rocm72-dense-moe-active-contracts-20260620.md](docs/rocm72-dense-moe-active-contracts-20260620.md).
- QC methodology:
  [docs/qc-methodology.md](docs/qc-methodology.md).
- Hardware verification standards:
  [docs/hardware-verification-standards.md](docs/hardware-verification-standards.md).
- GFX906 / MI50 VRAM QC field-check tool:
  [tools/gfx906-mi50-vram-qc](tools/gfx906-mi50-vram-qc/).

## How To Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

See [CITATION.cff](CITATION.cff).

## License

Code is licensed under [Apache-2.0](LICENSE). Documentation licensing has a public
notice at [docs/LICENSE-DOCS.md](docs/LICENSE-DOCS.md).

## Reproduce Latest vNext Release

Use the vNext tag for the current stand-alone ROCm7.2 GGUF/HF release. This path
is designed to run from public inputs: a tagged repository checkout, the pinned
public Docker image, public GitHub Release assets for GGUF profiles, public Qwen
HF repositories for HF profiles, local SSD/NVMe storage selected by the user,
and a qualifying GFX906 host that satisfies the documented full-BAR/P2P-on
preflight.

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
- standard POSIX/core userland utilities
- `hf` from `huggingface_hub` when reproducing HF profiles

Clone the release checkout:

```bash
git clone --depth 1 --branch vnext-gfx906-rocm72-gguf-hf-repro https://github.com/joe2gaan/localaiservers.git
cd localaiservers
```

Choose large local SSD/NVMe-backed storage. Do not use a small root partition
for model weights, Hugging Face cache, or runtime compile/cache directories.

```bash
export LOCAL_MODEL_ROOT=/mnt/nvme/local-models
export LOCAL_HF_CACHE=/mnt/nvme/hf-cache
export LOCAL_RUNTIME_ROOT=/mnt/nvme/vnext-runtime

mkdir -p "$LOCAL_MODEL_ROOT" "$LOCAL_HF_CACHE" "$LOCAL_RUNTIME_ROOT"
```

Inspect the published profiles without starting containers:

```bash
cd qwen36-gfx906
./vnext_profile_inspect.sh list
./vnext_profile_inspect.sh show gguf-dense27b-tp8
```

Confirm the public release asset URLs are present:

```bash
RELEASE_TAG=vnext-gfx906-rocm72-gguf-hf-repro
./verify_vnext_release_asset_urls.sh "$RELEASE_TAG"
```

Run the full release-readiness path, including serving benchmarks:

```bash
STAGE_HF_PUBLIC_INPUTS=1 \
RUN_SERVING_BENCHMARKS=1 \
./verify_vnext_release_readiness.sh \
  --release-tag "$RELEASE_TAG" \
  --model-root "$LOCAL_MODEL_ROOT" \
  --hf-cache "$LOCAL_HF_CACHE" \
  --runtime-root "$LOCAL_RUNTIME_ROOT"
```

That command sequences serviceability checks, contract matrix validation,
public GGUF asset staging, public HF input staging/validation, launch artifact
generation, vLLM argument-schema validation, host platform preflight, and the
normal serving benchmark ladder:

```text
8 warmups -> c1_128 uncapped strict -> c1_2000 -> c1_10000
```

It prints `vNext full release reproduction path completed` only after the
serving benchmark ladder runs. If `RUN_SERVING_BENCHMARKS=1` is omitted, the
script intentionally refuses to report full release reproduction success.

To run a generated profile as a normal `vllm serve` endpoint after staging
inputs, generate a launch artifact for the profile you want to serve. Start one
profile at a time because the published profiles default to port `8001`:

```bash
LOCAL_MODEL_ROOT="$LOCAL_MODEL_ROOT" \
./vnext_repro_launcher.sh \
  --profile gguf-dense27b-tp8 \
  --out "$LOCAL_RUNTIME_ROOT/vnext-launch-runs/gguf-dense27b-tp8"
```

```bash
HOST_MODEL_ROOT="$LOCAL_MODEL_ROOT" \
HOST_HF_CACHE="$LOCAL_HF_CACHE" \
HOST_RUNTIME_ROOT="$LOCAL_RUNTIME_ROOT/gguf-dense27b-tp8" \
VNEXT_CONTAINER_NAME=vnext_repro_current \
"$LOCAL_RUNTIME_ROOT/vnext-launch-runs/gguf-dense27b-tp8/docker_run.sh"
```

The generated wrapper starts a normal OpenAI-compatible vLLM endpoint. The
begin-think proxy is part of the Qwen benchmark validation harness only and is
not required for ordinary inference.

Full reproduction details:
[qwen36-gfx906/profiles/vnext/README.md](qwen36-gfx906/profiles/vnext/README.md).

## Reproduce Historical v0.2 Results

Use v0.2.1 for the historical v0.2 ROCm7.2 Dense/MoE deployment instructions,
host preflight helper, and prerequisite disclosures. The
`v0.2.1-gfx906-rocm72-dense-moe-repro` tag remains the named reproduction
package for the 2026-06-22 script/report state. The `deploy.sh` script needs
the bundled files under `qwen36-gfx906/files/`; downloading only `deploy.sh` is
not enough for the v0.2 runtime.

```bash
git clone https://github.com/joe2gaan/localaiservers.git
cd localaiservers/qwen36-gfx906
```

This checkout contains the `run_v02_profile_benchmark.sh` scorer path, the
bundled begin-think proxy, and the 2026-06-22 reproduction report. The v0.2.0
benchmark release tag and Docker image identity remain unchanged.

Before deployment, run the read-only host platform preflight. It checks the
visible full-BAR/P2P host state needed for comparable results, but it does not
patch amdgpu, flash firmware, or change host settings:

```bash
./check_host_platform_prereqs.sh
```

The host amdgpu source state required by the full-BAR/P2P-on lane is not bundled
in v0.2.1. The standardized full-BAR GFX906 VBIOS revision recorded for the
public v0.2 host-platform record is `113-D1631700-111`. The recovered amdgpu
state is pinned to `ROCm/ROCK-Kernel-Driver` tag `rocm-7.2.1` for source
review, but it is not a bundled installer or host module package. See
[docs/gfx906-host-platform-prereqs-v02.md](docs/gfx906-host-platform-prereqs-v02.md).

Choose one published v0.2 profile:

```bash
# Dense 27B TP8 full-BAR/P2P-on
export QWEN36_PROFILE=dense27b_tp8_fullbar_p2pon

# MoE TP8 full-BAR/P2P-on
# export QWEN36_PROFILE=moe35b_tp8_fullbar_p2pon

# MoE TP4 full-BAR/P2P-on
# export QWEN36_PROFILE=moe35b_tp4_fullbar_p2pon
```

Deploy the published v0.2 image:

```bash
DEPLOY_IMAGE=joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b \
DOCKER_ISOLATED_DAEMON_ENABLED=0 \
HF_HUB_DISABLE_XET=1 \
USE_PREBUILT_IMAGE=1 \
PREBUILT_IMAGE_PULL=1 \
AUTO_STAGE_MODEL=1 \
./deploy.sh
```

If the host Docker root is intentionally managed and the image/model cache already
has enough space, `SKIP_DISK_SPACE_CHECK=1` can be added. If the Docker root is too
small for normal operation, use the isolated Docker mode documented in
[qwen36-gfx906/README.md](qwen36-gfx906/README.md).

After readiness:

```bash
./smoke-test.sh

./run_v02_profile_benchmark.sh
```

The v0.2 profile benchmark runs the release scorer path: eight 2000-token
pre-measure warmups, the uncapped `c1_128` strict prompt, `c1_2000`, and
`c1_10000` through the bundled begin-think proxy. Use GitHub Releases as the
published claim boundary when comparing local measurements to release numbers.
The older `run_qwen36_live_tps.py` helper remains available for legacy
fixed-token checks, but it is not the v0.2 release scorer.

## Canonical v0.1 Reproduction Instructions

The canonical deployment package is [qwen36-gfx906/README.md](qwen36-gfx906/README.md).
These commands preserve the published v0.1.0 TP4 runtime reproduction path. The
latest vNext GGUF/HF release and the historical v0.2.0 Dense/MoE TPS values are
summarized in Published Releases above. v0.2 details remain in the
[ROCm7.2 active-contract notes](docs/rocm72-dense-moe-active-contracts-20260620.md).
This v0.1 section is retained for historical reproducibility and should not be
read as the latest TPS summary.

The published v0.1 runtime target is:

```text
Model: Qwen/Qwen3.6-35B-A3B
Hardware target: 4x AMD Instinct MI50 32GB
Parallelism: TP4
Runtime image: joe2gaan/localaiservers:qwen36-gfx906-c1-topk8-runtime-archive-aa34cb675f83
Docker Hub digest: sha256:f5e69ee127b766960e386e0e4eda8e26c399bd02f57c494847cb9a92ce04d8ac
```

Start here:

```bash
cd qwen36-gfx906
```

To run from the prebuilt Docker Hub runtime image without rebuilding:

```bash
mkdir -p ~/qwen36-gfx906-run
cd ~/qwen36-gfx906-run

curl -fsSL https://raw.githubusercontent.com/joe2gaan/localaiservers/main/qwen36-gfx906/deploy.sh -o deploy.sh
curl -fsSL https://raw.githubusercontent.com/joe2gaan/localaiservers/main/qwen36-gfx906/run_qwen36_live_tps.py -o run_qwen36_live_tps.py
chmod +x deploy.sh

DEPLOY_IMAGE=joe2gaan/localaiservers:qwen36-gfx906-c1-topk8-runtime-archive-aa34cb675f83 \
DOCKER_ISOLATED_DAEMON_ENABLED=0 \
HF_HUB_DISABLE_XET=1 \
USE_PREBUILT_IMAGE=1 \
PREBUILT_IMAGE_PULL=1 \
AUTO_STAGE_MODEL=1 \
./deploy.sh
```

`DOCKER_ISOLATED_DAEMON_ENABLED=0` uses the host Docker daemon for the prebuilt image
path, which is the correct path for hosts where the user is in the Docker group but
does not have noninteractive sudo. `HF_HUB_DISABLE_XET=1` uses the standard Hugging
Face download path for first-run model staging. With `AUTO_STAGE_MODEL=1`, the script
downloads and verifies the model snapshot before launching vLLM. Byte-for-byte source
rebuild validation remains a separate release-reproduction path.

The image entrypoint launches vLLM with the tested TP4/O3/Tree-LL command:

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --served-model-name Qwen/Qwen3.6-35B-A3B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --dtype half \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  --generation-config vllm \
  -O=3 \
  --async-scheduling \
  --reasoning-parser qwen3 \
  --language-model-only
```

The currently published Docker Hub image is the 29-layer runtime associated with the
strict byte-for-byte source-archive validation target
`aa34cb675f83ff6cade31cbbb357b1c31d793bee18da491f501d7c39fda3612a`. Source rebuilds
should be treated as release-reproduction evidence only when the exported Docker archive
matches that SHA-256. The live `main` deploy script defaults to
`BYTE_FOR_BYTE_VALIDATION_MODE=auto`, which records the archive SHA and enforces a
byte-for-byte target only when `EXPECTED_REPRO_DOCKER_ARCHIVE_SHA256` is set. Set
`EXPECTED_REPRO_DOCKER_ARCHIVE_SHA256=aa34cb675f83ff6cade31cbbb357b1c31d793bee18da491f501d7c39fda3612a`
for strict v0.1.0 source reproduction. Set `BYTE_FOR_BYTE_VALIDATION_MODE=0` only for
non-canonical local deploys where no release-reproduction claim is made. The
prebuilt image path is validated separately by Docker Hub manifest digest:
`sha256:f5e69ee127b766960e386e0e4eda8e26c399bd02f57c494847cb9a92ce04d8ac`, and the
registry config digest matches the tested local image ID:
`sha256:e45309183e6f35cae6fb8f9d8d6f016253f281a5e7187e1f11a57e5e28ef5e86`.

After the vLLM service is ready:

```bash
python3 ./run_qwen36_live_tps.py
```

Released v0.1.0 fixed-token single-request publication baseline on the validated 4x
MI50 32GB lane:

```text
c1_10000: 90+ TPS sustained backend decode publication baseline
```

Earlier interim TP4 validation numbers are superseded in the root README by the
published release table above. Use GitHub Releases for published claim boundaries
and the Published Releases section above for the latest vNext and historical
v0.2.0 Dense/MoE TPS summaries.

See [qwen36-gfx906/README.md](qwen36-gfx906/README.md) for the full deployment and
reproduction notes, including build pins, Docker archive hashes, runtime defaults, disk
checks, and limitations.
