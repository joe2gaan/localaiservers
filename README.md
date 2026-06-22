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
the latest benchmark announcement. The canonical deployment and reproducibility
package is [qwen36-gfx906/README.md](qwen36-gfx906/README.md).

### v0.1.0 Historical Published Artifact

`v0.1.0-gfx906-qwen36-mi50` is the Qwen3.6 / GFX906 / MI50 TP4 reproducibility
artifact. It remains the historical 90+ TPS sustained 10K backend decode
publication baseline, not the latest ROCm7.2 Dense/MoE result.

Canonical reproduction package:
[qwen36-gfx906/README.md](qwen36-gfx906/README.md).

### v0.2.0 Published ROCm7.2 Dense/MoE Artifact

`v0.2.0-gfx906-rocm72-dense-moe` is the current ROCm7.2 Dense/MoE GFX906
active-contract release.

| Release | Profile | Strict backend TPS | `c1_2000` backend TPS | `c1_10000` backend TPS | Status |
| --- | --- | ---: | ---: | ---: | --- |
| v0.2.0 | Dense 27B TP8 full-BAR/P2P-on | 69.514 | 70.347 | 66.069 | strict-valid; ai-info 10K gate cleared |
| v0.2.0 | Qwen3.6 35B-A3B MoE TP8 full-BAR/P2P-on | 94.907 | 97.028 | 91.290 | strict-valid MoE publication bar |
| v0.2.0 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | invalid/runaway | 116.146 | 109.283 | capped fixed-token only; not strict-valid |

`91.290` is the MoE TP8 `c1_10000` backend TPS; the MoE TP8 strict backend
TPS is `94.907`. MoE TP4 is capped-only and is not strict-valid.

Post-v0.2 validation note: a follow-up MoE TP4 strict repeatability study
passed `6/6` strict repeats across `.20` and `.30` under the native
`moe35b_tp4_fullbar_p2pon` release profile, with strict backend TPS from
`113.196` to `115.995`. This is post-v0.2 validation evidence and does not
change the published v0.2.0 release boundary. The report is in
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

## Canonical v0.1 Reproduction Instructions

The canonical deployment package is [qwen36-gfx906/README.md](qwen36-gfx906/README.md).
These commands preserve the published v0.1.0 TP4 runtime reproduction path. The
latest published v0.2.0 Dense/MoE TPS values are summarized in Published Releases
above and in the
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
non-canonical local deploys where no release-reproduction claim is being made. The
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

The canonical Qwen3.6 README also preserves a newer 95+ TPS 10K validation
result for the same runtime lane. That result is not part of the v0.1.0
publication release. Use GitHub Releases for published claim boundaries and the
Published Releases section above for the latest v0.2.0 Dense/MoE TPS summary:

```text
c1_2000:  101.47 TPS backend decode
c1_10000:  95.66 TPS backend decode
c1_10000:  95.36 client wall TPS
```

See [qwen36-gfx906/README.md](qwen36-gfx906/README.md) for the full deployment and
reproduction notes, including build pins, Docker archive hashes, runtime defaults, disk
checks, and limitations.
