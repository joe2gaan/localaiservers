# LocalAIServers

LocalAIServers is a 501(c)(3) public charity providing public education and open-source
infrastructure for locally hosted AI systems.

LocalAIServers preserves affordable AI research infrastructure by maintaining controlled
air-gapped GFX906 compute for benchmarking, hardware verification, and reproducibility
work. The cluster is not a public login service. Public benefit is delivered through
published outputs: open-source deployment scripts, reproducible benchmark reports, QC
methods, source-level findings, hardware verification standards, and educational
documentation.

## What This Repository Provides

- Reproducible local AI server configurations.
- GFX906 / ROCm runtime maintenance notes.
- Public benchmark reports.
- Source-level kernel and graph-runtime findings.
- Hardware verification and QC methodology.
- Educational documentation for locally hosted AI systems.

## Current Flagship Artifact

The current flagship public artifact is the Qwen3.6 / GFX906 / MI50 TP4 runtime work:

- Hardware target: 4x AMD Instinct MI50 32GB / GFX906.
- Model: `Qwen/Qwen3.6-35B-A3B`.
- Runtime: reproducible Docker/vLLM/ROCm deployment package.
- Public benchmark summary: sustained 90+ TPS backend decode on the documented
  fixed-token long-decode run, with a shorter fixed-token backend decode result over 100
  TPS.

Canonical technical deployment package:
[qwen36-gfx906/README.md](qwen36-gfx906/README.md)

Stable benchmark artifact:
[benchmarks/qwen36-gfx906-mi50-tp4/](benchmarks/qwen36-gfx906-mi50-tp4/)

## Controlled Air-Gapped Compute Model

The LocalAIServers GFX906 cluster is controlled research and verification
infrastructure, not an open public login environment. It is used to validate hardware,
reproduce AI workloads, test source/runtime changes, and publish public outputs. This
allows the public to benefit from the methods, code, benchmark reports, QC standards,
and findings without requiring direct access to the cluster.

See [docs/controlled-air-gapped-compute.md](docs/controlled-air-gapped-compute.md).

## Source-Level GFX906 Maintenance

LocalAIServers documents source-level kernel/runtime adaptation for GFX906-class
systems. The work includes MoE fastpath analysis, dense RowParallel/RCCL
collective-boundary research, rejected-path evidence, promotion methodology, and
technical progress reporting under [docs/](docs/).

Current source inventory and key-learning documents are represented as placeholders
until the source files are added to the public repository:

- [GFX906 source kernel inventory](docs/gfx906-source-kernel-inventory-20260612.md)
- [GFX906 key learnings](docs/gfx906-key-learnings-20260606.md)
- [Technical progress summary](docs/gfx906-technical-progress-summary.md)
- [Experimental methodology](docs/gfx906-experimental-methodology.md)

## Public Outputs

Public benefit is delivered through:

- Deployment scripts.
- Docker/runtime details.
- Reproducible benchmark methods.
- QC and hardware verification methods.
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

## How To Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

See [CITATION.cff](CITATION.cff).

## License

Code is licensed under [Apache-2.0](LICENSE). Documentation licensing still has a
human-review placeholder at [docs/LICENSE-DOCS.md](docs/LICENSE-DOCS.md), and the
recommended long-term documentation/data license is summarized in
[docs/license-recommendation.md](docs/license-recommendation.md).

## Funding and Program Boundaries

LocalAIServers is an IRS-recognized 501(c)(3) public charity. Charitable donations
support public documentation, open-source maintenance, QC methodology, benchmark
reporting, and community AI research access. Program payments for hardware verification,
pass-through costs, or fulfillment are separate from donations.

Donations do not provide hardware, preferential treatment, discounts, procurement
access, or private benefits.

See [docs/funding.md](docs/funding.md) and
[docs/program-boundaries.md](docs/program-boundaries.md).

## Existing Qwen3.6 Reproducibility Instructions

The canonical deployment package is [qwen36-gfx906/README.md](qwen36-gfx906/README.md).
The root README preserves the existing quick-start instructions below for continuity.

The current public runtime target is:

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
USE_PREBUILT_IMAGE=1 \
PREBUILT_IMAGE_PULL=1 \
AUTO_STAGE_MODEL=1 \
./deploy.sh
```

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

The currently published Docker Hub image is the 29-layer runtime produced by clean
source rebuilds on two independent GFX906 hosts. Both rebuilds produced the same source
archive SHA-256: `aa34cb675f83ff6cade31cbbb357b1c31d793bee18da491f501d7c39fda3612a`. The
public Docker Hub manifest digest is
`sha256:f5e69ee127b766960e386e0e4eda8e26c399bd02f57c494847cb9a92ce04d8ac`, and the
registry config digest matches the tested local image ID:
`sha256:e45309183e6f35cae6fb8f9d8d6f016253f281a5e7187e1f11a57e5e28ef5e86`.

After the vLLM service is ready:

```bash
python3 ./run_qwen36_live_tps.py
```

Reference fixed-token single-request results on the validated 4x MI50 32GB lane:

```text
c1_2000:  101.47 TPS backend decode
c1_10000:  95.66 TPS backend decode
c1_10000:  95.36 client wall TPS
```

See [qwen36-gfx906/README.md](qwen36-gfx906/README.md) for the full deployment and
reproduction notes, including build pins, Docker archive hashes, runtime defaults, disk
checks, and limitations.
