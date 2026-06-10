# LocalAIServers

LocalAIServers is an IRS-recognized 501(c)(3) public charity focused on public education
around locally hosted AI servers. This repository maintains open documentation,
reproducible local AI deployment notes, benchmark reporting, and hardware verification
methodology so students, educators, independent researchers, and community builders can
access practical AI infrastructure without enterprise budgets.

## Current Technical Focus

The current public main-branch work centers on reproducible local AI deployments for
GFX906 / AMD Instinct MI50-class hardware. The first technical target is a
Qwen3.6-35B-A3B runtime profile using vLLM on 4x AMD Instinct MI50 32GB GPUs, with
published Docker image identity, rebuild notes, benchmark reporting, and deployment
scripts.

This is community documentation and open-source maintenance work. It is not official
ROCm support, AMD endorsement, or a claim that every GFX906 system will reproduce the
same behavior without local validation.

## What Is In This Repo

- `qwen36-gfx906/`: deployment script, smoke test, runtime notes, manifest, Docker Hub
  publishing helper, and benchmark harness for the current GFX906 runtime work.
- `qwen36-gfx906/media/`: public benchmark video preview and GitHub Pages media page.
- `docs/`: public-benefit, funding, program-boundary, wording-audit, GitHub settings,
  and risk-review documentation.
- `grants/`: grant-facing overview and concept note material based on committed repo
  evidence.
- `communications/`: copy/paste-ready public update templates.
- `issues/`: contributor backlog items that can be copied into GitHub issues.
- `.github/`: funding example, issue templates, and pull request template.

## Quick Links

- [Qwen3.6 GFX906 reproduction notes](qwen36-gfx906/README.md)
- [Live TPS video page](https://joe2gaan.github.io/localaiservers/qwen36-gfx906/media/)
- [Impact overview](docs/impact.md)
- [Funding and donation language](docs/funding.md)
- [Program boundaries](docs/program-boundaries.md)
- [Main-branch wording audit](docs/main-branch-wording-audit.md)
- [Public risk review](docs/main-public-risk-review.md)
- [Grant concept note](grants/concept-note-gfx906-community-ai-access.md)
- [Contributor backlog](issues/README.md)

## Current Reproducible Runtime Work

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

## GFX906 / MI50 Documentation Focus

GFX906 / MI50-class hardware remains useful for learning, experimentation, and community
AI research access when deployment details are documented carefully. This repo focuses
on practical notes for local AI server operation: runtime image identity, source rebuild
controls, cache and Docker-root behavior, GPU memory gates, smoke tests, and repeatable
benchmark reporting.

The documentation is intentionally evidence-based. Where a claim depends on a specific
committed script, image digest, command, or benchmark output, the repo should point to
that source. Planned work should be labeled as planned or future direction.

## Hardware Verification and QC Methodology

LocalAIServers documents hardware verification standards as a public education activity.
Current and planned methodology includes:

- GPU identity and VRAM checks before expensive builds or model launches.
- Free-VRAM checks to avoid confusing occupied-device failures with runtime failures.
- Reproducibility checks based on Docker archive SHA-256 values and published image
  digests.
- Smoke tests that validate container health and model endpoint availability.
- Redaction expectations for logs and screenshots before public sharing.

This repository does not publish private participant records, payment records, vendor
quotes, home addresses, legal filing details, or operational fulfillment data.

## Benchmark Reporting

Benchmark reporting in this repo should be reproducible, narrow, and clearly scoped.
Reports should include the model, hardware, runtime image or build, command used,
context length, prompt/decode performance, VRAM usage, temperature or power notes when
available, and redacted logs or screenshots.

The current published TPS figures are fixed-token, single-request reference results for
the documented runtime lane. They are not general claims about all prompts, all
workloads, all GFX906 systems, or official vendor-supported performance.

## Public Benefit / Why This Matters

Modern AI infrastructure often assumes access to enterprise budgets, hosted services, or
current-generation accelerators. Public documentation for locally hosted AI servers
helps communities learn how AI systems are deployed, measured, maintained, and verified
on accessible hardware.

LocalAIServers supports public education, open-source maintenance, reproducible local AI
deployments, hardware verification standards, benchmark reporting, affordable AI
research access, and community AI research access.

## Who Benefits

- Students and educators learning how local AI servers are assembled and operated.
- Independent researchers who need reproducible runtime notes without enterprise
  infrastructure.
- Community builders who maintain local compute for workshops, labs, or public-interest
  projects.
- Open-source contributors improving deployment documentation, issue templates, smoke
  tests, and benchmark reporting.
- Donors and grant reviewers evaluating a public-benefit AI infrastructure project with
  clear boundaries.

## Contributing

Contributions are welcome when they improve public documentation, reproducibility,
benchmark clarity, repo hygiene, or hardware verification methodology. Start with
[CONTRIBUTING.md](CONTRIBUTING.md), review the [Code of Conduct](CODE_OF_CONDUCT.md),
and use the issue templates when reporting bugs, documentation problems, or benchmark
results.

Before opening an issue or pull request, redact secrets, tokens, private paths, private
hostnames, payment data, participant data, vendor data, and any other non-public
operational details.

## Funding and Donations

LocalAIServers is an IRS-recognized 501(c)(3) public charity. Donations support
open-source maintenance, documentation, QC methodology, benchmark reporting, and
community AI research access. Program payments for hardware verification, pass-through
costs, or fulfillment are separate from charitable donations.

Donation language and the placeholder donation URL are maintained in
[docs/funding.md](docs/funding.md). This repository intentionally does not include real
payment links, routing details, account details, or filing documents.

## Program Boundaries

- Donations do not provide hardware.
- Donations do not provide priority access.
- Donations do not provide discounts.
- Donations do not provide private benefits.
- Hardware verification/program payments, if any, are separate from charitable
  donations.

See [docs/program-boundaries.md](docs/program-boundaries.md) for the longer boundary
statement.

## Safety and Limitations

- This project documents community runtime work; it does not claim official ROCm support
  or AMD endorsement.
- Model weights are not included in the published runtime image and may require separate
  access under the model provider's terms.
- ROCm, vLLM, Docker, model downloads, and native builds can be large, slow, and
  hardware-sensitive.
- Benchmark results should be treated as scoped reference measurements, not universal
  guarantees.
- Do not post secrets, private logs, participant/payment/vendor data, personal
  addresses, or legal filing details in issues or pull requests.
