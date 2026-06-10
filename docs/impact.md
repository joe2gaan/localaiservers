# Impact

LocalAIServers is an IRS-recognized 501(c)(3) public charity focused on public education
around locally hosted AI servers. The public work in this repository helps people learn
how local AI infrastructure is deployed, verified, benchmarked, and maintained without
requiring enterprise budgets.

## Public Education

The repository documents practical local AI server work in a form that students,
educators, independent researchers, and community builders can inspect and reproduce.
Public notes are preferred over private operational knowledge because they make
deployment decisions, hardware constraints, and benchmark limitations easier to review.

## Reproducible Local AI Deployment Notes

Current repo evidence includes a GFX906 runtime bundle for `Qwen/Qwen3.6-35B-A3B` on 4x
AMD Instinct MI50 32GB hardware, with deployment commands, image identity, Docker
archive hashes, runtime defaults, and smoke-test guidance in
[qwen36-gfx906/README.md](../qwen36-gfx906/README.md).

## Affordable AI Research Access

Affordable AI research access means lowering the documentation and maintenance barrier
for community-scale infrastructure. This repository focuses on reproducible local AI
deployments and benchmark reporting so users can evaluate practical AI server work
without relying only on hosted enterprise platforms.

## GFX906 / MI50 Documentation

GFX906 / MI50-class hardware can be difficult to operate for modern AI workloads because
the practical path depends on exact runtime versions, build pins, model settings, memory
behavior, and local validation. The repo documents community runtime work for this
hardware class without claiming official ROCm support or AMD endorsement.

## Benchmark Reporting

Benchmark notes should be specific, reproducible, and scoped. Reports should identify
the model, hardware, runtime image or build, command used, context length, prompt/decode
performance, VRAM usage, temperature or power notes when available, and redacted logs or
screenshots.

## Community Learning

The project supports community learning by keeping reproducible notes, issue templates,
contribution guidance, and public risk boundaries in the open. The goal is to make local
AI infrastructure easier to understand, inspect, and improve.

## Hardware Verification Standards

Hardware verification standards include GPU identity checks, VRAM and free-VRAM checks,
smoke tests, reproducibility hashes, and clear redaction rules for logs. These standards
help separate hardware issues, runtime issues, and documentation gaps.

## Open-Source Maintenance

Donations and grants can support maintenance of deployment notes, issue triage,
documentation cleanup, benchmark templates, QC methodology, and public education
material. Maintenance work should stay public-benefit focused and avoid private
operational data.
