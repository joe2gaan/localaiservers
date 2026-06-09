# LocalAIServers GFX906 Documentation & Community AI Research Access Initiative

## Problem

Students, educators, independent researchers, and community builders often lack practical documentation for locally hosted AI servers. Modern AI infrastructure guidance frequently assumes enterprise budgets, hosted services, or current-generation accelerators. GFX906 / AMD Instinct MI50-class hardware can support learning and community AI research access, but practical deployment requires careful runtime notes, hardware verification, benchmark reporting, and open-source maintenance.

## Solution

LocalAIServers will maintain public documentation, reproducible local AI deployment notes, benchmark reporting formats, and hardware verification methodology for GFX906 / MI50-class local AI servers. The initiative focuses on public education and affordable AI research access rather than private access, hardware distribution, or vendor-specific promotion.

## Current Evidence From Committed Repo

- [qwen36-gfx906/README.md](../qwen36-gfx906/README.md) documents a Qwen3.6-35B-A3B runtime profile for 4x AMD Instinct MI50 32GB hardware.
- The repo includes a published Docker Hub runtime image tag and manifest digest for the current runtime notes.
- The GFX906 README preserves source rebuild pins, Docker archive SHA-256 values, runtime defaults, smoke-test guidance, and reference TPS results.
- [qwen36-gfx906/run_qwen36_live_tps.py](../qwen36-gfx906/run_qwen36_live_tps.py) provides a benchmark harness for the documented runtime path.
- [qwen36-gfx906/smoke-test.sh](../qwen36-gfx906/smoke-test.sh) provides a basic health and model endpoint check.
- [qwen36-gfx906/media/](../qwen36-gfx906/media/) includes public benchmark media for review.

Future work should be labeled as planned or future direction unless it is evidenced by committed files.

## Target Beneficiaries

- Students and educators studying locally hosted AI infrastructure.
- Independent researchers evaluating practical local AI server deployments.
- Community builders maintaining shared research or workshop infrastructure.
- Open-source contributors improving documentation, benchmark reporting, and QC methodology.

## Core Activities

- Maintain reproducible local AI deployment notes for GFX906 / MI50-class hardware.
- Improve hardware verification standards and QC methodology.
- Expand benchmark reporting templates and review guidance.
- Maintain public documentation, issue templates, and contributor guidance.
- Publish grant- and donor-readable summaries that accurately reflect committed evidence.

## Expected Outputs

- Updated public documentation and docs index.
- Reproducibility notes tied to image tags, digests, commands, and hashes.
- Benchmark reporting templates for community submissions.
- Hardware verification methodology for GPU identity, VRAM, free-VRAM, smoke tests, and log redaction.
- Contributor backlog for documentation and benchmark improvements.

## Expected Outcomes

- Improved public understanding of locally hosted AI servers.
- More reproducible community AI infrastructure documentation.
- Better benchmark comparability for scoped local AI workloads.
- Lower documentation barriers for affordable AI research access.
- Stronger public repository credibility for donors, grant reviewers, and contributors.

## Use of Funds

Funds may support open-source maintenance, documentation work, benchmark reporting, QC methodology, public education material, community review, and project infrastructure needed to maintain public local AI server documentation.

Funds should not be described as purchasing private benefits for donors or as providing hardware, discounts, procurement access, or program placement.

## Why LocalAIServers

LocalAIServers is an IRS-recognized 501(c)(3) public charity focused on public education around locally hosted AI servers. The repo already contains committed GFX906 runtime documentation, image identity details, benchmark notes, and reproducibility material that can be expanded into a stronger public education and community research access resource.

## Donation/Program-Payment Boundary

Charitable donations support public documentation, open-source maintenance, QC methodology, benchmark reporting, and community AI research access. Program payments for hardware verification, pass-through costs, or fulfillment are separate from donations and must not be represented as charitable gifts.
