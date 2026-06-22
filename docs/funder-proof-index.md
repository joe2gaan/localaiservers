# Funder Proof Index

## Organization

LocalAIServers is a 501(c)(3) public charity providing public education and open-source
infrastructure for locally hosted AI systems.

## Program

GFX906 Preservation & Public AI Research Infrastructure

## Existing Proof

- [Root README](../README.md)
- [Canonical Qwen3.6 GFX906 deployment package](../qwen36-gfx906/README.md)
- [Benchmark artifact](../benchmarks/qwen36-gfx906-mi50-tp4/)
- [Source kernel inventory](gfx906-source-kernel-inventory-20260612.md)
- [Key learnings](gfx906-key-learnings-20260606.md)
- [Technical progress summary](gfx906-technical-progress-summary.md)
- [ROCm7.2 Dense/MoE active contracts](rocm72-dense-moe-active-contracts-20260620.md)
- Post-v0.2 MoE TP4 strict repeatability report:
  [`test-reports/qwen36-gfx906-moe-tp4-strict-runaway/`](../test-reports/qwen36-gfx906-moe-tp4-strict-runaway/)
- [Experimental methodology](gfx906-experimental-methodology.md)
- [Current GFX906 research roadmap](gfx906-current-research-roadmap.md)
- [Controlled air-gapped compute model](controlled-air-gapped-compute.md)
- [Public output model](public-output-model.md)
- [QC methodology](qc-methodology.md)
- [Hardware verification standards](hardware-verification-standards.md)
- [GFX906 / MI50 32GB VRAM QC field-check tool](../tools/gfx906-mi50-vram-qc/)
- [Roadmap](roadmap.md)

## Current Infrastructure

LocalAIServers operates controlled air-gapped GFX906 compute infrastructure with 40
active GPUs and 32 additional GFX906 GPUs staged for deployment.

This infrastructure is a verification and reproducibility testbed, not an
interactive-use service. Public benefit is delivered through code, documentation,
benchmark reports, source-level findings, QC methods, hardware verification standards,
and reproducibility workflows.

## What The Evidence Shows

- Existing controlled GFX906 infrastructure is available for verification and
  reproducibility work.
- The canonical Qwen3.6 deployment package preserves Docker/runtime identity,
  hashes, commands, benchmark method, and limitations.
- The benchmark folder packages the Qwen3.6 / GFX906 / MI50 TP4 artifact as a stable
  proof point while linking back to the canonical deployment package.
- The published v0.2.0 release records ROCm7.2 dense/MoE validation:
  Dense 27B TP8 clears the ai-info 10K gate at `MAX_MODEL_LEN=131072`, and
  Qwen3.6 35B-A3B MoE TP8 establishes a strict-valid full-BAR/P2P-on bar.
- The source inventory and key learnings show source-level kernel/runtime maintenance,
  not only benchmark posting.
- The experimental methodology records strict promotion rules for optimized serving
  paths, backend metrics, correctness, profiles, and decode ladders.
- The technical progress summary separates the older published v0.1.0 release boundary
  from the published v0.2.0 ROCm7.2 Dense/MoE boundary. GitHub Releases remain
  canonical for published claim boundaries.
- The same ROCm7.2 experimental release image covers dense and MoE active contracts
  with model-specific env and overlays. Platform remediation for the full-BAR/P2P-on
  lane required official AMD VBIOS standardization, not modified BIOS images, plus
  amdgpu source patching. Docker Hub remains an evergreen artifact distribution
  channel; TPS claims should stay in GitHub Releases, repository docs, and benchmark
  artifacts. This is not a user instruction to flash cards; the public repo does not
  redistribute BIOS binaries or imply warranty or certification.
- The controlled compute model explains why public benefit comes from published
  outputs rather than interactive host use.
- The QC and hardware verification docs explain how LocalAIServers turns controlled
  testing into public standards that can reduce avoidable hardware failures.

## Public-Benefit Outputs

- Open-source deployment scripts.
- Docker/runtime details.
- Reproducible benchmark reports.
- Source-level findings.
- QC and verification methods.
- Educational documentation.

## Current Review Posture

LocalAIServers already has controlled GFX906 infrastructure, public benchmark proof,
reproducible deployment artifacts, source-level GFX906 preservation work, strict
experimental methodology, QC / hardware verification methods, and a public-output model.

## What Additional Support Would Scale

- More public documentation.
- Benchmark automation.
- QC tooling.
- Reproducibility workflows.
- Source-level GFX906 preservation.
- Controlled infrastructure reliability.
