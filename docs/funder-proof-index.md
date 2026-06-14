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
- [Experimental methodology](gfx906-experimental-methodology.md)
- [Current GFX906 research roadmap](gfx906-current-research-roadmap.md)
- [Controlled air-gapped compute model](controlled-air-gapped-compute.md)
- [Public output model](public-output-model.md)
- [QC methodology](qc-methodology.md)
- [Hardware verification standards](hardware-verification-standards.md)
- [Roadmap](roadmap.md)

## Current Infrastructure

LocalAIServers operates controlled air-gapped GFX906 compute infrastructure with 40
active GPUs and 32 additional GFX906 GPUs staged for deployment.

This infrastructure is a verification and reproducibility testbed, not a public login
service. Public benefit is delivered through code, documentation, benchmark reports,
source-level findings, QC methods, hardware verification standards, and reproducibility
workflows.

## What The Evidence Shows

- Existing controlled GFX906 infrastructure is available for verification and
  reproducibility work.
- The canonical Qwen3.6 deployment package preserves Docker/runtime identity,
  hashes, commands, benchmark method, and limitations.
- The benchmark folder packages the Qwen3.6 / GFX906 / MI50 TP4 artifact as a stable
  proof point while linking back to the canonical deployment package.
- The source inventory and key learnings show source-level kernel/runtime maintenance,
  not only benchmark posting.
- The experimental methodology records strict promotion rules for optimized serving
  paths, backend metrics, correctness, profiles, and decode ladders.
- The technical progress summary separates MoE publication evidence from open dense
  RowParallel/RCCL research.
- The controlled compute model explains why public benefit comes from published
  outputs rather than direct machine access.
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

## What Funding Would Scale

- More public documentation.
- Benchmark automation.
- QC tooling.
- Reproducibility workflows.
- Source-level GFX906 preservation.
- Controlled infrastructure reliability.
