# GFX906 Technical Progress Summary

## Summary

LocalAIServers performs source-level GFX906 preservation work across kernel, runtime,
graph, and collective communication paths. The public repository currently contains a
reproducible Qwen3.6-35B-A3B TP4 deployment package for 4x AMD Instinct MI50 32GB
hardware, while the dedicated source inventory and key learning files still need to be
imported before more detailed source claims can be made from repository evidence.

## Current Proof Points

Current proof points supported by committed public files:

- [qwen36-gfx906/README.md](../qwen36-gfx906/README.md) documents the canonical
  Qwen3.6-35B-A3B TP4 deployment package.
- The canonical package includes Docker image identity, Docker Hub digest, archive
  hashes, source pins, deploy commands, run commands, benchmark command, and
  limitations.
- The public benchmark lane reports fixed-token backend decode results for `c1_2000` and
  `c1_10000`, plus client wall TPS for `c1_10000`.
- LocalAIServers records benchmark scope, reproducibility evidence, and limitations
  instead of treating every experiment as a promoted serving result.

Source-specific proof points that need the missing source inventory/key-learning files
before publication:

- Detailed dense 27B gate status.
- Detailed RowParallel / RCCL or NCCL collective-boundary findings.
- Detailed source-level kernel inventories.
- Detailed rejected-path and active-lane records.

## Why This Matters

This is not one-off benchmarking. Source-level GFX906 preservation work helps
communities understand what still works on affordable AI research hardware, what
requires source/runtime adaptation, and which paths have already been rejected. Negative
results reduce duplicated community effort and make public benchmark claims more
trustworthy.

## Current Technical Direction

Current roadmap direction, pending source inventory publication:

- Structural RowParallel collective-boundary work for dense paths.
- Lower-latency GFX906 small-message collective investigation.
- Legal reduction-count changes where semantics are preserved.
- Producer/collective/consumer fusion below graph overhead where source evidence
  supports it.
- Prefill and concurrency work tracked separately from single-request decode.

Detailed claims should be upgraded only after sanitized source inventory and key
learning files are added to this repository.
