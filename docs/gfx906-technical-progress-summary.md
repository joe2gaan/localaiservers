# GFX906 Technical Progress Summary

## Summary

LocalAIServers performs source-level GFX906 preservation work across kernel, runtime,
graph, and collective communication paths. The public repository now includes a
sanitized source inventory and key-learning record showing that the work is not only a
launch/config sweep or benchmark post. It documents MoE fastpath work, dense
RowParallel/RCCL collective-boundary investigation, graph-runtime integration,
rejected paths, source milestones, and promotion methodology.

## Current Proof Points

- [qwen36-gfx906/README.md](../qwen36-gfx906/README.md) documents the canonical
  Qwen3.6-35B-A3B TP4 deployment package.
- The canonical package includes Docker image identity, Docker Hub digest, archive
  hashes, source pins, deploy commands, run commands, benchmark command, and
  limitations.
- [gfx906-source-kernel-inventory-20260612.md](gfx906-source-kernel-inventory-20260612.md)
  records source-level kernel, graph-runtime, RowParallel, RCCL/NCCL, MoE, dense,
  prefill, and diagnostic lanes.
- [gfx906-key-learnings-20260606.md](gfx906-key-learnings-20260606.md) records durable
  promotion rules, rejected paths, diagnostic limitations, and current technical
  direction.
- The source inventory reports that the MoE 35B TP4 publication gate is cleared for the
  C1 topk8 MoE fastpath stack, with the exact public runtime documented separately in
  the canonical deployment package.
- The dense 27B gate remains open. The source record describes the current dense work as
  a RowParallel / RCCL or NCCL collective-boundary problem, especially around MLP/down
  and repeated `1x5120` allreduce behavior.
- Current dense profiles are described as NCCL/RCCL dominated, with source notes
  separating source milestones, serving milestones, active lanes, and rejected paths.
- Negative results are preserved as reusable public evidence so other community
  builders do not repeat already-tested paths.

Exact benchmark numbers should be cited from the canonical deployment package,
benchmark artifact, or source inventory with run context. This summary intentionally
uses cautious language rather than promoting every source milestone as a serving result.

## Why This Matters

This is not one-off benchmarking. Source-level GFX906 preservation work helps
communities understand what still works on affordable AI research hardware, what
requires source/runtime adaptation, and which paths have already been rejected. Negative
results reduce duplicated community effort and make public benchmark claims more
trustworthy.

## Current Technical Direction

Current roadmap direction from the source inventory and key-learning record:

- Dense 27B work remains focused on RowParallel collective-boundary reduction or
  acceleration, including RCCL/Tree/LL source work and graph-native boundary changes.
- MoE work should protect and document the cleared TP4 publication path while testing
  source changes that reduce communication count without changing semantics.
- Prefill and concurrency work should remain separate from single-request c1 decode
  claims.
- Consumer-only cleanup is useful as component evidence but is not treated as a
  gate-clear path by itself.
- Grouped or coalesced collectives are useful lower-bound evidence but are not treated
  as legal serving results unless graph semantics support them.
- Promotion evidence should come from optimized serving paths with backend vLLM metric
  support, correctness validation, and clear separation between prefill, decode, and
  end-to-end measurements.
