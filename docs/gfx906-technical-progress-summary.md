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
  2026-06-20 ROCm7.2 Dense/MoE runner on current main.
- The canonical package includes Docker image identity, Docker Hub digest,
  archive hashes, source pins, deploy commands, run commands, benchmark command,
  active profile names, and limitations.
- v0.1.0 remains the older published GitHub Release boundary. The dense 27B and
  MoE values below are post-v0.1 main-branch validation notes until a separate
  release is published. GitHub Releases remain canonical for published claim
  boundaries.
- [gfx906-source-kernel-inventory-20260612.md](gfx906-source-kernel-inventory-20260612.md)
  records source-level kernel, graph-runtime, RowParallel, RCCL/NCCL, MoE, dense,
  prefill, and diagnostic lanes.
- [gfx906-key-learnings-20260606.md](gfx906-key-learnings-20260606.md) records durable
  promotion rules, rejected paths, diagnostic limitations, and current technical
  direction.
- The same ROCm7.2 experimental release image covers both active contracts with
  model-specific env and overlays:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`,
  Docker Hub manifest digest
  `sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e`.
  Docker Hub remains an evergreen artifact distribution channel; TPS claims
  should stay in GitHub Releases, repository docs, and benchmark artifacts.
- Dense 27B TP8 clears the ai-info 10K gate in post-v0.1 validation at
  `MAX_MODEL_LEN=131072` with eight pre-measure warmups: strict backend TPS
  `69.514`, `c1_2000` backend TPS `70.347`, and `c1_10000` backend TPS
  `66.069`; note `strict gate valid`.
- Qwen3.6 35B-A3B MoE TP8 establishes the current full-BAR/P2P-on bar at
  `MAX_MODEL_LEN=131072` with eight pre-measure warmups: strict backend TPS
  `94.907`, `c1_2000` backend TPS `97.028`, and `c1_10000` backend TPS
  `91.290`; note `strict gate valid`.
- Qwen3.6 35B-A3B MoE TP4 remains valuable as a capped warm-performance lane:
  strict is `invalid/runaway`, `c1_2000` backend TPS is `116.146`, and
  `c1_10000` backend TPS is `109.283`; note `uncapped strict prompt did not
  stop after >60K tokens`.
- Platform remediation for the current full-BAR/P2P-on lane required official
  AMD VBIOS standardization, not modified BIOS images, plus amdgpu source
  patching. This is not a user instruction to flash cards; the public repo does
  not redistribute BIOS binaries or imply warranty or certification.
- Earlier singlework/LL16/prims-inline RCCL, RowParallel, sidecar Tree/LL, and
  call-site routing work remains important source evidence, but those entries
  are superseded for current dense 27B status by the 2026-06-20 ROCm7.2
  Dense/MoE runner.
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

- Dense 27B work should now protect the post-v0.1 gate-clear path, keep the
  `MAX_MODEL_LEN=131072` serving contract intact, and continue source cleanup
  around RowParallel, sidecar Tree/LL, RCCL/Tree/LL, call-site routing, and
  graph-native boundary changes.
- MoE work should protect the strict-valid TP8 full-BAR/P2P-on bar and preserve
  the corrected post-v0.2 TP4 repeatability evidence: the release-time TP4
  fixed-token result remains `109.283` c1_10000 backend TPS, and a follow-up
  study found that the earlier TP4 strict runaway did not reproduce across
  `6/6` strict repeats.
- Prefill and concurrency work should remain separate from single-request c1 decode
  claims.
- Consumer-only cleanup is useful as component evidence but is not treated as a
  release-promotion path by itself.
- Grouped or coalesced collectives are useful lower-bound evidence but are not treated
  as legal serving results unless graph semantics support them.
- Promotion evidence should come from optimized serving paths with backend vLLM metric
  support, correctness validation, and clear separation between prefill, decode, and
  end-to-end measurements.
