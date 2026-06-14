# GFX906 Current Research Roadmap

## Dense 27B Status

Dense 27B gate remains open.

The current documented source high-water is the graph-safe non-resident sidecar Tree/LL
allreduce runtime plus exact public-RCCL fallback for captured attention
`call_site=1664`, with the `2048/1536/1536` sidecar descriptor split. The source
inventory describes this as a serving milestone and current source high-water, not a
dense gate clear.

The remaining gap is framed as RowParallel / RCCL / NCCL collective-boundary work,
especially around repeated small allreduce behavior, launch/count cost, tail behavior,
and graph-runtime integration.

## MoE 35B Status

MoE 35B TP4 publication gate is cleared.

The current public source record identifies the C1 topk8 MoE fastpath as the cleared
publication path and records corrected hard-thinking validation. The source inventory
reports `95.661` backend TPS on `c1_10000` for that validation context. Cite this with
the source context and the canonical Qwen3.6 deployment package rather than generalizing
it to unrelated workloads.

## Primary Dense Direction

- RowParallel collective-boundary work.
- Non-resident sidecar / public-RCCL fallback / per-node primitive selection where it
  preserves graph semantics.
- Tree/LL and RCCL source work for small-message collective behavior.
- Collective launch-count or reduction-count changes that preserve correctness.
- Legal coalescing or graph-native RowParallel boundary changes.
- Producer/collective/consumer fusion below graph-op overhead where source evidence
  supports it.
- Exact small-message collective improvements that survive optimized serving-path
  validation.

## Secondary / Side Lanes

- Prefill and concurrency milestones should remain separate from single-request c1
  decode claims.
- LL128 and wider-row bandwidth work are side lanes unless they improve the hot dense
  single-request path.
- Reduce-scatter or sequence-parallel work should be treated as prefill/concurrency
  evidence unless serving decode evidence changes that status.
- MoE side lanes should remain separate from dense 27B claims.

## Closed Or Deprioritized Paths

These paths remain useful public learning, but the current source record does not treat
them as the active dense gate-clear path:

- Consumer-only RMS/logits cleanup as a standalone gate path.
- Public send/recv composition.
- Raw peer-memory or HIP IPC collectives on the current GFX906 hosts.
- AITER or symmetric-memory substrate work on the current GFX906 hosts.
- Simple NCCL/RCCL environment knob sweeps.
- TP4 dense serving as the primary dense path.
- Model-forward split endpoint wrappers.
- Chunked RowParallel overlap that increases allreduce launch count.
- Direct median descriptor wins without serving promotion.
- Earlier singlework/LL16 branches as the global current winner when superseded by the
  latest sidecar/public1664 source inventory.

## Public Outputs

The roadmap should produce public-benefit outputs rather than private access claims:

- Benchmark documentation.
- Sanitized source inventory.
- Sanitized key learnings.
- QC methodology.
- Hardware verification standards.
- Reproducibility documentation.
- Canonical deployment notes.

## Near-Term Milestones

- Maintain the public source-kernel inventory.
- Maintain the public key-learnings record.
- Publish benchmark methodology updates.
- Publish QC methodology updates.
- Formalize benchmark artifacts.
- Maintain reproducible Docker/runtime docs.
- Continue source-level collective/boundary work.
