# GFX906 Current Research Roadmap

## Primary Dense Direction

Dense 27B remains focused on RowParallel / collective-boundary work. The sanitized
source inventory and key-learning record point toward RCCL/Tree/LL source work,
lower-level sidecar/RCCL boundary changes, graph-native RowParallel boundary reduction,
and collective-count or launch-count reduction.

Current dense language should remain cautious: the dense gate remains open, and source
milestones should not be described as gate clears.

## MoE Direction

Protect and document the cleared MoE 35B TP4 publication path. Future MoE work should
test source changes that reduce communication count or improve fastpath behavior without
changing model semantics.

## Prefill/Concurrency Direction

Prefill and concurrency work should remain separate from single-request c1 decode. Each
category needs its own metric scope, correctness gate, and promotion language. Multi-row
or prefill/concurrency milestones should not be collapsed into single-request decode
claims.

## Current Source Directions

- RCCL/Tree/LL source work remains a primary dense direction.
- Consumer-only cleanup is useful component evidence, but the current record treats it
  as insufficient for dense gate clear by itself.
- Grouped or coalesced collectives are useful lower-bound evidence, but not legal in the
  current serving graph unless semantics change or the graph boundary is restructured.
- Sidecar/RCCL work should focus on the hot RowParallel boundary, collective primitive,
  launch count, and tail behavior rather than simple descriptor-only retuning.
- Rejected paths should remain documented when they narrow the search space.

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

- Publish source-kernel inventory.
- Publish key learnings.
- Publish benchmark methodology.
- Publish QC methodology.
- Formalize benchmark artifacts.
- Maintain reproducible Docker/runtime docs.
- Continue source-level collective/boundary work.
