# GFX906 Current Research Roadmap

## Primary Dense Direction

Dense 27B remains focused on RowParallel collective-boundary work. Detailed claim
language should remain cautious until the sanitized source inventory and key-learning
files are added to the public repository.

## MoE Direction

Protect the MoE 35B TP4 publication artifact and test source changes that reduce
communication count without changing semantics.

## Prefill/Concurrency Direction

Prefill and concurrency work should remain separate from single-request decode. Each
category needs its own metric scope, correctness gate, and promotion language.

## Near-Term Milestones

- Publish source-kernel inventory.
- Publish key learnings.
- Publish benchmark methodology.
- Publish QC methodology.
- Formalize benchmark artifacts.
- Maintain reproducible Docker/runtime docs.
- Continue source-level collective/boundary work.
