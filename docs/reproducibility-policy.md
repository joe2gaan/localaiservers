# Reproducibility Policy

This policy keeps public benchmark claims narrow, inspectable, and useful.

## Promotion Rules

- Optimized serving paths are required for promotion claims.
- `-O=0` is diagnostic only.
- Backend decode TPS is the primary decode promotion metric.
- Client wall time is secondary.
- Separate prefill, decode, and end-to-end TPS claims.
- Microbench results are source evidence, not serving promotion by themselves.
- Source milestones and serving winners are separate categories.

## Correctness Rules

- Thinking-model correctness must be uncapped and evaluated after the closing reasoning
  section.
- Capped runs are invalid for hard correctness if the answer ends before the model
  closes its reasoning stream.

## Evidence Rules

- Near-tie results are not promoted unless repeated standard-tier evidence clears the
  incumbent band.
- Profiles are valid only when the profiling window overlaps the request and emits
  usable kernel rows.
- Negative results should be recorded when they prevent duplicated work.

## Public-Benefit Rule

Benchmark claims should help readers reproduce, compare, or avoid known failure paths. A
result that is useful only as an internal note should be labeled as diagnostic, active,
superseded, or source evidence rather than promoted.
