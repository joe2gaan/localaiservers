# GFX906 Experimental Methodology

This document summarizes the promotion methodology reflected in the sanitized
[GFX906 key learnings](gfx906-key-learnings-20260606.md). It is meant to make benchmark
claims more trustworthy by separating serving winners, source milestones, diagnostic
evidence, and rejected paths.

## Promotion Rules

- Promotion evidence must use optimized serving paths.
- `-O=0` is diagnostic only and is not used for winner or public-performance claims.
- Backend vLLM decode TPS is the primary decode promotion metric.
- Client wall TPS is secondary and should be reported separately.
- Thinking-model correctness must be uncapped and evaluated after the model closes its
  reasoning section.
- Capped runs are invalid for hard correctness if the answer ends before the model
  closes its reasoning stream.
- Near-ties are source milestones, not serving winners, unless repeated standard-tier
  evidence clears the incumbent band.
- Profiles require active request windows with usable kernel rows. Empty or
  non-overlapping profile captures are attribution failures, not performance evidence.
- Negative results are preserved as public learning when they prevent duplicated work.
- Source milestones and serving winners are distinct categories.
- Prefill, decode, and end-to-end TPS claims should not be mixed.
- Microbench results are source evidence, not serving promotion by themselves.

## Promotion Ladder

1. Hypothesis or source path identified.
2. Direct replay or component evidence collected.
3. Optimized serving path tested.
4. Backend metric evidence collected.
5. Correctness validated.
6. Standard decode ladder run where relevant.
7. Result categorized as promoted, rejected, source milestone, diagnostic, active lane,
   or superseded.

## Disposition Labels

- Promoted: serving-path result with sufficient metric and correctness evidence.
- Rejected: tested path that did not meet the relevant gate.
- Source milestone: source-level result that is useful but not a serving winner.
- Diagnostic only: evidence collected from a path not suitable for promotion.
- Active lane: ongoing work that has not reached a final disposition.
- Superseded: older result replaced by stronger evidence.
- Needs replay: result that requires controlled rerun before use.
- Correctness-only: correctness evidence without sufficient performance evidence.
- Performance regression: path that worsened the relevant metric.
- Serving milestone: serving-path progress that may not be the final winner.
- Publication gate: evidence threshold for publishing a reproducible public artifact.

## Negative Results

Rejected paths are preserved because they prevent duplicated effort, define the search
space, and make public benchmark claims more trustworthy. A negative result can still be
a public-benefit output when it is documented clearly and redacted safely.

Examples from the current source record include consumer-only cleanup paths, grouped
collective lower bounds, descriptor-routing variants, and diagnostic profile lanes that
were exact enough to learn from but not sufficient to clear the dense serving gate.
