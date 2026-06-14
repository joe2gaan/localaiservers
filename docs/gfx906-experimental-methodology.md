# GFX906 Experimental Methodology

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
