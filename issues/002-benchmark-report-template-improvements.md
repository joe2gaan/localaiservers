# Title

Improve benchmark report template and examples

## Context

The repo includes a benchmark issue template that asks for model, hardware, runtime
image/build, command, context length, prompt/decode performance, VRAM usage,
temperature/power notes, and redacted logs/screenshots. Additional wording or example
structure could make reports easier to compare.

## Desired Outcome

Improve the benchmark report template or add a short documentation page that explains
how to submit scoped, reproducible benchmark reports.

## Acceptance Criteria

- Keeps benchmark claims scoped to the documented model, hardware, runtime, command, and
  workload.
- Requests prompt/decode performance without implying unreleased benchmark tooling.
- Includes redaction guidance for logs and screenshots.
- Avoids generalized performance claims.
- Links to the existing benchmark issue template.

## Suggested Labels

`documentation`, `benchmark`, `reproducibility`

## Difficulty

Intermediate

## Privacy Reminder

Logs and screenshots must be redacted before sharing. Remove credential material,
private paths, private host details, participant data, payment data, and supplier data.
