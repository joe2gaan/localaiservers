# Methodology

## Environment

The canonical environment is documented in
[qwen36-gfx906/README.md](../../qwen36-gfx906/README.md). This benchmark summary does
not replace that deployment package.

## Model

`Qwen/Qwen3.6-35B-A3B`

## Hardware

4x AMD Instinct MI50 32GB on GFX906, using tensor parallel size 4.

## Runtime

The public runtime artifact is:

```text
joe2gaan/localaiservers:qwen36-gfx906-c1-topk8-runtime-archive-aa34cb675f83
```

The Docker Hub digest is:

```text
sha256:f5e69ee127b766960e386e0e4eda8e26c399bd02f57c494847cb9a92ce04d8ac
```

## Metrics

- Backend decode TPS is the primary decode metric for promotion language.
- Client wall TPS is secondary and should be reported separately.
- Benchmark claims should identify context, command, runtime, and metric source.

## Backend Decode TPS Vs Client Wall TPS

Backend decode TPS describes the serving backend measurement. Client wall TPS includes
client-side timing and should not be mixed into backend decode claims.

## Correctness

Correctness should be evaluated under the same policy used by the project: optimized
serving-path validation, metric evidence, and task-appropriate output review.
Diagnostic-only paths are not promotion evidence.

## Tiers

- Serving milestone: a useful serving-path result with scoped evidence.
- Publication gate: enough evidence to publish a reproducible public artifact.
- Source milestone: useful source-level evidence that is not a serving result by itself.
- Diagnostic only: useful for debugging, not for promotion claims.
