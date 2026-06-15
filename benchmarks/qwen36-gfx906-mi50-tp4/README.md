# Qwen3.6 GFX906 MI50 TP4 Benchmark

Reproducible 90+ TPS sustained GFX906 local inference on Qwen3.6-35B-A3B using 4x AMD
Instinct MI50 32GB.

## Hardware Target

- 4x AMD Instinct MI50 32GB.
- GFX906.
- Tensor parallel size: 4.

## Runtime Target

- Model: `Qwen/Qwen3.6-35B-A3B`.
- Runtime: vLLM on ROCm/GFX906.
- Docker/runtime package:
  `joe2gaan/localaiservers:qwen36-gfx906-c1-topk8-runtime-archive-aa34cb675f83`.
- Docker Hub digest:
  `sha256:f5e69ee127b766960e386e0e4eda8e26c399bd02f57c494847cb9a92ce04d8ac`.

## Publication Result

This benchmark artifact should be cited as the reproducible 90+ TPS sustained 10K
publication baseline for the documented GFX906 MI50 TP4 runtime lane.

```text
c1_10000: 90+ TPS sustained backend decode publication baseline
```

The canonical technical README preserves exact run tables, commands, and reproduction
details. This benchmark summary is intentionally framed as the publication baseline,
not as a latest-result announcement or general claim about all prompts, workloads,
GFX906 systems, or officially supported performance.

LocalAIServers has reached 95+ TPS on Qwen3.6 10K decode, but that version is outside
this v0.1.0 publication release and should be cited only after it is published
separately.

## Reproduction Link

The canonical technical deployment package is
[qwen36-gfx906/README.md](../../qwen36-gfx906/README.md). Use that file for the full
Docker image, digest, archive hash, source-pin, deploy, run, benchmark, and limitation
details.

## Limitations

See [limitations.md](limitations.md).

## Public-Benefit Note

This benchmark is a public proof artifact for affordable local AI research
infrastructure. It documents a reproducible method and scoped result so others can
inspect the runtime path, compare evidence, and avoid repeating setup work.
