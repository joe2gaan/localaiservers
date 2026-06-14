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

## Reference Results

The canonical technical README reports:

```text
c1_2000:  101.47 TPS backend decode
c1_10000:  95.66 TPS backend decode
c1_10000:  95.36 client wall TPS
```

These are fixed-token single-request reference results for the documented runtime lane.
They are not general claims about all prompts, all workloads, all GFX906 systems, or
officially supported performance.

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
