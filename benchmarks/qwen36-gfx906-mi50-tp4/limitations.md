# Limitations

## Hardware And Software Scope

This benchmark is scoped to the documented GFX906 / MI50 TP4 runtime lane. It is not a
general warranty for all GFX906 systems, all ROCm configurations, all prompts, or all
model-serving workloads.

## ROCm / GFX906 Constraints

GFX906 paths can be sensitive to runtime versions, source patches, kernel availability,
collective communication behavior, memory pressure, and host configuration. The
canonical README documents the exact public runtime identity and reproduction contract.

## MoE Vs Dense

This benchmark covers the public Qwen3.6-35B-A3B MoE runtime artifact. Dense model
results should not be inferred from this benchmark.

## Public Access Boundary

This benchmark is not a public cloud service and does not imply direct public machine
access. Public benefit is delivered through the published deployment method, benchmark
report, runtime details, and reproducibility workflow.

## Hardware Warranty Boundary

The benchmark is not a hardware warranty, resale support claim, or guarantee of future
compatibility. It is a scoped public reproducibility artifact.
