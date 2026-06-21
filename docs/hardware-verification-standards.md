# Hardware Verification Standards

This is the current draft public standard for hardware verification in LocalAIServers
documentation. It is subject to update as more hardware is tested.

## Why Used AI Hardware Verification Matters

Used AI hardware can be mislabeled, partially functional, thermally constrained, or
difficult to initialize in modern software stacks. Public verification standards help
communities reduce avoidable failures and compare evidence.

## GFX906 / MI50-Specific Considerations

GFX906 / MI50-class systems require careful attention to ROCm visibility, measured VRAM,
multi-GPU communication, software image identity, and workload scope. A device name
alone is not enough to establish readiness for a given AI runtime.

## Measured VRAM Over Product Strings

Verification should prefer measured VRAM in bytes over product-name strings. This helps
identify mismatches and incomplete enumeration.

## Device Enumeration And Topology Caveats

Reports should include device count, visible GPU identifiers, and any topology
constraints that affect multi-GPU workloads. Private hostnames, network details, and
local paths should be redacted.

## ROCm Initialization Requirements

Reports should confirm that ROCm initializes the expected devices and that the runtime
can see the intended architecture.

## Multi-GPU Communication Sanity Checks

Multi-GPU validation should include at least one communication-sensitive runtime or
benchmark check when the target workload depends on tensor parallelism or collective
communication.

## Reproducible Benchmark Checks

Benchmark evidence should identify the runtime image or build, command, model, hardware
target, metric source, and workload scope.

## Known Failure Classes

- Device not visible to ROCm.
- VRAM mismatch or incomplete memory reporting.
- Runtime image mismatch.
- Thermal or cooling instability.
- Communication-sensitive workload failure.
- Benchmark evidence without backend metric support.

## GFX906 / MI50 32GB VRAM QC Field Check

LocalAIServers maintains a public field-check tool for MI50 32GB / GFX906-class
cards:

[tools/gfx906-mi50-vram-qc](../tools/gfx906-mi50-vram-qc/)

The tool is educational QC methodology. It is not a certification, warranty, formal
assurance program, support contract, buying/selling workflow, or official AMD
validation.

## Reporting Format

Reports should include hardware class, measured VRAM, runtime stack, command, metric
source, pass/fail category, and redacted supporting evidence.
