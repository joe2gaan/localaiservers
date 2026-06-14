# QC Methodology

This is a draft public QC methodology for local AI server hardware. It is not a private
test fixture and does not include sourcing, payment, or supplier details.

## Purpose

QC reduces fraud, reduces avoidable failures, and helps the public evaluate used AI
hardware for reproducible local AI systems.

## Scope

The current public focus is GPU-based local AI inference infrastructure, with particular
attention to GFX906 / MI50-class systems.

## Hardware Classes

QC notes should identify the tested hardware class, GPU count, measured VRAM, host
platform constraints, and relevant cooling or power assumptions. Product names alone are
not sufficient evidence.

## Intake Checks

- Record device enumeration.
- Record measured VRAM in bytes where possible.
- Confirm that the system can initialize the expected runtime stack.
- Record obvious mismatch, damage, or missing-device conditions.

## Visual And Physical Inspection

Public reporting may describe inspection categories, but should not publish private
sourcing or fulfillment details. Useful categories include visible damage, cooling
condition, connector condition, and whether the device can be seated and powered safely.

## VRAM Detection

VRAM should be detected by measured bytes rather than product-name strings. This helps
catch mislabeled, damaged, or partially enumerated hardware.

## ROCm Detection

The validation path should confirm that ROCm-visible devices initialize, report expected
architecture, and remain visible during basic runtime checks.

## Thermal And Cooling Awareness

Thermal notes should identify whether the system can sustain validation runs without
obvious throttling or unsafe temperatures. Public reports should avoid private site
details.

## Stress And Validation Runs

Validation should include device visibility, basic memory checks where available,
multi-GPU sanity checks, and at least one workload that exercises the intended AI
runtime path.

## Benchmark Reproducibility

Benchmark reports should include model, runtime image or build, command, metrics,
context length, backend decode TPS, client wall TPS when relevant, and redacted
evidence.

## Pass/Fail Categories

- Pass: hardware and runtime meet the documented validation path.
- Conditional pass: usable with documented limitations.
- Needs retest: evidence is incomplete or the run was interrupted.
- Fail: hardware or runtime failed a required validation gate.
- Diagnostic only: evidence is useful but not sufficient for pass/fail.

## Public Reporting

Public QC reports should publish methods, categories, and redacted evidence. They should
not publish private participant data, payment data, supplier details, or private network
information.

## Limitations

QC cannot provide a warranty, guarantee future compatibility, or prove all possible
workloads. It can reduce avoidable risk by documenting measured evidence and known
failure classes.
