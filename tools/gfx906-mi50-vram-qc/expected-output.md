# Expected Output

This page shows representative output shapes for the GFX906 / MI50 32GB VRAM QC
field check. Exact device names, card numbers, temperatures, and ROCm output vary
by host.

## Passing One-Device Run

```text
GFX906 / MI50 32GB VRAM QC Field Check

Purpose: public education and best-effort hardware verification methodology.
Safety: allocates/checks transient GPU memory through HIP only.
This is not certification, warranty testing, resale support, or official validation.

== Tool configuration ==
hipcc: /opt/rocm/bin/hipcc
target allocation: 30 GiB
device selection: 0

== Kernel-reported VRAM totals ==
/sys/class/drm/card0: 34342961152 bytes (31.98 GiB)

== rocm-smi visibility ==
======================= ROCm System Management Interface =======================
GPU[0]          : Card series: AMD Instinct MI50
GPU[0]          : VRAM Total Memory (B): 34342961152
===============================================================================

== Build ==
build directory: /tmp/mi50-vram-qc.ABC123

== HIP VRAM allocation/check ==
HIP devices visible: 1
Selected devices: 0

Device 0:
  name: AMD Instinct MI50
  totalGlobalMem: 34342961152 bytes
  hipMemGetInfo free: 33800000000 bytes
  hipMemGetInfo total: 34342961152 bytes
  target allocation: 30.00 GiB
  checked words: 8053063680
  result: PASS

== Final result ==
PASS: selected device checks passed.
```

## Passing Multi-Device Run

```text
GFX906 / MI50 32GB VRAM QC Field Check

Purpose: public education and best-effort hardware verification methodology.
Safety: allocates/checks transient GPU memory through HIP only.
This is not certification, warranty testing, resale support, or official validation.

== Tool configuration ==
hipcc: /opt/rocm/bin/hipcc
target allocation: 30 GiB
device selection: all

== Kernel-reported VRAM totals ==
/sys/class/drm/card0: 34342961152 bytes (31.98 GiB)
/sys/class/drm/card1: 34342961152 bytes (31.98 GiB)

== HIP VRAM allocation/check ==
HIP devices visible: 2
Selected devices: 0 1

Device 0:
  name: AMD Instinct MI50
  totalGlobalMem: 34342961152 bytes
  hipMemGetInfo free: 33800000000 bytes
  hipMemGetInfo total: 34342961152 bytes
  target allocation: 30.00 GiB
  checked words: 8053063680
  result: PASS

Device 1:
  name: AMD Instinct MI50
  totalGlobalMem: 34342961152 bytes
  hipMemGetInfo free: 33800000000 bytes
  hipMemGetInfo total: 34342961152 bytes
  target allocation: 30.00 GiB
  checked words: 8053063680
  result: PASS

== Final result ==
PASS: selected device checks passed.
```

## Allocation Failure

```text
Device 0:
  name: AMD Instinct MI50
  totalGlobalMem: 34342961152 bytes
  hipMemGetInfo free: 17000000000 bytes
  hipMemGetInfo total: 34342961152 bytes
  target allocation: 30.00 GiB
  result: FAIL
  error: hipMalloc(32212254720 bytes) failed: hipErrorOutOfMemory: out of memory

== Final result ==
FAIL: one or more selected device checks failed.
```

## Missing hipcc

```text
error: hipcc was not found.

Install ROCm HIP development tools for your distribution, or set HIPCC:
  HIPCC=/opt/rocm/bin/hipcc ./mi50_vram_qc.sh

On apt-based ROCm systems, you may opt in to a best-effort install attempt:
  ./mi50_vram_qc.sh --install-deps
```

## Missing rocm-smi But Check Still Runs

```text
== rocm-smi visibility ==
rocm-smi not available; skipping.

== Build ==
build directory: /tmp/mi50-vram-qc.DEF456

== HIP VRAM allocation/check ==
HIP devices visible: 1
Selected devices: 0

Device 0:
  name: AMD Instinct MI50
  totalGlobalMem: 34342961152 bytes
  hipMemGetInfo free: 33800000000 bytes
  hipMemGetInfo total: 34342961152 bytes
  target allocation: 30.00 GiB
  checked words: 8053063680
  result: PASS

== Final result ==
PASS: selected device checks passed.
```

## Sanitized `.20` Validation Example

This example was captured from the maintained repository tool on the `.20` GFX906
server and sanitized before publication. It is an educational example only. It is not
certification, warranty evidence, official AMD validation, procurement support, resale
support, or a guarantee of AI workload performance.

Device 0 30 GiB excerpt:

```text
HIP devices visible: 8
Selected devices: 0

Device 0:
  name: AMD Instinct MI50/MI60
  totalGlobalMem: 34342961152 bytes
  target allocation: 30.00 GiB
  checked words: 8053063680
  result: PASS

== Final result ==
PASS: selected device checks passed.
```

All-device 30 GiB excerpt:

```text
HIP devices visible: 8
Selected devices: 0 1 2 3 4 5 6 7

Device 0:
  target allocation: 30.00 GiB
  checked words: 8053063680
  result: PASS

Device 7:
  target allocation: 30.00 GiB
  checked words: 8053063680
  result: PASS

== Final result ==
PASS: selected device checks passed.
```

## Interpreting Results

`PASS` means the selected HIP device could allocate the requested transient VRAM
region and pass a deterministic fill/check pass during this run.

`FAIL` means the selected device did not complete this limited diagnostic. Common
causes include insufficient free VRAM, missing ROCm/HIP setup, driver instability,
device reset, or memory errors.

This output is educational diagnostic evidence only. It is not certification,
warranty evidence, official AMD validation, a purchase recommendation, or proof of
AI workload performance.
