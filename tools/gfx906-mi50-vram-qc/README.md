# GFX906 / MI50 32GB VRAM QC Field Check

## Purpose

This tool is a public educational QC sanity check for AMD Instinct MI50 32GB /
GFX906-class cards. It helps users verify that the system can see the card, that
ROCm/HIP can talk to it, and that the card can allocate and correctly use a large
VRAM region.

LocalAIServers maintains this as a best-effort hardware verification methodology
tool for affordable local AI learning infrastructure. It is intended for public
education, reproducibility support, and hands-on local AI learning.

## What it checks

- Vega20 / GFX906-style evidence from firmware, kernel, and ROCm where available.
- Kernel-reported VRAM totals from `/sys/class/drm/card*/device/mem_info_vram_total`.
- HIP visibility and device enumeration.
- Large VRAM allocation and deterministic memory check, default `30 GiB`.
- `rocm-smi` visibility when available.

## What it does not prove

This tool is not:

- a certification or warranty
- a formal assurance program
- a support contract
- a complete memory burn-in
- a thermal stress test
- an official AMD validation
- a guarantee of AI workload performance
- a hardware purchase recommendation
- a buying or selling workflow

Passing this check means the selected device passed this limited HIP allocation
and pattern-check run on the current host configuration. It does not prove that
the card is free of intermittent faults or suitable for every workload.

## Safety

The tool only allocates and verifies transient GPU memory through HIP. It does not
flash firmware, modify BIOS/VBIOS, alter PCIe configuration, change driver source,
write to block devices, upload logs, or collect personal data.

The check may still crash or expose instability on faulty hardware, so users should
run it only on systems they control.

## Quick start

```bash
cd tools/gfx906-mi50-vram-qc
./mi50_vram_qc.sh --target-gib 30 --device all
```

If dependencies are missing:

```bash
./mi50_vram_qc.sh --install-deps
```

Run one HIP device only:

```bash
./mi50_vram_qc.sh --target-gib 30 --device 0
```

Skip optional evidence collection:

```bash
./mi50_vram_qc.sh --skip-dmesg --skip-rocm-smi
```

Keep the temporary build directory for inspection:

```bash
./mi50_vram_qc.sh --keep-build
```

## Dependencies

The script needs `hipcc` and the HIP runtime headers/libraries. It looks for `hipcc`
in this order:

1. `$HIPCC`
2. `command -v hipcc`
3. `/opt/rocm/bin/hipcc`

The script does not install packages by default. If you are on an apt-based system
and intentionally want the helper to try installing dependencies, pass:

```bash
./mi50_vram_qc.sh --install-deps
```

Review your distribution's ROCm packaging before using that flag. Package names
and repository setup vary across systems.

## Output

The HIP checker prints one result block per selected device:

```text
Device 0:
  name: AMD Instinct MI50
  totalGlobalMem: 34342961152 bytes
  target allocation: 30.00 GiB
  result: PASS
```

See [expected-output.md](expected-output.md) for a fuller sample.

## Interpreting results

`PASS` means:

- HIP allocation/check completed for the selected device or devices.
- No pattern errors were detected in the tested allocation region.

`FAIL` means:

- Allocation failed.
- Kernel launch or check failed.
- HIP device visibility failed.
- Memory pattern errors were detected.
- Another required command failed.

## Relationship to LocalAIServers

LocalAIServers publishes this tool as part of its public hardware-verification and
QC methodology for affordable local AI systems. The goal is to help people learn how
to inspect, verify, and reason about local AI hardware before relying on benchmark
or sales claims.

## Provenance

This tool is based on a LocalAIServers community QC script originally published by
the project maintainer and then cleaned up for repository use. It is maintained here
as an Apache-2.0 public hardware-verification education tool.

## Privacy

Before sharing output publicly, review it for host-specific details. The tool avoids
serial-number collection and does not upload logs anywhere, but kernel and ROCm logs
can still reveal local configuration details.

## License

Code in this directory is licensed under the repository Apache-2.0 license.
