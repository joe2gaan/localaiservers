# GFX906 Host Platform Prerequisites for v0.2

This note describes the host-side platform state expected by the ROCm7.2
Dense/MoE v0.2 reproduction package.

The v0.2 Docker image, model package, runtime overlays, and benchmark scorer are
public. The host platform work is separate: the published numbers were measured
on prepared GFX906 hosts with full-BAR/P2P-on platform state.

## Required Host State

For comparable v0.2 reproduction runs, the selected host should have:

- Enough `gfx906` GPUs for the selected profile:
  - `dense27b_tp8_fullbar_p2pon`: 8 GPUs.
  - `moe35b_tp8_fullbar_p2pon`: 8 GPUs.
  - `moe35b_tp4_fullbar_p2pon`: 4 GPUs.
- Full-BAR exposure for the selected GPUs. For MI50 32GB-class cards, the
  practical check is a largest visible PCI BAR of at least 32 GiB per selected
  GPU.
- P2P/topology state visible to ROCm.
- Official AMD VBIOS standardization where needed. The standardized full-BAR
  GFX906 VBIOS revision recorded for the public v0.2 host-platform record is
  `113-D1631700-111`.
- Host amdgpu source state matching the pinned `ROCm/ROCK-Kernel-Driver`
  `rocm-7.2.1` source lock recorded below. Reconstructed patch evidence is
  published for source review, but the public reproduction package still does
  not bundle or apply a host amdgpu patch.

## GPU BIOS / VBIOS Revision

The exact public GPU VBIOS revision used for the standardized full-BAR GFX906
platform state is:

```text
113-D1631700-111
```

This value is a VBIOS revision string, not a card serial number. Do not publish
per-card serial numbers, private inventory labels, or BIOS/VBIOS binaries in
this repository.

The external MI50 32GB VBIOS matrix maintained by evilJazz also records
`113-D1631700-111` as a ReBAR / 32 GiB BAR-capable AMD MI50 32GB VBIOS and
distinguishes it from `113-D1631711-100`, which is documented there as
non-ReBAR / 16 GiB BAR-visible. That external reference is a VBIOS matrix, not
part of this release package:
<https://gist.github.com/evilJazz/14a4c82a67f2c52a6bb5f9cea02f5e13>

The public preflight script checks for this revision through `rocm-smi
--showvbios` when ROCm-SMI can initialize on the host:

```bash
EXPECTED_GPU_VBIOS=113-D1631700-111 ./check_host_platform_prereqs.sh
```

If a future host uses a different official AMD VBIOS revision, record the exact
public revision string here before comparing that host with the v0.2 published
numbers.

## What Is Not Bundled

The v0.2.1 public reproduction package does not bundle:

- BIOS or VBIOS binaries.
- A host amdgpu patch installer or binary package.
- DKMS packages.
- Linux kernel packages.
- Instructions to flash cards or modify firmware.

The public repository now includes a pinned amdgpu source-state lock and
reconstructed patch evidence for source review:

- Reconstruction note:
  [`docs/gfx906-amdgpu-fullbar-p2p-reconstruction-20260619.md`](gfx906-amdgpu-fullbar-p2p-reconstruction-20260619.md)
- Locked source: `ROCm/ROCK-Kernel-Driver` tag `rocm-7.2.1`, peeled commit
  `d2762fd86fce89f0b32614aeca806a548c7f6993`
- Reconstructed patch:
  [`patches/gfx906-amdgpu-fullbar-p2p-reconstructed-20260619.patch`](../patches/gfx906-amdgpu-fullbar-p2p-reconstructed-20260619.patch)
- Reconstructed patch SHA-256:
  `7e08477b0160c0487d0fd24746ee26a577d70990edd537ae45a916b76d4ef4a1`

The original standalone patch file has not yet been recovered. The recovered
`.c` files match the locked official source state; treat that source state as
the host amdgpu prerequisite until a proper host module package is published.
Do not apply the reconstructed patch to the locked source without checking
whether the relevant hunks are already present.

## Read-Only Preflight

Run the host platform preflight before deploying:

```bash
git clone https://github.com/joe2gaan/localaiservers.git
cd localaiservers/qwen36-gfx906

export QWEN36_PROFILE=dense27b_tp8_fullbar_p2pon
./check_host_platform_prereqs.sh
```

The existing `v0.2.1-gfx906-rocm72-dense-moe-repro` tag remains the named
reproduction package for the 2026-06-22 script/report state, but this preflight
helper was added after that tag. Use current `main` or a later reproduction tag
when you need the helper in the checkout.

For MoE TP4:

```bash
export QWEN36_PROFILE=moe35b_tp4_fullbar_p2pon
./check_host_platform_prereqs.sh
```

For MoE TP8:

```bash
export QWEN36_PROFILE=moe35b_tp8_fullbar_p2pon
./check_host_platform_prereqs.sh
```

The script is read-only. It does not patch amdgpu, flash firmware, change
kernel settings, start containers, or run model workloads.

## Interpreting Results

A passing preflight means the host exposes the basic platform signals expected
by the v0.2 reproduction package:

- `amdgpu` is loaded.
- The expected number of likely `gfx906` PCI devices is visible.
- Each selected device appears to expose a large enough PCI BAR.
- ROCm/KFD topology checks are available where host tools expose them.
- ROCm-SMI can report the expected standardized VBIOS revision, if ROCm-SMI is
  available and can initialize on the host.

A passing preflight is not a performance guarantee. It does not prove that every
GFX906 system will reproduce the same numbers.

A failing preflight means local measurements should not be compared with the
published v0.2 numbers until the host platform state is corrected.

## Current Reproduction Boundary

The public v0.2.1 reproduction package uses the same v0.2.0 Docker image:

```text
joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b
```

Docker Hub manifest digest:

```text
sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e
```

The host platform prerequisite is outside that image. The release does not
provide public compute access, hardware, procurement support, resale support,
warranty, certification, official AMD validation, compensation, or professional
services.
