# GFX906 amdgpu Full-BAR/P2P Patch Reconstruction

This note records the current public reconstruction of the host-side amdgpu
source patch that was part of the GFX906 full-BAR/P2P-on platform state used
for the v0.2 ROCm7.2 Dense/MoE results.

## Status

- A reconstructed patch artifact is now public at
  [`patches/gfx906-amdgpu-fullbar-p2p-reconstructed-20260619.patch`](../patches/gfx906-amdgpu-fullbar-p2p-reconstructed-20260619.patch).
- Patch SHA-256:
  `04149c2944f322476b5cf272d18a27bd796cb6ffc169ac2a2d2f0b1162ce0f8e`.
- The original standalone patch file has not been recovered.
- The exact clean source checkout and commit hash used for the local host build
  still need independent confirmation.
- This artifact is evidence for source-level review, not an installation
  instruction and not a bundled host patch.

## Evidence Used

The reconstruction is based on a 2026-06-19 dry-run log and local ROCm 7.2
amdgpu source snapshots from the same work period.

The dry-run log showed a patch check against these files:

```text
checking file amd/amdgpu/amdgpu.h
checking file amd/amdgpu/amdgpu_drv.c
Hunk #1 succeeded at 894 (offset 1 line).
checking file amd/amdgpu/amdgpu_device.c
checking file amd/amdgpu/amdgpu_amdkfd_gpuvm.c
Hunk #1 succeeded at 903 (offset -9 lines).
Hunk #2 succeeded at 952 (offset -4 lines).
```

Recovered source state showed the following functional pieces:

- `amdgpu_drv.c` added a `pcie_p2p` module parameter defaulting to enabled.
- `amdgpu_device.c` allowed large-BAR peer-access addressability checks to fall
  back outside `CONFIG_HSA_AMD_P2P` while still honoring `pcie_p2p`.
- `amdgpu_amdkfd_gpuvm.c` added `reuse_dmamap()` and reused the original DMA map
  for direct-mapped or same-IOMMU-group userptr/GTT mappings.
- `amdgpu.h` was listed by the dry-run target list. The reconstructed hunk adds
  the `extern bool pcie_p2p;` declaration needed by the recovered `pcie_p2p`
  use. The matching header snapshot was not recovered locally.

## What The Patch Appears To Do

The reconstructed patch appears to make the GFX906 full-BAR/P2P-on lane usable
by combining:

- a host-visible `pcie_p2p` module parameter;
- a large-BAR peer-access test that can still succeed through legacy
  addressability when the kernel does not provide full HSA P2P DMA mapping
  support;
- KFD GPUVM attachment behavior that can share the original DMA map for
  direct-mapped or same-IOMMU-group userptr/GTT mappings.

This matches the public v0.2 platform note: the Docker image alone was not the
whole performance story. The measured lane depended on host firmware state,
full-BAR exposure, P2P/topology state, and amdgpu source behavior.

## What Still Needs Verification

Before this can be treated as a reproducible host patch package, the remaining
work is:

- identify the exact clean upstream or ROCm kernel source base;
- re-apply the reconstructed patch against that source;
- confirm the recovered hunks match the observed dry-run offsets;
- build and load the resulting amdgpu module on a controlled test host;
- run the read-only host preflight and the normal v0.2 benchmark ladder;
- record the resulting source hash, build hash, and sanitized validation report.

## Public Boundary

This repository still does not bundle a host kernel package, DKMS package,
BIOS/VBIOS binary, firmware modification workflow, or host installer. The
reconstructed patch is provided for public technical review and reproducibility
work only.

This is not official AMD validation, not a warranty or certification, not a
hardware recommendation, and not a guarantee that every GFX906 system will
reproduce the v0.2 numbers.
