# GFX906 amdgpu Full-BAR/P2P Patch Reconstruction and Source Lock

This note records the current public reconstruction of the host-side amdgpu
source patch that was part of the GFX906 full-BAR/P2P-on platform state used
for the v0.2 ROCm7.2 Dense/MoE results.

## Status

- The recovered ROCm 7.2 amdgpu source state is now pinned to the official
  `ROCm/ROCK-Kernel-Driver` source state below for reproduction review.
- A reconstructed patch artifact is now public at
  [`patches/gfx906-amdgpu-fullbar-p2p-reconstructed-20260619.patch`](../patches/gfx906-amdgpu-fullbar-p2p-reconstructed-20260619.patch).
- Patch SHA-256:
  `7e08477b0160c0487d0fd24746ee26a577d70990edd537ae45a916b76d4ef4a1`.
- The original standalone patch file has not been recovered.
- The reconstructed patch is evidence for source-level review against older or
  incomplete source bases, not an installation instruction and not a bundled
  host patch.
- Do not apply the reconstructed patch to the locked source state below without
  first checking whether the relevant hunks are already present.

## Locked Source State

The recovered source files match the official ROCm kernel-driver source for the
core public evidence files:

- Repository: `https://github.com/ROCm/ROCK-Kernel-Driver`
- Pinned tag: `rocm-7.2.1`
- Pinned peeled commit: `d2762fd86fce89f0b32614aeca806a548c7f6993`
- Same relevant file content was also observed in `rocm-7.2.2`; the recovered
  `.c` files also match `rocm-7.2.3` and `rocm-7.2.4`, but v0.2 reproduction
  should pin the earliest exact matched ROCm 7.2 tag unless a later source base
  is separately validated.

Relevant file hashes for the locked source state:

| File | SHA-256 |
| --- | --- |
| `drivers/gpu/drm/amd/amdgpu/amdgpu.h` | `f87b32d10e2cf141afad7a52d3fa1a79b5d74273130ebe5e923921c6e23a1cf3` |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_drv.c` | `414e5eb04a4ea17247c4596f452999340c3ea7e842c691cc41839d6cd3dff8da` |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_device.c` | `4d0edc4b714c005e911596e0e2e616be7fdbbb3526069938e4cc078eaba83673` |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gpuvm.c` | `c7cca2ee47a08c99bb73906662d82dd7d0b5738468fbef54848e5e6dd62ba50d` |

The local recovered snapshot did not include `amdgpu.h`, but the official
locked source contains `extern bool pcie_p2p;` in that header. The recovered
`.c` files match the hashes above exactly.

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

Recovered source state and the locked official source contain the following
functional pieces:

- `amdgpu_drv.c` added a `pcie_p2p` module parameter defaulting to enabled.
- `amdgpu_device.c` allowed large-BAR peer-access addressability checks to fall
  back outside `CONFIG_HSA_AMD_P2P` while still honoring `pcie_p2p`.
- `amdgpu_amdkfd_gpuvm.c` added `reuse_dmamap()` and reused the original DMA map
  for direct-mapped or same-IOMMU-group userptr/GTT mappings.
- `amdgpu.h` exposes the `pcie_p2p` declaration in the locked official source.

## What The Patch Appears To Do

The locked source state appears to make the GFX906 full-BAR/P2P-on lane usable
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

Before this can be treated as a reproducible host module package, the remaining
work is:

- build the amdgpu module from the pinned `rocm-7.2.1` source state;
- build and load the resulting amdgpu module on a controlled test host;
- run the read-only host preflight and the normal v0.2 benchmark ladder;
- record the resulting source hash, build hash, module metadata, and sanitized
  validation report.

The reconstructed patch remains useful for understanding the older dry-run
artifact, but reproduction should now lock to the pinned official source state
instead of treating the patch as an unversioned freestanding diff.

## Public Boundary

This repository still does not bundle a host kernel package, DKMS package,
BIOS/VBIOS binary, firmware modification workflow, or host installer. The source
lock and reconstructed patch are provided for public technical review and
reproducibility work only.

This is not official AMD validation, not a warranty or certification, not a
hardware recommendation, and not a guarantee that every GFX906 system will
reproduce the v0.2 numbers.
