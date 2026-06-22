# ROCm7.2 Dense/MoE Active Contracts - 2026-06-20

This document records the ROCm7.2 Dense/MoE active contracts copied from
[qwen36-gfx906/README.md](../qwen36-gfx906/README.md), which is the
authoritative latest technical source for the ROCm7.2 Dense/MoE runner.

v0.1.0 remains the older published GitHub Release boundary. The dense and MoE
active-contract values below are covered by the published v0.2.0 release
boundary. GitHub Releases remain canonical for published claim boundaries.

## Image Identity

Current `deploy.sh` SHA256:

```text
c8e8ef99ec39a0232f74a7bd0fe0efe0316c0e0678992a1c104eff3c05513c9a  deploy.sh
```

Default pushed image:

```text
joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b
```

Docker Hub manifest digest:

```text
sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e
```

Final deterministic release-tag archive from the patched deploy path:

```text
5316c3f6202fcb77987dabbf1e14e7369441ea127efed4f6def30259a09cfcb9  finaldeploy-patched.docker.tar
```

Image identity inside that archive:

```text
manifest sha256:7dadf367ec86fe2eb1dc22fb3af3002c3514514833b52329595a26e7a80ae247
config   sha256:45decd88eb7c10c0408327438e07c2a655e45cc7534f8b662e5c4089a6b88568
created  2025-11-24T16:00:00Z
layers   19
```

The same ROCm7.2 experimental release image covers dense and MoE through
model-specific env and overlay selection.

Docker Hub remains an evergreen artifact distribution channel; TPS claims should
stay in GitHub Releases, repository docs, and benchmark artifacts.

## Verified Portable Performance

Verified portable performance at `MAX_MODEL_LEN=131072` with eight pre-measure
warmups:

| profile | TP degree | host | strict backend TPS | c1_2000 backend TPS | c1_10000 backend TPS | note |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `dense27b_tp8_fullbar_p2pon` | 8 | `.20` | `69.514` | `70.347` | `66.069` | strict gate valid |
| `moe35b_tp8_fullbar_p2pon` | 8 | `.30` | `94.907` | `97.028` | `91.290` | strict gate valid |
| `moe35b_tp4_fullbar_p2pon` | 4 | `.30` | initial caveat corrected below | `116.146` | `109.283` | release-time fixed-token result; post-v0.2 strict repeatability passed |

Dense 27B TP8 therefore clears the ai-info 10K gate at the published v0.2.0
release boundary. MoE TP8 is the current strict-valid Qwen3.6 35B-A3B
full-BAR/P2P-on bar. The MoE TP4 release-time fixed-token result remains
`109.283` c1_10000 backend TPS, and the follow-up below corrects the earlier
strict-runaway caveat.

## Post-v0.2 TP4 Strict Repeatability Follow-Up

TP4 strict runaway did not reproduce in the follow-up study under the native
`moe35b_tp4_fullbar_p2pon` release profile.

- `6/6` strict repeats passed across `.20` and `.30`.
- All repeats finished with `finish_reason=stop`.
- All repeats had `qwen_gate_valid=true`.
- Strict backend TPS range: `113.196` to `115.995`.
- `.20` fixed sanity: c1_2000 `116.787`, c1_10000 `109.622`.
- `.30` fixed sanity: c1_2000 `115.770`, c1_10000 `108.950`.

This is post-v0.2 validation evidence. It does not rewrite the v0.2.0 release
boundary. No code, Docker image, tag, or runtime artifact changed.

## Platform Preconditions

The full-BAR/P2P-on lane required official AMD VBIOS standardization, not
modified BIOS images, plus amdgpu source patching. Keep that platform
remediation separate from model-performance claims. This is not a user
instruction to flash cards; the public repo does not redistribute BIOS binaries
or imply warranty or certification.
