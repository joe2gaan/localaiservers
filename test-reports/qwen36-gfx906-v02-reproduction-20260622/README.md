# Qwen3.6 GFX906 v0.2 Reproduction Check

## Summary

- Date UTC: 2026-06-22
- Release tag tested: `v0.2.0-gfx906-rocm72-dense-moe`
- Image tag: `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`
- Image digest: `sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e`
- Benchmark method: eight pre-measure warmups, uncapped `c1_128` strict prompt, `c1_2000`, and `c1_10000`
- Hosts tested: `.20`, `.30`

## Results

| Profile | Host | Strict backend TPS | `c1_2000` backend TPS | `c1_10000` backend TPS | Result |
| --- | --- | ---: | ---: | ---: | --- |
| `dense27b_tp8_fullbar_p2pon` | `.20` | `68.962` | `69.914` | `65.645` | dense ai-info 10K gate confirmed |
| `moe35b_tp8_fullbar_p2pon` | `.20` | `94.637` | `94.742` | `89.489` | strict-valid bar reproduced; fixed-token tiers lower |
| `moe35b_tp8_fullbar_p2pon` | `.30` | `91.773` | `93.476` | `88.935` | lower fixed-token repeat on this host/run |
| `moe35b_tp4_fullbar_p2pon` | `.20` | `114.725` | `116.429` | `109.531` | corrected TP4 strict/fixed band confirmed |
| `moe35b_tp4_fullbar_p2pon` | `.30` | `114.289` | `115.966` | `109.131` | corrected TP4 strict/fixed band confirmed |

## Interpretation

Dense TP8 reproduced the v0.2 gate clear: the `c1_10000` backend TPS remained
above the 65 TPS ai-info gate.

MoE TP4 reproduced the corrected post-v0.2 strict/fixed-token band on both
hosts tested.

MoE TP8 strict validity reproduced on `.20`, but the fixed-token tiers repeated
below the original v0.2 release-time high point on both `.20` and `.30`. Treat
the published MoE TP8 fixed-token values as measured release-time points rather
than repeatability floors.

## Boundaries

This is reproducibility and benchmark-methodology evidence only. It does not
provide public compute access, hardware, procurement, resale, warranty,
certification, or official AMD validation. It does not guarantee that every
GFX906 system will reproduce these numbers.
