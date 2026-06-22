# Qwen3.6 35B-A3B MoE TP4 Strict Runaway Repeatability

## Summary

- Date UTC: 2026-06-22
- Release tag tested: `v0.2.0-gfx906-rocm72-dense-moe`
- Image tag: `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`
- Image digest: `sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e`
- Model: `Qwen/Qwen3.6-35B-A3B`
- TP4 profile: `moe35b_tp4_fullbar_p2pon`
- TP8 control profile: `moe35b_tp8_fullbar_p2pon`
- Hosts tested: `.20`, `.30`
- `MAX_MODEL_LEN`: `131072`
- Warmup method: eight pre-measure warmup requests, each capped at 2000 generated tokens with prompt repeat 32
- Strict prompt cutoff: 720 seconds; no strict TP4 repeat reached the cutoff
- Classification: **Strict-valid repeatable**

## Why This Test Was Run

The published v0.2.0 release treats MoE TP4 as capped fixed-token only because
the uncapped strict prompt previously did not stop and was classified as
invalid/runaway. This repeatability study checked whether that behavior is
consistent, intermittent, resolved under the current controlled profile, or
blocked by unrelated infrastructure.

This test used the published ROCm7.2 deploy profile and benchmark scorer path.
No BIOS/VBIOS changes, amdgpu changes, host reboot, release-note changes, or
public-claim changes were made.

## Test Matrix

| host | profile | run type | repeat | generated tokens | finish reason | qwen gate valid | backend TPS | decision |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | --- |
| `.20` | `moe35b_tp4_fullbar_p2pon` | strict uncapped | 1 | 5823 | stop | true | 113.560 | strict-valid pass |
| `.20` | `moe35b_tp4_fullbar_p2pon` | strict uncapped | 2 | 3109 | stop | true | 115.995 | strict-valid pass |
| `.20` | `moe35b_tp4_fullbar_p2pon` | strict uncapped | 3 | 3518 | stop | true | 115.759 | strict-valid pass |
| `.30` | `moe35b_tp4_fullbar_p2pon` | strict uncapped | 1 | 2809 | stop | true | 115.369 | strict-valid pass |
| `.30` | `moe35b_tp4_fullbar_p2pon` | strict uncapped | 2 | 3227 | stop | true | 115.242 | strict-valid pass |
| `.30` | `moe35b_tp4_fullbar_p2pon` | strict uncapped | 3 | 4807 | stop | true | 113.196 | strict-valid pass |
| `.20` | `moe35b_tp4_fullbar_p2pon` | c1_2000 fixed-token | 0 | 2000 | length | false | 116.787 | fixed-token sanity pass |
| `.20` | `moe35b_tp4_fullbar_p2pon` | c1_10000 fixed-token | 0 | 10000 | length | false | 109.622 | fixed-token sanity pass |
| `.30` | `moe35b_tp4_fullbar_p2pon` | c1_2000 fixed-token | 0 | 2000 | length | false | 115.770 | fixed-token sanity pass |
| `.30` | `moe35b_tp4_fullbar_p2pon` | c1_10000 fixed-token | 0 | 10000 | length | false | 108.950 | fixed-token sanity pass |
| `.30` | `moe35b_tp8_fullbar_p2pon` | strict uncapped control | 1 | 2974 | stop | true | 94.546 | strict-valid control pass |

## TP4 Strict Results

All six TP4 strict uncapped repeats stopped normally and passed the Qwen strict
gate. No strict run was safety-cut off. No strict run reproduced the previously
published runaway behavior under this controlled native TP4 full-BAR/P2P-on
profile.

Strict backend TPS ranged from `113.196` to `115.995` across the six repeats.
Generated-token counts varied from `2809` to `5823`, but every run ended with
`finish_reason=stop` and `qwen_gate_valid=true`.

Sanitized output signatures:

| host | repeat | text SHA256 prefix |
| --- | ---: | --- |
| `.20` | 1 | `ab75acb37e67` |
| `.20` | 2 | `9bfdf708f297` |
| `.20` | 3 | `26cf23a95815` |
| `.30` | 1 | `58860d2f3072` |
| `.30` | 2 | `4cbd9361628d` |
| `.30` | 3 | `18026e7dd9e6` |

## TP4 Fixed-Token Sanity Results

The capped TP4 fixed-token sanity runs completed on both hosts and stayed close
to the published v0.2.0 capped TP4 band:

| host | c1_2000 backend TPS | c1_10000 backend TPS |
| --- | ---: | ---: |
| `.20` | 116.787 | 109.622 |
| `.30` | 115.770 | 108.950 |

These fixed-token runs are not strict-valid evidence. They use an explicit
client token cap and therefore correctly report strict-gate invalid reasons
such as `client_max_tokens_cap` and `finish_reason_length`.

## TP8 Strict Control

The MoE TP8 strict-valid control on `.30` passed:

- Profile: `moe35b_tp8_fullbar_p2pon`
- Generated tokens: `2974`
- Finish reason: `stop`
- Qwen gate valid: `true`
- Backend TPS: `94.546`
- Text SHA256 prefix: `85c6ce79369e`

This confirms the strict prompt and scorer path worked during the study window.

## Interpretation

MoE TP4 strict behavior may have improved under the tested conditions, but this
is post-v0.2 validation evidence and requires a separate docs/release update
before changing public claims. The current v0.2.0 release claim should not be
changed by this report alone.

The tested native TP4 release profile did not show consistent runaway behavior.
It also did not show intermittent behavior in this six-repeat matrix. Under
these conditions, TP4 strict behavior was repeatably strict-valid across both
tested hosts.

## Boundaries

This is reproducibility and benchmark-methodology evidence only. It does not
provide public compute access, hardware, procurement, resale, warranty,
certification, or official AMD validation. It does not guarantee that every
GFX906 system will reproduce these numbers.
