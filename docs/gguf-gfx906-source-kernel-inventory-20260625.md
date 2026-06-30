# GGUF / GFX906 Source Kernel Inventory - 2026-06-25

This is the GGUF-specific source inventory for the Qwen3.6 / GFX906 work. It
tracks source paths, overlays, knobs, and diagnostics tested while trying to
make GGUF models reproduce the same kind of correctness and throughput evidence
as the FP16 ROCm7.2 Dense/MoE release path.

This document is an active investigation record, not a release note.

## Current Readout

- The FP16 ROCm7.2 Dense/MoE path remains the canonical published reproduction
  path.
- `.10`, `.20`, and `.30` are the same EPYC / 8x MI50 32GB class. The published
  benchmark host remains `.20` unless a run explicitly records another lane. Use
  the InfiniBand address family for host-to-host and benchmark-lane work.
- The `.20` GGUF workspace is `<validation-workspace>` on NVMe.
  `LLM_STORE_VOL` is a `.40`-only source-debug workspace and must not be created
  or used on `.10`-`.30`. The separate `.40` 8x MI60 lane is source-debugging
  only, not benchmark reproduction evidence.
- The MoE F16 GGUF artifact and Qwen3.5-MoE experimental patch bundle are now
  staged on `.20` NVMe. The staged MoE GGUF hash matched the `.40` source copy:
  `1f2443bb0ff958943d091410c61120c181a0579b3bc85192029aa51d821d141c`.
- Historical Q4_0 GGUF TP paths proved useful for isolating GatedDeltaNet
  tensor-parallel drift, but they did not promote.
- Dense FP16/half GGUF now produces coherent output and passes the strict gate
  under the Qwen3.5 CausalLM path with GGUF tensor-layout fixes and the release
  overlay.
- Dense FP16/half GGUF is now benchmark-promoted for the dense reproduction
  target after the `lm_head` logits-path fix. The current best dense GGUF result
  is strict `70.505`, `c1_2000` `71.589`, and `c1_10000` `66.967` backend TPS,
  which beats the published dense release values on the `.20` lane.
- The reduced vNext Dense package path has moved from preflight to runtime
  evidence on `.30`: `dense27b_pkg_dot30_20260628T174515Z` launched from the
  contract-generated Docker wrapper and pinned minimal bundle, then completed
  the standard ladder at strict `70.163`, `c1_2000` `70.975`, and `c1_10000`
  `66.495` backend TPS.
- The clean vNext Dense GGUF TP8 contract run
  `20260629T003817Z-gguf-dense27b-tp8` supersedes the reduced-package-only
  status. It completed on `.20` and `.30` from the generated launcher, pinned
  minimal bundle, generated Docker wrapper, generated benchmark environment,
  TP8, FP16, P2P-on, and `MAX_MODEL_LEN=131072`: `.20` strict `69.914`,
  c1_2000 `70.759`, c1_10000 `66.316`; `.30` strict `69.851`, c1_2000
  `70.959`, c1_10000 `66.437`. Both strict runs passed.
- MoE FP16/half GGUF TP4 now has a release-overlay combo promotion candidate.
  Under `moe35b_tp4_fullbar_p2pon`, P2P-on, and the normal benchmark ladder, it
  produced `.20` strict `118.754`, c1_2000 `119.896`, and c1_10000 `112.747`
  backend TPS, and `.30` strict `119.917`, c1_2000 `120.781`, and c1_10000
  `113.605` backend TPS.
- The clean vNext MoE GGUF TP4 contract run
  `20260629T003817Z-gguf-moe35b-tp4` now validates the MoE package path across
  `.20` and `.30`: `.20` strict `119.333`, c1_2000 `120.565`, c1_10000
  `113.366`; `.30` strict `119.521`, c1_2000 `120.460`, c1_10000 `113.258`.
  Both strict runs passed. The source logs show the custom top-k8 MoE fastpath
  rejected larger graph shapes by shape/layout but activated token-sized decode
  shapes for `tokens=4`, `tokens=2`, and `tokens=1`.
- The public-staged vNext GGUF reproduction audit on `.30` verified that the
  release path can run from split F16 GGUF assets, SHA256 manifests, the Dense
  text-config archive, generated launch artifacts, and the normal benchmark
  ladder. Dense GGUF TP8 produced strict `70.072`, c1_2000 `71.031`, and
  c1_10000 `66.487`. MoE GGUF TP4 produced strict `120.319`, c1_2000
  `121.018`, and c1_10000 `113.760` after replacing an internal tokenizer
  snapshot path with public `Qwen/Qwen3.6-35B-A3B` plus
  `TOKENIZER_REVISION=995ad96eacd98c81ed38be0c5b274b04031597b0`.
- The POSIX-runner repeat of the same public-staged path on `.30` then
  verified the final vNext benchmark-runner shape without delegating to the
  older Bash-only v0.2 helper. Dense GGUF TP8 produced strict `69.986`,
  c1_2000 `70.935`, and c1_10000 `66.403`; MoE GGUF TP4 produced strict
  `120.822`, c1_2000 `121.733`, and c1_10000 `114.442`.
- The clean vNext HF MoE TP8 contract run
  `20260629T003817Z-hf-moe35b-tp8` validates the non-GGUF TP8 baseline path on
  `.20` and `.30`: `.20` strict `115.041`, c1_2000 `115.549`, c1_10000
  `108.805`; `.30` strict `114.698`, c1_2000 `115.532`, c1_10000 `108.673`.
  Both strict runs passed under `PATCH_TARGET_GROUPS="common moe hf moe_tp8"`.
  Startup still warns that an `E=256,N=64` tuned MoE config is not bundled, but
  that warning did not block high-band TP8 reproduction in this run.
- The vNext launcher now maps known container-required paths back to
  host-visible mount roots during preflight: `<container-hf-cache>/...`
  through `HOST_HF_CACHE`, and `/opt/vllm_patch_bundle/...` through
  `PATCH_BUNDLE_PATH`. This preserves container-path contracts while keeping
  host preflight strict.
- The official release image plus the same GGUF/release overlay performs in the
  same band as the experimental image, so image-base drift is rejected.
- A clean HF-weight dense TP8 control on the same host lane, official image,
  release overlay, P2P-on path, and benchmark harness reproduces the published
  band: strict `69.652`, `c1_2000` `70.202`, and `c1_10000` `65.952` backend
  TPS.
- Restoring release attention routing did not improve dense GGUF decode.
- ConditionalGeneration architecture alignment did not improve decode and made
  prefill worse.
- The F16 merged-single-matmul hypothesis was rejected because the major dense
  linears already load as normal `weight` params rather than GGUF `qweight`.
- Removing disabled forward trace calls from the GGUF Qwen3.5 path did not
  improve decode.
- The coherent GGUF release-attention runs log a Transformers FLA /
  causal-conv1d fallback warning, but later source inspection localized that
  warning to GGUF loader dummy-model construction rather than the proven serving
  hot path. The active gap remains a GGUF source-path parity issue rather than
  host state, image base, or benchmark harness drift.
- A fused-GDN GGUF parity bundle can load qkv/z/b/a into the HF release fused
  projection shape and is serving-capable with a patient startup/smoke window.
  It improves dense decode but does not close the FP16/HF release gap.
- Combining fused-GDN with native GGUF SwiGLU was the previous dense GGUF source
  baseline. It was superseded by the dense `lm_head` unquantized logits-path
  result, which closes the dense reproduction target.
- Patch-bundle routing can overwrite individually mounted Qwen source files at
  container startup, so future tests must use one coherent bundle.
- The leading source target is now Qwen3.5 / Qwen3.6 GatedDeltaNet and
  linear-attention execution-path performance with GGUF weights after tensor
  materialization.
- The strongest current source clue is mixed-quant split granularity around
  Qwen3.5 / Qwen3.6 GatedDeltaNet: upstream llama.cpp needed an `ssm_out`
  fallback for qkv/gate split config, while the local GGUF path has Q4_0
  qkv/gate tensors and a Q8_0 `ssm_out` tensor.
- MoE GGUF has moved from artifact absence to source-route investigation. The
  F16 MoE GGUF artifact is present and tokenizer metadata is sane. A copied
  experimental source route now maps `qwen35moe` into the Qwen3.5-MoE text
  class and reaches API health under graph mode with forced unquantized
  FusedMoE, restored text M-RoPE, and the full inverse Qwen3.5
  linear-attention GGUF layout bridge applied to both direct and packed
  projection loads. Before the runtime norm-offset fix, it did not return
  coherent client-visible text and could not enter benchmark warmups. With the
  runtime norm-offset fix, raw deterministic probes are coherent. An env-gated
  graph KV-layout bypass gets the 131K path healthy. The later `A_log` loader
  repair plus full-attention q/k norm repair makes the clean 131K graph-mode
  ladder strict-valid and fixed-token-complete, but backend TPS remains far
  below the FP16 v0.2.1 TP4 reference band.
- The `.20` official v0.2 image route now loads the staged Qwen3.5-MoE F16 GGUF
  artifact when the active source files are mounted directly. It originally
  failed before API health in hybrid KV-cache initialization with a
  five-dimensional cache tensor layout ambiguity. Explicit prefix-cache
  alignment did not fix that startup blocker, but an env-gated source bypass
  that assumes leading-KV layout for the ambiguous tensor lets graph mode reach
  health.
- A request-gated final-logit trace bundle is staged and has fired on a real
  `.20` eager-mode one-token request. The request still returned `_`, and all
  four TP workers agreed on the same bad top-token sequence. This is
  request-level evidence that the remaining MoE GGUF corruption is upstream of
  sampling.
- A ROCm llama.cpp control with the same `.20` MoE F16 GGUF artifact returns
  comma for raw `Hello`, so the artifact/tokenizer is coherent. In vLLM, comma
  token id `11` ranks only `25` while underscore token id `62` ranks `1`.
- The same vLLM watch-id probe at TP8 returned underscore, with comma only
  rank `20`, before the norm fix. That rejected a TP4-specific explanation for
  the pre-fix MoE GGUF corruption.
- A clean HF-weight vLLM TP4 control on the same `.20` MI50 lane, official v0.2
  image, P2P-on, eager mode, and raw one-token `Hello` probe returns comma and
  ranks comma id `11` first. Underscore id `62` ranks `369`. This rejects the
  shared Qwen3.5-MoE execution path, sampler, and raw prompt as the primary
  explanation; the remaining bug is specific to GGUF materialization/state
  formation before final logits.
- Boundary tracing showed the first measured GGUF/HF divergence before the
  GatedDeltaNet core: layer-0 linear-attention input norm was GGUF `91.739395`
  versus HF `46.638695`.
- Decoder-layer norm tracing found the cause: GGUF and HF enter layer 0 with
  the same pre-layernorm embedding norm `0.610113`, but GGUF loads non-offset
  RMSNorm values into vLLM's `GemmaRMSNorm`, which already applies
  `1 + weight`. The GGUF-only runtime norm-offset correction changes the
  layer-0 post-norm to `46.637737`, matching HF `46.638695`, changes final
  hidden to `114.727249`, matching HF `114.722717`, and changes raw `Hello`
  from `_` to comma.
- The 131K benchmark-path tests with the runtime norm-offset fix rejected both
  eager and current graph-mode candidates. Eager mode reaches health and stays
  coherent on tiny probes, but the normal warmups decode only at
  `12.455`-`14.060` backend TPS and the uncapped strict request does not close
  normally. The env-gated graph KV-layout bypass gets past the previous
  `torch.Size([2, 2, 528, 1, 256])` hybrid cache assertion and raises warmups to
  `73.579`-`73.655` backend TPS, but those warmups still produce no visible
  answer or think close. The uncapped strict request was cut off after the live
  metric estimate crossed `60071` generated tokens with no close.
- A shortest-prompt comparator rejects prompt length and capped-output behavior
  as sufficient explanations for the MoE GGUF strict failure. GGUF
  `prompt_repeat=0` stayed in the reasoning-only path and timed out after an
  estimated `18019` generated tokens for the active request, while a clean
  HF-weight TP4 eager comparator closed the same shortest prompt after `2957`
  generated tokens with `finish_reason=stop`, `qwen_gate_valid=true`, `10125`
  reasoning chars, and `3930` visible answer chars. The next source boundary is
  the GGUF reasoning-to-answer delimiter/stop transition under benchmark-task
  prefill.
- A direct thinking-disabled diagnostic narrows that boundary further. With
  default thinking enabled, GGUF `max_tokens=128` produced `621` reasoning chars
  and `0` content chars. With `chat_template_kwargs.enable_thinking=false`, the
  same prompt produced visible content immediately: `700` content chars at
  `128` tokens and `2158` content chars at `512` tokens. The 512-token tail was
  repetitive, so this is not a benchmark path, but it proves the current graph
  path can generate content if the parser/template begins after the reasoning
  end. Tokenizer inspection pins `<think>` to id `248068` and `</think>` to id
  `248069`.
- The first faithful chat branch divergence is now reproduced in both graph and
  eager serving shapes: after shared token `248046`, GGUF prefers `Here` over
  `Thinking`, while HF prefers `Thinking` over `Here`. The eager boundary probe
  still captured only `model.final_hidden`, so the current hook shape is
  rejected for layer localization. The next diagnostic should move the
  request-gated summaries deeper into late-layer residual, MoE/router/expert,
  and GatedDeltaNet/Mamba state, or use an offline reduced comparison.

## Source Paths And Dispositions

| Source family | Files / overlays / knobs | What was tested | Evidence | Disposition |
| --- | --- | --- | --- | --- |
| Text-only Qwen3.6 config for GGUF | Qwen3.6 text-only config used with GGUF weights | Avoided multimodal / M-RoPE model-path selection while loading GGUF files. | TP1 and TP2 first-token probes became useful without unrelated config behavior. | Promoted diagnostic precondition. |
| GGUF loader with Q4_0 dense weights | Qwen3.6 27B Dense Q4_0 GGUF | Loaded and served simple deterministic probes at TP1, TP2, TP4, and TP8. | TP1/TP2 first-token behavior was coherent; TP4 degraded; TP8 was wrong. | Active source lane; not benchmark-promoted. |
| GGUF FP16/half dense weights | Qwen3.6 27B Dense FP16/half GGUF candidate | Converted from cached source weights, scanner-verified as FP16/half, and served through the experimental local-file config patch. | TP8 full-context warmups completed at about `49.69` to `49.74` backend decode TPS after first-request overhead, but all warmups produced deterministic repeated reasoning text with no visible answer and no think close. The strict request was stopped manually and no fixed-token tiers were promoted. | Rejected as benchmark candidate; promoted as evidence that the failure is not only Q4_0 quantization. |
| QK repeat behavior | Qwen3.5 / Qwen3.6 attention and GatedDeltaNet path | Kept QK repeat semantics in the scratch path. | Earlier no-repeat behavior caused shape or dummy-path failures. | Promoted source precondition. |
| KV replica mode | `VLLM_QWEN35_KV_REPLICA_MODE=mod` | Tested whether TP4 first-token drift was a K/V replica-mode problem. | TP4 output did not repair. | Rejected as fix. |
| Async scheduling removal | Direct no-async launch shape | Tested whether async scheduling was the source of the TP4 drift. | Engine did not initialize cleanly under the tested path. | Rejected as reproduction path. |
| Final hidden/logit tracing | Scratch Qwen3.5 model instrumentation | Compared TP1 and TP4 final hidden values and top-token IDs. | TP4 hidden state had already diverged before logits. | Promoted diagnostic evidence. |
| Layer-0 GatedDeltaNet tracing | Scratch layer trace hooks | Compared layer-0 normalization, attention output, residual, post-attention norm, FFN output, and post-FFN materialization. | TP1 and TP4 shared the same input-normalization sum, then diverged at layer-0 GatedDeltaNet attention output. | Promoted current root-cause boundary. |
| Official GGUF format reference | Upstream GGUF specification: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md> | Checked scanner assumptions against the public file format. | The spec confirms aligned tensor-data offsets and type IDs relevant to this work: `Q4_0` = `2` and `Q8_0` = `8` for the active audit. | Promoted source-orientation evidence. |
| llama.cpp GGUF reader implementation | `ggml/src/gguf.cpp`: <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/gguf.cpp> | Reviewed the C++ reader behavior around tensor metadata, alignment, duplicate names, type validation, row-size divisibility, and `no_alloc` data attachment. | Reference implementation validates contiguous aligned offsets and attaches tensor data relative to the GGUF data section. | Promoted implementation reference for scanner parity. |
| Low-level GGUF tensor-info scanner | Scratch C scanner built from GGUF header, KV, and tensor-info parsing | Inspected Q4_0 and FP16/half split tensor names, dimensions, types, and offsets without relying on vLLM loader state. | Layer-0 Qwen3.6 GGUF tensors use llama.cpp-style names: `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`, and `ssm_out`. | Promoted source diagnostic. |
| llama.cpp Qwen3.5 layout comparison | Qwen3.5 tensor-name and graph conventions | Compared GGUF tensor names and graph layout against vLLM scratch fusion points. | q/k/v are represented as one contiguous `attn_qkv` tensor and z/gate is a separate projection; this weakens the simple qkv group-reorder hypothesis. | Promote as source-orientation evidence. |
| vLLM GGUF public support boundary | vLLM GGUF docs: <https://docs.vllm.ai/en/stable/features/quantization/gguf/> | Checked public GGUF support posture, tokenizer/config guidance, and tensor-parallel example. | GGUF remains documented as experimental; docs recommend base-model tokenizer and `--hf-config-path` when metadata conversion is insufficient. | Promoted launch guardrail; not a correctness fix. |
| Qwen3.5 GGUF loader mapping PR | vLLM issue #38122 and PR #38140 | Reviewed public Qwen3.5 GGUF failure class: `qwen3_5` versus `qwen35` naming and Qwen3.5 vision-config `depth` fallback. | The PR addresses startup/config loading and has tests for those cases, but does not address TP4/TP8 GatedDeltaNet correctness after serving starts. | Useful upstream prerequisite; not the local root cause. |
| Local-file GGUF config bypass issue | vLLM issue #36456 | Checked a separate public local-file failure where config probing can bypass `--hf-config-path`. | This supports keeping the current explicit repo/quant-style launch while debugging; the local failure is later in execution. | Launch guardrail. |
| Current vLLM Qwen3.5 GatedDeltaNet source | `QwenGatedDeltaNetAttention`, `Qwen3_5ForCausalLMBase`, `gqa_interleaved_layout=False` | Compared current source against the release FP16 overlay. | Current source builds `in_proj_qkvz` as `[q, k, v, z]` and `in_proj_ba` as `[b, a]`; ROCm fast path explicitly treats Qwen3.5 as non-interleaved and falls back to generic split/rearrange where needed. | Active source fork point. |
| FP16 release fused qkvzba overlay | `qwen36-gfx906/deploy.sh` embedded Qwen3.5 overlay | Compared the published FP16 overlay against current vLLM Qwen3.5 source. | The FP16 path fuses b/a into `in_proj_qkvz` and maps qkv with tuple shard IDs. That path is proven for FP16 release reproduction, but is risky for GGUF until materialization and shard semantics are proven. | Do not blindly reuse for GGUF; test split path. |
| GGUF packed-shard loader restrictions | `qwen36-gfx906/files/gfx906_runtime/python_overlays/persistent_ar_kindroute_20260613/linear.py` | Reviewed GGUF-specific packed weight and weight-type loading behavior. | The release overlay has GGUF special cases and restrictions around multi-index shard IDs; tuple shard loading can be incompatible with GGUF materialization. | Active loader-risk boundary. |
| Forced dequantized GGUF matmul fallback | `VLLM_GGUF_FORCE_DEQUANT_MATMUL=1` in scratch GGUF loader path | Bypassed the quantized GGUF matmul path to test whether ROCm GGUF MMVQ/MMQ was the main error. | TP4 still failed first-token sanity. | Rejected as fix; useful negative evidence. |
| Torch GatedDeltaNet reference path | `VLLM_QWEN35_TORCH_GDN_PREFILL=1` in scratch Qwen3.5 path | Replaced the custom GatedDeltaNet core with a torch reference path for prefill. | TP4 still failed first-token sanity. | Rejected as fix; promoted evidence that input tensors or TP/GGUF layout remain suspect. |
| GGUF loader fallback warning | Transformers Qwen3.5 / Qwen3.5-MoE constructors and vLLM GGUF loader | Localized the FLA / causal-conv warning seen in GGUF logs. | The warning is emitted while the GGUF loader builds a meta-device `AutoModelForCausalLM` dummy model for state-dict/name mapping; vLLM serving `qwen3_5.py` uses `torch.ops.vllm.gdn_attention_core` and does not contain the warning path. | Reject package installation as a blind fix; use request-window profiling for hot-path proof. |
| Logical GGUF shard-order patch | GGUF linear-method ordering around integer shard IDs | Tested whether sorting logical shard IDs before linear application restored TP4 first-token sanity. | TP4 still returned the wrong first-token ordering under P2P-on; `.` was top and `,` second. | Rejected as fix; useful negative evidence. |
| TP1 forward-time checksum control | Same experimental image, logical-order patch, and layer-0 hooks as TP4 | Reran the deterministic one-token `Hello` probe at TP1 after TP4 failed. | TP1 returned `,` as top token and also had physical qkvz order `[3, 0, 1, 2]`. | Promoted control evidence; physical qkvz order alone is not the root cause. |
| llama.cpp Qwen3.5 / Qwen3.6 TP granularity fix | PR #23843: <https://github.com/ggml-org/llama.cpp/pull/23843>; local `src/llama-model.cpp` split config | Reviewed the upstream fix for Qwen3.5 / Qwen3.6 heterogeneous quant tensor-parallel splitting. | qkv and gate split config can fall back to `ssm_out.weight`; the PR says wrong tensor granularity caused split inconsistencies. | Promoted external source clue; audit vLLM GGUF split boundaries against this behavior. |
| C split audit helper | Scratch C parser / split auditor built from GGUF header, KV, and tensor-info parsing | Calculated layer-0 tensor bytes, quant block sizes, and TP2/TP4/TP8 split sizes for the active Q4_0 GGUF file. | Q4_0 qkv/gate and Q8_0 `ssm_out` split cleanly by raw block size at TP4 and TP8. | Reject simple quant-block remainder theory; continue segment-level audit. |
| TP4 all-rank layer-0 trace | Same image and P2P-on path with rank-0-only tracing disabled | Reran the deterministic one-token probe and captured rank-local layer-0 activity. | TP4 reproduced the wrong `.` top token; all ranks agreed on final wrong top IDs, with rank-local GDN differences before synchronization. | Promoted failure-boundary evidence; not a fix. |
| TP4 segment-labeled trace | Scratch Qwen3.5 overlay with q/k/v/z and beta/alpha trace labels | Reran the same deterministic one-token probe with explicit logical segment labels. | q/k/v/z and beta/alpha were populated across ranks, but TP4 still returned `.` first. | Reject empty-segment theory; inspect segment ordering, split semantics, and GDN state/reassembly next. |
| TP1 segment-labeled control | Same scratch Qwen3.5 segmenttrace overlay, TP1, same deterministic prompt | Reran the segment-labeled hook on a coherent single-rank path. | TP1 returned `,` as top token while TP4 under the same hook returned `.` first and `,` second. | Promote as control evidence; instrumentation is not the failure source. |
| TP1/TP4 segment-checksum diagnostic | Scratch Qwen3.5 checksum overlay, TP1 and TP4, same deterministic prompt | Compared compact q/k/v/z/b/a/core/out sums at the final one-token decode step. | TP4 q/k/v/z/b/a sums aggregate to TP1, but GatedDeltaNet core/output checksums diverge and TP4 still returns `.`. | Promote narrowed root-cause evidence; inspect GatedDeltaNet core/state and output reassembly. |
| TP1 chunk versus TP4 rank post-conv comparison | Scratch shared-core hook after `causal_conv1d` | Compared TP1 rank-equivalent chunks with TP4 rank-local post-conv q/k/v, b/a, `A_log`, and `dt_bias`. | Value, b/a, `A_log`, and `dt_bias` match by chunk/rank; q/k do not. | Historical divergence boundary; later checks supersede the simple conv-weight-order hypothesis. |
| TP1/TP4 pre-conv and conv-weight checksum comparison | Scratch shared-core hook before and after `causal_conv1d` | Compared TP1 chunk-equivalent q/k/v pre-conv inputs and q/k/v `conv1d.weight` slices against TP4 rank-local data. | Pre-conv inputs and conv weights match exactly by TP1 chunk versus TP4 rank, while tiled-repeat TP4 still returns `.`. | Reject simple conv-weight loader/order fault; focus on Q/K repeat expansion semantics. |
| Q/K repeat-interleave candidate | Existing Qwen3.5 overlay with `VLLM_QWEN35_TILE_QK_REPEAT=0` | Tested TP1, TP2, TP4, and TP8 one-token sanity under repeat-interleave instead of tiled repeat. | TP4 changed from `.` to `,`; TP8 no longer produced the earlier catastrophic first-token behavior; TP1/TP2/TP4/TP8 returned `,`. | First correctness candidate; must still clear full-context warmups, strict prompt, and fixed-token tiers. |
| TP8 full-context repeat-interleave benchmark gate | Existing Qwen3.5 overlay with `VLLM_QWEN35_TILE_QK_REPEAT=0`, Q4_0 GGUF, `VLLM_DTYPE=half`, P2P-on | Ran the normal Qwen begin-think sequence on `.20`: eight warmups followed by `c1_128` uncapped strict. | Eight 2000-token warmups completed with backend decode TPS around `54.31` to `54.35`; `c1_128` strict stopped after 12 tokens with no visible answer, no think close, and `qwen_gate_valid=false`. | Reject as benchmark candidate; do not promote `c1_2000` / `c1_10000` until strict correctness is repaired. |
| TP8 full-context FP16/half GGUF benchmark gate | Same repeat-interleave Qwen3.5 overlay, FP16/half GGUF, `VLLM_DTYPE=half`, P2P-on | Ran the normal Qwen begin-think warmups on `.20`; strict started only after eight warmups. | Warmups completed but were deterministic garbage: repeated reasoning text, no visible answer, no think close, and identical output hash. Warmup 2-8 backend decode stayed around `49.687` to `49.744` TPS. | Reject as benchmark candidate; inspect GGUF config/materialization/model execution semantics before more throughput tiers. |
| Local-file GGUF speculator probe bypass | `transformers_utils/config.py` experimental mount | Skipped speculator detection for local GGUF files so the explicit Hugging Face config path could be used. | This moved the FP16/half local GGUF path past startup and into serving. It did not fix output semantics. | Promoted as launch diagnostic only; not a model correctness fix. |
| TP4 FP16/half GGUF smoke localization | Small-context TP4 launch, repeat-interleave, P2P-on, direct completion and chat probes | Tested raw one-token completion, direct chat, and raw synthetic benchmark prompt. | Raw `Hello` completion returned the coherent comma token. Direct chat returned an empty think section. Raw benchmark-prompt completion degenerated into a repeated phrase loop. | Promote as localization evidence; the failure is context/prefill semantics, not universal file unreadability. |
| Force-unquantized GGUF linear/embedding experiment | Env-gated `GGUFConfig` patch returning unquantized methods for linear and embedding layers | Tested whether FP16/half GGUF should bypass the GGUF quantized-linear abstraction entirely. | The same one-token pass and chat/benchmark-prompt failures remained. Runtime still exposed GGUF qweight handling in the active Qwen path. | Reject as fix; inspect Qwen overlay/load path and GatedDeltaNet state semantics instead. |
| Clean patch-bundle routing | Image entrypoint patch application and `/opt/vllm_patch_bundle` contents | Checked whether individual source mounts were overwritten by the patch bundle. | The entrypoint copies bundle files after mounts are present, so bundle contents can override individually mounted Qwen files. | Promotion runs must use a single coherent bundle and verify the active source file. |
| Direct QwenNext architecture launch | Clean bundle plus `Qwen3NextForCausalLM` architecture override | Tried to bypass the stale Qwen3.5 wrapper and use the clean QwenNext file directly. | Initialization failed before serving because the model received the outer Qwen3.5 config object without direct text fields such as `hidden_size`. | Reject as launch fix; needs a config adapter if this route is pursued. |
| Direct QwenNext text-config adapter and split-GDN reorder | Clean direct-next diagnostic bundle | Added the missing text-config adapter and split-GDN reorder route, then ran a raw `Hello` smoke. | The server reached inference and the reorder marker fired, but output was invalid byte/glyph garbage. | Reject as benchmark route; direct-next is not faithful for the `qwen3_5` dense model without deeper compatibility work. |
| Active FP16/half GGUF tensor scan | Low-level GGUF scanner | Verified sampled tensors in the active `Qwen3.6-27B-F16.gguf` file. | Key tensors, including embedding, lm-head, qkv, gate, SSM output, alpha, and beta, are `F16(1)`. | Promote dtype evidence; do not blame current garbage output on BF16 unless new evidence contradicts the scanner. |
| Qwen3.5 patched-bundle TP4 smoke | Older `qwen35-gguf-corechecksum` bundle plus repeat-interleave, P2P-on, small context | Tested FP16/half raw and chat behavior through the patched Qwen3.5 route. | Raw `Hello` produced repeated greeting text; direct chat timed out after the FLA native short-sequence warning and a shared-memory wait. | Reject as benchmark path; promote as evidence that the patched route still fails prefill/chat correctness. |
| Torch GDN prefill fallback | `VLLM_QWEN35_TORCH_GDN_PREFILL=1` under the same patched-bundle route | Tested whether replacing the custom GDN prefill path repaired semantics. | The hang disappeared, but raw output became repeated commas and chat returned only `<think>`. | Reject as fix; use only as a diagnostic contrast. |
| Minimal no-bundle Qwen3.5 launch | Experimental image with model/cache mounts and local-GGUF config bypass only | Tested whether the old Qwen3.5 diagnostic bundle could be removed entirely. | Startup failed before serving on an uninitialized GGUF embedding parameter. | Promote as startup-boundary evidence; extract only required embedding/lm-head materialization into the next minimal bundle. |
| F16 materialized qweight type correction | Copy of old Qwen3.5 bundle with materialized qweights changed from `WeightType.BF16` to `WeightType.F16` | Tested whether the active F16 GGUF file was being corrupted by a BF16 materialization label. | Server loaded with qweight type `1`, but raw output repeated greeting tokens, chat returned an empty think block, and longer chat echoed the prompt inside `<think>`. | Reject as semantic fix; keep as consistency requirement for future minimal bundles. |
| Minimal F16 Qwen3.5 bundle | Image-base `qwen3_5.py` plus F16 GGUF embedding/lm-head materialization and Q/K repeat-interleave | Tested whether only the startup fixes and one intentional Qwen3.5 correctness patch were enough. | Server loaded and answered raw/chat probes, but output was invalid byte/glyph garbage. | Reject as benchmark candidate; promote as clean semantic failure evidence. |
| Minimal F16 bundle plus logical GGUF shard order | Minimal Qwen3.5 bundle plus logical shard-order `gguf.py` patch | Tested whether logical shard ordering repaired the minimal FP16/half route. | Output remained invalid byte/glyph garbage. | Reject logical shard-order as sufficient. |
| Minimal F16 bundle plus old GGUF loader/registry without cache helpers | Minimal Qwen3.5 bundle, older `gguf_loader.py` and `registry.py`, logical shard patch | Tested whether old loader/registry behavior was required. | Engine failed before serving during hybrid KV/Mamba page-size setup. | Reject as serving path; promote cache-helper dependency for that loader route. |
| Minimal F16 bundle plus cache helpers and old GGUF loader/registry | Minimal Qwen3.5 bundle with `IsHybrid`/Mamba cache helper behavior, older loader/registry, logical shard patch | Tested whether old loader/registry plus cache helpers restored coherent output. | Server reached health, but raw `Hello`, chat `Hello`, and a short factual prompt produced invalid byte/glyph garbage. | Reject as benchmark candidate; startup/cache fixes do not solve semantic mapping. |
| Wrapper load delegation | Minimal F16 bundle changed to delegate wrapper `load_weights` into `self.language_model` after stripping `language_model.` | Tested whether the outer AutoWeightsLoader mapper caused the byte-garbage output. | Active materialization markers fired and raw `Hello` returned the same invalid byte/glyph prefix as before. | Reject as semantic fix. |
| llama.cpp CPU artifact smoke | CPU-only `llama-cli` against the same F16 GGUF file | Tested whether an independent GGUF reader could quickly prove file-level coherence. | The 51 GB F16 model timed out after 15 minutes before a useful answer. | Inconclusive; needs ROCm-capable llama.cpp or another practical independent runner. |
| llama.cpp ROCm artifact smoke | ROCm7.2 llama.cpp build in `build-rocm72-gfx906`, `GGML_HIP=ON`, `AMDGPU_TARGETS=gfx906` | Tested the same F16 GGUF with independent GPU-offloaded GGUF execution. | `llama-completion` returned coherent Qwen-style text for `Hello`: an empty think section followed by a normal greeting/help response. | Promote as artifact-coherence evidence; focus vLLM source path. |
| Tokenizer mapping audit | GGUF `tokenizer.ggml.tokens` versus HF tokenizer from the text-config path | Checked whether vLLM's bad token strings came from tokenizer ID drift. | Shared ID range matches for sampled IDs and strings such as `Hello`, `<think>`, `</think>`, `_manifest`, and `hh`. | Reject simple tokenizer-order mismatch. |
| TP4 F16 GGUF embedding/lm-head row diagnostic | Minimal F16 Qwen3.5 delegate source with env-gated row checksum overlay | Compared vLLM rank-local `token_embd.weight` and `output.weight` materialized rows against direct GGUF reader rows for sampled token IDs. | Sampled rows matched direct GGUF data on the owning TP ranks, but raw `Hello` still produced deterministic byte/glyph garbage. | Reject simple embedding/lm-head row-offset mismatch; trace layer-0 execution and GatedDeltaNet state next. |
| TP4 F16 GGUF layer-0 activity diagnostic | Minimal F16 Qwen3.5 delegate source with layer-0 projection/core/output summaries | Checked whether the bad raw `Hello` request was caused by a dead layer-0 path or zeroed GatedDeltaNet/output projection. | The request still produced deterministic byte/glyph garbage, but q/k/v/z, beta/alpha, GatedDeltaNet core output, output-projection input, and output-projection output were all nonzero on the actual request path. | Reject all-zero/dead-core theory; move to structured rank-filtered prefill/state comparison. |
| TP4 F16 GGUF benchmark-prompt rank0 trace | Same layer-0 diagnostic source with `VLLM_QWEN35_TRACE_RANK0_ONLY=1` | Sent the normal synthetic warmup prompt shape with one forced output token to classify prefill without running a full warmup. | The 431-token chat prefill returned only `;` after shared-memory broadcast waits; layer-0 rank0 projections, GatedDeltaNet core, and output projection were nonzero. | Reject current F16 GGUF path for warmups; active source target is semantic prefill/state/logit divergence, not missing layer activity. |
| TP2 F16 GGUF vLLM capacity test | Minimal delegate bundle, TP2, two GPUs, `MAX_MODEL_LEN=4096` | Checked whether F16 GGUF vLLM failure is TP4-specific. | Startup failed with `0.0 GiB` available KV cache memory after weight load. | Inconclusive for correctness; TP2 is not practical for this F16 diagnostic on 2x MI50. |
| vLLM GGUF partition-shape inspection | `MergedColumnParallelLinear`, `RowParallelLinear`, `GGUFLinearMethod` | Compared traced TP4 qkvz and `out_proj` shapes against vLLM partition rules. | qkvz local rows and `out_proj` packed local columns match expected TP4 sizes. | Reject simple loader-size mismatch; inspect per-rank content and offsets next. |
| Load-time checksum placement | GGUF loader and Qwen3.5 layer load hook | Tried to checksum qkvz, beta/alpha, conv, and `out_proj` at the end of `load_weights`. | qkvz and `out_proj` qweights were still uninitialized at that point. | Rejected as final diagnostic placement; promoted materialization-timing evidence. |
| Forward-time checksum trace | First-forward layer-0 checksum hook under TP4 P2P-on | Checked materialized qkvz, beta/alpha, conv, and `out_proj` tensors after GGUF quantized weights were usable. | qkvz materialized with physical shard order `[3, 0, 1, 2]`, while the deterministic `Hello` probe still returned `.` as top token. | Promoted source boundary evidence; not a serving fix. |
| Logical GGUF shard layout around GDN | `in_proj_qkvz`, `in_proj_ba`, q/k/v/z split and repeat, beta/gate tensors, GDN output projection | Still active after scanner and shard-order patch results. | First divergence appears inside the GatedDeltaNet path after identical layer-0 input normalization, and the current logical-order patch did not repair it. | Active source hypothesis. |
| Combined release/GGUF dense overlay | Release runtime overlay plus proven GGUF Qwen3.5 tensor fixes | Mounted as one coherent patch bundle on the experimental image and official release image. | Dense TP8 became coherent and strict-valid, but backend TPS stayed around `60.7` strict, `61.4` on `c1_2000`, and `58.0` on `c1_10000`. | Promote correctness; reject benchmark performance. |
| Official release image control | Official v0.2 runtime image plus combined release/GGUF overlay | Checked whether the experimental GGUF image base caused the dense throughput gap. | Official image results matched the experimental image band: strict `60.656`, `c1_2000` `61.399`, `c1_10000` `57.966`. | Reject image-base drift. |
| Release attention subclass merge | GGUF-correct Qwen3.5 source plus release `Qwen3_5Attention` subclass/routing | Tested whether missing release attention routing explained the decode gap. | Full ladder stayed in the same band: strict `60.764`, `c1_2000` `61.362`, `c1_10000` `57.965`. | Reject attention-class routing as the missing lever. |
| F16 merged-single-matmul GGUF loader patch | Experimental `GGUF_F16_MERGED_SINGLE_MATMUL` loader path | Tried to materialize F16 GGUF merged shards once in logical order and clear shard maps. | Marker did not fire for the important dense linears; `in_proj_qkvz`, `in_proj_ba`, `conv1d`, and `out_proj` already load as normal `weight` params. | Reject assumed GGUF `qweight` matmul overhead for those linears. |
| ConditionalGeneration architecture override | Architecture override to `Qwen3_5ForConditionalGeneration` with release-attention GGUF bundle | Matched release-style attention block size `400` and Mamba padding `2.17%`. | Warmup decode remained about `60.59` TPS and first-request prefill was much worse. | Reject as dense GGUF performance route. |
| No-trace release-attention GGUF bundle | Release-attention GGUF bundle with `_qwen35_trace_tensor(...)` calls removed from forward | Tested whether disabled trace hooks and repeated environment checks were the decode ceiling. | Strict remained valid, but full ladder stayed in the same band: strict `60.678`, `c1_2000` `61.289`, `c1_10000` `57.896`. | Reject trace-hook overhead as the missing performance lever. |
| Clean HF-weight dense TP8 release-overlay control | Official release image, clean release patch bundle, complete HF-weight dense snapshot, P2P-on, TP8, normal benchmark ladder | Checked whether the host lane, official image, release overlay, P2P-on state, or harness explained the GGUF decode gap. | The HF-weight path reproduced the dense band: strict `69.652`, `c1_2000` `70.202`, and `c1_10000` `65.952` backend TPS. | Promote as control evidence; reject host/image/harness drift as the GGUF performance explanation. |
| FLA / causal-conv1d fallback clue | Qwen3.5 linear-attention startup logs in coherent dense GGUF runs and the HF-weight control | Checked startup/runtime logs for the traced, no-trace, and HF-weight release-overlay paths, then inspected the image source. | The warning comes from GGUF loader dummy-model construction, not from a proven request-time Transformers fallback. The clean HF-weight control still proves the host/image/harness are capable of release-band decode on the same lane. | Reject as standalone package-install work; profile actual serving-path timing instead. |
| Fused-GDN GGUF projection parity | Experimental bundle using the proven GGUF tensor conversions plus HF release-style fused qkv/z/b/a `in_proj_qkvz` and beta/alpha shard IDs `4` / `5` | Tested whether the split `in_proj_ba` path was the missing decode-performance lever. | Initial short-timeout smoke looked hung, but a patient rerun served direct completion, passed short-chat coherence, and completed the full ladder: strict `63.201`, `c1_2000` `63.836`, `c1_10000` `60.169` backend TPS. | Promote as source evidence; reject as benchmark promotion because it remains below the FP16/HF release band. |
| Native interleaved SwiGLU for F16 GGUF | `qwen2_moe_interleaved_swiglu_20260608.py` guard relaxed only under `VLLM_GFX906_QWEN_MLP_INTERLEAVED_SWIGLU_ALLOW_GGUF_F16=1` | Tested whether the dense GGUF gap came from GGUF `quant_config` disabling the release MLP native SwiGLU extension. | Official image run reached health, logged 8 extension loads and 512 layer/rank enablements, and completed the full ladder: strict `61.053`, `c1_2000` `61.780`, `c1_10000` `58.266` backend TPS. | Reject as standalone performance fix; promote evidence that MLP native extension is not enough and GatedDeltaNet / FLA path parity remains the active target. |
| Fused-GDN plus native GGUF SwiGLU | Combined coherent patch bundle with fused qkv/z/b/a GatedDeltaNet projection and explicit F16 GGUF native SwiGLU guard | Tested whether the two partial dense levers were additive under the official release image, P2P-on, TP8, and normal warmups -> strict -> fixed-token ladder. | The run completed eight warmups, passed strict validity, and produced strict `63.679`, `c1_2000` `64.353`, and `c1_10000` `60.617` backend TPS. | Promote as current best dense GGUF source baseline; reject as benchmark promotion because it remains below the FP16/HF release band. |
| ConditionalGeneration with fused-GDN plus native GGUF SwiGLU | Same combined source bundle as the current best CausalLM path, but with `Qwen3_5ForConditionalGeneration` in `--hf-overrides` | Tested whether matching the HF-control architecture wrapper after the latest GGUF fixes closed the dense decode gap. | The run was coherent and strict-valid, but slower than CausalLM: strict `63.003`, `c1_2000` `63.656`, and `c1_10000` `60.109` backend TPS. | Reject architecture override as the missing dense GGUF lever; keep the CausalLM combined bundle as baseline. |
| HF-control versus best-GGUF log comparison | Existing logs from the clean HF-weight dense control and `GGUF-070` best-GGUF run | Compared launch geometry, graph capture, fallback messages, and bundle hashes without starting a new container. | Both paths used TP8, P2P-on, `MAX_MODEL_LEN=131072`, chunked prefill, graph-capture mode, capture sizes `[1, 2, 4]`, and about 2 seconds of graph capture. HF used `Qwen3_5ForConditionalGeneration`, attention block size `400`, and Mamba padding `2.17%`; GGUF used `Qwen3_5ForCausalLM`, attention block size `208`, Mamba padding `4.26%`, and `load_format=gguf`. | Reject graph setup as a standalone explanation. Wrapper/page geometry differs, but the modern ConditionalGeneration rerun already rejected wrapper matching as the missing lever; profile the current CausalLM baseline hot path next. |
| Python timing hooks in compiled forward | Timing-only copies of the current best GGUF bundle and clean HF bundle | Added optional Python timing around projection, `torch.ops.vllm.gdn_attention_core`, norm/gate/rearrange, and output projection. | The synchronized variant failed startup because Dynamo rejected `torch.cuda.synchronize()` inside the compiled forward. The no-sync variant failed startup because Dynamo rejected `time.perf_counter()` inside the compiled forward. | Reject Python timing in the compiled model path; use external ROCm profiling, compiler-safe C++/custom-op markers, or built-in tracing instead. |
| `gdn_attention_core` custom-op wrapper timing | Copied installed QwenNext source, current best MoE GGUF TP4 graph path, and HF-weight TP4 comparator | Timed the registered direct custom-op body rather than inserting timing inside the compiled model forward. | Synchronized timing invalidated HIP graph capture, but the no-sync wrapper reached health on both GGUF and HF. Both faithful two-token branch requests returned `Thinking Process` and added `240` timing records. Excluding the shared layer-0 first-request/prefill outlier, GGUF averaged `1,039,559` ns and HF averaged `1,049,320` ns across per-layer average `last_ns` values; max non-layer-0 `last_ns` was `1,344,764` for GGUF and `1,376,768` for HF. | Promote as request-window timing evidence; reject GDN core timing as the current MoE GGUF bottleneck. Next profile projections, MoE/shared-expert execution, residual/norm, graph/kernel selection, logits, and whole decode-loop scheduling. |
| FusedMoE direct custom-op wrapper timing | Copied installed `DefaultMoERunner` source, current best MoE GGUF TP4 graph/eager paths, and HF-weight TP4 comparators | Timed `torch.ops.vllm.moe_forward` and `torch.ops.vllm.moe_forward_shared` registered bodies around `layer.runner.forward_impl(...)`. | Graph-mode GGUF reached health and returned `Thinking Process`, but live requests added zero timing rows because the Python body ran during profiling/capture and not during graph replay. Eager-mode GGUF and HF each added `480` request-window timing rows. Token-sized shapes were at parity: `(1, 2048)` averaged `645,803` ns for GGUF and `649,962` ns for HF; `(32, 2048)` averaged `678,506` ns for GGUF and `694,144` ns for HF. The apparent `(15, 2048)` gap was layer-0 first-request noise; excluding layer 0, GGUF averaged `808,102` ns and HF averaged `801,966` ns. | Promote as timing-boundary evidence; reject the steady unquantized FusedMoE runner body as the primary current MoE GGUF bottleneck. Next use graph-visible or whole-decode profiling rather than another unchanged `DefaultMoERunner` timing pass. |
| Post-forward logits and sampler timing | Copied installed `LogitsProcessor` and v1 `Sampler` source with env-gated synchronized timing | Timed `lm_head.quant_method.apply(...)`, TP logits gather, total logits processing, `Sampler.sample(...)`, and total sampler forward on the faithful two-token branch request. | GGUF graph mode reached health after recompilation and HF graph mode required the same KV-layout bypass template as earlier successful comparators. Both returned `Thinking Process` and added `60` timing rows. GGUF versus HF averages were: `lm_head_apply` `393,201` ns versus `373,102` ns; `gather_logits` `236,504` ns versus `230,403` ns; `get_logits_total` `735,715` ns versus `727,748` ns; `sampler_forward_total` `250,274` ns versus `232,938` ns; `sample_inner` `76,387` ns versus `79,262` ns. | Reject logits processing and sampling as the primary MoE GGUF throughput bottleneck. Continue toward graph-visible whole-model timing, structural graph comparison, or model-runner scheduling/projection regions outside the already-timed direct ops. |
| ROCm profiler API-server wrapper | ROCm 7.2.1 `rocprof --hip-trace --timestamp on --stats` around the official image entrypoint with the `GGUF-070` bundle | Tested whether a broad external profiler wrapper could capture useful dense GGUF hot-path timing without source changes. | The server loaded and entered graph compilation, but the profile captured startup/parent-process HIP API activity only. No request-level decode window, no kernel-summary CSV, and no parent-process `hipLaunchKernel` records were produced. | Reject API-server-wrapper profiling as hot-path evidence. Use request-window, worker-targeted, profile-input, or single-process/offline profiling instead. |
| ROCm profiler PID attach | ROCm 7.2.1 `rocprofv3 --attach` against the TP0 worker process during fixed-token `c1_2000` requests on the current best dense GGUF bundle | Tested whether worker-targeted request-window attach could capture decode kernels while preserving the official image, P2P-on, TP8, and `GGUF-070` source baseline. | The benchmark requests completed in the expected current-best GGUF band, with backend decode TPS `64.386`, `64.362`, and `64.217`. A tiny Torch matmul sanity run proved `rocprofv3` can emit kernel/HIP CSV and JSON when the missing `libdw.so.1` is supplied read-only, but PID attach to the vLLM TP0 worker still emitted no files even after relaunching the server with that mount and using `-f csv json`. | Reject this PID-attach shape as a profiling route until artifact emission is proven. Next profiler work should use an offline/single-process path, profile-input workflow, known-good worker launcher shape, or compiler-safe lower-level markers. Account for the release image's missing `libdw.so.1` if using `rocprofv3`, without mutating the image. |
| ROCm profiler launch-time wrapper controls | ROCm 7.2.1 `rocprofv3` around tiny direct, `spawn`, and `fork` Torch GPU workloads | Checked whether `rocprofv3` can capture child-process GPU work when it wraps the parent from startup. | Direct execution and a Python `multiprocessing` `spawn` child both emitted kernel/HIP CSV and JSON. The `fork` child did not emit profile files. | Promote as profiler boundary evidence. vLLM uses `spawn`, so launch-time wrapping can follow workers in principle; attach failure is not the only possible route. |
| ROCm profiler launch-time full-server wrapper | `rocprofv3` around the full current best dense GGUF vLLM server with read-only `libdw.so.1` mount | Tested whether launch-time profiling could capture vLLM spawned worker kernels while preserving the release-like P2P-on TP8 `GGUF-070` path. | The server reached GGUF tensor conversion and graph compile, then faulted during persistent all-reduce startup with GPU memory access faults across workers. No request was sent and no profile artifacts were emitted. | Reject unchanged full-server wrapping as unsafe for comparable profiling. Use lower-impact markers, a reduced offline/single-worker reproducer, or non-AR diagnostic profiling only for source localization. |
| ROCm profiler launch-time no-AR diagnostic | `rocprofv3` around the current best dense GGUF vLLM server with GFX906 Persistent-AR and mutable RowParallel AR disabled | Tested whether removing the Persistent-AR startup fault would let a launch-time profiler capture a request. | The server eventually reached health, but the collection window mainly covered startup. A short capped request produced about `12.37` vLLM decode TPS, hit `finish_reason=length`, failed `qwen_gate_valid`, and still emitted no useful profile artifacts when the container was stopped. | Reject as profiling and benchmark evidence. It is non-comparable to the release-like P2P-on lane, much slower, misses the request window, and remains fileless in the tested stop path. |
| ROCTX marker trace control | `torch.cuda.nvtx.range_push/pop` in a tiny compiled Torch matmul under `rocprofv3 --marker-trace --kernel-trace` | Checked whether the release image and profiler can emit marker and kernel files at all. | With read-only `libdw.so.1` supplied, the run emitted kernel CSV/JSON and marker API CSV/JSON; the `tiny_marker` range appeared in the marker trace. | Promote as profiler capability evidence. Marker tracing works in a simple process shape, but not yet in the full model path. |
| Raw Python markers in Qwen forward | Copied current best dense GGUF bundle with raw `torch.cuda.nvtx.range_push/pop` around layer-0 GatedDeltaNet boundaries | Tested whether simple marker ranges could be inserted into the compiled model forward. | The server failed during AOT capture before readiness. Dynamo rejected `range_push` because it returns a non-Tensor. | Reject raw Python marker insertion inside Qwen forward. It is incompatible with vLLM's fullgraph/AOT path. |
| Tensor-returning Python custom marker op | Tiny `torch.library` marker op plus copied Qwen marker bundle | Tested whether returning a Tensor from a custom op would preserve fullgraph tracing and provide marker events. | The tiny fullgraph control compiled and emitted `custom_marker_1` / `custom_marker_2` under `rocprofv3`, but warned about input/output aliasing. The model bundle failed runtime shape checks with `wrong number of dimensions1`; the alias-annotated schema variant failed the tiny Inductor control. | Reject ad hoc Python custom marker ops for the model path. A future marker route needs a proper lower-level custom op with correct alias metadata or a reduced offline reproducer that avoids perturbing the full vLLM AOT graph. |
| Built-in PyTorch profiler record-function ops | `torch.ops.profiler._record_function_enter/_exit` tiny compiled controls | Tested whether built-in record-function ops could act as compiler-safe marker boundaries without a custom extension. | Keeping the returned handle alive failed Inductor graph lowering; leaving the call unused compiled but was removed and emitted no marker API records under `rocprofv3`. | Reject record-function hooks for the compiled GGUF Qwen path. They either break lowering or fail to survive as trace markers. |
| `GGUFLinearMethod.apply()` hot-path diagnostic | Disposable `gguf.py` overlay with `GGUF_APPLY_DIAG` logging | Sampled graph profiling and one short coherent chat request on the official image plus release-attention GGUF bundle. | Logged calls were `ParallelLMHead`; sampled hot path did not show internal Qwen block linears using `GGUFLinearMethod.apply()`. The `transformers` fast-path warning was traced to GGUF loader dummy-model name mapping rather than proven serving hot path. | Reject broad internal GGUF matmul overhead as the current dense-gap explanation; inspect lm-head, GatedDeltaNet/core timing, graph capture, and CausalLM versus HF release path. |
| lm-head-only unquantized method | `gguf.py` overlay forcing only `lm_head` to `UnquantizedEmbeddingMethod` | Tested whether the visible `ParallelLMHead` GGUF apply path was the missing dense TPS lever. | Marker fired on all 8 TP ranks and the normal ladder stayed strict-valid, but results remained in the prior band: strict `60.676`, `c1_2000` `61.309`, `c1_10000` `57.888` backend TPS. | Reject isolated lm-head method selection as the missing lever; inspect GatedDeltaNet/core timing, graph capture, sampler/logits, and CausalLM-versus-HF release execution. |
| F16 GGUF row-parallel residual pre-fold | Copied current-best dense GGUF bundle with `VLLM_GFX906_MLP_DOWN_LLMM1_RESIDUAL_PREFOLD` allowed for GGUF F16 and the native row-parallel residual extension supplied from the bundle | Tested whether the old HF release residual pre-fold path was blocked only by `quant_config == gguf`. | The path enabled at startup and completed the normal ladder strict-valid, but results were strict `63.631`, `c1_2000` `64.296`, and `c1_10000` `60.554` backend TPS. | Reject as a dense GGUF performance lever. It is slightly slower than fused-GDN plus native GGUF SwiGLU without residual pre-fold. |
| MoE GGUF artifact readiness | Qwen3.6 35B-A3B F16 GGUF conversion and metadata audit | Produced the MoE F16 GGUF artifact and checked architecture, file type, block/expert metadata, and sampled tokenizer IDs against the HF tokenizer. | The file reports `general.architecture=qwen35moe`, F16 data, 41 blocks, 256 experts, 8 active experts, and tokenizer metadata. Sampled token IDs matched HF for representative strings. | Promote artifact readiness only; serving still requires source support. |
| Transformers GGUF parser architecture map | `modeling_gguf_pytorch_utils.py` / GGUF architecture dispatch | Tested baseline parser behavior and a narrow experimental alias. | Baseline rejects `qwen35moe` as unsupported. Aliasing it to older Qwen3 MoE bypasses the first error but sends the model to the wrong config/model path. | Add a real `qwen35moe` / Qwen3.5-MoE parser route instead of aliasing to Qwen3 MoE. |
| vLLM GGUF model-loader mapping | `model_executor/model_loader/gguf_loader.py`, model registry, tensor-name mapping | Probed local and ai-infos images for Qwen3.5-MoE GGUF support. | ai-infos has Qwen3.5/Qwen3.5-MoE HF model/config source files, but its GGUF parser still rejects `qwen35moe`; local vLLM also lacks the route. | Port/reconstruct Qwen3.5-MoE GGUF mapping before any MoE benchmark ladder. |
| Qwen3.5-MoE FusedMoE weight loading | `model_executor/layers/quantization/gguf.py`, FusedMoE load methods, Qwen3-MoE versus Qwen3.5-MoE config | Added experimental F16/unquantized embedding and MoE method shims after the alias got past architecture rejection. | The run progressed into expert loading, then failed with expert-shard range/shape errors because Qwen3.6 35B-A3B MoE uses Qwen3.5-MoE-style dimensions such as `moe_intermediate_size=512`. | Reject plain-Qwen3-MoE aliasing. Correct support must preserve Qwen3.5-MoE text config, linear-attention layer types, and expert tensor sharding. |
| Rotary import fallback | `model_executor/layers/rotary_embedding/common.py` | Added a narrow fallback around missing `flash_attn.ops.triton.rotary` during the experimental alias path. | It allowed the wrong-architecture alias path to progress farther, but it did not make the model serve and is not sufficient source support. | Keep only as a diagnostic startup observation unless a correct Qwen3.5-MoE path hits the same import edge. |
| Qwen3.5-MoE parser-only name-map probe | `modeling_gguf_pytorch_utils.py`, gguf-py `qwen35moe` tensor-name map, Qwen3.5-MoE text config | Added a no-tensor-load metadata shim and mapped a text-only dummy model against the GGUF name map. | Parsed expected config fields and mapped `663` of `693` text parameters. The unmapped set was the `30` `model.layers.*.linear_attn.dt_bias` parameters; these correspond to GGUF `blk.*.ssm_dt.bias`. | Promote as the next narrow patch target: add `ssm_dt.bias` mapping, keep text-only Qwen3.5-MoE config, then test gate/up expert merge and FusedMoE TP sharding. |
| Qwen3.5-MoE text GGUF route | `model_executor/model_loader/gguf_loader.py`, `model_executor/models/registry.py`, `model_executor/models/config.py`, `model_executor/models/qwen3_5.py`, `modeling_gguf_pytorch_utils.py` | Added a real experimental route for `qwen35moe`: text-model registry selection, parser config, manual `ssm_dt.bias` mapping, expert gate/up/down stack loading, `conv1d.weight` shape guard, hybrid cache verification, and text-class Mamba state helpers. | The route moves past unsupported architecture, parser gaps, expert-shard shape failures, and Mamba page-size setup. It reaches API health under eager mode with the same TP4 full-BAR/P2P-on intent. | Promote as source-bridge evidence only. It is not a coherent serving path. |
| Qwen3.5-MoE graph-capture path | Same Qwen3.5-MoE text route, graph mode, TP4, P2P-on | Tested the serving path without eager mode after hybrid cache setup and Mamba state shape helpers were added. | Model load succeeded and the page-size error was gone, but graph capture failed with a HIP stream-capture error inside the slow F16 GGUF MoE fallback. | Reject graph serving until F16/unquantized GGUF MoE can use a capture-safe fast path instead of `GGUFMoEMethod` slow fallback. |
| Qwen3.5-MoE eager request path | Same Qwen3.5-MoE text route, `--enforce-eager`, M-RoPE stripped from text config, text-class Mamba state-copy helper | Tested direct completion and 8-token streaming completion after API health. | Initial eager request exposed accidental M-RoPE, then missing state-copy helper. After both were fixed, HTTP requests were accepted but emitted no client-visible bytes before timeout, including a 240-second 8-token streaming probe. | Reject coherence and benchmark testing. Next source target is a reduced first-token reproducer plus a capture-safe fast/unquantized F16 GGUF MoE path. |
| Qwen3.5-MoE unquantized GGUF method selection | `model_executor/layers/quantization/gguf.py` plus the Qwen3.5-MoE text route | Tested two ways to route F16 GGUF MoE expert layers away from the slow `GGUFMoEMethod` fallback. Prefix-based skipped-layer selection did not catch the expert layers and the slow fallback warning remained. An explicit diagnostic `VLLM_GFX906_GGUF_FORCE_UNQUANT_MOE=1` switch selected `UnquantizedFusedMoEMethod` for the expert layers. | The explicit switch let the TP4 full-BAR/P2P-on path load, compile, graph-capture, and serve. It does not produce coherent output. | Promote as graph-startup source evidence; reject as model-correctness or benchmark evidence. Method selection is now past the stream-capture blocker, but semantic correctness remains unresolved. |
| Qwen3.5-MoE text M-RoPE restoration | `model_executor/models/qwen3_5.py` and the experimental text config | Restored the original text config M-RoPE fields and added a minimal text-only `supports_mrope` / `get_mrope_input_positions` implementation. | The server reached health in graph mode, but a one-token `Hello` completion returned `amed`, and a 16-token math completion timed out after `180` seconds with zero response bytes. | Reject M-RoPE restoration as sufficient. The next source target is an offline or reduced first-token comparison of router logits, selected experts, expert tensor orientation, recurrent state-copy, and final logits against a coherent control. |
| Qwen3.5-MoE `dt_bias` GGUF permutation | `model_executor/models/qwen3_5.py` load path for `linear_attn.dt_bias` | Compared HF safetensors and GGUF tensor data for layer 0. Router, shared expert, expert gate/up/down slices, and `conv1d.weight` matched. `blk.0.ssm_dt.bias` matched HF `linear_attn.dt_bias` only after inverting even-then-odd packing. | Adding the inverse permutation before loading `linear_attn.dt_bias` preserved graph startup but did not repair output: `Hello` still returned `amed`, and a 16-token math prompt timed out. | Keep the inverse permutation as a required correctness patch, but reject it as sufficient. Continue with reduced first-token comparisons for router logits, expert materialization, recurrent state, and final logits. |
| Qwen3.5-MoE full linear-attention GGUF inverse layout | `model_executor/models/qwen3_5.py` load path for Qwen3.5 linear-attention tensors | Compared llama.cpp's Qwen3.5 converter logic against HF safetensors and the F16 MoE GGUF. The inverse V-head reorder reconstructs HF layout for QKV V rows, Z, alpha, beta, `dt_bias`, `A_log`, conv V channels, and out-proj columns within FP32 comparison noise. The first implementation only covered the direct load branch; the corrected implementation also covers the stacked load branch for fused `in_proj_qkvz` and `in_proj_ba`. | Both implementations preserved graph startup and health, but deterministic probes still failed. With stacked coverage, `Hello` returned `_`, and a 16-token math prompt timed out after `180` seconds with no response bytes. | Keep the inverse bridge as required source evidence, but reject it as sufficient. The active blocker is deeper than known linear-attention tensor packing: compare router logits, selected experts, expert outputs, recurrent state update/copy, output norm/gate, and final logits against a coherent control. |
| Qwen3.5-MoE request-gated final-logit trace | `model_executor/models/qwen3_5.py` final-logit trace hook plus `VLLM_GFX906_GGUF_TRACE_GATE_FILE` | Copied the current MoE GGUF source bundle on `.20` NVMe and added a gate file so final-logit tracing waits for a client request instead of writing during startup/profile paths. | The instrumentation is staged and verified in source. It successfully captured a real eager-mode one-token request on `.20`, but that request still returned `_` and the final-logit top-token sequence was bad. | Promote as diagnostic infrastructure and request-level evidence only. Do not interpret startup/profile tensors as prompt evidence, and do not run warmups until the deterministic probe is coherent. |
| Qwen3.5-MoE release-image direct mounts | Official v0.2 image, direct-mounted Qwen3.5-MoE GGUF source files, TP4, P2P-on | Ran staged `.20` MoE F16 GGUF through the release image rather than the source-debug lane. Initial config-path and text-config wrapper launches failed before model setup. The direct-mount launch then loaded weights and reached compile/cache initialization. | Engine startup failed before API health with hybrid KV-cache layout ambiguity for tensor shape `torch.Size([2, 2, 528, 1, 256])`. Repeating with `--enable-prefix-caching --mamba-cache-mode align` kept `mamba_cache_mode='align'` but failed with the same assertion. | Reject as serving path for now; promote as narrowed release-image startup blocker. Next source work should target the hybrid attention/Mamba KV layout discriminator or use an eager diagnostic only to capture the gated one-token request trace. |
| Qwen3.5-MoE eager request-level trace | Official v0.2 image, direct mounts, TP4, P2P-on, `--enforce-eager`, request-gated final-logit trace | Bypassed graph-cache startup to see whether the release-image route could serve one deterministic request and capture final-logit evidence. | Eager mode reached API health. A gated one-token raw `Hello` completion returned `_` with `finish_reason=length`. Four worker traces agreed on hidden shape `(1, 2048)`, logits shape `(1, 248320)`, hidden norm about `142.27`, and a shared top-token sequence headed by token id `62`. | Reject eager mode as a coherence fix; promote request-level final-logit evidence. The failure is upstream of sampling. Compare router logits, recurrent state, output norm/gate, and final logits against HF or llama.cpp before running more server warmups. |
| Qwen3.5-MoE llama.cpp raw first-token control | ROCm llama.cpp `llama-completion`, same `.20` F16 MoE GGUF artifact, four-device layer split, raw `Hello`, one generated token | Checked whether the staged GGUF artifact itself can generate a coherent raw first token outside vLLM. | The raw control generated `Hello,`; comma is token id `11` in the shared tokenizer. A default chat-template control also produced coherent Qwen-style `<think>` output. | Promote as artifact coherence control. The source target is vLLM's Qwen3.5-MoE GGUF execution path, not the GGUF file or tokenizer. |
| Qwen3.5-MoE watched token logits | `VLLM_GFX906_GGUF_TRACE_WATCH_IDS` in the copied request-trace bundle | Recorded logits and ranks for comma token id `11`, underscore token id `62`, and the bad top-token set during the `.20` eager one-token vLLM request. | All four TP workers ranked underscore id `62` first with logit `12.867188`; comma id `11` had logit `8.882812` and rank `25`. | Promote as exact final-logit divergence. The next source comparison should move upstream from `compute_logits` into final hidden state formation, output norm/gate, recurrent state update/copy, and MoE/router/expert outputs. |
| Qwen3.5-MoE TP8 watched token discriminator | Same watch-id bundle, official v0.2 image, eager mode, P2P-on, `TP_SIZE=8` | Checked whether increasing tensor parallel degree from TP4 to TP8 repairs the raw first-token corruption. | TP8 still returned `_`. All eight workers ranked underscore id `62` first with logit `13.000000`; comma id `11` ranked `20` with logit `9.000000`. | Reject TP-degree flip as a fix. The bug is general to the current vLLM Qwen3.5-MoE GGUF execution path, not a TP4-only sharding issue. |
| Qwen3.5-MoE HF-weight watched token control | Clean trace overlay from the official v0.2 Qwen3.5-MoE source, local HF snapshot, TP4, P2P-on, eager mode, same raw one-token `Hello` probe | Checked whether the same vLLM model code and reduced request are coherent with HF weights. | HF-weight vLLM returned comma. All four TP workers ranked comma id `11` first with logit `15.039062`; underscore id `62` ranked `369` with logit `5.906250`. Hidden norm was about `114.72`, versus about `142.27` in the GGUF trace. | Promote as shared-code control evidence. The active failure is not sampler behavior, the raw prompt, or TP4 sharding; compare GGUF versus HF materialization/state boundaries before final logits. |
| Qwen3.5-MoE boundary trace | Copied GGUF and HF watch-id bundles with request-gated tensor summaries in `Qwen3_5GatedDeltaNet.forward()` and model final hidden | Compared the same raw one-token request through GGUF and HF under TP4, P2P-on, eager mode, and the official v0.2 image. | GGUF returned `_`; HF returned comma. Layer-0 GatedDeltaNet input norm was GGUF `91.739395` versus HF `46.638695`; `mixed_qkvz` was `220.456894` versus `113.987366`; `z_pre_reshape` was `107.963158` versus `55.807575`; final hidden was `142.269989` versus `114.722717`. | Promote as the current first measured divergence. The next source trace should move one step earlier into the decoder-layer `input_layernorm` input/output and loaded norm parameter. |
| Qwen3.5-MoE Qwen35 norm-minus-2 check | Copied the GGUF boundary bundle and changed only `Qwen35MoeTensorProcessor` non-SSM norm handling from `weights - 1` to `weights - 2` | Tested whether the 2x layer-0 scale came directly from loading HF-style norm weights into vLLM `GemmaRMSNorm`, which applies `1 + weight`. | The live container showed the edited Qwen35 processor line, but the response and traces were unchanged: output `_`, final hidden norm `142.269989`, token id `62` rank `1`, and comma id `11` rank `25`. | Reject blind processor-line edits. Trace actual decoder-layer norm inputs/outputs and params to determine whether the active tensors bypass the processor, are later overwritten, or diverge before the norm. |
| Qwen3.5-MoE runtime norm-offset fix | `qwen35moe-text-gguf-runtime-norm-offset-fix-20260626`; env `VLLM_QWEN35_GGUF_RUNTIME_NORM_OFFSET_FIX=1` | Added a GGUF-only runtime correction that subtracts `1.0` from loaded `GemmaRMSNorm` parameters before the runtime applies `1 + weight`. Compared GGUF and HF decoder-layer traces under TP4, P2P-on, eager mode, and the official v0.2 image. | GGUF and HF enter layer 0 with the same pre-norm embedding norm `0.610113`. The corrected GGUF post-input-layernorm norm is `46.637737`, matching HF `46.638695`; corrected final hidden is `114.727249`, matching HF `114.722717`. Raw `Hello` changes from `_` to comma, and a short no-trace prompt returns coherent text. | Promote as the MoE GGUF coherence repair candidate. Do not promote throughput yet; build a clean benchmark overlay from this fix and run deterministic probes, warmups, `c1_128` uncapped strict, `c1_2000`, and `c1_10000`. |
| Qwen3.5-MoE norm-fix 131K benchmark path | `moe35b_gguf_tp4_normfix_graph_131k_dot20_*` and `moe35b_gguf_tp4_normfix_eager_131k_dot20_*` launch shapes | Tested the norm-offset fix with release-length `MAX_MODEL_LEN=131072`, TP4, P2P-on, release MoE NCCL settings, and the existing benchmark harness. Graph mode was tested first, then eager mode after graph startup failed. | Graph mode failed before health with the hybrid KV-cache layout assertion for shape `torch.Size([2, 2, 528, 1, 256])`. Eager mode passed raw `Hello`, then completed eight warmups at `12.455`-`14.060` backend TPS with repeated no-close reasoning-loop output. The uncapped strict request was stopped as runaway/no-close after live metrics showed roughly 3K generated tokens beyond the warmup pattern. | Reject as benchmark candidate. Do not rerun eager ladders unchanged. Next source work is graph-mode hybrid cache layout repair and benchmark-prompt/think-close behavior under the norm fix. |
| Qwen3.5-MoE graph hybrid KV-layout bypass | `qwen35moe-text-gguf-runtime-norm-offset-kvfix-20260626`; env `VLLM_GFX906_ASSUME_HYBRID_ATTENTION_LEADING_KV=1` with `VLLM_QWEN35_GGUF_RUNTIME_NORM_OFFSET_FIX=1` | Added a narrow env-gated bypass in `_update_hybrid_attention_mamba_layout` for the ambiguous attention KV tensor shape where both the KV dimension and block count are `2`. Tested the official v0.2 image, TP4, P2P-on, `MAX_MODEL_LEN=131072`, release MoE TP4 NCCL settings, GGUF loader/tokenizer/config route, graph mode, raw `Hello`, eight warmups, and uncapped strict. | Graph mode reached health and raw `Hello` returned comma. The normal warmups improved to `73.579`-`73.655` backend TPS, but every warmup ended at `finish_reason=length`, `2000` generated tokens, `9449` reasoning chars, `0` visible/answer chars, no think close, `qwen_gate_valid=false`, and the same output hash. The uncapped strict request did not stop and was cut off after the live metric estimate crossed `60071` generated tokens. | Promote as graph-startup and decode-speed evidence only. Reject as benchmark path because strict does not close and fixed-token tiers were not run. Next source target is benchmark-prompt/think-close behavior plus MoE GGUF hot-path throughput under graph mode. |
| Qwen3.5-MoE shortest-prompt comparator | Same GGUF norm-offset plus KV-layout graph route and a clean HF-weight TP4 eager comparator on `.20` | Swept GGUF benchmark prompts with `prompt_repeat` values `0`, `1`, `4`, `8`, `16`, and `32` at `max_tokens=128`, then ran uncapped strict for GGUF `prompt_repeat=0` and the same shortest prompt on the HF comparator. | GGUF ranked `Here` first on the shortest prompt, produced reasoning-only capped outputs for every repeat length, and timed out uncapped after an estimated `18019` generated tokens. HF also did not close inside the 128-token cap, but uncapped HF closed after `2957` generated tokens with `finish_reason=stop`, `qwen_gate_valid=true`, `10125` reasoning chars, and `3930` visible answer chars. | Promote as semantic split evidence. Reject prompt length, begin-think proxy behavior, and capped warmup invalidity as sufficient explanations. Next compare GGUF versus HF around the reasoning-to-answer delimiter logits/state and stop transition. |
| Qwen3.5-MoE thinking-disabled transition probe | Same GGUF norm-offset plus KV-layout graph route on `.20`; direct vLLM chat with `chat_template_kwargs.enable_thinking=false` | Compared default thinking-enabled direct chat against a thinking-disabled diagnostic on the same benchmark prompt. | Default thinking-enabled `max_tokens=128` returned `finish_reason=length`, `621` reasoning chars, and `0` content chars. Thinking-disabled `max_tokens=128` returned `0` reasoning chars and `700` content chars; thinking-disabled `max_tokens=512` returned `0` reasoning chars and `2158` content chars, but the tail became repetitive short statements. Tokenizer inspection pins `<think>` to id `248068` and `</think>` to id `248069`. | Promote as transition-boundary evidence. Reject as benchmark path. The active source target is not generic content generation, parser-only routing, or prompt length; it is the thinking-enabled GGUF failure to emit token id `248069` / content transition with stable long-output semantics. |
| Qwen3.5-MoE `</think>` watch-token trace | Copied GGUF compute-logits trace overlay, TP4 graph-mode norm-offset plus KV-layout bypass, P2P-on on `.20`; HF-weight TP4 trace as control | Disabled boundary tracing inside the compiled graph path and logged sampled logits/ranks for `<think>` `248068`, `</think>` `248069`, comma `11`, and underscore `62` during the shortest benchmark prompt. | GGUF served but hit `finish_reason=length` at `4096` completion tokens with `20218` reasoning chars and `0` content chars. The best sampled `</think>` rank was `198`. The HF-weight control closed after `2957` completion tokens with `finish_reason=stop` and sampled `</think>` rank `6`. | Promote as exact delimiter-transition evidence. Reject parser-only, prompt-length, P2P, and TP4 explanations. The next source target is the late-layer/logit path that keeps token id `248069` non-competitive in GGUF. |
| Qwen3.5-MoE exact HF-prefix delimiter score | vLLM completion endpoint with token-list `prompt`, HF stride-1 trace, GGUF TP4 graph-mode norm-offset plus KV-layout bypass | Used the original chat prompt token IDs plus the first `35` HF-generated token IDs as an exact forced prefix, avoiding special-token detokenization drift. Scored one next token on HF and GGUF with `</think>` watched. Then traced GGUF self-generation for the first `80` tokens. | HF ranked `</think>` id `248069` at `3` under the forced prefix, and GGUF also ranked it at `3`. GGUF self-generation diverged from HF at generated token index `1`: HF began `<|im_end|>Thinking Process:`, while GGUF began `<|im_end|>Here's a thinking process:`. GGUF self-generation stayed reasoning-only through the `80`-token cap. | Promote as trajectory-boundary evidence. Supersedes the blanket delimiter-scoring theory. The next source target is early-step trajectory divergence and why the GGUF self path later fails to recover into a reasoning close. |
| Qwen3.5-MoE faithful chat first-divergence probe | Chat endpoint, TP4, P2P-on, HF and GGUF trace stride `1`, watch IDs `90700`, `8160`, `8340`, `579`, `248069`, `11`, `62` | Compared the first two chat-generated tokens on the faithful chat path after rejecting token-list completion as a proxy for this hybrid decode state. | Both HF and GGUF first selected token `248046`. On the next step, HF ranked `Thinking` id `90700` first with logit `25.546875` and `Here` id `8160` second with logit `25.203125`. GGUF ranked `Here` first with logit `26.234375` and `Thinking` second with logit `22.953125`. Branch hidden norms were close: HF `108.96`, GGUF `108.02`. | Promote as the first actionable branch boundary. The next source target is token-specific logit contribution at this step: output norm/gate, lm-head rows for ids `90700` and `8160`, MoE/router expert output, and GatedDeltaNet/Mamba decode state. |
| Qwen3.5-MoE branch-row live instrumentation | Copied HF/GGUF trace overlays with request-gated `compute_logits()` `lm_head` shard/row logging | Tried to inspect token-row ownership and row dots for the `90700` / `8160` branch directly inside the live HF TP4 serving path. | The HF chat request failed with worker shutdown after one prefill/profile compute event. The only useful line reported `ParallelLMHead`, local weight shape `(62080, 2048)`, `org_start:124160`, `org_end:186240`, and watched IDs remote on that shard. | Reject live row-read instrumentation in the vLLM server path. Use offline row/materialization comparison or a reduced single-process diagnostic, and keep live serving hooks limited to low-risk hidden/logit trajectory capture. |
| Qwen3.5-MoE branch hidden-vector comparison | Copied HF/GGUF trace overlays with env-gated hidden-vector output only | Compared the hidden row at the first post-special-token branch using the same faithful two-token chat request, without reading `lm_head.weight` in live workers. | HF and GGUF both reproduced the `Thinking`/`Here` branch flip. Branch hidden norms were close (`108.961647` HF, `108.671486` GGUF), but vector direction differed: cosine `0.890908`, centered correlation `0.890758`, L2 diff `50.829083`, mean absolute diff `0.878714`, max component diff `5.335938`. | Promote as hidden-stream drift evidence. The next source target moves upstream of final logits/lm-head row ownership to output norm/gate, layer-39 MoE/router/expert output, GatedDeltaNet/Mamba state carry, and residual-stream formation before `compute_logits()`. |
| Qwen3.5-MoE eager branch boundary attempt | GGUF TP4 norm-offset plus KV-layout hidden-vector overlay, `MAX_MODEL_LEN=4096`, `--enforce-eager`, P2P-on, faithful two-token chat request | Checked whether eager mode reproduces the graph branch flip and whether layer-level boundary hooks fire outside graph capture. | Eager reached health after about `154` seconds. It reproduced the branch flip: event 0 selected token `248046`, event 1 ranked `Here` id `8160` first and `Thinking` id `90700` second, and event 2 ranked `'s` id `579` first. Boundary output still contained only `model.final_hidden`, not layer-0 or layer-39 internals. | Promote as graph/eager branch agreement; reject current eager boundary-hook shape as a layer-localization method. Do not repeat unchanged. Move summaries deeper into model-internal late-layer code or use offline reduced diagnostics. |
| Qwen3.5-MoE non-SSM norm-offset processor | `transformers/modeling_gguf_pytorch_utils.py` tensor processor route for `qwen35moe` | Compared GGUF embeddings, lm-head, output norm, block norms, attention q/k norms, and SSM norms against the HF snapshot. Embeddings and lm-head rows matched directly. Non-SSM RMSNorm-family weights matched HF only after subtracting `1.0`; `ssm_norm` matched directly. Added a `qwen35moe` tensor processor that shifts only non-SSM norm weights and inherits the existing MoE expert merge logic. Also simulated TP4 `w13`/`w2` expert shards for all four ranks and matched HF within FP32 noise. | The graph-mode TP4 forced-unquantized M-RoPE server reached health, but one-token `Hello` still returned `_`, and the 16-token math prompt timed out after `180` seconds with no bytes. | Promote the norm processor as necessary source evidence; reject it as sufficient. The next diagnostic should compare router/top-k outputs, recurrent state update/copy, output norm/gate application, and final logits against a coherent HF or llama.cpp control. |

## Reproduction Gate For Future GGUF Work

GGUF source changes should not be promoted directly from first-token probes to
throughput claims. The promotion gate is:

1. TP1 deterministic probe sanity.
2. TP2 deterministic probe sanity.
3. TP4 deterministic probe sanity.
4. TP8 deterministic probe sanity.
5. Normal release-style warmups.
6. `c1_128` uncapped strict prompt.
7. `c1_2000`.
8. `c1_10000`.
9. Backend metric comparison against the FP16 v0.2.1 reproduction path.

Any source patch that does not clear the deterministic first-token probes should
stay in the rejected or active-source category and should not be used for
public throughput claims.

## Next Source Work

The next source pass should compare the active coherent dense GGUF CausalLM path
against the published HF-weight FP16 path before running more benchmark ladders.
The relevant Qwen3.5 / Qwen3.6 GatedDeltaNet and linear-attention path is:

- `mixed_qkvz`
- `mixed_ba`
- q/k/v/z after split and repeat
- beta and gate tensors
- GatedDeltaNet core output before normalization
- normalized/gated output before flattening
- output projection input and output

The goal is no longer just to determine why TP4 diverges from TP1; the corrected
dense path is coherent. The patient fused-GDN path reduces the dense gap but
still decodes below the published FP16 path. The remaining question is whether
the overhead is specific to GGUF materialization, Qwen3.5 CausalLM routing,
graph/logits/sampler execution, remaining GatedDeltaNet parity, or release-overlay
parity outside the fused projection.

The next implementation candidates should prefer small, source-level C/C++ or
loader-path changes over broad launch-flag changes. In particular, inspect:

- adding a narrow timing diagnostic around the existing fused-GDN forward path
  in the current `GGUF-070` CausalLM baseline. The active GGUF bundle and clean
  HF release bundle both reach `torch.ops.vllm.gdn_attention_core`; the next
  useful hook should measure projection, GatedDeltaNet core, output projection,
  and logits/sampler boundaries without changing model semantics. Direct Python
  timing inside the compiled forward is rejected because Dynamo blocks
  `torch.cuda.synchronize()` and `time.perf_counter()`, so prefer `rocprof`,
  compiler-safe C++/custom-op markers, or built-in tracing. If `rocprof` is
  used, target a decode request window or worker/offline process, but first
  prove that the selected profiler shape emits artifacts. Do not repeat the
  broad API-server-wrapper run that captured only startup/parent-process HIP
  API activity, and do not repeat the fileless `rocprofv3 --attach` PID shape
  unchanged. The release image lacks `libdw.so.1`, so `rocprofv3` wrapper
  attempts need a read-only dependency mount or equivalent non-mutating setup.
  Do not wrap the full Persistent-AR server unchanged under `rocprofv3`; the
  current evidence shows that shape can fault the workers before readiness. Do
  not use a Persistent-AR-disabled wrapper as a substitute; that path is
  non-comparable, very slow, and still did not emit artifacts in the tested
  stop path;
- treating ROCTX marker output as viable only in controlled process shapes for
  now. Tiny controls emitted marker files, but raw Python markers, built-in
  record-function profiler hooks, and ad hoc Python custom marker ops all fail
  the compiled Qwen model path or fail to emit marker records. Any next marker
  implementation should be lower-level and alias-correct, or moved into a
  reduced offline/single-worker reproducer that does not perturb the full vLLM
  server graph;
- treating the `transformers` fast-path warning in GGUF logs as loader-time
  evidence unless a serving hot-path trace proves otherwise. vLLM's GGUF loader
  builds a dummy Transformers model to map GGUF names, which can instantiate the
  Transformers Qwen3Next class and print that warning before serving starts;
- treating broad internal GGUF matmul overhead as rejected for the coherent dense
  path unless a deeper trace contradicts it. The `GGUFLinearMethod.apply`
  diagnostic sampled graph profiling and one short decode request and recorded
  only `ParallelLMHead` hits, not internal Qwen block linears;
- treating isolated lm-head method selection as rejected for the coherent dense
  path. Forcing only `lm_head` to the unquantized embedding method kept strict
  validity but did not improve decode beyond the prior GGUF band;
- treating fused-GDN plus native GGUF SwiGLU as the current best dense GGUF
  source baseline, not as a completed performance fix. It removes the extra
  split `in_proj_ba` projection, restores the native MLP extension under an
  explicit F16 GGUF guard, and improves throughput, but still misses
  release-band dense TPS;
- profiling the coherent dense GGUF path against the published HF-weight FP16
  path at GatedDeltaNet, linear-attention, output projection, and sampler
  boundaries;
- treating the FLA / causal-conv warning as loader-boundary noise until a
  request-window profile proves it is on the hot path;
- treating the modern GGUF `Qwen3_5ForConditionalGeneration` rerun as rejected;
  future comparisons should profile the current GGUF `Qwen3_5ForCausalLM`
  baseline against HF-control internals instead of relaunching architecture
  overrides;
- treating graph-capture setup as weak evidence for the remaining gap. HF and
  best-GGUF logs use the same capture mode, capture sizes, and about the same
  capture duration, so the next check needs hot-path timing rather than more
  startup log comparison;
- treating native MLP SwiGLU enablement as already tested: it was disabled by
  GGUF `quant_config`, can be enabled under an explicit F16 GGUF guard, but does
  not close the benchmark gap by itself;
- treating MLP-down residual pre-fold as a rejected GGUF revisit unless a
  profiler identifies that exact boundary. A copied F16 GGUF bundle that relaxed
  the GGUF guard and supplied the native row-parallel residual extension
  completed the full ladder strict-valid, but was slightly slower than the
  current best GGUF baseline;
- treating optional FLA / `causal_conv1d` package installation as a rejected
  first move. The release image lacks those importable modules in both GGUF and
  HF-control contexts, but only GGUF remains below release-band TPS. The GGUF
  fallback warnings should guide profiler placement rather than a blind image
  mutation;
- partition axis selection for `ssm_out`;
- companion-tensor split granularity for Q4_0 `attn_qkv` / `attn_gate`
  against Q8_0 `ssm_out`;
- parity between the local C scanner and llama.cpp's offset/block-size
  validation rules;
- logical segment labels for q/k/v/z, beta, alpha, and `ssm_out`, since raw
  TP4 shapes and quant-block divisibility already pass;
- per-rank segment checksums from the segment-labeled TP1 control against the
  failing TP4 run, so populated segments can be compared for semantic parity
  instead of only presence;
- the current vLLM split `in_proj_qkvz` plus `in_proj_ba` Qwen3.5 path versus
  the FP16 release overlay's fused qkvzba path;
- why the combined fused-GDN plus native GGUF SwiGLU path still misses the
  HF-weight FP16 band, especially whether the remaining gap is GatedDeltaNet
  core input/state handling, graph/logits/sampler cost, or GGUF
  adapter/materialization overhead;
- GGUF packed shard ID handling where Qwen3.5 uses tuple shard mappings for
  q/k/v in FP16 but GGUF may need explicit split materialization;
- TP1 versus TP4 per-rank shard offset maps and tensor content checksums for
  q/k/v/z, beta/alpha, and `ssm_out`;
- the chunked-prefill to one-token decode transition in layer-0 GatedDeltaNet;
- GatedDeltaNet core/state equivalence after q/k/v/z/b/a inputs are formed,
  since TP4 aggregate input checksums match TP1 but core/output checksums do
  not;
- Q/K repeat expansion semantics, since pre-conv inputs and conv weights match
  but tiled repeat makes TP1 global chunks differ from TP4 local ranks, while
  repeat-interleave repairs only first-token sanity and fails the full-context
  strict gate;
- full-context benchmark-prompt behavior only after a source candidate changes
  the measured decode ceiling;
- FP16/half GGUF prefill behavior only if profiling shows prefill bottlenecks
  matter for the measured benchmark tiers;
- why the force-unquantized GGUF method selection did not remove GGUF qweight
  handling from the active Qwen path;
- coherent patch-bundle construction, since entrypoint patch application can
  override individual file mounts;
- a possible direct-QwenNext text-config adapter if the Qwen3.5 wrapper remains
  too stale for FP16/half GGUF source work;
- whether fused `in_proj_qkvz` should be split into separate qkv and z paths
  for GGUF;
- whether fused `in_proj_ba` should be split into separate beta and alpha
  paths for GGUF;
- whether the GGUF loader is applying row-parallel and column-parallel shards
  with the same semantics used by the FP16 path, without repeating the rejected
  broad `_fused_mul_mat_gguf` overhead theory for internal block linears;
- continuing the real `qwen35moe` / Qwen3.5-MoE GGUF architecture route from
  the runtime norm-offset repair candidate. The same reduced raw `Hello` probe
  is now coherent with both HF weights and GGUF weights under the fix, so the
  next target is a clean benchmark overlay rather than another launch-shape
  flip;
- preserving the decoder-layer `input_layernorm` trace evidence. The trace
  showed the active bug was GGUF non-offset norm values entering vLLM
  `GemmaRMSNorm`, which applies `1 + weight`. Runtime correction repaired the
  layer-0 post-norm and final-hidden parity. Do not repeat blind
  tensor-processor edits unless a future loader path proves the active
  parameters have moved there;
- replacing or bypassing the slow F16 GGUF MoE fallback with a capture-safe
  fast/unquantized path that can use normal FusedMoE-style tensors after GGUF
  materialization;
- preserving the Qwen3.5-MoE text config fields from GGUF metadata, including
  linear-attention layer types and the `512` MoE intermediate dimension;
- preserving the explicit `model.layers.*.linear_attn.dt_bias` to
  `blk.*.ssm_dt.bias` mapping and the full inverse llama.cpp Qwen3.5
  V-head layout transform for QKV V rows, Z, alpha, beta, `dt_bias`, `A_log`,
  conv V channels, and out-proj columns in both direct and stacked load
  branches;
- preserving the `qwen35moe` tensor processor rule that subtracts `1.0` from
  non-SSM RMSNorm-family GGUF tensors while leaving `ssm_norm` direct;
- preserving the runtime GGUF-only `GemmaRMSNorm` offset correction until the
  loader-side processor route is proven to affect the active serving
  parameters;
- treating the env-gated graph-mode hybrid KV-layout bypass as diagnostic
  evidence, not a benchmark fix. It gets the Qwen3.5-MoE GGUF route healthy and
  improves warmup decode from the rejected eager band to about `73.6` backend
  TPS, but it still does not close the strict benchmark prompt;
- preserving the shortest-prompt comparator evidence: GGUF `prompt_repeat=0`
  does not close after an estimated `18019` generated tokens, while the
  HF-weight comparator closes the same prompt after `2957` generated tokens
  with `qwen_gate_valid=true`;
- preserving the thinking-disabled diagnostic as a source boundary only:
  GGUF can produce visible content when reasoning is disabled, but this is not
  a valid benchmark/release path and the 512-token cap already shows repetitive
  content-mode degradation. The follow-up watch-token trace showed Qwen
  `</think>` id `248069` reached only sampled rank `198` in GGUF, versus rank
  `6` in the HF-weight control. The exact-prefix follow-up then showed GGUF can
  rank `</think>` at `3` under the HF-generated prefix, so the active target is
  self-trajectory divergence and recovery rather than a universal delimiter
  logit failure. The faithful chat trace narrows the first branch to the
  `Thinking` id `90700` versus `Here` id `8160` rank flip after token `248046`;
- preserving the text-only Qwen3.5-MoE `IsHybrid` class marker so vLLM runs
  hybrid attention/Mamba cache verification. The 2026-06-26 TP4 graph-mode
  smoke showed this changes startup into the expected hybrid cache path, but
  it was not sufficient before the runtime norm-offset fix;
- replacing the current final-logit trace placement with a request-gated trace.
  The existing hook fired during vLLM's internal compile/profile path on `.40`,
  before the client prompt, so it is rejected as an explanation for incoherent
  `Hello` output;
- preserving Qwen3.5-MoE expert tensor layout and TP sharding for gate/up/down
  expert weights before any TP4 or TP8 benchmark warmups;
- whether MoE GGUF inherits the dense execution overhead or needs a separate
  model-specific overlay after deterministic probes, warmups, strict, and
  fixed-token tiers are complete.
- request-gated row-vector tracing for Qwen3.5-MoE GGUF versus the HF-weight
  comparator. The current GGUF model-loop trace captures layer 0, layer 39,
  loop-final norm, and final hidden row values while reproducing the
  `Thinking` versus `Here` branch flip. Aggregate stats match closely enough
  that the next target is directional row-vector drift by layer boundary, not a
  gross norm or tokenizer failure. The first HF-only row-value comparator
  preserves the known-good branch and shows `layer39.post_mlp` row cosine of
  about `0.984005` against GGUF, but it is not decode-call localized. Add a
  call-indexed decoder boundary trace before claiming a layer-local root cause:
  capture call 0/1/2 `pre_input_layernorm`, `post_input_layernorm`,
  `post_attention`, `post_mlp`, final hidden, and branch logits for both HF and
  GGUF under the same two-token request.
- call-indexed decoder-boundary evidence from `.20` now narrows the MoE GGUF
  branch flip to the first post-special-token decode call. HF and GGUF match
  exactly through layer-0 call-1 input and nearly through post-input-layernorm,
  then diverge after layer-0 attention/MLP: `layers.0.call1.post_attention`
  cosine `0.900811`, `layers.0.call1.post_mlp` cosine `0.879649`,
  `layers.39.call1.post_mlp` cosine `0.955899`, and
  `compute_hidden.call1` cosine `0.890804`. Treat layer-0 attention,
  GatedDeltaNet/Mamba state carry, q/k/v/z/beta/alpha construction, and residual
  add order on call 1 as the next source targets. Do not spend another full
  warmup/strict/fixed-token ladder until this trajectory branch is repaired.
- all-head GatedDeltaNet tracing and static exact tensor checks refine the
  target again. The z gate is aligned (`z_for_norm` sampled pair cosine
  `1.000000`), static inverse transforms for V/Z/out-projection, `dt_bias`,
  `A_log`, `ssm_norm`, and conv V channels match HF within FP32 noise, and
  offline reconstruction proves the out-projection mapping reproduces the traced
  HF/GGUF projection outputs. The mismatch is already present in
  `core_attn_out_raw` from `torch.ops.vllm.gdn_attention_core`, with sampled
  call-1 rank-pair cosine as low as `0.837480`, and worsens after the
  normalized core output. The next source hook should inspect runtime-loaded
  GDN static parameters, recurrent state/cache identity, and TP-local state
  ordering inside or immediately around `gdn_attention_core`.
- runtime GatedDeltaNet state tracing found a served-loader bug in the active
  MoE GGUF path. Before repair, runtime `dt_bias`, conv weights, `ssm_norm`,
  mixed q/k/v, beta, and initial state matched HF, but GGUF runtime `A_log` was
  rank-constant while HF runtime `A_log` was rank-varying. The first attempted
  fix, moving conversion before an equal-head early return, did not reach the
  served path. The active repair adds explicit `blk.*.ssm_a` to
  `model.layers.*.linear_attn.A_log` mapping and applies `log(-GGUF_A)` in the
  top-level `Qwen3_5ForCausalLMBase.load_weights` path before
  `AutoWeightsLoader`.
- the repaired `A_log` path supersedes the earlier layer-0 GatedDeltaNet drift
  result. With runtime `A_log` matching HF, layer-0 call-1 now matches HF
  through GatedDeltaNet inputs, runtime params, recurrent core,
  post-attention, and post-MLP with cosine effectively `1.000000`. The branch
  still flips, and layer-39 call-1 now shows the meaningful drift:
  `pre_input_layernorm` min cosine `0.990509`, `post_input_layernorm`
  `0.970792`, `post_attention` `0.986151`, and `post_mlp` `0.891285`. Next
  source work should bisect intermediate layers under the repaired loader path
  before running warmups or fixed-token tiers.
- intermediate bisection under the repaired `A_log` path moved the active
  mismatch to the first full-attention layer. Layers 0-2 stayed aligned, while
  layer 3 call 1 drifted immediately after `self_attn`: `post_attention`
  cosine `0.859239` and `post_mlp` cosine `0.750201`. Layer 3 is
  `full_attention` in the model's layer schedule. The active cause was another
  GGUF norm-offset hole: imported `Qwen3NextAttention` instantiated
  `Qwen3NextRMSNorm` for `self_attn.q_norm` and `self_attn.k_norm`, bypassing
  the GGUF-aware `Qwen3_5RMSNorm` runtime correction used for decoder input and
  post-attention norms. Replacing q/k norms with `Qwen3_5RMSNorm` under the
  existing runtime norm-fix env repaired the two-token branch probe and aligned
  layers 0-8 against HF with cosine effectively `1.000000`.
- current MoE GGUF source-candidate overlay on `.20`:
  `<validation-workspace>/experimental-patches/qwen35moe-text-gguf-runtime-norm-offset-kvfix-gdnstate-alogloadfix-fullattnqknorm-20260626`.
  It combines the runtime non-SSM norm-offset fix, graph KV-layout diagnostic
  bypass, explicit `ssm_a` / `A_log` mapping plus top-level loader transform,
  and full-attention q/k norm replacement. It now passes the clean 131K
  graph-mode correctness ladder on `.20`: strict `71.916`, `c1_2000`
  `73.380`, and `c1_10000` `65.749` backend TPS with the strict gate valid.
  Treat it as a correctness candidate and performance rejection until GGUF MoE
  execution speed approaches the FP16 v0.2.1 TP4 reference band. A trace-gate
  disabled rerun with the same source bundle landed in the same band: strict
  `69.961`, `c1_2000` `73.228`, and `c1_10000` `65.643` backend TPS. This
  rejects disabled trace overhead as the missing performance path and moves the
  next source target to GGUF F16 materialization and MoE/GatedDeltaNet hot-path
  parity with HF weights.
- the quant-method audit copy of that overlay rejects a broad GGUF adapter
  overhead theory for MoE TP4. Across the four TP workers, all `760`
  `LinearBase` selections used `UnquantizedLinearMethod`, all `160` FusedMoE
  selections used forced `UnquantizedFusedMoEMethod`, and no hot serving module
  selected `GGUFLinearMethod`, `GGUFEmbeddingMethod`, or `GGUFMoEMethod`.
  The remaining MoE gap is therefore more likely in compiled execution parity,
  GatedDeltaNet/linear-attention behavior, shared-expert/routing structure,
  graph structure, or residual/norm path differences after GGUF tensor
  materialization.
- the default-scheduler / broad graph-capture rerun rejects launch graph sizing
  as the missing MoE GGUF throughput lever. Removing `--max-num-seqs 1
  --max-num-batched-tokens 1024` restored the HF-like graph capture settings
  (`max_num_batched_tokens=2048`, capture sizes through `512`, and
  `max_cudagraph_capture_size=512`) and kept the strict gate valid, but the
  fixed-token ladder stayed in the same band: strict `71.030`, `c1_2000`
  `73.130`, and `c1_10000` `65.552` backend TPS. Keep this launch shape for
  future comparable tests, but continue source inspection around the GGUF
  CausalLM / Qwen3.5-MoE hot path and the remaining FLA / causal-conv fallback
  warnings.
- a c1 topk8 MoE fastpath diagnostic rejects that env knob as a current release
  image explanation. A TP4 eager 4K smoke with
  `VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH=force` reached health and produced coherent
  text, but the image source contained no qwen c1 topk8 overlay strings and the
  runtime logs emitted no active/rejected fastpath messages. Until a topk8 patch
  is intentionally mounted, this variable is inert and should not be treated as
  the missing GGUF performance path.
- the FLA / causal-conv fallback warning is localized to GGUF loader
  name-mapping, not the proven request hot path. The warning string is in the
  vendored Transformers Qwen3.5 / Qwen3.5-MoE constructors. The GGUF loader then
  constructs a meta-device `AutoModelForCausalLM` dummy model to obtain parameter
  names and checkpoint conversion mappings. The vLLM serving `qwen3_5.py` path
  imports vLLM QwenNext classes and calls `torch.ops.vllm.gdn_attention_core`.
  Treat missing `fla` / `causal-conv1d` as rejected standalone package-install
  work until request-window profiling proves otherwise.
- the active MoE GGUF overlay is not still on the stock `GGUFMoEMethod` path.
  vLLM globally sets `quantization=gguf` and `load_format=gguf`, but the active
  overlay returns `UnquantizedFusedMoEMethod` for F16/BF16/F32 GGUF expert
  layers when the force-unquantized path is enabled. The remaining source target
  is the compiled serving graph and request-time timing, not another unchanged
  quant-method selection audit.
- `gdn_attention_core` is a viable first timing boundary because it is already a
  Python-registered direct custom op. In the official release image, the Qwen
  source registers `torch.ops.vllm.gdn_attention_core` with
  `direct_register_custom_op`; the registered body calls `self._forward_core(...)`
  and the installed image does not expose a separate C++ extension symbol for
  that op. A tiny `.20` ROCm control showed that timing and optional
  `torch.cuda.synchronize()` inside a direct custom-op body survive a
  `torch.compile(fullgraph=True)` caller. This promotes an env-gated
  `gdn_attention_core` wrapper timing diagnostic as the next lower-impact path,
  while leaving projection, output projection, logits, and sampler timing for
  later instrumentation.
- FusedMoE direct custom-op timing is now also a rejected bottleneck target in
  the steady request path. Timing `torch.ops.vllm.moe_forward_shared` around
  `DefaultMoERunner.forward_impl(...)` showed graph-mode Python wrappers can
  miss live graph replay entirely, but eager request-window comparators were
  observable and put GGUF at parity with HF after excluding layer-0
  first-request noise: `(1, 2048)` averaged `645,803` ns versus `649,962` ns,
  `(32, 2048)` averaged `678,506` ns versus `694,144` ns, and non-layer-0
  `(15, 2048)` averaged `808,102` ns versus `801,966` ns. The next source work
  should use graph-visible or whole-decode profiling rather than another
  unchanged `DefaultMoERunner` pass.
- Post-forward logits and sampler timing is also rejected as the primary MoE
  GGUF gap. Synchronized timing around `lm_head.apply`, TP logits gather, and
  v1 sampling on the faithful two-token request stayed sub-millisecond and
  effectively matched HF. The remaining performance work should move to
  whole-model graph replay, model-runner scheduling, CausalLM-versus-HF graph
  structure, or regions outside the already-timed GDN, FusedMoE, logits, and
  sampler boundaries.
- `GPUModelRunner` whole-forward timing has now narrowed the runner envelope.
  The synchronized diagnostic changed the expected two-token branch output and
  is rejected. The no-sync diagnostic preserved output and showed that
  `preprocess` and `postprocess` were small, while `model_forward` dominated the
  runner envelope. With broad graph capture aligned, the HF-weight comparator
  was slower than GGUF in the same short request window (`model_forward`
  HF/GGUF `1.107` at tokens `15`, `1.132` at tokens `32`). Because this is
  no-sync Python-side timing, it is not final GPU replay evidence. Treat it as a
  source-inventory boundary: do not keep instrumenting the same Python runner
  path until a source change needs a smoke check. The next useful evidence
  should come from ROCm/HIP graph-visible profiling, kernel traces, or
  fixed-token decode-window profiling.
- `rocprofv3` is not yet a usable out-of-the-box profiler path for the current
  vLLM multiprocess server. Worker PID attach reports success but produces no
  files, and parent-process delayed wrapping does not flush usable output from
  the child worker tree. The release image also lacks `libdw.so.1`, so a
  read-only host-library mount is needed even to start the wrapper. Tracing can
  perturb decode severely: c1_2000 fell from the normal GGUF band to `32.892`
  backend TPS when traced, while a shorter c1_512 kept `73.019` backend TPS but
  still hung in profiler shutdown. Future profiler source work should be
  worker-entrypoint-aware or use explicit profiler pause/resume hooks around a
  request window; do not spend another pass on unchanged parent wrapping or
  PID attach.
- Full MoE-block Python timing is a rejected source target. Instrumenting
  `Qwen3NextSparseMoeBlock.forward` in the graph-mode TP4 GGUF path failed
  before health because `torch.compile` cannot trace `time.perf_counter_ns()`
  inside the compiled block. The same instrumentation in eager 4K GGUF and HF
  comparators preserved the `Thinking Process` branch and showed parity at the
  stable decode shape: `(32, 2048)` total block averages were `2,486,178` ns
  for GGUF and `2,479,019` ns for HF, while the internal-router expert stage
  averaged `2,241,665` ns and `2,239,985` ns. The wrapper, shared-add, and TP
  all-reduce stages do not explain the GGUF fixed-token TPS gap.
- external dense GGUF patch references from
  `Kausik-A/qwen3.6-27b-mi50-vllm`: `qwen3_5` and `qwen3_5_moe` config
  registration, `qwen35` / `qwen35moe` GGUF aliases, `ssm_dt.bias` mapping,
  text-only M-RoPE stripping, quantized `embed_tokens` / `lm_head` plumbing,
  conv1d 2D-to-3D reshaping, `IsHybrid` hooks on text-only Qwen3.5 classes,
  tuple-shard splitting in `MergedColumnParallelLinear`, and MiniMax M2 GGUF
  expert aliases. These are source-reference items only; the project targets a
  single-MI50/eGPU, eager-mode, ROCm 6.3-era, 4096-context dense GGUF launch
  and should not be treated as a ROCm7.2 TP8/TP4 release reproduction path.
  It also does not map MoE `ssm_a` / `A_log`, and its top-level MoE CausalLM
  load path still uses plain `AutoWeightsLoader`, so it does not solve the
  active Qwen3.6 35B-A3B MoE GGUF branch flip.
- runtime non-SSM norm-offset subtraction is now a confirmed graph-structure
  difference between MoE GGUF and HF-weight paths. The current correctness
  baseline uses `VLLM_QWEN35_GGUF_RUNTIME_NORM_OFFSET_FIX=1`; its compiled
  graph contains `_qwen35_effective_weight` sites that materialize
  `weight - 1.0`, unlike the HF-weight graph. This is a real graph-parity
  target, but not a proven whole-gap explanation.
- load-time norm-offset variants are rejected for now. Disabling runtime offset
  produced incoherent smoke output. Applying a load-time non-SSM norm offset
  while disabling runtime offset reached warmups but failed `c1_128` strict
  immediately with no valid thinking close. Adding full-attention q/k norm
  class replacement still returned an empty one-token smoke response. These
  variants reduced graph size versus the runtime-offset graph but lost Qwen
  strict/parser semantics.
- next source inventory item: build a tensor-level norm audit before another
  serving launch. Compare loaded input/post/final/full-attention q/k/SSM norm
  tensors across runtime-offset GGUF, load-time GGUF, and HF controls to
  identify which weights are shifted, double-shifted, or still using runtime
  `GemmaRMSNorm` convention.
- raw tensor norm audit result: the staged Qwen3.6 35B-A3B F16 GGUF file maps
  cleanly to the local HF snapshot for all `131` norm tensors. Non-SSM norms
  need exactly `-1.0`; SSM norms need no shift. The current load-time hook
  selects the correct mapped raw tensors, so the load-time serving failure must
  be investigated after vLLM module materialization rather than by changing the
  raw GGUF name rule again.
- live module norm audit result: the first load-time hook fired twice in the
  active vLLM path and also shifted the mapped SSM name
  `linear_attn.norm.weight`. Correct source shape is a single top-level
  load-time hook plus exclusions for both raw `ssm_norm.weight` and mapped
  `linear_attn.norm.weight`, with q/k norm class replacement active under the
  load-time switch. This restores live module norm stats and strict validity,
  but the ladder remains in the same slow MoE GGUF band: strict `71.995`,
  `c1_2000` `73.005`, and `c1_10000` `65.456` backend TPS.
- corrected load-time norm plus broad scheduler is the current clean MoE GGUF
  profiling baseline. Removing the narrow scheduler override restored
  `max_num_batched_tokens=2048`, `PIECEWISE=51`, `FULL=35`, and cache
  `686a95799f`; the ladder stayed strict-valid at `71.785`, `c1_2000`
  `73.396`, and `c1_10000` `65.796` backend TPS. This slightly improves the
  observed GGUF `c1_2000` band but remains far below FP16 TP4. Treat this as
  source-baseline cleanup, not as performance promotion.
- broad GGUF and aligned HF graph signatures match at the major source
  boundaries already tested: `242` parameter inputs, `60` GDN custom-op calls,
  `160` routed MoE calls, `120` shared-MoE calls, and `244` all-reduce
  occurrences on rank 0. Remaining source work should inspect graph replay /
  generated kernels or Inductor code regions rather than repeating broad
  scheduler, FusedMoE, GDN, logits, sampler, runner-envelope, or norm-offset
  timing experiments unchanged.
- vLLM compile debug is now usable through a disposable wrapper entrypoint
  rather than the normal extra-argument string split. The accepted diagnostic
  route passes `--compilation-config` as one argv element from an environment
  variable, leaving the release image unchanged. The first successful MoE GGUF
  TP4 broad corrected compile-debug run captured a debug dump of about `26M`
  across `244` files and an unpacked compile cache of about `94M` across
  `100` files. Each rank graph had `9235` lines, each rank had `61` generated
  debug-kernel files, and rank-0 generated kernel files had no `tl.dot`
  occurrences. The AOT graph hash was
  `b5fab22e93990a3258b763cf688ae25969b823a138e729f0bb129d0f60e29b3a`.
- the same compile-debug artifact confirms corrected load-time norm cleanup at
  the generated-source level. Rank-0 `computation_graph.py` still referenced
  `_qwen35_effective_weight` `101` times, but the captured function body
  returned `self.weight.data` and did not contain the runtime `weight - 1.0`
  subtraction. Runtime norm subtraction should no longer be treated as the
  active MoE GGUF performance explanation for the corrected load-time path.
- external source reference `Kausik-A/qwen3.6-27b-mi50-vllm` at commit
  `61b273d` contains five patch surfaces worth keeping in the GGUF source
  checklist: config registration, GGUF loader aliases and tensor-name mapping,
  Qwen3.5 text/MoE model hooks, registry entries, and tuple-shard
  `MergedColumnParallelLinear` loading. The useful deltas are
  `qwen35`/`qwen35moe` aliases, `ssm_dt.bias` mapping, text-only M-RoPE
  stripping, quantized `embed_tokens`/`lm_head`, GGUF `conv1d.weight`
  2D-to-3D reshape, text-only `IsHybrid` hooks, tuple-shard splitting, and
  MiniMax M2 expert aliases. Its launch shape is single-MI50/eGPU, eager mode,
  and 4096 context, so it is not a substitute for the `.20` TP4/TP8
  full-BAR/P2P-on release-reproduction path.
- aligned HF TP4 compile-debug control: using the same wrapper diagnostic route
  as the corrected-load-time GGUF baseline, the HF-weight TP4 path reached
  `PIECEWISE=51` and `FULL=35`, emitted about `26M` of debug dump and `94M` of
  unpacked compile cache, and produced rank graphs of `9145` lines. The rank-0
  graph had `162` `torch.ops.vllm.all_reduce`, `110`
  `torch.ops.vllm.rocm_unquantized_gemm`, `60`
  `torch.ops.vllm.gdn_attention_core`, `40`
  `torch.ops.vllm.moe_forward_shared`, `20`
  `torch.ops.vllm.unified_kv_cache_update`, and `20`
  `torch.ops.vllm.unified_attention_with_output` calls. Those counts match the
  corrected GGUF graph exactly at the measured custom-op boundary. HF had no
  `_qwen35_effective_weight` call sites; GGUF had them, but the corrected body
  returned `self.weight.data` with no runtime `weight - 1.0` subtraction. Both
  paths had `61` rank-0 generated debug-kernel files and no `tl.dot`
  occurrences. This promotes HF as the current source-control comparator and
  rejects missing major compiled graph regions as the whole-gap explanation.
- trace-gated Q/K/V split cleanup: the GGUF Qwen3.5 source previously split
  `mixed_qkv` into Q, K, and V before checking whether layer-0 tracing was
  active. Moving that split under the `trace_active` guard removes a disabled
  trace-only graph wart and is valid cleanup. The `.20` TP4 broad GGUF ladder
  still produced strict `70.766`, `c1_2000` `73.104`, and `c1_10000` `65.546`
  backend TPS after eight warmups, with `PIECEWISE=51`, `FULL=35`, AOT hash
  `224188a6b6a29c71a85d735f590bd2ce704392d50cb0a3a24287867b7d630094`, and
  KV capacity `302016` tokens. Promote as source hygiene only; reject as a
  performance explanation or release-update candidate.
- dense GGUF batch-invariant startup test: `VLLM_BATCH_INVARIANT=1` is not
  currently usable in the dense TP8 GGUF current-best path. Without an explicit
  backend it fails vLLM's batch-invariant initialization because the attention
  backend is unset. With `--attention-backend FLASH_ATTN`, the path fails before
  serving during TorchInductor autotuning: no valid Triton config fits gfx906
  shared memory, with `163840` bytes required versus a `65536` byte limit.
  Reject unchanged batch-invariant repeats; any future attempt needs a concrete
  Triton/Inductor source or configuration change first.
- direct RMSNorm weight lookup: replacing the corrected-load-time MoE GGUF
  RMSNorm helper call with direct `self.weight.data` removes a known residual
  graph wrapper difference from HF, but it does not change the throughput class.
  The `.20` TP4 ladder produced strict `71.489`, `c1_2000` `73.015`, and
  `c1_10000` `65.454` backend TPS after eight warmups, with `PIECEWISE=51`,
  `FULL=35`, and `302016` KV tokens. The compiled cache had zero
  `_qwen35_effective_weight` references, so the wrapper was actually removed
  from the captured graph. Treat direct lookup as optional source cleanup only;
  reject it as the active performance explanation.
- MoE GGUF ConditionalGeneration architecture smoke: changing the text config
  architecture to `Qwen3_5MoeForConditionalGeneration` resolved that model
  class, but failed before model load because the multimodal processor expected
  an outer `Qwen3_5MoeConfig` and received the text-only
  `Qwen3_5MoeTextConfig`. The path selected the same attention block size
  (`528`) and Mamba padding (`0.76%`) as the working CausalLM MoE GGUF
  baseline, so the page-geometry explanation is rejected. Do not repeat an
  architecture-only ConditionalGeneration flip; any future route needs a real
  text-config adapter or multimodal-processor bypass first.
- generated-kernel source diff: rank-0 HF TP4 and corrected GGUF TP4
  compile-debug artifacts both had `14` generated kernel files with identical
  generated function-name sets. `11/14` files were byte-identical; the other
  three differed only in serialized `ModuleName` metadata for the expert module
  prefix. After removing comments and that metadata line, all generated kernel
  files matched. Reject static generated-kernel body differences as the active
  MoE GGUF performance explanation. The next boundary is replay-time behavior:
  HIP kernel mix/timing, graph replay, memory movement, or fixed-token
  decode-window traces.
- tuned MoE config lookup: the same corrected GGUF TP4 broad graph artifact and
  aligned HF TP4 broad graph control each logged exactly one use of
  `E=256,N=128,device_name=AMD_GFX906.json`. The GGUF run additionally logged
  `160` expected force-unquantized FusedMoE expert-layer events for F16 GGUF,
  while HF logged none. This rejects missing tuned GFX906 MoE config selection
  as the active throughput explanation and narrows the next source boundary to
  replay-time behavior rather than static config lookup.
- pre-grad graph signature/body boundary: corrected GGUF TP4 and aligned HF
  TP4 compile-debug artifacts had matching `BEFORE_PRE_GRAD` file counts,
  matching extracted forward signatures, the same combined signature hash
  (`9e5949826d4fbc75ea48e48959b9d875c0113b023c204e148dc11f261a33bb9c`), and
  matching measured counts for all-reduce, shared MoE, and ROCm unquantized
  GEMM custom ops. No GGUF materialization symbols appeared in either compiled
  pre-grad body. Reject unchanged static pre-grad signature/body diffing; the
  next boundary is completed replay/runtime behavior.
- no-sync `GPUModelRunner` timing: a full normal MoE GGUF TP4 ladder with the
  runner timing overlay stayed slow (`71.668` strict, `72.936` `c1_2000`,
  `65.400` `c1_10000` backend TPS). Decode `model_forward` timing averaged
  only about `0.042` ms while backend TPS stayed around `65`-`73`, proving the
  hook measures enqueue/bookkeeping rather than completed graph replay. Reject
  Python runner timing as the profiler path; use HIP kernel mix/timing, graph
  replay timing, memory movement, or a reduced completed-work reproducer next.
- `CUDAGraphWrapper` replay timing: a disposable env-gated overlay around
  `entry.cudagraph.replay()` produced completed replay timings on the corrected
  MoE GGUF TP4 graph path. The normal `.20` ladder remained in the slow GGUF
  band but stayed strict-valid: strict `72.063`, `c1_2000` `73.265`, and
  `c1_10000` `65.685` backend TPS. The timing file had `2008` rows, including
  `1964` `FULL` single-token replay samples averaging `13.542115 ms` and
  `44` `PIECEWISE` samples. Promote this replay boundary as the next comparator
  against aligned HF-weight TP4. Reject the timed run itself as benchmark
  evidence because sampled CUDA-event timing synchronizes the serving path.
- aligned HF `CUDAGraphWrapper` replay timing: running the same sampled replay
  overlay on the HF-weight TP4 control produced strict `71.784`, `c1_2000`
  `72.653`, and `c1_10000` `65.151` backend TPS, with `1944` `FULL`
  single-token replay samples averaging `13.655169 ms`. The HF and GGUF replay
  averages are effectively identical under this intrusive hook, while
  non-instrumented HF normally reproduces the higher release band. Reject
  unchanged sampled replay timing as the next profiler path. The next
  diagnostic should avoid per-token synchronization or move to request cadence,
  logits/sampler/postprocess, memory movement outside replay, or HIP kernel
  mix/timing.
- lagged `CUDAGraphWrapper` event timing: replacing immediate synchronize with
  a `32`-sample delayed event queue still perturbed the aligned HF TP4 control.
  The ladder produced strict `68.482`, `c1_2000` `73.352`, and `c1_10000`
  `65.724` backend TPS, with `2068` `FULL` replay samples averaging
  `13.466812 ms`. Reject event-based replay timing around
  `entry.cudagraph.replay()` for full-ladder comparisons. Future profiling must
  avoid injecting CUDA events into sampled replay calls.
- server-side reasoning parser toggle: removing the vLLM server-side `qwen3`
  reasoning parser from the corrected load-time MoE GGUF TP4 launch did not
  change the throughput class. The `.20` diagnostic still used the normal
  begin-think proxy benchmark path and produced strict `72.230` and `c1_2000`
  `73.023` backend TPS after the normal warmups. Reject server-side parser
  overhead as the missing source-level lever; continue below the launch-shape
  boundary with HIP/kernel mix, request cadence, logits/sampler/postprocess, or
  reduced replay diagnostics.
- MoE expert tensor materialization layout: a disposable load-time audit overlay
  around `UnquantizedFusedMoEMethod.process_weights_after_loading()` compared
  corrected F16 GGUF TP4 and HF TP4 on `.20` with the same published v0.2
  image, P2P-on lane, and tuned GFX906 MoE config. Both routes emitted `1280`
  rows: `40` MoE layers, `4` TP workers, `4` stages, and `2` expert tensors.
  The post-load summaries matched exactly. `w13_weight` starts contiguous as
  `(256, 256, 2048)` with stride `(524288, 2048, 1)`, then ROCm padding makes
  it non-contiguous with stride `(557056, 2176, 1)`; `w2_weight` remains
  contiguous as `(256, 2048, 128)` with stride `(262144, 128, 1)`. All audited
  rows were `torch.float16` and aligned at the sampled 16/64/128/256-byte
  boundaries. Reject expert materialization layout, stride, ROCm padding, dtype,
  contiguity, and basic pointer alignment as the current MoE GGUF TP4 gap.
  Continue with request cadence, HIP/kernel mix or memory movement, sampler /
  postprocess accounting, or a reduced completed-work replay reproducer.
- external `Kausik-A/qwen3.6-27b-mi50-vllm` source bundle: reviewed local head
  `61b273d` as compatibility context. Useful source ideas are `qwen35` /
  `qwen35moe` aliases, text-only M-RoPE suppression, `ssm_dt.bias` mapping,
  GGUF `conv1d.weight` reshape, tuple-shard support for fused QKVZ in
  `MergedColumnParallelLinear`, and MiniMax M2 GGUF aliases. Reject its
  eager-mode single-MI50 deployment shape as a reproduction or performance path
  for the `.20` ROCm7.2 full-BAR/P2P-on graph-mode lanes.
- existing request-artifact timing review: corrected GGUF TP4 fixed-token logs
  show the slow band in both wall timing and vLLM metrics (`c1_2000` about
  `71.934` wall / `73.015` decode TPS, `c1_10000` about `65.278` wall /
  `65.454` decode TPS). Aligned HF CUDAGraph replay timing artifacts are
  contaminated because the same event instrumentation collapses HF to strict
  `71.784`, `c1_2000` `72.653`, and `c1_10000` `65.151`, while the clean `.20`
  HF reproduction path is strict `114.725`, `c1_2000` `116.429`, and
  `c1_10000` `109.531`. Reject pure client cadence, metrics-settle, Python
  runner timing, and replay-event timing as next full-ladder explanations.
- Qwen3.5 GGUF method-selection audit: an existing `.20` load-time
  `quant_method_audit.tsv` confirms current fused module dispatch is already
  unquantized for the relevant F16 GGUF hot paths. Counts were `760`
  `UnquantizedLinearMethod` rows, `160` forced `UnquantizedFusedMoEMethod`
  rows, and `120` attention rows with no quant method. Sampled unquantized
  prefixes include `linear_attn.in_proj_qkvz`, `linear_attn.in_proj_ba`,
  `linear_attn.out_proj`, shared-expert `gate_up_proj`, shared-expert
  `down_proj`, and attention `qkv_proj` / `o_proj`. Reject missing
  packed-module mapping, fused-name lookup, or accidental GGUFLinear wrapper
  dispatch as the current MoE TP4 speed gap.
- MoE TP4 launch-shape closure: a direct-norm GGUF run on `.20` changed only
  the launch shape to `--max-num-seqs 1`, `--max-num-batched-tokens 1024`, and
  CUDAGraph capture sizes `[1, 2]`. The normal warmups -> uncapped strict ->
  `c1_2000` -> `c1_10000` ladder produced strict `72.020`, `c1_2000`
  `73.495`, and `c1_10000` `65.840` backend TPS. Reject launch-shape mismatch
  as the current GGUF-versus-HF MoE TP4 performance gap; continue with
  HIP/kernel mix, memory movement, request cadence, or reduced completed-work
  replay diagnostics that preserve the clean HF control.
- MoE TP4 debug-dump static graph comparison: existing `.20` broad graph
  debug-dump artifacts for corrected GGUF and HF TP4 compile to matching
  generated executable kernel bodies after comment, UUID/hash, and module-name
  prefix normalization. Both sides have `61` files, `14` generated kernel
  files, identical top vLLM op counts (`1134` all-reduce, `770` ROCm
  unquantized GEMM, `360` GatedDeltaNet core, `280` shared MoE, `120` unified
  KV-cache update, `120` unified attention with output), identical ATen op
  counts, and identical graph pattern counts. All `14` normalized generated
  executable kernel diffs have `0` lines. Promote static graph comparison as a
  low-perturbation diagnostic, but reject it as a complete explanation: the HF
  debug comparator used debug-dump compilation plus explicit
  `--enable-prefix-caching --mamba-cache-mode align`, and it is not yet proven
  to be the same launch as the clean fast v0.2.1 reproduction path.
- MoE TP4 release fastpath overlay on GGUF: a `.20` diagnostic mounted the
  current release `runtime/patches/fused_moe.py` TP4 top-k 8 MoE fastpath into
  the corrected GGUF TP4 path on the official release image. The overlay loaded
  and activated for single-token decode on all four TP workers, while rejecting
  larger graph-capture/prefill token shapes. The normal ladder completed
  strict-valid but below the HF/v0.2.1 TP4 band: strict `82.571`, `c1_2000`
  `84.284`, and `c1_10000` `74.402` backend TPS. Promote the fastpath mount as
  required for comparable future MoE GGUF TP4 tests. Reject "fastpath missing"
  as the complete throughput explanation; the remaining gap sits below the
  launch layer in source-path parity, kernel mix, memory movement, scheduler
  cadence, or other work outside the single-token FusedMoE fastpath.
- MoE TP4 generalized fastpath overlay on GGUF: a follow-up `.20` diagnostic
  mounted the TP8 overlay `files/gfx906_runtime/moe_tp8_overlays/fused_moe_tp8.py`
  into the same corrected TP4 GGUF route. The generalized overlay activated for
  token groups `4`, `2`, and `1` on all four TP workers, with `12` active
  fastpath entries and `196` larger-shape rejections. The normal benchmark
  ladder remained below the HF/v0.2.1 band: strict `82.958`, `c1_2000`
  `84.230`, and `c1_10000` `74.342` backend TPS. Reject the fastpath
  token-count guard as the remaining throughput explanation. The next source
  boundary should be HIP kernel mix/timing without replay-loop synchronization,
  memory movement around routing/logits/postprocess, scheduler cadence, or a
  reduced completed-work reproducer that preserves clean HF performance.
- MoE TP4 exact HF scheduler shape plus release fastpath: the `.20` run
  `moe35b_gguf_tp4_fastpath_exact_hfshape_dot20_20260627T065134Z` combined the
  release TP4 fastpath mount with `--max-num-seqs 1`,
  `--max-num-batched-tokens 1024`, prefix caching, `mamba-cache-mode=align`,
  and CUDAGraph capture sizes `[1, 2]`, matching the clean HF TP4 control
  scheduler shape. The ladder remained strict-valid but in the same GGUF
  post-fastpath band: strict `82.543`, `c1_2000` `84.195`, and `c1_10000`
  `74.314` backend TPS. Reject exact scheduler/capture shape, broad graph
  capture, and release fastpath absence as the remaining whole-gap explanation.
  Continue with lower-level HIP/kernel mix, memory movement, or reduced
  completed-work diagnostics that preserve clean HF performance.
- clean HF TP4 control artifact boundary: searched the `.20` GGUF workspace and
  local public checkout for raw artifacts matching the public high-band TP4
  reproduction values. The report value remains strict `114.725`, `c1_2000`
  `116.429`, and `c1_10000` `109.531`, but the nearby HF source/profiler
  directories are not clean controls. The broad launch comparator differs in
  scheduler/prefix-caching/parser shape, the exact-shape entrypoint failed
  before benchmarking on the hybrid KV layout ambiguity, and replay/debug
  comparators perturb the serving path. Any source-kernel comparison that needs
  a high-band HF baseline should first regenerate a clean HF control through
  the public v0.2.1 deploy plus benchmark path and then compare GGUF only after
  a new source-level change.
- regenerated clean HF TP4 control: ran the public v0.2.1 deployment and
  benchmark sequence on `.20` from commit `0c0ee21` with profile
  `moe35b_tp4_fullbar_p2pon`, the published image digest
  `sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e`,
  TP4, P2P-on, FP16, `MAX_MODEL_LEN=131072`, graph mode, async scheduling, and
  the release Qwen C1 top-k 8 MoE fastpath. Run directory:
  `<validation-workspace>/runs/hf_moe35b_tp4_clean_repro_20260627T074206Z/v02-profile-runs/moe35b_tp4_fullbar_p2pon_20260627T074206Z`.
  The normal warmups -> uncapped strict -> `c1_2000` -> `c1_10000` ladder
  produced strict `114.187`, `c1_2000` `116.623`, and `c1_10000` `109.748`
  backend TPS. Promote this as the clean HF control for future GGUF
  source-kernel comparisons. Reject additional unchanged HF controls or
  instrumented replay timing as substitutes for this high-band baseline.
- forced unquantized GGUF Linear dispatch: the `.20` scratch source run
  `moe35b_gguf_tp4_force_unquant_linear_dot20_20260627T075850Z` modified only
  GGUF quant-method dispatch so F16 `LinearBase` layers selected
  `UnquantizedLinearMethod`. Logs confirmed `760` forced unquantized linear
  selections plus `160` forced unquantized FusedMoE selections. The normal
  ladder remained strict-valid but in the same GGUF post-fastpath band: strict
  `82.065`, `c1_2000` `83.962`, and `c1_10000` `74.167` backend TPS. Reject
  residual GGUF linear wrapper dispatch as the remaining whole-gap source
  explanation. Continue below method selection: HIP/kernel mix, decode memory
  movement, graph-capture envelope, or reduced completed-work parity against
  the regenerated clean HF control.
- Kausik-A Qwen3.6 27B MI50 vLLM bundle:
  <https://github.com/Kausik-A/qwen3.6-27b-mi50-vllm>. The read-only inspected
  commit was `61b273d`. It vendors five Python patch files over
  `aiinfos/vllm-gfx906-mobydick` for single-MI50/eGPU Qwen3.6 27B GGUF
  serving. Useful source-reference items are `qwen35` / `qwen35moe` aliases,
  explicit HF tokenizer/config use, text-only Qwen3.5 override, M-RoPE bypass,
  `ssm_dt.bias` mapping, 2D-to-3D `conv1d.weight` reshape, `quant_config`
  plumbing for embedding/lm-head, text-only `IsHybrid` hooks, and GGUF
  tuple-shard fused QKVZ loading. Reject as a performance reproduction path for
  this goal because it is ROCm 6.3, eager mode, single GPU/eGPU, 4096 context,
  and quantized dense GGUF rather than the `.20` ROCm7.2 full-BAR/P2P-on TP8
  dense / TP4 MoE FP16-GGUF target.
- Dense GGUF forced SSM cache float32: the `.20` scratch run
  `dense27b_gguf_tp8_ssmfloat32_dot20_20260627T082830Z` added only
  `--mamba-ssm-cache-dtype float32` to the previous best dense GGUF CausalLM
  route. It corrected startup geometry to attention block size `400`, GPU KV
  cache `377,200` tokens, max concurrency `11.40x`, and successful graph
  capture. The normal ladder produced strict `62.992`, `c1_2000` `63.628`, and
  `c1_10000` `60.118` backend TPS, below the prior best GGUF dense values of
  strict `63.679`, `c1_2000` `64.353`, and `c1_10000` `60.617`. Promote as
  evidence that the GGUF CausalLM path can be forced to HF-like cache/page
  geometry. Reject cache dtype/page geometry as the remaining dense performance
  gap. Continue with lower-level decode kernel mix, memory movement,
  GDN/state-conversion work, graph lowering, or GGUF-loaded weight layout.
- Dense GGUF versus HF compile-debug graph surface: bounded `.20` debug runs
  `dense27b_gguf_tp8_compile_debug_20260627T090014Z` and
  `dense27b_hf_tp8_compile_debug_20260627T090636Z` reached health and wrote
  compile artifacts without a benchmark ladder. Both produced `616` debug-dump
  files, `77` files per rank, and equal counts for the major GDN/SwiGLU source
  markers (`gdn_attention_core`, `swiglu`, `qwen2_moe`, `in_proj_qkvz`).
  Reject missing GDN/SwiGLU lowering or broad graph-file shape as the remaining
  dense gap. Promote the GGUF-only embedding/materialization surface as the next
  narrow source candidate: `_apply_gguf_embedding` appeared `176` times in GGUF
  and `0` times in HF, with GGUF-only `masked_fill` / `bitwise_or` and a larger
  rank-0 `kernel_0`. Prior row-content checks mean this should be treated as a
  potential per-token overhead / graph-lowering issue, not a simple wrong-row
  correctness bug. Next source work should test an embedding-only unquantized
  dispatch path or low-overhead timing probe before any full benchmark ladder.
- Dense GGUF embed-tokens-only unquantized dispatch: the `.20` scratch run
  `dense27b_gguf_tp8_embedtokens_unquant_compile_debug_20260627T092322Z`
  mounted an env-gated `gguf.py` variant and forced only `model.embed_tokens`
  to `UnquantizedEmbeddingMethod`. All eight TP workers logged the forced
  dispatch and the server reached health, but the debug surface was unchanged
  from the prior GGUF compile-debug run: `_apply_gguf_embedding` `176`,
  `masked_fill` `232`, `bitwise_or` `168`, `vllm.all_reduce` `14680`, `616`
  debug-dump files, `176` compile-cache files, and rank-0 `kernel_0` `53,906`
  bytes. Reject simple `embed_tokens` method selection as the fix. Continue
  deeper in the GGUF embedding parameter/materialization path or generated
  custom-op construction; do not spend a full benchmark ladder on this patch.
- Dense GGUF skip-embed-qweight source probe: the `.20` scratch run
  `dense27b_gguf_tp8_skipembedqweight_compile_debug_20260627T093747Z` copied
  the current best dense GGUF source bundle and added an env-gated Qwen3.5
  overlay change:
  `VLLM_QWEN35_GGUF_SKIP_EMBED_QWEIGHT=1`. Together with
  `VLLM_GGUF_EMBED_TOKENS_UNQUANT=1`, this skipped only
  `model.embed_tokens` GGUF qweight override/materialization while leaving
  `lm_head` on the existing GGUF qweight path. It removed the
  `_apply_gguf_embedding` compile surface (`0` hits), wrote `856` debug-dump
  files and `208` compile-cache files, and shrank rank-0 `kernel_0` to `4,287`
  bytes. The normal ladder stayed coherent and strict-valid but did not
  improve: strict `63.606`, `c1_2000` `64.432`, and `c1_10000` `60.691`
  backend TPS. Reject embedding qweight removal as a dense throughput fix.
  Future source work should compare lower-level decode kernel mix, memory
  movement, GDN/state conversion, or GGUF-loaded weight layout against the
  clean HF dense control rather than repeating embedding dispatch surgery.
- Dense GGUF wrapper/prefix parity probes: the `.20` scratch runs
  `dense27b_gguf_tp8_conditional_wrapper_probe_20260627T101323Z` and
  `dense27b_gguf_tp8_langprefix_probe_20260627T101829Z` checked whether the
  remaining dense gap was caused by the GGUF path using text-only
  `Qwen3_5ForCausalLM` / `model.layers.*` prefixes while the HF control uses
  `Qwen3_5ForConditionalGeneration` / `language_model.model.layers.*` prefixes.
  The ConditionalGeneration probe did not reach health because vLLM treated the
  GGUF artifact as multimodal during memory profiling and entered the visual
  dummy path, where visual GGUF parameters were uninitialized. The narrower
  prefix-only CausalLM probe also failed before health because forcing the
  internal prefix to `language_model` left GGUF parameters uninitialized during
  compile/profile. Reject naive wrapper/prefix parity as a benchmark path.
  Promote the source boundary: the current GGUF loader/materialization route is
  coupled to the unwrapped text-only CausalLM prefix, and wrapper parity would
  require a real text-only wrapper that preserves GGUF materialization without
  advertising multimodal inputs. Continue below wrapper, embedding, cache/page,
  and launch-shape explanations.
- Dense `gdn_attention_core` custom-op timing: the `.20` request-window runs
  `dense27b_gguf_tp8_gdnop_timing_dot20_20260627T103024Z` and
  `dense27b_hf_tp8_gdnop_timing_dot20_20260627T104237Z` mounted the same
  direct custom-op timing `qwen3_next.py` over the published v0.2 image. After
  health, startup/profiling timing rows were cleared, a small warmup request was
  sent, timing rows were cleared again, and one measured 128-token request was
  sent. Both windows produced `1152` GDN timing rows. GGUF averaged
  `1,755,368` ns `last_ns`; HF averaged `1,951,620` ns. Layer-0 averaged
  `11,951,702` ns for GGUF and `16,107,229` ns for HF. Reject the registered
  GDN custom-op body as the remaining dense GGUF throughput gap. Continue below
  the core op with whole-decode kernel mix, memory movement, residual/norm and
  projection boundaries, scheduler/replay overhead, or loaded-weight layout
  outside the measured GDN body.
- Dense GGUF/HF generated-kernel diff review: an existing `.20` comparison
  artifact at
  `<validation-workspace>/runs/dense27b_gguf_hf_kernel_compare_20260627T110302Z`
  normalized the prior dense compile-debug dumps and compared generated bodies.
  Unlike the broad marker review, every generated source body differed:
  `48/48` generated kernel files and `520/520` pre-grad files had normalized
  diffs, with `33,582` and `143,824` diff lines respectively. GGUF retained
  `_apply_gguf_embedding` `176`, `masked_fill` `232`, and `bitwise_or` `168`
  marker hits, while HF had `0` for those markers; both sides still had
  `gdn_attention_core` `4,608` and `swiglu` `15,208` hits. Promote this as
  evidence that broad marker-count parity is insufficient. Reject it as final
  root-cause proof because the debug pair also differs by wrapper/cache
  geometry (`Qwen3_5ForCausalLM` GGUF versus `Qwen3_5ForConditionalGeneration`
  HF, with different KV-cache/concurrency values), and prior cache-geometry
  forcing did not improve GGUF. The next dense source step should build an
  aligned comparator before running another full benchmark ladder.
- Dense HF text-only CausalLM comparator: the `.20` run
  `dense27b_hf_tp8_textonly_causallm_compare_20260627T111054Z` used HF dense
  weights but forced the text-only `Qwen3_5ForCausalLM` path with
  `--language-model-only` and HF overrides. It kept TP8, full-BAR/P2P-on, FP16,
  `MAX_MODEL_LEN=131072`, graph capture, and the release dense overlay. The
  normal fixture produced strict `70.353`, `c1_2000` `70.978`, and `c1_10000`
  `66.428` backend TPS. Promote this as the aligned HF-side source comparator
  for future generated-kernel and replay analysis. Reject wrapper class alone
  as the dense GGUF gap and reject importing the external single-GPU/eager
  Kausik launch profile as a release-path performance fix.
- Dense GGUF vs HF text-only debug-dump comparison: the `.20` artifact
  `dense27b_gguf_hf_textonly_debugdump_compare_20260627T115023Z` compares the
  current best dense GGUF debug surface against the aligned HF text-only
  comparator with explicit vLLM debug-dump output. The accepted HF capture was
  `dense27b_hf_tp8_textonly_compile_debugdump_env_20260627T114602Z`; an earlier
  HF capture without explicit `--compilation-config` debug-dump output was
  rejected. The clean comparison found `520` pre-grad files on both sides, but
  all `520` differed. GGUF generated `48` kernel files (`6` per rank), while HF
  text-only generated `368` (`46` per rank). `gdn_attention_core` markers
  matched at `4608` each, but HF text-only had more `swiglu` and
  `vllm.all_reduce` markers, while GGUF alone retained `_apply_gguf_embedding`,
  `gguf_embedding`, and `index_select` surfaces. Promote this as the current
  best source comparison boundary and continue into generated-kernel
  partitioning, GGUF-loaded weight layout, embedding/index-select residue,
  graph replay/scheduler behavior, and memory-movement boundaries.
- Dense GGUF Inductor graph partitioning: the `.20` source probe
  `dense27b_gguf_tp8_graphpartition_probe_20260627T120119Z` added
  `use_inductor_graph_partition=true` to the current best dense GGUF bundle
  while preserving the release-like TP8 full-BAR/P2P-on lane. Generated kernels
  increased from the previous GGUF `48` total (`6` per rank) to `280` total
  (`35` per rank), compared with `368` total (`46` per rank) for the clean HF
  text-only comparator. GGUF-specific embedding/index-select residue remained:
  `_apply_gguf_embedding` `336`, `gguf_embedding` `472`, and `index_select`
  `3584` marker hits. The normal benchmark fixture produced strict `63.795`,
  `c1_2000` `64.556`, and `c1_10000` `60.826` backend TPS. Reject graph
  partitioning alone as a release-path fix. Promote it as source evidence that
  codegen partitioning matters but is not sufficient without changing lower
  level kernel mix, replay/memory movement, embedding/index residue, or
  GGUF-loaded weight layout.
- MoE GGUF TP4 Inductor graph partitioning: the `.20` source probe
  `moe35b_gguf_tp4_graphpartition_exact_hfshape_dot20_20260627T122551Z`
  preserved the exact-HF scheduler shape, TP4, P2P-on, FP16, release TP4 MoE
  fastpath, forced unquantized F16 expert path, and `MAX_MODEL_LEN=131072`,
  then added `use_inductor_graph_partition=true`. It reached health and wrote
  `184` generated kernels (`46` per rank), with marker counts including
  `gdn_attention_core` `4468`, `moe_forward_shared` `6944`,
  `rocm_unquantized_gemm` `16092`, and `vllm.all_reduce` `12492`. The normal
  benchmark fixture produced strict `82.412`, `c1_2000` `84.054`, and
  `c1_10000` `74.228` backend TPS. Reject graph partitioning alone as the MoE
  GGUF TP4 performance fix. Promote it as source evidence that generated-kernel
  count can be changed without moving the throughput class.
- GGUF existing timing/debug artifact review: reviewed the `.20` MoE runner /
  cudagraph replay timing artifacts and the dense/MoE graph-partition debug
  dumps before launching another run. Synchronized replay timing collapses HF
  into the GGUF band, so it is not a clean performance comparator: the
  timing-instrumented HF MoE ladder recorded strict `71.784`, `c1_2000`
  `72.653`, and `c1_10000` `65.151`, while the GGUF runner-timing ladder
  recorded strict `71.668`, `c1_2000` `72.936`, and `c1_10000` `65.400`.
  Dense rank-0 graph-partition debug counts still show GGUF-only
  `_apply_gguf_embedding`, `gguf_embedding`, and `index_select` residue, plus
  more `rocm_unquantized_gemm` and `all_reduce_inplace_kind` references than
  the aligned HF text-only comparator. The embedding-only fixes already failed
  to promote, so treat this as a broader GGUF materialization/generated-body
  boundary. Next work should use a reduced completed-work comparator or
  lower-level tensor-layout / memory-movement analysis, not another replay
  timing run.
- External Qwen3.6 GGUF compatibility reference: reviewed
  `Kausik-A/qwen3.6-27b-mi50-vllm` at public commit `61b273d`. Promote its
  source-level compatibility points for comparison: `qwen35` aliasing,
  text-only Qwen3.5/3.6 model registration, M-RoPE stripping, `ssm_dt.bias` to
  `linear_attn.dt_bias` mapping, MoE expert tensor aliases, GGUF `conv1d`
  reshape, `quant_config` propagation to `embed_tokens` / `lm_head`, and tuple
  shard handling for pre-fused GDN QKVZ tensors. Reject the repo's launch
  profile as a performance reference for the release path because it is
  single-GPU/eGPU, ROCm 6.3, Q6 GGUF, 4096 context, and eager execution. The
  relevant source follow-up is to compare its fused-loader and F16
  materialization assumptions against our current GGUF graph surfaces, not to
  copy its deployment settings.
- Dense generated-body standalone microbench boundary: rank-0 static scanning
  of the dense GGUF graph-partition dump versus the HF text-only comparator
  showed real GGUF-only generated surfaces, including `_apply_gguf_embedding`
  and rotary-cache `index_select`, plus much higher clone/reshape/GDN/GEMM
  generated-line counts. However, `GGUF-189` already proved that removing the
  embedding custom-op surface does not close the dense benchmark gap. A
  short-lived official-image container can run Torch/ROCm and can register the
  GGUF/QwenNext custom ops, but direct execution of the generated graph still
  requires vLLM `ForwardContext` and live no-compile GDN layer objects.
  Stubbing GDN would invalidate the measurement. Treat unchanged generated
  script microbenching as rejected until a real forward-context reproducer or
  non-perturbing live request-window profiler is built.
- Dense graph partition plus embed-qweight skip: the `.20` source probe
  `dense27b_gguf_tp8_graphpartition_skipembed_dot20_20260627T131645Z`
  combined `use_inductor_graph_partition=true` with
  `VLLM_QWEN35_GGUF_SKIP_EMBED_QWEIGHT=1` and
  `VLLM_GGUF_EMBED_TOKENS_UNQUANT=1` under the release-like dense TP8
  full-BAR/P2P-on lane. Startup succeeded, graph capture completed, and the
  normal fixture produced strict `63.747`, `c1_2000` `64.587`, and
  `c1_10000` `60.866` backend TPS. Reject this as a promotion path. It
  strengthens the source boundary: the remaining dense GGUF gap is below
  wrapper selection, M-RoPE stripping, embedding qweight materialization alone,
  and generated-kernel partition count alone. Continue with GGUF-loaded F16
  tensor layout, clone/reshape/memory movement, live request-window profiling,
  or a reduced completed-work forward-context comparator.
- Dense GGUF post-load tensor layout diagnostic: the `.20` source probe
  `dense27b_gguf_tp8_layoutdiag_dot20_20260627T135511Z` extended the GGUF
  loader to print layer-0 GatedDeltaNet tensor dtype, shape, stride,
  contiguity, and data pointer after weight loading/processing. On every TP
  rank, `linear_attn.in_proj_qkvz.weight`, `linear_attn.conv1d.weight`, and
  `linear_attn.out_proj.weight` were CUDA FP16 and contiguous with expected
  row-major strides. The server reached health and served a short coherent
  Qwen reasoning probe. Reject simple reload-time `.contiguous()` or tensor
  copying of those GDN weights as the next performance lever. The remaining
  dense GGUF source work should target live request-window profiling,
  generated graph/replay behavior, clone/reshape memory movement, or a reduced
  forward-context comparator that preserves real vLLM layer objects.
- Dense GGUF M-RoPE config restoration probe: the `.20` source probe
  `dense27b_gguf_tp8_graphpartition_skipembed_mropeprobe_dot20_20260627T141330Z`
  added the upstream Qwen3.6 M-RoPE config fields to the local GGUF text config
  while keeping the same image, F16 GGUF model, TP8, P2P-on, graph-partition,
  and embedding-qweight-skip launch. Startup, graph compilation, graph capture,
  and health all succeeded. The first request failed with the engine reporting
  that M-RoPE support was not implemented because `req.mrope_positions` was not
  available in the GGUF/text-only request path. Reject config-only M-RoPE
  restoration. Promote the source boundary: matching the HF M-RoPE graph path
  would require request/scheduler position plumbing, not only loader or config
  changes.
- Dense GGUF lm-head unquantized logits path: the `.20` source probe
  `dense27b_gguf_tp8_lmheadunquant_dot20_20260627T142800Z` kept the same dense
  F16 GGUF TP8 full-BAR/P2P-on graph-partition path and changed only the final
  logits path after `lm_head` F16 materialization by assigning
  `UnquantizedEmbeddingMethod()` to `self.lm_head.quant_method`. The normal
  fixture promoted with strict `70.350`, c1_2000 `71.396`, and c1_10000
  `66.842` backend TPS. A later `.20` lane-validation run from the same
  promoted path superseded those raw values with strict `70.505`, c1_2000
  `71.589`, and c1_10000 `66.967` backend TPS. This makes `lm_head`
  quant-method residue a confirmed dense GGUF throughput lever and shifts the
  next source check to whether MoE GGUF has the same final-logits overhead.
- MoE GGUF lm-head unquantized logits path: the `.20` source probe
  `moe35b_gguf_tp4_lmheadunquant_dot20_20260627T145000Z` copied the prior
  exact-HF-shape MoE GGUF graph-partition path and changed only
  `self.lm_head.quant_method` after GGUF load. The marker fired on all four TP
  ranks, but the normal fixture stayed in the MoE GGUF low-throughput band:
  strict `82.024`, c1_2000 `84.061`, c1_10000 `74.230` backend TPS. This
  rejects final-logits overhead as the MoE GGUF bottleneck and keeps the active
  source target in expert routing, MoE fastpath layout, memory movement, or a
  completed-work comparator against the clean HF MoE TP4 control.
- MoE GGUF static graph marker comparison: existing `.20` rank-0 debug dumps
  show the current GGUF MoE lm-head-unquant path with `46` generated kernel
  files and `1` pre-grad file, while the available HF broad graph debug dump has
  `14` generated kernel files and `41` pre-grad files. GGUF has many more
  marker references for `gdn_attention_core`, `moe_forward_shared`,
  `rocm_unquantized_gemm`, and `vllm.all_reduce`, plus `909` references to
  `_qwen35_effective_weight`. Treat this as source-boundary evidence only until
  a matched HF graph dump or timing-compatible comparator confirms whether the
  expanded GGUF graph surface is hot-path cost. Next inspect
  `_qwen35_effective_weight` for per-token reconstruction or extra graph-visible
  work.
- MoE GGUF base RMSNorm alias probe: the `.20` source probe
  `moe35b_gguf_tp4_base_rmsnorm_lmheadunquant_dot20_20260627T153000Z` replaced
  the custom GGUF `Qwen3_5RMSNorm` wrapper with
  `Qwen3_5RMSNorm = _Qwen3_5GemmaRMSNorm` while keeping load-time norm-offset
  correction and the lm-head unquantized logits path enabled. This removed
  `_qwen35_effective_weight` from the rank-0 debug dump, but the normal fixture
  stayed in the low MoE GGUF band: strict `81.808`, c1_2000 `84.403`, and
  c1_10000 `74.510` backend TPS. Reject `_qwen35_effective_weight` as the
  primary MoE bottleneck. Promote expert-layout / fastpath acceptance as the
  next source boundary because the run logged `24` fastpath layout rejections
  for GGUF expert tensors with `w1_shape=(256, 256, 2048)`,
  `w1_stride=(557056, 2176, 1)`, `w2_shape=(256, 2048, 128)`, and
  `w2_stride=(262144, 128, 1)`.
- MoE GGUF batch TopK8 fastpath probe: the `.20` source probe
  `moe35b_gguf_tp4_batch_fastpath_base_rmsnorm_lmheadunquant_dot20_20260627T153500Z`
  added batch-aware TopK8 kernels to the copied `fused_moe.py` overlay so the
  GGUF path accepted grouped token counts instead of requiring
  `hidden_states.size(0) == 1`. The run confirmed `fastpath_active=4` and
  `fastpath_rejects=0`, but the normal fixture stayed in the low band: strict
  `79.186`, c1_2000 `84.044`, and c1_10000 `74.194` backend TPS. Reject
  admission-only fastpath acceptance as the MoE promotion path. Continue below
  the call boundary: compare HF-vs-GGUF expert tensor materialization, generated
  kernel body, activation/reduction memory movement, and any layout-dependent
  loads inside the admitted path.
- External Qwen3.6 27B MI50 GGUF bundle: reviewed
  `Kausik-A/qwen3.6-27b-mi50-vllm` at commit `61b273d`. Its patch set covers
  Qwen3.5 / Qwen3.6 GGUF compatibility concerns: registry entries for
  `Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`, `qwen35` / `qwen35moe`
  GGUF aliases, text-only M-RoPE stripping, `ssm_dt.bias` to
  `linear_attn.dt_bias`, MoE gate/up/down aliases, GGUF `conv1d.weight`
  2D-to-3D reshaping, `quant_config` propagation into embeddings / lm-head, and
  tuple-shard fused projection loading. Promote as compatibility checklist
  evidence only. Reject its deployment shape as a LocalAIServers benchmark path:
  it is single-card / eGPU / eager / 4096-context oriented, while our target is
  the `.20` full-BAR/P2P-on ROCm7.2 release lane.
- MoE expert layout/value audit: the `.20` HF run
  `moe35b_hf_tp4_value_audit_dot20_20260627T160500Z` and GGUF run
  `moe35b_gguf_tp4_value_audit_dot20_20260627T161500Z` used a copied
  `unquantized_fused_moe_method.py` audit hook to sample expert tensors at
  `after_super`, `after_rocm_padding`, and `after_setup_kernel`. The existing
  layout audit had already shown identical HF/GGUF shape/stride/contiguity:
  `w13_weight` is ROCm-padded to stride `(557056, 2176, 1)` in both paths and
  `w2_weight` remains `(262144, 128, 1)`. The new value audit sampled layers
  `0`, `20`, and `39`, experts `0`, `7`, `127`, and `255`, and produced `576`
  rows per path. After ignoring TP-rank/device print order, the sampled values
  matched exactly as a multiset. Reject expert weight materialization as the
  primary MoE GGUF gap. The next source target is below admission and
  materialization: generated kernel body, prepare/finalize work, shared-expert
  work, activation/reduction memory movement, or a reduced completed-work
  timing comparator.
- MoE matched HF/GGUF static graph dump: the `.20` HF diagnostic
  `moe35b_hf_tp4_matched_graphdump_dot20_20260627T161707Z` used the same MoE
  TP4 graph-partition shape as the current GGUF path and produced matching
  rank-0 source-surface counts: `117` top-level files, `63` Python files, `46`
  generated kernel files, `1` pre-grad file, and `1` post-grad file in both
  paths. Marker counts also matched for `gdn_attention_core`,
  `moe_forward_shared`, `moe_forward`, `rocm_unquantized_gemm`, and
  `vllm.all_reduce`. The diagnostic failed before health at the known hybrid KV
  layout assertion, so this is static source evidence only. Reject the earlier
  broad graph-count expansion theory. Promote the narrower source leads:
  layer-name strings passed into custom ops differ between HF and GGUF,
  `_qwen35_effective_weight` remains visible but is already rejected by
  GGUF-206, and three `.best_config` files select different Triton configs after
  timing noise is stripped.
- MoE HF-layer-name alias probe: the `.20` source probe
  `moe35b_gguf_tp4_hflayeralias_lmheadunquant_dot20_20260627T170500Z` added
  `VLLM_QWEN35_GGUF_HF_LAYERNAME_ALIAS=1` to alias GGUF GDN and attention
  `static_forward_context` keys and custom-op names to HF-style
  `language_model.model.layers.N...` strings while preserving the existing
  lm-head-unquant MoE GGUF path. The alias markers fired and the normal
  fixture remained strict-valid, but measured throughput stayed unchanged:
  strict `82.488`, c1_2000 `84.001`, and c1_10000 `74.205` backend TPS.
  Reject GDN/attention layer-name keying as the main MoE gap. The active source
  inventory now narrows to the three semantic Triton `.best_config`
  differences, completed MoE prepare/finalize work, shared-expert work,
  activation/reduction movement, and a reduced HF/GGUF comparator that measures
  actual per-token work inside the admitted path.
- MoE Triton `.best_config` triage: the three semantic config differences from
  the matched HF/GGUF graph dump are not stable HF-versus-GGUF signatures.
  `35568...`, `9ca0...`, and `bde...` each showed rank-local config shuffling
  across HF and GGUF rather than a clean model-format split, and the debug dump
  did not map the config IDs directly to one generated source body. Reject
  forcing those configs as the next source patch. The stronger current
  inventory item is release-shape parity: compare GGUF MoE under the clean HF
  deployment shape before adding more lower-level kernel forcing.
- MoE GGUF release-shape scheduler probe: the `.20` source probe
  `moe35b_gguf_tp4_release_shape_lmheadunquant_dot20_20260627T170000Z`
  removed the constrained debug scheduler flags and used the cleaner
  release-style GGUF argument set while keeping FP16 GGUF weights, TP4
  full-BAR/P2P-on, `MAX_MODEL_LEN=131072`, and the normal benchmark sequence.
  Startup reached health and strict passed, but the ladder stayed in the low
  GGUF band: strict `81.329`, c1_2000 `85.178`, and c1_10000 `75.106` backend
  TPS. Reject launch-shape parity as the primary MoE GGUF bottleneck. Continue
  below that boundary: completed MoE work, shared/expert execution,
  prepare/finalize, activation/reduction movement, graph replay, request
  cadence, or HIP kernel mix/timing.
- HF MoE TP4 isolated-cache cache-capture control: the `.20` probe
  `moe35b_hf_tp4_cache_capture_dot20_20260627T172011Z` launched the official
  image with per-run TorchInductor, Triton, and vLLM caches to try to capture a
  clean high-band HF artifact set. It reached health and generated cache
  files, but the warmup decode band collapsed to the low diagnostic class:
  warmup 1 decoded at `73.819` backend TPS after heavy first-request prefill
  work, and warmup 2 decoded at `73.879` backend TPS. The log showed the raw
  image entrypoint path, not the full deploy-package/runtime-patch path:
  `max_num_batched_tokens=2048`, `enable_prefix_caching=False`, broad capture,
  and an unknown vLLM-core warning for
  `VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH`. Reject this run's cache files as an HF
  comparator for GGUF source work. Future high-band HF artifact captures must
  preserve the full deploy package and first prove release-band TPS.
- HF MoE TP4 full deploy-package cache-capture control: the `.20` probe
  `moe35b_hf_tp4_deploypkg_cache_capture_dot20_20260627T173555Z` copied the
  release deployment package and launched the same image through the package
  path. Startup showed the Qwen C1 TopK8 MoE fastpath applying on
  `E=256`, `N=256`, `K=2048`, `top_k=8`, FP16 shapes and completed CUDA graph
  capture. A Python capture-client invocation stalled before reaching vLLM;
  metrics stayed at zero running/waiting requests and zero prompt/decode
  counters, so that client-path probe is rejected. Replaying the generated
  request through the begin-think proxy with `curl` completed `2000` generated
  tokens from a `431` token prompt in about `18` seconds, with the vLLM logger
  reporting a steady generation interval of `115.7` tokens/s. Promote this as
  the HF control path for future source-level comparisons. The remaining GGUF
  inventory should focus below admitted MoE execution: prepare/finalize,
  shared/expert work, activation/reduction movement, graph replay, and HIP
  kernel mix/timing.
- MoE GGUF ConditionalGeneration wrapper loader probe: the `.20` disposable
  patch `qwen35moe-wrappermapper-conditional-20260627` attempted to load the
  FP16 GGUF through `Qwen3_5MoeForConditionalGeneration` instead of the current
  `Qwen3_5MoeForCausalLM` path. The patch added a wrapper
  `hf_to_vllm_mapper`, a missing `WeightsMapper` import, a `qwen3_5_moe` GGUF
  model-type alias, and a `vision_config.depth` fallback. It then failed before
  health inside `_get_gguf_weights_map(...)` because the internal GGUF dummy
  model map still saw wrapper-prefixed language parameters and vision
  parameters as unmapped. Reject this class-level mapper approach. Keep the
  wrapper/causal split as unresolved until a loader-map-level patch can build
  a text-only Qwen3.5-MoE map and translate the resulting text parameter names
  to wrapper names before the missing-parameter check.
- MoE GGUF ConditionalGeneration wrapper benchmark: the loader-map-level patch
  was completed in the same disposable patch tree by mapping
  `model.language_model.*` text names, sideloading unused visual parameters,
  skipping absent `mm_proj` sidecar loading for the text-only `qwen3_5_moe`
  GGUF, and delegating wrapper text weights into the existing CausalLM MoE
  loader. The server reached health and produced coherent Qwen output, then
  completed the normal TP4 benchmark ladder. Results remained low-band:
  strict `83.140`, c1_2000 `85.061`, and c1_10000 `74.996` backend TPS, with
  strict gate valid. Reject ConditionalGeneration wrapper parity as the MoE
  GGUF performance lever. The active inventory is now below wrapper selection:
  completed expert/shared-expert execution, prepare/finalize, activation /
  reduction movement, graph replay, request cadence, and HIP kernel mix/timing.
- MoE GGUF GatedDeltaNet projection graph parity: the `.20` static graph
  comparison between the high-band HF deploy-package control and the low-band
  GGUF wrapper path found matched counts for `all_reduce`, `moe_forward_shared`,
  `moe_forward`, and `gdn_attention_core`, but the GGUF graph had `440`
  `rocm_unquantized_gemm` references while the HF graph had `320`. The
  difference localized to split GGUF `in_proj_qkvz` plus `in_proj_ba` versus
  the HF release-style fused qkv/z/b/a projection. A disposable MoE GGUF patch
  `qwen35moe-fusedgdn-qkvzba-lmheadunquant-20260627` folded beta/alpha into
  `in_proj_qkvz`, loaded them with shard IDs `4` and `5`, and removed
  `in_proj_ba`. The resulting graph also had `320` `rocm_unquantized_gemm`
  references, reached health, returned coherent Qwen text, and completed the
  normal TP4 ladder. Results stayed low-band: strict `83.030`, c1_2000
  `84.778`, and c1_10000 `74.789` backend TPS. Reject extra GDN projection
  GEMMs and fused qkv/z/b/a graph shape as the primary MoE GGUF bottleneck.
  Continue below static graph marker parity: graph replay cadence, completed
  MoE prepare/finalize, shared-expert work, activation/reduction movement, HIP
  kernel mix/timing, or a reduced request-time comparator.
- MoE GGUF static graph parity with base RMSNorm: the follow-up `.20` patch
  `qwen35moe-fusedgdn-qkvzba-base-rmsnorm-lmheadunquant-20260627` combined the
  fused qkv/z/b/a projection with the base `GemmaRMSNorm` alias so the compiled
  graph no longer referenced `_qwen35_effective_weight`. The graph retained
  HF-style coarse marker counts: `rocm_unquantized_gemm` `320`,
  `moe_forward_shared` `200`, `moe_forward` `40`, `gdn_attention_core` `120`,
  `all_reduce` `486`, and `_qwen35_effective_weight` `0`. The normal ladder
  still landed at strict `82.534`, c1_2000 `84.892`, and c1_10000 `74.872`
  backend TPS. Reject further RMSNorm wrapper/static graph cleanup work as the
  active promotion path. The remaining inventory is runtime-level behavior:
  graph replay cadence, topk8/fused-MoE fastpath behavior, completed MoE
  prepare/finalize, shared-expert work, activation/reduction movement, HIP
  kernel mix/timing, or a reduced request-time comparator.
- MoE GGUF no-split static graph parity: the `.20` patch
  `qwen35moe-fusedgdn-qkvzba-nosplit-base-rmsnorm-lmheadunquant-20260627`
  moved the unused `mixed_qkv.split([q_size, k_size, v_size], dim=-1)` into
  the trace-only branch. This removed the final normalized graph operation
  difference seen against the high-band HF deploy-package control:
  `mixed_qkv.split([512, 512, 1024])` occurrences dropped to `0`, and the
  rank-0 graph line count matched HF at `8935`. The benchmark still stayed
  low-band: strict `81.220`, c1_2000 `84.852`, and c1_10000 `74.845` backend
  TPS. The release fastpath pattern remained grouped-shape rejection plus
  one-token activation. Reject static graph parity cleanup as the missing MoE
  GGUF lever. Next source/kernel work should inspect generated kernel body,
  graph replay cadence, expert layout consumption, activation/reduction
  movement, and HIP kernel mix/timing.
- MoE GGUF AWQ cache-factor probe: the `.20` no-split follow-up
  `moe35b_gguf_tp4_nosplit_awqoff_dot20_20260627T195932Z` attempted to force
  `VLLM_USE_TRITON_AWQ=0` while preserving the same FP16 GGUF, TP4
  full-BAR/P2P-on lane, `MAX_MODEL_LEN=131072`, P2P-on setting, release image,
  no-split static-parity patch, and normal benchmark ladder. The container
  environment showed the variable set to `0`, but all compiled rank
  `cache_key_factors.json` files still recorded `VLLM_USE_TRITON_AWQ=true`.
  The ladder remained low-band: strict `82.803`, c1_2000 `84.557`, and
  c1_10000 `74.608` backend TPS. Reject the env-only override as a valid
  AWQ-off source test. The useful source item is now the copied ROCm patch:
  `verify_quantization(...)` only guards the warning behind `quant == "awq"`,
  but unconditionally assigns `os.environ["VLLM_USE_TRITON_AWQ"] = "1"`.
  Patch that assignment and verify the cache factor before spending another
  full ladder on AWQ-cache behavior.
- MoE GGUF source-level AWQ false probe: the `.20` disposable patch bundle
  `qwen35moe-nosplit-awqflagrespect-20260627` added a ROCm patch guard
  controlled by `VLLM_GFX906_SKIP_ROCM_AWQ_AUTOENABLE=1`, then relaunched the
  same no-split FP16 GGUF TP4 full-BAR/P2P-on lane. All compiled rank cache
  factors recorded `VLLM_USE_TRITON_AWQ=false`, proving the earlier
  environment-only probe had not actually tested AWQ-off behavior. The normal
  ladder still stayed low-band: strict `83.940`, c1_2000 `85.177`, and
  c1_10000 `75.079` backend TPS. Reject AWQ cache-factor cleanup as the main
  missing MoE lever.
- MoE topk8 fastpath source boundary: source inspection of the release
  `fused_moe.py` shows `_try_qwen_c1_topk8_fastpath(...)` is deliberately a
  one-token path because it requires `hidden_states.size(0) == 1` and
  `topk_ids.size(0) == 1`. Grouped capture/decode shapes are therefore not
  expected to activate that kernel. A follow-up source/log audit also showed
  that `shape_or_layout` rejection messages are controlled by the separate
  `force` / `debug` setting, while `shape seen` messages require only the
  enable flag. This means the earlier HF-versus-GGUF grouped-rejection count is
  not a fair standalone comparator unless the debug mode is matched. The
  low-band GGUF logs still show padded `w1` stride `(557056, 2176, 1)` in the
  grouped standard path, but do not treat the absence of HF grouped-rejection
  logs as proof that HF activates a grouped fastpath. Do not chase grouped
  fastpath activation blindly. The remaining inventory should prioritize
  comparable high-band HF cache/debug capture, generated kernel body, grouped
  expert layout consumption in the standard fused-MoE path, graph replay
  cadence, activation/reduction movement, and HIP kernel mix/timing.
- MoE matched-debug HF fastpath control: the `.20` HF run
  `moe35b_hf_tp4_force_debug_cache_capture_dot20_20260627T211519Z` launched
  the native release TP4 profile with
  `VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH=force`. It logged `4` one-token fastpath
  activations and `55` grouped `shape_or_layout` rejections. The grouped
  rejection lines used the same padded expert layout as GGUF:
  `w1_shape=(256, 256, 2048)`, `w1_stride=(557056, 2176, 1)`,
  `w2_shape=(256, 2048, 128)`, and `w2_stride=(262144, 128, 1)`. This rejects
  padded grouped stride / grouped rejection visibility as a GGUF-only root
  cause. The same run is not a valid performance comparator: force-debug mode
  slowed the first warmup to `62.777` backend decode TPS with `206.191`
  prefill seconds. Keep debug visibility separate from benchmark promotion.
- MoE HF/GGUF summary-metrics comparator: read-only summary comparison between
  the clean `.20` HF TP4 control and the latest low-band GGUF TP4 probe shows
  the remaining gap is decode-phase work. Prompt tokens matched at `431` and
  prefill stayed near parity (`~0.32` seconds). HF c1_10000 used `91.118`
  decode seconds for `109.748` backend TPS, while GGUF used `136.944` decode
  seconds for `73.023` backend TPS. Reject prompt/prefill/proxy accounting as
  the primary gap. Continue with decode-window graph replay, HIP/Triton kernel
  mix, MoE prepare/finalize/postprocess memory movement, or a reduced
  completed-work comparator.
- MoE GGUF NCCL path cache-factor probe: the `.20` run
  `moe35b_gguf_tp4_nosplit_awqflagrespect_no_ncclpath_dot20_20260627T204804Z`
  copied the no-split/AWQ-false TP4 GGUF lane and removed only the explicit
  `VLLM_NCCL_SO_PATH=/rccl-overlay/install/lib/librccl.so.1` environment
  setting. All four compiled rank cache factors recorded an empty
  `VLLM_NCCL_SO_PATH` and `VLLM_USE_TRITON_AWQ=false`, matching the relevant
  high-band HF cache-factor values. The normal ladder still stayed low-band:
  strict `81.425`, c1_2000 `82.634`, and c1_10000 `73.023` backend TPS. The
  fastpath pattern was unchanged (`4` active, `204` rejected, `208` shape-seen).
  Reject NCCL-path env/cache parity as the primary source of the HF/GGUF split.
  Continue below environment/cache-key cleanup: steady decode HIP/Triton kernel
  mix, grouped expert execution in the standard fused-MoE path,
  activation/reduction movement, graph replay cadence, and completed-work
  comparison against the high-band HF TP4 control.
- MoE GGUF FusedMoE HF layer-name alias probe: the `.20` source probe
  `moe35b_gguf_tp4_fusedmoe_hfname_no_ncclpath_dot20_20260627T213900Z`
  targeted the final normalized TorchInductor cache-name difference by
  rewriting GGUF FusedMoE layer names to the HF-style
  `language_model.model.layers.*.mlp.experts` form after weight load. Startup
  recorded `160` alias events, then loaded and captured normally. The standard
  benchmark ladder remained strict-valid but low-band: strict `80.661`,
  c1_2000 `82.737`, and c1_10000 `73.111` backend TPS. Reject FusedMoE
  `ModuleName` / cache key string parity as the missing MoE GGUF lever. The
  current source inventory remains runtime decode work: generated kernel body,
  graph replay cadence, grouped standard fused-MoE execution, activation /
  reduction movement, and HIP/Triton kernel mix.
- MoE HF/GGUF generated `.kernel_perf` artifact comparator: a read-only join
  of the high-band HF deploy-package cache and the latest GGUF alias-probe
  cache found `76` `.kernel_perf` files on each side with no missing joined
  keys. Summed kernel-perf values were HF `2.106399` and GGUF `2.210239`,
  a `1.049x` ratio. This is too small to explain the observed request-level
  fixed-token decode gap by itself. Reject a single TorchInductor autotune miss
  as the complete MoE GGUF bottleneck. Keep high-ratio generated pointwise /
  reduction kernels as secondary clues, but prioritize request-window kernel
  counts, graph replay cadence, and completed-work accounting.
- MoE GGUF current CUDAGraph replay timing: the `.20` run
  `moe35b_gguf_tp4_current_cgreplay_timing_dot20_20260627T221003Z` sampled
  CUDAGraph replay while running the normal MoE TP4 ladder on the current
  strict-valid GGUF path. The ladder remained low-band: strict `81.280`,
  c1_2000 `82.538`, and c1_10000 `73.114` backend TPS. Sampled `FULL` replay
  count was `1952`, mean `12.054565` ms, min `10.880300` ms, max `15.272617`
  ms; sampled `PIECEWISE` replay count was `28`, mean `7.677879` ms. `FULL`
  replay drifted from roughly `11.51` ms early to `14.47` ms in the final call
  band, with all four TP ranks tracking the same range. Promote this as
  request-window replay-boundary evidence. Reject a single bad rank, proxy,
  tokenizer, prefill, NCCL-path, layer-name, or `.kernel_perf` aggregate
  explanation. The next useful source inventory item is lower-level decode work
  inside the replayed graph: generated kernel bodies, FusedMoE grouped expert
  execution, shared-expert/reduction movement, and extra GGUF-only completed
  work per generated token.
- MoE HF/GGUF CUDAGraph input metadata probe: the `.20` metadata runs
  `moe35b_hf_tp4_graphinput_meta_dot20_20260627T224016Z` and
  `moe35b_gguf_tp4_graphinput_meta_fixed_dot20_20260627T225531Z` captured
  graph-input signatures without tensor values. The useful summarized counts
  are HF `20764` metadata lines and GGUF `27588` metadata lines. HF showed
  `1044` entries with `shape=(N, 8, 128)` and stride `(3072, 128, 1)`,
  `1048` entries with `shape=(3072, 2048)`, and `1048` entries with
  `shape=(16, 2048)`. GGUF showed `1464` entries with `shape=(N, 8, 128)` and
  stride `(3088, 128, 1)`, `1464` entries with `shape=(3088, 2048)`, and no
  matching `shape=(16, 2048)` entries. Promote this as the first graph-input
  contract difference after static generated-source and `.kernel_perf`
  comparisons were insufficient, but source-state verification rejected it as a
  high-band comparator. The HF metadata launch mounted only the CUDAGraph
  metadata patch; it used the release image's stock split `qwen3_5.py` hash
  `587e313270cf78336b7ef9a6ab4df3201ce2008369bb6ae14e208d1d6d606d6f`. The
  release high-band patch hash was
  `71eaf52b5f85c022380599ae80ce0f478e989b6052a4f14a88ef4305edd3c046`, and the
  GGUF metadata source hash was
  `4359d9c3958047873b40b04cd174309df08266dabc0b81edc9cdbd92c19c16d7`; both
  release-patched HF and GGUF metadata sources use the fused `3088` qkv/z/b/a
  form. Therefore do not chase the `3072 + 16` versus `3088` contrast as the
  current performance lever. Rerun graph-input metadata only if it uses the
  release-patched HF source, or move directly to lower-level decode graph work:
  generated kernels, FusedMoE grouped expert execution, shared-expert/reduction
  movement, and extra GGUF-only work per token.
- MoE HF/GGUF high-band CUDAGraph replay comparator: the `.20` HF control
  `moe35b_hf_tp4_releasepatch_cgreplay_dot20_20260627T232912Z` mounted the
  release patch bundle and the replay-timing patch, then ran the normal
  benchmark ladder with curl POST mode after the first benchmark process hung
  before backend submission. It produced high-band HF numbers: strict
  `119.928`, `c1_2000` `121.255`, and `c1_10000` `114.020` backend TPS. The
  sampled FULL replay mean was `7.887792` ms over `1980` samples, with min
  `7.481852` and max `9.125901`; PIECEWISE mean was `7.153875` ms. The current
  GGUF comparator
  `moe35b_gguf_tp4_current_cgreplay_timing_dot20_20260627T221003Z` sampled
  FULL replay at `12.054565` ms over `1952` samples, with min `10.880300` and
  max `15.272617`; PIECEWISE mean was `7.677879` ms. HF FULL replay drifted
  from `7.730574` ms to `8.664663` ms by 250-sample bins, while GGUF drifted
  from `11.515907` ms to `14.436282` ms. Promote replayed decode graph /
  adjacent completed work as the active source inventory target. Reject the
  stale stock-source HF replay run and reject static `.kernel_perf` aggregate
  timing as sufficient. Continue with request-window HIP/Triton kernel counts,
  generated-kernel body diffs, standard FusedMoE grouped expert execution,
  shared/routed expert reduction movement, and GGUF layout/materialization work
  inside the replayed decode path.
- MoE generated graph/source identity and corrected metadata comparator: direct
  cache comparison between the high-band HF replay-control and current GGUF
  replay-control found `76` joined `.kernel_perf` keys with no HF-only or
  GGUF-only keys. It also found `208` generated Python files on each side and
  `208` common generated Python content hashes, with zero HF-only or GGUF-only
  generated Python hashes. Joined `.kernel_perf` sums were HF
  `2.227507998238` and GGUF `1.860795992897` (`0.835371x` GGUF/HF), so static
  generated code and static generated-kernel perf do not explain the slower
  GGUF replay. The corrected release-patched HF metadata run
  `moe35b_hf_tp4_releasepatch_graphinput_meta_dot20_20260627T235328Z` produced
  `28136` metadata lines and confirmed the fused `3088` graph-input family:
  `shape=(3088, 2048)` appeared `1496` times, `shape=(3072, 2048)` appeared
  `0` times, and `shape=(16, 2048)` appeared `0` times. The GGUF metadata run
  had `27588` lines, `1464` `shape=(3088, 2048)` entries, and also `0`
  `3072` or separate `16 x 2048` entries. Promote runtime invocation/count and
  custom-op/materialization profiling as the active source target. Reject
  generated Python source diffs, missing generated-kernel cache entries, and
  the stale stock-source metadata contrast as current bottlenecks.
- MoE branch-internal trace guardrail: internal decoder/GatedDeltaNet trace
  activation must call `_qwen35_boundary_trace_dir(trace_prefix)` before marking
  the one-shot trace as done. A scratch env-only trace activation was rejected
  because server profiling consumed the trace before the request gate existed,
  leaving only `model.final_hidden` output. The corrected GGUF scratch patch
  `qwen35moe-gguf-gatedinternaltrace-20260628T001219Z` captured layer-0 and
  layer-39 summaries for the faithful two-token branch probe without changing
  model semantics. It reproduced the `Here` over `Thinking` branch split while
  matching HF activation norms at layer 0 and layer 39. Promote full last-token
  vector dumps plus layer-wise cosine comparison as the next source diagnostic.
  Reject another norm-offset, graph/eager, or gross activation-scale fix until
  hidden direction drift is localized.
- MoE layer-wise vector cosine trace: scratch vector-trace patches
  `qwen35moe-hf-vectortrace-20260628T002050Z` and
  `qwen35moe-gguf-vectortrace-20260628T002050Z` added request-gated
  `row_values` dumps to the boundary trace helper, then replayed the faithful
  two-token HF/GGUF branch prompt on `.20`, TP4, P2P-on, eager mode. The
  comparison artifact is
  `moe35b_vector_cosine_compare_20260628T002050Z/cosine_summary.tsv`. The
  first strong signal is early residual direction drift, not magnitude: layer-0
  input and post-input norm cosines are `1.000000000` and `0.999999943`, but
  layer-0 `post_attention` drops to `0.988626168`,
  `post_attention_layernorm` to `0.982377650`, and low-norm `post_mlp` to
  `0.849558298`. By layer 10, `pre_input_layernorm` is `0.897712368`; by layer
  30 it is `0.788824286`. Promote layer-0 through layer-10 sparse
  MoE/router/expert output and GatedDeltaNet/residual contribution tracing as
  the next source target. Reject another final-logit-only, tokenizer, norm, or
  graph/eager experiment as the next step.
- MoE sparse component trace: scratch component-trace patches
  `qwen35moe-hf-componenttraceprefix-20260628T011015Z` and
  `qwen35moe-gguf-componenttraceprefix-20260628T011015Z` fixed the first
  component wrapper by explicitly assigning `self.mlp.prefix` after sparse
  block construction. The corrected `.20` trace pair
  `moe35b_hf_tp4_componenttraceprefix_eager4k_dot20_20260628T011830Z` and
  `moe35b_gguf_tp4_componenttraceprefix_eager4k_dot20_20260628T011830Z`
  captured `moe_input`, `routed_expert_output`, `shared_expert_output`,
  `pre_reduce_output`, and `post_reduce_output` for layers `0`, `1`, `2`,
  `5`, and `10`. The compare artifact
  `moe35b_component_cosine_compare_20260628T011830Z/cosine_summary_with_header.tsv`
  shows layer-0 `mlp.moe_input` remains close (`0.982377650`), while layer-0
  `mlp.shared_expert_output` averages `0.706940579` and layer-1
  `mlp.routed_expert_output` averages `0.693710627` with a minimum of
  `0.298821094`. Layer-2 `post_attention` is already `0.655290126`, and
  layer-2 `post_mlp` is `0.756801243`. This promotes GGUF expert/shared-expert
  materialization, expert shard ordering, and internal-router visibility as
  active source inventory targets. It rejects reduce-only debugging as the
  next primary move because TP-local expert outputs diverge before the reduce
  step.
- MoE GGUF expert-loader source inspection: `gguf_loader.py` routes
  `blk.*.ffn_down_exps.weight`, `blk.*.ffn_gate_exps.weight`, and
  `blk.*.ffn_up_exps.weight` into `model.layers.*.mlp.experts.*` parameters.
  `Qwen35MoeTensorProcessor` inherits the MiniMax-style merge path for
  `gate_up_proj`: `gate` and `up` are written into contiguous halves along
  shard dimension `1`, while `down` is stored separately. This is now the most
  concrete correctness hypothesis to test against HF: verify selected expert
  IDs and inspect whether GGUF's stored `gate`, `up`, and `down` expert tensor
  layout/order matches vLLM's expected `w13_qweight` and `w2_qweight` layout.
  If routing IDs match but outputs do not, prioritize expert tensor
  materialization and shard-order fixes before any new benchmark ladder.
- MoE router top-k trace: scratch overlays
  `qwen35moe-hf-routertopk-20260628T012050Z` and
  `qwen35moe-gguf-routertopk-20260628T012050Z` mounted patched
  `fused_moe/layer.py` and `router/base_router.py` so each FusedMoE router
  records request-gated logical `topk_ids` and `topk_weights` before EPLB
  mapping and before expert execution. The comparison artifact
  `moe35b_routertopk_compare_20260628T012050Z/topk_compare_with_header.tsv`
  compared `480` rows. Ordered top-k matches were `0`; top-1 matched in
  `268` rows; average set overlap was `5.933 / 8`, with minimum `1 / 8`.
  Event-0 layer-0 HF IDs were `222,223,186,73,199,20,175,92`; GGUF IDs were
  `223,73,199,234,161,92,222,186`. This promotes internal-router/gate
  materialization, gate input, score-correction bias, and top-k scoring math as
  active source targets. It rejects expert-kernel-only and reduce-only paths
  until selected experts match.
- MoE internal-gate logit trace: scratch overlays
  `qwen35moe-hf-gatelogits-20260628T020000Z` and
  `qwen35moe-gguf-gatelogits-20260628T020000Z` added a request-gated trace to
  `fused_moe/runner/default_moe_runner.py` immediately after the internal
  gate call. The comparison artifacts in
  `moe35b_gatelogits_compare_20260628T020000Z/` show
  `e_score_correction_bias=None` on both HF and GGUF, `192` compared
  rank/layer/event rows, `0` exact top-16 matches, `108` top-1 matches,
  average router-logit cosine `0.998827477`, and minimum router-logit cosine
  `0.995624130`. The same run's boundary trace shows the pre-gate MoE input is
  already drifting: layer-0 cosine `0.982377650`, layer-1 `0.935772504`, and
  layer-2 `0.903625660`. This rejects a current `exp_probs_b` /
  score-correction-bias fix and moves the active source inventory target
  upstream to layer-0/layer-1 attention, GatedDeltaNet contribution ordering,
  residual add, and post-attention norm before the first MoE gate.
- MoE boundary-trace first-drift mining: the gate-logit run's existing
  boundary traces were parsed into
  `moe35b_gatelogits_compare_20260628T020000Z/boundary_all_common_cosines.tsv`.
  Layer-0 is effectively identical through `post_input_layernorm`,
  `linear_attn.input_hidden`, `mixed_qkvz`, `ba`, and
  `core_attn_out_raw`. The first notable layer-0 direction drop is
  `linear_attn.out_proj_input` at cosine `0.878082862`; `post_attention` is
  `0.988626168`, first MoE input is `0.982377650`, and layer-0 `post_mlp`
  falls to `0.849558298`. Raw non-SSM norm-weight traces differ by the expected
  GGUF offset format and are not the next fix target because the active runtime
  bridge produces matching effective norm behavior at the first input norm.
  Promote layer-0 output-projection input construction / shape-stride
  semantics and controlled first-MoE route forcing as the next source fork.
- MoE layer-0 route-forcing diagnostic: scratch overlay
  `qwen35moe-gguf-forcehf-l0route-20260628T023000Z` patched
  `fused_moe/router/base_router.py` to force the GGUF layer-0 event-0
  final-row top-k route to the HF route before expert execution. The forced
  route used IDs `222,223,186,73,199,20,175,92` and improved layer-0
  `post_mlp` agreement to `0.977170526`, with layer-1 input also at
  `0.977170526`. The branch still followed the GGUF wrong path and later
  early-layer routed outputs diverged again. Promote route replay as a
  correctness diagnostic and early-layer top-k amplification as confirmed.
  Reject layer-0-only route forcing as a fix or benchmark promotion. The next
  source experiment should build a small HF route-map replay for layers `0`
  through `2` and compare branch output plus layer-2/layer-5 cosines before any
  performance ladder.
- MoE multi-layer route replay: valid every-call replays for layers `0`
  through `2`
  (`moe35b_gguf_tp4_forcehf_l0l2route_everycall_eager4k_dot20_20260628T021341Z`)
  and layers `0` through `5`
  (`moe35b_gguf_tp4_forcehf_l0l5route_everycall_eager4k_dot20_20260628T021826Z`)
  confirmed HF routes in the actual request trace but did not recover the HF
  branch. Both produced a blank/newline token path. l0-l5 replay improved
  layer-5 `post_mlp` to `0.928562586` and layer-10 `post_mlp` to
  `0.907384666`, but layer-2 `post_attention` stayed low at `0.646410402`.
  Reject route replay as a source fix and promote layer-1/layer-2
  attention/residual instrumentation: GatedDeltaNet core output, `z`/norm
  layout, output-projection input construction, output-projection result, and
  residual add before the post-attention norm. Also record the rejected
  profile-consumed run pattern: do not create `trace_gate` before launch for
  request-gated diagnostics.
- MoE attention/residual mining after l0-l5 route replay:
  `moe35b_attention_residual_compare_20260628T021826Z` compares the HF
  gate-logit trace to both unforced GGUF and valid l0-l5 route replay. Route
  replay improves MoE residual outputs but not the linear-attention projection
  boundary. Layer-2 `linear_attn.out_proj_input` is `0.668013175` unforced and
  `0.665483755` after l0-l5 replay; layer-2 `post_attention` is `0.655290126`
  unforced and `0.646410402` after replay. A local 8-head permutation probe on
  `out_proj_input` rejected simple head-order mismatch because identity was
  already best for layers `0`, `1`, `2`, and `5`. Promote a trace-only source
  diagnostic around the full final-token `core_attn_out_normed` vector before
  reshape/rearrange and the final `out_proj_input` vector. This should decide
  whether the next fix is the reshape/rearrange contract or an earlier
  GatedDeltaNet core / `z`-norm interaction.
- MoE full-token output-projection trace: scratch overlays
  `qwen35moe-hf-fulltokenoutproj-20260628T022943Z` and
  `qwen35moe-gguf-fulltokenoutproj-20260628T022943Z` added request-gated
  final-token traces around `core_attn_out_normed` and `out_proj_input`.
  The comparison artifacts in
  `moe35b_fulltokenoutproj_compare_20260628T022943Z/` show a strong
  layout-adjacent signal. Layer-0 `core_attn_out_normed_last_token_flat`
  first-rank cosine is `0.878082862`, matching current `out_proj_input`, while
  `core_attn_out_normed_last_token_alt_heads_first` is `0.999999746`. Layer-2
  shows current `out_proj_input` first-rank cosine `0.505315936` versus
  alternate heads-first `0.856964249`. This promotes `qwen3_5.py` shape/stride
  semantics around `core_attn_out`, norm, reshape, and `out_proj_input` as the
  active source inventory item. A copied candidate patch
  `qwen35moe-gguf-stateconvert-20260628T024100Z` globally reshaped the
  post-norm tensor as heads-first/token-second when
  `VLLM_QWEN35_GGUF_LINEAR_ATTN_STATE_CONVERT=1`; the run
  `moe35b_gguf_tp4_stateconvert_eager4k_dot20_20260628T024100Z` returned a
  different wrong branch and dropped layer-0 `post_attention` cosine to
  `0.210117847`. Reject that global conversion. The next source experiment
  must trace sequence-wide shape and stride before/after the GatedDeltaNet
  backend, norm, reshape, and flatten so a narrower boundary-correct fix can be
  tested.
- MoE activation/weight conversion rejects: two additional copied candidates
  confirm that the close heads-first final-token view is not sufficient as a
  direct patch. `qwen35moe-gguf-prenormstateconvert-20260628T031000Z` converted
  the GDN output before flattening and before norm pairing; the run
  `moe35b_gguf_tp4_prenormstateconvert_eager4k_dot20_20260628T031000Z`
  returned the wrong `SpaceItemSpaceItem` branch and dropped layer-0
  `post_attention` cosine to `0.112992007`. The paired
  `qwen35moe-gguf-postnorm-nooutprojinv-20260628T032000Z` candidate combined
  post-norm heads-first activation conversion with disabling inverse-column
  conversion for `linear_attn.out_proj.weight`; the run
  `moe35b_gguf_tp4_postnorm_nooutprojinv_eager4k_dot20_20260628T032000Z`
  returned the wrong `ymar Coll` branch and dropped layer-0 `post_attention`
  cosine to `-0.096042309`. Reject pre-norm conversion and paired post-norm /
  non-inverted-output-projection conversion. The active source inventory target
  is now the combined contract among GGUF V-head reorder helpers
  (`_qwen35_inverse_v_rows`, `_qwen35_inverse_v_cols`), GDN output storage,
  `z`/norm pairing, and output-projection column layout.
- MoE static linear-attention weight contract probe: read-only comparison
  artifacts
  `moe35b_linear_attn_weight_contract_lazy_20260628T031701Z/` and
  `moe35b_ssm_scalar_transform_probe_20260628T031923Z/` confirm that direct
  q/k layout is correct and the existing inverse value-head row/column helpers
  exactly recover HF layout for `v`, `z`, `conv_v`, and `out_proj`. This
  rejects another raw weight-column patch as the next source step. The raw
  GGUF SSM scalar/projection tensors still differ from HF, but those values
  have loader-specific semantics. The next inventory item should inspect
  active loaded values or graph inputs after loader transforms, not raw GGUF
  bytes alone.
- MoE graph replay/structural profiling: comparable graph dumps from
  `moe35b_gguf_tp4_hflayeralias_lmheadunquant_dot20_20260627T170500Z` and
  `moe35b_hf_tp4_matched_graphdump_dot20_20260627T161707Z` have identical
  high-level counts for `gdn_attention_core`, `fused_moe`, `moe_forward`,
  `rocm_unquantized_gemm`, qkvz/ba/out projection parameters, reshape/view/
  contiguous patterns, and all-reduce calls. Replay timing still differs
  materially: GGUF FULL replay averages `12.054565` ms, while the HF
  release-patch control averages `7.887792` ms. Together with prior GDN,
  FusedMoE, logits, and sampler timing probes, this promotes kernel selection,
  graph-visible input stride/value metadata, and replay drift as the current
  performance inventory target.
- MoE compile-cache artifact comparison:
  `moe35b_graph_kernel_surface_compare_20260628T032546Z/` now includes
  `concise-content-summary.md`, `kernel-normalized-compare.md`, and
  `best-config-distribution.md`. GGUF and HF generated the same kernel-family
  marker counts: `rocm_unquantized_gemm` `1360`, `all_reduce` `980`,
  `moe_forward_shared` `832`, `async_compile.triton` `200`,
  `gdn_attention_core` `120`, `unified_kv_cache_update` `80`, and
  `unified_attention_with_output` `48`. The generated Python totals were close
  (`7,780,819` bytes / `90,990` lines for GGUF versus `7,761,367` bytes /
  `90,630` lines for HF), and normalized first diffs were symbol-numbering
  differences. The real divergence is lower-level: GGUF has `97` unique
  `.best_config` hashes, HF has `89`, and only `60` are common. Promote
  graph-input manifest and best-config-by-kernel-role comparison. Reject
  missing generated kernel families or high-level graph shape as the next
  explanation.
- MoE cache-key factor normalization: rank-0 `cache_key_factors.json` differs
  in `code_hash`, `config_hash`, and `VLLM_USE_TRITON_AWQ`. The AWQ difference
  is already covered by the env-only and source-level AWQ-false probes, which
  did not promote. Keep it normalized in future graph comparisons, but reject
  AWQ-only work as the next source slice.
- MoE graph-input and best-config manifest: the forward graph signatures are
  identical (`576` inputs each, same manifest hash), and generated-kernel
  tensor metadata is identical (`2112` entries, `112` unique shape/stride/
  device strings, same manifest hash). The same `136` `.best_config` paths
  exist on both sides. Raw best-config hashes differ on `81` paths, but after
  normalizing out timing/cache-hash noise only `14` paths differ in actual
  autotune parameters. Promote a small compile-cache normalization experiment
  that forces those `14` GGUF choices to the HF parameter values before
  recompilation. Reject broad graph-input/stride rewrites based on the current
  evidence.
- MoE best-config preseed probe: the disposable `.20` run
  `moe35b_gguf_tp4_bestconfig_hfpreseed_dot20_20260628T040600Z` copied the HF
  `.best_config` files for the `14` path-normalized parameter mismatches into
  a fresh GGUF compile cache before launch. After the normal warmup ladder, the
  final cache retained HF choices for `9` paths and reverted to GGUF choices
  for `5` paths. Benchmark output remained low-band: strict `82.686`,
  `c1_2000` `84.181`, and `c1_10000` `74.301` backend TPS. FULL graph replay
  averaged `11.771824` ms, only a small improvement over the earlier GGUF
  `12.054565` ms and still far from the HF `7.887792` ms. Reject best-config
  selection as the main source-kernel blocker. The next source inventory should
  focus on runtime value/state or replay scheduling after graph capture, not
  another standalone autotune-config copy.
- MoE replay-control generated-code parity: a read-only `.20` follow-up
  compared the current GGUF replay-control run with the release-patched HF
  replay-control run and the corrected release-patched HF graph-input metadata.
  The valid HF comparator uses the same fused `3088` graph-input family as
  GGUF; the stale stock-HF `3072 + 16` contrast is rejected as a live patch
  target. Both replay-control roots had the same generated-source marker
  counts (`208` `.py`, `140` `.best_config`, `76` `.kernel_perf`, `4372`
  `triton_` markers, `812` `all_reduce`, `444` `moe_forward`, `100` clone,
  and `672` copy markers). Hashing generated `.py` files by content produced
  `208/208` common hashes with no HF-only or GGUF-only Python source bodies.
  Per-rank FULL replay timings were balanced on each path, while GGUF stayed
  around `12.054565` ms versus HF around `7.887792` ms. Reject generated
  Python code-body selection and a single-rank failure as the remaining
  performance explanation. Promote graph-input value signatures, runtime
  state/activation layout after loader transforms, and replay-time scheduling
  around those values as the next source-kernel slice.
- MoE graph-input value signatures: the `.20` HF and GGUF value-signature
  probes both logged `8092` lines, `384` capture headers, and `288` replay
  headers. Replay header structure matched exactly, while sampled values
  differed in expected graph-input tensors such as projection/weight-shaped
  arguments. This promotes value/state inspection only as a narrowed source
  slice. It rejects missing graph-input replay families, replay-header
  scheduling differences, and raw sampled-value differences as direct patch
  targets.
- MoE TopK8 debug boundary: the first value-signature run showed GGUF grouped
  `shape_or_layout` rejections while HF did not, but later matched-debug HF
  evidence shows grouped rejections are not GGUF-only. The release TopK8 helper
  remains one-token-only. Do not spend another source pass on grouped TopK8
  rejection unless a new patch also changes correctness, replay timing, or a
  grouped-token kernel contract.
- MoE current fused-`qkvzba` post-norm gather reject: scratch patch
  `qwen35moe-gguf-current-fusedqkvzba-postnorm-headgather-20260628T0525Z`
  tested `VLLM_QWEN35_GGUF_OUTPROJ_HEADS_FIRST_GATHER=1` against the current
  fused GGUF branch. The short branch prompt returned a non-coherent glyph, so
  the candidate was rejected before the benchmark ladder. The source target is
  lower than final `out_proj` row gathering: inspect the GatedDeltaNet custom
  op write/consume contract, active loaded SSM/GDN state, and norm pairing with
  `z`.
- MoE GDN boundary layout metadata: scratch patch
  `qwen35moe-gguf-gdn-layoutmeta-20260628T0115Z` traced stride, storage offset,
  and contiguity for `core_attn_out_raw`, `z_for_norm`,
  `core_attn_out_normed`, final-token views, `out_proj_input`, and
  `out_proj_output` on selected layers. The coherent short request showed
  normal contiguous layouts and expected strides through the Python-visible
  boundary; the only nonzero storage offset was the intentional final-token row
  view. Reject a visible stride/storage-offset anomaly as the current MoE GGUF
  blocker. Promote lower-level inspection of `gdn_attention_core` write/consume
  order, recurrent GDN state/value semantics, and replay scheduling around
  those values.
- MoE replay timing bucket shape: a read-only re-read of existing GGUF, HF,
  and HF-best-config-preseed replay logs shows the gap exists early and widens
  late. GGUF baseline FULL replay averages `11.220266` ms in calls `0-511`
  and `14.970377` ms near calls `31232-31743`; the HF release-patch control is
  `7.601556` ms early and `8.891808` ms late. Best-config preseed only reduces
  the GGUF buckets to `11.024393` ms early and `14.737254` ms late. Promote
  early/late state-cache and replay-window diagnostics. Reject startup-only,
  warmup-only, and best-config-only explanations.
- MoE sparse call-set state trace: scratch patch
  `qwen35moe-gguf-callset-state-20260628T0535Z` sampled GGUF decoder/GDN
  state at calls `2`, `64`, `128`, `256`, and `511` on `.20` under a bounded
  eager 4K diagnostic. It showed recurrent SSM state norms growing sharply
  across decode while post-layernorm and post-MLP norms stayed bounded. This
  promotes a paired HF/GGUF call-set comparator around recurrent state/cache
  semantics and the `gdn_attention_core` write/consume contract. It rejects a
  GGUF-only conclusion from the scratch trace and also rejects the current
  substring prefix matcher for formal comparison; tighten prefix matching to
  exact path-token matches before the HF comparator.
- MoE clean-HF call-set comparator: scratch patch
  `qwen35moe-hf-callset-counterfix-20260628T0815Z` fixed the clean-HF trace
  counter by incrementing outside the trace-active block, then ran the bounded
  `.20` HF TP4 eager diagnostic
  `moe35b_hf_tp4_callset_counterfix_eager4k_tuned_dot20_20260628T0825Z`.
  The useful comparator mounted the tuned MoE config, used P2P-on and FP16,
  and intentionally did not mount the release patch bundle because that bundle
  replaces the same `qwen3_5.py` path used by the trace source. The run
  captured calls `0`, `1`, `2`, `64`, `128`, `256`, and `511` and returned
  coherent capped text. HF and GGUF recurrent SSM state growth are the same
  order of magnitude, with layer-0 essentially identical early and layer-10
  within about five percent late. This rejects a broad GGUF-only SSM state
  explosion or SSM rescale patch as the next source target. The remaining
  source inventory target is lower-level replay/runtime behavior and late
  downstream signal consumption: post-attention layernorm and out-projection
  input attenuation, graph replay cadence, generated kernel runtime behavior,
  memory movement, or value materialization outside the matching generated
  Python graph bodies.
- MoE replay value-signature group mining: read-only summaries
  `value_signature_grouped_20260628.tsv` and
  `value_signature_diff_counts_20260628.tsv` were generated under the existing
  `.20` `moe35b_valuesig_replay_compare_20260628T0450Z` artifact root. The
  pass paired normalized HF/GGUF replay tensor arguments and grouped by
  mode/token count/argc/tensor-arg count/arg index/shape/stride. It found no
  missing replay family and many exact static/vector matches. The strongest
  repeated delta is the activation-shaped
  `argc=11 tensor=8 arg=4 shape=(16, 2048)` family: `36/36` records differed,
  with average sampled absolute sums `349.976942` for HF and `92.737499` for
  GGUF. The generated-source map
  `argc11_tensor8_generated_source_map_20260628.txt` identifies that family as
  a compiled MoE expert fragment around `moe_forward_shared`, all-reduce, and
  `rocm_unquantized_gemm_1`; the high-delta `arg4` is an output/mutation
  buffer, not a consumed semantic activation. Promote the fragment as a
  source-boundary target, but reject the stale output-buffer value as a patch
  target. The next useful probe should inspect consumed inputs and returned
  buffers around this fragment before any unchanged benchmark ladder.
- MoE replay pre/post value-signature probe: scratch patch
  `cudagraph-replay-prepost-value-signature-20260628T064447Z` extended the
  cuda-graph logger to emit `replay_post` records immediately after
  `entry.cudagraph.replay()`. The `.20` GGUF TP4 run
  `moe35b_gguf_tp4_prepost_valuesig_dot20_20260628T064605Z` produced balanced
  `384` replay and `384` post records from a coherent short request. The only
  paired pre/post value changes were output/mutation slots:
  `argc=12 arg[5]`, `argc=17 arg[5]`, `argc=11 arg[4]`, and `argc=9 arg[4]`.
  The matching metadata maps those slots to large `(tokens, 2048)`
  activation/output buffers. This rejects those pre/post differences as
  semantic consumed-input bugs and confirms the stale output-buffer guardrail.
  The same run clarifies the TopK8 helper boundary: the current helper requires
  `hidden_states.size(0) == 1`, so grouped multi-token `shape_or_layout`
  rejections during capture/prefill are expected. Promote a single-token
  decode-focused HF/GGUF source comparator that records active MoE kernel path,
  expert-weight layout/strides, generated kernel identities, and per-call FULL
  replay timing.
- MoE active-layout decode comparator: scratch patch
  `fused-moe-active-layout-debug-20260628T070403Z` inserted one-time detail
  logging at the release TopK8 helper's active branch. The matched `.20` HF
  and GGUF probes
  `moe35b_hf_tp4_decode_active_layout_dot20_20260628T070537Z` and
  `moe35b_gguf_tp4_decode_active_layout_dot20_20260628T070537Z` both activated
  the one-token fastpath on all four TP workers and logged the same visible
  active layout: hidden `(1, 2048)` stride `(2048, 1)`, `w1`
  `(256, 256, 2048)` stride `(557056, 2176, 1)`, and `w2`
  `(256, 2048, 128)` stride `(262144, 128, 1)`. Both also logged `204`
  grouped capture/prefill rejections. This rejects Python-visible expert
  weight shape/stride/layout and grouped TopK8 rejection differences as the
  primary source target. The timing gap remained: HF `FULL num_tokens=1`
  averaged `7.776122` ms and GGUF averaged `11.212760` ms over `1020` calls.
  Promote lower-level inspection of generated kernel runtime behavior, GGUF
  weight materialization, memory movement, or replay state below this matching
  tensor metadata.
- MoE GGUF RCCL-only launch parity probe: scratch bundle
  `qwen35moe-gguf-plus-rcclonly-20260628T072640Z` copied only the release
  `native/rccl` assets into the GGUF patch bundle and set
  `VLLM_NCCL_SO_PATH=/rccl-overlay/install/lib/librccl.so.1`. A prior
  full-native bundle copy was rejected because the GGUF launch mounts
  `<validation-nvme-root>` read-only and the swiglu native copy path writes under
  that tree. RCCL-only parity served correctly and trimmed GGUF `FULL
  num_tokens=1` replay to `10.864359` ms, but it did not approach the HF
  control's `7.776122` ms. Promote RCCL parity as launch hygiene for future
  GGUF probes; reject missing `VLLM_NCCL_SO_PATH` as the primary performance
  root cause.
- MoE GGUF force-unquantized-linear probe: scratch bundle
  `qwen35moe-gguf-force-unquant-linear-20260628T073755Z` patched
  `vllm/model_executor/layers/quantization/gguf.py` so opt-in
  `VLLM_GFX906_GGUF_FORCE_UNQUANT_LINEAR=1` forces `LinearBase` to
  `UnquantizedLinearMethod()` and `VocabParallelEmbedding` to
  `UnquantizedEmbeddingMethod()`. The run
  `moe35b_gguf_tp4_force_unquant_linear_dot20_20260628T073755Z` loaded and
  logged `760` forced-linear selections, so the hook reached normal attention,
  linear-attention, and shared-expert linears. The short request was coherent,
  but `FULL num_tokens=1` replay averaged `10.871309` ms, unchanged from the
  RCCL-only `10.864359` ms band and still behind the HF `7.776122` ms control.
  Reject regular GGUF linear method dispatch as the primary remaining gap.
  Promote lower-level generated code/runtime inspection that survives after
  MoE layout, RCCL parity, and regular linear dispatch are aligned.
- MoE generated-kernel and replay-pointer probe: scratch patch
  `cudagraph-replay-ptr-signature-20260628T080517Z` extended the existing
  CUDA graph value-signature patch to log `data_ptr`, `ptr_mod256`,
  `ptr_mod4096`, and `storage_offset` for graph replay tensors. The matched
  `.20` runs `moe35b_gguf_tp4_ptrsig_dot20_20260628T080540Z` and
  `moe35b_hf_tp4_ptrsig_dot20_20260628T081443Z` used the same MoE TP4
  P2P-on FP16 profile. `rocprofv3 --attach` was rejected as a probe because it
  attached to `Worker_TP0` but emitted no trace or summary output even with
  explicit runtime/kernel trace, CSV/JSON, summary, and zero minimum-output
  settings. The TorchInductor generated-kernel surface paired exactly: HF and
  GGUF each had `76` `.kernel_perf` records and `216` generated Python files.
  The largest GGUF-slower generated-kernel timing deltas had byte-identical
  generated Python source in HF and GGUF, so source generation mismatch is not
  the current root. The top-level FULL replay pointer diagnostic also matched:
  `input_ids` used contiguous shape `(1,)` with `ptr_mod256=0` and
  `ptr_mod4096=3584`; `positions` used shape `(3, 1)`, stride `(2049, 1)`,
  `ptr_mod256=0`, and `ptr_mod4096=512`. The gap remained with HF averaging
  `8.003061` ms and GGUF averaging `10.827265` ms for FULL `num_tokens=1`.
  Promote internal captured-graph/model-state inspection below the
  `CUDAGraphWrapper` argument boundary: activation/KV/recurrent buffers,
  graph state, weight materialization/residency, or runtime scheduling inside
  the captured graph. Reject attach-mode `rocprofv3`, generated-kernel source
  mismatch, and top-level replay pointer alignment as primary targets.
- MoE active fastpath value-signature comparator: scratch patch
  `fused-moe-active-value-debug-20260628T083222Z` extended the release TopK8
  one-token fastpath to log active expert IDs, routing weights, hidden-state
  samples, pointer alignment, and small selected expert-weight signatures under
  `VLLM_GFX906_MOE_ACTIVE_VALUE_FILE`. The matched `.20` runs were
  `moe35b_gguf_tp4_active_value_dot20_20260628T083258Z` and
  `moe35b_hf_tp4_active_value_dot20_20260628T083258Z`. Both emitted `320`
  active records, `80` per TP worker. The first active call matched between HF
  and GGUF at the visible boundary: identical top-k IDs
  `101,196,108,249,216,197,135,242`, identical routing weights, identical
  hidden signature, matching active tensor shape/stride, and matching selected
  expert-weight signatures. Calls `1` through roughly `8` stayed close; clear
  routing divergence appeared around calls `9` and `10`. GGUF remained
  coherent and replayed at `10.893246` ms average over `508` FULL
  `num_tokens=1` calls. The HF completion request under this diagnostic hung
  after readiness with a shared-memory wait, so the CPU-copy logger is rejected
  as a safe timing probe for HF. Promote upstream captured-state/value drift as
  the next source target; reject first-call active MoE routing/layout/selected
  expert materialization as primary roots. Future source probes should avoid
  CPU copies inside CUDA graph capture and should inspect upstream consumed
  inputs or device-side summaries without perturbing replay timing.
- MoE HF eager active-value control rejection: the follow-up
  `moe35b_hf_tp4_active_value_eager_dot20_20260628T085833Z` disabled CUDA
  graph capture with `--enforce-eager` and used the same HF TP4 release
  profile, FP16, P2P-on state, and RCCL path. The server reached readiness, but
  the bounded request hung with `0%` GPU use, empty completion output, no
  active-value signature file, and repeated shared-memory wait messages. The
  matching GGUF eager request emitted `512` active-value records, but the HF
  eager path did not produce a comparable request trace. Reject HF eager
  CPU-copy active-value logging as a source method. Next probes should move the
  comparator to device-side summaries, pinned-buffer snapshots, or lower-impact
  upstream router-input metadata rather than copying active tensors to CPU in
  HF request execution.
- MoE launch-wrapped `rocprofv3` shutdown rejection: the GGUF TP4 run
  `moe35b_gguf_tp4_rocprof_kernel_short_dot20_20260628T092554Z` wrapped the
  release image server with `/opt/rocm/bin/rocprofv3`, supplied the missing
  `libdw.so.1` as a read-only mount, and wrote profiler output to a mounted
  directory. The server reached readiness, activated the one-token MoE TopK8
  fastpath after graph capture, and served a coherent 64-token request. Docker
  stop then delivered signal 15; `rocprofv3` waited for child processes, the
  container exited `137`, and no profiler CSV/JSON/summary artifacts were
  emitted. Promote the `libdw.so.1` mount and launch-wrapper readiness as
  profiler hygiene. Reject Docker-stop shutdown and this run as kernel
  profiling evidence. A future profiler route needs a self-terminating
  in-container wrapper or reduced/offline worker path that exits cleanly enough
  for `rocprofv3` to flush.
- MoE self-terminating `rocprofv3` wrapper rejection: the follow-up run
  `moe35b_gguf_tp4_rocprof_selfterm_dot20_20260628T095756Z` used an
  in-container wrapper to start vLLM, wait for `/v1/models`, send one bounded
  short completion request, and signal the server child before exit. The
  request was coherent, the wrapper reached `request_complete` and
  `child_exited`, the container exited status `0`, and TP4 VRAM cleared. No
  profiler CSV/JSON/summary/kernel artifacts were emitted. This rejects the
  idea that Docker-stop signal 15 was the only blocker and rejects broad
  launch-wrapped server profiling as a repeat path. Wrapped timing was not
  usable either: FULL replay averaged `66.231362` ms, compared with normal GGUF
  force-unquant `10.871309` ms and HF control `7.887792` ms. The run also
  confirmed the shape boundary for the Qwen TopK8 c1 MoE fastpath:
  capture/prefill and multi-token shapes reject it, while FULL one-token decode
  activates it on all four TP workers. Promote a reduced worker/replay
  reproducer or lower-overhead C/C++ or device-side summaries for the FULL
  `num_tokens=1` path.
- MoE packed-weight contract audit: run
  `moe35b_gguf_packed_weight_contract_20260628T104313Z` used the correct
  Qwen3.5-MoE linear-attention geometry (`16` key heads, `32` value heads,
  `128`-wide linear heads) to reconstruct HF and GGUF layer-0 TP4 packed views.
  It compared transformed GDN tensors, TP4 `qkvzba` / `out_proj` / `conv1d`
  / `A_log` / `dt_bias` shards, and MoE expert/shared-expert tensors. Result:
  `36` checks, `0` review rows, and the swapped gate/up expert negative
  control rejected. Promote static packed-weight parity as current evidence.
  Reject another raw tensor permutation or layer-0 packed-shard audit unless a
  later-layer or runtime-only trace contradicts this result.
- MoE graph-surface compare: run
  `moe35b_gguf_graph_surface_compare_20260628T104424Z` compared the active GGUF
  graph against the HF replay-control graph. The checked rank-0 counts matched:
  `81` submodules, `31` folded `qkvzba` weight shapes, `30` gemms to `3088`,
  `40` gemms to `2048`, `60` GDN core calls, `162` all-reduces, `80`
  fused-MoE calls, and `120` shared-MoE calls. Promote the next source target
  below graph shape: runtime materialization, generated-kernel/cache factors,
  captured recurrent/KV state, or a reduced FULL `num_tokens=1` replay worker.
- MoE batched TopK8 tile variants: scratch overlays
  `fused-moe-batched-topk8-20260628T115300Z`,
  `fused-moe-batched-topk8-bi16-20260628T121200Z`, and
  `fused-moe-batched-topk8-bi32-20260628T122800Z` changed the one-token TopK8
  fastpath tile geometry for the GGUF TP4 path. Results stayed coherent and
  strict-valid but low-band: BI8 `83.785` strict / `74.844` c1_10000, BI16
  `84.972` strict / `76.483` c1_10000, and BI32 `85.607` strict / `76.271`
  c1_10000. Reject this tile family as a release reproduction fix. Promote it
  as evidence that simple TopK8 tile geometry is not the main native-parity
  blocker.
- Release native SwiGLU boundary: the release image contains
  `/opt/gfx906_release_runtime/python_overlays/qwen2_moe_interleaved_swiglu_20260608.py`
  and
  `/opt/gfx906_release_runtime/native/swiglu/gfx906_swiglu_gemv_ext_20260607.so`.
  The Python overlay is guarded for the Qwen MoE shared-expert path with TP8
  and shape-specific checks: `quant_config is None`, `hidden_size == 5120`,
  `intermediate_size == 17408`, `tp_size == 8`, and native call shape
  `(4352, 5120)`. Reject forcing this TP8 shared-expert overlay into TP4 GGUF
  routed-expert work. Keep it as release source context until a trace points
  directly at shared-expert execution.
- Exact HF release TopK8 fastpath on GGUF TP4: scratch overlay
  `fused-moe-hf-release-exact-20260628T1330Z/fused_moe.py`, checksum
  `66f63f74406c2a805e78eeb28e9dac76c3f98bc4ce9046cd3be5d604224a0a0e`,
  copied the HF release TP4 fastpath source into the current GGUF
  force-unquantized TP4 route on `.20` and `.30`. Logs showed expected
  larger-shape rejection and one-token decode activation for token counts
  `4`, `2`, and `1`. The normal ladders stayed low-band: `.20` strict
  `81.036`, c1_2000 `84.783`, c1_10000 `74.793`; `.30` strict `83.302`,
  c1_2000 `84.751`, c1_10000 `74.762`. Reject exact fastpath source mismatch
  and host-lane choice as the remaining TP4 GGUF gap. Keep the source target
  below high-level fastpath selection: replay body, captured state, runtime
  materialization, or expert prepare/finalize memory movement.
- MoE TopK8 active pointer alignment diagnostic: scratch overlay
  `fused-moe-active-ptr-debug-20260628T1340Z/fused_moe.py`, checksum
  `f5e20bd6e450371dbf36db9f6a2908da873aebf88c8cb367a2a6345568b19270`,
  logged storage offsets and pointer alignment for the one-token active
  fastpath. HF and GGUF both returned coherent short-prompt text and both had
  zero storage offsets, 256-byte alignment on hidden/top-k tensors, and
  4K-aligned packed expert `w1` / `w2` tensors. Diagnostic replay still favored
  HF (`7.518159` ms average) over GGUF (`9.979224` ms average). Reject visible
  active tensor pointer alignment as the primary MoE GGUF TP4 performance gap.
  Continue below the visible tensor surface into replay body, captured state,
  runtime materialization, or expert prepare/finalize memory movement.
- HF TorchInductor cache preseed diagnostic: run
  `moe35b_gguf_tp4_hf_inductor_cache_dot20_20260628T1405Z` copied the HF
  `torchinductor_root` into the GGUF route before launch. HF and GGUF had
  identical generated `*.py` source files in the pointer diagnostic; the
  remaining differences were autotune metadata. The preseeded run retained
  normalized HF `*.best_config` contents exactly and served coherent text, but
  only improved the second request to `9.190717` ms average, still slower than
  HF at `7.518159` ms. Reject best-config selection and generated Python source
  mismatch as the sole bottleneck. Keep investigating captured inputs/state and
  runtime materialization costs around the one-token FULL replay.
- CUDA graph replay input-boundary diagnostic: patch
  `cudagraph-input-debug-20260628T1420Z/vllm/compilation/cuda_graph.py`,
  checksum
  `5a55161238a3b7579d94490d7626453b82fa1f6617357119665eb5dfae58ad32`, logged
  FULL replay kwargs immediately before `entry.cudagraph.replay()`. Runs
  `moe35b_hf_tp4_graph_input_dot20_20260628T1420Z` and
  `moe35b_gguf_tp4_graph_input_dot20_20260628T1420Z` matched for the first
  eight one-token replay calls: `input_ids` `90700`, `8340`, `25`, `271`, `16`,
  `13`, `220`, `2972`; positions `20` through `27`; `input_ids` shape `(1,)`
  `torch.int32` stride `(1,)`; and `positions` shape `(3, 1)` `torch.int64`
  stride `(2049, 1)`. The response message hashes and usage counts also
  matched between HF and GGUF for this request. Replay timing remained split:
  HF `8.031084` ms average versus GGUF `10.843224` ms over `252` FULL rows
  each. Reject top-level replay input divergence and response divergence as the
  root cause. Next source work should move below `cuda_graph.py` replay kwargs
  into captured lower-level state, expert prepare/finalize materialization,
  recurrent/KV state plumbing, or a reduced C/C++-friendly FULL replay
  reproducer.
- Runner metadata trace rejection: HF patch checksum
  `853f61cdc813583a520ea64d28858e497e0dfec64de6b28fdfa41f47938d0022` and
  prepared GGUF patch checksum
  `630018c8454a77e69f0af20dea71bbcc1e76170f06a85fb74524f21440405ce0` added a
  guarded metadata logger to `vllm/v1/worker/gpu_model_runner.py`. The HF
  control initially failed due a missing `os` import in the release-image base,
  then reached readiness after the import fix. The logger recorded graph-capture
  dummy rows (`input_ids=[0]`, `positions=[0, 0, 0]`) and the real request
  stalled with no GPU activity. Reject this in-process runner metadata patch as
  a valid comparator and do not run the GGUF side. Future low-level work should
  avoid GPU-to-CPU reads at this model-runner point and should prefer a
  request-replay guard or a reduced explicit replay reproducer.
- Active-pointer `.kernel_perf` artifact recheck: the HF cache
  `moe35b_hf_tp4_active_ptr_dot20_20260628T1340Z/runtime/tmp/torchinductor_root`
  and GGUF cache
  `moe35b_gguf_tp4_active_ptr_dot20_20260628T1340Z/torchinductor_root` each had
  `76` `.kernel_perf` files with all paths joined. Summed metadata values were
  HF `2.060957000` and GGUF `1.980158011`, so the aggregate metadata direction
  slightly favored GGUF while live replay favored HF. Reject `.kernel_perf`
  aggregate timing as the current root cause. Keep future source work focused
  on request-window HIP kernel mix, captured state, runtime scheduling, or a
  reduced replay reproducer.
- Replay value-signature decode audit: scratch audit
  `moe35b_valuesig_replay_decode_audit_dot20_20260628T1515Z` generated
  `summary.md`, `piecewise_arg_shape_map.txt`, and `sha256.txt` from existing
  HF/GGUF value-signature artifacts. It found that the FULL one-token replay
  logging surface had `124` rows with `argc=0`; the tensor-value diffs were all
  PIECEWISE rows (`num_tokens=16` for HF/GGUF and `num_tokens=24` for GGUF
  pre/post). Keep the shape map as source context for generated PIECEWISE graph
  arguments, but reject it as direct evidence for the FULL decode performance
  gap. Do not repeat broad value-signature scans unless the probe captures
  tensor state inside a reduced FULL replay worker.
- HIP graph `LD_PRELOAD` trace rejection: C/C++ tool
  `hip_graph_trace_20260628T1525Z` intercepted HIP graph APIs and wrote
  process-local summaries. A standalone HIP graph smoke test succeeded, but the
  full GGUF TP4 vLLM run
  `moe35b_gguf_tp4_hipgraph_dot_dot20_20260628T1540Z` failed during KV-cache
  initialization before readiness. The trace directory contained summary files
  only, no graph node files, no DOT files, and zero graph instantiate / launch
  counters for worker processes. Reject process-wide graph interposition for
  full-server vLLM. Prefer a reduced FULL replay worker or a request-window
  trace that avoids preloading into every process.
- MoE TP4 release-overlay combo promotion: scratch bundle
  `qwen35moe-gguf-release-overlay-combo-20260628T1615Z` combined the active
  release overlay bundle, native runtime setup, GGUF loader/model repairs, and
  the exact HF release TP4 fastpath source. The first `.20` launch with
  `<validation-nvme-root>:ro` failed because the release entrypoint copies native
  SwiGLU runtime material into the NVMe work area; relaunching with the
  release-style read-write mount reached health. Under `moe35b_tp4_fullbar_p2pon`,
  FP16 GGUF, P2P-on, `MAX_MODEL_LEN=131072`, default broad graph capture, and
  normal warmups -> strict -> c1 tiers, the combo produced `.20` strict
  `118.754`, c1_2000 `119.896`, c1_10000 `112.747`, and `.30` strict
  `119.917`, c1_2000 `120.781`, c1_10000 `113.605` backend TPS. Promote the
  full release-overlay composition as the current MoE TP4 GGUF source path.
  Reject single-file fastpath swaps and host-lane choice as the old low-band
  explanation. Next source work should shrink the scratch combo into a
  reproducible minimal bundle before any public release/docs update.
- MoE TP4 reduced package-path promotion: vNext run
  `moe35b_pkg_dot30_20260628T180700Z` used the pinned minimal bundle under
  `qwen36-gfx906/overlays/gguf-moe/minimal-bundle/`, manifest hash
  `794cb8760003cfa7dafeb0f06ee01823e5e8eebb3f9d398d8bc8bd7697143f0b`, and
  the generated launch artifacts from `gguf-moe35b-tp4`. The first package
  launch failed before serving because the installed Transformers GGUF helper
  rejected the Qwen3.6 MoE GGUF architecture string. The required source
  packaging fix is to copy
  `qwen36-python/transformers/modeling_gguf_pytorch_utils.py` from the bundle
  directly into the installed Transformers paths, not only expose it through
  `PYTHONPATH`. With that fix, `.30` completed the normal package-path ladder:
  strict `119.508`, c1_2000 `120.758`, c1_10000 `113.271` backend TPS. Promote
  this as the reduced MoE bundle's current source inventory. Reject carrying
  the large scratch combo as the release candidate now that the reduced bundle
  reproduces the high-band result.
- HF Dense vNext baseline protection rejection: the generated `hf-dense27b-tp8`
  profile was aligned with the historical text-only comparator by using
  `Qwen3_5ForCausalLM`, `--language-model-only`, the CausalLM HF override, TP8,
  P2P-on, FP16, `MAX_MODEL_LEN=131072`, balanced `RCCL_TREES`, dense
  RowParallel/Persistent-AR envs, and an isolated HF release bundle under
  `qwen36-gfx906/overlays/hf/minimal-bundle/` with manifest hash
  `9fb16c0edfd57d908f2bff6eb51063b6e8cf7d2c27de252bb4f691c67d2f5a84`.
  Artifact validation passed, runtime arg-schema validation passed, GGUF env
  leakage was absent, and persistent all-reduce route replacement appeared in
  startup logs. Performance still rejected: `.30` no-bundle reached strict
  `59.988`, c1_2000 `62.853`, c1_10000 `54.294`; `.30` HF-bundle reached
  strict `63.232`, c1_2000 `65.051`, c1_10000 `55.985`; `.20` HF-bundle
  reached strict `62.962`, c1_2000 `64.991`, c1_10000 `55.945`. These remain
  below the historical `.20` text-only HF comparator (`70.353`, `70.978`,
  `66.428`). Promote the guardrail and isolated HF-bundle structure; reject the
  current HF generated package path as release-ready baseline protection until
  the old comparator/runtime delta is explained.
- HF Dense vNext baseline protection recovery: the old `.20` text-only
  comparator used the broader clean release patch-bundle composition. The
  first-pass reduced HF bundle omitted model, platform, attention, utility, and
  MoE fastpath overlay files from that composition. Expanding
  `qwen36-gfx906/overlays/hf/minimal-bundle/` to mirror the old clean release
  bundle produced manifest hash
  `d9e1168f67731ab66ee53671f89dfb31fd3c74c8f61e2f0d0b7c4f0704c298be`.
  With the generated vNext launcher on `.20`, the expanded bundle reproduced
  the HF Dense band using both shared release cache (`70.346`, `70.961`,
  `66.428`) and fresh per-run cache (`70.168`, `71.316`, `66.824`) for strict,
  c1_2000, and c1_10000 backend TPS respectively. Promote the expanded clean
  HF release bundle as the source inventory for non-GGUF baseline protection.
  Reject the reduced HF bundle and reject cache reuse as the necessary decode
  speed lever.
- HF MoE contract pinning: `hf-moe35b-tp4` and `hf-moe35b-tp8` now require the
  same expanded clean HF release bundle with manifest hash
  `d9e1168f67731ab66ee53671f89dfb31fd3c74c8f61e2f0d0b7c4f0704c298be`. Local
  synthetic preflight generated and verified launch artifacts for HF Dense, HF
  MoE TP4, HF MoE TP8, GGUF Dense, and GGUF MoE. This promotes profile
  contract symmetry and developer inspectability; it does not replace runtime
  HF MoE reproduction reruns.
- vNext serviceability gate: `qwen36-gfx906/verify_vnext_serviceability.sh`
  now validates all five profiles without launching Docker or touching live
  GPUs. It verifies image tag/digest pins, P2P-on profile state,
  `MAX_MODEL_LEN=131072`, FP16, HF/GGUF env separation, patch-bundle manifest
  hashes, and absence of private host/path references in developer-facing
  vNext files. Promote this as a required gate for developer-serviceable image
  paths.
- HF MoE TP4 generated-path runtime rejection: `.30` reruns showed that the
  vNext HF MoE TP4 package path is functionally valid but still below the
  published release-performance band. Without the tuned MoE config in the HF
  bundle it produced strict `95.824`, c1_2000 `96.158`, c1_10000 `91.524`.
  With the tuned MoE config added and the HF bundle pinned to
  `d9e1168f67731ab66ee53671f89dfb31fd3c74c8f61e2f0d0b7c4f0704c298be`, it
  produced strict `96.159`, c1_2000 `97.031`, c1_10000 `92.284`, with the
  strict run passing `finish_reason=stop` and `qwen_gate_valid=true`. Reject
  this generated path as release-performance-equivalent until the exact
  historical HF MoE TP4 performance settings are recovered.
- HF MoE TP4 deploy-shaped wrapper rejection: updating the vNext wrapper to
  preserve deploy-style generic runtime defaults (`VLLM_TARGET_DEVICE=rocm`,
  `VLLM_VERSION_OVERRIDE=0.0.0+gfx906`, `PYTORCH_ROCM_ARCH=gfx906`,
  `GPU_ARCHS=gfx906`, `OMP_NUM_THREADS=4`,
  `TORCH_BLAS_PREFER_HIPBLASLT=0`, and privileged container mode) still
  produced only strict `94.860`, c1_2000 `95.613`, and c1_10000 `91.036`
  backend TPS on `.30`, with strict validity passing. Promote the runtime
  defaults as serviceability guardrails, but reject them as the missing
  high-band source/runtime factor.
- Public image serviceability check: the public runtime tag
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`
  resolves to pinned manifest digest
  `sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e`.
  A clean container check confirmed the image contains the native runtime
  paths referenced by the profile contracts:
  `/opt/gfx906/libgfx906_persistent_tree_ll_ar_default_20260613.so`,
  `<validation-nvme-root>/kernel_labs/gfx906_swiglu_gemv_ext_native_runtime_20260608/gfx906_swiglu_gemv_ext_20260607.so`,
  and `/rccl-overlay/install/lib/librccl.so.1`. Promote the optional
  `CHECK_RUNTIME_IMAGE=1 CHECK_RUNTIME_PATHS=1 ./verify_vnext_serviceability.sh`
  gate for developer handoff. Reject hidden local images or native-library
  paths as release-profile inputs.
- Contract matrix gate: `qwen36-gfx906/verify_vnext_contract_matrix.sh` builds
  the C model-format probe, creates synthetic GGUF/safetensors model
  signatures, generates and verifies launch artifacts for all five profiles,
  and checks fail-closed behavior for format mismatch, Dense/MoE family
  mismatch, GGUF env leakage into HF, and patch-bundle hash mismatch. It also
  runs synthetic deploy/profile parity for HF MoE TP4 and TP8. Promote this as
  the source-package guardrail for launcher/profile edits. Reject changes that
  only pass profile metadata checks without proving generated artifacts,
  deploy parity, and mismatch failures.
- Deploy-profile parity gate: MoE vNext profiles now use the deploy-style
  tuned-config path `/opt/vllm_tuned_moe_configs`, and
  `qwen36-gfx906/vnext_repro_launcher.sh` copies
  `vllm_tuned_moe_configs` from the mounted patch bundle into that path before
  vLLM starts. New helper
  `qwen36-gfx906/verify_vnext_deploy_profile_parity.sh` compares captured
  `.deploy.runtime.env` files against vNext profile contracts and generated
  argv. Promote this as the source-level parity gate for recovering the exact
  HF MoE high-band lane from `deploy.sh` artifacts. Reject vNext-specific
  tuned-config paths for MoE profiles unless a later runtime result proves they
  are intentionally different and equivalent.
- Deploy-compat runtime patch surface: `qwen36-gfx906/vnext_repro_launcher.sh`
  now generates a `deploy_compat_patches.py` artifact and runs it before vLLM
  starts. The patch surface mirrors deploy-time runtime compatibility handling
  for missing custom-op APIs, ROCm/GFX906 capability, batch-invariant
  compatibility, fused-MoE utilities, activation/MoE activation/RMSNorm/rotary
  fallbacks, and missing-op reporting. Promote this as source inventory and
  developer-serviceability protection. Reject it as the HF MoE TP4 high-band
  performance fix: the `.30` deploy-compat runtime rerun completed the normal
  ladder at strict `92.731`, c1_2000 `93.272`, and c1_10000 `88.832` backend
  TPS. The strict run was valid, so the remaining gap is the exact high-band
  deploy contract, not basic runtime correctness.
- Profile-gated patch target activation: every vNext profile now declares
  `PATCH_TARGET_GROUPS`, and the generated container entrypoint uses those
  groups before copying files from a mounted bundle. Dense native sidecar,
  dense SWiGLU, MoE tuned-config/fused-MoE, MoE TP8-only, and GGUF loader
  targets are now group-labeled. This keeps the broad HF bundle serviceable
  while preventing dense-only files from activating in HF MoE launches, and
  prevents GGUF-only loader files from activating in HF launches. Promote this
  as part of the source inventory for overlay isolation. HF MoE TP4 reruns
  with `PATCH_TARGET_GROUPS="common moe hf"` recovered high-band performance
  on both primary lanes: `.20` strict `115.105`, c1_2000 `115.933`, c1_10000
  `109.095`; `.30` strict `114.409`, c1_2000 `115.685`, c1_10000 `108.918`.
  This confirms that group-gated activation is the source-contract fix for the
  previous low-band HF MoE TP4 vNext runs.

## Recent Source Inventory Addendum - 2026-06-29

- Container-path preflight mapping: `qwen36-gfx906/vnext_repro_launcher.sh`
  now maps `<container-hf-cache>/...` through `HOST_HF_CACHE` and
  `/opt/vllm_patch_bundle/...` through `PATCH_BUNDLE_PATH` during host-side
  preflight. Promote this as a launcher serviceability fix. Reject replacing
  runtime container paths in profile contracts with host-specific paths.
- Clean Dense GGUF contract package: `gguf-dense27b-tp8` with overlay
  `gguf_dense`, TP8, FP16, P2P-on, `MAX_MODEL_LEN=131072`, and bundle manifest
  `ce2c6f2d974de34c7abf4c25a67985b0eccc452a97f1a887ca97a8de1687f8d2`
  completed on `.20` and `.30`. Promote it as the current Dense GGUF package
  evidence: `.20` strict `69.914`, c1_2000 `70.759`, c1_10000 `66.316`; `.30`
  strict `69.851`, c1_2000 `70.959`, c1_10000 `66.437`. Reject older
  scratch-only runs as the primary package evidence.
- Clean MoE GGUF contract package: `gguf-moe35b-tp4` with overlay `gguf_moe`,
  TP4, FP16, P2P-on, `MAX_MODEL_LEN=131072`, and bundle manifest
  `794cb8760003cfa7dafeb0f06ee01823e5e8eebb3f9d398d8bc8bd7697143f0b`
  completed on `.20` and `.30`. Promote it as current MoE GGUF package
  evidence: `.20` strict `119.333`, c1_2000 `120.565`, c1_10000 `113.366`;
  `.30` strict `119.521`, c1_2000 `120.460`, c1_10000 `113.258`.
- Clean HF MoE TP8 contract package: `hf-moe35b-tp8` with overlay
  `hf_release`, TP8, FP16, P2P-on, `MAX_MODEL_LEN=131072`, patch target groups
  `common moe hf moe_tp8`, and bundle manifest
  `d9e1168f67731ab66ee53671f89dfb31fd3c74c8f61e2f0d0b7c4f0704c298be`
  completed on `.20` and `.30`. Promote it as current non-GGUF TP8 baseline
  evidence: `.20` strict `115.041`, c1_2000 `115.549`, c1_10000 `108.805`;
  `.30` strict `114.698`, c1_2000 `115.532`, c1_10000 `108.673`.
- Public-staged HF source path: a `.30` clean audit downloaded
  `Qwen/Qwen3.6-27B` and `Qwen/Qwen3.6-35B-A3B` with `hf download`, generated
  fresh vNext launch artifacts, validated runtime argv against
  `vllm serve --help=all`, and ran the POSIX benchmark ladder. Promote public
  Hugging Face downloads as the HF reproduction input path. Results: HF Dense
  TP8 strict `69.682`, c1_2000 `71.346`, c1_10000 `66.729`; HF MoE TP4 strict
  `111.964`, c1_2000 `115.937`, c1_10000 `108.848`; HF MoE TP8 strict
  `115.488`, c1_2000 `116.258`, c1_10000 `109.242`. Reject retained
  validation-host snapshots as public release inputs.
- MoE top-k8 fastpath disposition: clean contract logs show larger graph
  shapes rejected by shape/layout, while token-sized decode shapes activate.
  Promote this as a source-path nuance for future kernel work. Reject broad
  "all graph shapes use top-k8" claims unless a later profile proves it.
- Developer-serviceable image requirement: generated launch artifacts record
  image tag/digest, model format, profile, overlay, patch target groups, P2P
  state, `MAX_MODEL_LEN`, final command, and benchmark environment. Promote the
  generated artifact plus verifier path as the expected handoff surface for
  other developers. Reject hidden lab-only paths, unpinned bundles, and
  production `EXTRA_VLLM_ARGS`.
