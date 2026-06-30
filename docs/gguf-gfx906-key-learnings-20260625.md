# GGUF / GFX906 Key Learnings - 2026-06-25

This document starts the GGUF-specific source record for Qwen3.6 on GFX906.
It is separate from the published FP16 ROCm7.2 Dense/MoE release notes and
should not be read as a release claim.

The purpose is to preserve what was tried, what failed, what promoted as
diagnostic evidence, and what remains active source work.

## Scope

- Target stack: vLLM GGUF loading on ROCm / GFX906.
- Target comparison models: Qwen3.6 27B Dense and Qwen3.6 35B-A3B MoE.
- Current concrete test artifacts: Qwen3.6 27B Dense GGUF, with Q4_0 and
  FP16/half GGUF files available for source investigation; and Qwen3.6
  35B-A3B MoE F16 GGUF for architecture-loader investigation.
- Current source lanes: contract-generated Dense GGUF TP8 and MoE GGUF TP4
  reproduction packages after the `lm_head` logits-path fix, MoE semantic
  correctness repair, overlay isolation, patch-target group gating, and
  host/container preflight path mapping.
- Not yet promoted: public GGUF publication claims. The current GGUF evidence
  now includes clean package-path runs on `.20` and `.30`, but it remains
  experiment-stream evidence until a separate release explicitly publishes the
  GGUF path.

The published FP16 release path remains the canonical reproduction path for
v0.2.0/v0.2.1 until GGUF correctness and performance pass the same promotion
standard.

## Current GGUF Status

The dense GGUF path has now produced a promoted match/beat result for the
published dense benchmark numbers. Early work found tensor-parallel correctness
drift after the model entered the Qwen3.5 / Qwen3.6 GatedDeltaNet path. Later
FP16/half dense work repaired semantic correctness with explicit
tokenizer/config handling, GGUF tensor-layout fixes, and the release overlay.

Current status:

- Historical Q4_0 paths remain rejected: repeat-interleave repaired simple
  TP1/TP2/TP4/TP8 first-token sanity, but the full-context strict gate failed.
- FP16/half dense GGUF now produces coherent output, passes the strict gate, and
  beats the published dense release values on the CausalLM/Qwen3.5 source path
  after the `lm_head` logits-path fix. The final `.20` lane-validation run
  `dense27b_gguf_tp8_lmheadunquant_dot20_lanevalidate_20260628T110342Z`
  produced:
  - strict: best observed `70.505` TPS versus published `69.514`;
  - `c1_2000`: best observed `71.589` TPS versus published `70.347`;
  - `c1_10000`: best observed `66.967` TPS versus published `66.069`.
- The clean vNext Dense GGUF TP8 package path has now run successfully on both
  `.20` and `.30` from the contract launcher, pinned minimal bundle, generated
  Docker wrapper, generated benchmark environment, and normal `8` warmups ->
  `c1_128` strict -> `c1_2000` -> `c1_10000` ladder. Run
  `20260629T003817Z-gguf-dense27b-tp8` produced `.20` strict `69.914`,
  `c1_2000` `70.759`, and `c1_10000` `66.316`; and `.30` strict `69.851`,
  `c1_2000` `70.959`, and `c1_10000` `66.437`. Strict passed on both lanes,
  and both c1_10000 values beat the published Dense release value `66.069`.
  Promote the Dense GGUF package path as clean contract evidence, not yet as a
  public release claim.
- The clean public-staged HF path also reproduces from public upstream model
  downloads rather than retained validation-host snapshots. A `.30` audit using
  `hf download`, generated vNext launch artifacts, runtime vLLM argv-schema
  validation, and the normal POSIX benchmark ladder produced HF Dense TP8
  strict `69.682`, `c1_2000` `71.346`, and `c1_10000` `66.729`; HF MoE TP4
  strict `111.964`, `c1_2000` `115.937`, and `c1_10000` `108.848`; and HF MoE
  TP8 strict `115.488`, `c1_2000` `116.258`, and `c1_10000` `109.242`.
  Promote public `hf download` staging as the portable HF input path.
- Official release image plus the same GGUF/release overlay performs in the same
  band as the experimental image, so image-base drift is rejected.
- Restoring the release attention subclass does not improve decode throughput.
- ConditionalGeneration architecture alignment does not improve decode and
  worsens prefill.
- Removing disabled forward trace calls does not improve decode throughput.
- A clean HF-weight dense TP8 control on the same host lane, official image,
  release overlay, P2P-on path, and benchmark harness reproduced the published
  band: strict `69.652`, `c1_2000` `70.202`, and `c1_10000` `65.952` backend
  TPS.
- The coherent dense GGUF release-attention runs log a Transformers FLA /
  causal-conv1d fallback warning, but later source inspection localized that
  warning to GGUF loader dummy-model construction. The active gap is still
  source-path parity, not host state or image base, but the warning itself is not
  proof of request-time torch fallback.
- The HF-weight control resolves as `Qwen3_5ForConditionalGeneration`, while
  the coherent GGUF path resolves as `Qwen3_5ForCausalLM`; direct GGUF
  ConditionalGeneration override was already rejected, so the next source pass
  should compare execution paths instead of repeating that launch-shape change.
- Fusing GGUF GatedDeltaNet beta/alpha into the HF release qkv/z/b/a projection
  shape is a real improvement after a patient rerun. It improved the dense
  ladder to strict `63.201`, `c1_2000` `63.836`, and `c1_10000` `60.169`
  backend TPS, but still does not match the FP16/HF release band.
- Enabling the GFX906 native interleaved SwiGLU extension for F16 GGUF works
  and keeps strict validity, but the dense ladder remains below the HF-weight
  control: strict `61.053`, `c1_2000` `61.780`, and `c1_10000` `58.266`
  backend TPS. This rejects MLP SwiGLU enablement as the missing standalone
  performance lever.
- Combining fused-GDN with native GGUF SwiGLU was the previous dense GGUF
  baseline: strict `63.679`, `c1_2000` `64.353`, and `c1_10000` `60.617`
  backend TPS. It was superseded by the dense `lm_head` unquantized logits path,
  which reached strict `70.350`, `c1_2000` `71.396`, and `c1_10000` `66.842`
  backend TPS. A later `.20` lane-validation run from the same promoted source
  path superseded that with strict `70.505`, `c1_2000` `71.589`, and
  `c1_10000` `66.967` backend TPS.
- Rechecking `Qwen3_5ForConditionalGeneration` with the current fused-GDN plus
  native GGUF SwiGLU bundle remains strict-valid but is slower: strict
  `63.003`, `c1_2000` `63.656`, and `c1_10000` `60.109` backend TPS. This
  rejects architecture override as the missing dense GGUF lever after the latest
  source improvements.
- HF-control versus best-GGUF log comparison shows equivalent graph-capture
  mode, capture sizes, and capture duration. The logs still differ in wrapper
  type, page geometry, fallback source, and GGUF materialization/execution path
  details, but graph setup alone is not a credible explanation for the current
  dense TPS gap.
- Python timing hooks inside the compiled Qwen3.5 GatedDeltaNet forward are not
  viable. Dynamo rejects both `torch.cuda.synchronize()` and
  `time.perf_counter()` during startup profiling, before the server becomes
  ready.
- `rocprofv3 --attach` request-window probes also failed as currently shaped.
  They attached successfully to the TP0 worker during fixed-token `c1_2000`
  requests and confirmed the best dense GGUF bundle remains in the expected
  band (`64.386`, `64.362`, and `64.217` backend TPS), but emitted no kernel,
  HIP, RCCL, CSV, JSON, or summary artifacts. A separate tiny Torch sanity run
  proved `rocprofv3` can emit artifacts in the release image if the missing
  `libdw.so.1` dependency is supplied read-only, but relaunching the GGUF server
  with that mount still did not make PID attach write files. Do not repeat that
  PID-attach shape unchanged.
- A tiny multiprocessing control refined the profiler boundary. Launching the
  parent Python process under `rocprofv3` captured kernel/HIP artifacts for a
  `spawn` child doing GPU work, but not for a `fork` child. Because vLLM uses
  `spawn`, launch-time wrapping can follow child workers in principle.
- Launch-time wrapping still failed for the full current GGUF vLLM server. A
  profiler-wrapped `GGUF-070` server reached tensor conversion and graph
  compile, then hit GPU memory access faults during persistent all-reduce
  startup before readiness. No request or profile artifacts were produced. This
  rejects the full-server `rocprofv3` wrapper shape as a safe comparable
  profiling route.
- Disabling Persistent-AR to make the launch-time profiler shape reach health
  is not a usable comparison. The no-AR diagnostic eventually served a short
  capped request, but only at about `12.37` vLLM decode TPS, failed the strict
  gate because the request was capped, and still produced no useful profiler
  artifacts on shutdown. Treat it as a rejected profiler-shape experiment, not
  as model or release-path evidence.
- ROCTX marker tracing works in a tiny compiled Torch control when
  `rocprofv3 --marker-trace --kernel-trace` wraps the process and the missing
  `libdw.so.1` is supplied read-only. The marker API CSV/JSON files include the
  expected marker ranges.
- Raw `torch.cuda.nvtx.range_push/pop` calls cannot be inserted directly into
  the compiled Qwen forward. vLLM's AOT/fullgraph path rejects `range_push`
  because it returns a non-Tensor.
- A Tensor-returning Python custom marker op is also not ready for the model
  path. It compiles and emits markers in a tiny fullgraph control, but the
  model bundle fails runtime shape checks because the ad hoc op aliases its
  input. The alias-annotated schema variant failed the tiny Inductor control.
  Do not insert Python marker ops into Qwen forward again.
- Built-in PyTorch profiler record-function ops are also rejected for this
  compiled path. If the returned handle is consumed, Inductor fails during graph
  lowering; if the call is left unused, it compiles but is removed and emits no
  ROCm marker records. Do not use `_record_function_enter/_exit` as the next
  marker strategy.
- Residual pre-fold is not the next GGUF ladder. The active combined GGUF
  bundle still keeps the `VLLM_GFX906_MLP_DOWN_LLMM1_RESIDUAL_PREFOLD` path
  behind `quant_config is None`, and the historical FP16 dense record rejects
  MLP-down residual pre-folding as a serving promotion. It is legitimate source
  evidence, but not a current promotion candidate without profiler evidence.
- A direct F16 GGUF residual pre-fold candidate was later tested anyway as a
  controlled source check. It enabled the native row-parallel residual extension
  under the current best fused-GDN plus native GGUF SwiGLU bundle and completed
  the normal warmups -> strict -> fixed-token ladder, but it was slightly slower:
  strict `63.631`, `c1_2000` `64.296`, and `c1_10000` `60.554` backend TPS.
  Reject residual pre-fold as the missing dense GGUF lever.
- Optional FLA / `causal_conv1d` package absence is not enough to explain the
  GGUF gap. The v0.2 release image is missing `fla`, `causal_conv1d`, and
  `flash_linear_attention`, yet the clean HF-weight control on the same image
  reaches release-band TPS. GGUF runs emit the Transformers fallback warning and
  short-sequence format warnings; those are hot-path profiling clues, not a
  standalone installation fix.
- MoE TP4 default-scheduler / broad graph capture is not the missing
  performance lever. Removing the restrictive `--max-num-seqs 1
  --max-num-batched-tokens 1024` launch args restored the HF-style graph shape
  (`max_num_batched_tokens=2048`, broad capture sizes through `512`, and
  `max_cudagraph_capture_size=512`) and kept strict validity, but the fixed-token
  results stayed flat at `73.130` backend TPS for `c1_2000` and `65.552` for
  `c1_10000`. Keep the launch-shape correction for future comparability, but
  treat the remaining gap as source-path parity work.
- The c1 topk8 MoE fastpath is not a current explanation for the MoE GGUF
  throughput gap. A diagnostic TP4 eager 4K launch with
  `VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH=force` reached health and produced coherent
  output, but the published image source contained no qwen c1 topk8 overlay
  strings and the logs emitted no active/rejected fastpath messages. The env var
  is inert in this release image, so it cannot explain the HF-versus-GGUF TP4
  gap.
- The recurring Transformers FLA / causal-conv warning is loader-boundary
  evidence, not proof of a request-time torch fallback. Source inspection shows
  the warning lives in the vendored Transformers Qwen3.5 constructors, while the
  GGUF loader explicitly instantiates a meta-device `AutoModelForCausalLM` dummy
  model to extract parameter names and conversion mappings. The vLLM serving
  Qwen3.5 path uses vLLM QwenNext classes and `torch.ops.vllm.gdn_attention_core`.
  Do not spend time installing `fla` / `causal-conv1d` as a blind throughput fix
  unless a request-window profile proves those functions are on the hot path.
- The active MoE GGUF overlay is already past the stock `GGUFMoEMethod` issue.
  vLLM still sets `quantization=gguf` and `load_format=gguf`, but the active
  overlay returns `UnquantizedFusedMoEMethod` for the F16/BF16/F32 MoE expert
  layers under the force-unquantized path. The remaining MoE gap is therefore
  not another quant-method-selection audit; it needs graph/timing parity against
  HF around GatedDeltaNet, FusedMoE experts, shared expert, residual/norm, and
  logits.
- The next timing route should start at the existing `gdn_attention_core`
  custom-op wrapper, not inside the compiled Qwen forward. Source inspection of
  the release image shows `gdn_attention_core` is registered in Python through
  `direct_register_custom_op` and calls `self._forward_core(...)`; it is not
  exposed as a separate C++ extension symbol in the installed image. A tiny
  `.20` ROCm control proved that a `direct_register_custom_op` body can record
  `time.perf_counter_ns()` and can also call `torch.cuda.synchronize()` while
  the caller remains `torch.compile(fullgraph=True)`. This promotes an
  env-gated timing wrapper around `gdn_attention_core` as the next low-impact
  diagnostic route. It does not yet time projection, output projection, logits,
  or sampler, and it is not benchmark evidence.
- MoE GGUF has moved beyond artifact absence and unsupported-architecture
  failure, but it remains blocked before benchmark warmups. A Qwen3.6 35B-A3B
  F16 GGUF artifact exists on NVMe and reports
  `general.architecture=qwen35moe`, F16 file type, 41 blocks, 256 experts, and
  8 active experts. Its embedded tokenizer metadata is present, and sampled
  HF-versus-GGUF token IDs matched.
- The baseline vLLM smoke rejects `qwen35moe` as an unsupported GGUF
  architecture. A copied experimental image that aliases `qwen35moe` toward
  plain Qwen3 MoE gets farther, but fails during expert weight loading because
  Qwen3.6 35B-A3B uses Qwen3.5-MoE-style config and `512` MoE intermediate
  dimensions. This rejects architecture aliasing as a serving fix.
- The ai-infos image inspected during this pass has Qwen3.5/Qwen3.5-MoE HF
  model and config source support, but it still rejects `qwen35moe` in the
  GGUF parser path. The useful takeaway is source-level Qwen3.5-MoE support,
  not a ready GGUF runtime.
- A parser-only `qwen35moe` metadata shim can read the MoE GGUF without loading
  tensors and extracts the expected text-model fields: hidden size `2048`,
  `40` text layers, `16` attention heads, `2` KV heads, head dim `256`,
  `512` MoE intermediate size, `512` shared-expert intermediate size, `256`
  experts, `8` experts per token, and `rope_theta=10000000.0`.
- Using the Qwen3.5-MoE text config with gguf-py's `qwen35moe` tensor-name map
  maps `663` of `693` text parameters before loading tensor data. The unmapped
  parameters are the `30` linear-attention `dt_bias` tensors, which need an
  explicit `ssm_dt.bias` mapping. This is the next narrow source bridge before
  any serving launch.
- A real experimental Qwen3.5-MoE text GGUF route now parses, maps the missing
  `ssm_dt.bias` tensors, loads expert gate/up/down stacks, and reaches API
  health in eager mode. It required disabling accidental M-RoPE from the
  text-only config and adding the missing Mamba state-copy helper to the
  text-only base class.
- That route is still rejected for coherence and throughput. The graph path
  fails inside the slow F16 GGUF MoE fallback during HIP stream capture, and the
  eager path accepts requests but returns no client-visible bytes before
  timeouts, including an 8-token streaming probe. It cannot enter warmups,
  `c1_128`, `c1_2000`, or `c1_10000`.
- For graph mode, forcing F16 GGUF MoE expert layers onto the normal
  unquantized FusedMoE method gets past the stream-capture failure and reaches
  API health. It still does not produce coherent text.
- The missing Qwen3.5 linear-attention GGUF layout bridge has now been proven
  offline. Inverting llama.cpp's V-head reorder reconstructs HF tensors for
  QKV V rows, Z, alpha, beta, `dt_bias`, `A_log`, conv V channels, and
  out-proj columns. Applying that full inverse transform in both the direct and
  stacked Qwen3.5 load branches preserved startup but still failed
  deterministic probes: `Hello` returned incoherent tokens and the tiny math
  prompt timed out. This rejects the simple theory that the MoE GGUF failure is
  only unresolved linear-attention tensor packing.

The active dense question is now source-path performance, not "why is the output
garbage?" The leading target is vLLM GGUF/Qwen3.5 execution overhead or routing
after tensor materialization, especially the remaining gap between the
fused-GDN plus native GGUF SwiGLU path and the published HF-weight FP16 path.

The active MoE question is no longer just parser availability. vLLM now has an
experimental `qwen35moe` / Qwen3.5-MoE GGUF route that can reach graph-mode
health with forced unquantized FusedMoE, restored text M-RoPE, and the full
inverse Qwen3.5 linear-attention GGUF layout bridge applied to both direct and
packed projection loads. The coherent controls now cover both the artifact path
and the shared vLLM model code path: llama.cpp returns comma from the same GGUF
file, and HF-weight vLLM returns comma under the same reduced raw `Hello` probe.
The live vLLM GGUF route needed one more correction: Qwen3.5-MoE GGUF norm
parameters arrive as non-offset RMSNorm values, while vLLM's `GemmaRMSNorm`
runtime applies `1 + weight`. A narrow runtime norm-offset correction made the
GGUF layer-0 post-norm and final hidden match HF and changed raw `Hello` from
`_` to comma. A short no-trace sanity prompt returned coherent text. A clean
full-length benchmark overlay from the norm-offset fix was tried next: graph
mode first failed at the hybrid KV-cache layout assertion, and eager mode
completed the eight warmups but only at `12.455`-`14.060` backend TPS with
identical no-close reasoning-loop output. An env-gated graph KV-layout bypass
then got graph mode healthy and raised warmups to `73.579`-`73.655` backend
TPS, but the benchmark prompt still did not close. The uncapped strict request
was stopped after the live metric estimate crossed `60071` generated tokens.
A faithful branch probe now narrows the active MoE GGUF bug to the first
post-special-token decode step: HF prefers `Thinking` while GGUF prefers
`Here`. Eager mode reproduces that same branch flip, so the failure is not
graph-only. However, the current boundary hooks still capture only
`final_hidden` instead of layer-0/layer-39 internals, so eager boundary tracing
is rejected as a localization method until the hook is moved deeper into the
model path or replaced by an offline reduced diagnostic. The next useful source
work is benchmark-prompt/think-close behavior, late-layer hidden-stream
localization, and MoE GGUF hot-path throughput under graph mode, not another
eager benchmark ladder.

A later quant-method audit rejects the simple "F16 GGUF is still using the
GGUF adapter for hot linears" performance theory. Under the same TP4 graph-mode
candidate, all `760` recorded `LinearBase` method selections across four TP
workers selected `UnquantizedLinearMethod`; all `160` `FusedMoE` selections used
forced `UnquantizedFusedMoEMethod`; and there were zero hot-path
`GGUFLinearMethod` / `GGUFMoEMethod` selections. The next MoE performance work
should compare compiled execution structure and Qwen3.5-MoE/GatedDeltaNet path
parity against HF weights, not another broad GGUFLinear bypass.

The request-gated Qwen3.5-MoE model-loop trace rejects gross scale drift as the
remaining MoE GGUF blocker. With the runtime non-SSM RMSNorm offset fix and the
graph KV-layout bypass, the TP4 route reaches health and reproduces the branch
flip after shared token id `248046`: HF ranks `Thinking` id `90700` first,
while GGUF ranks `Here` id `8160` first. Layer 0 and layer 39 aggregate stats
stay close to the HF-weight comparator, including corrected layer-0
post-input-layernorm output and similar late-layer norms. The active problem is
therefore hidden-vector direction / trajectory drift before final logits, not a
simple final-row lookup, tokenizer ID map, or norm-scale failure. The next
diagnostic should compare request-gated row vectors against an HF row-vector
run at layer 0, layer 39, loop-final norm, and final hidden.

The first HF row-vector comparator is only a partial localization. Reusing the
GGUF modeltrace overlay on HF weights corrupted the comparator path: it reached
health but returned `!!` and emitted NaN trace values. A narrower HF-only
hidden-vector overlay with row-value emission preserved the known-good branch:
the response was `Thinking Process`, and event 1 ranked `Thinking` id `90700`
above `Here` id `8160`. It showed that GGUF remains close at some late rows
(`layer39.post_mlp` cosine about `0.984005`) while the branch logits still
flip. Treat those rows as first-call/request evidence, not final proof of the
decode-step boundary. The next trace must record call-indexed decoder rows
for call 0/1/2 so the layer comparison lines up with the exact compute-logit
event where the branch splits.

The external `Kausik-A/qwen3.6-27b-mi50-vllm` bundle is useful as a source
reference but not as a release path. It confirms several fixes already present
or independently rediscovered in this investigation: `qwen35` /
`qwen35moe` GGUF aliases, `ssm_dt.bias` to `linear_attn.dt_bias`, text-only
M-RoPE suppression, quantized embedding/lm-head plumbing, conv1d 2D-to-3D
reshape, `IsHybrid` hooks, and tuple-shard handling in
`MergedColumnParallelLinear` for fused GGUF QKVZ tensors. Its launch target is
single-GPU, eager-mode, ROCm 6.3-era, 4096-context, quantized dense GGUF on one
MI50/eGPU. Treat it as a packaging and loader-reference source, not evidence
for our ROCm7.2 TP8/TP4 full-BAR/P2P-on 131K benchmark path. The MiniMax M2
aliases in its GGUF loader are worth comparing when resuming MiniMax GGUF
compatibility work.

The current MoE GGUF TP4 source candidate combines the runtime non-SSM
RMSNorm offset fix, explicit `ssm_a` / `A_log` loading, and full-attention
q/k norm replacement. That bundle repairs the short branch trajectory and now
passes the full 131K graph-mode benchmark correctness ladder on `.20`: the
uncapped strict run stopped normally with `qwen_gate_valid=true`, `3428`
completion tokens, and `71.916` backend TPS; `c1_2000` completed at `73.380`
backend TPS; and `c1_10000` completed at `65.749` backend TPS. Promote it as
the first MoE GGUF strict-valid correctness candidate. Reject it as a
performance candidate because it remains far below the FP16 v0.2.1 TP4
reference band: strict `114.725`, `c1_2000` `116.429`, and `c1_10000`
`109.531` backend TPS on `.20`. A follow-up run with
`VLLM_GFX906_INTERNAL_BOUNDARY_TRACE=0` also stayed in the same band: strict
`69.961`, `c1_2000` `73.228`, and `c1_10000` `65.643` backend TPS. Do not
rerun trace/no-trace variants. The next source change needs to target GGUF MoE
execution speed, especially forced unquantized FusedMoE materialization and
whether GGUF-loaded F16 tensors can use the same ordinary FP16 fast path as HF
weights.

GGUF normally packages tokenizer metadata with the model file, and the local
F16 GGUF tokenizer ID audit did not find a simple shared-vocab ordering shift.
For vLLM reproduction work, explicit HF tokenizer/config pinning still remains
the safer launch rule because model config, special tokens, and chat-template
handling can differ even when a GGUF file includes tokenizer fields.

The historical sections below are preserved because they explain how the current
source boundary was reached.

## What Promoted As Useful Evidence

### Text-only configuration

Outcome:

- A text-only Qwen3.6 config is required for this source lane.
- It avoids falling into the wrong multimodal / M-RoPE behavior during GGUF
  testing.

Promotion:

- Promoted as a required diagnostic precondition.

Reason:

- It separates GGUF tensor-parallel behavior from unrelated model-config
  path drift.

### TP1 / TP2 first-token sanity

Outcome:

- TP1 Q4_0 GGUF returned the expected comma-like first-token behavior for a
  simple `Hello` probe.
- TP2 Q4_0 GGUF also preserved the same first-token candidate shape.

Promotion:

- Promoted as baseline sanity evidence.

Reason:

- The loader, tokenizer, basic decode loop, and small-TP execution path are not
  universally broken.

### Final hidden-state and logit tracing

Outcome:

- TP1 and TP4 final hidden/logit traces diverge before the final token choice.
- TP1 top-token behavior remains coherent.
- TP4 shifts the first token toward newline / punctuation behavior.

Promotion:

- Promoted as diagnostic evidence.

Reason:

- The error is not primarily an lm-head gather or tokenizer ID problem.

### Layer-0 GatedDeltaNet tracing

Outcome:

- TP1 and TP4 start from the same input-normalization sum at layer 0.
- The first material divergence appears at the layer-0 GatedDeltaNet attention
  output.

Promotion:

- Promoted as the current root-cause boundary.

Reason:

- This narrows the next source work to the GatedDeltaNet tensor-parallel /
  GGUF-weight path, not the final logits path.

### Low-level GGUF tensor-info scanning

Outcome:

- A small C scanner was used to inspect GGUF tensor metadata directly from the
  Q4_0 and FP16/half split files.
- Layer-0 GGUF names and shapes matched the Qwen3.5 llama.cpp convention:
  `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`, and `ssm_out`.
- The Q4_0 file stores `attn_qkv` as `5120 x 10240`, `attn_gate` as
  `5120 x 6144`, `ssm_alpha` / `ssm_beta` as `5120 x 48`, and `ssm_out` as
  `6144 x 5120`.

Promotion:

- Promoted as source diagnostic evidence.

Reason:

- The file-level layout makes a simple q/k/v group-reorder explanation less
  likely. The more credible target is how vLLM partitions and fuses those GGUF
  tensors under TP4+.

### Official GGUF format reference

Outcome:

- The upstream GGUF specification was checked against the low-level scanner:
  <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>.
- The spec confirms that tensor data is located through explicit tensor-info
  entries and aligned tensor-data offsets.
- The spec confirms the active tensor type IDs seen in the scanner and runtime
  traces: `Q4_0` is type `2` and `Q8_0` is type `8`.

Promotion:

- Promoted as source-orientation evidence.

Reason:

- The file-level scanner is interpreting the relevant GGUF type IDs and tensor
  offsets consistently with the public GGUF specification. The active failure
  is therefore more likely in vLLM materialization or tensor-parallel execution
  than in basic GGUF header parsing.

### External vLLM GGUF support boundary

Outcome:

- vLLM's public GGUF documentation still describes GGUF support as
  experimental and under-optimized:
  <https://docs.vllm.ai/en/stable/features/quantization/gguf/>.
- The same documentation recommends using the base-model tokenizer instead of
  extracting a tokenizer from GGUF, and using `--hf-config-path` when the GGUF
  metadata cannot be converted into a usable Hugging Face config.
- The public Qwen3.5 GGUF issue and PR focus on startup/config problems:
  `qwen3_5` versus `qwen35` GGUF naming, and `depth` fallback for Qwen3.5
  vision configs:
  <https://github.com/vllm-project/vllm/issues/38122> and
  <https://github.com/vllm-project/vllm/pull/38140>.
- A local-GGUF-file issue shows a separate failure mode where a raw `.gguf`
  file path can bypass `--hf-config-path` during config handling:
  <https://github.com/vllm-project/vllm/issues/36456>.

Promotion:

- Promoted as an external boundary and launch-style guardrail.

Reason:

- The local dense GGUF path is past the public startup failure class: it loads,
  serves, and the later FP16 dense path now produces coherent strict-valid
  output. Upstream `qwen3_5` to `qwen35` mapping is still useful, but it is not
  the remaining performance root cause after loading and tensor-layout repair.
- The active launch style should stay on explicit tokenizer/config handling and
  avoid changing to local-file GGUF paths until correctness is fixed.

### Forward-time checksum tracing

Outcome:

- Load-time checksum instrumentation fired too early: qkvz and `out_proj`
  qweights were still uninitialized at the end of `load_weights`.
- Forward-time instrumentation observed materialized layer-0 qkvz, beta/alpha,
  conv, and output-projection tensors.
- TP4 qkvz physical shard order appears as `[3, 0, 1, 2]`, with the first
  physical region corresponding to z/gate and the later regions corresponding
  to q, k, and v.
- The logical shard-order patch is directionally necessary because the runtime
  apply path must present q, k, v, z to the model code.
- The same TP4 one-token `Hello` probe still returned `.` as the top token, so
  materialized checksum visibility did not become a serving fix.

Promotion:

- Promoted as source boundary evidence.

Reason:

- Gross tensor shape and basic materialization are no longer enough to explain
  the failure. The next credible boundary is the GatedDeltaNet prefill/decode
  state path or a more subtle shard semantic mismatch after qkvz materializes.

### TP1 control under the same forward-time instrumentation

Outcome:

- TP1 was rerun with the same experimental image, logical GGUF shard-order
  patch, forward-time checksum hooks, and deterministic one-token `Hello`
  probe used for the TP4 failure reproducer.
- TP1 returned `,` as the top token, with `!`, ` everyone`, ` all`, and
  ` there` behind it in the top-logprob set.
- TP1 also materialized qkvz with physical shard order `[3, 0, 1, 2]`; the
  offset map was the full local z, q, k, and v tensor rather than the TP4 local
  shard sizes.
- TP4 under the same image returned `.` first and `,` second.

Promotion:

- Promoted as the current control for GGUF correctness debugging.

Reason:

- The physical z/q/k/v GGUF order is present in the coherent TP1 run too. That
  makes a simple "sort the qkvz shard IDs" fix insufficient.
- The remaining source target is the tensor-parallel shard boundary and
  reassembly behavior that appears only once qkvz/ba/`ssm_out` are split across
  ranks.

### llama.cpp Qwen3.5 / Qwen3.6 split-granularity fix

Outcome:

- The upstream llama.cpp PR
  <https://github.com/ggml-org/llama.cpp/pull/23843> is merged and explicitly
  fixes Qwen3.5 / Qwen3.6 tensor-parallel split granularity for heterogeneous
  quant mixes.
- The PR says the wrong tensors were being used to determine granularity when
  splitting quantized tensors across devices.
- The local llama.cpp checkout includes this behavior: qkv and gate split
  configuration can fall back to `ssm_out.weight`.
- The scanned Qwen3.6 27B Q4_0 GGUF file has layer-0 `attn_qkv` and
  `attn_gate` stored as Q4_0, while `ssm_out` is Q8_0.

Promotion:

- Promoted as the highest-value external source clue found so far.

Reason:

- The local TP4 failure is also a Qwen3.5 / Qwen3.6 tensor-parallel GGUF path
  with a mixed-quant GatedDeltaNet block.
- vLLM's GGUF merged-loader path slices each loaded quantized tensor by its own
  local shard size. It does not yet show the same companion-tensor granularity
  logic that llama.cpp added for this model family.
- This does not prove the vLLM bug is identical, but it gives a concrete
  source-level diagnostic direction: compare vLLM's qkv/gate/beta/alpha and
  `ssm_out` split boundaries against llama.cpp-style Qwen3.5 / Qwen3.6 split
  segments.

### C split audit against the active Q4_0 GGUF file

Outcome:

- A scratch C auditor was used to read the active GGUF file directly and
  calculate alignment, packed bytes, quant block sizes, and TP2/TP4/TP8 split
  sizes for layer-0 `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`, and
  `ssm_out`.
- `attn_qkv` and `attn_gate` are Q4_0 with 32-element / 18-byte blocks.
- `ssm_out` is Q8_0 with 32-element / 34-byte blocks.
- The TP4 and TP8 axis-local split sizes are cleanly divisible by the relevant
  quant blocks for these tensors.

Promotion:

- Promoted as split-audit evidence.

Reason:

- This rejects the simple "GGUF quant block remainder" theory.
- It keeps the llama.cpp clue alive at the model-family segment level: the
  issue may be how qkv/gate/beta/alpha segments are paired with `ssm_out`
  granularity, not whether raw tensor sizes divide by TP.

### TP4 all-rank trace

Outcome:

- TP4 was rerun with all-rank layer-0 tracing enabled under the same
  experimental image and P2P-on path.
- The deterministic one-token `Hello` probe reproduced the same failure:
  `.` remained the top token and `,` remained second.
- All ranks agreed on the final wrong top IDs after synchronization.
- Rank-local qkv, beta/alpha, GatedDeltaNet output, and output-projection
  activity differed before the synchronized final hidden state.

Promotion:

- Promoted as failure-boundary evidence, not as a fix.

Reason:

- The TP4 failure is not a single-rank final-logit artifact and not a final
  gather-only bug.
- The next trace needs cleaner logical labels for q/k/v/z, beta, alpha, and
  `ssm_out` slices so the source work can distinguish a segment-order issue
  from a GatedDeltaNet state/reassembly issue.

### TP4 segment-labeled trace

Outcome:

- A follow-up TP4 run added explicit q, k, v, z, beta, and alpha labels to the
  rank-local layer-0 trace.
- The deterministic one-token `Hello` probe still returned `.` as the top token
  and `,` second.
- The labeled trace showed populated q/k/v/z and beta/alpha segments before the
  GatedDeltaNet core.
- All ranks still agreed on the final wrong logits after synchronization.

Promotion:

- Promoted as source-boundary evidence, not as a fix.

Reason:

- The failure is not an obviously empty q, k, v, z, beta, or alpha segment.
- This keeps the active source target on segment ordering/granularity semantics,
  GatedDeltaNet state handling, and tensor-parallel reassembly rather than
  another launch-flag experiment.

### Qwen3.5 GatedDeltaNet source fork point

Outcome:

- Current vLLM Qwen3.5 code uses `QwenGatedDeltaNetAttention` with
  `gqa_interleaved_layout=False`.
- That path builds `in_proj_qkvz` as four independent output sizes for
  `[q, k, v, z]`, and `in_proj_ba` as two independent output sizes for
  `[b, a]`.
- The FP16 release overlay used for the published reproduction package fuses
  b/a into `in_proj_qkvz` for its runtime path.
- The release runtime linear overlay also contains GGUF-specific packed-shard
  handling, including restrictions around multi-index shard IDs.

Promotion:

- Promoted as the highest-value source comparison.

Reason:

- The FP16 fused projection shape is a proven release path for FP16, but GGUF
  tensor materialization is more sensitive to packed shard IDs and late
  quantized-weight initialization.
- The next GGUF source candidate should not blindly reuse the FP16 fused qkvzba
  shape. It should compare a current-style split `in_proj_qkvz` / `in_proj_ba`
  path against the fused overlay while tracing q/k/v/z and b/a before the
  GatedDeltaNet core.

## Rejected Paths

### KV replica mode change

Outcome:

- `VLLM_QWEN35_KV_REPLICA_MODE=mod` did not repair TP4 behavior.

Disposition:

- Rejected as a fix.

Reason:

- TP4 does not appear to be failing because of the tested K/V replica mode.

### No-async scheduling path

Outcome:

- Disabling async scheduling did not produce a usable TP4 path. The engine did
  not initialize cleanly under the tested no-async launch shape.

Disposition:

- Rejected as a practical reproduction path.

Reason:

- It failed before a meaningful correctness or throughput comparison.

### Forced GGUF dequantized matmul fallback

Outcome:

- Forcing dequantized matmul fallback changed the TP4 output but did not restore
  TP1-like correctness.

Disposition:

- Rejected as a fix.

Reason:

- The issue is not only the ROCm GGUF quantized matmul kernel.

### Torch GatedDeltaNet reference path

Outcome:

- Forcing a torch GatedDeltaNet prefill path did not repair TP4 first-token
  correctness.

Disposition:

- Rejected as a fix; promoted as root-cause evidence.

Reason:

- The failure is likely in tensors entering the GatedDeltaNet path or the
  tensor-parallel GGUF shard layout around that path, not only in the custom
  GatedDeltaNet core implementation.

### Logical GGUF shard-order patch

Outcome:

- Sorting logical GGUF shard IDs before linear application did not repair TP4.
- A TP4 P2P-on one-token probe returned `.` as the top token and `,` as the
  second token, while TP1 remains the expected reference for this prompt.
- The layer-0 trace still showed divergence before any benchmark-worthy output.
- Later forward-time checksum tracing showed why the patch was plausible:
  physical qkvz shard order is not logical q, k, v, z order. However, applying
  that logical order alone did not repair the TP4 path.

Disposition:

- Rejected as a fix; promoted as negative evidence.

Reason:

- The failure is not only unordered shard application at the GGUF linear-method
  boundary.

### vLLM partition-shape source inspection

Outcome:

- The traced TP4 `in_proj_qkvz` row count matches the expected local q/k/v/z
  partition under vLLM's `MergedColumnParallelLinear`.
- The traced TP4 `out_proj` Q8_0 packed input columns match the expected
  `RowParallelLinear` local input partition.

Disposition:

- Rejected the simple loader-size mismatch hypothesis.

Reason:

- The remaining bug is more likely actual per-rank tensor content, shard offset
  semantics, GGUF quantization unpacking, or GatedDeltaNet core inputs than a
  gross full-size versus local-size partition error.

### Load-time checksum placement

Outcome:

- Load-time checksums were attempted before the final GGUF quantized tensors
  were materialized.
- qkvz and `out_proj` qweights showed as uninitialized at that point.

Disposition:

- Rejected as the final diagnostic location.

Reason:

- The checksum hook must run during or after the first forward path to reflect
  the actual tensors used by GGUF linear execution.

### Full benchmark ladder before correctness

Outcome:

- The normal benchmark ladder was intentionally not run for GGUF TP4/TP8 after
  first-token sanity failed.

Disposition:

- Rejected until first-token correctness passes.

Reason:

- Running warmups, strict, `c1_2000`, and `c1_10000` would produce throughput
  numbers for an incorrect model path.

## Prior FP16 Source-Work Guardrails

The existing dense FP16 key-learnings record matters for GGUF because it shows
which ideas should not be repeated before correctness is repaired:

- GDN projection dual-streaming was semantically valid but a decode regression.
- Replacing dense GDN output zeroing with `torch.empty(...)` was semantically
  safe, but not a 65 TPS lever.
- Upstream-style GDN output flattening was useful source alignment, not a
  dense gate-clear lever.
- The current FP16 winner's profile already identified the Qwen3.5
  linear-attention `in_proj_qkvz` group as a real runtime shape, but the
  material bottleneck remained broader RowParallel/collective behavior.

Disposition:

- Promoted as a guardrail against repetitive experiments.

Reason:

- GGUF TP4/TP8 is currently blocked on correctness, so repeating throughput
  overlays that did not promote in FP16 would only create noisy failures. The
  GGUF lane should focus on file-layout invariants, GGUF loader materialization,
  qkvz/ba shard semantics, and TP state behavior first.

## Active Source Hypothesis

The next credible source target is the GGUF tensor-parallel execution around
the Qwen3.5 / Qwen3.6 GatedDeltaNet path. The file-level tensor scan and
llama.cpp layout comparison suggest q/k/v are contiguous in `attn_qkv` and
z/gate is a separate projection. Forward-time checksums show that vLLM
materializes TP4 qkvz with physical shard order `[3, 0, 1, 2]`, so q/k/v/z
logical ordering matters, but the logical-order patch alone did not restore
correctness.

The upstream/current Qwen3.5 source keeps qkvz and b/a as separate projection
families for non-interleaved Qwen3.5 layout, while the FP16 release overlay
used a fused qkvzba runtime shape. The next source pass should treat that as an
explicit fork point: either prove the fused shape can be made GGUF-correct, or
move the GGUF path toward the current split qkvz/ba source shape.

The highest-value boundaries are:

- `in_proj_qkvz`
- `in_proj_ba`
- GGUF packed shard ID handling for q/k/v/z and b/a
- TP1 versus TP4 forward-time tensor checksums
- q/k/v/z split and repeat semantics
- beta / gate tensors
- GatedDeltaNet output before normalization
- output projection after the GatedDeltaNet block

The current hypothesis is that TP4+ GGUF sharding, split granularity,
prefill/decode state, or GatedDeltaNet reassembly around this path is wrong for
Qwen3.6, while TP1/TP2 avoid or reduce the failing shape. The llama.cpp
Qwen3.5 / Qwen3.6 fix makes split granularity a first-class suspect because it
was needed for heterogeneous quant mixes around the same model family.

The next diagnostic should audit split boundaries against llama.cpp-style
Qwen3.5 / Qwen3.6 segments, then compare TP4 all-rank qkvz/ba/`ssm_out` traces
against the coherent TP1 control with logical q/k/v/z and beta/alpha labels.
Shape checks and basic quant-block divisibility checks are no longer enough.

## TP1 Segment-Labeled Control

Outcome:

- The same segmenttrace hook used for the TP4 failure was run at TP1.
- TP1 returned the coherent deterministic `Hello` top token `,`.
- TP4 under the same hook and P2P-on settings returned `.` first and `,`
  second.
- q, k, v, z, beta, and alpha traces were populated in both paths.

Disposition:

- Promoted as control evidence.
- Rejected as a fix.

Reason:

- The trace hook is not the reason TP4 is wrong.
- Segment population by itself is not enough to prove correctness; TP4 still
  diverges from the coherent TP1 path.
- The next source target should compare exact GGUF tensor ranges and
  per-rank logical segment semantics, then inspect GatedDeltaNet reassembly
  rather than adding more launch flags.

## TP1 Versus TP4 Segment Checksums

Outcome:

- A compact checksum-only trace was run for TP1 and TP4 under the same image,
  GGUF model, P2P-on settings, and deterministic one-token prompt.
- TP1 returned `,`; TP4 returned `.`.
- At the final one-token decode step, TP4 rank-local q/k/v/z/b/a inputs
  aggregate back to the coherent TP1 sums:
  - q: TP4 aggregate `19.639277`, TP1 `19.639275`
  - k: TP4 aggregate `8.790018`, TP1 `8.790018`
  - v: TP4 aggregate `300.671662`, TP1 `300.671661`
  - z: TP4 aggregate `-150.053726`, TP1 `-150.053726`
  - b: TP4 aggregate `80.371098`, TP1 `80.371094`
  - a: TP4 aggregate `161.471282`, TP1 `161.471283`
- The GatedDeltaNet core/output does not match:
  - TP1 core sum: `5.396257`
  - TP4 rank-local core-sum aggregate: `-2.240752`
  - TP1 output sum: `9.024781`
  - TP4 replicated output sum: `12.022104`

Disposition:

- Promoted as narrowed source evidence.
- Rejects benchmark throughput testing until TP correctness is repaired.

Reason:

- The q/k/v/z/b/a loader and split path now looks substantially less likely
  to be the whole failure: the TP4 inputs cover the same aggregate data as TP1.
- The divergence appears after those inputs are formed, at the GatedDeltaNet
  core/state boundary or output reassembly.
- The next source experiment should instrument the GatedDeltaNet core chunks
  and cached state across prefill/decode, not run warmups or benchmark ladders.

## Post-Conv Q/K Divergence And Repeat-Order Fix Candidate

Outcome:

- A TP1 rank-equivalent chunk hook compared TP1 four-way head chunks against
  TP4 ranks after the internal `causal_conv1d` step and before the recurrent
  GatedDeltaNet update.
- Value chunks, b/a chunks, `A_log`, and `dt_bias` matched by TP1 chunk and TP4
  rank.
- Q and K did not match by chunk/rank:
  - TP1 chunk 0 query/key sums: `13.044325` / `7.324649`
  - TP4 rank 0 query/key sums: `11.564686` / `14.409418`
  - TP1 chunk 1 query/key sums: `15.954918` / `8.375532`
  - TP4 rank 1 query/key sums: `16.515209` / `6.321072`
  - TP1 chunk 2 query/key sums: `14.134207` / `6.682993`
  - TP4 rank 2 query/key sums: `11.053076` / `1.243456`
  - TP1 chunk 3 query/key sums: `15.784380` / `3.986878`
  - TP4 rank 3 query/key sums: `19.784855` / `4.396105`
- The follow-up conv-weight and pre-conv input checks rejected a simple
  `conv1d.weight` loader fault:
  - TP1 chunk-equivalent pre-conv q/k/v inputs matched TP4 rank-local q/k/v
    inputs exactly.
  - TP1 chunk-equivalent q/k/v `conv1d.weight` slices matched TP4 rank-local
    conv weights exactly.
- The mismatch was tied to Q/K expansion order. With tiled repeat enabled,
  TP1 repeats Q/K over the global head list while TP4 repeats only each rank's
  local Q/K head list. With repeat-interleave enabled, TP1 chunk summaries line
  up with TP4 rank summaries and both TP1 and TP4 return `,` on the
  deterministic `Hello` one-token probe.
- The same repeat-interleave candidate also passed TP2 and TP8 first-token
  sanity. TP8 no longer produced the earlier catastrophic first-token behavior.

Disposition:

- Promoted as the first source-level correctness candidate.
- Rejects additional throughput runs until the repeat-interleave candidate
  clears broader correctness checks.

Reason:

- The GGUF in-projection, value slice, b/a slice, recurrent scalar parameters,
  pre-conv q/k/v inputs, and q/k/v conv weights are no longer broad suspects
  for the deterministic first-token failure.
- The active target is Q/K repeat ordering for Qwen3.5 / Qwen3.6 GatedDeltaNet
  under tensor parallelism.
- Repeat-interleave is a candidate fix only. It has passed TP1, TP2, TP4, and
  TP8 first-token sanity, but not full-context warmups, uncapped strict,
  `c1_2000`, or `c1_10000`.

## Full-Context TP8 Repeat-Interleave Rejection

Outcome:

- A clean TP8 full-context run on `.20` used the same Q4_0 GGUF model with
  `VLLM_DTYPE=half`, P2P-on, `MAX_MODEL_LEN=131072`,
  `VLLM_QWEN35_TILE_QK_REPEAT=0`, and the normal Qwen begin-think benchmark
  harness.
- The server loaded successfully and reported the full 131072-token context.
- Eight 2000-token pre-measure warmups completed.
- Warmup backend decode TPS was stable around `54.31` to `54.35`.
- The next `c1_128` uncapped strict request stopped after 12 completion tokens.
- The strict result had `qwen_gate_valid=false`, no visible answer, no think
  close, and a degenerate reasoning fragment.
- Because strict correctness failed, `c1_2000` and `c1_10000` were not promoted
  from this run.

Disposition:

- Reject repeat-interleave as a benchmark-ready GGUF path.
- Keep it as a narrowed source clue because it fixed first-token sanity.

Reason:

- First-token agreement is not strong enough for Qwen3.6 GatedDeltaNet GGUF
  correctness.
- The warmup throughput is stable but below the FP16 reproduction path, and the
  strict gate fails before any fixed-token throughput can be treated as useful
  benchmark evidence.
- The next source work should explain why the benchmark prompt produces
  degenerate reasoning under TP8 full-context even though simple first-token
  probes align across TP1/TP2/TP4/TP8.

## Full-Context FP16/Half GGUF Rejection

Outcome:

- A true FP16/half GGUF dense candidate was converted from the cached Qwen3.6
  27B source weights.
- The low-level scanner verified sampled tensors as FP16/half tensor data.
- A local-file GGUF startup path failed before serving because config probing
  read the GGUF architecture string before the explicit Hugging Face config
  path was applied.
- A minimal experimental patch skipped speculator probing for local GGUF files,
  allowing the explicit config path to drive startup while the GGUF loader still
  used the local FP16/half GGUF file.
- The model then served on `.20` with TP8, P2P-on, `MAX_MODEL_LEN=131072`,
  `VLLM_DTYPE=half`, repeat-interleave, and the normal Qwen begin-think
  benchmark harness.
- Eight 2000-token warmups completed.
- Warmup 1 included first-request overhead and measured `49.438` backend decode
  TPS.
- Warmups 2-8 were stable around `49.687` to `49.744` backend decode TPS.
- All warmups produced the same degenerate repeated reasoning text, with no
  visible answer and no think close.
- The strict request was stopped manually after the warmup degeneration made the
  outcome non-useful.
- `c1_2000` and `c1_10000` were not run or promoted from this candidate.

Disposition:

- Reject as a benchmark candidate.
- Promote as source evidence.

Reason:

- FP16/half GGUF did not repair semantic correctness.
- The steady warmup throughput was also below the published FP16 reproduction
  path.
- This rejects the narrow theory that the Q4_0 quantization format alone caused
  the full-context benchmark failure.
- The remaining source target is the local GGUF config / materialization /
  Qwen3.5 / Qwen3.6 GatedDeltaNet execution path, compared against the
  HF-weight FP16 path.

## FP16/Half TP4 Smoke Localization

Outcome:

- A small-context TP4 FP16/half GGUF launch returned the coherent comma token
  for a raw one-token `Hello` completion.
- Direct chat `Hello` returned only an empty think section.
- A raw completion using the synthetic benchmark prompt degenerated into a
  repeated phrase loop.
- An env-gated force-unquantized linear/embedding experiment produced the same
  outcomes.

Disposition:

- Promote as localization evidence.
- Reject the force-unquantized patch as a fix.

Reason:

- The FP16/half GGUF file and tokenizer path are not globally unreadable.
- The failure appears when meaningful prefill/chat/benchmark context enters the
  Qwen3.5 / Qwen3.6 execution path.
- Forcing normal unquantized linear/embedding method selection did not restore
  semantics, so the next source target is recurrent-state and GatedDeltaNet
  prefill/decode behavior against the known-good HF-weight FP16 path.

## Patch-Bundle Routing And Direct QwenNext Rejection

Outcome:

- The experimental image entrypoint applies the mounted patch bundle after
  Docker mounts are present.
- A bundle-provided Qwen source file can therefore overwrite an individually
  mounted Qwen source file at startup.
- A clean patch bundle was prepared and used for a direct
  `Qwen3NextForCausalLM` launch.
- The mixed-mount failure was removed, but direct QwenNext initialization failed
  before serving because the model received the outer Qwen3.5 config object
  instead of the text config fields expected by QwenNext.

Disposition:

- Promote as source-routing evidence.
- Reject direct QwenNext architecture override as a launch fix.

Reason:

- Future GGUF experiments must prove which Qwen source file is active before
  interpreting output quality.
- Direct QwenNext is still a possible cleanup route, but it requires a small
  config adapter rather than only changing `architectures`.
- Mixed patch bundles and individual file mounts should not be used for
  promotion testing.

## Direct QwenNext And Minimal Qwen3.5 Bundle Boundary

Outcome:

- A later direct `Qwen3NextForCausalLM` diagnostic added the missing text-config
  adapter and split-GDN reorder route.
- That route reached serving and the expected reorder marker fired, but raw
  `Hello` output was invalid byte/glyph garbage.
- The model config is `qwen3_5`, so direct QwenNext remains a diagnostic path,
  not a faithful promotion route.
- A TP4 FP16/half Qwen3.5 smoke using the older `qwen35-gguf-corechecksum`
  bundle, repeat-interleave, and torch GDN disabled returned repeated greeting
  text on raw `Hello`, then hung on a direct chat `Hello` request.
- The chat hang followed the FLA native warning that a short sequence had fewer
  tokens than heads, but that warning is not proof by itself because very short
  prompts can naturally satisfy that condition.
- Enabling the torch GDN prefill fallback avoided the hang but produced
  repeated commas for raw `Hello` and only `<think>` for chat.
- A minimal image-base Qwen3.5 launch without the old bundle failed during
  startup on an uninitialized GGUF embedding parameter.
- A copy of the older bundle was changed so materialized GGUF qweights use
  `WeightType.F16` instead of `WeightType.BF16`, matching the active scanner
  evidence for `Qwen3.6-27B-F16.gguf`.
- That F16 materialization path loaded and answered probes, but raw `Hello`
  still repeated greeting tokens, chat `Hello` returned an empty think block,
  and a longer simple chat prompt echoed part of the prompt inside `<think>`.

Disposition:

- Reject direct QwenNext as a benchmark candidate.
- Reject torch GDN prefill fallback as a semantic fix.
- Reject F16 qweight-type correction as a semantic fix.
- Promote the no-bundle startup failure as source-boundary evidence.
- Promote F16 materialization as a necessary consistency fix for future
  minimal bundles.

Reason:

- The old bundle is not clean: it replaces `qwen3_5.py` and contains trace
  hooks, Q/K repeat behavior, GDN fallback toggles, Mamba-state helpers, and
  GGUF embedding / lm-head materialization logic.
- The no-bundle failure proves at least part of that materialization logic is
  required to serve the active GGUF file.
- The patched-bundle outputs prove that reaching serving is not enough; the
  active Qwen3.5 GGUF path still fails semantic prefill/chat correctness.
- Correcting the materialized qweight type removes a real dtype inconsistency
  in the diagnostic bundle, but it does not explain the garbage output by
  itself.

Next:

- Build a minimal coherent bundle instead of mixing old diagnostics:
  - GGUF embedding / lm-head materialization needed for startup.
  - F16 qweight typing for the active F16 GGUF file.
  - Local-file GGUF config bypass if the explicit config path still requires it.
  - One intentional Qwen3.5 correctness patch at a time.
- Prove active source-file identity in the container before every promotion
  test.
- Do not run the throughput ladder until raw, chat, warmup, and strict
  correctness all pass.

## Minimal Bundle And Garbage-Output Boundary

Outcome:

- A minimal Qwen3.5 bundle with only F16 GGUF embedding/lm-head
  materialization and Q/K repeat-interleave can reach serving.
- That minimal route produces invalid byte/glyph text on raw and chat probes.
- Adding the logical GGUF shard-order patch does not repair the output.
- Adding the older GGUF loader/registry without cache helpers fails before
  serving during hybrid KV/Mamba page-size setup.
- Adding the cache-helper behavior gets the server healthy again, but the raw
  `Hello`, chat `Hello`, and short factual chat probes still produce invalid
  byte/glyph garbage.
- Changing the outer wrapper loader to delegate into `self.language_model` after
  stripping the `language_model.` prefix does not move the raw `Hello` failure;
  it returns the same invalid byte/glyph prefix.
- A CPU-only llama.cpp file-level smoke on the 51 GB F16 GGUF timed out before
  producing a useful answer, so it is inconclusive for artifact correctness.
- A ROCm7.2 llama.cpp build under the same NVMe checkout can run the same F16
  GGUF with GPU offload and produce coherent Qwen-style output from `Hello`.

Disposition:

- Reject the minimal route as a benchmark candidate.
- Promote F16 embedding/lm-head materialization as a startup precondition.
- Promote cache helpers as a startup precondition only when the older
  loader/registry path is used.
- Reject logical shard-order and cache-helper additions as semantic fixes.
- Reject wrapper load delegation as a semantic fix.
- Treat the CPU llama.cpp check as an inconclusive artifact check, not a pass or
  fail.
- Promote the ROCm llama.cpp run as artifact-coherence evidence.

Reason:

- This failure is past simple process startup. The API can answer requests, but
  the token distribution is corrupted.
- Deterministic garbage on trivial probes is not a benchmark warmup issue and
  should not be papered over with more requests.
- The active failure is still a GGUF Qwen3.5/Qwen3.6 semantic mapping issue:
  tensor materialization, config/architecture alignment, GatedDeltaNet
  state/cache behavior, or lm-head/embedding interpretation.
- The next useful evidence needs a boundary comparison or a GPU-capable
  independent GGUF runner; more wrapper-level launch changes are unlikely to
  promote without a trace showing movement.
- The GPU-capable independent runner now exists and points away from the GGUF
  file as the primary suspect. The active target is vLLM's GGUF Qwen3.5 /
  Qwen3.6 execution path.
- The GGUF embedded tokenizer and the HF tokenizer agree over the shared ID
  range, including the tokens involved in the bad vLLM response. The vLLM
  garbage output is therefore not a simple tokenizer-ID shift.
- TP2 is not a practical F16 GGUF discriminator on 2x MI50 for this model:
  after weight load it leaves no KV-cache memory at even 4K context.

Next:

- Compare the minimal GGUF route against the known-good HF-weight FP16 path at
  the earliest stable boundaries:
  - token IDs and embedding output,
  - layer-0 normalized input,
  - q/k/v/z and beta/alpha projections,
  - GatedDeltaNet recurrent/cache state,
  - output projection and lm-head logits.
- Do not run warmups or throughput tiers until raw and chat probes are
  coherent.
- Use the ROCm llama.cpp result as an external correctness control when
  deciding whether a future vLLM source patch actually moves semantics.
- Keep TP4 as the minimum viable vLLM F16 GGUF correctness lane on `.20` unless
  a smaller representative GGUF or more aggressive context/memory reduction is
  introduced only for diagnostics.

### TP4 F16 GGUF embedding / lm-head row diagnostic

Outcome:

- A TP4 P2P-on diagnostic overlay was added to the minimal F16 Qwen3.5 route.
- The overlay printed rank-local GGUF materialization metadata for
  `token_embd.weight` and `output.weight`.
- Sampled rows matched a direct GGUF reader baseline for token IDs `9419`,
  `19748`, `71741`, `248068`, and `248069`.
- The same diagnostic run still produced deterministic garbage for raw
  `Hello`: `�..��&#hh_manifest`.

Disposition:

- Reject a simple embedding/lm-head row-offset mismatch as the primary fix.
- Promote the row diagnostic as boundary evidence.

Reason:

- The F16 GGUF artifact is coherent under ROCm llama.cpp.
- GGUF and HF tokenizer IDs match over the sampled shared vocabulary range.
- vLLM's rank-local sampled embedding and lm-head rows match the direct GGUF
  reader once materialized.
- The failure is therefore downstream of row loading and upstream of coherent
  logits: layer-0 execution, GatedDeltaNet state/cache behavior, internal
  projection ordering, or TP reassembly remain the active source targets.

Next:

- Trace the F16 TP4 path at:
  - token embedding output,
  - layer-0 normalized input,
  - q/k/v/z and beta/alpha projections,
  - GatedDeltaNet recurrent/cache state,
  - output projection,
  - lm-head logits.
- Do not run warmups or throughput tiers until raw and chat probes are coherent.

### TP4 F16 GGUF layer-0 activity diagnostic

Outcome:

- A TP4 P2P-on layer diagnostic overlay traced the same F16/half minimal
  Qwen3.5 route after the embedding/lm-head row checks.
- The raw `Hello` request still produced deterministic byte/glyph garbage.
- The actual request path showed nonzero q/k/v/z segments, beta/alpha,
  GatedDeltaNet core output, output-projection input, and replicated
  output-projection output.
- The all-rank log was heavily interleaved, so it is useful for activity
  classification but not clean enough for precise per-rank numeric comparison.

Disposition:

- Reject an all-zero layer-0, dead-core, or dropped-output theory.
- Promote the layer diagnostic as source-boundary evidence.

Reason:

- The F16 GGUF artifact is coherent under ROCm llama.cpp.
- Tokenizer IDs and sampled embedding/lm-head rows now match direct GGUF
  baselines.
- Layer-0 is active during the bad vLLM request. The remaining failure is
  semantic: GatedDeltaNet state/cache handling, projection ordering, or tensor
  parallel reassembly is producing the wrong token distribution even though the
  obvious inputs and outputs are nonzero.

Next:

- Use a rank-filtered or structured trace for the benchmark-prompt prefill
  path, not just a one-token `Hello` probe.
- Compare the first divergent boundary against a coherent control before
  running more warmups.
- Keep fixed-token and strict benchmark tiers blocked until warmup output is
  coherent.

### Historical benchmark-prompt prefill failure boundary

Outcome:

- The normal v0.2 profile warmup prompt comes from `run_chat_capture.py`.
- It is a chat request containing a fixed synthetic benchmark instruction plus
  the calibration phrase repeated `32` times.
- In the rank0-only TP4 F16 GGUF trace, that prompt produced a 431-token
  prefill with `max_tokens=1`, `min_tokens=1`, and `ignore_eos=true`.
- The response was only `;`, not coherent Qwen reasoning.
- The request emitted shared-memory broadcast wait warnings before it
  completed.
- Layer-0 rank0 traces for the 431-token prefill showed nonzero normalized
  input, q/k/v/z, beta/alpha, GatedDeltaNet core output, output-projection
  input, and output-projection output.

Disposition:

- Reject the current F16 GGUF route as a warmup candidate.
- Promote the 431-token benchmark-prompt trace as the current source boundary.

Reason:

- The model is not failing because the warmup prompt was guessed incorrectly:
  the prompt is now tied back to the reproduction harness.
- The failure is also not a fully inactive layer-0 path. The full-context
  prompt reaches active projections and core/output stages.
- The semantic failure appears after active prefill computation, likely in
  GatedDeltaNet recurrent state semantics, chunked prefill/decode transition,
  or downstream logit formation.
- Since the warmup output is already garbage, the uncapped strict prompt cannot
  be promoted from this path.

Next:

- Compare the benchmark-prompt prefill state against a coherent control before
  running another warmup ladder.
- If a source candidate changes this boundary, require it to pass warmup
  coherence before `c1_128`, `c1_2000`, or `c1_10000`.

### Dense FP16 GGUF tensor fixes promoted correctness but not throughput

Outcome:

- The dense FP16 GGUF path became coherent after the Qwen3.5 tensor-layout fixes
  were carried into a single release-overlay bundle.
- The active fixed path uses explicit HF tokenizer/config handling for
  reproducibility instead of relying only on GGUF tokenizer metadata.
- The tensor fixes include:
  - F16 materialization where GGUF embedding/lm-head helpers are needed;
  - `norm.weight - 1` conversion for normal RMSNorm weights;
  - preserving `linear_attn.norm.weight` raw;
  - Qwen3.5 / Qwen3.6 linear-attention head/state permutation for QKV/Z,
    beta/alpha, `dt_bias`, `conv1d.weight`, and `A_log`;
  - release overlay hooks applied as one coherent bundle.
- Dense TP8 then produced coherent outputs and passed the strict gate.

Disposition:

- Promote dense FP16 GGUF semantic correctness.
- Reject benchmark promotion because throughput is still below the FP16 release
  path.

Evidence:

- Experimental image plus combined release/GGUF overlay:
  - strict backend TPS: `60.863`
  - `c1_2000` backend TPS: `61.383`
  - `c1_10000` backend TPS: `57.986`
- Official release image plus the same combined overlay:
  - strict backend TPS: `60.656`
  - `c1_2000` backend TPS: `61.399`
  - `c1_10000` backend TPS: `57.966`
- Official release image plus no-trace release-attention GGUF overlay:
  - strict backend TPS: `60.678`
  - `c1_2000` backend TPS: `61.289`
  - `c1_10000` backend TPS: `57.896`
- Clean HF-weight official-image release-overlay control:
  - strict backend TPS: `69.652`
  - `c1_2000` backend TPS: `70.202`
  - `c1_10000` backend TPS: `65.952`
- Published dense FP16 comparison:
  - strict backend TPS: `69.514`
  - `c1_2000` backend TPS: `70.347`
  - `c1_10000` backend TPS: `66.069`

Reason:

- The output-garbage failure is not the current dense blocker.
- The remaining gap is about `8` to `9` backend TPS depending on tier.
- The official image control shows the gap is not caused by the experimental
  image base.
- The HF-weight control shows the same host lane, P2P-on path, official image,
  release overlay, and benchmark harness can still reproduce the dense release
  band, so the remaining GGUF gap is source/materialization/execution-path
  specific.

Next:

- Compare active GGUF CausalLM execution against the published HF-weight FP16
  path at the source/profiling level.
- Focus on Qwen3.5 GatedDeltaNet / linear-attention execution and
  GGUF-specific adapter overhead after tensor materialization.

### Rejected dense GGUF performance hypotheses

Outcome:

- Restoring the release `Qwen3_5Attention` subclass/routing did not improve
  dense GGUF decode TPS.
- A F16 merged-single-matmul GGUF loader patch did not affect the major dense
  linears because `in_proj_qkvz.weight`, `in_proj_ba.weight`, `conv1d.weight`,
  and `out_proj.weight` already load as normal `weight` params rather than
  GGUF `qweight`.
- Switching to `Qwen3_5ForConditionalGeneration` matched release-style
  architecture/page-sizing indicators in an earlier run but left decode around
  `60.59` TPS and worsened first-request prefill.
- Removing `_qwen35_trace_tensor(...)` calls from the forward path left strict
  validity intact but did not improve decode: strict `60.678`, `c1_2000`
  `61.289`, and `c1_10000` `57.896` backend TPS.
- Relaxing the release MLP SwiGLU guard for explicit F16 GGUF testing enabled
  the native extension across the dense stack and produced coherent output, but
  decode stayed in the same slow band: strict `61.053`, `c1_2000` `61.780`,
  and `c1_10000` `58.266` backend TPS.
- Both the traced and no-trace release-attention runs logged the Transformers
  FLA / causal-conv warning, but later source inspection localized it to GGUF
  loader dummy-model construction. Request-window profiling is required before
  treating fast-path package availability as a serving hot-path issue.
- A fused-GDN GGUF parity bundle routed qkv/z/b/a into one release-shaped
  `in_proj_qkvz` projection and loaded with layer-0 per-rank shape
  `(2060, 5120)`. The first short-timeout smoke looked hung, but a patient rerun
  served direct completion, passed short-chat coherence, and completed the full
  ladder: strict `63.201`, `c1_2000` `63.836`, and `c1_10000` `60.169` backend
  TPS.
- Combining the fused-GDN path with native GGUF SwiGLU completed the same
  normal ladder and improved the current best to strict `63.679`, `c1_2000`
  `64.353`, and `c1_10000` `60.617` backend TPS.
- Repeating the ConditionalGeneration override after those improvements stayed
  coherent and strict-valid but landed lower: strict `63.003`, `c1_2000`
  `63.656`, and `c1_10000` `60.109` backend TPS.
- The `transformers` fast-path warning in GGUF logs was traced to vLLM's GGUF
  loader creating a dummy Transformers model for name mapping, not to proven
  serving hot-path execution.
- A disposable `GGUFLinearMethod.apply()` diagnostic sampled graph profiling and
  one short coherent decode request. The recorded calls were `ParallelLMHead`;
  sampled internal Qwen block linears did not appear as `GGUFLinearMethod.apply`
  callers.
- A narrow lm-head-only patch forced `lm_head` to use
  `UnquantizedEmbeddingMethod`. The marker fired on all 8 TP ranks and strict
  remained valid, but the full ladder stayed in the prior slow band: strict
  `60.676`, `c1_2000` `61.309`, and `c1_10000` `57.888` backend TPS.

Disposition:

- Reject attention-class routing as the missing performance lever.
- Reject the assumed GGUF `qweight` shard-matmul overhead for the main dense
  linears.
- Reject ConditionalGeneration as a dense GGUF performance route.
- Reject disabled trace-call overhead as the missing dense GGUF performance
  lever.
- Reject native MLP SwiGLU enablement as the missing standalone dense GGUF
  performance lever.
- Promote fused-GDN plus native GGUF SwiGLU as the best dense GGUF source
  baseline so far, but reject it as a benchmark-promotion candidate because it
  remains below the FP16/HF release band.
- Reject broad internal GGUF matmul overhead as the current dense-gap
  explanation unless a deeper trace contradicts the sampled hot-path diagnostic.
- Reject isolated lm-head method selection as the missing dense GGUF performance
  lever.

Reason:

- All three paths leave decode in the same slow band or make prefill worse.
- The no-trace run leaves decode in the same slow band as the traced run.
- The GGUF-SwiGLU run proves the native extension was disabled by GGUF
  `quant_config`, but the remaining performance gap persists after it is
  enabled.
- The fused-GDN rerun proves the extra split `in_proj_ba` projection was a real
  overhead component, but removing it recovers only part of the missing TPS.
- The initial fused-GDN "hang" classification was too strong; the path is
  serving-capable with a longer patience window and coherent smoke validation.
- The combined fused-GDN plus native GGUF SwiGLU run proves those two levers are
  additive, but together they still recover only part of the missing dense TPS.
- The modern ConditionalGeneration rerun proves the remaining gap is not solved
  by matching the HF-control architecture wrapper after fused-GDN and native
  SwiGLU are already active.
- The HF-control and best-GGUF logs both use TP8, P2P-on,
  `MAX_MODEL_LEN=131072`, chunked prefill, graph-capture mode, capture sizes
  `[1, 2, 4]`, and about 2 seconds of graph capture. The fallback messages
  differ in source and timing, so raw fallback count is too weak without
  hot-path timing.
- The rejected timing-hook run shows that source timing must not use ordinary
  Python wall-clock calls inside the compiled forward path. A comparable
  diagnostic needs external ROCm profiling, compiler-safe C++/custom-op
  markers, or built-in tracing that does not break Dynamo.
- `rocprof` is available on `.20` and inside the ROCm7.2 release image, so the
  next practical timing route is an external ROCm trace/counter run rather than
  another Python-source timing hook.
- A first `rocprof` wrapper around the API server entrypoint did not produce
  useful hot-path evidence. It captured startup and parent-process HIP API
  activity, but no request-level decode kernel window and no kernel-summary CSV.
  Future ROCm profiling needs to target a request window, worker process, or
  single-process/offline invocation instead of wrapping the whole API server.
- Later `rocprofv3 --attach` request-window attempts against the TP0 worker
  also failed to emit files even with explicit CSV/JSON formats and with the
  missing `libdw.so.1` supplied to the container. This is useful negative
  evidence about the PID-attach profiler shape, not about the model path. The
  next profiler run should first prove artifact emission on the intended
  process shape before burning another full request window.
- A launch-time `rocprofv3` wrapper can profile tiny `spawn` child GPU work, but
  wrapping the full vLLM/Persistent-AR server destabilized the lane with GPU
  memory access faults before health. This rejects the unchanged full-server
  wrapper shape for release-like profiling.
- The Persistent-AR-disabled `rocprofv3` wrapper is also rejected as a
  promotion or profiler route. It changes the release-like lane, reaches only a
  very slow capped request, misses the request window, and still does not emit
  useful artifacts.
- Marker instrumentation needs to move lower level before the next model run.
  The useful control is that ROCTX output works; the rejected part is Python
  marker insertion inside the compiled Qwen path. A viable follow-up should use
  a proper lower-level custom op with correct alias/side-effect metadata or a
  reduced offline reproducer where markers do not perturb the full vLLM AOT
  graph.
- The GGUF apply diagnostic shows the sampled serving path is not dominated by
  internal block linears calling `GGUFLinearMethod.apply`; the visible GGUF
  application point was the lm-head path.
- Forcing the visible lm-head path to the unquantized embedding method did not
  move strict or fixed-token backend TPS, so the visible apply point is not
  enough to explain the `8` to `9` TPS dense gap.
- The fast-path warning alone is not enough evidence to blame serving-time
  Transformers fallback because it can be emitted by GGUF loader dummy-model
  construction before serving starts.
- More launch-shape churn is unlikely to close the published FP16 gap without a
  source-level explanation.

Next:

- Use profiling and source comparison before starting another full benchmark
  ladder.
- Compare lm-head, GatedDeltaNet/core, graph-capture, sampler, and
  GGUF adapter costs against the clean HF-weight release path, using fused-GDN
  plus native GGUF SwiGLU as the dense GGUF baseline.
- Start with a narrow source timing hook around the existing fused-GDN forward
  path: projection, `torch.ops.vllm.gdn_attention_core`, output projection, and
  logits/sampler boundaries. Both HF and GGUF reach the same GDN op name, so the
  next useful evidence is boundary timing, not another launch-shape change.
- Implement that timing with `rocprof`/external ROCm profiling or a
  compiler-safe lower-level marker; do not put `time.perf_counter()` or
  `torch.cuda.synchronize()` directly in the compiled Python forward. Do not
  repeat broad API-server-wrapper profiling or the fileless `rocprofv3
  --attach` PID shape. Use a request-window trace, profile input file,
  worker-targeted run, or comparable single-process/offline invocation that
  first demonstrates it captures decode kernels and writes artifacts. If using
  `rocprofv3` inside the release image, handle the missing `libdw.so.1`
  dependency without mutating the image. Do not wrap the full Persistent-AR
  server unchanged under `rocprofv3`; the current evidence shows that shape can
  fault the workers before readiness. Do not use a Persistent-AR-disabled
  wrapper as a substitute for release-like profiling; it is non-comparable and
  still fileless in the tested stop path. Do not use raw Python NVTX calls,
  built-in `_record_function_enter/_exit` hooks, or ad hoc Python custom marker
  ops inside the compiled Qwen forward; all have now failed the model path or
  failed to emit marker records.
- Do not relaunch another architecture-wrapper test until profiling identifies
  a source-level reason; the modern ConditionalGeneration rerun already rejected
  wrapper matching as the missing dense GGUF performance lever.
- Do not run another residual pre-fold ladder just because the GGUF
  `quant_config` guard differs from HF. A controlled F16 GGUF candidate with
  the guard relaxed completed the full ladder and was slightly slower than the
  current best GGUF baseline.
- Do not mutate the release image to install optional FLA or causal-conv1d
  packages as a first move. The stronger test is request-window profiling of
  the current GGUF baseline versus HF control, because the observed warning is
  loader-boundary evidence unless a profile proves otherwise.
- Keep MoE GGUF blocked from throughput testing until the experimental
  `qwen35moe` / Qwen3.5-MoE route returns coherent deterministic probes. Do not
  run warmups or capped requests on the slow eager fallback, the older alias
  path, or the current graph-mode forced-unquantized path.

The latest MoE graph-mode source pass changes the startup boundary, not the
promotion boundary:

- Prefix-based skipped-layer selection did not select the unquantized MoE method
  for the GGUF expert layers; the slow fallback warning remained.
- An explicit diagnostic switch to force unquantized FusedMoE moved TP4 MoE GGUF
  past the stream-capture failure and into graph-mode API health.
- With M-RoPE stripped, a one-token `Hello` probe returned the unrelated token
  `amed`, while a 16-token math prompt returned multilingual/glyph garbage after
  a long delay.
- Restoring the text M-RoPE fields with a minimal text-only M-RoPE position
  implementation still produced `amed` for `Hello`, and the 16-token math prompt
  timed out after `180` seconds with no response bytes.
- Offline tensor comparison then found a real manual-bridge bug:
  `blk.0.ssm_dt.bias` is not HF `linear_attn.dt_bias` in direct order. It is
  exactly HF `dt_bias` packed even-then-odd. Inverting that permutation before
  loading restored the tensor match, but the same graph-mode TP4 probe still
  returned `amed` for `Hello` and timed out on the 16-token math prompt.
- A follow-up parser check found another real tensor-processing issue:
  Qwen3.5-MoE GGUF stores most RMSNorm-family tensors as HF `+ 1.0`, while
  `ssm_norm` tensors are already direct HF layout. Registering a `qwen35moe`
  processor that subtracts `1.0` only from non-SSM norm weights restored the
  sampled norm tensors to HF layout. The same pass also rejected the simple
  expert TP-shard theory: simulated TP4 `w13` and `w2` shards for all four
  ranks matched HF within FP32 comparison noise. The graph-mode TP4 server
  still returned `_` for one-token `Hello`, and the 16-token math prompt timed
  out after `180` seconds with zero response bytes.
- Adding the `IsHybrid` marker to the text-only Qwen3.5-MoE CausalLM base is a
  real integration fix but not a coherence fix. Before the patch, the saved
  model info reported `has_inner_state=true` and `is_hybrid=false`; after the
  patch, startup ran hybrid cache verification and configured a `528`-token
  attention block size with matched mamba page sizing. The server still returned
  `_` for one-token `Hello`, and the 16-token math prompt timed out after
  `180` seconds with zero response bytes. Keep this as a source-state
  correction, but do not treat it as a benchmark candidate.
- `.10`, `.20`, and `.30` are the same EPYC / 8x MI50 32GB class. The published
  benchmark host remains `.20` unless a run explicitly records another lane. Use
  the InfiniBand address family for future host-to-host and benchmark-lane work.
  The `.20` GGUF workspace is `<validation-workspace>` on NVMe;
  `LLM_STORE_VOL` is a `.40`-only source-debug workspace and must not be created
  or used on `.10`-`.30`. The separate `.40` 8x MI60 lane is useful only for
  source debugging and must not be used as benchmark reproduction evidence.
- The MoE F16 GGUF artifact and Qwen3.5-MoE experimental patch bundle are now
  staged on `.20` NVMe. The staged MoE GGUF hash matched the `.40` source copy:
  `1f2443bb0ff958943d091410c61120c181a0579b3bc85192029aa51d821d141c`.
- A final-logit trace hook fired during vLLM's internal compile/profile path,
  not during the client prompt. The saved tensors are useful only as a rejected
  trace-placement diagnostic; request-level coherence still requires a gated
  first-token trace or a reduced HF/llama.cpp comparison.
- A request-gated final-logit trace bundle is now staged on `.20` NVMe. The
  hook waits for `VLLM_GFX906_GGUF_TRACE_GATE_FILE` before writing, so future
  traces should target the actual client request rather than vLLM startup
  profiling. This is diagnostic infrastructure only.
- The `.20` official v0.2 image can now parse and weight-load the staged
  Qwen3.5-MoE F16 GGUF artifact when the source files are mounted directly.
  The original graph-mode route failed before API health in hybrid KV-cache
  initialization because the cache tensor shape
  `torch.Size([2, 2, 528, 1, 256])` was ambiguous between leading KV and
  leading block layout.
- Explicit `--enable-prefix-caching --mamba-cache-mode align` makes the release
  image report `enable_prefix_caching=True` and `mamba_cache_mode='align'`, but
  it does not fix the same hybrid cache layout assertion. Prefix-cache
  alignment alone is rejected as a MoE GGUF startup fix.
- Eager mode on the `.20` official image reaches API health with the same
  direct-mounted MoE GGUF route and fires the request-gated final-logit trace on
  a real one-token `Hello` request. The visible output is still `_`; all four TP
  workers agree on the same top-token sequence headed by token id `62`. This
  rejects eager mode as a correctness fix and promotes the trace as request-level
  evidence that corruption is upstream of sampling.
- A ROCm llama.cpp control using the same `.20` F16 MoE GGUF artifact gives a
  coherent raw-completion first token for `Hello`: comma, token id `11`.
  Therefore the current vLLM result is not an artifact/tokenizer failure.
- The watched vLLM final-logit trace now makes the divergence exact: all four TP
  workers rank underscore token id `62` first with logit `12.867188`, while the
  expected comma token id `11` ranks only `25` with logit `8.882812`. This is a
  material final-hidden/logit boundary error before sampling, not a near-tie or
  sampler issue.
- Before the runtime norm-offset fix, TP8 did not repair the MoE GGUF vLLM
  path. With the same watch-id trace on all eight GPUs, vLLM returned `_`,
  ranked underscore token id `62` first with logit `13.000000`, and ranked
  comma id `11` only `20` with logit `9.000000`. This rejected a TP4-only
  sharding explanation.
- A clean HF-weight vLLM TP4 control on the same `.20` MI50 lane, official
  v0.2 image, P2P-on, eager mode, and raw one-token `Hello` probe returns the
  expected comma token. It ranks comma id `11` first with logit `15.039062`,
  while underscore id `62` ranks `369` with logit `5.906250`. This rejects the
  shared vLLM Qwen3.5-MoE execution path, sampler, and raw prompt as the main
  explanation for the GGUF failure. The remaining MoE blocker is specific to
  GGUF materialization/state formation before final logits.
- Boundary tracing moved the first measured divergence before the GatedDeltaNet
  core. The layer-0 GatedDeltaNet input norm is nearly 2x in GGUF versus HF:
  GGUF `91.739395` and HF `46.638695`. The same scale difference appears in
  `mixed_qkvz` (`220.456894` versus `113.987366`) and `z_pre_reshape`
  (`107.963158` versus `55.807575`). Final hidden remains divergent at GGUF
  `142.269989` versus HF `114.722717`.
- A narrow edit changing the Qwen35-MoE non-SSM norm processor from
  `weights - 1` to `weights - 2` did not change the response or trace, even
  though the live container showed the edited processor line. That rejects
  blind processor-line edits as the next move and requires tracing the actual
  decoder-layer norm input, norm output, and loaded norm parameter.
- The decoder-layer trace showed the exact scale bug. GGUF and HF enter layer 0
  with the same pre-layernorm embedding norm `0.610113`, but GGUF loaded
  `input_layernorm_weight` with mean `1.031100` while HF loaded the offset
  parameter with mean `0.031111`. Applying a runtime GGUF-only correction before
  vLLM's `1 + weight` rule changed GGUF post-input-layernorm from `91.739395`
  to `46.637737`, matching HF `46.638695`, and final hidden from `142.269989`
  to `114.727249`, matching HF `114.722717`. Raw `Hello` changed from `_` to
  comma, and a short sanity prompt produced coherent text.
- Therefore M-RoPE omission, direct `dt_bias` order, non-SSM norm offsets, and
  the hybrid marker were all real source issues. The latest live-serving
  coherence repair is the Qwen3.5-MoE GGUF runtime norm-offset correction. Do
  not promote throughput from this finding alone. The first clean 131K
  benchmark overlay rejected eager mode: graph mode initially failed before
  health, and eager mode was coherent only for tiny probes while the benchmark
  warmups decoded at `12.455`-`14.060` backend TPS and strict did not close. An
  env-gated graph-mode KV-layout bypass then got the same 131K path healthy and
  raised warmups to `73.579`-`73.655` backend TPS, but strict still did not
  close and was cut off after the live metric estimate crossed `60071`
  generated tokens. A shortest-prompt follow-up rules out prompt length as the
  primary explanation: GGUF with `prompt_repeat=0` still timed out after an
  estimated `18019` generated tokens for the active strict request, while a
  clean HF-weight TP4 eager comparator on the same `.20` release lane closed
  the same shortest prompt after `2957` generated tokens with
  `finish_reason=stop`, `qwen_gate_valid=true`, `10125` reasoning chars, and
  `3930` visible answer chars. The remaining source target is the GGUF
  reasoning-to-answer delimiter/stop transition under benchmark-task prefill,
  not another full ladder or prompt-length sweep. A direct vLLM
  thinking-disabled diagnostic confirms the boundary: default thinking-enabled
  GGUF at `max_tokens=128` produced `621` reasoning chars and `0` content
  chars, while `chat_template_kwargs.enable_thinking=false` produced visible
  content immediately (`700` content chars at `128` tokens and `2158` content
  chars at `512` tokens). The 512-token tail became repetitive, so this is not
  a benchmark path; it only proves the model can enter content mode when the
  Qwen3 parser is told the prompt already ended reasoning. Tokenizer inspection
  pins `<think>` to id `248068` and `</think>` to id `248069`.

  A request-gated `</think>` watch-token trace now rejects parser-only and
  prompt-length explanations more directly. The TP4 graph-mode GGUF path with
  the norm-offset fix and KV-layout bypass reached health on `.20`, P2P-on, and
  the shortest benchmark prompt, but the capped trace hit `finish_reason=length`
  at `4096` completion tokens with `20218` reasoning chars and `0` content
  chars. The best sampled rank for `</think>` token id `248069` was only `198`.
  The HF-weight TP4 control on the same prompt closed after `2957` completion
  tokens with `finish_reason=stop`, and sampled `</think>` at rank `6`. The
  active blocker is therefore a GGUF model/logit-path failure to make the
  reasoning-end delimiter competitive under the thinking-enabled benchmark
  prompt, not a reason to rerun warmups or fixed-token tiers.

  The exact-prefix follow-up refined that conclusion. A stride-1 HF trace found
  `</think>` rank `3` at generated index `35`. Using the exact chat prompt
  tokens plus the first `35` HF-generated token IDs as a token-list prompt,
  GGUF also ranked `</think>` at `3`. The problem is therefore not that GGUF
  can never score the delimiter. The GGUF self trajectory diverges from HF at
  generated token index `1`: HF begins `<|im_end|>Thinking Process:`, while
  GGUF begins `<|im_end|>Here's a thinking process:`. That trajectory remains
  reasoning-only under the benchmark prompt and fails to recover into a close.

  A faithful chat-endpoint first-divergence probe then made the branch precise.
  The exact token-list completion endpoint was rejected as a proxy because it
  did not reproduce the HF chat branch after forcing the generated special
  token. Under the real chat decode path, both HF and GGUF first select token
  `248046`; on the next step HF ranks `Thinking` (`90700`) first and `Here`
  (`8160`) second, while GGUF flips the same pair. HF only prefers `Thinking`
  by `0.343750` logit, but GGUF prefers `Here` by `3.281250` logit. Hidden
  norms at that branch are close (`108.96` HF versus `108.02` GGUF), so this is
  not another gross norm-scale failure.

  The first branch-row diagnostic attempted to read `lm_head` shard rows and
  row dots from inside live `compute_logits()`. It is rejected as an
  instrumentation shape: the HF TP4 chat request failed with worker shutdown
  after one prefill/profile event, and only partial shard evidence was written.
  The useful evidence is limited to the fact that one TP shard reported
  `ParallelLMHead` local rows `124160` through `186240`, with the watched
  branch tokens remote on that shard. Do not repeat live `lm_head` row access in
  the serving path unchanged. The next safer path is minimal hidden/logit
  capture followed by offline token-row/materialization comparison, or a reduced
  single-process row diagnostic.

  The hidden-vector follow-up points upstream of the final lm-head. It used the
  same faithful two-token chat request and wrote only the detached hidden row
  already used by the safe trace hook. The branch after shared token `248046`
  reproduced: HF ranked `Thinking` id `90700` first and `Here` id `8160`
  second, while GGUF flipped them. Hidden norms were still close
  (`108.961647` HF versus `108.671486` GGUF), but the vectors were not close
  enough to treat the issue as a final-row lookup: cosine `0.890908`, centered
  correlation `0.890758`, L2 diff `50.829083`, mean absolute diff `0.878714`,
  and max component diff `5.335938`. The active source target is now the
  late-layer hidden stream before logits: output norm/gate, layer-39
  MoE/router/expert output, GatedDeltaNet/Mamba state carry, and residual
  stream around the first post-special-token decode step.

  The call-indexed decoder-boundary trace localizes that first aligned drift
  further. With the same two-token chat request, HF returned `Thinking Process`
  and GGUF still returned `finish_reason=length` with no visible or reasoning
  content. The first generated token remained shared at `248046`; on the next
  call HF preferred `Thinking` id `90700` over `Here` id `8160` by `0.343750`,
  while GGUF preferred `Here` over `Thinking` by `3.515625`. Call 0 layer-0
  input rows were identical, and call 1 post-input-layernorm was still
  effectively identical. The first useful aligned drift appears after layer-0
  attention and MLP on call 1: `layers.0.call1.post_attention` cosine
  `0.900811`, `layers.0.call1.post_mlp` cosine `0.879649`,
  `layers.39.call1.post_mlp` cosine `0.955899`, and
  `compute_hidden.call1` cosine `0.890804`. Call 2 is no longer comparable
  because the generated tokens have already diverged. The next source target is
  therefore layer-0 attention/GatedDeltaNet/Mamba state formation and residual
  add order on call 1, not another benchmark warmup or fixed-token run.

  The all-head GatedDeltaNet follow-up narrows the call-1 boundary further.
  HF and GGUF still flip the same `Thinking` / `Here` branch, but the z gate
  itself is not the culprit: sampled `z_for_norm` rank pairs have cosine
  `1.000000`. Drift is already present in `core_attn_out_raw`, where sampled
  call-1 rank-pair cosines range from `0.837480` to `0.988653`, and it becomes
  sharper after normalization/gating, with sampled `core_attn_out_normed`
  cosines from `0.539105` to `0.998626`. Separate static checks showed that the
  inverse GGUF transforms for V rows, Z rows, output-projection columns,
  `dt_bias`, `A_log`, `ssm_norm`, and conv V channels can exactly match HF
  within FP32 noise. Offline reconstruction with the HF out-projection weight
  also reproduced the traced HF and GGUF projection outputs. The current source
  target is therefore runtime `gdn_attention_core` input/state/cache behavior
  and runtime-loaded GDN parameters on call 1, not Kausik-style registry aliases,
  tuple-shard loading, tokenizer metadata, z gating, or out-projection mapping.

For MoE GGUF, preserve these boundaries before relaunching a full server:

- router logits and selected experts versus an HF or llama.cpp control;
- decoder-layer input before `input_layernorm`, output after
  `input_layernorm`, and loaded `input_layernorm.weight` values;
- the GGUF-only runtime norm-offset correction for vLLM `GemmaRMSNorm`;
- expert gate/up/down tensor orientation and TP sharding;
- `linear_attn.dt_bias` / GGUF `ssm_dt.bias` mapping;
- non-SSM norm offset handling versus direct `ssm_norm` handling;
- Mamba/GatedDeltaNet state-copy semantics;
- class-level hybrid model registration and cache sizing, preserving the
  `IsHybrid` marker while investigating the remaining semantic break;
- final logits before sampling.
- request-level top-token traces from the eager `.20` run, starting with the
  shared top token id `62`, against expected comma token id `11` from the
  coherent llama.cpp and HF-weight vLLM controls.
- the 131K norm-fix benchmark-path rejection: graph mode originally hit the
  hybrid KV-cache layout assertion for shape `torch.Size([2, 2, 528, 1, 256])`,
  while eager mode was far below target and did not close the benchmark gate.
  The env-gated graph KV-layout bypass gets graph mode healthy and improves
  warmup decode to about `73.6` backend TPS, but it also fails the benchmark
  gate because the strict request exceeded `60071` generated tokens without a
  close.
- the shortest-prompt comparator: GGUF with `prompt_repeat=0` still timed out
  after an estimated `18019` generated tokens, while the HF-weight comparator
  closed the same shortest prompt after `2957` generated tokens with
  `qwen_gate_valid=true`.
- the thinking-disabled diagnostic: `enable_thinking=false` routes GGUF output
  to visible content immediately, proving the active failure is the
  thinking-enabled `</think>` / content-transition path, not a total inability
  to answer. The follow-up watch-token trace showed `</think>` token id
  `248069` only reached sampled rank `198` in GGUF, while the HF-weight control
  sampled it at rank `6`.
- the exact HF-prefix delimiter score: when the HF trajectory prefix is forced
  into GGUF as token IDs, `</think>` also reaches rank `3`. The active source
  target is early trajectory divergence and later loop recovery, not a blanket
  inability to score the delimiter token.
- the faithful chat first-divergence rank flip: after the shared first special
  token, HF ranks `Thinking` id `90700` above `Here` id `8160`, while GGUF ranks
  `Here` above `Thinking` with a much larger margin. Token-list completion
  prompts do not faithfully reproduce this hybrid chat decode state.
- the runtime `A_log` loader fault and its partial repair. Request-gated
  GatedDeltaNet state tracing showed that the served MoE GGUF path matched HF
  for runtime `dt_bias`, conv weights, `ssm_norm`, mixed q/k/v, beta, and
  initial state, but loaded `A_log` as a rank-constant value instead of the
  rank-varying HF tensor. Moving conversion before the equal-head helper return
  did not affect the served path. The active fix was explicit
  `blk.*.ssm_a` to `model.layers.*.linear_attn.A_log` mapping plus a top-level
  `Qwen3_5ForCausalLMBase.load_weights` wrapper that applies
  `log(-GGUF_A)` before `AutoWeightsLoader`. With that repair, layer-0 call-1
  matches HF through GatedDeltaNet runtime params, recurrent core,
  post-attention, and post-MLP.
- the remaining branch flip after the `A_log` repair localized to the first
  full-attention layer, not to the next GatedDeltaNet layer. A layers `0-8`
  bisection showed layers 0-2 remained effectively aligned, while layer 3
  call-1 drifted immediately after full attention: `post_attention` cosine
  `0.859239` and `post_mlp` cosine `0.750201`. Layer 3 is the first
  `full_attention` layer in the Qwen3.6 35B-A3B layer schedule.
- the full-attention drift was repaired by making `self_attn.q_norm` and
  `self_attn.k_norm` use the same GGUF-aware `Qwen3_5RMSNorm` runtime offset
  behavior as the decoder input/post-attention norms. The imported
  `Qwen3NextAttention` class instantiated plain `Qwen3NextRMSNorm`, so the
  active GGUF runtime norm correction did not cover q/k norms. With the q/k norm
  repair, the same two-token branch probe returns `Thinking Process`, layers
  0-8 match HF nearly exactly, layer-3 `post_attention` improves to cosine
  `0.999999`, and event-1 logits rank `Thinking` id `90700` first and `Here`
  id `8160` second. This is a source fix candidate, not a benchmark claim yet.
- the `Kausik-A/qwen3.6-27b-mi50-vllm` repo is useful source context but not a
  direct fix for the active MoE path. It carries dense GGUF compatibility
  pieces such as Qwen3.5 aliases, text-only M-RoPE suppression, `ssm_dt.bias`
  mapping, conv1d reshaping, text-only `IsHybrid` hooks, tuple-shard QKVZ
  loading, and MiniMax M2 GGUF aliases. It does not map MoE `ssm_a` / `A_log`
  and its top-level MoE CausalLM load path still uses plain
  `AutoWeightsLoader`, which is exactly where the served MoE `A_log` transform
  was bypassed locally.
- request-window timing around the registered `gdn_attention_core` body rejects
  the steady GatedDeltaNet core as the current MoE GGUF throughput bottleneck.
  The synchronized timing wrapper invalidated HIP graph capture and is rejected
  for graph-mode serving. The no-sync wrapper reached health for both the GGUF
  TP4 graph path and an HF-weight comparator using the same hybrid KV-layout
  bypass. Both paths returned the faithful two-token branch response
  `Thinking Process` and added `240` timing records. Excluding the shared
  layer-0 first-request/prefill outlier, GGUF averaged `1,039,559` ns across
  per-layer average `last_ns` values and HF averaged `1,049,320` ns; max
  non-layer-0 `last_ns` was `1,344,764` for GGUF and `1,376,768` for HF.
  The next timing target should move outside the GDN core into projections,
  MoE/shared-expert execution, residual/norm, graph/kernel selection, logits,
  and whole decode-loop scheduling.
- request-window timing around the registered FusedMoE direct custom-op body
  rejects the steady `DefaultMoERunner.forward_impl(...)` path as the primary
  current MoE GGUF throughput bottleneck. In graph mode, the GGUF Python timing
  body ran during profiling/capture but not during live graph replay, so that
  hook is not sufficient for graph replay timing. In eager request-window
  comparators, GGUF and HF both returned `Thinking Process` and each added
  `480` timing rows. GGUF was effectively at parity on `(1, 2048)` and
  `(32, 2048)` shapes: `645,803` ns versus `649,962` ns and `678,506` ns
  versus `694,144` ns. The apparent `(15, 2048)` gap was a layer-0
  first-request outlier; excluding layer 0, GGUF averaged `808,102` ns and HF
  averaged `801,966` ns. Do not spend the next pass blindly optimizing the
  unquantized FusedMoE method without lower-level graph-visible evidence.
- post-forward logits and sampler timing also rejects the visible
  logits/sampling path as the missing MoE GGUF throughput lever. With
  synchronized env-gated timing on the faithful two-token branch request, GGUF
  and HF were effectively at parity: `lm_head_apply` `393,201` ns versus
  `373,102` ns, TP logits gather `236,504` ns versus `230,403` ns,
  `get_logits_total` `735,715` ns versus `727,748` ns, sampler forward
  `250,274` ns versus `232,938` ns, and inner sample `76,387` ns versus
  `79,262` ns. These sub-millisecond differences do not explain the fixed-token
  MoE TP4 TPS gap.
- whole-runner timing around vLLM v1 `GPUModelRunner.execute_model` is now also
  a rejected source route. The synchronized timing variant changed the faithful
  branch output and must not be used for promotion evidence. The no-sync variant
  preserved `Thinking Process`, showed preprocessing and postprocessing were
  small, and put the broad graph-capture HF comparator slower than GGUF in the
  short request window (`model_forward` HF/GGUF ratios `1.107` at tokens `15`
  and `1.132` at tokens `32`). Because no-sync timing is not a synchronized GPU
  duration, it should be treated as a boundary only: Python runner envelope
  timing does not explain the GGUF fixed-token gap. Move the next pass to
  graph-visible whole-decode profiling, ROCm/HIP kernel traces, or longer
  fixed-token request-window profiling.
- naive `rocprofv3` request-window profiling is rejected for this vLLM
  multiprocess server shape. PID attach reported successful injection into the
  four TP workers during a normal c1_2000 request at `72.740` backend decode
  TPS but emitted no usable files. Parent-process delayed wrapping needed a
  read-only `libdw.so.1` diagnostic mount, then either heavily perturbed c1_2000
  decode to `32.892` backend TPS or hung waiting for child workers after SIGINT
  on a c1_512 run that otherwise decoded at `73.019` backend TPS. Do not repeat
  unchanged parent-wrapper or worker-attach `rocprofv3` routes. The next
  profiler attempt needs worker-entrypoint awareness, explicit pause/resume
  hooks, lower-overhead counters, or graph-safe markers that survive replay.
- full `Qwen3NextSparseMoeBlock.forward` timing is also rejected as the primary
  MoE GGUF throughput gap. A graph-mode timer at that level failed during
  engine initialization because Dynamo cannot trace `time.perf_counter_ns()`
  inside the compiled MoE block. Eager 4K comparators on `.20` preserved the
  faithful `Thinking Process` branch for both GGUF and HF and showed parity at
  the stable `(32, 2048)` decode shape: total block time averaged `2,486,178`
  ns for GGUF and `2,479,019` ns for HF; the internal-router expert stage
  averaged `2,241,665` ns for GGUF and `2,239,985` ns for HF. Shared add and
  TP all-reduce were small. Do not repeat unchanged Python-level MoE-block
  timing; the next source work needs graph-visible replay, kernel-level
  evidence, CausalLM-versus-HF graph structure, or a longer fixed-token
  decode-window profiler.

## Promotion Standard For GGUF

GGUF can move from active source investigation to benchmark candidate only
after all of the following pass:

1. TP4 first-token sanity matches TP1 on simple deterministic prompts.
2. TP8 first-token sanity is coherent.
3. The normal benchmark warmup sequence runs without correctness failures.
4. The `c1_128` uncapped strict prompt passes the same gate logic used for the
   FP16 release path.
5. `c1_2000` and `c1_10000` complete with stable backend metrics.
6. Results are compared against the FP16 v0.2.1 reproduction path without
   mixing correctness-only, fixed-token, and strict-valid claims.

Dense FP16 GGUF now passes the correctness portions of this gate, including
strict validity and fixed-token completion, but fails the final promotion
requirement because backend TPS is below the FP16 v0.2.1 reproduction path.
MoE GGUF has an artifact, tokenizer metadata, and an experimental text route
that reaches API health. The runtime norm-offset correction made the raw
one-token probe and a short sanity prompt coherent. The env-gated graph
KV-layout bypass made the 131K TP4 path healthy, but the first graph ladder
still failed to close. Later source work repaired the trajectory bug by
combining the top-level `A_log` loader fix with the full-attention q/k norm
runtime-offset fix. The resulting clean non-tracing 131K graph-mode ladder is
strict-valid and completes the fixed-token tiers: strict `71.916`, `c1_2000`
`73.380`, and `c1_10000` `65.749` backend TPS. It is therefore promoted as a
MoE GGUF correctness candidate and rejected as a performance candidate because
it remains far below the FP16 v0.2.1 TP4 reproduction path. The next required
gate is not another identical ladder. The trace-gate-off rerun landed at
strict `69.961`, `c1_2000` `73.228`, and `c1_10000` `65.643`, so disabled
trace overhead is rejected as the missing performance path. The next source
change must move GGUF MoE execution speed toward the HF-weight fast path. A
default-scheduler rerun restored HF-like graph capture and landed in the same
fixed-token band (`73.130` / `65.552`), which rejects scheduler sizing and broad
graph capture as the missing MoE GGUF throughput lever. A c1 topk8 fastpath
diagnostic then showed the release image does not include that overlay at all;
forcing the env var produced no active/reject logs, so topk8 fastpath absence is
image-boundary evidence rather than the specific GGUF performance root cause.
The FLA / causal-conv warning was also localized to GGUF loader dummy-model
construction, so the next source work should profile or instrument the actual
serving path rather than treating that warning as a proven hot-path fallback.
The global GGUF quantization audit also shows the current MoE overlay already
forces unquantized FusedMoE for F16-class expert weights, so the next test should
not be another quant-method selection rerun.

## 2026-06-27 Norm-Offset Graph Parity Update

The MoE GGUF runtime norm-offset fix is both necessary and structurally visible
in the compiled graph. Comparing graph caches showed the current GGUF graph
contains runtime `_qwen35_effective_weight` subtraction sites, while the
HF-weight graph does not. The GGUF runtime-offset cache was larger
(`9336` lines and about `32M`) than the HF cache (`9145` lines and about
`23M`).

That makes runtime norm-offset removal a legitimate performance hypothesis, but
the first two load-time attempts did not preserve Qwen behavior:

- Disabling the runtime offset without a load-time replacement produced
  incoherent multilingual/glyph smoke output.
- A load-time-only non-SSM norm offset reached the warmup path at about the
  same `73` backend TPS band, but failed `c1_128` strict immediately with one
  completion token, `qwen_gate_valid=false`, and no valid thinking close or
  parser answer.
- Adding full-attention q/k norm class replacement to the load-time offset
  path shrank the graph versus the runtime-offset path, but still returned an
  empty one-token smoke response.

Conclusion: the runtime norm-offset correction remains the active MoE GGUF
correctness baseline. Moving the correction to load time is still a promising
graph-parity target, but it needs a tensor-level audit first. The next source
pass should compare representative loaded norm tensors across runtime-offset
GGUF, load-time GGUF, and HF controls before another serving launch.

The first raw tensor audit is now complete. It covered all `131` mapped
Qwen3.6 35B-A3B MoE norm tensors in the staged F16 GGUF and local HF snapshot.
It confirms the file-level rule exactly: non-SSM norms are stored as
`HF + 1.0`, while `ssm_norm` tensors already match HF directly. The current
load-time hook's name rule shifts every mapped non-SSM norm and no mapped SSM
norm. Therefore the rejected load-time variants did not fail because the raw
GGUF/HF norm convention was misunderstood.

Next target: live module materialization. The next audit should run after
vLLM weight loading and compare actual module parameter stats for
runtime-offset GGUF, load-time GGUF, and the HF control. Do not run another
serving ladder until those live module stats explain why the mathematically
correct load-time file transform still loses Qwen strict/parser behavior.

The live module audit found the exact load-time failure. The original
load-time hook was applied twice in the vLLM path and also shifted the mapped
SSM name `linear_attn.norm.weight`, so live params moved by about `-2.0` from
the runtime baseline. A corrected overlay that applies the hook once and
excludes `linear_attn.norm.weight` restored live module stats to HF convention:
non-SSM norms shifted by `-1.0`, SSM unchanged.

That corrected path is strict-valid, but it is not a performance promotion.
It completed the normal MoE TP4 GGUF ladder at strict `71.995`, `c1_2000`
`73.005`, and `c1_10000` `65.456` backend TPS. This is in the same band as the
current MoE GGUF baseline and far below the FP16 v0.2.1 TP4 path. Runtime
norm-offset graph subtraction is therefore not the missing TPS lever. Keep the
corrected load-time rule as a source-cleanup candidate, but move performance
work to graph-visible replay, CausalLM-versus-HF graph structure, model-runner
decode scheduling, or fixed-token kernel profiling.

The corrected load-time path was then rerun with the broad default scheduler
instead of the restrictive `--max-num-seqs 1 --max-num-batched-tokens 1024`
shape. It reached the intended broad graph capture (`PIECEWISE=51`, `FULL=35`,
cache `686a95799f`) and stayed strict-valid. The normal ladder produced strict
`71.785`, `c1_2000` `73.396`, and `c1_10000` `65.796` backend TPS. This makes
the broad corrected load-time path the cleanest MoE GGUF profiling baseline,
but it is still not a parity path: it only nudges `c1_2000` and `c1_10000`
inside the same slow GGUF band and remains far below the FP16 TP4 reproduction
target.

The broad GGUF and aligned HF compile graphs now look structurally close at the
major measured boundaries. Rank-0 graphs have `242` parameter inputs, `60`
`gdn_attention_core` calls, `160` `moe_forward` calls, `120`
`moe_forward_shared` calls, and `244` all-reduce occurrences. That rejects
missing broad graph capture or missing major graph regions as the whole-gap
explanation. The next pass should move to graph-replay/kernel mix evidence or
generated Inductor code comparison, not another scheduler or norm variant.

## 2026-06-27 Compile-Debug And External Patch Review

The official image entrypoint's free-form extra-argument path is not safe for
JSON-bearing vLLM flags. Two attempts to pass `--compilation-config` through the
existing string split failed before model load: one preserved literal single
quotes and one stripped double quotes. Treat those as launch-form rejects, not
model or source failures.

A disposable wrapper entrypoint that appends
`--compilation-config "${VLLM_COMPILATION_CONFIG_JSON}"` as a single Bash-array
element works. It let the broad corrected MoE GGUF TP4 path reach health and
emit readable debug-dump plus unpacked compile-cache artifacts without modifying
the release image. That is now the preferred source-inspection route for graph
and generated-kernel comparison.

The compile-debug artifact matters because it confirms the corrected load-time
norm path removes runtime norm subtraction from the compiled graph. The captured
graph still references `_qwen35_effective_weight`, but the function body returns
`self.weight.data`; it does not contain the old `weight - 1.0` runtime body.
Therefore the remaining MoE GGUF performance gap should not be chased as repeat
runtime norm-subtraction overhead.

The unpacked graph and debug kernels point away from missing high-level graph
regions. The diagnostic captured broad graph mode with `PIECEWISE=51` and
`FULL=35`, per-rank graphs of `9235` lines, `60` `gdn_attention_core` calls,
and no `tl.dot` occurrences in the generated debug kernel files. The next
source work should compare generated Inductor regions and replay/kernel mix
against an HF-weight TP4 control, or instrument graph-visible request windows.
Do not repeat norm-offset, scheduler, FusedMoE, GDN-wrapper, logits, sampler,
or Python runner timing variants unchanged.

The external `Kausik-A/qwen3.6-27b-mi50-vllm` repository is useful as a
compatibility checklist, not as a release-performance recipe. It independently
documents several fixes that overlap with this investigation:

- `qwen35` / `qwen35moe` GGUF aliases;
- `ssm_dt.bias -> linear_attn.dt_bias` mapping;
- text-only M-RoPE disabling for Qwen3.5 CausalLM/MoE;
- GGUF quantized embedding and lm-head plumbing;
- `conv1d.weight` 2D-to-3D reshape;
- text-only `IsHybrid` hooks;
- tuple-shard splitting in `MergedColumnParallelLinear`;
- MiniMax M2 GGUF expert aliases.

It should not be copied as a throughput path. Its deployment target is
single-MI50/eGPU, eager mode, 4096 context, and ROCm 6.3-era runtime behavior.
The LocalAIServers goal remains ROCm7.2, TP4/TP8, full-BAR/P2P-on, normal
benchmark warmups, uncapped strict validity, and fixed-token reproduction on
the `.20` release lane.

The aligned HF TP4 compile-debug control now narrows the remaining GGUF gap
below the graph-summary level. The HF control reached the same broad graph
shape as the corrected-load-time GGUF baseline (`PIECEWISE=51`, `FULL=35`),
the same rank-0 custom-op counts for all measured vLLM ops, and the same
absence of `tl.dot` in generated debug kernels. The useful difference is that
GGUF still carries `_qwen35_effective_weight` call sites while HF does not, but
the corrected GGUF body returns `self.weight.data` and contains no runtime
`weight - 1.0` subtraction. Do not chase missing graph capture, missing major
GDN/MoE/all-reduce regions, or runtime norm-subtraction overhead as the active
whole-gap explanation. The next source work should inspect generated Inductor
regions, graph replay behavior, HIP kernel mix, or fixed-token decode-window
traces.

The trace-gated Q/K/V split cleanup removed another small graph mismatch but
did not change the MoE GGUF throughput class. The GGUF source had prepared
trace-only Q/K/V splits even when `VLLM_QWEN35_TRACE_LAYER0=0`; moving that
split under the `trace_active` guard is the right source hygiene, but the
normal `.20` ladder stayed in the same slow band: strict `70.766`, `c1_2000`
`73.104`, and `c1_10000` `65.546` backend TPS after eight warmups. Promote the
change only as cleanup. Reject it as a performance fix. The remaining source
work is still below the high-level graph-summary boundary.

`VLLM_BATCH_INVARIANT` is rejected for the current dense GGUF TP8 path. With no
explicit attention backend, vLLM refuses to initialize batch-invariant mode.
With explicit `FLASH_ATTN`, the launch reaches model initialization but fails
before serving because TorchInductor autotuning selects no valid gfx906 Triton
configuration: the attempted path requires `163840` bytes of shared memory
against a `65536` byte hardware limit. Do not repeat this switch unchanged. Any
future revisit needs a lower-level Triton/Inductor source change or config that
reduces shared-memory demand before it is worth another benchmark ladder.

Removing the vestigial `_qwen35_effective_weight()` call from the corrected
load-time MoE GGUF RMSNorm path does not unlock performance. The direct
`self.weight.data` variant reached the normal `.20` TP4 graph profile with
`PIECEWISE=51`, `FULL=35`, and `302016` KV tokens, then produced strict
`71.489`, `c1_2000` `73.015`, and `c1_10000` `65.454` backend TPS after eight
warmups. The compiled cache for this run had zero
`_qwen35_effective_weight` references, so the wrapper-call hypothesis is
closed. Promote only as possible source cleanup. Reject it as a performance
explanation or release-update candidate.

The MoE GGUF ConditionalGeneration architecture flip is also rejected as a
simple performance path. The smoke resolved
`Qwen3_5MoeForConditionalGeneration`, but failed before model load because the
ConditionalGeneration multimodal processor expected an outer
`Qwen3_5MoeConfig` while the text-only GGUF config presented
`Qwen3_5MoeTextConfig`. It selected the same attention block size (`528`) and
Mamba padding (`0.76%`) as the working CausalLM MoE GGUF baseline before
failing, so page geometry does not justify repeating the architecture-only
override. Any future ConditionalGeneration route needs a deliberate text-config
adapter or multimodal-processor bypass before a benchmark ladder.

Static generated-kernel source diffing now also points away from codegen-body
differences as the MoE GGUF explanation. Comparing the existing `.20`
compile-debug artifacts for corrected GGUF and HF TP4 controls found identical
rank-0 generated kernel counts (`14` each), identical generated function-name
sets, and `11/14` byte-identical generated kernel files. The remaining three
files differed only by serialized model-module metadata
(`language_model.model.layers.*.mlp.experts` versus
`model.layers.*.mlp.experts`); after removing comments and that serialized
`ModuleName` line, all generated kernel files matched. The next useful
diagnostic should inspect replay-time behavior, HIP kernel mix/timing, memory
movement, or fixed-token decode-window traces.

The tuned MoE config path is also closed as a likely explanation. Both the
corrected GGUF TP4 broad graph run and the aligned HF TP4 broad graph control
logged exactly one load of the same
`E=256,N=128,device_name=AMD_GFX906.json` tuned GFX906 MoE config. GGUF also
logged `160` expected force-unquantized FusedMoE expert-layer events for the
F16 GGUF path, while HF logged none. That difference is method routing, not a
missed tuned config. Do not repeat tuned-config lookup debugging unless future
source work changes MoE method selection or the config lookup inputs.

The pre-grad graph signature/body boundary is now also closed as a likely
whole-gap explanation. Comparing the same corrected GGUF and aligned HF TP4
compile-debug artifacts found identical `BEFORE_PRE_GRAD` file counts (`41`
each), identical extracted `def forward` signature lines, and the same combined
signature hash (`9e5949826d4fbc75ea48e48959b9d875c0113b023c204e148dc11f261a33bb9c`).
The measured `torch.ops.vllm` counts matched for all-reduce, shared MoE, and
ROCm unquantized GEMM calls. No GGUF materialization strings such as `gguf`,
`qweight`, `qweight_type`, `weight_type`, or `data_container` appeared in
either compiled pre-grad body. GGUF still had the known
`_qwen35_effective_weight` references in this older artifact, but the later
direct-lookup variant removed those without improving throughput. Do not repeat
static pre-grad signature/body diffing unless a future patch changes graph
construction.

Whole-runner Python timing is rejected as a useful profiler for the current
MoE GGUF fixed-token gap. A full normal ladder with a no-sync
`GPUModelRunner` timing overlay stayed in the slow GGUF band: strict `71.668`,
`c1_2000` `72.936`, and `c1_10000` `65.400` backend TPS after the normal
warmups. The timing file showed decode `model_forward` averages around
`0.042` ms while backend TPS stayed around `65`-`73`, proving the hook measures
Python enqueue/bookkeeping rather than completed graph replay. The next useful
diagnostic needs graph-visible or lower-level completed-work evidence: HIP
kernel mix/timing, graph replay timing, memory movement, or a reduced worker
replay reproducer that can time completed GPU work without perturbing output.

`CUDAGraphWrapper` replay timing is the first useful graph-visible timing
boundary, but it is diagnostic only. A sampled CUDA-event hook around
`entry.cudagraph.replay()` on the corrected MoE GGUF TP4 path produced a normal
ladder of strict `72.063`, `c1_2000` `73.265`, and `c1_10000` `65.685`
backend TPS, with strict gate valid and coherent text. The sampled replay file
contained `2008` rows: `1964` `FULL` single-token replay samples averaged
`13.542115 ms` with min `12.455204 ms` and max `16.630726 ms`; `PIECEWISE`
samples averaged `1.663234 ms` for `num_tokens=16` and `7.766220 ms` for
`num_tokens=416`. Promote this boundary for an aligned HF comparison. Reject
the run as benchmark evidence because sampled CUDA-event timing synchronizes
and perturbs the serving path.

The aligned HF replay comparison closes completed graph replay as the current
whole-gap explanation. With the same replay-timing hook, HF TP4 produced strict
`71.784`, `c1_2000` `72.653`, and `c1_10000` `65.151` backend TPS, while
`FULL` single-token replay averaged `13.655169 ms`. That is effectively the
same timing class as the GGUF run (`13.542115 ms`). Because non-instrumented HF
is normally much faster, the sampled CUDA-event hook is too intrusive for
promotion and should not be repeated unchanged. The next source target should be
outside this sampled replay boundary: request cadence, synchronization side
effects, logits/sampler/postprocess, memory movement not captured by the wrapper,
or HIP kernel mix/timing without per-token synchronization.

Lagged CUDA-event replay timing is also rejected. A variant that queued replay
event pairs and flushed only samples older than `32` sampled events still
collapsed the aligned HF TP4 control to strict `68.482`, `c1_2000` `73.352`,
and `c1_10000` `65.724` backend TPS. It produced `2112` timing rows and a
`FULL` single-token replay average of `13.466812 ms`, but the control no longer
represented the non-instrumented HF reproduction band. Do not insert CUDA event
timing into `CUDAGraphWrapper` for another full ladder. The next profiler route
must avoid replay-loop event injection.

Server-side reasoning parser removal is rejected as a MoE GGUF throughput
lever. The corrected load-time MoE GGUF TP4 path on `.20` reached health with
`reasoning_parser=''`, used the normal warmups and begin-think proxy benchmark
path, then produced strict `72.230` and `c1_2000` `73.023` backend TPS. That is
the same known slow GGUF band, so the server-side parser is not the missing
source-level explanation. Do not repeat reasoning-parser toggles unless a future
source change moves parser behavior into the measured serving hot path.

MoE expert materialization layout is also rejected as the current TP4 GGUF
throughput explanation. A `.20` load-only audit compared corrected F16 GGUF and
HF TP4 under the same published v0.2 image, same P2P-on lane, same tuned GFX906
MoE config, and a disposable env-gated `UnquantizedFusedMoEMethod` metadata
hook. Both routes produced `1280` rows covering `40` MoE layers, `4` TP workers,
`4` post-load stages, and `2` expert tensors. The summarized layout was
identical: `w13_weight` starts contiguous as `(256, 256, 2048)` with stride
`(524288, 2048, 1)`, then ROCm padding changes it to non-contiguous stride
`(557056, 2176, 1)`; `w2_weight` remains contiguous as `(256, 2048, 128)` with
stride `(262144, 128, 1)`. All rows were `torch.float16`, and pointer alignment
was clean at 16/64/128/256-byte boundaries. Although the GGUF caller uses the
full-stack 3-D expert-load path and HF enters per expert, both converge to the
same tensor layout after `process_weights_after_loading()`. Do not repeat
unchanged expert-layout/materialization audits unless future source work changes
GGUF expert loading, method selection, or ROCm padding.

Existing artifact timing now closes two more easy explanations for the MoE GGUF
gap. The corrected GGUF TP4 `c1_2000` and `c1_10000` summaries show matching
slow behavior in both client wall timing and vLLM decode metrics: about
`71.934` wall / `73.015` decode TPS for `c1_2000`, and `65.278` wall /
`65.454` decode TPS for `c1_10000`. That rejects a pure client-side cadence or
metrics-settle artifact. The aligned HF replay-timing comparator is also not a
clean control: the same instrumentation collapses HF to strict `71.784`,
`c1_2000` `72.653`, and `c1_10000` `65.151`, far below the clean `.20` HF
reproduction band of strict `114.725`, `c1_2000` `116.429`, and `c1_10000`
`109.531`. Do not use CUDAGraph replay event timing as a full-ladder comparator
unless a reduced method preserves clean HF performance.

Qwen3.5 GGUF fused-module method selection is not the current MoE TP4 gap. The
existing `.20` `quant_method_audit.tsv` shows `760` linear modules selecting
`UnquantizedLinearMethod`, including hot fused prefixes such as
`linear_attn.in_proj_qkvz`, `linear_attn.in_proj_ba`, shared-expert
`gate_up_proj`, and attention `qkv_proj`. It also shows `160` MoE expert modules
using forced `UnquantizedFusedMoEMethod`. Reject missing GGUF packed-module
mapping or accidental F16 GGUF wrapper dispatch as the current throughput
explanation unless future source work changes model naming or quant-method
construction.

Narrowing the MoE GGUF TP4 launch shape to the clean HF comparator shape is not
the missing performance lever. The `.20` `btok1024/seq1` run used
`--max-num-seqs 1`, `--max-num-batched-tokens 1024`, and CUDAGraph capture
sizes `[1, 2]`, then completed the normal warmups -> uncapped strict ->
`c1_2000` -> `c1_10000` ladder at strict `72.020`, `c1_2000` `73.495`, and
`c1_10000` `65.840` backend TPS. Reject unchanged graph-size and sequence-limit
toggles as the current GGUF-versus-HF MoE TP4 gap.

Static debug-dump graph comparison now closes another easy source-level bucket
for the MoE GGUF TP4 gap. Existing `.20` broad graph debug-dump artifacts for
corrected GGUF and HF TP4 both produced `61` files and `14` generated kernel
files. Their top vLLM op counts matched exactly: `1134` all-reduce ops, `770`
ROCm unquantized GEMM ops, `360` GatedDeltaNet core ops, `280` shared MoE ops,
and `120` each for unified KV-cache update and unified attention with output.
ATen op counts and graph pattern counts also matched. After stripping comments,
UUID/hash-like metadata, and module-name prefix noise, all `14` generated
executable kernel bodies had `0` normalized diff lines. Reject generated
Inductor executable bodies, top-level vLLM op mix, and ATen op mix as the
current explanation for this comparator pair.

The caveat is that the HF debug-dump comparator is not proven to be the exact
clean fast v0.2.1 reproduction path: it used debug-dump compilation and
explicit `--enable-prefix-caching --mamba-cache-mode align`, while the clean
`.20` published report band remains strict `114.725`, `c1_2000` `116.429`, and
`c1_10000` `109.531` backend TPS. The next step is to find raw launch/config
evidence for the actual v0.2.1 reproduction run or build a reduced comparator
that preserves that clean HF band before attributing the gap outside generated
kernels.

Forcing the current release TP4 MoE fastpath overlay into the corrected GGUF
TP4 path is a real improvement, but not enough. The `.20` run
`moe35b_gguf_tp4_directnorm_fastpath_force_dot20_20260627T061255Z` used the
official release image, TP4, P2P-on, `MAX_MODEL_LEN=131072`, the native
`moe35b_tp4_fullbar_p2pon` benchmark sequence, and a direct mount of the
release `runtime/patches/fused_moe.py` fastpath. Logs showed the overlay loaded
and activated for single-token decode on all four TP workers, while rejecting
larger graph-capture/prefill token shapes. The normal ladder was strict-valid
but still below the HF/v0.2.1 band: strict `82.571`, `c1_2000` `84.284`, and
`c1_10000` `74.402` backend TPS. Promote the fastpath mount as a required
comparability condition for future MoE GGUF TP4 tests. Reject "fastpath missing"
as the complete throughput explanation; the remaining gap still requires
source-path work below the launch layer.

Generalizing the MoE fastpath token-count guard is also not enough. The `.20`
run `moe35b_gguf_tp4_directnorm_fastpath_tp8multi_dot20_20260627T063217Z`
mounted the TP8 overlay `files/gfx906_runtime/moe_tp8_overlays/fused_moe_tp8.py`
into the same corrected TP4 GGUF route to allow fastpath activation for token
groups `4`, `2`, and `1`. Logs confirmed `12` active fastpath entries and
`196` larger-shape rejections, and the strict request was valid. The normal
ladder still landed in the same post-fastpath GGUF band: strict `82.958`,
`c1_2000` `84.230`, and `c1_10000` `74.342` backend TPS. Promote the run as
evidence that generalized token-count activation works and should be logged;
reject the token-count guard as the remaining MoE GGUF TP4 throughput
explanation.

Combining the release fastpath with the exact clean HF TP4 scheduler shape is
also rejected. The `.20` run
`moe35b_gguf_tp4_fastpath_exact_hfshape_dot20_20260627T065134Z` used
`--max-num-seqs 1 --max-num-batched-tokens 1024 --enable-prefix-caching
--mamba-cache-mode align`, produced CUDAGraph capture sizes `[1, 2]`, and
mounted the release TP4 MoE fastpath. The normal ladder was strict-valid but
unchanged in throughput class: strict `82.543`, `c1_2000` `84.195`, and
`c1_10000` `74.314` backend TPS. Promote this as closure for the combined
launch-shape/fastpath hypothesis. Reject exact HF scheduler shape as the
remaining MoE GGUF TP4 gap.

Clean HF TP4 artifact boundary is now explicit. The public `.20` reproduction
report records the high HF-weight TP4 band as strict `114.725`, `c1_2000`
`116.429`, and `c1_10000` `109.531` backend TPS. The nearby local HF
directories under the GGUF workspace are not that raw high-band benchmark
artifact: one broad comparator used different launch shape, the exact-shape
entrypoint failed on the hybrid KV layout ambiguity before benchmarking, and
the debug/profiler comparators intentionally perturb or alter the control. Do
not use those directories as the clean HF control. Regenerate a fresh HF control
from the public v0.2.1 reproduction path before running the next source-level
GGUF comparator.

The fresh HF TP4 control has now been regenerated from the public v0.2.1
deploy and benchmark path on `.20`. The run
`hf_moe35b_tp4_clean_repro_20260627T074206Z` used the published image, TP4,
P2P-on, FP16, `MAX_MODEL_LEN=131072`, graph mode, async scheduling, the release
Qwen C1 top-k 8 MoE fastpath, eight pre-measure warmups, uncapped strict, and
the fixed-token tiers through the bundled begin-think proxy. It reproduced the
high band: strict `114.187`, `c1_2000` `116.623`, and `c1_10000` `109.748`
backend TPS. Treat this as the current clean HF control. Future GGUF runs
should compare against this artifact only after a new source-level change below
launch shape, fastpath activation, and graph-capture shape.

Forcing all F16 GGUF `LinearBase` layers to `UnquantizedLinearMethod` is not
the missing MoE TP4 performance fix. The `.20` run
`moe35b_gguf_tp4_force_unquant_linear_dot20_20260627T075850Z` activated forced
unquantized dispatch for `760` linear selections and kept `160` forced
unquantized FusedMoE selections active. The normal benchmark ladder stayed in
the same post-fastpath GGUF band: strict `82.065`, `c1_2000` `83.962`, and
`c1_10000` `74.167` backend TPS. Reject residual GGUF linear method dispatch as
the remaining whole-gap explanation. Promote the result as source-boundary
evidence: the next useful comparison is lower-level HIP/kernel mix and memory
movement during decode, or a reduced completed-work path that preserves the
clean HF control while varying only GGUF-loaded weights/source layout.

The Kausik-A `qwen3.6-27b-mi50-vllm` repo is a useful GGUF compatibility
reference, not a comparable `.20` performance path. It targets a single MI50
eGPU, ROCm 6.3, eager mode, 4096 context, and quantized dense GGUF. Its useful
source ideas are startup and loader bridges: `qwen35` / `qwen35moe` aliases,
explicit tokenizer/config handling, text-only Qwen3.5 override, M-RoPE bypass,
`ssm_dt.bias` mapping, `conv1d.weight` reshape, embedding/lm-head
`quant_config` plumbing, text-only `IsHybrid` hooks, and tuple-shard fused QKVZ
handling. Most of those concepts are already represented in the current local
source work. Do not use the repo's deployment settings as benchmark evidence for
the ROCm7.2 full-BAR/P2P-on TP8/TP4 goal.

Dense GGUF page geometry is no longer the primary suspect. The `.20` run
`dense27b_gguf_tp8_ssmfloat32_dot20_20260627T082830Z` forced
`--mamba-ssm-cache-dtype float32` on the previous best dense GGUF CausalLM
route. Startup moved to the HF-like geometry: attention block size `400`, GPU
KV cache `377,200` tokens, maximum concurrency `11.40x`, and successful graph
capture. The normal warmups -> uncapped strict -> `c1_2000` -> `c1_10000`
ladder remained strict-valid but slower than the prior best GGUF band: strict
`62.992`, `c1_2000` `63.628`, and `c1_10000` `60.118` backend TPS. Reject
cache dtype/page geometry as the remaining dense GGUF whole-gap explanation.
Future dense work should compare kernel mix, memory movement, GDN/state
conversion, graph lowering, or GGUF-loaded weight layout against the clean HF
dense control.

The dense GGUF/HF compile-debug comparison narrows the next source candidate.
The `.20` bounded debug runs
`dense27b_gguf_tp8_compile_debug_20260627T090014Z` and
`dense27b_hf_tp8_compile_debug_20260627T090636Z` both wrote `616` debug-dump
files and the same rank/file classes, and both contained equal counts for the
major GDN/SwiGLU markers (`gdn_attention_core`, `swiglu`, `in_proj_qkvz`).
That rejects a gross compile-shape or missing-GDN lowering explanation. The
clear difference is GGUF-only embedding/materialization surface:
`_apply_gguf_embedding` appeared `176` times in GGUF and `0` times in HF, with
GGUF-only `masked_fill` / `bitwise_or` hits and a larger rank-0 `kernel_0`.
Treat this as the active dense source candidate: not "embedding rows are wrong"
but "the GGUF embedding custom-op/mask/all-reduce route may be decode overhead
or graph-lowering drag." The next test should be embedding-only dispatch
surgery or a timing probe, not another unchanged full ladder.

The first embedding-only dispatch surgery did not change the graph. The `.20`
run `dense27b_gguf_tp8_embedtokens_unquant_compile_debug_20260627T092322Z`
forced only `model.embed_tokens` to `UnquantizedEmbeddingMethod` on all eight
TP workers and reached health, but the compile surface stayed unchanged:
`_apply_gguf_embedding` `176`, `masked_fill` `232`, `bitwise_or` `168`,
`vllm.all_reduce` `14680`, `616` debug-dump files, `176` compile-cache files,
and rank-0 `kernel_0` `53,906` bytes. Reject `embed_tokens` method selection
as the next performance fix. The source issue is likely in GGUF embedding
parameter/materialization or generated custom-op graph construction, not a
simple `get_quant_method()` branch.
This is reinforced by source reading: unquantized embedding should use
`layer.weight` through `F.embedding`, but the captured graph still referenced
`embed_tokens` `qweight`.

Removing the dense GGUF embedding custom-op surface is not enough to close the
dense gap. The `.20` run
`dense27b_gguf_tp8_skipembedqweight_compile_debug_20260627T093747Z` skipped the
Qwen3.5 GGUF `embed_tokens` qweight override/materialization while leaving
`lm_head` and the rest of the release-like dense TP8 path unchanged. This
actually changed the compile surface: `_apply_gguf_embedding` dropped to `0`,
debug dump files increased to `856`, compile-cache files to `208`, and rank-0
`kernel_0` shrank to `4,287` bytes. A short request remained coherent and the
normal benchmark ladder stayed strict-valid, but results were strict `63.606`,
`c1_2000` `64.432`, and `c1_10000` `60.691` backend TPS. Reject embedding
qweight removal as the missing dense performance lever. Promote it as source
boundary evidence: the remaining GGUF dense gap sits below broad graph-shape and
embedding-dispatch explanations, likely in decode kernel mix, memory movement,
GDN/state conversion, GGUF-loaded weight layout, or another lower-level
execution detail.

Dense wrapper parity is also not a quick fix. The `.20` conditional-wrapper
probe `dense27b_gguf_tp8_conditional_wrapper_probe_20260627T101323Z` changed
the GGUF launch to `Qwen3_5ForConditionalGeneration` with the full HF config,
but the server failed before health because vLLM executed the multimodal visual
dummy path and hit uninitialized visual GGUF parameters. A narrower text-only
prefix probe `dense27b_gguf_tp8_langprefix_probe_20260627T101829Z` kept
`Qwen3_5ForCausalLM` and only forced `language_model` source prefixes, but it
also failed before health with uninitialized GGUF parameters during
compile/profile. Reject naive wrapper/prefix parity as a benchmark path. If this
bucket is revisited, it needs a real text-only wrapper that avoids multimodal
profiling and preserves GGUF materialization.

Dense GatedDeltaNet core timing is not the current throughput explanation. The
`.20` request-window timing runs
`dense27b_gguf_tp8_gdnop_timing_dot20_20260627T103024Z` and
`dense27b_hf_tp8_gdnop_timing_dot20_20260627T104237Z` mounted the same
direct-custom-op timing `qwen3_next.py` over the published v0.2 image and sent
the same warmup plus measured request after health. Both paths produced `1152`
timing rows. GGUF averaged `1,755,368` ns `last_ns` across the measured GDN
rows, while HF averaged `1,951,620` ns. Layer 0 also favored GGUF in this
window: `11,951,702` ns versus `16,107,229` ns. Reject
`gdn_attention_core` body time as the missing dense GGUF TPS lever. The dense
gap is now below wrapper, embedding, cache/page, broad graph shape, and GDN
custom-op body explanations; continue with whole-decode kernel mix, memory
movement, residual/norm/projection boundaries, scheduler/replay overhead, or
loaded-weight layout outside the core op.

Dense generated-kernel parity is still not closed. A normalized review of the
existing `.20` dense compile-debug dumps found all `48` generated kernel files
and all `520` pre-grad source files differed after UUID/path/hash-like noise was
removed. Broad markers still match for major GDN/SwiGLU surfaces, but GGUF
keeps an extra token-id remap / embedding-associated region in rank-0
`kernel_0`, with `_apply_gguf_embedding` `176`, `masked_fill` `232`, and
`bitwise_or` `168` hits versus `0` on HF. Treat this as source-boundary
evidence, not final root cause: the debug pair also differs by wrapper/cache
geometry (`Qwen3_5ForCausalLM` GGUF versus `Qwen3_5ForConditionalGeneration`
HF, with different KV-cache geometry). The next useful dense probe should build
an aligned comparator before another full ladder.

Dense text-only wrapper shape is not the missing throughput lever. The `.20`
HF comparator
`dense27b_hf_tp8_textonly_causallm_compare_20260627T111054Z` used HF weights
with the text-only `Qwen3_5ForCausalLM` route, `--language-model-only`, TP8,
full-BAR/P2P-on, FP16, `MAX_MODEL_LEN=131072`, and the same dense release graph
path. It passed the normal ladder with strict `70.353`, `c1_2000` `70.978`,
and `c1_10000` `66.428` backend TPS. Promote this as a clean aligned HF-side
comparator and reject wrapper class alone as the dense GGUF gap. The
Kausik-style text-only route is useful compatibility plumbing, but not a
performance explanation by itself.

The aligned dense debug-dump comparison closes the main caveat on the prior
generated-kernel review. The accepted `.20` HF text-only debug run
`dense27b_hf_tp8_textonly_compile_debugdump_env_20260627T114602Z` used the same
release-like CausalLM comparator and explicit vLLM debug-dump config, then was
compared against `dense27b_gguf_tp8_compile_debug_20260627T090014Z`. Pre-grad
file counts aligned exactly (`520` each), but every common pre-grad file still
differed. Generated kernel partitioning diverged sharply: GGUF produced `48`
kernels (`6` per rank), while HF text-only produced `368` kernels (`46` per
rank). `gdn_attention_core` counts matched at `4608` each, but HF text-only had
more `swiglu` and `vllm.all_reduce` markers, while GGUF alone retained
`_apply_gguf_embedding`, `gguf_embedding`, and `index_select` surfaces. Promote
this as the strongest current source-boundary evidence: the dense GGUF gap is
not wrapper class, not M-RoPE stripping, not broad GDN marker count, and not the
direct GDN custom-op body. Continue with generated-kernel partitioning,
GGUF-loaded weight layout, embedding/index-select residue, scheduler/replay
behavior, or memory movement around residual/norm/projection boundaries.

Dense GGUF Inductor graph partitioning is a useful source clue but not the
standalone fix. The `.20` run
`dense27b_gguf_tp8_graphpartition_probe_20260627T120119Z` added only
`use_inductor_graph_partition=true` plus debug-dump output to the current best
dense GGUF source bundle. The server reached health and materially changed
codegen shape: generated kernels increased from the prior GGUF `48` total
(`6` per rank) to `280` total (`35` per rank), closer to the HF text-only
comparator's `368` total (`46` per rank). The normal benchmark fixture remained
strict-valid and produced strict `63.795`, `c1_2000` `64.556`, and `c1_10000`
`60.826` backend TPS. This is a slight improvement over the previous dense
GGUF band but still misses the v0.2.1 `.20` reproduction target (`65.645`
`c1_10000`) and the HF text-only comparator (`66.428` `c1_10000`). Promote
graph partitioning as a lower-level source boundary. Reject it as a release-path
update or standalone reproduction fix. Continue with kernel mix, graph replay,
memory movement, embedding/index residue, or GGUF-loaded weight layout.

MoE GGUF TP4 Inductor graph partitioning is also rejected as a standalone fix.
The `.20` run
`moe35b_gguf_tp4_graphpartition_exact_hfshape_dot20_20260627T122551Z` preserved
the exact HF scheduler shape, release TP4 MoE fastpath, TP4, P2P-on, FP16, and
`MAX_MODEL_LEN=131072`, then added only `use_inductor_graph_partition=true`
with debug output. It reached health, kept the same fastpath pattern as the
exact-HF-shape baseline (`4` active, `16` rejected), and changed generated
kernels to `184` total (`46` per rank). The normal fixture produced strict
`82.412`, `c1_2000` `84.054`, and `c1_10000` `74.228` backend TPS, effectively
the same as the prior exact-HF-shape fastpath GGUF band (`82.543`, `84.195`,
`74.314`) and far below the clean HF TP4 control (`114.187`, `116.623`,
`109.748`). Reject generated-kernel count/partitioning alone as the MoE TP4
gap. Continue with HIP/kernel mix, memory movement around expert routing and
postprocess, or a reduced completed-work comparator.

The existing timing artifacts should not be reused as clean performance
comparators. Synchronized runner / cudagraph replay timing slows the HF-weight
MoE TP4 path into the same low band as GGUF: the timing-instrumented HF ladder
recorded strict `71.784`, `c1_2000` `72.653`, and `c1_10000` `65.151`
backend TPS, while the nearby GGUF runner-timing ladder recorded strict
`71.668`, `c1_2000` `72.936`, and `c1_10000` `65.400`. That is diagnostic
overhead, not the clean HF control. No-sync timing mostly measures the CPU
envelope and is also insufficient for bottleneck proof.

The latest debug-artifact review keeps the source target below launch flags.
Dense rank-0 graph-partition dumps still show GGUF-only embedding/index
residue and materially different generated bodies after embedding-dispatch
surgery and graph partitioning failed to promote. MoE rank-0 graph-partition
dumps show the expected generated-kernel count but still stay in the GGUF
throughput band. The next useful work should be a reduced completed-work
comparator, GGUF-loaded F16 tensor layout inspection around the linears that
become `rocm_unquantized_gemm`, or memory-movement / replay-boundary analysis
that does not synchronize every replay and destroy the HF control.

The external `Kausik-A/qwen3.6-27b-mi50-vllm` repo is useful as a compact
GGUF compatibility reference, not as a performance target. It reinforces the
needed Qwen3.5/3.6 GGUF plumbing: `qwen35` aliases, text-only
`Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM` registration, M-RoPE stripping,
`ssm_dt.bias` mapping, `quant_config` through `embed_tokens` / `lm_head`,
GGUF GDN `conv1d` reshape, and tuple-shard handling for pre-fused QKVZ loads.
Reject its single-GPU/eGPU, ROCm 6.3, Q6, 4096-context, eager launch profile as
a release-lane fix. Our HF text-only comparator already proves that text-only
wrapper routing can hit the dense release band with HF weights, so the
remaining GGUF gap is lower in loader/materialization, generated graph/body
shape, memory movement, or replay boundaries.

The dense generated-body route now has a clearer boundary. A static scan of the
current dense GGUF graph-partition dump versus the aligned HF text-only
comparator showed real GGUF-only generated surfaces, including
`_apply_gguf_embedding` and rotary-cache `index_select`, plus much higher
clone/reshape/GDN/GEMM generated-line counts. Do not treat these counts as
timings. They are source-surface evidence only.

The important guardrail is that embedding custom-op removal already failed to
promote in `GGUF-189`. That run removed `_apply_gguf_embedding` from the graph,
remained coherent and strict-valid, and still landed at strict `63.606`,
c1_2000 `64.432`, and c1_10000 `60.691` backend TPS. Do not repeat embedding
qweight surgery unchanged.

Standalone generated-script microbenching is also not yet a valid comparator.
The official image can run Torch/ROCm with the entrypoint overridden, and
importing the GGUF plus QwenNext modules registers `_apply_gguf_embedding` and
`gdn_attention_core`. However, the generated graph requires a real vLLM
`ForwardContext` and live no-compile layer objects for `gdn_attention_core`.
Stubbing all-reduce to identity is acceptable only for an isolated single-GPU
diagnostic; stubbing GDN would remove the work being measured.

Next dense source work should use either a reduced completed-work comparator
with real forward context/no-compile layers, or a live-server request-window
profiler/marker route that does not synchronize every replay and collapse the
HF control.

Combining dense GGUF graph partitioning with embedding qweight-skip does not
promote. The `.20` run
`dense27b_gguf_tp8_graphpartition_skipembed_dot20_20260627T131645Z` reached
health under the release-like TP8 full-BAR/P2P-on lane with FP16,
`MAX_MODEL_LEN=131072`, P2P-on, async scheduling, graph capture, Inductor graph
partitioning, and the GGUF embedding qweight-skip/unquantized embedding
environment. The normal fixture produced strict `63.747`, `c1_2000` `64.587`,
and `c1_10000` `60.866` backend TPS. The strict gate was valid. Reject this as
a release-path update because it remains in the prior dense GGUF throughput
band and below the v0.2.1 `.20` dense reproduction target and HF text-only
comparator. Promote it as a clean negative result: neither embedding qweight
materialization removal nor generated-kernel partitioning, alone or together,
explains the remaining dense GGUF gap.

Simple post-load tensor layout is also not the dense GGUF gap. The `.20`
diagnostic run `dense27b_gguf_tp8_layoutdiag_dot20_20260627T135511Z` used the
same release-like dense GGUF TP8 full-BAR/P2P-on graph-partition path and
extended the GGUF loader to print layer-0 GatedDeltaNet tensor layout after
weight processing. Across all eight TP ranks, `in_proj_qkvz.weight`,
`conv1d.weight`, and `out_proj.weight` were CUDA FP16 tensors with contiguous
row-major strides: `(2060, 5120)` / `(5120, 1)`, `(1280, 1, 4)` / `(4, 4, 1)`,
and `(5120, 768)` / `(768, 1)`. The server reached health and a short smoke
request produced coherent Qwen reasoning text before hitting the intentionally
low token cap. Reject `.contiguous()` or reload-time copying of those GDN
weights as an unchanged performance experiment. Continue with request-window
profiling, generated graph/replay boundaries, clone/reshape memory movement, or
a reduced forward-context comparator.

M-RoPE is a real source-surface difference, but not a launch-flag fix. The HF
Qwen3.6-27B snapshot includes `mrope_interleaved=true` and
`mrope_section=[11,11,10]`; the current GGUF text config omits those fields and
therefore uses the plain/base rotary path. A `.20` M-RoPE config restoration
probe reached health, but the first request failed because the GGUF/text-only
request path did not provide `req.mrope_positions`. Reject config-only M-RoPE
restoration as a benchmark path. If M-RoPE is revisited, it needs scheduler /
request-position plumbing equivalent to the HF path. Otherwise, keep source
work below the current plain-RoPE GGUF path and focus on request-window,
memory-movement, generated-body, or forward-context comparators.

Dense GGUF finally promoted when the logits path was fixed, not when the model
body was retuned. The `.20` run
`dense27b_gguf_tp8_lmheadunquant_dot20_20260627T142800Z` kept the release-like
dense TP8 full-BAR/P2P-on path, graph partitioning, embedding-qweight skip, F16
GGUF weights, FP16, async scheduling, graph capture, and `MAX_MODEL_LEN=131072`,
then changed only the post-materialization `lm_head.quant_method` to
`UnquantizedEmbeddingMethod()`. The normal fixture produced strict `70.350`,
c1_2000 `71.396`, and c1_10000 `66.842` backend TPS. Promote this as the first
dense GGUF source path that beats the published dense c1_10000 gate, the
v0.2.1 `.20` reproduction, and the aligned HF text-only c1_10000 comparator.
The later `.20` lane-validation run
`dense27b_gguf_tp8_lmheadunquant_dot20_lanevalidate_20260628T110342Z`
superseded the raw values with strict `70.505`, c1_2000 `71.589`, and
c1_10000 `66.967` backend TPS. The next MoE GGUF task should test whether the
MoE `lm_head` has the same GGUF-quant-method residue before changing any
broader kernel path.

MoE GGUF does not share the dense `lm_head` bottleneck. The `.20` run
`moe35b_gguf_tp4_lmheadunquant_dot20_20260627T145000Z` copied the exact-HF-shape
graph-partition MoE GGUF patch bundle and changed only the post-load
`lm_head.quant_method` under `VLLM_QWEN35_GGUF_LM_HEAD_UNQUANT=1`. The marker
fired on all four TP ranks and the normal fixture remained strict-valid, but
the results stayed in the prior MoE GGUF band: strict `82.024`, c1_2000
`84.061`, and c1_10000 `74.230` backend TPS. Reject lm-head unquantization as
a standalone MoE GGUF fix. Continue below the logits path, especially expert
routing / fastpath layout, memory movement around expert postprocess, or a
completed-work comparator against the clean HF MoE TP4 control.

A static MoE graph marker scan gives the next source boundary but not proof of
root cause. The available HF rank-0 debug dump has `14` generated kernel files
and `41` pre-grad files, while the current GGUF lm-head-unquant dump has `46`
generated kernel files and only `1` pre-grad file. GGUF also has substantially
more marker references for `gdn_attention_core`, `moe_forward_shared`,
`rocm_unquantized_gemm`, and `vllm.all_reduce`, plus `909` references to the
GGUF-only `_qwen35_effective_weight` helper. Promote this as a low-perturbation
source boundary, not as timing proof. The next useful source inspection is
whether `_qwen35_effective_weight` creates graph-visible per-token work or is
only harmless materialization/debug residue.

The `_qwen35_effective_weight` helper is not the MoE GGUF bottleneck. The `.20`
source probe
`moe35b_gguf_tp4_base_rmsnorm_lmheadunquant_dot20_20260627T153000Z` replaced
the custom GGUF RMSNorm wrapper with the base Gemma RMSNorm alias while keeping
load-time norm-offset correction enabled, runtime norm-offset correction
disabled, and all other MoE TP4 settings fixed. The helper disappeared from the
rank-0 debug dump, but the normal fixture still landed at strict `81.808`,
c1_2000 `84.403`, and c1_10000 `74.510` backend TPS. Reject no-op runtime norm
wrapper cleanup as the MoE promotion path. The stronger clue is that the
release MoE fastpath logged layout rejections for GGUF expert tensors, including
`w1_shape=(256, 256, 2048)` with stride `(557056, 2176, 1)` and
`w2_shape=(256, 2048, 128)` with stride `(262144, 128, 1)`. Next MoE source
work should compare HF and GGUF expert tensor layout after load and then either
materialize GGUF experts once into the fastpath-accepted layout or extend the
fastpath to consume the GGUF stride pattern without per-token copies.

Fastpath admission alone is also not the MoE GGUF fix. The `.20` source probe
`moe35b_gguf_tp4_batch_fastpath_base_rmsnorm_lmheadunquant_dot20_20260627T153500Z`
added batch-aware TopK8 Triton kernels so the GGUF MoE path accepted grouped
token counts instead of rejecting everything except one-token calls. The log
confirmed `fastpath_active=4` and `fastpath_rejects=0`, including accepted
groups such as `1024`, `16`, `6`, `2`, and `1`. The normal fixture still landed
at strict `79.186`, c1_2000 `84.044`, and c1_10000 `74.194` backend TPS.
Reject admission-only fastpath patches. The remaining gap is inside completed
work: expert tensor layout/value materialization, generated kernel body,
activation/reduction memory movement, or another lower-level path that differs
from the clean HF MoE TP4 control after the call is admitted.

External GGUF compatibility bundles are useful for checklist coverage, not for
benchmark settings. The `Kausik-A/qwen3.6-27b-mi50-vllm` review at commit
`61b273d` independently highlighted the same Qwen3.5/Qwen3.6 GGUF source
surface we have been repairing: `qwen35` / `qwen35moe` architecture aliases,
text-only M-RoPE stripping, `ssm_dt.bias` to `linear_attn.dt_bias` mapping,
MoE expert tensor aliases, GGUF `conv1d.weight` 2D-to-3D reshaping,
`quant_config` propagation into embeddings / lm-head, and tuple-shard fused
projection loading. Promote those as compatibility checklist items. Reject the
repo's documented runtime settings as performance guidance for our path because
they target single-card / eGPU / eager / 4096-context serving rather than the
`.20` full-BAR/P2P-on ROCm7.2 release lane.

MoE GGUF expert tensors now have measured layout and sampled-value parity with
the HF path after load/materialization. The `.20` HF value audit
`moe35b_hf_tp4_value_audit_dot20_20260627T160500Z` and GGUF value audit
`moe35b_gguf_tp4_value_audit_dot20_20260627T161500Z` each produced `576`
sampled value rows over layers `0`, `20`, and `39`, experts `0`, `7`, `127`,
and `255`, and stages `after_super`, `after_rocm_padding`, and
`after_setup_kernel`. Ignoring TP-rank print order, the sampled values matched
exactly. The earlier layout audit also matched shape/stride/contiguity:
ROCm-padded `w13_weight` is `(557056, 2176, 1)` in both paths, and `w2_weight`
stays `(262144, 128, 1)`. Reject expert tensor materialization as the main MoE
GGUF throughput bottleneck. Continue into generated kernel body, actual
activation/reduction memory movement, FusedMoE prepare/finalize, shared expert
work, or a reduced completed-work comparator.

The earlier MoE static graph expansion hypothesis is superseded by the matched
HF graph dump. The broad HF dump used a different compile shape, so its lower
kernel/marker counts were not a fair baseline. The matched HF diagnostic
`moe35b_hf_tp4_matched_graphdump_dot20_20260627T161707Z` used the same MoE TP4
shape and graph-partition settings as the current GGUF path and produced the
same rank-0 file count, generated-kernel count, pre/post-grad count, and marker
counts for `gdn_attention_core`, `moe_forward_shared`, `moe_forward`,
`rocm_unquantized_gemm`, and `vllm.all_reduce`. The diagnostic failed before
health at the known hybrid KV layout assertion, so it is static source evidence
only, not benchmark evidence. Reject broad static graph inflation as the main
MoE GGUF bottleneck. The narrower remaining source leads are runtime
layer-name-keyed custom-op/cache behavior, three semantic Triton
`.best_config` differences, request-window completed work, and a reduced
forward-context comparator that preserves the real vLLM path.

HF-style GDN/attention layer-name aliasing is not the MoE GGUF promotion path.
The `.20` source probe
`moe35b_gguf_tp4_hflayeralias_lmheadunquant_dot20_20260627T170500Z` rewired
GGUF GDN and attention `static_forward_context` keys from `model.layers.N...`
to `language_model.model.layers.N...` under
`VLLM_QWEN35_GGUF_HF_LAYERNAME_ALIAS=1` while keeping the same MoE TP4
full-BAR/P2P-on image, graph-partition path, lm-head unquantization, FP16
weights, and release fixture. Startup succeeded, the alias markers fired, the
strict output was coherent, and the strict gate passed. The normal ladder still
landed at strict `82.488`, c1_2000 `84.001`, and c1_10000 `74.205` backend TPS,
which is effectively unchanged from the prior lm-head-unquant MoE GGUF run.
Reject GDN/attention layer-name cache keying as the primary gap. The remaining
MoE source work should focus on completed work below the call boundary:
semantic Triton `.best_config` differences, MoE prepare/finalize,
shared-expert work, activation/reduction movement, and a reduced comparator
against the clean HF MoE TP4 path.

The three matched-graph `.best_config` differences are weak evidence, not a
promotion target. A rank-by-rank check showed that the differing XBLOCK and
num_warps choices are shuffled across HF and GGUF ranks instead of forming a
stable model-format split. The debug dumps also do not directly map those
config hashes back to one generated source body. Reject forcing those Triton
configs without a hot-kernel mapping and repeated evidence. Before deeper
kernel forcing, verify release-shape parity: the clean HF MoE TP4 repro used
the standard release deployment shape and produced c1_10000 `109.748`, while
some GGUF MoE probes used constrained debug shape settings.

Release-shape parity is now rejected as the main MoE GGUF gap. The `.20` probe
`moe35b_gguf_tp4_release_shape_lmheadunquant_dot20_20260627T170000Z` removed
the constrained debug launch settings and used the cleaner release-style GGUF
argument set while preserving the same FP16 GGUF, TP4 full-BAR/P2P-on lane, and
normal benchmark ladder. It reached health and strict passed, but the ladder
landed at strict `81.329`, c1_2000 `85.178`, and c1_10000 `75.106` backend
TPS. This is only a small improvement over the prior low GGUF band and remains
far below the clean HF/release TP4 reproduction. Continue below launch shape:
completed MoE work, shared/expert execution, prepare/finalize, activation /
reduction movement, graph replay, request cadence, or HIP kernel mix/timing.

Raw-image isolated-cache HF controls are rejected as high-band comparators. The
`.20` probe `moe35b_hf_tp4_cache_capture_dot20_20260627T172011Z` launched the
official image with fresh per-run TorchInductor, Triton, and vLLM caches, then
started the normal benchmark warmups. It reached health, but warmup 2 decoded
at only `73.879` backend TPS. The log showed the raw image entrypoint path with
broad capture, `max_num_batched_tokens=2048`, `enable_prefix_caching=False`,
and an unknown core-environment warning for
`VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH`. Reject this as an HF baseline for GGUF
comparison. Future HF controls must preserve the full deploy-package/runtime
patch path and prove high-band TPS before their cache or generated artifacts
are used as evidence.

Full deploy-package HF controls restore high-band MoE behavior and sharpen the
remaining GGUF target. The `.20` probe
`moe35b_hf_tp4_deploypkg_cache_capture_dot20_20260627T173555Z` launched the
same image through the copied release deployment package and observed the Qwen
C1 TopK8 MoE fastpath applying on the expected `E=256`, `N=256`, `K=2048`,
`top_k=8`, FP16 shape. CUDA graph capture completed and the API reached health.
The first Python capture-client attempt stalled before any request reached
vLLM, with zero running/waiting requests and zero prompt/decode token counters.
Replaying that generated request with `curl` through the begin-think proxy
completed `2000` generated tokens from a `431` token prompt in about `18`
seconds, and the vLLM logger reported a steady generation interval of `115.7`
tokens/s. Promote full deploy-package controls for HF-vs-GGUF profiling.
Reject the stalled Python client invocation as harness blockage. The remaining
MoE GGUF gap is now below launch shape, fastpath admission, expert layout/value,
lm-head/norm mapping, and proxy/tokenizer correctness.

The first MoE GGUF ConditionalGeneration wrapper probe is rejected as
implemented but preserves a useful source boundary. The `.20` disposable patch
`qwen35moe-wrappermapper-conditional-20260627` tried to route FP16 GGUF through
`Qwen3_5MoeForConditionalGeneration` by adding a wrapper mapper and GGUF loader
aliases. The launch progressed through several loader fixes, then failed before
health in `_get_gguf_weights_map(...)` with unmapped wrapper-shaped language
parameters and vision parameters. No request completed. This rejects the
class-level mapper approach because it runs too late for GGUF's internal dummy
model map. It does not reject wrapper/causal construction as a possible MoE
gap. The next wrapper-parity test, if pursued, must patch the GGUF loader map
itself: build a text-only Qwen3.5-MoE tensor map, translate names to the
wrapper's `language_model.model.*` parameters before the missing-parameter
check, and only then run the normal benchmark ladder.

ConditionalGeneration wrapper parity is now rejected as a MoE GGUF performance
route, not only as a loader failure. The follow-up `.20` patch added the
loader-map fixes, skipped the absent `mm_proj` sidecar for the text-only GGUF
wrapper probe, and delegated wrapper text weights into the working
`Qwen3_5MoeForCausalLM` loader. The server reached health, a small Qwen smoke
was coherent, and the full benchmark ladder completed. Results stayed in the
low GGUF band: strict `83.140`, c1_2000 `85.061`, and c1_10000 `74.996`
backend TPS, with strict gate valid. This rejects wrapper/CausalLM construction
as the main remaining MoE gap. Continue below the admitted MoE path: expert and
shared-expert execution, prepare/finalize, activation/reduction movement, graph
replay, request cadence, or HIP kernel mix/timing.

Static graph marker parity is necessary but not sufficient for MoE GGUF
throughput parity. The high-band HF deploy-package graph and low-band GGUF
wrapper graph matched on `all_reduce`, `moe_forward_shared`,
`gdn_attention_core`, and `moe_forward`, but GGUF carried `440`
`rocm_unquantized_gemm` references versus HF's `320`. That localized to the
split GGUF GatedDeltaNet beta/alpha projection. A MoE-specific fused qkv/z/b/a
GGUF probe removed that difference: the fused graph also had `320`
`rocm_unquantized_gemm` references and stayed coherent/strict-valid. Throughput
still stayed low-band: strict `83.030`, c1_2000 `84.778`, and c1_10000
`74.789` backend TPS. Reject extra GDN projection GEMMs and fused-projection
shape parity as the primary remaining MoE GGUF bottleneck. The next useful
work is below static graph shape: graph replay cadence, completed MoE
prepare/finalize, shared-expert work, activation/reduction movement, HIP kernel
mix/timing, or a reduced comparator that can isolate request-time work without
perturbing graph capture.

The combined fused-qkvzba plus base-RMSNorm control rejects the remaining
visible static graph marker difference as well. The `.20` run
`moe35b_gguf_tp4_fusedgdn_qkvzba_base_rmsnorm_dot20_20260627T193000Z`
removed `_qwen35_effective_weight` from the compiled graph while keeping the
HF-style GDN projection marker counts. The ladder stayed low-band: strict
`82.534`, c1_2000 `84.892`, and c1_10000 `74.872` backend TPS. This means the
coarse Python graph marker surface can now look close to HF while runtime
throughput remains far below HF/release. Stop chasing RMSNorm wrapper removal,
fused projection shape, or similar static graph-cleanup combinations without
new evidence. The active work must move to runtime behavior: graph replay
cadence, topk8/fused-MoE fastpath behavior, completed MoE prepare/finalize,
shared-expert work, activation/reduction movement, HIP kernel mix/timing, or a
reduced request-time comparator.

The final visible normalized graph operation difference is also rejected as the
MoE GGUF bottleneck. The `.20` no-split probe
`moe35b_gguf_tp4_fusedgdn_qkvzba_nosplit_base_rmsnorm_dot20_20260627T201500Z`
moved the unused `mixed_qkv.split([q_size, k_size, v_size], dim=-1)` behind
the trace-only branch. The rank-0 graph then had zero
`mixed_qkv.split([512, 512, 1024])` occurrences and matched the high-band HF
control line count at `8935`, but the benchmark stayed low-band: strict
`81.220`, c1_2000 `84.852`, and c1_10000 `74.845` backend TPS. The release
fastpath pattern also stayed unchanged: grouped GGUF expert shapes rejected and
one-token shapes activated. This promotes static graph parity as completed
diagnosis, not as a performance solution. The remaining gap is now a
runtime/kernel question: generated kernel body, graph replay cadence, expert
layout consumption, activation/reduction movement, or HIP kernel mix/timing.

The env-only `VLLM_USE_TRITON_AWQ=0` probe is rejected as implemented, but it
adds an important source boundary. The `.20` no-split AWQ-off run
`moe35b_gguf_tp4_nosplit_awqoff_dot20_20260627T195932Z` contained
`VLLM_USE_TRITON_AWQ=0` in the container environment, yet every compiled rank
cache still recorded `VLLM_USE_TRITON_AWQ=true`. The benchmark stayed low-band:
strict `82.803`, c1_2000 `84.557`, and c1_10000 `74.608` backend TPS. Do not
repeat that launch unchanged. If the AWQ/cache-key hypothesis is tested again,
it must be source-level: the copied ROCm patch's `verify_quantization(...)`
sets `os.environ["VLLM_USE_TRITON_AWQ"] = "1"` outside the `quant == "awq"`
conditional, so the launch-time `0` cannot win. First patch that assignment and
verify the cache factor flips to false; only then run another full ladder.
Otherwise continue into generated kernel body, graph replay cadence, expert
layout consumption, activation/reduction movement, and HIP kernel mix/timing.

The source-level AWQ cache-factor false probe is also rejected as the main MoE
GGUF promotion path. The `.20` run
`moe35b_gguf_tp4_nosplit_awqflagrespect_dot20_20260627T202100Z` added a
disposable ROCm patch guard so the compiled cache factor actually recorded
`VLLM_USE_TRITON_AWQ=false` on all ranks. That proved the cache-key hypothesis
was testable, but performance only moved to strict `83.940`, c1_2000 `85.177`,
and c1_10000 `75.079` backend TPS. This is a marginal improvement, not a
promotion. Reject more AWQ/cache-factor cleanup without new runtime evidence.
The fastpath source inspection refines the remaining target. The release
topk8 fastpath kernel is a one-token path (`hidden_states.size(0) == 1`), so
not every grouped `shape_or_layout` rejection is itself a bug. However, the
high-band HF deploy-package debug control logged grouped shape-seen events
without grouped rejections and then activated the one-token fastpath, while the
low-band GGUF runs repeatedly rejected grouped shapes with padded `w1` expert
stride `(557056, 2176, 1)`. The main remaining path is therefore
kernel/runtime-level layout consumption: why the GGUF expert layout feeds a
slower grouped standard path while HF stays high-band, plus fused-MoE/shared
expert work, activation/reduction movement, graph replay cadence, and HIP
kernel mix/timing.

Matching the HF `VLLM_NCCL_SO_PATH` cache factor does not close the MoE GGUF
gap. The `.20` run
`moe35b_gguf_tp4_nosplit_awqflagrespect_no_ncclpath_dot20_20260627T204804Z`
removed the explicit `/rccl-overlay/install/lib/librccl.so.1` setting from the
`GGUF-223` no-split/AWQ-false lane while preserving the FP16 GGUF, TP4,
full-BAR/P2P-on, normal warmup/strict/fixed-token ladder, and all other source
patches. All four compiled cache-key files recorded an empty
`VLLM_NCCL_SO_PATH` and `VLLM_USE_TRITON_AWQ=false`, matching the relevant
high-band HF cache-factor values. The ladder remained low-band: strict
`81.425`, c1_2000 `82.634`, and c1_10000 `73.023` backend TPS. The fastpath
pattern was unchanged (`4` active, `204` rejected, `208` shape-seen). Reject
NCCL-path env/cache parity as the missing MoE GGUF lever. Do not repeat
NCCL-path probes without new all-reduce or kernel-level evidence; keep moving
toward steady decode kernel mix, grouped expert execution, activation/reduction
movement, and completed-work comparison against the high-band HF TP4 control.

Do not over-interpret the grouped TopK8 fastpath rejection logs. A follow-up
source audit showed that the release TopK8 helper is a one-token path by
construction, and that `shape_or_layout` rejection messages require the
separate `force` / `debug` setting while `shape seen` messages only require the
enable flag. Therefore the earlier "HF grouped shape-seen without rejection"
versus "GGUF grouped rejection" contrast is not a fair standalone comparator
unless both runs use the same debug mode. The remaining useful target is still
runtime-level MoE work, but the next control should capture a high-band HF TP4
run with comparable debug/cache visibility or use lower-level generated-kernel
evidence.

Matched-debug HF confirms that grouped TopK8 rejection logs are not GGUF-only.
The `.20` HF control
`moe35b_hf_tp4_force_debug_cache_capture_dot20_20260627T211519Z` launched the
native `moe35b_tp4_fullbar_p2pon` release profile with
`VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH=force`. It logged `4` one-token fastpath
activations, `55` grouped `shape_or_layout` rejections, and the same padded
expert stride seen in GGUF (`w1_stride=(557056, 2176, 1)`). That rejects the
grouped rejection count and padded stride as a GGUF-only root cause. The same
run also proves force-debug mode is too perturbing for throughput comparison:
the first warmup fell to `62.777` backend decode TPS with `206.191` seconds of
prefill time. Future work should use normal high-band HF/GGUF runs for
performance and separate low-overhead profiling or generated-kernel inspection
for the source-level difference.

The remaining MoE GGUF gap is now pinned to steady decode, not prefill or
client accounting. A read-only comparison of the clean HF TP4 summaries against
the latest GGUF no-NCCL-path probe showed the same `431` prompt tokens and
nearly identical prefill time (`0.317`-`0.319` seconds HF versus
`0.328`-`0.331` seconds GGUF). The fixed-token gap is decode seconds:
`c1_2000` HF `17.149` seconds versus GGUF `24.203`, and `c1_10000` HF
`91.118` seconds versus GGUF `136.944`. This rejects prompt/prefill/proxy
accounting as the main issue and promotes decode-phase graph replay,
HIP/Triton kernel mix, MoE postprocess memory movement, and completed-work
comparison as the active boundary.

FusedMoE HF-style layer-name aliasing is now rejected as a MoE GGUF promotion
path. The `.20` probe
`moe35b_gguf_tp4_fusedmoe_hfname_no_ncclpath_dot20_20260627T213900Z` rewrote
`160` GGUF FusedMoE expert-layer names from `model.layers.*.mlp.experts` to
`language_model.model.layers.*.mlp.experts`, matching the remaining normalized
HF/GGUF TorchInductor cache-name difference. The server loaded, captured
graphs, reached health, and completed the normal benchmark ladder with strict
validity, but stayed low-band: strict `80.661`, c1_2000 `82.737`, and
c1_10000 `73.111` backend TPS. This rejects `ModuleName` / layer-name cache
metadata as the missing decode lever. Continue with runtime decode work:
generated kernel body, graph replay cadence, grouped expert execution,
activation/reduction movement, and HIP/Triton kernel mix under normal,
non-force-debug runs.

TorchInductor `.kernel_perf` aggregate timing is also not the whole MoE GGUF
gap. A read-only join of the high-band HF deploy-package cache and the latest
GGUF alias-probe cache found `76` `.kernel_perf` files on each side with no
missing joined keys. The summed values were HF `2.106399` and GGUF `2.210239`,
a `1.049x` ratio. That is far smaller than the request-level decode gap
(`91.118` HF c1_10000 decode seconds versus roughly `136.8`-`136.9` GGUF
seconds). Keep `.kernel_perf` high-ratio files as secondary clues, but the next
measurement needs request-window kernel counts, replay cadence, and
completed-work accounting.

The current CUDAGraph replay-timing probe confirms the MoE GGUF performance
boundary is below the launch/proxy/prefill layer. The `.20` run
`moe35b_gguf_tp4_current_cgreplay_timing_dot20_20260627T221003Z` reused the
current strict-valid MoE GGUF TP4 path, completed the normal warmups -> strict
-> `c1_2000` -> `c1_10000` ladder, and stayed low-band: strict `81.280`,
`c1_2000` `82.538`, and `c1_10000` `73.114` backend TPS. Sampled CUDAGraph
`FULL` replay averaged `12.054565` ms, with all TP ranks tracking together.
The one-token replay cost drifted upward late in the request window from about
`11.51` ms to `14.47` ms, matching the long-tier drop. Promote this as evidence
that the next MoE GGUF work must inspect decode graph/custom-op work inside or
immediately around replay: generated kernel body, standard fused-MoE grouped
expert execution, shared-expert work, activation/reduction movement, and any
extra GGUF-only completed work per token. Reject another unchanged full ladder
or more metadata/cache-key cleanup without a source-level decode-path change.

The follow-up graph-input metadata probe narrowed the investigation and exposed
a comparator mismatch. Captured metadata summaries showed one HF run using
`3072`-wide GatedDeltaNet-like inputs with a separate `16 x 2048` tensor, while
the GGUF run used a `3088`-wide folded input family and did not show that
separate `16 x 2048` input in the same signature family. The useful counts were
HF `20764` metadata lines versus GGUF `27588` metadata lines; `shape=(N, 8,
128)` with stride `(3072, 128, 1)` appeared `1044` times for HF and `0` for
GGUF, while stride `(3088, 128, 1)` appeared `0` times for HF and `1464` times
for GGUF. Follow-up source-state verification showed that the HF metadata run
mounted only the CUDAGraph metadata patch and used the release image's stock
split `qwen3_5.py` source hash
`587e313270cf78336b7ef9a6ab4df3201ce2008369bb6ae14e208d1d6d606d6f`, while the
high-band release patch hash
`71eaf52b5f85c022380599ae80ce0f478e989b6052a4f14a88ef4305edd3c046` and the
GGUF metadata source hash
`4359d9c3958047873b40b04cd174309df08266dabc0b81edc9cdbd92c19c16d7` both use
the fused `3088` qkv/z/b/a form. Therefore reject the `3072 + 16` versus `3088`
contrast as a high-band performance hypothesis. Continue with lower-level
decode graph work inside FusedMoE, shared expert, activation/reduction, and
generated kernels, or rerun metadata only with the release-patched HF source
mounted.

The valid high-band CUDAGraph replay comparator shows that the MoE GGUF gap is
inside the replayed decode path or work immediately adjacent to it. The `.20`
HF control
`moe35b_hf_tp4_releasepatch_cgreplay_dot20_20260627T232912Z` mounted the release
patch bundle plus the replay-timing patch and reran the normal warmups ->
strict -> `c1_2000` -> `c1_10000` ladder with curl POST mode after the initial
urllib benchmark path hung before backend submission. The control was
high-band: strict `119.928`, `c1_2000` `121.255`, and `c1_10000` `114.020`
backend TPS. Its sampled FULL CUDAGraph replay averaged `7.887792` ms, while
the current GGUF replay-timing run averaged `12.054565` ms. HF drifted from
`7.730574` ms in the first 250 FULL samples to `8.664663` ms in the final bin;
GGUF drifted from `11.515907` ms to `14.436282` ms. Promote this as the current
best explanation for the 114-versus-73/75 TPS split. Reject more launch,
proxy, prompt, tokenizer, NCCL, layer-name, or static `.kernel_perf` cleanup as
the primary path unless it changes replayed decode work. The next useful work
is request-window kernel counts, generated kernel bodies, FusedMoE grouped
expert execution, shared/routed expert reduction movement, and GGUF-specific
layout/materialization inside the replayed graph.

The generated graph/source comparison now rejects generated Python differences
as the MoE GGUF bottleneck. A direct comparison of the high-band HF replay
cache and current GGUF replay cache found `76/76` joined `.kernel_perf` keys,
zero HF-only keys, zero GGUF-only keys, and `208/208` common generated Python
content hashes. The HF `.kernel_perf` sum was `2.227507998238`; the GGUF sum
was `1.860795992897`, a GGUF/HF ratio of `0.835371`. In other words, the
static generated kernels are identical by source hash and do not benchmark
slower on the GGUF side. A corrected release-patched HF graph-input metadata
run also removed the stale `3072 + 16` contrast: both release-patched HF and
GGUF use the fused `3088` qkv/z/b/a graph-input family, with no
`shape=(3072, 2048)` or `shape=(16, 2048)` entries. The remaining gap is
runtime work inside or adjacent to the replayed graph: kernel invocation counts,
custom-op work, tensor materialization, layout/stride effects, grouped expert
execution, and shared/routed reduction movement. The next experiment should
measure per-request invocation counts or add low-overhead counters around
FusedMoE/custom-op boundaries, not rerun unchanged ladders.

The MoE branch-localization trace now rejects gross activation scale as the
reason GGUF chooses the wrong second token in the faithful chat branch probe.
The corrected GGUF gated-internal trace
`moe35b_gguf_tp4_gatedinternaltrace_eager4k_dot20_20260628T001219Z` reproduced
the split after the shared special token `248046`: GGUF ranked `Here` (`8160`)
first at `26.468750` and `Thinking` (`90700`) second at `22.953125`. However,
selected HF-versus-GGUF layer norms were essentially identical: layer-0
`pre_input_layernorm` `2.867060` versus `2.867060`, layer-0
`post_input_layernorm` `278.260071` versus `278.259125`, layer-39
`post_input_layernorm` `269.081757` versus `269.077484`, layer-39 `post_mlp`
`273.929382` versus `274.308777`, and layer-39 `post_attention` `48.878269`
versus `49.109112`. This promotes hidden-direction / trajectory drift as the
active correctness boundary and rejects another norm-scale or gross-layer-output
fix. The next diagnostic should dump comparable last-token vectors at selected
layer boundaries and compute layer-wise cosine against HF.

The layer-wise vector-cosine probe has now localized that hidden-direction drift
to the early residual path. The `.20` HF/GGUF vector trace pair
`moe35b_hf_tp4_vectortrace_eager4k_dot20_20260628T002050Z` and
`moe35b_gguf_tp4_vectortrace_eager4k_dot20_20260628T002050Z` used the same
faithful two-token branch prompt. HF produced `Thinking Process`; GGUF
reproduced the null-content branch. Full last-token vector comparison showed
perfect agreement at layer-0 input (`1.000000000`) and post-input norm
(`0.999999943`), then drifted immediately after layer-0 attention/MLP:
layer-0 `post_attention` cosine `0.988626168`, layer-0
`post_attention_layernorm` `0.982377650`, and low-norm layer-0 `post_mlp`
`0.849558298`. The residual trajectory is already materially off by layer 10
(`pre_input_layernorm` `0.897712368`) and layer 30 (`pre_input_layernorm`
`0.788824286`). This rejects final-logit-only debugging and narrows the next
useful source work to sparse MoE/router/expert output, GatedDeltaNet
contribution ordering, and residual updates in the layer-0 to layer-10 window.

The corrected component trace now puts the strongest suspicion on the sparse
MoE expert path rather than on final logits, tokenizer, graph/eager mode, or a
reduce-only issue. The first component-trace patch failed silently because the
wrapped sparse block did not retain the layer prefix and the request-gated
prefix filter dropped the fallback `mlp` label. After explicitly setting
`self.mlp.prefix = f"{prefix}.mlp"`, the `.20` HF/GGUF trace pair
`moe35b_hf_tp4_componenttraceprefix_eager4k_dot20_20260628T011830Z` and
`moe35b_gguf_tp4_componenttraceprefix_eager4k_dot20_20260628T011830Z`
captured MoE internals for layers `0`, `1`, `2`, `5`, and `10`. Layer-0
`mlp.moe_input` still had high cosine (`0.982377650`), but layer-0
`mlp.shared_expert_output` averaged only `0.706940579`, and layer-1
`mlp.routed_expert_output` averaged `0.693710627` with one rank down at
`0.298821094`. Layer-2 `post_attention` was already `0.655290126`, and
layer-2 `post_mlp` was `0.756801243`. Promote expert/shared-expert
materialization, expert shard ordering, and internal-router visibility as the
next source target. Reject another unchanged ladder, generated-source diff,
metadata-only diff, or allreduce-only fix as the next step.

The router top-k trace shifts the active MoE GGUF correctness target upstream
from expert output alone to internal-router/gate materialization. The `.20`
router trace pair
`moe35b_hf_tp4_routertopk_eager4k_dot20_20260628T012050Z` and
`moe35b_gguf_tp4_routertopk_eager4k_dot20_20260628T012050Z` compared `480`
rank/layer/event rows. Exact ordered top-k matches were `0`; top-1 matched in
`268/480` rows; average set overlap was `5.933 / 8`, with a minimum overlap of
`1 / 8`. Event-0 layer-0 already differed: HF selected
`222,223,186,73,199,20,175,92`, while GGUF selected
`223,73,199,234,161,92,222,186`. Promote router input, gate tensor loading,
`e_score_correction_bias`, and top-k scoring as the next source target. Reject
expert-kernel-only or reduce-only debugging until selected experts match.

The follow-up gate-logit trace rejects `e_score_correction_bias` /
`exp_probs_b` as the live MoE GGUF blocker for the current TP4 profile. The
`.20` HF/GGUF gate trace pair
`moe35b_hf_tp4_gatelogits_eager4k_dot20_20260628T020000Z` and
`moe35b_gguf_tp4_gatelogits_eager4k_dot20_20260628T020000Z` instrumented the
internal `DefaultMoERunner` gate call immediately after
`router_logits, _ = self.gate(hidden_states)`. Both paths reported
`e_score_correction_bias=None`. Across `192` rank/layer/event rows, exact
top-16 matches were `0`, top-1 matched in `108` rows, average router-logit
cosine was `0.998827477`, and minimum router-logit cosine was `0.995624130`.
The paired boundary trace shows the gate input was already off before top-k:
layer-0 MoE input cosine was `0.982377650`, layer-1 was `0.935772504`, and
layer-2 was `0.903625660`. Promote upstream hidden-state trajectory drift as
the active correctness target. Reject a bias-only, gate-logit-only, or
expert-output-only fix until layer-0/layer-1 attention, GatedDeltaNet,
residual-add, and post-attention norm are compared and brought closer to HF.

Mining the same boundary traces gives the current first-drift map. Layer-0
matches HF through `pre_input_layernorm`, `post_input_layernorm`,
`linear_attn.input_hidden`, `linear_attn.mixed_qkvz`, `linear_attn.ba`, and
`linear_attn.core_attn_out_raw` with cosines at or effectively equal to `1.0`.
The first notable layer-0 direction drop is `linear_attn.out_proj_input` at
`0.878082862`; after the output projection the residual is closer but still off
(`post_attention` `0.988626168`), and the first MoE gate input is
`0.982377650`. Layer-0 MoE then amplifies the drift:
`mlp.shared_expert_output` is `0.622324722`, and `post_mlp` is `0.849558298`.
Layer 1 is therefore downstream of an already-misaligned residual. The active
source target is now layer-0 output-projection input construction /
shape-stride semantics and first-MoE route amplification, not layer-1-first
debugging or another unchanged benchmark ladder.

The layer-0 route-forcing diagnostic confirms that sparse top-k amplification
is not just a trace artifact. Forcing the GGUF layer-0 event-0 final-row route
to the HF route moved layer-0 `post_mlp` agreement from the earlier
`0.849558298` boundary to `0.977170526` and carried that improvement into
layer-1 input. The branch still did not flip back to HF because downstream
layers diverged again, so this is a diagnostic promotion rather than a
benchmark promotion. The next correctness experiment should replay HF routes
for layers `0` through `2` under the same faithful branch prompt, then extend
to `0` through `5` only if needed. Reject another unchanged MoE GGUF ladder
until branch correctness is restored.

The multi-layer route-replay diagnostics reject route forcing as a sufficient
GGUF MoE fix. A valid l0-l2 every-call replay and a valid l0-l5 every-call
replay both confirmed HF routes in the request trace, but neither restored the
HF `Thinking Process` branch. Both moved the failure into a blank/newline token
path instead. Route replay improves residual agreement, for example l0-l5
replay reaches layer-5 `post_mlp` cosine `0.928562586` and layer-10 `post_mlp`
cosine `0.907384666`, but layer-2 `post_attention` remains low at
`0.646410402`. Promote layer-1/layer-2 attention, GatedDeltaNet contribution,
output-projection input construction, and residual-add semantics as the next
source slice. Reject further broad route replay and any performance ladder
until branch correctness returns.

Mining the valid l0-l5 route-replay trace narrows the next source target to
Qwen3.5 linear-attention output-projection input construction. Route replay
improves sparse-MoE residual agreement, but it does not repair
`linear_attn.core_attn_out_normed -> linear_attn.out_proj_input ->
linear_attn.out_proj_output`. Layer-2 `linear_attn.out_proj_input` remains low
cosine (`0.665483755`) and layer-2 `post_attention` remains `0.646410402`
after l0-l5 replay. An exhaustive local 8-head permutation probe on the
`out_proj_input` row chose identity as the best permutation for layers `0`,
`1`, `2`, and `5`, rejecting a simple local head-order fix. The next diagnostic
must trace the full final-token all-head vector before and after the
reshape/rearrange into `out_proj_input`.

The full-token output-projection trace confirms that the GGUF MoE blocker is
layout-adjacent but rejects a naive global conversion. The HF control
`moe35b_hf_tp4_fulltokenoutproj_eager4k_dot20_20260628T022943Z` returned the
expected `Thinking Process` branch, while the GGUF trace
`moe35b_gguf_tp4_fulltokenoutproj_eager4k_dot20_20260628T022943Z` returned the
known wrong `Here's` branch. In that pair, the current layer-0
`linear_attn.out_proj_input` first-rank cosine is `0.878082862`, but the
alternate heads-first final-token view reaches `0.999999746`. Layer 2 shows the
same pattern: current `out_proj_input` first-rank cosine is `0.505315936`, while
the alternate heads-first final-token view is `0.856964249`. This promotes the
shape/stride contract around `core_attn_out` and `out_proj_input` as a real
source target. A copied state-conversion patch
`moe35b_gguf_tp4_stateconvert_eager4k_dot20_20260628T024100Z` consumed
`VLLM_QWEN35_GGUF_LINEAR_ATTN_STATE_CONVERT=1` and globally reshaped the
post-norm value output as heads-first/token-second before `out_proj`; it
returned a different wrong branch and collapsed layer-0 `post_attention`
cosine to `0.210117847`. Reject global post-norm heads-first reshaping as a
fix. The next scalable source slice is sequence-wide layout/stride
instrumentation before norm, after norm, before flatten, and after flatten.

Two follow-up variants also reject simple activation-view fixes. The pre-norm
state-conversion run
`moe35b_gguf_tp4_prenormstateconvert_eager4k_dot20_20260628T031000Z` converted
the GDN output before flattening and before pairing with `z`; it returned the
wrong `SpaceItemSpaceItem` branch and dropped layer-0 `post_attention` cosine
to `0.112992007`. The paired
`moe35b_gguf_tp4_postnorm_nooutprojinv_eager4k_dot20_20260628T032000Z` run
combined post-norm activation conversion with disabling the GGUF inverse-column
conversion for `linear_attn.out_proj.weight`; it returned the wrong
`ymar Coll` branch and made layer-0 `post_attention` cosine negative
(`-0.096042309`). The close heads-first final-token trace remains useful, but
it is not a standalone fix. The next source slice must inspect the combined
contract among GGUF V-head weight reordering, GDN output storage, `z`/norm
pairing, and output-projection column layout.

The static linear-attention weight contract probe narrows that source slice
further. Direct q/k rows match HF, while `v`, `z`, `conv_v`, and
`linear_attn.out_proj.weight` match HF only through the existing inverse
value-head layout rules. Layer-0 mean absolute differences after the correct
mapping are effectively zero: `v` `6.570227e-12`, `z` `5.64922848e-12`,
`conv_v` `0`, and `out_proj` `6.73216864e-12`. This rejects another broad
weight-column or head-row patch as the next move. Raw `ssm_a`, `ssm_dt.bias`,
`ssm_alpha.weight`, and `ssm_beta.weight` are not direct HF mirrors, but that
is a loader-semantics problem rather than proof of a fresh raw GGUF tensor
layout bug. The next useful check is after the active loader transforms, at
runtime graph inputs or compiled graph values.

The profiling readout now points to whole-graph replay rather than a single
already-timed Python-level op. The comparable GGUF and HF graph dumps have the
same high-level counts for GDN, FusedMoE, unquantized GEMM, qkvz/ba/out
projection, reshape/view/contiguous, and all-reduce patterns. The replay gap
remains large: GGUF TP4 FULL graph replay averages `12.054565` ms while the HF
release-patch control averages `7.887792` ms. Earlier request-window timing
already rejected the steady GDN custom op, steady FusedMoE runner body, logits
processing, and sampler as standalone bottlenecks. The active performance
target is therefore graph-visible value/stride inputs, kernel selection, and
replay drift inside the same high-level graph shape.

The compile-cache comparison sharpens that target. GGUF and HF both emit the
same generated-kernel family counts, including `1360` `rocm_unquantized_gemm`
calls, `832` `moe_forward_shared` calls, `200` Triton compile sites, `120`
`gdn_attention_core` calls, and matching KV-cache / attention update markers.
The generated Python surface is close in size and line count, and the first
normalized source diffs are internal FX/ATen symbol numbering rather than
missing operation families. The remaining profiler signal is lower-level:
GGUF has `97` unique `.best_config` hashes versus HF's `89`, with only `60`
common. Treat the MoE GGUF performance gap as a graph-input metadata /
autotune-config / replay-scheduling problem until proven otherwise.

The latest cache-key diff also explains why AWQ should not be re-tested as a
standalone hunch. The current GGUF graph has `VLLM_USE_TRITON_AWQ=true` while
the HF control has it false, but earlier source-level AWQ-false work already
proved the flag can be normalized without moving MoE GGUF into the release
band. Keep the flag normalized for clean comparisons, but do not spend another
benchmark ladder on AWQ alone.

The graph-input and lowered tensor metadata manifests further narrow the
performance gap. HF and GGUF have identical forward-signature manifests:
`576` inputs and the same type/shape counts. The generated-kernel tensor
metadata manifests are also identical: `2112` metadata entries and `112`
unique shape/stride/device strings on each side. After normalizing
`.best_config` files by path and removing timing/cache-hash noise, only `14`
of `136` config paths choose different actual autotune parameters. The next
performance experiment should force those GGUF config choices to the HF values
and measure graph replay timing. If that does not move the replay gap, stop
chasing compile selection and inspect runtime value/state differences.

The HF best-config preseed probe rejects those `14` normalized best-config
differences as the primary MoE GGUF performance blocker. In the disposable
`.20` run
`moe35b_gguf_tp4_bestconfig_hfpreseed_dot20_20260628T040600Z`, `9` of the
`14` mismatched paths retained the HF parameter choice and `5` reverted to the
GGUF baseline choice. The normal warmup -> strict -> `c1_2000` -> `c1_10000`
ladder still landed in the low GGUF band: strict `82.686`, `c1_2000`
`84.181`, and `c1_10000` `74.301` backend TPS. FULL replay timing improved
only from the earlier GGUF `12.054565` ms average to `11.771824` ms, still far
from the HF `7.887792` ms average. Stop treating compile-cache best-config
choice as the main gap. Promote runtime value/state or replay scheduling
inside the already-matching high-level graph as the next performance slice.

The replay-control generated-code parity check narrows that slice further.
Using the release-patched HF comparator, both HF and GGUF use the fused `3088`
graph-input family; the earlier stock-HF `3072 + 16` metadata contrast is not
a live source target. The replay-control roots also contain identical generated
Python source bodies as a content multiset: `208/208` generated `.py` hashes
match, with `0` GGUF-only and `0` HF-only source bodies. Per-rank replay timing
is balanced, so this is not a single slow rank. The remaining MoE GGUF gap is
therefore lower than generated Python code selection and static graph-input
shape metadata: inspect runtime values/state, graph-input contents, activation
layout after loader transforms, and replay-time scheduling around those values.

The graph-input value-signature probe confirms structural replay parity but
does not provide a standalone fix. HF and GGUF value-signature runs each logged
`8092` lines, `384` capture headers, and `288` replay headers, with an empty
replay-header diff. The sampled values differed in expected weight/activation
tensors, but the replay call structure stayed aligned. Treat this as a
narrowing result: the remaining MoE GGUF blocker is active values/state,
GatedDeltaNet layout semantics, or lower replay-time behavior inside the same
graph, not a missing graph family.

Do not revive the grouped TopK8 fastpath rejection as the primary theory. The
value-signature run showed `4` GGUF one-token activations and `204` grouped
rejections, while the first HF value-signature run showed no grouped
rejections. Later matched-debug evidence shows the release TopK8 helper is
one-token-only and HF can also show grouped `shape_or_layout` rejections when
launched with the same debug path. Grouped rejection logs are diagnostic noise
unless a candidate also changes branch correctness or settled replay timing.

The current fused-`qkvzba` post-norm head-gather candidate is rejected. It
tested the close alternate heads-first final-token trace as an opt-in
`VLLM_QWEN35_GGUF_OUTPROJ_HEADS_FIRST_GATHER=1` patch on the current fused
GGUF branch. The short two-token branch prompt returned a non-coherent glyph
instead of the HF `Thinking Process` branch, so no benchmark ladder was run.
The close alternate view remains useful evidence of a layout-adjacent boundary,
but it is not a direct output-projection gather fix.

The GDN boundary layout-metadata trace rejects a visible Python tensor layout
bug at that same boundary. On the current fused GGUF path, selected layers
`0`, `1`, `2`, `5`, and `10` expose contiguous `core_attn_out_raw`,
`z_for_norm`, `core_attn_out_normed`, `out_proj_input`, and `out_proj_output`
tensors with expected strides; only the intentional final-token row view has a
nonzero storage offset. This means the earlier value-direction drift is not
explained by `out_proj_input` being non-contiguous or storage-offset shifted.
Move the MoE source slice lower, toward the `gdn_attention_core` custom-op
write/consume contract, recurrent GDN state/value semantics, or replay-time
scheduling around those values.

The replay bucket re-read adds a length-dependent performance signal. GGUF
FULL replay is already slower than HF early in the decode window (`11.220266`
ms versus `7.601556` ms in the first 512-call bucket), and the gap widens late
(`14.970377` ms versus `8.891808` ms in the final observed buckets). The
HF-best-config preseed only trims about `0.2` ms from those buckets. Treat the
MoE gap as both a steady replay overhead and a long-decode drift problem:
future probes need early/late state signatures, not just one-token branch
checks or static compile-cache comparisons.

The sparse call-set state trace gives us that probe shape, but not a standalone
fix. The GGUF-only eager diagnostic on `.20` captured calls `2`, `64`, `128`,
`256`, and `511` and showed recurrent GDN/SSM state norms growing from
hundreds or tens into the multi-million range by late decode, while downstream
normalized output/residual norms stayed bounded. That supports early/late
state inspection as the next source slice, but it does not prove a GGUF-only
bug without the same call-set probe on HF. Tighten prefix matching first; the
current substring filter overmatches layer names and is useful only as a
scratch diagnostic.

The counter-fixed clean-HF call-set comparator rejects recurrent GDN/SSM state
growth as a GGUF-only root cause. The first HF attempt that reused a
GGUF-derived model file was rejected, and the first clean-HF patch had a trace
counter bug that captured only calls `0`, `1`, and `2`. After moving the
decoder trace counter increment outside the trace-active block, the tuned HF
TP4 diagnostic on `.20` completed the same 512-token capped prompt and captured
calls `0`, `1`, `2`, `64`, `128`, `256`, and `511`. Early HF/GGUF signals are
effectively identical: layer-0 SSM max norm is `929.961182` HF versus
`929.961365` GGUF at call `2`, and post-attention layernorm / post-MLP values
match at the same call. Late recurrent state also stays in the same order of
magnitude: layer-10 call `511` is `10110661.000000` HF versus
`10560669.000000` GGUF, while layer-20 call `511` is `28720954.000000` HF
versus `34658824.000000` GGUF. The clearer late divergence is downstream:
GGUF post-attention layernorm norms are lower at call `511` for layers `0`,
`10`, and `20` (`0.724`, `0.760`, and `0.752` of HF respectively), and
layer-10 out-projection input is `0.620` of HF. Promote late downstream
attenuation plus graph replay/kernel/runtime behavior as the next source slice.
Reject broad recurrent-state rescaling, another standalone SSM-state patch,
and using eager call-set traces as throughput evidence.

The replay value-signature mining pass adds one concrete next target without
promoting a fix. A read-only `awk` grouping pass over the existing HF/GGUF
replay value-signature logs found no missing replay-input family and confirmed
that several static/vector arguments are exactly equal. The largest repeated
delta is an activation-shaped family:
`argc=11 tensor=8 arg=4 shape=(16, 2048)`, with `36/36` paired records
differing and average sampled absolute sums of `349.976942` for HF versus
`92.737499` for GGUF. The follow-up generated-source map shows this exact
argument is an output/mutation buffer inside a compiled MoE expert fragment
around `moe_forward_shared`, all-reduce, and `rocm_unquantized_gemm_1`, not a
semantic consumed activation. Treat the mapped fragment as the next source
boundary, but reject the stale output-buffer value itself as a patch target.
Do not patch raw weight-shaped sampled diffs, and do not rerun the full
benchmark ladder until a candidate changes graph replay timing or the consumed
inputs/returned buffers around this fragment.

The pre/post replay value-signature probe confirms that conclusion. The `.20`
GGUF MoE TP4 diagnostic
`moe35b_gguf_tp4_prepost_valuesig_dot20_20260628T064605Z` added balanced
`384` `replay` and `384` `replay_post` records after a coherent short request.
All meaningful pre/post value changes concentrated in output/mutation slots:
`argc=12 arg[5]` (`80` pairs), `argc=17 arg[5]` (`40` pairs),
`argc=11 arg[4]` (`36` pairs), and `argc=9 arg[4]` (`4` pairs). The matching
metadata identifies those slots as large `(tokens, 2048)` activation/output
buffers. Promote the stale-mutation-buffer interpretation and stop treating
those value deltas as semantic input mismatches. The remaining MoE GGUF target
is decode replay/runtime cost: generated graph bodies and headers match, but
GGUF FULL decode replay remains much slower than HF. The current TopK8 helper
is intentionally one-token-only (`hidden_states.size(0) == 1`), so grouped
multi-token `shape_or_layout` rejections during capture/prefill are expected
diagnostic noise unless a future candidate changes settled replay timing or
decode correctness.

The active-layout decode comparator rejects a visible expert-layout mismatch
as the primary MoE GGUF gap. The matched `.20` HF/GGUF probes
`moe35b_hf_tp4_decode_active_layout_dot20_20260628T070537Z` and
`moe35b_gguf_tp4_decode_active_layout_dot20_20260628T070537Z` both activated
the release TopK8 one-token fastpath on all four TP workers. Both paths logged
the same active layout: `hidden_shape=(1, 2048)`,
`hidden_stride=(2048, 1)`, `w1_shape=(256, 256, 2048)`,
`w1_stride=(557056, 2176, 1)`, `w2_shape=(256, 2048, 128)`, and
`w2_stride=(262144, 128, 1)`. They also both logged `204` grouped
capture/prefill rejections, confirming that those rejections are not
GGUF-only. The replay timing gap remained: HF `FULL num_tokens=1` averaged
`7.776122` ms while GGUF averaged `11.212760` ms over `1020` calls. Promote
the matching active-layout evidence and move the source target below
Python-visible tensor metadata: GGUF weight materialization, generated kernel
runtime behavior, memory movement, or replay state.

The RCCL-only launch parity probe is a minor hygiene promotion, not the fix.
The HF control uses `VLLM_NCCL_SO_PATH=/rccl-overlay/install/lib/librccl.so.1`;
the current GGUF timing script did not. Copying the full release native bundle
into the GGUF patch path was rejected because the GGUF launch mounts
`<validation-nvme-root>` read-only and the swiglu native copy path tried to write
there. A narrowed RCCL-only composite bundle
`qwen35moe-gguf-plus-rcclonly-20260628T072640Z` served correctly and improved
GGUF `FULL num_tokens=1` replay from `11.212760` ms to `10.864359` ms, but it
still remained far behind HF's `7.776122` ms. Carry RCCL parity forward for
future GGUF probes, but reject missing `VLLM_NCCL_SO_PATH` as the primary
performance root cause.

The force-unquantized-linear probe rejects regular GGUF linear dispatch as the
primary remaining gap. Patch bundle
`qwen35moe-gguf-force-unquant-linear-20260628T073755Z` forced every GGUF
`LinearBase` and `VocabParallelEmbedding` to use unquantized methods and kept
the RCCL-only parity from the previous probe. It loaded successfully and logged
`760` forced-linear selections across attention projections, linear-attention
output projections, and shared-expert linears. The short request stayed
coherent, but `FULL num_tokens=1` replay averaged `10.871309` ms, effectively
unchanged from RCCL-only `10.864359` ms and still far from HF `7.776122` ms.
This narrows the next target below MoE layout, RCCL launch parity, and regular
GGUF linear method dispatch.

The generated-kernel and replay-pointer probe rejects two more visible
boundaries as the MoE GGUF root cause. `rocprofv3 --attach` successfully
attached to the running GGUF worker but emitted no usable trace files or
summary files, so attach-mode profiling is not a reliable probe for this
server path. The matched HF and GGUF runs both produced `76` generated-kernel
perf records and `216` generated Python files, and the generated-kernel IDs
paired one-for-one. The top GGUF-slower generated-kernel perf entries had
byte-identical generated Python source to HF. A pointer-signature diagnostic
then showed that top-level FULL replay `input_ids` and `positions` have the
same shape, stride, contiguity, and alignment pattern in HF and GGUF:
`input_ids` are contiguous with `ptr_mod256=0` and `ptr_mod4096=3584`;
`positions` use stride `(2049, 1)` with `ptr_mod256=0` and
`ptr_mod4096=512`. The replay gap still held: HF averaged `8.003061` ms while
GGUF averaged `10.827265` ms for FULL `num_tokens=1` in the pointer
diagnostic. Promote internal captured-graph/model-state inspection below the
`CUDAGraphWrapper` argument boundary. Reject generated-kernel source mismatch,
top-level replay pointer alignment, and attach-mode `rocprofv3` as the next
primary target.

The active fastpath value-signature comparator narrows the target further.
Scratch patch `fused-moe-active-value-debug-20260628T083222Z` logged active
TopK8 state from matched `.20` HF and GGUF TP4 runs. Both emitted `320`
active signature records, `80` per TP worker. The first one-token active MoE
call matched exactly at the visible boundary: top-k IDs
`101,196,108,249,216,197,135,242`, the same routing weights, the same first
hidden-state signature, the same visible hidden/`w1`/`w2` shape and stride,
and matching selected expert-weight signatures. Calls `1` through roughly `8`
stayed close with small FP16-level drift; clear routing divergence appeared
around calls `9` and `10`. The GGUF short request stayed coherent and replayed
in the same slow band (`508` FULL calls, `10.893246` ms average). The HF
completion request under this CPU-copy diagnostic hung after readiness with a
shared-memory wait and no GPU progress, so that request is rejected as a
diagnostic-method failure. Promote upstream captured-state/value drift as the
next source target; reject immediate active MoE expert layout, first-call
routing, and selected expert-weight materialization as primary roots. Future
probes must avoid CPU copies inside CUDA graph capture, for example by using
pinned buffers, device-side summaries, or an explicitly non-timing eager
diagnostic.

The explicit HF eager follow-up rejects that eager fallback as implemented.
Run `moe35b_hf_tp4_active_value_eager_dot20_20260628T085833Z` loaded the HF
TP4 release profile with `--enforce-eager`, P2P-on, FP16, and RCCL parity, but
the bounded request hung after readiness with `0%` GPU use, empty completion
output, no active-value records, and repeated shared-memory wait messages. The
matching GGUF eager request had already completed coherently and emitted `512`
active-value records, but no valid call-index comparison exists because the HF
eager diagnostic failed before request execution. Reject HF eager CPU-copy
active-value logging as a repeat path. The next source slice should use
lower-overhead device-side summaries, pinned-buffer snapshots, or narrower
router-input metadata that avoids tensor CPU copies in HF request execution.

The MoE launch-wrapped `rocprofv3` probe narrows the profiling failure to
shutdown/artifact collection. Run
`moe35b_gguf_tp4_rocprof_kernel_short_dot20_20260628T092554Z` preserved the
release image, FP16, P2P-on, TP4 GGUF path, and a read-only `libdw.so.1` mount.
The server reached readiness and returned a coherent 64-token short completion,
and the one-token TopK8 MoE fastpath became active after graph capture.
However, Docker stop delivered signal 15, the profiler waited for children,
the container exited `137`, and no profiler files were emitted under the
mounted output directory. Promote the `libdw.so.1` mount and launch-wrapper
readiness evidence; reject Docker-stop shutdown and the current run as profiler
evidence. Any future profiler route needs a self-terminating wrapper or
reduced/offline worker path that lets `rocprofv3` flush cleanly.

The self-terminating `rocprofv3` wrapper rejects broad server profiling more
strongly. Run
`moe35b_gguf_tp4_rocprof_selfterm_dot20_20260628T095756Z` started the GGUF TP4
server under `rocprofv3`, waited for readiness, sent the short deterministic
completion request, and then signaled the server child from inside the
container. The request was coherent and the container exited status `0` with
VRAM released, but the profiler output directory remained empty. This rejects
Docker-stop shutdown as the only explanation for missing profiler output. It
also records a useful shape boundary: graph-capture and multi-token shapes
reject the Qwen TopK8 c1 MoE fastpath, while `num_tokens=1` activates it on
all four TP workers. The route also contaminates timing: wrapped GGUF FULL
replay averaged `66.231362` ms, versus the normal GGUF force-unquant
`10.871309` ms and HF control `7.887792` ms. The next useful source slice
should stop broad launch-wrapped profiling and instead build a reduced
worker/replay reproducer or lower-overhead C/C++ or device-side summaries for
the FULL one-token decode path.

The corrected packed-weight contract audit rejects another tempting repeat
path. Run
`moe35b_gguf_packed_weight_contract_20260628T104313Z` reconstructed layer-0
HF and GGUF TP4 packed views using the actual Qwen3.5-MoE linear-attention
geometry: `16` key heads, `32` value heads, and `128`-wide linear heads. The
audit compared transformed `in_proj_qkv`, `in_proj_z`, `in_proj_b`,
`in_proj_a`, `A_log`, `dt_bias`, `conv1d`, `out_proj`, TP4 packed `qkvzba`
shards, and MoE expert/shared-expert tensors. It found `0` review rows across
`36` checks, and the intentionally swapped gate/up expert negative control was
rejected. Promote the result as evidence that the current low-band MoE GGUF
path is not explained by a layer-0 raw tensor transform or TP4 packed-shard
assembly error. Reject repeating static Qwen3.5 linear-attention permutation
work until a later-layer or runtime-only packed-state trace points back there.

The current HF/GGUF graph-surface compare also points below high-level graph
shape. Run
`moe35b_gguf_graph_surface_compare_20260628T104424Z` compared the active GGUF
rank-0 computation graph with the HF replay-control graph. The checked counts
matched: `81` submodules, `31` folded `qkvzba` weight shapes, `30`
`rocm_unquantized_gemm` calls producing `3088`, `40` producing `2048`, `60`
GDN core calls, `162` all-reduces, `80` fused-MoE calls, and `120`
shared-MoE calls. Promote runtime materialization, generated-kernel/cache
factors, captured recurrent/KV state, or a reduced FULL one-token replay
worker as the next source target. Reject more graph-shape/op-count audits as
the likely near-term fix.

The MoE TopK8 tile experiments reject a simple fastpath tile-size explanation.
The BI8, BI16, and BI32 overlays all kept the GGUF TP4 path coherent and
strict-valid, but stayed in the low band. BI16 was the best c1 variant
(`86.952` c1_2000 and `76.483` c1_10000), while BI32 was the best strict
variant (`85.607`). That is still far below native TP4 (`109.283` c1_10000
release-time fixed-token and `113.196` to `115.995` strict in the post-v0.2
repeatability report). Promote the result as evidence that the one-token
TopK8 tile can move only a small amount of TPS; reject BI8/BI16/BI32 tile
patches as release-candidate fixes.

The release-image native SwiGLU overlay is not a drop-in TP4 GGUF MoE fix.
`qwen2_moe_interleaved_swiglu_20260608.py` is guarded for the shared-expert
path with TP8 and shape-specific checks, including `hidden_size == 5120`,
`intermediate_size == 17408`, `tp_size == 8`, and a native call shape of
`(4352, 5120)`. That boundary should be documented, but forcing it into TP4
GGUF would be unsafe. The remaining MoE GGUF target stays below graph shape,
regular GGUF linear dispatch, first-call active expert materialization, and
simple TopK8 tile geometry.

The exact HF release TopK8 fastpath is also rejected as the missing MoE GGUF
TP4 lever. The `.20` and `.30` runs
`moe35b_gguf_tp4_hf_release_fastpath_exact_dot20_20260628T1330Z` and
`moe35b_gguf_tp4_hf_release_fastpath_exact_dot30_20260628T1330Z` mounted the
same fastpath source used by the HF release TP4 control, with checksum
`66f63f74406c2a805e78eeb28e9dac76c3f98bc4ce9046cd3be5d604224a0a0e`.
Both lanes reached health, activated the one-token TopK8 fastpath after graph
capture, and completed the normal warmups -> strict -> c1 tiers. Results stayed
low-band: `.20` strict `81.036`, c1_2000 `84.783`, c1_10000 `74.793`; `.30`
strict `83.302`, c1_2000 `84.751`, c1_10000 `74.762`. Promote this as a closed
comparability test and as evidence that `.30` is equivalent for this source
slice. Reject exact fastpath source mismatch, lane choice, and simple TopK8
helper selection as the remaining native-parity explanation.

The MoE TopK8 active pointer diagnostic rejects visible storage alignment as
the missing native-parity lever. Run
`moe35b_hf_tp4_active_ptr_dot20_20260628T1340Z` compared against
`moe35b_gguf_tp4_active_ptr_dot20_20260628T1340Z` using pointer-debug source
checksum `f5e20bd6e450371dbf36db9f6a2908da873aebf88c8cb367a2a6345568b19270`.
Both routes returned coherent text and exposed the same active tensor storage
contract: zero storage offsets, `hidden` / `topk_ids` / `topk_weights`
256-byte alignment, and 4K-aligned packed `w1` / `w2`. HF still replayed faster
in the diagnostic request (`7.518159` ms average versus GGUF `9.979224` ms).
Reject `w1` / `w2` pointer alignment, top-k pointer alignment, and storage
offsets as the primary MoE GGUF TP4 bottleneck. Keep the next target below
visible tensor surfaces: generated replay body, runtime materialization around
expert prepare/finalize, captured state, or a reduced FULL replay worker.

HF TorchInductor cache preseeding is also insufficient as a MoE GGUF TP4 fix.
The HF and GGUF pointer diagnostics generated identical `*.py` inductor source
files, but differed in autotune metadata. Run
`moe35b_gguf_tp4_hf_inductor_cache_dot20_20260628T1405Z` copied the HF
`torchinductor_root` into the GGUF run before launch and retained normalized
HF `*.best_config` files exactly. The first request had a large outlier and
averaged `13.746053` ms; the second request settled to `9.190717` ms over
`864` replay rows. That is a modest improvement over the normal GGUF pointer
diagnostic (`9.979224` ms), but still above HF (`7.518159` ms). Promote the
cache-preseed result as a closed diagnostic and reject best-config selection
alone as the native-parity blocker.

The CUDA graph replay input-boundary diagnostic rejects request-stream drift as
the current MoE GGUF TP4 explanation. Run
`moe35b_hf_tp4_graph_input_dot20_20260628T1420Z` compared against
`moe35b_gguf_tp4_graph_input_dot20_20260628T1420Z` with
`cudagraph-input-debug-20260628T1420Z`, checksum
`5a55161238a3b7579d94490d7626453b82fa1f6617357119665eb5dfae58ad32`. The first
eight one-token FULL replay calls had identical `input_ids` and `positions`
between HF and GGUF: token IDs `90700`, `8340`, `25`, `271`, `16`, `13`, `220`,
and `2972`, with positions `20` through `27`. Tensor shapes, strides, storage
offsets, and pointer buckets also matched at this boundary. The response
message hashes and usage counts also matched: same reasoning hash, same empty
content hash, `20` prompt tokens, `64` completion tokens, and `84` total
tokens. GGUF still replayed
slower (`10.843224` ms average over `252` FULL rows) than HF (`8.031084` ms
over the same row count). Promote the replay input boundary as closed for this
prompt. Reject tokenizer/request stream divergence, response divergence, and
top-level `input_ids`/`positions` surfaces as the primary bottleneck. Keep the
source target below this boundary: captured lower-level state, runtime
materialization around expert prepare/finalize, recurrent/KV state plumbing,
or a reduced FULL replay worker.

The first runner metadata probe is rejected as instrumentation, not as model
evidence. Run `moe35b_hf_tp4_runner_meta_dot20_20260628T1505Z` used a
metadata-only HF runner patch, checksum
`853f61cdc813583a520ea64d28858e497e0dfec64de6b28fdfa41f47938d0022`. The first
patch missed an `os` import and failed worker initialization; the corrected
patch reached readiness, but filled its log during CUDA graph capture with
dummy values such as `input_ids=[0]` and `positions=[0, 0, 0]`. The request then
stalled with no GPU activity and a shared-memory wait. Reject this patch and do
not run the GGUF half. Promote the lesson that runner-level probes must be
request-replay gated or moved into a reduced replay worker; `FULL` mode plus
`num_tokens_padded == 1` is not enough to distinguish capture dummy inputs from
real request replay.

The latest active-pointer `.kernel_perf` recheck strengthens the rejection of
TorchInductor artifact metadata as the current root cause. Comparing
`moe35b_hf_tp4_active_ptr_dot20_20260628T1340Z` against
`moe35b_gguf_tp4_active_ptr_dot20_20260628T1340Z` found `76` joined
`.kernel_perf` paths on each side. Summing all values produced HF
`2.060957000` and GGUF `1.980158011`, a `0.961x` GGUF/HF ratio, even though
live replay favored HF. Reject `.kernel_perf` aggregate timing and individual
small pointwise/reduction metadata deltas as the next source target. Continue
toward request-window kernel mix or a reduced replay reproducer.

The replay value-signature decode audit closes another tempting repeat path.
Run `moe35b_valuesig_replay_decode_audit_dot20_20260628T1515Z` reinterpreted
the older HF/GGUF replay value-signature artifacts and the GGUF pre/post
signature run. FULL one-token replay rows had `124` rows but `argc=0`; all
available tensor-value diffs came from PIECEWISE rows, including HF/GGUF
`num_tokens=16` groups and GGUF pre/post `num_tokens=24` groups. Promote this
as a closed interpretation of the old value-signature evidence. Reject those
diffs as proof of the current one-token decode bottleneck. The next useful
source slice remains request-window kernel mix, captured-state inspection
without Python CPU copies, or a reduced FULL replay reproducer.

The C/C++ HIP graph `LD_PRELOAD` route is rejected for the full vLLM server.
Tool `hip_graph_trace_20260628T1525Z` passed a standalone one-kernel HIP graph
smoke test, but the GGUF TP4 vLLM run
`moe35b_gguf_tp4_hipgraph_dot_dot20_20260628T1540Z` failed during KV-cache
initialization with a HIP unknown error before readiness. The shim emitted no
graph node files or DOT files and recorded zero graph instantiate / launch
counters for the worker processes. Promote only the small-shim smoke result and
the operational lesson. Reject full-server `LD_PRELOAD` graph interposition as
a repeat path; keep the target on a reduced FULL replay worker or a
request-window trace that does not preload into every process.

The MoE TP4 GGUF performance blocker has a promotion candidate now. The
release-overlay combo bundle
`qwen35moe-gguf-release-overlay-combo-20260628T1615Z` combines the full release
overlay stack, native runtime setup, GGUF loader/model repairs, and the exact
HF release TP4 fastpath source. Under the native `moe35b_tp4_fullbar_p2pon`
profile, P2P-on, FP16, `MAX_MODEL_LEN=131072`, default broad graph capture,
and normal warmups -> strict -> fixed-token ladder, it produced:

- `.20`: strict `118.754`, c1_2000 `119.896`, c1_10000 `112.747` backend TPS;
- `.30`: strict `119.917`, c1_2000 `120.781`, c1_10000 `113.605` backend TPS.

Promote the full overlay-composition hypothesis and `.30` as an equivalent
performing validation lane for this path. Reject the earlier exact-fastpath-only
low-band result as representative of the complete release-overlay GGUF route:
the standalone fastpath file was not enough, while the full release overlay
composition crosses the native TP4 band on both hosts. The next task is
reproducibility engineering: reduce the scratch combo to the minimal required
source bundle, write the deployment path, and rerun from that path before
public claims change.

The reduced MoE GGUF package path now reproduces the high-band result on `.30`.
Run `moe35b_pkg_dot30_20260628T180700Z` used the contract profile
`gguf-moe35b-tp4`, the binary GGUF probe, `moe35b_tp4_fullbar_p2pon`, TP4,
FP16, P2P-on, `MAX_MODEL_LEN=131072`, the pinned MoE minimal bundle manifest
`794cb8760003cfa7dafeb0f06ee01823e5e8eebb3f9d398d8bc8bd7697143f0b`, and the
normal warmups -> strict -> fixed-token ladder. It produced strict `119.508`,
c1_2000 `120.758`, and c1_10000 `113.271` backend TPS. Promote the reduced MoE
bundle as a working package-path candidate on `.30`. Reject `PYTHONPATH`-only
sidecar injection: the first package launch failed because installed
Transformers did not accept the GGUF MoE architecture string until
`modeling_gguf_pytorch_utils.py` was copied directly into the installed
Transformers paths. The remaining work for this slice is release engineering:
rerun clean package paths as needed on `.20` and `.30`, verify HF baselines,
verify mismatch failures, and document the exact user-facing path.

HF baseline protection is still open for the vNext package path. The generated
HF Dense profile was corrected toward the historical text-only comparator by
forcing `Qwen3_5ForCausalLM` with `--language-model-only`, adding the dense
release env set, and requiring an isolated HF release bundle with manifest hash
`9fb16c0edfd57d908f2bff6eb51063b6e8cf7d2c27de252bb4f691c67d2f5a84`. This
removed the no-bundle launch mistake and activated persistent all-reduce route
replacement, but it still did not reproduce the historical `.20` comparator
band. Current generated reruns were `.30` no-bundle `59.988` / `62.853` /
`54.294`, `.30` HF-bundle `63.232` / `65.051` / `55.985`, and `.20` HF-bundle
`62.962` / `64.991` / `55.945` backend TPS for strict / c1_2000 / c1_10000.
Reject the HF generated path as release-ready until it matches the earlier
text-only HF comparator (`70.353` / `70.978` / `66.428`). Promote the guardrail
work itself: the launcher validates HF/GGUF format separation, rejects
mismatches, saves argv, verifies args against runtime help, and keeps GGUF envs
out of HF launches.

Update: HF baseline protection recovered after the HF bundle was expanded to
mirror the old clean release patch-bundle composition. The first-pass reduced
HF bundle was too small; it activated persistent all-reduce routing but omitted
release-path model/platform/attention/utility/Fused-MoE overlay files that were
present in the historical control environment. The corrected HF bundle
manifest hash is
`d9e1168f67731ab66ee53671f89dfb31fd3c74c8f61e2f0d0b7c4f0704c298be`.
With that bundle, the generated `.20` HF Dense package path produced strict
`70.346`, c1_2000 `70.961`, and c1_10000 `66.428` using the shared release
runtime cache, and strict `70.168`, c1_2000 `71.316`, and c1_10000 `66.824`
with a fresh per-run runtime cache. Promote the expanded clean HF release
bundle as the HF baseline-protection requirement. Reject the reduced HF bundle
and reject cache reuse as the decode-TPS requirement; fresh cache pays startup
compile/prefill cost but still reaches the decode band after normal warmups.

HF MoE TP4/TP8 profile contracts now require the same expanded clean HF release
bundle. Local synthetic preflight generated and verified launch artifacts for
all five vNext profiles: HF Dense, HF MoE TP4, HF MoE TP8, GGUF Dense, and GGUF
MoE. Promote this as contract-level baseline protection and developer
serviceability, not as runtime MoE HF reproduction evidence. Runtime HF MoE
reruns remain required before claiming the whole vNext release package is
complete.

Developer serviceability is now a separate gate from benchmark promotion.
`qwen36-gfx906/verify_vnext_serviceability.sh` passed for all five profiles and
checks profile presence, image tag/digest pins, P2P-on state, FP16,
`MAX_MODEL_LEN=131072`, HF/GGUF env separation, patch-bundle manifest hashes,
and private host/path leakage without launching Docker or touching GPUs.
Promote this gate for sharing the image/profile path with other developers.
Reject any vNext package that cannot pass this read-only serviceability check.

HF MoE TP4 runtime reproduction remains unresolved for performance. The `.30`
generated package path with the expanded HF bundle but without the tuned MoE
config produced strict `95.824`, c1_2000 `96.158`, and c1_10000 `91.524`
backend TPS. After adding the tuned MoE config to the HF bundle, the corrected
path produced strict `96.159`, c1_2000 `97.031`, and c1_10000 `92.284`
backend TPS, with the strict run passing `finish_reason=stop` and
`qwen_gate_valid=true`. Promote the corrected path as functionally strict-valid
but reject it as release-performance-equivalent to the published HF MoE TP4
capped band. The exact historical TP4 performance lane still needs to be
recovered.

Matching the generic `deploy.sh` runtime wrapper shape did not recover HF MoE
TP4 performance. A deploy-shaped vNext wrapper that added ROCm target defaults,
`gfx906` arch exports, `OMP_NUM_THREADS=4`,
`TORCH_BLAS_PREFER_HIPBLASLT=0`, logging/privacy defaults, and privileged
container mode produced strict `94.860`, c1_2000 `95.613`, and c1_10000
`91.036` backend TPS on `.30`, with strict validity still passing. Promote
those defaults as serviceability/runtime-shape guardrails, but reject them as
the missing high-band performance factor. The next useful HF MoE task is to
derive a contract directly from the full `deploy.sh` emitted runtime env and
compose/package files rather than continuing one-off wrapper approximations.

Public image serviceability was verified independently from benchmark
promotion. A clean Docker pull of
`joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`
resolved to pinned digest
`sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e`,
and the image contained the baseline native runtime paths for persistent AR,
SwiGLU, and RCCL. Promote `CHECK_RUNTIME_IMAGE=1 CHECK_RUNTIME_PATHS=1
./verify_vnext_serviceability.sh` as the opt-in clean-developer image check.
Reject relying on locally renamed images or hidden native paths. This proves
the image/profile path is serviceable, not that every runtime profile has
recovered its target performance band.

The vNext launcher now has a CPU-only contract matrix gate. It creates
synthetic GGUF and safetensors byte signatures, generates launch artifacts for
all five profiles, verifies those artifacts, and confirms expected failures:
GGUF-under-HF, HF-under-GGUF, Dense-under-MoE, MoE-under-Dense, GGUF env leakage
into HF, and patch-bundle hash mismatch. Promote
`qwen36-gfx906/verify_vnext_contract_matrix.sh` as the cheap local gate for
launcher/profile changes. It now also runs synthetic deploy/profile parity for
HF MoE TP4 and TP8. Reject profile changes that only pass metadata
serviceability but have not proven valid artifact generation, deploy parity,
and fail-closed mismatch behavior.

Deploy/vNext parity now has a mechanical check. The MoE vNext profiles were
aligned with the published deploy path by using
`/opt/vllm_tuned_moe_configs` for `VLLM_TUNED_CONFIG_FOLDER`; the generated
entrypoint copies bundled tuned configs there from the mounted patch bundle.
`qwen36-gfx906/verify_vnext_deploy_profile_parity.sh` compares a captured
`.deploy.runtime.env` against a vNext profile and generated artifact. Promote
this as the gate for deriving the remaining HF MoE high-band contract from
`deploy.sh` artifacts. Reject treating the tuned-config path correction as a
performance promotion until a clean HF MoE runtime rerun proves the band.

The first HF MoE TP4 vNext runtime reruns after deploy-path alignment did not
recover the high-band TP4 lane. On `.30`, the deploy-style tuned-config path
completed the normal warmups -> strict -> fixed-token ladder at strict
`93.013`, c1_2000 `93.839`, and c1_10000 `89.312` backend TPS. Adding a
generated deploy-compat runtime patch step, including the deploy-time custom-op
fallback and ROCm/GFX906 compatibility surface, produced strict `92.731`,
c1_2000 `93.272`, and c1_10000 `88.832`. In both runs the strict request
stopped and passed the Qwen gate, so the failure is performance parity, not
correctness. Promote both attempts as negative evidence. Reject further
one-off wrapper approximation; the next useful source task is to capture the
known-good `deploy.sh` runtime env/package artifacts and diff them mechanically
against the vNext launch artifact.

Patch-bundle hashes are not enough to prove overlay isolation. The generated
entrypoint must also know which files from a broad bundle are allowed to
activate for the selected profile. vNext now pins `PATCH_TARGET_GROUPS` in each
profile and group-gates patch-copy targets in the entrypoint. This fixes a
real source-contract weakness: the broad HF bundle could otherwise copy
dense-only persistent-AR/SWiGLU targets into HF MoE launches. Promote
profile-gated patch activation as a required serviceability rule. Reject file
presence as an activation rule.

The `.20` and `.30` HF MoE TP4 runtime reruns confirmed that profile-gated
patch activation is not just cosmetic. With `PATCH_TARGET_GROUPS="common moe
hf"`, the generated package path recovered the high-band TP4 lane on both
hosts: `.20` produced strict `115.105`, c1_2000 `115.933`, and c1_10000
`109.095`; `.30` produced strict `114.409`, c1_2000 `115.685`, and c1_10000
`108.918`. The strict runs stopped normally and passed the Qwen gate. This
promotes group-gated patch activation as the missing source-contract fix for
HF MoE TP4. It also rejects the earlier low-band tuned-config/deploy-compat
runs as overlay-bleed artifacts, not evidence that vNext could not reproduce
HF MoE TP4.

The vNext launcher now maps known container-required paths back to host-visible
mount roots during preflight. This fixed the MoE GGUF profile without weakening
the profile contract: `<container-hf-cache>/...` maps through
`HOST_HF_CACHE`, and `/opt/vllm_patch_bundle/...` maps through
`PATCH_BUNDLE_PATH`. Promote this as a serviceability fix. Reject replacing
runtime container paths with host-specific defaults in profiles.

The clean vNext MoE GGUF TP4 package path now validates across `.20` and `.30`.
Run `20260629T003817Z-gguf-moe35b-tp4` used the `gguf-moe35b-tp4` contract,
TP4, P2P-on, FP16, `MAX_MODEL_LEN=131072`, overlay `gguf_moe`, and manifest
hash `794cb8760003cfa7dafeb0f06ee01823e5e8eebb3f9d398d8bc8bd7697143f0b`.
It produced `.20` strict `119.333`, c1_2000 `120.565`, and c1_10000
`113.366`; and `.30` strict `119.521`, c1_2000 `120.460`, and c1_10000
`113.258`. Strict passed on both lanes. Promote the MoE GGUF package path as
clean contract evidence and as proof that `.30` is an equivalent performing
lane for this profile after repaste work. Keep it separate from public release
claims until a GGUF release path is explicitly published.

The MoE GGUF top-k8 fastpath evidence is mixed and should be described
precisely. In the clean contract run, larger graph shapes were rejected by
shape/layout checks, while token-sized decode shapes became active for
`tokens=4`, `tokens=2`, and `tokens=1`. Promote that as source-path nuance.
Reject broad claims that the whole MoE graph is on the custom top-k8 fastpath.

HF MoE TP8 is now runtime-validated under the generated vNext contract on both
primary lanes. Run `20260629T003817Z-hf-moe35b-tp8` used the `hf-moe35b-tp8`
contract, TP8, P2P-on, FP16, `MAX_MODEL_LEN=131072`, overlay `hf_release`,
patch target groups `common moe hf moe_tp8`, and the expanded HF bundle
manifest `d9e1168f67731ab66ee53671f89dfb31fd3c74c8f61e2f0d0b7c4f0704c298be`.
It produced `.20` strict `115.041`, c1_2000 `115.549`, and c1_10000
`108.805`; and `.30` strict `114.698`, c1_2000 `115.532`, and c1_10000
`108.673`. Strict passed on both lanes. Promote HF MoE TP8 as package-path
runtime evidence. The startup warning about a missing `E=256,N=64` tuned MoE
config should be tracked, but it did not prevent high-band TP8 reproduction in
this run.

The public-staged GGUF reproduction audit on `.30` turned portability into a
hard release requirement. Dense GGUF TP8 reproduced from split F16 GGUF assets,
part manifests, final SHA256 checks, and the Dense text-config archive at
strict `70.072`, c1_2000 `71.031`, and c1_10000 `66.487` backend TPS. MoE GGUF
TP4 initially failed because the generated profile still pointed `TOKENIZER` at
a validation-host Hugging Face snapshot path. That path is rejected for public
release. After replacing it with the public `Qwen/Qwen3.6-35B-A3B` tokenizer
repo pinned by `TOKENIZER_REVISION=995ad96eacd98c81ed38be0c5b274b04031597b0`,
MoE GGUF TP4 reproduced at strict `120.319`, c1_2000 `121.018`, and c1_10000
`113.760` backend TPS. Promote public release assets plus pinned public
tokenizer/config references as the only acceptable GGUF reproduction path.
Reject any release wording or profile that requires LocalAIServers internal
hosts, private mount paths, or pre-existing local model caches.

The follow-up public-staged POSIX-runner repeat on `.30` closes a second
portability gap: the vNext benchmark runner no longer delegates to the older
Bash-only v0.2 helper. With the same staged split GGUF assets, Dense text-config
archive, generated launch artifacts, P2P-on profiles, and normal warmups ->
strict -> fixed-token ladder, Dense GGUF TP8 produced strict `69.986`,
c1_2000 `70.935`, and c1_10000 `66.403`; MoE GGUF TP4 produced strict
`120.822`, c1_2000 `121.733`, and c1_10000 `114.442`. Promote the POSIX
vNext runner as the release reproduction runner. Reject any vNext wording that
requires Bash for the benchmark ladder.

The `.20` full vNext readiness replay promoted the current stand-alone package
shape for end-to-end deployment testing. The documented readiness wrapper
completed serviceability, image/path validation, contract matrix checks, GGUF
asset staging, HF input validation, launch-artifact generation, runtime vLLM
argument-schema validation, host preflight, and all serving benchmark ladders.
Results were: GGUF Dense TP8 strict `69.980`, c1_2000 `70.975`, c1_10000
`66.493`; GGUF MoE TP4 strict `118.252`, c1_2000 `119.198`, c1_10000
`112.115`; HF Dense TP8 strict `70.454`, c1_2000 `71.516`, c1_10000
`66.930`; HF MoE TP4 strict `111.702`, c1_2000 `116.415`, c1_10000
`109.442`; and HF MoE TP8 strict `116.301`, c1_2000 `116.681`, c1_10000
`109.761`. Promote the readiness wrapper as the successful full deployment
test for the stand-alone vNext draft. Reject treating those numbers as v0.2.1
updates.

The same `.20` replay exposed a real HF snapshot portability bug before the
final pass. A top-level HF model symlink can be visible inside the container
while the safetensors shard links remain broken because Hugging Face snapshots
point shards through a relative `blobs` directory. vNext now generates a
runtime `model-bind-root` shim, mounts the resolved snapshot under
`/opt/vnext-models/...`, and mounts the matching snapshot `blobs` directory
read-only so the container sees the same files the host verifier accepted.
Promote fail-closed shard visibility checks and the shim/blob mount pattern.
Reject assuming a visible model directory is enough for HF serviceability.

The fresh `.20` replay after the HF staging clarification completed the full
release-readiness path again from a clean checkout. First-time HF staging and
repeat validation are now separate modes: use `STAGE_HF_PUBLIC_INPUTS=1` only
when the target HF model directories are absent or writable regular
directories, and use `STAGE_HF_PUBLIC_INPUTS=0` when pinned public HF packages
are already staged and verified, including symlinked local snapshots. Reject
rerunning `hf download --local-dir` into symlinked model aliases. With the
correct repeat mode, all five profiles passed the normal warmups -> strict ->
`c1_2000` -> `c1_10000` ladder: GGUF Dense TP8 `69.908` / `70.881` /
`66.446`, GGUF MoE TP4 `118.280` / `120.576` / `113.347`, HF Dense TP8
`70.153` / `71.395` / `66.876`, HF MoE TP4 `113.968` / `116.585` /
`108.544`, and HF MoE TP8 `115.673` / `116.542` / `109.672` backend TPS. The
remaining gate is still public packaging: rerun the same standalone vNext
command from the final tag and public GitHub Release asset URLs before
claiming completed public reproduction.

The vNext hosted-asset package is locally prepared but not yet public. The
simulated upload bundle contains `28` Dense GGUF parts, `36` MoE GGUF parts,
two part manifests, two final hash sidecars, and the Dense text-config archive.
All part hashes verified, and streaming the parts in manifest order reproduced
the locked Dense and MoE final GGUF hashes. The reusable
`verify_vnext_release_asset_bundle.sh` pre-upload check now verifies that same
inventory, and `list_vnext_release_asset_uploads.sh` prints the deterministic
`69`-file upload list. Promote this as upload-preparation evidence. Reject it
as public reproduction until those exact files are attached to the stand-alone
GitHub Release and the release-readiness command is rerun from public release
URLs.

Post-upload public asset verification is now separate from full reproduction.
`verify_vnext_release_asset_urls.sh` checks that the exact expected release
asset names resolve for a tag without downloading the full GGUF parts, and it
fails closed when the first required asset is missing. Promote it as the cheap
post-upload gate before full staging/replay. Reject URL existence alone as a
benchmark reproduction claim.

The vNext release asset upload step is now guarded by a dry-run helper.
`upload_vnext_release_assets.sh` reruns the complete bundle verifier, confirms
the deterministic `69`-file upload inventory, and prints the upload plan by
default. It only mutates GitHub when `UPLOAD_VNEXT_RELEASE_ASSETS=1` is set,
and a real upload now also requires `GH_RELEASE_REPO=owner/repo`. Uploading to
`joe2gaan/localaiservers` is maintainer-only and requires the additional
`ALLOW_LOCALAISERVERS_RELEASE_UPLOAD=1` guard. The helper does not create tags,
create releases, edit notes, delete assets, or use `--clobber`. Promote this
as the guarded upload path for maintainers and for users publishing equivalent
assets to their own repositories. Reject ad hoc manual upload selection, reject
public-user instructions that can accidentally target the LocalAIServers
release namespace, and reject treating a successful upload plan as public
reproduction.

The `.30` scratch-checkout full replay gives second-lane confidence in the
stand-alone vNext package path. From the release-note readiness wrapper, with
normal warmups -> strict -> `c1_2000` -> `c1_10000`, `.30` produced GGUF Dense
TP8 `70.123` / `71.122` / `66.607`, GGUF MoE TP4 `119.473` / `120.620` /
`113.409`, HF Dense TP8 `70.259` / `71.445` / `66.909`, HF MoE TP4 `115.560`
/ `116.777` / `109.882`, and HF MoE TP8 `115.592` / `116.111` / `109.292`
backend TPS. Promote `.30` as an equivalent-performing validation lane for the
current vNext release-candidate package. Reject treating this as completed
public reproduction because it still used maintainer local `file://` simulated
release assets instead of final public GitHub Release asset URLs.

The rebuilt vNext release-asset bundle also passed a full local
`verify_vnext_public_assets.sh` staging gate with
`ALLOW_LOCAL_ASSET_PREFLIGHT=1` and a `file://` release-asset base. The gate
reconstructed both final GGUF files from the split parts, verified the Dense
and MoE final SHA256 values, and accepted both `gguf-dense27b-tp8` and
`gguf-moe35b-tp4` profile contracts from a fresh model root. Promote this as
local asset-shape proof for the stand-alone release. Reject it as public
reproduction until the same gate runs from public GitHub Release URLs and the
full serving replay passes with `RUN_SERVING_BENCHMARKS=1`.
