# GGUF / GFX906 Experiment Log - 2026-06-25

This log captures the GGUF/vLLM experiments from the first Qwen3.6 dense GGUF
source pass. It uses the required public source-record shape:

- Outcome
- Promote / Reject
- Reason
- Next

The log is sanitized. Host labels such as `.20` are run identifiers, not
reachable host addresses.

## Baseline

- Host label: `.20`
- Primary model under test: Qwen3.6 27B Dense GGUF
- GGUF variants prepared: Q4_0 and FP16/half split GGUF
- Published comparison path: FP16 ROCm7.2 v0.2.1 reproduction path
- Benchmark ladder expected after correctness: warmups, `c1_128` uncapped
  strict, `c1_2000`, `c1_10000`

The full benchmark ladder was not promoted for early GGUF TP4/TP8 paths. Those
paths failed first-token correctness, and the later repeat-interleave candidate
passed first-token sanity but failed the release-style uncapped strict gate
after warmups.

Later FP16/half dense GGUF source work repaired semantic correctness under the
Qwen3.5 path, explicit HF tokenizer/config, GGUF tensor fixes, and release
overlay. That path is coherent and strict-valid, but it remains slower than the
published FP16 ROCm7.2 reproduction numbers. MoE GGUF artifact work has started,
but serving is blocked before benchmark warmups by missing Qwen3.6 35B-A3B MoE
GGUF architecture support in the current vLLM/Transformers path.

## Experiment Ledger

| ID | Experiment | Outcome | Promote / Reject | Reason | Next |
| --- | --- | --- | --- | --- | --- |
| GGUF-001 | Load dense Q4_0 GGUF with a text-only Qwen3.6 config. | Model path reached deterministic prompt testing. | Promote as diagnostic precondition. | It avoids mixing GGUF behavior with multimodal / M-RoPE config drift. | Keep text-only config for all dense GGUF source tests. |
| GGUF-002 | TP1 Q4_0 first-token sanity on `Hello`. | Returned coherent comma-like first-token behavior. | Promote as baseline sanity. | Loader, tokenizer, decode loop, and TP1 graph are not universally broken. | Use TP1 as reference for TP4/TP8 traces. |
| GGUF-003 | TP2 Q4_0 first-token sanity on `Hello`. | Returned coherent first-token behavior. | Promote as baseline sanity. | The issue does not appear at all low TP counts. | Compare TP2 traces with TP4 before assuming TP-only failure is linear. |
| GGUF-004 | TP4 Q4_0 first-token sanity on `Hello`. | Returned degraded punctuation/newline behavior instead of matching TP1. | Reject as serving candidate; promote as failure reproducer. | Correctness fails before benchmark warmups or decode tiers. | Trace where TP4 diverges. |
| GGUF-005 | TP8 GGUF sanity. | Produced catastrophically wrong token behavior. | Reject as serving candidate. | TP8 correctness is worse than TP4 and cannot support benchmark claims. | Repair TP4 first, then recheck TP8. |
| GGUF-006 | `VLLM_QWEN35_KV_REPLICA_MODE=mod` under TP4. | Did not repair first-token behavior. | Reject as fix. | The observed TP4 failure is not explained by this K/V replica mode. | Stop spending time on replica mode until traces point back there. |
| GGUF-007 | No-async launch shape. | Engine did not initialize cleanly enough for a meaningful comparison. | Reject as reproduction path. | It failed before correctness or throughput evidence. | Keep standard serving path unless a trace requires otherwise. |
| GGUF-008 | Final hidden/logit trace TP1 versus TP4. | TP4 hidden state had already diverged before logits. | Promote as diagnostic evidence. | The failure is not primarily tokenizer ID mapping or final lm-head gather. | Move tracing earlier into layer 0. |
| GGUF-009 | Layer-0 GatedDeltaNet trace TP1 versus TP4. | TP1 and TP4 shared the same input-normalization sum, then diverged at layer-0 GatedDeltaNet attention output. | Promote as current root-cause boundary. | The earliest clean divergence is inside the GatedDeltaNet path. | Instrument q/k/v/z, beta/gate, and output projection boundaries. |
| GGUF-010 | Force dequantized GGUF matmul fallback. | TP4 output changed but did not match TP1 correctness. | Reject as fix; promote as negative evidence. | The issue is not simply ROCm GGUF quantized matmul. | Continue into GatedDeltaNet tensor layout rather than matmul-only fixes. |
| GGUF-011 | Force torch GatedDeltaNet reference path. | TP4 still failed first-token sanity. | Reject as fix; promote as root-cause evidence. | The custom GatedDeltaNet core is not the only suspect; tensors entering the path or GGUF TP layout remain suspect. | Trace `in_proj_qkvz`, `in_proj_ba`, q/k/v/z, beta/gate, and output projection. |
| GGUF-012 | Low-level C GGUF tensor-info scanner against Q4_0 and FP16/half split files. | Confirmed Qwen3.6 GGUF tensor names and dimensions for layer-0 `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`, and `ssm_out`. | Promote as source diagnostic. | The GGUF file layout matches llama.cpp-style Qwen3.5 naming and shows q/k/v as one contiguous `attn_qkv` tensor plus a separate gate/z projection, so a simple qkv group-reorder theory is less likely. | Focus on vLLM GGUF partition handling for fused `in_proj_qkvz`, `in_proj_ba`, and `out_proj`. |
| GGUF-013 | TP4 Q4_0 runtime trace with logical GGUF shard-order patch and P2P-on. | Deterministic `Hello` one-token probe returned `.` as the top token with `,` second. Layer-0 trace still diverged and no full benchmark ladder was run. | Reject as fix; promote as negative evidence. | Sorting logical shard IDs before GGUF linear application was not enough to restore TP1-like first-token correctness. | Inspect partition axis and load order for `ssm_out`, split the fused z/beta-alpha path if needed, and compare against llama.cpp Qwen3.5 graph semantics. |
| GGUF-014 | Source inspection of vLLM GGUF column/row parallel partition math. | TP4 traced qkvz rows and `out_proj` packed input columns match expected vLLM partition sizes. | Reject the simple loader-size mismatch hypothesis; promote as source boundary evidence. | `in_proj_qkvz` local rows line up as q/k/v/z under TP4, and row-parallel `out_proj` Q8_0 packed shape matches the expected local input partition. | Add shard-offset/checksum instrumentation and compare actual per-rank loaded tensor contents, not just shapes. |
| GGUF-015 | Load-time shard checksum instrumentation. | The qkvz and `out_proj` qweights were still uninitialized at the end of `load_weights`; beta/alpha and conv checksums were not a final materialized view. | Reject as final diagnostic location; promote as materialization-timing evidence. | vLLM GGUF quantized tensors are materialized later than the first load hook, so load-time checksums can mislead the investigation. | Move checksum instrumentation to the first forward path where quantized linear weights are materialized. |
| GGUF-016 | Forward-time TP4 shard checksum trace with logical shard-order patch and P2P-on. | Forward-time checksums showed materialized qkvz, beta/alpha, conv, and output-projection tensors, but the one-token `Hello` probe still returned `.` as the top token. | Reject as fix; promote as source boundary evidence. | qkvz physical shard order appears as `[3, 0, 1, 2]` with offsets for z, q, k, and v; logical reordering to q, k, v, z is necessary but not sufficient. The remaining failure likely sits in GatedDeltaNet state/cache semantics or post-qkvz execution rather than gross tensor shape. | Compare TP1 versus TP4 forward-time checksums and traces under the same instrumentation, then isolate chunked-prefill to decode state behavior in layer-0 GatedDeltaNet. |
| GGUF-017 | llama.cpp `gguf.cpp` reader implementation review. | Confirmed the reference reader checks duplicate tensor names, tensor type range, row divisibility by quantization block size, aligned contiguous tensor offsets, and optional data attachment through `no_alloc`. | Promote as implementation reference. | Our scanner and vLLM trace should be compared against the same invariants; the current evidence does not point to a basic GGUF tensor-info parse failure. | Add a small offset/block-size parity check to the low-level scanner before the next TP1/TP4 forward comparison. |
| GGUF-018 | vLLM public GGUF support and Qwen3.5 issue review. | Upstream documentation still labels GGUF support experimental, recommends using the base tokenizer, and documents `--hf-config-path` for models whose metadata cannot be converted. Public Qwen3.5 GGUF reports focus on `qwen3_5` versus `qwen35` model-type mapping and vision-config depth fallback. | Promote as external boundary evidence. | Our local path is past the startup/config failure class: the model loads and serves, then fails TP4/TP8 correctness inside execution. The upstream loader mapping PR is useful but does not explain the layer-0 GatedDeltaNet divergence. | Keep the repo/quant plus explicit tokenizer/config launch style while debugging; do not chase local-file config bugs unless the launch path changes. |
| GGUF-019 | Current vLLM Qwen3.5 GatedDeltaNet source comparison. | Current vLLM Qwen3.5 code uses non-interleaved `[q, k, v, z]` qkvz plus separate `[b, a]` projection. The FP16 release overlay fuses b/a into qkvz for its runtime path. | Promote as source fork evidence; not a fix. | GGUF weights and vLLM GGUF packing are sensitive to shard IDs and materialization order, so reusing the FP16 fused overlay is a correctness risk until a split qkvz/ba path is tested. | Build the next experimental path around the current separate `in_proj_qkvz` plus `in_proj_ba` Qwen3.5 source shape and compare it to the fused FP16 overlay. |
| GGUF-020 | GGUF packed-shard loader restriction review. | The release runtime linear overlay contains GGUF-specific handling for packed shard IDs and rejects multi-index GGUF shard IDs in some paths. | Promote as source-risk evidence. | Qwen3.5 fused qkv loading uses tuple shard IDs such as `(0, 1, 2)`. That is normal for FP16 packed modules, but it is a high-risk shape for GGUF materialization unless the GGUF loader explicitly handles it. | Prefer split or explicitly mapped GGUF q/k/v/z and b/a loading over broad tuple-shard reuse. |
| GGUF-021 | Prior FP16 dense source-work mining. | Existing key-learnings records already rejected GDN projection dual-streaming, GDN output zeroing, simple flattening, and many RowParallel scheduling variants as promotion levers. | Promote as guardrail. | GGUF work should not repeat those throughput experiments before correctness is fixed; the active failure is first-token correctness, not sub-percent dense TPS tuning. | Keep GGUF work on parser/loader invariants, qkvz/ba shard semantics, TP state behavior, and only then resume throughput tuning. |
| GGUF-022 | TP1 forward-time checksum control under the same GGUF instrumentation. | TP1 stayed coherent on the deterministic `Hello` probe and returned `,` as the top token. Its qkvz physical shard order was also `[3, 0, 1, 2]`, with offsets for z, q, k, and v over the full local tensor. | Promote as control evidence. | The physical GGUF qkvz order alone is not the full root cause: TP1 has the same physical order and still behaves coherently, while TP4 returns `.` first under the same image and logical-order patch. | Compare TP4 against TP1 at rank-local qkvz/ba/out-projection content and GatedDeltaNet state semantics rather than chasing a simple physical-order fix. |
| GGUF-023 | llama.cpp Qwen3.5/Qwen3.6 tensor-parallel split-granularity research. | Upstream llama.cpp PR #23843 is merged and fixes Qwen3.5/Qwen3.6 tensor-parallel granularity for heterogeneous quant mixes by selecting the correct tensor for block-size/granularity decisions. The local llama.cpp checkout includes the `ssm_out.weight` fallback for qkv/gate split config. | Promote as highest-value external source clue; not yet a vLLM fix. | The scanned GGUF file stores layer-0 `attn_qkv` and `attn_gate` as Q4_0 while `ssm_out` is Q8_0. vLLM's GGUF merged loader slices each loaded tensor directly by local shard size, so it may be missing the companion-tensor granularity logic llama.cpp needed for this model family. | Add a split-audit diagnostic that compares vLLM GGUF shard boundaries for qkv, gate/z, beta/alpha, and `ssm_out` against llama.cpp-style Qwen3.5/Qwen3.6 split segments before attempting a throughput run. |
| GGUF-024 | Low-level C split audit for layer-0 Q4_0/Q8_0 tensors. | A scratch C auditor calculated GGUF alignment, packed bytes, block sizes, and TP2/TP4/TP8 axis-local splits for layer-0 `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`, and `ssm_out`. No basic block-size remainder appeared for TP4 or TP8. | Reject the simple quant-block remainder theory; promote as split-audit evidence. | Raw tensor dimensions and quant block sizes are cleanly divisible, so the bug is not just an obvious GGUF row/column block alignment failure. The remaining llama.cpp clue is subtler segment/granularity selection across companion tensors, not a plain divisibility issue. | Compare model-family segment semantics, not only scalar dimensions: Qwen3.5/Qwen3.6 qkv/gate/beta/alpha splits should be audited against the `ssm_out` reference behavior. |
| GGUF-025 | TP4 all-rank layer-0 trace under the same failing image. | TP4 reproduced the same wrong first-token result: `.` was top and `,` second. All ranks agreed on the final wrong top IDs, while rank-local qkv, beta/alpha, and GatedDeltaNet outputs differed before the final synchronized hidden state. | Reject as fix; promote as failure-boundary evidence. | The failure is not a single-rank final-logit artifact. The trace points back to rank-local GatedDeltaNet inputs/outputs and their TP reassembly before logits. | Run a cleaner rank-local split-semantic trace that labels q/k/v/z, beta, alpha, and `ssm_out` slices by logical tensor segment rather than only by flat row offsets. |
| GGUF-026 | TP4 segment-labeled q/k/v/z and beta/alpha trace. | TP4 again returned `.` as top token and `,` second. The trace labeled q, k, v, z, beta, and alpha segments across ranks and showed populated rank-local segments before the synchronized wrong logits. | Reject as fix; promote as source-boundary evidence. | The issue is not an obviously empty q/k/v/z or beta/alpha segment. The failure remains after correct-looking segment population, so the likely target is segment ordering/granularity semantics, GatedDeltaNet state handling, or TP reassembly across those segments. | Compare segment-labeled TP1 and TP4 under the same hook, then test a minimal split-semantics patch rather than more launch flags. |
| GGUF-027 | TP1 segment-labeled control under the same q/k/v/z and beta/alpha trace hook. | TP1 returned the coherent top token `,` on the same deterministic `Hello` probe. The same trace hook populated q, k, v, z, beta, and alpha segments and produced the expected TP1 logprob ordering, while the TP4 segmenttrace run under P2P-on returned `.` first and `,` second. | Promote as control evidence; not a fix. | The segmenttrace hook itself is not causing the TP4 failure, and populated segments alone are not sufficient proof of correct TP semantics. TP1 and TP4 now differ under the same instrumentation, so the next source target is the exact per-rank split semantics and GatedDeltaNet reassembly path. | Build a minimal loader/model diagnostic that maps each GGUF tensor range to logical q/k/v/z, beta, alpha, and `ssm_out` segments per rank, then compare against llama.cpp Qwen3.5/Qwen3.6 split semantics. |
| GGUF-028 | TP1 versus TP4 compact segment-checksum diagnostic. | TP1 again returned `,`; TP4 again returned `.`. For the final one-token decode step, TP4 rank-local q/k/v/z/b/a sums aggregate back to the TP1 sums: q `19.639277` vs TP1 `19.639275`, k `8.790018` vs `8.790018`, v `300.671662` vs `300.671661`, z `-150.053726` vs `-150.053726`, b `80.371098` vs `80.371094`, and a `161.471282` vs `161.471283`. The GatedDeltaNet core/output diverged: TP1 core sum `5.396257` and output sum `9.024781`, while TP4 rank-local core sums totaled `-2.240752` and the replicated output sum was `12.022104`. | Promote as narrowed root-cause evidence; reject throughput testing. | The loader/split inputs appear to cover the same q/k/v/z/b/a data in aggregate, but TP4 does not produce the same GatedDeltaNet core/output behavior. This shifts the active source target away from raw GGUF tensor loading and toward TP GatedDeltaNet core state, local-head semantics, or output reassembly. | Instrument or patch the GatedDeltaNet core boundary next: compare local core chunks against TP1 chunks, inspect state/cache behavior across prefill/decode, and only then test a source fix. |
| GGUF-029 | TP1 rank-equivalent chunk comparison after `causal_conv1d`. | TP1 rank-equivalent chunks were compared against TP4 ranks at the post-conv/pre-recurrent boundary. Value, b/a, `A_log`, and `dt_bias` matched by chunk/rank. Q and K did not: for example TP1 chunk 0 query/key sums were `13.044325` / `7.324649`, while TP4 rank 0 query/key sums were `11.564686` / `14.409418`; TP1 chunk 1 was `15.954918` / `8.375532`, while TP4 rank 1 was `16.515209` / `6.321072`. | Promote as first clean divergence boundary; reject benchmark testing. | The pre-recurrent value and parameter shards line up, but q/k after the internal causal convolution do not. That points to the Q/K `conv1d` sharded weight/order path for GGUF TP4, not to value projection, b/a, A/dt parameters, or generic launch flags. | Instrument `conv1d.weight` q/k/v shard checksums and test a minimal Q/K conv weight order fix under TP4. |
| GGUF-030 | TP1/TP4 pre-conv input and `conv1d.weight` checksum comparison. | TP1 chunk-equivalent pre-conv q/k/v inputs matched TP4 rank-local q/k/v inputs exactly. TP1 chunk-equivalent `conv1d.weight` q/k/v slices also matched TP4 rank-local conv weights exactly. TP4 still returned `.` when Q/K used the tiled repeat mode. | Reject `conv1d.weight` loader/order as the primary fault; promote repeat-order suspicion. | The data entering `causal_conv1d` and the conv weights are rank-equivalent, so the earlier post-conv Q/K mismatch is not caused by a simple GGUF conv tensor shard mismatch. The remaining difference is how smaller Q/K head groups are expanded to the value-head space under TP. | Test Q/K repeat-interleave instead of tiled repeat and compare TP1/TP4 first-token behavior before any benchmark ladder. |
| GGUF-031 | TP1/TP4 Q/K repeat-interleave candidate. | With `VLLM_QWEN35_TILE_QK_REPEAT=0`, TP4 returned `,` on the deterministic `Hello` one-token probe instead of the previous `.`. TP1 under the same repeat-interleave setting also returned `,`, and TP1 chunk summaries aligned with TP4 rank summaries at the post-conv/pre-recurrent boundary. | Promote as first correctness candidate; still reject benchmark claims. | Repeat-interleave makes TP1 chunk ordering and TP4 rank-local Q/K ordering consistent for the simple first-token probe. This repairs the immediate TP4 first-token failure but has not yet cleared TP2/TP8 sanity, warmups, uncapped strict, or fixed-token benchmark tiers. | Run TP2, TP4, and TP8 first-token sanity with repeat-interleave, then normal warmups -> `c1_128` uncapped strict -> `c1_2000` -> `c1_10000` only if correctness remains stable. |
| GGUF-032 | TP2 and TP8 repeat-interleave first-token sanity. | TP2 and TP8 both returned `,` on the deterministic `Hello` one-token probe with `VLLM_QWEN35_TILE_QK_REPEAT=0`, P2P-on, and the same Q4_0 GGUF image. TP8 no longer produced the earlier catastrophic first-token behavior. | Promote as first-token correctness gate pass; still reject benchmark claims. | Repeat-interleave now passes TP1, TP2, TP4, and TP8 first-token sanity. These were diagnostic launches, not full-context benchmark launches, so they do not prove throughput or strict-prompt validity. | Move to a clean full-context launch without tracing, then run normal warmups -> `c1_128` uncapped strict -> `c1_2000` -> `c1_10000`. |
| GGUF-033 | TP8 full-context repeat-interleave warmups and uncapped strict gate. | Clean TP8 full-context launch on `.20` used `VLLM_DTYPE=half`, P2P-on, `MAX_MODEL_LEN=131072`, Q4_0 GGUF, repeat-interleave, and the normal Qwen begin-think benchmark harness. Eight 2000-token warmups completed with stable backend decode TPS around `54.31` to `54.35`. The following `c1_128` uncapped strict request stopped after 12 completion tokens with no visible answer, no think close, and `qwen_gate_valid=false`. The generated reasoning text was degenerate and did not satisfy the strict parser. | Reject as benchmark candidate. | Repeat-interleave fixes simple first-token probes but not full benchmark correctness. The strict gate failed before `c1_2000` and `c1_10000`, so no GGUF throughput result from this path should be compared to the FP16 release numbers. | Debug full-context Qwen thinking correctness next: compare the benchmark prompt path against TP1, inspect reasoning-parser/proxy interaction only after confirming model output semantics, and continue source-level GatedDeltaNet/state investigation before running more throughput tiers. |
| GGUF-034 | TP8 full-context FP16/half GGUF candidate with repeat-interleave. | A true FP16/half GGUF was converted from the cached Qwen3.6 27B source weights and verified by a low-level scanner as FP16/half tensor data. A local-file GGUF config-path issue required a minimal experimental patch that skipped speculator probing for local GGUF files so the explicit Hugging Face config path could be used. The server loaded on `.20` with `VLLM_DTYPE=half`, P2P-on, `MAX_MODEL_LEN=131072`, TP8, and repeat-interleave. Eight 2000-token warmups completed after normal startup delay: warmup 1 paid first-request overhead and reached `49.438` backend decode TPS; warmups 2-8 stayed around `49.687` to `49.744` backend decode TPS. All warmups produced the same degenerate repeated reasoning text, no visible answer, and no think close. The strict request was stopped manually after the warmup degeneration made it non-useful; no `c1_2000` or `c1_10000` tier was promoted. | Reject as benchmark candidate; promote as FP16 GGUF source evidence. | Moving from Q4_0 to FP16/half GGUF did not restore Qwen benchmark semantics and was slower than the published FP16 v0.2.1 reproduction path. The failure is therefore not only quantization loss; the local GGUF loader/model execution path still produces degenerate benchmark-prompt reasoning under TP8 full context. | Inspect local-file GGUF config handling, GGUF tensor materialization, and Qwen3.5 / Qwen3.6 GatedDeltaNet execution against the HF-weight FP16 path. Do not run fixed-token throughput tiers from this candidate until semantic correctness is repaired. |
| GGUF-035 | TP4 FP16/half GGUF smoke localization and force-unquantized check. | A small-context TP4 FP16/half GGUF launch on `.20` returned the coherent comma token for a raw one-token `Hello` completion. Direct chat `Hello` was malformed and returned only an empty think section. A raw completion using the synthetic benchmark prompt degenerated into a repeated `calibration manual warranty` loop. An env-gated force-unquantized GGUF linear/embedding patch produced the same outcomes: coherent one-token `Hello`, malformed chat, and the same benchmark-prompt phrase loop. | Reject force-unquantized as a fix; promote as localization evidence. | The FP16 GGUF file, tokenizer, and first-token raw completion path are not globally broken. The failure appears when real prefill/chat/benchmark context enters the Qwen3.5 / Qwen3.6 execution path. Forcing F16 GGUF layers toward unquantized methods did not repair semantics, so the next source target is not a simple quantized-kernel bypass. | Compare GGUF prefill and decode state against the known-good HF-weight FP16 path at layer-0 GatedDeltaNet, chat-template tokenization, recurrent state update, and output projection boundaries. |
| GGUF-036 | Clean patch-bundle audit and direct QwenNext launch attempt. | The container entrypoint copies files from the patch bundle after Docker mounts are applied. That means individual Qwen source-file mounts can be overwritten by the bundle at startup. A clean bundle was built with the intended QwenNext source file and used for a direct `Qwen3NextForCausalLM` launch. The mixed-mount failure was removed, but direct QwenNext initialization failed before serving because the model received the outer Qwen3.5 config object, which lacks direct text fields such as `hidden_size`. | Reject as launch fix; promote as source-routing evidence. | The previous TP4/TP8 GGUF tests may have been affected by patch-bundle source routing. However, simply switching the architecture override to direct QwenNext is insufficient; it needs a config adapter that passes the text config into the QwenNext model path while preserving GGUF name mapping. | Build a clean Qwen3.5 wrapper or direct-QwenNext adapter instead of mixing patch bundles and individual mounts. The next run should prove which source file is active before any benchmark request. |
| GGUF-037 | Direct QwenNext split-GDN reorder smoke after adding the text-config adapter. | The direct `Qwen3NextForCausalLM` route served far enough to answer a raw `Hello` probe, and the split-GDN QKV reorder marker fired for the expected local tensor shape. The returned text was invalid byte/glyph garbage rather than a coherent token sequence. | Reject direct QwenNext as a benchmark candidate; promote as negative source evidence. | The direct-next route is not faithful for this Qwen3.6 dense GGUF model even after startup and reorder issues are bypassed. The authoritative config remains `qwen3_5`, so direct-next should remain diagnostic only unless a source-level compatibility proof is produced. | Return to the Qwen3.5 path and isolate prefill/GatedDeltaNet semantics rather than trying to promote direct-next output. |
| GGUF-038 | Active FP16/half GGUF tensor-type verification. | The low-level scanner verified the active single-file GGUF at `Qwen3.6-27B-F16.gguf` as GGUF v3 with sampled key tensors stored as `F16(1)`, including `token_embd`, `output.weight`, `blk.0.attn_qkv.weight`, `blk.0.attn_gate.weight`, `blk.0.ssm_out.weight`, `blk.0.ssm_alpha.weight`, and `blk.0.ssm_beta.weight`. | Promote as dtype evidence. | The confusing runtime `qweight_type` value is not evidence that this active file is BF16. The current garbage-output failure should not be attributed to BF16 conversion unless a later scanner contradicts this. | Keep using FP16/half terminology for this GGUF file and focus debugging on loader/model execution semantics. |
| GGUF-039 | TP4 FP16/half Qwen3.5 patched-bundle smoke with repeat-interleave and torch GDN disabled. | The run used P2P-on, TP4, `MAX_MODEL_LEN=4096`, the FP16/half GGUF file, explicit Qwen3.5 tokenizer/config, local-GGUF config bypass, logical-shard patch, and the older `qwen35-gguf-corechecksum` bundle. A raw `Hello` completion returned repeated greeting text instead of invalid bytes. A chat `Hello` request timed out after the FLA native path warned that the short sequence length was smaller than the head count, followed later by a shared-memory wait warning. | Reject as benchmark candidate; promote as patch-contamination evidence. | This was not a truly clean Qwen3.5 run because the diagnostic bundle replaced `qwen3_5.py`. It still shows that the faithful Qwen3.5 route is closer than direct-next but fails chat/prefill before warmups. The short-prompt FLA warning is not conclusive by itself because short prompts can naturally have fewer tokens than heads. | Split required startup fixes from diagnostic behavior changes, then test Qwen3.5 prefill with a coherent minimal bundle. |
| GGUF-040 | TP4 FP16/half Qwen3.5 patched-bundle smoke with torch GDN prefill fallback. | Enabling `VLLM_QWEN35_TORCH_GDN_PREFILL=1` avoided the chat timeout but changed the failure shape: raw `Hello` produced repeated commas and chat returned only `<think>` with a stop finish. | Reject torch GDN prefill fallback as a fix. | Bypassing the custom GDN prefill path does not restore semantic output. The root cause is not just one custom GDN kernel call; it is likely in GGUF Qwen3.5 materialization, state semantics, or the patched wrapper behavior around prefill/decode. | Use torch fallback only as a diagnostic contrast, not as a benchmark candidate. |
| GGUF-041 | TP4 FP16/half minimal image-base launch without the old Qwen3.5 patch bundle. | The run mounted only the model/cache and local-GGUF config bypass, without `qwen35-gguf-corechecksum`. It failed during engine startup before serving with an uninitialized GGUF embedding parameter in `_apply_gguf_embedding`. | Reject as a serving path; promote as startup-boundary evidence. | The older bundle is carrying at least one required GGUF embedding/materialization workaround, not only trace code. However, it also changes Qwen3.5 behavior, so promotion tests need a smaller coherent bundle that contains only required startup fixes plus intentional correctness changes. | Build a minimal source bundle: GGUF embedding/lm-head materialization, local config bypass if still needed, and a single Qwen3.5 correctness patch under test. |
| GGUF-042 | TP4 FP16/half Qwen3.5 patched bundle with materialized GGUF qweights labeled F16 instead of BF16. | The active GGUF scanner proves the file is F16, so the old materialization helper was copied and changed from `WeightType.BF16` to `WeightType.F16`. The server loaded with embedding and lm-head qweight type `1`. Raw `Hello` still repeated greeting tokens. Chat `Hello` returned only an empty think block. A longer simple chat prompt echoed part of the prompt inside `<think>`. | Reject F16 qweight-type correction as a semantic fix; promote as dtype/materialization evidence. | Correcting the materialized qweight type from BF16 to F16 removes one inconsistency and avoids treating the active file as BF16, but it does not restore meaningful output. The current blocker remains Qwen3.5 GGUF prefill/state/model semantics, not a single embedding/lm-head dtype label. | Keep F16 materialization in any future minimal bundle because it matches the file, but continue isolating GatedDeltaNet prefill/decode behavior and GGUF loader materialization. |
| GGUF-127 | Qwen3.6 35B-A3B MoE GGUF TP4 model-loop trace with request-gated row dumps. | The copied modeltrace overlay reached `/health` after normal load time, reproduced the branch flip after shared token `248046`, and wrote request-gated row values for layer 0, layer 39, final hidden, and loop-final norm. Aggregate layer stats stayed close to the HF-weight control from layer 0 through layer 39, but GGUF still ranked `Here` id `8160` above `Thinking` id `90700`. | Promote as diagnostic evidence; reject as fix. | The runtime norm-offset repair and graph KV-layout bypass are enough for startup, health, and short coherent probes, but the thinking-enabled trajectory still diverges. Aggregate scale/stat parity rejects a gross norm or final lm-head lookup theory; the remaining evidence points to directional hidden drift before logits. | Run the same request-gated row-vector trace on the HF-weight comparator and compute layer-by-layer cosine/diff for layer 0, layer 39, loop-final norm, and final hidden. |
| GGUF-128 | External source review of `Kausik-A/qwen3.6-27b-mi50-vllm`. | The project provides a single-MI50/eGPU GGUF deployment bundle for dense Qwen3.6 27B Q6_K_XL on `aiinfos/vllm-gfx906-mobydick`, with patches for qwen35/qwen35moe registry/config aliases, text-only M-RoPE suppression, GGUF `ssm_dt.bias` mapping, expert aliases, quantized embedding/lm-head plumbing, conv1d 2D-to-3D reshape, `IsHybrid` hooks, and tuple-shard GGUF fused QKVZ loading. | Promote as source-reference inventory; reject as a direct benchmark/release path. | Its assumptions are single-GPU, eager-mode, ROCm 6.3-era, 4096 context, and quantized dense GGUF. It does not match our ROCm7.2 multi-MI50 TP8/TP4, full-BAR/P2P-on, 131K active-contract path, and it does not resolve our MoE GGUF hidden-direction branch flip. | Compare its tuple-shard `MergedColumnParallelLinear` and MiniMax GGUF aliases against our current experimental overlays, but do not copy the launch profile into release reproduction. |
| GGUF-129 | HF-weight MoE TP4 row-vector comparator for the GGUF branch-flip run. | First, mounting the GGUF modeltrace overlay directly onto the HF-weight comparator was rejected: it reached health after `225` seconds but returned `!!`, emitted NaN trace values, and no longer matched the known-good HF branch. A narrower HF-only copy of the hidden-vector overlay with only `row_values` added reached health after `195` seconds and reproduced the known-good branch: response `Thinking Process`, event 1 `Thinking` id `90700` rank 1 and `Here` id `8160` rank 2. Comparing row vectors against the GGUF modeltrace run showed layer-0 `post_mlp` cosine about `0.849558` and layer-39 `post_mlp` cosine about `0.984005`; the HF event-0 hidden row versus GGUF final hidden row had cosine about `0.947924`. | Promote as partial diagnostic evidence; reject as final localization. | The valid HF row overlay proves the branch comparison can be reproduced without perturbing HF, and it rejects the broad GGUF-modeltrace overlay for HF comparator use. However, the layer boundary rows are still first-call/request rows, not guaranteed to be the decode call that flips `Thinking` versus `Here`. The true branch-local drift still needs call-indexed decoder boundary rows. | Build a decode-step boundary trace that records call 0/1/2 layer rows for both HF and GGUF, then compare layer-0/layer-39 rows at the exact compute-logit event where HF picks `Thinking` and GGUF picks `Here`. |
| GGUF-130 | Call-indexed MoE TP4 decoder-boundary rows for HF versus GGUF. | Built separate request-gated call-row overlays for the HF-weight comparator and the GGUF path with runtime norm-offset plus graph KV-layout bypass. HF reached health after `195` seconds and returned `Thinking Process`; GGUF reached health after `100` seconds and returned `finish_reason=length` with no visible or reasoning content for the two-token cap. The branch reproduced exactly: after shared token `248046`, HF ranked `Thinking` id `90700` first at `25.546875` and `Here` id `8160` second at `25.203125`, while GGUF ranked `Here` first at `26.468750` and `Thinking` second at `22.953125`. Call-indexed row comparisons show call 0 starts identical at layer 0, but call 1 diverges by layer-0 attention/MLP: `layers.0.call1.post_attention` cosine `0.900811`, `layers.0.call1.post_mlp` cosine `0.879649`, `layers.39.call1.post_mlp` cosine `0.955899`, and `compute_hidden.call1` cosine `0.890804`. Call 2 is no longer comparable because the generated tokens have already diverged. | Promote as branch-local source evidence; reject as benchmark candidate. | The branch flip is not caused by tokenizer metadata, gross input embedding mismatch, initial layer-norm scale, graph-only execution, or final lm-head row lookup alone. The earliest useful aligned drift now appears immediately after layer-0 attention/MLP on the first post-special-token decode call, with late-layer hidden direction still materially different before logits. Warmups and fixed-token tiers remain premature. | Next compare layer-0 attention inputs/outputs and GatedDeltaNet/Mamba state carry at call 1, including q/k/v/z/beta/alpha, conv/state tensors, and residual add order. Keep the HF and GGUF call-row traces as the aligned comparator baseline. |
| GGUF-131 | MoE TP4 all-head GatedDeltaNet trace plus static tensor cross-check. | Added all-head request-gated traces around layer-0 GatedDeltaNet `core_attn_out_raw`, `z_for_norm`, and `core_attn_out_normed`, then compared HF and GGUF on the same two-token branch probe. The branch flip remained. The z gate matched across ranks (`z_for_norm` cosine `1.000000` on the sampled rank pairs), while GatedDeltaNet raw core output already drifted on call 1, with sampled pair cosines from `0.837480` to `0.988653`; normalized core output then ranged from `0.539105` to `0.998626`. Separate exact static checks showed the inverse V/Z/out-projection, `dt_bias`, `A_log`, `ssm_norm`, and conv V-channel transforms can match HF within FP32 noise. Offline reconstruction with the HF out-projection weight reproduced both HF and GGUF traced projection outputs, proving the out-projection mapping itself is not the branch-flip root cause. | Promote as source localization; reject as a benchmark candidate. | Kausik-style registry, tuple-shard, conv reshape, and tensor-alias patches are useful prerequisites, but this result points below those compatibility fixes. The remaining mismatch starts inside or immediately around `torch.ops.vllm.gdn_attention_core`, its loaded runtime parameters, or its recurrent state/cache semantics, not in tokenizer IDs, lm-head rows, z gating, or static out-projection mapping. | Trace the runtime-loaded GatedDeltaNet static tensors and state/cache identity on call 1, then inspect `gdn_attention_core` state ordering and TP-local state carry before running any more warmups or fixed-token tiers. |
| GGUF-132 | MoE TP4 runtime GatedDeltaNet state trace before `A_log` loader repair. | Added request-gated `_forward_core` traces for runtime `dt_bias`, `A_log`, conv weights, `ssm_norm`, state before/after, mixed q/k/v, gating, and recurrent outputs. HF returned `Thinking Process`; GGUF returned `Here's`. Runtime `dt_bias`, conv weights, `ssm_norm`, mixed q/k/v, beta, and initial state matched closely, but GGUF runtime `A_log` was a rank-constant value while HF runtime `A_log` was rank-varying. | Promote as source localization; reject as benchmark candidate. | The active served MoE GGUF path was not applying the intended GGUF `ssm_a` to vLLM `linear_attn.A_log` transform. This explains the earlier `gdn_attention_core` drift and proves the next fix belongs in the active loader path, not in launch flags, tokenizer metadata, or warmup repetition. | Identify whether the transform is skipped by equal-head early return, GGUF name mapping, or top-level MoE loader bypass before changing any benchmark path. |
| GGUF-133 | MoE TP4 `A_log` loader-path repair. | First moving `A_log` conversion before the equal-head early return did not change runtime `A_log`, proving that helper was not reached by the served MoE path. Adding explicit GGUF `blk.*.ssm_a` to `model.layers.*.linear_attn.A_log` mapping and wrapping the top-level `Qwen3_5ForCausalLMBase.load_weights` path repaired runtime `A_log`: GGUF now matched HF rank-varying `A_log` exactly. Layer-0 call-1 then matched HF through GatedDeltaNet runtime params, recurrent output, post-attention, and post-MLP with cosine effectively `1.000000`. The two-token branch still flipped: HF ranked `Thinking` id `90700` above `Here` id `8160`, while GGUF ranked `Here` above `Thinking`. | Promote as source fix candidate and diagnostic evidence; reject as benchmark candidate. | The loader repair fixes a real correctness bug and invalidates the previous layer-0 drift evidence, but it is not sufficient for benchmark promotion. The remaining branch flip now enters later than layer 0, so another source boundary is needed before any warmup or fixed-token ladder. | Compare later decoder boundaries under the repaired `A_log` path, starting with layer 39 and final logits, then bisect earlier layers only if layer 39 still diverges. |
| GGUF-134 | MoE TP4 repaired-`A_log` layer-39 comparator. | Reran the same two-token HF/GGUF branch probe with layer-0 and layer-39 request-gated traces after the top-level `A_log` loader repair. HF returned `Thinking Process`; GGUF still returned `Here's`. Layer-0 call-1 remained aligned with min cosine `1.000000` across traced GatedDeltaNet inputs, runtime params, recurrent core, post-attention, and post-MLP. By layer 39 call 1, drift was material again: `pre_input_layernorm` min cosine `0.990509`, `post_input_layernorm` `0.970792`, `post_attention` `0.986151`, and `post_mlp` `0.891285`. The branch logits remained flipped: HF event 1 had `Thinking` id `90700` rank 1 at `25.546875` and `Here` id `8160` rank 2 at `25.203125`; GGUF had `Here` rank 1 at `23.812500` and `Thinking` rank 2 at `23.468750`. | Promote as updated source boundary; reject as benchmark candidate. | The linked dense GGUF repo provides useful compatibility references, but it does not solve this MoE path: its MoE mapping lacks `ssm_a` / `A_log`, and its top-level MoE loader still uses plain `AutoWeightsLoader`. Our repaired path moves the mismatch beyond layer 0, so the next root cause is an accumulating later-layer transform, expert-routing/expert-weight layout, residual path, or layer-local GDN behavior after the first layer. | Stop the diagnostic server, keep the `A_log` loader repair as a candidate, and launch a narrower bisection trace over intermediate MoE layers before running normal warmups. |
| GGUF-135 | MoE TP4 layer bisection and full-attention q/k norm repair. | Bisection over layers `0,8,16,24,32,39` showed layer 0 stayed aligned after the `A_log` fix, but layer 8 was already materially drifted. A narrower layers `0-8` trace localized the first major drift to layer 3, the first `full_attention` layer: without the repair, layer-3 call-1 `post_attention` cosine was `0.859239` and `post_mlp` cosine was `0.750201`, while layers 0-2 remained effectively aligned. Source inspection showed full attention uses `Qwen3NextRMSNorm` for `self_attn.q_norm` and `self_attn.k_norm`, bypassing the GGUF-aware `Qwen3_5RMSNorm` runtime offset fix used by decoder input and post-attention norms. Replacing full-attention q/k norm modules with `Qwen3_5RMSNorm` under the same GGUF runtime norm-fix env repaired the branch probe: GGUF returned `Thinking Process`, matching HF. Layers 0-8 then matched HF nearly exactly; layer-3 call-1 `post_attention` improved to min cosine `0.999999`, `post_mlp` to `0.999998`, and event-1 logits ranked `Thinking` id `90700` first and `Here` id `8160` second. | Promote as source fix candidate; still reject as benchmark candidate until the full ladder passes. | This identifies a second real served-path GGUF norm-offset bug. The fix is model-source-level, not a launch flag: full-attention q/k norms were using a norm class that applies vLLM's `1 + weight` behavior without the GGUF correction. The short branch probe is now semantically aligned, but it is not yet proof of the 131K graph-mode benchmark path. | Build a clean non-tracing full-context TP4 graph-mode GGUF launch with both the `A_log` loader repair and full-attention q/k norm repair, then run normal benchmark warmups -> `c1_128` uncapped strict -> `c1_2000` -> `c1_10000` before comparing against FP16 v0.2.1 numbers. |
| GGUF-136 | MoE TP4 clean 131K graph-mode ladder with `A_log` plus full-attention q/k norm repairs. | Launched the clean non-tracing TP4 graph-mode path on `.20` with P2P-on, `MAX_MODEL_LEN=131072`, the release MoE tuned config, the staged Qwen3.6 35B-A3B F16 GGUF artifact, and the combined `A_log` plus full-attention q/k norm candidate overlay. The same two-token branch probe returned `Thinking Process` without tracing. The normal `run_v02_profile_benchmark.sh` ladder then ran eight 2000-token warmups, uncapped strict, `c1_2000`, and `c1_10000`. Warmups stabilized around `73.36` backend TPS. The uncapped strict gate passed with `finish_reason=stop`, `qwen_gate_valid=true`, `3428` completion tokens, and `71.916` backend TPS. Fixed-token tiers completed at `73.380` backend TPS for `c1_2000` and `65.749` backend TPS for `c1_10000`. | Promote as MoE GGUF correctness candidate; reject as performance candidate. | This is the first MoE GGUF TP4 path in this run set that clears the strict thinking gate under the full 131K graph-mode benchmark harness. It still misses the FP16 v0.2.1 TP4 target by a large margin: reference TP4 is strict `114.725`, c1_2000 `116.429`, and c1_10000 `109.531` backend TPS on `.20`. The active blocker has moved from semantic correctness to GGUF MoE execution speed. | Do not rerun the same ladder without a performance-path change. Next source target is the GGUF MoE execution path, especially the forced unquantized MoE/fused-expert path, FusedMoE tensor materialization, graph capture overhead, and whether the GGUF loaded F16 expert tensors can enter the same fast path as HF weights. |
| GGUF-137 | MoE TP4 trace-gate-off ladder. | Relaunched the same `.20` TP4 graph-mode GGUF source candidate with `VLLM_GFX906_INTERNAL_BOUNDARY_TRACE=0`, P2P-on, `MAX_MODEL_LEN=131072`, the release MoE tuned config, and the normal benchmark ladder. A direct smoke prompt produced coherent content and `finish_reason=stop`. Eight warmups settled at about `73.22` to `73.25` backend TPS. The uncapped strict request passed with `finish_reason=stop`, `qwen_gate_valid=true`, `5302` completion tokens, and `69.961` backend TPS. Fixed-token tiers completed at `73.228` backend TPS for `c1_2000` and `65.643` backend TPS for `c1_10000`. | Reject as performance fix; promote as negative evidence. | Explicitly disabling the request-gated trace path did not improve decode. The c1_10000 result stayed in the same band as GGUF-136 and strict was slower because the model generated a longer strict answer. The performance gap is not explained by the active trace env gate alone. | Stop repeating trace/no-trace launch variants. Next source work should inspect GGUF quant/materialization and MoE/GatedDeltaNet hot-path parity against HF weights, especially whether F16 GGUF tensors are still routed through GGUF adapter methods instead of ordinary FP16 module weights. |
| GGUF-138 | MoE TP4 quant-method selection audit. | Copied the GGUF-136/137 source candidate into a quant-method audit overlay and launched the same `.20` TP4 graph-mode path with P2P-on, `MAX_MODEL_LEN=131072`, release MoE tuned config, and an append-only `GGUFConfig.get_quant_method()` audit file. The server reached health. The audit recorded `1040` method selections across the four TP workers: `760` `LinearBase` entries all selected `UnquantizedLinearMethod`, `160` `FusedMoE` entries all selected forced `UnquantizedFusedMoEMethod`, and `120` attention entries had no GGUF quant method. There were zero `GGUFLinearMethod`, `GGUFEmbeddingMethod`, or `GGUFMoEMethod` records for the hot serving modules. A raw `Hello` completion returned coherent comma-led text; a short thinking-mode chat capped at 128 tokens stayed inside reasoning text. | Reject GGUF adapter-method overhead as the missing MoE performance path; promote as materialization evidence. | The slow MoE GGUF TP4 band is not explained by F16 linears accidentally running through the GGUF quantized linear adapter or by experts missing the unquantized FusedMoE method. Hot linears such as `linear_attn.in_proj_qkvz`, `linear_attn.in_proj_ba`, `linear_attn.out_proj`, full-attention `qkv_proj` / `o_proj`, and shared-expert linears were already ordinary unquantized FP16 methods; MoE experts were already using the tuned `AMD_GFX906` unquantized FusedMoE path. | Move the next performance investigation away from `GGUFLinearMethod` bypasses. Compare GGUF versus HF execution-path parity inside Qwen3.5-MoE after materialization: GatedDeltaNet/linear-attention kernels, shared-expert routing, residual/normalization path, graph capture differences, and whether loaded F16 tensors enter the same compiled graph structure as HF weights. |
| GGUF-139 | MoE TP4 default-scheduler / broad graph-capture ladder. | Relaunched the GGUF-136/137 source candidate on `.20` with P2P-on, `MAX_MODEL_LEN=131072`, the official v0.2 image, release MoE tuned config, and the native `moe35b_tp4_fullbar_p2pon` GGUF route, but removed the restrictive `--max-num-seqs 1 --max-num-batched-tokens 1024` launch args. Startup restored the HF-style scheduler and graph-capture shape: `max_num_batched_tokens=2048`, broad cudagraph capture sizes through `512`, `max_cudagraph_capture_size=512`, `PIECEWISE=51`, `FULL=35`, and graph capture finished in `168` seconds. The normal benchmark ladder ran eight warmups at about `72.793` to `73.145` backend TPS after first-request overhead. Strict passed with `finish_reason=stop`, `qwen_gate_valid=true`, `4124` generated tokens, and `71.030` backend TPS. Fixed-token tiers completed at `73.130` backend TPS for `c1_2000` and `65.552` backend TPS for `c1_10000`. | Promote launch-shape correction; reject as performance fix. | Removing the restrictive scheduler args restores the HF-like graph config and slightly improves strict versus GGUF-137, but fixed-token throughput stays in the same slow band as GGUF-136/137. The remaining MoE GGUF gap is not explained by narrow graph capture sizes or low `max_num_batched_tokens`. The same Transformers FLA / causal-conv fallback warning still appears on the GGUF route. | Keep the default scheduler shape for future comparable MoE GGUF launches, but do not repeat this ladder unchanged. Next source target is the GGUF CausalLM / Qwen3.5-MoE execution path that still logs the fallback warning while the HF-weight release path reaches the FP16 TP4 band. |
| GGUF-140 | MoE TP4 c1 topk8 fastpath debug / eager 4K smoke. | Launched a diagnostic `.20` TP4 server with P2P-on, `MAX_MODEL_LEN=4096`, `--enforce-eager`, the same official v0.2 image, the active GGUF MoE source overlay, `VLLM_GFX906_GGUF_FORCE_UNQUANT_MOE=1`, and `VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH=force`. The server reached health and a direct raw `Hello` completion returned coherent comma-led text, proving the diagnostic route was live enough for source logging. Startup still resolved `Qwen3_5MoeForCausalLM` with `quantization=gguf`, selected the Triton unquantized FusedMoE path, and emitted the Transformers FLA / causal-conv fallback warning. Grepping the image source found no qwen c1 topk8 overlay strings, and the serving logs contained no topk8 active or rejected messages. | Reject c1 topk8 fastpath as the current MoE GGUF performance explanation; promote as image/source-boundary evidence. | The published image does not contain the c1 topk8 MoE overlay, so `VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH` is inert for both HF and GGUF in this release image. It cannot explain why HF-weight TP4 reaches the FP16 release band while GGUF TP4 remains around `73` / `65` backend TPS. | Do not repeat topk8 diagnostics until a topk8 patch is intentionally mounted. Continue source-path parity work around GGUF global quantization, CausalLM versus HF Conditional route differences, and whether the Transformers fallback warning is conversion-only or part of the hot inference path. |
| GGUF-141 | MoE/Dense GGUF FLA / causal-conv fallback source audit. | Inspected the published image source without starting a model server. The warning string appears in the vendored Transformers `modeling_qwen3_5.py` and `modeling_qwen3_5_moe.py` constructors. The vLLM GGUF loader creates a meta-device `AutoModelForCausalLM.from_config(...)` dummy model only to extract parameter names and state-dict conversion mappings. The vLLM serving `qwen3_5.py` path imports vLLM QwenNext classes and its Qwen3.5 GatedDeltaNet forward uses `torch.ops.vllm.gdn_attention_core`; the fallback warning string and Transformers torch fallback functions are not in that serving model file. | Reject missing FLA / causal-conv packages as a standalone GGUF throughput fix; promote as loader-versus-hot-path boundary evidence. | The repeated warning is best explained by GGUF loader name-map construction, not by proof that request-time decode is running through the Transformers torch GatedDeltaNet fallback. Installing `fla` or `causal-conv1d` blindly would not target the measured GGUF MoE TP4 performance gap. | Continue with measurable hot-path parity checks: GGUF `load_format` / global quantization effects, CausalLM page geometry versus HF ConditionalGeneration, materialized tensor graph structure, and request-window timing around GatedDeltaNet, MoE experts, residual/norm, and logits. |
| GGUF-142 | GGUF global quantization / load-format source audit. | Inspected the published image and active MoE GGUF overlay for `load_format=gguf`, `quantization=gguf`, `GGUFConfig`, and `quant_config is None` branches. vLLM automatically sets both `quantization` and `load_format` to `gguf` for GGUF files, and `get_quant_config()` returns `GGUFConfig()`. The stock `GGUFConfig` would choose `GGUFMoEMethod` for FusedMoE, but the active MoE overlay explicitly returns `UnquantizedFusedMoEMethod` under `VLLM_GFX906_GGUF_FORCE_UNQUANT_MOE=1` or skipped unquantized F16/BF16/F32 layers. The active MoE `qwen3_5.py` overlay has GGUF-specific A-log, norm-offset, trace, and loader fixes but no remaining broad `quant_config is None` serving fast-path guard. The dense combined overlay still has known `quant_config is None` guard points, including the native SwiGLU gate that was already tested and rejected as a standalone performance fix. | Reject a vague "GGUF quant method is still active" theory for MoE; promote a narrower graph/timing parity target. | The MoE GGUF slowdown is not explained by accidentally running the stock `GGUFMoEMethod` after the force-unquantized hook. The remaining quantization/load-format risk is subtler: non-None `GGUFConfig`, CausalLM page geometry, graph structure, materialized tensor layout, or request-time kernel selection may still differ from HF even when expert and linear layer methods are unquantized. | Next useful experiment should measure hot-path timing or graph/kernels for the current best GGUF baseline versus the HF control, preferably around GatedDeltaNet, FusedMoE experts, shared expert, residual/norm, and logits. Do not repeat another quant-method selection audit unchanged. |
| GGUF-143 | `gdn_attention_core` custom-op timing viability control. | Inspected the official release image source and ran a tiny `.20` ROCm control with `direct_register_custom_op` under `torch.compile(fullgraph=True)`. The image does not expose `gdn_attention_core` as a C++ extension symbol; vLLM registers it in Python as a direct custom op whose body calls `self._forward_core(...)` outside the compiled forward. A tiny custom op that records `time.perf_counter_ns()` inside the registered op body compiled and ran three times. A second variant with `torch.cuda.synchronize()` before and after the small tensor operation also compiled and ran three times. | Promote as timing-instrumentation route evidence; reject as benchmark or performance evidence. | This is the first timing route that survives the same direct custom-op registration shape without inserting Python timing calls into the compiled Qwen forward. It does not time the real model yet, and it does not cover projection, output projection, logits, or sampler. It does show that an env-gated wrapper around the existing `gdn_attention_core` custom-op body is more viable than the rejected raw Python timing hooks, raw NVTX calls, or profiler record-function ops inside model forward. | Build the next diagnostic as a copied source bundle that times only the registered `gdn_attention_core` wrapper first, with an env gate and append-only per-rank summary. Compare GGUF versus HF under the smallest faithful request window before attempting a full warmup or fixed-token ladder. |
| GGUF-144 | MoE TP4 `gdn_attention_core` request-window timing comparison. | Copied the installed QwenNext source and added an env-gated timing wrapper around the registered `gdn_attention_core` body. The first real-model sync variant fixed the missing-import bug but failed during graph/memory profiling with HIP stream capture invalidation, so synchronized timing is rejected for graph-mode serving. The no-sync GGUF variant reached health under the native TP4/P2P-on/131K graph path and returned the faithful two-token branch response `Thinking Process`. The HF-weight comparator needed the same env-gated hybrid KV-layout bypass used by the GGUF graph route, then reached health and returned the same response. Both request windows added `240` GDN timing records. Excluding the shared layer-0 first-request/prefill outlier, GGUF averaged `1,039,559` ns across per-layer average `last_ns` values and HF averaged `1,049,320` ns; max non-layer-0 `last_ns` was `1,344,764` for GGUF and `1,376,768` for HF. | Promote as request-time timing-boundary evidence; reject GDN core timing as the current MoE GGUF performance bottleneck. | The steady GatedDeltaNet custom-op body is effectively at parity between the current best GGUF path and the HF-weight comparator on this faithful two-token request. The large layer-0 outlier appears on both paths and should be treated as first-request/prefill/startup noise, not a GGUF-only slowdown. | Move the next timing target outside `gdn_attention_core`: input/output projections, MoE/shared-expert execution, residual/norm, graph/kernel selection, logits, and whole decode-loop scheduling. Prefer lower-level or external profiling if Python wrapper side effects interfere with graph capture. |
| GGUF-145 | MoE TP4 FusedMoE direct custom-op timing comparison. | Copied the installed `DefaultMoERunner` source and added an env-gated no-sync timing wrapper around the registered `torch.ops.vllm.moe_forward` and `torch.ops.vllm.moe_forward_shared` bodies, which call `layer.runner.forward_impl(...)`. A GGUF graph-mode TP4/P2P-on/131K run reached health and returned `Thinking Process`, but live requests added zero timing rows because the Python body ran during profiling/capture and not during graph replay. The same hook on an HF-weight graph comparator did add request-window rows. Eager-mode comparators for GGUF and HF both reached health, returned `Thinking Process`, and added `480` request-window rows each. | Promote as timing-boundary evidence; reject the steady FusedMoE direct custom-op body as the primary remaining GGUF-vs-HF throughput bottleneck. | In eager request-window timing, GGUF and HF were at parity on token-sized shapes: `(1, 2048)` averaged `645,803` ns for GGUF and `649,962` ns for HF, while `(32, 2048)` averaged `678,506` ns for GGUF and `694,144` ns for HF. The only large shape-level gap was `(15, 2048)`, where GGUF averaged `7,337,458` ns and HF averaged `2,759,635` ns, but this was a layer-0 first-request outlier. Excluding layer 0, `(15, 2048)` averaged `808,102` ns for GGUF and `801,966` ns for HF. The graph-mode GGUF wrapper limitation also proves Python direct-op timing can miss replay-time behavior. | Do not spend the next pass blindly optimizing `UnquantizedFusedMoEMethod` or `DefaultMoERunner.forward_impl` without lower-level evidence. The next source target should move to graph-replay-visible timing or whole-decode profiling around residual/norm, projections outside the direct op, logits/sampling, scheduler/replay overhead, and CausalLM-versus-HF graph structure. |
| GGUF-146 | MoE TP4 post-forward logits and sampler timing comparison. | Copied the installed `LogitsProcessor` and v1 `Sampler` source files and added env-gated synchronized timing around `lm_head.quant_method.apply(...)`, TP logits gather, `Sampler.sample(...)`, and total sampler forward. The GGUF TP4/P2P-on/131K graph path reached health after recompilation and returned the faithful branch response `Thinking Process`. A clean HF comparator without the hybrid KV-layout bypass failed at the known ambiguous cache layout assertion, so the successful HF comparator used the same KV-layout bypass template from earlier graph diagnostics. The HF request used the HF served model name and returned the same `Thinking Process` response. Both request windows added `60` timing rows. | Promote as post-forward timing evidence; reject logits processing and sampler as the primary current MoE GGUF throughput bottleneck. | GGUF and HF were effectively at parity: `lm_head_apply` averaged `393,201` ns for GGUF and `373,102` ns for HF; `gather_logits` averaged `236,504` ns versus `230,403` ns; `get_logits_total` averaged `735,715` ns versus `727,748` ns; `sampler_forward_total` averaged `250,274` ns versus `232,938` ns; and `sample_inner` averaged `76,387` ns versus `79,262` ns. These sub-millisecond differences cannot explain the much larger fixed-token GGUF-vs-HF MoE TP4 TPS gap. | Stop targeting visible post-forward logits/sampler work unless a longer-window profiler contradicts this. The next source target should be graph-visible whole-model timing or structural graph comparison: CausalLM-versus-HF wrapper/page geometry, graph replay/kernels, residual/norm/projection regions outside the already-timed direct ops, or model-runner scheduling overhead. |
| GGUF-043 | TP4 FP16/half minimal Qwen3.5 source bundle. | A minimal `qwen3_5.py` was built from the image-base source with only GGUF embedding/lm-head F16 materialization and Q/K repeat-interleave. It loaded and served, but raw and chat probes returned invalid byte/glyph garbage. | Reject as benchmark candidate; promote as clean failure evidence. | The old bundle is not required merely to produce output, but the minimal materialization route still corrupts model semantics. This points back to GGUF Qwen3.5 execution/materialization details rather than a broad launch issue. | Add one source dependency at a time from the old bundle and keep rejecting any run whose warmup probes are incoherent. |
| GGUF-044 | TP4 FP16/half minimal Qwen3.5 source plus logical GGUF shard-order patch. | The same minimal source was paired with the logical shard-order `gguf.py` patch. It still served and produced invalid byte/glyph garbage on raw and chat probes. | Reject logical shard-order as sufficient. | Logical shard ordering alone does not repair the FP16/half GGUF semantic failure. | Continue isolating loader/materialization and Qwen3.5 cache helpers before running warmups. |
| GGUF-045 | TP4 FP16/half minimal Qwen3.5 source with old GGUF loader/registry and logical shard patch, without cache helpers. | Engine startup failed during hybrid KV/Mamba cache setup with a page-size divisibility error before serving. | Reject as serving path; promote cache-helper dependency. | The old loader/registry pair changes the cache setup requirements. Without the Qwen3.5 Mamba/cache helper behavior from the old diagnostic bundle, the server cannot reach requests. | Add only the required cache-helper behavior to the minimal bundle and retest correctness. |
| GGUF-046 | TP4 FP16/half minimal Qwen3.5 source with cache helpers plus old GGUF loader/registry and logical shard patch. | The server reached health and answered deterministic probes, but raw `Hello`, chat `Hello`, and a short factual chat prompt all produced invalid byte/glyph garbage. | Reject as benchmark candidate. | Cache helpers solved the engine-init/page-size class of failure but did not repair GGUF semantic correctness. The model can decode tokens, but the learned distribution is still corrupted before any warmup or benchmark tier. | Stop this patch line as a promotion candidate. Next source work should compare the minimal route against the known-good HF-weight FP16 path at embedding, first layer input, GatedDeltaNet state/cache, and lm-head boundaries. |
| GGUF-047 | TP4 FP16/half minimal Qwen3.5 source with wrapper `load_weights` delegation. | The wrapper loader was changed to delegate into `self.language_model.load_weights()` after stripping the `language_model.` prefix, matching the older diagnostic bundle. The server reached health and the F16 GGUF embedding/lm-head materialization markers fired. Raw `Hello` still returned the same invalid byte/glyph prefix as GGUF-046, and chat probes remained invalid. | Reject wrapper delegation as semantic fix. | The active source was patched and the output did not move, so the minimal-byte-garbage failure is not explained by only the outer wrapper weight-loader mapping. | Continue comparing runtime boundaries rather than adding more broad bundle behavior. |
| GGUF-048 | llama.cpp CPU file-level smoke on the same FP16/half GGUF. | A local CPU `llama-cli` build was run against `Qwen3.6-27B-F16.gguf` with `Hello`, `-n 8`, and temperature 0. The run hit the 15-minute timeout before producing a useful answer. | Inconclusive. | The model is 51 GB F16 and the available llama.cpp build is CPU-only, so timeout is not proof that the GGUF file is bad. It also does not prove the file is coherent. | Use a ROCm-enabled llama.cpp build or a smaller representative GGUF slice if file-level validation is needed; otherwise keep focusing on vLLM boundary traces. |
| GGUF-049 | ROCm7.2 llama.cpp artifact smoke on the same FP16/half GGUF. | A separate ROCm7.2 llama.cpp build was created under the NVMe checkout with `GGML_HIP=ON` and `AMDGPU_TARGETS=gfx906`. The first `llama-cli` run loaded the model across four GPUs and produced coherent Qwen-style text before entering an interactive prompt loop. A clean `llama-completion` run on the same file returned `<think></think>` followed by `Hello! How can I help you today?` with GPU offload. | Promote as artifact-coherence evidence. | The exact same F16 GGUF file can produce coherent text in an independent ROCm GGUF runner. The current garbage-output blocker is therefore in the vLLM GGUF/Qwen3.5 execution or materialization path, not simply a corrupt GGUF artifact. | Compare vLLM against llama.cpp / HF-weight boundaries at token IDs, embedding output, layer-0 GatedDeltaNet inputs/state, output projection, and lm-head logits. |
| GGUF-050 | GGUF embedded tokenizer versus HF tokenizer comparison. | The `tokenizer.ggml.tokens` array in the F16 GGUF was decoded and compared against the HF tokenizer mounted from the text-config path. The shared ID range matches for sampled IDs and known strings, including `Hello`, `<think>`, `</think>`, `_manifest`, `hh`, and the byte-fallback token for `阅`. The GGUF token array has extra padded/special entries beyond the HF tokenizer length, but the shared ordering matches. | Reject tokenizer-order mismatch as the primary failure. | vLLM's weird raw output is not explained by a simple HF-versus-GGUF vocab ID shift. The model is actually assigning high probability to bad tokens under the matching tokenizer IDs. | Continue at model execution boundaries rather than tokenizer replacement. |
| GGUF-051 | TP2 F16/half vLLM minimal delegate smoke. | After clearing stale llama.cpp VRAM, TP2 was retried with the same minimal delegate source, P2P-on, two MI50s, `MAX_MODEL_LEN=4096`, and lower memory utilization. Startup failed because the F16 model left `0.0 GiB` available KV cache memory; even one request at 4K needed more KV cache than available. | Inconclusive for correctness; reject TP2 as practical F16 discriminator on 2x MI50. | The failure is capacity, not semantic output. TP2 cannot be used to classify the F16 GGUF vLLM bug on this hardware without more aggressive context/memory surgery that would no longer match the benchmark direction. | Keep TP4 as the minimum viable F16 GGUF vLLM discriminator on `.20`; use llama.cpp as the artifact control. |
| GGUF-052 | TP4 F16/half vLLM row-checksum diagnostic for GGUF embedding and lm-head. | A separate diagnostic overlay printed rank-local metadata and sampled row checksums for `token_embd.weight` and `output.weight` after vLLM materialized GGUF qweights. The sampled rows matched a direct GGUF reader baseline for token IDs `9419`, `19748`, `71741`, `248068`, and `248069`. The same run still returned deterministic garbage for raw `Hello`: `�..��&#hh_manifest` with high probability on byte/punctuation tokens. | Reject embedding/lm-head row-offset mismatch as the primary fix; promote as boundary evidence. | The active F16 GGUF file is coherent in ROCm llama.cpp, tokenizer IDs match over the shared range, and vLLM's TP4 embedding/lm-head sampled rows now match direct GGUF data. The corruption therefore occurs after correct-looking token embedding and before final token distribution, most likely in Qwen3.5/Qwen3.6 layer execution, GatedDeltaNet state/cache behavior, or an internal projection/reassembly path. | Trace the F16 TP4 path at layer-0 normalized input, q/k/v/z plus beta/alpha projections, GatedDeltaNet core/state, output projection, and logits; do not run warmups or throughput tiers until raw/chat probes are coherent. |
| GGUF-053 | TP4 F16/half vLLM layer-0 diagnostic after row checks. | A follow-up overlay traced layer-0 normalized input, q/k/v segments, z, beta/alpha, GatedDeltaNet core output, output-projection input, and output-projection output for the same raw `Hello` request. The output remained deterministic byte/glyph garbage, but the actual request path showed nonzero q/k/v/z, beta/alpha, nonzero GatedDeltaNet core output, nonzero output-projection input, and replicated nonzero output-projection output. | Reject the all-zero or dead-core theory; promote layer-activity boundary evidence. | The vLLM F16 GGUF path is not simply dropping layer-0 activity. Combined with GGUF-052, the next target is semantic divergence inside GatedDeltaNet state/cache handling, projection ordering, or TP reassembly after active projections are formed. The all-rank log is interleaved, so it is not yet clean enough for exact per-rank numeric comparison. | Run a rank-filtered or structured trace on the benchmark-prompt prefill path and compare it against the coherent TP1/llama.cpp/HF-weight boundary before any more warmups or throughput tiers. |
| GGUF-054 | TP4 F16/half rank0-only layer trace on the benchmark warmup prompt. | The normal warmup prompt was confirmed from `run_chat_capture.py`: a synthetic instruction plus the calibration phrase repeated 32 times, producing a 431-token chat prefill in this run. A rank0-only TP4 trace with `max_tokens=1`, `min_tokens=1`, and `ignore_eos=true` returned only `;`, not coherent Qwen reasoning. The request emitted shared-memory broadcast wait warnings before completion. Layer-0 rank0 traces showed nonzero normalized input, q/k/v/z, beta/alpha, GatedDeltaNet core output, output-projection input, and output-projection output for the 431-token prefill. | Reject the current F16 GGUF path as a warmup candidate; promote the benchmark-prompt prefill trace as the active boundary. | The failure is not an absent prefill or zeroed layer. The prompt path runs through layer-0 but produces semantically wrong logits after a slow/stalled prefill window. This explains why full warmups were garbage and confirms that strict cannot promote from this state. | Compare benchmark-prompt prefill/state against a coherent control and inspect Qwen3.5 GatedDeltaNet recurrent state update, chunked prefill/decode transition, and final logit distribution before running any more warmups. |
| GGUF-055 | HF dense TP8 official-image release-overlay control. | A control launch against the HF-weight dense model resolved `Qwen3_5ForConditionalGeneration` and showed release overlay hooks, but the manual launch stalled/idled before API readiness and was stopped. | Reject as calibration path. | The manual startup plumbing was not healthy enough to compare against published numbers, and no benchmark tier completed. | If an HF control is needed again, use the exact release deploy path or a known-good local snapshot instead of ad hoc manual launch. |
| GGUF-056 | Combined release overlay plus GGUF tensor-fix bundle. | A combined patch bundle was built from the release runtime overlay with the proven GGUF Qwen3.5 tensor fixes: norm handling, linear-attention head/state permutation, F16 materialization, and explicit tokenizer/config route. | Promote as dense GGUF correctness precondition. | This is the first bundle that keeps release hooks while preserving the GGUF tensor-layout fixes required for coherent output. | Use this bundle as the baseline for dense GGUF benchmark attempts. |
| GGUF-057 | Dense FP16 GGUF TP8 combined overlay benchmark on experimental image. | The normal sequence completed with coherent output and a valid strict gate. Results: `c1_128` strict backend `60.863` TPS, `c1_2000` backend `61.383` TPS, and `c1_10000` backend `57.986` TPS. | Reject performance; promote correctness evidence. | Strict validity is now solved for this path, but decode is below published dense FP16 targets: strict `69.514`, `c1_2000` `70.347`, and `c1_10000` `66.069`. | Compare the active GGUF source path against release HF path before changing launch flags. |
| GGUF-058 | Dense FP16 GGUF TP8 combined overlay benchmark on official release image. | The same combined overlay was mounted on the official release image. Results: `c1_128` strict backend `60.656` TPS, `c1_2000` backend `61.399` TPS, and `c1_10000` backend `57.966` TPS. | Reject performance; promote image-control evidence. | The throughput gap is not explained by using the older experimental image. Official image plus the same source overlay stays in the same decode band. | Treat the remaining gap as source/execution-path overhead, not image-base drift. |
| GGUF-059 | Release-attention merge for GGUF dense. | A new bundle preserved the GGUF tensor fixes while restoring the release `Qwen3_5Attention` subclass/routing. Full ladder results: `c1_128` strict backend `60.764` TPS, `c1_2000` backend `61.362` TPS, and `c1_10000` backend `57.965` TPS. | Reject performance; promote negative source evidence. | Restoring the release attention subclass did not move the decode ceiling. The bottleneck is not simply missing that attention wrapper. | Continue profiling GatedDeltaNet / GGUF execution path rather than attention-class routing alone. |
| GGUF-060 | Experimental F16 merged-single-matmul GGUF quantization patch. | A GGUF loader patch tried to materialize F16 merged shards once in logical order and clear the shard map. The marker did not fire for the important dense linears because `in_proj_qkvz.weight`, `in_proj_ba.weight`, `conv1d.weight`, and `out_proj.weight` already load as normal `weight` params, not GGUF `qweight`. | Reject hypothesis; no benchmark promoted. | The major Qwen3.5 dense linear weights are not paying a quantized GGUF shard-matmul cost in the way this patch assumed. | Stop chasing this path unless profiling contradicts the source-level evidence. |
| GGUF-061 | ConditionalGeneration architecture override for dense FP16 GGUF. | The architecture override resolved `Qwen3_5ForConditionalGeneration`, matched release-style attention block size `400`, and reported Mamba padding `2.17%`. Warmup decode stayed around `60.59` TPS and first-request prefill was much worse than the CausalLM path. | Reject early. | Architecture alignment did not improve decode throughput and degraded prefill behavior. | Keep the coherent CausalLM/GGUF tensor-fix path as the active dense baseline. |
| GGUF-062 | Dense FP16 GGUF no-trace release-attention bundle. | A copy of the release-attention GGUF bundle removed all `_qwen35_trace_tensor(...)` calls from the forward path while preserving tensor fixes, release overlay hooks, P2P-on, TP8, and the official release image. Full ladder results: `c1_128` strict backend `60.678` TPS, `c1_2000` backend `61.289` TPS, and `c1_10000` backend `57.896` TPS. Strict gate remained valid. | Reject performance; promote negative source evidence. | Removing trace-call environment checks did not improve decode TPS. The same launch also logged the FLA / causal-conv1d fallback seen in the previous release-attention run. | Stop chasing trace overhead. Compare fast-path availability/parity and GatedDeltaNet execution against the published HF-weight FP16 path. |
| GGUF-063 | Clean HF-weight dense TP8 official-image release-overlay control. | The official release image was launched with a clean release patch bundle and the complete HF-weight dense snapshot, P2P-on, TP8, `MAX_MODEL_LEN=131072`, and the normal warmups -> `c1_128` strict -> `c1_2000` -> `c1_10000` ladder. Results: `c1_128` strict backend `69.652` TPS with `qwen_gate_valid=true`, `c1_2000` backend `70.202` TPS, and `c1_10000` backend `65.952` TPS. | Promote as control evidence; reject host/image explanation for GGUF gap. | The same host lane, official image, release overlay, benchmark harness, and P2P-on path reproduce the published dense band with HF weights while coherent GGUF stays around `58` to `61` backend TPS. The remaining gap is therefore specific to the GGUF source/materialization/execution path. | Profile GGUF versus HF at Qwen3.5 GatedDeltaNet / linear-attention boundaries and confirm why GGUF logs the slower fallback path while HF reaches the release decode band. |
| GGUF-064 | Dense FP16 GGUF fused-GDN projection parity experiment. | A new GGUF patch bundle kept the proven tensor conversions but changed Qwen3.5 GatedDeltaNet back to the HF release fused qkv/z/b/a projection shape. The loader routed `in_proj_b` and `in_proj_a` into fused shard IDs `4` and `5`, and layer-0 `in_proj_qkvz.weight` materialized as `(2060, 5120)` per rank. The server reached API readiness, but the first short chat smoke request did not return, GPUs stayed idle after entry, and the response file remained empty until the test was stopped. | Reject as serving candidate; promote as source-boundary evidence. | Fusing b/a into the release projection shape is not a drop-in GGUF performance fix. The path can load and compile, and it reaches the vLLM FLA warning path, but request execution hangs before usable output or benchmark tiers. | Do not run warmups or throughput tiers from this bundle. The next source target is the active GatedDeltaNet call/input layout and state semantics when fused projection shape is used, or a lower-level profile of the current coherent split-GDN path. |
| GGUF-065 | Dense FP16 GGUF native SwiGLU enablement for GGUF load-format. | A disposable patch relaxed the release `Qwen2MoeMLP` SwiGLU guard only when `VLLM_GFX906_QWEN_MLP_INTERLEAVED_SWIGLU_ALLOW_GGUF_F16=1` and GGUF quant config was active. The official image loaded, reached API readiness, logged 8 native extension loads and 512 layer/rank SwiGLU enablements, and a short smoke returned coherent Qwen-style reasoning. Full ladder results: `c1_128` strict backend `61.053` TPS with `qwen_gate_valid=true`, `c1_2000` backend `61.780` TPS, and `c1_10000` backend `58.266` TPS. | Reject performance; promote source-boundary evidence. | GGUF load-format had been disabling the native MLP SwiGLU extension through the `quant_config is None` guard, but enabling that extension is not enough to close the dense gap. The run still logged the FLA / causal-conv1d fallback and stayed far below the HF-weight release-overlay control. | Keep the native SwiGLU enablement as a possible small component only. The next source target is GatedDeltaNet / linear-attention path parity, especially why GGUF keeps the fallback path while HF reaches the release band. |
| GGUF-066 | Dense FP16 GGUF `GGUFLinearMethod.apply()` hot-path diagnostic. | A disposable `gguf.py` overlay logged the first `GGUFLinearMethod.apply()` calls during graph profiling and one short coherent chat request on the official image plus coherent release-attention GGUF bundle. The loader-time `transformers` fast-path warning was traced to vLLM's GGUF name-map dummy model construction. During the sampled hot path, all recorded `GGUF_APPLY_DIAG` calls were `ParallelLMHead`; no internal transformer-layer linear appeared in the sample. | Reject broad internal GGUF matmul overhead as the primary dense gap; promote hot-path boundary evidence. | The measured dense gap is not explained by every Qwen block linear running through `_fused_mul_mat_gguf`. The remaining likely targets are lm-head GGUF application cost, Qwen3.5 CausalLM versus ConditionalGeneration execution differences, GatedDeltaNet input/state/core timing, graph-capture behavior, or another source-level route after internal projections have materialized. | Do not repeat broad GGUF-linear speculation. If profiling continues, measure lm-head cost separately and compare GatedDeltaNet/core timings against the clean HF-weight release path. |
| GGUF-067 | Dense FP16 GGUF lm-head-only unquantized embedding method. | A narrow `gguf.py` overlay forced only `lm_head` to use `UnquantizedEmbeddingMethod` while keeping the official image, release-attention/no-trace GGUF bundle, P2P-on, TP8, and normal warmups -> strict -> fixed-token ladder. The marker fired on all 8 TP ranks. Full ladder results: `c1_128` strict backend `60.676` TPS with `qwen_gate_valid=true`, `c1_2000` backend `61.309` TPS, and `c1_10000` backend `57.888` TPS. | Reject performance; promote lm-head boundary evidence. | Removing the visible lm-head `GGUFLinearMethod.apply()` path did not improve dense decode. The remaining gap is not explained by lm-head GGUF matmul overhead alone. | Stop chasing isolated lm-head method selection as the missing lever. Next compare GatedDeltaNet/core timing, graph capture, sampler/logits cost, and CausalLM-versus-HF release execution. |
| GGUF-068 | MoE GGUF artifact readiness inventory on `.20`. | Targeted listings under the known NVMe model/cache roots found the dense 27B F16 GGUF and older 27B GGUF/header material, but did not find a Qwen3.6 35B-A3B MoE GGUF artifact. | Promote as readiness evidence; no MoE benchmark attempted. | The MoE side of the GGUF goal cannot start from the current `.20` artifact set without first staging or producing a MoE GGUF package. | Before MoE GGUF testing, stage the exact intended Qwen3.6 35B-A3B MoE GGUF package on NVMe, pin its tokenizer/config route, and run the same warmups -> strict -> fixed-token ladder only after coherence passes. |
| GGUF-069 | Patient rerun of dense FP16 GGUF fused-GDN projection parity. | The earlier fused-GDN path was rerun with a longer patience window. Direct one-token completion returned immediately, a short chat smoke returned coherent Qwen-style reasoning, and the normal warmups -> strict -> fixed-token ladder completed. Results: `c1_128` strict backend `63.201` TPS with `qwen_gate_valid=true`, `c1_2000` backend `63.836` TPS, and `c1_10000` backend `60.169` TPS. | Reject benchmark performance; promote fused-GDN source evidence. | Fusing beta/alpha into the release-shaped GatedDeltaNet projection is a real improvement over the split-GDN GGUF band, but it still does not match or beat the published FP16/HF dense release numbers. The prior "hang" classification was too strong; the path is serving-capable with patience. | Superseded by `GGUF-070` as the current best dense GGUF source baseline. |
| GGUF-070 | Dense FP16 GGUF fused-GDN plus native GGUF SwiGLU combination. | A combined patch bundle used the patient fused-GDN projection path and enabled the GFX906 native interleaved SwiGLU extension under the explicit F16 GGUF guard. The official release image reached API readiness, completed eight normal warmups, passed the uncapped strict gate, and completed the fixed-token tiers. Results: `c1_128` strict backend `63.679` TPS with `qwen_gate_valid=true`, `c1_2000` backend `64.353` TPS, and `c1_10000` backend `60.617` TPS. | Reject benchmark performance; promote as the current best dense GGUF source baseline. | Combining fused-GDN with native GGUF SwiGLU is additive and improves over fused-GDN alone, but it remains below the FP16/HF release-band control and below the published dense targets. Warmups were stable around `64.2` to `64.35` backend TPS, so the path is coherent and serving-capable; it is not fast enough for promotion. | Use this combined bundle as the next dense GGUF baseline for profiling. The next source target is the remaining GGUF-specific execution gap: GatedDeltaNet / linear-attention fast-path parity, CausalLM versus HF release execution, graph/logits/sampler cost, and any residual GGUF adapter overhead after materialization. |
| GGUF-071 | Dense FP16 GGUF ConditionalGeneration rerun with fused-GDN plus native GGUF SwiGLU. | The `GGUF-070` bundle was relaunched with `Qwen3_5ForConditionalGeneration` instead of `Qwen3_5ForCausalLM`, keeping P2P-on, TP8, official release image, fused-GDN, native GGUF SwiGLU, and the normal warmups -> strict -> fixed-token ladder. The run was coherent and strict-valid. Results: `c1_128` strict backend `63.003` TPS with `qwen_gate_valid=true`, `c1_2000` backend `63.656` TPS, and `c1_10000` backend `60.109` TPS. | Reject performance; promote negative source evidence. | Rechecking ConditionalGeneration after fused-GDN and native GGUF SwiGLU does not close the gap. It is slower than the `GGUF-070` CausalLM baseline and remains below the FP16/HF release path, so architecture override is not the missing dense GGUF lever. | Keep `GGUF-070` as the baseline. Next work should profile the remaining CausalLM baseline gap instead of repeating architecture flips. |
| GGUF-072 | HF-control versus best-GGUF log comparison. | Compared the clean HF-weight dense TP8 control log against the `GGUF-070` best-GGUF log. Both use the same official image, P2P-on lane, TP8, `MAX_MODEL_LEN=131072`, chunked prefill, graph-capture mode, capture sizes `[1, 2, 4]`, and graph capture finished in about 2 seconds with about `0.42 GiB`. HF resolved as `Qwen3_5ForConditionalGeneration`, `load_format=auto`, `quantization=None`, attention block size `400`, Mamba padding `2.17%`, and hit eight ROCm GELU fallback warnings. GGUF resolved as `Qwen3_5ForCausalLM`, `load_format=gguf`, `quantization=gguf`, attention block size `208`, Mamba padding `4.26%`, and hit eight Transformers fast-path fallback warnings during model setup. The GGUF combined bundle hashes were `qwen3_5.py` `e88db243d01de8592e107a17a7d6195ac89af4fea5c475cf8ec454104e28a5ea` and `qwen2_moe_interleaved_swiglu_20260608.py` `d7022e939598730de1506d9a31db98be33921652ec561e728665b2bc72766fb5`; the clean release control `qwen3_5.py` hash was `71eaf52b5f85c022380599ae80ce0f478e989b6052a4f14a88ef4305edd3c046`. | Reject as a direct fix; promote diagnostic boundary evidence. | Graph-capture configuration and capture duration look equivalent, so graph setup alone is not a strong explanation for the `8` to `9` TPS dense gap. Page geometry and wrapper type differ, but `GGUF-071` already showed that forcing the GGUF path to ConditionalGeneration is slower. The fallback strings differ in source and timing, so raw fallback count is not useful without hot-path timing. | Profile the current `GGUF-070` CausalLM baseline against HF-control internals at GatedDeltaNet/core, linear-attention, output projection, logits/sampler, and GGUF adapter/materialization boundaries before running another full ladder. |
| GGUF-073 | Python timing hooks inside the compiled Qwen3.5 GatedDeltaNet forward. | Created timing-only experimental patch bundles from the current best GGUF bundle and the clean HF bundle. The hook added optional prints around input projection, `torch.ops.vllm.gdn_attention_core`, norm/gate/rearrange, and output projection. The synchronized variant failed during engine startup because Dynamo rejected `torch.cuda.synchronize()` inside the compiled forward. The no-sync variant also failed during engine startup because Dynamo rejected `time.perf_counter()` inside the compiled forward. Both containers were stopped. | Reject Python timing inside compiled forward. | The vLLM compile path traces this forward during memory profiling/startup, so Python timing calls cannot live directly in the compiled model path without changing compile behavior or forcing eager execution. That would not be comparable to the release path. | Use external ROCm profiling, compiler-safe C++/custom-op markers, or built-in vLLM/NVTX-style tracing instead of Python timing calls for the next boundary measurement. |
| GGUF-074 | ROCm profiler wrapper around the dense FP16 GGUF API server. | Wrapped the official release image entrypoint with ROCm 7.2.1 `rocprof --hip-trace --timestamp on --stats` and the `GGUF-070` best-GGUF bundle. The server loaded the model and entered graph compilation, but no benchmark request was sent before shutdown. The profiler output contained startup logs, per-process handle files, and a parent-process HIP API trace, but no kernel-summary CSV and no request-level decode window. The HIP trace showed no `hipLaunchKernel` or `hipMemcpy` lines in the captured parent process. | Reject as hot-path evidence; promote profiler-shape evidence. | Wrapping the API server process is too broad and did not capture the worker decode kernels needed to explain the dense GGUF gap. The run is useful only because it proves this profiler shape can waste time while missing the child-worker hot path. | Do not repeat API-server-wrapper profiling. If using `rocprof`, use a narrow request-window/offline invocation, a profile input file, or a worker/single-process path that actually captures decode kernels. |
| GGUF-075 | `rocprofv3 --attach` request-window probe against the current best dense GGUF bundle. | Relaunched the `GGUF-070` fused-GDN plus native GGUF SwiGLU baseline on `.20` with the official release image, P2P-on, TP8, `MAX_MODEL_LEN=131072`, explicit HF tokenizer/config, and the normal begin-think proxy. Fixed-token `c1_2000` request-window probes ran while attaching `rocprofv3` to the TP0 worker. Backend decode TPS was `64.386`, `64.362`, and `64.217`, matching the current-best GGUF band. The first profiler sanity check showed the release image is missing `libdw.so.1`; a tiny Torch matmul produced kernel/HIP CSV and JSON only after mounting the host `libdw.so.1` read-only. However, relaunching the GGUF server with the same read-only `libdw` mount still left `rocprofv3 --attach` fileless: attach reported success, but no kernel, HIP, RCCL, CSV, JSON, or summary artifacts were emitted. | Reject `rocprofv3 --attach` as currently used; promote the request result and `libdw` sanity result as profiler-shape evidence. | The request confirms the best GGUF bundle still serves coherently in the expected fixed-token throughput band. The separate tiny Torch sanity proves `rocprofv3` can write artifacts in this image when launched as a wrapper with `libdw` available. The remaining failure is specific to the attach shape, not a general inability to run ROCm profiling. | Next profiler attempt should avoid PID attach in this form. Use an offline/single-process invocation, `rocprofv3` around the exact worker/launcher shape that emits files, a profile-input workflow, or compiler-safe lower-level markers around the GatedDeltaNet/logits boundaries. If `rocprofv3` is used inside the release image, account for the missing `libdw.so.1` without mutating the image. |
| GGUF-076 | `rocprofv3` multiprocessing boundary control. | Ran tiny Torch FP16 matmul controls in the official release image with a read-only `libdw.so.1` mount. Direct execution under `rocprofv3` emitted kernel/HIP CSV and JSON. A Python `multiprocessing` `spawn` child also emitted kernel/HIP CSV and JSON when the parent was launched under `rocprofv3`. A `fork` child did not emit profile files. | Promote as profiler-shape evidence. | vLLM uses `spawn`, so wrapping from process start can in principle follow spawned Python workers and write child kernel traces. The failure of PID attach is not the only possible ROCm profiler route. | If profiling vLLM with `rocprofv3`, prefer launch-time wrapper or a purpose-built worker/offline launcher over attach. First prove the selected shape writes artifacts before running a full model request. |
| GGUF-077 | Launch-time `rocprofv3` wrapper around the current best dense GGUF vLLM server. | Generated a profiler-wrapped copy of the `GGUF-070` launch script using the official release image, P2P-on, TP8, `MAX_MODEL_LEN=131072`, fused-GDN plus native GGUF SwiGLU bundle, explicit HF tokenizer/config, and a read-only `libdw.so.1` mount. The server reached GGUF tensor conversion, compiled the graph, and then faulted during persistent all-reduce startup with GPU memory access faults across workers. No request was sent and no profile files were emitted. The diagnostic container was stopped and removed. | Reject this full-server wrapper shape. | Although the tiny `spawn` control proves launch-time `rocprofv3` can profile child GPU work, wrapping the full vLLM/Persistent-AR server destabilized the active release-like lane before readiness. It is not a comparable profiling route for promotion or throughput diagnosis. | Do not run the full server under `rocprofv3` unchanged. Next options are lower-impact markers, a reduced offline/single-worker reproducer that still reaches the relevant GatedDeltaNet kernels, or profiling a non-AR diagnostic variant only as source localization, not as benchmark evidence. |
| GGUF-078 | Launch-time `rocprofv3` wrapper with Persistent-AR disabled as a non-promotion diagnostic. | Reused the current best dense GGUF launch shape but disabled the GFX906 Persistent-AR and mutable row-parallel AR knobs, then wrapped startup with `rocprofv3` and a read-only `libdw.so.1` mount. The server eventually reached health, but the collection window mostly covered startup. A short capped request completed only `128` tokens in about `50.34` seconds, with vLLM decode TPS about `12.37`, `finish_reason=length`, and `qwen_gate_valid=false`. Stopping the container still did not produce usable profiler artifacts. | Reject as profiling and benchmark evidence; promote only as a negative profiler-shape diagnostic. | Disabling Persistent-AR changes the lane too much to compare with the release-like P2P-on path and makes the request far too slow. The profiling window missed the meaningful request interval and the stop path still did not emit files. | Do not use no-AR `rocprofv3` wrapping as a promotion route. The next profiler direction should be lower-impact markers, profile-input/request-window control with proven artifact emission, or a reduced offline/single-worker reproducer rather than full-server wrapping or PID attach. |
| GGUF-079 | Tiny ROCTX/NVTX marker control under `rocprofv3`. | A tiny Torch FP16 matmul function with `torch.cuda.nvtx.range_push/pop` compiled under `torch.compile(..., fullgraph=False)`. Wrapped with `rocprofv3 --marker-trace --kernel-trace --stats`, it emitted kernel CSV/JSON plus marker API CSV/JSON, and the `tiny_marker` range appeared in the trace. | Promote as profiler capability evidence. | The release image can emit ROCTX marker artifacts when the process shape is simple and the missing `libdw.so.1` is supplied read-only. | Marker tracing is viable in principle, but it must be inserted in a way that survives vLLM's stricter AOT/fullgraph path. |
| GGUF-080 | Raw `torch.cuda.nvtx.range_push/pop` markers inside the current best Qwen3.5 dense GGUF bundle. | Created a copied marker bundle that inserted raw marker ranges around layer-0 `in_proj_qkvz`, `gdn_attention_core`, norm/gate, and `out_proj`. The server failed during startup AOT capture before readiness. Dynamo rejected `range_push` because it returns a non-Tensor. The diagnostic container was stopped and removed. | Reject raw Python marker calls inside compiled Qwen forward. | The tiny control used `fullgraph=False`; vLLM uses a stricter fullgraph/AOT path. Non-Tensor marker calls cannot live directly inside that compiled forward. | Do not repeat raw `torch.cuda.nvtx.range_push/pop` inside Qwen model code. |
| GGUF-081 | Traceable Python custom-op marker control. | A tiny `torch.library` custom op returning its input tensor compiled under `torch.compile(..., fullgraph=True)` and emitted `custom_marker_1` / `custom_marker_2` marker records under `rocprofv3`. PyTorch warned that the custom op output aliases its input. An alias-annotated schema variant failed under Inductor. | Promote only as partial marker-route evidence; reject as ready for model insertion. | A Tensor-returning custom op can survive a tiny fullgraph compile and emit markers, but alias semantics are fragile. The alias-correct schema did not compile in the tiny control. | If this route continues, use a proper lower-level custom op with correct alias/side-effect semantics, not an ad hoc Python aliasing op. |
| GGUF-082 | Python custom-op markers inside the current best Qwen3.5 dense GGUF bundle. | Created a copied bundle with Tensor-returning custom marker boundary points around layer-0 GatedDeltaNet. The server got farther than the raw-marker bundle and recompiled, but failed during runtime shape checks with `wrong number of dimensions1`. The diagnostic container was stopped and removed. | Reject Python custom-op marker insertion for the release-like model path. | The aliasing custom op perturbs vLLM/AOT runtime shape assumptions even though the tiny control compiles and emits markers. This is not a comparable profiling path. | Next timing work should use a true lower-level marker/custom op with correct alias metadata, a profile-input/offline reproducer outside vLLM's full server path, or a reduced single-worker reproducer. Do not insert Python marker ops into the compiled Qwen forward again. |
| GGUF-083 | Built-in PyTorch profiler record-function ops as trace markers. | Tested `torch.ops.profiler._record_function_enter`, `_record_function_enter_new`, and `_record_function_exit` in tiny compiled controls. Consuming the handle through a Tensor dependency failed Inductor with a node-erasure error. Leaving `_record_function_enter` unused compiled, but Dynamo/Inductor removed the side effect and `rocprofv3 --marker-trace --kernel-trace` emitted no marker API records. | Reject built-in profiler record-function ops for this compiled GGUF path. | The built-in record-function hooks are either not compiler-safe when their handle is kept alive, or are dead-code-eliminated when unused. They do not provide a low-impact boundary marker for the vLLM fullgraph/AOT Qwen path. | Do not repeat `_record_function_enter/_exit` as a marker strategy. Use a lower-level alias-correct custom op, a profile-input/offline reproducer, or a reduced single-worker path with proven artifact emission. |
| GGUF-084 | F16 GGUF row-parallel residual pre-fold candidate. | Created a copied current-best dense GGUF bundle that allowed `VLLM_GFX906_MLP_DOWN_LLMM1_RESIDUAL_PREFOLD` when `quant_config` is GGUF and supplied the native row-parallel residual extension from the copied bundle. The server reached health on `.20` with the official release image, P2P-on, TP8, `MAX_MODEL_LEN=131072`, fused-GDN, native GGUF SwiGLU, and the normal benchmark ladder. Startup logs confirmed the residual pre-fold path enabled for the MLP down projection. Full ladder results were strict `63.631`, `c1_2000` `64.296`, and `c1_10000` `60.554` backend TPS. The strict gate was valid. | Reject as performance candidate; promote as negative source evidence. | Allowing the native row-parallel residual pre-fold path for F16 GGUF does not close the dense gap and is slightly slower than the current best GGUF baseline (`63.679` / `64.353` / `60.617`). The missing dense GGUF TPS is not the old residual pre-fold guard by itself. | Keep the fused-GDN plus native GGUF SwiGLU bundle as current best. Next source work should return to timing or reduced reproducer evidence around GatedDeltaNet/core, lm-head/logits, sampler, or GGUF adapter boundaries. |
| GGUF-085 | Qwen3.6 35B-A3B MoE F16 GGUF conversion and tokenizer audit. | Produced a F16 GGUF artifact from the cached Qwen3.6 35B-A3B source snapshot. Metadata reports `general.architecture=qwen35moe`, `general.file_type=1`, `qwen35moe.block_count=41`, `qwen35moe.expert_count=256`, and `qwen35moe.expert_used_count=8`. The embedded tokenizer metadata is present, and sampled HF-versus-GGUF token IDs matched for representative strings including `Hello`, `<think>`, `</think>`, `_manifest`, and `hh`. | Promote as artifact-readiness evidence only. | The GGUF package and tokenizer metadata look structurally sane, but that does not prove vLLM can map the architecture or serve coherent text. | Keep the artifact on NVMe and pin the matching HF tokenizer/config for vLLM tests. Do not run throughput tiers until startup and coherence pass. |
| GGUF-086 | Baseline MoE TP4 GGUF vLLM smoke on the local release-derived image. | Launching the MoE F16 GGUF with the native TP4 full-BAR/P2P-on intent failed before GPU allocation with `GGUF model with architecture qwen35moe is not supported yet`. | Reject as serving path; promote as source boundary evidence. | The failure happens before warmups or correctness probes, so it is not a TP4 throughput, P2P, warmup, or tokenizer result. The current local stack lacks a GGUF parser/model mapping for `qwen35moe`. | Add real `qwen35moe` / Qwen3.5-MoE GGUF support instead of aliasing it as an older MoE architecture. |
| GGUF-087 | Experimental `qwen35moe` to plain Qwen3 MoE alias with startup shims. | A copied experimental image aliased `qwen35moe` toward the older Qwen3 MoE route and added narrow startup shims for rotary fallback, F16 GGUF embedding materialization, and unquantized FusedMoE selection. The run progressed beyond architecture rejection and into weight loading, but failed with expert-shard shape/range errors while loading FusedMoE weights. | Reject as architecture fix; promote as negative source evidence. | Qwen3.6 35B-A3B MoE is not plain Qwen3 MoE. Its config path is Qwen3.5-MoE-style, with `moe_intermediate_size=512`, linear-attention layer types, and different expert tensor semantics. Aliasing to the older model class reaches the wrong loader and corrupts expert sharding. | Build proper Qwen3.5-MoE GGUF parser/loader support. Do not benchmark this alias path and do not treat it as a tokenizer problem. |
| GGUF-088 | ai-infos container source probe for Qwen3.5 MoE support. | The probed ai-infos image has native Qwen3.5/Qwen3.5-MoE HF model and config source files, including Qwen3.5 MoE registry entries. Its Transformers GGUF parser still rejects `qwen35moe`, and its vLLM GGUF mapping still lacks a Qwen3.5-MoE GGUF route. | Promote as upstream/source direction evidence; not a usable runtime fix. | That image appears ahead on HF Qwen3.5-MoE model support, but it does not make the GGUF artifact serve as-is. The useful takeaway is source-level architecture support, not a container swap or benchmark shortcut. | Port or reconstruct the required Qwen3.5-MoE GGUF mapping: parser config fields, tensor-name map, text-only model class selection, and expert sharding that preserves the `512` MoE intermediate dimension. |
| GGUF-089 | Parser-only Qwen3.5-MoE GGUF metadata and text-name-map probe. | A no-tensor-load parser shim for `qwen35moe` extracted the key config fields: hidden size `2048`, `40` text layers, `16` attention heads, `2` KV heads, `256` head dim, `512` MoE intermediate size, `512` shared-expert intermediate size, `256` experts, `8` experts per token, `MAX_MODEL_LEN` metadata `262144`, and `rope_theta=10000000.0`. A text-only dummy model with gguf-py's `qwen35moe` tensor-name map matched `663` of `693` text parameters. The unmapped set was the `30` linear-attention `dt_bias` tensors. | Promote as parser and mapping evidence only. | This proves the metadata and most GGUF/HF tensor names are close enough for a real route, but it also identifies the next missing bridge: map `linear_attn.dt_bias` to GGUF `ssm_dt.bias`, and keep the text-only Qwen3.5-MoE config instead of the multimodal wrapper. | Patch the experimental route to use the Qwen3.5-MoE text config, `qwen35moe` gguf-py architecture map, manual `ssm_dt.bias` mapping, and proper gate/up expert merge before any long vLLM launch. |
| GGUF-090 | Qwen3.5-MoE text GGUF route with manual tensor-name bridges. | Added an experimental `qwen35moe` route against the Qwen3.5-MoE text model class, including registry/model-loader mapping, a GGUF parser config, manual `ssm_dt.bias` to `linear_attn.dt_bias` mapping, full-stack expert gate/up/down loading, and a `conv1d.weight` shape guard. | Promote as source bridge evidence only. | The route moves beyond the unsupported-architecture and wrong-Qwen3-MoE alias failures, but it is not a serving or coherence proof by itself. | Launch the native TP4 full-BAR/P2P-on profile and classify the next runtime blocker before any warmups. |
| GGUF-091 | TP4 Qwen3.5-MoE GGUF launch with `--mamba-cache-mode align`. | The server reached model initialization but failed during cache setup with a page-size unification error: the layer page size was not divisible by the maximum page size. | Reject as serving path; promote as cache-geometry evidence. | `--mamba-cache-mode align` alone is insufficient for the text-only Qwen3.5-MoE route. | Wire the Qwen3.5 hybrid attention/Mamba config helper instead of relying on the launch flag alone. |
| GGUF-092 | Qwen3.5-MoE config-cache map without hybrid helper. | Mapping the text model class to the Qwen3.5 config helper still failed with the same page-size unification error. | Reject as complete cache fix. | The Qwen3.5 helper only updated SSM cache dtype; it did not invoke the hybrid attention/Mamba page-size adjustment path. | Call `HybridAttentionMambaModelConfig.verify_and_update_config(...)` for the Qwen3.5-MoE text route. |
| GGUF-093 | Hybrid cache helper plus text-model state-shape helpers. | The hybrid helper fired, logged attention block-size adjustment and Mamba page padding, loaded the model, and passed the previous page-size failure. Graph-capture startup then failed in the F16 GGUF MoE slow path with a HIP stream-capture error inside `fused_moe_gguf`. | Promote as startup-progress evidence; reject as graph-serving path. | The page-size blocker is fixed, but unquantized/F16 GGUF MoE currently routes through `GGUFMoEMethod` slow fallback, which is not graph-capture safe on this lane. | Use eager mode only for semantic diagnostics; for performance, implement a capture-safe fast/unquantized F16 GGUF MoE path rather than benchmarking the slow fallback. |
| GGUF-094 | Eager TP4 Qwen3.5-MoE GGUF request with original text config. | Eager startup reached API health, but the first direct completion request failed HTTP 500 with `M-RoPE support is not implemented`. | Reject as coherence path; promote as config-boundary evidence. | The text config still carried `rope_parameters.mrope_section`, causing vLLM to initialize M-RoPE positions even though the selected text-only model class does not implement `SupportsMRoPE`. | Strip only the M-RoPE section from the experimental text config while preserving ordinary RoPE theta and partial-rotary settings. |
| GGUF-095 | Eager TP4 Qwen3.5-MoE GGUF request with M-RoPE disabled. | Startup again reached API health, and the M-RoPE assertion disappeared. The first direct completion request then failed HTTP 500 because the text-only model class lacked `get_mamba_state_copy_func`. | Reject as coherence path; promote as Mamba-state interface evidence. | The request path now reaches Mamba state handling. The text-only base had dtype and shape helpers but not the matching state-copy helper present on the conditional-generation class. | Add the same gated-delta-net Mamba state-copy helper to the text-only base class. |
| GGUF-096 | Eager TP4 Qwen3.5-MoE GGUF with M-RoPE disabled and state-copy helper. | Startup reached health and the immediate request exceptions were gone. A 64-token direct completion returned no client bytes before a 300-second timeout; logs showed the request eventually leaving the running state just after the timeout. A follow-up 8-token streaming completion also returned HTTP 200 but emitted zero SSE bytes before a 240-second timeout. | Reject as deterministic probe candidate; promote as first-request stall evidence. | The route can now parse, load, initialize cache, and accept requests, but it does not deliver coherent text to the client and therefore cannot enter warmups or benchmark tiers. The slow F16 GGUF MoE fallback remains a likely source-path blocker, and request/state handling still needs work. | Do not run benchmark warmups on this path. Next source target is a capture-safe fast/unquantized F16 GGUF MoE path and a reduced first-token reproducer that can distinguish MoE slow-kernel deadlock from scheduler/state-copy behavior. |
| GGUF-097 | Graph-mode TP4 Qwen3.5-MoE GGUF with prefix-skip unquantized MoE selector. | Replaced the slow GGUF MoE fallback only when `is_layer_skipped_gguf(...)` matched the layer prefix. The native TP4 full-BAR/P2P-on graph launch still logged the slow GGUF MoE fallback warning and did not reach a useful health/probe point in the patient window. | Reject prefix-skip selector as a functional MoE fix; promote as method-selection evidence. | The GGUF MoE layer prefixes do not match the existing skipped-layer selector strongly enough to select `UnquantizedFusedMoEMethod`. The graph path remains tied to the slow fallback unless method selection is made explicit for the MoE expert layers. | Use an explicit diagnostic switch for F16/unquantized MoE method selection, then test graph startup and deterministic probes before any warmups. |
| GGUF-098 | Graph-mode TP4 Qwen3.5-MoE GGUF with forced unquantized FusedMoE method and M-RoPE stripped. | Added a diagnostic `VLLM_GFX906_GGUF_FORCE_UNQUANT_MOE=1` switch so GGUF F16 MoE expert layers use the normal unquantized FusedMoE method. The server reached API health in graph mode after model load and graph capture. A one-token raw `Hello` completion returned `amed` in about `0.38` seconds. A 16-token raw math completion completed after about `80` seconds but produced multilingual/glyph garbage rather than a coherent answer. A chat sanity request timed out with zero response bytes. | Reject as coherence and performance candidate; promote as graph-startup/source progress. | Forcing unquantized FusedMoE gets past the stream-capture failure and proves the slow fallback is not required for startup, but output semantics are still wrong and multi-token throughput is unusable. The remaining bug is not just capture safety; it is still in Qwen3.5-MoE GGUF config/state/tensor mapping or execution semantics. | Restore text M-RoPE fields with a text-only M-RoPE implementation and retest only short deterministic probes. Do not run warmups until probes are coherent. |
| GGUF-099 | Graph-mode TP4 Qwen3.5-MoE GGUF with forced unquantized FusedMoE and restored text M-RoPE. | Restored the original Qwen3.5-MoE text M-RoPE fields and added a text-only `supports_mrope` / `get_mrope_input_positions` implementation. The server reached health in graph mode: model load used about `17.22` GiB, torch compile finished, mixed prefill/decode and decode graph capture completed, and the API started normally. A one-token raw `Hello` completion still returned `amed`; a 16-token raw math completion timed out after `180` seconds with zero response bytes, while metrics showed only the one generated token from the first probe. | Reject as deterministic probe and benchmark candidate; promote as config-boundary evidence. | Restoring M-RoPE avoids the earlier assertion and preserves the text config, but it does not repair the garbage/timeout behavior. The tokenizer and basic serving shell are not the active blocker; the likely failure remains tensor mapping/layout, Mamba/GatedDeltaNet state handling, or expert routing under the Qwen3.5-MoE GGUF path. | Stop this container and do not run warmups. The next MoE GGUF work should be an offline or reduced first-token comparison against HF/llama.cpp boundaries: router logits, selected experts, expert gate/up/down tensor orientation, `linear_attn.dt_bias`, recurrent state copy, and final logits. |
| GGUF-100 | Offline MoE tensor comparison plus `dt_bias` permutation patch. | Compared layer-0 MoE GGUF tensors against the HF safetensors snapshot. Router weights, shared expert weights, expert gate/up/down slices for sampled experts, and `conv1d.weight` matched HF within FP32 comparison noise. `blk.0.ssm_dt.bias` did not match `linear_attn.dt_bias` directly; it exactly matched HF `dt_bias` in even-then-odd order, with inverse permutation restoring a zero-difference match. Added an experimental loader fix that restores HF order before loading `linear_attn.dt_bias`, then relaunched the same TP4 graph-mode forced-unquantized M-RoPE path. The server reached health and graph capture again, but the one-token raw `Hello` completion still returned `amed`, and the 16-token raw math prompt timed out after `180` seconds with zero response bytes. | Reject `dt_bias` permutation alone as a coherence fix; promote as confirmed tensor-mapping bug and narrowed source evidence. | The manual `ssm_dt.bias` bridge was wrong and should not be loaded directly, but fixing it alone does not repair the visible MoE GGUF output. The remaining corruption is probably downstream or parallel to `dt_bias`: selected-expert routing, expert tensor sharding/orientation after vLLM FusedMoE materialization, recurrent state copy/update, or final logits. | Keep the `dt_bias` inverse permutation in the experimental source tree for future tests. Next run should be an offline/reduced first-token comparison of router logits, top-k experts, expert outputs, recurrent state, and logits before another full vLLM server launch. |
| GGUF-101 | Direct-branch full inverse Qwen3.5 linear-attention GGUF layout patch. | Compared the llama.cpp Qwen3.5 conversion logic against the MoE F16 GGUF and HF safetensors. The inverse V-head reorder reconstructed HF layout for `in_proj_qkv` V rows, `in_proj_z`, `in_proj_a`, `in_proj_b`, `dt_bias`, `A_log`, `conv1d` V channels, and `out_proj` columns within FP32 comparison noise. Added those inverse transforms to the direct-parameter load branch and relaunched native TP4 graph mode with forced unquantized MoE. The server loaded, compiled, captured graphs, and reached health. A one-token raw `Hello` completion returned ` dota`; a 16-token raw math prompt timed out after `180` seconds with zero response bytes. | Reject as deterministic probe and benchmark candidate; promote as confirmed layout-bridge evidence plus loader-coverage warning. | The direct branch did not cover packed Qwen3.5 projections loaded through `stacked_params_mapping`, so this run did not actually invert `in_proj_qkv`, `in_proj_z`, `in_proj_b`, or `in_proj_a` before loading them into fused parameters. | Apply the same inverse transform before the stacked-parameter branch loads packed QKV/Z and B/A projections, then rerun only deterministic probes. |
| GGUF-102 | Stacked-branch full inverse Qwen3.5 linear-attention GGUF layout patch. | Moved the inverse layout transform ahead of the stacked-parameter load branch so `in_proj_qkv`, `in_proj_z`, `in_proj_b`, and `in_proj_a` are transformed before loading into fused `in_proj_qkvz` and `in_proj_ba`. The server again loaded, compiled, captured graphs, and reached health. The one-token raw `Hello` completion changed from the previous bad tokens to `_`, but it was still incoherent. The 16-token raw math prompt again timed out after `180` seconds with zero response bytes. | Reject as deterministic probe and benchmark candidate; promote as stronger source-boundary evidence. | The known llama.cpp Qwen3.5 linear-attention GGUF layout transform is now covered in both direct and packed load branches, but MoE GGUF output is still wrong. The active blocker is deeper than the confirmed linear-attention packing bridge. | Do not run warmups. Next work should build a reduced first-token comparison against a coherent control: router logits, selected experts, expert outputs after FusedMoE materialization, recurrent state update/copy, output norm/gate, and final logits. |
| GGUF-103 | Qwen3.5-MoE GGUF non-SSM norm-offset processor plus expert-shard verification. | Compared sampled embeddings, lm-head rows, final norm, block norms, attention q/k norms, and SSM norms against HF. Embeddings and lm-head matched directly. Most RMSNorm-family GGUF tensors matched HF only after subtracting `1.0`; `ssm_norm` tensors matched HF directly and must not be shifted. Added an experimental `qwen35moe` tensor processor that subtracts `1.0` from non-SSM norm weights while preserving the existing MoE expert merge logic. A separate offline TP4 shard simulation showed `w13` and `w2` expert shards for all four TP ranks match HF within FP32 comparison noise. The graph-mode TP4 forced-unquantized M-RoPE server reached health, but a one-token raw `Hello` completion still returned `_`, and the 16-token raw math prompt timed out after `180` seconds with zero response bytes. | Reject as deterministic probe and benchmark candidate; promote as confirmed parser/tensor-processing evidence. | The Qwen3.5-MoE GGUF route did have a real non-SSM norm offset bug, and expert TP shard orientation is now unlikely to be the primary corruption. Fixing the norm offset still does not recover coherent client-visible output. | Keep the norm-offset processor in the experimental source tree for future tests. Next work should compare first-token router outputs, recurrent state update/copy, output norm/gate application, and final logits against a coherent HF or llama.cpp control before another full server launch. |

| GGUF-104 | Benchmark-host identity and storage correction. | Confirmed that `.10`, `.20`, and `.30` are the same EPYC / 8x MI50 32GB class. The published benchmark host remains `.20` unless a run explicitly records another lane. Future host-to-host and benchmark-lane work should use the InfiniBand address family. The `.20` NVMe workspace is `<validation-workspace>`; `LLM_STORE_VOL` is a `.40`-only source-debug workspace and must not be created or used on `.10`-`.30`. The separate `.40` 8x MI60 lane is source-debugging only. | Promote as environment-control evidence; reject any benchmark conclusion that depends on treating `.40` as `.20` or mixing MI60 source-debug results with MI50 benchmark evidence. | Dense GGUF artifacts, run history, release-repo scripts, the MoE F16 GGUF artifact, and the Qwen3.5-MoE experimental patch bundle are now staged on `.20` NVMe. The MoE GGUF file hash matched the `.40` source copy: `1f2443bb0ff958943d091410c61120c181a0579b3bc85192029aa51d821d141c`. | Use `.20` for benchmark reproduction and claim comparisons. `.10` and `.30` can be used as same-class MI50 validation lanes when explicitly labeled. If MoE GGUF testing continues, use `<validation-workspace>` on the target MI50 host, not root-backed storage and not `LLM_STORE_VOL`. |
| GGUF-105 | Request-trace placement check for Qwen3.5-MoE GGUF final logits. | Added an experimental final-logit trace hook and launched the native TP4 graph-mode route on `.40` with the current Qwen3.5-MoE GGUF patch bundle. Startup loaded the model, compiled, captured graphs, and reached API health, but the saved trace fired during vLLM's internal compile/profile path before a client prompt. The traced hidden states had synthetic/profile-like statistics and identical top-token sets across TP workers; no valid client-visible `Hello` or math probe was captured before the container was stopped. | Reject this trace placement as a request-level coherence diagnostic; promote it as evidence that final-logit hooks must be gated to the actual request path. | The hook is too early in the serving lifecycle and records vLLM profiling tensors rather than the user prompt. It cannot explain why the client-visible `Hello` token remains `_` or why the 16-token math prompt times out. | Move the diagnostic to a request-gated boundary or build a reduced first-token comparison against HF/llama.cpp before another full server launch. Do not run warmups until deterministic probes are coherent. |
| GGUF-106 | Request-gated final-logit trace bundle staged on `.20`. | Copied the Qwen3.5-MoE GGUF patch bundle on `.20` NVMe and added a gate-file check around the final-logit trace hook. The hook now returns without writing until `VLLM_GFX906_GGUF_TRACE_GATE_FILE` exists, so startup/profile tensors should not be mistaken for client-request tensors. | Promote as diagnostic infrastructure only. No model request was served from this bundle yet. | This fixes the trace-placement problem from `GGUF-105` at the instrumentation level, but it is not a correctness or performance result. | Use the gated bundle only after the server reaches API health. Arm the gate immediately before a deterministic one-token request, then disarm or stop the container. |
| GGUF-107 | `.20` official-image MoE GGUF TP4 request-trace launch-shape checks. | Tried to run the staged Qwen3.5-MoE F16 GGUF artifact on `.20` with the official v0.2 image, TP4, P2P-on, and the request-gated trace bundle. The first launch used the wrong config directory and failed before model setup. The corrected text-config launch then hit a renderer/config wrapper mismatch: vLLM expected the outer Qwen3.5-MoE config class but received the text config directly. | Reject as launch-shape errors; promote as release-image integration evidence. | These failures happen before coherence or benchmark evaluation. They show the `.20` release-image route cannot simply reuse the source-debug launch arguments unchanged. | Keep using `.20` NVMe artifacts, but mount the active source files directly into the release image and preserve the wrapper/config shape expected by vLLM. |
| GGUF-108 | `.20` official-image MoE GGUF TP4 direct-mount and prefix-cache attempts. | Direct-mounted the Qwen3.5-MoE GGUF source files into the official release image so the staged artifact could load on `.20`. The server loaded weights and reached compile/cache initialization, but failed before API health with hybrid KV-cache layout ambiguity for tensor shape `torch.Size([2, 2, 528, 1, 256])`. Repeating with `--enable-prefix-caching --mamba-cache-mode align` made the config show `enable_prefix_caching=True` and `mamba_cache_mode='align'`, but the same layout assertion remained. | Reject as serving and benchmark candidate; promote as narrowed cache-integration blocker. | The release-image path is now past unsupported architecture, artifact absence, and basic weight-load issues, but it still cannot serve a client request on `.20`. No warmups, strict prompt, or fixed-token tiers were run. Prefix-cache alignment alone is not enough. | Next useful experiment is either a diagnostic eager-mode launch to get a request-gated one-token trace, or a source-level fix around the hybrid attention/Mamba KV layout discriminator for the observed five-dimensional cache tensor. Do not run warmups until deterministic probes are coherent. |
| GGUF-109 | `.20` official-image MoE GGUF TP4 eager request-gated trace. | Relaunched the direct-mounted release-image route with `--enforce-eager` while keeping TP4, P2P-on, prefix-cache alignment, the staged F16 MoE GGUF, and the request-gated final-logit hook. Eager mode reached API health. After arming the trace gate, a raw one-token `Hello` completion returned `_` with `finish_reason=length`. Four worker trace files were written from the client request. All workers reported the same hidden shape `(1, 2048)`, logits shape `(1, 248320)`, hidden norm about `142.27`, and the same bad top-token sequence headed by token id `62`. | Reject as coherence and benchmark candidate; promote as request-level trace evidence. | Eager mode bypasses the graph/cache startup assertion enough to reach a client request, but the output is still incoherent. The bad token is agreed across TP workers, so the visible corruption is upstream of sampling and not a single-rank sampler accident. No warmups or throughput tiers were run. | Keep the trace evidence and stop repeating full launches. The next source target is a reduced first-token comparison against HF or llama.cpp at router logits, recurrent state, output norm/gate, and final logits. |
| GGUF-110 | llama.cpp raw MoE GGUF control on `.20`. | Ran the same Qwen3.6 35B-A3B F16 GGUF artifact through the existing ROCm llama.cpp build inside the v0.2 release image with ROCm devices mounted. The first `llama-cli` attempt accidentally entered interactive conversation mode and was stopped. `llama-completion` with the default chat template produced coherent Qwen-style `<think>` output for `Hello`. The raw-completion control used `llama-completion -no-cnv`, TP4-class four-device layer split, `-n 1`, `--temp 0`, `--top-k 1`, and `--no-warmup`; it produced `Hello,`. | Promote as artifact/coherent-control evidence; reject as a benchmark throughput result. | The GGUF file and tokenizer are capable of coherent first-token behavior under an independent ROCm GGUF runner. The expected raw first generated token after `Hello` is comma, token id `11`. This isolates the vLLM failure to the vLLM GGUF/Qwen3.5-MoE execution path rather than the GGUF artifact itself. | Use comma token id `11` as the control token in vLLM request-level traces. Continue with reduced source comparisons; do not run warmups while vLLM still ranks the wrong token first. |
| GGUF-111 | `.20` vLLM MoE GGUF watch-id final-logit trace. | Copied the request-gated trace bundle and added `VLLM_GFX906_GGUF_TRACE_WATCH_IDS` to record targeted logits/ranks for the llama.cpp control token id `11`, the wrong vLLM token id `62`, and the other bad top IDs. Relaunched the official-image direct-mount route in eager mode, TP4, P2P-on, and sent the same raw one-token `Hello` request. | Reject as coherence and benchmark candidate; promote as exact request-level divergence evidence. | vLLM again returned `_`. On every TP worker, token id `62` ranked `1` with logit `12.867188`, while the llama.cpp control token id `11` ranked only `25` with logit `8.882812`. The same bad top-token set appeared on all four workers. This proves the vLLM path is not merely selecting a wrong token from nearly tied logits; the final hidden/logit boundary is materially wrong before sampling. | Next source target is upstream of `compute_logits`: compare final hidden state formation, output norm/gate, recurrent state copy/update, router/expert outputs, and layer outputs against a coherent control or reduced HF path. |
| GGUF-112 | `.20` vLLM MoE GGUF TP8 watch-id discriminator. | Reused the watch-id request trace bundle with the official v0.2 image, eager mode, P2P-on, all eight GPUs, and `TP_SIZE=8`. After the server reached API health, the same raw one-token `Hello` request was sent on a separate port. | Reject as coherence and benchmark candidate; promote as TP-degree discriminator evidence. | TP8 also returned `_`. Across all eight workers, token id `62` ranked `1` with logit `13.000000`, while expected comma token id `11` ranked only `20` with logit `9.000000`. This rules out a TP4-only sharding bug; the remaining corruption is general to the current vLLM Qwen3.5-MoE GGUF path. No warmups or throughput tiers were run. | Stop repeating TP-degree flips. Move upstream into reduced source comparison of final hidden formation, GatedDeltaNet/Mamba state, output norm/gate, and MoE/router/expert outputs. |
| GGUF-113 | `.20` HF-weight vLLM MoE TP4 watch-id control. | Built a clean HF-only trace overlay from the official v0.2 image's own Qwen3.5-MoE source, mounted the local HF snapshot, and launched TP4 eager mode with the same request-gated watch-id hook, P2P-on, prefix-cache alignment, and raw one-token `Hello` request. | Promote as shared-code control evidence; reject sampler, TP4 sharding, and raw-prompt objections as the primary GGUF explanation. | HF-weight vLLM returned the expected comma token. All four workers ranked comma token id `11` first with logit `15.039062`; underscore token id `62` ranked `369` with logit `5.906250`. Hidden norm was about `114.72`, not the GGUF trace's about `142.27`. The same vLLM Qwen3.5-MoE execution path is coherent with HF weights under the same reduced probe. | Narrow the active MoE GGUF bug to GGUF materialization/state formation before final logits. Compare live GGUF versus HF at layer outputs, output norm/gate, Mamba/GatedDeltaNet state copy/update, router/expert outputs, and selected tensor rows before running any warmups. |
| GGUF-114 | `.20` GGUF versus HF boundary-trace comparison. | Added the same request-gated boundary trace to copied GGUF and HF patch bundles, then reran the one-token raw `Hello` probe with TP4, P2P-on, eager mode, and the official v0.2 image. | Promote as the current best divergence boundary; reject another warmup/benchmark ladder. | GGUF still returned `_`; HF still returned comma. The first traced layer-0 GatedDeltaNet input already differs in scale: GGUF norm `91.739395`, HF norm `46.638695`. Downstream layer-0 projections are similarly scaled: GGUF `mixed_qkvz` norm `220.456894` versus HF `113.987366`, and GGUF `z_pre_reshape` norm `107.963158` versus HF `55.807575`. Final hidden stays divergent: GGUF norm `142.269989`, HF norm `114.722717`. | The next trace should capture decoder-layer input before `input_layernorm`, output after `input_layernorm`, and the loaded `input_layernorm.weight` parameter. The observed near-2x layer-0 scale points at a norm/runtime-parameter convention boundary before GatedDeltaNet, not at sampler or TP degree. |
| GGUF-115 | Qwen35-MoE non-SSM norm-minus-2 processor check. | Copied the GGUF boundary-trace bundle and changed only the `Qwen35MoeTensorProcessor` non-SSM norm rule from `weights - 1` to `weights - 2`, reflecting vLLM `GemmaRMSNorm`'s zero-centered `1 + weight` runtime convention. Relaunched the same TP4/P2P-on/eager one-token probe. | Reject as a fix; promote as source-routing evidence. | The live container showed the edited Qwen35 processor line, but the response and traces were byte-for-byte equivalent at the measured boundaries: output `_`, final hidden norm `142.269989`, token id `62` rank `1`, and comma id `11` rank `25`. The first accidental norm-minus-2 attempt had changed a different processor line and was also unchanged. | Do not repeat blind tensor-processor edits. Instrument the actual decoder-layer norm input/output and loaded norm parameter, then decide whether the wrong value is caused by a processor branch not firing for the active tensors, a later loader override/cache, or a different pre-GDN scale source. |
| GGUF-116 | Qwen35-MoE decoder-norm trace and runtime norm-offset fix. | Added request-gated decoder-layer traces for GGUF and HF controls, then tested a narrow GGUF-only runtime correction that subtracts `1.0` from loaded `GemmaRMSNorm` parameters before vLLM applies its `1 + weight` rule. The run used the same official v0.2 image, TP4, P2P-on, eager mode, and raw one-token `Hello` probe on `.20`. | Promote as MoE GGUF coherence repair candidate; reject throughput or release-claim promotion until the normal ladder runs. | The decoder trace showed GGUF and HF enter layer 0 with the same pre-layernorm embedding norm `0.610113`. GGUF loaded `input_layernorm_weight` with mean `1.031100` and norm `46.714878`, while HF loaded the offset parameter with mean `0.031111` and norm `2.625840`. Without the runtime correction, GGUF post-input-layernorm norm was `91.739395`; with the correction it became `46.637737`, matching the HF control `46.638695`. Final hidden moved from `142.269989` to `114.727249`, matching HF `114.722717`, and raw `Hello` changed from `_` to comma. A short no-trace sanity prompt returned coherent text: `Paris, a city...`. | Build a clean benchmark overlay from the runtime norm-offset fix, then run deterministic probes before the normal benchmark sequence: warmups -> `c1_128` uncapped strict -> `c1_2000` -> `c1_10000`. Do not promote MoE GGUF performance until those tiers are complete and comparable. |
| GGUF-117 | MoE GGUF TP4 norm-fix 131K benchmark-path test. | Built a full-length TP4/P2P-on launch from the norm-offset fix with `MAX_MODEL_LEN=131072`, release MoE TP4 NCCL settings, the GGUF loader/tokenizer/config route, and the existing benchmark harness. Tested graph mode first, then eager mode after graph startup failed. | Reject as benchmark candidate; promote as next source boundary. | Graph mode failed before health at the known hybrid KV-cache layout assertion for shape `torch.Size([2, 2, 528, 1, 256])`. Eager mode reached health and raw `Hello` returned comma, but the normal eight 2000-token warmups ran at only `12.455`-`14.060` backend TPS. All warmups produced the same self-referential reasoning-loop output, had no visible answer, and had no think close. The uncapped strict request was stopped as a runaway/no-close path after live metrics showed it had generated roughly 3K additional tokens beyond the warmup pattern. No `c1_2000` or `c1_10000` fixed-token tiers were run. | Do not spend another ladder on eager mode. The next source target is graph-mode hybrid cache layout for Qwen3.5-MoE GGUF, plus the benchmark-prompt reasoning-loop behavior under the norm fix. Throughput parity is impossible until graph mode starts and strict closes normally. |
| GGUF-118 | MoE GGUF TP4 norm-fix plus graph hybrid KV-layout bypass. | Built a copied experimental overlay from `qwen35moe-text-gguf-runtime-norm-offset-fix-20260626` and added an env-gated graph-mode cache layout bypass for the ambiguous `(2, 2, 528, 1, 256)` attention KV tensor. The launch used the official v0.2 image, TP4, P2P-on, `MAX_MODEL_LEN=131072`, release MoE TP4 NCCL settings, the same GGUF loader/tokenizer/config route, `VLLM_QWEN35_GGUF_RUNTIME_NORM_OFFSET_FIX=1`, and `VLLM_GFX906_ASSUME_HYBRID_ATTENTION_LEADING_KV=1`. | Reject as benchmark candidate; promote as graph-startup and strict-failure evidence. | The env-gated KV-layout bypass got graph mode past the previous hybrid cache assertion and the server reached health. A raw `Hello` probe returned comma. The normal eight 2000-token warmups completed much faster than eager, at `73.579`-`73.655` backend TPS after first-request overhead, but every warmup produced the same reasoning-only benchmark-prompt loop: `2000` generated tokens, `finish_reason=length`, `9449` reasoning chars, `0` visible/answer chars, no think close, `qwen_gate_valid=false`, and text hash `76ff50907f90dea8b16219bfb425858cc08904a8f955cfb430105128532f0d24`. The uncapped strict request did not stop and was cut off after the live metric estimate crossed `60071` generated tokens for that request. No `c1_2000` or `c1_10000` fixed-token tiers were run. | Preserve the graph KV-layout patch as a startup/performance diagnostic only. The next source target is benchmark-prompt/think-close behavior under the coherent norm-fix path, then MoE GGUF throughput. Do not run fixed-token tiers or claim MoE GGUF parity until the strict prompt closes normally. |
| GGUF-119 | MoE GGUF shortest-prompt sweep and HF TP4 eager comparator. | Relaunched the same graph-mode GGUF TP4 norm-fix plus KV-layout bypass on `.20` and ran the benchmark prompt with `prompt_repeat` values `0`, `1`, `4`, `8`, `16`, and `32` through the same chat/proxy path using `max_tokens=128` with EOS allowed. Then launched a clean HF-weight TP4 eager comparator with the same model snapshot, parser, P2P-on, TP4, and release MoE settings to check the shortest prompt without relying on graph mode. | Reject prompt-length and capped-output explanations; promote as exact semantic split evidence. | GGUF failed even at `prompt_repeat=0`: one-token top-logprobs ranked `Here` first with logprob about `-0.041`; every capped sweep case stopped at `finish_reason=length`, had no visible content, and stayed in reasoning-only text. `prompt_repeat=0` uncapped strict then timed out after `300` seconds with one active request and an estimated `18019` generated tokens for that request. The HF eager comparator also did not close inside a 128-token cap, so capped warmup invalidity alone is not diagnostic. But the same HF shortest prompt closed uncapped after `2957` generated tokens with `finish_reason=stop`, `qwen_gate_valid=true`, `10125` reasoning chars, and `3930` visible answer chars. | The live GGUF bug is not just long prompt length, begin-think proxy behavior, or the model choosing to start with meta-reasoning. HF also starts by analyzing the benchmark prompt and still transitions to an answer. The next source target is the GGUF failure to emit the Qwen reasoning/content transition or stop condition under benchmark-task prefill. Compare GGUF versus HF around the reasoning-to-answer delimiter logits/state, not another full ladder. |
| GGUF-120 | MoE GGUF Qwen3 thinking-disabled transition probe. | Relaunched the same graph-mode GGUF TP4 norm-fix plus KV-layout bypass on `.20` and used direct vLLM chat requests against port `8071` to compare the default Qwen3 reasoning parser path with `chat_template_kwargs.enable_thinking=false`. | Promote as transition-boundary evidence; reject as benchmark candidate. | Default thinking-enabled `max_tokens=128` reproduced the reasoning-only failure: `finish_reason=length`, `621` reasoning chars, and `0` content chars. With `enable_thinking=false`, the same prompt produced visible content immediately: `max_tokens=128` had `0` reasoning chars and `700` content chars, and `max_tokens=512` had `0` reasoning chars and `2158` content chars. The 512-token tail degraded into short repeated statements, so this does not prove long-form semantic quality. Tokenizer inspection pinned `<think>` to id `248068` and `</think>` to id `248069`. | The current graph path can generate visible content when the Qwen3 parser is put into content mode, so the benchmark failure is narrower than "GGUF cannot answer." It is the thinking-enabled path failing to emit the `</think>` reasoning end / content transition, with additional long-output quality concerns. Do not use thinking-disabled output for release benchmark claims. Next compare GGUF versus HF logits/state around token id `248069`. |
| GGUF-121 | MoE GGUF `</think>` watch-token trace versus HF-weight control. | Added request-gated compute-logits tracing for watch IDs `<think>` `248068`, `</think>` `248069`, comma `11`, and underscore `62`. The copied GGUF overlay disabled boundary tracing inside the compiled graph path so only compute-logits tracing remained. Ran the TP4 graph-mode GGUF norm-offset plus KV-layout bypass on `.20` with P2P-on and the shortest benchmark prompt at `max_tokens=4096`; compared against the prior HF-weight TP4 trace of the same prompt. | Promote as exact transition-failure evidence; reject as benchmark path. | GGUF served but hit `finish_reason=length` at `4096` completion tokens with `20218` reasoning chars and `0` content chars. The best sampled rank for `</think>` was only `198` at decode index `1664`; sampled ranks were usually much worse. The HF-weight control closed with `finish_reason=stop` after `2957` completion tokens and had sampled `</think>` rank `6` at decode index `224`. | The current GGUF blocker is not generic content generation, prompt length, P2P, TP4, or a parser-only setting. It is a model/logit-path failure to make `</think>` competitive under the thinking-enabled benchmark prompt. Next source work should compare late-layer state/logits around the delimiter transition rather than running warmups or fixed-token tiers. |
| GGUF-122 | MoE GGUF exact HF-prefix delimiter scoring and first-divergence check. | Reran the HF-weight TP4 comparator with trace stride `1` on the shortest benchmark prompt, extracted the original chat prompt token IDs plus the first `35` HF-generated token IDs, and used vLLM's exact token-list `prompt` form to avoid lossy special-token text round trips. Scored that same `82`-token forced prefix on HF and GGUF, then ran a GGUF self-generated `80`-token trace from the original chat prompt. | Promote as trajectory-boundary evidence; reject as benchmark path. | HF made `</think>` token id `248069` rank `3` at generated index `35`. Under the exact HF prefix, GGUF also ranked `</think>` at `3` and returned the same top visible token class as HF. The GGUF self trace diverged from HF at generated token index `1`: HF began `<|im_end|>Thinking Process:`, while GGUF began `<|im_end|>Here's a thinking process:`. The GGUF self request stayed reasoning-only through the `80`-token cap (`370` reasoning chars, `0` content chars). | This supersedes the simpler "GGUF cannot score `</think>`" theory. GGUF can make the delimiter competitive under a known-good HF trajectory; the active bug is early trajectory drift plus failure to recover into a closing reasoning/content transition. Next source work should compare early-step logits/state around generated index `1`, then follow the GGUF self trajectory to the point where it enters the loop. Do not run warmups or fixed-token tiers yet. |
| GGUF-123 | MoE GGUF faithful chat first-divergence rank probe. | Rejected the token-list completion endpoint as a faithful stand-in after it failed to reproduce the HF chat branch even with `add_special_tokens=false`. Relaunched HF and GGUF through the chat endpoint with trace stride `1`, max `2` tokens, and watch IDs for HF's `Thinking` token `90700`, GGUF's `Here` token `8160`, `Process` token `8340`, `'s` token `579`, `</think>` `248069`, comma `11`, and underscore `62`. | Promote as first actionable divergence boundary; reject benchmark path. | Both HF and GGUF first selected the same special token `248046`. On the next faithful chat-decode step, HF ranked `Thinking` (`90700`) first with logit `25.546875` and `Here` (`8160`) second with logit `25.203125`, a narrow `0.343750` HF preference. GGUF flipped the same competition much harder: `Here` ranked first with logit `26.234375`, while `Thinking` ranked second with logit `22.953125`, a `3.281250` GGUF preference. Hidden norms at that branch were close (`108.96` HF versus `108.02` GGUF), so the next target is not a gross norm-scale failure. | The active source target is the first post-special-token decode step. Compare late-layer hidden/logit contributors for tokens `90700` and `8160` at this branch: output norm/gate, lm-head rows, MoE/router expert outputs, and GatedDeltaNet/Mamba state. Do not use token-list forced-prefix completion as a benchmark-path proxy for this hybrid chat decode state. |
| GGUF-124 | MoE HF branch-row live instrumentation attempt. | Copied the HF and GGUF trace overlays and added request-gated `lm_head` shard/row logging inside `compute_logits()` to explain the `Thinking` (`90700`) versus `Here` (`8160`) rank flip from `GGUF-123`. Ran the HF TP4 chat endpoint first on `.20`, P2P-on, max `2` tokens. | Reject the live row-read hook; promote only the partial shard evidence. | The request failed with HTTP 500 after worker shutdown. One worker wrote a single prefill/profile compute event with `ParallelLMHead`, local weight shape `(62080, 2048)`, `org_start:124160`, `org_end:186240`, and all watch IDs marked remote. No successful branch-step row/dot evidence was captured. | Do not repeat live `lm_head` row access in the serving path unchanged. The safer next diagnostic is to capture hidden/logit trajectory with minimal hooks, then compare token rows and materialization offline or in a reduced single-process script. |
| GGUF-125 | MoE HF versus GGUF branch hidden-vector comparison. | Copied the safe compute-logits trace overlays and added an env-gated hidden-vector dump without touching `lm_head.weight`. Replayed the same faithful two-token chat request on HF TP4 eager and GGUF TP4 graph mode on `.20`, P2P-on. | Promote as branch-direction evidence; reject benchmark path. | The branch event after shared token `248046` reproduced the prior split. HF ranked `Thinking` first and `Here` second; GGUF ranked `Here` first and `Thinking` second. Hidden norms remained close: HF `108.961647`, GGUF `108.671486`. But the hidden-vector cosine was only `0.890908`, centered correlation `0.890758`, L2 diff `50.829083`, mean absolute diff `0.878714`, and max absolute component diff `5.335938`. | The branch flip is not explained by gross norm scale and should not be chased first as a final lm-head row lookup. The hidden direction has materially drifted before final logits. Next compare late-layer hidden formation: output norm/gate, layer-39 MoE/router/expert output, GatedDeltaNet/Mamba state carry, and residual stream around the first post-special-token decode step. |
| GGUF-126 | MoE GGUF eager-mode branch boundary attempt. | Relaunched the GGUF TP4 norm-offset plus KV-layout hidden-vector overlay in eager mode with `MAX_MODEL_LEN=4096` on `.20`, P2P-on, and the same faithful two-token chat request used for the branch comparison. | Promote as graph/eager branch agreement; reject as layer-localization evidence. | Eager reached API health after about `154` seconds and reproduced the same early branch flip. The first event selected special token `248046`; the next event ranked `Here` (`8160`) first with logit `26.468750` and `Thinking` (`90700`) second with logit `22.953125`; the following event ranked `'s` (`579`) first. The response ended at `finish_reason=length` with no visible or reasoning content. Boundary files still captured only `model.final_hidden`, not layer-0 or layer-39 internals. | Do not repeat the current eager boundary-hook shape unchanged. The failure is not graph-only because eager and graph agree on the branch flip, but localization still needs request-gated model-internal summaries or offline reduced diagnostics around late-layer residual, MoE/router/expert, and GatedDeltaNet/Mamba state. |
| GGUF-127 | MoE GGUF gated internal branch trace. | Rebuilt the GGUF hidden-vector overlay as a scratch patch so decoder and GatedDeltaNet internal traces only activate when the request trace gate is live, then reran the same `.20` TP4/P2P-on eager `MAX_MODEL_LEN=4096` two-token chat branch probe. | Promote as scale/gross-layer-output rejection; reject as benchmark path. | The run reproduced the branch split: after shared token `248046`, GGUF ranked `Here` (`8160`) first at `26.468750` and `Thinking` (`90700`) second at `22.953125`. The internal trace finally captured layer-0 and layer-39 summaries. HF and GGUF norms were effectively matched at selected boundaries: layer-0 `pre_input_layernorm` `2.867060` versus `2.867060`, layer-0 `post_input_layernorm` `278.260071` versus `278.259125`, layer-39 `post_input_layernorm` `269.081757` versus `269.077484`, layer-39 `post_mlp` `273.929382` versus `274.308777`, and layer-39 `post_attention` `48.878269` versus `49.109112`. | The remaining branch bug is hidden direction / trajectory drift, not RMSNorm scale, gross activation magnitude, graph-only behavior, or a late layer-39 blow-up. The next diagnostic should dump comparable last-token vectors at selected layer boundaries and compute layer-wise cosine against HF to find where the direction diverges. Do not run another full ladder until that boundary is sharper. |
| GGUF-128 | MoE HF/GGUF layer-wise vector cosine probe. | Added a scratch `row_values` dump to the gated boundary trace helper, then reran the faithful `.20` TP4/P2P-on eager two-token branch probe for HF and GGUF with full last-token vectors at layers 0, 10, 20, 30, and 39. | Promote early residual/MoE trajectory localization; reject final-logit-only and norm-only explanations. | HF produced the expected `Thinking Process` branch, while GGUF reproduced the null-content branch. Layer-wise cosine showed exact agreement through layer-0 input (`1.000000000`) and post-input norm (`0.999999943`), then drift after layer-0 attention/MLP: layer-0 `post_attention` `0.988626168`, `post_attention_layernorm` `0.982377650`, and low-norm `post_mlp` `0.849558298`. By layer 10, `pre_input_layernorm` was `0.897712368`; by layer 20 it was `0.905732166`; by layer 30 it dropped to `0.788824286`. Layer-39 boundaries partially realign but remain off-direction: `post_input_layernorm` `0.973970366`, `post_attention` `0.972754927`, `post_attention_layernorm` `0.938494202`, and `post_mlp` `0.984005033`. | The first actionable drift is early, before the final logits and long before layer 39. The next source target is the layer-0 to layer-10 residual trajectory, especially sparse MoE/router/expert output and GatedDeltaNet contribution ordering under GGUF. Instrument router logits, selected experts, routed/shared expert outputs, and residual updates before another benchmark ladder. |

## Current Classification

Classification: active source investigation. Dense FP16 GGUF is now coherent,
strict-valid, and benchmark-promoted for the dense reproduction target under
the CausalLM/Qwen3.5 path with GGUF tensor fixes, release overlay, graph
partitioning, embedding qweight-skip, and the `lm_head` unquantized logits-path
fix. Current best dense GGUF result is strict `70.505`, `c1_2000` `71.589`,
and `c1_10000` `66.967` backend TPS on `.20`, which beats the published dense
values (`69.514`, `70.347`, `66.069`). Dense still needs release-package
polish before it becomes a public GGUF reproduction path, but the benchmark
target itself has been met.

MoE GGUF has moved beyond artifact readiness, unsupported architecture,
semantic-correctness failure, and the earlier low decode band. The experimental
Qwen3.5-MoE text route can parse the F16 GGUF, load the model, initialize cache,
force a graph-capture-safe unquantized FusedMoE path, reach graph-mode health,
and pass the strict benchmark gate under the native TP4 full-BAR/P2P-on profile.
The initial MoE blockers were real: `qwen35moe` metadata support, text M-RoPE
handling, Qwen3.5 linear-attention GGUF layout inversion, non-SSM RMSNorm offset
handling, full-attention q/k norm handling, and hybrid KV-cache layout all had
to be addressed before the normal benchmark ladder became valid.

The MoE TP4 release-overlay combo bundle is the first performance promotion
candidate for GGUF MoE. Under the native `moe35b_tp4_fullbar_p2pon` profile,
P2P-on, FP16 GGUF, `MAX_MODEL_LEN=131072`, default graph capture, and normal
warmups -> strict -> `c1_2000` -> `c1_10000`, it produced `.20` strict
`118.754`, `c1_2000` `119.896`, and `c1_10000` `112.747` backend TPS. The
same path on `.30` produced strict `119.917`, `c1_2000` `120.781`, and
`c1_10000` `113.605` backend TPS. This matches or exceeds the native TP4 band
for the current goal, but still needs reproducibility engineering before public
GGUF claims change.

Historical Q4_0 GGUF TP4/TP8 was not benchmark-ready. Q/K repeat-interleave for
Qwen3.5 / Qwen3.6 GatedDeltaNet TP consistency cleared TP1, TP2, TP4, and TP8
simple first-token sanity, and the TP8 full-context Q4_0 launch completed stable
2000-token warmups. It did not clear the release-style uncapped strict prompt:
the `c1_128` strict gate stopped after 12 tokens without a visible answer or
think close. Fixed-token `c1_2000` and `c1_10000` tiers were intentionally not
promoted after that correctness failure.

The later true FP16/half GGUF candidate superseded that garbage-output state.
The proven correction set is:

- explicit HF tokenizer/text config instead of relying only on embedded GGUF
  metadata;
- F16 materialization for GGUF embedding/lm-head where needed;
- norm conversion with `norm.weight - 1` except for
  `linear_attn.norm.weight`;
- linear-attention QKV/Z, beta/alpha, `dt_bias`, `conv1d.weight`, and `A_log`
  layout/state conversions for Qwen3.5 / Qwen3.6;
- release overlay applied as one coherent bundle.

That path now produces coherent output and passes the strict gate. The later
`GGUF-203` `lm_head` unquantized logits-path result supersedes the older
`63.679` / `64.353` / `60.617` dense baseline and reaches strict `70.350`,
`c1_2000` `71.396`, and `c1_10000` `66.842` backend TPS. This beats the
published dense values and confirms the dense target is achievable in GGUF on
the same `.20` lane. A later `.20` lane-validation run from the same promoted
path superseded those raw values with strict `70.505`, c1_2000 `71.589`, and
c1_10000 `66.967` backend TPS.

The fused-GDN GGUF parity experiment rejected another tempting shortcut. It
made the GGUF projection shape match the HF release fused qkv/z/b/a path and
loaded successfully, but the first short chat request hung before producing a
response. That means the performance gap cannot be solved by mechanically
fusing beta/alpha into the release projection shape without also handling the
GGUF GatedDeltaNet execution/state semantics.

The smaller TP4 FP16/half smoke and later CausalLM TP8 runs narrowed the failure
from correctness to performance. Earlier raw one-token and chat probes were
malformed before the Qwen3.5 tensor fixes. After those fixes, the dense lane is
semantically usable but still decode-limited relative to the published HF-weight
FP16 path.

The latest minimal-bundle tests split startup fixes from semantic fixes. F16
GGUF embedding/lm-head materialization is necessary for serving, and cache
helpers can be necessary when older GGUF loader/registry code is used. Neither
condition restores coherent output. Invalid byte/glyph text from deterministic
raw and chat probes means this line fails before warmups and should not be used
for benchmark throughput.

The independent ROCm llama.cpp smoke changes the artifact diagnosis: the F16
GGUF file itself is capable of coherent output on `.20`. The GGUF file also
packages tokenizer metadata, and a local token-ID audit matched the embedded
GGUF tokenizer ordering against the HF tokenizer over sampled shared IDs. For
vLLM reproducibility we still pin the HF tokenizer/config explicitly, because
chat-template and model-config behavior can differ even when the embedded GGUF
vocab is present. The active failure is now specifically a vLLM GGUF/Qwen3.5
source-path performance gap. Future work should stop treating file conversion
or tokenizer packaging as the leading suspect unless new evidence contradicts
the llama.cpp, tokenizer-audit, and vLLM coherence results.

The latest source-routing checks make the next step more precise. Direct
QwenNext can now be made to start, but its output is invalid bytes and should
not be promoted for this `qwen3_5` model. A no-bundle Qwen3.5 launch fails
before serving on an uninitialized GGUF embedding parameter, proving that the
old diagnostic bundle contains necessary startup materialization logic.
However, that same bundle also changes Qwen3.5 behavior. The next experiment
should therefore create a smaller coherent source bundle that separates
required GGUF startup fixes from the actual Qwen3.5 correctness hypothesis.

The F16 materialization test corrected an obvious mismatch in the diagnostic
bundle: materialized GGUF embedding/lm-head qweights should be labeled F16 for
the active F16 file, not BF16. That change is worth carrying forward in a
minimal bundle, but it did not repair semantics. The next blocker remains the
Qwen3.5 GGUF execution path for real prefill/chat contexts in the older bundle;
the later tensor-fix/release-overlay bundle repaired semantics but not
throughput.

Patch-bundle routing is now part of the active source checklist. The image
entrypoint applies the mounted patch bundle after Docker file mounts, so a
bundle can overwrite an individually mounted Qwen source file. A direct
QwenNext launch with a clean bundle failed before serving because it needs a
text-config adapter, not only an architecture override.

The official GGUF specification confirms the relevant tensor type IDs used by
the scanner and runtime traces: `Q4_0` is type `2` and `Q8_0` is type `8` for the active audit. It also confirms the tensor-info and aligned tensor-data
structure used by the low-level scanner:
<https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>.

The llama.cpp GGUF reader implementation is the current C++ behavior reference
for parser invariants and data attachment:
<https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/gguf.cpp>.

The vLLM public GGUF documentation is an external support boundary, not a
promotion claim. It documents GGUF as experimental, recommends the base-model
tokenizer, and shows `--hf-config-path` when metadata conversion is not enough:
<https://docs.vllm.ai/en/stable/features/quantization/gguf/>.

The public Qwen3.5 GGUF upstream issue and PR currently address loader/config
startup failures, especially `qwen3_5` to `qwen35` naming and vision-config
`depth` fallback:
<https://github.com/vllm-project/vllm/issues/38122> and
<https://github.com/vllm-project/vllm/pull/38140>. Those findings explain why
the launch path must be explicit, but they do not resolve the local TP4/TP8
first-token correctness drift.

The upstream llama.cpp Qwen3.5/Qwen3.6 tensor-parallel fix is a stronger source
clue for the current failure:
<https://github.com/ggml-org/llama.cpp/pull/23843>. The PR states that wrong
tensors were used to determine split granularity for Qwen3.5/Qwen3.6 with
heterogeneous quant mixes. The local GGUF file has exactly that kind of mixed
quantization around layer-0 GatedDeltaNet: `attn_qkv` and `attn_gate` are Q4_0,
while `ssm_out` is Q8_0.

The first C split audit rejected a basic divisibility failure: the relevant
Q4_0, Q8_0, and F32 layer-0 tensors split cleanly for TP4 and TP8. That does
not remove the llama.cpp clue; it narrows it to Qwen-specific segment and
companion-tensor split semantics rather than a trivial block-size remainder.

The all-rank TP4 trace reproduced the same wrong first-token result and showed
rank agreement on the final wrong logits. The failure should therefore be
treated as rank-local GatedDeltaNet input/output or reassembly drift before
the final logits, not as a final gather-only issue.

The segment-labeled TP4 trace did not reveal a dead segment. q, k, v, z, beta,
and alpha are all populated before the GatedDeltaNet core, yet the same wrong
top token remains. The matching TP1 segment-labeled control returned the
coherent comma token under the same hook, so the instrumentation itself is not
the cause. That keeps the active source target on how populated segments are
ordered, split, consumed by the GatedDeltaNet core, and reassembled under TP4.

The compact segment-checksum diagnostic narrowed the failure further. For the
final one-token decode step, TP4 rank-local q/k/v/z/b/a inputs sum back to the
coherent TP1 inputs, but the GatedDeltaNet core and output projection checksums
do not match TP1. The current target is therefore no longer a simple GGUF tensor
range or missing-segment problem; it is the TP GatedDeltaNet core/state or
reassembly behavior after those inputs are formed.

The TP1 chunk comparison first pointed at Q/K after `causal_conv1d`, but the
follow-up pre-conv and conv-weight checks rejected a simple conv-weight loader
fault: TP1 chunk-equivalent q/k/v inputs and q/k/v conv weights match TP4
rank-local data exactly. The mismatch comes from Q/K expansion order. Tiled
repeat makes TP1 global chunks and TP4 local ranks disagree; repeat-interleave
made TP1 and TP4 agree on the deterministic first-token probe.

The latest dense TP8 benchmark attempts reject three performance hypotheses:

- official release image plus combined GGUF/release overlay performs the same
  as the experimental image, so the gap is not image-base drift;
- restoring the release attention subclass does not move decode TPS, so the
  missing performance is not simply attention-class routing;
- the F16 merged-single-matmul GGUF patch did not apply to the important dense
  linears because those tensors already load as normal `weight` params, so the
  gap is not the assumed GGUF `qweight` shard-matmul overhead;
- removing disabled trace hooks from the forward path does not improve decode
  TPS, so trace-call overhead is not the missing performance lever.
- enabling the GFX906 native interleaved SwiGLU path for F16 GGUF works and
  stays strict-valid, but the full ladder still lands at `61.053` strict,
  `61.780` on `c1_2000`, and `58.266` on `c1_10000`, so MLP SwiGLU alone is
  not the missing performance lever.
- the `GGUFLinearMethod.apply()` diagnostic sampled graph profiling and a short
  decode request and only saw `ParallelLMHead` hits. That rejects broad internal
  block-linear GGUF matmul overhead as the primary dense gap.
- forcing only `lm_head` to use the unquantized embedding method removed the
  visible GGUF lm-head path but did not improve decode: strict `60.676`,
  `c1_2000` `61.309`, and `c1_10000` `57.888` backend TPS.
- rerunning the fused-GDN projection path with a patient smoke and full ladder
  showed it is serving-capable and faster than split-GDN: strict `63.201`,
  `c1_2000` `63.836`, and `c1_10000` `60.169` backend TPS. It is still below
  the FP16/HF release band.
- combining fused-GDN with native GGUF SwiGLU improved the dense ladder again:
  strict `63.679`, `c1_2000` `64.353`, and `c1_10000` `60.617` backend TPS.
  It is now the best dense GGUF baseline, but it still misses the FP16/HF
  release band.
- comparing HF-control and best-GGUF logs shows equivalent graph-capture mode,
  capture sizes, and capture duration, so graph setup alone is not the current
  explanation. The logs still differ in wrapper type, page geometry, and
  fallback source, but the ConditionalGeneration rerun already rejected wrapper
  matching as a performance route.
- Python timing hooks inside the compiled Qwen3.5 GatedDeltaNet forward are not
  viable: Dynamo rejects both `torch.cuda.synchronize()` and
  `time.perf_counter()` during startup profiling. The next timing evidence needs
  an external ROCm profiler or a compiler-safe lower-level marker.
- the GGUF residual pre-fold revisit is closed for now. The active combined
  bundle still guards `VLLM_GFX906_MLP_DOWN_LLMM1_RESIDUAL_PREFOLD` behind
  `quant_config is None`, while the historical FP16 dense source record already
  rejects MLP-down residual pre-folding as a serving promotion after exact
  microbench and full-ladder testing. Do not spend another full GGUF ladder on
  this path unless a request-window profiler points directly at MLP-down
  residual/add work as the remaining gap.
- optional FLA / `causal_conv1d` package absence is not by itself the dense
  GGUF explanation. A lightweight release-image check found `fla`,
  `causal_conv1d`, and `flash_linear_attention` all absent, while the same
  image and clean HF-weight control still reproduced the release band. Stopped
  GGUF containers emitted the Transformers "fast path is not available" warning
  eight times per run and many short-sequence format warnings, while the HF
  control emitted none of those warning strings and logged 512 native SwiGLU
  enablements. Treat these logs as a profiling clue, not a reason to mutate the
  release image or install packages blindly.

ConditionalGeneration architecture alignment also failed to improve decode and
worsened prefill. The clean HF-weight control resolves as
`Qwen3_5ForConditionalGeneration`, while the coherent GGUF path resolves as
`Qwen3_5ForCausalLM`; however, both the earlier ConditionalGeneration override
and the modern fused-GDN plus native GGUF SwiGLU ConditionalGeneration rerun
have been rejected. The next source work should therefore profile the active
GGUF CausalLM baseline against the published HF-weight FP16 release path instead
of flipping the architecture override again, especially GatedDeltaNet /
linear-attention execution, request-window timing, graph structure, and any
GGUF-specific adapter overhead after tensor materialization. The FLA /
causal-conv warning is now treated as loader-boundary evidence unless profiling
proves otherwise.

The MoE GGUF source lane is now distinct from the dense performance lane. The
Qwen3.6 35B-A3B F16 GGUF artifact is present and its tokenizer metadata passes
sampled HF/GGUF ID checks. The first vLLM smoke failed because `qwen35moe` was
unsupported, and the alias-to-Qwen3-MoE experiment failed later on expert-shard
shape semantics. The current experimental Qwen3.5-MoE text route fixes those
early boundaries but still cannot produce client-visible text. It reaches
health only in eager mode after disabling accidental M-RoPE and adding the
missing text-class Mamba state-copy helper; then even an 8-token streaming probe
returns no bytes before timeout. That means the next useful MoE work is not
warmup repetition or fixed-token throughput. It is a source-level first-token
reproducer and a capture-safe fast/unquantized F16 GGUF MoE implementation.

The first parser-only Qwen3.5-MoE GGUF probe is encouraging but not sufficient.
It parses the expected metadata without loading tensors and maps most text-model
parameters through gguf-py's `qwen35moe` map. The remaining explicit mapping
gap is `linear_attn.dt_bias` versus GGUF `ssm_dt.bias`; the remaining serving
gap is correct expert gate/up merge and FusedMoE sharding under the Qwen3.5-MoE
model class.

The 2026-06-26 TP4 graph-mode `IsHybrid` marker run changed startup behavior
but did not fix output coherence. The experimental text-only
`Qwen3_5ForCausalLMBase` previously inherited `HasInnerState` but not
`IsHybrid`, while the stock conditional-generation wrapper advertises hybrid
behavior. Adding `IsHybrid` caused vLLM to run hybrid attention/Mamba cache
verification, set the attention block size to `528` tokens, and align the
mamba page size before graph capture. The server reached `/health` with the
same TP4 GGUF image, restored M-RoPE fields, non-SSM norm offset fix, full
linear-attention inverse mapping, and forced unquantized MoE path. The
deterministic probes still failed: raw `Hello` with `max_tokens=1` returned
`_` in `0.36` seconds, and the math prompt timed out after `180.01` seconds
with zero response bytes. Outcome: preserve the hybrid marker as a likely
correct text-model integration fix, but reject it as a semantic/coherence
promotion. Do not run warmups or benchmark tiers from this state.

## Promotion Criteria

The next source patch should be considered only a diagnostic success until it
passes:

1. TP1 first-token sanity.
2. TP2 first-token sanity.
3. TP4 first-token sanity.
4. TP8 first-token sanity.
5. Normal benchmark warmups.
6. `c1_128` uncapped strict.
7. `c1_2000`.
8. `c1_10000`.

Dense FP16 GGUF now clears the first six steps in the coherent CausalLM path and
also completes `c1_2000` and `c1_10000`, but it fails the final promotion test
because backend TPS is below the FP16 v0.2.1 reproduction numbers. MoE GGUF has
cleared several architecture-mapping and startup failures, can serve reduced
coherent probes, and can score the `</think>` delimiter correctly under an exact
HF-generated prefix. It still fails the thinking-enabled benchmark trajectory:
self-generation diverges immediately and does not close into visible content.
It must repair that trajectory before it can enter the warmup and fixed-token
promotion ladder.

## GGUF-147 - GPUModelRunner Whole-Forward Timing

### Hypothesis

After rejecting steady GatedDeltaNet core timing, steady FusedMoE runner timing,
and post-forward logits/sampler timing as the dominant MoE GGUF throughput gap,
the next boundary was the vLLM v1 GPU model runner itself. The working
hypothesis was that GGUF might be losing time in runner preprocessing,
postprocessing, graph replay setup, or a broader model-forward envelope that was
not visible from the narrower Python hooks.

### Method

An env-gated diagnostic copy of `vllm/v1/worker/gpu_model_runner.py` was mounted
only for the timing experiment. It recorded request-window timing for:

- `preprocess`
- `model_forward`
- `postprocess`
- `execute_model_total`

Two variants were tested on `.20`:

- synchronized timing, which inserted device synchronization around the measured
  regions;
- no-sync timing, which avoided synchronization and preserved the serving
  trajectory.

The GGUF TP4 graph-mode path was compared against HF-weight TP4 controls using
the same short faithful branch request. An aligned HF comparator also restored
the broad graph-capture shape so that capture sizing was not the explanation.

### Results

The synchronized GGUF timing run reached `/health`, but the request output
changed from the expected `Thinking Process` branch to `Here's`. That diagnostic
shape is rejected because the instrumentation perturbed correctness or graph
behavior.

The no-sync GGUF timing run preserved correctness and returned
`Thinking Process`. Request-window rows showed that runner preprocessing and
postprocessing were small compared with the model-forward envelope:

- tokens `1`, FULL graph: `model_forward` about `77,795` ns;
- tokens `15`, PIECEWISE graph: `model_forward` about `3,455,214,660` ns;
- tokens `32`, PIECEWISE graph: `model_forward` about `10,317,543,110` ns.

The aligned HF no-sync comparator also returned `Thinking Process`. With broad
graph capture aligned, HF was not faster in this short request window:

- `model_forward` HF/GGUF ratio, tokens `1`: `0.983`;
- `model_forward` HF/GGUF ratio, tokens `15`: `1.107`;
- `model_forward` HF/GGUF ratio, tokens `32`: `1.132`;
- `preprocess` and `postprocess` remained small on both paths.

### Outcome

Reject `GPUModelRunner` Python-side preprocessing, postprocessing, and short
request no-sync model-forward envelope timing as the missing MoE GGUF
performance explanation.

The no-sync measurements are not definitive GPU durations because they do not
synchronize graph replay. They are still useful as a boundary: the visible
Python runner path is not where the FP16/HF versus GGUF fixed-token throughput
gap obviously appears, and an aligned HF short request was slower than GGUF
under the same runner timing hook.

### Promote / Reject

Reject as a promotion path.

Do not repeat the synchronized runner timer because it changed branch output.
Do not repeat the same no-sync runner timer unless a new source change needs a
quick runner-envelope smoke. The next timing pass should move to
graph-visible whole-decode profiling, ROCm/HIP kernel traces, or longer
fixed-token request-window profiling that can compare actual graph replay and
kernel mix per decode token.

## GGUF-148 - `rocprofv3` Request-Window Profiling Route

### Hypothesis

The Python-side timing hooks had narrowed several boundaries but still did not
show actual GPU graph replay duration. The next hypothesis was that a
request-window ROCm kernel trace could reveal whether the MoE GGUF path is
using a different kernel mix or graph replay pattern from the HF-weight TP4
reference path.

### Method

Two `rocprofv3` routes were tested on `.20` against the current MoE GGUF TP4
default-scheduler path:

1. Attach `rocprofv3` to each live TP worker PID during a normal c1_2000
   request.
2. Launch the server under `rocprofv3` with delayed collection so the trace
   window starts after model load and graph capture.

The PID-attach route reported successful injection into all four TP workers
during a c1_2000 request. The request itself completed in the normal GGUF band:
`2000` completion tokens, `72.740` backend decode TPS.

The delayed wrapper route initially failed before server startup because the
release image does not include `libdw.so.1`, a dependency of
`librocprofiler-sdk.so`. A read-only host `libdw.so.1` mount fixed the startup
dependency for diagnostic use only. With that mount:

- delayed c1_2000 tracing completed the request but heavily perturbed decode:
  backend decode TPS fell to `32.892`;
- no trace or summary files were emitted after the default container stop;
- a shorter delayed c1_512 retry kept decode in the normal GGUF band at
  `73.019` backend decode TPS, but `rocprofv3` remained stuck waiting for child
  processes after SIGINT and still emitted no trace or summary files.

### Outcome

Reject this `rocprofv3` attach/wrapper route for the current vLLM multiprocess
server.

The failure is methodological, not a model-performance finding. PID attach can
report success without producing usable files, while wrapping the parent server
does not flush cleanly for the child-worker process tree. Kernel tracing also
can materially perturb decode when the collection window covers a long request.

### Promote / Reject

Reject as a repeatable profiling route.

Do not repeat parent-process `rocprofv3` wrapping or worker PID attach unchanged.
The next graph-visible evidence should come from a worker-entrypoint-aware
profiler strategy, explicit in-process profiler pause/resume hooks, a
lower-overhead kernel counter route, or a targeted source-level marker that
survives graph replay without changing output correctness.

## GGUF-149 - Full MoE Block Timing Boundary

### Hypothesis

After rejecting GatedDeltaNet core timing, direct FusedMoE timing,
post-forward logits/sampler timing, and the visible `GPUModelRunner` envelope,
the next hypothesis was that the larger Qwen3.5-MoE block wrapper might still
hide the GGUF gap. That wrapper includes router handling, the routed/shared
expert call, shared-expert addition, and tensor-parallel all-reduce.

### Method

An env-gated diagnostic copy of `qwen3_next.py` was mounted on `.20` and added
timing around `Qwen3NextSparseMoeBlock.forward`. The first attempt used the
normal TP4/P2P-on graph-mode GGUF path. That failed during engine initialization
because `torch.compile` cannot trace `time.perf_counter_ns()` inside the
compiled MoE block forward.

The diagnostic was then rerun as an eager-only 4K source comparator, not as a
benchmark claim:

- GGUF TP4 used the current best MoE GGUF overlay, P2P-on, forced
  unquantized MoE, and the same faithful two-token branch probe.
- HF TP4 used the same host lane, same timing overlay, same request shape, and
  the HF-weight v0.2 release model path.
- Both probes returned the expected `Thinking Process` branch.

### Results

The request-window aggregate comparison used `160` rows per source/stage/shape
after excluding startup rows.

Stable `(32, 2048)` MoE-block totals were effectively identical:

- GGUF total: `2,486,178` ns average, min `2,256,212`, max `3,135,504`.
- HF total: `2,479,019` ns average, min `2,251,235`, max `3,046,707`.

The dominant `(32, 2048)` internal-router expert stage also matched:

- GGUF `experts_internal_router`: `2,241,665` ns average.
- HF `experts_internal_router`: `2,239,985` ns average.

Shared add and TP all-reduce were small at the stable decode shape:

- GGUF `shared_add`: `46,113` ns average.
- HF `shared_add`: `48,811` ns average.
- GGUF `tp_all_reduce`: `169,843` ns average.
- HF `tp_all_reduce`: `159,616` ns average.

The `(15, 2048)` HF totals contained large all-reduce outliers and are treated
as eager diagnostic noise, not promotion evidence.

### Outcome

Reject the full `Qwen3NextSparseMoeBlock.forward` wrapper as the current
primary MoE GGUF throughput bottleneck.

Graph-mode Python timing at this level is not viable because it breaks Dynamo
compilation. Eager-mode timing is useful as a source boundary, and it shows that
the full MoE wrapper is already at parity between GGUF and HF for the stable
decode shape on the faithful two-token probe.

### Promote / Reject

Reject as a performance fix and as a benchmark path.

Do not repeat unchanged full-MoE-block Python timing. The next useful source
work should move below Python wrapper timing toward graph-visible replay,
kernel-level evidence, CausalLM-versus-HF graph structure, or a longer
fixed-token decode-window profiler that can attribute actual per-token replay
cost without perturbing output correctness.

## GGUF-150 - MoE Graph Cache Runtime Norm-Offset Comparison

### Hypothesis

After the MoE GGUF path became strict-valid but slow, the next structural
hypothesis was that the GGUF-only runtime RMSNorm offset correction might be
inflating the compiled graph. The runtime fix subtracts `1.0` from non-SSM
norm weights inside the active model path so vLLM's `GemmaRMSNorm` runtime
receives the offset-style parameter shape it expects.

### Method

Compared cached compiled graph artifacts from the current MoE GGUF graph path
against the HF-weight comparator:

- GGUF runtime-offset graph cache hash: `007b10778a`.
- HF-weight graph cache hash: `a1b5bf3ed3`.

The comparison looked only at graph structure, local source references, and
cache size. No benchmark run was promoted from this inspection.

### Results

The GGUF graph carried `_qwen35_effective_weight` calls that materialize
`self.weight.data - 1.0` inside the compiled graph for many non-SSM
Qwen3.5-MoE RMSNorm sites. The HF graph uses the normal layernorm weight
directly.

Observed graph/cache sizes:

- GGUF runtime-offset graph: `9336` lines, about `32M` cache size.
- HF-weight graph: `9145` lines, about `23M` cache size.

### Outcome

Promote as source evidence that the current correctness-preserving runtime
norm-offset path is a real graph-structure difference from the HF-weight path.

### Promote / Reject

Promote as a performance hypothesis and graph parity target.

Reject as a standalone fix. A graph difference does not prove that the runtime
offset accounts for the full GGUF/HF TPS gap. The next step was to test whether
the same offset could be moved to load time without breaking semantics.

## GGUF-151 - MoE Runtime Norm-Offset Disabled Smoke

### Hypothesis

If the runtime subtraction is only a performance artifact and the loaded GGUF
norm tensors are already in the right convention, disabling
`VLLM_QWEN35_GGUF_RUNTIME_NORM_OFFSET_FIX` should preserve coherence while
shrinking the graph.

### Method

Relaunched the active MoE GGUF TP4 graph path on `.20` with the runtime
norm-offset switch disabled and sent a short direct smoke prompt. No benchmark
ladder was run because the smoke result determines whether the path is even a
candidate.

### Results

The server reached a request path, but the smoke output degraded into
multilingual/glyph garbage and the request ended at the token cap rather than
a normal coherent stop.

### Outcome

Reject disabling the runtime norm-offset fix.

### Promote / Reject

Reject as a correctness path. The runtime offset is still required for the
active MoE GGUF source route. The next test must preserve semantics while
attempting to move the offset out of the compiled forward graph.

## GGUF-152 - MoE Load-Time Norm Offset Benchmark Attempt

### Hypothesis

Apply the non-SSM norm `-1.0` correction once during GGUF weight loading, then
disable the runtime offset path. If equivalent, this should remove the graph
subtractions while preserving Qwen strict output behavior.

### Method

Created a disposable load-time norm-offset overlay. The loader subtracted
`1.0` for floating non-SSM `norm.weight` tensors and preserved `ssm_norm`
directly. Launched the native MoE TP4 full-BAR/P2P-on GGUF path with
`MAX_MODEL_LEN=131072`, runtime offset off, and load-time offset on, then ran
the standard harness until the first correctness failure.

### Results

The server reached health. A short smoke prompt stopped immediately with an
empty one-token response. The normal benchmark warmups completed in the same
rough fixed-token throughput band as the current GGUF path:

- warmup 1 backend TPS: `71.421`
- later warmups: about `73.49` to `73.53`

However, warmups had no visible answer field, no parser-complete visible
answer, and reused the same visible-output hash. The `c1_128` uncapped strict
gate failed immediately:

- completion tokens: `1`
- finish reason: `stop`
- `qwen_gate_valid=false`
- invalid reason: missing thinking close or parser answer

No `c1_2000` or `c1_10000` tier was promoted from this variant.

### Outcome

Reject the load-time-only norm-offset variant.

### Promote / Reject

Reject as a correctness path and benchmark candidate. Moving the offset solely
into the current load hook is not semantically equivalent to the runtime
correction. The failure shape suggests either the load hook double-shifts some
weights, misses others, or interacts with vLLM's active parameter materializers
after initial weight iteration.

## GGUF-153 - MoE Load-Time Norm Offset Plus Q/K Norm Class Smoke

### Hypothesis

The load-time-only path may have failed because full-attention q/k norms still
used the wrong runtime class. Replacing full-attention q/k norm modules with
the Qwen3.5 RMSNorm class while using load-time offset might restore semantics
without runtime graph subtraction.

### Method

Created a second disposable overlay from the load-time norm-offset branch. The
first broad edit accidentally made `_qwen35_effective_weight` fire for both
runtime and load-time switches; that was corrected before launch so runtime
subtraction stayed disabled. The q/k norm module replacement was enabled when
either the runtime or load-time norm switch was set.

### Results

The server reached health, but the smoke prompt again returned an empty
one-token stop with no content. The full benchmark ladder was not run because
the previous load-time variant had already failed the strict gate and this
smoke reproduced the same pathological output shape.

Graph-cache inspection showed the intended structural difference was present:

- load-time-only graph cache: `9235` lines, about `28M`, no runtime
  `weight - 1.0` body;
- load-time plus q/k class graph cache: `9235` lines, about `28M`, no runtime
  `weight - 1.0` body;
- prior runtime-offset graph cache: `9336` lines, about `32M`, explicit
  `weight - 1.0` bodies;
- HF graph cache: `9145` lines, about `23M`.

### Outcome

Reject load-time offset plus q/k norm class replacement as a serving path.

### Promote / Reject

Promote the graph comparison as source evidence. Reject the overlay as a
correctness fix. The runtime norm-offset path remains the active correctness
baseline until a tensor-level audit proves which non-SSM norm tensors can be
shifted at load time without changing Qwen strict/parser behavior.

### Next

Run a tensor-level norm audit before another launch. Compare representative
loaded weights for input layernorm, post-attention layernorm, full-attention
q/k norms, final norm, and SSM norms between the runtime-offset baseline, the
load-time variants, and the HF-weight control. Do not run another warmup ladder
until that audit identifies the exact tensor convention mismatch.

## GGUF-154 - MoE GGUF Raw Norm Tensor Audit

### Hypothesis

The load-time norm-offset variants may have failed because the broad
`name.endswith("norm.weight")` hook shifted the wrong tensors, missed active
non-SSM tensors, or double-shifted tensors that already matched HF layout.

### Method

Ran a read-only tensor audit on `.20` against the staged
Qwen3.6 35B-A3B F16 GGUF file and the local HF safetensors snapshot. The audit
mapped GGUF norm names to HF/vLLM names:

- `blk.N.attn_norm.weight` -> `model.language_model.layers.N.input_layernorm.weight`
- `blk.N.post_attention_norm.weight` -> `model.language_model.layers.N.post_attention_layernorm.weight`
- `blk.N.ssm_norm.weight` -> `model.language_model.layers.N.linear_attn.norm.weight`
- `blk.N.attn_q_norm.weight` -> `model.language_model.layers.N.self_attn.q_norm.weight`
- `blk.N.attn_k_norm.weight` -> `model.language_model.layers.N.self_attn.k_norm.weight`
- `output_norm.weight` -> `model.language_model.norm.weight`

For each mapped tensor, the audit compared raw GGUF to HF and
`raw GGUF - 1.0` to HF. It also applied the current load-time hook decision:
shift if the mapped name ends with `norm.weight` and is not `ssm_norm`.

### Results

The audit covered `131` mapped norm tensors:

- input layer norms: `40`
- post-attention norms: `40`
- SSM norms: `30`
- full-attention q norms: `10`
- full-attention k norms: `10`
- final norm: `1`

The file-level convention is clean:

- non-SSM norms differ from HF by exactly `+1.0` in raw GGUF form;
- subtracting `1.0` makes input, post-attention, q norm, k norm, and final
  norm tensors match HF within FP32 comparison noise;
- SSM norms already match HF directly and become wrong if shifted by `-1.0`;
- the current load-time hook has no misses for non-SSM norms;
- the current load-time hook does not shift any SSM norm.

Representative rows:

- `blk.0.attn_norm.weight` mean `1.031111` versus HF
  `model.language_model.layers.0.input_layernorm.weight` mean `0.031111`;
  `raw - 1.0` matches.
- `blk.0.post_attention_norm.weight` mean `0.895086` versus HF
  `model.language_model.layers.0.post_attention_layernorm.weight` mean
  `-0.104914`; `raw - 1.0` matches.
- `blk.0.ssm_norm.weight` mean `0.884155` versus HF
  `model.language_model.layers.0.linear_attn.norm.weight` mean `0.884155`;
  raw matches and `raw - 1.0` is wrong.
- `blk.3.attn_q_norm.weight` mean `1.325485` versus HF
  `model.language_model.layers.3.self_attn.q_norm.weight` mean `0.325485`;
  `raw - 1.0` matches.
- `output_norm.weight` mean `2.627906` versus HF
  `model.language_model.norm.weight` mean `1.627906`; `raw - 1.0` matches.

### Outcome

Promote the load-time hook's raw tensor selection rule as mathematically
correct at the GGUF/HF file level.

Reject the theory that the load-time failures were caused by a broad norm-name
selection error, a missed non-SSM norm tensor, or an SSM norm being shifted.

### Promote / Reject

Promote as source evidence, not as a serving fix. The load-time variants still
failed smoke/strict behavior even though the raw tensor rule is correct. The
remaining failure is likely in live module materialization or runtime class
semantics, not in the raw GGUF tensor convention.

### Next

Add a live module-param audit before the next serving run. It should dump
aggregate stats for loaded module parameters after vLLM weight loading for:

- input layernorm;
- post-attention layernorm;
- full-attention q/k norms;
- final norm;
- SSM norm.

Compare runtime-offset GGUF, load-time GGUF, and HF control module parameters.
Only after live module stats match should another warmups -> `c1_128` strict
-> `c1_2000` -> `c1_10000` ladder be run.

## GGUF-155 - Corrected Load-Time Norm Offset And Full Ladder

### Hypothesis

The failed load-time norm-offset variants may have been structurally correct at
the file level but wrong in the live vLLM module path because the hook fired
twice and shifted the mapped SSM norm name.

### Method

Created disposable `.20` audit overlays from the runtime-offset and load-time
q/k-class variants. Each overlay dumped live module norm-parameter stats after
vLLM weight loading and before any request.

The first live audit compared:

- runtime-offset GGUF path;
- original load-time q/k-class path.

Then a corrected load-time overlay was created:

- removed the load-time hook from `Qwen3_5Model.load_weights`;
- kept the hook in the top-level CausalLM iterator;
- excluded both `ssm_norm.weight` and the mapped live name
  `linear_attn.norm.weight`;
- kept full-attention q/k norm class replacement active for the load-time
  switch.

After live parameter stats matched the expected convention, the corrected path
ran the normal benchmark sequence: warmups -> `c1_128` uncapped strict ->
`c1_2000` -> `c1_10000`.

### Results

The original load-time q/k-class variant was live-wrong:

- runtime path stored raw GGUF-convention params, for example layer-0 input
  norm mean `1.031100`;
- original load-time path stored layer-0 input norm mean `-0.968885`, a
  `-1.999985` delta from runtime;
- SSM was also shifted incorrectly: layer-0 `linear_attn.norm.weight` moved
  from `0.884155` to `-1.115845`, a `-2.000000` delta.

That explains the earlier empty/invalid load-time output: the hook was applied
twice, and the SSM exclusion missed the mapped vLLM name.

The corrected load-time overlay fixed live module stats:

- layer-0 input norm: runtime `1.031100`, corrected `0.031111`;
- layer-0 post-attention norm: runtime `0.895085`, corrected `-0.104914`;
- layer-0 SSM norm: runtime `0.884155`, corrected `0.884155`;
- layer-3 q norm: runtime `1.325489`, corrected `0.325485`;
- layer-3 k norm: runtime `1.314066`, corrected `0.314066`;
- final norm: runtime `2.627906`, corrected `1.627906`.

The corrected path reached health. A tiny capped smoke generated coherent
reasoning text but did not close inside the cap. The normal benchmark ladder
then completed:

| case | backend decode TPS | completion tokens | finish | strict gate valid |
| --- | ---: | ---: | --- | --- |
| `c1_128_strict` | `71.995` | `3053` | stop | true |
| `c1_2000` | `73.005` | `2000` | length | false |
| `c1_10000` | `65.456` | `10000` | length | false |

The corrected graph cache hash was `8e32cfe515`; its computation graph had
`9235` lines, matching the earlier load-time graph size and removing the
runtime `weight - 1.0` graph bodies.

### Outcome

Promote the corrected load-time norm-offset implementation as a strict-valid
source candidate and as the confirmed fix for the previous load-time semantic
failure.

Reject it as a performance promotion. It did not beat the current best MoE
GGUF result (`71.916` strict, `73.380` `c1_2000`, `65.749` `c1_10000`) and is
still far below the FP16 v0.2.1 TP4 reproduction path.

### Promote / Reject

Promote as a source cleanup candidate only: it shrinks the graph and preserves
strict correctness when applied once with the mapped SSM exclusion.

Reject as the path to benchmark parity. The remaining gap is not the runtime
norm subtraction graph body. The next source target should move to
CausalLM-versus-HF graph structure, model-runner/decode replay, or a
graph-visible fixed-token profiling route.

## GGUF-156 - Corrected Load-Time Norm Offset With Broad Scheduler

### Hypothesis

The corrected load-time norm-offset path in GGUF-155 preserved strict
correctness but used the restrictive release-style scheduler arguments
`--max-num-seqs 1 --max-num-batched-tokens 1024`, which limited CUDA graph
capture to `PIECEWISE=2` and `FULL=1`. A broad scheduler launch might recover
some of the older default-scheduler GGUF throughput while keeping the cleaner
load-time norm graph.

### Method

Relaunched the corrected load-time overlay on `.20` with:

- `moe35b_tp4_fullbar_p2pon`;
- P2P-on;
- `MAX_MODEL_LEN=131072`;
- same v0.2 image and MoE tuned config;
- load-time norm offset applied once;
- mapped SSM norm excluded from the shift;
- full-attention q/k norm class replacement enabled;
- no `--max-num-seqs 1 --max-num-batched-tokens 1024` override.

The server reached health with broad graph capture:

- chunked prefill `max_num_batched_tokens=2048`;
- `PIECEWISE=51`, largest `512`;
- `FULL=35`, largest `256`;
- torch compile cache hash `686a95799f`.

The rank-0 graph signature for the broad corrected path was:

- computation graph lines: `9235`;
- parameter inputs: `242`;
- `gdn_attention_core`: `60`;
- `moe_forward`: `160`;
- `moe_forward_shared`: `120`;
- `all_reduce`: `244`;
- runtime `weight - 1`: `0`.

Then ran the normal benchmark sequence: warmups -> `c1_128` uncapped strict ->
`c1_2000` -> `c1_10000`.

### Results

Warmups settled in the same broad GGUF band after the first request:

- warmup 1 backend decode TPS: `70.363`;
- warmups 2-8 backend decode TPS: about `73.345` to `73.445`.

Final ladder:

| case | backend decode TPS | completion tokens | finish | strict gate valid |
| --- | ---: | ---: | --- | --- |
| `c1_128_strict` | `71.785` | `3642` | stop | true |
| `c1_2000` | `73.396` | `2000` | length | false |
| `c1_10000` | `65.796` | `10000` | length | false |

### Outcome

Promote broad scheduler shape as the right launch shape for corrected
load-time GGUF experiments. It preserves strict validity and avoids the narrow
`PIECEWISE=2` / `FULL=1` capture shape.

Reject this as a benchmark-parity path. It slightly improves the best observed
GGUF `c1_2000` number (`73.396` versus previous `73.380`) and improves
corrected load-time `c1_10000` over GGUF-155 (`65.796` versus `65.456`), but it
does not approach the FP16 TP4 reproduction path and does not materially change
the slow GGUF band.

### Promote / Reject

Promote as the current clean source baseline for future MoE GGUF profiling:
load-time norm offset once, mapped SSM exclusion, q/k norm class repair, broad
graph capture.

Reject scheduler shape as the missing performance lever. Broad graph capture is
necessary for comparable testing, but the remaining gap is deeper than the
restrictive capture shape.

### Next

Continue with graph-visible whole-decode or kernel-level evidence. The graph
operation counts between broad GGUF and aligned HF controls are nearly
structurally identical for the major boundaries already measured
(`gdn_attention_core`, `moe_forward`, `moe_forward_shared`, and all-reduce), so
the next experiment should measure replay/kernel mix or compare generated
Inductor code regions rather than re-running scheduler or norm variants.

## GGUF-157 - Compile-Debug Config Via Single-Quoted JSON

### Hypothesis

vLLM's `--compilation-config` can emit readable torch.compile debug dumps and
unpacked compile-cache artifacts for the current MoE GGUF broad corrected path.

### Method

Launched the current `.20` MoE GGUF TP4 broad corrected source baseline with a
single-quoted JSON `--compilation-config` string passed through the existing
container entrypoint argument path. This was a launch-form test only; no model
benchmark was attempted.

### Results

The launch failed before model loading. vLLM rejected the argument as invalid
JSON because the literal single quotes survived into the CLI value and the
object keys were not parsed as JSON strings.

### Outcome

Reject this launch form.

### Promote / Reject

Reject as a syntax / entrypoint quoting failure, not as model, source, or
benchmark evidence.

### Next

Do not repeat single-quoted JSON through the existing entrypoint argument
split. Use an entrypoint-safe argument path before trying compile debug again.

## GGUF-158 - Compile-Debug Config Via Escaped Double Quotes

### Hypothesis

Escaped double quotes inside the existing entrypoint argument path might
preserve valid JSON for vLLM `--compilation-config`.

### Method

Relaunched the same `.20` MoE GGUF TP4 broad corrected source baseline with an
escaped double-quoted JSON `--compilation-config` string through the existing
entrypoint argument split. This was again a launch-form test only.

### Results

The launch failed before model loading. The entrypoint/read path stripped or
split the quotes so vLLM again saw invalid JSON and reported that the key must
be a string.

### Outcome

Reject this launch form.

### Promote / Reject

Reject as another syntax / entrypoint quoting failure. It does not say anything
about GGUF model correctness or performance.

### Next

Use a safer argument append route that passes the JSON string as one argv item.

## GGUF-159 - Compile-Debug Wrapper Entrypoint

### Hypothesis

A disposable wrapper entrypoint that appends
`--compilation-config "${VLLM_COMPILATION_CONFIG_JSON}"` as a Bash-array element
can pass a valid compilation config without modifying the release image or
benchmark source.

### Method

Copied the image entrypoint into the `.20` GGUF run workspace, added an
env-gated Bash-array append for `VLLM_COMPILATION_CONFIG_JSON`, and launched the
current broad corrected MoE GGUF TP4 baseline with:

- profile: `moe35b_tp4_fullbar_p2pon`;
- TP4;
- P2P-on;
- `MAX_MODEL_LEN=131072`;
- same v0.2 image;
- same broad corrected GGUF MoE overlay as GGUF-156;
- debug dump path and unpacked compile-cache path under the run workspace.

This was a source-inspection run, not a benchmark run. After API health, one
short smoke prompt was sent to confirm the server was coherent enough to justify
keeping the artifacts. The prompt was:

```text
Hello. In one short sentence, say what local AI reproducibility means.
```

### Results

The wrapper passed the compilation config correctly. vLLM logged the accepted
debug dump path, compile-cache path, and `compile_cache_save_format='unpacked'`.

Startup and graph signatures:

- model class resolved as `Qwen3_5MoeForCausalLM`;
- dtype `torch.float16`;
- graph mode reached health;
- model load: about `17.23 GiB`, about `73.19` seconds;
- torch compile total: about `33.65` seconds;
- graph capture: `PIECEWISE=51`, largest `512`; `FULL=35`, largest `256`;
- KV-cache capacity: `302016` tokens;
- max concurrency: about `8.98x`;
- AOT graph hash:
  `b5fab22e93990a3258b763cf688ae25969b823a138e729f0bb129d0f60e29b3a`.

Artifact summary:

- debug dump: about `26M`, `244` files;
- compile cache: about `94M`, `100` files;
- each rank graph contained `9235` lines;
- each rank had `61` generated debug kernel files;
- rank-0 debug kernel files contained no `tl.dot` occurrences.

Rank-0 graph operation counts in the unpacked computation graph:

- `torch.ops.vllm.gdn_attention_core`: `60`;
- `torch.ops.vllm.moe_forward_shared`: `40`;
- `torch.ops.vllm.moe_forward`: `40`;
- `all_reduce(` occurrences: `162`;
- `_qwen35_effective_weight` references: `101`.

The `_qwen35_effective_weight` source body in the captured graph returned
`self.weight.data`; it did not contain the runtime `weight - 1.0` subtraction
body. This confirms the corrected load-time norm path moved the norm correction
out of the compiled runtime graph for this diagnostic shape.

The smoke response was coherent Qwen-style text, although the short cap stopped
the response by length. No benchmark ladder was run.

### Outcome

Promote the wrapper entrypoint as a source-inspection route for future
compile-debug runs. It captures readable graph and kernel artifacts without
modifying the release image.

Reject as benchmark or performance evidence. No warmups, strict run,
`c1_2000`, or `c1_10000` tier was run.

### Promote / Reject

Promote as a diagnostic artifact route.

Reject as a performance promotion. It only proves that the broad corrected MoE
GGUF path can emit readable compile artifacts and coherent smoke output.

### Next

Compare the GGUF compile graph and generated kernels against a similarly dumped
HF-weight TP4 control, or instrument graph-visible replay/kernel mix around the
already narrowed boundaries. Do not rerun norm-offset or scheduler variants
unchanged.

## GGUF-160 - External Qwen3.6 MI50 GGUF Patch Bundle Review

### Hypothesis

The external `Kausik-A/qwen3.6-27b-mi50-vllm` repository might contain
source-level compatibility changes useful to the GFX906 GGUF investigation.

### Method

Reviewed the repository at commit `61b273d` outside this repo. Inspected its
README, compose launch, and five vLLM patch files:

- `patches/config.py`;
- `patches/gguf_loader.py`;
- `patches/linear.py`;
- `patches/qwen3_5.py`;
- `patches/registry.py`.

### Results

Useful source ideas:

- `qwen3_5` / `qwen3_5_moe` config and registry plumbing;
- `qwen35` and `qwen35moe` GGUF architecture aliases;
- `ssm_dt.bias -> linear_attn.dt_bias` mapping;
- text-only M-RoPE stripping for Qwen3.5 CausalLM/MoE;
- quantized `embed_tokens` and `lm_head` plumbing;
- GGUF `conv1d.weight` 2D-to-3D reshape;
- `IsHybrid` hooks on text-only Qwen3.5 classes;
- tuple-shard splitting in `MergedColumnParallelLinear`;
- MiniMax M2 GGUF expert-name aliases.

Non-promoting differences:

- the project targets a single MI50/eGPU path, not TP4/TP8 full-BAR/P2P-on;
- it uses eager mode and a 4096 context deployment shape;
- it is ROCm 6.3-era rather than the ROCm7.2 release path;
- it does not provide the release benchmark warmup/strict/fixed-token ladder;
- it does not solve the active MoE `A_log`/norm/q-k parity and branch-flip
  issues that were already reconstructed in our source lane.

### Outcome

Promote as an external compatibility checklist and MiniMax/Qwen GGUF mapping
reference.

Reject as a throughput or release-reproduction route for this goal.

### Promote / Reject

Promote specific loader compatibility ideas only where they survive direct
comparison with our current source path.

Reject copying its launch profile or treating it as proof of GFX906 TP
benchmark parity.

### Next

Use it as a checklist when auditing new GGUF model-family support, especially
MiniMax and Qwen3.5 text-only GGUF loading. Do not replace the current `.20`
release-reproduction path with the eGPU/eager launch shape.

## GGUF-161 - HF TP4 Compile-Debug Control

### Hypothesis

If the broad corrected MoE GGUF TP4 path is missing a major graph region or
custom-op selection relative to the HF-weight TP4 release path, an aligned
compile-debug control should show it in the unpacked graph and generated source
artifacts.

### Method

Launched an HF-weight Qwen3.6 35B-A3B TP4 control on `.20` using the same
release image, the native `moe35b_tp4_fullbar_p2pon` profile shape, P2P-on,
`MAX_MODEL_LEN=131072`, broad scheduler defaults, graph mode, and the same
wrapper-entrypoint compile-debug route from GGUF-159.

This was a source-control run, not a benchmark run. After API health, one short
smoke prompt was sent to confirm the server was responsive before collecting
artifacts.

### Results

Startup and graph signatures:

- model class resolved as `Qwen3_5MoeForConditionalGeneration`;
- dtype `torch.float16`;
- model load: about `17.23 GiB`, about `45.97` seconds;
- torch compile total: about `26.23` seconds;
- graph capture: `PIECEWISE=51`, largest `512`; `FULL=35`, largest `256`;
- KV-cache capacity: `302016` tokens;
- max concurrency: about `8.98x`;
- AOT graph hash:
  `d9ac113acb4972d7ae2b3daf7ee4764594c1242d5b30252658e95699fae13974`.

Artifact summary:

- debug dump: about `26M`, `244` files;
- compile cache: about `94M`, `100` files;
- each rank graph contained `9145` lines;
- rank-0 generated debug-kernel files: `61`;
- rank-0 debug kernel files contained no `tl.dot` occurrences.

Rank-0 graph operation counts in the unpacked computation graph:

- `torch.ops.vllm.gdn_attention_core`: `60`;
- `torch.ops.vllm.moe_forward_shared`: `40`;
- `torch.ops.vllm.moe_forward`: `40`;
- `torch.ops.vllm.all_reduce`: `162`;
- `_qwen35_effective_weight` references: `0`.

Comparison with the GGUF-159 broad corrected GGUF graph:

- high-level vLLM op counts matched exactly for all measured custom ops:
  `all_reduce`, `rocm_unquantized_gemm`, `gdn_attention_core`,
  `moe_forward_shared`, `unified_kv_cache_update`, and
  `unified_attention_with_output`;
- both paths captured the same broad graph shape: `PIECEWISE=51`, `FULL=35`;
- both paths emitted `61` rank-0 generated debug-kernel files with no
  `tl.dot` occurrences;
- GGUF still had a larger rank graph (`9235` lines versus `9145`) because it
  carried `_qwen35_effective_weight` call sites, but those sites returned
  `self.weight.data` and contained no runtime `weight - 1.0` subtraction.

The smoke request reached Qwen-style reasoning text and stopped at the short
token cap. It was not a benchmark response and was not used for promotion.

### Outcome

Promote as an HF source-control artifact for the broad TP4 profile. The major
compiled custom-op regions match between HF and corrected-load-time GGUF at this
inspection level.

Reject missing broad graph capture, missing GDN/MoE/all-reduce regions, and
runtime norm-subtraction overhead as the current whole-gap explanation.

### Promote / Reject

Promote the HF control as the comparator for future generated-source or
graph-replay investigations.

Reject as performance evidence. No warmups, strict run, `c1_2000`, or
`c1_10000` tier was run.

### Next

Move source work below the current graph-summary level. Compare generated
Inductor regions, graph replay behavior, HIP kernel mix, or fixed-token
decode-window traces. Do not rerun unchanged scheduler, norm-offset,
FusedMoE/GDN wrapper, logits, sampler, or Python runner timing variants.

## GGUF-162 - MoE TP4 Trace-Gated QKV Split Cleanup

### Hypothesis

The corrected GGUF graph still carried unused Q/K/V split nodes that the HF
graph did not carry. Those nodes came from trace-only tensor logging prep in
the GGUF Qwen3.5 path: the code split `mixed_qkv` into Q, K, and V before
checking whether layer-0 tracing was enabled. Moving that split under the
`trace_active` guard should make the disabled-trace serving graph closer to the
HF path.

This was expected to be a source hygiene cleanup. It was not expected to fix
the whole throughput gap unless the unused split forced enough graph or memory
traffic to affect decode replay.

### Method

Copied the current corrected-load-time MoE GGUF patch bundle on `.20` and made
one surgical source change in `qwen3_5.py`: the trace-only Q/K/V split now runs
only when `trace_active` is true.

Launched the normal MoE GGUF TP4 broad profile on `.20`:

- model: Qwen3.6 35B-A3B F16 GGUF;
- profile: `moe35b_tp4_fullbar_p2pon`;
- tensor parallel: TP4;
- P2P: on;
- dtype: FP16;
- max model length: `131072`;
- graph mode: broad default capture, `PIECEWISE=51`, `FULL=35`;
- same release image and model-specific GGUF env/overlays;
- trace disabled: `VLLM_QWEN35_TRACE_LAYER0=0`;
- benchmark sequence: eight normal warmups, uncapped strict, `c1_2000`,
  `c1_10000`.

### Results

Startup and graph signatures:

- resolved architecture: `Qwen3_5MoeForCausalLM`;
- model load: about `17.23 GiB`, about `94.53` seconds;
- torch compile total: about `30.58` seconds;
- AOT graph hash:
  `224188a6b6a29c71a85d735f590bd2ce704392d50cb0a3a24287867b7d630094`;
- graph capture: `PIECEWISE=51`, largest `512`; `FULL=35`, largest `256`;
- KV-cache capacity: `302016` tokens;
- max concurrency: about `8.98x`;
- graph capture completed in about `167` seconds and used about `2.11 GiB`.

Warmups were stable in the same GGUF band:

- warmup 1 backend TPS: `70.533`;
- warmups 2-8 backend TPS range: `73.007` to `73.158`.

Normal ladder:

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | 4367 | 70.297 | 70.766 | 62.122 | stop | true |
| `c1_2000` | 2000 | 72.021 | 73.104 | 27.770 | length | false |
| `c1_10000` | 10000 | 65.369 | 65.546 | 152.979 | length | false |

### Outcome

Promote the patch as source hygiene: disabled tracing no longer creates the
unused Q/K/V split nodes that separated the GGUF graph from the HF graph.

Reject it as the performance unlock. It did not improve the MoE GGUF TP4
throughput class. The result stayed in the same band as the corrected
load-time broad baseline (`71.785`, `73.396`, `65.796`) and remains far below
the FP16 TP4 reproduction target.

### Promote / Reject

Promote only as a cleanup candidate for a future GGUF source branch.

Reject as a benchmark promotion, release-doc update, or container update. The
release image and public release claims should not change from this result.

### Next

Do not repeat trace-gating, norm-offset, scheduler-width, FusedMoE/GDN wrapper,
logits, sampler, or Python-runner timing variants unchanged. The next useful
source work is below the current graph-summary level: generated Inductor
regions, graph replay behavior, HIP kernel mix, or fixed-token decode-window
traces against the aligned HF TP4 comparator.

## GGUF-163 - Dense TP8 `VLLM_BATCH_INVARIANT` Startup Test

### Hypothesis

The external Qwen3.6 MI50 GGUF patch review left one untested vLLM source-path
switch in the dense GGUF current-best path: `VLLM_BATCH_INVARIANT`. Since the
dense GGUF gap is smaller than the MoE gap and the benchmark ladder uses
single-client decode, the batch-invariant linear path was worth one controlled
startup test.

### Method

Copied the current-best dense GGUF TP8 launch on `.20`:

- model: Qwen3.6 27B F16 GGUF;
- profile: dense TP8 full-BAR/P2P-on;
- tensor parallel: TP8;
- P2P: on;
- dtype: FP16;
- max model length: `131072`;
- graph mode: release dense GGUF source bundle with fused GDN and native GGUF
  SwiGLU;
- benchmark sequence intended if serving reached health: eight warmups,
  uncapped strict, `c1_2000`, `c1_10000`.

Two startup variants were tried:

1. `VLLM_BATCH_INVARIANT=1` with the existing launch arguments.
2. `VLLM_BATCH_INVARIANT=1` plus explicit `--attention-backend FLASH_ATTN`,
   because the first variant refused to initialize without an explicit
   supported attention backend.

### Results

The first variant failed before model load. vLLM rejected batch-invariant mode
because `attention_config.backend` was `None` and requested an explicit backend
from the supported set.

The second variant passed that gate and reached model initialization, but
failed before serving. TorchInductor autotuning reported no valid Triton config
for gfx906 because the generated path required `163840` bytes of shared memory
against a hardware limit of `65536` bytes.

No benchmark warmups were run, no strict request was sent, and no fixed-token
tiers were promoted.

### Outcome

Reject `VLLM_BATCH_INVARIANT` for the current dense GGUF TP8 reproduction path.
The mode is not merely neutral; it currently prevents the `.20` dense GGUF
current-best path from reaching health unless additional lower-level Triton /
Inductor tuning is done.

### Promote / Reject

Promote only as negative source-path evidence.

Reject as a performance fix, benchmark candidate, release-doc update, or
container update.

### Next

Do not repeat batch-invariant unchanged. A future attempt would need a specific
gfx906-safe Triton/Inductor configuration or source patch that reduces shared
memory use. Continue below the graph-summary boundary: generated Inductor
regions, graph replay behavior, HIP kernel mix, or decode-window traces against
the aligned HF comparator.

## GGUF-164 - MoE TP4 Direct RMSNorm Weight Lookup

### Hypothesis

The corrected load-time MoE GGUF graph still carried
`_qwen35_effective_weight` call sites while the aligned HF TP4 control did not.
After the load-time norm-offset fix, the helper body only returned
`self.weight.data`, so replacing the wrapper call in `forward_native()` with a
direct weight lookup should make the GGUF graph closer to the HF graph at that
known residual difference.

This was expected to be a source cleanup and a possible small timing win. It
was not expected to solve the whole MoE GGUF gap unless those wrapper sites
were enough to perturb graph capture or replay.

### Method

Copied the trace-gated corrected-load-time MoE GGUF patch bundle on `.20` and
made one surgical source change in `qwen3_5.py`:

```text
weight = self._qwen35_effective_weight()
```

became:

```text
weight = self.weight.data
```

Launched the normal MoE GGUF TP4 broad profile:

- model: Qwen3.6 35B-A3B F16 GGUF;
- profile: `moe35b_tp4_fullbar_p2pon`;
- tensor parallel: TP4;
- P2P: on;
- dtype: FP16;
- max model length: `131072`;
- graph mode: broad default capture, `PIECEWISE=51`, `FULL=35`;
- same release image, tuned MoE config, and model-specific GGUF env/overlays;
- benchmark sequence: eight normal warmups, uncapped strict, `c1_2000`,
  `c1_10000`.

### Results

Startup and graph signatures:

- model load: about `17.23 GiB`, about `92.50` seconds;
- torch compile total: about `29.78` seconds;
- graph capture: `PIECEWISE=51`, largest `512`; `FULL=35`, largest `256`;
- KV-cache capacity: `302016` tokens;
- max concurrency: about `8.98x`;
- graph capture completed in about `169` seconds and used about `2.11 GiB`;
- the resulting compile cache had zero `_qwen35_effective_weight` references.

Warmups stayed in the existing GGUF band:

- warmup 1 backend TPS: `73.014`;
- warmups 2-8 backend TPS range: `72.943` to `73.041`.

Normal ladder:

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | 3544 | 70.899 | 71.489 | 49.987 | stop | true |
| `c1_2000` | 2000 | 71.934 | 73.015 | 27.803 | length | false |
| `c1_10000` | 10000 | 65.278 | 65.454 | 153.191 | length | false |

### Outcome

Reject direct RMSNorm weight lookup as a performance unlock. It did not move
the MoE GGUF TP4 path out of the slow band and remains far below the FP16 TP4
reproduction target.

The strict result improved slightly versus the trace-gated cleanup run
(`71.489` versus `70.766` backend TPS), but `c1_2000` and `c1_10000` did not
improve versus the corrected-load-time broad baseline (`73.396` and `65.796`).
That is not enough to promote.

### Promote / Reject

Promote only as possible source hygiene if a future cleanup branch wants to
remove vestigial wrapper calls after load-time norm correction.

Reject as a benchmark promotion, release-doc update, or container update. The
release image and public release claims should not change from this result.

### Next

Do not repeat direct RMSNorm wrapper removal unchanged. The remaining gap is
still below the high-level graph-summary boundary. Next useful evidence should
come from graph replay / HIP kernel mix, generated Inductor region comparison,
or request-window profiling against the aligned HF TP4 comparator.

## GGUF-165 - MoE TP4 ConditionalGeneration Architecture Smoke

### Hypothesis

The aligned HF TP4 control resolves as `Qwen3_5MoeForConditionalGeneration`,
while the working MoE GGUF TP4 path resolves as `Qwen3_5MoeForCausalLM`.
Dense ConditionalGeneration overrides were already rejected, but the MoE GGUF
architecture flip itself had not been tested. If the wrapper changed page
geometry, renderer setup, or hybrid-cache alignment, it could explain part of
the MoE GGUF performance gap.

### Method

Copied the current trace-gated corrected-load-time MoE GGUF patch bundle on
`.20` and changed only the text config architecture from:

```text
Qwen3_5MoeForCausalLM
```

to:

```text
Qwen3_5MoeForConditionalGeneration
```

Launched the same MoE GGUF TP4 smoke profile:

- model: Qwen3.6 35B-A3B F16 GGUF;
- profile: `moe35b_tp4_fullbar_p2pon`;
- tensor parallel: TP4;
- P2P: on;
- dtype: FP16;
- max model length: `131072`;
- graph mode and tuned MoE config unchanged;
- no benchmark requests were sent unless the server reached health.

### Results

The server resolved the requested architecture:

```text
Resolved architecture: Qwen3_5MoeForConditionalGeneration
```

It also selected the same page geometry as the working CausalLM MoE GGUF
baseline:

- attention block size: `528` tokens;
- Mamba page padding: `0.76%`.

Startup failed before model load and before the API socket opened. vLLM entered
the multimodal renderer/processor path for the ConditionalGeneration class, but
the text-only GGUF config was a `Qwen3_5MoeTextConfig` where that processor
expected the outer `Qwen3_5MoeConfig`.

No warmups, strict request, `c1_2000`, or `c1_10000` benchmark was run.

### Outcome

Reject the simple MoE GGUF ConditionalGeneration architecture flip. It is not a
drop-in performance route for the current text-only GGUF path.

This smoke also rejects page geometry as the reason to repeat the architecture
flip: both the failed ConditionalGeneration path and the working CausalLM path
selected the same attention block size and Mamba padding.

### Promote / Reject

Promote as source-routing evidence only. A future ConditionalGeneration route
would need a real text-config adapter or multimodal-processor bypass, not just
an `architectures` change.

Reject as a benchmark promotion, release-doc update, or container update.

### Next

Do not repeat MoE ConditionalGeneration as an architecture-only override. The
next useful MoE GGUF work should compare generated Inductor regions, graph
replay / HIP kernel mix, or fixed-token decode-window traces against the HF TP4
control. If the ConditionalGeneration route is revisited, first implement a
specific text-config adapter and prove it reaches health before any benchmark
ladder.

## GGUF-166 - HF Versus GGUF Generated Kernel Source Diff

### Hypothesis

The HF TP4 control and corrected-load-time GGUF TP4 path have matching broad
graph shape and matching measured vLLM custom-op counts, but GGUF remains much
slower. A generated-kernel source diff might expose a lower-level codegen
difference that the high-level graph summary missed.

### Method

Compared the existing `.20` compile-debug artifacts from:

- GGUF corrected-load-time broad graph source-inspection run;
- HF TP4 broad graph source-control run.

No server was launched. This was a read-only artifact comparison.

The comparison checked rank-0 debug-dump files for:

- generated kernel file count;
- generated kernel symbol names;
- line and byte counts;
- raw file equality;
- normalized executable equality after removing comments, generated UUID/hash
  strings, placeholder numbering, and the serialized `ModuleName` line.

### Results

Rank-0 generated kernel summary:

- GGUF generated kernel files: `14`;
- HF generated kernel files: `14`;
- generated function-name multiset: identical;
- measured vLLM custom-op counts: still identical at the prior inspection
  boundary;
- `11/14` generated kernel files were byte-identical before normalization.

The only three raw kernel-file differences had identical line counts and byte
counts:

- `__compiled_fn_1.kernel_1.py`;
- `__compiled_fn_1.kernel_4.py`;
- `__compiled_fn_1.kernel_9.py`.

The executable difference in those files was limited to a serialized
`ModuleName` prefix:

- HF: `language_model.model.layers.*.mlp.experts`;
- GGUF: `model.layers.*.mlp.experts`.

After removing comments and the serialized `ModuleName` line, all rank-0
generated kernel files matched.

### Outcome

Reject generated Inductor kernel source differences as the current explanation
for the MoE GGUF TP4 throughput gap at this artifact level.

The generated kernel bodies are effectively identical once model-name metadata
is ignored. The remaining performance gap is more likely in graph replay,
runtime scheduling, HIP kernel launch mix/timing, memory movement, or request
window behavior outside these generated source bodies.

### Promote / Reject

Promote this as a source-boundary result: high-level graph shape, custom-op
counts, generated kernel symbol names, and generated kernel bodies are all
aligned between the HF TP4 control and corrected GGUF path.

Reject further static generated-kernel body diffing unless a new source patch
changes the generated code.

### Next

Move below static source artifacts. The next useful diagnostic should inspect
replay-time behavior: HIP kernel mix/timing, graph replay boundaries, memory
movement, or fixed-token decode-window traces against the HF TP4 control.

## GGUF-167 - MoE Tuned-Config Lookup And ModuleName Hypothesis

### Hypothesis

The generated-kernel source diff found that the remaining HF/GGUF raw source
difference was serialized model-module metadata:

- HF: `language_model.model.layers.*.mlp.experts`;
- GGUF: `model.layers.*.mlp.experts`.

If tuned MoE config selection depended on that module-name prefix, GGUF could
silently miss the tuned GFX906 MoE configuration even while the generated
kernel bodies looked identical.

### Method

Checked the existing `.20` compile-debug container logs for the corrected GGUF
TP4 broad graph run and the aligned HF TP4 broad graph control. No server was
launched.

The check counted exact tuned-config messages for:

`E=256,N=128,device_name=AMD_GFX906.json`

and counted GGUF force-unquantized MoE method messages separately.

### Results

- GGUF corrected broad graph run:
  - tuned MoE config load messages: `1`;
  - `Forcing unquantized FusedMoE method` messages: `160`.
- HF broad graph control:
  - tuned MoE config load messages: `1`;
  - `Forcing unquantized FusedMoE method` messages: `0`.
- Both paths used the same tuned config file:
  `E=256,N=128,device_name=AMD_GFX906.json`.

### Outcome

Reject missing tuned MoE config selection as the current MoE GGUF throughput
explanation.

The `ModuleName` prefix difference in generated kernel metadata does not prevent
the tuned GFX906 MoE config from loading. The GGUF force-unquantized count is
the expected F16 GGUF expert-method routing, not evidence that the tuned config
was skipped.

### Promote / Reject

Promote this as a source-boundary result: HF and GGUF both reach the tuned MoE
configuration boundary.

Reject more tuned-config lookup debugging unless a future source patch changes
MoE method selection or config lookup inputs.

### Next

Continue below static graph/config selection. The next useful diagnostic remains
replay-time behavior: HIP kernel mix/timing, graph replay, memory movement, or
fixed-token decode-window traces against the HF TP4 control.

## GGUF-168 - MoE HF/GGUF Pre-Grad Signature And Body Boundary

### Hypothesis

The generated kernel bodies matched after metadata normalization, but the
larger `BEFORE_PRE_GRAD` graph files might still expose visible tensor
signature or materialization differences between the corrected GGUF TP4 path
and the HF TP4 control.

### Method

Compared existing `.20` rank-0 debug artifacts from:

- corrected GGUF TP4 broad graph source-inspection run;
- aligned HF TP4 broad graph source-control run.

No server was launched. The comparison used POSIX shell extraction of the
single `def forward` signature line from every
`__compiled_fn_1.BEFORE_PRE_GRAD.*.py` file and then compared raw and normalized
signature hashes.

### Results

- `BEFORE_PRE_GRAD` file count: `41` for GGUF and `41` for HF.
- Raw signature differences: `0`.
- Normalized signature differences: `0`.
- Combined signature hash for both paths:
  `9e5949826d4fbc75ea48e48959b9d875c0113b023c204e148dc11f261a33bb9c`.
- `torch.ops.vllm` counts in the pre-grad files matched for the measured ops:
  - `all_reduce`: `162`;
  - `moe_forward_shared`: `40`;
  - `rocm_unquantized_gemm`: `110`.
- No `gguf`, `qweight`, `qweight_type`, `weight_type`, or `data_container`
  strings appeared in either compiled pre-grad graph body.
- GGUF still had `_qwen35_effective_weight` references in the pre-grad graph
  files; HF had none. That helper path was already tested separately by the
  direct RMSNorm lookup experiment and did not improve throughput.

### Outcome

Reject visible graph input-signature or obvious GGUF materialization symbols as
the current MoE GGUF throughput explanation.

The remaining visible pre-grad body difference is the known norm helper
surface. A direct lookup variant removed that surface from the captured graph
and stayed in the same slow throughput class, so this is not the current whole
gap.

### Promote / Reject

Promote this as another static-source boundary: signatures, visible tensor
shapes, and measured custom-op counts are aligned between HF and corrected
GGUF.

Reject more static signature comparison unless a future patch changes graph
inputs.

### Next

Move to graph-visible runtime behavior. Static signatures and static generated
kernel bodies are now too similar to explain the remaining fixed-token gap.

## GGUF-169 - MoE TP4 Full-Ladder No-Sync Runner Timing

### Hypothesis

The short faithful branch request did not reproduce the HF-over-GGUF throughput
gap, so the no-sync `GPUModelRunner` timing hook needed to be tested across the
normal benchmark ladder. If the hook captured useful long-window timing, it
might reveal runner preprocessing, postprocessing, or model-forward enqueue
behavior that scales differently during fixed-token decode.

### Method

Launched the current corrected GGUF TP4 graph path on `.20` with:

- published v0.2 image;
- Qwen3.6 35B-A3B F16 GGUF;
- TP4, full-BAR/P2P-on path;
- `MAX_MODEL_LEN=131072`;
- broad graph capture;
- tuned GFX906 MoE config;
- no-sync `GPUModelRunner` timing overlay.

The normal benchmark ladder was then run:

1. eight `2000`-token warmups;
2. `c1_128` uncapped strict;
3. `c1_2000`;
4. `c1_10000`.

The diagnostic container was stopped after the run.

### Results

Benchmark ladder summary:

| case | completion tokens | client TPS | backend decode TPS | finish | strict gate valid |
| --- | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | `3292` | `71.028` | `71.668` | `stop` | `true` |
| `c1_2000` | `2000` | `71.852` | `72.936` | `length` | `false` |
| `c1_10000` | `10000` | `65.223` | `65.400` | `length` | `false` |

The result stayed in the same slow GGUF class and did not approach the FP16/HF
TP4 band.

The no-sync runner timing file contained about `500k` rows. Stage/shape
summary showed:

- FULL decode, one token:
  - `execute_model_total` average: about `1.49 ms`;
  - `model_forward` average: about `0.042 ms`;
  - `preprocess` average: about `1.10 ms`;
  - `postprocess` average: about `0.21 ms`.
- PIECEWISE startup/request shapes were much larger, but they do not explain
  steady fixed-token decode throughput.

### Outcome

Reject no-sync `GPUModelRunner` timing as an explanation for the fixed-token
MoE GGUF gap.

The hook is mostly measuring Python-side enqueue and bookkeeping, not completed
GPU graph replay time. It is useful for proving that preprocessing and
postprocessing are not dominant, but it cannot account for the actual
`c1_2000` / `c1_10000` wall and backend decode throughput gap.

### Promote / Reject

Reject as a performance fix and as a sufficient profiler.

Promote the full ladder as another reproducible slow-class confirmation for the
current GGUF path.

### Next

The next diagnostic must be graph-visible and lower level: HIP kernel
mix/timing, graph replay timing, or a reduced worker/replay reproducer that can
measure completed GPU work without perturbing output correctness.

## GGUF-170 - MoE TP4 CUDAGraph Replay Timing

### Hypothesis

The no-sync runner timing hook was measuring Python enqueue/bookkeeping, not
completed GPU replay. Timing `CUDAGraphWrapper` around
`entry.cudagraph.replay()` with CUDA events should expose completed replay cost
for the same corrected MoE GGUF TP4 path.

### Method

Created a disposable release-image source overlay for
`vllm/compilation/cuda_graph.py` that writes env-gated replay timings. The hook
records sampled CUDA-event elapsed time around graph replay and is not part of a
benchmark candidate because sampled calls synchronize.

Launched the current corrected GGUF TP4 graph path on `.20` with:

- published v0.2 image;
- Qwen3.6 35B-A3B F16 GGUF;
- TP4, full-BAR/P2P-on path;
- `MAX_MODEL_LEN=131072`;
- broad graph capture;
- tuned GFX906 MoE config;
- corrected load-time norm source patch;
- sampled `CUDAGraphWrapper` replay timing.

The normal benchmark ladder was run:

1. eight `2000`-token warmups;
2. `c1_128` uncapped strict;
3. `c1_2000`;
4. `c1_10000`.

The diagnostic container was stopped after the run.

### Results

Benchmark ladder summary:

| case | completion tokens | client TPS | backend decode TPS | finish | strict gate valid |
| --- | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | `3244` | `71.405` | `72.063` | `stop` | `true` |
| `c1_2000` | `2000` | `72.173` | `73.265` | `length` | `false` |
| `c1_10000` | `10000` | `65.506` | `65.685` | `length` | `false` |

The strict output was coherent benchmark-style text and passed the Qwen gate.
The fixed-token tiers stayed in the known slow GGUF MoE class and did not
approach the FP16/HF TP4 reproduction band.

Replay timing file:

- total sampled rows: `2008`;
- `FULL` mode rows: `1964`;
- `PIECEWISE` mode rows: `44`;
- overall sampled replay average: `13.342608 ms`;
- `FULL` single-token replay average: `13.542115 ms`, min `12.455204 ms`,
  max `16.630726 ms`;
- `PIECEWISE` `num_tokens=16` average: `1.663234 ms`;
- `PIECEWISE` `num_tokens=416` average: `7.766220 ms`.

### Outcome

Promote replay timing as a better low-level diagnostic than Python runner
timing. It measures completed graph replay instead of only enqueue/bookkeeping.

Reject the timed run as benchmark evidence. The hook synchronizes sampled calls
and therefore perturbs the serving path.

### Promote / Reject

Promote the `CUDAGraphWrapper` replay boundary for future GGUF/HF comparison.

Reject this run as a performance fix or release-candidate result.

### Next

Run the same replay-timing overlay against an aligned HF-weight MoE TP4 control
or a shorter comparable fixed-token control. Compare completed graph replay
timing against the GGUF timing above. If HF replay timing is similar, look above
or below replay boundaries: scheduler request cadence, logits/sampler,
memory movement, or HIP kernel mix. If HF replay timing is materially faster,
inspect graph replay contents and kernel mix at the completed-work boundary.

## GGUF-171 - MoE TP4 HF CUDAGraph Replay Timing Comparator

### Hypothesis

If the corrected MoE GGUF TP4 path is slow because its completed graph replay is
slower than HF, an aligned HF-weight TP4 run with the same replay-timing hook
should show materially lower full-graph replay time.

### Method

Launched an aligned HF-weight TP4 control on `.20` with:

- published v0.2 image;
- Qwen3.6 35B-A3B HF snapshot;
- TP4, full-BAR/P2P-on path;
- `MAX_MODEL_LEN=131072`;
- broad graph capture;
- tuned GFX906 MoE config;
- same sampled `CUDAGraphWrapper` replay timing overlay as `GGUF-170`.

The launch shape matched the broad GGUF graph profile: `PIECEWISE=51`,
`FULL=35`, and `302016` KV-cache tokens. The normal benchmark ladder was run:

1. eight `2000`-token warmups;
2. `c1_128` uncapped strict;
3. `c1_2000`;
4. `c1_10000`.

The diagnostic container was stopped after the run.

### Results

Benchmark ladder summary under the intrusive timing hook:

| case | completion tokens | client TPS | backend decode TPS | finish | strict gate valid |
| --- | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | `2951` | `71.070` | `71.784` | `stop` | `true` |
| `c1_2000` | `2000` | `71.579` | `72.653` | `length` | `false` |
| `c1_10000` | `10000` | `64.975` | `65.151` | `length` | `false` |

Replay timing file:

- total sampled rows: `1988`;
- `FULL` mode rows: `1944`;
- `PIECEWISE` mode rows: `44`;
- overall sampled replay average: `13.450060 ms`;
- `FULL` single-token replay average: `13.655169 ms`, min `12.533114 ms`,
  max `16.748783 ms`;
- `PIECEWISE` `num_tokens=16` average: `1.640725 ms`;
- `PIECEWISE` `num_tokens=416` average: `7.684730 ms`.

Aligned comparison to `GGUF-170`:

| path | strict backend TPS | c1_2000 backend TPS | c1_10000 backend TPS | FULL replay avg ms |
| --- | ---: | ---: | ---: | ---: |
| GGUF TP4 | `72.063` | `73.265` | `65.685` | `13.542115` |
| HF TP4 | `71.784` | `72.653` | `65.151` | `13.655169` |

The timing hook slows the HF control into the same observed TPS class as GGUF,
so these numbers must not be compared to the non-instrumented HF reproduction
band. The useful comparison is the near-identical replay timing under identical
instrumentation.

### Outcome

Reject completed CUDAGraph replay time as the main GGUF-versus-HF MoE TP4
performance explanation in this diagnostic shape.

The intrusive timing hook collapses both HF and GGUF into the same slow band and
their sampled full-graph replay times are effectively the same. This points away
from graph replay contents alone and toward request cadence, synchronization
side effects, logits/sampler path, metrics accounting, memory movement outside
sampled replay, or another boundary not captured by this wrapper.

### Promote / Reject

Promote the aligned replay comparison as a source-boundary closure.

Reject more unchanged `CUDAGraphWrapper` sampled timing ladders.

### Next

Do not repeat the same timing hook for another full ladder. The next useful
diagnostic should avoid synchronizing sampled decode calls, or should measure a
different boundary: HIP kernel mix/timing without per-token synchronization,
host-side request cadence, logits/sampler/postprocess cost, or a reduced replay
reproducer that can compare HF and GGUF without perturbing live serving.

## GGUF-172 - HF TP4 Lagged CUDAGraph Event Timing

### Hypothesis

The sampled replay timing hook in `GGUF-170` / `GGUF-171` synchronized each
sampled call and collapsed HF throughput into the GGUF band. A lagged event
queue might avoid that perturbation by recording CUDA event pairs around graph
replay and only writing elapsed times after the event pairs are at least `32`
samples old.

### Method

Created a disposable `vllm/compilation/cuda_graph.py` overlay that:

- records start/end CUDA events around `entry.cudagraph.replay()`;
- queues sampled event pairs without calling `synchronize()`;
- flushes only event pairs older than `32` sampled events;
- samples every `64` graph replay calls;
- writes elapsed times only when the queued end event reports complete.

Launched the aligned HF-weight TP4 control on `.20` with:

- published v0.2 image;
- Qwen3.6 35B-A3B HF snapshot;
- TP4, full-BAR/P2P-on path;
- `MAX_MODEL_LEN=131072`;
- broad graph capture;
- tuned GFX906 MoE config;
- lagged `CUDAGraphWrapper` replay timing overlay.

The server reached health with the expected graph profile:

- `PIECEWISE=51`;
- `FULL=35`;
- `302016` KV-cache tokens;
- actual graph pool about `2.13 GiB`.

The normal benchmark ladder was run:

1. eight `2000`-token warmups;
2. `c1_128` uncapped strict;
3. `c1_2000`;
4. `c1_10000`.

The diagnostic container was stopped after the run.

### Results

Benchmark ladder summary:

| case | completion tokens | client TPS | backend decode TPS | finish | strict gate valid |
| --- | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | `6952` | `68.206` | `68.482` | `stop` | `true` |
| `c1_2000` | `2000` | `72.252` | `73.352` | `length` | `false` |
| `c1_10000` | `10000` | `65.546` | `65.724` | `length` | `false` |

Timing file:

- total sampled rows: `2112`;
- `FULL` mode rows: `2068`;
- `PIECEWISE` mode rows: `44`;
- overall sampled replay average: `13.278601 ms`;
- `FULL` single-token replay average: `13.466812 ms`, min `12.329097 ms`,
  max `15.769899 ms`;
- `PIECEWISE` average: `4.432659 ms`.

### Outcome

Reject lagged CUDA-event replay timing as a usable live-serving profiler. It
still collapses the HF control into the same slow band as GGUF and changes
strict output length/shape enough that it cannot be treated as low-perturbation.

### Promote / Reject

Reject event-based replay timing in `CUDAGraphWrapper` for full ladder
comparisons, whether synchronized immediately or flushed after a lag.

Promote the negative result as profiler-boundary evidence: the next timing route
must avoid inserting CUDA events into every sampled replay call.

### Next

Use an external or lower-level profiler that does not alter graph replay source:

- HIP / rocprof kernel mix and timing, if a stable request-window capture can
  be built;
- host-side request cadence and metrics accounting from outside the replay
  loop;
- reduced worker replay reproducer that can be profiled without a live vLLM
  server;
- source comparison around graph/replay launch structure without injecting
  per-token CUDA events.

## GGUF-173 - MoE TP4 No Server-Side Reasoning Parser Diagnostic

### Hypothesis

The server-side `qwen3` reasoning parser might add enough request-loop friction
to explain part of the MoE GGUF fixed-token gap.

### Method

Launched the corrected load-time MoE GGUF TP4 path on `.20` with the same
published v0.2 image, same native `moe35b_tp4_fullbar_p2pon` profile, same
P2P-on path, same tuned GFX906 MoE config, same patch bundle, and same
begin-think proxy benchmark path as the current clean GGUF baseline.

The only intended launch-shape change was removing the server-side reasoning
parser from the vLLM command. The launched server reported
`reasoning_parser=''`. The tool-call parser remained enabled to avoid changing
more than one server-side parser variable.

The benchmark sequence used the normal warmup behavior, then stopped after the
strict and `c1_2000` checks because the `c1_2000` result stayed in the existing
slow GGUF band.

### Results

| case | completion tokens | client TPS | backend decode TPS | finish | strict gate valid |
| --- | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | `2837` | `71.479` | `72.230` | `stop` | `true` |
| `c1_2000` | `2000` | `71.940` | `73.023` | `length` | `false` |

The eight pre-measure warmups stabilized around `73.04` backend TPS after the
first cold warmup, matching the existing corrected-load-time GGUF MoE TP4 band.

### Outcome

Reject server-side reasoning parser removal as the missing performance lever.
It does not move MoE GGUF TP4 out of the known `~73` backend TPS `c1_2000`
band and is far below the FP16/HF TP4 reproduction path.

### Promote / Reject

Reject as a performance fix or release candidate.

Promote as launch-shape closure: the server-side reasoning parser is not the
source of the MoE GGUF throughput gap. Continue using the normal begin-think
proxy for Qwen strict-gate benchmarking, but do not spend more full-ladder time
on reasoning-parser toggles unless a future source change makes parser behavior
part of the serving hot path.

### Next

Return to source-level investigation below the launch-shape boundary:

- lower-perturbation HIP/kernel mix evidence;
- logits/sampler/postprocess and request-cadence boundaries outside graph
  replay event injection;
- reduced worker replay or offline graph reproduction that can time completed
  GPU work without collapsing the HF control;
- MoE GGUF materialization and residual/state paths that still differ from the
  HF-weight TP4 reproduction lane.

## GGUF-174 - MoE TP4 Expert Layout Audit

### Hypothesis

The MoE GGUF TP4 throughput gap might come from expert tensor
materialization/layout rather than request scheduling. The active GGUF overlay
loads 3-D GGUF expert tensors through the full-stack `FusedMoE.weight_loader`
path, while the stock HF loader iterates one expert at a time before calling
the same loader. A stride, contiguity, ROCm padding, or setup-kernel layout
difference could explain why corrected GGUF stays far below the HF TP4
reproduction band.

### Method

Created a disposable `.20` load-time audit overlay from the published v0.2
runtime image by copying only
`unquantized_fused_moe_method.py` and adding an env-gated metadata writer around
`process_weights_after_loading()`.

The audit recorded, for `w13_weight` and `w2_weight` at every MoE layer on each
TP worker:

- stage: before superclass post-load, after superclass post-load, after ROCm
  padding, and after kernel setup;
- tensor shape;
- stride;
- dtype;
- device;
- contiguity;
- storage offset;
- data pointer alignment modulo 16/64/128/256.

Ran two load-only diagnostics on `.20`, both P2P-on and TP4:

- GGUF path: corrected load-time Qwen3.6 35B-A3B F16 GGUF with the native
  `moe35b_tp4_fullbar_p2pon` profile and active GGUF patch bundle.
- HF path: Qwen3.6 35B-A3B HF snapshot with the same published image, same TP4
  host lane, same tuned GFX906 MoE config, and the same audit overlay.

No benchmark requests were sent. Both diagnostic containers were stopped after
the load-time audit rows were captured.

### Results

Both GGUF and HF produced `1280` audit rows:

- `40` MoE layers;
- `4` TP workers;
- `4` audit stages;
- `2` expert tensors per layer.

The summarized tensor layout was identical:

| stage | tensor | shape | stride | contiguous | row count |
| --- | --- | --- | --- | --- | ---: |
| before superclass post-load | `w13_weight` | `(256, 256, 2048)` | `(524288, 2048, 1)` | `true` | `160` |
| before superclass post-load | `w2_weight` | `(256, 2048, 128)` | `(262144, 128, 1)` | `true` | `160` |
| after superclass post-load | `w13_weight` | `(256, 256, 2048)` | `(524288, 2048, 1)` | `true` | `160` |
| after superclass post-load | `w2_weight` | `(256, 2048, 128)` | `(262144, 128, 1)` | `true` | `160` |
| after ROCm padding | `w13_weight` | `(256, 256, 2048)` | `(557056, 2176, 1)` | `false` | `160` |
| after ROCm padding | `w2_weight` | `(256, 2048, 128)` | `(262144, 128, 1)` | `true` | `160` |
| after kernel setup | `w13_weight` | `(256, 256, 2048)` | `(557056, 2176, 1)` | `false` | `160` |
| after kernel setup | `w2_weight` | `(256, 2048, 128)` | `(262144, 128, 1)` | `true` | `160` |

All rows were `torch.float16`; pointer alignment was clean at the sampled
16/64/128/256-byte boundaries. The only expected non-contiguity was the ROCm
padding applied to `w13_weight`, and it appeared in both GGUF and HF in the
same shape.

### Outcome

Reject MoE expert materialization layout, stride, ROCm padding, dtype,
contiguity, and basic alignment as the current MoE GGUF TP4 throughput
explanation.

The active GGUF full-stack expert load path is different from the stock HF
per-expert caller, but after `process_weights_after_loading()` both routes land
on the same expert tensor layout presented to the unquantized MoE kernel.

### Promote / Reject

Promote as source-boundary closure.

Reject another unchanged expert-layout or FusedMoE materialization audit unless
a future patch changes GGUF expert loading, MoE method selection, or ROCm
padding behavior.

### Next

Continue below this boundary:

- request cadence and queueing around fixed-token decode;
- HIP/kernel mix or memory movement that can be captured without replay-loop
  synchronization;
- logits/sampler/postprocess accounting outside the already-rejected parser
  toggle;
- a reduced replay reproducer that can time completed GPU work without
  collapsing the HF control.

## GGUF-175 - Artifact Timing Review After Kausik Repo Check

### Setup

Reviewed the external `Kausik-A/qwen3.6-27b-mi50-vllm` source bundle at local
head `61b273d` and rechecked existing `.20` benchmark artifacts before
launching another server.

The Kausik bundle was treated as source context only. Its useful patches are
compatibility bridges for Qwen3.5/3.6 GGUF loading:

- `qwen35` / `qwen35moe` architecture aliases;
- text-only M-RoPE suppression;
- `ssm_dt.bias` to `linear_attn.dt_bias` mapping;
- GGUF `conv1d.weight` 2-D to 3-D reshape;
- GGUF-aware tuple-shard loading in `MergedColumnParallelLinear`;
- MiniMax M2 GGUF expert-name aliases.

Its runtime target is a single MI50-class eager-mode deployment, so it is not a
drop-in reproduction path for the `.20` ROCm7.2 full-BAR/P2P-on, graph-mode,
TP8/TP4 release benchmark lanes.

### Existing Artifact Review

Read the current corrected MoE GGUF TP4 fixed-token summaries from
`moe35b_gguf_tp4_directnorm_broad_dot20_20260627T023204Z`.

The GGUF slowdown appears in both client wall timing and vLLM metrics:

| case | wall completion TPS | vLLM decode TPS | note |
| --- | ---: | ---: | --- |
| `c1_2000` | `71.934` | `73.015` | slow GGUF band |
| `c1_10000` | `65.278` | `65.454` | slow GGUF band |

Also reviewed the aligned HF CUDAGraph replay timing comparator artifacts. The
instrumented HF control fell into the same slow class:

| case | HF vLLM decode TPS under replay-timing hook | note |
| --- | ---: | --- |
| strict | `71.784` | strict-valid but contaminated by timing hook |
| `c1_2000` | `72.653` | below clean HF reproduction band |
| `c1_10000` | `65.151` | below clean HF reproduction band |

This differs from the clean `.20` HF reproduction path in
`test-reports/qwen36-gfx906-v02-reproduction-20260622/README.md`, which records
MoE TP4 strict `114.725`, `c1_2000` `116.429`, and `c1_10000` `109.531`.

### Outcome

Reject a pure client-side cadence or metrics-settle explanation for the current
GGUF MoE TP4 throughput gap. The low TPS is visible in both wall time and vLLM
decode metrics.

Reject CUDAGraph replay timing hooks as a trustworthy full-ladder comparator.
Even aligned HF weights collapse from the clean release band to the slow GGUF
band under the replay-timing instrumentation.

### Promote / Reject

Promote the Kausik repository as a compatibility checklist and source cross-check
only.

Reject importing the Kausik deployment shape or eager-mode launch as a
performance path for this goal.

Reject unchanged replay-event timing and Python runner timing as next
full-ladder diagnostics.

### Next

The next useful diagnostic must avoid perturbing replay timing. Prefer one of:

- offline/static comparison of compile artifacts from non-instrumented runs;
- HIP/kernel mix capture that does not inject per-token replay events;
- memory movement/accounting outside the replay loop;
- reduced completed-work replay reproducer that preserves clean HF performance;
- sampler/logits/postprocess accounting that does not alter graph replay.

## GGUF-176 - Qwen3.5 GGUF Quant-Method Mapping Audit

### Setup

Revisited the existing `.20` load-time quant-method audit after comparing the
external `Kausik-A/qwen3.6-27b-mi50-vllm` source bundle with the current
experimental overlay.

The audit artifact was:

`<validation-workspace>/runs/moe35b_gguf_tp4_quantmethod_audit_dot20_20260626T192907Z/quant_method_audit.tsv`

The run used the same published v0.2 image family, TP4, full-BAR/P2P-on lane,
GGUF load format, and Qwen3.5-MoE text overlay class used by the current MoE
GGUF experiments.

### Source Check

The current overlay already carries the important Kausik-style compatibility
pieces:

- Qwen3.5/Qwen3.5-MoE architecture aliases;
- GGUF linear-attention tensor-name mapping;
- tuple-shard handling for fused `in_proj_qkvz`;
- fused GDN mappings for `in_proj_qkvz` and `in_proj_ba`;
- fused MLP mapping for `gate_up_proj`;
- forced unquantized MoE expert method for F16 GGUF experts.

A fresh `GGUFConfig()` starts with an empty `packed_modules_mapping`, but vLLM's
`SupportsQuant` path updates the quant config from the model class during model
construction. The audit confirms the runtime handoff is happening for the hot
Qwen3.5 modules.

### Audit Results

The quant-method audit counted:

| method | count |
| --- | ---: |
| `UnquantizedLinearMethod` | `760` |
| `UnquantizedFusedMoEMethod(force)` | `160` |
| `None` for attention modules | `120` |

Sampled hot prefixes selected `UnquantizedLinearMethod`:

- `model.layers.0.linear_attn.in_proj_qkvz`
- `model.layers.0.linear_attn.in_proj_ba`
- `model.layers.0.linear_attn.out_proj`
- `model.layers.0.mlp.shared_expert.gate_up_proj`
- `model.layers.0.mlp.shared_expert.down_proj`
- `model.layers.3.self_attn.qkv_proj`
- `model.layers.3.self_attn.o_proj`

MoE expert modules selected forced `UnquantizedFusedMoEMethod`.

### Outcome

Reject missing fused-module GGUF skip mapping as the current MoE GGUF TP4
throughput explanation. The instantiated model is already using unquantized
linear methods for the fused Qwen3.5 GDN and shared-expert paths, and the MoE
experts are already forced through the unquantized FusedMoE method.

This also reduces the likelihood that importing more of the external Kausik
bundle will close the current speed gap. The useful compatibility ideas are
already represented in the current overlay or are irrelevant to the `.20`
graph-mode release lane.

### Promote / Reject

Promote this as a source-boundary closure for GGUF method selection.

Reject another unchanged quant-method or packed-module mapping audit unless a
future patch changes the model class, `GGUFConfig`, `SupportsQuant`, or fused
projection naming.

### Next

Continue below the method-selection boundary. The remaining plausible buckets
are:

- HIP/kernel mix and memory movement without per-token replay synchronization;
- request cadence or scheduler behavior that differs between HF and GGUF despite
  matching graph bodies;
- logits/sampler/postprocess accounting that avoids replay perturbation;
- reduced completed-work reproducer that preserves clean HF performance.

## GGUF-177 - MoE TP4 btok1024/seq1 Launch-Shape Test

### Setup

Launched the corrected direct-norm MoE GGUF TP4 path on `.20` with the same
published v0.2 image family, TP4, full-BAR/P2P-on lane,
`MAX_MODEL_LEN=131072`, release MoE tuned config, GGUF load format, and
model-specific Qwen3.5-MoE overlay used by the current successful GGUF strict
path.

This run changed only the launch shape to match the clean HF comparator more
closely:

- `--max-num-seqs 1`
- `--max-num-batched-tokens 1024`
- CUDAGraph capture sizes reduced to `[1, 2]`

Run artifact:

`<validation-workspace>/runs/moe35b_gguf_tp4_directnorm_btok1024_seq1_dot20_20260627T054620Z`

The normal benchmark sequence was used:

- eight 2000-token pre-measure warmups;
- `c1_128` uncapped strict through the begin-think proxy;
- `c1_2000`;
- `c1_10000`.

### Results

Warmups stabilized after the first compile-heavy request at about `73.48`
backend decode TPS.

| case | completion tokens | client TPS | backend decode TPS | finish | strict gate valid |
| --- | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | `3460` | `71.373` | `72.020` | `stop` | `true` |
| `c1_2000` | `2000` | `72.342` | `73.495` | `length` | `false` |
| `c1_10000` | `10000` | `65.652` | `65.840` | `length` | `false` |

### Outcome

Reject narrowed launch shape as the missing MoE GGUF TP4 performance lever. The
run stayed in the same known corrected-GGUF band despite using the smaller
`max_num_batched_tokens=1024`, `max_num_seqs=1`, and `[1, 2]` CUDAGraph capture
shape that distinguished the clean HF comparator.

This result also rejects repeating unchanged graph-size toggles as a productive
next step.

### Promote / Reject

Promote the run as a clean launch-boundary closure.

Reject launch-shape mismatch between the GGUF broad run and HF comparator as the
current explanation for the GGUF-versus-HF MoE TP4 gap.

### Next

Continue below launch configuration and method selection. The next useful work
should collect lower-level evidence without perturbing the HF control:

- HIP/kernel mix or memory movement profiling;
- request cadence outside sampled replay-event timing;
- reduced completed-work replay reproducer that preserves clean HF performance;
- static graph/artifact comparison from non-instrumented GGUF and HF runs.

## GGUF-178 - MoE TP4 Debug-Dump Static Graph Comparison

### Setup

Compared existing `.20` debug-dump graph artifacts without launching a new
model server.

GGUF artifact:

`<validation-workspace>/runs/moe35b_gguf_tp4_loadtime_once_broad_graph_debugentry_dot20_20260627T012347Z/debug_dump/rank_0_dp_0`

HF artifact:

`<validation-workspace>/runs/moe35b_hf_tp4_broad_graph_debugentry_dot20_20260627T013947Z/debug_dump/rank_0_dp_0`

Comparison artifact:

`<validation-workspace>/runs/gguf178_moe35b_tp4_debugdump_static_compare_20260627T060110Z`

Both source runs used the published v0.2 image family, TP4,
full-BAR/P2P-on, `MAX_MODEL_LEN=131072`, the tuned GFX906 MoE config,
`NCCL_ALGO=Tree`, `NCCL_PROTO=LL`, `NCCL_P2P_DISABLE=0`, graph mode, and
debug-dump compilation. The GGUF side used `load_format=gguf` and
`quantization=gguf`; the HF side used `load_format=auto` and
`quantization=None`.

Both debug comparators also used explicit prefix-cache/Mamba alignment:

- `--enable-prefix-caching`
- `--mamba-cache-mode align`
- `--language-model-only`

### Results

The debug-dump shape was close:

| item | GGUF | HF |
| --- | ---: | ---: |
| files | `61` | `61` |
| bytes | `6534768` | `6506847` |
| after-split lines | `9236` | `9146` |
| before-split lines | `8630` | `8540` |
| generated kernel files | `14` | `14` |

The top vLLM op counts matched exactly:

| op | count |
| --- | ---: |
| `torch.ops.vllm.all_reduce` | `1134` |
| `torch.ops.vllm.rocm_unquantized_gemm` | `770` |
| `torch.ops.vllm.gdn_attention_core` | `360` |
| `torch.ops.vllm.moe_forward_shared` | `280` |
| `torch.ops.vllm.unified_kv_cache_update` | `120` |
| `torch.ops.vllm.unified_attention_with_output` | `120` |

The ATen op counts also matched exactly:

| op | count |
| --- | ---: |
| `torch.ops.aten.add` | `96` |
| `torch.ops.aten.mul` | `64` |
| `torch.ops.aten.rsqrt` | `28` |
| `torch.ops.aten.pow` | `28` |
| `torch.ops.aten.mean` | `28` |
| `torch.ops.aten.reshape` | `20` |
| `torch.ops.aten.copy_` | `10` |
| `torch.ops.aten.neg` | `6` |
| `torch.ops.aten.exp` | `6` |
| `torch.ops.aten.div` | `6` |
| `torch.ops.aten.clone` | `6` |
| `torch.ops.aten.sigmoid` | `2` |

Key graph pattern counts matched exactly:

| pattern | count |
| --- | ---: |
| `cos_sin_cache` | `202` |
| `l_positions_` | `136` |
| `unified_attention` | `120` |
| `gdn_attention_core` | `360` |
| `moe_forward` | `1231` |
| `all_reduce` | `1579` |
| `cat` | `701` |
| `clone` | `584` |
| `contiguous` | `940` |
| `reshape` | `2773` |
| `view` | `3573` |
| `mm` | `2263` |

After stripping comments, UUID/hash-like metadata, and normalizing module-name
prefix noise, every generated executable kernel body compared equal:

| generated kernel | normalized executable diff lines |
| --- | ---: |
| `__compiled_fn_1.kernel_0.py` | `0` |
| `__compiled_fn_1.kernel_1.py` | `0` |
| `__compiled_fn_1.kernel_2.py` | `0` |
| `__compiled_fn_1.kernel_3.py` | `0` |
| `__compiled_fn_1.kernel_4.py` | `0` |
| `__compiled_fn_1.kernel_5.py` | `0` |
| `__compiled_fn_1.kernel_6.py` | `0` |
| `__compiled_fn_1.kernel_7.py` | `0` |
| `__compiled_fn_1.kernel_8.py` | `0` |
| `__compiled_fn_1.kernel_9.py` | `0` |
| `__compiled_fn_1.kernel_10.py` | `0` |
| `__compiled_fn_1.kernel_11.py` | `0` |
| `__compiled_fn_1.kernel_12.py` | `0` |
| `__compiled_fn_1.kernel_13.py` | `0` |

A repeated GGUF-only IR difference remains in the pre-executable debug files:
the GGUF dump contains dead-looking `split([512, 512, 1024], dim=-1)` patterns
around the GatedDeltaNet projection path that do not appear in the HF dump.
This difference did not survive into the normalized generated executable kernel
bodies in this comparison.

### Outcome

Promote the static debug-dump comparison as evidence that these broad GGUF and
HF debug comparators compile to the same generated executable kernel bodies and
the same measured vLLM/ATen graph op mix.

Reject generated Inductor executable body differences, top-level vLLM op-count
mix, and ATen op-count mix as the current explanation for the MoE GGUF TP4
throughput gap in this comparator pair.

Do not over-promote the result. The HF debug comparator is not proven to be the
same as the clean fast v0.2.1 reproduction path. It used debug-dump compilation
and explicit prefix-cache/Mamba alignment, while the published `.20`
reproduction evidence is still the report band of strict `114.725`, `c1_2000`
`116.429`, and `c1_10000` `109.531` backend TPS.

### Promote / Reject

Promote static graph artifact comparison as a low-perturbation diagnostic.

Reject another unchanged generated-kernel body diff between these same
debug-dump artifacts.

Reject the GGUF-only dead-looking split in debug IR as a sufficient explanation
unless a future runtime trace proves it executes or changes generated kernels.

### Next

Find the raw launch/config evidence for the actual v0.2.1 MoE TP4 reproduction
run, or create a reduced comparator that preserves the clean HF reproduction
band before attributing the GGUF gap outside generated kernels. If raw evidence
is unavailable, the next candidate is a GGUF launch that matches the current
`deploy.sh` MoE TP4 defaults more tightly and then runs the normal warmups ->
`c1_128` uncapped strict -> `c1_2000` -> `c1_10000` sequence only if the launch
materially changes the hypothesis.

## GGUF-179 - MoE TP4 Release Fastpath Overlay Forced Onto GGUF

### Setup

- Host label: `.20`
- Run directory:
  `<validation-workspace>/runs/moe35b_gguf_tp4_directnorm_fastpath_force_dot20_20260627T061255Z`
- Benchmark directory:
  `<validation-workspace>/runs/moe35b_gguf_tp4_directnorm_fastpath_force_dot20_20260627T061255Z/v02-profile-runs/moe35b_tp4_fastpath_force_20260627T062050Z`
- Container image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`
- Model artifact: `Qwen3.6-35B-A3B-F16-GGUF`
- Profile intent: `moe35b_tp4_fullbar_p2pon`
- Tensor parallelism: TP4
- Platform lane: full-BAR/P2P-on
- `MAX_MODEL_LEN=131072`
- Normal benchmark order: eight 2000-token warmups, `c1_128` uncapped strict,
  `c1_2000`, then `c1_10000`

This run corrected an earlier comparison gap. The current v0.2.1 deployment
path generates and mounts a Qwen c1 top-k 8 MoE fastpath overlay for the TP4
MoE profile, while earlier GGUF runs did not actually have that overlay active.
For this diagnostic, the release `runtime/patches/fused_moe.py` fastpath was
mounted into the official release image alongside the corrected Qwen3.5-MoE
GGUF source route.

### Evidence

The server reached API health. Startup logs showed:

- `qwen c1 topk8 MoE fastpath overlay loaded`
- `qwen c1 topk8 MoE fastpath active` for `num_tokens=1` on all four TP
  workers
- `204` larger-token fastpath rejections during graph capture/prefill shape
  checks
- `4` active fastpath log entries for the single-token decode shape

The fastpath shape logs confirmed the expert layout is compatible with the
single-token guard:

- hidden shape: `(1, 2048)` for active decode
- `w1_shape=(256, 256, 2048)`
- `w2_shape=(256, 2048, 128)`
- top-k: `8`
- dtype: `torch.float16`

Larger capture shapes such as `num_tokens=56`, `48`, `40`, `32`, `24`, `16`,
`8`, `4`, and `2` were rejected by the fastpath shape guard. That is expected
for the current TP4 fastpath shape, but it means the overlay does not accelerate
all graph-capture and prefill shapes.

### Results

Warmups stabilized in the low-80s backend TPS after graph capture. The promoted
benchmark summary was:

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | `3322` | `81.731` | `82.571` | `40.646` | stop | true |
| `c1_2000` | `2000` | `82.840` | `84.284` | `24.143` | length | false |
| `c1_10000` | `10000` | `74.174` | `74.402` | `134.819` | length | false |

The strict request completed normally with `finish_reason=stop` and
`qwen_gate_valid=true`.

### Outcome

Promote the release TP4 MoE fastpath mount as a real GGUF performance lever.
It moves the corrected MoE GGUF TP4 path above the prior slow band and proves
that the missing fastpath overlay was a meaningful comparison error in earlier
GGUF runs.

Reject the single-token top-k 8 fastpath as sufficient for release-parity
performance. The best fixed-token result from this run, `c1_2000` `84.284`
backend TPS, remains far below the clean `.20` HF/v0.2.1 reproduction band
of strict `114.725`, `c1_2000` `116.429`, and `c1_10000` `109.531` backend
TPS.

### Promote / Reject

Promote:

- corrected comparison requirement: future MoE GGUF TP4 tests must include the
  same release MoE fastpath overlay unless the experiment is explicitly testing
  its absence;
- evidence that the GGUF expert layout can enter the single-token release
  fastpath;
- evidence that MoE GGUF strict correctness can pass under this fastpath-mounted
  path.

Reject:

- "fastpath missing" as the complete MoE GGUF performance explanation;
- unchanged runs that omit the release TP4 MoE fastpath unless they are
  intentional negative controls;
- another full ladder focused only on launch flags while the remaining gap is
  still roughly 25-35 backend TPS.

### Next

Compare the release fastpath implementation against the GGUF execution path
below the Python launch layer. The next useful source target is not another
generic deployment toggle; it is why a path with matching expert layout and
single-token fastpath activation still decodes substantially slower than the
HF-weight TP4 reproduction path. Candidate boundaries are HIP kernel mix,
memory movement around expert routing, scheduler/token cadence after fastpath
activation, and any remaining GGUF-specific work outside the single-token
FusedMoE fastpath.

## GGUF-180 - MoE TP4 Generalized TP8 Fastpath Overlay Forced Onto GGUF

### Setup

- Host label: `.20`
- Run directory:
  `<validation-workspace>/runs/moe35b_gguf_tp4_directnorm_fastpath_tp8multi_dot20_20260627T063217Z`
- Benchmark directory:
  `<validation-workspace>/runs/moe35b_gguf_tp4_directnorm_fastpath_tp8multi_dot20_20260627T063217Z/v02-profile-runs/moe35b_tp4_tp8multi_fastpath_20260627T063838Z`
- Container image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`
- Model artifact: `Qwen3.6-35B-A3B-F16-GGUF`
- Profile intent: `moe35b_tp4_fullbar_p2pon`
- Tensor parallelism: TP4
- Platform lane: full-BAR/P2P-on
- `MAX_MODEL_LEN=131072`
- Normal benchmark order: eight 2000-token warmups, `c1_128` uncapped strict,
  `c1_2000`, then `c1_10000`

This diagnostic tested whether the TP4 MoE GGUF path was limited by the current
release fastpath's single-token guard. The TP8 overlay
`files/gfx906_runtime/moe_tp8_overlays/fused_moe_tp8.py` supports
`1 <= num_tokens <= 4`, so it was mounted into the same TP4 GGUF launch used
for GGUF-179.

An initial launch on port `8080` was rejected as a false start because health
checks could hit an existing host service. The promoted diagnostic used port
`8091`.

### Evidence

The server reached API health and logs confirmed that the generalized fastpath
entered more decode shapes than the release TP4 overlay:

- active fastpath log entries: `12`
- rejected fastpath log entries: `196`
- active token counts observed: `4`, `2`, and `1`
- rejected token counts included larger graph-capture and prefill shapes down
  to `8`

Representative active shape:

- hidden shape: `(4, 2048)`, `(2, 2048)`, or `(1, 2048)`
- `w1_shape=(256, 256, 2048)`
- `w2_shape=(256, 2048, 128)`
- top-k: `8`
- dtype: `torch.float16`

Warmup output had visible answer text after the begin-think proxy split, unlike
some earlier GGUF-179 warmups that had zero visible answer characters. That
improved output shape did not translate into release-band throughput.

### Results

Warmups stabilized around `84.2` backend TPS for 2000-token capped requests.
The promoted benchmark summary was:

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | `2985` | `82.015` | `82.958` | `36.396` | stop | true |
| `c1_2000` | `2000` | `82.789` | `84.230` | `24.158` | length | false |
| `c1_10000` | `10000` | `74.114` | `74.342` | `134.927` | length | false |

The strict request completed normally with `finish_reason=stop` and
`qwen_gate_valid=true`.

### Outcome

Promote the generalized TP8 fastpath overlay as evidence that GGUF TP4 can
enter the top-k 8 fastpath for token groups `4`, `2`, and `1`. Also promote it
as a useful output-shape diagnostic because the warmups produced visible answer
text under the same begin-think proxy path.

Reject the generalized token-count guard as a throughput fix. The result is
effectively the same throughput class as GGUF-179: strict low-80s backend TPS,
`c1_2000` about `84` backend TPS, and `c1_10000` about `74` backend TPS. This
remains far below the clean `.20` HF/v0.2.1 TP4 reproduction band of strict
`114.725`, `c1_2000` `116.429`, and `c1_10000` `109.531` backend TPS.

### Promote / Reject

Promote:

- future fastpath diagnostics should explicitly record which token-count shapes
  enter the overlay;
- generalized `1..4` token support is compatible with the corrected GGUF expert
  layout;
- strict correctness can pass with the generalized overlay mounted into the TP4
  GGUF route.

Reject:

- another full ladder that only changes the fastpath token-count guard;
- treating visible warmup answer text as a throughput proxy;
- "single-token-only fastpath" as the complete remaining GGUF-versus-HF MoE
  TP4 gap.

### Next

Move below the Python fastpath guard. The remaining gap is not explained by
model aliasing, expert layout, method dispatch, launch-shape narrowing, static
generated kernels, server-side reasoning parser toggles, missing single-token
fastpath activation, or the `1..4` fastpath token-count guard. The next useful
source-level work is HIP kernel mix/timing without replay-loop synchronization,
memory movement around routing and logits/postprocess, or a reduced completed
work reproducer that preserves the clean HF TP4 fast path while comparing GGUF
and HF under the same source-level instrumentation.

## GGUF-181 - MoE TP4 Exact HF Scheduler Shape Plus Release Fastpath

### Setup

- Host label: `.20`
- Run directory:
  `<validation-workspace>/runs/moe35b_gguf_tp4_fastpath_exact_hfshape_dot20_20260627T065134Z`
- Benchmark directory:
  `<validation-workspace>/runs/moe35b_gguf_tp4_fastpath_exact_hfshape_dot20_20260627T065134Z/v02-profile-runs/moe35b_tp4_fastpath_exact_hfshape_20260627T065533Z`
- Container image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`
- Model artifact: `Qwen3.6-35B-A3B-F16-GGUF`
- Profile intent: `moe35b_tp4_fullbar_p2pon`
- Tensor parallelism: TP4
- Platform lane: full-BAR/P2P-on
- `MAX_MODEL_LEN=131072`
- Normal benchmark order: eight 2000-token warmups, `c1_128` uncapped strict,
  `c1_2000`, then `c1_10000`

This diagnostic combined two previously separate conditions:

- the current release TP4 MoE fastpath overlay from GGUF-179;
- the clean HF TP4 control scheduler shape:
  `--max-num-seqs 1 --max-num-batched-tokens 1024 --enable-prefix-caching
  --mamba-cache-mode align`.

The purpose was to test whether the release fastpath only failed to promote
because the GGUF fastpath-mounted runs had broad graph capture sizes, while the
clean HF control captured only `[1, 2]`.

### Evidence

The launch matched the clean HF scheduler/capture shape:

- `max_num_batched_tokens=1024`
- `max_num_seqs=1`
- `enable_prefix_caching=True`
- `mamba_cache_mode=align`
- CUDAGraph capture sizes: `[1, 2]`
- `load_format=gguf`
- `quantization=gguf`
- architecture: `Qwen3_5MoeForCausalLM`

The release fastpath overlay was mounted and loaded. Logs recorded:

- active fastpath log entries: `4`
- rejected fastpath log entries: `16`
- active token count observed: `1`
- rejected token counts included `1024`, `416`, `15`, and `2`

This confirms that the exact HF scheduler shape was active, but the release TP4
fastpath still only promoted the single-token shape in this run.

### Results

Warmups stabilized around `84.2` backend TPS after the first prefill-heavy
request. The promoted benchmark summary was:

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | `3256` | `81.639` | `82.543` | `39.883` | stop | true |
| `c1_2000` | `2000` | `82.676` | `84.195` | `24.191` | length | false |
| `c1_10000` | `10000` | `74.073` | `74.314` | `135.002` | length | false |

The strict request completed normally with `finish_reason=stop` and
`qwen_gate_valid=true`.

### Outcome

Reject the combined exact-HF scheduler shape plus release-fastpath hypothesis.
The result is effectively identical to the prior fastpath-mounted GGUF TP4 band:
strict low-80s backend TPS, `c1_2000` about `84` backend TPS, and `c1_10000`
about `74` backend TPS. It remains far below the clean `.20` HF/v0.2.1 TP4
reproduction band of strict `114.725`, `c1_2000` `116.429`, and `c1_10000`
`109.531` backend TPS.

Also record a correctness/output caveat: the capped warmups under this exact
HF shape produced zero visible answer characters, while the strict request and
the capped `c1_10000` request produced visible answer text after the
begin-think proxy split. This is not a throughput explanation, but it is useful
evidence that scheduler shape can affect the visible-output profile even when
decode TPS stays in the same band.

### Promote / Reject

Promote:

- future comparisons should keep exact scheduler/capture shape and fastpath
  state in the same run record;
- exact HF control shape is now closed as a missing fastpath-mounted GGUF
  condition;
- the remaining gap is below the broad launch-shape boundary.

Reject:

- repeating exact-HF scheduler shape runs without a source-path change;
- treating broad graph capture as the current MoE GGUF-versus-HF TP4 gap;
- treating the release fastpath mount plus `[1, 2]` CUDAGraph capture as enough
  to reach the HF/v0.2.1 band.

### Next

Continue below launch shape and Python fastpath selection. The next useful work
is a lower-perturbation HIP/kernel or memory-movement comparison that preserves
the clean HF TP4 band, or a reduced completed-work reproducer that can compare
GGUF and HF without per-token replay synchronization. Static graph bodies,
method selection, expert tensor layout, exact scheduler shape, and fastpath
activation have all been checked and rejected as complete explanations.

## GGUF-182 - Clean HF TP4 Artifact Boundary Review

### Setup

- Host label: `.20`
- Scope: read-only artifact search and launch-log comparison
- Repo checkout inspected on `.20`:
  `<validation-workspace>/repo/localaiservers`
- Relevant public report:
  `test-reports/qwen36-gfx906-v02-reproduction-20260622/README.md`
- Relevant local diagnostic runs:
  - `moe35b_hf_tp4_release_comparator_dot20_20260626T131844Z`
  - `moe35b_hf_tp4_release_comparator_entrypoint_dot20_20260626T133559Z`
  - `moe35b_hf_tp4_broad_graph_debugentry_dot20_20260627T013947Z`
  - `moe35b_hf_tp4_cgreplay_timing_dot20_20260627T040136Z`
  - `moe35b_hf_tp4_cgreplay_lag_timing_dot20_20260627T042121Z`

This review checked whether the clean high-band HF TP4 reproduction artifacts
were present near the GGUF diagnostic directories, and whether those artifacts
could be used as a non-instrumented comparator for the current GGUF MoE TP4
gap.

### Evidence

The current local public reproduction report records the `.20` HF-weight TP4
band as:

| profile | host | strict backend TPS | c1_2000 backend TPS | c1_10000 backend TPS |
| --- | --- | ---: | ---: | ---: |
| `moe35b_tp4_fullbar_p2pon` | `.20` | `114.725` | `116.429` | `109.531` |

The nearby HF diagnostic directories do not provide the original clean
benchmark-ladder summaries for that high-band run:

- `moe35b_hf_tp4_release_comparator_dot20_20260626T131844Z` used the HF model,
  broad graph capture, `max_num_batched_tokens=2048`,
  `enable_prefix_caching=False`, `reasoning_parser=''`, and CUDAGraph capture
  sizes through `512`. It is a launch/control diagnostic, not the exact public
  reproduction report artifact.
- `moe35b_hf_tp4_release_comparator_entrypoint_dot20_20260626T133559Z` matched
  the later exact scheduler shape more closely
  (`--max-num-seqs 1 --max-num-batched-tokens 1024 --enable-prefix-caching
  --mamba-cache-mode align --language-model-only`), but it failed before a
  benchmark ladder because the hybrid KV layout ambiguity raised:
  `Fail to determine whether the layout is (2, num_blocks, ...) or
  (num_blocks, 2, ...)`.
- The later broad/debug and replay-timing HF directories are known diagnostic
  comparators. They either include debug-dump/profiler instrumentation or event
  timing hooks that collapse the HF lane into the same low-70s band as GGUF,
  so they cannot stand in for the clean high-band reproduction path.

The actual high-band values are therefore documented in the public
reproduction report, but the raw local run directory containing those exact
`114.725` / `116.429` / `109.531` summaries was not found during this targeted
search.

### Outcome

Promote this as an artifact-boundary finding: the high HF TP4 control is real
as a public reproduction-report value, but the local diagnostic directories
available under the GGUF workspace are not clean enough to compare against the
current GGUF path as raw source artifacts.

Reject reusing the existing HF debug/profiler directories as proof that static
graph parity or replay parity explains the whole gap. They are useful negative
evidence, but they do not preserve the clean `.20` HF TP4 performance band.

### Promote / Reject

Promote:

- any next HF-vs-GGUF comparator must regenerate a clean HF control from the
  public v0.2.1 reproduction path, preserving the normal benchmark order:
  eight warmups, uncapped `c1_128` strict, `c1_2000`, and `c1_10000`;
- the regenerated HF control should save launch command, container log,
  benchmark summaries, and effective vLLM engine configuration before a GGUF
  run is compared against it;
- a GGUF comparator should only be launched after one new source-level change
  is identified, not as another unchanged fastpath/scheduler repeat.

Reject:

- treating the HF debug-dump or replay-timing comparators as the clean
  high-band control;
- another unchanged exact-scheduler GGUF ladder;
- another broad artifact grep over old ROCm/RCCL source trees.

### Next

Regenerate a clean HF TP4 control from the public v0.2.1 deploy and benchmark
path if we need a fresh comparator artifact. Then run a GGUF comparator only
with a new source-level hypothesis below the launch/fastpath layer: HIP kernel
mix without replay-loop event injection, memory movement around routing or
postprocess, or a reduced completed-work reproducer that can keep the clean HF
band intact.

## GGUF-183 - Clean HF TP4 Control Regenerated From v0.2.1 Path

### Setup

- Host label: `.20`
- Source path:
  `<validation-workspace>/repo/localaiservers/qwen36-gfx906`
- Repo commit on `.20`: `0c0ee21`
- Profile: `moe35b_tp4_fullbar_p2pon`
- Runtime image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`
- Docker manifest digest verified by deploy:
  `sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e`
- Run directory:
  `<validation-workspace>/runs/hf_moe35b_tp4_clean_repro_20260627T074206Z/v02-profile-runs/moe35b_tp4_fullbar_p2pon_20260627T074206Z`

### Method

Followed the public v0.2.1 reproduction path on `.20`:

1. Ran the read-only host preflight.
2. Deployed the published image with:
   - `QWEN36_PROFILE=moe35b_tp4_fullbar_p2pon`
   - `DOCKER_ISOLATED_DAEMON_ENABLED=0`
   - `HF_HUB_DISABLE_XET=1`
   - `USE_PREBUILT_IMAGE=1`
   - `PREBUILT_IMAGE_PULL=1`
   - `AUTO_STAGE_MODEL=1`
3. Ran `./smoke-test.sh`.
4. Ran `./run_v02_profile_benchmark.sh`, preserving the normal release order:
   eight 2000-token pre-measure warmups, uncapped `c1_128` strict,
   `c1_2000`, and `c1_10000` through the bundled begin-think proxy.

The host preflight passed with eight visible gfx906 agents, 32 GiB BAR on all
eight GPUs, KFD topology visible, and VBIOS `113-D1631700-111`.

### Results

Deploy-time source/shape facts:

- auto-selected `HIP_VISIBLE_DEVICES=0,1,2,3`;
- TP size `4`;
- P2P on (`NCCL_P2P_DISABLE=0`);
- dtype `torch.float16`;
- `MAX_MODEL_LEN=131072`;
- asynchronous scheduling enabled;
- graph mode enabled (`enforce_eager=False`);
- CUDAGraph capture included `PIECEWISE=51` and `FULL=35`;
- Qwen C1 top-k 8 MoE fastpath loaded and became active;
- model loading took about `45.75` seconds and `17.32 GiB` per TP rank;
- first warmup absorbed first-request compile/capture overhead, then later
  warmups settled at about `116.6` backend decode TPS.

Benchmark summary:

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | `5183` | `113.369` | `114.187` | `45.718` | `stop` | `True` |
| `c1_2000` | `2000` | `114.455` | `116.623` | `17.474` | `length` | `False` |
| `c1_10000` | `10000` | `109.357` | `109.748` | `91.444` | `length` | `False` |

The `c1_2000` and `c1_10000` fixed-token values slightly exceed the prior
public `.20` reproduction report values (`116.429` and `109.531`). The strict
value is still in the expected high HF TP4 band.

The benchmark container was stopped and removed after the run. A follow-up
check showed no vLLM worker processes and all GPUs back to `0%` VRAM/use.

### Outcome

Promote this run as the clean high-band HF TP4 control artifact for the current
GGUF source investigation.

Reject any explanation that depends on the clean HF TP4 path being inherently
low-70s or low-80s. The public v0.2.1 path still reproduces the high band when
run without replay-event timing hooks, debug instrumentation, or altered GGUF
source overlays.

### Promote / Reject

Promote:

- use this run directory as the high-band HF control for the next GGUF
  source-level comparator;
- preserve the release benchmark order and begin-think proxy for Qwen strict
  runs;
- require any GGUF improvement claim to compare against this clean control,
  not against instrumented HF replay/debug runs.

Reject:

- repeating clean HF TP4 control immediately without a source change;
- replay-loop event timing as a full-ladder comparator;
- launch-shape, fastpath activation, and broad graph-capture explanations as
  sufficient for the remaining MoE GGUF gap.

### Next

Run the next GGUF TP4 comparator only after a new source-level change below the
launch/fastpath layer. The most useful next hypotheses are:

- a low-perturbation HIP/kernel mix capture that does not instrument replay
  per token;
- memory movement around routing or postprocess;
- a reduced completed-work reproducer that preserves the clean HF band while
  swapping HF and GGUF-loaded weights through the same source path.

## GGUF-184 - Force F16 GGUF Linear Layers To UnquantizedLinearMethod

Date: 2026-06-27 UTC

Host: `.20`

Run directory:
`<validation-workspace>/runs/moe35b_gguf_tp4_force_unquant_linear_dot20_20260627T075850Z`

Benchmark directory:
`<validation-workspace>/runs/moe35b_gguf_tp4_force_unquant_linear_dot20_20260627T075850Z/v02-profile-runs/moe35b_tp4_fullbar_p2pon_force_unquant_linear_20260627T080430Z`

### Hypothesis

The remaining MoE GGUF TP4 gap might be caused by residual F16 GGUF
`LinearBase` wrapper dispatch outside the unquantized FusedMoE path. Force all
F16 GGUF linear layers to use `UnquantizedLinearMethod`, while leaving the
forced unquantized FusedMoE method active.

### Change

Created a scratch source copy under:

`<validation-workspace>/experimental-patches/qwen35moe-force-unquant-linear-20260627`

Modified `vllm/model_executor/layers/quantization/gguf.py` in the scratch copy
only. The launch set:

- `VLLM_GFX906_GGUF_FORCE_UNQUANT_MOE=1`
- `VLLM_GFX906_GGUF_FORCE_UNQUANT_LINEAR=1`

The run used the published v0.2.x ROCm7.2 image, the GGUF F16
Qwen3.6-35B-A3B model, profile `moe35b_tp4_fullbar_p2pon`, TP4, P2P-on, FP16,
`MAX_MODEL_LEN=131072`, graph mode, async scheduling, and the normal benchmark
ladder: warmups, uncapped strict, `c1_2000`, and `c1_10000`.

Dispatch evidence from the container log:

- forced unquantized Linear selections: `760`
- forced unquantized FusedMoE selections: `160`
- model load time: about `101.08` seconds
- CUDAGraph capture completed successfully

### Results

Warmups settled in the same post-fastpath GGUF band as the prior runs, around
`84` backend decode TPS after the first-request overhead.

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | `3477` | `81.229` | `82.065` | `42.805` | `stop` | `True` |
| `c1_2000` | `2000` | `82.439` | `83.962` | `24.260` | `length` | `False` |
| `c1_10000` | `10000` | `73.928` | `74.167` | `135.268` | `length` | `False` |

The scratch container was stopped and removed after the run. No vLLM or Ray
worker processes remained on `.20`.

### Outcome

Reject forced unquantized `LinearBase` dispatch as the MoE GGUF TP4 performance
fix. It does not improve the prior post-fastpath GGUF band and remains far
below the regenerated clean HF TP4 control:

- HF control strict: `114.187`
- HF control `c1_2000`: `116.623`
- HF control `c1_10000`: `109.748`

### Promote / Reject

Promote:

- the dispatch audit result that F16 GGUF linear wrappers are not the primary
  remaining throughput limiter once forced unquantized dispatch is active;
- the run as source-boundary evidence for future comparisons.

Reject:

- repeating this exact forced-unquantized-linear source change;
- treating generic GGUF `LinearBase` method selection as the remaining whole
  MoE TP4 gap.

### Next

Compare the clean HF TP4 control and the best GGUF TP4 path at the next lower
boundary: HIP/kernel mix and memory movement during decode, preferably without
per-token replay synchronization. Also investigate why the GGUF path still
captures a small decode graph envelope while the clean HF release path captures
a larger one, but do not repeat launch-shape-only tests without a new source
reason.

## GGUF-185 - Kausik-A Qwen3.6 27B MI50 vLLM Source Reference

External source reference:
`https://github.com/Kausik-A/qwen3.6-27b-mi50-vllm`

Local read-only clone:
`/tmp/kausik-qwen36-mi50-vllm`

Inspected commit: `61b273d` (`README: tighten TL;DR bullets`)

### Hypothesis

The Kausik-A single-MI50 GGUF bundle might contain compatibility patches that
explain part of the dense or MoE GGUF gap on the `.20` full-BAR/P2P-on release
lane.

### Findings

The repo targets a different deployment envelope from the `.20` benchmark
lane:

- single MI50 eGPU rather than 8x MI50 server tensor parallelism;
- ROCm 6.3 rather than the ROCm7.2 release path;
- Qwen3.6 27B quantized GGUF at `MAX_MODEL_LEN=4096`;
- `--enforce-eager`;
- no full-BAR/P2P-on TP8 dense or TP4 MoE reproduction target.

The useful source ideas are compatibility-level items, not a direct performance
path:

- `qwen35` / `qwen35moe` GGUF architecture aliases;
- explicit HF tokenizer/config route for GGUF;
- text-only Qwen3.5 / Qwen3.6 architecture override;
- M-RoPE bypass for text-only GGUF;
- `ssm_dt.bias` to `linear_attn.dt_bias` tensor mapping;
- 2D to 3D `conv1d.weight` reshape;
- `quant_config` plumbing through embeddings and `lm_head`;
- `IsHybrid` hooks on text-only Qwen3.5 / Qwen3.5-MoE classes;
- tuple-shard handling for GGUF fused QKVZ tensors.

### Outcome

Promote as source-reference evidence only.

### Promote / Reject

Promote:

- keeping the compatibility checklist above visible for future GGUF loader
  work;
- using the repo as a sanity reference for text-only Qwen3.5 GGUF startup
  fixes.

Reject:

- direct transplant as a `.20` reproduction path;
- treating its reported launch behavior as comparable benchmark evidence;
- repeating its eager/single-GPU/4096-context setup for the current goal.

### Reason

Most of the useful ideas are already present in, or have close equivalents in,
the local GGUF source work. The remaining dense and MoE gaps are below startup
compatibility: dense is strict-valid but slower than HF, while MoE TP4 is
strict-valid in GGUF only after multiple local source fixes and remains far
below the clean HF TP4 control.

### Next

Do not run a full benchmark from this repo as-is. Continue source-path
comparison against the regenerated `.20` clean HF control, focusing on
lower-level kernel mix, memory movement, graph envelope, and reduced
completed-work parity rather than single-GPU/eager deployment settings.

## GGUF-186 - Dense 27B GGUF TP8 Forced SSM Cache Float32

Host label: `.20`

Profile: `dense27b_tp8_fullbar_p2pon`

Model: `Qwen3.6-27B-F16-GGUF`

Run directory:
`<validation-workspace>/runs/dense27b_gguf_tp8_ssmfloat32_dot20_20260627T082830Z`

Benchmark directory:
`<validation-workspace>/runs/dense27b_gguf_tp8_ssmfloat32_dot20_20260627T082830Z/v02-profile-runs/dense27b_tp8_fullbar_p2pon_20260627T083420Z_ssmfloat32`

### Hypothesis

The best dense GGUF path was still using attention block size `208`, while the
clean HF dense path uses block size `400`. Forcing
`--mamba-ssm-cache-dtype float32` on the best dense GGUF CausalLM route might
align GGUF SSM cache/page geometry with HF and close the remaining dense
throughput gap.

### Configuration Delta

Started from the previous best dense GGUF source bundle and launch shape:

- TP8 full-BAR/P2P-on
- FP16 model weights
- graph mode
- async scheduling
- `MAX_MODEL_LEN=131072`
- `--load-format gguf`
- `--hf-overrides {"architectures":["Qwen3_5ForCausalLM"]}`
- `--max-num-seqs 2`
- `--max-num-batched-tokens 4`
- begin-think proxy benchmark path
- eight pre-measure warmups

Only launch delta:

- added `--mamba-ssm-cache-dtype float32`

### Startup Evidence

The geometry change worked:

- `mamba_ssm_cache_dtype`: `float32`
- attention block size: `400`
- GPU KV cache size: `377,200` tokens
- maximum concurrency at `131,072` tokens/request: `11.40x`
- CUDAGraph capture finished successfully
- API startup completed

### Results

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | 3542 | 50.533 | 62.992 | 70.093 | stop | True |
| `c1_2000` | 2000 | 44.231 | 63.628 | 45.217 | length | False |
| `c1_10000` | 10000 | 55.646 | 60.118 | 179.706 | length | False |

Warmup decode stabilized around `63.64` backend TPS.

### Outcome

Reject as a performance improvement.

### Promote / Reject

Promote:

- `--mamba-ssm-cache-dtype float32` as a diagnostic proving the dense GGUF
  CausalLM route can be forced onto HF-like block/page geometry.

Reject:

- forced SSM cache `float32` as a throughput fix;
- page/block geometry mismatch as the remaining dense GGUF whole-gap
  explanation.

### Reason

The flag corrected the visible page geometry, but the measured ladder was
slower than the previous best dense GGUF run:

- previous best GGUF: strict `63.679`, `c1_2000` `64.353`, `c1_10000`
  `60.617` backend TPS;
- forced SSM float32: strict `62.992`, `c1_2000` `63.628`, `c1_10000`
  `60.118` backend TPS.

It also remains below the clean HF dense control and published release band.
The remaining dense GGUF gap is below page geometry: likely kernel mix,
GGUF-loaded weight layout, graph/lowering details, state conversion cost, or
decode memory movement.

### Next

Do not repeat cache-dtype/page-geometry-only dense tests without a new source
reason. The next useful dense step is a lower-level comparison between the
best GGUF dense route and the clean HF dense control for decode kernel mix,
memory movement, and GDN/state-conversion work, preferably with a reduced
completed-work reproducer that preserves the HF fast path.

## GGUF-187 - Dense 27B GGUF Versus HF Compile-Debug Graph Comparison

Host label: `.20`

Profile: `dense27b_tp8_fullbar_p2pon`

GGUF debug run directory:
`<validation-workspace>/runs/dense27b_gguf_tp8_compile_debug_20260627T090014Z`

HF debug run directory:
`<validation-workspace>/runs/dense27b_hf_tp8_compile_debug_20260627T090636Z`

### Hypothesis

The remaining dense GGUF gap might be visible in the compiled graph surface
even before another full benchmark ladder: missing GDN/SwiGLU lowering,
different graph-file shape, extra GGUF materialization work, or a decode-path
custom-op surface that the HF-weight control does not carry.

### Method

Started one bounded compile-debug server for the current best dense GGUF
CausalLM route and one bounded compile-debug server for the clean dense
HF-weight control. Both used:

- TP8 full-BAR/P2P-on;
- FP16;
- graph mode;
- async scheduling;
- `MAX_MODEL_LEN=131072`;
- release dense overlay and P2P/RCCL settings;
- compile debug dumps and compile cache under the run directory.

No warmup or benchmark ladder was run. Both servers were stopped after reaching
health and writing compile artifacts.

### Startup Evidence

GGUF:

- debug dump: `616` files, `77` files per rank;
- compile cache: `176` files, about `83M`;
- attention block size: `208`;
- GPU KV cache: `377,520` tokens;
- max concurrency: `11.45x`;
- graph memory pool: about `0.42 GiB`;
- engine init: `85.61` seconds.

HF:

- debug dump: `616` files, `77` files per rank;
- compile cache: `280` files, about `129M`;
- attention block size: `400`;
- GPU KV cache: `354,800` tokens;
- max concurrency: `10.72x`;
- graph memory pool: about `0.42 GiB`;
- engine init: `150.48` seconds.

### Static Graph Counts

Both debug dumps contained the same normalized rank/file classes and the same
major GDN/SwiGLU markers:

- `torch.ops.vllm.gdn_attention_core`: GGUF `4608`, HF `4608`;
- `gdn_attention_core`: GGUF `4608`, HF `4608`;
- `silu`: GGUF `8424`, HF `8424`;
- `swiglu`: GGUF `15208`, HF `15208`;
- `qwen2_moe`: GGUF `11264`, HF `11264`;
- `in_proj_qkvz`: GGUF `6960`, HF `6960`.

Key differences:

- `_apply_gguf_embedding`: GGUF `176`, HF `0`;
- `masked_fill`: GGUF `232`, HF `0`;
- `bitwise_or`: GGUF `168`, HF `0`;
- `vllm.all_reduce`: GGUF `14680`, HF `14496`;
- `language_model`: GGUF `0`, HF `3840`;
- `embed_tokens`: GGUF `200`, HF `0`;
- rank-0 `kernel_0` size: GGUF `53,906` bytes, HF `28,717` bytes.

The GGUF-only `_apply_gguf_embedding` path appears in eight files per rank:
`BEFORE_PRE_GRAD.0`, generated compiled function, `after_split`,
`before_split`, `kernel_0`, `post_split_module`, pre-insert, and pre-split
artifacts.

### Outcome

Promote as source-boundary evidence, not as a performance result.

### Promote / Reject

Promote:

- the compile-debug artifact comparison as a repeatable, low-risk source
  diagnostic;
- the GGUF embedding custom-op / mask / all-reduce surface as the next narrow
  dense source candidate.

Reject:

- missing GDN/SwiGLU compile lowering as the explanation;
- gross graph-file-shape mismatch as the explanation;
- page/cache geometry as the whole-gap explanation, already rejected by
  `GGUF-186`;
- another full benchmark ladder before a narrow source change or timing probe.

### Reason

The GGUF and HF controls compile the same broad dense model surface and both
reach the GDN/SwiGLU paths. The remaining visible difference is not startup
compatibility. It is the GGUF-specific embedding/materialization surface that
survives into compiled artifacts, plus a small all-reduce count difference.

Earlier row-content diagnostics showed sampled embedding and lm-head rows
matched direct GGUF data, and an lm-head-only unquantized test did not improve
throughput. Therefore the next source hypothesis is not "the embedding rows are
wrong"; it is that the GGUF embedding custom-op/mask/all-reduce route may add
per-token decode overhead or alter graph lowering relative to the HF control.

### Next

Inspect a narrow embedding-only route that replaces only GGUF
`VocabParallelEmbedding` dispatch with the normal unquantized embedding path
for F16 GGUF, while leaving linear/GDN/lm-head semantics unchanged. Before a
full ladder, prove the patch changes the compile surface by removing or
reducing `_apply_gguf_embedding` artifacts. If the graph surface changes
without semantic regression, run the normal warmups -> `c1_128` uncapped strict
-> `c1_2000` -> `c1_10000` benchmark order.

## GGUF-188 - Dense 27B Embed-Tokens-Only Unquantized Dispatch Probe

Host label: `.20`

Profile: `dense27b_tp8_fullbar_p2pon`

Patch directory:
`<validation-workspace>/experimental-patches/gguf-embedtokens-unquant-20260627/`

Run directory:
`<validation-workspace>/runs/dense27b_gguf_tp8_embedtokens_unquant_compile_debug_20260627T092322Z`

### Hypothesis

If the GGUF-only `_apply_gguf_embedding` compile surface from `GGUF-187` came
from `model.embed_tokens`, then forcing only `embed_tokens` to
`UnquantizedEmbeddingMethod` should remove or reduce that custom-op surface
without touching linear layers, GDN, or `lm_head`.

### Configuration Delta

Started from the same dense GGUF TP8 compile-debug shape as `GGUF-187`.

Only source/runtime deltas:

- mounted a copy of the active `gguf-logical-shard-order/gguf.py`;
- added env-gated dispatch:
  `VLLM_GGUF_EMBED_TOKENS_UNQUANT=1`;
- forced only prefixes ending in `embed_tokens` to
  `UnquantizedEmbeddingMethod`.

The run did not execute the benchmark ladder. It reached health, wrote compile
artifacts, and was stopped afterward.

### Startup Evidence

- all eight TP workers logged:
  `VLLM_GGUF_EMBED_TOKENS_UNQUANT forcing model.embed_tokens to
  UnquantizedEmbeddingMethod`;
- GPU KV cache: `377,520` tokens;
- maximum concurrency: `11.45x`;
- graph capture finished in `3` seconds;
- engine init: `95.35` seconds;
- API startup completed.

A short direct chat request returned reasoning text and did not show garbage
output, but it was capped at `32` tokens and was not a strict benchmark
request.

### Compile Surface Result

The compile artifacts were unchanged versus `GGUF-187`:

- debug dump: `616` files;
- compile cache: `176` files, about `83M`;
- `_apply_gguf_embedding`: `176`;
- `masked_fill`: `232`;
- `bitwise_or`: `168`;
- `vllm.all_reduce`: `14680`;
- `gdn_attention_core`: `4608`;
- `swiglu`: `15208`;
- `qwen2_moe`: `11264`;
- `in_proj_qkvz`: `6960`;
- rank-0 `kernel_0`: `53,906` bytes.

Rank-0 debug dump still showed `_apply_gguf_embedding` operating on
`l_self_modules_embed_tokens_parameters_qweight_`, followed by mask handling
and TP all-reduce.

### Outcome

Reject `embed_tokens`-only unquantized dispatch as the next dense performance
fix.

### Promote / Reject

Promote:

- the finding that `embed_tokens` dispatch selection alone is not enough to
  remove the GGUF embedding custom-op graph surface;
- the compile-surface gate before spending time on full benchmark ladders.

Reject:

- running a full warmup/strict/fixed-token ladder for this patch;
- treating `embed_tokens` method selection as the remaining dense gap.

### Reason

The patch activated on every TP worker, but the compiled graph remained
byte-for-byte equivalent at the coarse artifact level and retained the same
GGUF embedding custom-op counts and `kernel_0` size. The issue is therefore
below the simple `get_quant_method()` branch for `embed_tokens`: either the
parameter/materialization path still supplies GGUF qweight semantics to the
compiled forward, Dynamo preserves the custom op despite method selection, or
the active graph surface is produced before the replacement takes effect.

Source reading supports this interpretation: `UnquantizedEmbeddingMethod`
creates `layer.weight` and calls `F.embedding(input_, layer.weight)`, while the
debug dump still references `l_self_modules_embed_tokens_parameters_qweight_`
and `torch.ops.vllm._apply_gguf_embedding`. The next fix cannot stop at method
selection; it has to prove that the active parameter path changes from `qweight`
to `weight`, or that the custom op itself is removed from the generated graph.
The mask and TP all-reduce around embedding are standard
`VocabParallelEmbedding.forward_native()` behavior; the source-specific suspect
is the GGUF custom op / `qweight` materialization path, not the existence of
the wrapper mask/reduce itself.

### Next

Do not repeat `embed_tokens`-only dispatch. The next narrow source test should
trace or alter the actual `VocabParallelEmbedding` parameter/materialization
path for F16 GGUF, or compare the generated graph around the exact
`l_self_modules_embed_tokens_parameters_qweight_` source against HF. A separate
lm-head-only unquantized test was already rejected as a throughput fix, so the
next candidate must target the embedded custom-op generation path itself rather
than another broad method-selection flag.

## GGUF-189 - Dense 27B Skip Embed Qweight Compile And Ladder

Host label: `.20`

Profile: `dense27b_tp8_fullbar_p2pon`

Patch directories:

- `<validation-workspace>/experimental-patches/gguf-embedtokens-unquant-20260627/`
- `<validation-workspace>/experimental-patches/combined-release-dense-gguf-skipembedqweight-20260627/`

Run directory:
`<validation-workspace>/runs/dense27b_gguf_tp8_skipembedqweight_compile_debug_20260627T093747Z`

Benchmark directory:
`<validation-workspace>/runs/dense27b_gguf_tp8_skipembedqweight_compile_debug_20260627T093747Z/v02-profile-runs/dense27b_tp8_fullbar_p2pon_20260627T093747Z_skipembedq`

### Hypothesis

`GGUF-188` proved that changing the embedding quant method was not enough,
because the Qwen3.5 dense overlay reinstalled GGUF `qweight` materialization for
`model.embed_tokens`. This test skipped only the `embed_tokens` qweight override
and materialization while leaving `lm_head`, GDN, linears, release attention,
FP16, graph mode, P2P-on, and `MAX_MODEL_LEN=131072` unchanged.

### Configuration Delta

The source bundle added an env-gated Qwen3.5 overlay change:

- `VLLM_QWEN35_GGUF_SKIP_EMBED_QWEIGHT=1`
- `VLLM_GGUF_EMBED_TOKENS_UNQUANT=1`

The patch logged skip markers for `model.embed_tokens` on all TP workers.
`lm_head` still materialized the GGUF qweight path and was intentionally not
changed.

### Startup And Compile Evidence

The server reached API health and graph capture completed.

- GPU KV cache: `372,944` tokens;
- maximum concurrency: `11.32x`;
- engine init: `90.13` seconds;
- debug dump: `856` files;
- compile cache: `208` files;
- `_apply_gguf_embedding`: `0`;
- `gdn_attention_core`: `4608`;
- `swiglu`: `15208`;
- `qwen2_moe`: `11264`;
- `in_proj_qkvz`: `6960`;
- rank-0 `kernel_0`: `4,287` bytes.

The generated graph changed materially: the prior GGUF `_apply_gguf_embedding`
custom-op surface disappeared and the rank-0 kernel shrank. A short direct
request produced coherent reasoning text, so the patch was allowed to run the
normal benchmark order.

### Benchmark Result

The normal ladder used eight pre-measure warmups, then uncapped strict, then
`c1_2000`, then `c1_10000`.

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | 3658 | 51.229 | 63.606 | 71.405 | stop | True |
| `c1_2000` | 2000 | 44.515 | 64.432 | 44.929 | length | False |
| `c1_10000` | 10000 | 55.956 | 60.691 | 178.713 | length | False |

Warmups were stable around the same backend decode band and did not indicate a
late ramp that would explain the final numbers.

### Outcome

Reject as a dense throughput fix; promote as source-boundary evidence.

### Promote / Reject

Promote:

- the source patch as proof that the GGUF embedding custom-op graph surface can
  be removed without breaking Qwen strict correctness;
- the benchmark ladder as evidence that removing `_apply_gguf_embedding` is not
  enough to close the dense GGUF performance gap;
- the Kausik-A source bundle as a compatibility checklist only, not as a
  comparable release path.

Reject:

- treating the GGUF embedding custom op as the whole dense performance gap;
- repeating embed/qweight surgery without a lower-level timing or kernel-mix
  explanation;
- importing the Kausik-A single-GPU/eager launch shape into the ROCm7.2
  full-BAR/P2P-on benchmark path.

### Reason

This was the first dense GGUF probe that actually changed the compile surface:
`_apply_gguf_embedding` went to zero and the generated rank-0 kernel shrank
substantially. If that custom op were the dominant per-token drag, the benchmark
ladder should have moved upward. Instead it came in slightly below the previous
best dense GGUF result (`63.679` strict, `64.353` c1_2000, `60.617` c1_10000)
and still below the clean HF/v0.2.1 dense band.

The result narrows the next useful work below broad graph-shape and embedding
dispatch. The remaining gap is more likely in decode kernel mix, memory
movement, GDN/state conversion, GGUF-loaded weight layout, or other
source-level execution details that survive after embedding qweight removal.

### Next

Do not repeat embedding-only qweight removal as a performance path. The next
dense work should compare lower-level kernel mix or timing between the current
best GGUF path and the clean HF control, preferably with a reduced diagnostic
that does not perturb the full release-like graph. Kausik-A's repository should
remain a source-reference checklist for compatibility patches, not a direct
benchmark profile.

## GGUF-190 - Dense 27B Wrapper/Prefix Parity Probes

Host label: `.20`

Profile intent: `dense27b_tp8_fullbar_p2pon`

Probe directories:

- `<validation-workspace>/runs/dense27b_gguf_tp8_conditional_wrapper_probe_20260627T101323Z`
- `<validation-workspace>/runs/dense27b_gguf_tp8_langprefix_probe_20260627T101829Z`

### Hypothesis

The dense HF compile-debug control resolved to `Qwen3_5ForConditionalGeneration`
and emitted `language_model.model.layers.*` source prefixes, while the current
best dense GGUF path resolves to `Qwen3_5ForCausalLM` and unwrapped
`model.layers.*` prefixes. This test checked whether wrapper/prefix parity was
a missing source condition before running another benchmark ladder.

### Probe A: Conditional Wrapper

The first probe kept the release-like GGUF TP8/P2P-on launch but changed the
architecture override to `Qwen3_5ForConditionalGeneration`, used the full HF
config from cache, and removed `--language-model-only`.

Outcome: reject before benchmark.

Reason: the server did not reach health. During memory profiling, vLLM treated
the model as multimodal and executed the visual tower dummy path. The GGUF file
does not contain initialized visual weights, so the run failed with an
uninitialized GGUF parameter in the visual `qkv` projection.

### Probe B: Text-Only Language Prefix

The second probe copied the current best dense GGUF source bundle and added an
env-gated CausalLM prefix change:

- `VLLM_QWEN35_FORCE_LANGUAGE_MODEL_PREFIX=1`

This kept the text-only `Qwen3_5ForCausalLM` architecture and only forced the
internal source prefix to `language_model` to test whether the HF wrapper prefix
was itself a performance condition.

Outcome: reject before benchmark.

Reason: the server did not reach health. The prefix change caused GGUF
parameters to remain uninitialized during compile/profile, producing an
uninitialized `GGUFUninitializedParameter` error. This means the current GGUF
loader/materialization path is coupled to the unwrapped CausalLM prefix and
cannot be changed by prefix alone.

### Promote / Reject

Promote:

- wrapper/prefix parity as a documented source boundary;
- the finding that naive ConditionalGeneration enables an unwanted multimodal
  profile path for the GGUF artifact;
- the finding that prefix-only CausalLM parity breaks GGUF parameter
  materialization before health.

Reject:

- using the full multimodal conditional wrapper as a dense GGUF benchmark path;
- using a prefix-only CausalLM change as a benchmark path;
- spending a full warmup/strict/fixed-token ladder on wrapper parity until a
  real text-only wrapper class also repairs GGUF materialization.

### Next

If wrapper parity is revisited, it needs a real text-only wrapper that provides
`language_model.*` source prefixes without advertising multimodal inputs and
without breaking GGUF weight materialization. Until then, continue below
wrapper/embedding/launch-shape explanations: kernel mix, memory movement,
GDN/state conversion, or loaded-weight layout against the clean HF controls.

## GGUF-191 - Dense 27B GDN Custom-Op Timing Window

Host label: `.20`

Profile intent: `dense27b_tp8_fullbar_p2pon`

Run directories:

- `<validation-workspace>/runs/dense27b_gguf_tp8_gdnop_timing_dot20_20260627T103024Z`
- `<validation-workspace>/runs/dense27b_hf_tp8_gdnop_timing_dot20_20260627T104237Z`

### Hypothesis

After wrapper, cache geometry, embedding dispatch, broad graph shape, and
launch-shape explanations were rejected, this test checked whether dense GGUF
was spending extra request-window time inside the registered
`torch.ops.vllm.gdn_attention_core` body.

### Method

Both runs used the published v0.2 image, `.20`, P2P-on, TP8, FP16,
`MAX_MODEL_LEN=131072`, graph mode, persistent all-reduce, and the dense
release overlay path. The only diagnostic change was a read-only mount of the
existing direct custom-op timing `qwen3_next.py` over the container source with:

- `VLLM_GDN_OP_TIMING=1`
- `VLLM_GDN_OP_TIMING_SYNC=0`
- `VLLM_GDN_OP_TIMING_FLUSH_EVERY=1`

The server was allowed to reach health. Startup/profiling timing rows were
deleted, then a small warmup request was sent, timing rows were deleted again,
and one measured 128-token request was sent.

This was not a benchmark ladder and is not a publication TPS result.

### Result

Both request windows completed and produced `1152` timing rows.

Summary from `gdn_op_timing_summary.txt`:

| Path | Rows | Avg `last_ns` | Avg `last_ns` under 100 ms | Max `last_ns` |
| --- | ---: | ---: | ---: | ---: |
| Dense GGUF TP8 | 1152 | 1,755,368 | 1,755,368 | 31,437,127 |
| Dense HF TP8 control | 1152 | 1,951,620 | 1,951,620 | 55,090,078 |

Layer-0 averaged `11,951,702` ns for GGUF and `16,107,229` ns for HF in this
window. Most non-layer-0 averages were in the `1.5` to `1.9` ms range.

### Promote / Reject

Promote:

- direct custom-op timing as a low-impact dense diagnostic route;
- the run directories as request-window evidence for the dense GDN custom-op
  boundary;
- the finding that the measured dense GGUF request path is not slower inside
  the registered GDN core body.

Reject:

- `gdn_attention_core` body time as the remaining dense GGUF throughput gap;
- repeating GDN custom-op timing without a new source-level reason;
- using this request-window timing as a substitute for normal warmup -> strict
  -> fixed-token benchmark tiers.

### Next

Continue below the GDN custom-op body. The next dense source target should be
whole-decode kernel mix, memory movement around projections/residual/norm,
graph replay/scheduler overhead, loaded-weight layout outside the core op, or
another reduced diagnostic that can preserve the clean HF control.

## GGUF-192 - Dense 27B GGUF/HF Generated Kernel Diff Review

Host label: `.20`

Profile intent: `dense27b_tp8_fullbar_p2pon`

Input artifacts:

- `<validation-workspace>/runs/dense27b_gguf_tp8_compile_debug_20260627T090014Z`
- `<validation-workspace>/runs/dense27b_hf_tp8_compile_debug_20260627T090636Z`

Comparison artifact:

- `<validation-workspace>/runs/dense27b_gguf_hf_kernel_compare_20260627T110302Z`

### Hypothesis

The earlier dense compile-debug comparison proved broad file counts and marker
counts were similar, but that could still hide differences in generated kernel
bodies. This review normalized UUID/path/hash-like noise and compiled-function
metadata, then compared the generated kernel and pre-grad source bodies.

### Method

No new serving container or benchmark ladder was started. The review used
existing compile-debug dumps from `.20` and compared:

- `__compiled_fn_1.kernel_*.py` files;
- `__compiled_fn_1.BEFORE_PRE_GRAD.*.py` files;
- broad marker counts for the same dense GGUF/HF debug pair.

This is a source diagnostic only. It is not a throughput result and does not
replace the normal warmups -> uncapped strict -> `c1_2000` -> `c1_10000`
benchmark sequence.

### Result

The normalized comparison found generated-body differences across the full
debug pair:

| File class | Total | Same | Different | Missing | Diff lines |
| --- | ---: | ---: | ---: | ---: | ---: |
| generated kernels | 48 | 0 | 48 | 0 | 33,582 |
| pre-grad sources | 520 | 0 | 520 | 0 | 143,824 |

Marker counts:

| Marker | GGUF | HF |
| --- | ---: | ---: |
| `gdn_attention_core` | 4,608 | 4,608 |
| `swiglu` | 15,208 | 15,208 |
| `vllm.all_reduce` | 14,680 | 14,496 |
| `_apply_gguf_embedding` | 176 | 0 |
| `masked_fill` | 232 | 0 |
| `bitwise_or` | 168 | 0 |
| `qwen35_effective_weight` | 0 | 0 |

The clearest localized difference remains rank-0 `kernel_0`: GGUF includes an
extra token-id remap / embedding-associated pointwise region with
`aten.ge`, `aten.lt`, `aten.bitwise_and`, `aten.bitwise_or`,
`aten.bitwise_not`, `aten.unsqueeze`, `aten.masked_fill`, and
`vllm.all_reduce`. HF does not have that surface in the matching kernel.

Many other diffs appear to be driven by symbolic capture geometry: the HF
debug files use symbolic dimension `s18` in many generated regions, while the
GGUF files use `s72`.

The launch logs also show an important caveat. The compared debug artifacts are
not geometry-identical:

| Path | Resolved architecture | Attention block size / cache evidence |
| --- | --- | --- |
| GGUF debug | `Qwen3_5ForCausalLM` with `--language-model-only` and `--load-format gguf` | GPU KV cache `377,520` tokens, max concurrency `11.45x` |
| HF debug | `Qwen3_5ForConditionalGeneration` | GPU KV cache `354,800` tokens, max concurrency `10.72x` |

Earlier forced-cache testing (`GGUF-186`) showed moving GGUF to HF-like cache
geometry did not improve the dense ladder, so this diagnostic should not be
overread as "cache geometry is the root cause."

### Promote / Reject

Promote:

- the normalized kernel comparison as useful source-boundary evidence;
- the finding that broad marker-count parity is not enough to prove generated
  body parity;
- the finding that GGUF-only token/embedding remap work is still visible in
  generated `kernel_0`;
- the finding that the current dense debug pair also differs by wrapper/cache
  geometry, which must be controlled before making a final root-cause claim.

Reject:

- treating this comparison as a benchmark result;
- treating the generated-body diffs as a proven root cause without an aligned
  comparator;
- repeating unchanged dense benchmark ladders from the same GGUF source state;
- returning to broad marker-count parity as sufficient closure.

### Next

Build a cleaner aligned comparator before the next full dense ladder. The most
useful next probe is either:

- a text-only HF comparator that preserves the fast HF dense band while using a
  CausalLM/source-prefix shape closer to GGUF; or
- a GGUF comparator that preserves the current best GGUF source state while
  matching the HF debug geometry without reintroducing the visual dummy path.

If that aligned comparator still shows generated-body differences, continue
with lower-level whole-decode kernel mix, memory movement, residual/norm and
projection boundaries, graph replay/scheduler overhead, or loaded-weight layout
outside the already rejected GDN custom-op body.

## GGUF-193 - Dense HF Text-Only CausalLM Comparator

### Question

Could the remaining dense GGUF gap be explained by the GGUF path using the
text-only `Qwen3_5ForCausalLM` wrapper while the clean HF control uses the
`Qwen3_5ForConditionalGeneration` wrapper?

The external `Kausik-A/qwen3.6-27b-mi50-vllm` source bundle also uses the
text-only `Qwen3_5ForCausalLM` route for dense Qwen3.6 GGUF, so this is the
right control before attributing the dense gap to wrapper shape.

### Setup

Host: `.20`

Run directory:

```text
<validation-workspace>/runs/dense27b_hf_tp8_textonly_causallm_compare_20260627T111054Z
```

Container:

```text
vllm_qwen36_dense27b_hf_tp8_textonly_causallm_compare
```

Key launch properties:

- published v0.2 ROCm7.2 image;
- HF dense Qwen3.6 27B weights, not GGUF;
- `Qwen3_5ForCausalLM` via `--language-model-only` and HF overrides;
- TP8, full-BAR/P2P-on;
- `MAX_MODEL_LEN=131072`;
- FP16;
- same dense release overlay and graph path;
- `--max-num-seqs 2`;
- `--max-num-batched-tokens 4`;
- prefix caching disabled.

A short direct coherence probe returned coherent reasoning text before the
benchmark ladder started.

The normal release fixture was then run:

- 8 warmups;
- `c1_128` uncapped strict;
- `c1_2000`;
- `c1_10000`.

### Result

Benchmark result directory:

```text
<validation-workspace>/runs/dense27b_hf_tp8_textonly_causallm_compare_20260627T111054Z/v02-profile-runs/dense27b_tp8_fullbar_p2pon_20260627T111054Z_textonly_causallm
```

| Case | Completion tokens | Client TPS | Backend decode TPS | Finish | Strict gate valid |
| --- | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | 3119 | 53.768 | 70.353 | stop | true |
| `c1_2000` | 2000 | 48.318 | 70.978 | length | false |
| `c1_10000` | 10000 | 60.870 | 66.428 | length | false |

Warmups were stable in the same band, with backend decode TPS near 71 on the
2K warmup requests.

The test container was stopped after the run.

### Promote / Reject

Promote:

- the HF text-only `Qwen3_5ForCausalLM` comparator as a valid control;
- the result that the text-only CausalLM wrapper can still match or exceed the
  dense HF/release band on `.20`;
- the Kausik-style text-only wrapper idea as compatibility plumbing, not as a
  throughput explanation.

Reject:

- wrapper class alone as the dense GGUF throughput gap;
- repeating ConditionalGeneration-versus-CausalLM wrapper probes without a more
  specific lower-level hypothesis;
- importing the external single-GPU eager launch profile into the release path.

### Next

The remaining dense GGUF gap sits below wrapper choice, broad graph marker
counts, cache/page geometry, embedding dispatch removal, and direct
`gdn_attention_core` body timing. Continue with whole-decode kernel mix,
memory movement, residual/norm/projection boundaries, graph replay/scheduler
overhead, or GGUF-loaded weight layout. The aligned comparator should be used
as the HF side for future generated-kernel diff work.

## GGUF-194 - Dense GGUF vs HF Text-Only Debug-Dump Comparison

### Purpose

Compare the current best dense GGUF compile surface against an aligned HF
text-only `Qwen3_5ForCausalLM` control. This follows the source-reference review
of `Kausik-A/qwen3.6-27b-mi50-vllm`, which uses the same broad compatibility
ideas: text-only Qwen3.5/Qwen3.6 registry entries, `qwen35` GGUF aliases,
M-RoPE stripping for the text-only head, `ssm_dt.bias` mapping, MoE expert
aliases, `quant_config` plumbing, conv1d reshape, `IsHybrid` hooks, and a
GGUF-aware fused QKVZ loader.

The external package is useful as compatibility evidence, but its deployment
profile is not comparable to the release path: single MI50/eGPU, ROCm 6.3,
4096 context, Q6 GGUF, and `--enforce-eager`. Our target remains `.20`, the
published ROCm7.2 image, TP8 full-BAR/P2P-on, FP16, graph capture,
`MAX_MODEL_LEN=131072`, and the release benchmark fixture.

### Setup

GGUF debug source:

```text
dense27b_gguf_tp8_compile_debug_20260627T090014Z
```

HF text-only debug-dump source:

```text
dense27b_hf_tp8_textonly_compile_debugdump_env_20260627T114602Z
```

Comparison artifact:

```text
dense27b_gguf_hf_textonly_debugdump_compare_20260627T115023Z
```

The first HF debug run was rejected because it did not pass vLLM's explicit
`--compilation-config` debug-dump options. The accepted run used the same HF
text-only comparator settings that already matched the release TPS band, then
added only:

```text
{"debug_dump_path":"/runwork/debug_dump","cache_dir":"/runwork/compile_cache","compile_cache_save_format":"unpacked"}
```

The accepted container reached health and was stopped after artifact capture.

### Result

| Artifact class | GGUF files | HF text-only files | Common | GGUF-only | HF-only | Same | Different |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Generated kernels | 48 | 368 | 48 | 0 | 320 | 0 | 48 |
| Pre-grad source | 520 | 520 | 520 | 0 | 0 | 0 | 520 |
| All Python debug files | 616 | 1416 | 600 | 16 | 816 | 0 | 600 |

Marker counts across all debug-dump Python files:

| Marker | GGUF | HF text-only |
| --- | ---: | ---: |
| `gdn_attention_core` | 4608 | 4608 |
| `swiglu` | 15208 | 15768 |
| `vllm.all_reduce` | 14680 | 16408 |
| `_apply_gguf_embedding` | 176 | 0 |
| `gguf_embedding` | 232 | 0 |
| `index_select` | 2184 | 0 |
| `masked_fill` | 232 | 272 |
| `bitwise_or` | 168 | 256 |

Per-rank generated-kernel count:

| Path | Kernels per rank |
| --- | ---: |
| GGUF | 6 |
| HF text-only | 46 |

The common kernel filenames were not equivalent bodies. For example,
`kernel_3.py` was roughly `1915` lines on GGUF but only `59` lines on HF
text-only across each rank. HF text-only had many HF-only generated kernels
containing SwiGLU and all-reduce call sites; GGUF instead kept a much larger
set of logic inside the six common kernel bodies and retained GGUF-specific
embedding/index-select machinery.

### Promote / Reject

Promote:

- the HF text-only comparator as the correct aligned HF side for future
  generated-kernel, graph replay, and scheduler analysis;
- the Kausik-style source ideas as compatibility plumbing for loading Qwen3.6
  GGUFs through vLLM;
- the conclusion that broad `gdn_attention_core` marker parity does not imply
  decode-surface parity.

Reject:

- text-only `Qwen3_5ForCausalLM` wrapper choice as the dense GGUF throughput
  fix;
- importing the external single-GPU eager launch settings into the release
  performance path;
- another wrapper or M-RoPE-only experiment unless it is tied to a lower-level
  generated-kernel or replay hypothesis.

### Next

The dense GGUF gap now points at lower-level execution differences in generated
kernel partitioning, GGUF-loaded weight layout, embedding/index-select
residue, scheduler/replay behavior, or memory movement between residual/norm
and projection boundaries. The next useful source probe should instrument or
replace one of those boundaries directly rather than repeating wrapper,
embedding-dispatch, cache-geometry, or GDN-body timing experiments.

## GGUF-195 - Dense GGUF Inductor Graph-Partition Probe

### Purpose

Test whether the dense GGUF throughput gap is caused by the GGUF graph being
lowered into too few large generated kernels. The aligned debug comparison in
GGUF-194 showed the current dense GGUF path producing only `48` generated
kernels (`6` per rank), while the clean HF text-only comparator produced `368`
generated kernels (`46` per rank) under an otherwise comparable TP8
full-BAR/P2P-on release lane.

### Setup

Host label: `.20`

Model:

```text
Qwen3.6-27B-F16-GGUF
```

Profile:

```text
dense27b_tp8_fullbar_p2pon
```

Run directory:

```text
<validation-workspace>/runs/dense27b_gguf_tp8_graphpartition_probe_20260627T120119Z
```

Benchmark directory:

```text
<validation-workspace>/runs/dense27b_gguf_tp8_graphpartition_probe_20260627T120119Z/v02-profile-runs/dense27b_tp8_fullbar_p2pon_20260627T120903Z_graphpartition
```

The probe used the published v0.2 image, TP8, P2P-on, FP16,
`MAX_MODEL_LEN=131072`, graph mode, async scheduling, the best dense GGUF
source bundle, and the normal benchmark fixture sequence:

```text
warmups -> c1_128 uncapped strict -> c1_2000 -> c1_10000
```

Only the vLLM compilation config was changed:

```text
{"debug_dump_path":"/runwork/debug_dump","cache_dir":"/runwork/compile_cache","compile_cache_save_format":"unpacked","use_inductor_graph_partition":true}
```

### Compile Surface

The probe reached health and changed the generated-kernel partitioning
materially.

| Artifact | Count |
| --- | ---: |
| Debug-dump Python files | `416` |
| Generated kernels | `280` |
| Pre-grad files | `8` |
| Generated kernels per rank | `35` |

Marker counts across debug-dump Python files:

| Marker | Count |
| --- | ---: |
| `gdn_attention_core` | `14272` |
| `swiglu` | `24704` |
| `vllm.all_reduce` | `37208` |
| `_apply_gguf_embedding` | `336` |
| `gguf_embedding` | `472` |
| `index_select` | `3584` |
| `masked_fill` | `392` |
| `bitwise_or` | `424` |

This moved GGUF from `6` generated kernels per rank to `35` per rank, closer
to the HF text-only comparator's `46` per rank. It did not remove GGUF-specific
embedding/index-select residue.

### Benchmark Result

Eight 2000-token warmups completed before measurement. Warmups 2 through 8
were stable around `64.5` backend decode TPS.

| Case | Completion tokens | Client TPS | Backend decode TPS | Finish | Strict gate valid |
| --- | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | `3611` | `51.883` | `63.795` | `stop` | `true` |
| `c1_2000` | `2000` | `46.075` | `64.556` | `length` | `false` |
| `c1_10000` | `10000` | `56.105` | `60.826` | `length` | `false` |

Comparison points:

| Path | Strict backend TPS | c1_2000 backend TPS | c1_10000 backend TPS |
| --- | ---: | ---: | ---: |
| Dense GGUF graph-partition probe | `63.795` | `64.556` | `60.826` |
| Previous best dense GGUF | `63.679` | `64.353` | `60.617` |
| HF text-only comparator | `70.353` | `70.978` | `66.428` |
| v0.2.1 `.20` reproduction target | `68.962` | `69.914` | `65.645` |

### Outcome

Reject Inductor graph partitioning alone as the dense GGUF reproduction fix.
It materially changes the generated-kernel surface and gives a small uplift,
but the long-tier result remains in the same dense GGUF band and still misses
the v0.2.1 reproduction path by about `4.819` backend TPS on `c1_10000`.

### Promote / Reject

Promote:

- generated-kernel partitioning as a real source boundary worth understanding;
- the graph-partition probe as evidence that kernel partition shape affects
  dense GGUF throughput slightly;
- lower-level kernel mix / replay / memory movement analysis using this probe
  and the HF text-only comparator as paired references.

Reject:

- `use_inductor_graph_partition=true` as a release-path update;
- another full graph-partition benchmark unless a source patch also changes
  GGUF-loaded weight layout, embedding/index-select residue, or replay/memory
  movement;
- updating README, release notes, or the container from this result.

### Next

Continue below the graph-partition toggle. The remaining dense GGUF gap is now
smaller but still persistent after rejecting wrapper choice, cache/page
geometry, embedding qweight removal, GDN custom-op body timing, and graph
partitioning alone. The next useful direction is low-level comparison of kernel
mix, replay boundaries, memory movement around residual/norm/projection, or
GGUF-loaded weight layout against the clean HF text-only comparator.

## GGUF-196 - MoE TP4 Exact-HF-Shape Inductor Graph-Partition Probe

### Purpose

Test whether the remaining MoE GGUF TP4 gap is caused by Inductor graph
partitioning below the Python launch and fastpath layer. GGUF-181 already
matched the clean HF scheduler shape and mounted the release TP4 MoE fastpath,
but left `use_inductor_graph_partition=false`. This probe changed only the
compilation config to enable Inductor graph partitioning and capture debug
artifacts.

### Setup

Host label: `.20`

Model:

```text
Qwen3.6-35B-A3B-F16-GGUF
```

Profile:

```text
moe35b_tp4_fullbar_p2pon
```

Run directory:

```text
<validation-workspace>/runs/moe35b_gguf_tp4_graphpartition_exact_hfshape_dot20_20260627T122551Z
```

Benchmark directory:

```text
<validation-workspace>/runs/moe35b_gguf_tp4_graphpartition_exact_hfshape_dot20_20260627T122551Z/v02-profile-runs/moe35b_tp4_fullbar_p2pon_20260627T123306Z_graphpartition
```

Launch conditions preserved from the best fastpath-mounted GGUF MoE comparator:

- published v0.2 image;
- TP4, P2P-on, FP16;
- `MAX_MODEL_LEN=131072`;
- exact clean-HF scheduler shape:
  `--max-num-seqs 1 --max-num-batched-tokens 1024`;
- `--enable-prefix-caching`;
- `--mamba-cache-mode align`;
- `--language-model-only`;
- release TP4 MoE fastpath mounted;
- forced unquantized F16 MoE expert path;
- normal benchmark order:
  warmups -> `c1_128` uncapped strict -> `c1_2000` -> `c1_10000`.

The only intended compile change was:

```text
{"debug_dump_path":"/runwork/debug_dump","cache_dir":"/runwork/compile_cache","compile_cache_save_format":"unpacked","use_inductor_graph_partition":true}
```

One false start had malformed JSON for `--compilation-config`; it was stopped
and removed before model load. The accepted launch showed
`use_inductor_graph_partition: True` in vLLM's parsed engine config.

### Compile Surface

The server reached health. Graph capture remained `[1, 2]`, and the fastpath
pattern matched the earlier exact-HF-shape GGUF run:

- active fastpath log entries: `4`;
- rejected fastpath log entries: `16`;
- active token count observed: `1`;
- rejected token count included `1024` and `2`.

The debug dump showed a materially different partition shape from the older
MoE debug comparison:

| Artifact | Count |
| --- | ---: |
| Debug-dump Python files | `252` |
| Generated kernels | `184` |
| Pre-grad files | `4` |
| Generated kernels per rank | `46` |

Marker counts:

| Marker | Count |
| --- | ---: |
| `gdn_attention_core` | `4468` |
| `moe_forward_shared` | `6944` |
| `rocm_unquantized_gemm` | `16092` |
| `vllm.all_reduce` | `12492` |
| `_qwen35_effective_weight` | `3636` |
| `_apply_gguf_embedding` | `0` |
| `index_select` | `0` |

### Benchmark Result

Warmups 2 through 8 were stable at about `84.05` backend decode TPS.

| Case | Completion tokens | Client TPS | Backend decode TPS | Finish | Strict gate valid |
| --- | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | `3278` | `81.499` | `82.412` | `stop` | `true` |
| `c1_2000` | `2000` | `82.558` | `84.054` | `length` | `false` |
| `c1_10000` | `10000` | `73.990` | `74.228` | `length` | `false` |

Comparison points:

| Path | Strict backend TPS | c1_2000 backend TPS | c1_10000 backend TPS |
| --- | ---: | ---: | ---: |
| MoE GGUF graph-partition exact-HF-shape probe | `82.412` | `84.054` | `74.228` |
| MoE GGUF exact-HF-shape fastpath baseline | `82.543` | `84.195` | `74.314` |
| Clean HF TP4 control | `114.187` | `116.623` | `109.748` |

### Outcome

Reject Inductor graph partitioning as the MoE GGUF TP4 reproduction fix. It
changed the generated-kernel partitioning substantially, but the benchmark
numbers remained effectively identical to the earlier exact-HF-shape fastpath
GGUF band and far below the clean HF TP4 control.

### Promote / Reject

Promote:

- graph partitioning as another closed source boundary for MoE GGUF TP4;
- the observation that the MoE gap persists even when the generated-kernel
  count is moved to `46` per rank;
- keeping exact scheduler shape, release fastpath state, and compile
  partitioning in the same future run records.

Reject:

- `use_inductor_graph_partition=true` as a release-path update;
- repeating graph-partition-only MoE ladders;
- treating generated-kernel count alone as the current MoE GGUF TP4 bottleneck.

### Next

The remaining MoE GGUF gap is now below launch shape, fastpath activation,
method selection, broad generated-kernel body parity, and graph partitioning.
The next useful direction is lower-level HIP/kernel mix, memory movement around
expert routing and postprocess, or a reduced completed-work reproducer that can
compare HF-loaded and GGUF-loaded weights without changing the high-band HF
control behavior.

## GGUF-197 - Existing Timing and Debug Artifact Review

### Purpose

Review the existing `.20` timing and debug-dump artifacts before launching
another GGUF run. The goal was to avoid repeating known-bad profiler or replay
timing shapes, and to identify the next source-level target from already
captured evidence.

### Inputs Reviewed

MoE timing artifacts:

- `moe35b_gguf_tp4_runner_timing_full_ladder_dot20_20260627T031446Z`
- `moe35b_hf_tp4_cgreplay_timing_dot20_20260627T040136Z`
- `moe35b_hf_tp4_cgreplay_lag_timing_dot20_20260627T042121Z`
- earlier no-sync runner timing artifacts for HF and GGUF MoE TP4

Dense debug artifacts:

- `dense27b_gguf_tp8_graphpartition_probe_20260627T120119Z`
- `dense27b_hf_tp8_textonly_compile_debugdump_env_20260627T114602Z`

MoE debug artifacts:

- `moe35b_gguf_tp4_graphpartition_exact_hfshape_dot20_20260627T122551Z`
- `moe35b_hf_tp4_broad_graph_debugentry_dot20_20260627T013947Z`

### Timing Readout

The runner / cudagraph replay timing path is diagnostic-only. Once
synchronizing replay timing is inserted, even the HF-weight TP4 path collapses
into the same low throughput band as GGUF:

| Path | Strict backend TPS | c1_2000 backend TPS | c1_10000 backend TPS |
| --- | ---: | ---: | ---: |
| GGUF runner timing ladder | `71.668` | `72.936` | `65.400` |
| HF cudagraph replay timing ladder | `71.784` | `72.653` | `65.151` |

That is far below the clean HF TP4 control (`114.187`, `116.623`, `109.748`),
so this timing shape cannot be used as a clean HF-versus-GGUF performance
differential. The no-sync runner timing variants mostly measure CPU envelope
and omit enough GPU work to be misleading as bottleneck evidence.

### Dense Rank-0 Graph Surface

One-rank counts from the dense graph-partition debug dumps:

| Path | Python files | Kernel files | Kernel perf files |
| --- | ---: | ---: | ---: |
| Dense GGUF graph partition | `52` | `35` | `14` |
| Dense HF text-only comparator | `177` | `46` | `19` |

Selected rank-0 marker / op references:

| Marker / op | Dense GGUF | Dense HF text-only |
| --- | ---: | ---: |
| `torch.ops.vllm.all_reduce_inplace_kind` | `4480` | `2002` |
| `torch.ops.vllm.rocm_unquantized_gemm` | `3854` | `1557` |
| `torch.ops.vllm.gdn_attention_core` | `1680` | `576` |
| `index_select` | `560` | `0` |
| `_apply_gguf_embedding` | `52` | `0` |
| `torch.ops.vllm._apply_gguf_embedding` | `20` | `0` |
| `gguf_embedding` | `17` | `0` |

The embedding/index-select surface is still present in the dense GGUF
graph-partition dump, even though earlier embedding-dispatch surgery showed
that removing the embedding qweight path alone did not close the benchmark
gap. Treat this as evidence of a broader GGUF graph/materialization route
difference, not as a simple embedding correctness bug.

### MoE Rank-0 Graph Surface

One-rank counts from the MoE graph-partition GGUF dump:

| Path | Python files | Kernel files | Kernel perf files |
| --- | ---: | ---: | ---: |
| MoE GGUF graph partition | `63` | `46` | `20` |

Selected rank-0 marker / op references:

| Marker / op | MoE GGUF graph partition |
| --- | ---: |
| `torch.ops.vllm.all_reduce` | `2999` |
| `torch.ops.vllm.rocm_unquantized_gemm` | `2210` |
| `torch.ops.vllm.gdn_attention_core` | `1050` |
| `_qwen35_effective_weight` | `909` |
| `torch.ops.vllm.moe_forward_shared` | `888` |

The nearby HF broad debug-entry artifact is not a clean high-band comparator:
it has only `14` kernel files and no kernel perf files for the checked rank.
Use it only as orientation, not as proof of the HF release-band kernel mix.

### Outcome

No new benchmark run was launched. The existing artifacts narrow two things:

- synchronized replay/runner timing is rejected as a clean performance
  comparator because it slows HF into the GGUF band;
- graph-partition debug dumps still show GGUF-specific materialization and
  generated-body differences after launch shape, fastpath activation, method
  selection, embedding-dispatch surgery, and graph partitioning have all been
  tested.

### Promote / Reject

Promote:

- existing timing artifacts as a warning against synchronized replay timing;
- dense rank-0 debug-dump review as evidence that GGUF still takes a different
  generated graph/materialization route from the aligned HF text-only control;
- future lower-level work on GGUF-loaded weight layout, generated-body
  structure, memory movement, and reduced completed-work comparators.

Reject:

- using the replay-timing HF run as the clean HF control;
- another replay-timing or no-sync timing run without a lower-level source
  change;
- interpreting GGUF embedding/index-select residue as a standalone fix after
  embedding qweight removal failed to promote.

### Next

The next useful experiment should be source-level and narrower than another
full ladder. Candidate directions:

- a reduced completed-work comparator that preserves the clean HF control and
  swaps only GGUF-loaded weights or layout;
- lower-level inspection of GGUF-loaded F16 tensor layout around the linears
  that still become `rocm_unquantized_gemm`;
- memory-movement / replay-boundary analysis that does not synchronize every
  replay and collapse HF throughput;
- generated-body diffing around residual/norm/projection boundaries instead of
  another launch-flag change.

## GGUF-198: External Qwen3.6 GGUF Compatibility Reference Review

### Scope

Reviewed `Kausik-A/qwen3.6-27b-mi50-vllm` at public commit `61b273d` as a
source-compatibility reference. No benchmark run was launched and no container
was modified.

The project is a small Docker/bind-mount patch bundle over
`aiinfos/vllm-gfx906-mobydick`. It targets `unsloth/Qwen3.6-27B-GGUF` on a
single MI50/eGPU, ROCm 6.3, Q6 GGUF, FP16, 4096 context, and
`--enforce-eager`.

### Useful Findings

- It confirms the text-only Qwen3.5/3.6 GGUF plumbing points: registry entries
  for `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM`, `qwen35` architecture
  aliasing, M-RoPE stripping for text-only use, and `ssm_dt.bias` mapping to
  `linear_attn.dt_bias`.
- It carries MoE GGUF tensor-name aliases for Qwen3.5-style expert tensors:
  `ffn_down_exps`, `ffn_gate_exps`, `ffn_up_exps`, and `ffn_gate_inp`.
- It wires `quant_config` through `VocabParallelEmbedding` and
  `ParallelLMHead`, and reshapes GGUF GDN `conv1d.weight` from `[C, K]` to
  `[C, 1, K]`.
- It extends `MergedColumnParallelLinear.weight_loader` to accept tuple shard
  IDs so a pre-fused GGUF GDN tensor can be split across Q/K/V/Z shards.
- It records a concrete correctness warning: if text-only Qwen still carries
  `mrope_section` / `mrope_interleaved`, RoPE can take a multimodal path and
  produce garbage output.

### Outcome

Promote as compatibility and source-orientation material only. Do not promote
its launch profile or performance assumptions into the release path:

- single-GPU/eGPU, ROCm 6.3, Q6 GGUF, 4096 context, and eager execution are not
  comparable to the `.20` TP8/TP4 full-BAR/P2P-on ROCm 7.2 release lane;
- our accepted HF text-only comparator already rejects the text-only wrapper as
  the dense performance gap by reaching strict `70.353`, c1_2000 `70.978`, and
  c1_10000 `66.428` backend TPS;
- our current GGUF work has already tested and rejected wrapper/class,
  cache/page geometry, embedding-only dispatch, GDN custom-op body timing,
  exact HF scheduler shape, forced unquantized F16 method selection, and graph
  partitioning as standalone fixes.

### Promote / Reject

Promote:

- the tensor-name alias and M-RoPE stripping checks as compatibility guardrails;
- the fused QKVZ tuple-loader pattern as a concrete source area to compare
  against our dense GGUF loader;
- the warning that coherent warmup output is a tokenizer/config/RoPE validation
  signal before strict benchmarking.

Reject:

- importing the single-GPU/eager launch profile into the release reproduction
  path;
- treating text-only CausalLM routing as the remaining dense TPS lever;
- treating the repo as MoE TP4 performance evidence, since it does not test our
  TP4 full-BAR/P2P-on release lane.

### Next

Continue below compatibility plumbing:

- inspect whether all F16 GGUF tensors that should become normal unquantized
  weights fully bypass GGUF `qweight`/custom-op graph surfaces;
- compare pre-fused GDN tuple loading and materialization against our current
  dense GGUF source bundle;
- build a reduced completed-work comparator before another full benchmark
  ladder.

## GGUF-199: Dense Generated-Body Static And Standalone Microbench Boundary

### Purpose

Use the existing `.20` debug-dump artifacts to compare the current dense GGUF
generated body against the aligned HF text-only comparator before launching
another full model server. This was intended to answer whether the next source
probe should be another embedding/materialization patch, a generated-kernel
microbench, or a lower-level forward-context reproducer.

### Static Artifact Inputs

Compared rank-0 debug dumps from:

- dense GGUF graph-partition probe
  `dense27b_gguf_tp8_graphpartition_probe_20260627T120119Z`;
- dense HF text-only comparator
  `dense27b_hf_tp8_textonly_compile_debugdump_env_20260627T114602Z`.

The static scan counted executable-looking non-comment lines in generated
Python files. These are not benchmark timings, but they are useful source
surface evidence.

| Marker | Dense GGUF graph-partition rank 0 | Dense HF text-only rank 0 |
| --- | ---: | ---: |
| Python files | `52` | `177` |
| Kernel files | `35` | `46` |
| `_apply_gguf_embedding` | `36` | `0` |
| `gguf_embedding` | `36` | `0` |
| `index_select` | `160` | `0` |
| `torch.ops.vllm.gdn_attention_core` | `864` | `288` |
| `torch.ops.vllm.rocm_unquantized_gemm.default` | `2496` | `195` |
| `torch.ops.vllm.all_reduce.default` | `13` | `13` |
| `torch.ops.gfx906_swiglu.interleaved` | `1280` | `513` |
| `torch.ops.aten.clone.default` | `1760` | `150` |
| `torch.ops.aten.copy_.default` | `0` | `50` |
| `torch.ops.aten.reshape.default` | `2352` | `183` |

The GGUF-only `index_select` hits were rotary cache lookups for full-attention
layers, not GGUF tensor loading. The GGUF `_apply_gguf_embedding` call was
present as actual generated work in the graph-partition dump, but this is not
sufficient by itself to explain the dense decode gap because `GGUF-189`
removed the embedding custom-op surface entirely and still produced only
strict `63.606`, c1_2000 `64.432`, and c1_10000 `60.691` backend TPS.

### Standalone Generated-Script Probe

Tried to execute the generated dense GGUF graph-partition script directly in a
short-lived official v0.2 image container with one visible GPU and no model
server. This was intentionally a local generated-code probe, not a benchmark
deployment.

Observed sequence:

- the release image entrypoint had to be overridden; otherwise it starts the
  normal vLLM server;
- a tiny Torch/ROCm sanity check succeeded under the override;
- importing `vllm.model_executor.layers.quantization.gguf` registered
  `torch.ops.vllm._apply_gguf_embedding`;
- the generated script then required a `tp:0` process group for
  `torch.ops.vllm.all_reduce`;
- stubbing all-reduce to identity allowed the script to advance to
  `torch.ops.vllm.gdn_attention_core`;
- importing `vllm.model_executor.models.qwen3_next` registered
  `gdn_attention_core`;
- the generated script then failed because `gdn_attention_core` requires a
  real vLLM `ForwardContext` with `no_compile_layers[layer_name]` pointing to a
  live GatedDeltaNet layer object.

### Outcome

Reject unchanged standalone generated-script microbenching as the next
performance comparator. It is not enough to run the dump file by itself:
`gdn_attention_core`, unified attention, and similar direct custom ops depend
on live vLLM forward context and model layer objects. Stubbing those ops would
remove the work we need to measure and would not be valid performance evidence.

Promote the static comparison as useful source-boundary evidence:

- dense GGUF graph-partition still carries real generated GGUF embedding
  custom-op work, but embedding removal has already failed to promote;
- the denser GGUF generated body shows much more clone/reshape/GDN/GEMM surface
  in the generated file than the HF text-only comparator, but these counts are
  structural source evidence, not timings;
- a valid reduced completed-work comparator must either construct the real
  vLLM forward context and no-compile GDN/attention layers, or use a lower-level
  profiler/marker route that observes the live server without changing the hot
  graph.

### Promote / Reject

Promote:

- static generated-body comparison as a safe pre-run filter;
- the requirement that any microbench comparator include real forward context
  for direct custom ops;
- the finding that all-reduce can be identity-stubbed only for isolated
  single-GPU generated-code diagnostics, not for release performance evidence.

Reject:

- repeating embedding-only qweight removal;
- treating generated-script execution with stubbed GDN as valid evidence;
- launching another full dense ladder without a source change below the
  generated-body / forward-context boundary.

### Next

Build the next source probe around one of these routes:

1. a reduced completed-work comparator that instantiates the real GDN
   `no_compile_layers` objects and forward context for GGUF versus HF;
2. a live-server, request-window marker/profiler route that does not synchronize
   every replay and collapse the HF control;
3. a narrower static/generated-body diff around the clone/reshape/GDN state
   movement that remains after embedding qweight removal.

## GGUF-200: Dense Graph Partition Plus Embed-QWeight Skip Full Ladder

### Purpose

Test whether combining the two previously isolated dense GGUF changes moves
the `.20` TP8 full-BAR/P2P-on release lane into the published HF/release TPS
band:

- Inductor graph partitioning from `GGUF-196`;
- `VLLM_QWEN35_GGUF_SKIP_EMBED_QWEIGHT=1` plus
  `VLLM_GGUF_EMBED_TOKENS_UNQUANT=1` from `GGUF-189`.

This run intentionally used the normal benchmark order: pre-measure warmups,
uncapped strict `c1_128`, `c1_2000`, then `c1_10000`.

### Configuration

- Host label: `.20`
- Image: published v0.2 ROCm7.2 Dense/MoE runtime image
- Model file: dense 27B F16 GGUF
- Profile shape: `dense27b_tp8_fullbar_p2pon`
- TP: `8`
- P2P: on
- `MAX_MODEL_LEN=131072`
- FP16
- async scheduling on
- graph capture on
- `use_inductor_graph_partition=true`
- begin-think proxy enabled for Qwen strict validation

Run directory:

`<validation-workspace>/runs/dense27b_gguf_tp8_graphpartition_skipembed_dot20_20260627T131645Z/v02-profile-runs/dense27b_tp8_fullbar_p2pon_20260627T132824Z_graphpartition_skipembed`

Startup reached health. Model load completed, graph compile completed, CUDA
graph memory profiling completed, KV cache was allocated for 131072-token
requests, and the API served requests.

### Results

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | `3861` | `52.555` | `63.747` | `73.465` | `stop` | `true` |
| `c1_2000` | `2000` | `44.493` | `64.587` | `44.951` | `length` | `false` |
| `c1_10000` | `10000` | `55.991` | `60.866` | `178.601` | `length` | `false` |

The eight warmups settled in the same dense GGUF decode band, with backend TPS
roughly `64.57` to `64.60` after the first compile-affected request.

### Outcome

Reject graph partition plus embed-qweight skip as a promotion path. The run is
coherent and strict-valid, but it remains far below:

- the v0.2.1 `.20` dense reproduction target, c1_10000 `65.645`;
- the aligned HF text-only comparator, c1_10000 `66.428`;
- the published v0.2 dense gate lane, c1_10000 `66.069`.

The combined result is essentially the same throughput class as the previous
dense GGUF attempts:

- graph partition alone: strict `63.795`, c1_2000 `64.556`, c1_10000
  `60.826`;
- embed-qweight skip alone: strict `63.606`, c1_2000 `64.432`, c1_10000
  `60.691`.

### Promote / Reject

Promote:

- the combined run as a clean negative result;
- the finding that the remaining dense GGUF gap is not explained by embedding
  qweight materialization alone;
- the finding that increasing generated-kernel partitioning plus removing the
  embedding qweight surface still does not close the GGUF/HF gap.

Reject:

- updating the release/reproduction path with this GGUF configuration;
- repeating graph-partition or embedding-only variants unchanged;
- treating the external single-GPU/eager Qwen3.6 GGUF deployment profile as a
  performance recipe for the TP8 release lane.

### Next

Continue below launch flags and wrapper selection. The next useful source work
is a lower-level comparison of GGUF-loaded F16 tensor layout, generated graph
memory movement, clone/reshape boundaries, or a reduced completed-work
forward-context reproducer that keeps real GDN/attention layer objects alive.

## GGUF-201 - Dense GGUF Post-Load Tensor Layout Diagnostic

### Question

Is the remaining dense GGUF throughput gap caused by non-contiguous or otherwise
unusual post-load tensor layout for the layer-0 GatedDeltaNet linears?

### Setup

- Host label: `.20`
- Run directory:
  `<validation-workspace>/runs/dense27b_gguf_tp8_layoutdiag_dot20_20260627T135511Z`
- Image: published v0.2 ROCm7.2 Dense/MoE runtime image
- Model file: dense 27B F16 GGUF
- Profile shape: `dense27b_tp8_fullbar_p2pon`
- TP: `8`
- P2P: on
- `MAX_MODEL_LEN=131072`
- FP16
- async scheduling on
- graph capture on
- `use_inductor_graph_partition=true`
- `VLLM_QWEN35_GGUF_SKIP_EMBED_QWEIGHT=1`
- `VLLM_GGUF_EMBED_TOKENS_UNQUANT=1`

The only source change was a local diagnostic extension in the GGUF loader that
printed dtype, shape, stride, contiguity, and data pointer for key layer-0
linear-attention tensors after `process_weights_after_loading(...)`.

### Observed Layout

For all eight TP ranks, the key layer-0 tensors were materialized as CUDA FP16
tensors with contiguous row-major layout:

| tensor | dtype | shape | stride | contiguous |
| --- | --- | --- | --- | --- |
| `linear_attn.in_proj_qkvz.weight` | `torch.float16` | `(2060, 5120)` | `(5120, 1)` | `True` |
| `linear_attn.conv1d.weight` | `torch.float16` | `(1280, 1, 4)` | `(4, 4, 1)` | `True` |
| `linear_attn.out_proj.weight` | `torch.float16` | `(5120, 768)` | `(768, 1)` | `True` |

The server reached health after the normal long startup/compile path. A short
smoke request returned coherent Qwen reasoning text and stopped only because the
probe capped generation at 64 tokens.

### Outcome

Reject simple post-load stride/contiguity mismatch as the dense GGUF throughput
root cause. The important GDN linears already enter runtime as contiguous FP16
CUDA tensors on every rank.

### Promote / Reject

Promote:

- the layout diagnostic as source-boundary evidence;
- the finding that the next dense GGUF investigation should move below simple
  tensor contiguity.

Reject:

- adding `.contiguous()` or reload-time tensor copying as an unchanged
  performance experiment for these GDN weights;
- rerunning this diagnostic unchanged before checking deeper request-window,
  graph replay, or memory-movement boundaries.

### Next

Focus on lower-level generated graph/replay behavior or a live request-window
profiler that can localize clone/reshape/all-reduce/GEMM boundaries without
synchronizing every replay and collapsing the HF control.

## GGUF-202 - Dense GGUF M-RoPE Config Restoration Probe

### Question

Is part of the dense GGUF throughput gap caused by the GGUF text config using
plain/default RoPE while the aligned HF text-only comparator keeps the upstream
Qwen3.6 M-RoPE fields?

### Setup

- Host label: `.20`
- Run directory:
  `<validation-workspace>/runs/dense27b_gguf_tp8_graphpartition_skipembed_mropeprobe_dot20_20260627T141330Z`
- Image: published v0.2 ROCm7.2 Dense/MoE runtime image
- Model file: dense 27B F16 GGUF
- Profile shape: `dense27b_tp8_fullbar_p2pon`
- TP: `8`
- P2P: on
- `MAX_MODEL_LEN=131072`
- FP16
- async scheduling on
- graph capture on
- `use_inductor_graph_partition=true`
- `VLLM_QWEN35_GGUF_SKIP_EMBED_QWEIGHT=1`
- `VLLM_GGUF_EMBED_TOKENS_UNQUANT=1`

A throwaway local config directory was created from the existing GGUF
`qwen36-27b-text-config-eosfix` config, adding the upstream HF
`mrope_interleaved=true` and `mrope_section=[11,11,10]` fields. No image,
container source, model file, or repo file was changed.

### Observed Behavior

The server accepted the config, loaded weights, completed graph compilation,
captured graphs, and reached `/health`.

The first short chat request failed with HTTP 500. The engine-side failure was:

```text
RuntimeError: Worker failed with error 'M-RoPE support is not implemented.'
```

The stack trace reached `_calc_mrope_positions(...)` and asserted that request
M-RoPE positions were absent. This means config-only M-RoPE restoration flips
the model path into M-RoPE mode, but the current GGUF/text-only request path
does not supply the per-request M-RoPE position tensors needed to execute it.

### Outcome

Reject config-only M-RoPE restoration as a benchmark path. It is not a valid
performance experiment because the server cannot serve a request in this mode.

### Promote / Reject

Promote:

- the source boundary that the HF comparator and GGUF config differ on M-RoPE
  fields;
- the finding that a usable M-RoPE GGUF path would require request/scheduler
  plumbing, not only a JSON config change;
- the external `Kausik-A/qwen3.6-27b-mi50-vllm` repo's M-RoPE stripping as a
  compatibility clue, not a performance recipe.

Reject:

- forcing M-RoPE fields into the current GGUF text config without adding
  request-position support;
- treating the HF comparator's M-RoPE-generated graph surface as directly
  reachable through current GGUF launch flags.

### Next

Do not repeat M-RoPE config-only runs. If M-RoPE remains interesting, the next
source task is to identify how the HF request path produces
`req.mrope_positions` and whether that can be implemented safely for GGUF
text-only requests. Otherwise continue below the current plain-RoPE GGUF path
with live request-window profiling, clone/reshape memory-movement analysis, or
a reduced forward-context comparator.

## GGUF-203 - Dense GGUF LM-Head Unquantized Logits Path

### Question

Is the remaining dense GGUF decode gap partly outside the captured model graph,
in the logits processor calling the GGUF `lm_head` quant method instead of the
normal unquantized embedding / GEMM method?

### Setup

- Host label: `.20`
- Run directory:
  `<validation-workspace>/runs/dense27b_gguf_tp8_lmheadunquant_dot20_20260627T142800Z`
- Benchmark directory:
  `<validation-workspace>/runs/dense27b_gguf_tp8_lmheadunquant_dot20_20260627T142800Z/benchmarks/dense27b_tp8_fullbar_p2pon_20260627T143511Z`
- Image: published v0.2 ROCm7.2 Dense/MoE runtime image
- Model file: dense 27B F16 GGUF
- Profile shape: `dense27b_tp8_fullbar_p2pon`
- TP: `8`
- P2P: on
- `MAX_MODEL_LEN=131072`
- FP16
- async scheduling on
- graph capture on
- `use_inductor_graph_partition=true`
- `VLLM_QWEN35_GGUF_SKIP_EMBED_QWEIGHT=1`
- `VLLM_GGUF_EMBED_TOKENS_UNQUANT=1`
- `VLLM_QWEN35_GGUF_LM_HEAD_UNQUANT=1`

The source change was intentionally narrow: after the dense GGUF `lm_head`
F16 weight was materialized, the local Qwen3.5 loader switched only
`self.lm_head.quant_method` to `UnquantizedEmbeddingMethod()`. The goal was to
leave the model body, graph partitioning, and embedding path unchanged while
routing final logits through the normal unquantized logits path.

Before the benchmark ladder, a short smoke request returned coherent Qwen-style
reasoning text and stopped only because the probe capped generation at 64
tokens.

### Benchmark Result

The normal release benchmark fixture was used:

- eight pre-measure 2000-token warmups through the begin-think proxy;
- `c1_128` uncapped strict;
- `c1_2000`;
- `c1_10000`.

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | 3897 | 56.308 | 70.350 | 69.209 | stop | True |
| `c1_2000` | 2000 | 49.395 | 71.396 | 40.490 | length | False |
| `c1_10000` | 10000 | 61.559 | 66.842 | 162.446 | length | False |

### Outcome

Promote for dense GGUF. The unquantized `lm_head` logits path is the first
dense GGUF source change that crosses the published dense c1_10000 gate under
the normal warmup / strict / fixed-token ladder.

It beats:

- published v0.2 dense: strict `69.514`, c1_2000 `70.347`, c1_10000 `66.069`;
- v0.2.1 `.20` reproduction: strict `68.962`, c1_2000 `69.914`, c1_10000
  `65.645`;
- aligned HF text-only comparator on c1_10000: `66.428`.

### Promote / Reject

Promote:

- unquantizing only the GGUF `lm_head` logits path after F16 materialization;
- retaining the graph-partition plus embedding-qweight-skip dense GGUF body;
- the normal warmup -> strict -> c1_2000 -> c1_10000 fixture as the decision
  gate for dense GGUF work.

Reject:

- repeating embedding-only qweight removal as the dense throughput lever;
- config-only M-RoPE restoration;
- treating the external single-GPU/eGPU GGUF deployment profile as the
  performance explanation.

### Next

Port the same narrow logits-path idea into the MoE GGUF TP4 profile only if the
MoE source path still routes `lm_head` through GGUF quant methods after F16
materialization. Use the same normal benchmark ladder and compare against the
clean HF MoE TP4 control before considering release documentation changes.

## GGUF-204 - MoE GGUF TP4 LM-Head Unquantized Logits Path

### Question

Does the dense GGUF `lm_head` logits-path fix also close the Qwen3.6
35B-A3B MoE TP4 GGUF throughput gap?

### Setup

- Host label: `.20`
- Run directory:
  `<validation-workspace>/runs/moe35b_gguf_tp4_lmheadunquant_dot20_20260627T145000Z`
- Benchmark directory:
  `<validation-workspace>/runs/moe35b_gguf_tp4_lmheadunquant_dot20_20260627T145000Z/benchmarks/moe35b_tp4_fullbar_p2pon_20260627T145556Z`
- Image: published v0.2 ROCm7.2 Dense/MoE runtime image
- Model file: Qwen3.6 35B-A3B F16 GGUF
- Profile shape: `moe35b_tp4_fullbar_p2pon`
- TP: `4`
- P2P: on
- `MAX_MODEL_LEN=131072`
- FP16
- async scheduling on
- graph capture on
- release TP4 MoE fastpath enabled
- `use_inductor_graph_partition=true`
- `VLLM_GFX906_GGUF_FORCE_UNQUANT_MOE=1`
- `VLLM_QWEN35_GGUF_LM_HEAD_UNQUANT=1`

The patch bundle was copied from the prior exact-HF-shape graph-partition MoE
GGUF run and changed only `Qwen3_5ForCausalLMBase.load_weights(...)`: after the
GGUF weights load, `self.lm_head.quant_method` is switched to
`UnquantizedEmbeddingMethod()` when the env gate is enabled.

The server reached health. The `GGUF_QWEN35_LM_HEAD_UNQUANTIZED` marker fired
on all four TP ranks. A short smoke request produced coherent Qwen-style
reasoning text and stopped only because the probe capped generation.

### Benchmark Result

The normal release benchmark fixture was used:

- eight pre-measure 2000-token warmups through the begin-think proxy;
- `c1_128` uncapped strict;
- `c1_2000`;
- `c1_10000`.

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | 3574 | 81.218 | 82.024 | 44.005 | stop | True |
| `c1_2000` | 2000 | 82.556 | 84.061 | 24.226 | length | False |
| `c1_10000` | 10000 | 73.994 | 74.230 | 135.147 | length | False |

### Outcome

Reject as a MoE GGUF performance fix. The result is effectively unchanged from
the prior exact-HF-shape graph-partition GGUF MoE band and remains far below
the clean HF MoE TP4 control.

Reference comparisons:

- prior MoE GGUF exact-HF-shape / fastpath band: strict `82.543`, c1_2000
  `84.195`, c1_10000 `74.314`;
- prior MoE GGUF graph-partition exact-HF-shape band: strict `82.412`,
  c1_2000 `84.054`, c1_10000 `74.228`;
- clean HF MoE TP4 control: strict `114.187`, c1_2000 `116.623`, c1_10000
  `109.748`.

### Promote / Reject

Promote:

- the finding that MoE GGUF reaches a strict-valid but low-throughput plateau
  under this exact-HF-shape TP4 path;
- the evidence that the dense `lm_head` logits-path lever does not transfer as
  the MoE bottleneck fix.

Reject:

- `lm_head` unquantization as a standalone MoE GGUF promotion path;
- rerunning the same MoE lm-head-unquant profile unchanged;
- changing release claims based on this MoE GGUF result.

### Next

Continue MoE GGUF below the logits path. The active source target remains expert
routing / MoE fastpath layout, memory movement around expert postprocess, or a
reduced completed-work comparator against the clean HF MoE TP4 control.

## GGUF-205 - MoE GGUF Versus HF Static Graph Marker Comparison

### Question

Can existing debug dumps point to the next MoE GGUF source boundary without
rerunning a perturbing profiler?

### Setup

Compared existing rank-0 debug dump artifacts on `.20`:

- HF MoE:
  `<validation-workspace>/runs/moe35b_hf_tp4_broad_graph_debugentry_dot20_20260627T013947Z/debug_dump/rank_0_dp_0`
- current GGUF MoE lm-head-unquant:
  `<validation-workspace>/runs/moe35b_gguf_tp4_lmheadunquant_dot20_20260627T145000Z/debug_dump/rank_0_dp_0`
- previous GGUF MoE graph-partition exact-HF-shape:
  `<validation-workspace>/runs/moe35b_gguf_tp4_graphpartition_exact_hfshape_dot20_20260627T122551Z/debug_dump/rank_0_dp_0`

The comparison was static grep/count analysis only. No container was launched
and no benchmark request was run.

### Observed Counts

| artifact | `.py` files | generated kernel files | pre-grad files |
| --- | ---: | ---: | ---: |
| HF broad graph debug | 61 | 14 | 41 |
| GGUF lm-head-unquant | 63 | 46 | 1 |
| GGUF graph-partition exact-HF-shape | 63 | 46 | 1 |

Marker counts:

| marker | HF broad | GGUF lm-head-unquant | GGUF graph-partition |
| --- | ---: | ---: | ---: |
| `gdn_attention_core` | 360 | 1117 | 1117 |
| `moe_forward_shared` | 951 | 1736 | 1736 |
| `moe_forward` | 1231 | 2416 | 2416 |
| `rocm_unquantized_gemm` | 1257 | 4023 | 4023 |
| `vllm.all_reduce` | 1204 | 3123 | 3123 |
| `_qwen35_effective_weight` | 0 | 909 | 909 |
| `_apply_gguf_embedding` | 0 | 0 | 0 |
| `gguf_embedding` | 0 | 0 | 0 |
| `index_select` | 0 | 0 | 0 |

### Outcome

Promote as source-boundary evidence only. The GGUF graph surface has materially
more GDN, MoE, GEMM, and all-reduce references than the available HF debug dump,
and it contains GGUF-only `_qwen35_effective_weight` references. That is a real
structural difference worth following.

Reject as proof of root cause because the HF artifact is a broad graph debug
entrypoint dump, not a freshly generated exact-HF-shape / graph-partition
counterpart from the same comparison run.

### Promote / Reject

Promote:

- static graph marker comparison as a low-perturbation next-step tool;
- `_qwen35_effective_weight` and expanded GGUF graph surfaces as the next source
  area to inspect;
- generating or locating a truly matched HF graph dump before claiming the
  marker count difference explains throughput.

Reject:

- treating marker counts as timings;
- claiming the GGUF-vs-HF MoE throughput gap is solved or fully explained by
  this static scan;
- running another broad profiler before narrowing the graph-source difference.

### Next

Inspect `_qwen35_effective_weight` use in the GGUF MoE source path and compare
it to the HF loaded-weight path. If it is only debug/materialization residue,
reject it quickly. If it introduces per-token effective-weight reconstruction or
extra graph-visible work, build a narrow source patch to materialize the same
weight view once after load.

## GGUF-206 - MoE GGUF Base RMSNorm Alias With Load-Time Norm Offset

### Question

Does the GGUF-only `_qwen35_effective_weight` helper explain the MoE GGUF TP4
throughput gap after load-time norm-offset correction is already enabled?

### Setup

Host `.20`, native MoE TP4 full-BAR/P2P-on lane.

Copied the current MoE GGUF lm-head-unquant patch bundle:

`qwen35moe-loadtime-norm-once-qknormclass-paramaudit-lmheadunquant-20260627`

to:

`qwen35moe-loadtime-norm-once-qknormclass-base-rmsnorm-lmheadunquant-20260627`

and changed only the local Qwen3.5 RMSNorm definition:

```python
Qwen3_5RMSNorm = _Qwen3_5GemmaRMSNorm
```

All other release-like settings stayed fixed:

- image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`;
- model: Qwen3.6 35B-A3B F16 GGUF;
- profile: `moe35b_tp4_fullbar_p2pon`;
- TP4, FP16, P2P-on, async scheduling, graph capture;
- `MAX_MODEL_LEN=131072`;
- exact-HF scheduler shape:
  `--max-num-seqs 1 --max-num-batched-tokens 1024`;
- `--enable-prefix-caching --mamba-cache-mode align --language-model-only`;
- `use_inductor_graph_partition=true`;
- release TP4 MoE fastpath overlay;
- forced unquantized F16 expert path;
- load-time norm-offset fix enabled;
- runtime norm-offset fix disabled;
- `lm_head` unquantized logits path enabled.

Run directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_base_rmsnorm_lmheadunquant_dot20_20260627T153000Z`

Benchmark directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_base_rmsnorm_lmheadunquant_dot20_20260627T153000Z/benchmarks/moe35b_tp4_fullbar_p2pon_20260627T151630Z`

### Observations

- Startup reached health.
- A short smoke request produced coherent Qwen reasoning text.
- `GGUF_QWEN35_LM_HEAD_UNQUANTIZED` fired on all four TP ranks.
- Static debug dump check found `_qwen35_effective_weight` count `0`.
- Container log recorded `24` MoE fastpath layout rejections. The runtime
  shape/layout line showed GGUF expert tensors with:
  - `w1_shape=(256, 256, 2048)`;
  - `w1_stride=(557056, 2176, 1)`;
  - `w2_shape=(256, 2048, 128)`;
  - `w2_stride=(262144, 128, 1)`.

### Benchmark Result

Normal benchmark ladder: warmups, uncapped strict, `c1_2000`, `c1_10000`.

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | 3958 | 81.077 | 81.808 | 48.818 | stop | True |
| `c1_2000` | 2000 | 82.895 | 84.403 | 24.127 | length | False |
| `c1_10000` | 10000 | 74.271 | 74.510 | 134.642 | length | False |

### Outcome

Reject as a MoE GGUF performance fix. Removing the runtime RMSNorm wrapper does
remove the `_qwen35_effective_weight` graph marker, but it does not move the
MoE GGUF throughput class. The run remains far below the clean HF MoE TP4
control band of roughly strict `114`, `c1_2000` `116`, and `c1_10000` `109`
backend TPS.

### Promote / Reject

Promote:

- base RMSNorm aliasing as a harmless cleanup candidate only if needed later;
- the fastpath-layout rejection signal as the strongest new source clue;
- the conclusion that `_qwen35_effective_weight` marker volume was not the
  primary MoE GGUF bottleneck.

Reject:

- rerunning this base-RMSNorm profile unchanged;
- treating static graph marker count reduction as sufficient proof of hot-path
  improvement;
- focusing further MoE GGUF work on final logits or no-op runtime norm wrapper
  cleanup before expert layout is addressed.

### Next

Move MoE GGUF source work to the expert tensor layout / fastpath boundary. The
current release fastpath expects the HF expert layout class, but the GGUF-loaded
F16 expert tensors still present strides that reject the fastpath. The next
candidate should compare HF and GGUF expert tensor shape/stride/storage after
load, then either materialize the GGUF experts into the fastpath-accepted layout
once at load time or teach the fastpath to accept the GGUF stride pattern
without adding per-token copies.

## GGUF-207 - MoE GGUF Batch TopK8 Fastpath Acceptance

### Question

Is the MoE GGUF TP4 gap mainly caused by falling out of the release `top_k=8`
fastpath when vLLM presents grouped token counts instead of a single token?

### Setup

Host `.20`, native MoE TP4 full-BAR/P2P-on lane.

Started from the `GGUF-206` base-RMSNorm / lm-head-unquant path and changed
only the mounted `fused_moe.py` fastpath overlay. The new copied overlay added
batch-aware Triton kernels:

- `_qwen_c1_topk8_batch_w1_act_kernel`;
- `_qwen_c1_topk8_batch_w2_reduce_kernel`.

The copied fastpath relaxed the prior `hidden_states.size(0) == 1` and
`topk_ids.size(0) == 1` checks to accept grouped token counts where
`topk_ids.size(0) == hidden_states.size(0)`.

All other release-like settings stayed fixed:

- image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`;
- model: Qwen3.6 35B-A3B F16 GGUF;
- profile: `moe35b_tp4_fullbar_p2pon`;
- TP4, FP16, P2P-on, async scheduling, graph capture;
- `MAX_MODEL_LEN=131072`;
- exact-HF scheduler shape:
  `--max-num-seqs 1 --max-num-batched-tokens 1024`;
- `--enable-prefix-caching --mamba-cache-mode align --language-model-only`;
- `use_inductor_graph_partition=true`;
- forced unquantized F16 expert path;
- load-time norm-offset fix enabled;
- runtime norm-offset fix disabled;
- `lm_head` unquantized logits path enabled.

Run directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_batch_fastpath_base_rmsnorm_lmheadunquant_dot20_20260627T153500Z`

Benchmark directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_batch_fastpath_base_rmsnorm_lmheadunquant_dot20_20260627T153500Z/benchmarks/moe35b_tp4_fullbar_p2pon_20260627T153543Z`

### Observations

- Startup reached health.
- A short smoke request produced coherent Qwen reasoning text.
- Fastpath acceptance was confirmed:
  - `fastpath_active=4`;
  - `fastpath_rejects=0`.
- The log showed grouped token counts being accepted, including `1024`, `16`,
  `6`, `2`, and `1`.

### Benchmark Result

Normal benchmark ladder: warmups, uncapped strict, `c1_2000`, `c1_10000`.

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | 5747 | 78.604 | 79.186 | 73.114 | stop | True |
| `c1_2000` | 2000 | 82.180 | 84.044 | 24.337 | length | False |
| `c1_10000` | 10000 | 73.899 | 74.194 | 135.319 | length | False |

### Outcome

Reject as a MoE GGUF performance fix. The batch fastpath removed the fastpath
rejection signal, but it did not move throughput out of the prior GGUF MoE
band. Strict-valid throughput regressed slightly, while c1_2000 and c1_10000
remained effectively unchanged.

Reference comparisons:

- `GGUF-206` base RMSNorm / lm-head-unquant: strict `81.808`, c1_2000
  `84.403`, c1_10000 `74.510`;
- clean HF MoE TP4 control: strict `114.187`, c1_2000 `116.623`, c1_10000
  `109.748`.

### Promote / Reject

Promote:

- grouped-token fastpath acceptance as a tested and working source change;
- the conclusion that prior fastpath rejection logs were not the primary MoE
  GGUF bottleneck;
- the need to inspect completed work inside the accepted path, not only its
  admission checks.

Reject:

- rerunning this batch-fastpath overlay unchanged;
- treating fastpath admission as equivalent to matching the HF MoE path;
- changing release claims based on this MoE GGUF result.

### Next

Move below fastpath admission. The next useful comparator should measure or
reduce completed work inside the MoE body: HF-vs-GGUF expert tensor values and
layout after load, generated kernel body differences, memory movement around
expert activation/reduction, and whether the GGUF expert layout forces worse
loads even when the same shape is admitted. Avoid another admission-only patch.

## GGUF-208 - External Qwen3.6 27B MI50 vLLM Bundle Review

Reviewed `Kausik-A/qwen3.6-27b-mi50-vllm` at commit `61b273d` as an external
GGUF compatibility reference.

### Setup

No container was launched and no benchmark was run. This was a source and
deployment-shape review only.

Repository shape:

- single-MI50 / eGPU-oriented deployment bundle;
- ROCm 6.3-era `aiinfos/vllm-gfx906-mobydick` base;
- GGUF dense target: `unsloth/Qwen3.6-27B-GGUF`;
- documented launch uses `--enforce-eager`, 4096 context, Q6 GGUF, and
  single-GPU assumptions.

Patch files reviewed:

- `patches/config.py`;
- `patches/gguf_loader.py`;
- `patches/linear.py`;
- `patches/qwen3_5.py`;
- `patches/registry.py`.

### Observations

Useful compatibility mechanisms:

- registers `Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`;
- maps GGUF short architecture names such as `qwen35` and `qwen35moe`;
- strips text-only M-RoPE fields when routing Qwen3.5 / Qwen3.6 through the
  CausalLM path;
- maps `ssm_dt.bias` to `linear_attn.dt_bias`;
- adds MoE expert GGUF aliases for gate/up/down tensors;
- passes GGUF `quant_config` into `embed_tokens` and `lm_head`;
- reshapes GGUF `conv1d.weight` from 2D to the 3D vLLM Conv1d shape;
- extends `MergedColumnParallelLinear.weight_loader` to accept tuple shard IDs
  for pre-fused GatedDeltaNet projections.

### Outcome

Promote as source compatibility reference. The repo independently confirms the
same class of Qwen3.5/Qwen3.6 GGUF loader and text-model issues that our source
work encountered.

Reject as a performance reproduction path for LocalAIServers benchmarks. Its
documented runtime target is a single-card / eGPU / eager / 4096-context shape,
not the `.20` TP8/TP4 full-BAR/P2P-on ROCm7.2 release lane.

### Promote / Reject

Promote:

- the compatibility checklist above as a sanity guide for future GGUF source
  patches;
- the `qwen35moe` alias and `ssm_dt.bias` mapping as already-aligned with our
  MoE GGUF source route;
- tuple-shard fused projection loading as a source area to audit carefully.

Reject:

- importing the deployment settings as benchmark settings;
- treating ROCm 6.3 / single-card / eager results as comparable to the
  published LocalAIServers ROCm7.2 full-BAR/P2P-on path;
- rerunning a launch-shape experiment based only on this bundle.

### Next

Keep using this repo as a compatibility checklist only. The active MoE gap is
already past parser/model registration and fastpath admission; next work should
stay below the admitted MoE call boundary and compare completed work against
the clean HF MoE TP4 control.

## GGUF-209 - MoE HF/GGUF Expert Layout And Value Audit

Compared the HF and GGUF MoE TP4 expert tensors after load/materialization on
the `.20` release lane.

### Setup

No benchmark requests were sent. This was a startup/load-stage audit only.

HF run directory:

`<validation-workspace>/runs/moe35b_hf_tp4_value_audit_dot20_20260627T160500Z`

GGUF run directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_value_audit_dot20_20260627T161500Z`

Audit inputs:

- image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`;
- profile shape: MoE TP4, FP16, P2P-on, release tuned MoE config;
- sampled layers: `0`, `20`, `39`;
- sampled experts: `0`, `7`, `127`, `255`;
- sampled stages: `after_super`, `after_rocm_padding`, `after_setup_kernel`;
- sampled tensors: `w13_weight` gate/up slices and `w2_weight` low/mid slices.

Both containers were stopped after the load-stage evidence was captured, and
VRAM returned to zero use.

### Observations

The existing layout audit showed identical HF and GGUF expert tensor layout
summaries:

- `before_super` / `after_super` `w13_weight` stride:
  `(524288, 2048, 1)`, contiguous;
- `after_rocm_padding` / `after_setup_kernel` `w13_weight` stride:
  `(557056, 2176, 1)`, non-contiguous ROCm-padded view;
- all audited `w2_weight` stages:
  `(262144, 128, 1)`, contiguous;
- each audit file contained `1280` layout rows.

The new sampled value audit produced `576` rows for HF and `576` rows for GGUF.
After ignoring TP-rank/device print order, the sampled value rows matched
exactly as a multiset.

### Outcome

Reject expert tensor shape, stride, contiguity, and sampled loaded values as the
primary explanation for the MoE GGUF throughput gap.

The GGUF MoE path remains far below the HF TP4 reference band, but this audit
shows that the gap is not explained by a simple expert-weight materialization
mistake in the audited regions.

### Promote / Reject

Promote:

- HF/GGUF expert tensor layout parity as measured evidence;
- HF/GGUF sampled expert value parity as measured evidence;
- the conclusion that the next source target must be below or around completed
  work inside the admitted MoE path.

Reject:

- repeating expert layout-only audits unchanged;
- materializing GGUF expert tensors again solely to fix shape/stride;
- assuming the prior fastpath layout-rejection logs identified the main cost.

### Next

Move to lower-level completed-work comparisons:

- generated kernel body and graph differences for the admitted MoE path;
- activation/reduction memory movement around `w13` / `w2` use;
- FusedMoE prepare/finalize and shared-expert work split;
- a reduced forward-context comparator that times or counts actual work without
  perturbing the full benchmark path.

## GGUF-210 - Matched HF/GGUF MoE Static Graph Dump

Generated a matched HF MoE TP4 static graph dump on `.20` using the same
release-like shape as the current GGUF MoE path.

### Setup

This was a compile/debug-dump diagnostic, not a benchmark run.

HF run directory:

`<validation-workspace>/runs/moe35b_hf_tp4_matched_graphdump_dot20_20260627T161707Z`

GGUF comparison directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_lmheadunquant_dot20_20260627T145000Z`

The HF diagnostic used the same broad release image, MoE TP4, FP16, P2P-on,
`MAX_MODEL_LEN=131072`, tuned MoE config folder, `--max-num-seqs 1`,
`--max-num-batched-tokens 1024`, prefix caching, aligned mamba cache mode,
language-model-only path, and `use_inductor_graph_partition=true`.

The HF diagnostic failed before health at the known hybrid KV layout assertion:

`Fail to determine whether the layout is (2, num_blocks, ...) or (num_blocks, 2, ...)`

The debug dump was still usable as static compile/source-surface evidence. It is
not performance evidence.

### Observations

With matched shape and graph partitioning, the HF and GGUF rank-0 dumps had the
same top-level source surface:

| Item | HF | GGUF |
| --- | ---: | ---: |
| top-level files | 117 | 117 |
| Python files | 63 | 63 |
| generated kernel files | 46 | 46 |
| pre-grad files | 1 | 1 |
| post-grad files | 1 | 1 |

Marker counts also matched for the hot path markers that previously looked
expanded in GGUF:

| Marker | HF | GGUF |
| --- | ---: | ---: |
| `gdn_attention_core` | 1117 | 1117 |
| `moe_forward_shared` | 1736 | 1736 |
| `moe_forward` | 2416 | 2416 |
| `rocm_unquantized_gemm` | 4023 | 4023 |
| `vllm.all_reduce` | 3123 | 3123 |

The remaining visible differences were:

- GGUF still had `909` references to `_qwen35_effective_weight`; prior
  GGUF-206 removed that helper without improving MoE throughput, so this remains
  rejected as the primary bottleneck.
- Only one generated kernel file differed by normalized source hash. The visible
  diff was dominated by layer-name strings such as
  `language_model.model.layers.N...` in HF versus `model.layers.N...` in GGUF.
- The `.best_config` file set matched at `34` files per path. Fourteen files
  were byte-identical. Twenty differed raw, but most differences were
  `time_taken_ms`. After removing timing noise, three files still selected
  different Triton configs: two `XBLOCK` swaps between `512` and `1024`, and one
  smaller kernel changing from `XBLOCK=128, num_warps=2` to
  `XBLOCK=256, num_warps=1`.

### Outcome

Reject the earlier static graph expansion theory as the main MoE GGUF
throughput explanation. The earlier HF dump used a different graph shape, so its
lower kernel/marker counts were not a fair comparison.

Promote the matched graph evidence as a source boundary: HF and GGUF are now
structurally close at static graph level, so the remaining MoE gap likely lives
in runtime keyed state, custom-op naming/cache behavior, runtime request-window
work, or a small number of tuned generated kernels rather than broad graph
inflation.

### Promote / Reject

Promote:

- matched HF/GGUF static graph file and marker-count parity;
- `.best_config` semantic differences as a narrow source lead;
- layer-name string differences passed through custom ops as a source lead;
- a runtime completed-work comparator as the next useful class of experiment.

Reject:

- rerunning broad unmatched HF-vs-GGUF graph count comparisons;
- treating `_qwen35_effective_weight` as the primary MoE bottleneck without new
  evidence;
- treating this failed HF diagnostic as performance evidence.

### Next

Investigate runtime differences that do not appear as broad static graph
surface changes:

- whether custom ops key caches or state by layer-name string;
- whether the three semantic `.best_config` differences correspond to hot
  decode kernels;
- whether a reduced forward-context comparator can measure completed work
  without changing the request path;
- whether forcing GGUF layer-name strings to match HF names changes runtime
  behavior before launching a full benchmark ladder.

## GGUF-211 - MoE GGUF HF-Layer-Name Alias Probe

Tested whether the remaining MoE GGUF TP4 throughput gap was caused by
runtime state or custom-op caches keyed by GGUF layer-name strings instead of
the HF-style `language_model.model.layers.N...` names.

### Setup

Host: `.20`

Run directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_hflayeralias_lmheadunquant_dot20_20260627T170500Z`

Benchmark directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_hflayeralias_lmheadunquant_dot20_20260627T170500Z/benchmarks/moe35b_tp4_fullbar_p2pon_20260627T164500Z`

Patch bundle:

`<validation-workspace>/experimental-patches/qwen35moe-hf-layername-alias-lmheadunquant-20260627T170500Z`

The run kept the release image, MoE TP4, FP16, P2P-on,
`MAX_MODEL_LEN=131072`, graph partition, prefix caching, aligned mamba cache
mode, tuned MoE config, lm-head unquantization, and the release benchmark
fixture. The only source change was a gated alias helper under
`VLLM_QWEN35_GGUF_HF_LAYERNAME_ALIAS=1` that renamed GDN and attention
`static_forward_context` keys from `model.layers.N...` to
`language_model.model.layers.N...` and updated the corresponding custom-op
`prefix` / `layer_name` fields. The probe intentionally did not rename
`FusedMoE.layer_name`, because that path is also tied to weight loading and
`static_all_moe_layers`.

The first benchmark attempt failed before useful work because the release
harness defaulted to the HF MoE model id while the GGUF server advertised
`Qwen3.6-35B-A3B-F16-GGUF-lmheadunquant`. The ladder was rerun with only
`MODEL` set to that served GGUF id.

### Observations

Startup succeeded. The log confirmed the alias helper was active on the GDN
and attention layers, and `GGUF_QWEN35_LM_HEAD_UNQUANTIZED` fired on all four
TP ranks.

The log also showed that the TopK8 fastpath still rejected grouped token shapes
such as `1024` and `2` but activated for single-token decode. That matches the
current lm-head-unquant GGUF MoE behavior and does not by itself explain a
promotion.

The normal benchmark fixture completed:

| case | backend decode TPS | finish | strict gate valid |
| --- | ---: | --- | --- |
| `c1_128_strict` | `82.488` | `stop` | `true` |
| `c1_2000` | `84.001` | `length` | `false` |
| `c1_10000` | `74.205` | `length` | `false` |

The strict response was coherent Qwen reasoning text and the strict parser gate
passed. This is therefore a performance rejection, not a correctness rejection.

For comparison, the prior lm-head-unquant MoE GGUF run landed at strict
`82.024`, c1_2000 `84.061`, and c1_10000 `74.230` backend TPS. The alias probe
is within noise and does not move the c1_10000 result toward the HF/release TP4
band around `109` backend TPS.

The test container was stopped and removed after the run.

### Outcome

Reject HF-layer-name aliasing as a MoE GGUF promotion path.

The layer-name string difference observed in the matched graph dump is real,
but changing the GDN and attention custom-op keys to HF-style names did not
improve the measured MoE TP4 throughput. This makes runtime custom-op cache
keying by those strings unlikely to be the primary cause of the gap.

### Promote / Reject

Promote:

- the exact served GGUF model id requirement for the release harness;
- strict-valid GGUF MoE correctness under the alias probe;
- the conclusion that GDN/attention layer-name keying is not the primary MoE
  throughput bottleneck.

Reject:

- repeating HF-layer-name alias probes unchanged;
- treating static layer-name string diffs as performance evidence without a
  runtime TPS improvement;
- widening the alias to `FusedMoE.layer_name` before proving it would not
  perturb weight loading or `static_all_moe_layers`.

### Next

Continue with lower-level completed-work analysis:

- identify which generated decode kernels correspond to the three semantic
  `.best_config` differences from GGUF-210;
- compare actual per-token runtime work inside MoE prepare/finalize,
  shared-expert work, and activation/reduction movement;
- build a reduced completed-work comparator against the clean HF MoE TP4
  control that preserves the real vLLM request path.

## GGUF-212 - MoE HF/GGUF Triton Best-Config Triage

Followed up on the three semantic `.best_config` differences reported by
GGUF-210.

### Setup

Compared the matched static graph dumps:

- HF:
  `<validation-workspace>/runs/moe35b_hf_tp4_matched_graphdump_dot20_20260627T161707Z/debug_dump`
- GGUF:
  `<validation-workspace>/runs/moe35b_gguf_tp4_lmheadunquant_dot20_20260627T145000Z/debug_dump`

Inspected the three config IDs that still differed after stripping timing noise:

- `35568a6238301f1158a611395eeff6e4b4aed8ef90183ab94517ffb45dbb9df9`
- `9ca0ecd74596078dc32e4972a37e675cbe179c3823ea9d34c054d5e64aeb7ac1`
- `bde31149a549f82a0422766cc1dd9670997c33b25939604ec346743401c79c30`

### Observations

The rank distribution was not a clean HF-versus-GGUF split.

For `35568...`, HF used `XBLOCK=1024` on ranks 0-2 and `XBLOCK=512` on rank 3,
while GGUF used `XBLOCK=512` on ranks 0 and 3 and `XBLOCK=1024` on ranks 1-2.

For `9ca0...`, HF used `XBLOCK=512` only on rank 0 and `XBLOCK=1024` on ranks
1-3, while GGUF used `XBLOCK=512` only on rank 1 and `XBLOCK=1024` on ranks
0, 2, and 3.

For `bde...`, HF used `XBLOCK=256,num_warps=1` only on rank 2 and
`XBLOCK=128,num_warps=2` on ranks 0, 1, and 3. GGUF used
`XBLOCK=256,num_warps=1` on ranks 0 and 3 and `XBLOCK=128,num_warps=2` on
ranks 1 and 2.

The `.best_config` files did not provide a direct stable source-file mapping in
the debug dump. Grepping the config IDs and Triton cache hashes only mapped
back to `.best_config` files, not to a specific generated kernel body.

### Outcome

Reject the three `.best_config` differences as a standalone promotion target
until there is stronger evidence that they map to a hot decode kernel and are
stable across ranks/runs.

The evidence is more consistent with rank-local autotune variability than a
deterministic HF/GGUF source difference. Forcing these configs would be a noisy
experiment and would not directly explain the large MoE gap.

### Promote / Reject

Promote:

- treating `.best_config` differences as weak leads unless they can be mapped
  to hot generated source and repeated consistently;
- checking release-shape parity before more low-level kernel forcing.

Reject:

- forcing the three Triton configs based only on one matched debug dump;
- treating rank-local autotune config differences as evidence of the main MoE
  GGUF bottleneck.

### Next

Compare the current GGUF launch shape with the clean HF MoE TP4 reproduction
shape. The clean HF TP4 run used the standard release deployment profile and
produced strict `114.187`, c1_2000 `116.623`, and c1_10000 `109.748` backend
TPS, while the current GGUF path has often used constrained debug settings such
as `--max-num-batched-tokens 1024`, prefix caching, mamba-cache alignment, and
graph partitioning. A release-shape GGUF run is the next lower-risk test before
more kernel work.

## GGUF-213 - MoE GGUF Release-Shape Scheduler Probe

Tested whether the current MoE GGUF gap was caused by the constrained debug
launch shape rather than the lower-level source path.

### Setup

Run:

`<validation-workspace>/runs/moe35b_gguf_tp4_release_shape_lmheadunquant_dot20_20260627T170000Z`

Benchmark:

`<validation-workspace>/runs/moe35b_gguf_tp4_release_shape_lmheadunquant_dot20_20260627T170000Z/benchmarks/moe35b_tp4_fullbar_p2pon_20260627T170600Z`

The run kept the same GGUF source bundle and lm-head unquantization path, but
changed the server launch to the cleaner release-shaped argument set:

- `--load-format gguf`
- HF tokenizer and text config path
- `--language-model-only`
- no explicit `--max-num-batched-tokens 1024`
- no explicit prefix caching
- no mamba-cache alignment override
- no Inductor graph partition override

The run still used the normal MoE TP4 full-BAR/P2P-on profile, FP16 GGUF
weights, `MAX_MODEL_LEN=131072`, and the standard warmups -> strict -> c1_2000
-> c1_10000 benchmark sequence.

### Observations

Startup reached health. The server used the default chunked-prefill shape with
`max_num_batched_tokens=2048`, `enable_prefix_caching=False`, and
`use_inductor_graph_partition=False`.

The release-shape capture logs still rejected the TopK8 fastpath for grouped
token counts and only admitted the one-token decode shape.

The normal benchmark ladder produced:

| case | backend decode TPS | finish | qwen gate valid |
| --- | ---: | --- | --- |
| `c1_128_strict` | `81.329` | `stop` | `true` |
| `c1_2000` | `85.178` | `length` | `false` |
| `c1_10000` | `75.106` | `length` | `false` |

Warmups after the first request settled around `85.1` backend TPS, which is
slightly better than the constrained lm-head-unquant MoE GGUF path but still
far below the clean HF/release MoE TP4 band.

### Outcome

Reject release-shape scheduler parity as sufficient to close the MoE GGUF
performance gap. The run is strict-valid, but it does not approach the current
HF/release TP4 reproduction numbers.

This result points away from launch-shape settings as the primary bottleneck.
The remaining gap is below that boundary: completed MoE work, shared/expert
execution, prepare/finalize, activation/reduction movement, graph replay,
request cadence, or HIP kernel mix/timing.

### Promote / Reject

Promote:

- using release-shape GGUF results as the current cleaner MoE GGUF performance
  baseline;
- focusing next work on non-perturbing decode-window evidence and lower-level
  MoE execution cost;
- treating correctness as good enough for performance triage, since strict
  completed with `qwen_gate_valid=true`.

Reject:

- repeating release-default scheduler-shape tests as a likely promotion path;
- treating prefix-caching, `max_num_batched_tokens=1024`, mamba-cache
  alignment, or graph-partition overrides as the main MoE GGUF throughput
  explanation.

### Next

Build a lower-overhead comparator for completed MoE work that does not collapse
the HF control. The useful boundary is now HIP/kernel mix, memory movement,
MoE prepare/finalize, shared-expert work, or graph replay/request cadence,
not another high-level launch setting.

## GGUF-214 - HF MoE TP4 Isolated-Cache Control Rejected

Tested whether a clean HF-weight MoE TP4 control could be relaunched with
per-run TorchInductor, Triton, and vLLM cache directories to produce comparable
static artifacts without perturbing the HF performance band.

### Setup

Host: `.20`

Run directory:

`<validation-workspace>/runs/moe35b_hf_tp4_cache_capture_dot20_20260627T172011Z`

Benchmark partial directory:

`<validation-workspace>/runs/moe35b_hf_tp4_cache_capture_dot20_20260627T172011Z/v02-profile-runs/moe35b_tp4_fullbar_p2pon_20260627T172706Z_cachecapture`

The run used the official v0.2 image, HF Qwen3.6 35B-A3B weights, TP4,
P2P-on, FP16, `MAX_MODEL_LEN=131072`, the normal begin-think benchmark harness,
and fresh per-run cache directories:

- `vllm_cache`
- `triton_cache`
- `torchinductor_root`

The intent was to preserve the high-band HF behavior while capturing a clean
cache tree for static comparison against the GGUF release-shape run.

### Observations

The container reached API health after normal weight load, compile, KV-cache
allocation, and CUDA graph capture. The first benchmark request spent about
`205.287` seconds in prefill/cache work, then decoded at `73.819` backend TPS.
The second warmup removed the first-request prefill cost but still decoded at
only `73.879` backend TPS.

The log showed this was the raw image entrypoint path, not the full
deploy-package path:

- the startup command used the image default port and broad capture shape;
- `max_num_batched_tokens=2048`;
- `enable_prefix_caching=False`;
- the vLLM process warned that `VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH` was an
  unknown core environment variable;
- tuned MoE config loading occurred, but the decode band still matched the low
  GGUF/HF diagnostic band rather than the published high-band reproduction.

The partial benchmark was stopped after warmup 2. The container was stopped and
removed, and the `.20` GPU lane returned to idle.

### Outcome

Reject isolated-cache raw-image HF launch as a valid high-band comparator.

This probe collapsed the HF control into the same approximate `73`-TPS
c1-style decode band as the low GGUF runs. It therefore cannot be used to
compare hot HF versus GGUF generated kernels or cache artifacts for the real
release-performance path.

### Promote / Reject

Promote:

- the requirement that future HF controls must be launched through the full
  deploy-package/runtime-patch path, not by directly running the raw image
  entrypoint;
- keeping per-run cache isolation only if the launch still reproduces the
  high-band HF MoE TP4 numbers;
- treating first-request cache generation as potentially valid only after
  steady-state warmups remain in the high band.

Reject:

- using this run's `torchinductor_root`, `triton_cache`, or `vllm_cache` as the
  HF baseline for GGUF comparison;
- repeating the same raw-image isolated-cache shape;
- interpreting the low HF band from this run as a model or host limitation.

### Next

The next HF-vs-GGUF comparator must preserve the full deploy-package path. If a
cache capture is needed, copy or parameterize the release deployment package so
that it mounts the same runtime patch bundle and keeps the same observed
high-band behavior before comparing generated artifacts. Otherwise continue
with lower-level GGUF-only probes below the already rejected launch-shape,
lm-head, expert-layout, and fastpath-admission boundaries.

## GGUF-215 - HF MoE TP4 Full Deploy-Package Path and Client-Path Probe

Host: `.20`

Run directory:

`<validation-workspace>/runs/moe35b_hf_tp4_deploypkg_cache_capture_dot20_20260627T173555Z`

The run copied the release deployment package into an isolated NVMe run
directory and launched the official v0.2 image through the package deploy path
instead of the raw image entrypoint. It used the native
`moe35b_tp4_fullbar_p2pon` profile, HF Qwen3.6 35B-A3B weights, TP4,
full-BAR/P2P-on, FP16, `MAX_MODEL_LEN=131072`, and the release runtime patch
bundle.

### Observations

The deploy-package path showed the Qwen C1 TopK8 MoE fastpath applying during
startup on the expected expert shape:

- `E=256`
- `N=256`
- `K=2048`
- `top_k=8`
- `dtype=torch.float16`

CUDA graph capture completed, the API reached health, and the backend accepted
direct requests. A minimal direct backend smoke decoded successfully, proving
the container and vLLM engine were live.

An attempted `run_chat_capture.py` warmup through the begin-think proxy wrote
the generated request JSON but never reached the vLLM engine. During the stall,
vLLM metrics showed:

- `num_requests_running=0`
- `num_requests_waiting=0`
- `prompt_tokens_total=0`
- `generation_tokens_total=0`

The same generated request replayed with `curl` through the begin-think proxy
completed successfully. It produced `2000` completion tokens from a `431` token
prompt in about `18` seconds wall time. The vLLM logger reported a steady
generation interval of `115.7` tokens/s during the request.

### Outcome

Promote the full deploy-package/runtime-patch path as the high-band HF control
shape. Reject the blocked `run_chat_capture.py` invocation as a client-path
failure, not a model, kernel, or GGUF/HF performance signal.

The profiling now separates three states:

- raw image entrypoint with isolated caches: low-band HF control, rejected;
- full deploy package with generated request replay: high-band HF behavior
  returns;
- current MoE GGUF path: still low-band despite correctness, expert-layout
  parity, launch-shape parity, and fastpath admission probes.

### Promote / Reject

Promote:

- launch HF comparators through the full deploy package before comparing
  generated artifacts;
- use direct request replay or a repaired benchmark client when the Python
  capture harness blocks before the backend receives a request;
- treat vLLM request counters and GPU activity as the first check for whether a
  run is actually decoding.

Reject:

- interpreting a stalled benchmark client as decode throughput;
- using raw-image isolated-cache HF artifacts for GGUF performance comparison;
- pursuing tokenizer, begin-think proxy, lm-head, norm alias, expert
  materialization, release-shape parity, or fastpath-admission as the primary
  remaining MoE GGUF gap.

### Next

The remaining useful source work is below the admitted MoE call path:
completed per-token work, shared/expert execution, prepare/finalize,
activation/reduction movement, graph replay, or HIP kernel mix/timing. For
benchmark controls, first confirm that the request reaches vLLM and that
generation counters move before collecting timing evidence.

## GGUF-216 - MoE GGUF ConditionalGeneration Wrapper Loader Probe

Host: `.20`

Run directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_wrappermapper_conditional_dot20_20260627T181000Z`

Patch directory:

`<validation-workspace>/experimental-patches/qwen35moe-wrappermapper-conditional-20260627`

### Hypothesis

The high-band HF MoE TP4 path resolves as
`Qwen3_5MoeForConditionalGeneration`, while the current serving GGUF path
resolves as `Qwen3_5MoeForCausalLM`. The remaining MoE GGUF gap might be tied
to wrapper-level model construction, static context, or graph/cache behavior
that is not visible in the matched source-surface counts.

### Method

The probe copied the current MoE GGUF patch bundle and tried to route the FP16
GGUF through the ConditionalGeneration wrapper while preserving the same
published v0.2 image, native `moe35b_tp4_fullbar_p2pon` profile, TP4,
full-BAR/P2P-on, `MAX_MODEL_LEN=131072`, release-style scheduler shape, and
model-specific GGUF env/overlays.

Changes made in the disposable patch tree:

- added a wrapper `hf_to_vllm_mapper` that maps plain text weights under
  `model.*` to `language_model.model.*`;
- imported `WeightsMapper` for the wrapper mapper;
- extended the GGUF loader's Qwen3.5-MoE model-type aliases from
  `qwen3_5_moe_text` to include `qwen3_5_moe`;
- allowed Qwen3.5-MoE vision configs that expose `depth` instead of
  `num_hidden_layers`.

### Observations

The first launch failed on a missing `WeightsMapper` import. After fixing that,
the loader recognized the GGUF model type only after the `qwen3_5_moe` alias
was added. The next launch reached the multimodal dummy-model construction and
failed on the vision depth attribute until the depth fallback was added.

The final launch progressed to worker model loading, then failed before health
with GGUF parameter-map rejection. The unmapped set included both vision
parameters and wrapper-prefixed language-model parameters. This failure occurs
inside the loader's internal `_get_gguf_weights_map(...)` path before the
model-class `load_weights(...)` mapper can repair names.

No serving smoke, warmup, strict, `c1_2000`, or `c1_10000` request completed
from this probe.

### Outcome

Reject this wrapper-loader attempt as implemented. It is not a performance
result and does not prove that the wrapper/causal split is irrelevant. It only
proves that adding a model-class mapper is too late for GGUF wrapper loading:
the loader's internal dummy-model map must first understand the wrapper shape.

### Promote / Reject

Promote:

- wrapper-versus-causal construction remains a plausible source boundary until
  a clean wrapper-loading probe reaches health;
- the failure location: `_get_gguf_weights_map(...)`, not the later model
  `load_weights(...)` call;
- keeping the next patch inside the GGUF loader map if wrapper parity is tested
  again.

Reject:

- treating this failed launch as MoE throughput evidence;
- rerunning the same class-level mapper unchanged;
- using a multimodal wrapper dummy map without either filtering unused vision
  parameters or explicitly translating wrapper-prefixed language names.

### Next

If wrapper parity is still worth testing, build the next disposable patch at the
loader map boundary. The cleanest route is likely to construct a text-only
Qwen3.5-MoE dummy map for GGUF tensor names, then translate mapped text names
to the wrapper's actual `language_model.model.*` parameter names before the
missing-parameter check. If that reaches health, run the normal warmups ->
`c1_128` uncapped strict -> `c1_2000` -> `c1_10000` ladder before making any
performance claim.

## GGUF-217 - MoE GGUF ConditionalGeneration Wrapper Loader Benchmark

Host: `.20`

Run directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_wrappermapper_conditional_dot20_20260627T181000Z`

Benchmark directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_wrappermapper_conditional_dot20_20260627T181000Z/benchmarks/moe35b_tp4_fullbar_p2pon_20260627T183100Z`

### Hypothesis

After the `GGUF-216` loader-map failure, the remaining wrapper test was to make
the ConditionalGeneration route actually serve and then measure it. If the
MoE GGUF gap was caused by CausalLM-versus-wrapper graph construction, a
wrapper-loaded text-only GGUF path should move toward the HF/release TP4 band.

### Method

The same disposable patch bundle was extended at the loader boundary:

- the GGUF loader maps Qwen3.5-MoE wrapper dummy names under
  `model.language_model.*`;
- unused visual / projector parameters are sideloaded for the text-only GGUF;
- `mm_proj` weight-type and weight-iterator loading is skipped only for the
  text-only `qwen3_5_moe` wrapper probe;
- `Qwen3_5MoeForConditionalGeneration.load_weights(...)` delegates text weights
  into the already-working `self.language_model.load_weights(...)` CausalLM
  loader after stripping the wrapper prefix.

The run preserved the same release image, FP16 GGUF file, native
`moe35b_tp4_fullbar_p2pon` profile, TP4, full-BAR/P2P-on, P2P enabled,
`MAX_MODEL_LEN=131072`, model-specific overlays, and normal benchmark sequence:
warmups -> `c1_128` uncapped strict -> `c1_2000` -> `c1_10000`.

### Observations

The patch got past the previous unmapped-parameter failure and the later
`mm_proj` sidecar assertion. The server reached `/health`. A direct small Qwen
smoke produced coherent text and stopped normally when given enough tokens.

Eight pre-measure warmups completed. Warmups settled around `85.1` backend TPS.
The full benchmark ladder produced:

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | 3427 | 82.436 | 83.140 | 41.572 | stop | True |
| `c1_2000` | 2000 | 83.810 | 85.061 | 23.864 | length | False |
| `c1_10000` | 10000 | 74.798 | 74.996 | 133.693 | length | False |

### Outcome

Reject ConditionalGeneration wrapper parity as the MoE GGUF promotion path.
This is now a performance rejection, not just a loader/startup rejection. The
wrapper route is coherent and strict-valid, but it remains in the same low GGUF
band as the release-shape CausalLM path and is far below the clean HF/release
TP4 reproduction band.

### Promote / Reject

Promote:

- the loader-map patch as evidence that a text-only GGUF can be forced through
  the Qwen3.5-MoE wrapper without garbage output;
- the benchmark result as a real wrapper-parity measurement;
- the conclusion that the remaining MoE GGUF gap is below wrapper selection.

Reject:

- further ConditionalGeneration wrapper-loader work as a likely TPS promotion
  route;
- treating wrapper/CausalLM construction, `mm_proj` sidecar handling, or
  high-level model architecture naming as the primary remaining MoE gap;
- running more wrapper variants unless a lower-level trace shows a specific
  wrapper-only hot-path difference.

### Next

Return to lower-level evidence inside the admitted MoE path: completed expert
work, shared-expert execution, prepare/finalize, activation/reduction movement,
graph replay, request cadence, or HIP kernel mix/timing. The next useful source
work should avoid Python timing inside compiled Qwen code and should first
prove profiler/marker artifact emission on a reduced or non-perturbing request
window.

## GGUF-218 - MoE HF/GGUF Compile-Cache Graph Marker Comparison

### Question

Does the remaining MoE GGUF throughput gap show up as a static compiled-graph
structure difference when compared against a high-band HF deploy-package
control?

### Setup

Compared rank-0 `computation_graph.py` files from:

- HF high-band deploy-package control:
  `<validation-workspace>/runs/moe35b_hf_tp4_deploypkg_cache_capture_dot20_20260627T173555Z`
- GGUF ConditionalGeneration wrapper benchmark:
  `<validation-workspace>/runs/moe35b_gguf_tp4_wrappermapper_conditional_dot20_20260627T181000Z`

Both paths used the same `.20` TP4 full-BAR/P2P-on lane, release image family,
Qwen3.6 35B-A3B MoE profile, and normal serving shape. The HF path was first
validated as high-band by replaying a generated request through the proxy and
observing a steady generation interval around `115.7` tokens/s.

### Observations

The marker counts matched for the major admitted path boundaries:

| marker | HF deploy-package graph | GGUF wrapper graph |
| --- | ---: | ---: |
| `all_reduce` | 486 | 486 |
| `moe_forward_shared` | 200 | 200 |
| `gdn_attention_core` | 120 | 120 |
| `moe_forward` | 40 | 40 |

The visible difference was in unquantized projection GEMMs:

| marker | HF deploy-package graph | GGUF wrapper graph |
| --- | ---: | ---: |
| `rocm_unquantized_gemm` | 320 | 440 |

The localized source difference was the Qwen3.5 GatedDeltaNet projection shape.
The HF release-style graph uses a fused qkv/z/b/a projection with output shape
`3088` and split sizes `[2048, 1024, 8, 8]`. The GGUF graph used split
projection: qkv/z output shape `3072` with split sizes `[2048, 1024]`, plus a
separate beta/alpha projection output shape `16`.

### Outcome

Promote this as a narrow source clue, not a fix. The extra `120`
`rocm_unquantized_gemm` references give a concrete candidate to test: fold
GGUF beta/alpha into the same fused `in_proj_qkvz` shape used by the HF release
path.

### Promote / Reject

Promote:

- the high-band HF deploy-package run as the current valid graph comparator;
- the static graph comparison as evidence that GGUF still differed at the
  GatedDeltaNet input-projection boundary;
- the fused qkv/z/b/a projection shape as a bounded follow-up probe.

Reject:

- treating the graph marker comparison alone as proof of the bottleneck;
- reusing the older dense fused-GDN rejection as sufficient evidence against
  the MoE-specific graph difference.

### Next

Run a MoE-specific fused qkv/z/b/a GGUF probe that changes only this projection
shape, then use the normal benchmark ladder to decide whether removing the
extra projection GEMMs moves the TP4 GGUF path toward the HF/release band.

## GGUF-219 - MoE GGUF Fused QKVZBA Projection Probe

### Question

If the split GGUF `in_proj_ba` path is folded into the HF-style fused
qkv/z/b/a projection, does MoE GGUF TP4 move toward the HF/release throughput
band?

### Setup

Created a disposable patch bundle on `.20` by copying the current best
CausalLM MoE GGUF patch:

`<validation-workspace>/experimental-patches/qwen35moe-fusedgdn-qkvzba-lmheadunquant-20260627`

The patch preserved the known correctness repairs and changed only the
GatedDeltaNet projection structure:

- deleted `in_proj_ba` after parent initialization;
- expanded `in_proj_qkvz` output sizes to include q/k/v/z/b/a;
- loaded `in_proj_b` and `in_proj_a` into `in_proj_qkvz` shard IDs `4` and
  `5`;
- updated packed module mappings so GGUF qkv/z/b/a load into the fused
  projection;
- kept the same release image, FP16 GGUF file, TP4 full-BAR/P2P-on lane,
  `MAX_MODEL_LEN=131072`, P2P-on settings, model-specific overlays, and normal
  benchmark ladder.

Run directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_fusedgdn_qkvzba_lmheadunquant_dot20_20260627T190000Z`

Benchmark directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_fusedgdn_qkvzba_lmheadunquant_dot20_20260627T190000Z/benchmarks/moe35b_tp4_fullbar_p2pon_20260627T190900Z`

### Observations

The patch compiled, loaded, reached `/health`, and returned coherent Qwen text
for a small direct smoke request. The normal benchmark ladder completed:

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | 3314 | 82.340 | 83.030 | 40.248 | stop | True |
| `c1_2000` | 2000 | 83.592 | 84.778 | 23.926 | length | False |
| `c1_10000` | 10000 | 74.601 | 74.789 | 134.046 | length | False |

The compiled rank-0 graph now matched the HF-style projection marker count:

| marker | fused-qkvzba GGUF graph |
| --- | ---: |
| `rocm_unquantized_gemm` | 320 |
| `moe_forward_shared` | 200 |
| `gdn_attention_core` | 120 |
| `all_reduce` | 486 |

This confirms that the experiment removed the extra GGUF projection GEMMs seen
in `GGUF-218`, but the throughput remained in the same low MoE GGUF band.

### Outcome

Reject fused qkv/z/b/a GatedDeltaNet projection as the MoE GGUF promotion
lever. The graph-level projection difference was real, but eliminating it did
not reproduce the HF/release TP4 band. The path stays far below the `.20`
reference TP4 result class around strict `114+`, c1_2000 `116+`, and c1_10000
`109+` backend TPS.

### Promote / Reject

Promote:

- the fused qkv/z/b/a patch as proof that GGUF can be made graph-closer to the
  HF release projection shape without breaking coherence;
- the marker comparison as proof that the extra projection GEMM count can be
  removed;
- the rejection boundary: extra GDN projection GEMMs are not the primary
  current MoE GGUF throughput gap.

Reject:

- further fused-GDN projection-only variants as likely promotion paths;
- treating static `rocm_unquantized_gemm` count parity as sufficient evidence
  of runtime parity;
- rerunning the same fused qkv/z/b/a ladder without a lower-level change.

### Next

Continue below the now-rejected graph-shape boundary: graph replay cadence,
completed MoE prepare/finalize, shared-expert work, activation/reduction
movement, HIP kernel mix/timing, or a reduced comparator that can isolate
request-time work without perturbing graph capture.

## GGUF-220 - MoE GGUF Fused QKVZBA Plus Base RMSNorm Static-Parity Probe

### Question

After `GGUF-219` removed the extra GDN projection GEMMs, does also removing the
remaining GGUF-only `_qwen35_effective_weight` RMSNorm wrapper references move
MoE GGUF TP4 into the HF/release band?

### Setup

Created a disposable patch bundle on `.20`:

`<validation-workspace>/experimental-patches/qwen35moe-fusedgdn-qkvzba-base-rmsnorm-lmheadunquant-20260627`

The patch copied the fused qkv/z/b/a path from `GGUF-219` and changed only the
RMSNorm wrapper shape:

- kept load-time norm offset handling;
- aliased `Qwen3_5RMSNorm` back to the base `GemmaRMSNorm`;
- removed the source-level `_qwen35_effective_weight` helper;
- kept the same release image, FP16 GGUF file, TP4 full-BAR/P2P-on lane,
  `MAX_MODEL_LEN=131072`, P2P-on settings, model-specific overlays, and normal
  benchmark ladder.

Run directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_fusedgdn_qkvzba_base_rmsnorm_dot20_20260627T193000Z`

Benchmark directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_fusedgdn_qkvzba_base_rmsnorm_dot20_20260627T193000Z/benchmarks/moe35b_tp4_fullbar_p2pon_20260627T193900Z`

### Observations

The patch compiled, loaded, reached `/health`, and returned coherent Qwen text
for a small direct smoke request. The normal benchmark ladder completed:

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | 3773 | 81.934 | 82.534 | 46.050 | stop | True |
| `c1_2000` | 2000 | 83.713 | 84.892 | 23.891 | length | False |
| `c1_10000` | 10000 | 74.684 | 74.872 | 133.897 | length | False |

The rank-0 compiled graph retained the HF-style marker counts and removed the
RMSNorm helper marker:

| marker | fused-qkvzba + base-RMSNorm GGUF graph |
| --- | ---: |
| `rocm_unquantized_gemm` | 320 |
| `moe_forward_shared` | 200 |
| `moe_forward` | 40 |
| `gdn_attention_core` | 120 |
| `all_reduce` | 486 |
| `_qwen35_effective_weight` | 0 |

### Outcome

Reject base RMSNorm plus fused qkv/z/b/a as the MoE GGUF promotion path.
Removing the remaining visible static graph marker difference did not move the
throughput out of the low GGUF band. This reinforces that the active gap is not
visible in the coarse Python graph marker counts we have been using.

### Promote / Reject

Promote:

- the patch as a clean static-parity control;
- the conclusion that `_qwen35_effective_weight` is not the bottleneck, even in
  combination with fused qkv/z/b/a;
- the boundary that further static graph shape edits need new evidence before
  another full ladder.

Reject:

- continuing to chase RMSNorm wrapper removal;
- repeating fused projection plus norm-alias combinations unchanged;
- treating Python-level graph marker parity as sufficient proof of runtime
  parity.

### Next

Move to runtime-level evidence. The remaining likely classes are graph replay
cadence, topk8/fused-MoE fastpath behavior, completed MoE prepare/finalize,
shared-expert work, activation/reduction movement, HIP kernel mix/timing, or a
reduced comparator that can isolate request-time work without changing the
release-like serving lane.

## GGUF-221 - MoE GGUF TP4 Unused q/k/v Split Removal

### Question

After `GGUF-220`, the normalized rank-0 compiled-graph diff against the
high-band HF deploy-package control showed one remaining non-comment operation
difference: GGUF still emitted an unused
`mixed_qkv.split([512, 512, 1024], dim=-1)` in each GatedDeltaNet layer. Does
moving that split behind the trace-only branch remove the last static graph
operation difference and promote MoE GGUF TP4 into the HF/release band?

### Setup

Created a disposable patch bundle on `.20`:

`<validation-workspace>/experimental-patches/qwen35moe-fusedgdn-qkvzba-nosplit-base-rmsnorm-lmheadunquant-20260627`

The patch copied `GGUF-220` and changed only the trace-only q/k/v unpack:

- kept fused qkv/z/b/a projection loading;
- kept base `GemmaRMSNorm` alias and load-time norm offset handling;
- moved the `mixed_qkv.split([q_size, k_size, v_size], dim=-1)` call inside
  `if trace_active`;
- kept the same release image, FP16 GGUF file, TP4 full-BAR/P2P-on lane,
  `MAX_MODEL_LEN=131072`, P2P-on settings, model-specific overlays, release
  MoE fastpath overlay, and normal benchmark ladder.

Run directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_fusedgdn_qkvzba_nosplit_base_rmsnorm_dot20_20260627T201500Z`

Benchmark directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_fusedgdn_qkvzba_nosplit_base_rmsnorm_dot20_20260627T201500Z/benchmarks/moe35b_tp4_fullbar_p2pon_20260627T194600Z`

### Observations

The patch compiled, loaded, reached `/health`, and completed the normal
benchmark ladder:

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | 4738 | 80.756 | 81.220 | 58.671 | stop | True |
| `c1_2000` | 2000 | 83.666 | 84.852 | 23.905 | length | False |
| `c1_10000` | 10000 | 74.657 | 74.845 | 133.946 | length | False |

The rank-0 compiled graph no longer contained the unused q/k/v split:

| check | result |
| --- | ---: |
| `mixed_qkv.split([512, 512, 1024])` occurrences | 0 |
| rank-0 graph lines | 8935 |
| rank-0 graph bytes | 899212 |

The release fastpath behavior stayed the same as the static-parity baseline:

| fastpath log class | count |
| --- | ---: |
| active | 4 |
| rejected | 204 |
| apply shape seen | 208 |

Grouped GGUF expert shapes still rejected the release fastpath on layout/shape,
while one-token shapes activated.

### Outcome

Reject unused q/k/v split removal as a MoE GGUF promotion path. Removing the
last normalized static graph operation difference did not move throughput out
of the low GGUF band.

### Promote / Reject

Promote:

- the no-split patch as the cleanest static graph parity control so far;
- the conclusion that HF/GGUF static graph operation parity is not sufficient
  evidence of runtime parity;
- runtime/kernel-level investigation as the required next boundary.

Reject:

- chasing remaining Python graph cleanup as the active MoE performance lever;
- repeating q/k/v split, fused projection, RMSNorm alias, or wrapper-parity
  probes without new lower-level evidence.

### Next

Move below the Python graph body. The useful next evidence should come from
graph-replay-visible timing, HIP/kernel mix, generated kernel body comparison,
or a reduced request-time comparator around expert/shared-expert execution,
activation/reduction movement, and the topk8/fused-MoE path.

## GGUF-222 - MoE GGUF TP4 AWQ Cache-Key Override Probe

### Question

The high-band HF deploy-package control and low-band GGUF no-split control
differed in `cache_key_factors.json`: the GGUF cache recorded
`VLLM_USE_TRITON_AWQ=true` while the HF cache did not. Does forcing
`VLLM_USE_TRITON_AWQ=0` in the GGUF no-split lane change the compiled cache
factor and move MoE GGUF TP4 toward the HF/release band?

### Setup

Started from the `GGUF-221` no-split static-parity patch bundle on `.20`:

`<validation-workspace>/experimental-patches/qwen35moe-fusedgdn-qkvzba-nosplit-base-rmsnorm-lmheadunquant-20260627`

The launch changed only the attempted AWQ cache-key override:

- release image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`
- FP16 GGUF:
  `Qwen3.6-35B-A3B-F16.gguf`
- profile: `moe35b_tp4_fullbar_p2pon`
- TP degree: `4`
- `MAX_MODEL_LEN=131072`
- P2P enabled via `NCCL_P2P_DISABLE=0`
- `VLLM_GFX906_GGUF_FORCE_UNQUANT_MOE=1`
- container environment contained `VLLM_USE_TRITON_AWQ=0`
- normal benchmark ladder: eight warmups, uncapped strict `c1_128`,
  `c1_2000`, then `c1_10000`

Run directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_nosplit_awqoff_dot20_20260627T195932Z`

Benchmark directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_nosplit_awqoff_dot20_20260627T195932Z/benchmarks/moe35b_tp4_fullbar_p2pon_20260627T200700Z`

### Observations

Startup reached `/v1/models`, loaded the model, completed graph capture, and
ran the normal release-style ladder. Warmups 5 through 8 were stable in the
same low GGUF MoE band: `84.587`, `84.599`, `84.577`, and `84.604` backend
TPS.

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | 3333 | 82.119 | 82.803 | 40.588 | stop | True |
| `c1_2000` | 2000 | 83.380 | 84.557 | 23.987 | length | False |
| `c1_10000` | 10000 | 74.423 | 74.608 | 134.367 | length | False |

The container environment did contain `VLLM_USE_TRITON_AWQ=0`, but all four
compiled rank cache-key files still recorded:

```text
"VLLM_USE_TRITON_AWQ": true
```

Source inspection explains the mismatch. In the copied release ROCm patch,
`verify_quantization(...)` logs the AWQ warning only under
`if quant == "awq" and not envs.VLLM_USE_TRITON_AWQ`, but the assignment
`os.environ["VLLM_USE_TRITON_AWQ"] = "1"` is outside that conditional. That
means the ROCm patch forces the flag during quantization verification instead
of respecting the launch-time `0` override.

The topk8 MoE fastpath behavior also remained unchanged from `GGUF-221`:

| fastpath log class | count |
| --- | ---: |
| active | 4 |
| rejected | 204 |
| apply shape seen | 208 |

Grouped GGUF expert shapes continued to reject the release fastpath for
`shape_or_layout`; one-token shapes still activated.

### Outcome

Reject this env-only AWQ-off probe as a MoE GGUF promotion path. It did not
flip the compiled vLLM cache factor and did not move throughput out of the low
GGUF band.

This does not fully reject the AWQ/cache-key hypothesis. It rejects only the
assumption that setting `VLLM_USE_TRITON_AWQ=0` in the environment is enough
for this GGUF MoE path. A real test of that hypothesis needs a source-level
ROCm patch change that stops the unconditional assignment and then proves the
compiled cache factor actually becomes false.

### Promote / Reject

Promote:

- the benchmark ladder as a clean repeat of the no-split static-parity lane;
- the observation that runtime environment and compiled cache factors can
  disagree for `VLLM_USE_TRITON_AWQ`;
- source-level cache-factor/quant-method inspection as the next boundary.

Reject:

- treating this env-only AWQ override as a valid AWQ-off experiment;
- repeating the no-split AWQ-off launch unchanged;
- continuing Python graph cleanup before checking lower-level quant/cache and
  runtime-kernel behavior.

### Next

Patch the copied ROCm hotfix so `VLLM_USE_TRITON_AWQ` is not unconditionally
forced to `1`, relaunch the same no-split GGUF MoE lane, and verify the
compiled cache factor before spending a full ladder. If the cache factor flips
to false, run the normal ladder and compare generated kernels / graph replay
against the high-band HF deploy-package control. If it still does not move,
continue into HIP/kernel-mix profiling around fused MoE expert/shared-expert
execution and activation / reduction movement.

## GGUF-223 - MoE GGUF TP4 Source-Level AWQ Cache-Factor False Probe

### Question

After `GGUF-222` showed that the environment-only AWQ override did not change
the compiled cache factor, does a source-level ROCm patch that actually leaves
`VLLM_USE_TRITON_AWQ=false` move MoE GGUF TP4 toward the HF/release band?

### Setup

Created a disposable patch bundle on `.20`:

`<validation-workspace>/experimental-patches/qwen35moe-nosplit-awqflagrespect-20260627`

The bundle copied the `GGUF-221` no-split static-parity patch and added a
top-level `rocm.py` override. The only intended source change was in
`verify_quantization(...)`: when
`VLLM_GFX906_SKIP_ROCM_AWQ_AUTOENABLE=1` is present, return before the release
patch forces `os.environ["VLLM_USE_TRITON_AWQ"] = "1"`.

The launch kept the same serving lane otherwise:

- release image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`
- FP16 GGUF:
  `Qwen3.6-35B-A3B-F16.gguf`
- profile: `moe35b_tp4_fullbar_p2pon`
- TP degree: `4`
- `MAX_MODEL_LEN=131072`
- P2P enabled via `NCCL_P2P_DISABLE=0`
- `VLLM_GFX906_GGUF_FORCE_UNQUANT_MOE=1`
- `VLLM_USE_TRITON_AWQ=0`
- `VLLM_GFX906_SKIP_ROCM_AWQ_AUTOENABLE=1`
- normal benchmark ladder: eight warmups, uncapped strict `c1_128`,
  `c1_2000`, then `c1_10000`

Run directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_nosplit_awqflagrespect_dot20_20260627T202100Z`

Benchmark directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_nosplit_awqflagrespect_dot20_20260627T202100Z/benchmarks/moe35b_tp4_fullbar_p2pon_20260627T202800Z`

### Observations

Startup reached `/v1/models`, loaded the model, and completed graph capture.
The patched ROCm path logged that Triton AWQ auto-enable was skipped for the
controlled probe. All four compiled rank cache-key files recorded:

```text
"VLLM_USE_TRITON_AWQ": false
```

Warmups 5 through 8 were stable but still low-band: `85.067`, `85.231`,
`85.210`, and `85.199` backend TPS.

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | 2946 | 83.148 | 83.940 | 35.431 | stop | True |
| `c1_2000` | 2000 | 83.983 | 85.177 | 23.814 | length | False |
| `c1_10000` | 10000 | 74.891 | 75.079 | 133.528 | length | False |

The topk8 MoE fastpath pattern remained unchanged:

| fastpath log class | count |
| --- | ---: |
| active | 4 |
| rejected | 204 |
| apply shape seen | 208 |

Grouped GGUF expert shapes continued to reject the release fastpath for
`shape_or_layout`; one-token shapes still activated.

### Outcome

Reject `VLLM_USE_TRITON_AWQ=false` as the main MoE GGUF promotion path. This
source-level probe proved the cache factor can be flipped to false, and it
produced only a marginal improvement over `GGUF-222`: c1_10000 moved from
`74.608` to `75.079` backend TPS, still far below the clean HF/release TP4
band around `109.7` backend TPS on c1_10000.

### Promote / Reject

Promote:

- the ROCm patch diagnosis from `GGUF-222`;
- the source-level method for flipping the cache factor to false;
- the conclusion that AWQ cache-factor cleanup is not the missing MoE GGUF
  throughput lever by itself.

Reject:

- further AWQ-cache-factor probes without new kernel-level evidence;
- treating the tiny c1_10000 lift as a promotion;
- continuing static graph/cache-factor cleanup as the main path to match MoE
  HF/release numbers.

### Next

Move to lower-level runtime evidence. The strongest remaining targets are:

- generated HIP/Triton kernel body differences between high-band HF and
  low-band GGUF;
- graph replay cadence and launch mix during steady decode;
- expert/shared-expert execution and activation/reduction movement;
- physical expert layout consumption inside the grouped topk8/fused-MoE path.

The grouped-shape release fastpath rejection remains visible in every low-band
GGUF MoE run and should be treated as a primary runtime/kernel suspect.

## GGUF-224 - MoE GGUF TP4 NCCL Path Cache-Factor Probe

### Question

Does matching the high-band HF cache factor for `VLLM_NCCL_SO_PATH` move the
MoE GGUF TP4 path toward the release/HF band?

The previous source-level cache comparison found that the high-band HF TP4
control and low-band GGUF TP4 probe matched on the main MoE environment knobs,
including `VLLM_USE_TRITON_AWQ=false`, activation chunking, MoE DP chunking,
DeepGEMM, shared-expert stream behavior, and symm-mem all-reduce. The remaining
environment cache-factor difference was:

```text
HF:   VLLM_NCCL_SO_PATH=""
GGUF: VLLM_NCCL_SO_PATH="/rccl-overlay/install/lib/librccl.so.1"
```

### Setup

Created a disposable `.20` run by copying the `GGUF-223` no-split/AWQ-false
launch and changing only the NCCL library path environment:

- removed the explicit `VLLM_NCCL_SO_PATH=/rccl-overlay/install/lib/librccl.so.1`
  export;
- kept the same release image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`;
- kept the same FP16 GGUF:
  `Qwen3.6-35B-A3B-F16.gguf`;
- kept profile `moe35b_tp4_fullbar_p2pon`;
- kept TP degree `4`;
- kept `MAX_MODEL_LEN=131072`;
- kept P2P enabled via `NCCL_P2P_DISABLE=0`;
- kept `VLLM_GFX906_GGUF_FORCE_UNQUANT_MOE=1`;
- kept `VLLM_USE_TRITON_AWQ=0`;
- kept `VLLM_GFX906_SKIP_ROCM_AWQ_AUTOENABLE=1`;
- used the normal ladder: eight warmups, uncapped strict `c1_128`, `c1_2000`,
  then `c1_10000`.

Run directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_nosplit_awqflagrespect_no_ncclpath_dot20_20260627T204804Z`

Benchmark directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_nosplit_awqflagrespect_no_ncclpath_dot20_20260627T204804Z/benchmarks/moe35b_tp4_fullbar_p2pon_20260627T204804Z_no_ncclpath`

### Observations

Startup reached `/v1/models`, loaded the model, completed mixed prefill/decode
and decode graph capture, and served the benchmark ladder. The compiled cache
factor changed as intended on all four ranks:

```text
"VLLM_NCCL_SO_PATH": ""
"VLLM_USE_TRITON_AWQ": false
```

Warmups 2 through 8 were stable in the same low GGUF band: about `82.47` to
`82.66` backend TPS after the first warmup.

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | 2993 | 80.680 | 81.425 | 37.097 | stop | True |
| `c1_2000` | 2000 | 81.502 | 82.634 | 24.539 | length | False |
| `c1_10000` | 10000 | 72.843 | 73.023 | 137.282 | length | False |

The topk8 MoE fastpath pattern remained unchanged:

| fastpath log class | count |
| --- | ---: |
| active | 4 |
| rejected | 204 |
| apply shape seen | 208 |

### Outcome

Reject `VLLM_NCCL_SO_PATH` cache-factor mismatch as the MoE GGUF promotion path.
The probe matched the HF cache-factor value for the NCCL path, but performance
did not move toward the clean HF/release TP4 band. The result was slightly worse
than `GGUF-223` on every measured tier.

### Promote / Reject

Promote:

- the cache-factor comparison as useful evidence;
- the conclusion that the remaining MoE GGUF gap is not explained by
  `VLLM_NCCL_SO_PATH`;
- continued focus on lower-level runtime behavior after graph capture.

Reject:

- further NCCL-path env probes without new all-reduce or kernel-level evidence;
- treating cache-factor parity alone as enough to explain the HF/GGUF split;
- repeating full ladders for env-only changes that do not alter completed work.

### Next

The active MoE GGUF gap remains below config/cache-factor parity. Continue with
lower-level evidence around:

- steady decode HIP/Triton kernel mix and timing;
- grouped expert execution in the standard fused-MoE path;
- activation/reduction memory movement around expert postprocess;
- completed-work comparators against the high-band HF TP4 control.

## GGUF-225 - MoE TopK8 Fastpath Debug-Log Comparability Audit

### Question

Can the previously observed HF/GGUF grouped TopK8 fastpath log difference be
used as proof that HF and GGUF are taking different grouped expert execution
paths?

### Setup

No benchmark run was launched. This was a source/log audit of:

- high-band HF deploy-package control:
  `<validation-workspace>/runs/moe35b_hf_tp4_deploypkg_cache_capture_dot20_20260627T173555Z`
- low-band GGUF NCCL-path cache-factor probe:
  `<validation-workspace>/runs/moe35b_gguf_tp4_nosplit_awqflagrespect_no_ncclpath_dot20_20260627T204804Z`
- release `fused_moe.py` TopK8 helper:
  `_try_qwen_c1_topk8_fastpath(...)`

### Observations

The source guard for `_try_qwen_c1_topk8_fastpath(...)` is explicitly
one-token:

- `hidden_states.size(0) == 1`
- `topk_ids.size(0) == 1`

Grouped capture/decode shapes are therefore not expected to activate that
kernel. The audit also found that `shape seen` logging only requires the
fastpath enable flag, while `shape_or_layout` rejection logging is gated by the
separate `force` / `debug` setting. The latest GGUF probe used
`VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH=force`, so grouped rejections were visible.
The HF deploy-package control did not preserve comparable debug-mode evidence
in the saved launch artifacts.

### Outcome

Reject the earlier grouped rejection count as a standalone HF/GGUF proof. It is
not fair to compare "HF grouped shapes with no rejection lines" against "GGUF
grouped shapes with rejection lines" unless both runs use the same debug mode.

This does not clear GGUF. The low-band GGUF logs still show padded `w1` expert
stride `(557056, 2176, 1)` in the grouped standard path, and MoE GGUF remains
far below the HF/release TP4 band. The promoted conclusion is narrower: the next
comparison needs a high-band HF control with comparable cache/debug capture or
lower-level generated-kernel evidence.

### Promote / Reject

Promote:

- the one-token-only boundary for the release TopK8 fastpath;
- the need for comparable debug settings before interpreting rejection counts;
- high-band HF cache/debug capture as the next control target.

Reject:

- treating grouped `shape_or_layout` rejection logs alone as the root cause;
- chasing grouped fastpath activation without kernel/runtime evidence;
- launching another full GGUF ladder from this log asymmetry alone.

### Next

Capture a high-band HF TP4 control with isolated TorchInductor/Triton cache and
matched fastpath debug visibility, or use another graph-visible profiler that
can compare steady decode kernel mix without perturbing graph replay.

## GGUF-226 - HF TP4 Force-Debug Fastpath Control

### Question

If the high-band HF TP4 path is launched with the same TopK8 fastpath debug
visibility used by the low-band GGUF probes, does it also log grouped
`shape_or_layout` rejections and the padded expert stride?

### Setup

Created a disposable `.20` HF control by copying the release deploy package
from the high-band HF TP4 cache-capture lane and changing only the fastpath
debug visibility:

- release image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`;
- model: `Qwen/Qwen3.6-35B-A3B`;
- profile: `moe35b_tp4_fullbar_p2pon`;
- TP degree: `4`;
- `MAX_MODEL_LEN=131072`;
- P2P enabled through the release profile;
- `VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH=force`;
- normal benchmark harness sequence was started: warmups, uncapped strict,
  `c1_2000`, then `c1_10000`.

Run directory:

`<validation-workspace>/runs/moe35b_hf_tp4_force_debug_cache_capture_dot20_20260627T211519Z`

Benchmark directory:

`<validation-workspace>/runs/moe35b_hf_tp4_force_debug_cache_capture_dot20_20260627T211519Z/v02-profile-runs/moe35b_tp4_fullbar_p2pon_20260627T211519Z_force_debug`

### Observations

The server reached health and began the normal warmup ladder, but force-debug
mode perturbed the live request path enough that the control is not a fair
performance comparator. The first fixed-token warmup completed, but it took
`238.081` seconds wall time for `2000` completion tokens. vLLM metrics recorded
`431` prompt tokens, `2000` generation tokens, `206.191` prefill seconds,
`31.859` decode seconds, and `62.777` backend decode TPS. The request finished
by length, did not close the Qwen thinking gate, and was not a valid strict
measurement.

The useful part of the run is the matched debug visibility. The HF control
logged:

| fastpath log class | count |
| --- | ---: |
| active | 4 |
| rejected | 55 |
| apply shape seen | 59 |
| `shape_or_layout` | 55 |

The grouped HF rejection lines showed the same padded expert layout that had
previously looked suspicious in GGUF:

```text
w1_shape=(256, 256, 2048)
w1_stride=(557056, 2176, 1)
w2_shape=(256, 2048, 128)
w2_stride=(262144, 128, 1)
```

Only rank cache-factor files were captured; generated graph source artifacts
were not preserved in this run, so this is a log/source-boundary control rather
than a generated-kernel comparator.

### Outcome

Reject grouped `shape_or_layout` rejection and padded `w1` stride as a
GGUF-only explanation for the MoE throughput gap. When debug visibility is
matched, HF also logs grouped rejection lines with the same padded stride while
still having a separate high-band release path in normal, non-force operation.

Also reject `VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH=force` as a live benchmark
capture mode. It is useful for visibility, but it perturbs the request path and
does not reproduce the high-band HF TP4 throughput.

### Promote / Reject

Promote:

- matched-debug controls before comparing fastpath rejection counts;
- the conclusion that grouped fastpath rejection logs are expected for
  multi-token shapes because the release TopK8 helper is one-token only;
- moving the MoE GGUF investigation below Python-visible layout/log evidence.

Reject:

- treating the padded grouped `w1` stride by itself as the GGUF root cause;
- treating absence of HF rejection logs in non-debug artifacts as proof of a
  different grouped execution path;
- force-debug mode as a throughput/profile capture setting.

### Next

The remaining MoE GGUF gap is still real, but the current evidence now points
below visible graph and fastpath-log parity. The next useful probe should be a
lower-level, minimally perturbing comparison of normal HF TP4 versus GGUF TP4:

- steady decode HIP/Triton kernel mix and timing;
- graph replay cadence and request scheduling;
- completed MoE prepare/finalize and shared/expert work;
- activation/reduction movement around fused-MoE postprocess.

## GGUF-227 - HF/GGUF TP4 Summary-Metrics Completed-Work Comparator

### Question

Before launching another instrumented run, do the existing clean HF TP4 and
latest low-band GGUF TP4 benchmark summaries show the gap in prefill/setup,
client accounting, or steady decode?

### Setup

No server was launched. This was a read-only comparison of existing `.20`
summary artifacts:

- clean HF TP4 control:
  `<validation-workspace>/runs/hf_moe35b_tp4_clean_repro_20260627T074206Z/v02-profile-runs/moe35b_tp4_fullbar_p2pon_20260627T074206Z`
- low-band GGUF TP4 NCCL-path/cache-factor probe:
  `<validation-workspace>/runs/moe35b_gguf_tp4_nosplit_awqflagrespect_no_ncclpath_dot20_20260627T204804Z/benchmarks/moe35b_tp4_fullbar_p2pon_20260627T204804Z_no_ncclpath`

Both paths used the normal benchmark artifacts with a `431` token prompt.

### Observations

| Path | Case | Completion tokens | Finish | Qwen gate valid | Prefill seconds | Decode seconds | Backend decode TPS |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| HF | `c1_128_strict` | 5183 | stop | true | 0.319 | 45.390 | 114.187 |
| HF | `c1_2000` | 2000 | length | false | 0.317 | 17.149 | 116.623 |
| HF | `c1_10000` | 10000 | length | false | 0.317 | 91.118 | 109.748 |
| GGUF | `c1_128_strict` | 2993 | stop | true | 0.331 | 36.758 | 81.425 |
| GGUF | `c1_2000` | 2000 | length | false | 0.328 | 24.203 | 82.634 |
| GGUF | `c1_10000` | 10000 | length | false | 0.329 | 136.944 | 73.023 |

Prompt length and prefill time are effectively matched. The c1 fixed-token
gap is almost entirely decode seconds:

- `c1_2000`: HF `17.149` decode seconds versus GGUF `24.203`
- `c1_10000`: HF `91.118` decode seconds versus GGUF `136.944`

The client-visible elapsed time tracks the same pattern, so this is not a
begin-think proxy or client-accounting artifact.

### Outcome

Promote steady decode replay/kernel work as the next source boundary. The
remaining MoE GGUF gap is not explained by prompt length, prefill duration, or
client/proxy timing. It persists inside the long decode window after the same
prompt has been admitted and after strict correctness has already been proven
for GGUF.

### Promote / Reject

Promote:

- lower-level comparison of decode-phase HIP/Triton kernel mix and timing;
- graph replay cadence and scheduling/completed-work comparison;
- activation/reduction movement around fused-MoE postprocess.

Reject:

- spending more time on prompt/prefill-token accounting as the primary gap;
- treating the begin-think proxy or summary parsing as a throughput explanation;
- launching another full ladder unless the source change targets decode-phase
  completed work.

### Next

The next executable experiment should be a minimally perturbing decode-window
comparator. It should compare normal, high-band HF TP4 and low-band GGUF TP4
without `force` debug mode, Python synchronization in replay, or full-server
`rocprofv3` wrapping. Candidate routes:

- a reduced forward-context comparator that preserves real vLLM layer objects;
- a graph-safe lower-level marker or C/C++ custom-op marker around MoE
  prepare/finalize and postprocess;
- a request-window HIP/Triton kernel-count comparator only after artifact
  emission is proven on the intended worker shape.

## GGUF-228 - MoE TP4 FusedMoE HF Layer-Name Alias Probe

### Question

The HF and GGUF TorchInductor cache comparison left one normalized source
difference after removing path, device, stream, and guard noise: the embedded
FusedMoE module name was `language_model.model.layers.*.mlp.experts` for the
HF deploy-package cache and `model.layers.*.mlp.experts` for the GGUF cache.
Can forcing the GGUF FusedMoE runtime layer names to the HF-style
`language_model.*` prefix move MoE GGUF TP4 out of the low decode band?

### Setup

Disposable `.20` source patch:

`qwen35moe-nosplit-awqflagrespect-fusedmoe-hfname-20260627`

Run directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_fusedmoe_hfname_no_ncclpath_dot20_20260627T213900Z`

Benchmark directory:

`<validation-workspace>/runs/moe35b_gguf_tp4_fusedmoe_hfname_no_ncclpath_dot20_20260627T213900Z/benchmarks/moe35b_tp4_fullbar_p2pon_20260627T213900Z_fusedmoe_hfname`

The launch preserved the current low-band comparison shape:

- official v0.2 runtime image;
- FP16 MoE GGUF artifact;
- TP4, full-BAR/P2P-on, `MAX_MODEL_LEN=131072`;
- no explicit `VLLM_NCCL_SO_PATH`;
- `VLLM_USE_TRITON_AWQ=false` in compiled cache factors via the source-level
  AWQ guard;
- forced unquantized F16 GGUF expert path;
- release Qwen C1 TopK8 MoE fastpath overlay;
- normal benchmark ladder: eight warmups, uncapped strict, `c1_2000`, and
  `c1_10000`.

The patch added a post-load helper guarded by
`VLLM_QWEN35_GGUF_HF_MOE_LAYERNAME_ALIAS=1`. It rewrote each GGUF FusedMoE
expert layer name from `model.layers.*.mlp.experts` to
`language_model.model.layers.*.mlp.experts` and updated the static MoE layer
registry when present.

### Observations

Startup showed `160` layer-name alias messages across the four TP workers.
Model load, graph compile, mixed prefill/decode graph capture, decode graph
capture, and API health all completed.

The release TopK8 fastpath behavior did not become a multi-token grouped
fastpath. During startup/capture the larger decode shapes still emitted
`shape_or_layout` rejections with the same padded expert layout. Only the
one-token shape activated the release helper, which matches prior source
inspection of the one-token guard.

The normal ladder completed:

| case | completion tokens | finish | strict gate valid | backend decode TPS |
| --- | ---: | --- | --- | ---: |
| `c1_128_strict` | 3629 | stop | true | 80.661 |
| `c1_2000` | 2000 | length | false | 82.737 |
| `c1_10000` | 10000 | length | false | 73.111 |

The fixed-token prefill time remained near the same parity band as prior GGUF
runs, while decode time remained low-band.

### Outcome

Reject FusedMoE HF-style layer-name aliasing as the MoE GGUF performance
lever. It preserves correctness and strict-gate validity, but the benchmark
numbers remain effectively the same as the prior no-NCCL-path low-band run
(`81.425`, `82.634`, `73.023` backend TPS).

The cache-name difference was real and testable, but it is not the missing
decode throughput source. The profiling/source boundary remains below visible
module-name metadata and prompt/prefill accounting.

### Promote / Reject

Promote:

- cache-diff normalization before launching source probes;
- the conclusion that FusedMoE module-name strings are not the current MoE
  GGUF bottleneck;
- the decode-phase source boundary from `GGUF-227`.

Reject:

- repeating GDN/attention/FusedMoE layer-name alias probes without new
  evidence;
- treating TorchInductor cache `ModuleName` string differences as sufficient
  proof of runtime throughput impact;
- chasing prompt/prefill/proxy changes for this gap.

### Next

The current evidence points to runtime-level decode work:

- generated kernel body differences;
- graph replay cadence;
- grouped expert execution in the standard fused-MoE path;
- activation/reduction movement around MoE prepare/finalize/postprocess;
- HIP/Triton kernel mix and timing under normal, non-force-debug runs.

Do not launch another full ladder unless the source change targets one of
those decode-phase items.

## GGUF-229 - HF/GGUF TP4 Generated Kernel-Perf Artifact Comparator

### Question

Do existing TorchInductor `.kernel_perf` autotune artifacts show a single
obvious generated-kernel timing miss that explains the HF-versus-GGUF MoE TP4
decode gap?

### Setup

No server was launched. This was a read-only comparison of existing `.20`
TorchInductor cache artifacts:

- high-band HF deploy-package cache-capture control:
  `<validation-workspace>/runs/moe35b_hf_tp4_deploypkg_cache_capture_dot20_20260627T173555Z/runtime/tmp/torchinductor_root`
- GGUF FusedMoE HF-layer-name alias probe:
  `<validation-workspace>/runs/moe35b_gguf_tp4_fusedmoe_hfname_no_ncclpath_dot20_20260627T213900Z/torchinductor_root`
- prior GGUF no-NCCL-path low-band probe:
  `<validation-workspace>/runs/moe35b_gguf_tp4_nosplit_awqflagrespect_no_ncclpath_dot20_20260627T204804Z/torchinductor_root`

The comparison summed each `.kernel_perf` file by relative cache key and joined
HF and GGUF keys.

### Observations

HF and GGUF each had `76` `.kernel_perf` files and no missing joined keys in
the HF versus GGUF alias-probe comparison.

The sum of joined `.kernel_perf` values was:

| path | summed kernel-perf value |
| --- | ---: |
| HF deploy-package control | 2.106399 |
| GGUF alias probe | 2.210239 |

The GGUF/HF summed ratio was `1.049`.

Several individual generated kernels were slower in the GGUF artifact set,
including pointwise/reduction kernels with ratios above `2x`, but the aggregate
autotune hint is far smaller than the observed request-time decode gap. The
fixed-token request summaries still show the real gap in decode seconds:

- HF `c1_10000`: `91.118` decode seconds
- GGUF `c1_10000`: `136.944` to `136.779` decode seconds in the latest
  low-band runs

### Outcome

Reject `.kernel_perf` aggregate timing as the complete explanation for the
MoE GGUF gap. The artifacts are useful for narrowing generated-kernel families,
but their joined aggregate ratio is only about `5%`, while request-level decode
time is roughly `50%` longer on the low-band GGUF path.

### Promote / Reject

Promote:

- request-window kernel mix/count comparison as a stronger next measurement;
- replay-frequency and completed-work accounting, not only per-kernel autotune
  timing;
- inspecting high-ratio generated pointwise/reduction kernels as secondary
  leads.

Reject:

- assuming a single missing TorchInductor autotune config explains the whole
  MoE GGUF throughput gap;
- another full benchmark ladder without a source change or lower-level
  decode-window profiler.

### Next

The next practical source target remains normal-run decode profiling: compare
HF and GGUF request windows for kernel counts, graph replay cadence, standard
fused-MoE grouped expert work, activation/reduction movement, and any extra
completed work per generated token.

## GGUF-230 - MoE TP4 Current CUDAGraph Replay Timing Probe

### Question

Does the current strict-valid MoE GGUF TP4 path still sit in the same low
decode band when CUDAGraph replay timing is sampled, and does the timing point
above or below the replayed graph boundary?

### Setup

The run used the current `.20` MoE GGUF TP4 source path from `GGUF-228`:

- run root:
  `<validation-workspace>/runs/moe35b_gguf_tp4_current_cgreplay_timing_dot20_20260627T221003Z`
- profile: `moe35b_tp4_fullbar_p2pon`
- TP: `4`
- host label: `.20`
- full-BAR/P2P-on release lane
- `MAX_MODEL_LEN=131072`
- graph mode
- normal benchmark sequence: eight warmups, `c1_128` uncapped strict,
  `c1_2000`, `c1_10000`
- begin-think proxy enabled for Qwen benchmark scoring
- replay timing patch:
  `<validation-workspace>/experimental-patches/cudagraph-replay-timing-20260627`

The timing patch samples CUDAGraph replay with CUDA events and periodic
synchronization. It is useful as source-localization evidence, but it is not a
zero-overhead profiler and should not be treated as release benchmark evidence.

### Observations

The normal ladder completed and stayed strict-valid but low-band:

| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate valid |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | 3281 | 80.602 | 81.280 | 40.706 | stop | True |
| `c1_2000` | 2000 | 81.402 | 82.538 | 24.569 | length | False |
| `c1_10000` | 10000 | 72.933 | 73.114 | 137.112 | length | False |

Replay timing file:

`<validation-workspace>/runs/moe35b_gguf_tp4_current_cgreplay_timing_dot20_20260627T221003Z/cudagraph_replay_timing.tsv`

Sampled replay timing:

| mode | samples | mean ms | min ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `FULL` | 1952 | 12.054565 | 10.880300 | 15.272617 |
| `PIECEWISE` | 28 | 7.677879 | 7.356147 | 8.293589 |

`FULL` one-token replay drifted during the request window:

| sampled call band | samples | mean ms | min ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| `00000-04095` | 248 | 11.519913 | 11.048459 | 11.972619 |
| `04096-08191` | 248 | 11.509498 | 11.016139 | 12.023980 |
| `08192-12287` | 248 | 11.529527 | 11.013265 | 12.019024 |
| `12288-16383` | 256 | 11.511758 | 11.055339 | 12.067661 |
| `16384-20479` | 252 | 11.680154 | 11.071987 | 12.491820 |
| `20480-24575` | 252 | 11.685275 | 10.880300 | 12.315665 |
| `24576-28671` | 256 | 13.072891 | 12.168937 | 13.900939 |
| `28672-32767` | 192 | 14.469455 | 13.679334 | 15.272617 |

All four TP ranks tracked the same timing band. Per-rank `FULL` means were
`12.053268` to `12.055619` ms, so this does not look like a single bad rank or
single bad GPU.

For scale, the clean HF TP4 control's `c1_10000` decode window was about
`91.118` seconds for `10000` generated tokens, or roughly `9.11` ms/token.
The GGUF replay samples are already slower than that early in the run and drift
farther away in the long tier.

Older HF timing probes are not clean controls for this measurement. The
available HF replay-timing runs were heavily perturbed and collapsed to roughly
`65` backend TPS on `c1_10000`, so they cannot be used as proof that the timing
patch is non-invasive on the high-band HF lane.

### Outcome

Reject the current MoE GGUF path as a performance promotion. It remains
strict-valid, but it is still far below the HF/v0.2.1 TP4 band and does not
match or beat the current MoE target.

Promote the profiling boundary: the live gap is below prompt/proxy/prefill and
inside the replayed decode graph or work immediately attached to it. The
request-level symptom is not a metadata/cache-name issue and not a single-rank
thermal or clock issue. The low band aligns with per-token `FULL` replay cost,
and the long-tier degradation lines up with the late-run replay drift.

### Promote / Reject

Promote:

- current MoE GGUF correctness evidence: strict finishes normally and
  `qwen_gate_valid=true`;
- graph replay timing as a useful localization signal for the low-band MoE GGUF
  path;
- late-run replay drift as an explanation for why `c1_10000` drops below the
  shorter strict and `c1_2000` tiers;
- the source boundary from launch settings into decode graph/custom-op work.

Reject:

- treating the current MoE GGUF path as a matched/beat result;
- rerunning the same full ladder without a decode-path source change;
- using the old HF replay-timing runs as clean performance controls;
- chasing proxy, tokenizer, prompt token count, prefill, NCCL path, module-name
  aliasing, or cache-key metadata as the next primary lever.

### Next

Move one level deeper than CUDAGraph replay timing. The next source work should
compare or instrument request-window decode work inside the replayed graph:

- generated kernel body and kernel mix;
- standard fused-MoE grouped expert execution;
- shared-expert and MoE prepare/finalize/postprocess movement;
- reduction and activation memory movement;
- any extra GGUF-only completed work per generated token.

If a profiler is used, avoid the known perturbed full-server and PID-attach
shapes unless artifact emission and overhead are proven first. Prefer a reduced
decode-window comparator or lower-level C/C++/ROCm instrumentation that can
separate GatedDeltaNet, FusedMoE, shared expert, residual/norm, logits, and
sampler work without changing the benchmark lane.

## GGUF-231 - MoE TP4 HF/GGUF CUDAGraph Input Metadata Probe

### Question

Do the high-band HF TP4 path and the current strict-valid GGUF TP4 path feed
the same tensor shapes and argument signatures into captured CUDAGraph replay?

### Setup

The probe used a disposable metadata-only `CUDAGraphWrapper` patch that records
argument dtype, device, shape, stride, contiguity, and argument count at graph
capture. It does not record tensor values or model output.

Runs:

- HF control:
  `<validation-workspace>/runs/moe35b_hf_tp4_graphinput_meta_dot20_20260627T224016Z`
- GGUF first wiring attempt:
  `<validation-workspace>/runs/moe35b_gguf_tp4_graphinput_meta_dot20_20260627T224418Z`
- GGUF fixed metadata run:
  `<validation-workspace>/runs/moe35b_gguf_tp4_graphinput_meta_fixed_dot20_20260627T225531Z`
- comparison scratch:
  `<validation-workspace>/runs/moe35b_graphinput_meta_compare_20260627T2304Z`

Both useful runs used the `.20` TP4 full-BAR/P2P-on lane, graph mode,
`MAX_MODEL_LEN=131072`, `max_num_batched_tokens=2048`, prefix caching disabled,
and the native `moe35b_tp4_fullbar_p2pon` profile shape. The first GGUF attempt
is rejected as instrumentation wiring only because the metadata environment was
not passed into the container.

### Observations

Captured metadata line counts:

| path | metadata lines |
| --- | ---: |
| HF | 20764 |
| GGUF fixed | 27588 |

Selected graph-input signature counts from the captured summaries:

| signature | HF | GGUF fixed |
| --- | ---: | ---: |
| `shape=(N, 8, 128)`, `stride=(3072, 128, 1)` | 1044 | 0 |
| `shape=(N, 8, 128)`, `stride=(3088, 128, 1)` | 0 | 1464 |
| `shape=(3072, 2048)` | 1048 | 0 |
| `shape=(3088, 2048)` | 0 | 1464 |
| `shape=(16, 2048)` | 1048 | 0 |
| `argc=6` entries | 0 | 52 |
| `argc=7` entries | 36 | 0 |
| `argc=12` entries | 316 | 976 |
| `argc=13` entries | 696 | 0 |

This is a graph-input-level divergence after the previous static
generated-source and `.kernel_perf` comparisons failed to explain the gap. The
captured HF metadata presents a `3072`-wide GatedDeltaNet-like projection shape
plus a separate `16 x 2048` input, while the captured GGUF metadata presents a
`3088`-wide shape and does not show the separate `16 x 2048` input in the same
signature family.

Source reconciliation was required before treating this as a root-cause fix.
The follow-up source-state check showed that the HF metadata launch mounted
only the `CUDAGraphWrapper` metadata patch and did not mount the release
`qwen3_5.py` patch. The release image's built-in `qwen3_5.py` hash was
`587e313270cf78336b7ef9a6ab4df3201ce2008369bb6ae14e208d1d6d606d6f` and used
the stock split `in_proj_qkvz` plus `in_proj_ba` path. The high-band release
patch hash was `71eaf52b5f85c022380599ae80ce0f478e989b6052a4f14a88ef4305edd3c046`
and used the fused `3088` qkv/z/b/a form. The GGUF metadata source hash was
`4359d9c3958047873b40b04cd174309df08266dabc0b81edc9cdbd92c19c16d7` and also
used the fused form.

That means the `3072 + separate 16` versus `3088 folded` metadata contrast is
not a valid high-band HF-versus-GGUF performance hypothesis. It compared the
current GGUF fused source against the stock/split image source, not against the
release-patched high-band HF source.

### Outcome

Promote the metadata probe as a concrete comparator-hygiene and source-routing
finding. Reject the `3072 + separate 16` versus `3088 folded` contrast as a
high-band HF-versus-GGUF root cause because the HF metadata run used stock
image source, not the release-patched high-band source.

Do not treat this as a throughput promotion. The GGUF path still has not
matched or beaten the MoE target.

### Promote / Reject

Promote:

- graph-input metadata as a useful source-state and comparator-hygiene tool;
- the release-patched HF source hash `71eaf52b5f85c022380599ae80ce0f478e989b6052a4f14a88ef4305edd3c046`
  as the required high-band comparator source for future HF probes;
- the current GGUF metadata source hash `4359d9c3958047873b40b04cd174309df08266dabc0b81edc9cdbd92c19c16d7`
  as fused-form source evidence.

Reject:

- treating the stock/split HF metadata run as a high-band comparator;
- another full benchmark ladder without a source-level change or a lower-level
  profiling target;
- assuming `.kernel_perf`, prompt/prefill/proxy, NCCL path, module-name alias,
  or layer-name cache metadata is still the primary lever.

### Next

If graph-input metadata is needed again, rerun a small HF metadata capture with
the release-patched `qwen3_5.py` mounted. Otherwise, continue below the source
contract level: compare the replayed decode graph's kernel mix, completed work,
FusedMoE grouped expert execution, shared-expert/reduction movement, and
generated kernels between the high-band HF path and the strict-valid GGUF path.

## 2026-06-27 - MoE TP4 high-band HF CUDAGraph replay comparator

### Question

The current MoE GGUF TP4 path is strict-valid but low-band. The best GGUF
fixed-token result on `.20` remains around `75.106` backend TPS on `c1_10000`,
while the clean HF TP4 reproduction reaches `109.748` backend TPS and the
release-time corrected native profile can run strict-valid.

Earlier profiling rejected prompt/proxy/prefill, tokenizer accounting, one bad
rank, NCCL-path parity, layer-name cache metadata, and static `.kernel_perf`
aggregate timing as sufficient explanations. The previous HF CUDAGraph replay
timing run was also invalid as a high-band comparator because it mounted only
the replay-timing patch, used the release image's stock split `qwen3_5.py`
source, and produced low-band request results.

The open question was whether a release-patched high-band HF control has a much
faster replay boundary than the GGUF path.

### Method

Started a temporary HF control on `.20` using the published v0.2 image, the
native `moe35b_tp4_fullbar_p2pon` profile, P2P-on settings, the release patch
bundle mounted through `/opt/vllm_patch_bundle`, and the CUDAGraph replay timing
patch mounted over `vllm/compilation/cuda_graph.py`.

The first benchmark attempt hung before any backend request reached vLLM.
Metrics showed zero prompt and generation tokens, zero running requests, and no
replay timing rows. The benchmark process was stopped without touching the
loaded server and the same ladder was rerun with `CHAT_CAPTURE_USE_CURL=1`,
which uses the harness' explicit curl POST path.

Normal benchmark order was preserved:

- pre-measure warmups;
- `c1_128` uncapped strict;
- `c1_2000`;
- `c1_10000`.

### Artifacts

- HF replay-control run:
  `<validation-workspace>/runs/moe35b_hf_tp4_releasepatch_cgreplay_dot20_20260627T232912Z`
- HF benchmark ladder:
  `<validation-workspace>/runs/moe35b_hf_tp4_releasepatch_cgreplay_dot20_20260627T232912Z/benchmark_ladder_curl`
- GGUF comparator run:
  `<validation-workspace>/runs/moe35b_gguf_tp4_current_cgreplay_timing_dot20_20260627T221003Z`

### Results

The release-patched HF replay-control was high-band:

| case | completion tokens | backend decode TPS | finish | strict gate valid |
| --- | ---: | ---: | --- | --- |
| `c1_128_strict` | 3739 | 119.928 | stop | True |
| `c1_2000` | 2000 | 121.255 | length | False |
| `c1_10000` | 10000 | 114.020 | length | False |

CUDAGraph replay timing comparison:

| path | FULL count | FULL mean ms | FULL min ms | FULL max ms | PIECEWISE mean ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF release-patched control | 1980 | 7.887792 | 7.481852 | 9.125901 | 7.153875 |
| GGUF current path | 1952 | 12.054565 | 10.880300 | 15.272617 | 7.677879 |

FULL replay drift by 250-sample bins:

| path | early FULL mean ms | late FULL mean ms |
| --- | ---: | ---: |
| HF release-patched control | 7.730574 | 8.664663 |
| GGUF current path | 11.515907 | 14.436282 |

### Outcome

Promote the release-patched HF replay comparator as the first valid high-band
HF-versus-GGUF replay-boundary measurement. The replay timing difference is
large enough to explain most of the MoE GGUF fixed-token throughput gap:
HF `c1_10000` is `114.020` backend TPS with `7.887792` ms average FULL replay,
while GGUF remains around `73`-`75` backend TPS with `12.054565` ms average FULL
replay.

This moves the active bottleneck below launch/proxy/prefill/cache-key cleanup
and into the replayed decode graph or work immediately adjacent to it.

### Promote / Reject

Promote:

- release-patched HF replay timing as the required comparator;
- CUDAGraph FULL replay timing as an explanatory measurement for the MoE GGUF
  gap;
- the curl POST mode for this probe's benchmark ladder when the urllib path
  hangs before backend submission;
- lower-level decode graph / completed-work analysis as the next source target.

Reject:

- the stale stock-source HF replay run as a comparator;
- another unchanged GGUF full ladder without a source-level decode-path change;
- prompt/proxy/prefill/tokenizer accounting as the main gap;
- `.kernel_perf` aggregate timing as sufficient to explain the gap;
- C1 fastpath absence as the explanation, because both HF and GGUF show the
  one-token fastpath active.

### Next

Compare high-band HF and low-band GGUF inside the replayed decode graph:

- request-window kernel counts and HIP/Triton kernel mix;
- generated kernel bodies for high-ratio kernels;
- FusedMoE grouped expert execution;
- shared-expert and routed-expert reduction movement;
- extra GGUF-only completed work per generated token;
- layout/materialization differences feeding the standard fused-MoE path.

## 2026-06-27 - MoE TP4 generated graph and corrected graph-input metadata comparison

### Question

The release-patched HF replay-control is high-band while the current GGUF path
is low-band, but both use the same model family, profile shape, and P2P-on host.
The replay gap could come from generated kernel source, graph input contract,
or runtime work around otherwise identical graph code.

### Method

Compared the exact cache trees from the high-band HF replay-control and the
current GGUF replay-control:

- HF:
  `<validation-workspace>/runs/moe35b_hf_tp4_releasepatch_cgreplay_dot20_20260627T232912Z/runtime/tmp/torchinductor_root`
- GGUF:
  `<validation-workspace>/runs/moe35b_gguf_tp4_current_cgreplay_timing_dot20_20260627T221003Z/torchinductor_root`

Then reran HF graph-input metadata with the release patch bundle mounted,
because the earlier HF metadata run used stock image source and was not a valid
high-band comparator.

Corrected HF metadata artifact:

`<validation-workspace>/runs/moe35b_hf_tp4_releasepatch_graphinput_meta_dot20_20260627T235328Z/graph_input_meta.tsv`

Existing GGUF metadata artifact:

`<validation-workspace>/runs/moe35b_gguf_tp4_graphinput_meta_fixed_dot20_20260627T225531Z/graph_input_meta.tsv`

### Results

Generated cache comparison:

| item | HF | GGUF |
| --- | ---: | ---: |
| `.kernel_perf` files | 76 | 76 |
| joined `.kernel_perf` keys | 76 | 76 |
| HF-only `.kernel_perf` keys | 0 | - |
| GGUF-only `.kernel_perf` keys | - | 0 |
| generated Python files | 208 | 208 |
| common generated Python content hashes | 208 | 208 |
| HF-only generated Python hashes | 0 | - |
| GGUF-only generated Python hashes | - | 0 |

The joined `.kernel_perf` sums were HF `2.227507998238` and GGUF
`1.860795992897`, a GGUF/HF ratio of `0.835371`. That means the static
generated Python/kernel cache is not only identical by source hash; the
benchmark metadata does not show the GGUF generated kernels as slower.

Corrected graph-input metadata comparison:

| signature | HF release-patched metadata | GGUF metadata |
| --- | ---: | ---: |
| metadata lines | 28136 | 27588 |
| `shape=(3088, 2048)` | 1496 | 1464 |
| `shape=(3072, 2048)` | 0 | 0 |
| `shape=(16, 2048)` | 0 | 0 |
| `stride=(3088, 128, 1)` | 1492 | 1464 |
| `stride=(3072, 128, 1)` | 0 | 0 |
| `argc=6` | 52 | 52 |
| `argc=7` | 0 | 0 |
| `argc=12` | 996 | 976 |
| `argc=13` | 0 | 0 |

Both corrected paths use the fused `3088` qkv/z/b/a form. The earlier
`3072 + separate 16` contrast is confirmed as a stock-source comparator
artifact, not a high-band HF-versus-GGUF performance difference.

### Outcome

Promote the generated-code identity result as a hard boundary: the MoE GGUF gap
is not explained by different TorchInductor-generated Python sources or missing
generated-kernel cache entries.

Promote the corrected metadata result as a comparator hygiene fix: release-
patched HF and GGUF agree on the fused `3088` graph-input family.

The remaining gap is runtime behavior with otherwise identical generated code:
captured custom-op work, kernel invocation counts, tensor materialization,
layout/stride details not visible in the coarse metadata counts, standard
FusedMoE grouped-expert execution, or shared/routed expert reduction movement.

### Promote / Reject

Promote:

- full generated-Python content-hash comparison as a required guard before
  chasing generated-kernel source diffs;
- corrected release-patched HF graph-input metadata as the valid metadata
  comparator;
- request-window kernel invocation counts or lower-overhead custom-op counters
  as the next measurement.

Reject:

- generated Python source diffs as the explanation for the current MoE GGUF
  gap;
- `.kernel_perf` sums as an explanatory bottleneck;
- the earlier stock-source `3072 + 16` metadata contrast;
- another unchanged full ladder without a source-level or profiler-targeted
  change.

### Next

Instrument invocation counts inside the replayed decode graph without changing
the model semantics. The best next slice is a low-overhead counter around the
standard FusedMoE path and adjacent custom ops, or a short `rocprofv3` request
window if attach/output can be made reliable. The question to answer is whether
GGUF launches more of the same kernels/custom ops per generated token, performs
extra materialization/reductions, or feeds the same graph with layout that costs
more at runtime despite identical generated Python.

## 2026-06-28 - MoE TP4 component trace for early sparse-expert divergence

### Question

The layer-wise vector trace localized the GGUF correctness drift to the early
residual path: layer-0 input matched HF, but layer-0 `post_mlp` and the later
layer-1/layer-2 trajectory were already off-direction. The next question was
whether the divergence comes from the sparse MoE path itself, the shared expert,
or tensor-parallel reduction after expert execution.

### Method

Built scratch HF and GGUF component-trace patches for the Qwen3.6 35B-A3B MoE
TP4 profile. The first patch inserted a wrapper around `Qwen3NextSparseMoeBlock`
but did not emit MoE-internal labels because the wrapped block did not retain
its layer prefix and the request-gated prefix filter dropped the fallback
`mlp` label.

Corrected scratch patches set `self.mlp.prefix = f"{prefix}.mlp"` after
constructing the sparse block, then reran the faithful two-token branch probe
on the same TP4/P2P-on/eager settings:

- HF run:
  `<validation-workspace>/runs/moe35b_hf_tp4_componenttraceprefix_eager4k_dot20_20260628T011830Z`
- GGUF run:
  `<validation-workspace>/runs/moe35b_gguf_tp4_componenttraceprefix_eager4k_dot20_20260628T011830Z`
- comparison artifact:
  `<validation-workspace>/runs/moe35b_component_cosine_compare_20260628T011830Z/cosine_summary_with_header.tsv`

Captured components included `moe_input`, `routed_expert_output`,
`shared_expert_output`, `pre_reduce_output`, and `post_reduce_output` for exact
layers `0`, `1`, `2`, `5`, and `10`. The current wrapper still does not expose
router logits when the standard FusedMoE layer uses its internal-router path.

### Results

Both corrected traces produced `539` labels per TP rank and included the MoE
component labels. The HF branch response remained `Thinking Process`; GGUF kept
the null-content branch on the same two-token request.

Important aggregate cosines across four TP ranks:

| component | avg cosine | min cosine | max cosine |
| --- | ---: | ---: | ---: |
| layer-0 `mlp.moe_input` | `0.982377650` | `0.982377650` | `0.982377650` |
| layer-0 `mlp.shared_expert_output` | `0.706940579` | `0.568671570` | `0.868551856` |
| layer-0 `mlp.post_reduce_output` / `post_mlp` | `0.849558298` | `0.849558298` | `0.849558298` |
| layer-1 `mlp.routed_expert_output` | `0.693710627` | `0.298821094` | `0.951999182` |
| layer-1 `mlp.post_reduce_output` / `post_mlp` | `0.895443136` | `0.895443136` | `0.895443136` |
| layer-2 `post_attention` | `0.655290126` | `0.655290126` | `0.655290126` |
| layer-2 `mlp.shared_expert_output` | `0.729080872` | `0.656994970` | `0.819076134` |
| layer-2 `mlp.post_reduce_output` / `post_mlp` | `0.756801243` | `0.756801243` | `0.756801243` |
| layer-5 `mlp.post_reduce_output` / `post_mlp` | `0.901195543` | `0.901195543` | `0.901195543` |
| layer-10 `mlp.post_reduce_output` / `post_mlp` | `0.906997090` | `0.906997090` | `0.906997090` |

The result is not a pure post-reduce/allreduce mismatch. Some TP-local expert
components are already divergent before the reduce step, especially layer-0
shared expert output and rank-local layer-1 routed expert output. Layer-2
post-attention is also already low, so the error is being fed forward through
the residual path quickly.

### Outcome

Promote sparse-expert/shared-expert materialization as the active correctness
target. The GGUF path reaches layer-0 MoE input with high cosine, but the first
MoE outputs are not HF-equivalent. The most suspicious area is GGUF loading /
layout / sharding of the MoE expert and shared-expert weights, plus the internal
router/expert path that the current trace wrapper cannot yet split into gate
logits and selected expert IDs.

Reject a reduce-only hypothesis for now. `post_reduce_output` reflects the
divergence, but `routed_expert_output`, `shared_expert_output`, and
`pre_reduce_output` are already mismatched.

Reject another unchanged GGUF benchmark ladder until a source-level MoE
materialization or expert-routing hypothesis is tested.

### Promote / Reject

Promote:

- corrected request-gated component trace with explicit sparse-block prefix;
- MoE shared/routed expert output comparison as the next correctness guardrail;
- expert weight materialization, expert shard ordering, and internal-router
  visibility as the next source-work slice.

Reject:

- the first component-trace patch without explicit `self.mlp.prefix`;
- generated Python source diffs as the explanation;
- static graph-input metadata diffs as the current explanation;
- allreduce/reduce-only debugging as the primary next step.

### Next

Add a lower-level MoE source probe that exposes router logits or selected
experts even when the FusedMoE layer uses the internal-router path, then compare
HF and GGUF expert IDs and per-expert weight slices. If routing matches, inspect
GGUF expert tensor loading, shard ordering, transpose/layout handling, and
shared-expert materialization before returning to performance ladders.

## 2026-06-28 - MoE TP4 router top-k trace

### Question

The component trace showed early divergence inside the sparse MoE path, but did
not expose selected experts because Qwen3.6 MoE uses the FusedMoE
internal-router path. The next question was whether HF and GGUF route the same
token to the same experts before expert execution.

### Method

Built scratch HF/GGUF router-topk overlays from the corrected component-trace
patches. The overlays mount two additional vLLM modules without modifying the
release image:

- `vllm/model_executor/layers/fused_moe/layer.py`
- `vllm/model_executor/layers/fused_moe/router/base_router.py`

The `FusedMoE` layer tags its router with the layer prefix, and
`BaseRouter.select_experts()` writes request-gated `topk_ids` and
`topk_weights` rows before EPLB mapping and before expert execution.

Runs:

- HF:
  `<validation-workspace>/runs/moe35b_hf_tp4_routertopk_eager4k_dot20_20260628T012050Z`
- GGUF:
  `<validation-workspace>/runs/moe35b_gguf_tp4_routertopk_eager4k_dot20_20260628T012050Z`
- comparison artifact:
  `<validation-workspace>/runs/moe35b_routertopk_compare_20260628T012050Z/topk_compare_with_header.tsv`

Both runs used the same faithful two-token branch prompt, TP4, P2P-on, eager
mode, and the same release image.

### Results

HF produced `Thinking Process`; GGUF again produced the null-content branch.

Router comparison summary:

| item | value |
| --- | ---: |
| compared rows | `480` |
| exact ordered top-k matches | `0` |
| top-1 matches | `268` |
| average set overlap | `5.933 / 8` |
| minimum set overlap | `1 / 8` |
| maximum set overlap | `8 / 8` |

Early event-0 examples were already different:

| layer | HF top-k IDs | GGUF top-k IDs | overlap |
| --- | --- | --- | ---: |
| 0 | `222,223,186,73,199,20,175,92` | `223,73,199,234,161,92,222,186` | `6 / 8` |
| 1 | `98,49,245,153,178,204,45,175` | `49,98,245,153,126,204,178,123` | `6 / 8` |
| 2 | `193,216,76,16,158,3,187,71` | `216,193,16,76,42,158,71,157` | `6 / 8` |
| 5 | `170,98,208,150,45,62,96,115` | `170,98,62,150,96,115,84,208` | `7 / 8` |
| 10 | `81,232,165,234,188,205,66,187` | `81,165,232,234,188,205,187,111` | `7 / 8` |

The ordered routing is never identical in the compared window. Some rows share
most of the set, but the priority order and weights differ; others lose several
experts from the set.

### Outcome

Promote router/gate materialization as the active correctness target. The
earlier expert-output mismatch is not purely an expert-kernel or reduce issue:
HF and GGUF often select different experts before expert execution. Since the
layer-0 MoE input vector was still high-cosine but top-k IDs already differed,
the next probe should inspect the FusedMoE internal router input, gate weight
materialization, score-correction bias, and top-k scoring math.

Reject a pure expert-weight-output hypothesis until routing is understood.
Expert weight layout may still be wrong, but selected experts differ first.

Reject another performance ladder until a source-level routing/materialization
hypothesis changes the selected experts or improves the branch probe.

### Promote / Reject

Promote:

- router top-k tracing as a required correctness guardrail for MoE GGUF;
- internal-router/gate weight and score-correction-bias materialization as the
  next source target;
- comparing selected experts before comparing expert output performance.

Reject:

- reduce-only explanations;
- expert-output-only debugging before checking selected expert IDs;
- unchanged MoE GGUF benchmark reruns.

### Next

Trace or directly compare the internal router inputs and gate parameters:

- confirm whether HF and GGUF `e_score_correction_bias` values match;
- compare gate/top-k scoring inputs around the first selected token;
- verify GGUF mapping of `ffn_gate_inp.weight`, `exp_probs_b.bias`, and any
  router-specific tensors for Qwen3.6 MoE;
- only after routing matches, return to expert gate/up/down shard layout.

## 2026-06-28 - MoE TP4 gate-logit and router-input trace

### Question

The router top-k trace showed that HF and GGUF select different experts before
expert execution. The next question was whether this was caused by missing or
misloaded score-correction bias, gate-logit materialization, or upstream hidden
state drift feeding the internal gate.

### Method

Built scratch HF/GGUF gate-logit overlays from the router-topk patches without
modifying the release image:

- HF overlay:
  `<validation-workspace>/experimental-patches/qwen35moe-hf-gatelogits-20260628T020000Z`
- GGUF overlay:
  `<validation-workspace>/experimental-patches/qwen35moe-gguf-gatelogits-20260628T020000Z`

The overlay added a request-gated trace in
`fused_moe/runner/default_moe_runner.py` immediately after the internal gate
call:

```python
router_logits, _ = self.gate(hidden_states)
```

The trace recorded the final token's internal-gate input summary, full
`256`-expert router-logit row, top choice IDs/values, scoring function, and
loaded `e_score_correction_bias` state. It reused the same faithful two-token
branch prompt, TP4, P2P-on, eager mode, and release image shape as the previous
router-topk trace.

Runs:

- HF:
  `<validation-workspace>/runs/moe35b_hf_tp4_gatelogits_eager4k_dot20_20260628T020000Z`
- GGUF:
  `<validation-workspace>/runs/moe35b_gguf_tp4_gatelogits_eager4k_dot20_20260628T020000Z`
- comparison artifacts:
  `<validation-workspace>/runs/moe35b_gatelogits_compare_20260628T020000Z/summary.txt`
  and
  `<validation-workspace>/runs/moe35b_gatelogits_compare_20260628T020000Z/gate_compare_event0.tsv`

### Results

HF again produced `Thinking Process`; GGUF produced null content with
reasoning text `Here's`.

The traced internal gate calls reported `e_score_correction_bias=None` on both
HF and GGUF for the compared rows. That rejects a missing `exp_probs_b` /
score-correction-bias value as the live cause of the branch split in this
profile.

Router-logit comparison across rank/layer/event rows:

| item | value |
| --- | ---: |
| compared rows | `192` |
| exact top-16 matches | `0` |
| top-1 matches | `108` |
| average logit cosine | `0.998827477` |
| minimum logit cosine | `0.995624130` |
| maximum absolute logit diff | `2.735352` |
| average top-16 set overlap | `13.021 / 16` |

Event-0 examples:

| layer | logit cosine | max abs diff | HF top-8 IDs | GGUF top-8 IDs |
| --- | ---: | ---: | --- | --- |
| 0 | `0.999179904` | `0.996094` | `222,223,186,73,199,175,20,92` | `223,73,199,234,161,92,222,186` |
| 1 | `0.995985865` | `1.882812` | `98,49,245,153,178,204,45,175` | `49,98,245,153,126,204,178,123` |
| 2 | `0.995624130` | `1.875000` | `193,216,76,16,158,3,187,71` | `216,193,16,76,42,158,157,71` |
| 5 | `0.999132578` | `0.812500` | `170,98,208,150,62,45,96,115` | `170,98,62,150,96,115,84,208` |
| 10 | `0.999380984` | `0.894531` | `81,232,165,234,188,205,66,187` | `81,165,232,234,188,205,187,111` |

The paired boundary trace from the same run also shows that the internal-gate
input is already drifting:

| layer | MoE input cosine | max abs diff |
| --- | ---: | ---: |
| 0 | `0.982377650` | `1.101563` |
| 1 | `0.935772504` | `1.919922` |
| 2 | `0.903625660` | `2.072754` |
| 5 | `0.915851469` | `1.233399` |
| 10 | `0.925721323` | `1.173828` |

The router logits are more aligned than the hidden inputs, but top-k selection
is sensitive enough that those upstream hidden-state differences still change
the expert order and sometimes the selected set.

### Outcome

Promote upstream hidden-state drift before the internal gate as the active
correctness target. The live route has no score-correction bias to fix, and the
gate logits are high-cosine rather than grossly remapped. The expert-selection
split is now best explained as a top-k amplification of earlier residual /
attention / GatedDeltaNet trajectory drift.

Reject a bias-only or `exp_probs_b`-only fix for the current profile. The trace
showed `e_score_correction_bias=None` in both HF and GGUF internal-router calls.

Reject a gate-weight-first explanation unless a later direct weight audit shows
the HF and GGUF gate tensors differ. The current evidence says the gate input
is already off before the router selects experts.

### Promote / Reject

Promote:

- request-gated internal-gate trace in `DefaultMoERunner.forward_impl`;
- router-logit cosine plus selected-expert comparison as the next correctness
  guardrail;
- layer-0 through layer-2 upstream residual / attention / GatedDeltaNet trace
  as the next source-work slice.

Reject:

- `e_score_correction_bias` / `exp_probs_b` as the live blocker for this
  profile;
- another unchanged MoE GGUF performance ladder;
- expert-output-only debugging before the upstream hidden trajectory is fixed.

### Next

Move the source probe upstream from FusedMoE routing:

- compare layer-0 and layer-1 attention / GatedDeltaNet contribution vectors
  with enough precision to identify the first operation that moves off HF;
- separate attention output, GatedDeltaNet output, residual add, and
  post-attention norm before the first MoE gate;
- only revisit gate/expert tensor layout after the pre-gate hidden trajectory
  matches HF more closely.

## 2026-06-28 - Boundary-trace mining for the first upstream drift point

### Question

The gate-logit trace showed that pre-gate hidden drift is already present. The
next question was whether the already-captured boundary traces were precise
enough to locate the first layer-0 operation that diverges, without launching a
new server run.

### Method

Parsed the `row_values` fields from the paired HF/GGUF boundary traces in the
gate-logit runs and normalized label prefixes before cosine comparison.

Comparison artifact:

`<validation-workspace>/runs/moe35b_gatelogits_compare_20260628T020000Z/boundary_all_common_cosines.tsv`

### Results

Rank-0 ordered layer-0/layer-1 highlights:

| label | cosine | max abs diff |
| --- | ---: | ---: |
| layer-0 `pre_input_layernorm` | `1.000000000` | `0.000000` |
| layer-0 `post_input_layernorm` | `0.999999943` | `0.001954` |
| layer-0 `linear_attn.input_hidden` | `0.999999943` | `0.001954` |
| layer-0 `linear_attn.mixed_qkvz` | `0.999999978` | `0.007813` |
| layer-0 `linear_attn.ba` | `0.999999999` | `0.000489` |
| layer-0 `linear_attn.core_attn_out_raw` | `1.000000000` | `0.001221` |
| layer-0 `linear_attn.out_proj_input` | `0.878082862` | `0.072204` |
| layer-0 `linear_attn.out_proj_output` / `post_attention` | `0.988626168` | `0.022034` |
| layer-0 `post_attention_layernorm` / `mlp.moe_input` | `0.982377650` | `1.101563` |
| layer-0 `mlp.routed_expert_output` | `0.923882178` | `0.005173` |
| layer-0 `mlp.shared_expert_output` | `0.622324722` | `0.003537` |
| layer-0 `mlp.post_reduce_output` / `post_mlp` | `0.849558298` | `0.008163` |
| layer-1 `pre_input_layernorm` | `0.849558298` | `0.008163` |
| layer-1 `linear_attn.ba` | `0.925086100` | `1.444336` |
| layer-1 `linear_attn.a` | `-0.034166718` | `1.444336` |
| layer-1 `linear_attn.out_proj_output` / `post_attention` | `0.817134116` | `0.072266` |
| layer-1 `post_attention_layernorm` / `mlp.moe_input` | `0.935772504` | `1.919922` |

The trace also showed raw non-SSM RMSNorm-family weights offset by about `+1`
on the GGUF side. That is expected for this scratch path: the GGUF processor
and runtime norm-offset bridge intentionally handle Qwen3.5-MoE non-SSM norm
weights differently from SSM norm weights. The active runtime uses
`VLLM_QWEN35_GGUF_RUNTIME_NORM_OFFSET_FIX=1`, and the output tensors around the
first input norm remain effectively identical. Treat raw norm-weight trace
cosines as an offset-format diagnostic, not as a direct correctness failure.

### Outcome

Promote layer-0 linear-attention output projection / post-attention residual and
the first MoE gate as the current source boundary. The model is exact through
most layer-0 linear-attention internals, but the vector presented to the
attention output projection is not direction-equivalent, and the resulting
post-attention vector is different enough for the MoE gate to select a
different expert order.

Reject raw non-SSM RMSNorm weight mismatch as the next fix target. It is an
expected GGUF offset-format artifact under the current runtime bridge, and the
first input norm output remains effectively identical.

Reject layer-1-first debugging. Layer 1 is already downstream of a low-cosine
layer-0 MoE result, so the next source change should focus on layer 0 before
using layer 1 as a fix target.

### Promote / Reject

Promote:

- boundary-trace mining as a low-cost comparator before launching another
  server run;
- layer-0 output-projection input/reshape, post-attention residual, and first
  MoE gate as the next source targets;
- raw/effective RMSNorm weight distinction as required trace interpretation.

Reject:

- norm-offset-only fixes based on raw weight traces;
- layer-1-only probes before layer-0 divergence is reduced;
- another unchanged performance ladder.

### Next

Inspect the layer-0 linear-attention output projection input construction and
shape/stride semantics, especially the transition from `core_attn_out_normed`
and `z_for_norm` to `out_proj_input`. If that path cannot explain the drift,
move immediately to a controlled layer-0 MoE route-forcing experiment to
separate "small pre-gate drift amplified by top-k" from "expert materialization
wrong even when route is forced."

## 2026-06-28 - MoE TP4 layer-0 route-forcing diagnostic

### Question

The boundary trace showed that GGUF reaches the first MoE gate with a
high-cosine but not identical hidden vector, and the router top-k trace showed
that this is enough to alter selected experts. The next question was whether
forcing the GGUF layer-0 route to the HF route would materially improve the
layer-0 residual trajectory or whether expert materialization would remain
wrong even under the same selected experts.

### Method

Built a scratch GGUF route-forcing overlay without modifying the release image:

`<validation-workspace>/experimental-patches/qwen35moe-gguf-forcehf-l0route-20260628T023000Z`

The overlay patched `fused_moe/router/base_router.py` so the GGUF TP4 run
forced the layer-0 event-0 final-row route to the HF route observed in the
router/gate traces. The forced route was applied before expert execution and
logged in the router trace.

Run:

`<validation-workspace>/runs/moe35b_gguf_tp4_forcehf_l0route_eager4k_dot20_20260628T023000Z`

### Results

The forced route log confirmed that layer-0 event-0 used the HF route:

| item | value |
| --- | --- |
| forced expert IDs | `222,223,186,73,199,20,175,92` |
| forced weights | `0.155119717,0.152416825,0.128598765,0.124885872,0.122231394,0.106404103,0.106404103,0.103939250` |

The response still followed the GGUF wrong branch: null content with reasoning
text beginning `Here's`.

Compared with the earlier unforced boundary trace, forcing only the layer-0
final-row route materially improved the layer-0 residual trajectory:

| label | forced-route cosine versus HF |
| --- | ---: |
| layer-0 `mlp.routed_expert_output`, rank 0 | `0.923882178` |
| layer-0 `mlp.routed_expert_output`, rank 1 | `0.921115167` |
| layer-0 `mlp.routed_expert_output`, rank 2 | `0.963771056` |
| layer-0 `mlp.routed_expert_output`, rank 3 | `0.959121649` |
| layer-0 `mlp.shared_expert_output`, rank 0 | `0.970252042` |
| layer-0 `mlp.shared_expert_output`, rank 1 | `0.988431334` |
| layer-0 `mlp.shared_expert_output`, rank 2 | `0.976093244` |
| layer-0 `mlp.shared_expert_output`, rank 3 | `0.946853812` |
| layer-0 `mlp.post_reduce_output` / `post_mlp` | `0.977170526` |
| layer-1 `pre_input_layernorm` | `0.977170526` |

Layer 1 then diverged again. The forced run still had low routed-output
cosines on later early-layer ranks, including layer-1 rank 2 at `0.310366085`
and layer-1 rank 3 at `0.594683211`.

### Outcome

Promote top-k route amplification as a real correctness mechanism. Forcing the
first layer's route improved layer-0 `post_mlp` from the earlier `0.849558298`
boundary to `0.977170526` and carried that improvement into layer-1 input.
That is strong evidence that the small pre-gate vector drift is being amplified
by sparse expert selection.

Reject layer-0-only route forcing as a complete fix. It did not flip the branch
back to the HF `Thinking Process` path, and layer-1/layer-2 still diverge.

Do not overinterpret the routed-expert component rows as a complete selected
expert equivalence proof. The patch forced the final-row route for the selected
layer/event, while some component comparisons include TP-rank-local details and
other rows. The safe conclusion is that route forcing materially improves the
residual trajectory, not that every routed expert tensor is proven identical.

### Promote / Reject

Promote:

- route replay/forcing as a diagnostic for MoE GGUF correctness;
- early-layer top-k route amplification as a confirmed contributor;
- multi-layer HF-route replay for layers `0` through `2`, and then `0` through
  `5` if the branch remains wrong.

Reject:

- claiming MoE GGUF correctness or performance parity from a forced diagnostic;
- treating layer-0-only route forcing as a source fix;
- another unchanged MoE GGUF performance ladder before branch correctness is
  restored.

### Next

Generate a small HF route map from the existing router traces, then run a GGUF
diagnostic that replays HF final-row routes for layers `0` through `2` under
the same TP4 faithful branch prompt. If that restores the HF branch or
substantially improves layer-2/layer-5 cosine, extend replay to layers `0`
through `5`. If multi-layer route replay does not recover the branch, return to
the layer-0 output-projection input construction and residual update path.

## 2026-06-28 - MoE TP4 multi-layer route-replay diagnostics

### Question

Layer-0 route forcing improved the residual trajectory but did not restore the
HF branch. The next question was whether replaying the HF route map across more
early layers would recover the branch or whether the GGUF divergence must be
fixed upstream of routing / inside the attention-residual path.

### Method

Two bounded route-replay diagnostics were run on `.20` with the same release
image, GGUF model, TP4, P2P-on, eager-mode branch probe, and scratch overlay
pattern as the layer-0 diagnostic.

Invalid attempts were also recorded:

- `moe35b_gguf_tp4_forcehf_l0l2route_eager4k_dot20_20260628T020336Z` did not
  include the new force env var and is rejected as an unforced control.
- `moe35b_gguf_tp4_forcehf_l0l2route_eager4k_dot20_20260628T020745Z` created
  `trace_gate` before launch, causing startup/profile routing to consume the
  first trace/force condition at shape `(1024, 8)`. It is rejected as an
  invalid request-branch diagnostic.

Valid diagnostics:

- layers `0` through `2` every-call route replay:
  `<validation-workspace>/runs/moe35b_gguf_tp4_forcehf_l0l2route_everycall_eager4k_dot20_20260628T021341Z`
- layers `0` through `5` every-call route replay:
  `<validation-workspace>/runs/moe35b_gguf_tp4_forcehf_l0l5route_everycall_eager4k_dot20_20260628T021826Z`

The valid runs made route forcing independent of the trace counter so profiling
could not consume the diagnostic. `trace_gate` was created only after the
server reached `/v1/models`, immediately before the single branch request.

### Results

Both valid runs confirmed that the intended HF routes were replayed for the
target layers in the actual request trace.

The `0` through `2` route replay did not restore the HF branch. It produced
blank content rather than `Thinking Process`. The branch logits moved away from
the original `Here` path but still did not promote: at the later logged branch
point, watched `Thinking` remained far from rank 1.

The `0` through `5` route replay also did not restore the HF branch. It produced
the same blank two-token path. The request trace confirmed HF routes for layers
`0` through `5`; layer `6` and later routes were still GGUF-derived.

Route replay did improve some residual alignment, but not enough to recover
the branch:

| label | l0-l2 replay cosine | l0-l5 replay cosine |
| --- | ---: | ---: |
| layer-0 `post_mlp` | `0.977170526` | `0.977170526` |
| layer-1 `post_mlp` | `0.922040945` | `0.922040945` |
| layer-2 `post_attention` | `0.646410402` | `0.646410402` |
| layer-2 `post_mlp` | `0.834355903` | `0.834355903` |
| layer-5 `pre_input_layernorm` | `0.911286193` | `0.942882641` |
| layer-5 `post_mlp` | `0.908002698` | `0.928562586` |
| layer-10 `pre_input_layernorm` | `0.885273314` | `0.897774157` |
| layer-10 `post_mlp` | `0.902844431` | `0.907384666` |

Logit trace highlights:

- l0-l2 replay: first generated path moved to the blank/newline branch; the
  watched `Thinking` id reached rank `60` at one logged step and rank `2321`
  at the next, but never ranked first.
- l0-l5 replay: `Thinking` improved to rank `11` at one logged step and rank
  `854` at the next, but still did not become the selected branch.

### Outcome

Promote early-route amplification as a contributor, but reject route replay as
a sufficient correctness fix. Forcing HF routes through layer `5` changes the
failure mode and improves some downstream cosines, but it does not recover the
HF `Thinking Process` branch.

Promote layer-1/layer-2 attention and residual trajectory as the next source
target. Even after early route replay, layer-2 `post_attention` remains low
cosine (`0.646410402`), which is upstream of later MoE decisions and explains
why simply replaying selected experts is not enough.

Reject another route-forcing expansion as the next experiment unless it is
paired with a specific attention/residual hypothesis. Replaying routes farther
downstream risks creating artificial token paths without identifying the source
bug.

### Promote / Reject

Promote:

- request-gated route replay with force independent of trace counters;
- the trace-gate startup pitfall as a documented reject condition;
- layer-1/layer-2 attention, GatedDeltaNet contribution, residual add, and
  output-projection shape/stride semantics as the next correctness slice.

Reject:

- unforced or profile-consumed route-replay runs as evidence;
- layer-0, layer-0-through-2, or layer-0-through-5 route replay as a fix;
- any GGUF MoE performance ladder until the faithful branch returns.

### Next

Instrument the layer-1 and layer-2 attention/residual path after the improved
layer-0 route-replay state. The next source experiment should separate:

- GatedDeltaNet core output;
- `z` / norm / output-projection input construction;
- output-projection result;
- residual add before post-attention norm.

The goal is to explain why layer-2 `post_attention` remains low-cosine even
when layers `0` through `2` use the HF route map.

## 2026-06-28 - MoE TP4 attention/residual mining after route replay

### Question

The valid l0-l5 route replay confirmed that early expert routes can be forced
to match HF, but the branch still does not recover. The next question was
whether the existing traces already show the remaining divergence inside
attention/residual flow, especially around the Qwen3.5 GatedDeltaNet
output-projection input.

### Method

No new server was launched. The existing HF gate-logit trace was compared
against both:

- unforced GGUF gate-logit trace:
  `<validation-workspace>/runs/moe35b_gguf_tp4_gatelogits_eager4k_dot20_20260628T020000Z`
- valid l0-l5 every-call route replay:
  `<validation-workspace>/runs/moe35b_gguf_tp4_forcehf_l0l5route_everycall_eager4k_dot20_20260628T021826Z`

Comparison artifacts:

- `<validation-workspace>/runs/moe35b_attention_residual_compare_20260628T021826Z/attention_residual_join.tsv`
- `<validation-workspace>/runs/moe35b_attention_residual_compare_20260628T021826Z/attention_residual_selected.tsv`
- `<validation-workspace>/runs/moe35b_attention_residual_compare_20260628T021826Z/headperm_probe.tsv`

The comparison used existing `row_values` traces for:

- decoder residual and norm boundaries;
- `linear_attn.input_hidden`;
- `linear_attn.mixed_qkvz`;
- `linear_attn.z_pre_reshape`;
- `linear_attn.core_attn_out_normed`;
- `linear_attn.out_proj_input`;
- `linear_attn.out_proj_output`;
- MoE routed/shared/reduced outputs.

### Results

Route replay improves sparse-MoE residual agreement, but the attention path
still contains the larger remaining divergence.

Selected unforced versus l0-l5 route-replay cosines:

| label | unforced cosine | l0-l5 replay cosine | delta |
| --- | ---: | ---: | ---: |
| layer-0 `linear_attn.core_attn_out_normed` | `0.999903782` | `0.999903782` | `0.000000000` |
| layer-0 `linear_attn.out_proj_input` | `0.944018924` | `0.944018924` | `0.000000000` |
| layer-0 `post_mlp` | `0.849558298` | `0.977170526` | `0.127612227` |
| layer-1 `linear_attn.core_attn_out_normed` | `0.790346254` | `0.794759707` | `0.004413453` |
| layer-1 `linear_attn.out_proj_input` | `0.780464966` | `0.778301128` | `-0.002163839` |
| layer-1 `post_mlp` | `0.895443136` | `0.922040945` | `0.026597809` |
| layer-2 `linear_attn.core_attn_out_normed` | `0.808045048` | `0.801679775` | `-0.006365273` |
| layer-2 `linear_attn.out_proj_input` | `0.668013175` | `0.665483755` | `-0.002529421` |
| layer-2 `post_attention` | `0.655290126` | `0.646410402` | `-0.008879724` |
| layer-2 `post_mlp` | `0.756801243` | `0.834355903` | `0.077554660` |
| layer-5 `linear_attn.core_attn_out_normed` | `0.801165527` | `0.824400256` | `0.023234728` |
| layer-5 `linear_attn.out_proj_input` | `0.732923293` | `0.735253862` | `0.002330568` |
| layer-5 `post_mlp` | `0.901195543` | `0.928562586` | `0.027367043` |

A simple local 8-head permutation probe on the `linear_attn.out_proj_input`
row did not explain the mismatch. For the first rank pair, exhaustive `8!`
head permutation search chose identity as the best permutation for layers `0`,
`1`, `2`, and `5`. Reversed-head order was strongly rejected:

| label | base cosine | best cosine | best head perm | reversed-head cosine |
| --- | ---: | ---: | --- | ---: |
| layer-0 `out_proj_input` | `0.878082862` | `0.878082862` | `0,1,2,3,4,5,6,7` | `0.040652165` |
| layer-1 `out_proj_input` | `0.839706517` | `0.839706517` | `0,1,2,3,4,5,6,7` | `-0.008928482` |
| layer-2 `out_proj_input` | `0.514428905` | `0.514428905` | `0,1,2,3,4,5,6,7` | `-0.029989344` |
| layer-5 `out_proj_input` | `0.594508355` | `0.594508355` | `0,1,2,3,4,5,6,7` | `-0.001191208` |

### Outcome

Promote the Qwen3.5 linear-attention output-projection input path as the next
source target. Route replay can repair parts of the MoE residual trajectory,
but it does not repair `linear_attn.core_attn_out_normed ->
linear_attn.out_proj_input -> linear_attn.out_proj_output`.

Reject simple local value-head permutation as the explanation for
`out_proj_input` mismatch. Identity was already the best 8-head order on the
tested rank row.

Reject broad route replay as the next source experiment. It improves some
downstream cosines but does not fix the attention/residual divergence or the
branch.

### Promote / Reject

Promote:

- full-token tracing around `core_attn_out_normed`, reshape/rearrange, and
  `out_proj_input`;
- Qwen3.5 V-head GGUF reorder and output-projection input construction as the
  next low-level source slice;
- comparing all local heads for the final token, not only the last flattened
  head row.

Reject:

- another route-replay expansion;
- simple 8-head reorder as a source fix;
- performance testing before the faithful branch returns.

### Next

Add a trace-only diagnostic that dumps the full final-token value vector before
and after the `core_attn_out.reshape(z_shape_og)` / `rearrange(... h d -> ...
(h d))` transition. The current trace only proves a selected flattened
`core_attn_out_normed` row; it does not yet prove the full final-token
all-head vector. If the full pre-rearrange vector matches HF but
`out_proj_input` does not, the fix target is the reshape/rearrange contract. If
the full pre-rearrange vector is already off, the target moves earlier into the
GatedDeltaNet core output or `z`/norm interaction.

## 2026-06-28 - MoE TP4 full-token output-projection trace

### Question

Does the GGUF MoE path lose agreement with HF because the final-token
GatedDeltaNet value vector is already wrong, or because the vector is reshaped
incorrectly before `linear_attn.out_proj`?

### Method

Two trace-only eager 4K runs were launched on the reference `.20` lane:

- HF control:
  `<validation-workspace>/runs/moe35b_hf_tp4_fulltokenoutproj_eager4k_dot20_20260628T022943Z`
- GGUF candidate:
  `<validation-workspace>/runs/moe35b_gguf_tp4_fulltokenoutproj_eager4k_dot20_20260628T022943Z`

Both runs used one gated strict prompt request. The patch added trace labels
around `core_attn_out_normed`:

- `linear_attn.core_attn_out_normed_last_token_flat`
- `linear_attn.core_attn_out_normed_last_token_alt_heads_first`
- existing `linear_attn.out_proj_input`
- existing `linear_attn.out_proj_output`

Comparison artifacts:

- `<validation-workspace>/runs/moe35b_fulltokenoutproj_compare_20260628T022943Z/fulltokenoutproj_compare.tsv`
- `<validation-workspace>/runs/moe35b_fulltokenoutproj_compare_20260628T022943Z/fulltokenoutproj_selected.tsv`

### Results

The HF control returned the expected `Thinking Process` branch. The GGUF run
returned the known wrong `Here's` branch.

Selected cosine results:

| label | first-rank cosine | mean cosine |
| --- | ---: | ---: |
| layer-0 `core_attn_out_normed_last_token_flat` | `0.878082862` | `0.944018924` |
| layer-0 `core_attn_out_normed_last_token_alt_heads_first` | `0.999999746` | `0.999945831` |
| layer-0 `out_proj_input` | `0.878082862` | `0.944018924` |
| layer-1 `core_attn_out_normed_last_token_flat` | `0.831878661` | `0.780464966` |
| layer-1 `core_attn_out_normed_last_token_alt_heads_first` | `0.853450533` | `0.901644111` |
| layer-1 `out_proj_input` | `0.831878661` | `0.780464966` |
| layer-2 `core_attn_out_normed_last_token_flat` | `0.505315936` | `0.668013175` |
| layer-2 `core_attn_out_normed_last_token_alt_heads_first` | `0.856964249` | `0.917695214` |
| layer-2 `out_proj_input` | `0.505315936` | `0.668013175` |
| layer-5 `core_attn_out_normed_last_token_flat` | `0.632915039` | `0.732923293` |
| layer-5 `core_attn_out_normed_last_token_alt_heads_first` | `0.901198906` | `0.860840415` |
| layer-5 `out_proj_input` | `0.632915039` | `0.732923293` |

The alternate heads-first final-token view is much closer to HF than the vector
currently fed into `out_proj`, especially at layers `0`, `2`, and `5`.

### Follow-up candidate

A copied patch bundle then consumed the existing
`VLLM_QWEN35_GGUF_LINEAR_ATTN_STATE_CONVERT=1` flag and globally reshaped the
normed GatedDeltaNet output as heads-first/token-second before
`linear_attn.out_proj`.

Run:

- `<validation-workspace>/runs/moe35b_gguf_tp4_stateconvert_eager4k_dot20_20260628T024100Z`

Comparison artifact:

- `<validation-workspace>/runs/moe35b_stateconvert_compare_20260628T024100Z/stateconvert_selected.tsv`

The candidate did not restore correctness. It returned the wrong `ancest`
branch, and the similarity collapsed after layer 0:

| label | first-rank cosine | mean cosine |
| --- | ---: | ---: |
| layer-0 `out_proj_input` | `0.483958924` | `0.226750694` |
| layer-0 `post_attention` | `0.210117847` | `0.210117847` |
| layer-1 `out_proj_input` | `0.027885782` | `0.028232840` |
| layer-2 `out_proj_input` | `0.023830630` | `-0.001040235` |
| layer-5 `out_proj_input` | `-0.013301968` | `-0.010325054` |

### Additional conversion candidates

Two narrower variants were tested after the first rejected post-norm
conversion.

#### Pre-norm state conversion

Run:

- `<validation-workspace>/runs/moe35b_gguf_tp4_prenormstateconvert_eager4k_dot20_20260628T031000Z`

Comparison artifact:

- `<validation-workspace>/runs/moe35b_prenormstateconvert_compare_20260628T031000Z/prenormstateconvert_selected.tsv`

This candidate interpreted the GatedDeltaNet output storage as
heads-first/token-second before flattening and before pairing with `z` in the
norm. It returned the wrong `SpaceItemSpaceItem` branch and made layer-0
agreement worse:

| label | first-rank cosine | mean cosine |
| --- | ---: | ---: |
| layer-0 `core_attn_out_normed_last_token_flat` | `0.182265430` | `0.097800157` |
| layer-0 `core_attn_out_normed_last_token_alt_heads_first` | `0.372663929` | `0.253684142` |
| layer-0 `out_proj_input` | `0.182265430` | `0.097800157` |
| layer-0 `post_attention` | `0.112992007` | `0.112992007` |
| layer-0 `post_mlp` | `0.000826810` | `0.000826810` |

#### Post-norm conversion plus non-inverted output projection

Run:

- `<validation-workspace>/runs/moe35b_gguf_tp4_postnorm_nooutprojinv_eager4k_dot20_20260628T032000Z`

Comparison artifact:

- `<validation-workspace>/runs/moe35b_postnorm_nooutprojinv_compare_20260628T032000Z/postnorm_nooutprojinv_selected.tsv`

This candidate paired the post-norm heads-first input conversion with disabling
the GGUF inverse-column conversion for `linear_attn.out_proj.weight`. It
returned the wrong `ymar Coll` branch and also made the residual path worse:

| label | first-rank cosine | mean cosine |
| --- | ---: | ---: |
| layer-0 `out_proj_input` | `0.483958924` | `0.226750694` |
| layer-0 `out_proj_output` | `-0.096042309` | `-0.096042309` |
| layer-0 `post_attention` | `-0.096042309` | `-0.096042309` |
| layer-1 `out_proj_input` | `-0.003516220` | `-0.002804815` |
| layer-2 `out_proj_input` | `-0.028061733` | `-0.024915432` |

### Outcome

Promote the output-projection layout boundary as a real source target. The
full-token trace shows that a heads-first final-token view is HF-aligned while
the current flattened `out_proj_input` is not.

Reject the naive global heads-first conversion as a fix. It makes the early
residual path substantially worse and does not recover the HF branch.

Reject the pre-norm conversion and the paired post-norm/non-inverted-output-
projection candidate. Both produce new wrong branches and degrade layer-0
agreement far below the original GGUF baseline.

### Promote / Reject

Promote:

- a narrower layout fix that preserves full sequence semantics, not only the
  final-token view;
- tracing `core_attn_out` stride/shape before and after the backend call, norm,
  and flatten;
- checking whether the GGUF backend writes `[heads, tokens, dim]` while HF/vLLM
  expects `[tokens, heads, dim]`, and whether the correct conversion must occur
  before norm or only for selected decode tokens.

Reject:

- global post-norm heads-first reshaping before `out_proj`;
- pre-norm heads-first reshaping before norm;
- pairing post-norm heads-first reshaping with a non-inverted
  `linear_attn.out_proj.weight`;
- another performance ladder before branch correctness returns;
- treating the close alternate final-token view as sufficient proof of a patch.

### Next

Instrument shape, stride, and selected multi-token slices around
`core_attn_out`, `z`, and `out_proj.weight`. The next patch should compare
sequence-wide layout and the paired weight layout instead of changing only the
activation view. The active suspicion is now a more specific interaction among
GGUF V-head weight reordering, GDN output storage, `z`/norm pairing, and
`out_proj` column layout. Do not run another throughput ladder until a strict
branch probe returns the HF `Thinking Process` path.

## GGUF-129 - MoE Linear-Attention Static Weight Contract Probe

Run artifacts:

- `.20` run:
  `<validation-workspace>/runs/moe35b_linear_attn_weight_contract_lazy_20260628T031701Z`
- scalar-transform follow-up:
  `<validation-workspace>/runs/moe35b_ssm_scalar_transform_probe_20260628T031923Z`

### Outcome

The read-only layer-0/layer-1 HF safetensor versus GGUF comparison confirms
that the major Qwen3.5-MoE linear-attention tensor layout rules are already
known and mostly matched by the current loader contract:

| Tensor family | Best static mapping | Result |
| --- | --- | --- |
| `q` rows | direct | exact within float roundoff |
| `k` rows | direct | exact within float roundoff |
| `v` rows | GGUF inverse value-head row order | exact within float roundoff |
| `z` / gate rows | GGUF inverse value-head row order | exact within float roundoff |
| `conv_q` / `conv_k` | direct | exact |
| `conv_v` | GGUF inverse value-head row order | exact |
| `linear_attn.out_proj.weight` | GGUF inverse value-head column order | exact within float roundoff |
| `ssm_norm` | direct | exact |

Representative layer-0 values:

- `q` direct mean absolute diff: `1.11540707e-11`
- `k` direct mean absolute diff: `9.80893144e-12`
- `v` inverse-row mean absolute diff: `6.570227e-12`
- `z` inverse-row mean absolute diff: `5.64922848e-12`
- `conv_v` inverse-row mean absolute diff: `0`
- `out_proj` inverse-column mean absolute diff: `6.73216864e-12`

The scalar/projection follow-up shows raw GGUF `ssm_a`, `ssm_dt.bias`,
`ssm_alpha.weight`, and `ssm_beta.weight` are not direct HF mirrors. That
matches the existing source-lineage note that GGUF stores some SSM values in a
converted form, and it means raw static parity is not the right acceptance test
for those tensors. Runtime validation still has to prove the loader conversion
and compiled GDN inputs are correct.

### Promote / Reject

Promote:

- the existing inverse value-head row mapping for `v`, `z`, and `conv_v`;
- the existing inverse value-head column mapping for `linear_attn.out_proj`;
- treating q/k direct row layout as already proven for layers 0 and 1;
- moving the next source slice away from broad raw-weight layout guessing and
  toward runtime GDN input/value-contract validation.

Reject:

- a new standalone output-projection weight-column patch;
- another global activation heads-first conversion based only on the
  close final-token alternate view;
- treating raw GGUF `ssm_a` / `dt` / alpha / beta mismatch as proof of a new
  bug without checking the active loader conversion.

### Next

Trace the runtime loaded tensors or compiled graph inputs after the active
loader transformations have been applied. The static probe says the easy
weight-layout mismatch is not the remaining MoE GGUF performance blocker.

## GGUF-130 - MoE Graph Replay And Structural Profiling Readout

Run artifacts:

- GGUF replay timing:
  `<validation-workspace>/runs/moe35b_gguf_tp4_current_cgreplay_timing_dot20_20260627T221003Z/cudagraph_replay_timing.tsv`
- HF release-patch replay timing:
  `<validation-workspace>/runs/moe35b_hf_tp4_releasepatch_cgreplay_dot20_20260627T232912Z/cudagraph_replay_timing.tsv`
- GGUF graph dump:
  `<validation-workspace>/runs/moe35b_gguf_tp4_hflayeralias_lmheadunquant_dot20_20260627T170500Z/compile_cache/rank_0_0/backbone/computation_graph.py`
- HF graph dump:
  `<validation-workspace>/runs/moe35b_hf_tp4_matched_graphdump_dot20_20260627T161707Z/compile_cache/rank_0_0/backbone/computation_graph.py`

### Outcome

The comparable compiled graph dumps have the same high-level operation counts:

| Pattern | GGUF count | HF count |
| --- | ---: | ---: |
| `gdn_attention_core` | `120` | `120` |
| `fused_moe` | `80` | `80` |
| `moe_forward` | `240` | `240` |
| `rocm_unquantized_gemm` | `440` | `440` |
| `in_proj_qkvz` | `240` | `240` |
| `in_proj_ba` | `240` | `240` |
| `out_proj` | `270` | `270` |
| `linear_attn` | `990` | `990` |
| `reshape` | `830` | `830` |
| `view` | `1190` | `1190` |
| `contiguous` | `300` | `300` |
| `all_reduce` | `487` | `487` |

The graph-level replay timing still shows the performance gap:

| Path | FULL records | Average replay ms | Early average ms | Late average ms |
| --- | ---: | ---: | ---: | ---: |
| GGUF TP4 | `1952` | `12.054565` | `11.220266` | `12.238473` |
| HF TP4 release-patch control | `1980` | `7.887792` | `7.601556` | `7.942541` |

Earlier timing probes already rejected the steady GDN core, steady FusedMoE
runner body, logits processing, and sampler as standalone bottlenecks. This
structural count rejects a gross compiled-graph-shape mismatch as well.

### Promote / Reject

Promote:

- whole-graph replay / kernel-scheduling as the active performance boundary;
- graph-input value/stride validation after loader transforms;
- lower-level kernel selection / replay drift inspection rather than another
  unchanged warmup ladder.

Reject:

- "GGUF compiled a totally different high-level model graph" as the explanation;
- logits/sampler timing as the primary throughput gap;
- steady unquantized FusedMoE runner timing as the primary throughput gap;
- another broad benchmark run until either graph replay time or branch
  correctness has a sharper source change to test.

### Next

Build the next diagnostic around graph-visible input metadata and replayed
kernel selection for the identical high-level graph. The target is to explain
why the GGUF replayed graph takes about `4.17` ms more per decode token than
the HF release-patch control while showing the same high-level operation shape.

## GGUF-131 - MoE Compile-Cache Kernel Artifact Comparison

Run artifacts:

- Summary root:
  `<validation-workspace>/runs/moe35b_graph_kernel_surface_compare_20260628T032546Z/`
- Concise content summary:
  `<validation-workspace>/runs/moe35b_graph_kernel_surface_compare_20260628T032546Z/concise-content-summary.md`
- Kernel normalized compare:
  `<validation-workspace>/runs/moe35b_graph_kernel_surface_compare_20260628T032546Z/kernel-normalized-compare.md`
- Best-config distribution:
  `<validation-workspace>/runs/moe35b_graph_kernel_surface_compare_20260628T032546Z/best-config-distribution.md`

### Outcome

The compile-cache comparison reinforces the graph-surface result. HF and GGUF
both produced `152` content-manifest entries. GGUF had `110` unique content
hashes, HF had `102`, and `60` unique hashes were common. The extension totals
matched exactly by class: each side had `136` `.best_config` files, `4` JSON
files, and `12` generated Python files.

The generated Python artifacts are almost the same size and shape. GGUF
generated Python totals were `7,780,819` bytes and `90,990` lines; HF generated
Python totals were `7,761,367` bytes and `90,630` lines. The largest generated
kernel files were all about `1.03` MB and `14,1xx` lines per rank on both
paths. A marker-count scan over the generated kernels matched exactly for the
important op families:

| Marker | GGUF count | HF count |
| --- | ---: | ---: |
| `torch.ops.vllm.rocm_unquantized_gemm` | `1360` | `1360` |
| `torch.ops.vllm.all_reduce` | `980` | `980` |
| `torch.ops.vllm.moe_forward_shared` | `832` | `832` |
| `async_compile.triton` | `200` | `200` |
| `torch.ops.vllm.gdn_attention_core` | `120` | `120` |
| `torch.ops.vllm.unified_kv_cache_update` | `80` | `80` |
| `torch.ops.vllm.unified_attention_with_output` | `48` | `48` |

After normalizing generated paths and source-line comments, the first
generated-source diffs were internal FX/ATen symbol-number differences such as
`add_128` versus `add_137`, not missing kernels or a different operation
family. The per-kernel marker inventory still matched.

The best-config distribution did differ. GGUF had `97` unique best-config
hashes, HF had `89`, and `60` were common. That leaves `37` GGUF-only unique
best-config hashes and `29` HF-only unique best-config hashes.

The cache-key factor diff also found one environment-factor difference besides
the expected `code_hash` and `config_hash`: the GGUF graph recorded
`VLLM_USE_TRITON_AWQ=true`, while the HF graph recorded
`VLLM_USE_TRITON_AWQ=false`. This is not a new promotion candidate by itself.
The earlier env-only AWQ-off probe showed the ROCm patch could still force the
cache factor back to true, and the source-level AWQ-false probe proved the flag
could be set false but still stayed in the low MoE GGUF band. Treat this as a
cache-key factor to keep normalized in future comparisons, not as a likely
standalone fix.

### Promote / Reject

Promote:

- graph-visible input metadata, tensor stride/layout, and autotune-config
  selection as the current performance slice;
- replay/kind selection inside an otherwise matching generated kernel surface;
- a reduced graph-input manifest comparison before another full benchmark
  ladder.
- normalizing the `VLLM_USE_TRITON_AWQ` cache factor for cleaner future HF/GGUF
  comparisons.

Reject:

- missing generated kernel families as the explanation for the MoE GGUF gap;
- a high-level graph mismatch as the explanation;
- another broad C++/Triton source rewrite without a graph-input or autotune
  target.
- repeating AWQ-off as a standalone performance experiment without new
  kernel-level evidence.

### Reason

The profiling now says GGUF and HF are compiling the same high-level graph and
the same generated-kernel family surface, but they do not share all generated
content hashes or autotune best-config hashes. That is consistent with the
observed replay gap being caused by graph-visible metadata, value/stride
differences, or per-shape autotune choices rather than a missing op.

### Next

Build a graph-input manifest for the GGUF and HF graph-capture paths. Compare
input shapes, strides, dtype, storage offset, contiguity, and aliases for the
compiled graph parameters and dynamic tensors. If the graph-input manifests are
equivalent, compare the differing `.best_config` files by normalized kernel
role and shape before changing source.

## GGUF-230 - MoE Graph Input And Best-Config Normalized Manifest

Run artifacts:

- Summary root:
  `<validation-workspace>/runs/moe35b_graph_kernel_surface_compare_20260628T032546Z/`
- Forward signature compare:
  `<validation-workspace>/runs/moe35b_graph_kernel_surface_compare_20260628T032546Z/forward-signature-manifest-compare.md`
- Generated tensor metadata compare:
  `<validation-workspace>/runs/moe35b_graph_kernel_surface_compare_20260628T032546Z/tensor-metadata-manifest-compare.md`
- Path-normalized best-config compare:
  `<validation-workspace>/runs/moe35b_graph_kernel_surface_compare_20260628T032546Z/best-config-path-normalized-compare.md`
- Best-config mismatch cache-hash probe:
  `<validation-workspace>/runs/moe35b_graph_kernel_surface_compare_20260628T032546Z/best-config-mismatch-triton-cachehash-probe.md`

### Outcome

The graph-input manifest rejects a graph signature mismatch. GGUF and HF both
have `576` graph inputs, and the normalized forward-signature manifests have
the same SHA-256 hash. The type counts also match exactly:

| Type | Count |
| --- | ---: |
| `f16[2048]` | `162` |
| `vllm_utils_torch_utils_ModuleName` | `80` |
| `f16[2048, 1024]` | `80` |
| `f16[3072, 2048]` | `60` |
| `f16[16, 2048]` | `60` |
| `f16[128]` | `60` |
| `f16[256]` | `40` |
| `f16[2560, 2048]` | `20` |

The generated-kernel tensor metadata also matches exactly. Both paths have
`2112` generated metadata entries, `112` unique metadata strings, and the same
metadata-count manifest hash. This includes the lowered tensor
shape/stride/device strings in the generated kernel fragments, so the current
cache does not show a graph-visible stride or device mismatch.

The best-config path comparison is narrower than the earlier raw content-hash
comparison implied. Both paths have the same `136` `.best_config` paths. Raw
file hashes differ on `81` paths, but most of that is timing/cache-hash noise.
After normalizing out `time_taken_ms`, only `14` paths still differ in the
actual autotune parameters. Representative mismatches include:

| Key | HF params | GGUF params |
| --- | --- | --- |
| `733b5481...` rank 0 | `XBLOCK=4, warps=1` | `XBLOCK=8, warps=1` |
| `8c94c32e...` ranks 0/1 | `XBLOCK=1024, warps=4` | `XBLOCK=512, warps=4` |
| `a616ff67...` ranks 0/3 | `XBLOCK=256, warps=4` | `XBLOCK=512, warps=2` |
| `d7cbaff6...` rank 0 | `XBLOCK=512, warps=4` | `XBLOCK=1024, warps=4` |

Some mismatched choices appear rank-swapped rather than absent. For example,
`8c94c32e...` uses `XBLOCK=512` on HF rank 3 and GGUF ranks 0/1, while
`XBLOCK=1024` appears on HF ranks 0/1 and GGUF rank 3. The config IDs and
Triton cache hashes are not referenced textually in the generated Python
source, so this pass did not map each mismatch to a named generated function.

### Promote / Reject

Promote:

- the `14` normalized best-config parameter mismatches as the current
  compile-cache performance target;
- a controlled compile-cache/autotune experiment that forces GGUF to use the
  HF parameter choices for only those mismatched keys;
- graph replay timing after best-config normalization before making another
  source change.

Reject:

- graph-input shape/type mismatch as the explanation;
- generated-kernel tensor shape/stride/device mismatch as the explanation;
- raw best-config content-hash mismatch as sufficient evidence, because most
  raw mismatches are timing/cache-hash noise;
- another broad benchmark ladder before either best-config normalization or a
  runtime value/state difference is tested.

### Reason

The active MoE GGUF performance gap now has a narrower compile-cache boundary:
same graph signature, same generated tensor metadata, same generated kernel
families, same best-config path set, but `14` normalized autotune parameter
choices differ. That is a tractable experiment. If forcing those choices does
not move replay timing toward HF, the remaining gap is no longer compile
selection and should move to runtime value/state or replay scheduling.

### Next

Build a disposable GGUF compile-cache normalization test. The safest version is
to force only the `14` mismatched `.best_config` parameter choices to the HF
values before GGUF recompilation, then compare graph replay timing before
running the full benchmark ladder. Do not promote MoE GGUF until the normal
warmups -> strict -> `c1_2000` -> `c1_10000` sequence matches or beats the
current MoE target.

## GGUF-232 - MoE HF Best-Config Preseed Performance Probe

Run artifacts:

- Run root:
  `<validation-workspace>/runs/moe35b_gguf_tp4_bestconfig_hfpreseed_dot20_20260628T040600Z/`
- Benchmark summary:
  `<validation-workspace>/runs/moe35b_gguf_tp4_bestconfig_hfpreseed_dot20_20260628T040600Z/benchmark_ladder/summary.md`
- Replay timing:
  `<validation-workspace>/runs/moe35b_gguf_tp4_bestconfig_hfpreseed_dot20_20260628T040600Z/cudagraph_replay_timing.tsv`
- Best-config final compare:
  `<validation-workspace>/runs/moe35b_gguf_tp4_bestconfig_hfpreseed_dot20_20260628T040600Z/best-config-preseed-final-compare.tsv`

### Setup

This was a disposable performance-slice probe on `.20`. The base launcher was
the GGUF MoE TP4 hflayeralias/lm-head-unquant graph-dump launcher, with the
CUDA-graph replay timing patch added. Before launch, the run preseeded the
`14` parameter-mismatched `.best_config` paths from the matched HF TP4 graph
dump into the new GGUF compile cache. The base launcher still inherited the
state-convert branch that later correctness probes rejected, so this run is not
a final promotion candidate. It is a compile-cache hypothesis test only.

The run used the normal benchmark sequence:

- `8` pre-measure warmups;
- uncapped `c1_128` strict;
- `c1_2000`;
- `c1_10000`.

### Outcome

The best-config preseed did not move MoE GGUF into the published TP4 band.

| case | backend decode TPS | finish | strict gate valid |
| --- | ---: | --- | --- |
| `c1_128_strict` | `82.686` | `stop` | `True` |
| `c1_2000` | `84.181` | `length` | `False` |
| `c1_10000` | `74.301` | `length` | `False` |

Warmups after first compile settled around `84.15` to `84.22` backend TPS,
which matched the same low GGUF MoE band seen in earlier runs. Replay timing
improved only slightly:

| replay set | FULL rows | average ms | min ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| GGUF baseline graph compare | `1952` | `12.054565` | `11.220266` | `12.238473` |
| HF control graph compare | `1980` | `7.887792` | `7.601556` | `7.942541` |
| GGUF HF-best-config preseed | `1944` | `11.771824` | `10.706855` | `14.868934` |

After the run completed, the compile cache held `136` `.best_config` files.
Of the original `14` mismatched paths, `9` retained the HF parameter choice
and `5` reverted to the GGUF baseline choice.

### Promote / Reject

Promote:

- compile-cache preseed as a useful diagnostic for proving that the
  path-normalized `.best_config` differences are not sufficient by themselves;
- runtime value/state or replay scheduling differences after graph capture as
  the next performance slice;
- keeping the normal warmup ladder for performance probes, because first-run
  compile/warmup behavior materially differs from settled decode.

Reject:

- the `14` best-config parameter mismatches as the primary explanation for the
  MoE GGUF gap;
- another standalone best-config copy/preseed run without a stronger hook that
  also changes replay scheduling or runtime values;
- promoting this branch, because the state-convert family remains a
  correctness-rejected diagnostic branch.

### Reason

The experiment normalized most of the known best-config parameter mismatches
yet left MoE GGUF in the same `~74` backend TPS c1_10000 band. The replay
average moved only about `0.28` ms toward HF, while the original replay gap was
about `4.17` ms. That is too small to explain the benchmark gap. The remaining
MoE blocker is therefore not just TorchInductor choosing a few different
pointwise/autotune configs.

### Next

Move the active MoE performance work away from raw best-config selection.
The next useful target is runtime value/state or replay scheduling inside the
already-matching high-level graph: compare graph-captured runtime values and
per-call replay timing around GDN/linear-attention state, residual updates,
and MoE route amplification. Do not run another full benchmark ladder until a
source or runtime-state change has evidence that it changes settled replay
timing or restores a closer HF trajectory.

## GGUF-233 - MoE Replay-Control Generated-Code Parity Check

Run artifacts:

- GGUF replay-control run:
  `<validation-workspace>/runs/moe35b_gguf_tp4_current_cgreplay_timing_dot20_20260627T221003Z/`
- HF release-patch replay-control run:
  `<validation-workspace>/runs/moe35b_hf_tp4_releasepatch_cgreplay_dot20_20260627T232912Z/`
- GGUF graph-input metadata:
  `<validation-workspace>/runs/moe35b_gguf_tp4_graphinput_meta_fixed_dot20_20260627T225531Z/graph_input_meta.tsv`
- HF release-patch graph-input metadata:
  `<validation-workspace>/runs/moe35b_hf_tp4_releasepatch_graphinput_meta_dot20_20260627T235328Z/graph_input_meta.tsv`

### Setup

This was a read-only profiling pass on existing `.20` artifacts. It did not
launch a new server or run another benchmark ladder. The goal was to verify
whether the remaining MoE GGUF replay gap maps to generated-code selection,
stale graph-input metadata, or runtime values/state flowing through the
already-captured graph.

The comparison used the valid release-patched HF graph-input metadata. The
older stock-source HF metadata still shows a stale `3072 + 16` split, but that
is not the active comparator for the release-patched MoE path.

### Outcome

Replay timing remains the main performance boundary:

| path | FULL replay rows | average ms | min ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| GGUF current replay-control | `1952` | `12.054565` | `10.880300` | `15.272617` |
| HF release-patch replay-control | `1980` | `7.887792` | `7.481852` | `9.125901` |
| GGUF HF-best-config preseed | `1944` | `11.771824` | `10.706855` | `14.868934` |

Per-rank replay timing was balanced on both paths. GGUF had `488` FULL samples
per rank with means from `12.053268` to `12.055619` ms. HF had `495` FULL
samples per rank with means from `7.886261` to `7.889269` ms. This rejects a
single bad rank or single slow GPU as the replay explanation.

The corrected graph-input metadata comparison rejects the stale `3072 + 16`
split as a live patch target:

| graph-input marker | GGUF metadata | HF release-patch metadata |
| --- | ---: | ---: |
| `shape=(3088, 2048)` | `1464` | `1496` |
| `shape=(3072, 2048)` | `0` | `0` |
| `shape=(16, 2048)` | `0` | `0` |
| `stride=(3088, 128, 1)` | `1464` | `1492` |
| `stride=(3072, 128, 1)` | `0` | `0` |

The generated TorchInductor Python source bodies are identical as a content
multiset. Both replay-control roots had:

- `208` generated `.py` files;
- `140` `.best_config` files;
- `76` `.kernel_perf` files;
- `4372` `triton_` source markers;
- `812` `all_reduce` markers;
- `444` `moe_forward` markers;
- `100` clone markers;
- `672` copy markers.

Hashing all generated `.py` files by content produced `208/208` common hashes
with `0` GGUF-only and `0` HF-only Python source bodies.

### Promote / Reject

Promote:

- runtime tensor value/state tracing inside the already-matching graph as the
  next MoE performance slice;
- graph-input value or compact tensor-signature probes that include norms,
  hashes, strides, and selected activation/state summaries after loader
  transforms;
- replay-control comparisons using the release-patched HF source bundle only.

Reject:

- generated Python code body selection as the remaining MoE GGUF explanation;
- the stale stock-HF `3072 + 16` graph-input contrast as a patch target;
- a single-rank or host-lane explanation for the current replay gap;
- another broad benchmark ladder without a new runtime value/state hypothesis.

### Reason

The generated graph code is now proven identical at the Python source-body
level, and the valid release-patched HF metadata uses the same fused `3088`
graph-input family as GGUF. Yet completed FULL replay remains roughly
`4.17` ms slower for GGUF. The remaining explanation must be lower than
generated-code selection: runtime values, captured tensor state, graph input
contents/strides not visible in the existing metadata, or replay-time
scheduling around those values.

### Next

Build a small value-signature diagnostic instead of another performance
ladder. The next probe should log compact graph-input or layer-boundary
signatures for GGUF and release-patched HF during the same strict prompt
window: dtype, shape, stride, contiguity, finite counts, norm/mean, and a small
stable sample hash for GDN/linear-attention state, post-attention residual,
router outputs, and MoE block inputs. Run it first on a short correctness
window; only rerun the normal warmups -> strict -> `c1_2000` -> `c1_10000`
ladder after a source/runtime-state change moves those signatures or replay
timing toward HF.

## GGUF-234 - MoE Replay Graph-Input Value Signature Probe

Run artifacts:

- HF value-signature run:
  `<validation-workspace>/runs/moe35b_hf_tp4_valuesig_replay_dot20_20260628T042651Z/`
- GGUF value-signature run:
  `<validation-workspace>/runs/moe35b_gguf_tp4_valuesig_replay_dot20_20260628T044000Z/`
- Comparison summary:
  `<validation-workspace>/runs/moe35b_valuesig_replay_compare_20260628T0450Z/summary.md`

### Setup

This was a short `.20` diagnostic using a CUDA-graph input value-signature
overlay. The overlay logged compact graph-input metadata and sampled values
during capture and replay without full tensor reductions. Both HF and GGUF
runs used the same short prompt:

```text
Write one concise sentence about local AI reproducibility.
```

Both returned coherent `Thinking Process` starts. The probe was diagnostic
only; it was not a benchmark ladder.

### Outcome

HF and GGUF had the same graph-input replay structure:

| path | lines | capture headers | replay headers |
| --- | ---: | ---: | ---: |
| HF | `8092` | `384` | `288` |
| GGUF | `8092` | `384` | `288` |

The replay header diff was empty. Both paths had the same replay header
distribution, including `124` FULL replay entries and matching PIECEWISE
argument-count families.

The sampled replay values differed, as expected for HF versus GGUF-loaded
weights and activations, while the same sampled tensor shapes and strides were
present. Representative differing sampled graph inputs included contiguous
projection/weight-shaped tensors such as `(2048, 1024)` and `(3088, 2048)`.

The fastpath logs initially looked asymmetric:

| path | active | rejected |
| --- | ---: | ---: |
| HF value-signature run | `4` | `0` |
| GGUF value-signature run | `4` | `204` |

However, later source inspection and matched-debug HF runs show this must not
be over-interpreted: the release TopK8 helper is a one-token path, and grouped
`shape_or_layout` rejections are visible when HF is launched with the same
debug/force path.

### Promote / Reject

Promote:

- graph-input replay structure parity as further evidence that the remaining
  MoE GGUF gap is below high-level graph selection;
- runtime value/state tracing as useful, but only when tied to a specific
  source hypothesis;
- keeping the corrected release-patched HF comparator for future graph-value
  probes.

Reject:

- treating value differences alone as a patch target;
- treating grouped TopK8 rejection logs from this run as the root cause;
- running another benchmark ladder without a candidate that first improves
  branch correctness, replay timing, or a narrowed trace boundary.

### Reason

The probe confirms that HF and GGUF replay the same graph-input structure, but
not the same values. That is useful only as a narrowing result: the remaining
gap is not a missing graph family or a broad replay-header mismatch. It points
back to active loaded values/state, GatedDeltaNet layout semantics, or lower
replay-time kernel behavior inside a structurally matching graph.

### Next

Do not chase the TopK8 grouped rejection by itself. Continue from the
linear-attention trace boundary: layer-0 q/k/v/z inputs match closely, but the
full-token output-projection input assembly diverges. Candidate fixes must be
tested first on a short correctness/trace window before any full ladder.

## GGUF-235 - Current Fused-QKVZBA Post-Norm Head-Gather Candidate

Run artifacts:

- Scratch patch:
  `<validation-workspace>/experimental-patches/qwen35moe-gguf-current-fusedqkvzba-postnorm-headgather-20260628T0525Z/`
- Run root:
  `<validation-workspace>/runs/moe35b_gguf_tp4_current_fusedqkvzba_headgather_eager4k_dot20_20260628T0525Z/`

### Setup

This was a narrow source candidate against the current fused `qkvzba` GGUF
branch, not the older split-`ba` trace branch. The only intended behavioral
change was an opt-in output-projection gather:

```text
VLLM_QWEN35_GGUF_OUTPROJ_HEADS_FIRST_GATHER=1
```

After `self.norm(core_attn_out, z)`, the candidate gathered normalized GDN
rows as heads-first/token-second before `out_proj`. The test used the same
short two-token branch prompt as the earlier HF/GGUF full-token trace and
ran in 4K eager mode with TP4 and P2P on. It did not run the benchmark ladder.

### Outcome

The candidate failed the short correctness check. The response reasoning field
was a non-coherent single glyph rather than the HF `Thinking Process` branch.

### Promote / Reject

Promote:

- the current fused-`qkvzba` candidate as a clean rejection of post-norm
  heads-first gathering as a standalone fix;
- the need to test source candidates on the current fused branch, because older
  split-`ba` candidates are not directly comparable;
- moving the next source slice lower than the output-projection gather.

Reject:

- `VLLM_QWEN35_GGUF_OUTPROJ_HEADS_FIRST_GATHER=1` as a correctness or
  benchmark candidate;
- a full warmup/strict/`c1_2000`/`c1_10000` ladder for this patch;
- interpreting the close alternate heads-first final-token trace as a direct
  patch recipe.

### Reason

The close alternate heads-first trace remains a useful diagnostic, but a simple
post-norm gather corrupts the branch on the current fused path. The actual
boundary is lower than a final gather: GatedDeltaNet output storage, norm
pairing with `z`, sequence/head layout, or the custom op's write/consume
contract.

### Next

Inspect the GatedDeltaNet custom-op contract and the active loaded state values
after loader transforms. A useful next candidate should improve the short
branch or trace boundary first; only then run the normal warmups -> `c1_128`
uncapped strict -> `c1_2000` -> `c1_10000` ladder.

## GGUF-236 - MoE GDN Boundary Layout-Metadata Trace

Run artifacts:

- Scratch patch:
  `<validation-workspace>/experimental-patches/qwen35moe-gguf-gdn-layoutmeta-20260628T0115Z/`
- Run root:
  `<validation-workspace>/runs/moe35b_gguf_tp4_gdn_layoutmeta_eager4k_dot20_20260628T0120Z/`
- Compact layout summary:
  `<validation-workspace>/runs/moe35b_gguf_tp4_gdn_layoutmeta_eager4k_dot20_20260628T0120Z/gdn_layout_unique.tsv`

### Setup

This was a trace-only follow-up to `GGUF-235` on the current fused `qkvzba`
GGUF branch. It did not change model math or run a benchmark ladder. The patch
extended the existing tensor trace helper to include stride, storage offset,
and contiguity metadata at the GatedDeltaNet / output-projection boundary.

The run used the official v0.2 image, TP4, P2P-on, FP16, 4K eager mode, and the
same two-token branch prompt used by the recent HF/GGUF full-token trace
experiments.

### Outcome

The short request remained coherent: the response started the expected
`Thinking Process` branch and stopped by `max_tokens` length. The trace files
captured the selected layers and tensors.

For layers `0`, `1`, `2`, `5`, and `10`, the compact summary contained `35`
unique `label | shape | layout` triples. Every selected boundary tensor was
contiguous with the expected local stride and `storage_offset=0`, except the
intentional final-token row view:

| tensor family | observed layout |
| --- | --- |
| `core_attn_out_raw` | `shape=(32, 8, 128)`, `stride=(1024, 128, 1)`, contiguous |
| `z_for_norm` | `shape=(256, 128)`, `stride=(128, 1)`, contiguous |
| `core_attn_out_normed` | `shape=(256, 128)`, `stride=(128, 1)`, contiguous |
| `core_attn_out_normed_last_token_flat` | `shape=(1024,)`, `stride=(1,)`, contiguous row view with `storage_offset=31744` |
| `core_attn_out_normed_last_token_alt_heads_first` | `shape=(1024,)`, `stride=(1,)`, contiguous materialized alternate view |
| `out_proj_input` | `shape=(32, 1024)`, `stride=(1024, 1)`, contiguous |
| `out_proj_output` | `shape=(32, 2048)`, `stride=(2048, 1)`, contiguous |

### Promote / Reject

Promote:

- the layout-metadata trace as a clean rejection of a visible Python tensor
  stride/storage-offset mismatch at the GDN/output-projection boundary;
- the existing full-token value trace as the still-relevant evidence that the
  values differ even when the exposed tensor layouts are normal;
- moving the next source slice below this Python-visible boundary.

Reject:

- another standalone reshape/gather patch based only on tensor contiguity;
- treating `out_proj_input` as non-contiguous or storage-offset shifted in the
  current fused GGUF path;
- running the normal throughput ladder for this trace-only diagnostic.

### Reason

The previous full-token trace showed value-direction drift between
`core_attn_out_normed` views and `out_proj_input`. This trace shows that drift
is not explained by an obvious Python-level stride, contiguity, or storage
offset bug in the selected GDN boundary tensors. The remaining performance and
trajectory gap is therefore lower than this surface: custom-op write/consume
contract, recurrent GDN state/value semantics, or replay-time scheduling around
those values.

### Next

Move the next MoE GGUF source slice lower than Python-boundary layout:

- inspect the C++/custom-op `gdn_attention_core` write contract and whether the
  replayed graph consumes token/head order identically for HF and GGUF;
- add lower-level timing or compact value markers around GDN state,
  post-attention residual, and MoE route amplification only if they can run
  without breaking graph capture;
- do not run another full warmup -> `c1_128` strict -> `c1_2000` ->
  `c1_10000` ladder until a branch probe or replay-timing probe moves toward
  the HF trajectory.

## GGUF-237 - MoE Replay-Timing Bucket Re-Read

Run artifacts:

- GGUF baseline replay timing:
  `<validation-workspace>/runs/moe35b_gguf_tp4_current_cgreplay_timing_dot20_20260627T221003Z/cudagraph_replay_timing.tsv`
- HF release-patch replay timing:
  `<validation-workspace>/runs/moe35b_hf_tp4_releasepatch_cgreplay_dot20_20260627T232912Z/cudagraph_replay_timing.tsv`
- GGUF HF-best-config preseed replay timing:
  `<validation-workspace>/runs/moe35b_gguf_tp4_bestconfig_hfpreseed_dot20_20260628T040600Z/cudagraph_replay_timing.tsv`

### Setup

This was a read-only analysis pass over existing `.20` replay logs. No server
was launched and no benchmark ladder was run. The timing files are key-value
logs, not true TSV, so this pass grouped `elapsed_ms` by `mode` and 512-call
buckets.

### Outcome

The replay gap is stable and decode-window dependent, not a one-time startup
artifact.

Overall FULL replay:

| path | FULL rows | average ms | min ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| GGUF baseline | `1952` | `12.054565` | `10.880300` | `15.272617` |
| HF release-patch control | `1980` | `7.887792` | `7.481852` | `9.125901` |
| GGUF HF-best-config preseed | `1944` | `11.771824` | `10.706855` | `14.868934` |

Representative call buckets:

| path | early FULL bucket | late FULL bucket |
| --- | --- | --- |
| GGUF baseline | calls `0-511`: `11.220266` ms | calls `31232-31743`: `14.970377` ms |
| HF control | calls `0-511`: `7.601556` ms | calls `31744-32255`: `8.891808` ms |
| GGUF best-config preseed | calls `0-511`: `11.024393` ms | calls `30720-31231`: `14.737254` ms |

GGUF is already about `3.62` ms slower than HF in the early FULL bucket, and
the late-decode gap grows to about `6.08` ms. The best-config preseed reduces
early replay time by roughly `0.20` ms and late replay time by roughly
`0.23` ms, which is useful but far too small to explain the MoE gap.

### Promote / Reject

Promote:

- long-decode replay drift as a real part of the MoE GGUF performance gap;
- probes that distinguish steady one-token graph replay from state/cache
  effects that accumulate across long decode;
- lower-level profiling around recurrent GDN state, residual stream, and
  MoE route/output amplification over time.

Reject:

- treating the GGUF gap as only a first-request or warmup artifact;
- expecting best-config preseed alone to close the c1_10000 gap;
- another full ladder unless a candidate first changes replay timing or
  branch trajectory.

### Reason

The early replay gap means GGUF enters steady FULL replay slower than HF even
before long-context drift accumulates. The late replay buckets widen the gap
substantially, which matches the benchmark pattern where c1_10000 falls farther
behind than short strict/c1_2000 windows. The active source target should
therefore include replay-time state/cache behavior across decode length, not
only static graph shape or compile-cache selection.

### Next

Build the next diagnostic around time-evolving replay state: compact per-window
signatures for GDN recurrent state, post-attention residual, router/expert
outputs, and graph replay timing buckets. Prefer a short controlled prompt
that can sample early and late decode windows before spending another full
benchmark ladder.

## GGUF-238 - MoE GGUF Call-Set Early/Late State Trace

Run artifacts:

- Patch:
  `<validation-workspace>/experimental-patches/qwen35moe-gguf-callset-state-20260628T0535Z`
- Run root:
  `<validation-workspace>/runs/moe35b_gguf_tp4_callset_state_eager4k_dot20_20260628T0538Z`
- Trace summary:
  `<validation-workspace>/runs/moe35b_gguf_tp4_callset_state_eager4k_dot20_20260628T0538Z/callset_state_summary.tsv`
- Compact readout:
  `<validation-workspace>/runs/moe35b_gguf_tp4_callset_state_eager4k_dot20_20260628T0538Z/callset_compact_readout.tsv`

### Setup

This was a bounded `.20` GGUF-only diagnostic, not a benchmark ladder. It used
the published ROCm7.2 Dense/MoE runtime image with TP4, P2P-on
(`NCCL_P2P_DISABLE=0`), FP16, `--enforce-eager`, `MAX_MODEL_LEN=4096`, and
`--max-num-batched-tokens 1024`. The request used the normal benchmark-style
chat wrapper with a 512-token cap so the tracer could sample early and late
decode calls without spending a full c1_10000 run.

The patch added a sparse call-set gate:

- `VLLM_GFX906_CALL_TRACE_SET=0,1,2,64,128,256,511`
- `VLLM_GFX906_CALL_TRACE_LAYERS=0,1,2,5,10,20,39`
- `VLLM_GFX906_CALL_TRACE_MAX_EVENTS=1`

The decoder call counter now increments outside the normal trace-active block,
so late calls can be sampled without logging every token. The current patch
only traced decoder and linear-attention state; it did not add router/MoE
wrapper tracing because the active fused source lacks the sequence-parallel
helper imports needed for a low-risk wrapper in this pass.

### Outcome

The request completed cleanly:

- `finish_reason=length`
- prompt tokens: `143`
- completion tokens: `512`
- content characters: `0`
- reasoning characters: `2382`
- reasoning head: `Here's a thinking process:`

This was coherent enough for a trace harness, but it remained reasoning-only
under the short cap and is not a correctness or performance promotion.

Trace coverage:

- captured call labels: `2`, `64`, `128`, `256`, `511`
- labels per captured call: `3412`
- missing requested labels: `0`, `1`

The missing `0` and `1` labels likely mean the counter was already consumed by
setup/pre-request calls before the actual request window. Use call `2` as the
first captured request window for this run.

The prefix filter also overmatched: a prefix such as `model.layers.1` captured
layers such as `10`, `12`, and `13` because the current helper accepts
substring matches. The trace is still useful for a GGUF-only readout, but the
filter must be tightened before a paired HF/GGUF comparator.

Selected compact readout:

| signal | call `2` norm | call `64` norm | call `128` norm | call `256` norm | call `511` norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| layer 0 SSM state | `595.545525` | `335409.206055` | `384549.646485` | `500465.988282` | `610356.753906` |
| layer 10 SSM state | `18.612862` | `4340766.687500` | `4816365.000000` | `5892098.375000` | `7107090.625000` |
| layer 20 SSM state | `25.892348` | `20300971.250000` | `20795605.750000` | `23133636.000000` | `24548591.000000` |

The normalized downstream signals were much smaller after the early call:

| signal | call `2` norm | call `511` norm |
| --- | ---: | ---: |
| layer 0 post-attention layernorm | `326.962769` | `20.992693` |
| layer 39 post-MLP | `285.294495` | `11.129422` |

### Promote / Reject

Promote:

- sparse call-set tracing as a practical mechanism for early/late decode
  state probes;
- recurrent GDN/SSM state growth as the next paired-comparator target;
- a strict prefix-matching cleanup before running the same trace on HF and
  GGUF side by side.

Reject:

- treating this GGUF-only state growth as a proven root cause before an HF
  comparator exists;
- running a full benchmark ladder from this diagnostic-only eager setup;
- reusing the current substring prefix filter for a formal comparator.

### Reason

The tracer exposed very large recurrent-state norms that grow across decode,
while the normalized output/residual signals stay bounded. That shape is
consistent with the replay-bucket finding that the MoE GGUF gap exists early
and widens late, but it is not yet enough to identify a GGUF-specific bug.
HF may show the same internal state scale. The next useful experiment must
compare the same sparse call windows and layer labels against an HF control.

### Next

Tighten boundary prefix matching to exact path-token matches, then run a
paired HF comparator with the same call-set windows and selected layers. If HF
shows similar recurrent-state growth, move back toward replay scheduling,
kernel timing, or memory behavior. If HF differs materially, target recurrent
GDN state/cache semantics and the custom-op write/consume contract.

## GGUF-239 - MoE HF/GGUF Call-Set Comparator With Counter Fix

Run artifacts:

- Rejected HF patch using GGUF-derived model source:
  `<validation-workspace>/experimental-patches/qwen35moe-hf-callset-prefixfix-20260628T0640Z`
- Rejected HF run:
  `<validation-workspace>/runs/moe35b_hf_tp4_callset_state_eager4k_dot20_20260628T0640Z`
- Clean HF call-set patch with counter bug:
  `<validation-workspace>/experimental-patches/qwen35moe-hf-callset-clean-20260628T0720Z`
- Clean HF call-set run with counter bug:
  `<validation-workspace>/runs/moe35b_hf_tp4_callset_clean_eager4k_dot20_20260628T0720Z`
- Counter-fixed HF patch:
  `<validation-workspace>/experimental-patches/qwen35moe-hf-callset-counterfix-20260628T0815Z`
- Counter-fixed tuned HF run:
  `<validation-workspace>/runs/moe35b_hf_tp4_callset_counterfix_eager4k_tuned_dot20_20260628T0825Z`
- HF/GGUF compact comparison:
  `<validation-workspace>/runs/moe35b_hf_tp4_callset_counterfix_eager4k_tuned_dot20_20260628T0825Z/callset_hf_gguf_compare.md`

### Setup

This was a `.20` profiling follow-up to `GGUF-238`, not a benchmark ladder.
The goal was to determine whether the large recurrent GDN/SSM state norms seen
in the GGUF-only call-set trace were GGUF-specific or also present in an HF
control.

The first HF attempt reused a GGUF-derived `qwen3_5.py` and was rejected: it
returned non-coherent text and wrote no usable trace. The next clean-HF patch
used the HF source base and returned coherent text, but it only captured calls
`0`, `1`, and `2` because the decoder trace counter advanced only inside the
trace-active block.

The promoted diagnostic patch moved the decoder trace counter increment
outside the trace-active block, matching the working GGUF patch. The final
counter-fixed run used:

- published ROCm7.2 Dense/MoE runtime image;
- HF Qwen3.6 35B-A3B model cache;
- TP4 on `.20`;
- P2P-on (`NCCL_P2P_DISABLE=0`);
- FP16;
- tuned MoE config mounted;
- `MAX_MODEL_LEN=4096`;
- `--enforce-eager`;
- 512-token capped benchmark-style prompt.

The release patch bundle was intentionally not mounted in the final comparator
because it tries to replace `qwen3_5.py`, which conflicts with the trace source
mounted at the same path. This makes the run a state-comparison diagnostic,
not a performance comparator.

### Outcome

The counter-fixed HF request completed:

- `finish_reason=length`
- prompt tokens: `143`
- completion tokens: `512`
- content characters: `2371`
- reasoning characters: `0`
- content head: `Here's a thinking process:`

Trace coverage reached all requested call labels:

- `304` rows for call `0`
- `304` rows for call `1`
- `616` rows for call `2`
- `616` rows for call `64`
- `616` rows for call `128`
- `616` rows for call `256`
- `616` rows for call `511`

Selected SSM max-norm comparison:

| layer | call | HF max norm | GGUF max norm | GGUF/HF |
| --- | ---: | ---: | ---: | ---: |
| layer 0 | `2` | `929.961182` | `929.961365` | `1.000` |
| layer 0 | `64` | `612587.875000` | `613222.000000` | `1.001` |
| layer 0 | `128` | `701444.937500` | `701755.062500` | `1.000` |
| layer 0 | `256` | `974358.562500` | `922935.500000` | `0.947` |
| layer 0 | `511` | `1347038.000000` | `1130861.125000` | `0.840` |
| layer 10 | `511` | `10110661.000000` | `10560669.000000` | `1.045` |
| layer 20 | `511` | `28720954.000000` | `34658824.000000` | `1.207` |

The important result is that recurrent SSM state growth is not GGUF-only. HF
shows the same shape and the same order of magnitude; layer 0 is essentially
identical through call `128`, layer 10 stays within about five percent at late
decode, and layer 20 GGUF is about twenty percent higher late.

Selected downstream max-norm comparison:

| signal | layer | call | HF max norm | GGUF max norm | GGUF/HF |
| --- | --- | ---: | ---: | ---: | ---: |
| post-attention layernorm | layer 0 | `2` | `326.962555` | `326.962769` | `1.000` |
| post-attention layernorm | layer 0 | `511` | `28.997663` | `20.992693` | `0.724` |
| post-attention layernorm | layer 10 | `511` | `35.674820` | `27.124575` | `0.760` |
| post-attention layernorm | layer 20 | `511` | `47.194199` | `35.481808` | `0.752` |
| post-MLP | layer 39 | `511` | `6.414737` | `11.129422` | `1.735` |
| out-projection input | layer 10 | `511` | `0.624816` | `0.387380` | `0.620` |

The early downstream signals match almost exactly, which is the same pattern
seen in the earlier branch and boundary traces. By late decode, GGUF tends to
show lower post-attention layernorm norms and lower layer-10 out-projection
input, while recurrent SSM state growth itself is not a unique GGUF failure.

### Promote / Reject

Promote:

- the counter-fixed clean-HF call-set patch as the working comparator shape for
  short bounded state traces;
- early HF/GGUF state parity as evidence that the remaining gap is not a gross
  loader or initial hidden-state mismatch;
- late downstream attenuation, replay scheduling, kernel behavior, and
  graph-consumed value/state semantics as the next source slice.

Reject:

- the GGUF-derived HF model file as a comparator base;
- the clean-HF counter-bug run as a late-window comparator;
- a broad "GGUF recurrent state explosion" theory;
- a GDN/SSM rescale patch based only on the large recurrent state norms;
- using eager call-set traces as benchmark or throughput evidence.

### Reason

The profiling now separates two facts. First, the recurrent state norms grow
very large on both HF and GGUF, so that growth alone is not the MoE GGUF
throughput root cause. Second, late normalized downstream signals diverge while
the replay timing gap widens across long decode. That points below static
loader and graph-shape parity but above a single obvious Python tensor layout
bug: the next useful work is graph-visible replay behavior, kernel/runtime
scheduling, or how late GDN state and downstream activations are consumed in
the replayed graph.

### Next

Do not run another full warmup -> `c1_128` strict -> `c1_2000` ->
`c1_10000` ladder until a candidate changes graph replay timing or late
downstream state. The next source pass should inspect lower-level replay
inputs or generated kernel/runtime behavior around:

- GDN state/value consumption after graph capture;
- late post-attention layernorm and out-projection input attenuation;
- replay scheduler cadence and per-window kernel timing;
- memory movement or value materialization outside the already-matching
  generated Python graph bodies.

## GGUF-240 - MoE Replay Value-Signature Group Mining

Run artifacts:

- Existing HF/GGUF replay value-signature comparison:
  `<validation-workspace>/runs/moe35b_valuesig_replay_compare_20260628T0450Z`
- New grouped value-signature summary:
  `<validation-workspace>/runs/moe35b_valuesig_replay_compare_20260628T0450Z/value_signature_grouped_20260628.tsv`
- New line-diff count summary:
  `<validation-workspace>/runs/moe35b_valuesig_replay_compare_20260628T0450Z/value_signature_diff_counts_20260628.tsv`
- Generated-source map for the highest-delta replay family:
  `<validation-workspace>/runs/moe35b_valuesig_replay_compare_20260628T0450Z/argc11_tensor8_generated_source_map_20260628.txt`

### Setup

This was a read-only mining pass over the existing `.20` replay value-signature
artifacts from `GGUF-234`. No server was launched, no benchmark ladder was run,
and no model files or source files were changed.

The pass parsed normalized HF and GGUF replay graph-input records with POSIX
shell and `awk`, paired records by replay order, and grouped tensor arguments
by mode, token count, argument count, tensor-argument count, argument index,
shape, and stride. The purpose was to separate repeated value families from
raw diff noise before choosing another source experiment.

### Outcome

Replay structure stayed aligned:

- HF and GGUF still have identical replay headers.
- The grouped pass found no missing graph-input family.
- Several static/vector arguments are exactly equal across all paired records,
  including repeated `2048`-wide vector arguments and the `argc=6/tensor=4`
  replay family.

The strongest repeated value delta is not a static weight-shaped tensor. It is
an activation-shaped graph argument:

| key | count | diff count | average HF abs sum | average GGUF abs sum | average abs delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `argc=11 tensor=8 arg=4 shape=(16, 2048)` | `36` | `36` | `349.976942` | `92.737499` | `257.266042` |
| `argc=12 tensor=9 arg=0 shape=(16, 8, 128)` | `80` | `79` | `10.236491` | `10.295423` | `4.115438` |
| `argc=17 tensor=13 arg=0 shape=(16, 8, 128)` | `40` | `40` | `10.513276` | `10.640548` | `2.817278` |
| `argc=11 tensor=8 arg=0 shape=(16, 4, 256)` | `36` | `36` | `2.454427` | `2.507891` | `1.724874` |
| `argc=11 tensor=8 arg=2 shape=(16, 1024)` | `36` | `36` | `20.671586` | `20.699270` | `1.596972` |

Representative weight-shaped arguments such as `(2048, 1024)`,
`(3088, 2048)`, and `(2560, 2048)` differ by sampled values, but their average
sampled absolute sums are similar between HF and GGUF. That makes them poor
direct patch targets without a stronger mapping to a named generated region.

The generated-source map changes the interpretation of the highest-delta
`arg=4` row. In the matching generated `argc=11/tensor=8` fragment, this family
is the compiled MoE expert fragment around `moe_forward_shared`, an all-reduce,
and `rocm_unquantized_gemm_1` for `language_model.model.layers.3.mlp.experts`.
The `arg4_1` tensor is passed as an output/mutation pointer to the final
`triton_red_fused__to_copy_add_copy__mean_mul_pow_rocm_unquantized_gemm_rsqrt_3`
kernel, not as the semantic activation consumed by the graph. The value
signature therefore sampled stale mutable output-buffer contents before replay
overwrote them.

### Promote / Reject

Promote:

- the existing replay value-signature logs as useful for grouping repeated
  activation/value families;
- source mapping of the `argc=11/tensor=8` family to a generated MoE
  shared/routed-expert fragment;
- the output-buffer finding as a guardrail against chasing stale mutable graph
  inputs as semantic activations;
- consumed inputs around the same fragment, especially `arg0`, `arg2`, `arg6`,
  and the downstream returned buffers, as the next inspection boundary.

Reject:

- a new full warmup -> strict -> `c1_2000` -> `c1_10000` ladder before a
  source candidate changes replay timing or this activation family;
- treating the high-delta `arg4` output buffer as a patch target by itself;
- treating raw `(2048, 1024)` / `(3088, 2048)` sampled weight differences as a
  patch target by themselves;
- returning to generated Python body diffs, replay-header diffs, or broad
  graph-input shape diffs as the primary MoE GGUF explanation.

### Reason

The profiling picture is now more constrained. The remaining MoE GGUF gap is
not missing graph structure, not generated Python source selection, not a
single-rank issue, not a broad best-config choice, and not a gross recurrent
state explosion unique to GGUF. The generated-source map shows that the
largest sampled delta is an output buffer for a MoE expert fragment rather than
a consumed activation. That makes the next useful source slice narrower:
instrument consumed inputs and returned buffers around this fragment and check
whether they connect to the late post-attention layernorm / out-projection
input attenuation already seen in the call-set comparator.

### Next

Do not launch another benchmark ladder yet. The next bounded diagnostic should
trace consumed inputs and returned buffers around the mapped
`argc=11/tensor=8` MoE expert fragment in a short request window. Only run the
normal benchmark ladder after that source slice either changes replay timing
toward the HF band or restores a closer late downstream trajectory.

## GGUF-241 - MoE Replay Pre/Post Value Signature Probe

### Setup

- Date: 2026-06-28 UTC.
- Host: `.20`.
- Run root:
  `<validation-workspace>/runs/moe35b_gguf_tp4_prepost_valuesig_dot20_20260628T064605Z`.
- Container:
  `vllm_qwen36_moe35b_gguf_tp4_prepost_20260628T064605Z`.
- Model: `Qwen3.6-35B-A3B-F16-GGUF-fusedgdn-base-rmsnorm`.
- Profile: GGUF MoE TP4, P2P-on, FP16, `MAX_MODEL_LEN=131072`.
- Patch: disposable cuda-graph logger overlay copied from the existing
  value-signature patch and extended to log `replay_post` immediately after
  `entry.cudagraph.replay()`.

### Question

The previous value-signature mining pass showed the largest HF/GGUF sampled
delta on output-shaped MoE graph arguments, especially
`argc=11/tensor=8 arg=4`. This run asked whether those values were semantic
inputs consumed by replay or stale output/mutation buffers overwritten during
replay.

### Procedure

The server was allowed to finish graph capture. A short direct vLLM chat
request was sent:

- prompt: `Write one concise sentence about local AI reproducibility.`
- `max_tokens=64`
- `temperature=0.0`

The response was coherent in the reasoning stream and stopped by the local
length cap, so the run was accepted as a valid short diagnostic. The signature
file grew from `5212` lines before the request to `11932` lines after the
request.

### Observations

Replay logging produced balanced pre/post records:

| phase | record count |
| --- | ---: |
| `capture` | `384` |
| `replay` | `384` |
| `replay_post` | `384` |

The short request split into:

| replay shape | count |
| --- | ---: |
| `FULL num_tokens=1 argc=0` | `220` replay + `220` post |
| `PIECEWISE num_tokens=24 argc=12` | `80` replay + `80` post |
| `PIECEWISE num_tokens=24 argc=17` | `40` replay + `40` post |
| `PIECEWISE num_tokens=24 argc=11` | `36` replay + `36` post |
| `PIECEWISE num_tokens=24 argc=9` | `4` replay + `4` post |
| `PIECEWISE num_tokens=24 argc=6` | `4` replay + `4` post |

The paired pre/post scan found `160` changed tensor records, concentrated in
the expected output/mutation slots:

| changed slot | changed pairs |
| --- | ---: |
| `argc=12 arg[5]` | `80` |
| `argc=17 arg[5]` | `40` |
| `argc=11 arg[4]` | `36` |
| `argc=9 arg[4]` | `4` |

The matching metadata identifies those changed slots as large
`(tokens, 2048)` activation/output buffers. Stable consumed inputs in the same
blocks did not show this pre/post mutation pattern. This confirms that the
large deltas seen in GGUF-240 are primarily stale output-buffer samples before
replay overwrites them, not direct semantic input mismatches.

The same run also made the TopK8 fastpath boundary clearer. The installed
overlay's `_try_qwen_c1_topk8_fastpath` requires `hidden_states.size(0) == 1`,
so it is a decode-single-token helper. During graph capture and prefill it
logged `204` grouped `shape_or_layout` rejections across multi-token shapes
and `4` activations at `num_tokens=1`. Those grouped rejections are expected
from the predicate and are not by themselves the GGUF/HF performance gap.

### Promote / Reject

Promote:

- the pre/post value-signature probe as confirmation that the high-delta MoE
  graph arguments are mutation buffers;
- the previous source-map interpretation of `argc=11 arg[4]` as an
  output/mutation pointer;
- the replay timing evidence as the stronger performance signal: GGUF FULL
  decode replay is still materially slower than the HF control;
- the next source slice around decode replay cost, generated kernel runtime
  behavior, and GGUF expert-weight layout/materialization in the single-token
  path.

Reject:

- treating `argc=11 arg[4]`, `argc=12 arg[5]`, `argc=17 arg[5]`, or
  `argc=9 arg[4]` pre/post changes as semantic correctness bugs;
- another benchmark ladder before a candidate changes the replay timing band;
- chasing grouped TopK8 multi-token rejections as a standalone fix, because
  the current helper is intentionally one-token-only.

### Reason

The profiling picture now says the graph/value surface is mostly exonerated.
Generated graph structure, source bodies, replay headers, recurrent state
growth, and the largest sampled activation deltas have all failed to produce a
direct GGUF-only patch target. The remaining actionable gap is runtime
performance in the decode replay lane. The next source experiment should
inspect why the GGUF single-token decode path still replays slower than HF
despite matching generated graph bodies, with special attention to expert
weight layout/materialization and generated kernel runtime behavior.

### Next

Keep the benchmark ladder parked. The next source probe should be a
single-token decode-focused HF/GGUF comparator that records the active MoE
kernel path, expert-weight strides/layout, generated kernel identities, and
per-call replay timing for the single-token `FULL` decode graph.

## GGUF-242 - MoE Active Layout Decode Comparator

### Setup

- Date: 2026-06-28 UTC.
- Host: `.20`.
- HF run root:
  `<validation-workspace>/runs/moe35b_hf_tp4_decode_active_layout_dot20_20260628T070537Z`.
- GGUF run root:
  `<validation-workspace>/runs/moe35b_gguf_tp4_decode_active_layout_dot20_20260628T070537Z`.
- Compare summary:
  `<validation-workspace>/runs/moe35b_decode_active_layout_compare_20260628T070537Z.txt`.
- Debug overlay:
  `<validation-workspace>/experimental-patches/fused-moe-active-layout-debug-20260628T070403Z`.
- Model pair:
  - HF: `Qwen/Qwen3.6-35B-A3B`
  - GGUF: `Qwen3.6-35B-A3B-F16-GGUF-fusedgdn-base-rmsnorm`
- Profile: MoE TP4, P2P-on, FP16, `MAX_MODEL_LEN=131072`.

### Question

After the pre/post value-signature probe ruled out stale replay output buffers
as semantic inputs, the next question was whether GGUF enters a different
single-token MoE fastpath layout from HF. The probe added one-time detail
logging at the release TopK8 helper's `num_tokens=1` active branch and sampled
every `FULL` decode replay call.

### Procedure

The HF and GGUF servers were launched sequentially on `.20`, not in parallel.
Both used the same short request:

- prompt: `Write one concise sentence about local AI reproducibility.`
- `max_tokens=256`
- `temperature=0.0`

Both returned coherent reasoning-stream output. The containers were stopped
after each request.

### Observations

Both HF and GGUF produced `4` active-detail records, one for each TP worker.
Both also produced `204` grouped `shape_or_layout` rejections during graph
capture/prefill. The active single-token layout was identical:

```text
hidden_shape=(1, 2048) hidden_stride=(2048, 1)
w1_shape=(256, 256, 2048) w1_stride=(557056, 2176, 1)
w2_shape=(256, 2048, 128) w2_stride=(262144, 128, 1)
topk_ids_shape=(1, 8) topk_weights_shape=(1, 8)
```

The `FULL num_tokens=1` replay timing still diverged:

| path | count | average ms | min ms | max ms | early avg ms | late avg ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HF | `1020` | `7.776122` | `7.398875` | `9.576159` | `7.875851` | `7.761949` |
| GGUF | `1020` | `11.212760` | `10.743509` | `14.567984` | `11.377638` | `11.218778` |

### Promote / Reject

Promote:

- active-layout equivalence as a real finding: the visible TopK8 MoE
  single-token path, expert weight shape, and expert weight stride match HF;
- the replay timing gap as the primary source signal;
- deeper inspection of GGUF runtime/materialization or generated kernel runtime
  behavior below the Python-visible layout boundary.

Reject:

- a simple expert tensor stride/layout mismatch as the MoE GGUF performance
  root cause;
- grouped TopK8 prefill rejections as a GGUF-only issue;
- another full benchmark ladder before a candidate changes the single-token
  replay timing band.

### Reason

This probe removes the most direct layout hypothesis. GGUF activates the same
one-token fastpath layout that HF uses, but its `FULL` decode replay remains
about `44%` slower on the same prompt and profile. The next source slice should
look below visible tensor metadata: GGUF weight materialization, generated
kernel runtime behavior, memory movement, or other runtime state not exposed by
the identical shape/stride logs.

## GGUF-243 - MoE GGUF RCCL-Only Native Overlay Probe

### Setup

- Date: 2026-06-28 UTC.
- Host: `.20`.
- Failed full-native scratch run:
  `<validation-workspace>/runs/moe35b_gguf_tp4_decode_layout_rcclnative_dot20_20260628T072507Z`.
- Accepted RCCL-only run:
  `<validation-workspace>/runs/moe35b_gguf_tp4_decode_layout_rcclonly_dot20_20260628T072640Z`.
- Composite patch bundle:
  `<validation-workspace>/experimental-patches/qwen35moe-gguf-plus-rcclonly-20260628T072640Z`.
- Profile: GGUF MoE TP4, P2P-on, FP16, `MAX_MODEL_LEN=131072`.

### Question

The HF release-patch control sets
`VLLM_NCCL_SO_PATH=/rccl-overlay/install/lib/librccl.so.1`, while the current
GGUF timing script did not. This probe asked whether the missing release RCCL
overlay explains the GGUF replay gap.

### Procedure

A first scratch attempt copied the full release native bundle into the GGUF
patch bundle. That was rejected before serving because the GGUF launch mounts
`<validation-nvme-root>` read-only and the native swiglu copy path tried to write
under that tree. The accepted run narrowed the patch bundle to release
`native/rccl` only and added the same `VLLM_NCCL_SO_PATH` environment variable.

The same short request was used:

- prompt: `Write one concise sentence about local AI reproducibility.`
- `max_tokens=256`
- `temperature=0.0`

The response was coherent, and the container was stopped after the request.

### Observations

The RCCL-only run stayed on the intended GGUF path:

- forced unquantized FusedMoE method was active;
- the TopK8 one-token fastpath activated on all TP workers;
- visible active layout still matched HF:

```text
hidden_shape=(1, 2048) hidden_stride=(2048, 1)
w1_shape=(256, 256, 2048) w1_stride=(557056, 2176, 1)
w2_shape=(256, 2048, 128) w2_stride=(262144, 128, 1)
```

The `FULL num_tokens=1` replay timing improved slightly but did not close the
gap:

| path | count | average ms | min ms | max ms | early avg ms | late avg ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GGUF baseline active-layout probe | `1020` | `11.212760` | `10.743509` | `14.567984` | `11.377638` | `11.218778` |
| GGUF + RCCL-only native overlay | `1020` | `10.864359` | `10.379656` | `14.610685` | `10.989412` | `10.932782` |
| HF active-layout control | `1020` | `7.776122` | `7.398875` | `9.576159` | `7.875851` | `7.761949` |

### Promote / Reject

Promote:

- carrying the release RCCL overlay into GGUF repro hygiene, because it trims
  roughly `0.35` ms from the short active-layout probe;
- keeping this as a launch-environment parity requirement for future GGUF MoE
  probes.

Reject:

- missing `VLLM_NCCL_SO_PATH` as the primary MoE GGUF performance root cause;
- copying the full native release bundle into the read-only GGUF launch
  unchanged;
- another full benchmark ladder from this small timing improvement alone.

### Reason

RCCL parity helps but is not enough. The GGUF replay path remains about
`40%` slower than the HF control after the RCCL-only fix. The next useful
source experiment should compare the GGUF and HF generated kernel/runtime
surface below the matching active MoE layout: materialized expert storage,
kernel identities, memory movement, or replay-time state that is not visible in
Python tensor shape/stride metadata.

## GGUF-244 - MoE GGUF Force-Unquantized Linear Probe

### Setup

- Date: 2026-06-28 UTC.
- Host: `.20`.
- Run root:
  `<validation-workspace>/runs/moe35b_gguf_tp4_force_unquant_linear_dot20_20260628T073755Z`.
- Patch bundle:
  `<validation-workspace>/experimental-patches/qwen35moe-gguf-force-unquant-linear-20260628T073755Z`.
- Base: RCCL-only GGUF patch bundle from GGUF-243.
- Added env: `VLLM_GFX906_GGUF_FORCE_UNQUANT_LINEAR=1`.
- Profile: GGUF MoE TP4, P2P-on, FP16, `MAX_MODEL_LEN=131072`.

### Question

The GGUF loader should add F16 GGUF `.weight` tensors to
`unquantized_modules`, but the remaining replay gap could still have come from
some non-MoE linears running through `GGUFLinearMethod`. This probe forced
every GGUF `LinearBase` and `VocabParallelEmbedding` module to return
unquantized methods before weight loading.

### Procedure

The patch inserted opt-in method selection in
`vllm/model_executor/layers/quantization/gguf.py`:

- `LinearBase` -> `UnquantizedLinearMethod()`
- `VocabParallelEmbedding` -> `UnquantizedEmbeddingMethod()`

The same short request was used:

- prompt: `Write one concise sentence about local AI reproducibility.`
- `max_tokens=256`
- `temperature=0.0`

### Observations

The patch was active and loaded successfully:

- the container logged `760` forced-linear method selections;
- examples included attention projections, linear-attention output
  projections, and shared-expert linears;
- the response was coherent;
- TopK8 one-token MoE fastpath layout remained unchanged.

The `FULL num_tokens=1` replay timing did not improve beyond the RCCL-only
probe:

| path | count | average ms | min ms | max ms | early avg ms | late avg ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GGUF + RCCL-only native overlay | `1020` | `10.864359` | `10.379656` | `14.610685` | `10.989412` | `10.932782` |
| GGUF + RCCL-only + force-unquantized linears | `1020` | `10.871309` | `10.401414` | `15.010851` | `10.971306` | `10.978598` |
| HF active-layout control | `1020` | `7.776122` | `7.398875` | `9.576159` | `7.875851` | `7.761949` |

### Promote / Reject

Promote:

- the forced-linear probe as evidence that the patch can switch regular GGUF
  linears and embeddings onto unquantized methods without breaking this short
  decode request;
- the conclusion that the remaining gap is below this GGUF linear method
  boundary.

Reject:

- `GGUFLinearMethod` dispatch for F16 regular linears as the primary remaining
  replay-time gap;
- another full benchmark ladder from this patch;
- a broad "force unquantized all linears" release candidate unless a later
  source change also moves replay timing.

### Reason

The patch affected hundreds of regular linears but left the settled
single-token replay band unchanged relative to RCCL-only. This narrows the
remaining work toward lower-level generated code/runtime differences that
survive after MoE layout, RCCL, and regular GGUF linear dispatch are aligned.

## GGUF-245 - MoE GGUF Generated-Kernel and Replay Pointer Probe

### Setup

- Date: 2026-06-28 UTC.
- Host: `.20`.
- GGUF run:
  `<validation-workspace>/runs/moe35b_gguf_tp4_ptrsig_dot20_20260628T080540Z`.
- HF run:
  `<validation-workspace>/runs/moe35b_hf_tp4_ptrsig_dot20_20260628T081443Z`.
- Pointer diagnostic patch:
  `<validation-workspace>/experimental-patches/cudagraph-replay-ptr-signature-20260628T080517Z`.
- GGUF base patch:
  `<validation-workspace>/experimental-patches/qwen35moe-gguf-plus-rcclonly-20260628T072640Z`.
- Profile: MoE TP4, P2P-on, FP16, `MAX_MODEL_LEN=131072`.

### Question

After GGUF-242 through GGUF-244, the visible MoE layout, RCCL path, and
regular GGUF linear method were no longer good explanations for the MoE GGUF
gap. This probe asked whether the gap was visible in:

- kernel-level profiler output;
- generated TorchInductor source or generated-kernel timing;
- top-level CUDA graph replay argument pointer alignment.

### Procedure

First, `rocprofv3 --attach` was tried against `Worker_TP0` inside the running
GGUF TP4 container. The command attached successfully, but no trace files or
summary files were emitted, even after explicit `csv`/`json`, runtime trace,
kernel trace, summary, and zero minimum-output settings. That path was rejected
as a useful probe for this server process.

Next, existing TorchInductor `.kernel_perf` files were paired between matched
HF and GGUF runs. The current GGUF+RCCL run and the HF active-layout control
both had `76` generated-kernel perf records and `216` generated Python files.
The generated kernel IDs paired one-for-one.

The top GGUF-slower generated-kernel perf entries were:

| kernel id | HF timing | GGUF timing | ratio | delta |
| --- | ---: | ---: | ---: | ---: |
| `v6/cv62j5...py` | `0.014560000` | `0.051199999` | `3.516` | `0.036639999` |
| `qx/cqxlvi...py` | `0.011200000` | `0.046239998` | `4.129` | `0.035039999` |
| `sq/csqysk...py` | `0.022399999` | `0.053280000` | `2.379` | `0.030880000` |
| `b6/cb6zn2...py` | `0.036479998` | `0.058400001` | `1.601` | `0.021920003` |
| `ws/cwsrym...py` | `0.014400000` | `0.028640000` | `1.989` | `0.014240000` |
| `dl/cdlbqg...py` | `0.022080000` | `0.035680000` | `1.616` | `0.013599999` |

The corresponding generated Python sources were byte-identical for these top
delta kernels. The aggregate paired `.kernel_perf` first-line timings were not
worse for GGUF:

| path | paired count | timing sum |
| --- | ---: | ---: |
| HF | `76` | `1.862880006` |
| GGUF | `76` | `1.624796998` |

Finally, a pointer-signature diagnostic was created by extending the existing
graph-input value-signature patch to log `data_ptr`, `ptr_mod256`,
`ptr_mod4096`, and `storage_offset`. Matched HF and GGUF short decode requests
were run after graph capture.

### Observations

The top-level FULL replay inputs were the same visible shape/alignment pattern
for HF and GGUF:

- `input_ids`: shape `(1,)`, contiguous, `ptr_mod256=0`,
  `ptr_mod4096=3584` on all four TP workers.
- `positions`: shape `(3, 1)`, stride `(2049, 1)`, non-contiguous,
  `ptr_mod256=0`, `ptr_mod4096=512` on all four TP workers.

The pointer diagnostic confirmed the replay gap while rejecting those top-level
arguments as the cause:

| path | FULL `num_tokens=1` count | average ms | min ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| HF pointer diagnostic | `64` | `8.003061` | `7.659992` | `11.062552` |
| GGUF pointer diagnostic | `508` | `10.827265` | `10.412477` | `20.420797` |

The GGUF short response remained coherent. The HF first request hit the startup
edge and timed out; a second short request after the server was fully ready
completed normally.

### Promote / Reject

Promote:

- the generated-kernel ID pairing as evidence that HF and GGUF use the same
  generated kernel surface for this TP4 decode path;
- the byte-identical top-delta generated sources as evidence that the remaining
  gap is not caused by Python source generation differences in those kernels;
- the pointer-signature diagnostic as evidence that top-level FULL replay
  `input_ids` and `positions` pointer alignment do not explain the gap.

Reject:

- `rocprofv3 --attach` as a useful profiling path for these long-running vLLM
  server workers in this setup;
- generated-kernel source mismatch as the current root cause;
- top-level FULL replay argument pointer alignment as the current root cause;
- another full benchmark ladder until a candidate changes settled replay
  timing or a lower-level internal buffer/state difference is found.

### Reason

The gap survives after matching visible MoE layout, RCCL launch state, regular
linear method dispatch, generated-kernel IDs, generated-kernel source bodies,
and top-level CUDA graph replay input alignment. The next source slice should
look inside the captured graph body and model internals rather than at the
outer replay call: internal activation/KV/recurrent buffers, graph state,
weight materialization/residency, or runtime scheduling below the
`CUDAGraphWrapper` argument boundary.

## GGUF-246 - MoE Active Fastpath Value-Signature Comparator

### Setup

- Date: 2026-06-28 UTC.
- Host: `.20`.
- Diagnostic patch:
  `<validation-workspace>/experimental-patches/fused-moe-active-value-debug-20260628T083222Z`.
- GGUF run:
  `<validation-workspace>/runs/moe35b_gguf_tp4_active_value_dot20_20260628T083258Z`.
- HF run:
  `<validation-workspace>/runs/moe35b_hf_tp4_active_value_dot20_20260628T083258Z`.
- GGUF patch base:
  `<validation-workspace>/experimental-patches/qwen35moe-gguf-plus-rcclonly-20260628T072640Z`.
- HF patch base:
  `<validation-workspace>/experimental-patches/release-patch-bundle-active-value-debug-20260628T083258Z`.
- Profile: MoE TP4, P2P-on, FP16, `MAX_MODEL_LEN=131072`.

### Question

GGUF-245 rejected generated-kernel source mismatch and top-level replay input
pointer alignment as the current root. This probe asked whether HF and GGUF
enter the one-token TopK8 MoE fastpath with different active experts, routing
weights, hidden-state values, or expert-weight slice signatures.

### Procedure

The release TopK8 one-token fastpath was extended to write a small active-state
record when `VLLM_GFX906_MOE_ACTIVE_VALUE_FILE` is set. Each record contains
the TP worker PID, per-worker call index, pointer alignment, the eight active
expert IDs, the eight routing weights, the first eight hidden-state values,
and small selected `w1`/`w2` expert-slice value signatures.

The first implementation copied CUDA tensors to CPU from inside the active
branch. During graph capture, PyTorch/ROCm reported:

`Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned.`

Despite that warning during capture, both HF and GGUF emitted `320` active
signature lines, or `80` lines per TP worker. The GGUF endpoint completed a
coherent short completion request. The HF endpoint reached readiness, but the
first completion request under this diagnostic hung with a shared-memory wait
and no GPU progress; that HF request is rejected as a diagnostic-method
failure, not as a model-performance result.

### Observations

The first active fastpath call matched between HF and GGUF:

- TopK IDs: `101,196,108,249,216,197,135,242`.
- TopK weights:
  `0.311032623,0.21544601,0.126901776,0.0917616785,0.082980752,0.0682581142,0.0558740981,0.0477450006`.
- Hidden-state signature:
  `0.11554,-0.0568237,0.70166,-1.2793,0.270996,0.732422,-1.56738,-0.678223`.
- Visible tensor layout remained the same as GGUF-242:
  hidden `(1, 2048)` stride `(2048, 1)`, `w1`
  `(256, 256, 2048)` stride `(557056, 2176, 1)`, and `w2`
  `(256, 2048, 128)` stride `(262144, 128, 1)`.

Calls `1` through roughly `8` stayed close, with small FP16-level drift in
routing weights and hidden signatures. The first clear routing divergence
appeared around calls `9` and `10`; for example, call `9` differed in the last
selected expert and call `10` changed both ordering and selected experts.

The GGUF short request remained coherent and produced this replay timing under
the diagnostic:

| path | FULL `num_tokens=1` count | average ms | min ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| GGUF active-value diagnostic | `508` | `10.893246` | `10.468469` | `14.597109` |

No valid HF replay timing was accepted from this diagnostic because the HF
request hung after readiness.

### Promote / Reject

Promote:

- the active-value comparator as evidence that the first one-token MoE
  fastpath call has matching expert IDs, routing weights, hidden signature,
  visible tensor layout, and selected expert-weight signatures between HF and
  GGUF;
- the later call-index divergence as evidence that small upstream state/value
  drift accumulates before or around the active MoE fastpath;
- the GGUF coherent response and `10.893246` ms replay timing as another
  confirmation that the GGUF performance band remains in the same slower range
  after adding this diagnostic.

Reject:

- visible first-call expert routing, tensor layout, or selected expert-weight
  materialization as the current primary root;
- the CPU-copy active-value logger as a safe timing probe for HF request
  execution;
- the failed HF completion request in this diagnostic as evidence of an HF
  model or release-profile problem.

### Reason

The first active MoE decode call starts from a matched HF/GGUF state, so the
remaining MoE GGUF gap is not explained by an immediate wrong expert layout,
wrong first routing vector, or wrong selected expert slice. The later routing
divergence points back upstream into captured model state or earlier operations
that feed the router. The next useful source slice should avoid CPU copies
inside CUDA graph capture and should inspect upstream consumed inputs or
runtime scheduling without perturbing HF replay.

## GGUF-247 - HF Eager Active-Value Control Rejection

### Setup

- Date: 2026-06-28 UTC.
- Host: `.20`.
- HF eager run:
  `<validation-workspace>/runs/moe35b_hf_tp4_active_value_eager_dot20_20260628T085833Z`.
- Matching GGUF eager run:
  `<validation-workspace>/runs/moe35b_gguf_tp4_active_value_eager_dot20_20260628T085833Z`.
- Diagnostic patch:
  `<validation-workspace>/experimental-patches/fused-moe-active-value-debug-20260628T083222Z`.
- Profile: MoE TP4, P2P-on, FP16, `MAX_MODEL_LEN=131072`.

### Question

GGUF-246 found matching first-call active MoE state but later routing
divergence. This follow-up asked whether disabling CUDA graph capture with
`--enforce-eager` would let the same active-value CPU-copy logger collect a
request-time HF control without the CUDA graph capture warning.

### Procedure

The HF launch was adjusted only in the generated run directory so
`--enforce-eager` was passed through `EXTRA_VLLM_ARGS` instead of being parsed
as a Docker flag. The server then loaded the HF model with the release TP4
profile, RCCL path, P2P-on state, FP16 dtype, and eager execution enabled.

After readiness, the active-value signature file and completion output were
cleared. The same bounded diagnostic prompt used for the GGUF eager run was
sent:

`Write one concise sentence about local AI reproducibility.`

with `max_tokens=64` and `temperature=0.0`.

### Observations

The HF eager server reached readiness and confirmed:

- `Enforce eager set, disabling torch.compile and CUDAGraphs`;
- `Cudagraph is disabled under eager mode`;
- the release RCCL library was loaded through `VLLM_NCCL_SO_PATH`;
- the four TP workers allocated model memory on the expected GPUs.

The bounded request did not complete. GPU use remained at `0%`, the completion
file stayed empty, and no `moe_active_value_signature.tsv` records were
written. The server emitted repeated shared-memory wait messages after
readiness. Stopping the diagnostic container cleared VRAM cleanly.

The matching GGUF eager run had already completed a coherent bounded request
and emitted `512` active-value records, but without a successful HF request log
there is no valid eager HF/GGUF call-index comparison from this method.

### Promote / Reject

Promote:

- the launch-script quoting correction as run-directory hygiene for future
  eager diagnostics;
- the observation that HF eager plus the active-value CPU-copy logger is not a
  usable comparator in this form.

Reject:

- HF eager active-value CPU-copy logging as the next diagnostic path;
- using this HF eager request hang as evidence about HF model quality,
  release-profile quality, or normal graph-captured HF throughput;
- repeating the same CPU-copy logger in HF request execution without changing
  the instrumentation method.

### Reason

The eager switch removed CUDA graph capture from the request path, but the HF
request still hung with no GPU progress and no active-value records. The
diagnostic therefore failed before it could compare the call where GGUF-246 saw
later routing divergence. The next source slice should use lower-overhead
instrumentation: device-side summaries, pinned-buffer snapshots, log-only
metadata that avoids tensor CPU copies in the hot path, or a narrower upstream
router-input comparator that does not perturb HF request execution.

## GGUF-248 - MoE ROCProfiler Launch-Wrapper Shutdown Rejection

### Setup

- Date: 2026-06-28 UTC.
- Host: `.20`.
- Run:
  `<validation-workspace>/runs/moe35b_gguf_tp4_rocprof_kernel_short_dot20_20260628T092554Z`.
- Image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Profile: MoE TP4, P2P-on, FP16, GGUF release-shape path.
- Profiler route: launch-time `/opt/rocm/bin/rocprofv3` wrapper with a
  read-only host `libdw.so.1` mount and a mounted `/rocprof-out` output
  directory.
- Request:
  `Write one concise sentence about local AI reproducibility.`
  with `max_tokens=64` and `temperature=0.0`.

### Question

GGUF-245 and earlier dense probes rejected `rocprofv3 --attach` and several
broad server-wrapper shapes. This probe asked whether launch-wrapping the MoE
GGUF server from process start, while supplying the missing `libdw.so.1`, could
produce kernel-level profiler artifacts for a short coherent request.

### Procedure

The active GGUF TP4 launch script was wrapped with `rocprofv3`, mounted a
dedicated profiler output directory, and preserved the release image, FP16,
P2P-on, TP4 profile, and GGUF source bundle. After readiness, the short
deterministic prompt above was sent through the local completions API. The
container was then stopped through Docker.

### Observations

The server reached readiness and served the short completion request. The
response was coherent Qwen-style thinking text, with:

- prompt tokens: `10`;
- completion tokens: `64`;
- total tokens: `74`;
- `finish_reason`: `length`.

The profiler wrapper did not emit kernel, HIP, RCCL, CSV, JSON, or summary
files under the mounted profiler output directory. The log shows
`rocprofv3 caught signal 15` and waited for child processes after Docker stop.
The container exited with status `137`.

The run did emit normal TorchInductor cache artifacts and confirmed the one
token MoE TopK8 fastpath became active after graph capture. This makes the
model/server part of the route usable, but the profiler shutdown method did
not preserve output.

### Promote / Reject

Promote:

- the read-only `libdw.so.1` mount as required profiler launch hygiene for the
  release image;
- launch-time wrapping as capable of taking the MoE GGUF server through graph
  capture, readiness, and a coherent short request;
- the observation that the one-token MoE fastpath was active in the wrapped
  server.

Reject:

- Docker stop / signal-15 shutdown as a way to collect useful `rocprofv3`
  output for this multiprocess vLLM route;
- the current run as profiler evidence, because no profiler artifacts were
  emitted;
- repeating the same launch-wrapper plus Docker-stop shape.

### Reason

The profiler route failed at the shutdown/artifact boundary, not at model
loading or request correctness. The next profiler attempt, if any, needs a
self-terminating in-container wrapper or a reduced/offline worker path that lets
`rocprofv3` flush cleanly after a bounded request. Until that is proven, the
source slice should continue with lower-overhead C/C++ or device-side summaries
instead of another broad server-wrapper run.

## GGUF-249 - MoE ROCProfiler Self-Terminating Wrapper Rejection

### Setup

- Date: 2026-06-28 UTC.
- Host: `.20`.
- Run:
  `<validation-workspace>/runs/moe35b_gguf_tp4_rocprof_selfterm_dot20_20260628T095756Z`.
- Image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Profile: MoE TP4, P2P-on, FP16, GGUF release-shape path.
- Profiler route: launch-time `/opt/rocm/bin/rocprofv3` wrapper with a
  read-only host `libdw.so.1` mount, a mounted profiler output directory, and
  an in-container self-test script.
- Request:
  `Write one concise sentence about local AI reproducibility.`
  with `max_tokens=64` and `temperature=0.0`.

### Question

GGUF-248 rejected Docker-stop shutdown because `rocprofv3` caught signal 15,
waited for children, and emitted no trace files. This probe asked whether an
in-container wrapper could start vLLM, wait for readiness, send one bounded
completion request, send `SIGINT` to the server child, and let `rocprofv3`
flush normally.

### Procedure

The self-test entrypoint started the existing GGUF TP4 launch path as a child
process, waited for `/v1/models`, sent the deterministic short completion
request, slept briefly, and then signaled the child. The wrapper allowed a long
shutdown window before escalating from `SIGINT` to `TERM`.

### Observations

The server loaded, completed graph capture, reached readiness, and returned a
coherent 64-token completion. The completion had:

- prompt tokens: `10`;
- completion tokens: `64`;
- total tokens: `74`;
- `finish_reason`: `length`.

The graph-capture logs provide one useful shape boundary:

- larger prefill and capture shapes such as `368`, `352`, `336`, `8`, `4`,
  and `2` tokens rejected the Qwen TopK8 c1 MoE fastpath;
- `num_tokens=1` activated the Qwen TopK8 c1 MoE fastpath on all four TP
  workers.

The profiler still emitted no CSV, JSON, summary, HIP, RCCL, or kernel trace
files under the mounted `rocprof` output directory. The wrapper reached:

- `request_complete`;
- `child_exited`;
- container exit status `0`;
- GPU VRAM cleared on the TP4 devices.

The logs show `rocprofv3` caught signal 2, waited for child processes, later
caught signal 15, and still produced no profiler artifacts. The run directory
contains TorchInductor generated files, replay timing logs, server logs, and
the short completion JSON, but no usable profiler output.

The replay timing file is also contaminated by profiler overhead and is not
usable as benchmark evidence:

- wrapped GGUF self-test FULL replay average: `66.231362` ms over `252` calls;
- normal GGUF force-unquant FULL replay average: `10.871309` ms over `1020`
  calls;
- HF TP4 release-control FULL replay average: `7.887792` ms over `1980`
  calls.

### Promote / Reject

Promote:

- the self-terminating wrapper as safer operational hygiene than Docker-stop
  shutdown because it exited cleanly and released VRAM;
- the correctness observation that the short GGUF MoE TP4 request remained
  coherent under the profiler wrapper;
- the fastpath shape boundary: capture/prefill and multi-token decode shapes
  reject the c1 TopK8 path, while one-token decode activates it.

Reject:

- broad launch-wrapped `rocprofv3` as a useful profiler route for this
  multiprocess vLLM server path;
- wrapped-server replay timings as benchmark evidence;
- repeating launch-wrapper profiler probes without a narrower worker or
  reproducer boundary;
- treating the run as kernel-level profiling evidence, because no profiler
  artifacts were emitted.

### Reason

This eliminates the previous Docker-stop explanation. The profiler failure is
not just signal-15 shutdown; even a clean self-terminating wrapper reached a
request, exited status `0`, and produced no `rocprofv3` trace files. The next
source slice should move away from broad server profiling and toward a reduced
worker/replay reproducer or lower-overhead C/C++ or device-side summaries that
can inspect the FULL `num_tokens=1` path without relying on `rocprofv3` to
trace the whole vLLM process tree.

## GGUF-250 - MoE Packed Weight Contract Audit

### Setup

- Date: 2026-06-28 UTC.
- Host: `.20`.
- Run:
  `<validation-workspace>/runs/moe35b_gguf_packed_weight_contract_20260628T104313Z`.
- Image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Comparison: HF BF16 snapshot versus the staged Qwen3.6 35B-A3B F16 GGUF
  artifact, both cast to FP32 for metric calculation.
- Layer audited: layer 0.
- TP shape: TP4, full-BAR/P2P-on target geometry.

### Question

Earlier raw tensor checks and live graph checks disagreed enough that the next
source risk was whether the active GGUF path still assembled a different
runtime packed view from HF after the Qwen3.5 linear-attention transforms.
This audit reconstructed the TP4 rank-local packed views directly from HF and
GGUF tensors without launching a server.

### Procedure

The audit used the model config values, not hard-coded dense attention
geometry:

- `linear_num_key_heads=16`;
- `linear_num_value_heads=32`;
- `linear_key_head_dim=128`;
- `linear_value_head_dim=128`.

It applied the same inverse Qwen3.5 value-head reorder used by the active GGUF
overlay, then compared:

- full `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `A_log`,
  `dt_bias`, `conv1d`, and `out_proj`;
- reconstructed TP4 packed `qkvzba` shards for ranks 0 through 3;
- TP4 `out_proj`, `conv1d`, `A_log`, and `dt_bias` shards;
- layer-0 MoE expert `gate_up`, `down`, router, shared expert, and shared
  router tensors.

The audit also included a swapped gate/up expert comparison as a negative
control.

### Observations

The final clean run audited 36 rows:

- rows needing review: `0`;
- negative controls rejected: `1`;
- all audited transformed GGUF tensors and reconstructed TP4 packed shards were
  near-equivalent to the HF snapshot under BF16-to-F16 tolerance;
- the intentionally swapped gate/up expert comparison was rejected as expected.

Representative rows:

- `full.in_proj_qkv`: max abs `2.9802322e-08`, mean abs `8.525864e-12`;
- `tp0.packed_in_proj_qkvzba`: shape `(3088, 2048)`, max abs
  `2.9802322e-08`;
- `tp3.packed_in_proj_qkvzba`: shape `(3088, 2048)`, max abs
  `2.9802322e-08`;
- `moe.experts.gate_up_direct`: near-equivalent;
- `moe.experts.up_gate_swapped`: rejected negative control.

### Promote / Reject

Promote:

- the corrected MoE linear-attention geometry for any future static audit;
- the static packed-weight contract result as evidence that layer-0 raw tensor
  transforms and TP4 packed projection assembly are not the active low-band
  MoE GGUF root cause;
- the negative-control shape, because it proves the audit can reject a
  real gate/up ordering error.

Reject:

- repeating raw Qwen3.5 linear-attention permutation work as the next source
  slice;
- treating the remaining MoE GGUF TPS gap as an obvious layer-0 packed weight
  assembly issue;
- using the earlier incorrect 16-value-head / 256-head-dim audit attempt as
  evidence.

### Reason

The current MoE GGUF path has the right transformed layer-0 tensors and the
right reconstructed TP4 packed views for the main GatedDeltaNet and MoE expert
weights that were audited. The remaining gap should be pursued below static
weight mapping: runtime materialization, generated kernel/code cache behavior,
captured graph state, later-layer runtime-only packed state, or reduced replay
worker diagnostics.

## GGUF-251 - MoE HF/GGUF Graph Surface Compare

### Setup

- Date: 2026-06-28 UTC.
- Host: `.20`.
- Run:
  `<validation-workspace>/runs/moe35b_gguf_graph_surface_compare_20260628T104424Z`.
- GGUF source graph:
  `<validation-workspace>/runs/moe35b_gguf_tp4_force_unquant_linear_dot20_20260628T073755Z/vllm_cache/torch_compile_cache/3289447a2e/rank_0_0/backbone/computation_graph.py`.
- HF source graph:
  `<validation-workspace>/runs/moe35b_hf_tp4_releasepatch_cgreplay_dot20_20260627T232912Z/runtime/root/.cache/vllm/torch_compile_cache/83e072eb23/rank_0_0/backbone/computation_graph.py`.

### Question

After the packed-weight contract audit rejected a layer-0 static mapping
mistake, the next question was whether the active GGUF graph still differs
from the HF replay-control at the high-level compiled graph surface.

### Observations

Rank-0 graph counts matched across the checked surface:

| label | submods | qkvzba weight shape | gemm->3088 | gemm->2048 | gdn core | all_reduce | fused_moe | shared_moe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GGUF | 81 | 31 | 30 | 40 | 60 | 162 | 80 | 120 |
| HF | 81 | 31 | 30 | 40 | 60 | 162 | 80 | 120 |

The active GGUF graph also uses the same folded `qkvzba` per-rank shape seen
in the HF replay-control path:

- `in_proj_qkvz_parameters_weight_`: `f16[3088, 2048]`;
- split: q/k/v, z, b, a in the Qwen3.5 fused path;
- one-token decode activates `gdn_attention_core` and the release MoE path.

### Promote / Reject

Promote:

- high-level graph surface parity as current evidence;
- lower-level runtime/materialization/codegen inspection as the next source
  target.

Reject:

- repeating graph-shape or op-count audits as the next likely fix;
- treating the current MoE GGUF TPS gap as a missing high-level GDN/MoE call
  or obvious folded-projection shape problem.

### Reason

The active GGUF and HF replay-control graphs match on the checked structural
surface, while replay timing still differs substantially. Combined with the
packed-weight contract audit, this shifts the source target below visible graph
shape and static tensor assembly. The next useful experiment should inspect
runtime materialization, generated-kernel/cache factors, captured recurrent or
KV state, or a reduced replay worker that can time the FULL `num_tokens=1`
path without broad server profiling.

## GGUF-252 - MoE Batched TopK8 Tile Variants

### Setup

- Date: 2026-06-28 UTC.
- Host: `.20`.
- Profile: `moe35b_tp4_fullbar_p2pon` GGUF TP4 full-BAR/P2P-on.
- Image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Baseline: force-unquantized GGUF MoE path with the one-token TopK8 c1
  fastpath active.
- Variants:
  - BI8 overlay:
    `<validation-workspace>/experimental-patches/fused-moe-batched-topk8-20260628T115300Z/fused_moe.py`.
  - BI16 overlay:
    `<validation-workspace>/experimental-patches/fused-moe-batched-topk8-bi16-20260628T121200Z/fused_moe.py`.
  - BI32 overlay:
    `<validation-workspace>/experimental-patches/fused-moe-batched-topk8-bi32-20260628T122800Z/fused_moe.py`.
- Ladder: normal warmups, `c1_128` uncapped strict, `c1_2000`, and
  `c1_10000`.

### Question

After graph-surface parity and packed-weight parity rejected the visible graph
and static tensor paths, the next source question was whether the one-token
TopK8 MoE fastpath tile shape was suppressing GGUF TP4 MoE throughput.

### Observations

| Variant | Strict backend TPS | c1_2000 backend TPS | c1_10000 backend TPS | Strict valid | FULL replay avg ms |
| --- | ---: | ---: | ---: | --- | ---: |
| GGUF force-unquant baseline | `83.728` | `85.203` | `75.112` | yes | `10.871309` |
| BI8 TopK8 tile | `83.785` | `84.877` | `74.844` | yes | `12.547244` |
| BI16 TopK8 tile | `84.972` | `86.952` | `76.483` | yes | `12.557635` |
| BI32 TopK8 tile | `85.607` | `86.732` | `76.271` | yes | `10.869545` |

Reference native TP4 remains much higher:

- v0.2 release-time fixed c1_10000: `109.283` backend TPS;
- post-v0.2 strict repeatability range: `113.196` to `115.995` backend TPS.

BI16 was the best c1_2000/c1_10000 tile variant. BI32 was the best strict
variant. None moved GGUF MoE out of the low band.

### Promote / Reject

Promote:

- BI16 as the best c1 tile variant in this family;
- BI32 as the best strict variant in this family;
- the result as evidence that TopK8 tile shape can move a small amount of TPS
  but is not the primary missing native-parity mechanism.

Reject:

- promoting any TopK8 BI8/BI16/BI32 overlay as a release reproduction path;
- treating one-token TopK8 tile shape as the main MoE GGUF performance root;
- repeating this tile family before a lower-level replay/materialization
  finding points back to the fastpath.

### Reason

The tile variants only moved MoE GGUF by a few TPS and stayed roughly
`33` backend TPS below the native TP4 c1_10000 release-time value. The
remaining source target is still below high-level fused-MoE graph shape,
regular GGUF linear dispatch, first-call active TopK8 materialization, and
simple TopK8 tile geometry. Continue with lower-level captured-state,
runtime-materialization, or reduced replay-worker probes.

## GGUF-253 - Release Native SwiGLU Applicability Check

### Setup

- Date: 2026-06-28 UTC.
- Source inspected inside the release image:
  `/opt/gfx906_release_runtime/python_overlays/qwen2_moe_interleaved_swiglu_20260608.py`.
- Related release native artifact:
  `/opt/gfx906_release_runtime/native/swiglu/gfx906_swiglu_gemv_ext_20260607.so`.

### Question

Could the native SwiGLU overlay from the published release image explain the
remaining native-vs-GGUF MoE TP4 gap, and should it be forced into the GGUF
TP4 path?

### Observations

The overlay is guarded for the Qwen MoE shared-expert path and only enables the
native SwiGLU call when the release-specific shape and runtime conditions are
met:

- `VLLM_GFX906_QWEN_MLP_INTERLEAVED_SWIGLU` is enabled;
- `quant_config is None`;
- `hidden_size == 5120`;
- `intermediate_size == 17408`;
- tensor parallel size is `8`;
- the `gate_up` weight exists;
- the native call receives the expected `(4352, 5120)` shape.

The routed-expert path still goes through the fused-MoE path. The shape and TP
guards make this overlay a TP8 shared-expert optimization, not a direct TP4
GGUF routed-expert fix.

### Promote / Reject

Promote:

- the native SwiGLU guard as a documented release-image boundary;
- targeted inspection of release native overlays before attempting to port or
  force them into GGUF.

Reject:

- forcing the TP8 shared-expert native SwiGLU overlay into the TP4 GGUF MoE
  path;
- treating this overlay as the obvious missing mechanism for the TP4 GGUF
  routed-expert gap.

### Reason

The native SwiGLU overlay is too shape- and TP-specific to apply blindly to the
current TP4 GGUF MoE problem. It remains relevant release source context, but
the active MoE GGUF gap should stay focused on the routed-expert/captured-state
and replay-materialization path unless a later trace points back to shared
expert execution.

## GGUF-254 - Exact HF Release TopK8 Fastpath on GGUF TP4

### Setup

- Date: 2026-06-28 UTC.
- Hosts: `.20` and `.30`.
- Profile: `moe35b_tp4_fullbar_p2pon` GGUF TP4 full-BAR/P2P-on.
- Image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Exact fastpath source:
  `<validation-workspace>/experimental-patches/fused-moe-hf-release-exact-20260628T1330Z/fused_moe.py`.
- Source checksum:
  `66f63f74406c2a805e78eeb28e9dac76c3f98bc4ce9046cd3be5d604224a0a0e`.
- Run directories:
  - `.20`: `moe35b_gguf_tp4_hf_release_fastpath_exact_dot20_20260628T1330Z`
  - `.30`: `moe35b_gguf_tp4_hf_release_fastpath_exact_dot30_20260628T1330Z`
- Ladder: normal warmups, `c1_128` uncapped strict, `c1_2000`, and
  `c1_10000`.

### Question

Earlier TopK8 tile and scheduler probes used copied or modified fastpath
overlays. This run checked the exact HF TP4 release fastpath file against the
current GGUF force-unquantized loader to rule out a mismatched source copy as
the reason GGUF MoE TP4 stays below native TP4 throughput.

### Observations

Both hosts reached API health under the release image and activated the exact
one-token TopK8 fastpath after graph capture. The exact fastpath rejected larger
prefill/capture shapes as expected and logged activation for decode token counts
`4`, `2`, and `1`.

| Host | Strict backend TPS | c1_2000 backend TPS | c1_10000 backend TPS | Strict valid |
| --- | ---: | ---: | ---: | --- |
| `.20` | `81.036` | `84.783` | `74.793` | yes |
| `.30` | `83.302` | `84.751` | `74.762` | yes |

For comparison, the prior GGUF force-unquantized baseline on `.20` was strict
`83.728`, `c1_2000` `85.203`, and `c1_10000` `75.112` backend TPS. Native TP4
remains much higher: the v0.2 release-time fixed `c1_10000` value is `109.283`
backend TPS, and post-v0.2 strict repeatability landed from `113.196` to
`115.995` backend TPS.

### Promote / Reject

Promote:

- the exact HF release fastpath run as a closed comparability check;
- `.30` as an equivalent lane for this source test, because its results
  matched `.20` within the same low band;
- the result as evidence that missing or mismatched TopK8 fastpath source is
  not the remaining MoE GGUF TP4 performance explanation.

Reject:

- the exact HF release TopK8 fastpath as a GGUF TP4 performance fix;
- repeating fastpath-source-copy experiments unless a later trace points back
  to the one-token TopK8 helper;
- treating the `.20` versus `.30` lane difference as the root cause of the
  MoE GGUF gap.

### Reason

The exact HF release fastpath activates correctly on GGUF TP4, but throughput
stays in the same low band and slightly trails the prior GGUF baseline on the
important `c1_10000` tier. This closes the "wrong fastpath source" hypothesis.
The remaining target stays below source-file selection and high-level graph
shape: runtime materialization, generated replay body, captured state,
memory movement around expert prepare/finalize, or a reduced FULL replay
diagnostic.

## GGUF-255 - MoE TopK8 Active Pointer Alignment Diagnostic

### Setup

- Date: 2026-06-28 UTC.
- Host: `.20`.
- Profile: `moe35b_tp4_fullbar_p2pon`.
- Image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- HF control run:
  `moe35b_hf_tp4_active_ptr_dot20_20260628T1340Z`.
- GGUF run:
  `moe35b_gguf_tp4_active_ptr_dot20_20260628T1340Z`.
- Instrumented source:
  `<validation-workspace>/experimental-patches/fused-moe-active-ptr-debug-20260628T1340Z/fused_moe.py`.
- Source checksum:
  `f5e20bd6e450371dbf36db9f6a2908da873aebf88c8cb367a2a6345568b19270`.
- Request shape: short chat completion after graph capture, enough to activate
  the one-token TopK8 path and emit pointer/storage diagnostics.

### Question

Static packed-weight checks, graph-surface checks, and exact fastpath-source
checks did not explain why GGUF TP4 replay stays slower than the HF/native
control. This diagnostic checked whether the active one-token TopK8 tensors
were entering the fastpath with different storage offsets or pointer alignment.

### Observations

Both runs returned coherent text for the same short prompt. The HF control used
the high-band native-weight route, while the GGUF run used the current
force-unquantized GGUF route.

| Route | Active detail rows | FULL replay avg ms | `w1` offset/mod256/mod4096 | `w2` offset/mod256/mod4096 |
| --- | ---: | ---: | --- | --- |
| HF control | `448` | `7.518159` | `0 / 0 / 0` | `0 / 0 / 0` |
| GGUF | `220` | `9.979224` | `0 / 0 / 0` | `0 / 0 / 0` |

For both routes:

- `hidden_storage_offset=0` and `hidden_ptr_mod256=0`;
- `w1_storage_offset=0`, `w1_ptr_mod256=0`, and `w1_ptr_mod4096=0`;
- `w2_storage_offset=0`, `w2_ptr_mod256=0`, and `w2_ptr_mod4096=0`;
- `topk_ids_storage_offset=0` and `topk_ids_ptr_mod256=0`;
- `topk_weights_storage_offset=0` and `topk_weights_ptr_mod256=0`.

The 4K buckets for hidden, top-k IDs, and top-k weights varied in both runs,
which matches normal allocator behavior for transient active tensors. The
packed expert tensors stayed 4K-aligned in both runs.

### Promote / Reject

Promote:

- the pointer diagnostic as a closed comparability check for the active
  one-token TopK8 path;
- the current evidence that the GGUF replay gap is still present even when
  visible storage offsets and pointer alignment match the HF control;
- the next source target below tensor pointer alignment: generated replay body,
  runtime materialization around expert prepare/finalize, captured state, or a
  reduced FULL replay worker.

Reject:

- `w1` / `w2` storage offset or pointer alignment as the primary explanation
  for the GGUF TP4 MoE gap;
- repeating active pointer-bucket checks unless a later source change alters
  tensor materialization or allocation.

### Reason

The hypothesis predicted that GGUF might be paying a replay penalty because
active expert tensors were not laid out like HF tensors. The measured active
storage contract did not support that. Both routes had zero storage offsets,
256-byte alignment for active tensors, and 4K-aligned packed `w1` / `w2`.
GGUF remained slower, so the next useful work should inspect what the captured
FULL replay actually does with otherwise equivalent visible tensor surfaces.

## GGUF-256 - HF Inductor Cache Preseed on GGUF TP4

### Setup

- Date: 2026-06-28 UTC.
- Host: `.20`.
- Profile: `moe35b_tp4_fullbar_p2pon`.
- Image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Base GGUF run:
  `moe35b_gguf_tp4_active_ptr_dot20_20260628T1340Z`.
- HF cache source:
  `moe35b_hf_tp4_active_ptr_dot20_20260628T1340Z`.
- Preseeded GGUF run:
  `moe35b_gguf_tp4_hf_inductor_cache_dot20_20260628T1405Z`.
- Instrumented source:
  `<validation-workspace>/experimental-patches/fused-moe-active-ptr-debug-20260628T1340Z/fused_moe.py`.
- Source checksum:
  `f5e20bd6e450371dbf36db9f6a2908da873aebf88c8cb367a2a6345568b19270`.

### Question

The HF and GGUF runs generated the same number of TorchInductor artifacts, and
all `*.py` generated source files matched byte-for-byte. The remaining
difference was autotune metadata: several `*.best_config` and `*.kernel_perf`
files differed. This run checked whether seeding the GGUF run with the HF
TorchInductor cache and best configs could move replay toward HF performance.

### Observations

The run copied all `488` HF TorchInductor artifacts into the GGUF run before
launch. After serving, the preseeded GGUF run still had `208` generated Python
files, `140` `*.best_config` files, and `76` `*.kernel_perf` files. Its
normalized `*.best_config` contents matched the HF control exactly after
removing only timing metadata.

The preseeded GGUF run returned coherent short-prompt text.

| Route | Request | FULL replay avg ms | Min ms | Max ms | Replay rows |
| --- | --- | ---: | ---: | ---: | ---: |
| GGUF normal pointer diagnostic | first request | `9.979224` | `0.220960` | `204.865204` | `1184` |
| GGUF HF-cache preseed | first request | `13.746053` | `0.202080` | `1098.631592` | `1184` |
| GGUF HF-cache preseed | second request | `9.190717` | `0.220640` | `11.713107` | `864` |
| HF pointer diagnostic | first request | `7.518159` | `0.229920` | `273.329346` | `1184` |

The second preseeded request removed the large first-request outlier and showed
a modest steady replay improvement over the normal GGUF pointer diagnostic, but
it remained materially slower than HF.

### Promote / Reject

Promote:

- the fact that generated TorchInductor Python source matches HF and GGUF for
  this diagnostic slice;
- the fact that HF-normalized best configs can be retained by the GGUF run;
- HF cache preseeding as a minor diagnostic lever that can shift steady replay
  but not as a release fix.

Reject:

- HF inductor-cache preseeding as a sufficient GGUF TP4 MoE performance fix;
- spending a full warmup / strict / c1 ladder on this cache-preseeded route
  unless a later source change makes replay much closer to HF first;
- treating autotune config selection alone as the native-parity blocker.

### Reason

If the main problem were simply that GGUF selected worse generated kernel
configs, the HF-preseeded cache should have moved replay close to the HF
control. It did not. The steady second request improved to `9.190717` ms but
stayed well above the HF `7.518159` ms diagnostic band. This points below
generated Python source and best-config selection: captured inputs/state,
runtime materialization around expert preparation/finalization, or another
host-side replay setup cost that is not encoded in the visible inductor source
or config cache.

## GGUF-257 - CUDA Graph Replay Input Boundary Diagnostic

### Setup

- Date: 2026-06-28 UTC.
- Host: `.20`.
- Profile: `moe35b_tp4_fullbar_p2pon`.
- Image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- HF control run:
  `moe35b_hf_tp4_graph_input_dot20_20260628T1420Z`.
- GGUF run:
  `moe35b_gguf_tp4_graph_input_dot20_20260628T1420Z`.
- CUDA graph replay input patch:
  `<validation-workspace>/experimental-patches/cudagraph-input-debug-20260628T1420Z/vllm/compilation/cuda_graph.py`.
- Source checksum:
  `5a55161238a3b7579d94490d7626453b82fa1f6617357119665eb5dfae58ad32`.

### Question

The previous diagnostics showed that HF and GGUF generated identical
TorchInductor Python source and had matching visible one-token active tensor
alignment, but GGUF still replayed slower. This run moved the probe up to the
CUDA graph replay boundary and logged the request-time FULL replay inputs
immediately before `entry.cudagraph.replay()`.

The question was whether the GGUF route was replaying a different actual
decode stream, or whether the slowdown lived below the top-level `input_ids`
and `positions` boundary.

### Observations

The first eight logged one-token FULL replay calls matched between HF and GGUF
for both `input_ids` and `positions`.

| Call | HF input ID | GGUF input ID | HF position | GGUF position |
| ---: | ---: | ---: | ---: | ---: |
| 42 | `90700` | `90700` | `20` | `20` |
| 43 | `8340` | `8340` | `21` | `21` |
| 44 | `25` | `25` | `22` | `22` |
| 45 | `271` | `271` | `23` | `23` |
| 46 | `16` | `16` | `24` | `24` |
| 47 | `13` | `13` | `25` | `25` |
| 48 | `220` | `220` | `26` | `26` |
| 49 | `2972` | `2972` | `27` | `27` |

The tensor surfaces also matched:

- `input_ids`: shape `(1,)`, dtype `torch.int32`, stride `(1,)`,
  storage offset `0`, `ptr_mod256=0`, `ptr_mod4096=3584`;
- `positions`: shape `(3, 1)`, dtype `torch.int64`, stride `(2049, 1)`,
  storage offset `0`, `ptr_mod256=0`, `ptr_mod4096=512`.

The short GGUF request returned coherent Qwen-style reasoning text, but the
64-token cap was spent in the reasoning field before a final content message.
That is not a correctness failure for this diagnostic; the replay input stream
matched the HF control for the measured calls.

The HF and GGUF response message hashes also matched for this request:

- reasoning SHA256:
  `5f225a515d6d56f0f0badb37a31f76ee1967b3b4530be1747576d637a57348b7`;
- content SHA256:
  `01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b`;
- usage: `20` prompt tokens, `64` completion tokens, `84` total tokens.

| Route | FULL replay avg ms | Min ms | Max ms | Replay rows |
| --- | ---: | ---: | ---: | ---: |
| HF control | `8.031084` | `7.494225` | `14.327335` | `252` |
| GGUF | `10.843224` | `10.424463` | `21.287489` | `252` |

The logs also showed that both HF and GGUF reject the Qwen TopK8 c1 fastpath
for larger capture/prefill shapes because those are not the one-token decode
case. That rejection is therefore not GGUF-specific. One-token decode still
reaches the active detail path on both routes.

### Promote / Reject

Promote:

- the CUDA graph replay input boundary as a closed comparator for the short
  request;
- the conclusion that HF and GGUF replay the same top-level request token and
  position stream for the measured calls;
- the remaining source target below this boundary: captured lower-level state,
  runtime materialization around expert prepare/finalize, recurrent/KV state
  plumbing, or a reduced FULL replay worker.

Reject:

- request-time `input_ids` divergence as the cause of the GGUF TP4 MoE replay
  gap for this prompt;
- response-text divergence as the cause of the GGUF TP4 MoE replay gap for
  this prompt;
- request-time `positions` shape, stride, offset, or pointer alignment as the
  cause for this prompt;
- larger-shape Qwen TopK8 fastpath rejection as a GGUF-specific issue, since
  HF shows the same expected rejection outside one-token decode.

### Reason

If GGUF were slower because it generated a different decode token stream or
different top-level replay inputs, the first request-time FULL replay calls
would have diverged at `input_ids` or `positions`. They did not. GGUF still
averaged `10.843224` ms versus HF at `8.031084` ms over the same number of
FULL replay rows. The next useful work should therefore move below the replay
input boundary and avoid repeating tokenizer, request stream, visible pointer
alignment, generated Python source, or best-config cache checks unless a later
source change alters those surfaces.

## GGUF-258 - Runner Metadata Trace Rejection

### Setup

- Date: 2026-06-28 UTC.
- Host: `.20`.
- Intended profile: `moe35b_tp4_fullbar_p2pon`.
- HF attempted run:
  `moe35b_hf_tp4_runner_meta_dot20_20260628T1505Z`.
- HF runner metadata patch checksum:
  `853f61cdc813583a520ea64d28858e497e0dfec64de6b28fdfa41f47938d0022`.
- GGUF runner metadata patch checksum, prepared but not run:
  `630018c8454a77e69f0af20dea71bbcc1e76170f06a85fb74524f21440405ce0`.

### Question

After the CUDA graph replay input-boundary diagnostic, the next candidate was
small runner metadata immediately before the model call: input IDs, positions,
query starts, sequence lengths, discard masks, input-batch counters, slot
mappings, and model-kwargs tensor surfaces. The intent was to compare
lower-level metadata without copying large activations.

### Observations

The first HF patch attempt used a release-image runner base that did not import
`os`; worker initialization failed immediately. That patch was fixed by adding
the missing import.

The corrected HF control reached readiness, but the metadata logger filled its
limit during CUDA graph capture with dummy capture rows rather than request
replay rows. The recorded rows had `input_ids` values `[0]`, `positions`
values `[0, 0, 0]`, and graph-capture metadata such as `query_start_loc`
`[0, 1, 2, 3, 4, 5, 6, 7]`.

After readiness, the short request stalled with `0%` GPU use and a shared-memory
wait warning. Stopping the container returned an empty response to the client.
No GGUF run was launched with this patch because the HF control was already
invalid.

### Promote / Reject

Promote:

- the need to distinguish CUDA graph capture-time dummy inputs from request-time
  replay inputs in any runner-level probe;
- the constraint that runner metadata logging must be gated on request replay,
  not merely `CUDAGraphMode.FULL` and `num_tokens_padded == 1`.

Reject:

- this runner metadata patch as a valid HF/GGUF comparator;
- copying GPU metadata at the model-runner call site during capture/replay as a
  safe next step in this form;
- running the matching GGUF route with this patch.

### Reason

The patch answered a tooling question, not a model-performance question. It
showed that the chosen insertion point observes graph-capture dummy rows before
it observes real request rows, and its small GPU-to-CPU metadata copies were
enough to perturb the HF request path. The next runner-level probe should either
hook after graph capture with a request-only guard from the scheduler/forward
context, or avoid in-process GPU reads and use a reduced replay worker where
state snapshots are explicit.

## GGUF-259 - Active-Pointer Kernel-Perf Artifact Recheck

### Setup

- Date: 2026-06-28 UTC.
- Host: `.20`.
- HF cache:
  `moe35b_hf_tp4_active_ptr_dot20_20260628T1340Z/runtime/tmp/torchinductor_root`.
- GGUF cache:
  `moe35b_gguf_tp4_active_ptr_dot20_20260628T1340Z/torchinductor_root`.

### Question

The active-pointer diagnostic showed a live replay gap even though visible
tensor alignment matched. This read-only artifact recheck asked whether the
generated TorchInductor `.kernel_perf` files from that same run pair explained
the gap.

### Observations

Both caches had `76` `.kernel_perf` files and all paths joined. Summing all
numeric entries by joined relative path produced:

| Route | `.kernel_perf` sum |
| --- | ---: |
| HF | `2.060957000` |
| GGUF | `1.980158011` |

The aggregate ratio was `0.961x` for GGUF versus HF, meaning this metadata
slightly favored GGUF even though live replay favored HF.

The largest individual GGUF-slower metadata deltas were small pointwise or
reduction kernels, such as:

- `./ea/ceaxnbyew7o4cln4eezj7lj7ovss5njpl2ozhjxvafcjvo7azfng.kernel_perf`:
  HF `0.007840000`, GGUF `0.051840000`;
- `./b7/cb7krozuo6oqh5bk3z3cfk5bqju2niwvmbvckhhn5z5zirj376jj.kernel_perf`:
  HF `0.012960000`, GGUF `0.048640002`;
- `./pv/cpvudhw4q3mrthtjciujguj2tb5mfkzvdu2fxh574tqzh3jqhcax.kernel_perf`:
  HF `0.014559000`, GGUF `0.049919002`.

### Promote / Reject

Promote:

- `.kernel_perf` artifact comparison as a useful secondary sanity check;
- the fact that generated-kernel metadata is not aligned with the live replay
  direction for this run pair.

Reject:

- `.kernel_perf` aggregate timing as the explanation for the MoE GGUF TP4
  replay gap;
- chasing individual pointwise/reduction `.kernel_perf` deltas before obtaining
  request-window kernel mix or a reduced replay reproducer.

### Reason

If generated-kernel benchmark metadata were the primary cause, the aggregate
direction should have matched the live replay direction. It did not. The
request-time gap remains a captured execution/context problem rather than a
simple TorchInductor artifact metadata problem.

## GGUF-260 - Replay Value-Signature Decode Audit

### Setup

- Date: 2026-06-28 UTC.
- Host: `.20`.
- Audit run:
  `moe35b_valuesig_replay_decode_audit_dot20_20260628T1515Z`.
- Source HF/GGUF replay compare:
  `moe35b_valuesig_replay_compare_20260628T0450Z`.
- Source GGUF pre/post value-signature run:
  `moe35b_gguf_tp4_prepost_valuesig_dot20_20260628T064605Z`.
- Audit artifacts:
  - `summary.md`;
  - `piecewise_arg_shape_map.txt`;
  - `sha256.txt`.

### Question

The previous CUDA graph input-boundary diagnostic closed the top-level
`input_ids` and `positions` surface for a short one-token replay request. The
remaining question was whether the older value-signature diffs contained useful
FULL decode evidence below that boundary, or whether those diffs came from
PIECEWISE capture/replay rows that should not drive the current one-token
decode investigation.

### Observations

The replay header counts showed:

| Replay mode | Batch shape | Rows | Args |
| --- | --- | ---: | ---: |
| FULL | `num_tokens=1`, `num_reqs=1` | `124` | `0` |
| PIECEWISE | `num_tokens=16` | `80` | `12` |
| PIECEWISE | `num_tokens=16` | `40` | `17` |
| PIECEWISE | `num_tokens=16` | `36` | `11` |
| PIECEWISE | `num_tokens=16` | `4` | `9` |
| PIECEWISE | `num_tokens=16` | `4` | `6` |

The HF/GGUF value-signature diff table contained `42` grouped rows, all under
PIECEWISE `num_tokens=16` keys. The GGUF pre/post signature run contained four
diff groups:

| Count | Key |
| ---: | --- |
| `80` | `argc=12 arg[5]` |
| `40` | `argc=17 arg[5]` |
| `36` | `argc=11 arg[4]` |
| `4` | `argc=9 arg[4]` |

Those pre/post rows were PIECEWISE `num_tokens=24` signatures, not FULL
one-token decode rows. The shape map still helps identify the involved surfaces
as generated PIECEWISE graph arguments, including hidden-state buffers,
projection weights, bias vectors, recurrent-state-like buffers, and
`ModuleName` expert references. It does not prove that the FULL
`num_tokens=1` decode graph is receiving different lower-level values.

### Promote / Reject

Promote:

- the read-only value-signature audit as a closed interpretation of the older
  value-signature artifacts;
- the conclusion that these artifacts are useful for PIECEWISE graph-state
  context, but not as direct FULL decode root-cause evidence;
- the need for a request-window kernel mix trace or a reduced FULL
  `num_tokens=1` replay worker if we want evidence below the
  `cuda_graph.py` replay input boundary.

Reject:

- treating the older value-signature diffs as proof of the current GGUF TP4 MoE
  one-token replay bottleneck;
- using the GGUF pre/post `arg[4]` / `arg[5]` diffs as a release-candidate fix
  target without a FULL decode reproducer;
- repeating broad value-signature scans that only expose PIECEWISE rows.

### Reason

The active performance gap is in the normal decode ladder, where the useful
comparator has been FULL one-token replay timing. In this logging surface, the
FULL rows carry `argc=0`, so the value-signature tensors available for
comparison are not the tensors inside the FULL captured replay. The observed
value diffs are real, but they sit on PIECEWISE rows and can describe
prefill/capture-like or multi-token replay surfaces. The next useful step is
not another value-signature scrape; it is a lower-overhead way to observe the
request-window kernel mix or a reduced FULL replay reproducer that exposes the
critical captured state without Python CPU copies.

## GGUF-261 - HIP Graph LD_PRELOAD Trace Rejection

### Setup

- Date: 2026-06-28 UTC.
- Host: `.20`.
- Tool path:
  `tools/hip_graph_trace_20260628T1525Z`.
- Linked shim checksum:
  `3a7274dd3164b6df26933425c76cf808cacd31433073fe66e7f900b5d76fc4ab`.
- No-link shim was rebuilt with `g++`, `-D__HIP_PLATFORM_AMD__`, and no
  `libamdhip64` runtime dependency.
- Smoke test: `smoke_graph` captured, instantiated, and launched a one-kernel
  HIP graph successfully under the shim.
- Failed GGUF run:
  `moe35b_gguf_tp4_hipgraph_dot_dot20_20260628T1540Z`.

### Question

The goal was to avoid another Python-side timing probe by using a C/C++
`LD_PRELOAD` shim to observe HIP graph instantiation, graph node counts, DOT
graph dumps, and graph-launch host timing from outside the release container.

### Observations

The first linked shim could not be used as `LD_PRELOAD` because the dynamic
loader tried to resolve `libamdhip64.so.6` before the entrypoint had established
the release container `LD_LIBRARY_PATH`.

The no-link shim loaded successfully and passed a standalone HIP graph smoke
test:

- one graph instantiated;
- one graph launched;
- one DOT file emitted;
- graph launch host time was recorded.

The full vLLM GGUF TP4 server did not reach readiness under the no-link shim.
It finished model loading and entered compile/graph-cache setup, then a worker
failed during KV-cache initialization with a HIP unknown error. After exit:

- no graph node files were emitted;
- no DOT files were emitted;
- shim summaries showed only process load/unload rows and zero graph
  instantiate / graph launch counters for the worker processes;
- GPU VRAM and activity returned to idle;
- the HF half was not launched because the GGUF route was already invalid.

### Promote / Reject

Promote:

- the standalone C/C++ HIP graph shim smoke test as proof that the interposer
  can work for a small isolated HIP graph;
- the operational lesson that full-server `LD_PRELOAD` graph interposition is
  not safe enough in this form for vLLM/GFX906 MoE startup.

Reject:

- the GGUF vLLM run as model-performance evidence;
- this `LD_PRELOAD` graph interposition route as a repeat path for full-server
  MoE GGUF debugging;
- launching the HF control with the same shim.

### Reason

The shim perturbed the full vLLM server before it produced any graph evidence.
Because graph counters remained at zero and the server failed before readiness,
the run does not tell us whether HF and GGUF captured different graph topology
or kernel mix. The next lower-level route should avoid process-wide
`LD_PRELOAD` in the full server. A reduced replay worker or a deliberately
scoped in-process trace around a known request window is still the better path.

## GGUF-262 - MoE TP4 Release-Overlay Combo Promotion

### Setup

- Date: 2026-06-28 UTC.
- Hosts: `.20` and `.30`.
- Profile: `moe35b_tp4_fullbar_p2pon`.
- Image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Mode: FP16 GGUF, TP4, P2P-on, `MAX_MODEL_LEN=131072`, graph mode, async
  scheduling, release MoE tuned config, normal benchmark warmups, then
  `c1_128` uncapped strict, `c1_2000`, and `c1_10000`.
- Combo bundle:
  `qwen35moe-gguf-release-overlay-combo-20260628T1615Z`.
- `.20` run:
  `moe35b_gguf_tp4_release_overlay_combo_dot20_20260628T1635Z`.
- `.30` run:
  `moe35b_gguf_tp4_release_overlay_combo_dot30_20260628T1645Z`.

The combo bundle started from the active release overlay bundle and overlaid
the GGUF compatibility fixes plus the exact HF release TP4 fastpath source.
The first `.20` launch mounted `<validation-nvme-root>` read-only and failed before
startup because the release entrypoint copies the native SwiGLU runtime into
the NVMe model/work area. Relaunching with the release-style read-write
`<validation-nvme-root>` mount fixed the startup issue without mutating the image.

### Question

The exact HF release TopK8 fastpath alone had not moved GGUF MoE TP4 out of
the low band. The question was whether the missing piece was the full release
overlay composition, not a single `fused_moe.py` file: release Python overlays,
native runtime setup, GGUF loader/model repairs, and the release fastpath all
mounted together under the native MoE TP4 profile.

### Observations

Both hosts reached health after the expected long model load and graph-capture
window. Logs showed the same shape boundary on both lanes:

- larger graph-capture / prefill shapes rejected the Qwen c1 TopK8 fastpath on
  layout;
- small decode shapes activated it for token counts `4`, `2`, and `1`;
- the normal warmups stabilized around the native TP4 speed band before the
  measured tiers.

The completed ladder results were:

| Host | Strict backend TPS | c1_2000 backend TPS | c1_10000 backend TPS | Strict gate |
| --- | ---: | ---: | ---: | --- |
| `.20` | `118.754` | `119.896` | `112.747` | valid |
| `.30` | `119.917` | `120.781` | `113.605` | valid |

The `.30` lane was not slower than `.20` under the same combo path. In this run
it was slightly faster across all three measured tiers.

For comparison, the prior `.30` exact-HF-fastpath-only run stayed low-band:
strict `83.302`, c1_2000 `84.751`, and c1_10000 `74.762`. The difference is
therefore not lane choice alone and not the standalone fastpath file alone.

### Promote / Reject

Promote:

- the release-overlay combo bundle as the first MoE GGUF TP4 path in this work
  stream that matches or exceeds the native TP4 performance band on both `.20`
  and `.30`;
- `.30` as an equivalent-performing validation lane for this profile after
  repaste;
- the full overlay-composition hypothesis: GGUF correctness fixes plus the
  complete release overlay stack and exact release fastpath must be applied
  together for high-band TP4 MoE GGUF.

Reject:

- the earlier low-band exact-fastpath-only result as representative of the full
  release-overlay GGUF path;
- lane choice as the remaining explanation for the old MoE GGUF TP4 gap;
- repeating single-file fastpath swaps without the full release overlay
  composition.

### Reason

This result changes the active MoE GGUF status from performance-blocked to
promotion-candidate. The one-token TopK8 fastpath was already known to activate
in previous diagnostics, but exact fastpath source alone did not recover
throughput. The high-band result appears only when the full release overlay
bundle, native runtime setup, GGUF loader/model repairs, default graph shape,
P2P-on platform state, and exact release fastpath are combined.

The next step is to make this path reproducible instead of treating the scratch
combo as a hand-built artifact: reduce the bundle to the minimal required
source changes, write a clear deployment path, rerun both hosts from that path,
and only then consider public documentation or release-note changes.

## GGUF-263 - `.20` Dense and MoE Goal Closure Audit

### Setup

- Date: 2026-06-28 UTC.
- Host: `.20`.
- Image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Platform mode: full-BAR/P2P-on with `NCCL_P2P_DISABLE=0`.
- Benchmark sequence: normal pre-measure warmups, then `c1_128` uncapped
  strict, `c1_2000`, and `c1_10000`.
- Strict tier:
  `uncapped_no_max_tokens_require_think_close`.
- `MAX_MODEL_LEN=131072`.

Dense run:

- Profile: `dense27b_tp8_fullbar_p2pon`.
- Model file:
  `Qwen3.6-27B-F16.gguf`.
- TP: `8`.
- Served model:
  `Qwen3.6-27B-F16-GGUF-lmheadunquant`.
- Run:
  `dense27b_gguf_tp8_lmheadunquant_dot20_lanevalidate_20260628T110342Z`.

MoE run:

- Profile: `moe35b_tp4_fullbar_p2pon`.
- Model file:
  `Qwen3.6-35B-A3B-F16.gguf`.
- TP: `4`.
- Served model:
  `Qwen3.6-35B-A3B-F16-GGUF-fusedgdn-base-rmsnorm`.
- Patch bundle:
  `qwen35moe-gguf-release-overlay-combo-20260628T1615Z`.
- Run:
  `moe35b_gguf_tp4_release_overlay_combo_dot20_20260628T1635Z`.

### Results

| Model path | Strict backend TPS | c1_2000 backend TPS | c1_10000 backend TPS | Strict gate |
| --- | ---: | ---: | ---: | --- |
| Dense 27B GGUF TP8 | `70.505` | `71.589` | `66.967` | valid |
| MoE 35B-A3B GGUF TP4 | `118.754` | `119.896` | `112.747` | valid |

### Comparison Targets

Published/current dense target:

- strict `69.514`;
- c1_2000 `70.347`;
- c1_10000 `66.069`.

Current MoE TP4 native/reference band:

- v0.2 release fixed-token: c1_2000 `116.146`, c1_10000 `109.283`;
- post-v0.2 `.20` fixed sanity: c1_2000 `116.787`, c1_10000 `109.622`;
- post-v0.2 strict range: `113.196` to `115.995`.

### Outcome

Promote for the current GGUF goal: on `.20`, both GGUF paths match or beat the
current dense and MoE benchmark targets using vLLM, the release image, P2P-on,
normal warmups, uncapped strict, and the fixed-token decode tiers.

### Reason

Dense no longer only has a near-threshold result; the final `.20` lane
validation run clears all three published dense tiers. MoE no longer remains in
the earlier low band; the release-overlay combo crosses the native TP4 band on
strict and fixed-token tiers. Both result sets used the same benchmark ladder
shape as the release work and preserved `MAX_MODEL_LEN=131072`.

### Next

Do not change public release claims directly from these scratch-path results.
The next engineering step is to reduce the dense and MoE GGUF overlays into a
minimal reproducible package, rerun the package path on `.20` and `.30`, and
only then decide whether to publish a GGUF reproduction release.

## GGUF-264 - vNext Contract Launcher Preflight

### Setup

- Date: 2026-06-28 UTC.
- Scope: reproduction scaffolding only; no vLLM server launch and no public
  claim update.
- New flow under test:
  binary model probe -> profile contract validation -> overlay/env selection ->
  generated vLLM command -> saved effective config -> benchmark ladder handoff.
- Profiles checked:
  - `hf-dense27b-tp8`
  - `hf-moe35b-tp4`
  - `hf-moe35b-tp8`
  - `gguf-dense27b-tp8`
  - `gguf-moe35b-tp4`

### Tests

Local synthetic tests:

- C model-format probe classified GGUF, safetensors, and legacy PyTorch bytes.
- Mixed-format directory failed closed with `confidence=conflict`.
- HF Dense and HF MoE profile artifacts generated and verified.
- GGUF Dense and GGUF MoE profile artifacts generated and verified when an
  explicit development patch-bundle hash override was provided.
- Unpinned GGUF Dense and GGUF MoE profiles failed closed.
- GGUF bytes under an HF profile failed closed.
- MoE GGUF bytes under the Dense GGUF profile failed closed.
- Leaked `VLLM_QWEN35_GGUF_*` env into an HF launch failed closed.
- A temporary bad bundle manifest using absolute paths was rejected.

Real-model preflight:

| Host | Profile | Model format | Detected family | TP | P2P | Artifact verifier |
| --- | --- | --- | --- | ---: | --- | --- |
| `.20` | `gguf-dense27b-tp8` | `gguf` | `qwen36_dense` | `8` | on | valid |
| `.20` | `gguf-moe35b-tp4` | `gguf` | `qwen36_moe` | `4` | on | valid |
| `.30` | `gguf-dense27b-tp8` | `gguf` | `qwen36_dense` | `8` | on | valid |
| `.30` | `gguf-moe35b-tp4` | `gguf` | `qwen36_moe` | `4` | on | valid |

### Outcome

Promote:

- the contract-driven launcher shape as the correct next release direction;
- binary model probing instead of extension parsing;
- separated `hf_release`, `gguf_dense`, and `gguf_moe` overlay scopes;
- generated launch commands and saved effective configs;
- fail-closed profile/format/family/overlay validation;
- `.20` and `.30` as valid preflight lanes for the current real GGUF model
  artifacts.

Reject:

- treating the current scratch patch bundles as release-ready reproduction
  bundles;
- production use of `EXTRA_VLLM_ARGS` for the vNext path;
- publishing GGUF claims from scratch overlays before minimal bundle reduction
  and clean reruns.

### Reason

The launcher now protects the HF baseline from GGUF env leakage and prevents
Dense/MoE overlay mismatch before vLLM starts. It also records the image
tag/digest, TP degree, P2P state, `MAX_MODEL_LEN=131072`, dtype, selected
overlay, patch-bundle fields, generated vLLM command, and benchmark ladder
shape.

The remaining blocker is packaging discipline, not basic profile detection:
the scratch overlay bundles must be reduced to minimal bundles with deterministic
relative-path manifests that exclude `sha256sum.txt` and pass
`sha256sum -c sha256sum.txt` from the bundle root. After that, rerun the full
warmup -> strict -> c1_2000 -> c1_10000 ladder on `.20` and `.30`, then rerun
the HF baseline profiles to prove the GGUF overlay work did not regress the
published non-GGUF path.

## GGUF-265 - Minimal Bundle Pinning Preflight

### Setup

- Date: 2026-06-28 UTC.
- Scope: package-path preflight only; no vLLM server launch and no public claim
  update.
- Dense minimal bundle:
  `qwen36-gfx906/overlays/gguf-dense/minimal-bundle/`.
- MoE minimal bundle:
  `qwen36-gfx906/overlays/gguf-moe/minimal-bundle/`.
- Bundle policy:
  - deterministic relative-path `sha256sum.txt`;
  - no self-inclusion of `sha256sum.txt`;
  - no absolute paths;
  - `sha256sum -c sha256sum.txt` must pass from the bundle root.

### Bundle Hashes

| Bundle | File count | Manifest SHA256 |
| --- | ---: | --- |
| Dense GGUF TP8 minimal bundle | `14` | `ce2c6f2d974de34c7abf4c25a67985b0eccc452a97f1a887ca97a8de1687f8d2` |
| MoE GGUF TP4 minimal bundle | `18` | `794cb8760003cfa7dafeb0f06ee01823e5e8eebb3f9d398d8bc8bd7697143f0b` |

The MoE bundle now carries its tuned C1 config under
`vllm_tuned_moe_configs/`, so the profile no longer depends on a host-only
scratch mount for `VLLM_TUNED_CONFIG_FOLDER`.

### Tests

Local pinned-bundle synthetic preflight:

- Dense GGUF TP8 generated and verified a launch artifact with the pinned dense
  manifest hash.
- MoE GGUF TP4 generated and verified a launch artifact with the pinned MoE
  manifest hash.

Real-model pinned-bundle preflight:

| Host | Profile | Model format | Detected family | Bundle hash verified | Artifact verifier |
| --- | --- | --- | --- | --- | --- |
| `.20` | `gguf-dense27b-tp8` | `gguf` | `qwen36_dense` | yes | valid |
| `.20` | `gguf-moe35b-tp4` | `gguf` | `qwen36_moe` | yes | valid |
| `.30` | `gguf-dense27b-tp8` | `gguf` | `qwen36_dense` | yes | valid |
| `.30` | `gguf-moe35b-tp4` | `gguf` | `qwen36_moe` | yes | valid |

The first host-side preflight draft used `MODEL` as the host-visible file path,
which would have generated a container command pointing at a host path. That was
rejected as a reproducibility bug. The launcher now separates runtime `MODEL`
from `MODEL_PROBE_PATH`: generated commands keep the container path under
`/opt/local-models/...`, while host-side probing reads the host-visible GGUF
file.

### Outcome

Promote:

- the pinned minimal-bundle layout as the next package-path candidate;
- keeping native runtime binaries out of the GGUF bundles because the release
  image already contains the referenced RCCL, sidecar, and SwiGLU runtime paths;
- embedding the MoE tuned config inside the MoE GGUF bundle.

Reject:

- the old scratch `sha256sum.txt` files as release manifests because they used
  absolute paths and included `sha256sum.txt` itself;
- relying on host-only tuned-config mounts for the MoE GGUF vNext profile;
- using the host probe path as the runtime model path in generated commands;
- treating preflight success as benchmark promotion.

### Reason

This closes the basic packaging-discipline gap from GGUF-264: the launcher can
now verify pinned, relative-path bundle manifests on `.20` and `.30` against
the real model files. The remaining promotion gate is runtime evidence from the
full benchmark ladder using these pinned bundles, followed by an HF baseline
rerun to prove the new GGUF bundle path does not contaminate the non-GGUF
release path.

## GGUF-266 - `.30` Dense GGUF vNext Package-Path Ladder

### Setup

- Date: 2026-06-28 UTC.
- Host label: `.30`.
- Profile: `gguf-dense27b-tp8`.
- Runtime profile: `dense27b_tp8_fullbar_p2pon`.
- Model format: GGUF, detected by the binary model probe.
- Model family: `qwen36_dense`.
- Tensor parallel degree: `8`.
- P2P state: on.
- `MAX_MODEL_LEN`: `131072`.
- Dtype: `half`.
- Image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Image digest:
  `sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e`.
- Patch bundle:
  `qwen36-gfx906/overlays/gguf-dense/minimal-bundle/`.
- Patch bundle manifest SHA256:
  `ce2c6f2d974de34c7abf4c25a67985b0eccc452a97f1a887ca97a8de1687f8d2`.
- Generated package run:
  `dense27b_pkg_dot30_20260628T174515Z`.
- Benchmark ladder:
  `8` warmups -> `c1_128` uncapped strict -> `c1_2000` -> `c1_10000`.

The package-path launch was generated by `vnext_repro_launcher.sh`, verified by
`verify_vnext_launch_artifact.sh`, then run through the generated
`docker_run.sh` and `benchmark_env.sh`. No production `EXTRA_VLLM_ARGS` were
used.

### Result

| Case | Completion tokens | Client TPS | Backend decode TPS | Finish | Strict gate valid |
| --- | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | `3445` | `55.639` | `70.163` | `stop` | true |
| `c1_2000` | `2000` | `49.983` | `70.975` | `length` | false |
| `c1_10000` | `10000` | `61.753` | `66.495` | `length` | false |

Warmup backend decode TPS after first-request startup settled around
`70.94` to `71.00`. The first warmup paid the slow package-path startup /
prefill cost but still decoded at `70.990` backend TPS.

### Outcome

Promote:

- the `.30` Dense GGUF generated package path as a successful runtime
  reproduction candidate;
- the pinned Dense minimal bundle as sufficient for a real server launch and
  benchmark ladder on `.30`;
- the generated launcher artifacts as capable of reproducing the standard
  benchmark sequence without hand-assembled `EXTRA_VLLM_ARGS`;
- the Dense GGUF package path as clearing the published Dense c1_10000
  threshold (`66.495` versus published `66.069`).

Reject:

- treating this single `.30` Dense package-path run as full vNext release
  readiness by itself;
- treating the earlier scratch-overlay best `.30` Dense result as already
  reproduced by the reduced package path, because this package run is lower
  than the earlier best scratch-lane result while still clearing the published
  target.

### Reason

This run proves the contract-driven launcher can move beyond preflight and
start a real vLLM server from the reduced Dense bundle, preserve P2P-on,
preserve `MAX_MODEL_LEN=131072`, keep the generated benchmark ladder intact,
and produce a strict-valid Dense result on `.30`. The remaining release
readiness work is to run the MoE package path, rerun the clean package path on
`.20` and `.30` as needed, and rerun HF baselines to prove GGUF overlays do not
contaminate the published non-GGUF path.

## GGUF-267 - `.30` MoE GGUF vNext Package-Path Ladder

### Setup

- Date: 2026-06-28 UTC.
- Host label: `.30`.
- Profile: `gguf-moe35b-tp4`.
- Runtime profile: `moe35b_tp4_fullbar_p2pon`.
- Model format: GGUF, detected by the binary model probe.
- Model family: `qwen36_moe`.
- Tensor parallel degree: `4`.
- P2P state: on.
- `MAX_MODEL_LEN`: `131072`.
- Dtype: `half`.
- Image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Image digest:
  `sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e`.
- Patch bundle:
  `qwen36-gfx906/overlays/gguf-moe/minimal-bundle/`.
- Patch bundle manifest SHA256:
  `794cb8760003cfa7dafeb0f06ee01823e5e8eebb3f9d398d8bc8bd7697143f0b`.
- Generated package run:
  `moe35b_pkg_dot30_20260628T180700Z`.
- Benchmark ladder:
  `8` warmups -> `c1_128` uncapped strict -> `c1_2000` -> `c1_10000`.

The first package-path launch reached the GGUF loader and then failed before
serving because the installed Transformers side did not accept the GGUF
architecture string for the Qwen3.6 MoE package. The sidecar
`modeling_gguf_pytorch_utils.py` file was present in the minimal bundle, but
`PYTHONPATH=/opt/qwen36-python` was not sufficient to override the installed
Transformers module. The launcher and release deploy helper now copy that
sidecar directly into the installed Transformers paths before vLLM starts.

After regenerating the launch artifact with that fix, the container resolved
`Qwen3_5MoeForCausalLM`, loaded the GGUF package, completed graph capture, and
served from the generated `docker_run.sh` path. No production
`EXTRA_VLLM_ARGS` were used.

### Result

| Case | Completion tokens | Client TPS | Backend decode TPS | Finish | Strict gate valid |
| --- | ---: | ---: | ---: | --- | --- |
| `c1_128_strict` | `3786` | `118.307` | `119.508` | `stop` | true |
| `c1_2000` | `2000` | `118.452` | `120.758` | `length` | false |
| `c1_10000` | `10000` | `112.858` | `113.271` | `length` | false |

Warmups after the first compile-heavy request settled around `120.74` to
`120.81` backend decode TPS. The first warmup paid a large prefill / compile
cost but decoded at `117.262` backend TPS.

### Outcome

Promote:

- the `.30` MoE GGUF generated package path as a successful runtime
  reproduction candidate;
- the pinned MoE minimal bundle as sufficient for a real server launch and
  benchmark ladder on `.30`;
- the direct installed-Transformers sidecar copy as required for the GGUF MoE
  package path;
- the generated launcher artifacts as capable of reproducing the standard
  MoE benchmark sequence without hand-assembled `EXTRA_VLLM_ARGS`;
- the MoE GGUF package path as beating the native/current TP4 c1_10000 band
  (`113.271` versus the post-v0.2 repeatability `.30` fixed sanity value
  `108.950` and the v0.2 release-time capped value `109.283`).

Reject:

- relying on `PYTHONPATH=/opt/qwen36-python` alone to override the installed
  Transformers GGUF helper;
- treating architecture resolution as a sufficient benchmark result before
  graph capture and the full ladder complete;
- treating this single `.30` MoE package-path run as full vNext release
  readiness by itself.

### Reason

This run proves the contract-driven launcher can start the MoE GGUF package
path from the reduced bundle, preserve P2P-on, preserve
`MAX_MODEL_LEN=131072`, keep the standard warmups -> strict -> fixed-token
ladder intact, and reproduce the high-band MoE TP4 GGUF result on `.30`.
The remaining release-readiness work is to rerun package-path lanes as needed
on `.20` and `.30`, verify the HF baseline profiles still reproduce current
release bands, and keep mismatch tests failing closed.

## GGUF-268 - HF Dense vNext Baseline Protection Reruns

### Setup

- Date: 2026-06-28 UTC.
- Profile: `hf-dense27b-tp8`.
- Runtime profile: `dense27b_tp8_fullbar_p2pon`.
- Model format: HF safetensors, detected by the binary model probe.
- Model family: `qwen36_dense`.
- Tensor parallel degree: `8`.
- P2P state: on.
- `MAX_MODEL_LEN`: `131072`.
- Dtype: `half`.
- Image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Image digest:
  `sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e`.
- Benchmark ladder:
  `8` warmups -> `c1_128` uncapped strict -> `c1_2000` -> `c1_10000`.

The first vNext HF Dense profile used the release image, CausalLM override,
P2P-on, and the standard generated benchmark ladder, but it did not mount an
explicit HF release overlay bundle. It was valid as a launch artifact and had
no GGUF env leakage, but it did not reproduce the historical text-only HF
comparator band.

The profile was then corrected to:

- force `Qwen3_5ForCausalLM` with `--language-model-only` and
  `--hf-overrides`;
- include the dense release env set such as balanced `RCCL_TREES`, persistent
  all-reduce routing, RowParallel boundary settings, and native SwiGLU path;
- require an isolated HF release patch bundle under
  `qwen36-gfx906/overlays/hf/minimal-bundle/`;
- pin the HF bundle manifest hash:
  `9fb16c0edfd57d908f2bff6eb51063b6e8cf7d2c27de252bb4f691c67d2f5a84`;
- keep GGUF loader/model/quantization patches out of the HF launch.

One `.20` host-side cache path exposed only HF metadata from the host view and
was rejected as a `MODEL_PROBE_PATH`; the rerun used the complete HF snapshot
path so the binary probe could inspect actual safetensors bytes. This is a
model-probe input lesson, not a runtime result.

### Results

| Host | Launch shape | Strict backend TPS | `c1_2000` backend TPS | `c1_10000` backend TPS | Strict valid | Outcome |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `.30` | CausalLM, no explicit HF bundle | `59.988` | `62.853` | `54.294` | true | reject |
| `.30` | CausalLM, pinned HF bundle | `63.232` | `65.051` | `55.985` | true | reject |
| `.20` | CausalLM, pinned HF bundle, complete snapshot mount | `62.962` | `64.991` | `55.945` | true | reject |

The pinned HF bundle was active: startup logs showed persistent all-reduce
route replacement under the generated launcher path. The artifact verifier and
runtime vLLM arg-schema verifier both passed, and intentional HF/GGUF
format-mismatch tests failed closed locally.

### Promote / Reject

Promote:

- the CausalLM / text-only / HF bundle shape as the correct direction for HF
  baseline protection;
- the isolated HF release bundle as required for generated HF launches;
- the local guardrails: binary probe, launch artifact verifier, arg-schema
  verifier, and mismatch fail-closed behavior;
- using a complete byte-visible HF snapshot for `MODEL_PROBE_PATH`.

Reject:

- the no-bundle HF generated path as a release-baseline reproduction path;
- the pinned HF bundle path as release-ready baseline protection until it
  reproduces the historical text-only comparator band;
- treating launch validity or overlay activation as performance reproduction;
- treating the GGUF package-path success as proof that the HF baseline is still
  protected.

### Reason

The historical `.20` text-only HF comparator remains the benchmark reference
for this control: strict `70.353`, `c1_2000` `70.978`, and `c1_10000`
`66.428` backend TPS. The current generated HF package path is strict-valid and
overlay-active, but it remains materially below that comparator, especially on
`c1_10000`.

This keeps the vNext release-readiness gate open. The GGUF Dense and MoE
package paths have promoted as runtime candidates, but the non-GGUF HF
baseline protection requirement is not yet satisfied. Next work should compare
the old text-only comparator environment and current generated HF bundle path
at the runtime/env level, then rerun the HF control before public release
claims change.

## GGUF-269 - HF Dense vNext Baseline Protection Recovery

### Setup

- Date: 2026-06-28 UTC.
- Profile: `hf-dense27b-tp8`.
- Runtime profile: `dense27b_tp8_fullbar_p2pon`.
- Model format: HF safetensors, detected by the binary model probe.
- Model family: `qwen36_dense`.
- Tensor parallel degree: `8`.
- P2P state: on.
- `MAX_MODEL_LEN`: `131072`.
- Dtype: `half`.
- Benchmark ladder:
  `8` warmups -> `c1_128` uncapped strict -> `c1_2000` -> `c1_10000`.

The old `.20` text-only comparator was inspected through its retained Docker
metadata and logs. It used the published runtime image plus the broader clean
release patch bundle, not the first-pass reduced HF bundle. The reduced HF
bundle from GGUF-268 had omitted release-path files such as model, platform,
attention, utility, and MoE fastpath overlays. Even though the Dense path does
not use the MoE fastpath as its promoted kernel, the omission was enough to
make the generated HF package path non-equivalent to the historical release
control.

The HF bundle under `qwen36-gfx906/overlays/hf/minimal-bundle/` was therefore
expanded to mirror the old clean release patch-bundle composition. Its new
manifest hash is:

`d9e1168f67731ab66ee53671f89dfb31fd3c74c8f61e2f0d0b7c4f0704c298be`

### Results

Two `.20` reruns used the generated vNext launch artifact, the expanded HF
bundle, the old served-model name, normal warmups, and the standard benchmark
ladder:

| Host | Cache mode | Strict backend TPS | `c1_2000` backend TPS | `c1_10000` backend TPS | Strict valid | Outcome |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `.20` | shared release runtime cache | `70.346` | `70.961` | `66.428` | true | promote |
| `.20` | fresh per-run runtime cache | `70.168` | `71.316` | `66.824` | true | promote |

The shared-cache rerun exactly matched the historical c1_10000 comparator
band. The fresh-cache rerun paid a large first-warmup compile/prefill cost, but
the decode path still settled into the high-band result during the required
warmup ladder. This shows that the expanded release bundle, not a preseeded
cache, was the missing HF baseline requirement.

### Promote / Reject

Promote:

- the expanded clean HF release bundle as the required HF baseline bundle for
  vNext package-path reproduction;
- the new HF bundle manifest hash
  `d9e1168f67731ab66ee53671f89dfb31fd3c74c8f61e2f0d0b7c4f0704c298be`;
- the generated HF Dense package path as baseline-protected on `.20` after the
  expanded bundle correction;
- fresh per-run runtime caches as acceptable for decode TPS after the normal
  warmup ladder, while noting the first warmup can pay compile/prefill cost.

Reject:

- the first-pass reduced HF bundle as too small for HF baseline protection;
- treating persistent all-reduce route replacement alone as proof of HF
  release equivalence;
- treating shared runtime-cache reuse as required for the HF decode TPS claim.

### Reason

The objective requires GGUF support without breaking the original non-GGUF HF
release path. The earlier reduced HF bundle kept GGUF env leakage out of HF
launches, but it did not reproduce the old Dense HF band. Mirroring the clean
release patch-bundle composition restored the historical HF numbers while
preserving the contract launcher guardrails: binary model probing, profile
validation, isolated overlay selection, generated argv, runtime arg-schema
verification, and mismatch fail-closed behavior.

## GGUF-270 - HF MoE Contract Pinning and Developer Inspection

### Setup

- Date: 2026-06-28 UTC.
- Scope: local contract validation only; no vLLM server launch and no public
  release-claim update.
- Profiles updated:
  - `hf-moe35b-tp4`
  - `hf-moe35b-tp8`
- Shared HF release bundle manifest:
  `d9e1168f67731ab66ee53671f89dfb31fd3c74c8f61e2f0d0b7c4f0704c298be`.
- New helper: `qwen36-gfx906/vnext_profile_inspect.sh`.

### Tests

Local synthetic model fixtures were used so profile validation could run
without host model paths:

- HF Dense config plus safetensors bytes generated a valid `hf-dense27b-tp8`
  launch artifact.
- HF MoE config plus safetensors bytes generated valid `hf-moe35b-tp4` and
  `hf-moe35b-tp8` launch artifacts.
- GGUF Dense and MoE byte headers generated valid `gguf-dense27b-tp8` and
  `gguf-moe35b-tp4` launch artifacts.
- `verify_vnext_launch_artifact.sh` passed for all five profiles.
- `vnext_profile_inspect.sh list` and `vnext_profile_inspect.sh show
  hf-moe35b-tp4` ran without starting Docker or mutating runtime state.

### Promote / Reject

Promote:

- requiring the expanded clean HF release bundle for HF MoE TP4/TP8, matching
  the HF Dense baseline-protection bundle;
- the read-only profile inspector as a developer-serviceability aid;
- local synthetic preflight as a cheap guardrail for profile/format/overlay
  regressions before long host runs.

Reject:

- treating synthetic preflight as a runtime MoE HF reproduction result;
- unpinned HF MoE patch-bundle selection;
- any design where the image becomes useful only through hidden benchmark
  harness assumptions.

### Reason

The vNext objective protects the original non-GGUF release models as well as
the new GGUF paths. Leaving HF MoE TP4/TP8 with `PATCH_BUNDLE_REQUIRED=0` made
those contracts weaker than HF Dense and left too much runtime state implicit.
Pinning both HF MoE profiles to the expanded HF release bundle keeps the
non-GGUF path explicit, inspectable, and isolated from GGUF-only env or source
files.

The new inspector supports the serviceability requirement: developers can list
and inspect supported contracts, image tags/digests, overlays, TP degree, P2P
requirements, and patch-bundle hashes before running the launcher or entering
the container.

## GGUF-271 - Developer Serviceability Gate and HF MoE TP4 Runtime Rerun

### Setup

- Date: 2026-06-28 UTC.
- Host lane: `.30`.
- Profile: `hf-moe35b-tp4`.
- Runtime image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Runtime digest:
  `sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e`.
- Shared HF release bundle manifest:
  `d9e1168f67731ab66ee53671f89dfb31fd3c74c8f61e2f0d0b7c4f0704c298be`.
- New helper: `qwen36-gfx906/verify_vnext_serviceability.sh`.

### Serviceability Gate

`verify_vnext_serviceability.sh` is a read-only developer gate. It builds only
the local C model-format probe and verifies:

- all five expected vNext profiles are present;
- each profile pins the public runtime image tag and digest;
- `P2P_REQUIRED=1` and `NCCL_P2P_DISABLE=0`;
- `MAX_MODEL_LEN=131072` and `VLLM_DTYPE=half`;
- HF profiles do not leak GGUF-only environment variables;
- pinned patch-bundle manifests verify;
- developer-facing vNext files do not contain private host/path references.

It does not start Docker, patch a host, inspect live GPUs, mutate model caches,
or run benchmarks. The check passed for:

- `gguf-dense27b-tp8`
- `gguf-moe35b-tp4`
- `hf-dense27b-tp8`
- `hf-moe35b-tp4`
- `hf-moe35b-tp8`

### Runtime Reruns

First HF MoE TP4 runtime rerun used the expanded clean HF bundle before the
tuned MoE config was added to the bundle. It completed but stayed in the
low-performance band:

| Variant | Strict backend TPS | `c1_2000` backend TPS | `c1_10000` backend TPS | Strict valid | Outcome |
| --- | ---: | ---: | ---: | --- | --- |
| expanded HF bundle without tuned MoE config | `95.824` | `96.158` | `91.524` | true | reject |

The HF bundle was then corrected to include the tuned MoE config used by the
release MoE path, keeping the same manifest hash family now pinned by the HF
profiles:

`d9e1168f67731ab66ee53671f89dfb31fd3c74c8f61e2f0d0b7c4f0704c298be`

The corrected HF MoE TP4 rerun used the generated launch artifact, normal
warmups, uncapped strict, `c1_2000`, and `c1_10000`. Startup reached readiness
after long graph capture and first-warmup prefill/compile work. The run then
settled into a stable low band:

| Variant | Strict backend TPS | `c1_2000` backend TPS | `c1_10000` backend TPS | Strict valid | Outcome |
| --- | ---: | ---: | ---: | --- | --- |
| expanded HF bundle with tuned MoE config | `96.159` | `97.031` | `92.284` | true | reject |

The strict run finished with `finish_reason=stop` and `qwen_gate_valid=true`,
so the corrected profile is functionally valid. It does not reproduce the
published TP4 fixed-token band and must not be promoted as the exact release
MoE TP4 reproduction path.

### Promote / Reject

Promote:

- `verify_vnext_serviceability.sh` as a required local gate before sharing or
  publishing a vNext package path;
- profile inspection plus patch-bundle manifest verification as the way to
  keep the image serviceable for other developers;
- the corrected HF MoE TP4 profile as a functionally valid strict path.

Reject:

- the HF MoE TP4 vNext generated package path as release-performance-equivalent
  to the published TP4 capped fixed-token result;
- the low-band no-tuned-config run;
- assuming that the expanded HF bundle plus tuned MoE config fully reconstructs
  the exact historical TP4 release-performance lane.

### Reason

The serviceability problem and the reproduction problem are separate. The
image/profile contract is now inspectable and verifiable without hidden lab
state, which makes it serviceable for developers. However, the HF MoE TP4
runtime result is still materially below the published `c1_10000` TP4 band.
The next source/repro task is to recover the exact release MoE TP4 performance
settings rather than promote a clean but slower contract path.

## GGUF-272 - HF MoE TP4 Deploy-Shaped vNext Runtime Rejected

### Setup

- Date: 2026-06-28 UTC.
- Host lane: `.30`.
- Profile: `hf-moe35b-tp4`.
- Runtime image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Runtime digest:
  `sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e`.
- Shared HF release bundle manifest:
  `d9e1168f67731ab66ee53671f89dfb31fd3c74c8f61e2f0d0b7c4f0704c298be`.

### Hypothesis

The earlier vNext HF MoE TP4 generated path might have been low-band because
the generated Docker wrapper was not close enough to the `deploy.sh` runtime
shape. The launcher was updated to preserve deploy-style generic runtime
defaults:

- `VLLM_TARGET_DEVICE=rocm`;
- `VLLM_VERSION_OVERRIDE=0.0.0+gfx906`;
- `PYTORCH_ROCM_ARCH=gfx906`;
- `GPU_ARCHS=gfx906`;
- `OMP_NUM_THREADS=4`;
- `TORCH_BLAS_PREFER_HIPBLASLT=0`;
- `DO_NOT_TRACK=1`;
- `VLLM_LOGGING_LEVEL=INFO`;
- privileged container mode.

The vLLM command remained profile-generated and still did not use production
`EXTRA_VLLM_ARGS`.

### Result

The fresh deploy-shaped generated artifact passed `verify_vnext_launch_artifact.sh`
and the staged tree passed `verify_vnext_serviceability.sh` on `.30`. The
container reached readiness after normal compile/KV-cache/graph-capture work,
the MoE fastpath overlay loaded, and the normal ladder ran:

| Variant | Strict backend TPS | `c1_2000` backend TPS | `c1_10000` backend TPS | Strict valid | Outcome |
| --- | ---: | ---: | ---: | --- | --- |
| deploy-shaped vNext HF MoE TP4 wrapper | `94.860` | `95.613` | `91.036` | true | reject |

The strict run finished with `finish_reason=stop` and `qwen_gate_valid=true`.
The run is functionally correct but remains below the published HF MoE TP4
fixed-token release band.

### Promote / Reject

Promote:

- deploy-shaped generic runtime defaults in the vNext wrapper as closer to the
  public deploy path and better for developer serviceability;
- the updated artifact verifier checks for those generic runtime defaults;
- the finding that the wrapper/runtime-default delta alone does not explain
  the HF MoE TP4 performance gap.

Reject:

- the deploy-shaped vNext HF MoE TP4 generated path as release-performance
  equivalent;
- repeating wrapper/privileged/default-env changes as the next hypothesis
  without a new concrete delta from the full deploy-package path.

### Reason

This closes another likely reproduction gap. The high-band HF MoE TP4 control
still appears tied to something in the full deploy-package/runtime-patch path
that is not captured by the current vNext generated artifact, even after
matching generic runtime defaults and privileged container mode. The remaining
work should compare the generated artifact against the full deploy package at
the copied source/runtime file level or run the full `deploy.sh` path as the
source of truth and mechanically derive a profile contract from its emitted
runtime env and compose file.

## GGUF-273 - Public Image Serviceability Check

### Setup

- Date: 2026-06-28 UTC.
- Scope: developer serviceability, not benchmark promotion.
- Runtime image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Runtime digest:
  `sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e`.
- Profiles checked:
  - `gguf-dense27b-tp8`
  - `gguf-moe35b-tp4`
  - `hf-dense27b-tp8`
  - `hf-moe35b-tp4`
  - `hf-moe35b-tp8`

### Result

The local read-only serviceability gate passed for all five profiles. A clean
Docker pull of the public runtime image resolved to the pinned manifest digest:

`sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e`

The pulled public image also contained the baseline native runtime paths used
by the profile contracts:

- `/opt/gfx906/libgfx906_persistent_tree_ll_ar_default_20260613.so`
- `<validation-nvme-root>/kernel_labs/gfx906_swiglu_gemv_ext_native_runtime_20260608/gfx906_swiglu_gemv_ext_20260607.so`
- `/rccl-overlay/install/lib/librccl.so.1`

`verify_vnext_serviceability.sh` was updated with opt-in Docker checks:

```sh
CHECK_RUNTIME_IMAGE=1 CHECK_RUNTIME_PATHS=1 ./verify_vnext_serviceability.sh
```

The default check remains lightweight and does not start Docker, touch GPUs,
mutate model caches, or run benchmarks. The opt-in mode verifies that the
public Docker tag resolves to the pinned digest and that the baseline native
runtime paths exist inside the public image.

### Promote / Reject

Promote:

- the public image tag and digest as serviceable for clean developers;
- the optional Docker manifest/native-path check as a release-sharing gate;
- keeping serviceability validation separate from performance promotion.

Reject:

- relying on locally renamed or experimental images;
- relying on hidden native library paths that are not visible in the public
  image or copied by the generated launcher;
- treating image serviceability as proof that every profile has recovered its
  target benchmark band.

### Reason

The image can be serviced by another developer because the public tag resolves
to the pinned digest and the baseline native runtime dependencies referenced by
the profiles are present in the clean image. This closes the image availability
and native-path concern. It does not close the HF MoE TP4 performance gap; that
remains a separate release-readiness problem requiring derivation from the full
`deploy.sh` emitted runtime package.

## GGUF-274 - vNext Contract Matrix Fail-Closed Gate

### Setup

- Date: 2026-06-28 UTC.
- Scope: launcher/profile contract validation, not runtime benchmarking.
- New helper: `qwen36-gfx906/verify_vnext_contract_matrix.sh`.
- Profiles checked:
  - `gguf-dense27b-tp8`
  - `gguf-moe35b-tp4`
  - `hf-dense27b-tp8`
  - `hf-moe35b-tp4`
  - `hf-moe35b-tp8`

### Result

The verifier builds the C model-format probe and creates tiny synthetic model
signatures instead of requiring full model weights:

- GGUF files begin with the `GGUF` magic bytes and include bounded family
  marker text.
- HF directories include `config.json` plus a minimal safetensors byte
  signature with a valid little-endian header length and JSON tensor metadata.
- A mixed GGUF+safetensors directory is expected to fail as a conflict.

Valid generated artifacts passed for all five profiles:

- `gguf-dense27b-tp8`
- `gguf-moe35b-tp4`
- `hf-dense27b-tp8`
- `hf-moe35b-tp4`
- `hf-moe35b-tp8`

Synthetic deploy/profile parity checks also passed for:

- `hf-moe35b-tp4`
- `hf-moe35b-tp8`

Expected failures also failed closed:

- GGUF model bytes under an HF profile.
- HF/safetensors model bytes under a GGUF profile.
- Dense GGUF model bytes under a MoE GGUF profile.
- MoE GGUF model bytes under a Dense GGUF profile.
- GGUF-only environment leakage into an HF profile.
- Patch-bundle manifest hash mismatch.

The command completed with:

```text
vNext contract matrix passed
```

### Promote / Reject

Promote:

- `verify_vnext_contract_matrix.sh` as the cheap local gate for launcher and
  profile logic changes;
- deploy/profile parity checks inside the matrix for HF MoE TP4/TP8;
- synthetic binary signatures as enough evidence for format/family routing and
  fail-closed contract behavior;
- keeping this gate separate from GPU runtime benchmarking.

Reject:

- using full model weights or live GPUs to test simple profile-contract
  regressions;
- relying only on serviceability metadata checks to prove launcher behavior;
- accepting profile changes that cannot pass both valid generation and expected
  mismatch failures.

### Reason

The contract matrix covers the failure modes that would make GGUF support
dangerous to the original HF release path: overlay bleed, format mismatch,
family mismatch, leaked GGUF envs, and unpinned or mismatched patch bundles.
This makes the launcher safer for developers before long benchmark runs begin.
It does not prove release TPS reproduction; benchmark promotion still requires
clean `.20` and `.30` package-path runs.

## GGUF-275 - Deploy-Style Tuned MoE Config Path and Parity Gate

### Setup

- Date: 2026-06-28 UTC.
- Scope: vNext/deploy contract parity, not GPU runtime benchmarking.
- Profiles touched:
  - `gguf-moe35b-tp4`
  - `hf-moe35b-tp4`
  - `hf-moe35b-tp8`
- New helper: `qwen36-gfx906/verify_vnext_deploy_profile_parity.sh`.

### Result

The checked-in `deploy.sh` runtime path mounts tuned MoE configs at:

`/opt/vllm_tuned_moe_configs`

The vNext MoE profiles previously pointed `VLLM_TUNED_CONFIG_FOLDER` inside
the patch bundle:

`/opt/vllm_patch_bundle/vllm_tuned_moe_configs`

The MoE vNext profiles were aligned to the deploy-style path, and the generated
container entrypoint now copies bundled tuned MoE configs from:

`$VLLM_PATCH_BUNDLE/vllm_tuned_moe_configs`

to:

`/opt/vllm_tuned_moe_configs`

before vLLM starts.

`verify_vnext_deploy_profile_parity.sh` was added to compare a captured
`.deploy.runtime.env` against a selected vNext profile and optional generated
launch artifact. It checks:

- profile name;
- TP degree;
- `MAX_MODEL_LEN`;
- dtype;
- async-scheduling toggle;
- P2P/NCCL settings;
- MoE fastpath flag;
- deploy-style tuned-config path;
- `--language-model-only` mapping;
- generated argv shape.

A synthetic HF MoE TP4 deploy-env comparison passed against a generated
`hf-moe35b-tp4` vNext artifact:

```text
vNext deploy/profile parity passed
```

### Promote / Reject

Promote:

- `/opt/vllm_tuned_moe_configs` as the vNext MoE tuned-config path to match
  `deploy.sh`;
- copying tuned configs from the mounted patch bundle into that path at
  container-entrypoint time;
- the deploy/profile parity checker as the mechanical comparison gate for the
  remaining HF MoE high-band derivation work.

Reject:

- keeping tuned MoE configs hidden under a vNext-specific patch-bundle path
  when the published deploy package uses `/opt/vllm_tuned_moe_configs`;
- manual grep-based deploy/vNext comparisons as the only evidence for future
  high-band profile changes;
- treating this contract-path correction as proof of recovered HF MoE TP4
  performance before a real runtime rerun.

### Reason

This removes one concrete deploy/vNext runtime-shape delta and gives future
HF MoE reruns a stronger contract comparison against captured deploy artifacts.
It is a release-readiness improvement, not a benchmark promotion. The HF MoE
TP4 path still needs a clean runtime rerun from the updated vNext artifact
before the performance gap can be closed.

## GGUF-276 - HF MoE TP4 Tuned-Config Path Runtime Rerun

### Setup

- Date: 2026-06-28 UTC.
- Host lane: `.30`.
- Profile: `hf-moe35b-tp4`.
- Runtime contract:
  - HF model package;
  - TP4;
  - FP16;
  - P2P-on;
  - `MAX_MODEL_LEN=131072`;
  - expanded clean HF bundle manifest
    `d9e1168f67731ab66ee53671f89dfb31fd3c74c8f61e2f0d0b7c4f0704c298be`;
  - deploy-style tuned MoE config path `/opt/vllm_tuned_moe_configs`.
- Benchmark ladder: normal warmups, then `c1_128` uncapped strict, `c1_2000`,
  and `c1_10000`.

### Result

The service reached readiness and completed the normal benchmark ladder. The
strict run finished normally and passed the Qwen gate, but decode performance
remained in the low band:

| case | backend decode TPS | finish | strict gate valid |
| --- | ---: | --- | --- |
| `c1_128_strict` | `93.013` | stop | true |
| `c1_2000` | `93.839` | length | false |
| `c1_10000` | `89.312` | length | false |

### Promote / Reject

Promote:

- deploy-style tuned-config path alignment as the correct runtime-shape
  contract for MoE profiles;
- the generated artifact and benchmark ladder as functional.

Reject:

- tuned-config path alignment alone as the missing high-band TP4 performance
  factor;
- treating this HF MoE TP4 vNext path as release-performance-equivalent.

### Reason

The run was functionally valid, and the strict request stopped cleanly, but the
published high-band TP4 repeatability result is around the `113` to `116`
backend TPS strict range. This rerun stayed near the TP8-like low band. The
next experiment needed to compare more of the deploy runtime wrapper surface,
not rerun the same tuned-config-only package.

## GGUF-277 - HF MoE TP4 Deploy-Compatibility Runtime Patch Rerun

### Setup

- Date: 2026-06-28 UTC.
- Host lane: `.30`.
- Profile: `hf-moe35b-tp4`.
- Runtime contract:
  - same HF model package and profile as GGUF-276;
  - deploy-style tuned MoE config path;
  - expanded clean HF bundle manifest
    `d9e1168f67731ab66ee53671f89dfb31fd3c74c8f61e2f0d0b7c4f0704c298be`;
  - generated `deploy_compat_patches.py` entrypoint step mirroring deploy-time
    runtime compatibility patches for missing custom-op surfaces, ROCm/GFX906
    capability, batch-invariant compatibility, fused-MoE utilities,
    activation/MoE activation/RMSNorm/rotary fallbacks, and missing-op
    reporting.
- Benchmark ladder: normal warmups, then `c1_128` uncapped strict, `c1_2000`,
  and `c1_10000`.

### Result

The generated launch artifact validated, the container reached readiness, and
the normal ladder completed. The eight warmups settled around `93.25` backend
decode TPS. The strict request was valid, but the final decode results remained
low-band:

| case | backend decode TPS | finish | strict gate valid |
| --- | ---: | --- | --- |
| `c1_128_strict` | `92.731` | stop | true |
| `c1_2000` | `93.272` | length | false |
| `c1_10000` | `88.832` | length | false |

The test container was stopped and removed after capture.

### Promote / Reject

Promote:

- the generated deploy-compat patch artifact as a serviceability/parity
  guardrail when vNext needs to mimic deploy-time runtime patch surfaces;
- the result as negative evidence that the compatibility patch surface by
  itself is not the missing TP4 high-band lever.

Reject:

- deploy-compat runtime patching as sufficient to recover published HF MoE TP4
  throughput;
- further one-off wrapper approximation without deriving the full contract
  from the exact `deploy.sh` emitted runtime env and package artifacts.

### Reason

This experiment removed another concrete deploy/vNext delta, but performance
did not move meaningfully from GGUF-276. The strict run remained correct while
throughput stayed low. The most useful next step is not another small wrapper
edit; it is to capture the full known-good deploy environment and diff it
mechanically against the generated vNext launch artifact.

## GGUF-278 - Profile-Gated Patch Target Activation

### Setup

- Date: 2026-06-28 UTC.
- Scope: launcher/source contract; no GPU runtime benchmark.
- Affected profiles:
  - `gguf-dense27b-tp8`
  - `gguf-moe35b-tp4`
  - `hf-dense27b-tp8`
  - `hf-moe35b-tp4`
  - `hf-moe35b-tp8`

### Result

The vNext profiles now declare `PATCH_TARGET_GROUPS`, and the generated
container entrypoint only copies patch files whose group is allowed by the
profile. The current groups are:

| profile | patch target groups |
| --- | --- |
| `gguf-dense27b-tp8` | `common dense gguf` |
| `gguf-moe35b-tp4` | `common moe gguf` |
| `hf-dense27b-tp8` | `common dense hf` |
| `hf-moe35b-tp4` | `common moe hf` |
| `hf-moe35b-tp8` | `common moe hf moe_tp8` |

The generated artifact verifier now checks that the group set is pinned into
the entrypoint, that dense native sidecar/SWiGLU copies are group-gated, that
MoE tuned-config and fused-MoE copies are group-gated, and that GGUF loader
copy targets are labeled as GGUF-only.

### Promote / Reject

Promote:

- profile-gated patch target activation as the correct way to keep broad
  bundles serviceable without overlay bleed;
- verifier coverage for generated entrypoint group labels and unresolved
  placeholders.

Reject:

- using file presence inside a patch bundle as sufficient evidence that the
  file should be copied for every profile;
- broad HF bundle activation without a profile-specific copy filter.

### Reason

The HF bundle currently contains files needed by different HF lanes. The
previous generated entrypoint copied every recognized file if it was present,
which allowed dense-only persistent-AR/SWiGLU targets to activate in HF MoE
launches. That is a profile isolation bug even when the command-line and env
contract look correct. This source fix improves serviceability and creates a
better next HF MoE TP4 runtime candidate, but it is not a benchmark promotion
until a clean `.20`/`.30` rerun recovers the high-band TP4 numbers.

## GGUF-279 - HF MoE TP4 Group-Gated Runtime Promotion on `.20` / `.30`

### Setup

- Date: 2026-06-29 UTC.
- Host lanes: `.20` and `.30`.
- Profile: `hf-moe35b-tp4`.
- Runtime contract:
  - HF model package;
  - TP4;
  - FP16;
  - P2P-on;
  - `MAX_MODEL_LEN=131072`;
  - expanded clean HF bundle manifest
    `d9e1168f67731ab66ee53671f89dfb31fd3c74c8f61e2f0d0b7c4f0704c298be`;
  - deploy-style tuned MoE config path `/opt/vllm_tuned_moe_configs`;
  - `PATCH_TARGET_GROUPS="common moe hf"`.
- Benchmark ladder: normal warmups, then `c1_128` uncapped strict, `c1_2000`,
  and `c1_10000`.

### Result

The generated artifact validated and pinned `patch_target_groups='common moe hf'`.
The entrypoint still carried dense-only target rows for developer-serviceable
broad bundles, but those rows were gated behind the `dense` target group and
therefore did not activate for the HF MoE TP4 run.

Both servers completed graph capture and the normal benchmark ladder. Warmups
settled around `115.7` to `116.0` backend decode TPS after the first
cache/prefill pass.

| host | case | backend decode TPS | finish | strict gate valid |
| --- | --- | ---: | --- | --- |
| `.20` | `c1_128_strict` | `115.105` | stop | true |
| `.20` | `c1_2000` | `115.933` | length | false |
| `.20` | `c1_10000` | `109.095` | length | false |
| `.30` | `c1_128_strict` | `114.409` | stop | true |
| `.30` | `c1_2000` | `115.685` | length | false |
| `.30` | `c1_10000` | `108.918` | length | false |

The test containers were stopped and removed after capture, and VRAM returned
to idle on both lanes.

### Promote / Reject

Promote:

- profile-gated patch target activation as the missing HF MoE TP4 high-band
  source-contract fix on `.20` and `.30`;
- `PATCH_TARGET_GROUPS="common moe hf"` for `hf-moe35b-tp4`;
- the generated vNext HF MoE TP4 package path as performance-equivalent on
  `.20` and `.30` to the post-v0.2 TP4 repeatability lane.

Reject:

- the earlier broad-bundle entrypoint that copied dense-only targets into HF
  MoE launches;
- the hypothesis that deploy-compat runtime patching alone was the missing
  high-band factor.

### Reason

The only meaningful launcher/source change between GGUF-277 and this run was
profile-gated patch activation. The benchmark moved from the low `88` to `93`
backend TPS band back to the high `108` to `115` band on `.30`, and reproduced
the same high band on `.20`. That ties the previous HF MoE TP4 performance gap
to overlay activation bleed, especially dense-only targets that should not have
been copied into a MoE TP4 launch. This promotes the generated HF MoE TP4
profile-contract path across the two primary lanes. Remaining release-readiness
work is to keep the HF Dense and GGUF lanes green under the same current
contract set.

## GGUF-280 - vNext Host-Path Preflight Mapping Fix

### Setup

- Date: 2026-06-29 UTC.
- Scope: launcher/source contract; no GPU runtime benchmark.
- Affected launcher: `qwen36-gfx906/vnext_repro_launcher.sh`.
- Triggering profile: `gguf-moe35b-tp4`.

### Result

The MoE GGUF profile initially failed host-side preflight because profile
contracts correctly referenced container paths for tokenizer/config and
patch-bundle files:

- `<container-hf-cache>/...`
- `/opt/vllm_patch_bundle/...`

Those paths are valid after the generated Docker wrapper mounts the NVMe model
root, Hugging Face cache, and patch bundle, but they do not exist as literal
host paths before launch. The launcher now maps container-required paths back
to the host-visible mount roots during preflight:

- `<container-hf-cache>/...` -> `HOST_HF_CACHE/...`
- `/opt/vllm_patch_bundle/...` -> `PATCH_BUNDLE_PATH/...`

### Promote / Reject

Promote:

- container-to-host required-path mapping as a serviceability fix for generated
  launch artifacts;
- keeping tokenizer/config paths in profile contracts as container paths,
  because those are the paths vLLM actually sees at runtime.

Reject:

- turning container runtime paths into host-local defaults inside profiles;
- disabling all required-path checks just because one path is container-scoped.

### Reason

The correct contract has two different path views: the runtime container path
used by vLLM, and the host path used by preflight. Mapping only the known mount
roots keeps the check strict for model bytes and patch manifests while avoiding
false negatives for valid container-only paths.

## GGUF-281 - Clean vNext Dense GGUF TP8 Contract Run on `.20` / `.30`

### Setup

- Date: 2026-06-29 UTC.
- Run label: `20260629T003817Z-gguf-dense27b-tp8`.
- Host lanes: `.20` and `.30`.
- Profile: `gguf-dense27b-tp8`.
- Runtime contract:
  - model format: GGUF;
  - model family: Qwen3.6 27B Dense;
  - profile: `dense27b_tp8_fullbar_p2pon`;
  - TP8;
  - FP16;
  - P2P-on;
  - `MAX_MODEL_LEN=131072`;
  - overlay: `gguf_dense`;
  - patch target groups: `common dense gguf`;
  - Dense GGUF bundle manifest
    `ce2c6f2d974de34c7abf4c25a67985b0eccc452a97f1a887ca97a8de1687f8d2`.
- Benchmark ladder: `8` warmups, `c1_128` uncapped strict, `c1_2000`, and
  `c1_10000`.

### Result

Both generated artifacts validated, both generated Docker wrappers reached
`/v1/models`, and both lanes completed the normal ladder. The test containers
were stopped after capture and VRAM returned to idle.

| host | case | backend decode TPS | finish | strict gate valid |
| --- | --- | ---: | --- | --- |
| `.20` | `c1_128_strict` | `69.914` | stop | true |
| `.20` | `c1_2000` | `70.759` | length | false |
| `.20` | `c1_10000` | `66.316` | length | false |
| `.30` | `c1_128_strict` | `69.851` | stop | true |
| `.30` | `c1_2000` | `70.959` | length | false |
| `.30` | `c1_10000` | `66.437` | length | false |

The strict requests stopped normally and passed the Qwen gate. The fixed-token
tiers are capped throughput tiers, so their strict-gate field is intentionally
false.

### Promote / Reject

Promote:

- `gguf-dense27b-tp8` as a clean contract-run GGUF Dense path across `.20` and
  `.30`;
- the generated launcher, minimal bundle, overlay isolation, and normal ladder
  as sufficient package-path evidence for Dense GGUF;
- Dense GGUF c1_10000 as beating the published Dense release gate value
  `66.069` on both primary lanes.

Reject:

- treating earlier scratch-overlay Dense GGUF runs as the only evidence now
  that the clean contract path has reproduced the band;
- any Dense GGUF path that bypasses profile generation or hand-assembles
  production `EXTRA_VLLM_ARGS`.

### Reason

This run used the same vNext contract machinery intended for developer
handoff: binary model probe, profile validation, patch-bundle manifest check,
generated Docker wrapper, generated benchmark environment, and the standard
warmup/strict/fixed-token ladder. It proves the Dense GGUF result is no longer
only a scratch source-work artifact.

## GGUF-282 - Clean vNext MoE GGUF TP4 Contract Run on `.20` / `.30`

### Setup

- Date: 2026-06-29 UTC.
- Run label: `20260629T003817Z-gguf-moe35b-tp4`.
- Host lanes: `.20` and `.30`.
- Profile: `gguf-moe35b-tp4`.
- Runtime contract:
  - model format: GGUF;
  - model family: Qwen3.6 35B-A3B MoE;
  - profile: `moe35b_tp4_fullbar_p2pon`;
  - TP4;
  - FP16;
  - P2P-on;
  - `MAX_MODEL_LEN=131072`;
  - overlay: `gguf_moe`;
  - patch target groups: `common moe gguf`;
  - MoE GGUF bundle manifest
    `794cb8760003cfa7dafeb0f06ee01823e5e8eebb3f9d398d8bc8bd7697143f0b`.
- Benchmark ladder: `8` warmups, `c1_128` uncapped strict, `c1_2000`, and
  `c1_10000`.

### Result

Both generated artifacts validated, both generated Docker wrappers reached
`/v1/models`, and both lanes completed the normal ladder. The startup logs
confirmed that the GGUF MoE overlay was active, BF16 metadata was cast to FP16,
async scheduling was enabled, and F16 GGUF Linear/FusedMoE layers used the
unquantized path.

The custom top-k8 MoE fastpath showed a mixed disposition: larger graph shapes
were rejected by shape/layout checks, while token-sized decode shapes became
active for `tokens=4`, `tokens=2`, and `tokens=1`. Keep this as source-path
nuance rather than a standalone performance claim.

| host | case | backend decode TPS | finish | strict gate valid |
| --- | --- | ---: | --- | --- |
| `.20` | `c1_128_strict` | `119.333` | stop | true |
| `.20` | `c1_2000` | `120.565` | length | false |
| `.20` | `c1_10000` | `113.366` | length | false |
| `.30` | `c1_128_strict` | `119.521` | stop | true |
| `.30` | `c1_2000` | `120.460` | length | false |
| `.30` | `c1_10000` | `113.258` | length | false |

The strict requests stopped normally and passed the Qwen gate. The test
containers were stopped after capture and VRAM returned to idle on both lanes.

### Promote / Reject

Promote:

- `gguf-moe35b-tp4` as a clean contract-run GGUF MoE path across `.20` and
  `.30`;
- the generated launcher, minimal bundle, overlay isolation, and normal ladder
  as sufficient package-path evidence for MoE GGUF;
- GGUF MoE TP4 as strict-valid under the generated vNext contract, with
  c1_10000 above the native/current TP4 band on both primary lanes.

Reject:

- interpreting token-sized top-k8 fastpath activation as proof that every graph
  shape uses that path;
- any MoE GGUF reproduction path that imports Dense-only overlay settings or
  relies on manual production `EXTRA_VLLM_ARGS`.

### Reason

This run validates the MoE GGUF path under the same contract machinery as the
Dense GGUF run and confirms `.30` is an equivalent performing lane after
repaste work for this profile. It also preserves the important source nuance:
the MoE GGUF path is not just "Dense settings plus a larger model"; it needs
its own overlay, profile contract, patch-target groups, and graph/fastpath
handling.

## GGUF-283 - HF MoE TP8 Group-Gated Runtime Baseline on `.20` / `.30`

### Setup

- Date: 2026-06-29 UTC.
- Run label: `20260629T003817Z-hf-moe35b-tp8`.
- Host lanes: `.20` and `.30`.
- Profile: `hf-moe35b-tp8`.
- Runtime contract:
  - model format: HF safetensors;
  - model family: Qwen3.6 35B-A3B MoE;
  - profile: `moe35b_tp8_fullbar_p2pon`;
  - TP8;
  - FP16;
  - P2P-on;
  - `MAX_MODEL_LEN=131072`;
  - overlay: `hf_release`;
  - patch target groups: `common moe hf moe_tp8`;
  - expanded clean HF bundle manifest
    `d9e1168f67731ab66ee53671f89dfb31fd3c74c8f61e2f0d0b7c4f0704c298be`.
- Benchmark ladder: `8` warmups, `c1_128` uncapped strict, `c1_2000`, and
  `c1_10000`.

### Result

Both generated artifacts validated, both generated Docker wrappers reached
`/v1/models`, and both lanes completed the normal ladder. The test containers
were stopped after capture and VRAM returned to idle on both lanes.

The TP8 startup logs showed the same important MoE fastpath nuance seen in
other MoE paths: larger graph shapes were rejected by shape/layout checks, and
token-sized decode shapes became active. The logs also looked for an
`E=256,N=64` tuned MoE config that is not currently bundled. Despite that
missing tuned-config file, the profile reproduced a high-band TP8 lane under
the generated contract.

| host | case | backend decode TPS | finish | strict gate valid |
| --- | --- | ---: | --- | --- |
| `.20` | `c1_128_strict` | `115.041` | stop | true |
| `.20` | `c1_2000` | `115.549` | length | false |
| `.20` | `c1_10000` | `108.805` | length | false |
| `.30` | `c1_128_strict` | `114.698` | stop | true |
| `.30` | `c1_2000` | `115.532` | length | false |
| `.30` | `c1_10000` | `108.673` | length | false |

### Promote / Reject

Promote:

- `hf-moe35b-tp8` as runtime-validated under the generated vNext contract on
  `.20` and `.30`;
- `PATCH_TARGET_GROUPS="common moe hf moe_tp8"` as sufficient to keep HF MoE
  TP8 isolated from GGUF overlays while preserving the TP8-specific MoE env
  surface;
- the current `.20`/`.30` pair as equivalent performing lanes for this profile.

Reject:

- leaving HF MoE TP8 listed as only synthetic/preflight validated;
- assuming the missing `E=256,N=64` tuned-config warning is a hard blocker for
  high-band TP8 reproduction.

### Reason

This closes the main HF baseline-protection gap for vNext: HF Dense, HF MoE
TP4, HF MoE TP8, GGUF Dense, and GGUF MoE have all now been exercised through
the contract-generated path, with fail-closed profile/format guardrails still
covered by local synthetic tests. The remaining release decision is not basic
runtime viability; it is whether and how to publish the GGUF path as a public
claim.

## GGUF-284 - Runtime vLLM Arg-Schema Checks on `.20` / `.30`

### Setup

- Date: 2026-06-29 UTC.
- Host lanes: `.20` and `.30`.
- Runtime image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Check: capture `vllm serve --help=all` from the pinned runtime image with
  GPU devices exposed, then run `verify_vllm_arg_schema.sh` against generated
  launch artifacts.

### Result

The runtime help capture succeeded on both lanes only when the throwaway help
container was given GPU device access; without devices, vLLM's ROCm platform
import fails before printing help.

Schema checks passed on both `.20` and `.30` for:

- `gguf-dense27b-tp8`
- `gguf-moe35b-tp4`
- `hf-moe35b-tp4`
- `hf-moe35b-tp8`

### Promote / Reject

Promote:

- runtime-help schema validation as a required guardrail before publishing or
  sharing generated launch artifacts;
- GPU-device-enabled help capture for this ROCm image, even though no model is
  loaded.

Reject:

- assuming `vllm serve --help` is CPU-only for this image;
- trusting generated argv without checking it against the exact runtime image.

### Reason

This check directly addresses vLLM flag drift. It validates the generated
argument names against the image that will run them, instead of depending on
memory or a stale help page.

## GGUF-285 - Public-Staged GGUF Reproduction Path Audit on `.30`

### Setup

- Date: 2026-06-29 UTC.
- Host lane: `.30`.
- Runtime image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Local model root was staged from a simulated public release-asset directory:
  split F16 GGUF parts, part manifests, final SHA256 checks, and the Dense
  text-config archive.
- Benchmark sequence:
  `8` warmups -> `c1_128` uncapped strict -> `c1_2000` -> `c1_10000`.

### Result

Dense GGUF TP8 reproduced from staged public-style assets:

| case | backend decode TPS | finish | strict gate valid |
| --- | ---: | --- | --- |
| `c1_128_strict` | `70.072` | stop | true |
| `c1_2000` | `71.031` | length | false |
| `c1_10000` | `66.487` | length | false |

The first Dense warmup had a large setup/prefill cost, then later warmups
settled near the expected decode band. This confirms that the documented
warmups are required for representative fixed-token numbers.

MoE GGUF TP4 initially failed because `TOKENIZER` pointed at a validation-host
Hugging Face snapshot path. That was rejected as non-portable. The profile was
corrected to use the public `Qwen/Qwen3.6-35B-A3B` tokenizer repo pinned by
`TOKENIZER_REVISION=995ad96eacd98c81ed38be0c5b274b04031597b0`.

After that correction, MoE GGUF TP4 reproduced:

| case | backend decode TPS | finish | strict gate valid |
| --- | ---: | --- | --- |
| `c1_128_strict` | `120.319` | stop | true |
| `c1_2000` | `121.018` | length | false |
| `c1_10000` | `113.760` | length | false |

### Promote / Reject

Promote:

- split release assets plus SHA256 manifests as the portable GGUF staging path;
- the Dense text-config archive as required for Dense GGUF reproduction;
- public pinned MoE tokenizer repo/revision as the portable replacement for
  validation-host snapshot paths;
- `CHECK_REQUIRED_PATHS=0 CHECK_PATCH_BUNDLE_PATHS=1` for host-side GGUF
  launch-artifact generation, while still verifying host-visible GGUF bytes and
  patch-bundle manifests.

Reject:

- internal Hugging Face cache snapshot paths in public profiles;
- treating a single local NVMe model directory as a public reproduction source;
- measuring Dense/MoE GGUF without the normal warmups, because first-request
  setup costs materially distort the results.

### Reason

This audit proves the release path can be made portable without mutating the
runtime image: public/staged model artifacts, generated launch wrappers,
runtime argument-schema validation, P2P-on profiles, and the normal benchmark
ladder reproduce the expected GGUF bands on a clean `.30` run.

## GGUF-286 - Public-Staged GGUF POSIX-Runner Repeat on `.30`

### Setup

- Date: 2026-06-29 UTC.
- Host lane: `.30`.
- Runtime image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Local model root was staged from the same simulated public release-asset
  directory used by GGUF-285: split F16 GGUF parts, SHA256 manifests, final
  assembled-file checks, and the Dense text-config archive.
- The repo copy was updated after the vNext benchmark runner was converted to
  run the normal ladder directly under POSIX `/bin/sh` instead of delegating to
  the older Bash-only v0.2 helper.
- Benchmark sequence:
  `8` warmups -> `c1_128` uncapped strict -> `c1_2000` -> `c1_10000`.

### Result

Dense GGUF TP8 reproduced from staged public-style assets with the POSIX
runner:

| case | backend decode TPS | finish | strict gate valid |
| --- | ---: | --- | --- |
| `c1_128_strict` | `69.986` | stop | true |
| `c1_2000` | `70.935` | length | false |
| `c1_10000` | `66.403` | length | false |

MoE GGUF TP4 reproduced from staged public-style assets and the public pinned
MoE tokenizer with the POSIX runner:

| case | backend decode TPS | finish | strict gate valid |
| --- | ---: | --- | --- |
| `c1_128_strict` | `120.822` | stop | true |
| `c1_2000` | `121.733` | length | false |
| `c1_10000` | `114.442` | length | false |

### Promote / Reject

Promote:

- the POSIX vNext benchmark runner as the public reproduction runner for the
  vNext package;
- the staged split-asset GGUF path plus Dense text-config archive;
- the public `Qwen/Qwen3.6-35B-A3B` tokenizer repo pinned by
  `TOKENIZER_REVISION=995ad96eacd98c81ed38be0c5b274b04031597b0`;
- the normal Qwen benchmark ladder and bundled begin-think proxy for these
  Qwen3.6 profiles.

Reject:

- delegating vNext reproduction to the older Bash-only helper;
- treating validation-host cache paths as public reproduction inputs;
- measuring without the normal warmups.

### Reason

This repeat closes the script-portability gap discovered after GGUF-285. The
same public-staged assets and generated launch artifacts reproduce the Dense
and MoE GGUF bands when the benchmark ladder is executed directly by the
vNext POSIX-shell runner. No runtime image, Docker digest, model file, or patch
bundle changed.

## GGUF-287 - Public-Staged HF Reproduction Path Audit on `.30`

### Setup

- Date: 2026-06-29 UTC.
- Host lane: `.30`.
- Runtime image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Public model source:
  - `Qwen/Qwen3.6-27B`, downloaded with `hf download`;
  - `Qwen/Qwen3.6-35B-A3B`, downloaded with `hf download`.
- The Hugging Face packages were staged into a clean local model root under
  the public-staging area, not copied from retained validation-host snapshots.
- Fresh vNext launch artifacts were generated for:
  - `hf-dense27b-tp8`;
  - `hf-moe35b-tp4`;
  - `hf-moe35b-tp8`.
- Runtime vLLM argument-schema validation passed against the exact public
  image before launching benchmarks.
- Benchmark sequence:
  `8` warmups -> `c1_128` uncapped strict -> `c1_2000` -> `c1_10000`.

### Result

HF Dense TP8 reproduced from public upstream model files:

| case | backend decode TPS | finish | strict gate valid |
| --- | ---: | --- | --- |
| `c1_128_strict` | `69.682` | stop | true |
| `c1_2000` | `71.346` | length | false |
| `c1_10000` | `66.729` | length | false |

HF MoE TP4 reproduced from public upstream model files:

| case | backend decode TPS | finish | strict gate valid |
| --- | ---: | --- | --- |
| `c1_128_strict` | `111.964` | stop | true |
| `c1_2000` | `115.937` | length | false |
| `c1_10000` | `108.848` | length | false |

HF MoE TP8 reproduced from public upstream model files:

| case | backend decode TPS | finish | strict gate valid |
| --- | ---: | --- | --- |
| `c1_128_strict` | `115.488` | stop | true |
| `c1_2000` | `116.258` | length | false |
| `c1_10000` | `109.242` | length | false |

### Promote / Reject

Promote:

- public Hugging Face staging with `hf download` as the portable HF model
  source path;
- generated vNext launch artifacts plus runtime argument-schema validation for
  HF profiles;
- the normal POSIX-runner benchmark ladder for HF Dense and MoE profiles;
- documenting strict backend TPS and gate validity separately from capped
  fixed-token tiers.

Reject:

- retained Hugging Face snapshot directories as public reproduction inputs;
- assuming a validation-host model cache is equivalent to a clean public
  staging flow;
- judging fixed-token tiers before the documented warmups complete.

### Reason

This audit confirms that the HF side of the vNext package is portable in the
same sense as the earlier releases: a fresh checkout, public image, public
model repositories, generated profile artifacts, and local NVMe-backed storage
are enough to reproduce the expected Dense and MoE bands on a qualifying
GFX906 host.

## GGUF-288 - Clean-Room GGUF Package-Path Audit on `.30`

### Setup

- Date: 2026-06-29 UTC.
- Host lane: `.30`.
- Runtime image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- A fresh checkout directory was created outside the working repository.
- GGUF inputs were staged through the split-release-asset layout, including
  part manifests, final SHA256 verification, and the Dense text-config archive.
- `RELEASE_ASSET_BASE=file://...` was used only as a maintainer-side simulation
  of final GitHub Release assets. This is not acceptable as a public release
  dependency.
- Launch artifacts were generated using profile defaults and only
  `LOCAL_MODEL_ROOT` for the public host model-path mapping.
- Runtime vLLM argument-schema validation passed against the exact runtime
  image before the benchmark ladder.
- Benchmark sequence:
  `8` warmups -> `c1_128` uncapped strict -> `c1_2000` -> `c1_10000`.

### Result

Dense GGUF TP8 reproduced from the clean-room package path:

| case | backend decode TPS | finish | strict gate valid |
| --- | ---: | --- | --- |
| `c1_128_strict` | `69.580` | stop | true |
| `c1_2000` | `70.837` | length | false |
| `c1_10000` | `66.309` | length | false |

MoE GGUF TP4 reproduced from the clean-room package path:

| case | backend decode TPS | finish | strict gate valid |
| --- | ---: | --- | --- |
| `c1_128_strict` | `117.731` | stop | true |
| `c1_2000` | `119.649` | length | false |
| `c1_10000` | `112.568` | length | false |

### Promote / Reject

Promote:

- profile-default launch generation with `LOCAL_MODEL_ROOT` as the normal
  public host-specific model path input;
- split-release-asset staging plus final SHA256 verification for GGUF inputs;
- generated Docker wrappers, runtime schema validation, and the POSIX benchmark
  ladder as the clean-room reproduction path;
- Dense GGUF TP8 and MoE GGUF TP4 as portable mechanics candidates pending real
  GitHub Release asset upload.

Reject:

- relying on `.20`, `.30`, InfiniBand addresses, validation-host model caches,
  private mounts, or retained local artifacts as public reproduction inputs;
- treating the `file://` release-asset simulation as a published public input;
- skipping the eight warmups before strict and fixed-token benchmark tiers.

### Reason

This audit proves the vNext GGUF path can behave like the prior public
releases: a clean checkout, pinned public Docker image, profile-generated
artifacts, user-controlled NVMe-backed staging, and the normal benchmark ladder
are sufficient to reproduce the Dense and MoE GGUF bands. The remaining
publication gate is operational, not source-level: upload the exact split GGUF
assets and manifests to GitHub Releases and verify this same flow against those
public URLs.

## GGUF-289 - Stand-Alone vNext GGUF Release-Package Audit on `.30`

### Setup

- Date: 2026-06-29 UTC.
- Host lane: `.30`.
- Runtime image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- The vNext draft was treated as a stand-alone release package, not a v0.2.1
  addendum.
- The draft checkout was copied to NVMe-backed storage outside the working
  repository.
- GGUF inputs were staged through the split-release-asset layout using
  `stage_vnext_gguf_assets.sh`.
- Dense and MoE part manifests, part hashes, final assembled-file SHA256
  values, and the Dense text-config archive were verified.
- `RELEASE_ASSET_BASE=file://...` was used only as a maintainer-side
  simulation of final GitHub Release assets. It remains rejected as a public
  dependency.
- `verify_vnext_serviceability.sh`, `verify_vnext_contract_matrix.sh`,
  generated launch-artifact validation, runtime image/digest validation, and
  vLLM argument-schema validation passed before the benchmark ladder.
- Benchmark sequence:
  `8` warmups -> `c1_128` uncapped strict -> `c1_2000` -> `c1_10000`.

### Result

Dense GGUF TP8 stand-alone package-path result:

| case | backend decode TPS | finish | strict gate valid |
| --- | ---: | --- | --- |
| `c1_128_strict` | `70.070` | stop | true |
| `c1_2000` | `71.129` | length | false |
| `c1_10000` | `66.530` | length | false |

MoE GGUF TP4 stand-alone package-path result:

| case | backend decode TPS | finish | strict gate valid |
| --- | ---: | --- | --- |
| `c1_128_strict` | `119.197` | stop | true |
| `c1_2000` | `120.618` | length | false |
| `c1_10000` | `113.377` | length | false |

Startup observation:

- The Dense GGUF server exceeded an initial five-minute health polling window
  during first-run graph compilation.
- The container was not hung: logs advanced through model loading, graph
  compile, and cache-save events, and GPU activity showed the workload was
  active.
- The benchmark runner's backend-readiness loop handled the eventual service
  readiness correctly before starting warmups.

### Promote / Reject

Promote:

- vNext as a stand-alone release package with its own tag, assets,
  instructions, validation table, and release boundary;
- the profile-generated Docker wrapper and POSIX benchmark runner as the
  public reproduction path;
- binary format/family detection, profile-scoped overlays, patch-bundle
  manifest checks, and runtime argument-schema validation as required gates;
- Dense GGUF TP8 and MoE GGUF TP4 as strict-valid release-candidate lanes when
  public release assets are present.

Reject:

- describing vNext as a v0.2.1 patch or addendum;
- treating simulated `file://` release assets as public proof;
- using validation-lane paths, retained model caches, InfiniBand addresses, or
  private mounts as public reproduction inputs;
- classifying first-run shared-memory wait warnings as failure while compile
  logs and GPU activity still show progress.

### Reason

The stand-alone audit reproduced both GGUF lanes from the vNext release
machinery and normal benchmark ladder. Dense cleared the ai-info 10K gate at
`66.530` backend TPS on `c1_10000`, and MoE TP4 remained strict-valid with
`113.377` backend TPS on `c1_10000`. The remaining publication blocker is
release packaging: the exact GGUF split assets and hashes must be attached to
the final GitHub Release and retested through public URLs.

## GGUF-290 - Clean-Room vNext Release-Note Preflight Without Model Bytes

### Setup

- Date: 2026-06-29 UTC.
- Scope: release-note mechanics, not runtime benchmark reproduction.
- Source checkout: current working tree copied to a fresh `/tmp` clean-room
  directory without `.git`.
- Model inputs: tiny synthetic GGUF and HF/safetensors signatures only.
- Runtime image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Profiles tested:
  - `gguf-dense27b-tp8`
  - `gguf-moe35b-tp4`
  - `hf-dense27b-tp8`
  - `hf-moe35b-tp4`
  - `hf-moe35b-tp8`

### Result

The clean-room preflight followed the non-serving portion of the draft vNext
release flow:

- built `tools/model-format-probe`;
- ran `vnext_profile_inspect.sh`;
- ran `verify_vnext_serviceability.sh`, including the
  `release-note-locks-ok` check tying artifact hashes, tokenizer revisions, and
  patch-bundle manifests back to the draft release note and scripts;
- ran `verify_vnext_contract_matrix.sh`;
- captured `vllm serve --help=all` from the pinned runtime image with GPU
  devices exposed;
- generated launch artifacts for all five profiles using `LOCAL_MODEL_ROOT`;
- validated every generated launch artifact;
- validated every generated `vllm_argv.txt` against the runtime image's vLLM
  help output.

All five profiles passed launch-artifact and runtime argument-schema
validation:

| Profile | Launch artifact | Runtime arg schema |
| --- | --- | --- |
| `gguf-dense27b-tp8` | pass | pass |
| `gguf-moe35b-tp4` | pass | pass |
| `hf-dense27b-tp8` | pass | pass |
| `hf-moe35b-tp4` | pass | pass |
| `hf-moe35b-tp8` | pass | pass |

### Promote / Reject

Promote:

- the current vNext profile/launcher/serviceability/schema flow as internally
  consistent from a fresh checkout;
- `LOCAL_MODEL_ROOT` inference as the default public host-path mechanism;
- synthetic model signatures as sufficient for release-note command-shape and
  argument-schema validation before large model staging begins;
- the requirement that final publication must rerun the same flow with real
  public release assets and model bytes.

Reject:

- treating this preflight as benchmark reproduction;
- treating synthetic GGUF/HF signatures as substitutes for the final model
  assets;
- treating the maintainer clean-room copy as public release proof;
- publishing before the final split GGUF assets, manifests, final SHA256 files,
  and Dense text-config archive are attached to the GitHub Release and retested
  through public URLs.

### Reason

The release-note mechanics are now stronger than a local working-tree check:
the command flow can be copied into a clean directory, generate profile-specific
launch artifacts for every published vNext profile, and validate those
artifacts against the real runtime image's vLLM argument schema. The remaining
gap is intentionally outside this preflight: exact public model assets and
runtime benchmark reproduction from those assets.

## GGUF-291 - vNext Split-Asset Staging Fixture and Manifest Hardening

### Setup

- Date: 2026-06-29 UTC.
- Scope: `stage_vnext_gguf_assets.sh` split-asset mechanics, not benchmark
  reproduction.
- Method: generated tiny local GGUF fixture files and a tiny Dense text-config
  archive, then ran a temporary hash-substituted copy of the staging script
  against `RELEASE_ASSET_BASE=file://...`.
- Negative fixture: replaced one dense part manifest row with a `../` path.

### Result

The positive fixture passed:

- downloaded the Dense text-config archive from the local asset base;
- verified the text-config archive hash;
- extracted `config.json` and `tokenizer.json`;
- downloaded both GGUF part manifests;
- downloaded and verified every listed part;
- assembled each GGUF file from the manifest-listed part order;
- verified the final assembled GGUF hashes.

The negative fixture failed closed before part download when the manifest
listed `../evil.part-0000`.

### Promote / Reject

Promote:

- manifest-listed assembly order instead of shell glob assembly;
- rejecting part names with paths, `..`, whitespace, or unexpected prefixes;
- removing stale split outputs in the publisher checklist before generating
  release parts;
- text-config archive path checks before extraction.

Reject:

- relying on whatever `*.part-*` files happen to be present in the staging
  directory;
- accepting stale publisher split outputs;
- accepting manifest rows that could write outside the model staging
  directory.

### Reason

The fixture proves the release-asset staging logic can work from a clean
release-asset-style source and that obvious unsafe manifest rows are rejected.
It still does not prove final public reproduction because the real GGUF
release assets have not been attached to a GitHub Release and retested through
public URLs.

## GGUF-292 - vNext All-Profile Release-Note Reproduction Loop

### Setup

- Date: 2026-06-29 UTC.
- Scope: release-note command accuracy for all vNext profiles.
- Files checked:
  - `releases/draft-vnext-gfx906-rocm72-gguf-hf-repro.md`
  - `qwen36-gfx906/profiles/vnext/README.md`
  - `qwen36-gfx906/vnext_repro_launcher.sh`
  - `qwen36-gfx906/run_vnext_profile_benchmark.sh`
- Profiles covered:
  - `gguf-dense27b-tp8`
  - `gguf-moe35b-tp4`
  - `hf-dense27b-tp8`
  - `hf-moe35b-tp4`
  - `hf-moe35b-tp8`

### Result

The draft release note previously generated all five launch artifacts, but
only showed verify/start/benchmark commands for `gguf-dense27b-tp8`. That was
too weak for a stand-alone release note whose goal is to reproduce every
published profile result.

The release-note flow now explicitly:

- verifies every generated launch artifact;
- captures one runtime `vllm serve --help=all` schema from the pinned image;
- validates every profile's generated `vllm_argv.txt` against that schema;
- states that profiles default to port `8001`;
- runs one profile at a time with reusable container name
  `vnext_repro_current`;
- assigns a separate `HOST_RUNTIME_ROOT` per profile; and
- runs the normal benchmark ladder for each profile:
  `8` warmups -> `c1_128` uncapped strict -> `c1_2000` -> `c1_10000`.

Validation after the release-note edit:

- POSIX shell syntax check for vNext scripts: pass.
- `verify_vnext_serviceability.sh`: pass, including `release-note-locks-ok`.
- `verify_vnext_contract_matrix.sh`: pass.
- `CHECK_RUNTIME_IMAGE=1 CHECK_RUNTIME_PATHS=1 verify_vnext_serviceability.sh`:
  pass against the pinned public runtime image digest.

### Promote / Reject

Promote:

- documenting the all-profile loop as the required stand-alone reproduction
  shape;
- using a reusable container name so default port `8001` profiles do not
  collide;
- keeping runtime caches separate per profile;
- validating all generated argv files before starting benchmark containers.

Reject:

- publishing vNext with only a single-profile benchmark example;
- treating concurrent same-port profile launches as the normal public path;
- treating local release-note command validation as full public reproduction
  before final GitHub Release assets exist.

### Reason

The release note now carries a user from public input staging through
all-profile launch-artifact validation and all-profile benchmark execution.
This is required for a stand-alone release. The remaining blocker is not the
command shape; it is the absence of final public GGUF release assets and a
tagged release checkout that can be retested from public URLs.

## GGUF-293 - Clean-Room vNext Release-Note Command Replay

### Setup

- Date: 2026-06-29 UTC.
- Scope: replay the current draft vNext release-note command flow from a fresh
  non-git checkout copy.
- Model inputs: tiny synthetic GGUF and safetensors signatures staged under
  the same `LOCAL_MODEL_ROOT` layout required by the release note.
- Runtime image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Profiles replayed:
  - `gguf-dense27b-tp8`
  - `gguf-moe35b-tp4`
  - `hf-dense27b-tp8`
  - `hf-moe35b-tp4`
  - `hf-moe35b-tp8`

### Result

The clean-room replay followed the draft release-note flow through the
non-serving validation boundary:

- built `tools/model-format-probe`;
- inspected vNext profiles;
- ran `verify_vnext_serviceability.sh`;
- ran `verify_vnext_contract_matrix.sh`;
- generated launch artifacts for all five profiles with `LOCAL_MODEL_ROOT`;
- verified every generated launch artifact;
- captured `vllm serve --help=all` from the pinned runtime image with GPU
  devices exposed; and
- validated every generated profile argv against the captured runtime schema.

All five profiles passed command-shape and runtime-argument validation from
the clean-room copy.

### Promote / Reject

Promote:

- the current draft release-note command sequence through all-profile
  launch-artifact and schema validation;
- the one-profile-at-a-time container flow as the documented path for the
  shared default port;
- synthetic model signatures as enough to prove command shape before full
  model-byte staging.

Reject:

- treating this as full benchmark reproduction;
- treating synthetic model signatures as substitutes for the real GGUF/HF
  model bytes;
- marking vNext complete before final GitHub Release assets and a tagged
  checkout are retested through public URLs.

### Reason

This replay proves the release-note instructions are internally executable up
to the point where real model bytes and serving are required. It does not prove
the benchmark numbers from scratch because the final public GGUF release
assets do not yet exist, and the serving benchmark ladder was intentionally
not run against synthetic model signatures.

## GGUF-294 - vNext Public GGUF Asset Gate

### Setup

- Date: 2026-06-29 UTC.
- Scope: make the remaining public-asset blocker executable instead of relying
  only on release-note prose.
- Script added: `qwen36-gfx906/verify_vnext_public_assets.sh`.
- Related script: `qwen36-gfx906/stage_vnext_gguf_assets.sh`.

### Result

The new public asset gate:

- requires a final release tag instead of `<release-tag>`;
- rejects any angle-bracket placeholder tag;
- rejects unapproved `file://` asset bases unless
  `ALLOW_LOCAL_ASSET_PREFLIGHT=1` is set;
- stages GGUF assets through the same public staging script used by the release
  instructions;
- verifies the staged Dense GGUF, MoE GGUF, and Dense text-config paths;
- builds the binary model-format probe; and
- checks that the staged GGUF files satisfy the `gguf-dense27b-tp8` and
  `gguf-moe35b-tp4` profile contracts.

Fail-closed checks passed:

- placeholder tag `<release-tag>` failed before any download;
- angle-bracket placeholder variants are rejected before any download;
- `RELEASE_ASSET_BASE=file://...` failed unless local preflight was explicitly
  allowed.

Follow-up validation:

- POSIX shell syntax check: pass.
- `verify_vnext_serviceability.sh`: pass, including the public-asset gate lock.
- `verify_vnext_contract_matrix.sh`: pass.
- `git diff --check`: pass.

### Promote / Reject

Promote:

- `verify_vnext_public_assets.sh` as the required publication gate for the
  hosted GGUF asset path;
- fail-closed handling for placeholder tags and accidental local asset bases;
- checking staged assets against profile contracts before benchmark serving.

Reject:

- publishing vNext while this gate cannot pass against final GitHub Release
  asset URLs;
- treating local `file://` preflight as public reproduction evidence;
- treating the gate as a benchmark substitute.

### Reason

The release notes already identified the missing public assets as the blocker.
The new gate makes that blocker mechanical: once the final split GGUF assets,
part manifests, final hash files, and Dense text-config archive are attached to
the GitHub Release, this script must pass before the GGUF results can be
called public reproduction claims. It still does not complete the active goal
because final public assets and serving benchmark reruns are not available yet.

## GGUF-295 - vNext Profile README Command Alignment

### Setup

- Date: 2026-06-29 UTC.
- Scope: compare the vNext profile README command snippets against the draft
  vNext release-note reproduction flow.
- Files checked:
  - `qwen36-gfx906/profiles/vnext/README.md`
  - `releases/draft-vnext-gfx906-rocm72-gguf-hf-repro.md`

### Result

The profile README still had a single-profile verification/start/benchmark
example and used the older `dense-gguf-check` launch directory name, while the
release note had moved to an all-profile loop with canonical profile directory
names.

The profile README now:

- defines `RELEASE_TAG` near the public asset gate example;
- provides a default `LOCAL_MODEL_ROOT` in that gate snippet;
- stages GGUF assets with `"$RELEASE_TAG"`;
- uses `./vnext-launch-runs/gguf-dense27b-tp8` for the Dense example;
- verifies every generated launch artifact;
- captures one runtime schema and validates every profile's argv against it;
- documents one-profile-at-a-time serving for the shared default port; and
- runs the normal benchmark ladder for all five profiles.

Follow-up validation:

- POSIX shell syntax check: pass.
- `verify_vnext_serviceability.sh`: pass.
- `CHECK_RUNTIME_IMAGE=1 CHECK_RUNTIME_PATHS=1 verify_vnext_serviceability.sh`:
  pass.
- `verify_vnext_contract_matrix.sh`: pass.
- release-facing targeted scan: only expected negative-boundary language.

### Promote / Reject

Promote:

- using canonical profile directory names in support docs;
- keeping the profile README and release note aligned on the all-profile
  reproduction flow;
- documenting one-profile-at-a-time serving anywhere public instructions
  mention the shared default port.

Reject:

- maintaining separate single-profile examples that use different launch
  directory names from the release note;
- leaving `$RELEASE_TAG` implicit in copy/paste snippets.

### Reason

The active goal is to follow release notes and reproduce results from scratch.
Support docs that drift from the release-note command sequence create avoidable
user error. This update keeps the profile README as a helper for the same
stand-alone release flow rather than a parallel reproduction path.

## GGUF-296 - Public Asset Gate Positive Fixture

### Setup

- Date: 2026-06-29 UTC.
- Scope: positive-path validation for
  `qwen36-gfx906/verify_vnext_public_assets.sh`.
- Method:
  - copied the current working tree to a temporary clean-room checkout;
  - generated tiny release-asset-style GGUF fixtures;
  - generated split part files and `*.parts.sha256` manifests;
  - generated a tiny Dense text-config archive;
  - substituted fixture hashes only in the temporary copy of
    `stage_vnext_gguf_assets.sh`;
  - ran the public asset gate with
    `ALLOW_LOCAL_ASSET_PREFLIGHT=1` and `RELEASE_ASSET_BASE=file://...`.

### Result

The positive fixture passed:

- downloaded the Dense text-config archive through the staging script;
- verified the archive hash and extracted `config.json` / `tokenizer.json`;
- downloaded both GGUF part manifests;
- downloaded and verified every listed split part;
- assembled Dense and MoE GGUF files from manifest order;
- verified final assembled GGUF hashes;
- generated the `gguf-dense27b-tp8` launch artifact from the staged assets;
- generated the `gguf-moe35b-tp4` launch artifact from the staged assets; and
- verified both generated launch artifacts.

The gate printed:

```text
public-asset-profile-ok: gguf-dense27b-tp8
public-asset-profile-ok: gguf-moe35b-tp4
vNext public GGUF asset gate passed for tag: vnext-fixture
```

### Promote / Reject

Promote:

- the public asset gate wrapper as mechanically valid when a release-asset
  layout is present;
- manifest-order assembly and profile-contract verification as part of the
  publication gate.

Reject:

- treating the fixture as a benchmark reproduction;
- treating `file://` local preflight as public evidence;
- substituting fixture hashes into the real release script.

### Reason

The earlier gate tests proved fail-closed behavior. This positive fixture
proves the same gate can pass when release-style assets, manifests, and hashes
are available. The active goal remains open because the final public GGUF
assets have not been attached to a real GitHub Release and the full serving
benchmark ladder has not been rerun from those public URLs.

## GGUF-297 - vNext Public HF Input Gate

### Setup

- Date: 2026-06-29 UTC.
- Scope: add a public-input gate for the HF side of the vNext reproduction
  package.
- Script added: `qwen36-gfx906/verify_vnext_public_hf_inputs.sh`.
- Profiles covered:
  - `hf-dense27b-tp8`
  - `hf-moe35b-tp4`
  - `hf-moe35b-tp8`

### Result

The HF input gate:

- rejects unsafe or overly broad model roots;
- rejects HF revision overrides that do not match the pinned release values;
- can optionally run the pinned public `hf download` commands with
  `STAGE_HF_PUBLIC_INPUTS=1`;
- verifies staged HF Dense and MoE directories;
- generates all three HF launch artifacts from the staged model root; and
- verifies all three generated launch artifacts.

Fail-closed checks passed:

- `/` as `LOCAL_MODEL_ROOT` failed before profile validation;
- `HF_DENSE_REVISION=main` failed before profile validation.

A positive synthetic HF fixture also passed:

```text
public-hf-profile-ok: hf-dense27b-tp8
public-hf-profile-ok: hf-moe35b-tp4
public-hf-profile-ok: hf-moe35b-tp8
vNext public HF input gate passed
```

### Promote / Reject

Promote:

- `verify_vnext_public_hf_inputs.sh` as the HF counterpart to the GGUF public
  asset gate;
- explicit pinned-revision checks before HF profile validation;
- optional one-command HF download plus contract validation for users who want
  the gate to stage public inputs.

Reject:

- treating a staged HF directory as release-valid without checking all three
  HF profile contracts;
- allowing floating HF `main` revisions in a reproduction claim;
- treating synthetic HF fixtures as benchmark reproduction.

### Reason

The vNext release path has both hosted GGUF assets and public HF downloads.
The GGUF side now had a public asset gate, but the HF side only had raw
download commands plus later launcher preflight. This gate makes the HF input
validation equally explicit. It still does not complete the active goal because
real full-size HF model bytes must be downloaded from the pinned public repos
and benchmarked through the serving ladder before publication.

## GGUF-298 - HF Gate Download-Mode Fixture

### Setup

- Date: 2026-06-29 UTC.
- Scope: validate the `STAGE_HF_PUBLIC_INPUTS=1` branch of
  `verify_vnext_public_hf_inputs.sh`.
- Method:
  - created a temporary fake `hf` executable;
  - accepted only the pinned public repo/revision pairs from the release note;
  - generated tiny HF-style Dense and MoE directory fixtures; and
  - ran the HF gate with `STAGE_HF_PUBLIC_INPUTS=1`.

### Result

The fake downloader was invoked for the two expected public inputs:

```text
fake-hf-download-ok: Qwen/Qwen3.6-27B
fake-hf-download-ok: Qwen/Qwen3.6-35B-A3B
```

The HF gate then passed all profile checks:

```text
public-hf-profile-ok: hf-dense27b-tp8
public-hf-profile-ok: hf-moe35b-tp4
public-hf-profile-ok: hf-moe35b-tp8
vNext public HF input gate passed
```

### Promote / Reject

Promote:

- `STAGE_HF_PUBLIC_INPUTS=1` as a mechanically valid one-command staging and
  validation path for HF profiles;
- the pinned repo/revision pairs as the only accepted HF download inputs for
  this release package.

Reject:

- treating the fake downloader fixture as real Hugging Face download evidence;
- treating synthetic model files as benchmark reproduction;
- allowing the release note to rely on unstaged HF directories without running
  the HF input gate.

### Reason

The previous HF gate test proved staged-directory validation. This test proves
the optional download branch invokes the intended pinned public repo/revision
pairs before validation. Full completion still requires running the same gate
with the real Hugging Face CLI and full model bytes, then serving benchmark
reruns through the normal ladder.

## GGUF-299 - vNext Stand-Alone Release Boundary Lock

### Setup

- Date: 2026-06-29 UTC.
- Scope: prevent vNext release-package wording and profile documentation from
  drifting back into a v0.2.1 patch/addendum shape.
- Script updated: `qwen36-gfx906/verify_vnext_serviceability.sh`.
- Docs checked:
  - `releases/draft-vnext-gfx906-rocm72-gguf-hf-repro.md`
  - `qwen36-gfx906/profiles/vnext/README.md`

### Result

The serviceability gate now verifies that:

- the draft release note describes vNext as a stand-alone reproduction-package
  release;
- the draft release note says vNext is not a patch, addendum, or silent
  replacement for v0.2.0/v0.2.1;
- the profile README describes vNext as a future stand-alone release path;
- the profile README says vNext is not a patch, addendum, or silent
  replacement;
- the profile README says vNext must be reproducible from its own release tag;
- the profile README includes the same public reproduction anchors as the
  release note:
  - `verify_vnext_public_assets.sh`
  - `verify_vnext_public_hf_inputs.sh`
  - `STAGE_HF_PUBLIC_INPUTS=1`
  - `RELEASE_TAG=<release-tag>`
  - `vnext_repro_current`
  - `vllm-serve-help.txt`
  - `run_vnext_profile_benchmark.sh`
  - all five vNext profile ids.

The gate prints:

```text
profile-doc-locks-ok
```

### Promote / Reject

Promote:

- vNext as a stand-alone release package with its own tag, public assets,
  profile contracts, reproduction instructions, validation tables, and claim
  boundary;
- serviceability checks that lock both release-note and profile-README
  boundary wording;
- profile README lock checks as part of the developer serviceability gate.

Reject:

- describing vNext as a v0.2.1 patch, addendum, or silent replacement;
- relying on manually maintained profile docs without a drift check;
- publishing vNext before the final public tag/assets and real benchmark reruns
  can satisfy the same reproduction flow.

### Reason

The next package must be portable and publicly reproducible without depending
on internal hosts, retained caches, private mounts, or the historical v0.2.1
checkout. The profile README is a major helper surface for developers, so it
needs the same release-boundary locks as the draft release note. This does not
complete the active goal because final GitHub Release assets, a final tag, and
full real benchmark reruns from public inputs are still required.

## GGUF-300 - Clean-Room vNext Command Replay After Stand-Alone Boundary Lock

### Setup

- Date: 2026-06-29 UTC.
- Scope: replay the current vNext release-note command flow from a fresh
  temporary checkout after locking vNext as a stand-alone release line.
- Clean-room copy:
  `<local-clean-room-root>`
- Replay log:
  `<local-clean-room-root>`
- Method:
  - copied the current worktree to `/tmp`;
  - removed the built `tools/model-format-probe/model-format-probe` binary from
    the copy so `make -C tools/model-format-probe clean all` rebuilt it;
  - ran serviceability and contract checks before any fixture hash
    substitution;
  - staged synthetic HF fixtures under the public `LOCAL_MODEL_ROOT` layout;
  - staged synthetic split GGUF assets through `RELEASE_ASSET_BASE=file://...`
    with `ALLOW_LOCAL_ASSET_PREFLIGHT=1`;
  - temporarily substituted fixture hashes in the clean-room copy of
    `stage_vnext_gguf_assets.sh` only, so the split-asset mechanics could be
    exercised without final public release assets;
  - generated and verified launch artifacts for all five vNext profiles; and
  - schema-checked generated argv files against a synthetic help fixture.

### Result

The clean-room replay passed these release-note surfaces:

```text
release-note-locks-ok
profile-doc-locks-ok
vNext developer serviceability check passed
vNext contract matrix passed
public-hf-profile-ok: hf-dense27b-tp8
public-hf-profile-ok: hf-moe35b-tp4
public-hf-profile-ok: hf-moe35b-tp8
vNext public HF input gate passed
public-asset-profile-ok: gguf-dense27b-tp8
public-asset-profile-ok: gguf-moe35b-tp4
vNext public GGUF asset gate passed for tag: vnext-fixture
artifact-ok=gguf-dense27b-tp8
artifact-ok=gguf-moe35b-tp4
artifact-ok=hf-dense27b-tp8
artifact-ok=hf-moe35b-tp4
artifact-ok=hf-moe35b-tp8
schema-ok=gguf-dense27b-tp8
schema-ok=gguf-moe35b-tp4
schema-ok=hf-dense27b-tp8
schema-ok=hf-moe35b-tp4
schema-ok=hf-moe35b-tp8
benchmark-shape-ok=gguf-dense27b-tp8
benchmark-shape-ok=gguf-moe35b-tp4
benchmark-shape-ok=hf-dense27b-tp8
benchmark-shape-ok=hf-moe35b-tp4
benchmark-shape-ok=hf-moe35b-tp8
cleanroom-replay=passed
```

For every generated profile, the artifact preserved the benchmark ladder:

```text
8 warmups -> c1_128 uncapped strict -> c1_2000 -> c1_10000
```

and the generated `benchmark_env.sh` preserved:

- `PRE_MEASURE_WARMUP_REQUESTS=8`
- `PRE_MEASURE_WARMUP_MAX_TOKENS=2000`
- `USE_BEGIN_THINK_PROXY=1`

### Promote / Reject

Promote:

- the current release-note command order for serviceability, HF input gating,
  GGUF split-asset staging, all-profile artifact generation, artifact
  verification, argv-schema validation, and benchmark-ladder handoff;
- the stand-alone vNext boundary wording as compatible with a clean-room
  command replay;
- fixture-based release-note command replay as a useful pre-publication guard.

Reject:

- treating this fixture replay as real model-byte staging evidence;
- treating the synthetic help file as a replacement for runtime
  `vllm serve --help=all` capture;
- treating generated benchmark environment checks as benchmark reproduction;
- marking vNext release complete before final public GitHub Release assets,
  final tag checkout, real public HF/GGUF inputs, and full serving benchmark
  reruns pass.

### Reason

This replay proves the release-note command flow is internally coherent from a
fresh checkout and that all five profiles can be staged, generated, verified,
and handed to the benchmark runner shape without depending on internal host
paths. It does not satisfy the full objective because the final release tag,
final public GGUF assets, real HF model downloads, runtime help capture from
the final image, and full serving benchmark numbers still need to be verified
from public inputs.

## GGUF-301 - Real Runtime vLLM Help Schema Capture

### Setup

- Date: 2026-06-29 UTC.
- Scope: test the exact vNext release-note command that captures the runtime
  image's `vllm serve --help=all` output with GPU devices exposed.
- Clean-room copy:
  `<local-clean-room-root>`
- Runtime image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`
- Runtime digest already present locally:
  `sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e`
- Command shape:

```sh
docker run --rm \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --security-opt seccomp=unconfined \
  --security-opt label=disable \
  --entrypoint vllm \
  joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b \
  serve --help=all > ./vnext-launch-runs/vllm-serve-help.real.txt
```

### Result

The runtime help capture succeeded:

```text
docker_help_rc=0
86382 ./vnext-launch-runs/vllm-serve-help.real.txt
0 ./vnext-launch-runs/vllm-serve-help.real.err
```

All five generated vNext argv files validated against the real help output:

```text
vllm_arg_schema=valid
run_dir=./vnext-launch-runs/gguf-dense27b-tp8
vllm_arg_schema=valid
run_dir=./vnext-launch-runs/gguf-moe35b-tp4
vllm_arg_schema=valid
run_dir=./vnext-launch-runs/hf-dense27b-tp8
vllm_arg_schema=valid
run_dir=./vnext-launch-runs/hf-moe35b-tp4
vllm_arg_schema=valid
run_dir=./vnext-launch-runs/hf-moe35b-tp8
```

### Promote / Reject

Promote:

- the release-note runtime help-capture command as tested against the pinned
  runtime image on a host with `/dev/kfd` and `/dev/dri`;
- `verify_vllm_arg_schema.sh` as a real runtime-schema guard for all five
  generated profiles;
- the generated vNext argv surface as compatible with the pinned vLLM runtime.

Reject:

- treating a synthetic help fixture as sufficient final schema evidence now
  that the real runtime help command has passed;
- treating schema validity as benchmark reproduction;
- marking vNext complete before final public model inputs and serving
  benchmark reruns complete.

### Reason

The previous clean-room replay used a synthetic help fixture to validate the
schema-checking mechanics. This run exercises the actual Docker command in the
draft release note and proves that all generated profile argv files use options
recognized by the pinned runtime image. The remaining blocker is model-byte and
serving evidence: final public GGUF release assets, real pinned HF downloads,
final tag checkout, and full benchmark reruns from public inputs.

## GGUF-302 - Real Public HF Input Staging Gate

### Setup

- Date: 2026-06-29 UTC.
- Scope: test the vNext release-note HF public input path with real model bytes
  instead of synthetic fixtures.
- Temporary HF CLI:
  `<local-clean-room-root>`
- Clean model root:
  `<local-clean-room-root>`
- HF cache:
  `<local-clean-room-root>`
- Gate log:
  `<local-clean-room-root>`
- Command shape:

```sh
PATH="<local-clean-room-root>$PATH" \
HF_HOME=<local-clean-room-root> \
STAGE_HF_PUBLIC_INPUTS=1 \
  ./verify_vnext_public_hf_inputs.sh \
  <local-clean-room-root>
```

The gate used the pinned public release inputs:

- `Qwen/Qwen3.6-27B`
  at `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- `Qwen/Qwen3.6-35B-A3B`
  at `995ad96eacd98c81ed38be0c5b274b04031597b0`

### Result

The real HF public input gate passed:

```text
public-hf-profile-ok: hf-dense27b-tp8
public-hf-profile-ok: hf-moe35b-tp4
public-hf-profile-ok: hf-moe35b-tp8
vNext public HF input gate passed
```

Download/staging summary:

| Public input | Size on disk | Safetensor shards | Index file | Binary probe |
| --- | ---: | ---: | --- | --- |
| `qwen36-27b-hf` | `52G` | `15` | present | `hf_safetensors`, high confidence |
| `qwen36-35b-a3b-hf` | `67G` | `26` | present | `hf_safetensors`, high confidence |

The gate-created launch artifacts also passed artifact and real runtime-schema
validation against the pinned image's captured `vllm serve --help=all` output:

```text
real-hf-artifact-schema-ok=hf-dense27b-tp8
real-hf-artifact-schema-ok=hf-moe35b-tp4
real-hf-artifact-schema-ok=hf-moe35b-tp8
```

### Promote / Reject

Promote:

- the release-note HF staging path using `STAGE_HF_PUBLIC_INPUTS=1`;
- the pinned HF repo/revision values for Dense and MoE;
- the generated HF profile artifacts as compatible with the pinned runtime
  image's real argv schema.

Reject:

- treating HF dry-run evidence as sufficient now that the real public HF
  model-byte staging gate has passed;
- treating HF input staging as GGUF asset evidence;
- treating input staging and argv schema validation as benchmark reproduction.

### Reason

This removes one major public-input blocker for the HF profiles. Another
developer can now follow the release-note HF staging path from public upstream
repos and get valid generated artifacts for `hf-dense27b-tp8`,
`hf-moe35b-tp4`, and `hf-moe35b-tp8`. The active goal still requires final
public GGUF release assets, a final release tag checkout, and full serving
benchmark reruns from the staged public inputs before the release notes can be
called fully reproduced from scratch.

## GGUF-303 - vNext Host Platform Preflight Gate

### Setup

- Date: 2026-06-29 UTC.
- Scope: replay the host-platform portion of the vNext release-note flow on
  the current shell before attempting serving benchmarks.
- Files touched:
  - `qwen36-gfx906/check_host_platform_prereqs.sh`
  - `qwen36-gfx906/verify_vnext_serviceability.sh`
  - `releases/draft-vnext-gfx906-rocm72-gguf-hf-repro.md`
  - `qwen36-gfx906/profiles/vnext/README.md`
  - `docs/gfx906-host-platform-prereqs-v02.md`
- Method:
  - converted the read-only host preflight from Bash-only syntax to POSIX
    `/bin/sh`;
  - added the preflight to the vNext release-note and profile README flow
    before starting benchmark servers;
  - added serviceability locks requiring the preflight script to be present,
    executable, POSIX-syntax clean, and documented.

Command shape now required before serving:

```sh
for profile in \
  dense27b_tp8_fullbar_p2pon \
  moe35b_tp4_fullbar_p2pon \
  moe35b_tp8_fullbar_p2pon
do
  QWEN36_PROFILE="$profile" ./check_host_platform_prereqs.sh
done
```

### Result

The POSIX conversion and documentation locks passed:

```text
vnext-posix-syntax-ok
release-note-locks-ok
profile-doc-locks-ok
vNext developer serviceability check passed
vNext contract matrix passed
```

The current shell failed the host platform preflight for all three profile
families, as expected for a non-qualifying reproduction host:

| Profile family | Required GPUs | Visible likely GFX906 GPUs | Full-BAR GPUs | Decision |
| --- | ---: | ---: | ---: | --- |
| `dense27b_tp8_fullbar_p2pon` | 8 | 1 | 0 | reject serving benchmark on this host |
| `moe35b_tp4_fullbar_p2pon` | 4 | 1 | 0 | reject serving benchmark on this host |
| `moe35b_tp8_fullbar_p2pon` | 8 | 1 | 0 | reject serving benchmark on this host |

Each run reported the amdgpu module loaded and KFD topology visible, but the
GPU count and largest visible BAR checks failed. The preflight therefore
correctly prevented comparing this shell with the published full-BAR/P2P-on
benchmark tables.

### Promote / Reject

Promote:

- `check_host_platform_prereqs.sh` as a release-flow fail-fast gate before
  serving benchmarks;
- POSIX `/bin/sh` compatibility for the host preflight;
- serviceability locks that prevent the preflight from drifting out of the
  release-note/profile documentation.

Reject:

- attempting Dense TP8, MoE TP4, or MoE TP8 serving benchmarks on a host that
  fails the required full-BAR/P2P preflight;
- treating model staging, artifact generation, or runtime schema validation as
  sufficient to prove benchmark reproduction on a non-qualifying host;
- leaving host prerequisite checks as background prose instead of an explicit
  command in the reproduction flow.

### Reason

The goal is public reproduction, not just a successful command sequence on the
maintainer's validation lanes. A user following the release notes needs a
clear, read-only, early check that tells them whether their host is inside the
recorded full-BAR/P2P-on platform boundary before they spend time staging
large model files, loading weights, and running the benchmark ladder. This
entry does not complete the full reproduction goal; it makes the release-note
flow more accurate by rejecting an invalid serving environment earlier.

## GGUF-304 - Clean-Room vNext Replay With Host-Preflight Gate

### Setup

- Date: 2026-06-29 UTC.
- Scope: replay the updated vNext release-note command flow from a clean-room
  copy after adding the explicit host-preflight gate.
- Clean-room copy:
  `<local-clean-room-root>`
- Fixture note: the real final vNext GitHub Release assets do not exist yet.
  The GGUF split-asset mechanics were tested with tiny local `file://` fixtures
  and clean-room-only hash substitutions in `stage_vnext_gguf_assets.sh`.
  Those substitutions were not made in the working repository and are not
  public-asset proof.

### Result

The unmodified clean-room release locks passed before any fixture substitution:

```text
release-note-locks-ok
profile-doc-locks-ok
vNext developer serviceability check passed
vNext contract matrix passed
```

The local fixture asset pass exercised the public GGUF staging mechanics:

```text
vNext GGUF assets staged under: <local-clean-room-root>
public-asset-profile-ok: gguf-dense27b-tp8
public-asset-profile-ok: gguf-moe35b-tp4
vNext public GGUF asset gate passed for tag: vnext-fixture-20260629
```

Synthetic HF input validation passed for the three HF profiles:

```text
public-hf-profile-ok: hf-dense27b-tp8
public-hf-profile-ok: hf-moe35b-tp4
public-hf-profile-ok: hf-moe35b-tp8
vNext public HF input gate passed
```

Launch artifacts were generated and verified for all five profiles:

```text
vnext_launch_artifact=valid: gguf-dense27b-tp8
vnext_launch_artifact=valid: gguf-moe35b-tp4
vnext_launch_artifact=valid: hf-dense27b-tp8
vnext_launch_artifact=valid: hf-moe35b-tp4
vnext_launch_artifact=valid: hf-moe35b-tp8
```

The exact runtime help-capture command from the release note completed against
the pinned image, and all generated argv files validated against that real help
output:

```text
vllm_arg_schema=valid: gguf-dense27b-tp8
vllm_arg_schema=valid: gguf-moe35b-tp4
vllm_arg_schema=valid: hf-dense27b-tp8
vllm_arg_schema=valid: hf-moe35b-tp4
vllm_arg_schema=valid: hf-moe35b-tp8
```

The generated benchmark environments preserved the intended Qwen benchmark
shape for every profile:

```text
PRE_MEASURE_WARMUP_REQUESTS=8
PRE_MEASURE_WARMUP_MAX_TOKENS=2000
PROMPT_REPEAT=32
USE_BEGIN_THINK_PROXY=1
```

The runner itself still declares and executes:

```text
8 warmups -> c1_128 uncapped strict -> c1_2000 -> c1_10000
```

The updated host-preflight block then rejected serving on the current shell for
all three profile families:

| Profile family | Decision |
| --- | --- |
| `dense27b_tp8_fullbar_p2pon` | preflight failed; do not benchmark here |
| `moe35b_tp4_fullbar_p2pon` | preflight failed; do not benchmark here |
| `moe35b_tp8_fullbar_p2pon` | preflight failed; do not benchmark here |

### Promote / Reject

Promote:

- the updated release-note command flow through serviceability, contract
  matrix, public-input mechanics, launch-artifact generation, real runtime
  argv-schema validation, and host-preflight rejection;
- explicit preflight rejection as the correct stopping point on a
  non-qualifying host;
- the generated benchmark environment as preserving warmup count, warmup token
  budget, begin-think proxy usage, and strict/fixed ladder order.

Reject:

- treating local `file://` fixture assets as final public GGUF asset proof;
- starting Docker serving or running benchmark numbers on the current shell
  after the host-preflight failure;
- marking the vNext release notes fully reproduced before final public assets,
  final tag checkout, and full serving benchmark reruns pass on a qualifying
  full-BAR/P2P host.

### Reason

This replay proves the updated release-note flow is more accurate than the
previous version: it can proceed from clean inputs through the non-serving
contract gates, validate the generated runtime command surface against the
pinned image, and then stop before serving when the host is outside the
published platform boundary. The active goal remains open because benchmark
reproduction from scratch still requires final public GGUF assets and a
qualifying host run.

## GGUF-305 - Runtime Image and Launch-Artifact Portability Boundary

### Setup

- Date: 2026-06-29 UTC.
- Scope: test the release-note Docker image verification path and inspect the
  generated launch artifacts for public portability boundaries.
- Command:

```sh
CHECK_RUNTIME_IMAGE=1 CHECK_RUNTIME_PATHS=1 ./verify_vnext_serviceability.sh
```

### Result

The pinned runtime image and native runtime-path check passed:

```text
runtime-image-digest-ok: joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b@sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e
runtime-native-paths-ok: joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b
vNext developer serviceability check passed
```

Inspection of the clean-room launch artifacts showed:

- `effective_config.env` records local preflight paths such as the model probe
  path and patch-bundle source used to generate that artifact.
- `docker_run.sh` still requires user-provided `HOST_MODEL_ROOT`,
  `HOST_HF_CACHE`, and `HOST_RUNTIME_ROOT` values at launch time.
- the generated vLLM command uses container paths under `/opt/local-models` for
  model inputs.
- the generated container entrypoint can create a container-internal native
  extension path under `/usr/share/ollama/kernel_labs/...` from the mounted
  patch bundle.

This exposed a wording gap: the docs correctly reject validation-host storage
roots as public inputs, but they also need to explain that container-internal
runtime paths created from the release patch bundle are legitimate and are not
host validation-workspace dependencies.

### Promote / Reject

Promote:

- the pinned runtime image and digest as verified by Docker;
- `CHECK_RUNTIME_IMAGE=1 CHECK_RUNTIME_PATHS=1` as a release-note validation
  command that proves the expected native runtime paths exist in the public
  image;
- generated launch artifacts as local, auditable outputs rather than release
  assets to copy between hosts;
- documentation that separates container-internal runtime paths from forbidden
  host validation-workspace paths.

Reject:

- treating `effective_config.env` local probe paths as reusable public release
  paths;
- treating container-internal `/usr/share/ollama/kernel_labs/...` paths as
  evidence of a host dependency;
- publishing wording that bans all visible `/usr/share/ollama` strings without
  explaining the container-runtime distinction.

### Reason

The public reproduction path must not depend on LocalAIServers validation-host
storage, but the runtime image and generated entrypoint still need stable
container-internal paths for native extension placement. The release note and
profile README now state that distinction explicitly, and
`verify_vnext_serviceability.sh` locks the wording so it does not regress.

## GGUF-306 - vNext Release-Readiness Verifier

### Setup

- Date: 2026-06-29 UTC.
- Scope: make the draft vNext release-note reproduction path mechanically
  auditable instead of relying on manual sequencing.
- Added tool:
  `qwen36-gfx906/verify_vnext_release_readiness.sh`.

### Result

The verifier sequences the public release-note path:

```text
serviceability
-> contract matrix
-> public GGUF asset staging
-> public HF input staging/validation
-> launch artifact generation
-> runtime vLLM argument-schema validation
-> host platform preflight
-> optional serving benchmark ladder
```

The script requires a final release tag, user-selected model root, HF cache,
and runtime root. It rejects placeholder tags and unsafe broad roots. It exits
nonzero before benchmark launch if public assets are unavailable or if the host
fails the full-BAR/P2P-on preflight. It also refuses to report full release
reproduction success unless `RUN_SERVING_BENCHMARKS=1` is set; non-serving
dry runs require explicit `ALLOW_PRE_SERVING_ONLY=1`.

### Promote / Reject

Promote:

- `verify_vnext_release_readiness.sh` as the release-note sequencing gate;
- `RUN_SERVING_BENCHMARKS=1` as the explicit switch for full reproduction
  success;
- `ALLOW_PRE_SERVING_ONLY=1` only as a maintainer dry-run mode;
- documentation that says final vNext publication still requires real GitHub
  Release asset URLs and a qualifying full-BAR/P2P host benchmark run.

Reject:

- treating serviceability, schema validation, or clean-room fixture staging as
  full benchmark reproduction;
- treating `file://` asset preflight as a public release input;
- treating a non-qualifying host preflight failure as comparable with the
  recorded `.20` / `.30` benchmark lanes.

### Reason

The active goal is to follow the release notes from scratch and reproduce the
results. A single verifier makes that goal testable: it follows the same
sequence documented for users and stops at the first missing public asset,
invalid generated artifact, mismatched runtime schema, host-platform failure,
or omitted serving benchmark ladder.

## GGUF-307 - vNext Local Asset Lock Audit

### Setup

- Date: 2026-06-29 UTC.
- Scope: inspect local maintainer-side artifacts before attempting another
  release-note replay.
- Checked:
  - full Dense GGUF F16 file;
  - full MoE GGUF F16 file;
  - Dense text-config/tokenizer archives and extracted directories.

### Result

The local full GGUF files match the vNext locks:

```text
Qwen3.6-27B-F16.gguf
b2347376b9bb7d12cf5f1d31c53ac4c60dd3d4b95068a09351187b328b5e9d89

Qwen3.6-35B-A3B-F16.gguf
1f2443bb0ff958943d091410c61120c181a0579b3bc85192029aa51d821d141c
```

The local Dense text-config/tokenizer candidates found during this audit were
fixture or intermediate variants. None matched the current locked archive hash
or config hash used by `stage_vnext_gguf_assets.sh`:

```text
Qwen3.6-27B-text-config-eosfix.tar.gz
9a85f0a18a012ed37f0eb4c42549569234af880c5f6e3ce908e0e45cca835719

qwen36-27b-text-config-eosfix/config.json
370a0c6b21a09e288bd3301581e126956d1b2dd410e791920884c3130bfe6e0a
```

### Promote / Reject

Promote:

- the full GGUF file locks as locally verified;
- the release-note warning that final GGUF Dense publication still needs the
  exact Dense text-config/tokenizer asset or a reproducible public generation
  recipe.

Reject:

- using tiny text-config fixtures as public release assets;
- treating a local full-GGUF mirror as sufficient public reproduction evidence;
- publishing the GGUF Dense path until the text-config/tokenizer asset gate can
  pass without fixture hash substitution.

### Reason

The active release-note replay cannot pass the unmodified public GGUF asset
gate without the exact Dense text-config/tokenizer archive. The correct next
step is either to recover/build and attach that locked asset or replace the
asset requirement with a deterministic public generation recipe from the pinned
`Qwen/Qwen3.6-27B` revision.

## GGUF-308 - Dense Text-Config Public Reconstruction Probe

### Setup

- Date: 2026-06-29 UTC.
- Scope: test whether the missing Dense GGUF text-config asset can be
  reconstructed deterministically from the pinned public
  `Qwen/Qwen3.6-27B` config instead of recovered as an archive.
- Public input checked:
  `Qwen/Qwen3.6-27B` revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Target locks:

```text
Qwen3.6-27B-text-config-eosfix.tar.gz
9a85f0a18a012ed37f0eb4c42549569234af880c5f6e3ce908e0e45cca835719

qwen36-27b-text-config-eosfix/config.json
370a0c6b21a09e288bd3301581e126956d1b2dd410e791920884c3130bfe6e0a
```

### Result

The pinned public HF `config.json` was available and had raw SHA256:

```text
69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c
```

Candidate JSON variants were generated from the public `text_config`,
including:

- exact `text_config` with sorted and unsorted compact JSON;
- pretty-printed variants;
- `model_type` variants for `qwen3_5_text`, `qwen3_5`, and `qwen3`;
- architecture overrides for `Qwen3_5ForCausalLM` and `Qwen3ForCausalLM`;
- dtype variants for `bfloat16`, `float16`, `half`, and omitted dtype;
- top-level wrapped variants with `language_model_only=true`;
- plain-RoPE variants with `rope_parameters` and related rotary keys removed.

None of the `97` candidate configs matched the locked config hash
`370a0c6b21a09e288bd3301581e126956d1b2dd410e791920884c3130bfe6e0a`.

Representative non-matching hashes:

```text
Qwen3_5ForCausalLM / qwen3_5 / float16
edc0262c24efafb3e1aebbca50e952919f07d45d7dad6559800337fdd05c2fde

Qwen3_5ForCausalLM / qwen3_5 / original dtype
4e90b3d7288f9e233c964ad7b3ef790026308329c7970fce1cb1d67d52ec9ab5

Qwen3_5ForCausalLM / qwen3_5 / rope_parameters stripped
71bd042c9d958ea49e5deb55121e502094706fcb534c99c15d33ce0bb95b8052
```

A follow-up filename-based recovery pass over likely local vNext staging,
clean-room, replay, and temporary cache roots found `66` candidate
`Qwen3.6-27B-text-config-eosfix.tar.gz` or
`qwen36-27b-text-config-eosfix/config.json` files. None matched the current
archive lock and none matched the current config lock.

### Promote / Reject

Promote:

- the conclusion that the current locked Dense text-config archive is not
  trivially reproducible from obvious public-HF `text_config` transformations;
- the conclusion that the likely local staging/replay candidate set does not
  contain the current locked archive/config pair;
- the vNext draft release guard that a regenerated config must not be
  substituted silently.

Reject:

- replacing the locked Dense text-config asset with any unbenchmarked generated
  variant;
- publishing the GGUF Dense result as portable until the exact locked archive
  is recovered and attached, or a new deterministic public generation recipe is
  benchmarked through the full ladder and documented as the stand-alone vNext
  release input.

### Reason

vNext is a stand-alone release. It cannot depend on validation-host caches or
unstated local transformations. The GGUF Dense benchmark path needs either the
exact text-config/tokenizer archive that produced the existing evidence, or a
new public generation recipe whose output is rerun through the normal
`8` warmups -> `c1_128` uncapped strict -> `c1_2000` -> `c1_10000` ladder
before publication.

## GGUF-309 - vNext Readiness Replay Against Unpublished Tag

### Setup

- Date: 2026-06-29 UTC.
- Scope: replay the documented vNext readiness verifier from the release-note
  flow using temporary local roots and an intentionally unpublished tag.
- Command shape:

```sh
cd qwen36-gfx906
CHECK_RUNTIME_IMAGE=0 \
CHECK_RUNTIME_PATHS=0 \
CAPTURE_VLLM_HELP=0 \
ALLOW_PRE_SERVING_ONLY=1 \
./verify_vnext_release_readiness.sh \
  --release-tag vnext-readiness-replay-not-published \
  --model-root <temporary-model-root> \
  --hf-cache <temporary-hf-cache> \
  --runtime-root <temporary-runtime-root>
```

The Docker image and runtime-help capture were skipped for this replay because
the target was release-note sequencing and public asset failure behavior, not a
GPU serving run.

### Result

The readiness verifier ran the documented gates in order:

```text
serviceability: pass
contract matrix: pass
public GGUF assets: fail
```

The public GGUF asset gate attempted to fetch assets for the supplied tag and
received a `404`. The verifier then exited nonzero with the intended
actionable error:

```text
public GGUF asset gate failed; check release tag, release assets, part manifests, and final SHA256 locks
```

No HF staging, launch-artifact generation, runtime schema capture, host
preflight, or serving benchmark ladder was run after the public asset gate
failed.

### Promote / Reject

Promote:

- `verify_vnext_release_readiness.sh` as the documented sequence gate;
- the current fail-closed behavior for unpublished/missing public GGUF release
  assets;
- the release-note statement that final vNext publication requires final
  GitHub Release asset URLs and a qualifying full-BAR/P2P host benchmark run.

Reject:

- treating a serviceability plus contract-matrix pass as release reproduction;
- bypassing the public GGUF asset gate with retained local model files;
- calling the vNext release notes reproduced before the final release tag and
  public assets exist.

### Reason

This replay followed the release-note path far enough to prove that the
stand-alone vNext release currently stops at the correct boundary. The result
is not a full reproduction, but it is a useful release-readiness check: the
published instructions will not silently proceed from private caches or a
placeholder release tag.

## GGUF-310 - Dense Text-Config Archive Candidate Audit

### Setup

- Date: 2026-06-29 UTC.
- Scope: determine whether older clean-room or replay text-config archives
  recovered from local staging evidence satisfy the current vNext Dense
  text-config lock.
- Current vNext locks:
  - Dense text-config archive:
    `9a85f0a18a012ed37f0eb4c42549569234af880c5f6e3ce908e0e45cca835719`
  - Dense text-config `config.json`:
    `370a0c6b21a09e288bd3301581e126956d1b2dd410e791920884c3130bfe6e0a`

### Result

Two older clean-room/replay archive candidates were inspected. Neither matched
the current vNext lock:

| Candidate role | Archive SHA256 | Extracted `config.json` SHA256 | Notes |
| --- | --- | --- | --- |
| fixture-era clean-room archive | `97079fb1618807c5f9c8c681796f295b062f5b531ca4d9ea942181302fac105f` | `a87cd840493f5dc058668d70a2fef679449b820dd0e88a641566a40c17cbe82d` | Created during a fixture hash-substitution flow. |
| replay archive | `3b6e1b4c078910e5b5c4840d43d581443e4645a66dcb10d59736c75d19787acb` | `7c87e49055ec9c5772ba2159b85c3f64f2f3c529ee9568b06068769f04e96745` | Created during a local replay flow. |

The fixture-era archive used a tiny placeholder tokenizer/config path and the
replay archive used a different minimal config. These archives are useful for
understanding how the split-asset checks were exercised, but they are not the
current Dense text-config archive and cannot be promoted as public vNext
release assets.

### Promote / Reject

Promote:

- the conclusion that historical clean-room/replay archive candidates do not
  satisfy the current vNext text-config lock;
- the release-note warning that `file://` staging, clean-room simulations, and
  fixture hash substitutions are maintainer evidence only;
- the requirement that vNext remain a standalone release whose GGUF Dense path
  is reproduced from final public assets or a documented public generation
  recipe.

Reject:

- treating the older clean-room/replay archives as substitutes for the current
  locked Dense text-config archive;
- citing the existing GGUF Dense benchmark table as a final portable public
  reproduction claim before the Dense text-config public-input gap is closed;
- changing the current text-config lock without rerunning the normal
  `8` warmups -> `c1_128` uncapped strict -> `c1_2000` -> `c1_10000`
  benchmark ladder.

### Reason

vNext is a standalone release. It cannot inherit trust from v0.2.1, internal
validation hosts, retained caches, or temporary fixture substitutions. The
release draft should keep the strong benchmark evidence, but the public claim
boundary remains blocked until the exact locked Dense text-config archive is
published or a deterministic public generation recipe replaces it and is
benchmarked end to end.

## GGUF-311 - vNext Release-Note Readiness Replay After Standalone Boundary Tightening

### Setup

- Date: 2026-06-29 UTC.
- Scope: replay the documented vNext release-readiness sequence after tightening
  the release-note boundary around standalone public inputs.
- Command shape:

```sh
cd qwen36-gfx906
CHECK_RUNTIME_IMAGE=0 \
CHECK_RUNTIME_PATHS=0 \
CAPTURE_VLLM_HELP=0 \
ALLOW_PRE_SERVING_ONLY=1 \
./verify_vnext_release_readiness.sh \
  --release-tag vnext-readiness-replay-not-published \
  --model-root <temporary-model-root> \
  --hf-cache <temporary-hf-cache> \
  --runtime-root <temporary-runtime-root>
```

Runtime image/path checks and runtime vLLM help capture were disabled because
this was a local non-serving replay of the release-note gate order, not a
qualifying GFX906 benchmark run.

### Result

The readiness verifier ran the expected gates:

```text
serviceability: pass
contract matrix: pass
public GGUF assets: fail
```

The public GGUF asset gate attempted to fetch assets for the intentionally
unpublished tag and received a `404`. The verifier exited with the expected
nonzero error:

```text
public GGUF asset gate failed; check release tag, release assets, part manifests, and final SHA256 locks
```

No HF staging, launch-artifact generation, runtime schema capture, host
preflight, or serving benchmark ladder ran after the public asset gate failed.

### Promote / Reject

Promote:

- the current release-note sequence as fail-closed when the standalone vNext
  public assets do not exist;
- the serviceability and contract matrix checks as passing non-serving gates;
- the draft wording that final vNext publication requires final GitHub Release
  asset URLs and a qualifying full-BAR/P2P host benchmark run.

Reject:

- treating this replay as a full reproduction;
- bypassing the public GGUF asset gate with private local model files;
- publishing vNext before the final standalone tag, release assets, Dense
  text-config public input, and full serving benchmark replay are in place.

### Reason

This replay proves that the documented release path stops at the correct
boundary today. The failure is expected and useful: it prevents a release-note
reader from accidentally reproducing from internal caches or unpublished local
assets while thinking they followed the public vNext package.

## GGUF-312 - Direct GGUF Staging Local-Asset Guard

### Setup

- Date: 2026-06-29 UTC.
- Scope: verify that both the public GGUF asset wrapper and the lower-level
  staging script reject `file://` asset bases unless the maintainer explicitly
  opts into local preflight.
- Commands tested:

```sh
RELEASE_ASSET_BASE=file://<temporary-assets> \
  ./verify_vnext_public_assets.sh vnext-file-guard-test <temporary-model-root>

RELEASE_ASSET_BASE=file://<temporary-assets> \
  ./stage_vnext_gguf_assets.sh vnext-file-guard-test <temporary-model-root>
```

### Result

Before this pass, `verify_vnext_public_assets.sh` rejected the local asset base
with:

```text
file:// RELEASE_ASSET_BASE requires ALLOW_LOCAL_ASSET_PREFLIGHT=1
```

The lower-level `stage_vnext_gguf_assets.sh` did not reject it first; it handed
the `file://` path to `curl`, which then failed while trying to open the local
file. The staging script was updated so direct use has the same explicit guard
as the wrapper. The vNext release draft, profile README, and serviceability
lock were also updated so the guard stays visible and mechanically checked.

### Promote / Reject

Promote:

- requiring `ALLOW_LOCAL_ASSET_PREFLIGHT=1` for every `file://`
  `RELEASE_ASSET_BASE` path, including direct calls to
  `stage_vnext_gguf_assets.sh`;
- locking that requirement in `verify_vnext_serviceability.sh`;
- treating `file://` as a maintainer preflight hook only, never as a public
  reproduction dependency.

Reject:

- relying on the wrapper alone to protect public-release instructions;
- allowing a direct staging-script invocation to silently try private local
  files.

### Reason

The draft release notes expose both the public asset gate and the staging
script. A user following the notes directly must get the same fail-closed
behavior from either entry point. vNext is standalone and must not depend on
internal asset directories or retained local files.

## GGUF-313 - Direct GGUF Staging Placeholder and Unsafe-Root Guard

### Setup

- Date: 2026-06-29 UTC.
- Scope: test whether direct calls to `stage_vnext_gguf_assets.sh` reject
  placeholder release tags and unsafe model roots before network or filesystem
  side effects.
- Commands tested:

```sh
./stage_vnext_gguf_assets.sh '<release-tag>' <temporary-model-root>
./stage_vnext_gguf_assets.sh vnext-stage-guard-test /
./stage_vnext_gguf_assets.sh vnext-stage-guard-test /home
```

### Result

Before this pass, all three direct staging calls attempted a release-asset
download and failed later with a `404`. The wrapper and release-readiness
scripts already had stricter guards, but the release notes expose direct
staging as a documented command.

The staging script was updated to fail before download with explicit errors:

```text
release tag must be the final published tag, not a placeholder
local model root is unsafe or too broad
```

The release draft, profile README, and serviceability lock were updated to keep
that behavior visible and enforced.

### Promote / Reject

Promote:

- direct staging rejecting placeholder tags before any download attempt;
- direct staging rejecting broad roots such as `/` and `/home`;
- locking the direct-staging guards in `verify_vnext_serviceability.sh`.

Reject:

- relying on `curl`/404 failures as placeholder-tag validation;
- letting a direct staging command create or overwrite release asset files under
  an unsafe broad root.

### Reason

The public reproduction flow must be robust when followed literally from the
release notes. Direct staging should fail for bad inputs at the same boundary
as the wrapper and readiness verifier, not by accidentally touching the network
or broad filesystem paths first.

## GGUF-314 - HF Public-Input Verifier Negative Guards

### Setup

- Date: 2026-06-29 UTC.
- Scope: test the HF side of the vNext release-note flow for fail-closed
  behavior before public downloads or launch-artifact generation.
- Commands tested:

```sh
./verify_vnext_public_hf_inputs.sh /
./verify_vnext_public_hf_inputs.sh /home
./verify_vnext_public_hf_inputs.sh <temporary-empty-model-root>
STAGE_HF_PUBLIC_INPUTS=2 ./verify_vnext_public_hf_inputs.sh <temporary-model-root>
HF_DENSE_REVISION=bad ./verify_vnext_public_hf_inputs.sh <temporary-model-root>
```

### Result

The existing verifier already rejected:

```text
local model root is unsafe or too broad
missing Dense HF config.json
STAGE_HF_PUBLIC_INPUTS must be 0 or 1
HF_DENSE_REVISION does not match the pinned release value
```

The verifier was then tightened so `STAGE_HF_PUBLIC_INPUTS=1` also validates
the selected `HF_HOME` cache root before invoking `hf download`. Unsafe cache
roots such as `/` and `/home` are now rejected before any download attempt.
The release draft, profile README, and serviceability lock were updated to keep
that behavior visible and enforced.

### Promote / Reject

Promote:

- the HF verifier as the public HF input gate;
- pinned HF revision rejection for Dense and MoE profiles;
- explicit unsafe model-root and unsafe HF-cache-root rejection;
- treating missing staged HF configs as an actionable pre-launch failure.

Reject:

- running HF profile launch artifacts from empty or private model roots;
- allowing broad HF cache roots during staged public downloads;
- treating floating HF revisions as equivalent reproduction inputs.

### Reason

The standalone vNext release must be reproducible from public Hugging Face
inputs and user-selected local storage. The HF verifier should fail before
network/download or launch work when the user gives an unsafe root, a bad
revision, or an empty staging directory.

## GGUF-315 - Stand-Alone vNext Readiness Guard Replay

### Setup

- Date: 2026-06-29 UTC.
- Scope: confirm the current draft treats vNext as a stand-alone release, not a
  v0.2.1 patch/addendum, and replay the non-serving readiness gates after the
  direct GGUF staging and HF cache-root guard updates.
- Commands tested:

```sh
STAGE_HF_PUBLIC_INPUTS=1 HF_HOME=/ \
  ./verify_vnext_public_hf_inputs.sh <temporary-model-root>

STAGE_HF_PUBLIC_INPUTS=1 HF_HOME=/home \
  ./verify_vnext_public_hf_inputs.sh <temporary-model-root>

./verify_vnext_serviceability.sh

./verify_vnext_contract_matrix.sh

CHECK_RUNTIME_IMAGE=0 CHECK_RUNTIME_PATHS=0 CAPTURE_VLLM_HELP=0 \
ALLOW_PRE_SERVING_ONLY=1 \
  ./verify_vnext_release_readiness.sh \
    --release-tag vnext-unpublished-local-dryrun \
    --model-root <temporary-model-root> \
    --hf-cache <temporary-hf-cache> \
    --runtime-root <temporary-runtime-root>
```

### Result

The HF public-input verifier rejected broad cache roots before invoking
`hf download`:

```text
HF cache root is unsafe or too broad: /
HF cache root is unsafe or too broad: /home
```

The serviceability gate passed with the stand-alone vNext release locks in
place:

```text
release-note-locks-ok
profile-doc-locks-ok
serviceable: gguf-dense27b-tp8
serviceable: gguf-moe35b-tp4
serviceable: hf-dense27b-tp8
serviceable: hf-moe35b-tp4
serviceable: hf-moe35b-tp8
vNext developer serviceability check passed
```

The CPU-only contract matrix passed, including the expected fail-closed cases
for GGUF/HF format mismatch, Dense/MoE family mismatch, HF GGUF-env leakage,
and patch-bundle hash mismatch.

The top-level readiness driver passed serviceability and contract-matrix
stages, then failed at the public GGUF asset gate with a `404` for the
unpublished dry-run tag. That is the expected result before a final vNext tag
and public GGUF/text-config release assets exist.

### Promote / Reject

Promote:

- vNext wording as a stand-alone release package with its own release tag,
  assets, portability gate, reproduction instructions, validation tables, and
  claim boundary;
- HF cache-root guards for public `hf download` staging;
- the top-level readiness driver as the correct gate order:
  serviceability -> contract matrix -> public GGUF asset gate -> public HF gate
  -> launch artifacts -> schema -> host preflight -> optional serving ladder.

Reject:

- describing vNext as a v0.2.1 patch, addendum, or silent replacement;
- publishing before the final public GGUF split assets and Dense text-config
  input are attached or replaced by an exact public generation recipe;
- treating a pre-serving dry run as benchmark reproduction success.

### Reason

vNext must be public and portable from its own release inputs. The current
non-serving checks now enforce that boundary locally. Full promotion still
requires final public assets and a serving benchmark replay from the published
vNext tag with `RUN_SERVING_BENCHMARKS=1` on a qualifying full-BAR/P2P-on
GFX906 host.

## GGUF-316 - Literal Release-Readiness Tag Boundary Replay

### Setup

- Date: 2026-06-29 UTC.
- Scope: follow the draft release-note readiness command literally enough to
  confirm the user-facing tag substitution and unpublished-asset boundaries.
- Commands tested:

```sh
CHECK_RUNTIME_IMAGE=0 CHECK_RUNTIME_PATHS=0 CAPTURE_VLLM_HELP=0 \
ALLOW_PRE_SERVING_ONLY=1 \
  ./verify_vnext_release_readiness.sh \
    --release-tag '<release-tag>' \
    --model-root <temporary-model-root> \
    --hf-cache <temporary-hf-cache> \
    --runtime-root <temporary-runtime-root>

CHECK_RUNTIME_IMAGE=0 CHECK_RUNTIME_PATHS=0 CAPTURE_VLLM_HELP=0 \
ALLOW_PRE_SERVING_ONLY=1 \
  ./verify_vnext_release_readiness.sh \
    --release-tag vnext-unpublished-local-dryrun \
    --model-root <temporary-model-root> \
    --hf-cache <temporary-hf-cache> \
    --runtime-root <temporary-runtime-root>
```

### Result

The literal placeholder tag failed before any staging work:

```text
release tag must be the final published tag, not a placeholder
```

After substituting a final-looking but unpublished dry-run tag, the readiness
driver passed serviceability and contract-matrix checks, then stopped at the
public GGUF asset gate with a release-asset `404`.

### Promote / Reject

Promote:

- keeping `<release-tag>` visibly rejected until a real final vNext tag exists;
- preserving the public GGUF asset gate as the first hard publication blocker
  after local serviceability and contract checks.

Reject:

- allowing placeholder release tags to proceed into network or filesystem
  staging;
- treating an unpublished dry-run tag as a substitute for final GitHub Release
  assets.

### Reason

The release-note path should be safe when followed literally by a reader. A
placeholder must fail early with a clear fix, and an unpublished tag must not
fall back to local caches or validation-host paths.

## GGUF-317 - Clean-Copy Release-Note Replay

### Setup

- Date: 2026-06-29 UTC.
- Scope: copy the current candidate tree to a temporary clean directory without
  `.git`, without a prebuilt `model-format-probe` binary, and without generated
  launch runs; then follow the draft release-note commands through the
  non-serving gates.
- Clean copy location during the test:
  `/tmp/vnext-clean-copy.<temporary>/localaiservers`.
- Commands tested:

```sh
make -C tools/model-format-probe

cd qwen36-gfx906
./vnext_profile_inspect.sh list
./vnext_profile_inspect.sh show gguf-dense27b-tp8
./verify_vnext_serviceability.sh
./verify_vnext_contract_matrix.sh

CHECK_RUNTIME_IMAGE=0 CHECK_RUNTIME_PATHS=0 CAPTURE_VLLM_HELP=0 \
ALLOW_PRE_SERVING_ONLY=1 \
  ./verify_vnext_release_readiness.sh \
    --release-tag '<release-tag>' \
    --model-root <temporary-model-root> \
    --hf-cache <temporary-hf-cache> \
    --runtime-root <temporary-runtime-root>

CHECK_RUNTIME_IMAGE=0 CHECK_RUNTIME_PATHS=0 CAPTURE_VLLM_HELP=0 \
ALLOW_PRE_SERVING_ONLY=1 \
  ./verify_vnext_release_readiness.sh \
    --release-tag vnext-unpublished-local-dryrun \
    --model-root <temporary-model-root> \
    --hf-cache <temporary-hf-cache> \
    --runtime-root <temporary-runtime-root>

STAGE_HF_PUBLIC_INPUTS=1 HF_HOME=/ \
  ./verify_vnext_public_hf_inputs.sh <temporary-model-root>
```

### Result

The clean copy rebuilt `tools/model-format-probe/model-format-probe` from
source, listed all five vNext profiles, and inspected `gguf-dense27b-tp8`
without using generated artifacts from the working tree.

The serviceability and contract-matrix checks passed from the clean copy:

```text
vNext developer serviceability check passed
vNext contract matrix passed
```

The placeholder release tag failed before staging:

```text
release tag must be the final published tag, not a placeholder
```

The final-looking unpublished dry-run tag passed serviceability and
contract-matrix stages, then stopped at the public GGUF asset gate with a
release-asset `404`, as expected before final vNext release assets exist.

The HF public-input verifier failed closed from the clean copy for unsafe model
roots, unsafe HF cache roots, missing staged configs, invalid
`STAGE_HF_PUBLIC_INPUTS`, and non-pinned Dense/MoE revisions.

### Promote / Reject

Promote:

- the current release-note command sequence as mechanically runnable from a
  clean candidate checkout through the non-serving gates;
- rebuilding the binary model-format probe from source as part of the public
  path;
- keeping the missing public GGUF assets as the current first hard blocker for
  full vNext reproduction.

Reject:

- claiming full release reproduction from these clean-copy checks alone;
- treating local working-tree generated binaries, launch artifacts, model
  caches, or unpublished release assets as public reproduction inputs.

### Reason

This clean-copy replay is closer to the release user's path than running from
the active development tree. It proves the candidate instructions do not need a
prebuilt local probe binary or `.git` metadata for the pre-serving gates. It
does not prove final release reproduction because final public assets and the
serving benchmark ladder from the published tag are still required.

## GGUF-318 - Direct GGUF Staging and Host-Preflight Replay

### Setup

- Date: 2026-06-29 UTC.
- Scope: test the direct GGUF staging helpers and host-preflight command blocks
  that are documented separately from the top-level readiness wrapper.
- Commands tested:

```sh
./stage_vnext_gguf_assets.sh '<release-tag>' <temporary-model-root>
./stage_vnext_gguf_assets.sh vnext-direct-gate /
./stage_vnext_gguf_assets.sh vnext-direct-gate /home
RELEASE_ASSET_BASE=file://<temporary-assets> \
  ./stage_vnext_gguf_assets.sh vnext-direct-gate <temporary-model-root>
./stage_vnext_gguf_assets.sh vnext-unpublished-local-dryrun <temporary-model-root>

./verify_vnext_public_assets.sh '<release-tag>' <temporary-model-root>
./verify_vnext_public_assets.sh vnext-direct-gate /
RELEASE_ASSET_BASE=file://<temporary-assets> \
  ./verify_vnext_public_assets.sh vnext-direct-gate <temporary-model-root>
./verify_vnext_public_assets.sh vnext-unpublished-local-dryrun <temporary-model-root>

QWEN36_PROFILE=bad_profile ./check_host_platform_prereqs.sh
QWEN36_PROFILE=dense27b_tp8_fullbar_p2pon ./check_host_platform_prereqs.sh
QWEN36_PROFILE=moe35b_tp4_fullbar_p2pon ./check_host_platform_prereqs.sh
QWEN36_PROFILE=moe35b_tp8_fullbar_p2pon ./check_host_platform_prereqs.sh
```

### Result

Direct GGUF staging rejected unsafe or non-public inputs before attempting a
download:

```text
release tag must be the final published tag, not a placeholder
local model root is unsafe or too broad: /
local model root is unsafe or too broad: /home
file:// RELEASE_ASSET_BASE requires ALLOW_LOCAL_ASSET_PREFLIGHT=1
```

The direct public-asset verifier produced the same early rejection for
placeholder tags, unsafe roots, and unapproved `file://` asset bases.

A syntactically valid but unpublished dry-run tag reached GitHub asset
download and failed with curl `404`, which is expected before final vNext
GitHub Release assets exist.

The host preflight rejected an unknown profile with the valid profile list. On
the current non-qualifying local host, all real profile families failed before
benchmark work because only one likely gfx906 PCI device was visible and its
largest BAR was `0.25` GiB, below the required full-BAR boundary.

The preflight output was corrected from the narrow phrase "published v0.2
numbers" to the release-generic "published full-BAR/P2P release numbers" so
the same helper reads correctly from the standalone vNext release notes.

### Promote / Reject

Promote:

- direct staging guards for placeholder tags, unsafe model roots, and
  unapproved local asset mirrors;
- host preflight as a read-only gate before any serving benchmark comparison;
- release-generic host-preflight wording for both v0.2 and vNext full-BAR/P2P
  claims.

Reject:

- using `file://` release assets as public reproduction inputs;
- treating curl `404` from an unpublished tag as evidence of a valid release
  asset set;
- running or comparing benchmarks after a full-BAR/P2P preflight failure.

### Reason

The release notes expose direct commands as well as the top-level readiness
wrapper. Those direct commands must fail at clear boundaries for unsafe input,
and the host preflight must tell users that a failing host is outside the
published full-BAR/P2P release comparison boundary.

## GGUF-319 - Runtime Image Digest, Native Paths, and vLLM Schema Replay

### Setup

- Date: 2026-06-29 UTC.
- Scope: test the runtime-image portions of the draft release-note path against
  the public Docker image and validate generated profile argv against the
  image's own `vllm serve --help=all` output.
- Commands tested:

```sh
CHECK_RUNTIME_IMAGE=1 CHECK_RUNTIME_PATHS=0 \
  ./verify_vnext_serviceability.sh

CHECK_RUNTIME_IMAGE=1 CHECK_RUNTIME_PATHS=1 \
  ./verify_vnext_serviceability.sh

docker run --rm \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --security-opt seccomp=unconfined \
  --security-opt label=disable \
  --entrypoint vllm \
  joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b \
  serve --help=all > <temporary-vllm-serve-help.txt>

./verify_vllm_arg_schema.sh <generated-profile-run-dir> \
  <temporary-vllm-serve-help.txt>
```

Synthetic local model signatures were used only to generate launch artifacts
for schema validation. No model server or benchmark workload was started.

### Result

The public Docker tag resolved to the pinned manifest digest:

```text
runtime-image-digest-ok: joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b@sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e
```

The native runtime path check passed inside the image:

```text
runtime-native-paths-ok: joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b
```

The documented `vllm serve --help=all` capture succeeded with GPU devices
exposed and produced a runtime schema file. Generated argv for all five vNext
profiles validated against that schema:

```text
schema-ok: gguf-dense27b-tp8
schema-ok: gguf-moe35b-tp4
schema-ok: hf-dense27b-tp8
schema-ok: hf-moe35b-tp4
schema-ok: hf-moe35b-tp8
```

The generated help file and synthetic launch artifacts were removed after the
check.

### Promote / Reject

Promote:

- the public Docker tag/digest pair as mechanically verifiable from the current
  release-note path;
- runtime native-path verification as a useful pre-benchmark guard;
- runtime argv-schema validation against the image's own `vllm serve --help=all`
  output for every generated profile.

Reject:

- assuming vLLM CLI compatibility from static docs alone;
- treating schema validation as serving benchmark reproduction;
- retaining generated help files or synthetic launch artifacts in the repo.

### Reason

The release notes tell users to validate the runtime image and generated argv
before launching benchmark servers. This replay confirms that the public image
identity, native runtime paths, and generated profile argv are internally
consistent on the current candidate path. Full release reproduction still
requires real public model assets and the serving benchmark ladder on a
qualifying full-BAR/P2P-on host.

## GGUF-320 - Benchmark Runner Fixture Ladder Replay

### Setup

- Date: 2026-06-29 UTC.
- Scope: test the vNext benchmark runner's observable ladder order and summary
  output against a temporary local OpenAI/vLLM-style fixture endpoint.
- Fixture behavior:
  - `GET /v1/models` returned ready.
  - `POST /v1/chat/completions` returned a Qwen-style response with a
    `reasoning_content` field, visible answer text, `finish_reason=stop`, and
    usage counters.
  - `GET /metrics` returned vLLM-compatible prompt/generation/prefill/decode
    counters that advanced once per request.
- Commands tested:

```sh
./run_vnext_profile_benchmark.sh <temporary-launch-run-dir>
```

The temporary `benchmark_env.sh` selected
`dense27b_tp8_fullbar_p2pon`, localhost fixture endpoints, short timeouts, and
the default begin-think proxy path. No model server, Docker container, or GPU
workload was started.

### Result

The runner completed successfully against the fixture and sent exactly `11`
chat requests:

1. eight warmups with `max_tokens=2000`, `min_tokens=2000`, and
   `ignore_eos=true`;
2. one uncapped strict request with no `max_tokens`, no `min_tokens`, and no
   `ignore_eos`;
3. one fixed-token `c1_2000` request with `max_tokens=2000`,
   `min_tokens=2000`, and `ignore_eos=true`;
4. one fixed-token `c1_10000` request with `max_tokens=10000`,
   `min_tokens=10000`, and `ignore_eos=true`.

All requests used the expected model id for the dense profile and
`prompt_repeat=32`. The begin-think proxy path was exercised and the strict
request passed the Qwen gate through the parser-split reasoning/content shape.

The first fixture run showed a user-facing summary issue: fixed-token rows
reported `strict gate valid=False` because capped requests intentionally fail
strict-gate semantics. That was accurate raw JSON but misleading in the
Markdown summary. The runner summary was updated to label that column
`strict gate status` and render fixed-token rows as `n/a (fixed-token)` while
leaving each tier's raw `summary.json` unchanged.

The rerun produced:

```text
| `c1_128_strict` | 128 | ... | ... | ... | stop | True |
| `c1_2000` | 2000 | ... | ... | ... | stop | n/a (fixed-token) |
| `c1_10000` | 10000 | ... | ... | ... | stop | n/a (fixed-token) |
```

### Promote / Reject

Promote:

- the vNext benchmark runner's default request ladder:
  `8` warmups -> uncapped strict -> `c1_2000` -> `c1_10000`;
- the begin-think proxy path for Qwen strict-gate validation;
- explicit `n/a (fixed-token)` summary status for non-strict capped tiers.

Reject:

- reading capped fixed-token rows as failed strict-gate evidence;
- treating fixture throughput numbers as benchmark results;
- using this fixture replay as a substitute for real serving on a qualifying
  full-BAR/P2P-on GFX906 host.

### Reason

The release notes promise the normal benchmark ladder, and users read the
generated `summary.md` after a run. The fixture replay verifies the runner's
request order and makes the summary output align with the documented strict
versus fixed-token claim boundary.

## GGUF-321 - Draft Release Local Path and Link Gate Audit

### Setup

- Date: 2026-06-29 UTC.
- Scope: audit the vNext draft release note as a reader-facing instruction
  surface by checking named local scripts/docs and the current state of draft
  absolute GitHub links.
- Commands tested:

```sh
rg -o '([A-Za-z0-9_./-]+\\.(sh|py|md|txt|env|json|so)|tools/model-format-probe)' \
  releases/draft-vnext-gfx906-rocm72-gguf-hf-repro.md

test -e <referenced-local-file>

curl -fsS -o /dev/null -w '%{http_code}' \
  https://raw.githubusercontent.com/joe2gaan/localaiservers/main/<linked-path>
```

### Result

All repo-local files named by the draft release note exist in the current
candidate tree, including:

- vNext launcher, inspector, staging, serviceability, public-HF, public-GGUF,
  launch-artifact, schema, readiness, and benchmark scripts;
- `begin_think_proxy.py` and `run_chat_capture.py`;
- `tools/model-format-probe/Makefile` and source;
- vNext profile docs and the GGUF experiment/key-learning/source-inventory
  docs.

The draft `Links` section currently points at GitHub `main` for owner-review
convenience. Links to already-published files such as
`docs/gfx906-host-platform-prereqs-v02.md` and `qwen36-gfx906/README.md`
resolve on current `main`, but links to new vNext files return `404` until the
vNext docs/scripts are committed and published.

The release draft already warned that `main` links must be replaced with the
final release tag or pinned release commit before publishing. This audit made
that requirement more explicit in the validation section:

```text
The `Links` section below currently uses `main` URLs for draft owner review;
before publishing, replace those links with the final vNext release tag or a
pinned release commit and verify that every linked page resolves.
```

`verify_vnext_serviceability.sh` now locks that final-link publication gate so
the warning cannot silently disappear.

### Promote / Reject

Promote:

- keeping draft `main` links only as owner-review placeholders;
- requiring final vNext release links to use the final tag or pinned release
  commit;
- verifying every linked page before publication.

Reject:

- publishing the current draft `Links` section unchanged;
- treating uncommitted local vNext docs as live public links;
- using working-tree path existence as proof that GitHub release-body links
  will resolve.

### Reason

The release-note reproduction path includes both commands and links. Local path
checks prove the candidate tree is internally complete, but public release notes
must link to immutable, resolvable public pages after the release tag exists.

## GGUF-322 - Stand-Alone vNext Prerequisite Coverage Audit

### Setup

- Date: 2026-06-29 UTC.
- Scope: audit whether the stand-alone vNext public reproduction instructions
  name the host tools required by the release-note scripts.
- Commands tested:

```sh
rg -n "git |docker|make|compiler|python3|curl|sha256sum|tar|split|cat|grep|sed|awk|find|sort|wc|mktemp|dd|POSIX|standard" \
  releases/draft-vnext-gfx906-rocm72-gguf-hf-repro.md \
  qwen36-gfx906/profiles/vnext/README.md \
  qwen36-gfx906/*.sh
```

### Result

The release draft and vNext profile README already listed the main external
tools: POSIX `/bin/sh`, `git`, `docker`, `make`, a C compiler, `python3`,
`curl`, `sha256sum`, `tar`, `split`, `cat`, and `hf` for HF staging.

The script audit showed the public flow also relies on common POSIX/core
userland utilities such as `grep`, `sed`, `awk`, `find`, `sort`, `wc`,
`mktemp`, `mkdir`, `rm`, `mv`, `cp`, `ln`, `chmod`, `touch`, `date`, `sleep`,
`kill`, `wait`, `dd`, `dirname`, `basename`, and `tail`.

The release draft and profile README now name that standard utility
requirement explicitly. This keeps the vNext release package self-contained as
a stand-alone release path instead of assuming a validation-host shell
environment.

### Promote / Reject

Promote:

- vNext as a stand-alone release with explicit host-tool prerequisites;
- POSIX-shell scripts plus ordinary POSIX/core utilities as the expected user
  environment;
- `hf` only where HF staging or tokenizer resolution needs it.

Reject:

- treating a maintainer validation host as the implicit prerequisite list;
- hiding required tools inside scripts without documenting the host-side
  baseline;
- publishing vNext as a v0.2.1 patch/addendum.

### Reason

A stand-alone reproduction package should let another developer prepare the
host before running the release flow. Naming the ordinary utility baseline is a
small documentation fix that reduces first-run friction without changing any
runtime claims, Docker image, model artifact, benchmark result, or release
boundary.

## GGUF-323 - vNext Release-Readiness Wrapper Replay

### Setup

- Date: 2026-06-29 UTC.
- Scope: follow the draft vNext release-note readiness wrapper from clean
  temporary model/cache/runtime roots and verify that it stops at the correct
  public boundary before final release assets exist.
- Commands tested:

```sh
tmp=$(mktemp -d /tmp/vnext-readiness-replay.XXXXXX)
cd qwen36-gfx906
./verify_vnext_release_readiness.sh \
  --release-tag '<release-tag>' \
  --model-root "$tmp/models" \
  --hf-cache "$tmp/hf" \
  --runtime-root "$tmp/runtime"
```

```sh
tmp=$(mktemp -d /tmp/vnext-readiness-replay.XXXXXX)
cd qwen36-gfx906
CHECK_RUNTIME_IMAGE=0 \
CHECK_RUNTIME_PATHS=0 \
CAPTURE_VLLM_HELP=0 \
ALLOW_PRE_SERVING_ONLY=1 \
./verify_vnext_release_readiness.sh \
  --release-tag vnext-unpublished-fixture \
  --model-root "$tmp/models" \
  --hf-cache "$tmp/hf" \
  --runtime-root "$tmp/runtime"
```

### Result

The placeholder-tag run failed immediately with:

```text
error: release tag must be the final published tag, not a placeholder
```

The unpublished-tag dry run passed the local release gates:

- serviceability;
- contract matrix;
- inferred-profile checks;
- deploy/profile parity checks;
- expected fail-closed profile mismatch cases.

It then stopped at the public GGUF asset gate with a GitHub release-asset
`404`, followed by:

```text
error: public GGUF asset gate failed; check release tag, release assets, part manifests, and final SHA256 locks
```

This is the correct current boundary because the final stand-alone vNext tag
and public release assets do not exist yet.

### Promote / Reject

Promote:

- the wrapper as the documented top-level release-readiness command;
- placeholder-tag rejection before any network/model work;
- public-asset failure before any benchmark or host-specific success claim;
- keeping `ALLOW_PRE_SERVING_ONLY=1` as a maintainer dry-run mode, not a full
  reproduction success path.

Reject:

- calling the draft release reproduced while the public GGUF asset gate returns
  `404`;
- treating `ALLOW_PRE_SERVING_ONLY=1` as a release success mode;
- using a local maintainer asset mirror as the published reproduction source.

### Reason

The release notes now have a single high-level command that follows the intended
sequence. The current replay proves that the command is fail-closed: it can
validate local scaffolding but cannot be used to claim public vNext
reproduction until the final standalone tag and release assets are published.

## GGUF-324 - vNext Command Surface and Executability Audit

### Setup

- Date: 2026-06-29 UTC.
- Scope: compare the documented vNext release-note command surface against the
  actual scripts, executable bits, and POSIX syntax before another readiness
  replay.
- Commands tested:

```sh
for f in \
  qwen36-gfx906/hash_vnext_patch_bundle.sh \
  qwen36-gfx906/run_vnext_profile_benchmark.sh \
  qwen36-gfx906/stage_vnext_gguf_assets.sh \
  qwen36-gfx906/verify_vllm_arg_schema.sh \
  qwen36-gfx906/verify_vnext_contract_matrix.sh \
  qwen36-gfx906/verify_vnext_deploy_profile_parity.sh \
  qwen36-gfx906/verify_vnext_launch_artifact.sh \
  qwen36-gfx906/verify_vnext_public_assets.sh \
  qwen36-gfx906/verify_vnext_public_hf_inputs.sh \
  qwen36-gfx906/verify_vnext_release_readiness.sh \
  qwen36-gfx906/verify_vnext_serviceability.sh \
  qwen36-gfx906/vnext_profile_inspect.sh \
  qwen36-gfx906/vnext_repro_launcher.sh \
  qwen36-gfx906/check_host_platform_prereqs.sh
do
  test -x "$f"
  sh -n "$f"
done
```

```sh
./qwen36-gfx906/verify_vnext_release_readiness.sh --help
```

### Result

Every documented vNext shell script is executable and passed `sh -n` syntax
checking. The release-readiness wrapper help matches the documented gate order:

```text
serviceability -> contract matrix -> public GGUF asset gate -> public HF gate
-> launch artifact generation -> vLLM argv schema -> host preflight
-> optional serving benchmark ladder
```

The direct command scan found a few standard utilities that were used by the
scripts but not yet named in the prerequisite list. The release draft and
profile README now explicitly include `cp`, `ln`, `chmod`, `touch`, and
`tail`, in addition to the previously listed POSIX/core utilities.

### Promote / Reject

Promote:

- documenting the wrapper as the top-level reproduction command;
- requiring all release-facing scripts to be executable and POSIX syntax clean;
- keeping the host prerequisite list tied to the actual script command
  surface.

Reject:

- publishing a standalone release whose scripts rely on unnamed host-side
  utilities;
- documenting a gate order that differs from `verify_vnext_release_readiness.sh`;
- depending on generated `docker_run.sh` files before the launcher creates
  them from the public profile contracts.

### Reason

The release notes are meant to be followed directly. This audit keeps the
reader-facing command surface honest before the remaining asset and serving
benchmark blockers are removed.

## GGUF-325 - Current Local Locked-Artifact Recheck

### Setup

- Date: 2026-06-29 UTC.
- Scope: recheck local artifact mirrors found during the active release-note
  reproduction goal before attempting a maintainer-side `file://` asset-gate
  replay.
- Checked:
  - local real-sized Dense GGUF candidate;
  - local real-sized MoE GGUF candidate;
  - all found `Qwen3.6-27B-text-config-eosfix.tar.gz` candidates under the
    active temporary/replay roots.

### Result

The local artifact mirror contains real-sized GGUF files matching the current
vNext locks:

```text
Qwen3.6-27B-F16.gguf
b2347376b9bb7d12cf5f1d31c53ac4c60dd3d4b95068a09351187b328b5e9d89

Qwen3.6-35B-A3B-F16.gguf
1f2443bb0ff958943d091410c61120c181a0579b3bc85192029aa51d821d141c
```

The candidate Dense text-config archives found locally were stale or synthetic
fixtures. None matched the current release lock:

```text
Qwen3.6-27B-text-config-eosfix.tar.gz
9a85f0a18a012ed37f0eb4c42549569234af880c5f6e3ce908e0e45cca835719
```

Representative stale text-config archive hashes observed in the current pass:

```text
06d28eaf1fdbed2edb2ef0f6117b941e3cfac0be26dbed581049c1ec43f09d77
0c985566e5458ab38601a1fcbb50bedba74005e942199473fde63818724c039c
3b6e1b4c078910e5b5c4840d43d581443e4645a66dcb10d59736c75d19787acb
97079fb1618807c5f9c8c681796f295b062f5b531ca4d9ea942181302fac105f
```

A prior tiny split-asset fixture also had nonmatching GGUF hashes and was
rejected as release-reproduction evidence.

### Promote / Reject

Promote:

- the locked full GGUF hashes as locally reverified;
- the current release-note blocker that final vNext publication still requires
  the exact Dense text-config archive or a benchmarked deterministic public
  generation recipe;
- refusing to use stale or synthetic fixtures for public release evidence.

Reject:

- replaying the public GGUF asset gate against a stale text-config archive;
- treating a local full-GGUF mirror as sufficient for public reproduction;
- changing the text-config lock without rerunning the normal benchmark ladder
  from the changed public input.

### Reason

The asset path is now narrowed: the full GGUF bytes can be verified locally,
but the Dense GGUF profile still depends on the missing locked text-config
archive. The release notes should continue to fail closed at the public asset
gate until that public input is recovered, attached, or replaced by a
benchmarked deterministic generation recipe.

## GGUF-326 - HF Public-Input Gate Completeness Hardening

### Setup

- Date: 2026-06-29 UTC.
- Scope: verify whether the HF public-input gate proves real staged public HF
  model packages, not just minimal config directories.
- Tested roots:
  - complete public-HF staging root with real `Qwen/Qwen3.6-27B` and
    `Qwen/Qwen3.6-35B-A3B` files;
  - older tiny synthetic HF fixture previously used for contract-gate smoke.
- Commands tested:

```sh
qwen36-gfx906/verify_vnext_public_hf_inputs.sh <complete-hf-model-root>
qwen36-gfx906/verify_vnext_public_hf_inputs.sh <tiny-hf-fixture-root>
sh -n qwen36-gfx906/verify_vnext_public_hf_inputs.sh
```

### Result

The original HF gate was too weak for public reproduction because a tiny
config-oriented fixture could satisfy profile generation even though it was
not a real HF model package.

The HF gate now verifies:

- `tokenizer.json`;
- `tokenizer_config.json`;
- `model.safetensors.index.json`;
- every non-empty shard named by the safetensors index;
- no unresolved Git LFS pointer files in those shards.

The complete public-HF staging root passed:

```text
public-hf-shards-ok: Dense HF shards=15
public-hf-shards-ok: MoE HF shards=26
public-hf-profile-ok: hf-dense27b-tp8
public-hf-profile-ok: hf-moe35b-tp4
public-hf-profile-ok: hf-moe35b-tp8
vNext public HF input gate passed
```

The older tiny fixture now fails before profile generation:

```text
error: missing Dense HF tokenizer.json
```

The release draft and profile README now describe the stronger HF package
completeness check.

### Promote / Reject

Promote:

- complete public HF package staging as a prerequisite for HF profile
  reproduction;
- safetensors-index-based shard validation instead of a bare `config.json`
  check;
- rejecting unresolved Git LFS pointer files before launch-artifact
  generation.

Reject:

- treating config-only or tiny synthetic HF directories as public reproduction
  inputs;
- using profile-generation success alone as proof that model weights were
  staged;
- publishing HF reproduction instructions whose verifier can pass without real
  model shards.

### Reason

The release notes tell users to stage public HF model packages. A verifier that
accepts only `config.json` would allow a false-positive preflight and defer
failure until container startup. Tightening the gate makes the release-note
path more accurate and fail-closed before benchmark work begins.

## GGUF-327 - Stand-Alone vNext HF Completeness Lock Replay

### Setup

- Date: 2026-06-29 UTC.
- Scope: enforce the strengthened HF package-completeness checks through the
  vNext serviceability gate and replay the standalone release-readiness
  boundaries.
- Commands tested:

```sh
sh -n qwen36-gfx906/verify_vnext_serviceability.sh \
  qwen36-gfx906/verify_vnext_public_hf_inputs.sh \
  qwen36-gfx906/verify_vnext_release_readiness.sh \
  qwen36-gfx906/stage_vnext_gguf_assets.sh \
  qwen36-gfx906/run_vnext_profile_benchmark.sh

qwen36-gfx906/verify_vnext_serviceability.sh
qwen36-gfx906/verify_vnext_contract_matrix.sh
qwen36-gfx906/verify_vnext_public_hf_inputs.sh <complete-hf-model-root>
qwen36-gfx906/verify_vnext_public_hf_inputs.sh <tiny-hf-fixture-root>

ALLOW_PRE_SERVING_ONLY=1 CHECK_RUNTIME_IMAGE=0 CHECK_RUNTIME_PATHS=0 \
  CAPTURE_VLLM_HELP=0 qwen36-gfx906/verify_vnext_release_readiness.sh \
  --release-tag '<release-tag>' \
  --model-root <complete-hf-model-root> \
  --hf-cache <hf-cache-root> \
  --runtime-root <runtime-root>

ALLOW_PRE_SERVING_ONLY=1 CHECK_RUNTIME_IMAGE=0 CHECK_RUNTIME_PATHS=0 \
  CAPTURE_VLLM_HELP=0 qwen36-gfx906/verify_vnext_release_readiness.sh \
  --release-tag vnext-dry-run-20260629 \
  --model-root <complete-hf-model-root> \
  --hf-cache <hf-cache-root> \
  --runtime-root <runtime-root>
```

### Result

The serviceability gate now locks:

- the draft release note's explicit `model.safetensors.index.json`
  requirement;
- the draft release note's unresolved Git LFS pointer rejection;
- the HF verifier's shard-index and LFS-pointer checks; and
- the profile README's matching public HF package-completeness wording.

The complete public-HF staging root still passed:

```text
public-hf-shards-ok: Dense HF shards=15
public-hf-shards-ok: MoE HF shards=26
public-hf-profile-ok: hf-dense27b-tp8
public-hf-profile-ok: hf-moe35b-tp4
public-hf-profile-ok: hf-moe35b-tp8
vNext public HF input gate passed
```

The older tiny fixture still failed correctly:

```text
error: missing Dense HF tokenizer.json
```

The release-readiness wrapper rejected the placeholder tag:

```text
error: release tag must be the final published tag, not a placeholder
```

The unpublished final-looking tag replay passed serviceability and contract
matrix, then stopped at the public GGUF asset gate with a GitHub asset `404`,
which is the expected current boundary before final vNext release assets exist.

### Promote / Reject

Promote:

- vNext as a stand-alone release whose HF inputs must be real public HF model
  packages;
- serviceability locks that prevent the public HF package gate from drifting
  back to config-only checks;
- the release-readiness wrapper's placeholder-tag rejection and public-asset
  fail-closed behavior.

Reject:

- treating a partial HF fixture as a public model package;
- publishing vNext before the final release tag and public GGUF assets exist;
- calling local dry-run readiness a successful public standalone reproduction.

### Reason

Standalone release wording is not enough by itself. The release package needs
mechanical checks that prove public HF inputs and public GGUF assets are
materialized before benchmark claims are made. This replay keeps the current
vNext path fail-closed: HF package staging can now be verified, while the GGUF
side still waits for final public release assets and the locked Dense
text-config input or a benchmarked deterministic replacement recipe.

## GGUF-328 - Dense Text-Config Runtime Reconstruction Probe

### Setup

- Date: 2026-06-29 UTC.
- Scope: test whether the locked Dense GGUF text-config asset can be
  regenerated deterministically from the pinned public `Qwen/Qwen3.6-27B`
  config using the pinned release runtime image's Qwen3.5 config classes.
- Runtime image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`
- Target config hash:
  `370a0c6b21a09e288bd3301581e126956d1b2dd410e791920884c3130bfe6e0a`
- Target archive hash:
  `9a85f0a18a012ed37f0eb4c42549569234af880c5f6e3ce908e0e45cca835719`

### Result

No discovered local text-config archive matched the current archive lock. The
observed archive hashes were stale fixture or prior-candidate outputs:

```text
97079fb1618807c5f9c8c681796f295b062f5b531ca4d9ea942181302fac105f
0c985566e5458ab38601a1fcbb50bedba74005e942199473fde63818724c039c
3b6e1b4c078910e5b5c4840d43d581443e4645a66dcb10d59736c75d19787acb
06d28eaf1fdbed2edb2ef0f6117b941e3cfac0be26dbed581049c1ec43f09d77
```

No discovered extracted Dense text-config `config.json` matched the current
config lock. The observed config hashes were:

```text
7c87e49055ec9c5772ba2159b85c3f64f2f3c529ee9568b06068769f04e96745
a87cd840493f5dc058668d70a2fef679449b820dd0e88a641566a40c17cbe82d
f242072892095e27875e9482323ef5e73b0a86f851d1891215e555fd7fc10ecb
```

A runtime-image reconstruction probe generated candidate text configs from
the public HF `text_config` and top-level config using:

- `vllm.transformers_utils.configs.qwen3_5.Qwen3_5TextConfig`;
- `vllm.transformers_utils.configs.qwen3_5.Qwen3_5Config`;
- `to_dict()` and `to_diff_dict()`;
- architecture overrides including `Qwen3_5ForCausalLM`,
  `Qwen3ForCausalLM`, and `Qwen3_5ForConditionalGeneration`;
- `model_type` variants `qwen3_5`, `qwen3_5_text`, and `qwen3`; and
- common EOS variants.

The runtime image accepted those config classes, but none of the generated
candidates matched the locked `370a...` config hash.

### Promote / Reject

Promote:

- the current release-note blocker that the Dense text-config archive must be
  recovered as the locked public asset or replaced by a benchmarked
  deterministic public recipe;
- treating runtime-class reconstruction as tested and currently non-promoted.

Reject:

- substituting any currently discovered stale text-config archive;
- substituting the obvious compact text-only config variants;
- substituting a newly generated runtime-class config without rerunning the
  full normal benchmark ladder from that changed public input.

### Reason

The Dense GGUF path is not blocked by the full GGUF bytes; those hashes have
been locally verified. It is blocked by the text-config package needed for the
Dense GGUF profile. Since neither local archive discovery nor runtime-image
config-class reconstruction reproduced the current lock, the stand-alone
vNext release notes should continue to fail closed until the exact asset is
recovered or a replacement config recipe is benchmarked and documented as the
new public release input.

Supersession note: GGUF-329 later recovered and verified the exact locked
Dense text-config archive from validation-lane model-store volumes. The active
blocker shifted from asset recovery to publishing the recovered asset as part
of the final stand-alone vNext GitHub Release and rerunning the readiness flow
from those public release inputs.

## GGUF-329 - Stand-Alone vNext Real Asset Recovery and Tar Portability Probe

### Setup

- Date: 2026-06-29 UTC.
- Scope: recover the currently locked stand-alone vNext GGUF release inputs
  from validation-lane model-store volumes and replay the public GGUF asset
  gate through a local `file://` release-asset mirror.
- Hosts checked: `.20` and `.30`.
- Target Dense text-config archive hash:
  `9a85f0a18a012ed37f0eb4c42549569234af880c5f6e3ce908e0e45cca835719`
- Target Dense text-config `config.json` hash:
  `370a0c6b21a09e288bd3301581e126956d1b2dd410e791920884c3130bfe6e0a`
- Target Dense GGUF hash:
  `b2347376b9bb7d12cf5f1d31c53ac4c60dd3d4b95068a09351187b328b5e9d89`
- Target MoE GGUF hash:
  `1f2443bb0ff958943d091410c61120c181a0579b3bc85192029aa51d821d141c`

### Result

The validation-lane model stores contained the exact current Dense text-config
archive lock, the extracted Dense `config.json` lock, and both final GGUF
file locks. The recovered artifacts were copied into a local private asset
mirror and verified by SHA256.

The full GGUF files were then split into release-asset-shaped parts:

- Dense GGUF: `13` parts
- MoE GGUF: `17` parts

The first public GGUF asset-gate replay failed during Dense text-config
extraction. The tar archive stores uid/gid metadata, and this root/unshare
environment rejected ownership restoration during `tar -xzf`.

After updating the staging script to use `tar --no-same-owner` when supported,
the same local split-asset mirror passed the public GGUF asset gate:

```text
Qwen3.6-27B-F16.gguf: OK
Qwen3.6-35B-A3B-F16.gguf: OK
public-asset-profile-ok: gguf-dense27b-tp8
public-asset-profile-ok: gguf-moe35b-tp4
vNext public GGUF asset gate passed for tag: vnext-local-real-assets
```

### Promote / Reject

Promote:

- the recovered Dense text-config archive as the current public-release asset
  candidate;
- the recovered Dense and MoE GGUF files as matching the current vNext hash
  locks;
- a staging-script portability fix that uses `tar --no-same-owner` when the
  local tar implementation supports it.

Reject:

- assuming tar extraction is portable just because it works on the validation
  host where the archive was created.

### Reason

The asset hashes are no longer the blocker: the exact locked inputs were
recovered and locally verified. The first replay exposed a release-package
portability issue, and the corrected staging path now passes the real local
public GGUF asset gate. A public reproducer may run the staging command as
root, in a container, or inside an id-mapped environment, so the text-config
extraction must not depend on restoring archive ownership metadata.

## GGUF-330 - Combined vNext Release-Readiness Replay With Real Local Assets

### Setup

- Date: 2026-06-29 UTC.
- Scope: run the documented `verify_vnext_release_readiness.sh` sequence using
  the recovered real GGUF split-asset mirror and real staged public HF inputs.
- Release tag argument: `vnext-local-real-assets`
- Asset source: maintainer-only `file://` mirror with
  `ALLOW_LOCAL_ASSET_PREFLIGHT=1`
- Serving benchmark mode: disabled for this dry run.
- Runtime vLLM schema capture: disabled for this dry run.
- Host: local non-release workstation environment.

### Result

The combined readiness wrapper passed:

- serviceability checks;
- contract matrix checks;
- public GGUF asset staging from the real local split mirror;
- public HF input validation from real staged HF packages;
- launch artifact generation for all five vNext profiles:
  - `gguf-dense27b-tp8`
  - `gguf-moe35b-tp4`
  - `hf-dense27b-tp8`
  - `hf-moe35b-tp4`
  - `hf-moe35b-tp8`

The wrapper then stopped at host platform preflight, which is the correct
behavior for this local environment. The preflight saw only one likely GFX906
device and no full 32 GiB BAR coverage, so the host is not comparable with the
published full-BAR/P2P-on release lane.

### Promote / Reject

Promote:

- the combined release-readiness sequence as correctly ordered through
  launch-artifact generation;
- the recovered real GGUF and HF assets as sufficient for pre-serving
  readiness replay;
- the host preflight fail-closed behavior as correct when the current machine
  is not an 8-GPU full-BAR/P2P GFX906 host.

Reject:

- interpreting this dry run as full vNext release reproduction;
- running serving benchmarks on a host that fails the full-BAR/P2P preflight;
- publishing final vNext claims before the same wrapper passes with
  `RUN_SERVING_BENCHMARKS=1` from final public release assets on a qualifying
  host.

### Reason

This run proves that the release-note sequence can advance past asset staging
and artifact generation with the real locked inputs. It does not prove final
release reproduction because full success still requires the final public
GitHub Release assets and a qualifying full-BAR/P2P-on GFX906 host for the
serving benchmark ladder.

## GGUF-331 - Validation-Lane Host Preflight Replay

### Setup

- Date: 2026-06-29 UTC.
- Scope: run the read-only host-platform preflight on `.20` and `.30` before
  attempting any serving benchmark replay from the vNext package.
- Profiles checked:
  - `dense27b_tp8_fullbar_p2pon`
  - `moe35b_tp4_fullbar_p2pon`
  - `moe35b_tp8_fullbar_p2pon`

### Result

Both `.20` and `.30` passed the host-platform preflight for all three release
profiles:

- `amdgpu` module loaded;
- `amdgpu` module version reported as `6.8.5`;
- `rocminfo` reported `8` gfx906 agents;
- `8` likely GFX906 PCI display/controller devices found;
- all `8` likely GFX906 devices reported largest BAR `32.00` GiB;
- KFD topology visible with `9` nodes and `16` io-link property files;
- `rocm-smi` reported `8` GPUs with VBIOS `113-D1631700-111`;
- `rocm-smi --showtopo` completed.

### Promote / Reject

Promote:

- `.20` and `.30` as qualifying full-BAR/P2P-on validation lanes for the next
  vNext serving replay;
- the host preflight script as sufficient to catch the local non-release
  workstation mismatch observed in GGUF-330.

Reject:

- running the full serving ladder on the local non-release workstation;
- treating host preflight success as proof of final benchmark reproduction.

### Reason

The remote validation lanes satisfy the platform prerequisites that the local
combined readiness dry run failed. The next serving replay should run on one
of these hosts from the vNext launch artifacts and locked public-input shape.

## GGUF-332 - Final GGUF Reuse Gate

### Setup

- Date: 2026-06-30 UTC.
- Scope: verify that the GGUF staging helper can reuse an already staged final
  GGUF file when its SHA256 matches the locked release value, without requiring
  split parts to be present in the model root.
- Asset source: maintainer-only `file://` mirror for the Dense text-config
  archive.
- Model root: temporary directory containing symlinks to the verified final
  Dense and MoE GGUF files, with no split part files staged.

### Result

The public GGUF asset gate passed:

```text
using existing verified model: Qwen3.6-27B-F16.gguf
using existing verified model: Qwen3.6-35B-A3B-F16.gguf
public-asset-profile-ok: gguf-dense27b-tp8
public-asset-profile-ok: gguf-moe35b-tp4
vNext public GGUF asset gate passed for tag: vnext-local-real-assets
```

### Promote / Reject

Promote:

- final-file reuse as a release-package convenience for reruns and resumed
  staging;
- final SHA256 verification as sufficient to skip split-part storage when the
  final file already exists locally.

Reject:

- treating final-file reuse as a replacement for publishing the split assets;
- publishing a from-scratch release path that depends on pre-existing local
  GGUF files.

### Reason

From-scratch public reproduction still needs release-hosted split parts or a
public conversion recipe. Final-file reuse makes the instructions easier and
less wasteful for users who rerun verification or pre-stage the final files,
while preserving the same final SHA256 claim boundary.

## GGUF-333 - Dot20 Pre-Serving vNext Release-Readiness Replay

### Setup

- Date: 2026-06-30 UTC.
- Scope: run the documented vNext release-readiness wrapper on `.20` from the
  scratch repository copy and model root prepared under the local NVMe-backed
  model-store volume.
- Asset source: maintainer-only `file://` mirror containing the recovered
  Dense text-config archive.
- GGUF inputs: final verified Dense and MoE GGUF files reused by SHA256.
- HF inputs: complete staged public HF package directories selected by shard
  completeness and validated by `verify_vnext_public_hf_inputs.sh`.
- Serving benchmark mode: disabled for this pre-serving replay.
- Runtime vLLM schema capture: disabled for this pre-serving replay.

### Result

The `.20` release-readiness wrapper passed:

- serviceability checks;
- contract matrix checks;
- public GGUF asset gate with final-file reuse;
- public HF input gate;
- launch artifact generation for all five vNext profiles;
- host preflight for:
  - `dense27b_tp8_fullbar_p2pon`
  - `moe35b_tp4_fullbar_p2pon`
  - `moe35b_tp8_fullbar_p2pon`

The first `.20` pre-serving attempt rejected the initially selected Dense HF
candidate because it had tokenizer/index files but was missing the indexed
weight shards. A second selector required every shard named by
`model.safetensors.index.json` to exist and be non-empty. With that complete
Dense HF package and a complete MoE HF package linked into the model root, the
HF gate passed:

```text
public-hf-shards-ok: Dense HF shards=15
public-hf-shards-ok: MoE HF shards=26
public-hf-profile-ok: hf-dense27b-tp8
public-hf-profile-ok: hf-moe35b-tp4
public-hf-profile-ok: hf-moe35b-tp8
vNext public HF input gate passed
```

The wrapper ended with the intended pre-serving boundary:

```text
pre-serving-vnext-release-gates-passed
full-serving-benchmark-not-run: set RUN_SERVING_BENCHMARKS=1 for release reproduction success
```

### Promote / Reject

Promote:

- `.20` as ready for the full serving benchmark ladder from the vNext package;
- the HF completeness gate as necessary, since tokenizer/index presence alone
  accepted an incomplete Dense candidate until shard checks ran;
- final-file GGUF reuse for validation-lane replay where the final GGUF files
  already match the locked hashes.

Reject:

- treating pre-serving readiness as benchmark reproduction;
- selecting HF packages by directory name or tokenizer/index presence alone;
- starting the serving ladder before checking for unrelated active workloads.

### Reason

This is the strongest non-serving evidence so far. It follows the documented
release-readiness flow on a qualifying host and reaches the explicit boundary
that full reproduction still requires `RUN_SERVING_BENCHMARKS=1`.

## GGUF-334 - Dot20 HF Snapshot Symlink Visibility Failure

### Setup

- Date: 2026-06-30 UTC.
- Scope: run the full vNext release-readiness flow on `.20` with serving
  benchmarks enabled after the pre-serving gates passed.
- Profile blocked first: `hf-dense27b-tp8`.
- Model layout: the local HF model alias under the selected model root resolved
  to a Hugging Face cache snapshot whose safetensors shards were relative
  symlinks into the snapshot `blobs` directory.

### Result

The generated Docker wrapper preserved the top-level HF model symlink from the
parent model-root mount. The container could see the model alias, but the
snapshot shard symlinks did not resolve inside the container, so the HF Dense
server failed before benchmark warmups with a missing safetensors shard.

A one-shot container check reproduced the failure shape: the container saw the
HF model alias and a shard symlink, but the shard target was not visible from
inside the bind-mounted model root.

### Promote / Reject

Promote:

- HF snapshot shard visibility as a required release-readiness check.

Reject:

- mounting only the parent model root when an HF model alias is a symlink to a
  snapshot directory containing relative shard symlinks;
- treating a visible top-level model alias as proof that all indexed shards are
  readable inside the container.

### Reason

This was a real portability bug. It would affect any user whose local
SSD/NVMe-backed model directory points at a Hugging Face snapshot layout
instead of containing regular shard files. Public reproduction cannot depend on
the validation host's symlink topology.

## GGUF-335 - HF Runtime Model-Root Shim And Blob Mount

### Setup

- Date: 2026-06-30 UTC.
- Scope: repair HF snapshot portability without changing the runtime image or
  relying on a validation-host path.
- Implementation path: generated `docker_run.sh` from
  `vnext_repro_launcher.sh`.

### Result

The launcher now detects a symlinked HF model path, resolves the target
snapshot, checks for symlinked `*.safetensors` shards, resolves the matching
snapshot `blobs` directory, and creates a runtime `model-bind-root` shim below
the selected runtime root. The generated Docker wrapper mounts:

- the shim root at `/opt/local-models`;
- the resolved snapshot under `/opt/vnext-models/<model-alias>`;
- the matching snapshot `blobs` directory read-only at `/opt/blobs`.

Broken or non-directory model symlinks fail closed. Unresolved snapshot shard
links also fail closed before benchmark launch.

A one-shot container check after the fix showed the HF model alias resolving to
`/opt/vnext-models/...` and the first safetensors shard resolving through
`/opt/blobs`.

### Promote / Reject

Promote:

- the runtime model-root shim and read-only blob mount as the portable HF
  snapshot handling mechanism for vNext;
- fail-closed HF shard visibility checks before benchmark launch.

Reject:

- relying on Docker mount ordering to replace a symlink preserved from a parent
  bind mount;
- accepting unresolved Hugging Face snapshot shard links as a serviceable
  launch artifact.

### Reason

This keeps vNext portable for normal Hugging Face cache layouts while still
letting users choose their own model root, HF cache, and runtime root. It does
not modify the container image or the published model packages.

## GGUF-336 - Dot20 Full vNext Release-Readiness Serving Replay

### Setup

- Date: 2026-06-30 UTC.
- Scope: follow the vNext release notes/readiness flow with serving benchmarks
  enabled on `.20`.
- Release boundary: stand-alone vNext draft path; not v0.2.0 or v0.2.1.
- Asset source: maintainer-only local mirror of future release assets for
  pre-publication validation.
- GGUF inputs: final verified Dense and MoE F16 GGUF files and Dense text-config
  archive staged through the release-asset layout.
- HF inputs: public upstream HF packages staged into a local model root and
  verified for complete safetensors shard sets.
- Benchmark ladder: normal warmups -> `c1_128` uncapped strict -> `c1_2000` ->
  `c1_10000`.

### Result

The full vNext release-readiness flow completed:

- serviceability checks;
- runtime image digest/native path checks;
- contract matrix checks;
- public GGUF asset gate;
- public HF input gate;
- launch artifact generation;
- runtime vLLM argument-schema validation;
- host platform preflight for Dense TP8, MoE TP4, and MoE TP8;
- serving benchmark ladders for all five vNext profiles.

| Format | Profile | Host | Strict backend TPS | `c1_2000` backend TPS | `c1_10000` backend TPS | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| GGUF F16 | Dense 27B TP8 full-BAR/P2P-on | `.20` | `69.980` | `70.975` | `66.493` | strict-valid; ai-info 10K gate cleared |
| GGUF F16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | `.20` | `118.252` | `119.198` | `112.115` | strict-valid |
| HF FP16 | Dense 27B TP8 full-BAR/P2P-on | `.20` | `70.454` | `71.516` | `66.930` | strict-valid; ai-info 10K gate cleared |
| HF FP16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | `.20` | `111.702` | `116.415` | `109.442` | strict-valid |
| HF FP16 | Qwen3.6 35B-A3B MoE TP8 full-BAR/P2P-on | `.20` | `116.301` | `116.681` | `109.761` | strict-valid |

The readiness wrapper ended with:

```text
vNext full release reproduction path completed
```

### Promote / Reject

Promote:

- the vNext readiness wrapper as the successful end-to-end deployment and
  benchmark reproduction path for the stand-alone vNext draft package;
- the HF snapshot shim and shard visibility checks as required public
  portability behavior;
- `.20` as a successful full-ladder validation lane for all five profiles.

Reject:

- classifying vNext as an update to v0.2.1;
- publishing the GGUF rows before the same flow is rerun from public release
  asset URLs rather than a maintainer-only local mirror;
- treating pre-serving readiness as sufficient once the serving ladder is part
  of the release goal.

### Reason

The documented vNext flow now proves that a qualifying host can generate the
launch artifacts, pass preflight, start the runtime, complete the warmups, and
reproduce the benchmark ladder for both GGUF and HF profiles. The remaining
publication gate is packaging the exact public release assets and rerunning the
same flow against the final stand-alone vNext tag and asset URLs.

## GGUF-337 - vNext Stand-Alone Boundary Check

### Setup

- Date: 2026-06-30 UTC.
- Scope: verify whether the full readiness work changes the published v0.2.1
  release boundary.

### Result

The vNext work is a separate release package. It uses its own draft release
notes, profiles, generated launch artifacts, public asset gates, and benchmark
tables. It does not retarget, replace, or silently update v0.2.1.

### Promote / Reject

Promote:

- vNext as a stand-alone release boundary.

Reject:

- describing vNext as a v0.2.1 patch, addendum, or replacement;
- updating v0.2.1 claims with vNext benchmark numbers.

### Reason

v0.2.1 remains the historical ROCm7.2 Dense/MoE reproduction-package release.
vNext can use v0.2.1 as a comparator, but it must be reproducible from its own
tag, release assets, public inputs, and documentation.

## GGUF-338 - Dot20 HF Staging Mode Reproduction Failure

### Setup

- Date: 2026-06-30 UTC.
- Scope: repeat the full vNext release-readiness flow from a fresh checkout on
  `.20` after the initial full replay completed.
- Storage: SSD/NVMe-backed model, cache, runtime, and release-asset roots.
- HF package state: pinned public HF packages were already staged and verified;
  the selected HF model aliases resolved through local snapshot symlinks.
- Initial command shape: release-note readiness wrapper with
  `RUN_SERVING_BENCHMARKS=1` and `STAGE_HF_PUBLIC_INPUTS=1`.

### Result

The first repeat attempt failed before serving benchmarks because the clean
scratch environment did not have the `hf` CLI on `PATH`. That is a legitimate
host prerequisite failure for `STAGE_HF_PUBLIC_INPUTS=1`, not a benchmark or
container failure.

After installing the Hugging Face CLI in an SSD/NVMe scratch virtual
environment, the second attempt still failed before serving benchmarks. The
target HF model directories were symlinked to already-staged snapshot
directories, and `hf download --local-dir` attempted to materialize files
through those symlink targets. That is not a safe public reproduction mode:
`STAGE_HF_PUBLIC_INPUTS=1` should stage into absent or writable regular
directories, while already-staged packages should be verified without
re-downloading into symlink targets.

### Promote / Reject

Promote:

- fail-fast validation that `STAGE_HF_PUBLIC_INPUTS=1` requires writable
  non-symlink HF model directories;
- `STAGE_HF_PUBLIC_INPUTS=0` for reruns against already-staged and verified HF
  packages, including symlinked local snapshots.

Reject:

- treating a missing `hf` CLI as a runtime-image problem;
- running `hf download --local-dir` into symlinked model-package aliases;
- letting public docs imply that first-time HF staging and already-staged HF
  verification are the same operation.

### Reason

This was a meaningful reproduction-doc fix. The release path now distinguishes
first-time public HF staging from repeat validation over already-staged inputs.
That keeps the vNext path portable while avoiding host-specific assumptions
about local Hugging Face cache topology.

## GGUF-339 - Dot20 Fresh-Checkout Full vNext Replay After HF Staging Fix

### Setup

- Date: 2026-06-30 UTC.
- Scope: rerun the full vNext release-readiness flow from a fresh checkout on
  `.20` after tightening HF staging mode in the verifier and documentation.
- Release boundary: stand-alone vNext draft path; not v0.2.0 or v0.2.1.
- Asset source: maintainer-only local mirror of future release assets for
  pre-publication validation.
- HF mode: `STAGE_HF_PUBLIC_INPUTS=0` because the pinned public HF packages
  were already staged and verified.
- Benchmark ladder: normal warmups -> `c1_128` uncapped strict -> `c1_2000` ->
  `c1_10000`.

### Result

The fresh-checkout full vNext release-readiness flow completed:

- serviceability checks;
- runtime image digest/native path checks;
- contract matrix checks;
- public GGUF asset gate;
- public HF input gate, including complete shard validation for symlinked HF
  snapshots;
- launch artifact generation;
- runtime vLLM argument-schema validation;
- host platform preflight for Dense TP8, MoE TP4, and MoE TP8;
- serving benchmark ladders for all five vNext profiles.

| Format | Profile | Host | Strict backend TPS | `c1_2000` backend TPS | `c1_10000` backend TPS | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| GGUF F16 | Dense 27B TP8 full-BAR/P2P-on | `.20` | `69.908` | `70.881` | `66.446` | strict-valid; ai-info 10K gate cleared |
| GGUF F16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | `.20` | `118.280` | `120.576` | `113.347` | strict-valid |
| HF FP16 | Dense 27B TP8 full-BAR/P2P-on | `.20` | `70.153` | `71.395` | `66.876` | strict-valid; ai-info 10K gate cleared |
| HF FP16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | `.20` | `113.968` | `116.585` | `108.544` | strict-valid |
| HF FP16 | Qwen3.6 35B-A3B MoE TP8 full-BAR/P2P-on | `.20` | `115.673` | `116.542` | `109.672` | strict-valid |

The readiness wrapper ended with:

```text
vNext full release reproduction path completed
```

### Promote / Reject

Promote:

- the release-readiness wrapper as the successful fresh-checkout deployment and
  benchmark reproduction path after the HF staging clarification;
- `STAGE_HF_PUBLIC_INPUTS=0` as the correct repeat mode for already-staged HF
  packages;
- all five vNext profile contracts as internally reproducible on `.20` using
  the stand-alone vNext package shape.

Reject:

- publishing vNext as a completed public reproduction before the same flow is
  rerun from final public GitHub Release asset URLs;
- updating v0.2.1 with vNext benchmark numbers;
- treating local `file://` release-asset validation as equivalent to a public
  release tag and asset URL replay.

### Reason

This repeat run verifies that the reproduction-doc fix was sufficient: no
further script or release-note changes were needed for the complete benchmark
ladder to pass. The remaining publication gate is not runtime behavior; it is
publishing the stand-alone vNext assets and rerunning the same command from the
final tag and public asset URLs.

## GGUF-340 - Public Asset Gate Fail-Closed Check

### Setup

- Date: 2026-06-30 UTC.
- Scope: verify that the release-readiness wrapper does not fall back to local
  files when a final-looking tag has no published GitHub Release assets.
- Serving benchmarks: disabled for this negative gate check.
- Runtime image/path and vLLM help capture: disabled so the check could focus
  on release-note gate order and public asset behavior.

### Result

The readiness wrapper passed the early local gates:

- serviceability;
- contract matrix.

It then stopped at the public GGUF asset gate when the release asset URL for
the Dense text-config archive returned `404`. The wrapper exited nonzero with
the intended public asset error and did not continue into HF staging, launch
artifact generation, host preflight, or serving benchmarks.

### Promote / Reject

Promote:

- the public GGUF asset gate as fail-closed when the final tag/assets are not
  published;
- the current gate order as preventing accidental fallback to retained local
  model files for public reproduction.

Reject:

- calling the vNext public reproduction complete before the final tag and
  GitHub Release assets exist;
- using maintainer-only local mirrors as the published reproduction source.

### Reason

This check confirms the remaining blocker is external release packaging, not a
hidden successful public path. The next completion step is to publish the
stand-alone vNext assets and rerun the same readiness command against the
public release URLs.

## GGUF-341 - vNext Split Asset Bundle Verification

### Setup

- Date: 2026-06-30 UTC.
- Scope: verify that the local simulated vNext release-asset bundle contains
  the uploadable files required by the draft release notes.
- Host label: `.30`.
- Asset class: maintainer-side simulated GitHub Release asset directory, not a
  public release URL.

### Result

The simulated upload set contains:

- Dense GGUF split parts: `28`
- MoE GGUF split parts: `36`
- Dense part manifest: `Qwen3.6-27B-F16.gguf.parts.sha256`
- MoE part manifest: `Qwen3.6-35B-A3B-F16.gguf.parts.sha256`
- Dense final hash sidecar: `Qwen3.6-27B-F16.gguf.sha256`
- MoE final hash sidecar: `Qwen3.6-35B-A3B-F16.gguf.sha256`
- Dense text-config archive: `Qwen3.6-27B-text-config-eosfix.tar.gz`

Part verification passed:

- Dense manifest rows verified: `28/28`
- MoE manifest rows verified: `36/36`

Streaming the parts in manifest order produced the final hashes required by
the draft release notes:

- Dense GGUF final SHA256:
  `b2347376b9bb7d12cf5f1d31c53ac4c60dd3d4b95068a09351187b328b5e9d89`
- MoE GGUF final SHA256:
  `1f2443bb0ff958943d091410c61120c181a0579b3bc85192029aa51d821d141c`
- Dense text-config archive SHA256:
  `9a85f0a18a012ed37f0eb4c42549569234af880c5f6e3ce908e0e45cca835719`

The reusable upload-bundle verifier also passed against the same simulated
asset directory:

```text
asset-text-config-ok: Qwen3.6-27B-text-config-eosfix.tar.gz sha=9a85f0a18a012ed37f0eb4c42549569234af880c5f6e3ce908e0e45cca835719
asset-parts-ok: Qwen3.6-27B-F16.gguf parts=28 final_sha=b2347376b9bb7d12cf5f1d31c53ac4c60dd3d4b95068a09351187b328b5e9d89
asset-parts-ok: Qwen3.6-35B-A3B-F16.gguf parts=36 final_sha=1f2443bb0ff958943d091410c61120c181a0579b3bc85192029aa51d821d141c
vNext release asset bundle verified: files=69 dense_parts=28 moe_parts=36
```

The deterministic upload-list helper also printed exactly `69` files from the
same bundle, starting with the Dense text-config archive and Dense manifest /
hash sidecars, then Dense parts, MoE manifest / hash sidecars, and MoE parts.

### Promote / Reject

Promote:

- the `.30` simulated asset bundle as hash-correct upload-preparation
  evidence;
- manifest-order streaming as a disk-safe way to verify final assembled GGUF
  hashes without duplicating the 27B and MoE files.
- `verify_vnext_release_asset_bundle.sh` as the pre-upload safety check for
  the hosted-asset path.
- `list_vnext_release_asset_uploads.sh` as the deterministic upload inventory
  helper for maintainer review.

Reject:

- treating the simulated asset directory as public reproduction evidence;
- declaring vNext complete before these files are attached to the final
  stand-alone GitHub Release and replayed from public release URLs.

### Reason

This closes the local asset-preparation evidence gap: the files needed for the
public GGUF asset path exist locally and match the release-note hashes. The
remaining blocker is publication and public replay, not missing local parts or
manifest/hash uncertainty.

## GGUF-342 - vNext Release Asset URL Verifier

### Setup

- Date: 2026-06-30 UTC.
- Scope: add and test a post-upload URL verifier for the `69` expected vNext
  GitHub Release assets.
- Positive test: maintainer local `file://` preflight against the simulated
  upload bundle.
- Negative test: final-looking public tag with no uploaded assets.

### Result

The URL verifier passed in explicit local preflight mode against the simulated
asset bundle and checked the exact expected asset names without downloading the
full multi-GB files:

```text
release-asset-url-ok: Qwen3.6-27B-text-config-eosfix.tar.gz
release-asset-url-ok: Qwen3.6-27B-F16.gguf.parts.sha256
release-asset-url-ok: Qwen3.6-27B-F16.gguf.sha256
...
vNext release asset URLs verified for tag: vnext-local-url-check
```

The same verifier failed closed against a non-existent public tag:

```text
error: release asset URL did not resolve: Qwen3.6-27B-text-config-eosfix.tar.gz
```

### Promote / Reject

Promote:

- `verify_vnext_release_asset_urls.sh` as the post-upload public URL inventory
  check before the expensive full release-readiness replay;
- explicit `ALLOW_LOCAL_ASSET_PREFLIGHT=1` for local `file://` simulations.

Reject:

- treating URL existence as benchmark reproduction;
- continuing to full public replay when the first required release asset URL
  is missing.

### Reason

This closes the cheap post-upload verification gap. After assets are attached
to the stand-alone GitHub Release, maintainers can confirm the expected public
asset URLs exist before spending time and bandwidth on full staging and
serving-benchmark replay.

## GGUF-343 - Dry-Run vNext Release Asset Upload Helper

### Setup

- Date: 2026-06-30 UTC.
- Scope: add a maintainer-side helper that wraps the vNext upload-bundle
  verifier, deterministic upload list, optional `gh release upload`, and
  post-upload URL verifier.
- Boundary: no tag, GitHub Release, or remote asset was created or modified by
  adding the helper.

### Result

`upload_vnext_release_assets.sh` was added as a dry-run-by-default upload
guard for the stand-alone vNext release asset set. In default mode, it verifies
the complete `69`-file release asset bundle, confirms the upload-list count,
and prints the upload plan without touching GitHub. A real upload requires the
maintainer to set `UPLOAD_VNEXT_RELEASE_ASSETS=1` after the stand-alone tag and
GitHub Release object exist. Later hardening also requires
`GH_RELEASE_REPO=owner/repo` for any real upload, and uploading to
`joe2gaan/localaiservers` is maintainer-only behind
`ALLOW_LOCALAISERVERS_RELEASE_UPLOAD=1`.

The helper does not create tags, create releases, edit release notes, delete
assets, or use `--clobber`.

The helper dry-run passed against the `.30` simulated vNext release-asset
bundle. It reported:

```text
dry-run: vNext release asset bundle verified for tag: vnext-local-upload-check
dry-run: upload file count: 69
```

No GitHub asset upload was attempted.

### Promote / Reject

Promote:

- `upload_vnext_release_assets.sh` as the guarded maintainer upload helper for
  the stand-alone vNext release asset path and as a guarded own-repository
  upload helper for users who publish equivalent assets in their own namespace;
- dry-run default behavior as the required first review step before any
  GitHub asset mutation;
- explicit post-upload URL verification before the full public readiness
  replay.

Reject:

- manual ad hoc selection of the `69` release asset files without the
  verifier/list helper;
- public-user instructions that can upload to the LocalAIServers release
  namespace without maintainer approval;
- treating a successful upload plan as proof of public reproduction;
- publishing vNext before the final public asset URLs pass and the full
  `verify_vnext_release_readiness.sh` replay succeeds from those URLs.

### Reason

The asset upload step is now scripted in the same fail-closed style as the
rest of the vNext package. This lowers the chance of a missing split part,
stale hash sidecar, or accidental overwrite while preserving the release
boundary: vNext remains incomplete until public release assets exist and the
public replay passes.

## GGUF-344 - Stand-Alone Public Readiness Command Fails Closed Before Assets

### Setup

- Date: 2026-06-30 UTC.
- Scope: run the documented stand-alone vNext release-readiness command without
  a local `RELEASE_ASSET_BASE` override, using a final-looking tag that has no
  public assets.
- Benchmark mode: `RUN_SERVING_BENCHMARKS=1`.
- HF staging mode: `STAGE_HF_PUBLIC_INPUTS=1`.

### Result

The readiness wrapper followed the documented release-note gate order:

1. serviceability;
2. contract matrix;
3. public GGUF asset gate.

Serviceability passed, including runtime image digest and runtime native-path
checks. The contract matrix passed, including positive profile contracts and
negative fail-closed cases for HF/GGUF format mismatch, Dense/MoE mismatch,
GGUF environment leakage into HF profiles, and patch-bundle hash mismatch.

The run then failed at the public GGUF asset gate when the first required
GitHub Release asset URL returned `404`:

```text
error: public GGUF asset gate failed; check release tag, release assets, part manifests, and final SHA256 locks
```

No HF download, launch-artifact generation, host preflight, container launch,
or serving benchmark ladder was attempted after the missing public asset.

### Promote / Reject

Promote:

- the documented release-readiness wrapper as fail-closed when public vNext
  assets are absent;
- the current gate order as preventing fallback to retained local GGUF files or
  validation-host paths before any benchmark claim is made.

Reject:

- treating local file-based asset simulations as completed public
  reproduction;
- publishing the stand-alone vNext claim before the exact release assets exist
  and the readiness command replays successfully from public URLs.

### Reason

This is the expected pre-publication behavior. The release notes and readiness
wrapper do not currently need a meaningful wording change for this failure
mode: they already require final public assets, final public URLs, and a full
`RUN_SERVING_BENCHMARKS=1` replay before the stand-alone vNext release can be
called reproduced from scratch.

## GGUF-345 - Dot30 Scratch-Repo Pre-Serving Readiness Replay

### Setup

- Date: 2026-06-30 UTC.
- Scope: copy the current draft repository state to a scratch checkout on
  `.30` and run the documented vNext readiness wrapper with the maintainer
  local asset preflight hook.
- Model root: existing NVMe-backed staged public-input root containing verified
  GGUF files and pinned HF model directories.
- Asset source: simulated `file://` GitHub Release asset bundle with explicit
  `ALLOW_LOCAL_ASSET_PREFLIGHT=1`.
- HF staging mode: `STAGE_HF_PUBLIC_INPUTS=0` because pinned HF model packages
  were already staged and verified.
- Serving benchmark mode: `RUN_SERVING_BENCHMARKS=0` with
  `ALLOW_PRE_SERVING_ONLY=1`.

### Result

The readiness wrapper completed every pre-serving gate:

- serviceability;
- runtime image digest check;
- runtime native-path check;
- contract matrix;
- public GGUF asset staging/reuse and profile checks;
- public HF input checks;
- launch artifact generation for all five profiles;
- runtime `vllm serve --help=all` argument-schema checks;
- host platform preflight for Dense TP8, MoE TP4, and MoE TP8 full-BAR/P2P-on
  profiles.

Observed gate output included:

```text
vNext public GGUF asset gate passed for tag: vnext-local-preflight-check
vNext public HF input gate passed
launch-artifact-ok: gguf-dense27b-tp8
launch-artifact-ok: gguf-moe35b-tp4
launch-artifact-ok: hf-dense27b-tp8
launch-artifact-ok: hf-moe35b-tp4
launch-artifact-ok: hf-moe35b-tp8
vllm-schema-ok: gguf-dense27b-tp8
vllm-schema-ok: gguf-moe35b-tp4
vllm-schema-ok: hf-dense27b-tp8
vllm-schema-ok: hf-moe35b-tp4
vllm-schema-ok: hf-moe35b-tp8
pre-serving-vnext-release-gates-passed
full-serving-benchmark-not-run: set RUN_SERVING_BENCHMARKS=1 for release reproduction success
```

The host preflight saw eight `gfx906` agents, eight likely GFX906 PCI
controller devices with `32 GiB` largest BAR, visible KFD topology, VBIOS
revision `113-D1631700-111`, and completed `rocm-smi --showtopo` for Dense TP8,
MoE TP4, and MoE TP8 release profiles.

### Promote / Reject

Promote:

- the readiness wrapper's pre-serving gate order as internally coherent from a
  scratch checkout;
- the existing GGUF asset reuse path for avoiding duplicate >100 GiB model
  copies when verified final GGUF files already exist in the model root;
- the wrapper's explicit refusal to claim full reproduction when the serving
  benchmark ladder is not run.

Reject:

- treating this pre-serving replay as completed benchmark reproduction;
- treating maintainer local `file://` assets as public release inputs;
- rerunning the same pre-serving-only replay as the next release proof unless
  the release notes or gate scripts change.

### Reason

This confirms that the release-note command sequence is actionable through the
deployment and validation gates that can run before serving. The next proof
step is not another pre-serving check; it is a full
`RUN_SERVING_BENCHMARKS=1` replay from public GitHub Release asset URLs after
the stand-alone vNext assets exist.

## GGUF-346 - Dot30 Scratch-Repo Full vNext Release-Readiness Replay

### Setup

- Date: 2026-06-30 UTC.
- Scope: run the documented stand-alone vNext release-readiness wrapper from
  the `.30` scratch checkout through the full serving benchmark ladder.
- Asset source: simulated `file://` GitHub Release asset bundle with explicit
  `ALLOW_LOCAL_ASSET_PREFLIGHT=1`.
- HF staging mode: `STAGE_HF_PUBLIC_INPUTS=0` because pinned public HF model
  packages were already staged and verified.
- Serving benchmark mode: `RUN_SERVING_BENCHMARKS=1`.
- Benchmark ladder: eight 2000-token warmups, uncapped strict, `c1_2000`, and
  `c1_10000`.

### Result

The readiness wrapper completed the same gate sequence as the release notes:

- serviceability;
- runtime image digest and native-path checks;
- contract matrix;
- GGUF release asset staging/reuse and hash checks;
- public HF input checks;
- launch artifact generation for all five profiles;
- runtime `vllm serve --help=all` argument-schema checks;
- host preflight for Dense TP8, MoE TP4, and MoE TP8;
- serving benchmark ladders for all five profiles.

The wrapper ended with:

```text
vNext full release reproduction path completed
```

Measured `.30` backend decode TPS:

| Format | Profile | Strict backend TPS | `c1_2000` backend TPS | `c1_10000` backend TPS | Status |
| --- | --- | ---: | ---: | ---: | --- |
| GGUF F16 | Dense 27B TP8 full-BAR/P2P-on | `70.123` | `71.122` | `66.607` | strict-valid; ai-info 10K gate cleared |
| GGUF F16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | `119.473` | `120.620` | `113.409` | strict-valid |
| HF FP16 | Dense 27B TP8 full-BAR/P2P-on | `70.259` | `71.445` | `66.909` | strict-valid; ai-info 10K gate cleared |
| HF FP16 | Qwen3.6 35B-A3B MoE TP4 full-BAR/P2P-on | `115.560` | `116.777` | `109.882` | strict-valid |
| HF FP16 | Qwen3.6 35B-A3B MoE TP8 full-BAR/P2P-on | `115.592` | `116.111` | `109.292` | strict-valid |

### Promote / Reject

Promote:

- `.30` as an equivalent-performing validation lane for the current vNext
  release-candidate package;
- the documented readiness wrapper as capable of replaying every current
  vNext profile through deployment, warmups, strict, and fixed-token tiers;
- the normal benchmark ladder as the release-candidate replay path.

Reject:

- treating this as completed public reproduction before final stand-alone
  vNext GitHub Release assets exist;
- treating maintainer local `file://` assets as public release inputs;
- updating v0.2.0 or v0.2.1 with these values.

### Reason

This replay is strong package-path evidence because it starts from a scratch
checkout and exercises every generated profile artifact and benchmark tier on a
second host. The remaining blocker is public packaging, not local deployment:
the same command must pass from the final stand-alone vNext tag and public
GitHub Release asset URLs before vNext is called publicly reproduced.

## GGUF-347 - Stand-Alone vNext Public-Asset Fail-Closed Replay

### Setup

- Date: 2026-06-30 UTC.
- Scope: rerun the documented stand-alone vNext release-readiness wrapper after
  adding the final public replay checklist.
- Release tag argument: `vnext-public-assets-not-yet-published-check`.
- Asset source: default public GitHub Release asset URLs; no
  `RELEASE_ASSET_BASE` override.
- Local asset preflight: not enabled.
- HF staging mode: `STAGE_HF_PUBLIC_INPUTS=1`.
- Serving benchmark mode: `RUN_SERVING_BENCHMARKS=1`.
- Runtime image and vLLM-help capture were skipped for this boundary check with
  `CHECK_RUNTIME_IMAGE=0`, `CHECK_RUNTIME_PATHS=0`, and
  `CAPTURE_VLLM_HELP=0`.

### Result

The readiness wrapper passed the early non-serving gates:

- serviceability;
- contract matrix;
- intentional mismatch rejection cases.

It then stopped at the public GGUF asset gate before any serving benchmark
launch because the stand-alone vNext public release assets do not exist yet.

Observed boundary:

```text
vNext release readiness: public GGUF assets
curl: (22) The requested URL returned error: 404
error: public GGUF asset gate failed; check release tag, release assets, part manifests, and final SHA256 locks
```

The command returned nonzero with exit code `2`.

### Promote / Reject

Promote:

- the final public replay checklist as aligned with current gate behavior;
- the readiness wrapper's fail-closed behavior before serving launch when the
  required public GGUF release assets are absent.

Reject:

- treating local simulated release assets as public reproduction evidence;
- calling vNext complete before the final stand-alone tag, GitHub Release
  assets, and public URL verifier are present;
- rerunning another public-input readiness check until the final assets or
  release tag change.

### Reason

This confirms the current draft remains honest about the public reproduction
boundary. vNext is internally reproducible from maintainer-staged assets, but
public reproduction is still blocked on publishing the stand-alone vNext tag
and the exact release assets, then rerunning the same readiness command without
local asset overrides.

## GGUF-348 - vNext Release Asset Bundle Rebuild And Upload Dry-Run Recheck

### Setup

- Date: 2026-06-30 UTC.
- Scope: follow the draft vNext publisher split procedure from the release
  notes and revalidate the upload bundle shape.
- Source inputs:
  - full Dense F16 GGUF file with final SHA256
    `b2347376b9bb7d12cf5f1d31c53ac4c60dd3d4b95068a09351187b328b5e9d89`;
  - full MoE F16 GGUF file with final SHA256
    `1f2443bb0ff958943d091410c61120c181a0579b3bc85192029aa51d821d141c`;
  - Dense text-config archive with SHA256
    `9a85f0a18a012ed37f0eb4c42549569234af880c5f6e3ce908e0e45cca835719`.
- Split command shape: `split -b 1900M -d -a 4`.
- Upload helper mode: dry run only; no GitHub Release, tag, asset, or remote
  repository mutation.

### Result

The split procedure rebuilt the expected upload inventory:

- Dense GGUF part files: `28`
- MoE GGUF part files: `36`
- part manifests: `2`
- final assembled-file hash sidecars: `2`
- Dense text-config archive: `1`
- total uploadable files: `69`

The bundle verifier passed:

```text
asset-text-config-ok: Qwen3.6-27B-text-config-eosfix.tar.gz sha=9a85f0a18a012ed37f0eb4c42549569234af880c5f6e3ce908e0e45cca835719
asset-parts-ok: Qwen3.6-27B-F16.gguf parts=28 final_sha=b2347376b9bb7d12cf5f1d31c53ac4c60dd3d4b95068a09351187b328b5e9d89
asset-parts-ok: Qwen3.6-35B-A3B-F16.gguf parts=36 final_sha=1f2443bb0ff958943d091410c61120c181a0579b3bc85192029aa51d821d141c
vNext release asset bundle verified: files=69 dense_parts=28 moe_parts=36
```

The deterministic upload-list helper printed exactly `69` file paths. The
guarded upload helper also passed in dry-run mode:

```text
dry-run: vNext release asset bundle verified for tag: vnext-local-upload-recheck
dry-run: upload file count: 69
```

### Promote / Reject

Promote:

- the release-note split procedure as capable of recreating the expected
  hosted-asset bundle from the full GGUF files and Dense text-config archive;
- `verify_vnext_release_asset_bundle.sh` as the pre-upload integrity gate;
- `list_vnext_release_asset_uploads.sh` as the deterministic `69`-file upload
  inventory;
- `upload_vnext_release_assets.sh` dry-run mode as a safe maintainer guard
  before any GitHub asset upload.

Reject:

- treating this dry-run as public reproduction;
- treating the local rebuilt bundle as a public input before it is attached to
  the final stand-alone vNext GitHub Release;
- changing v0.2.0 or v0.2.1 release claims based on this asset-packaging
  check.

### Reason

The publisher-side asset instructions now have current local evidence, not
only stale prior logs. The remaining public blocker is external publication:
create the final stand-alone vNext tag and GitHub Release, attach these exact
assets, verify the public release URLs, and rerun
`verify_vnext_release_readiness.sh` with `RUN_SERVING_BENCHMARKS=1` without
local asset overrides.

## GGUF-349 - Local File-Backed vNext Public Asset Gate Recheck

### Setup

- Date: 2026-06-30 UTC.
- Scope: run the public GGUF asset staging gate against the freshly rebuilt
  `69`-file release-asset bundle using a local `file://` base.
- Release tag argument: `vnext-local-public-asset-recheck`.
- Asset source: `/tmp/localaiservers-vnext-asset-bundle-recheck-20260630T085951Z/assets`.
- Target model root:
  `/tmp/localaiservers-vnext-public-assets-recheck-20260630T095631Z/model-root`.
- Local asset mode: `ALLOW_LOCAL_ASSET_PREFLIGHT=1`.
- Remote mutation: none. No GitHub tag, release, or asset was created or
  modified.

### Result

The public asset staging gate reconstructed and verified the hosted-asset
inputs from the local simulated release asset base:

```text
Qwen3.6-27B-F16.gguf.tmp: OK
Qwen3.6-27B-F16.gguf: OK
Qwen3.6-35B-A3B-F16.gguf.tmp: OK
Qwen3.6-35B-A3B-F16.gguf: OK
vNext GGUF assets staged under: /tmp/localaiservers-vnext-public-assets-recheck-20260630T095631Z/model-root
public-asset-profile-ok: gguf-dense27b-tp8
public-asset-profile-ok: gguf-moe35b-tp4
vNext public GGUF asset gate passed for tag: vnext-local-public-asset-recheck
```

The run verified:

- Dense text-config archive;
- Dense GGUF part manifest and final assembled GGUF SHA256;
- MoE GGUF part manifest and final assembled GGUF SHA256;
- GGUF Dense TP8 profile compatibility with the staged asset tree;
- GGUF MoE TP4 profile compatibility with the staged asset tree.

### Promote / Reject

Promote:

- `verify_vnext_public_assets.sh` as a working staging gate for the exact
  hosted-asset layout expected by the stand-alone vNext release;
- the rebuilt `69`-file bundle as internally consistent with the locked Dense
  and MoE full-file hashes;
- the fail-closed profile checks for GGUF Dense and GGUF MoE after asset
  staging.

Reject:

- treating this as public reproduction; the run used a local `file://` asset
  base rather than public GitHub Release URLs;
- treating URL or local asset staging as benchmark reproduction; the serving
  benchmark ladder was not run in this check;
- uploading assets to the LocalAIServers remote from public-user instructions.
  Users may upload equivalent assets to their own repository, but
  `joe2gaan/localaiservers` publication remains maintainer-only.

### Reason

This closes the local asset-shape loop after rebuilding the upload bundle from
the final full GGUF files. The remaining gate is still public and external:
publish the stand-alone vNext tag and release assets, verify the public asset
URLs without `RELEASE_ASSET_BASE`, and rerun
`verify_vnext_release_readiness.sh` with `RUN_SERVING_BENCHMARKS=1` from the
public inputs.

## GGUF-350 - vNext Release-Note Instruction Audit

### Setup

- Date: 2026-06-30 UTC.
- Scope: audit the draft vNext release note and profile README for the explicit
  public reproduction path after the normal-inference wording update.
- Files checked:
  - `releases/draft-vnext-gfx906-rocm72-gguf-hf-repro.md`
  - `qwen36-gfx906/profiles/vnext/README.md`
  - `qwen36-gfx906/verify_vnext_release_readiness.sh`
  - `qwen36-gfx906/verify_vnext_release_asset_urls.sh`
- Remote mutation: none. No GitHub tag, release, or asset was created or
  modified.

### Result

The vNext release note now describes:

- clean-checkout public inputs;
- large local SSD/NVMe-backed storage setup;
- public HF staging and validation;
- public GGUF split-asset staging and validation;
- profile-driven launch artifact generation;
- normal `vllm serve` usage through the OpenAI-compatible endpoint;
- benchmark validation through the normal warmup, `c1_128` uncapped strict,
  `c1_2000`, and `c1_10000` ladder; and
- the requirement that full release success needs
  `RUN_SERVING_BENCHMARKS=1`.

The profile README had one practical gap: it exported `LOCAL_MODEL_ROOT`,
`LOCAL_HF_CACHE`, and `LOCAL_RUNTIME_ROOT`, but did not explicitly create those
directories before the example launcher commands. The profile README was updated
to run:

```sh
mkdir -p "$LOCAL_MODEL_ROOT" "$LOCAL_HF_CACHE" "$LOCAL_RUNTIME_ROOT"
```

after the exports, matching the release-note setup block.

The checks rerun after the edit passed:

```text
vNext developer serviceability check passed
vNext contract matrix passed
vNext release asset URLs verified for tag: vnext-local-public-asset-recheck
```

The default URL check also failed closed for a non-existent public tag:

```text
error: release asset URL did not resolve: Qwen3.6-27B-text-config-eosfix.tar.gz
```

### Promote / Reject

Promote:

- the vNext draft release note as the primary step-by-step public reproduction
  path for the unpublished stand-alone release;
- the profile README as a mirrored developer-facing path after adding the
  directory creation step;
- the normal-inference wording that clarifies the generated wrapper starts
  ordinary `vllm serve` and the begin-think proxy is only a benchmark harness;
- the asset URL checker's fail-closed behavior before publication.

Reject:

- claiming final public reproduction before the stand-alone vNext tag and
  public GitHub Release assets exist;
- treating `file://` asset preflight as public reproduction;
- treating non-serving gates as a complete benchmark reproduction.

### Reason

The release-note path is now clearer for a public user who starts from a clean
checkout and a local NVMe-backed workspace. The full goal remains open because
the final stand-alone vNext release assets are not yet public and the full
`RUN_SERVING_BENCHMARKS=1` replay has not been run from those public URLs.

## GGUF-351 - vNext Readiness Wrapper Replay To Host Preflight

### Setup

- Date: 2026-06-30 UTC.
- Scope: run the release-note readiness wrapper as far as possible on the local
  workstation using previously staged public-shape inputs.
- GGUF source: local simulated release asset base
  `/tmp/localaiservers-vnext-asset-bundle-recheck-20260630T085951Z/assets`.
- Model root:
  `/tmp/localaiservers-vnext-public-assets-recheck-20260630T095631Z/model-root`.
- HF inputs: symlinked staged public HF downloads under the model root, pointing
  at `/tmp/localaiservers-vnext-real-hf-20260629T192018Z/models`.
- Local asset mode: `ALLOW_LOCAL_ASSET_PREFLIGHT=1`.
- Runtime schema: skipped with `CAPTURE_VLLM_HELP=0` for this non-serving local
  workstation replay.
- Serving benchmarks: not run.
- Remote mutation: none. No GitHub tag, release, or asset was created or
  modified.

Command shape:

```sh
ALLOW_LOCAL_ASSET_PREFLIGHT=1 \
RELEASE_ASSET_BASE=file:///tmp/localaiservers-vnext-asset-bundle-recheck-20260630T085951Z/assets \
CHECK_RUNTIME_IMAGE=0 \
CHECK_RUNTIME_PATHS=0 \
CAPTURE_VLLM_HELP=0 \
ALLOW_PRE_SERVING_ONLY=1 \
RUN_SERVING_BENCHMARKS=0 \
qwen36-gfx906/verify_vnext_release_readiness.sh \
  --release-tag vnext-local-public-asset-recheck \
  --model-root /tmp/localaiservers-vnext-public-assets-recheck-20260630T095631Z/model-root \
  --hf-cache /tmp/localaiservers-vnext-real-hf-20260629T192018Z/hf-cache \
  --runtime-root /tmp/localaiservers-vnext-readiness-local-preflight-20260630T105229Z
```

### Result

The readiness wrapper passed:

```text
vNext release readiness: serviceability
vNext developer serviceability check passed
vNext release readiness: contract matrix
vNext contract matrix passed
vNext release readiness: public GGUF assets
vNext public GGUF asset gate passed for tag: vnext-local-public-asset-recheck
vNext release readiness: public HF inputs
vNext public HF input gate passed
vNext release readiness: launch artifacts
launch-artifact-ok: gguf-dense27b-tp8
launch-artifact-ok: gguf-moe35b-tp4
launch-artifact-ok: hf-dense27b-tp8
launch-artifact-ok: hf-moe35b-tp4
launch-artifact-ok: hf-moe35b-tp8
```

It then failed at the host platform preflight, as expected for this local
workstation:

```text
profile=dense27b_tp8_fullbar_p2pon
required_gpu_count=8
min_gpu_bar_gib=32
expected_gpu_vbios=113-D1631700-111
FAIL: found 1 likely gfx906 PCI devices; expected at least 8
FAIL: 0 likely gfx906 GPUs have largest BAR >= 32 GiB; expected at least 8
error: host platform preflight failed for dense27b_tp8_fullbar_p2pon; this host is not comparable with the full-BAR/P2P-on release lane
```

### Promote / Reject

Promote:

- `verify_vnext_release_readiness.sh` as correctly sequencing the documented
  release-note gates through public GGUF assets, public HF inputs, and launch
  artifact generation;
- symlinked public HF staging directories when they resolve to real directories
  with complete shard sets;
- the host preflight fail-closed behavior on a non-comparable local workstation.

Reject:

- claiming this as full release reproduction; the public assets were local
  `file://` preflight assets, runtime schema was skipped, host preflight failed,
  and serving benchmarks were not run;
- using this workstation as evidence for the full-BAR/P2P-on release lane;
- treating `ALLOW_PRE_SERVING_ONLY=1` as a publication-complete path.

### Reason

This replay proves the readiness wrapper can carry a public-style input tree
through the non-serving gates and stops for the right reason on non-comparable
hardware. The remaining proof must be run on a comparable full-BAR/P2P-on GFX906
lane from final public GitHub Release asset URLs with `RUN_SERVING_BENCHMARKS=1`.

## GGUF-352 - Comparable-Lane Host Preflight From Current Script

### Setup

- Date: 2026-06-30 UTC.
- Scope: copy the current host-platform preflight script to `.20` and `.30` and
  run it read-only for the three published full-BAR/P2P-on profile families.
- Profiles checked:
  - `dense27b_tp8_fullbar_p2pon`
  - `moe35b_tp4_fullbar_p2pon`
  - `moe35b_tp8_fullbar_p2pon`
- Remote mutation: only a temporary preflight directory under `/tmp` was
  created on each host. No containers, benchmarks, tags, releases, or model
  downloads were started.

### Result

Both `.20` and `.30` passed host platform preflight for all three profiles:

- amdgpu kernel module loaded;
- amdgpu module version reported as `6.8.5`;
- `rocminfo` reported `8` gfx906 agents;
- `8` likely gfx906 PCI devices were found;
- all `8` likely gfx906 GPUs reported largest BAR `32.00` GiB;
- KFD topology was visible;
- `rocm-smi` reported `8` GPUs with VBIOS `113-D1631700-111`;
- `rocm-smi --showtopo` completed.

### Promote / Reject

Promote:

- `.20` and `.30` as comparable full-BAR/P2P-on lanes for the host preflight
  portion of the vNext release-note reproduction path;
- the current `check_host_platform_prereqs.sh` preflight checks as correctly
  accepting both validation lanes and rejecting the non-comparable local
  workstation in `GGUF-351`.

Reject:

- treating host preflight as benchmark reproduction;
- treating current host readiness as proof that the final public asset replay
  has been completed;
- starting a full readiness replay before enough NVMe workspace is available
  for the complete public model tree and runtime artifacts.

### Reason

The hardware/platform part of the release-note path is now validated on both
intended lanes. The next blocker is operational rather than platform-related:
the visible NVMe mount on both hosts is currently nearly full, so a full
all-profile replay from public assets requires either cleanup or a different
large local workspace before running `RUN_SERVING_BENCHMARKS=1`.

## GGUF-353 - Comparable-Lane vNext Readiness Replay Through Pre-Serving Gates

### Setup

- Date: 2026-06-30 UTC.
- Host: `.30`.
- Scope: copy the current worktree to a temporary directory on `.30` and run the
  vNext readiness wrapper through all non-serving gates on a comparable
  full-BAR/P2P-on lane.
- Model root:
  `/usr/share/ollama/vnext-release-repro/public-staging-20260629/local-models`.
- HF cache:
  `/usr/share/ollama/vnext-release-repro/public-staging-20260629/hf-cache`.
- Runtime root:
  `/usr/share/ollama/vnext-release-repro/public-staging-20260629/runtime-current-20260630T110859Z`.
- GGUF release assets: local simulated release-asset base under
  `/usr/share/ollama/vnext-release-repro/release-assets/vnext-gguf-sim`.
- Local asset mode: `ALLOW_LOCAL_ASSET_PREFLIGHT=1`.
- Serving benchmarks: not run; `RUN_SERVING_BENCHMARKS=0` and
  `ALLOW_PRE_SERVING_ONLY=1`.
- Remote mutation: temporary current-code copy and runtime output only. No
  GitHub tag, release, remote branch, or public asset was created or modified.

### Result

The readiness wrapper exited with status `0` for the pre-serving gate set:

```text
runtime-image-digest-ok: joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b@sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e
runtime-native-paths-ok: joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b
vNext developer serviceability check passed
vNext contract matrix passed
vNext public GGUF asset gate passed for tag: vnext-local-public-asset-recheck
vNext public HF input gate passed
launch-artifact-ok: gguf-dense27b-tp8
launch-artifact-ok: gguf-moe35b-tp4
launch-artifact-ok: hf-dense27b-tp8
launch-artifact-ok: hf-moe35b-tp4
launch-artifact-ok: hf-moe35b-tp8
vllm-schema-ok: gguf-dense27b-tp8
vllm-schema-ok: gguf-moe35b-tp4
vllm-schema-ok: hf-dense27b-tp8
vllm-schema-ok: hf-moe35b-tp4
vllm-schema-ok: hf-moe35b-tp8
host platform preflight completed
pre-serving-vnext-release-gates-passed
full-serving-benchmark-not-run: set RUN_SERVING_BENCHMARKS=1 for release reproduction success
```

The host preflight passed for:

- `dense27b_tp8_fullbar_p2pon`;
- `moe35b_tp4_fullbar_p2pon`;
- `moe35b_tp8_fullbar_p2pon`.

### Promote / Reject

Promote:

- the current readiness wrapper sequence through all non-serving gates on a
  comparable lane;
- the runtime image digest/native-path verification;
- vLLM schema validation for all five profiles;
- `.30` as ready for the next full serving replay, subject to workspace space
  and ordinary workload checks.

Reject:

- claiming full release reproduction; the serving benchmark ladder was
  intentionally not run in this pass;
- claiming public reproduction; the GGUF assets still came from a local
  simulated release-asset base rather than final public GitHub Release URLs.

### Reason

This is the strongest non-serving proof so far: on comparable hardware, the
current release-note path reaches the exact designed boundary and refuses to
claim full success without `RUN_SERVING_BENCHMARKS=1`. The next step is to run
the same wrapper on `.30` with `RUN_SERVING_BENCHMARKS=1` and compare the
resulting benchmark summaries against the vNext table.

## GGUF-354 - vNext Full Serving Reproduction Replay From Public-Shaped Local Assets

### Setup

- Date: 2026-06-30 UTC.
- Host: `.30`.
- Scope: run the vNext release-note deployment path end-to-end with
  `RUN_SERVING_BENCHMARKS=1` across all five launch profiles.
- Runtime image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Public-shape inputs:
  - GGUF assets staged from a local file-backed release-asset mirror;
  - HF model inputs staged from the public-input layout used by the vNext
    wrappers;
  - `ALLOW_LOCAL_ASSET_PREFLIGHT=1` because final public release URLs do not
    exist yet.
- Benchmark ladder per profile:
  - normal pre-measure warmups;
  - `c1_128_strict`;
  - `c1_2000`;
  - `c1_10000`.
- Launch behavior: one profile at a time through the generated `vllm serve`
  wrapper. The begin-think proxy was used only by the Qwen benchmark harness,
  not as the normal serving interface.
- Remote mutation: temporary runtime outputs only. No GitHub tag, release,
  remote branch, or public asset was created or modified.

### Result

The wrapper completed with `readiness_rc=0` and produced benchmark summaries
for all five profiles.

| Profile | Strict backend TPS | c1_2000 backend TPS | c1_10000 backend TPS | Strict gate |
| --- | ---: | ---: | ---: | --- |
| `gguf-dense27b-tp8` | 69.873 | 71.161 | 66.638 | True |
| `gguf-moe35b-tp4` | 119.363 | 119.808 | 112.755 | True |
| `hf-dense27b-tp8` | 70.474 | 71.326 | 66.810 | True |
| `hf-moe35b-tp4` | 115.592 | 116.062 | 109.240 | True |
| `hf-moe35b-tp8` | 114.652 | 115.895 | 109.001 | True |

The normal `vllm serve` endpoint came up for each profile, and the wrapper
cleaned up the current-profile container after the replay.

### Promote / Reject

Promote:

- the vNext release-note deployment shape as executable end-to-end on a
  comparable full-BAR/P2P-on lane;
- the generated profile wrappers as normal `vllm serve` launchers for both
  GGUF and HF model formats;
- the benchmark ladder ordering: warmups, strict, `c1_2000`, then `c1_10000`;
- the GGUF dense and GGUF MoE profile numbers as reproduced from
  public-shaped local assets;
- the HF dense and HF MoE profile numbers as reproduced from public-shaped
  local inputs.

Reject:

- claiming final public release reproduction until the same replay is run
  against live public release URLs without `ALLOW_LOCAL_ASSET_PREFLIGHT=1`;
- making the begin-think proxy part of normal serving documentation;
- treating fixed-token warmup or fixed-token tiers as strict-gate evidence;
- changing v0.2.1 or vNext public claims solely from this internal-lane replay.

### Reason

This pass proves that the written deployment path is operational: the release
notes can describe staging inputs, generating launch artifacts, starting normal
`vllm serve`, and running the benchmark ladder without relying on ad hoc manual
launch commands. The remaining promotion blocker is publication state, not
runtime behavior: final public reproduction still requires the same flow from
live release assets and a stable public tag.

## GGUF-355 - vNext Public Asset URL Gate Boundary Check

### Setup

- Date: 2026-06-30 UTC.
- Scope: verify whether the final public-URL precondition for vNext release
  reproduction is currently satisfied.
- Command path:
  - `./verify_vnext_release_asset_urls.sh vnext-local-public-asset-recheck`
    without a release-asset override;
  - the same command with `ALLOW_LOCAL_ASSET_PREFLIGHT=1` and a local
    file-backed release-asset mirror.

### Result

The public GitHub Release URL gate failed without the local override:

```text
error: release asset URL did not resolve: Qwen3.6-27B-text-config-eosfix.tar.gz
```

The same gate passed against the local file-backed mirror and verified the
expected `69` release-asset URLs/files for the internal replay tag.

### Promote / Reject

Promote:

- the URL gate as correctly distinguishing real public release assets from
  maintainer-local mirrors;
- the local mirror as coherent with the expected vNext asset manifest;
- the release-note boundary that final publication requires the same replay
  without `ALLOW_LOCAL_ASSET_PREFLIGHT=1`.

Reject:

- calling the vNext package publicly reproduced before the final stand-alone
  tag and GitHub Release assets exist;
- rerunning the expensive serving ladder as a substitute for publishing and
  verifying the public asset URLs.

### Reason

The full serving replay passed from public-shaped local assets, but the final
public proof still depends on publication state. The next completion attempt
must start from the final vNext tag or pinned release commit, verify the real
GitHub Release asset URLs, and then run `RUN_SERVING_BENCHMARKS=1` without the
local asset override.

## GGUF-356 - Clean Checkout vNext Tree Visibility Gate

### Setup

- Date: 2026-06-30 UTC.
- Scope: test whether a clean checkout of current `main` contains the vNext
  release-note reproduction package.
- Method: clone current `main` into a temporary clean directory and check for
  the vNext draft release note, profile README, readiness script, and
  experiment log.

### Result

The clean checkout was clean, but the vNext package files were not present in
the tracked tree at current `main`:

```text
missing releases/draft-vnext-gfx906-rocm72-gguf-hf-repro.md
missing qwen36-gfx906/verify_vnext_release_readiness.sh
missing qwen36-gfx906/profiles/vnext/README.md
missing docs/gguf-gfx906-experiment-log-20260625.md
```

### Promote / Reject

Promote:

- a fresh-checkout tree-content gate before final publication;
- the release-note requirement that the final vNext tag or pinned release commit
  must contain the vNext profile directory and verification/launcher scripts.

Reject:

- treating the current dirty local worktree as proof of final release
  reproducibility;
- starting another expensive serving replay from current `main` until the
  final release tree and public release assets exist.

### Reason

The runtime path has been exercised successfully, but final reproduction is
stronger than runtime success from a local working directory. The release tag
must contain the scripts, profile files, overlays, and release notes needed for
a user to start from a clean checkout before public asset URL verification and
full serving replay can be meaningful.

## GGUF-357 - Local Proof Commit Clean Checkout Gate

### Setup

- Date: 2026-06-30 UTC.
- Scope: test whether the vNext package is self-contained once the current
  release-note, profile, overlay, launcher, verifier, and experiment-log files
  are committed to a local proof branch.
- Local proof branch: `vnext-local-repro-package-proof`.
- Package proof commit: `04c2c51`.
- The branch was re-cloned and rechecked after this log entry was added.
- Method:
  - clone the local proof branch into a temporary clean checkout;
  - check that the vNext release note, profile directory, overlays, launcher,
    benchmark runner, readiness verifier, and experiment log exist;
  - run `./verify_vnext_serviceability.sh`;
  - run `./verify_vnext_contract_matrix.sh`;
  - run `./verify_vnext_release_asset_urls.sh` without a local override;
  - rerun the same asset URL gate against the maintainer-only local asset
    mirror with `ALLOW_LOCAL_ASSET_PREFLIGHT=1`.

### Result

The clean checkout of the current local proof tip contained the required vNext
package files:

```text
present releases/draft-vnext-gfx906-rocm72-gguf-hf-repro.md
present qwen36-gfx906/verify_vnext_release_readiness.sh
present qwen36-gfx906/vnext_repro_launcher.sh
present qwen36-gfx906/run_vnext_profile_benchmark.sh
present qwen36-gfx906/profiles/vnext
present qwen36-gfx906/overlays
present docs/gguf-gfx906-experiment-log-20260625.md
```

The clean-checkout package passed both lightweight gates:

```text
vNext developer serviceability check passed
vNext contract matrix passed
```

The public GitHub Release URL gate still failed without a published vNext
asset set:

```text
error: release asset URL did not resolve: Qwen3.6-27B-text-config-eosfix.tar.gz
```

The same clean-checkout gate passed against the maintainer-only local asset
mirror and verified all `69` expected asset URLs/files for the internal replay
tag.

Both validation lanes were idle at the GPU level during this check, but neither
exposed the large model/asset staging roots needed for another five-profile
serving replay without restaging large assets first.

### Promote / Reject

Promote:

- the local proof commit as a valid tree-content proof for the vNext package;
- the serviceability and contract-matrix gates as clean-checkout checks, not
  dirty-worktree-only checks;
- the release-note requirement that final public replay must use real GitHub
  Release asset URLs with no local override.

Reject:

- calling the goal complete from the local proof commit alone;
- treating `ALLOW_LOCAL_ASSET_PREFLIGHT=1` as a public reproduction path;
- rerunning the full serving ladder before the final public tag and asset URLs
  exist, unless the model/asset staging roots are deliberately restored for a
  targeted validation replay.

### Reason

The package is now self-contained at the Git tree level in a local proof
commit, which closes the earlier clean-checkout visibility gap. The remaining
release-note accuracy blocker is publication state: the final vNext tag and
GitHub Release assets must exist, then the same readiness path must be replayed
without local asset overrides and with `RUN_SERVING_BENCHMARKS=1`.

## GGUF-358 - Clean Checkout Full Serving Replay Repeatability

### Setup

- Date: 2026-06-30 UTC.
- Host: `.30`.
- Scope: repeat the vNext release-note deployment path from a clean checkout of
  the local proof commit with `RUN_SERVING_BENCHMARKS=1`.
- Runtime image:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
- Public-shape inputs:
  - GGUF assets staged from a local file-backed release-asset mirror;
  - HF model inputs staged from the public-input layout used by the vNext
    wrappers;
  - `ALLOW_LOCAL_ASSET_PREFLIGHT=1` because final public release URLs do not
    exist yet.
- Benchmark ladder per profile:
  - normal pre-measure warmups;
  - `c1_128_strict`;
  - `c1_2000`;
  - `c1_10000`.
- Launch behavior: one profile at a time through the generated `vllm serve`
  wrapper. The begin-think proxy was used only by the Qwen benchmark harness,
  not as the normal serving interface.

### Result

The wrapper completed with `readiness_rc=0`, ended with
`vNext full release reproduction path completed`, and produced benchmark
summaries for all five profiles.

| Profile | Strict backend TPS | c1_2000 backend TPS | c1_10000 backend TPS | Strict gate |
| --- | ---: | ---: | ---: | --- |
| `gguf-dense27b-tp8` | 70.097 | 71.081 | 66.543 | True |
| `gguf-moe35b-tp4` | 118.948 | 120.961 | 113.684 | True |
| `hf-dense27b-tp8` | 70.306 | 71.254 | 66.796 | True |
| `hf-moe35b-tp4` | 114.557 | 113.436 | 108.044 | True |
| `hf-moe35b-tp8` | 115.509 | 116.704 | 109.754 | True |

The normal `vllm serve` endpoint came up for each profile. The replay confirmed
that normal inference is available directly through the OpenAI-compatible vLLM
endpoint and that the begin-think proxy is benchmark-harness-only.

### Promote / Reject

Promote:

- the current release-note sequence as repeatable from a clean checkout when
  model inputs and release-shaped assets are present;
- the generated wrappers as normal `vllm serve` launchers for GGUF and HF
  profiles;
- the benchmark ladder ordering: warmups, strict, `c1_2000`, then `c1_10000`;
- using a validation band instead of exact TPS equality for repeat runs, while
  still requiring strict rows to pass the Qwen gate and Dense c1_10000 to clear
  the 65 TPS gate.

Reject:

- calling this final public reproduction because it still used
  `ALLOW_LOCAL_ASSET_PREFLIGHT=1` and a local file-backed asset mirror;
- treating minor TPS drift between successful repeats as a release-note failure;
- making the begin-think proxy part of ordinary inference instructions;
- updating v0.2.0 or v0.2.1 claims from this vNext release-candidate replay.

### Reason

The repeat proves that the runtime path and instructions are not a one-off lane
success. The remaining blocker is publication state, not launch behavior: the
stand-alone vNext tag and public GitHub Release assets must exist, and the same
readiness path must pass without local asset overrides.
