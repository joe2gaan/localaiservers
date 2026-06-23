# GFX906 Source Kernel Inventory - 2026-06-12

This is a sanitized public copy of the source note used to document
LocalAIServers GFX906 preservation work. It is published as evidence of source-level
kernel, runtime, graph, and collective-communication investigation. Experimental
terms such as "winner" or "high-water" are source-record labels;
funder-facing claims should use the cautious summaries in
[gfx906-technical-progress-summary.md](gfx906-technical-progress-summary.md),
[gfx906-experimental-methodology.md](gfx906-experimental-methodology.md), and
[gfx906-current-research-roadmap.md](gfx906-current-research-roadmap.md).

## Sanitization Note

The public copy removes or generalizes private IPs, local home-directory paths,
credentials, and site-identifying details when present. Anonymized benchmark lane
labels such as `.10`, `.20`, `.30`, and `.50` are preserved because they function as
run identifiers rather than reachable host addresses.

## Sanitized Source Content

# gfx906 Source Kernel Inventory - 2026-06-12

This inventory lists the source-level kernel and graph-runtime work that has
been adapted, patched, or directly tested for gfx906 during the Qwen3.6
campaign. It intentionally excludes pure launch/config sweeps unless the config
was tied to a source path.

## Current Readout

- As of 2026-06-20,
  [qwen36-gfx906/README.md](../qwen36-gfx906/README.md) is the authoritative
  latest technical source. Dense 27B clears the ai-info 10K gate at the
  published v0.2.0 release boundary with `MAX_MODEL_LEN=131072` and eight
  pre-measure warmups.
- v0.1.0 remains the older published GitHub Release boundary. v0.2.0 is the
  published ROCm7.2 Dense/MoE release boundary. GitHub Releases remain canonical
  for published claim boundaries.
- The same ROCm7.2 experimental release image covers both active contracts with
  model-specific env and overlay selection:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`,
  Docker Hub manifest digest
  `sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e`.
  Docker Hub remains an evergreen artifact distribution channel; TPS claims
  should stay in GitHub Releases, repository docs, and benchmark artifacts.
- Dense 27B TP8 full-BAR/P2P-on portable performance on `.20`: strict backend
  TPS `69.514`, `c1_2000` backend TPS `70.347`, `c1_10000` backend TPS
  `66.069`; note `strict gate valid`.
- Qwen3.6 35B-A3B MoE full-BAR/P2P-on portable performance on `.30`: TP8
  strict backend TPS `94.907`, `c1_2000` backend TPS `97.028`, `c1_10000`
  backend TPS `91.290`; note `strict gate valid`. TP4 release-time fixed-token
  performance is `c1_2000` backend TPS `116.146` and `c1_10000` backend TPS
  `109.283`. A post-v0.2 repeatability study found that the earlier TP4 strict
  runaway did not reproduce: `6/6` strict repeats passed across `.20` and `.30`,
  all with `finish_reason=stop` and `qwen_gate_valid=true`, with strict backend
  TPS from `113.196` to `115.995`. No code, Docker image, tag, model package, or
  runtime artifact changed.
- Platform remediation for the full-BAR/P2P-on lane required official AMD VBIOS
  standardization, not modified BIOS images, plus amdgpu source patching. The
  source inventory should keep this hardware/software precondition separate
  from model-performance claims. This is not a user instruction to flash cards;
  the public repo does not redistribute BIOS binaries or imply warranty or
  certification.
- Earlier dense source rows below are preserved as superseded historical
  context for why RowParallel, sidecar Tree/LL, and RCCL/NCCL boundary work
  mattered. They no longer describe the current dense 27B gate status.

## MoE Expert Kernels

| Source family | Files / overlays / kernels | What was adapted for gfx906 | Disposition |
| --- | --- | --- | --- |
| Qwen3.6 C1 topk8 MoE fastpath | `overlays/c1_topk8_moe_fastpath_20260526/fused_moe.py`; `_qwen_c1_topk8_w1_act_kernel.kd`; `_qwen_c1_topk8_w2_reduce_kernel.kd` | Hard-coded the active C1 topk8 unquantized path to real tensor strides, `I=w2.size(2)`, TP4 expert shard shape, and the tuned W1/W2 scheduling envelope. | Promoted for Qwen3.6-35B-A3B TP4 single-request decode. |
| Qwen3.6 TP8 semantic MoE TP4x2-in-TP8 | `overlays/qwen36_moe_tp4x2_in_tp8_20260526/qwen3_next.py` | Kept dense layers at TP8 while routing MoE math through two TP4 local groups; included shared-expert TP4x2 loader fixes and fused shared/routed local-sum reduce. | User-promoted semantic TP8 branch, but not a strict 90 TPS exact-hash closure. |
| Qwen3-Next topk10 MoE fastpath | `overlays/qwen3_next_c1_topk10_moe_fastpath_*` | Extended the C1 MoE fastpath concept to Qwen3-Next topk10 routing, EP maps, remote-skip/block-tune probes, and M-size variants. | Source canaries useful; serving gate not cleared. |
| MTP-aware MoE fastpath | `overlays/c1_topk8_moe_fastpath_mtp_m4_*` | Extended fastpath routing for MTP-shaped inputs and BF16 canaries. | Source evidence only; serving correctness or performance did not promote. |
| MoE prefill prefix kernel | `overlays/prefix_prefill_blockm64_20260605/prefix_prefill.py` | Changed non-CUDA Triton prefix prefill `BLOCK_M` from `128` to `64` to reduce gfx906 pressure. | MoE prefill/concurrency milestone; not a decode promotion. |

## Dense GEMV / LLGemm / SwiGLU Kernels

| Source family | Files / overlays / kernels | What was adapted for gfx906 | Disposition |
| --- | --- | --- | --- |
| LLMM1 helper routing | `utils_llmm1_rpb2`, LLMM1 shape census, exact-shape microbenches | Verified `ops.LLMM1(..., rows_per_block=2)` is the best current route for dominant `n=1` dense decode shapes such as `m=31040,k=5120` and RowParallel shards. | Keep `LLMM1_rpb2`; routing changes not promoted. |
| LLMM1 dot2 replacement | `patches/gfx906_llmm1_dot2_20260607.patch` | Replaced unsupported gfx9 half-dot path with gfx906-compatible `v_dot2_f32_f16` style operands. | Correct/compiled, but slower on dominant shape. |
| LLMM1 loop variants | `patches/gfx906_llmm1_loop_variants_20260607.patch` | Tested strided-K loop thread counts and reduction layouts. | Rejected; dominant shape regressed. |
| LLMM1 scalar / rows-per-block variants | `gfx906_llmm1_scalar_rpb5_20260607.patch`, rpb sweeps | Tested nonstandard row scheduling and scalar output-store ideas. | Rejected; `rpb2` remains best. |
| LLMM1 packed-FP16 reduction | `runs/llmm1_half_reduce_host30_20260607_203057` | Kept partial sums packed in `half2` through reduction. | Rejected; slower on gate/up dominant shapes. |
| Exact-K LLMM1 specialization | `llmm1_k5120_specialized_*` lab builds | Specialized K=5120 and shape-selective dispatch paths. | Rejected or ABI-blocked on ROCm 7.2; not a dense promotion. |
| wvSplitK | `patches/gfx906_enable_wvsplitk_env_20260607.patch` | Fixed gfx906 assembly incompatibility by replacing `v_dot2c_f32_f16` use with gfx906-safe dot operands; enabled guarded wvSplitK probes. | Concurrency/prefill milestone for `n=2..4`; rejected for c1 decode and unsafe on corrected dense shape stack. |
| Standalone fused SwiGLU GEMV | `microbenches/gfx906_swiglu_gemv_bench_20260607.py` | Fused gate/up GEMV and activation locally. | Rejected as standalone; did not beat tuned LLMM1 route. |
| Interleaved SwiGLU native runtime | `overlays/qwen2_moe_interleaved_swiglu_20260608/qwen2_moe.py`; native op lab | Moved N==1/N>1 branching into the native runtime path and used interleaved gate/up handling. | Dense compute milestone; now part of the dense high-water stack, but not enough alone. |
| Corrected fused gate/up lower bound | corrected D=2176,K=5120 MEP | Proved interleaved LLMM1-style fused epilogue saves roughly `3.2-3.5 us/layer`. | Source lower-bound milestone; serving requires packed/interleaved weights or a layout conversion plan. |

## Attention / Context / Prefill Kernels

| Source family | Files / overlays / kernels | What was adapted for gfx906 | Disposition |
| --- | --- | --- | --- |
| ROCm 7.2-style attention backport | `overlays/rocm72_attention_backport_20260605/triton_attn.py`, `triton_unified_attention.py` | Backported the 7.2-style attention files into the 6.3 MoE/dense package for controlled A/B. | MoE production not promoted; dense early milestone superseded by later stack. |
| Attention softmax segment count | `overlays/attention_softmax_segments_20260607/seg32`, `seg128` | Tested TRITON attention segment counts against gfx906 decode behavior. | Rejected for dense decode. |
| Attention decode tile size | `overlays/attention_decode_tile_20260607/tile4`, `tile16`, `tile32` | Tested decode tile variants around incumbent tile8. | Rejected for dense decode. |
| TRITON decode context parallelism | `overlays/triton_dcp_lse_work_20260607/*` | Added decode LSE output, DCP local context lengths, non-causal context attention, Flash-style LSE merge, and a TRITON suffix kernel. | Correct enough to serve, but far below incumbent decode TPS; source milestone only. |
| FlashAttention ROCm shims | FA compatibility overlays and `flashinfer_comm_rocm_shim` | Attempted to enable ROCm FlashAttention/DCP lanes despite API/signature drift. | Rejected because current AMD Triton FA2 path lacks paged attention support. |
| V2 hybrid/Mamba cache reshape | `overlays/attn_utils_v2_hybrid_mamba_20260607/attn_utils.py` | Fixed V2 startup by unwrapping cache specs and handling Mamba state layout. | V2 source milestone; not a dense decode promotion. |

## RowParallel / Graph Runtime / Consumer Fusion

| Source family | Files / overlays / kernels | What was adapted for gfx906 | Disposition |
| --- | --- | --- | --- |
| Mutable RowParallel allreduce | `overlays/rowparallel_mutable_ar_20260607/*` | Made RowParallel reductions mutation-aware and graph-stable so lower-level collective experiments could be attached. | Part of current dense high-water envelope. |
| Post-allreduce fused Gemma/Qwen RMS | `patches/gfx906_gemma_fused_add_rms_norm_20260606.patch`; `gfx906_gemma_add_rms_norm_fp32_residual_kernel` | Fused residual add and RMSNorm after allreduce for Qwen/Gemma-style graphs. | Correct lower-bound milestone; serving consumer-only fusion did not clear gate. |
| Post-IR direct MLP AR/RMS fusion | `patches/gfx906_post_ir_direct_mlp_ar_rms_fusion_hostimage_20260606.patch`; `overlays/postir_mutable_fusion_20260608/*` | Matched lowered MLP-down allreduce-consumer chains and replaced decomposed residual/RMS nodes. | Semantically viable at 64/64 MLP and 128/128 MLP+attention coverage; performance stayed below winner. |
| Split-endpoint RMS Python path | `overlays/postir_split_endpoint_rms_20260611/gfx906_post_ir_ar_rms_fusion.py` | Split allreduce endpoint and RMS work with process-local scratch to avoid graph allocation. | Source milestone; Python/ctypes overhead lost the component win. |
| Split-endpoint RMS native torch op | `gfx906_ext::split_endpoint_rms` lab build | Registered C++/HIP torch op for split endpoint RMS and inserted through post-IR. | Correct, but still not a serving promotion. |
| MLP-down residual pre-fold + split4 | `microbenches/rccl_llmm1_ar_rms_boundary_bench_20260612.cpp`; `llmm1_residual_rpb2_kernel`; RMS-only split endpoint; `overlays/qwen2_moe_interleaved_swiglu_20260608/qwen2_moe.py` | Tested `allreduce(LLMM1 + residual/TP) -> RMS-only` against normal `allreduce(LLMM1) + residual -> RMS` under direct graph replay, then composed the native residual prefold with the non-resident sidecar allreduce serving stack. | Exact on `.20/.30`, but not additive with split4 or the sidecar winner. Corrected active serving smokes logged `gfx906 MLP down LLMM1 residual pre-fold enabled` and reached `.20` `58.171` / `.30` `55.601` backend TPS, only noise-level over same-session controls. Full `.20` ladder was `57.930` strict smoke, `58.811` c1_2000, `55.789` c1_10000 backend TPS. Rejected as a serving promotion. |
| Fused LLMM1 producer plus sidecar allreduce | `microbenches/rccl_llmm1_ar_rms_boundary_bench_20260612.cpp`; `fused_llmm1_sidecar_tree_ll_1x5120_kernel` | Combined LLMM1 producer blocks and the three-channel `nWarps=4` sidecar Tree/LL allreduce into one graph node with monotonic producer-ready counters, then tested the `BOUNDARY_FUSED_CHANNEL_READY=1` variant where each sidecar channel waits only for its own producer row slice. The corrected lower-bound separates producer logical threads from the larger sidecar block, so attention `K=768` no longer overuses the sidecar warp count for producer reduction. Later parameterized the bench with `SIDECAR_COUNT_*` and descriptor-aware channel-ready row mapping so the promoted `2048/1536/1536` descriptor could be checked directly. | Exact, but rejected. Full-producer readiness regressed: `.20` sidecar `attn_5120x768`/`mlp_down_5120x2176` `0.053225/0.069524 ms` became fused `0.066186/0.095438 ms`; `.30` reproduced `0.053273/0.069546 -> 0.066458/0.095453 ms`. Channel-ready also rejects: `.20` `0.052666/0.069239 -> 0.065661/0.093162`, and `.30` MLP-down `0.069257 -> 0.093228` while attention remained slower than same-run sidecar AR. The repaired producer-thread version still rejects: `.20` attention/MLP-down `0.052392/0.068851 -> 0.066298/0.093176 ms`, and `.30` `0.052528/0.068663 -> 0.066501/0.093752 ms`. The current-descriptor revisit also rejects: `.20` `0.051948/0.068380 -> 0.064869/0.091892`, `.30` `0.052071/0.069377 -> 0.095290/0.092117`. Do not pursue coarse in-kernel spin-barrier producer/collective fusion; future fusion would need true chunk/LL-line overlap or captured launch-count removal. |
| Fused LLMM1 producer plus sidecar block order | `.tmp/llmm1_sidecar_fused_sidecarlast_20260614/`; `microbenches/rccl_llmm1_ar_rms_boundary_bench_20260612.cpp` | Added guarded `BOUNDARY_FUSED_SIDECAR_LAST=1`, moving producer CTAs before sidecar channel blocks inside the fused `LLMM1 -> sidecar Tree/LL allreduce` lower-bound kernel. | Exact but rejected before serving. `.20` was only a near-tie on attention (`0.052874 -> 0.052863 ms`) and a slight loss on MLP/down (`0.079530 -> 0.079605 ms`); `.30` corrected one anomalous same-source control outlier but remained in the normal fused band. The fused path is still slower than separate `LLMM1 + sidecar allreduce`, so simple fused-grid block ordering is closed. |
| RCCL post-RMS fusion | `rccl_721_gfx906_tree_ll_post_rms_*` patches | Tried to fuse post-allreduce RMS into RCCL Tree/LL boundary. | Rejected; one-channel and phase/rendezvous costs were too high. |
| Down-projection residual pre-folding | `qwen3_5_*residual_prefold*`, `qwen35_gdn_flatten_downprefold_*` | Tried to fold local residual/add work around down-projection and GDN boundaries. | Rejected; did not survive full serving ladder. |
| Greedy logits gather cleanup | `greedy_top1_mintokens_*` overlays | Replaced full-vocab logits all-gather with local top1 plus tiny gather for greedy decode semantics. | Correct serving milestone; not enough and did not stack with current winner. |
| Grouped allreduce lower bound | direct graph grouped allreduce harnesses | Proved grouping independent reductions can cut `1x5120` allreduce cost sharply. | Not integrable in current dense graph because dependencies/intervening uses block grouping. |
| Selective skip upper bounds | `rowparallel_selective_skip_20260609`, `skip_*allreduce*` | Removed semantic allreduces by category to locate the gate-scale boundary. | Invalid by design, but decisive evidence that MLP/down RowParallel is the primary dense target. |

## RCCL / Collective Kernels

| Source family | Files / overlays / kernels | What was adapted for gfx906 | Disposition |
| --- | --- | --- | --- |
| Specialized Tree/LL allreduce dispatch | `rccl_721_gfx906_tree_ll_specialized_kernel_include_ar_*` | Forced gfx906 fp16 Tree/LL allreduce into specialized include/dispatch paths and exported overlays for direct graph replay. | Base of the current dense source stack. |
| Explicit balanced `RCCL_TREES` | runtime tree string plus specialized RCCL overlay | Replaced RCCL's default one-child chain with an explicit binary tree matched to the TP8 hosts. | Superseded historical dense decode milestone. |
| Odd-root balanced `RCCL_TREES` rotations | `runs/rccl_sidecar_tree_ll_work_probe_host{20,30}_tree_rot{1,3,5,7}_h{20,30}_20260613` | Rotated the promoted balanced tree string to use odd roots while preserving the current `nWarps=4` / 256-thread sidecar descriptor. | Exact direct-replay diagnostic only. The best portable shifts are sub-microsecond and below serving-promotion scale; keep the promoted balanced tree string. |
| Tree/LL leaf-up/down scheduling | `rccl_overlay_721_gfx906_tree_ll_leaf_updown_20260611_*`; `rccl_overlay_721_gfx906_tree_ll_singlework_leaf_updown_20260612_h30`; sidecar SO `libgfx906_sidecar_tree_ll_ar_multirow_count_2048_1536_1536_leafupdown_20260614.so` | Revisited the leaf-side Tree/LL source variant after tree-role wall-clock attribution showed public1664 sidecar run-wall ownership is dominated by leaf roles, especially logical device `7` across active channels and device `6` as leaf/intermediate. | Exact but rejected for serving. The `.30` sidecar C API smoke was exact, and the runtime survived O3 graph capture, but strict c1 with public fallback `1664` reached only `61.448` backend TPS versus `.30` public1664 support `62.197`. The diagnostic is valuable because it closes root-only and existing leaf-up/down scheduling fixes; the remaining path needs a different sidecar primitive or RowParallel launch-count reduction. |
| Tree/LL force-inline and direct dispatch | `tree_ll_forceinline`, `direct_dispatch_forceinline`, `direct_dispatch_no_table`, `primsll_forceinline` | Reduced wrapper/table overhead around the hot Tree/LL path. | Some component wins; no standalone gate clear. |
| LL line count and step variants | `ll_lines4`, `ll_lines16`, `steps4`, `steps16`, `cleanmask` | Changed LL line granularity and step scheduling for tiny fp16 allreduces. | Mostly near-ties or rejects; LL16 composes inside current high-water side branch. |
| Singlework Tree/LL | `singlework_fastpath`, `singlework_steps4`, `singlework_ll_lines16`, `singlework_ll_lines16_primsll_inline`, `rccl_721_gfx906_tree_ll_singlework_nwarps4_after_20260612.patch` | Collapsed/reshaped work launching and primitive inlining for hot small allreduces; also tested carrying the sidecar `nWarps=4` occupancy into the public RCCL planner by forcing gfx906 fp16 sum Tree/LL allreduce work records to `4 * comm->WarpSize`. | Best `.20` source high-water side branch remains singlework/LL16/prims-inline with balanced trees. The real-path nWarps4 planner override is exact but rejected: `.20` tied the high-water row family and `.30` regressed `4x5120`. |
| RCCL devComm sidecar Tree/LL work-loop | `microbenches/rccl_devcomm_channel_probe_20260612.cpp`; `microbenches/rccl_sidecar_tree_ll_work_probe_20260612.cpp`; `microbenches/rccl_llmm1_ar_rms_boundary_bench_20260612.cpp` | Read live `ncclDevCommAndChannels`, derived the three-channel `1x5120` descriptor, executed exact Tree/LL fp16 sum through `RunWorkColl`, then tested descriptor occupancy, batching/serial variants, a direct `LLMM1 -> sidecar allreduce -> RMS` boundary, and inline sidecar allreduce+RMS endpoint work. | New one-work promotion: `work->nWarps=4` plus a 256-thread launch is exact and beats balanced stock on the `1x5120` microbench (`.20` `0.029549` vs stock `0.030739`; `.30` count-1 batch `0.029751` vs stock `0.030717`). Manual `nWarps=4` coll `RunWorkBatch >1` is rejected due timeout and is not planner-legal because RCCL caps coll batch bytes at one descriptor. Serial reset mode is promoted as a source microbench milestone: count `128` is exact at `.20` `0.027902` and `.30` `0.027888 ms/work` versus stock `~0.03125 ms/call`. Reusable descriptor serial mode is the latest source milestone: count `128` stayed exact at `.20` `0.027699` and `.30` `0.027609 ms/work`, proving descriptor setup can be amortized safely in an explicitly serial work sequence. Boundary integration is exact but too small for serving promotion: `.20/.30` `LLMM1+AR` and split-boundary savings are only about `0.8-1.3 us` per call. Inline sidecar RMS is also exact, but adds only about `0.1 us` over sidecar+split; consumer-only sidecar fusion is rejected as the primary gate path. |
| Four-warp batched sidecar barrier reset | `.tmp/sidecar_batch_reset_20260614/rccl_sidecar_tree_ll_work_probe_batch_reset_20260614.cpp` | Added a throwaway `SIDECAR_MULTIWORK_MODE=batch_reset` harness path that fills up to eight `ncclDevWorkColl` descriptors once, then executes them with `nWarps=4` / 256 threads while resetting `ncclShmem.barrier_pat` and `ncclShmem.groups[*].barrier` before each work item. | Promoted source diagnostic, not serving. Raw `RunWorkBatch` with `nWarps=4` timed out at count `2` on both `.20` and `.30`; `batch_reset` made counts `2/4/8` exact on both hosts, with count `8` graph time `.20` `0.028334` and `.30` `0.028365 ms/work`. Same-run serial reuse still wins slightly (`.20` `0.028216`, `.30` `0.028206`), and serial-reuse serving was already below high-water. Use this as a construction rule for any future grouped sidecar: dependent work items need explicit LL barrier/group reset unless the RCCL planner path proves equivalent state cleanup. |
| Single-work full group-barrier reset | `.tmp/sidecar_full_group_reset_20260614/gfx906_sidecar_tree_ll_ar_runtime_full_group_reset_20260614.cpp` | Tested whether the hot single-work sidecar should clear every `ncclShmem.groups[*].barrier` at initialization instead of only `groups[0]`, based on the multiwork barrier-reset finding. | Exact but rejected before serving. Matched C API smokes with the promoted `2048/1536/1536` descriptor regressed `.20` from `0.056210` to `0.058163 ms/call` and `.30` from `0.056647` to `0.056749`. The all-group reset rule applies between dependent work items, not to ordinary single-work initialization. |
| Sidecar descriptor split tuning | `microbenches/rccl_sidecar_tree_ll_work_probe_20260612.cpp` | Added compile-time `SIDECAR_COUNT_LO/MID/HI` and `SIDECAR_CHUNK_GRAINS_LO/MID/HI` macros to sweep the fixed `1x5120` Tree/LL descriptor without changing serving code. | Exact but rejected as a gate-scale lever. Corrected sequential per-host sweeps showed the best count split `2048/2048/1024` only moved graph-mode sidecar from `.20` `0.043890` to `0.043853 ms/call` and `.30` `0.043945` to `0.043868 ms/call`; grain variants were neutral or regressed. Keep the macros for diagnostics, but do not build a serving runtime from this tiny component near-tie. |
| RCCL persistent Tree/LL sidecar worker | `microbenches/rccl_persistent_tree_ll_worker_probe_20260612.cpp`; `microbenches/gfx906_persistent_tree_ll_ar_runtime_20260612.cpp`; `microbenches/gfx906_persistent_ar_capi_smoke_20260612.cpp`; `overlays/gfx906_persistent_ar_20260612/*` | Launched a resident three-channel worker that keeps RCCL channel state alive and executes serial dependent `1x5120` commands submitted by a graph-captured control stream; later exposed `PERSISTENT_WORKER_NWARPS`, block threads, pointer-submit, a dlopened C ABI runtime, a guarded vLLM serving overlay, an opt-in idle `s_sleep` loop via `GFX906_PERSISTENT_AR_IDLE_SLEEP_ITERS`, a deferred post-capture worker lifecycle, snapshot/watchdog counters for command/status progress, an execute-model hook that can arm workers only after the first real prefill, and an opt-in `VLLM_GFX906_PERSISTENT_AR_START_ON_FIRST_USE` diagnostic for deferred handles created after the hook fires. | Promoted as a source milestone after carrying over the sidecar `nWarps=4` occupancy. `nWarps=2` was rejected at `.20` `0.039367 ms/call`, but `nWarps=4`/256 threads is exact and reproduced at `.20` `0.028754` and `.30` `0.028727 ms/call`, beating the public `1x5120` graph comparator. The dlopened C ABI is exact. Per-call start/stop is rejected at `~3.60 ms/call`. First-create, preinit, skip-profile, and yielding `idle_sleep_iters=4` serving smokes all hung in/after O3 graph capture. Deferred post-capture launch survives startup and opens the API, but starting all 8 idle workers immediately after capture makes the first strict c1 request time out before any persistent command is submitted (`request=0/done=0`). The start-after-prefill lifecycle fixes that timeout and completes strict valid smokes while the watchdog shows command progress: `.20 nWarps=4` `58.272`, `.20 nWarps=2` `57.023`, and `.30 nWarps=3` `54.258` backend TPS. The later `.20` idle-sleep16 retries completed valid c1 at `57.170` and `57.249` TPS; logs showed the post-prefill hook started `0` workers and the first-use opt-in did not produce replacement logs. These remain below the non-resident high-water. Keep the lifecycle and diagnostics; reject further simple resident lifecycle sweeps until source tracing proves the in-place persistent replacement path is entered under the real decode graph. |
| Non-resident sidecar Tree/LL allreduce runtime | `microbenches/gfx906_sidecar_tree_ll_ar_runtime_20260612.cpp`; `run_gfx906_sidecar_ar_runtime_build_20260612.sh`; `overlays/gfx906_persistent_ar_20260612/*`; `overlays/rowparallel_mutable_ar_20260607/*` | Reused the exact `nWarps=4`/256-thread Tree/LL sidecar descriptor without a permanent resident wait loop, exposed it through a dlopened runtime, and routed fixed-shape `numel=5120` RowParallel reductions through a graph-capture-safe vLLM custom op. Added target-selective RowParallel routing for `all`, `mlp`, `attn`, or `none` while preserving a public PyNccl fallback custom op for non-target callsites. Corrected attention target detection to include `input_size_per_partition=768`. Tested throwaway descriptor partition variants for `countLo/Mid/Hi` and `chunkGrainsLo/Mid/Hi`, but restored the default runtime source to the promoted constant `1024/2048/2048` count split and `2048/2048/2048` grains. | Superseded historical dense source high-water and serving milestone: full all-sidecar ladders reached `.20` `61.959/62.957/59.431` and `.30` `61.983/62.777/59.054` backend TPS. Corrected target-selective full ladders reject selective routing: `.20` attention-only `57.793/58.527/55.468`, `.30` `mlp,attn` `55.509/56.187/53.417`, `.20` `mlp`-only `57.420/58.130/55.138`, `.30` `mlp`-only `55.238/55.824/52.996`, and `.10` `other`-only `53.825/52.562/48.853` backend TPS for strict c1 / c1_2000 / c1_10000. Worker-shape tuning is closed for this runtime: `4/256` remains best, `3/192` is slower, and `4/320`, `5/320`, `6/384`, `8/512` launch-fail. Channel partition tuning is exact but rejected as non-portable: `.20` default C API smoke was `0.07249 ms/call`, while tested partitions regressed to `0.0865-0.0893`; `.30` improved from `0.08481` to `0.07743`, which is host-specific and not enough for promotion. Channel-count expansion is also closed: static four-channel `1024/1024/1024/2048` replay was exact but slower than the original three-channel primitive on both hosts (`.20` `0.030045` vs `0.029670` ms/call; `.30` `0.029983` vs `0.029743`). Prebuilt work templates are rejected: `.20` improved only `0.044033 -> 0.043997 ms/call` while `.30` regressed `0.043986 -> 0.044024-0.044049`. |
| Tree/LL microchunking | `.tmp/sidecar_tree_ll_microchunk_20260614/all_reduce.h`; `run_gfx906_sidecar_ar_runtime_build_20260612.sh` `EXTRA_LOCAL_FILES`; builds `tree_ll_micro{0,512,1024}_h{20,30}_20260614` | Added a throwaway `GFX906_TREE_LL_MICROCHUNK` knob to the promoted RCCL `all_reduce.h` so the hot sidecar `runTreeSplit` path can force smaller LL chunks while preserving the current `2048/1536/1536` descriptor and balanced Tree/LL envelope. | Exact but rejected before serving. Fresh `micro0` controls were `.20` `0.062040` and `.30` `0.059337 ms/call`. `micro1024` gave `.20` a small component win (`0.061247`) but regressed `.30` (`0.065649`); `micro512` regressed both hosts (`0.067117`/`0.067623`). This closes coarse `runTreeSplit` chunk-loop retuning; the next sidecar primitive work must touch the LL first-read datapath or remove/reduce RowParallel collective boundaries. |
| LL receive polling/backoff | `.tmp/sidecar_ll_pollbackoff_20260614/prims_ll.h`; staged `all_reduce.h`/`primitives.h`; builds `llpoll_sleep{0,256}` and `llpoll_bcast{0,256}` on `.20/.30` | Added guarded `GFX906_LL_READ_SLEEP_SPIN_INTERVAL`, `GFX906_LL_READ_SLEEP_VALUE`, and `GFX906_LL_READ_SLEEP_BROADCAST_ONLY` controls to test whether light `s_sleep` backoff in `readLL`/`readLLFinish` improves the first receive/read wait tail. | Exact but rejected before serving. Blanket `sleep256` regressed `.30` (`0.060595 -> 0.063853 ms/call`) and did not improve `.20`. Broadcast-only `bcast256` kept reduce-up polling unchanged but still lost to 512-iteration controls on both hosts (`.20` `0.060173 -> 0.060460`, `.30` `0.057978 -> 0.058274`). This closes simple local spin/yield policy as the gate lever. |
| LL FIFO 128-bit receive load | `.tmp/sidecar_ll_b128load_20260614/prims_ll.h` | Added guarded `GFX906_LL_FORCE_DWORDX4_LOAD=1` so the gfx9 LL receive path loads each 16-byte FIFO line with one `global_load_dwordx4 ... glc slc` instead of two non-temporal `u64` loads. | Exact but rejected before serving. The candidate compiled on gfx906 but regressed both lanes under the promoted `2048/1536/1536` descriptor: `.20` `0.056449 -> 0.061807 ms/call`, `.30` `0.067920 -> 0.077688`. Keep the current two-`u64` receive path. |
| LL FIFO `u64` receive load order | `.tmp/sidecar_ll_readorder_20260614/prims_ll.h` | Added guarded `GFX906_LL_REVERSE_U64_LOAD_ORDER=1`, changing only the gfx9 two-`u64` receive order in `readLL`, `readLLBeginAll`, and `readLLFinish` so the high word is loaded before the low word. | Exact but rejected before serving. After discarding an invalid simultaneous control/candidate run, clean sequential C API repeats with the promoted `2048/1536/1536` descriptor showed `.20` regressed from `0.056300` to `0.060437 ms/call`; `.30` moved only within noise from `0.056179` to `0.055990`. Keep the normal low-then-high load order. |
| Sidecar Tree/LL128 protocol substitution | `.tmp/sidecar_tree_ll128_20260614/gfx906_sidecar_tree_ll_ar_runtime_ll128_20260614.cpp` | Changed only the sidecar `RunWorkColl` protocol template from `NCCL_PROTO_LL` to `NCCL_PROTO_LL128` while preserving the promoted `2048/1536/1536` descriptor, `nWarps=4`, and 256-thread launch. | Exact but rejected before serving. Matched C API controls were `.20` `0.073089` and `.30` `0.072796 ms/call`; LL128 was much slower at `.20` `0.104334` and `.30` `0.105641`. This closes the sidecar protocol-swap shortcut for the hot `1x5120` path. |
| LL broadcast send-slot order | `.tmp/sidecar_ll_sendorder_20260614/prims_ll.h`; staged `all_reduce.h`/`primitives.h`; builds `llsend_slot0{last,first}` on `.20/.30` | Added guarded `GFX906_LL_SEND_SLOT0_FIRST` to test whether writing Tree/LL broadcast send slot `0` before slots `1..N` improves first-line propagation. | Exact but rejected before serving. `slot0_first` regressed `.20` (`0.057677 -> 0.058452 ms/call`) and only moved `.30` within noise (`0.057279 -> 0.057553`). This closes simple send-slot ordering as the gate lever. |
| Broadcast-only LL send-slot order | `.tmp/sidecar_ll_bcast_sendorder_20260614/prims_ll.h` | Added guarded `GFX906_LL_BCAST_SEND_SLOT0_FIRST`, applying slot `0` first only to the broadcast `recvCopySend` LL primitive instance while leaving reduce-up/root behavior unchanged. | Exact but rejected before serving. Fresh controls initially made the branch look useful (`.20` `0.060769 -> 0.057145`, `.30` `0.061277 -> 0.057074 ms/call`), but reverse controls moved into the same or better warmed band (`.20` `0.056413`, `.30` `0.057005`). Small sidecar primitive wins require reverse-order C API repeats before promotion. |
| Specialized LL broadcast `recvCopySend` | `.tmp/sidecar_ll_bcast_specialized_20260614/prims_ll.h` | Added guarded `GFX906_LL_BCAST_SPECIALIZED=1`, keeping wait/post/step semantics while replacing the generic broadcast `LLGenericOp<1,1,-1,Output>` wrapper with a narrower hot-path receive-copy-send implementation. | Exact but rejected before serving. `.20` control/candidate/reverse-control were `0.072962/0.074048/0.071260 ms/call`, and `.30` `0.072644/0.071191/0.070815`; the candidate did not beat warmed controls. Wrapper specialization around `recvCopySend` is not enough to fix first-line propagation. |
| Tree-path-targeted LL broadcast send order | `.tmp/sidecar_ll_treepath_sendorder_20260614/prims_ll.h`; builds `lltreepath_{control,candidate}_h{20,30}_20260614_1439` | Added guarded `GFX906_LL_BCAST_TREEPATH_FIRST`, applying slot0-first only to the TP8 channel/rank pairs where the promoted balanced tree sends the slow leaf path through child slot 0. | Exact but rejected before serving. Longer 4096-iteration C API runs were mixed: `.20` candidate/control `0.057114/0.057869 ms/call`, but `.30` candidate/control `0.056916/0.056371`. Blanket, broadcast-only, and tree-path-targeted send-order tweaks are now closed as portable dense-gate levers. |
| Broadcast `waitSend` no-poll | `.tmp/sidecar_ll_bcast_waitsend_20260614/prims_ll.h`; builds `llbcast_waitsend_{control,assume}_h{20,30}_20260614_1446` | Added guarded `GFX906_LL_BCAST_ASSUME_SEND_SPACE` so LL broadcast `recv-copy-send` can publish FIFO size and advance the send head without polling the remote head first. | Exact but rejected before serving. `.20` showed a small local C API win (`0.057044 -> 0.056386 ms/call`), but `.30` regressed materially (`0.056717 -> 0.063378`). Do not bypass queue-space polling in the serving path; any future change must preserve equivalent queue semantics while improving propagation. |
| Broadcast LL FIFO scalar store order | `.tmp/sidecar_ll_bcast_storeorder_20260614/prims_ll.h`; builds `llbcast_storeorder_{control,rev}_h{20,30}_20260614_1500` | Added guarded `GFX906_LL_BCAST_STORE_REVERSE_WORDS`, preserving two scalar `u64` FIFO stores but reversing their order only for broadcast `recv-copy-send`. | Exact but rejected before serving. Longer 4096-iteration C API runs were mixed: `.20` regressed (`0.056507 -> 0.056790 ms/call`) while `.30` improved (`0.057086 -> 0.056511`). Combined with the older rejected vector-store patch, this closes LL FIFO store publication order as a robust gate-scale lever. |
| Rank-selective sidecar channel order | `.tmp/sidecar_rank_selective_channel_order_20260614/gfx906_sidecar_tree_ll_ar_runtime_rank_selective_channel_order_20260614.cpp`; builds `rankselect67_021_h{20,30}_20260614` and `rankselect7_021_h{20,30}_20260614` | Added guarded `SIDECAR_RANK_SELECTIVE_*` compile-time controls so only the logical ranks implicated by the public1664 wall-clock trace use sidecar channel block order `0,2,1` while all other ranks preserve promoted order `0,1,2`. Tested rank range `6-7` and a narrower rank-7-only follow-up. | Exact but rejected before serving. Initial rank `6-7` 512-iteration C API looked neutral/winning (`.20` `0.057084 -> 0.057039`, `.30` noisy `0.064731 -> 0.060549`), but the longer reversed-order 1024-iteration repeat lost on both hosts: `.20` candidate/control `0.056924/0.056868`, `.30` `0.056538/0.056085`. Rank-7-only also rejects: `.20` `0.057329` only beat one noisy slow control and is slower than established `.20` controls, while `.30` regressed versus same-session control (`0.057933/0.056634`). This closes tail-rank-only channel reorder as a stable primitive improvement. |
| Multi-row non-resident sidecar capture path | `.tmp/sidecar_multirow_runtime_20260613/gfx906_sidecar_tree_ll_ar_runtime_multirow_20260613.cpp`; `.tmp/sidecar_multirow_runtime_20260613/gfx906_persistent_ar.py`; `.tmp/sidecar_splitloop_multirow_runtime_20260613/gfx906_sidecar_tree_ll_ar_runtime_multirow_20260613.cpp`; `.tmp/sidecar_serial_multirow_runtime_20260613/gfx906_sidecar_tree_ll_ar_runtime_multirow_20260613.cpp`; `.tmp/sidecar_serial_reuse_runtime_20260613/gfx906_sidecar_tree_ll_ar_runtime_20260612.cpp`; `microbenches/gfx906_persistent_ar_capi_smoke_20260612.cpp`; `analyze_sidecar_tail_20260613.py`; `analyze_sidecar_position_shape_20260613.py`; `analyze_sidecar_pair_gpu_skew_20260613.py` | Left the hot `numel=5120` path on the promoted `nWarps=4`/256-thread sidecar kernel, added an opt-in eligibility/runtime path for `2/4/8 * 5120` rows, then added C API count control, an in-place probe mode, a split-loop fallback that launches the exact `1x5120` kernel once per slice, serial-reset and serial-reuse multi-row kernels, and optional serving env forwarding for serial multiwork `nWarps`/block settings. Added tail analysis that filters malformed rocprof rows and avoids cross-GPU skew claims because rocprof GPU clocks are not synchronized, local-sequence previous/next-kernel shape analysis for sidecar p99 rows, and pair/GPU skew analysis for local producer duration versus sidecar tail. | Superseded historical dense source milestone and pre-2026-06-20 standard-envelope high-water. Full ladders for the original widened-capture candidate reached `.20` `62.491/63.308/59.783` and `.30` `62.192/62.919/59.127` backend TPS for strict c1 / c1_2000 / c1_10000. Saved logs confirm graph-capture `numel=40960` replacements (`.20` `486`, `.30` `484`). `NCCL_NTHREADS` unset does not promote with this path: `.20` `62.218/63.237/59.704`, `.30` `62.031/62.948/59.237`. The graph-capture skip-context sweep is closed: skip `0` on `.20` produced a valid c1 smoke at `62.285` backend TPS and skip `2` on `.30` produced `60.973`, both below same-host high-water, so keep skip `1`. Follow-up `.30` profile `qwen36_27b_sidecar_multirow_winner_rocprof_h30_20260613` was valid (`806,243` filtered rows, `25.723s`), but still showed only `sidecar_tree_ll_1x5120_kernel` as the dominant sidecar row (`14.128s`, `98,073` launches) and no `sidecar_tree_ll_multiwork_*` rows. Tail analysis puts the all-GPU `sidecar_tree_ll_1x5120_kernel` p50/p95/p99 at `.20` winner `42/183/3051 us`, `.30` original `46/172/3575 us`, and `.30` multi-row `46/179/3644 us`, so multi-row capture has not fixed the hot tail. Position/shape analysis shows the tail is producer-shape concentrated: the two previous `LLGemm1_kernel` shapes `grid=327680 wg=128` and `grid=819200 wg=320` explain `.30` `961/981` and `.20` `962/981` sidecar p99 rows, while modulo-16 local sequence position is less decisive. Pair/GPU skew analysis sharpens this: attention `LLGemm1 grid=327680 -> interleaved RMS vgpr52` p99 rows almost never follow a slow local producer (`.30` `6/356`, `.20` `9/356`), and local launch gaps stay around `9-10 us`, so the worst tail is most likely inside sidecar collective wait/serialization or rank-arrival behavior. Raising `MAX_NUM_SEQS` from `4` to `8` changed FULL graph profiling from largest `4` to largest `8`, but did not promote: `.20` valid smoke reached `62.256` backend TPS versus `62.491`, and `.30` reached `61.796` versus `62.192`. Split-loop was exact on `.20/.30` for `5120/10240/20480/40960`, but serving rejected it at `.20` `62.015/62.997/59.498` and `.30` `62.294/63.041/59.489`. The earlier serial-reset serving run at `.20` `58.096/58.835/55.785` is superseded as methodology evidence because it mounted the base `5120` Python overlay and had empty `RCCL_TREES`. Corrected serial-reuse with the multi-row Python overlay, balanced tree, `nWarps=4`, and 256 threads is exact and materially faster in C API (`40960` `.20` `0.267425`, `.30` `0.271618 ms/call`) but still rejected for serving: `.20` `62.013/62.858/59.392` and `.30` `62.106/62.893/59.241`. Treat this family as widened capture plus mild long-decode gain until a real multi-row launch-count collapse or the sidecar collective wait/rank-arrival path is fixed. |
| Multi-row sidecar `2048/1536/1536` split | `microbenches/gfx906_sidecar_tree_ll_ar_runtime_20260612.cpp`; `.tmp/sidecar_multirow_runtime_20260613/gfx906_sidecar_tree_ll_ar_runtime_multirow_20260613.cpp`; build SOs `libgfx906_sidecar_tree_ll_ar_multirow_count_2048_1536_1536_20260613.so` on `.20/.30` | Added compile-time `SIDECAR_COUNT_LO/MID/HI` and `SIDECAR_CHUNK_GRAINS_LO/MID/HI` defaults to both base and multi-row runtimes, then used op-count trace evidence to retest a less channel-imbalanced fixed Tree/LL descriptor split. | Superseded `.20` serving milestone. Direct graph probes improved from `.20` default `0.029616` to `0.029084 ms/call` and `.30` default `0.029628` to `0.029076`, exact on both hosts. The full `.20` ladder improved the prior high-water from `62.491/63.308/59.783` to `62.806/63.807/60.209` backend TPS for strict c1 / c1_2000 / c1_10000. The `.30` strict c1 smoke was valid but below same-host high-water at `62.098`, so do not promote as a cross-host global gate solution. |
| Multi-row sidecar op-count trace instrumentation | `.tmp/sidecar_multirow_runtime_20260613/gfx906_sidecar_tree_ll_ar_runtime_multirow_20260613.cpp`; `analyze_sidecar_opcount_trace_20260613.py`; runs `qwen36_27b_decode_tiers_rocm72_dense_sidecar_multirow_trace_2048_1536_1536_h{20,30}_graceful_*_20260613_193643` | Extended the sidecar trace entries with `work_count` and dump support for the promoted multi-row runtime, then ran `.20/.30` capped strict diagnostics with graceful container stop so destructor-based trace CSV export completed. | Diagnostic milestone. Both hosts produced equal per-rank CSVs with `199,296` traced launches per device and no overflow. The `op_count >= 10000` steady decode cut had `178,693` complete `(channel, op_count, work_count)` groups and zero incomplete groups on both hosts, and every steady group was `work_count=1`. Multi-work rows were confined to early graph/warmup ranges (`work_count=8` at `646-1663`, `4` at `1671-3978`, `2` at `2184-3338`). This proves the current high-water is not a steady c1 launch-count collapse; future gate work must make the strict c1 graph call a fused/multi-work boundary or improve the hot `1x5120` primitive/tail. |
| Sidecar launch-bounds specialization | `.tmp/sidecar_multirow_runtime_20260613/gfx906_sidecar_tree_ll_ar_runtime_multirow_20260613.cpp`; build SOs `libgfx906_sidecar_tree_ll_ar_multirow_lb256_2048_1536_1536_20260613.so` and `libgfx906_sidecar_tree_ll_ar_multirow_lb256x2_2048_1536_1536_20260613.so` on `.20/.30` | Added guarded `SIDECAR_LAUNCH_BOUNDS_THREADS` and `SIDECAR_LAUNCH_BOUNDS_MIN_BLOCKS` macros to both sidecar kernels so compiler resource assumptions can be varied without changing the promoted descriptor split or serving overlay. | Exact but rejected for promotion. `256,1` was nearly neutral on `.20` C API (`0.073859` vs `0.074258 ms/call`) and regressed `.30` (`0.073374` vs `0.062261`). `256,2` gave a strong isolated `.20` C API win (`0.059460`) but still regressed `.30` (`0.072244`). Strict serving rejected the only plausible variant: `.20` `62.524` backend TPS versus high-water `62.806`, and `.30` `60.733` versus `62.192`. Keep the macros for future lab builds; do not promote compiler-resource tweaks without O3 serving reproduction. |
| Sidecar captured call-site tracing and routing | `.tmp/sidecar_multirow_runtime_20260613/gfx906_sidecar_tree_ll_ar_runtime_multirow_20260613.cpp`; `analyze_sidecar_opcount_trace_20260613.py`; build SOs `libgfx906_sidecar_tree_ll_ar_multirow_callsite_trace_2048_1536_1536_20260614.so` and `libgfx906_sidecar_tree_ll_ar_multirow_callsite_mod4alt1792_20260614.so` on `.20/.30` | Added a per-handle `launch_sequence` passed into sidecar kernels as `call_site`, emitted it in trace CSVs, extended the analyzer to rank captured call sites by global/device-local p99 rows, and added guarded compile-time routing macros (`SIDECAR_CALLSITE_BASE`, `SIDECAR_CALLSITE_ROUTE_MOD`, `SIDECAR_CALLSITE_ROUTE_REM`, `SIDECAR_ALT_COUNT_*`) so selected captured graph nodes can use alternate descriptor splits. | Diagnostic milestone; route rejected. Trace smokes were strict-valid and showed exactly `128` steady call sites in `1664-1791`. `.20` p99 rows were dominated by `(call_site - 1664) % 4 == 0`; `.30` was more mixed but still mod0-led. The targeted mod0 route kept default `2048/1536/1536` and switched only mod0 to `1792/1664/1664`, but full serving rejected it: `.20` `62.594/63.622/60.015` and `.30` `61.935/62.915/59.200`. Exact single-call-site variants also rejected: `.20` `1664` alt `62.792`, `.30` `1664` alt `62.031`, and `.30` host-local `1672` alt `62.132`. Keep call-site tracing/routing as infrastructure; descriptor-only p99 routing is not the gate fix. |
| Sidecar wall-clock trace instrumentation | `.tmp/sidecar_wallclock_trace_runtime_20260614/gfx906_sidecar_tree_ll_ar_runtime_wallclock_trace_20260614.cpp`; `analyze_sidecar_wallclock_trace_20260614.py`; runs `qwen36_27b_decode_tiers_rocm72_dense_sidecar_wallclock_public1664_capped_h20_host20_20260614_wallclock_public1664_capped_h20` and `qwen36_27b_decode_tiers_rocm72_dense_sidecar_wallclock_public1664_uncapped_flush_h20_host20_20260614_wallclock_public1664_uncapped_flush_h20` | Added disabled-by-default trace fields using the RCCL/ROCm wall-clock source: `entry_wall_clock`, `run_start_wall_clock`, `run_end_wall_clock`, `setup_wall_ticks`, and `run_wall_ticks`. The analyzer estimates median per-device offsets from complete cross-rank cohorts, joins call-site labels, and reports adjusted arrival spread, exit spread, run-wall spread, owner counts, channel summaries, per-channel owner counts, and top call sites. | Diagnostic/source milestone. The C ABI smoke was exact at about `0.0836 ms` scoped lifecycle time. The capped serving trace loaded `2,362,392` entries with `274,834` complete cohorts and pointed toward residual MLP sidecar nodes. MLP public-fallback follow-ups are rejected: `1664,1711` `.20/.30` `62.871/61.769`, `1664,1759` `62.777/61.918`, and `1664,1743` `62.877/61.795`; exact MLP routing of `1711,1759,1743` rejected on full `.20` ladder at `62.993/63.851/60.279`. The later uncapped strict-valid trace loaded `11,558,208` entries and `1,424,311` complete cohorts, with run-wall ownership dominated by devices `7/6` and channel `1` showing the highest entry-run p99. Channel block-order smokes rejected: `0,2,1` `.20/.30` `63.034/62.163`, and `1,0,2` `63.006/61.838`. Keep wall-clock tracing for attribution; do not promote MLP public fallback, exact MLP sidecar routing, or simple channel block-order changes. |
| Sidecar call-site public bypass | `.tmp/sidecar_kindroute_runtime_20260613/gfx906_persistent_ar.py`; `.tmp/sidecar_kindroute_runtime_20260613/parallel_state.py`; `run_qwen36_27b_sidecar_kindroute_decode_lane_20260613.sh` | Added guarded runtime controls that preserve the sidecar launch sequence but return `False` for selected captured call sites, letting the existing PyNccl/RCCL public allreduce handle that node. Controls include `VLLM_GFX906_PERSISTENT_AR_BYPASS_KIND`, `*_CALLSITE_LO/HI`, `*_CALLSITE_LIST`, and optional modulo routing. | Superseded narrow `.20` serving milestone. Broad even-attention public fallback was valid but slow (`.20` strict c1 `61.781`). Bypassing only the worst mapped attention node, `call_site=1664`, improved `.20` from `62.806/63.807/60.209` to `63.084/63.993/60.380`; `.30` reached `62.197/63.034/59.525`. A no-trace repeat supported but did not beat it (`.20` `63.070/63.883/60.296`). Sparse and host-local singleton follow-ups rejected broader public fallback: `.20` `1664,1668` smoke `62.909`, `.20` `1668` smoke `62.723`, `.20` `1692` full ladder `62.909/63.788/60.169`, `.20` `1664,1692` smoke `62.997`, `.30` top-4 smoke `61.948`, `.30` `1692` smoke `61.781`, `.30` host-local `1672` smoke `61.913`, `.30` `1664,1672` smoke `61.320`, and `.30` `1688` smoke `61.334`; hybrid public `1664` plus alt descriptor `1668` also lost at `.20` `62.604`. Keep `1664` as source evidence that per-node primitive selection is viable; do not route whole attention classes, host-local p99 singletons, or sparse top-N sets to public RCCL. |
| RowParallel prefix/callsite mapping | `.tmp/sidecar_kindroute_runtime_20260613/linear.py`; `rowprefix_callsite_map_dev0.csv`; `rowprefix_callsite_mapping_summary.md` | Added a static-construction `RowParallelLinear` prefix trace for O3-safe callsite mapping and guarded the old forward-time trace behind `VLLM_GFX906_ROWPART_PREFIX_TRACE_RUNTIME=1` after Dynamo rejected filesystem calls in compiled forward. | Diagnostic/source milestone. The `.20` mapping run was strict-valid at `63.016` backend TPS and mapped the captured `1x5120` ranges to two passes over the `128` language RowParallel sequence. Public fallback `1664` is exactly `language_model.model.layers.0.linear_attn.out_proj`; `1665/1666/1667` are layer-0 MLP, layer-1 linear-attn, and layer-1 MLP. Same-class early linear-attn fallback `1664,1666,1668` rejected on both hosts (`.20` `62.726`, `.30` `61.451`). Use this mapping for exact prefix/layer source rewrites; do not keep expanding public fallback lists. |
| MLP-only alternate sidecar kind-route | `.tmp/sidecar_kindroute_runtime_20260613/*`; `run_qwen36_27b_sidecar_kindroute_decode_lane_20260613.sh`; sidecar libraries `2048/1536/1536` and `1792/1664/1664` | Kept the promoted default sidecar for most calls, retained public fallback for `kind=attn, call_site=1664`, and routed only MLP/down kinded RowParallel calls through the direct-replay-faster `1792/1664/1664` sidecar library. | Exact but rejected before c1_10000. Strict c1 was valid but slower than same-host public1664: `.20` `62.597` versus `63.084`, `.30` `61.073` versus `62.197`. The c1_2000 tier also lost (`.20` `63.558` versus `63.993`, `.30` `62.839` versus `63.034`). This closes simple MLP-only alternate-descriptor routing; direct descriptor medians are not enough without tail/launch-count improvement. |
| Multi-row sidecar `1792/1664/1664` neighbor split | `microbenches/rccl_sidecar_tree_ll_work_probe_20260612.cpp`; `.tmp/sidecar_multirow_runtime_20260613/gfx906_sidecar_tree_ll_ar_runtime_multirow_20260613.cpp`; build SOs `libgfx906_sidecar_tree_ll_ar_multirow_count_1792_1664_1664_20260613.so` on `.20/.30` | Swept neighboring count-only splits around the `.20` high-water while keeping default chunk grains, then built the best direct replay candidate into the same multi-row serving runtime. | Exact but rejected for serving. Direct graph replay improved to `.20` `0.028766` and `.30` `0.028765 ms/call`, faster than `2048/1536/1536`, but full serving ladders regressed: `.20` `62.270/63.377/59.833` and `.30` `61.691/62.768/58.975` backend TPS for strict c1 / c1_2000 / c1_10000. Treat this as evidence that direct median descriptor timing is no longer enough; the gate needs p99/tail or launch-count reduction. |
| Sidecar op-count trace instrumentation | `microbenches/gfx906_sidecar_tree_ll_ar_runtime_20260612.cpp`; `analyze_sidecar_opcount_trace_20260613.py`; `runs/qwen36_27b_decode_tiers_rocm72_dense_sidecar_traceopcount_h20_host20_traceopcount_h20_20260613` | Added disabled-by-default `GFX906_SIDECAR_TRACE` support to dump `device`, `channel`, RCCL `op_count`, device `clock64()` start/end, and launch shape from inside the exact sidecar Tree/LL kernel. The analyzer groups complete `(channel, op_count)` cohorts across all TP ranks and uses device-local tail thresholds so raw GPU clocks are not compared directly across devices. | Diagnostic milestone. The `.20` capped trace run produced `8` equal files with `196,992` entries each, zero overflow, and `196,992` complete logical cohorts. In the `op_count >= 10000` steady cut, device-local p99 multiplicity was one-device `5557` versus multi-device `3134`, so rank/device-local arrival or occupancy sensitivity remains plausible while multi-rank tail behavior is still meaningful. Keep this trace path for source attribution; do not enable it in winner runs. |
| Sidecar channel-offset partition | `.tmp/sidecar_channel_offset_candidate_20260613/rccl_sidecar_tree_ll_work_probe_channel_offset_20260613.cpp`; `.tmp/sidecar_channel_offset_candidate_20260613/gfx906_sidecar_tree_ll_ar_runtime_channel_offset_20260613.cpp`; `.tmp/sidecar_channel_offset_candidate_20260613/gfx906_sidecar_tree_ll_ar_runtime_multirow_channel_offset_20260613.cpp` | Added opt-in channel offset controls so the promoted three-channel sidecar descriptor can use channels `1,2,3` instead of `0,1,2` while preserving the same count split, `nWarps=4`, and 256-thread launch shape. | Exact but rejected for serving. Balanced-tree replay improved only slightly (`.20` `0.0295653 -> 0.0293985 ms/call`; `.30` `0.0294441 -> 0.0293987`), and strict multi-row sidecar smokes did not beat the same-host high-water: `.20` `62.428` backend TPS versus `62.491`, `.30` `62.183` versus `62.192`. Keep the promoted default channel start unless a later primitive rewrite changes the payload or channel topology. |
| Sidecar skinny shared-memory load | `.tmp/sidecar_skinny_shmem_candidate_20260613/rccl_sidecar_tree_ll_work_probe_skinny_shmem_20260613.cpp`; `.tmp/sidecar_skinny_shmem_candidate_20260613/gfx906_sidecar_tree_ll_ar_runtime_multirow_skinny_shmem_20260613.cpp` | Added opt-in `SIDECAR_SKINNY_SHMEM` / `VLLM_GFX906_PERSISTENT_AR_SKINNY_SHMEM` so the sidecar Tree/LL kernel skips full `ncclDevComm`/`ncclDevChannel` copies and initializes only the fields consumed by the LL primitive: rank/node metadata, LL buffer size, abort flag, channel peer pointer, tree, and work counter. | Exact source milestone but rejected for serving promotion. Direct graph replay improved `.20` `0.0296748 -> 0.0295084 ms/call` and `.30` `0.0296938 -> 0.0295101`; C API in-place smokes were exact. Strict c1 serving reached `.20` `62.362` backend TPS versus high-water `62.491`, and `.30` `62.238` versus `62.192`, so the direct gain is below serving variance and does not clear the tail-driven gate. |
| Sidecar skinny shared-memory plus no-zero descriptor | `.tmp/sidecar_skinny_shmem_candidate_20260613/rccl_sidecar_tree_ll_work_probe_skinny_nozero_20260613.cpp`; `.tmp/sidecar_skinny_shmem_candidate_20260613/gfx906_sidecar_tree_ll_ar_runtime_multirow_skinny_nozero_20260613.cpp` | Kept the skinny shared-memory path and removed the sidecar `ncclDevWorkColl` byte-zero loop, relying on explicit assignments for the fields used by the fixed Tree/LL descriptor. | Exact direct source milestone but rejected for serving. Balanced replay improved to `.20` `0.0292063 ms/call` and `.30` `0.0292124`, about another `1%` over skinny-only. C API in-place smokes were exact on both hosts. Strict serving smokes were valid but below high-water: `.20` `62.183` backend TPS versus `62.491`, and `.30` `62.172` versus `62.192`. This closes descriptor-init cleanup as a gate-scale lever. |
| Sidecar descriptor flags | `.tmp/sidecar_descriptor_flags_candidate_20260613/rccl_sidecar_tree_ll_work_probe_descriptor_flags_20260613.cpp`; `.tmp/sidecar_descriptor_flags_candidate_20260613/gfx906_sidecar_tree_ll_ar_runtime_descriptor_flags_20260613.cpp` | Added opt-in controls for the fixed sidecar `ncclDevWorkColl` descriptor fields `rcclUseOneSlice` and `gfx9CheapFenceOff`, preserving the promoted balanced-tree, three-channel, `nWarps=4`/256-thread launch shape. | Exact but rejected before serving. Direct replay on `.20` showed tiny hot-row wins versus baseline `0.029676425 ms/call`: one-slice `0.029630200`, cheap-fence-off `0.029591338`, both `0.029592262`. `.30` did not reproduce: baseline `0.029613550`, one-slice `0.029620125`, cheap-fence-off `0.029659450`, both `0.029671525`. This is below the promotion threshold and not portable, so do not build a serving candidate from these flags. |
| Sidecar skinny/no-zero plus channel offset | `.tmp/sidecar_skinny_nozero_chanoffset_candidate_20260613/rccl_sidecar_tree_ll_work_probe_skinny_nozero_chanoffset_20260613.cpp`; `.tmp/sidecar_skinny_nozero_chanoffset_candidate_20260613/gfx906_sidecar_tree_ll_ar_runtime_multirow_skinny_nozero_chanoffset_20260613.cpp` | Combined the skinny shared-memory load, no-zero hot descriptor setup, and opt-in channel offset so the fixed three-channel Tree/LL sidecar can use channels `1,2,3` while preserving `nWarps=4`/256 threads. | Direct replay stacked the independent source wins and is a useful microbench milestone: offset `1` reached `.20` `0.0291214 ms/call` and `.30` `0.0290683`, faster than offset `0` on both hosts. Runtime C API `5120` smokes were exact with offset `1` and skinny shmem. Serving did not promote: strict c1 smoke reached `.20` `62.338` backend TPS versus high-water `62.491`, and `.30` `62.358` versus `62.192`. Keep as evidence that descriptor-side median wins are now below serving variance unless they reduce launch count or tail latency. |
| Public communication trace hook | `overlays/gfx906_persistent_ar_20260612/parallel_state.py` | Added disabled-by-default metadata tracing for public `all_reduce`, public/miss `all_reduce_inplace`, `all_gather`, and `reduce_scatter` paths using `VLLM_GFX906_PUBLIC_COMM_TRACE=1`. | Diagnostic milestone. `.20` capped trace runs showed public fallback is dominated by prefill/profiling `2048x5120` and CUDA graph capture `2/4/8x5120`, while exact `1x5120` decode replacements still hit the sidecar. Keep the hook for future attribution; do not enable for winner runs. |
| Sidecar direct primitive bypass | `.tmp/sidecar_direct_prims_candidate_20260613/rccl_sidecar_tree_ll_work_probe_direct_prims_20260613.cpp` | Bypassed `RunWorkColl` in a throwaway fixed-shape sidecar source and directly instantiated Tree/LL reduce and broadcast `Primitives` for `fp16 count=5120` while preserving the promoted three-channel descriptor and `4/256` launch. | Exact but rejected as too small: same-session graph replay moved `.20` `0.029680 -> 0.029666` and `.30` `0.029658 -> 0.029632` ms/call. This proves generic `RunWorkColl` wrapper overhead is not the gate-scale source. |
| Sidecar role-specialized primitive arity | `.tmp/sidecar_role_specialized_candidate_20260613/rccl_sidecar_tree_ll_work_probe_role_specialized_20260613.cpp`; `.tmp/sidecar_role_specialized_runtime_20260613/gfx906_sidecar_tree_ll_ar_runtime_role_specialized_20260613.cpp` | Narrowed Tree/LL primitive fan templates by each rank's actual fixed tree role while preserving the promoted three-channel sidecar descriptor and `4/256` launch; later parameterized the runtime with `SIDECAR_COUNT_*` and `SIDECAR_ROLE_UP_THREADS` so the current `2048/1536/1536` descriptor and up/down split variants could be checked directly. | Exact microbench milestone but rejected for serving: repeat replay improved `.20` `0.0294769 -> 0.0293125` and `.30` `0.0295417 -> 0.0293225` ms/call, but full vLLM ladders regressed to `.20` `57.807/58.800/55.799` and `.30` `56.102/56.705/53.850` backend TPS for strict c1 / c1_2000 / c1_10000. The current-descriptor split revisit also rejects before serving: `.20` split `32/96` `0.089512`, split `64/64` `0.086988`, split `96/32` timeout; `.30` split `32/96` `0.081877`, split `64/64` `0.098859`, split `96/32` timeout. This closes generic primitive fan-template and up/down split-allocation overhead as gate-scale sources. |
| One-block serial-channel sidecar | `.tmp/sidecar_oneblock_serial_channels_candidate_20260613/rccl_sidecar_tree_ll_work_probe_oneblock_serial_channels_20260613.cpp` | Collapsed the sidecar channel launch from three channel-parallel blocks to one block that loops over channels `0,1,2`, testing whether a lower-occupancy resident/parked worker could preserve the allreduce floor. | Exact but rejected: `.20` `0.128805 ms/call` and `.30` `0.125897 ms/call`, both far slower than the promoted three-block sidecar floor and slower than same-session stock RCCL. Do not serialize channels as a parked-worker strategy. |
| ROCm hardware queue clamp side lane | `run_qwen36_27b_persistent_ar_decode_lane_20260612.sh`; runtime env `GPU_MAX_HW_QUEUES=1` via `EXTRA_DOCKER_ENV_ARGS_APPEND` | Tested whether reducing ROCm hardware queues could flatten rank-arrival skew at the sidecar collective boundary without source changes. | Rejected: `.20` full ladder reached only `57.709/58.724/55.657` backend TPS for strict c1 / c1_2000 / c1_10000, below the multi-row sidecar high-water `62.491/63.308/59.783`. Queue knobs are not a substitute for source-level scheduling or placement changes. |
| Sidecar comm-stream fork/join | `.tmp/sidecar_commstream_runtime_20260613/gfx906_sidecar_tree_ll_ar_runtime_multirow_20260613.cpp`; env `VLLM_GFX906_PERSISTENT_AR_COMM_STREAM` | Added an opt-in ready-event / separate HIP stream / done-event fork around the sidecar allreduce to test whether high-priority communication stream placement could reduce rank-skew tails under serving. | Rejected. Small `5120` C API correctness passed, but high-priority keep-handle timing regressed to `0.2295 ms/call` versus current-stream `0.1018`; normal-priority was `10.756 ms/call`. The `.20` serving run hung in PIECEWISE graph capture after `40960` replacement logs and never reached readiness. Do not use external HIP stream/event choreography inside the graph-captured custom op. |
| Separate rank-stagger delay kernel | `.tmp/sidecar_stagger_runtime_20260613/gfx906_sidecar_tree_ll_ar_runtime_stagger_20260613.cpp`; env `VLLM_GFX906_PERSISTENT_AR_STAGGER_ITERS` | Added an opt-in one-block `s_sleep` delay kernel before the sidecar allreduce, with per-device iteration lists derived from the `.30` sidecar rank-tail profile. | Rejected for serving graph-capture liveness. C API smokes were exact: zero-delay `.20` mean `0.077210 ms/call`; uniform `1000` iterations mean `0.121354 ms/call`. But serving hung before readiness on `.20` with calibrated `1081,229,697,0,611,814,995,475` and on `.30` with tiny `80,20,50,0,45,60,75,35`, both in PIECEWISE graph capture after `40960` replacement logs. Do not add a separate captured delay node before the collective. |
| In-kernel rank-stagger sidecar delay | `.tmp/sidecar_instagger_runtime_20260613/gfx906_sidecar_tree_ll_ar_runtime_instagger_20260613.cpp`; env `VLLM_GFX906_PERSISTENT_AR_STAGGER_ITERS` | Moved the per-rank `s_sleep` delay into the existing sidecar kernels before `RunWorkColl` / `RunWorkBatch`, avoiding the extra captured delay kernel node. | Rejected for serving graph-capture liveness. `.20` C API smokes were exact: zero-delay mean `0.0792557875 ms/call`, rough list `80,20,50,0,45,60,75,35` mean `0.076963375 ms/call`. Serving still hung in PIECEWISE graph capture after `40960` replacement logs and never opened the backend port. This closes simple delay-based rank staggering as a near-term path; the issue is not only the extra graph node. |
| RCCL `ncclKernelMain` args-storage parity | `microbenches/rccl_kernelmain_args_parity_probe_20260612.cpp`; `run_rccl_kernelmain_args_parity_probe_20260612.sh` | Built a stock-style `ncclDevKernelArgs4K` packet with three `ncclDevWorkBatch` entries and one shared `ncclDevWorkColl`, then entered a private `ncclKernelMain` wrapper. Corrected the gfx906 wave64 launch requirement from `128` threads, which faults, to `192+` threads so `loadWorkBatchToShmem` has a loader wavefront. | Exact on `.20/.30`, but rejected for speed: private wrapper `~0.0488 ms/call`, slower than one-work sidecar and clean stock. Useful integration-shape milestone only; next work must instrument/patch real RCCL enqueue/kernel path. |
| RCCL single-work fastload | `patches/rccl_721_gfx906_tree_ll_singlework_fastload_after_20260613.patch` | Added a guarded shortcut in `loadWorkBatchToShmem` for the common public RCCL single-work coll batch (`funcId=0`, `offsetBitset=1`, no extensions), copying one `ncclDevWorkColl` directly into shared work storage. | Rejected for dense c1 decode. Reduced `.30` build with `ONLY_FUNCS="AllReduce TREE LL Sum f16"` was exact, but direct graph replay regressed the hot `1x5120` row to `0.044682 ms/call` graph-out versus the current baseline family around `0.0307`; wider rows improved (`32x5120 0.257113`), but that does not help the single-request decode gate. Full `.20` build was stopped after the reduced replay rejected the patch. |
| Single-chunk Tree/LL | `rccl_721_gfx906_tree_ll_singlework_ll16_primsinline_singlechunk_20260612.patch`; `rccl_721_gfx906_tree_ll_singlechunk_no_runtreesplit_forceinline_after_20260612.patch` | Added a guarded single-chunk fast path inside `runTreeSplit` on top of singlework/LL16/prims-inline; also tested a reduced-inline fallback to avoid extra compile pressure. | Exact direct replay, but rejected: `.20` hot rows `1x5120` and `2x5120` regressed while wider-row wins were too small. |
| Chain-fan specialization | `chainfan1`, `static_chainfan1`, `chainfan1_half2_sum` | Specialized the one-child Tree/LL chain and combined it with half2 sum. | Correct but not a dense c1 promotion; static branch unsafe. |
| Binary-fan specialization | `binaryfan2` | Narrowed primitive fan templates to binary fan for promoted binary tree. | Useful component; standalone and combined serving did not beat `.20` high-water. |
| Half2 sum specialization | `half2_sum`, `chainfan1_half2_sum` | Added guarded gfx906 `FuncSum<half>` half2 reduce specialization using `__hadd2`. | Source-valid, but row gains were too small/mixed. |
| Aligned half DataLoader | `rccl_721_gfx906_tree_ll_aligned_half_loader_20260612.patch` | Added an aligned u64 load path for `eltN==EltPerLine` fp16 LL data. | Rejected; did not improve the row family. |
| Volatile 128-bit LL FIFO load | `rccl_721_gfx906_tree_ll_singlework_volatile_b128_load_after_20260612.patch` | Replaced the gfx906 LL FIFO pair of 64-bit non-temporal loads with a guarded volatile `uint32_t[4]` vector load on top of singlework/LL16/prims-inline. | Rejected. It compiled and was exact on completed rows, but `.30` direct replay regressed `1x5120` from `0.030711` to `0.035552 ms` and `4x5120` from `0.052496` to `0.065026 ms`, then stalled before the wider row family completed. |
| LL128 threshold | `tree_ll128_threshold*` | Routed wider rows to LL128-style behavior while preserving hot Tree/LL rows. | Prefill/concurrency lever only; serving decode regressed. |
| Channel/planner partitioning | `force4_10kb`, `force3balanced_10kb`, channel count sweeps | Forced hot `count=5120` fp16 rows to different channel splits. | Rejected; current three-channel planner split with min/max 4 is better. |
| Kind-routed sidecar library selection | `.tmp/sidecar_kindroute_runtime_20260613/*`; `run_qwen36_27b_sidecar_kindroute_decode_lane_20260613.sh` | Added `all_reduce_inplace_kind` and route-aware `VLLM_GFX906_PERSISTENT_AR_LIB_ATTN` / `VLLM_GFX906_PERSISTENT_AR_LIB_MLP` loading so attention and MLP RowParallel calls can use different sidecar `.so` files while non-target calls keep the default. | Infrastructure milestone only. O3 graph capture and strict serving were valid, but using the `1792/1664/1664` split for attention-only reached `.20 62.362` / `.30 61.887` and MLP-only reached `.20 62.419` / `.30 61.995`, all below same-host high-water. Keep the routing substrate for future per-family variants; reject this alternate split. |
| LLMM1 sidecar ready/event window | `microbenches/rccl_llmm1_ar_rms_boundary_bench_20260612.cpp`; `run_rccl_llmm1_ar_rms_boundary_bench_20260612.sh` | Added `llmm1_rpb2_ready_kernel`, optional ready-counter polling in `boundary_sidecar_tree_ll_1x5120_kernel`, a two-stream graph replay harness for event-gated sidecar launch, and `BOUNDARY_FUSED_CHANNEL_READY=1` for channel-local producer readiness inside the fused LLMM1+sidecar kernel. Later corrected the producer-readiness trace from raw `clock64()` to RCCL `globaltimer()` and added explicit `timer_source` / `*_wall_ticks` fields. | Ready-counter spin windows are rejected: volatile and atomic `.20` smokes both hard-stalled with all GPUs at `100%` and were stopped with exit `137`. Event-gated sidecar launch is exact but not a promotion: full `.20/.30` deltas versus same-stream sidecar are only `+0.000003844/-0.000024919 ms` and `+0.000006445/+0.000014985 ms` for `attn_5120x768` / `mlp_down_5120x2176`. Channel-local fused readiness is exact but also rejected: `.20/.30` fused channel-ready remains `+13 us` on attention and `+24 us` on MLP-down versus the separate `LLMM1 -> sidecar allreduce` boundary. The corrected wall-clock trace is promoted as a diagnostic/source milestone: `.20/.30` each produced `48` valid trace rows with no wall rows exceeding HIP event duration. Channel `0` is fully ready around `4-5 us`, channel `1` around `6-7 us`, and channel `2` at the full `7-8 us` producer span. Future overlap work must be true channel/LL-line or chunk-level propagation without a CTA spin barrier or extra launch. |
| RCCL AllReduceWithBias | `rccl_721_allreduce_with_bias_*`; `overlays/rccl_allreduce_with_bias_*` | Added API/export/acc paths for allreduce plus bias/residual-like accumulation. | Correct enough to serve, but performance rejected. |
| RCCL LLMM1 precompute API | `rccl_721_gfx906_tree_ll_llmm1_precompute_20260612.patch` | Added public API-table export and metadata plumbing for in-collective producer precompute. | Infrastructure milestone; scalar producer inside RCCL geometry is far too slow. |
| ReduceScatter + AllGather equivalence | `rccl_rs_ag_equiv_graph_bench_20260612.cpp` | Tested SP-like substitute collectives against promoted allreduce. | Rejected for dense c1 decode; keep SP as prefill side lane. |
| Public send/recv recursive doubling | direct allreduce graph harness method switches | Tested XOR/ring/allgather schedules through public RCCL calls. | Rejected; slower or P2P/communicator failures. |
| Symmetric/cumem/custom allreduce | `rccl_721_symmetric_f16_allreduce_only_*`, `force_rocm_pcie_custom_ar_*`, AITER probes | Tried ROCm symmetric memory, vLLM custom allreduce, and AITER-style fused AR/RMS substrates. | Rejected/blocked by gfx906 PCIe peer/IPC substrate and invalid device pointer paths. |

## Model-Structure Source Paths

| Source family | Files / overlays / kernels | What was adapted for gfx906 | Disposition |
| --- | --- | --- | --- |
| Replicated attention output projection | `attention_repl_oproj_20260608` MEP | Replaced attention o-proj RowParallel allreduce with gather plus replicated full GEMV. | Rejected; about 2.8x slower than current attention boundary. |
| MLP activation-gather plus replicated down-proj | MLP full-projection MEPs | Tested avoiding MLP/down allreduce by gathering activations and computing replicated output. | Rejected for bandwidth/compute cost. |
| TP8-to-TP4 MLP bridge | narrow bridge MEP | Tried zero-padded pair allreduce and paired down-projection groups. | Rejected; extra bridge work outweighed any collective savings. |
| Static SP / sequence-parallel side lane | SP threshold/runtime overlays | Tested reduce-scatter/all-gather rewrite and static compile constraints. | Controlled prefill/no-regression side lane only; not dense c1 decode. |
| Post-IR collective blocker trace | `overlays/postir_blocker_trace_20260613/gfx906_post_ir_ar_rms_fusion.py` | Added shortest-path diagnostics from each collective output to the next collective input, preserving dry-run behavior. | Promoted diagnostic milestone: all `128` adjacent dense collective pairs are blocked by true hidden-state dependence into `rocm_gemm_5120` producers (`64` MLP, `47` normal attention, `16` GDN/linear attention, `1` initial path). Standalone grouped allreduce is not viable for the current graph. |
| Post-IR attention shape-classified split endpoint | `overlays/postir_split_endpoint_rms_20260611/gfx906_post_ir_ar_rms_fusion.py`; `runs/qwen36_27b_decode_tiers_postir_attention_split_endpoint_torchop_kshape2_limit64_split4_h{20,30}_experimental_host{20,30}_20260613_attention_kshape2_torchop_h{20,30}` | Repaired current-graph attention matching by classifying RowParallel producers from local K dimension (`K=768` attention, `K=2176` MLP/down) and allowing attention to use the immediate `float -> residual add -> RMS` chain instead of the older `setitem` path. | Source infrastructure milestone only. The pass now selects `64/64` attention boundaries on `.20` and `.30`, but strict smokes were only `.20` `55.585` and `.30` `53.449` backend TPS. Keep the classifier; reject consumer-only split-endpoint RMS as a serving path. |

## Practical Lessons For Next Source Work

1. Tiny RCCL primitive tweaks are now near the floor. Sub-one-percent Tree/LL
   row wins do not translate reliably to serving.
2. The most credible dense gate path is still exact MLP/down RowParallel:
   reduce launch/count cost, legally coalesce below dependency boundaries, or
   fuse producer/collective/consumer below graph-op overhead.
3. Consumer-only RMS/logits cleanup is worth keeping as a component milestone,
   but it is too small to clear `65 TPS` without a larger collective change.
4. MoE and dense configs must stay separate. Dense TP8 likes balanced trees,
   four channels, and `NCCL_NTHREADS=128`; MoE TP4 remains best around
   one-channel Tree/LL with `NCCL_NTHREADS` unset.
5. New kernel work should be promoted only after direct replay and then the
   full strict decode ladder: uncapped c1_128, c1_2000, c1_10000.
6. The current dense sidecar gap is tail-driven. The `.30` multi-row winner
   profile has `sidecar_tree_ll_1x5120_kernel` median `45.8 us` but p99
   `3.64 ms`; next source work should flatten rank-skew/barrier tails rather
   than chase another tiny direct-replay median win.
7. Simple queue clamping is not enough. `GPU_MAX_HW_QUEUES=1` regressed the
   `.20` multi-row sidecar ladder to `57.709/58.724/55.657` backend TPS, so
   the next credible work needs source-level scheduling, placement, or
   launch-count changes.
8. External HIP comm-stream fork/join is not graph-capture safe here. The
   high-priority stream variant hung before serving readiness, so future tail
   work should stay graph-native or patch the lower-level RCCL launch path.
9. The sidecar p99 tail is now tied to two producer shapes, not to arbitrary
   descriptor setup or a single modulo position. Future source work should map
   `LLGemm1 grid=327680/wg=128` and `LLGemm1 grid=819200/wg=320` back to the
   current attention/MLP RowParallel callsites and patch that full boundary.
10. The attention callsite mapping is repaired in post-IR, but replacing only
    the RMS/residual consumer remains too slow. Next work should move below the
    post-IR wrapper level into producer/collective scheduling or a lower-level
    fused/overlapped boundary implementation.
11. The latest pair/GPU skew evidence rules out slow local `LLGemm1` producers
    as the primary cause of the worst attention tail. Prioritize sidecar
    rank/topology behavior, launch-count collapse, or lower-level graph-native
    collective boundary work over standalone producer GEMM tuning.
12. Basic rank-to-GPU reordering does not promote the dense sidecar stack.
    Reverse and even/odd `HIP_VISIBLE_DEVICES` orders stayed at or below the
    same-host high-water (`.20` best near-tie `62.436` vs `62.491`, `.30` best
    `61.312` vs `62.192` strict c1). Keep rank placement as context, but spend
    source time on the collective schedule or graph-native boundary.
13. Odd-root balanced tree rotations do not promote the dense sidecar stack.
    Switching the balanced `RCCL_TREES` roots from even ranks to odd ranks was
    valid but slower: `.20` reached `62.000` versus `62.491`, and `.30`
    reached `61.408` versus `62.192` strict c1 backend TPS. Simple tree-root
    shifts are closed; future collective work needs to change the sidecar
    primitive, schedule, launch count, or graph-native boundary.
14. Multi-row capture is not the single-request c1 launch-count fix. The
    `40960` sidecar replacement logs are from the size-8 graph capture path;
    strict c1 decode replays the size-1 graph and still profiles as roughly
    `98k` `sidecar_tree_ll_1x5120_kernel` rows. Keep multi-row as a
    multi-request/prefill milestone, but the dense decode gate needs hot
    `1x5120` primitive or boundary work.
15. Descriptor split tuning is below serving scale. The best exact
    `countLo/countMid/countHi` variant, `2048/2048/1024`, saves only tens of
    nanoseconds per direct graph call on `.20/.30`; chunk-grain variants are
    neutral or worse. Do not spend serving-lane time on `cbd` partition changes
    unless a future tail-specific profile points there.
16. Cross-rank sidecar tail attribution needs explicit sequence IDs. The new
    `analyze_sidecar_crossrank_sequence_20260613.py` and
    `analyze_sidecar_crossrank_laggard_20260613.py` reports now guard against
    unsafe occurrence-index joins when per-GPU sidecar row counts are
    imbalanced. The existing `.20`/`.30` profiles have count spreads of
    `16912`, `15180`, and `11429` rows, so they cannot prove one-rank versus
    multi-rank tail ownership. The next credible source step is graph-safe
    sequence/arrival instrumentation around the hot sidecar allreduce, then a
    mitigation based on that stronger evidence.
17. The promoted multi-row runtime is not enough by itself for dense c1 decode.
    Trace-enabled `.20/.30` `2048/1536/1536` runs show multi-work sidecar rows
    only during early graph/warmup ranges, while the clean `op_count >= 10000`
    steady gate region is entirely `work_count=1`. The next launch-count
    breakthrough requires changing graph lowering/capture or the RowParallel
    boundary so strict single-request decode issues a fused/multi-work
    allreduce, otherwise the work remains a hot `1x5120` primitive/tail
    problem.
18. Launch-bound compiler wins must be serving-validated. The
    `SIDECAR_LAUNCH_BOUNDS_*` experiment made the isolated `.20` C API path
    look much better, but the same change regressed `.30` and failed strict
    serving on both hosts. Treat compiler-resource knobs as diagnostics unless
    they improve the promoted O3 vLLM ladder.
19. Captured call-site labels are now the preferred sidecar tail diagnostic.
    They identify the 128 graph-captured steady allreduce nodes directly and
    expose p99-heavy classes without relying on cross-GPU timestamp alignment
    or occurrence-index joins. The first simple route, mod4-to-`1792/1664/1664`,
    did not promote, so use the call-site signal to map layer/family and alter
    the boundary or primitive rather than only descriptor counts.
20. The current dense 27B tail target is attention RowParallel output
    allreduce. The call-site/kind join shows the steady `1664-1791` graph-node
    range alternates `attn, mlp`, and the dominant `.20` p99 sites are all
    even-layer attention call sites. Future source work should therefore focus
    on attention producer/collective scheduling, graph-native boundary fusion,
    or a lower-latency `1x5120` attention allreduce path before revisiting MLP
    or broad descriptor routing.
21. Per-node primitive selection is viable, but only as a scalpel. Moving the
    whole even-attention class to public RCCL regresses, while moving only
    `call_site=1664` improves `.20` by `+0.278/+0.186/+0.171` TPS across the
    standard ladder. Follow-up sparse and host-local singleton tests also
    reject adding more public fallback nodes: `.20` `1664,1668`, `1668`,
    full-ladder `1692`, and `1664,1692` all lose, and `.30`
    top-4/`1692`/`1672`/`1688` do not show a promotion signal. Exact
    alternate-descriptor routes are also rejected: `.20` exact `1664` alt
    `62.792`, `.30` exact `1664` alt `62.031`, `.30` exact `1672` alt
    `62.132`, and `.20` public `1664` plus alt `1668` hybrid `62.604`. The
    next likely gain is not more public fallback or `1792/1664/1664` routing;
    it is making the
    sidecar/RCCL path for that pathological attention node cheaper or less
    tail-sensitive.
22. The latest public-1664 profile says the next source target is sidecar
    tail behavior, not another fallback permutation. With `call_site=1664`
    on public RCCL, the `.20` profile still spends `47.74%` of filtered
    kernel time in `sidecar_tree_ll_1x5120_kernel`; public NCCL is already
    visible at `11.18%`. The corresponding sidecar trace shows devices `6/7`
    own most max-duration groups and have much higher p99 ticks than devices
    `0-5`, while device-local p99 groups are mostly one-device events. Future
    work should therefore prioritize rank-arrival skew mitigation,
    tree/root/channel scheduling, or graph-native attention RowParallel
    boundary changes.
23. Public1664 plus skinny/no-zero channel-offset is closed. The correct
    multi-row runtime survived O3 graph capture with
    `VLLM_GFX906_PERSISTENT_AR_CHANNEL_OFFSET=1`,
    `VLLM_GFX906_PERSISTENT_AR_SKINNY_SHMEM=1`, and public fallback only for
    `kind=attn, call_site=1664`, but strict c1 serving lost on both primary
    hosts: `.20` `62.282` versus public1664 high-water `63.084`, and `.30`
    `61.211` versus public1664 support `62.197`. Do not return to
    descriptor-cleanup/channel-offset composition without a new tail-specific
    mechanism.
24. Public1664 plus odd-root tree rotation 3 is closed. The rotated
    `RCCL_TREES` string survived graph capture, but strict c1 serving lost:
    `.20` `62.383` versus public1664 high-water `63.084`, and `.30`
    `61.432` versus public1664 support `62.197`. Whole-tree root rotation is
    not the targeted sidecar-tail fix; retain the promoted even-root balanced
    tree string.
25. Exact captured-callsite library routing is available for future source
    experiments. The kind-route overlay now supports
    `VLLM_GFX906_PERSISTENT_AR_LIB_CALLSITE` with exact/range/mod callsite
    selectors while preserving global sequence numbering and public-bypass
    behavior. Routing only `kind=attn, call_site=1664` to the old
    role-specialized Tree/LL primitive passed O3 graph capture and strict c1
    on both primary hosts, but did not promote: `.20` `62.648` versus
    public1664 `63.084`, and `.30` `61.681` versus public1664 `62.197`. Use
    this router for truly different exact-node primitives or boundary changes,
    not for more generic descriptor/fan-template permutations.
26. The public-1664 stack still keeps `NCCL_NTHREADS=128`. The kind-route
    runner now treats `NCCL_NTHREADS=unset|none|default` as a real omitted env
    var. With public fallback only for `kind=attn, call_site=1664`, unset
    produced `.20` full-ladder `63.053/63.889/60.294`, below the promoted
    `63.084/63.993/60.380`, and `.30` strict c1 `61.889` versus public1664
    `62.197`. Do not revisit public-thread-count tuning without a new public
    primitive or callsite mix.
27. Exact-node skinny/no-zero/channel-offset sidecar routing is closed.
    Routing only `kind=attn, call_site=1664` through
    `libgfx906_sidecar_tree_ll_ar_multirow_skinny_nozero_chanoffset_20260613.so`
    passed O3 graph capture and strict c1 on both hosts, but lost:
    `.20` `62.873` and `.30` `61.442` versus public1664 `63.084` and
    `62.197`. The callsite router is useful, but this source variant is still
    only a median/descriptor cleanup and not the tail-sensitive `1664` fix.
28. Fixed-size sidecar primitives need shape guards before serving tests.
    The channel-base `1x5120` candidates looked valid in direct C API tests,
    but broad serving routes hit unsupported larger tensors and failed with
    `rc=2`. The Python runtime now supports numel-gated callsite routing, so
    exact `kind=attn, call_site=1664, numel=5120` can be sent to a fixed-size
    library without affecting unrelated reductions. The corrected channel-base
    smokes did not promote (`.20` `62.860`, `.30` `61.653`), but the guard is
    required infrastructure for any future single-shape collective kernel.
29. The public RCCL side of the public1664 hybrid should stay four-channel.
    Exact residual fallback for `1664,1669` did not promote, and lowering the
    public path to three channels was only a near-tie on `.20` while losing on
    `.30`. Lowering to two channels is not a performance candidate: both
    primary hosts hit GPU memory access faults during graph capture. Future
    dense source work should return to the hot `1x5120` sidecar primitive,
    explicit arrival instrumentation, or graph-native RowParallel boundary
    reduction rather than more public-channel permutations.
30. Wall-clock sidecar tracing is now the preferred cross-rank attribution
    tool. The new trace adds wall-clock entry/run-start/run-end ticks and
    median per-device offset correction, avoiding unsafe raw `clock64()`
    comparisons across GPUs. On the public1664 `.20` serving trace, adjusted
    arrival spread was visible but smaller than run-wall dominance; devices
    `7` and `6` owned most max run-wall groups, and top p99 call sites shifted
    toward `mlp/sidecar/5120`. The first resulting MLP fallback test,
    `1664,1711`, was valid but rejected (`.20` `62.871`, `.30` `61.769`).
    Follow-up public fallback for `1759` and `1743` also rejected, and exact
    skinny/no-zero/channel-offset sidecar routing for `1711,1759,1743` failed
    the full `.20` ladder after one small smoke bump. The breakthrough is
    measurement quality, not a new winner.
31. Simple sidecar channel block-order changes are closed. The uncapped
    public1664 trace showed channel `1` with the highest entry-run p99, but
    moving it later (`0,2,1`) reached only `.20/.30` `63.034/62.163`, and
    moving it earlier (`1,0,2`) reached `63.006/61.838`, all below same-host
    public1664. The tail is not removed by reordering the same three Tree/LL
    blocks. Continue with deeper sidecar schedule/primitive work,
    graph-native launch-count reduction, or RowParallel boundary fusion.
32. Host `.50` ROCm 7.2 requires selective gfx906 device exposure. The host
    has a `gfx803` display adapter at physical GPU0 and eight MI50 gfx906
    devices at physical GPUs 1-8. Broad `/dev/dri` exposure or `--privileged`
    makes ROCm 7.2 fail HSA init inside the container; exposing only
    `/dev/dri/card1-8`, `/dev/dri/renderD129-D136`, `/dev/kfd`, and the
    render/video groups makes PyTorch ROCm 7.2 usable with in-container
    ordinals `0-7`. The sidecar decode runners now encode this `.50`
    behavior, and the sidecar build, C API smoke, and replicated RowParallel
    microbench helpers now use the same selective `.50` exposure. This is lane
    infrastructure, not a source-kernel promotion: the first `.50` public1664
    strict smoke was valid but only `56.129` backend TPS, while the corrected
    `.50` C API helper smoke passed exactly.
33. Contiguous sidecar channel-3 inclusion is closed. The multi-row runtime
    now has compile-time `SIDECAR_CHANNEL_LO/HI` in addition to
    `SIDECAR_CHANNEL_MAP0/1/2`, preserving the default `0-2` map while
    allowing contiguous sets such as `1-3`. The `1,3,2` candidate was exact
    on `.20/.30`, but C API timing was not better than control and strict
    public1664 serving reached only `.20` `62.638` and `.30` `62.197`
    backend TPS. Do not spend more time on simple active-channel selection
    unless the underlying Tree/LL primitive or work descriptor semantics
    change.
34. Host `.50` does not currently support peer-memory collectives despite the
    PCIe-switch topology and GPU8 repair. Under selective ROCm 7.2 exposure,
    HIP reports eight gfx906 devices but a zero off-diagonal
    `hipDeviceCanAccessPeer` matrix, unchanged by
    `HSA_FORCE_FINE_GRAIN_PCIE=1`. The RCCL send/recv P2P graph probe fails at
    `ncclGroupEnd` before timing. Keep `.50` as a compatibility/diagnostic
    lane, but do not use it as the basis for P2P/symmetric-memory dense gate
    source work unless a future driver or BIOS change alters peer access.
35. Variable active-channel sidecar support is lab infrastructure, not a
    promotion. The multi-row runtime now has compile-time `SIDECAR_NUM_CHANNELS`
    and `SIDECAR_CHANNEL_MAP3`, while default behavior remains the promoted
    three-channel map. Two-channel `2560/2560` proved exact and faster for
    `.20` `5120` C API timing, but failed as a portable source path:
    `.30` component timing regressed, `.20` 5120-only serving fell to
    `61.021` backend TPS, and the multi-row `10240` C API path timed out on
    both primary hosts. Four active channels were exact but did not beat
    control. Do not revisit active channel count without a deeper primitive
    change that also handles multiwork safely.
36. Targeted device-7-depth1 `RCCL_TREES` topology is a rejected scheduling
    overlay, not a source-kernel winner. It used the existing promoted
    `2048/1536/1536` multi-row sidecar binary and changed only tree labels so
    device `7` was a depth-1 node on active channels. Direct C API timing was exact
    and strong on `.20/.30`, but strict vLLM public1664 serving rejected it on
    the strongest lane (`.20` `62.536` versus `63.084` high-water). Keep the
    evidence as support for deeper primitive/tail work, not as a deployable
    topology change.
37. The corrected serving wall-clock trace for device-7-depth1 topology
    closes tree-label retuning more strongly. It improved arrival spread but
    left the run-wall tail on devices `6/7`; depth-1 rows became the
    expensive role. This points away from more `RCCL_TREES` permutations and
    toward changing the sidecar Tree/LL primitive itself or eliminating the
    captured RowParallel collective launch boundary.
38. C API worker occupancy controls were added to the smoke harness
    (`CAPI_WORKER_NWARPS`, `CAPI_BLOCK_THREADS`) and used to retest the current
    public1664 sidecar primitive. `nWarps=4`, `block_threads=256` remains the
    best exact shape on `.20/.30`; lower worker counts are slower and wider
    blocks fail at launch. This closes worker-count retuning as a source-kernel
    promotion path for the current primitive.
39. Current-descriptor resident sidecar support is diagnostic infrastructure,
    not a promotion. The resident runtime was rebuilt with the current
    `2048/1536/1536` hot descriptor and the kind-route launcher now supports
    optional post-capture resident lifecycle hooks. Exact C API timing did not
    justify serving: `.20` `0.057944` and `.30` `0.063888 ms/call`. A future
    resident branch must first beat the non-resident control in C API or change
    the worker primitive; simple lifecycle retries are closed.
40. Current-winner post-IR grouping diagnostics close simple collective
    coalescing on the existing graph. The post-IR pass now sees the
    kind-routed `all_reduce_inplace_kind` nodes in the public1664 high-water
    stack, but `.20` dry-run census found `129` collective ops and zero legal
    groupable pairs, with every adjacent candidate blocked by dependency and
    intervening use. Keep the pass as a boundary diagnostic tool, but do not
    treat it as a standalone coalescing path unless a model/executor schedule
    rewrite first creates independent adjacent collectives.
41. Sidecar Tree/LL phase tracing is promoted as diagnostic infrastructure.
    The runtime
    `.tmp/sidecar_phase_trace_runtime_20260614/gfx906_sidecar_tree_ll_ar_runtime_phase_trace_20260614.cpp`
    adds reduce-side and broadcast-side timestamps to the hot `1x5120`
    sidecar path while preserving the current `2048/1536/1536` descriptor for
    C API diagnosis. Exact `.20/.30` smokes show leaf reduce-up is stable
    (`316.0` and `296.0` P99 ticks), while leaf broadcast owns the long tail
    (`2648.8` and `25385.5` P99 ticks). This is not a serving candidate because
    the traced helper is slower than production `RunWorkColl`, but it sharpens
    the next source target: LL broadcast/recv datapath changes or a
    RowParallel boundary rewrite, not send-side wait, thread split, or
    descriptor setup retuning.
42. Sidecar LL broadcast subphase tracing is promoted as diagnostic
    infrastructure. The runtime
    `.tmp/sidecar_ll_subphase_trace_runtime_20260614/gfx906_sidecar_tree_ll_ar_runtime_ll_subphase_trace_20260614.cpp`
    and patched throwaway header
    `.tmp/sidecar_ll_subphase_trace_runtime_20260614/device/prims_ll.h`
    instrument `LLGenericOp` inside the broadcast branch. The include-order
    lesson matters: because RCCL's `primitives.h` includes `"prims_ll.h"` from
    its own directory, the successful build copies the overlay
    `build/hipify/src/device` directory into `/probe/device` and replaces only
    `/probe/device/prims_ll.h`. Exact `.20/.30` C API smokes show the
    broadcast P99 is first receive/read wait dominated: `.20`
    broadcast/read/store/post-to-end P99 `5543.6/4326.48/68/8` ticks, and
    `.30` `14268.0/12330.16/64/8`. Adjusted parent-child timing showed the
    parent had usually started before the child began its first-read wait
    (`3.34%`/`3.04%` parent-not-started on `.20/.30`), but had not finished
    by the child's first-read end (`99.93%` on both hosts). This closes local
    store/forward/post retreads and points the next source work upstream to
    first-line production/propagation through the tree, launch grouping, or
    RowParallel collective-boundary reduction.
43. RowParallel prefix-metadata routing is promoted as source control-surface
    infrastructure. The kind-route `linear.py` overlay can emit
    `kind|prefix=...` metadata for RowParallel calls, and
    `gfx906_persistent_ar.py` now strips that metadata for base sidecar routing
    while allowing opt-in prefix bypass lists. The serving check proved O3
    compatibility and strict correctness, but prefix-only public fallback for
    `language_model.model.layers.0.linear_attn.out_proj` rejected on
    performance (`.20` `62.964`, `.30` `61.862` backend TPS) because it
    selects every captured instance of that prefix instead of exact captured
    callsite `1664`. Use this for layer-targeted diagnostics and source
    rewrites, not as a promoted serving runtime.
44. Public-bypass numel filtering is promoted as diagnostic routing
    infrastructure. `gfx906_persistent_ar.py` now supports
    `VLLM_GFX906_PERSISTENT_AR_BYPASS_NUMEL_LIST/LO/HI` so public fallback can
    be constrained by tensor size while preserving default behavior. The first
    serving check used prefix metadata for
    `language_model.model.layers.0.linear_attn.out_proj` plus
    `BYPASS_NUMEL_LIST=5120`; it remained O3-capturable and strict-valid but
    rejected on performance (`.20` `62.751`, `.30` `62.137` backend TPS). Keep
    the filter for controlled attribution and future exactness checks; it is
    not a dense-gate promotion by itself.
45. LL flag-only receive polling is exact but rejected. The throwaway overlay
    `staging/minimal_fasttrack/model_selection_campaign_20260520/.tmp/sidecar_ll_flagpoll_20260614/prims_ll.h`
    added guarded `GFX906_LL_FLAG_POLL` and
    `GFX906_LL_FLAG_POLL_BROADCAST_ONLY` so the broadcast receive path polls
    only the two LL flags before loading the full 16-byte LL line. Matching
    C API controls and candidates on `.20/.30` all passed exactness, but the
    flag-poll candidate regressed materially: `.20` `0.064309 -> 0.074171`
    and `.30` `0.065196 -> 0.081752 ms/call`. Do not reopen narrower polling
    loads for the current Tree/LL primitive; the next source work still needs
    earlier first-line production/propagation or RowParallel boundary/count
    reduction.
46. Current-winner decode clocks are not stuck low under auto. The `.20`
    public1664 clock probe
    `runs/qwen36_27b_clock_probe_public1664_h20_20260614` reproduced the
    high-water band (`c1_2000` `63.922`, `c1_10000` `60.320` backend TPS)
    while sampling `rocm-smi` every two seconds. During fixed decode, all
    eight GPUs were near full utilization (`93.3%` on `c1_2000`, `97.4%` on
    `c1_10000`), and clocks ramped to the top levels (`mclk` about
    `983-1000 MHz`, `sclk` about `1704-1725 MHz`, `fclk` about
    `1259-1278 MHz`). Treat this as a diagnostic closure: do not chase low
    auto-clock explanations for the dense gate unless a future run contradicts
    the sampler.
47. Sidecar data-path prewarm is exact but not a promotion. The throwaway
    overlay `.tmp/sidecar_prewarm_runtime_20260614/gfx906_persistent_ar.py`
    adds guarded `VLLM_GFX906_PERSISTENT_AR_PREWARM_ON_GRAPH_CAPTURE=1`,
    which creates the sidecar handle and runs one scratch `1x5120` fp16
    sidecar allreduce per route/device before graph capture. This directly
    tested whether the current public-1664 win is mostly first-use queue,
    counter, or cache warmup. The prewarm path fired on all ranks and preserved
    strict thinking-gate correctness, but serving did not beat the incumbent:
    `.20` public1664+prewarm reached only `62.055` strict c1 backend TPS,
    `.20` no-bypass/default prewarm reached `63.046`, and `.30`
    no-bypass/default prewarm reached `62.326`. Keep handle preinit enabled,
    but do not add data-path prewarm to the winner path unless a future
    boundary/count rewrite gives a new reason to revisit it.
48. ai-infos Torch SDPA decode routing is exact enough to run but rejected for
    dense decode. The isolated overlay
    `.tmp/attention_sdpa_decode_overlay_20260614/candidate/` added only the
    newer fork's `TRITON_ATTN` Torch SDPA decode branch on top of the current
    `seg64/tile8` hotfixes, with compatibility shims for the older env and
    `kv_cache_dtype` surfaces. The branch activated on all eight ranks during
    graph setup, so this was not an inert env-var test. Streaming capture was
    incompatible with the overlay, but the non-streaming strict smoke on `.20`
    completed correctly: `3523` completion tokens, strict Qwen gate valid,
    answer SHA `c18d0cf06fe4b6bee4e06a0f84441b15d26fb81ba2e143efd93241cb801ffa5e`,
    and `62.567` backend decode TPS. This is below the `.20` public1664
    strict high-water `63.084`, so do not promote SDPA decode or spend a full
    ladder on it without a materially different upstream implementation. The
    attention-side fork-feature escape hatch is now closed; keep the primary
    source path on RowParallel boundary/count reduction or deeper Tree/LL
    first-line production/propagation work.
49. Broadcast-only LL send-slot order is exact but rejected before serving.
    The throwaway overlay
    `.tmp/sidecar_ll_bcast_sendorder_20260614/` added guarded
    `GFX906_LL_BCAST_SEND_SLOT0_FIRST`, applying slot `0` first only to the
    broadcast `recvCopySend` LL primitive instance (`RECV && SEND && !SRC &&
    DST`) while leaving reduce-up/root behavior unchanged. Fresh `.20/.30`
    controls initially made the candidate look useful (`.20` `0.060769 ->
    0.057145`, `.30` `0.061277 -> 0.057074 ms/call`), but reverse controls
    moved into the same or better warmed band (`.20` `0.056413`, `.30`
    `0.057005`). Do not run serving ladders for this branch. The useful
    lesson is methodological: small sidecar primitive wins need reverse-order
    C API repeats before promotion, because cold first-pass timing can produce
    false positives.
50. Fused producer/sidecar sidecar-last scheduling is exact but rejected before
    serving. The throwaway boundary-bench overlay
    `.tmp/llmm1_sidecar_fused_sidecarlast_20260614/` added guarded
    `BOUNDARY_FUSED_SIDECAR_LAST=1`, moving producer CTAs before sidecar
    channel blocks in the fused `LLMM1 -> sidecar Tree/LL allreduce`
    lower-bound kernel. On `.20`, the candidate was only a near-tie on
    attention (`0.052874 -> 0.052863 ms`) and a slight loss on MLP/down
    (`0.079530 -> 0.079605 ms`). On `.30`, the candidate corrected one
    anomalous same-source control outlier but landed in the normal fused band
    (`0.079306 ms`) rather than proving a portable win. The fused path remains
    slower than separate `LLMM1 + sidecar allreduce` by about `0.0148 ms`
    attention and `0.0257 ms` MLP/down on `.20`. This closes simple fused-grid
    block ordering as the missing dense-gate lever; continue with true
    RowParallel collective-boundary/count reduction or a materially different
    producer/collective implementation.
51. Sidecar Tree/LL128 protocol substitution is exact but rejected before
    serving. The isolated runtime source
    `.tmp/sidecar_tree_ll128_20260614/gfx906_sidecar_tree_ll_ar_runtime_ll128_20260614.cpp`
    changes only the sidecar `RunWorkColl` protocol template from
    `NCCL_PROTO_LL` to `NCCL_PROTO_LL128` while preserving the promoted
    `2048/1536/1536` descriptor, `nWarps=4`, and 256-thread launch. Matched
    C API controls were exact at `.20` `0.073089` and `.30` `0.072796`
    ms/call; LL128 was exact but much slower at `.20` `0.104334` and `.30`
    `0.105641` ms/call. This closes the sidecar protocol-swap shortcut. Do
    not run serving ladders for LL128; the remaining source path needs a
    genuinely different `1x5120` collective or a RowParallel collective-count
    reduction.
52. Specialized LL broadcast `recvCopySend` is exact but rejected before
    serving. The throwaway overlay
    `.tmp/sidecar_ll_bcast_specialized_20260614/prims_ll.h` adds guarded
    `GFX906_LL_BCAST_SPECIALIZED=1`, keeping the same wait/post/step semantics
    while replacing the generic broadcast `LLGenericOp<1,1,-1,Output>` wrapper
    with a narrower `1x5120` receive-copy-send path. Matched `.20/.30` builds
    used the promoted `2048/1536/1536` descriptor. C API results with `64`
    warmup and `1024` timed calls were exact, but not promotable: `.20`
    control/candidate/reverse-control `0.072962/0.074048/0.071260`
    ms/call, and `.30` `0.072644/0.071191/0.070815`. Do not run serving
    ladders for this branch. The generic template was not the missing
    first-line propagation lever; continue toward a materially different
    Tree/LL primitive or RowParallel boundary/count reduction.
53. Semantic RowParallel prefix public fallback is promoted as source
    infrastructure but rejected as the dense serving winner. The kind-route
    overlay now supports prefix metadata via
    `VLLM_GFX906_PERSISTENT_AR_KIND_PREFIX_METADATA=1` and semantic bypass
    with `VLLM_GFX906_PERSISTENT_AR_BYPASS_PREFIX_LIST`. Full decode ladders
    routed
    `language_model.model.layers.0.linear_attn.out_proj` to public RCCL while
    preserving sidecar handling for the rest of the fixed-shape `1x5120`
    RowParallel path. The runs were strict-valid and O3-compatible, but did
    not beat exact call-site `1664`: `.20` reached
    `62.735/63.837/60.219` and `.30` reached `62.407/63.020/59.209`
    backend TPS for strict c1 / `c1_2000` / `c1_10000`. Keep prefix routing
    for model-semantic attribution and source rewrites. Do not use broad
    prefix fallback as a replacement for the exact public1664 high-water,
    because it selects every captured instance of the same module prefix
    rather than only the captured node that won.
54. Prefix-occurrence public fallback is promoted as source infrastructure,
    not as a serving winner. The same kind-route overlay now adds opt-in
    occurrence filters:
    `VLLM_GFX906_PERSISTENT_AR_BYPASS_PREFIX_OCCURRENCE_LIST`,
    `VLLM_GFX906_PERSISTENT_AR_BYPASS_PREFIX_OCCURRENCE_LO`, and
    `VLLM_GFX906_PERSISTENT_AR_BYPASS_PREFIX_OCCURRENCE_HI`. Targeting
    occurrence `13` zero-based of
    `language_model.model.layers.0.linear_attn.out_proj` expresses the same
    semantic node that the static prefix map labeled as public fallback
    `call_site=1664`. The patch is syntax-clean and was staged on `.20/.30`.
    Strict smokes were valid (`.20` `62.763`, `.30` `61.693` backend TPS).
    Full ladders were also valid but not promotable: `.20`
    `62.576/63.795/60.188` and `.30` `60.996/63.082/59.254` backend TPS for
    strict c1 / `c1_2000` / `c1_10000`. Keep this as the preferred way to
    describe exact semantic captured-node experiments. Do not replace the
    current public1664 high-water with it, and do not expect selector
    granularity alone to close the dense `65 TPS` gate.
55. Corrected LLMM1 producer-readiness tracing is promoted as diagnostic/source
    infrastructure. The first attempt used raw `clock64()` and produced
    impossible cross-CTA spans, so the boundary bench now records block
    start/end with RCCL `globaltimer()` and emits `timer_source="globaltimer"`
    plus `*_rel_wall_ticks` aliases. The corrected `.20` and `.30` traces each
    produced `48` valid rows. Across both hosts, the dominant `5120`-wide
    producer spans only about `7-8 us`; channel `0` is fully ready by roughly
    `4-5 us`, channel `1` by `6-7 us`, and channel `2` at the full producer
    span. This supports a narrow true-overlap research branch, but rejects
    coarse spin-barrier fusion as a gate path because the measured overhead was
    larger than the available producer-overlap budget. The next overlap attempt,
    if pursued, must be a materially different LL-line/chunk primitive or a
    RowParallel launch-count reduction.
56. Generation-correct LL line-ready producer/sidecar fusion is exact but
    rejected before serving. The line-ready branch in
    `microbenches/rccl_llmm1_ar_rms_boundary_bench_20260612.cpp` and the
    throwaway RCCL header overlay
    `.tmp/ll_line_ready_include_20260614/prims_ll.h` changed the fused
    `LLMM1 -> sidecar Tree/LL` path from stale binary ready flags to monotonic
    producer-block generation counters. The repaired path is graph-replay exact
    on both `.20` and `.30`, closing the correctness hole from the first
    line-ready attempt. Longer confirmation with `512` iterations and `64`
    calls/replay rejects it: `.20` sidecar/fused line-ready was
    `0.052021/0.065216 ms` for `attn_5120x768` and
    `0.068476/0.086947 ms` for `mlp_down_5120x2176`; `.30` was
    `0.052136/0.065298` and `0.068466/0.089223`. Once correctness is enforced,
    the per-line wait and fused-grid scheduling overhead exceed the available
    overlap. Do not pursue additional spin-wait producer/sidecar fusion variants;
    shift the primary source path to RowParallel collective-count reduction or a
    different `1x5120` collective primitive.
57. Canonical resident persistent Tree/LL runtime descriptor roll-forward is
    promoted as source hygiene, not as a serving winner. The canonical
    `microbenches/gfx906_persistent_tree_ll_ar_runtime_20260612.cpp` no longer
    hardcodes the old `1024/2048/2048` split; it accepts `SIDECAR_COUNT_*` and
    `SIDECAR_CHUNK_GRAINS_*` compile-time controls while preserving old defaults.
    Fresh `.20/.30` builds with the promoted `2048/1536/1536` split passed, and
    keep-handle C API smokes were exact: `.20` mean/max
    `0.058535/0.058948 ms/call`, `.30` `0.059020/0.059329 ms/call`. Do not run
    a serving ladder for this exact resident-worker variant. The timing is still
    not a component win over the best non-resident sidecar control band, and the
    next persistent path needs a different command/scheduling primitive rather
    than another lifecycle retry with the same worker.
58. Sidecar LL broadcast edge attribution is promoted as diagnostic/source
    infrastructure. `analyze_sidecar_ll_subphase_edges_20260614.py` reuses the
    existing `.20/.30` LL subphase trace CSVs and groups broadcast first-read
    wait by exact Tree edge, child, parent, role, and channel. The output at
    `runs/gfx906_sidecar_ll_subphase_edge_attribution_20260614.md` shows the
    portable bottleneck is leaf first-read wait, not one stable edge/channel:
    `.20` leaf p99 was `5187.36` ticks with top child `dev7` at `6316.84`, while
    `.30` leaf p99 was `13922.28` ticks with top child `dev5` at `13276.96` and
    worst edges spread across multiple devices. This closes further
    rank-specific/channel-specific Tree permutations as a primary path unless a
    future trace contradicts it. The next high-value source direction is
    RowParallel collective-boundary/count reduction or a fundamentally different
    leaf broadcast/first-line propagation primitive.
59. ROCm7.2 dense/MoE portable release image is now bundled in
    `qwen36-gfx906/deploy.sh` with source/runtime artifacts under
    `qwen36-gfx906/files/gfx906_runtime`. The final deploy script SHA256 is
    `c8e8ef99ec39a0232f74a7bd0fe0efe0316c0e0678992a1c104eff3c05513c9a`.

    The release image tag is
    `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`.
    Docker Hub reports manifest digest
    `sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e`.
    Final deterministic export from the patched deploy path produced archive
    SHA256 `5316c3f6202fcb77987dabbf1e14e7369441ea127efed4f6def30259a09cfcb9`
    with OCI image manifest
    `sha256:7dadf367ec86fe2eb1dc22fb3af3002c3514514833b52329595a26e7a80ae247`
    and config
    `sha256:45decd88eb7c10c0408327438e07c2a655e45cc7534f8b662e5c4089a6b88568`.

    Portable performance at `MAX_MODEL_LEN=131072` with eight pre-measure
    warmups: dense 27B TP8 on `.20` reached strict-valid `69.514` backend TPS,
    c1_2000 `70.347`, and c1_10000 `66.069`; MoE 35B-A3B TP8 on `.30`
    reached strict-valid `94.907`, c1_2000 `97.028`, and c1_10000 `91.290`.
    MoE TP4 on `.30` has a release-time fixed-token result, c1_2000 `116.146`
    and c1_10000 `109.283`. The earlier strict-runaway caveat has a post-v0.2
    correction: the native TP4 profile passed `6/6` strict repeats across `.20`
    and `.30`, all with `finish_reason=stop` and `qwen_gate_valid=true`, with
    strict backend TPS from `113.196` to `115.995`. No code, Docker image, tag,
    model package, or runtime artifact changed.

    Deploy-source note: local Hugging Face snapshot resolution now validates
    every shard referenced by `model.safetensors.index.json` before selecting a
    snapshot path. Incomplete caches stay on the repo id, and `AUTO_STAGE_MODEL=1`
    uses retry plus post-download shard validation.
