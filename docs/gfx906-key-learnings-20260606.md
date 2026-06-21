# GFX906 Key Learnings - 2026-06-06

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

# gfx906 Key Learnings - 2026-06-06

This document captures the durable lessons from the Qwen3.6 / gfx906 campaign.
It is intentionally about decisions and evidence, not every run.

Related index:

- Source-level kernel and graph-runtime inventory:
  `gfx906_source_kernel_inventory_20260612.md`.

## Promotion Rules That Matter

- Promotion evidence must use optimized serving paths. `-O=0` is diagnostic
  only and must not be used for winner claims.
- Thinking-model correctness checks must be uncapped and evaluated after the
  closing thinking section. Capped `c1_128` hashes are invalid if the response
  ends before the model closes its reasoning stream.
- Throughput claims need backend metric evidence, not only client wall time.
  The main decode gate metric is backend decode TPS from vLLM request metrics.
- Near-tie throughput is not promotion evidence by itself. A variance-band run
  can still be promoted as a source milestone if it proves a reusable mechanism,
  composes with other patches, or removes a blocker for a credible follow-on
  optimization. It becomes a serving winner only after repeated standard-tier
  evidence clears the incumbent band.
- Profiles are valid only when the active profiling window overlaps the request
  and emits kernel rows. Runs with raw files but zero kernel CSV rows are
  attribution failures, not performance evidence.

## Current Winners And Milestones

- As of 2026-06-20,
  [qwen36-gfx906/README.md](../qwen36-gfx906/README.md) is the authoritative
  latest technical source for current ROCm7.2 dense/MoE status on main.
- Dense 27B TP8 clears the ai-info 10K gate in post-v0.1 main-branch validation.
  The active contract is `dense27b_tp8_fullbar_p2pon` on `.20` at
  `MAX_MODEL_LEN=131072` with eight pre-measure warmups: strict backend TPS
  `69.514`, `c1_2000` backend TPS `70.347`, and `c1_10000` backend TPS
  `66.069`; note `strict gate valid`.
- Qwen3.6 35B-A3B MoE TP8 establishes the strict-valid full-BAR/P2P-on bar.
  The active contract is `moe35b_tp8_fullbar_p2pon` on `.30` at
  `MAX_MODEL_LEN=131072` with eight pre-measure warmups: strict backend TPS
  `94.907`, `c1_2000` backend TPS `97.028`, and `c1_10000` backend TPS
  `91.290`; note `strict gate valid`.
- Qwen3.6 35B-A3B MoE TP4 remains capped-only until the strict runaway is
  resolved. The active TP4 lane is `moe35b_tp4_fullbar_p2pon` on `.30`:
  strict is `invalid/runaway`, `c1_2000` backend TPS is `116.146`, and
  `c1_10000` backend TPS is `109.283`; note `uncapped strict prompt did not
  stop after >60K tokens`.
- The same ROCm7.2 experimental release image covers both active contracts with
  model-specific env and overlays:
  `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`,
  Docker Hub manifest digest
  `sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e`.
  Docker Hub remains an evergreen artifact distribution channel; TPS claims
  should stay in GitHub Releases, repository docs, and benchmark artifacts.
  v0.1.0 remains the older published GitHub Release boundary; these are
  post-v0.1 main-branch validation notes until a separate release is published.
  GitHub Releases remain canonical for published claim boundaries.
- Platform remediation for the current full-BAR/P2P-on lane required official
  AMD VBIOS standardization, not modified BIOS images, plus amdgpu source
  patching. This is not a user instruction to flash cards; the public repo does
  not redistribute BIOS binaries or imply warranty or certification.
- Older dense source entries in this log remain useful historical source
  evidence, but they are superseded for current dense 27B status by the
  2026-06-20 ROCm7.2 Dense/MoE runner.
- The latest profile of the actual non-resident sidecar winner sharpened the
  dense target again. On `.30`, run
  `qwen36_27b_sidecar_winner_rocprof_h30_20260613_004845` emitted `806,396`
  filtered kernel rows; `sidecar_tree_ll_1x5120_kernel` alone was `13.767s`,
  `54.14%` of filtered kernel time, with `98,062` launches averaging about
  `140 us`. This makes sidecar collective launch/count cost the primary dense
  source target, not more RMS/MLP cleanup.
- A same-stack `.20` profile reproduced the sidecar-winner structure and
  narrowed the remaining target. Run
  `qwen36_27b_sidecar_winner_rocprof_h20_20260613_sidecar_winner_h20_profile`
  emitted `806,637` used rows and `23.719s` filtered kernel time. The sidecar
  kernel was still dominant at `12.483s`, `52.63%`, `98,059` launches, and
  `127 us` average; public `ncclDevKernel_Generic_4` was still visible at
  `2.014s`, `8.49%`, and `1,672` launches. Compared with `.30`
  (`140 us` sidecar average and `2.292s` public NCCL), `.20` is faster mainly
  in the collective path while LLGemm and SwiGLU remain nearly unchanged.
  Public-communication tracing showed the residual public NCCL calls are mostly
  prefill/profiling and graph-capture rows, while exact `1x5120` decode
  allreduces still route through the sidecar.
- Sidecar tail analysis now points away from a simple synchronized rank-arrival
  stall. The `.20` winner has all-GPU sidecar p50/p95/p99 duration of about
  `42/183/3051 us`; `.30` original is about `46/172/3575 us`; `.30` multi-row
  is about `46/179/3644 us`. Normal per-GPU launch spacing stays near
  `220 us`, while the p99 tail and rare long spikes are local and not cleanly
  aligned across ranks. Treat this as a collective primitive/count problem:
  reduce the hot `1x5120` launch count or change the sidecar primitive, rather
  than spending another cycle on external rank staggering or residual cleanup.
- Direct sidecar descriptor median wins are now too weak to trust without
  serving tiers. The `1792/1664/1664` count split beat the promoted
  `2048/1536/1536` split in direct graph replay on both `.20` and `.30`
  (`~0.028766 ms/call`), but full serving ladders regressed to `.20`
  `62.270/63.377/59.833` and `.30` `61.691/62.768/58.975` backend TPS.
  This keeps the current `.20` serving high-water on `2048/1536/1536` and
  reinforces that the next gate-sized work must reduce p99 tail or launch
  count, not only median descriptor work.
- Sidecar data-path prewarm is not the missing first-node fix. An opt-in
  scratch `1x5120` sidecar allreduce before graph capture fired correctly and
  preserved strict c1 correctness, but did not beat the winner band:
  `.20` public1664+prewarm `62.055`, `.20` no-bypass/default prewarm `63.046`,
  and `.30` no-bypass/default prewarm `62.326` backend TPS. Keep the existing
  handle preinit, but do not spend more cycles on first-use prewarm without a
  deeper boundary/count change.
- Rechecking `NCCL_NTHREADS` unset on the current non-resident sidecar winner
  did not promote. On `.20` it reached strict c1 `62.213`, `c1_2000 63.195`,
  and `c1_10000 59.685`; on `.30` it reached strict c1 `61.972`,
  `c1_2000 62.786`, and `c1_10000 59.137` backend TPS. This is a tie/noise
  result against the promoted `.20/.30` high-water, so keep `NCCL_NTHREADS=128`
  in the dense sidecar envelope.
- Explicit ROCm 7.2 RCCL override variables do not promote the current dense
  sidecar winner. Adding `RCCL_OVERRIDE_ALGO=Tree` and
  `RCCL_OVERRIDE_PROTO=LL` produced valid strict c1 smokes at `.20` `62.275`
  and `.30` `62.036` backend TPS, below the same-host high-water. Keep the
  existing `NCCL_ALGO=Tree` / `NCCL_PROTO=LL` envelope without the RCCL
  override variables.
- Correcting the target-selective RowParallel selector to recognize attention
  `LLMM1(5120x768)` allreduces is a source correctness fix, but selective
  routing is not a serving breakthrough. Revisited full ladders after the fix
  produced `.20` attention-only `57.793/58.527/55.468` and `.30` `mlp,attn`
  `55.509/56.187/53.417` backend TPS for strict c1 / c1_2000 / c1_10000.
  All-sidecar remains the dense high-water.
- Expanding the sidecar descriptor from three to four channels is closed. RCCL
  supports the four-channel continuous-byte split and the static variant is
  exact, but it is slower than the original three-channel primitive on both
  promotion hosts: `.20` `0.030045` vs `0.029670` ms/call and `.30`
  `0.029983` vs `0.029743` ms/call in direct graph replay.
- Bypassing the sidecar `RunWorkColl` wrapper is also closed as a promotion
  path. A throwaway direct-`Primitives` fixed-shape `1x5120` Tree/LL candidate
  was exact, but same-session controls showed only a tiny graph replay shift:
  `.20` `0.029666` vs `0.029680` ms/call and `.30` `0.029632` vs `0.029658`.
  The remaining dense gap is not generic wrapper overhead.
- Dense 27B prefill can be much higher under ai-infos-style batch settings. A
  same-host `.30` no-SP control reached about `832` prompt TPS for `4096+1`,
  but decode remained around `37.8` TPS for `4096+512`.
- Host `.50` is not currently qualified for ROCm 7.2 dense-winner comparisons.
  It has a small `gfx803` display GPU ahead of eight 32 GiB `gfx906` cards.
  ROCm 6.3 can initialize a selected MI50 there, but the ROCm 7.2 image fails
  HIP initialization with `hsa_init failed with 1000` / `hipErrorNoDevice` even
  under one-GPU and UUID-style masks. Treat `.50` as a ROCm 6.3 side lane until
  the host/topology issue is fixed; do not mix it into ROCm 7.2 dense promotion
  evidence.

## gfx906 Communication Lessons

- The bottleneck is not simply raw RCCL latency. Small collectives are fast in
  microbench, but serving performance depends on launch sequencing, graph
  capture, overlap, and whether reductions are serial consumers.
- Microbench wins do not automatically promote. Ring/LL and grouped collectives
  produced real primitive wins, but dense serving did not improve unless the
  model graph had legal independent callsites.
- Grouped PyNccl is a reusable primitive for adjacent independent reductions.
  It is not directly applicable to dense 27B RowParallel layers because each
  allreduce is immediately consumed by the next residual/RMSNorm boundary.
- This was confirmed on all three prepared hosts after fixing post-IR pass
  registration and forcing fresh compile caches. The current dense winner graph
  reports `129` collective ops per rank, `0` groupable adjacent pairs,
  `128` dependency blockers, and `128` intervening-use blockers on `.10`,
  `.20`, and `.30`. Treat graph-level grouping/coalescing as closed for the
  current lowered graph; further gains need per-collective latency reduction,
  producer/consumer overlap, or a deeper graph/model rewrite.
- The 35B MoE TP4 winner benefits from reducing communication count and forcing
  a stable low-latency NCCL shape. The dense model needs either fewer
  RowParallel boundaries or a lower-latency same-semantics collective.
- Dense RowParallel replay testing shows why this path is still worth source
  work. Replay-56 amortizes graph overhead and estimates roughly `6.25 ms/token`
  of allreduce/boundary component across Qwen3.6-27B's `64` text layers. The
  `45 -> 65 TPS` gap is about `6.84 ms/token`, so allreduce work is near the
  needed scale but likely needs consumer fusion or reduction-count reduction too.
- A full TP8 post-allreduce boundary replay for dense hidden size `5120`
  sharpened that estimate. At c1 decode shape `1x5120`, the direct MLP-down
  boundary costs `0.0893 ms` per call: `0.0594 ms` allreduce and `0.0325 ms`
  decomposed Qwen/Gemma residual RMSNorm consumer. Across the `64` direct
  MLP-down boundaries this gives a bounded first source target of about
  `2.08 ms/token`, assuming the consumer launch/memory pass can be removed
  cleanly while keeping RCCL/PyNccl for the collective.
- Sidecar channel partitioning is not a portable breakthrough. The opt-in
  runtime split knobs were exact, but `.20` regressed from default
  `0.07249 ms/call` to `0.0865-0.0893`, while `.30` improved from
  `0.08481` to as low as `0.07743`. Keep the knobs for diagnostics, but do not
  promote a host-specific partition without a serving-tier cross-host win.

## Sequence Parallelism Findings

- Hidden-shard reduce-scatter is rejected for c1 decode. At `1x5120` TP8 it was
  materially slower than allreduce on both `.20` and `.30`.
- Token-sharded sequence parallelism matches upstream vLLM SP semantics and is
  a real prefill lower-bound win in microbench. At `64-256x5120`, it improved
  boundary time by roughly `35-59%` depending on host and row count.
- Raw upstream SP is rejected as a serving patch today. Same-host `.30` results:
  - no SP, `4096+512`: `37.804` backend decode TPS
  - SP threshold `64`: `13.373` backend decode TPS
  - SP threshold `4096`: `12.969` backend decode TPS
  - SP threshold `4096` plus requested captures `[1,2,4,8,16]`: `13.407` TPS
- vLLM overrides explicit SP cudagraph capture sizes back to `[8,16]`, so c1
  decode loses the `num_tokens=1` graph path.
- A threshold-only ROCm/gfx906 SP default would be harmful. SP must either
  preserve single-token graph capture or gain a ROCm fused GEMM-collective path
  before promotion.
- Static SP is now a side lane only. Its job is to find a prefill milestone that
  does not regress the normal dense decode ladder.
- Static compile without SP is now source-viable but not a decode promotion. The
  `static_compile_contig` overlay fixed the gfx906/ROCm stride mismatch by
  making small non-contiguous two-dimensional static inputs contiguous before
  dispatch. With that patch, `compile_sizes=[1]` passed strict streaming c1 on
  `.20` at `45.623` backend TPS, and `compile_sizes=[1,2,4,8]` passed strict
  streaming c1 on `.30` at `43.484` backend TPS. That proves the cold-start
  failures were patchable, but the throughput stays baseline-like or slower.
  The later full ladder on `.30` confirmed the reject: strict c1 `43.365`,
  c1_2000 `44.312`, and c1_10000 `42.494` backend TPS.

## Fused Collective Support

- The existing vLLM fused follow-up path is `AsyncTPPass`.
- In the tested image, `AsyncTPPass` is imported only under
  `current_platform.is_cuda()` and uses `torch.distributed._symmetric_memory`
  plus `torch.ops.symm_mem.*`.
- The existing ROCm `quickreduce` substrate is not a gfx906 shortcut. A direct
  TP4 probe that bypassed only the Python gfx94/gfx95 guard failed inside the
  HIP IPC / raw buffer-load path with `HIP error 17` invalid device pointer and
  rank SIGSEGV.
- A standalone HIP copy/stage/sum allreduce MEP is correct but not useful for
  the dense gate. On `.30`, TP4 `1x5120` took `299.16 us/call`, TP8 `1x5120`
  took `1486.52 us/call`, and TP8 `431x5120` took `28396.60 us/call`.
  Direct-peer kernels faulted with a GPU memory access error. This closes the
  naive peer-copy/direct-peer source branch; any future collective backend must
  use a different ROCm/gfx906 design than raw cross-GPU loads or staged peer
  copies.
- The existing vLLM custom allreduce substrate is also not a gfx906 TP8 PCIe
  shortcut. On `.20` with all eight 32 GB gfx906 cards:
  - forcing `use_custom_allreduce=True` first failed worker initialization while
    opening peer IPC handles, with `hipErrorInvalidDevicePointer`
  - adding an init fallback avoided the immediate crash, but serving still hung
    after the first `431`-token prefill batch
  - guarding `should_custom_ar()` was insufficient because reporting platform
    custom-allreduce support changes ROCm communicator setup before runtime
  - the safe hotfix now requires both `VLLM_GFX906_PCIE_CUSTOM_AR=1` and
    `VLLM_GFX906_PCIE_CUSTOM_AR_FORCE_RUNTIME=1` before gfx906 reports custom
    allreduce support; the second flag is unsafe and diagnostic-only
  - early short-window "hang" calls were too aggressive for source-overlay
    lanes. After adding import-time scrubbing of `VLLM_CUSTOM_ALLREDUCE_ALGO`,
    a bounded `32`-token smoke with the diagnostic env/mounts took `208.79s`
    wall time because of cold first-request work, but completed with backend
    decode `48.48 TPS`. Treat cold readiness / compile / graph-capture delay
    separately from steady decode and from hard failures.
  - the full standard dense ladder for the same guarded diagnostic/no-runtime
    path also completed:
    `runs/qwen36_27b_decode_tiers_dense_fullold_pcie_custom_ar_guard_full_c1_h20_host20_20260606_190347`.
    Results were strict c1 valid at `46.29` backend TPS, c1_2000 at `47.07`
    backend TPS, and c1_10000 at `45.05` backend TPS. This is baseline-like
    and not a promotion, but it invalidates the earlier short-window "hung"
    classification for guarded/no-runtime diagnostic configs.
  - the exact platform-guard impatience revisit also completed under the same
    long-patience rule:
    `runs/qwen36_27b_decode_tiers_dense_fullold_pcie_custom_ar_platform_guard_revisit_c1_h20_host20_20260606_193555`.
    Startup plus first uncapped smoke took `295.98s` wall time, but the smoke
    was valid and produced backend decode `46.68` TPS. The full ladder then
    produced c1_2000 `47.41` backend TPS and c1_10000 `45.22` backend TPS.
    This is valid-but-not-promoted evidence: reject it on throughput, not on
    cold-start behavior.
  Conclusion: a production gfx906 small-message collective cannot be a simple
  Python guard bypass of vLLM's CUDA/MI300 custom AR path. It needs a separate
  ROCm/gfx906 design with its own peer-access proof, timeout-safe init, and
  model-output validation.
- The existing CUDA/FlashInfer `AllReduceFusionPass` matches the right graph
  pattern, `allreduce -> residual add -> RMSNorm`, but its backend is gated to
  CUDA/Hopper/Blackwell plus FlashInfer. For gfx906, this is a source template,
  not a usable runtime switch.
- gfx906 support therefore requires real source work:
  - for dense decode, implement or overlap ROCm-safe RowParallel
    `GEMM + AllReduce` / allreduce-consumer paths
  - for SP-prefill, implement ROCm-safe fused `GEMM + ReduceScatter` and
    `AllGather + GEMM`
  - guard by platform, dtype, TP size, and static shape
  - preserve c1 decode graph capture
  - rerun strict dense and MoE ladders before any promotion
- The Gemma/Qwen fused add+RMSNorm standalone MEP was useful but rejected for
  serving after an active dense ladder regression. Do not promote it without a
  new source change and full ladder repeat.
- The dense 27B lowered compile graph gives a precise allreduce-consumer split:
  `129` allreduces per rank graph, consisting of `64` attention allreduces that
  first write through `self_attention_output[...] = all_reduce`, `64` direct
  MLP-down allreduces consumed by decomposed residual/Gemma RMSNorm, and `1`
  embedding allreduce. The first source target is the direct MLP-down boundary;
  the attention path needs separate mutation-aware handling.
- The post-grad high-level matcher is not the integration hook on the current
  image. It runs, but it sees `0` high-level fused patterns and `0` lowered
  `torch.ops.vllm.all_reduce` nodes. Use the emitted compile graph/lower-level
  backend path for implementation evidence.
- Consumer-only post-allreduce fusion is most credible as a dense decode
  milestone. For prefill rows `64-432x5120`, allreduce dominates the same
  boundary replay, so prompt-processing gains need SP/CP or GEMM-collective
  work rather than only a better RMSNorm consumer.
- The post-IR fused consumer is a valid source milestone in lower-bound replay.
  In `runs/pynccl_post_ar_rms_graph_bench_host20_20260606_201135`, TP8
  `allreduce -> gfx906_gemma_add_rms_norm_fp32_residual` beat decomposed
  `allreduce -> residual add -> Gemma RMSNorm` at all tested shapes:
  `1x5120` improved by `0.0229 ms/call`, `64x5120` by `0.0526 ms/call`,
  `128x5120` by `0.0719 ms/call`, and `431x5120` by `0.2168 ms/call`, with
  zero output/residual error. The c1 direct-MLP-only bound is about
  `1.47 ms/token`, so this is a meaningful milestone but still not a solo
  `65` TPS gate-clear path.
- The first post-IR serving integration did not activate. The original
  `MAX_TOKENS=16` guard skipped compile range `(1, 2048)`. The corrected
  `MAX_TOKENS=2048` revisit compiled the range but matched `0` boundaries and
  stayed baseline-like (`47.374` c1_2000 backend TPS, `45.158` c1_10000). The
  failure is the brittle lowered matcher, not proof against the fused kernel.
  The matcher must follow downstream structure, not `down_proj` string
  preservation in lowered GEMM args.
- The later pre-split lowered matcher did activate, but raw FX substitution is
  rejected as a serving integration method. Image
  `qwen36-gfx906-postir-direct-mlp-ar-rms:20260606-presplit14` selected `1/64`
  direct MLP boundaries with fresh compile cache, yet strict streaming c1
  immediately produced `<think>` followed by repeated `!`. Repeating with
  tensor metadata dropped produced the same corruption. This rejects stale
  compile cache, custom-op math, residual-output replacement, and fake-tensor
  metadata as the likely primary causes. Future integration must use vLLM's
  native pattern-replacement contract or a lower-level backend path, not broad
  hand-spliced lowered-node replacement.
- The presplit corruption was traced to unsafe graph-wide cleanup, not to the
  direct MLP boundary matcher itself. A no-op selected-match control with global
  `graph.eliminate_dead_code()` still produced repeated `!`; disabling that
  global DCE restored coherent strict streaming output.
- The safer presplit16 source path adds a true dry-run mode plus bounded local
  cleanup that only erases dead pure nodes in the matched residual/RMS chain.
  This path is semantically viable at full dense coverage:
  `runs/qwen36_27b_decode_tiers_dense_fullold_postir_presplit16_fused_both_cleanup_limit64_fulltiers_h20_host20_20260607_012512`
  selected `64/64` direct MLP boundaries on every TP rank and locally erased
  `768` dead pure nodes per rank.
- Full-coverage direct MLP allreduce-consumer cleanup is a source milestone, not
  a gate clear. The same run produced strict uncapped c1 valid output at
  `46.003` backend TPS, `c1_2000` at `47.103` backend TPS, and `c1_10000` at
  `44.853` backend TPS. That keeps the dense 27B gate open and shows the main
  remaining limiter is not only post-allreduce RMS/residual compute; the
  communication schedule, launch sequencing, or RowParallel collective path
  still has to move.
- A valid targeted rocprof sample of that full-coverage presplit16 path confirms
  the pivot. Run
  `runs/qwen36_27b_presplit16_full64_decode_rocprof_host20_20260607_014244`
  emitted `998,526` used kernel rows. Filtered kernel time was dominated by
  NCCL `54.17%` and `LLGemm1_kernel` `27.51%`; the custom
  `gfx906_gemma_add_rms_norm_fp32_residual_kernel` was only `1.60%`. This makes
  another consumer-only RMS rewrite a low-probability gate-clear path.
- The presplit17 attention-capable matcher completed the bounded semantic
  milestone and closed the consumer-only branch. Image
  `qwen36-gfx906-postir-direct-mlp-ar-rms:20260607-presplit17-attn` with
  `VLLM_GFX906_POST_IR_AR_RMS_TARGETS=both` selected `128/128` lowered
  boundaries per TP rank (`64` direct MLP plus `64` attention), locally erased
  `1536` dead pure nodes per rank, and produced valid strict uncapped c1 output
  at `44.958` backend TPS. The matcher is semantically safe at full coverage,
  but it does not move the dense decode gate.
- The LLMM1 helper-routing lane has a narrow useful readout. Current routing
  uses `ops.LLMM1(weight, x_view, 2)` for `n == 1`, `k <= 8192`, and
  `m % 4 == 0`; synthetic gfx906 benches confirm `rows_per_block=2` is best for
  the large dense decode GEMV shapes. `rows_per_block=1` is not a real in-tree
  variant and falls back to the rpb4 path, producing only baseline-like strict
  c1 results on `.20` (`45.806` TPS) and `.30` (`45.396` TPS). Do not promote
  helper routing without runtime shape evidence showing a meaningful small
  shape population.
- Runtime shape census on the dense 27B full-old four-hotfix path confirms that
  the dominant decode GEMV is `n=1,m=31040,k=5120,float16`, with `4096`
  calls/rank in the c1 smoke. Exact-shape microbenching on `.20` and `.30`
  keeps `LLMM1_rpb2` as the right path for that shape: about `381.6 us`,
  versus `396 us` for rpb4, `405 us` for rpb8, `678 us` for rpb16, and
  `930 us` for Triton. Small-shape autorouting and the `m5120,k2176` torch
  fallback were both valid but baseline-like in serving (`46.516` and
  `46.275` backend TPS), so they are not decode promotions.
- Two additional LLMM1 source probes reinforce that conclusion. A gfx906
  `v_dot2_f32_f16` replacement for the current half2 multiply-add path compiled
  but regressed the dominant shape (`387.840 us` vs the `~381-382 us` rpb2
  control band). A strided-K loop kernel with rpb2 and `256/320/384/512`
  threads also regressed the dominant shape
  (`408.961/399.841/397.921/393.601 us`). Do not revisit these LLMM1
  decompositions without new profile evidence.
- Direct PyTorch/rocBLAS GEMV is not a hidden win for the `n=1` dense decode
  wall. On `.20`, `torch.mv`/`torch.addmv` took about `3.62-3.63 ms` for
  `n1,m31040,k5120`, while `LLMM1_rpb2` took about `0.381 ms`. `torch_linear`
  and `torch.matmul(x, W.T)` were about `1.76 ms`. Keep `LLMM1_rpb2` unless a
  new custom kernel beats it on exact measured shapes.
- PyNccl output allocation is not the dense gate limiter. A guarded rotating
  output-buffer cache improved the TP8 `1x5120` graph lower bound only from
  `0.049899 ms` to `0.049225 ms`, and the full dense winner envelope regressed
  slightly at `c1_10000` (`45.182` backend decode TPS versus incumbent
  `45.476`). Keep the source effort on collective structure rather than
  allocator reuse.
- `wvSplitK` was not safe as a blind dispatcher toggle, but the source issue
  is now better understood. Upstream excludes gfx906 from the GFX9 `wvSplitK`
  path because its half-dot macro uses `v_dot2c_f32_f16`, which the gfx906
  assembler rejects. A gfx906-specific `v_dot2_f32_f16` replacement plus raw
  half-pair operands and a corrected lane reduction gives valid FP16 output.
  On `.20`, the patched `qwen36-gfx906-wvsplitk-env:20260607` image produced
  target-shape microbench wins for `n=2..4`, for example `n2,m5120,k2176`
  `57.599 us` vs torch `125.759 us` and Triton `142.719 us`, and
  `n4,m4352,k5120` `83.199 us` vs torch `381.117 us` and Triton
  `234.398 us`. This is a meaningful small-batch/concurrency milestone, not a
  single-request c1 decode gate clearance, because the dense gate is still
  dominated by `n=1` LLMM1 and RowParallel collectives.
- Naive RowParallel output-column chunking is rejected. On `.20`, the
  pipelined RowParallel MEP showed the full `LLMM1_rpb2` route at about
  `0.045 ms` for the attention `5120x640` shard and `0.087 ms` for the MLP
  `5120x3696` shard, while two-way chunking was already slower and
  stream-pipelined chunks were several times slower. Do not create more
  per-layer allreduce launches to chase overlap; the source path must fuse or
  reduce the original collective boundary.
- A valid skip-TP-allreduce dense 27B profile on `.30`
  (`runs/qwen36_27b_skip_tp_allreduce_upper_rocprof_host30_20260607_052545`)
  reduced NCCL from `28.875s / 67491` calls to `3.890s / 519` calls versus the
  normal full-old profile. That is an `86.5%` NCCL time reduction and removes
  about `66972` collective calls in the `17408+512` profile target. The
  remaining wall is no longer mostly RCCL: attention is `9.316s`, hipBLAS/GEMM
  other is `11.027s`, and LLGemm is `4.458s`. This confirms that RowParallel
  collectives are necessary to attack, but not sufficient alone for the dense
  65 TPS gate.
- Explicit balanced `RCCL_TREES` is the current dense decode milestone. The
  same-day `.20` no-tree control measured `57.223/58.107/55.111` backend TPS on
  `c1_128/c1_2000/c1_10000`; the balanced tree measured
  `61.252/62.236/58.780`, a `+7.04%/+7.11%/+6.66%` decode gain. It still does
  not clear the `65 TPS` sustained gate, but it proves the local RCCL Tree/LL
  chain topology was leaving meaningful decode performance on the table.
- The valid balanced-tree `rocprofv2` c1_768 profile on `.20`
  (`runs/qwen36_27b_c1_rocprofv2_balanced_tree_h20_20260612_030409`) still
  shows NCCL as the dominant bucket: `12.619s`, `57.86%`, `99,658` calls. The
  next buckets are `LLGemm1_kernel` (`3.838s`, `17.60%`) and native interleaved
  SwiGLU (`2.837s`, `13.01%`). Because those buckets are nearly unchanged from
  the prior current-winner profile, use clean serving A/B for the topology
  promotion and use the profile only to select the next residual target.

## Kernel And Memory-Access Lessons

- We should prefer contiguous row-major tensor layouts and shape-specific paths.
  The microbenches use contiguous tensors and expose clear row-count effects.
- We cannot assume cross-GPU cache coherency for correctness. Cross-rank
  visibility must be established by collectives, stream/event ordering, or
  explicit synchronization.
- Coalesced access should be verified per kernel with profiles or source
  inspection. It is not enough to know the input tensor is contiguous; the
  kernel's thread-to-element mapping must also be coalesced.
- gfx906 is bandwidth-rich but sensitive to launch overhead and small-message
  communication. Optimizations should reduce launches, avoid extra round trips,
  and preserve graph capture.

## Profiling Lessons

- Rocprof active-window timing must be anchored to the profiler's actual delay
  timestamp, not the host launch epoch.
- `rocprofv3` unfiltered traces can fail above a token threshold because the
  exporter produces raw files but no usable kernel CSVs.
- Include-regex profiling can salvage some attribution lanes, but filtered
  exports are not always complete. Treat them as targeted evidence, not full
  critical-path profiles.
- Profiling overhead can destroy throughput numbers. Use profiled runs for
  attribution and clean O3 runs for promotion.
- Do not reject a source-overlay config only because readiness or the first
  post-prefill decode marker is late. Reject only on hard process failure,
  traceback, correctness failure, or a documented long no-progress window with
  unchanged logs and no GPU/compile activity. Some valid configs perform
  expensive one-time work that can reduce later runtime work.

## Hardware And Lane Hygiene

- Detect hardware by measured VRAM and stable device identifiers, not brittle
  product-name strings.
- The intended deploy target for the public MoE package is `4x MI50 32GB`
  gfx906.
- `.10` GPU1 stale VRAM accounting was cleared by reboot on 2026-06-06; all
  eight cards returned to `0%` VRAM after the host came back.
- `.30` root-fill cleanup was verified on 2026-06-12 after stale Docker/build
  artifacts were cleared: `/` had `305 GiB` free, `/usr/share/ollama` had
  `510 GiB` free, and all eight GPUs were idle except driver allocations.
- Avoid leaving profiler loops, stale containers, or old isolated Docker roots
  running. They pollute performance measurements and lane availability.

## Next Technical Direction

1. Protect the 35B MoE TP4 winner and the dense 27B four-hotfix milestone.
2. Do not promote raw SP on ROCm.
3. For dense 27B, prioritize structural RowParallel work:
   - graph-captured RowParallel fused/overlapped `GEMM + AllReduce`, or
   - a lower-latency gfx906 small-message collective for the validated hidden
     width reductions, or
   - a true reduction-count change in the dense graph.
   Avoid chunked RowParallel overlap that increases allreduce launch count.
   Pair any major communication improvement with attention or dense GEMM/LLGemm
   work, because the skip-allreduce upper bound still does not clear `65` TPS.
4. For MoE, continue testing source changes that reduce communication count
   without changing semantics.
5. For public claims, separate prefill, decode, and end-to-end total-token TPS.
   They are different performance surfaces.
6. Dense 27B attention softmax segment count is not currently a decode-gate
   lever. Revisited `REF=FALSE`, source seg32, and source seg128 all stayed in
   the existing `44-45 TPS` c1_10000 class with strict streaming smoke, so the
   source effort should move to decode tiling, communication boundaries, GEMM,
   or launch-count reduction.
7. Dense 27B attention decode tile size is also closed for the current profile.
   Source tile4, tile16, and tile32 all regressed c1_10000 to about
   `28-36 TPS`; keep the incumbent tile8 unless a new profile identifies a
   different attention shape.
8. Lower TP degree is not automatically a dense decode win. A clean TP4
   full-old rerun on `.30` reached only `35.705` backend TPS on `c1_10000`,
   and a TP2 16K-context rerun with the hybrid-KV hotfix reached only `17.356`
   backend TPS. The reduced collective fanout does not compensate for the
   extra per-rank dense compute.
9. vLLM's hybrid KV-cache layout guard can falsely reject low-TP `TRITON_ATTN`
   profiles when the profiled cache has exactly two GPU blocks. For Triton,
   shape `[2, 2, ...]` can be the valid `(num_blocks, 2, ...)` layout, not an
   ambiguous FlashAttention layout. The opt-in
   `gpu_model_runner_hybrid_kv_20260607` hotfix fixes startup enough to produce
   real TP2/PP2xTP2 evidence.
10. PP2xTP2 is not a useful dense c1 decode topology on the current stack. Once
    startup was fixed, the strict smoke still timed out after `300s` with only
    the injected `<think>` visible and `0` backend generation tokens. Pipeline
    parallelism adds a prompt/P2P schedule cost that is incompatible with this
    single-request gate.
11. Omitting the vLLM `-O` CLI flag is not a dense decode win on the current
    image. The run still used piecewise compile ranges and scored `45.067`
    backend TPS on `c1_10000`, matching the existing TP8 milestone band.
12. V2 runner startup for Qwen3.6 hybrid attention plus Mamba/GDN can be fixed
    with a targeted `attn_utils.py` cache-reshape overlay that unwraps
    `UniformTypeKVCacheSpecs` and handles `MambaSpec` pages using the older
    runner's state-tensor layout. This is a useful source milestone, but the
    first valid TP8 smoke was only `43.715` backend TPS, so V2 is not a dense
    decode promotion by itself.
13. Full direct+attention RowParallel consumer matching is semantically viable
    but not sufficient for the dense gate. The pre-split17 post-IR source lane
    selected all `128/128` candidate boundaries and passed strict thinking
    smoke, but the full ladder stayed at `44.545` backend TPS on `c1_10000`.
14. Decode context parallelism is a real source target, but the current
    `TRITON_ATTN` backend cannot be enabled by configuration alone. It fails
    compatibility because `TritonAttentionImpl` does not return decode LSE.
    A correct TRITON DCP implementation must add LSE output and reproduce the
    FlashAttention context/query split plus LSE merge semantics, not merely
    bypass the assert.
15. The current ROCm `FLASH_ATTN` path in the deploy image cannot serve
    Qwen3.6-27B paged decode attention. Python compatibility shims can get past
    `fa_version=None`, `out=`, and signature mismatches, but both plain
    `FLASH_ATTN` and `FLASH_ATTN` + DCP2 ultimately fail inside AMD Triton FA2
    with `block_table / paged attention is not supported`. DCP cannot be
    promoted through this backend unless a paged-attention-capable ROCm
    FlashAttention kernel is supplied.
16. SP-style reduce-scatter is shape-sensitive on gfx906. Corrected replay64
    boundary evidence shows it regresses dense c1 decode rows `1-8`, starts to
    become useful around rows `16-32`, and becomes a strong
    prompt-processing/concurrency candidate at rows `64+` with `32-51%`
    graph-boundary gains. Treat SP as a prefill/concurrency source lane, not
    the single-request decode gate path.
17. Ring/LL is not a dense decode promotion even though replay64 RowParallel
    MEPs showed a small c1 lower-bound edge over Tree/LL. The exact full-old
    serving ladder on `.30` reached only `44.325` backend TPS at `c1_10000`,
    so Tree/LL remains the dense communication default.
18. Tree/LL128 is also not a dense decode promotion. It produced the best
    `.30` RowParallel replay64 microbench numbers in the cheap NCCL sweep, but
    the full-old serving ladder reached only `44.315` backend TPS at
    `c1_10000`. Microbench wins on isolated RowParallel boundaries must still
    clear the strict serving ladder before they influence defaults.
19. Dense TP8 performs one full-vocab logits all-gather per generated token per
    rank on the full-old path. The all-gather trace on `.20`
    (`qwen36_27b_decode_tiers_dense_fullold_allgather_trace_c2000_h20_20260607_host20_20260607_084111`)
    showed `[1,31040]` fp16 shard gathers from
    `LogitsProcessor._gather_logits` at exactly `1.0` call/token/rank. This is
    a real semantic reduction target for greedy decode, but prior upper bounds
    prove it is not enough alone to clear the `65 TPS` dense gate.
20. The logits-gather replacement ceiling is modest but measurable. The `.20`
    `pynccl_logits_gather_bench_host20_20260607_090823` replay64 MEP measured
    graph full logits all-gather at `0.219404 ms/call` and graph local-argmax
    plus tiny pair gather at `0.075929 ms/call`, so the exact greedy cleanup is
    worth a milestone but saves only about `0.143 ms/token` on the gate shape.
21. The exact greedy logits-gather replacement now works in real serving, but
    it is not a dense decode promotion. The source overlay
    `greedy_top1_mintokens_20260607` had to expose `get_top_tokens()` through
    `Qwen3_5ForConditionalGeneration`, support `LogitBiasLogitsProcessor`, and
    cast local min-token `-inf` writes to the FP16 logits dtype. The corrected
    `.30` full-old ladder
    `qwen36_27b_decode_tiers_dense_fullold_greedy_top1_mintokens_logitbias_dtypefix_h30_20260607_host30_20260607_093529`
    produced a valid strict smoke at `45.233` backend TPS, `c1_2000` at
    `46.395`, and `c1_10000` at `44.375`. This confirms the semantic reduction
    path, but it stays in the existing dense milestone band.
22. TRITON decode-context parallelism can be made correct enough to serve on
    gfx906, but it is not a single-request decode gate path. The overlay
    `triton_dcp_lse_work_20260607` added decode LSE output, DCP local context
    lengths, non-causal context attention, Flash-style LSE correction/merge,
    and later a TRITON decode suffix kernel. It passed a strict uncapped smoke
    on `.30`
    `qwen36_27b_decode_tiers_dense_triton_dcp2_lse_bootstrap_h30_20260607_host30_20260607_100000`
    with coherent output and `qwen_gate_valid=true`, but only reached `8.601`
    backend TPS. Replacing the torch suffix bootstrap with a TRITON suffix
    kernel improved a capped diagnostic on `.20` from `8.533` to `9.718`
    backend TPS, still far below the `44-45 TPS` incumbent. Treat DCP/TRITON as
    a source milestone and possible prefill/concurrency reference, not a dense
    c1 decode promotion.
23. Exact greedy logits cleanup does not stack with omitted `-O`. The `.30`
    full-old ladder
    `qwen36_27b_decode_tiers_dense_fullold_greedy_top1_no_o_combo_h30_20260607_102632_host30_20260607_102632`
    reached `45.922` backend TPS on strict smoke, `46.635` on `c1_2000`, and
    `44.718` on `c1_10000`, so no-`O` remains closed for dense decode.
24. Presplit17 direct+attention consumer fusion also does not stack with exact
    greedy logits cleanup. A `.20` retry was invalid because stale ROCm holders
    left GPUs `0` and `2` pinned after a container exit, but the clean `.30`
    retest
    `qwen36_27b_decode_tiers_dense_fullold_presplit17_greedy_top1_timeout1800_h30_20260607_105154_host30_20260607_105154`
    passed strict streaming smoke at `45.873`, then scored `46.111` on
    `c1_2000` and `44.219` on `c1_10000`. Local RowParallel consumer cleanup
    plus logits-gather cleanup stays in the current milestone band.
25. Some source candidates need a long first-request compile window. The
    presplit17+greedy clean `.30` run emitted only `<think>` for several
    minutes, while each worker used roughly `60-70%` CPU, then resumed coherent
    streaming and completed the ladder. Treat CPU-active first-execute waits as
    compile evidence, not immediate hangs; reserve rejection for a concrete
    timeout, crash, or stalled process state.
26. Smaller static serving envelopes do not explain the dense decode gap. The
    `.30` full-old `MAX_MODEL_LEN=16384`, `MAX_NUM_SEQS=4`,
    `MAX_NUM_BATCHED_TOKENS=2048`, prefix-cache-off ladder
    `qwen36_27b_decode_tiers_dense_fullold_maxlen16k_static_envelope_h30_20260607_110910_host30_20260607_110910`
    passed strict streaming smoke at `46.161` backend TPS, then reached
    `47.670` on `c1_2000` and `45.476` on `c1_10000`. This closes cache
    envelope reduction as a single-request decode promotion path.
27. Exact-shape RCCL primitive wins are not sufficient by themselves. The `.30`
    TP8 `1x5120` fp16 HIP graph-replay sweep
    `pynccl_ar_env_sweep_host30_20260607_112945` found Ring/LL with
    `NCCL_NTHREADS` unset at `0.043364 ms/call`, about `10.6%` faster than the
    incumbent Tree/LL/one-channel/128-thread case at `0.048504 ms/call`.
    Serving validation
    `qwen36_27b_decode_tiers_dense_fullold_ring_ll_unset_microbench_candidate_h30_host30_20260607_113623`
    still landed at `46.576` strict, `47.085` on `c1_2000`, and `44.881` on
    `c1_10000`. The tied cleared-algo/proto one-channel candidate
    `qwen36_27b_decode_tiers_dense_fullold_cleared_ch1_p2poff_microbench_candidate_h30_host30_20260607_114750`
    reached `46.981` strict, `47.494` on `c1_2000`, and `45.223` on
    `c1_10000`. Treat standalone allreduce microbench wins as necessary
    evidence, not promotion evidence; the next communication work must alter
    graph-level scheduling, legal boundary count, or the collective substrate.
28. Standalone fused dense SwiGLU GEMV does not beat the tuned LLMM1 route. The
    `.30` MEP `microbenches/gfx906_swiglu_gemv_bench_20260607.py` tested the
    dominant dense decode MLP shape `x[1,5120] @ W[31040,5120].T -> silu*mul`.
    Baseline LLMM1 rpb2 plus `silu_and_mul` was `385.120 us` median. The best
    fused pair-wave t512 kernel was `415.680 us`, and the two-pair wave t512
    kernel was `420.481 us`. A tiny deterministic debug case matched the
    activation/layout contract; the full random-shape `32` max-abs drift is the
    same long-reduction scale seen between LLMM1 and torch. Close standalone
    consumer fusion as a promotion path. Only revisit if modifying LLMM1 itself
    so the existing tuned reduction structure is preserved and the epilogue
    writes the activated half directly.
29. Even preserving LLMM1's gfx906 `v_dot2` reduction and interleaving gate/up
    rows is not enough for dense decode promotion. The follow-up SwiGLU GEMV
    MEP added an LLMM1-style fused epilogue and then an interleaved gate/up
    weight layout so each block reads adjacent row pairs. The long `.30`
    two-method run measured incumbent LLMM1 rpb2 plus `silu_and_mul` at
    `388.800 us` and the fused interleaved LLMM1 epilogue at `389.120 us`.
    This is a useful source lesson: Qwen gate/up weight-layout specialization
    can tie the current path, but it does not create gate-clearing headroom.
    Keep the main lane on attention/GEMM wall reduction or true RowParallel
    collective structure.
30. Removing the GDN piecewise split is valid but insufficient. The `.30`
    dense full-old ladder with
    `--compilation-config '{"splitting_ops":[]}'`
    (`qwen36_27b_decode_tiers_dense_fullold_splitting_ops_empty_fullgraph_ladder_h30_host30_20260607_123645`)
    reached readiness after a patient compile/warmup, streamed coherent
    parser-split thinking output, and scored `46.903` strict, `47.645` on
    `c1_2000`, and `45.297` on `c1_10000`. This keeps full-graph GDN in the
    milestone bucket, not the promotion bucket. Further GDN work needs
    source-level overlap/fusion or kernel changes, not only a compile-boundary
    toggle.
31. GDN projection dual-streaming is a valid source experiment but a decode
    regression. Storing `torch.cuda.Stream` / `torch.cuda.Event` directly on
    the dense GDN module made vLLM's compile-cache artifact non-serializable;
    moving that state into a process-local global map keyed by layer prefix
    fixed startup. The fixed `.30` smoke
    `qwen36_27b_decode_tiers_dense_fullold_qwen35_gdn_dualstream_statefix_smoke_h30_host30_20260607_125516`
    was semantically valid but only `42.072` backend TPS. Do not pursue
    separate-stream overlap for the two tiny decode-side GDN input projections
    unless a later profile shows enough larger independent work to amortize the
    event/stream overhead.
32. Dense GDN output zeroing is not the 65 TPS lever. Replacing
    `Qwen3_5GatedDeltaNet.forward` `core_attn_out = torch.zeros(...)` with
    `torch.empty(...)` was semantically safe in the strict streaming ladder, but
    `qwen36_27b_decode_tiers_dense_fullold_qwen35_gdn_empty_core_ladder_h30_host30_20260607_131319`
    still scored only `46.965` strict, `47.109` on `c1_2000`, and `45.341` on
    `c1_10000`. The sustained dense gap is not explained by this per-layer
    buffer fill.
33. The decode-focused profile re-centers the dense 27B work on ordinary TP
    communication plus LLMM1/GEMV. The valid `.30` rocprof run
    `dense27b_decodefocus_rocprof_fullold_h30_20260607_132702` used a
    `512+768` request on the full-old TP8 stack and emitted `845031` kernel
    rows. Filtered kernel time was NCCL `12.176s / 56.11% / 99760` calls and
    `LLGemm1_kernel` `6.639s / 30.59% / 233506` calls. GDN/linear-attention
    kernels together were only `0.424s / 1.95%`. This closes simple GDN wrapper
    cleanup as a likely gate-clear path and makes collective schedule/substrate
    work or exact-shape LLMM1/GEMV improvement the primary branch.
34. RCCL custom ring order is not the missing dense decode knob on this
    four-root-complex MI50 host. The `.30` topology is four GPU pairs
    (`0/1`, `2/3`, `4/5`, `6/7`), so `NCCL_RINGS` orders were tested against
    replay64 c1 RowParallel shapes in
    `runs/rccl_custom_rings_20260607_134332`. Bus order, pair-flip,
    root-interleave, root-pairs, and reverse all completed correctly, but none
    improved both `1x640x5120` and `1x3696x5120`. The best single attention
    boundary was only `0.075851 ms` and came with an MLP regression; the best
    MLP boundary was only mixed/tied. Close topology-order tuning unless the
    RCCL stack or physical GPU ordering changes.
35. Dense `PP2 x TP4` is a valid cold-start patience lesson but not a decode
    promotion. The `.30` rerun
    `qwen36_27b_decode_tiers_dense_fullold_pp2_tp4_ladder_h30_20260607_host30_20260607_135649`
    waited through the long first-request path, streamed coherent parser-split
    thinking output, and passed strict `c1_128`. Sustained decode was only
    `32.693` strict, `32.983` on `c1_2000`, and `31.193` on `c1_10000`.
    Logs showed unbatched pipeline P2P send/recv communicator setup and steady
    generation around `29-33` tokens/s. For the single-request gate, the
    pipeline-stage overhead outweighs the smaller TP4 collectives.
36. The dense max-length envelope win is not monotonic toward the smallest
    context that fits the request. The `.30` `max_model_len=12288` rerun
    `qwen36_27b_decode_tiers_dense_fullold_maxlen12288_envelope_h30_20260607_host30_20260607_141543`
    passed strict smoke but scored `45.106` strict, `46.881` on `c1_2000`, and
    `44.950` on `c1_10000`, below the existing `16384` milestone at `45.476`
    sustained. Keep the current `16k` envelope as the config milestone unless a
    nearby bracket beats it.
37. The nearby `15360` max-length bracket also does not beat `16k`.
    `qwen36_27b_decode_tiers_dense_fullold_maxlen15360_envelope_h30_20260607_host30_20260607_142619`
    scored `44.953` strict, `47.085` on `c1_2000`, and `45.133` on
    `c1_10000`. This closes simple max-length bracketing for the dense c1 gate;
    use `16384` as the current config milestone and return to source-level
    communication/GEMV work for larger movement.
38. Full-graph compile and the dense `16k` max-length envelope do not combine
    into a new sustained decode winner. After fixing the runner so
    `COMPILATION_CONFIG_JSON='{"splitting_ops":[]}'` is passed as valid JSON,
    the `.30` combo run
    `qwen36_27b_decode_tiers_dense_fullold_maxlen16k_fullgraph_combo_h30_20260607_rerun_host30_20260607_143939`
    streamed coherent strict output and scored `47.031` strict, `47.556` on
    `c1_2000`, and `45.390` on `c1_10000`. This is close but below the current
    `16k` milestone (`45.476` sustained), so the next high-value work remains
    source-level RCCL/RowParallel communication or exact-shape LLMM1/GEMV
    improvement rather than more compile-envelope stacking.
39. HIP IPC peer-load collectives are rejected for this sprint, including the
    narrower adjacent-GPU case. The same-process `ipc_peer` addition to
    `gfx906_peer_allreduce_bench_20260606.cpp` failed `hipIpcOpenMemHandle`
    with `invalid device context` on TP4/TP8 `1x2048` and `1x5120`. The true
    multi-process `ipc_peer_mp` mode, with one process per GPU and atomic
    64-byte handle publish, failed every peer open with `invalid device pointer`
    for TP4, TP8, and TP2 on GPUs `0,1`. Runs:
    `gfx906_small_ar_microbench_host30_20260607_145242`,
    `gfx906_small_ar_microbench_host30_20260607_145651`, and
    `gfx906_small_ar_microbench_host30_20260607_145820`. This is stronger than
    the earlier direct-pointer memory fault: do not build the dense gate path on
    raw peer reads or HIP IPC handles unless a lower ROCr/memory-pool design is
    proven separately.
40. RCCL MSCCL++ small-message routing is not a dense serving promotion on this
    stack. AMD documents `RCCL_MSCCLPP_ENABLE=1` and a threshold for
    allreduce/allgather messages, so `.30` tested it on the exact dense
    RowParallel graph shapes. Forced Tree/LL plus MSCCL++ was neutral. Clearing
    `NCCL_ALGO`, `NCCL_PROTO`, and `NCCL_NTHREADS` while setting
    `RCCL_MSCCL_FORCE_ENABLE=1`, `RCCL_MSCCLPP_ENABLE=1`, and
    `RCCL_MSCCLPP_THRESHOLD=65536` improved isolated AR-only latency from about
    `0.051/0.053 ms` to `0.046/0.049 ms` for `1x640x5120` and
    `1x3696x5120`, but the full graph path only clearly improved the smaller
    attention shape. The serving ladder
    `qwen36_27b_decode_tiers_dense_fullold_maxlen16k_auto_mscclpp_h30_20260607_host30_20260607_151614`
    scored `45.627` strict, `47.473` on `c1_2000`, and `45.108` on
    `c1_10000`, below the current `16k` milestone. Do not promote MSCCL++ or
    `HSA_FORCE_FINE_GRAIN_PCIE=1` from this branch.
41. Exact LLMM1 RowParallel boundary testing closes chunked output overlap as a
    high-value branch. On `.30`, the real single-token dense boundaries with
    `rows_per_block=2` are only `0.063868 ms` for
    `LLMM1(5120x640)->allreduce(1x5120)` and `0.106050 ms` for
    `LLMM1(5120x3696)->allreduce(1x5120)` under Tree/LL. Splitting the output
    dimension into chunks is much worse: the best chunked Tree/LL cases are
    `0.111269 ms` and `0.154759 ms`, and two-stream pipelining is slower still.
    Auto MSCCL++ trims the full attention boundary to `0.060192 ms` but is
    neutral on MLP-down and already failed the serving ladder. The next
    RowParallel source test should target vLLM's out-of-place allreduce
    contract, not partial-output chunking.
42. Mutation-only allreduce is not the missing RowParallel win. PyNccl
    same-buffer allreduce is correct for the fresh `LLMM1` output tensor, but
    the exact `.30` boundary probe showed noise-level deltas: Tree/LL attention
    `+0.000127 ms`, Tree/LL MLP-down `-0.000047 ms`, auto+MSCCL++ attention
    `-0.000288 ms`, and auto+MSCCL++ MLP-down `+0.000977 ms`. Do not spend
    serving-patch time changing vLLM's allreduce custom-op mutation contract
    unless a different profile exposes a real allocation/copy cost. Return to
    exact-shape `LLGemm1_kernel`/GEMV work or structural communication overlap.
43. Dense LLMM1 scalar `rows_per_block=5` is not a decode promotion. The
    source patch initially failed because the private opcode `105` was used as
    the physical row block in launch geometry, creating an invalid oversized
    block and bogus `13 us` timings with garbage output. After translating
    opcode `105` to an effective five-row block before computing `NUM_THREADS`
    and `NUM_BLOCKS`, the kernel became numerically valid but slower than rpb2:
    `397.280 us` vs `382.080 us` on `n1_m31040_k5120_float16`, and
    `37.120 us` vs `36.000 us` on `n1_m5120_k2176_float16`. Keep rpb2 as the
    exact-shape LLMM1 winner.
44. vLLM CustomAllreduce cannot be promoted by relaxing policy gates on this
    gfx906 ROCm stack. A direct vLLM probe failed on TP4 and even TP2 while
    opening the first remote metadata IPC buffer. Patches for normal
    `hipMalloc`, explicit peer enable, and `cudaIpcMemHandle_t` `memcpy`
    handling did not change the failure: `hipIpcOpenMemHandle` still returned
    `invalid device pointer`. The standalone HIP IPC microbench confirmed the
    substrate issue without vLLM: TP2 adjacent GPUs `0,1` failed with
    `invalid device pointer` using lazy peer-open flag `1`, and `invalid
    argument` with flag `0`. Do not spend serving-lane time on custom
    allreduce paths that require imported peer pointers unless a lower ROCr or
    memory-pool primitive first proves valid peer handle import.
45. RowParallel mutable custom-op allreduce is graph-safe but not a decode
    promotion. The overlay
    `overlays/rowparallel_mutable_ar_20260607` added a mutating
    `vllm::all_reduce_inplace` custom op, exported
    `tensor_model_parallel_all_reduce_inplace(...)`, and used it only for
    guarded fresh `RowParallelLinear` outputs. It survived O3 startup,
    piecewise/full CUDA graph capture, and strict uncapped Qwen thinking output
    on `.30`. The full ladder
    `qwen36_27b_decode_tiers_dense_fullold_maxlen16k_rowpar_mutable_ar_fulltiers_h30_20260607_171759_host30_20260607_171759`
    produced strict `46.330`, `c1_2000` `47.052`, and `c1_10000` `45.047`
    backend decode TPS. This is below the current `16k` incumbent
    (`45.476` on `c1_10000`). A repeat run produced strict `46.754`,
    `c1_2000` `47.780`, and `c1_10000` `45.607`, but the paired static
    incumbent repeat produced strict `46.620`, `c1_2000` `47.600`, and
    `c1_10000` `45.471`; the `0.136 TPS` 10k difference is run-to-run
    variance, not a deployment promotion. Keep the patch as an extensible
    source milestone: mutable RowParallel collectives can be represented safely
    in the graph, but alias/mutation contract cleanup alone is not the missing
    `65 TPS` lever.
46. Raw HIP peer-buffer custom collectives are rejected on the current gfx906
    PCIe topology. The TP8 multiprocess IPC peer probe failed every remote
    buffer open with `invalid device pointer`; single-process direct peer reads
    faulted with GPU memory access faults; and the correct single-process
    copy-peer staging path measured `1364.02 us` for `1x5120`, versus
    `51-58 us` PyNccl allreduce-only time on the exact RowParallel decode
    boundaries. Do not spend serving-source time on a raw peer-buffer collective
    until a separate memory-sharing substrate first proves valid remote device
    buffer import across worker processes.
47. LLMM1 rows-per-block tuning does not clear dense decode. A fresh `.30`
    exact RowParallel boundary sweep found `rpb4` slightly better than `rpb2`
    for MLP-down (`0.100348 ms` vs `0.104681 ms`) and neutral for attention, but
    the full serving ladder rejected it: strict `46.025`, `c1_2000` `46.457`,
    and `c1_10000` `44.468` backend TPS versus the paired `rpb2` incumbent
    repeat's `45.471` on `c1_10000`. Keep `rpb2`; future GEMV work needs a
    real kernel-structure change, not just output rows-per-block selection.
48. Shape-selective LLMM1 routing is safe but still variance-band for sustained
    dense decode. The overlay
    `hotfixes/utils_llmm1_selective_rowpar_rpb4/utils_llmm1_selective_rowpar_rpb4.py`
    kept `rpb2` globally and routed only exact RowParallel decode boundaries
    `m=5120,k in (640,3696)` through `rpb4`. It passed strict thinking output
    twice and produced `c1_10000` `45.493` then `45.371` backend TPS. That is
    not a repeatable win over the paired `rpb2` repeat (`45.471`), but it is a
    useful source milestone proving shape-selective GEMV routing can be used
    safely for future kernel experiments.
49. LLMM1 scalar output-store is not a dense decode promotion. The standalone
    scalar-store MEP
    `runs/llmm1_scalar_store_shape_host30_20260607_191927` was numerically
    exact versus `LLMM1_rpb2`, but lost on the dominant
    `n1_m31040_k5120_float16` gate/up projection (`385.600 us` vs
    `381.121 us`). It showed isolated wins on smaller `5120xK` shapes, so it
    was carried into a full RowParallel boundary repeat instead of being
    rejected prematurely.
50. The full RowParallel scalar-store repeat closes that branch. Run
    `runs/llmm1_scalar_store_rowparallel_host30_20260607_192416_repeat` showed
    attention `5120x640` is best with incumbent `rpb2` (`0.061562 ms/call`);
    scalar-store `rpb2` regressed to `0.062524`, and `rpb4` regressed to
    `0.062018`. MLP-down `5120x3696` is best with existing `rpb4`
    (`0.099966 ms/call`) rather than scalar-store. The refined serving overlay
    `hotfixes/utils_llmm1_selective_mlpdown_rpb4/utils_llmm1_selective_mlpdown_rpb4.py`
    routed only `m=5120,k=3696` through `rpb4`; it passed strict thinking output
    but scored `c1_10000` `45.465`, tied with the paired `rpb2` incumbent
    repeat (`45.471`). Do not promote scalar-store or MLP-down-only rpb4 as the
    dense winner.
51. Presplit17 post-IR RowParallel consumer fusion remains a source milestone,
    not a serving winner, even under the current 16k envelope. The `.30` retest
    with `qwen36-gfx906-postir-direct-mlp-ar-rms:20260607-presplit17-attn`,
    direct+attention targets, strict local cleanup, and the same four-hotfix
    dense stack passed the strict thinking smoke but reached only `46.720`
    backend TPS on `c1_2000` and `44.748` on `c1_10000`. That is below the
    current `rpb2` incumbent repeat band (`45.471-45.476` on `c1_10000`). The
    post-IR matcher is useful because it proves legal graph rewrites can target
    both RowParallel consumer classes, but the gate now needs cheaper/fewer
    collectives or a real exact-shape GEMV wall-time reduction.
52. Simple exact-K LLMM1 specialization is not enough for the dense gate. The
    standalone `.30` MEP
    `runs/llmm1_k5120_specialized_host30_20260607_200904` tested exact
    `K=5120` variants against stock `ops.LLMM1(..., rpb2)`. They were
    numerically correct and won small shapes such as `m=2048,k=5120`
    (`33.760 us` vs `35.360 us`), but all variants lost on the dominant
    `n1_m31040_k5120_float16` gate/up shape: stock was `380.800 us`, while the
    best exact-K variant was `384.959 us`. Do not promote this branch into
    serving. Keep it as negative source evidence that the next GEMV attempt
    needs a deeper arithmetic/reduction/launch strategy, not just generic-K
    branch removal or a `float4` B-load path.
53. `wvSplitK` remains a concurrency/prefill source milestone, not a c1 dense
    decode path. The patched `.20` image
    `qwen36-gfx906-wvsplitk-env:20260607` was revisited for exact `n=1`
    decode shapes in
    `runs/wvsplitk_n1_revisit_host20_20260607_201646`. It was numerically
    valid but slower than LLMM1 everywhere: dominant
    `n1_m31040_k5120_float16` was `446.879 us` for `wvSplitK` versus
    `380.638 us` for best LLMM1, and `n1_m5120_k640_float16` was `19.520 us`
    versus `17.120 us`. Keep the earlier `n=2..4` small-batch result for
    concurrency lanes, but do not route c1 through `wvSplitK`.
54. Serial row-pair LLMM1 workgroups do not improve gfx906 c1 GEMV. The `.30`
    MEP `runs/llmm1_serial_pairs_nobarrier_host30_20260607_202658` kept the
    proven two-row accumulator but processed multiple row pairs inside one
    workgroup to reuse the single-token activation vector. The no-extra-barrier
    variant was numerically identical to stock LLMM1, but still lost every
    tested `n=1` shape: dominant `n1_m31040_k5120_float16` was `401.761 us`
    versus stock `382.080 us`, and `n1_m5120_k640_float16` was `18.080 us`
    versus `17.120 us`. This closes activation-vector reuse via larger
    workgroups; future LLMM1 work needs a different reduction/arithmetic model
    or composition with collective/consumer work.
55. Packed-FP16 LLMM1 reduction is not the missing arithmetic lever. The `.30`
    MEP `runs/llmm1_half_reduce_host30_20260607_203057` kept partial sums in
    `half2` through warp/block reduction and converted only at final store. It
    introduced extra numeric drift (`0.25` max abs vs stock on major K=5120
    shapes) and still lost the dominant `n1_m31040_k5120_float16` shape:
    `390.561 us` versus stock `381.440 us`. Isolated wins on smaller shapes do
    not justify serving promotion. The current LLMM1 wall is not cleared by
    replacing the fp32 scalar reduction with packed FP16 reduction.
56. Extensibility is evaluated separately from immediate promotion. The
    mutable RowParallel path did not win by itself, but it became a real source
    base: `overlays/postir_mutable_fusion_20260608` patched the post-IR matcher
    to understand `all_reduce_inplace(gemm)` as a collective boundary whose
    consumer value is the mutated GEMM tensor. The full `.30` ladder
    `runs/qwen36_27b_decode_tiers_dense_fullold_maxlen16k_postir_mutable_inplace_patch_fulltiers_h30_20260607_205534_host30_20260607_205534`
    selected `64/64` direct MLP boundaries on every rank and passed strict
    thinking output, proving that the milestones compose correctly. It still
    did not promote: `c1_2000` was `47.614` backend TPS and `c1_10000` was
    `45.415`, below the current incumbent band. This keeps the mutable/post-IR
    composition as source evidence only and moves the primary path back to
    cheaper collectives, true `GEMV + collective` overlap/fusion, legal
    reduction-count changes, or a materially faster exact-shape GEMV.
57. Shape-selective exact-K LLMM1 dispatch is graph-safe but not a dense decode
    promotion. The `.30` overlay
    `hotfixes/utils_llmm1_k5120_selective/utils_llmm1_k5120_selective.py`
    routed only `n=1,k=5120,m in (2048,4352)` through the specialized
    `llmm1_k5120_b4` extension while keeping stock `LLMM1_rpb2` for the
    dominant `m=31040,k=5120` logits/gate shape. The full ladder
    `runs/qwen36_27b_decode_tiers_dense_fullold_maxlen16k_k5120_selective_fulltiers_nostream_h30_20260607_213846_host30_20260607_213846`
    passed strict uncapped thinking output at `46.326` backend TPS, then
    produced `47.255` on `c1_2000` and `45.215` on `c1_10000`. This proves
    selective external GEMV dispatch can survive O3 graph serving, but the
    microbench win on smaller K5120 shapes does not survive the full decode
    schedule. Keep it as extensibility evidence, not the global winner.
58. Stacking small neutral source milestones is not enough. The `.30`
    composition
    `runs/qwen36_27b_decode_tiers_dense_fullold_maxlen16k_mutable_k5120_combo_h30_20260607_215827_host30_20260607_215827`
    combined mutable RowParallel allreduce with the shape-selective K5120 LLMM1
    extension. The extension loaded on all eight ranks, the mounted mutable
    overlay preserved strict thinking output, and the run produced strict
    `46.950`, `c1_2000` `47.234`, and `c1_10000` `45.299` backend TPS. This is
    a valid composition milestone, but not a promotion. Future dense-gate work
    should stop expecting additive gains from independent variance-band hooks
    and return to larger structural changes: fewer legal RowParallel
    reductions, a genuinely faster dominant `m=31040,k=5120` GEMV, or a real
    original-boundary `GEMV + collective` implementation.
59. LLMM1 wavegroup row-pair scheduling is not the missing dense decode lever.
    The `.30` MEP
    `runs/llmm1_wavegroup_pairs_host30_20260608_022316` split one CTA into
    multiple independent wavegroup lanes so each lane group could keep the
    proven two-row accumulator shape while reducing CTA count. It was
    numerically inside the stock fp16 reduction-order band, but every K5120
    shape lost to stock `LLMM1_rpb2`. The dominant
    `n1_m31040_k5120_float16` shape was `380.641 us` for stock versus
    `396.962 us` for the best wavegroup variant, with a quick dominant repeat
    at `378.562 us` stock versus `397.602 us` best wavegroup. The isolated
    `rpb1_one_row` win on `n1_m5120_k3696_float16` is too small and too
    narrow to reopen the rows-per-block serving lane, which already failed
    selective RowParallel promotion. Future work should move away from LLMM1
    scheduling variants unless the exact dominant shape improves first.
60. `ncclAllReduceWithBias` is a promising RCCL-version lead, not an immediate
    patch for the current image. Current RCCL docs list
    `ncclAllReduceWithBias(..., acc)`, which could map to the dense boundary
    `allreduce -> residual add -> RMSNorm` without relying on broken HIP peer
    IPC or PyTorch symm_mem fused collectives. The deployed ROCm 6.3 image on
    `.30` does not export the symbol from either PyTorch's bundled
    `librccl.so` or `/opt/rocm/lib/librccl.so*`; standard `ncclAllReduce` is
    present. Treat this as a future RCCL upgrade/backport lane. The required
    next evidence is a gfx906-capable RCCL image where the symbol exists, then
    a TP8 graph-replay comparison of normal allreduce, allreduce-with-bias, and
    allreduce-with-bias plus fused RMSNorm consumer.
61. The AllReduceWithBias lane needs an RCCL build artifact before serving
    integration. ROCm `7.1.1` source carries `acc` through the API, task, device
    work struct, generated dispatch key, and Simple/LL/LL128 device primitives;
    ROCm `6.3.3` does not. The `.30` probe
    `runs/pynccl_allreduce_with_bias_probe_host30_20260607_223813` cleanly
    reports `symbol_missing` on the current deployment image, while preserving a
    reusable TP8 graph-replay harness for future builds. The device path casts
    `acc` to the collective dtype, so same-dtype residual/bias fusion is the
    first valid target; FP32 residual semantics require separate evidence before
    this can replace the current Qwen/Gemma residual boundary.
62. Path extensibility must be checked against the already-fused ROCm baseline.
    Qwen-family decode sends RowParallel allreduce outputs into
    `RMSNorm.forward_hip(..., residual)`, and vLLM already dispatches that
    residual add through the ROCm fused add+RMSNorm path. Therefore
    `ncclAllReduceWithBias` is not automatically "one less kernel" for the
    main dense residual boundary. A viable serving patch would need to prove
    that moving the same-dtype residual add into RCCL and then running plain
    RMSNorm is faster or more composable than the current
    `allreduce -> fused_add_rms_norm` schedule. Keep this as a high-extensibility
    source lane, but require microbench evidence before serving integration.
63. ROCm `7.1.1` RCCL `ncclAllReduceWithBias` is now a real gfx906 source
    milestone. The `.30` overlay build
    `/usr/share/ollama/rccl_overlay_711_gfx906_20260608_0240` completed after
    `10458.7` seconds, installed `librccl.so`, and reports
    `ncclGetVersion=22707` with `ncclAllReduceWithBias` exported. Loaded through
    `VLLM_NCCL_SO_PATH`, the TP8 graph replay probe is correct on all tested
    shapes. It wins strongly at `1x5120` (`0.265434` vs `0.397772` ms/call),
    wins at rows `2-4`, is variance-band at row `8`, and regresses at rows
    `16+`. This is not a broad prefill/concurrency win, but it is a viable
    decode-shaped base for an env-gated, shape-gated source patch.
64. Close TPS deltas are evaluated by extensibility, not just immediate rank.
    The AllReduceWithBias lane is the current example: the first O3 serving
    overlay loaded the ROCm `7.1.1` RCCL library through `VLLM_NCCL_SO_PATH`,
    passed strict uncapped smoke, and reached `46.375` backend TPS, but the
    low-row Python model hook did not select the fused path in serving. That is
    not a serving promotion, but it remains valuable because it proves the
    deployment image can use a newer RCCL ABI and gives us a reusable PyNccl
    binding for future graph/pass-level work.
65. Do not put normal vLLM allreduce fallback inside a graph custom op for
    prefill. The all-rows custom-op experiment that routed prefill through
    `group._all_reduce_out_place(input_) + residual` wedged the executor:
    the first request entered a `431` token prefill batch, metrics stayed at
    zero, workers spun near `90%` CPU, and the shared-memory broadcast poller
    timed out. This is a source-path rejection, not run variance.
66. The strict all-rows RCCL variant proves the fused primitive can be selected
    in O3 serving at the decode shape. During warmup on `.30`, all eight TP
    ranks logged
    `gfx906 AllReduceWithBias custom op path=rccl shape=(1, 5120)
    dtype=torch.float16`. The same route still stalled on the real `431` token
    prefill request, so the missing engineering piece is prefill-safe routing:
    keep prefill on the stock allreduce/fused-add-RMSNorm path and apply
    `ncclAllReduceWithBias` only to decode-sized graph ranges/callsites.
67. Current `qwen36-gfx906-c1-topk8-fastpath-reproducible:latest` does not
    include AITER. A Docker probe reports `aiter_spec None`. Upstream vLLM has a
    ROCm AITER allreduce/RMSNorm fusion pass behind `VLLM_ROCM_USE_AITER=1`, but
    that is not a config-only fix for this image. Treat AITER as a separate
    image/backport lane, while the primary source path should use our proven
    RCCL overlay plus an FX/pass-level decode-only replacement.
68. Path extensibility matters when the TPS result is variance-band. The
    post-IR attention-boundary `AllReduceWithBias` lane is a good example:
    it is not a dense decode winner, but it is a valid source milestone because
    O3 serving emitted `torch.ops.vllm.gfx906_all_reduce_with_bias_postir` in
    the compiled graph and logged the ROCm `7.1.1` RCCL path on real serving
    shapes, including `(431, 5120)`.
69. The attention-boundary `AllReduceWithBias` lane also exposes the next real
    blocker. Strict uncapped smoke
    `qwen36_27b_decode_tiers_dense_fullold_maxlen16k_postir_ar_bias_attention_strict_h30_20260608_host30_20260608_postir_arbias_attention_strict`
    passed the hard thinking gate with answer hash
    `ccda5f21326af8140c17e3323a22e16644dd50eebd532e04fcdd20d9d5f38938`
    and `46.267` backend decode TPS. That is baseline-band, not promotion.
    The source reason is clear from the generated graph: only the first
    attention residual has a same-dtype FP16 residual usable by RCCL
    `ncclAllReduceWithBias`; later direct MLP and attention residual chains
    carry FP32 residual state. Extending this path requires a mixed-dtype
    fused collective or a different residual/collective schedule.
70. Full-boundary replay rejected same-dtype `ncclAllReduceWithBias` as the
    dense residual replacement. Run
    `runs/pynccl_post_ar_rms_graph_bench_host30_20260608_041253` compared the
    current `AllReduce -> fused_add_rms_norm` boundary with
    `AllReduceWithBias -> fused zero-residual RMSNorm` on TP8 `.30` using the
    RCCL `7.1.1` overlay. The biased path was slower at every tested shape:
    `+0.002349 ms` at `1x5120`, `+0.029819 ms` at `8x5120`, `+0.055933 ms` at
    `16x5120`, and `+0.477225 ms` at `128x5120` versus the current fused
    boundary. The residual-downcast approximation also introduced
    `0.000976562` max error versus the FP32-residual reference. This keeps the
    source milestone but closes same-dtype AR-bias as a serving promotion path.
71. Mixed-dtype RCCL bias fusion is source-viable but not wrapper-sized.
    Inspection of the ROCm `7.1.1` RCCL overlay shows the public API,
    `ncclInfo`, task planner, `ncclDevWorkColl`, generated device-function key,
    and Simple/LL/LL128 device primitives all carry only one collective dtype
    plus an `acc` pointer/enable bit. The device paths type `acc` as the same
    `T` as the payload. Preserving dense FP32 residual semantics therefore
    requires a real RCCL source extension with accumulator dtype metadata and
    new device kernels, not a Python/PyNccl shim. Keep it as a future serious
    source branch, but the near-term primary path should return to
    original-boundary RowParallel overlap/fusion, legal reduction-count changes,
    or a materially faster exact-shape `LLGemm1_kernel`/GEMV path.
72. Separate-stream LLMM1/allreduce overlap is not a fine-grained RowParallel
    fix. The `.30` TP8 lower-bound run
    `runs/llmm1_allreduce_overlap_host30_20260608_043418` launched independent
    LLMM1 compute and a `1x5120` PyNccl allreduce on separate HIP streams. The
    heavy `gate_up_31040x5120` shape showed partial substrate support,
    improving serial `0.422576 ms` to `0.390539 ms`, but the smaller
    RowParallel-sized shapes regressed badly: `attn_out_5120x640` went from
    `0.061625` serial to `0.157413` overlapped, and
    `mlp_down_5120x3696` went from `0.104530` to `0.163864`. Overlap remains
    viable only for a legal coarse schedule with independent heavy work; a
    per-boundary `compute_stream`/`comm_stream` wrapper should not be promoted.
73. ROCm 7.2 full-stack dense TP8 is a valid pre-release stack milestone,
    later superseded by the 2026-06-20 ROCm7.2 Dense/MoE validation.
    Exact-shape `.30` microbenching compared the 6.3 reproducible stack
    (`runs/llmm1_shape_stack_compare_rocm63_repro_exactshape_host30_20260608_044440`)
    with the 7.2 experimental image
    (`runs/llmm1_shape_stack_compare_rocm72_experimental_exactshape_host30_20260608_044548`).
    The dominant `n1_m31040_k5120_float16` `LLMM1_rpb2` path improved only
    `381.758 -> 380.798 us` (`-0.25%`), while smaller RowParallel-like shapes
    improved by about `0.9-2.8%`. The full 7.2 serving ladder
    `runs/qwen36_27b_decode_tiers_rocm72_fullold_maxlen16k_fulltiers_host30_20260608_050020`
    passed strict thinking output and reached `47.138` strict, `47.594` on
    `c1_2000`, and `45.737` on `c1_10000`. This is the best single sustained
    number so far, but only `~0.13 TPS` above the previous best repeat and
    still far below the `65 TPS` dense gate. Keep ROCm 7.2 as an extensible
    runtime/library lane; do not treat the variance-band sustained result as a
    serving promotion without repeat and two-host evidence.
74. ROCm 7.2 plus mutable RowParallel is the current dense sustained source
    milestone. The `.30` composition runs
    `qwen36_27b_decode_tiers_rocm72_mutable_rowpar_maxlen16k_fulltiers_20260608`
    and
    `qwen36_27b_decode_tiers_rocm72_mutable_rowpar_maxlen16k_repeat_20260608`
    used the same dense full-old TP8 `16k` envelope, the 7.2 experimental
    image, and `VLLM_GFX906_ROWPAR_MUTABLE_AR=1`. Both strict uncapped smokes
    were parser-valid. The repeated fixed tiers reached `48.577/48.719`
    backend TPS on `c1_2000` and `46.389/46.214` on `c1_10000`. The same image
    ID was then streamed to `.20` and reproduced at `50.454/49.863` on
    `c1_2000` and `48.131/47.582` on `c1_10000`. This is above the previous
    sustained band (`45.471-45.737`) and proves that near-tie source milestones
    can compose. It is still not a `65 TPS` gate clear and not
    production-portable until the 7.2 stack has a real build/deploy path.
75. Do not trust early zero metrics as a hang signal on ROCm 7.2 decode. In
    the `.20` cross-host `c1_10000` run, vLLM metrics initially lagged while
    `rocm-smi` showed `96-99%` GPU use and worker logs showed steady
    `num_tokens=1` graph execution. The request completed at `48.131` backend
    TPS with valid backend counter deltas. Use metrics, GPU utilization, and
    worker logs together before deciding a long decode is wedged.
76. The exact c1 profile of the current dense winner keeps the primary source
    branch on communication. Run
    `qwen36_27b_c1_rocprof_rocm72_mutable_rowpar_c1exact_20260608_host20_20260608_064240`
    used the ROCm 7.2 + mutable RowParallel `.20` winner path and emitted a
    valid `1,141,659` row kernel trace for capped c1_1024. The profile TPS
    (`20.687`) is attribution-only because rocprof slows serving, but the kernel
    mix is decisive: NCCL is `21.158s / 62.67% / 132,882` rows and
    `LLGemm1_kernel` is `9.052s / 26.81% / 311,470` rows. Attention plus
    GDN/linear attention is only about `2.63%` of filtered kernel time. For
    dense c1 decode, prioritize legal RowParallel reduction-count changes,
    grouped/coalesced allreduce, or fused collective support before attention
    microtuning.
77. Exact greedy logits cleanup does not meaningfully compose with the ROCm 7.2
    + mutable RowParallel winner. The `.20` composition run
    `qwen36_27b_decode_tiers_rocm72_mutable_rowpar_greedy_top1_compose_20260608_host20_20260608_070744`
    passed strict thinking smoke and reached `49.900` backend TPS there,
    `49.962` on `c1_2000`, and `47.692` on `c1_10000`. That sustained result is
    only about `+0.23%` over the `.20` repeat baseline (`47.582`) and below the
    first `.20` pass (`48.131`), so it is not a promotion. The branch remains a
    correctness-valid source milestone but should not be carried as part of the
    current global winner unless a larger collective change makes logits
    reduction materially more valuable.
78. `VLLM_LOGGING_LEVEL=DEBUG` was suppressing dense decode performance enough
    to matter. The current ROCm 7.2 + mutable RowParallel winner inherited DEBUG
    from the ai-infos command shape and emitted per-token worker batch logs
    during c1 decode. Changing only `VLLM_LOGGING_LEVEL=INFO` produced `.20`
    `c1_10000` results of `49.668` and `49.627` backend TPS, versus the prior
    `.20` DEBUG repeat at `47.582`. `.30` cross-host reached `47.604`, also
    above its prior DEBUG band but below `.20`. Promote INFO as the
    performance-candidate default; keep DEBUG for diagnostics only.
79. Metrics lag is not a rejection signal by itself. During the INFO-winner
    rocprof follow-up, `/metrics` stalled while GPU occupancy and worker logs
    showed active decode work. The run ultimately failed because rocprof caused
    a vLLM shared-memory/RPC executor timeout during warmup, not because the
    INFO serving path was invalid. For long/profiler lanes, use GPU occupancy,
    worker logs, response artifacts, and final decode counter deltas as the
    authority; only reject on a hard crash, timeout, bad output, or completed
    TPS evidence.
80. ROCm/RCCL overlays must match the runtime closely enough to initialize
    communicators, not just export the target symbol. The ROCm `7.1.1`
    `ncclAllReduceWithBias` overlay could be force-loaded inside the ROCm 7.2
    image with soname aliases, and `ctypes` confirmed the symbol, but vLLM TP8
    startup failed at `ncclCommInitRank` with `NCCL error: invalid usage`.
    Build the fused collective substrate from matching ROCm 7.2 RCCL source;
    `rocm-7.2.1` already carries the API, so this is a matched-build problem
    before it is a vLLM pass problem.
81. Matched ROCm 7.2 RCCL is useful for ordinary allreduce but not for
    `AllReduceWithBias` on this gfx906 stack. The `.30` gfx906 build at
    `/usr/share/ollama/rccl_overlay_721_gfx906_20260608_082322` exports
    `ncclAllReduceWithBias` and loads through `VLLM_NCCL_SO_PATH`. Ordinary
    decomposed PyNccl allreduce improved slightly versus native ROCm 7.2
    (`1x5120` `0.117685 ms` vs `0.124857 ms`; `8x5120` `0.161931 ms` vs
    `0.184381 ms`). But the fused-bias call SIGSEGVs for world size 2+ and
    the direct C++ world-size-1 probe returns success while ignoring the bias
    (`max_abs_error=0.191`). Reject `AllReduceWithBias` for serving until RCCL
    source correctness is fixed.
82. The matched RCCL 7.2 plain-overlay serving run is a variance-band milestone,
    not a global winner. On `.30`, the ROCm 7.2 + mutable RowParallel + INFO
    envelope with `VLLM_NCCL_SO_PATH=/rccl-overlay/install/lib/librccl.so` and
    `NCCL_NTHREADS=256` reached strict c1 `49.631`, `c1_2000` `50.225`, and
    `c1_10000` `47.849` backend TPS. That is slightly above the prior `.30`
    cross-host INFO run (`47.604`) but below the `.20` global sustained result
    (`49.668`), so do not promote it as the current default.
83. The ROCm 7.2 RCCL `ncclAllReduceWithBias` failure was source-fixable on
    gfx906. The unpatched code first crashed in an unsupported-architecture
    warning path that used a null `task` pointer, then cleanly reported gfx906
    unsupported once that warning used the aggregate fields. Extending the
    non-LL128 host/device guards to include gfx906 made f16 Tree/LL
    `AllReduceWithBias` pass direct C++ TP8 tests at rows `1`, `16`, and `128`
    with half-precision error bounded by `0.0150146`. Keep this distinction:
    the old rejection applied to unpatched RCCL; the patched primitive is now
    correctness-valid.
84. A serving-capable RCCL fused-bias overlay must include f32 as well as f16.
    PyNccl initializes with a tiny default `torch.zeros(1)` warmup allreduce,
    which is float32. The f16-only minimal RCCL overlay aborted during PyNccl
    communicator startup with a GPU memory fault before any benchmark rows were
    emitted. The combined f16+f32 Tree/LL plus Ring/SIMPLE overlay
    `/usr/share/ollama/rccl_overlay_721_gfx906_acc_minimal_f16f32_20260608`
    initializes correctly and exports both `ncclAllReduce` and
    `ncclAllReduceWithBias`.
85. `ncclAllReduceWithBias` is an extensibility milestone, not yet a dense-gate
    answer. The PyNccl graph probe on `.20` passed but showed the fused-bias
    call slower than decomposed allreduce plus add: `1x5120` was `0.099075 ms`
    vs `0.088541 ms`, `16x5120` was `0.293163 ms` vs `0.215048 ms`, and
    `128x5120` was `1.757810 ms` vs `1.188194 ms`. This argues against a blind
    full-boundary promotion, but the primitive is still valuable as a substrate
    for fused collective scheduling and mixed consumer fusion.
86. The fair serving test must compose source paths rather than overwrite them.
    Mutable RowParallel and AR-bias both replace `communication_op.py`; stacking
    mounts naively disables one path. The combined shim
    `overlays/rccl_ar_bias_plus_mutable_rowpar_20260608/communication_op.py`
    preserves mutable in-place RowParallel and lets the in-place call consume a
    pending residual through RCCL `AllReduceWithBias`. Strict serving logs
    confirmed the fused RCCL path at `(2048,5120)`, `(8,5120)`, `(4,5120)`,
    `(2,5120)`, `(1,5120)`, and real prompt `(431,5120)` shapes.
87. The integrated mutable+AR-bias serving path is not a global dense winner
    after repeat. On `.20`, the first run reached strict c1 `51.457`,
    `c1_2000` `52.460`, and `c1_10000` `49.809` backend TPS, narrowly above
    the prior `.20` INFO winner band (`49.668`/`49.627`). The `.20` repeat
    reached strict c1 `51.440`, `c1_2000` `51.611`, and `c1_10000` `49.365`;
    `.30` was lower at `47.766` sustained. Promote the RCCL source fix as a
    primitive milestone, but keep the INFO mutable RowParallel config as the
    serving winner.
88. Upstream-style GDN output flattening is a clean source no-op/tiny cleanup,
    not a dense decode lever. Overlay
    `overlays/qwen35_gdn_flatten_20260608/qwen3_5.py` replaced the Qwen3.5/3.6
    GDN `einops.rearrange(... h d -> ... (h d))` path with
    `Tensor.flatten(-2)` and removed the runtime `einops` import. The first
    parallel attempt was invalid because both hosts accidentally used local
    proxy port `18797`, causing `.30` traffic to hit the `.20` proxy and
    contaminating backend deltas. The corrected two-host rerun used isolated
    proxy ports and one-request metric deltas. `.20` reached strict c1
    `50.969`, `c1_2000` `51.880`, and `c1_10000` `49.459`; `.30` reached
    strict c1 `49.368`, `c1_2000` `49.829`, and `c1_10000` `47.697`. This is
    below the `.20` INFO winner band, so keep the change as upstream-alignment
    evidence only. The earlier exact profile already showed GDN/linear
    attention at only about `2.63%` of filtered kernel time, and the serving
    result confirms that simple GDN wrapper cleanup is not the `65 TPS` path.
89. The ai-infos `8192` batch/cache shape still does not promote after the
    ROCm 7.2 + mutable RowParallel + INFO milestone. A clean `.20` isolation
    run changed only the current winner's batching width to
    `max_num_batched_tokens=8192` while keeping `max_num_seqs=4` and prefix
    cache disabled; it reached strict c1 `51.263`, `c1_2000` `51.594`, and
    `c1_10000` `49.219` backend TPS. That is close but below the `.20` INFO
    winner band (`49.668`/`49.627`). A fuller ai-infos-style `.30` run with
    `max_num_seqs=8`, `max_num_batched_tokens=8192`, and prefix cache enabled
    reached strict c1 `47.852`, `c1_2000` `48.562`, and `c1_10000` `46.133`.
    Treat 8192-wide capture as a prefill/concurrency lane, not the current
    single-request decode winner.
90. ROCm 7.2 does not reopen the rejected LLMM1 exact-shape variants. The
    shape-stack rerun inside
    `qwen36-gfx906-c1-topk8-fastpath-rocm72-experimental:latest` produced
    the same decision as the older stack. On `.20`, stock `LLMM1_rpb2` was
    best for all tested meaningful shapes: `379.520 us` for
    `n1_m31040_k5120`, `68.800 us` for `n1_m5120_k5120`, `59.680 us` for
    `n1_m4352_k5120`, `52.320 us` for `n1_m5120_k3696`, and `16.320 us` for
    `n1_m5120_k640`. `.30` matched the same pattern, with dominant
    `n1_m31040_k5120` at `380.798 us`. Keep the exact-shape GEMV target in
    the source backlog, but do not revive rows-per-block, serial-pair,
    wavegroup, or simple K5120-load specialization as primary paths without a
    materially new kernel structure.
91. Async scheduling is now promoted on the ROCm 7.2 + mutable RowParallel +
    INFO dense stack. This is a stack-dependent result: old-stack async was
    previously rejected, but after the ROCm 7.2 image, mutable RowParallel
    overlay, INFO logging, TP8, `max_model_len=16384`, `max_num_seqs=4`,
    `max_num_batched_tokens=2048`, prefix cache off, Tree/LL, one channel,
    P2P disabled, and `NCCL_NTHREADS=128`, `--async-scheduling` repeatedly
    moved sustained decode. `.20` reached strict c1 `55.681`, `c1_2000`
    `56.391`, and `c1_10000` `53.511` in the first run; the repeat reached
    strict c1 `55.732`, `c1_2000` `56.479`, and `c1_10000` `53.573`.
    Cross-host `.30` reached strict c1 `53.101`, `c1_2000` `54.008`, and
    `c1_10000` `51.313`. Promote this as the current global dense
    source/config milestone, still not a `65 TPS` gate clear. The sibling
    same-stack `.30` check with `NCCL_NTHREADS` unset only reached strict c1
    `49.492`, `c1_2000` `50.085`, and `c1_10000` `47.614`, so keep
    `NCCL_NTHREADS=128` in the current winner envelope.
92. Whole-server `rocprofv2`/`rocprofv3` wrapping remains too intrusive for
    the current dense async profile lane.
    The `.20` attempt
    `runs/qwen36_27b_c1_rocprof_rocm72_mutable_rowpar_info_async_c1profile_20260608_host20_20260608_140108`
    started cleanly and reached serving readiness, but the first capped warmup
    under rocprof repeatedly emitted shared-memory broadcast waits and then
    failed with `TimeoutError: RPC call to sample_tokens timed out`, followed
    by `EngineDeadError` and HTTP 500. The follow-up `rocprofv3` attempt
    `runs/qwen36_27b_c1_rocprof_rocm72_mutable_rowpar_info_async_rocprofv3_c1profile_libdw_20260608_host20_20260608_142033`
    required mounting host `libdw.so.1` into the container, then reached
    readiness, accepted a 431-token warmup request, kept metrics at zero while
    repeating the shared-memory broadcast warning, and failed the same
    `sample_tokens` RPC path with HTTP 500. Treat both as profiler-induced
    executor failures, not serving regressions. Future attribution should use
    filtered capture, post-run cache artifacts, or standalone graph-replay
    microbenches rather than whole-server rocprof wrapping.
93. `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` with
    `--gpu-memory-utilization 0.9667` is rejected as a dense async decode
    serving config for now. On `.30`, the lane increased available KV cache
    from `374,400` to `382,400` tokens and maximum 16K concurrency from
    `85.09x` to `86.98x`, but the first strict request and an independent
    direct one-token backend request both hung before vLLM reported any
    running/waiting request or prompt/generation tokens. This may be useful for
    future concurrency/prefill work after the first-request stall is understood,
    but it is not a decode-gate candidate.

## Symmetric-Memory Fused Collective Rejection - 2026-06-07

- Probe:
  `microbenches/symm_mem_fused_collective_probe_20260607.py`, run on `.20`
  GPUs `0-3` inside `qwen36-gfx906-wvsplitk-env:20260607`.
- Actual hardware check: all eight `.20` cards report
  `34342961152` bytes of VRAM via `rocm-smi --showmeminfo vram`, even though
  product-name strings label cards `6/7` as `16GB`. Continue using measured
  VRAM and stable DID/rev data, not product strings.
- PyTorch stack: `torch 2.9.1+rocm6.3`, HIP `6.3.42134-a9a80e791`.
- `torch.distributed` NCCL process-group init succeeds on TP4.
- `torch.distributed._symmetric_memory.enable_symm_mem_for_group("0")`
  succeeds on every rank.
- Both upstream AsyncTP fused primitives are unusable on this stack:
  - `_fused_matmul_reduce_scatter(...)` entered on all four ranks and did not
    return on tiny tensors; GPUs `0-3` sat at `100%` use with negligible VRAM.
  - `_fused_all_gather_matmul(...)` behaved the same way.

Decision:

- Do not port vLLM `AsyncTPPass` to gfx906 by simply relaxing CUDA/device
  gates. The required PyTorch symm_mem fused compute/collective substrate
  exists enough to enable, but the actual fused primitives spin on gfx906 ROCm
  6.3.
- Any future fused `GEMM + ReduceScatter` or `AllGather + GEMM` path for gfx906
  needs a different backend implementation, most likely RCCL/PyNccl graph-safe
  composition or a new HIP/C++ primitive, not `torch.ops.symm_mem`.

93. 2026-06-08 tiny-prompt rocprofv3 check confirms the metrics-lag symptom is a profiler-induced pre-metrics stall, not hidden decode progress. The reduced prompt run `qwen36_27b_c1_rocprof_rocm72_mutable_rowpar_info_async_rocprofv3_tinyprompt_20260608_host20_20260608_143534` reached readiness but produced HTTP 500 before any prompt/decode counters moved, with repeated shared-memory broadcast warnings. Close whole-server rocprof wrapping for the async dense winner.
94. Rebased ROCm 7.2 post-IR fused RMS consumer source work is buildable and serviceable, but the old matcher is a no-op on the current mutable RowParallel graph. The derivative `.30` image `qwen36-gfx906-rocm72-postir-rms:20260608` built from `ai-infos/vllm-gfx906-mobydick@6a52668c14dc9cb94a270d94bfec735fe627ed0c` and exposes `torch.ops._C.gfx906_gemma_add_rms_norm_fp32_residual` after importing `vllm._C`. The decode-tier run `qwen36_27b_decode_tiers_rocm72_mutable_rowpar_info_async_postir_rms_20260608_host30_20260608_150559` produced a valid strict smoke (`53.124` backend TPS, answer hash `111fabc12c44ab9bc483d882a8d1554b7395536072fe557f8b664f19d014ac8b`), `c1_2000` `54.026`, and `c1_10000` `51.375`, which matches the `.30` async baseline but does not improve the gate. Logs showed `gfx906 post-IR allreduce RMS fusion selected 0/0 boundaries`; next source work must add raw collective diagnostics and adapt matching to the current mutable `all_reduce_inplace` representation.
95. Metrics can legitimately lag during the first uncapped strict smoke even without a profiler. In the post-IR ROCm 7.2 run, vLLM counters initially stayed at zero while VRAM was committed and GPUs reached full utilization; later the same request reported `num_requests_running=1`, `prompt_tokens_total=431`, and generation tokens increasing, then completed validly. Treat early zero metrics as inconclusive during cold compile/pre-TTFT windows; confirm with logs, process activity, and later counters before killing a lane.
96. First-request patience is now a required methodology rule for the ROCm 7.2 post-IR dense lane. On `.30`, dry-run/no-rewrite, `replace=both`, `norm_only`, and `residual_only` all spent about `207-208s` in the first post-ready 431-token prefill/admission path at roughly `2.08 prompt TPS`; warm requests on the same backend returned to roughly `450-457 prompt TPS`. Do not reject these lanes from zero metrics or slow first-request behavior alone.
97. The mutable post-IR RMS/residual consumer source path is correctness-valid but not promotable. Overlay `postir_mutable_fusion_20260608` (`sha256:6c9b7d3208772d665434e584224b622c5080f870895c297271fefbb9a7977e5d`) now selects `2/2` mutable RowParallel boundaries and runs without topology, metadata, or dtype failures. Warm `.30` `c1_2000` decode was `49.559 TPS` for no-rewrite dry-run, `48.489` for `replace=both`, `45.264` for `norm_only`, and `44.291` for `residual_only`. Keep this as source infrastructure evidence, not a dense gate candidate.
98. Consumer-only fusion is not enough for the dense `65 TPS` gate. The post-IR custom op fuses work after the RowParallel allreduce, but it does not reduce the allreduce launch count or coalesce the small-message collective schedule. The data points back to fused collective support, collective coalescing, or a lower-overhead RowParallel communication primitive as the next serious source path.
99. The combined allreduce/allgather/RowParallel census is now the source-path index, not a promotion artifact. `runs/gfx906_allreduce_shape_census_20260608_172909/allreduce_shape_census.md` produced `133` rows and confirmed strong physical small-message allreduce evidence for MoE/Next `1x2048` traffic. Dense 27B still needs a better current-stack decode count trace: existing dense O3 traces under-attribute exact decode-shaped collectives, while graph benches identify real RowParallel boundary shapes such as `1x3696x5120` and `1x640x5120`.
100. PyNccl output-buffer allocation caching is rejected on the current ROCm 7.2 async dense winner. The `.20` run `qwen36_27b_decode_tiers_rocm72_mutable_rowpar_info_async_pynccl_cache_20260608_host20_20260608_173235` used the same winner envelope plus `pynccl_cached_output_20260607` and reached valid serving, but warm decode regressed to `52.147 TPS` on `c1_2000` and `46.119 TPS` on `c1_10000`, compared with the current `.20` winner at `56.479` and `53.573`. Allocation avoidance around PyNccl outputs is not enough; continue toward actual fused/small-message collective launch/count/coalescing work.
101. Runtime Python allreduce hooks are closed as the dense O3 async count method. High-level wrapping breaks graph capture with Dynamo rejecting the `torch.compiler.disable()`-wrapped `tensor_model_parallel_all_reduce` path, while PyNccl-only wrapping serves but only sees a `[431,5120]` batch-shaped escape at `0.0645` calls/token/rank. The regenerated census `gfx906_allreduce_shape_census_20260608_180854` has `134` rows and includes this limitation. Continue count/source work from pass/graph-level lowered collective visibility, not runtime Python wrappers.
102. Be patient with ROCm 7.2 O3 source lanes. The fresh `.30` post-IR collective census run `qwen36_27b_decode_tiers_rocm72_postir_collective_census_patience_20260608_host30_20260608_181900` stayed alive through several minutes of failed health probes before normal engine init, post-IR diagnostics, O3 compile, and CUDA graph capture completed. Zero metrics or failed curl health checks during this cold window are not enough to reject a lane.
103. The current dense winner graph already uses graph-visible in-place RowParallel collectives. The same post-IR census parsed `48` selection events: `40` selected `2/2` mutable `vllm.all_reduce_inplace.default` boundaries, all direct residual/RMS-adjacent and all fp16 `[s18,5120]` GEMM outputs; `8` events had `0/0` boundaries. Each TP rank accumulated `10` selected direct boundaries in the sampled compile logs. This confirms that PyNccl output allocation avoidance and consumer-only residual/RMS fusion are not enough. The next gate-relevant source work must reduce or accelerate the RowParallel collective itself.
104. The old dense RowParallel shape assumptions are superseded for the current `Qwen/Qwen3.6-27B` stack. The active config is `hidden_size=5120`, `num_hidden_layers=64`, `intermediate_size=17408`, `num_attention_heads=24`, `num_key_value_heads=4`, and `head_dim=256`. Current lowered graph evidence points to attention `LLMM1(5120x768) -> all_reduce_inplace(1x5120)`, gate/up `LLMM1(4352x5120)`, and MLP down `LLMM1(5120x2176) -> all_reduce_inplace(1x5120)`. Older `5120x640` and `5120x3696` rows are historical only.
105. Direct C++/HIP RCCL graph replay does not beat PyNccl/RCCL enough to clear the dense gate. The new `rccl_direct_allreduce_graph_bench_20260608.cpp` passed zero-error TP8 Tree/LL tests on `.20` and `.30`, but `1x5120` graph allreduce stayed at about `0.0486-0.0488 ms/call`, with seeded in-place about `0.0499 ms/call`. The language boundary is not the limiter; the primitive/schedule/count is.
106. Current actual-shape RowParallel boundaries are much smaller than the stale MLP-down estimate but still gate-relevant. Serving-equivalent LLMM1 plus allreduce on ROCm 7.2 measured `attn_5120x768` at `0.0661-0.0664 ms/call` and `mlp_down_5120x2176` at `0.0813-0.0822 ms/call` across `.20`/`.30`. That is about `0.147-0.148 ms/layer`, or roughly `9.4 ms/token` over `64` layers.
107. Corrected-shape NCCL env variants are not promotion candidates. Default-cleared/auto on `.20` slightly improved `attn_5120x768` but worsened `mlp_down_5120x2176`; Ring/LL on `.30` showed the same shape of result. Keep the current dense winner on Tree/LL and stop treating the old Ring/LL primitive win as extensible evidence for this serving path.
108. `wvSplitK` is unsafe on the corrected dense shape stack. Enabling `LLMM1_INCLUDE_WVSPLITK=1` for `5120x768`, `4352x5120`, `5120x2176`, and `31040x5120` crashed both `.20` and `.30` with a device-side assertion in `wvSplitK_hf_sml_` at `skinny_gemms.hip:530`. Close wvSplitK for this path unless its shape guard is fixed upstream/source-side first.
109. Corrected `D=2176,K=5120` gate/up fused SwiGLU is a real source lower-bound milestone. After patching the MEP to avoid the missing ROCm 7.2 `C10_HIP_KERNEL_LAUNCH_CHECK` macro, long two-method runs showed the interleaved LLMM1-style fused epilogue at `60.320 us` on `.20` and `59.840 us` on `.30`, versus incumbent LLMM1 plus `silu_and_mul` at `63.520` and `63.360 us`. This saves about `3.2-3.5 us/layer`, or roughly `0.20-0.23 ms/token`.
110. The corrected fused gate/up win is not a drop-in serving promotion. It depends on interleaved gate/up rows; the non-interleaved fused variants are slower. A serving implementation would need either extra packed interleaved weights, likely multiple GiB across all layers/ranks, or an in-place layout conversion plus custom activation handling for decode and prefill. Treat it as composable source evidence, while keeping the main gate path on RowParallel collective structure and graph scheduling.
111. Corrected-shape stream overlap is rejected. After patching `run_llmm1_allreduce_overlap_microbench_20260608.sh` to pass explicit shapes, TP8 Tree/LL overlap runs on `.20` and `.30` showed explicit stream-overlap slower than serial for `attn_out_5120x768`, `gate_up_4352x5120`, and `mlp_down_5120x2176`. The old partial-overlap signal came from a stale heavy `31040x5120` gate/up assumption. Do not build a simple overlap pass for the current dense graph.
112. Upstream AITER fused custom allreduce is compile-reachable on gfx906 but runtime-blocked by HIP IPC peer import. Probe artifacts are under `/usr/share/ollama/gfx906_aiter_probe` on `.20`, with local runners `run_aiter_custom_ar_probe_20260608.sh` and `aiter_custom_ar_multigpu_probe_20260608.py`. Probe-only source patches added `gfx906` to AITER's arch maps, guarded FP8 conversion helpers that require unavailable FP8 conversion instructions, forced the metadata buffer through the torch cached allocator, and explicitly called `hipDeviceEnablePeerAccess()` before `hipIpcOpenMemHandle()`. Single-process JIT then succeeded and `module_custom_all_reduce` returned `meta_size 5504`, but TP2 `(1,5120)` fp16 initialization still aborted on both ranks at `custom_all_reduce.cuh` `hipIpcOpenMemHandle(...) -> invalid device pointer`. Treat AITER custom allreduce / fused allreduce+RMS as blocked on the same gfx906 PCIe peer-memory substrate as the previous raw IPC probes, not on Python/vLLM wiring.
113. Replicating dense attention `o_proj` to trade the attention RowParallel allreduce for an all-gather is rejected. New microbench `microbenches/attention_replicated_oproj_bench_20260608.py` ran on `.20` TP8 at `/usr/share/ollama/kernel_labs/attention_repl_oproj_20260608_195419` with the dense winner NCCL envelope. Graph means were: current `LLMM1(5120x768)+allreduce(1x5120)` `0.062960 ms`, `all_gather(1x768->1x6144)` alone `0.065471 ms`, replicated full `LLMM1(5120x6144)` alone `0.085702 ms`, and combined all-gather plus replicated GEMV `0.176393 ms`. This is about `2.8x` slower than the current attention boundary before accounting for the extra replicated weight memory, so do not pursue replicated attention output projection as a dense gate path.
114. Interleaved SwiGLU serving integration is a mild pre-release exact-shape compute milestone. The first Python-branch and native-op integrations both served correctly but regressed to about `25 TPS`, showing that a microbench win can be erased by graph/runtime integration cost. Moving the N==1/N>1 branch fully inside the native op recovered winner-class performance on `.20`: run `qwen36_27b_decode_tiers_rocm72_mutable_rowpar_info_async_interleaved_swiglu_native_runtime_20260608_host20_20260608_211553` produced valid strict c1 `55.719 TPS`, `c1_2000` `56.800 TPS`, and `c1_10000` `53.908 TPS`. Repeat run `qwen36_27b_decode_tiers_rocm72_mutable_rowpar_info_async_interleaved_swiglu_native_runtime_repeat2_20260608_host20_20260608_213641` produced valid strict c1 `56.214 TPS`, `c1_2000` `56.906 TPS`, and `c1_10000` `54.012 TPS`. Host `.30` repeat `qwen36_27b_decode_tiers_rocm72_mutable_rowpar_info_async_interleaved_swiglu_native_runtime_repeat_20260608_host30_20260608_213547` was lower but still near its same-family baseline: valid strict c1 `53.579 TPS`, `c1_2000` `54.249 TPS`, and `c1_10000` `51.499 TPS`. Promote only the methodology/source lesson: keep decode-shape branches out of Python/model code and make source experiments graph-stable before judging the kernel idea. The gate-relevant path remains RowParallel collective structure and scheduling.
115. Public RCCL send/recv composition is rejected as the gfx906 small-message collective replacement. New MEP `microbenches/rccl_p2p_allreduce_graph_bench_20260609.cpp` and runner `run_rccl_p2p_allreduce_graph_bench_20260609.sh` compared generic `ncclAllReduce`, grouped full-buffer send/recv allgather-sum, and explicit ring reduce-scatter/allgather on exact TP8 decode payloads. With the safe winner envelope `NCCL_P2P_DISABLE=1`, `.20` graph replay measured `1x5120` generic `0.049656 ms`, allgather-sum `0.207861 ms`, ring `0.203515 ms`; `1x2048` generic `0.044924 ms`, allgather-sum `0.118955 ms`, ring `0.194659 ms`. `.30` reproduced the same shape: `1x5120` generic `0.049754 ms`, allgather-sum `0.206677 ms`, ring `0.204213 ms`; `1x2048` generic `0.043924 ms`, allgather-sum `0.117044 ms`, ring `0.190827 ms`. Enabling RCCL P2P for this MEP produced HIP `invalid argument` / `invalid device pointer` failures and a stuck container, which was cleaned. Conclusion: do not replace generic RCCL allreduce with public send/recv composition for dense c1. A useful custom collective must be deeper than public P2P scheduling or must reduce the boundary count.
116. RCCL `AllReduceWithBias` is source-valid enough to serve but remains a performance rejection on the current dense winner. The acc-enabled ROCm 7.2 overlay `rccl_overlay_721_gfx906_acc_minimal_f16f32_20260608` exports and executes `ncclAllReduceWithBias` for TP8 fp16; the C probe passes at FP16 tolerance `0.02` with max abs error `0.0150146` across `1x5120` and larger rows. Serving overlay `rccl_ar_bias_plus_mutable_rowpar_20260608` did hit the `path=rccl` custom op at decode shapes `(1,5120)`, `(2,5120)`, `(4,5120)`, etc., and strict output stayed valid. Performance regressed: `.20` full run `qwen36_27b_decode_tiers_rocm72_mutable_plus_rccl_ar_bias_fulltiers_h20_20260608_host20_20260608_1156_ar_bias_plus_mutable_fulltiers_h20` produced `c1_10000` `49.809 TPS`, and repeat `..._repeat_h20_...` produced `49.365 TPS`, versus the current `.20` mutable/interleaved winner band around `53.6-54.0 TPS`. Do not keep debugging this as a gate path unless RCCL-level profiling shows a way to make the acc path faster than generic allreduce plus existing fused consumers.
117. `max_num_seqs=8` is rejected for the current ROCm 7.2 mutable RowParallel + interleaved SwiGLU dense winner despite a valid patient cold start. Host `.30` maxseq8 was slightly above its same-family maxseq4 repeat (`54.018` strict c1, `54.635` c1_2000, `51.917` c1_10000), so `.20` was retested with patience through O3 compile, graph capture, and uncapped strict thinking. The `.20` run `qwen36_27b_decode_tiers_rocm72_mutable_rowpar_info_async_interleaved_swiglu_maxseq8_h20_20260608_host20_20260608_222059` completed cleanly but landed at strict c1 `55.679`, c1_2000 `56.724`, and c1_10000 `53.799`, below the maxseq4 `.20` repeat `56.214` / `56.906` / `54.012`. Keep `max_num_seqs=4` in the dense decode winner envelope. `max_num_seqs=1` remains a hard config rejection on this hybrid attention/mamba stack because KV-cache profiling hits the layout ambiguity assertion for shape `[2, 2, 400, 1, 256]`.
118. Current-stack invalid upper-bound split confirms the dense `65 TPS` gate is primarily RowParallel/allreduce, not raw model compute or logits/sampling. On the same ROCm 7.2 TP8 mutable RowParallel + INFO + async + interleaved SwiGLU base, fixed-token-only removed sampler/logits work but kept normal allreduces and reached only `.20` `c1_2000` `59.080` and `c1_10000` `55.898` backend TPS. Skip-allreduce-only kept normal sampling/logits but removed TP allreduces and reached `.30` `c1_2000` `81.940` and `c1_10000` `75.081` backend TPS. The combined skip-allreduce plus fixed-token upper bound previously reached `.20` `c1_10000` `82.448` and `.30` `79.779`. These are correctness-invalid upper bounds, but they show the current stack has enough raw decode headroom and that the next serious source path must replace, reduce, or coalesce RowParallel collectives while preserving correctness. Logits/sampling cleanup is secondary unless it is stacked after a collective improvement.
119. TP4 is rejected again on the current ROCm 7.2 dense stack after a patient full-tier run. Host `.20` GPUs `0,1,2,3` served TP4 with mutable RowParallel + INFO + async + interleaved SwiGLU and completed the standard tiers in `qwen36_27b_decode_tiers_rocm72_mutable_interleaved_tp4_h20_20260608_231048_host20_20260608_231048`: strict c1 was valid but only `39.520 TPS`, `c1_2000` was `39.676 TPS`, and `c1_10000` was `38.331 TPS`. The first uncapped smoke had slow admission/prefill, but warm forced tiers recovered normal prefill around `279 prompt TPS` while decode remained low. This closes TP4 as an impatience-revisit path for dense single-request decode; the gate remains a TP8 RowParallel collective structure problem.
    Repeat `.20` run
    `qwen36_27b_decode_tiers_rocm72_mutable_interleaved_tp4_valid_h20_20260609_host20_20260609_1322`
    used the same current-stack overlays and patient harness and reproduced the
    rejection: strict smoke valid at `39.418 TPS`, `c1_2000` `39.596 TPS`,
    and `c1_10000` `38.245 TPS`. Reducing TP communication to TP4 does not pay
    for the larger per-rank GEMV on this model/config.
120. ROCm 7.2 does not rescue PyTorch symmetric-memory fused collectives on gfx906. Tiny TP8 probes on `.20` and `.30` using `qwen36-gfx906-c1-topk8-fastpath-rocm72-experimental:latest` reported `torch 2.11.0+rocm7.2`, HIP `7.2.26015`, `torch.ops.symm_mem.fused_matmul_reduce_scatter` present, and `_symmetric_memory.enable_symm_mem_for_group("0")` succeeded on all ranks. Both hosts then entered `private_fused_matmul_reduce_scatter` and stayed there until the 300s timeout, with all eight GPUs at `100%` and essentially no VRAM use. Runs: `symm_mem_rocm72_tp8_tiny_host20_20260609` and `symm_mem_rocm72_tp8_tiny_host30_20260609`. Continue treating upstream `AsyncTPPass` / `torch.ops.symm_mem` fused GEMM-collective as substrate-blocked on gfx906; any fused collective source path must replace that primitive rather than just relaxing vLLM's CUDA gates.
121. Selective RowParallel skip attribution narrows the dense gate target to
    the MLP/down-projection collective. The correctness-invalid selective skip
    overlay removed only matching `RowParallelLinear` allreduces while keeping
    the rest of the current ROCm 7.2 TP8 mutable RowParallel + INFO + async +
    interleaved SwiGLU stack. Attention-only skip stayed baseline-class:
    `.20` `c1_2000` `59.103`, `c1_10000` `55.931`; `.30` `c1_2000`
    `56.655`, `c1_10000` `53.708`. MLP/down-proj skip was much larger and
    near the gate: `.30` `c1_2000` `65.159`, `c1_10000` `61.311`; `.20`
    `c1_2000` `68.000`, `c1_10000` `63.763`. These runs are invalid for
    quality because layer reductions are removed, but they show that attention
    `o_proj` is not the next high-value target. The primary source branch
    should focus on correctness-preserving MLP/down-proj RowParallel collective
    reduction, coalescing, or semantic restructuring first; logits/sampling and
    attention-only work are secondary.
122. Scaled local MLP/down-proj partials are rejected as an output strategy but
    reinforce the MLP collective diagnosis. The diagnostic overlay
    `rowparallel_mlp_approx_20260609` initially failed because runtime
    string/regex classification entered the O3 compiled forward path; after
    moving classification to import/init time, patient smoke probes served.
    Scaling all MLP local partials on `.20` produced backend decode `60.975`
    TPS but filled the context with no close-think and only `7` answer chars.
    Keeping first/last `4` MLP layers exact on `.30` produced `61.890` backend
    TPS but was also strict-invalid with only `7` answer chars. Do not pursue
    approximate local partials for quality. The useful signal is that the
    speed response remains in the same near-gate band, so the primary source
    path must make the MLP/down-proj reduction exact and cheaper rather than
    removing or approximating it.
123. Current-stack MSCCL++ is rejected for dense decode. Because the old
    MSCCL++ rejection used an earlier/stale dense stack, it was retested on the
    ROCm 7.2 TP8 mutable RowParallel + INFO + async + native-runtime
    interleaved SwiGLU winner envelope with `NCCL_ALGO`, `NCCL_PROTO`, and
    `NCCL_NTHREADS` cleared, `NCCL_P2P_DISABLE=1`,
    `NCCL_MAX_NCHANNELS=1`, `RCCL_MSCCL_FORCE_ENABLE=1`,
    `RCCL_MSCCLPP_ENABLE=1`, and `RCCL_MSCCLPP_THRESHOLD=65536`.
    Strict smokes were valid, but sustained decode regressed: `.20`
    `c1_10000` `49.970` and `.30` `49.685`, versus the current winner band
    of `.20` `54.012` and `.30` `51.499`. Do not spend more primary time on
    RCCL/MSCCL env routing for this gate; the needed improvement is still
    an exact MLP/down-proj collective/source change.
124. Corrected current-shape PyNccl graph replay supports the same MLP-first
    source target. Runs
    `pynccl_rowparallel_graph_bench_host20_20260609_011056_corr_h20_localdocker`
    and
    `pynccl_rowparallel_graph_bench_host30_20260609_011056_corr_h30_localdocker`
    replayed TP8 Tree/LL exact matmul+allreduce boundaries with zero numerical
    error. On `.20`, graph `1x2176x5120` was `0.357870 ms/call` versus
    attention `1x768x5120` `0.089483`; `18x2176x5120` was `0.529519` versus
    `18x768x5120` `0.241773`. On `.30`, the corresponding values were
    `0.381338` versus `0.092627`, and `0.545556` versus `0.242423`. Large
    `431`-row rows are allreduce-payload dominated, but MLP still carries the
    extra local GEMM cost. This closes high-level Python/allocation cleanup as
    the next serious path and keeps the gate work focused on exact
    MLP/down-proj RowParallel collective count/cost/scheduling.
125. `RCCL_ENABLE_CONTEXT_TRACKING=1` is rejected for the current dense gate.
    AMD documents context tracking as a possible performance knob, so it was
    checked on the corrected TP8 Tree/LL RowParallel graph replay path in
    `pynccl_rowparallel_graph_bench_host20_20260609_0138_context_h20` and
    `pynccl_rowparallel_graph_bench_host30_20260609_0138_context_h30`.
    `.20` stayed baseline-class or slightly slower (`1x2176x5120` graph
    `0.358921` versus prior `0.357870`; `18x2176x5120` `0.531209` versus
    `0.529519`). `.30` had a noisy single-row MLP improvement but regressed
    the more relevant `18`/`128` row graph paths (`18x2176x5120` `0.555751`
    versus `0.545556`; `128x2176x5120` `1.458471` versus `1.451595`). Do not
    spend a full serving lane on RCCL context tracking for this stack.
126. Bundled RCCL/MSCCL XML allreduce is not a deployable shortcut for the
    dense gate on this ROCm 7.2 gfx906 stack. `NCCL_ALGO=MSCCL` is invalid in
    this RCCL build; omitting `NCCL_ALGO` lets MSCCL initialize, but the
    default PyNccl 4-byte out-of-place warmup can hit XMLs that declare only
    `inplace=1`. After adding a benchmark-only in-place warmup patch and then
    staging a custom `MSCCL_ALGO_DIR` with only
    `allreduce-allpairs-8n-ll-32tb.xml` and `minBytes=8192`, TP8 still
    SIGSEGVs on the first real `1x2176x5120` RowParallel in-place allreduce.
    Runs:
    `pynccl_rowparallel_graph_bench_host20_20260609_0210_msccl_xml32_inplace_noalgo_h20`,
    `..._0220_msccl_xml32_inplace_warmfix_h20`,
    `..._0240_msccl_progress_ch32_h20`, and
    `..._0250_msccl_custom_min8192_h20`. Treat built-in MSCCL XML as
    substrate-blocked unless we generate/debug a gfx906-specific XML or patch
    RCCL/MSCCL itself.
127. Serving-equivalent in-place Tree/LL RowParallel graph replay is stable
    but does not reveal hidden allocation headroom. The benchmark was patched
    with `ROWPAR_GRAPH_INPLACE_AR=1` to match the mutable RowParallel serving
    path and rerun on `.20`/`.30` in
    `pynccl_rowparallel_graph_bench_host20_20260609_0300_tree_ll_inplace_corrected_h20`
    and
    `pynccl_rowparallel_graph_bench_host30_20260609_0300_tree_ll_inplace_corrected_h30`.
    `.20` rank-mean graph values were `1x768` `0.089541`,
    `1x2176` `0.356334`, `18x768` `0.241364`, `18x2176` `0.530438`,
    `128x768` `1.238121`, `128x2176` `1.437297`, `431x768` `4.001533`,
    and `431x2176` `4.371874` ms/call. `.30` was similar:
    `1x768` `0.094553`, `1x2176` `0.362963`, `18x768` `0.253763`,
    `18x2176` `0.533054`, `128x768` `1.265370`, `128x2176` `1.454057`,
    `431x768` `4.039896`, and `431x2176` `4.414282`. This confirms the
    current source path already uses the low-allocation in-place form; the
    remaining gate work must reduce/coalesce/replace exact RowParallel
    collective structure, especially MLP/down-proj, rather than chase PyNccl
    output allocation cleanup.
128. Exact MLP TP4x2 inside global TP8 is correctness-valid but performance
    rejected. Overlay
    `overlays/qwen2_moe_mlp_tp4x2_20260609/qwen2_moe.py` creates two local MLP
    TP4 groups `[0,1,2,3]` and `[4,5,6,7]`, builds Qwen dense MLP projections
    under those groups, and allreduces MLP down-proj only inside the local TP4
    group. After fixing `logger.info_once` rank-list hashing and staging the
    native SwiGLU extension on `.30`, strict smokes were valid on two hosts,
    but decode collapsed: `.20` `27.034` backend TPS, `.30` `26.302` backend
    TPS. The lower-bound microbench's local allreduce win is outweighed in
    serving by duplicated MLP gate/up and down-proj compute. Do not pursue
    TP4x2 replicated MLP as a gate path; keep TP8 compute partitioning and
    attack the MLP/down-proj RowParallel collective boundary directly.
129. Host-mapped shared-memory collectives are correctness-valid but too slow
    for the dense gate. `gfx906_peer_allreduce_bench_20260606.cpp` now has
    POSIX-shm + `hipHostRegisterMapped` multi-process modes that avoid HIP IPC:
    `mapped_host_mp` uses a rank-0 broker, while `mapped_host_allrank_mp` has
    every rank sum from mapped host slots. Both passed on `.20` and `.30`, but
    TP8 `1x5120` was `267-268 us/call` for the broker and `240-241 us/call`
    for all-rank sum, versus the current RCCL Tree/LL floor near `49 us`.
    This closes host-mapped exchange as a serving substrate. The remaining
    serious source path must keep TP8 compute partitioning and use a lower-level
    collective primitive, graph-visible MLP/down-proj boundary change, or fused
    GEMM-plus-reduce path with a faster non-IPC exchange mechanism.
130. Current-shape LLMM1 remeasurement rejects reviving stale `rpb4` routing.
    After correcting the lowered dense graph shapes, the active ROCm 7.2 TP8
    path uses gate/up `4352x5120`, MLP down `5120x2176`, and attention output
    `5120x768`. Fresh `.20` and `.30` stack compares showed `LLMM1_rpb2` best
    for all three current shapes: `.20` `59.840/33.600/17.760 us` and `.30`
    `59.680/33.439/17.600 us`, respectively. The old selective `rpb4` hint was
    tied to obsolete `5120x3696` MLP-down assumptions and should stay closed.
    Do not spend more serving lanes on rows-per-block retuning unless a new
    profile exposes a different current shape; focus on exact MLP/down-proj
    collective/source structure instead.
131. RCCL 7.2.1 gfx906 one-slice opt-in is correctness-clean but performance
     rejected for the dense gate. Full patched RCCL overlays built successfully
     on `.20` and `.30` from
     `patches/rccl_721_gfx906_one_slice_optin_20260609.patch`, with
     `RCCL_GFX906_ONE_SLICE_ENABLE=1` default-off and symbol checks passing for
     `ncclAllReduce`, `ncclAllReduceWithBias`, and `ncclGetVersion=22707`.
     Direct TP8 Tree/LL allreduce graph replay showed no useful primitive-floor
     win: `.20` `1x5120` graph out was effectively neutral
     (`0.048762 -> 0.048755 ms`), but `128x5120` regressed
     (`1.140974 -> 1.143278 ms`) and `431x5120` regressed
     (`3.740560 -> 3.751531 ms`). `.30` matched the same pattern:
     `1x5120` tiny noise win (`0.049266 -> 0.049157 ms`), while `128x5120`
     regressed (`1.137023 -> 1.141505 ms`) and `431x5120` regressed
     (`3.733625 -> 3.745480 ms`). Do not spend RowParallel or serving lanes on
     this overlay; keep the source path focused on exact TP8 MLP/down-proj
     collective/source changes with a positive lower-bound signal.
132. RCCL Tree/LL UpDown is rejected for gfx906 TP8 dense decode. Patch
     `patches/rccl_721_tree_ll_updown_experiment_20260609.patch` changed the
     Tree/LL AllReduce device dispatch from `runTreeSplit` to
     `runTreeUpDown` and added the missing `directRecvReduceCopy` wrapper
     needed for `ProtoLL` compilation. Overlays built cleanly on `.20` and
     `.30`, with symbol checks passing for `ncclAllReduce`,
     `ncclAllReduceWithBias`, and `ncclGetVersion=22707`. Direct TP8
     Tree/LL allreduce graph replay was exact, but slower on the decode and
     prefill shapes that matter. `.20` `1x5120` graph out regressed
     `0.048762 -> 0.050315 ms`, `128x5120` regressed
     `1.140974 -> 1.244298 ms`, and `431x5120` regressed
     `3.740560 -> 4.076114 ms`. `.30` matched the pattern:
     `0.049266 -> 0.050387 ms`, `1.137023 -> 1.249448 ms`, and
     `3.733625 -> 4.086535 ms`. Do not run serving tiers with this overlay.
     The split Tree/LL schedule is better for this PCIe gfx906 TP8 path.
133. Public RCCL send/recv composition is rejected as a small-message
     allreduce substrate. `run_rccl_p2p_allreduce_graph_bench_20260609.sh`
     compared normal `ncclAllReduce` against two exact public send/recv
     compositions: allgather-then-sum and ring reduce-scatter/allgather. Both
     were graph-capturable and exact on `.20`/`.30`, but both were far slower
     than Tree/LL. At `1x5120`, `.20` graph replay was `0.049565 ms` for
     `ncclAllReduce`, `0.206539 ms` for allgather-sum, and `0.203032 ms` for
     send/recv ring. `.30` was `0.050020 ms`, `0.207062 ms`, and
     `0.208703 ms`. This closes public send/recv composition as a serving
     backend; future small-message source work should stay inside RCCL's
     generated collective machinery or use a graph-level RowParallel boundary
     change with a positive lower-bound signal.
134. RCCL symmetric-memory AllReduce is not a viable TP8 gfx906 PCIe gate path
     on the current hosts. Full symmetric-kernel generation reached the
     `librccl.so` link step on `.30` and was still linking after 50+ minutes;
     a narrowed f16 AllReduce-only generator is the practical iteration loop
     and links `librccl.so` in about `9.7 s`. Stock ROCm RCCL hard-disables
     `ncclCuMemEnable()` on HIP, so `ncclCommWindowRegister()` returns null
     unless a local explicit opt-in patch is applied. With
     `patches/rccl_721_sym_f16_allreduce_rocm_cumem_optin_20260609.patch`,
     `NCCL_LOCAL_REGISTER=1`, `NCCL_CUMEM_ENABLE=1`, and 2 MiB padded backing
     allocations, TP8 P2P-off runs get non-null windows and exact graph replay
     results, but `Check P2P Type isAllDirectP2p 0` and no
     `AllReduce [Symmetric]` tuning line appears. That means the normal RCCL
     Tree/LL path is still used. P2P-on plus cuMem fails on this PCIe topology:
     RCCL reports peer-access failures across devices and segfaults before any
     timing rows. Cleanup also remains unsafe after ROCm cuMem/window
     registration; skipping window deregistration and HIP frees still crashes,
     while skipping `ncclCommDestroy` exits cleanly, isolating a communicator
     teardown issue. Reject symmetric-memory RCCL for the TP8 dense gate unless
     a future hardware lane has full direct P2P across all ranks. Continue with
     TP8-preserving RowParallel boundary fusion/coalescing or a collective
     source path that does not require direct peer loads.
135. Do not reopen `ncclAllReduceWithBias` as a dense gate path without new
     RCCL-level evidence. The 2026-06-09 f16-only recheck intentionally rebuilt
     a narrow Tree/LL accumulator overlay, but initially missed the matching
     host-side `rcclIsArchSupportedForFunc()` gfx906 guard. That reproduced the
     prior host-planning failure: the bias call logged a non-null `acc` pointer
     and crashed before selected-algorithm logging, while `world_size=1`
     returned the exact missing-bias error magnitude. The existing 2026-06-08
     known-good source patch already fixes both device generation and host arch
     validation, passes the direct TP8 C probe at f16 tolerance `0.02`, and was
     rejected only because serving performance regressed (`c1_10000` about
     `49.4-49.8 TPS` versus the mutable/interleaved winner band around
     `53.6-54.0 TPS`). Keep the primitive as a substrate milestone, not the
     active gate-clearing branch.
136. Do not lower gfx906 Tree/LL AllReduce below RCCL's 4-warp correctness
     floor by only overriding host `nThreads`. Patch
     `patches/rccl_721_gfx906_tree_ll_nthreads128_optin_20260609.patch`
     built successfully as both a full `.20` overlay and a narrowed `.30`
     f16 Tree/LL overlay. With the opt-in disabled, the narrowed overlay stayed
     exact through the direct TP8 graph replay rows observed so far:
     `1x5120 graph_out=0.048153 ms`, `8x5120=0.107699 ms`, and
     `64x5120=0.584640 ms`, all with zero error. With
     `RCCL_GFX906_LL_NTHREADS_128=1`, the full overlay failed before any timing
     row: ranks 1-7 reported GPU memory access faults and rank 0 stayed busy.
     Source inspection explains the failure: RCCL's kernel prologue and
     Tree/LL split path assume the normal 4-warp gfx906 launch structure, while
     the opt-in changed only the collective work's thread count. Treat the
     256-thread Tree/LL worker floor as a correctness constraint unless a deeper
     device-kernel/planner rewrite is done. Continue gate work through
     RowParallel boundary coalescing/fusion, not by shrinking Tree/LL blocks.
137. Legal RowParallel coalescing would have enough primitive-level headroom to
     matter, but the current dense graph still has real dependencies between
     the per-layer collectives. Direct TP8 Tree/LL graph replay on `.20`
     (`runs/rccl_direct_allreduce_graph_bench_host20_20260609_direct_ar_coalesce_upper_h20`)
     measured zero-error `graph_out` costs of `0.048380 ms` for `1x5120`,
     `0.055876 ms` for `2x5120`, `0.073880 ms` for `4x5120`, and
     `0.107948 ms` for `8x5120`. Two independent `1x5120` calls therefore
     cost about `0.0968 ms`, while one `2x5120` call costs about `0.0559 ms`.
     This confirms that batching multiple same-shape reductions is a real
     lower-bound opportunity. It is not directly promotable because attention
     and MLP RowParallel outputs are separated by residual/RMS and layer
     dependencies. Future source work should only use this headroom through a
     legal semantic schedule, static-SP/partitioned-hidden-state design, or a
     fused primitive whose consumer dependency is preserved.
138. Use the serving-equivalent LLMM1 graph replay for dense RowParallel
     boundary estimates. The generic PyTorch `torch.mm` RowParallel graph bench
     remains valid for RCCL/allreduce behavior, but it overstates the MLP
     compute portion on the promoted vLLM image. The corrected `.20` LLMM1
     graph replay
     (`runs/llmm1_pipelined_rowparallel_host20_20260609_llmm1_graph_actual_h20`)
     used TP8 Tree/LL, P2P-off, `NCCL_NTHREADS=128`, and `64` calls per graph
     replay. It measured exact `graph_full_llmm1_allreduce` costs of
     `0.057309 ms` for `attn_5120x768` and `0.073390 ms` for
     `mlp_down_5120x2176`; in-place was `0.057258 ms` and `0.073955 ms`.
     The current exact per-layer RowParallel boundary estimate is therefore
     about `0.1307 ms`, or `8.36 ms/token` across `64` layers. MLP/down-proj
     remains the first target, but MLP-only improvement is near-gate rather
     than decisively gate clearing; expect to stack it with attention,
     sampler/logits cleanup, or a broader legal schedule change.
139. `HSA_FORCE_FINE_GRAIN_PCIE=1` does not make P2P-on RCCL viable on the
     current 8x gfx906 hosts. AMD documents this variable as the PCIe P2P
     enable switch when GPUs and large BAR placement support peer access, so it
     was tested as a bounded `.20` direct RCCL graph replay with
     `NCCL_P2P_DISABLE=0`, Tree/LL, one channel, and `NCCL_NTHREADS=128`
     (`runs/rccl_direct_allreduce_graph_bench_host20_20260609_hsa_fine_pcie_p2pon_h20`).
     The run produced no timing rows; rank logs showed `HIP failure:
     'invalid argument'` and unhandled RCCL CUDA/system errors at the allreduce
     call. Keep `NCCL_P2P_DISABLE=1` in promoted dense configs. This reinforces
     the HIP IPC, AITER, and RCCL symmetric-memory rejections: direct peer
     access is not a safe substrate for this topology.
140. Exact greedy/logits cleanup does not stack with the current ROCm 7.2 dense
     winner. The invalid fixed-token-only lane remains useful attribution
     evidence, but the legal current-stack retest on `.20`
     (`runs/qwen36_27b_decode_tiers_rocm72_mutable_interleaved_greedy_current_h20_20260609_host20_20260609_1258`)
     regressed versus the winner: strict uncapped smoke was valid at
     `55.873` backend TPS, `c1_2000` was `51.606`, and `c1_10000` was
     `49.058` versus the `.20` winner's `54.012`. Keep sampler/logits cleanup
     closed until a later profile shows it has become material after
     RowParallel collective work; primary source effort stays on exact
     MLP/down-proj RowParallel structure.
141. Do not coalesce the two mutable RowParallel collectives seen in one
     current-stack post-IR graph event. The detailed census sample in
     `runs/qwen36_27b_decode_tiers_rocm72_postir_collective_census_patience_20260608_host30_20260608_181900/docker_ready.log`
     shows the first `all_reduce_inplace` feeding `add_70` and its RMS/MLP
     chain before the second `rocm_unquantized_gemm_2 -> all_reduce_inplace`
     boundary feeds `add_119`. They are same-shape and same graph event, but
     not independent. The `2x5120` direct coalescing win remains a lower bound
     for a different schedule, not a legal rewrite for the current graph.
142. Hidden-sharded decode is not a local RowParallel patch. Avoiding the
     all-gather after reduce-scatter only works if the following projections
     are redesigned too, because the current column-parallel Qwen3.6
     projections expect a full hidden vector on every TP rank. Otherwise the
     collective cost simply moves from the RowParallel boundary into the next
     projection. Treat hidden-sharded decode as a full layer-schedule rewrite
     that needs its own layer-level microbench before any serving lane.
143. The hidden-sharded next-projection lower bound is negative, so do not
     spend serving lanes on hidden-sharded dense decode for this gate. New
     microbench
     `llmm1_hidden_shard_projection_bench_20260609.py` compared the current
     full-hidden `LLMM1(weight[out,5120], hidden[1,5120])` against a
     hypothetical hidden-sharded consumer
     `LLMM1(weight_shard[out,640], hidden_shard[1,640]) + allreduce(1,out)`
     on `.20` TP8 Tree/LL/P2P-off/NTHREADS=128. Gate/up, the most favorable
     current projection, was still slower: `0.056332 ms` versus `0.052825 ms`
     graph. `2048x5120` was `1.916x` slower and `12x5120` was `18.279x`
     slower. The gather fallback was `2.139x-24.967x` slower. Hidden-sharding
     moves the cost into the next projection instead of removing it; keep the
     primary source path on TP8 RowParallel MLP/down-proj collective/boundary
     work.
144. RCCL Tree/LL split-ratio tuning is rejected for the dense TP8 decode
     gate. The patient `.30` stock control completed exact direct graph replay
     with `graph_out` means of `0.0481479 ms` for `1x5120`,
     `0.1076725 ms` for `8x5120`, `0.584640875 ms` for `64x5120`,
     `1.13025 ms` for `128x5120`, and `3.71013875 ms` for `431x5120`.
     Two ROCm 7.2 RCCL source overlays were built on `.20` for
     `AllReduce TREE LL Sum f16`: `3 reduce / 1 broadcast` and
     `1 reduce / 3 broadcast`. Both were exact and slower than stock:
     3/1 measured `0.05439125`, `0.155148`, `0.978794375`, and
     `1.91732875 ms` for `1/8/64/128x5120`; 1/3 measured `0.055965525`
     and `0.158567 ms` for `1/8x5120`. Do not spend serving lanes on
     split-ratio overlays; the current effective `2 reduce / 2 broadcast`
     Tree/LL structure is locally best. The remaining positive path is at the
     RowParallel boundary, not inside single-call Tree/LL warp allocation.
145. The MiniMax Lamport fused allreduce/RMS path in
     `ai-infos/vllm-gfx906-mobydick` is not a direct dense TP8 substrate for
     these hosts. It is valuable as a design reference, but the implementation
     shares workspace buffers through CUDA IPC style peer-memory handles with
     lazy peer access. That is the same substrate class already rejected by HIP
     IPC, AITER custom allreduce, PyTorch symmetric memory, RCCL symmetric
     memory, and P2P-on tests on the current PCIe-only gfx906 topology. Do not
     transplant it into Qwen3.6 serving until a standalone peer-import
     microbench passes on the target hosts; keep the near-term source path on
     legal RowParallel boundary work using functioning RCCL/LLMM1 primitives.
146. `NCCL_MAX_NCHANNELS=4` is promoted as a small current-stack dense TP8
     serving milestone. It was retested patiently against the ROCm 7.2 mutable
     RowParallel + INFO + async + native-runtime interleaved SwiGLU winner with
     only max channels changed from `1` to `4`. On `.20`, run
     `qwen36_27b_decode_tiers_rocm72_mutable_interleaved_maxch4_h20_20260609_host20_20260609_1438`
     produced valid strict c1 `56.670`, `c1_2000 57.408`, and `c1_10000
     54.444` backend TPS versus the one-channel `.20` repeat at `56.214`,
     `56.906`, and `54.012`. On `.30`, run
     `qwen36_27b_decode_tiers_rocm72_mutable_interleaved_maxch4_h30_20260609_host30_20260609_1438`
     produced valid strict c1 `54.213`, `c1_2000 54.857`, and `c1_10000
     52.081` versus the same-family one-channel `.30` control at `53.579`,
     `54.249`, and `51.499`. This is not gate-scale, but it is a consistent
     same-host improvement, so make `NCCL_MAX_NCHANNELS=4` the current dense
     serving envelope while continuing source work on RowParallel boundaries.
147. Forcing `NCCL_MIN_NCHANNELS=4` on top of `NCCL_MAX_NCHANNELS=4` is a
     further small sustained-decode promotion. On `.20`, run
     `qwen36_27b_decode_tiers_rocm72_mutable_interleaved_maxch4_minch4_h20_20260609_host20_20260609_1458`
     produced valid strict c1 `56.885`, `c1_2000 57.554`, and `c1_10000
     54.599` backend TPS versus maxch4-only at `56.670`, `57.408`, and
     `54.444`. On `.30`, run
     `qwen36_27b_decode_tiers_rocm72_mutable_interleaved_maxch4_minch4_h30_20260609_host30_20260609_1458`
     produced valid strict c1 `53.936`, `c1_2000 54.908`, and `c1_10000
     52.156` versus maxch4-only at `54.213`, `54.857`, and `52.081`. The
     `.30` strict decode TPS is lower, but that uncapped response generated
     `4428` tokens versus `3251` in the maxch4-only strict run; the fixed
     c1_2000/c1_10000 tiers improved on both hosts. For the sustained dense
     gate, use `NCCL_MIN_NCHANNELS=4` and `NCCL_MAX_NCHANNELS=4` as the
     current baseline.
148. The refreshed current-winner dense TP8 profile keeps the source path on
     collectives. Run
     `qwen36_27b_c1_rocprof_rocm72_mutable_interleaved_minmax4_h20_20260609_20260609_145834`
     profiled the promoted `.20` ROCm 7.2 mutable RowParallel plus
     native-runtime interleaved SwiGLU baseline with
     `NCCL_MIN_NCHANNELS=4` and `NCCL_MAX_NCHANNELS=4`. The capped `c1_768`
     request ran fully inside the rocprof active window and emitted a valid
     `92.4 MB` kernel trace with `819750` used rows. Under instrumentation it
     scored `20.572` backend decode TPS, so absolute throughput is diagnostic
     only. Aggregate filtered kernel attribution is the important evidence:
     NCCL `19.3097s` (`67.08%`, `25.143 ms/token` summed across traced GPU
     kernel time), LLGemm `3.9732s` (`13.80%`, `5.173 ms/token`), the
     interleaved SwiGLU native kernel `2.9190s` (`10.14%`,
     `3.801 ms/token`), attention `0.2523s` (`0.88%`), and
     GDN/linear-attention `0.4274s` (`1.48%`). Per-worker row counts were
     imbalanced in the rocprof export, so do not claim a tail-rank diagnosis
     from this run. The robust conclusion is that the remaining gate-scale
     path is still TP8 collective count/latency or a legal RowParallel
     boundary rewrite; another isolated MLP microkernel tweak is unlikely to
     clear the dense `65 TPS` gate by itself.
149. The dense TP8 NCCL channel-count knee is closed around four channels for
     the current stack. After the min/max=4 promotion, `.20` min/max=2 run
     `qwen36_27b_decode_tiers_rocm72_mutable_interleaved_minmax2_h20_20260609_host20_20260609_152229`
     produced valid strict c1 `56.346`, `c1_2000 57.024`, and `c1_10000
     54.064` backend TPS, below the `.20` min/max=4 winner at `56.885`,
     `57.554`, and `54.599`. `.30` min/max=8 run
     `qwen36_27b_decode_tiers_rocm72_mutable_interleaved_minmax8_h30_20260609_host30_20260609_152229`
     produced valid strict c1 `54.637`, `c1_2000 55.289`, and `c1_10000
     52.503`; it is not a global promotion and does not justify moving away
     from four channels. Keep `NCCL_MIN_NCHANNELS=4` and
     `NCCL_MAX_NCHANNELS=4` as the dense sustained-decode baseline, then spend
     serious effort on collective/boundary source changes. The capture harness
     now waits for post-response `/metrics` counters to settle so long
     in-flight decode requests are not scored from lagging metrics snapshots.
150. `ncclAllReduceWithBias` is rejected again under the final min/max=4 dense
     winner envelope, including a new exact MLP-side implementation. The fair
     current-stack attention-side recheck on `.20`,
     `qwen36_27b_decode_tiers_rocm72_mutable_interleaved_arbias_minmax4_h20_20260609_host20_20260609_1542_arbias_minmax4_h20`,
     produced valid strict c1 `52.031`, `c1_2000 53.776`, and `c1_10000
     47.429` backend TPS, far below the `.20` min/max=4 winner at `56.885`,
     `57.554`, and `54.599`. A disabled-by-default MLP AR-bias hook was added
     to `overlays/qwen2_moe_interleaved_swiglu_20260608/qwen2_moe.py` via
     `VLLM_GFX906_MLP_AR_BIAS_ENABLE=1`; the `.30` source lane
     `qwen36_27b_decode_tiers_rocm72_mlp_arbias_minmax4_h30_20260609_host30_20260609_1546_mlp_arbias_minmax4_h30`
     was exact at strict c1 but only reached `50.638`, `51.869`, and
     `45.931`. Conclusion: fusing the residual through RCCL's AR-bias API
     removes an immediate consumer add but makes the collective path slower
     than stock Tree/LL. Keep the patch only as a negative source artifact;
     the next viable source work needs a different exact collective primitive,
     a graph-visible schedule change, or a broader legal boundary rewrite.
151. The final min/max=4 channel envelope does not materially change the
     standalone RowParallel lower bound. Serving-equivalent LLMM1 plus PyNccl
     allreduce graph replay on `.20`
     `llmm1_pipelined_rowparallel_host20_20260609_llmm1_graph_minmax4_h20`
     measured `attn_5120x768` at `0.055389 ms/call` and
     `mlp_down_5120x2176` at `0.071916 ms/call`; `.30`
     `llmm1_pipelined_rowparallel_host30_20260609_llmm1_graph_minmax4_h30`
     measured `0.055645` and `0.072491`. In-place allreduce variants were
     essentially flat or slightly slower. This reinforces that min/max=4 is a
     serving-envelope promotion, not a raw-boundary breakthrough. The dense
     `65 TPS` gap is still in full-graph sequencing, collective scheduling,
     launch/capture behavior, or a larger legal boundary rewrite rather than
     the isolated LLMM1 plus allreduce primitive alone.
152. Omitting the `-O` flag is rejected again under the exact final ROCm 7.2
     mutable/interleaved dense winner envelope. The `.20` run
     `qwen36_27b_decode_tiers_rocm72_mutable_interleaved_no_o_final_h20_20260609_host20_20260609_1612_no_o_final_h20`
     used the current global stack except `OMIT_OPT_FLAG=1`: TP8 fp16,
     async, INFO logging, max model len `16384`, max seqs `4`,
     max batched tokens `2048`, prefix cache off, Tree/LL, P2P off,
     `NCCL_MIN_NCHANNELS=4`, `NCCL_MAX_NCHANNELS=4`, `NCCL_NTHREADS=128`,
     mutable RowParallel, and native-runtime interleaved SwiGLU. It passed
     strict c1 with `52.007` backend TPS, then scored `c1_2000 53.276` and
     `c1_10000 46.994`, all with `metrics_settled=true`. This is below the
     current `.20` winner `56.885/57.554/54.599`, so keep explicit `-O=3`.
     The run also validates the patience rule: live `/metrics` stayed at zero
     during cold startup and early request admission while logs showed compile,
     graph capture, and later successful completion; score only settled
     summaries.
153. The final-envelope reduce-scatter/RMSNorm boundary recheck closes
     hidden-sharded collectives as a c1 decode route, but promotes them as a
     real prefill/concurrency source milestone. Under the ROCm 7.2 dense
     winner envelope with Tree/LL, P2P disabled, `NCCL_MIN_NCHANNELS=4`,
     `NCCL_MAX_NCHANNELS=4`, and `NCCL_NTHREADS=128`, `.20` measured
     `1x5120` graph baseline `0.068277 ms` versus `rs_ag 0.139445 ms` and
     `sharded_rms 0.189293 ms`; `.30` measured `0.068442 ms` versus
     `0.139620 ms` and `0.189472 ms`. Rows `1-16` stayed slower than stock
     allreduce on both hosts, so reject this for the single-request
     `65 TPS` decode gate. At prefill-sized rows the result flips:
     `.20 64x5120` baseline `0.533577 ms` versus `rs_ag 0.412037 ms`,
     `sharded_rms 0.407010 ms`, and `token_rs_rms 0.354226 ms`; `.30`
     baseline `0.540981 ms` versus `0.409328 ms`, `0.405966 ms`, and
     `0.353377 ms`. Keep reduce-scatter/RMSNorm for prompt-processing and
     concurrency work, not sustained c1 decode.
154. The direct RCCL coalescing lower bound remains positive under the final
     dense min/max=4 channel envelope. After patching
     `run_rccl_direct_allreduce_graph_bench_20260608.sh` to expose
     `NCCL_MIN_NCHANNELS`, `.20` run
     `rccl_direct_allreduce_graph_bench_host20_20260609_direct_ar_coalesce_minmax4_h20`
     measured graph out `0.046187/0.052658/0.064633/0.091558/0.146178/
     0.259412/0.486475/0.939650 ms` for `1/2/4/8/16/32/64/128 x 5120`;
     `.30` measured `0.046396/0.052690/0.064731/0.091482/0.145920/
     0.258300/0.483646/0.934067 ms`. All rows were exact. Compared with the
     prior one-channel `.20` rows, min/max=4 improved graph out by about
     `4.5%` at `1x5120`, `5.8%` at `2x5120`, `12.5%` at `4x5120`, and
     `15.2%` at `8x5120`. This is a source milestone, not a serving
     promotion: one coalesced `2x5120` allreduce is far cheaper than two
     separate `1x5120` calls, but the current Qwen3.6 dense graph does not
     legally expose adjacent independent same-shape RowParallel reductions.
     The implementation path must therefore change schedule/boundary legality
     or build a fused collective/consumer primitive, not just flip another
     RCCL env knob.
155. RCCL topology and rank-order tuning is closed as a near-term dense decode
     gate path. The `.20` debug run
     `rccl_direct_allreduce_graph_bench_host20_20260609_rccl_topology_debug_minmax4_h20`
     confirmed that RCCL sees the host as four two-GPU PCIe islands
     (`0/1`, `2/3`, `4/5`, `6/7`) and, under the current Tree/LL
     min=max=4 envelope, uses four collective channels all ordered
     `0 1 2 3 4 5 6 7`. Direct graph-replay rank-order sweeps then tested
     current, reverse, pair-reversed, even/odd-striped, root-zigzag, and
     cross-pair visible orders. The best `1x5120` graph-out change was only
     about `0.4%`, and the same stripe variant regressed `8x5120` from about
     `0.0917 ms` to `0.0950 ms`. Other orders were flat or slightly worse.
     Do not spend serving lanes on `HIP_VISIBLE_DEVICES` permutations unless a
     future direct primitive sweep first shows a material shift; topology
     ordering is not the missing `65 TPS` step.
156. MLP residual pre-folding is semantically viable but not a dense decode
     promotion by itself. The source rewrite folds the MLP residual into the
     local RowParallel partial as `allreduce(partial + residual / TP)`, which
     is algebraically equivalent to `allreduce(partial) + residual`. The first
     Qwen3.5 smoke found and fixed a real final-norm edge case:
     `GemmaRMSNorm(hidden_states, None)` returns a tensor, so a successful
     last-layer pre-fold cannot reuse the inherited unconditional tuple
     unpack. The clean `.20` run
     `qwen36_27b_decode_tiers_rocm72_qwen35_mlp_residual_prefold_fulltiers_h20_fix1_20260609_host20_20260609_1819_q35_prefold_fulltiers_h20_fix1`
     passed strict uncapped c1 with `56.429` backend TPS and then measured
     `c1_2000 57.502` and `c1_10000 54.561`, versus the current winner
     `56.885/57.554/54.599`. Keep this as a valid source component for a
     future schedule/consumer-fusion rewrite, but do not promote it as the
     global winner because it does not reduce collective launch count and does
     not clear the `65 TPS` gate.
157. Attention residual pre-folding is also legal but not a standalone dense
     decode promotion. The repaired Qwen3.5 overlay can fold the attention
     residual into the local attention RowParallel partial and preserve the
     folded post-attention residual state before the one-argument
     `post_attention_layernorm` call. The corrected `.20` attention-only
     smoke
     `qwen36_27b_decode_tiers_rocm72_attn_residual_prefold_smoke_h20_fix2_20260609_host20_20260609_1842_attn_prefold_smoke_h20_fix2`
     passed the strict uncapped thinking gate with `56.393` backend decode
     TPS. The `.30` attention+MLP stack
     `qwen36_27b_decode_tiers_rocm72_attn_mlp_residual_prefold_smoke_h30_fix2_20260609_host30_20260609_1842_attn_mlp_prefold_smoke_h30_fix2`
     was also valid but slower at `54.192` backend decode TPS. This closes
     residual pre-folding as a standalone gate path: the transformation is
     exact and useful as source evidence, but it does not reduce collective
     launch count or move the current winner toward `65 TPS`.
158. Final-envelope RCCL runtime knob tuning is closed for the dense c1 decode
     gate. Under the current Tree/LL, P2P-off, `NCCL_MIN_NCHANNELS=4`,
     `NCCL_MAX_NCHANNELS=4`, `NCCL_NTHREADS=128` winner, direct graph replay
     rejected channel counts 3/5/6/7, LL buffer sizes 131K/524K/1M,
     `NCCL_GRAPH_MIXING_SUPPORT=1`, and `NCCL_LOCAL_REGISTER=1`. The only
     borderline item, `NCCL_GRAPH_REGISTER=0`, looked slightly better on
     `.30` but did not reproduce on `.20`: the repeat measured graph-out
     `1/2/4/8/16 x 5120` at
     `0.046332/0.052756/0.064857/0.091491/0.146392 ms`, versus the `.20`
     min/max=4 baseline
     `0.046187/0.052658/0.064633/0.091558/0.146178 ms`. Do not spend more
     serving lanes on RCCL env sweeps unless a new direct primitive first
     shows a material decode-row improvement. The next serious path remains
     source work that reduces, fuses, or legally coalesces RowParallel
     collective boundaries.
159. The down-projection `LLMM1 + residual` fused local source path is a
     microbench win but not a serving promotion. The new
     `gfx906_rowpar::llmm1_residual` extension became bit-exact after fixing
     row-pair stores (`qthreadid == 0`, not `lane == 0`) and matching fp16
     epilogue rounding. On `.20` it improved the local decode microbench from
     `37.120 us` median for separate `LLMM1 + residual add` to `33.280 us`
     median for the fused kernel. Under the current dense winner envelope,
     however, the clean full-tier run
     `qwen36_27b_decode_tiers_rocm72_mutable_interleaved_minmax4_downllmm1resid_full_h20_20260609_host20_20260609_201938`
     measured strict c1 `56.521`, c1_2000 `57.322`, and c1_10000 `54.366`,
     below the incumbent `.20` `56.885/57.554/54.599`. This closes local
     residual/add saving as a standalone gate path. A source path now has to
     change the RowParallel collective boundary itself, not only make the work
     around the collective cheaper.
160. The upstream AsyncTP path can be made to instantiate on ROCm, but forced
     SP/AsyncTP is a hard reject for single-request dense decode. The installed
     ROCm 7.2 image contains `AsyncTPPass`, yet `pass_manager.py` imports it
     only under `current_platform.is_cuda()`. The
     `pass_manager_rocm_collective_import_20260609` overlay moves the import
     into the CUDA-alike branch, and the `.20` smoke
     `qwen36_27b_decode_tiers_rocm72_mutable_interleaved_minmax4_asynctp_sp1_smoke_h20_20260609_host20_20260609_203859`
     confirmed `Enabled custom fusions: gemm_comms`, `enable_sp=True`, and
     `fuse_gemm_comms=True` without the old `AsyncTPPass` NameError. The
     strict c1 output was valid, but backend decode was only `10.562 TPS`.
     The log warning is decisive for c1: sequence parallelism removes batch
     sizes `[1, 2, 4]` because they are not multiples of TP size `8`, so the
     single-request lane loses the incumbent graph shape. Keep the import fix
     for future prefill/concurrency work, but do not pursue forced SP/AsyncTP
     for the `65 TPS` c1 decode gate unless a later source change preserves
     batch-size-1 decode on the fast path.
161. The upstream `fuse_allreduce_rms` pass is not an immediate gfx906 gate
     path in the ROCm 7.2 image. The same pass-manager import overlay lets vLLM
     accept `{"pass_config":{"fuse_allreduce_rms":true}}`, and the `.20` smoke
     `qwen36_27b_decode_tiers_rocm72_mutable_interleaved_minmax4_allreduce_rms_smoke_h20_20260609_host20_20260609_205731`
     passed strict c1 with `57.068` backend TPS. However, every worker logged
     `Flashinfer is not installed or comm module not found, skipping allreduce
     fusion pass` and then `AllReduce fusion pass is disabled.` Treat the smoke
     as no-op pass plumbing, not a promotion. The upstream pass remains a useful
     semantic template for `AllReduce -> RMSNorm` fusion, but the current image
     needs a native ROCm/gfx906 fused collective/consumer backend or a legal
     reduction-count change before this path can affect dense decode TPS.
162. The model generation-config warning is not a dense decode lever under the
     current benchmark. A `.20` current-envelope smoke with
     `--generation-config vllm`
     (`qwen36_27b_decode_tiers_rocm72_mutable_interleaved_minmax4_generation_config_vllm_smoke_h20_20260609_host20_20260609_210629`)
     passed strict c1 but measured only `56.235` backend TPS, below the current
     `.20` winner `56.885`. The harness already sends `temperature: 0`; do not
     spend more gate lanes on sampler/default-generation cleanup unless the
     request payload itself changes for a separate benchmark objective.
163. Hidden-sharding the dense MLP boundary by repartitioning the next gate/up
     weights is not a viable `65 TPS` single-request decode path. The MEP
     `mlp_hidden_shard_gateup_boundary_bench_20260610.py` tested the exact
     structural idea on `.20` and `.30`: baseline
     `allreduce([1,5120]) -> RMSNorm -> LLMM1([4352,5120])` versus
     `reduce_scatter([5120,1]) -> scalar allreduce -> LLMM1([34816,640]) -> reduce_scatter([34816,1])`.
     Graph replay regressed from `0.164287 ms` to `0.271864 ms` on `.20`
     and from `0.170345 ms` to `0.273169 ms` on `.30`. The reason is
     structural, not tuning noise: the alternative replaces one small hidden
     allreduce with two collectives, one of which moves the much larger gate/up
     output vector. Continue source work on the narrower MLP/down collective
     boundary rather than a full hidden-sharded gate/up rewrite.
164. A custom peer-memory allreduce is blocked for the TP8 dense lane by HIP
     peer-access reality on `.20`. The new
     `gfx906_ipc_flag_allreduce_bench_20260610.cpp` adds a ready/done
     device-flag protocol around peer reads, but it cannot time the full TP8
     collective because `hipIpcOpenMemHandle` fails for devices `0-5`.
     The follow-up `hip_peer_access_matrix_20260610.cpp` proves this is not a
     visibility-mask artifact: inside the ROCm 7.2 image, all eight devices are
     `gfx906:sramecc+:xnack-` with `34342961152` bytes of VRAM, but
     `hipDeviceCanAccessPeer` reports no direct peer access among `0-5`; only
     `6/7` are mutually peer-accessible. Repeating the matrix with
     `HSA_FORCE_FINE_GRAIN_PCIE=1` did not change it. This explains why P2P-on
     lanes have not promoted and why the current winner keeps
     `NCCL_P2P_DISABLE=1`. Do not spend TP8 gate time on peer-load collectives
     unless the driver or hardware peer-access matrix changes.
165. Host-staged small-message allreduce is exact but far too slow for the
     dense `65 TPS` gate. The new
     `gfx906_host_mapped_allreduce_bench_20260610.cpp` avoids HIP IPC peer
     imports by writing partials into mapped host memory and synchronizing with
     device-side ready/done flags. On ROCm 7.2 it completed correctly with no
     timeouts, but `1x5120` measured `0.277351 ms` on `.20` and `0.279685 ms`
     on `.30` with `HSA_FORCE_FINE_GRAIN_PCIE=1`, versus the repeated RCCL
     Tree/LL min=max=4 floor near `0.0463 ms`. `HIP_FORCE_DEV_KERNARG=1` was
     also checked on the hot rows and was slightly slower than baseline on
     both hosts. Continue toward native RCCL/RowParallel boundary work; do not
     revisit host-mapped or kernarg env paths for c1 decode unless the ROCm
     memory model changes substantially.
166. `.50` does not reopen peer-memory collectives despite the VRAM upgrade.
     Hardware inventory shows one 4 GiB display GPU plus eight gfx906 devices
     with `34342961152` bytes of VRAM each; the product-name strings still
     mislabel two of them as 16 GB, so use VRAM/gfx/DID checks instead of
     product strings. The available ROCm 6.3 diagnostic image, with GPU0
     skipped, exposed seven 32 GiB gfx906 devices and still showed no full
     mutual peer-access matrix: most devices could only access the last visible
     device one-way. The all-visible run crashed, consistent with the display
     GPU contaminating HIP enumeration. Before any `.50` lane, skip GPU0 and
     re-run peer/runtime checks under the exact image; do not assume `.50`
     solves the TP8 peer-memory substrate problem.
167. Bundled MSCCL allpairs XML is rejected even with direct C++ base pointers.
     The `.20` run `rccl_direct_ar_msccl_xml32_baseptr_host20_20260609_220342`
     staged only `allreduce-allpairs-8n-ll-32tb.xml`, cleared
     `NCCL_ALGO/PROTO`, enabled `RCCL_MSCCL_FORCE_ENABLE=1`, and used the
     direct `hipMalloc` RCCL graph harness for `1x5120,2x5120`. No timing rows
     were emitted: six ranks segfaulted before the first row and the remaining
     two ranks spun on GPUs 6/7 until cleanup. This closes the allocator-offset
     theory for the earlier PyNccl/vLLM XML crash. A future MSCCL branch must
     generate/debug gfx906 XML or patch RCCL/MSCCL itself; do not route serving
     through bundled XML until the direct base-pointer harness is correct and
     faster than Tree/LL.
168. Ring/LL remains rejected under the final dense winner envelope. The final
     recheck with Tree/LL's promoted channel policy changed only
     `NCCL_ALGO=Ring`, keeping `LL`, P2P off, `NCCL_MIN_NCHANNELS=4`,
     `NCCL_MAX_NCHANNELS=4`, and `NCCL_NTHREADS=128`. Direct C++ graph replay
     measured `1x5120` at `.20 0.048686 ms` and `.30 0.048792 ms`, slower than
     the repeated Tree/LL min=max=4 floor near `0.0463 ms`. Do not spend another
     serving lane on Ring/LL unless a source patch changes its primitive
     behavior, not just the runtime env.
169. A direct fused residual/RMSNorm consumer is a valid component milestone,
     not the main `65 TPS` path. New MEP
     `rccl_ar_fused_rms_boundary_bench_20260610.cpp` captured graph replay of
     `RCCL allreduce -> fused residual/RMSNorm` under the final Tree/LL
     min=max=4 winner envelope. It was correct on `.20` and `.30`; `1x5120`
     allreduce-only stayed at about `0.0462 ms`, and the fused consumer raised
     the boundary to about `0.06336 ms`, a `16.5-17.1 us` per-boundary add
     across `1/2/4/8 x 5120`. This proves a compact ROCm consumer kernel is
     feasible, but it cannot bridge the sustained gap alone. Stack consumer
     fusion only after a collective improvement or use it as an integration
     milestone; the primary dense decode path remains reducing or materially
     accelerating RowParallel collectives.
170. Tree/SIMPLE is rejected under the final dense winner envelope. Direct C++
     graph replay on `.20` and `.30` used `NCCL_ALGO=Tree`,
     `NCCL_PROTO=Simple`, P2P off, min=max channels 4, and
     `NCCL_NTHREADS=128`. The `1x5120` graph-out row was `.20 0.095698 ms`
     and `.30 0.095836 ms`, roughly twice the Tree/LL floor near
     `0.0463 ms`; `2/4/8 x 5120` rows were also slower. Because the inspected
     RCCL source maps pipelined LL/LL128 generated kernels back to the
     non-pipelined primary, and SIMPLE is directly too slow, do not pursue
     RCCL pipelining/SIMPLE as a dense gate-clear path unless source work
     changes the generated primitive itself.
171. Explicit subgroup hierarchy is exact but rejected for dense TP8 c1
     decode. The new `pynccl_hierarchical_allreduce_bench_20260610.py` MEP
     used pair allreduces, leader allreduce, then pair fanout under the current
     Tree/LL, P2P-off, min=max=4, `NCCL_NTHREADS=128` envelope. It returned
     max error `0.0`, but graph replay on `.20` measured `1x5120` world
     allreduce `0.047094 ms` versus hierarchy `0.089798 ms`; `.30` measured
     `0.047204 ms` versus `0.087101 ms`. Larger rows were also about
     `1.77-1.89x` slower. Do not integrate this schedule into vLLM; the
     topology-aware launch count increase outweighs any smaller-subgroup
     benefit at the decode row.
172. The upstream `fuse_allreduce_rms` activation path is now proven on ROCm,
     but the current upstream matcher does not replace the dense winner graph.
     A forced `fi_allreduce_fusion_max_size_mb=1.0` created a false-negative
     graph shape by splitting the dense TP8 compile range into `(1,102)` and
     `(103,2048)`, which regressed even when FlashInfer was absent. Raising the
     threshold to `20.0` MiB preserves the winner's single `(1,2048)` range for
     Qwen3.6-27B fp16 (`2048 * 5120 * 2` bytes). Valid threshold-20 smokes
     were therefore compile-shape/activation checks, not fusion evidence: `.20`
     shim strict `c1_128` backend decode was `51.352 TPS`, while the matched
     `.30` no-shim control produced `48.862 TPS`. A fresh DEBUG compile with
     `20.5` MiB initialized FlashInfer workspaces at `max_token_num=2048`, but
     logged `AllReduceFusionPass completed` with `Replaced 0 patterns`.
     Inspection of rank-0 `computation_graph.py` found `388`
     `torch.ops.vllm.all_reduce` sites, `0` `fused_add_rms` sites, and `0`
     `rms_norm` sites; RMSNorm/residual is static-expanded into float/add,
     pow/mean/rsqrt/mul/to operations. Decision: keep the threshold and
     activation lessons, reject the Python shim as a no-op for promotion, and
     move source work to matching the actual ROCm graph shape before writing a
     native ROCm fused op.
173. `use_inductor_graph_partition=true` is the first confirmed way to expose
     the dense TP8 out-of-place allreduce/RMSNorm boundary to the vLLM fusion
     pass, but registration-based pattern replacement still misses. This lane
     did not mount the current mutable RowParallel overlay, so treat it as
     upstream pass evidence rather than a new measurement of the global winner
     graph. The default
     piecewise path runs `AllReduceFusionPass` on small subgraphs that contain
     only `1-2` allreduces and no downstream RMSNorm window, so the upstream
     pass cannot fuse this out-of-place graph there. With graph partition
     enabled, the pass sees the full lowered ROCm graph: each rank logs
     `129` `torch.ops.vllm.all_reduce` sites, `129` direct float users, and
     `129` static-expanded residual RMS windows. Valid smokes stayed correct,
     but upstream replacement and a gated-lowered MEP replacement both logged
     `Replaced 0 patterns`; the latter produced only `49.467` backend TPS.
     This promotes graph partition as a source-enabling milestone only. The
     next real source step is direct FX graph surgery or a pre-Inductor
     model-level boundary rewrite, not more blind `register_replacement`
     variants. Also note the Qwen3.6/Gemma RMSNorm semantic: the weight path is
     `(1.0 + weight)`, so any native fused backend must support that gated RMS
     variant rather than assuming plain `normalized * weight`.
174. The graph-partition visibility milestone also holds for the current dense
     global-winner mutable RowParallel stack. The `.30` compile-only census
     `qwen36_27b_decode_tiers_rocm72_mutable_postir_graph_partition_census_h30_host30_20260610_0018_mutable_graph_partition_census_h30`
     mounted the mutable RowParallel overlays plus native-runtime interleaved
     SwiGLU under the final winner envelope: TP8 fp16, explicit `-O=3`, async
     scheduling, `max_model_len=16384`, `max_num_seqs=4`,
     `max_num_batched_tokens=2048`, Tree/LL, P2P disabled, min=max channels
     `4`, and `NCCL_NTHREADS=128`. With
     `use_inductor_graph_partition=true` and `SKIP_REQUEST=1`, the backend
     reached readiness after a patient compile/warmup and exited cleanly. Each
     rank reported `5591` graph nodes, `5070` call_function nodes, `129`
     collective value nodes, mutable RowParallel collectives represented as
     `auto_functionalized(vllm.all_reduce_inplace.default)`, and the post-IR
     pass selected `128/128` direct allreduce/RMS boundaries. This is not a
     throughput promotion because `replace=none` and no request was sent. It is
     the strongest current source evidence that the next patch should target
     the mutable RowParallel `all_reduce_inplace -> residual/RMSNorm` boundary
     directly with FX graph surgery or a model-level native ROCm fused call
     site.
175. Active mutable post-IR replacement is now proven but rejected as a
     consumer-only gate path. The `.30` one-boundary smoke
     `qwen36_27b_decode_tiers_rocm72_mutable_postir_fused_one_boundary_h30_host30_20260610_0032_mutable_postir_fused_one_h30`
     selected `1/128` direct mutable boundaries, erased `10` local pure nodes
     per rank, compiled, served, and passed the strict thinking gate at
     `54.129` backend TPS. The `.30` all-boundary smoke
     `qwen36_27b_decode_tiers_rocm72_mutable_postir_fused_all_boundaries_h30_host30_20260610_0040_mutable_postir_fused_all_h30`
     selected `128/128` direct boundaries, erased `1280` local pure nodes per
     rank, compiled, served, and passed the strict gate at only `53.018`
     backend TPS. This proves the mutable FX replacement infrastructure is
     real, but replacing residual/RMSNorm consumers without improving the
     collective regresses relative to the current dense winner. Continue source
     work on allreduce primitive cost, overlap, or a legal collective-count
     change rather than more consumer-only fusion variants.
176. Dense TP4 is rejected under the current ROCm 7.2 global-winner stack. Two
     full-tier runs used the promoted dense settings with only TP/world size
     changed to `TENSOR_PARALLEL_SIZE=4` on GPUs `0,1,2,3`: explicit `-O=3`,
     async scheduling, `max_model_len=16384`, `max_num_seqs=4`,
     `max_num_batched_tokens=2048`, prefix cache off, Tree/LL, P2P disabled,
     min=max channels `4`, `NCCL_NTHREADS=128`, mutable RowParallel overlays,
     and native-runtime interleaved SwiGLU. `.20`
     `qwen36_27b_decode_tiers_rocm72_dense_tp4_mutable_interleaved_minmax4_h20_host20_20260610_005251`
     passed strict smoke at `40.165` backend TPS, then produced `40.298`
     c1_2000 and `38.904` c1_10000. `.30`
     `qwen36_27b_decode_tiers_rocm72_dense_tp4_mutable_interleaved_minmax4_h30_host30_20260610_005252`
     passed strict smoke at `40.156` backend TPS, then produced `40.270`
     c1_2000 and `38.875` c1_10000. The rejection is clean: correctness passed
     on both hosts, but the heavier per-rank dense compute dominates the
     reduced collective world size. Keep TP8 as the dense base and focus source
     effort on lowering or overlapping RowParallel collective cost.
177. RCCL Tree/LL split-warp source variants are closed under the final dense
     communication envelope. The `.20` rerun
     `rccl_direct_allreduce_graph_bench_host20_20260610_split3r1b_minmax4_h20`
     used the previously built `split3r1b` RCCL overlay with TP8 fp16,
     Tree/LL, P2P disabled, min=max channels `4`, and `NCCL_NTHREADS=128`.
     It was exact, but graph-out rows regressed versus the final `.20`
     min=max4 baseline: `1x5120 0.049289 ms` versus `0.046187`, `8x5120
     0.108940` versus `0.091558`, `64x5120 0.631419` versus `0.486475`, and
     `128x5120 1.226013` versus `0.939650`. Do not run serving tiers for
     split3r1b, and do not spend a new min=max4 lane on split1r3b because it
     was already worse in one-channel evidence while split3r1b is dominated by
     the final baseline.
178. The current dense global winner is a usable long-prefill baseline, but
     not a prefill-specialized config. On `.20`, the persistent ROCm 7.2 TP8
     winner backend with `max_num_seqs=4`, `max_num_batched_tokens=2048`,
     prefix cache off, Tree/LL, P2P off, min=max channels `4`,
     `NCCL_NTHREADS=128`, mutable RowParallel overlays, and native-runtime
     interleaved SwiGLU produced `8192`-token vLLM-summed prefill TPS of
     `603.627/533.751/504.927/491.352` at concurrency `1/2/4/8`. Client wall
     prompt throughput was stable near `602-603` TPS for those `8192` cases.
     Because the backend caps active sequences at `4`, concurrency `8` is
     partly queueing evidence rather than proof of eight-way simultaneous
     prefill. Use this as the current-winner prefill baseline before testing
     SP, larger batch-token envelopes, or ai-infos-style prefill knobs.
179. `use_inductor_graph_partition=true` is not runtime-neutral on the current
     dense winner. A full strict/2000/10000 dry-run on `.30` with graph
     partition plus post-IR no-rewrite instrumentation stayed correct but
     measured only `50.430` strict backend decode TPS, `51.447` on c1_2000,
     and `45.561` on c1_10000. The current winner's `.20` c1_10000 reference
     is `54.599`, so graph partition remains source-visibility evidence only.
     Do not build a deployable gate path that requires graph partition unless
     the source change pays back this regression.
180. Current-winner concurrent decode saturates at the configured active
     sequence limit. On `.20`, the warm ROCm 7.2 TP8 winner backend produced
     aggregate client wall TPS near `55.9` for `c1_2000` at concurrency `4`
     and `8`, while per-request TPOT-derived throughput fell near `14` TPS.
     The c8 case is two queued waves because `max_num_seqs=4`; it preserves
     aggregate throughput but doubles tail latency. Treat this as workload
     behavior and not as a single-request gate regression.
181. Current-winner `rocprofv3 --attach` is closed for this ROCm 7.2 image.
     Two `.30` attach lanes launched the correct dense TP8 winner, waited
     through cold compile, completed warmup and a `1297+768` decode request,
     and saw exact metrics deltas from `1297/1` to `2594/769` prompt/generation
     tokens. Kernel-only attach
     `qwen36_27b_currentwinner_attach_rocprofv3_h30_20260610_032305` served the
     profile request in `17.508s`; runtime-trace attach plus a `90s` output
     settle
     `qwen36_27b_currentwinner_attach_runtime_rocprofv3_h30_20260610_033633`
     served it in `17.529s`. In both runs all eight worker attach logs reported
     `librocprofv3-attach.so :: success`, but no CSV or raw trace output was
     materialized. Treat this as a profiler-output limitation, not a model
     failure. Continue dense gate work from graph/source evidence and
     exact-shape MLP/down RowParallel microbenchmarks instead of another
     attach-based profiling rerun.
182. MLP residual pre-fold plus mutable post-IR direct allreduce/RMS fusion is
     source-valid but rejected as a serving path. `.30`
     `qwen36_27b_decode_tiers_rocm72_prefold_postir_direct_smoke_h30_host30_20260610_0358_prefold_postir_smoke_h30`
     selected `64/64` direct MLP/down boundaries on every rank and passed the
     strict uncapped thinking gate, proving the rewrites compose structurally.
     Backend decode was only `49.863 TPS` on the strict smoke. This reinforces
     the graph-partition lesson: exposing the graph for post-IR replacement is
     still more expensive than the local residual/RMS cleanup. Future dense
     gate lanes should avoid graph-partition-dependent runtime wins unless the
     source patch also removes that overhead.
183. Mutable post-IR direct allreduce/RMS fusion without graph partition is
     also rejected as a serving path. `.30`
     `qwen36_27b_decode_tiers_rocm72_postir_direct_nograph_smoke_h30_host30_20260610_0416_postir_direct_nograph_smoke_h30`
     ran under the normal piecewise compile path and passed strict uncapped
     thinking output, but backend decode was only `49.142 TPS`. The pass saw
     partial per-fragment windows (`40` log lines of `selected 2/2`, `8` lines
     of `selected 0/0`) and compiled much faster than graph partition, but the
     active replacement still regressed. Close consumer-only post-IR fusion as
     the primary dense-gate path; the remaining gap requires collective
     primitive work or a legal collective-count/size change.
184. Final-envelope LL128 is rejected for dense c1 decode. Direct C++
     graph replay on `.20` and `.30` used Tree/LL128 and Ring/LL128, P2P disabled,
     `NCCL_MIN_NCHANNELS=4`, `NCCL_MAX_NCHANNELS=4`, and `NCCL_NTHREADS=128`.
     The hot `1x5120` row was exact but about `0.1085-0.1091 ms` for Tree and
     `0.1085-0.1086 ms` for Ring, versus the current Tree/LL min=max4 floor
     around `0.0462 ms`. Tree `2x5120`, `4x5120`, and `8x5120` were also
     slower. Tree `64x5120` was faster than Tree/LL, so LL128 remains a
     possible prefill-specific primitive, but it does not earn a dense
     single-request decode serving lane.
185. Final-envelope `HSA_ENABLE_SDMA=0` is closed for dense decode. Direct
     graph replay on `.20` and `.30` used the current Tree/LL, P2P-off,
     min=max channels `4`, `NCCL_NTHREADS=128` envelope plus SDMA disabled.
     The hot `1x5120` row was exact but tied with the current floor:
     `.20 0.046373 ms` versus `0.046187`, `.30 0.046379` versus `0.046396`.
     Other hot rows were tied or worse, and `.30 2x5120`/`64x5120` regressed.
     Do not spend serving lanes on SDMA-off under the ROCm 7.2 winner stack.
186. Final-envelope `NCCL_NTHREADS` unset is closed for dense decode. The
     direct harness now supports explicitly omitting the env var, and `.20`
     / `.30` graph replay used Tree/LL, P2P off, min=max channels `4`.
     `.20` was effectively tied (`1x5120 0.046301 ms` versus `0.046187`;
     `64x5120 0.485340` versus `0.486475`), while `.30` regressed and showed
     a bad `1x5120` graph-out anomaly at `0.096799 ms`. Keep
     `NCCL_NTHREADS=128` in the dense winner envelope.
187. The memory-bandwidth path should be split by workload. Local `.10`
     inventory confirms eight `gfx906:sramecc+:xnack-` devices with 60 CUs,
     32 GiB VRAM each, 256 max workgroup size, 64-lane preferred wave multiple,
     and PCIe `16.0GT/s x16` links. This favors contiguous/coalesced kernels
     and larger row collectives, but the c1 decode RowParallel payload is only
     `1x5120` fp16, about 10 KiB per rank per boundary. Dense single-request
     decode remains latency/count dominated until source work legally coalesces
     or fuses boundaries. Prefill/concurrency is where the card bandwidth can
     be exploited directly: LL128 flipped positive at `64x5120`, and prior
     SP/reduce-scatter lower bounds improved at prefill-sized rows.
188. The final-envelope LL128 crossover is between `8x5120` and `16x5120`.
     Tree/LL128 remained a decode rejection at `1x5120` and was slower at
     `8x5120`, but direct graph replay showed exact wins at `16x5120`
     (`.20 0.133938` vs Tree/LL `0.146178`, `.30 0.133909` vs `0.145920`),
     `32x5120` (`.20 0.193796` vs `0.259412`, `.30 0.196160` vs `0.258300`),
     and `64x5120` (`.20 0.299024` vs `0.486475`, `.30 0.302414` vs
     `0.483646`). This is a concrete bandwidth/source milestone: use LL128 or
     similar bandwidth-friendly collectives only for row counts `>=16`, and
     keep Tree/LL for single-request c1 decode unless source work legally
     coalesces enough rows.
189. `NCCL_PROTO` auto-selection is closed for the dense c1 decode gate. With
     Tree fixed, P2P disabled, min=max channels `4`, and `NCCL_NTHREADS=128`,
     omitting `NCCL_PROTO` moved the hot `1x5120` graph allreduce to
     `.20 0.095668 ms` and `.30 0.099944 ms`, versus the current forced
     Tree/LL floor around `0.0462 ms`. This reinforces the row-threshold
     conclusion: bandwidth-friendly protocol changes need source gating for
     larger contiguous rows, not a global protocol unset.
190. Wide-row direct replay proves the gfx906 bandwidth path is material when
     collectives are large enough. Tree/LL128 versus Tree/LL graph-out times
     were `0.133915` vs `0.146257 ms` at `16x5120`, `0.301159` vs
     `0.490513 ms` at `64x5120`, `0.479991` vs `0.941800 ms` at `128x5120`,
     and `1.751425` vs `3.660298 ms` at `512x5120`, all exact. This promotes
     row-thresholded LL128 as a prefill/concurrency/source milestone while
     preserving the c1 decode rule: forced Tree/LL remains required below the
     `16x5120` crossover.
191. Dynamic Tree/LL to Tree/LL128 switching must be admitted in RCCL's
     communicator protocol matrix before the per-call threshold mutates the
     collective. The first gfx906 threshold patch generated LL128 device code
     but segfaulted at the first thresholded `16x5120` call because the
     communicator had been built from an LL-only protocol matrix. The corrected
     patch also exposes gfx906 non-acc LL128 runtime support and enables
     AllReduce/Tree/LL128 in the matrix when
     `RCCL_GFX906_TREE_LL128_MIN_BYTES > 0`. This fixed the crash on `.20` and
     `.30`. The correct default threshold is not the raw crossover at
     `163840` bytes: under the safe matrix path, `16x5120` and `32x5120`
     regressed. Promote `RCCL_GFX906_TREE_LL128_MIN_BYTES=655360`
     (`64x5120` fp16) as the source milestone. It preserves Tree/LL at
     `1/8/16/32 x5120` and improves graph-out by about `7.3%`, `17.1%`, and
     `21.6%` at `64/128/256 x5120` respectively in direct replay.
192. The corrected RCCL threshold overlay is a real dense prefill/concurrency
     milestone, not a dense decode gate clear. With the same ROCm 7.2 dense
     TP8 winner surface and only
     `VLLM_NCCL_SO_PATH=/rccl-overlay/install/lib/librccl.so.1` plus
     `RCCL_GFX906_TREE_LL128_MIN_BYTES=655360` added, `.20` full decode tiers
     landed at valid strict `56.925`, `c1_2000 57.655`, and `c1_10000
     54.623`, essentially tied/slightly above the prior `.20` winner
     `56.885/57.554/54.599`. `.30` landed at `53.391/54.540/52.070`, slightly
     below its prior host-local `53.936/54.908/52.156`. The hot c1 decode row
     still stays Tree/LL and the `65 TPS` dense gate remains uncleared.
     Prefill is the meaningful promotion: `.20` 8192-token vLLM-summed prefill
     TPS improved to `702.992/620.882/586.435/570.701` at concurrency
     `1/2/4/8`, versus the prior `.20` current-winner baseline
     `603.627/533.751/504.927/491.352`, about `16%` better across the grid.
     `.30` reproduced the positive prefill direction at
     `659.983/578.457/544.232/525.326`. First `4096/c1` entries were cold
     compile artifacts around `19 TPS`; warmed repeats were `.20 729.681` and
     `.30 710.276` vLLM-summed TPS. Promote the overlay for bandwidth-sized
     prefill rows and future legal row-coalesced decode work, but keep
     collective-boundary source work as the primary decode-gate path.
193. Multi-request dense burst testing confirms that gfx906 bandwidth is
     exploitable when enough rows are live, but this is a separate milestone
     from the single-request c1 decode gate. Using a `1024` token prompt plus
     fixed `1000` token decode, `.30` with the threshold RCCL overlay,
     `max_num_seqs=64`, and `max_num_batched_tokens=8192` reached `239.113`
     then `240.700` client-wall generation TPS at c64, versus `.20` stock
     Tree/LL maxseq64 at `224.772`. The same `.30` lane improved c64 prefill
     TPS to `59.314`/`58.133` versus `.20` `50.551`. At c16, a `.10`
     threshold maxseq16 warm repeat reached `137.340` wall generation TPS and
     `102.717` prompt TPS after an initial cold prompt-compile artifact
     (`222s` mean TTFT) was excluded. A controlled `.20`/`.30` maxseq16 pair
     confirmed the config effect: warm c16 rose to `.20 136.872` and `.30
     139.360` wall generation TPS, compared with the maxseq64 c16 band around
     `122 TPS`. The c32 maxseq32 retest was valid but did not promote:
     `.30` threshold maxseq32 warm reached `171.392` wall generation TPS and
     `197.184` overlapped TPOT TPS, slightly below `.30` threshold maxseq64 c32
     at `172.893` wall and `198.539` overlapped. Raising
     `max_num_batched_tokens` from `8192` to `16384` at maxseq64/c64 also did
     not promote: `.30` threshold warm tied/slightly missed wall TPS
     (`240.275` versus prior `240.700`) while prompt throughput dropped from
     `58.133` to `31.871`, even though overlapped TPOT rose to `291.502`.
     Lowering `max_num_batched_tokens` to `4096` did not promote either:
     `.30` threshold warm improved prompt throughput to `87.346`, but wall TPS
     fell to `226.426` and overlapped TPOT to `274.658`. Lesson: tune
     `max_num_seqs` to the target concurrency, keep
     `max_num_batched_tokens=8192` for the current c64 burst milestone because
     it is the best measured wall-throughput balance, keep cold prompt-compile
     artifacts separate from steady-state evidence, and continue treating c1
     decode as a small-row collective-boundary source problem.
194. INFO logging is not a dense c64 burst promotion, even though INFO remains
     useful for single-request decode candidate surfaces. A controlled c64
     retest on the current maxseq64/mnbt8192 burst surface showed `.30`
     threshold INFO warm at `239.806` wall generation TPS and `285.720`
     overlapped TPOT TPS, slightly below the DEBUG/winner repeat at `240.700`
     and `285.890`. `.20` Tree/LL INFO warm was essentially tied with its
     stock result (`224.937` versus `224.772` wall TPS), not a transferable
     improvement. Keep the measured burst milestone as maxseq64/mnbt8192 with
     the logging level used by that run, and reserve INFO for c1 decode lanes
     where earlier evidence showed less logging overhead.
195. Static-SP remains a controlled side lane, not a dense single-request
     decode promotion. The threshold-gated runtime assertion patch let the
     SP-threshold configs get past the original batch-divisibility failure, but
     the first maxseq4 runs were forced into a PIECEWISE-only decode path and
     strict c1 decode regressed to about `12.6 TPS` on `.20/.30`. Maxseq16 with
     `FULL_AND_PIECEWISE` restored FULL captures but padded c1 to an 8-token
     graph and stayed around `12.5 TPS`. A follow-up source hotfix preserved
     sub-threshold decode capture sizes while compiling static `4096/8192`
     SP rows, but the `.10` startup failed before readiness with an Inductor
     stride assertion on the 8192 path:
     `expected size 3==3, stride 8193==8192 at dim=0`. This is useful source
     evidence for the static-SP/meta-kernel lane, but it does not change the
     gate status: the dense 27B single-request `65 TPS` decode gate remains
     open, with the best valid standard tiers still in the mid/high 50 TPS
     band.
196. Selective MLP/down-proj out-place RowParallel routing is not a dense
     decode promotion. On top of the current ROCm 7.2 threshold winner,
     `VLLM_GFX906_ROWPAR_MLP_OUTPLACE=1` routed only `*.mlp.down_proj` through
     exact out-of-place allreduce while leaving other RowParallel boundaries on
     mutable in-place. `.20` regressed from the promoted threshold
     `56.925/57.655/54.623` standard tiers to `56.351/57.348/54.351`.
     `.30` was mixed at `53.322/54.871/52.244`, but still below the gate and
     below the `.20` global band. The serving result closes the simple
     out-place-vs-in-place MLP branch; remaining dense gate work must reduce
     the original collective cost, legally coalesce/reduce boundaries, or
     introduce a fused collective/consumer backend with real measured gain.
197. `.10` does not hide a better dense 27B single-request decode result for
     the final threshold surface. A full standard ladder on `.10` with the
     current ROCm 7.2 TP8 fp16 threshold winner, mutable RowParallel,
     native-runtime interleaved SwiGLU, Tree/LL, P2P disabled,
     `NCCL_MIN_NCHANNELS=4`, `NCCL_MAX_NCHANNELS=4`, `NCCL_NTHREADS=128`, and
     `RCCL_GFX906_TREE_LL128_MIN_BYTES=655360` produced valid strict c1
     `49.942`, `c1_2000 50.840`, and sustained `c1_10000 44.883` backend
     decode TPS. This is materially below the `.20` promoted reference
     `56.925/57.655/54.623`, so keep `.20` as the global standard-ladder
     reference for this surface. A Tree/LL `2 reduce / 2 broadcast` RCCL split
     should not be built as a separate source branch: with gfx906's 64-lane
     wavefront, RCCL's existing LL split rounds the normal 256-thread launch to
     `128/128`, so the apparent 2/2 variant is already the default. The
     measured 1/3 and 3/1 split patches bracketed that default and regressed.
198. Grouped small-allreduce is a real lower-bound milestone under the final
     dense comm envelope, but the current dense decode graph has no legal
     scheduler-only grouping point. The final-envelope MEP on `.20` and `.30`
     used the ROCm 7.2 threshold winner surface with Tree/LL, P2P disabled,
     `NCCL_MIN_NCHANNELS=4`, `NCCL_MAX_NCHANNELS=4`, and the threshold RCCL
     overlay. For TP8 `1x5120`, `.20` graph replay improved from `0.050883`
     ungrouped to `0.020220 ms/call` at group size 4, `0.048613` to
     `0.017844` at group size 8, and `0.046815` to `0.015689` at group size
     32. `.30` reproduced the same direction: group size 4 improved from
     `0.051634` to `0.020767`, group size 8 from `0.050166` to `0.018016`,
     and group size 32 from `0.047182` to `0.015742`. However, the post-IR
     cluster census on the current dense graph reported `collective_ops=129`,
     `groupable_pairs=0`, `blocked_by_dependency=128`,
     `blocked_by_intervening_use=128`, `max_cluster_size=1`,
     `clusters_ge_2=0`, and `clusters_ge_4=0`, while still selecting the same
     `128/128` RowParallel boundaries for direct analysis. Decision: promote
     grouped allreduce as a source lower bound and reject naive
     `ncclGroupStart/ncclGroupEnd` around existing RowParallel calls. Clearing
     the dense `65 TPS` c1 gate still requires changing graph semantics,
     reducing collective count, or implementing a lower-overhead exact
     collective/fused collective primitive.
199. RCCL launch-control/env tuning is closed under the final dense envelope.
     Direct C++/HIP RCCL graph replay on `.20` and `.30` used the promoted
     TP8 fp16 communication surface: Tree/LL, P2P disabled,
     `NCCL_MIN_NCHANNELS=4`, `NCCL_MAX_NCHANNELS=4`, `NCCL_NTHREADS=128`,
     the threshold RCCL overlay, and
     `RCCL_GFX906_TREE_LL128_MIN_BYTES=655360`. The baseline `1x5120`
     graph allreduce floor was `0.046195 ms/call` on `.20` and
     `0.046119 ms/call` on `.30`. `NCCL_CTA_POLICY=0/1`,
     fixed `NCCL_MIN_CTAS=NCCL_MAX_CTAS` from 1 through 8,
     `RCCL_GFX9_CHEAP_FENCE_OFF=1`, `NCCL_GRAPH_HELPER_DISABLE=1`,
     `RCCL_PIPELINE_ALL_DATA_TYPES=1`, and
     `RCCL_DISABLE_REDUCE_COPY_PIPELINING=1` did not produce a
     reproducible floor reduction across both hosts.
     `NCCL_LAUNCH_ORDER_IMPLICIT=1` segfaulted ranks before timing rows on
     both hosts, and fixed CTA count 8 worsened `8x5120`. Decision: stop
     spending serving lanes on RCCL launch-policy knobs for this gate. The next
     dense gate branch must be a real generated-kernel/algorithm source change
     inside RCCL, a legal graph rewrite that changes collective count, or a
     primitive that first proves a sub-`0.044 ms` TP8 `1x5120` floor on both
     `.20` and `.30`.
200. Cached user-buffer loads inside RCCL LL are correctness-clean but not a
     dense decode primitive win. Patch
     `rccl_721_gfx906_tree_ll128_threshold_ll_user_cached_load_20260610.patch`
     replaced gfx906 LL user-buffer `__builtin_nontemporal_load` reads with
     ordinary global loads while leaving LL FIFO receive synchronization
     untouched. The ROCm 7.2 overlays built on `.20` and `.30` and direct TP8
     graph replay was exact. Results stayed in the existing floor band:
     `.20 1x5120` was `0.046152 ms/call` and `.30 1x5120` was
     `0.046139 ms/call`; paired baseline was about `0.046195` and `0.046119`.
     Larger rows were also only noise-level mixed. Do not promote this to
     serving; local user-buffer cache hinting is not the missing `65 TPS` gate
     lever.
201. Cached/volatile LL FIFO receive loads are a hard dense gate rejection.
     Patch `rccl_721_gfx906_ll_fifo_volatile_cached_load_20260610.patch`
     changed the LL FIFO receive path from non-temporal loads to volatile
     ordinary global loads, while preserving exact Tree/LL semantics. It built
     on `.20` for `AllReduce TREE LL Sum f16` and passed a bounded TP8
     `1x5120` correctness replay with `max_error=0.0`, but graph out-of-place
     latency regressed to `0.077101 ms/call` versus the current `~0.046 ms`
     floor. Do not expand this branch to `.30` or serving. Together with the
     neutral user-buffer load probe, this closes the simple LL cache-hint axis;
     the next source work must change the generated algorithm, collective
     count, or graph boundary rather than just load cache policy.
202. Gfx906 LL block-scope barrier fencing is not a dense gate primitive win.
     Patch `rccl_721_gfx906_ll_barrier_block_fence_20260610.patch` changed the
     LL/LL128 barrier path to use `__threadfence_block()` on gfx906 instead of
     the generic device-wide fence. The `.20` RCCL overlay built cleanly and
     exact direct TP8 graph replay produced `1x5120 0.046337 ms/call`,
     `2x5120 0.052114`, `4x5120 0.063874`, `8x5120 0.090052`, and
     `16x5120 0.143529`. This stays in the same `0.046 ms` floor band and is
     slightly slower than the `.20` final-envelope baseline. Do not expand this
     branch to `.30` or serving; cheap LL load/fence tweaks are now closed as
     primary paths. The remaining dense `65 TPS` work needs generated
     collective algorithm changes, legal collective-count reduction, or graph
     boundary changes.
203. Gfx906 Tree/LL 192-thread opt-in is correctness-clean but slower. Patch
     `rccl_721_gfx906_tree_ll_nthreads192_optin_20260610.patch` added
     `RCCL_GFX906_LL_NTHREADS_192=1` and set Tree/LL AllReduce launches to
     `3 * warpSize` only for single-node gfx906. The `.20` overlay built
     cleanly and avoided the memory faults seen in the earlier 128-thread
     branch, but direct TP8 graph replay regressed every tested row:
     `1x5120 0.049110 ms/call`, `2x5120 0.057039`,
     `4x5120 0.071609`, `8x5120 0.105498`, and `16x5120 0.175504`.
     Decision: reject and do not expand to `.30` or serving. Thread-count
     changes inside `rcclOptThreadBlockSize` are not sufficient without a
     deeper generated Tree/LL algorithm rewrite.
204. `NCCL_LL_BUFFSIZE` does not move the dense c1 small-message floor.
     A serial `.20` direct TP8 Tree/LL graph replay sweep under the final
     threshold envelope tested `32768`, `65536`, `131072`, `262144`, `524288`,
     and `1048576` bytes against a paired unset control. The control was
     `1x5120 0.046067 ms/call`; the best tuned point was only
     `0.046047 ms/call` at `524288`, while `32768` regressed to `0.048299`.
     All points were exact, but none approached the sub-`0.044 ms/call`
     primitive bar. Close LL buffer geometry as a dense single-request decode
     gate lever; continue with generated algorithm/dataflow work.
205. Gfx906 Tree/LL `runTreeSplit` force-inlining is a real RCCL source
     primitive milestone, but it does not clear the dense 27B gate by itself.
     Patch `rccl_721_gfx906_tree_ll_forceinline_20260610.patch` changes
     `runTreeSplit` to `__device__ __forceinline__` only for `__gfx906__`.
     The ROCm 7.2 RCCL overlays built on `.20` and `.30`, exporting
     `ncclAllReduce`, `ncclAllReduceWithBias`, and RCCL version `22707`.
     Direct TP8 Tree/LL graph replay was exact on both hosts. `.20` measured
     `1x5120 0.045104`, `2x5120 0.051396`, `4x5120 0.063153`,
     `8x5120 0.090022`, and `16x5120 0.144666 ms/call`; `.30` measured
     `1x5120 0.045300`, `2x5120 0.051310`, `4x5120 0.064096`,
     `8x5120 0.090511`, and `16x5120 0.145477 ms/call`. Promote this as the
     first cross-host exact RCCL Tree/LL source patch to lower the hot
     `1x5120` floor, but do not run serving tiers yet: the result is still
     above the sub-`0.044 ms/call` primitive bar and must be integrated with
     the promoted LL128 threshold patch before becoming a candidate image.
206. Combining the LL128 threshold overlay with `runTreeSplit` force-inlining
     is correctness-clean, but it does not improve the dense decode primitive
     enough to justify serving tiers. Patch
     `rccl_721_gfx906_tree_ll128_threshold_forceinline_20260610.patch` built
     full Tree/LL plus Tree/LL128 overlays on `.20` and `.30`. Partial full
     replay plus a short threshold replay showed exact output throughout.
     Small-row means were `.20 1x5120 0.045253`, `2x5120 0.051513`,
     `4x5120 0.063483`, `8x5120 0.090220`, `16x5120 0.145077`; `.30`
     measured `1x5120 0.045248`, `2x5120 0.051481`,
     `4x5120 0.063554`, `8x5120 0.090473`, `16x5120 0.145319`. Threshold
     rows remained in the existing threshold band: `64x5120` was about
     `0.458/0.457 ms/call` on `.20/.30`, and short `128x5120` was
     `0.790/0.795`. Decision: keep force-inline as a primitive milestone, but
     do not promote this combined overlay to serving because the hot row still
     misses the sub-`0.044` bar.
207. Generated direct-dispatch removal of `ncclDevFuncTable_[f]()` is not the
     missing dense decode lever. The first patch variant force-inlined the
     generated `ncclDevFunc_*` wrappers and failed link because cross-TU
     force-inlined device symbols were not emitted. The corrected
     `rccl_721_gfx906_tree_ll_direct_dispatch_no_table_20260610.patch` kept
     emitted wrapper symbols but generated explicit `Caller<idx, idx+1>`
     specializations that call the symbol directly instead of loading it from
     the table. It built on `.20` and was exact, but direct replay remained in
     the same band: `1x5120 0.045125`, `2x5120 0.051302`,
     `4x5120 0.063240`, `8x5120 0.090180`, and `16x5120 0.144790`.
     Reject without expanding to `.30` or serving. The next source branch
     needs to alter actual Tree/LL primitive dataflow or collective count, not
     just the generated dispatch wrapper.
208. Specializing the generic RCCL Tree/LL f16 all-reduce kernel is
     correctness-sensitive and only a small primitive milestone after fixing
     header instantiation. Patch
     `rccl_721_gfx906_tree_ll_specialized_kernel_20260610.patch` looked
     impossibly fast on `.20`, but was invalid: direct replay reported
     nonzero per-rank errors up to `35.0`, so the kernel had not performed the
     intended reduction. The corrected patch
     `rccl_721_gfx906_tree_ll_specialized_kernel_include_ar_20260610.patch`
     explicitly includes `all_reduce.h` in `common.cu` before instantiating
     the specialized `RunWorkBatch` path. That version built on `.20` and
     `.30`, exported `ncclAllReduce`, `ncclAllReduceWithBias`, and RCCL
     `22707`, and direct TP8 Tree/LL replay was exact on both hosts. `.20`
     measured `1x5120 0.044735`, `2x5120 0.050842`,
     `4x5120 0.062528`, `8x5120 0.089508`, and
     `16x5120 0.144416 ms/call`; `.30` measured
     `1x5120 0.045226`, `2x5120 0.051023`, `4x5120 0.063054`,
     `8x5120 0.089626`, and `16x5120 0.144760`. Promote this only as a
     source primitive milestone: it is the first exact branch to dip below
     `0.045 ms/call` on `.20`, but cross-host it still misses the
     sub-`0.044 ms/call` serving promotion bar and does not clear the dense
     27B `65 TPS` gate.
209. Do not set RCCL Tree kernel launch bounds below the actual HIP block
     size. Patch
     `rccl_721_gfx906_tree_ll_specialized_kernel_include_ar_lb128_20260610.patch`
     built on `.20`, but direct TP8 Tree/LL replay failed before producing
     timing rows with `HIP failure: 'unspecified launch failure'` on every
     rank. The reason is structural, not a transient startup issue:
     `enqueue.cc` forces Tree launches to `NCCL_MAX_NTHREADS`, and the HIP
     path defines `NCCL_MAX_NTHREADS` as `512`; rocprof traces also show
     `ncclDevKernel_Generic_4` running with a `512` workgroup. Therefore
     `__launch_bounds__(128)` is illegal for this kernel even when the serving
     environment sets `NCCL_NTHREADS=128`. Reject the launch-bound branch and
     do not test `256` either; both are below the actual Tree launch size.
210. Balanced 2-reduce/2-broadcast Tree/LL split is not the missing gfx906
     dense decode dataflow change. Patch
     `rccl_721_gfx906_tree_ll_specialized_kernel_include_ar_split2r2b_20260610.patch`
     stacked the corrected generic-kernel specialization with an explicit
     `nthreadsSplit = nthreads/2` LL Tree split. This corrects an earlier
     assumption error: the active HIP Tree launch uses `512` threads, so this
     is a real `256/256` split rather than a duplicate of the default. The
     overlay built on `.20` and `.30` and exported `ncclAllReduce`,
     `ncclAllReduceWithBias`, and RCCL `22707`. Direct `.20` TP8 Tree/LL
     graph replay was exact for the completed hot rows, but slower than the
     corrected specialization: `1x5120 0.044772`, `2x5120 0.051051`,
     `4x5120 0.062777`, and `8x5120 0.089943` ms/call. The run was stopped
     before spending minutes on `64x/128x` because the single-request hot rows
     had already rejected the branch. Do not promote or run serving tiers; the
     current best primitive remains the corrected specialized include branch,
     and the dense `65 TPS` gate remains open.
211. Broad LL primitive force-inlining does not promote beyond the corrected
     specialized generic kernel. Patch
     `rccl_721_gfx906_tree_ll_specialized_kernel_include_ar_primsll_forceinline_20260610.patch`
     stacks the corrected generic Tree/LL f16 all-reduce specialization with
     gfx906 `runTreeSplit` force-inlining and `__forceinline__` on hot
     `Primitives<..., ProtoLL>` helpers (`readLL`, `storeLL`, `LLGenericOp`,
     constructor/destructor, and send/recv wrappers). The first build caught a
     malformed destructor hunk; after fixing the patch, the `.20` overlay
     `/usr/share/ollama/rccl_overlay_721_gfx906_tree_ll_specialized_kernel_primsll_forceinline_20260610b_h20`
     built cleanly and exported `ncclAllReduce`, `ncclAllReduceWithBias`, and
     RCCL `22707`. Direct TP8 Tree/LL replay was exact, but the rank-mean
     graph out-of-place results were only noise-level wins at the smallest
     rows and regressions at the wider hot rows: `1x5120 0.044600`,
     `2x5120 0.050743`, `4x5120 0.062668`, `8x5120 0.089784`, and
     `16x5120 0.144529` ms/call versus the corrected specialization baseline
     `0.044735`, `0.050842`, `0.062528`, `0.089508`, and `0.144416`.
     Reject without `.30` or serving expansion. The next gate-relevant source
     target must change Tree/LL dataflow, reduce legal collective boundaries,
     or implement a fused collective/consumer primitive; more call-boundary
     cleanup is not enough.
212. The corrected specialized RCCL Tree/LL f16 kernel is now a serving
     milestone, not just a primitive milestone. The standard dense 27B serving
     ladder on `.20` with the same ROCm 7.2 TP8 fp16 surface, mutable
     RowParallel, native-runtime interleaved SwiGLU, Tree/LL, P2P disabled,
     `NCCL_MIN_NCHANNELS=4`, `NCCL_MAX_NCHANNELS=4`, and `NCCL_NTHREADS=128`,
     but with
     `/usr/share/ollama/rccl_overlay_721_gfx906_tree_ll_specialized_kernel_include_20260610_h20`,
     reached valid strict c1 `57.344`, `c1_2000 58.195`, and `c1_10000
     55.228` backend decode TPS. A `.20` repeat confirmed and slightly
     improved the result at `57.456/58.245/55.321`. That modestly but
     consistently supersedes the prior `.20` threshold ladder of
     `56.925/57.655/54.623`. The paired `.30` check reached
     `54.923/55.599/52.821`, so this is not a cross-host gate-clear and the
     dense `65 TPS` gate remains open. Carry the corrected specialized RCCL
     overlay as the current `.20` serving baseline, while continuing primary
     work on collective-boundary/dataflow changes.
213. The corrected specialized RCCL kernel does not change the promoted
     channel-count envelope. A `.30` direct TP8 Tree/LL resweep over
     `NCCL_MIN_NCHANNELS=NCCL_MAX_NCHANNELS=1..4` with the specialized overlay
     produced graph out-of-place `1x5120` times of `0.055137`, `0.046785`,
     `0.044807`, and `0.044767` ms/call respectively. Channel count 3 tied the
     hot row but lost at `4x/8x/16x`; channel 4 remained best balanced
     (`2x5120 0.052738`, `4x5120 0.062701`, `8x5120 0.090449`,
     `16x5120 0.145155`). Keep `NCCL_MIN_NCHANNELS=4` and
     `NCCL_MAX_NCHANNELS=4` for the current serving baseline; do not spend a
     serving lane on channel 3 unless a later source patch makes wider rows
     irrelevant or changes the collective shape.
214. Do not combine LL128 thresholding into the dense c1 decode winner. Patch
     `rccl_721_gfx906_tree_ll128_threshold_specialized_kernel_include_ar_20260610.patch`
     composed the corrected specialized Tree/LL kernel with the gfx906
     row-thresholded LL128 support. A first long replay accidentally omitted
     `RCCL_GFX906_TREE_LL128_MIN_BYTES=655360` from Docker and therefore only
     measured forced Tree/LL no-regression. A corrected smoke injected the env
     and was exact with `1x5120 0.045176`, `64x5120 0.458100`, and
     `128x5120 0.791199` ms/call, proving the threshold path still works.
     Serving rejected the composition: `.20` standard ladder was strict c1
     `52.301`, `c1_2000 53.463`, and `c1_10000 50.801` backend decode TPS,
     well below the specialized-only repeat of `57.456/58.245/55.321`. Keep
     LL128 thresholding as a future prefill/concurrency lever; keep the
     corrected specialized Tree/LL overlay as the dense decode baseline.
215. Current-baseline fused residual/RMS consumer lower-bound is too small for
     another consumer-only serving push. Re-running
     `rccl_ar_fused_rms_boundary_bench_20260610` with the promoted specialized
     RCCL overlays on `.20` and `.30` was exact. `.20` allreduce-only versus
     allreduce+fused-RMS was `1x5120 0.044629 -> 0.061718`,
     `2x5120 0.051014 -> 0.067531`, `4x5120 0.062816 -> 0.079396`,
     `8x5120 0.089384 -> 0.105893`, and
     `16x5120 0.144508 -> 0.161045` ms/call. The consumer delta is therefore
     only about `16-17 us` per boundary under the current baseline. MLP-only
     removal would bound around `1.06 ms/token`, and even both direct and
     attention consumers would bound around `2.1 ms/token`, below the roughly
     `2.69 ms/token` gap from current `.20 c1_10000 55.321 TPS` to `65 TPS`.
     Do not run more RMS/residual consumer-only serving lanes; the gate path
     must reduce collective cost/count or fuse below the public collective API.
216. The corrected specialized Tree/LL kernel does not benefit from simple
     split-ratio bracketing. Patch
     `rccl_721_gfx906_tree_ll_specialized_kernel_include_ar_split5r3b_20260610.patch`
     built cleanly on `.20`, but direct TP8 graph replay produced zero rank
     rows after roughly eight minutes at 100% GPU and was stopped with
     `exit_code=137`. Patch
     `rccl_721_gfx906_tree_ll_specialized_kernel_include_ar_split3r1b_20260610.patch`
     built cleanly on `.30` and was exact, but regressed every hot row:
     `1x5120 0.048817`, `2x5120 0.059558`, `4x5120 0.073663`,
     `8x5120 0.108958`, and `16x5120 0.182901` ms/call versus the `.30`
     corrected specialization baseline `0.045226/0.051023/0.063054/0.089626/
     0.144760`. Together with the prior balanced split rejection, this closes
     simple Tree/LL reduce/broadcast split tuning for the dense gate. Keep the
     corrected specialized Tree/LL overlay as the decode baseline and move
     source effort to deeper collective/dataflow or collective-count changes.
217. Grouped small-allreduce remains the largest current lower-bound primitive
     win after the specialized RCCL promotion. Re-running
     `pynccl_grouped_allreduce_bench_20260606` with the current specialized
     RCCL overlays on `.20` and `.30`, Tree/LL, P2P disabled, and
     min/max channels `4` preserved the TP8 `1x5120` grouped speedup. On `.20`,
     graph group size `32` was `0.015937 ms/call` versus `0.047514` ungrouped
     (`66.445%` faster). On `.30`, graph group size `32` was `0.015991`
     versus `0.047017` ungrouped (`65.982%` faster). This keeps grouped
     submission as the high-upside structural target, but not as a direct
     serving patch: the current graph cluster census still has
     `groupable_pairs=0` with every candidate blocked by dependency and
     intervening use. The next viable implementation must create legal
     independent same-shape reductions or achieve equivalent launch-count
     reduction in a native fused backend.
218. Intra-vector grouped slicing is not the legal shortcut to use grouped
     RCCL inside one RowParallel boundary. Testing the current specialized
     overlays on `.20` and `.30` with equivalent logical `1x5120` slice plans
     showed every plan slower than one contiguous call. On `.20`, `2 x
     1x2560` cost `0.107748 ms`, `4 x 1x1280` cost `0.075844 ms`,
     `8 x 1x640` cost `0.122800 ms`, and `16 x 1x320` cost `0.188448 ms`
     versus same-run contiguous `0.063910 ms`. On `.30`, the same plans cost
     `0.113130`, `0.085340`, `0.130672`, and `0.195648 ms` versus
     `0.066767 ms`. Grouped submission remains valuable for multiple
     independent reductions, but artificial slicing of one hidden vector is
     rejected.
219. Compile-time `NCCL_STEPS` tuning does not improve the corrected
     specialized Tree/LL f16 dense decode baseline. Steps4 built and was exact
     on `.20`, but direct replay was neutral/noise versus the current
     specialization: `1x5120 0.044645`, `2x5120 0.050684`,
     `4x5120 0.062455`, `8x5120 0.089641`, and `16x5120 0.144563`
     ms/call versus `.20` baseline `0.044735/0.050842/0.062528/0.089508/
     0.144416`. A naive steps16 patch failed RCCL's
     `NCCL_LL_CLEAN_MASK % NCCL_STEPS` assertion; the corrected steps16 patch
     aligned the mask to `0x7ffffff0` and built cleanly, but regressed on
     `.30`: `1x5120 0.087529`, `2x5120 0.052207`, `4x5120 0.065251`,
     `8x5120 0.090645`, and `16x5120 0.146331` ms/call versus `.30`
     baseline `0.045226/0.051023/0.063054/0.089626/0.144760`. Close the
     simple FIFO-depth branch; the next source path needs collective-count
     reduction, native fused collective support, or a deeper Tree/LL dataflow
     change.
220. `NCCL_LL_LINES_PER_THREAD` bracketing is not a dense decode milestone.
     Lines4 built and was exact on `.20`, but regressed wider replay rows:
     `1x5120 0.044663`, `2x5120 0.050952`, `4x5120 0.063760`,
     `8x5120 0.091339`, and `16x5120 0.147896` ms/call versus `.20`
     baseline `0.044735/0.050842/0.062528/0.089508/0.144416`. Lines16 was
     correctness-clean and mixed: `.20` measured `0.044739/0.051035/0.062822/
     0.088139/0.143922`, while `.30` measured
     `0.044640/0.051667/0.062914/0.088837/0.144619`. Because the `.20`
     `1x5120` hot row did not improve, do not promote this bracket to a
     serving milestone. The broad/latest RCCL Tree/LL primitive force-inline
     branch should be recorded as source evidence only; the actual milestone is
     the corrected specialized overlay that combines gfx906 `runTreeSplit`
     force-inline with specialized generic f16 Tree/LL AllReduce dispatch.
221. Attention TP4x2 is rejected by lower-bound before serving. The current
     specialized RCCL overlay, Tree/LL, P2P-off, min=max channels `4`, and
     corrected in-place RowParallel graph replay on the ROCm72 image showed
     current TP8 attention `1x768x5120` faster than the TP4x2-equivalent
     `1x1536x5120` on both hosts: `.20 0.081094` versus `0.096770`
     ms/call, and `.30 0.084336` versus `0.095355`. The MLP control still
     shows the isolated TP4x2 boundary lower-bound (`.20 0.351138 ->
     0.210598`, `.30 0.357302 -> 0.208520`), but the real MLP TP4x2 serving
     branch already failed because duplicated MLP compute is too expensive. Do
     not implement attention TP4x2 overlays; continue on TP8-preserving
     collective-count, launch-count, or generated Tree/LL dataflow work. The
     first in-place PyNccl replay from this lane was invalid because it used
     the old ROCm 6.3 image against a ROCm 7.2 overlay; PyNccl failed to load
     NCCL and disabled itself, causing no-op max errors `35` and `9`.
222. Dual no-acc/acc generic Tree/LL specialization is a source checkpoint,
     not a serving milestone. The patch
     `rccl_721_gfx906_tree_ll_specialized_dual_acc_20260611.patch` adds a
     second `ncclKernelMain` specialized function id so the current no-acc f16
     Tree/LL fast path for function id `0` can coexist with an acc/WithBias
     fast path for function id `1`. Minimal ROCm 7.2.1 builds on `.20` and
     `.30` exported both `ncclAllReduce` and `ncclAllReduceWithBias`. The
     `.20` no-acc direct replay preserved the promoted primitive floor
     (`1x5120 0.045199`, `16x5120 0.144484` ms/call), but the `.30` minimal
     dual overlay showed a repeated out-of-place `1x5120` anomaly
     (`0.068271`, then `0.086393`) while the current promoted overlay rechecked
     clean at `0.044719`. PyNccl WithBias improved only `1x5120` by less than
     `0.5 us` on both hosts (`.30 -0.000459 ms`, `.20 -0.000371 ms`) and
     regressed every tested `2x+` row. Keep AR-bias serving closed unless a
     future source branch fixes the wider accumulator path or proves strict
     single-row-only serving.
223. The single-work Tree/LL wrapper is a valid source insertion point, but
     not a dense decode milestone by itself. Patch
     `rccl_721_gfx906_tree_ll_singlework_fastpath_20260611.patch` adds a
     `ncclShmem.nWorks == 1` branch inside the specialized generic f16 Tree/LL
     dispatch and directly calls the Tree/LL allreduce worker for the first
     `ncclDevWorkColl`. The first version failed the ROCm/RCCL HIP transform
     because `add_unroll.sh` expands `RunWorkColl` to include `USE_ACC`,
     `COLL_UNROLL`, and `Pipeline`; the corrected patch uses
     `RunWorkColl<..., NCCL_PROTO_LL, 0, Unroll, 0>` and built cleanly on
     `.20` and `.30`. Direct graph replay was exact but neutral/mixed:
     `.20 0.044681/0.050996/0.063003/0.090022/0.144867` and
     `.30 0.044707/0.051016/0.062784/0.090090/0.145175` ms/call for
     `1x/2x/4x/8x/16x5120`. This does not justify a serving ladder or replace
     the promoted overlay. Keep the insertion point for deeper Tree/LL
     dataflow work; pure wrapper-loop shaving is too small to clear the
     dense `65 TPS` gate.
224. Widening gfx906 LL FIFO receive loads to a 128-bit vector nontemporal
     load is source-valid but not a dense decode milestone. The first
     `int4*` version failed HIP nontemporal-load typing; the corrected
     `uint32_t __attribute__((ext_vector_type(4)))` version built on `.20`
     as
     `/usr/share/ollama/rccl_overlay_721_gfx906_tree_ll_fifo_vector_load_v2_20260611_h20`
     and exported RCCL `22707`. Direct graph replay was exact but mixed
     against the promoted `.20` baseline: vector-load
     `0.044726/0.051066/0.062967/0.089907/0.144632` versus baseline
     `0.044735/0.050842/0.062528/0.089508/0.144416` ms/call for
     `1x/2x/4x/8x/16x5120`. This closes simple LL FIFO receive-load widening;
     future RCCL work needs collective-count reduction, launch-count
     reduction, fused collectives, or a deeper Tree/LL dataflow change.
225. Lowering the promoted specialized Tree/LL overlay from
     `NCCL_NTHREADS=128` to `64` is not a dense decode milestone. The `.20`
     direct graph guardrail was exact but effectively tied the promoted
     baseline: `0.044750/0.050965/0.062592/0.089497/0.144290` versus
     `0.044735/0.050842/0.062528/0.089508/0.144416` ms/call for
     `1x/2x/4x/8x/16x5120`. Keep `NCCL_NTHREADS=128`; the remaining gate
     work needs a material collective-boundary change, not a lower requested
     worker thread count.
226. Hybrid leaf up/down Tree/LL is source-valid but not a dense decode
     milestone. The branch keeps the promoted gfx906 `runTreeSplit`
     force-inline and specialized f16 Tree/LL dispatch, but lets leaf ranks run
     all threads for reduce-up and then all threads for broadcast-down. The
     `.20` direct replay was exact and measured
     `0.044668/0.051558/0.066323/0.098559/0.164204` ms/call versus promoted
     `.20` baseline `0.044735/0.050842/0.062528/0.089508/0.144416` for
     `1x/2x/4x/8x/16x5120`. The hot-row improvement is below one tenth of a
     microsecond and every wider row regresses materially, so do not replicate
     on `.30` or run a serving ladder. Simple leaf-only phase splitting is not
     enough; future Tree/LL work must change full-rank dataflow, collective
     count, launch count, or fused-collective semantics.
227. Combining the single-work Tree/LL wrapper with `NCCL_STEPS=4` is not a
     dense decode milestone. The `.20` overlay built cleanly as
     `/usr/share/ollama/rccl_overlay_721_gfx906_tree_ll_singlework_steps4_20260611_h20`
     and direct replay was exact, but measured
     `0.044764/0.050825/0.062742/0.089540/0.144269` ms/call versus promoted
     `.20` baseline `0.044735/0.050842/0.062528/0.089508/0.144416` for
     `1x/2x/4x/8x/16x5120`. The hot `1x5120` row is slightly slower, and the
     wider-row changes are mixed/noise-scale. This closes the obvious safe
     composition of two prior RCCL micro-branches; further gate progress needs
     a boundary-level collective/count/fusion change.
228. Upstream RCCL `develop_deprecated` is not a gfx906 dense decode
     milestone as a wholesale branch replacement. A compatibility patch
     (`rccl_develop_deprecated_rocm72_compat_20260611.patch`) was required to
     build it in the ROCm 7.2 test image: include `nccl_tuner.h` before
     `rccl_common.h` uses tuner constants, make `nvtx.h` honor the
     `NVTX_NO_IMPL` stub path, add the missing `NCCL_NVTX3_FUNC_RANGE` no-op,
     and fix the fallback typo `ncclSymGetKernelPtr` ->
     `ncclSymkGetKernelPtr`. The `.20` minimal overlay built and loaded as
     `/usr/share/ollama/rccl_overlay_upstream_devdep_compat_symstub_gfx906_20260611_h20`
     with `ncclGetVersion 22803`, but the direct AllReduce graph replay hung
     before the first `1x5120` row and had to be killed with exit `137` while
     all eight GPUs were busy. Enabling generated symmetric kernels exposed a
     broader incompatible device path (`ncclCoopCta`, `cuda::memory_order`,
     `ncclSymPtr` arithmetic). Keep the promoted ROCm 7.2.1 specialized
     Tree/LL overlay as baseline; future upstream work should be selective
     cherry-picks, not branch replacement, unless the branch first passes
     direct no-acc replay.
229. Upstream RCCL `ENABLE_WARP_SPEED` is not a dense TP8 milestone for this
     gfx906 path. It is Ring-only in upstream `develop_deprecated`, not a
     Tree/LL improvement. A `.20` overlay built with
     `-DENABLE_WARP_SPEED=ON` and
     `ONLY_FUNCS="AllReduce RING LL Sum f16"` exported the expected symbols,
     and a tiny `1x5120` smoke passed, but the comparable direct replay was
     slower than the promoted Tree/LL overlay (`1x5120 0.048850` and
     `2x5120 0.070821` versus promoted `.20` Tree/LL
     `0.044735 / 0.050842`) and then stalled before `4x5120`. Do not use
     upstream WarpSpeed as a serving base; any future Ring path must first beat
     the promoted Tree/LL direct row set.
230. Channel-split endpoint RMS is a source lower-bound milestone. Forcing
     one Tree/LL channel is not viable (`1x5120` allreduce-only regressed to
     `0.0529-0.0600 ms` and `16x5120` to about `0.2957 ms`), so any real
     fused collective/consumer must preserve the four-channel schedule. A
     new split-endpoint simulator in
     `rccl_ar_fused_rms_boundary_bench_20260610.cpp` models one block per
     `(row, channel)` plus a global row-completion counter, uses FP32
     residual input/output to match the dense boundary, and saved roughly
     `5 us` per boundary versus the current separate
     `AllReduce -> fused_add_rms_norm` graph node on `.20` and `.30` stable
     rows. Promote this as a lower-bound/source milestone only; it is not a
     dense `65 TPS` gate clear until a real RCCL Tree/LL post-RMS metadata
     path preserves the same advantage in serving.
231. The RCCL Tree/LL post-RMS metadata branch is an API plumbing milestone
     but not a decode performance milestone. The first implementation failed
     export validation because ROCm 7.2 public collective wrappers are emitted
     through `misc/api_trace.cc` and `rcclApiFuncTable`; adding only
     `collectives.cc` and `nccl.h.in` was insufficient. After adding the API
     trace typedef/table slot/wrapper, both `.20` and `.30` minimal overlays
     exported `ncclAllReduceWithPostRms`, and vLLM's PyNccl path can call the
     symbol. However, the earlier near-free post-RMS timing rows were invalid:
     the C++ bench reused output buffers from the standalone split-endpoint
     simulator, so a skipped post-RMS hook could inherit correct outputs.
     After clearing outputs before the post-RMS mode, split-4 metadata is
     correctly rejected (`residual_error=36.0896`, hook skipped) because RCCL
     plans the hot `1x5120` Tree/LL work item as `channel{Lo..Hi}={0..2}`.
     The valid split-3 path is correct but slow: C++ `.20` measured
     AllReduce-only `0.054626 ms`, separate fused RMS `0.070951 ms`,
     split-endpoint simulator `0.066264 ms`, and RCCL post-RMS metadata
     `0.074058 ms`; PyNccl stable split-3 measured `0.046939 -> 0.065983 ms`.
     Do not run a serving ladder with this metadata kernel as-is. The next
     source step should preserve the split-endpoint lower-bound algorithm
     while making the hook/graph op aware of the actual RCCL work partition,
     or implement the split endpoint as a separate guarded vLLM custom graph
     op.
232. The standalone PyNccl split-endpoint RMS bridge is the corrected source
     milestone for the dense post-attention boundary. The bridge compiles a
     tiny HIP shared library inside the ROCm 7.2 image and times
     PyNccl AllReduce-only, PyNccl AllReduce plus one-block fused add/RMS, and
     PyNccl AllReduce plus split-endpoint add/RMS under HIP graph replay. On
     `.20`, `1x5120` measured `0.046896 ms` AR-only, `0.064505 ms` AR+fused,
     and `0.058956 ms` AR+split; on `.10`, it measured `0.046959`,
     `0.063868`, and `0.058937 ms`. Correctness was clean on both hosts
     (`residual_error=0`, norm error `0.000976562`). Promote this as a
     component/source milestone: it saves about `4.9-5.5 us` per hot boundary
     versus the one-block fused RMS consumer in the same PyNccl graph context.
     It is not enough alone to clear the dense `65 TPS` gate; the next
     implementation step is a guarded vLLM Qwen3.5 post-attention overlay for
     `rows == 1`, `cols == 5120`, fp16 allreduce output, fp32 residual, fp16
     effective gamma, TP8 Tree/LL, and split count `3`, with fallback for every
     other shape.
233. The Qwen3.5 split-endpoint RMS serving overlay is not a dense decode
     milestone. The guarded overlay was source-aligned to the ROCm 7.2 image
     `qwen3_5.py`, mounted the HIP bridge from the `.20` PyNccl milestone, and
     was correctness-valid on strict c1, but it did not improve fixed-token
     serving under the corrected async-on parent. The initial same-day
     overlay/control pair was async-off and therefore not comparable to the
     promoted dense winner. The corrected `.20` no-overlay async-on control
     reproduced the winner band at strict c1 `57.183`, `c1_2000 58.244`, and
     `c1_10000 55.272` backend TPS. The overlay async-on direct/curl run
     reached strict c1 `57.236`, `c1_2000 58.034`, and `c1_10000 55.089`.
     Treat this as a reject for serving: a component-level endpoint RMS win
     does not automatically survive a Python/model-forward override under O3
     graph capture. Keep async scheduling recorded in dense runner settings,
     and keep `CHAT_CAPTURE_USE_CURL=1` available for runs where the local
     Python HTTP POST path opens a socket but does not register a request. The
     next serious path should stay below the model forward boundary:
     collective count reduction, coalescing, or RCCL Tree/LL primitive work.
234. The latest/broad RCCL Tree/LL primitive force-inline branch should not be
     promoted as its own milestone. The promoted substrate milestone remains
     the corrected specialized generic f16 Tree/LL AllReduce path with gfx906
     `runTreeSplit` force-inline, because that is the branch tied to measured
     serving movement. Broad force-inline is supporting source evidence unless
     it independently improves the standard dense ladder.
235. Short `NCCL_DEBUG` runtime-only shape logging under graph capture does not
     prove steady decode shape frequency. The `.20` runtime-only diagnostic
     zeroed RCCL logs after readiness and then sent a capped request; the clean
     histogram showed AllReduce `count=102400`, dtype `6`, exactly
     `20x5120`, matching the 20 prompt tokens, plus AllGather `count=31040`.
     No pure `1x5120` steady decode AllReduce row was exposed. The next primary
     source branch should either support the row family seen during
     capture/request setup or use a stronger decode-isolated diagnostic before
     specializing only the one-row path.
236. Streaming decode confirms that `NCCL_DEBUG` is not a steady replay
     attribution tool under the current O3 captured dense graph. In
     `qwen36_27b_decode_stream_shape_diag_h20_20260611_034732`, chat streaming
     produced 129 SSE chunks and finished in `36.645s`; after the first
     non-empty stream chunk, all eight RCCL debug files were cleared from
     inside the container and stayed at zero bytes through the remaining
     decode. Treat RCCL debug output as construction/capture/request evidence,
     not per-token replay evidence. Future dense source branches should use
     component replays or vLLM graph/source instrumentation and should be
     guarded for the observed row family, not a blind strict `1x5120` fast path.
237. The split-endpoint RMS bridge does not generalize cleanly enough to become
     the next row-family serving base. On `.20`, the row-family bridge
     (`1/2/4/8/20/64/128x5120`) remained correctness-clean and was
     `3.7-6.3 us` faster than the one-block fused RMS consumer at every tested
     shape. On `.30`, correctness was still clean, but the split endpoint
     regressed badly at `2x5120` and especially `20x5120` (`0.360384 ms`
     versus fused RMS `0.194843 ms`). Keep the original hot `1x5120`
     split-endpoint bridge as a source/component milestone only. The next
     serious serving source path should use the simpler fused post-allreduce
     consumer through a safer native vLLM graph replacement path, guarded by
     row family, rather than the current global-counter split-endpoint design.
238. The Qwen3.5 MLP-input split-endpoint serving overlay is correctness-clean
     but not a dense decode promotion. The overlay pre-normalizes the next
     layer input after a layer returns, using the split-endpoint RMS bridge at
     the direct MLP-down boundary while preserving the current dense winner
     envelope. A clean `.20` direct/curl ladder reached strict c1 `57.052`,
     c1_2000 `57.916`, and c1_10000 `54.957` backend TPS; a prior direct smoke
     reached c1 `57.427`. This is below the current async-on no-overlay control
     (`57.183 / 58.244 / 55.272`) and below the corrected RCCL overlay winner
     (`57.456 / 58.245 / 55.321`). Reject additional model-forward
     split-endpoint placements as the primary path; component-level
     microsecond savings are being consumed in serving integration. Future
     gate work should stay below the model-forward boundary or reduce
     collective count/coalescing directly in vLLM/RCCL.
239. Host `.10` is prepared as a third current-winner dense test lane,
     and `RCCL_OVERRIDE_ALGO/PROTO` is rejected as a serving promotion. `.10`
     already had the ROCm 7.2 experimental image, Qwen3.6-27B snapshot,
     mutable RowParallel overlays, and native interleaved SwiGLU extension; the
     current specialized RCCL overlay was copied from `.20` to
     `/usr/share/ollama/rccl_overlay_721_gfx906_tree_ll_specialized_kernel_include_20260610_h10`.
     The `.10` direct sanity replay was exact and matched the winner band:
     `1x/2x/4x/8x/16x5120 = 0.044710/0.050998/0.062900/0.089617/0.144482`.
     The strict uncapped `.10` serving smoke was also valid through the Qwen
     thinking gate with `3419` completion tokens, `45.093` client TPS, `53.840`
     backend decode TPS, `35.103` backend prefill TPS, `finish_reason=stop`,
     and `qwen_gate_valid=true`. Treat `.10` as usable for serving/source
     checks but not as the performance reference; `.20` remains the current
     global dense winner.
     Cross-host override component checks with
     `RCCL_OVERRIDE_ALGO=TREE` and `RCCL_OVERRIDE_PROTO=LL` were exact but
     mixed: `.20` measured `0.044710/0.051101/0.062909/0.090008/0.144612`,
     and `.30` measured `0.044676/0.051217/0.063031/0.090019/0.144725`.
     The hot row ties/slightly improves, while wider rows regress or mix, so
     do not burn serving ladders on override-only tuning. Keep the serving
     envelope on `NCCL_ALGO=Tree` / `NCCL_PROTO=LL` unless a future RCCL source
     branch first produces a real row-family primitive win.
240. `.10` is usable as a third lane, but not as the dense performance
     reference, and the single-work plus `NCCL_LL_LINES_PER_THREAD=16` RCCL
     merge is only a component checkpoint. The `.10` full current-winner ladder
     `qwen36_27b_decode_tiers_rocm72_lane_prep_h10_full_host10_20260611`
     reached strict c1 `53.913`, `c1_2000 53.206`, and `c1_10000 49.862`
     backend TPS, below the `.20` reference band. The merged RCCL patch
     `rccl_721_gfx906_tree_ll_singlework_ll_lines16_20260611.patch` built
     cleanly on `.20`/`.30` and was exact in direct graph replay. It improved
     `8x5120` and `16x5120` on both hosts (`.20` `0.088381/0.144009`,
     `.30` `0.088340/0.144556`) and tied the hot row after the `.30` repeat
     (`0.044813`), but regressed `2x5120` and `4x5120`. Keep it as source
     evidence for wider-row/prefill/concurrency work; do not promote it to the
     dense single-request decode stack.
241. The three-lane single-work + LL16 + `prims_ll` force-inline composition
     is exact but still not a dense decode promotion. Patch
     `rccl_721_gfx906_tree_ll_singlework_ll_lines16_primsll_inline_20260611.patch`
     built cleanly on `.10`, `.20`, and `.30` and exported RCCL `22707`.
     Direct graph replay produced `.10`
     `0.044642/0.051331/0.063288/0.088532/0.144272`, `.20`
     `0.044662/0.051296/0.062788/0.088538/0.144189`, and `.30`
     `0.076983/0.051173/0.064402/0.088464/0.144097` for
     `1x/2x/4x/8x/16x5120`, all exact. The branch helps `8x/16x` rows, but
     it regresses `2x/4x` on every host and repeats the `.30` out-of-place
     hot-row anomaly. The in-place graph column also stays mixed; `.20`
     regresses `1x/2x/4x` versus the promoted overlay and only helps `8x/16x`.
     Do not run serving tiers. Keep it as wider-row source evidence; dense
     single-request decode still needs collective count, launch count, or a
     fused backend path with a much larger row-family win.
242. The single-work + LL16 + `prims_ll` RCCL branch is rejected at the
     serving layer by measured TPS, not by token-generation failure. The first
     `.20` current-envelope launch was too impatient and should not be treated
     as the branch verdict. A patient three-lane rerun using fresh runtime
     roots and
     `rccl_overlay_721_gfx906_tree_ll_singlework_ll16_primsinline_20260611`
     served valid strict c1 on all hosts. Results were `.10`
     `53.189/52.325/47.998`, `.20` `57.546/58.234/55.274`, and `.30`
     `55.034/55.663/52.910` for strict c1, c1_2000, and c1_10000 backend
     decode TPS. The `.20` warm tiers are slightly below the promoted global
     overlay (`57.456/58.245/55.321` on the reference repeat), and `.10`/`.30`
     are lower. Keep this branch as wider-row source evidence only; it does
     not replace the global dense winner.
243. The ROCm 7.2 post-IR RMS image is not a current-envelope third-lane
     promotion. On `.30`, `qwen36-gfx906-rocm72-postir-rms:20260608` plus the
     promoted specialized RCCL overlay, mutable RowParallel overlays,
     native-runtime interleaved SwiGLU, async scheduling, and maxseq4 completed
     cold start and graph capture, but the post-IR pass selected zero current
     direct MLP boundaries and the strict smoke produced only the begin-think
     marker with zero engine token counters. Do not copy this image to `.10`
     for full tiers; future post-IR work must first prove current-stack boundary
     selection and a valid strict smoke before taking performance lanes.
244. The RCCL post-RMS phase-counter correction is a source correctness
     milestone, not a dense decode promotion. Patch
     `rccl_721_gfx906_tree_ll_post_rms_parallel_norm_phase_20260611.patch`
     keeps the public `ncclAllReduceWithPostRms` ABI stable but changes the
     backing buffers to `rows * (split_channels+1)` partial/inverse slots and
     `rows * 2` arrival/release counters. It fixes the previous `inf`
     normalization failure for matched split-2 and split-3 rows across
     `1/2/4/8/20x5120`, and all three hosts built/exported the symbol. The
     performance result rejects the path for serving: `.20` split-3 `1x5120`
     costs `+12.838 us` over AllReduce, worse than the external split-endpoint
     parallel endpoint at `+8.026 us`, and wider rows balloon to
     `+29.170/+72.178/+162.054/+434.827 us` at `2/4/8/20x5120`. Do not put
     row-level spin rendezvous inside the Tree/LL kernel on the dense decode
     path; the next serious work should reduce/coalesce RowParallel collective
     count, move fusion below the public API with a non-rendezvous dataflow, or
     improve the Tree/LL primitive itself.
245. Corrected-shape RowParallel output chunking is rejected on all three
     hosts. The current dense boundaries are `attn_5120x768` and
     `mlp_down_5120x2176`, not the stale `5120x640/3696` shapes. Under the
     promoted min/max-channel-4 Tree/LL envelope, full graph MLP-down measured
     `.10/.20/.30` at `0.073183/0.071996/0.073772 ms`. The best 2-way
     sequential chunked MLP-down was `0.131430/0.130157/0.133979 ms`, and the
     best stream-pipelined chunked MLP-down was
     `0.366226/0.272128/0.353873 ms`. Attention showed the same rejection.
     Do not slice one exact RowParallel output vector into multiple
     allreduces; the extra collective launches erase any overlap opportunity.
     Any useful compute/collective fusion must avoid increasing collective
     count.
246. Grouped allreduce remains a serious source-planning lever for small and
     medium decode row families, but not for huge prefill-like rows. After
     preparing `.10` as a third lane, `pynccl_grouped_allreduce_bench` was
     rerun on `.10` and `.30` with the promoted ROCm 7.2 specialized Tree/LL
     overlay, TP8, min=max channels `4`, and row families
     `1/2/4/8/20/21/431x5120`. At group size `32`, both hosts showed about
     `67%` speedup for `1x5120`, `58%` for `2x5120`, `45%` for `4x5120`,
     `30%` for `8x5120`, and `14%` for `20/21x5120`. The `431x5120` family
     improved only about `1%`. Keep grouped/count-reduction work focused on
     legal same-shape decode reductions or an equivalent launch-count
     reduction; do not treat grouping as a large-prefill fix.
247. The gfx906 LL FIFO vector-store branch is rejected on three lanes. Patch
     `rccl_721_gfx906_tree_ll_specialized_kernel_include_ar_ll_fifo_vector_store_20260611.patch`
     replaced the two scalar `u64` FIFO stores in `storeLL` with one
     `uint32x4` vector store while preserving the promoted specialized f16
     Tree/LL kernel. It built and exported RCCL `22707` on `.10`, `.20`, and
     `.30`, and direct graph replay was exact for `1/2/4/8/16x5120`.
     However, the key graph-out column regressed on all hosts: `.10`
     `0.045344/0.051555/0.063244/0.090262/0.145197`, `.20`
     `0.045367/0.051559/0.063237/0.090350/0.144894`, and `.30`
     `0.045292/0.051741/0.063321/0.090326/0.145068`, versus promoted
     baselines `.10` `0.044710/0.050998/0.062900/0.089617/0.144482`,
     `.20` `0.044735/0.050842/0.062528/0.089508/0.144416`, and `.30`
     `0.045226/0.051023/0.063054/0.089626/0.144760`. This is useful negative
     source evidence: the compiler/RCCL scalar-store form is better for the LL
     FIFO on gfx906 than the forced vector store. Do not use this patch as a
     serving or component promotion.
248. The one-send Tree/LL fastpath is exact but not a source promotion. Patch
     `rccl_721_gfx906_tree_ll_specialized_kernel_include_ar_one_send_fastpath_20260611.patch`
     built on `.10`, `.20`, and `.30` and preserved exact direct graph replay,
     but the graph-out column only tied or slightly regressed the promoted
     specialized overlay. Results for `1/2/4/8/16/20x5120` were `.10`
     `0.044722/0.051076/0.062886/0.089658/0.144505/0.173057`, `.20`
     `0.044588/0.050990/0.062850/0.089683/0.144832/0.174065`, and `.30`
     `0.044695/0.051120/0.062901/0.089675/0.144782/0.173668`. The promoted
     baselines remain `.10` `0.044710/0.050998/0.062900/0.089617/0.144482`,
     `.20` `0.044735/0.050842/0.062528/0.089508/0.144416`, and `.30`
     `0.045226/0.051023/0.063054/0.089626/0.144760`. Reject this branch; it
     does not move the small-message floor enough to justify serving.
249. The corrected PyNccl split-endpoint RMS bridge is a three-lane source
     milestone after fixing the runner to mount the host RCCL overlay into the
     container. The earlier bridge attempts were invalid because PyNccl could
     not load the intended RCCL and the CUDA graph could become empty. The
     corrected runner records `max_ar_err`, rejects empty graphs, mounts
     `/rccl-overlay`, and uses the promoted host-specific RCCL overlays. Valid
     bridge runs were
     `pynccl_split_endpoint_rms_bridge_host10_20260611_split_endpoint_bridge_valid_h10`,
     `...h20`, and `...h30`. For `1x5120`, `.10` measured AR-only `0.045434`,
     AR+fused `0.062962`, AR+split `0.057351`; `.20` measured
     `0.045594/0.062170/0.057480`; `.30` measured
     `0.045935/0.062678/0.057056`. The split endpoint also stayed faster than
     the one-block fused consumer across `2/4/8/16/20x5120`, with zero AR
     error and residual error and fp16 norm error `0.000976562`. Keep this as
     source evidence for endpoint consumer fusion outside RCCL row-level spin,
     not as a serving win by itself.
250. Host `.10` is now active as a third lane for the dense source campaign,
     and the post-attention split-endpoint serving overlay is rejected by the
     standard three-host decode ladder. Actual VRAM capacity was checked by
     bytes, not product-name strings: all eight GPUs on `.10`, `.20`, and `.30`
     report `34342961152` bytes total. `.10` has the ROCm 7.2 image, Qwen3.6
     27B weights, promoted RCCL overlay, Qwen3.5 split-endpoint overlay, and
     `libgfx906_split_endpoint_rms.so`. The current post-attention overlay
     served valid strict c1 on all hosts, but fixed-token decode did not beat
     the `.20` global winner (`57.456/58.245/55.321` backend TPS). Results:
     `.10` `53.878/52.785/48.106`, `.20` `56.882/57.957/54.956`, and `.30`
     `54.875/55.628/52.822` for strict c1, c1_2000, and c1_10000 backend
     decode TPS. The serving integration is neutral to negative despite the
     component bridge win. Continue the primary path below the model-forward
     boundary: collective count reduction, legal coalescing, or deeper RCCL
     Tree/LL/fused-endpoint source work.
251. QuickReduce is rejected on the current ROCm 7.2 gfx906 TP8 stack. The
     runner first exposed a portability bug: `REMOTE_DOCKER_ENV_FILE=none`
     was treated as a real file, unlike the newer campaign runners. That is
     fixed in `run_quickreduce_direct_microbench_20260606.sh`. After the fix,
     `.10`, `.20`, and `.30` all failed the exact dense TP8 row-family sweep
     (`1/2/4/8/16/20x5120`, FP16) with HIP errors and rank segfaults; every
     shape had zero passing rank groups. Run dirs:
     `quickreduce_direct_bench_host10_20260611_quickreduce_current_fixed_h10`,
     `...h20`, and `...h30`. No stale containers or VRAM remained after the
     sweep. Do not spend serving-source time wiring QuickReduce into the dense
     winner until its direct TP8 FP16 path is made correctness-valid on gfx906.
     The near-term gate path remains the promoted RCCL Tree/LL base plus legal
     collective schedule changes or a deeper exact fused endpoint primitive.
252. The focused three-lane promoted Tree/LL graph-out baseline is now
     refreshed with `.10` included. The first attempt included `431x5120` at
     `2000*64` graph calls and was intentionally stopped because that
     prefill-sized point would consume the lanes for minutes without deciding
     the single-request decode gate. The bounded rerun used the promoted
     host-specific RCCL overlays, TP8 FP16, Tree/LL, P2P disabled,
     `NCCL_MIN_NCHANNELS=4`, `NCCL_MAX_NCHANNELS=4`, `NCCL_NTHREADS=128`, and
     graph-out only for rows `1/2/4/8/16/20/21/32x5120`. All hosts were exact.
     Cross-host graph-out means were `1x5120 0.044758`, `2x5120 0.051052`,
     `4x5120 0.062892`, `8x5120 0.089850`, `16x5120 0.144508`,
     `20x5120 0.173230`, `21x5120 0.181381`, and `32x5120 0.258667` ms/call.
     This is the current component floor for dense source branches. With
     `128` RowParallel reductions per token, the hot `1x5120` floor alone is
     about `5.73 ms/token`, so the `65 TPS` gate still needs a structural
     collective/count/fusion change rather than another percent-level LL
     micro-edit.
253. Host `.50` is staged as a topology-variant lane, but it is not the same
     promotion class as `.10/.20/.30`. It has the ROCm 7.2 experimental image
     `sha256:8c91d37aacef...`, the Qwen3.6-27B model cache, the promoted
     specialized RCCL overlay copied as
     `rccl_overlay_721_gfx906_tree_ll_specialized_kernel_include_20260610_h50`,
     and the current overlay/hotfix files. The hardware difference matters:
     `.50` is a dual-socket Intel E5-2667 v4 host with two PLX four-GPU switch
     groups, while `.10/.20/.30` are single-socket EPYC 7F32 hosts with four
     two-GPU switch/root-complex groups. The eight MI50 links still report
     `16.0 GT/s PCIex16`, but the topology and CPU/NUMA behavior differ.
     `.50` also has a separate 4 GiB display adapter at ROCm-SMI GPU0
     (`/dev/dri/card1`, `/dev/dri/renderD128`). Per user instruction, do not
     use that device. ROCm 7.2 inside Docker fails if the container is run
     `--privileged` with all of `/dev/dri`, because ROCr sees the non-gfx906
     adapter and PyTorch raises `RuntimeError: No CUDA GPUs are available`
     during HIP init. The working policy is non-privileged Docker plus
     explicit MI50 device nodes only. Under that policy, Torch initializes and
     runs kernels on all eight MI50s. RCCL is narrower: TP4 graph replay on
     ROCm-SMI GPUs 1-4 passed (`1x5120 0.037004`, `8x5120 0.115374`,
     `32x5120 0.374567` ms/call), and pair smokes for GPUs 5-6 and 6-7
     passed, but pairs involving ROCm-SMI GPU8 failed during communicator
     setup under both the promoted RCCL overlay and stock ROCm 7.2 RCCL.
     A ROCm 6.3 compatibility check was inconclusive because the newer graph
     benchmark source uses `ncclWindow_t`, which is absent from the older 6.3
     headers, so it did not reach communicator setup. A later isolated GPU8
     compute sanity check exposed only `/dev/dri/card8` and
     `/dev/dri/renderD136` and completed repeated PyTorch fp16 matmuls under
     ROCm 7.2. The full `.50` TP8 direct graph run therefore failed before
     producing valid rows, but current evidence points to a `.50` host/GPU8
     RCCL communicator/topology issue rather than a proven dead GPU. Treat
     `.50` as staged and Torch-ready, but not TP8-RCCL-ready for the dense gate
     until the host/GPU8 RCCL communicator issue is resolved without exposing
     ROCm-SMI GPU0. Promotions remain anchored on the identical `.10/.20/.30`
     hosts.
254. Grouped-allreduce is now split into "real lower-bound" and "not legal in
     the current compiled serving graph." The `.20` reference refresh
     `runs/pynccl_grouped_allreduce_bench_host20_20260611_promoted_decode_rows_h20`
     used the promoted specialized Tree/LL overlay, TP8 FP16, P2P disabled,
     min=max channels `4`, and decode rows `1/2/4/8/20/21x5120`. At group
     size `32`, rank-mean timings again showed a major launch-amortization win:
     `1x5120` `0.046789 -> 0.015254 ms/call` (`67.399%` faster),
     `2x5120` `0.053055 -> 0.022072` (`58.397%`), `4x5120`
     `0.064825 -> 0.035843` (`44.709%`), `8x5120`
     `0.092036 -> 0.064035` (`30.424%`), and `20/21x5120` about
     `14-15%`. However, the current winner post-IR grouping census on
     `.10/.20/.30` found `129` collective ops per rank and `0` groupable
     adjacent pairs on every rank, with all `128` adjacent pairs blocked by
     both dependency and intervening use:
     `qwen36_27b_decode_tiers_currentwinner_grouping_census_h{10,20,30}_host{10,20,30}_20260611_postir_register_freshcache_h{10,20,30}/postir_collective_census.md`.
     This promotes grouped small-allreduce only as a lower-bound primitive and
     rejects a naive grouped PyNccl/RCCL serving wrapper. The next source work
     must either create independent reductions through a semantic schedule
     change, reduce the RowParallel collective count by construction, or keep
     digging into a deeper exact-shape RCCL/fused-endpoint primitive.
255. Wider Tree/LL channel forcing is rejected on the promoted specialized
     overlay. The focused `.20/.30` direct graph sweeps
     `rccl_direct_allreduce_graph_bench_host20_20260611_specialized_minmax8_graphout_h20`
     and
     `rccl_direct_allreduce_graph_bench_host30_20260611_specialized_minmax8_graphout_h30`
     used TP8 FP16, Tree/LL, P2P disabled, `NCCL_MIN_NCHANNELS=8`,
     `NCCL_MAX_NCHANNELS=8`, `NCCL_NTHREADS=128`, graph-out only, and decode
     rows `1/2/4/8/16/20/21/32x5120`. Both hosts were exact, but results
     regressed the three-lane min=max=4 floor: `.20` measured
     `0.044879/0.051533/0.071080/0.100191/0.156954/0.183567/0.190617/0.272875`,
     and `.30` measured
     `0.045131/0.051744/0.070960/0.100421/0.156668/0.183440/0.190411/0.272918`
     ms/call. Compared with promoted means of
     `0.044758/0.051052/0.062892/0.089850/0.144508/0.173230/0.181381/0.258667`,
     extra channels do not improve the hot row and materially hurt medium
     rows. Keep min=max channels `4`; do not use channel widening as the fused
     endpoint or RowParallel collective base.
256. Two cold-start patience revisits close without promotion. The `.10`
     current-winner `max_num_seqs=8` recheck
     `qwen36_27b_decode_tiers_rocm72_dense_currentwinner_maxseq8_recheck_h10_host10_20260611_121036`
     completed after the expected slow first uncapped thinking request and
     produced strict c1 `53.866`, c1_2000 `53.817`, and c1_10000 `50.566`
     backend TPS. This is below the `.20` global winner and keeps
     `max_num_seqs=4` as the dense decode default. The corrected `.30`
     windowed active post-IR RMS/residual replacement
     `qwen36_27b_decode_tiers_postir_window_fused_both_start64_limit16_h30_rmsimage_host30_20260611_121431`
     used the RMS-capable image `qwen36-gfx906-rocm72-postir-rms:20260608`,
     selected `16/128` direct mutable RowParallel boundaries, and completed a
     valid strict smoke at `54.849` backend TPS. This proves the windowed
     matcher/replacement path can actively rewrite current-stack boundaries,
     but the result is still below the winner. Consumer-only post-IR fusion
     remains source infrastructure, not a gate path; keep source effort on
     RowParallel collective count, coalescing legality, or lower-level exact
     small-message collective/fused-endpoint work.
257. Keep decode-runner defaults aligned with the promoted overlay. The
     `singlework_ll16_primsinline` branch is source-valid but rejected by the
     patient three-lane serving results in item 242. The helper
     `run_qwen36_27b_rccl_overlay_decode_lane_20260611.sh` has been reset to
     default `OVERLAY_KIND=currentwinner`; use
     `OVERLAY_KIND=singlework_ll16_primsinline` only for explicit guardrail or
     wider-row experiments.
258. Split-endpoint MLP-input serving integration is valid but not a dense
     decode promotion. Host `.20` run
     `qwen36_27b_decode_tiers_rocm72_split_endpoint_mlp_input_direct_full_h20_host20_20260611_044349`
     used the current ROCm 7.2 TP8 winner envelope plus the split-endpoint
     RMS custom op at the `input_layernorm`/MLP-input boundary. It produced
     valid strict c1 `57.052`, c1_2000 `57.916`, and c1_10000 `54.957`
     backend TPS, below the `.20` reference `57.456/58.245/55.321`. Together
     with the rejected post-attention split-endpoint overlay, this closes
     Python/model-overlay placement of the bridge as a gate path. Any remaining
     endpoint work must happen inside or immediately underneath RCCL/Tree/LL,
     not as another model-forward custom-op placement.
259. Final-envelope `NCCL_NTHREADS=256/512` is closed as a guardrail, not a
     promotion. New exact graph-out runs used the promoted specialized
     Tree/LL overlays, TP8 FP16, P2P disabled, min=max channels `4`, and rows
     `1/2/4/8/16/20/21/32x5120`: `.10`
     `rccl_direct_allreduce_graph_bench_host10_20260611_specialized_nthreads256_graphout_h10`,
     `.20` `...nthreads256_graphout_h20`, `.30`
     `...nthreads256_graphout_h30_repeat`, plus `.20/.30`
     `...nthreads512_graphout_h{20,30}`. All rows were exact. The best
     `256` data shaved sub-microsecond amounts on `.10/.20` wider rows, but
     `.30` repeated a real wider-row loss (`8x5120 0.092215` vs promoted
     `.30` `0.089690`, and `20x5120 0.174553` vs `0.172791`). `512` was also
     mixed and worse on `.30` (`8x5120 0.099362`). Keep
     `NCCL_NTHREADS=128` in the dense serving envelope. Thread-count tuning is
     not a credible remaining path to the `65 TPS` gate; continue with
     collective count/coalescing legality or a deeper exact Tree/LL/fused
     endpoint primitive.
260. Host `.50` GPU8 is compute-good but TP8-RCCL-transport-bad under the
     normal fast path. The isolated diagnostic
     `runs/host50_gpu8_diag_20260611_121452` used non-privileged Docker with
     explicit MI50 DRM nodes only, never exposing ROCm-SMI GPU0
     (`0000:01:00.0`, `0x699f`, `/dev/dri/card1`, `renderD128`). Static
     inventory showed GPU8 as BDF `0000:1f:00.0`, DID `0x66a1`, VRAM
     `34342961152` bytes, PCIe `16.0 GT/s x16`, NUMA `0`, replay count `0`.
     Isolated PyTorch/HIP compute passed on all allowed GPUs `1..8`; GPU8
     exposed as only `/dev/dri/card8` plus `/dev/dri/renderD136` completed
     fp16 matmul, a `20640169984` byte allocation/touch, and a 20 second
     stress loop. Corrected stock ROCm 7.2 RCCL baseline passed adjacent pairs
     `1-2` through `6-7`, GPU8 pairs `1/2/3/4-8`, TP4 groups `1,2,3,4`,
     `4,5,6,7`, and `1,2,3,8`; it failed GPU8 pairs `5/6/7-8`, group
     `5,6,7,8`, and full `1..8` with RCCL `transport/p2p.cc` HIP
     `invalid argument` plus SHM attach failures against `/dev/shm/nccl-`.
     The requested env variations `NCCL_P2P_DISABLE=1`, Tree/LL, channel
     count `1`, channel count `4`, and the h50 specialized overlay did not
     fix pair `6,8`. Diagnostic `NCCL_SHM_DISABLE=1` did fix all failing
     cases, including full `1..8`, but graph-out `1x5120` was slow:
     pairs `5/6/7-8` were `0.167541/0.179511/0.181177` ms, TP4 `5,6,7,8`
     was `0.209532` ms, and full `1..8` was `0.326495` ms. Classification:
     `.50` has a topology/RCCL SHM/P2P communicator issue, not a proven GPU8
     hardware failure. Keep `.50` as diagnostic-only for TP8 unless the
     transport issue is fixed without the slow NET fallback; dense promotions
     remain anchored on `.10/.20/.30`.
261. Current-winner RowParallel allreduces are not legally groupable at the
     post-IR graph point, even with a non-adjacent scan. The new diagnostic
     overlay
     `overlays/postir_mutable_fusion_nonadjacent_census_20260611/gfx906_post_ir_ar_rms_fusion.py`
     and parser update
     `parse_postir_collective_census_20260608.py` examined all later
     collective pairs up to a `128` collective gap. `.10` completed normally in
     `runs/qwen36_27b_decode_tiers_currentwinner_nonadjacent_grouping_census_h10_host10_20260611_124855`;
     `.20`, the dense reference lane, reproduced the census in
     `runs/qwen36_27b_decode_tiers_currentwinner_nonadjacent_grouping_census_h20_host20_20260611_125617`
     before a post-warmup API-readiness hang, which was captured under
     `live_status/` and cleaned up by interrupting only the instance-owned
     runner. Both lanes reported eight ranks with `collective_ops=129`,
     adjacent `groupable_pairs=0`, and non-adjacent
     `pairs_examined=8256`, `nonadjacent_pairs_examined=8128`,
     `pair_groupable=0`, `nonadjacent_groupable=0`. Every examined pair was
     blocked by dependency and intervening use. Do not integrate grouped
     allreduce in the current dense graph. Keep the grouped microbench result
     as valid only for independent reductions; the dense gate path now returns
     to exact-shape RCCL/LL work or structural collective-count reduction.
262. Gfx906 Tree/LL chain-fan specialization is not a dense single-request
     decode promotion. NCCL debug output shows the primary TP8 Tree/LL
     topology is a one-child chain, so
     `rccl_721_gfx906_tree_ll_specialized_kernel_include_ar_chainfan1_20260611.patch`
     bound the guarded chain case to `FanSymmetric<1>` and
     `FanAsymmetric<1,1>`. The `.20` overlay
     `/usr/share/ollama/rccl_overlay_721_gfx906_tree_ll_chainfan1_20260611_h20_v2`
     built cleanly and was exact, but direct graph-out results were mixed:
     `1/2/4x5120` regressed slightly to `0.044833/0.050886/0.062583`, while
     `8/16/20/21/32x5120` improved to
     `0.089396/0.144295/0.172659/0.180428/0.256976`. This is useful
     wider-row source evidence, not a c1 decode gate path. The stricter static
     chain-fan patch built but the direct replay produced no result rows after
     several minutes with all ranks busy and required container teardown
     (`exit_code=137`), so static chain-fan is unsafe. Do not run serving tiers
     for either branch; continue with structural collective-count/coalescing or
     fused endpoint work.
263. Gfx906 Tree/LL half2 sum specialization is source-valid but not a dense
     decode promotion. Patch
     `rccl_721_gfx906_tree_ll_specialized_kernel_include_ar_half2_sum_20260611.patch`
     added a guarded gfx906 `FuncSum<half>` half2 reduce specialization with
     `__hadd2` on top of the promoted Tree/LL f16 include/force-inline path.
     It built cleanly on `.10/.20/.30` and all direct graph-out rows were
     exact. `.20` improved most rows only by noise-scale amounts
     (`1x5120 0.044544` vs promoted `.20` `0.044617`; `32x5120 0.257590`
     vs `0.258963`), while `.10` was mixed and `.30` repeatedly regressed the
     important wider decode row (`8x5120 0.103065`, repeat `0.099530`, vs
     promoted `.30` `0.089690`). Do not run serving tiers for this branch.
     Keep it as evidence that the half arithmetic path is reachable, but the
     primary path must change collective count, row distribution, or fuse work
     below the Python model-forward boundary.
264. Current upstream vLLM ROCm AITER allreduce/RMS fusion is directionally
     relevant but not a runnable gfx906 TP8 gate config. A sparse clone of
     `vllm-project/vllm@b8142294b7e757f3a39729c4f400bafaed534681` shows
     `RocmAiterAllReduceFusionPass` in `pass_manager.py`,
     `rocm_aiter_fused_allreduce_rmsnorm` registration in `_aiter_ops.py`, and
     AITER allreduce/RMS patterns in `allreduce_rms_fusion.py`; upstream issue
     `#43224` also names this ROCm path. The local ROCm 7.2 image on `.20`
     reports `torch 2.11.0+rocm7.2`, but `aiter` is not installed and the
     installed vLLM lacks the current `rocm_aiter_ops.get_aiter_allreduce`
     accessor. More importantly, upstream still requires vLLM/AITER custom
     allreduce initialization, which maps to the already-rejected gfx906 PCIe
     HIP IPC substrate. The prior AITER source probe built enough to reach
     `module_custom_all_reduce`, but TP2 `(1,5120)` failed at
     `hipIpcOpenMemHandle(...) -> invalid device pointer`. Do not burn a
     serving lane on flipping `fuse_allreduce_rms` or backporting the pass
     alone. AITER is only worth revisiting after a tiny TP2/TP8 fused-ar-rms
     probe proves the peer-memory/custom-allreduce substrate works on gfx906.
265. `.50` cannot be made a normal TP8 dense lane from userspace software in
     its current hardware/firmware/topology state. The run
     `runs/host50_make_usable_20260611_132446` reproduced GPU8's failure below
     RCCL at the HIP peer/IPC layer. GPU8 remains compute-usable and static
     checks are normal (`34342961152` VRAM bytes, DID `0x66a1`, PCIe
     `16.0 GT/s x16`, replay count `0`), but it is the only MI50 with a `32G`
     BAR0/downstream window while the others are `16G`. ROCm 7.2 and ROCm
     6.3.4 both show the same GPU8 pair signature: non-GPU8 -> GPU8 reports
     `hipDeviceCanAccessPeer=1`, GPU8 -> non-GPU8 reports `0`, peer-enabled
     p2p segfaults in both copy directions, non-GPU8 export / GPU8 import
     fails at `hipIpcOpenMemHandle` with `hipErrorInvalidDevicePointer`, and
     GPU8 export / non-GPU8 import succeeds. HSA env variations did not change
     this. Classify the remaining blocker as hardware/firmware/PCIe
     topology/BAR allocation until a maintenance cycle proves otherwise. `.50`
     is usable only for TP2/TP4 diagnostics that avoid GPU8, single-GPU GPU8
     compute checks, or slow health diagnostics; dense TP8 promotions remain on
     `.10/.20/.30`.
266. `ngram_gpu` speculative decoding is rejected for dense single-request
     decode. The local `Qwen3.6-27B` checkpoint has no MTP/draft-head metadata,
     so vLLM's no-extra-model `ngram_gpu` path was the only plausible
     speculative schedule lane. It starts under the promoted async/O3 ROCm 7.2
     dense winner envelope, but adds drafter graph capture and collapses
     throughput: `.10` k=2 strict c1/c1_2000 was `17.448/15.363` TPS, `.20`
     k=4 was `17.796/14.164`, and `.30` k=8 was `17.797/13.002`, versus the
     current `.20` winner `57.456/58.245/55.321`. Do not spend more primary
     lane time on ngram speculation for this dense gate; return to RCCL
     Tree/LL source work, fused endpoint work below the Python model-forward
     boundary, or a real semantic collective-count/schedule change.
267. Rebooting `.50` did not remediate GPU8. In
     `runs/host50_reboot_retest_20260611_140745`, `.50` was rebooted and
     retested from a fresh run directory. Post-reboot topology was unchanged:
     GPU8 remained BDF `0000:1f:00.0`, VRAM `34342961152`, replay count `0`,
     and the only MI50 with a `32G` BAR0/downstream prefetchable window while
     the other MI50s remained `16G`. All allowed GPUs `1..8` passed isolated
     single-GPU ROCm 7.2 PyTorch compute with 4 GiB FP16 allocation and repeated
     FP16 matmul, including GPU8. HIP peer/IPC still failed for every ROCm 7.2
     GPU8 pair retested (`1,8`, `4,8`, `5,8`, `6,8`, `7,8`) and reproduced on
     ROCm 6.3.4 for `6,8` and `1,8`: non-GPU8 -> GPU8 reports
     `hipDeviceCanAccessPeer=1`, GPU8 -> non-GPU8 reports `0`, peer-enabled p2p
     segfaults, non-GPU8 export / GPU8 import fails at
     `hipIpcOpenMemHandle`, and GPU8 export / non-GPU8 import succeeds. The
     remaining `.50` TP8 blocker is therefore not cleared by reboot; keep it
     classified as hardware/firmware/PCIe topology/BAR allocation until a
     physical maintenance cycle changes card/slot/riser/BIOS state.
268. Post-IR split-endpoint RMS is correct but not a dense serving promotion
     in the current Python/ctypes form. The overlay
     `overlays/postir_split_endpoint_rms_20260611/gfx906_post_ir_ar_rms_fusion.py`
     successfully replaced one RowParallel allreduce/RMS boundary under the
     promoted ROCm 7.2 TP8 dense envelope after moving split scratch allocation
     out of the FX graph into a process-local per-device cache. `.20` run
     `runs/qwen36_27b_decode_tiers_postir_split_endpoint_cached_scratch_h20_host20_20260611_143553`
     compiled, captured CUDA graphs, served an uncapped strict thinking smoke,
     and produced `4347` completion tokens with valid parser-split answer hash.
     Backend decode was `57.029` TPS, below the current global dense winner
     smoke of `57.456` TPS. Keep this as a source milestone: endpoint fusion
     belongs below the Python model-forward boundary, but the next viable
     version should be a C++/HIP/vLLM-registered op with capture-safe scratch
     or a real collective-count/schedule reduction rather than another
     Python-side wrapper.
269. Scaling the Python/ctypes post-IR split endpoint confirms overhead, not
     correctness, is the blocker. Smoke-only strict thinking runs replaced
     `4/128`, `32/128`, and `128/128` RowParallel boundaries on `.10`, `.20`,
     and `.30` respectively. All compiled, captured, served valid parser-split
     responses, and cleaned up, but backend decode fell to `53.818`, `54.944`,
     and `50.549` TPS. Do not keep sweeping match limits in this
     implementation. Treat the 128-boundary success as a source milestone
     proving the graph rewrite is structurally legal, and move any endpoint
     continuation into a low-overhead registered C++/HIP/vLLM op or RCCL-side
     fused endpoint.
270. Native torch-op split-endpoint RMS is also not enough to promote. The
     runner built
     `/usr/share/ollama/kernel_labs/gfx906_split_endpoint_rms_torchop_20260611/build/gfx906_split_endpoint_rms_torchop_20260611.so`
     on `.10`, `.20`, and `.30`, registered
     `gfx906_ext::split_endpoint_rms`, and inserted it through the post-IR pass
     in `split_endpoint_torchop` mode. `.10` one-boundary smoke was valid but
     reached only `53.530` backend TPS. `.20` one-boundary smoke was also valid
     and reached `56.651` backend TPS, below both the Python/ctypes one-boundary
     `.20` result (`57.029`) and the current `.20` global winner smoke
     (`57.456`). Removing the Python/ctypes per-call counter zero helped the
     full-replacement path only to `54.159`. Close endpoint-wrapper tuning as a
     primary path; further progress likely requires reducing or rescheduling
     collective boundaries, changing RCCL launch behavior, or fusing below the
     Python/model-forward layer rather than wrapping the same boundary more
     cheaply.
271. The RCCL chain-fan plus half2 composition is source-valid but not a dense
     serving candidate. Patch
     `rccl_721_gfx906_tree_ll_chainfan1_half2_sum_20260611.patch` combined the
     one-child Tree/LL chain-fan specialization with the guarded gfx906
     `half2` sum specialization and built cleanly on `.20/.30` with RCCL
     `22707`. Direct graph replay stayed exact. `.20` produced
     `1/2/4/8/16/20/21/32x5120 =
     0.044760/0.050968/0.062748/0.089361/0.143931/0.172819/0.180600/0.256010`;
     `.30` produced
     `0.044794/0.058745/0.062532/0.089542/0.144103/0.173005/0.181202/0.256601`.
     This improves selected wider rows but regresses important hot decode rows,
     including a large `.30` `2x5120` regression. Do not run serving tiers for
     this branch. Keep it as evidence that narrow primitive tweaks are running
     out of useful single-request decode headroom; the gate path needs
     collective-count reduction, launch/schedule reduction, or a real fused
     producer/collective/consumer boundary below the Python model-forward layer.
272. `.50` is usable for current P2P-disabled Tree/LL dense lanes only when
     the unsupported 4 GiB gfx803 display GPU is excluded at the DRI-device
     level. Broad `/dev/dri` or `--privileged` can leak host GPU0 into ROCm
     initialization and make PyTorch fail with `No CUDA GPUs are available`.
     After the GPU8 VBIOS flash/reboot, ROCm-SMI still shows the display as
     `card0`, but DRM now maps the display to `/dev/dri/card1` plus
     `/dev/dri/renderD128`. The MI50/gfx906 devices are `/dev/dri/card0`,
     `/dev/dri/card2-8`, and `/dev/dri/renderD129-136`. The validated
     override exposes those eight MI50s as container devices `0-7`, with host
     render gid `992` and `DOCKER_PRIVILEGE_ARGS=''`. Direct RCCL TP8 Tree/LL
     P2P-off graph replay then passes on `.50` after the VBIOS flash, including
     the large `431x5120` row-family shape. Run
     `runs/rccl_direct_allreduce_graph_bench_host50_postflash_20260611_200230`
     produced exact zero-error graph results: `1x5120` graph-out
     `0.067968` ms/call and `431x5120` graph-out `5.843020` ms/call
     (`5.855701` in-place). Current-winner vLLM smoke also passes with strict
     parser-split validity, `3608` completion tokens, and `48.520` backend
     decode TPS. Keep `.50` excluded from peer/IPC custom-collective claims
     unless those paths are retested directly.
273. NPKIT attribution on the narrow ROCm 7.2 Tree/LL specialized RCCL build
     shifts the source target away from wait/spin tuning. In
     `runs/rccl_npkit_specialized_narrow_direct_h20_20260611`,
     `PRIM_LL_WAIT_SEND` is tiny (`p95` about `3.75` trace units), while
     Tree split broadcast, recv/copy/send, recv-reduce, and LL data-process
     dominate. Future RCCL-side source work should target Tree/LL data
     movement and broadcast/copy behavior, or move up a level to
     collective-count, launch/schedule, or producer/collective/consumer fusion.
274. `.50` is operational again for P2P-disabled Tree/LL dense lanes, but it is
     not an equivalent promotion host. The fixed container contract excludes
     the 4 GiB gfx803 display adapter at the DRI level and exposes the eight
     MI50/gfx906 devices as container devices `0-7`; with that contract,
     current-winner vLLM smoke and direct TP8 RCCL graph replay pass. A
     bounded graph-only row family replay,
     `runs/rccl_direct_allreduce_graph_bench_host50_20260611_171347`, was
     exact but materially slower than `.10/.20/.30`: graph-out `1x5120`
     `0.067662 ms` versus the promoted three-host mean `0.044758`, and
     `8x5120` `0.165264 ms` versus `0.089850`. Use `.50` for
     portability/topology validation and no-GPU0 DRI regression checks, not as
     same-platform promotion evidence for the dense `65 TPS` gate.
275. MLP activation-gather plus replicated full down-projection is rejected as
     a memory-bandwidth alternative to the MLP/down RowParallel output
     allreduce. The new runner
     `run_replicated_rowparallel_gather_microbench_20260611.sh` tested
     all-gathering `1x2176` activations and running a replicated
     `5120x17408` projection on `.20/.30`. The LLMM1 full-projection path
     fails with HIP `invalid configuration argument`; the `torch_linear`
     fallback runs but is not an exact replacement in the MEP
     (`max_abs_error=260.25`) and is over an order of magnitude slower:
     `.20` current boundary `0.078856 ms` vs all-gather+replicated
     `1.315014 ms`, `.30` `0.081858` vs `1.324168`. Keep the gate path on the
     narrow `1x5120` MLP/down output collective rather than moving work to a
     full replicated projection.
276. The narrow TP8-to-TP4 MLP down-projection bridge is rejected. New MEP
     `mlp_tp8_to_tp4_downproj_bridge_bench_20260611.py` kept TP8 gate/up
     shards, exchanged adjacent `1x2176` activation shards through an exact
     zero-padded pair allreduce, computed paired `LLMM1(5120x4352)`, then ran
     two independent TP4 `1x5120` allreduces over even/odd ranks. It was
     exact after replacing the invalid subgroup `all_gather` attempt, but
     slower than the current TP8 down boundary: `.20` baseline `0.074606 ms`
     vs bridge `0.107827`, `.30` baseline `0.077538` vs bridge `0.103723`.
     The earlier TP4 down-proj lower-bound only applies when the `1x4352`
     activation already exists; constructing it inside the TP8 model erases
     the benefit. Do not implement this model overlay.
277. Down-projection `LLMM1 + residual` pre-folding is still rejected after
     retesting on the final specialized RCCL dense winner. The `.20` smoke
     run
     `qwen36_27b_decode_tiers_rocm72_dense_specialized_downllmm1resid_recheck_h20_host20_20260611_174740`
     used the current ROCm 7.2 TP8 mutable RowParallel, native-runtime
     interleaved SwiGLU, specialized Tree/LL overlay, min/max channels `4`,
     and `NCCL_NTHREADS=128`, plus the strict
     `gfx906_rowpar_llmm1_residual_ext_20260609.so` path. It passed the
     thinking-safe smoke gate but reached only `55.823` backend decode TPS,
     below the `.20` current winner smoke of `57.456`. Treat local
     residual-add cleanup as closed for dense single-request decode unless it
     is part of a larger producer/collective/consumer boundary change.
278. Dense and MoE do not share the same NCCL channel/thread optimum. The
     opportunistic MoE side-lane translated the dense ROCm 7.2 channel envelope
     (`NCCL_MAX_NCHANNELS=4`, `NCCL_NTHREADS=128`) to Qwen3.6-35B-A3B TP4
     with ROCm7.2 attention-backport patches. `.10` reached
     `98.509/100.225/94.968` TPS for strict c1/c1_2000/c1_10000, a small
     `+0.41%` host-local 10K gain over its old baseline but below the global
     MoE references. `.30` reached `98.725/99.400/94.233`, a `-1.12%` 10K
     regression versus its old baseline. Do not promote dense maxch4 plus
     nthreads128 as the MoE default; MoE side-lanes need sustained c1_10000
     wins, not only short-tier gains.
279. The `.20` current-winner dense rocprofv3 run
     `qwen36_27b_c1_rocprof_currentwinner_rocm72_h20_side_20260611_180110`
     is invalid profiling evidence. It served the capped c1_768 request but
     exported no kernel trace; rocprofv3 caught SIGINT, waited for child
     processes, then required docker-stop SIGTERM and produced an empty profile
     artifact. The depressed profiled decode TPS (`18.830`) is
     instrumentation-only and must not be used for performance decisions.
     Rerun current-winner profiling only with a corrected exporter/window or a
     narrower profiling method that demonstrably produces kernel CSVs.
280. The corrected `.20` current-winner dense rocprofv2 run is valid and keeps
     the serious source path on collectives. Run
     `qwen36_27b_c1_rocprofv2_currentwinner_h20_20260611_185225` used the
     current ROCm 7.2 TP8 dense winner envelope and emitted `806382` filtered
     kernel rows for capped `c1_768`. Profiled throughput was only
     `25.838` backend TPS, so it is attribution-only. Filtered kernel time was
     NCCL/RCCL `12.6186s / 57.689% / 99640` rows, `LLGemm1_kernel`
     `3.8584s / 17.640% / 184680`, and
     `swiglu_gemv_llmm1_interleaved_pair_kernel`
     `2.8385s / 12.977% / 48991`. This closes wrapper/local-residual cleanup
     as the next serious dense gate path and points back to collective-count
     reduction, legal coalescing, or deeper Tree/LL data movement work.
281. The MoE one-channel ROCm7.2 image/backport side-lane produced a real
     decode milestone on `.20`, but remains a side winner rather than a public
     deploy default until reproducibility/portability is checked. `.10` run
     `qwen36_moe_decode_tiers_moe_rocm72image_attn_ch1_side_h10_host10_20260611_185604`
     reached strict c1 `100.048`, c1_2000 `100.405`, and c1_10000
     `95.182` backend TPS, a `+0.64%` host-local 10K gain over the old `.10`
     baseline (`94.578`) and slightly better than the `.10` dense-channel
     side-lane (`94.968`). `.20` run
     `qwen36_moe_decode_tiers_moe_rocm72image_attn_ch1_side_h20_host20_20260611_190859`
     reached strict c1 `104.135`, c1_2000 `106.037`, and c1_10000
     `100.304` backend TPS after a valid uncapped thinking smoke. That is
     `+3.54%` over the earlier `.20` ROCm7.2 reference (`96.878`) and
     `+4.85%` over the corrected Tree/LL publication winner (`95.661`).
     A same-host repeat,
     `qwen36_moe_decode_tiers_moe_rocm72image_attn_ch1_repeat_h20_host20_20260611_192801`,
     reached strict c1 `104.279`, c1_2000 `105.269`, and c1_10000
     `99.544` backend TPS. That repeat is still `+2.75%` over the earlier
     `.20` ROCm7.2 reference and `+4.06%` over the corrected Tree/LL
     publication winner, but `-0.76%` below the first `.20` side run. Treat the
     side winner as a real `99.544-100.304` c1_10000 milestone band rather than
     exact-run-stable until portability is checked.
     `.30` run
     `qwen36_moe_decode_tiers_moe_rocm72image_attn_ch1_side_h30_host30_20260611_185604`
     reached only `98.405/99.487/94.327`, a `-1.02%` 10K regression versus
     its host baseline (`95.302`). Promote the `.20` result as the current MoE
     single-request decode side winner and milestone, while keeping the stable
     deployable publication image unchanged until this ROCm7.2 image/backport
     shape is reproduced. The MoE communication envelope should stay
     one-channel Tree/LL; dense maxch4/nthreads128 remains rejected for MoE
     decode.
282. `NCCL_NTHREADS=128` does not improve the MoE one-channel decode envelope.
     The `.10` isolation run
     `qwen36_moe_decode_tiers_moe_rocm72image_attn_ch1_nthreads128_h10_host10_20260611_192801`
     used the same ROCm7.2 image/backport one-channel Tree/LL path but forced
     `NCCL_NTHREADS=128`. It reached strict c1 `99.666`, c1_2000 `100.174`,
     and c1_10000 `94.954` backend TPS, which is `-0.24%` versus the `.10`
     one-channel unset-thread result (`95.182`) and only `+0.40%` over the old
     `.10` baseline (`94.578`). Keep `NCCL_NTHREADS` unset for MoE ch1; do not
     import the dense thread-count setting.
283. The MoE decode-tier runner cleanup path must honor the selected Docker
     source mode. `run_qwen36_moe_decode_tiers_20260605.sh` now uses
     `REMOTE_DOCKER_ENV_SOURCE` in cleanup instead of unconditionally sourcing
     `.deploy.docker-host.env`. This prevents `REMOTE_DOCKER_ENV_FILE=host`
     runs from leaving containers alive when a package-local deploy env points
     at a stale isolated Docker socket.
284. `.50` was rechecked after the GPU8 VBIOS flash and the device is
     compute-usable under the corrected explicit DRI contract. Live inventory
     on `2026-06-11T19:56:17-04:00` showed ROCm-SMI `card1-card8` as DID
     `0x66a1` with `34342961152` bytes VRAM and VBIOS
     `113-D1631711-100`; product strings for two cards still say `16GB`, so
     keep using VRAM/DID rather than product names. Inside the ROCm 7.2 image,
     exposing `/dev/dri/card0`, `/dev/dri/card2-8`, and
     `/dev/dri/renderD129-136` produced `torch.cuda.device_count()==8`, all
     visible devices reported `34342961152` bytes, and an FP16 matmul on
     container `cuda:7` completed. Local `.50` runners have been updated from
     the stale `card1-8` mapping to this post-flash DRI mapping.
285. Shortening MoE `max_model_len` to `16384` does not improve the current
     one-channel ROCm7.2 MoE side winner. The `.10` run
     `qwen36_moe_decode_tiers_moe_rocm72image_attn_ch1_len16k_h10_host10_20260611_194251`
     used the ROCm7.2 image/backport, `NCCL_MAX_NCHANNELS=1`,
     `NCCL_NTHREADS` unset, and `MAX_MODEL_LEN=16384`. It reached strict c1
     `99.265`, c1_2000 `100.206`, and c1_10000 `94.976` backend TPS. That is
     a slight regression versus the same-host one-channel unset-thread 128K
     result (`95.182` at c1_10000), so keep the MoE side winner on its normal
     long-context setting unless a separate memory-pressure target requires a
     shorter context.
286. The refreshed `.20` grouped-allreduce row-family replay preserves the
     lower-bound story but still does not create a legal serving promotion by
     itself. Run
     `pynccl_grouped_allreduce_bench_host20_20260611_rowfamily_h20_hostdocker`
     showed group-size-32 eager speedups of `67.299%` for `1x5120`,
     `58.401%` for `2x5120`, `44.685%` for `4x5120`, `30.047%` for
     `8x5120`, `15.062%` for `20x5120`, `14.421%` for `21x5120`, and
     `0.930%` for `431x5120`. This reinforces grouped collectives as a
     worthwhile primitive when independent adjacent reductions exist, but the
     dense 27B lowered graph still has immediate consumer dependencies between
     RowParallel reductions. Promotion still requires source work that changes
     the legal boundary schedule, count, or producer/collective/consumer shape.
287. Host `.30` root filled because dense 27B runtime/cache directories were
     still defaulting to `[redacted-local-path]`. On
     `2026-06-11`, `/dev/sda2` was `100%` full (`418G` used, `0` available).
     Safe user-owned cleanup of pip/Triton caches and stale venv zip archives
     recovered `35G`, then the stale root-owned Docker runtime directories
     under `[redacted-local-path]` were removed after
     explicit sudo approval. Final state: `/dev/sda2` `231G` used, `186G`
     available, `56%` full; `[redacted-local-path]` is `12K`.
     The dense 27B decode/census runners now default runtime caches to
     `/usr/share/ollama/qwen36_27b_decode_tiers`, leaving the heavy
     TorchInductor/vLLM/Triton artifacts on the NVMe mount unless explicitly
     overridden.
288. The current-winner LLGemm profile shapes are mapped well enough to keep
     source effort focused. The valid `.20` rocprof profile
     `qwen36_27b_c1_rocprofv2_currentwinner_h20_20260611_185225` showed five
     `LLGemm1_kernel` groups: `M=5120,K=2176` is MLP down RowParallel local
     GEMV; `M=5120,K=768` is full-attention `o_proj`; `M=2048,K=5120` is
     Qwen3.5 linear-attention `in_proj_qkvz`; `M=12,K=5120` is
     linear-attention `in_proj_ba`; and `M=31040,K=5120` appears to be a
     sparse warmup/legacy full gate-up shape, not the active recurrent serving
     path. The material runtime buckets remain RCCL/NCCL first, then
     `LLGemm1`, then interleaved SwiGLU. Do not move the primary path away
     from collective count/coalescing unless a local GEMV change proves a
     whole-tier serving gain.
289. The old K5120 selective LLMM1 extension is rejected on the ROCm 7.2 dense
     winner path until rebuilt against the ROCm 7 ABI. Both `.20` and `.30`
     selective serving attempts failed before readiness with
     `ImportError: libamdhip64.so.6: cannot open shared object file`; the
     extension at
     `/usr/share/ollama/kernel_labs/llmm1_k5120_specialized_20260607_200904`
     was built against the ROCm 6 HIP runtime while the current winner image is
     ROCm 7.2. This is an ABI rejection, not a performance rejection of a
     rebuilt ROCm 7.2 extension. The baseline profile-mapped shape-stack
     microbench did not show enough isolated upside to make this a primary
     gate path; treat any ROCm 7.2 rebuild as a minor milestone candidate.
290. Dense Tree/LL `NCCL_NTHREADS` remains a guardrail, not a new promotion
     lever. The `.10/.30` exact-row graph-out sweep
     `rccl_direct_allreduce_graph_bench_host{10,30}_20260611_nthreads_sweep_*`
     tested `unset`, `64`, `96`, `128`, `160`, `192`, and `256` on the
     promoted specialized Tree/LL overlay for
     `1/2/4/8/16/20/21x5120`. Host `.10` showed a small best-case signal for
     `96` (`1-8x5120` mean `0.061910 ms` versus `0.062062 ms` at `128`), but
     the all-row mean was essentially tied (`0.106736` versus `0.106700`) and
     host `.30` produced shape-specific regressions/outliers for alternative
     settings (`96` regressed `4x5120`, `160` regressed `2x5120`, `256`
     regressed `1x5120`). No thread count produced a consistent multi-host,
     multi-shape improvement large enough to justify full serving tiers. Keep
     dense decode at `NCCL_NTHREADS=128`.
291. Full-boundary `LLMM1 -> AllReduce -> residual/RMS` split-endpoint replay
     is a source milestone, but still not a standalone dense gate clear. New
     harness `llmm1_allreduce_rms_boundary_bench_20260612.py` includes the
     actual serving producer shapes (`attn 5120x768`, `mlp_down 5120x2176`)
     before the PyNccl allreduce and compares one-block fused RMS against the
     split-endpoint RMS bridge under HIP graph replay. Split3 reproduced the
     expected `~5.1-6.3 us` win on `.10/.20` and for `.30` MLP-down, but `.30`
     attention split3 was repeatably slow (`~0.156-0.165 ms`). Switching the
     external endpoint to split4 fixed that host/shape sensitivity: `.20`
     attention split4 `0.070329 -> 0.064811 ms`, `.30` attention split4
     `0.069953 -> 0.064302 ms`, `.20` MLP split4 `0.085917 -> 0.080295 ms`,
     and `.30` MLP split4 `0.086081 -> 0.080588 ms`, all exact with residual
     error `0` and norm error `0.000976562`. Use split4, not hard-coded
     split3, for the next native endpoint integration. The maximum full-model
     upside is roughly `5.5 us * 128` boundaries, about `0.7 ms/token`, so this
     can be a meaningful component milestone or stackable source patch but it
     cannot by itself bridge the remaining dense `65 TPS` gap.
292. Full `128/128` native torch-op split-endpoint serving is rejected despite
     the split4 lower-bound. The `.30` run
     `qwen36_27b_decode_tiers_postir_split_endpoint_torchop_limit128_split4_h30_rerun_host30_20260611_212301`
     used the post-IR RMS image, selected `128/128` boundaries in
     `split_endpoint_torchop` mode with `split_channels=4`, compiled/captured
     successfully after a long but valid cold start, and passed the strict
     thinking gate. The smoke produced `3139` completion tokens at `52.197`
     backend decode TPS, below the `.30` promoted-specialized baseline band
     and far below the `.20` global winner. This proves the external split4
     endpoint component win is being lost in current post-IR/native-op serving
     integration. Close endpoint-wrapper optimization as the primary path;
     continue toward true collective count/schedule changes or a fused
     producer/collective/consumer implementation below the current graph-op
     overhead.
293. Single-channel RCCL post-RMS fusion is rejected as a dense decode path.
     The split-2/3/4 post-RMS branch had already shown that row-level channel
     rendezvous is too expensive, so the remaining useful question was whether
     `NCCL_MIN_NCHANNELS=1` / `NCCL_MAX_NCHANNELS=1` could make the Tree/LL
     kernel own the full `1x5120` row and avoid cross-channel synchronization.
     Three-host runs
     `rccl_ar_fused_rms_boundary_host{10,20,30}_20260612_postrms_split1_h{10,20,30}`
     were correctness-clean, but the one-channel allreduce floor was too slow
     and in-RCCL post-RMS was slower again. At `1x5120`, `.10/.20/.30`
     measured allreduce-only `0.053698/0.053531/0.053641 ms`,
     one-channel `AllReduce -> fused RMS` `0.070422/0.070304/0.070449 ms`,
     and `AllReduceWithPostRms` `0.077414/0.077493/0.089528 ms`. Wider rows
     were worse: `20x5120` post-RMS was about `0.815-0.817 ms` versus
     one-channel allreduce+fused RMS at about `0.379-0.380 ms`. Do not revisit
     in-RCCL post-RMS as a serving path. The useful fused-consumer direction
     must avoid row/channel rendezvous and also preserve the promoted
     four-channel Tree/LL allreduce floor.
294. Combining two weak RCCL micro-signals did not create a dense decode base.
     The `.30` `singlework_leaf_updown` overlay
     `/usr/share/ollama/rccl_overlay_721_gfx906_tree_ll_singlework_leaf_updown_20260612_h30`
     built cleanly and was exact in direct graph replay, but it only improved
     `1x5120` from the promoted `.30` comparator `0.044954` to `0.044667`
     ms/call. It regressed every wider row: `2x5120 0.051669` vs `0.051044`,
     `4x5120 0.066416` vs `0.062838`, `8x5120 0.098849` vs `0.089690`,
     `16x5120 0.164614` vs `0.144252`, `20x5120 0.198372` vs `0.172791`,
     and `32x5120 0.301709` vs `0.258422`. Do not promote this branch or run
     serving tiers. The lesson is that wrapper shaving plus leaf sequential
     phase splitting is below the needed lever size; next source work needs a
     real Tree/LL data-movement change, collective count/launch-count
     reduction, or producer/collective/consumer fusion below graph-op overhead.
295. The aligned-half `DataLoader` fast path is rejected as a dense decode
     lever. Patch
     `rccl_721_gfx906_tree_ll_aligned_half_loader_20260612.patch` added a
     guarded gfx906 fp16 `eltN==EltPerLine` 8-byte-aligned `u64` load path to
     skip generic half misalignment and funnel-shift bookkeeping. It built and
     exported cleanly on `.30`, and direct graph replay was exact, but it did
     not improve the row family: candidate graph-out
     `1/2/4/8/16/20/32x5120 = 0.045440/0.051301/0.062985/0.089836/0.144949/0.174016/0.259826`
     versus promoted `.30` comparators
     `0.045226/0.051023/0.063054/0.089626/0.144760/0.172791/0.258422`.
     Do not run serving tiers. This closes the hypothesis that the remaining
     `PRIM_LL_DATA_PROCESS` cost is mostly local half-load alignment
     bookkeeping; the next useful path must change dataflow or collective
     scheduling more substantially.
296. ReduceScatter plus AllGather is rejected as a dense single-request decode
     gate path. New harness
     `rccl_rs_ag_equiv_graph_bench_20260612.cpp` showed that the promoted
     Tree/LL overlay cannot be used for non-allreduce collectives because its
     generic kernels are hardwired to
     `RunWorkBatch<ncclFuncAllReduce, half, Sum, Tree, LL>`; under that overlay
     `ReduceScatter` produced the correct `36`, but `AllGather` reduced again
     and produced `288`. Stock full RCCL made `RS+AG` exact, but it was slower
     than stock allreduce on decode-hot rows and still slower than the
     promoted allreduce floor: `.10/.20/.30` stock `1x5120 RS+AG` measured
     `0.123745/0.123202/0.406612 ms`, while promoted allreduce is about
     `0.045 ms`; `20x5120 RS+AG` was only neutral to slightly positive versus
     stock allreduce (`0.185653/0.180035/0.182599 ms`) and still not better
     than the promoted allreduce row family. Static SP can remain a controlled
     prefill/no-regression side lane, but it is not a credible primary path for
     clearing the dense decode gate unless a future safe multi-collective
     overlay radically changes ReduceScatter/AllGather performance.
297. `.30` root pressure was checked after the source-lane churn and was not
     the active blocker. On 2026-06-11, `/` on `.30` had `295 GiB` free
     (`30%` used). Low-risk Docker cleanup removed stopped containers only;
     active `open-webui` and `openedai-speech` containers were left untouched,
     build cache was already `0B`, and `/` remained at `295 GiB` free. Do not
     spend source-lane time on `.30` root cleanup unless a fresh `df -h /`
     shows real pressure.
298. `.50` remains a topology/portability lane after the GPU8 VBIOS fix, not
     a dense `65 TPS` promotion host. The corrected no-privileged container
     contract excludes the display GPU at the DRI level and exposes the eight
     MI50/gfx906 devices as container-local ordinals `0-7`; using host-style
     ordinals `1-8` under that DRI filter caused `invalid device ordinal` on
     ranks 6/7 and is invalid. Corrected direct TP8 graph replay with the
     current promoted allreduce overlay was exact, but much slower than
     `.10/.20/.30`: `.50` graph-out
     `1/2/4/8/16/20/21/32x5120 =
     0.063429/0.082256/0.109139/0.164705/0.268789/0.323902/0.335420/0.479297`
     ms/call. The normal promotion lanes are roughly
     `0.045/0.051/0.063/0.090/0.145/0.173/0.181/0.258` on the same row
     family. Keep `.50` for topology regression checks and no-GPU0 deployment
     validation, not for same-platform dense decode promotion evidence.
299. Direct C++ `LLMM1 -> RCCL AllReduce -> RMS` lower-bound testing confirms
     split4 endpoint RMS as a component win, but closes it as a primary dense
     gate path. New harness
     `rccl_llmm1_ar_rms_boundary_bench_20260612.cpp` removes Python/PyNccl and
     post-IR wrapper overhead from the boundary and links directly against the
     promoted RCCL overlay. On `.20`, graph replay measured
     `attn_5120x768` `LLMM1+AR 0.052658`, `+fused RMS 0.069157`,
     `+split4 RMS 0.063871`; `mlp_down_5120x2176` measured `0.069148`,
     `0.084998`, and `0.079484` ms. `.30` reproduced the attention result in
     a repeat at `0.053097`, `0.069569`, and `0.064254`; its MLP row measured
     `0.069460`, `0.085615`, and `0.080239`. All rows were exact. Split4
     saves about `5.3-5.5 us` versus one-block fused RMS, but still adds about
     `10-11 us` over `LLMM1+AR`. This matches the previous Python lower bound:
     endpoint fusion is a stackable source milestone, not enough to clear the
     dense gate and not worth another serving overlay by itself. The next
     serious path must change the producer/collective schedule or collective
     count, not only the post-allreduce consumer wrapper.
300. The hidden-sharded dense decode route remains closed under the current
     winner envelope. Existing exact MEPs are sufficient and do not need another
     serving lane: the next-projection hidden-shard test made the favorable
     gate/up path slower on `.20` (`0.052825 -> 0.056332 ms`) and made smaller
     projection consumers much worse; the final-envelope
     reduce-scatter/RMSNorm test was also slower at `1x5120` (`0.068277 ms`
     baseline versus `0.139445 ms` for `rs_ag` and `0.189293 ms` for
     `sharded_rms`). At prefill-sized rows those reduce-scatter/RMSNorm paths
     can win, so keep them for prompt-processing/concurrency milestones, but
     do not spend single-request decode promotion time on hidden-sharded serving
     unless the whole layer schedule changes.
301. NPKIT attribution on the promoted specialized Tree/LL path points at real
     Tree/LL reduce/broadcast data movement, not wrapper overhead. The clean
     `.20` `1x5120` graph-out trace shows the hot cost inside
     `ALL_REDUCE_TREE_SPLIT_*` and `PRIM_LL_DATA_PROCESS` phases; wait-send
     bookkeeping is tiny by comparison. This validates why loader tweaks,
     single-work wrappers, leaf-only up/down, and endpoint consumers have all
     stayed below gate-clearing size. The next serious dense path must either
     legally reduce/coalesce RowParallel collectives, change full-rank Tree/LL
     dataflow, or implement a fused producer/collective/consumer below the
     current graph-op boundary.
302. `.30` cleanup status after the source-lane churn: Docker's data root is
     already on NVMe at `/usr/share/ollama/docker-data`, not on `/`. Stale
     unused test image tags were removed while retaining the current ROCm 7.2
     experimental winner image and the ROCm 7.2 base image. User-owned
     Hugging Face Xet cache and stale `/tmp` scratch directories were cleared,
     reducing `/` from `122 GiB` used to `112 GiB` used with `305 GiB` free.
     The two long-running service containers (`open-webui` and
     `openedai-speech`) were left untouched, and all eight GPUs were idle.
303. Forcing the hot `10240` byte Tree/LL row to all four channels is rejected.
     Patch
     `rccl_721_gfx906_tree_ll_specialized_kernel_include_ar_force4_10kb_20260612.patch`
     stacked on the promoted specialized f16 Tree/LL dispatch and changed the
     host planner for TP8 fp16 `AllReduce(count=5120)` to an even
     `1280/1280/1280/1280` split. The TUNING-log smoke confirmed the candidate
     planned `channel{Lo..Hi}={0..3}`, while the current promoted overlay plans
     the same row as `channel{Lo..Hi}={0..2}`. Direct `.30` graph replay was
     exact but slower on the key row family: candidate
     `1/2/4/8/16/20/32x5120 =
     0.051213/0.051172/0.063219/0.090235/0.144936/0.174240/0.258984 ms`
     versus promoted `.30` comparators around
     `0.045226/0.051023/0.063054/0.089626/0.144760/0.172791/0.258422 ms`.
     Do not force four work channels for the `1x5120` row; the planner's
     three-channel split is locally better even though the communicator uses the
     min=max-4 serving envelope.
304. Balanced three-channel partitioning for the same hot row is also rejected.
     Patch
     `rccl_721_gfx906_tree_ll_specialized_kernel_include_ar_force3balanced_10kb_20260612.patch`
     kept the promoted three-channel shape but forced a balanced
     `1704/1704/1712` element split for TP8 fp16
     `AllReduce(count=5120)`. The full `.30` graph replay was exact and showed
     a tiny apparent `1x5120` win (`0.045009 ms`), but the focused high-iteration
     repeat showed run variance rather than signal: current `1x5120` repeated
     at `0.045889 ms`, while balanced-three repeated at `0.046378 ms`. Do not
     promote or run serving tiers. The planner-partition branch is closed unless
     a future profile shows a different payload or topology-specific channel
     imbalance.
305. The guarded RCCL `ncclAllReduceWithLlmm1Precompute` metadata path now
     builds and exports correctly, but the scalar in-collective producer is
     rejected as a performance path. Patch
     `rccl_721_gfx906_tree_ll_llmm1_precompute_20260612.patch` adds the public
     API-table export and device-work metadata plumbing, and the `.20` exact
     `AllReduce TREE LL Sum f16` overlay exported
     `ncclAllReduceWithLlmm1Precompute`. Direct TP8 graph replay was exact, but
     far slower than the existing optimized producer plus RCCL allreduce:
     `attn_5120x768` measured `LLMM1+AR 0.052905 ms` versus
     `precompute+AR 0.547903 ms`; `mlp_down_5120x2176` measured
     `0.069374 ms` versus `1.445859 ms`. This closes the simple "do the GEMV
     inside Tree/LL" branch. Keep the API/metadata plumbing as reusable
     infrastructure, but any future producer/collective fusion must use a real
     tiled producer or a different multi-block collective design; RCCL's
     current small number of channel threads is the wrong execution geometry
     for the dense projection itself.
306. Public RCCL send/recv recursive-doubling is not a viable dense decode
     lower bound on the current stack. The harness now supports isolated method
     switches (`AR_GRAPH_RUN_P2P_ALLGATHER`, `AR_GRAPH_RUN_RING`,
     `AR_GRAPH_RUN_XOR`) so the XOR schedule can be tested without the older
     allgather/ring paths. With stock full RCCL, P2P disabled, Tree/LL,
     `NCCL_MAX_NCHANNELS=4`, and `NCCL_NTHREADS=128`, recursive doubling was
     exact but slower than stock `ncclAllReduce`: for `2/4/8x5120`, stock
     graph allreduce measured `0.053366/0.066118/0.092429 ms` while XOR graph
     measured `0.086210/0.095753/0.132288 ms`; the `1x5120` eager comparison
     was `0.048403 ms` stock allreduce versus `0.115924 ms` XOR. P2P-enabled
     XOR-only failed before measurement with RCCL/HIP `invalid argument` at
     communicator setup, and the promoted allreduce-specialized overlay failed
     immediately on the public send/recv group call. This closes public P2P
     recursive doubling as a gate path. Any recursive-doubling style schedule
     would need to be implemented inside RCCL's allreduce primitive, not by
     composing public send/recv calls from serving.
307. Explicit binary `RCCL_TREES` can materially improve dense TP8
     single-request decode by replacing RCCL's default one-child local Tree/LL
     chain with a serving-valid balanced tree. The validated tree string is
     `(0(1(3)(4))(2(5)(6(7))))|(2(3(5)(6))(4(7)(0(1))))|(4(5(7)(0))(6(1)(2(3))))|(6(7(1)(2))(0(3)(4(5))))`.
     Direct `.20` graph replay moved `1x5120` from `0.044676 ms` to
     `0.030709 ms` and `2x5120` from `0.050897 ms` to `0.038196 ms`, while
     regressing wider rows (`16x5120+`). Clean serving repeated on `.20` at
     `c1_128/c1_2000/c1_10000 = 61.252/62.236/58.780` backend TPS and on
     `.30` at `61.000/61.907/58.226`. This is the current dense decode
     milestone but not promoted for the pre-release dense gate. The `.10` repeat was valid but slower
     (`58.831/57.446/53.301`). Arbitrary 3-child/asymmetric tree strings are
     unsafe for serving: ternary and pair-aware hand trees hung direct replay
     with no rank rows and were killed. Binary pair-aware variants were exact
     but did not improve enough to justify serving tiers.
308. For ROCm 7.2 C/HIP harnesses, physical GPU ordinals and local HIP ordinals
     must be separated when targeting a subset such as host GPUs `4,5,6,7`.
     The reliable mapping is `ROCR_VISIBLE_DEVICES=4,5,6,7` with local
     `HIP_VISIBLE_DEVICES=0,1,2,3`, `CUDA_VISIBLE_DEVICES=0,1,2,3`, and
     `GPU_DEVICE_ORDINAL=0,1,2,3`. A compiled HIP count test returned zero
     devices when physical `4,5,6,7` were used directly as HIP ordinals, while
     the filtered-local mapping returned four `gfx906` devices. The direct
     allreduce and MoE decode runners now support this split mapping.
309. `RCCL_TREES` strings name RCCL GPU device IDs, not local rank ordinals.
     On the filtered `.20` TP4 side lane, `(0(1)(2(3)))` failed in
     `parseGraphLight(...)` because the visible GPUs still mapped to physical
     device IDs `4,5,6,7`; `(4(5)(6(7)))` passed direct allreduce. The serving
     retest with physical-device tree syntax and split ROCR/HIP mapping scored
     MoE `c1_128/c1_2000/c1_10000 =
     104.387/105.149/99.362` backend TPS. That is not a MoE promotion versus
     the existing `100.304` sustained side winner, but it closes the TP4
     tree-syntax failure and records the portable subset-GPU launch rule.
310. Grouped allreduce remains the strongest dense source lower bound after
     balanced topology, but only if implemented below a legality-preserving
     boundary. On `.30`, graph-captured group-size-64 no-tree `1x5120` moved
     from `0.045392` ungrouped to `0.014956 ms/call`; balanced tree moved
     `1x5120` from `0.030999` to `0.012724`. For `2x5120`, grouping was still
     strong and topology-neutral; for `4x5120/8x5120`, balanced grouped was
     worse than no-tree grouped. This means source coalescing should be
     dependency-aware and shape-aware, not a blind global group rule.
311. Reducing balanced-tree dense TP8 to fewer RCCL channels is closed. On
     `.30`, `NCCL_MIN/MAX_NCHANNELS=3` was exact but only tied the hot
     `1x5120` row (`0.030930 ms`) while regressing `2x/4x/8x5120` to
     `0.042750/0.062052/0.108095 ms` and wider rows more severely.
     `NCCL_MIN/MAX_NCHANNELS=2` regressed every measured row. Keep min=max `4`
     for the dense balanced-tree serving path.
312. Balanced-tree dense TP8 should keep `NCCL_NTHREADS=128`. Same-host `.30`
     direct replays of unset, `64`, and `256` were exact but mixed; all three
     regressed the important `8x5120` row to roughly `0.111-0.116 ms`, while
     the `128` control kept it at `0.090059 ms`. Tiny-row differences were not
     enough to justify serving tiers.
313. Prior RCCL source variants mostly remain near-ties after adding balanced
     `RCCL_TREES`. Direct graph replay under the balanced tree showed
     singlework+LL16+prims-inline and chainfan1+half2 were exact and slightly
     positive on some `.20/.30` rows, but not a primitive breakout. The best
     serving composition, balanced tree plus singlework+LL16+prims-inline,
     measured `.20` `61.581/62.390/58.947` and `.30`
     `61.149/61.857/58.275` for `c1_128/c1_2000/c1_10000`. This creates a
     `.20` sustained high-water mark but not a durable promoted winner: the
     `.20` margin over balanced-tree-only `58.780` is small and `.30` is
     tied/slightly mixed. Keep balanced `RCCL_TREES` plus the corrected
     specialized Tree/LL overlay as the promoted surface unless a later repeat
     or source patch changes the row mix. The follow-up balanced
     chainfan1+half2 serving check also starts cleanly and is exact, but it
     scored only `.20` `61.155/62.311/58.792` and `.30`
     `61.025/61.884/58.234`, so it is closed as non-promoting.
314. Balanced `RCCL_TREES` is confirmed by NPKIT as a real RCCL data-movement
     improvement, not a wrapper timing artifact. Using the same NPKIT-enabled
     specialized overlay and the same warmup trim, balanced topology moved
     instrumented graph-out from `0.083004/0.119770/0.166254 ms` to
     `0.065522/0.091807/0.122236 ms` for `1/2/4x5120`, while regressing
     `8x5120` from `0.174261` to `0.249671 ms`. The dominant NPKIT families
     dropped sharply: `tree_split_total -70.05%`,
     `prim_ll_data_process -73.82%`, `tree_split_broadcast -70.28%`, and
     `recv -70.93%`. `send` rose to `1.88%` share and
     `prim_ll_wait_send` remained only `0.46%`, so wait/spin tuning is still
     the wrong primary target. The next source work should focus on Tree/LL
     data path cost and legal collective-count reduction/coalescing below the
     public graph boundary.
315. Do not promote LL128 thresholding even after composing it with the
     balanced `RCCL_TREES` milestone. On `.20`, the threshold-specialized
     overlay plus balanced trees preserved the hot rows and improved only the
     very wide direct graph rows: `64x5120` moved from `0.599810` to
     `0.463402 ms/call`, and `128x5120` moved from `1.174703` to
     `0.891793 ms/call`. The serving ladder regressed versus the promoted
     balanced-tree surface, measuring strict c1/c1_2000/c1_10000 backend TPS
     of `60.800/61.846/57.814` versus `.20` balanced clean
     `61.252/62.236/58.780` and the `.20` singlework high-water
     `61.581/62.390/58.947`. Keep LL128 thresholding as a prefill/concurrency
     lever only; it is not part of the dense single-request decode winner.
316. Binary-fan specialization is exact but not a standalone dense decode
     promotion. Patch
     `rccl_721_gfx906_tree_ll_specialized_kernel_include_ar_binaryfan2_20260612.patch`
     narrows gfx906 Tree/LL primitive bounds to `FanSymmetric<2>`,
     `FanAsymmetric<2,1>`, and `FanAsymmetric<1,2>` when the promoted tree is
     binary (`tree->down[2] == -1`). The `.30` overlay built and same-settings
     direct replay versus the current specialized balanced overlay measured
     graph-out `1x5120 0.030772 -> 0.030649`, `2x5120 0.037979 -> 0.037804`,
     `8x5120 0.089675 -> 0.089609`, but `16x5120 0.162237 -> 0.186767`
     ms/call. Wider rows were mildly positive (`64x5120 -0.99%`,
     `128x5120 -0.88%`). This is a useful topology-aligned source component,
     but the c1 decode-dominant row moved only `0.40%`, so do not spend a
     serving lane on binary-fan alone. Consider it only as a component inside a
     larger singlework/coalescing source patch.
317. Combining binary-fan2 with the prior singlework/LL16/prims-inline branch
     produces a `.30` pre-release sustained decode milestone. The
     temporary patch series applied
     `rccl_721_gfx906_tree_ll_singlework_ll_lines16_primsll_inline_20260611.patch`
     plus the binary-fan insertion. Direct `.30` replay versus the prior
     singlework overlay improved most rows slightly (`1x5120 -0.47%`,
     `8x5120 -0.45%`, `16x5120 -0.48%`, `32x5120 -0.69%`) but regressed
     `2x5120` by `+5.14%`. The `.30` standard decode ladder was strict-valid
     and measured `60.968/62.045/58.520` backend TPS for
     `c1_128/c1_2000/c1_10000`, improving the prior `.30`
     balanced+singlework ladder of `61.149/61.857/58.275` at the long tier but
     not the strict smoke. This is a host-local sustained high-water on `.30`,
     not a global promotion; it needs `.20` confirmation before changing the
     global winner.
318. `.20` confirmation of singlework/LL16/prims-inline plus binary-fan2
     closes the branch as non-promoting. The `.20` direct graph replay was
     exact and improved every measured row versus prior `.20` singlework:
     `1x5120 -0.70%`, `2x5120 -0.37%`, `4x5120 -0.85%`,
     `8x5120 -0.86%`, `16x5120 -0.95%`, `20x5120 -0.97%`, and
     `32x5120 -0.92%`. Serving did not fully translate: the standard ladder
     measured strict-valid `61.309/62.307/58.831` backend TPS for
     `c1_128/c1_2000/c1_10000`. That is slightly above `.20` balanced clean
     sustained (`58.780`) but below the `.20` singlework high-water
     (`58.947`). Keep the promoted durable dense surface at balanced
     `RCCL_TREES` plus corrected specialized Tree/LL overlay, with
     singlework/LL16/prims-inline retained as the highest `.20` source
     high-water branch and binary-fan2 retained only as a component candidate.
319. Re-running the MLP/down selective skip upper bound on the current `.20`
     source high-water stack shows the gate is reachable if the exact MLP
     down-projection RowParallel boundary is materially reduced, but it also
     confirms the skip itself is not promotable. The lane used balanced
     `RCCL_TREES`, singlework/LL16/prims-inline RCCL, mutable RowParallel,
     native-runtime interleaved SwiGLU, O3 async, and skipped only
     MLP/down-proj RowParallel reductions. It measured backend decode
     `65.387/70.872/66.365` TPS for `c1_128/c1_2000/c1_10000`; `c1_128`
     generated reasoning only and fixed tiers were capped, so correctness is
     invalid by design. Treat this as an upper-bound source target, not a
     config: the next serious work is an exact TP8 MLP/down `1x5120`
     RowParallel collective/launch/count improvement under the current winner,
     then a full strict three-tier ladder if the direct evidence moves enough.
320. `.30` confirms the current-stack MLP/down skip upper bound across hosts.
     With the same balanced tree, singlework/LL16/prims-inline, mutable
     RowParallel, native-runtime interleaved SwiGLU, and MLP/down selective
     skip envelope, `.30` measured backend decode `62.899/69.421/65.134` TPS
     for `c1_128/c1_2000/c1_10000`. This is again correctness-invalid
     (`c1_128` filled the context with reasoning and no parser answer; fixed
     tiers are capped), but it clears the sustained `65 TPS` upper bound on a
     second host. The source conclusion is now stronger: the dense 27B gate is
     not waiting on another tiny RCCL primitive tweak; it needs an exact
     MLP/down RowParallel boundary change large enough to preserve TP8 compute
     while saving roughly the skipped-collective cost.
321. The clean-window profile of the current `.20` source high-water stack
     confirms the same source direction under the latest balanced-tree plus
     singlework branch. Run
     `qwen36_27b_c1_rocprofv2_balanced_singlework_highwater_cleanwindow_h20_20260612`
     used a 420s rocprofv2 delay so the capped c1_768 request landed after the
     strict warmup contamination window. The profile is attribution-only
     (`24.840` backend decode TPS under profiler overhead, capped/incomplete
     thinking output), but the kernel CSV is valid: `806,315` filtered rows and
     `22.678s` aggregate filtered kernel time. The top buckets were NCCL
     `13.388s / 59.03% / 99,702 calls`, LLGemm `3.868s / 17.06% / 184,523
     calls`, and interleaved SwiGLU `2.844s / 12.54% / 49,010 calls`. Compared
     with the earlier clean balanced-tree profile, the NCCL call count and
     share remain dominant; the gate decision does not change. The next serious
     source work must reduce, legally coalesce, or fuse the exact MLP/down
     RowParallel collective boundary rather than chase sub-one-percent Tree/LL
     variants.
322. The single-chunk Tree/LL branch is closed as non-promoting, and the build
     hygiene lesson is now explicit. The first `.20/.30/.10` build attempt
     accidentally omitted `ONLY_FUNCS`, so RCCL tried to compile and link the
     full generated collective matrix; the final LTO link stayed CPU-active for
     more than an hour. Fresh-root rebuilds with
     `ONLY_FUNCS="AllReduce TREE LL Sum f16"` completed and direct replayed
     correctly. Patch
     `rccl_721_gfx906_tree_ll_singlework_ll16_primsinline_singlechunk_20260612.patch`
     is exact, but not worth serving tiers. On `.20`, graph-out changed versus
     the prior singlework high-water as follows: `1x5120 0.030700 -> 0.031208`
     (`+1.65%`), `2x5120 0.037917 -> 0.038133` (`+0.57%`),
     `4x5120 0.052564 -> 0.052341` (`-0.42%`), `8x5120 0.088060 -> 0.087444`
     (`-0.70%`), `16x5120 0.160930 -> 0.160635` (`-0.18%`),
     `20x5120 0.196101 -> 0.195856` (`-0.12%`), and
     `32x5120 0.304225 -> 0.303952` (`-0.09%`). `.30` was similarly mixed:
     `1x5120 0.031225`, `2x5120 0.038160`, `4x5120 0.052412`,
     `8x5120 0.087582`, `16x5120 0.160787`, `20x5120 0.196437`, and
     `32x5120 0.303929`. The reduced-inline fallback on `.10` was exact but
     not better (`1x5120 0.032535`, `2x5120 0.039414`, `8x5120 0.088737`).
     Conclusion: guarded loop removal inside `runTreeSplit` is not a
     gate-scale change. Keep `ONLY_FUNCS` mandatory for RCCL source lanes, and
     return to MLP/down collective-count, launch, or producer/collective
     boundary work.
323. Grouped small-allreduce remains the strongest lower-bound primitive under
     the exact current high-water stack, but it is still not directly
     deployable. The refreshed `.20/.30` runs used balanced `RCCL_TREES`,
     singlework/LL16/prims-inline RCCL, Tree/LL, P2P disabled,
     `NCCL_MIN_NCHANNELS=4`, `NCCL_MAX_NCHANNELS=4`, and
     `NCCL_NTHREADS=128`. At group size `32`, graph replay reduced `1x5120`
     from `0.031948` to `0.013161 ms/call` on `.20` and from `0.031286` to
     `0.013182 ms/call` on `.30`. Wider rows still improve, but less:
     `.20` `8x5120` moves `0.089515 -> 0.075670`, and `32x5120` moves
     `0.307062 -> 0.290656`. Promote this as refreshed source-planning
     evidence only. The dense compiled graph still has no legal independent
     same-shape RowParallel clusters, and the single-call group wrapper
     regressed serving. The next source branch must create the grouped
     condition by construction: semantic schedule/count reduction, or a
     below-public-API generated/persistent collective path that amortizes
     fixed Tree/LL setup while preserving current dependencies.
324. Dummy-work grouping does not turn the grouped lower bound into a serving
     shortcut. The new dummy-group microbench grouped one useful `1x5120`
     allreduce with `0/1/3/7/15/31` dummy `1x1` allreduces and scored latency
     per useful boundary under the exact high-water overlay. On `.20`, graph
     useful-call latency went `0.043509`, `0.056585`, `0.056899`,
     `0.073609`, `0.114321`, `0.197275 ms` as dummy count increased. On
     `.30`, it went `0.044704`, `0.058376`, `0.059801`, `0.074599`,
     `0.115383`, `0.194561 ms`. All rows were correct, but all dummy padding
     regressed the useful boundary. The grouped lower-bound requires real
     independent reductions; padding a serial boundary is closed.
325. Full split4 native endpoint replacement is now a current-envelope
     rejection, not a stale unknown. The corrected `.20` retest used ROCm 7.2
     TP8 fp16, O3 async, balanced `RCCL_TREES`, the
     `singlework_ll16_primsinline` RCCL overlay, mutable RowParallel,
     native-runtime interleaved SwiGLU, and
     `POSTIR_REWRITE_MODE=split_endpoint_torchop` with `POSTIR_MATCH_LIMIT=128`
     and `split_channels=4`. The pass compiled, captured, and selected
     `128/128` direct boundaries on all ranks, and the strict uncapped smoke
     produced a valid parser-split thinking answer. The metric was polluted by
     a concurrent tiny diagnostic probe (`vllm_decode_request_count_delta=2`),
     so it is not a clean promotion datum, but the aggregate backend decode was
     only `45.809 TPS`, the smoke client rate was `13.850 TPS`, and the tiny
     `max_tokens=4` probe itself took `42.9s`. This branch is far below the
     current `.20` high-water `61.581 TPS` c1_128 backend decode and does not
     merit a full `c1_2000/c1_10000` ladder unless the endpoint implementation
     changes substantially. The retest also exposed runner hygiene issues:
     dense tier runs now accept `RCCL_TREES` directly, and the post-IR wrapper
     now honors caller-supplied `RCCL_OVERLAY_DIR` instead of silently remounting
     the older specialized RCCL overlay.
326. The post-split4 source surface is now narrowed to fixed-cost collective
     removal or a genuinely lower-floor primitive. The current high-water
     `1x5120` allreduce is about `30.7 us`, but the near-empty Tree/LL floor is
     already `~21.5 us`, so payload-only work inside the existing public
     allreduce can save only about `9.2 us` per MLP/down boundary, or
     `~0.59 ms/token` across the `64` MLP/down reductions. The sustained gate
     gap is about `1.58 ms/token`. Python/model-layer hooks are therefore
     exhausted as primary candidates: mutable in-place RowParallel is already
     active, residual pre-fold is exact but flat, consumer split-endpoint
     serving regressed, dummy grouping regressed, and RCCL LLMM1 precompute
     proved API plumbing but used the wrong execution geometry. The next
     promotable source milestone must either beat the `~21.5 us` public
     Tree/LL empty-call floor for TP8 `1x5120`, or legally remove/coalesce
     whole MLP/down collective boundaries.
327. The `.10` repeat of the current dense high-water software stack is a
     promotion rejection and a lane-quality data point. Host `.10` ran the ROCm
     7.2 TP8 fp16 O3 async path with explicit balanced `RCCL_TREES`, the
     host-specific `singlework_ll16_primsinline` RCCL overlay, mutable
     RowParallel, and native-runtime interleaved SwiGLU, but produced only
     `54.572/55.868/48.603` backend TPS for
     `c1_128/c1_2000/c1_10000`. The same branch remains the global source
     high-water on `.20` at `61.581/62.390/58.947` and is cross-supported by
     `.30` at `61.149/61.857/58.275`. Keep `.10` for component/topology
     experiments, not as the primary promotion lane for this Tree/LL branch.
328. Current-high-water NPKIT attribution closes further wait/send tuning on
     the actual `singlework_ll16_primsinline` Tree/LL branch. Under the `.20`
     high-water envelope, `prim_ll_wait_send` totaled only `19319.25` time
     units with mean `1.917` and p95 `3.25`, while `prim_ll_data_process`,
     receive, reduce, and broadcast families dominated. The bottleneck is now
     LL data movement/granularity, public collective count, or the public
     Tree/LL small-message floor, not spin waiting.
329. The LL32 primitive-width after-patch is rejected as a dense-gate source
     candidate. Changing `NCCL_LL_LINES_PER_THREAD` from `16` to `32` on top of
     the high-water branch produced `.20` direct graph replay of
     `1x5120=0.030724 ms`, `2x5120=0.037889 ms`,
     `4x5120=0.052669 ms`, `8x5120=0.088393 ms`,
     `16x5120=0.160572 ms`, `20x5120=0.197042 ms`, and
     `32x5120=0.305180 ms`. The LL16 baseline was
     `0.030700/0.037917/0.052564/0.088060/0.160930/0.196101/0.304225`.
     This is tied/noise at tiny rows and worse at several larger rows, so it
     does not merit `.30` confirmation or a serving ladder.
330. Split-thread allocation is now closed under the final high-water Tree/LL
     surface. The current NPKIT profile made `1 reduce / 3 broadcast` worth a
     corrected revisit, but stacking `split1r3b` on the balanced
     `singlework_ll16_primsinline` branch regressed every measured row on
     `.20` and `.30`. `.20` moved from
     `0.030700/0.037917/0.052564/0.088060/0.160930/0.196101/0.304225` to
     `0.041228/0.054081/0.078861/0.138707/0.259281/0.319120/0.499301`.
     `.30` was similarly worse at
     `0.041323/0.054149/0.084847/0.138599/0.258125/0.318109/0.497547`.
     More broadcast threads are not the answer; the default 2/2 split remains
     the local best allocation until the underlying dataflow changes.
331. Generic4 in-kernel unroll lowering is closed on the current high-water
     branch. The `Generic4 -> Unroll2` after-patch built and replayed
     correctly on `.20`, but it only tied the current floor:
     `0.030703/0.037841/0.052573/0.088071/0.161042/0.196171/0.303944`
     versus the high-water
     `0.030700/0.037917/0.052564/0.088060/0.160930/0.196101/0.304225`.
     The `Generic4 -> Unroll1` after-patch built and replayed correctly on
     `.30`, but it regressed the hottest row from `0.030711` to
     `0.041794 ms`. Simple in-kernel unroll retuning is therefore not a
     promotion path; the remaining dense gate work must change collective
     dataflow, public collective count, or launch grouping/coalescing.
332. Grouped RowParallel scale evidence changes the primary source target from
     MLP/down-only to broad RowParallel lower-floor coverage. Under the current
     `.20/.30` high-water RCCL overlays and balanced `RCCL_TREES`, grouped
     `1x5120` allreduce flattens at about `12.69 us/call` from group size
     `64` through `128`. Relative to the corrected high-water floor of about
     `30.74 us`, that saves roughly `18.05 us` per reduction. Covering only
     the `64` MLP/down reductions projects about `63.25 TPS` on the `.20`
     sustained tier, still short of the `65 TPS` gate. Covering `96+` of the
     `128` RowParallel reductions projects gate clearance, with all `128`
     projecting about `68.2 TPS` before secondary overheads. The next serious
     implementation should therefore be an exact generated/persistent/queued
     RowParallel primitive that can approach `13 us/call` across both
     attention and MLP/down reductions; MLP/down remains the first validation
     site, not the final scope.
333. A graph-submitted resident GPU queue is cheap enough to justify deeper
     RCCL-native persistent RowParallel source work. New harness
     `gfx906_persistent_queue_overhead_bench_20260612.cpp` captures `128`
     serial submit/wait kernels while one resident worker block per GPU polls
     device memory. On `.20/.30`, pure submit/wait overhead is
     `3.48/3.47 us/call`; submit/wait plus one local `1x5120` fp16 payload
     copy is `8.68/8.62 us/call`, with p95 about `8.74 us/call` and zero
     payload error. This is not a collective and not a serving result, but it
     removes the queue front-end as the immediate blocker. The next source
     step should wire the same graph-submitted descriptor idea into RCCL's
     existing Tree/LL connection/FIFO substrate or an equivalent generated
     work-loop and test exact `1x5120` cross-rank output.
334. The RCCL resident-worker path can be prototyped as a version-coupled
     sidecar against the exact high-water overlay. New harness
     `rccl_devcomm_channel_probe_20260612.cpp` and runner
     `run_rccl_devcomm_channel_probe_20260612.sh` compile against the built
     overlay's generated hipified RCCL headers, initialize an 8-rank
     communicator, run one exact `1x5120` fp16 allreduce, then launch a device
     kernel that reads `comm->devComm` as `ncclDevCommAndChannels`. Both `.20`
     and `.30` passed with rank/world matches, visible ring/tree metadata, and
     channel0 work counter `1` on every rank. This is not a TPS result, but it
     removes a major source-risk: the next descriptor/worker prototype can
     reuse RCCL's live communicator and Tree/LL topology instead of recreating
     topology or returning to raw HIP peer-memory paths.
335. The exact hot `1x5120` dense RowParallel allreduce uses three RCCL
     channels under the current high-water Tree/LL environment. The extended
     `.20` sidecar rerun
     `rccl_devcomm_channel_probe_host20_devcomm_probe_h20_chancounters_20260612_151510`
     reported channel counters `[1,1,1,0,0,0,0,0]` on every rank after one
     allreduce. The next manual descriptor or resident-worker prototype should
     reproduce that three-channel schedule first; assuming one channel or all
     four forced channels would be a false target.
336. The first manual Tree/LL descriptor target is now concrete enough to
     implement and falsify. Combining the RCCL scheduler source with the
     three-channel counter evidence gives a derived `1x5120` descriptor:
     active channels `0,1,2`, `countLo=1024`, `countMid=2048`,
     `countHi=2048`, LL chunk size `32768` bytes, LL chunk grains `2048`,
     `nWarps=2`, and specialized `devFuncId=0` for half-sum Tree/LL
     allreduce. This is not yet proof of a worker path; it is the exact-output
     target the next resident/generated work-loop must reproduce before timing
     matters.
337. A sidecar-generated Tree/LL work loop can now produce exact cross-rank
     `1x5120` fp16 output through RCCL's live `devComm` substrate. New harness
     `rccl_sidecar_tree_ll_work_probe_20260612.cpp` builds against the exact
     high-water overlay's generated headers, constructs the derived
     three-channel descriptor in device shared memory, and calls
     `RunWorkColl<AllReduce, half, sum, Tree, LL>`. `.20` and `.30` both
     passed with zero tensor error, and `.20` also passed without a normal
     warmup allreduce. This is the first proof that an out-of-tree/generated
     worker can execute a real RCCL Tree/LL collective path correctly.
338. The sidecar work-loop is a correctness milestone, not a speed promotion
     yet. The first timing runs accidentally omitted the high-water
     `RCCL_TREES`, which made stock graph timing look like `~44.6 us`. With
     the balanced tree string restored, the stock control returns to
     `~30.7 us/call`, while the sidecar graph path is `~38.9-39.2 us/call`
     and exact. `-O3`, `NDEBUG`, kernarg preload, zero dynamic shared memory,
     removing timed diagnostic writes, `RunWorkColl` unroll `4`, and
     `__launch_bounds__(NCCL_MAX_NTHREADS, 1)` did not close the gap.
339. Prebuilding the sidecar descriptor is not the missing `~8 us`. On `.20`
     with balanced trees, mode `0` per-call descriptor setup was
     `0.0389996 ms`, prebuilt parallel copy was `0.0390177 ms`, and thread-0
     struct copy was `0.0390365 ms`, all exact. Setup-only graph timing is
     only `~0.00216-0.00219 ms/call` on `.20/.30`. The gap to stock is
     therefore not dominated by descriptor construction or shared-memory setup.
     The next serious source step should target multi-work/batched Tree/LL
     execution or an embedded RCCL-side path that reuses the stock
     `ncclKernelMain`/singlework shape, rather than further polishing the
     one-work sidecar prologue.
340. Batched sidecar Tree/LL work is the first exact source-side collective
     variant to beat the corrected stock balanced RCCL microbench. Extending
     `rccl_sidecar_tree_ll_work_probe_20260612.cpp` to put multiple
     independent `1x5120` work descriptors in `ncclShmem.workStorage` and run
     `RunWorkBatch` showed exact output for multi-work counts `2`, `4`, and
     `8`. On `.20`, count `8` measured `0.02952-0.02962 ms/work` versus stock
     `0.03073-0.03086 ms/call`; the `.30` repeat measured
     `0.02958 ms/work` versus stock `0.03087 ms/call`. This is only a
     microbench/source milestone, not a serving gate clearance, but it proves
     that changing work shape can lower the small allreduce floor where
     one-work sidecar prologue tuning could not.
341. A resident sidecar worker is exact but does not fix the dependent
     single-reduction floor. New harness
     `rccl_persistent_tree_ll_worker_probe_20260612.cpp` launches a
     three-channel persistent Tree/LL sidecar worker per rank and feeds serial
     dependent `1x5120` commands through graph-submitted submit/wait kernels.
     It compiled and ran exactly on `.20` and `.30`, proving persistent
     sidecar sequencing is feasible. Timing rejects it as a speed path:
     `.20` stock graph `0.030868 ms/call` versus persistent worker
     `0.039302 ms/call`; `.30` stock graph `0.030844` versus persistent worker
     `0.039344`. The sidecar gap therefore is not launch overhead or
     one-time `devComm` copy. The next source work must move inside or more
     faithfully reproduce RCCL's stock `ncclKernelMain`/singlework execution
     shape instead of adding another sidecar queue layer.
342. The standalone `ncclKernelMain` args-packet parity probe is exact but not
     a speed promotion. New harness
     `rccl_kernelmain_args_parity_probe_20260612.cpp` builds a stock-style
     `ncclDevKernelArgs4K` packet in argument storage with three
     `ncclDevWorkBatch` entries and one shared `ncclDevWorkColl` descriptor.
     The first private-wrapper launch used `128` threads and faulted because
     gfx906 is wave64: `ncclKernelMain` spends the first two wavefronts on
     comm/channel loading, leaving no threads for `loadWorkBatchToShmem`.
     Retrying with `192` and `256` threads fixed correctness; `.20` and `.30`
     both produced exact `1x5120` fp16 output. Timing rejects the standalone
     wrapper: `.20`/`.30` private graph timing is about `0.0488 ms/call`,
     slower than the one-work sidecar `~0.039 ms` and well behind the clean
     stock comparator `~0.0308 ms`. Treat this as an integration-shape
     learning: `NCCL_NTHREADS=128`/`work->nWarps=2` describes the work
     wavefronts, not the full `ncclKernelMain` block size on wave64. The next
     attempt should instrument or modify the real RCCL enqueue/kernel path
     rather than approximating stock from a standalone wrapper.
343. The sidecar Tree/LL descriptor has a new source performance promotion:
     stock-like work occupancy matters. Updating
     `rccl_sidecar_tree_ll_work_probe_20260612.cpp` so the descriptor can use
     `work->nWarps=4` and a 256-thread launch made the single-work sidecar
     exact and faster than the balanced stock `1x5120` graph comparator.
     `.20` confirmed `0.029549 ms/call` sidecar versus `0.030739 ms/call`
     stock; `.30` confirmed the equivalent `RunWorkBatch(count=1)` path at
     `0.029751 ms/work` versus `0.030717 ms/call` stock. This overturns the
     earlier assumption that one-work sidecar was structurally stuck at
     `~0.039 ms`. It also shows why the old wording was too simple:
     `nWarps=2` was sufficient for exactness, but it under-filled the fast
     single-work execution shape on gfx906. The limit is now batching:
     combining `nWarps=4` with `RunWorkBatch` counts `4` or `8` timed out
     before final JSON, while the older `nWarps=2` batch path remains a
     separate source milestone. The next source step is to reconcile these two
     facts: preserve the `nWarps=4` single-work floor while making a legal
     multi-descriptor schedule, or move the occupancy fix into the real RCCL
     enqueue path before serving integration.
344. A second sidecar source milestone combines `nWarps=4` with long serial
     descriptor execution by avoiding the unsupported coll `RunWorkBatch`
     shape. The probe now has `SIDECAR_MULTIWORK_MODE=serial`, which reuses the
     copied `devComm`/channel state, resets `ncclShmem` barrier state for each
     descriptor, and runs one `RunWorkColl` at a time inside the same sidecar
     kernel. This is exact and repeatable: count `8` measured `.20`
     `0.028270 ms/work` and `.30` `0.028325 ms/work`; count `128` measured
     `.20` `0.027902 ms/work` and `.30` `0.027888 ms/work`, all with zero
     tensor error. The local stock comparator in the count-128 runs was
     `0.03127/.03125 ms/call`, so serial sidecar is about `10.7-10.8%` faster
     per `1x5120` work at the 128-work scale. Do not claim serving gate clear:
     dense post-IR analysis still says the actual RowParallel boundaries are
     dependency/intervening-use blocked, so this kernel cannot simply replace
     the current dependent sequence. The milestone does prove that a custom
     fused-collective worker can lower the small allreduce floor when it owns a
     legal sequence of ready descriptors; next source work must either create
     such a legal ready descriptor window in vLLM or fuse producer/consumer
     work into the serial sidecar boundary.
345. Wiring the promoted `nWarps=4` sidecar primitive into the direct
     `LLMM1 -> allreduce -> RMS` boundary bench confirms exactness under a
     gate-relevant producer/consumer shape, but it also bounds the serving
     payoff. The updated
     `rccl_llmm1_ar_rms_boundary_bench_20260612.cpp` compiles against the
     high-water RCCL overlay and measures public RCCL versus sidecar allreduce
     inside the same graph-replayed boundary. Full `.20/.30` runs with balanced
     Tree/LL, `NCCL_NTHREADS=128`, `work->nWarps=4`, and 256-thread sidecar
     launch were exact for both `attn_5120x768` and `mlp_down_5120x2176`.
     Savings were consistent but small: `.20` mean `mlp_down`
     `LLMM1+AR` improved `0.054750 -> 0.053942 ms`, and split RMS boundary
     improved `0.067222 -> 0.065891 ms`; `.30` improved `0.055067 ->
     0.053888 ms` and `0.066779 -> 0.065819 ms`. This promotes the boundary
     integration as correct source evidence, but rejects sidecar-only serving
     integration as the primary gate path: even broad coverage at roughly
     `0.8-1.3 us` per hot boundary is not enough to move the current
     `~58.95 TPS` c1_10000 regime to `65 TPS`. The next source work should
     target larger semantic removal or deeper producer/collective fusion,
     not just swapping dependent public allreduces for this one-work sidecar.
346. Inline sidecar `allreduce + RMS` is exact but does not unlock a larger
     consumer-fusion win. A new sidecar boundary kernel performs the Tree/LL
     allreduce and then, inside the same three-channel sidecar launch,
     computes residual add, per-channel sum-of-squares, global RMS via an
     atomic endpoint, residual output, and normalized output. Full `.20/.30`
     graph replay with the same balanced Tree/LL stack was exact for
     `attn_5120x768` and `mlp_down_5120x2176`. The measured improvement over
     sidecar allreduce plus the separate split-RMS endpoint is only about
     `0.09-0.12 us` per boundary; total improvement versus public
     `LLMM1+AR+splitRMS` is about `1.19 us` on `.20` attention, `1.23 us` on
     `.20` MLP-down, `1.42 us` on `.30` attention, and `1.54 us` on `.30`
     MLP-down. Promote this as correctness evidence for inline endpoint work,
     but reject it as the dense gate path. The main remaining gap is not the
     separate RMS endpoint launch; it is the number and placement of dependent
     RowParallel collective boundaries.
347. Carrying the `nWarps=4` occupancy idea into the real public RCCL planner
     is exact but rejected. Patch
     `rccl_721_gfx906_tree_ll_singlework_nwarps4_after_20260612.patch` stacks
     on the singlework/LL16/prims-inline high-water patch and forces
     single-node gfx906 fp16 sum Tree/LL allreduce work records to
     `nThreads = 4 * comm->WarpSize`. The overlays built on `.20` and `.30`
     and direct graph replay was exact, but not promotable. `.20` moved from
     `0.030700/0.037917/0.052564/0.088060/0.160930/0.196101/0.304225` to
     `0.030726/0.037884/0.052569/0.088024/0.161111/0.196316/0.304241` for
     `1/2/4/8/16/20/32x5120`. `.30` moved from
     `0.030711/0.043593/0.052496/0.087969/0.160635/0.195827/0.303766` to
     `0.030719/0.037942/0.058060/0.087949/0.160725/0.196162/0.305066`. The
     `.20` result is a tie and `.30` has a large `4x5120` regression, so no
     serving ladder is justified. The sidecar occupancy result remains useful,
     but not as a simple public-RCCL planner override.
348. `NCCL_CHUNK_SIZE` tuning is closed for the current high-water dense
     decode surface. The hot `1x5120` descriptor uses three active channels
     under the forced min/max four-channel envelope, so chunk size was
     retested with explicit balanced `RCCL_TREES` and the
     singlework/LL16/prims-inline overlay. The existing `4096` check was a
     tie. New `.20/.30` direct graph replays with `NCCL_CHUNK_SIZE=2048`
     measured `.20` `0.030711/0.038060/0.052550/0.088080/0.161046` and
     `.30` `0.030720/0.038230/0.052645/0.113412/0.160886` for
     `1/2/4/8/16x5120`. `NCCL_CHUNK_SIZE=8192` measured `.20`
     `0.030718/0.037898/0.052701/0.088057/0.160956` and `.30`
     `0.030890/0.038068/0.052700/0.105946/0.160849`. All rows were exact,
     but none beat the baseline row family on both hosts, and `.30` regressed
     `8x5120`. Do not spend serving lanes on chunk-size-only variants.
349. The current dense high-water does not have an unused simple in-place
     RowParallel fix waiting to be enabled. The `.20` and `.30` remote
     `rowparallel_mutable_ar_*_20260608.py` overlays are hash-identical,
     mounted by the high-water launchers, and run with
     `VLLM_GFX906_ROWPAR_MUTABLE_AR=1`. The deployed `RowParallelLinear`
     sends contiguous CUDA fp16/bf16 outputs through
     `tensor_model_parallel_all_reduce_inplace`, and the deployed
     `parallel_state.py` path calls `pynccl_comm.all_reduce(input_, input_)`
     when PyNccl is present. This closes the hidden extra-output-buffer
     hypothesis for the current `.20/.30` high-water branch. Future source work
     must change the collective boundary, not merely ask RowParallel to mutate
     its output buffer.
350. Volatile 128-bit LL FIFO reads are rejected for the current Tree/LL
     receiver path. The candidate patch
     `rccl_721_gfx906_tree_ll_singlework_volatile_b128_load_after_20260612.patch`
     stacked on singlework/LL16/prims-inline and replaced the gfx906 LL FIFO
     pair of 64-bit non-temporal loads with a guarded volatile
     `uint32_t[4]` vector load. It compiled on `.30`, but direct TP8 graph
     replay regressed the hot rows before stalling on the wider row family:
     `1x5120` moved from the `.30` high-water `0.030711 ms` to
     `0.035552 ms`, `2x5120` to `0.045443 ms`, and `4x5120` from
     `0.052496 ms` to `0.065026 ms`; all completed rows were exact. Artifact:
     `runs/rccl_direct_allreduce_graph_bench_host30_volatile_b128_v3_h30_20260612_181737`.
     This closes the simple "make the LL FIFO load wider/volatile" idea. The
     NPKIT recv/data-process signal still matters, but the current
     non-temporal pair load appears better for gfx906 than this vector load
     form.
351. The resident Tree/LL sidecar worker becomes a real source milestone when
     it uses the sidecar-proven occupancy. The persistent worker probe was
     updated to expose `PERSISTENT_WORKER_NWARPS` and
     `PERSISTENT_WORKER_BLOCK_THREADS`. Under the promoted balanced Tree/LL
     envelope, `.20` `nWarps=2` measured `0.039367 ms/call`, while `.20`
     `nWarps=4` and 256 worker threads measured `0.028754 ms/call`; `.30`
     reproduced `0.028727 ms/call`. All rows were exact. Artifacts:
     `runs/rccl_persistent_tree_ll_worker_host20_persistent_nw2_control_h20_20260612_182756`,
     `runs/rccl_persistent_tree_ll_worker_host20_persistent_nw4_h20_repeat_20260612_182721`,
     and
     `runs/rccl_persistent_tree_ll_worker_host30_persistent_nw4_h30_repeat_20260612_182721`.
     This is superseded historical dense source evidence, but it is now the first exact persistent
     collective substrate that beats the public `1x5120` graph comparator on
     the good `.20/.30` topology. The next implementation question is serving
     integration for fixed-shape RowParallel decode allreduces, not more
     sidecar occupancy tuning.
352. Pointer-carrying persistent worker submit is also exact and stays below
     the public allreduce floor, which makes a serving overlay technically
     worth attempting. The probe now supports `PERSISTENT_WORKER_SET_PTRS=1`,
     causing each graph-captured submit kernel to write `send/recv`, fence,
     then increment the request counter. This models vLLM RowParallel decode
     better than the fixed-pointer worker. Results: `.20`
     `0.028995 ms/call` and `.30` `0.028876 ms/call`, both exact. Artifacts:
     `runs/rccl_persistent_tree_ll_worker_host20_persistent_nw4_setptr_h20_20260612_183106`
     and
     `runs/rccl_persistent_tree_ll_worker_host30_persistent_nw4_setptr_h30_20260612_183108`.
     The pointer/fence cost is small enough to proceed to a guarded vLLM
     prototype for fp16 in-place `count=5120` RowParallel allreduces.
353. The persistent worker C ABI is viable, but its lifetime must be scoped.
     A dlopened shared runtime built from
     `gfx906_persistent_tree_ll_ar_runtime_20260612.cpp` can create the
     worker, submit an in-place `fp16 count=5120` allreduce, synchronize, and
     return exact output through a plain C++ executable. The passing artifact
     is
     `runs/gfx906_persistent_ar_capi_smoke_host20_persistent_ar_capi_prefill_h20_20260612_190127`.
     However, the same C API smoke hung when an ordinary fill kernel was
     launched after `gfx906_persistent_ar_create`; rank logs stopped at
     `fill_begin` in
     `runs/gfx906_persistent_ar_capi_smoke_host20_persistent_ar_capi_markers_h20_20260612_185858`.
     This means the previous Python/PyNccl hang is not a ctypes-only problem:
     a resident worker started too early can block normal compute kernels.
     The source milestone remains promoted, but serving integration must use
     lazy/scoped worker lifetime or another design that leaves LLMM1/GEMM
     kernels free between collective windows.
354. Scoped persistent-worker lifetime has two sharply different outcomes.
     Per-call `create -> submit/sync -> destroy` is exact but rejected at
     about `3.60 ms` per allreduce on `.20`
     (`runs/gfx906_persistent_ar_capi_smoke_host20_persistent_ar_capi_lifecycle10_h20_20260612_190411`).
     A diagnostic post-create fill also passes after the fill kernel has been
     prewarmed, which points to code-object first launch as the cause of the
     earlier post-create fill hang
     (`runs/gfx906_persistent_ar_capi_smoke_host20_persistent_ar_capi_postfill_h20_20260612_190510`).
     Most importantly, a keep-alive worker with prewarmed normal fill kernels
     interleaved before every persistent allreduce remains exact
     (`runs/gfx906_persistent_ar_capi_smoke_host20_persistent_ar_capi_keephandle_h20_20260612_190621`).
     The C host-launch path measures about `0.070 ms/call`, slower than the
     graph-captured sidecar `~0.029 ms/call`, so serving integration must rely
     on vLLM graph capture of the submit path. Do not pursue per-call worker
     teardown; pursue prewarm plus keep-alive or a yielding resident worker.
355. The permanent resident worker is not serving-safe under the current vLLM
     O3 CUDA graph capture path. Three `.20` serving smokes all reached the
     relevant vLLM startup/capture phase and then deadlocked with worker
     heartbeats and saturated GPUs: first-create inside graph capture
     (`runs/qwen36_27b_decode_tiers_rocm72_dense_persistent_ar_smoke_h20_host20_persistent_ar_smoke_h20_20260612_191000`),
     explicit preinit on graph capture
     (`runs/qwen36_27b_decode_tiers_rocm72_dense_persistent_ar_preinit_smoke_h20_host20_persistent_ar_preinit_smoke_h20_20260612_192238`),
     and skip-profile plus final-capture preinit
     (`runs/qwen36_27b_decode_tiers_rocm72_dense_persistent_ar_skipprofile_smoke_h20_host20_persistent_ar_skipprofile_smoke_h20_20260612_193259`).
     The third run is decisive: public RCCL allreduce survived the graph
     profiling pass, KV cache allocation completed, and the hang started only
     after persistent AR preinitialized for the later/final graph-capture
     context. Keep the `nWarps=4` worker as a source primitive, but reject the
     current permanent-resident serving overlay. The next collective path must
     be graph-capture-compatible without a resident wait loop, or must move the
     improvement into the public RCCL kernel/enqueue path.
356. The non-resident sidecar allreduce runtime is the latest dense serving
     milestone. Replacing fixed-shape RowParallel `numel=5120` reductions with
     a dlopened one-shot sidecar Tree/LL path avoids the permanent-worker
     graph-capture deadlock and completes the standard O3/async dense ladder on
     both `.20` and `.30`. Results: `.20`
     `61.959/62.957/59.431` and `.30` `61.983/62.777/59.054` backend TPS for
     strict c1, `c1_2000`, and `c1_10000`. This is a serving/source
     high-water, but not a `65 TPS` gate clear. Artifacts:
     `qwen36_27b_decode_tiers_rocm72_dense_sidecar_ar_balanced_h20_host20_sidecar_ar_balanced_h20_20260612_200429`
     and
     `qwen36_27b_decode_tiers_rocm72_dense_sidecar_ar_balanced_h30_host30_sidecar_ar_balanced_h30_20260612_200429`.
357. Target-selective sidecar routing is now graph-safe, but simple MLP-only
     and attention-only targeting do not promote. The first selector attempt
     mutated `os.environ` inside
     `RowParallelLinear.forward()` and Dynamo rejected `posix.putenv`; the
     direct PyNccl bypass then hit Dynamo's ctypes `_SimpleCData.__new__`
     guard. The correct integration is a second custom op,
     `torch.ops.vllm.all_reduce_inplace_public`, so non-target RowParallel
     callsites can use the normal PyNccl path without leaving the compiled
     graph. With that fix, `.20` MLP-only strict c1 completed validly at
     `61.692` backend TPS and `.30` attention-only strict c1 completed validly
     at `61.154` backend TPS. That proves the selector infrastructure, but the
     all-sidecar route remains the current high-water. Move back to reducing
     RowParallel boundary count or lowering collective/enqueue overhead rather
     than spending more lanes on simple MLP-vs-attention selector variants.
358. Sidecar worker-shape tuning is closed for the current one-shot Tree/LL
     runtime. Exact-shape probes on `.20` and `.30` show that `nWarps=4` with a
     256-thread block remains the only promotable sidecar launch shape for the
     `1x5120` boundary. `nWarps=3` with 192 threads is exact but slower
     (`~0.0344 ms/work` versus `~0.0296 ms/work`), while `nWarps=5/6/8` with
     matching 320/384/512-thread blocks and the extra-block `nWarps=4` /
     320-thread probe fail at kernel launch on both hosts. Artifacts include
     `runs/rccl_sidecar_tree_ll_work_probe_host20_sidecar_exactshape_nw3_bt192_h20_20260612_211540`,
     `...nw4_bt256...`, `...nw5_bt320...`, `...nw6_bt384...`, `...nw8_bt512...`,
     and the matching `.30` runs. This rejects worker geometry as a likely
     gate-clearing branch; the remaining high-value path is still exact
     MLP/down RowParallel boundary count, schedule, or producer/collective
     integration.
359. Adding a low-duty wait loop to the permanent resident worker does not fix
     the vLLM O3 graph-capture hang. The runtime now has an opt-in
     `GFX906_PERSISTENT_AR_IDLE_SLEEP_ITERS` knob using `s_sleep` while idle;
     `.20` built
     `runs/gfx906_persistent_ar_runtime_build_host20_yield_sleep4_h20_20260612_212349`
     and passed a C ABI smoke exactly with `idle_sleep_iters=4`
     (`runs/gfx906_persistent_ar_capi_smoke_host20_yield_sleep4_h20_20260612_212419`).
     Serving still reproduced the old failure signature: after KV allocation
     and persistent handle preinit for graph capture, logs stopped at
     `preinitialized for graph_capture`, shared-memory broadcast warnings
     repeated, the port never opened, and all eight GPUs sat at 100% until the
     lane was torn down. Artifact:
     `runs/qwen36_27b_decode_tiers_rocm72_dense_persistent_ar_yield_sleep4_h20_20260612_212438_host20_20260612_212438`.
     This closes "busy-wait starvation" as the simple explanation. The next
     resident-style attempt would need a different lifecycle, such as launching
     the worker outside graph capture after capture completion, or a
     graph-captured queue primitive that does not require a permanent live
     worker during capture.
360. Deferred resident-worker lifecycle is now proven to pass O3 startup and
     final CUDA graph capture, but it is not serving-correct yet. The `.20`
     `defer_envfix` run fixed two real integration bugs: the container was not
     receiving `GFX906_PERSISTENT_AR_DEFER_WORKER=1` or
     `VLLM_GFX906_PERSISTENT_AR_START_AFTER_CAPTURE=1`, and the post-capture
     hook was not visibly active. After wiring those through
     `run_qwen36_27b_persistent_ar_decode_lane_20260612.sh`, the run showed
     `sitecustomize active`, disabled persistent AR during
     `profile_cudagraph_memory`, completed final graph capture, and started
     one deferred worker on each of 8 ranks after capture. This is a real source
     lifecycle milestone. The first strict c1 request then failed with
     `TimeoutError: RPC call to sample_tokens timed out` and HTTP 500 after a
     single scheduled prefill of 431 tokens. Artifact:
     `runs/qwen36_27b_decode_tiers_rocm72_dense_persistent_ar_defer_envfix_h20_20260612_215641_host20_20260612_215641`.
     Current interpretation: the remaining problem is post-capture runtime
     synchronization between captured submit kernels and the resident worker,
     not model load, graph-memory profiling, or final graph capture.
361. Snapshot diagnostics isolated the deferred resident-worker failure to the
     post-capture idle worker, not to command publication. The ordered-atomic
     runtime built on `.20` and passed C ABI exact smoke, but serving still
     timed out in `sample_tokens`. A snapshot ABI plus Python watchdog then
     showed every rank live with `ready=3`, `wait=1`, stable tree counters, and
     `request=0/done=0/send=0/recv=0` all the way to the HTTP 500 timeout.
     That means the first request never reaches the graph-captured persistent
     allreduce submit node. The decisive no-start control used the same
     capture-only replacement path but did not launch deferred workers after
     capture; it completed instead of timing out, producing `15953` tokens with
     backend decode `72.295 TPS`, but strict gate invalid because it ended at
     `finish_reason=length` with no parser-visible answer. Artifacts:
     `runs/qwen36_27b_decode_tiers_rocm72_dense_persistent_ar_snapshot_watchdog_synced_h20_20260612_224754_host20_20260612_224754`
     and
     `runs/qwen36_27b_decode_tiers_rocm72_dense_persistent_ar_capture_only_no_start_control_h20_20260612_225919_host20_20260612_225919`.
     New interpretation: an idle resident RCCL Tree/LL worker launched after
     capture interferes with the first normal prefill/model execution before
     any captured replacement allreduce is submitted. The next resident branch
     should start workers only after prefill, use a parked worker that does not
     occupy execution resources, or move the speedup into the non-resident or
     public RCCL enqueue path.
362. Starting deferred persistent allreduce workers only after the first real
     prefill fixes the serving timeout and proves the captured command path can
     execute correctly, but the current worker footprint is too expensive for
     promotion. The `.20` start-after-prefill run kept persistent AR disabled
     during graph-memory profiling and final graph capture, then armed the
     workers after the strict request scheduled a real 431-token prefill. The
     watchdog showed live command progress after arming (`done_count` tracking
     `request * 3` across the three Tree/LL channels), and the strict smoke
     completed validly with `finish_reason=stop`, parser-split post-think
     answer, `3222` completion tokens, and `58.272` backend decode TPS.
     Artifact:
     `runs/qwen36_27b_decode_tiers_rocm72_dense_persistent_ar_start_after_prefill_h20_20260612_232234_host20_20260612_232234`.
     This is a source lifecycle milestone, not a dense gate milestone: it is
     well below the non-resident sidecar high-water (`~63` backend TPS on
     c1_2000). The next test should reduce resident-worker occupancy
     (`nWarps=2/128`, `nWarps=3/192`) and promote only if correctness stays
     valid while decode moves back toward the non-resident winner.
363. Lower-occupancy start-after-prefill resident workers did not recover the
     dense decode high-water. The `.20` `nWarps=2/block_threads=128` strict
     smoke completed validity-clean with `3228` completion tokens,
     `finish_reason=stop`, parser-split post-think answer, and `57.023`
     backend decode TPS. The `.30` `nWarps=3/block_threads=192` strict smoke
     also completed validity-clean with `3729` completion tokens and `54.258`
     backend decode TPS. Artifacts:
     `runs/qwen36_27b_decode_tiers_rocm72_dense_persistent_ar_start_after_prefill_nw2_h20_20260612_233739_host20_20260612_233739`
     and
     `runs/qwen36_27b_decode_tiers_rocm72_dense_persistent_ar_start_after_prefill_nw3_h30_20260612_233739_host30_20260612_233739`.
     This rejects the simple "use fewer resident worker warps" fix. The
     resident path is now a correctness/source-lifecycle milestone, but the
     primary gate path should move back to non-resident sidecar/public RCCL
     enqueue overhead, parked-worker design, or graph-level reduction count
     reduction rather than more resident worker shape sweeps.
364. MLP-down LLMM1 residual pre-fold composes correctly with the current
     non-resident sidecar allreduce stack, but it does not promote. The first
     `.20` attempt was invalid as a source test because the remote
     `qwen2_moe_interleaved_swiglu_20260608.py` overlay was older than the
     local overlay and did not contain the down-prefold path; it measured
     `57.761` backend TPS, matching the same-host sidecar control at `57.604`.
     After staging the corrected overlay to `.20` and `.30`, both hosts logged
     `gfx906 MLP down LLMM1 residual pre-fold enabled` for MLP down
     projections and completed strict uncapped smokes cleanly. Corrected
     results: `.20` `3251` completion tokens, `finish_reason=stop`,
     parser-split answer, `58.171` backend decode TPS; `.30` `3975` completion
     tokens, `finish_reason=stop`, parser-split answer, `55.601` backend
     decode TPS. A forced `.20` full ladder with the corrected active path then
     produced valid strict smoke `3890` tokens at `57.930` backend decode TPS,
     `c1_2000` at `58.811` TPS, and `c1_10000` at `55.789` TPS. Artifacts:
     `runs/qwen36_27b_decode_tiers_rocm72_dense_sidecar_downprefold_corrected_h20_20260613_001738_host20_20260613_001738`
     and
     `runs/qwen36_27b_decode_tiers_rocm72_dense_sidecar_downprefold_corrected_h30_20260613_001738_host30_20260613_001738`;
     full ladder:
     `runs/qwen36_27b_decode_tiers_rocm72_dense_sidecar_downprefold_corrected_full_h20_20260613_002953_host20_20260613_002953`.
     Decision: reject this composition as a serving promotion. The math is
     exact and active, but the saved residual/RMS-side work is too small at the
     current collective boundary. The next serious source work should not add
     more consumer-side polish; it should reduce RowParallel collective launch
     count/overhead, move the non-resident sidecar closer to the public RCCL
     enqueue path, or design a parked worker that avoids the resident-worker
     occupancy penalty.
365. The `.10` overlay-fix rerun is a lane-quality rejection, not a
     high-water reproduction. After fixing the remote overlay staging hashes,
     host `.10` ran the same non-resident sidecar high-water stack and produced
     strict c1 `54.843` backend decode TPS, `c1_2000` `54.760`, and
     `c1_10000` `51.270`. Artifact:
     `runs/qwen36_27b_decode_tiers_rocm72_dense_sidecar_ar_h10_overlayfix_20260613_011202_host10_20260613_011202`.
     Direct VRAM-byte checks showed all eight `.10` GPUs expose
     `34342961152` bytes, so the older rocm-smi product-name/product-string
     ambiguity was not a 16 GB card problem. The likely issue is platform and
     topology difference, not model/config reproduction. Keep `.10` useful for
     component/topology probing, but promote dense serving winners only after
     `.20/.30` reproduction unless a `.10` path independently beats them.
366. Fusing the LLMM1 producer and sidecar Tree/LL allreduce into one kernel is
     exact but slower, closing the simple spin-barrier fusion branch. The new
     `fused_llmm1_sidecar_tree_ll_1x5120_kernel` waits for all producer blocks
     through monotonic counters and then runs the same three-channel
     `RunWorkColl<AllReduce, half, Sum, Tree, LL>` sidecar work inside the
     producer launch. Smoke2 results on `.20` were exact but regressed
     `attn_5120x768` from sidecar `0.053225 ms` to fused `0.066186 ms`, and
     `mlp_down_5120x2176` from sidecar `0.069524 ms` to fused `0.095438 ms`.
     `.30` reproduced the same direction: `0.053273 -> 0.066458 ms` for
     attention and `0.069546 -> 0.095453 ms` for MLP-down. Artifacts:
     `runs/rccl_llmm1_ar_rms_boundary_host20_20260613_013344_fused_llmm1_sidecar_smoke2_h20`
     and
     `runs/rccl_llmm1_ar_rms_boundary_host30_20260613_013345_fused_llmm1_sidecar_smoke2_h30`.
     Decision: do not pursue a single-kernel producer-ready spin barrier. The
     next collective path needs a parked-worker design that does not occupy
     execution resources while idle, a lower-overhead non-resident/public RCCL
     enqueue path, or a legal graph-level reduction-count change.
367. Role-specializing the sidecar Tree/LL primitive by fixed tree arity is an
     exact microbench milestone but not a serving breakthrough. The throwaway
     candidate narrowed each rank to the actual root/leaf/depth-1 reduce and
     broadcast fan shape while keeping the promoted three-channel descriptor
     and `nWarps=4` / `block_threads=256`. Repeat graph replay improved `.20`
     from `0.0294769` to `0.0293125 ms/call` and `.30` from `0.0295417` to
     `0.0293225 ms/call`, but the full dense vLLM ladder regressed: `.20`
     produced strict c1 `57.807`, c1_2000 `58.800`, and c1_10000 `55.799`
     backend TPS; `.30` produced `56.102`, `56.705`, and `53.850`. Artifacts:
     `runs/qwen36_27b_decode_tiers_rocm72_dense_sidecar_role_specialized_h20_host20_sidecar_role_specialized_serving_h20_20260613_023437`
     and
     `runs/qwen36_27b_decode_tiers_rocm72_dense_sidecar_role_specialized_h30_host30_sidecar_role_specialized_serving_h30_20260613_023437`.
     This rejects generic fan-template overhead as a gate-scale source. Keep
     the learning, but do not promote the role-specialized runtime.
368. A one-block serial-channel sidecar is exact but far too slow, so it is not
     a viable base for a lower-occupancy parked worker. The throwaway
     `sidecar_tree_ll_1x5120_oneblock_serial_channels_kernel` kept the
     promoted descriptor and `4/256` launch geometry but launched one block per
     rank and processed channels `0,1,2` serially. `.20` measured
     `0.128805 ms/call` graph replay versus same-session stock RCCL
     `0.049891`; `.30` measured `0.125897` versus `0.044675`. Both were exact.
     This closes the simple "reduce resident occupancy by serializing
     channels" branch. A parked-worker design must preserve channel parallelism
     without occupying CUs while idle, or the main path should return to
     graph-level RowParallel count/schedule reduction and non-resident launch
     boundary work.
369. The post-IR blocker trace proves grouped allreduce is structurally blocked
     in the current dense graph, not merely missing a local cleanup rewrite.
     The diagnostic overlay
     `overlays/postir_blocker_trace_20260613/gfx906_post_ir_ar_rms_fusion.py`
     ran on `.20` in
     `runs/qwen36_27b_decode_tiers_currentwinner_blocker_trace_h20_host20_20260613_blocker_trace_h20`
     and reached backend readiness. All eight ranks reported the same
     `128` adjacent collective-pair histogram: `64` MLP paths
     (`getitem -> convert -> add -> mul -> convert -> swiglu -> rocm_gemm_5120`),
     `47` normal attention/out-projection paths, `16` GDN/linear-attention
     paths, and `1` initial embedding/attention path. Every next collective
     input is a `rocm_gemm_5120`, and `127/128` first dependent users are the
     previous allreduce's output `getitem`. This means the next allreduce input
     truly depends on the previous hidden state. Do not pursue standalone
     grouped-AR integration for this graph; pursue full boundary fusion,
     lower non-resident enqueue overhead, or a schedule-level change that
     legally reduces RowParallel count.
370. Target-selective sidecar routing is graph-safe infrastructure, but not a
     dense gate path by itself. Full ladders closed the misleading `mlp`-only
     smoke signal: `.20` `mlp`-only reached only strict/c1_2000/c1_10000
     `57.420/58.130/55.138`, `.30` `mlp`-only reached
     `55.238/55.824/52.996`, and `.10` `other`-only reached
     `53.825/52.562/48.853`. These all underperform the all-sidecar
     high-water of `.20` `61.959/62.957/59.431` and `.30`
     `61.983/62.777/59.054`. Keep the second public custom op and corrected
     attention selector for future experiments, but do not reopen simple
     MLP/attention/other target routing as a promotion lane.
371. Prebuilding the sidecar `ncclDevWorkColl` descriptor is exact but not
     portable or large enough to matter. The promoted `4/256` sidecar geometry
     was retested with mode `0` baseline, mode `1` byte-copy template, and mode
     `2` single-thread struct copy. `.20` improved only from `0.044033` to
     `0.043997 ms/call`, while `.30` regressed from `0.043986` to
     `0.044024-0.044049 ms/call`. Do not carry prebuilt-work-template logic
     into the serving runtime; descriptor setup is not the gate-scale cost.
372. A sidecar collective kernel cannot be parked ahead of the LLMM1 producer
     with a device spin-ready counter on the current gfx906 ROCm 7.2 stack. The
     volatile-ready and atomic-ready `.20` smokes both compiled, then
     hard-stalled with all GPUs at `100%` and no rank progress before manual
     cleanup. The safe event-gated two-stream variant is exact but not useful:
     full-count `.20/.30` deltas versus same-stream sidecar are only
     `+0.000003844/-0.000024919 ms` and
     `+0.000006445/+0.000014985 ms` for `attn_5120x768` /
     `mlp_down_5120x2176`. Do not promote two-stream sidecar launch or
     spin-ready parked sidecar windows to serving; keep them as diagnostic
     harness knobs only.
373. `NCCL_NTHREADS` unset remains rejected on the current non-resident
     sidecar high-water. The reproducible wrapper
     `run_qwen36_27b_sidecar_decode_lane_20260613.sh` reran the full strict
     c1 / `c1_2000` / `c1_10000` ladder on `.20` and `.30` with the same
     ROCm 7.2, balanced-tree, O3 async, mutable RowParallel, interleaved
     SwiGLU, and sidecar runtime stack, changing only `NCCL_NTHREADS` to
     unset. Results were `.20` `62.213/63.195/59.685` and `.30`
     `61.972/62.786/59.137` backend TPS. This does not beat the promoted
     high-water by a durable margin and keeps the dense serving envelope on
     `NCCL_NTHREADS=128`.
374. Channel-local producer readiness inside the fused LLMM1+sidecar boundary
     is exact but still not a promotion. The microbench now exposes
     `BOUNDARY_FUSED_CHANNEL_READY=1`, making each sidecar channel wait only
     for its own output slice (`0:1024`, `1024:3072`, `3072:5120`) instead of
     all LLMM1 producer blocks. Runs
     `rccl_llmm1_ar_rms_boundary_host20_20260613_channel_ready_h20_channel_ready_h20`
     and
     `rccl_llmm1_ar_rms_boundary_host30_20260613_channel_ready_h30_channel_ready_h30`
     were zero-error on both hosts. `.20` fused-channel-ready measured
     `0.065661 ms` for attention and `0.093162 ms` for MLP-down, while the
     same-run separate `LLMM1 -> sidecar allreduce` boundary measured
     `0.052329 ms` and `0.068780 ms`. `.30` reproduced the same gap:
     `0.065718/0.093228 ms` fused versus `0.052316/0.068721 ms` separate.
     Keep the flag as a diagnostic knob, but close simple in-kernel
     producer/collective overlap: sidecar blocks spinning inside the producer
     launch still lose far more than channel-local readiness recovers.
375. The same sidecar-winner rocprof on `.20` confirms the latest dense
     bottleneck split is durable across hosts. Run
     `qwen36_27b_sidecar_winner_rocprof_h20_20260613_sidecar_winner_h20_profile`
     produced `806,637` used rows and `23.719s` filtered kernel time. The top
     rows were `sidecar_tree_ll_1x5120_kernel` at `12.483s`, `52.63%`,
     `98,059` launches, `127.3 us` average; `LLGemm1_kernel` at `3.863s`,
     `16.29%`; interleaved SwiGLU at `2.851s`, `12.02%`; and public
     `ncclDevKernel_Generic_4` at `2.014s`, `8.49%`, `1,672` launches,
     `1204.7 us` average. The `.30` profile has the same ordering but slower
     collective rows (`140.4 us` sidecar average and `2.292s` public NCCL).
     Treat this as a diagnostic breakthrough: `.20/.30` agree that the next
     serious work is residual collective attribution and sidecar data path
     reduction, not another small elementwise/RMS cleanup.
376. Public NCCL attribution on the sidecar winner shows the residual public
     bucket is mostly prefill and CUDA-graph capture, not unhandled c1 decode.
     A disabled-by-default trace hook in
     `overlays/gfx906_persistent_ar_20260612/parallel_state.py` recorded
     public communication metadata during two capped diagnostics on `.20`.
     Run
     `qwen36_27b_decode_tiers_rocm72_dense_sidecar_public_comm_trace_h20_host20_20260613_public_comm_trace_h20`
     showed runtime prefill misses at `2048x5120` and graph-capture misses at
     `8x5120`. The follow-up
     `qwen36_27b_decode_tiers_rocm72_dense_sidecar_public_comm_trace_decode16_h20_host20_20260613_public_comm_trace_decode16_h20`
     showed the same structure plus capture misses for `2x5120` and
     `4x5120`; exact `1x5120` allreduces were still replaced by the sidecar.
     Aggregated trace rows from the 16-token diagnostic were: `2048` capture
     misses for `8x5120`, `4096` for `2x5120`, `4723` for `4x5120`, `1024`
     runtime prefill misses for `2048x5120`, and only logits all-gather at
     runtime after prefill. This closes "mystery public NCCL" as a primary
     single-request decode target. Multi-row sidecar support may be useful for
     prefill/concurrency milestones, but the c1 decode gate still depends on
     reducing the hot `1x5120` sidecar path or the legal RowParallel schedule.
377. Multi-row non-resident sidecar support is now a promoted minor dense
     source milestone and the current standard-envelope high-water, but not a
     gate clear. The throwaway candidate
     `.tmp/sidecar_multirow_runtime_20260613/gfx906_sidecar_tree_ll_ar_runtime_multirow_20260613.cpp`
     keeps `count=5120` on the promoted `nWarps=4`/256-thread single-work
     kernel and routes `2/4/8 * 5120` rows through the older exact
     `RunWorkBatch` shape with `nWarps=2`/128 threads. The matching Python
     overlay widens only the sidecar eligibility test. Full strict ladders
     were valid on both hosts: `.20`
     `qwen36_27b_decode_tiers_rocm72_dense_sidecar_multirow_h20_host20_multirow_h20_20260613_full`
     reached `62.491/63.308/59.783` backend TPS for strict c1 / `c1_2000` /
     `c1_10000`, and `.30`
     `qwen36_27b_decode_tiers_rocm72_dense_sidecar_multirow_h30_host30_multirow_h30_20260613_full`
     reached `62.192/62.919/59.127`. Saved docker logs confirm the widened
     path executed during graph capture (`numel=40960` replacements:
     `.20` `486`, `.30` `484`). This is a small repeatable improvement over
     the prior `5120`-only sidecar ladders, not a new gate-scale mechanism.
378. `NCCL_NTHREADS` unset does not compose into the new multi-row sidecar
     winner. The paired interaction test kept the same multi-row sidecar
     runtime and changed only `NCCL_NTHREADS` to empty. `.20`
     `qwen36_27b_decode_tiers_rocm72_dense_sidecar_multirow_nthreads_unset_h20_host20_multirow_nthreads_unset_h20_20260613_full`
     reached `62.218/63.237/59.704`; `.30`
     `qwen36_27b_decode_tiers_rocm72_dense_sidecar_multirow_nthreads_unset_h30_host30_multirow_nthreads_unset_h30_20260613_full`
     reached `62.031/62.948/59.237`. That is mixed and lowers strict smoke on
     both hosts, so keep `NCCL_NTHREADS=128` in the dense sidecar envelope.
379. Profiling the multi-row sidecar high-water shows it is a useful widened
     capture milestone, not yet true launch coalescing. `.30`
     `qwen36_27b_sidecar_multirow_winner_rocprof_h30_20260613` was a valid
     rocprof run with `806,243` filtered kernel rows and `25.723s` filtered
     kernel time. The profile still shows `sidecar_tree_ll_1x5120_kernel` as
     the dominant row (`14.128s`, `54.92%`, `98,073` launches, `144.1 us`
     average) and no `sidecar_tree_ll_multiwork_*` rows, even though docker
     logs confirm `numel=40960` sidecar replacements. Public NCCL improved
     only slightly versus the prior `.30` sidecar profile (`2.292s`/`1,670`
     launches to `2.209s`/`1,663`). The next source target is to make the
     `2/4/8x5120` path emit a distinct multiwork kernel and actually reduce
     launch count; otherwise treat multi-row as a minor long-decode gain only.
380. The correctness-safe split-loop multi-row sidecar fallback is exact but
     not a serving promotion. C API count configurability in
     `microbenches/gfx906_persistent_ar_capi_smoke_20260612.cpp` exposed
     `PERSISTENT_CAPI_COUNT`, which made the broken raw `RunWorkBatch >1` and
     serial same-kernel multiwork candidates fail clearly: both were exact at
     `5120`, then timed out or left high-rank local values unchanged at
     `10240+`. The split-loop runtime
     `.tmp/sidecar_splitloop_multirow_runtime_20260613/gfx906_sidecar_tree_ll_ar_runtime_multirow_20260613.cpp`
     instead launches the promoted `sidecar_tree_ll_1x5120_kernel` once per
     slice and was exact on `.20` and `.30` for `5120`, `10240`, `20480`, and
     `40960`. Full serving ladders rejected it for promotion: `.30`
     `qwen36_27b_decode_tiers_rocm72_dense_sidecar_splitloop_multirow_h30_host30_splitloop_multirow_h30_20260613_full`
     reached `62.294/63.041/59.489`, and `.20`
     `qwen36_27b_decode_tiers_rocm72_dense_sidecar_splitloop_multirow_h20_host20_splitloop_multirow_h20_20260613_full`
     reached `62.015/62.997/59.498` backend TPS for strict c1 / `c1_2000` /
     `c1_10000`. The split-loop path is a safe fallback and a useful test
     harness, but it does not collapse launches and does not beat the current
     multi-row high-water of `.20` `62.491/63.308/59.783`.
381. Resident persistent AR idle-sleep plus post-prefill start remains
     rejected, and the follow-up exposed a concrete lifecycle ordering bug.
     Rebuilt the current runtime on `.20` with the deferred-start ABI at
     `/usr/share/ollama/kernel_labs/gfx906_persistent_ar_runtime_startabi_sleep_h20_20260613/libgfx906_persistent_tree_ll_ar_startabi_sleep_20260613.so`
     (`gfx906_persistent_ar_create/start/snapshot/allreduce_inplace/destroy`
     exported). The first strict smoke used `DEFER_WORKER=1`,
     `START_AFTER_PREFILL=1`, `IDLE_SLEEP_ITERS=16`, `nWarps=4`, and 256
     block threads; it completed valid c1 at `57.170` backend decode TPS.
     Logs showed `start_after_prefill started_deferred_workers=0`, so the
     hook fired before any deferred handles existed. A guarded source patch
     added `VLLM_GFX906_PERSISTENT_AR_START_ON_FIRST_USE=1` so the first
     eligible post-prefill `1x5120` replacement can create and start its own
     deferred handle. The retry was also valid but stayed at `57.249` backend
     decode TPS, with no `created handle` / `replaced allreduce` logs, meaning
     the current graph schedule did not invoke the in-place persistent
     replacement. This closes simple idle-sleep / first-use lifecycle patching
     as a gate path. Keep the patch as diagnostics, but do not promote resident
     workers until source work proves the replacement path is actually entered
     and improves strict c1 above the non-resident sidecar high-water.
382. Serial multi-row sidecar needed a serving-relevant correction: the earlier
     C API rejection was partly an unsafe default, not proof that the promoted
     serial primitive is invalid. Added `SIDECAR_INPLACE=1` to
     `microbenches/rccl_sidecar_tree_ll_work_probe_20260612.cpp` and threaded
     it through `run_rccl_sidecar_tree_ll_work_probe_20260612.sh`. The direct
     probe now proves in-place serial `nWarps=4`/256 is exact for count `2`
     and count `8` (`sidecar_serial2_inplace_h20_20260613`,
     `sidecar_serial8_inplace_h20_20260613`), and in-place raw batch
     `nWarps=2`/128 count `8` is also exact on `.30`
     (`sidecar_batch8_inplace_h30_20260613`). Rebuilt the serial runtime on
     `.20` and reran C API count `40960`: default multiwork `nWarps=2` still
     failed asymmetrically (rank `0` hung while ranks `1..7` returned local
     values), but pinning `VLLM_GFX906_PERSISTENT_AR_MULTIWORK_NWARPS=4` and
     `...BLOCK_THREADS=256` made the one-call C API exact. The full serving
     ladder with the corrected serial runtime was valid but rejected:
     `qwen36_27b_decode_tiers_rocm72_dense_sidecar_serial_multirow_nw4_revisit_h20_host20_serial_multirow_nw4_revisit_h20_20260613_full`
     reached only `58.096/58.835/55.785` backend TPS for strict c1 /
     `c1_2000` / `c1_10000`. This corrects the diagnosis but does not promote
     serial multiwork for serving; the current standard-envelope high-water
     remains the original `.20` widened-capture run at
     `62.491/63.308/59.783`.
383. The dense serving sidecar gap is dominated by tail latency, not uniform
     kernel compute. Reparsed the valid `.30` multi-row winner profile
     `qwen36_27b_sidecar_multirow_winner_rocprof_h30_20260613` with exact
     kernel-name matching and positive-duration filtering. The cleaned
     `sidecar_tree_ll_1x5120_kernel` rows show `98,077` launches,
     `14.005s` total, mean `142.8 us`, median `45.8 us`, p95 `179.2 us`,
     p99 `3.64 ms`, and max `83.3 ms`. Per-second bins keep the median around
     `45-48 us` throughout the request, while p99 stays multi-ms across
     decode; the tail is not just cold start. Public `ncclDevKernel_Generic_4`
     rows are much worse in the tail (`p50 154.9 us`, p95 `5.57 ms`), but the
     sidecar still loses most of its theoretical microbench floor to rank
     skew/barrier wait. Next serious source work should target tail
     flattening, legal scheduling/overlap, or producer/collective/consumer
     placement; another sub-microsecond Tree/LL primitive tweak is unlikely to
     move the dense gate.
384. `GPU_MAX_HW_QUEUES=1` does not fix the sidecar tail in serving. The
     hypothesis was that limiting ROCm hardware queues might reduce rank
     arrival skew at the collective boundary. Reused the current `.20`
     multi-row sidecar high-water runtime and added only
     `GPU_MAX_HW_QUEUES=1` through `EXTRA_DOCKER_ENV_ARGS_APPEND`, then ran
     the full strict decode ladder in
     `qwen36_27b_decode_tiers_rocm72_dense_sidecar_multirow_gpu_hw_queues1_h20_host20_multirow_gpu_hw_queues1_h20_20260613_full`.
     Results were `57.709/58.724/55.657` backend TPS for strict c1 /
     `c1_2000` / `c1_10000`, well below the current standard-envelope
     high-water of `.20` `62.491/63.308/59.783`. The lane was cleanly torn
     down and `.20` returned to zero VRAM. Queue-count clamping is rejected as
     a dense decode promotion path; keep the result as evidence that the tail
     problem likely needs source-level scheduling/placement changes rather
     than a simple ROCm queue knob.
385. Cross-stream fork/join inside the sidecar runtime is not graph-capture
     safe enough to pursue as the next tail fix. Built
     `.tmp/sidecar_commstream_runtime_20260613`, adding opt-in
     `VLLM_GFX906_PERSISTENT_AR_COMM_STREAM` to record a ready event on the
     captured graph stream, run the sidecar allreduce on a separate
     nonblocking/high-priority HIP stream, then join back with a done event.
     Small-shape C API correctness passed, but direct keep-handle timing
     regressed from current-stream `0.1018 ms/call` to high-priority
     `0.2295 ms/call`; normal-priority stream was far worse at
     `10.756 ms/call`. The serving run
     `qwen36_27b_decode_tiers_rocm72_dense_sidecar_commstream_prio_h20_host20_commstream_prio_h20_20260613_full`
     never reached readiness: it hung in PIECEWISE CUDA graph capture after
     logging `40960` sidecar replacement attempts, with GPUs pegged at 100%
     and no memory activity. The lane was killed after the extended wait and
     `.20` returned to zero VRAM. Reject stream fork/join as a shortcut; the
     next source path needs a graph-native schedule change or a lower-level
     collective rewrite, not external HIP stream/event choreography inside the
     custom op.
386. RCCL `loadWorkBatchToShmem` single-work fastload is rejected for c1
     decode. Patch
     `patches/rccl_721_gfx906_tree_ll_singlework_fastload_after_20260613.patch`
     added a guarded shortcut for the common public RCCL single-work batch:
     coll work type, `funcId=0`, `offsetBitset=1`, and no extension batches.
     The full `.20` overlay build was still in AMDGPU `lld` after about
     `81` minutes, so a reduced `.30` build with
     `ONLY_FUNCS="AllReduce TREE LL Sum f16"` was used for direct replay.
     Build and symbol checks passed, and the result was exact, but the
     c1-critical hot row regressed: `.30`
     `rccl_direct_allreduce_graph_bench_host30_fastload_min_h30_direct_20260613`
     measured graph-out `1x5120 0.044682 ms`, `2x5120 0.051114 ms`,
     `4x5120 0.063254 ms`, `8x5120 0.088971 ms`, `16x5120 0.144615 ms`,
     `20x5120 0.172032 ms`, and `32x5120 0.257113 ms`. The wider-row gains
     do not matter for the single-request dense decode gate because the hot
     `1x5120` row is roughly `45%` slower than the current
     singlework/LL16/prims-inline baseline family. The full `.20` build was
     stopped after this direct rejection. Do not promote the fastload patch for
     dense c1; keep it only as evidence that bypassing generic batch loading
     is not a free win for the hot RCCL path.
387. Separate rank-stagger delay kernels are C API-correct but not serving
     graph-capture live. Built
     `.tmp/sidecar_stagger_runtime_20260613`, adding
     `VLLM_GFX906_PERSISTENT_AR_STAGGER_ITERS` as a per-device delay list
     before the sidecar allreduce. Zero-delay C API smoke on `.20` was exact
     at mean `0.077210 ms/call`; uniform `1000` iterations was exact at mean
     `0.121354 ms/call`, so `1000` iterations costs about `44 us/call`.
     Serving rejected the design before any c1 result. The `.20` calibrated
     list `1081,229,697,0,611,814,995,475` and the `.30` tiny list
     `80,20,50,0,45,60,75,35` both reached PIECEWISE CUDA graph capture,
     logged `40960` sidecar replacement attempts, then stopped making progress
     with all GPUs pegged and repeated shared-memory broadcast warnings. Run
     evidence is in
     `qwen36_27b_decode_tiers_rocm72_dense_sidecar_stagger_rankmean_h20_host20_stagger_rankmean_h20_20260613`
     and
     `qwen36_27b_decode_tiers_rocm72_dense_sidecar_stagger_rough_h30_host30_stagger_rough_h30_20260613`.
     Reject a separate delay kernel/node as a tail-flattening strategy. If
     stagger is revisited, keep it graph-native inside the existing sidecar
     kernel or patch lower-level scheduling without adding an extra captured
     kernel before the collective.
388. Graph-native in-kernel sidecar rank staggering is also rejected for
     serving liveness. Built `.tmp/sidecar_instagger_runtime_20260613`, moving
     the per-device `s_sleep` delay into the existing
     `sidecar_tree_ll_1x5120_kernel` and `sidecar_tree_ll_multiwork_1x5120_kernel`
     before `RunWorkColl` / `RunWorkBatch` instead of launching a separate
     delay kernel. `.20` C API smokes were exact: zero-delay mean
     `0.0792557875 ms/call`, and the rough stagger list
     `80,20,50,0,45,60,75,35` mean `0.076963375 ms/call`. Serving run
     `qwen36_27b_decode_tiers_rocm72_dense_sidecar_instagger_rough_h20_host20_instagger_rough_h20_20260613`
     loaded, compiled, entered PIECEWISE CUDA graph capture, created handles on
     all eight ranks, and logged `numel=40960` replacement attempts, but never
     opened the backend port. After roughly fifteen minutes it was still
     emitting only shared-memory broadcast warnings with all GPUs busy and
     `94-95%` VRAM allocated. This closes simple delay-based rank staggering:
     the capture hang is not only caused by an extra captured delay node.
389. Offset channel partitioning is exact but not a serving promotion. Built
     `.tmp/sidecar_channel_offset_candidate_20260613`, adding opt-in
     `SIDECAR_CHANNEL_OFFSET` for the direct work-loop probe and
     `VLLM_GFX906_PERSISTENT_AR_CHANNEL_OFFSET` for the non-resident sidecar
     runtime. The hypothesis was to keep three active Tree/LL channels but use
     channels `1,2,3` instead of `0,1,2`, allowing the hot sidecar row to touch
     the fourth balanced tree root without increasing channel count. Under the
     promoted balanced `RCCL_TREES`, exact replay improved only slightly:
     `.20` offset `0` was `0.0295653 ms/call` and offset `1` was
     `0.0293985`; `.30` offset `0` was `0.0294441` and offset `1` was
     `0.0293987`. Strict serving smokes with the multi-row sidecar runtime did
     not promote: `.20` reached `62.428` backend TPS versus the same-host
     high-water `62.491`, and `.30` reached `62.183` versus `62.192`.
     Keep channel offset closed as a useful negative result. Tiny median
     microbench improvements are not enough; the next source step still needs
     to reduce the hot `1x5120` sidecar launch/count cost or change the
     collective primitive/tail behavior.
390. Skinny shared-memory loading inside the sidecar runtime is an exact
     microbench improvement but not a serving promotion. Built
     `.tmp/sidecar_skinny_shmem_candidate_20260613`, adding opt-in
     `SIDECAR_SKINNY_SHMEM` to the direct work-loop probe and
     `VLLM_GFX906_PERSISTENT_AR_SKINNY_SHMEM` to the multi-row sidecar
     runtime. The patch avoids copying the full `ncclDevComm` and
     `ncclDevChannel` structs into `ncclShmem` for the Tree/LL sidecar path,
     instead setting the fields actually read by the LL primitive: rank/node
     metadata, LL buffer size, abort flag, peer table pointer, tree, and
     work counter. Balanced direct graph replay was exact and improved the hot
     row from `.20` `0.0296748 -> 0.0295084 ms/call` and `.30`
     `0.0296938 -> 0.0295101`. C API in-place smokes were exact on both
     hosts. Serving did not promote: strict c1 reached `.20` `62.362`
     backend TPS versus the high-water `62.491`, and `.30` `62.238` versus
     `62.192`. This is a useful source milestone because it proves the full
     comm/channel copy has measurable direct cost, but the gain is still
     below serving variance and does not address the p99 tail enough to clear
     the `65 TPS` gate.
391. Removing the sidecar work-descriptor zero-fill stacks with skinny shared
     memory in direct replay, but still does not promote in serving. Built
     `.tmp/sidecar_skinny_shmem_candidate_20260613` no-zero variants of the
     direct probe and multi-row runtime. The patch keeps explicit descriptor
     field assignments but skips the normal `ncclDevWorkColl` byte-zero loop
     in the sidecar path. Balanced direct graph replay was exact and improved
     to `.20` `0.0292063 ms/call` and `.30` `0.0292124`, roughly another
     `1%` better than skinny-only and about `1.6%` better than the same
     candidate baseline. C API in-place smokes were exact on both hosts.
     Strict serving smokes were valid but below the high-water: `.20`
     `62.183` backend TPS versus `62.491`, and `.30` `62.172` versus
     `62.192`. Reject this as a serving promotion. It is still useful source
     evidence: descriptor initialization overhead is measurable in isolation,
     but we are now below the threshold where tiny median-sidecar wins move
     the dense `65 TPS` gate.
392. Increasing the dense sidecar serving envelope to `--max-num-seqs 8` is
     rejected as a promotion. The test was motivated by the profile/log gap:
     the original multi-row winner logged `40960` sidecar replacements during
     mixed/piecewise graph capture, but the valid profile still showed the real
     single-request decode hot path dominated by `sidecar_tree_ll_1x5120_kernel`
     rather than `sidecar_tree_ll_multiwork_*`. Raising `MAX_NUM_SEQS` from
     `4` to `8` did change graph profiling from `FULL=3 (largest=4)` to
     `FULL=4 (largest=8)` while preserving strict uncapped thinking-gate
     validity, but serving did not improve: `.20` reached `62.256` backend TPS
     versus the same-host high-water `62.491`, and `.30` reached `61.796`
     versus `62.192`. This closes simple graph-envelope widening as the next
     dense decode lever. The remaining gate path is still the hot `1x5120`
     sidecar tail/launch/count behavior, not just capture-size eligibility.
393. Sidecar descriptor flags are not a portable dense decode lever. Built
     `.tmp/sidecar_descriptor_flags_candidate_20260613`, adding opt-in direct
     probe and runtime controls for the fixed sidecar `ncclDevWorkColl`
     fields `rcclUseOneSlice` and `gfx9CheapFenceOff`. All direct work-loop
     variants were exact under the promoted balanced tree, `nWarps=4`, and
     256-thread sidecar shape. `.20` improved only marginally versus baseline
     `0.029676425 ms/call`: one-slice `0.029630200`, cheap-fence-off
     `0.029591338`, and both `0.029592262`. `.30` did not reproduce the win:
     baseline was `0.029613550`, one-slice `0.029620125`, cheap-fence-off
     `0.029659450`, and both `0.029671525`. Reject this branch before serving.
     The result closes another descriptor-level cleanup path and reinforces
     that the dense `65 TPS` gap is not likely to be solved by sub-percent
     median-sidecar tweaks unless they also reduce launch count or p99 tail.
394. Combining skinny shared-memory, no-zero descriptor initialization, and
     channel offset is a direct-replay source milestone but not a serving
     breakthrough. Built
     `.tmp/sidecar_skinny_nozero_chanoffset_candidate_20260613`, keeping the
     hot `1x5120` sidecar path on skinny shared-memory and explicit descriptor
     assignment while allowing `SIDECAR_CHANNEL_OFFSET=1` /
     `VLLM_GFX906_PERSISTENT_AR_CHANNEL_OFFSET=1`. Balanced direct replay
     stacked the prior independent wins: offset `0` measured `.20`
     `0.0292045 ms/call` and `.30` `0.0291916-0.0291917`, while offset `1`
     improved to `.20` `0.0291214` and `.30` `0.0290683`. Runtime C API
     `5120` smokes were exact on both hosts with offset `1` and skinny shmem,
     but the strict serving smoke did not beat the durable high-water: `.20`
     reached `62.338` backend TPS versus `62.491`, and `.30` reached
     `62.358` versus `62.192`. Reject as a serving promotion. The useful
     lesson is that descriptor-side median wins continue to translate poorly
     unless the real decode graph sees fewer launches, a lower p99 sidecar
     tail, or a different collective schedule.
395. Sidecar p99 tail context now has two-host evidence and points at the
     producer/collective boundary rather than local launch overhead. Added
     `analyze_sidecar_tail_context_20260613.py` and ran it on the valid `.30`
     multi-row winner profile plus the `.20` sidecar winner profile. On `.30`,
     p99 sidecar rows were almost always locally ordered as
     `LLGemm1_kernel -> sidecar_tree_ll_1x5120_kernel -> RMS`, with `964/982`
     outliers immediately following `LLGemm1_kernel`, zero same-GPU cross-queue
     overlaps, and local previous-kernel duration not materially worse than
     normal rows (`p99 33.470 us` for outliers versus `34.400 us` for normal
     `<=p95`). `.20` reproduced the same pattern: `964/982` outliers
     immediately followed `LLGemm1_kernel`, zero same-GPU cross-queue overlap,
     and previous-kernel duration remained normal (`p99 33.760 us` versus
     `34.720 us`). This strengthens the rank-arrival interpretation: a rank
     reporting a long sidecar is usually not locally delayed by its own GEMM;
     it is likely waiting inside the collective for another rank's producer or
     schedule path. Delay/stagger branches are already serving-rejected, so the
     next credible gate-scale source work should change producer/collective
     dataflow, legal collective count, or the small-message collective itself,
     not shave descriptor setup or add artificial waits.
396. Channel-slice producer readiness does not rescue fused
     `LLMM1 + sidecar` as implemented. The existing boundary harness already
     has `BOUNDARY_FUSED_CHANNEL_READY=1`, where each sidecar channel waits
     only for its own producer row slice rather than all producer blocks. The
     `.20` exact run
     `rccl_llmm1_ar_rms_boundary_host20_20260613_channel_ready_h20_channel_ready_h20`
     still regressed: attention `LLMM1+AR` averaged `0.052666 ms` while fused
     channel-ready took `0.065661`, and MLP-down averaged `0.069239` while
     fused channel-ready took `0.093162`. The `.30` run
     `rccl_llmm1_ar_rms_boundary_host30_20260613_channel_ready_h30_channel_ready_h30`
     reproduced the MLP-down rejection (`0.069257 -> 0.093228`); its attention
     public baseline was noisy, but fused channel-ready remained slower than
     the same-run sidecar AR path (`0.065718` versus `0.052316`). This closes
     simple channel-level producer/collective fusion. A future fusion would
     need true chunk/LL-line-level overlap or a different dataflow, not just a
     coarser per-channel readiness counter before `RunWorkColl`.
397. Odd-root balanced `RCCL_TREES` rotations are exact but not a gate-scale
     sidecar lever. Ran bounded direct replay sweeps on `.20` and `.30` with
     rank-rotated versions of the promoted balanced tree string at offsets
     `1`, `3`, `5`, and `7`, preserving the current `nWarps=4` /
     256-thread sidecar descriptor. All variants were exact on both hosts. The
     mean `sidecar_graph_ms_per_call` results were `.20` offset
     `1/3/5/7 = 0.029592/0.029513/0.029568/0.029502` and `.30` offset
     `1/3/5/7 = 0.029562/0.029531/0.029576/0.096919`; the `.30` offset-7
     direct metric is an anomaly because its count-1 batch path remained
     normal at `0.029756 ms/work`. The only portable direct signal is a tiny
     sub-microsecond shift, below the size of previously serving-rejected
     channel-offset and descriptor-cleanup wins. Do not spend serving lanes on
     odd-root tree rotations unless a later profile shows a topology-specific
     tail failure. Keep the promoted balanced tree string and continue toward
     structural RCCL work: lower launch/count cost, a real multi-descriptor
     path, or deeper producer/collective dataflow.
398. Reusable sidecar descriptor serial multiwork is exact and is the latest
     source microbench milestone, but not yet a serving breakthrough. Added
     `SIDECAR_MULTIWORK_MODE=reuse`, which preinitializes one fixed
     `ncclDevWorkColl` descriptor, updates only the live buffer pointers and
     `opCount`, and executes 128 dependent `1x5120` Tree/LL works through the
     same graph-captured sidecar path. It completed exactly on both primary
     hosts with the promoted balanced tree and `nWarps=4` / 256-thread launch:
     `.20` measured `0.0276987 ms/work` and `.30` measured
     `0.0276088 ms/work`, versus same-run stock RCCL graph calls around
     `0.03071 ms/call`. This confirms the previous serial reset result was
     not a one-off and that descriptor setup can be amortized safely when the
     work sequence is explicitly serial. Do not promote directly to serving
     yet: the dense decode graph still lacks a legal independent multiwork
     collapse at the hot RowParallel boundaries, and prior direct-replay
     median wins failed to clear the p99 tail. The value is source evidence for
     a future lower-level RCCL path that can carry a reusable work descriptor
     across real decode collectives without changing graph semantics.
399. Serial-reuse multi-row sidecar is exact and fixes the prior serial retest
     methodology, but still does not beat the current serving winner. The older
     `.20` serial multi-row serving rejection is not valid evidence for the
     intended multi-row path because its settings mounted the base
     `gfx906_persistent_ar_20260612` Python overlay, which accepts only
     `numel=5120`, and left `rccl_trees=` empty instead of using the promoted
     balanced tree. Built a corrected candidate at
     `.tmp/sidecar_serial_reuse_runtime_20260613`, replacing the multi-row
     `RunWorkBatch` kernel with the reusable serial descriptor path. Also
     patched `run_gfx906_persistent_ar_capi_smoke_20260612.sh` to pass and log
     `RCCL_TREES` safely so the C API evidence matches serving. C API
     `count=40960`, `nWarps=4`, 256-thread, balanced-tree smokes were exact on
     `.20` and `.30`: `.20` averaged `0.267425 ms/call`, `.30` averaged
     `0.271618 ms/call`, a real improvement over the split-loop C API around
     `0.434 ms` and the prior one-call serial-reset smoke around `1.27 ms`.
     Full serving ladders with the correct multi-row Python overlay and
     balanced tree were valid but below high-water: `.20`
     `62.013/62.858/59.392` and `.30` `62.106/62.893/59.241` backend TPS for
     strict c1 / c1_2000 / c1_10000. Reject for promotion. The useful result
     is that descriptor reuse is correct and portable, but serializing the
     multi-row work inside the sidecar kernel does not fix the real serving
     tail or clear the `65 TPS` gate.
400. Sidecar p99 stalls are producer-shape concentrated, not modulo-position
     concentrated. Added `analyze_sidecar_position_shape_20260613.py` and ran it
     on the current `.20` and `.30` multi-row sidecar winner profiles. The
     report uses only local per-GPU sequence ordering, exact
     `sidecar_tree_ll_1x5120_kernel` rows, and adjacent kernel shapes, avoiding
     cross-GPU timestamp alignment. On `.30`
     `sidecar_position_shape_analysis_20260613.md` loaded `806813` rows with
     `98078` exact sidecar rows; sidecar p50/p95/p99 was
     `45.9/179.2/3643.6 us`. On `.20` it loaded `807227` rows with `98066`
     exact sidecar rows; sidecar p50/p95/p99 was `42.2/183.2/3051.0 us`.
     Across both hosts, the same two previous-kernel shapes account for almost
     all tail rows: `LLGemm1_kernel grid=327680 wg=128` and
     `LLGemm1_kernel grid=819200 wg=320`. They account for `.30`
     `4847/4897` p95 rows and `961/981` p99 rows, and `.20` `4850/4899` p95
     rows and `962/981` p99 rows. Pair analysis maps the highest p99 group to
     attention first: `LLGemm1 grid=327680` followed by the interleaved RMS
     consumer has `.30` `597` and `.20` `577` p99 rows. The MLP/down
     `LLGemm1 grid=819200` tail is real but split across several RMS consumer
     variants (`143/102/36/23` p99 rows on `.30`, `153/127/39/17/16` on
     `.20`). Local sequence modulo-16 positions are much less decisive; p99
     rows are spread across the positions rather than collapsing onto one slot.
     This promotes a stronger source direction: target the real attention
     `LLGemm1 -> sidecar allreduce -> interleaved RMS` boundary first, then
     MLP/down, instead of doing more generic descriptor-init or arbitrary
     position-stagger work.
401. The current-graph attention post-IR matcher is repaired, but attention
     split-endpoint RMS replacement is not a serving win. The old attention
     matcher expected a `setitem` buffer path and selected `0/0` boundaries on
     the current mutable RowParallel graph. Patched
     `overlays/postir_split_endpoint_rms_20260611/gfx906_post_ir_ar_rms_fusion.py`
     to classify RowParallel producers by local K dimension (`K=768` attention,
     `K=2176` MLP/down) and to match the immediate
     `float -> residual add -> RMS` consumer chain for attention. Retests on
     `.20` and `.30` selected `64/64` attention boundaries on every rank and
     completed strict thinking-gate smokes, proving the matcher repair. Serving
     speed still rejected the replacement: `.20` strict c1 was `55.585` backend
     TPS and `.30` was `53.449`, well below the multi-row sidecar high-water.
     Keep the classifier as source infrastructure, but do not pursue more
     consumer-only split-endpoint placement as the next gate path.
402. Sidecar p99 tails are pair-specific but not locally producer-duration
     driven. Added `analyze_sidecar_pair_gpu_skew_20260613.py` and ran it on
     the `.20` and `.30` multi-row sidecar winner profiles. The main attention
     pair `LLGemm1 grid=327680 -> interleaved RMS vgpr52` has sidecar p99
     `5220.869 us` on `.30` and `4629.755 us` on `.20`, while the immediately
     preceding producer p95 is only about `13-14 us`. Among pair-p99 attention
     rows, only `.30` `6/356` (`1.7%`) and `.20` `9/356` (`2.5%`) followed a
     producer above that pair's producer p95; local previous/next launch gaps
     stayed around `9-10 us`. MLP/down RMS2 tails show the same direction, with
     `.30` only `2/62` (`3.2%`) and `.20` `9/59` (`15.3%`) following slow
     producers. This promotes a sharper diagnosis: the hot tail is dominated by
     sidecar collective wait/serialization or rank-arrival behavior, not
     `LLGemm1` compute. Next dense source work should target sidecar
     rank/topology behavior or a lower-level graph-native collective boundary,
     not standalone producer GEMM or consumer-only RMS rewrites.
403. Basic rank-to-GPU reordering is valid but not a dense decode promotion.
     Tested the current multi-row sidecar high-water stack with only
     `HIP_VISIBLE_DEVICES` order changed, keeping O3 async, the explicit
     balanced `RCCL_TREES`, Tree/LL, four channels, and `NCCL_NTHREADS=128`.
     `.20` reverse `7,6,5,4,3,2,1,0` reached `62.135` strict c1 backend TPS;
     `.20` even/odd `0,2,4,6,1,3,5,7` reached `62.436`; `.30` even/odd
     reached `61.210`; `.30` reverse reached `61.312`. The `.20` even/odd
     result is a near-tie but still below the same-host high-water `62.491`,
     and the `.30` variants clearly regress versus `.30` high-water `62.192`.
     Do not spend full ladders on basic process/device reordering unless a
     smoke beats same-host high-water. The remaining rank/topology path must
     alter the sidecar/RCCL schedule or lower-level graph-native boundary, not
     just the physical GPU order.
404. Odd-root balanced Tree/LL scheduling is valid but regresses the dense
     sidecar stack. Tested a rank-shifted `RCCL_TREES` string with roots
     `1/3/5/7` instead of the promoted even-root `0/2/4/6`, leaving GPU order,
     O3 async, and the multi-row sidecar runtime unchanged. `.20` reached
     `62.000` strict c1 backend TPS and `.30` reached `61.408`, both below
     same-host high-water (`62.491` and `62.192`). The `.20` first attempt also
     exposed lane hygiene debt: stale decode-tier `runtime_*` caches had filled
     `/usr/share/ollama`; cleaning `67` stale runtime directories through a
     root Docker cleanup container freed about `90G`. The easy topology surface
     is now closed: process/device order and simple balanced-tree root shifts
     do not clear the sidecar tail. Continue with source-level collective
     schedule, launch-count, or graph-native boundary work.
405. Multi-row sidecar capture is real but not the single-request c1
     launch-count fix. The `numel=40960` replacement logs line up with the
     size-8 CUDA graph capture path, while strict c1 decode replays the size-1
     graph and still profiles as roughly `98k`
     `sidecar_tree_ll_1x5120_kernel` rows with no useful
     `sidecar_tree_ll_multiwork_*` rows. Keep widened capture as a
     prefill/multi-request and mild high-water milestone, but put primary dense
     decode source effort into the hot `1x5120` collective primitive,
     rank-arrival tail, or lower-level graph-native boundary.
406. Sidecar descriptor `cbd` split tuning is exact but below serving scale.
     Added compile-time `SIDECAR_COUNT_*` and `SIDECAR_CHUNK_GRAINS_*` macros
     to `rccl_sidecar_tree_ll_work_probe_20260612.cpp`, then reran corrected
     sequential sweeps on `.20` and `.30` with one 8-GPU job per host at a
     time. The best split, `countLo/countMid/countHi=2048/2048/1024`, moved
     graph-mode sidecar from `.20` `0.043890` to `0.043853 ms/call` and `.30`
     `0.043945` to `0.043868 ms/call`. That is exact but only tens of
     nanoseconds per call, below serving promotion scale. Do not build a
     runtime from descriptor partitioning unless a later profile shows a much
     larger tail-specific effect.
407. Cross-rank sidecar sequence analysis needs explicit sequence IDs before
     it can drive a source promotion. Added
     `analyze_sidecar_crossrank_sequence_20260613.py` and
     `analyze_sidecar_crossrank_laggard_20260613.py`, with guards that mark
     occurrence-index alignment as exploratory when per-GPU sidecar row counts
     differ by more than `1%`. The existing `.20`/`.30` winner profiles fail
     that guard badly: per-GPU sidecar count spreads are `16912`, `15180`, and
     `11429` rows. Therefore the current combined rocprof CSVs cannot prove
     one-rank versus multi-rank tail ownership. The useful milestone is that
     the analysis tooling now prevents this unsafe inference and points the
     next source step at graph-safe sequence/arrival instrumentation around the
     hot sidecar allreduce.
408. Sidecar `op_count` tracing is now a usable diagnostic substrate, but the
     first root-reorder mitigation is rejected. Added optional
     `GFX906_SIDECAR_TRACE` support to
     `gfx906_sidecar_tree_ll_ar_runtime_20260612.cpp`, dumping per-rank CSVs
     with `channel`, RCCL `op_count`, and device `clock64()` duration. The
     `.20` capped diagnostic produced `8` equal files with `196,992` entries
     each and zero overflow; `analyze_sidecar_opcount_trace_20260613.py`
     grouped `196,992` complete `(channel, op_count)` cohorts. In the
     `op_count >= 10000` steady cut, device-local p99 multiplicity was
     one-device `5557` versus multi-device `3134`, so local/rank-arrival
     sensitivity remains plausible. Reordering the four-tree string from
     `0/2/4/6` to `0/2/6/4` so the sidecar uses the root-6 tree was valid but
     did not promote: `.20` strict c1 `61.955` and `.30` `61.696` backend TPS,
     both below same-host high-water.
409. Kind-routed sidecar selection is graph-capture viable, but the
     `1792/1664/1664` direct-median winner is not a per-family serving win.
     Added `all_reduce_inplace_kind` and route-aware sidecar library loading so
     attention and MLP RowParallel calls can use different `.so` files while the
     non-target family keeps the promoted `2048/1536/1536` default. O3 strict
     serving was valid on `.20`/`.30`, but attention-only routing reached only
     `.20 62.362` and `.30 61.887`, and MLP-only routing reached `.20 62.419`
     and `.30 61.995`, all below same-host high-water. Keep the infrastructure;
     reject this alternate split.
410. Multi-row sidecar tracing proves the current dense decode high-water is
     still a one-row steady-state path. Added `GFX906_SIDECAR_TRACE` to the
     promoted `2048/1536/1536` multi-row sidecar runtime and reran capped
     strict diagnostics on `.20`/`.30` with graceful container stop so the
     destructor could dump per-rank CSVs. Both hosts produced `1,594,368`
     loaded entries with zero incomplete full-trace groups; the clean
     `op_count >= 10000` steady cut had `1,429,544` entries and `178,693`
     complete `(channel, op_count, work_count)` groups. All steady gate-region
     launches were `work_count=1`; `work_count=8/4/2` only appeared in the
     early graph/warmup op-count ranges. This keeps the `.20`
     `62.806/63.807/60.209` ladder as a meaningful pre-release high-water, but it
     was later superseded by the 2026-06-20 ROCm7.2 Dense/MoE validation. At this
     pre-release point, source work still needed to make the
     strict c1 graph issue fewer or fused steady allreduces, not merely widen a
     runtime path that the size-1 decode graph does not call.
411. Launch-bounds specialization is a useful compiler-resource diagnostic,
     not a dense serving promotion. Added
     `SIDECAR_LAUNCH_BOUNDS_THREADS` and
     `SIDECAR_LAUNCH_BOUNDS_MIN_BLOCKS` to the promoted multi-row sidecar
     runtime and built `256,1` plus `256,2` variants on `.20/.30`. The
     `256,2` C API smoke was exact and looked strong on `.20`
     (`0.059460 ms/call` versus `0.074258` baseline), but it regressed the
     `.30` component path (`0.072244` versus `0.062261`) and did not promote
     in strict serving: `.20` reached `62.524` backend TPS versus the
     `.20` high-water `62.806`, while `.30` reached `60.733` versus
     `62.192`. Keep the macros for lab builds, but do not trust isolated C API
     compiler wins until they reproduce in real O3 vLLM serving.
412. Captured sidecar call-site tracing is a real diagnostic breakthrough.
     Passing a host-side `launch_sequence` into the sidecar kernels gives each
     CUDA-graph-captured allreduce node a stable `call_site` label. Trace
     smokes on `.20/.30` stayed strict-valid and produced `128` steady
     callsites in the compact range `1664-1791`, matching the expected two
     RowParallel allreduces per dense layer. The `.20` p99 rows were strongly
     concentrated in `(call_site - 1664) % 4 == 0` (`20,435` p99 rows versus
     `5,773` for mod2 and near-zero for mod1/mod3). `.30` was more mixed but
     still mod0-led. This replaces unsafe rocprof occurrence-index inference
     with a graph-node label that can drive targeted source work.
413. Call-site-targeted descriptor routing is not enough. Added a compile-time
     route that keeps the promoted `2048/1536/1536` split globally but switches
     only `(call_site - 1664) % 4 == 0` to the direct-replay-faster
     `1792/1664/1664` split. Strict smokes were valid, and `.30` had one
     near-term c1 smoke at `62.321`, but the full ladder rejected the idea:
     `.20` `62.594/63.622/60.015` and `.30` `61.935/62.915/59.200` backend
     TPS for strict c1 / `c1_2000` / `c1_10000`. The useful result is the
     trace substrate, not the descriptor route. The next source path should map
     those 128 callsites back to layer/family and change the boundary or
     primitive, not just the count split.
414. The hot dense sidecar call-site class maps to attention output allreduce,
     not MLP. Added Python-side kind mapping to the graph-safe kind-route
     overlay and a runner append hook for diagnostic mounts. A strict `.20`
     mapping smoke was valid (`62.785` backend TPS) and wrote eight per-rank
     `callsite_kind` CSVs. Joining those rows to the `.20` call-site trace
     shows the steady range `1664-1791` alternates exactly `attn, mlp` for 64
     graph layers: `graph_layer=(call_site-1664)//2`, even call sites are
     attention, odd call sites are MLP. The top 12 tail sites by local/global
     p99 concentration are all even-layer attention output allreduces
     (`1664`, `1668`, `1692`, `1772`, `1740`, `1716`, `1700`, `1732`,
     `1724`, `1788`, `1676`, `1756`). This changes the next serious source
     target from generic descriptor tuning to attention RowParallel
     arrival/scheduling or graph-native collective boundary work.
415. Single-call-site primitive bypass is the latest narrow `.20` high-water,
     but only as a pre-2026-06-20 narrow high-water. Added guarded sidecar bypass controls to the
     kind-route overlay so selected captured nodes can return `False` and use
     the existing PyNccl/RCCL public allreduce path while preserving the global
     launch sequence. Broad even-attention bypass is rejected (`.20` strict c1
     `61.781`). Bypassing only `kind=attn, call_site=1664` is valid and
     improves the `.20` standard ladder from `62.806/63.807/60.209` to
     `63.084/63.993/60.380` backend TPS for strict c1 / `c1_2000` /
     `c1_10000`. `.30` reproduces as support but not a strong promotion:
     `62.197/63.034/59.525`. The no-trace `.20` repeat supports the promotion
     but does not beat it (`63.070/63.883/60.296`). Added exact-list controls
     with `VLLM_GFX906_PERSISTENT_AR_BYPASS_CALLSITE_LIST`; sparse follow-ups
     reject extending the public fallback set: `.20` `1664,1668` smoke
     `62.909`, `.20` `1668` smoke `62.723`, `.30` top-4 smoke `61.948`, and
     `.30` `1692` smoke `61.781`. The second follow-up batch closed the
     remaining obvious host-local singleton gaps: `.20` `1692` full ladder
     reached `62.909/63.788/60.169`, `.20` `1664,1692` smoke reached
     `62.997`, `.30` host-local `1672` smoke reached `61.913`, `.30`
     `1664,1672` reached `61.320`, and `.30` host-local `1688` reached
     `61.334`. The new lesson is precise: one pathological captured attention
     node can be moved without breaking O3 graph capture, but public RCCL is
     too expensive for broad, sparse top-N, or host-local singleton expansion.
     Continue with sidecar/RCCL-boundary work on the worst attention node; this
     pre-release branch still had an open dense status before the later
     2026-06-20 ROCm7.2 Dense/MoE validation.
416. Exact alternate-descriptor routing does not explain the single-node
     `1664` win. Built exact `1792/1664/1664` sidecar variants for selected
     captured call sites and strict-smoked them after the public-bypass
     promotion. `.20` exact `1664` reached only `62.792`, `.30` exact `1664`
     reached `62.031`, and `.30` host-local p99 leader `1672` reached
     `62.132`. A hybrid `.20` route with public fallback for `1664` plus
     alternate descriptor for neighbor `1668` reached `62.604`. This closes
     the current descriptor-route combination branch: the latest gain comes
     from avoiding the sidecar path for one pathological node, not from the
     `1792/1664/1664` split. Keep the exact call-site machinery, but put source
     effort into the sidecar/RCCL boundary or a genuinely faster hot attention
     collective primitive.
417. The public-1664 winner profile closes the next public-permutation branch
     and points back into the sidecar/RCCL boundary. A valid `.20` rocprofv2
     profile of the current public-1664 path still spends `10.643s` /
     `47.74%` of filtered kernel time in `sidecar_tree_ll_1x5120_kernel`,
     with public NCCL now visible at `2.493s` / `11.18%`. Compared to the
     prior all-sidecar profile, public-1664 trades roughly `-1.84s` sidecar
     time for `+0.48s` public NCCL time, which explains the small clean
     serving gain but also shows why broad fallback loses. The trace-enabled
     public-1664 diagnostic loaded `5.44M` steady sidecar rows and showed
     strong device-local skew: devices `6/7` own most raw max-duration groups
     and have p99 ticks around `70k` while devices `0-5` are around `50-52k`.
     Most device-local p99 groups are one-device events (`25,168` one-device
     versus `11,375` multi-device). The next primary source path is therefore
     sidecar rank-arrival/tail mitigation or a graph-native attention
     RowParallel boundary change, not another descriptor split or sparse
     public-fallback sweep.
418. Skinny/no-zero channel-offset does not compose with public1664 in
     serving. Rechecked the correct multi-row runtime
     `libgfx906_sidecar_tree_ll_ar_multirow_skinny_nozero_chanoffset_20260613.so`
     with `VLLM_GFX906_PERSISTENT_AR_CHANNEL_OFFSET=1`,
     `VLLM_GFX906_PERSISTENT_AR_SKINNY_SHMEM=1`, and public fallback only for
     `kind=attn, call_site=1664`. Both hosts survived O3 graph capture and
     strict c1 was valid, but performance lost: `.20` reached `62.282`
     backend TPS versus the public1664 high-water `63.084`, and `.30` reached
     `61.211` versus the public1664 support run `62.197`. This closes the
     obvious channel-offset composition after the public1664 trace; direct
     descriptor-side wins are still below serving-promotion scale unless they
     also reduce the sidecar tail.
419. Odd-root `RCCL_TREES` rotation does not compose with public1664 in
     serving. After the public1664 trace showed device `6/7` and channel tail
     skew, rechecked the previously exact rotation-3 tree string with the
     current public1664 stack. Both hosts survived graph capture and strict c1
     was valid, but `.20` reached only `62.383` backend TPS versus the
     public1664 high-water `63.084`, and `.30` reached `61.432` versus
     `62.197`. The tail problem is not solved by rotating the whole balanced
     tree root schedule; keep the promoted even-root balanced `RCCL_TREES` and
     use more targeted sidecar or attention-boundary work.
420. Per-callsite sidecar library routing is now graph-safe infrastructure,
     but the first exact-node primitive swap did not promote. Extended the
     kind-route overlay with `VLLM_GFX906_PERSISTENT_AR_LIB_CALLSITE` and
     exact/range/mod callsite route controls while preserving stable global
     launch-sequence numbering and public-bypass semantics. Routing only
     `kind=attn, call_site=1664` to the prior role-specialized Tree/LL sidecar
     runtime survived O3 graph capture and strict serving on `.20/.30`, but
     reached only `.20` `62.648` and `.30` `61.681` backend TPS. Keep the
     router as a source milestone for exact-node experiments; reject generic
     role-specialized fan-template routing as the `1664` fix.
421. Public-1664 still wants `NCCL_NTHREADS=128`. After fixing the kind-route
     runner so `NCCL_NTHREADS=unset|none|default` really omits the env var,
     the `.20` smoke briefly reached `63.173` backend TPS, but the full ladder
     reverted below the high-water: `63.053/63.889/60.294` for strict c1 /
     `c1_2000` / `c1_10000` versus public1664 `63.084/63.993/60.380`.
     `.30` also rejected unset with strict c1 `61.889` versus `62.197`.
     Keep `NCCL_NTHREADS=128`; the public-1664 single-node fallback does not
     change the thread-count conclusion.
422. Exact-node skinny/no-zero/channel-offset routing does not fix
     `call_site=1664`. Routing only `kind=attn, call_site=1664` to
     `libgfx906_sidecar_tree_ll_ar_multirow_skinny_nozero_chanoffset_20260613.so`
     while keeping all other nodes on the high-water sidecar survived O3 graph
     capture and strict serving, but reached only `.20` `62.873` and `.30`
     `61.442` backend TPS. This closes the exact-node version of that
     descriptor-cleanup branch; median sidecar wins remain below the
     tail-driven serving threshold.
423. Numel-guarded exact-node routing fixes the channel-base test method, but
     not the dense gate. The channel-base sidecar libraries only support the
     `1x5120` fp16 attention reduction; broad routing accidentally applied
     them to larger graph-captured tensors and failed with
     `persistent AR allreduce failed rc=2`. The kind-route overlay now supports
     `VLLM_GFX906_PERSISTENT_AR_CALLSITE_ROUTE_NUMEL_LIST/LO/HI`, allowing
     exact `kind=attn, call_site=1664, numel=5120` routing. Corrected smokes
     were strict-valid but below the public1664 high-water: `.20` `62.860`
     and `.30` `61.653`. Keep the numel guard as source/testing
     infrastructure; reject channel-base exact `1664` as the serving fix.
424. Residual public fallback and public-channel reductions do not extend the
     public1664 winner. The post-public1664 trace pointed at residual p99 site
     `1669`, but adding public fallback for `1664,1669` reached only `.20`
     `63.062` and `.30` `61.394` backend TPS. Keeping only `1664` public but
     lowering public RCCL to three channels reached `.20` `63.068` and `.30`
     `62.053`, below the promoted four-channel public1664 result. Two-channel
     public RCCL is unsafe in this graph-captured serving stack: both `.20`
     and `.30` faulted during startup graph capture with GPU memory access
     faults. Keep `NCCL_MIN/MAX_NCHANNELS=4`; do not revisit channel counts
     below `3` without a separate RCCL correctness lane.
425. Wall-clock sidecar trace is the latest real source breakthrough, but not
     a performance promotion. Added wall-clock entry/run-start/run-end ticks to
     the sidecar runtime and a new analyzer that estimates median per-device
     offsets from complete cohorts. The `.20` public1664 capped serving trace
     produced `2,362,392` entries, `274,834` complete cohorts, and zero
     incomplete cohorts. After offset correction, the data points away from
     another broad attention fallback and toward residual MLP sidecar nodes:
     top p99 call sites were mostly `mlp/sidecar/5120`, led by `1711`, `1759`,
     `1743`, `1783`, and `1767`. The first exact follow-up, public fallback
     `1664,1711`, was strict-valid but rejected on both primary hosts (`.20`
     `62.871`, `.30` `61.769`). Keep wall-clock trace as the cross-rank
     attribution tool; keep the dense global high-water unchanged at public
     `kind=attn, call_site=1664`.
426. The wall-clock MLP branch is now closed for simple public fallback and
     the first exact sidecar-library route. Public fallback for the next two
     ranked MLP sites also rejected: `1664,1759` reached `.20/.30`
     `62.777/61.918`, and `1664,1743` reached `62.877/61.795`, all below
     same-host public1664. Routing the top three MLP sites
     `1711,1759,1743` through the skinny/no-zero/channel-offset sidecar
     library while keeping `1664` public produced one `.20` smoke bump
     (`63.133`), but `.30` did not reproduce (`61.676`) and the full `.20`
     ladder rejected it at `62.993/63.851/60.279` versus public1664
     `63.084/63.993/60.380`. The next source work should not be another
     public-fallback permutation; it needs to change the hot sidecar primitive,
     collective schedule, or graph boundary.
427. The uncapped public1664 wall-clock trace confirms the tail is a
     sidecar schedule/primitive problem, not simple block order. A valid
     uncapped `.20` trace loaded `11,558,208` sidecar entries and
     `1,424,311` complete cohorts after `op_count >= 10000`. The complete
     report showed run-wall ownership dominated by device `7` and then
     device `6`, while arrival max ownership was mostly devices `5`, `4`,
     `6`, and `0`. Channel `1` had the highest entry-run p99 (`1060`
     ticks), but moving it later with block order `0,2,1` rejected at
     `.20/.30` `63.034/62.163`, and moving it earlier with `1,0,2` rejected
     at `63.006/61.838`, all below same-host public1664. Keep the channel
     permutation macros as lab infrastructure, but the next serious source
     path must change the hot sidecar collective schedule/primitive, reduce
     launch count, or fuse the RowParallel boundary instead of reordering the
     existing three blocks.
428. Public1664 sidecar tail ownership is leaf-heavy, not root-heavy. Extending
     the wall-clock analyzer with promoted `RCCL_TREES` role parsing showed
     active sidecar roots `{0:0, 1:2, 2:4}` with root p99 only `728` ticks,
     while leaf depth-2 and depth-3 roles owned the long run-wall cohorts.
     Logical device `7` is leaf on all active sidecar channels and dominates
     max-run ownership, with device `6` also contributing as leaf/depth-1.
     This is a source/diagnostic breakthrough because it rules out more
     root-only scheduling work. The direct sidecar leaf-up/down follow-up was
     exact and graph-safe, but rejected in serving on `.30` at `61.448`
     backend TPS versus the public1664 support run `62.197`. The next serious
     dense path must change the hot sidecar primitive or reduce the repeated
     `1x5120` RowParallel launch count; simple root rotation, channel order,
     public fallback expansion, and existing leaf-up/down scheduling are now
     closed.
429. Repaired producer/collective fusion is exact but still the wrong dense
     gate shape. The older fused `LLMM1 -> sidecar Tree/LL` lower-bound used
     the sidecar block shape to infer producer warp count, which could
     overstate the attention producer work for `K=768`. After separating
     producer logical threads from the sidecar `4`-warp block, both `.20` and
     `.30` reproduced the same conclusion: fused attention stayed about
     `13.9 us` slower than separate `LLMM1 + sidecar`, and MLP/down stayed
     about `24-25 us` slower. Immediate public-neighbor fallback
     `1664,1665,1666,1667` also rejected (`.20` `62.969`, `.30` `62.186`)
     versus the current public1664 high-water/support rows. This closes the
     coarse in-kernel producer/collective fusion and neighbor-public branches;
     any future fusion must create real chunk-level overlap or remove captured
     launch count, not just combine full producer completion with the current
     sidecar spin barrier.
430. Public1664 repeat variance did not clear the dense gate, but `.50` is now
     a usable ROCm 7.2 lane. Repeating the exact current winner on `.20/.30`
     reached `.20` `62.751/63.818/60.256` and `.30`
     `62.352/63.038/59.201` backend TPS for strict c1 / `c1_2000` /
     `c1_10000`, below the `.20` high-water
     `63.084/63.993/60.380`. The `.50` ROCm 7.2 failure was traced to broad
     exposure of physical GPU0, a `gfx803` display adapter. Exposing only
     `/dev/dri/card1-8` and `/dev/dri/renderD129-D136` plus `/dev/kfd`
     makes ROCm 7.2 initialize cleanly with the MI50s as container ordinals
     `0-7`; both dense sidecar runners now default to that selective exposure
     on `.50`. The `.50` public1664 strict smoke was valid at `56.129` TPS,
     so `.50` is qualified for diagnostics and compatibility but is not a
     performance-equivalent promotion lane for this stack. The same selective
     exposure is now also patched into the sidecar runtime build helper, C API
     smoke helper, and replicated RowParallel microbench helper; the corrected
     `.50` C API helper smoke `h50_selective_capi_smoke2_20260614` passed
     exactly.
431. Including sidecar channel `3` without launching all four channels is
     exact but not a promotion. The multi-row runtime now supports
     compile-time `SIDECAR_CHANNEL_LO/HI` so contiguous channel sets can be
     tested without changing the default `0-2` behavior. The first leaf-tail
     candidate used channels `1-3` in block order `1,3,2`, preserving the
     promoted `2048/1536/1536` count split. C API smokes were exact, but same
     method timing was tied/slower than control (`.20` `0.056321` versus
     `0.056054`, `.30` `0.065960` versus `0.058092` ms/call), and strict
     public1664 serving did not promote (`.20` `62.638`, `.30` `62.197`
     backend TPS). This closes the remaining simple channel-set branch:
     channel selection/order is not enough to remove the leaf-heavy tail.
432. The fixed `.50` lane does not reopen peer-memory collectives. With
     selective ROCm 7.2 device exposure, `.50` cleanly exposes eight 32 GiB
     gfx906 devices, but `hipDeviceCanAccessPeer` is zero for every
     off-diagonal pair. `HSA_FORCE_FINE_GRAIN_PCIE=1` does not change the
     matrix. The existing RCCL send/recv P2P allreduce graph probe also fails
     before timing rows at `ncclGroupEnd` with `invalid usage`, regardless of
     `NCCL_P2P_DISABLE=1` or `0`. This closes the post-vBIOS `.50` P2P
     revisit under the current driver stack; do not pursue peer-memory or
     symmetric-memory source work unless the peer matrix changes.
433. Active sidecar channel count is closed as a simple fix. A 5120-only
     two-channel `2560/2560` sidecar was exact and faster than same-method
     current control on `.20` (`0.074309` versus `0.085362` ms/call), but it
     regressed on `.30` (`0.084178` versus `0.080051`), and the `.20` strict
     serving smoke with public1664 fallback reached only `61.021` backend TPS.
     The fair multi-row two-channel runtime stayed exact for `5120`, but
     timed out on `10240` on both `.20` and `.30`. Four active channels were
     exact but no faster than control. Keep the promoted three-channel
     `2048/1536/1536` sidecar; future gains require a different primitive
     schedule, real launch-count reduction, or chunk/LL-line overlap.
434. Role-specialized Tree/LL up/down worker splitting is closed for the
     current dense gate. The earlier role-specialized source was corrected to
     accept the promoted `2048/1536/1536` descriptor and a compile-time
     `SIDECAR_ROLE_UP_THREADS` split, then tested on `.20/.30` at `32/96`,
     `64/64`, and `96/32`. The exact variants were slower or host-specific
     (`.20` best `0.086988 ms/call`, `.30` best `0.081877 ms/call`), and
     `96/32` timed out on both primary hosts. Do not reopen role-template or
     split-allocation tuning unless the primitive semantics change below the
     existing up/down Tree/LL template.
435. Current-descriptor fused channel-ready RowParallel boundary is closed as
     a coarse fusion path. The boundary bench now accepts compile-time
     `SIDECAR_COUNT_*`/`SIDECAR_CHUNK_GRAINS_*`, and channel-ready producer
     mapping follows the promoted `2048/1536/1536` descriptor. Exact graph
     replay still rejected on both primary hosts: `.20` separate
     `LLMM1 + sidecar` `0.051948/0.068380 ms` became fused
     `0.064869/0.091892 ms` for attention/MLP-down, and `.30`
     `0.052071/0.069377` became `0.095290/0.092117`. Do not pursue
     block-level producer-ready spin-barrier fusion; any remaining fusion path
     must overlap below the current block/channel granularity or remove
     captured launch count.
436. Targeted device-7-depth1 Tree/LL topology is closed as a whole-tree
     scheduling fix. Swapping labels in the balanced `RCCL_TREES` shape so
     logical device `7` becomes a depth-1 node on active sidecar
     channels `0-2` gave exact and fast direct component timing (`.20`
     `0.072494`, `.30` `0.070809 ms/call`), but strict public1664 serving did
     not promote: `.20` reached only `62.536` backend TPS versus the
     public1664 high-water `63.084`, while `.30` reached `62.432` as a local
     support bump only. Component-level tree reshaping is not enough; the
     next source work needs a lower-level sidecar primitive schedule,
     chunk/LL-line overlap, or graph-native RowParallel launch-count
     reduction.
437. The device-7-depth1 wall-clock follow-up explains why that topology did
     not promote. The corrected `.20` trace loaded `9,247,824` sidecar entries
     and `1,135,513` complete cohorts after graceful flush. Arrival p99
     improved to `888` ticks, but run-wall tail stayed high (`1008` ticks
     p99), and max-run ownership shifted to depth-1-node work on devices `6`
     and `7` instead of vanishing. This reinforces that the gate is below
     whole-tree rank assignment: the next real source target is the sidecar
     primitive schedule, chunk/LL-line overlap, or graph-native RowParallel
     launch-count reduction.
438. Simple sidecar worker occupancy retuning is closed for the public1664
     dense stack. The C API smoke now accepts `CAPI_WORKER_NWARPS` and
     `CAPI_BLOCK_THREADS`; sweeping `.20/.30` against the current
     `2048/1536/1536` sidecar showed `nWarps=4`, `block_threads=256` remains
     fastest (`.20` `0.058072`, `.30` `0.061172 ms/call`). Exact lower
     occupancy shapes were slower, and wider `320+` thread launches failed.
     Do not spend serving lanes on occupancy variants unless the primitive
     itself changes.
439. The resident/persistent sidecar worker does not become a promotion
     candidate merely by updating it to the current `2048/1536/1536`
     descriptor. A current-descriptor resident runtime built cleanly on
     `.20/.30` and passed exact C API smokes, but timing was only `.20`
     `0.057944` and `.30` `0.063888 ms/call`, below the bar for a serving
     retry. Keep the new kind-route post-capture wiring as diagnostics; do not
     run more simple resident lifecycle sweeps without a C API primitive win.
440. Current-winner post-IR collective coalescing is blocked by graph
     semantics, not pass visibility. On `.20` with the public1664 high-water
     stack, the diagnostic pass saw `all_reduce_inplace_kind(... kind='attn')`
     nodes and selected `128/128` direct allreduce/RMS boundaries in dry-run
     mode, but the grouping census found `collective_ops=129`,
     `groupable_pairs=0`, `blocked_by_dependency=128`,
     `blocked_by_intervening_use=128`, and no cluster larger than one. This
     closes simple post-IR coalescing over the current graph. The next dense
     source work must change the RowParallel schedule/boundary or primitive
     cost, not just make another pass over the same dependent node sequence.
441. Additional trace-selected public RCCL fallback does not currently compose
     with the dense public1664 winner. Mixed `1664+1689` and same-kind
     attention `1664+1694` strict c1 smokes were valid on `.20/.30`, but both
     lost versus same-host public1664 (`.20` `62.247`/`62.625`, `.30`
     `61.900`/`61.963`). The residual wall-clock table remains useful for
     attribution, but the next promotion-scale path is source work on the
     sidecar/RCCL boundary, not more sparse public-fallback routing.
442. The current `1x5120` sidecar Tree/LL tail is broadcast/recv dominated,
     not leaf reduce-up dominated. The throwaway phase-trace runtime
     `gfx906_sidecar_tree_ll_ar_runtime_phase_trace_20260614.cpp` passed exact
     C API smokes on `.20/.30` with the current `2048/1536/1536` descriptor.
     On `.20`, whole-run/reduce/broadcast P99 was
     `2744.2/1251.2/2712.6` ticks, while leaf reduce P99 was only `316.0`
     ticks versus leaf broadcast `2648.8`. `.30` amplified the same pattern:
     whole-run/reduce/broadcast P99 `25444.0/9268.0/25409.8`, leaf reduce
     `296.0`, leaf broadcast `25385.5`. This agrees with NPKIT and closes
     retreads around wait-send, simple role thread split, and pure reduce-up
     tuning. The next credible source patch should alter the LL
     broadcast/recv datapath itself or change/remove the RowParallel
     collective boundary.
443. The LL broadcast subphase trace narrows the remaining sidecar tail to
     first receive/read wait, not local store/forward/post work. The diagnostic
     runtime
     `gfx906_sidecar_tree_ll_ar_runtime_ll_subphase_trace_20260614.cpp` plus
     patched throwaway `prims_ll.h` passed exact C API smokes on `.20/.30`.
     On `.20`, broadcast P99 was `5543.6` ticks, first read/wait P99 was
     `4326.48`, first store/forward P99 was only `68`, and post-to-end P99 was
     `8`. On `.30`, the same shape held at `14268.0`, `12330.16`, `64`, and
     `8` ticks. Do not chase local FIFO store coalescing, child-forwarding, or
     post-step overhead for this primitive. Adjusted parent-child timing
     showed the parent had usually started broadcasting before the child waited
     (`3.34%`/`3.04%` parent-not-started fraction on `.20/.30`), but had not
     finished by child first-read end (`99.93%` on both hosts). The next
     gate-scale source work should therefore treat this as first-line
     production/propagation latency through the tree, with launch grouping or
     RowParallel collective-boundary reduction still attractive because they
     reduce how often that propagation cost is paid.
444. Forced Tree/LL microchunking is exact but not a portable serving path for
     the current sidecar primitive. The throwaway
     `.tmp/sidecar_tree_ll_microchunk_20260614/all_reduce.h` starts from the
     promoted `.20/.30` `all_reduce.h` and adds
     `GFX906_TREE_LL_MICROCHUNK` while preserving the current
     `2048/1536/1536` descriptor. Fresh `micro0` controls measured `.20`
     `0.062040` and `.30` `0.059337 ms/call`. `micro1024` gave `.20` only a
     small component win (`0.061247`) but regressed `.30` to `0.065649`;
     `micro512` regressed both hosts (`0.067117`/`0.067623`). This closes
     coarse `runTreeSplit` chunk-loop retuning. The remaining source work must
     operate lower in the LL first-read datapath, or reduce how often the graph
     pays the RowParallel collective boundary.
445. Simple LL receive polling backoff is exact but rejected. The throwaway
     `.tmp/sidecar_ll_pollbackoff_20260614/prims_ll.h` adds guarded
     `GFX906_LL_READ_SLEEP_*` controls and was staged with matching
     `all_reduce.h`/`primitives.h` so the override actually reaches
     `LLGenericOp`. Blanket `sleep256` regressed `.30` (`0.060595 ->
     0.063853 ms/call`) and did not improve `.20`. Broadcast-only
     `sleep256` avoided reduce-up changes but still lost to its 512-iteration
     controls on both hosts: `.20` `0.060173 -> 0.060460`, `.30`
     `0.057978 -> 0.058274`. This closes the simple local spin/yield policy
     path; the first-read wait is more likely tree production/propagation
     ordering or RowParallel boundary count.
446. Simple LL broadcast send-slot reordering is exact but rejected. The
     throwaway `.tmp/sidecar_ll_sendorder_20260614/prims_ll.h` adds
     `GFX906_LL_SEND_SLOT0_FIRST` to test whether writing send slot `0` before
     slots `1..N` improves broadcast first-line propagation. It did not:
     `slot0_first` regressed `.20` from `0.057677` to `0.058452 ms/call` and
     only moved `.30` from `0.057279` to `0.057553`. This closes the obvious
     "slot 0 is late because it is written last" hypothesis.
447. MLP-only `1792/1664/1664` sidecar kind-routing does not compose with the
     public1664 dense winner. The branch kept the promoted default
     `2048/1536/1536` sidecar, retained public RCCL fallback for
     `kind=attn, call_site=1664`, and routed only MLP/down kinded RowParallel
     calls through the `1792/1664/1664` sidecar library. Strict c1 was valid on
     both hosts but slower than same-host public1664: `.20` `62.597` and `.30`
     `61.073` backend TPS. The automatically started c1_2000 tier also lost:
     `.20` `63.558` versus `63.993`, and `.30` `62.839` versus `63.034`.
     The branch was stopped before c1_10000 because both earlier tiers
     rejected. This closes the simple "better direct-replay descriptor only for
     MLP/down" route; the remaining gate work still needs sidecar tail/primitive
     changes or RowParallel boundary/count reduction.
448. RowParallel prefix/callsite mapping is now exact enough for layer-targeted
     source work. A static-construction trace in the kind-route `linear.py`
     overlay records the `128` language `RowParallelLinear` rows outside the O3
     compiled forward path. The first forward-time trace attempt reproduced the
     known Dynamo limitation by failing on `posix.stat` inside `os.makedirs()`,
     so runtime forward tracing is now behind
     `VLLM_GFX906_ROWPART_PREFIX_TRACE_RUNTIME=1`. The valid `.20` mapping run
     `qwen36_27b_decode_tiers_rocm72_dense_sidecar_public1664_staticprefix_h20_host20_20260614_staticprefix_public1664_h20`
     showed the captured `1x5120` ranges `768-1023` and `1536-1791` are each
     two passes over the same `128` language RowParallel sequence. The promoted
     public fallback `call_site=1664` maps to
     `language_model.model.layers.0.linear_attn.out_proj`; `1665` is layer-0
     `mlp.down_proj`, `1666` is layer-1 `linear_attn.out_proj`, and `1667` is
     layer-1 `mlp.down_proj`.
449. Same-class early linear-attention public fallback does not compose with
     the public1664 dense winner. The precise mapping made it possible to test
     `1664,1666,1668`, corresponding to `linear_attn.out_proj` layers `0-2`
     without the MLP neighbors from the older `1664-1667` probe. Both primary
     hosts were strict-valid but slower than same-host public1664: `.20`
     `62.726` backend TPS versus `63.084`, and `.30` `61.451` versus `62.197`.
     This closes the immediate "same-class early attention" fallback branch and
     further lowers the value of sparse public-RCCL lists. Future work should
     use the mapping for exact layer/prefix source rewrites or primitive
     attribution, not as a reason to keep adding public fallback nodes.
450. Prefix-metadata RowParallel routing is useful infrastructure but not a
     public1664 replacement. The kind-route overlay can now attach module
     prefixes to RowParallel kind metadata under
     `VLLM_GFX906_PERSISTENT_AR_KIND_PREFIX_METADATA=1`, and the sidecar helper
     can bypass by `VLLM_GFX906_PERSISTENT_AR_BYPASS_PREFIX_LIST` while still
     routing by the base kind. Strict smokes with public fallback for
     `language_model.model.layers.0.linear_attn.out_proj` were valid but slower
     than callsite-only public1664: `.20` `62.964` versus `63.084`, and `.30`
     `61.862` versus `62.197`. The reason is structural: prefix-only matching
     hits the layer-0 linear-attention row in every captured decode shape/range,
     not only captured `1x5120` callsite `1664`. Keep prefix metadata for
     layer-targeted source work and diagnostics; do not promote prefix-only
     bypass for serving.
451. Prefix plus numel-filtered public bypass is also a serving rejection. The
     sidecar helper now supports opt-in
     `VLLM_GFX906_PERSISTENT_AR_BYPASS_NUMEL_LIST/LO/HI` so prefix or callsite
     public bypass can be restricted by tensor size without changing default
     behavior. Testing `language_model.model.layers.0.linear_attn.out_proj`
     with `BYPASS_NUMEL_LIST=5120` was strict-valid but below the public1664
     high-water: `.20` `62.751` versus `63.084`, and `.30` `62.137` versus
     `62.197`. This narrows the lesson: even layer-prefix plus decode-shape
     matching is still not equivalent to captured callsite `1664`, likely
     because it hits multiple captured `1x5120` passes. Keep exact callsite
     routing for the current winner; use prefix+numel as diagnostic
     infrastructure only.
452. Rank-selective sidecar channel ordering is exact but not a stable
     primitive win. The throwaway
     `.tmp/sidecar_rank_selective_channel_order_20260614/gfx906_sidecar_tree_ll_ar_runtime_rank_selective_channel_order_20260614.cpp`
     added guarded `SIDECAR_RANK_SELECTIVE_*` controls so only selected
     tail ranks use channel block order `0,2,1`, based on the public1664
     wall-clock trace showing devices `7/6` owning most sidecar run-wall
     groups. Initial rank `6-7` 512-iteration C API results looked
     neutral/winning (`.20` `0.057084 -> 0.057039`, `.30` noisy
     `0.064731 -> 0.060549`), but the longer reversed-order 1024-iteration
     repeat lost on both hosts: `.20` candidate/control
     `0.056924/0.056868`, `.30` `0.056538/0.056085`. The rank-7-only
     follow-up also rejects: `.20` `0.057329` beat only one noisy slow control
     and is slower than established `.20` controls, while `.30` regressed
     versus same-session control (`0.057933/0.056634`). Do not spend serving
     lanes on tail-rank-only channel reordering. The remaining dense path
     still needs a deeper Tree/LL primitive change or RowParallel
     launch-count/boundary reduction.
453. LL flag-only polling is not the missing first-read fix. The isolated
     `sidecar_ll_flagpoll_20260614` overlay changed `prims_ll.h` so the
     broadcast receive path can poll only the two LL flags, then load the full
     LL line once the flags match. This was exact on `.20/.30`, but slower on
     both hosts: `.20` `0.064309 -> 0.074171 ms/call` and `.30`
     `0.065196 -> 0.081752 ms/call`. The first-line wait is not improved by
     narrower per-iteration polling loads; continue toward parent-line
     production/propagation changes or RowParallel collective-boundary
     reduction.
454. The current dense public1664 winner does ramp clocks under auto during
     fixed decode. The `.20` clock probe reproduced normal throughput
     (`c1_2000` `63.922`, `c1_10000` `60.320` backend TPS) while sampling
     `rocm-smi`. During `c1_10000`, all GPUs averaged `97.4%` GPU use and
     ran near top clocks: `mclk` about `983-991/1000 MHz`, `sclk` about
     `1704-1715/1725 MHz`, and `fclk` about `1259-1268/1278 MHz`. The
     remaining dense gap is not explained by idle memory/core clocks; keep the
     primary path on RowParallel boundary/count reduction or the Tree/LL
     first-line propagation path.
455. Same-config public1664 variance does not clear the dense 27B gate. A
     clean idle-lane rerun on `.20/.30` kept the current winner in the known
     band rather than crossing `65 TPS`: `.20`
     `62.601/63.858/60.263` and `.30` `61.183/62.945/59.232` backend TPS for
     strict c1 / `c1_2000` / `c1_10000`. Keep the global high-water unchanged
     at prior `.20` public1664 `63.084/63.993/60.380`, and continue source
     work on RowParallel boundary/count reduction or Tree/LL first-line
     production/propagation rather than waiting for noise to promote the
     existing config.
456. Layer-resolved wall-clock attribution closes the "one more sparse
     callsite list" path for the current dense winner. The full-group analyzer
     `analyze_sidecar_wallclock_layer_tail_fullgroup_20260614.py` joins the
     uncapped public1664 sidecar wall-clock trace to the exact RowParallel
     prefix map using the same complete-cohort method as the authoritative
     wall-clock report. It used `1424311` complete cohorts, all in hot decode
     range/pass `1536-1791` pass `1` shape `1x5120`. Arrival tail is broad:
     top 64 by arrival p99 span layers `1-61` and include `44` MLP/down,
     `11` self-attention, and `9` linear-attention rows. Run-max tail is even
     more class-specific: top 64 by run-max p99 are `63` MLP/down rows and
     `1` attention row, spanning layers `0-62`; aggregate run-max p99 is
     `1108` ticks for MLP/down versus `884` for attention. This argues for
     model-wide RowParallel boundary or primitive work, not another guessed
     sparse layer fallback.
457. Exact hot-pass MLP/down public fallback is rejected. To test whether the
     MLP-heavy run-max tail could be solved by normal public RCCL, exact public
     fallback was applied to `call_site=1664` plus all MLP/down callsites in
     hot `1x5120` range/pass `1536-1791`, pass `1`
     (`1665,1667,...,1791`). The smokes were strict-valid but slower than the
     public1664 high-water: `.20` `61.392` backend TPS and `.30` `61.463`.
     Keep public fallback only for the current narrow `kind=attn,
     call_site=1664` milestone. The broad MLP/down issue needs a source-level
     primitive or collective-boundary change.
458. Fused producer/sidecar sidecar-last scheduling does not rescue the fused
     boundary. The throwaway
     `.tmp/llmm1_sidecar_fused_sidecarlast_20260614/rccl_llmm1_ar_rms_boundary_bench_20260612.cpp`
     moved producer CTAs ahead of sidecar CTAs inside the fused
     LLMM1+sidecar kernel. `.20` showed attention as a near-tie
     (`0.052874 -> 0.052863 ms`) and MLP/down as a slight loss
     (`0.079530 -> 0.079605 ms`); `.30` landed in the same sane fused band
     after rejecting an anomalous control. Simple CTA ordering is not why the
     fused producer/collective path is slower. Future fusion would need true
     LL-line/chunk overlap or launch-count removal, not just block reordering.
459. Sidecar Tree/LL128 is exact but much slower for the hot `1x5120` dense
     decode primitive. The isolated
     `.tmp/sidecar_tree_ll128_20260614/gfx906_sidecar_tree_ll_ar_runtime_ll128_20260614.cpp`
     swapped the sidecar `RunWorkColl` from `NCCL_PROTO_LL` to
     `NCCL_PROTO_LL128` while preserving the promoted `2048/1536/1536`
     descriptor. Matched C API controls were `.20` `0.073089` and `.30`
     `0.072796 ms/call`; LL128 was exact but regressed to `.20` `0.104334`
     and `.30` `0.105641`. Do not use LL128 for single-request dense decode;
     keep LL128 only as a prefill/concurrency lever where rows are large enough
     to amortize its wider protocol.
460. Hand-specializing the LL broadcast `recvCopySend` wrapper is exact but
     not useful enough. The throwaway
     `.tmp/sidecar_ll_bcast_specialized_20260614/prims_ll.h` added guarded
     `GFX906_LL_BCAST_SPECIALIZED=1`, replacing the generic
     `LLGenericOp<1,1,-1,Output>` path with a narrower broadcast helper for the
     hot sidecar. C API results with reverse controls rejected it before
     serving: `.20` control/candidate/reverse-control
     `0.072962/0.074048/0.071260 ms/call`, and `.30`
     `0.072644/0.071191/0.070815`. The compiler is already removing enough of
     the generic template overhead; the remaining dense gap needs a materially
     different Tree/LL line-production primitive or fewer RowParallel
     collective boundaries.
461. Tree-path-targeted LL broadcast send ordering is exact but not portable
     enough to promote. The throwaway
     `.tmp/sidecar_ll_treepath_sendorder_20260614/prims_ll.h` added guarded
     `GFX906_LL_BCAST_TREEPATH_FIRST=1`, applying slot0-first only to TP8
     channel/rank pairs that carry the slow leaf path in the promoted balanced
     tree. Matched 4096-iteration C API runs were mixed: `.20` candidate/control
     `0.057114/0.057869 ms/call`, but `.30` candidate/control
     `0.056916/0.056371`. Do not run serving for this branch. Blanket,
     broadcast-only, and tree-path-targeted send-order tweaks are now closed;
     the next dense source work must change line-production/propagation more
     materially or reduce RowParallel collective boundary count.
462. Broadcast-side `waitSend` no-poll is exact but not portable. The throwaway
     `.tmp/sidecar_ll_bcast_waitsend_20260614/prims_ll.h` added guarded
     `GFX906_LL_BCAST_ASSUME_SEND_SPACE=1` so LL broadcast `recv-copy-send`
     still publishes FIFO size, advances the send head, and keeps the block
     barrier, but skips the remote-head polling loop. C API exactness passed,
     but the result split by host: `.20` control/candidate
     `0.057044/0.056386 ms/call`, while `.30` control/candidate
     `0.056717/0.063378`. Do not serve this branch. Send-head polling is not a
     safe portable lever; future Tree/LL work needs equivalent queue semantics
     with a different propagation schedule, or RowParallel collective-count
     reduction above the primitive.
463. Broadcast LL FIFO scalar store-order reversal is exact but mixed. The
     throwaway `.tmp/sidecar_ll_bcast_storeorder_20260614/prims_ll.h` added
     guarded `GFX906_LL_BCAST_STORE_REVERSE_WORDS=1`, preserving two scalar
     `u64` FIFO stores while reversing their order only for broadcast
     `recv-copy-send`. Longer 4096-iteration C API runs split by host:
     `.20` control/candidate `0.056507/0.056790 ms/call`, while `.30`
     `0.057086/0.056511`. Do not serve this branch. Together with the older
     rejected vector-store path, this closes LL FIFO store publication order as
     a robust dense-gate lever for the current Tree/LL sidecar primitive.
464. Four-warp sidecar batching is recoverable only when LL barrier/group state
     is reset between dependent work items. Raw `RunWorkBatch` with
     `nWarps=4` timed out even at count `2` on both `.20` and `.30` under the
     promoted balanced Tree/LL envelope. The throwaway
     `.tmp/sidecar_batch_reset_20260614/rccl_sidecar_tree_ll_work_probe_batch_reset_20260614.cpp`
     filled up to eight `ncclDevWorkColl` descriptors once, then ran each
     descriptor with a per-work reset of `ncclShmem.barrier_pat` and
     `ncclShmem.groups[*].barrier`. This made counts `2`, `4`, and `8` exact
     on both hosts: count `8` graph time was `.20` `0.028334` and `.30`
     `0.028365 ms/work`. Same-run serial reuse still wins slightly at count
     `8` (`.20` `0.028216`, `.30` `0.028206`), and its corrected serving
     ladder already rejected below the global high-water. Keep `batch_reset`
     as a source diagnostic milestone and future grouped-sidecar construction
     rule, not a serving promotion.
465. Full group-barrier reset is not a single-work optimization. The
     throwaway
     `.tmp/sidecar_full_group_reset_20260614/gfx906_sidecar_tree_ll_ar_runtime_full_group_reset_20260614.cpp`
     changed only the hot sidecar kernel initialization so
     `tid < NCCL_MAX_GROUPS` clears every `ncclShmem.groups[tid].barrier`.
     Matched C API smokes with the promoted `2048/1536/1536` descriptor were
     exact but slower: `.20` control/candidate
     `0.056210/0.058163 ms/call`, `.30` `0.056647/0.056749`. Keep all-group
     reset as a required between-work cleanup for manually serialized
     multiwork kernels; do not add it to the ordinary single-work serving
     sidecar.
466. Forcing a single 128-bit LL FIFO line load is not a gfx906 Tree/LL
     receive-path fix. The throwaway
     `.tmp/sidecar_ll_b128load_20260614/prims_ll.h` added guarded
     `GFX906_LL_FORCE_DWORDX4_LOAD=1`, replacing the normal two
     non-temporal 64-bit gfx9 receives in `readLL`, `readLLBeginAll`, and
     `readLLFinish` with `global_load_dwordx4 ... glc slc` plus
     `s_waitcnt vmcnt(0)`. The candidate compiled and was correctness-exact,
     but regressed both hosts with the promoted `2048/1536/1536` descriptor:
     `.20` control/candidate `0.056449/0.061807 ms/call`, and `.30`
     `0.067920/0.077688`. Keep the current two-`u64` load path; the remaining
     first-read/tail issue needs a different primitive or RowParallel
     launch-count reduction.
467. Reversing the two gfx9 LL FIFO `u64` receive loads is not a portable
     first-read fix. The throwaway
     `.tmp/sidecar_ll_readorder_20260614/prims_ll.h` added guarded
     `GFX906_LL_REVERSE_U64_LOAD_ORDER=1` and changed only `readLL`,
     `readLLBeginAll`, and `readLLFinish` so the high word is loaded before
     the low word. After discarding an invalid simultaneous control/candidate
     run that oversubscribed each host, clean sequential C API repeats were
     exact but not promotable: `.20` control/candidate
     `0.056300/0.060437 ms/call`, while `.30` was only a noise-level
     `0.056179/0.055990`. Keep the normal low-then-high two-`u64` order; the
     remaining LL broadcast first-read tail needs a different Tree/LL
     primitive or RowParallel boundary/count reduction.
468. The 2026-06-20 ROCm7.2 Dense/MoE runner supersedes the pre-release dense
     open status. `qwen36-gfx906/README.md` records deploy script SHA256
     `c8e8ef99ec39a0232f74a7bd0fe0efe0316c0e0678992a1c104eff3c05513c9a`,
     pushed image
     `joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`,
     Docker Hub manifest digest
     `sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e`,
     deterministic archive SHA256
     `5316c3f6202fcb77987dabbf1e14e7369441ea127efed4f6def30259a09cfcb9`,
     image manifest
     `sha256:7dadf367ec86fe2eb1dc22fb3af3002c3514514833b52329595a26e7a80ae247`,
     config
     `sha256:45decd88eb7c10c0408327438e07c2a655e45cc7534f8b662e5c4089a6b88568`,
     `created 2025-11-24T16:00:00Z`, and `layers 19`. At
     `MAX_MODEL_LEN=131072` with eight pre-measure warmups,
     `dense27b_tp8_fullbar_p2pon` on `.20` reached strict backend TPS
     `69.514`, `c1_2000` backend TPS `70.347`, and `c1_10000` backend TPS
     `66.069`; note `strict gate valid`. `moe35b_tp8_fullbar_p2pon` on `.30`
     reached strict backend TPS `94.907`, `c1_2000` backend TPS `97.028`, and
     `c1_10000` backend TPS `91.290`; note `strict gate valid`.
     `moe35b_tp4_fullbar_p2pon` on `.30` had strict `invalid/runaway`,
     `c1_2000` backend TPS `116.146`, and `c1_10000` backend TPS `109.283`;
     note `uncapped strict prompt did not stop after >60K tokens`. The same
     ROCm7.2 experimental release image covers dense and MoE through
     model-specific env and overlays. Docker Hub remains an evergreen artifact
     distribution channel; TPS claims should stay in GitHub Releases,
     repository docs, and benchmark artifacts. The full-BAR/P2P-on lane
     required official AMD VBIOS standardization, not modified BIOS images,
     plus amdgpu source patching. This is not a user instruction to flash
     cards; the public repo does not redistribute BIOS binaries or imply
     warranty or certification. v0.1.0 remains the older published GitHub Release
     boundary; this entry is post-v0.1 main-branch validation until a separate
     release is published.
