# GFX906 Current Research Roadmap

## Dense 27B Status

Dense 27B clears the ai-info 10K gate in post-v0.1 main-branch validation.

[qwen36-gfx906/README.md](../qwen36-gfx906/README.md) is the authoritative latest
technical source. It records the 2026-06-20 ROCm7.2 Dense/MoE runner at
`MAX_MODEL_LEN=131072` with eight pre-measure warmups. The dense active contract is
`dense27b_tp8_fullbar_p2pon` on `.20`: strict backend TPS `69.514`, `c1_2000`
backend TPS `70.347`, and `c1_10000` backend TPS `66.069`; note `strict gate
valid`.

v0.1.0 remains the older published GitHub Release boundary. These dense 27B numbers
are post-v0.1 main-branch validation notes until a separate release is published.
GitHub Releases remain canonical for published claim boundaries.

The next dense work is to protect this usable 128K-context contract, improve margin,
and keep reducing RowParallel / RCCL / NCCL collective-boundary cost without breaking
text correctness or the strict serving profile.

## MoE 35B Status

Qwen3.6 35B-A3B MoE now has a full-BAR/P2P-on TP8 strict-valid bar and a TP4
capped-performance bar under the same ROCm7.2 experimental release image.

The current TP8 active contract is `moe35b_tp8_fullbar_p2pon` on `.30` at
`MAX_MODEL_LEN=131072` with eight pre-measure warmups: strict backend TPS `94.907`,
`c1_2000` backend TPS `97.028`, and `c1_10000` backend TPS `91.290`; note `strict
gate valid`.

The TP4 lane is `moe35b_tp4_fullbar_p2pon` on `.30`: strict is `invalid/runaway`,
`c1_2000` backend TPS is `116.146`, and `c1_10000` backend TPS is `109.283`; note
`uncapped strict prompt did not stop after >60K tokens`. Treat TP4 as capped-only
until a strict-valid run is produced.

The same ROCm7.2 experimental release image covers dense and MoE with model-specific
env and overlays:
`joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-0a2dbd6b7f0b`,
Docker Hub manifest digest
`sha256:8c380e9ca48943d8617de5a2e2eaf32a26dcc2c341e4b4f4f8c45294a72b8f1e`.
Docker Hub remains an evergreen artifact distribution channel; TPS claims should stay
in GitHub Releases, repository docs, and benchmark artifacts.

Platform remediation for the full-BAR/P2P-on lane required official AMD VBIOS
standardization, not modified BIOS images, plus amdgpu source patching. This is not
a user instruction to flash cards; the public repo does not redistribute BIOS
binaries or imply warranty or certification.

## Primary Dense Direction

- Preserve the dense 27B ai-info gate-clear contract at 128K context.
- Maintain byte-for-byte reproducible ROCm7.2 image and overlay selection.
- RowParallel collective-boundary improvement.
- Non-resident sidecar / public-RCCL fallback / per-node primitive selection where it
  preserves graph semantics.
- Tree/LL and RCCL source work for small-message collective behavior.
- Collective launch-count or reduction-count changes that preserve correctness.
- Legal coalescing or graph-native RowParallel boundary changes.
- Producer/collective/consumer fusion below graph-op overhead where source evidence
  supports it.
- Exact small-message collective improvements that survive optimized serving-path
  validation.

## Secondary / Side Lanes

- Prefill and concurrency milestones should remain separate from single-request c1
  decode claims.
- LL128 and wider-row bandwidth work are side lanes unless they improve the hot dense
  single-request path.
- Reduce-scatter or sequence-parallel work should be treated as prefill/concurrency
  evidence unless serving decode evidence changes that status.
- MoE side lanes should remain separate from dense 27B claims.

## Closed Or Deprioritized Paths

These paths remain useful public learning, but the current source record does not treat
them as the active dense release-margin path:

- Consumer-only RMS/logits cleanup as a standalone gate path.
- Public send/recv composition.
- Raw peer-memory or HIP IPC collectives on the current GFX906 hosts.
- AITER or symmetric-memory substrate work on the current GFX906 hosts.
- Simple NCCL/RCCL environment knob sweeps.
- TP4 dense serving as the primary dense path.
- Model-forward split endpoint wrappers.
- Chunked RowParallel overlap that increases allreduce launch count.
- Direct median descriptor wins without serving promotion.
- Earlier singlework/LL16 branches as the global current winner when superseded by the
  latest sidecar/public1664 source inventory.

## Public Outputs

The roadmap should produce public-benefit outputs rather than private access claims:

- Benchmark documentation.
- Sanitized source inventory.
- Sanitized key learnings.
- QC methodology.
- Hardware verification standards.
- Reproducibility documentation.
- Canonical deployment notes.

## Near-Term Milestones

- Maintain the public source-kernel inventory.
- Maintain the public key-learnings record.
- Maintain the post-v0.1 active-contract record.
- Publish benchmark methodology updates.
- Publish QC methodology updates.
- Formalize benchmark artifacts.
- Maintain reproducible Docker/runtime docs.
- Continue source-level collective/boundary work.
