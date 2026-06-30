# HF / Non-GGUF Overlay Scope

This scope is for the original HF-weight release reproduction path.

GGUF-only files and `VLLM_QWEN35_GGUF_*` / `VLLM_GFX906_GGUF_*` behavior must
not be selected by HF/non-GGUF profiles.

`minimal-bundle/` contains the clean release-path HF runtime pieces needed by
the contract launcher. This bundle intentionally mirrors the broader
v0.2/v0.2.1 release patch composition used by the historical HF Dense control
and the original HF MoE release profiles rather than the smaller first-pass HF
subset.

- ROCm/RCCL overlay library;
- GFX906 persistent all-reduce sidecar library;
- native GFX906 SwiGLU extension;
- release Python overlays for RowParallel / persistent all-reduce routing;
- release model, platform, attention, utility, and MoE fastpath overlays that
  are part of the original clean HF patch bundle.
- the tuned MoE config used by the original HF MoE release profile.

It intentionally excludes GGUF loader, GGUF quantization, and GGUF model
registry patches.
