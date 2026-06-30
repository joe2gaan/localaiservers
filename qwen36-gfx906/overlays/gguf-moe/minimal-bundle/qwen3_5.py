# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3.5 Series compatible with HuggingFace weights."""

import os
import typing
from itertools import islice
from collections.abc import Callable, Iterable

import torch
from einops import rearrange
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    VllmConfig,
)
from vllm.distributed import (
    get_pp_group,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating_delta_rule_update
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as _Qwen3_5GemmaRMSNorm,
)

def _qwen35_apply_gguf_loadtime_norm_offset(name: str, loaded_weight: torch.Tensor) -> torch.Tensor:
    if os.environ.get("VLLM_QWEN35_GGUF_LOADTIME_NORM_OFFSET_FIX") == "1":
        if name.endswith("norm.weight") and "ssm_norm.weight" not in name and "linear_attn.norm.weight" not in name:
            if torch.is_floating_point(loaded_weight):
                return loaded_weight - 1.0
    return loaded_weight

Qwen3_5RMSNorm = _Qwen3_5GemmaRMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
)
from vllm.transformers_utils.configs.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsPP,
    _require_is_multimodal,
)
from .qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from .qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextGatedDeltaNet,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
    fused_gdn_gating,
)
from .qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    _merge_multimodal_embeddings,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


def _qwen35_norm_param_audit(model: nn.Module, label: str) -> None:
    path = os.environ.get("VLLM_QWEN35_NORM_PARAM_AUDIT_PATH")
    if not path:
        return
    try:
        rows = []
        for name, param in model.named_parameters():
            if not (
                name.endswith("input_layernorm.weight")
                or name.endswith("post_attention_layernorm.weight")
                or name.endswith("linear_attn.norm.weight")
                or name.endswith("self_attn.q_norm.weight")
                or name.endswith("self_attn.k_norm.weight")
                or name.endswith("model.norm.weight")
                or name.endswith("language_model.norm.weight")
                or name.endswith(".norm.weight")
            ):
                continue
            tensor = param.detach().float().cpu()
            if tensor.numel() == 0:
                continue
            rows.append((
                name,
                tuple(tensor.shape),
                float(tensor.mean().item()),
                float(tensor.abs().max().item()),
                float(tensor.min().item()),
                float(tensor.max().item()),
            ))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"audit_label={label} pid={os.getpid()} rows={len(rows)}\n")
            fh.write("name\tshape\tmean\tabsmax\tmin\tmax\n")
            for name, shape, mean, absmax, minv, maxv in sorted(rows):
                fh.write(
                    f"{name}\t{shape}\t{mean:.6f}\t{absmax:.6f}\t"
                    f"{minv:.6f}\t{maxv:.6f}\n"
                )
    except Exception as exc:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(f"audit_error={type(exc).__name__}:{exc}\n")
        except Exception:
            pass



def _qwen35_boundary_trace_dir(prefix: str) -> str | None:
    trace_dir = os.environ.get("VLLM_GFX906_BOUNDARY_TRACE_DIR") or os.environ.get(
        "VLLM_GFX906_GGUF_TRACE_DIR"
    )
    if not trace_dir:
        return None
    trace_gate = os.environ.get("VLLM_GFX906_GGUF_TRACE_GATE_FILE")
    if trace_gate and not os.path.exists(trace_gate):
        return None
    prefixes = os.environ.get("VLLM_GFX906_BOUNDARY_TRACE_PREFIXES", "")
    if prefixes:
        wanted = [p.strip() for p in prefixes.split(",") if p.strip()]
        if wanted and not any(prefix.startswith(p) or p in prefix for p in wanted):
            return None
    return trace_dir


def _qwen35_trace_tensor(label: str, tensor: torch.Tensor | None) -> None:
    trace_dir = _qwen35_boundary_trace_dir(label)
    if not trace_dir:
        return
    try:
        os.makedirs(trace_dir, exist_ok=True)
        out = os.path.join(trace_dir, f"boundary_trace_pid_{os.getpid()}.txt")
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"label={label}\n")
            if tensor is None:
                fh.write("tensor=None\n")
                return
            t = tensor.detach()
            fh.write(f"shape={tuple(t.shape)} dtype={t.dtype}\n")
            tf = t.float()
            fh.write(
                "stats="
                f"min:{tf.min().item():.6f} "
                f"max:{tf.max().item():.6f} "
                f"mean:{tf.mean().item():.6f} "
                f"norm:{tf.norm().item():.6f}\n"
            )
            row = tf.reshape(-1, tf.shape[-1])[-1] if tf.dim() > 1 else tf.reshape(-1)
            sample = row[: min(16, row.numel())].tolist()
            fh.write("sample=" + ",".join(f"{x:.6f}" for x in sample) + "\n")
            if os.environ.get("VLLM_GFX906_BOUNDARY_TRACE_ROW_VALUES") == "1":
                labels_env = os.environ.get("VLLM_GFX906_BOUNDARY_TRACE_ROW_VALUE_LABELS", "")
                labels = [part.strip() for part in labels_env.split(",") if part.strip()]
                if not labels or any(label == part or label.endswith(part) for part in labels):
                    fh.write("row_values=" + ",".join(f"{float(x):.6f}" for x in row.cpu()) + "\n")
            if os.environ.get("VLLM_GFX906_BOUNDARY_TRACE_ALL_ROWS") == "1" and tf.dim() > 1:
                labels_env = os.environ.get("VLLM_GFX906_BOUNDARY_TRACE_ALL_ROW_LABELS", "")
                labels = [part.strip() for part in labels_env.split(",") if part.strip()]
                if not labels or any(label == part or label.endswith(part) for part in labels):
                    rows = tf.reshape(-1, tf.shape[-1]).cpu()
                    encoded_rows = []
                    for trace_row in rows:
                        encoded_rows.append(",".join(f"{float(x):.6f}" for x in trace_row))
                    fh.write("all_row_values=" + ";".join(encoded_rows) + "\n")
    except Exception as exc:
        try:
            os.makedirs(trace_dir, exist_ok=True)
            with open(
                os.path.join(trace_dir, f"boundary_trace_error_pid_{os.getpid()}.txt"),
                "a",
                encoding="utf-8",
            ) as fh:
                fh.write(f"label={label} error={type(exc).__name__}:{exc}\n")
        except Exception:
            pass



def _qwen35_v_head_inverse_permutation(
    num_key_heads: int,
    num_value_heads: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Inverse of llama.cpp's Qwen3.5 V-head GGUF reorder."""
    value_per_key = num_value_heads // num_key_heads
    perm = torch.arange(num_value_heads * head_dim, device=device)
    perm = perm.reshape(num_key_heads, value_per_key, head_dim)
    perm = perm.transpose(0, 1).contiguous().reshape(-1)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=device)
    return inv_perm


def _qwen35_inverse_v_rows(
    weight: torch.Tensor,
    config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig,
    head_dim: int,
) -> torch.Tensor:
    perm = _qwen35_v_head_inverse_permutation(
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        head_dim,
        weight.device,
    )
    return weight.index_select(0, perm)


def _qwen35_inverse_v_cols(
    weight: torch.Tensor,
    config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig,
    head_dim: int,
) -> torch.Tensor:
    perm = _qwen35_v_head_inverse_permutation(
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        head_dim,
        weight.device,
    )
    return weight.index_select(1, perm)


def _qwen35_unpermute_linear_attention_gguf_weight(
    name: str,
    loaded_weight: torch.Tensor,
    config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig,
) -> torch.Tensor:
    """Convert qwen35/qwen35moe GGUF linear-attention layout to HF layout.

    llama.cpp reorders V-like Qwen3.5 linear-attention heads while writing GGUF
    so its runtime can use tiled broadcast. vLLM's Qwen3.5 module expects the
    original HF grouped-by-K-head layout, so GGUF weights need the inverse here.
    """
    if name.endswith(".linear_attn.A_log"):
        if torch.is_floating_point(loaded_weight) and bool((loaded_weight < 0).all()):
            loaded_weight = torch.log((-loaded_weight).to(torch.float32)).to(loaded_weight.dtype)
        if config.linear_num_key_heads != config.linear_num_value_heads:
            loaded_weight = _qwen35_inverse_v_rows(loaded_weight, config, 1)
        return loaded_weight

    if config.linear_num_key_heads == config.linear_num_value_heads:
        return loaded_weight

    key_dim = config.linear_key_head_dim
    value_dim = config.linear_value_head_dim
    key_total = key_dim * config.linear_num_key_heads

    if name.endswith(".linear_attn.in_proj_qkv.weight"):
        q = loaded_weight[:key_total]
        k = loaded_weight[key_total:2 * key_total]
        v = _qwen35_inverse_v_rows(loaded_weight[2 * key_total:], config,
                                   value_dim)
        return torch.cat([q, k, v], dim=0)

    if name.endswith(".linear_attn.in_proj_z.weight"):
        return _qwen35_inverse_v_rows(loaded_weight, config, value_dim)

    if name.endswith((".linear_attn.in_proj_a.weight",
                      ".linear_attn.in_proj_b.weight")):
        return _qwen35_inverse_v_rows(loaded_weight, config, 1)

    if name.endswith(".linear_attn.dt_bias"):
        return _qwen35_inverse_v_rows(loaded_weight, config, 1)


    if name.endswith(".linear_attn.conv1d.weight") and loaded_weight.dim() == 2:
        qk_channels = 2 * key_total
        qk_part = loaded_weight[:qk_channels]
        v_part = _qwen35_inverse_v_rows(loaded_weight[qk_channels:], config,
                                        value_dim)
        return torch.cat([qk_part, v_part], dim=0).unsqueeze(1)

    if name.endswith(".linear_attn.out_proj.weight"):
        return _qwen35_inverse_v_cols(loaded_weight, config, value_dim)

    return loaded_weight


class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5Config)


class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5MoeConfig)


class Qwen3_5GatedDeltaNet(Qwen3NextGatedDeltaNet):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Experimental GGUF MoE parity path: fold beta/alpha into the
        # qkvz projection so graph structure matches the HF release path.
        if hasattr(self, "in_proj_ba"):
            delattr(self, "in_proj_ba")

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        raise NotImplementedError(
            "Qwen3.5 Series dont need to fix query key value ordering"
        )

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[
                key_dim,
                key_dim,
                value_dim,
                value_dim,
                self.num_v_heads,
                self.num_v_heads,
            ],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # Qwen3.5 has separate in_proj_b and in_proj_a weights in the
        # checkpoint, which are loaded into the fused in_proj_ba parameter
        # via stacked_params_mapping with shard_id 0 and 1 respectively.
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[num_v_heads] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )


    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """
        Core attention computation with request-gated runtime-state tracing.
        """
        trace_base = getattr(
            self,
            "_gguf_parent_trace_label",
            getattr(self, "prefix", "linear_attn"),
        )
        trace_prefix = f"{trace_base}.linear_attn_core"
        trace_call_idx = int(getattr(self, "_gguf_parent_call_idx", 0))
        trace_max = int(os.environ.get("VLLM_GFX906_CALL_TRACE_MAX_EVENTS", "3"))
        trace_active = (
            os.environ.get("VLLM_GFX906_INTERNAL_BOUNDARY_TRACE", "1") == "1"
            and _qwen35_boundary_trace_dir(trace_prefix) is not None
            and trace_call_idx < trace_max
        )

        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if trace_active:
            _qwen35_trace_tensor(f"{trace_prefix}.runtime.dt_bias", self.dt_bias)
            _qwen35_trace_tensor(f"{trace_prefix}.runtime.A_log", self.A_log)
            _qwen35_trace_tensor(f"{trace_prefix}.runtime.conv_weights", conv_weights)
            _qwen35_trace_tensor(f"{trace_prefix}.runtime.conv_bias", self.conv1d.bias)
            _qwen35_trace_tensor(f"{trace_prefix}.runtime.ssm_norm_weight", self.norm.weight)
            _qwen35_trace_tensor(f"{trace_prefix}.state.conv_before", conv_state)
            _qwen35_trace_tensor(f"{trace_prefix}.state.ssm_before", ssm_state)
            _qwen35_trace_tensor(f"{trace_prefix}.input.mixed_qkv_sliced", mixed_qkv)
            _qwen35_trace_tensor(f"{trace_prefix}.input.b_sliced", b)
            _qwen35_trace_tensor(f"{trace_prefix}.input.a_sliced", a)

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        if trace_active:
            _qwen35_trace_tensor(f"{trace_prefix}.input.mixed_qkv_non_spec_preconv", mixed_qkv_non_spec)
            _qwen35_trace_tensor(f"{trace_prefix}.input.mixed_qkv_spec_preconv", mixed_qkv_spec)

        if spec_sequence_masks is not None:
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][
                    : attn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        if attn_metadata.num_prefills > 0:
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[
                    : attn_metadata.num_actual_tokens
                ],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        if trace_active:
            _qwen35_trace_tensor(f"{trace_prefix}.postconv.mixed_qkv_non_spec", mixed_qkv_non_spec)
            _qwen35_trace_tensor(f"{trace_prefix}.postconv.mixed_qkv_spec", mixed_qkv_spec)
            _qwen35_trace_tensor(f"{trace_prefix}.state.conv_after_conv", conv_state)

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec
        )

        if trace_active:
            _qwen35_trace_tensor(f"{trace_prefix}.postconv.query_non_spec", query_non_spec)
            _qwen35_trace_tensor(f"{trace_prefix}.postconv.key_non_spec", key_non_spec)
            _qwen35_trace_tensor(f"{trace_prefix}.postconv.value_non_spec", value_non_spec)

        if attn_metadata.num_prefills > 0:
            g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
            if spec_sequence_masks is not None:
                g_non_spec = g.index_select(1, non_spec_token_indx)
                beta_non_spec = beta.index_select(1, non_spec_token_indx)
            else:
                g_non_spec = g
                beta_non_spec = beta
        else:
            g_non_spec = None
            beta_non_spec = None

        if trace_active:
            _qwen35_trace_tensor(f"{trace_prefix}.gating.g_non_spec", g_non_spec)
            _qwen35_trace_tensor(f"{trace_prefix}.gating.beta_non_spec", beta_non_spec)

        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[
                        : attn_metadata.num_spec_decodes + 1
                    ],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        if attn_metadata.num_prefills > 0:
            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
            initial_state[~has_initial_state, ...] = 0
            if trace_active:
                _qwen35_trace_tensor(f"{trace_prefix}.state.initial_non_spec", initial_state)
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = self.chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=non_spec_query_start_loc,
                use_qk_l2norm_in_kernel=True,
            )
            if trace_active:
                _qwen35_trace_tensor(f"{trace_prefix}.recurrent.last_state_prefill", last_recurrent_state)
            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(
                ssm_state.dtype
            )
        elif attn_metadata.num_decodes > 0:
            if trace_active:
                _qwen35_trace_tensor(f"{trace_prefix}.state.decode_ssm_before", ssm_state)
            core_attn_out_non_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[
                        : attn_metadata.num_decodes + 1
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            )
            if trace_active:
                _qwen35_trace_tensor(f"{trace_prefix}.recurrent.last_state_decode", last_recurrent_state)
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        if trace_active:
            _qwen35_trace_tensor(f"{trace_prefix}.recurrent.core_non_spec", core_attn_out_non_spec)
            _qwen35_trace_tensor(f"{trace_prefix}.recurrent.core_spec", core_attn_out_spec)
            _qwen35_trace_tensor(f"{trace_prefix}.state.ssm_after_recurrent", ssm_state)

        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

        if trace_active:
            _qwen35_trace_tensor(f"{trace_prefix}.output.core_attn_out", core_attn_out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        trace_base = getattr(
            self,
            "_gguf_parent_trace_label",
            getattr(self, "prefix", "linear_attn"),
        )
        trace_prefix = f"{trace_base}.linear_attn_detail"
        trace_call_idx = int(getattr(self, "_gguf_parent_call_idx", 0))
        trace_max = int(os.environ.get("VLLM_GFX906_CALL_TRACE_MAX_EVENTS", "3"))
        trace_active = (
            os.environ.get("VLLM_GFX906_INTERNAL_BOUNDARY_TRACE", "1") == "1"
            and _qwen35_boundary_trace_dir(trace_prefix) is not None
            and trace_call_idx < trace_max
        )
        if trace_active:
            _qwen35_trace_tensor(f"{trace_prefix}.input_hidden", hidden_states)

        mixed_qkvzba, _ = self.in_proj_qkvz(hidden_states)
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        ba_size = self.num_v_heads // self.tp_size
        mixed_qkv, z, b, a = mixed_qkvzba.split(
            [qkv_size, z_size, ba_size, ba_size], dim=-1
        )
        if trace_active:
            q_size = self.key_dim // self.tp_size
            k_size = self.key_dim // self.tp_size
            v_size = self.value_dim // self.tp_size
            q_part, k_part, v_part = mixed_qkv.split(
                [q_size, k_size, v_size], dim=-1
            )
            _qwen35_trace_tensor(f"{trace_prefix}.mixed_qkvzba", mixed_qkvzba)
            _qwen35_trace_tensor(f"{trace_prefix}.mixed_qkv", mixed_qkv)
            _qwen35_trace_tensor(f"{trace_prefix}.q", q_part)
            _qwen35_trace_tensor(f"{trace_prefix}.k", k_part)
            _qwen35_trace_tensor(f"{trace_prefix}.v", v_part)
            _qwen35_trace_tensor(f"{trace_prefix}.z_pre_reshape", z)
            _qwen35_trace_tensor(f"{trace_prefix}.b", b)
            _qwen35_trace_tensor(f"{trace_prefix}.a", a)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.contiguous()
        a = a.contiguous()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
        )
        if trace_active:
            _qwen35_trace_tensor(f"{trace_prefix}.core_attn_out_raw", core_attn_out)

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        if trace_active:
            _qwen35_trace_tensor(f"{trace_prefix}.z_for_norm", z)
        core_attn_out = self.norm(core_attn_out, z)
        if trace_active:
            _qwen35_trace_tensor(f"{trace_prefix}.core_attn_out_normed", core_attn_out)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        if trace_active:
            _qwen35_trace_tensor(f"{trace_prefix}.out_proj_input", core_attn_out)
        output[:num_tokens], _ = self.out_proj(core_attn_out)
        if trace_active:
            _qwen35_trace_tensor(f"{trace_prefix}.out_proj_output", output[:num_tokens])



def _qwen35_hf_moe_layer_alias_postload(model: nn.Module) -> None:
    if os.environ.get("VLLM_QWEN35_GGUF_HF_MOE_LAYERNAME_ALIAS") != "1":
        return
    vllm_config = getattr(model, "vllm_config", None)
    if vllm_config is None:
        return
    compilation_config = vllm_config.compilation_config
    ctx = compilation_config.static_forward_context
    all_moe_layers = compilation_config.static_all_moe_layers
    root_model = getattr(model, "model", None)
    layers = getattr(root_model, "layers", [])
    for layer in layers:
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "experts"):
            continue
        experts = layer.mlp.experts
        original = getattr(experts, "layer_name", None)
        if not isinstance(original, str) or not original.startswith("model."):
            continue
        alias = "language_model." + original
        if ctx.get(original) is experts:
            del ctx[original]
        ctx[alias] = experts
        for i, name in enumerate(list(all_moe_layers)):
            if name == original:
                all_moe_layers[i] = alias
        experts.layer_name = alias
        runner = getattr(experts, "runner", None)
        if runner is not None and hasattr(runner, "layer_name"):
            runner.layer_name = alias
        print("GGUF_QWEN35_HF_MOE_LAYER_ALIAS", original, "->", alias, flush=True)

class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        speculative_config = vllm_config.speculative_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        self.prefix = prefix

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                speculative_config=speculative_config,
                prefix=f"{prefix}.linear_attn",
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
            if os.environ.get("VLLM_QWEN35_GGUF_RUNTIME_NORM_OFFSET_FIX") == "1" or os.environ.get("VLLM_QWEN35_GGUF_LOADTIME_NORM_OFFSET_FIX") == "1":
                self.self_attn.q_norm = Qwen3_5RMSNorm(
                    self.self_attn.head_dim, eps=config.rms_norm_eps
                )
                self.self_attn.k_norm = Qwen3_5RMSNorm(
                    self.self_attn.head_dim, eps=config.rms_norm_eps
                )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # NOTE: Determine the MLP type based on the model type
        # Qwen3.5 use all layers for MLP / Qwen3.5-MoE use sparse MoE blocks
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        trace_prefix = getattr(self, "prefix", "decoder")
        trace_call_idx = getattr(self, "_gguf_decoder_trace_call_idx", 0)
        trace_max = int(os.environ.get("VLLM_GFX906_CALL_TRACE_MAX_EVENTS", "3"))
        trace_layer_filter_env = os.environ.get("VLLM_GFX906_CALL_TRACE_LAYERS", "")
        trace_layers = set()
        for raw_part in trace_layer_filter_env.split(","):
            raw_part = raw_part.strip()
            if not raw_part:
                continue
            try:
                trace_layers.add(int(raw_part))
            except ValueError:
                pass
        trace_layer_idx = getattr(self, "layer_idx", -1)
        trace_layer_ok = not trace_layers or trace_layer_idx in trace_layers
        trace_active = (
            os.environ.get("VLLM_GFX906_INTERNAL_BOUNDARY_TRACE", "1") == "1"
            and _qwen35_boundary_trace_dir(trace_prefix) is not None
            and trace_call_idx < trace_max
            and trace_layer_ok
        )
        trace_label = f"{trace_prefix}.call{trace_call_idx}"
        if trace_active:
            _qwen35_trace_tensor(f"{trace_label}.pre_input_layernorm", hidden_states)
            _qwen35_trace_tensor(
                f"{trace_label}.input_layernorm_weight",
                getattr(self.input_layernorm, "weight", None),
            )
            if residual is not None:
                _qwen35_trace_tensor(f"{trace_label}.pre_residual", residual)

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if trace_active:
            _qwen35_trace_tensor(f"{trace_label}.post_input_layernorm", hidden_states)

        self_attention_output = torch.empty_like(hidden_states)
        if self.layer_type == "linear_attention":
            try:
                setattr(self.linear_attn, "_gguf_parent_trace_label", trace_label)
                setattr(self.linear_attn, "_gguf_parent_call_idx", trace_call_idx)
            except Exception:
                pass
            self.linear_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
            )
        elif self.layer_type == "full_attention":
            self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output

        if trace_active:
            _qwen35_trace_tensor(f"{trace_label}.post_attention", hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if trace_active:
            _qwen35_trace_tensor(
                f"{trace_label}.post_attention_layernorm_weight",
                getattr(self.post_attention_layernorm, "weight", None),
            )
            _qwen35_trace_tensor(f"{trace_label}.post_attention_layernorm", hidden_states)
        hidden_states = self.mlp(hidden_states)

        if trace_active:
            _qwen35_trace_tensor(f"{trace_label}.post_mlp", hidden_states)
            self._gguf_decoder_trace_call_idx = trace_call_idx + 1

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3_5Model(Qwen3NextModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3NextModel, self).__init__()

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = (
            vllm_config.model_config.hf_text_config
        )
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3_5DecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        trace_active = (
            os.environ.get("VLLM_GFX906_MODEL_LOOP_TRACE") == "1"
            and _qwen35_boundary_trace_dir("model.layers.0.loop_pre") is not None
        )
        layer_filter_env = os.environ.get("VLLM_GFX906_MODEL_LOOP_TRACE_LAYERS", "")
        trace_layers = set()
        for raw_part in layer_filter_env.split(","):
            raw_part = raw_part.strip()
            if not raw_part:
                continue
            try:
                trace_layers.add(int(raw_part))
            except ValueError:
                pass

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            layer_idx = getattr(layer, "layer_idx", -1)
            should_trace_layer = trace_active and (
                not trace_layers or layer_idx in trace_layers
            ) and not getattr(layer, "_gguf_model_loop_trace_done", False)
            if should_trace_layer:
                _qwen35_trace_tensor(f"model.layers.{layer_idx}.loop_pre", hidden_states)
                if residual is not None:
                    _qwen35_trace_tensor(f"model.layers.{layer_idx}.loop_residual_pre", residual)
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            if should_trace_layer:
                _qwen35_trace_tensor(f"model.layers.{layer_idx}.loop_post", hidden_states)
                if residual is not None:
                    _qwen35_trace_tensor(f"model.layers.{layer_idx}.loop_residual_post", residual)
                layer._gguf_model_loop_trace_done = True

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if trace_active and not getattr(self, "_gguf_model_loop_final_trace_done", False):
            _qwen35_trace_tensor("model.loop_final_norm", hidden_states)
            self._gguf_model_loop_final_trace_done = True
        return hidden_states

    def load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        param = params_dict[name]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        loaded_local_expert = False
        for expert_id in range(num_experts):
            if is_gguf_weight and loaded_weight.dim() == 3:
                curr_expert_weight = loaded_weight
            else:
                curr_expert_weight = loaded_weight[expert_id]
            success = weight_loader(
                param,
                curr_expert_weight,
                name,
                shard_id,
                expert_id,
                return_success=True,
            )
            if success:
                loaded_local_expert = True
                if is_gguf_weight and loaded_weight.dim() == 3:
                    break

        return loaded_local_expert

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # self attention
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # mlp
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN
            ("in_proj_qkvz", "in_proj_qkv", (0, 1, 2)),
            ("in_proj_qkvz", "in_proj_z", 3),
            ("in_proj_qkvz", "in_proj_b", 4),
            ("in_proj_qkvz", "in_proj_a", 5),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        is_fused_expert = False
        has_gguf_moe_qweight = any(
            param_name.endswith("experts.w13_qweight")
            for param_name in params_dict
        )
        if has_gguf_moe_qweight:
            fused_expert_params_mapping = [
                ("experts.w13_qweight", "experts.gate_up_proj", 0, "w13"),
                ("experts.w13_qweight", "experts.gate_proj", 0, "w1"),
                ("experts.w13_qweight", "experts.up_proj", 0, "w3"),
                ("experts.w2_qweight", "experts.down_proj", 0, "w2"),
            ]
        else:
            fused_expert_params_mapping = [
                ("experts.w13_weight", "experts.gate_up_proj", 0, "w13"),
                ("experts.w13_weight", "experts.gate_proj", 0, "w1"),
                ("experts.w13_weight", "experts.up_proj", 0, "w3"),
                ("experts.w2_weight", "experts.down_proj", 0, "w2"),
            ]
        num_experts = (
            self.config.num_experts if hasattr(self.config, "num_experts") else 0
        )
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            # Remapping the name of FP8 kv-scale.
            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if any(
                    expert_name in name
                    for expert_name in (
                        "experts.gate_up_proj",
                        "experts.gate_proj",
                        "experts.up_proj",
                        "experts.down_proj",
                    )
                ):
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                loaded_weight = _qwen35_unpermute_linear_attention_gguf_weight(
                    name, loaded_weight, self.config
                )
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # name = apply_attn_prefix(name, params_dict)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if is_fused_expert:
                        if name_mapped not in params_dict:
                            continue
                        # qwen3.5 no need to transpose
                        # loaded_weight = loaded_weight.transpose(-1, -2)
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            success_w1 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            success_w3 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                            success = success_w1 and success_w3
                        else:
                            success = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                        if success:
                            name = name_mapped
                            break
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if (
                            name_mapped.endswith(".bias")
                            or name_mapped.endswith("_bias")
                        ) and name_mapped not in params_dict:
                            continue
                        param = params_dict[name_mapped]
                        weight_loader = param.weight_loader
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"
                        )
                        continue
                    loaded_weight = _qwen35_unpermute_linear_attention_gguf_weight(
                        name, loaded_weight, self.config
                    )
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsLoRA,
    SupportsPP,
):
    supports_mrope = True

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # Experimental GGUF MoE parity path: load qkv, z, b, and a into
        # the same fused projection shape used by the HF release path.
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.5 currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def get_mrope_input_positions(
        self, input_tokens: list[int], mm_features: list[object]
    ) -> tuple[torch.Tensor, int]:
        # Text-only Qwen3.5-MoE uses the same token index for T/H/W M-RoPE axes.
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        model_trace_idx = getattr(self, "_gguf_boundary_model_trace_call_idx", 0)
        model_trace_max = int(os.environ.get("VLLM_GFX906_CALL_TRACE_MAX_EVENTS", "3"))
        if (
            _qwen35_boundary_trace_dir("model.final_hidden") is not None
            and model_trace_idx < model_trace_max
        ):
            _qwen35_trace_tensor(f"model.final_hidden.call{model_trace_idx}", hidden_states)
            self._gguf_boundary_model_trace_call_idx = model_trace_idx + 1

        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )


    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        trace_dir = os.environ.get("VLLM_GFX906_GGUF_TRACE_DIR")
        trace_gate = os.environ.get("VLLM_GFX906_GGUF_TRACE_GATE_FILE")
        if trace_gate and not os.path.exists(trace_gate):
            return logits
        trace_index = int(getattr(self, "_gguf_logits_trace_count", 0))
        self._gguf_logits_trace_count = trace_index + 1
        try:
            trace_stride = int(os.environ.get("VLLM_GFX906_GGUF_TRACE_EVERY_N", "0") or "0")
        except ValueError:
            trace_stride = 0
        try:
            trace_max = int(os.environ.get("VLLM_GFX906_GGUF_TRACE_MAX_EVENTS", "1") or "1")
        except ValueError:
            trace_max = 1
        should_trace = False
        if trace_dir and logits is not None:
            if trace_stride > 0:
                should_trace = trace_index < trace_max and (trace_index % trace_stride) == 0
            else:
                should_trace = not getattr(self, "_gguf_logits_trace_done", False)
        if should_trace:
            try:
                os.makedirs(trace_dir, exist_ok=True)
                logits_row = logits.detach().reshape(-1, logits.shape[-1])[-1].float()
                hidden_row = hidden_states.detach().reshape(
                    -1, hidden_states.shape[-1]
                )[-1].float()
                k = min(8, logits_row.numel())
                values, indices = torch.topk(logits_row, k=k)
                path = os.path.join(
                    trace_dir, f"compute_logits_pid_{os.getpid()}.txt"
                )
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(f"event=compute_logits_call index={trace_index}\n")
                    fh.write(f"hidden_shape={tuple(hidden_states.shape)}\n")
                    fh.write(f"logits_shape={tuple(logits.shape)}\n")
                    fh.write(
                        "hidden_stats="
                        f"min:{float(hidden_row.min())} "
                        f"max:{float(hidden_row.max())} "
                        f"mean:{float(hidden_row.mean())} "
                        f"norm:{float(torch.linalg.vector_norm(hidden_row))}\n"
                    )
                    if os.environ.get("VLLM_GFX906_GGUF_TRACE_HIDDEN_VALUES") == "1":
                        fh.write("hidden_values=" + ",".join(f"{float(x):.6f}" for x in hidden_row.cpu()) + "\n")
                    fh.write(
                        "top_ids="
                        + ",".join(str(int(x)) for x in indices.cpu())
                        + "\n"
                    )
                    fh.write(
                        "top_values="
                        + ",".join(f"{float(x):.6f}" for x in values.cpu())
                        + "\n"
                    )
                    watch_ids_env = os.environ.get(
                        "VLLM_GFX906_GGUF_TRACE_WATCH_IDS", ""
                    )
                    watch_ids = []
                    for raw_part in watch_ids_env.split(","):
                        raw_part = raw_part.strip()
                        if not raw_part:
                            continue
                        try:
                            token_id = int(raw_part)
                        except ValueError:
                            continue
                        if 0 <= token_id < logits_row.numel():
                            watch_ids.append(token_id)
                    if watch_ids:
                        watch_values = []
                        watch_ranks = []
                        for token_id in watch_ids:
                            token_value = logits_row[token_id]
                            watch_values.append(float(token_value))
                            watch_ranks.append(
                                int((logits_row > token_value).sum().item()) + 1
                            )
                        fh.write(
                            "watch_ids="
                            + ",".join(str(x) for x in watch_ids)
                            + "\n"
                        )
                        fh.write(
                            "watch_values="
                            + ",".join(f"{x:.6f}" for x in watch_values)
                            + "\n"
                        )
                        fh.write(
                            "watch_ranks="
                            + ",".join(str(x) for x in watch_ranks)
                            + "\n"
                        )
                if trace_stride <= 0:
                    self._gguf_logits_trace_done = True
            except Exception as exc:
                try:
                    os.makedirs(trace_dir, exist_ok=True)
                    with open(
                        os.path.join(trace_dir, f"trace_error_pid_{os.getpid()}.txt"),
                        "a",
                        encoding="utf-8",
                    ) as fh:
                        fh.write(f"compute_logits_trace_error={type(exc).__name__}:{exc}\n")
                except Exception:
                    pass
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def _qwen35_gguf_linear_attn_state_iter(source):
            for name, loaded_weight in source:
                loaded_weight = _qwen35_apply_gguf_loadtime_norm_offset(name, loaded_weight)
                if name.endswith("linear_attn.A_log") and torch.is_floating_point(loaded_weight) and bool((loaded_weight < 0).all()):
                    loaded_weight = torch.log((-loaded_weight).to(torch.float32)).to(loaded_weight.dtype)
                yield name, loaded_weight
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        loaded = loader.load_weights(_qwen35_gguf_linear_attn_state_iter(weights))
        if os.environ.get("VLLM_QWEN35_GGUF_LM_HEAD_UNQUANT") == "1":
            from vllm.model_executor.layers.vocab_parallel_embedding import (
                UnquantizedEmbeddingMethod,
            )

            if hasattr(self, "lm_head") and not isinstance(self.lm_head, PPMissingLayer):
                self.lm_head.quant_method = UnquantizedEmbeddingMethod()
                print(
                    "GGUF_QWEN35_LM_HEAD_UNQUANTIZED",
                    getattr(self.lm_head, "weight", None).device if hasattr(getattr(self, "lm_head", None), "weight") else "unknown",
                    flush=True,
                )
        _qwen35_hf_moe_layer_alias_postload(self)
        _qwen35_norm_param_audit(self, "causal_lm_base")
        return loaded


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # set MoE hyperparameters
        self.set_moe_parameters()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


########################################################
# Qwen3_5-Dense
########################################################


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    # Qwen3.5 does not support multimodal pruning (EVS).
    supports_multimodal_pruning = False

    packed_modules_mapping = Qwen3VLForConditionalGeneration.packed_modules_mapping | {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        # Qwen3.5 does not support multimodal pruning (EVS).
        self.is_multimodal_pruning_enabled = False

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5ForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def recompute_mrope_positions(self, *args, **kwargs):
        raise NotImplementedError(
            "Qwen3.5 does not support multimodal pruning (EVS). "
            "recompute_mrope_positions should never be called."
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for Qwen3.5.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen3VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            intermediate_tensors: Intermediate tensors from previous pipeline
                stages.
            inputs_embeds: Pre-computed input embeddings.
            **kwargs: Additional keyword arguments including:
                - pixel_values: Pixel values to be fed to a model.
                    `None` if no images are passed.
                - image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in
                    LLM. `None` if no images are passed.
                - pixel_values_videos: Pixel values of videos to be fed to a
                    model. `None` if no videos are passed.
                - video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in
                    LLM. `None` if no videos are passed.
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        _qwen35_norm_param_audit(self, "conditional_generation")
        return loaded

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


########################################################
# Qwen3_5-MoE
########################################################


class Qwen3_5_MoeMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.language_model.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.expert_weights = []

        self.moe_layers = []
        example_moe = None
        for layer in self.language_model.model.layers:
            if isinstance(layer, Qwen3_5DecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError(
                "No Qwen3_5 layer found in the language_model.model.layers."
            )

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MoeForConditionalGeneration(
    Qwen3_5ForConditionalGeneration, Qwen3_5_MoeMixtureOfExperts
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5MoeConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        # Qwen3.5 does not support multimodal pruning (EVS).
        self.is_multimodal_pruning_enabled = False

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5MoeForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # set MoE hyperparameters
        self.set_moe_parameters()
