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
from collections.abc import Callable, Iterable

from gguf import GGMLQuantizationType as WeightType
import torch
from einops import rearrange
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    VllmConfig,
)
from vllm.distributed import (
    get_pp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3_5RMSNorm,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
)
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




def _qwen35_gguf_head_inverse_indices(
    n_heads: int, group: int, device: torch.device
) -> torch.Tensor:
    base = n_heads // group
    values = [(idx % group) * base + idx // group for idx in range(n_heads)]
    return torch.tensor(values, dtype=torch.long, device=device)


def _qwen35_gguf_row_inverse_indices(
    n_heads: int, group: int, head_dim: int, device: torch.device
) -> torch.Tensor:
    head_idx = _qwen35_gguf_head_inverse_indices(n_heads, group, device).tolist()
    values: list[int] = []
    for gguf_head in head_idx:
        start = gguf_head * head_dim
        values.extend(range(start, start + head_dim))
    return torch.tensor(values, dtype=torch.long, device=device)


def _maybe_apply_gguf_linear_attn_head_permutation(
    name: str, loaded_weight: torch.Tensor
) -> torch.Tensor:
    """Convert Qwen3.5 GGUF linear-attention head layout to HF/vLLM layout.

    The tested GGUF stores 48 linear-attention heads in 16-by-3 order. HF/vLLM
    expects 3-by-16 order. q and k are unaffected by this local TP4 slice, but
    v, z, beta, alpha, and output-projection input columns need reordering.
    """
    if os.environ.get("VLLM_QWEN35_GGUF_LINEAR_ATTN_HEAD_PERMUTE") != "1":
        return loaded_weight
    if not isinstance(loaded_weight, torch.Tensor) or loaded_weight.ndim != 2:
        return loaded_weight
    n_heads = 48
    group = 3
    head_dim = 128
    changed = False
    if "linear_attn.in_proj_qkv.weight" in name and loaded_weight.shape[0] == 10240:
        idx = _qwen35_gguf_row_inverse_indices(n_heads, group, head_dim, loaded_weight.device)
        loaded_weight = torch.cat(
            [loaded_weight[:4096], loaded_weight[4096:].index_select(0, idx)],
            dim=0,
        )
        changed = True
    elif "linear_attn.in_proj_z.weight" in name and loaded_weight.shape[0] == 6144:
        idx = _qwen35_gguf_row_inverse_indices(n_heads, group, head_dim, loaded_weight.device)
        loaded_weight = loaded_weight.index_select(0, idx)
        changed = True
    elif (
        ("linear_attn.in_proj_b.weight" in name or "linear_attn.in_proj_a.weight" in name)
        and loaded_weight.shape[0] == 48
    ):
        idx = _qwen35_gguf_head_inverse_indices(n_heads, group, loaded_weight.device)
        loaded_weight = loaded_weight.index_select(0, idx)
        changed = True
    elif "linear_attn.out_proj.weight" in name and loaded_weight.shape[1] == 6144:
        idx = _qwen35_gguf_row_inverse_indices(n_heads, group, head_dim, loaded_weight.device)
        loaded_weight = loaded_weight.index_select(1, idx)
        changed = True
    if changed:
        print(
            "GGUF_QWEN35_LINEAR_ATTN_HEAD_PERMUTE",
            name,
            tuple(loaded_weight.shape),
            flush=True,
        )
    return loaded_weight



def _maybe_apply_gguf_linear_attn_state_conventions(
    name: str, loaded_weight: torch.Tensor
) -> torch.Tensor:
    """Convert GGUF linear-attention recurrent-state tensors to HF/vLLM layout."""
    if os.environ.get("VLLM_QWEN35_GGUF_LINEAR_ATTN_STATE_CONVERT") != "1":
        return loaded_weight
    if not isinstance(loaded_weight, torch.Tensor):
        return loaded_weight
    n_heads = 48
    group = 3
    head_dim = 128
    changed = False
    if "linear_attn.A_log" in name and loaded_weight.ndim == 1 and loaded_weight.shape[0] == 48:
        idx = _qwen35_gguf_head_inverse_indices(n_heads, group, loaded_weight.device)
        loaded_weight = torch.log((-loaded_weight).float()).index_select(0, idx)
        changed = True
    elif "linear_attn.dt_bias" in name and loaded_weight.ndim == 1 and loaded_weight.shape[0] == 48:
        idx = _qwen35_gguf_head_inverse_indices(n_heads, group, loaded_weight.device)
        loaded_weight = loaded_weight.index_select(0, idx)
        changed = True
    elif "linear_attn.conv1d.weight" in name and loaded_weight.shape[0] == 10240:
        idx = _qwen35_gguf_row_inverse_indices(n_heads, group, head_dim, loaded_weight.device)
        loaded_weight = loaded_weight.clone()
        loaded_weight[4096:] = loaded_weight[4096:].index_select(0, idx)
        changed = True
    if changed:
        print(
            "GGUF_QWEN35_LINEAR_ATTN_STATE_CONVERT",
            name,
            tuple(loaded_weight.shape),
            flush=True,
        )
    return loaded_weight

def _maybe_apply_gguf_gemma_norm_convention(
    name: str, loaded_weight: torch.Tensor
) -> torch.Tensor:
    """Convert GGUF effective RMSNorm weights to GemmaRMSNorm storage.

    Qwen3.5 uses GemmaRMSNorm in vLLM, whose parameter stores the residual
    delta and adds 1 during forward. GGUF stores the effective norm multiplier,
    so loading it directly into GemmaRMSNorm doubles the scale.
    """
    if os.environ.get("VLLM_QWEN35_GGUF_NORM_MINUS_ONE") != "1":
        return loaded_weight
    if not isinstance(loaded_weight, torch.Tensor):
        return loaded_weight
    if loaded_weight.ndim != 1:
        return loaded_weight
    if not torch.is_floating_point(loaded_weight):
        return loaded_weight
    if "linear_attn.norm.weight" in name:
        return loaded_weight
    if not (name.endswith("norm.weight") or name.endswith("layernorm.weight")):
        return loaded_weight
    adjusted = loaded_weight - 1
    print(
        "GGUF_QWEN35_NORM_MINUS_ONE",
        name,
        tuple(loaded_weight.shape),
        flush=True,
    )
    return adjusted


def _qwen35_trace_layer_enabled(layer_idx: int) -> bool:
    if os.environ.get("VLLM_QWEN35_TRACE_LAYER0") != "1":
        return False
    trace_layer_env = os.environ.get("VLLM_QWEN35_TRACE_LAYER_INDEX", "0")
    if trace_layer_env.strip().lower() == "all":
        return True
    try:
        trace_layer_idxs = {
            int(part.strip()) for part in trace_layer_env.split(",") if part.strip()
        }
    except ValueError:
        trace_layer_idxs = {0}
    return layer_idx in trace_layer_idxs


def _qwen35_trace_rank_enabled() -> bool:
    if os.environ.get("VLLM_QWEN35_TRACE_RANK0_ONLY") != "1":
        return True
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
    except Exception:
        return True
    return True


def _qwen35_trace_rank_label() -> str:
    try:
        cuda_rank = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    except Exception:
        cuda_rank = "unknown"
    try:
        dist_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available()
            and torch.distributed.is_initialized()
            else "na"
        )
    except Exception:
        dist_rank = "unknown"
    return f"cuda={cuda_rank} dist={dist_rank}"


    if not _qwen35_trace_layer_enabled(layer_idx) or not _qwen35_trace_rank_enabled():
        return
    try:
        data = tensor.detach().float()
        flat = data.reshape(-1)
        n = int(flat.numel())
        sample_idxs = []
        for idx in (0, 1, 2, 3, 4, 5, 31, 32, 63, 64, 127, 128, n - 3, n - 2, n - 1):
            if 0 <= idx < n and idx not in sample_idxs:
                sample_idxs.append(idx)
        if sample_idxs:
            idx_tensor = torch.tensor(sample_idxs, device=flat.device, dtype=torch.long)
            samples = [round(float(x), 6) for x in flat[idx_tensor].cpu()]
        else:
            samples = []
        print(
            "GGUF_QWEN35_LAYER_DIAG",
            f"layer={layer_idx}",
            _qwen35_trace_rank_label(),
            f"label={label}",
            f"shape={tuple(tensor.shape)}",
            f"dtype={tensor.dtype}",
            f"sum={float(data.sum().cpu()):.9f}",
            f"abs={float(data.abs().sum().cpu()):.9f}",
            f"mean={float(data.mean().cpu()):.9f}",
            f"min={float(data.min().cpu()):.9f}",
            f"max={float(data.max().cpu()):.9f}",
            f"idxs={sample_idxs}",
            f"vals={samples}",
            flush=True,
        )
    except Exception as exc:
        print(
            "GGUF_QWEN35_LAYER_DIAG_ERROR",
            f"layer={layer_idx}",
            _qwen35_trace_rank_label(),
            f"label={label}",
            type(exc).__name__,
            str(exc),
            flush=True,
        )


def _materialize_f16_gguf_qweight(module: nn.Module, label: str) -> None:
    qweight = getattr(module, "qweight", None)
    weight = getattr(module, "weight", None)
    if qweight is None or weight is None:
        return
    try:
        tuple(qweight.shape)
        return
    except RuntimeError as exc:
        if "uninitialized" not in str(exc).lower():
            raise

    qweight_param = nn.Parameter(weight.detach(), requires_grad=False)
    qweight_param.tensor_shape = tuple(qweight_param.shape)
    qweight_param.is_gguf_weight = True
    qweight_param.data_container = []
    qweight_param.shard_id = []
    qweight_param.shard_id_map = {}
    module.register_parameter("qweight", qweight_param)

    qweight_type = getattr(module, "qweight_type", None)
    if qweight_type is not None:
        qweight_type.weight_type = int(WeightType.F16)
        try:
            qweight_type.data.fill_(int(WeightType.F16))
        except Exception:
            pass
    print("GGUF_QWEN35_MINIMAL_F16_MATERIALIZED", label, tuple(qweight_param.shape), int(WeightType.F16), flush=True)


class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5Config)


class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5MoeConfig)


class Qwen3_5GatedDeltaNet(Qwen3NextGatedDeltaNet):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Experimental GGUF parity path: match the HF release fused qkvzba
        # projection shape, then remove the separately-created ba projection.
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
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[num_v_heads] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        num_tokens = hidden_states.size(0)
        mixed_qkvzba, _ = self.in_proj_qkvz(hidden_states)
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        ba_size = self.num_v_heads // self.tp_size
        mixed_qkv, z, b, a = mixed_qkvzba.split(
            [qkv_size, z_size, ba_size, ba_size], dim=-1
        )
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.contiguous()
        a = a.contiguous()

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

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)


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
            self.self_attn = Qwen3_5Attention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
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


class Qwen3_5Attention(Qwen3NextAttention):
    def __init__(
        self,
        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

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
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        if os.environ.get("VLLM_QWEN35_GGUF_SKIP_EMBED_QWEIGHT") == "1":
            print(
                "GGUF_QWEN35_SKIP_EMBED_QWEIGHT_INIT model.embed_tokens",
                flush=True,
            )
        else:
            from vllm.model_executor.layers.quantization.gguf import (
                GGUFConfig,
                GGUFEmbeddingMethod,
            )

            gguf_quant_config = vllm_config.quant_config or GGUFConfig()
            if getattr(gguf_quant_config, "get_name", lambda: None)() != "gguf":
                gguf_quant_config = GGUFConfig()
            if "qweight_type" not in self.embed_tokens._parameters:
                self.embed_tokens.quant_method = GGUFEmbeddingMethod(gguf_quant_config)
                self.embed_tokens.quant_method.create_weights(
                    self.embed_tokens,
                    self.embed_tokens.embedding_dim,
                    [self.embed_tokens.num_embeddings_per_partition],
                    self.embed_tokens.embedding_dim,
                    self.embed_tokens.num_embeddings_padded,
                    params_dtype=torch.get_default_dtype(),
                    weight_loader=self.embed_tokens.weight_loader,
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
        loaded_local_expert = False
        for expert_id in range(num_experts):
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
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]
        num_experts = (
            self.config.num_experts if hasattr(self.config, "num_experts") else 0
        )
        for name, loaded_weight in weights:
            loaded_weight = _maybe_apply_gguf_linear_attn_head_permutation(name, loaded_weight)
            loaded_weight = _maybe_apply_gguf_linear_attn_state_conventions(name, loaded_weight)
            loaded_weight = _maybe_apply_gguf_gemma_norm_convention(name, loaded_weight)
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
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

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
                if (
                    shard_id in ("k", "v")
                    and getattr(param, "is_gguf_weight", False)
                ):
                    owner_module_name = name.rsplit(".", 1)[0]
                    owner_module = self.get_submodule(owner_module_name)
                    output_dim = getattr(param, "output_dim", None)
                    if (
                        output_dim is not None
                        and getattr(owner_module, "num_kv_head_replicas", 1) > 1
                    ):
                        shard_size = owner_module._get_shard_size_mapping(shard_id)
                        replica_rank = (
                            owner_module.tp_rank
                            // owner_module.num_kv_head_replicas
                        )
                        start_idx = replica_rank * shard_size
                        loaded_weight = loaded_weight.narrow(
                            output_dim, start_idx, shard_size
                        )
                        param.shard_id.append(shard_id)
                        param.shard_id_map[shard_id] = len(param.data_container)
                        param.data_container.append(loaded_weight)
                        break
                if shard_id is None:
                    weight_loader(param, loaded_weight)
                elif isinstance(shard_id, tuple):
                    owner_module_name = name.rsplit(".", 1)[0]
                    owner_module = self.get_submodule(owner_module_name)
                    if getattr(param, "is_gguf_weight_type", False):
                        for one_shard_id in shard_id:
                            weight_loader(param, loaded_weight, one_shard_id)
                    elif getattr(param, "is_gguf_weight", False):
                        output_dim = getattr(param, "output_dim", 0)
                        shard_offset = 0
                        for one_shard_id in shard_id:
                            shard_size = owner_module.output_sizes[one_shard_id]
                            loaded_weight_shard = loaded_weight.narrow(
                                output_dim, shard_offset, shard_size
                            )
                            weight_loader(param, loaded_weight_shard, one_shard_id)
                            shard_offset += shard_size
                    else:
                        weight_loader_v2 = getattr(owner_module, "weight_loader_v2", None)
                        if weight_loader_v2 is None:
                            weight_loader(param, loaded_weight, shard_id)
                        else:
                            weight_loader_v2(param, loaded_weight, shard_id)
                else:
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
                            # down_proj
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
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    if (
                        name.endswith("conv1d.weight")
                        and loaded_weight.ndim == 2
                        and getattr(param, "data", None) is not None
                        and param.data.ndim == 3
                        and param.data.shape[1] == 1
                    ):
                        loaded_weight = loaded_weight.unsqueeze(1)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        if (
            hasattr(self, "embed_tokens")
            and os.environ.get("VLLM_QWEN35_GGUF_SKIP_EMBED_QWEIGHT") != "1"
        ):
            _materialize_f16_gguf_qweight(self.embed_tokens, "EMBED")
        elif hasattr(self, "embed_tokens"):
            print(
                "GGUF_QWEN35_SKIP_EMBED_QWEIGHT_MATERIALIZE model.embed_tokens",
                flush=True,
            )
        return loaded_params


class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    IsHybrid,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # Experimental GGUF parity path: load qkv, z, b, and a into the
        # same fused projection shape used by the HF release path.
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            print(
                "GGUF_QWEN35_MINIMAL_ALLOW_MAMBA_CACHE_ALL",
                "experimental",
                flush=True,
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
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
                from vllm.model_executor.layers.quantization.gguf import (
                    GGUFConfig,
                    GGUFEmbeddingMethod,
                )

                gguf_quant_config = self.quant_config or GGUFConfig()
                if getattr(gguf_quant_config, "get_name", lambda: None)() != "gguf":
                    gguf_quant_config = GGUFConfig()
                # Leave the original dense weight parameter in place; the
                # GGUF loader will populate qweight/qweight_type and the
                # logits processor uses lm_head.quant_method.apply().
                for param_name in ("qweight", "qweight_type"):
                    if param_name in self.lm_head._parameters:
                        del self.lm_head._parameters[param_name]
                self.lm_head.quant_method = GGUFEmbeddingMethod(gguf_quant_config)
                self.lm_head.quant_method.create_weights(
                    self.lm_head,
                    self.lm_head.embedding_dim,
                    [self.lm_head.num_embeddings_per_partition],
                    self.lm_head.embedding_dim,
                    self.lm_head.num_embeddings_padded,
                    params_dtype=torch.get_default_dtype(),
                    weight_loader=self.lm_head.weight_loader,
                )
                self.lm_head.quant_config = gguf_quant_config
                self.lm_head.quant_method = GGUFEmbeddingMethod(gguf_quant_config)
                if "qweight_type" not in self.lm_head._parameters:
                    raise RuntimeError("GGUF lm_head qweight_type registration failed")
                if "qweight" not in self.lm_head._parameters:
                    raise RuntimeError("GGUF lm_head qweight registration failed")
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

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
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if hasattr(self, "lm_head") and hasattr(self.lm_head, "weight_loader"):
            from vllm.model_executor.layers.quantization.gguf import (
                GGUFConfig,
                GGUFEmbeddingMethod,
            )

            gguf_quant_config = self.quant_config or GGUFConfig()
            if getattr(gguf_quant_config, "get_name", lambda: None)() != "gguf":
                gguf_quant_config = GGUFConfig()
            for param_name in ("qweight", "qweight_type"):
                if param_name in self.lm_head._parameters:
                    del self.lm_head._parameters[param_name]
            self.lm_head.quant_method = GGUFEmbeddingMethod(gguf_quant_config)
            self.lm_head.quant_method.create_weights(
                self.lm_head,
                self.lm_head.embedding_dim,
                [self.lm_head.num_embeddings_per_partition],
                self.lm_head.embedding_dim,
                self.lm_head.num_embeddings_padded,
                params_dtype=torch.get_default_dtype(),
                weight_loader=self.lm_head.weight_loader,
            )
            self.lm_head.quant_config = gguf_quant_config
            print(
                "GGUF_QWEN35_LM_HEAD_PARAMS",
                sorted(self.lm_head._parameters.keys()),
                sorted(name for name, _ in self.lm_head.named_parameters(recurse=False)),
                flush=True,
            )
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        loaded = loader.load_weights(weights)
        if hasattr(self, "lm_head"):
            _materialize_f16_gguf_qweight(self.lm_head, "LM_HEAD")
            target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
            moved = []
            for param_name in ("qweight", "qweight_type"):
                param = getattr(self.lm_head, param_name, None)
                if param is not None and getattr(param, "device", None) is not None:
                    if param.device.type == "cpu":
                        param.data = param.data.to(target_device, non_blocking=True)
                        moved.append(param_name)
            if moved:
                print(
                    "GGUF_QWEN35_LM_HEAD_MOVED_TO_DEVICE",
                    target_device,
                    moved,
                    flush=True,
                )
            if os.environ.get("VLLM_QWEN35_GGUF_LM_HEAD_UNQUANT") == "1":
                from vllm.model_executor.layers.vocab_parallel_embedding import (
                    UnquantizedEmbeddingMethod,
                )

                self.lm_head.quant_method = UnquantizedEmbeddingMethod()
                print(
                    "GGUF_QWEN35_LM_HEAD_UNQUANTIZED",
                    target_device,
                    flush=True,
                )
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
        def language_model_weights():
            for name, weight in weights:
                if name.startswith("language_model."):
                    yield name[len("language_model.") :], weight
                elif name.startswith(("mtp.", "visual.")):
                    continue
                else:
                    yield name, weight

        loaded = self.language_model.load_weights(language_model_weights())
        return {f"language_model.{name}" for name in loaded}

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
