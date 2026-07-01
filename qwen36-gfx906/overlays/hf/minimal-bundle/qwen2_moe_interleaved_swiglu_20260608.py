# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Inference-only Qwen2MoE model compatible with HuggingFace weights."""

from collections.abc import Iterable
import importlib.util
from itertools import islice
import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen2MoeConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce_inplace,
)
try:
    from vllm.distributed.communication_op import (
        begin_tensor_model_parallel_all_reduce_with_bias,
        clear_tensor_model_parallel_all_reduce_with_bias,
        tensor_model_parallel_all_reduce_with_bias_was_consumed,
    )
except ImportError:
    def begin_tensor_model_parallel_all_reduce_with_bias(
        residual: torch.Tensor | None,
    ) -> None:
        return None

    def clear_tensor_model_parallel_all_reduce_with_bias() -> None:
        return None

    def tensor_model_parallel_all_reduce_with_bias_was_consumed() -> bool:
        return False

try:
    from vllm.distributed.communication_op import (
        begin_tensor_model_parallel_all_reduce_prefold,
        clear_tensor_model_parallel_all_reduce_prefold,
        tensor_model_parallel_all_reduce_prefold_was_consumed,
    )
except ImportError:
    def begin_tensor_model_parallel_all_reduce_prefold(
        residual: torch.Tensor | None,
    ) -> None:
        return None

    def clear_tensor_model_parallel_all_reduce_prefold() -> None:
        return None

    def tensor_model_parallel_all_reduce_prefold_was_consumed() -> bool:
        return False

from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

_GFX906_SWIGLU_EXT = None
_GFX906_SWIGLU_EXT_LOAD_ATTEMPTED = False
_GFX906_SWIGLU_FAKE_REGISTERED = False
_GFX906_ROWPAR_EXT = None
_GFX906_ROWPAR_EXT_LOAD_ATTEMPTED = False
_GFX906_ROWPAR_FAKE_REGISTERED = False


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() in ("1", "true", "yes", "on")


def _load_gfx906_swiglu_ext():
    global _GFX906_SWIGLU_EXT
    global _GFX906_SWIGLU_EXT_LOAD_ATTEMPTED
    if _GFX906_SWIGLU_EXT_LOAD_ATTEMPTED:
        return _GFX906_SWIGLU_EXT
    _GFX906_SWIGLU_EXT_LOAD_ATTEMPTED = True
    path = os.environ.get("VLLM_GFX906_QWEN_MLP_INTERLEAVED_SWIGLU_EXT_PATH", "")
    if not path:
        logger.warning_once("gfx906 interleaved SwiGLU extension path is not set")
        return None
    if not os.path.exists(path):
        logger.warning_once("gfx906 interleaved SwiGLU extension missing: %s", path)
        return None
    spec = importlib.util.spec_from_file_location(
        "gfx906_swiglu_gemv_ext_20260607", path
    )
    if spec is None or spec.loader is None:
        logger.warning_once(
            "gfx906 interleaved SwiGLU extension load spec failed: %s", path
        )
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _GFX906_SWIGLU_EXT = module
    logger.info_once("gfx906 interleaved SwiGLU extension loaded: %s", path)
    return _GFX906_SWIGLU_EXT


def _gfx906_interleaved_swiglu_impl(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    ext = _load_gfx906_swiglu_ext()
    if ext is not None:
        return torch.ops.gfx906_swiglu.interleaved(x, weight)
    if _env_truthy("VLLM_GFX906_QWEN_MLP_INTERLEAVED_SWIGLU_STRICT"):
        raise RuntimeError("gfx906 interleaved SwiGLU extension is unavailable")
    gate_up = F.linear(x, weight, None)
    return F.silu(gate_up[:, 0::2]) * gate_up[:, 1::2]


def _gfx906_interleaved_swiglu_fake(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0] // 2))


def _ensure_gfx906_swiglu_native_op() -> bool:
    global _GFX906_SWIGLU_FAKE_REGISTERED
    ext = _load_gfx906_swiglu_ext()
    if ext is None:
        if _env_truthy("VLLM_GFX906_QWEN_MLP_INTERLEAVED_SWIGLU_STRICT"):
            raise RuntimeError("gfx906 interleaved SwiGLU extension is unavailable")
        return False
    if not _GFX906_SWIGLU_FAKE_REGISTERED:
        try:
            torch.library.register_fake("gfx906_swiglu::interleaved")(
                _gfx906_interleaved_swiglu_fake
            )
        except RuntimeError as exc:
            if "already" not in str(exc).lower():
                raise
        _GFX906_SWIGLU_FAKE_REGISTERED = True
    return True


def _load_gfx906_rowpar_ext():
    global _GFX906_ROWPAR_EXT
    global _GFX906_ROWPAR_EXT_LOAD_ATTEMPTED
    if _GFX906_ROWPAR_EXT_LOAD_ATTEMPTED:
        return _GFX906_ROWPAR_EXT
    _GFX906_ROWPAR_EXT_LOAD_ATTEMPTED = True
    path = os.environ.get("VLLM_GFX906_ROWPAR_LLMM1_RESIDUAL_EXT_PATH", "")
    if not path:
        logger.warning_once("gfx906 RowParallel residual extension path is not set")
        return None
    if not os.path.exists(path):
        logger.warning_once("gfx906 RowParallel residual extension missing: %s", path)
        return None
    spec = importlib.util.spec_from_file_location(
        "gfx906_rowpar_llmm1_residual_ext_20260609", path
    )
    if spec is None or spec.loader is None:
        logger.warning_once(
            "gfx906 RowParallel residual extension load spec failed: %s", path
        )
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _GFX906_ROWPAR_EXT = module
    logger.info_once("gfx906 RowParallel residual extension loaded: %s", path)
    return _GFX906_ROWPAR_EXT


def _gfx906_rowpar_llmm1_residual_fake(
    weight: torch.Tensor,
    x: torch.Tensor,
    residual: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    del scale
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _ensure_gfx906_rowpar_native_op() -> bool:
    global _GFX906_ROWPAR_FAKE_REGISTERED
    ext = _load_gfx906_rowpar_ext()
    if ext is None:
        if _env_truthy("VLLM_GFX906_ROWPAR_LLMM1_RESIDUAL_STRICT"):
            raise RuntimeError("gfx906 RowParallel residual extension is unavailable")
        return False
    if not _GFX906_ROWPAR_FAKE_REGISTERED:
        try:
            torch.library.register_fake("gfx906_rowpar::llmm1_residual")(
                _gfx906_rowpar_llmm1_residual_fake
            )
        except RuntimeError as exc:
            if "already" not in str(exc).lower():
                raise
        _GFX906_ROWPAR_FAKE_REGISTERED = True
    return True


class Qwen2MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        expert_gate: torch.nn.Linear | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.expert_gate = expert_gate
        self._gfx906_interleaved_swiglu_enabled = (
            _env_truthy("VLLM_GFX906_QWEN_MLP_INTERLEAVED_SWIGLU")
            and quant_config is None
            and hidden_size == 5120
            and intermediate_size == 17408
            and getattr(self.gate_up_proj, "tp_size", None) == 8
            and hasattr(self.gate_up_proj, "weight")
        )
        self._gfx906_original_gate_up_weight_loader = None
        self._gfx906_native_swiglu_available = False
        self._gfx906_rowpar_residual_available = False
        if self._gfx906_interleaved_swiglu_enabled:
            self._gfx906_native_swiglu_available = _ensure_gfx906_swiglu_native_op()
            self._gfx906_original_gate_up_weight_loader = (
                self.gate_up_proj.weight.weight_loader
            )
            self.gate_up_proj.weight.weight_loader = (
                self._gfx906_interleaved_gate_up_weight_loader
            )
            logger.info_once(
                "gfx906 interleaved SwiGLU enabled for %s",
                f"{prefix}.gate_up_proj",
            )
        self._gfx906_down_residual_prefold_enabled = (
            _env_truthy("VLLM_GFX906_MLP_DOWN_LLMM1_RESIDUAL_PREFOLD")
            and quant_config is None
            and hidden_size == 5120
            and intermediate_size == 17408
            and getattr(self.down_proj, "tp_size", None) == 8
            and getattr(self.down_proj, "input_is_parallel", True)
            and hasattr(self.down_proj, "weight")
        )
        if self._gfx906_down_residual_prefold_enabled:
            self._gfx906_rowpar_residual_available = _ensure_gfx906_rowpar_native_op()
            logger.info_once(
                "gfx906 MLP down LLMM1 residual pre-fold enabled for %s",
                f"{prefix}.down_proj",
            )

    def _gfx906_interleaved_gate_up_weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | tuple[int, ...] | None = None,
    ):
        original_loader = self._gfx906_original_gate_up_weight_loader
        if (
            original_loader is None
            or loaded_shard_id not in (0, 1)
            or getattr(param, "output_dim", None) != 0
            or loaded_weight.dim() != 2
        ):
            assert original_loader is not None
            return original_loader(param, loaded_weight, loaded_shard_id)

        shard_size = self.gate_up_proj.output_sizes[loaded_shard_id]
        shard_size //= self.gate_up_proj.tp_size
        start_idx = self.gate_up_proj.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        target = param.data[loaded_shard_id::2]
        assert target.shape == loaded_weight.shape, (
            f"gfx906 interleaved SwiGLU loader shape mismatch: "
            f"target={target.shape} loaded={loaded_weight.shape} "
            f"shard_id={loaded_shard_id}"
        )
        target.copy_(loaded_weight)

    def _gfx906_interleaved_swiglu(self, x: torch.Tensor) -> torch.Tensor:
        x_view = x.reshape(-1, x.shape[-1])
        weight = self.gate_up_proj.weight
        if (
            x_view.shape[1] == 5120
            and weight.shape == (4352, 5120)
            and x_view.dtype == torch.float16
            and weight.dtype == torch.float16
            and x_view.is_cuda
            and weight.is_cuda
            and self._gfx906_native_swiglu_available
        ):
            out = torch.ops.gfx906_swiglu.interleaved(x_view.contiguous(), weight)
        else:
            gate_up = F.linear(x_view, weight, None)
            out = F.silu(gate_up[:, 0::2]) * gate_up[:, 1::2]
        return out.view(*x.shape[:-1], weight.shape[0] // 2)

    def forward(self, x):
        if self._gfx906_interleaved_swiglu_enabled:
            out = self._gfx906_interleaved_swiglu(x)
        else:
            gate_up, _ = self.gate_up_proj(x)
            out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)

        if self.expert_gate is not None:
            out = F.sigmoid(self.expert_gate(x)[0]) * out

        return out

    def forward_with_residual_prefold(
        self, x: torch.Tensor, residual: torch.Tensor | None
    ) -> tuple[torch.Tensor, bool]:
        if not self._gfx906_down_residual_prefold_enabled or residual is None:
            return self.forward(x), False
        if self.expert_gate is not None:
            return self.forward(x), False

        if self._gfx906_interleaved_swiglu_enabled:
            out = self._gfx906_interleaved_swiglu(x)
        else:
            gate_up, _ = self.gate_up_proj(x)
            out = self.act_fn(gate_up)

        out_view = out.reshape(-1, out.shape[-1])
        residual_view = residual.reshape(-1, residual.shape[-1])
        weight = self.down_proj.weight
        can_fuse = (
            self._gfx906_rowpar_residual_available
            and out_view.shape == (1, 2176)
            and residual_view.shape == (1, 5120)
            and weight.shape == (5120, 2176)
            and out_view.dtype == torch.float16
            and residual_view.dtype == torch.float16
            and weight.dtype == torch.float16
            and out_view.is_cuda
            and residual_view.is_cuda
            and weight.is_cuda
            and out_view.is_contiguous()
            and residual_view.is_contiguous()
            and weight.is_contiguous()
        )
        if not can_fuse:
            return self.forward(x), False

        scale = 1.0 / float(get_tensor_model_parallel_world_size())
        local = torch.ops.gfx906_rowpar.llmm1_residual(
            weight, out_view, residual_view, scale
        )
        reduced = tensor_model_parallel_all_reduce_inplace(local)
        return reduced.view_as(residual), True


class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: Qwen2MoeConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )

        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                prefix=f"{prefix}.shared_expert",
            )
        else:
            self.shared_expert = None

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_expert,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        if self.shared_expert is not None:
            final_hidden_states = final_hidden_states[0] + final_hidden_states[1]
        if self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(  # noqa E501
                final_hidden_states
            )

        return final_hidden_states.view(orig_shape)


class Qwen2MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2MoeConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = Qwen2MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        # Note: Qwen/Qwen2-57B-A14B-Instruct does not have
        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen2MoeSparseMoeBlock(
                config=config, quant_config=quant_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        use_mlp_down_llmm1_prefold = (
            _env_truthy("VLLM_GFX906_MLP_DOWN_LLMM1_RESIDUAL_PREFOLD")
            and isinstance(self.mlp, Qwen2MoeMLP)
        )
        use_mlp_ar_bias = (
            _env_truthy("VLLM_GFX906_MLP_AR_BIAS_ENABLE")
            and isinstance(self.mlp, Qwen2MoeMLP)
            and not use_mlp_down_llmm1_prefold
        )
        use_mlp_prefold = (
            _env_truthy("VLLM_GFX906_MLP_RESIDUAL_PREFOLD_ENABLE")
            and isinstance(self.mlp, Qwen2MoeMLP)
            and not use_mlp_ar_bias
            and not use_mlp_down_llmm1_prefold
        )
        if use_mlp_ar_bias:
            begin_tensor_model_parallel_all_reduce_with_bias(residual)
        if use_mlp_prefold:
            begin_tensor_model_parallel_all_reduce_prefold(residual)
        try:
            if use_mlp_down_llmm1_prefold:
                hidden_states, mlp_down_llmm1_prefold_consumed = (
                    self.mlp.forward_with_residual_prefold(hidden_states, residual)
                )
            else:
                hidden_states = self.mlp(hidden_states)
                mlp_down_llmm1_prefold_consumed = False
            mlp_ar_bias_consumed = (
                tensor_model_parallel_all_reduce_with_bias_was_consumed()
                if use_mlp_ar_bias
                else False
            )
            mlp_prefold_consumed = (
                tensor_model_parallel_all_reduce_prefold_was_consumed()
                if use_mlp_prefold
                else False
            )
        finally:
            if use_mlp_ar_bias:
                clear_tensor_model_parallel_all_reduce_with_bias()
            if use_mlp_prefold:
                clear_tensor_model_parallel_all_reduce_prefold()
        if mlp_ar_bias_consumed:
            residual = None
        if mlp_prefold_consumed:
            residual = None
        if mlp_down_llmm1_prefold_consumed:
            residual = None
        return hidden_states, residual


@support_torch_compile
class Qwen2MoeModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen2MoeDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
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
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        norm_output = self.norm(hidden_states, residual)
        if isinstance(norm_output, tuple):
            hidden_states, _ = norm_output
        else:
            hidden_states = norm_output
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale"
                        )
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv_scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv_scale is not loaded.",  #  noqa: E501
                                name,
                                remapped_kv_scale_name,
                            )
                            continue
                        else:
                            name = remapped_kv_scale_name
                    # GGUF: make sure that shared_expert_gate is a 2D tensor.
                    if (
                        "mlp.shared_expert_gate" in name
                        and len(loaded_weight.shape) == 1
                    ):
                        loaded_weight = loaded_weight[None, :]
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen2MoeForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    fall_back_to_pt_during_load = False
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        # Only perform the following mapping when Qwen2MoeMLP exists
        if (
            getattr(config, "mlp_only_layers", [])
            or config.shared_expert_intermediate_size > 0
        ):
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]

        self.model = Qwen2MoeModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
