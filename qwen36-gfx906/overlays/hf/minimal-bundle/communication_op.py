# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any
import os

import torch
import torch.distributed

from .parallel_state import get_tp_group

_PENDING_AR_PREFOLD: torch.Tensor | None = None
_LAST_AR_PREFOLD_CONSUMED = False


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def begin_tensor_model_parallel_all_reduce_prefold(
    residual: torch.Tensor | None,
) -> None:
    global _LAST_AR_PREFOLD_CONSUMED, _PENDING_AR_PREFOLD
    _PENDING_AR_PREFOLD = residual
    _LAST_AR_PREFOLD_CONSUMED = False


def clear_tensor_model_parallel_all_reduce_prefold() -> None:
    global _PENDING_AR_PREFOLD
    _PENDING_AR_PREFOLD = None


def tensor_model_parallel_all_reduce_prefold_was_consumed() -> bool:
    return _LAST_AR_PREFOLD_CONSUMED


def _consume_all_reduce_prefold(input_: torch.Tensor) -> torch.Tensor | None:
    global _LAST_AR_PREFOLD_CONSUMED, _PENDING_AR_PREFOLD
    residual = _PENDING_AR_PREFOLD
    if not _env_flag("VLLM_GFX906_AR_PREFOLD_ENABLE"):
        return None
    if residual is None:
        return None
    if input_.shape != residual.shape:
        return None
    if input_.dim() != 2 or input_.shape[-1] != 5120:
        return None
    if input_.dtype not in (torch.float16, torch.bfloat16):
        return None
    if not input_.is_cuda or not residual.is_cuda:
        return None
    if not input_.is_contiguous() or not residual.is_contiguous():
        if _env_flag("VLLM_GFX906_AR_PREFOLD_STRICT"):
            raise RuntimeError(
                "gfx906 residual pre-fold requires contiguous CUDA tensors: "
                f"input={tuple(input_.shape)} input_contig={input_.is_contiguous()} "
                f"residual_contig={residual.is_contiguous()}"
            )
        return None

    group = get_tp_group()
    world_size = getattr(group, "world_size", 1)
    if world_size <= 1:
        return None

    # Exact algebraic target:
    #   allreduce(partial + residual / TP) == allreduce(partial) + residual.
    # This keeps the stock fast allreduce primitive. It is intentionally
    # strict opt-in because residual dtype/rounding semantics must be validated
    # by the strict c1 gate before performance numbers are trusted.
    if residual.dtype != input_.dtype:
        residual = residual.to(dtype=input_.dtype)
    input_.add_(residual, alpha=1.0 / float(world_size))
    _PENDING_AR_PREFOLD = None
    _LAST_AR_PREFOLD_CONSUMED = True
    return group.all_reduce_inplace(input_)


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    prefold = _consume_all_reduce_prefold(input_)
    if prefold is not None:
        return prefold
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_reduce_inplace(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor in-place across model parallel group."""
    prefold = _consume_all_reduce_prefold(input_)
    if prefold is not None:
        return prefold
    return get_tp_group().all_reduce_inplace(input_)


def tensor_model_parallel_all_reduce_inplace_kind(
    input_: torch.Tensor, kind: str
) -> torch.Tensor:
    """In-place TP all-reduce with a static RowParallel family label.

    The custom op receives ``kind`` as a compile-time string argument, which lets
    gfx906 source experiments route attention and MLP reductions to different
    sidecar runtimes without using mutable Python globals during graph capture.
    """
    prefold = _consume_all_reduce_prefold(input_)
    if prefold is not None:
        return prefold

    group = get_tp_group()
    if group.world_size == 1:
        return input_
    torch.ops.vllm.all_reduce_inplace_kind(
        input_, group_name=group.unique_name, kind=kind
    )
    return input_


def tensor_model_parallel_all_reduce_inplace_public(
    input_: torch.Tensor,
) -> torch.Tensor:
    """In-place TP all-reduce that bypasses gfx906 sidecar hooks.

    This keeps non-target RowParallel families on the public RCCL path while a
    targeted sidecar experiment is active for another family.
    """
    prefold = _consume_all_reduce_prefold(input_)
    if prefold is not None:
        return prefold

    group = get_tp_group()
    if group.world_size == 1:
        return input_
    torch.ops.vllm.all_reduce_inplace_public(input_, group_name=group.unique_name)
    return input_


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
