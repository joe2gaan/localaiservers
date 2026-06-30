"""Start deferred gfx906 persistent AR workers after vLLM graph capture.

This is intentionally loaded as ``sitecustomize`` for source experiments. It
keeps the vLLM runner file untouched while adding one narrow post-capture hook.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
from types import ModuleType


TARGET = "vllm.v1.worker.gpu_model_runner"


def _env_on(name: str) -> bool:
    return os.getenv(name, "0").lower() in {"1", "true", "yes", "on"}


def _postcapture_enabled() -> bool:
    return os.getenv(
        "VLLM_GFX906_PERSISTENT_AR_START_AFTER_CAPTURE", "0"
    ).lower() in {"1", "true", "yes", "on"}


def _prefill_enabled() -> bool:
    return _env_on("VLLM_GFX906_PERSISTENT_AR_START_AFTER_PREFILL")


def _enabled() -> bool:
    return _postcapture_enabled() or _prefill_enabled()


def _persistent_ar_enabled() -> bool:
    return _env_on("VLLM_GFX906_PERSISTENT_AR")


def _in_graph_capture() -> bool:
    return _env_on("VLLM_GFX906_PERSISTENT_AR_IN_GRAPH_CAPTURE")


def _with_persistent_ar_disabled(fn, label: str):
    def wrapped(self, *args, **kwargs):
        restore_enabled = os.environ.get("VLLM_GFX906_PERSISTENT_AR")
        restore_capture = os.environ.get("VLLM_GFX906_PERSISTENT_AR_IN_GRAPH_CAPTURE")
        os.environ["VLLM_GFX906_PERSISTENT_AR"] = "0"
        os.environ.pop("VLLM_GFX906_PERSISTENT_AR_IN_GRAPH_CAPTURE", None)
        try:
            print(
                f"[gfx906_persistent_ar_postcapture] disabled during {label}",
                file=sys.stderr,
                flush=True,
            )
            return fn(self, *args, **kwargs)
        finally:
            if restore_enabled is None:
                os.environ.pop("VLLM_GFX906_PERSISTENT_AR", None)
            else:
                os.environ["VLLM_GFX906_PERSISTENT_AR"] = restore_enabled
            if restore_capture is None:
                os.environ.pop("VLLM_GFX906_PERSISTENT_AR_IN_GRAPH_CAPTURE", None)
            else:
                os.environ["VLLM_GFX906_PERSISTENT_AR_IN_GRAPH_CAPTURE"] = (
                    restore_capture
                )

    wrapped._gfx906_profile_disable_wrapped = True
    return wrapped


def _scheduled_token_values(scheduler_output) -> list[int]:
    raw = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
    if hasattr(raw, "values"):
        values = raw.values()
    else:
        values = raw
    out = []
    for value in values:
        try:
            out.append(int(value))
        except Exception:
            pass
    return out


def _has_real_request_id(scheduler_output) -> bool:
    prefix = os.getenv(
        "VLLM_GFX906_PERSISTENT_AR_START_AFTER_PREFILL_REQ_PREFIX",
        "chatcmpl-",
    )
    if prefix in {"", "*"}:
        return True
    raw = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
    keys = raw.keys() if hasattr(raw, "keys") else []
    for req_id in keys:
        if str(req_id).startswith(prefix):
            return True
    for req in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
        req_id = getattr(req, "req_id", "")
        if str(req_id).startswith(prefix):
            return True
    return False


def _should_start_after_prefill(scheduler_output) -> bool:
    if not _prefill_enabled() or not _persistent_ar_enabled() or _in_graph_capture():
        return False
    if os.getenv("VLLM_GFX906_PERSISTENT_AR_PREFILL_STARTED", "0") == "1":
        return False
    if not _has_real_request_id(scheduler_output):
        return False
    min_tokens = int(
        os.getenv("VLLM_GFX906_PERSISTENT_AR_START_AFTER_PREFILL_MIN_TOKENS", "2")
    )
    values = _scheduled_token_values(scheduler_output)
    if values and max(values) >= min_tokens:
        return True
    total = getattr(scheduler_output, "total_num_scheduled_tokens", 0)
    try:
        return int(total) >= min_tokens
    except Exception:
        return False


def _wrap_execute_model(fn):
    def wrapped(self, scheduler_output, *args, **kwargs):
        result = fn(self, scheduler_output, *args, **kwargs)
        if not _should_start_after_prefill(scheduler_output):
            return result
        try:
            from vllm.distributed.device_communicators import gfx906_persistent_ar

            started = gfx906_persistent_ar.start_deferred_workers()
            os.environ["VLLM_GFX906_PERSISTENT_AR_PREFILL_STARTED"] = "1"
            values = _scheduled_token_values(scheduler_output)
            print(
                "[gfx906_persistent_ar_postcapture] "
                "start_after_prefill "
                f"started_deferred_workers={started} "
                f"scheduled_tokens={values}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - runtime diagnostics only
            print(
                "[gfx906_persistent_ar_postcapture] start_after_prefill failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if _env_on("VLLM_GFX906_PERSISTENT_AR_STRICT"):
                raise
        return result

    wrapped._gfx906_start_after_prefill_wrapped = True
    return wrapped


def _patch(module: ModuleType) -> None:
    if not _enabled():
        return
    cls = getattr(module, "GPUModelRunner", None)
    if cls is None:
        return
    profile_original = getattr(cls, "profile_cudagraph_memory", None)
    if profile_original is not None and not getattr(
        profile_original, "_gfx906_profile_disable_wrapped", False
    ):
        cls.profile_cudagraph_memory = _with_persistent_ar_disabled(
            profile_original, "profile_cudagraph_memory"
        )

    if _postcapture_enabled():
        original = getattr(cls, "capture_model", None)
        if original is not None and not getattr(
            original, "_gfx906_postcapture_wrapped", False
        ):

            def wrapped(self, *args, **kwargs):
                result = original(self, *args, **kwargs)
                try:
                    from vllm.distributed.device_communicators import (
                        gfx906_persistent_ar,
                    )

                    started = gfx906_persistent_ar.start_deferred_workers()
                    print(
                        "[gfx906_persistent_ar_postcapture] "
                        f"started_deferred_workers={started}",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception as exc:  # pragma: no cover - runtime diagnostics only
                    print(
                        "[gfx906_persistent_ar_postcapture] start failed: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if _env_on("VLLM_GFX906_PERSISTENT_AR_STRICT"):
                        raise
                return result

            wrapped._gfx906_postcapture_wrapped = True
            cls.capture_model = wrapped

    execute_original = getattr(cls, "execute_model", None)
    if (
        _prefill_enabled()
        and execute_original is not None
        and not getattr(execute_original, "_gfx906_start_after_prefill_wrapped", False)
    ):
        cls.execute_model = _wrap_execute_model(execute_original)


class _Loader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec):
        if hasattr(self._wrapped, "create_module"):
            return self._wrapped.create_module(spec)
        return None

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is None:
                continue
            if spec.loader is not None:
                spec.loader = _Loader(spec.loader)
            return spec
        return None


if TARGET in sys.modules:
    _patch(sys.modules[TARGET])
elif _enabled():
    sys.meta_path.insert(0, _Finder())

if _enabled():
    print(
        "[gfx906_persistent_ar_postcapture] sitecustomize active",
        file=sys.stderr,
        flush=True,
    )
