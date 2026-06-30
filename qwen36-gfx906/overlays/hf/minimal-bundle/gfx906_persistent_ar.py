import atexit
import ctypes
import os
import sys
import threading
import time
from typing import Any

import torch


_libs: dict[str, ctypes.CDLL] = {}
_handles: dict[tuple[str, int, int], ctypes.c_void_p] = {}
_started_handles: set[tuple[str, int, int]] = set()
_launch_sequences: dict[tuple[str, int, int], int] = {}
_trace_files: dict[int, Any] = {}
_trace_lock = threading.Lock()
_bypass_callsite_list_cache: tuple[str, frozenset[int]] | None = None
_bypass_prefix_list_cache: tuple[str, frozenset[str]] | None = None
_bypass_prefix_occurrence_list_cache: tuple[str, frozenset[int]] | None = None
_prefix_occurrences: dict[tuple[str, int, int], int] = {}
_log_count = 0
_reject_log_count = 0
_watchdog_started = False


def _log(message: str) -> None:
    global _log_count
    limit = int(os.getenv("VLLM_GFX906_PERSISTENT_AR_LOG_LIMIT", "8"))
    if _log_count >= limit:
        return
    _log_count += 1
    print(
        f"[gfx906_persistent_ar pid={os.getpid()}] {message}",
        file=sys.stderr,
        flush=True,
    )


def _reject_add_rms(reason: str) -> bool:
    global _reject_log_count
    limit = int(os.getenv("VLLM_GFX906_PERSISTENT_AR_REJECT_LOG_LIMIT", "16"))
    if _reject_log_count < limit:
        _reject_log_count += 1
        print(
            f"[gfx906_persistent_ar pid={os.getpid()}] add_rms reject: {reason}",
            file=sys.stderr,
            flush=True,
        )
    return False


def _enabled() -> bool:
    return os.getenv("VLLM_GFX906_PERSISTENT_AR", "0").lower() in ("1", "true", "yes")


def _strict() -> bool:
    return os.getenv("VLLM_GFX906_PERSISTENT_AR_STRICT", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def _kind_base(kind: str | None = None) -> str | None:
    if not kind:
        return None
    return kind.split("|", 1)[0]


def _kind_prefix(kind: str | None = None) -> str | None:
    if not kind or "|" not in kind:
        return None
    for part in kind.split("|")[1:]:
        key, sep, value = part.partition("=")
        if sep and key.strip().lower() == "prefix":
            return value.strip()
    return None


def _route_name(kind: str | None = None) -> str:
    base = _kind_base(kind)
    if not base:
        return "default"
    normalized = "".join(ch for ch in base.lower() if ch.isalnum() or ch == "_")
    return normalized or "default"


def _route_path(kind: str | None = None) -> str:
    route = _route_name(kind)
    if route != "default":
        routed = os.getenv(f"VLLM_GFX906_PERSISTENT_AR_LIB_{route.upper()}", "")
        if routed:
            return routed
    return os.getenv("VLLM_GFX906_PERSISTENT_AR_LIB", "")


def _configured_routes() -> list[str | None]:
    routes: list[str | None] = [None]
    for route in ("attn", "mlp", "other", "callsite"):
        if os.getenv(f"VLLM_GFX906_PERSISTENT_AR_LIB_{route.upper()}", ""):
            routes.append(route)
    return routes


def _load_lib(kind: str | None = None):
    path = _route_path(kind)
    if not path:
        if _strict():
            raise RuntimeError("VLLM_GFX906_PERSISTENT_AR_LIB is not set")
        return None
    if path in _libs:
        return _libs[path]
    if not os.path.exists(path):
        if _strict():
            raise RuntimeError(f"persistent AR runtime not found: {path}")
        return None
    lib = ctypes.CDLL(path)
    lib.gfx906_persistent_ar_create.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.gfx906_persistent_ar_create.restype = ctypes.c_int
    lib.gfx906_persistent_ar_allreduce_inplace.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib.gfx906_persistent_ar_allreduce_inplace.restype = ctypes.c_int
    if hasattr(lib, "gfx906_persistent_ar_add_rms_f16"):
        lib.gfx906_persistent_ar_add_rms_f16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_void_p,
        ]
        lib.gfx906_persistent_ar_add_rms_f16.restype = ctypes.c_int
    if hasattr(lib, "gfx906_persistent_ar_add_rms_llmm1_f16"):
        lib.gfx906_persistent_ar_add_rms_llmm1_f16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.gfx906_persistent_ar_add_rms_llmm1_f16.restype = ctypes.c_int
    if hasattr(lib, "gfx906_persistent_ar_start"):
        lib.gfx906_persistent_ar_start.argtypes = [ctypes.c_void_p]
        lib.gfx906_persistent_ar_start.restype = ctypes.c_int
    if hasattr(lib, "gfx906_persistent_ar_snapshot"):
        lib.gfx906_persistent_ar_snapshot.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.c_size_t,
        ]
        lib.gfx906_persistent_ar_snapshot.restype = ctypes.c_int
    lib.gfx906_persistent_ar_destroy.argtypes = [ctypes.c_void_p]
    lib.gfx906_persistent_ar_destroy.restype = ctypes.c_int
    _libs[path] = lib
    return lib


def _defer_worker() -> bool:
    return os.getenv("GFX906_PERSISTENT_AR_DEFER_WORKER", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def _in_graph_capture() -> bool:
    return os.getenv("VLLM_GFX906_PERSISTENT_AR_IN_GRAPH_CAPTURE", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def _capture_only() -> bool:
    return os.getenv("VLLM_GFX906_PERSISTENT_AR_CAPTURE_ONLY", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def _watchdog_enabled() -> bool:
    return os.getenv("VLLM_GFX906_PERSISTENT_AR_WATCHDOG", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def _callsite_trace_enabled() -> bool:
    return os.getenv("VLLM_GFX906_PERSISTENT_AR_CALLSITE_TRACE", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        return int(raw, 0)
    except ValueError:
        if _strict():
            raise
        return default


def _bypass_kind_matches(kind: str | None) -> bool:
    target = os.getenv("VLLM_GFX906_PERSISTENT_AR_BYPASS_KIND", "").strip().lower()
    if not target or target in ("*", "all", "any"):
        return True
    return _route_name(kind) == target


def _bypass_callsite_set() -> frozenset[int]:
    global _bypass_callsite_list_cache
    raw = os.getenv("VLLM_GFX906_PERSISTENT_AR_BYPASS_CALLSITE_LIST", "").strip()
    if _bypass_callsite_list_cache is not None and _bypass_callsite_list_cache[0] == raw:
        return _bypass_callsite_list_cache[1]
    callsites: set[int] = set()
    if raw:
        for item in raw.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                callsites.add(int(item, 0))
            except ValueError:
                if _strict():
                    raise
    parsed = frozenset(callsites)
    _bypass_callsite_list_cache = (raw, parsed)
    return parsed


def _bypass_prefix_set() -> frozenset[str]:
    global _bypass_prefix_list_cache
    raw = os.getenv("VLLM_GFX906_PERSISTENT_AR_BYPASS_PREFIX_LIST", "").strip()
    if _bypass_prefix_list_cache is not None and _bypass_prefix_list_cache[0] == raw:
        return _bypass_prefix_list_cache[1]
    prefixes: set[str] = set()
    if raw:
        for item in raw.replace(";", ",").split(","):
            item = item.strip()
            if item:
                prefixes.add(item)
    parsed = frozenset(prefixes)
    _bypass_prefix_list_cache = (raw, parsed)
    return parsed


def _bypass_prefix_occurrence_set() -> frozenset[int]:
    global _bypass_prefix_occurrence_list_cache
    raw = os.getenv(
        "VLLM_GFX906_PERSISTENT_AR_BYPASS_PREFIX_OCCURRENCE_LIST", ""
    ).strip()
    if (
        _bypass_prefix_occurrence_list_cache is not None
        and _bypass_prefix_occurrence_list_cache[0] == raw
    ):
        return _bypass_prefix_occurrence_list_cache[1]
    occurrences: set[int] = set()
    if raw:
        for item in raw.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                occurrences.add(int(item, 0))
            except ValueError:
                if _strict():
                    raise
    parsed = frozenset(occurrences)
    _bypass_prefix_occurrence_list_cache = (raw, parsed)
    return parsed


def _bypass_prefix_occurrence_matches(prefix_occurrence: int | None) -> bool:
    occurrence_set = _bypass_prefix_occurrence_set()
    lo = _env_int("VLLM_GFX906_PERSISTENT_AR_BYPASS_PREFIX_OCCURRENCE_LO", -1)
    hi = _env_int("VLLM_GFX906_PERSISTENT_AR_BYPASS_PREFIX_OCCURRENCE_HI", -1)
    if not occurrence_set and lo < 0 and hi < 0:
        return True
    if prefix_occurrence is None:
        return False
    if occurrence_set and prefix_occurrence not in occurrence_set:
        return False
    if lo >= 0 and prefix_occurrence < lo:
        return False
    if hi >= 0 and prefix_occurrence > hi:
        return False
    return True


def _bypass_numel_matches(numel: int) -> bool:
    raw = os.getenv("VLLM_GFX906_PERSISTENT_AR_BYPASS_NUMEL_LIST", "").strip()
    if raw:
        for item in raw.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                if numel == int(item, 0):
                    return True
            except ValueError:
                if _strict():
                    raise
        return False

    lo = _env_int("VLLM_GFX906_PERSISTENT_AR_BYPASS_NUMEL_LO", -1)
    hi = _env_int("VLLM_GFX906_PERSISTENT_AR_BYPASS_NUMEL_HI", -1)
    if lo >= 0 and numel < lo:
        return False
    if hi >= 0 and numel > hi:
        return False
    return True


def _bypass_numel_configured() -> bool:
    if os.getenv("VLLM_GFX906_PERSISTENT_AR_BYPASS_NUMEL_LIST", "").strip():
        return True
    return (
        _env_int("VLLM_GFX906_PERSISTENT_AR_BYPASS_NUMEL_LO", -1) >= 0
        or _env_int("VLLM_GFX906_PERSISTENT_AR_BYPASS_NUMEL_HI", -1) >= 0
    )


def _bypass_callsite(
    kind: str | None,
    call_site: int,
    numel: int,
    prefix_occurrence: int | None = None,
) -> bool:
    if not _bypass_kind_matches(kind):
        return False
    if not _bypass_numel_matches(numel):
        return False

    prefix_set = _bypass_prefix_set()
    if prefix_set:
        prefix = _kind_prefix(kind)
        if prefix in prefix_set:
            return _bypass_prefix_occurrence_matches(prefix_occurrence)

    callsite_set = _bypass_callsite_set()
    if callsite_set:
        return call_site in callsite_set

    lo = _env_int("VLLM_GFX906_PERSISTENT_AR_BYPASS_CALLSITE_LO", -1)
    hi = _env_int("VLLM_GFX906_PERSISTENT_AR_BYPASS_CALLSITE_HI", -1)
    if lo >= 0 and call_site < lo:
        return False
    if hi >= 0 and call_site > hi:
        return False

    base = _env_int("VLLM_GFX906_PERSISTENT_AR_BYPASS_CALLSITE_BASE", -1)
    mod = _env_int("VLLM_GFX906_PERSISTENT_AR_BYPASS_CALLSITE_MOD", 0)
    rem = _env_int("VLLM_GFX906_PERSISTENT_AR_BYPASS_CALLSITE_REM", 0)
    if base >= 0 and mod > 0:
        return (call_site - base) % mod == rem

    return _bypass_numel_configured()


def _numel_route_matches(numel: int) -> bool:
    raw = os.getenv("VLLM_GFX906_PERSISTENT_AR_CALLSITE_ROUTE_NUMEL_LIST", "").strip()
    if raw:
        for item in raw.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                if numel == int(item, 0):
                    return True
            except ValueError:
                if _strict():
                    raise
        return False

    lo = _env_int("VLLM_GFX906_PERSISTENT_AR_CALLSITE_ROUTE_NUMEL_LO", -1)
    hi = _env_int("VLLM_GFX906_PERSISTENT_AR_CALLSITE_ROUTE_NUMEL_HI", -1)
    if lo >= 0 and numel < lo:
        return False
    if hi >= 0 and numel > hi:
        return False
    return True


def _numel_route_configured() -> bool:
    if os.getenv("VLLM_GFX906_PERSISTENT_AR_CALLSITE_ROUTE_NUMEL_LIST", "").strip():
        return True
    return (
        _env_int("VLLM_GFX906_PERSISTENT_AR_CALLSITE_ROUTE_NUMEL_LO", -1) >= 0
        or _env_int("VLLM_GFX906_PERSISTENT_AR_CALLSITE_ROUTE_NUMEL_HI", -1) >= 0
    )


def _callsite_route_matches(kind: str | None, call_site: int, numel: int) -> bool:
    if not _numel_route_matches(numel):
        return False

    target = os.getenv("VLLM_GFX906_PERSISTENT_AR_CALLSITE_ROUTE_KIND", "").strip()
    if target and target.lower() not in ("*", "all", "any"):
        if _route_name(kind) != _route_name(target):
            return False

    raw = os.getenv("VLLM_GFX906_PERSISTENT_AR_CALLSITE_ROUTE_LIST", "").strip()
    if raw:
        for item in raw.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                if call_site == int(item, 0):
                    return True
            except ValueError:
                if _strict():
                    raise
        return False

    lo = _env_int("VLLM_GFX906_PERSISTENT_AR_CALLSITE_ROUTE_LO", -1)
    hi = _env_int("VLLM_GFX906_PERSISTENT_AR_CALLSITE_ROUTE_HI", -1)
    if lo >= 0 and call_site < lo:
        return False
    if hi >= 0 and call_site > hi:
        return False

    base = _env_int("VLLM_GFX906_PERSISTENT_AR_CALLSITE_ROUTE_BASE", -1)
    mod = _env_int("VLLM_GFX906_PERSISTENT_AR_CALLSITE_ROUTE_MOD", 0)
    rem = _env_int("VLLM_GFX906_PERSISTENT_AR_CALLSITE_ROUTE_REM", 0)
    if base >= 0 and mod > 0:
        return (call_site - base) % mod == rem

    return _numel_route_configured()


def _callsite_route_kind(kind: str | None, call_site: int, numel: int) -> str | None:
    if not os.getenv("VLLM_GFX906_PERSISTENT_AR_LIB_CALLSITE", ""):
        return kind
    if _callsite_route_matches(kind, call_site, numel):
        return "callsite"
    return kind


def _trace_file_for_device(device: int):
    trace_dir = os.getenv("VLLM_GFX906_PERSISTENT_AR_CALLSITE_TRACE_DIR", "")
    if not trace_dir:
        return None
    with _trace_lock:
        handle = _trace_files.get(device)
        if handle is not None:
            return handle
        os.makedirs(trace_dir, exist_ok=True)
        path = os.path.join(
            trace_dir,
            f"callsite_kind_pid{os.getpid()}_dev{device}.csv",
        )
        is_new = not os.path.exists(path) or os.path.getsize(path) == 0
        handle = open(path, "a", buffering=1, encoding="utf-8")
        if is_new:
            handle.write(
                "pid,device,call_site,kind,route,numel,shape,capture,stream,route_path,action\n"
            )
        _trace_files[device] = handle
        return handle


def _trace_callsite(
    *,
    key: tuple[str, int, int],
    tensor: torch.Tensor,
    kind: str | None,
    stream: Any,
    call_site: int,
    action: str = "sidecar",
    route_kind: str | None = None,
) -> None:
    if not _callsite_trace_enabled():
        return
    device = key[1]
    handle = _trace_file_for_device(device)
    if handle is None:
        return
    shape = "x".join(str(dim) for dim in tuple(tensor.shape))
    normalized_kind = (kind or "default").replace(",", "_")
    route = _route_name(route_kind if route_kind is not None else kind).replace(",", "_")
    route_path = key[0].replace(",", "_")
    handle.write(
        f"{os.getpid()},{device},{call_site},{normalized_kind},{route},"
        f"{int(tensor.numel())},{shape},{int(_in_graph_capture())},"
        f"0x{int(stream.cuda_stream):x},{route_path},{action}\n"
    )


def _multirow_enabled() -> bool:
    return os.getenv("VLLM_GFX906_PERSISTENT_AR_MULTIROW", "1").lower() in (
        "1",
        "true",
        "yes",
    )


def _supported_numel(numel: int) -> bool:
    if numel == 5120:
        return True
    return _multirow_enabled() and numel in (10240, 20480, 40960)


def _comm_value(pynccl_comm: Any) -> int:
    comm = getattr(pynccl_comm, "comm", None)
    value = getattr(comm, "value", comm)
    if value is None:
        return 0
    return int(value)


def _device_index(tensor: torch.Tensor) -> int:
    index = tensor.device.index
    if index is None:
        index = torch.cuda.current_device()
    return int(index)


def _get_handle_for_device(pynccl_comm: Any, device: int, kind: str | None = None):
    lib = _load_lib(kind)
    if lib is None:
        return None
    comm_value = _comm_value(pynccl_comm)
    if comm_value == 0:
        if _strict():
            raise RuntimeError("PyNccl communicator has null comm pointer")
        return None
    route_path = _route_path(kind)
    key = (route_path, device, comm_value)
    handle = _handles.get(key)
    if handle is not None:
        return handle
    worker_nwarps = int(os.getenv("VLLM_GFX906_PERSISTENT_AR_NWARPS", "4"))
    block_threads = int(
        os.getenv("VLLM_GFX906_PERSISTENT_AR_BLOCK_THREADS", str(worker_nwarps * 64))
    )
    out = ctypes.c_void_p()
    rc = lib.gfx906_persistent_ar_create(
        ctypes.c_void_p(comm_value),
        ctypes.c_int(device),
        ctypes.c_int(worker_nwarps),
        ctypes.c_int(block_threads),
        ctypes.byref(out),
    )
    if rc != 0 or not out.value:
        if _strict():
            raise RuntimeError(f"persistent AR create failed rc={rc}")
        return None
    _handles[key] = out
    _log(
        "created handle "
        f"route={_route_name(kind)} device={device} comm=0x{comm_value:x} "
        f"nwarps={worker_nwarps} block_threads={block_threads}"
    )
    return out


def _get_handle(pynccl_comm: Any, tensor: torch.Tensor, kind: str | None = None):
    return _get_handle_for_device(pynccl_comm, _device_index(tensor), kind)


def try_all_reduce_inplace(
    pynccl_comm: Any, tensor: torch.Tensor, kind: str | None = None
) -> bool:
    if not _enabled():
        return False
    if tensor.dtype is not torch.float16:
        return False
    if not tensor.is_cuda or not tensor.is_contiguous():
        return False
    numel = int(tensor.numel())
    if not _supported_numel(numel):
        return False
    if getattr(pynccl_comm, "world_size", None) != 8:
        return False
    device = _device_index(tensor)
    comm_value = _comm_value(pynccl_comm)
    if comm_value == 0:
        if _strict():
            raise RuntimeError("PyNccl communicator has null comm pointer")
        return False
    seq_key = ("__sequence__", device, comm_value)
    call_site = _launch_sequences.get(seq_key, 0)
    if _capture_only() and not _in_graph_capture():
        return False
    stream = torch.cuda.current_stream(tensor.device)
    prefix_occurrence = None
    prefix = _kind_prefix(kind)
    if prefix:
        prefix_key = (prefix, device, comm_value)
        prefix_occurrence = _prefix_occurrences.get(prefix_key, 0)
        _prefix_occurrences[prefix_key] = prefix_occurrence + 1
    if _bypass_callsite(kind, call_site, numel, prefix_occurrence):
        route_kind = kind
        key = (_route_path(route_kind), device, comm_value)
        _launch_sequences[seq_key] = call_site + 1
        _trace_callsite(
            key=key,
            tensor=tensor,
            kind=kind,
            stream=stream,
            call_site=call_site,
            action="bypass_public",
            route_kind=route_kind,
        )
        _log(
            "bypassed allreduce "
            f"route={_route_name(kind)} device={_device_index(tensor)} "
            f"call_site={call_site} prefix_occurrence={prefix_occurrence} "
            f"numel={numel} "
            f"stream=0x{int(stream.cuda_stream):x}"
        )
        return False
    route_kind = _callsite_route_kind(kind, call_site, numel)
    handle = _get_handle_for_device(pynccl_comm, device, route_kind)
    if handle is None:
        return False
    key = (_route_path(route_kind), device, comm_value)
    if _defer_worker() and key not in _started_handles and not _in_graph_capture():
        return False
    lib = _load_lib(route_kind)
    if lib is None:
        return False
    rc = lib.gfx906_persistent_ar_allreduce_inplace(
        handle,
        ctypes.c_void_p(tensor.data_ptr()),
        ctypes.c_size_t(tensor.numel()),
        ctypes.c_void_p(stream.cuda_stream),
    )
    if rc != 0:
        if _strict():
            raise RuntimeError(f"persistent AR allreduce failed rc={rc}")
        return False
    _launch_sequences[seq_key] = call_site + 1
    _trace_callsite(
        key=key,
        tensor=tensor,
        kind=kind,
        stream=stream,
        call_site=call_site,
        action="sidecar",
        route_kind=route_kind,
    )
    _log(
        "replaced allreduce "
        f"route={_route_name(route_kind)} device={_device_index(tensor)} "
        f"call_site={call_site} "
        f"numel={numel} "
        f"stream=0x{int(stream.cuda_stream):x}"
    )
    return True


def try_add_rms_f16(
    pynccl_comm: Any,
    tensor: torch.Tensor,
    recv: torch.Tensor,
    residual_in: torch.Tensor,
    gamma: torch.Tensor,
    norm_out: torch.Tensor,
    residual_out: torch.Tensor,
    partial_sums: torch.Tensor,
    counters: torch.Tensor,
    epsilon: float,
    kind: str | None = None,
    require_enabled: bool = True,
) -> bool:
    if require_enabled and not _enabled():
        return _reject_add_rms("disabled")
    if tensor.dtype is not torch.float16 or recv.dtype is not torch.float16:
        return _reject_add_rms(
            f"tensor/recv dtype tensor={tensor.dtype} recv={recv.dtype}"
        )
    if gamma.dtype is not torch.float16 or norm_out.dtype is not torch.float16:
        return _reject_add_rms(
            f"gamma/norm dtype gamma={gamma.dtype} norm={norm_out.dtype}"
        )
    if residual_in.dtype is not torch.float32 or residual_out.dtype is not torch.float32:
        return _reject_add_rms(
            "residual dtype "
            f"in={residual_in.dtype} out={residual_out.dtype}"
        )
    if partial_sums.dtype is not torch.float32 or counters.dtype is not torch.int32:
        return _reject_add_rms(
            f"scratch dtype partial={partial_sums.dtype} counters={counters.dtype}"
        )
    tensors = (
        tensor,
        recv,
        residual_in,
        gamma,
        norm_out,
        residual_out,
        partial_sums,
        counters,
    )
    if any((not t.is_cuda) or (not t.is_contiguous()) for t in tensors):
        return _reject_add_rms(
            "device/contiguous "
            + " ".join(
                f"{idx}:cuda={t.is_cuda},contig={t.is_contiguous()}"
                for idx, t in enumerate(tensors)
            )
        )
    tensor_numel = int(tensor.numel())
    if tensor_numel == 0 or (tensor_numel % 5120) != 0:
        return _reject_add_rms(f"tensor numel={tensor_numel} not multiple of 5120")
    if int(gamma.numel()) < 5120:
        return _reject_add_rms(f"gamma numel={int(gamma.numel())} < 5120")
    for name, candidate in (
        ("recv", recv),
        ("residual_in", residual_in),
        ("norm_out", norm_out),
        ("residual_out", residual_out),
    ):
        if int(candidate.numel()) < tensor_numel:
            return _reject_add_rms(
                f"{name} numel={int(candidate.numel())} < tensor numel={tensor_numel}"
            )
    if int(partial_sums.numel()) < 3 or int(counters.numel()) < 1:
        return _reject_add_rms(
            f"scratch numel partial={int(partial_sums.numel())} counters={int(counters.numel())}"
        )
    if getattr(pynccl_comm, "world_size", None) != 8:
        return _reject_add_rms(
            f"world_size={getattr(pynccl_comm, 'world_size', None)}"
        )
    if _capture_only() and not _in_graph_capture():
        return _reject_add_rms("capture_only outside graph capture")
    device = _device_index(tensor)
    handle = _get_handle_for_device(pynccl_comm, device, kind)
    if handle is None:
        return _reject_add_rms(f"missing handle device={device} kind={kind}")
    lib = _load_lib(kind)
    if lib is None:
        return _reject_add_rms(f"missing lib kind={kind}")
    add_rms = getattr(lib, "gfx906_persistent_ar_add_rms_f16", None)
    if add_rms is None:
        if _strict():
            raise RuntimeError("persistent AR runtime does not export add_rms_f16")
        return _reject_add_rms("missing add_rms_f16 symbol")
    stream = torch.cuda.current_stream(tensor.device)
    rc = add_rms(
        handle,
        ctypes.c_void_p(tensor.data_ptr()),
        ctypes.c_void_p(recv.data_ptr()),
        ctypes.c_void_p(residual_in.data_ptr()),
        ctypes.c_void_p(gamma.data_ptr()),
        ctypes.c_void_p(norm_out.data_ptr()),
        ctypes.c_void_p(residual_out.data_ptr()),
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_void_p(counters.data_ptr()),
        ctypes.c_size_t(tensor.numel()),
        ctypes.c_float(float(epsilon)),
        ctypes.c_void_p(stream.cuda_stream),
    )
    if rc != 0:
        if _strict():
            raise RuntimeError(f"persistent AR add_rms failed rc={rc}")
        return _reject_add_rms(f"runtime rc={rc}")
    _log(
        "replaced allreduce+rms "
        f"route={_route_name(kind)} device={device} "
        f"numel={int(tensor.numel())} stream=0x{int(stream.cuda_stream):x}"
    )
    return True


def try_add_rms_llmm1_f16(
    pynccl_comm: Any,
    tensor: torch.Tensor,
    recv: torch.Tensor,
    residual_in: torch.Tensor,
    gamma: torch.Tensor,
    norm_out: torch.Tensor,
    residual_out: torch.Tensor,
    partial_sums: torch.Tensor,
    counters: torch.Tensor,
    consumer_weight: torch.Tensor,
    consumer_out: torch.Tensor,
    epsilon: float,
    kind: str | None = None,
    require_enabled: bool = True,
) -> bool:
    if require_enabled and not _enabled():
        return _reject_add_rms("disabled")
    if tensor.dtype is not torch.float16 or recv.dtype is not torch.float16:
        return _reject_add_rms(
            f"tensor/recv dtype tensor={tensor.dtype} recv={recv.dtype}"
        )
    if gamma.dtype is not torch.float16 or norm_out.dtype is not torch.float16:
        return _reject_add_rms(
            f"gamma/norm dtype gamma={gamma.dtype} norm={norm_out.dtype}"
        )
    if residual_in.dtype is not torch.float32 or residual_out.dtype is not torch.float32:
        return _reject_add_rms(
            "residual dtype "
            f"in={residual_in.dtype} out={residual_out.dtype}"
        )
    if consumer_weight.dtype is not torch.float16 or consumer_out.dtype is not torch.float16:
        return _reject_add_rms(
            "consumer dtype "
            f"weight={consumer_weight.dtype} out={consumer_out.dtype}"
        )
    if partial_sums.dtype is not torch.float32 or counters.dtype is not torch.int32:
        return _reject_add_rms(
            f"scratch dtype partial={partial_sums.dtype} counters={counters.dtype}"
        )
    tensors = (
        tensor,
        recv,
        residual_in,
        gamma,
        norm_out,
        residual_out,
        partial_sums,
        counters,
        consumer_weight,
        consumer_out,
    )
    if any((not t.is_cuda) or (not t.is_contiguous()) for t in tensors):
        return _reject_add_rms(
            "device/contiguous "
            + " ".join(
                f"{idx}:cuda={t.is_cuda},contig={t.is_contiguous()}"
                for idx, t in enumerate(tensors)
            )
        )
    tensor_numel = int(tensor.numel())
    if tensor_numel != 5120:
        return _reject_add_rms(f"add_rms_llmm1 tensor numel={tensor_numel} != 5120")
    if int(gamma.numel()) < 5120:
        return _reject_add_rms(f"gamma numel={int(gamma.numel())} < 5120")
    for name, candidate in (
        ("recv", recv),
        ("residual_in", residual_in),
        ("norm_out", norm_out),
        ("residual_out", residual_out),
    ):
        if int(candidate.numel()) < tensor_numel:
            return _reject_add_rms(
                f"{name} numel={int(candidate.numel())} < tensor numel={tensor_numel}"
            )
    if len(consumer_weight.shape) < 2 or int(consumer_weight.shape[-1]) != 5120:
        return _reject_add_rms(
            f"consumer weight shape={tuple(consumer_weight.shape)} expected K=5120"
        )
    consumer_rows = int(consumer_weight.numel() // 5120)
    if consumer_rows <= 0 or int(consumer_out.numel()) < consumer_rows:
        return _reject_add_rms(
            f"consumer rows={consumer_rows} out_numel={int(consumer_out.numel())}"
        )
    if int(partial_sums.numel()) < 3 or int(counters.numel()) < 1:
        return _reject_add_rms(
            f"scratch numel partial={int(partial_sums.numel())} counters={int(counters.numel())}"
        )
    if getattr(pynccl_comm, "world_size", None) != 8:
        return _reject_add_rms(
            f"world_size={getattr(pynccl_comm, 'world_size', None)}"
        )
    if _capture_only() and not _in_graph_capture():
        return _reject_add_rms("capture_only outside graph capture")
    device = _device_index(tensor)
    handle = _get_handle_for_device(pynccl_comm, device, kind)
    if handle is None:
        return _reject_add_rms(f"missing handle device={device} kind={kind}")
    lib = _load_lib(kind)
    if lib is None:
        return _reject_add_rms(f"missing lib kind={kind}")
    add_rms_llmm1 = getattr(lib, "gfx906_persistent_ar_add_rms_llmm1_f16", None)
    if add_rms_llmm1 is None:
        if _strict():
            raise RuntimeError("persistent AR runtime does not export add_rms_llmm1_f16")
        return _reject_add_rms("missing add_rms_llmm1_f16 symbol")
    stream = torch.cuda.current_stream(tensor.device)
    rc = add_rms_llmm1(
        handle,
        ctypes.c_void_p(tensor.data_ptr()),
        ctypes.c_void_p(recv.data_ptr()),
        ctypes.c_void_p(residual_in.data_ptr()),
        ctypes.c_void_p(gamma.data_ptr()),
        ctypes.c_void_p(norm_out.data_ptr()),
        ctypes.c_void_p(residual_out.data_ptr()),
        ctypes.c_void_p(partial_sums.data_ptr()),
        ctypes.c_void_p(counters.data_ptr()),
        ctypes.c_size_t(tensor.numel()),
        ctypes.c_float(float(epsilon)),
        ctypes.c_void_p(consumer_weight.data_ptr()),
        ctypes.c_void_p(consumer_out.data_ptr()),
        ctypes.c_int(consumer_rows),
        ctypes.c_void_p(stream.cuda_stream),
    )
    if rc != 0:
        if _strict():
            raise RuntimeError(f"persistent AR add_rms_llmm1 failed rc={rc}")
        return _reject_add_rms(f"runtime rc={rc}")
    _log(
        "replaced allreduce+rms+llmm1 "
        f"route={_route_name(kind)} device={device} "
        f"numel={int(tensor.numel())} consumer_rows={consumer_rows} "
        f"stream=0x{int(stream.cuda_stream):x}"
    )
    return True


def preinit_for_graph_capture(pynccl_comm: Any, device: int | None = None) -> bool:
    if not _enabled():
        return False
    if getattr(pynccl_comm, "world_size", None) != 8:
        return False
    if device is None:
        device = torch.cuda.current_device()
    ok = False
    for route in _configured_routes():
        handle = _get_handle_for_device(pynccl_comm, int(device), route)
        ok = ok or handle is not None
    if ok:
        _log(f"preinitialized for graph_capture device={int(device)}")
    return ok


def ensure_initialized(
    pynccl_comm: Any, tensor: torch.Tensor, kind: str | None = None
) -> bool:
    if not _enabled():
        return False
    if tensor.dtype is not torch.float16:
        return False
    if not tensor.is_cuda or not tensor.is_contiguous():
        return False
    if not _supported_numel(int(tensor.numel())):
        return False
    if getattr(pynccl_comm, "world_size", None) != 8:
        return False
    return _get_handle(pynccl_comm, tensor, kind) is not None


def start_deferred_workers() -> int:
    if not _enabled() or not _defer_worker():
        return 0
    started = 0
    for key, handle in list(_handles.items()):
        if key in _started_handles:
            continue
        if handle is None or not handle.value:
            continue
        lib = _libs.get(key[0])
        if lib is None:
            continue
        start = getattr(lib, "gfx906_persistent_ar_start", None)
        if start is None:
            if _strict():
                raise RuntimeError("persistent AR runtime does not export start")
            continue
        rc = start(handle)
        if rc != 0:
            if _strict():
                raise RuntimeError(f"persistent AR deferred start failed rc={rc}")
            continue
        _started_handles.add(key)
        started += 1
        _log(
            "started deferred worker "
            f"lib={key[0]} device={key[1]} comm=0x{key[2]:x}"
        )
    _start_watchdog_if_requested()
    return started


def _format_snapshot(key: tuple[str, int, int], values: list[int]) -> str:
    send_ptr = values[7] & ((1 << 64) - 1)
    recv_ptr = values[8] & ((1 << 64) - 1)
    channels = []
    offset = 9
    for _ in range(3):
        rank = values[offset]
        channel = values[offset + 1]
        before = values[offset + 2]
        after = values[offset + 3]
        tree = (
            values[offset + 4],
            values[offset + 5],
            values[offset + 6],
            values[offset + 7],
        )
        channels.append(
            f"ch{channel}:rank={rank} before={before} after={after} tree={tree}"
        )
        offset += 8
    return (
        f"key_lib={key[0]} key_device={key[1]} comm=0x{key[2]:x} "
        f"device={values[0]} launched={values[1]} "
        f"request={values[2]} done={values[3]} stop={values[4]} "
        f"ready={values[5]} wait={values[6]} "
        f"send=0x{send_ptr:x} recv=0x{recv_ptr:x} "
        + " ".join(channels)
    )


def _snapshot_lines() -> list[str]:
    lines = []
    for key, handle in list(_handles.items()):
        if handle is None or not handle.value:
            continue
        lib = _libs.get(key[0])
        if lib is None:
            lines.append(f"key_lib={key[0]} runtime library not loaded")
            continue
        snapshot = getattr(lib, "gfx906_persistent_ar_snapshot", None)
        if snapshot is None:
            lines.append(f"key_lib={key[0]} snapshot ABI unavailable")
            continue
        out = (ctypes.c_longlong * 33)()
        rc = snapshot(handle, out, ctypes.c_size_t(len(out)))
        if rc != 0:
            lines.append(
                f"key_lib={key[0]} key_device={key[1]} comm=0x{key[2]:x} rc={rc}"
            )
            continue
        lines.append(_format_snapshot(key, [int(x) for x in out]))
    if not lines:
        lines.append("snapshot unavailable: no handles")
    return lines


def _watchdog_loop() -> None:
    interval = float(os.getenv("VLLM_GFX906_PERSISTENT_AR_WATCHDOG_INTERVAL", "15"))
    max_snapshots = int(os.getenv("VLLM_GFX906_PERSISTENT_AR_WATCHDOG_MAX", "40"))
    for index in range(max_snapshots):
        if index > 0:
            time.sleep(interval)
        try:
            for line in _snapshot_lines():
                print(
                    f"[gfx906_persistent_ar_watchdog pid={os.getpid()} "
                    f"snapshot={index}] {line}",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as exc:  # pragma: no cover - runtime diagnostics only
            print(
                f"[gfx906_persistent_ar_watchdog pid={os.getpid()}] "
                f"snapshot failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )


def _start_watchdog_if_requested() -> None:
    global _watchdog_started
    if _watchdog_started or not _watchdog_enabled():
        return
    _watchdog_started = True
    thread = threading.Thread(
        target=_watchdog_loop,
        name="gfx906-persistent-ar-watchdog",
        daemon=True,
    )
    thread.start()
    print(
        f"[gfx906_persistent_ar_watchdog pid={os.getpid()}] started",
        file=sys.stderr,
        flush=True,
    )


def _destroy_handles() -> None:
    for handle in list(_trace_files.values()):
        try:
            handle.close()
        except Exception:
            pass
    _trace_files.clear()
    for key, handle in list(_handles.items()):
        if handle is not None and handle.value:
            lib = _libs.get(key[0])
            if lib is not None:
                lib.gfx906_persistent_ar_destroy(handle)
    _handles.clear()
    _started_handles.clear()
    _libs.clear()


atexit.register(_destroy_handles)
