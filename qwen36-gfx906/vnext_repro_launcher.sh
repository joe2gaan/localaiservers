#!/bin/sh
set -eu

usage() {
    cat >&2 <<'EOF'
usage: vnext_repro_launcher.sh --profile NAME [--out DIR] [--run]

Environment overrides:
  MODEL=/container/model-or-dir
  MODEL_PROBE_PATH=/host/model-or-dir
  LOCAL_MODEL_ROOT=/host/local-models
  SERVED_MODEL_NAME=name
  MODEL_FORMAT_PROBE=/path/to/model-format-probe
  PATCH_BUNDLE_PATH=/path/to/minimal-bundle
  VNEXT_PATCH_BUNDLE_MANIFEST_SHA256=<sha256>
  CHECK_REQUIRED_PATHS=0
  CHECK_PATCH_BUNDLE_PATHS=1
  ALLOW_MODEL_FORMAT_MISMATCH=1
  ALLOW_UNPINNED_PATCH_BUNDLE=1

Default action is preflight plus command generation. Use --run to exec vLLM.
EOF
}

die() {
    printf '%s\n' "error: $*" >&2
    exit 2
}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "${script_dir}/.." && pwd)

profile_name=
run_now=0
run_dir=

while [ "$#" -gt 0 ]; do
    case "$1" in
        --profile)
            [ "$#" -ge 2 ] || die "--profile needs a value"
            profile_name=$2
            shift 2
            ;;
        --out)
            [ "$#" -ge 2 ] || die "--out needs a value"
            run_dir=$2
            shift 2
            ;;
        --run)
            run_now=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[ -n "$profile_name" ] || {
    usage
    exit 2
}

profile_file="${script_dir}/profiles/vnext/${profile_name}.env"
[ -f "$profile_file" ] || die "profile not found: $profile_file"

model_probe_path_was_set=0
if [ "${MODEL_PROBE_PATH+x}" = x ]; then
    model_probe_path_was_set=1
fi

patch_bundle_path_was_set=0
if [ "${PATCH_BUNDLE_PATH+x}" = x ]; then
    patch_bundle_path_was_set=1
fi

# shellcheck disable=SC1090
. "$profile_file"

MODEL=${MODEL:-${MODEL_DEFAULT:-}}
if [ "$model_probe_path_was_set" = "1" ]; then
    MODEL_PROBE_PATH=${MODEL_PROBE_PATH:-$MODEL}
else
    case "$MODEL" in
        /opt/local-models/*)
            if [ -n "${LOCAL_MODEL_ROOT:-}" ]; then
                model_suffix=${MODEL#/opt/local-models/}
                MODEL_PROBE_PATH=$LOCAL_MODEL_ROOT/$model_suffix
            else
                MODEL_PROBE_PATH=$MODEL
            fi
            ;;
        *)
            MODEL_PROBE_PATH=$MODEL
            ;;
    esac
fi
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-${SERVED_MODEL_NAME_DEFAULT:-}}
PATCH_BUNDLE_PATH=${PATCH_BUNDLE_PATH:-${PATCH_BUNDLE_PATH_DEFAULT:-}}
if [ -n "${VNEXT_PATCH_BUNDLE_MANIFEST_SHA256:-}" ]; then
    PATCH_BUNDLE_MANIFEST_SHA256=$VNEXT_PATCH_BUNDLE_MANIFEST_SHA256
fi
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8001}
PROXY_PORT=${PROXY_PORT:-19080}
CHECK_REQUIRED_PATHS=${CHECK_REQUIRED_PATHS:-1}
CHECK_PATCH_BUNDLE_PATHS=${CHECK_PATCH_BUNDLE_PATHS:-$CHECK_REQUIRED_PATHS}
ALLOW_MODEL_FORMAT_MISMATCH=${ALLOW_MODEL_FORMAT_MISMATCH:-0}
ALLOW_UNPINNED_PATCH_BUNDLE=${ALLOW_UNPINNED_PATCH_BUNDLE:-0}

[ -n "$MODEL" ] || die "MODEL is required"
[ -n "$SERVED_MODEL_NAME" ] || die "SERVED_MODEL_NAME is required"
[ -n "${MODEL_FORMAT:-}" ] || die "profile missing MODEL_FORMAT"
[ -n "${MODEL_FAMILY:-}" ] || die "profile missing MODEL_FAMILY"
[ -n "${PROFILE:-}" ] || die "profile missing PROFILE"
[ -n "${OVERLAY:-}" ] || die "profile missing OVERLAY"
[ -n "${PATCH_TARGET_GROUPS:-}" ] || die "profile missing PATCH_TARGET_GROUPS"
[ -n "${TP_SIZE:-}" ] || die "profile missing TP_SIZE"
[ -n "${MAX_MODEL_LEN:-}" ] || die "profile missing MAX_MODEL_LEN"
[ -n "${VLLM_DTYPE:-}" ] || die "profile missing VLLM_DTYPE"
[ -n "${LOAD_FORMAT:-}" ] || die "profile missing LOAD_FORMAT"
[ -n "${RUNTIME_IMAGE:-}" ] || die "profile missing RUNTIME_IMAGE"
[ -n "${RUNTIME_IMAGE_DIGEST:-}" ] || die "profile missing RUNTIME_IMAGE_DIGEST"

if [ "$patch_bundle_path_was_set" = "0" ]; then
    case "$PATCH_BUNDLE_PATH:$OVERLAY" in
        /opt/vllm_patch_bundle:hf_release)
            PATCH_BUNDLE_PATH=${script_dir}/overlays/hf/minimal-bundle
            ;;
        /opt/vllm_patch_bundle:gguf_dense)
            PATCH_BUNDLE_PATH=${script_dir}/overlays/gguf-dense/minimal-bundle
            ;;
        /opt/vllm_patch_bundle:gguf_moe)
            PATCH_BUNDLE_PATH=${script_dir}/overlays/gguf-moe/minimal-bundle
            ;;
    esac
fi

probe=${MODEL_FORMAT_PROBE:-"${repo_root}/tools/model-format-probe/model-format-probe"}
if [ ! -x "$probe" ]; then
    if [ -f "${repo_root}/tools/model-format-probe/Makefile" ]; then
        make -C "${repo_root}/tools/model-format-probe" model-format-probe >/dev/null
    fi
fi
[ -x "$probe" ] || die "model-format-probe is not executable: $probe"

probe_output=$("$probe" "$MODEL_PROBE_PATH") || {
    printf '%s\n' "$probe_output" >&2
    die "binary model format probe failed for MODEL_PROBE_PATH=$MODEL_PROBE_PATH"
}

detected_format=$(printf '%s\n' "$probe_output" | sed -n 's/^model_format=//p' | sed -n '1p')
probe_source=$(printf '%s\n' "$probe_output" | sed -n 's/^probe_source=//p' | sed -n '1p')
probe_confidence=$(printf '%s\n' "$probe_output" | sed -n 's/^confidence=//p' | sed -n '1p')

[ -n "$detected_format" ] || die "probe did not emit model_format"

detect_model_family() {
    model_path=$1
    detected=unknown

    if [ -d "$model_path" ] && [ -f "$model_path/config.json" ]; then
        if grep -Eiq '"num_experts"|"n_routed_experts"|moe|Qwen3.*Moe|Qwen.*A3B' "$model_path/config.json"; then
            detected=qwen36_moe
        elif grep -Eiq 'Qwen3|qwen3|Qwen3_5' "$model_path/config.json"; then
            detected=qwen36_dense
        fi
    elif [ -f "$model_path" ] && [ "$detected_format" = "gguf" ]; then
        if dd if="$model_path" bs=1048576 count=64 2>/dev/null | grep -aEiq '35B-A3B|Qwen3.*Moe|qwen3.*moe|num_experts|n_routed_experts'; then
            detected=qwen36_moe
        elif dd if="$model_path" bs=1048576 count=64 2>/dev/null | grep -aEiq '27B|Qwen3|qwen3'; then
            detected=qwen36_dense
        fi
    fi

    printf '%s\n' "$detected"
}

detected_family=$(detect_model_family "$MODEL_PROBE_PATH")
if [ "$detected_family" != "unknown" ] && [ "$detected_family" != "$MODEL_FAMILY" ]; then
    die "model family mismatch: detected=$detected_family expected=$MODEL_FAMILY"
fi

if [ "$detected_format" != "$MODEL_FORMAT" ]; then
    if [ "$ALLOW_MODEL_FORMAT_MISMATCH" != "1" ]; then
        die "model format mismatch: detected=$detected_format expected=$MODEL_FORMAT"
    fi
fi

case "$LOAD_FORMAT:$detected_format" in
    gguf:gguf) ;;
    gguf:*) die "LOAD_FORMAT=gguf but detected model format is $detected_format" ;;
    auto:gguf) die "GGUF model requires LOAD_FORMAT=gguf" ;;
    auto:*) ;;
    *) die "unsupported LOAD_FORMAT=$LOAD_FORMAT" ;;
esac

[ "$MAX_MODEL_LEN" = "131072" ] || die "MAX_MODEL_LEN must be 131072, got $MAX_MODEL_LEN"
[ "$VLLM_DTYPE" = "half" ] || die "VLLM_DTYPE must be half, got $VLLM_DTYPE"

if [ "${P2P_REQUIRED:-0}" = "1" ]; then
    [ "${NCCL_P2P_DISABLE:-}" = "0" ] || die "P2P is required but NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-unset}"
fi

case "$MODEL_FAMILY:$OVERLAY" in
    qwen36_dense:gguf_moe|qwen36_moe:gguf_dense)
        die "profile family $MODEL_FAMILY cannot use overlay $OVERLAY"
        ;;
esac

case "$MODEL_FORMAT:$OVERLAY" in
    gguf:gguf_dense|gguf:gguf_moe) ;;
    gguf:*) die "GGUF model cannot use non-GGUF overlay $OVERLAY" ;;
    hf_safetensors:hf_release|hf_torch_legacy:hf_release) ;;
    hf_safetensors:gguf_*|hf_torch_legacy:gguf_*)
        die "HF/non-GGUF model cannot use GGUF overlay $OVERLAY"
        ;;
esac

case " $PATCH_TARGET_GROUPS " in
    *" common "*) ;;
    *) die "PATCH_TARGET_GROUPS must include common" ;;
esac
case "$MODEL_FORMAT:$PATCH_TARGET_GROUPS" in
    gguf:*gguf*) ;;
    gguf:*) die "GGUF profile missing gguf patch target group" ;;
    hf_safetensors:*gguf*|hf_torch_legacy:*gguf*)
        die "HF/non-GGUF profile cannot include gguf patch target group"
        ;;
esac
case "$MODEL_FAMILY:$PATCH_TARGET_GROUPS" in
    qwen36_dense:*moe*) die "dense profile cannot include moe patch target group" ;;
    qwen36_moe:*dense*) die "MoE profile cannot include dense patch target group" ;;
esac

for var in ${PROFILE_ENV_VARS:-}; do
    case "$var" in
        EXTRA_VLLM_ARGS)
            die "PROFILE_ENV_VARS must not include EXTRA_VLLM_ARGS"
            ;;
        ''|*[!A-Za-z0-9_]*)
            die "invalid profile environment variable name: $var"
            ;;
        [A-Za-z_]*) ;;
        *)
            die "invalid profile environment variable name: $var"
            ;;
    esac
    case "$var" in
        VLLM_QWEN35_GGUF_*|VLLM_GFX906_GGUF_*|VLLM_GGUF_*)
            [ "$MODEL_FORMAT" = "gguf" ] || die "GGUF env $var is not allowed for MODEL_FORMAT=$MODEL_FORMAT"
            ;;
        VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH|VLLM_QWEN36_MOE_*)
            [ "$MODEL_FAMILY" = "qwen36_moe" ] || die "MoE env $var is not allowed for MODEL_FAMILY=$MODEL_FAMILY"
            ;;
    esac
done

if [ "$MODEL_FORMAT" != "gguf" ]; then
    if env | grep -E '^(VLLM_QWEN35_GGUF_|VLLM_GFX906_GGUF_|VLLM_GGUF_)' >/dev/null 2>&1; then
        die "current environment contains GGUF-only variables for an HF/non-GGUF launch"
    fi
fi

map_container_required_path() {
    required_path=$1

    case "$required_path" in
        /root/.cache/huggingface/*)
            [ -n "${HOST_HF_CACHE:-}" ] || {
                printf '%s\n' "$required_path"
                return
            }
            suffix=${required_path#/root/.cache/huggingface/}
            printf '%s/%s\n' "$HOST_HF_CACHE" "$suffix"
            ;;
        /opt/vllm_patch_bundle/*)
            [ -n "$PATCH_BUNDLE_PATH" ] || {
                printf '%s\n' "$required_path"
                return
            }
            suffix=${required_path#/opt/vllm_patch_bundle/}
            printf '%s/%s\n' "$PATCH_BUNDLE_PATH" "$suffix"
            ;;
        /opt/local-models/*)
            [ -n "${LOCAL_MODEL_ROOT:-}" ] || {
                printf '%s\n' "$required_path"
                return
            }
            suffix=${required_path#/opt/local-models/}
            printf '%s/%s\n' "$LOCAL_MODEL_ROOT" "$suffix"
            ;;
        *)
            printf '%s\n' "$required_path"
            ;;
    esac
}

check_required_path() {
    path_label=$1
    required_path=$2

    case "$required_path" in
        /*|./*|../*)
            host_required_path=$(map_container_required_path "$required_path")
            [ -e "$host_required_path" ] || die "required ${path_label} missing: ${required_path}"
            ;;
    esac
}

if [ "$CHECK_REQUIRED_PATHS" = "1" ]; then
    [ -z "${TOKENIZER:-}" ] || check_required_path "tokenizer path" "$TOKENIZER"
    [ -z "${HF_CONFIG_PATH:-}" ] || check_required_path "HF_CONFIG_PATH" "$HF_CONFIG_PATH"
fi

if [ "${PATCH_BUNDLE_REQUIRED:-0}" = "1" ]; then
    [ -n "$PATCH_BUNDLE_PATH" ] || die "patch bundle path is required"
    if [ "$CHECK_PATCH_BUNDLE_PATHS" = "1" ]; then
        [ -d "$PATCH_BUNDLE_PATH" ] || die "patch bundle directory missing: $PATCH_BUNDLE_PATH"
        [ -f "$PATCH_BUNDLE_PATH/sha256sum.txt" ] || die "patch bundle manifest missing: $PATCH_BUNDLE_PATH/sha256sum.txt"
    fi
    if [ -z "${PATCH_BUNDLE_MANIFEST_SHA256:-}" ]; then
        [ "$ALLOW_UNPINNED_PATCH_BUNDLE" = "1" ] || die "patch bundle manifest hash is not pinned in profile"
    elif [ "$CHECK_PATCH_BUNDLE_PATHS" = "1" ]; then
        actual_manifest_hash=$(sha256sum "$PATCH_BUNDLE_PATH/sha256sum.txt" | awk '{print $1}')
        [ "$actual_manifest_hash" = "$PATCH_BUNDLE_MANIFEST_SHA256" ] || {
            die "patch bundle manifest hash mismatch: actual=$actual_manifest_hash expected=$PATCH_BUNDLE_MANIFEST_SHA256"
        }
        if grep -Eq '  /| sha256sum[.]txt$|/sha256sum[.]txt$' "$PATCH_BUNDLE_PATH/sha256sum.txt"; then
            die "patch bundle manifest must use relative paths and must not include sha256sum.txt"
        fi
        (
            cd "$PATCH_BUNDLE_PATH"
            sha256sum -c sha256sum.txt >/dev/null
        ) || die "patch bundle file hash verification failed"
    fi
fi

quote_arg() {
    case "$1" in
        *[!A-Za-z0-9_./:=,+@%-]*|'')
            printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
            ;;
        *)
            printf '%s' "$1"
            ;;
    esac
}

print_command() {
    first=1
    for arg do
        if [ "$first" = "1" ]; then
            first=0
        else
            printf ' '
        fi
        quote_arg "$arg"
    done
    printf '\n'
}

write_env_line() {
    printf '%s=' "$1"
    quote_arg "$2"
    printf '\n'
}

set -- vllm serve "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_CALL_PARSER:-hermes}" \
    --dtype "$VLLM_DTYPE" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.95}" \
    --trust-remote-code \
    --generation-config vllm \
    -O=3 \
    --reasoning-parser "${REASONING_PARSER:-qwen3}"

[ "${DISABLE_ASYNC_SCHEDULING:-0}" = "1" ] || set -- "$@" --async-scheduling
[ "$LOAD_FORMAT" = "gguf" ] && set -- "$@" --load-format gguf
[ -n "${TOKENIZER:-}" ] && set -- "$@" --tokenizer "$TOKENIZER"
[ -n "${TOKENIZER_REVISION:-}" ] && set -- "$@" --tokenizer-revision "$TOKENIZER_REVISION"
[ -n "${HF_CONFIG_PATH:-}" ] && set -- "$@" --hf-config-path "$HF_CONFIG_PATH"
[ -n "${HF_OVERRIDES_JSON:-}" ] && set -- "$@" --hf-overrides "$HF_OVERRIDES_JSON"
[ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ] && set -- "$@" --language-model-only
[ -n "${MAX_NUM_SEQS:-}" ] && set -- "$@" --max-num-seqs "$MAX_NUM_SEQS"
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && set -- "$@" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
[ "${ENABLE_PREFIX_CACHING:-1}" = "0" ] && set -- "$@" --no-enable-prefix-caching
[ -n "${COMPILATION_CONFIG:-}" ] && set -- "$@" --compilation-config "$COMPILATION_CONFIG"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir=${run_dir:-"${script_dir}/vnext-launch-runs/${profile_name}_${stamp}"}
mkdir -p "$run_dir"

{
    write_env_line profile_name "$profile_name"
    write_env_line profile "$PROFILE"
    write_env_line model "$MODEL"
    write_env_line model_probe_path "$MODEL_PROBE_PATH"
    write_env_line served_model_name "$SERVED_MODEL_NAME"
    write_env_line detected_model_format "$detected_format"
    write_env_line expected_model_format "$MODEL_FORMAT"
    write_env_line probe_source "$probe_source"
    write_env_line probe_confidence "$probe_confidence"
    write_env_line model_family "$MODEL_FAMILY"
    write_env_line detected_model_family "$detected_family"
    write_env_line selected_overlay "$OVERLAY"
    write_env_line patch_target_groups "$PATCH_TARGET_GROUPS"
    write_env_line runtime_image "$RUNTIME_IMAGE"
    write_env_line runtime_image_digest "$RUNTIME_IMAGE_DIGEST"
    write_env_line tp_size "$TP_SIZE"
    write_env_line max_model_len "$MAX_MODEL_LEN"
    write_env_line dtype "$VLLM_DTYPE"
    write_env_line p2p "$(if [ "${NCCL_P2P_DISABLE:-}" = "0" ]; then printf on; else printf off; fi)"
    write_env_line load_format "$LOAD_FORMAT"
    write_env_line patch_bundle_path "$PATCH_BUNDLE_PATH"
    write_env_line patch_bundle_manifest_sha256 "${PATCH_BUNDLE_MANIFEST_SHA256:-}"
    write_env_line benchmark_sequence '8 warmups -> c1_128 uncapped strict -> c1_2000 -> c1_10000'
} >"$run_dir/effective_config.env"

{
    printf '#!/bin/sh\n'
    printf 'set -eu\n'
    for var in ${PROFILE_ENV_VARS:-}; do
        eval "value=\${$var-}"
        [ -n "$value" ] || continue
        printf 'export %s=' "$var"
        quote_arg "$value"
        printf '\n'
    done
    print_command "$@"
} >"$run_dir/vllm_command.sh"
chmod +x "$run_dir/vllm_command.sh"

{
    for arg do
        printf '%s\n' "$arg"
    done
} >"$run_dir/vllm_argv.txt"

cat >"$run_dir/deploy_compat_patches.py" <<'EOF_DEPLOY_COMPAT_PATCHES'
from pathlib import Path
import importlib
import torch


def patch_matcher_utils_missing_c_ops() -> None:
    paths = [
        Path("/opt/venv/lib/python3.12/site-packages/vllm/compilation/passes/fusion/matcher_utils.py"),
        Path("/opt/src/vllm/vllm/compilation/passes/fusion/matcher_utils.py"),
        Path("/workspace/vllm/vllm/compilation/passes/fusion/matcher_utils.py"),
    ]

    old_ops = '''RMS_OP = torch.ops._C.rms_norm.default
RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default
ROTARY_OP = torch.ops._C.rotary_embedding.default
FLASHINFER_ROTARY_OP = torch.ops.vllm.flashinfer_rotary_embedding.default

QUANT_OPS: dict[QuantKey, OpOverload] = {
    kFp8StaticTensorSym: torch.ops._C.static_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTensorSym: torch.ops._C.dynamic_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTokenSym: torch.ops._C.dynamic_per_token_scaled_fp8_quant.default,  # noqa: E501
}

if current_platform.is_cuda() and hasattr(torch.ops._C, "scaled_fp4_quant"):
    QUANT_OPS[kNvfp4Dynamic] = torch.ops._C.scaled_fp4_quant.default  # noqa: E501

if current_platform.is_cuda():
    QUANT_OPS[kFp8Dynamic128Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501
    QUANT_OPS[kFp8Dynamic64Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501

SILU_MUL_OP = torch.ops._C.silu_and_mul.default
'''

    new_ops = '''def _vllm_gfx906_op_default(namespace: str, name: str):
    try:
        op = getattr(getattr(torch.ops, namespace), name)
        return op.default
    except AttributeError:
        return None


RMS_OP = _vllm_gfx906_op_default("_C", "rms_norm")
RMS_ADD_OP = _vllm_gfx906_op_default("_C", "fused_add_rms_norm")
ROTARY_OP = _vllm_gfx906_op_default("_C", "rotary_embedding")
FLASHINFER_ROTARY_OP = _vllm_gfx906_op_default(
    "vllm", "flashinfer_rotary_embedding")

QUANT_OPS: dict[QuantKey, OpOverload] = {}
for _key, _name in (
    (kFp8StaticTensorSym, "static_scaled_fp8_quant"),
    (kFp8DynamicTensorSym, "dynamic_scaled_fp8_quant"),
    (kFp8DynamicTokenSym, "dynamic_per_token_scaled_fp8_quant"),
):
    _op = _vllm_gfx906_op_default("_C", _name)
    if _op is not None:
        QUANT_OPS[_key] = _op

if current_platform.is_cuda():
    _op = _vllm_gfx906_op_default("_C", "scaled_fp4_quant")
    if _op is not None:
        QUANT_OPS[kNvfp4Dynamic] = _op

    _op = _vllm_gfx906_op_default("_C", "per_token_group_fp8_quant")
    if _op is not None:
        QUANT_OPS[kFp8Dynamic128Sym] = _op
        QUANT_OPS[kFp8Dynamic64Sym] = _op

SILU_MUL_OP = _vllm_gfx906_op_default("_C", "silu_and_mul")
'''

    replacements = [
        (old_ops, new_ops),
        ('''        if enabled is None:
            enabled = RotaryEmbedding.enabled()
        if match_rocm_aiter is None:
            match_rocm_aiter = rocm_aiter_ops.is_triton_rotary_embed_enabled()

        super().__init__(enabled)
''',
'''        if enabled is None:
            enabled = RotaryEmbedding.enabled()
        if match_rocm_aiter is None:
            match_rocm_aiter = rocm_aiter_ops.is_triton_rotary_embed_enabled()
        if enabled and not use_flashinfer and not match_rocm_aiter and ROTARY_OP is None:
            enabled = False
        if enabled and use_flashinfer and FLASHINFER_ROTARY_OP is None:
            enabled = False

        super().__init__(enabled)
'''),
        ('''        if enabled is None:
            enabled = RMSNorm.enabled()

        super().__init__(enabled)
        self.epsilon = epsilon
        self._rmsnorm_op = RMS_OP
        self.match_rocm_aiter = match_rocm_aiter
''',
'''        if enabled is None:
            enabled = RMSNorm.enabled()
        if enabled and not match_rocm_aiter and RMS_OP is None:
            enabled = False

        super().__init__(enabled)
        self.epsilon = epsilon
        self._rmsnorm_op = RMS_OP
        self.match_rocm_aiter = match_rocm_aiter
'''),
        ('''        if enabled is None:
            enabled = RMSNorm.enabled()

        super().__init__(enabled)
        self.epsilon = epsilon
        self.match_rocm_aiter = match_rocm_aiter

        self._rmsnorm_op = RMS_ADD_OP
''',
'''        if enabled is None:
            enabled = RMSNorm.enabled()
        if enabled and not match_rocm_aiter and RMS_ADD_OP is None:
            enabled = False

        super().__init__(enabled)
        self.epsilon = epsilon
        self.match_rocm_aiter = match_rocm_aiter

        self._rmsnorm_op = RMS_ADD_OP
'''),
        ('''        if enabled is None:
            enabled = QuantFP8.enabled()

        super().__init__(enabled)
''',
'''        if enabled is None:
            enabled = QuantFP8.enabled()
        if enabled and not match_rocm_aiter and quant_key not in QUANT_OPS:
            enabled = False

        super().__init__(enabled)
'''),
        ('''        else:
            assert quant_key in QUANT_OPS, (
                f"unsupported quantization scheme {quant_key}"
            )
            self.QUANT_OP = QUANT_OPS[quant_key]

            assert quant_key.dtype == current_platform.fp8_dtype(), (
                "Only QuantFP8 supported by"
            )
            assert quant_key.scale2 is None

        self.quant_fp8 = QuantFP8(
''',
'''        else:
            if self.enabled:
                assert quant_key in QUANT_OPS, (
                    f"unsupported quantization scheme {quant_key}"
                )
                self.QUANT_OP = QUANT_OPS[quant_key]

                assert quant_key.dtype == current_platform.fp8_dtype(), (
                    "Only QuantFP8 supported by"
                )
                assert quant_key.scale2 is None
            else:
                self.QUANT_OP = None

        self.quant_fp8 = QuantFP8(
'''),
        ('''        if enabled is None:
            enabled = SiluAndMul.enabled()
        super().__init__(enabled)
''',
'''        if enabled is None:
            enabled = SiluAndMul.enabled()
        if enabled and SILU_MUL_OP is None:
            enabled = False
        super().__init__(enabled)
'''),
    ]

    for path in paths:
        if not path.exists():
            continue
        text = path.read_text()
        if "_vllm_gfx906_op_default" in text:
            continue
        new_text = text
        for old, new in replacements:
            if old not in new_text:
                raise SystemExit(f"matcher_utils patch anchor not found in {path}")
            new_text = new_text.replace(old, new, 1)
        path.write_text(new_text)


def patch_rocm_gfx906_capability() -> None:
    paths = [
        Path("/opt/venv/lib/python3.12/site-packages/vllm/platforms/rocm.py"),
        Path("/opt/src/vllm/vllm/platforms/rocm.py"),
        Path("/workspace/vllm/vllm/platforms/rocm.py"),
    ]
    old_on_gfx9 = '_ON_GFX9 = any(arch in _GCN_ARCH for arch in ["gfx90a", "gfx942", "gfx950"])'
    new_on_gfx9 = '_ON_GFX9 = any(arch in _GCN_ARCH for arch in ["gfx9", "gfx90a", "gfx942", "gfx950"])'
    old_capability = '''    elif n == 4:
        # 2-digit major: gfx10xx, gfx11xx, gfx12xx
        # major(2) + minor(1) + stepping(1)
        major = int(digits[:2])
        minor = int(digits[2])
'''
    new_capability = '''    elif n == 4:
        # AMD SMI on gfx906/MI50 can report "gfx9006".
        # Treat gfx90xx as the gfx9 one-digit-major layout rather than
        # the gfx10/11/12 two-digit-major layout.
        if digits.startswith("90"):
            major = int(digits[0])
            minor = int(digits[1])
        else:
            # 2-digit major: gfx10xx, gfx11xx, gfx12xx
            # major(2) + minor(1) + stepping(1)
            major = int(digits[:2])
            minor = int(digits[2])
'''
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text()
        text = text.replace(old_on_gfx9, new_on_gfx9)
        if "AMD SMI on gfx906/MI50 can report" not in text:
            text = text.replace(old_capability, new_capability)
        if "def on_gfx906(" not in text:
            text = text.rstrip() + '''

def on_gfx906() -> bool:
    return "gfx906" in _GCN_ARCH or "gfx9006" in _GCN_ARCH
''' + "\n"
        path.write_text(text)


def append_if_missing(paths, marker, shim) -> None:
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text()
        if marker not in text:
            path.write_text(text.rstrip() + shim + "\n")


def patch_batch_invariant_compat() -> None:
    append_if_missing(
        [
            Path("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/batch_invariant.py"),
            Path("/opt/src/vllm/vllm/model_executor/layers/batch_invariant.py"),
            Path("/workspace/vllm/vllm/model_executor/layers/batch_invariant.py"),
        ],
        "def vllm_is_batch_invariant(",
        '''

def vllm_is_batch_invariant() -> bool:
    """Compatibility shim for MoE hotfixes from newer vLLM branches."""
    return False
''',
    )


def patch_fused_moe_utils_compat() -> None:
    append_if_missing(
        [
            Path("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/utils.py"),
            Path("/opt/src/vllm/vllm/model_executor/layers/fused_moe/utils.py"),
            Path("/workspace/vllm/vllm/model_executor/layers/fused_moe/utils.py"),
        ],
        "def disable_inplace(",
        '''

def disable_inplace() -> bool:
    """Compatibility shim for MoE hotfixes from newer vLLM branches."""
    return False
''',
    )


def replace_once(paths, old, new, marker) -> None:
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text()
        if marker in text:
            continue
        if old not in text:
            continue
        path.write_text(text.replace(old, new, 1))


def patch_activation_custom_op_fallbacks() -> None:
    replace_once(
        [
            Path("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/activation.py"),
            Path("/opt/src/vllm/vllm/model_executor/layers/activation.py"),
            Path("/workspace/vllm/vllm/model_executor/layers/activation.py"),
        ],
        '''        if current_platform.is_cuda_alike() or current_platform.is_xpu():
            self.op = torch.ops._C.silu_and_mul
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native
''',
        '''        if (
            (current_platform.is_cuda_alike() or current_platform.is_xpu())
            and hasattr(torch.ops._C, "silu_and_mul")
        ):
            self.op = torch.ops._C.silu_and_mul
        else:
            self._forward_method = self.forward_native
''',
        'hasattr(torch.ops._C, "silu_and_mul")',
    )


def patch_moe_activation_custom_op_fallbacks() -> None:
    replace_once(
        [
            Path("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/activation.py"),
            Path("/opt/src/vllm/vllm/model_executor/layers/fused_moe/activation.py"),
            Path("/workspace/vllm/vllm/model_executor/layers/fused_moe/activation.py"),
        ],
        '''    if activation == MoEActivation.SILU:
        torch.ops._C.silu_and_mul(output, input)
''',
        '''    if activation == MoEActivation.SILU:
        if hasattr(torch.ops._C, "silu_and_mul"):
            torch.ops._C.silu_and_mul(output, input)
        else:
            d = input.shape[-1] // 2
            output.copy_(F.silu(input[:, :d]) * input[:, d:])
''',
        'hasattr(torch.ops._C, "silu_and_mul")',
    )


def patch_rms_norm_custom_op_fallbacks() -> None:
    paths = [
        Path("/opt/venv/lib/python3.12/site-packages/vllm/kernels/vllm_c.py"),
        Path("/opt/src/vllm/vllm/kernels/vllm_c.py"),
        Path("/workspace/vllm/vllm/kernels/vllm_c.py"),
    ]
    rms_old = '''    assert variance_size is None
    # ROCm's vLLM C RMSNorm kernel operates on contiguous 2D tensors.
'''
    rms_new = '''    assert variance_size is None
    if not hasattr(torch.ops._C, "rms_norm"):
        return ir.ops.rms_norm.impls["native"].impl_fn(
            x, weight, epsilon, variance_size
        )
    # ROCm's vLLM C RMSNorm kernel operates on contiguous 2D tensors.
'''
    fused_old = '''    assert variance_size is None
    if IS_ROCM and (not x.is_contiguous() or not x_residual.is_contiguous()):
'''
    fused_new = '''    assert variance_size is None
    if not hasattr(torch.ops._C, "fused_add_rms_norm"):
        output, residual = ir.ops.fused_add_rms_norm.impls["native"].impl_fn(
            x, x_residual, weight, epsilon, variance_size
        )
        x.copy_(output)
        x_residual.copy_(residual)
        return x, x_residual
    if IS_ROCM and (not x.is_contiguous() or not x_residual.is_contiguous()):
'''
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text()
        if 'not hasattr(torch.ops._C, "rms_norm")' not in text:
            text = text.replace(rms_old, rms_new)
        if 'not hasattr(torch.ops._C, "fused_add_rms_norm")' not in text:
            text = text.replace(fused_old, fused_new)
        path.write_text(text)


def patch_rotary_custom_op_fallbacks() -> None:
    replace_once(
        [
            Path("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/base.py"),
            Path("/opt/src/vllm/vllm/model_executor/layers/rotary_embedding/base.py"),
            Path("/workspace/vllm/vllm/model_executor/layers/rotary_embedding/base.py"),
        ],
        '''        if self.use_aiter:
            cos_sin_cache = self._match_cos_sin_cache_dtype(query)
            self.rocm_aiter_triton_rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                cos_sin_cache,
                self.is_neox_style,
            )
            return query, key
        return self.forward_cuda(positions, query, key)
''',
        '''        if self.use_aiter:
            cos_sin_cache = self._match_cos_sin_cache_dtype(query)
            self.rocm_aiter_triton_rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                cos_sin_cache,
                self.is_neox_style,
            )
            return query, key
        if not hasattr(torch.ops._C, "rotary_embedding"):
            return self.forward_native(positions, query, key)
        return self.forward_cuda(positions, query, key)
''',
        'not hasattr(torch.ops._C, "rotary_embedding")',
    )


def report_reference_custom_op_surface() -> None:
    for module in ("vllm._C", "vllm._rocm_C"):
        try:
            importlib.import_module(module)
        except Exception:
            pass
    missing = []
    for name in ("rotary_embedding", "silu_and_mul", "rms_norm", "fused_add_rms_norm"):
        if not hasattr(torch.ops._C, name):
            missing.append(name)
    if missing:
        print(
            "info: vLLM _C op(s) missing as on the reference image: "
            + ",".join(missing),
            flush=True,
        )


patch_matcher_utils_missing_c_ops()
patch_rocm_gfx906_capability()
patch_batch_invariant_compat()
patch_fused_moe_utils_compat()
patch_activation_custom_op_fallbacks()
patch_moe_activation_custom_op_fallbacks()
patch_rms_norm_custom_op_fallbacks()
patch_rotary_custom_op_fallbacks()
report_reference_custom_op_surface()
EOF_DEPLOY_COMPAT_PATCHES

{
    write_env_line QWEN36_PROFILE "$PROFILE"
    write_env_line MODEL "$SERVED_MODEL_NAME"
    write_env_line HOST "${BENCHMARK_HOST:-127.0.0.1}"
    write_env_line PORT "$PORT"
    write_env_line PROXY_PORT "$PROXY_PORT"
    write_env_line RUNTIME_IMAGE "$RUNTIME_IMAGE"
    write_env_line RUNTIME_IMAGE_DIGEST "$RUNTIME_IMAGE_DIGEST"
    printf 'PRE_MEASURE_WARMUP_REQUESTS=8\n'
    printf 'PRE_MEASURE_WARMUP_MAX_TOKENS=2000\n'
    printf 'PROMPT_REPEAT=32\n'
    printf 'USE_BEGIN_THINK_PROXY=1\n'
} >"$run_dir/benchmark_env.sh"

cat >"$run_dir/container_entrypoint.sh" <<'EOF_CONTAINER_ENTRYPOINT'
#!/bin/sh
set -eu

bundle=${VLLM_PATCH_BUNDLE:-/opt/vllm_patch_bundle}
patch_target_groups='__PATCH_TARGET_GROUPS__'

copy_if_present() {
    src=$1
    dst=$2
    if [ -f "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        cp -f "$src" "$dst"
    fi
}

copy_tree_if_present() {
    src=$1
    dst=$2
    if [ -d "$src" ]; then
        rm -rf "$dst"
        mkdir -p "$dst"
        cp -pR "$src/." "$dst/"
    fi
}

target_group_enabled() {
    groups=$1
    old_ifs=$IFS
    IFS=','
    for group in $groups; do
        IFS=$old_ifs
        case " $patch_target_groups " in
            *" $group "*) return 0 ;;
        esac
        IFS=','
    done
    IFS=$old_ifs
    return 1
}

if [ -f "$bundle/sha256sum.txt" ]; then
    (
        cd "$bundle"
        sha256sum -c sha256sum.txt >/dev/null
    )
fi

if [ -f "$bundle/native/rccl/lib/librccl.so.1.0" ]; then
    mkdir -p /rccl-overlay/install/lib
    cp -p "$bundle/native/rccl/lib/librccl.so.1.0" /rccl-overlay/install/lib/librccl.so.1.0
    ln -sfn librccl.so.1.0 /rccl-overlay/install/lib/librccl.so.1
    ln -sfn librccl.so.1 /rccl-overlay/install/lib/librccl.so
fi
if target_group_enabled dense && [ -f "$bundle/native/sidecar_ar/libgfx906_sidecar_tree_ll_ar_multirow_count_2048_1536_1536_20260613.so" ]; then
    mkdir -p /opt/gfx906
    cp -p "$bundle/native/sidecar_ar/libgfx906_sidecar_tree_ll_ar_multirow_count_2048_1536_1536_20260613.so" /opt/gfx906/libgfx906_persistent_tree_ll_ar_default_20260613.so
fi
if target_group_enabled dense && [ -f "$bundle/native/swiglu/gfx906_swiglu_gemv_ext_20260607.so" ]; then
    mkdir -p /usr/share/ollama/kernel_labs/gfx906_swiglu_gemv_ext_native_runtime_20260608
    cp -p "$bundle/native/swiglu/gfx906_swiglu_gemv_ext_20260607.so" /usr/share/ollama/kernel_labs/gfx906_swiglu_gemv_ext_native_runtime_20260608/gfx906_swiglu_gemv_ext_20260607.so
fi
if target_group_enabled moe; then
    copy_tree_if_present "$bundle/vllm_tuned_moe_configs" /opt/vllm_tuned_moe_configs
fi

while IFS='|' read -r groups src_rel dst; do
    [ -n "$groups" ] || continue
    target_group_enabled "$groups" || continue
    [ -n "$src_rel" ] || continue
    copy_if_present "$bundle/$src_rel" "$dst"
done <<'EOF_TARGETS'
moe|shared_expert_gate.py|/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/shared_expert_gate.py
moe|shared_expert_gate.py|/opt/src/vllm/vllm/model_executor/layers/fused_moe/shared_expert_gate.py
moe|shared_expert_gate.py|/workspace/vllm/vllm/model_executor/layers/fused_moe/shared_expert_gate.py
common|utils.py|/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/utils.py
common|utils.py|/opt/src/vllm/vllm/model_executor/layers/utils.py
common|utils.py|/workspace/vllm/vllm/model_executor/layers/utils.py
common|rocm.py|/opt/venv/lib/python3.12/site-packages/vllm/platforms/rocm.py
common|rocm.py|/opt/src/vllm/vllm/platforms/rocm.py
common|rocm.py|/workspace/vllm/vllm/platforms/rocm.py
common|triton_attn.py|/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/triton_attn.py
common|triton_attn.py|/opt/src/vllm/vllm/v1/attention/backends/triton_attn.py
common|triton_attn.py|/workspace/vllm/vllm/v1/attention/backends/triton_attn.py
common|triton_unified_attention.py|/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py
common|triton_unified_attention.py|/opt/src/vllm/vllm/v1/attention/ops/triton_unified_attention.py
common|triton_unified_attention.py|/workspace/vllm/vllm/v1/attention/ops/triton_unified_attention.py
moe|qwen2_moe.py|/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen2_moe.py
moe|qwen2_moe.py|/opt/src/vllm/vllm/model_executor/models/qwen2_moe.py
moe|qwen2_moe.py|/workspace/vllm/vllm/model_executor/models/qwen2_moe.py
moe|qwen3_moe.py|/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_moe.py
moe|qwen3_moe.py|/opt/src/vllm/vllm/model_executor/models/qwen3_moe.py
moe|qwen3_moe.py|/workspace/vllm/vllm/model_executor/models/qwen3_moe.py
common|qwen3_5.py|/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5.py
common|qwen3_5.py|/opt/src/vllm/vllm/model_executor/models/qwen3_5.py
common|qwen3_5.py|/workspace/vllm/vllm/model_executor/models/qwen3_5.py
dense|parallel_state.py|/opt/venv/lib/python3.12/site-packages/vllm/distributed/parallel_state.py
dense|parallel_state.py|/opt/src/vllm/vllm/distributed/parallel_state.py
dense|parallel_state.py|/workspace/vllm/vllm/distributed/parallel_state.py
dense|communication_op.py|/opt/venv/lib/python3.12/site-packages/vllm/distributed/communication_op.py
dense|communication_op.py|/opt/src/vllm/vllm/distributed/communication_op.py
dense|communication_op.py|/workspace/vllm/vllm/distributed/communication_op.py
dense|linear.py|/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/linear.py
dense|linear.py|/opt/src/vllm/vllm/model_executor/layers/linear.py
dense|linear.py|/workspace/vllm/vllm/model_executor/layers/linear.py
dense|gfx906_persistent_ar.py|/opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/gfx906_persistent_ar.py
dense|gfx906_persistent_ar.py|/opt/src/vllm/vllm/distributed/device_communicators/gfx906_persistent_ar.py
dense|gfx906_persistent_ar.py|/workspace/vllm/vllm/distributed/device_communicators/gfx906_persistent_ar.py
dense|qwen2_moe_interleaved_swiglu_20260608.py|/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen2_moe.py
dense|qwen2_moe_interleaved_swiglu_20260608.py|/opt/src/vllm/vllm/model_executor/models/qwen2_moe.py
dense|qwen2_moe_interleaved_swiglu_20260608.py|/workspace/vllm/vllm/model_executor/models/qwen2_moe.py
moe_tp8|qwen3_next.py|/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py
moe_tp8|qwen3_next.py|/opt/src/vllm/vllm/model_executor/models/qwen3_next.py
moe_tp8|qwen3_next.py|/workspace/vllm/vllm/model_executor/models/qwen3_next.py
gguf|vllm/model_executor/model_loader/gguf_loader.py|/opt/venv/lib/python3.12/site-packages/vllm/model_executor/model_loader/gguf_loader.py
gguf|vllm/model_executor/model_loader/gguf_loader.py|/opt/src/vllm/vllm/model_executor/model_loader/gguf_loader.py
gguf|vllm/model_executor/model_loader/gguf_loader.py|/workspace/vllm/vllm/model_executor/model_loader/gguf_loader.py
gguf|vllm/model_executor/models/registry.py|/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/registry.py
gguf|vllm/model_executor/models/registry.py|/opt/src/vllm/vllm/model_executor/models/registry.py
gguf|vllm/model_executor/models/registry.py|/workspace/vllm/vllm/model_executor/models/registry.py
gguf|vllm/model_executor/models/config.py|/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/config.py
gguf|vllm/model_executor/models/config.py|/opt/src/vllm/vllm/model_executor/models/config.py
gguf|vllm/model_executor/models/config.py|/workspace/vllm/vllm/model_executor/models/config.py
gguf|vllm/model_executor/layers/quantization/gguf.py|/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/gguf.py
gguf|vllm/model_executor/layers/quantization/gguf.py|/opt/src/vllm/vllm/model_executor/layers/quantization/gguf.py
gguf|vllm/model_executor/layers/quantization/gguf.py|/workspace/vllm/vllm/model_executor/layers/quantization/gguf.py
gguf|vllm/v1/worker/gpu_model_runner.py|/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
gguf|vllm/v1/worker/gpu_model_runner.py|/opt/src/vllm/vllm/v1/worker/gpu_model_runner.py
gguf|vllm/v1/worker/gpu_model_runner.py|/workspace/vllm/vllm/v1/worker/gpu_model_runner.py
gguf|vllm/compilation/cuda_graph.py|/opt/venv/lib/python3.12/site-packages/vllm/compilation/cuda_graph.py
gguf|vllm/compilation/cuda_graph.py|/opt/src/vllm/vllm/compilation/cuda_graph.py
gguf|vllm/compilation/cuda_graph.py|/workspace/vllm/vllm/compilation/cuda_graph.py
gguf|vllm/transformers_utils/config.py|/opt/venv/lib/python3.12/site-packages/vllm/transformers_utils/config.py
gguf|vllm/transformers_utils/config.py|/opt/src/vllm/vllm/transformers_utils/config.py
gguf|vllm/transformers_utils/config.py|/workspace/vllm/vllm/transformers_utils/config.py
EOF_TARGETS

fused_src="$bundle/fused_moe.py"
if [ ! -f "$fused_src" ] && [ -f "$bundle/fused_moe_pr39016.py" ]; then
    fused_src="$bundle/fused_moe_pr39016.py"
fi
if target_group_enabled moe && [ -f "$fused_src" ]; then
    copy_if_present "$fused_src" /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/fused_moe.py
    copy_if_present "$fused_src" /opt/src/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py
    copy_if_present "$fused_src" /workspace/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py
fi

if target_group_enabled gguf; then
    copy_tree_if_present "$bundle/qwen36-python" /opt/qwen36-python
    if [ -d /opt/qwen36-python ]; then
        export PYTHONPATH="/opt/qwen36-python${PYTHONPATH:+:$PYTHONPATH}"
    fi
    copy_if_present "$bundle/qwen36-python/transformers/modeling_gguf_pytorch_utils.py" /opt/venv/lib/python3.12/site-packages/transformers/modeling_gguf_pytorch_utils.py
    copy_if_present "$bundle/qwen36-python/transformers/modeling_gguf_pytorch_utils.py" /opt/src/transformers/src/transformers/modeling_gguf_pytorch_utils.py
fi

if target_group_enabled dense; then
    copy_if_present "$bundle/sitecustomize_postcapture_start.py" /opt/vllm_trace_site/sitecustomize.py
fi
/opt/venv/bin/python /runwork/deploy_compat_patches.py

export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-rocm}"
export VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-0.0.0+gfx906}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx906}"
export GPU_ARCHS="${GPU_ARCHS:-gfx906}"
export DO_NOT_TRACK="${DO_NOT_TRACK:-1}"
export TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

exec /runwork/vllm_command.sh
EOF_CONTAINER_ENTRYPOINT
escaped_patch_target_groups=$(printf '%s' "$PATCH_TARGET_GROUPS" | sed 's/[\\&|]/\\&/g')
tmp_entrypoint="${run_dir}/container_entrypoint.sh.tmp"
sed "s|__PATCH_TARGET_GROUPS__|${escaped_patch_target_groups}|g" \
    "$run_dir/container_entrypoint.sh" >"$tmp_entrypoint"
mv "$tmp_entrypoint" "$run_dir/container_entrypoint.sh"
chmod +x "$run_dir/container_entrypoint.sh"

cat >"$run_dir/docker_run.sh" <<'EOF_DOCKER_RUN'
#!/bin/sh
set -eu

run_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# shellcheck disable=SC1090
. "$run_dir/effective_config.env"

: "${HOST_MODEL_ROOT:?set HOST_MODEL_ROOT to the host directory mounted at /opt/local-models}"
: "${HOST_HF_CACHE:?set HOST_HF_CACHE to the host Hugging Face cache directory}"
: "${HOST_PATCH_BUNDLE_PATH:=$patch_bundle_path}"
: "${HOST_RUNTIME_ROOT:=$run_dir/runtime}"
: "${VNEXT_CONTAINER_NAME:=vnext_${profile_name}}"
: "${VNEXT_REPLACE_CONTAINER:=1}"

case "$tp_size" in
    4) default_devices=0,1,2,3 ;;
    8) default_devices=0,1,2,3,4,5,6,7 ;;
    *) default_devices=0 ;;
esac
: "${HIP_VISIBLE_DEVICES:=$default_devices}"

resolve_symlink_target() {
    path=$1
    target=$(readlink "$path") || return 1
    case "$target" in
        /*)
            printf '%s\n' "$target"
            ;;
        *)
            path_dir=$(CDPATH= cd -- "$(dirname -- "$path")" && pwd)
            printf '%s/%s\n' "$path_dir" "$target"
            ;;
    esac
}

has_safetensors_symlink() {
    model_dir=$1
    for shard in "$model_dir"/*.safetensors; do
        [ -L "$shard" ] && return 0
    done
    return 1
}

resolve_hf_snapshot_blobs() {
    model_dir=$1
    case "$model_dir" in
        */snapshots/*) ;;
        *) return 1 ;;
    esac
    blobs_dir=$model_dir/../../blobs
    [ -d "$blobs_dir" ] || return 1
    CDPATH= cd -- "$blobs_dir" && pwd -P
}

mkdir -p \
    "$HOST_HF_CACHE" \
    "$HOST_RUNTIME_ROOT/root/.cache/vllm" \
    "$HOST_RUNTIME_ROOT/root/.triton" \
    "$HOST_RUNTIME_ROOT/tmp/torchinductor_root"

model_root_mount=$HOST_MODEL_ROOT
model_is_symlink=0
model_suffix=
resolved_model_path=
resolved_blobs_path=

case "$model" in
    /opt/local-models/*)
        model_suffix=${model#/opt/local-models/}
        host_model_path=$HOST_MODEL_ROOT/$model_suffix
        if [ -L "$host_model_path" ]; then
            resolved_model_path=$(resolve_symlink_target "$host_model_path") || {
                printf '%s\n' "error: could not resolve symlinked model path: $host_model_path" >&2
                exit 2
            }
            [ -d "$resolved_model_path" ] || {
                printf '%s\n' "error: symlinked model path is not a directory: $host_model_path" >&2
                exit 2
            }

            model_is_symlink=1
            model_root_mount=$HOST_RUNTIME_ROOT/model-bind-root/$profile_name
            model_link=$model_root_mount/$model_suffix
            rm -rf "$model_root_mount"
            mkdir -p "$(dirname -- "$model_link")"
            ln -s "/opt/vnext-models/$model_suffix" "$model_link"

            if has_safetensors_symlink "$resolved_model_path"; then
                resolved_blobs_path=$(resolve_hf_snapshot_blobs "$resolved_model_path") || {
                    printf '%s\n' "error: symlinked HF snapshot shards require a readable blobs directory: $host_model_path" >&2
                    exit 2
                }
            fi
        fi
        ;;
esac

if [ "$VNEXT_REPLACE_CONTAINER" = "1" ]; then
    docker stop "$VNEXT_CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$VNEXT_CONTAINER_NAME" >/dev/null 2>&1 || true
fi

set -- docker run -d \
    --name "$VNEXT_CONTAINER_NAME" \
    --network host \
    --ipc host \
    --privileged \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --cap-add SYS_PTRACE \
    --cap-add IPC_LOCK \
    --security-opt seccomp=unconfined \
    --security-opt label=disable \
    --ulimit memlock=-1:-1 \
    -v "$HOST_HF_CACHE:/root/.cache/huggingface" \
    -v "$model_root_mount:/opt/local-models:ro" \
    -v "$run_dir:/runwork" \
    -v "$HOST_RUNTIME_ROOT/root/.cache/vllm:/root/.cache/vllm" \
    -v "$HOST_RUNTIME_ROOT/root/.triton:/root/.triton" \
    -v "$HOST_RUNTIME_ROOT/tmp/torchinductor_root:/tmp/torchinductor_root" \
    -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
    -e LD_LIBRARY_PATH=/opt/gfx906:/rccl-overlay/install/lib:/usr/local/lib:/opt/rocm/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    --entrypoint /runwork/container_entrypoint.sh

if [ "$model_is_symlink" = "1" ]; then
    set -- "$@" -v "$resolved_model_path:/opt/vnext-models/$model_suffix:ro"
    if [ -n "$resolved_blobs_path" ]; then
        set -- "$@" -v "$resolved_blobs_path:/opt/blobs:ro"
    fi
fi

if [ -n "$HOST_PATCH_BUNDLE_PATH" ]; then
    set -- "$@" \
        -v "$HOST_PATCH_BUNDLE_PATH:/opt/vllm_patch_bundle:ro" \
        -e VLLM_PATCH_BUNDLE=/opt/vllm_patch_bundle
fi

set -- "$@" "$runtime_image"

"$@" > "$run_dir/container_id.txt"

docker logs -f "$VNEXT_CONTAINER_NAME" > "$run_dir/container.log" 2>&1 &
echo "$!" > "$run_dir/log_tail_pid.txt"
echo "$VNEXT_CONTAINER_NAME" > "$run_dir/container_name.txt"
printf '%s\n' "$VNEXT_CONTAINER_NAME"
EOF_DOCKER_RUN
chmod +x "$run_dir/docker_run.sh"

{
    printf 'model_format=%s\n' "$detected_format"
    printf 'model_family=%s\n' "$MODEL_FAMILY"
    printf 'detected_model_family=%s\n' "$detected_family"
    printf 'profile=%s\n' "$PROFILE"
    printf 'selected_overlay=%s\n' "$OVERLAY"
    printf 'runtime_image=%s\n' "$RUNTIME_IMAGE"
    printf 'runtime_image_digest=%s\n' "$RUNTIME_IMAGE_DIGEST"
    printf 'tp_size=%s\n' "$TP_SIZE"
    printf 'max_model_len=%s\n' "$MAX_MODEL_LEN"
    printf 'dtype=%s\n' "$VLLM_DTYPE"
    printf 'p2p=%s\n' "$(if [ "${NCCL_P2P_DISABLE:-}" = "0" ]; then printf on; else printf off; fi)"
    printf 'load_format=%s\n' "$LOAD_FORMAT"
    printf 'run_dir=%s\n' "$run_dir"
} | tee "$run_dir/preflight.txt"

if [ "$run_now" = "1" ]; then
    for var in ${PROFILE_ENV_VARS:-}; do
        eval "export $var"
    done
    exec "$@"
fi
