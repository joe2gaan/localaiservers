#!/bin/sh
set -eu

usage() {
    cat >&2 <<'EOF'
usage: verify_vnext_deploy_profile_parity.sh <profile-name> <deploy-runtime-env> [vnext-launch-run-dir]

Compares a deploy.sh-produced .deploy.runtime.env against a vNext profile.
This is a read-only contract comparison; it does not run Docker or benchmarks.
EOF
}

die() {
    printf '%s\n' "error: $*" >&2
    exit 2
}

say() {
    printf '%s\n' "$*"
}

[ "$#" -ge 2 ] && [ "$#" -le 3 ] || {
    usage
    exit 2
}

profile_name=$1
deploy_env=$2
run_dir=${3:-}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
profile_file="${script_dir}/profiles/vnext/${profile_name}.env"

[ -f "$profile_file" ] || die "profile not found: $profile_file"
[ -f "$deploy_env" ] || die "deploy runtime env not found: $deploy_env"

# shellcheck disable=SC1090
. "$profile_file"

get_deploy_value() {
    key=$1
    sed -n "s/^${key}=//p" "$deploy_env" | sed -n '1p'
}

profile_value() {
    key=$1
    eval "printf '%s\n' \"\${$key-}\""
}

check_equal() {
    deploy_key=$1
    profile_key=$2
    label=$3

    deploy_value=$(get_deploy_value "$deploy_key")
    expected=$(profile_value "$profile_key")
    [ "$deploy_value" = "$expected" ] || {
        die "$label mismatch: deploy ${deploy_key}='$deploy_value' profile ${profile_key}='$expected'"
    }
    say "parity-ok: $label=$deploy_value"
}

check_empty_or_equal() {
    deploy_key=$1
    profile_key=$2
    label=$3

    deploy_value=$(get_deploy_value "$deploy_key")
    expected=$(profile_value "$profile_key")
    [ -z "$deploy_value" ] || [ "$deploy_value" = "$expected" ] || {
        die "$label mismatch: deploy ${deploy_key}='$deploy_value' profile ${profile_key}='$expected'"
    }
    say "parity-ok: $label=${deploy_value:-<empty>}"
}

check_profile_name() {
    deploy_profile=$(get_deploy_value QWEN36_PROFILE)
    [ "$deploy_profile" = "$PROFILE" ] || {
        die "profile mismatch: deploy QWEN36_PROFILE='$deploy_profile' profile PROFILE='$PROFILE'"
    }
    say "parity-ok: profile=$PROFILE"
}

check_language_model_only() {
    extra=$(get_deploy_value EXTRA_VLLM_ARGS)
    case " $extra " in
        *" --language-model-only "*)
            [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ] ||
                die "deploy uses --language-model-only but profile LANGUAGE_MODEL_ONLY is not 1"
            ;;
        *)
            [ "${LANGUAGE_MODEL_ONLY:-0}" != "1" ] ||
                die "profile uses LANGUAGE_MODEL_ONLY=1 but deploy EXTRA_VLLM_ARGS='$extra'"
            ;;
    esac
    say "parity-ok: language-model-only=${LANGUAGE_MODEL_ONLY:-0}"
}

check_generated_artifact() {
    [ -n "$run_dir" ] || return 0

    command_file="${run_dir}/vllm_command.sh"
    argv_file="${run_dir}/vllm_argv.txt"
    config_file="${run_dir}/effective_config.env"
    [ -f "$command_file" ] || die "missing generated command file: $command_file"
    [ -f "$argv_file" ] || die "missing generated argv file: $argv_file"
    [ -f "$config_file" ] || die "missing generated effective config: $config_file"

    if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ]; then
        grep -qx -- '--language-model-only' "$argv_file" ||
            die "generated argv missing --language-model-only"
    fi
    if [ "${LOAD_FORMAT:-}" = "gguf" ]; then
        grep -qx -- '--load-format' "$argv_file" ||
            die "generated argv missing --load-format for GGUF"
        grep -qx -- 'gguf' "$argv_file" ||
            die "generated argv missing gguf load-format value"
    elif grep -qx -- '--load-format' "$argv_file"; then
        die "generated HF/non-GGUF argv contains --load-format"
    fi
    if grep -Eq 'EXTRA_VLLM_ARGS' "$command_file"; then
        die "generated command uses EXTRA_VLLM_ARGS"
    fi
    if printf '%s\n' "${PROFILE_ENV_VARS:-}" | grep -Eq '(^| )VLLM_TUNED_CONFIG_FOLDER( |$)'; then
        grep -q 'VLLM_TUNED_CONFIG_FOLDER=.*/opt/vllm_tuned_moe_configs' "$command_file" ||
            die "generated command does not use deploy-style tuned config folder"
    fi
    say "parity-ok: generated-artifact=$run_dir"
}

check_profile_name
check_equal TP_SIZE TP_SIZE "tp-size"
check_equal MAX_MODEL_LEN MAX_MODEL_LEN "max-model-len"
check_equal VLLM_DTYPE VLLM_DTYPE "dtype"
check_equal DISABLE_ASYNC_SCHEDULING DISABLE_ASYNC_SCHEDULING "async-scheduling-toggle"
check_equal NCCL_ALGO NCCL_ALGO "nccl-algo"
check_equal NCCL_PROTO NCCL_PROTO "nccl-proto"
check_equal NCCL_P2P_DISABLE NCCL_P2P_DISABLE "p2p"
check_empty_or_equal NCCL_MIN_NCHANNELS NCCL_MIN_NCHANNELS "nccl-min-nchannels"
check_equal NCCL_MAX_NCHANNELS NCCL_MAX_NCHANNELS "nccl-max-nchannels"
check_empty_or_equal NCCL_NTHREADS NCCL_NTHREADS "nccl-nthreads"
check_empty_or_equal RCCL_TREES RCCL_TREES "rccl-trees"
check_empty_or_equal VLLM_NCCL_SO_PATH VLLM_NCCL_SO_PATH "vllm-nccl-so-path"
check_equal VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH "moe-fastpath"
if printf '%s\n' "${PROFILE_ENV_VARS:-}" | grep -Eq '(^| )VLLM_TUNED_CONFIG_FOLDER( |$)'; then
    check_equal VLLM_TUNED_CONFIG_FOLDER VLLM_TUNED_CONFIG_FOLDER "tuned-config-folder"
fi
check_language_model_only
check_generated_artifact

say "vNext deploy/profile parity passed"
