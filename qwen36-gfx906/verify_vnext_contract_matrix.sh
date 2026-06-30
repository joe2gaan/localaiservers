#!/bin/sh
set -eu

usage() {
    cat >&2 <<'EOF'
usage: verify_vnext_contract_matrix.sh

CPU-only vNext launcher contract matrix. This creates tiny synthetic model
signatures, generates vNext launch artifacts, verifies them, and confirms
expected format/family/env/hash mismatches fail closed.

It does not start Docker, touch GPUs, mutate model caches, or run benchmarks.
EOF
}

die() {
    printf '%s\n' "error: $*" >&2
    exit 2
}

say() {
    printf '%s\n' "$*"
}

[ "$#" -eq 0 ] || {
    usage
    exit 2
}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "${script_dir}/.." && pwd)

tmp_root=$(mktemp -d "${TMPDIR:-/tmp}/vnext-contract-matrix.XXXXXX") ||
    die "failed to create temporary directory"
trap 'rm -rf "$tmp_root"' EXIT HUP INT TERM

probe="${repo_root}/tools/model-format-probe/model-format-probe"
make -C "${repo_root}/tools/model-format-probe" model-format-probe >/dev/null
[ -x "$probe" ] || die "model-format-probe did not build"

write_gguf() {
    path=$1
    marker=$2
    mkdir -p "$(dirname -- "$path")"
    {
        printf 'GGUF'
        printf '%s\n' "$marker"
    } >"$path"
}

write_safetensors() {
    path=$1
    mkdir -p "$(dirname -- "$path")"
    header='{"w":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}'
    {
        # 64-byte little-endian safetensors header length.
        printf '\100\000\000\000\000\000\000\000'
        printf '%-64s' "$header"
        printf '\000\000'
    } >"$path"
}

write_hf_model_dir() {
    dir=$1
    family=$2
    mkdir -p "$dir"
    case "$family" in
        dense)
            cat >"$dir/config.json" <<'EOF_DENSE_CONFIG'
{"architectures":["Qwen3_5ForCausalLM"],"model_type":"qwen3","hidden_size":5120}
EOF_DENSE_CONFIG
            ;;
        moe)
            cat >"$dir/config.json" <<'EOF_MOE_CONFIG'
{"architectures":["Qwen3MoeForCausalLM"],"model_type":"qwen3_moe","num_experts":256,"n_routed_experts":256}
EOF_MOE_CONFIG
            ;;
        *)
            die "unknown synthetic HF family: $family"
            ;;
    esac
    write_safetensors "$dir/model-00001-of-00001.safetensors"
}

profile_bundle_path() {
    profile=$1
    case "$profile" in
        gguf-dense27b-tp8)
            printf '%s\n' "${script_dir}/overlays/gguf-dense/minimal-bundle"
            ;;
        gguf-moe35b-tp4)
            printf '%s\n' "${script_dir}/overlays/gguf-moe/minimal-bundle"
            ;;
        hf-dense27b-tp8|hf-moe35b-tp4|hf-moe35b-tp8)
            printf '%s\n' "${script_dir}/overlays/hf/minimal-bundle"
            ;;
        *)
            die "unknown profile for patch bundle mapping: $profile"
            ;;
    esac
}

run_launcher_ok() {
    profile=$1
    model_probe_path=$2
    out_dir="${tmp_root}/out/${profile}"
    patch_bundle_path=$(profile_bundle_path "$profile")

    rm -rf "$out_dir"
    MODEL="/opt/local-models/synthetic/${profile}" \
    MODEL_PROBE_PATH="$model_probe_path" \
    PATCH_BUNDLE_PATH="$patch_bundle_path" \
    CHECK_REQUIRED_PATHS=0 \
    CHECK_PATCH_BUNDLE_PATHS=1 \
    MODEL_FORMAT_PROBE="$probe" \
    "${script_dir}/vnext_repro_launcher.sh" \
        --profile "$profile" \
        --out "$out_dir" >/dev/null
    "${script_dir}/verify_vnext_launch_artifact.sh" "$out_dir" >/dev/null
    say "contract-ok: $profile"
}

run_launcher_inferred_ok() {
    profile=$1
    out_dir="${tmp_root}/portable-out/${profile}"

    rm -rf "$out_dir"
    LOCAL_MODEL_ROOT="${tmp_root}/portable-local-models" \
    CHECK_REQUIRED_PATHS=1 \
    CHECK_PATCH_BUNDLE_PATHS=1 \
    MODEL_FORMAT_PROBE="$probe" \
    "${script_dir}/vnext_repro_launcher.sh" \
        --profile "$profile" \
        --out "$out_dir" >/dev/null
    "${script_dir}/verify_vnext_launch_artifact.sh" "$out_dir" >/dev/null
    say "contract-inferred-ok: $profile"
}

run_launcher_fail() {
    name=$1
    profile=$2
    model_probe_path=$3
    out_dir="${tmp_root}/fail/${name}"
    patch_bundle_path=$(profile_bundle_path "$profile")

    rm -rf "$out_dir"
    set +e
    MODEL="/opt/local-models/synthetic/${name}" \
    MODEL_PROBE_PATH="$model_probe_path" \
    PATCH_BUNDLE_PATH="$patch_bundle_path" \
    CHECK_REQUIRED_PATHS=0 \
    CHECK_PATCH_BUNDLE_PATHS=1 \
    MODEL_FORMAT_PROBE="$probe" \
    "${script_dir}/vnext_repro_launcher.sh" \
        --profile "$profile" \
        --out "$out_dir" >"${tmp_root}/${name}.out" 2>"${tmp_root}/${name}.err"
    status=$?
    set -e

    if [ "$status" -eq 0 ]; then
        cat "${tmp_root}/${name}.out" >&2
        cat "${tmp_root}/${name}.err" >&2
        die "expected contract failure passed unexpectedly: $name"
    fi
    say "contract-fail-ok: $name"
}

run_env_leak_fail() {
    name=hf-gguf-env-leak
    profile=hf-dense27b-tp8
    model_probe_path=$1
    out_dir="${tmp_root}/fail/${name}"
    patch_bundle_path=$(profile_bundle_path "$profile")

    rm -rf "$out_dir"
    set +e
    VLLM_QWEN35_GGUF_NORM_MINUS_ONE=1 \
    MODEL="/opt/local-models/synthetic/${name}" \
    MODEL_PROBE_PATH="$model_probe_path" \
    PATCH_BUNDLE_PATH="$patch_bundle_path" \
    CHECK_REQUIRED_PATHS=0 \
    CHECK_PATCH_BUNDLE_PATHS=1 \
    MODEL_FORMAT_PROBE="$probe" \
    "${script_dir}/vnext_repro_launcher.sh" \
        --profile "$profile" \
        --out "$out_dir" >"${tmp_root}/${name}.out" 2>"${tmp_root}/${name}.err"
    status=$?
    set -e

    if [ "$status" -eq 0 ]; then
        cat "${tmp_root}/${name}.out" >&2
        cat "${tmp_root}/${name}.err" >&2
        die "expected HF GGUF-env leak failure passed unexpectedly"
    fi
    say "contract-fail-ok: $name"
}

run_bad_hash_fail() {
    name=patch-bundle-hash-mismatch
    profile=gguf-dense27b-tp8
    model_probe_path=$1
    out_dir="${tmp_root}/fail/${name}"
    patch_bundle_path=$(profile_bundle_path "$profile")

    rm -rf "$out_dir"
    set +e
    MODEL="/opt/local-models/synthetic/${name}" \
    MODEL_PROBE_PATH="$model_probe_path" \
    PATCH_BUNDLE_PATH="$patch_bundle_path" \
    CHECK_REQUIRED_PATHS=0 \
    CHECK_PATCH_BUNDLE_PATHS=1 \
    MODEL_FORMAT_PROBE="$probe" \
    VNEXT_PATCH_BUNDLE_MANIFEST_SHA256=0000000000000000000000000000000000000000000000000000000000000000 \
    "${script_dir}/vnext_repro_launcher.sh" \
        --profile "$profile" \
        --out "$out_dir" >"${tmp_root}/${name}.out" 2>"${tmp_root}/${name}.err"
    status=$?
    set -e

    if [ "$status" -eq 0 ]; then
        cat "${tmp_root}/${name}.out" >&2
        cat "${tmp_root}/${name}.err" >&2
        die "expected patch-bundle hash failure passed unexpectedly"
    fi
    say "contract-fail-ok: $name"
}

write_deploy_runtime_env() {
    path=$1
    profile=$2
    tp_size=$3
    disable_async=$4
    fastpath=$5

    cat >"$path" <<EOF_DEPLOY_ENV
QWEN36_PROFILE=$profile
TP_SIZE=$tp_size
MAX_MODEL_LEN=131072
VLLM_DTYPE=half
DISABLE_ASYNC_SCHEDULING=$disable_async
NCCL_ALGO=Tree
NCCL_PROTO=LL
NCCL_P2P_DISABLE=0
NCCL_MIN_NCHANNELS=
NCCL_MAX_NCHANNELS=1
NCCL_NTHREADS=
RCCL_TREES=
VLLM_NCCL_SO_PATH=
VLLM_ENABLE_C1_TOPK8_MOE_FASTPATH=$fastpath
VLLM_TUNED_CONFIG_FOLDER=/opt/vllm_tuned_moe_configs
EXTRA_VLLM_ARGS=--language-model-only
EOF_DEPLOY_ENV
}

run_deploy_parity_ok() {
    name=$1
    profile_name=$2
    deploy_profile=$3
    tp_size=$4
    disable_async=$5
    fastpath=$6
    deploy_env="${tmp_root}/${name}.deploy.runtime.env"
    run_dir="${tmp_root}/out/${profile_name}"

    write_deploy_runtime_env "$deploy_env" "$deploy_profile" "$tp_size" "$disable_async" "$fastpath"
    "${script_dir}/verify_vnext_deploy_profile_parity.sh" "$profile_name" "$deploy_env" "$run_dir" >/dev/null
    say "deploy-parity-ok: $profile_name"
}

models="${tmp_root}/models"
write_gguf "${models}/gguf-dense/Qwen3.6-27B-F16.gguf" "Qwen3.6-27B dense synthetic metadata"
write_gguf "${models}/gguf-moe/Qwen3.6-35B-A3B-F16.gguf" "Qwen3.6-35B-A3B qwen3 moe num_experts synthetic metadata"
write_hf_model_dir "${models}/hf-dense" dense
write_hf_model_dir "${models}/hf-moe" moe
portable_models="${tmp_root}/portable-local-models"
write_gguf "${portable_models}/qwen36-27b-f16-gguf/Qwen3.6-27B-F16.gguf" "Qwen3.6-27B dense synthetic metadata"
write_gguf "${portable_models}/qwen36-35b-a3b-f16-gguf/Qwen3.6-35B-A3B-F16.gguf" "Qwen3.6-35B-A3B qwen3 moe num_experts synthetic metadata"
write_hf_model_dir "${portable_models}/qwen36-27b-hf" dense
write_hf_model_dir "${portable_models}/qwen36-35b-a3b-hf" moe
mkdir -p "${portable_models}/qwen36-27b-text-config-eosfix"
printf '%s\n' '{"architectures":["Qwen3_5ForCausalLM"]}' \
    >"${portable_models}/qwen36-27b-text-config-eosfix/config.json"
printf '%s\n' '{}' >"${portable_models}/qwen36-27b-text-config-eosfix/tokenizer.json"
mkdir -p "${models}/conflict"
write_gguf "${models}/conflict/model.gguf" "Qwen3.6-27B"
write_safetensors "${models}/conflict/model.safetensors"

"$probe" "${models}/gguf-dense/Qwen3.6-27B-F16.gguf" | grep -q '^model_format=gguf$' ||
    die "synthetic GGUF probe failed"
"$probe" "${models}/hf-dense" | grep -q '^model_format=hf_safetensors$' ||
    die "synthetic safetensors probe failed"
if "$probe" "${models}/conflict" >/dev/null 2>&1; then
    die "conflicting synthetic model directory did not fail closed"
fi
say "binary-probe-ok"

run_launcher_ok gguf-dense27b-tp8 "${models}/gguf-dense/Qwen3.6-27B-F16.gguf"
run_launcher_ok gguf-moe35b-tp4 "${models}/gguf-moe/Qwen3.6-35B-A3B-F16.gguf"
run_launcher_ok hf-dense27b-tp8 "${models}/hf-dense"
run_launcher_ok hf-moe35b-tp4 "${models}/hf-moe"
run_launcher_ok hf-moe35b-tp8 "${models}/hf-moe"

run_launcher_inferred_ok gguf-dense27b-tp8
run_launcher_inferred_ok gguf-moe35b-tp4
run_launcher_inferred_ok hf-dense27b-tp8
run_launcher_inferred_ok hf-moe35b-tp4
run_launcher_inferred_ok hf-moe35b-tp8

run_deploy_parity_ok hf-moe35b-tp4 hf-moe35b-tp4 moe35b_tp4_fullbar_p2pon 4 0 1
run_deploy_parity_ok hf-moe35b-tp8 hf-moe35b-tp8 moe35b_tp8_fullbar_p2pon 8 1 force

run_launcher_fail gguf-model-under-hf-profile hf-dense27b-tp8 "${models}/gguf-dense/Qwen3.6-27B-F16.gguf"
run_launcher_fail hf-model-under-gguf-profile gguf-dense27b-tp8 "${models}/hf-dense"
run_launcher_fail dense-model-under-moe-profile gguf-moe35b-tp4 "${models}/gguf-dense/Qwen3.6-27B-F16.gguf"
run_launcher_fail moe-model-under-dense-profile gguf-dense27b-tp8 "${models}/gguf-moe/Qwen3.6-35B-A3B-F16.gguf"
run_env_leak_fail "${models}/hf-dense"
run_bad_hash_fail "${models}/gguf-dense/Qwen3.6-27B-F16.gguf"

say "vNext contract matrix passed"
