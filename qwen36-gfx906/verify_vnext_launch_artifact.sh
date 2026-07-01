#!/bin/sh
set -eu

usage() {
    printf '%s\n' "usage: verify_vnext_launch_artifact.sh <vnext-launch-run-dir>" >&2
}

die() {
    printf '%s\n' "error: $*" >&2
    exit 2
}

[ "$#" -eq 1 ] || {
    usage
    exit 2
}

run_dir=$1
config="${run_dir}/effective_config.env"
command_file="${run_dir}/vllm_command.sh"
argv_file="${run_dir}/vllm_argv.txt"
benchmark_env="${run_dir}/benchmark_env.sh"
preflight="${run_dir}/preflight.txt"
container_entrypoint="${run_dir}/container_entrypoint.sh"
docker_run="${run_dir}/docker_run.sh"
deploy_compat_patches="${run_dir}/deploy_compat_patches.py"

[ -f "$config" ] || die "missing effective_config.env"
[ -f "$command_file" ] || die "missing vllm_command.sh"
[ -f "$argv_file" ] || die "missing vllm_argv.txt"
[ -f "$benchmark_env" ] || die "missing benchmark_env.sh"
[ -f "$preflight" ] || die "missing preflight.txt"
[ -x "$container_entrypoint" ] || die "missing executable container_entrypoint.sh"
[ -x "$docker_run" ] || die "missing executable docker_run.sh"
[ -f "$deploy_compat_patches" ] || die "missing deploy_compat_patches.py"

# shellcheck disable=SC1090
. "$config"

[ -n "${detected_model_format:-}" ] || die "effective_config missing detected_model_format"
[ -n "${expected_model_format:-}" ] || die "effective_config missing expected_model_format"
[ -n "${model:-}" ] || die "effective_config missing model"
[ -n "${model_probe_path:-}" ] || die "effective_config missing model_probe_path"
[ -n "${profile:-}" ] || die "effective_config missing profile"
[ -n "${model_family:-}" ] || die "effective_config missing model_family"
[ -n "${selected_overlay:-}" ] || die "effective_config missing selected_overlay"
[ -n "${patch_target_groups:-}" ] || die "effective_config missing patch_target_groups"
[ -n "${runtime_image:-}" ] || die "effective_config missing runtime_image"
[ -n "${runtime_image_digest:-}" ] || die "effective_config missing runtime_image_digest"
[ -n "${tp_size:-}" ] || die "effective_config missing tp_size"
[ -n "${max_model_len:-}" ] || die "effective_config missing max_model_len"
[ -n "${dtype:-}" ] || die "effective_config missing dtype"
[ -n "${p2p:-}" ] || die "effective_config missing p2p"
[ "$max_model_len" = "131072" ] || die "max_model_len must be 131072"
[ "$dtype" = "half" ] || die "dtype must be half"
[ "$p2p" = "on" ] || die "p2p must be on"

if [ "$detected_model_format" != "$expected_model_format" ]; then
    die "detected/expected model format mismatch in generated artifact"
fi

case "$detected_model_format:$selected_overlay" in
    gguf:gguf_dense|gguf:gguf_moe) ;;
    hf_safetensors:hf_release|hf_torch_legacy:hf_release) ;;
    *) die "invalid format/overlay pair: $detected_model_format/$selected_overlay" ;;
esac

case " $patch_target_groups " in
    *" common "*) ;;
    *) die "patch_target_groups must include common" ;;
esac
case "$detected_model_format:$patch_target_groups" in
    gguf:*gguf*) ;;
    gguf:*) die "GGUF artifact missing gguf patch target group" ;;
    hf_safetensors:*gguf*|hf_torch_legacy:*gguf*)
        die "HF/non-GGUF artifact includes gguf patch target group"
        ;;
esac
case "$model_family:$patch_target_groups" in
    qwen36_dense:*moe*) die "dense artifact includes moe patch target group" ;;
    qwen36_moe:*dense*) die "MoE artifact includes dense patch target group" ;;
esac

grep -q "patch_target_groups='$patch_target_groups'" "$container_entrypoint" ||
    die "container_entrypoint does not pin generated patch target groups"
if grep -q '__PATCH_TARGET_GROUPS__' "$container_entrypoint"; then
    die "container_entrypoint contains unresolved patch target placeholder"
fi
grep -q 'target_group_enabled dense &&.*native/sidecar_ar' "$container_entrypoint" ||
    die "dense native sidecar copy is not group-gated"
grep -q 'target_group_enabled dense &&.*native/swiglu' "$container_entrypoint" ||
    die "dense native swiglu copy is not group-gated"
grep -q 'target_group_enabled moe' "$container_entrypoint" ||
    die "MoE tuned-config/fused copy paths are not group-gated"
grep -q '^dense|qwen2_moe_interleaved_swiglu_20260608.py|' "$container_entrypoint" ||
    die "dense-only qwen2_moe_interleaved copy target is not group-labeled"
grep -q '^gguf|vllm/model_executor/model_loader/gguf_loader.py|' "$container_entrypoint" ||
    die "GGUF loader copy target is not group-labeled"

if [ "$detected_model_format" = "gguf" ]; then
    grep -q -- '--load-format gguf' "$command_file" || die "GGUF command missing --load-format gguf"
    grep -qx -- '--load-format' "$argv_file" || die "GGUF argv missing --load-format"
    grep -qx -- 'gguf' "$argv_file" || die "GGUF argv missing load format value"
else
    if grep -q -- '--load-format gguf' "$command_file"; then
        die "HF/non-GGUF command contains --load-format gguf"
    fi
    if grep -qx -- '--load-format' "$argv_file"; then
        die "HF/non-GGUF argv contains --load-format"
    fi
    if grep -Eq 'VLLM_QWEN35_GGUF_|VLLM_GFX906_GGUF_|VLLM_GGUF_' "$command_file"; then
        die "HF/non-GGUF command exports GGUF-only environment"
    fi
fi

if grep -Eq 'EXTRA_VLLM_ARGS' "$command_file" "$container_entrypoint" "$docker_run"; then
    die "vNext launch artifact must not use EXTRA_VLLM_ARGS"
fi

grep -q 'VLLM_TARGET_DEVICE=.*rocm' "$container_entrypoint" ||
    die "container_entrypoint does not preserve ROCm target-device default"
grep -q 'PYTORCH_ROCM_ARCH=.*gfx906' "$container_entrypoint" ||
    die "container_entrypoint does not preserve gfx906 PyTorch ROCm arch default"
grep -q 'GPU_ARCHS=.*gfx906' "$container_entrypoint" ||
    die "container_entrypoint does not preserve gfx906 GPU_ARCHS default"
grep -q 'TORCH_BLAS_PREFER_HIPBLASLT=.*0' "$container_entrypoint" ||
    die "container_entrypoint does not preserve HIPBLASLT preference default"
grep -q -- '--privileged' "$docker_run" ||
    die "docker_run does not match the release privileged runtime shape"
grep -q 'resolve_symlink_target' "$docker_run" ||
    die "docker_run does not include symlinked model directory resolver"
grep -q 'symlinked model path is not a directory' "$docker_run" ||
    die "docker_run does not fail closed on broken symlinked model directories"
grep -q 'has_safetensors_symlink' "$docker_run" ||
    die "docker_run does not detect symlinked HF shard files"
grep -q 'resolve_hf_snapshot_blobs' "$docker_run" ||
    die "docker_run does not resolve HF snapshot blobs directories"
grep -q 'symlinked HF snapshot shards require a readable blobs directory' "$docker_run" ||
    die "docker_run does not fail closed on unresolved HF snapshot shard symlinks"
grep -q 'model-bind-root' "$docker_run" ||
    die "docker_run does not create a runtime model-root shim for symlinked model directories"
grep -q '/opt/vnext-models/' "$docker_run" ||
    die "docker_run does not mount resolved symlinked models outside /opt/local-models"
grep -q '/opt/blobs:ro' "$docker_run" ||
    die "docker_run does not mount HF snapshot blobs read-only"
grep -q '/runwork/deploy_compat_patches.py' "$container_entrypoint" ||
    die "container_entrypoint does not run deploy compatibility patches"
grep -q 'patch_matcher_utils_missing_c_ops' "$deploy_compat_patches" ||
    die "deploy compatibility patches do not include matcher_utils missing-op guard"
grep -q 'patch_rotary_custom_op_fallbacks' "$deploy_compat_patches" ||
    die "deploy compatibility patches do not include rotary fallback guard"

if grep -q 'VLLM_TUNED_CONFIG_FOLDER=.*opt/vllm_tuned_moe_configs' "$command_file"; then
    grep -q 'copy_tree_if_present "$bundle/vllm_tuned_moe_configs" /opt/vllm_tuned_moe_configs' "$container_entrypoint" ||
        die "container_entrypoint does not populate deploy-style tuned MoE config path"
fi

grep -q '^PRE_MEASURE_WARMUP_REQUESTS=8$' "$benchmark_env" ||
    die "benchmark_env does not request 8 warmups"
grep -q '^PRE_MEASURE_WARMUP_MAX_TOKENS=2000$' "$benchmark_env" ||
    die "benchmark_env does not request 2000-token warmups"
grep -q '^PROMPT_REPEAT=32$' "$benchmark_env" ||
    die "benchmark_env does not set prompt repeat 32"
grep -q '^USE_BEGIN_THINK_PROXY=1$' "$benchmark_env" ||
    die "benchmark_env does not enable begin-think proxy"

printf '%s\n' "vnext_launch_artifact=valid"
printf 'run_dir=%s\n' "$run_dir"
printf 'profile=%s\n' "$profile"
printf 'model_format=%s\n' "$detected_model_format"
printf 'overlay=%s\n' "$selected_overlay"
