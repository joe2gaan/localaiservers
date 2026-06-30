#!/bin/sh
set -eu

usage() {
    cat >&2 <<'EOF'
usage:
  verify_vnext_release_readiness.sh \
    --release-tag TAG \
    --model-root DIR \
    --hf-cache DIR \
    --runtime-root DIR

Runs the vNext release-note reproduction gates in order:
  serviceability -> contract matrix -> public GGUF asset gate -> public HF gate
  -> launch artifact generation -> vLLM argv schema -> host preflight
  -> optional serving benchmark ladder

Environment:
  RUN_SERVING_BENCHMARKS=1    Required for full reproduction success.
  ALLOW_PRE_SERVING_ONLY=1    Allow success after non-serving gates/preflight.
  STAGE_HF_PUBLIC_INPUTS=1    Let the HF gate download pinned public inputs.
  RELEASE_ASSET_BASE=...      Optional maintainer test hook for asset staging.
  ALLOW_LOCAL_ASSET_PREFLIGHT=1 required for file:// RELEASE_ASSET_BASE.
  CHECK_RUNTIME_IMAGE=0       Skip Docker image digest check. Default: 1.
  CHECK_RUNTIME_PATHS=0       Skip Docker runtime path check. Default: 1.
  CAPTURE_VLLM_HELP=0         Skip runtime vLLM help/schema check. Default: 1.
EOF
}

die() {
    printf '%s\n' "error: $*" >&2
    exit 2
}

say() {
    printf '%s\n' "$*"
}

release_tag=
local_model_root=
local_hf_cache=
local_runtime_root=

while [ "$#" -gt 0 ]; do
    case "$1" in
        --release-tag)
            [ "$#" -ge 2 ] || die "--release-tag needs a value"
            release_tag=$2
            shift 2
            ;;
        --model-root)
            [ "$#" -ge 2 ] || die "--model-root needs a value"
            local_model_root=$2
            shift 2
            ;;
        --hf-cache)
            [ "$#" -ge 2 ] || die "--hf-cache needs a value"
            local_hf_cache=$2
            shift 2
            ;;
        --runtime-root)
            [ "$#" -ge 2 ] || die "--runtime-root needs a value"
            local_runtime_root=$2
            shift 2
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

[ -n "$release_tag" ] || die "--release-tag is required"
[ -n "$local_model_root" ] || die "--model-root is required"
[ -n "$local_hf_cache" ] || die "--hf-cache is required"
[ -n "$local_runtime_root" ] || die "--runtime-root is required"

case "$release_tag" in
    ''|'<release-tag>'|*'<'*|*'>'*|*' '*)
        die "release tag must be the final published tag, not a placeholder"
        ;;
esac

safe_root() {
    label=$1
    path=$2
    slash=/
    home_root=${slash}home

    case "$path" in
        ''|/|/root|/root/|"${home_root}"|"${home_root}/")
            die "$label is unsafe or too broad: $path"
            ;;
    esac
}

safe_root "model root" "$local_model_root"
safe_root "HF cache" "$local_hf_cache"
safe_root "runtime root" "$local_runtime_root"

RUN_SERVING_BENCHMARKS=${RUN_SERVING_BENCHMARKS:-0}
ALLOW_PRE_SERVING_ONLY=${ALLOW_PRE_SERVING_ONLY:-0}
CHECK_RUNTIME_IMAGE=${CHECK_RUNTIME_IMAGE:-1}
CHECK_RUNTIME_PATHS=${CHECK_RUNTIME_PATHS:-1}
CAPTURE_VLLM_HELP=${CAPTURE_VLLM_HELP:-1}

case "$RUN_SERVING_BENCHMARKS" in 0|1) ;; *) die "RUN_SERVING_BENCHMARKS must be 0 or 1" ;; esac
case "$ALLOW_PRE_SERVING_ONLY" in 0|1) ;; *) die "ALLOW_PRE_SERVING_ONLY must be 0 or 1" ;; esac
case "$CHECK_RUNTIME_IMAGE" in 0|1) ;; *) die "CHECK_RUNTIME_IMAGE must be 0 or 1" ;; esac
case "$CHECK_RUNTIME_PATHS" in 0|1) ;; *) die "CHECK_RUNTIME_PATHS must be 0 or 1" ;; esac
case "$CAPTURE_VLLM_HELP" in 0|1) ;; *) die "CAPTURE_VLLM_HELP must be 0 or 1" ;; esac

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
profile_dir="${script_dir}/profiles/vnext"
launch_stamp=$(date -u +%Y%m%dT%H%M%SZ)
launch_root="${local_runtime_root}/vnext-launch-runs/${launch_stamp}-$$"
help_file="${launch_root}/vllm-serve-help.txt"

mkdir -p "$local_model_root" "$local_hf_cache" "$local_runtime_root" "$launch_root"

say "vNext release readiness: serviceability"
CHECK_RUNTIME_IMAGE="$CHECK_RUNTIME_IMAGE" \
CHECK_RUNTIME_PATHS="$CHECK_RUNTIME_PATHS" \
    "${script_dir}/verify_vnext_serviceability.sh" ||
    die "serviceability gate failed"

say "vNext release readiness: contract matrix"
"${script_dir}/verify_vnext_contract_matrix.sh" ||
    die "contract matrix failed"

say "vNext release readiness: public GGUF assets"
"${script_dir}/verify_vnext_public_assets.sh" "$release_tag" "$local_model_root" ||
    die "public GGUF asset gate failed; check release tag, release assets, part manifests, and final SHA256 locks"

say "vNext release readiness: public HF inputs"
HF_HOME="$local_hf_cache" \
    "${script_dir}/verify_vnext_public_hf_inputs.sh" "$local_model_root" ||
    die "public HF input gate failed; install hf and use STAGE_HF_PUBLIC_INPUTS=1 with writable non-symlink staging directories, or pre-stage the pinned public HF repos and rerun with STAGE_HF_PUBLIC_INPUTS=0"

profiles="gguf-dense27b-tp8 gguf-moe35b-tp4 hf-dense27b-tp8 hf-moe35b-tp4 hf-moe35b-tp8"

say "vNext release readiness: launch artifacts"
for profile in $profiles; do
    run_dir="${launch_root}/${profile}"
    rm -rf "$run_dir"
    LOCAL_MODEL_ROOT="$local_model_root" \
    HOST_HF_CACHE="$local_hf_cache" \
        "${script_dir}/vnext_repro_launcher.sh" \
            --profile "$profile" \
            --out "$run_dir" >/dev/null ||
        die "launch artifact generation failed for $profile"
    "${script_dir}/verify_vnext_launch_artifact.sh" "$run_dir" >/dev/null ||
        die "launch artifact verification failed for $profile"
    printf 'launch-artifact-ok: %s\n' "$profile"
done

if [ "$CAPTURE_VLLM_HELP" = "1" ]; then
    runtime_image=$(sed -n 's/^RUNTIME_IMAGE=//p' "${profile_dir}/hf-dense27b-tp8.env" | sed -n '1p')
    [ -n "$runtime_image" ] || die "could not read RUNTIME_IMAGE from hf-dense27b-tp8 profile"

    say "vNext release readiness: runtime vLLM schema"
    command -v docker >/dev/null 2>&1 || die "docker is required for runtime vLLM schema capture"
    docker run --rm \
        --device /dev/kfd \
        --device /dev/dri \
        --group-add video \
        --security-opt seccomp=unconfined \
        --security-opt label=disable \
        --entrypoint vllm \
        "$runtime_image" \
        serve --help=all >"$help_file" ||
        die "runtime vLLM help capture failed; expose GPU devices or set CAPTURE_VLLM_HELP=0 for a non-serving dry run"

    for profile in $profiles; do
        "${script_dir}/verify_vllm_arg_schema.sh" \
            "${launch_root}/${profile}" \
            "$help_file" >/dev/null ||
            die "runtime vLLM argv schema check failed for $profile"
        printf 'vllm-schema-ok: %s\n' "$profile"
    done
else
    say "vNext release readiness: runtime vLLM schema skipped by CAPTURE_VLLM_HELP=0"
fi

say "vNext release readiness: host platform preflight"
for release_profile in \
    dense27b_tp8_fullbar_p2pon \
    moe35b_tp4_fullbar_p2pon \
    moe35b_tp8_fullbar_p2pon
do
    QWEN36_PROFILE="$release_profile" "${script_dir}/check_host_platform_prereqs.sh" ||
        die "host platform preflight failed for $release_profile; this host is not comparable with the full-BAR/P2P-on release lane"
done

if [ "$RUN_SERVING_BENCHMARKS" = "1" ]; then
    say "vNext release readiness: serving benchmark ladder"
    for profile in $profiles; do
        docker stop vnext_repro_current >/dev/null 2>&1 || true
        docker rm vnext_repro_current >/dev/null 2>&1 || true

        HOST_MODEL_ROOT="$local_model_root" \
        HOST_HF_CACHE="$local_hf_cache" \
        HOST_RUNTIME_ROOT="${local_runtime_root}/${profile}" \
        VNEXT_CONTAINER_NAME=vnext_repro_current \
            "${launch_root}/${profile}/docker_run.sh" ||
            die "container launch failed for $profile"

        "${script_dir}/run_vnext_profile_benchmark.sh" "${launch_root}/${profile}" ||
            die "serving benchmark ladder failed for $profile"

        docker stop vnext_repro_current >/dev/null 2>&1 || true
        docker rm vnext_repro_current >/dev/null 2>&1 || true
    done

    say "vNext full release reproduction path completed"
else
    say "pre-serving-vnext-release-gates-passed"
    say "full-serving-benchmark-not-run: set RUN_SERVING_BENCHMARKS=1 for release reproduction success"
    [ "$ALLOW_PRE_SERVING_ONLY" = "1" ] ||
        die "full serving benchmark ladder was not run"
fi
