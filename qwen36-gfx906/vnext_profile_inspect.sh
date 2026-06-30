#!/bin/sh
set -eu

usage() {
    cat >&2 <<'EOF'
usage:
  vnext_profile_inspect.sh list
  vnext_profile_inspect.sh show <profile-name>

Read-only helper for inspecting vNext reproduction contracts. It does not probe
models, apply overlays, start Docker, or run benchmarks.
EOF
}

die() {
    printf '%s\n' "error: $*" >&2
    exit 2
}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
profile_dir="${script_dir}/profiles/vnext"

[ "$#" -ge 1 ] || {
    usage
    exit 2
}

case "$1" in
    list)
        [ "$#" -eq 1 ] || die "list takes no extra arguments"
        find "$profile_dir" -type f -name '*.env' | LC_ALL=C sort |
        while IFS= read -r file; do
            name=${file##*/}
            name=${name%.env}
            printf '%s\n' "$name"
        done
        ;;
    show)
        [ "$#" -eq 2 ] || die "show requires exactly one profile name"
        profile_name=$2
        case "$profile_name" in
            ''|*/*|*..*|*[!A-Za-z0-9_-]*)
                die "invalid profile name: $profile_name"
                ;;
        esac
        profile_file="${profile_dir}/${profile_name}.env"
        [ -f "$profile_file" ] || die "profile not found: $profile_name"

        # shellcheck disable=SC1090
        . "$profile_file"

        printf 'profile_name=%s\n' "$profile_name"
        printf 'profile=%s\n' "${PROFILE:-}"
        printf 'model_format=%s\n' "${MODEL_FORMAT:-}"
        printf 'model_family=%s\n' "${MODEL_FAMILY:-}"
        printf 'overlay=%s\n' "${OVERLAY:-}"
        printf 'runtime_image=%s\n' "${RUNTIME_IMAGE:-}"
        printf 'runtime_image_digest=%s\n' "${RUNTIME_IMAGE_DIGEST:-}"
        printf 'model_default=%s\n' "${MODEL_DEFAULT:-}"
        printf 'served_model_name_default=%s\n' "${SERVED_MODEL_NAME_DEFAULT:-}"
        printf 'tokenizer=%s\n' "${TOKENIZER:-}"
        printf 'tokenizer_revision=%s\n' "${TOKENIZER_REVISION:-}"
        printf 'tp_size=%s\n' "${TP_SIZE:-}"
        printf 'max_model_len=%s\n' "${MAX_MODEL_LEN:-}"
        printf 'vllm_dtype=%s\n' "${VLLM_DTYPE:-}"
        printf 'load_format=%s\n' "${LOAD_FORMAT:-}"
        printf 'p2p_required=%s\n' "${P2P_REQUIRED:-}"
        printf 'patch_bundle_required=%s\n' "${PATCH_BUNDLE_REQUIRED:-0}"
        printf 'patch_bundle_path_default=%s\n' "${PATCH_BUNDLE_PATH_DEFAULT:-}"
        printf 'patch_bundle_manifest_sha256=%s\n' "${PATCH_BUNDLE_MANIFEST_SHA256:-}"
        printf 'profile_env_vars=%s\n' "${PROFILE_ENV_VARS:-}"
        ;;
    -h|--help)
        usage
        ;;
    *)
        usage
        exit 2
        ;;
esac
