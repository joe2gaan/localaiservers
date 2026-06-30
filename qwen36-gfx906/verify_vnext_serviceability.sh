#!/bin/sh
set -eu

usage() {
    cat >&2 <<'EOF'
usage: verify_vnext_serviceability.sh

Read-only developer serviceability check for vNext reproduction profiles.
This does not start Docker, probe live GPUs, mutate model caches, or run
benchmarks.

Optional checks:
  CHECK_RUNTIME_IMAGE=1  verify each public Docker tag resolves to its pinned digest
  CHECK_RUNTIME_PATHS=1  also start a short /bin/sh container to verify baseline
                         native runtime paths exist in the public image
EOF
}

die() {
    printf '%s\n' "error: $*" >&2
    exit 2
}

say() {
    printf '%s\n' "$*"
}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "${script_dir}/.." && pwd)
profile_dir="${script_dir}/profiles/vnext"
runtime_ref_file="${TMPDIR:-/tmp}/vnext-serviceability-runtime-refs.$$"

[ "$#" -eq 0 ] || {
    usage
    exit 2
}

trap 'rm -f "$runtime_ref_file"' EXIT HUP INT TERM
: >"$runtime_ref_file"

CHECK_RUNTIME_IMAGE=${CHECK_RUNTIME_IMAGE:-0}
CHECK_RUNTIME_PATHS=${CHECK_RUNTIME_PATHS:-0}
case "$CHECK_RUNTIME_IMAGE" in
    0|1) ;;
    *) die "CHECK_RUNTIME_IMAGE must be 0 or 1" ;;
esac
case "$CHECK_RUNTIME_PATHS" in
    0|1) ;;
    *) die "CHECK_RUNTIME_PATHS must be 0 or 1" ;;
esac
if [ "$CHECK_RUNTIME_PATHS" = "1" ]; then
    CHECK_RUNTIME_IMAGE=1
fi

[ -d "$profile_dir" ] || die "profile directory missing: $profile_dir"
[ -x "${script_dir}/list_vnext_release_asset_uploads.sh" ] || die "list_vnext_release_asset_uploads.sh is not executable"
[ -x "${script_dir}/upload_vnext_release_assets.sh" ] || die "upload_vnext_release_assets.sh is not executable"
[ -x "${script_dir}/vnext_profile_inspect.sh" ] || die "vnext_profile_inspect.sh is not executable"
[ -x "${script_dir}/hash_vnext_patch_bundle.sh" ] || die "hash_vnext_patch_bundle.sh is not executable"
[ -x "${script_dir}/stage_vnext_gguf_assets.sh" ] || die "stage_vnext_gguf_assets.sh is not executable"
[ -x "${script_dir}/verify_vnext_release_asset_bundle.sh" ] || die "verify_vnext_release_asset_bundle.sh is not executable"
[ -x "${script_dir}/verify_vnext_release_asset_urls.sh" ] || die "verify_vnext_release_asset_urls.sh is not executable"
[ -x "${script_dir}/verify_vnext_public_assets.sh" ] || die "verify_vnext_public_assets.sh is not executable"
[ -x "${script_dir}/verify_vnext_public_hf_inputs.sh" ] || die "verify_vnext_public_hf_inputs.sh is not executable"
[ -x "${script_dir}/verify_vnext_release_readiness.sh" ] || die "verify_vnext_release_readiness.sh is not executable"
[ -x "${script_dir}/check_host_platform_prereqs.sh" ] || die "check_host_platform_prereqs.sh is not executable"
[ -f "${repo_root}/tools/model-format-probe/Makefile" ] || die "model-format-probe Makefile missing"

sh -n "${script_dir}/check_host_platform_prereqs.sh" || die "check_host_platform_prereqs.sh is not POSIX-shell syntax clean"

make -C "${repo_root}/tools/model-format-probe" model-format-probe >/dev/null
[ -x "${repo_root}/tools/model-format-probe/model-format-probe" ] || die "model-format-probe did not build"

scan_paths="${script_dir}/profiles/vnext ${script_dir}/overlays ${script_dir}/list_vnext_release_asset_uploads.sh ${script_dir}/upload_vnext_release_assets.sh ${script_dir}/vnext_profile_inspect.sh ${script_dir}/vnext_repro_launcher.sh ${script_dir}/run_vnext_profile_benchmark.sh ${script_dir}/stage_vnext_gguf_assets.sh ${script_dir}/verify_vnext_release_asset_bundle.sh ${script_dir}/verify_vnext_release_asset_urls.sh ${script_dir}/verify_vnext_public_assets.sh ${script_dir}/verify_vnext_public_hf_inputs.sh ${script_dir}/check_host_platform_prereqs.sh"
private_ref_re='/home/|ai@[A-Za-z0-9._-]+:|(^|[^0-9])(10[.][0-9]{1,3}[.][0-9]{1,3}[.][0-9]{1,3}|192[.]168[.][0-9]{1,3}[.][0-9]{1,3}|172[.]16[.][0-9]{1,3}[.][0-9]{1,3})([^0-9]|$)'
if grep -RInE "$private_ref_re" $scan_paths >/tmp/vnext-serviceability-private-scan.$$ 2>/dev/null; then
    cat /tmp/vnext-serviceability-private-scan.$$ >&2
    rm -f /tmp/vnext-serviceability-private-scan.$$
    die "developer-facing vNext files contain private host/path references"
fi
rm -f /tmp/vnext-serviceability-private-scan.$$

profiles=$("${script_dir}/vnext_profile_inspect.sh" list)
[ -n "$profiles" ] || die "no vNext profiles found"

for required in hf-dense27b-tp8 hf-moe35b-tp4 hf-moe35b-tp8 gguf-dense27b-tp8 gguf-moe35b-tp4; do
    printf '%s\n' "$profiles" | grep -Fx "$required" >/dev/null || die "required profile missing: $required"
done

require_lock_text() {
    file=$1
    lock=$2
    label=$3

    grep -F -- "$lock" "$file" >/dev/null || die "$label missing from $file: $lock"
}

verify_release_note_locks() {
    release_note="${repo_root}/releases/draft-vnext-gfx906-rocm72-gguf-hf-repro.md"
    stage_script="${script_dir}/stage_vnext_gguf_assets.sh"
    hf_script="${script_dir}/verify_vnext_public_hf_inputs.sh"

    [ -f "$release_note" ] || return 0

    require_lock_text "$release_note" "stand-alone vNext reproduction-package release" "stand-alone boundary"
    require_lock_text "$release_note" "not a patch, addendum, or silent replacement" "v0.2 boundary"
    require_lock_text "$release_note" "RELEASE_ASSET_BASE" "release-asset test-hook disclosure"
    require_lock_text "$release_note" "ALLOW_LOCAL_ASSET_PREFLIGHT=1" "local asset preflight guard disclosure"
    require_lock_text "$release_note" "verify_vnext_public_assets.sh" "public asset gate"
    require_lock_text "$release_note" "list_vnext_release_asset_uploads.sh" "release asset upload list helper"
    require_lock_text "$release_note" "upload_vnext_release_assets.sh" "release asset upload helper"
    require_lock_text "$release_note" "verify_vnext_release_asset_bundle.sh" "release asset bundle verifier"
    require_lock_text "$release_note" "verify_vnext_release_asset_urls.sh" "release asset URL verifier"
    require_lock_text "$release_note" "verify_vnext_public_hf_inputs.sh" "public HF input gate"
    require_lock_text "$release_note" "model.safetensors.index.json" "public HF shard-index requirement"
    require_lock_text "$release_note" "unresolved Git LFS pointer" "public HF LFS-pointer rejection"
    require_lock_text "$release_note" "unsafe HF cache roots" "public HF cache-root guard disclosure"
    require_lock_text "$release_note" "verify_vnext_release_readiness.sh" "release readiness verifier"
    require_lock_text "$release_note" "RUN_SERVING_BENCHMARKS=1" "full benchmark opt-in"
    require_lock_text "$release_note" "ALLOW_PRE_SERVING_ONLY=1" "pre-serving dry-run disclosure"
    require_lock_text "$release_note" "unique" "unique launch-artifact disclosure"
    require_lock_text "$release_note" "container-owned files from a previous attempt" "repeat-run cleanup disclosure"
    require_lock_text "$release_note" "check_host_platform_prereqs.sh" "host platform preflight"
    require_lock_text "$release_note" "Container-internal paths such as" "container-internal path boundary"
    require_lock_text "$release_note" "HOST_MODEL_ROOT" "generated docker wrapper host root"
    require_lock_text "$release_note" "Final publication still requires the public GGUF asset gate" "final-publication boundary"
    require_lock_text "$release_note" "final GitHub Release asset URLs before publication" "final-public-asset gate"
    require_lock_text "$release_note" "replace those links with the final vNext release tag" "final-link publication gate"
    require_lock_text "$release_note" "--no-same-owner" "text-config tar ownership portability disclosure"
    require_lock_text "$release_note" "regular GGUF file already exists" "final GGUF reuse disclosure"
    require_lock_text "$release_note" "split parts again" "final GGUF reuse disclosure"
    require_lock_text "$release_note" "Symlinked GGUF files are materialized" "final GGUF symlink materialization disclosure"
    require_lock_text "$release_note" "Symlinked HF model" "HF symlink-directory disclosure"
    require_lock_text "$release_note" "they resolve to real local directories" "HF symlink-directory disclosure"
    require_lock_text "$release_note" "symlinked safetensors shards" "HF snapshot shard disclosure"
    require_lock_text "$release_note" 'model-bind-root' "HF runtime model-root shim disclosure"
    require_lock_text "$release_note" '/opt/vnext-models/...' "HF resolved model mount disclosure"
    require_lock_text "$release_note" 'snapshot `blobs` directory read-only' "HF snapshot blobs mount disclosure"
    require_lock_text "$release_note" "unresolved snapshot shard links" "HF snapshot shard fail-closed disclosure"
    require_lock_text "$release_note" "Broken or" "HF symlink fail-closed disclosure"
    require_lock_text "$release_note" "non-directory model symlinks fail closed" "HF symlink fail-closed disclosure"

    for lock in \
        b2347376b9bb7d12cf5f1d31c53ac4c60dd3d4b95068a09351187b328b5e9d89 \
        1f2443bb0ff958943d091410c61120c181a0579b3bc85192029aa51d821d141c \
        9a85f0a18a012ed37f0eb4c42549569234af880c5f6e3ce908e0e45cca835719
    do
        require_lock_text "$release_note" "$lock" "GGUF/text-config artifact lock"
        require_lock_text "$stage_script" "$lock" "staging-script artifact lock"
    done

    require_lock_text "$stage_script" "ALLOW_LOCAL_ASSET_PREFLIGHT=1" "staging-script local asset guard"
    require_lock_text "$stage_script" "file:// RELEASE_ASSET_BASE requires ALLOW_LOCAL_ASSET_PREFLIGHT=1" "staging-script file asset rejection"
    require_lock_text "$stage_script" "release tag must be the final published tag, not a placeholder" "staging-script placeholder tag rejection"
    require_lock_text "$stage_script" "local model root is unsafe or too broad" "staging-script unsafe root rejection"
    require_lock_text "$stage_script" "--no-same-owner" "staging-script text-config tar ownership portability"
    require_lock_text "$stage_script" "using existing verified model" "staging-script final GGUF reuse"
    require_lock_text "$stage_script" "existing staged model is a symlink" "staging-script symlink materialization"
    require_lock_text "$hf_script" "HF cache root" "HF verifier cache-root guard"
    require_lock_text "$hf_script" "STAGE_HF_PUBLIC_INPUTS must be 0 or 1" "HF verifier stage-flag guard"
    require_lock_text "$hf_script" "HF_DENSE_REVISION does not match the pinned release value" "HF verifier dense revision guard"
    require_lock_text "$hf_script" "HF_MOE_REVISION does not match the pinned release value" "HF verifier MoE revision guard"
    require_lock_text "$hf_script" "model.safetensors.index.json" "HF verifier shard-index guard"
    require_lock_text "$hf_script" "unresolved Git LFS pointer" "HF verifier LFS-pointer guard"
    require_lock_text "$hf_script" "public-hf-symlink-dir-ok" "HF verifier symlink-directory guard"
    require_lock_text "${script_dir}/verify_vnext_release_readiness.sh" "launch_stamp=" "readiness unique launch root"
    require_lock_text "${script_dir}/verify_vnext_release_readiness.sh" 'vnext-launch-runs/${launch_stamp}-$$' "readiness unique launch root"

    for profile_file in "$profile_dir"/*.env; do
        [ -f "$profile_file" ] || continue
        manifest_lock=$(sed -n 's/^PATCH_BUNDLE_MANIFEST_SHA256=//p' "$profile_file" | sed -n '1p')
        tokenizer_revision=$(sed -n 's/^TOKENIZER_REVISION=//p' "$profile_file" | sed -n '1p')

        [ -z "$manifest_lock" ] ||
            require_lock_text "$release_note" "$manifest_lock" "profile patch-bundle manifest lock"
        [ -z "$tokenizer_revision" ] ||
            require_lock_text "$release_note" "$tokenizer_revision" "profile tokenizer revision lock"
    done

    say "release-note-locks-ok"
}

verify_profile_doc_locks() {
    profile_doc="${profile_dir}/README.md"

    [ -f "$profile_doc" ] || die "profile README missing: $profile_doc"

    for lock in \
        "future stand-alone vNext release package" \
        "not a patch, addendum, or silent replacement" \
        "reproducible from its own release tag" \
        "tables, and claim boundary" \
        "verify_vnext_public_assets.sh" \
        "verify_vnext_public_hf_inputs.sh" \
        "model.safetensors.index.json" \
        "unresolved Git LFS pointer" \
        "verify_vnext_release_readiness.sh" \
        "RUN_SERVING_BENCHMARKS=1" \
        "ALLOW_PRE_SERVING_ONLY=1" \
        "unique" \
        "container-owned files from a previous attempt" \
        "check_host_platform_prereqs.sh" \
        "Generated launch artifacts are local outputs" \
        "Container-internal paths such as" \
        "HOST_MODEL_ROOT" \
        "STAGE_HF_PUBLIC_INPUTS=1" \
        "unsafe HF cache roots" \
        "RELEASE_TAG=<release-tag>" \
        "ALLOW_LOCAL_ASSET_PREFLIGHT=1" \
        "--no-same-owner" \
        "regular GGUF file already exists" \
        "split parts again" \
        "materialized as regular files" \
        "symlinked model" \
        "directory below" \
        "Broken" \
        "non-directory model symlinks fail closed" \
        "Symlinked HF model" \
        "they resolve to real local directories" \
        "symlinked safetensors shards" \
        'model-bind-root' \
        '/opt/vnext-models/...' \
        'snapshot `blobs` directory read-only' \
        "unresolved snapshot shard links" \
        "rejects placeholder release tags" \
        "model roots such as" \
        "vnext_repro_current" \
        "vllm-serve-help.txt" \
        "run_vnext_profile_benchmark.sh" \
        "gguf-dense27b-tp8" \
        "gguf-moe35b-tp4" \
        "hf-dense27b-tp8" \
        "hf-moe35b-tp4" \
        "hf-moe35b-tp8"
    do
        require_lock_text "$profile_doc" "$lock" "profile README reproduction lock"
    done

    say "profile-doc-locks-ok"
}

verify_release_note_locks
verify_profile_doc_locks

verify_bundle() {
    overlay=$1
    expected=$2

    case "$overlay" in
        hf_release)
            bundle="${script_dir}/overlays/hf/minimal-bundle"
            ;;
        gguf_dense)
            bundle="${script_dir}/overlays/gguf-dense/minimal-bundle"
            ;;
        gguf_moe)
            bundle="${script_dir}/overlays/gguf-moe/minimal-bundle"
            ;;
        *)
            die "unknown overlay for bundle verification: $overlay"
            ;;
    esac

    "${script_dir}/hash_vnext_patch_bundle.sh" verify "$bundle" "$expected" >/dev/null
}

bundle_for_overlay() {
    overlay=$1

    case "$overlay" in
        hf_release)
            printf '%s\n' "${script_dir}/overlays/hf/minimal-bundle"
            ;;
        gguf_dense)
            printf '%s\n' "${script_dir}/overlays/gguf-dense/minimal-bundle"
            ;;
        gguf_moe)
            printf '%s\n' "${script_dir}/overlays/gguf-moe/minimal-bundle"
            ;;
        *)
            die "unknown overlay for bundle lookup: $overlay"
            ;;
    esac
}

verify_runtime_manifest() {
    image=$1
    expected_digest=$2

    command -v docker >/dev/null 2>&1 || die "docker is required for CHECK_RUNTIME_IMAGE=1"
    manifest=$(docker manifest inspect --verbose "$image" 2>&1) || {
        printf '%s\n' "$manifest" >&2
        die "failed to inspect public runtime image: $image"
    }
    printf '%s\n' "$manifest" | grep -F "\"digest\": \"$expected_digest\"" >/dev/null || {
        die "runtime image digest mismatch or missing: image=$image expected=$expected_digest"
    }
    say "runtime-image-digest-ok: $image@$expected_digest"
}

verify_runtime_paths() {
    image=$1

    command -v docker >/dev/null 2>&1 || die "docker is required for CHECK_RUNTIME_PATHS=1"
    docker run --rm --entrypoint /bin/sh "$image" -c '
        set -eu
        for path in \
            /opt/gfx906/libgfx906_persistent_tree_ll_ar_default_20260613.so \
            /usr/share/ollama/kernel_labs/gfx906_swiglu_gemv_ext_native_runtime_20260608/gfx906_swiglu_gemv_ext_20260607.so \
            /rccl-overlay/install/lib/librccl.so.1
        do
            [ -e "$path" ] || {
                printf "%s\n" "missing runtime path: $path" >&2
                exit 2
            }
        done
    ' >/dev/null
    say "runtime-native-paths-ok: $image"
}

printf '%s\n' "$profiles" |
while IFS= read -r profile_name; do
    [ -n "$profile_name" ] || continue
    profile_file="${profile_dir}/${profile_name}.env"
    [ -f "$profile_file" ] || die "profile file missing: $profile_file"

    # shellcheck disable=SC1090
    . "$profile_file"

    [ -n "${PROFILE:-}" ] || die "$profile_name missing PROFILE"
    [ -n "${MODEL_FORMAT:-}" ] || die "$profile_name missing MODEL_FORMAT"
    [ -n "${MODEL_FAMILY:-}" ] || die "$profile_name missing MODEL_FAMILY"
    [ -n "${OVERLAY:-}" ] || die "$profile_name missing OVERLAY"
    [ -n "${RUNTIME_IMAGE:-}" ] || die "$profile_name missing RUNTIME_IMAGE"
    [ -n "${RUNTIME_IMAGE_DIGEST:-}" ] || die "$profile_name missing RUNTIME_IMAGE_DIGEST"
    [ -n "${TP_SIZE:-}" ] || die "$profile_name missing TP_SIZE"
    [ -n "${MAX_MODEL_LEN:-}" ] || die "$profile_name missing MAX_MODEL_LEN"
    [ -n "${VLLM_DTYPE:-}" ] || die "$profile_name missing VLLM_DTYPE"
    [ -n "${LOAD_FORMAT:-}" ] || die "$profile_name missing LOAD_FORMAT"
    [ -n "${PATCH_BUNDLE_MANIFEST_SHA256:-}" ] || die "$profile_name missing PATCH_BUNDLE_MANIFEST_SHA256"

    case "$RUNTIME_IMAGE" in
        joe2gaan/localaiservers:*) ;;
        *) die "$profile_name runtime image is not the public LocalAIServers image: $RUNTIME_IMAGE" ;;
    esac
    case "$RUNTIME_IMAGE_DIGEST" in
        sha256:????????????????????????????????????????????????????????????????) ;;
        *) die "$profile_name runtime image digest is not pinned: $RUNTIME_IMAGE_DIGEST" ;;
    esac

    [ "$MAX_MODEL_LEN" = "131072" ] || die "$profile_name MAX_MODEL_LEN is not 131072"
    [ "$VLLM_DTYPE" = "half" ] || die "$profile_name VLLM_DTYPE is not half"
    [ "${P2P_REQUIRED:-0}" = "1" ] || die "$profile_name does not declare P2P_REQUIRED=1"
    [ "${NCCL_P2P_DISABLE:-unset}" = "0" ] || die "$profile_name does not keep P2P on"
    [ "${PATCH_BUNDLE_REQUIRED:-0}" = "1" ] || die "$profile_name does not require a patch bundle"

    case "$MODEL_FORMAT:$LOAD_FORMAT:$OVERLAY" in
        gguf:gguf:gguf_dense|gguf:gguf:gguf_moe) ;;
        hf_safetensors:auto:hf_release|hf_torch_legacy:auto:hf_release) ;;
        *) die "$profile_name has incompatible format/load/overlay: $MODEL_FORMAT/$LOAD_FORMAT/$OVERLAY" ;;
    esac

    case "$MODEL_FORMAT:${TOKENIZER:-}" in
        gguf:Qwen/*)
            case "${TOKENIZER_REVISION:-}" in
                ????????????????????????????????????????) ;;
                *) die "$profile_name GGUF tokenizer repo is not pinned by TOKENIZER_REVISION" ;;
            esac
            ;;
    esac

    if printf '%s\n' "${PROFILE_ENV_VARS:-}" | grep -Eq '(^| )VLLM_TUNED_CONFIG_FOLDER( |$)'; then
        [ "${VLLM_TUNED_CONFIG_FOLDER:-}" = "/opt/vllm_tuned_moe_configs" ] ||
            die "$profile_name does not use deploy-style VLLM_TUNED_CONFIG_FOLDER"
        bundle=$(bundle_for_overlay "$OVERLAY")
        [ -d "$bundle/vllm_tuned_moe_configs" ] ||
            die "$profile_name requires tuned MoE configs but bundle lacks vllm_tuned_moe_configs"
        find "$bundle/vllm_tuned_moe_configs" -type f -name '*.json' | grep . >/dev/null ||
            die "$profile_name tuned MoE config bundle contains no JSON config"
    fi

    case "$MODEL_FORMAT:${PROFILE_ENV_VARS:-}" in
        hf_*:*VLLM_QWEN35_GGUF_*|hf_*:*VLLM_GFX906_GGUF_*|hf_*:*VLLM_GGUF_*)
            die "$profile_name leaks GGUF-only env into an HF profile"
            ;;
    esac

    verify_bundle "$OVERLAY" "$PATCH_BUNDLE_MANIFEST_SHA256"
    printf '%s %s\n' "$RUNTIME_IMAGE" "$RUNTIME_IMAGE_DIGEST" >>"$runtime_ref_file"
    "${script_dir}/vnext_profile_inspect.sh" show "$profile_name" >/dev/null
    say "serviceable: $profile_name"
done

if [ "$CHECK_RUNTIME_IMAGE" = "1" ]; then
    sort -u "$runtime_ref_file" |
    while read -r image expected_digest; do
        [ -n "$image" ] || continue
        verify_runtime_manifest "$image" "$expected_digest"
        if [ "$CHECK_RUNTIME_PATHS" = "1" ]; then
            verify_runtime_paths "$image"
        fi
    done
fi

say "vNext developer serviceability check passed"
