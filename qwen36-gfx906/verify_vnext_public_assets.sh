#!/bin/sh
set -eu

usage() {
  cat >&2 <<'EOF'
Usage:
  verify_vnext_public_assets.sh <release-tag> <local-model-root>

Downloads/stages the vNext GGUF release assets through
stage_vnext_gguf_assets.sh, verifies their hashes, then checks that the staged
assets satisfy the GGUF Dense and MoE vNext profile contracts.

Environment:
  RELEASE_ASSET_BASE              Optional release asset base URL.
  ALLOW_LOCAL_ASSET_PREFLIGHT=1   Required when RELEASE_ASSET_BASE is file://.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

[ "$#" -eq 2 ] || {
  usage
  exit 2
}

release_tag=$1
local_model_root=$2

case "$release_tag" in
  ''|'<release-tag>'|*'<'*|*'>'*|*' '*)
    die "release tag must be the final published tag, not a placeholder"
    ;;
esac

case "${RELEASE_ASSET_BASE:-}" in
  file://*)
    [ "${ALLOW_LOCAL_ASSET_PREFLIGHT:-0}" = "1" ] ||
      die "file:// RELEASE_ASSET_BASE requires ALLOW_LOCAL_ASSET_PREFLIGHT=1"
    ;;
esac

slash=/
home_root=${slash}home
case "$local_model_root" in
  ''|/|/root|/root/|"${home_root}"|"${home_root}/")
    die "local model root is unsafe or too broad: $local_model_root"
    ;;
esac

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
launch_root="$local_model_root/.vnext-public-asset-gate"

mkdir -p "$local_model_root" "$launch_root"

"$script_dir/stage_vnext_gguf_assets.sh" "$release_tag" "$local_model_root"

test -f "$local_model_root/qwen36-27b-f16-gguf/Qwen3.6-27B-F16.gguf" ||
  die "missing staged Dense GGUF"
test -f "$local_model_root/qwen36-35b-a3b-f16-gguf/Qwen3.6-35B-A3B-F16.gguf" ||
  die "missing staged MoE GGUF"
test -f "$local_model_root/qwen36-27b-text-config-eosfix/config.json" ||
  die "missing staged Dense text config"

make -C "$repo_root/tools/model-format-probe" model-format-probe >/dev/null

for profile in gguf-dense27b-tp8 gguf-moe35b-tp4; do
  run_dir="$launch_root/$profile"
  rm -rf "$run_dir"
  LOCAL_MODEL_ROOT="$local_model_root" \
    "$script_dir/vnext_repro_launcher.sh" \
      --profile "$profile" \
      --out "$run_dir" >/dev/null
  "$script_dir/verify_vnext_launch_artifact.sh" "$run_dir" >/dev/null
  printf 'public-asset-profile-ok: %s\n' "$profile"
done

printf 'vNext public GGUF asset gate passed for tag: %s\n' "$release_tag"
