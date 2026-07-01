#!/bin/sh
set -eu

usage() {
  cat >&2 <<'EOF'
Usage:
  verify_vnext_release_asset_urls.sh <release-tag>

Checks that every expected vNext release asset URL resolves for the selected
tag without downloading the full multi-GB GGUF parts. Use this after uploading
the 69 release assets and before running the full readiness replay.

Environment:
  RELEASE_ASSET_BASE              Override release asset base URL.
  ALLOW_LOCAL_ASSET_PREFLIGHT=1   Required when RELEASE_ASSET_BASE is file://.
  CURL                            curl-compatible URL checker. Default: curl.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

[ "$#" -eq 1 ] || {
  usage
  exit 2
}

release_tag=$1
case "$release_tag" in
  ''|'<release-tag>'|*'<'*|*'>'*|*' '*)
    die "release tag must be the final published tag, not a placeholder"
    ;;
esac

curl_bin=${CURL:-curl}
release_asset_base=${RELEASE_ASSET_BASE:-"https://github.com/joe2gaan/localaiservers/releases/download/$release_tag"}

local_asset_dir=
case "$release_asset_base" in
  file://*)
    [ "${ALLOW_LOCAL_ASSET_PREFLIGHT:-0}" = "1" ] ||
      die "file:// RELEASE_ASSET_BASE requires ALLOW_LOCAL_ASSET_PREFLIGHT=1"
    local_asset_dir=${release_asset_base#file://}
    test -d "$local_asset_dir" || die "local release asset base does not exist: $local_asset_dir"
    ;;
  *)
    command -v "$curl_bin" >/dev/null 2>&1 || die "required command not found: $curl_bin"
    ;;
esac

dense_model=Qwen3.6-27B-F16.gguf
moe_model=Qwen3.6-35B-A3B-F16.gguf
text_archive=Qwen3.6-27B-text-config-eosfix.tar.gz

check_name() {
  asset_name=$1

  case "$asset_name" in
    ''|*/*|*..*|*' '*)
      die "unsafe release asset name: $asset_name"
      ;;
  esac

  if [ -n "$local_asset_dir" ]; then
    test -f "$local_asset_dir/$asset_name" ||
      die "missing local release asset: $asset_name"
  else
    "$curl_bin" -fsIL -o /dev/null "$release_asset_base/$asset_name" ||
      die "release asset URL did not resolve: $asset_name"
  fi

  printf 'release-asset-url-ok: %s\n' "$asset_name"
}

check_parts() {
  model_file=$1
  last_index=$2

  i=0
  while [ "$i" -le "$last_index" ]; do
    suffix=$(printf '%04d' "$i")
    check_name "$model_file.part-$suffix"
    i=$((i + 1))
  done
}

check_name "$text_archive"

check_name "$dense_model.parts.sha256"
check_name "$dense_model.sha256"
check_parts "$dense_model" 27

check_name "$moe_model.parts.sha256"
check_name "$moe_model.sha256"
check_parts "$moe_model" 35

printf 'vNext release asset URLs verified for tag: %s\n' "$release_tag"
