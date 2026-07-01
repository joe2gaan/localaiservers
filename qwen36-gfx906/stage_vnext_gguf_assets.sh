#!/bin/sh
set -eu

usage() {
  cat >&2 <<'EOF'
Usage:
  stage_vnext_gguf_assets.sh <release-tag> <local-model-root>

Downloads vNext GGUF release assets, verifies every part by SHA256, assembles
the final F16 GGUF files, verifies the final file hashes, and stages the dense
text-config/tokenizer asset needed by the Dense GGUF profile.

Environment:
  RELEASE_ASSET_BASE              Override release asset base URL.
  ALLOW_LOCAL_ASSET_PREFLIGHT=1   Required when RELEASE_ASSET_BASE is file://.
  CURL                            curl-compatible downloader. Default: curl.
  SHA256SUM                       sha256sum-compatible verifier. Default: sha256sum.
EOF
}

if [ "$#" -ne 2 ]; then
  usage
  exit 2
fi

release_tag=$1
local_model_root=$2

case "$release_tag" in
  ''|'<release-tag>'|*'<'*|*'>'*|*' '*)
    printf '%s\n' "error: release tag must be the final published tag, not a placeholder" >&2
    exit 2
    ;;
esac

slash=/
home_root=${slash}home
case "$local_model_root" in
  ''|/|/root|/root/|"${home_root}"|"${home_root}/")
    printf '%s\n' "error: local model root is unsafe or too broad: $local_model_root" >&2
    exit 2
    ;;
esac

curl_bin=${CURL:-curl}
sha256sum_bin=${SHA256SUM:-sha256sum}

release_asset_base=${RELEASE_ASSET_BASE:-"https://github.com/joe2gaan/localaiservers/releases/download/$release_tag"}

case "$release_asset_base" in
  file://*)
    [ "${ALLOW_LOCAL_ASSET_PREFLIGHT:-0}" = "1" ] || {
      printf '%s\n' "error: file:// RELEASE_ASSET_BASE requires ALLOW_LOCAL_ASSET_PREFLIGHT=1" >&2
      exit 2
    }
    ;;
esac

require_cmd() {
  cmd=$1

  command -v "$cmd" >/dev/null 2>&1 || {
    printf 'error: required command not found: %s\n' "$cmd" >&2
    exit 2
  }
}

require_cmd "$curl_bin"
require_cmd "$sha256sum_bin"
require_cmd tar
require_cmd cat
require_cmd grep
require_cmd rm
require_cmd mv
require_cmd mkdir

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

verify_one() {
  expected_sha256=$1
  file_path=$2

  printf '%s  %s\n' "$expected_sha256" "$file_path" | "$sha256sum_bin" -c -
}

download_one_part() {
  part_sha256=$1
  part_file=$2

  if [ -f "$part_file" ] &&
    verify_one "$part_sha256" "$part_file" >/dev/null 2>&1; then
    printf 'using existing verified part: %s\n' "$part_file"
    return 0
  fi

  "$curl_bin" -fL -o "$part_file" "$release_asset_base/$part_file"
  verify_one "$part_sha256" "$part_file"
}

download_split_gguf() {
  model_dir=$1
  model_file=$2
  final_sha256=$3
  parts_list="$model_file.parts.list"

  mkdir -p "$model_dir"
  (
    cd "$model_dir" || exit 1

    if [ -L "$model_file" ]; then
      if verify_one "$final_sha256" "$model_file" >/dev/null 2>&1; then
        printf 'existing staged model is a symlink; materializing regular file: %s\n' "$model_file"
        rm -f "$model_file.tmp"
        cat "$model_file" > "$model_file.tmp"
        verify_one "$final_sha256" "$model_file.tmp"
        mv -f "$model_file.tmp" "$model_file"
        verify_one "$final_sha256" "$model_file"
        return 0
      fi
      printf 'existing staged model symlink is not a verified model; downloading parts: %s\n' "$model_file"
    elif [ -f "$model_file" ] &&
      verify_one "$final_sha256" "$model_file" >/dev/null 2>&1; then
      printf 'using existing verified model: %s\n' "$model_file"
      return 0
    fi

    "$curl_bin" -fL -o "$model_file.parts.sha256" \
      "$release_asset_base/$model_file.parts.sha256"

    rm -f "$parts_list"
    while read -r part_sha256 part_file extra; do
      test -n "$part_sha256" || continue
      test -n "$part_file" || die "malformed part manifest row for $model_file"
      test -z "${extra:-}" || die "part filenames must not contain whitespace: $part_file"
      printf '%s\n' "$part_sha256" | grep -Eq '^[0-9A-Fa-f]{64}$' ||
        die "invalid part hash for $model_file: $part_sha256"
      case "$part_file" in
        "$model_file".part-*) ;;
        *) die "unexpected part name for $model_file: $part_file" ;;
      esac
      case "$part_file" in
        */*|*..*|'') die "unsafe part name for $model_file: $part_file" ;;
      esac
      download_one_part "$part_sha256" "$part_file"
      printf '%s\n' "$part_file" >> "$parts_list"
    done < "$model_file.parts.sha256"
    test -s "$parts_list" || die "part manifest contains no parts: $model_file.parts.sha256"

    "$sha256sum_bin" -c "$model_file.parts.sha256"

    rm -f "$model_file.tmp"
    while IFS= read -r part_file; do
      cat "$part_file" >> "$model_file.tmp"
    done < "$parts_list"
    verify_one "$final_sha256" "$model_file.tmp"
    mv -f "$model_file.tmp" "$model_file"
    verify_one "$final_sha256" "$model_file"
    rm -f "$parts_list"
  )
}

download_dense_text_config() {
  archive=Qwen3.6-27B-text-config-eosfix.tar.gz
  expected_sha256=9a85f0a18a012ed37f0eb4c42549569234af880c5f6e3ce908e0e45cca835719
  target_dir=$local_model_root/qwen36-27b-text-config-eosfix

  mkdir -p "$local_model_root"
  (
    cd "$local_model_root" || exit 1

    if [ -f "$archive" ] &&
      verify_one "$expected_sha256" "$archive" >/dev/null 2>&1; then
      printf 'using existing verified text-config archive: %s\n' "$archive"
    else
      "$curl_bin" -fL -o "$archive" "$release_asset_base/$archive"
      verify_one "$expected_sha256" "$archive"
    fi

    tar -tzf "$archive" | grep -Eq '(^/|(^|/)[.][.](/|$))' &&
      die "unsafe path in text-config archive: $archive"

    rm -rf "$target_dir"
    if tar --help 2>/dev/null | grep -q -- '--no-same-owner'; then
      tar --no-same-owner -xzf "$archive"
    else
      tar -xzf "$archive"
    fi
    test -f "$target_dir/config.json"
    test -f "$target_dir/tokenizer.json"
    verify_one 370a0c6b21a09e288bd3301581e126956d1b2dd410e791920884c3130bfe6e0a "$target_dir/config.json"
  )
}

download_dense_text_config

download_split_gguf \
  "$local_model_root/qwen36-27b-f16-gguf" \
  "Qwen3.6-27B-F16.gguf" \
  "b2347376b9bb7d12cf5f1d31c53ac4c60dd3d4b95068a09351187b328b5e9d89"

download_split_gguf \
  "$local_model_root/qwen36-35b-a3b-f16-gguf" \
  "Qwen3.6-35B-A3B-F16.gguf" \
  "1f2443bb0ff958943d091410c61120c181a0579b3bc85192029aa51d821d141c"

printf 'vNext GGUF assets staged under: %s\n' "$local_model_root"
