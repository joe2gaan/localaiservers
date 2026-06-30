#!/bin/sh
set -eu

usage() {
  cat >&2 <<'EOF'
Usage:
  verify_vnext_release_asset_bundle.sh <release-asset-dir>

Verifies the complete vNext GGUF release upload bundle before publishing:
  - expected file inventory
  - Dense and MoE part manifests
  - every split-part SHA256
  - final assembled GGUF SHA256 by streaming parts in manifest order
  - Dense text-config archive SHA256

The script reads the split parts but does not assemble another full GGUF copy.
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

asset_dir=$1
slash=/
home_root=${slash}home

case "$asset_dir" in
  ''|/|/root|/root/|"${home_root}"|"${home_root}/")
    die "release asset dir is unsafe or too broad: $asset_dir"
    ;;
esac

test -d "$asset_dir" || die "release asset dir does not exist: $asset_dir"

command -v sha256sum >/dev/null 2>&1 || die "sha256sum not found"
command -v find >/dev/null 2>&1 || die "find not found"
command -v wc >/dev/null 2>&1 || die "wc not found"
command -v awk >/dev/null 2>&1 || die "awk not found"
command -v grep >/dev/null 2>&1 || die "grep not found"
command -v cat >/dev/null 2>&1 || die "cat not found"
command -v rm >/dev/null 2>&1 || die "rm not found"

dense_model=Qwen3.6-27B-F16.gguf
moe_model=Qwen3.6-35B-A3B-F16.gguf
text_archive=Qwen3.6-27B-text-config-eosfix.tar.gz

dense_final_sha=b2347376b9bb7d12cf5f1d31c53ac4c60dd3d4b95068a09351187b328b5e9d89
moe_final_sha=1f2443bb0ff958943d091410c61120c181a0579b3bc85192029aa51d821d141c
text_archive_sha=9a85f0a18a012ed37f0eb4c42549569234af880c5f6e3ce908e0e45cca835719

tmp_dense=${TMPDIR:-/tmp}/vnext-dense-parts.$$
tmp_moe=${TMPDIR:-/tmp}/vnext-moe-parts.$$
trap 'rm -f "$tmp_dense" "$tmp_moe"' EXIT HUP INT TERM

verify_one() {
  expected_sha=$1
  file_path=$2

  printf '%s  %s\n' "$expected_sha" "$file_path" | sha256sum -c -
}

count_files() {
  pattern=$1

  find . -maxdepth 1 -type f -name "$pattern" | wc -l | awk '{print $1}'
}

require_sidecar() {
  file_name=$1
  expected_sha=$2

  test -f "$file_name.sha256" || die "missing final hash sidecar: $file_name.sha256"

  set -- $(sed -n '1p' "$file_name.sha256")
  found_sha=${1:-}
  found_name=${2:-}
  extra=${3:-}

  test "$found_sha" = "$expected_sha" ||
    die "sidecar hash mismatch for $file_name: $found_sha"
  test "$found_name" = "$file_name" ||
    die "sidecar filename mismatch for $file_name: $found_name"
  test -z "$extra" ||
    die "sidecar has unexpected extra fields for $file_name"
}

verify_manifest() {
  model_file=$1
  expected_count=$2
  expected_final_sha=$3
  parts_file=$4

  manifest=$model_file.parts.sha256
  test -f "$manifest" || die "missing part manifest: $manifest"
  : >"$parts_file"

  count=0
  while read -r part_sha part_file extra; do
    test -n "$part_sha" || continue
    test -n "$part_file" || die "malformed manifest row in $manifest"
    test -z "${extra:-}" || die "part filenames must not contain whitespace: $part_file"
    printf '%s\n' "$part_sha" | grep -Eq '^[0-9A-Fa-f]{64}$' ||
      die "invalid part hash for $model_file: $part_sha"
    case "$part_file" in
      "$model_file".part-*) ;;
      *) die "unexpected part name for $model_file: $part_file" ;;
    esac
    case "$part_file" in
      */*|*..*|'') die "unsafe part name for $model_file: $part_file" ;;
    esac
    test -f "$part_file" || die "missing part file: $part_file"
    verify_one "$part_sha" "$part_file" >/dev/null
    printf '%s\n' "$part_file" >>"$parts_file"
    count=$((count + 1))
  done < "$manifest"

  test "$count" -eq "$expected_count" ||
    die "$model_file manifest row count mismatch: expected $expected_count got $count"

  found_parts=$(count_files "$model_file.part-*")
  test "$found_parts" -eq "$expected_count" ||
    die "$model_file part file count mismatch: expected $expected_count got $found_parts"

  final_sha=$(while IFS= read -r part_file; do cat "$part_file"; done < "$parts_file" | sha256sum | awk '{print $1}')
  test "$final_sha" = "$expected_final_sha" ||
    die "$model_file streamed final hash mismatch: $final_sha"

  printf 'asset-parts-ok: %s parts=%s final_sha=%s\n' \
    "$model_file" "$count" "$final_sha"
}

(
  cd "$asset_dir" || exit 1

  total_files=$(find . -maxdepth 1 -type f | wc -l | awk '{print $1}')
  test "$total_files" -eq 69 ||
    die "release asset file count mismatch: expected 69 got $total_files"

  test -f "$text_archive" || die "missing Dense text-config archive: $text_archive"
  verify_one "$text_archive_sha" "$text_archive" >/dev/null
  printf 'asset-text-config-ok: %s sha=%s\n' "$text_archive" "$text_archive_sha"

  require_sidecar "$dense_model" "$dense_final_sha"
  require_sidecar "$moe_model" "$moe_final_sha"

  verify_manifest "$dense_model" 28 "$dense_final_sha" "$tmp_dense"
  verify_manifest "$moe_model" 36 "$moe_final_sha" "$tmp_moe"

  printf 'vNext release asset bundle verified: files=%s dense_parts=28 moe_parts=36\n' "$total_files"
)
