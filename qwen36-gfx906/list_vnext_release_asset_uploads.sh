#!/bin/sh
set -eu

usage() {
  cat >&2 <<'EOF'
Usage:
  list_vnext_release_asset_uploads.sh <release-asset-dir>

Prints the exact vNext release asset file paths to upload, one per line, in a
stable order. Run verify_vnext_release_asset_bundle.sh first for full hash
validation.
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

dense_model=Qwen3.6-27B-F16.gguf
moe_model=Qwen3.6-35B-A3B-F16.gguf
text_archive=Qwen3.6-27B-text-config-eosfix.tar.gz

print_required() {
  file_name=$1
  path=$asset_dir/$file_name

  test -f "$path" || die "missing release asset: $file_name"
  printf '%s\n' "$path"
}

print_parts() {
  model_file=$1
  expected_count=$2
  manifest=$asset_dir/$model_file.parts.sha256

  test -f "$manifest" || die "missing part manifest: $model_file.parts.sha256"

  count=0
  while read -r part_sha part_file extra; do
    test -n "$part_sha" || continue
    test -n "$part_file" || die "malformed manifest row in $model_file.parts.sha256"
    test -z "${extra:-}" || die "part filenames must not contain whitespace: $part_file"
    case "$part_file" in
      "$model_file".part-*) ;;
      *) die "unexpected part name for $model_file: $part_file" ;;
    esac
    case "$part_file" in
      */*|*..*|'') die "unsafe part name for $model_file: $part_file" ;;
    esac
    print_required "$part_file"
    count=$((count + 1))
  done < "$manifest"

  test "$count" -eq "$expected_count" ||
    die "$model_file manifest row count mismatch: expected $expected_count got $count"
}

print_required "$text_archive"

print_required "$dense_model.parts.sha256"
print_required "$dense_model.sha256"
print_parts "$dense_model" 28

print_required "$moe_model.parts.sha256"
print_required "$moe_model.sha256"
print_parts "$moe_model" 36
