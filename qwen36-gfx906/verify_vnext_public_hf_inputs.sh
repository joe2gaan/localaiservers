#!/bin/sh
set -eu

usage() {
  cat >&2 <<'EOF'
Usage:
  verify_vnext_public_hf_inputs.sh <local-model-root>

Verifies that the public HF model directories staged under <local-model-root>
satisfy the vNext HF profile contracts. Set STAGE_HF_PUBLIC_INPUTS=1 to run
the pinned public `hf download` commands before validation.

Environment:
  HF_HOME                 Hugging Face cache directory.
  STAGE_HF_PUBLIC_INPUTS  Set to 1 to download public HF repos first.
  HF_DENSE_REVISION       Optional override; defaults to the pinned release value.
  HF_MOE_REVISION         Optional override; defaults to the pinned release value.
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

local_model_root=$1
dense_revision=${HF_DENSE_REVISION:-6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}
moe_revision=${HF_MOE_REVISION:-995ad96eacd98c81ed38be0c5b274b04031597b0}

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

case "$dense_revision" in
  6a9e13bd6fc8f0983b9b99948120bc37f49c13e9) ;;
  *) die "HF_DENSE_REVISION does not match the pinned release value" ;;
esac
case "$moe_revision" in
  995ad96eacd98c81ed38be0c5b274b04031597b0) ;;
  *) die "HF_MOE_REVISION does not match the pinned release value" ;;
esac

safe_root "local model root" "$local_model_root"

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
dense_dir="$local_model_root/qwen36-27b-hf"
moe_dir="$local_model_root/qwen36-35b-a3b-hf"
launch_root="$local_model_root/.vnext-public-hf-gate"

check_symlinked_model_dir() {
  label=$1
  model_dir=$2

  if [ -L "$model_dir" ]; then
    target=$(readlink "$model_dir") ||
      die "could not resolve symlinked $label directory"
    case "$target" in
      /*)
        resolved=$target
        ;;
      *)
        resolved=$(CDPATH= cd -- "$(dirname -- "$model_dir")" && pwd)/$target
        ;;
    esac
    test -d "$resolved" ||
      die "symlinked $label directory does not resolve to a directory"
    printf 'public-hf-symlink-dir-ok: %s\n' "$label"
  fi
}

case "${STAGE_HF_PUBLIC_INPUTS:-0}" in
  0|1) ;;
  *) die "STAGE_HF_PUBLIC_INPUTS must be 0 or 1" ;;
esac

if [ "${STAGE_HF_PUBLIC_INPUTS:-0}" = "1" ]; then
  hf_home=${HF_HOME:-$local_model_root/.hf-cache}
  safe_root "HF cache root" "$hf_home"
  command -v hf >/dev/null 2>&1 || die "hf command not found"
  mkdir -p "$local_model_root" "$hf_home"
  for stage_dir in "$dense_dir" "$moe_dir"; do
    if [ -L "$stage_dir" ]; then
      die "STAGE_HF_PUBLIC_INPUTS=1 requires writable non-symlink local-dir targets; remove $stage_dir or rerun with STAGE_HF_PUBLIC_INPUTS=0 after the pinned HF package is already staged"
    fi
    if [ -e "$stage_dir" ] && [ ! -w "$stage_dir" ]; then
      die "STAGE_HF_PUBLIC_INPUTS=1 target is not writable: $stage_dir"
    fi
  done
  HF_HOME="$hf_home" hf download Qwen/Qwen3.6-27B \
    --revision "$dense_revision" \
    --local-dir "$dense_dir"
  HF_HOME="$hf_home" hf download Qwen/Qwen3.6-35B-A3B \
    --revision "$moe_revision" \
    --local-dir "$moe_dir"
fi

check_symlinked_model_dir "Dense HF" "$dense_dir"
check_symlinked_model_dir "MoE HF" "$moe_dir"

test -f "$dense_dir/config.json" || die "missing Dense HF config.json: $dense_dir"
test -f "$moe_dir/config.json" || die "missing MoE HF config.json: $moe_dir"

check_hf_package_complete() {
  label=$1
  model_dir=$2

  test -f "$model_dir/tokenizer.json" ||
    die "missing $label tokenizer.json: $model_dir"
  test -f "$model_dir/tokenizer_config.json" ||
    die "missing $label tokenizer_config.json: $model_dir"
  test -f "$model_dir/model.safetensors.index.json" ||
    die "missing $label model.safetensors.index.json: $model_dir"

  python3 - "$model_dir" "$label" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
label = sys.argv[2]
index_path = root / "model.safetensors.index.json"

try:
    index = json.loads(index_path.read_text())
except Exception as exc:
    raise SystemExit(f"error: could not parse {label} safetensors index: {exc}")

weight_map = index.get("weight_map")
if not isinstance(weight_map, dict) or not weight_map:
    raise SystemExit(f"error: {label} safetensors index has no weight_map")

shards = sorted(set(weight_map.values()))
if not shards:
    raise SystemExit(f"error: {label} safetensors index has no shard files")

missing = []
empty = []
pointers = []
for shard in shards:
    if "/" in shard or "\\" in shard or ".." in shard:
        raise SystemExit(f"error: unsafe {label} shard name in index: {shard}")
    path = root / shard
    if not path.is_file():
        missing.append(shard)
        continue
    size = path.stat().st_size
    if size <= 0:
        empty.append(shard)
        continue
    with path.open("rb") as fh:
        prefix = fh.read(96)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        pointers.append(shard)

if missing:
    raise SystemExit(f"error: missing {label} shard(s): {', '.join(missing[:8])}")
if empty:
    raise SystemExit(f"error: empty {label} shard(s): {', '.join(empty[:8])}")
if pointers:
    raise SystemExit(f"error: unresolved Git LFS pointer in {label} shard(s): {', '.join(pointers[:8])}")

print(f"public-hf-shards-ok: {label} shards={len(shards)}")
PY
}

check_hf_package_complete "Dense HF" "$dense_dir"
check_hf_package_complete "MoE HF" "$moe_dir"

make -C "$repo_root/tools/model-format-probe" model-format-probe >/dev/null

for profile in hf-dense27b-tp8 hf-moe35b-tp4 hf-moe35b-tp8; do
  run_dir="$launch_root/$profile"
  rm -rf "$run_dir"
  LOCAL_MODEL_ROOT="$local_model_root" \
    "$script_dir/vnext_repro_launcher.sh" \
      --profile "$profile" \
      --out "$run_dir" >/dev/null
  "$script_dir/verify_vnext_launch_artifact.sh" "$run_dir" >/dev/null
  printf 'public-hf-profile-ok: %s\n' "$profile"
done

printf 'vNext public HF input gate passed\n'
