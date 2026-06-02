#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" != "0" ]]; then
  echo "error: run as root" >&2
  exit 1
fi

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 BUILD_DIR ROOT_NAME..." >&2
  exit 1
fi

build_dir="$1"
shift

case "${build_dir}" in
  */q36b) ;;
  *)
    echo "error: refusing cleanup outside a q36b build dir: ${build_dir}" >&2
    exit 1
    ;;
esac

for root_name in "$@"; do
  case "${root_name}" in
    .d|.d-*) ;;
    *)
      echo "error: refusing unexpected root name: ${root_name}" >&2
      exit 1
      ;;
  esac
done

for root_name in "$@"; do
  root="${build_dir}/${root_name}"
  if [[ -e "${root}" ]]; then
    pkill -f "${root}" || true
  fi
done

sleep 2

lazy_unmount_tree() {
  local root="$1"
  awk -v root="${root}" '$5 == root || index($5, root "/") == 1 { print $5 }' /proc/self/mountinfo \
    | sort -r \
    | while IFS= read -r mountpoint; do
        [[ -n "${mountpoint}" ]] && umount -l "${mountpoint}" || true
      done || true

  if command -v findmnt >/dev/null 2>&1; then
    findmnt -R -n -o TARGET "${root}" 2>/dev/null \
      | sort -r \
      | while IFS= read -r mountpoint; do
          [[ -n "${mountpoint}" ]] && umount -l "${mountpoint}" || true
        done || true
  fi
}

for root_name in "$@"; do
  root="${build_dir}/${root_name}"
  if [[ -e "${root}" ]]; then
    lazy_unmount_tree "${root}"
    rm -rf "${root}" || {
      lazy_unmount_tree "${root}"
      rm -rf "${root}"
    }
  fi
done

archive_dir="${build_dir}/.repro-docker-archives"
rm -f "${archive_dir}/qwen36-gfx906-c1-topk8-fastpath-reproducible.docker.tar"
rm -f "${archive_dir}/qwen36-gfx906-c1-topk8-fastpath-reproducible.docker.tar.sha256"
rm -f "${archive_dir}/qwen36-layer-file-manifest.tsv"

df -h "${build_dir}"
