#!/usr/bin/env bash
set -euo pipefail

TARGET_GIB="30"
DEVICE="all"
INSTALL_DEPS=0
SKIP_DMESG=0
SKIP_ROCM_SMI=0
KEEP_BUILD=0
BUILD_DIR=""

usage() {
  cat <<'USAGE'
GFX906 / MI50 32GB VRAM QC Field Check

Best-effort educational diagnostic for MI50 32GB / GFX906-class cards.
This tool only allocates and verifies transient GPU memory through HIP.
It does not flash firmware, modify BIOS/VBIOS, alter PCIe configuration,
write to block devices, upload logs, or make certification/warranty claims.

Usage:
  ./mi50_vram_qc.sh [options]

Options:
  --target-gib <value>  Allocation size per selected device in GiB (default: 30)
  --device <id|all>     HIP device id to test, or all devices (default: all)
  --install-deps        Explicitly try apt dependency installation
  --skip-dmesg          Skip kernel-log evidence collection
  --skip-rocm-smi       Skip rocm-smi visibility output
  --keep-build          Keep the temporary build directory
  --help                Show this help

Environment:
  HIPCC                 Optional path to hipcc

Exit status:
  0 if all selected HIP allocation/check tests pass.
  Nonzero if setup fails or any selected device fails.
USAGE
}

log_section() {
  printf '\n== %s ==\n' "$1"
}

cleanup() {
  if [[ -n "${BUILD_DIR}" && -d "${BUILD_DIR}" && "${KEEP_BUILD}" != "1" ]]; then
    rm -rf "${BUILD_DIR}"
  elif [[ -n "${BUILD_DIR}" && -d "${BUILD_DIR}" ]]; then
    printf 'kept build directory: %s\n' "${BUILD_DIR}"
  fi
}
trap cleanup EXIT

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

is_number() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-gib)
      [[ $# -ge 2 ]] || die "--target-gib requires a value"
      TARGET_GIB="$2"
      shift 2
      ;;
    --device)
      [[ $# -ge 2 ]] || die "--device requires all or a numeric HIP device id"
      DEVICE="$2"
      shift 2
      ;;
    --install-deps)
      INSTALL_DEPS=1
      shift
      ;;
    --skip-dmesg)
      SKIP_DMESG=1
      shift
      ;;
    --skip-rocm-smi)
      SKIP_ROCM_SMI=1
      shift
      ;;
    --keep-build)
      KEEP_BUILD=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

is_number "${TARGET_GIB}" || die "--target-gib must be a positive number"
awk -v value="${TARGET_GIB}" 'BEGIN { exit !(value > 0) }' || die "--target-gib must be greater than zero"
if [[ "${DEVICE}" != "all" && ! "${DEVICE}" =~ ^[0-9]+$ ]]; then
  die "--device must be all or a numeric HIP device id"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SRC="${SCRIPT_DIR}/vram_check.cpp"
[[ -f "${SRC}" ]] || die "missing source file: ${SRC}"

cat <<'INTRO'
GFX906 / MI50 32GB VRAM QC Field Check

Purpose: public education and best-effort hardware verification methodology.
Safety: allocates/checks transient GPU memory through HIP only.
This is not certification, warranty testing, resale support, or official validation.
INTRO

install_deps() {
  log_section "Optional dependency installation"
  if ! command -v apt-get >/dev/null 2>&1; then
    die "--install-deps was requested, but apt-get is not available"
  fi

  local runner=()
  if [[ "${EUID}" == "0" ]]; then
    runner=(apt-get)
  elif command -v sudo >/dev/null 2>&1; then
    runner=(sudo apt-get)
  else
    die "--install-deps requires root or sudo"
  fi

  printf 'Installing suggested packages with apt. Package names may vary by ROCm repository setup.\n'
  "${runner[@]}" update
  "${runner[@]}" install -y rocm-smi libamdhip64-dev hipcc
}

find_hipcc() {
  if [[ -n "${HIPCC:-}" ]]; then
    [[ -x "${HIPCC}" ]] || die "HIPCC is set but not executable: ${HIPCC}"
    printf '%s\n' "${HIPCC}"
    return 0
  fi
  if command -v hipcc >/dev/null 2>&1; then
    command -v hipcc
    return 0
  fi
  if [[ -x /opt/rocm/bin/hipcc ]]; then
    printf '%s\n' /opt/rocm/bin/hipcc
    return 0
  fi
  return 1
}

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  install_deps
fi

HIPCC_BIN="$(find_hipcc || true)"
if [[ -z "${HIPCC_BIN}" ]]; then
  cat >&2 <<'EOF'
error: hipcc was not found.

Install ROCm HIP development tools for your distribution, or set HIPCC:
  HIPCC=/opt/rocm/bin/hipcc ./mi50_vram_qc.sh

On apt-based ROCm systems, you may opt in to a best-effort install attempt:
  ./mi50_vram_qc.sh --install-deps
EOF
  exit 1
fi

log_section "Tool configuration"
printf 'hipcc: %s\n' "${HIPCC_BIN}"
printf 'target allocation: %s GiB\n' "${TARGET_GIB}"
printf 'device selection: %s\n' "${DEVICE}"

log_section "Firmware / device-name hints"
if command -v fwupdmgr >/dev/null 2>&1; then
  if fwupdmgr get-devices --json 2>/dev/null | grep -Eio 'Vega20|VEGA20|GFX906|MI50|Instinct' | sort -u; then
    :
  else
    printf 'No Vega20 / GFX906 / MI50 hints found in fwupdmgr output.\n'
  fi
else
  printf 'fwupdmgr not available; skipping firmware/device-name hints.\n'
fi

if [[ "${SKIP_DMESG}" != "1" ]]; then
  log_section "Kernel-log hints"
  if dmesg >/dev/null 2>&1; then
    if dmesg | grep -Ei 'vega20|gfx906|mi50|instinct|amdgpu|kfd' | tail -50; then
      :
    else
      printf 'No matching kernel-log hints found.\n'
    fi
  else
    printf 'dmesg is unavailable or requires privileges on this host; skipping.\n'
  fi
else
  log_section "Kernel-log hints"
  printf 'skipped by --skip-dmesg\n'
fi

log_section "Kernel-reported VRAM totals"
found_vram=0
for mem_file in /sys/class/drm/card*/device/mem_info_vram_total; do
  [[ -e "${mem_file}" ]] || continue
  found_vram=1
  card_dir="$(dirname "$(dirname "${mem_file}")")"
  bytes="$(cat "${mem_file}" 2>/dev/null || printf 'unknown')"
  if [[ "${bytes}" =~ ^[0-9]+$ ]]; then
    gib="$(awk -v bytes="${bytes}" 'BEGIN { printf "%.2f", bytes / 1073741824 }')"
    printf '%s: %s bytes (%s GiB)\n' "${card_dir}" "${bytes}" "${gib}"
  else
    printf '%s: %s\n' "${card_dir}" "${bytes}"
  fi
done
if [[ "${found_vram}" == "0" ]]; then
  printf 'No /sys/class/drm/card*/device/mem_info_vram_total files found.\n'
fi

if [[ "${SKIP_ROCM_SMI}" != "1" ]]; then
  log_section "rocm-smi visibility"
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showproductname --showmeminfo vram --showuse --showtemp 2>/dev/null || rocm-smi || true
  else
    printf 'rocm-smi not available; skipping.\n'
  fi
else
  log_section "rocm-smi visibility"
  printf 'skipped by --skip-rocm-smi\n'
fi

BUILD_DIR="$(mktemp -d -t mi50-vram-qc.XXXXXX)"
BIN="${BUILD_DIR}/vram_check"

log_section "Build"
printf 'build directory: %s\n' "${BUILD_DIR}"
"${HIPCC_BIN}" -O2 -std=c++17 "${SRC}" -o "${BIN}"

log_section "HIP VRAM allocation/check"
if "${BIN}" --target-gib "${TARGET_GIB}" --device "${DEVICE}"; then
  log_section "Final result"
  printf 'PASS: selected device checks passed.\n'
  exit 0
else
  status=$?
  log_section "Final result"
  printf 'FAIL: one or more selected device checks failed.\n'
  exit "${status}"
fi
