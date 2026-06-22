#!/usr/bin/env bash
set -euo pipefail

# Read-only host preflight for the v0.2 ROCm7.2 GFX906 profiles.
# This script does not patch amdgpu, flash VBIOS, change kernel settings, or
# start workloads. It checks the visible host state needed before comparing
# local measurements with the published release numbers.

PROFILE="${QWEN36_PROFILE:-moe35b_tp4_fullbar_p2pon}"
MIN_GPU_BAR_GIB="${MIN_GPU_BAR_GIB:-32}"
STRICT_PLATFORM_PREREQS="${STRICT_PLATFORM_PREREQS:-0}"
EXPECTED_GPU_VBIOS="${EXPECTED_GPU_VBIOS:-113-D1631700-111}"

case "${PROFILE}" in
  dense27b_tp8_fullbar_p2pon|moe35b_tp8_fullbar_p2pon)
    REQUIRED_GPU_COUNT="${REQUIRED_GPU_COUNT:-8}"
    ;;
  moe35b_tp4_fullbar_p2pon)
    REQUIRED_GPU_COUNT="${REQUIRED_GPU_COUNT:-4}"
    ;;
  *)
    echo "error: unknown QWEN36_PROFILE=${PROFILE}" >&2
    echo "valid profiles: dense27b_tp8_fullbar_p2pon, moe35b_tp4_fullbar_p2pon, moe35b_tp8_fullbar_p2pon" >&2
    exit 2
    ;;
esac

failures=0
warnings=0

pass() {
  printf 'PASS: %s\n' "$*"
}

warn() {
  warnings=$((warnings + 1))
  printf 'WARN: %s\n' "$*"
}

fail() {
  failures=$((failures + 1))
  printf 'FAIL: %s\n' "$*"
}

human_gib() {
  awk -v bytes="$1" 'BEGIN { printf "%.2f", bytes / 1024 / 1024 / 1024 }'
}

largest_bar_bytes() {
  local resource_file="$1"
  python3 - "${resource_file}" <<'PY'
import sys

max_size = 0
with open(sys.argv[1], "r", encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        parts = line.split()
        if len(parts) < 2:
            continue
        start = int(parts[0], 16)
        end = int(parts[1], 16)
        if end >= start:
            max_size = max(max_size, end - start + 1)

print(max_size)
PY
}

echo "GFX906 v0.2 host platform preflight"
echo "profile=${PROFILE}"
echo "required_gpu_count=${REQUIRED_GPU_COUNT}"
echo "min_gpu_bar_gib=${MIN_GPU_BAR_GIB}"
echo "expected_gpu_vbios=${EXPECTED_GPU_VBIOS}"
echo

if [[ -d /sys/module/amdgpu ]]; then
  pass "amdgpu kernel module is loaded"
else
  fail "amdgpu kernel module is not loaded"
fi

if command -v modinfo >/dev/null 2>&1; then
  amdgpu_version="$(modinfo -F version amdgpu 2>/dev/null || true)"
  if [[ -n "${amdgpu_version}" ]]; then
    echo "amdgpu module version: ${amdgpu_version}"
  else
    warn "could not read amdgpu module version"
  fi
else
  warn "modinfo is unavailable; amdgpu module version not checked"
fi

if command -v rocminfo >/dev/null 2>&1; then
  gfx906_count="$(rocminfo 2>/dev/null | awk '/Name:[[:space:]]+gfx906/ {count++} END {print count + 0}')"
  if [[ "${gfx906_count}" -ge "${REQUIRED_GPU_COUNT}" ]]; then
    pass "rocminfo reports ${gfx906_count} gfx906 agents"
  else
    warn "rocminfo reports ${gfx906_count} gfx906 agents; expected at least ${REQUIRED_GPU_COUNT}"
  fi
else
  warn "rocminfo is unavailable; gfx906 agent count not checked"
fi

declare -a gfx906_devices=()
for dev in /sys/bus/pci/devices/*; do
  [[ -r "${dev}/vendor" && -r "${dev}/device" && -r "${dev}/class" ]] || continue
  vendor="$(<"${dev}/vendor")"
  device="$(<"${dev}/device")"
  class="$(<"${dev}/class")"
  [[ "${vendor}" == "0x1002" ]] || continue
  [[ "${class}" == 0x03* ]] || continue
  case "${device}" in
    0x66a0|0x66a1|0x66a3|0x66a7|0x66af)
      gfx906_devices+=("$(basename "${dev}")")
      ;;
  esac
done

if [[ "${#gfx906_devices[@]}" -ge "${REQUIRED_GPU_COUNT}" ]]; then
  pass "found ${#gfx906_devices[@]} likely gfx906 PCI display/controller devices"
else
  fail "found ${#gfx906_devices[@]} likely gfx906 PCI devices; expected at least ${REQUIRED_GPU_COUNT}"
fi

full_bar_count=0
for bdf in "${gfx906_devices[@]}"; do
  resource_file="/sys/bus/pci/devices/${bdf}/resource"
  if [[ ! -r "${resource_file}" ]]; then
    warn "${bdf}: resource file unavailable"
    continue
  fi
  largest_bytes="$(largest_bar_bytes "${resource_file}")"
  largest_gib="$(human_gib "${largest_bytes}")"
  printf 'GPU %s largest_BAR_GiB=%s\n' "${bdf}" "${largest_gib}"
  if awk -v got="${largest_gib}" -v need="${MIN_GPU_BAR_GIB}" 'BEGIN { exit !(got + 0 >= need + 0) }'; then
    full_bar_count=$((full_bar_count + 1))
  fi
done

if [[ "${full_bar_count}" -ge "${REQUIRED_GPU_COUNT}" ]]; then
  pass "${full_bar_count} likely gfx906 GPUs have largest BAR >= ${MIN_GPU_BAR_GIB} GiB"
else
  fail "${full_bar_count} likely gfx906 GPUs have largest BAR >= ${MIN_GPU_BAR_GIB} GiB; expected at least ${REQUIRED_GPU_COUNT}"
fi

if [[ -d /sys/class/kfd/kfd/topology/nodes ]]; then
  kfd_gpu_nodes="$(find /sys/class/kfd/kfd/topology/nodes -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l | awk '{print $1}')"
  io_link_count="$(find /sys/class/kfd/kfd/topology/nodes -path '*/io_links/*/properties' -type f 2>/dev/null | wc -l | awk '{print $1}')"
  if [[ "${kfd_gpu_nodes}" -gt 0 ]]; then
    pass "KFD topology is visible (${kfd_gpu_nodes} nodes, ${io_link_count} io-link property files)"
  else
    warn "KFD topology directory exists but no nodes were found"
  fi
else
  warn "KFD topology is unavailable; ROCm peer/topology state not checked"
fi

if command -v rocm-smi >/dev/null 2>&1; then
  if rocm-smi --showvbios >/tmp/gfx906-host-platform-rocm-smi-vbios.$$ 2>/tmp/gfx906-host-platform-rocm-smi-vbios.err.$$; then
    vbios_match_count="$(grep -F "${EXPECTED_GPU_VBIOS}" /tmp/gfx906-host-platform-rocm-smi-vbios.$$ | wc -l | awk '{print $1}')"
    if [[ "${vbios_match_count}" -ge "${REQUIRED_GPU_COUNT}" ]]; then
      pass "rocm-smi reports ${vbios_match_count} GPUs with VBIOS ${EXPECTED_GPU_VBIOS}"
    else
      warn "rocm-smi reports ${vbios_match_count} GPUs with VBIOS ${EXPECTED_GPU_VBIOS}; expected at least ${REQUIRED_GPU_COUNT}"
    fi
    rm -f /tmp/gfx906-host-platform-rocm-smi-vbios.$$ /tmp/gfx906-host-platform-rocm-smi-vbios.err.$$
  else
    warn "rocm-smi --showvbios failed; GPU VBIOS revision needs manual review"
    rm -f /tmp/gfx906-host-platform-rocm-smi-vbios.$$ /tmp/gfx906-host-platform-rocm-smi-vbios.err.$$
  fi

  if rocm-smi --showtopo >/tmp/gfx906-host-platform-rocm-smi-topo.$$ 2>/tmp/gfx906-host-platform-rocm-smi-topo.err.$$; then
    pass "rocm-smi --showtopo completed"
    rm -f /tmp/gfx906-host-platform-rocm-smi-topo.$$ /tmp/gfx906-host-platform-rocm-smi-topo.err.$$
  else
    warn "rocm-smi --showtopo failed; topology/P2P state needs manual review"
    rm -f /tmp/gfx906-host-platform-rocm-smi-topo.$$ /tmp/gfx906-host-platform-rocm-smi-topo.err.$$
  fi
else
  warn "rocm-smi is unavailable; topology/P2P state not checked"
fi

echo
echo "Host amdgpu patch status:"
echo "- The v0.2.1 public repo does not bundle the host amdgpu source patch."
echo "- This preflight cannot prove the exact amdgpu patch hash or source base."
echo "- Treat a failing full-BAR/P2P preflight as not comparable with the published v0.2 numbers."
echo "- Expected standardized GPU VBIOS revision: ${EXPECTED_GPU_VBIOS}."

if [[ "${warnings}" -gt 0 ]]; then
  echo
  echo "warnings=${warnings}"
fi

if [[ "${failures}" -gt 0 ]]; then
  echo "failures=${failures}"
  exit 1
fi

if [[ "${STRICT_PLATFORM_PREREQS}" == "1" && "${warnings}" -gt 0 ]]; then
  echo "STRICT_PLATFORM_PREREQS=1 and warnings were emitted"
  exit 1
fi

echo "host platform preflight completed"
