#!/bin/sh
set -eu

usage() {
    cat >&2 <<'EOF'
usage: run_vnext_profile_benchmark.sh <vnext-launch-run-dir>

Runs the generated vNext benchmark environment through the normal Qwen
promotion ladder:
  8 warmups -> c1_128 uncapped strict -> c1_2000 -> c1_10000
EOF
}

die() {
    printf '%s\n' "error: $*" >&2
    exit 2
}

[ "$#" -eq 1 ] || {
    usage
    exit 2
}

run_dir=$1
env_file="${run_dir}/benchmark_env.sh"

[ -f "$env_file" ] || die "missing ${env_file}"

# shellcheck disable=SC1090
. "$env_file"

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)

PROFILE=${QWEN36_PROFILE:-${PROFILE:-}}
[ -n "${PROFILE:-}" ] || die "benchmark_env.sh did not define QWEN36_PROFILE"

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}
PROXY_PORT=${PROXY_PORT:-19080}
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-${run_dir}/benchmark_ladder}
RUN_DIR=${RUN_DIR:-${RUN_ROOT}/${PROFILE}_${STAMP}}

PRE_MEASURE_WARMUP_REQUESTS=${PRE_MEASURE_WARMUP_REQUESTS:-8}
PRE_MEASURE_WARMUP_MAX_TOKENS=${PRE_MEASURE_WARMUP_MAX_TOKENS:-2000}
PRE_MEASURE_WARMUP_PROMPT_REPEAT=${PRE_MEASURE_WARMUP_PROMPT_REPEAT:-32}
PRE_MEASURE_WARMUP_TIMEOUT=${PRE_MEASURE_WARMUP_TIMEOUT:-3600}
READY_TIMEOUT=${READY_TIMEOUT:-60}
STRICT_TIMEOUT=${STRICT_TIMEOUT:-2400}
LONG_TIMEOUT=${LONG_TIMEOUT:-7200}
PROMPT_REPEAT=${PROMPT_REPEAT:-32}
USE_BEGIN_THINK_PROXY=${USE_BEGIN_THINK_PROXY:-1}
MODEL=${MODEL:-}

case "$PROFILE" in
    dense27b_tp8_fullbar_p2pon)
        MODEL=${MODEL:-Qwen3.6-27B}
        TITLE="Qwen3.6 27B Dense Portable Decode Tiers"
        ;;
    moe35b_tp4_fullbar_p2pon|moe35b_tp8_fullbar_p2pon)
        MODEL=${MODEL:-Qwen/Qwen3.6-35B-A3B}
        TITLE="Qwen3.6 35B MoE Portable Decode Tiers"
        ;;
    *)
        die "unknown PROFILE=${PROFILE}"
        ;;
esac

mkdir -p "$RUN_DIR"

PROXY_PID=
cleanup() {
    if [ -n "$PROXY_PID" ]; then
        kill "$PROXY_PID" >/dev/null 2>&1 || true
        wait "$PROXY_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup 0 HUP INT TERM

cat >"${RUN_DIR}/settings.txt" <<EOF_SETTINGS
profile=${PROFILE}
host=${HOST}
port=${PORT}
proxy_port=${PROXY_PORT}
model=${MODEL}
pre_measure_warmup_requests=${PRE_MEASURE_WARMUP_REQUESTS}
pre_measure_warmup_max_tokens=${PRE_MEASURE_WARMUP_MAX_TOKENS}
pre_measure_warmup_prompt_repeat=${PRE_MEASURE_WARMUP_PROMPT_REPEAT}
pre_measure_warmup_timeout=${PRE_MEASURE_WARMUP_TIMEOUT}
prompt_repeat=${PROMPT_REPEAT}
ready_timeout=${READY_TIMEOUT}
strict_timeout=${STRICT_TIMEOUT}
long_timeout=${LONG_TIMEOUT}
use_begin_think_proxy=${USE_BEGIN_THINK_PROXY}
strict_tier=uncapped_no_max_tokens_require_think_close
EOF_SETTINGS

printf '%s start %s\n' "$TITLE" "$(date -u +%F_%T_UTC)"
printf 'Run dir: %s\n' "$RUN_DIR"

while ! curl -fsS --max-time 5 "http://${HOST}:${PORT}/v1/models" >/dev/null; do
    printf 'backend not ready at %s\n' "$(date -u +%F_%T_UTC)"
    sleep 5
done

capture_endpoint="http://${HOST}:${PORT}/v1"
if [ "$USE_BEGIN_THINK_PROXY" = "1" ]; then
    python3 "${script_dir}/begin_think_proxy.py" \
        --listen-host 127.0.0.1 \
        --listen-port "$PROXY_PORT" \
        --backend "http://${HOST}:${PORT}" \
        --quiet >"${RUN_DIR}/begin_think_proxy.log" 2>&1 &
    PROXY_PID=$!

    while ! curl -fsS --max-time 5 "http://127.0.0.1:${PROXY_PORT}/v1/models" >/dev/null; do
        if ! kill -0 "$PROXY_PID" >/dev/null 2>&1; then
            printf '%s\n' "begin-think proxy exited early" >&2
            tail -n 80 "${RUN_DIR}/begin_think_proxy.log" >&2 || true
            exit 1
        fi
        sleep 2
    done
    capture_endpoint="http://127.0.0.1:${PROXY_PORT}/v1"
fi

warmup_index=1
while [ "$warmup_index" -le "$PRE_MEASURE_WARMUP_REQUESTS" ]; do
    warmup_dir="${RUN_DIR}/pre_measure_warmup_${warmup_index}_proxy"
    printf 'pre-measure warmup %s/%s\n' "$warmup_index" "$PRE_MEASURE_WARMUP_REQUESTS"
    python3 "${script_dir}/run_chat_capture.py" \
        --endpoint "$capture_endpoint" \
        --metrics-url "http://${HOST}:${PORT}/metrics" \
        --model "$MODEL" \
        --out-dir "$warmup_dir" \
        --max-tokens "$PRE_MEASURE_WARMUP_MAX_TOKENS" \
        --prompt-repeat "$PRE_MEASURE_WARMUP_PROMPT_REPEAT" \
        --ready-timeout "$READY_TIMEOUT" \
        --request-timeout "$PRE_MEASURE_WARMUP_TIMEOUT" \
        >"${RUN_DIR}/pre_measure_warmup_${warmup_index}.log" 2>&1 || {
            cat "${RUN_DIR}/pre_measure_warmup_${warmup_index}.log"
            exit 1
        }
    cat "${RUN_DIR}/pre_measure_warmup_${warmup_index}.log"
    warmup_index=$((warmup_index + 1))
done

python3 "${script_dir}/run_chat_capture.py" \
    --endpoint "$capture_endpoint" \
    --metrics-url "http://${HOST}:${PORT}/metrics" \
    --model "$MODEL" \
    --out-dir "${RUN_DIR}/c1_128_strict_proxy" \
    --no-max-tokens \
    --prompt-repeat "$PROMPT_REPEAT" \
    --ready-timeout "$READY_TIMEOUT" \
    --request-timeout "$STRICT_TIMEOUT" \
    --require-think-close >"${RUN_DIR}/c1_128_strict.log" 2>&1 || {
        status=$?
        cat "${RUN_DIR}/c1_128_strict.log"
        printf '%s\n' "$status" >"${RUN_DIR}/c1_128_status.txt"
        exit "$status"
    }
cat "${RUN_DIR}/c1_128_strict.log"
printf '%s\n' 0 >"${RUN_DIR}/c1_128_status.txt"

for tokens in 2000 10000; do
    python3 "${script_dir}/run_chat_capture.py" \
        --endpoint "$capture_endpoint" \
        --metrics-url "http://${HOST}:${PORT}/metrics" \
        --model "$MODEL" \
        --out-dir "${RUN_DIR}/c1_${tokens}_proxy" \
        --max-tokens "$tokens" \
        --prompt-repeat "$PROMPT_REPEAT" \
        --ready-timeout "$READY_TIMEOUT" \
        --request-timeout "$LONG_TIMEOUT" >"${RUN_DIR}/c1_${tokens}.log" 2>&1 || {
            cat "${RUN_DIR}/c1_${tokens}.log"
            exit 1
        }
    cat "${RUN_DIR}/c1_${tokens}.log"
done

python3 - "$RUN_DIR" "$TITLE" >"${RUN_DIR}/summary.md" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
title = sys.argv[2]
print(f"# {title}")
print()
print(f"Run dir: `{root}`")
print()
print("| case | completion tokens | client TPS | backend decode TPS | elapsed s | finish | strict gate status |")
print("| --- | ---: | ---: | ---: | ---: | --- | --- |")
for label, subdir in [
    ("c1_128_strict", "c1_128_strict_proxy"),
    ("c1_2000", "c1_2000_proxy"),
    ("c1_10000", "c1_10000_proxy"),
]:
    data = json.loads((root / subdir / "summary.json").read_text())
    strict_status = data.get("qwen_gate_valid")
    if label != "c1_128_strict":
        strict_status = "n/a (fixed-token)"
    print(
        f"| `{label}` | {data.get('completion_tokens')} | "
        f"{data.get('wall_completion_tps', 0):.3f} | "
        f"{data.get('vllm_decode_tps', 0):.3f} | "
        f"{data.get('elapsed_seconds', 0):.3f} | "
        f"{data.get('finish_reason')} | {strict_status} |"
    )
PY

cat "${RUN_DIR}/summary.md"
