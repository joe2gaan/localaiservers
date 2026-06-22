#!/usr/bin/env python3
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request


API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8001")
MODEL = os.environ.get("MODEL", "")
METRIC_MODEL = os.environ.get("METRIC_MODEL", "")
PROMPT_TEXT = os.environ.get(
    "PROMPT_TEXT",
    "Throughput benchmark. Continue generating plain neutral benchmark prose. "
    "Do not summarize and do not stop early.\n\n",
)
PROMPT_REPEAT = int(os.environ.get("PROMPT_REPEAT", "1"))
WARMUP_REQUESTS = int(os.environ.get("WARMUP_REQUESTS", "0"))
WARMUP_TOKENS = int(os.environ.get("WARMUP_TOKENS", "2000"))


def parse_cases():
    raw = os.environ.get("CASES", "c1_128:128,c1_2000:2000,c1_10000:10000")
    cases = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid CASES item, expected label:tokens: {item}")
        label, tokens = item.split(":", 1)
        cases.append((label.strip(), int(tokens.strip())))
    if not cases:
        raise ValueError("CASES did not contain any benchmark cases")
    return cases


CASES = parse_cases()


def http_get(path, timeout=10):
    with urllib.request.urlopen(API_BASE + path, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def detect_model():
    data = json.loads(http_get("/v1/models", timeout=10))
    models = data.get("data") or []
    if not models:
        raise RuntimeError("no models returned by /v1/models")
    return models[0]["id"]


def metric_value(name, text=None):
    if text is None:
        text = http_get("/metrics", timeout=10)
    pattern = re.compile(
        r"^"
        + re.escape(name)
        + r"\{[^}]*model_name=\""
        + re.escape(METRIC_MODEL)
        + r"\"[^}]*\}\s+([-+0-9.eE]+)$"
    )
    total = 0.0
    found = False
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            total += float(match.group(1))
            found = True
    if not found:
        raise RuntimeError(f"metric not found: {name}")
    return total


def metrics_snapshot():
    text = http_get("/metrics", timeout=10)
    names = [
        "vllm:generation_tokens_total",
        "vllm:request_generation_tokens_sum",
        "vllm:request_generation_tokens_count",
        "vllm:request_time_per_output_token_seconds_sum",
        "vllm:request_time_per_output_token_seconds_count",
        "vllm:e2e_request_latency_seconds_sum",
        "vllm:e2e_request_latency_seconds_count",
    ]
    return {name: metric_value(name, text) for name in names}


def post_stream(body):
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        API_BASE + "/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=7200)


def wait_idle():
    for _ in range(300):
        text = http_get("/metrics", timeout=10)
        running = metric_value("vllm:num_requests_running", text)
        waiting = metric_value("vllm:num_requests_waiting", text)
        if running == 0 and waiting == 0:
            return
        print(f"waiting for idle: running={running:.0f} waiting={waiting:.0f}", flush=True)
        time.sleep(1)


def run_case(label, target_tokens):
    wait_idle()
    before = metrics_snapshot()
    base_generation = before["vllm:generation_tokens_total"]
    stop = threading.Event()
    live_state = {
        "first_token_time": None,
        "last_tokens": 0.0,
        "last_time": None,
    }

    print("", flush=True)
    print("=" * 96, flush=True)
    print(f"START {label}: single request, target generation tokens={target_tokens}", flush=True)
    print(f"model={MODEL} endpoint={API_BASE}/v1/completions", flush=True)
    print("live TPS below is computed from vLLM generation_tokens_total while the stream is active", flush=True)
    print("=" * 96, flush=True)

    def poller():
        while not stop.is_set():
            now = time.monotonic()
            try:
                generated = metric_value("vllm:generation_tokens_total") - base_generation
                if generated > 0 and live_state["first_token_time"] is None:
                    live_state["first_token_time"] = now
                    live_state["last_time"] = now
                    live_state["last_tokens"] = generated
                if live_state["first_token_time"] is not None:
                    elapsed_decode = max(now - live_state["first_token_time"], 1e-9)
                    avg_tps = generated / elapsed_decode
                    if live_state["last_time"] is None:
                        inst_tps = 0.0
                    else:
                        dt = max(now - live_state["last_time"], 1e-9)
                        inst_tps = (generated - live_state["last_tokens"]) / dt
                    live_state["last_time"] = now
                    live_state["last_tokens"] = generated
                    print(
                        f"LIVE {label} elapsed_decode={elapsed_decode:7.2f}s "
                        f"server_tokens={generated:8.0f} avg_decode_tps={avg_tps:7.2f} "
                        f"instant_tps={inst_tps:7.2f}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"LIVE {label} metrics warning: {exc}", flush=True)
            stop.wait(1.0)

    body = {
        "model": MODEL,
        "prompt": PROMPT_TEXT * PROMPT_REPEAT,
        "temperature": 0.8,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": target_tokens,
        "min_tokens": target_tokens,
        "ignore_eos": True,
    }

    start = time.monotonic()
    first_chunk = None
    chunks = 0
    usage = None
    thread = threading.Thread(target=poller, daemon=True)
    thread.start()

    try:
        with post_stream(body) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                content = choices[0].get("text") or ""
                if content:
                    chunks += 1
                    if first_chunk is None:
                        first_chunk = time.monotonic()
                        print(f"FIRST_STREAM_CHUNK {label}: ttft={first_chunk - start:.3f}s", flush=True)
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        raise
    finally:
        stop.set()
        thread.join(timeout=3)

    end = time.monotonic()
    time.sleep(1.0)
    after = metrics_snapshot()
    metric_tokens = after["vllm:generation_tokens_total"] - before["vllm:generation_tokens_total"]
    request_tokens = after["vllm:request_generation_tokens_sum"] - before["vllm:request_generation_tokens_sum"]
    request_count = after["vllm:request_generation_tokens_count"] - before["vllm:request_generation_tokens_count"]
    tpot_sum = (
        after["vllm:request_time_per_output_token_seconds_sum"]
        - before["vllm:request_time_per_output_token_seconds_sum"]
    )
    tpot_count = (
        after["vllm:request_time_per_output_token_seconds_count"]
        - before["vllm:request_time_per_output_token_seconds_count"]
    )
    e2e_sum = after["vllm:e2e_request_latency_seconds_sum"] - before["vllm:e2e_request_latency_seconds_sum"]
    wall = end - start
    client_wall_tps = metric_tokens / wall if wall > 0 else 0.0
    decode_tps = (1.0 / (tpot_sum / tpot_count)) if tpot_count > 0 and tpot_sum > 0 else 0.0
    e2e_wall_tps = (request_tokens / e2e_sum) if e2e_sum > 0 else 0.0

    print("-" * 96, flush=True)
    print(
        f"RESULT {label}: metric_tokens={metric_tokens:.0f} request_tokens={request_tokens:.0f} "
        f"requests={request_count:.0f} chunks={chunks}",
        flush=True,
    )
    print(
        f"RESULT {label}: decode_tps_from_vllm_tpot={decode_tps:.2f} "
        f"client_wall_tps={client_wall_tps:.2f} e2e_metric_tps={e2e_wall_tps:.2f} wall={wall:.3f}s",
        flush=True,
    )
    if usage:
        print(f"RESULT {label}: stream_usage={json.dumps(usage, sort_keys=True)}", flush=True)
    print("-" * 96, flush=True)


def main():
    global MODEL, METRIC_MODEL
    if not MODEL:
        MODEL = detect_model()
    if not METRIC_MODEL:
        METRIC_MODEL = MODEL

    print("Qwen3.6 single-request TPS harness", flush=True)
    print(f"model={MODEL} metric_model={METRIC_MODEL} api_base={API_BASE}", flush=True)
    print(f"cases={CASES}", flush=True)
    print(f"warmup_requests={WARMUP_REQUESTS} warmup_tokens={WARMUP_TOKENS} prompt_repeat={PROMPT_REPEAT}", flush=True)
    print("This is a fixed-token TPS harness: max_tokens=min_tokens and ignore_eos=true.", flush=True)
    for idx in range(1, WARMUP_REQUESTS + 1):
        run_case(f"warmup_{idx}", WARMUP_TOKENS)
        time.sleep(2)
    for label, target_tokens in CASES:
        run_case(label, target_tokens)
        time.sleep(5)
    print("ALL CASES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
