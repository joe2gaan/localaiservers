#!/usr/bin/env python3
import argparse
from collections import Counter
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


METRIC_NAMES = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_prefill_time_seconds_count",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_decode_time_seconds_count",
)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def http_get_text(url: str, timeout: float) -> str:
    request = urllib.request.Request(
        url,
        headers={"Connection": "close"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def http_post_json(url: str, payload: dict, timeout: float) -> tuple[dict, float]:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if os.environ.get("CHAT_CAPTURE_USE_CURL") == "1":
        start = time.monotonic()
        result = subprocess.run(
            [
                "curl",
                "-fsS",
                "--http1.1",
                "--max-time",
                str(timeout),
                "-H",
                "Content-Type: application/json",
                "-H",
                "Connection: close",
                "--data-binary",
                "@-",
                url,
            ],
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        elapsed = time.monotonic() - start
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"curl POST failed with rc={result.returncode}: {message}")
        return json.loads(result.stdout.decode("utf-8")), elapsed
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Connection": "close"},
        method="POST",
    )
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body.decode("utf-8")), time.monotonic() - start


def wait_ready(endpoint: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    models_url = endpoint.rstrip("/") + "/models"
    last_error = ""
    while time.monotonic() < deadline:
        try:
            http_get_text(models_url, timeout=5)
            return
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
            time.sleep(5)
    raise RuntimeError(f"backend not ready at {models_url}: {last_error}")


def parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        for name in METRIC_NAMES:
            if line.startswith(name):
                try:
                    value = float(line.rsplit(" ", 1)[1])
                except (IndexError, ValueError):
                    continue
                metrics[name] = metrics.get(name, 0.0) + value
                break
    return metrics


def metric_delta(after: dict[str, float], before: dict[str, float], name: str) -> float:
    return after.get(name, 0.0) - before.get(name, 0.0)


def read_metrics(metrics_url: str) -> dict[str, float]:
    return parse_metrics(http_get_text(metrics_url, timeout=10))


def wait_metrics_settled(
    metrics_url: str,
    before: dict[str, float],
    completion_tokens: int,
    timeout_seconds: float,
    interval_seconds: float,
) -> tuple[dict[str, float], bool, float]:
    interval_seconds = max(interval_seconds, 0.1)
    deadline = time.monotonic() + timeout_seconds
    start = time.monotonic()
    after = read_metrics(metrics_url)
    while True:
        generation_delta = metric_delta(
            after, before, "vllm:generation_tokens_total"
        )
        decode_count_delta = metric_delta(
            after, before, "vllm:request_decode_time_seconds_count"
        )
        has_generation = completion_tokens <= 0 or generation_delta >= completion_tokens
        has_decode_count = decode_count_delta >= 1
        if has_generation and has_decode_count:
            return after, True, time.monotonic() - start
        if time.monotonic() >= deadline:
            return after, False, time.monotonic() - start
        time.sleep(interval_seconds)


def extract_response_parts(parsed: dict) -> tuple[str, str]:
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return "", ""
    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str):
        reasoning = message.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = ""
    content = message.get("content")
    if not isinstance(content, str):
        content = ""
    return reasoning, content


def extract_response_text(parsed: dict) -> str:
    reasoning, content = extract_response_parts(parsed)
    return "\n".join(part for part in (reasoning, content) if part)


def visible_after_qwen_thinking(reasoning: str, content: str) -> tuple[str, dict[str, bool]]:
    raw_text = "\n".join(part for part in (reasoning, content) if part)
    flags = {
        "has_reasoning_field": bool(reasoning.strip()),
        "content_has_think_close": "</think>" in content,
        "raw_has_think_close": "</think>" in raw_text,
        "qwen_visible_parse_complete": False,
    }
    if "</think>" in content:
        flags["qwen_visible_parse_complete"] = True
        return content.split("</think>", 1)[1].lstrip(), flags
    if reasoning.strip():
        flags["qwen_visible_parse_complete"] = True
        return content.lstrip(), flags
    if "</think>" in raw_text:
        flags["qwen_visible_parse_complete"] = True
        return raw_text.split("</think>", 1)[1].lstrip(), flags
    return raw_text, flags


def degenerate_output_reasons(
    reasoning: str,
    content: str,
    answer: str,
    parser_split_answer: bool,
) -> list[str]:
    reasons: list[str] = []

    compact_reasoning = re.sub(r"\s+", "", reasoning)
    if len(compact_reasoning) >= 512:
        char, count = Counter(compact_reasoning).most_common(1)[0]
        dominant_ratio = count / len(compact_reasoning)
        if dominant_ratio >= 0.85 and not char.isalnum():
            reasons.append(
                f"degenerate_repeated_reasoning_char_{repr(char)}_{dominant_ratio:.3f}"
            )

    if parser_split_answer and len(reasoning.strip()) >= 512 and len(answer.strip()) < 32:
        reasons.append("too_short_answer_after_long_reasoning")

    return reasons


def strict_qwen_answer(
    reasoning: str,
    content: str,
    max_tokens: int | None,
    finish_reason: str | None,
) -> tuple[str, dict[str, object]]:
    raw_text = "\n".join(part for part in (reasoning, content) if part)
    close_in_content = "</think>" in content
    close_in_raw = "</think>" in raw_text
    parser_split_answer = bool(reasoning.strip()) and bool(content.strip())

    answer = ""
    if close_in_content:
        answer = content.split("</think>", 1)[1].lstrip()
    elif close_in_raw:
        answer = raw_text.split("</think>", 1)[1].lstrip()
    elif parser_split_answer:
        # vLLM's reasoning parser consumes the delimiter and returns answer
        # text separately only after it has observed the model's close marker.
        answer = content.lstrip()

    invalid_reasons: list[str] = []
    if max_tokens is not None:
        invalid_reasons.append("client_max_tokens_cap")
    if finish_reason == "length":
        invalid_reasons.append("finish_reason_length")
    if max_tokens is None and finish_reason not in ("stop",):
        invalid_reasons.append(f"finish_reason_not_stop_{finish_reason}")
    if not (close_in_content or close_in_raw or parser_split_answer):
        invalid_reasons.append("missing_think_close_or_parser_answer")
    invalid_reasons.extend(
        degenerate_output_reasons(reasoning, content, answer, parser_split_answer)
    )

    return answer, {
        "qwen_gate_valid": not invalid_reasons,
        "qwen_gate_invalid_reasons": invalid_reasons,
        "qwen_gate_close_in_content": close_in_content,
        "qwen_gate_close_in_raw": close_in_raw,
        "qwen_gate_parser_split_answer": parser_split_answer,
    }


def response_completion_tokens(parsed: dict) -> int:
    usage = parsed.get("usage")
    if isinstance(usage, dict):
        for key in ("completion_tokens", "output_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                return value
    return 0


def build_payload(
    model: str,
    max_tokens: int | None,
    prompt_repeat: int,
    force_length: bool,
) -> dict:
    seed = (
        "You are participating in a synthetic decode-throughput benchmark. "
        "Continue with dense, factual numbered statements about precision instruments, "
        "calibration manuals, warranties, and source-grounded QA. Do not summarize. "
    )
    prompt = seed + (
        "calibration manual warranty infrared thermometer digital multimeter slide caliper. "
        * prompt_repeat
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
        if force_length:
            payload["min_tokens"] = max_tokens
            payload["ignore_eos"] = True
    return payload


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture one deterministic chat response.")
    parser.add_argument("--endpoint", required=True, help="Base endpoint ending in /v1")
    parser.add_argument("--metrics-url", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--no-max-tokens",
        action="store_true",
        help="Do not send max_tokens/min_tokens/ignore_eos; use for EOS-terminated quality runs.",
    )
    parser.add_argument(
        "--allow-eos",
        action="store_true",
        help="When max_tokens is set, do not force min_tokens=max_tokens or ignore_eos.",
    )
    parser.add_argument("--prompt-repeat", type=int, default=32)
    parser.add_argument("--ready-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument(
        "--metrics-settle-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait after the response for vLLM /metrics counters to catch up.",
    )
    parser.add_argument(
        "--metrics-settle-interval",
        type=float,
        default=1.0,
        help="Polling interval while waiting for post-response metrics to settle.",
    )
    parser.add_argument(
        "--require-think-close",
        action="store_true",
        help="Require an uncapped complete Qwen thinking response and hash only post-think answer text.",
    )
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    chat_url = endpoint + "/chat/completions"
    metrics_url = args.metrics_url or (re.sub(r"/v1/?$", "", endpoint) + "/metrics")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_tokens = None if args.no_max_tokens else args.max_tokens
    force_length = not args.allow_eos and max_tokens is not None
    payload = build_payload(args.model, max_tokens, args.prompt_repeat, force_length)
    write_json(out_dir / "request.json", payload)
    wait_ready(endpoint, args.ready_timeout)

    before = read_metrics(metrics_url)
    parsed, elapsed = http_post_json(chat_url, payload, args.request_timeout)

    reasoning, content = extract_response_parts(parsed)
    finish_reason = (parsed.get("choices") or [{}])[0].get("finish_reason")
    text = "\n".join(part for part in (reasoning, content) if part)
    visible_text, qwen_flags = visible_after_qwen_thinking(reasoning, content)
    answer_text, qwen_gate_flags = strict_qwen_answer(
        reasoning, content, max_tokens, finish_reason
    )
    completion_tokens = response_completion_tokens(parsed)
    after, metrics_settled, metrics_settle_elapsed = wait_metrics_settled(
        metrics_url,
        before,
        completion_tokens,
        args.metrics_settle_timeout,
        args.metrics_settle_interval,
    )
    summary = {
        "finished_at_utc": now_iso(),
        "endpoint": endpoint,
        "metrics_url": metrics_url,
        "model": args.model,
        "max_tokens": max_tokens,
        "force_length": force_length,
        "prompt_repeat": args.prompt_repeat,
        "elapsed_seconds": elapsed,
        "completion_tokens": completion_tokens,
        "metrics_settled": metrics_settled,
        "metrics_settle_elapsed_seconds": metrics_settle_elapsed,
        "metrics_settle_timeout_seconds": args.metrics_settle_timeout,
        "wall_completion_tps": completion_tokens / elapsed if elapsed > 0 else 0.0,
        "finish_reason": finish_reason,
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "text_chars": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "qwen_visible_text_chars": len(visible_text),
        "qwen_visible_text_sha256": hashlib.sha256(
            visible_text.encode("utf-8")
        ).hexdigest(),
        "qwen_answer_text_chars": len(answer_text),
        "qwen_answer_text_sha256": hashlib.sha256(
            answer_text.encode("utf-8")
        ).hexdigest(),
        **qwen_flags,
        **qwen_gate_flags,
        "vllm_generation_tokens_delta": metric_delta(
            after, before, "vllm:generation_tokens_total"
        ),
        "vllm_prompt_tokens_delta": metric_delta(
            after, before, "vllm:prompt_tokens_total"
        ),
        "vllm_prefill_seconds_delta": metric_delta(
            after, before, "vllm:request_prefill_time_seconds_sum"
        ),
        "vllm_prefill_request_count_delta": metric_delta(
            after, before, "vllm:request_prefill_time_seconds_count"
        ),
        "vllm_decode_seconds_delta": metric_delta(
            after, before, "vllm:request_decode_time_seconds_sum"
        ),
        "vllm_decode_request_count_delta": metric_delta(
            after, before, "vllm:request_decode_time_seconds_count"
        ),
    }
    decode_seconds = summary["vllm_decode_seconds_delta"]
    if decode_seconds > 0:
        summary["vllm_decode_tps"] = (
            summary["vllm_generation_tokens_delta"] / decode_seconds
        )
    else:
        summary["vllm_decode_tps"] = 0.0
    prefill_seconds = summary["vllm_prefill_seconds_delta"]
    if prefill_seconds > 0:
        summary["vllm_prefill_tps"] = (
            summary["vllm_prompt_tokens_delta"] / prefill_seconds
        )
    else:
        summary["vllm_prefill_tps"] = 0.0

    write_json(out_dir / "response.json", parsed)
    (out_dir / "response_text.txt").write_text(text, encoding="utf-8")
    (out_dir / "qwen_visible_text.txt").write_text(visible_text, encoding="utf-8")
    (out_dir / "qwen_answer_text.txt").write_text(answer_text, encoding="utf-8")
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, separators=(",", ":")), flush=True)
    if args.require_think_close and not summary["qwen_gate_valid"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
