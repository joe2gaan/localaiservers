#!/usr/bin/env python3
"""Small single-backend /v1/chat/completions proxy used by v0.2 benchmarks.

The v0.2 release benchmark path used a begin-think proxy. For the public
single-backend reproduction path we only need GET passthrough for /v1/models
and POST passthrough for /v1/chat/completions, plus the same non-streaming
response rewrite that prefixes message.content with "<think>".
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def rewrite_begin_think(payload: bytes) -> bytes:
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError:
        return payload

    changed = False
    choices = parsed.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and not content.startswith("<think>"):
                message["content"] = "<think>" + content
                changed = True

    if not changed:
        return payload
    return json.dumps(parsed, separators=(",", ":")).encode("utf-8")


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "LocalAIServersBeginThinkProxy/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        if self.server.quiet:  # type: ignore[attr-defined]
            return
        super().log_message(fmt, *args)

    def do_GET(self) -> None:  # noqa: N802
        self.forward(None)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b""
        self.forward(body)

    def forward(self, body: bytes | None) -> None:
        backend = self.server.backend.rstrip("/")  # type: ignore[attr-defined]
        target = backend + self.path
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "connection"}
        }
        if body is not None:
            headers.setdefault("Content-Type", "application/json")

        request = Request(
            target,
            data=body,
            headers=headers,
            method=self.command,
        )
        try:
            with urlopen(request, timeout=self.server.timeout_seconds) as response:  # type: ignore[attr-defined]
                payload = response.read()
                status = response.status
                response_headers = response.headers
        except HTTPError as exc:
            payload = exc.read()
            status = exc.code
            response_headers = exc.headers

        if (
            self.command == "POST"
            and self.path.startswith("/v1/chat/completions")
            and not self.server.disable_begin_think  # type: ignore[attr-defined]
            and b'"stream":true' not in (body or b"")
        ):
            payload = rewrite_begin_think(payload)

        self.send_response(status)
        content_type = response_headers.get("Content-Type", "application/json")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-backend begin-think proxy")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--backend", required=True, help="Backend base URL, e.g. http://127.0.0.1:8001")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--disable-begin-think", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.listen_host, args.listen_port), ProxyHandler)
    server.backend = args.backend  # type: ignore[attr-defined]
    server.timeout_seconds = args.timeout  # type: ignore[attr-defined]
    server.disable_begin_think = args.disable_begin_think  # type: ignore[attr-defined]
    server.quiet = args.quiet  # type: ignore[attr-defined]
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
