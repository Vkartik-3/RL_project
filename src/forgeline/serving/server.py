"""Standard-library HTTP server exposing the router (no web framework required)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator, Optional

from forgeline.serving.api import Router


def make_handler(router: Router):
    class Handler(BaseHTTPRequestHandler):
        server_version = "forgeline/0.1"

        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_sse(self, chunks: Iterator[str]) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(chunk.encode())
                self.wfile.flush()

        def _dispatch(self, method: str) -> None:
            body = None
            if method == "POST":
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    self._send_json(400, {"error": {"message": "invalid JSON body", "type": "invalid_request_error", "code": 400}})
                    return
            status, payload = router.handle(method, self.path.split("?")[0], body)
            if isinstance(payload, dict):
                self._send_json(status, payload)
            else:
                self._send_sse(payload)

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def log_message(self, format: str, *args: Any) -> None:  # quiet
            pass

    return Handler


class ServingServer:
    def __init__(self, router: Router, host: str = "127.0.0.1", port: int = 8000):
        self.router = router
        self.httpd = ThreadingHTTPServer((host, port), make_handler(router))
        self.httpd.daemon_threads = True
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    def start(self) -> None:
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
