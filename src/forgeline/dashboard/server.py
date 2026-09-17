"""Local, read-only dashboard server (standard library only).

Endpoints (all GET, JSON unless noted):

* ``/`` — single-page UI (HTML, no external assets)
* ``/api/overview`` — runs, registry and benchmark summaries
* ``/api/runs`` · ``/api/runs/<id>`` — run list · metric series, events, checkpoints, evaluation files of one run
* ``/api/registry`` — candidates grouped by model name with states, metrics and transition history
* ``/api/benchmarks`` — evidence records from ``benchmarks/**/results.json``
* ``/api/inspect?checkpoint=<path>&text=<prompt>`` — architecture, weight statistics, attention patterns, activation flow
  (only for checkpoints under a configured runs root)
* ``/health``

The server never mutates state; lifecycle changes go through ``forgeline registry``.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from forgeline.dashboard import data
from forgeline.dashboard.data import DashboardSources


class DashboardApp:
    """Routes requests to the data layer; usable without sockets (``handle``)."""

    def __init__(self, sources: DashboardSources, inspect_cache_size: int = 4):
        self.sources = sources
        self._inspect_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._cache_size = inspect_cache_size
        self._lock = threading.Lock()

    def _run_path(self, run_id: str) -> Optional[str]:
        return next((r["path"] for r in data.discover_runs(self.sources.runs) if r["id"] == run_id), None)

    def handle(self, path: str) -> Tuple[int, str, Any]:
        """Return ``(status, content_type, payload)``."""
        url = urlparse(path)
        route, query = url.path.rstrip("/") or "/", parse_qs(url.query)
        if route in ("/", "/index.html"):
            return 200, "text/html; charset=utf-8", index_html()
        if route == "/health":
            return 200, "application/json", {"status": "ok"}
        if route == "/api/overview":
            runs = data.discover_runs(self.sources.runs)
            registry = data.registry_view(self.sources.registry)
            benches = data.benchmark_records(self.sources.benchmarks)
            return 200, "application/json", {
                "sources": {"runs": self.sources.runs, "registry": self.sources.registry, "benchmarks": self.sources.benchmarks},
                "runs": runs, "registry": registry,
                "benchmarks": [{k: b[k] for k in ("record", "evidence", "revalidation_recommended", "published")} for b in benches],
            }
        if route == "/api/runs":
            return 200, "application/json", data.discover_runs(self.sources.runs)
        if route.startswith("/api/runs/"):
            run_path = self._run_path(route.rsplit("/", 1)[-1])
            if run_path is None:
                return 404, "application/json", {"error": "unknown run"}
            return 200, "application/json", data.run_detail(run_path)
        if route == "/api/registry":
            return 200, "application/json", data.registry_view(self.sources.registry)
        if route == "/api/benchmarks":
            return 200, "application/json", data.benchmark_records(self.sources.benchmarks)
        if route == "/api/inspect":
            ckpt = (query.get("checkpoint") or [""])[0]
            text = (query.get("text") or [""])[0][:2000]
            if not ckpt or not self.sources.checkpoint_allowed(ckpt):
                return 403, "application/json", {"error": "checkpoint must be a checkpoint directory under a configured runs root"}
            key = (str(Path(ckpt).resolve()), text)
            with self._lock:
                cached = self._inspect_cache.get(key)
            if cached is None:
                try:
                    cached = data.inspect_checkpoint(ckpt, text=text)
                except Exception as exc:  # noqa: BLE001
                    return 422, "application/json", {"error": f"{type(exc).__name__}: {exc}"}
                with self._lock:
                    if len(self._inspect_cache) >= self._cache_size:
                        self._inspect_cache.pop(next(iter(self._inspect_cache)))
                    self._inspect_cache[key] = cached
            return 200, "application/json", cached
        return 404, "application/json", {"error": f"no route {route}"}


def index_html() -> str:
    return resources.files("forgeline.dashboard").joinpath("static/index.html").read_text()


def make_handler(app: DashboardApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "forgeline-dashboard/0.1"

        def do_GET(self) -> None:  # noqa: N802
            status, ctype, payload = app.handle(self.path)
            body = payload.encode() if isinstance(payload, str) else json.dumps(payload, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # quiet
            pass

    return Handler


class DashboardServer:
    def __init__(self, app: DashboardApp, host: str = "127.0.0.1", port: int = 8765):
        self.httpd = ThreadingHTTPServer((host, port), make_handler(app))
        self.host, self.port = host, self.httpd.server_address[1]
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> "DashboardServer":
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
