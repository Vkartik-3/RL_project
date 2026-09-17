"""Transport-agnostic router for the OpenAI-compatible API.

``Router.handle(method, path, body)`` returns ``(status, payload)`` for
JSON responses or ``(status, iterator)`` for server-sent-event streams, so
the same object is exercised by unit tests and by the HTTP server.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

from forgeline.core.errors import BackendUnavailableError, ServingError
from forgeline.observability.events import Event
from forgeline.observability.logging import get_logger
from forgeline.observability.metrics import MetricsSink, NoOpMetricsSink
from forgeline.serving.backends import Backend
from forgeline.serving.schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    chat_chunk,
    chat_response,
    completion_chunk,
    completion_response,
    error_response,
    new_request_id,
)

log = get_logger("forgeline.serving")
Response = Tuple[int, Union[Dict[str, Any], Iterator[str]]]


class Router:
    def __init__(self, backends: Dict[str, Backend], default_model: str, route_fn: Optional[Callable[[str, Optional[str]], str]] = None,
                 metrics: Optional[MetricsSink] = None, owner: str = "forgeline"):
        if default_model not in backends:
            raise ServingError(f"default model {default_model!r} is not a registered backend")
        self.backends = dict(backends)
        self.default_model = default_model
        self.route_fn = route_fn
        self.metrics = metrics or NoOpMetricsSink()
        self.owner = owner
        self.requests_total = 0
        self.errors_total = 0
        self.latencies: list[float] = []

    # ── dispatch ─────────────────────────────────────────────────────────
    def handle(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Response:
        try:
            if method == "GET" and path == "/health":
                return 200, self.health()
            if method == "GET" and path == "/v1/models":
                return 200, self.models()
            if method == "GET" and path == "/metrics":
                return 200, self.metrics_summary()
            if method == "POST" and path == "/v1/completions":
                return self.completions(body or {})
            if method == "POST" and path == "/v1/chat/completions":
                return self.chat_completions(body or {})
            return 404, error_response(f"no route for {method} {path}", "not_found", 404)
        except BackendUnavailableError as exc:
            self._error(exc)
            return 503, error_response(exc.message, "backend_unavailable", 503)
        except ServingError as exc:
            self._error(exc)
            return 400, error_response(exc.message, "invalid_request_error", 400)
        except Exception as exc:  # noqa: BLE001
            self._error(exc)
            return 500, error_response(f"{type(exc).__name__}: {exc}", "server_error", 500)

    def _error(self, exc: BaseException) -> None:
        self.errors_total += 1
        self.metrics.increment("serving/errors")
        self.metrics.event(Event.SERVING_ERROR.value, {"error": repr(exc)})

    # ── endpoints ────────────────────────────────────────────────────────
    def health(self) -> Dict[str, Any]:
        status = {name: b.healthy() for name, b in self.backends.items()}
        return {"status": "ok" if status.get(self.default_model, False) else "degraded", "backends": status,
                "default_model": self.default_model}

    def models(self) -> Dict[str, Any]:
        return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": self.owner, "healthy": b.healthy()}
                                            for name, b in self.backends.items()]}

    def metrics_summary(self) -> Dict[str, Any]:
        lat = sorted(self.latencies[-1000:])
        p50 = lat[len(lat) // 2] if lat else 0.0
        p95 = lat[int(len(lat) * 0.95)] if lat else 0.0
        return {"requests_total": self.requests_total, "errors_total": self.errors_total,
                "latency_p50_s": p50, "latency_p95_s": p95, "backends": list(self.backends)}

    def _select_backend(self, request_id: str, requested: Optional[str]) -> Tuple[str, Backend]:
        name = self.route_fn(request_id, requested) if self.route_fn else (requested or self.default_model)
        if name not in self.backends:
            raise ServingError(f"unknown model {name!r}; available: {sorted(self.backends)}")
        backend = self.backends[name]
        if not backend.healthy():
            raise BackendUnavailableError(f"backend {name!r} is unavailable")
        return name, backend

    def completions(self, body: Dict[str, Any]) -> Response:
        req = CompletionRequest.from_dict(body)
        rid = req.request_id or new_request_id()
        name, backend = self._select_backend(rid, req.model)
        self.requests_total += 1
        t0 = time.perf_counter()
        if req.stream:
            return 200, self._sse(backend.stream(req.prompt, req.sampling(), rid), lambda t, fr: completion_chunk(rid, name, t, fr), t0)
        result = backend.complete(req.prompt, req.sampling(), rid)
        self._record(t0, result.completion_tokens, name)
        return 200, completion_response(rid, name, result.text, result.prompt_tokens, result.completion_tokens, result.finish_reason, name)

    def chat_completions(self, body: Dict[str, Any]) -> Response:
        req = ChatCompletionRequest.from_dict(body)
        rid = req.request_id or new_request_id()
        name, backend = self._select_backend(rid, req.model)
        self.requests_total += 1
        t0 = time.perf_counter()
        prompt = req.to_prompt()
        if req.stream:
            return 200, self._sse(backend.stream(prompt, req.sampling(), rid), lambda t, fr: chat_chunk(rid, name, t, fr), t0)
        result = backend.complete(prompt, req.sampling(), rid)
        self._record(t0, result.completion_tokens, name)
        return 200, chat_response(rid, name, result.text, result.prompt_tokens, result.completion_tokens, result.finish_reason, name)

    def _sse(self, chunks: Iterator[str], make_chunk: Callable[[str, Optional[str]], Dict[str, Any]], t0: float) -> Iterator[str]:
        n = 0
        for text in chunks:
            n += 1
            yield f"data: {json.dumps(make_chunk(text, None))}\n\n"
        yield f"data: {json.dumps(make_chunk('', 'stop'))}\n\n"
        yield "data: [DONE]\n\n"
        self._record(t0, n, "stream")

    def _record(self, t0: float, tokens: int, backend: str) -> None:
        latency = time.perf_counter() - t0
        self.latencies.append(latency)
        self.metrics.log({"serving/latency_s": latency, "serving/completion_tokens": float(tokens)})
        self.metrics.increment("serving/requests")
        self.metrics.event(Event.SERVING_REQUEST.value, {"backend": backend, "latency_s": latency, "tokens": tokens})
