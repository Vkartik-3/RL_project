# Serving

## What

An OpenAI-compatible HTTP API over one or more backends, with health, model listing, completions, chat completions, server-sent-event streaming and request metrics.

## Why

Local, dependency-free serving of trained checkpoints, with pluggable routing so champion/challenger rollouts apply to live traffic.

## How

| Endpoint | Method | Response |
|---|---|---|
| `/health` | GET | `status` (`ok` / `degraded`), per-backend health, default model |
| `/v1/models` | GET | OpenAI model list |
| `/v1/completions` | POST | `text_completion` with usage; SSE when `stream: true` |
| `/v1/chat/completions` | POST | `chat.completion`; messages rendered as `<|role|>\ncontent\n…<|assistant|>\n`; SSE chunks when streaming |
| `/metrics` | GET | request/error counts, latency p50/p95 |

* `Router(backends, default_model, route_fn=None, metrics=None)` — transport-agnostic: `handle(method, path, body)` returns `(status, json)` or `(status, iterator of SSE lines)`. `route_fn(request_id, requested_model)` selects the backend (e.g. `RoutingPolicy.route(...).primary`).
* `EngineBackend(name, engine, tokenizer, background=True)` — runs the inference engine loop in a thread; streams text as tokens decode cleanly.
* `EchoBackend` — deterministic backend for tests.
* `ServingServer(router, host, port)` — `ThreadingHTTPServer`; `port=0` picks a free port.
* Responses include `forgeline.backend` so clients and logs record which candidate served a request.

## Configuration

`forgeline serve --checkpoint DIR [--host 127.0.0.1] [--port 8000] [--model-name NAME] [--max-batch 8] [--max-seq-len N] [--routing routing.json] [--metrics local]`

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt": "Q: What is 2 + 2?\nA:", "max_tokens": 8, "temperature": 0}'
```

## Failure modes

| Condition | Status |
|---|---|
| invalid JSON / schema (empty prompt, bad role, `max_tokens < 1`, `top_p` out of range, bad `stop`) | 400 |
| unknown model | 400 |
| disabled / unhealthy backend | 503 |
| backend exception | 500 with the error message |
| unknown route | 404 |

## Local validation

`tests/unit/test_inference_serving.py` (schemas, router over echo backends including streaming, 400/404/503, metrics; engine backend completion and streaming consistency), `tests/integration/test_cli_pipeline.py::test_http_server_roundtrip` (real sockets: health, completion usage, streamed chat ending in `[DONE]`).

## Hardware requirements

CPU or a single GPU per backend process.

## Limitations

* No authentication, TLS or rate limiting; bind to localhost or place behind a gateway.
* Shadow traffic is decided by the routing policy but the router serves only the primary backend; use `offline_replay` for shadow comparisons.
