# Observability

## What

Structured logging, metrics sinks, lifecycle events, counters and timing spans.

## Why

Training, evaluation, rollout, serving and promotion decisions must be traceable locally without requiring external telemetry.

## How

* **Logging** — `get_logger("forgeline.x").info("event", key=value)`; text (`event key=value`) or JSON lines with `FORGELINE_LOG_FORMAT=json`; level via `FORGELINE_LOG_LEVEL`.
* **Metrics sinks** — `build_metrics_sink(backend)`:
  * `local` — `<output_dir>/metrics.jsonl` and `events.jsonl` (non-finite values written as `null`);
  * `none` — in-memory only;
  * `wandb` / `tensorboard` — optional `observability` extra.
  Every sink keeps history (`latest`, `series`), events and counters (`increment`).
* **Events** (`observability/events.py`) — `run.started|finished|failed`, `checkpoint.saved|loaded|failed`, `evaluation.started|finished`, `rollout.batch|failed`, `candidate.registered|promoted|rejected|rolled_back`, `serving.request|error`.
* **Tracing** — `Tracer.span(name)` records durations; `summary()` gives count/mean/max per span (the inference engine records `prefill` and `decode`).
* **Training metrics** — loss, learning rate, gradient norm and algorithm metrics (reward mean/std, KL, entropy, groups skipped, preference accuracy, response length, tool calls…), CUDA memory when available, `gradient_norms(model)` per selected layer.
* **Serving metrics** — request and error counters, latency, completion tokens, `/metrics` endpoint.

## Dashboard

`forgeline dashboard` serves a read-only local UI over run metrics and events, checkpoints, evaluation files, the registry, benchmark evidence and model inspection; `--export` writes the same data as JSON. See [dashboard.md](dashboard.md).

## Configuration

`forgeline train … --metrics local|none|wandb|tensorboard`; `forgeline serve … --metrics local --output-dir runs/serve`.

## Failure modes

Requesting `wandb`/`tensorboard` without the extra raises `OptionalDependencyError`; unknown backends raise `ConfigError`.

## Local validation

`tests/unit/test_observability_cli.py` (JSONL sink contents, events, counters, structured log output, spans, gradient norms); integration tests assert `run.started`, `checkpoint.saved`, `run.finished` and `run.failed` events.

## Hardware requirements

None.

## Limitations

No OpenTelemetry exporter; spans are in-process. The dashboard reads local files only (no W&B / TensorBoard ingestion).
