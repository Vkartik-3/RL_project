# Dashboard

An optional, read-only, local web UI over the files Forgeline already writes. Training, evaluation, serving and the registry never depend on it, and it adds no dependencies.

```bash
forgeline dashboard --runs runs --registry registry/candidates.json --benchmarks benchmarks --port 8765
forgeline dashboard --runs runs other_runs --export runs/dashboard.json     # JSON snapshot, no server
```

## Views

| View | Source | Shows |
|---|---|---|
| Runs | `metrics.jsonl`, `events.jsonl`, `manifest.yaml` under each `--runs` root | status (running / finished / failed from `run.*` events, "live" when updated in the last 2 min), algorithm, strategy, Ray orchestration, last step; one chart per logged metric; final summary; latest 60 events |
| Checkpoints (per run) | `manifest.json` + files | step, algorithm, size, `LATEST` marker, validity from `CheckpointManager.validate` (missing, truncated, incomplete files are flagged with the error) |
| Evaluations (per run) | any `save_results` JSON/JSONL in the run directory | suite metrics, sample counts, offline-sample flag |
| Registry | the registry JSON | candidates per model name, state, gate status, metrics, full transition history with reasons |
| Benchmarks | `benchmarks/**/results.json` + `summary.md` | evidence level, rerun-recommended and not-published flags, summaries |
| Inspect | a checkpoint under a runs root + a prompt | parameters, spec, per-parameter mean/std/norm, head-averaged attention heatmaps and per-head entropies for every layer, residual-stream norm per layer |

Metric series are downsampled to at most 1,500 points per key. The Runs view refreshes every 5 s while a run is live.

## API

| Route | Returns |
|---|---|
| `/` | the single-page UI (inline HTML/CSS/JS, no external assets) |
| `/health` | `{"status": "ok"}` |
| `/api/overview` | runs, registry, benchmark summaries |
| `/api/runs`, `/api/runs/<id>` | run list; series, events, summary, checkpoints, evaluations |
| `/api/registry` | registry view |
| `/api/benchmarks` | evidence records |
| `/api/inspect?checkpoint=<dir>&text=<prompt>` | model inspection (cached, 4 entries) |

Run ids are the first 10 hex digits of `sha1(resolved run path)`.

## Attention maps

`forgeline.evaluation.inspection.attention_patterns(model, ids, per_head=False)` captures each attention layer's input with a forward pre-hook and recomputes the attention weights from that layer's own `q/k` projections, rotary embedding, logit cap and causal or sliding-window mask. It therefore matches the forward pass even where the model uses fused SDPA (which never materialises weights). `tests/integration/test_dashboard.py` rebuilds every layer's output from the returned per-head weights and checks it equals the real output (`atol=1e-5`) for full and sliding-window/soft-capped variants. Latent (MLA) and native-sparse (NSA) layers are reported as unsupported.

## Safety

* Binds to `127.0.0.1` by default; no authentication or TLS.
* No route mutates anything.
* `torch.load` can execute pickled code, so `/api/inspect` only loads a directory that resolves inside a configured `--runs` root and contains `manifest.json`; anything else returns 403.
