# Evaluation

## What

Evaluators that return `EvaluationResult(suite, metrics, n_samples, details, per_item)`, serialisable to JSON/JSONL, plus regression rules and model inspection.

## Why

Promotion, regression gating and reporting all consume the same flat metric names (`suite/metric`), so every evaluator must emit structured results.

## How

| Suite | Module | Metrics |
|---|---|---|
| `HeldOutLossSuite` | `evaluation/quality.py` | `loss`, `perplexity` |
| `VerifierPassRateSuite` | `evaluation/quality.py` | `pass_rate`, `reward`, `malformed_rate` (+ per-item outputs) |
| `AgentAccuracySuite` | `evaluation/quality.py` | `accuracy`, `tool_use_rate` |
| `RecordRewardSuite` | `evaluation/reward.py` | `mean`, `std`, `min`, `max`, `pct_above_*`, `improvement_vs_baseline_pct`; `details.policy_dependent = false` |
| `TrajectoryRewardSuite` | `evaluation/reward.py` | same statistics over policy trajectories; `policy_dependent = true` |
| `PreferenceWinRateSuite` | `evaluation/reward.py` | `win_rate`, `mean_margin` |
| `PerformanceSuite` / `measure_generation` | `evaluation/performance.py` | `tokens_per_second`, `latency_mean_s`, `latency_p50_s`, `latency_max_s`, `peak_memory_gb` (CUDA) |
| `BenchmarkSuite` + benchmark | `evaluation/suites` | `accuracy`, `correct` |

### Standard benchmarks

`mmlu`, `hellaswag`, `arc` (challenge), `gsm8k`, `truthfulqa`, `humaneval`. Each supports few-shot formatting and task limits. With `offline=False` data is loaded from the HuggingFace hub (`huggingface` extra); otherwise a small built-in sample set is used for wiring checks — sample accuracies are not benchmark scores. HumanEval executes generated code in a subprocess.

### Regression rules

`evaluate_regression(candidate, baseline, [RegressionRule(metric, direction, max_regression, relative)])` — higher- or lower-is-better, absolute or relative tolerance; missing or non-finite metrics fail.

### Safety and quality gates

Safety regressions, malformed-output rate, KL drift and latency are expressed as metrics and enforced by promotion gates ([deployment](deployment.md)).

### Inspection

`weight_statistics` (per-parameter moments + histograms), `layer_summary`, `activation_flow` (residual-stream statistics per block).

### Results IO

`save_results(results, path.json|.jsonl)`, `load_results`, `merge_metrics` → `{"suite/metric": value}`.

## Configuration

`forgeline evaluate --checkpoint DIR --suites gsm8k mmlu humaneval [--offline] [--n-shot 5] [--max-tasks 200] [--heldout-data DIR] --output eval.json`

`configs/evaluation/quick_offline.yaml` lists the offline set.

## Failure modes

Dataset download failures fall back to samples (logged); code execution failures score as incorrect; regression and gate checks fail closed on missing/NaN metrics.

## Local validation

`tests/unit/test_evaluation.py` (statistics, IO, regression rules, held-out loss, pass rate, every benchmark offline, scoring functions, performance measurement).

## Hardware requirements

CPU for everything; full benchmark sets on large models need a GPU for reasonable time.

## Limitations

* Multiple-choice benchmarks are scored generatively (first emitted letter/number), not by log-likelihood ranking of choices.
* Offline sample sets are tiny and only verify wiring.
