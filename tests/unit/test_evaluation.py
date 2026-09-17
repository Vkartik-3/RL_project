import math

import pytest
import torch

from forgeline.core.protocols import EvaluationResult
from forgeline.data.pretraining import MemmapCorpus
from forgeline.data.records import VerifiableTask
from forgeline.evaluation.performance import measure_generation
from forgeline.evaluation.quality import HeldOutLossSuite, VerifierPassRateSuite
from forgeline.evaluation.regression import RegressionRule, evaluate_regression
from forgeline.evaluation.results import load_results, merge_metrics, save_results
from forgeline.evaluation.reward import PreferenceWinRateSuite, RecordRewardSuite, reward_statistics
from forgeline.evaluation.suites import BENCHMARKS, BenchmarkSuite, build_benchmark
from forgeline.rollouts.rewards import VerifierReward
from forgeline.rollouts.verifiers import MathVerifier


def test_reward_statistics():
    s = reward_statistics([0.5, 1.0, 0.8], baseline=0.5)
    assert s["mean"] == pytest.approx(0.7666, abs=1e-3) and s["pct_above_0.75"] == pytest.approx(66.66, abs=0.1)
    assert s["improvement_vs_baseline_pct"] > 0
    assert reward_statistics([])["n"] == 0


def test_record_and_preference_suites():
    r = RecordRewardSuite([{"v": 0.2}, {"v": 0.9}], lambda rec: rec["v"]).run()
    assert r.details["policy_dependent"] is False and r.metrics["max"] == 0.9
    w = PreferenceWinRateSuite([{"prompt": "p", "chosen": "long answer", "rejected": "x"}], lambda p, r: len(r)).run()
    assert w.metrics["win_rate"] == 1.0


def test_results_io_and_merge(tmp_path):
    res = [EvaluationResult("a", {"x": 1.0}, 3), EvaluationResult("b", {"y": 2.0})]
    for ext in ("json", "jsonl"):
        p = save_results(res, tmp_path / f"r.{ext}")
        back = load_results(p)
        assert [r.suite for r in back] == ["a", "b"] and back[0].n_samples == 3
    assert merge_metrics(res) == {"a/x": 1.0, "b/y": 2.0}


def test_regression_rules():
    rules = [RegressionRule("acc"), RegressionRule("latency", "lower_is_better", max_regression=0.1, relative=True)]
    ok = evaluate_regression({"acc": 0.9, "latency": 105}, {"acc": 0.9, "latency": 100}, rules)
    assert ok.passed
    bad = evaluate_regression({"acc": 0.8, "latency": 120}, {"acc": 0.9, "latency": 100}, rules)
    assert not bad.passed and len(bad.failures()) == 2
    assert not evaluate_regression({"acc": float("nan")}, {"acc": 1.0}, [RegressionRule("acc")]).passed
    assert not evaluate_regression({}, {"acc": 1.0}, [RegressionRule("acc")]).passed
    assert not evaluate_regression({"acc": 1.0}, {}, [RegressionRule("acc")]).passed


def test_heldout_loss_and_pass_rate(corpus_dir, policy):
    r = HeldOutLossSuite(MemmapCorpus(corpus_dir, "val", 64), batches=2, batch_size=2).run(policy.model)
    assert math.isfinite(r.metrics["loss"]) and r.metrics["perplexity"] > 1
    tasks = [VerifiableTask(prompt="What is 1 + 1?", answer="2")]
    r = VerifierPassRateSuite(tasks, VerifierReward(MathVerifier()), max_new_tokens=4).run(policy)
    assert set(r.metrics) == {"pass_rate", "reward", "malformed_rate"} and r.n_samples == 1 and r.per_item


def test_benchmarks_offline(policy):
    assert set(BENCHMARKS.names()) == {"arc", "gsm8k", "hellaswag", "humaneval", "mmlu", "truthfulqa"}
    for name in BENCHMARKS.names():
        b = build_benchmark(name, offline=True, max_tasks=2)
        assert len(b.tasks()) == 2
        r = BenchmarkSuite(b, max_new_tokens=4).run(policy)
        assert 0.0 <= r.metrics["accuracy"] <= 1.0 and r.details["offline_samples"] is True
    assert build_benchmark("mmlu").score({"answer": "B"}, " b. because") and not build_benchmark("gsm8k").score({"answer": "8"}, "nine")
    assert build_benchmark("gsm8k").score({"answer": "8"}, "so 3+5 = 8")
    assert build_benchmark("humaneval").score({"prompt": "def f(x):\n", "test": "assert f(2) == 4\n"}, "    return x * 2\n")


def test_measure_generation(policy):
    m = measure_generation(policy, torch.tensor(policy.encode("hi")), max_new_tokens=4, runs=2, warmup=0)
    assert m["tokens_per_second"] > 0 and m["runs"] == 2 and "peak_memory_gb" not in m or torch.cuda.is_available()
