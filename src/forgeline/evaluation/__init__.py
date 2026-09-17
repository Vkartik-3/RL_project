"""Evaluation: structured results, quality/reward/performance suites, benchmarks and regression rules."""

from forgeline.core.protocols import EvaluationResult
from forgeline.evaluation.performance import PerformanceSuite, measure_generation
from forgeline.evaluation.quality import AgentAccuracySuite, HeldOutLossSuite, VerifierPassRateSuite
from forgeline.evaluation.regression import RegressionReport, RegressionRule, evaluate_regression
from forgeline.evaluation.results import load_results, merge_metrics, save_results
from forgeline.evaluation.reward import PreferenceWinRateSuite, RecordRewardSuite, TrajectoryRewardSuite, reward_statistics
from forgeline.evaluation.suites import BENCHMARKS, BenchmarkSuite, build_benchmark

__all__ = [
    "EvaluationResult", "PerformanceSuite", "measure_generation", "AgentAccuracySuite", "HeldOutLossSuite",
    "VerifierPassRateSuite", "RegressionReport", "RegressionRule", "evaluate_regression", "load_results", "merge_metrics",
    "save_results", "PreferenceWinRateSuite", "RecordRewardSuite", "TrajectoryRewardSuite", "reward_statistics",
    "BENCHMARKS", "BenchmarkSuite", "build_benchmark",
]
