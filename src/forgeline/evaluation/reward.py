"""Reward statistics over scored records or trajectories."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Sequence

from forgeline.core.protocols import EvaluationResult, RewardProvider, Trajectory


def reward_statistics(rewards: Sequence[float], baseline: float = 0.5, thresholds: Sequence[float] = (0.6, 0.75)) -> Dict[str, float]:
    if not rewards:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    n = len(rewards)
    mean = sum(rewards) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / n)
    out = {"mean": mean, "std": std, "min": min(rewards), "max": max(rewards), "n": float(n),
           "improvement_vs_baseline_pct": (mean - baseline) / baseline * 100.0 if baseline else 0.0}
    for t in thresholds:
        out[f"pct_above_{t}"] = 100.0 * sum(1 for r in rewards if r > t) / n
    return out


class RecordRewardSuite:
    """Scores dataset records with a rule/learned scorer.

    This measures the *dataset*, not a policy: the numbers are policy-independent.
    ``details.policy_dependent`` is False so downstream reports can label it correctly.
    """

    name = "record_reward"

    def __init__(self, records: Sequence[Dict[str, Any]], score_fn: Callable[[Dict[str, Any]], float], baseline: float = 0.5):
        self.records, self.score_fn, self.baseline = list(records), score_fn, baseline

    def run(self, target: Any = None, **kwargs: Any) -> EvaluationResult:
        rewards = [self.score_fn(r) for r in self.records]
        return EvaluationResult(self.name, reward_statistics(rewards, self.baseline), n_samples=len(rewards),
                                details={"policy_dependent": False})


class TrajectoryRewardSuite:
    """Scores policy trajectories (policy-dependent)."""

    name = "trajectory_reward"

    def __init__(self, reward: RewardProvider, baseline: float = 0.5):
        self.reward, self.baseline = reward, baseline

    def run(self, trajectories: Sequence[Trajectory], **kwargs: Any) -> EvaluationResult:
        rewards = [self.reward.score(t).value for t in trajectories]
        return EvaluationResult(self.name, reward_statistics(rewards, self.baseline), n_samples=len(rewards),
                                details={"policy_dependent": True})


class PreferenceWinRateSuite:
    """Fraction of pairs where the scorer ranks ``chosen`` above ``rejected``."""

    name = "preference_win_rate"

    def __init__(self, pairs: Sequence[Dict[str, str]], score_fn: Callable[[str, str], float]):
        self.pairs, self.score_fn = list(pairs), score_fn

    def run(self, target: Any = None, **kwargs: Any) -> EvaluationResult:
        wins = 0
        margins: List[float] = []
        for p in self.pairs:
            c, r = self.score_fn(p["prompt"], p["chosen"]), self.score_fn(p["prompt"], p["rejected"])
            wins += int(c > r)
            margins.append(c - r)
        n = max(len(self.pairs), 1)
        return EvaluationResult(self.name, {"win_rate": wins / n, "mean_margin": sum(margins) / n}, n_samples=len(self.pairs))
