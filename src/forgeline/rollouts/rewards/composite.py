"""Compose verifiers and reward providers into one reward signal."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from forgeline.core.errors import InvalidRewardError
from forgeline.core.protocols import RewardProvider, RewardResult, Trajectory, Verifier
from forgeline.rollouts.verifiers import tag_format_reward


def _check_finite(value: float, source: str) -> float:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise InvalidRewardError(f"reward provider {source!r} returned non-finite value {value!r}")
    return float(value)


class VerifierReward:
    """Adapts a :class:`Verifier` to the :class:`RewardProvider` protocol.

    ``target_key`` selects which task field is the target (``answer`` or ``test_code``).
    """

    def __init__(self, verifier: Verifier, target_key: str = "answer", weight: float = 1.0):
        self.verifier = verifier
        self.target_key = target_key
        self.weight = weight
        self.name = f"verifier:{verifier.name}"

    def score(self, trajectory: Trajectory) -> RewardResult:
        target = trajectory.task.get(self.target_key)
        result = self.verifier.verify(trajectory.response_text, target)
        value = _check_finite(result.value, self.name) * self.weight
        return RewardResult(value=value, passed=result.passed, components={self.name: result.value}, info=result.info)


class TagFormatReward:
    """Small structural reward for ``<think>``/``<tool_call>``/``<final_answer>`` tags."""

    name = "tag_format"

    def score(self, trajectory: Trajectory) -> RewardResult:
        return RewardResult(value=tag_format_reward(trajectory.response_text))


class CompositeReward:
    """Weighted sum of providers; ``passed`` is taken from the first provider that reports it."""

    def __init__(self, providers: Sequence[Tuple[RewardProvider, float]], name: str = "composite",
                 clip: Optional[Tuple[float, float]] = None):
        if not providers:
            raise InvalidRewardError("composite reward needs at least one provider")
        self.providers = list(providers)
        self.name = name
        self.clip = clip

    def score(self, trajectory: Trajectory) -> RewardResult:
        total = 0.0
        components: Dict[str, float] = {}
        passed: Optional[bool] = None
        info: Dict[str, Any] = {}
        for provider, weight in self.providers:
            r = provider.score(trajectory)
            v = _check_finite(r.value, provider.name)
            components[provider.name] = v
            total += weight * v
            if passed is None and r.passed is not None:
                passed = r.passed
            if r.info:
                info[provider.name] = r.info
        if self.clip is not None:
            total = min(max(total, self.clip[0]), self.clip[1])
        return RewardResult(value=total, components=components, passed=passed, info=info)


class CallableReward:
    """Wrap a plain ``fn(trajectory) -> float`` as a provider."""

    def __init__(self, fn: Callable[[Trajectory], float], name: str = "callable"):
        self.fn = fn
        self.name = name

    def score(self, trajectory: Trajectory) -> RewardResult:
        return RewardResult(value=_check_finite(self.fn(trajectory), self.name))


def score_trajectories(provider: RewardProvider, trajectories: List[Trajectory]) -> List[float]:
    """Score in place (sets ``trajectory.reward``) and return the reward values.

    Providers with ``score_many(trajectories)`` (e.g. a Ray reward-worker pool) score the whole list in one call.
    """
    batch = getattr(provider, "score_many", None)
    results = batch(trajectories) if callable(batch) else [provider.score(t) for t in trajectories]
    values = []
    for t, r in zip(trajectories, results):
        _check_finite(r.value, provider.name)
        t.reward = r
        values.append(r.value)
    return values
