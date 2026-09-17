"""Reward providers backed by learned models (sequence reward model, blended rule/neural)."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from forgeline.core.protocols import RewardResult, Trajectory
from forgeline.models.reward import SequenceRewardModel


class RewardModelProvider:
    """Score ``prompt + response`` tokens with a :class:`SequenceRewardModel`."""

    name = "reward_model"

    def __init__(self, model: SequenceRewardModel, max_length: Optional[int] = None):
        self.model = model.eval()
        self.max_length = max_length or model.spec.block_size

    @torch.no_grad()
    def score(self, trajectory: Trajectory) -> RewardResult:
        ids = torch.cat([trajectory.prompt_ids, trajectory.response_ids]).unsqueeze(0)
        ids = ids[:, -self.max_length:].to(next(self.model.parameters()).device)
        value = float(self.model(ids).item())
        return RewardResult(value=value)


class BlendedReward:
    """``blend · neural + (1 − blend) · rule``, clipped to [0, 1]; falls back to rule-only."""

    name = "blended"

    def __init__(self, rule_fn: Callable[[Trajectory], float], neural: Optional[Callable[[Trajectory], float]] = None,
                 neural_weight: float = 0.6):
        self.rule_fn, self.neural, self.neural_weight = rule_fn, neural, neural_weight

    def score(self, trajectory: Trajectory) -> RewardResult:
        rule = float(self.rule_fn(trajectory))
        if self.neural is None:
            return RewardResult(value=rule, components={"rule": rule})
        neural = float(self.neural(trajectory))
        value = max(0.0, min(1.0, (1 - self.neural_weight) * rule + self.neural_weight * neural))
        return RewardResult(value=value, components={"rule": rule, "neural": neural})
