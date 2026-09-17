"""Advantage estimators shared by PPO / GRPO / DAPO."""

from __future__ import annotations

from typing import Tuple

import torch


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float = 0.99, lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over a ``[T]`` reward sequence with ``[T+1]`` values."""
    T = rewards.shape[0]
    advantages = torch.zeros(T, dtype=torch.float32)
    gae = 0.0
    for t in reversed(range(T)):
        delta = float(rewards[t]) + gamma * float(values[t + 1]) - float(values[t])
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values[:T].float()
    return advantages, returns


def normalize_advantages(adv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """``(a − mean) / (std + eps)``; a single element normalises to 0."""
    if adv.numel() <= 1:
        return torch.zeros_like(adv)
    return (adv - adv.mean()) / (adv.std() + eps)


def group_relative_advantages(rewards: torch.Tensor, eps: float = 1e-8, clamp_std: bool = False) -> torch.Tensor:
    """GRPO advantages within one group: ``(r − mean) / (std + eps)``.

    ``clamp_std=True`` uses ``std.clamp(min=eps)`` instead of ``std + eps``.
    """
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)
    std = rewards.std().clamp(min=eps) if clamp_std else rewards.std() + eps
    return (rewards - rewards.mean()) / std
