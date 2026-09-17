"""Sequence log-prob reductions and KL / entropy estimators used by RL objectives."""

from __future__ import annotations

from typing import Optional

import torch


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim)
    mask = mask.to(x.dtype)
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1.0)


def masked_sum(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1) -> torch.Tensor:
    if mask is None:
        return x.sum(dim=dim)
    return (x * mask.to(x.dtype)).sum(dim=dim)


def reduce_logprobs(token_lp: torch.Tensor, mask: Optional[torch.Tensor], reduction: str) -> torch.Tensor:
    """Reduce ``[B, R]`` token log-probs to ``[B]`` with ``sum`` or ``mean``."""
    if reduction == "sum":
        return masked_sum(token_lp, mask)
    if reduction == "mean":
        return masked_mean(token_lp, mask)
    raise ValueError(f"reduction must be sum|mean, got {reduction!r}")


def kl_logprob_diff(ref_lp: torch.Tensor, new_lp: torch.Tensor, mask: Optional[torch.Tensor] = None,
                    clamp_min: Optional[float] = 0.0) -> torch.Tensor:
    """Per-sequence ``mean(ref − new)`` estimator (optionally clamped at 0)."""
    kl = masked_mean(ref_lp - new_lp, mask)
    return kl.clamp(min=clamp_min) if clamp_min is not None else kl


def kl_k3(ref_lp: torch.Tensor, new_lp: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Unbiased low-variance k3 estimator: ``exp(ref−new) − (ref−new) − 1`` averaged per sequence."""
    log_ratio = ref_lp - new_lp
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    return masked_mean(kl, mask)


def kl_full_distribution(new_logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    """Exact per-token ``KL(new || ref)`` from full-vocabulary logits ``[B, T, V]``."""
    new_lp = torch.log_softmax(new_logits.float(), dim=-1)
    ref_lp = torch.log_softmax(ref_logits.float(), dim=-1)
    return (new_lp.exp() * (new_lp - ref_lp)).sum(-1)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Per-position entropy ``[B, T]`` of the full distribution."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    return -(lp.exp() * lp).sum(-1)
