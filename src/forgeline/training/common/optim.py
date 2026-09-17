"""Optimizer construction with weight-decay parameter groups."""

from __future__ import annotations

from typing import Iterable, List

import torch

from forgeline.core.config import OptimizerConfig


def build_optimizer(params: Iterable[torch.nn.Parameter], cfg: OptimizerConfig, device: torch.device) -> torch.optim.Optimizer:
    params = [p for p in params if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters")
    if cfg.decay_only_matrices:
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups: List[dict] = [{"params": decay, "weight_decay": cfg.weight_decay}]
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
    else:
        groups = [{"params": params, "weight_decay": cfg.weight_decay}]
    use_fused = (cfg.fused == "always") or (cfg.fused == "auto" and device.type == "cuda")
    kwargs = dict(lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
    if use_fused:
        kwargs["fused"] = True
    if cfg.name == "adam":
        return torch.optim.Adam(groups, **kwargs)
    return torch.optim.AdamW(groups, **kwargs)
