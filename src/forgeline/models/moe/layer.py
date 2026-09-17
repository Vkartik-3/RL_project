"""Mixture-of-experts layer with optional always-on shared experts."""

from __future__ import annotations

import torch
import torch.nn as nn

from forgeline.core.config import ModelSpec
from forgeline.models.moe.gate import ExpertGate
from forgeline.models.transformer.feedforward import make_ffn


class MoELayer(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.n_experts = spec.n_experts
        self.n_experts_active = spec.n_experts_active
        self.n_shared_experts = spec.n_shared_experts
        self.gate = ExpertGate(spec)
        self.experts = nn.ModuleList([make_ffn(spec) for _ in range(spec.n_experts)])
        self.shared_experts = make_ffn(spec) if spec.n_shared_experts > 0 else None

    @property
    def aux_loss(self):
        return self.gate.aux_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        weights, indices = self.gate(x_flat)
        y = torch.zeros_like(x_flat)
        for i in range(self.n_experts_active):
            expert_idx = indices[:, i]
            expert_weight = weights[:, i]
            for e in range(self.n_experts):
                mask = expert_idx == e
                if mask.any():
                    y[mask] += expert_weight[mask].unsqueeze(-1) * self.experts[e](x_flat[mask])
        if self.shared_experts is not None:
            y = y + self.shared_experts(x_flat)
        return y.view(B, T, C)
