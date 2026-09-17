"""Expert router: softmax/sigmoid scoring, aux-loss-free bias, group-limited routing."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from forgeline.core.config import ModelSpec


class ExpertGate(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.dim = spec.n_embd
        self.topk = spec.n_experts_active
        self.n_experts = spec.n_experts
        self.score_func = spec.score_func
        self.route_scale = spec.route_scale
        self.aux_loss_free = spec.aux_loss_free
        self.bias_update_speed = spec.bias_update_speed
        self.aux_loss_weight = spec.moe_aux_loss_weight
        self.n_expert_groups = spec.n_expert_groups
        self.n_limited_groups = spec.n_limited_groups

        self.weight = nn.Parameter(torch.empty(spec.n_experts, spec.n_embd))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.aux_loss_free:
            self.register_buffer("expert_bias", torch.zeros(spec.n_experts, dtype=torch.float32))
        self._aux_loss: Optional[torch.Tensor] = None

    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        return self._aux_loss

    def forward(self, x: torch.Tensor):
        scores = F.linear(x.float(), self.weight.float())
        scores = scores.softmax(dim=-1) if self.score_func == "softmax" else scores.sigmoid()
        original_scores = scores
        routing_scores = scores + self.expert_bias.unsqueeze(0) if self.aux_loss_free else scores

        if self.n_expert_groups > 1 and self.n_limited_groups < self.n_expert_groups:
            N = x.shape[0]
            per_group = self.n_experts // self.n_expert_groups
            group_scores = routing_scores.view(N, self.n_expert_groups, per_group)
            group_max = group_scores.max(dim=-1).values
            _, top_groups = group_max.topk(self.n_limited_groups, dim=-1)
            group_mask = torch.zeros(N, self.n_expert_groups, device=x.device)
            group_mask.scatter_(1, top_groups, 1.0)
            expert_mask = group_mask.unsqueeze(-1).expand(-1, -1, per_group).reshape(N, self.n_experts)
            routing_scores = routing_scores.masked_fill(expert_mask == 0, float("-inf"))

        _, indices = torch.topk(routing_scores, self.topk, dim=-1)
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        weights = weights * self.route_scale

        if self.training and self.aux_loss_free:
            with torch.no_grad():
                counts = torch.zeros(self.n_experts, device=x.device)
                for k in range(self.topk):
                    counts.scatter_add_(0, indices[:, k], torch.ones(indices.shape[0], device=x.device))
                avg_count = counts.float().mean()
                self.expert_bias += self.bias_update_speed * (avg_count - counts)
            self._aux_loss = torch.tensor(0.0, device=x.device)
        elif self.training:
            one_hot = F.one_hot(indices, self.n_experts).float()
            tokens_per_expert = one_hot.sum(dim=(0, 1))
            fraction_tokens = tokens_per_expert / (x.shape[0] * self.topk)
            fraction_probs = original_scores.mean(dim=0)
            self._aux_loss = self.aux_loss_weight * self.n_experts * (fraction_tokens * fraction_probs).sum()
        else:
            self._aux_loss = torch.tensor(0.0, device=x.device)

        return weights.type_as(x), indices
