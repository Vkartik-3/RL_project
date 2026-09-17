"""Feed-forward networks: SwiGLU and GELU."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from forgeline.core.config import ModelSpec


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * 4 * dim / 3)
            hidden_dim = 64 * ((hidden_dim + 63) // 64)
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class GELUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.c_fc = nn.Linear(dim, hidden_dim, bias=bias)
        self.c_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


def make_ffn(spec: ModelSpec, hidden_dim: Optional[int] = None) -> nn.Module:
    if spec.use_swiglu:
        return SwiGLUFeedForward(spec.n_embd, hidden_dim, spec.bias, spec.dropout)
    return GELUFeedForward(spec.n_embd, hidden_dim, spec.bias, spec.dropout)
