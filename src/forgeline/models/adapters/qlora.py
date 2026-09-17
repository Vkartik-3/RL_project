"""QLoRA: NF4-quantized frozen base weights + full-precision LoRA adapters."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from forgeline.core.errors import ModelError
from forgeline.models.adapters.lora import DEFAULT_TARGET_MODULES, _iter_linear_attrs
from forgeline.models.quantization.nf4 import NF4Linear


class QLoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.base = NF4Linear(original)
        self.lora_A = nn.Parameter(torch.zeros(original.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        if not self.enabled:
            return result
        return result + (self.dropout(x) @ self.lora_A.to(x.dtype) @ self.lora_B.to(x.dtype)) * self.scaling


def apply_qlora(model: nn.Module, rank: int = 16, alpha: float = 32.0, dropout: float = 0.0,
                target_modules: Optional[Sequence[str]] = None) -> nn.Module:
    targets = list(target_modules) if target_modules else [t for t in DEFAULT_TARGET_MODULES]
    for p in model.parameters():
        p.requires_grad = False
    count = 0
    for _, parent, attr_name, child in _iter_linear_attrs(model):
        if isinstance(child, nn.Linear) and any(t in attr_name for t in targets):
            setattr(parent, attr_name, QLoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            count += 1
    if count == 0:
        raise ModelError(f"No linear layers matched target_modules={targets}")
    return model
