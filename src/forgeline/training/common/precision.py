"""Mixed-precision helpers: autocast context + gradient scaler selection."""

from __future__ import annotations

from typing import Any, ContextManager

import torch

from forgeline.core.config import PrecisionConfig
from forgeline.core.runtime import autocast_context


class Precision:
    def __init__(self, cfg: PrecisionConfig, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        use_scaler = cfg.grad_scaler == "on" or (
            cfg.grad_scaler == "auto" and device.type == "cuda" and dtype == torch.float16
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=bool(use_scaler and device.type == "cuda"))

    def autocast(self) -> ContextManager[Any]:
        return autocast_context(self.device, self.dtype)

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()

    def unscale(self, optimizer: torch.optim.Optimizer) -> None:
        self.scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.scaler.step(optimizer)
        self.scaler.update()

    def state_dict(self) -> dict:
        return self.scaler.state_dict()

    def load_state_dict(self, state: dict) -> None:
        if state:
            self.scaler.load_state_dict(state)
