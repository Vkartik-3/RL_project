"""Learning-rate schedules: cosine with warmup, warmup-stable-decay, constant, linear decay."""

from __future__ import annotations

import math

from forgeline.core.config import ScheduleConfig


def learning_rate_at(step: int, peak_lr: float, cfg: ScheduleConfig) -> float:
    """Learning rate for ``step`` (0-based) under ``cfg``."""
    if cfg.name == "constant":
        if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
            return peak_lr * step / cfg.warmup_steps
        return peak_lr
    if cfg.name == "wsd":
        if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
            return peak_lr * step / cfg.warmup_steps
        stable_end = int(cfg.decay_steps * cfg.wsd_stable_fraction)
        if step < stable_end:
            return peak_lr
        decay_ratio = min((step - stable_end) / max(1, cfg.decay_steps - stable_end), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (peak_lr - cfg.min_lr)
    if cfg.name == "linear_decay":
        if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
            return peak_lr * step / cfg.warmup_steps
        if step >= cfg.decay_steps:
            return cfg.min_lr
        frac = (step - cfg.warmup_steps) / max(1, cfg.decay_steps - cfg.warmup_steps)
        return peak_lr + (cfg.min_lr - peak_lr) * min(max(frac, 0.0), 1.0)
    # cosine with linear warmup (default)
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return peak_lr * step / cfg.warmup_steps
    if step > cfg.decay_steps:
        return cfg.min_lr
    denom = max(1, cfg.decay_steps - cfg.warmup_steps)
    decay_ratio = (step - cfg.warmup_steps) / denom
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (peak_lr - cfg.min_lr)
