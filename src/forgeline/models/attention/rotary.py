"""Rotary positional embeddings with linear and YaRN context extension."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """RoPE with optional ``linear`` or ``yarn`` (NTK-aware) scaling.

    ``scaling_factor`` = target_context / original_context.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        scaling_type: str = "none",
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor

        if scaling_type == "yarn":
            base = base * (scaling_factor ** (dim / (dim - 2)))
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            freqs = torch.arange(0, dim, 2).float() / dim
            wavelen = 2 * math.pi * base ** freqs
            orig_max_seq_len = max_seq_len / scaling_factor
            ratio = orig_max_seq_len / wavelen
            smooth = torch.clamp((ratio - 1) / (scaling_factor - 1), 0.0, 1.0)
            yarn_scale = (1 - smooth) / scaling_factor + smooth
            inv_freq = inv_freq * yarn_scale
        elif scaling_type == "linear":
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            inv_freq = inv_freq / scaling_factor
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0):
        end = offset + seq_len
        if end > self.cos_cached.shape[0]:
            self._build_cache(end * 2)
        return self.cos_cached[offset:end], self.sin_cached[offset:end]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to one tensor of shape ``[B, H, T, D]``."""
    cos = cos.unsqueeze(0).unsqueeze(0).to(x.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


def apply_rotary_pos_emb_pair(q, k, cos, sin):
    return apply_rotary_pos_emb(q, cos, sin), apply_rotary_pos_emb(k, cos, sin)
