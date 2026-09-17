"""Grouped-query causal self-attention with KV cache, sliding window and logit cap."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from forgeline.core.config import ModelSpec
from forgeline.models.attention.rotary import RotaryEmbedding, apply_rotary_pos_emb_pair

KVCache = Tuple[torch.Tensor, torch.Tensor]


class CausalSelfAttention(nn.Module):
    """Multi-head / grouped-query attention.

    * full causal attention via SDPA when no cache / window / cap is used
    * sliding-window attention (optionally on odd layers only)
    * logit soft-capping
    * incremental decoding with a ``(k, v)`` cache
    """

    def __init__(self, spec: ModelSpec, layer_id: int = 0):
        super().__init__()
        self.n_head = spec.n_head
        self.n_kv_head = spec.n_kv_head
        self.n_embd = spec.n_embd
        self.head_dim = spec.n_embd // spec.n_head
        self.n_rep = spec.n_head // spec.n_kv_head
        self.dropout_rate = spec.dropout
        self.use_rope = spec.use_rope
        self.attn_logit_cap = spec.attn_logit_cap

        if spec.alternating_layers and layer_id % 2 == 1:
            self.window_size = spec.sliding_window
        elif not spec.alternating_layers and spec.sliding_window > 0:
            self.window_size = spec.sliding_window
        else:
            self.window_size = 0

        self.q_proj = nn.Linear(spec.n_embd, spec.n_head * self.head_dim, bias=spec.bias)
        self.k_proj = nn.Linear(spec.n_embd, spec.n_kv_head * self.head_dim, bias=spec.bias)
        self.v_proj = nn.Linear(spec.n_embd, spec.n_kv_head * self.head_dim, bias=spec.bias)
        self.c_proj = nn.Linear(spec.n_embd, spec.n_embd, bias=spec.bias)

        self.attn_dropout = nn.Dropout(spec.dropout)
        self.resid_dropout = nn.Dropout(spec.dropout)

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_seq_len=spec.block_size,
                scaling_type=spec.rope_scaling_type,
                scaling_factor=spec.rope_scaling_factor,
            )

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(spec.block_size, spec.block_size)).view(1, 1, spec.block_size, spec.block_size),
            persistent=False,
        )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        B, n_kv, T, D = x.shape
        return x[:, :, None, :, :].expand(B, n_kv, self.n_rep, T, D).reshape(B, self.n_head, T, D)

    def _sliding_window_mask(self, T: int, total_T: int, device) -> torch.Tensor:
        q_pos = torch.arange(total_T - T, total_T, device=device).unsqueeze(1)
        k_pos = torch.arange(total_T, device=device).unsqueeze(0)
        allowed = (k_pos <= q_pos) & (q_pos - k_pos < self.window_size)
        return allowed.view(1, 1, T, total_T)

    def forward(self, x: torch.Tensor, past_kv: Optional[KVCache] = None):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            offset = past_kv[0].shape[2] if past_kv is not None else 0
            cos, sin = self.rotary_emb(T, offset=offset)
            q, k = apply_rotary_pos_emb_pair(q, k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present_kv = (k, v)

        k_exp = self._repeat_kv(k)
        v_exp = self._repeat_kv(v)

        use_flash = past_kv is None and self.window_size == 0 and self.attn_logit_cap == 0
        if use_flash:
            y = F.scaled_dot_product_attention(
                q, k_exp, v_exp, attn_mask=None,
                dropout_p=self.dropout_rate if self.training else 0.0,
                is_causal=True,
            )
        else:
            total_len = k_exp.shape[2]
            att = (q @ k_exp.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if self.attn_logit_cap > 0:
                att = torch.tanh(att / self.attn_logit_cap) * self.attn_logit_cap
            if past_kv is None:
                if self.window_size > 0:
                    mask = self._sliding_window_mask(T, total_len, x.device)
                    att = att.masked_fill(~mask, float("-inf"))
                else:
                    att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
            elif T > 1:
                # cached path with more than one new token: causal over the new block
                mask = self._sliding_window_mask(T, total_len, x.device) if self.window_size > 0 else (
                    torch.arange(total_len, device=x.device).unsqueeze(0)
                    <= torch.arange(total_len - T, total_len, device=x.device).unsqueeze(1)
                ).view(1, 1, T, total_len)
                att = att.masked_fill(~mask, float("-inf"))
            elif self.window_size > 0:
                start = max(0, total_len - self.window_size)
                if start > 0:
                    att[..., :start] = float("-inf")
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v_exp

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, present_kv
