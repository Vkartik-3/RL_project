"""Multi-head latent attention (low-rank KV compression with decoupled RoPE)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from forgeline.core.config import ModelSpec
from forgeline.models.attention.rotary import RotaryEmbedding, apply_rotary_pos_emb
from forgeline.models.transformer.norm import RMSNorm

LatentCache = Tuple[torch.Tensor, torch.Tensor]  # (kv_latent [B,T,r], k_rope [B,1,T,d_rope])


class MultiHeadLatentAttention(nn.Module):
    """MLA: compresses K/V into a latent vector; caches only ``(c_kv, k_rope)``.

    Two attention modes:
      * naive — decompress the latent to per-head K/V (training / prefill)
      * absorbed — fold the up-projection into the query and attend in latent
        space (single-token decode)
    """

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.n_heads = spec.n_head
        self.dim = spec.n_embd
        self.kv_lora_rank = spec.kv_lora_rank
        self.q_lora_rank = spec.q_lora_rank
        self.qk_nope_head_dim = spec.qk_nope_head_dim
        self.qk_rope_head_dim = spec.qk_rope_head_dim
        self.qk_head_dim = spec.qk_nope_head_dim + spec.qk_rope_head_dim
        self.v_head_dim = spec.v_head_dim
        self.attn_logit_cap = spec.attn_logit_cap

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)

        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.qk_rope_head_dim, max_seq_len=spec.block_size,
            scaling_type=spec.rope_scaling_type, scaling_factor=spec.rope_scaling_factor,
        )
        self.softmax_scale = self.qk_head_dim ** -0.5
        self.attn_dropout = nn.Dropout(spec.dropout)
        self.resid_dropout = nn.Dropout(spec.dropout)

    def _cap(self, att: torch.Tensor) -> torch.Tensor:
        if self.attn_logit_cap > 0:
            att = torch.tanh(att / self.attn_logit_cap) * self.attn_logit_cap
        return att

    def _forward_naive(self, q_nope, q_rope, kv_latent, k_rope, B, T):
        kv_full = self.wkv_b(kv_latent).view(B, -1, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_full.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_nope = k_nope.transpose(1, 2)
        k = torch.cat([k_nope, k_rope.expand(-1, self.n_heads, -1, -1)], dim=-1)
        q = torch.cat([q_nope.transpose(1, 2), q_rope], dim=-1)
        v = v.transpose(1, 2)

        total_T = k.shape[2]
        att = self._cap((q @ k.transpose(-2, -1)) * self.softmax_scale)
        if T > 1:
            causal = torch.triu(torch.full((T, total_T), float("-inf"), device=q.device), diagonal=1 + total_T - T)
            att = att + causal.unsqueeze(0).unsqueeze(0)
        att = att.softmax(dim=-1, dtype=torch.float32).type_as(q)
        att = self.attn_dropout(att)
        return att @ v

    def _forward_absorbed(self, q_nope, q_rope, kv_latent, k_rope, B, T):
        total_T = kv_latent.shape[1]
        w_full = self.wkv_b.weight.view(self.n_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        w_k_nope = w_full[:, : self.qk_nope_head_dim, :]
        w_v = w_full[:, self.qk_nope_head_dim :, :]

        q_nope = q_nope.transpose(1, 2)
        q_absorbed = torch.einsum("bhtn,hnl->bhtl", q_nope, w_k_nope)
        kv_lat = kv_latent.unsqueeze(1)
        att_nope = torch.matmul(q_absorbed, kv_lat.transpose(-2, -1))
        att_rope = torch.matmul(q_rope, k_rope.expand(-1, self.n_heads, -1, -1).transpose(-2, -1))
        att = self._cap((att_nope + att_rope) * self.softmax_scale)
        if T > 1:
            causal = torch.triu(torch.full((T, total_T), float("-inf"), device=att.device), diagonal=1 + total_T - T)
            att = att + causal.unsqueeze(0).unsqueeze(0)
        att = att.softmax(dim=-1, dtype=torch.float32).type_as(q_nope)
        att = self.attn_dropout(att)
        latent_out = torch.matmul(att, kv_lat)
        return torch.einsum("bhtl,hvl->bhtv", latent_out, w_v)

    def forward(self, x: torch.Tensor, past_kv: Optional[LatentCache] = None):
        B, T, _ = x.size()
        q = self.wq(x) if self.q_lora_rank == 0 else self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(B, T, self.n_heads, self.qk_head_dim)
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv_combined = self.wkv_a(x)
        kv_latent, k_rope_raw = kv_combined.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_latent = self.kv_norm(kv_latent)

        offset = past_kv[0].shape[1] if past_kv is not None else 0
        cos, sin = self.rotary_emb(T, offset=offset)
        q_rope = apply_rotary_pos_emb(q_rope.transpose(1, 2), cos, sin)
        k_rope = apply_rotary_pos_emb(k_rope_raw.unsqueeze(2).transpose(1, 2), cos, sin)

        if past_kv is not None:
            past_latent, past_rope = past_kv
            kv_latent = torch.cat([past_latent, kv_latent], dim=1)
            k_rope = torch.cat([past_rope, k_rope], dim=2)
        present_kv = (kv_latent, k_rope)

        use_absorbed = (not self.training) and (T == 1)
        if use_absorbed:
            y = self._forward_absorbed(q_nope, q_rope, kv_latent, k_rope, B, T)
        else:
            y = self._forward_naive(q_nope, q_rope, kv_latent, k_rope, B, T)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.v_head_dim)
        return self.resid_dropout(self.wo(y)), present_kv
