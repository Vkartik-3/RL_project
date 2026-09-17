"""Native sparse attention: compressed + top-k selected + sliding-window branches."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from forgeline.core.config import ModelSpec
from forgeline.models.attention.rotary import RotaryEmbedding, apply_rotary_pos_emb_pair


class NativeSparseAttention(nn.Module):
    """Three-branch sparse attention fused by learned per-head gates.

    Limitation (inherited): the incremental KV-cache decode path is not
    implemented for this module; use full-sequence forward passes.
    """

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.n_head = spec.n_head
        self.head_dim = spec.n_embd // spec.n_head
        self.n_embd = spec.n_embd
        self.nsa_block_size = spec.nsa_block_size
        self.nsa_top_k = spec.nsa_top_k
        self.nsa_window_size = spec.nsa_window_size

        self.q_proj = nn.Linear(spec.n_embd, spec.n_embd, bias=spec.bias)
        self.k_proj = nn.Linear(spec.n_embd, spec.n_embd, bias=spec.bias)
        self.v_proj = nn.Linear(spec.n_embd, spec.n_embd, bias=spec.bias)
        self.c_proj = nn.Linear(spec.n_embd, spec.n_embd, bias=spec.bias)

        self.compress_k = nn.Linear(self.head_dim * self.nsa_block_size, self.head_dim, bias=False)
        self.compress_v = nn.Linear(self.head_dim * self.nsa_block_size, self.head_dim, bias=False)
        self.gate = nn.Linear(spec.n_embd, 3 * spec.n_head, bias=False)

        self.rope = (
            RotaryEmbedding(self.head_dim, max_seq_len=spec.block_size,
                            scaling_type=spec.rope_scaling_type, scaling_factor=spec.rope_scaling_factor)
            if spec.use_rope else None
        )
        self.attn_dropout = nn.Dropout(spec.dropout)
        self.resid_dropout = nn.Dropout(spec.dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _compress_kv(self, k, v):
        B, nh, T, hd = k.shape
        bs = self.nsa_block_size
        pad_len = (bs - T % bs) % bs
        if pad_len > 0:
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
        n_blocks = k.shape[2] // bs
        k_blocks = k.reshape(B, nh, n_blocks, bs * hd)
        v_blocks = v.reshape(B, nh, n_blocks, bs * hd)
        return self.compress_k(k_blocks), self.compress_v(v_blocks)

    def _select_topk(self, q, k, v):
        B, nh, T, hd = q.shape
        topk = min(self.nsa_top_k, k.shape[2])
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal = torch.triu(torch.ones(T, k.shape[2], dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        topk_scores, topk_idx = scores.topk(topk, dim=-1)
        idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, -1, hd)
        k_sel = torch.gather(k.unsqueeze(2).expand(-1, -1, T, -1, -1), 3, idx)
        v_sel = torch.gather(v.unsqueeze(2).expand(-1, -1, T, -1, -1), 3, idx)
        attn = self.attn_dropout(F.softmax(topk_scores, dim=-1))
        return torch.matmul(attn.unsqueeze(-2), v_sel).squeeze(-2)

    def _sliding_window(self, q, k, v):
        B, nh, T, hd = q.shape
        w = self.nsa_window_size
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        q_pos = torch.arange(T, device=q.device).unsqueeze(1)
        k_pos = torch.arange(k.shape[2], device=q.device).unsqueeze(0)
        mask = (k_pos <= q_pos) & (q_pos - k_pos < w)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        return torch.matmul(attn, v)

    def forward(self, x: torch.Tensor, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            cos, sin = self.rope(T)
            q, k = apply_rotary_pos_emb_pair(q, k, cos.to(x.device), sin.to(x.device))

        present_kv = (k, v)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        k_comp, v_comp = self._compress_kv(k, v)
        comp_scores = torch.matmul(q, k_comp.transpose(-2, -1)) * self.scale
        n_comp = k_comp.shape[2]
        comp_mask = torch.triu(torch.ones(T, n_comp, dtype=torch.bool, device=x.device), diagonal=max(1, n_comp - T + 1))
        comp_scores = comp_scores.masked_fill(comp_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        out_compressed = torch.matmul(F.softmax(comp_scores, dim=-1), v_comp)

        out_selected = self._select_topk(q, k, v)
        out_window = self._sliding_window(q, k, v)

        gates = F.softmax(self.gate(x).view(B, T, 3, self.n_head), dim=2).permute(0, 3, 1, 2)
        y = (
            gates[..., 0].unsqueeze(-1) * out_compressed
            + gates[..., 1].unsqueeze(-1) * out_selected
            + gates[..., 2].unsqueeze(-1) * out_window
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y)), present_kv
