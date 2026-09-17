"""Transformer block and multi-token-prediction module."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from forgeline.core.config import ModelSpec
from forgeline.models.attention.causal import CausalSelfAttention
from forgeline.models.attention.latent import MultiHeadLatentAttention
from forgeline.models.attention.sparse import NativeSparseAttention
from forgeline.models.moe.layer import MoELayer
from forgeline.models.transformer.feedforward import GELUFeedForward, SwiGLUFeedForward
from forgeline.models.transformer.norm import RMSNorm


def build_attention(spec: ModelSpec, layer_id: int = 0) -> nn.Module:
    if spec.use_mla:
        return MultiHeadLatentAttention(spec)
    if spec.use_nsa:
        return NativeSparseAttention(spec)
    return CausalSelfAttention(spec, layer_id=layer_id)


def build_dense_ffn(spec: ModelSpec) -> nn.Module:
    if spec.use_swiglu:
        return SwiGLUFeedForward(spec.n_embd, bias=spec.bias, dropout=spec.dropout)
    return GELUFeedForward(spec.n_embd, bias=spec.bias, dropout=spec.dropout)


class TransformerBlock(nn.Module):
    """Pre-norm block: attention (GQA | MLA | NSA) + FFN (dense | MoE)."""

    def __init__(self, layer_id: int, spec: ModelSpec):
        super().__init__()
        self.ln_1 = RMSNorm(spec.n_embd)
        self.ln_2 = RMSNorm(spec.n_embd)
        self.attn = build_attention(spec, layer_id)
        self.use_moe = bool(spec.use_moe and layer_id >= spec.n_dense_layers)
        self.ffn = MoELayer(spec) if self.use_moe else build_dense_ffn(spec)

    def forward(self, x: torch.Tensor, past_kv: Optional[Any] = None):
        attn_out, present_kv = self.attn(self.ln_1(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.ffn(self.ln_2(x))
        return x, present_kv


class MTPModule(nn.Module):
    """One additional prediction depth for multi-token prediction (training only)."""

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.proj = nn.Linear(spec.n_embd, spec.n_embd, bias=False)
        self.norm = RMSNorm(spec.n_embd)
        self.attn_norm = RMSNorm(spec.n_embd)
        self.attn = MultiHeadLatentAttention(spec) if spec.use_mla else CausalSelfAttention(spec)
        self.ffn_norm = RMSNorm(spec.n_embd)
        self.ffn = build_dense_ffn(spec)

    def forward(self, h: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        combined = self.norm(self.proj(h) + target_emb)
        attn_out, _ = self.attn(self.attn_norm(combined))
        combined = combined + attn_out
        return combined + self.ffn(self.ffn_norm(combined))
