"""Model inspection: weight statistics, layer summary, activation flow."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from forgeline.core.config import ModelSpec


def weight_statistics(model: torch.nn.Module, bins: int = 20) -> List[Dict[str, Any]]:
    stats = []
    for name, p in model.named_parameters():
        d = p.detach().float()
        hist = torch.histc(d.flatten().cpu(), bins=bins)
        lo, hi = float(d.min()), float(d.max())
        stats.append({"name": name, "shape": list(p.shape), "numel": p.numel(), "mean": float(d.mean()), "std": float(d.std()) if d.numel() > 1 else 0.0,
                      "min": lo, "max": hi, "abs_mean": float(d.abs().mean()), "norm": float(d.norm()),
                      "histogram": {"counts": hist.tolist(), "edges": [lo + i * (hi - lo) / bins for i in range(bins + 1)]}})
    return stats


def layer_summary(spec: ModelSpec) -> List[Dict[str, Any]]:
    layers: List[Dict[str, Any]] = [{"type": "embedding", "name": "token_embedding", "params": spec.vocab_size * spec.n_embd}]
    for i in range(spec.n_layer):
        attn = "MLA" if spec.use_mla else "NSA" if spec.use_nsa else ("GQA" if spec.n_kv_head < spec.n_head else "MHA")
        comp = [{"type": "attention", "name": attn, "n_heads": spec.n_head, "n_kv_heads": spec.n_kv_head}]
        if spec.use_moe and i >= spec.n_dense_layers:
            comp.append({"type": "moe", "n_experts": spec.n_experts, "n_active": spec.n_experts_active, "n_shared": spec.n_shared_experts})
        else:
            comp.append({"type": "ffn", "name": "SwiGLU" if spec.use_swiglu else "GELU"})
        layers.append({"type": "block", "layer_id": i, "components": comp})
    layers.append({"type": "output", "name": "lm_head", "params": spec.vocab_size * spec.n_embd, "tied": True})
    return layers


@torch.no_grad()
def activation_flow(model, ids: torch.Tensor) -> List[Dict[str, float]]:
    """Mean/std/max-abs/norm of the residual stream after the embedding and each block."""
    records: List[Dict[str, float]] = []

    def hook(name):
        def fn(m, i, o):
            out = o[0] if isinstance(o, tuple) else o
            records.append({"name": name, "mean": float(out.float().mean()), "std": float(out.float().std()),
                            "max_abs": float(out.float().abs().max()), "norm": float(out.float().norm())})
        return fn

    handles = [model.transformer.wte.register_forward_hook(hook("embedding"))]
    handles += [b.register_forward_hook(hook(f"block_{i}")) for i, b in enumerate(model.transformer.h)]
    handles.append(model.transformer.ln_f.register_forward_hook(hook("final_norm")))
    try:
        model(ids)
    finally:
        for h in handles:
            h.remove()
    return records


@torch.no_grad()
def attention_patterns(model, ids: torch.Tensor, max_tokens: int = 48, per_head: bool = False) -> List[Dict[str, Any]]:
    """Per-layer attention weights (head-averaged ``[T, T]``) and per-head mean entropy for a ``[1, T]`` prompt.

    Weights are recomputed from each grouped-query attention layer's own projections, rotary embedding, logit cap and
    causal / sliding-window mask, so they match the forward pass even when it runs through fused SDPA. Latent (MLA) and
    native-sparse (NSA) layers are reported as unsupported. ``per_head=True`` adds ``head_weights`` ``[H, T, T]``.
    """
    import math

    from forgeline.models.attention.causal import CausalSelfAttention
    from forgeline.models.attention.rotary import apply_rotary_pos_emb_pair

    ids = ids[:, -max_tokens:]
    captured: Dict[int, torch.Tensor] = {}
    blocks = list(model.transformer.h)
    handles = []
    for i, block in enumerate(blocks):
        if isinstance(block.attn, CausalSelfAttention):
            handles.append(block.attn.register_forward_pre_hook(
                lambda m, args, kwargs, i=i: captured.__setitem__(i, args[0].detach()), with_kwargs=True))
    try:
        model(ids)
    finally:
        for h in handles:
            h.remove()
    out: List[Dict[str, Any]] = []
    for i, block in enumerate(blocks):
        attn = block.attn
        if i not in captured:
            out.append({"layer": i, "type": type(attn).__name__, "supported": False})
            continue
        x = captured[i].float()
        B, T, _ = x.shape
        q = attn.q_proj(x).view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(B, T, attn.n_kv_head, attn.head_dim).transpose(1, 2)
        if attn.use_rope:
            cos, sin = attn.rotary_emb(T, offset=0)
            q, k = apply_rotary_pos_emb_pair(q, k, cos, sin)
        k = attn._repeat_kv(k)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
        if attn.attn_logit_cap > 0:
            att = torch.tanh(att / attn.attn_logit_cap) * attn.attn_logit_cap
        mask = attn._sliding_window_mask(T, T, x.device) if attn.window_size > 0 else \
            torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)).view(1, 1, T, T)
        att = torch.softmax(att.masked_fill(~mask, float("-inf")), dim=-1)[0]  # [H, T, T]
        entropy = -(att.clamp_min(1e-12).log() * att).sum(-1).mean(-1)  # [H]
        out.append({"layer": i, "type": "GQA" if attn.n_kv_head < attn.n_head else "MHA", "supported": True,
                    "window": attn.window_size, "weights": att.mean(0).tolist(), "head_entropy": entropy.tolist()})
        if per_head:
            out[-1]["head_weights"] = att.tolist()
    return out
