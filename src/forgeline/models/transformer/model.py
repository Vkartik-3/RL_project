"""The native decoder-only language model."""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from forgeline.core.config import ModelSpec
from forgeline.models.transformer.block import MTPModule, TransformerBlock
from forgeline.models.transformer.norm import RMSNorm


class TransformerLM(nn.Module):
    """Decoder-only transformer with tied embeddings, optional MoE/MLA/NSA/MTP.

    ``forward(idx, targets)`` returns ``(logits, loss)``; when ``targets`` is
    omitted only the last-position logits are returned (inference shortcut).
    ``forward_hidden`` exposes final hidden states for reward / value heads and
    ``step`` performs incremental decoding with a per-layer KV cache.
    """

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.aux_losses: List[torch.Tensor] = []
        self.gradient_checkpointing = False

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(spec.vocab_size, spec.n_embd),
                drop=nn.Dropout(spec.dropout),
                h=nn.ModuleList([TransformerBlock(i, spec) for i in range(spec.n_layer)]),
                ln_f=RMSNorm(spec.n_embd),
            )
        )
        self.wpe: Optional[nn.Embedding] = None
        if not spec.use_rope and not spec.use_mla:
            self.wpe = nn.Embedding(spec.block_size, spec.n_embd)
        self.lm_head = nn.Linear(spec.n_embd, spec.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.mtp_modules: Optional[nn.ModuleList] = None
        if spec.n_predict_tokens > 1:
            self.mtp_modules = nn.ModuleList([MTPModule(spec) for _ in range(spec.n_predict_tokens - 1)])

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("w2.weight") or pn.endswith("wo.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * spec.n_layer))
        self._last_hidden: Optional[torch.Tensor] = None

    # ── introspection ─────────────────────────────────────────────────────
    @property
    def config(self) -> ModelSpec:  # backwards-friendly alias used by tooling
        return self.spec

    def num_parameters(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.wpe is not None:
            n -= self.wpe.weight.numel()
        return n

    def num_active_parameters(self, non_embedding: bool = True) -> int:
        n = self.num_parameters(non_embedding)
        if not self.spec.use_moe:
            return n
        for block in self.transformer.h:
            if block.use_moe:
                expert_params = sum(p.numel() for p in block.ffn.experts[0].parameters())
                n -= expert_params * (self.spec.n_experts - self.spec.n_experts_active)
        return n

    def describe(self) -> str:
        s = self.spec
        feats = []
        if s.use_mla:
            feats.append(f"MLA(kv_rank={s.kv_lora_rank})")
        elif s.use_nsa:
            feats.append("NSA")
        elif s.n_kv_head < s.n_head:
            feats.append(f"GQA({s.n_head}Q/{s.n_kv_head}KV)")
        else:
            feats.append("MHA")
        if s.sliding_window > 0:
            feats.append(f"SlidingWindow({s.sliding_window},{'alternating' if s.alternating_layers else 'all'})")
        if s.attn_logit_cap > 0:
            feats.append(f"LogitCap({s.attn_logit_cap})")
        if s.rope_scaling_type != "none":
            feats.append(f"RoPE-{s.rope_scaling_type}({s.rope_scaling_factor}x)")
        if s.use_moe:
            feats.append(f"MoE({s.n_experts_active}/{s.n_experts}+{s.n_shared_experts}shared{',aux-free' if s.aux_loss_free else ''})")
        if s.n_predict_tokens > 1:
            feats.append(f"MTP({s.n_predict_tokens})")
        total = self.num_parameters() / 1e6
        if s.use_moe:
            return f"{total:.1f}M total / {self.num_active_parameters()/1e6:.1f}M active | " + " | ".join(feats)
        return f"{total:.1f}M params | " + " | ".join(feats)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    # ── forward passes ────────────────────────────────────────────────────
    def _embed(self, idx: torch.Tensor, offset: int = 0) -> torch.Tensor:
        tok_emb = self.transformer.wte(idx)
        if self.wpe is not None:
            pos = torch.arange(offset, offset + idx.size(1), dtype=torch.long, device=idx.device)
            return self.transformer.drop(tok_emb + self.wpe(pos))
        return self.transformer.drop(tok_emb)

    def forward_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        """Final normalised hidden states ``[B, T, n_embd]`` (collects MoE aux losses)."""
        B, T = idx.size()
        if T > self.spec.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.spec.block_size}")
        x = self._embed(idx)
        self.aux_losses = []
        for block in self.transformer.h:
            if self.gradient_checkpointing and self.training:
                x, _ = torch.utils.checkpoint.checkpoint(block, x, None, use_reentrant=False)
            else:
                x, _ = block(x)
            if block.use_moe:
                aux = block.ffn.aux_loss
                if aux is not None:
                    self.aux_losses.append(aux)
        h = self.transformer.ln_f(x)
        self._last_hidden = h.detach()
        return h

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        h = self.forward_hidden(idx)
        T = idx.size(1)
        if targets is None:
            return self.lm_head(h[:, [-1], :]), None

        logits = self.lm_head(h)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        if self.aux_losses:
            loss = loss + sum(self.aux_losses)
        if self.mtp_modules is not None and T > 1:
            mtp_total, n_mtp = 0.0, 0
            for d, mtp_mod in enumerate(self.mtp_modules):
                depth = d + 1
                if T - depth < 1:
                    break
                target_emb = self.transformer.wte(idx[:, depth:])
                h_mtp = mtp_mod(h[:, : T - depth, :], target_emb[:, : T - depth, :])
                mtp_logits = self.lm_head(h_mtp)
                mtp_total = mtp_total + F.cross_entropy(
                    mtp_logits.view(-1, mtp_logits.size(-1)),
                    targets[:, depth:][:, : T - depth].contiguous().view(-1),
                    ignore_index=-1,
                )
                n_mtp += 1
            if n_mtp > 0:
                loss = loss + self.spec.mtp_loss_weight * (mtp_total / n_mtp)
        return logits, loss

    def full_logits(self, idx: torch.Tensor) -> torch.Tensor:
        """Logits at every position ``[B, T, V]`` (used for log-prob computation)."""
        return self.lm_head(self.forward_hidden(idx))

    # ── incremental decoding ──────────────────────────────────────────────
    @torch.no_grad()
    def prefill(self, idx: torch.Tensor) -> Tuple[torch.Tensor, List[Any]]:
        """Run the prompt through the model; return last-position logits and the KV cache."""
        x = self._embed(idx)
        cache: List[Any] = []
        for block in self.transformer.h:
            x, present = block(x)
            cache.append(present)
        h = self.transformer.ln_f(x)
        return self.lm_head(h[:, -1, :]), cache

    @torch.no_grad()
    def step(self, next_ids: torch.Tensor, cache: List[Any], position: int) -> Tuple[torch.Tensor, List[Any]]:
        """Decode one token per sequence given the cache. ``next_ids``: ``[B, 1]``."""
        x = self._embed(next_ids, offset=position)
        new_cache: List[Any] = []
        for i, block in enumerate(self.transformer.h):
            x, present = block(x, past_kv=cache[i])
            new_cache.append(present)
        h = self.transformer.ln_f(x)
        return self.lm_head(h[:, -1, :]), new_cache

    # ── sampling-based generation (see models/generation/sampling.py) ─────
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None, top_p: Optional[float] = None, min_p: Optional[float] = None,
                 repetition_penalty: float = 1.0, use_cache: bool = True,
                 stop_token_ids: Tuple[int, ...] = ()) -> torch.Tensor:
        from forgeline.models.generation.sampling import generate_tokens

        return generate_tokens(
            self, idx, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
            min_p=min_p, repetition_penalty=repetition_penalty, use_cache=use_cache,
            stop_token_ids=stop_token_ids,
        )
