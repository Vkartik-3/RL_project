"""Speculative decoding: separate draft model, or self-drafting via MTP heads."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from forgeline.models.transformer.model import TransformerLM


def _topk_filter(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
    return logits


class SpeculativeGenerator:
    """Draft ``k`` tokens with a small model, verify with the target in one pass."""

    def __init__(self, target_model: TransformerLM, draft_model: TransformerLM, k: int = 5):
        self.target = target_model
        self.draft = draft_model
        self.k = k
        self.stats = {"proposed": 0, "accepted": 0}

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = 50):
        generated = 0
        temperature = max(temperature, 1e-8)
        while generated < max_new_tokens:
            draft_tokens, draft_probs = [], []
            draft_idx = idx.clone()
            for _ in range(self.k):
                ctx = draft_idx if draft_idx.size(1) <= self.draft.spec.block_size else draft_idx[:, -self.draft.spec.block_size:]
                logits, _ = self.draft(ctx)
                logits = _topk_filter(logits[:, -1, :] / temperature, top_k)
                probs = F.softmax(logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
                draft_tokens.append(token)
                draft_probs.append(probs)
                draft_idx = torch.cat([draft_idx, token], dim=1)

            verify_idx = torch.cat([idx] + draft_tokens, dim=1)
            if verify_idx.size(1) > self.target.spec.block_size:
                verify_idx = verify_idx[:, -self.target.spec.block_size:]
            target_logits = self.target.full_logits(verify_idx)

            n_accepted = 0
            prompt_len = idx.size(1)
            offset = verify_idx.size(1) - (prompt_len + self.k)  # 0 unless truncated
            for i in range(self.k):
                target_pos = prompt_len - 1 + i + offset
                if target_pos < 0 or target_pos >= target_logits.size(1):
                    break
                t_logit = _topk_filter(target_logits[:, target_pos, :] / temperature, top_k)
                target_prob = F.softmax(t_logit, dim=-1)
                d_token = draft_tokens[i]
                d_p = draft_probs[i].gather(1, d_token)
                t_p = target_prob.gather(1, d_token)
                ratio = (t_p / (d_p + 1e-10)).clamp(max=1.0)
                self.stats["proposed"] += 1
                if torch.rand(1, device=idx.device) < ratio:
                    n_accepted += 1
                    self.stats["accepted"] += 1
                    idx = torch.cat([idx, d_token], dim=1)
                    generated += 1
                    if generated >= max_new_tokens:
                        break
                else:
                    adjusted = torch.clamp(target_prob - draft_probs[i], min=0)
                    adjusted = adjusted / (adjusted.sum(dim=-1, keepdim=True) + 1e-10)
                    idx = torch.cat([idx, torch.multinomial(adjusted, num_samples=1)], dim=1)
                    generated += 1
                    break

            if n_accepted == self.k and generated < max_new_tokens:
                last_pos = prompt_len - 1 + self.k + offset
                if 0 <= last_pos < target_logits.size(1):
                    bonus = F.softmax(_topk_filter(target_logits[:, last_pos, :] / temperature, top_k), dim=-1)
                    idx = torch.cat([idx, torch.multinomial(bonus, num_samples=1)], dim=1)
                    generated += 1
            if idx.size(1) >= self.target.spec.block_size:
                break
        return idx

    @property
    def acceptance_rate(self) -> float:
        return self.stats["accepted"] / max(self.stats["proposed"], 1)


class MTPSpeculativeGenerator:
    """Self-speculative decoding using the model's own multi-token-prediction heads."""

    def __init__(self, model: TransformerLM):
        if model.mtp_modules is None or len(model.mtp_modules) == 0:
            raise ValueError("MTPSpeculativeGenerator requires a model with n_predict_tokens > 1")
        self.model = model
        self.n_drafts = len(model.mtp_modules)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 200, temperature: float = 1.0, top_k: Optional[int] = 50):
        self.model.eval()
        generated = 0
        temperature = max(temperature, 1e-8)
        block = self.model.spec.block_size
        while generated < max_new_tokens:
            ctx = idx if idx.size(1) <= block else idx[:, -block:]
            logits, _ = self.model(ctx)
            last_h = self.model._last_hidden
            main_probs = F.softmax(_topk_filter(logits[:, -1, :] / temperature, top_k), dim=-1)
            token_1 = torch.multinomial(main_probs, num_samples=1)
            draft_tokens, draft_probs = [token_1], [main_probs]

            prev_emb = self.model.transformer.wte(token_1)
            for mtp_mod in self.model.mtp_modules:
                h_mtp = mtp_mod(last_h[:, -1:, :], prev_emb)
                mtp_probs = F.softmax(_topk_filter(self.model.lm_head(h_mtp[:, -1, :]) / temperature, top_k), dim=-1)
                mtp_token = torch.multinomial(mtp_probs, num_samples=1)
                draft_tokens.append(mtp_token)
                draft_probs.append(mtp_probs)
                prev_emb = self.model.transformer.wte(mtp_token)

            idx = torch.cat([idx, token_1], dim=1)
            generated += 1
            if generated >= max_new_tokens:
                break

            if len(draft_tokens) > 1:
                verify_idx = torch.cat([idx, torch.cat(draft_tokens[1:], dim=1)], dim=1)
                if verify_idx.size(1) > block:
                    verify_idx = verify_idx[:, -block:]
                verify_logits = self.model.full_logits(verify_idx)
                prompt_len = idx.size(1)
                offset = verify_idx.size(1) - (prompt_len + len(draft_tokens) - 1)
                for i in range(len(draft_tokens) - 1):
                    if generated >= max_new_tokens:
                        break
                    pos = prompt_len - 1 + i + offset
                    if pos < 0 or pos >= verify_logits.size(1):
                        break
                    target_p = F.softmax(_topk_filter(verify_logits[:, pos, :] / temperature, top_k), dim=-1)
                    d_token = draft_tokens[i + 1]
                    ratio = (target_p.gather(1, d_token) / (draft_probs[i + 1].gather(1, d_token) + 1e-10)).clamp(max=1.0)
                    if torch.rand(1, device=idx.device) < ratio:
                        idx = torch.cat([idx, d_token], dim=1)
                        generated += 1
                    else:
                        adjusted = torch.clamp(target_p - draft_probs[i + 1], min=0)
                        adjusted = adjusted / (adjusted.sum(dim=-1, keepdim=True) + 1e-10)
                        idx = torch.cat([idx, torch.multinomial(adjusted, num_samples=1)], dim=1)
                        generated += 1
                        break
            if idx.size(1) >= block:
                break
        return idx
