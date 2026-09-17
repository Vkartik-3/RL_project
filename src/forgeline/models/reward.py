"""Sequence reward model: native transformer backbone + scalar head."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from forgeline.core.config import ModelSpec
from forgeline.models.transformer.model import TransformerLM


class SequenceRewardModel(nn.Module):
    """Scores a token sequence with a scalar; pooling uses the last position."""

    def __init__(self, spec: ModelSpec, backbone: Optional[TransformerLM] = None, pooling: str = "last"):
        super().__init__()
        if pooling not in ("last", "mean"):
            raise ValueError("pooling must be 'last' or 'mean'")
        self.spec = spec
        self.pooling = pooling
        self.backbone = backbone or TransformerLM(spec)
        self.reward_head = nn.Linear(spec.n_embd, 1, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.backbone.forward_hidden(input_ids)
        if self.pooling == "mean":
            if attention_mask is None:
                pooled = h.mean(dim=1)
            else:
                m = attention_mask.unsqueeze(-1).to(h.dtype)
                pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            if attention_mask is None:
                pooled = h[:, -1, :]
            else:
                last = attention_mask.long().sum(dim=1).clamp(min=1) - 1
                pooled = h[torch.arange(h.size(0), device=h.device), last]
        return self.reward_head(pooled).squeeze(-1)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def state_for_checkpoint(self) -> Dict[str, Any]:
        return {"model": self.state_dict(), "pooling": self.pooling}

    def load_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        self.load_state_dict(state["model"])


def bradley_terry_loss(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> torch.Tensor:
    """``-log σ(r_chosen − r_rejected)`` averaged over the batch."""
    return -F.logsigmoid(chosen_scores - rejected_scores).mean()


class EncoderRewardModel(nn.Module):
    """Transformer-encoder reward model on text (``huggingface`` extra).

    ``objective="regression"``: sigmoid head fit with MSE to scalar targets (e.g. rule-reward pseudo-labels).
    ``objective="preference"``: linear head fit with the Bradley-Terry loss on (chosen, rejected) texts.
    Architecture: CLS state → Dropout → Linear(h, h/2) → GELU → [Dropout] → Linear(h/2, 1) [→ Sigmoid].
    """

    def __init__(self, backbone: str = "bert-base-uncased", objective: str = "regression", dropout: float = 0.1, max_length: int = 128):
        super().__init__()
        from forgeline.core.errors import ConfigError, OptionalDependencyError

        if objective not in ("regression", "preference"):
            raise ConfigError("objective must be regression|preference")
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise OptionalDependencyError("transformers is not installed", hint="pip install 'forgeline[huggingface]'") from exc
        self.objective, self.max_length = objective, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.encoder = AutoModel.from_pretrained(backbone)
        h = self.encoder.config.hidden_size
        layers = [nn.Dropout(dropout), nn.Linear(h, h // 2), nn.GELU()]
        if objective == "regression":
            layers += [nn.Dropout(dropout), nn.Linear(h // 2, 1), nn.Sigmoid()]
        else:
            layers += [nn.Linear(h // 2, 1)]
        self.head = nn.Sequential(*layers)
        self.trained = False

    def _encode(self, texts):
        enc = self.tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        dev = next(self.parameters()).device
        return {k: v.to(dev) for k, v in enc.items()}

    def forward(self, texts) -> torch.Tensor:
        enc = self._encode(texts)
        out = self.encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        return self.head(out.last_hidden_state[:, 0, :]).squeeze(-1)

    @torch.no_grad()
    def score_texts(self, texts, probability: bool = True) -> torch.Tensor:
        self.eval()
        s = self(texts)
        return torch.sigmoid(s) if (probability and self.objective == "preference") else s

    def fit_regression(self, texts, targets, epochs: int = 3, lr: float = 1e-4, batch_size: int = 16) -> list:
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        y = torch.tensor(list(targets), dtype=torch.float32)
        losses = []
        self.train()
        for _ in range(epochs):
            total, n = 0.0, 0
            for i in range(0, len(texts), batch_size):
                pred = self(texts[i : i + batch_size])
                loss = F.mse_loss(pred, y[i : i + batch_size].to(pred.device))
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss.detach()); n += 1
            losses.append(total / max(n, 1))
        self.trained = True
        return losses

    def fit_preference(self, pairs, epochs: int = 3, lr: float = 2e-5, batch_size: int = 16, seed: int = 0) -> list:
        import random

        opt = torch.optim.AdamW(self.parameters(), lr=lr)
        rng = random.Random(seed)
        losses = []
        self.train()
        for _ in range(epochs):
            order = list(range(len(pairs)))
            rng.shuffle(order)
            total, n = 0.0, 0
            for i in range(0, len(order), batch_size):
                idx = order[i : i + batch_size]
                loss = bradley_terry_loss(self([pairs[j][0] for j in idx]), self([pairs[j][1] for j in idx]))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()
                total += float(loss.detach()); n += 1
            losses.append(total / max(n, 1))
        self.trained = True
        return losses
