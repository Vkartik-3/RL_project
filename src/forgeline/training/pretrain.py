"""Language-model pretraining on a memory-mapped or streaming token corpus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from forgeline.core.errors import ConfigError
from forgeline.data.pretraining import MemmapCorpus
from forgeline.models.transformer.model import TransformerLM
from forgeline.training.common.trainer import PostTrainingAlgorithm


@dataclass
class PretrainConfig:
    batch_size: int = 16
    eval_batches: int = 20
    seed: int = 0


class PretrainAlgorithm(PostTrainingAlgorithm):
    name = "pretrain"

    def __init__(self, model: TransformerLM, train_corpus: MemmapCorpus | Iterator[Tuple[torch.Tensor, torch.Tensor]],
                 config: PretrainConfig, val_corpus: Optional[MemmapCorpus] = None, device: torch.device | str = "cpu",
                 tokenizer_meta: Optional[Dict[str, Any]] = None):
        self.model = model
        self.train = train_corpus
        self.val = val_corpus
        self.cfg = config
        self.device = torch.device(device)
        self._tokenizer_meta = tokenizer_meta
        self.gen = torch.Generator().manual_seed(config.seed)
        self.tokens_seen = 0
        self.ema_loss: Optional[float] = None
        self.spec = model.spec
        if isinstance(train_corpus, MemmapCorpus) and train_corpus.block_size != model.spec.block_size:
            raise ConfigError(f"corpus block_size {train_corpus.block_size} != model block_size {model.spec.block_size}")

    def parameters(self) -> List[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def modules(self):
        return (self.model,)

    def collect(self, step: int):
        if isinstance(self.train, MemmapCorpus):
            return self.train.sample_batch(self.cfg.batch_size, self.device, self.gen)
        batch = next(self.train)  # streaming loader yields (x, y) batches
        x, y = batch
        return x.to(self.device), y.to(self.device)

    def loss(self, batch, micro_step: int, n_micro: int):
        x, y = batch
        if n_micro > 1:
            x, y = x[micro_step::n_micro], y[micro_step::n_micro]
        _, loss = self.model(x, y)
        self.tokens_seen += int(x.numel())
        val = float(loss.detach())
        self.ema_loss = val if self.ema_loss is None else 0.95 * self.ema_loss + 0.05 * val
        return loss, {"ema_loss": self.ema_loss, "tokens_seen": float(self.tokens_seen)}

    @torch.no_grad()
    def evaluate(self, step: int) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, corpus in (("train", self.train if isinstance(self.train, MemmapCorpus) else None), ("val", self.val)):
            if corpus is None:
                continue
            losses = []
            for _ in range(self.cfg.eval_batches):
                x, y = corpus.sample_batch(self.cfg.batch_size, self.device, self.gen)
                _, l = self.model(x, y)
                losses.append(float(l))
            out[f"{name}_loss"] = sum(losses) / len(losses)
        if "val_loss" in out:
            out["loss"] = out["val_loss"]
        return out

    def state(self) -> Dict[str, Any]:
        return {"model": self.model.state_dict()}

    def load_state(self, state: Mapping[str, Any]) -> None:
        from forgeline.checkpoints.manager import load_model_state

        load_model_state(self.model, state["model"])

    def model_spec(self) -> Dict[str, Any]:
        return getattr(self.model, "module", self.model).spec.to_dict()

    def tokenizer_meta(self):
        return self._tokenizer_meta
