"""Reward-model training with the Bradley-Terry pairwise objective."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn

from forgeline.core.errors import ConfigError
from forgeline.data.collators import pad_sequences
from forgeline.data.preference import PreferenceDataset
from forgeline.models.reward import SequenceRewardModel, bradley_terry_loss
from forgeline.training.common.trainer import PostTrainingAlgorithm


@dataclass
class RewardModelConfig:
    batch_size: int = 8
    freeze_backbone: bool = False
    max_length: int = 256
    seed: int = 0


class RewardModelAlgorithm(PostTrainingAlgorithm):
    name = "reward_model"

    def __init__(self, model: SequenceRewardModel, dataset: PreferenceDataset, config: RewardModelConfig,
                 eval_dataset: Optional[PreferenceDataset] = None, pad_token_id: int = 0,
                 tokenizer_meta: Optional[Dict[str, Any]] = None):
        if len(dataset) == 0:
            raise ConfigError("reward-model dataset is empty")
        self.model, self.dataset, self.eval_dataset, self.cfg = model, dataset, eval_dataset, config
        self.pad = pad_token_id
        self._tok_meta = tokenizer_meta
        if config.freeze_backbone:
            model.freeze_backbone()
        self.rng = random.Random(config.seed)

    def parameters(self) -> List[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def modules(self):
        return (self.model,)

    def collect(self, step: int):
        return [self.dataset[self.rng.randrange(len(self.dataset))] for _ in range(self.cfg.batch_size)]

    def _scores(self, examples, key: str) -> torch.Tensor:
        seqs = [(e["prompt"] + e[key])[-self.cfg.max_length:] for e in examples]
        ids, mask = pad_sequences(seqs, self.pad, side="right")
        dev = next(self.model.parameters()).device
        return self.model(ids.to(dev), mask.to(dev))

    def loss(self, batch, micro_step: int, n_micro: int):
        chunk = batch[micro_step::n_micro] if n_micro > 1 else batch
        c, r = self._scores(chunk, "chosen"), self._scores(chunk, "rejected")
        loss = bradley_terry_loss(c, r)
        return loss, {"accuracy": float((c > r).float().mean()), "margin": float((c - r).detach().mean())}

    @torch.no_grad()
    def evaluate(self, step: int) -> Dict[str, float]:
        ds = self.eval_dataset or self.dataset
        examples = [ds[i] for i in range(min(len(ds), 32))]
        c, r = self._scores(examples, "chosen"), self._scores(examples, "rejected")
        return {"loss": float(bradley_terry_loss(c, r)), "accuracy": float((c > r).float().mean())}

    def state(self) -> Dict[str, Any]:
        return self.model.state_for_checkpoint()

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.model.load_checkpoint_state(state)

    def model_spec(self) -> Dict[str, Any]:
        return self.model.spec.to_dict()

    def tokenizer_meta(self):
        return self._tok_meta
