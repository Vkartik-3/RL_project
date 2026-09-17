"""Supervised fine-tuning on (prompt, response) pairs with prompt-token masking.

Works for full fine-tuning and for LoRA/QLoRA adapters (the policy decides
which parameters are trainable). ``loss_reduction="mean"`` is the standard
token-mean cross-entropy; this is also the objective used by the STaR loop.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from forgeline.core.errors import ConfigError
from forgeline.data.collators import IGNORE_INDEX, collate_supervised
from forgeline.data.supervised import SupervisedDataset
from forgeline.models.policy import NativePolicy
from forgeline.training.common.trainer import PostTrainingAlgorithm


@dataclass
class SFTConfig:
    batch_size: int = 8
    max_length: int = 256
    mask_prompt: bool = True
    seed: int = 0


class SFTAlgorithm(PostTrainingAlgorithm):
    name = "sft"

    def __init__(self, policy: NativePolicy, dataset: SupervisedDataset, config: SFTConfig,
                 eval_dataset: Optional[SupervisedDataset] = None):
        if len(dataset) == 0:
            raise ConfigError("SFT dataset is empty")
        self.policy = policy
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.cfg = config
        self.rng = random.Random(config.seed)
        self._epoch_order: List[int] = []

    def parameters(self) -> List[nn.Parameter]:
        return self.policy.trainable_parameters()

    def modules(self):
        return (self.policy.model,)

    def _next_indices(self, n: int) -> List[int]:
        out = []
        while len(out) < n:
            if not self._epoch_order:
                self._epoch_order = list(range(len(self.dataset)))
                self.rng.shuffle(self._epoch_order)
            out.append(self._epoch_order.pop())
        return out

    def collect(self, step: int):
        examples = [self.dataset[i] for i in self._next_indices(self.cfg.batch_size)]
        return collate_supervised(examples, pad_value=self.policy.pad_token_id,
                                  max_length=min(self.cfg.max_length, self.policy.max_length + 1),
                                  mask_prompt=self.cfg.mask_prompt)

    def _ce(self, batch) -> torch.Tensor:
        logits = self.policy.token_logits(batch.input_ids)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), batch.targets.to(self.policy.device).reshape(-1),
                               ignore_index=IGNORE_INDEX)

    def loss(self, batch, micro_step: int, n_micro: int):
        if n_micro > 1:
            from forgeline.data.collators import SupervisedBatch

            sl = slice(micro_step, None, n_micro)
            batch = SupervisedBatch(batch.input_ids[sl], batch.targets[sl], batch.attention_mask[sl], batch.prompt_lengths[sl])
        loss = self._ce(batch)
        n_tokens = int((batch.targets != IGNORE_INDEX).sum())
        return loss, {"response_tokens": float(n_tokens)}

    @torch.no_grad()
    def evaluate(self, step: int) -> Dict[str, float]:
        ds = self.eval_dataset or self.dataset
        n = min(len(ds), 4 * self.cfg.batch_size)
        losses = []
        for i in range(0, n, self.cfg.batch_size):
            batch = collate_supervised([ds[j] for j in range(i, min(i + self.cfg.batch_size, n))],
                                       pad_value=self.policy.pad_token_id,
                                       max_length=min(self.cfg.max_length, self.policy.max_length + 1),
                                       mask_prompt=self.cfg.mask_prompt)
            losses.append(float(self._ce(batch)))
        return {"loss": sum(losses) / max(len(losses), 1)}

    def state(self) -> Dict[str, Any]:
        return self.policy.state_for_checkpoint()

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.policy.load_checkpoint_state(state)

    def model_spec(self) -> Dict[str, Any]:
        return self.policy.spec.to_dict()

    def tokenizer_meta(self):
        return self.policy.tokenizer.metadata()
