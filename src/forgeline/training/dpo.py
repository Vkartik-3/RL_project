"""Direct Preference Optimization.

    L = −log σ( β·(log π(y_w|x) − log π_ref(y_w|x)) − β·(log π(y_l|x) − log π_ref(y_l|x)) )

``logprob_reduction`` selects how token log-probs are reduced to a sequence
score: ``sum`` (standard DPO) or ``mean`` (length-normalised; the configuration
recorded under benchmarks/post_training/dpo). ``reference_free=True`` drops the
reference terms (used by the RLAIF self-training rounds).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from forgeline.core.errors import ConfigError
from forgeline.data.collators import pad_sequences
from forgeline.data.preference import PreferenceDataset
from forgeline.models.policy import NativePolicy
from forgeline.training.common.logprobs import reduce_logprobs
from forgeline.training.common.trainer import PostTrainingAlgorithm


@dataclass
class DPOConfig:
    beta: float = 0.1
    batch_size: int = 4
    logprob_reduction: str = "sum"  # sum | mean
    reference_free: bool = False
    label_smoothing: float = 0.0
    seed: int = 0

    def validate(self) -> None:
        if self.beta <= 0:
            raise ConfigError("DPO beta must be positive")
        if self.logprob_reduction not in ("sum", "mean"):
            raise ConfigError("logprob_reduction must be sum|mean")
        if not 0.0 <= self.label_smoothing < 0.5:
            raise ConfigError("label_smoothing must be in [0, 0.5)")


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor, ref_chosen: torch.Tensor,
             ref_rejected: torch.Tensor, beta: float, label_smoothing: float = 0.0) -> Dict[str, torch.Tensor]:
    """Batched DPO loss from sequence log-probs ``[B]``. Returns loss and implicit rewards."""
    chosen_reward = beta * (policy_chosen - ref_chosen)
    rejected_reward = beta * (policy_rejected - ref_rejected)
    logits = chosen_reward - rejected_reward
    if label_smoothing > 0:
        loss = -F.logsigmoid(logits) * (1 - label_smoothing) - F.logsigmoid(-logits) * label_smoothing
    else:
        loss = -F.logsigmoid(logits)
    return {
        "loss": loss.mean(),
        "chosen_reward": chosen_reward.detach(),
        "rejected_reward": rejected_reward.detach(),
        "margin": (chosen_reward - rejected_reward).detach(),
        "accuracy": (chosen_reward > rejected_reward).float().detach(),
    }


class DPOAlgorithm(PostTrainingAlgorithm):
    name = "dpo"

    def __init__(self, policy: NativePolicy, dataset: PreferenceDataset, config: DPOConfig,
                 eval_dataset: Optional[PreferenceDataset] = None):
        config.validate()
        self.policy = policy
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.cfg = config
        self.rng = random.Random(config.seed)
        if not config.reference_free and not policy.has_adapter and policy.reference_model is None:
            policy.with_frozen_reference()
        if len(dataset) == 0:
            raise ConfigError("DPO dataset is empty")

    def parameters(self) -> List[nn.Parameter]:
        return self.policy.trainable_parameters()

    def modules(self):
        return (self.policy.model,)

    def collect(self, step: int) -> List[Dict[str, List[int]]]:
        return [self.dataset[self.rng.randrange(len(self.dataset))] for _ in range(self.cfg.batch_size)]

    def _sequence_logprobs(self, examples: Sequence[Dict[str, List[int]]], key: str) -> torch.Tensor:
        """Sequence log-probs ``[B]`` of ``key`` responses given their prompts (left-padded prompts)."""
        out = []
        for ex in examples:  # prompts differ in length → per-example forward keeps semantics exact
            p = torch.tensor(ex["prompt"], dtype=torch.long).unsqueeze(0)
            r = torch.tensor(ex[key], dtype=torch.long).unsqueeze(0)
            lp = self.policy.logprobs(p, r)
            out.append(reduce_logprobs(lp, None, self.cfg.logprob_reduction).squeeze(0))
        return torch.stack(out)

    def loss(self, batch: List[Dict[str, List[int]]], micro_step: int, n_micro: int):
        chunk = batch[micro_step::n_micro] if n_micro > 1 else batch
        if not chunk:
            return torch.zeros((), device=self.policy.device, requires_grad=True), {}
        pc = self._sequence_logprobs(chunk, "chosen")
        pr = self._sequence_logprobs(chunk, "rejected")
        if self.cfg.reference_free:
            rc = torch.zeros_like(pc)
            rr = torch.zeros_like(pr)
        else:
            with torch.no_grad(), self.policy.reference():
                rc = self._sequence_logprobs(chunk, "chosen")
                rr = self._sequence_logprobs(chunk, "rejected")
        out = dpo_loss(pc, pr, rc, rr, self.cfg.beta, self.cfg.label_smoothing)
        metrics = {
            "reward_margin": float(out["margin"].mean()),
            "chosen_reward": float(out["chosen_reward"].mean()),
            "rejected_reward": float(out["rejected_reward"].mean()),
            "preference_accuracy": float(out["accuracy"].mean()),
        }
        return out["loss"], metrics

    @torch.no_grad()
    def evaluate(self, step: int) -> Dict[str, float]:
        ds = self.eval_dataset or self.dataset
        n = min(len(ds), 32)
        examples = [ds[i] for i in range(n)]
        pc = self._sequence_logprobs(examples, "chosen")
        pr = self._sequence_logprobs(examples, "rejected")
        if self.cfg.reference_free:
            rc, rr = torch.zeros_like(pc), torch.zeros_like(pr)
        else:
            with self.policy.reference():
                rc = self._sequence_logprobs(examples, "chosen")
                rr = self._sequence_logprobs(examples, "rejected")
        out = dpo_loss(pc, pr, rc, rr, self.cfg.beta)
        return {"loss": float(out["loss"]), "preference_accuracy": float(out["accuracy"].mean()),
                "reward_margin": float(out["margin"].mean())}

    def state(self) -> Dict[str, Any]:
        return self.policy.state_for_checkpoint()

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.policy.load_checkpoint_state(state)

    def model_spec(self) -> Dict[str, Any]:
        return self.policy.spec.to_dict()

    def tokenizer_meta(self):
        return self.policy.tokenizer.metadata()
