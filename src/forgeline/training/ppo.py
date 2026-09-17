"""Proximal Policy Optimization for language-model policies.

One implementation, three benchmark-preserving switches:

* ``ratio_level``: ``sequence_mean`` — the probability ratio is
  ``exp(mean_t(log π_new − log π_old))`` per sequence (the configuration
  recorded under benchmarks/post_training/ppo); ``token`` — per-token ratios
  with token-level clipping (standard PPO-for-LLMs).
* ``kl_mode``: ``monitor`` — KL is logged only (the value-head PPO runs that
  produced the recorded numbers used a detached KL term); ``penalty`` — KL is
  added to the loss with weight ``kl_coef``; ``reward`` — KL is subtracted from
  the reward before advantage estimation.
* ``entropy_mode``: ``sampled_logprob`` — entropy proxy ``−mean(log π(y_t))``
  over the sampled tokens; ``full`` — exact entropy of the full distribution.

Advantages are ``reward − value`` (normalised across the batch) with a value
MSE loss, or GAE when ``use_gae`` is set with per-token rewards.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from forgeline.core.errors import ConfigError
from forgeline.core.protocols import RewardProvider, Trajectory
from forgeline.models.policy import NativePolicy
from forgeline.rollouts.engine import RolloutConfig, RolloutEngine, stack_group, stack_old_logprobs
from forgeline.rollouts.rewards.composite import score_trajectories
from forgeline.training.common.advantages import normalize_advantages
from forgeline.training.common.logprobs import entropy_from_logits, kl_logprob_diff, masked_mean
from forgeline.training.common.trainer import PostTrainingAlgorithm


@dataclass
class PPOConfig:
    prompts_per_step: int = 4
    samples_per_prompt: int = 1
    ppo_epochs: int = 3
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.1
    ratio_level: str = "sequence_mean"  # sequence_mean | token
    kl_mode: str = "monitor"  # monitor | penalty | reward
    entropy_mode: str = "sampled_logprob"  # sampled_logprob | full
    normalize_advantages: bool = True
    seed: int = 0

    def validate(self) -> None:
        if self.ratio_level not in ("sequence_mean", "token"):
            raise ConfigError("ratio_level must be sequence_mean|token")
        if self.kl_mode not in ("monitor", "penalty", "reward"):
            raise ConfigError("kl_mode must be monitor|penalty|reward")
        if self.entropy_mode not in ("sampled_logprob", "full"):
            raise ConfigError("entropy_mode must be sampled_logprob|full")
        if not 0 < self.clip_ratio < 1:
            raise ConfigError("clip_ratio must be in (0, 1)")
        if self.ppo_epochs < 1 or self.prompts_per_step < 1 or self.samples_per_prompt < 1:
            raise ConfigError("ppo_epochs, prompts_per_step and samples_per_prompt must be >= 1")


def ppo_clipped_objective(ratio: torch.Tensor, advantages: torch.Tensor, clip_ratio: float) -> torch.Tensor:
    """``−min(r·A, clip(r, 1−ε, 1+ε)·A)`` elementwise (not reduced)."""
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    return -torch.min(surr1, surr2)


@dataclass
class PPOBatch:
    prompt_ids: torch.Tensor  # [N, P]
    response_ids: torch.Tensor  # [N, R]
    mask: torch.Tensor  # [N, R]
    old_logprobs: torch.Tensor  # [N, R]
    ref_logprobs: torch.Tensor  # [N, R]
    old_values: torch.Tensor  # [N]
    rewards: torch.Tensor  # [N]
    trajectories: List[Trajectory]


class PPOAlgorithm(PostTrainingAlgorithm):
    name = "ppo"

    def __init__(self, policy: NativePolicy, prompts: Sequence[torch.Tensor], reward: RewardProvider,
                 config: PPOConfig, rollout: Optional[RolloutConfig] = None,
                 tasks: Optional[Sequence[Dict[str, Any]]] = None):
        config.validate()
        if policy.value_head is None:
            raise ConfigError("PPO requires a policy with value_head=True")
        self.policy = policy
        if not policy.has_adapter and policy.reference_model is None:
            policy.with_frozen_reference()
        self.prompts = list(prompts)
        self.tasks = list(tasks) if tasks else [{} for _ in self.prompts]
        self.reward = reward
        self.cfg = config
        self.rollouts = RolloutEngine(policy, rollout or RolloutConfig(group_size=config.samples_per_prompt))
        self.rng = random.Random(config.seed)
        self._last_stats: Dict[str, float] = {}

    def parameters(self) -> List[nn.Parameter]:
        return self.policy.trainable_parameters()

    def modules(self):
        return (self.policy.model, self.policy.value_head)

    # ── rollout / collection ─────────────────────────────────────────────
    @torch.no_grad()
    def collect(self, step: int) -> PPOBatch:
        idx = [self.rng.randrange(len(self.prompts)) for _ in range(self.cfg.prompts_per_step)]
        groups = [self.rollouts.rollout(self.prompts[i], self.tasks[i], group_size=self.cfg.samples_per_prompt) for i in idx]
        trajectories = [t for g in groups for t in g]
        rewards = torch.tensor(score_trajectories(self.reward, trajectories), dtype=torch.float32)
        # pad everything into one batch (prompts may differ in length → left pad)
        P = max(t.prompt_ids.numel() for t in trajectories)
        R = max(t.response_length for t in trajectories)
        N = len(trajectories)
        pad = self.policy.pad_token_id
        prompt_ids = torch.full((N, P), pad, dtype=torch.long)
        response_ids = torch.full((N, R), pad, dtype=torch.long)
        mask = torch.zeros((N, R))
        for i, t in enumerate(trajectories):
            prompt_ids[i, P - t.prompt_ids.numel():] = t.prompt_ids
            response_ids[i, : t.response_length] = t.response_ids
            mask[i, : t.response_length] = 1.0
        old_lp, old_values = self.policy.logprobs_and_values(prompt_ids, response_ids)
        with self.policy.reference():
            ref_lp = self.policy.logprobs(prompt_ids, response_ids)
        if self.cfg.kl_mode == "reward":
            kl_seq = kl_logprob_diff(ref_lp, old_lp, mask, clamp_min=None).cpu()
            rewards = rewards - self.cfg.kl_coef * kl_seq
        self._last_stats = {"reward_mean": float(rewards.mean()), "reward_std": float(rewards.std()) if N > 1 else 0.0,
                            "response_len": float(mask.sum(1).mean())}
        return PPOBatch(prompt_ids, response_ids, mask, old_lp.detach().cpu(), ref_lp.detach().cpu(),
                        old_values.detach().cpu(), rewards, trajectories)

    # ── objective ────────────────────────────────────────────────────────
    def loss(self, batch: PPOBatch, micro_step: int, n_micro: int):
        dev = self.policy.device
        mask = batch.mask.to(dev)
        returns = batch.rewards.to(dev)
        advantages = returns - batch.old_values.to(dev)
        if self.cfg.normalize_advantages:
            advantages = normalize_advantages(advantages)
        advantages = advantages.detach()

        total = torch.zeros((), device=dev)
        agg: Dict[str, float] = {}
        for _ in range(self.cfg.ppo_epochs):
            new_lp, values = self.policy.logprobs_and_values(batch.prompt_ids, batch.response_ids)
            old_lp = batch.old_logprobs.to(dev)
            ref_lp = batch.ref_logprobs.to(dev)
            if self.cfg.ratio_level == "sequence_mean":
                log_ratio = masked_mean(new_lp - old_lp, mask)  # [N]
                policy_loss = ppo_clipped_objective(torch.exp(log_ratio), advantages, self.cfg.clip_ratio).mean()
            else:
                ratio = torch.exp(new_lp - old_lp)  # [N, R]
                per_token = ppo_clipped_objective(ratio, advantages.unsqueeze(1), self.cfg.clip_ratio)
                policy_loss = masked_mean(per_token, mask).mean()
            value_loss = F.mse_loss(values, returns.detach())
            if self.cfg.kl_mode == "penalty":
                kl = kl_logprob_diff(ref_lp, new_lp, mask).mean()
            else:  # monitor / reward: detached mean(log π_old − log π_ref), logged only (recorded configuration)
                kl = masked_mean(old_lp - ref_lp, mask).mean().detach()
            if self.cfg.entropy_mode == "full":
                logits = self.policy.full_sequence_logits(batch.prompt_ids, batch.response_ids)[:, batch.prompt_ids.shape[1] - 1 : -1, :]
                entropy = masked_mean(entropy_from_logits(logits), mask).mean()
            else:
                entropy = -masked_mean(new_lp, mask).mean()
            loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
            if self.cfg.kl_mode == "penalty":
                loss = loss + self.cfg.kl_coef * kl
            total = total + loss
            for k, v in {"policy_loss": policy_loss, "value_loss": value_loss, "kl": kl, "entropy": entropy}.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach()) / self.cfg.ppo_epochs
        agg.update(self._last_stats)
        agg["advantage_mean"] = float(advantages.mean())
        return total / self.cfg.ppo_epochs, agg

    def state(self) -> Dict[str, Any]:
        return self.policy.state_for_checkpoint()

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.policy.load_checkpoint_state(state)

    def model_spec(self) -> Dict[str, Any]:
        return self.policy.spec.to_dict()

    def tokenizer_meta(self):
        return self.policy.tokenizer.metadata()
