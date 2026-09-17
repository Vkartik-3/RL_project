"""DAPO — Decoupled Clip and Dynamic Sampling Policy Optimization.

Relative to GRPO:

1. Clip-Higher: asymmetric ratio bounds ``[1 − ε_low, 1 + ε_high]``.
2. Dynamic sampling: groups whose rewards have zero variance carry no signal
   and are dropped from the update.
3. Token-level policy gradient: per-token surrogate terms are summed across
   every response in the batch and normalised by the total token count, so
   long responses are not under-weighted.
4. Overlong reward shaping: responses that hit the generation budget receive
   a soft penalty proportional to the overshoot.
5. Optional entropy bonus on the sampled tokens' full distribution.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from forgeline.core.errors import ConfigError
from forgeline.core.protocols import RewardProvider, Trajectory
from forgeline.models.policy import NativePolicy
from forgeline.rollouts.engine import RolloutConfig, RolloutEngine, stack_group
from forgeline.rollouts.rewards.composite import score_trajectories
from forgeline.training.common.advantages import group_relative_advantages
from forgeline.training.common.logprobs import entropy_from_logits
from forgeline.training.common.trainer import PostTrainingAlgorithm


@dataclass
class DAPOConfig:
    group_size: int = 4
    prompts_per_step: int = 2
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.28
    kl_coef: float = 0.04
    entropy_coef: float = 0.0
    dynamic_sampling: bool = True
    overlong_penalty: float = -1.0
    max_response_len: Optional[int] = None  # defaults to rollout max_new_tokens
    seed: int = 0

    def validate(self) -> None:
        if self.group_size < 2:
            raise ConfigError("DAPO group_size must be >= 2")
        if not (0 < self.clip_eps_low < 1 and 0 < self.clip_eps_high < 1):
            raise ConfigError("clip epsilons must be in (0, 1)")
        if self.clip_eps_high < self.clip_eps_low:
            raise ConfigError("clip_eps_high must be >= clip_eps_low (clip-higher)")


@dataclass
class DAPOGroup:
    trajectories: List[Trajectory]
    rewards: torch.Tensor
    advantages: torch.Tensor
    old_logprobs: torch.Tensor  # [G, R]
    ref_logprobs: torch.Tensor  # [G, R]
    mask: torch.Tensor  # [G, R]
    skipped: bool = False


def dapo_token_objective(ratio: torch.Tensor, adv: torch.Tensor, eps_low: float, eps_high: float) -> torch.Tensor:
    """Per-token ``−min(r·A, clip(r, 1−ε_low, 1+ε_high)·A)``; ``adv`` is ``[G,1]``."""
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv
    return -torch.min(surr1, surr2)


class DAPOAlgorithm(PostTrainingAlgorithm):
    name = "dapo"

    def __init__(self, policy: NativePolicy, prompts: Sequence[torch.Tensor], reward: RewardProvider, config: DAPOConfig,
                 rollout: Optional[RolloutConfig] = None, tasks: Optional[Sequence[Dict[str, Any]]] = None):
        config.validate()
        self.policy = policy
        self.cfg = config
        self.reward = reward
        self.prompts = list(prompts)
        self.tasks = list(tasks) if tasks else [{} for _ in self.prompts]
        rc = rollout or RolloutConfig()
        rc.group_size = config.group_size
        self.rollouts = RolloutEngine(policy, rc)
        self.max_response_len = config.max_response_len or rc.max_new_tokens
        if config.kl_coef > 0 and not policy.has_adapter and policy.reference_model is None:
            policy.with_frozen_reference()
        self.rng = random.Random(config.seed)
        self._last_stats: Dict[str, float] = {}

    def parameters(self) -> List[nn.Parameter]:
        return self.policy.trainable_parameters()

    def modules(self):
        return (self.policy.model,)

    @torch.no_grad()
    def collect(self, step: int) -> List[DAPOGroup]:
        idx = [self.rng.randrange(len(self.prompts)) for _ in range(self.cfg.prompts_per_step)]
        groups: List[DAPOGroup] = []
        skipped = 0
        for i in idx:
            trajs = self.rollouts.rollout(self.prompts[i], self.tasks[i])
            rewards = torch.tensor(score_trajectories(self.reward, trajs), dtype=torch.float32)
            # overlong reward shaping
            for g, t in enumerate(trajs):
                if t.response_length >= self.max_response_len:
                    overshoot = t.response_length / self.max_response_len
                    rewards[g] += self.cfg.overlong_penalty * max(0.0, overshoot - 1.0)
                    t.truncated = True
            prompt, resp, mask = stack_group(trajs, self.policy.pad_token_id)
            old = self.policy.logprobs(prompt, resp).cpu() * mask
            if self.cfg.kl_coef > 0:
                with self.policy.reference():
                    ref = self.policy.logprobs(prompt, resp).cpu() * mask
            else:
                ref = torch.zeros_like(old)
            adv = group_relative_advantages(rewards, clamp_std=True)
            group = DAPOGroup(trajs, rewards, adv, old, ref, mask)
            if self.cfg.dynamic_sampling and float(rewards.std()) < 1e-6:
                group.skipped = True
                skipped += 1
            groups.append(group)
        all_r = torch.cat([g.rewards for g in groups])
        self._last_stats = {"reward_mean": float(all_r.mean()), "groups_skipped": float(skipped),
                            "active_groups": float(len(groups) - skipped),
                            "truncated": float(sum(t.truncated for g in groups for t in g.trajectories))}
        return groups

    def loss(self, groups: List[DAPOGroup], micro_step: int, n_micro: int):
        chunk = groups[micro_step::n_micro] if n_micro > 1 else groups
        dev = self.policy.device
        pg_total = torch.zeros((), device=dev)
        kl_total = torch.zeros((), device=dev)
        ent_total = torch.zeros((), device=dev)
        n_tokens = 0.0
        for group in chunk:
            if group.skipped:
                continue
            prompt, resp, _ = stack_group(group.trajectories, self.policy.pad_token_id)
            mask = group.mask.to(dev)
            new = self.policy.logprobs(prompt, resp) * mask
            ratio = torch.exp(new - group.old_logprobs.to(dev)) * mask
            per_token = dapo_token_objective(ratio, group.advantages.to(dev).unsqueeze(1), self.cfg.clip_eps_low, self.cfg.clip_eps_high)
            pg_total = pg_total + (per_token * mask).sum()
            if self.cfg.kl_coef > 0:
                # token-level KL(new || ref) estimated on sampled tokens: new − ref
                kl_total = kl_total + ((new - group.ref_logprobs.to(dev)) * mask).sum()
            if self.cfg.entropy_coef > 0:
                logits = self.policy.full_sequence_logits(prompt, resp)[:, prompt.shape[1] - 1 : -1, :]
                ent_total = ent_total + (entropy_from_logits(logits) * mask).sum()
            n_tokens += float(mask.sum())
        if n_tokens == 0:
            zero = sum(p.sum() for p in self.parameters()) * 0.0
            return zero, {**self._last_stats, "policy_loss": 0.0, "kl": 0.0, "entropy": 0.0, "tokens": 0.0}
        loss = (pg_total + self.cfg.kl_coef * kl_total - self.cfg.entropy_coef * ent_total) / n_tokens
        metrics = {**self._last_stats, "policy_loss": float(pg_total.detach()) / n_tokens,
                   "kl": float(kl_total.detach()) / n_tokens, "entropy": float(ent_total.detach()) / n_tokens,
                   "tokens": n_tokens}
        return loss, metrics

    @torch.no_grad()
    def evaluate(self, step: int) -> Dict[str, float]:
        n = min(8, len(self.prompts))
        passed, reward_sum = 0, 0.0
        for i in range(n):
            t = self.rollouts.rollout(self.prompts[i], self.tasks[i], group_size=1)[0]
            r = self.reward.score(t)
            reward_sum += r.value
            passed += int(bool(r.passed))
        return {"reward": reward_sum / max(n, 1), "pass_rate": passed / max(n, 1)}

    def state(self) -> Dict[str, Any]:
        return self.policy.state_for_checkpoint()

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.policy.load_checkpoint_state(state)

    def model_spec(self) -> Dict[str, Any]:
        return self.policy.spec.to_dict()

    def tokenizer_meta(self):
        return self.policy.tokenizer.metadata()
