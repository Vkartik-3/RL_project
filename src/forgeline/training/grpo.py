"""Group Relative Policy Optimization (single-turn and tool-using agent mode).

For each prompt ``G`` completions are sampled; advantages are group-normalised
``(r − mean) / (std + ε)`` — no value model.

``objective``:
* ``clipped`` — PPO-style clipped surrogate on the probability ratio plus a
  KL penalty ``β·clamp(mean(log π_ref − log π_new), 0)`` (the configuration
  recorded under benchmarks/post_training/grpo and agent_grpo).
* ``reinforce`` — ``−A · Σ_t log π(y_t)`` without clipping or KL (the
  configuration recorded for process-reward GRPO).

``ratio_level``: ``sequence_mean`` (ratio of mean token log-probs, recorded) or
``token`` (per-token ratios).

Agent mode (``agent=True``) runs multi-turn tool rollouts; the trajectory
log-prob is the average over model segments of each segment's mean token
log-prob, each conditioned on the full context including tool results.
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
from forgeline.rollouts.agent import AgentRolloutConfig, AgentRolloutEngine
from forgeline.rollouts.engine import RolloutConfig, RolloutEngine, stack_group, stack_old_logprobs
from forgeline.rollouts.rewards.composite import score_trajectories
from forgeline.training.common.advantages import group_relative_advantages
from forgeline.training.common.logprobs import kl_k3, kl_logprob_diff, masked_mean, masked_sum
from forgeline.training.common.trainer import PostTrainingAlgorithm
from forgeline.training.ppo import ppo_clipped_objective


@dataclass
class GRPOConfig:
    group_size: int = 4
    prompts_per_step: int = 2
    clip_ratio: float = 0.2
    kl_coef: float = 0.04
    objective: str = "clipped"  # clipped | reinforce
    ratio_level: str = "sequence_mean"  # sequence_mean | token
    kl_estimator: str = "logprob_diff"  # logprob_diff | k3
    advantage_eps: float = 1e-8
    skip_zero_variance_groups: bool = False
    agent: bool = False
    seed: int = 0

    def validate(self) -> None:
        if self.group_size < 2:
            raise ConfigError("GRPO group_size must be >= 2 (group-relative advantages need variance)")
        if self.objective not in ("clipped", "reinforce"):
            raise ConfigError("objective must be clipped|reinforce")
        if self.ratio_level not in ("sequence_mean", "token"):
            raise ConfigError("ratio_level must be sequence_mean|token")
        if self.kl_estimator not in ("logprob_diff", "k3"):
            raise ConfigError("kl_estimator must be logprob_diff|k3")


@dataclass
class GRPOGroup:
    trajectories: List[Trajectory]
    rewards: torch.Tensor  # [G]
    advantages: torch.Tensor  # [G]
    old_logprobs: Optional[torch.Tensor] = None  # [G, R] or [G] (agent)
    ref_logprobs: Optional[torch.Tensor] = None
    skipped: bool = False


class GRPOAlgorithm(PostTrainingAlgorithm):
    name = "grpo"

    def __init__(self, policy: NativePolicy, prompts: Sequence[torch.Tensor] | Sequence[Dict[str, Any]],
                 reward: RewardProvider, config: GRPOConfig, rollout: Optional[RolloutConfig] = None,
                 tasks: Optional[Sequence[Dict[str, Any]]] = None, agent_rollout: Optional[AgentRolloutConfig] = None):
        config.validate()
        self.policy = policy
        self.cfg = config
        self.reward = reward
        self.rng = random.Random(config.seed)
        needs_ref = config.objective == "clipped" and config.kl_coef > 0
        if needs_ref and not policy.has_adapter and policy.reference_model is None:
            policy.with_frozen_reference()
        if config.agent:
            self.tasks = list(prompts)  # type: ignore[arg-type]
            self.agent_engine = AgentRolloutEngine(policy, agent_rollout or AgentRolloutConfig())
            self.prompts: List[torch.Tensor] = []
        else:
            self.prompts = list(prompts)  # type: ignore[arg-type]
            self.tasks = list(tasks) if tasks else [{} for _ in self.prompts]
            rc = rollout or RolloutConfig()
            rc.group_size = config.group_size
            self.rollouts = RolloutEngine(policy, rc)
        self._last_stats: Dict[str, float] = {}

    def parameters(self) -> List[nn.Parameter]:
        return self.policy.trainable_parameters()

    def modules(self):
        return (self.policy.model,)

    # ── collection ───────────────────────────────────────────────────────
    @torch.no_grad()
    def collect(self, step: int) -> List[GRPOGroup]:
        n = len(self.tasks if self.cfg.agent else self.prompts)
        idx = [self.rng.randrange(n) for _ in range(self.cfg.prompts_per_step)]
        groups: List[GRPOGroup] = []
        skipped = 0
        for i in idx:
            if self.cfg.agent:
                trajs = self.agent_engine.rollout(self.tasks[i], self.cfg.group_size)
            else:
                trajs = self.rollouts.rollout(self.prompts[i], self.tasks[i])
            rewards = torch.tensor(score_trajectories(self.reward, trajs), dtype=torch.float32)
            adv = group_relative_advantages(rewards, eps=self.cfg.advantage_eps)
            group = GRPOGroup(trajs, rewards, adv)
            if self.cfg.skip_zero_variance_groups and float(rewards.std()) < 1e-6:
                group.skipped = True
                skipped += 1
            elif self.cfg.objective == "clipped":
                self._attach_logprobs(group)
            groups.append(group)
        all_r = torch.cat([g.rewards for g in groups])
        self._last_stats = {
            "reward_mean": float(all_r.mean()), "reward_std": float(all_r.std()) if all_r.numel() > 1 else 0.0,
            "group_std": float(torch.stack([g.rewards.std() for g in groups]).mean()),
            "groups_skipped": float(skipped),
            "response_len": float(sum(t.response_length for g in groups for t in g.trajectories) / max(1, sum(len(g.trajectories) for g in groups))),
        }
        if self.cfg.agent:
            self._last_stats["tool_calls"] = float(sum(len(t.tool_results) for g in groups for t in g.trajectories))
        return groups

    @torch.no_grad()
    def _attach_logprobs(self, group: GRPOGroup) -> None:
        if self.cfg.agent:
            old = torch.stack([self.agent_engine.trajectory_logprob(t) for t in group.trajectories]).cpu()
            with self.policy.reference():
                ref = torch.stack([self.agent_engine.trajectory_logprob(t) for t in group.trajectories]).cpu()
        else:
            prompt, resp, mask = stack_group(group.trajectories, self.policy.pad_token_id)
            old = self.policy.logprobs(prompt, resp).cpu() * mask
            with self.policy.reference():
                ref = self.policy.logprobs(prompt, resp).cpu() * mask
        group.old_logprobs, group.ref_logprobs = old, ref

    # ── objective ────────────────────────────────────────────────────────
    def loss(self, groups: List[GRPOGroup], micro_step: int, n_micro: int):
        chunk = groups[micro_step::n_micro] if n_micro > 1 else groups
        dev = self.policy.device
        total = torch.zeros((), device=dev)
        n_used = 0
        kl_sum, pl_sum = 0.0, 0.0
        for group in chunk:
            if group.skipped:
                continue
            adv = group.advantages.to(dev)
            if self.cfg.objective == "reinforce":
                loss = self._reinforce_loss(group, adv)
                kl = torch.zeros((), device=dev)
            else:
                loss, kl = self._clipped_loss(group, adv)
            total = total + loss
            pl_sum += float(loss.detach())
            kl_sum += float(kl.detach())
            n_used += 1
        if n_used == 0:
            # every group was skipped: return a zero loss that still participates in autograd
            zero = sum(p.sum() for p in self.parameters()) * 0.0
            return zero, {**self._last_stats, "policy_loss": 0.0, "kl": 0.0}
        metrics = {**self._last_stats, "policy_loss": pl_sum / n_used, "kl": kl_sum / n_used}
        return total / n_used, metrics

    def _kl(self, ref: torch.Tensor, new: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.cfg.kl_estimator == "k3":
            return kl_k3(ref, new, mask).mean()
        # clamp after averaging over the group (recorded configuration)
        return kl_logprob_diff(ref, new, mask, clamp_min=None).mean().clamp(min=0)

    def _clipped_loss(self, group: GRPOGroup, adv: torch.Tensor):
        dev = self.policy.device
        if self.cfg.agent:
            new = torch.stack([self.agent_engine.trajectory_logprob(t) for t in group.trajectories])
            old = group.old_logprobs.to(dev)
            ref = group.ref_logprobs.to(dev)
            ratio = torch.exp(new - old)
            policy_loss = ppo_clipped_objective(ratio, adv, self.cfg.clip_ratio).mean()
            kl = (ref - new).clamp(min=0).mean()
            return policy_loss + self.cfg.kl_coef * kl, kl
        prompt, resp, mask = stack_group(group.trajectories, self.policy.pad_token_id)
        mask = mask.to(dev)
        new = self.policy.logprobs(prompt, resp) * mask
        old = group.old_logprobs.to(dev)
        ref = group.ref_logprobs.to(dev)
        if self.cfg.ratio_level == "sequence_mean":
            ratio = torch.exp(masked_mean(new - old, mask))
            policy_loss = ppo_clipped_objective(ratio, adv, self.cfg.clip_ratio).mean()
        else:
            ratio = torch.exp(new - old)
            policy_loss = masked_mean(ppo_clipped_objective(ratio, adv.unsqueeze(1), self.cfg.clip_ratio), mask).mean()
        kl = self._kl(ref, new, mask)
        return policy_loss + self.cfg.kl_coef * kl, kl

    def _reinforce_loss(self, group: GRPOGroup, adv: torch.Tensor) -> torch.Tensor:
        dev = self.policy.device
        if self.cfg.agent:
            lp = torch.stack([self.agent_engine.trajectory_logprob(t) for t in group.trajectories])
            return (-adv * lp).sum()
        prompt, resp, mask = stack_group(group.trajectories, self.policy.pad_token_id)
        mask = mask.to(dev)
        seq_lp = masked_sum(self.policy.logprobs(prompt, resp), mask)
        return (-adv * seq_lp).sum()

    # ── evaluation ───────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, step: int) -> Dict[str, float]:
        n = min(8, len(self.tasks if self.cfg.agent else self.prompts))
        passed, total, reward_sum = 0, 0, 0.0
        for i in range(n):
            if self.cfg.agent:
                trajs = self.agent_engine.rollout(self.tasks[i], 1, temperature=0.0)
            else:
                trajs = self.rollouts.rollout(self.prompts[i], self.tasks[i], group_size=1)
            for t in trajs:
                r = self.reward.score(t)
                reward_sum += r.value
                passed += int(bool(r.passed))
                total += 1
        return {"reward": reward_sum / max(total, 1), "pass_rate": passed / max(total, 1)}

    def state(self) -> Dict[str, Any]:
        return self.policy.state_for_checkpoint()

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.policy.load_checkpoint_state(state)

    def model_spec(self) -> Dict[str, Any]:
        return self.policy.spec.to_dict()

    def tokenizer_meta(self):
        return self.policy.tokenizer.metadata()
