"""PPO for the allocation environment, run through the shared :class:`Trainer`.

This is *not* a second PPO implementation: the clipped objective (``ppo_clipped_objective``), GAE
(``compute_gae``), advantage normalisation, optimizer, schedules, precision, checkpointing, DDP synchronisation and
metrics all come from the existing training stack. This module only defines how a batch of environment episodes is
collected and how the per-step loss is formed for a stateless (MLP) or sequence-conditioned (GRU) actor-critic.

PPO epochs map onto trainer steps: ``collect`` samples fresh episodes every ``ppo_epochs`` trainer steps and returns
the same rollout (with old log-probs frozen at collection time) in between, so each rollout receives ``ppo_epochs``
optimizer steps — standard PPO with ``gradient_accumulation_steps`` acting as the number of minibatches.

Objective per step (γ = ``gamma``, λ = ``gae_lambda``)::

    A_t   = GAE(r, V_old)                  normalised across the rollout
    ρ_t   = exp(log π(a_t|h_t) − log π_old(a_t|h_t))
    L     = mean_t[ −min(ρ_t A_t, clip(ρ_t, 1−ε, 1+ε) A_t) ] + c_v · mean_t (V(h_t) − R_t)² − c_H · mean_t H[π(·|h_t)]
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from forgeline.core.errors import ConfigError
from forgeline.domains.allocation.env import OBS_DIM, AllocationEnvConfig, VectorEnv
from forgeline.domains.allocation.policies import (
    HistoryBuffer, NeuralPolicy, PolicyNetConfig, StepContext, build_actor_critic, history_feature_dim,
)
from forgeline.domains.allocation.rollout import aggregate_metrics, run_episodes, seeds_for
from forgeline.training.common.advantages import compute_gae, normalize_advantages
from forgeline.training.common.trainer import PostTrainingAlgorithm
from forgeline.training.ppo import ppo_clipped_objective


@dataclass
class AllocationPPOConfig:
    episodes_per_rollout: int = 16
    ppo_epochs: int = 4
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 1.0
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    eval_episodes: int = 64
    eval_seed: int = 10_000  # held-out seed family for periodic evaluation
    seed: int = 0

    def validate(self) -> None:
        if self.episodes_per_rollout < 1 or self.ppo_epochs < 1 or self.eval_episodes < 1:
            raise ConfigError("episodes_per_rollout, ppo_epochs and eval_episodes must be >= 1")
        if not 0 < self.clip_ratio < 1 or not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ConfigError("clip_ratio in (0,1), gamma in (0,1], gae_lambda in [0,1] required")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AllocationPPOConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ConfigError(f"unknown AllocationPPOConfig keys: {sorted(unknown)}")
        c = cls(**d)
        c.validate()
        return c


@dataclass
class RolloutBatch:
    obs: torch.Tensor  # [S, OBS_DIM]
    hist: Optional[torch.Tensor]  # [S, W, D] (GRU) or None
    lengths: Optional[torch.Tensor]  # [S]
    actions: torch.Tensor  # [S]
    old_logp: torch.Tensor  # [S]
    advantages: torch.Tensor  # [S]
    returns: torch.Tensor  # [S]
    perm: torch.Tensor  # minibatch permutation for this epoch
    stats: Dict[str, float]


class AllocationPPOAlgorithm(PostTrainingAlgorithm):
    name = "allocation_ppo"

    def __init__(self, env_cfg: AllocationEnvConfig, net_cfg: PolicyNetConfig, config: AllocationPPOConfig,
                 device: torch.device | str = "cpu"):
        env_cfg.validate(); net_cfg.validate(); config.validate()
        self.env_cfg, self.net_cfg, self.cfg = env_cfg, net_cfg, config
        self.device = torch.device(device)
        self.net = build_actor_critic(env_cfg.n_actions, net_cfg).to(self.device)
        self.policy = NeuralPolicy(self.net, env_cfg, self.device)
        self.rng = np.random.default_rng(config.seed)
        self._batch: Optional[RolloutBatch] = None
        self._rollouts = 0
        self._episodes_seen = 0

    # ── PostTrainingAlgorithm surface ────────────────────────────────────
    def parameters(self) -> List[nn.Parameter]:
        return [p for p in self.net.parameters() if p.requires_grad]

    def modules(self):
        return (self.net,)

    def collect(self, step: int) -> RolloutBatch:
        if self._batch is None or step % self.cfg.ppo_epochs == 0:
            self._batch = self._rollout()
            self._rollouts += 1
        S = self._batch.actions.shape[0]
        self._batch.perm = torch.as_tensor(self.rng.permutation(S), device=self.device)
        return self._batch

    @torch.no_grad()
    def _rollout(self) -> RolloutBatch:
        cfg, env_cfg = self.cfg, self.env_cfg
        n, T = cfg.episodes_per_rollout, env_cfg.horizon
        seeds = [int(s) for s in self.rng.integers(0, 2 ** 31 - 1, size=n)]
        self._episodes_seen += n
        venv = VectorEnv(env_cfg, n)
        obs = venv.reset(seeds)
        self.policy.start(n)
        seq = self.policy.is_sequence
        obs_buf = np.zeros((T, n, OBS_DIM), np.float32)
        hist_buf = np.zeros((T, n, self.net_cfg.window, history_feature_dim(env_cfg.n_actions)), np.float32) if seq else None
        len_buf = np.zeros((T, n), np.int64)
        act_buf = np.zeros((T, n), np.int64); logp_buf = np.zeros((T, n), np.float32)
        val_buf = np.zeros((T + 1, n), np.float32); rew_buf = np.zeros((T, n), np.float32)
        clipped = 0
        for t in range(T):
            if seq:
                self.policy.hist.set_obs(obs)
                hist_buf[t] = self.policy.hist.buf; len_buf[t] = self.policy.hist.lengths
            logits, value = self.policy.evaluate(obs)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            obs_buf[t] = obs; act_buf[t] = a.cpu().numpy(); logp_buf[t] = dist.log_prob(a).cpu().numpy(); val_buf[t] = value.cpu().numpy()
            obs, rewards, dones, infos = venv.step(act_buf[t])
            rew_buf[t] = rewards
            clipped += sum(i.clipped for i in infos)
            self.policy.observe(act_buf[t], rewards, np.array([i.spend for i in infos]), np.array([i.remaining_after for i in infos]))
        val_buf[T] = 0.0  # terminal
        adv = np.zeros((T, n), np.float32); ret = np.zeros((T, n), np.float32)
        for j in range(n):
            a_j, r_j = compute_gae(torch.as_tensor(rew_buf[:, j]), torch.as_tensor(val_buf[:, j]), cfg.gamma, cfg.gae_lambda)
            adv[:, j] = a_j.numpy(); ret[:, j] = r_j.numpy()
        flat = lambda x: torch.as_tensor(x.reshape(T * n, *x.shape[2:]), device=self.device)
        advantages = flat(adv)
        if cfg.normalize_advantages:
            advantages = normalize_advantages(advantages)
        ep_values = rew_buf.sum(0)
        stats = {"reward_mean": float(ep_values.mean()), "reward_std": float(ep_values.std()), "clipped_per_episode": clipped / n,
                 "episodes_seen": float(self._episodes_seen), "utilization": float(np.mean([e.cum_spend / env_cfg.budget for e in venv.envs]))}
        return RolloutBatch(flat(obs_buf), flat(hist_buf) if seq else None, flat(len_buf) if seq else None, flat(act_buf),
                            flat(logp_buf), advantages, flat(ret), torch.arange(T * n, device=self.device), stats)

    def loss(self, batch: RolloutBatch, micro_step: int, n_micro: int):
        idx = batch.perm[micro_step::n_micro] if n_micro > 1 else batch.perm
        feats = batch.obs[idx] if batch.hist is None else (batch.obs[idx], batch.hist[idx], batch.lengths[idx])
        logits, values = self.net(feats)
        logp_all = F.log_softmax(logits, dim=-1)
        logp = logp_all.gather(1, batch.actions[idx].unsqueeze(1)).squeeze(1)
        ratio = torch.exp(logp - batch.old_logp[idx])
        pg = ppo_clipped_objective(ratio, batch.advantages[idx], self.cfg.clip_ratio).mean()
        v_loss = F.mse_loss(values, batch.returns[idx])
        entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
        approx_kl = (batch.old_logp[idx] - logp).mean()
        loss = pg + self.cfg.value_coef * v_loss - self.cfg.entropy_coef * entropy
        metrics = {**batch.stats, "policy_loss": float(pg.detach()), "value_loss": float(v_loss.detach()), "entropy": float(entropy.detach()),
                   "approx_kl": float(approx_kl.detach()), "clip_fraction": float(((ratio - 1).abs() > self.cfg.clip_ratio).float().mean())}
        return loss, metrics

    @torch.no_grad()
    def evaluate(self, step: int) -> Dict[str, float]:
        """Held-out simulator evaluation on a fixed seed family (stochastic policy, its actual action distribution)."""
        seeds = seeds_for(self.cfg.eval_seed, self.cfg.eval_episodes)
        eps = run_episodes(self.policy, self.env_cfg, seeds, rng_seed=self.cfg.eval_seed)
        m = aggregate_metrics(eps, self.env_cfg)
        keep = {k: m[k] for k in ("value", "value_std", "utilization", "pacing_error", "early_exhaustion", "violations", "final_pressure")}
        keep["primary"] = -m["value"]  # the trainer keeps the checkpoint with the lowest primary metric
        return keep

    def state(self) -> Dict[str, Any]:
        return {"model": self.net.state_dict()}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.net.load_state_dict(state["model"])

    def model_spec(self) -> Dict[str, Any]:
        return {"family": "allocation_policy", "policy": self.net_cfg.to_dict(), "env": self.env_cfg.to_dict()}

    def tokenizer_meta(self):
        return None


def build_allocation_ppo(params: Dict[str, Any], device: torch.device | str = "cpu") -> AllocationPPOAlgorithm:
    """From ``algorithm_params``: ``{env: {...}, policy: {...}, <AllocationPPOConfig fields>}``."""
    p = dict(params)
    env_cfg = AllocationEnvConfig.from_dict(p.pop("env", {}) or {})
    net_cfg = PolicyNetConfig.from_dict(p.pop("policy", {}) or {})
    return AllocationPPOAlgorithm(env_cfg, net_cfg, AllocationPPOConfig.from_dict(p), device)


def load_allocation_policy(checkpoint: str, device: torch.device | str = "cpu") -> NeuralPolicy:
    """Rebuild a :class:`NeuralPolicy` from a checkpoint written by the trainer."""
    from forgeline.checkpoints.manager import CheckpointManager

    state = CheckpointManager.load(checkpoint, map_location="cpu")
    spec = state.model_spec
    if spec.get("family") != "allocation_policy":
        raise ConfigError(f"checkpoint {checkpoint} is not an allocation policy (model_spec.family={spec.get('family')!r})")
    env_cfg = AllocationEnvConfig.from_dict(spec["env"])
    net = build_actor_critic(env_cfg.n_actions, PolicyNetConfig.from_dict(spec["policy"]))
    net.load_state_dict(state.model_state["model"])
    net.eval()
    return NeuralPolicy(net, env_cfg, device)
