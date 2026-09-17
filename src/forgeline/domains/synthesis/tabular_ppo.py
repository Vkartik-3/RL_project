"""Actor-critic PPO on 128-d trajectory state vectors (tabular baseline).

Loss = clipped surrogate + 0.5·value MSE − 0.01·entropy; the surrogate ratio
is computed against detached log-probs of the same forward pass (so it is 1 on
the first pass, as in the recorded baseline runs).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from forgeline.core.errors import ConfigError, DatasetError
from forgeline.domains.synthesis.features import STATE_DIM, encode_trajectory
from forgeline.domains.synthesis.reward import rule_score


def load_trajectories(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"trajectory file not found: {path}")
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class TabularActorCritic(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        if state_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ConfigError("dimensions must be positive")
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim), nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        f = self.shared(state)
        return self.actor(f), self.critic(f)


@dataclass
class TabularPPOConfig:
    learning_rate: float = 1e-4
    clip_ratio: float = 0.2
    epochs: int = 5
    batch_size: int = 8
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    baseline_reward: float = 0.5


class TabularPPOTrainer:
    def __init__(self, policy: TabularActorCritic, cfg: TabularPPOConfig, reward_fn=rule_score, device: str = "cpu"):
        if cfg.learning_rate <= 0:
            raise ConfigError("learning_rate must be positive")
        self.policy = policy.to(device)
        self.cfg = cfg
        self.reward_fn = reward_fn
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)
        self.log = logging.getLogger("forgeline.synthesis.tabular_ppo")

    def train(self, trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not trajectories:
            raise DatasetError("empty trajectory list")
        self.policy.train()
        metrics: Dict[str, Any] = {"epoch_rewards": [], "epoch_losses": [], "epoch_policy_losses": [], "epoch_value_losses": []}
        for _ in range(self.cfg.epochs):
            sums = {"reward": 0.0, "loss": 0.0, "policy": 0.0, "value": 0.0}
            n = 0
            for i in range(0, len(trajectories), self.cfg.batch_size):
                batch = trajectories[i : i + self.cfg.batch_size]
                states = torch.stack([encode_trajectory(t) for t in batch]).to(self.device)
                rewards = torch.tensor([self.reward_fn(t) for t in batch], dtype=torch.float32, device=self.device)
                probs, values = self.policy(states)
                values = values.squeeze(-1)
                adv = rewards - values.detach()
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                log_probs = torch.log(probs.max(dim=1)[0] + 1e-8)
                ratio = torch.exp(log_probs - log_probs.detach())
                surr1, surr2 = ratio * adv, torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, rewards)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()
                sums["reward"] += float(rewards.mean()); sums["loss"] += float(loss.detach()); sums["policy"] += float(policy_loss.detach()); sums["value"] += float(value_loss.detach())
                n += 1
            metrics["epoch_rewards"].append(sums["reward"] / n)
            metrics["epoch_losses"].append(sums["loss"] / n)
            metrics["epoch_policy_losses"].append(sums["policy"] / n)
            metrics["epoch_value_losses"].append(sums["value"] / n)
        metrics["final_reward"] = metrics["epoch_rewards"][-1]
        metrics["improvement"] = (metrics["final_reward"] - self.cfg.baseline_reward) / self.cfg.baseline_reward * 100
        return metrics
