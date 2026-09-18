"""Policies for the budgeted allocation environment.

Every policy exposes the same batched interface so evaluation, logging, OPE and A/B code are policy-agnostic::

    policy.start(n)                          -> reset per-episode internal state for n parallel episodes
    policy.act(obs, ctx, rng, greedy=False)  -> (actions [n], probs [n, A])
    policy.observe(actions, rewards, spends, remaining) -> update internal state after the environment step

``ctx`` is the list of per-environment :class:`StepContext` (raw scalars the heuristics use). ``probs`` are the
behaviour probabilities that get logged as propensities, so stochastic policies must return their true action
distribution and deterministic policies return one-hot rows.

Policies:

* :class:`ThresholdPacingPolicy` — static heuristic: allocate when value/cost exceeds a threshold, gated by uniform
  pacing (spend so far must not exceed the pro-rata budget).
* :class:`DualPacingPolicy` — online primal-dual (Lagrangian) controller: maximises `E[value](m) − λ·cost(m)` per step
  and adapts the dual price λ to the budget rate constraint with a subgradient step.
* :class:`NeuralPolicy` with :class:`MLPActorCritic` — stateless actor-critic on the current observation (used by PPO).
* :class:`NeuralPolicy` with :class:`GRUActorCritic` — sequence-conditioned actor-critic: a GRU over the last `window` steps of
  (observation, one-hot action, reward, spend, remaining-budget) tuples.
* :class:`EpsilonMixPolicy` — mixes any policy with ε-uniform exploration so logged propensities have full support.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from forgeline.core.errors import ConfigError
from forgeline.domains.allocation.env import OBS_DIM, AllocationEnvConfig, response_gain


@dataclass
class StepContext:
    """Raw per-step scalars visible to the policy (the same information as the observation, unnormalised)."""

    t: int
    horizon: int
    remaining: float
    budget: float
    cum_spend: float
    value: float
    cost: float
    quality: float


class Policy:
    name = "policy"
    stochastic = False

    def start(self, n: int) -> None:  # noqa: D401
        pass

    def act(self, obs: np.ndarray, ctx: Sequence[StepContext], rng: np.random.Generator, greedy: bool = False):
        raise NotImplementedError

    def observe(self, actions: np.ndarray, rewards: np.ndarray, spends: np.ndarray, remaining: Optional[np.ndarray] = None) -> None:
        pass

    def action_probs(self, obs: np.ndarray, ctx: Sequence[StepContext]) -> np.ndarray:
        """Target-policy probabilities for OPE (no sampling, no state change)."""
        raise NotImplementedError


def _one_hot(actions: Sequence[int], n: int) -> np.ndarray:
    out = np.zeros((len(actions), n), dtype=np.float64)
    out[np.arange(len(actions)), list(actions)] = 1.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Static heuristic
# ─────────────────────────────────────────────────────────────────────────────

class ThresholdPacingPolicy(Policy):
    """Allocate ``level`` when ``value·quality / cost ≥ threshold`` and cumulative spend ≤ pro-rata budget, else abstain."""

    name = "threshold_pacing"

    def __init__(self, env_cfg: AllocationEnvConfig, threshold: float = 1.0, level: int = 2, pacing_slack: float = 0.05):
        if not 0 < level < env_cfg.n_actions:
            raise ConfigError("level must be a non-abstain action index")
        self.cfg, self.threshold, self.level, self.slack = env_cfg, threshold, level, pacing_slack

    def _decide(self, c: StepContext) -> int:
        ratio = c.value * c.quality / max(c.cost, 1e-9)
        on_pace = c.cum_spend <= c.budget * (c.t / c.horizon + self.slack)
        return self.level if (ratio >= self.threshold and on_pace and c.remaining > 0) else 0

    def act(self, obs, ctx, rng, greedy=False):
        actions = np.array([self._decide(c) for c in ctx])
        return actions, _one_hot(actions, self.cfg.n_actions)

    def action_probs(self, obs, ctx):
        return _one_hot([self._decide(c) for c in ctx], self.cfg.n_actions)


# ─────────────────────────────────────────────────────────────────────────────
#  Online primal-dual pacing controller
# ─────────────────────────────────────────────────────────────────────────────

class DualPacingPolicy(Policy):
    """Lagrangian pacing.

    Objective per episode: maximise Σ_t E[r_t | m_t] subject to Σ_t x_t ≤ B. With multiplier λ ≥ 0 the per-step
    Lagrangian is ``L(m) = v_t q_t g(m) − λ · m c_t`` (pressure is unobserved, so the base cost is used); the
    action is ``argmax_m L(m)``. The dual price adapts online by projected subgradient descent on the budget-rate
    constraint::

        λ ← max(0, λ + α · (x_t − B / T))        (spend above the uniform rate raises the price)

    with step size ``α`` (``learning_rate``) and initial price ``λ₀``. A late-horizon guard forces abstention when
    the budget is exhausted. This is a standard dual-descent budget pacer; it adapts to observed spend, not to
    the latent pressure.
    """

    name = "dual_pacing"

    def __init__(self, env_cfg: AllocationEnvConfig, learning_rate: float = 0.2, initial_price: float = 0.8):
        if learning_rate <= 0 or initial_price < 0:
            raise ConfigError("dual pacing needs learning_rate > 0 and initial_price >= 0")
        self.cfg, self.lr, self.lam0 = env_cfg, learning_rate, initial_price
        self.lam: np.ndarray = np.zeros(0)
        self._last_spend_target = env_cfg.budget / env_cfg.horizon

    def start(self, n: int) -> None:
        self.lam = np.full(n, self.lam0, dtype=np.float64)

    def _scores(self, c: StepContext, lam: float) -> np.ndarray:
        ms = np.array(self.cfg.multipliers)
        return c.value * c.quality * (1.0 - np.exp(-ms)) - lam * ms * c.cost

    def _decide(self, c: StepContext, lam: float) -> int:
        if c.remaining <= 0:
            return 0
        return int(np.argmax(self._scores(c, lam)))

    def act(self, obs, ctx, rng, greedy=False):
        if self.lam.shape[0] != len(ctx):
            self.start(len(ctx))
        actions = np.array([self._decide(c, self.lam[i]) for i, c in enumerate(ctx)])
        return actions, _one_hot(actions, self.cfg.n_actions)

    def observe(self, actions, rewards, spends, remaining=None):
        self.lam = np.maximum(0.0, self.lam + self.lr * (np.asarray(spends) - self._last_spend_target))

    def action_probs(self, obs, ctx):
        if self.lam.shape[0] != len(ctx):
            self.start(len(ctx))
        return _one_hot([self._decide(c, self.lam[i]) for i, c in enumerate(ctx)], self.cfg.n_actions)

    def prices(self) -> np.ndarray:
        return self.lam.copy()


# ─────────────────────────────────────────────────────────────────────────────
#  Neural actor-critics (PPO)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PolicyNetConfig:
    kind: str = "mlp"  # mlp | gru
    hidden: int = 64
    window: int = 8  # sequence length consumed by the GRU
    gru_layers: int = 1

    def validate(self) -> None:
        if self.kind not in ("mlp", "gru"):
            raise ConfigError("policy kind must be mlp|gru")
        if self.hidden < 1 or self.window < 2 or self.gru_layers < 1:
            raise ConfigError("hidden >= 1, window >= 2, gru_layers >= 1 required")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PolicyNetConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ConfigError(f"unknown policy keys: {sorted(unknown)}")
        c = cls(**d)
        c.validate()
        return c

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class ActorCritic(nn.Module):
    """Common surface: ``forward(features) -> (logits [n, A], value [n])``."""

    kind = "base"

    def __init__(self, n_actions: int, config: PolicyNetConfig):
        super().__init__()
        self.n_actions = n_actions
        self.config = config


class MLPActorCritic(ActorCritic):
    kind = "mlp"

    def __init__(self, n_actions: int, config: PolicyNetConfig):
        super().__init__(n_actions, config)
        h = config.hidden
        self.body = nn.Sequential(nn.Linear(OBS_DIM, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.pi = nn.Linear(h, n_actions)
        self.v = nn.Linear(h, 1)

    def forward(self, obs: torch.Tensor):  # obs [n, OBS_DIM]
        z = self.body(obs)
        return self.pi(z), self.v(z).squeeze(-1)


def history_feature_dim(n_actions: int) -> int:
    return OBS_DIM + n_actions + 3  # obs, one-hot action, reward, spend, remaining


class GRUActorCritic(ActorCritic):
    """GRU over a window of past (obs, action, reward, spend, remaining) tuples plus the current observation.

    The encoder input at step t is the sequence ``h_{t−W}, …, h_{t−1}`` where
    ``h_j = [obs_j, onehot(a_j), r_j/v̄, x_j/c̄, B_j/B]``; positions before the episode start are zero-padded and
    masked out. The final GRU state is concatenated with the current observation before the actor and critic heads,
    so the policy conditions on both the present opportunity and the realised history (from which the latent
    pressure can be inferred).
    """

    kind = "gru"

    def __init__(self, n_actions: int, config: PolicyNetConfig):
        super().__init__(n_actions, config)
        h = config.hidden
        self.in_dim = history_feature_dim(n_actions)
        self.gru = nn.GRU(self.in_dim, h, num_layers=config.gru_layers, batch_first=True)
        self.obs_proj = nn.Linear(OBS_DIM, h)
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.Tanh())
        self.pi = nn.Linear(h, n_actions)
        self.v = nn.Linear(h, 1)

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        obs, hist, lengths = features  # obs [n, OBS_DIM], hist [n, W, in_dim], lengths [n]
        n, W, _ = hist.shape
        if int(lengths.max()) == 0:
            enc = hist.new_zeros(n, self.gru.hidden_size)
        else:
            # the buffer is left-padded (most recent step last); shift each row so its valid steps start at 0,
            # then pack so the GRU only reads real history
            shift = (W - lengths).unsqueeze(1)  # [n, 1]
            idx = (torch.arange(W, device=hist.device).unsqueeze(0) + shift) % W
            aligned = torch.gather(hist, 1, idx.unsqueeze(-1).expand(-1, -1, hist.shape[-1]))
            packed = nn.utils.rnn.pack_padded_sequence(aligned, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
            _, h_n = self.gru(packed)
            enc = h_n[-1] * (lengths > 0).float().unsqueeze(-1)
        z = self.head(torch.cat([torch.tanh(self.obs_proj(obs)), enc], dim=-1))
        return self.pi(z), self.v(z).squeeze(-1)


def build_actor_critic(n_actions: int, config: PolicyNetConfig) -> ActorCritic:
    config.validate()
    return MLPActorCritic(n_actions, config) if config.kind == "mlp" else GRUActorCritic(n_actions, config)


class HistoryBuffer:
    """Per-environment rolling window of history features for the GRU policy."""

    def __init__(self, n: int, window: int, n_actions: int, env_cfg: AllocationEnvConfig):
        self.n, self.window, self.n_actions, self.cfg = n, window, n_actions, env_cfg
        self.dim = history_feature_dim(n_actions)
        self.buf = np.zeros((n, window, self.dim), dtype=np.float32)
        self.lengths = np.zeros(n, dtype=np.int64)
        self._pending_obs: Optional[np.ndarray] = None

    def set_obs(self, obs: np.ndarray) -> None:
        self._pending_obs = np.asarray(obs, dtype=np.float32)

    def push(self, actions: np.ndarray, rewards: np.ndarray, spends: np.ndarray, remaining: np.ndarray) -> None:
        feat = np.zeros((self.n, self.dim), dtype=np.float32)
        feat[:, :OBS_DIM] = self._pending_obs
        feat[np.arange(self.n), OBS_DIM + np.asarray(actions)] = 1.0
        feat[:, OBS_DIM + self.n_actions] = np.asarray(rewards) / self.cfg.value_mean
        feat[:, OBS_DIM + self.n_actions + 1] = np.asarray(spends) / self.cfg.cost_mean
        feat[:, OBS_DIM + self.n_actions + 2] = np.asarray(remaining) / self.cfg.budget
        self.buf = np.concatenate([self.buf[:, 1:], feat[:, None, :]], axis=1)
        self.lengths = np.minimum(self.lengths + 1, self.window)

    def tensors(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.as_tensor(self.buf, device=device), torch.as_tensor(self.lengths, device=device)


class NeuralPolicy(Policy):
    """Wraps an :class:`ActorCritic` (MLP or GRU) behind the batched policy interface."""

    stochastic = True

    def __init__(self, net: ActorCritic, env_cfg: AllocationEnvConfig, device: torch.device | str = "cpu"):
        self.net = net.to(device)
        self.cfg = env_cfg
        self.device = torch.device(device)
        self.name = f"ppo_{net.kind}"
        self.hist: Optional[HistoryBuffer] = None
        self._n = 0

    @property
    def is_sequence(self) -> bool:
        return self.net.kind == "gru"

    def start(self, n: int) -> None:
        self._n = n
        if self.is_sequence:
            self.hist = HistoryBuffer(n, self.net.config.window, self.cfg.n_actions, self.cfg)

    def features(self, obs: np.ndarray):
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        if not self.is_sequence:
            return obs_t
        if self.hist is None or self.hist.n != obs_t.shape[0]:
            self.start(obs_t.shape[0])
        h, l = self.hist.tensors(self.device)
        return (obs_t, h, l)

    @torch.no_grad()
    def evaluate(self, obs: np.ndarray):
        logits, value = self.net(self.features(obs))
        return logits, value

    def act(self, obs, ctx, rng, greedy=False):
        if self.is_sequence:
            if self.hist is None or self.hist.n != len(obs):
                self.start(len(obs))
            self.hist.set_obs(obs)
        logits, _ = self.evaluate(obs)
        probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
        if greedy:
            actions = probs.argmax(axis=1)
        else:
            cum = probs.cumsum(axis=1)
            u = rng.uniform(size=(len(obs), 1))
            actions = (u < cum).argmax(axis=1)
        return actions, probs

    def observe(self, actions, rewards, spends, remaining=None):
        if self.is_sequence and self.hist is not None:
            if remaining is None:
                raise ConfigError("GRU policy needs remaining budget in observe()")
            self.hist.push(actions, rewards, spends, remaining)

    def action_probs(self, obs, ctx):
        if self.is_sequence and (self.hist is None or self.hist.n != len(obs)):
            self.start(len(obs))
        logits, _ = self.evaluate(obs)
        return torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)


class EpsilonMixPolicy(Policy):
    """``(1 − ε)·base + ε·uniform`` — guarantees every action has propensity ≥ ε/A (behaviour policy for logging)."""

    stochastic = True

    def __init__(self, base: Policy, epsilon: float, n_actions: int):
        if not 0 < epsilon <= 1:
            raise ConfigError("epsilon must be in (0, 1]")
        self.base, self.eps, self.n_actions = base, epsilon, n_actions
        self.name = f"{base.name}_eps{epsilon:g}"

    def start(self, n):
        self.base.start(n)

    def _mix(self, probs: np.ndarray) -> np.ndarray:
        return (1.0 - self.eps) * probs + self.eps / self.n_actions

    def act(self, obs, ctx, rng, greedy=False):
        _, base_probs = self.base.act(obs, ctx, rng, greedy=False)
        probs = self._mix(base_probs)
        cum = probs.cumsum(axis=1)
        u = rng.uniform(size=(len(obs), 1))
        actions = (u < cum).argmax(axis=1)
        return actions, probs

    def observe(self, actions, rewards, spends, remaining=None):
        self.base.observe(actions, rewards, spends, remaining)

    def action_probs(self, obs, ctx):
        return self._mix(self.base.action_probs(obs, ctx))


def policy_spec(policy: Policy) -> Dict[str, Any]:
    if isinstance(policy, NeuralPolicy):
        return {"kind": policy.net.kind, "net": policy.net.config.to_dict()}
    return {"kind": policy.name}
