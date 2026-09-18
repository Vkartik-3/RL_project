"""BudgetedAllocationEnv — a finite-horizon, budget-constrained sequential allocation POMDP.

The decision problem
--------------------
An agent faces ``T`` opportunities in sequence and holds a finite budget ``B``. At each step it sees the current
opportunity and must choose an allocation intensity **before** any later opportunity is revealed. Spending is
irreversible; the objective is the total realised value over the horizon subject to ``Σ spend ≤ B``.

Per-step quantities (all scalars unless noted)::

    opportunity o_t = (v_t, c_t, q_t, s_t)     value scale, base cost, response quality, season phase
    action      a_t ∈ {0, 1, 2, 3}             allocation multiplier m_t = M[a_t], M = (0, 0.5, 1, 1.5)
    latent      p_t ≥ 0                        pressure — a hidden, policy-dependent state (see below)

    effective cost   ĉ_t = c_t · (1 + κ p_t)
    requested spend  m_t · ĉ_t
    realised spend   x_t = min(m_t · ĉ_t, B_t)                 (hard budget; the clip is recorded as a violation)
    intensity        u_t = x_t / ĉ_t                           (multiplier actually afforded)
    response rate    ρ_t = q_t · g(u_t) · max(0, 1 − φ p_t),    g(u) = 1 − exp(−u)     (concave, saturating)
    outcome          y_t ~ Bernoulli(ρ_t)                       (uncertain realisation)
    reward           r_t = y_t · v_t
    budget           B_{t+1} = B_t − x_t
    pressure         p_{t+1} = δ p_t + η u_t                     (endogenous feedback)

Opportunity process (exogenous, stochastic, non-stationary)::

    s_t = sin(2π (t + phase) / T)                                seasonal phase in [−1, 1]
    v_t = v̄ · exp(σ_v ε_t) · (1 + A_s s_t)                       log-normal value with seasonal drift
    c_t = c̄ · exp(σ_c ε'_t) · (1 + A_c s_t)                      log-normal base cost with (opposite) seasonal drift
    q_t ~ Beta(α_q, β_q)

Endogenous (policy-dependent) dynamics
--------------------------------------
Pressure ``p_t`` accumulates with the intensities the policy actually affords (``η u_t``) and decays geometrically
(``δ``). It raises every future effective cost by ``(1 + κ p_t)`` and lowers every future response rate by
``(1 − φ p_t)``. Aggressive early allocation therefore both drains the budget and degrades the quality of later
opportunities — the same initial seed yields different future states under different policies. Pressure is **not**
part of the observation; it must be inferred from the history of realised costs and outcomes, which makes the
problem partially observable and gives sequence context a genuine role.

Observation (``obs_dim`` = 14, all in roughly [−2, 2])::

    [remaining_budget/B, t/T, (T−t)/T, v_t/v̄ − 1, c_t/c̄ − 1, q_t, s_t,
     pacing_error = cum_spend/B − t/T, cum_value/(T·v̄), last_multiplier/1.5, last_reward/v̄, last_spend/c̄,
     mean(v/v̄ − 1) over the last k opportunities, mean(c/c̄ − 1) over the last k opportunities]

Termination: after step ``T − 1`` (fixed horizon). A depleted budget forces ``x_t = 0`` but the episode continues so
that every episode has exactly ``T`` decisions. Discount: none (undiscounted finite-horizon return); the PPO
estimator uses γ = 1 by default.

The oracle in ``oracle.py`` is an upper bound that ignores pressure and knows the whole opportunity stream.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from forgeline.core.errors import ConfigError

MULTIPLIERS: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5)
ACTION_NAMES: Tuple[str, ...] = ("abstain", "low", "medium", "high")
OBS_DIM = 14
OBS_NAMES: Tuple[str, ...] = (
    "remaining_frac", "elapsed_frac", "remaining_steps_frac", "value_rel", "cost_rel", "quality", "season",
    "pacing_error", "cum_value_norm", "last_multiplier", "last_reward_rel", "last_spend_rel", "recent_value_rel", "recent_cost_rel",
)


def response_gain(u: float) -> float:
    """Saturating response to allocation intensity, g(u) = 1 − e^{−u}."""
    return 1.0 - math.exp(-u)


@dataclass
class AllocationEnvConfig:
    horizon: int = 48
    budget: float = 24.0  # roughly T · c̄ · 0.5: enough for medium intensity on half the steps
    value_mean: float = 1.0
    value_sigma: float = 0.5
    cost_mean: float = 1.0
    cost_sigma: float = 0.3
    quality_alpha: float = 2.0
    quality_beta: float = 2.0
    season_amplitude_value: float = 0.4
    season_amplitude_cost: float = -0.2
    random_phase: bool = True
    # endogenous pressure
    pressure_decay: float = 0.85  # δ
    pressure_gain: float = 0.25  # η
    pressure_cost_coef: float = 0.6  # κ
    pressure_response_coef: float = 0.35  # φ
    history_window: int = 4  # k for the recent-opportunity features
    multipliers: Tuple[float, ...] = MULTIPLIERS

    def validate(self) -> None:
        if self.horizon < 2:
            raise ConfigError("allocation env horizon must be >= 2")
        if self.budget <= 0 or self.value_mean <= 0 or self.cost_mean <= 0:
            raise ConfigError("budget, value_mean and cost_mean must be positive")
        if not 0.0 <= self.pressure_decay < 1.0:
            raise ConfigError("pressure_decay must be in [0, 1)")
        if self.pressure_gain < 0 or self.pressure_cost_coef < 0 or self.pressure_response_coef < 0:
            raise ConfigError("pressure coefficients must be >= 0")
        if self.value_sigma < 0 or self.cost_sigma < 0 or self.quality_alpha <= 0 or self.quality_beta <= 0:
            raise ConfigError("invalid opportunity distribution parameters")
        if len(self.multipliers) < 2 or self.multipliers[0] != 0.0 or any(m < 0 for m in self.multipliers):
            raise ConfigError("multipliers must start with 0 (abstain) and be non-negative")
        if self.history_window < 1:
            raise ConfigError("history_window must be >= 1")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AllocationEnvConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ConfigError(f"unknown allocation env keys: {sorted(unknown)}")
        d = dict(data)
        if "multipliers" in d:
            d["multipliers"] = tuple(float(m) for m in d["multipliers"])
        cfg = cls(**d)
        cfg.validate()
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["multipliers"] = list(self.multipliers)
        return d

    @property
    def n_actions(self) -> int:
        return len(self.multipliers)


@dataclass
class Opportunity:
    value: float
    cost: float
    quality: float
    season: float


@dataclass
class StepInfo:
    """Everything about one transition, used for logging, metrics and the oracle."""

    t: int
    action: int
    multiplier: float
    intensity: float
    effective_cost: float
    spend: float
    reward: float
    outcome: int
    response_rate: float
    pressure_before: float
    pressure_after: float
    remaining_before: float
    remaining_after: float
    clipped: bool
    opportunity: Opportunity


class BudgetedAllocationEnv:
    """Single-episode environment. ``reset(seed)`` fixes the exogenous opportunity stream and outcome noise stream."""

    def __init__(self, config: Optional[AllocationEnvConfig] = None):
        self.cfg = config or AllocationEnvConfig()
        self.cfg.validate()
        self._rng: np.random.Generator = np.random.default_rng(0)
        self.t = 0
        self.remaining = self.cfg.budget
        self.pressure = 0.0
        self.cum_spend = 0.0
        self.cum_value = 0.0
        self.opportunities: List[Opportunity] = []
        self.history: List[StepInfo] = []
        self._noise: np.ndarray = np.zeros(0)
        self.done = True

    # ── episode generation ───────────────────────────────────────────────
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        cfg = self.cfg
        self._rng = np.random.default_rng(seed)
        T = cfg.horizon
        phase = self._rng.uniform(0, T) if cfg.random_phase else 0.0
        eps_v = self._rng.standard_normal(T)
        eps_c = self._rng.standard_normal(T)
        q = self._rng.beta(cfg.quality_alpha, cfg.quality_beta, size=T)
        self._noise = self._rng.uniform(0.0, 1.0, size=T)  # outcome uniforms: y_t = 1[u < ρ_t]
        self.opportunities = []
        for t in range(T):
            s = math.sin(2.0 * math.pi * (t + phase) / T)
            v = cfg.value_mean * math.exp(cfg.value_sigma * eps_v[t] - 0.5 * cfg.value_sigma ** 2) * (1.0 + cfg.season_amplitude_value * s)
            c = cfg.cost_mean * math.exp(cfg.cost_sigma * eps_c[t] - 0.5 * cfg.cost_sigma ** 2) * (1.0 + cfg.season_amplitude_cost * s)
            self.opportunities.append(Opportunity(max(v, 1e-6), max(c, 1e-6), float(q[t]), s))
        self.t = 0
        self.remaining = cfg.budget
        self.pressure = 0.0
        self.cum_spend = 0.0
        self.cum_value = 0.0
        self.history = []
        self.done = False
        return self.observation()

    # ── dynamics ─────────────────────────────────────────────────────────
    def transition(self, action: int) -> StepInfo:
        """Apply ``action`` at the current step. Pure function of the current state and the pre-drawn noise."""
        if self.done:
            raise ConfigError("episode is finished; call reset()")
        if not 0 <= int(action) < self.cfg.n_actions:
            raise ConfigError(f"invalid action {action!r}; expected 0..{self.cfg.n_actions - 1}")
        cfg = self.cfg
        t = self.t
        o = self.opportunities[t]
        m = cfg.multipliers[int(action)]
        eff_cost = o.cost * (1.0 + cfg.pressure_cost_coef * self.pressure)
        requested = m * eff_cost
        spend = min(requested, self.remaining)
        clipped = requested > self.remaining + 1e-12
        intensity = spend / eff_cost if eff_cost > 0 else 0.0
        rate = o.quality * response_gain(intensity) * max(0.0, 1.0 - cfg.pressure_response_coef * self.pressure)
        outcome = int(self._noise[t] < rate)
        reward = outcome * o.value
        p_before = self.pressure
        self.pressure = cfg.pressure_decay * self.pressure + cfg.pressure_gain * intensity
        remaining_before = self.remaining
        self.remaining -= spend
        self.cum_spend += spend
        self.cum_value += reward
        info = StepInfo(t, int(action), m, intensity, eff_cost, spend, reward, outcome, rate, p_before, self.pressure,
                        remaining_before, self.remaining, clipped, o)
        self.history.append(info)
        self.t += 1
        self.done = self.t >= cfg.horizon
        return info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, StepInfo]:
        info = self.transition(action)
        obs = self.observation() if not self.done else np.zeros(OBS_DIM, dtype=np.float32)
        return obs, info.reward, self.done, info

    # ── observation ──────────────────────────────────────────────────────
    def observation(self) -> np.ndarray:
        cfg = self.cfg
        T = cfg.horizon
        t = self.t
        o = self.opportunities[t]
        last = self.history[-1] if self.history else None
        k = cfg.history_window
        recent = self.opportunities[max(0, t - k):t]
        rv = float(np.mean([r.value / cfg.value_mean - 1.0 for r in recent])) if recent else 0.0
        rc = float(np.mean([r.cost / cfg.cost_mean - 1.0 for r in recent])) if recent else 0.0
        max_m = max(cfg.multipliers)
        obs = np.array([
            self.remaining / cfg.budget,
            t / T,
            (T - t) / T,
            o.value / cfg.value_mean - 1.0,
            o.cost / cfg.cost_mean - 1.0,
            o.quality,
            o.season,
            self.cum_spend / cfg.budget - t / T,
            self.cum_value / (T * cfg.value_mean),
            (last.multiplier / max_m) if last else 0.0,
            (last.reward / cfg.value_mean) if last else 0.0,
            (last.spend / cfg.cost_mean) if last else 0.0,
            rv,
            rc,
        ], dtype=np.float32)
        return obs

    def state_dict(self) -> Dict[str, float]:
        return {"t": self.t, "remaining": self.remaining, "pressure": self.pressure, "cum_spend": self.cum_spend,
                "cum_value": self.cum_value}

    # ── helpers for the oracle and pacing metrics ────────────────────────
    def opportunity_stream(self) -> List[Opportunity]:
        return list(self.opportunities)

    def outcome_uniforms(self) -> np.ndarray:
        return self._noise.copy()


class VectorEnv:
    """``n`` independent environments stepped in lockstep (batched policies)."""

    def __init__(self, config: AllocationEnvConfig, n: int):
        if n < 1:
            raise ConfigError("VectorEnv needs n >= 1")
        self.envs = [BudgetedAllocationEnv(config) for _ in range(n)]
        self.cfg = config

    def reset(self, seeds: Sequence[int]) -> np.ndarray:
        if len(seeds) != len(self.envs):
            raise ConfigError("one seed per environment is required")
        return np.stack([e.reset(int(s)) for e, s in zip(self.envs, seeds)])

    def step(self, actions: Sequence[int]):
        obs, rewards, dones, infos = [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, d, i = e.step(int(a))
            obs.append(o); rewards.append(r); dones.append(d); infos.append(i)
        return np.stack(obs), np.array(rewards, dtype=np.float32), np.array(dones), infos

    @property
    def done(self) -> bool:
        return all(e.done for e in self.envs)


def episode_seed(base_seed: int, episode: int) -> int:
    """Deterministic per-episode seed (stable across policies so comparisons share opportunity streams)."""
    return (int(base_seed) * 1_000_003 + int(episode) * 7919) % (2 ** 31 - 1)
