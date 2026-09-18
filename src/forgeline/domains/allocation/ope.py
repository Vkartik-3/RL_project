"""Offline (counterfactual) policy evaluation for logged allocation trajectories.

Notation: an episode has decisions ``t = 0..T−1`` with logged behaviour propensities ``b_t = π_b(a_t | h_t)`` and
target probabilities ``e_t = π_e(a_t | h_t)`` (computed by replaying the target policy over the logged history).
Per-decision importance ratios ``ρ_t = e_t / b_t`` and cumulative products ``w_t = Π_{k≤t} ρ_k``. Returns are
undiscounted sums of rewards (optionally ``γ``-discounted).

Estimators (``n`` episodes, ``G_i = Σ_t γ^t r_{i,t}``):

* **IPS** (trajectory importance sampling): ``V̂ = (1/n) Σ_i w_{i,T−1} G_i``
* **SNIPS** (self-normalised): ``V̂ = Σ_i w_{i,T−1} G_i / Σ_i w_{i,T−1}``
* **PDIS** (per-decision IS): ``V̂ = (1/n) Σ_i Σ_t γ^t w_{i,t} r_{i,t}``
* **DR** (per-decision doubly robust, Jiang & Li 2016):
  ``V̂ = (1/n) Σ_i Σ_t γ^t [ w_{i,t} (r_{i,t} − Q̂(h_{i,t}, a_{i,t})) + w_{i,t−1} V̂_e(h_{i,t}) ]``
  with ``V̂_e(h) = Σ_a π_e(a|h) Q̂(h, a)`` and ``w_{i,−1} = 1``. ``Q̂`` is a ridge regression of the logged
  reward-to-go on ``[obs, onehot(a)]`` fitted on the same log (a control variate; the estimator stays consistent for
  any Q̂ when the propensities are correct).

Importance weights may be clipped at ``max_weight`` (biased but variance-reducing; reported as a separate estimate).

Diagnostics: effective sample size ``ESS = (Σ w)² / Σ w²`` of the trajectory weights, maximum weight, the
**unsupported fraction** (logged decisions where the target puts probability on an action the behaviour policy could
not have taken, ``π_b(a) = 0 < π_e(a)`` — importance sampling is invalid there), the **zero-target fraction** (logged
actions with ``π_e = 0``; these only zero the weight and shrink the ESS), the fraction of trajectories whose weights
were clipped, and mean/min overlap ``Σ_a min(π_e, π_b)``. Estimates are refused (``OPEError``) for zero/invalid
behaviour propensities, non-finite values, or an unsupported fraction above ``max_unsupported``; they are returned
with ``warnings`` when support is weak (ESS below ``min_ess`` or ESS/n below ``min_ess_fraction``).

Confidence intervals: percentile bootstrap over episodes (``n_bootstrap`` resamples, seeded).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from forgeline.core.errors import ForgelineError
from forgeline.domains.allocation.env import AllocationEnvConfig
from forgeline.domains.allocation.policies import NeuralPolicy, Policy, StepContext
from forgeline.domains.allocation.rollout import Episode


class OPEError(ForgelineError):
    hint = "Check logged propensities and target-policy support; see the diagnostics in the message."


@dataclass
class OPEConfig:
    gamma: float = 1.0
    max_weight: Optional[float] = None  # clip cumulative importance weights
    min_ess: float = 10.0
    min_ess_fraction: float = 0.05
    max_unsupported: float = 0.0  # fraction of logged decisions with zero target probability tolerated
    n_bootstrap: int = 500
    bootstrap_seed: int = 0
    ridge: float = 1e-2


@dataclass
class OPEEstimate:
    estimator: str
    value: float
    ci_low: float
    ci_high: float
    std_error: float


@dataclass
class OPEDiagnostics:
    n_episodes: int
    n_decisions: int
    ess: float
    ess_fraction: float
    max_weight: float
    mean_weight: float
    unsupported_fraction: float
    zero_target_fraction: float
    clipped_fraction: float
    overlap_mean: float
    overlap_min: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class OPEReport:
    estimates: Dict[str, OPEEstimate]
    diagnostics: OPEDiagnostics
    logged_value: float  # on-policy mean return of the behaviour data
    target_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {"target": self.target_name, "logged_value": self.logged_value,
                "estimates": {k: dataclasses.asdict(v) for k, v in self.estimates.items()},
                "diagnostics": dataclasses.asdict(self.diagnostics)}


# ─────────────────────────────────────────────────────────────────────────────
#  Target-policy probabilities on logged histories
# ─────────────────────────────────────────────────────────────────────────────

def target_probabilities(policy: Policy, episodes: Sequence[Episode], env_cfg: AllocationEnvConfig) -> List[np.ndarray]:
    """Replay each logged episode through ``policy`` (feeding it the logged actions/rewards so stateful policies see the
    same history the behaviour policy produced) and return per-episode ``[T, A]`` target action distributions."""
    out: List[np.ndarray] = []
    for ep in episodes:
        policy.start(1)
        probs = np.zeros((len(ep.steps), env_cfg.n_actions))
        for k, s in enumerate(ep.steps):
            obs = np.asarray([s.obs], dtype=np.float32)
            ctx = [StepContext(s.t, env_cfg.horizon, s.remaining_budget + s.cost, env_cfg.budget, s.cum_spend - s.cost,
                               _value_from_obs(s.obs, env_cfg), _cost_from_obs(s.obs, env_cfg), s.obs[5])]
            if isinstance(policy, NeuralPolicy) and policy.is_sequence:
                policy.hist.set_obs(obs)
            probs[k] = policy.action_probs(obs, ctx)[0]
            policy.observe(np.array([s.action]), np.array([s.reward]), np.array([s.cost]), np.array([s.remaining_budget]))
        out.append(probs)
    return out


def _value_from_obs(obs: Sequence[float], cfg: AllocationEnvConfig) -> float:
    return (obs[3] + 1.0) * cfg.value_mean


def _cost_from_obs(obs: Sequence[float], cfg: AllocationEnvConfig) -> float:
    return (obs[4] + 1.0) * cfg.cost_mean


# ─────────────────────────────────────────────────────────────────────────────
#  Core estimators (pure functions on arrays; tested against hand-computed values)
# ─────────────────────────────────────────────────────────────────────────────

def cumulative_weights(ratios: np.ndarray, max_weight: Optional[float] = None) -> np.ndarray:
    """``w_t = Π_{k≤t} ρ_k`` for a ``[T]`` ratio vector, optionally clipped after each product."""
    w = np.empty_like(ratios, dtype=float)
    acc = 1.0
    with np.errstate(over="ignore"):  # overflow is reported by the caller as a non-finite estimate
        for t, r in enumerate(ratios):
            acc = float(acc * r)
            if max_weight is not None:
                acc = min(acc, max_weight)
            w[t] = acc
    return w


def ips_estimate(weights_final: np.ndarray, returns: np.ndarray) -> float:
    return float(np.mean(weights_final * returns))


def snips_estimate(weights_final: np.ndarray, returns: np.ndarray) -> float:
    denom = float(np.sum(weights_final))
    if denom <= 0:
        raise OPEError("SNIPS denominator is zero (no trajectory has positive weight)")
    return float(np.sum(weights_final * returns) / denom)


def pdis_estimate(weights: List[np.ndarray], rewards: List[np.ndarray], gamma: float) -> float:
    vals = []
    for w, r in zip(weights, rewards):
        disc = gamma ** np.arange(len(r))
        vals.append(float(np.sum(disc * w * r)))
    return float(np.mean(vals))


def dr_estimate(weights: List[np.ndarray], rewards: List[np.ndarray], q_taken: List[np.ndarray],
                v_target: List[np.ndarray], gamma: float) -> float:
    vals = []
    for w, r, q, v in zip(weights, rewards, q_taken, v_target):
        T = len(r)
        disc = gamma ** np.arange(T)
        w_prev = np.concatenate([[1.0], w[:-1]])
        vals.append(float(np.sum(disc * (w * (r - q) + w_prev * v))))
    return float(np.mean(vals))


def effective_sample_size(weights_final: np.ndarray) -> float:
    s2 = float(np.sum(weights_final ** 2))
    return float(np.sum(weights_final) ** 2 / s2) if s2 > 0 else 0.0


def bootstrap_ci(per_episode_values: np.ndarray, n_bootstrap: int, seed: int, level: float = 0.95,
                 statistic: Callable[[np.ndarray], float] = np.mean) -> tuple:
    """Percentile bootstrap over episodes for a statistic of per-episode contributions."""
    rng = np.random.default_rng(seed)
    n = len(per_episode_values)
    stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        stats[b] = statistic(per_episode_values[idx])
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha)), float(stats.std(ddof=1)) if n_bootstrap > 1 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Q̂ regression for DR
# ─────────────────────────────────────────────────────────────────────────────

def _features(obs: np.ndarray, action: int, n_actions: int) -> np.ndarray:
    oh = np.zeros(n_actions); oh[action] = 1.0
    return np.concatenate([obs, oh, [1.0]])


def fit_q_ridge(episodes: Sequence[Episode], n_actions: int, gamma: float, ridge: float) -> np.ndarray:
    """Least-squares ``Q̂(obs, a) ≈ reward-to-go`` with ridge penalty; returns the weight vector."""
    X, y = [], []
    for ep in episodes:
        r = np.array([s.reward for s in ep.steps]); T = len(r)
        rtg = np.array([float(np.sum(gamma ** np.arange(T - t) * r[t:])) for t in range(T)])
        for s, g in zip(ep.steps, rtg):
            X.append(_features(np.asarray(s.obs), s.action, n_actions)); y.append(g)
    X = np.asarray(X); y = np.asarray(y)
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + ridge * np.eye(d), X.T @ y)


def q_values(theta: np.ndarray, obs: np.ndarray, n_actions: int) -> np.ndarray:
    return np.array([float(_features(obs, a, n_actions) @ theta) for a in range(n_actions)])


# ─────────────────────────────────────────────────────────────────────────────
#  Full evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_policy(episodes: Sequence[Episode], target_probs: Sequence[np.ndarray], env_cfg: AllocationEnvConfig,
                    config: Optional[OPEConfig] = None, target_name: str = "target") -> OPEReport:
    cfg = config or OPEConfig()
    if len(episodes) == 0:
        raise OPEError("no logged episodes")
    if len(target_probs) != len(episodes):
        raise OPEError("one target probability matrix per episode is required")
    A = env_cfg.n_actions
    ratios, rewards, tp_list, overlaps = [], [], [], []
    n_dec = unsupported = zero_target = 0
    for ep, tp in zip(episodes, target_probs):
        T = len(ep.steps)
        if tp.shape != (T, A):
            raise OPEError(f"target probabilities for episode {ep.episode_id} have shape {tp.shape}, expected {(T, A)}")
        if not np.all(np.isfinite(tp)) or np.any(tp < -1e-12) or np.any(np.abs(tp.sum(1) - 1) > 1e-6):
            raise OPEError(f"target probabilities for episode {ep.episode_id} are not valid distributions")
        b = np.array([s.propensity for s in ep.steps]); a = np.array([s.action for s in ep.steps])
        if np.any(~np.isfinite(b)) or np.any(b <= 0):
            raise OPEError(f"episode {ep.episode_id} has zero or invalid behaviour propensities")
        e = tp[np.arange(T), a]
        bp = np.array([s.action_probs for s in ep.steps])
        unsupported += int(np.sum(np.any((bp <= 0) & (tp > 0), axis=1)))
        zero_target += int(np.sum(e <= 0)); n_dec += T
        ratios.append(e / b); rewards.append(np.array([s.reward for s in ep.steps])); tp_list.append(tp)
        overlaps.append(np.minimum(tp, bp).sum(1))
    if n_dec and unsupported / n_dec > cfg.max_unsupported:
        raise OPEError(f"target policy is unsupported by the log: at {unsupported}/{n_dec} logged decisions it puts probability on an "
                       f"action the behaviour policy could not take (tolerance {cfg.max_unsupported:.2%})")
    weights = [cumulative_weights(r, None) for r in ratios]
    w_final = np.array([w[-1] for w in weights])
    returns = np.array([float(np.sum(cfg.gamma ** np.arange(len(r)) * r)) for r in rewards])
    if not np.all(np.isfinite(w_final)):
        raise OPEError("importance weights overflowed to non-finite values (catastrophic weight explosion)")
    ess = effective_sample_size(w_final)
    ov = np.concatenate(overlaps)
    diag = OPEDiagnostics(len(episodes), n_dec, ess, ess / len(episodes), float(w_final.max()), float(w_final.mean()),
                          unsupported / max(n_dec, 1), zero_target / max(n_dec, 1), 0.0, float(ov.mean()), float(ov.min()))
    if zero_target / max(n_dec, 1) > 0.5:
        diag.warnings.append(f"target gives zero probability to {zero_target / n_dec:.0%} of logged actions: trajectory weights are mostly zero")
    if ess < cfg.min_ess:
        diag.warnings.append(f"effective sample size {ess:.1f} < {cfg.min_ess}: estimates unreliable")
    if ess / len(episodes) < cfg.min_ess_fraction:
        diag.warnings.append(f"ESS fraction {ess / len(episodes):.3f} < {cfg.min_ess_fraction}: weak overlap between behaviour and target")
    if diag.max_weight > 100.0:
        diag.warnings.append(f"maximum trajectory weight {diag.max_weight:.1f} dominates the estimate")

    estimates: Dict[str, OPEEstimate] = {}

    def add(name: str, per_episode: np.ndarray, value: float) -> None:
        if not math.isfinite(value):
            raise OPEError(f"{name} estimate is non-finite")
        lo, hi, se = bootstrap_ci(per_episode, cfg.n_bootstrap, cfg.bootstrap_seed)
        estimates[name] = OPEEstimate(name, value, lo, hi, se)

    add("ips", w_final * returns, ips_estimate(w_final, returns))
    # SNIPS: bootstrap the ratio statistic
    if float(np.sum(w_final)) > 0:
        snips_val = snips_estimate(w_final, returns)
        rng = np.random.default_rng(cfg.bootstrap_seed)
        n = len(returns)
        boots = []
        for _ in range(cfg.n_bootstrap):
            i = rng.integers(0, n, size=n)
            boots.append(snips_estimate(w_final[i], returns[i]) if float(np.sum(w_final[i])) > 0 else snips_val)
        boots = np.array(boots)
        estimates["snips"] = OPEEstimate("snips", snips_val, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), float(boots.std(ddof=1)))
    else:
        diag.warnings.append("SNIPS unavailable: every trajectory weight is zero (target never agrees with the full logged trajectory)")
    pd = np.array([float(np.sum(cfg.gamma ** np.arange(len(r)) * w * r)) for w, r in zip(weights, rewards)])
    add("pdis", pd, float(pd.mean()))
    theta = fit_q_ridge(episodes, A, cfg.gamma, cfg.ridge)
    q_taken, v_tgt = [], []
    for ep, tp in zip(episodes, tp_list):
        qs = np.array([q_values(theta, np.asarray(s.obs), A) for s in ep.steps])  # [T, A]
        q_taken.append(qs[np.arange(len(ep.steps)), [s.action for s in ep.steps]])
        v_tgt.append((tp * qs).sum(1))
    dr_per = []
    for w, r, q, v in zip(weights, rewards, q_taken, v_tgt):
        disc = cfg.gamma ** np.arange(len(r)); w_prev = np.concatenate([[1.0], w[:-1]])
        dr_per.append(float(np.sum(disc * (w * (r - q) + w_prev * v))))
    add("dr", np.array(dr_per), float(np.mean(dr_per)))
    if cfg.max_weight is not None:
        cw = [cumulative_weights(r, cfg.max_weight) for r in ratios]
        cwf = np.array([w[-1] for w in cw])
        diag.clipped_fraction = float(np.mean([np.any(np.abs(a - b) > 1e-12) for a, b in zip(cw, weights)]))
        add("ips_clipped", cwf * returns, ips_estimate(cwf, returns))
        pdc = np.array([float(np.sum(cfg.gamma ** np.arange(len(r)) * w * r)) for w, r in zip(cw, rewards)])
        add("pdis_clipped", pdc, float(pdc.mean()))
        drc = []
        for w, r, q, v in zip(cw, rewards, q_taken, v_tgt):
            disc = cfg.gamma ** np.arange(len(r)); w_prev = np.concatenate([[1.0], w[:-1]])
            drc.append(float(np.sum(disc * (w * (r - q) + w_prev * v))))
        add("dr_clipped", np.array(drc), float(np.mean(drc)))
    return OPEReport(estimates, diag, float(returns.mean()), target_name)


def evaluate_target_policy(policy: Policy, episodes: Sequence[Episode], env_cfg: AllocationEnvConfig,
                           config: Optional[OPEConfig] = None, target_name: Optional[str] = None) -> OPEReport:
    tp = target_probabilities(policy, episodes, env_cfg)
    return evaluate_policy(episodes, tp, env_cfg, config, target_name or policy.name)
