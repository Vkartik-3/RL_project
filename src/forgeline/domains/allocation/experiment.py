"""Controlled offline experiments: simulated A/B tests, shadow evaluation and lifecycle gating.

Everything here runs in the simulator. Wording that applies: *simulated A/B experiment*, *controlled offline policy
experiment*. Nothing here touches live traffic.

A/B design
----------
Episodes (identified by their seed) are assigned to the **incumbent** or **challenger** arm by the same deterministic
hash used for serving routing (``stable_bucket(salt:episode_id) mod 10_000 < challenger_percent × 100``), so the
assignment is reproducible and independent of the policies. Each arm runs its policy on its own episodes; per-episode
metrics (value, spend, pacing error, violations, regret against the hindsight oracle) are aggregated per arm.

Analysis
--------
* difference of arm means for every metric, with a percentile **bootstrap CI** over episodes;
* a **permutation (randomisation) test** for the primary metric: episodes are re-labelled at random ``n_permutations``
  times and the two-sided p-value is the fraction of permuted mean differences at least as extreme as the observed one;
* Welch-style standard error for reference;
* **minimum sample size** warning when either arm has fewer than ``min_episodes`` episodes;
* **guardrails**: the challenger is rejected when the primary metric's CI is unfavourable (the lower bound is below
  ``−tolerance`` for higher-is-better), when utilisation leaves ``utilization_bounds``, when violations or early
  exhaustion increase beyond their allowed deltas, or when any additional guardrail metric regresses.

The resulting :class:`ABResult` exposes ``gate_metrics()``: a flat mapping (``ab/value_delta``, ``ab/value_delta_ci_low``,
``ab/violations_delta``, …) that the existing :class:`PromotionGate` consumes, so promotion decisions stay in the one
lifecycle system.

Shadow evaluation
-----------------
:func:`shadow_evaluate` runs the incumbent on a set of episodes while the candidate proposes an action at every step
without affecting the environment (it sees the incumbent's trajectory, as a shadow deployment would), records the
divergence rate, and reports the candidate's estimated value from OPE on the incumbent's log and from simulator
replay on the same seeds.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from forgeline.core.errors import ConfigError
from forgeline.deployment.feature_flags import stable_bucket
from forgeline.domains.allocation.env import AllocationEnvConfig, VectorEnv
from forgeline.domains.allocation.ope import OPEConfig, evaluate_target_policy
from forgeline.domains.allocation.oracle import oracle_values
from forgeline.domains.allocation.policies import NeuralPolicy, Policy, StepContext
from forgeline.domains.allocation.rollout import Episode, _contexts, episode_metrics, run_episodes, seeds_for

PRIMARY = "value"


@dataclass
class ABConfig:
    n_episodes: int = 400
    challenger_percent: float = 50.0
    salt: str = "forgeline-ab"
    seed: int = 0
    min_episodes: int = 30
    n_bootstrap: int = 1000
    n_permutations: int = 1000
    tolerance: float = 0.0  # allowed regression of the primary metric (absolute)
    utilization_bounds: tuple = (0.5, 1.0)
    max_violation_increase: float = 0.5  # per-episode clipped attempts
    max_exhaustion_increase: float = 0.1  # early-exhaustion rate
    guardrails: Dict[str, float] = field(default_factory=dict)  # metric -> max allowed increase (higher is worse)
    greedy: bool = False

    def validate(self) -> None:
        if self.n_episodes < 2 or not 0 < self.challenger_percent < 100 or self.min_episodes < 2:
            raise ConfigError("invalid A/B configuration")


@dataclass
class ArmResult:
    name: str
    n: int
    metrics: Dict[str, float]
    per_episode: Dict[str, np.ndarray]


@dataclass
class MetricDelta:
    metric: str
    incumbent: float
    challenger: float
    delta: float
    ci_low: float
    ci_high: float
    std_error: float


@dataclass
class ABResult:
    incumbent: ArmResult
    challenger: ArmResult
    deltas: Dict[str, MetricDelta]
    p_value: float
    guardrail_failures: List[str]
    warnings: List[str]
    promote: bool
    elapsed_s: float

    def gate_metrics(self) -> Dict[str, float]:
        out: Dict[str, float] = {"ab/n_incumbent": self.incumbent.n, "ab/n_challenger": self.challenger.n, "ab/p_value": self.p_value,
                                 "ab/guardrail_failures": float(len(self.guardrail_failures))}
        for k, d in self.deltas.items():
            out[f"ab/{k}_delta"] = d.delta; out[f"ab/{k}_delta_ci_low"] = d.ci_low; out[f"ab/{k}_delta_ci_high"] = d.ci_high
            out[f"ab/{k}_challenger"] = d.challenger; out[f"ab/{k}_incumbent"] = d.incumbent
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"incumbent": {"name": self.incumbent.name, "n": self.incumbent.n, "metrics": self.incumbent.metrics},
                "challenger": {"name": self.challenger.name, "n": self.challenger.n, "metrics": self.challenger.metrics},
                "deltas": {k: dataclasses.asdict(v) for k, v in self.deltas.items()}, "p_value": self.p_value,
                "guardrail_failures": self.guardrail_failures, "warnings": self.warnings, "promote": self.promote,
                "elapsed_s": self.elapsed_s}


def assign_arms(seeds: Sequence[int], challenger_percent: float, salt: str) -> Dict[int, str]:
    return {int(s): ("challenger" if stable_bucket(str(s), salt) < challenger_percent * 100 else "incumbent") for s in seeds}


def _arm(name: str, policy: Policy, env_cfg: AllocationEnvConfig, seeds: List[int], oracle: np.ndarray, greedy: bool, rng_seed: int) -> ArmResult:
    eps = run_episodes(policy, env_cfg, seeds, greedy=greedy, rng_seed=rng_seed)
    rows = [episode_metrics(e, env_cfg) for e in eps]
    per = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    per["regret"] = oracle - per["value"]
    return ArmResult(name, len(eps), {k: float(v.mean()) for k, v in per.items()}, per)


def _boot_delta(a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator):
    stats = np.empty(n_boot)
    for i in range(n_boot):
        stats[i] = b[rng.integers(0, len(b), len(b))].mean() - a[rng.integers(0, len(a), len(a))].mean()
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975)), float(stats.std(ddof=1))


def permutation_test(a: np.ndarray, b: np.ndarray, n_permutations: int, rng: np.random.Generator) -> float:
    observed = abs(b.mean() - a.mean())
    pooled = np.concatenate([a, b]); n_a = len(a)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        if abs(pooled[n_a:].mean() - pooled[:n_a].mean()) >= observed - 1e-15:
            count += 1
    return (count + 1) / (n_permutations + 1)


def run_ab_experiment(incumbent: Policy, challenger: Policy, env_cfg: AllocationEnvConfig, config: Optional[ABConfig] = None,
                      incumbent_name: str = "incumbent", challenger_name: str = "challenger") -> ABResult:
    cfg = config or ABConfig()
    cfg.validate()
    t0 = time.time()
    seeds = seeds_for(cfg.seed, cfg.n_episodes)
    arms = assign_arms(seeds, cfg.challenger_percent, cfg.salt)
    inc_seeds = [s for s in seeds if arms[s] == "incumbent"]; ch_seeds = [s for s in seeds if arms[s] == "challenger"]
    warnings: List[str] = []
    if len(inc_seeds) < cfg.min_episodes or len(ch_seeds) < cfg.min_episodes:
        warnings.append(f"arm sizes {len(inc_seeds)}/{len(ch_seeds)} below the minimum {cfg.min_episodes}; decision withheld")
    inc = _arm(incumbent_name, incumbent, env_cfg, inc_seeds, oracle_values(env_cfg, inc_seeds), cfg.greedy, cfg.seed + 1)
    ch = _arm(challenger_name, challenger, env_cfg, ch_seeds, oracle_values(env_cfg, ch_seeds), cfg.greedy, cfg.seed + 2)
    rng = np.random.default_rng(cfg.seed)
    deltas: Dict[str, MetricDelta] = {}
    for k in ("value", "spend", "utilization", "value_per_budget", "pacing_error", "violations", "early_exhaustion", "regret", "unused_budget_frac"):
        lo, hi, se = _boot_delta(inc.per_episode[k], ch.per_episode[k], cfg.n_bootstrap, rng)
        deltas[k] = MetricDelta(k, inc.metrics[k], ch.metrics[k], ch.metrics[k] - inc.metrics[k], lo, hi, se)
    p = permutation_test(inc.per_episode[PRIMARY], ch.per_episode[PRIMARY], cfg.n_permutations, rng)
    fails: List[str] = []
    d = deltas[PRIMARY]
    if d.ci_low < -cfg.tolerance:
        fails.append(f"primary metric CI lower bound {d.ci_low:.4f} < -{cfg.tolerance}")
    if d.delta < 0:
        fails.append(f"primary metric regressed by {d.delta:.4f}")
    lo_u, hi_u = cfg.utilization_bounds
    if not lo_u <= ch.metrics["utilization"] <= hi_u:
        fails.append(f"challenger utilization {ch.metrics['utilization']:.3f} outside [{lo_u}, {hi_u}]")
    if deltas["violations"].delta > cfg.max_violation_increase:
        fails.append(f"violations increased by {deltas['violations'].delta:.3f} > {cfg.max_violation_increase}")
    if deltas["early_exhaustion"].delta > cfg.max_exhaustion_increase:
        fails.append(f"early exhaustion rate increased by {deltas['early_exhaustion'].delta:.3f} > {cfg.max_exhaustion_increase}")
    for metric, max_inc in cfg.guardrails.items():
        if metric not in deltas:
            raise ConfigError(f"unknown guardrail metric {metric!r}")
        if deltas[metric].delta > max_inc:
            fails.append(f"guardrail {metric} increased by {deltas[metric].delta:.4f} > {max_inc}")
    promote = not fails and not warnings
    return ABResult(inc, ch, deltas, p, fails, warnings, promote, time.time() - t0)


# ─────────────────────────────────────────────────────────────────────────────
#  Shadow evaluation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ShadowResult:
    n_episodes: int
    n_decisions: int
    divergence_rate: float
    divergence_by_third: List[float]
    incumbent_value: float
    candidate_replay_value: float
    candidate_ope: Dict[str, Any]
    action_confusion: List[List[int]]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def shadow_evaluate(incumbent: Policy, candidate: Policy, env_cfg: AllocationEnvConfig, seeds: Sequence[int],
                    ope_config: Optional[OPEConfig] = None, rng_seed: int = 0) -> ShadowResult:
    """Incumbent acts; candidate proposes in the shadow of the incumbent's trajectory (never applied)."""
    rng = np.random.default_rng(rng_seed)
    seeds = [int(s) for s in seeds]
    venv = VectorEnv(env_cfg, len(seeds))
    obs = venv.reset(seeds)
    incumbent.start(len(seeds)); candidate.start(len(seeds))
    A = env_cfg.n_actions
    confusion = np.zeros((A, A), dtype=int)
    T = env_cfg.horizon
    diverged = np.zeros(T)
    while not venv.done:
        ctx = _contexts(venv)
        t = venv.envs[0].t
        if isinstance(candidate, NeuralPolicy) and candidate.is_sequence:
            candidate.hist.set_obs(obs)
        cand_probs = candidate.action_probs(obs, ctx)
        cand_actions = cand_probs.argmax(1)
        actions, _ = incumbent.act(obs, ctx, rng)
        for a_i, a_c in zip(actions, cand_actions):
            confusion[a_i, a_c] += 1
        diverged[t] = float(np.mean(actions != cand_actions))
        next_obs, rewards, dones, infos = venv.step(actions)
        spends = np.array([i.spend for i in infos]); rem = np.array([i.remaining_after for i in infos])
        incumbent.observe(actions, rewards, spends, rem); candidate.observe(actions, rewards, spends, rem)
        obs = next_obs
    thirds = [float(np.mean(x)) for x in np.array_split(diverged, 3)]
    inc_eps = run_episodes(incumbent, env_cfg, seeds, rng_seed=rng_seed)
    cand_eps = run_episodes(candidate, env_cfg, seeds, rng_seed=rng_seed)
    try:
        ope = evaluate_target_policy(candidate, inc_eps, env_cfg, ope_config, target_name=candidate.name).to_dict()
    except Exception as exc:  # noqa: BLE001 - deterministic incumbents make OPE impossible; report why
        ope = {"error": f"{type(exc).__name__}: {exc}"}
    return ShadowResult(len(seeds), len(seeds) * T, float(diverged.mean()), thirds,
                        float(np.mean([e.total_reward for e in inc_eps])), float(np.mean([e.total_reward for e in cand_eps])),
                        ope, confusion.tolist())
