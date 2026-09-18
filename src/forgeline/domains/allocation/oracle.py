"""HINDSIGHT ORACLE / UPPER BOUND — not a deployable policy.

For a seeded episode the oracle sees the entire opportunity stream ``(v_t, c_t, q_t)`` in advance and solves the
expected-value fractional knapsack::

    maximise   Σ_t v_t q_t g(u_t)      subject to   Σ_t u_t c_t ≤ B,   0 ≤ u_t ≤ max multiplier

with ``g(u) = 1 − e^{−u}`` (concave). Because ``g`` is concave the optimum is greedy on the marginal value per unit
cost: fill intensity where ``v_t q_t g'(u_t) / c_t`` is highest until the budget binds. The KKT solution is
``u_t = clip(ln(v_t q_t / (λ c_t)), 0, u_max)`` for the dual price λ that exhausts the budget, found by bisection.

Three relaxations make this a strict upper bound on any realisable policy's expected return:

1. it knows the future (hindsight);
2. it ignores pressure, which can only raise costs and lower response rates;
3. it allocates continuous intensities rather than the discrete multiplier grid.

Outcome noise is replaced by its expectation, so the oracle bounds the *expected* return of the episode.
Regret is reported against this bound and against the concrete baselines in ``benchmark.py``.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from forgeline.domains.allocation.env import AllocationEnvConfig, BudgetedAllocationEnv, Opportunity


def hindsight_upper_bound(opps: List[Opportunity], budget: float, u_max: float) -> Dict[str, float]:
    if not opps:
        return {"oracle_value": 0.0, "oracle_spend": 0.0, "dual_price": 0.0, "oracle_mean_intensity": 0.0}
    v = np.array([o.value * o.quality for o in opps], dtype=float)  # expected value at full response
    c = np.array([o.cost for o in opps], dtype=float)
    u_free = np.full(len(v), u_max)  # unconstrained optimum: saturate everything
    if float((u_free * c).sum()) <= budget:
        u = u_free
        lam = 0.0
    else:
        def alloc(lam: float) -> np.ndarray:
            with np.errstate(divide="ignore"):
                x = np.log(np.maximum(v, 1e-300) / (lam * c))
            return np.clip(x, 0.0, u_max)

        lo, hi = 1e-9, float((v / c).max()) + 1.0  # at hi every u_t = 0
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            if float((alloc(mid) * c).sum()) > budget:
                lo = mid
            else:
                hi = mid
        lam = hi
        u = alloc(lam)
    value = float((v * (1.0 - np.exp(-u))).sum())
    return {"oracle_value": value, "oracle_spend": float((u * c).sum()), "dual_price": float(lam),
            "oracle_mean_intensity": float(u.mean())}


def oracle_for_seed(env_cfg: AllocationEnvConfig, seed: int) -> Dict[str, float]:
    env = BudgetedAllocationEnv(env_cfg)
    env.reset(seed)
    return hindsight_upper_bound(env.opportunity_stream(), env_cfg.budget, max(env_cfg.multipliers))


def oracle_values(env_cfg: AllocationEnvConfig, seeds: List[int]) -> np.ndarray:
    return np.array([oracle_for_seed(env_cfg, s)["oracle_value"] for s in seeds])
