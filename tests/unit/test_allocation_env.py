"""BudgetedAllocationEnv: transitions, rewards, budget conservation, termination, reproducibility, endogenous feedback."""

import math

import numpy as np
import pytest

from forgeline.core.errors import ConfigError
from forgeline.domains.allocation.env import OBS_DIM, AllocationEnvConfig, BudgetedAllocationEnv, VectorEnv, episode_seed, response_gain
from forgeline.domains.allocation.oracle import hindsight_upper_bound, oracle_for_seed
from forgeline.domains.allocation.policies import DualPacingPolicy, ThresholdPacingPolicy
from forgeline.domains.allocation.rollout import aggregate_metrics, episode_metrics, run_episodes, seeds_for


def test_transition_matches_documented_equations():
    cfg = AllocationEnvConfig(horizon=6, budget=100.0)
    env = BudgetedAllocationEnv(cfg)
    env.reset(3)
    o = env.opportunities[0]
    u = env.outcome_uniforms()
    info = env.transition(2)  # medium: multiplier 1.0, pressure 0 → effective cost = base cost
    assert info.effective_cost == pytest.approx(o.cost)
    assert info.spend == pytest.approx(o.cost)
    assert info.intensity == pytest.approx(1.0)
    rate = o.quality * response_gain(1.0)
    assert info.response_rate == pytest.approx(rate)
    assert info.outcome == int(u[0] < rate)
    assert info.reward == pytest.approx(info.outcome * o.value)
    assert env.pressure == pytest.approx(cfg.pressure_gain * 1.0)
    # second step: pressure raises the cost and lowers the response rate
    o1 = env.opportunities[1]
    info1 = env.transition(3)
    p = cfg.pressure_gain
    assert info1.effective_cost == pytest.approx(o1.cost * (1 + cfg.pressure_cost_coef * p))
    assert info1.response_rate == pytest.approx(o1.quality * response_gain(1.5) * (1 - cfg.pressure_response_coef * p))
    assert env.pressure == pytest.approx(cfg.pressure_decay * p + cfg.pressure_gain * 1.5)


def test_budget_conservation_and_clipping():
    cfg = AllocationEnvConfig(horizon=10, budget=2.0)
    env = BudgetedAllocationEnv(cfg)
    env.reset(0)
    total = 0.0
    for _ in range(10):
        info = env.transition(3)
        total += info.spend
        assert info.spend <= info.remaining_before + 1e-12
        assert info.remaining_after >= -1e-12
    assert total == pytest.approx(cfg.budget)  # fully spent, never overspent
    assert any(i.clipped for i in env.history)
    assert env.history[-1].spend == 0.0 and env.history[-1].reward == 0.0


def test_horizon_termination_and_invalid_use():
    env = BudgetedAllocationEnv(AllocationEnvConfig(horizon=3))
    env.reset(1)
    for k in range(3):
        _, _, done, _ = env.step(1)
        assert done == (k == 2)
    with pytest.raises(ConfigError):
        env.transition(0)
    env.reset(1)
    with pytest.raises(ConfigError):
        env.transition(7)


def test_seeded_reproducibility_and_observation_shape():
    cfg = AllocationEnvConfig()
    a, b = BudgetedAllocationEnv(cfg), BudgetedAllocationEnv(cfg)
    oa, ob = a.reset(42), b.reset(42)
    assert oa.shape == (OBS_DIM,) and np.array_equal(oa, ob)
    for act in (1, 3, 0, 2):
        ia, ib = a.transition(act), b.transition(act)
        assert ia == ib
    assert episode_seed(1, 2) != episode_seed(1, 3) and episode_seed(1, 2) == episode_seed(1, 2)


def test_endogenous_feedback_changes_future_state_under_same_seed():
    """Same seed and identical actions after step 5 → different costs/rates because early actions differed."""
    cfg = AllocationEnvConfig(horizon=12, budget=1e6)
    aggressive, cautious = BudgetedAllocationEnv(cfg), BudgetedAllocationEnv(cfg)
    aggressive.reset(9); cautious.reset(9)
    for _ in range(5):
        aggressive.transition(3); cautious.transition(0)
    assert aggressive.pressure > cautious.pressure == 0.0
    ia, ic = aggressive.transition(2), cautious.transition(2)
    assert ia.opportunity == ic.opportunity  # exogenous stream identical
    assert ia.effective_cost > ic.effective_cost
    assert ia.response_rate < ic.response_rate
    # and the induced future-state distribution differs across many seeds
    diffs = []
    for s in range(30):
        e1, e2 = BudgetedAllocationEnv(cfg), BudgetedAllocationEnv(cfg)
        e1.reset(s); e2.reset(s)
        for _ in range(6):
            e1.transition(3); e2.transition(1)
        diffs.append(e1.transition(2).effective_cost - e2.transition(2).effective_cost)
    assert min(diffs) > 0


def test_pressure_off_makes_dynamics_exogenous():
    cfg = AllocationEnvConfig(horizon=8, budget=1e6, pressure_gain=0.0)
    e1, e2 = BudgetedAllocationEnv(cfg), BudgetedAllocationEnv(cfg)
    e1.reset(4); e2.reset(4)
    for _ in range(4):
        e1.transition(3); e2.transition(0)
    assert e1.transition(2).effective_cost == pytest.approx(e2.transition(2).effective_cost)


def test_invalid_configuration():
    for bad in ({"horizon": 1}, {"budget": 0}, {"pressure_decay": 1.0}, {"multipliers": [0.5, 1.0]}, {"quality_alpha": 0}):
        with pytest.raises(ConfigError):
            AllocationEnvConfig.from_dict(bad)
    with pytest.raises(ConfigError):
        AllocationEnvConfig.from_dict({"nope": 1})


def test_vector_env_matches_single_env():
    cfg = AllocationEnvConfig(horizon=5)
    v = VectorEnv(cfg, 3)
    obs = v.reset([1, 2, 3])
    single = BudgetedAllocationEnv(cfg); s_obs = single.reset(2)
    assert np.array_equal(obs[1], s_obs)
    o, r, d, infos = v.step([2, 2, 2])
    assert infos[1] == single.transition(2)


def test_oracle_is_an_upper_bound_and_respects_budget():
    cfg = AllocationEnvConfig(horizon=24, budget=8.0)
    seeds = seeds_for(5, 40)
    for pol in (ThresholdPacingPolicy(cfg, 0.5), DualPacingPolicy(cfg)):
        eps = run_episodes(pol, cfg, seeds)
        for ep in eps:
            o = oracle_for_seed(cfg, ep.seed)
            assert o["oracle_spend"] <= cfg.budget + 1e-6
            # realised value is one Bernoulli draw; compare expected value of the realised plan instead
            env = BudgetedAllocationEnv(cfg); env.reset(ep.seed)
            expected = sum(env.opportunities[s.t].value * env.opportunities[s.t].quality * response_gain(s.cost / (env.opportunities[s.t].cost))
                           for s in ep.steps if s.cost > 0)  # ignores pressure, so ≥ the true expectation
            assert expected <= o["oracle_value"] + 1e-6
    # unconstrained case saturates every opportunity
    env = BudgetedAllocationEnv(cfg); env.reset(0)
    ob = hindsight_upper_bound(env.opportunity_stream(), 1e9, 1.5)
    assert ob["dual_price"] == 0.0 and ob["oracle_mean_intensity"] == pytest.approx(1.5)
    assert hindsight_upper_bound([], 1.0, 1.5)["oracle_value"] == 0.0


def test_pacing_metrics():
    cfg = AllocationEnvConfig(horizon=10, budget=5.0)
    eps = run_episodes(ThresholdPacingPolicy(cfg, threshold=0.0, level=3), cfg, seeds_for(1, 5))
    m = episode_metrics(eps[0], cfg)
    assert 0 <= m["utilization"] <= 1 + 1e-9 and m["spend"] <= cfg.budget + 1e-9
    assert abs(sum(m[f"action_frac_{i}"] for i in range(cfg.n_actions)) - 1) < 1e-9
    assert m["value_early"] + m["value_mid"] + m["value_late"] == pytest.approx(m["value"])
    agg = aggregate_metrics(eps, cfg)
    assert agg["n_episodes"] == 5 and "value_ci95_halfwidth" in agg
