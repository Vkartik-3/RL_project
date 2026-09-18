"""OPE estimators verified against hand-computed reference values on tiny trajectories."""

import math

import numpy as np
import pytest

from forgeline.core.errors import DatasetError
from forgeline.domains.allocation.env import AllocationEnvConfig
from forgeline.domains.allocation.ope import (
    OPEConfig, OPEError, bootstrap_ci, cumulative_weights, dr_estimate, effective_sample_size, evaluate_policy, ips_estimate,
    pdis_estimate, snips_estimate,
)
from forgeline.domains.allocation.rollout import Episode, LoggedStep, read_episodes, validate_step, write_episodes


def _step(ep, t, action, probs, reward, terminal=False):
    return LoggedStep(episode_id=ep, seed=0, t=t, obs=[0.0] * 14, history_summary={}, action=action, propensity=probs[action],
                      action_probs=list(probs), reward=reward, cost=0.5, next_obs=[0.0] * 14, cum_spend=0.0, cum_reward=0.0,
                      remaining_budget=1.0, terminal=terminal, clipped=False, pressure=0.0)


def _two_episode_log():
    # behaviour: uniform over 2 actions; episode A takes (0, 1) rewards (1, 2); episode B takes (1, 1) rewards (0, 4)
    b = [0.5, 0.5]
    ea = Episode("a", 0, [_step("a", 0, 0, b, 1.0), _step("a", 1, 1, b, 2.0, True)])
    eb = Episode("b", 1, [_step("b", 0, 1, b, 0.0), _step("b", 1, 1, b, 4.0, True)])
    # target: action 1 with prob 0.8 everywhere
    tp = [np.array([[0.2, 0.8], [0.2, 0.8]]), np.array([[0.2, 0.8], [0.2, 0.8]])]
    return [ea, eb], tp


def test_core_formulas_against_hand_values():
    # ratios episode A: 0.2/0.5=0.4 then 0.8/0.5=1.6 → w = [0.4, 0.64]; episode B: 1.6, 1.6 → w = [1.6, 2.56]
    wa, wb = cumulative_weights(np.array([0.4, 1.6])), cumulative_weights(np.array([1.6, 1.6]))
    assert np.allclose(wa, [0.4, 0.64]) and np.allclose(wb, [1.6, 2.56])
    wf = np.array([0.64, 2.56]); G = np.array([3.0, 4.0])
    assert ips_estimate(wf, G) == pytest.approx((0.64 * 3 + 2.56 * 4) / 2)  # 6.08
    assert snips_estimate(wf, G) == pytest.approx((0.64 * 3 + 2.56 * 4) / (0.64 + 2.56))  # 3.8
    assert pdis_estimate([wa, wb], [np.array([1.0, 2.0]), np.array([0.0, 4.0])], 1.0) == pytest.approx(((0.4 * 1 + 0.64 * 2) + (1.6 * 0 + 2.56 * 4)) / 2)  # 5.96
    # DR with Q̂ ≡ 1 and V̂ ≡ 1: per episode Σ_t [w_t (r_t − 1) + w_{t−1}·1]
    q = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]; v = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
    dr_a = 0.4 * (1 - 1) + 1.0 * 1 + 0.64 * (2 - 1) + 0.4 * 1  # 2.04
    dr_b = 1.6 * (0 - 1) + 1.0 * 1 + 2.56 * (4 - 1) + 1.6 * 1  # 8.68
    assert dr_estimate([wa, wb], [np.array([1.0, 2.0]), np.array([0.0, 4.0])], q, v, 1.0) == pytest.approx((dr_a + dr_b) / 2)
    assert effective_sample_size(wf) == pytest.approx((0.64 + 2.56) ** 2 / (0.64 ** 2 + 2.56 ** 2))
    assert effective_sample_size(np.array([1.0, 1.0, 1.0])) == pytest.approx(3.0)
    # clipping caps the running product
    assert np.allclose(cumulative_weights(np.array([2.0, 2.0, 2.0]), max_weight=3.0), [2.0, 3.0, 3.0])
    # discounting
    assert pdis_estimate([np.array([1.0, 1.0])], [np.array([1.0, 1.0])], 0.5) == pytest.approx(1.5)


def test_evaluate_policy_end_to_end_matches_hand_values():
    eps, tp = _two_episode_log()
    cfg = AllocationEnvConfig(multipliers=(0.0, 1.0))
    rep = evaluate_policy(eps, tp, cfg, OPEConfig(n_bootstrap=50, min_ess=0.0, min_ess_fraction=0.0))
    assert rep.estimates["ips"].value == pytest.approx(6.08)
    assert rep.estimates["snips"].value == pytest.approx(3.8)
    assert rep.estimates["pdis"].value == pytest.approx(5.96)
    assert rep.diagnostics.ess == pytest.approx((0.64 + 2.56) ** 2 / (0.64 ** 2 + 2.56 ** 2))
    assert rep.diagnostics.max_weight == pytest.approx(2.56) and rep.diagnostics.unsupported_fraction == 0.0
    assert rep.logged_value == pytest.approx(3.5)
    for e in rep.estimates.values():
        assert e.ci_low <= e.ci_high and math.isfinite(e.std_error)
    # identical target and behaviour → every estimator equals the logged value, ESS = n
    same = evaluate_policy(eps, [np.full((2, 2), 0.5)] * 2, cfg, OPEConfig(n_bootstrap=20, min_ess=0.0, min_ess_fraction=0.0))
    for k in ("ips", "snips", "pdis"):
        assert same.estimates[k].value == pytest.approx(3.5)
    assert same.diagnostics.ess == pytest.approx(2.0)


def test_bootstrap_ci_behaviour():
    vals = np.array([1.0] * 50)
    lo, hi, se = bootstrap_ci(vals, 100, 0)
    assert lo == hi == 1.0 and se == 0.0
    rng = np.random.default_rng(0)
    vals = rng.normal(0, 1, 400)
    lo, hi, se = bootstrap_ci(vals, 300, 1)
    assert lo < vals.mean() < hi and 0.03 < se < 0.08
    lo2, hi2, _ = bootstrap_ci(vals, 300, 1)
    assert (lo, hi) == (lo2, hi2)  # seeded


def test_support_and_validity_errors():
    eps, tp = _two_episode_log()
    cfg = AllocationEnvConfig(multipliers=(0.0, 1.0))
    # target mass on an action the behaviour never takes
    eps[0].steps[0].action_probs = [1.0, 0.0]; eps[0].steps[0].propensity = 1.0
    with pytest.raises(OPEError, match="unsupported"):
        evaluate_policy(eps, tp, cfg, OPEConfig(n_bootstrap=10))
    # tolerated when max_unsupported allows it
    evaluate_policy(eps, tp, cfg, OPEConfig(n_bootstrap=10, max_unsupported=0.5, min_ess=0.0, min_ess_fraction=0.0))
    eps, tp = _two_episode_log()
    eps[1].steps[1].propensity = 0.0
    with pytest.raises(OPEError, match="propensities"):
        evaluate_policy(eps, tp, cfg, OPEConfig(n_bootstrap=10))
    eps, tp = _two_episode_log()
    tp[0] = np.array([[0.7, 0.7], [0.2, 0.8]])
    with pytest.raises(OPEError, match="valid distributions"):
        evaluate_policy(eps, tp, cfg, OPEConfig(n_bootstrap=10))
    with pytest.raises(OPEError, match="one target"):
        evaluate_policy(eps, tp[:1], cfg)


def test_weak_support_warnings_and_zero_target():
    eps, _ = _two_episode_log()
    cfg = AllocationEnvConfig(multipliers=(0.0, 1.0))
    det = [np.array([[1.0, 0.0], [1.0, 0.0]])] * 2  # deterministic target: never matches full trajectories
    rep = evaluate_policy(eps, det, cfg, OPEConfig(n_bootstrap=10))
    assert rep.estimates["ips"].value == 0.0 and "snips" not in rep.estimates
    assert rep.diagnostics.zero_target_fraction == pytest.approx(0.75)
    assert any("effective sample size" in w for w in rep.diagnostics.warnings)


def test_weight_explosion_is_refused():
    T = 400
    steps = [_step("x", t, 1, [0.01, 0.99] if False else [1e-3, 1 - 1e-3], 1.0, t == T - 1) for t in range(T)]
    for s in steps:
        s.action = 0; s.propensity = 1e-3
    ep = Episode("x", 0, steps)
    tp = [np.tile([[1.0, 0.0]], (T, 1))]
    with pytest.raises(OPEError, match="non-finite"):
        evaluate_policy([ep], tp, AllocationEnvConfig(multipliers=(0.0, 1.0)), OPEConfig(n_bootstrap=5))


def test_logged_schema_validation_roundtrip(tmp_path):
    eps, _ = _two_episode_log()
    p = write_episodes(eps, tmp_path / "log.jsonl")
    back = read_episodes(p)
    assert [e.episode_id for e in back] == ["a", "b"] and back[0].steps[1].reward == 2.0
    bad = _step("c", 0, 0, [0.5, 0.5], 1.0, True); bad.propensity = 0.3
    with pytest.raises(DatasetError, match="does not match"):
        validate_step(bad)
    bad.propensity = 0.0
    with pytest.raises(DatasetError, match="propensity"):
        validate_step(bad)
    (tmp_path / "bad.jsonl").write_text('{"schema_version": 99}\n')
    with pytest.raises(DatasetError, match="schema_version"):
        read_episodes(tmp_path / "bad.jsonl")
    write_episodes([Episode("d", 0, [_step("d", 0, 0, [0.5, 0.5], 1.0, False)])], tmp_path / "open.jsonl")
    with pytest.raises(DatasetError, match="not terminated"):
        read_episodes(tmp_path / "open.jsonl")
