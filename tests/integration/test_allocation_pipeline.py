"""Baselines, PPO (stateless and sequence) through the shared Trainer, OPE on real logs, A/B, shadow, lifecycle, CLI."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from forgeline.cli.main import main
from forgeline.core.config import ExperimentManifest
from forgeline.core.errors import ConfigError, ForgelineError
from forgeline.core.lifecycle import RunContext
from forgeline.deployment.candidates import CandidateState, ModelCandidate
from forgeline.deployment.promotion import PromotionGate
from forgeline.deployment.registry import CandidateRegistry
from forgeline.deployment.rollback import rollback
from forgeline.domains.allocation.env import AllocationEnvConfig, BudgetedAllocationEnv
from forgeline.domains.allocation.experiment import ABConfig, assign_arms, permutation_test, run_ab_experiment, shadow_evaluate
from forgeline.domains.allocation.ope import OPEConfig, evaluate_target_policy, target_probabilities
from forgeline.domains.allocation.policies import (
    DualPacingPolicy, EpsilonMixPolicy, GRUActorCritic, HistoryBuffer, NeuralPolicy, PolicyNetConfig, StepContext, ThresholdPacingPolicy,
    build_actor_critic,
)
from forgeline.domains.allocation.ppo import AllocationPPOAlgorithm, AllocationPPOConfig, load_allocation_policy
from forgeline.domains.allocation.rollout import run_episodes, seeds_for, write_episodes
from forgeline.training.common.trainer import Trainer
from forgeline.training.factory import build_algorithm

ROOT = Path(__file__).resolve().parents[2]
ENV = AllocationEnvConfig(horizon=16, budget=8.0)


def _ctx(tmp_path, name, steps, **params):
    m = ExperimentManifest.from_dict({"name": name, "algorithm": "allocation_ppo", "output_dir": str(tmp_path / name),
                                      "trainer": {"max_steps": steps, "batch_size": 1, "gradient_accumulation_steps": 2, "eval_every": 0,
                                                  "log_every": 1, "device": "cpu", "checkpoint": {"save_every": 0},
                                                  "optimizer": {"learning_rate": 3e-3, "decay_only_matrices": False, "weight_decay": 0.0}},
                                      "algorithm_params": {"env": ENV.to_dict(), "episodes_per_rollout": 4, "ppo_epochs": 2, "eval_episodes": 4, **params}})
    return m, RunContext.create(m, metrics_backend="local")


# ── baselines ────────────────────────────────────────────────────────────────

def test_heuristic_and_dual_pacer_behave():
    seeds = seeds_for(0, 40)
    thr = run_episodes(ThresholdPacingPolicy(ENV, threshold=0.6), ENV, seeds)
    dual = run_episodes(DualPacingPolicy(ENV), ENV, seeds)
    assert all(e.total_spend <= ENV.budget + 1e-9 for e in thr + dual)
    # the heuristic never spends ahead of pace (plus slack)
    for e in thr:
        cum = 0.0
        for s in e.steps:
            assert cum <= ENV.budget * (s.t / ENV.horizon + 0.05) + 1e-9 or s.cost == 0.0
            cum += s.cost
    # dual price adapts: overspending raises λ, underspending lowers it
    pol = DualPacingPolicy(ENV, learning_rate=0.5, initial_price=1.0)
    pol.start(2)
    pol.observe(np.array([1, 0]), np.array([0.0, 0.0]), np.array([5.0, 0.0]))
    assert pol.prices()[0] > 1.0 > pol.prices()[1]
    with pytest.raises(ConfigError):
        DualPacingPolicy(ENV, learning_rate=0.0)
    with pytest.raises(ConfigError):
        ThresholdPacingPolicy(ENV, level=0)


def test_dual_pacer_adapts_spend_to_budget():
    tight = AllocationEnvConfig(horizon=16, budget=3.0)
    loose = AllocationEnvConfig(horizon=16, budget=30.0)
    seeds = seeds_for(1, 30)
    t_eps = run_episodes(DualPacingPolicy(tight), tight, seeds); l_eps = run_episodes(DualPacingPolicy(loose), loose, seeds)
    t = np.mean([e.total_spend for e in t_eps]); l = np.mean([e.total_spend for e in l_eps])
    assert 0.5 * 3.0 < t <= 3.0 + 1e-9  # spends a large share of a tight budget without exceeding it
    assert l > 3 * t  # a loose budget lets the price fall and spend rise


# ── neural policies ──────────────────────────────────────────────────────────

def test_gru_policy_consumes_history_window():
    torch.manual_seed(0)
    net = build_actor_critic(4, PolicyNetConfig(kind="gru", hidden=16, window=4))
    pol = NeuralPolicy(net, ENV)
    pol.start(2)
    obs = np.random.default_rng(0).normal(size=(2, 14)).astype(np.float32)
    base_probs = pol.action_probs(obs, [])
    # feed two different histories to the two environments; the current observation is identical
    same_obs = np.repeat(obs[:1], 2, axis=0)
    pol.start(2); pol.hist.set_obs(same_obs)
    pol.observe(np.array([3, 0]), np.array([1.0, 0.0]), np.array([1.5, 0.0]), np.array([4.0, 8.0]))
    pol.hist.set_obs(same_obs)
    pol.observe(np.array([3, 0]), np.array([1.0, 0.0]), np.array([1.5, 0.0]), np.array([2.5, 8.0]))
    p = pol.action_probs(same_obs, [])
    assert not np.allclose(p[0], p[1])  # history changed the decision
    assert pol.hist.lengths.tolist() == [2, 2]
    # window truncation keeps the most recent steps
    for _ in range(5):
        pol.hist.set_obs(same_obs); pol.observe(np.array([1, 1]), np.zeros(2), np.zeros(2), np.ones(2))
    assert pol.hist.lengths.tolist() == [4, 4]
    # stateless policy ignores history
    mlp = NeuralPolicy(build_actor_critic(4, PolicyNetConfig(kind="mlp", hidden=16)), ENV)
    mlp.start(2)
    assert np.allclose(mlp.action_probs(same_obs, [])[0], mlp.action_probs(same_obs, [])[1])


def test_epsilon_mix_has_full_support():
    pol = EpsilonMixPolicy(ThresholdPacingPolicy(ENV, 0.6), 0.2, 4)
    eps = run_episodes(pol, ENV, seeds_for(3, 5))
    for e in eps:
        for s in e.steps:
            assert min(s.action_probs) >= 0.05 - 1e-12 and s.propensity >= 0.05 - 1e-12


# ── PPO through the shared trainer ───────────────────────────────────────────

@pytest.mark.parametrize("kind", ["mlp", "gru"])
def test_ppo_trains_and_checkpoints_reload(tmp_path, kind):
    m, ctx = _ctx(tmp_path, f"ppo-{kind}", 6, policy={"kind": kind, "hidden": 16, "window": 4})
    alg = build_algorithm(m, ctx)
    assert isinstance(alg, AllocationPPOAlgorithm)
    trainer = Trainer(ctx, alg)
    summary = trainer.fit()
    assert np.isfinite(summary["final_loss"]) and alg._rollouts == 3  # 6 steps / 2 PPO epochs
    ctx.close()
    ckpt = Path(m.output_dir) / "checkpoints" / "final"
    policy = load_allocation_policy(ckpt)
    assert policy.net.kind == kind
    obs = BudgetedAllocationEnv(ENV).reset(0)[None]
    policy.start(1)
    a, p = policy.act(obs, [], np.random.default_rng(0))
    assert p.shape == (1, 4) and abs(p.sum() - 1) < 1e-6
    metrics = [json.loads(l) for l in (Path(m.output_dir) / "metrics.jsonl").read_text().splitlines()]
    assert any("train/reward_mean" in r for r in metrics)
    with pytest.raises(ForgelineError):
        load_allocation_policy(ROOT / "configs")  # not a checkpoint


def test_ppo_improves_over_random_init():
    torch.manual_seed(0)
    cfg = AllocationEnvConfig(horizon=16, budget=8.0)
    alg = AllocationPPOAlgorithm(cfg, PolicyNetConfig(kind="mlp", hidden=32), AllocationPPOConfig(episodes_per_rollout=16, ppo_epochs=4, eval_episodes=64))
    before = alg.evaluate(0)["value"]
    m = ExperimentManifest.from_dict({"name": "ppo-learn", "algorithm": "allocation_ppo", "output_dir": "runs/_test_ppo_learn",
                                      "trainer": {"max_steps": 240, "batch_size": 1, "gradient_accumulation_steps": 4, "eval_every": 0, "log_every": 0,
                                                  "device": "cpu", "checkpoint": {"save_every": 0},
                                                  "optimizer": {"learning_rate": 3e-3, "decay_only_matrices": False, "weight_decay": 0.0},
                                                  "schedule": {"name": "constant", "warmup_steps": 0}}})
    ctx = RunContext.create(m, metrics_backend="none")
    Trainer(ctx, alg).fit()
    after = alg.evaluate(0)["value"]
    assert after > before + 0.3


# ── OPE on real logs ─────────────────────────────────────────────────────────

def test_ope_recovers_simulator_value_for_supported_target():
    torch.manual_seed(1)
    cfg = AllocationEnvConfig(horizon=12, budget=6.0)
    target = NeuralPolicy(build_actor_critic(4, PolicyNetConfig(kind="mlp", hidden=16)), cfg)
    behaviour = EpsilonMixPolicy(target, 0.3, 4)
    seeds = seeds_for(2, 300)
    logged = run_episodes(behaviour, cfg, seeds, rng_seed=5)
    truth = np.mean([e.total_reward for e in run_episodes(target, cfg, seeds, rng_seed=6)])
    rep = evaluate_target_policy(target, logged, cfg, OPEConfig(n_bootstrap=200, max_weight=50.0))
    for k in ("snips", "pdis", "dr"):
        e = rep.estimates[k]
        assert abs(e.value - truth) < 0.6, (k, e.value, truth)
    assert rep.diagnostics.unsupported_fraction == 0.0 and rep.diagnostics.ess > 5
    # the GRU target replays the logged history consistently (probabilities are proper distributions)
    gru = NeuralPolicy(build_actor_critic(4, PolicyNetConfig(kind="gru", hidden=16, window=4)), cfg)
    tp = target_probabilities(gru, logged[:5], cfg)
    assert all(t.shape == (12, 4) and np.allclose(t.sum(1), 1) for t in tp)


# ── experiments ──────────────────────────────────────────────────────────────

def test_assignment_is_deterministic_and_balanced():
    seeds = seeds_for(0, 2000)
    a = assign_arms(seeds, 30.0, "salt")
    assert a == assign_arms(seeds, 30.0, "salt") and a != assign_arms(seeds, 30.0, "other")
    frac = np.mean([v == "challenger" for v in a.values()])
    assert abs(frac - 0.3) < 0.03


def test_permutation_test_and_ab_decisions():
    rng = np.random.default_rng(0)
    same = rng.normal(size=200); shifted = same + 1.0
    assert permutation_test(same, same.copy(), 300, rng) > 0.5
    assert permutation_test(same, shifted, 300, rng) < 0.01
    cfg = AllocationEnvConfig(horizon=16, budget=8.0)
    good, bad = DualPacingPolicy(cfg), ThresholdPacingPolicy(cfg, threshold=5.0)  # threshold 5 ≈ never allocates
    res = run_ab_experiment(good, bad, cfg, ABConfig(n_episodes=120, n_bootstrap=100, n_permutations=100))
    assert not res.promote and any("regressed" in f for f in res.guardrail_failures)
    assert res.deltas["value"].ci_high < 0 and res.p_value < 0.05
    res2 = run_ab_experiment(bad, good, cfg, ABConfig(n_episodes=120, n_bootstrap=100, n_permutations=100))
    assert res2.deltas["value"].ci_low > 0 and res2.guardrail_failures == [] and res2.promote
    gm = res2.gate_metrics()
    assert gm["ab/value_delta_ci_low"] > 0 and gm["ab/n_challenger"] >= 30
    small = run_ab_experiment(bad, good, cfg, ABConfig(n_episodes=20, min_episodes=30, n_bootstrap=50, n_permutations=50))
    assert small.warnings and not small.promote
    with pytest.raises(ConfigError):
        run_ab_experiment(bad, good, cfg, ABConfig(n_episodes=60, n_bootstrap=20, n_permutations=20, guardrails={"nope": 1.0}))


def test_shadow_evaluation_reports_divergence():
    cfg = AllocationEnvConfig(horizon=12, budget=6.0)
    inc = DualPacingPolicy(cfg)
    res = shadow_evaluate(inc, DualPacingPolicy(cfg), cfg, seeds_for(4, 10), OPEConfig(n_bootstrap=20))
    assert res.divergence_rate == 0.0 and res.candidate_replay_value == pytest.approx(res.incumbent_value)
    res = shadow_evaluate(inc, ThresholdPacingPolicy(cfg, 0.6), cfg, seeds_for(4, 10), OPEConfig(n_bootstrap=20))
    assert 0 < res.divergence_rate <= 1 and len(res.divergence_by_third) == 3 and sum(map(sum, res.action_confusion)) == res.n_decisions
    assert "error" in res.candidate_ope  # deterministic incumbent → no valid propensities for OPE


# ── lifecycle integration ────────────────────────────────────────────────────

def test_lifecycle_with_ab_gate(tmp_path):
    cfg = AllocationEnvConfig(horizon=16, budget=8.0)
    reg = CandidateRegistry(tmp_path / "reg.json")
    champ = reg.register(ModelCandidate("allocator", "v1", "dual_pacing", "builtin:dual"))
    reg.transition(champ.id, CandidateState.CHALLENGER); reg.transition(champ.id, CandidateState.CHAMPION)
    gate = PromotionGate.from_config(__import__("yaml").safe_load((ROOT / "configs/allocation/promotion_gate.yaml").read_text()))
    res = run_ab_experiment(DualPacingPolicy(cfg), ThresholdPacingPolicy(cfg, 5.0), cfg, ABConfig(n_episodes=120, n_bootstrap=100, n_permutations=100))
    cand = reg.register(ModelCandidate("allocator", "v2", "allocation_ppo", "ckpt", metrics=res.gate_metrics()))
    reg.transition(cand.id, CandidateState.CHALLENGER)
    report = gate.evaluate(cand.metrics, champ.metrics)
    assert not report.passed and any(c.metric == "ab/value_delta_ci_low" for c in report.failures())
    # a genuinely better challenger passes, is promoted, and can be rolled back
    res2 = run_ab_experiment(ThresholdPacingPolicy(cfg, 5.0), DualPacingPolicy(cfg), cfg, ABConfig(n_episodes=120, n_bootstrap=100, n_permutations=100))
    good = reg.register(ModelCandidate("allocator", "v3", "allocation_ppo", "ckpt3", metrics=res2.gate_metrics()))
    reg.transition(good.id, CandidateState.CHALLENGER)
    assert gate.evaluate(good.metrics).passed
    reg.transition(good.id, CandidateState.CHAMPION)
    assert reg.champion("allocator").id == good.id and reg.get(champ.id).state == CandidateState.RETIRED
    restored = rollback(reg, "allocator")
    assert restored.id == champ.id and reg.get(good.id).state == CandidateState.RETIRED


# ── CLI ──────────────────────────────────────────────────────────────────────

def test_cli_end_to_end(tmp_path):
    out = tmp_path / "bench"
    assert main(["allocation", "benchmark", "--config", str(ROOT / "configs/allocation/benchmark_tiny_cpu.yaml"), "--output-dir", str(out)]) == 0
    results = json.loads((out / "results.json").read_text())
    assert set(results["policies"]) == {"threshold_pacing", "dual_pacing", "ppo_mlp", "ppo_gru"} and (out / "summary.md").exists()
    ckpt = results["challenger"]["checkpoint"]
    env_yaml = tmp_path / "env.yaml"; env_yaml.write_text("horizon: 12\nbudget: 6.0\n")
    rc = main(["allocation", "ope", "--env", str(env_yaml), "--log", str(out / "logged_trajectories.jsonl"), "--target", ckpt, "--bootstrap", "20"])
    assert rc in (0, 3)
    gm = tmp_path / "gate.json"
    rc = main(["allocation", "ab", "--env", str(env_yaml), "--incumbent", "dual", "--challenger", ckpt, "--episodes", "60", "--gate-metrics", str(gm),
               "--metrics", "local", "--output-dir", str(tmp_path / "ab")])
    assert rc in (0, 2) and "ab/value_delta_ci_low" in json.loads(gm.read_text())
    assert (tmp_path / "ab" / "events.jsonl").exists()
    assert main(["allocation", "shadow", "--env", str(env_yaml), "--incumbent", "dual", "--candidate", ckpt, "--episodes", "5"]) == 0
    assert main(["train", str(ROOT / "configs/allocation/ppo_tiny_cpu.yaml"), "--output-dir", str(tmp_path / "train"), "--metrics", "none"]) == 0
