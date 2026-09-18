"""Sequential-decisioning lifecycle: train a sequence policy → held-out evaluation → OPE → simulated A/B → gated
registration → promotion or rejection → rollback. Runs on CPU in about a minute (small settings)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import yaml

from forgeline.core.config import ExperimentManifest
from forgeline.core.lifecycle import RunContext
from forgeline.deployment.candidates import CandidateState, ModelCandidate
from forgeline.deployment.promotion import PromotionGate
from forgeline.deployment.registry import CandidateRegistry
from forgeline.deployment.rollback import rollback
from forgeline.domains.allocation import (
    ABConfig, AllocationEnvConfig, DualPacingPolicy, EpsilonMixPolicy, OPEConfig, aggregate_metrics, evaluate_target_policy,
    load_allocation_policy, run_ab_experiment, run_episodes, seeds_for, shadow_evaluate,
)
from forgeline.training.common.trainer import Trainer
from forgeline.training.factory import build_algorithm

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="forgeline-alloc-"))
    env_cfg = AllocationEnvConfig(horizon=32, budget=16.0)

    # 1. train a GRU policy through the shared trainer (small budget for the example)
    manifest = ExperimentManifest.load(ROOT / "configs/allocation/ppo_gru.yaml")
    manifest.output_dir = str(work / "train"); manifest.trainer.max_steps = 240; manifest.trainer.schedule.decay_steps = 240
    manifest.trainer.eval_every = 0; manifest.algorithm_params["env"] = env_cfg.to_dict(); manifest.algorithm_params["episodes_per_rollout"] = 16
    ctx = RunContext.create(manifest, metrics_backend="local")
    Trainer(ctx, build_algorithm(manifest, ctx)).fit(); ctx.close()
    candidate = load_allocation_policy(work / "train/checkpoints/final")
    incumbent = DualPacingPolicy(env_cfg)

    # 2. held-out simulator evaluation on common seeds
    seeds = seeds_for(123, 200)
    for name, pol in (("incumbent (dual pacer)", incumbent), ("candidate (GRU PPO)", candidate)):
        m = aggregate_metrics(run_episodes(pol, env_cfg, seeds), env_cfg)
        print(f"{name}: value {m['value']:.3f} ± {m['value_ci95_halfwidth']:.3f}, utilisation {m['utilization']:.2f}, pacing error {m['pacing_error']:.3f}")

    # 3. offline policy evaluation from an ε-mixed behaviour log
    behaviour = EpsilonMixPolicy(candidate, 0.2, env_cfg.n_actions)
    logged = run_episodes(behaviour, env_cfg, seeds_for(7, 150))
    report = evaluate_target_policy(candidate, logged, env_cfg, OPEConfig(max_weight=20.0, n_bootstrap=200))
    print("OPE:", {k: round(v.value, 3) for k, v in report.estimates.items()}, "ESS", round(report.diagnostics.ess, 1), report.diagnostics.warnings)

    # 4. simulated A/B experiment → gate metrics
    ab = run_ab_experiment(incumbent, candidate, env_cfg, ABConfig(n_episodes=300, seed=5), "dual_pacing", "ppo_gru")
    print(f"A/B Δvalue {ab.deltas['value'].delta:+.3f} CI [{ab.deltas['value'].ci_low:+.3f}, {ab.deltas['value'].ci_high:+.3f}] p={ab.p_value:.3f} "
          f"guardrails={ab.guardrail_failures or 'none'} → {'promote' if ab.promote else 'reject'}")

    # 5. registry + gate
    registry = CandidateRegistry(work / "registry.json")
    champ = registry.register(ModelCandidate("allocator", "v1", "dual_pacing", "builtin:dual"))
    registry.transition(champ.id, CandidateState.CHALLENGER); registry.transition(champ.id, CandidateState.CHAMPION)
    cand = registry.register(ModelCandidate("allocator", "v2", "allocation_ppo", str(work / "train/checkpoints/final"), metrics=ab.gate_metrics()))
    registry.transition(cand.id, CandidateState.CHALLENGER)
    gate = PromotionGate.from_config(yaml.safe_load((ROOT / "configs/allocation/promotion_gate.yaml").read_text()))
    verdict = gate.evaluate(cand.metrics)
    print("gate:", "PASS" if verdict.passed else "REJECT", [c.metric for c in verdict.failures()])
    if verdict.passed:
        registry.transition(cand.id, CandidateState.CHAMPION)
        print("promoted; rolling back for demonstration →", rollback(registry, "allocator").key)

    # 6. shadow evaluation
    sh = shadow_evaluate(incumbent, candidate, env_cfg, seeds_for(9, 50), OPEConfig(n_bootstrap=50))
    print(f"shadow divergence {sh.divergence_rate:.3f}; incumbent {sh.incumbent_value:.3f} vs candidate replay {sh.candidate_replay_value:.3f}")
    print("artifacts in", work)


if __name__ == "__main__":
    main()
