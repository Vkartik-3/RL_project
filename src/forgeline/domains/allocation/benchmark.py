"""One-command benchmark: heuristic vs online optimizer vs stateless PPO vs sequence PPO, with OPE, A/B and shadow.

``run_benchmark(BenchmarkConfig)`` trains the two PPO variants for each training seed through the shared
:class:`Trainer`, evaluates every policy on a common held-out seed family (identical opportunity streams per seed),
computes regret against the hindsight oracle, checks OPE accuracy against simulator ground truth, runs a simulated
A/B test between the best baseline (incumbent) and the best PPO policy (challenger), a shadow evaluation, and records
throughput. Results are written as JSON plus a Markdown summary so they can be dropped into ``benchmarks/``.
"""

from __future__ import annotations

import dataclasses
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from forgeline.core.config import CheckpointConfig, ExperimentManifest, OptimizerConfig, ScheduleConfig, TrainerConfig
from forgeline.core.errors import ConfigError
from forgeline.core.lifecycle import RunContext
from forgeline.domains.allocation.env import ACTION_NAMES, AllocationEnvConfig
from forgeline.domains.allocation.experiment import ABConfig, run_ab_experiment, shadow_evaluate
from forgeline.domains.allocation.ope import OPEConfig, evaluate_target_policy
from forgeline.domains.allocation.oracle import oracle_values
from forgeline.domains.allocation.policies import DualPacingPolicy, EpsilonMixPolicy, NeuralPolicy, Policy, PolicyNetConfig, ThresholdPacingPolicy
from forgeline.domains.allocation.ppo import AllocationPPOAlgorithm, AllocationPPOConfig
from forgeline.domains.allocation.rollout import aggregate_metrics, run_episodes, seeds_for, write_episodes
from forgeline.training.common.trainer import Trainer


@dataclass
class BenchmarkConfig:
    env: Dict[str, Any] = field(default_factory=dict)
    training_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    eval_episodes: int = 300
    eval_seed: int = 20_000
    trainer_steps: int = 600
    episodes_per_rollout: int = 32
    ppo_epochs: int = 4
    learning_rate: float = 3e-3
    hidden: int = 64
    window: int = 8
    heuristic_threshold: float = 0.6
    dual_learning_rate: float = 0.2
    dual_initial_price: float = 0.8
    ope_log_episodes: int = 300
    ope_epsilon: float = 0.2
    ope_max_weight: Optional[float] = 20.0
    ab_episodes: int = 400
    shadow_episodes: int = 100
    output_dir: str = "runs/allocation-benchmark"
    metrics: str = "local"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ConfigError(f"unknown benchmark keys: {sorted(unknown)}")
        return cls(**d)


def train_policy(kind: str, env_cfg: AllocationEnvConfig, bcfg: BenchmarkConfig, seed: int, out_dir: Path, metrics: str) -> tuple:
    manifest = ExperimentManifest(
        name=f"allocation-ppo-{kind}-seed{seed}", algorithm="allocation_ppo", output_dir=str(out_dir),
        trainer=TrainerConfig(max_steps=bcfg.trainer_steps, batch_size=1, gradient_accumulation_steps=4, eval_every=0, log_every=50,
                              device="cpu", seed=seed,
                              optimizer=OptimizerConfig(learning_rate=bcfg.learning_rate, weight_decay=0.0, beta2=0.999, grad_clip=0.5, decay_only_matrices=False),
                              schedule=ScheduleConfig(name="linear_decay", warmup_steps=0, decay_steps=bcfg.trainer_steps, min_lr=bcfg.learning_rate / 10),
                              checkpoint=CheckpointConfig(directory="checkpoints", save_every=0)),
        algorithm_params={"env": env_cfg.to_dict(), "policy": {"kind": kind, "hidden": bcfg.hidden, "window": bcfg.window},
                          "episodes_per_rollout": bcfg.episodes_per_rollout, "ppo_epochs": bcfg.ppo_epochs, "seed": seed},
    )
    ctx = RunContext.create(manifest, metrics_backend=metrics)
    torch.manual_seed(seed)
    alg = AllocationPPOAlgorithm(env_cfg, PolicyNetConfig(kind=kind, hidden=bcfg.hidden, window=bcfg.window),
                                 AllocationPPOConfig(episodes_per_rollout=bcfg.episodes_per_rollout, ppo_epochs=bcfg.ppo_epochs, seed=seed))
    t0 = time.time()
    trainer = Trainer(ctx, alg)
    trainer.fit()
    ctx.close()
    return alg.policy, time.time() - t0, str(ctx.output_dir / "checkpoints" / "final")


def evaluate(policy: Policy, env_cfg: AllocationEnvConfig, seeds: List[int], oracle: np.ndarray) -> Dict[str, float]:
    t0 = time.time()
    eps = run_episodes(policy, env_cfg, seeds, rng_seed=seeds[0])
    m = aggregate_metrics(eps, env_cfg)
    values = np.array([e.total_reward for e in eps])
    m["regret_vs_oracle"] = float((oracle - values).mean())
    m["regret_vs_oracle_std"] = float((oracle - values).std(ddof=1))
    m["eval_seconds"] = time.time() - t0
    m["env_steps_per_sec"] = len(seeds) * env_cfg.horizon / max(m["eval_seconds"], 1e-9)
    return m


def run_benchmark(bcfg: BenchmarkConfig, log=print) -> Dict[str, Any]:
    env_cfg = AllocationEnvConfig.from_dict(bcfg.env)
    out = Path(bcfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    eval_seeds = seeds_for(bcfg.eval_seed, bcfg.eval_episodes)
    oracle = oracle_values(env_cfg, eval_seeds)
    results: Dict[str, Any] = {"config": dataclasses.asdict(bcfg), "env": env_cfg.to_dict(), "action_names": list(ACTION_NAMES),
                               "oracle": {"mean_value": float(oracle.mean()), "std": float(oracle.std(ddof=1))},
                               "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
                                               "threads": torch.get_num_threads()},
                               "policies": {}, "training": {}, "started_at": time.time()}
    # baselines (deterministic given the seeds; one evaluation)
    heuristic = ThresholdPacingPolicy(env_cfg, threshold=bcfg.heuristic_threshold)
    dual = DualPacingPolicy(env_cfg, learning_rate=bcfg.dual_learning_rate, initial_price=bcfg.dual_initial_price)
    for name, pol in (("threshold_pacing", heuristic), ("dual_pacing", dual)):
        results["policies"][name] = {"seeds": {"baseline": evaluate(pol, env_cfg, eval_seeds, oracle)}}
        log(f"{name}: value {results['policies'][name]['seeds']['baseline']['value']:.3f}")
    # PPO variants across training seeds
    trained: Dict[str, List[tuple]] = {"ppo_mlp": [], "ppo_gru": []}
    for kind, key in (("mlp", "ppo_mlp"), ("gru", "ppo_gru")):
        results["policies"][key] = {"seeds": {}}
        for seed in bcfg.training_seeds:
            pol, secs, ckpt = train_policy(kind, env_cfg, bcfg, seed, out / f"train-{kind}-seed{seed}", bcfg.metrics)
            m = evaluate(pol, env_cfg, eval_seeds, oracle)
            m["train_seconds"] = secs; m["checkpoint"] = ckpt
            results["policies"][key]["seeds"][str(seed)] = m
            trained[key].append((seed, pol, m["value"], ckpt))
            log(f"{key} seed {seed}: value {m['value']:.3f} (train {secs:.1f}s)")
    # across-seed summaries
    for key, entry in results["policies"].items():
        rows = list(entry["seeds"].values())
        summary = {}
        for k in ("value", "utilization", "pacing_error", "early_exhaustion", "violations", "regret_vs_oracle", "value_per_budget", "unused_budget_frac"):
            v = np.array([r[k] for r in rows]); summary[k] = float(v.mean()); summary[f"{k}_seed_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        summary["n_seeds"] = len(rows)
        entry["summary"] = summary
    # regret vs baselines
    base_vals = {k: results["policies"][k]["summary"]["value"] for k in results["policies"]}
    for key, entry in results["policies"].items():
        entry["summary"]["regret_vs_threshold"] = base_vals["threshold_pacing"] - entry["summary"]["value"]
        entry["summary"]["regret_vs_dual"] = base_vals["dual_pacing"] - entry["summary"]["value"]
    # best models
    best = {k: max(v, key=lambda x: x[2]) for k, v in trained.items()}
    incumbent_name = max(("threshold_pacing", "dual_pacing"), key=lambda k: base_vals[k])
    incumbent = heuristic if incumbent_name == "threshold_pacing" else dual
    ch_key = max(best, key=lambda k: best[k][2]); ch_seed, challenger, _, ch_ckpt = best[ch_key]
    results["incumbent"] = incumbent_name; results["challenger"] = {"policy": ch_key, "seed": ch_seed, "checkpoint": ch_ckpt}
    entry = results["policies"][ch_key]; entry["summary"]["regret_vs_incumbent"] = base_vals[incumbent_name] - entry["summary"]["value"]
    # OPE: behaviour = ε-mix of the best MLP policy; targets = every policy; truth = simulator on the same seeds
    behaviour = EpsilonMixPolicy(best["ppo_mlp"][1], bcfg.ope_epsilon, env_cfg.n_actions)
    log_seeds = seeds_for(bcfg.eval_seed + 1, bcfg.ope_log_episodes)
    t0 = time.time()
    logged = run_episodes(behaviour, env_cfg, log_seeds, rng_seed=7)
    write_episodes(logged, out / "logged_trajectories.jsonl")
    ope_cfg = OPEConfig(max_weight=bcfg.ope_max_weight)
    results["ope"] = {"behaviour": behaviour.name, "n_episodes": len(logged), "logged_value": float(np.mean([e.total_reward for e in logged])), "targets": {}}
    targets = {"ppo_mlp_best": best["ppo_mlp"][1], "ppo_gru_best": best["ppo_gru"][1], "dual_pacing": dual, "threshold_pacing": heuristic,
               "behaviour_itself": behaviour}
    for name, pol in targets.items():
        truth = float(np.mean([e.total_reward for e in run_episodes(pol, env_cfg, log_seeds, rng_seed=11)]))
        try:
            rep = evaluate_target_policy(pol, logged, env_cfg, ope_cfg, target_name=name).to_dict()
            rep["simulator_truth"] = truth
            rep["abs_error"] = {k: abs(v["value"] - truth) for k, v in rep["estimates"].items()}
            rep["ci_covers_truth"] = {k: bool(v["ci_low"] <= truth <= v["ci_high"]) for k, v in rep["estimates"].items()}
        except Exception as exc:  # noqa: BLE001
            rep = {"error": f"{type(exc).__name__}: {exc}", "simulator_truth": truth}
        results["ope"]["targets"][name] = rep
        log(f"OPE {name}: " + (", ".join(f"{k}={v['value']:.3f}" for k, v in rep.get("estimates", {}).items()) or rep.get("error", "")) + f" | truth {truth:.3f}")
    results["ope"]["seconds"] = time.time() - t0
    results["ope"]["trajectories_per_sec"] = len(logged) * len(targets) / max(results["ope"]["seconds"], 1e-9)
    # A/B: incumbent baseline vs challenger PPO
    ab = run_ab_experiment(incumbent, challenger, env_cfg, ABConfig(n_episodes=bcfg.ab_episodes, seed=bcfg.eval_seed + 2),
                           incumbent_name=incumbent_name, challenger_name=ch_key)
    results["ab"] = ab.to_dict()
    results["ab"]["episodes_per_sec"] = bcfg.ab_episodes / max(ab.elapsed_s, 1e-9)
    log(f"A/B {incumbent_name} vs {ch_key}: Δvalue {ab.deltas['value'].delta:+.3f} CI [{ab.deltas['value'].ci_low:+.3f}, {ab.deltas['value'].ci_high:+.3f}] p={ab.p_value:.3f} promote={ab.promote}")
    # shadow
    sh = shadow_evaluate(incumbent, challenger, env_cfg, seeds_for(bcfg.eval_seed + 3, bcfg.shadow_episodes), ope_cfg)
    results["shadow"] = sh.to_dict()
    results["elapsed_s"] = time.time() - results["started_at"]
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    (out / "summary.md").write_text(render_summary(results))
    return results


def render_summary(r: Dict[str, Any]) -> str:
    pols = r["policies"]
    lines = ["# Budgeted allocation benchmark", "",
             f"Environment: horizon {r['env']['horizon']}, budget {r['env']['budget']}, {r['config']['eval_episodes']} held-out episodes per policy, "
             f"{len(r['config']['training_seeds'])} training seeds for PPO. Oracle (hindsight upper bound) mean value {r['oracle']['mean_value']:.3f}.", "",
             "| Policy | Value (mean ± seed std) | Utilisation | Pacing error | Early exhaustion | Violations | Regret vs oracle | Regret vs dual pacer |",
             "|---|---|---|---|---|---|---|---|"]
    for k, e in pols.items():
        s = e["summary"]
        lines.append(f"| {k} | {s['value']:.3f} ± {s['value_seed_std']:.3f} | {s['utilization']:.3f} | {s['pacing_error']:.3f} | {s['early_exhaustion']:.3f} | "
                     f"{s['violations']:.2f} | {s['regret_vs_oracle']:.3f} | {s['regret_vs_dual']:+.3f} |")
    lines += ["", "## Offline policy evaluation (behaviour: " + r["ope"]["behaviour"] + f", {r['ope']['n_episodes']} logged episodes)", "",
              "| Target | Truth | IPS | SNIPS | PDIS | DR | DR clipped | ESS | max weight | warnings |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, rep in r["ope"]["targets"].items():
        if "error" in rep:
            lines.append(f"| {k} | {rep['simulator_truth']:.3f} | refused: {rep['error']} | | | | | | | |")
            continue
        est = rep["estimates"]; d = rep["diagnostics"]
        g = lambda n: f"{est[n]['value']:.3f} [{est[n]['ci_low']:.2f}, {est[n]['ci_high']:.2f}]" if n in est else "—"
        lines.append(f"| {k} | {rep['simulator_truth']:.3f} | {g('ips')} | {g('snips')} | {g('pdis')} | {g('dr')} | {g('dr_clipped')} | {d['ess']:.1f} | {d['max_weight']:.1f} | {len(d['warnings'])} |")
    ab = r["ab"]; dv = ab["deltas"]["value"]
    lines += ["", f"## Simulated A/B: {ab['incumbent']['name']} (n={ab['incumbent']['n']}) vs {ab['challenger']['name']} (n={ab['challenger']['n']})", "",
              f"Δvalue {dv['delta']:+.3f} (95% bootstrap CI [{dv['ci_low']:+.3f}, {dv['ci_high']:+.3f}]), permutation p = {ab['p_value']:.3f}; "
              f"guardrail failures: {ab['guardrail_failures'] or 'none'}; decision: {'promote' if ab['promote'] else 'reject'}.",
              "", f"## Shadow: divergence rate {r['shadow']['divergence_rate']:.3f} (by horizon third {[round(x, 3) for x in r['shadow']['divergence_by_third']]}); "
              f"incumbent value {r['shadow']['incumbent_value']:.3f}, candidate replay value {r['shadow']['candidate_replay_value']:.3f}",
              "", f"Throughput: evaluation {pols['dual_pacing']['seeds']['baseline']['env_steps_per_sec']:.0f} env steps/s; OPE {r['ope']['trajectories_per_sec']:.0f} trajectory-evaluations/s; "
              f"A/B {r['ab']['episodes_per_sec']:.0f} episodes/s; PPO training " + ", ".join(f"{k} {np.mean([v['train_seconds'] for v in pols[k]['seeds'].values()]):.0f}s/seed" for k in ("ppo_mlp", "ppo_gru")) + f". Total {r['elapsed_s']:.0f}s."]
    return "\n".join(lines) + "\n"
