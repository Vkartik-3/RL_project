"""Ray orchestration on a private local Ray instance (CPU only).

These tests exercise real actor processes, object-store weight transfer, placement groups and worker replacement
on one machine. They say nothing about multi-node scaling or GPU placement.
"""

import math
import os
from pathlib import Path

import pytest
import torch

ray = pytest.importorskip("ray")

from forgeline.checkpoints.manager import CheckpointManager, CheckpointState  # noqa: E402
from forgeline.core.errors import OrchestrationError  # noqa: E402
from forgeline.core.protocols import RewardResult, Trajectory  # noqa: E402
from forgeline.evaluation.suites import BenchmarkSuite, build_benchmark  # noqa: E402
from forgeline.models.loading import load_policy_from_checkpoint  # noqa: E402
from forgeline.models.policy import NativePolicy  # noqa: E402
from forgeline.models.transformer.model import TransformerLM  # noqa: E402
from forgeline.orchestration.config import RayConfig  # noqa: E402
from forgeline.orchestration.ray_backend import (  # noqa: E402
    RayEvaluator, RayRewardPool, RayRolloutEngine, RayToolPool, attach_to_algorithm, init_ray, shutdown_ray,
)
from forgeline.rollouts.engine import RolloutConfig, RolloutEngine  # noqa: E402
from forgeline.rollouts.rewards import build_reward_provider, score_trajectories  # noqa: E402
from forgeline.training import GRPOAlgorithm, GRPOConfig, Trainer  # noqa: E402

pytestmark = pytest.mark.optional_dependency
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def ray_instance():
    init_ray(RayConfig(num_cpus=4))
    yield
    shutdown_ray()


def _traj(text, answer):
    return Trajectory(prompt_ids=torch.tensor([1, 2]), response_ids=torch.tensor([3]), response_text=text, task={"answer": answer})


class CrashOnceReward:
    """Kills its worker process the first time it runs (marker file shared across processes)."""

    name = "crash_once"

    def __init__(self, marker):
        self.marker = str(marker)

    def score(self, trajectory):
        if not os.path.exists(self.marker):
            Path(self.marker).write_text("crashed")
            os._exit(1)
        return RewardResult(value=1.0)


class FailingReward:
    name = "failing"

    def score(self, trajectory):
        raise ValueError("bad trajectory")


def test_reward_pool_matches_in_process_scoring():
    cfg = {"type": "verifier", "verifier": "math", "target_key": "answer"}
    trajs = [_traj(f"The answer is {i if i % 3 else i + 1}", str(i)) for i in range(10)]
    local = [build_reward_provider(cfg).score(t) for t in trajs]
    with RayRewardPool(cfg, num_workers=2) as pool:
        values = score_trajectories(pool, trajs)
        assert values == [r.value for r in local]
        assert [t.reward.passed for t in trajs] == [r.passed for r in local]
        pids = pool.pids()
        assert len(set(pids)) == 2 and os.getpid() not in pids


def test_tool_pool_preserves_order():
    with RayToolPool(num_workers=2, timeout=10) as pool:
        out = pool.execute_many([f"print({i} * {i})" for i in range(5)])
    assert [o["stdout"] for o in out] == [str(i * i) for i in range(5)]


def test_rollout_workers_sample_from_current_learner_weights(tiny_spec, tokenizer):
    torch.manual_seed(0)
    policy = NativePolicy(TransformerLM(tiny_spec), tokenizer)
    rc = RolloutConfig(group_size=4, max_new_tokens=8, temperature=0.0)
    prompt = torch.tensor(tokenizer.encode("What is 2 + 2?"))
    with RayRolloutEngine(policy, rc, num_workers=2) as engine:
        remote = engine.rollout(prompt, {"answer": "4"})
        local = RolloutEngine(policy, rc).rollout(prompt, {"answer": "4"})
        assert [t.response_ids.tolist() for t in remote] == [t.response_ids.tolist() for t in local]
        pushes = engine.weight_pushes
        assert engine.rollout(prompt) and engine.weight_pushes == pushes  # unchanged weights → no push

        with torch.no_grad():  # simulate an optimizer step on the learner
            for p in policy.model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        remote = engine.rollout(prompt)
        local = RolloutEngine(policy, rc).rollout(prompt)
        assert engine.weight_pushes == pushes + 1
        assert [t.response_ids.tolist() for t in remote] == [t.response_ids.tolist() for t in local]
        assert len(set(engine.worker_versions())) == 1


def test_killed_rollout_worker_is_replaced_with_current_weights(tiny_spec, tokenizer):
    torch.manual_seed(1)
    policy = NativePolicy(TransformerLM(tiny_spec), tokenizer)
    rc = RolloutConfig(group_size=2, max_new_tokens=6, temperature=0.0)
    prompt = torch.tensor(tokenizer.encode("1 + 1 ="))
    with RayRolloutEngine(policy, rc, num_workers=2, config=RayConfig(max_restarts=0)) as engine:
        with torch.no_grad():
            for p in policy.model.parameters():
                p.mul_(1.1)
        engine.sync_weights()
        ray.kill(engine.workers[0], no_restart=True)
        remote = engine.rollout(prompt)
        assert engine.restarts == 1
        local = RolloutEngine(policy, rc).rollout(prompt)
        assert [t.response_ids.tolist() for t in remote] == [t.response_ids.tolist() for t in local]


def test_worker_process_crash_is_retried(tmp_path):
    marker = tmp_path / "crash.marker"
    with RayRewardPool(CrashOnceReward(marker), num_workers=1) as pool:
        assert pool.score_many([_traj("x", "1")])[0].value == 1.0
        assert pool.restarts == 1 and marker.exists()


def test_user_errors_propagate_without_retry():
    with RayRewardPool(FailingReward(), num_workers=1) as pool:
        with pytest.raises(ValueError, match="bad trajectory"):
            pool.score_many([_traj("x", "1")])
        assert pool.restarts == 0


def test_retry_budget_is_enforced(tmp_path):
    class AlwaysCrash:
        name = "always_crash"

        def score(self, trajectory):
            os._exit(1)

    with RayRewardPool(AlwaysCrash(), num_workers=1, config=RayConfig(max_restarts=0, max_call_retries=1)) as pool:
        with pytest.raises(OrchestrationError, match="failed after 2 attempts"):
            pool.score_many([_traj("x", "1")])


def test_placement_group_reservation_and_infeasible_request():
    with RayToolPool(num_workers=2, config=RayConfig(placement_strategy="PACK")) as pool:
        placement = pool.placement()
        assert placement["state"] == "CREATED" and len(placement["bundles"]) == 2
    with pytest.raises(OrchestrationError, match="could not be scheduled"):
        RayToolPool(num_workers=1, config=RayConfig(placement_strategy="PACK", cpus_per_worker=512, placement_timeout_s=2))


def test_sharded_evaluation_equals_serial(tmp_path, tiny_spec, tokenizer):
    torch.manual_seed(0)
    model = TransformerLM(tiny_spec)
    ckpt = CheckpointManager(tmp_path / "ckpt").save(CheckpointState(
        model_state={"model": model.state_dict()}, model_spec=tiny_spec.to_dict(), tokenizer_meta=tokenizer.metadata(),
        algorithm="pretrain"), name="final")
    serial = BenchmarkSuite(build_benchmark("arc", offline=True), max_new_tokens=4).run(load_policy_from_checkpoint(ckpt))
    with RayEvaluator(str(ckpt), num_workers=2) as evaluator:
        sharded = evaluator.run("arc", max_new_tokens=4, offline=True)
    assert sharded.n_samples == serial.n_samples
    assert sharded.metrics == serial.metrics
    assert [(i["index"], i["correct"], i["output"]) for i in sharded.per_item] == \
           [(i["index"], i["correct"], i["output"]) for i in serial.per_item]
    assert sharded.details["workers"] == 2


def test_grpo_trainer_with_ray_rollout_and_reward_workers(ctx_factory, tiny_spec, tokenizer):
    torch.manual_seed(0)
    policy = NativePolicy(TransformerLM(tiny_spec), tokenizer)
    prompts = [torch.tensor(tokenizer.encode(f"What is {i} + 1?")) for i in range(6)]
    tasks = [{"answer": str(i + 1)} for i in range(6)]
    reward_cfg = {"type": "rule", "name": "length", "target_length": 6}
    alg = GRPOAlgorithm(policy, prompts, build_reward_provider(reward_cfg), GRPOConfig(group_size=4, prompts_per_step=2),
                        RolloutConfig(max_new_tokens=8, temperature=0.9), tasks=tasks)
    attached = attach_to_algorithm(alg, {"type": "ray", "rollout_workers": 2, "reward_workers": 1}, reward_cfg)
    assert attached == {"backend": "ray", "reward_workers": 1, "rollout_workers": 2}
    assert isinstance(alg.rollouts, RayRolloutEngine) and isinstance(alg.reward, RayRewardPool)
    try:
        summary = Trainer(ctx_factory("grpo", run_name="grpo-ray", max_steps=3, eval_every=0, save_every=0), alg).fit()
        assert summary["final_loss"] is not None and math.isfinite(summary["final_loss"])
        # one push at start-up, then one after each optimizer step that preceded a rollout
        assert alg.rollouts.weight_pushes >= 3
    finally:
        alg.rollouts.shutdown()
        alg.reward.shutdown()


def test_cli_train_with_ray_manifest(tmp_path):
    from forgeline.cli.main import main

    rc = main(["train", str(ROOT / "configs/post_training/grpo_ray_tiny_cpu.yaml"), "--max-steps", "2",
               "--output-dir", str(tmp_path / "run"), "--set", f"data.path={ROOT / 'data/samples/verifiable/arithmetic_tasks.jsonl'}",
               "--set", "trainer.eval_every=0"])
    assert rc == 0
    assert (tmp_path / "run" / "metrics.jsonl").exists()
