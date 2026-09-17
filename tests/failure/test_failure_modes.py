"""Explicit, actionable errors for each failure category."""

import json
import shutil

import pytest
import torch

from forgeline.checkpoints.manager import CheckpointManager, CheckpointState
from forgeline.core.config import DistributedConfig, ExperimentManifest, model_spec_from_preset
from forgeline.core.errors import (
    CheckpointCorruptError, CheckpointMismatchError, CheckpointNotFoundError, ConfigError, DatasetError, DistributedConfigError,
    EvaluationError, ExportError, ForgelineError, IncompatibleStateError, InvalidRewardError, MalformedRecordError, PromotionError,
    QuantizationError, RolloutError, ServingError, VerifierError,
)
from forgeline.data.records import PreferenceRecord, read_jsonl
from forgeline.deployment.promotion import GateRule, PromotionGate
from forgeline.models.quantization.gguf import export_gguf
from forgeline.models.quantization.nf4 import quantize_nf4
from forgeline.models.transformer.model import TransformerLM
from forgeline.rollouts.agent import AgentRolloutConfig, AgentRolloutEngine
from forgeline.rollouts.engine import RolloutConfig, RolloutEngine
from forgeline.rollouts.rewards import CallableReward, score_trajectories
from forgeline.rollouts.verifiers import MathVerifier
from forgeline.serving.api import Router
from forgeline.serving.backends import EchoBackend


@pytest.fixture
def saved(tmp_path):
    m = TransformerLM(model_spec_from_preset("tiny", vocab_size=20))
    path = CheckpointManager(tmp_path).save(CheckpointState(model_state=m.state_dict(), step=5, algorithm="sft", model_spec=m.spec.to_dict()))
    return path, m


def test_all_errors_have_hints():
    for cls in ForgelineError.__subclasses__():
        assert cls("x").hint or cls is ForgelineError


def test_checkpoint_missing(tmp_path):
    with pytest.raises(CheckpointNotFoundError):
        CheckpointManager.load(tmp_path / "nope")


def test_checkpoint_bad_metadata(saved):
    path, _ = saved
    (path / "manifest.json").write_text("{broken")
    with pytest.raises(CheckpointCorruptError):
        CheckpointManager.load(path)


def test_checkpoint_incomplete_and_truncated(saved):
    path, _ = saved
    manifest = json.loads((path / "manifest.json").read_text())
    with open(path / "model.pt", "r+b") as f:
        f.truncate(10)
    with pytest.raises(CheckpointCorruptError, match="truncated"):
        CheckpointManager.load(path)
    (path / "model.pt").unlink()
    with pytest.raises(CheckpointCorruptError, match="missing model.pt"):
        CheckpointManager.load(path)
    (path / "manifest.json").unlink()
    with pytest.raises(CheckpointCorruptError, match="no manifest"):
        CheckpointManager.load(path)
    manifest["complete"] = False


def test_checkpoint_incomplete_flag_and_tmp_dir_ignored(saved, tmp_path):
    path, _ = saved
    m = json.loads((path / "manifest.json").read_text())
    m["complete"] = False
    (path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(CheckpointCorruptError, match="incomplete"):
        CheckpointManager.load(path)
    (tmp_path / "step_00000099.tmp").mkdir()  # crashed mid-save
    assert CheckpointManager(tmp_path).find_latest().name == path.name


def test_checkpoint_mismatched_config_and_state(saved):
    path, m = saved
    with pytest.raises(CheckpointMismatchError):
        CheckpointManager.load(path, expected_spec={"n_layer": 12})
    with pytest.raises(CheckpointMismatchError):
        CheckpointManager.load(path, expected_algorithm="dpo")
    from forgeline.checkpoints.manager import load_model_state

    with pytest.raises(IncompatibleStateError):
        load_model_state(TransformerLM(model_spec_from_preset("tiny", vocab_size=99)), CheckpointManager.load(path).model_state)


def test_dataset_failures(tmp_path):
    with pytest.raises(DatasetError):
        read_jsonl(tmp_path / "missing.jsonl", PreferenceRecord)
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"prompt": "p", "chosen": 1, "rejected": "r"}) + "\n")
    with pytest.raises(MalformedRecordError, match="bad.jsonl:1"):
        read_jsonl(p, PreferenceRecord)


def test_invalid_reward_and_verifier_exception(policy):
    from forgeline.core.protocols import Trajectory

    t = Trajectory(prompt_ids=torch.tensor([1]), response_ids=torch.tensor([2]), response_text="x")
    with pytest.raises(InvalidRewardError):
        score_trajectories(CallableReward(lambda _: float("nan")), [t])

    class Exploding(MathVerifier):
        def extract_answer(self, text):
            raise RuntimeError("boom")

    r = Exploding().verify("1", "1")
    assert r.passed is False and "boom" in r.info["error"]


def test_tool_execution_failure_is_injected(tokenizer):
    class Scripted:
        device = torch.device("cpu"); pad_token_id = 0; n = 0
        def encode(self, t): return tokenizer.encode(t)
        def decode(self, ids): return ['<tool_call>{"name": "python_executor", "args": {"code": "raise ValueError(42)"}}</tool_call>', "<final_answer>1</final_answer>"][self.n - 1]
        def generate(self, ids, s):
            self.n += 1
            return torch.zeros(1, 1, dtype=torch.long)

    def broken_tool(args):
        raise RuntimeError("tool crashed")

    _, blocks, _ = AgentRolloutEngine(Scripted(), AgentRolloutConfig()).run_episode("p")
    assert "ERROR" in blocks[0] and "ValueError" in blocks[0]
    _, blocks, _ = AgentRolloutEngine(Scripted(), AgentRolloutConfig(tools={"python_executor": broken_tool})).run_episode("p")
    assert "ERROR" in blocks[0] and "tool crashed" in blocks[0]


def test_rollout_failure(policy):
    class Broken:
        device = torch.device("cpu")
        def generate(self, *a): raise RuntimeError("OOM")
    with pytest.raises(RolloutError, match="OOM"):
        RolloutEngine(Broken(), RolloutConfig()).rollout(torch.tensor([1, 2]))
    with pytest.raises(RolloutError):
        RolloutEngine(policy, RolloutConfig()).rollout(torch.tensor([], dtype=torch.long))


def test_distributed_misconfiguration():
    from forgeline.distributed.strategies import TensorParallelStrategy, validate_distributed_config
    with pytest.raises(DistributedConfigError):
        validate_distributed_config(DistributedConfig(strategy="pipeline_parallel", pipeline_parallel_size=3), world_size=4)
    with pytest.raises(DistributedConfigError):
        TensorParallelStrategy(DistributedConfig(strategy="tensor_parallel", tensor_parallel_size=2)).wrap_model(torch.nn.Linear(1, 1))
    with pytest.raises(ConfigError):
        ExperimentManifest.from_dict({"distributed": {"strategy": "fsdp", "tensor_parallel_size": 0}})


def test_promotion_failure():
    with pytest.raises(PromotionError, match="reward"):
        PromotionGate([GateRule("reward", min=0.9)]).assert_promotable({"reward": 0.1})


def test_serving_backend_failure():
    class Failing(EchoBackend):
        def complete(self, *a, **k):
            raise RuntimeError("backend crashed")
    router = Router({"f": Failing("f")}, "f")
    status, body = router.handle("POST", "/v1/completions", {"prompt": "x"})
    assert status == 500 and "backend crashed" in body["error"]["message"]
    with pytest.raises(ServingError):
        Router({"a": EchoBackend()}, "missing")


def test_engine_isolates_failing_sequence(tiny_model):
    from forgeline.inference.engine import InferenceEngine
    from forgeline.inference.requests import RequestStatus, SamplingParams

    eng = InferenceEngine(tiny_model.eval(), max_batch=2)
    good = eng.submit([1, 2], SamplingParams(max_tokens=3, temperature=0))
    bad = eng.submit([10_000], SamplingParams(max_tokens=3, temperature=0))  # token id out of vocabulary
    eng.run_until_idle()
    assert good.status == RequestStatus.FINISHED and bad.status == RequestStatus.FAILED and bad.error


def test_quantization_and_export_errors(tmp_path):
    with pytest.raises(QuantizationError):
        quantize_nf4(torch.empty(0))
    m = TransformerLM(model_spec_from_preset("tiny", vocab_size=10))
    with pytest.raises(ExportError):
        export_gguf(m.state_dict(), m.spec, tmp_path / "x.gguf", quantize="q3_k")


def test_non_finite_loss_aborts_training(ctx_factory, tiny_spec):
    from forgeline.training.common.trainer import PostTrainingAlgorithm, Trainer

    class NaNAlg(PostTrainingAlgorithm):
        name = "pretrain"
        def __init__(self): self.p = torch.nn.Parameter(torch.ones(1))
        def parameters(self): return [self.p]
        def collect(self, step): return None
        def loss(self, batch, i, n): return self.p * float("nan"), {}

    ctx = ctx_factory("pretrain", eval_every=0)
    with pytest.raises(ConfigError, match="non-finite loss"):
        Trainer(ctx, NaNAlg()).fit()
    events = [json.loads(l)["event"] for l in (ctx.output_dir / "events.jsonl").read_text().splitlines()]
    assert "run.failed" in events


def test_step_trainer_rejects_unsupported_strategy_combinations(policy):
    from types import SimpleNamespace

    from forgeline.cli.main import _apply_strategy

    manifest = ExperimentManifest(algorithm="dpo", model=model_spec_from_preset("tiny"))
    for name in ("fsdp", "tensor_parallel", "pipeline_parallel", "deepspeed"):
        strategy = SimpleNamespace(name=name, rank=lambda: 0)
        with pytest.raises(DistributedConfigError):
            _apply_strategy(SimpleNamespace(), strategy, manifest)
