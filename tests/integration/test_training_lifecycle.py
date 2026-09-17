"""Every training stage runs through the shared Trainer on CPU with finite losses."""

import math

import pytest
import torch

from forgeline.core.config import AdapterConfig, model_spec_from_preset
from forgeline.data.pretraining import MemmapCorpus
from forgeline.data.preference import PreferenceDataset
from forgeline.data.records import PreferenceRecord, SFTRecord, VerifiableTask
from forgeline.data.supervised import SupervisedDataset
from forgeline.models.policy import NativePolicy
from forgeline.models.reward import SequenceRewardModel
from forgeline.models.transformer.model import TransformerLM
from forgeline.rollouts.agent import AgentRolloutConfig
from forgeline.rollouts.engine import RolloutConfig
from forgeline.rollouts.rewards import ConstantReward, LengthReward, VerifierReward, build_reward_provider
from forgeline.rollouts.verifiers import MathVerifier
from forgeline.training import (
    DAPOAlgorithm, DAPOConfig, DistillationAlgorithm, DistillConfig, DPOAlgorithm, DPOConfig, GRPOAlgorithm, GRPOConfig,
    PPOAlgorithm, PPOConfig, PretrainAlgorithm, PretrainConfig, RewardModelAlgorithm, RewardModelConfig, RLVRConfig,
    SFTAlgorithm, SFTConfig, Trainer, build_rlvr_algorithm,
)


def finite(summary):
    return summary["final_loss"] is not None and math.isfinite(summary["final_loss"])


def test_pretrain_loss_decreases(ctx_factory, tiny_spec, corpus_dir):
    ctx = ctx_factory("pretrain", max_steps=40, batch_size=8, eval_every=0, save_every=0,
                      optimizer=__import__("forgeline").core.config.OptimizerConfig(learning_rate=3e-3))
    model = TransformerLM(tiny_spec)
    alg = PretrainAlgorithm(model, MemmapCorpus(corpus_dir, "train", 64), PretrainConfig(batch_size=8), MemmapCorpus(corpus_dir, "val", 64))
    start = alg.evaluate(0)["val_loss"]
    Trainer(ctx, alg).fit()
    assert alg.evaluate(0)["val_loss"] < start


@pytest.mark.parametrize("adapter", ["none", "lora", "qlora"])
def test_sft(ctx_factory, tiny_spec, tokenizer, adapter):
    recs = [SFTRecord(f"Q: {i}+1?", f" {i+1}") for i in range(16)]
    pol = NativePolicy(TransformerLM(tiny_spec), tokenizer, adapter=AdapterConfig(method=adapter, rank=4) if adapter != "none" else None)
    before = {n: p.detach().clone() for n, p in pol.model.named_parameters() if not p.requires_grad}
    s = Trainer(ctx_factory("sft"), SFTAlgorithm(pol, SupervisedDataset(recs, tokenizer, 32), SFTConfig(batch_size=4))).fit()
    assert finite(s)
    if adapter != "none":
        for n, p in pol.model.named_parameters():
            if n in before:
                assert torch.equal(before[n], p)  # frozen base untouched


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_dpo_increases_preference_margin(ctx_factory, tiny_spec, tokenizer, reduction):
    prefs = [PreferenceRecord(f"Q{i}:", " yes", " no") for i in range(8)]
    pol = NativePolicy(TransformerLM(tiny_spec), tokenizer)
    ds = PreferenceDataset(prefs, tokenizer, 16, 8)
    from forgeline.core.config import OptimizerConfig
    alg = DPOAlgorithm(pol, ds, DPOConfig(batch_size=4, beta=0.5, logprob_reduction=reduction))
    m0 = alg.evaluate(0)["reward_margin"]
    Trainer(ctx_factory("dpo", max_steps=15, eval_every=0, save_every=0, optimizer=OptimizerConfig(learning_rate=5e-3, weight_decay=0.0)), alg).fit()
    ev = alg.evaluate(0)
    assert ev["reward_margin"] > m0 and ev["preference_accuracy"] == 1.0


def test_reward_model_learns_ranking(ctx_factory, tiny_spec, tokenizer):
    from forgeline.core.config import OptimizerConfig
    prefs = [PreferenceRecord(f"Q{i}:", " good good good", " bad") for i in range(8)]
    alg = RewardModelAlgorithm(SequenceRewardModel(tiny_spec), PreferenceDataset(prefs, tokenizer, 16, 16), RewardModelConfig(batch_size=4))
    Trainer(ctx_factory("reward_model", max_steps=20, eval_every=0, save_every=0, optimizer=OptimizerConfig(learning_rate=3e-3)), alg).fit()
    assert alg.evaluate(0)["accuracy"] == 1.0


@pytest.mark.parametrize("ratio_level,kl_mode,entropy_mode", [("sequence_mean", "monitor", "sampled_logprob"), ("token", "penalty", "full"), ("sequence_mean", "reward", "sampled_logprob")])
def test_ppo(ctx_factory, tiny_spec, tokenizer, ratio_level, kl_mode, entropy_mode):
    pol = NativePolicy(TransformerLM(tiny_spec), tokenizer, value_head=True)
    prompts = [torch.tensor(tokenizer.encode(f"Q{i}:")) for i in range(4)]
    alg = PPOAlgorithm(pol, prompts, LengthReward(6), PPOConfig(prompts_per_step=2, samples_per_prompt=2, ppo_epochs=2, ratio_level=ratio_level,
                                                               kl_mode=kl_mode, entropy_mode=entropy_mode), RolloutConfig(max_new_tokens=6))
    s = Trainer(ctx_factory("ppo", eval_every=0), alg).fit()
    assert finite(s)
    last = Trainer.__dict__  # trainer history is checked via summary
    assert "kl" in alg.loss(alg.collect(0), 0, 1)[1]


@pytest.mark.parametrize("objective,ratio_level", [("clipped", "sequence_mean"), ("clipped", "token"), ("reinforce", "sequence_mean")])
def test_grpo(ctx_factory, tiny_spec, tokenizer, objective, ratio_level):
    tasks = [{"prompt": f"What is {i}+1?", "answer": str(i + 1)} for i in range(4)]
    pol = NativePolicy(TransformerLM(tiny_spec), tokenizer)
    prompts = [torch.tensor(tokenizer.encode(t["prompt"])) for t in tasks]
    alg = GRPOAlgorithm(pol, prompts, VerifierReward(MathVerifier()), GRPOConfig(group_size=3, prompts_per_step=2, objective=objective, ratio_level=ratio_level),
                        RolloutConfig(max_new_tokens=6), tasks=tasks)
    assert finite(Trainer(ctx_factory("grpo"), alg).fit())


def test_agent_grpo_runs_tools(ctx_factory, tokenizer):
    spec = model_spec_from_preset("tiny", vocab_size=tokenizer.vocab_size, block_size=256)
    pol = NativePolicy(TransformerLM(spec), tokenizer)
    tasks = [{"prompt": "Compute 12 * 3.", "answer": "36"}]
    alg = GRPOAlgorithm(pol, tasks, build_reward_provider({"type": "tagged"}), GRPOConfig(group_size=2, prompts_per_step=1, agent=True),
                        agent_rollout=AgentRolloutConfig(max_turns=1, max_new_tokens=6, max_context_tokens=200))
    assert finite(Trainer(ctx_factory("grpo", max_steps=1, eval_every=0), alg).fit())


def test_agent_rollout_executes_tool_calls(tokenizer):
    """A scripted policy emits a tool call; the engine injects the executor result and conditions on it."""
    from forgeline.rollouts.agent import AgentRolloutEngine

    class Scripted:
        device = torch.device("cpu")
        pad_token_id = 0

        def __init__(self):
            self.calls = 0
            self.contexts = []

        def encode(self, t):
            return tokenizer.encode(t)

        def decode(self, ids):
            return self.outputs[self.calls - 1]

        outputs = ['<tool_call>{"name": "python_executor", "args": {"code": "print(12*3)"}}</tool_call>', "<final_answer>36</final_answer>"]

        def generate(self, ids, settings):
            self.contexts.append(tokenizer.decode(ids[0].tolist()))
            self.calls += 1
            return torch.zeros(1, 1, dtype=torch.long)

    pol = Scripted()
    segs, blocks, final = AgentRolloutEngine(pol, AgentRolloutConfig(max_context_tokens=100000)).run_episode("Solve")
    assert len(segs) == 2 and "36" in blocks[0] and "<tool_result>" in pol.contexts[1] and final.endswith("</final_answer>")


def test_dapo_dynamic_sampling_skips_constant_reward(ctx_factory, tiny_spec, tokenizer):
    pol = NativePolicy(TransformerLM(tiny_spec), tokenizer)
    prompts = [torch.tensor(tokenizer.encode("Q:"))]
    alg = DAPOAlgorithm(pol, prompts, ConstantReward(1.0), DAPOConfig(group_size=3, prompts_per_step=2), RolloutConfig(max_new_tokens=4))
    groups = alg.collect(0)
    assert all(g.skipped for g in groups)
    loss, metrics = alg.loss(groups, 0, 1)
    assert loss.item() == 0.0 and metrics["groups_skipped"] == 2.0
    alg2 = DAPOAlgorithm(pol, prompts, LengthReward(2), DAPOConfig(group_size=4, prompts_per_step=2, entropy_coef=0.01), RolloutConfig(max_new_tokens=6, temperature=1.5))
    assert finite(Trainer(ctx_factory("dapo"), alg2).fit())


def test_rlvr_with_format_verifier(ctx_factory, tokenizer):
    spec = model_spec_from_preset("tiny", vocab_size=tokenizer.vocab_size, block_size=64)
    pol = NativePolicy(TransformerLM(spec), tokenizer)
    tasks = [VerifiableTask(prompt=f"What is {i}+1?", answer=str(i + 1)) for i in range(4)]
    for opt in ("dapo", "grpo"):
        alg = build_rlvr_algorithm(pol, tasks, RLVRConfig(optimizer=opt, require_format=True, dapo=DAPOConfig(group_size=2, prompts_per_step=1),
                                                         grpo=GRPOConfig(group_size=2, prompts_per_step=1)), RolloutConfig(max_new_tokens=8))
        assert alg.name == "rlvr" and finite(Trainer(ctx_factory("rlvr", max_steps=2), alg).fit())


def test_distillation(ctx_factory, tiny_spec, corpus_dir):
    teacher = TransformerLM(tiny_spec)
    student = TransformerLM(model_spec_from_preset("tiny", vocab_size=tiny_spec.vocab_size, block_size=64, n_layer=1))
    alg = DistillationAlgorithm(student, teacher, MemmapCorpus(corpus_dir, "train", 64), DistillConfig(batch_size=4, feature_weight=0.1))
    assert finite(Trainer(ctx_factory("distill", eval_every=0), alg).fit())
