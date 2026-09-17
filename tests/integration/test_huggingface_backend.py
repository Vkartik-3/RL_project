"""HuggingFace backend on CPU with a tiny randomly initialised local model (no downloads)."""

import pytest
import torch

transformers = pytest.importorskip("transformers", reason="huggingface extra not installed")
pytest.importorskip("peft", reason="huggingface extra not installed")

pytestmark = pytest.mark.optional_dependency


@pytest.fixture(scope="module")
def tiny_hf_model(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import BertConfig, BertModel, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("hf")
    vocab = {c: i for i, c in enumerate(["<pad>", "<eos>", "[CLS]"] + [chr(i) for i in range(32, 127)] + ["\n"])}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    tk.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", cls_token="[CLS]")
    causal = d / "causal"
    fast.save_pretrained(causal)
    torch.manual_seed(0)
    LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                 num_key_value_heads=2, max_position_embeddings=512, pad_token_id=0, eos_token_id=1)).save_pretrained(causal)
    encoder = d / "encoder"
    fast.save_pretrained(encoder)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                         max_position_embeddings=512, pad_token_id=0)).save_pretrained(encoder)
    return d


def _manifest(tmp_path, model_dir, algorithm, data, **extra):
    from forgeline.core.config import ExperimentManifest

    return ExperimentManifest.from_dict({
        "name": f"hf-{algorithm}", "algorithm": algorithm, "output_dir": str(tmp_path / algorithm),
        "backend": {"type": "huggingface", "model_name": str(model_dir / "causal"), "lora_r": 2, "lora_alpha": 4, "gradient_checkpointing": False},
        "data": data, "trainer": {"max_steps": 2, "batch_size": 2, "eval_every": 0, "device": "cpu", "checkpoint": {"save_every": 0},
                                  "optimizer": {"learning_rate": 1.0e-3}}, **extra,
    })


@pytest.mark.parametrize("algorithm,extra", [
    ("sft", {}),
    ("dpo", {"algorithm_params": {"beta": 0.1, "logprob_reduction": "mean"}}),
    ("ppo", {"reward": {"type": "synthesis_rule"}, "algorithm_params": {"prompts_per_step": 2, "rollout": {"max_new_tokens": 8}}}),
    ("grpo", {"reward": {"type": "synthesis_rule"}, "algorithm_params": {"group_size": 2, "prompts_per_step": 1, "rollout": {"max_new_tokens": 8}}}),
    ("dapo", {"reward": {"type": "synthesis_rule"}, "algorithm_params": {"group_size": 2, "prompts_per_step": 1, "rollout": {"max_new_tokens": 8}}}),
])
def test_synthesis_pipelines_on_hf_backend(tmp_path, tiny_hf_model, algorithm, extra):
    import math

    from forgeline.core.lifecycle import RunContext
    from forgeline.training.common.trainer import Trainer
    from forgeline.training.factory import build_algorithm

    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    data = {"kind": "trajectories", "path": str(root / "data/samples/synthesis/trajectories_literature.jsonl"),
            "extra": {"reward_threshold": 0.88, "pairs_per_molecule": 2, "max_prompt_length": 160, "max_response_length": 96}}
    m = _manifest(tmp_path, tiny_hf_model, algorithm, data, **extra)
    ctx = RunContext.create(m, metrics_backend="none", device_override="cpu")
    alg = build_algorithm(m, ctx)
    frozen = {n: p.detach().clone() for n, p in alg.policy.model.named_parameters() if "lora_" not in n}
    summary = Trainer(ctx, alg).fit()
    assert math.isfinite(summary["final_loss"])
    for n, p in alg.policy.model.named_parameters():
        if n in frozen:
            assert torch.equal(frozen[n], p), n
    if algorithm == "sft":
        assert len(alg.dataset) == 139
    if algorithm == "dpo":
        assert len(alg.dataset) == 8  # 4 molecules in the ordered 80% split × 2 pairs


def test_hf_agent_grpo_and_checkpoint_roundtrip(tmp_path, tiny_hf_model):
    from forgeline.core.lifecycle import RunContext
    from forgeline.training.common.trainer import Trainer
    from forgeline.training.factory import build_algorithm

    tasks = tmp_path / "t.jsonl"
    tasks.write_text('{"prompt": "Compute 2*3.", "answer": "6"}\n')
    m = _manifest(tmp_path, tiny_hf_model, "grpo", {"kind": "verifiable", "path": str(tasks)}, reward={"type": "tagged"},
                  algorithm_params={"group_size": 2, "prompts_per_step": 1, "agent": True})
    ctx = RunContext.create(m, metrics_backend="none", device_override="cpu")
    alg = build_algorithm(m, ctx)
    alg.agent_engine.cfg.max_new_tokens = 6
    trainer = Trainer(ctx, alg)
    trainer.fit()
    lora = {n: p.detach().clone() for n, p in alg.policy.model.named_parameters() if "lora_" in n}
    m2 = _manifest(tmp_path / "b", tiny_hf_model, "grpo", {"kind": "verifiable", "path": str(tasks)}, reward={"type": "tagged"},
                   algorithm_params={"group_size": 2, "prompts_per_step": 1, "agent": True},
                   checkpoint_path=str(ctx.output_dir / "checkpoints/final"))
    alg2 = build_algorithm(m2, RunContext.create(m2, metrics_backend="none", device_override="cpu"))
    for n, p in alg2.policy.model.named_parameters():
        if n in lora:
            assert torch.allclose(p, lora[n])


def test_encoder_reward_models(tiny_hf_model):
    from forgeline.domains.synthesis import load_trajectories, rule_score, trajectory_to_text
    from forgeline.models.reward import EncoderRewardModel

    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    rows = load_trajectories(root / "data/samples/synthesis/trajectories_literature.jsonl")[:16]
    texts, targets = [trajectory_to_text(r) for r in rows], [rule_score(r) for r in rows]
    reg = EncoderRewardModel(str(tiny_hf_model / "encoder"), objective="regression")
    losses = reg.fit_regression(texts, targets, epochs=2, batch_size=8)
    assert len(losses) == 2 and all(math_isfinite(x) for x in losses)
    s = reg.score_texts(texts[:3])
    assert s.shape == (3,) and ((s >= 0) & (s <= 1)).all()
    pref = EncoderRewardModel(str(tiny_hf_model / "encoder"), objective="preference")
    pairs = [(texts[i], texts[i + 1]) for i in range(0, 8, 2)]
    assert len(pref.fit_preference(pairs, epochs=1, batch_size=2)) == 1


def math_isfinite(x):
    import math

    return math.isfinite(x)
