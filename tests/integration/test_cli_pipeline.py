"""End-to-end CLI: prepare → pretrain → SFT → evaluate → export → registry → serve (in-process)."""

import json
import urllib.request
from pathlib import Path

import pytest

from forgeline.cli.main import main

ROOT = Path(__file__).resolve().parents[2]


def run(argv):
    assert main([str(a) for a in argv]) == 0, argv


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cli")
    corpus = tmp / "corpus.txt"
    corpus.write_text("Q: What is 2 + 3? A: 5\nthe quick brown fox\n" * 200)
    run(["data", "prepare", "--input", corpus, "--output", tmp / "prepared", "--tokenizer", "char"])
    run(["train", ROOT / "configs/training/pretrain_tiny_cpu.yaml", "--max-steps", 6, "--output-dir", tmp / "pre",
         "--set", f"data.path={tmp / 'prepared'}", "--set", "trainer.eval_every=3", "--set", "trainer.checkpoint.save_every=0"])
    run(["train", ROOT / "configs/training/sft_tiny_cpu.yaml", "--max-steps", 4, "--output-dir", tmp / "sft",
         "--set", f"checkpoint_path={tmp / 'pre/checkpoints/final'}", "--set", f"data.path={ROOT / 'data/samples/supervised/arithmetic_sft.jsonl'}"])
    return tmp


def test_train_outputs(trained):
    assert (trained / "pre/checkpoints/final/manifest.json").exists() and (trained / "pre/manifest.yaml").exists()
    assert (trained / "pre/metrics.jsonl").read_text().strip()
    events = [json.loads(l)["event"] for l in (trained / "sft/events.jsonl").read_text().splitlines()]
    assert "run.started" in events and "run.finished" in events and "checkpoint.saved" in events


@pytest.mark.parametrize("manifest", sorted(p.relative_to(ROOT) for p in (ROOT / "configs").rglob("*.yaml") if p.parent.name in ("training", "post_training", "models")))
def test_all_manifests_validate(manifest):
    assert main(["validate", str(ROOT / manifest)]) == 0


def test_generate_evaluate_export_registry(trained, tmp_path, capsys):
    ck = trained / "sft/checkpoints/final"
    run(["generate", "--checkpoint", ck, "--prompt", "Q: What is 1 + 1?\nA:", "--max-tokens", 4, "--temperature", 0])
    run(["evaluate", "--checkpoint", ck, "--suites", "gsm8k", "arc", "--offline", "--max-tokens", 4, "--output", tmp_path / "eval.json"])
    assert json.loads((tmp_path / "eval.json").read_text())[0]["suite"] == "benchmark/gsm8k"
    run(["export", "--checkpoint", ck, "--output", tmp_path / "m.gguf", "--quantize", "q8_0"])
    reg = tmp_path / "reg.json"
    run(["registry", "--registry", reg, "register", "--name", "a", "--version", "v1", "--algorithm", "sft", "--checkpoint", ck,
         "--metrics-file", tmp_path / "eval.json"])
    cid = json.loads(reg.read_text())["candidates"][0]["id"]
    capsys.readouterr()
    assert main(["registry", "--registry", str(reg), "promote", "--id", cid, "--to", "champion", "--gate", str(ROOT / "configs/deployment/promotion_gate.yaml")]) == 2
    assert json.loads(capsys.readouterr().out)["promoted"] is False  # accuracy/reward metrics absent → fail closed


def test_http_server_roundtrip(trained):
    from forgeline.inference.engine import InferenceEngine
    from forgeline.models.loading import load_policy_from_checkpoint
    from forgeline.serving import EngineBackend, Router, ServingServer

    pol = load_policy_from_checkpoint(trained / "sft/checkpoints/final")
    backend = EngineBackend("tiny", InferenceEngine(pol.model, max_batch=2), pol.tokenizer)
    server = ServingServer(Router({"tiny": backend}, "tiny"), port=0)
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        assert json.loads(urllib.request.urlopen(base + "/health").read())["status"] == "ok"
        req = urllib.request.Request(base + "/v1/completions", data=json.dumps({"prompt": "Q:", "max_tokens": 3}).encode(),
                                     headers={"Content-Type": "application/json"})
        body = json.loads(urllib.request.urlopen(req).read())
        assert body["usage"]["completion_tokens"] == 3
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "stream": True}).encode())
        stream = urllib.request.urlopen(req).read().decode()
        assert stream.strip().endswith("data: [DONE]")
    finally:
        backend.close()
        server.stop()


def test_rlaif_and_synthesis_commands(tmp_path):
    run(["rlaif", "pairwise", "--output", tmp_path / "rlaif"])
    stats = json.loads((tmp_path / "rlaif/rlaif_stats.json").read_text())
    assert stats["n_pairs"] == 60 and stats["n_comparisons"] == 60
    run(["rlaif", "constitutional", "--output", tmp_path / "cai"])
    assert json.loads((tmp_path / "cai/constitutional_stats.json").read_text())["sft_records"] == 10
    run(["synthesis-ppo", "--data", ROOT / "data/samples/synthesis/trajectories_literature.jsonl", "--epochs", 1, "--output", tmp_path / "s.json"])
    out = json.loads((tmp_path / "s.json").read_text())
    assert out["heldout_record_reward"]["details"]["policy_dependent"] is False


def test_star_and_rlaif_trainers_run(tmp_path, policy, tokenizer):
    from forgeline.data.records import VerifiableTask
    from forgeline.training import RLAIFConfig, RLAIFTrainer, STaRConfig, STaRTrainer
    from forgeline.training.star import HillClimbConfig, HillClimber

    tasks = [VerifiableTask(prompt=f"{i}+1", answer=str(i + 1)) for i in range(4)]
    star = STaRTrainer(policy, tasks, STaRConfig(num_rounds=1, samples_per_problem=1, max_new_tokens=4, num_problems=2), tmp_path / "star").run()
    assert star["history"][0]["round"] == 0 and (tmp_path / "star/star_results.json").exists()
    rl = RLAIFTrainer(policy, tasks, RLAIFConfig(num_rounds=1, candidates_per_prompt=2, pair_gap=0.0, max_new_tokens=4, num_problems=2),
                      tmp_path / "rlaif", judge_fn=lambda p, r: len(r) / 10.0).run()
    assert rl["history"]
    from forgeline.core.config import model_spec_from_preset
    from forgeline.models import NativePolicy, TransformerLM
    from forgeline.rollouts.agent import AgentRolloutConfig

    big = NativePolicy(TransformerLM(model_spec_from_preset("tiny", vocab_size=tokenizer.vocab_size, block_size=512)), tokenizer)
    hc = HillClimber(big, tasks, HillClimbConfig(num_rounds=1, rollouts_per_problem=1, grpo_steps_per_round=1, grpo_group_size=2, grpo_prompts_per_step=1, eval_problems=1),
                     tmp_path / "hc", agent_config=AgentRolloutConfig(max_turns=0, max_new_tokens=4, max_context_tokens=400)).run()
    assert [r["round"] for r in hc] == [0, 1]
