"""Regression tests pinning the computations behind recorded measurements.

Each test re-implements the reference formula independently and checks the
framework produces the identical number, so a refactor cannot silently change
the mathematics a benchmark depends on.
"""

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from forgeline.core.config import model_spec_from_preset, ScheduleConfig
from forgeline.domains.synthesis import rule_score
from forgeline.evaluation.reward import reward_statistics
from forgeline.rollouts.rewards.process import compute_process_reward
from forgeline.rollouts.verifiers import TaggedAnswerVerifier, tag_format_reward
from forgeline.training.common.schedule import learning_rate_at
from forgeline.training.dpo import dpo_loss
from forgeline.training.ppo import ppo_clipped_objective
from forgeline.training.common.logprobs import masked_mean, reduce_logprobs
from forgeline.models.transformer.model import TransformerLM

ROOT = Path(__file__).resolve().parents[2]


def test_pretraining_loss_is_token_cross_entropy():
    m = TransformerLM(model_spec_from_preset("tiny", vocab_size=33))
    x = torch.randint(0, 33, (3, 10))
    logits, loss = m(x, x)
    assert loss.item() == pytest.approx(F.cross_entropy(logits.view(-1, 33), x.view(-1)).item(), rel=1e-6)


def test_cosine_schedule_reference_formula():
    cfg = ScheduleConfig(name="cosine", warmup_steps=100, decay_steps=5000, min_lr=3e-5)
    for it in (0, 50, 100, 1500, 4999, 5000, 6000):
        if it < 100:
            ref = 3e-4 * it / 100
        elif it > 5000:
            ref = 3e-5
        else:
            ref = 3e-5 + 0.5 * (1 + math.cos(math.pi * (it - 100) / 4900)) * (3e-4 - 3e-5)
        assert learning_rate_at(it, 3e-4, cfg) == pytest.approx(ref)


def test_synthesis_rule_reward_reference():
    rec = {"outcomes": {"yield": 0.943, "selectivity": 0.878, "safety_risk": 0.15, "steps": 3}}
    ref = 0.943 * 0.4 + 0.878 * 0.3 + 0.85 * 0.2 + (1 / 1.3) * 0.1
    assert rule_score(rec) == pytest.approx(ref)


def test_heldout_record_reward_reproduces_recorded_statistics():
    """The recorded held-out evaluation (mean 0.8080, 100/100 > 0.75) is the rule score of the last 100 literature records."""
    rows = [json.loads(l) for l in (ROOT / "data/samples/synthesis/trajectories_literature.jsonl").read_text().splitlines()]
    stats = reward_statistics([rule_score(r) for r in rows[400:]], baseline=0.5)
    assert stats["mean"] == pytest.approx(0.8079880769230771, abs=1e-9)
    assert stats["std"] == pytest.approx(0.015118150515191983, abs=1e-9)
    assert stats["pct_above_0.75"] == 100.0
    assert stats["improvement_vs_baseline_pct"] == pytest.approx(61.59761538461541, abs=1e-6)


def test_dpo_mean_reduction_reference():
    lp_c = torch.tensor([[-1.0, -2.0, -3.0]]); lp_r = torch.tensor([[-2.0, -2.0]])
    ref_c = torch.tensor([[-1.5, -2.0, -3.0]]); ref_r = torch.tensor([[-1.0, -2.0]])
    beta = 0.1
    ours = dpo_loss(reduce_logprobs(lp_c, None, "mean"), reduce_logprobs(lp_r, None, "mean"),
                    reduce_logprobs(ref_c, None, "mean"), reduce_logprobs(ref_r, None, "mean"), beta)["loss"]
    ref = -F.logsigmoid(beta * (lp_c.mean() - ref_c.mean()) - beta * (lp_r.mean() - ref_r.mean()))
    assert ours.item() == pytest.approx(ref.item())


def test_sequence_mean_ppo_ratio_reference():
    new = torch.tensor([[-1.0, -1.2], [-0.5, -0.7]]); old = torch.tensor([[-1.1, -1.0], [-0.6, -0.6]])
    adv = torch.tensor([1.0, -1.0])
    ratio = torch.exp((new - old).mean(dim=-1))
    ref = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).mean()
    ours = ppo_clipped_objective(torch.exp(masked_mean(new - old, None)), adv, 0.2).mean()
    assert ours.item() == pytest.approx(ref.item())


def test_tagged_answer_reward_schedule():
    v = TaggedAnswerVerifier()
    assert v.verify("<final_answer>5200</final_answer>", "5200").value + tag_format_reward("<final_answer>5200</final_answer>") == pytest.approx(1.02)
    assert v.verify("<tool_call>x</tool_call><tool_result>1</tool_result>", "3").value == 0.1


def test_process_reward_reference():
    resp = "Step 1: a = 3 * 8 = 24\n\nStep 2: b = 24 / 2 = 12\n\nStep 3: total = 24 + 12 = 36\n<final_answer>36</final_answer>"
    total, scores = compute_process_reward(resp, "36", gamma=0.9, step_weight=0.5, execute_code=False)
    steps = scores[:-1]
    T = len(steps)
    ref = 0.5 * sum(0.9 ** (T - 1 - t) * r for t, r in enumerate(steps)) + scores[-1]
    assert total == pytest.approx(ref) and scores[-1] == 1.0


def test_nf4_table_and_block_size_unchanged():
    from forgeline.models.quantization.nf4 import NF4_BLOCK_SIZE, NF4_TABLE

    assert NF4_BLOCK_SIZE == 64 and NF4_TABLE.numel() == 16 and NF4_TABLE[0] == -1.0 and NF4_TABLE[7] == 0.0 and NF4_TABLE[15] == 1.0


def test_leave_one_molecule_out_reproduces_recorded_values():
    """Per-epoch training reward of the tabular trainer is the rule reward of the training records, so the recorded
    leave-one-molecule-out table is fully determined by the data and must reproduce exactly."""
    from forgeline.domains.synthesis import TabularActorCritic, TabularPPOConfig, TabularPPOTrainer, load_trajectories
    from forgeline.evaluation import RecordRewardSuite

    recorded = {"Aspirin": (0.8248230838775634, 0.8965930769230771), "Ibuprofen": (0.8360318326950074, 0.851758076923077),
                "Naproxen": (0.858636827468872, 0.761338076923077)}
    rows = load_trajectories(ROOT / "data/samples/synthesis/trajectories_literature.jsonl")
    for mol, (train_ref, test_ref) in recorded.items():
        train = [r for r in rows if r["molecule"] != mol]
        test = [r for r in rows if r["molecule"] == mol]
        m = TabularPPOTrainer(TabularActorCritic(), TabularPPOConfig(epochs=1)).train(train)
        assert m["final_reward"] == pytest.approx(train_ref, abs=1e-6)
        assert RecordRewardSuite(test, rule_score).run().metrics["mean"] == pytest.approx(test_ref, abs=1e-9)


def test_synthesis_generator_is_deterministic():
    from forgeline.domains.synthesis.generator import simulate_dataset, yield_series_trajectories

    assert simulate_dataset(5, seed=3) == simulate_dataset(5, seed=3) and len(simulate_dataset(5)) == 25
    assert len(yield_series_trajectories()) == 50


def test_synthesis_rule_reward_ignores_generated_text_for_nested_records():
    """Pins a property the recorded value-head PPO numbers depend on: for records with nested ``outcomes`` the rule
    reward is determined by the dataset record, not by the conditions (or even outcome keys) the policy generates."""
    from forgeline.core.protocols import Trajectory
    from forgeline.domains.synthesis.reward import SynthesisRuleReward

    rows = [json.loads(l) for l in (ROOT / "data/samples/synthesis/trajectories_literature.jsonl").read_text().splitlines()[:20]]
    reward = SynthesisRuleReward(decode_from_text=True)
    texts = ['{"temperature_celsius": 25, "time_hours": 1}', '{"temperature_celsius": 299, "yield": 1.0, "selectivity": 1.0}',
             "no json at all"]
    for row in rows:
        values = {reward.score(Trajectory(torch.tensor([1]), torch.tensor([2]), response_text=t, task=row)).value for t in texts}
        assert values == {rule_score(row)}


def _split_means(rows, score, n_train):
    tr = [score(r) for r in rows[:n_train]]
    te = [score(r) for r in rows[n_train:]]
    return sum(tr) / len(tr), sum(te) / len(te), te


def test_labeled_record_statistics_reproduce_recorded_values():
    base = ROOT / "data/samples/synthesis"
    rows = [json.loads(l) for l in (base / "trajectories_labeled_expanded.jsonl").read_text().splitlines()]
    train, test, te = _split_means(rows, rule_score, 400)
    assert abs(train - 0.8576667964458465) < 1e-8 and abs(test - 0.8120846192985227) < 1e-12
    assert sum(x > 0.75 for x in te) == 100
    rows = [json.loads(l) for l in (base / "trajectories_labeled_small.jsonl").read_text().splitlines()]
    train, test, _ = _split_means(rows, rule_score, 80)
    assert abs(train - 0.8558842161075184) < 1e-8 and abs(test - 0.81810719368551) < 1e-12


def test_legacy_efficiency_term_reproduces_early_recorded_values():
    def legacy(r):
        o = r["outcomes"]
        return o["yield"] * 0.4 + o["selectivity"] * 0.3 + (1 - o["safety_risk"]) * 0.2 + (1 - o["steps"] / 10) * 0.1

    base = ROOT / "data/samples/synthesis"
    for name, (b, t, above) in {"trajectories_literature.jsonl": (0.84005125, 0.8010650000000001, 1),
                                "trajectories_improvable.jsonl": (0.6033490000000001, 0.5934710000000003, 30)}.items():
        rows = [json.loads(l) for l in (base / name).read_text().splitlines()]
        train, test, te = _split_means(rows, legacy, 400)
        assert abs(train - b) < 1e-9 and abs(test - t) < 1e-9
        assert sum(x > train for x in te) == above
