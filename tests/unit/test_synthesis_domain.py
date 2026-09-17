import json

import pytest
import torch

from forgeline.core.errors import ConfigError, DatasetError
from forgeline.core.protocols import Trajectory
from forgeline.domains.synthesis import (
    SynthesisRuleReward, TabularActorCritic, TabularPPOConfig, TabularPPOTrainer, batch_validate, build_preference_pairs,
    build_sft_records, build_synthesis_prompt, conditions_to_json, decode_conditions, encode_trajectory, filter_high_reward,
    rule_score, trajectory_to_text, validate_conditions, validity_penalty,
)
from forgeline.domains.synthesis.preferences import pairs_to_records

FLAT_HIGH = {"yield": 0.95, "selectivity": 0.90, "safety_risk": 0.05, "steps": 3}
FLAT_LOW = {"yield": 0.20, "selectivity": 0.30, "safety_risk": 0.90, "steps": 15}
ORD = {"molecule": "Aspirin", "cas_number": "X", "parameters": {"temperature_celsius": 90.0, "time_hours": 2.5, "catalyst_loading_M": 0.10, "solvent_ratio_ml_mmol": 2.0},
       "outcomes": {"yield": 0.94, "selectivity": 0.88, "safety_risk": 0.10, "steps": 3}}


def test_rule_score_flat_and_nested():
    assert rule_score(FLAT_HIGH) > 0.8 and rule_score(FLAT_LOW) < 0.4 and rule_score(ORD) > 0.75
    assert 0 <= rule_score({}) <= 1 and rule_score({"yield": 1.0, "selectivity": 1.0, "safety_risk": 0.0, "steps": 1}) > 0.9
    assert rule_score(ORD) == pytest.approx(0.94 * 0.4 + 0.88 * 0.3 + 0.9 * 0.2 + 0.1 / 1.3)
    assert "Aspirin" in trajectory_to_text(ORD)


def test_constraints():
    clipped, ok = validate_conditions({"temperature_celsius": 999.0, "time_hours": -2.0, "catalyst_loading_M": 0.05, "solvent_ratio_ml_mmol": 3.0})
    assert not ok and clipped["temperature_celsius"] == 300.0 and clipped["time_hours"] == 0.25
    assert validate_conditions({"temperature_celsius": 80.0, "time_hours": 4.0, "catalyst_loading_M": 0.05, "solvent_ratio_ml_mmol": 3.0})[1]
    assert validate_conditions({})[0]["temperature_celsius"] == 80.0
    assert not validate_conditions({"temperature_celsius": float("nan")})[1]
    _, flags = batch_validate([{}, {"temperature_celsius": "abc"}])
    assert flags == [False, False] and validity_penalty(False) == -0.05


def test_prompt_and_decode():
    prompt = build_synthesis_prompt(ORD)
    assert "Aspirin" in prompt and prompt.endswith("(JSON):\n")
    cond, ok = decode_conditions('blah {"temperature_celsius": 120, "time_hours": 3, "catalyst_loading_M": 0.1, "solvent_ratio_ml_mmol": 2} tail')
    assert ok and cond["temperature_celsius"] == 120
    cond, ok = decode_conditions("no json at all")
    # unparsable output falls back to the default conditions, which are in-bounds (valid)
    assert ok and cond["temperature_celsius"] == 80.0
    assert json.loads(conditions_to_json(ORD))["time_hours"] == 2.5


def test_features():
    v = encode_trajectory(ORD)
    assert v.shape == (128,) and abs(v[0].item() - 0.45) < 1e-5 and torch.isfinite(v).all()
    assert encode_trajectory({}).shape == (128,)


def test_preferences_and_sft():
    rows = [dict(ORD, molecule="Aspirin", outcomes=dict(ORD["outcomes"], yield_=0), procedure_id=i) for i in range(10)]
    for i, r in enumerate(rows):
        r["outcomes"] = {**ORD["outcomes"], "yield": 0.5 + 0.04 * i}
    pairs = build_preference_pairs(rows, pairs_per_molecule=5, seed=1)
    assert len(pairs) == 5 and all(c["outcomes"]["yield"] >= r["outcomes"]["yield"] for c, r in pairs)
    assert build_preference_pairs(rows, 5, seed=1) == pairs  # seeded
    assert build_preference_pairs([], 5) == []
    recs = pairs_to_records(pairs)
    assert all(r.chosen != r.rejected for r in recs)
    assert len(filter_high_reward(rows, 0.8)) <= 10 and len(build_sft_records(rows)) == 10


def test_reward_provider_and_tabular_ppo():
    t = Trajectory(prompt_ids=torch.tensor([1]), response_ids=torch.tensor([2]), response_text='{"temperature_celsius": 90, "time_hours": 2, "catalyst_loading_M": 0.1, "solvent_ratio_ml_mmol": 2}', task=ORD)
    r = SynthesisRuleReward().score(t)
    assert r.passed and r.info["conditions_valid"]
    rows = [{**ORD, "outcomes": {**ORD["outcomes"], "yield": 0.5 + 0.01 * i}} for i in range(32)]
    trainer = TabularPPOTrainer(TabularActorCritic(), TabularPPOConfig(epochs=2, batch_size=8))
    m = trainer.train(rows)
    assert len(m["epoch_rewards"]) == 2 and all(torch.isfinite(torch.tensor(m["epoch_losses"])))
    with pytest.raises(DatasetError):
        trainer.train([])
    with pytest.raises(ConfigError):
        TabularActorCritic(state_dim=0)
    with pytest.raises(ConfigError):
        TabularPPOTrainer(TabularActorCritic(), TabularPPOConfig(learning_rate=0))
