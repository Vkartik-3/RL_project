"""Rule-based synthesis reward: 0.40·yield + 0.30·selectivity + 0.20·(1−risk) + 0.10/(1+steps/10)."""

from __future__ import annotations

from typing import Any, Dict, List

from forgeline.core.protocols import RewardResult, Trajectory
from forgeline.domains.synthesis.prompts import decode_conditions

RULE_WEIGHTS = {"yield": 0.40, "selectivity": 0.30, "safety": 0.20, "efficiency": 0.10}


def extract_outcomes(traj: Dict[str, Any]) -> Dict[str, float]:
    """Normalise flat and nested (``parameters``/``outcomes``) trajectory records."""
    out = traj.get("outcomes", {}) or {}
    params = traj.get("parameters", {}) or {}
    return {
        "yield": float(out.get("yield", traj.get("yield", 0.5))),
        "selectivity": float(out.get("selectivity", traj.get("selectivity", 0.5))),
        "safety_risk": float(out.get("safety_risk", traj.get("safety_risk", 0.5))),
        "steps": float(out.get("steps", traj.get("steps", 5))),
        "temperature": float(params.get("temperature_celsius", traj.get("temperature_celsius", 80))),
        "time": float(params.get("time_hours", traj.get("time_hours", 4))),
        "catalyst": float(params.get("catalyst_loading_M", traj.get("catalyst_loading_M", 0.05))),
        "solvent": float(params.get("solvent_ratio_ml_mmol", traj.get("solvent_ratio_ml_mmol", 3.0))),
    }


def rule_score(traj: Dict[str, Any]) -> float:
    o = extract_outcomes(traj)
    score = (o["yield"] * RULE_WEIGHTS["yield"] + o["selectivity"] * RULE_WEIGHTS["selectivity"]
             + (1.0 - o["safety_risk"]) * RULE_WEIGHTS["safety"] + (1.0 / (1.0 + o["steps"] / 10.0)) * RULE_WEIGHTS["efficiency"])
    return float(max(0.0, min(1.0, score)))


def trajectory_to_text(traj: Dict[str, Any]) -> str:
    o = extract_outcomes(traj)
    return (f"Synthesis of {traj.get('molecule', 'compound')}: temp {o['temperature']:.0f}°C, time {o['time']:.1f}h, "
            f"catalyst {o['catalyst']:.3f}M, solvent {o['solvent']:.1f}. Yield {o['yield']:.2f}, selectivity "
            f"{o['selectivity']:.2f}, safety risk {o['safety_risk']:.2f}, steps {int(o['steps'])}.")


class SynthesisRuleReward:
    """Scores a trajectory whose ``task`` holds the record; generated conditions are merged in.

    When the policy output is text, JSON conditions are decoded and merged into
    the record before scoring (matching the recorded chemistry runs).
    """

    name = "synthesis_rule"

    def __init__(self, decode_from_text: bool = True):
        self.decode_from_text = decode_from_text

    def score_record(self, record: Dict[str, Any]) -> float:
        return rule_score(record)

    def score_batch(self, records: List[Dict[str, Any]]) -> List[float]:
        return [rule_score(r) for r in records]

    def score(self, trajectory: Trajectory) -> RewardResult:
        record = dict(trajectory.task)
        valid = True
        if self.decode_from_text and trajectory.response_text:
            cond, valid = decode_conditions(trajectory.response_text)
            record.update(cond)
        value = rule_score(record)
        return RewardResult(value=value, passed=value >= 0.75, components={"rule": value}, info={"conditions_valid": valid})
