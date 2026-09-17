"""Preference pairs and SFT records derived from trajectory yields."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from forgeline.data.records import PreferenceRecord, SFTRecord
from forgeline.domains.synthesis.prompts import build_synthesis_prompt, conditions_to_json
from forgeline.domains.synthesis.reward import rule_score


def _yield(t: Dict[str, Any]) -> float:
    return float(t.get("outcomes", t).get("yield", t.get("yield", 0.0)))


def build_preference_pairs(trajectories: List[Dict[str, Any]], pairs_per_molecule: int = 20, seed: int = 0) -> List[Tuple[Dict, Dict]]:
    """Group by molecule, sort by yield, sample (top-half, bottom-half) pairs. Seeded."""
    rng = random.Random(seed)
    by_mol: Dict[str, List[Dict]] = {}
    for t in trajectories:
        by_mol.setdefault(t.get("molecule", "unknown"), []).append(t)
    pairs: List[Tuple[Dict, Dict]] = []
    for trajs in by_mol.values():
        if len(trajs) < 2:
            continue
        ordered = sorted(trajs, key=_yield, reverse=True)
        top, bottom = ordered[: len(ordered) // 2], ordered[len(ordered) // 2 :]
        for _ in range(pairs_per_molecule):
            if not top or not bottom:
                break
            pairs.append((rng.choice(top), rng.choice(bottom)))
    return pairs


def pairs_to_records(pairs: List[Tuple[Dict, Dict]]) -> List[PreferenceRecord]:
    out = []
    for chosen, rejected in pairs:
        c, r = conditions_to_json(chosen), conditions_to_json(rejected)
        if c == r:
            continue
        out.append(PreferenceRecord(prompt=build_synthesis_prompt(chosen), chosen=c, rejected=r,
                                    meta={"molecule": chosen.get("molecule"), "chosen_yield": _yield(chosen), "rejected_yield": _yield(rejected)}))
    return out


def filter_high_reward(trajectories: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    return [t for t in trajectories if rule_score(t) >= threshold]


def build_sft_records(trajectories: List[Dict[str, Any]]) -> List[SFTRecord]:
    return [SFTRecord(prompt=build_synthesis_prompt(t), response=conditions_to_json(t), meta={"molecule": t.get("molecule")}) for t in trajectories]
