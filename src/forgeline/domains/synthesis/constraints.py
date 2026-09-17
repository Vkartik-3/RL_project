"""Hard bounds on generated synthesis conditions."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

CONDITION_BOUNDS: Dict[str, Tuple[float, float]] = {
    "temperature_celsius": (0.0, 300.0),
    "time_hours": (0.25, 48.0),
    "catalyst_loading_M": (0.001, 1.0),
    "solvent_ratio_ml_mmol": (0.5, 20.0),
}
CONDITION_DEFAULTS: Dict[str, float] = {
    "temperature_celsius": 80.0, "time_hours": 4.0, "catalyst_loading_M": 0.05, "solvent_ratio_ml_mmol": 3.0,
}


def validate_conditions(conditions: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Clip to feasible ranges; ``valid`` is False when any value was missing, non-numeric or clipped."""
    clipped = dict(conditions)
    valid = True
    for key, (lo, hi) in CONDITION_BOUNDS.items():
        raw = conditions.get(key)
        if raw is None:
            clipped[key] = CONDITION_DEFAULTS[key]
            valid = False
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            clipped[key] = CONDITION_DEFAULTS[key]
            valid = False
            continue
        if val != val or val < lo or val > hi:  # NaN or out of range
            clipped[key] = CONDITION_DEFAULTS[key] if val != val else max(lo, min(hi, val))
            valid = False
    return clipped, valid


def batch_validate(conditions_list: List[Dict[str, Any]]):
    results = [validate_conditions(c) for c in conditions_list]
    return [r[0] for r in results], [r[1] for r in results]


def validity_penalty(was_valid: bool, penalty: float = 0.05) -> float:
    return 0.0 if was_valid else -penalty
