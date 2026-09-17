"""Prompt template and JSON condition codec for the synthesis policy."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

from forgeline.domains.synthesis.constraints import CONDITION_DEFAULTS, validate_conditions

CONDITION_KEYS = ("temperature_celsius", "time_hours", "catalyst_loading_M", "solvent_ratio_ml_mmol")

PROMPT_TEMPLATE = """\
### Optimize pharmaceutical synthesis conditions

Molecule: {molecule}
CAS: {cas}
Objective: maximize yield and selectivity, minimize hazard and steps

### Optimal conditions (JSON):
"""


def build_synthesis_prompt(record: Dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(molecule=record.get("molecule", "Unknown"), cas=record.get("cas_number", "N/A"))


def conditions_to_json(record: Dict[str, Any]) -> str:
    params = record.get("parameters", record)
    cond = {k: params.get(k, record.get(k, CONDITION_DEFAULTS[k])) for k in CONDITION_KEYS}
    return json.dumps(cond)


def decode_conditions(text: str) -> Tuple[Dict[str, Any], bool]:
    """Parse the first JSON object in ``text``; fall back to defaults; clip to bounds."""
    try:
        m = re.search(r"\{[^}]+\}", text, re.DOTALL)
        raw = json.loads(m.group()) if m else dict(CONDITION_DEFAULTS)
        if not isinstance(raw, dict):
            raw = dict(CONDITION_DEFAULTS)
    except (json.JSONDecodeError, AttributeError):
        raw = dict(CONDITION_DEFAULTS)
    return validate_conditions(raw)
