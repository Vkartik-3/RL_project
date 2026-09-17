"""Simulated synthesis trajectories with chemistry-shaped response surfaces.

Two generators:

* :func:`simulate_trajectory` — random sub-optimal conditions with bell-shaped
  temperature, S-shaped time, and banded catalyst/solvent effects on yield.
* :func:`yield_series_trajectories` — conditions swept around per-molecule
  optima paired with a monotone yield series.

Both are seeded and deterministic.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

MOLECULES: Dict[str, Dict[str, float | str]] = {
    "Aspirin": {"name": "Acetylsalicylic acid", "cas": "50-78-2", "optimal_yield": 0.95, "optimal_temp": 80, "optimal_time": 2.0, "molar_mass": 180.16},
    "Ibuprofen": {"name": "2-(4-isobutylphenyl)propionic acid", "cas": "15687-27-1", "optimal_yield": 0.88, "optimal_temp": 110, "optimal_time": 4.0, "molar_mass": 206.28},
    "Paracetamol": {"name": "N-(4-hydroxyphenyl)acetamide", "cas": "103-90-2", "optimal_yield": 0.92, "optimal_temp": 70, "optimal_time": 2.5, "molar_mass": 151.16},
    "Naproxen": {"name": "2-(6-methoxynaphthalen-2-yl)propionic acid", "cas": "22204-53-1", "optimal_yield": 0.75, "optimal_temp": 120, "optimal_time": 6.0, "molar_mass": 230.26},
    "Ketoprofen": {"name": "2-(3-benzoylphenyl)propionic acid", "cas": "22071-15-4", "optimal_yield": 0.82, "optimal_temp": 100, "optimal_time": 4.5, "molar_mass": 254.28},
}

YIELD_SERIES = {
    "Aspirin": [0.45, 0.52, 0.58, 0.65, 0.72, 0.78, 0.82, 0.85, 0.88, 0.91],
    "Ibuprofen": [0.35, 0.42, 0.50, 0.58, 0.65, 0.72, 0.78, 0.82, 0.85, 0.88],
    "Paracetamol": [0.40, 0.48, 0.55, 0.62, 0.70, 0.76, 0.81, 0.85, 0.88, 0.91],
    "Naproxen": [0.25, 0.32, 0.40, 0.48, 0.55, 0.62, 0.68, 0.74, 0.79, 0.83],
    "Ketoprofen": [0.30, 0.38, 0.46, 0.54, 0.61, 0.68, 0.74, 0.79, 0.83, 0.87],
}
GOOD_CONDITIONS = {
    "Aspirin": {"temp": 80, "time": 2.0, "catalyst": 0.1, "solvent": 2.0},
    "Ibuprofen": {"temp": 110, "time": 4.0, "catalyst": 0.1, "solvent": 2.0},
    "Paracetamol": {"temp": 70, "time": 2.5, "catalyst": 0.1, "solvent": 2.0},
    "Naproxen": {"temp": 120, "time": 6.0, "catalyst": 0.1, "solvent": 2.0},
    "Ketoprofen": {"temp": 100, "time": 4.5, "catalyst": 0.1, "solvent": 2.0},
}


def simulate_trajectory(molecule: str, idx: int, rng: random.Random) -> Dict:
    m = MOLECULES[molecule]
    temperature = max(30, min(150, m["optimal_temp"] + rng.randint(-40, 40)))
    time_hours = max(0.5, min(12, m["optimal_time"] + rng.gauss(0, 2.0)))
    catalyst = rng.uniform(0.01, 0.3)
    solvent = rng.uniform(0.5, 6.0)

    temp_factor = math.exp(-((abs(temperature - m["optimal_temp"]) / 30) ** 2))
    t_opt = m["optimal_time"]
    if time_hours < t_opt * 0.3:
        time_factor = 0.3
    elif time_hours < t_opt:
        time_factor = 0.5 + 0.5 * (time_hours / t_opt)
    elif time_hours <= t_opt * 2.0:
        time_factor = 1.0 - 0.1 * ((time_hours - t_opt) / t_opt)
    else:
        time_factor = max(0.4, 1.0 - 0.3 * ((time_hours - t_opt) / t_opt))
    if 0.05 <= catalyst <= 0.15:
        catalyst_factor = 1.0
    elif 0.02 <= catalyst < 0.05:
        catalyst_factor = 0.8
    elif 0.15 < catalyst <= 0.25:
        catalyst_factor = 0.85
    else:
        catalyst_factor = 0.5
    if 1.0 <= solvent <= 3.0:
        solvent_factor = 1.0
    elif 0.5 <= solvent < 1.0:
        solvent_factor = 0.85
    elif 3.0 < solvent <= 5.0:
        solvent_factor = 0.9
    else:
        solvent_factor = 0.6

    y = m["optimal_yield"] * temp_factor * time_factor * catalyst_factor * solvent_factor * rng.gauss(1.0, 0.08)
    y = max(0.25, min(0.95, y))
    selectivity = max(0.5, min(0.98, 0.7 + (y - 0.25) * 0.3))
    safety = min(0.5, 0.1 + max(0, (temperature - 80) / 100) * 0.2 + max(0, (time_hours - 4) / 10) * 0.15)
    return {
        "molecule": molecule, "cas_number": m["cas"], "molar_mass": m["molar_mass"], "procedure_id": idx,
        "procedure": f"Synthesis of {m['name']}",
        "parameters": {"temperature_celsius": round(temperature, 1), "time_hours": round(time_hours, 2),
                       "catalyst_loading_M": round(catalyst, 3), "solvent_ratio_ml_mmol": round(solvent, 2),
                       "hazards": ["Thermal risk" if temperature > 100 else "Safe", "Long reaction" if time_hours > 6 else "Normal"]},
        "outcomes": {"yield": round(y, 3), "selectivity": round(selectivity, 3), "safety_risk": round(safety, 3),
                     "steps": 3 + (2 if time_hours > 8 else 0), "molar_mass_product": m["molar_mass"]},
        "data_source": "simulated",
    }


def simulate_dataset(per_molecule: int = 100, seed: int = 0, molecules: Optional[List[str]] = None) -> List[Dict]:
    rng = random.Random(seed)
    return [simulate_trajectory(mol, i, rng) for mol in (molecules or list(MOLECULES)) for i in range(per_molecule)]


def yield_series_trajectories(seed: int = 0) -> List[Dict]:
    rng = random.Random(seed)
    out = []
    for mol, yields in YIELD_SERIES.items():
        good = GOOD_CONDITIONS[mol]
        for i, yv in enumerate(yields):
            temperature = good["temp"] + (i - 5) * 8
            time_hours = good["time"] + (i - 5) * 0.6
            out.append({
                "molecule": mol, "procedure_id": i,
                "parameters": {"temperature_celsius": round(max(30, min(150, temperature)), 1), "time_hours": round(max(0.5, min(12, time_hours)), 2),
                               "catalyst_loading_M": round(good["catalyst"] + rng.gauss(0, 0.02), 3), "solvent_ratio_ml_mmol": round(good["solvent"] + rng.gauss(0, 0.3), 2)},
                "outcomes": {"yield": round(yv, 3), "selectivity": round(0.85 + yv * 0.1, 3),
                             "safety_risk": round(0.1 + abs(temperature - good["temp"]) / 200, 3), "steps": 3 + (1 if time_hours > 5 else 0)},
            })
    return out
