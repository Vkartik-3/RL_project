#!/usr/bin/env python3
"""Generate simulated synthesis trajectories (JSONL)."""

import argparse
import json
from pathlib import Path

from forgeline.domains.synthesis.generator import simulate_dataset, yield_series_trajectories
from forgeline.domains.synthesis.reward import rule_score


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["simulated", "yield_series"], default="simulated")
    p.add_argument("--per-molecule", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    rows = simulate_dataset(a.per_molecule, a.seed) if a.kind == "simulated" else yield_series_trajectories(a.seed)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    rewards = [rule_score(r) for r in rows]
    print(json.dumps({"written": len(rows), "mean_rule_reward": sum(rewards) / len(rewards)}, indent=2))


if __name__ == "__main__":
    main()
