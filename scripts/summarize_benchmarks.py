#!/usr/bin/env python3
"""Print (or write as CSV) a table of every benchmark record: path, evidence level, headline metrics."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "benchmarks"


def flatten(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, f"{prefix}{k}.")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield f"{prefix}{k}", v


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=None)
    a = p.parse_args()
    rows = []
    for f in sorted(ROOT.rglob("results.json")):
        data = json.loads(f.read_text())
        metrics = dict(flatten(data.get("metrics", {})))
        rows.append({"record": str(f.parent.relative_to(ROOT)), "evidence": data.get("evidence", ""),
                     "revalidation_recommended": data.get("revalidation_recommended", False),
                     "metrics": "; ".join(f"{k}={v:.6g}" for k, v in metrics.items())})
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    else:
        w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
