"""Structured evaluation results with JSON / JSONL IO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from forgeline.core.protocols import EvaluationResult


def save_results(results: Iterable[EvaluationResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.to_dict() for r in results]
    if path.suffix == ".jsonl":
        with path.open("w") as f:
            for r in payload:
                f.write(json.dumps(r, default=str) + "\n")
    else:
        path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load_results(path: str | Path) -> List[EvaluationResult]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    else:
        rows = json.loads(path.read_text())
    return [EvaluationResult(suite=r["suite"], metrics=r["metrics"], n_samples=r.get("n_samples", 0),
                             details=r.get("details", {}), per_item=r.get("per_item", [])) for r in rows]


def merge_metrics(results: Iterable[EvaluationResult]) -> Dict[str, float]:
    """Flatten ``suite/metric`` pairs into one mapping (used by promotion gates)."""
    out: Dict[str, float] = {}
    for r in results:
        for k, v in r.metrics.items():
            out[f"{r.suite}/{k}"] = v
    return out
