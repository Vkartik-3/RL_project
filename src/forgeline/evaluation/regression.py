"""Regression policies: compare a candidate's metrics with a baseline's, fail closed on missing/NaN."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional


@dataclass
class RegressionRule:
    metric: str
    direction: str = "higher_is_better"  # higher_is_better | lower_is_better
    max_regression: float = 0.0  # absolute allowed drop (or rise for lower_is_better)
    relative: bool = False  # interpret max_regression as a fraction of the baseline

    def validate(self) -> None:
        if self.direction not in ("higher_is_better", "lower_is_better"):
            raise ValueError("direction must be higher_is_better|lower_is_better")
        if self.max_regression < 0:
            raise ValueError("max_regression must be >= 0")


@dataclass
class RegressionCheck:
    metric: str
    passed: bool
    candidate: Optional[float]
    baseline: Optional[float]
    reason: str


@dataclass
class RegressionReport:
    passed: bool
    checks: List[RegressionCheck] = field(default_factory=list)

    def failures(self) -> List[RegressionCheck]:
        return [c for c in self.checks if not c.passed]


def _finite(v: Optional[float]) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))


def evaluate_regression(candidate: Mapping[str, float], baseline: Mapping[str, float], rules: List[RegressionRule]) -> RegressionReport:
    checks: List[RegressionCheck] = []
    for rule in rules:
        rule.validate()
        c, b = candidate.get(rule.metric), baseline.get(rule.metric)
        if not _finite(c):
            checks.append(RegressionCheck(rule.metric, False, c, b, "candidate metric missing or non-finite"))
            continue
        if not _finite(b):
            checks.append(RegressionCheck(rule.metric, False, c, b, "baseline metric missing or non-finite"))
            continue
        allowed = rule.max_regression * (abs(b) if rule.relative else 1.0)
        if rule.direction == "higher_is_better":
            ok = c >= b - allowed
            reason = f"{c:.6g} >= {b:.6g} - {allowed:.6g}" if ok else f"{c:.6g} < {b:.6g} - {allowed:.6g}"
        else:
            ok = c <= b + allowed
            reason = f"{c:.6g} <= {b:.6g} + {allowed:.6g}" if ok else f"{c:.6g} > {b:.6g} + {allowed:.6g}"
        checks.append(RegressionCheck(rule.metric, ok, c, b, reason))
    return RegressionReport(passed=all(c.passed for c in checks) and bool(rules), checks=checks)
