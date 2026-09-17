"""Fail-closed promotion gates.

A :class:`PromotionGate` is a list of :class:`GateRule`s over candidate (and
optionally baseline) metrics. A candidate is promoted only when every rule
passes; a missing or non-finite metric fails its rule. Rules can be
absolute (``min``/``max``) or relative to the baseline (``min_delta`` /
``max_increase``, absolute or as a fraction).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from forgeline.core.errors import ConfigError, PromotionError


@dataclass
class GateRule:
    metric: str
    min: Optional[float] = None  # candidate >= min
    max: Optional[float] = None  # candidate <= max
    min_delta: Optional[float] = None  # candidate >= baseline + min_delta
    max_increase: Optional[float] = None  # candidate <= baseline + max_increase
    relative: bool = False  # deltas as a fraction of the baseline
    description: str = ""

    def validate(self) -> None:
        if all(v is None for v in (self.min, self.max, self.min_delta, self.max_increase)):
            raise ConfigError(f"gate rule for {self.metric!r} has no constraint")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ConfigError(f"gate rule for {self.metric!r}: min > max")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GateRule":
        rule = cls(**dict(d))
        rule.validate()
        return rule


@dataclass
class GateCheck:
    metric: str
    passed: bool
    candidate: Optional[float]
    baseline: Optional[float]
    reason: str


@dataclass
class PromotionReport:
    passed: bool
    checks: List[GateCheck] = field(default_factory=list)

    def failures(self) -> List[GateCheck]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "checks": [c.__dict__ for c in self.checks]}


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


class PromotionGate:
    """``when_no_baseline``: ``fail`` (default, fail closed) or ``skip_relative`` — relative rules are
    skipped only when there is no incumbent at all (first promotion); absolute rules always apply."""

    def __init__(self, rules: List[GateRule], require_baseline: bool = False, when_no_baseline: str = "fail"):
        if not rules:
            raise ConfigError("promotion gate needs at least one rule")
        if when_no_baseline not in ("fail", "skip_relative"):
            raise ConfigError("when_no_baseline must be fail|skip_relative")
        for r in rules:
            r.validate()
        self.rules = list(rules)
        self.require_baseline = require_baseline
        self.when_no_baseline = when_no_baseline

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "PromotionGate":
        rules = config.get("rules")
        if not isinstance(rules, list) or not rules:
            raise ConfigError("promotion config needs a non-empty 'rules' list")
        return cls([GateRule.from_dict(r) for r in rules], require_baseline=bool(config.get("require_baseline", False)),
                   when_no_baseline=str(config.get("when_no_baseline", "fail")))

    def evaluate(self, candidate: Mapping[str, float], baseline: Optional[Mapping[str, float]] = None) -> PromotionReport:
        checks: List[GateCheck] = []
        if self.require_baseline and baseline is None:
            return PromotionReport(False, [GateCheck("*", False, None, None, "baseline metrics required but missing")])
        for rule in self.rules:
            c = candidate.get(rule.metric)
            b = baseline.get(rule.metric) if baseline is not None else None
            if not _finite(c):
                checks.append(GateCheck(rule.metric, False, None if c is None else float("nan"), b, "candidate metric missing or non-finite"))
                continue
            c = float(c)
            reasons: List[str] = []
            ok = True
            if rule.min is not None and c < rule.min:
                ok, reasons = False, reasons + [f"{c:.6g} < min {rule.min:.6g}"]
            if rule.max is not None and c > rule.max:
                ok, reasons = False, reasons + [f"{c:.6g} > max {rule.max:.6g}"]
            if rule.min_delta is not None or rule.max_increase is not None:
                if baseline is None and self.when_no_baseline == "skip_relative":
                    checks.append(GateCheck(rule.metric, ok, c, None, ("ok" if ok else "; ".join(reasons)) + " (relative rule skipped: no incumbent)"))
                    continue
                if not _finite(b):
                    checks.append(GateCheck(rule.metric, False, c, b, "baseline metric missing or non-finite"))
                    continue
                b = float(b)
                scale = abs(b) if rule.relative else 1.0
                if rule.min_delta is not None and c < b + rule.min_delta * scale:
                    ok, reasons = False, reasons + [f"{c:.6g} < baseline {b:.6g} + {rule.min_delta * scale:.6g}"]
                if rule.max_increase is not None and c > b + rule.max_increase * scale:
                    ok, reasons = False, reasons + [f"{c:.6g} > baseline {b:.6g} + {rule.max_increase * scale:.6g}"]
            checks.append(GateCheck(rule.metric, ok, c, b, "ok" if ok else "; ".join(reasons)))
        return PromotionReport(passed=all(ch.passed for ch in checks), checks=checks)

    def assert_promotable(self, candidate: Mapping[str, float], baseline: Optional[Mapping[str, float]] = None) -> PromotionReport:
        report = self.evaluate(candidate, baseline)
        if not report.passed:
            failing = ", ".join(f"{c.metric} ({c.reason})" for c in report.failures())
            raise PromotionError(f"promotion rejected: {failing}")
        return report
