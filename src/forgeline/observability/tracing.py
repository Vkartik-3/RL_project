"""Lightweight timing spans (no external tracing dependency)."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional


@dataclass
class Span:
    name: str
    start: float
    end: Optional[float] = None
    attributes: Dict[str, object] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.time()) - self.start) * 1000.0


class Tracer:
    """Collects spans in memory; suitable for latency metrics and tests."""

    def __init__(self) -> None:
        self.spans: List[Span] = []

    @contextmanager
    def span(self, name: str, **attributes: object) -> Iterator[Span]:
        s = Span(name=name, start=time.time(), attributes=dict(attributes))
        try:
            yield s
        finally:
            s.end = time.time()
            self.spans.append(s)

    def durations(self, name: str) -> List[float]:
        return [s.duration_ms for s in self.spans if s.name == name and s.end is not None]

    def summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for s in self.spans:
            d = out.setdefault(s.name, {"count": 0.0, "total_ms": 0.0, "max_ms": 0.0})
            d["count"] += 1
            d["total_ms"] += s.duration_ms
            d["max_ms"] = max(d["max_ms"], s.duration_ms)
        for d in out.values():
            d["mean_ms"] = d["total_ms"] / max(d["count"], 1.0)
        return out


_GLOBAL = Tracer()


def global_tracer() -> Tracer:
    return _GLOBAL
