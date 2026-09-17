"""Champion / challenger / shadow routing with deterministic request-id hashing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from forgeline.core.errors import ConfigError
from forgeline.deployment.feature_flags import stable_bucket


@dataclass
class RoutingDecision:
    primary: str  # backend that serves the response
    shadow: Optional[str] = None  # backend that also receives the request (response discarded)
    bucket: int = 0
    reason: str = ""


@dataclass
class RoutingPolicy:
    """``champion`` gets ``100 − challenger_percent`` of traffic; ``challenger`` the rest.

    ``shadow`` receives a copy of ``shadow_percent`` of requests without serving them.
    ``pinned`` maps explicit request ids / users to a backend (for replay and debugging).
    """

    champion: str
    challenger: Optional[str] = None
    challenger_percent: float = 0.0
    shadow: Optional[str] = None
    shadow_percent: float = 0.0
    salt: str = "forgeline-routing"
    pinned: Dict[str, str] = field(default_factory=dict)
    disabled: set = field(default_factory=set)

    def validate(self) -> None:
        if not 0.0 <= self.challenger_percent <= 100.0 or not 0.0 <= self.shadow_percent <= 100.0:
            raise ConfigError("percentages must be in [0, 100]")
        if self.challenger_percent > 0 and not self.challenger:
            raise ConfigError("challenger_percent > 0 requires a challenger backend")
        if self.shadow_percent > 0 and not self.shadow:
            raise ConfigError("shadow_percent > 0 requires a shadow backend")

    def route(self, request_id: str, requested: Optional[str] = None) -> RoutingDecision:
        self.validate()
        if requested and requested in self.pinned.values() and requested not in self.disabled:
            return RoutingDecision(primary=requested, bucket=-1, reason="explicit model request")
        if request_id in self.pinned and self.pinned[request_id] not in self.disabled:
            return RoutingDecision(primary=self.pinned[request_id], bucket=-1, reason="pinned")
        bucket = stable_bucket(request_id, self.salt)
        primary, reason = self.champion, "champion"
        if self.challenger and self.challenger not in self.disabled and bucket < self.challenger_percent * 100:
            primary, reason = self.challenger, f"challenger ({self.challenger_percent}%)"
        shadow = None
        if self.shadow and self.shadow not in self.disabled and self.shadow != primary:
            if stable_bucket(request_id, self.salt + ":shadow") < self.shadow_percent * 100:
                shadow = self.shadow
        return RoutingDecision(primary=primary, shadow=shadow, bucket=bucket, reason=reason)

    def disable_backend(self, name: str) -> None:
        self.disabled.add(name)

    def enable_backend(self, name: str) -> None:
        self.disabled.discard(name)

    def to_dict(self) -> Dict[str, Any]:
        return {"champion": self.champion, "challenger": self.challenger, "challenger_percent": self.challenger_percent,
                "shadow": self.shadow, "shadow_percent": self.shadow_percent, "salt": self.salt, "pinned": dict(self.pinned),
                "disabled": sorted(self.disabled)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RoutingPolicy":
        p = cls(champion=d["champion"], challenger=d.get("challenger"), challenger_percent=float(d.get("challenger_percent", 0.0)),
                shadow=d.get("shadow"), shadow_percent=float(d.get("shadow_percent", 0.0)), salt=d.get("salt", "forgeline-routing"),
                pinned=dict(d.get("pinned", {})), disabled=set(d.get("disabled", [])))
        p.validate()
        return p

    def save(self, path) -> None:
        from pathlib import Path

        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "RoutingPolicy":
        from pathlib import Path

        return cls.from_dict(json.loads(Path(path).read_text()))


def simulate_traffic(policy: RoutingPolicy, request_ids: Sequence[str]) -> Dict[str, float]:
    """Fraction of requests routed to each backend (offline replay / A-B sanity check)."""
    counts: Dict[str, int] = {}
    shadows = 0
    for rid in request_ids:
        d = policy.route(rid)
        counts[d.primary] = counts.get(d.primary, 0) + 1
        shadows += int(d.shadow is not None)
    n = max(len(request_ids), 1)
    out = {k: v / n for k, v in counts.items()}
    out["shadow_fraction"] = shadows / n
    return out


@dataclass
class ReplayResult:
    n: int
    per_backend: Dict[str, Dict[str, float]]


def offline_replay(backends: Dict[str, Any], prompts: Sequence[str], scorer, params=None) -> ReplayResult:
    """Send the same prompts to every backend and score outputs (offline A/B comparison)."""
    from forgeline.inference.requests import SamplingParams

    params = params or SamplingParams(max_tokens=32, temperature=0.0)
    per: Dict[str, Dict[str, float]] = {}
    for name, backend in backends.items():
        scores: List[float] = []
        for i, p in enumerate(prompts):
            result = backend.complete(p, params, request_id=f"replay-{i}")
            scores.append(float(scorer(p, result.text)))
        per[name] = {"mean_score": sum(scores) / max(len(scores), 1), "n": float(len(scores))}
    return ReplayResult(n=len(prompts), per_backend=per)
