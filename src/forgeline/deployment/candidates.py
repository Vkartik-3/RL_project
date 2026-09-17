"""Model candidates and their lifecycle states."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from forgeline.core.errors import RegistryError


class CandidateState(str, Enum):
    EXPERIMENTAL = "experimental"
    SHADOW = "shadow"
    CHALLENGER = "challenger"
    CHAMPION = "champion"
    RETIRED = "retired"


ALLOWED_TRANSITIONS: Dict[CandidateState, set] = {
    CandidateState.EXPERIMENTAL: {CandidateState.SHADOW, CandidateState.CHALLENGER, CandidateState.RETIRED},
    CandidateState.SHADOW: {CandidateState.CHALLENGER, CandidateState.EXPERIMENTAL, CandidateState.RETIRED},
    CandidateState.CHALLENGER: {CandidateState.CHAMPION, CandidateState.SHADOW, CandidateState.EXPERIMENTAL, CandidateState.RETIRED},
    CandidateState.CHAMPION: {CandidateState.RETIRED, CandidateState.CHALLENGER},
    # a retired candidate may only come back through an explicit rollback / restore
    CandidateState.RETIRED: {CandidateState.CHALLENGER, CandidateState.CHAMPION},
}


@dataclass
class ModelCandidate:
    name: str
    version: str
    algorithm: str
    checkpoint: str
    tokenizer: str = ""
    dataset: str = ""
    precision: str = "float32"
    distributed_strategy: str = "single"
    model_spec: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    state: CandidateState = CandidateState.EXPERIMENTAL
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    promotion_status: str = "none"  # none | passed | rejected
    promotion_report: Dict[str, Any] = field(default_factory=dict)
    rollback_target: Optional[str] = None  # candidate id to fall back to
    tags: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.name}:{self.version}"

    def transition(self, new_state: CandidateState, reason: str = "") -> None:
        if new_state not in ALLOWED_TRANSITIONS[self.state]:
            raise RegistryError(f"cannot move candidate {self.key} from {self.state.value} to {new_state.value}")
        self.history.append({"from": self.state.value, "to": new_state.value, "reason": reason, "ts": time.time()})
        self.state = new_state
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelCandidate":
        d = dict(d)
        d["state"] = CandidateState(d.get("state", "experimental"))
        return cls(**d)
