"""Local JSON-backed candidate registry (no database or cloud service)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from forgeline.core.errors import RegistryError
from forgeline.deployment.candidates import CandidateState, ModelCandidate
from forgeline.observability.events import Event
from forgeline.observability.logging import get_logger

log = get_logger("forgeline.registry")


class CandidateRegistry:
    def __init__(self, path: str | os.PathLike, metrics=None):
        self.path = Path(path)
        self.metrics = metrics
        self._candidates: Dict[str, ModelCandidate] = {}
        if self.path.exists():
            self._load()

    # ── persistence ──────────────────────────────────────────────────────
    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text())
        except json.JSONDecodeError as exc:
            raise RegistryError(f"registry file {self.path} is corrupt: {exc}") from None
        self._candidates = {c["id"]: ModelCandidate.from_dict(c) for c in data.get("candidates", [])}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 1, "candidates": [c.to_dict() for c in self._candidates.values()]}
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self.path)

    def _event(self, name: str, payload: dict) -> None:
        if self.metrics is not None:
            self.metrics.event(name, payload)

    # ── CRUD ─────────────────────────────────────────────────────────────
    def register(self, candidate: ModelCandidate) -> ModelCandidate:
        if any(c.key == candidate.key for c in self._candidates.values()):
            raise RegistryError(f"candidate {candidate.key} already registered")
        self._candidates[candidate.id] = candidate
        self._save()
        self._event(Event.CANDIDATE_REGISTERED.value, {"id": candidate.id, "key": candidate.key})
        log.info("candidate_registered", id=candidate.id, key=candidate.key)
        return candidate

    def get(self, candidate_id: str) -> ModelCandidate:
        try:
            return self._candidates[candidate_id]
        except KeyError:
            raise RegistryError(f"unknown candidate id {candidate_id!r}") from None

    def find(self, name: str, version: Optional[str] = None) -> List[ModelCandidate]:
        return [c for c in self._candidates.values() if c.name == name and (version is None or c.version == version)]

    def list(self, state: Optional[CandidateState] = None, name: Optional[str] = None) -> List[ModelCandidate]:
        out = [c for c in self._candidates.values() if (state is None or c.state == state) and (name is None or c.name == name)]
        return sorted(out, key=lambda c: c.created_at)

    def update(self, candidate: ModelCandidate) -> None:
        if candidate.id not in self._candidates:
            raise RegistryError(f"unknown candidate id {candidate.id!r}")
        self._candidates[candidate.id] = candidate
        self._save()

    def record_metrics(self, candidate_id: str, metrics: Dict[str, float]) -> ModelCandidate:
        c = self.get(candidate_id)
        c.metrics.update({k: float(v) for k, v in metrics.items()})
        self.update(c)
        return c

    # ── lifecycle ────────────────────────────────────────────────────────
    def champion(self, name: str) -> Optional[ModelCandidate]:
        champs = [c for c in self._candidates.values() if c.name == name and c.state == CandidateState.CHAMPION]
        if len(champs) > 1:
            raise RegistryError(f"registry has {len(champs)} champions for {name!r}; expected at most one")
        return champs[0] if champs else None

    def transition(self, candidate_id: str, new_state: CandidateState, reason: str = "") -> ModelCandidate:
        c = self.get(candidate_id)
        if new_state == CandidateState.CHAMPION:
            current = self.champion(c.name)
            if current is not None and current.id != c.id:
                current.transition(CandidateState.RETIRED, reason=f"replaced by {c.id}")
                c.rollback_target = current.id
        c.transition(new_state, reason)
        self._save()
        return c

    def retire(self, candidate_id: str, reason: str = "") -> ModelCandidate:
        return self.transition(candidate_id, CandidateState.RETIRED, reason)
