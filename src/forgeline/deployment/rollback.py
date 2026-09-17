"""Rollback: demote the current champion and restore its rollback target immediately."""

from __future__ import annotations

from typing import Optional

from forgeline.core.errors import RegistryError
from forgeline.deployment.candidates import CandidateState, ModelCandidate
from forgeline.deployment.registry import CandidateRegistry
from forgeline.deployment.rollout import RoutingPolicy
from forgeline.observability.events import Event


def rollback(registry: CandidateRegistry, name: str, reason: str = "manual rollback",
             routing: Optional[RoutingPolicy] = None) -> ModelCandidate:
    """Retire the champion of ``name`` and re-promote its ``rollback_target``.

    Returns the restored champion. Fails when there is no champion or no
    rollback target so callers can never end up with zero serving models.
    """
    current = registry.champion(name)
    if current is None:
        raise RegistryError(f"no champion to roll back for {name!r}")
    if not current.rollback_target:
        raise RegistryError(f"champion {current.key} has no rollback target")
    target = registry.get(current.rollback_target)
    current.transition(CandidateState.RETIRED, reason=reason)
    registry.update(current)
    target.transition(CandidateState.CHAMPION, reason=f"rollback from {current.id}: {reason}")
    target.rollback_target = None
    registry.update(target)
    if routing is not None:
        routing.champion = target.key
        if routing.challenger in (current.key, target.key):
            routing.challenger, routing.challenger_percent = None, 0.0
        routing.disable_backend(current.key)
    if registry.metrics is not None:
        registry.metrics.event(Event.CANDIDATE_ROLLED_BACK.value, {"from": current.id, "to": target.id, "reason": reason})
    return target


def disable_candidate(registry: CandidateRegistry, candidate_id: str, routing: Optional[RoutingPolicy] = None,
                      reason: str = "disabled") -> ModelCandidate:
    """Kill-switch: stop routing to a candidate without changing registry states."""
    c = registry.get(candidate_id)
    if routing is not None:
        routing.disable_backend(c.key)
    c.tags = sorted(set(c.tags) | {"disabled"})
    c.history.append({"from": c.state.value, "to": c.state.value, "reason": reason, "ts": __import__("time").time()})
    registry.update(c)
    return c
