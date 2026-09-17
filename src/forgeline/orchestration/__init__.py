"""Optional process orchestration around the learner.

The Ray backend (``pip install -e '.[ray]'``) runs rollout generation, reward/verifier scoring,
sandboxed tool execution and sharded evaluation in worker actors with resource placement and
worker recovery. Gradient synchronisation stays with the ``torch.distributed`` strategies
(single / DDP / FSDP / tensor / pipeline); Ray does not replace them.

Nothing in the core framework imports Ray. Import the pools from
``forgeline.orchestration.ray_backend``; that import fails with an actionable error when Ray is not installed.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Dict, Mapping

from forgeline.core.errors import ConfigError


def ray_available() -> bool:
    return importlib.util.find_spec("ray") is not None


ORCHESTRATION_TYPES = ("local", "ray")


def validate_orchestration(config: Mapping[str, Any]) -> None:
    """Validate a manifest ``orchestration`` mapping without importing Ray."""
    if not config:
        return
    kind = config.get("type", "local")
    if kind not in ORCHESTRATION_TYPES:
        raise ConfigError(f"orchestration.type must be one of {ORCHESTRATION_TYPES}, got {kind!r}")
    if kind == "ray":
        from forgeline.orchestration.config import RayConfig

        RayConfig.from_dict({k: v for k, v in config.items() if k != "type"})


def orchestration_summary(config: Mapping[str, Any]) -> Dict[str, Any]:
    kind = (config or {}).get("type", "local")
    return {"type": kind, "ray_installed": ray_available()}


__all__ = ["ORCHESTRATION_TYPES", "orchestration_summary", "ray_available", "validate_orchestration"]
