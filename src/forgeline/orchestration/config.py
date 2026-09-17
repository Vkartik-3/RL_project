"""Ray orchestration settings (importable without Ray)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from forgeline.core.errors import ConfigError

PLACEMENT_STRATEGIES = ("PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD")


@dataclass
class RayConfig:
    """Where Ray runs and how worker actors are sized, placed and recovered.

    * ``address`` — ``None`` starts a private local Ray instance; ``"auto"`` or ``ray://host:port`` joins an
      existing cluster (``num_cpus``/``num_gpus`` then describe nothing and must be unset).
    * ``rollout_workers`` / ``reward_workers`` — pool sizes used by ``forgeline train`` (0 disables a pool).
    * ``cpus_per_worker`` / ``memory_per_worker_mb`` — per-actor requests for every pool; ``gpus_per_worker`` applies to
      rollout and evaluation workers only (reward and tool workers never request GPUs).
    * ``placement_strategy`` — reserve one bundle per worker in a placement group (PACK, SPREAD, ...).
    * ``max_restarts`` — Ray-level actor restarts; ``max_call_retries`` — how often a failed call is resubmitted
      to a recreated worker (state such as policy weights is re-sent before the retry).
    """

    address: Optional[str] = None
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None
    rollout_workers: int = 0
    reward_workers: int = 0
    cpus_per_worker: float = 1.0
    gpus_per_worker: float = 0.0
    memory_per_worker_mb: Optional[int] = None
    placement_strategy: Optional[str] = None
    placement_timeout_s: float = 60.0
    max_restarts: int = 1
    max_call_retries: int = 2
    task_timeout_s: Optional[float] = None
    worker_torch_threads: int = 1
    namespace: str = "forgeline"

    def validate(self) -> None:
        if self.address is not None and (self.num_cpus is not None or self.num_gpus is not None):
            raise ConfigError("orchestration.num_cpus/num_gpus only apply to a local Ray instance (address unset)")
        if self.rollout_workers < 0 or self.reward_workers < 0:
            raise ConfigError("orchestration worker counts must be >= 0")
        if self.cpus_per_worker < 0 or self.gpus_per_worker < 0:
            raise ConfigError("orchestration per-worker resources must be >= 0")
        if self.placement_strategy is not None and self.placement_strategy not in PLACEMENT_STRATEGIES:
            raise ConfigError(f"placement_strategy must be one of {PLACEMENT_STRATEGIES}")
        if self.max_restarts < -1 or self.max_call_retries < 0:
            raise ConfigError("max_restarts must be >= -1 and max_call_retries >= 0")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RayConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ConfigError(f"unknown orchestration keys: {sorted(unknown)}")
        cfg = cls(**dict(data))
        cfg.validate()
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
