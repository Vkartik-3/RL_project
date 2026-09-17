"""Run context: the single object threaded through a training / eval run."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from forgeline.core.config import ExperimentManifest
from forgeline.core.runtime import resolve_device, resolve_dtype, seed_everything
from forgeline.observability.logging import get_logger
from forgeline.observability.metrics import MetricsSink, build_metrics_sink


class Stage(str, Enum):
    DATA = "data"
    PRETRAIN = "pretrain"
    SFT = "sft"
    REWARD_MODEL = "reward_model"
    PREFERENCE = "preference"
    RL = "rl"
    EVALUATION = "evaluation"
    EXPORT = "export"
    SERVING = "serving"
    DEPLOYMENT = "deployment"


@dataclass
class RunContext:
    """Seed, device, dtype, output directory, logger and metrics sink for a run."""

    manifest: ExperimentManifest
    device: torch.device
    dtype: torch.dtype
    output_dir: Path
    metrics: MetricsSink
    seed: int
    started_at: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        manifest: ExperimentManifest,
        *,
        metrics_backend: str = "local",
        distributed: bool = False,
        device_override: Optional[str] = None,
    ) -> "RunContext":
        manifest.validate()
        seed = manifest.trainer.seed
        seed_everything(seed)
        device = resolve_device(device_override or manifest.trainer.device, distributed=distributed)
        dtype = resolve_dtype(manifest.trainer.precision.dtype, device)
        output_dir = Path(manifest.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = build_metrics_sink(metrics_backend, output_dir=output_dir, run_name=manifest.name,
                                     config=manifest.to_dict())
        log = get_logger("forgeline.run")
        log.info("run_context", name=manifest.name, algorithm=manifest.algorithm,
                 device=str(device), dtype=str(dtype), seed=seed, output_dir=str(output_dir))
        return cls(manifest=manifest, device=device, dtype=dtype, output_dir=output_dir,
                   metrics=metrics, seed=seed)

    @property
    def logger(self):
        return get_logger(f"forgeline.{self.manifest.algorithm}")

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def close(self) -> None:
        self.metrics.close()
