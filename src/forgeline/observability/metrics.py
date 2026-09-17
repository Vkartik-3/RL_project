"""Metrics sinks: local JSONL (default), no-op, Weights & Biases, TensorBoard.

External telemetry is never required; ``wandb`` / ``tensorboard`` are loaded
lazily and only when explicitly requested.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch

from forgeline.core.errors import ConfigError, OptionalDependencyError
from forgeline.observability.logging import get_logger

log = get_logger("forgeline.metrics")


def _clean(metrics: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().float().mean().item()
        if isinstance(v, bool):
            v = float(v)
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


class MetricsSink:
    """Base sink: records metrics in memory and forwards to a backend."""

    backend = "memory"

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.counters: Dict[str, int] = {}
        self._step = 0

    def log(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        if step is not None:
            self._step = step
        record = {"step": self._step, "ts": time.time(), **_clean(metrics)}
        self.history.append(record)
        self._write(record)

    def event(self, name: str, payload: Mapping[str, Any] | None = None) -> None:
        record = {"event": name, "ts": time.time(), "step": self._step, **(dict(payload or {}))}
        self.events.append(record)
        self._write_event(record)

    def increment(self, counter: str, by: int = 1) -> int:
        self.counters[counter] = self.counters.get(counter, 0) + by
        return self.counters[counter]

    def latest(self, key: str) -> Optional[float]:
        for record in reversed(self.history):
            if key in record:
                return record[key]
        return None

    def series(self, key: str) -> List[float]:
        return [r[key] for r in self.history if key in r]

    def _write(self, record: Mapping[str, Any]) -> None:  # backend hook
        pass

    def _write_event(self, record: Mapping[str, Any]) -> None:  # backend hook
        pass

    def close(self) -> None:
        pass


class NoOpMetricsSink(MetricsSink):
    backend = "noop"


class LocalJsonlMetricsSink(MetricsSink):
    """Append-only ``metrics.jsonl`` + ``events.jsonl`` in the run directory."""

    backend = "local"

    def __init__(self, output_dir: str | Path, run_name: str = "run"):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.events_path = self.output_dir / "events.jsonl"
        self.run_name = run_name

    def _write(self, record: Mapping[str, Any]) -> None:
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(_finite(record)) + "\n")

    def _write_event(self, record: Mapping[str, Any]) -> None:
        with self.events_path.open("a") as f:
            f.write(json.dumps(_finite(record), default=str) + "\n")


def _finite(record: Mapping[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in record.items():
        if isinstance(v, float) and not math.isfinite(v):
            out[k] = None
        else:
            out[k] = v
    return out


class WandbMetricsSink(MetricsSink):
    backend = "wandb"

    def __init__(self, project: str, run_name: str, config: Mapping[str, Any] | None = None):
        super().__init__()
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - optional
            raise OptionalDependencyError("wandb is not installed", hint="pip install 'forgeline[observability]'") from exc
        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name, config=dict(config or {}), resume="allow")

    def _write(self, record: Mapping[str, Any]) -> None:  # pragma: no cover - optional
        payload = {k: v for k, v in record.items() if k not in ("ts",)}
        self._wandb.log(payload, step=int(record["step"]))

    def close(self) -> None:  # pragma: no cover - optional
        self._wandb.finish()


class TensorBoardMetricsSink(MetricsSink):
    backend = "tensorboard"

    def __init__(self, log_dir: str | Path):
        super().__init__()
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:  # pragma: no cover - optional
            raise OptionalDependencyError("tensorboard is not installed", hint="pip install 'forgeline[observability]'") from exc
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def _write(self, record: Mapping[str, Any]) -> None:  # pragma: no cover - optional
        for k, v in record.items():
            if k in ("step", "ts"):
                continue
            self._writer.add_scalar(k, v, int(record["step"]))

    def close(self) -> None:  # pragma: no cover - optional
        self._writer.close()


def build_metrics_sink(
    backend: str,
    *,
    output_dir: str | Path = "runs",
    run_name: str = "run",
    project: str = "forgeline",
    config: Mapping[str, Any] | None = None,
) -> MetricsSink:
    if backend in ("local", "jsonl"):
        return LocalJsonlMetricsSink(output_dir, run_name)
    if backend in ("none", "noop", "memory"):
        return NoOpMetricsSink()
    if backend == "wandb":
        return WandbMetricsSink(project=project, run_name=run_name, config=config)
    if backend == "tensorboard":
        return TensorBoardMetricsSink(Path(output_dir) / "tensorboard")
    raise ConfigError(f"Unknown metrics backend {backend!r}; use local|none|wandb|tensorboard")


def gradient_norms(model: torch.nn.Module, keys: tuple[str, ...] = ("wte", "ln_f", "lm_head")) -> Dict[str, float]:
    """Per-layer (selected) and total gradient norms for health monitoring."""
    out: Dict[str, float] = {}
    total = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        n = p.grad.detach().data.norm(2).item()
        total += n * n
        if any(k in name for k in keys):
            out[f"grad_norm/{name}"] = n
    out["grad_norm/total"] = total ** 0.5
    return out


def device_memory_metrics(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "memory/allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "memory/reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "memory/max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
    }
