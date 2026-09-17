"""Structured logging under the ``forgeline`` namespace.

``get_logger(name).info("event_name", key=value, ...)`` emits one line of
JSON (or key=value text) so logs are grep-able and machine-parsable.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

_CONFIGURED = False


class StructuredLogger:
    """Thin wrapper around :mod:`logging` with keyword fields."""

    def __init__(self, name: str):
        self._log = logging.getLogger(name)

    def _emit(self, level: int, event: str, **fields: Any) -> None:
        if not self._log.isEnabledFor(level):
            return
        payload = {"ts": round(time.time(), 3), "logger": self._log.name, "event": event}
        payload.update({k: _jsonable(v) for k, v in fields.items()})
        if os.environ.get("FORGELINE_LOG_FORMAT", "text") == "json":
            msg = json.dumps(payload, sort_keys=False)
        else:
            kv = " ".join(f"{k}={payload[k]}" for k in payload if k not in ("ts", "logger", "event"))
            msg = f"{event} {kv}".strip()
        self._log.log(level, msg)

    def debug(self, event: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit(logging.ERROR, event, **fields)

    def setLevel(self, level: int) -> None:  # noqa: N802 - logging API parity
        self._log.setLevel(level)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() == 1 else value.tolist()
    except Exception:  # pragma: no cover
        pass
    return str(value)


def configure_logging(level: str | int | None = None, stream=None) -> None:
    """Configure the root ``forgeline`` logger once (idempotent)."""
    global _CONFIGURED
    root = logging.getLogger("forgeline")
    if level is None:
        level = os.environ.get("FORGELINE_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(level)
    if not _CONFIGURED:
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True


def get_logger(name: str = "forgeline") -> StructuredLogger:
    if not name.startswith("forgeline"):
        name = f"forgeline.{name}"
    configure_logging()
    return StructuredLogger(name)
