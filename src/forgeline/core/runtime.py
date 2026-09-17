"""Device, dtype and seeding utilities shared by every subsystem."""

from __future__ import annotations

import os
import random
from contextlib import nullcontext
from typing import Any, ContextManager

import numpy as np
import torch

from forgeline.core.errors import ConfigError

DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def resolve_device(requested: str = "auto", *, distributed: bool = False) -> torch.device:
    """Pick the best available device.

    ``auto`` → CUDA if available, else MPS, else CPU. When running under a
    distributed launcher the local rank selects the CUDA device.
    """
    if distributed and torch.cuda.is_available():
        return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    if requested != "auto":
        dev = torch.device(requested)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise ConfigError("device=cuda requested but CUDA is not available", hint="Use device=cpu or auto.")
        return dev
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    """Mixed-precision dtype: bf16 on capable CUDA, fp16 on other CUDA, fp32 elsewhere."""
    if requested != "auto":
        if requested not in DTYPE_MAP:
            raise ConfigError(f"Unknown dtype {requested!r}; expected one of {sorted(DTYPE_MAP)}")
        return DTYPE_MAP[requested]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def autocast_context(device: torch.device, dtype: torch.dtype) -> ContextManager[Any]:
    """Autocast context for CUDA (bf16/fp16) or CPU (bf16); no-op otherwise."""
    if device.type == "cuda" and dtype in (torch.bfloat16, torch.float16):
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    if device.type == "cpu" and dtype == torch.bfloat16:
        return torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state() -> dict[str, Any]:
    """Capture RNG state for checkpointing."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def has_cuda() -> bool:
    return torch.cuda.is_available()


def cuda_device_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0
