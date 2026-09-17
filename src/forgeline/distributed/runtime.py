"""Process-group lifecycle and rank helpers (safe to call without a launcher)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from forgeline.core.errors import DistributedConfigError


@dataclass
class DistributedInfo:
    rank: int
    local_rank: int
    world_size: int
    backend: str
    launched: bool  # True when started under torchrun/deepspeed (env vars present)


def read_launcher_env() -> DistributedInfo:
    launched = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    return DistributedInfo(
        rank=int(os.environ.get("RANK", 0)), local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        world_size=int(os.environ.get("WORLD_SIZE", 1)), backend="", launched=launched,
    )


def select_backend(requested: str = "auto") -> str:
    if requested == "auto":
        return "nccl" if torch.cuda.is_available() else "gloo"
    if requested == "nccl" and not torch.cuda.is_available():
        raise DistributedConfigError("backend=nccl requires CUDA", hint="Use backend=gloo on CPU.")
    return requested


def init_process_group(backend: str = "auto", *, world_size: Optional[int] = None, rank: Optional[int] = None,
                       init_method: Optional[str] = None) -> DistributedInfo:
    """Initialise torch.distributed. Without a launcher, creates a 1-process gloo/nccl group."""
    info = read_launcher_env()
    backend = select_backend(backend)
    if not dist.is_initialized():
        ws = world_size if world_size is not None else info.world_size
        rk = rank if rank is not None else info.rank
        kwargs = {"backend": backend, "world_size": ws, "rank": rk}
        if init_method is not None:
            kwargs["init_method"] = init_method
        elif not info.launched:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", str(_free_port()))
            kwargs["init_method"] = "env://"
        dist.init_process_group(**kwargs)
    if torch.cuda.is_available():
        torch.cuda.set_device(info.local_rank)
    return DistributedInfo(rank=dist.get_rank(), local_rank=info.local_rank, world_size=dist.get_world_size(),
                           backend=backend, launched=info.launched)


def destroy_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_initialized() -> bool:
    return dist.is_initialized()


def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def barrier() -> None:
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return t
    t = t.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t / dist.get_world_size()


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def validate_topology(world: int, tensor_parallel: int, pipeline_parallel: int) -> int:
    """Return the data-parallel degree; raise when the mesh does not tile the world."""
    if tensor_parallel < 1 or pipeline_parallel < 1:
        raise DistributedConfigError("parallel degrees must be >= 1")
    if world % (tensor_parallel * pipeline_parallel) != 0:
        raise DistributedConfigError(
            f"world_size {world} is not divisible by tensor_parallel {tensor_parallel} × pipeline_parallel {pipeline_parallel}")
    return world // (tensor_parallel * pipeline_parallel)
