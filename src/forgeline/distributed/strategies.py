"""Distributed strategies behind one interface: single, ddp, fsdp, deepspeed,
tensor_parallel, pipeline_parallel.

Every strategy validates its configuration on CPU. ``setup`` only touches
``torch.distributed`` when a launcher is present (or a 1-process group is
explicitly requested), so unit tests never need GPUs.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from forgeline.core.config import DistributedConfig
from forgeline.core.errors import DistributedConfigError, OptionalDependencyError
from forgeline.distributed import runtime as rt
from forgeline.distributed.topology import ParallelMesh


class SingleProcessStrategy:
    name = "single"

    def __init__(self, cfg: Optional[DistributedConfig] = None):
        self.cfg = cfg or DistributedConfig()

    def setup(self) -> None:
        pass

    def wrap_model(self, model: nn.Module) -> nn.Module:
        return model

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return model

    def is_main_process(self) -> bool:
        return True

    def world_size(self) -> int:
        return 1

    def rank(self) -> int:
        return 0

    def barrier(self) -> None:
        pass

    def teardown(self) -> None:
        pass


class _ProcessGroupStrategy(SingleProcessStrategy):
    """Base for strategies that need a process group."""

    def __init__(self, cfg: DistributedConfig, allow_single_process: bool = True):
        super().__init__(cfg)
        self.allow_single_process = allow_single_process
        self.info: Optional[rt.DistributedInfo] = None

    def setup(self) -> None:
        env = rt.read_launcher_env()
        if not env.launched and not self.allow_single_process:
            raise DistributedConfigError(f"{self.name} strategy requires a torchrun launch (RANK/WORLD_SIZE not set)")
        self.info = rt.init_process_group(self.cfg.backend)

    def is_main_process(self) -> bool:
        return rt.is_main_process()

    def world_size(self) -> int:
        return rt.world_size()

    def rank(self) -> int:
        return rt.rank()

    def barrier(self) -> None:
        rt.barrier()

    def teardown(self) -> None:
        rt.destroy_process_group()

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return getattr(model, "module", model)


class DDPStrategy(_ProcessGroupStrategy):
    name = "ddp"

    def wrap_model(self, model: nn.Module) -> nn.Module:
        from torch.nn.parallel import DistributedDataParallel as DDP

        if not rt.is_initialized():
            raise DistributedConfigError("call setup() before wrap_model()")
        if rt.world_size() == 1:
            return model  # DDP over one process is a no-op; keep the raw module
        device_ids = [self.info.local_rank] if self.info and torch.cuda.is_available() else None
        return DDP(model, device_ids=device_ids, find_unused_parameters=True)


class FSDPStrategy(_ProcessGroupStrategy):
    name = "fsdp"

    def __init__(self, cfg: DistributedConfig, dtype: torch.dtype = torch.float32):
        super().__init__(cfg)
        self.dtype = dtype

    def wrap_model(self, model: nn.Module) -> nn.Module:
        if not torch.cuda.is_available():
            raise DistributedConfigError("FSDP requires CUDA devices", hint="Use strategy=single on CPU.")
        import functools

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        policy = functools.partial(size_based_auto_wrap_policy, min_num_params=self.cfg.fsdp_min_params_to_wrap)
        mp = MixedPrecision(param_dtype=self.dtype, reduce_dtype=self.dtype, buffer_dtype=self.dtype)
        return FSDP(model, auto_wrap_policy=policy, mixed_precision=mp, sharding_strategy=ShardingStrategy.FULL_SHARD,
                    device_id=self.info.local_rank if self.info else None)

    def clip_grad_norm(self, model: nn.Module, max_norm: float) -> torch.Tensor:
        return model.clip_grad_norm_(max_norm)


class DeepSpeedStrategy(_ProcessGroupStrategy):
    """ZeRO via DeepSpeed (optional extra). ``initialize`` returns the engine."""

    name = "deepspeed"

    def __init__(self, cfg: DistributedConfig):
        super().__init__(cfg)
        self.config_path = cfg.deepspeed_config

    @staticmethod
    def load_config(path: str) -> dict:
        import json
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            raise DistributedConfigError(f"deepspeed config not found: {path}")
        cfg = json.loads(p.read_text())
        if "zero_optimization" not in cfg:
            raise DistributedConfigError("deepspeed config must contain zero_optimization")
        return cfg

    def initialize(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Any:
        try:
            import deepspeed
        except ImportError as exc:
            raise OptionalDependencyError("deepspeed is not installed", hint="pip install 'forgeline[distributed]'") from exc
        cfg = self.load_config(self.config_path)
        engine, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=cfg,
                                                       model_parameters=[p for p in model.parameters() if p.requires_grad])
        return engine

    def wrap_model(self, model: nn.Module) -> nn.Module:
        return self.initialize(model)


class TensorParallelStrategy(_ProcessGroupStrategy):
    name = "tensor_parallel"

    def __init__(self, cfg: DistributedConfig):
        super().__init__(cfg)
        self.mesh: Optional[ParallelMesh] = None

    def setup(self) -> None:
        super().setup()
        self.mesh = ParallelMesh(self.cfg.tensor_parallel_size, self.cfg.pipeline_parallel_size)

    def wrap_model(self, model: nn.Module) -> nn.Module:
        from forgeline.distributed.tensor_parallel import apply_tensor_parallel

        if self.mesh is None:
            raise DistributedConfigError("call setup() before wrap_model()")
        for layer in model.transformer.h:
            apply_tensor_parallel(layer, self.mesh)
        return model


class PipelineParallelStrategy(_ProcessGroupStrategy):
    name = "pipeline_parallel"

    def __init__(self, cfg: DistributedConfig):
        super().__init__(cfg)
        self.mesh: Optional[ParallelMesh] = None

    def setup(self) -> None:
        super().setup()
        self.mesh = ParallelMesh(self.cfg.tensor_parallel_size, self.cfg.pipeline_parallel_size)

    def wrap_model(self, model: nn.Module) -> nn.Module:
        from forgeline.distributed.pipeline_parallel import build_pipeline_stage

        if self.mesh is None:
            raise DistributedConfigError("call setup() before wrap_model()")
        return build_pipeline_stage(model, self.mesh)

    def build_scheduler(self, stage: nn.Module):
        """1F1B scheduler for a wrapped stage, using ``distributed.pipeline_micro_batches``."""
        from forgeline.distributed.pipeline_parallel import PipelineScheduler

        if self.mesh is None:
            raise DistributedConfigError("call setup() before build_scheduler()")
        return PipelineScheduler(stage, self.mesh, n_micro=self.cfg.pipeline_micro_batches)


STRATEGIES = {
    "single": SingleProcessStrategy, "ddp": DDPStrategy, "fsdp": FSDPStrategy, "deepspeed": DeepSpeedStrategy,
    "tensor_parallel": TensorParallelStrategy, "pipeline_parallel": PipelineParallelStrategy,
}


def build_strategy(cfg: DistributedConfig, **kwargs: Any):
    cfg.validate()
    cls = STRATEGIES[cfg.strategy]
    return cls(cfg, **kwargs) if cfg.strategy != "single" else cls(cfg)


def validate_distributed_config(cfg: DistributedConfig, world_size: Optional[int] = None) -> dict:
    """CPU-only validation: returns the implied topology or raises DistributedConfigError."""
    cfg.validate()
    ws = world_size if world_size is not None else rt.read_launcher_env().world_size
    dp = rt.validate_topology(ws, cfg.tensor_parallel_size, cfg.pipeline_parallel_size)
    if cfg.strategy in ("tensor_parallel",) and cfg.tensor_parallel_size == 1:
        raise DistributedConfigError("tensor_parallel strategy needs tensor_parallel_size > 1")
    if cfg.strategy in ("pipeline_parallel",) and cfg.pipeline_parallel_size == 1:
        raise DistributedConfigError("pipeline_parallel strategy needs pipeline_parallel_size > 1")
    if cfg.strategy == "deepspeed":
        DeepSpeedStrategy.load_config(cfg.deepspeed_config)
    return {"world_size": ws, "tensor_parallel": cfg.tensor_parallel_size, "pipeline_parallel": cfg.pipeline_parallel_size, "data_parallel": dp}
