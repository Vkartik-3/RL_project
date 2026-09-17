"""3-D process mesh: [tensor, pipeline, data] parallel groups."""

from __future__ import annotations

from typing import Optional

import torch.distributed as dist

from forgeline.core.errors import DistributedConfigError
from forgeline.distributed.runtime import validate_topology


class ParallelMesh:
    """Organises ``world_size`` ranks into TP × PP × DP groups.

    Rank layout: ``global = dp * (tp*pp) + pp_rank * tp + tp_rank``.
    """

    def __init__(self, tp_size: int = 1, pp_size: int = 1, world_size: Optional[int] = None, global_rank: Optional[int] = None,
                 create_groups: bool = True):
        if world_size is None or global_rank is None:
            if not dist.is_initialized():
                raise DistributedConfigError("ParallelMesh needs torch.distributed initialised or explicit world_size/global_rank")
            world_size, global_rank = dist.get_world_size(), dist.get_rank()
        self.world_size, self.global_rank = world_size, global_rank
        self.tp_size, self.pp_size = tp_size, pp_size
        self.dp_size = validate_topology(world_size, tp_size, pp_size)
        self.tp_rank = global_rank % tp_size
        self.pp_rank = (global_rank // tp_size) % pp_size
        self.dp_rank = global_rank // (tp_size * pp_size)
        self.tp_group = self.pp_group = self.dp_group = None
        if create_groups and dist.is_initialized():
            self._create_groups()

    def ranks_in_tp_group(self, pp: int, dp: int):
        return [dp * self.tp_size * self.pp_size + pp * self.tp_size + tp for tp in range(self.tp_size)]

    def ranks_in_pp_group(self, tp: int, dp: int):
        return [dp * self.tp_size * self.pp_size + pp * self.tp_size + tp for pp in range(self.pp_size)]

    def ranks_in_dp_group(self, tp: int, pp: int):
        return [dp * self.tp_size * self.pp_size + pp * self.tp_size + tp for dp in range(self.dp_size)]

    def _create_groups(self) -> None:
        for pp in range(self.pp_size):
            for dp in range(self.dp_size):
                ranks = self.ranks_in_tp_group(pp, dp)
                g = dist.new_group(ranks)
                if self.global_rank in ranks:
                    self.tp_group = g
        for tp in range(self.tp_size):
            for dp in range(self.dp_size):
                ranks = self.ranks_in_pp_group(tp, dp)
                g = dist.new_group(ranks)
                if self.global_rank in ranks:
                    self.pp_group = g
        for tp in range(self.tp_size):
            for pp in range(self.pp_size):
                ranks = self.ranks_in_dp_group(tp, pp)
                g = dist.new_group(ranks)
                if self.global_rank in ranks:
                    self.dp_group = g

    @property
    def prev_pipeline_rank(self) -> Optional[int]:
        return None if self.pp_rank == 0 else self.global_rank - self.tp_size

    @property
    def next_pipeline_rank(self) -> Optional[int]:
        return None if self.pp_rank == self.pp_size - 1 else self.global_rank + self.tp_size

    def describe(self) -> str:
        return (f"rank {self.global_rank}/{self.world_size}: TP {self.tp_rank}/{self.tp_size} "
                f"PP {self.pp_rank}/{self.pp_size} DP {self.dp_rank}/{self.dp_size}")
