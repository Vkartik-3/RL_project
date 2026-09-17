import pytest
import torch

from forgeline.core.config import DistributedConfig
from forgeline.core.errors import DistributedConfigError
from forgeline.distributed.pipeline_parallel import one_f_one_b_schedule
from forgeline.distributed.runtime import read_launcher_env, validate_topology
from forgeline.distributed.strategies import DeepSpeedStrategy, SingleProcessStrategy, build_strategy, validate_distributed_config
from forgeline.distributed.topology import ParallelMesh


def test_mesh_layout():
    m = ParallelMesh(2, 2, world_size=8, global_rank=5, create_groups=False)
    assert (m.tp_rank, m.pp_rank, m.dp_rank, m.dp_size) == (1, 0, 1, 2)
    assert m.ranks_in_tp_group(0, 1) == [4, 5] and m.ranks_in_pp_group(1, 1) == [5, 7] and m.ranks_in_dp_group(1, 0) == [1, 5]
    assert m.next_pipeline_rank == 7 and m.prev_pipeline_rank is None
    with pytest.raises(DistributedConfigError):
        ParallelMesh(3, 1, world_size=8, global_rank=0, create_groups=False)


def test_schedule_orders():
    s = one_f_one_b_schedule(3, 0, 2)
    assert s == [("F", 0), ("F", 1), ("B", 0), ("F", 2), ("B", 1), ("B", 2)]
    assert [a for a, _ in one_f_one_b_schedule(2, 1, 2)] == ["F", "B", "F", "B"]


def test_topology_validation():
    assert validate_topology(8, 2, 2) == 2
    with pytest.raises(DistributedConfigError):
        validate_topology(6, 4, 1)
    info = validate_distributed_config(DistributedConfig(strategy="single"), world_size=1)
    assert info["data_parallel"] == 1
    with pytest.raises(DistributedConfigError):
        validate_distributed_config(DistributedConfig(strategy="tensor_parallel", tensor_parallel_size=1), world_size=2)
    with pytest.raises(DistributedConfigError):
        validate_distributed_config(DistributedConfig(strategy="deepspeed", deepspeed_config="/nope.json"), world_size=1)
    assert DeepSpeedStrategy.load_config("configs/distributed/deepspeed_zero2.json")["zero_optimization"]["stage"] == 2


def test_build_strategy_and_env():
    s = build_strategy(DistributedConfig(strategy="single"))
    assert isinstance(s, SingleProcessStrategy) and s.is_main_process() and s.world_size() == 1
    assert read_launcher_env().launched is False
    with pytest.raises(DistributedConfigError):
        build_strategy(DistributedConfig(strategy="fsdp")).wrap_model(torch.nn.Linear(2, 2)) if not torch.cuda.is_available() else None
