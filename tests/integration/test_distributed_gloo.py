"""Tensor/pipeline parallel code paths inside a real single-process gloo group."""

import pytest
import torch

from forgeline.core.config import DistributedConfig, model_spec_from_preset
from forgeline.distributed import (
    ParallelMesh, PipelineScheduler, apply_tensor_parallel, build_pipeline_stage, destroy_process_group, init_process_group,
)
from forgeline.distributed.strategies import DDPStrategy
from forgeline.models.transformer.model import TransformerLM


@pytest.fixture
def gloo():
    init_process_group("gloo")
    yield
    destroy_process_group()


def test_tp_size1_equivalence_and_pipeline(gloo):
    spec = model_spec_from_preset("tiny", vocab_size=30, block_size=16)
    model = TransformerLM(spec)
    x = torch.randint(0, 30, (4, 8))
    ref_logits, ref_loss = model(x, x)
    mesh = ParallelMesh(1, 1)
    for layer in model.transformer.h:
        apply_tensor_parallel(layer, mesh)
    logits, loss = model(x, x)
    assert torch.allclose(ref_logits, logits, atol=1e-5)
    stage = build_pipeline_stage(model, mesh)
    total = PipelineScheduler(stage, mesh, n_micro=2).run(x, x)
    assert abs(float(total) - float(loss.detach())) < 1e-4
    assert any(p.grad is not None for p in stage.parameters())


def test_pipeline_strategy_uses_configured_micro_batches(gloo):
    from forgeline.distributed.strategies import PipelineParallelStrategy

    strategy = PipelineParallelStrategy(DistributedConfig(strategy="pipeline_parallel", backend="gloo", pipeline_micro_batches=3))
    strategy.mesh = ParallelMesh(1, 1)
    model = TransformerLM(model_spec_from_preset("tiny", vocab_size=30, block_size=16))
    scheduler = strategy.build_scheduler(strategy.wrap_model(model))
    assert scheduler.n_micro == 3
    x = torch.randint(0, 30, (6, 8))
    assert torch.isfinite(scheduler.run(x, x))


def test_ddp_strategy_single_process(gloo):
    s = DDPStrategy(DistributedConfig(strategy="ddp", backend="gloo"))
    s.setup()
    m = torch.nn.Linear(2, 2)
    assert s.wrap_model(m) is m and s.is_main_process() and s.world_size() == 1
