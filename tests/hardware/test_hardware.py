"""Hardware-dependent paths. These skip cleanly (with a reason) when CUDA / multiple GPUs are absent."""

import os

import pytest
import torch

from forgeline.core.config import DistributedConfig, model_spec_from_preset
from forgeline.models.transformer.model import TransformerLM


@pytest.mark.cuda
def test_cuda_bf16_training_step():
    from forgeline.core.config import PrecisionConfig
    from forgeline.training.common.precision import Precision

    dev = torch.device("cuda")
    m = TransformerLM(model_spec_from_preset("tiny", vocab_size=50)).to(dev)
    prec = Precision(PrecisionConfig(dtype="bfloat16"), dev, torch.bfloat16)
    x = torch.randint(0, 50, (2, 16), device=dev)
    with prec.autocast():
        _, loss = m(x, x)
    prec.backward(loss)
    assert torch.isfinite(loss)


@pytest.mark.cuda
def test_cuda_fp16_grad_scaler_enabled():
    from forgeline.core.config import PrecisionConfig
    from forgeline.training.common.precision import Precision

    prec = Precision(PrecisionConfig(dtype="float16"), torch.device("cuda"), torch.float16)
    assert prec.scaler.is_enabled()


@pytest.mark.cuda
def test_cuda_inference_engine():
    from forgeline.inference.engine import InferenceEngine
    from forgeline.inference.requests import SamplingParams

    m = TransformerLM(model_spec_from_preset("tiny", vocab_size=50)).cuda().eval()
    r = InferenceEngine(m).generate([1, 2, 3], SamplingParams(max_tokens=4, temperature=0))
    assert len(r.generated_tokens) == 4


@pytest.mark.multi_gpu
@pytest.mark.distributed
def test_fsdp_wrap_under_torchrun():
    if "RANK" not in os.environ:
        pytest.skip("launch with: torchrun --nproc_per_node=2 -m pytest tests/hardware -m multi_gpu")
    from forgeline.distributed.strategies import FSDPStrategy

    s = FSDPStrategy(DistributedConfig(strategy="fsdp", backend="nccl"), dtype=torch.bfloat16)
    s.setup()
    wrapped = s.wrap_model(TransformerLM(model_spec_from_preset("tiny", vocab_size=50)).cuda())
    x = torch.randint(0, 50, (2, 16), device="cuda")
    _, loss = wrapped(x, x)
    loss.backward()
    s.teardown()


@pytest.mark.multi_gpu
@pytest.mark.distributed
def test_tensor_parallel_two_ranks():
    if "RANK" not in os.environ:
        pytest.skip("launch with: torchrun --nproc_per_node=2 -m pytest tests/hardware -m multi_gpu")
    from forgeline.distributed import ParallelMesh, apply_tensor_parallel, init_process_group

    init_process_group("nccl")
    mesh = ParallelMesh(2, 1)
    m = TransformerLM(model_spec_from_preset("tiny", vocab_size=50)).cuda()
    for layer in m.transformer.h:
        apply_tensor_parallel(layer, mesh)
    x = torch.randint(0, 50, (2, 16), device="cuda")
    _, loss = m(x, x)
    assert torch.isfinite(loss)


@pytest.mark.optional_dependency
def test_deepspeed_initialize():
    pytest.importorskip("deepspeed", reason="deepspeed extra not installed")
    if not torch.cuda.is_available():
        pytest.skip("DeepSpeed ZeRO requires CUDA")


@pytest.mark.cuda
def test_ray_rollout_worker_uses_assigned_gpu():
    ray = pytest.importorskip("ray")
    from forgeline.data.tokenizers import CharTokenizer
    from forgeline.models.policy import NativePolicy
    from forgeline.orchestration.config import RayConfig
    from forgeline.orchestration.ray_backend import RayRolloutEngine, shutdown_ray
    from forgeline.rollouts.engine import RolloutConfig

    tok = CharTokenizer.ascii()
    policy = NativePolicy(TransformerLM(model_spec_from_preset("tiny", vocab_size=tok.vocab_size, block_size=64)), tok)
    try:
        with RayRolloutEngine(policy, RolloutConfig(group_size=2, max_new_tokens=4, temperature=0.0), num_workers=1,
                              config=RayConfig(gpus_per_worker=1)) as engine:
            assert engine.broadcast("device") == ["cuda"]
            assert len(engine.rollout(torch.tensor(tok.encode("hi")))) == 2
    finally:
        shutdown_ray()
