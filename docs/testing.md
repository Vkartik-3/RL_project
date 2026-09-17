# Testing

## What

A CPU-first test suite organised by scope, with explicit hardware markers.

## Why

The framework must be verifiable without GPUs; hardware-only paths must skip with a reason instead of failing.

## How

| Directory | Scope |
|---|---|
| `tests/unit` | configs, data, collators, models and attention variants, adapters and quantization, training math, rollouts/verifiers/rewards, checkpoints, distributed planning, evaluation, deployment, inference and serving, observability, CLI, synthesis domain, benchmark-semantics regression tests |
| `tests/integration` | every training stage through the `Trainer`, checkpoint resume equivalence, two-process gloo data parallelism, HuggingFace backend with a tiny local model (SFT/DPO/PPO/GRPO/DAPO/agent GRPO, adapter checkpoint round trip, encoder reward models; skipped without the extra), CLI pipeline (prepare → pretrain → SFT → generate → evaluate → export → registry, HTTP server, AI-feedback and synthesis commands, STaR/RLAIF/hill-climb loops), gloo-backed tensor/pipeline parallel execution |
| `tests/failure` | every failure category with its typed error |
| `tests/smoke` | every module imports; public API; optional-extra error |
| `tests/hardware` | CUDA bf16/fp16, CUDA inference, FSDP and 2-rank tensor parallel under torchrun, DeepSpeed |

Markers (`pyproject.toml`): `cuda`, `multi_gpu`, `distributed`, `slow`, `optional_dependency`. `tests/conftest.py` skips `cuda` tests without CUDA and `multi_gpu` tests with fewer than two devices, printing the reason.

### Benchmark-semantics tests

`tests/unit/test_benchmark_semantics.py` independently re-implements the formulas behind recorded measurements (pretraining cross-entropy, cosine schedule, synthesis rule reward, length-normalised DPO, sequence-mean PPO ratio, tagged-answer reward, process reward, NF4 table) and reproduces the recorded held-out record statistics and leave-one-molecule-out values from the shipped data.

## Configuration

```bash
pip install -e ".[dev]"
python -m pytest tests                    # everything; hardware tests skip on CPU
python -m pytest tests -m "not cuda and not multi_gpu"
python -m pytest tests -rs                # show skip reasons
torchrun --nproc_per_node=2 -m pytest tests/hardware -m multi_gpu
scripts/local_validation.sh               # tests + end-to-end CLI pipeline
```

## Failure modes

A test that needs a missing optional extra either skips (`importorskip`) or asserts the `OptionalDependencyError`.

## Local validation

Current results on CPU (Apple Silicon, Python 3.12, PyTorch 2.14):

| Install | Result |
|---|---|
| `pip install -e ".[dev]"` | 185 passed, 7 skipped (HuggingFace extra, 5 CUDA/multi-GPU, DeepSpeed) in ~9 s |
| `pip install -e ".[dev,huggingface]"` | 192 passed, 6 skipped (5 CUDA/multi-GPU, DeepSpeed) in ~10 s |

## Hardware requirements

CPU for the default suite.

## Limitations

Multi-GPU behaviour is only exercised when `tests/hardware` is run on GPUs.
