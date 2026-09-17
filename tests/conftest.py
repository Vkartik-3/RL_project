"""Shared fixtures: tiny model spec, tokenizer, corpus, policy, run context."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from forgeline.core.config import CheckpointConfig, ExperimentManifest, TrainerConfig, model_spec_from_preset
from forgeline.core.lifecycle import RunContext
from forgeline.data.tokenizers import CharTokenizer, save_tokenizer_metadata
from forgeline.models.policy import NativePolicy
from forgeline.models.transformer.model import TransformerLM

HAS_CUDA = torch.cuda.is_available()
N_GPUS = torch.cuda.device_count() if HAS_CUDA else 0


def pytest_collection_modifyitems(config, items):
    """Skip hardware-marked tests with an explicit, human-readable reason."""
    for item in items:
        if "cuda" in item.keywords and not HAS_CUDA:
            item.add_marker(pytest.mark.skip(reason="requires a CUDA device (none available)"))
        if "multi_gpu" in item.keywords and N_GPUS < 2:
            item.add_marker(pytest.mark.skip(reason=f"requires >= 2 CUDA devices (found {N_GPUS})"))
        if "distributed" in item.keywords and not HAS_CUDA:
            item.add_marker(pytest.mark.skip(reason="requires a torchrun launch on CUDA devices"))


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)


@pytest.fixture
def tokenizer():
    return CharTokenizer.ascii()


@pytest.fixture
def tiny_spec(tokenizer):
    return model_spec_from_preset("tiny", vocab_size=tokenizer.vocab_size, block_size=64)


@pytest.fixture
def tiny_model(tiny_spec):
    return TransformerLM(tiny_spec)


@pytest.fixture
def policy(tiny_model, tokenizer):
    return NativePolicy(tiny_model, tokenizer)


@pytest.fixture
def corpus_dir(tmp_path, tokenizer):
    text = ("the quick brown fox jumps over the lazy dog. Q: What is 2 + 3? A: 5\n" * 120)
    ids = np.array(tokenizer.encode(text), dtype=np.uint16)
    ids[: int(len(ids) * 0.9)].tofile(tmp_path / "train.bin")
    ids[int(len(ids) * 0.9):].tofile(tmp_path / "val.bin")
    save_tokenizer_metadata(tokenizer, tmp_path)
    return tmp_path


@pytest.fixture
def sft_jsonl(tmp_path):
    p = tmp_path / "sft.jsonl"
    with p.open("w") as f:
        for i in range(40):
            f.write(json.dumps({"prompt": f"Q: What is {i} + 1?\nA:", "response": f" {i + 1}"}) + "\n")
    return p


@pytest.fixture
def preference_jsonl(tmp_path):
    p = tmp_path / "prefs.jsonl"
    with p.open("w") as f:
        for i in range(40):
            f.write(json.dumps({"prompt": f"Q{i}:", "chosen": f" good answer {i}", "rejected": f" bad {i}"}) + "\n")
    return p


@pytest.fixture
def tasks_jsonl(tmp_path):
    p = tmp_path / "tasks.jsonl"
    with p.open("w") as f:
        for i in range(12):
            f.write(json.dumps({"prompt": f"What is {i} + 2?", "answer": str(i + 2)}) + "\n")
    return p


def make_ctx(tmp_path, algorithm: str, spec, *, run_name=None, max_steps=3, batch_size=4, eval_every=2, save_every=2, **trainer_kwargs):
    manifest = ExperimentManifest(
        name=f"test-{algorithm}", algorithm=algorithm, model=spec, output_dir=str(tmp_path / (run_name or algorithm)),
        trainer=TrainerConfig(max_steps=max_steps, batch_size=batch_size, eval_every=eval_every, log_every=1, device="cpu",
                              checkpoint=CheckpointConfig(save_every=save_every, directory="ckpt"), **trainer_kwargs),
    )
    return RunContext.create(manifest, metrics_backend="local")


@pytest.fixture
def ctx_factory(tmp_path, tiny_spec):
    def factory(algorithm: str, **kw):
        return make_ctx(tmp_path, algorithm, tiny_spec, **kw)

    return factory
