"""Two CPU processes (gloo): gradient averaging keeps replicas identical while ranks sample different data."""

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world: int, port: int, out_dir: str, algorithm: str, q) -> None:
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "RANK": str(rank), "WORLD_SIZE": str(world), "LOCAL_RANK": "0"})
    import numpy as np

    from forgeline.cli.main import _apply_strategy
    from forgeline.core.config import CheckpointConfig, DistributedConfig, ExperimentManifest, OptimizerConfig, TrainerConfig, model_spec_from_preset
    from forgeline.core.lifecycle import RunContext
    from forgeline.data.tokenizers import CharTokenizer, save_tokenizer_metadata
    from forgeline.distributed.strategies import build_strategy
    from forgeline.training.common.trainer import Trainer
    from forgeline.training.factory import build_algorithm

    tok = CharTokenizer.ascii()
    data_dir = os.path.join(out_dir, "data")
    if algorithm == "pretrain":
        ids = np.array(tok.encode("abcdefghij klmnop qrstuv wxyz. " * 60), dtype=np.uint16)
        os.makedirs(data_dir, exist_ok=True)
        ids.tofile(os.path.join(data_dir, f"train.bin"))
        ids.tofile(os.path.join(data_dir, f"val.bin"))
        save_tokenizer_metadata(tok, data_dir)
        data = {"kind": "pretraining", "path": data_dir}
        params = {}
    else:
        data = {"kind": "verifiable", "path": os.path.join(out_dir, "tasks.jsonl"), "tokenizer": "char"}
        with open(data["path"], "w") as f:
            for i in range(6):
                f.write('{"prompt": "What is %d+1?", "answer": "%d"}\n' % (i, i + 1))
        params = {"group_size": 2, "prompts_per_step": 1, "rollout": {"max_new_tokens": 4}}
    manifest = ExperimentManifest.from_dict({
        "name": f"ddp-{rank}", "algorithm": algorithm, "model_preset": "tiny", "model": {"block_size": 32},
        "data": data, "algorithm_params": params, "reward": {"type": "rule", "name": "length", "target_length": 3},
        "distributed": {"strategy": "ddp", "backend": "gloo"}, "output_dir": os.path.join(out_dir, f"rank{rank}"),
        "trainer": {"max_steps": 3, "batch_size": 4, "eval_every": 0, "device": "cpu",
                    "optimizer": {"learning_rate": 1.0e-2}, "checkpoint": {"save_every": 0}},
    })
    strategy = build_strategy(manifest.distributed)
    strategy.setup()
    ctx = RunContext.create(manifest, metrics_backend="none")
    alg = build_algorithm(manifest, ctx)
    _apply_strategy(alg, strategy, manifest)
    trainer = Trainer(ctx, alg, strategy=strategy)
    first_batch = alg.collect(0)
    trainer.fit()
    flat = torch.cat([p.detach().flatten() for p in trainer.params])
    sample = first_batch[0][0].tolist() if algorithm == "pretrain" else None
    q.put((rank, flat.numpy(), sample))
    strategy.teardown()


@pytest.mark.parametrize("algorithm", ["pretrain", "grpo"])
def test_ddp_replicas_stay_identical(tmp_path, algorithm):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, 2, port, str(tmp_path), algorithm, q)) for r in range(2)]
    for p in procs:
        p.start()
    results = sorted(q.get(timeout=240) for _ in procs)
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    (_, w0, s0), (_, w1, s1) = results
    assert (abs(w0 - w1).max()) < 1e-6  # identical replicas after synchronized updates
    if algorithm == "pretrain":
        assert s0 != s1  # different data per rank
