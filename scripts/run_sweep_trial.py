#!/usr/bin/env python3
"""One hyper-parameter sweep trial: apply overrides to a manifest, train, report the final eval loss.

Used by configs/sweeps/dpo_wandb.yaml (wandb passes --beta=... --learning_rate=...).
"""

import argparse
import json

from forgeline.core.config import ExperimentManifest
from forgeline.core.lifecycle import RunContext
from forgeline.training.common.trainer import Trainer
from forgeline.training.factory import build_algorithm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--beta", type=float)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--max_steps", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--metrics", default="wandb")
    a = p.parse_args()
    m = ExperimentManifest.load(a.manifest)
    if a.beta is not None:
        m.algorithm_params["beta"] = a.beta
    if a.learning_rate is not None:
        m.trainer.optimizer.learning_rate = a.learning_rate
    if a.max_steps is not None:
        m.trainer.max_steps = a.max_steps
    if a.batch_size is not None:
        m.trainer.batch_size = a.batch_size
    m.trainer.eval_every = max(1, m.trainer.max_steps // 4)
    ctx = RunContext.create(m, metrics_backend=a.metrics)
    summary = Trainer(ctx, build_algorithm(m, ctx)).fit()
    ctx.close()
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
