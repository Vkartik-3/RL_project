"""The shared training lifecycle.

``Trainer`` owns everything algorithms have in common — optimizer, learning
rate schedule, mixed precision, gradient accumulation, clipping, checkpoint
save/resume, evaluation cadence and metrics — while a
:class:`PostTrainingAlgorithm` owns data collection and the loss.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
import torch.nn as nn

from forgeline.checkpoints.manager import CheckpointManager, CheckpointState
from forgeline.core.config import TrainerConfig
from forgeline.core.errors import CheckpointNotFoundError, ConfigError
from forgeline.core.lifecycle import RunContext
from forgeline.core.runtime import rng_state, set_rng_state
from forgeline.observability.events import Event
from forgeline.observability.metrics import device_memory_metrics
from forgeline.training.common.optim import build_optimizer
from forgeline.training.common.precision import Precision
from forgeline.training.common.schedule import learning_rate_at


@dataclass
class StepResult:
    loss: float
    metrics: Dict[str, float] = field(default_factory=dict)


class PostTrainingAlgorithm(ABC):
    """Contract every training stage implements."""

    name: str = "algorithm"

    @abstractmethod
    def parameters(self) -> List[nn.Parameter]:
        """Parameters the optimizer should update."""

    @abstractmethod
    def collect(self, step: int) -> Any:
        """Produce the data for one optimizer step (may run rollouts)."""

    @abstractmethod
    def loss(self, batch: Any, micro_step: int, n_micro: int) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute the loss for one micro-batch (``batch`` may be sliced by ``micro_step``)."""

    def modules(self) -> Iterable[nn.Module]:
        """Modules to toggle train()/eval() (default: none)."""
        return ()

    def evaluate(self, step: int) -> Dict[str, float]:
        return {}

    def state(self) -> Dict[str, Any]:
        """Model/adapter state to checkpoint."""
        return {}

    def load_state(self, state: Mapping[str, Any]) -> None:
        pass

    def model_spec(self) -> Dict[str, Any]:
        return {}

    def tokenizer_meta(self) -> Optional[Dict[str, Any]]:
        return None

    def on_step_end(self, step: int, result: StepResult) -> None:
        pass


class Trainer:
    def __init__(self, ctx: RunContext, algorithm: PostTrainingAlgorithm, config: Optional[TrainerConfig] = None,
                 checkpoint_manager: Optional[CheckpointManager] = None, strategy: Any = None):
        self.ctx = ctx
        self.alg = algorithm
        self.cfg = config or ctx.manifest.trainer
        self.cfg.validate()
        self.log = ctx.logger
        self.strategy = strategy
        self.params = [p for p in algorithm.parameters() if p.requires_grad]
        if not self.params:
            raise ConfigError("algorithm exposes no trainable parameters")
        self._sync_grads = bool(strategy is not None and getattr(strategy, "name", "") == "ddp" and strategy.world_size() > 1)
        self._sharded = bool(strategy is not None and getattr(strategy, "name", "") == "fsdp" and strategy.world_size() > 1)
        if self._sync_grads:
            self._broadcast_parameters()
        if self.cfg.compile_model:
            self._compile_modules()
        self.optimizer = build_optimizer(self.params, self.cfg.optimizer, ctx.device)
        self.precision = Precision(self.cfg.precision, ctx.device, ctx.dtype)
        ckpt_dir = ctx.output_dir / self.cfg.checkpoint.directory
        self.checkpoints = checkpoint_manager or CheckpointManager(
            ckpt_dir, keep_last=self.cfg.checkpoint.keep_last, async_save=self.cfg.checkpoint.async_save)
        self.step = 0
        self.best_metric: float = math.inf
        self.history: List[StepResult] = []

    def _compile_modules(self) -> None:
        """Compile each trained module's ``forward`` in place with ``torch.compile``.

        In-place compilation keeps parameter names, so checkpoints stay loadable without compilation. Cached
        generation paths (``prefill`` / ``step``) run eagerly.
        """
        if not hasattr(nn.Module, "compile"):
            raise ConfigError("trainer.compile_model needs torch>=2.2 (nn.Module.compile)")
        compiled = []
        for m in self.alg.modules():
            if m is not None:
                m.compile(backend=self.cfg.compile_backend)
                compiled.append(type(m).__name__)
        self.log.info("compiled", backend=self.cfg.compile_backend, modules=",".join(compiled))

    # ── data-parallel helpers ────────────────────────────────────────────
    def _broadcast_parameters(self) -> None:
        """Make every rank start from rank 0's parameters."""
        import torch.distributed as dist

        with torch.no_grad():
            for p in self.params:
                dist.broadcast(p.data, src=0)

    def _all_reduce_gradients(self) -> None:
        """Average gradients across ranks (data parallelism for any algorithm, including RL loops)."""
        import torch.distributed as dist

        world = self.strategy.world_size()
        for p in self.params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world)

    # ── schedule / lr ────────────────────────────────────────────────────
    def current_lr(self) -> float:
        return learning_rate_at(self.step, self.cfg.optimizer.learning_rate, self.cfg.schedule)

    def _set_lr(self, lr: float) -> None:
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def _set_train_mode(self, mode: bool) -> None:
        for m in self.alg.modules():
            m.train(mode)

    # ── one optimizer step ───────────────────────────────────────────────
    def train_step(self) -> StepResult:
        lr = self.current_lr()
        self._set_lr(lr)
        self._set_train_mode(True)
        batch = self.alg.collect(self.step)
        self.optimizer.zero_grad(set_to_none=True)
        n_micro = self.cfg.gradient_accumulation_steps
        total_loss = 0.0
        agg: Dict[str, float] = {}
        for micro in range(n_micro):
            with self.precision.autocast():
                loss, metrics = self.alg.loss(batch, micro, n_micro)
            if not torch.isfinite(loss):
                raise ConfigError(f"non-finite loss at step {self.step}: {loss.item()}",
                                  hint="Lower the learning rate or check the reward / data pipeline.")
            self.precision.backward(loss / n_micro)
            total_loss += float(loss.detach()) / n_micro
            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + float(v) / n_micro
        if self._sync_grads:
            self._all_reduce_gradients()
        grad_norm = 0.0
        if self.cfg.optimizer.grad_clip > 0:
            self.precision.unscale(self.optimizer)
            if self._sharded and hasattr(self.strategy, "clip_grad_norm"):
                grad_norm = float(self.strategy.clip_grad_norm(self._sharded_module(), self.cfg.optimizer.grad_clip))
            else:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.params, self.cfg.optimizer.grad_clip))
        self.precision.step(self.optimizer)
        agg.update({"lr": lr, "grad_norm": grad_norm})
        result = StepResult(loss=total_loss, metrics=agg)
        self.alg.on_step_end(self.step, result)
        self.step += 1
        self.history.append(result)
        return result

    # ── full loop ────────────────────────────────────────────────────────
    def fit(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        max_steps = self.cfg.max_steps if max_steps is None else max_steps
        self.ctx.metrics.event(Event.RUN_STARTED.value, {"algorithm": self.alg.name, "max_steps": max_steps})
        t0 = time.time()
        last_eval: Dict[str, float] = {}
        try:
            while self.step < max_steps:
                if self.cfg.eval_every and self.step % self.cfg.eval_every == 0 and (self._is_main() or self._sharded):
                    last_eval = self._run_eval()
                if self.cfg.checkpoint.save_every and self.step > 0 and self.step % self.cfg.checkpoint.save_every == 0 and (self._is_main() or self._sharded):
                    self.save_checkpoint()
                result = self.train_step()
                if self.cfg.log_every and self.step % self.cfg.log_every == 0 and self._is_main():
                    payload = {"train/loss": result.loss, **{f"train/{k}": v for k, v in result.metrics.items()}}
                    payload.update(device_memory_metrics(self.ctx.device))
                    self.ctx.metrics.log(payload, step=self.step)
                    self.log.info("step", step=self.step, loss=round(result.loss, 5),
                                  **{k: round(v, 5) for k, v in result.metrics.items() if isinstance(v, float)})
            if self._is_main() or self._sharded:
                last_eval = self._run_eval() if self.cfg.eval_every else last_eval
                self.save_checkpoint(name="final")
                if self._is_main():
                    self.checkpoints.wait()
        except BaseException as exc:
            self.ctx.metrics.event(Event.RUN_FAILED.value, {"error": repr(exc), "step": self.step})
            raise
        elapsed = time.time() - t0
        summary = {"steps": self.step, "elapsed_s": elapsed, "final_loss": self.history[-1].loss if self.history else None,
                   "best_metric": None if math.isinf(self.best_metric) else self.best_metric, "eval": last_eval}
        self.ctx.metrics.event(Event.RUN_FINISHED.value, summary)
        return summary

    def _is_main(self) -> bool:
        return self.strategy is None or self.strategy.is_main_process()

    def _run_eval(self) -> Dict[str, float]:
        self._set_train_mode(False)
        self.ctx.metrics.event(Event.EVAL_STARTED.value, {"step": self.step})
        with torch.no_grad():
            metrics = self.alg.evaluate(self.step)
        self._set_train_mode(True)
        if metrics and self._is_main():
            self.ctx.metrics.log({f"eval/{k}": v for k, v in metrics.items()}, step=self.step)
            self.log.info("eval", step=self.step, **{k: round(v, 5) for k, v in metrics.items()})
            primary = metrics.get("loss", metrics.get("primary"))
            if primary is not None and primary < self.best_metric and not self._sharded:
                self.best_metric = primary
                self.save_checkpoint(name="best_candidate", is_best=True)
        self.ctx.metrics.event(Event.EVAL_FINISHED.value, {"step": self.step, **metrics})
        return metrics

    # ── checkpoints ──────────────────────────────────────────────────────
    def _sharded_module(self) -> nn.Module:
        return getattr(self.alg, "model", None) or getattr(self.alg, "student")

    def _model_state(self) -> Dict[str, Any]:
        if not self._sharded:
            return self.alg.state()
        from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

        with FSDP.state_dict_type(self._sharded_module(), StateDictType.FULL_STATE_DICT,
                                  FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            return self.alg.state()

    def save_checkpoint(self, name: Optional[str] = None, is_best: bool = False):
        model_state = self._model_state()  # collective under FSDP: every rank participates
        if not self._is_main():
            return None
        state = CheckpointState(
            model_state=model_state, step=self.step, algorithm=self.alg.name,
            model_spec=self.alg.model_spec(), optimizer_state=None if self._sharded else self.optimizer.state_dict(),
            scaler_state=self.precision.state_dict(), rng_state=rng_state(),
            trainer_state={"best_metric": self.best_metric, "history_len": len(self.history)},
            tokenizer_meta=self.alg.tokenizer_meta(), experiment=self.ctx.manifest.to_dict(),
        )
        path = self.checkpoints.save(state, name=name, is_best=is_best)
        self.ctx.metrics.event(Event.CHECKPOINT_SAVED.value, {"path": str(path), "step": self.step})
        if name == "final" and self.cfg.adapter.merge_on_save:
            self._save_merged(state)
        return path

    def _save_merged(self, state: CheckpointState):
        """Write ``final_merged``: adapters folded into the base weights, loadable as a plain model checkpoint."""
        import copy

        from forgeline.models.adapters.lora import merge_lora

        policy = getattr(self.alg, "policy", None)
        if policy is None or not getattr(policy, "has_adapter", False) or not hasattr(policy.model, "spec"):
            self.log.warning("merge_on_save_skipped", reason="algorithm has no native policy with adapters")
            return None
        if self.cfg.adapter.method != "lora":
            self.log.warning("merge_on_save_skipped", reason=f"{self.cfg.adapter.method} base weights are quantized; export merges on dequantized weights")
            return None
        merged = merge_lora(copy.deepcopy(policy.model)).state_dict()
        merged_state = CheckpointState(model_state={"model": merged}, step=state.step, algorithm=state.algorithm,
                                       model_spec=state.model_spec, tokenizer_meta=state.tokenizer_meta,
                                       experiment=state.experiment, metadata={"merged_adapter": True})
        path = self.checkpoints.save(merged_state, name="final_merged")
        self.ctx.metrics.event(Event.CHECKPOINT_SAVED.value, {"path": str(path), "step": self.step, "merged_adapter": True})
        return path

    def resume(self, path: Optional[str] = None, strict_spec: bool = True) -> bool:
        """Resume from ``path`` or the latest checkpoint; returns True when something was loaded."""
        if self._sharded:
            raise ConfigError("resume is not supported for fsdp runs in the step trainer",
                              hint="Initialise from the checkpoint with `checkpoint_path` instead.")
        target = path or self.cfg.checkpoint.resume_from or None
        if target is None:
            if not self.cfg.checkpoint.auto_resume:
                return False
            latest = self.checkpoints.find_latest()
            if latest is None:
                return False
            target = str(latest)
        expected = self.alg.model_spec() if strict_spec else None
        try:
            state = CheckpointManager.load(target, map_location="cpu", expected_spec=expected or None,
                                           expected_algorithm=self.alg.name)
        except CheckpointNotFoundError:
            if path or self.cfg.checkpoint.resume_from:
                raise
            return False
        self.alg.load_state(state.model_state)
        if state.optimizer_state:
            self.optimizer.load_state_dict(state.optimizer_state)
        if state.scaler_state:
            self.precision.load_state_dict(state.scaler_state)
        if state.rng_state:
            set_rng_state(state.rng_state)
        self.step = state.step
        self.best_metric = float(state.trainer_state.get("best_metric", math.inf))
        self.ctx.metrics.event(Event.CHECKPOINT_LOADED.value, {"path": target, "step": self.step})
        self.log.info("resumed", path=target, step=self.step)
        return True
