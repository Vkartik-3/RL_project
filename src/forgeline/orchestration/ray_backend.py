"""Ray worker pools: rollouts, rewards/verifiers, tools and sharded evaluation.

Division of labour:

* the **learner** (``Trainer`` + a ``torch.distributed`` strategy) owns parameters, gradients and optimizer state;
* **rollout workers** hold a copy of the policy weights and sample completions; the learner pushes new weights
  through the object store whenever its parameters change (detected from tensor version counters);
* **reward workers** build a reward provider from its manifest mapping and score trajectories in shards;
* **tool workers** run sandboxed Python snippets;
* **evaluation workers** load a checkpoint and run disjoint shards of a benchmark; results are merged exactly.

Recovery: every call is tracked with the generation of the worker it was sent to. If the actor dies or the call
times out, the worker is replaced (once per generation), its state is restored (initial arguments plus the
latest weights for rollout workers) and the call is resubmitted, up to ``max_call_retries`` times. Errors raised
by user code inside a worker are not retried; they propagate to the caller.

Requires ``pip install -e '.[ray]'``.
"""

from __future__ import annotations

import math
import os
import weakref
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from forgeline.core.errors import OptionalDependencyError, OrchestrationError
from forgeline.core.protocols import EvaluationResult, RewardResult, Trajectory
from forgeline.observability.logging import get_logger
from forgeline.orchestration.config import RayConfig

try:
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise OptionalDependencyError("Ray orchestration requires Ray: pip install -e '.[ray]'") from exc

log = get_logger("forgeline.orchestration")

_RECOVERABLE: Tuple[type, ...] = tuple(
    t for t in (getattr(ray.exceptions, name, None) for name in
                ("RayActorError", "ActorUnavailableError", "WorkerCrashedError", "GetTimeoutError", "ObjectLostError"))
    if isinstance(t, type)
)

_POOLS: "weakref.WeakSet[_WorkerPool]" = weakref.WeakSet()


# ─────────────────────────────────────────────────────────────────────────────
#  Runtime
# ─────────────────────────────────────────────────────────────────────────────

def init_ray(config: Optional[RayConfig] = None) -> Dict[str, Any]:
    """Start (or join) Ray once per process and return the cluster resources."""
    cfg = config or RayConfig()
    if not ray.is_initialized():
        kwargs: Dict[str, Any] = dict(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False,
                                      namespace=cfg.namespace)
        if cfg.address is not None:
            kwargs["address"] = cfg.address
        else:
            if cfg.num_cpus is not None:
                kwargs["num_cpus"] = cfg.num_cpus
            if cfg.num_gpus is not None:
                kwargs["num_gpus"] = cfg.num_gpus
        ray.init(**kwargs)
        log.info("ray_initialized", address=cfg.address or "local", resources=ray.cluster_resources())
    return dict(ray.cluster_resources())


def shutdown_pools() -> None:
    for pool in list(_POOLS):
        pool.shutdown()


def shutdown_ray() -> None:
    shutdown_pools()
    if ray.is_initialized():
        ray.shutdown()


def _chunks(items: Sequence[Any], n: int) -> List[List[Any]]:
    """Split into at most ``n`` contiguous, nearly equal, non-empty chunks (order preserved)."""
    n = max(1, min(n, len(items)))
    size = math.ceil(len(items) / n) if items else 0
    return [list(items[i:i + size]) for i in range(0, len(items), size)] if items else []


# ─────────────────────────────────────────────────────────────────────────────
#  Actors (plain classes; wrapped with ray.remote by the pool)
# ─────────────────────────────────────────────────────────────────────────────

class _Worker:
    def __init__(self, torch_threads: int = 1):
        torch.set_num_threads(max(1, torch_threads))

    def pid(self) -> int:
        return os.getpid()

    def node(self) -> str:
        return ray.get_runtime_context().get_node_id()

    def ping(self) -> bool:
        return True

    def device(self) -> str:
        """``cuda`` when Ray assigned this actor a GPU and CUDA is usable, else ``cpu``."""
        return "cuda" if torch.cuda.is_available() and ray.get_gpu_ids() else "cpu"


class RewardWorker(_Worker):
    def __init__(self, reward: Any, torch_threads: int = 1):
        super().__init__(torch_threads)
        if isinstance(reward, Mapping):
            from forgeline.rollouts.rewards import build_reward_provider

            reward = build_reward_provider(dict(reward))
        self.provider = reward

    def score(self, trajectories: List[Trajectory]) -> List[RewardResult]:
        return [self.provider.score(t) for t in trajectories]


class ToolWorker(_Worker):
    def execute(self, codes: List[str], timeout: float) -> List[Dict[str, Any]]:
        from forgeline.rollouts.tools.python_executor import execute_python

        out = []
        for code in codes:
            r = execute_python(code, timeout=timeout)
            out.append({"stdout": r.stdout, "stderr": r.stderr, "timed_out": r.timed_out, "output": r.output})
        return out


class RolloutWorker(_Worker):
    def __init__(self, model: torch.nn.Module, tokenizer: Any, pad_token_id: int, rollout_config: Any, seed: int,
                 torch_threads: int = 1):
        super().__init__(torch_threads)
        from forgeline.models.policy import NativePolicy
        from forgeline.rollouts.engine import RolloutEngine

        torch.manual_seed(seed)
        self.policy = NativePolicy(model.to(self.device()).eval(), tokenizer, pad_token_id=pad_token_id)
        self.engine = RolloutEngine(self.policy, rollout_config)
        self.version = -1

    def load_weights(self, state: Dict[str, torch.Tensor], version: int) -> int:
        self.policy.model.load_state_dict(state, strict=True)
        self.version = version
        return version

    def rollout(self, prompt_ids: torch.Tensor, task: Optional[Dict[str, Any]], prompt_text: str,
                group_size: int) -> List[Trajectory]:
        return self.engine.rollout(prompt_ids, task, prompt_text, group_size=group_size)

    def get_version(self) -> int:
        return self.version


class EvaluationWorker(_Worker):
    def __init__(self, checkpoint: str, data_dir: Optional[str], torch_threads: int = 1):
        super().__init__(torch_threads)
        from forgeline.models.loading import load_policy_from_checkpoint

        self.policy = load_policy_from_checkpoint(checkpoint, self.device(), data_dir=data_dir)

    def run(self, benchmark: str, benchmark_kwargs: Dict[str, Any], indices: List[int], max_new_tokens: int) -> EvaluationResult:
        from forgeline.evaluation.suites import BenchmarkSuite, build_benchmark

        bench = build_benchmark(benchmark, **benchmark_kwargs)
        return BenchmarkSuite(bench, max_new_tokens=max_new_tokens).run(self.policy, task_indices=indices)


# ─────────────────────────────────────────────────────────────────────────────
#  Pool base: placement, calls with recovery
# ─────────────────────────────────────────────────────────────────────────────

class _WorkerPool:
    actor_class: type = _Worker
    kind = "worker"
    uses_gpu = False  # reward and tool workers never request GPUs

    def __init__(self, num_workers: int, config: Optional[RayConfig] = None):
        if num_workers < 1:
            raise OrchestrationError(f"{self.kind} pool needs at least one worker")
        self.ray_config = config or RayConfig()
        self.ray_config.validate()
        init_ray(self.ray_config)
        self.num_workers = num_workers
        self.restarts = 0
        self._remote = ray.remote(self.actor_class)
        self._pg = None
        if self.ray_config.placement_strategy:
            bundle: Dict[str, float] = {"CPU": self.ray_config.cpus_per_worker}
            if self.uses_gpu and self.ray_config.gpus_per_worker:
                bundle["GPU"] = self.ray_config.gpus_per_worker
            self._pg = placement_group([dict(bundle) for _ in range(num_workers)], strategy=self.ray_config.placement_strategy)
            try:
                ray.get(self._pg.ready(), timeout=self.ray_config.placement_timeout_s)
            except Exception as exc:  # noqa: BLE001
                remove_placement_group(self._pg)
                raise OrchestrationError(f"placement group {bundle}×{num_workers} ({self.ray_config.placement_strategy}) "
                                         f"could not be scheduled: {type(exc).__name__}",
                                         hint="Reduce workers/resources or add nodes; see ray.cluster_resources().") from None
        self.workers: List[Any] = []
        self.generations: List[int] = []
        _POOLS.add(self)

    # construction ----------------------------------------------------------
    def _init_args(self, index: int) -> Tuple[tuple, dict]:
        return (), {"torch_threads": self.ray_config.worker_torch_threads}

    def _restore(self, index: int) -> None:
        """Re-send state that the constructor does not carry (e.g. latest weights)."""

    def _options(self, index: int) -> Dict[str, Any]:
        opts: Dict[str, Any] = dict(num_cpus=self.ray_config.cpus_per_worker, num_gpus=self.ray_config.gpus_per_worker if self.uses_gpu else 0,
                                    max_restarts=self.ray_config.max_restarts)
        if self.ray_config.memory_per_worker_mb:
            opts["memory"] = int(self.ray_config.memory_per_worker_mb) * 1024 * 1024
        if self._pg is not None:
            opts["scheduling_strategy"] = PlacementGroupSchedulingStrategy(placement_group=self._pg,
                                                                           placement_group_bundle_index=index)
        return opts

    def _spawn(self, index: int):
        args, kwargs = self._init_args(index)
        return self._remote.options(**self._options(index)).remote(*args, **kwargs)

    def _start(self) -> None:
        self.workers = [self._spawn(i) for i in range(self.num_workers)]
        self.generations = [0] * self.num_workers
        ray.get([w.ping.remote() for w in self.workers])

    def _replace(self, index: int, generation: int, reason: BaseException) -> None:
        if self.generations[index] != generation:
            return  # already replaced after an earlier failure of the same actor
        try:
            ray.kill(self.workers[index], no_restart=True)
        except Exception:  # noqa: BLE001
            pass
        self.workers[index] = self._spawn(index)
        self.generations[index] += 1
        self.restarts += 1
        log.warning("worker_replaced", pool=self.kind, index=index, generation=self.generations[index],
                    reason=type(reason).__name__)
        self._restore(index)

    # calls -------------------------------------------------------------------
    def call(self, calls: Sequence[Tuple[int, str, tuple]]) -> List[Any]:
        """Run ``(worker_index, method, args)`` calls concurrently; results in call order."""
        pending = {j: (w, self.generations[w], getattr(self.workers[w], m).remote(*a)) for j, (w, m, a) in enumerate(calls)}
        attempts = [0] * len(calls)
        results: List[Any] = [None] * len(calls)
        while pending:
            refs = {ref: j for j, (_, _, ref) in pending.items()}
            ready, _ = ray.wait(list(refs), num_returns=1, timeout=self.ray_config.task_timeout_s)
            if not ready:  # timeout: treat every still-pending call as failed
                failed = list(pending)
                errors = {j: ray.exceptions.GetTimeoutError("task timeout") for j in failed}
            else:
                j = refs[ready[0]]
                try:
                    results[j] = ray.get(ready[0])
                    pending.pop(j)
                    continue
                except _RECOVERABLE as exc:
                    failed, errors = [j], {j: exc}
            for j in failed:
                w, gen, _ = pending[j]
                attempts[j] += 1
                if attempts[j] > self.ray_config.max_call_retries:
                    raise OrchestrationError(f"{self.kind} call {calls[j][1]!r} failed after {attempts[j]} attempts: "
                                             f"{type(errors[j]).__name__}: {errors[j]}")
                self._replace(w, gen, errors[j])
                _, m, a = calls[j]
                pending[j] = (w, self.generations[w], getattr(self.workers[w], m).remote(*a))
        return results

    def broadcast(self, method: str, *args: Any) -> List[Any]:
        return self.call([(i, method, args) for i in range(self.num_workers)])

    def pids(self) -> List[int]:
        return self.broadcast("pid")

    def shutdown(self) -> None:
        for w in self.workers:
            try:
                ray.kill(w, no_restart=True)
            except Exception:  # noqa: BLE001
                pass
        self.workers = []
        if self._pg is not None:
            try:
                remove_placement_group(self._pg)
            except Exception:  # noqa: BLE001
                pass
            self._pg = None
        _POOLS.discard(self)

    def placement(self) -> Optional[Dict[str, Any]]:
        if self._pg is None:
            return None
        table = ray.util.placement_group_table(self._pg)
        return {"strategy": table.get("strategy"), "state": table.get("state"), "bundles": table.get("bundles")}

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
#  Pools
# ─────────────────────────────────────────────────────────────────────────────

class RayRewardPool(_WorkerPool):
    """A :class:`RewardProvider` whose scoring runs in reward/verifier worker actors.

    ``reward`` is a manifest reward mapping (built inside each worker) or a picklable provider object.
    ``score_many`` shards trajectories across workers; ``score_trajectories`` uses it automatically.
    """

    actor_class = RewardWorker
    kind = "reward"

    def __init__(self, reward: Any, num_workers: int, config: Optional[RayConfig] = None):
        super().__init__(num_workers, config)
        self._reward_ref = ray.put(reward)
        self.name = f"ray:{reward.get('type', 'reward') if isinstance(reward, Mapping) else getattr(reward, 'name', 'reward')}"
        self._start()

    def _init_args(self, index):
        return (self._reward_ref,), {"torch_threads": self.ray_config.worker_torch_threads}

    def score_many(self, trajectories: Sequence[Trajectory]) -> List[RewardResult]:
        chunks = _chunks(list(trajectories), self.num_workers)
        parts = self.call([(i, "score", (chunk,)) for i, chunk in enumerate(chunks)])
        return [r for part in parts for r in part]

    def score(self, trajectory: Trajectory) -> RewardResult:
        return self.score_many([trajectory])[0]


class RayToolPool(_WorkerPool):
    """Sandboxed Python execution fanned out over worker actors."""

    actor_class = ToolWorker
    kind = "tool"

    def __init__(self, num_workers: int, config: Optional[RayConfig] = None, timeout: float = 5.0):
        super().__init__(num_workers, config)
        self.timeout = timeout
        self._start()

    def execute_many(self, codes: Sequence[str]) -> List[Dict[str, Any]]:
        chunks = _chunks(list(codes), self.num_workers)
        parts = self.call([(i, "execute", (chunk, self.timeout)) for i, chunk in enumerate(chunks)])
        return [r for part in parts for r in part]


def _weights_version(model: torch.nn.Module) -> int:
    """Sum of tensor version counters: changes whenever the optimizer (or load_state_dict) updates a parameter."""
    return int(sum(p._version for p in model.parameters()))


class RayRolloutEngine(_WorkerPool):
    """Drop-in replacement for :class:`RolloutEngine` that samples in rollout worker actors.

    Weights are pushed from the learner's policy before a rollout whenever its parameters changed since the last
    push, so every trajectory is sampled from the current policy (on-policy). Old/reference log-probs used by
    PPO/GRPO are recomputed on the learner, as with the in-process engine. Native policies only.
    """

    actor_class = RolloutWorker
    kind = "rollout"
    uses_gpu = True

    def __init__(self, policy: Any, rollout_config: Any, num_workers: int, config: Optional[RayConfig] = None,
                 seed: int = 0):
        if getattr(policy, "backend", "native") != "native":
            raise OrchestrationError("Ray rollout workers support native policies",
                                     hint="Use in-process rollouts for the huggingface backend.")
        super().__init__(num_workers, config)
        self.policy = policy
        self.cfg = rollout_config  # same attribute name as RolloutEngine.cfg
        self.seed = seed
        self._model_ref = ray.put(policy.model.cpu() if policy.device.type == "cpu" else _cpu_copy(policy.model))
        self._tokenizer_ref = ray.put(policy.tokenizer)
        self._state_ref = None
        self._version: Optional[int] = None
        self.weight_pushes = 0
        self._start()
        self.sync_weights(force=True)

    def _init_args(self, index):
        return ((self._model_ref, self._tokenizer_ref, self.policy.pad_token_id, self.cfg, self.seed + 1000 * index),
                {"torch_threads": self.ray_config.worker_torch_threads})

    def _restore(self, index: int) -> None:
        if self._state_ref is not None:
            getattr(self.workers[index], "load_weights").remote(self._state_ref, self._version)

    def sync_weights(self, force: bool = False) -> bool:
        version = _weights_version(self.policy.model)
        if not force and version == self._version:
            return False
        state = {k: v.detach().cpu().clone() for k, v in self.policy.model.state_dict().items()}
        self._state_ref = ray.put(state)
        self._version = version
        self.broadcast("load_weights", self._state_ref, version)
        self.weight_pushes += 1
        return True

    def worker_versions(self) -> List[int]:
        return self.broadcast("get_version")

    def rollout(self, prompt_ids: torch.Tensor, task: Optional[Dict[str, Any]] = None, prompt_text: str = "",
                group_size: Optional[int] = None) -> List[Trajectory]:
        self.sync_weights()
        G = group_size or self.cfg.group_size
        n = min(self.num_workers, G)
        counts = [G // n + (1 if i < G % n else 0) for i in range(n)]
        parts = self.call([(i, "rollout", (prompt_ids.cpu(), task, prompt_text, c)) for i, c in enumerate(counts)])
        return [t for part in parts for t in part]

    def rollout_many(self, prompts: Sequence[torch.Tensor], tasks: Optional[Sequence[Dict[str, Any]]] = None,
                     prompt_texts: Optional[Sequence[str]] = None) -> List[List[Trajectory]]:
        """One call per prompt, prompts distributed round-robin over workers."""
        self.sync_weights()
        G = self.cfg.group_size
        calls = [(i % self.num_workers, "rollout", (p.cpu(), tasks[i] if tasks else None,
                                                    prompt_texts[i] if prompt_texts else "", G)) for i, p in enumerate(prompts)]
        return self.call(calls)


def _cpu_copy(model: torch.nn.Module) -> torch.nn.Module:
    import copy

    return copy.deepcopy(model).cpu()


class RayEvaluator(_WorkerPool):
    """Shards a benchmark's tasks across evaluation workers and merges the per-item results exactly."""

    actor_class = EvaluationWorker
    kind = "evaluation"
    uses_gpu = True

    def __init__(self, checkpoint: str, num_workers: int, config: Optional[RayConfig] = None,
                 data_dir: Optional[str] = None):
        super().__init__(num_workers, config)
        self.checkpoint = str(checkpoint)
        self.data_dir = data_dir
        self._start()

    def _init_args(self, index):
        return (self.checkpoint, self.data_dir), {"torch_threads": self.ray_config.worker_torch_threads}

    def run(self, benchmark: str, max_new_tokens: int = 64, **benchmark_kwargs: Any) -> EvaluationResult:
        from forgeline.evaluation.suites import build_benchmark

        n_tasks = len(build_benchmark(benchmark, **benchmark_kwargs).tasks())
        shards = _chunks(list(range(n_tasks)), self.num_workers)
        parts: List[EvaluationResult] = self.call(
            [(i, "run", (benchmark, benchmark_kwargs, shard, max_new_tokens)) for i, shard in enumerate(shards)])
        per_item = sorted((it for p in parts for it in p.per_item), key=lambda it: it["index"])
        correct = sum(int(it["correct"]) for it in per_item)
        n = max(len(per_item), 1)
        details = dict(parts[0].details) if parts else {}
        details.update({"workers": len(shards), "shard_sizes": [len(s) for s in shards]})
        return EvaluationResult(parts[0].suite if parts else f"benchmark/{benchmark}",
                                {"accuracy": correct / n, "correct": float(correct)}, n_samples=len(per_item),
                                details=details, per_item=per_item)


# ─────────────────────────────────────────────────────────────────────────────
#  Manifest integration
# ─────────────────────────────────────────────────────────────────────────────

def attach_to_algorithm(algorithm: Any, orchestration: Mapping[str, Any], reward_config: Optional[Mapping[str, Any]],
                        seed: int = 0) -> Dict[str, Any]:
    """Swap an algorithm's in-process rollout engine / reward provider for Ray pools per the manifest.

    Supports algorithms exposing ``rollouts`` (a :class:`RolloutEngine`) and ``reward`` (PPO, GRPO, DAPO). Agent-mode
    GRPO keeps multi-turn rollouts in process (tool calls already run in subprocess sandboxes).
    """
    cfg = RayConfig.from_dict({k: v for k, v in orchestration.items() if k != "type"})
    attached: Dict[str, Any] = {"backend": "ray"}
    if cfg.reward_workers and hasattr(algorithm, "reward"):
        algorithm.reward = RayRewardPool(dict(reward_config) if reward_config else algorithm.reward, cfg.reward_workers, cfg)
        attached["reward_workers"] = cfg.reward_workers
    rollouts = getattr(algorithm, "rollouts", None)
    if cfg.rollout_workers and rollouts is not None and hasattr(algorithm, "policy"):
        algorithm.rollouts = RayRolloutEngine(algorithm.policy, rollouts.cfg, cfg.rollout_workers, cfg, seed=seed)
        attached["rollout_workers"] = cfg.rollout_workers
    elif cfg.rollout_workers:
        attached["rollout_workers_skipped"] = f"{type(algorithm).__name__} has no single-turn rollout engine"
    log.info("orchestration_attached", **{k: str(v) for k, v in attached.items()})
    return attached


__all__ = ["EvaluationWorker", "RayConfig", "RayEvaluator", "RayRewardPool", "RayRolloutEngine", "RayToolPool",
           "RewardWorker", "RolloutWorker", "ToolWorker", "attach_to_algorithm", "init_ray", "shutdown_pools", "shutdown_ray"]
