# Orchestration (optional Ray backend)

Forgeline separates two kinds of scale-out:

| Concern | Mechanism | Module |
|---|---|---|
| Gradient synchronisation, sharded parameters, tensor/pipeline model parallelism | `torch.distributed` strategies (`single`, `ddp`, `fsdp`, `deepspeed`, `tensor_parallel`, `pipeline_parallel`) | `forgeline.distributed` |
| Work around the learner: sampling completions, scoring them, running tools, evaluating checkpoints | Ray worker pools | `forgeline.orchestration.ray_backend` |

Ray never owns gradients or optimizer state. Install it only when you need it:

```bash
pip install -e ".[ray]"
```

Nothing in the core imports Ray (`tests/unit/test_orchestration_config.py` checks this in a subprocess). Without the extra, importing `forgeline.orchestration.ray_backend` raises `OptionalDependencyError` naming the extra.

## Pools

| Pool | Actor | What runs in the worker | Resources |
|---|---|---|---|
| `RayRolloutEngine` | `RolloutWorker` | a `NativePolicy` copy + `RolloutEngine`; samples `G` completions split across workers | `cpus_per_worker`, `gpus_per_worker` |
| `RayRewardPool` | `RewardWorker` | the manifest reward provider (`build_reward_provider`) or a picklable provider object; scores trajectory shards | `cpus_per_worker` (never GPUs) |
| `RayToolPool` | `ToolWorker` | `execute_python` (subprocess + timeout) for batches of snippets | `cpus_per_worker` (never GPUs) |
| `RayEvaluator` | `EvaluationWorker` | loads a checkpoint, runs a contiguous shard of a benchmark's tasks | `cpus_per_worker`, `gpus_per_worker` |

`RayRolloutEngine` has the same `rollout(prompt_ids, task, prompt_text, group_size)` / `rollout_many` surface and `cfg` attribute as `RolloutEngine`, and `RayRewardPool` implements the `RewardProvider` protocol plus `score_many(trajectories)`; `score_trajectories` uses `score_many` when a provider offers it. PPO, GRPO and DAPO therefore run unchanged.

A GPU-assigned worker (`ray.get_gpu_ids()` non-empty and CUDA available) places its model on `cuda`; otherwise on CPU.

## On-policy weight synchronisation

The learner's policy is registered once in the object store at pool start-up. Before each rollout the engine computes the sum of the parameters' tensor version counters; any optimizer step (or `load_state_dict`) changes it. When it changed, the engine copies the state dict to CPU, `ray.put`s it once and every worker loads it before sampling. `weight_pushes` counts transfers; `worker_versions()` reports what each worker holds.

Old and reference log-probs for PPO/GRPO/DAPO are recomputed by the learner from the returned token ids, exactly as with in-process rollouts, so importance ratios never mix worker-side and learner-side numerics.

## Recovery

Every call is submitted with the generation number of the worker it targets.

* A `RayActorError` / `ActorUnavailableError` / `WorkerCrashedError` / `ObjectLostError`, or a call exceeding `task_timeout_s`, marks the call failed.
* The worker is killed and respawned once per generation (other failed calls to the same dead actor reuse the new worker), `restarts` is incremented, and `_restore` re-sends state the constructor does not carry (latest weights for rollout workers).
* The call is resubmitted; after `max_call_retries` resubmissions an `OrchestrationError` is raised.
* Exceptions raised by user code inside a worker (e.g. a reward provider bug) propagate immediately and are not retried.

`max_restarts` is also passed to Ray so actors restart on their own when no call is in flight.

## Placement

`placement_strategy: PACK | SPREAD | STRICT_PACK | STRICT_SPREAD` creates a placement group with one bundle per worker (`{"CPU": cpus_per_worker, "GPU": gpus_per_worker}` for GPU pools) and schedules worker *i* into bundle *i*. If the group cannot be scheduled within `placement_timeout_s`, it is removed and `OrchestrationError` names the bundle shape.

## Configuration

```yaml
orchestration:
  type: ray                 # or local / omitted
  address: null             # null: private local instance; "auto" / "ray://host:10001": join a cluster
  num_cpus: 4               # local instance only
  rollout_workers: 2
  reward_workers: 2
  cpus_per_worker: 1
  gpus_per_worker: 0        # rollout / evaluation workers only
  memory_per_worker_mb: null
  placement_strategy: PACK
  placement_timeout_s: 60
  max_restarts: 1
  max_call_retries: 2
  task_timeout_s: null
  worker_torch_threads: 1
```

Unknown keys are errors. `configs/post_training/grpo_ray_tiny_cpu.yaml` is a runnable CPU example; `configs/orchestration/ray_cluster.yaml` is a starting block for an existing cluster.

```bash
forgeline train configs/post_training/grpo_ray_tiny_cpu.yaml
forgeline evaluate --checkpoint runs/sft-tiny-cpu/checkpoints/final --suites arc gsm8k --offline --ray-workers 4
```

Python:

```python
from forgeline.orchestration.config import RayConfig
from forgeline.orchestration.ray_backend import RayRewardPool, RayRolloutEngine, attach_to_algorithm

attach_to_algorithm(grpo, {"type": "ray", "rollout_workers": 4, "reward_workers": 8}, reward_config)
```

## Scope

* Supported: single-turn rollouts for PPO/GRPO/DAPO with native policies; reward/verifier scoring for any provider; tool batches; sharded benchmark evaluation.
* Agent-mode GRPO keeps multi-turn episodes in the learner process (tool calls already run in subprocesses).
* HuggingFace policies are not shipped to rollout workers.
* Ray Serve is not used: the built-in OpenAI-compatible server plus the routing policy cover serving, and nothing in the current serving path needs replica autoscaling.

## Validation

`tests/integration/test_ray_orchestration.py` runs on a private local Ray instance (4 CPUs):

| Test | Checks |
|---|---|
| reward pool | identical values and pass flags to in-process scoring; two distinct worker processes |
| tool pool | ordered results across shards |
| rollout weights | greedy worker rollouts equal learner rollouts before and after a parameter update; no push when weights are unchanged; all workers on the same version |
| killed worker | replacement restores the latest weights; outputs still equal the learner's |
| process crash | a worker that calls `os._exit` is replaced and the call succeeds |
| user errors | propagate without retries |
| retry budget | persistent crashes end in `OrchestrationError` |
| placement | bundles created for PACK; an infeasible request fails with a clear error |
| sharded evaluation | metrics and per-item outputs identical to serial evaluation |
| GRPO via Ray | `Trainer.fit` with 2 rollout + 1 reward workers; weights pushed after optimizer steps |
| CLI | `forgeline train` with the Ray manifest |

`tests/hardware/test_hardware.py::test_ray_rollout_worker_uses_assigned_gpu` (CUDA-marked) checks GPU placement. No multi-node Ray cluster has been exercised and no Ray throughput numbers exist.
