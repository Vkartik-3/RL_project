# Distributed execution

## What

One `DistributedStrategy` interface over single-process, DDP, FSDP, DeepSpeed, tensor-parallel and pipeline-parallel execution, plus the process-group runtime and a 3-D (tensor × pipeline × data) process mesh.

## Why

Scaling strategy is a deployment decision, not an algorithm change. Every training stage runs under any strategy selected in the manifest.

## How

### Strategies (`distributed/strategies.py`)

| `strategy` | Setup | Model wrapping |
|---|---|---|
| `single` | none | unchanged |
| `ddp` | process group (NCCL on CUDA, gloo on CPU) | `DistributedDataParallel` (no-op for world size 1) |
| `fsdp` | process group | `FullyShardedDataParallel`, size-based auto-wrap (`fsdp_min_params_to_wrap`), `MixedPrecision` (param/reduce/buffer dtype), `FULL_SHARD`; CUDA only |
| `deepspeed` | process group | `deepspeed.initialize` with a JSON config (ZeRO-2 example in `configs/distributed/deepspeed_zero2.json`) |
| `tensor_parallel` | process group + mesh | attention q/k/v (and latent-attention projections) → column-parallel; output projections → row-parallel; SwiGLU w1/w3 column, w2 row (GELU c_fc/c_proj likewise) |
| `pipeline_parallel` | process group + mesh | the rank's slice of blocks as a `PipelineStage` (first stage embeds, last stage owns norm + head), optional tensor parallelism inside the stage |

### Runtime (`distributed/runtime.py`)

`read_launcher_env()` (RANK, LOCAL_RANK, WORLD_SIZE), `init_process_group(backend)` (creates a 1-process group when no launcher is present), `is_main_process`, `world_size`, `rank`, `barrier`, `all_reduce_mean`, `validate_topology(world, tp, pp)` → data-parallel degree.

### Mesh (`distributed/topology.py`)

`global_rank = dp·(tp·pp) + pp_rank·tp + tp_rank`. `ParallelMesh` computes ranks and creates TP/PP/DP groups; `prev_pipeline_rank` / `next_pipeline_rank` identify pipeline neighbours. It can also be constructed with explicit world size/rank (no process group) for planning.

### Tensor parallelism (`distributed/tensor_parallel.py`)

`ColumnParallelLinear` shards output features (identity forward, all-reduce input gradients); `RowParallelLinear` shards input features (all-reduce outputs). `from_linear` slices existing weights, so a trained model can be parallelised.

### Pipeline parallelism (`distributed/pipeline_parallel.py`)

`one_f_one_b_schedule(n_micro, pp_rank, pp_size)` returns the forward/backward order (warm-up forwards, steady 1F1B, cool-down backwards). `PipelineScheduler.run(inputs, targets)` executes it with point-to-point sends and receives between stages and returns the averaged loss on the last stage. `PipelineParallelStrategy.build_scheduler(stage)` builds it with `distributed.pipeline_micro_batches`.

### Relation to Ray

Ray worker pools ([orchestration.md](orchestration.md)) run rollout generation, reward scoring, tools and evaluation outside the learner. They complement these strategies and never synchronise gradients.

## Configuration

```yaml
distributed: {strategy: fsdp, backend: nccl, fsdp_min_params_to_wrap: 100000}
distributed: {strategy: pipeline_parallel, backend: nccl, tensor_parallel_size: 2, pipeline_parallel_size: 4, pipeline_micro_batches: 4}
distributed: {strategy: deepspeed, deepspeed_config: configs/distributed/deepspeed_zero2.json}
```

### Step-trainer integration

| Strategy | `forgeline train` / `Trainer` |
|---|---|
| `single` | every stage |
| `ddp` | every stage, including PPO/GRPO/DAPO/RLVR rollouts: parameters are broadcast from rank 0, gradients are averaged across ranks after accumulation, and data-sampling and rollout seeds are offset by rank |
| `fsdp` | `pretrain` and `distill` (the model forward is wrapped); evaluation and full-state checkpoint gathering run on all ranks, files are written by rank 0; resume is initialised via `checkpoint_path` |
| `deepspeed`, `tensor_parallel`, `pipeline_parallel` | strategy objects for custom training loops; `forgeline train` rejects them with `DistributedConfigError` |

Launch: `torchrun --nproc_per_node=8 -m forgeline.cli.main train <manifest> --set distributed.strategy=ddp`.

`forgeline validate <manifest> --world-size 8` checks the topology without GPUs.

## Failure modes

| Condition | Error |
|---|---|
| world size not divisible by TP × PP | `DistributedConfigError` |
| tensor/pipeline strategy with degree 1 | `DistributedConfigError` |
| `nccl` without CUDA; FSDP without CUDA | `DistributedConfigError` |
| DeepSpeed config missing or without `zero_optimization` | `DistributedConfigError` |
| `wrap_model` before `setup` | `DistributedConfigError` |
| DeepSpeed not installed | `OptionalDependencyError` |

## Local validation

* `tests/unit/test_distributed.py` — mesh layouts, 1F1B orders, topology validation, strategy construction.
* `tests/integration/test_distributed_gloo.py` — inside a real single-process gloo group: tensor-parallel layers (degree 1) reproduce the original logits; the pipeline scheduler's loss equals the model loss and produces gradients; the pipeline strategy builds its scheduler with the configured micro-batch count; DDP strategy setup.
* `tests/integration/test_ddp_two_process.py` — two CPU processes with gloo run pretraining and GRPO under the ddp strategy; replicas remain identical after synchronised updates while ranks sample different batches.
* `tests/hardware/test_hardware.py` — FSDP and 2-rank tensor parallelism under torchrun (skipped without ≥ 2 GPUs).

## Hardware requirements

FSDP, DeepSpeed and NCCL-based tensor/pipeline parallelism need CUDA GPUs; tensor parallelism is intended within a node (fast interconnect).

## Limitations

* No multi-GPU measurements have been recorded; multi-rank correctness is covered by hardware-marked tests that must be run on GPUs.
* The pipeline scheduler assumes a fixed activation shape per stage and float32 activations.
* Tensor-parallel surgery covers the transformer blocks; embeddings and the LM head remain replicated.
* Expert parallelism for MoE layers is not provided.
* The step trainer's FSDP path does not save optimizer state and does not support `resume`.
