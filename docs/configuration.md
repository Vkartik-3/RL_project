# Configuration

## What

`ExperimentManifest` is the reproducible record of a run: model, tokenizer, dataset, algorithm and its parameters, seed, precision, adapters, distributed strategy, optimizer, schedule, batch size, gradient accumulation, checkpoint settings, evaluation suites, reward/verifier settings and runtime environment.

## Why

One human-readable file (YAML, TOML or JSON) reproduces any run, is saved next to its outputs, and is embedded in every checkpoint.

## How

```yaml
name: dpo-tiny-cpu
algorithm: dpo                 # pretrain sft distill reward_model dpo ppo grpo dapo rlvr (+ rlaif constitutional star hill_climb tabular_ppo via CLI)
model_preset: tiny             # optional; `model` values override preset values
model: {block_size: 64, dropout: 0.0}
checkpoint_path: ""            # initialise from a checkpoint directory
data:
  kind: preference
  path: data/samples/preference/arithmetic_pairs.jsonl
  tokenizer: char
  val_fraction: 0.1
  max_length: 16
  seed: 1337
  extra: {}
trainer:
  max_steps: 60
  batch_size: 4
  gradient_accumulation_steps: 1
  eval_every: 20
  eval_batches: 20
  log_every: 10
  seed: 1337
  device: auto                 # auto | cpu | cuda | mps
  compile_model: false          # torch.compile trained modules in place
  compile_backend: inductor      # inductor | eager | aot_eager
  gradient_checkpointing: false
  optimizer: {name: adamw, learning_rate: 5.0e-4, weight_decay: 0.01, beta1: 0.9, beta2: 0.95, eps: 1.0e-8, grad_clip: 1.0, decay_only_matrices: true, fused: auto}
  schedule: {name: cosine, warmup_steps: 100, decay_steps: 5000, min_lr: 3.0e-5, wsd_stable_fraction: 0.8}   # cosine | wsd | constant | linear_decay
  precision: {dtype: auto, grad_scaler: auto}   # auto → bf16 on capable CUDA, fp16 on other CUDA, fp32 elsewhere
  checkpoint: {directory: checkpoints, save_every: 1000, keep_last: 3, async_save: false, resume_from: "", auto_resume: true}
  adapter: {method: none, rank: 16, alpha: 32.0, dropout: 0.0, target_modules: [], merge_on_save: false}
distributed: {strategy: single, backend: auto, tensor_parallel_size: 1, pipeline_parallel_size: 1, pipeline_micro_batches: 4, fsdp_min_params_to_wrap: 100000, deepspeed_config: ""}
orchestration: {}              # optional: {type: ray, rollout_workers, reward_workers, ...} — see orchestration.md
backend: {}                    # native model; or {type: huggingface, model_name: ..., lora_r, lora_alpha, lora_dropout, target_modules, load_in_8bit, torch_dtype, gradient_checkpointing}
algorithm_params: {beta: 0.1, logprob_reduction: sum}
reward: {}                     # see rlhf.md / rlvr.md
evaluation: []
output_dir: runs/dpo-tiny-cpu
tags: []
```

* Unknown keys at any level raise `ConfigError`.
* `forgeline train m.yaml --set trainer.max_steps=10 --set algorithm_params.beta=0.2` applies dotted overrides (values parsed as YAML).
* `forgeline train` writes `<output_dir>/manifest.yaml` with the captured runtime environment (Python, platform, torch, CUDA availability and devices).

### Policy backend

With `backend.type: huggingface` the SFT, DPO, PPO, GRPO (including agent mode), DAPO and RLVR stages run on a HuggingFace causal LM with PEFT LoRA; `checkpoint_path` restores adapter (and value-head) state from a Forgeline checkpoint. `scripts/export_adapter.py` writes the adapter as a PEFT directory and can publish it to the Hub.

### Data kinds

`pretraining` (token directory), `supervised`, `preference`, `verifiable` (JSONL), and `trajectories` (synthesis records): SFT uses records with rule reward ≥ `extra.reward_threshold`; DPO builds per-molecule yield preference pairs (`extra.pairs_per_molecule`) from the first `extra.train_fraction` (default 0.8) of records; PPO/GRPO/DAPO prompt with the synthesis template and score generated conditions with `reward: {type: synthesis_rule}`.

### Rollout parameters

`algorithm_params.rollout` configures sampling for PPO/GRPO/DAPO/RLVR: `group_size`, `max_new_tokens`, `temperature`, `top_k`, `top_p`, `max_prompt_length`, `record_old_logprobs`, `stop_token_ids`.

### Provided manifests

| Directory | Files |
|---|---|
| `configs/models` | `tiny`, `small`, `large`, `latent_moe` |
| `configs/training` | pretraining (tiny CPU, small char, large GPU), SFT, LoRA, QLoRA, distillation |
| `configs/post_training` | CPU: reward model, DPO, length-normalised DPO, PPO, GRPO, agent GRPO with tools, process-reward GRPO, DAPO, RLVR. GPU (7B LoRA): synthesis SFT/DPO/PPO/GRPO/DAPO, agent GRPO on GSM8K |
| `configs/distributed` | single, DDP, FSDP, 3-D tensor×pipeline, DeepSpeed (+ ZeRO-2 JSON) |
| `configs/inference`, `configs/evaluation` | server defaults, offline benchmark set |
| `configs/deployment` | promotion gate, champion/challenger routing policy |
| `configs/sweeps` | W&B Bayesian sweep for DPO |

## Failure modes

`ConfigError` for invalid values, unknown keys, unknown algorithms and presets; `DistributedConfigError` for topologies that do not tile the world size.

## Local validation

`forgeline validate <manifest> [--world-size N]`; `tests/integration/test_cli_pipeline.py` validates every shipped manifest.

## Limitations

Manifests do not include variable interpolation or includes; compose with `--set`.
