# Capability matrix

Status: **Implemented** (CPU-validated), **Implemented · GPU-validated only by skipped hardware tests**, **Optional extra** (requires an extra dependency), **Not included** (with reason).
"Benchmark" refers to records under `benchmarks/`.

## Models and generation

| Capability | Module | Status | Tests | CPU validated | CUDA required | Multi-GPU required | Benchmark | Notes |
|---|---|---|---|---|---|---|---|---|
| Decoder-only transformer, RMSNorm, tied embeddings | `models/transformer` | Implemented | unit/test_models, unit/test_benchmark_semantics | ✓ | – | – | training/* | loss = token CE (+aux, +MTP) |
| Multi-head / grouped-query attention, SDPA path | `models/attention/causal.py` | Implemented | unit/test_models | ✓ | – | – | training/fineweb_edu_large_a40 | |
| Rotary embeddings, linear and YaRN scaling | `models/attention/rotary.py` | Implemented | unit/test_models | ✓ | – | – | – | |
| Learned absolute positions | `models/transformer/model.py` | Implemented | unit/test_models | ✓ | – | – | – | `use_rope: false` |
| Sliding window, alternating layers, logit soft-cap | `models/attention/causal.py` | Implemented | unit/test_models | ✓ | – | – | – | |
| Multi-head latent attention (naive + absorbed, compressed cache) | `models/attention/latent.py` | Implemented | unit/test_models, unit/test_inference_serving | ✓ | – | – | – | cached = uncached greedy |
| Native sparse attention | `models/attention/sparse.py` | Implemented | unit/test_models | ✓ | – | – | – | no cached decode path |
| SwiGLU / GELU FFN | `models/transformer/feedforward.py` | Implemented | unit/test_models | ✓ | – | – | – | |
| Mixture of experts (softmax/sigmoid, shared, aux-free bias, group-limited, dense first layers) | `models/moe` | Implemented | unit/test_models | ✓ | – | – | – | |
| Multi-token prediction | `models/transformer/block.py` | Implemented | unit/test_models | ✓ | – | – | – | |
| Presets (tiny … alternating_28l) | `core/config.py` | Implemented | unit/test_config | ✓ | – | – | – | |
| Gradient checkpointing | `TransformerLM.enable_gradient_checkpointing` | Implemented | integration (CLI manifests) | ✓ | – | – | – | |
| Sampling: temperature, top-k, top-p, min-p, repetition penalty, stop tokens | `models/generation/sampling.py` | Implemented | unit/test_models | ✓ | – | – | training/char_small_cpu (49 tok/s) | shared by rollouts and engine |
| KV-cached generation | `TransformerLM.prefill/step` | Implemented | unit/test_models | ✓ | – | – | – | |
| Speculative decoding (draft model) | `models/generation/speculative.py` | Implemented | unit/test_models | ✓ | – | – | – | acceptance rate reported |
| MTP self-speculative decoding | `models/generation/speculative.py` | Implemented | unit/test_models | ✓ | – | – | – | |
| Native policy with reference + value head | `models/policy.py::NativePolicy` | Implemented | integration/test_training_lifecycle | ✓ | – | – | – | |
| HuggingFace causal-LM policy + PEFT LoRA + 8-bit loading, manifest backend | `models/policy.py::HFPolicy`, `training/factory.py` | Optional extra | integration/test_huggingface_backend, smoke/test_smoke | ✓ (tiny local model) | 8-bit only | – | peft/lora_qwen2_5_7b, post_training/*, rlvr/* | `huggingface` extra |
| Transformer-encoder reward models (regression on pseudo-labels, Bradley-Terry) | `models/reward.py::EncoderRewardModel` | Optional extra | integration/test_huggingface_backend | ✓ (tiny local encoder) | – | – | – | `huggingface` extra |
| Adapter export / Hub publishing | `scripts/export_adapter.py` | Optional extra | – | – | – | – | – | needs `huggingface_hub` + token for publishing |
| Sequence reward model (Bradley-Terry) | `models/reward.py` | Implemented | integration/test_training_lifecycle | ✓ | – | – | – | |
| Checkpoint → model/policy/reward loading, tokenizer resolution, compile-prefix stripping | `models/loading.py` | Implemented | integration/test_cli_pipeline, unit/test_checkpoints | ✓ | – | – | – | |
| Model inspection (weights, layers, activations, exact attention maps) | `evaluation/inspection.py` | Implemented | integration/test_dashboard | ✓ | – | – | – | attention recomputed per layer; output reconstruction tested |

## Parameter efficiency and quantization

| Capability | Module | Status | Tests | CPU validated | CUDA required | Multi-GPU required | Benchmark | Notes |
|---|---|---|---|---|---|---|---|---|
| LoRA apply / merge / save / load / enable-disable / trainable report | `models/adapters/lora.py` | Implemented | unit/test_adapters_quantization, integration/test_training_lifecycle, integration/test_checkpoint_resume | ✓ | – | – | peft/lora_qwen2_5_7b | |
| QLoRA (NF4 base + LoRA) | `models/adapters/qlora.py` | Implemented | unit/test_adapters_quantization, integration/test_training_lifecycle | ✓ | – | – | – | |
| NF4 quantize / dequantize, memory report | `models/quantization/nf4.py` | Implemented | unit/test_adapters_quantization, unit/test_benchmark_semantics | ✓ | – | – | – | |
| GGUF export FP16 / Q8_0 / Q4_0 + read-back | `models/quantization/gguf.py` | Implemented | unit/test_adapters_quantization, integration/test_cli_pipeline | ✓ | – | – | – | aligned offsets |
| fp16 / bf16 autocast, GradScaler | `training/common/precision.py` | Implemented · GPU tests skipped | hardware/test_hardware | fp32/bf16 on CPU | fp16 | – | training/fineweb_edu_large_a40 | |

## Data

| Capability | Module | Status | Tests | CPU validated | CUDA required | Multi-GPU required | Benchmark | Notes |
|---|---|---|---|---|---|---|---|---|
| Char / tiktoken / HuggingFace tokenizers, metadata | `data/tokenizers.py` | Implemented (tiktoken, HF: optional extras) | unit/test_data | ✓ (char) | – | – | – | |
| Text corpus preparation (train/val `.bin`) | `data/preprocessing.py` | Implemented | unit/test_data, integration/test_cli_pipeline | ✓ | – | – | training/char_small_cpu | |
| HuggingFace corpus → shards (constant memory) | `data/preprocessing.py`, `data/streaming.py` | Optional extra | unit/test_data (writer) | ✓ (writer) | – | – | training/fineweb_edu_large_a40 | |
| Memmap random windows, dtype detection | `data/pretraining.py` | Implemented | unit/test_data | ✓ | – | – | training/* | |
| Sharded / text-glob / HF streaming datasets, shard splitting | `data/streaming.py` | Implemented (HF: optional) | unit/test_data | ✓ | – | – | – | rank/worker sharding |
| Record schemas + malformed handling, deterministic splits | `data/records.py` | Implemented | unit/test_data, failure/test_failure_modes | ✓ | – | – | – | |
| SFT / preference / verifiable datasets, collators | `data/supervised.py`, `preference.py`, `verifiable.py`, `collators.py` | Implemented | unit/test_data | ✓ | – | – | – | |
| GSM8K tool-task builder | `data/verifiable.py` | Optional extra | unit/test_data (answer extraction) | ✓ | – | – | rlvr/* | |
| Sample datasets (text, SFT, preference, verifiable, process, synthesis) | `data/samples` | Implemented | integration, unit/test_benchmark_semantics | ✓ | – | – | post_training/*, rlvr/* | |

## Training and post-training

| Capability | Module | Status | Tests | CPU validated | CUDA required | Multi-GPU required | Benchmark | Notes |
|---|---|---|---|---|---|---|---|---|
| Shared trainer (optimizer groups, schedules, accumulation, clipping, eval, checkpoints, events) | `training/common/trainer.py` | Implemented | integration/*, failure/test_failure_modes | ✓ | – | – | training/* | |
| `torch.compile` of trained modules (in place, checkpoint-compatible) | `Trainer._compile_modules` | Implemented | integration/test_checkpoint_resume | ✓ (eager backend) | inductor speed-up on GPU | – | training/fineweb_edu_large_a40 | losses identical to uncompiled |
| Merged-adapter checkpoint on save | `Trainer._save_merged` (`adapter.merge_on_save`) | Implemented | integration/test_checkpoint_resume | ✓ | – | – | – | LoRA only |
| Cosine / WSD / constant / linear schedules | `training/common/schedule.py` | Implemented | unit/test_training_utils, unit/test_benchmark_semantics | ✓ | – | – | training/* | |
| Pretraining | `training/pretrain.py` | Implemented | integration/test_training_lifecycle, integration/test_ddp_two_process | ✓ | – | – | training/* | |
| SFT (full / LoRA / QLoRA) | `training/sft.py` | Implemented | integration/test_training_lifecycle | ✓ | – | – | post_training/sft_synthesis | |
| Knowledge distillation (+ feature matching) | `training/distill.py` | Implemented | integration/test_training_lifecycle | ✓ | – | – | – | |
| Reward-model training | `training/reward_model.py` | Implemented | integration/test_training_lifecycle | ✓ | – | – | – | |
| DPO (sum / mean / reference-free / label smoothing) | `training/dpo.py` | Implemented | integration/test_training_lifecycle, unit/test_training_utils, unit/test_benchmark_semantics | ✓ | – | – | post_training/dpo_synthesis | |
| PPO (value head; ratio, KL, entropy modes), GAE | `training/ppo.py`, `training/common/advantages.py` | Implemented | integration/test_training_lifecycle, unit/test_training_utils, unit/test_benchmark_semantics | ✓ | – | – | post_training/ppo_synthesis* | reproduction recommended (normalisation constant) |
| GRPO (clipped / REINFORCE, sequence or token ratio, KL estimators) | `training/grpo.py` | Implemented | integration/test_training_lifecycle, integration/test_ddp_two_process | ✓ | – | – | rlvr/process_reward_grpo_gsm8k | |
| Agent GRPO with tools | `training/grpo.py` (`agent`), `rollouts/agent.py` | Implemented | integration/test_training_lifecycle | ✓ | – | – | rlvr/agent_grpo_gsm8k_tools | reproduction recommended |
| Critic-augmented agent rewards | `rollouts/rewards/critic.py` + `CompositeReward` | Implemented | unit/test_rollouts_rewards | ✓ | – | – | – | |
| Process-reward GRPO | `rollouts/rewards/process.py` + GRPO `reinforce` | Implemented | unit/test_rollouts_rewards, unit/test_benchmark_semantics | ✓ | – | – | rlvr/process_reward_grpo_gsm8k | |
| DAPO | `training/dapo.py` | Implemented | integration/test_training_lifecycle, unit/test_training_utils | ✓ | – | – | – | |
| RLVR builder + pass-rate evaluation | `training/rlvr.py` | Implemented | integration/test_training_lifecycle | ✓ | – | – | – | |
| Self-judge AI feedback → reference-free DPO rounds | `training/rlaif.py::RLAIFTrainer` | Implemented | integration/test_cli_pipeline | ✓ | – | – | – | |
| Pairwise judge preference generation | `training/rlaif.py::run_pairwise_rlaif` | Implemented | integration/test_cli_pipeline | ✓ | – | – | post_training/ai_feedback_data | exact reproduction |
| Constitutional critique → revise | `training/rlaif.py::run_constitutional` | Implemented | integration/test_cli_pipeline | ✓ | – | – | post_training/ai_feedback_data | exact reproduction |
| STaR | `training/star.py::STaRTrainer` | Implemented | integration/test_cli_pipeline | ✓ | – | – | – | |
| Hill climbing | `training/star.py::HillClimber` | Implemented | integration/test_cli_pipeline | ✓ | – | – | rlvr/hill_climb_gsm8k | |
| Manifest → algorithm factory | `training/factory.py` | Implemented | integration/test_cli_pipeline | ✓ | – | – | – | |
| Benchmark summary table / CSV | `scripts/summarize_benchmarks.py` | Implemented | – | ✓ | – | – | all | |
| Hyper-parameter sweeps | `configs/sweeps`, `scripts/run_sweep_trial.py` | Optional extra | – | – | – | – | – | `observability` extra (wandb) |

## Rollouts, verifiers, rewards, tools

| Capability | Module | Status | Tests | CPU validated | CUDA required | Multi-GPU required | Benchmark | Notes |
|---|---|---|---|---|---|---|---|---|
| Group rollout engine with sampled log-probs | `rollouts/engine.py` | Implemented | unit/test_rollouts_rewards | ✓ | – | – | – | |
| Multi-turn agent engine, segment log-probs | `rollouts/agent.py` | Implemented | integration/test_training_lifecycle, failure/test_failure_modes | ✓ | – | – | rlvr/agent_grpo_gsm8k_tools | |
| Python executor (subprocess, timeout, output cap) | `rollouts/tools/python_executor.py` | Implemented | unit/test_rollouts_rewards | ✓ | – | – | – | not a security sandbox |
| Tag parser | `rollouts/tools/parser.py` | Implemented | unit/test_rollouts_rewards | ✓ | – | – | – | |
| Math / exact-match / tagged-answer / code / format verifiers | `rollouts/verifiers` | Implemented | unit/test_rollouts_rewards, failure/test_failure_modes | ✓ | – | – | – | |
| Rule, composite, process, critic, reward-model, blended rewards | `rollouts/rewards` | Implemented | unit/test_rollouts_rewards | ✓ | – | – | – | |

## Distributed

| Capability | Module | Status | Tests | CPU validated | CUDA required | Multi-GPU required | Benchmark | Notes |
|---|---|---|---|---|---|---|---|---|
| Process-group runtime, topology validation | `distributed/runtime.py` | Implemented | unit/test_distributed, integration/test_distributed_gloo | ✓ | – | – | – | |
| DDP (step trainer, all algorithms) | `distributed/strategies.py`, `Trainer` | Implemented | integration/test_ddp_two_process | ✓ (2 gloo processes) | – | for NCCL | – | |
| FSDP (pretrain, distill) | `distributed/strategies.py` | Implemented · GPU tests skipped | hardware/test_hardware | config only | ✓ | ✓ | – | |
| DeepSpeed ZeRO | `distributed/strategies.py`, `configs/distributed/deepspeed_zero2.json` | Optional extra · GPU | unit/test_distributed (config), hardware/test_hardware | config only | ✓ | ✓ | – | custom loops |
| 3-D mesh | `distributed/topology.py` | Implemented | unit/test_distributed | ✓ | – | – | – | |
| Tensor parallel (column/row layers, block surgery) | `distributed/tensor_parallel.py` | Implemented · multi-rank GPU tests skipped | integration/test_distributed_gloo, hardware/test_hardware | ✓ (degree 1) | multi-rank | ✓ | – | custom loops |
| Pipeline parallel (stages, 1F1B, configured micro-batches) | `distributed/pipeline_parallel.py`, `PipelineParallelStrategy.build_scheduler` | Implemented | unit/test_distributed, integration/test_distributed_gloo | ✓ (1 stage) | – | multi-stage | – | custom loops |
| torchrun launching | CLI | Implemented | integration/test_ddp_two_process (launcher env) | ✓ | – | – | – | |

## Orchestration (optional Ray backend)

| Capability | Module | Status | Tests | CPU validated | CUDA required | Multi-GPU required | Benchmark | Notes |
|---|---|---|---|---|---|---|---|---|
| Ray runtime (local instance or cluster address), config validation without Ray | `orchestration/config.py`, `ray_backend.init_ray` | Optional extra | unit/test_orchestration_config, integration/test_ray_orchestration | ✓ (local, 4 CPUs) | – | – | – | `ray` extra; core never imports Ray |
| Rollout workers with on-policy weight sync | `RayRolloutEngine` | Optional extra | integration/test_ray_orchestration | ✓ | GPU placement test skipped | – | – | native policies; drop-in for PPO/GRPO/DAPO |
| Reward / verifier workers | `RayRewardPool` (`score_many`) | Optional extra | integration/test_ray_orchestration | ✓ | – | – | – | identical to in-process scoring |
| Tool workers | `RayToolPool` | Optional extra | integration/test_ray_orchestration | ✓ | – | – | – | subprocess isolation per call |
| Sharded evaluation | `RayEvaluator`, `forgeline evaluate --ray-workers` | Optional extra | integration/test_ray_orchestration | ✓ | – | – | – | per-item results equal serial |
| Placement groups, worker replacement, bounded retries | `_WorkerPool` | Optional extra | integration/test_ray_orchestration | ✓ | – | – | – | kill, crash, retry budget, infeasible placement tested |
| Manifest integration | `orchestration:` block, `training/factory.py` | Optional extra | integration/test_ray_orchestration (Trainer + CLI) | ✓ | – | – | – | agent-mode rollouts stay in process |

## Checkpoints, evaluation, inference, serving, lifecycle, observability

| Capability | Module | Status | Tests | CPU validated | CUDA required | Multi-GPU required | Benchmark | Notes |
|---|---|---|---|---|---|---|---|---|
| Atomic directory checkpoints, async save, rotation, best/final/latest | `checkpoints/manager.py` | Implemented | unit/test_checkpoints | ✓ | – | – | – | |
| Validation (missing, corrupt, truncated, incomplete, mismatched) | `checkpoints` | Implemented | failure/test_failure_modes | ✓ | – | – | – | |
| Exact resume (model, optimizer, scaler, RNG, adapters) | `Trainer.resume` | Implemented | integration/test_checkpoint_resume | ✓ | – | – | – | |
| Held-out loss, pass rate, malformed rate, agent accuracy | `evaluation/quality.py` | Implemented | unit/test_evaluation | ✓ | – | – | – | |
| Reward statistics, preference win rate | `evaluation/reward.py` | Implemented | unit/test_evaluation, unit/test_benchmark_semantics | ✓ | – | – | post_training/* | policy-dependence flag |
| Latency / throughput / memory | `evaluation/performance.py` | Implemented | unit/test_evaluation | ✓ | memory metrics | – | inference | |
| MMLU, HellaSwag, ARC, GSM8K, TruthfulQA, HumanEval | `evaluation/suites` | Implemented (full data: optional extra) | unit/test_evaluation | ✓ (offline samples) | – | – | – | |
| Regression rules | `evaluation/regression.py` | Implemented | unit/test_evaluation | ✓ | – | – | – | |
| Paged block allocator | `inference/paged_memory.py` | Implemented | unit/test_inference_serving | ✓ | – | – | – | |
| Continuous-batching scheduler, cancellation | `inference/scheduler.py` | Implemented | unit/test_inference_serving | ✓ | – | – | – | |
| KV-cached engine (batched decode = greedy) | `inference/engine.py` | Implemented | unit/test_inference_serving, failure/test_failure_modes, hardware/test_hardware | ✓ | – | – | – | |
| OpenAI-compatible API, SSE, metrics | `serving` | Implemented | unit/test_inference_serving, integration/test_cli_pipeline | ✓ (live HTTP) | – | – | – | |
| Candidate registry and states | `deployment/registry.py`, `candidates.py` | Implemented | unit/test_deployment | ✓ | – | – | – | |
| Promotion gates (fail closed) | `deployment/promotion.py` | Implemented | unit/test_deployment, integration/test_cli_pipeline | ✓ | – | – | – | |
| Feature flags | `deployment/feature_flags.py` | Implemented | unit/test_deployment | ✓ | – | – | – | |
| Champion / challenger / shadow routing, offline replay | `deployment/rollout.py` | Implemented | unit/test_deployment | ✓ | – | – | – | |
| Rollback, kill switch | `deployment/rollback.py` | Implemented | unit/test_deployment | ✓ | – | – | – | |
| Structured logging, JSONL metrics/events, spans | `observability` | Implemented | unit/test_observability_cli | ✓ | – | – | – | |
| W&B / TensorBoard sinks | `observability/metrics.py` | Optional extra | – | – | – | – | – | |
| Dashboard: runs, metric charts, events, checkpoint validity, evaluations, registry history, benchmark evidence, model inspection; JSON export | `dashboard/`, `forgeline dashboard` | Implemented (optional to run) | integration/test_dashboard | ✓ (live HTTP) | – | – | – | read-only; stdlib only; checkpoint loading restricted to run roots |
| CLI | `cli/main.py` | Implemented | unit/test_observability_cli, integration/test_cli_pipeline | ✓ | – | – | – | |

## Synthesis domain

| Capability | Module | Status | Tests | CPU validated | CUDA required | Multi-GPU required | Benchmark | Notes |
|---|---|---|---|---|---|---|---|---|
| Rule reward, reward provider for generated conditions | `domains/synthesis/reward.py` | Implemented | unit/test_synthesis_domain, unit/test_benchmark_semantics | ✓ | – | – | post_training/* | |
| Condition constraints and validity penalty | `domains/synthesis/constraints.py` | Implemented | unit/test_synthesis_domain | ✓ | – | – | – | |
| Prompt template, JSON condition codec | `domains/synthesis/prompts.py` | Implemented | unit/test_synthesis_domain | ✓ | – | – | – | |
| 128-d features + Morgan fingerprints | `domains/synthesis/features.py` | Implemented (RDKit optional) | unit/test_synthesis_domain | ✓ | – | – | – | |
| Preference pairs, SFT records, trajectory manifests (`data.kind: trajectories`) | `domains/synthesis/preferences.py`, `training/factory.py` | Implemented | unit/test_synthesis_domain, integration/test_huggingface_backend | ✓ | – | – | post_training/dpo_synthesis, sft_synthesis | reproduces 139 SFT / 80 DPO examples |
| Reaction-SMILES corpus download | `scripts/download_reaction_smiles.py` | Implemented | – | network | – | – | – | |
| Simulated / yield-series data generators | `domains/synthesis/generator.py`, `scripts/generate_synthesis_data.py` | Implemented | unit/test_benchmark_semantics | ✓ | – | – | – | |
| Tabular actor-critic PPO, leave-one-molecule-out | `domains/synthesis/tabular_ppo.py`, `examples/05_synthesis_generalization.py` | Implemented | unit/test_synthesis_domain, unit/test_benchmark_semantics | ✓ | – | – | post_training/tabular_ppo_synthesis | exact reproduction |

## Sequential decisioning (budgeted allocation)

| Capability | Module | Status | Tests | CPU validated | CUDA required | Multi-GPU required | Benchmark | Notes |
|---|---|---|---|---|---|---|---|---|
| Finite-horizon budgeted environment with hidden policy-dependent pressure | `domains/allocation/env.py` | Implemented | unit/test_allocation_env | ✓ | – | – | decisioning/budgeted_allocation | POMDP; endogenous feedback tested |
| Threshold heuristic, primal-dual pacer | `domains/allocation/policies.py` | Implemented | integration/test_allocation_pipeline | ✓ | – | – | same | |
| Stateless and GRU sequence PPO via the shared trainer | `domains/allocation/ppo.py`, `policies.py` | Implemented | integration/test_allocation_pipeline | ✓ | – | – | same | manifest algorithm `allocation_ppo` |
| Logged trajectories with propensities | `domains/allocation/rollout.py` | Implemented | unit/test_allocation_ope | ✓ | – | – | same | schema v1 |
| OPE: IPS, SNIPS, PDIS, DR, clipping, ESS/support diagnostics, bootstrap CIs | `domains/allocation/ope.py` | Implemented | unit/test_allocation_ope (hand-computed references) | ✓ | – | – | same | |
| Hindsight oracle / regret | `domains/allocation/oracle.py` | Implemented | unit/test_allocation_env | ✓ | – | – | same | upper bound, not a policy |
| Pacing / budget metrics | `domains/allocation/rollout.py` | Implemented | unit/test_allocation_env | ✓ | – | – | same | |
| Simulated A/B (deterministic assignment, bootstrap CI, permutation test, guardrails) → promotion gate | `domains/allocation/experiment.py`, `configs/allocation/promotion_gate.yaml` | Implemented | integration/test_allocation_pipeline | ✓ | – | – | same | |
| Shadow evaluation | `domains/allocation/experiment.py::shadow_evaluate` | Implemented | integration/test_allocation_pipeline | ✓ | – | – | same | |
| One-command benchmark and CLI (`forgeline allocation …`) | `domains/allocation/benchmark.py`, `cli/main.py` | Implemented | integration/test_allocation_pipeline | ✓ | – | – | same | |

## Not included

| Capability | Decision |
|---|---|
| FP8 training / FP8 linear layers | Outside the supported precision set; fp16/bf16, NF4 and 8-bit paths cover efficiency without Hopper-class hardware. |
| Vendor CUDA kernels for latent attention | The PyTorch latent-attention implementation (naive and absorbed paths) is used so models run on any device. |
| Expert-parallel MoE dispatch | MoE layers run under data parallelism; expert-parallel all-to-all is listed under technical debt. |
| Ray Serve | The built-in OpenAI-compatible server and routing policy cover serving; nothing in the serving path needs replica autoscaling. |
| Ray-based gradient synchronisation / Ray Train | Gradients stay with the torch.distributed strategies; Ray orchestrates rollouts, rewards, tools and evaluation. |
| Cloud-provider launch and monitoring scripts | Manifests (`configs/training/pretrain_large_gpu.yaml`) and `scripts/local_validation.sh` are the supported entry points. |
| Result charts as images | All numbers are structured data in `benchmarks/`; the dashboard renders run curves from the logs. |
