# Forgeline — System Report

## 1. Project name

**Forgeline** — post-training, distributed training, evaluation, inference and model-lifecycle framework for language models. Version 0.1.0, MIT license.

## 2. Package name

`forgeline` (src layout, `src/forgeline`), command-line entry point `forgeline`, logging namespace `forgeline.*`. Python ≥ 3.10; core dependencies `torch`, `numpy`, `pyyaml`; extras `training`, `huggingface`, `distributed`, `ray`, `serving`, `observability`, `chemistry`, `dev`.

Size: 134 source modules (~14.4k lines of Python plus the dashboard page), 26 test modules (229 tests), 40 configuration files, 5 runnable examples, 7 scripts, 21 documents including the complete guide.

## 3. Architecture

```
ExperimentManifest ─► RunContext (seed, device, dtype, output dir, metrics sink)
        │
        ├─ data ─────────────┐
        ├─ models / policies ┤
        ├─ rollouts ─────────┼─► PostTrainingAlgorithm ─► Trainer ─► checkpoints ─► evaluation ─► registry ─► promotion gate ─► routing
        ├─ distributed ──────┘                              │                     │                                                │
        └─ observability ◄──────────────────────────────────┘                     └─► quantization / export      inference engine ─► serving
```

Contracts: `PolicyModel` (native transformer and HuggingFace+PEFT implementations), `RewardProvider`, `Verifier`, `Trajectory`, `PostTrainingAlgorithm`, `DistributedStrategy`, `EvaluationSuite`/`EvaluationResult`, `MetricsSink`, `ModelCandidate`. The trainer owns optimizer grouping, schedules, precision, accumulation, clipping, evaluation cadence, checkpoints, RNG capture, data-parallel gradient synchronisation and lifecycle events; algorithms own data collection and loss mathematics.

## 4. Major subsystems

| Subsystem | Package | Summary |
|---|---|---|
| Core | `core` | model spec + 9 presets, trainer/data/distributed configs, manifest (YAML/TOML/JSON, strict keys), protocols, registry, runtime, run context, 27 typed errors with hints |
| Data | `data` | char/tiktoken/HF tokenizers; validated SFT, preference, verifiable, process and reward schemas; memmap, sharded, text-glob and HF streaming corpora; constant-memory shard writer; collators |
| Models | `models` | decoder-only transformer with GQA/MHA, latent attention, native sparse attention, RoPE/YaRN, sliding windows, logit cap, MoE, MTP; sampling; speculative decoding; LoRA/QLoRA; NF4; GGUF; policies; sequence and encoder reward models |
| Training | `training` | trainer; pretraining, SFT, distillation, reward model, DPO, PPO, GRPO, DAPO, RLVR, AI feedback, STaR, hill climbing; manifest factory |
| Rollouts | `rollouts` | group rollout engine, multi-turn tool agent, Python executor, tag parser, 5 verifiers, rule/composite/process/critic/learned rewards |
| Distributed | `distributed` | runtime, 3-D mesh, column/row-parallel layers, pipeline stages + 1F1B, 6 strategies |
| Checkpoints | `checkpoints` | atomic manifest checkpoints, validation, discovery, resume |
| Evaluation | `evaluation` | quality, reward, performance suites; 6 standard benchmarks; regression rules; inspection |
| Inference | `inference` | request lifecycle, paged block allocator, continuous-batching scheduler, KV-cached engine |
| Serving | `serving` | OpenAI-compatible router, engine backend, HTTP server with SSE |
| Deployment | `deployment` | candidate registry, fail-closed promotion gates, feature flags, champion/challenger/shadow routing, offline replay, rollback, kill switch |
| Observability | `observability` | structured logs, JSONL/no-op/W&B/TensorBoard sinks, events, spans |
| Orchestration (optional) | `orchestration` | Ray rollout, reward/verifier, tool and evaluation worker pools; on-policy weight sync; placement groups; worker replacement with bounded retries |
| Dashboard (optional) | `dashboard` | read-only local UI and JSON export over runs, checkpoints, evaluations, registry, benchmark evidence and model inspection |
| Sequential decisioning | `domains.allocation` | budgeted allocation POMDP with policy-dependent pressure, heuristic/primal-dual/PPO/GRU policies, logged trajectories, OPE, oracle, A/B, shadow |
| Synthesis domain | `domains.synthesis` | rule reward, constraints, prompt codec, features, preference/SFT construction, simulators, tabular actor-critic PPO |

## 5. Supported algorithms

| Family | Algorithms and variants |
|---|---|
| Supervised | pretraining (cross-entropy, MoE aux, MTP), SFT (full / LoRA / QLoRA), knowledge distillation (logit KL + feature MSE) |
| Preference | DPO (sum / length-normalised, reference-free, label smoothing); Bradley-Terry reward models (transformer backbone; encoder regression or preference) |
| Policy optimisation | PPO (value head; sequence-mean or token ratios; KL monitor/penalty/reward; sampled or full entropy; GAE utility); GRPO (clipped or REINFORCE; sequence or token ratios; log-prob-difference or k3 KL; optional zero-variance skipping); agent GRPO with tools; DAPO (clip-higher, dynamic sampling, token-level loss, overlong shaping, entropy bonus) |
| Verifiable rewards | RLVR builder (math, exact match, tagged answer, code execution, format verifiers; correctness + format weighting); process rewards; critic-augmented rewards |
| Self-improvement / AI feedback | self-judge candidate rounds → reference-free DPO; pairwise judge preference generation; constitutional critique → revise (SFT + DPO data); STaR; rejection-sampling hill climbing with agent GRPO |
| Domain baseline | tabular actor-critic PPO on synthesis state vectors |
| Sequential decisioning | stateless and GRU sequence-conditioned PPO (`allocation_ppo`), threshold heuristic, online primal-dual pacer; OPE (IPS/SNIPS/PDIS/DR); simulated A/B; shadow |

## 6. Supported distributed strategies

| Strategy | Step trainer | Validation status |
|---|---|---|
| single | all stages | CPU |
| ddp | all stages including RL (parameter broadcast, gradient averaging, per-rank seeds) | two CPU processes with gloo: replicas identical after updates, per-rank data differs |
| fsdp | pretrain, distill | configuration on CPU; execution tests require ≥ 2 GPUs (skipped) |
| deepspeed | custom loops (`DeepSpeedStrategy.initialize`) | config validation on CPU; execution requires the extra and CUDA (skipped) |
| tensor_parallel | custom loops | degree-1 numerical equivalence in a gloo group; multi-rank test requires GPUs (skipped) |
| pipeline_parallel | custom loops | single-stage 1F1B loss equivalence in a gloo group; schedules unit-tested |

Ray worker pools (optional) run alongside any strategy: rollout generation, reward scoring, tools and evaluation move to actors while gradients stay with the strategy. Validated on a local 4-CPU Ray instance.

## 7. Inference capabilities

KV-cached prefill/step for grouped-query and latent attention; continuous batching with FIFO admission bounded by `max_batch` and paged block capacity; length-grouped batched decoding (outputs equal greedy single-sequence generation); per-request sampling (temperature, top-k, top-p, min-p, repetition penalty, stop tokens); cancellation; failure isolation; background engine loop; speculative decoding with draft models or MTP heads; OpenAI-compatible `/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions` with SSE, and `/metrics`.

## 8. Model lifecycle capabilities

Local JSON candidate registry with metadata (version, algorithm, checkpoint, tokenizer, dataset, precision, strategy, spec, metrics, timestamps, promotion status/report, rollback target, history); states EXPERIMENTAL / SHADOW / CHALLENGER / CHAMPION / RETIRED with enforced transitions and a single champion; promotion gates over absolute and baseline-relative thresholds that fail closed on missing or NaN metrics; deterministic hash routing for champion/challenger splits, shadow mirroring decisions, pinning and disabling; traffic simulation and offline replay; persistent feature flags with percentage rollout and allow/deny lists; one-command rollback that also updates routing; kill switch. All available from Python and `forgeline registry` / `forgeline serve --routing`.

## 9. Test results

```
python -m pytest tests
221 passed, 8 skipped in 37 s   (229 collected; huggingface and ray extras installed)
```

Clean install with only `.[dev]`: 204 passed, 9 skipped (HuggingFace and Ray integration modules skip; Ray is never imported).

| Suite | Scope |
|---|---|
| unit (16 modules, 116 tests) | configs, data, models, adapters/quantization, training math, rollouts/verifiers/rewards, checkpoints, distributed planning, evaluation, deployment, inference/serving, observability/CLI, orchestration config, synthesis domain, benchmark semantics |
| integration (9 modules, 85 tests) | every training stage; exact resume; torch.compile equivalence; merged-adapter checkpoints; CLI pipeline with live HTTP server; two-process DDP; gloo tensor/pipeline execution; HuggingFace backend with a tiny local model; Ray worker pools; dashboard |
| failure (1 module) | checkpoint missing/bad metadata/truncated/incomplete/mismatched/incompatible; malformed data; invalid rewards; verifier and tool failures; rollout failure; distributed misconfiguration; unsupported strategy combinations; promotion rejection; serving backend failure; engine sequence isolation; quantization/export errors; non-finite loss |
| smoke (1 module, 4 tests) | all modules import (extras fail only with the actionable error); public API; optional-extra errors |
| hardware (1 module, 7 tests) | skipped with reasons: 4 × no CUDA device (incl. Ray GPU rollout worker), 2 × fewer than 2 CUDA devices, 1 × DeepSpeed extra not installed |

Environment: macOS on Apple Silicon (CPU), Python 3.12, PyTorch 2.14, transformers 5.17, PEFT 0.21, Ray 2.58.

## 10. CPU-local validation

Validated on CPU in this environment:

* forward, backward and generation for every attention / FFN / MoE / MTP / RoPE variant; cached = uncached decoding;
* LoRA identity at init, disable/enable, save/load, merge; NF4 and QLoRA; GGUF FP16/Q8_0/Q4_0 round trips with aligned offsets;
* pretraining loss decreases; SFT (full, LoRA with frozen base, QLoRA); distillation;
* DPO raises preference margin to 100% accuracy on a toy set; reward model learns ranking;
* PPO (3 configurations), GRPO (3 configurations), agent GRPO with executed tool calls, DAPO dynamic sampling, RLVR with format verification, STaR, self-judge AI feedback, hill climbing;
* HuggingFace backend: SFT, DPO, PPO, GRPO, DAPO and agent GRPO on a tiny local Llama with LoRA, adapter checkpoint round trip, encoder reward models;
* checkpoint save → reload → continue matches uninterrupted losses within 1e-5;
* two-process DDP (pretrain and GRPO); tensor/pipeline code paths in a gloo group;
* evaluation suites and all six benchmarks offline; regression rules;
* inference engine batching equivalence, stop tokens, cancellation, block accounting;
* HTTP serving round trip with streaming;
* registry, gates (success, rejection, boundaries, missing, NaN, invalid config), routing split accuracy, feature flags, rollback, kill switch;
* the complete CLI lifecycle and all five examples;
* Ray pools on a local instance: worker rollouts equal learner rollouts before and after weight updates, killed or crashed workers are replaced with current weights, placement groups, sharded evaluation identical to serial, GRPO through `Trainer` and `forgeline train` with Ray workers;
* dashboard data layer and HTTP API on real run outputs; attention maps that reconstruct each layer's output exactly;
* `torch.compile` (eager backend) produces losses identical to eager training with loadable checkpoints; merged-adapter checkpoints reproduce adapter logits;
* exact reproduction of recorded held-out record statistics (mean 0.80799, std 0.01512), leave-one-molecule-out values, RLAIF pairwise statistics (60 pairs) and constitutional statistics (10 SFT / 10 DPO), and the recorded SFT/DPO dataset sizes (139 / 80).

## 11. GPU-only capabilities

fp16 training with gradient scaling; bf16 autocast at scale; FSDP sharding; DeepSpeed ZeRO; NCCL tensor and pipeline parallelism across ranks; 8-bit base-model loading (bitsandbytes); 7B-class LoRA post-training runs (`configs/post_training/synthesis_*_qwen7b_lora.yaml`, `agent_grpo_gsm8k_qwen7b_lora.yaml`); `large`-preset pretraining (`configs/training/pretrain_large_gpu.yaml`).

## 12. Benchmark inventory

| Record | Evidence | Headline |
|---|---|---|
| training/char_small_cpu | run report | `small` 10.6M params, val loss 1.479, 49 tok/s generation (Apple M-series CPU) |
| training/fineweb_edu_large_a40 | run report | `large`, val loss 3.5834, 20.7k tok/s average (A40, bf16, compile) |
| peft/lora_qwen2_5_7b | run report | 5.05M / 7.62B trainable (0.066%) |
| post_training/sft_synthesis | raw log | loss 1.663 → 1.008, 139 pairs |
| post_training/dpo_synthesis | raw log | best margin 0.00034, 80 pairs |
| post_training/ppo_synthesis | raw log | 200 iterations completed; rewards are record statistics (pipeline evidence only) |
| post_training/ppo_synthesis_pilot | raw log | 50 iterations (pipeline evidence only) |
| rlvr/agent_grpo_gsm8k_tools | raw log | best training reward 0.5575; eval accuracy 0.15 → 0.05, tool use 0.5 → 0.9 |
| rlvr/process_reward_grpo_gsm8k | raw log | best reward 1.0659 at iteration 1 |
| rlvr/hill_climb_gsm8k | raw log | accuracy 0.125 → 0.0 |
| post_training/tabular_ppo_synthesis | raw log | training reward 0.8577; leave-one-molecule-out table |
| post_training/ai_feedback_data | raw log | 60 pairwise DPO pairs; 10 constitutional SFT/DPO records |
| decisioning/budgeted_allocation | raw log | dual pacer 7.71, sequence PPO 7.60 ± 0.12, stateless PPO 7.40 ± 0.10, heuristic 7.29; oracle 13.22; A/B rejected; DR OPE within 0.1–0.9 of truth (CPU, 3 min) |
| post_training/runs_without_retained_logs | not published | GRPO 0.823, RLAIF 0.814, STaR 0.791, SFT 0.412 — needs evidence recovery |
| distributed, quantization, inference (throughput), serving | none | no measurements recorded |

Revalidation: no record requires revalidation (no benchmark-producing computation was materially changed). Four records are marked `revalidation_recommended` because the objective is identical but a constant loss-normalisation factor (and, for process-reward GRPO, the log-prob tokenisation path) differs from the recorded run; Adam-family optimisers are invariant to constant gradient scale apart from epsilon and gradient clipping.

Held-out synthesis numbers that score dataset records are labelled policy-independent everywhere they appear. Every record carries `ledger_status` per number (CARRY_FORWARD, CARRY_FORWARD_WITH_INTERPRETATION_NOTE, NEEDS_EVIDENCE_RECOVERY, REVALIDATE, DO_NOT_USE); the full ledger with provenance is in `FORGELINE_COMPLETE_ENGINEERING_DOSSIER.md`.

## 13. Known limitations

* No multi-GPU, quantized-model accuracy/speed, or serving throughput measurements exist.
* The step trainer supports DDP for all stages and FSDP for pretraining/distillation; DeepSpeed, tensor and pipeline parallelism are strategy components for custom loops. FSDP runs do not save optimizer state or support `resume`.
* The inference engine keeps per-sequence cache tensors (block accounting, not block-indexed kernels); native sparse attention has no cached decode.
* The Python executor is process- and timeout-isolated, not a security sandbox.
* Multiple-choice benchmarks are scored generatively; offline benchmark samples verify wiring only.
* The registry is single-writer JSON; the router decides but does not dispatch shadow traffic; the HTTP server has no auth/TLS/rate limiting.
* GGUF files use the `forgeline` architecture tag and native tensor names; third-party runtimes need a matching architecture definition.
* In the synthesis domain the rule reward reads outcomes stored in each record; generated conditions do not change it, so synthesis RL rewards measure sampled records, not policies (pinned by a test). A condition-aware reward or outcome simulator is needed for policy-quality claims.
* Ray orchestration is validated on a single machine; no multi-node or GPU Ray execution has been measured. Agent-mode rollouts and HuggingFace policies are not dispatched to Ray workers.
* The dashboard reads local files only and has no authentication.

## 14. Remaining technical debt

* Batch rollouts across prompts (currently per prompt group).
* Per-token value estimation and GAE wired into the default PPO loss.
* Block-indexed paged attention kernels and cross-length batched decoding.
* Sharded (per-rank) checkpoint files and optimizer-state saving for FSDP; resume for FSDP runs.
* DeepSpeed, tensor and pipeline parallelism integrated into the step trainer; expert parallelism for MoE.
* Checkpointing the position of streaming data loaders.
* Vectorised MoE expert dispatch and fused NF4 dequantization.
* Log-likelihood scoring for multiple-choice benchmarks.
* Multi-writer registry backend and shadow-traffic dispatch in the router.
* Ray dispatch for multi-turn agent episodes and HuggingFace policies; asynchronous (off-policy) rollout buffers.
* Condition-aware synthesis reward (outcome model) so synthesis RL can be evaluated on policy outputs.

## 15. Recommended GPU validation

When hardware is available, in priority order:

1. `python -m pytest tests/hardware` on one CUDA GPU (bf16/fp16 precision, CUDA inference).
2. `torchrun --nproc_per_node=2 -m pytest tests/hardware -m multi_gpu` (FSDP wrap, 2-rank tensor parallelism).
3. Reproduce the four `revalidation_recommended` records with their manifests (`synthesis_ppo_qwen7b_lora`, `agent_grpo_gsm8k_qwen7b_lora`, process-reward GRPO with the HuggingFace backend) and replace the records with the new logs.
4. Run the synthesis GRPO and DAPO manifests and full AI-feedback / STaR loops to obtain first retained logs.
5. Measure continuous-batching throughput and latency with `measure_generation` and the HTTP server at several concurrency levels; record under `benchmarks/inference` and `benchmarks/serving`.
6. Measure NF4/QLoRA memory and Q8_0/Q4_0 accuracy against FP16 on a held-out set; record under `benchmarks/quantization`.
7. DDP and FSDP scaling (1, 2, 4, 8 GPUs) on `large` pretraining; record under `benchmarks/distributed`.
8. Run the full (non-offline) benchmark suites on trained checkpoints and gate promotions on them.
9. `python -m pytest tests/hardware -k ray` on a GPU node, then GRPO with `rollout_workers` on GPUs against a Ray cluster (`configs/orchestration/ray_cluster.yaml`), recording rollout throughput with 1/2/4 workers.
