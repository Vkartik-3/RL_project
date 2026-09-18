# Forgeline — Complete Engineering and Evidence Dossier

The permanent source of truth for what Forgeline is, what it implements, how every subsystem works, where every number comes from, which historical measurements remain valid, what was executed on which hardware, and what is safe to claim. It is written so that no one needs to rediscover anything from the code, run records or history; where a historical artifact is the evidence, it is named.

Path conventions: `src/forgeline/...` is abbreviated to the package-relative path (e.g. `training/ppo.py`). "Historical" refers to earlier internal versions of the code and to run records produced before the current architecture. Historical ledger record ids (`H01`…`H84`) refer to §30 and to `benchmarks/historical_ledger.json`.

---

## Contents

1. Executive snapshot · 2. Repository / environment · 3. Complete architecture · 4. Data system · 5. Model architectures · 6. Pretraining / SFT / distillation · 7. PEFT / LoRA / QLoRA · 8. Reward modeling · 9. DPO · 10. PPO · 11. GRPO · 12. RLVR · 13. Agent GRPO / tool use · 14. Process rewards · 15. RLAIF · 16. STaR · 17. DAPO · 18. Hill climbing and other post-training · 19. Distributed systems · 20. Ray orchestration · 21. Checkpointing / recovery · 22. Evaluation · 23. Quantization / export · 24. Inference engine · 25. Serving · 26. Model lifecycle · 27. Dashboard / observability · 28. Failure engineering · 29. Complete test evidence · 30. Complete benchmark ledger · 31. Historical carry-forward analysis · 32. Metrics requiring interpretation · 33. Metrics requiring evidence recovery · 34. Metrics requiring revalidation · 35. Bug-fix history · 36. Implementation ownership matrix · 37. Technology evidence matrix · 38. Claim bank · 39. Interview story bank · 40. Truth / boundary ledger · 41. Hardware validation matrix · 42. Future optional experiments · 43. GenAI infrastructure concept mapping · 44. Resume-safe summary · 45. Sequential decisioning under uncertainty: budgeted allocation

---

## 1. Executive snapshot

| Item | Value |
|---|---|
| Project | **Forgeline** — post-training, distributed training, evaluation, inference and model-lifecycle framework for language models (Python package `forgeline`, version 0.1.0, MIT) |
| Repository | `Meta_RL_Project/forgeline/`, branch `main`, pushed to `origin/main` |
| Commits | see `git log` |
| Size | 143 Python modules (~16.5k lines) + a 1-file dashboard page, 29 test modules / 259 tests, 47 config files, 6 examples, 7 scripts, 23 documents |
| Tests (full environment: torch 2.14, transformers 5.17, PEFT 0.21, Ray 2.58, macOS/Apple Silicon CPU) | **251 passed, 8 skipped** (~41 s); skips: 7 hardware tests (4 need CUDA, 2 need ≥2 GPUs, 1 needs the deepspeed extra) + 1 smoke test that only runs without Ray |
| Tests (clean `pip install -e ".[dev]"`, no HuggingFace/Ray) | **234 passed, 9 skipped**; Ray is never imported by the core |
| Language / framework | Python ≥ 3.10, PyTorch ≥ 2.1, NumPy, PyYAML; optional extras `training`, `huggingface`, `distributed`, `ray`, `serving`, `observability`, `chemistry`, `dev` |
| Architecture | one experiment manifest → one `RunContext` → one `Trainer` lifecycle over pluggable `PostTrainingAlgorithm`s, driven by contracts (`PolicyModel`, `RewardProvider`, `Verifier`, `Trajectory`, `DistributedStrategy`, `EvaluationResult`, `MetricsSink`, `ModelCandidate`); a native decoder-only transformer and a HuggingFace+PEFT policy behind the same surface |
| Most important capabilities | 15 training/post-training algorithms incl. PPO, GRPO, agent GRPO with tools, process-reward GRPO, DAPO, RLVR, RLAIF, STaR; DDP for every algorithm, FSDP, tensor/pipeline parallel components, DeepSpeed strategy; optional Ray rollout/reward/tool/evaluation workers with weight sync and recovery; atomic checkpoints with exact resume; NF4/QLoRA and GGUF export; continuous-batching KV-cached inference engine with paged block accounting; OpenAI-compatible server; candidate registry with fail-closed promotion gates, deterministic champion/challenger routing, rollback; read-only dashboard; a sequential-decisioning subsystem (budgeted allocation POMDP with policy-dependent dynamics, primal-dual pacer, stateless and GRU sequence PPO, OPE with IPS/SNIPS/PDIS/DR, hindsight oracle, simulated A/B with guardrails, shadow evaluation) |
| Most important measured results | 421M-parameter transformer pretrained on ~491M FineWeb-Edu tokens on one A40 to 3.5834 validation loss at ~20.7k tokens/s (H05–H11); 10.6M char-level model to 1.479 val loss on CPU (H02); Qwen2.5-7B LoRA (5.05M params) SFT loss 1.663→1.008 (H32), DPO and 200-iteration PPO executed on one GPU (H35, H43); GSM8K tool-agent GRPO: best batch reward 0.5575, tool-use rate 0.5→0.9 on held-out problems (H53, H56); process-reward GRPO best 1.0659 (H59); every synthesis "held-out reward" (0.808 etc.) is a dataset statistic, reproduced exactly and labelled as such (H39); budgeted-allocation benchmark (5 seeds × 300 episodes, CPU): dual pacer 7.71, GRU sequence PPO 7.60 ± 0.12, stateless PPO 7.40 ± 0.10, heuristic 7.29, oracle bound 13.22; DR OPE within 0.1–0.9 of simulator truth; simulated A/B rejected the sequence-PPO challenger (§45) |
| Historical metric ledger | 82 historical records (§30) + the current decisioning record (§45): 19 CARRY_FORWARD, 42 CARRY_FORWARD_WITH_INTERPRETATION_NOTE, 5 NEEDS_EVIDENCE_RECOVERY, 0 REVALIDATE, 16 DO_NOT_USE |

---

## 2. Repository / environment

| Item | Value |
|---|---|
| Path | `/Users/kartikvadhawana/Desktop/Meta_RL_Project/forgeline` |
| Python | 3.12.13 (venv `../.venv-forgeline`, Homebrew CPython) |
| PyTorch | 2.14.0 (CPU build on macOS arm64) |
| Transformers / PEFT / accelerate / tokenizers | 5.17.0 / 0.21.0 / 1.15.0 / 0.23.2 (optional `huggingface` extra) |
| Ray | 2.58.0 (optional `ray` extra; `ray[default]>=2.9`) |
| DeepSpeed | not installed here (optional `distributed` extra; strategy validates configs without it) |
| NumPy / PyYAML / pytest | 2.5.3 / 6.0.3 / 9.1.1 |
| Package layout | `src/` layout, `pyproject.toml` (setuptools), console script `forgeline = forgeline.cli.main:main`, package data `dashboard/static/*.html`, pytest markers `cuda`, `multi_gpu`, `distributed`, `slow`, `optional_dependency` |
| Installation modes | `pip install -e .` (core: torch, numpy, pyyaml); `.[dev]` adds pytest, pytest-timeout, httpx, ruff; `.[huggingface]` transformers, datasets, peft, accelerate; `.[ray]` ray[default]; `.[distributed]` deepspeed; `.[serving]` fastapi, uvicorn (not required — the built-in server is stdlib); `.[observability]` wandb, tensorboard; `.[training]` tqdm, tiktoken; `.[chemistry]` rdkit |
| Environment variables read | `RANK`, `LOCAL_RANK`, `WORLD_SIZE` (launcher detection, `distributed/runtime.py::read_launcher_env`), `MASTER_ADDR`/`MASTER_PORT` (set if absent for 1-process groups), `FORGELINE_LOG_FORMAT` (`text`|`json`), `FORGELINE_LOG_LEVEL` |
| Historical artifacts | run reports, results JSON files, statistics files, saved LoRA adapters and chart images from earlier internal versions; no notebooks, W&B exports or TensorBoard event files exist |

---

## 3. Complete architecture

```
src/forgeline/
├── core/            config.py (ModelSpec, presets, Trainer/Optimizer/Schedule/Precision/Checkpoint/Adapter/Distributed/Data configs,
│                    ExperimentManifest incl. `orchestration`), protocols.py (contracts), errors.py (27 typed errors), lifecycle.py (RunContext),
│                    registry.py, runtime.py (device/dtype/seeds/RNG state)
├── data/            tokenizers (char, tiktoken, HF), records + JSONL schemas, pretraining (memmap), streaming (shards, text glob, HF), supervised,
│                    preference, verifiable (GSM8K builder), collators, preprocessing
├── models/          transformer/ (norm, feedforward, block, model), attention/ (rotary, causal, latent, sparse), moe/ (gate, layer),
│                    generation/ (sampling, speculative), adapters/ (lora, qlora), quantization/ (nf4, gguf), policy.py (NativePolicy, HFPolicy),
│                    reward.py (SequenceRewardModel, EncoderRewardModel), loading.py
├── training/        common/ (trainer, optim, schedule, precision, advantages, logprobs); pretrain, sft, distill, reward_model, dpo, ppo, grpo,
│                    dapo, rlvr, rlaif, star (STaR + HillClimber), factory (manifest → algorithm, incl. Ray attachment)
├── rollouts/        engine (group rollouts), agent (multi-turn tool episodes), tools/ (python_executor, parser), verifiers/, rewards/
│                    (rules, composite, process, critic, learned)
├── distributed/     runtime, topology (3-D mesh), tensor_parallel, pipeline_parallel (1F1B), strategies (single/ddp/fsdp/deepspeed/tp/pp)
├── orchestration/   config (RayConfig), ray_backend (RayRolloutEngine, RayRewardPool, RayToolPool, RayEvaluator, attach_to_algorithm)  [optional]
├── checkpoints/     manager (atomic directory checkpoints, validation, resume), validation
├── evaluation/      results, quality, reward, regression, performance, inspection (weights/layers/activations/attention), suites/ (6 benchmarks)
├── inference/       requests, paged_memory, scheduler, engine, generation
├── serving/         schemas, backends, api (Router), server (stdlib HTTP + SSE)
├── deployment/      candidates (state machine), registry (JSON), promotion (gates), feature_flags, rollout (routing/replay), rollback
├── observability/   logging, metrics (sinks), events, tracing
├── dashboard/       data (file readers), server (routes), static/index.html  [optional to run]
├── domains/synthesis/  reward, constraints, prompts, features, preferences, generator, tabular_ppo
├── domains/allocation/ env (BudgetedAllocationEnv, VectorEnv), policies (threshold, dual pacer, MLP/GRU actor-critics, ε-mix), ppo (AllocationPPOAlgorithm),
│                    rollout (logged trajectories, pacing metrics), ope (IPS/SNIPS/PDIS/DR), oracle, experiment (A/B, shadow), benchmark
└── cli/main.py      presets, validate, data, train, generate, evaluate, serve, export, registry, rlaif, dashboard, allocation, synthesis-ppo
```

End-to-end flow and the code that owns each step:

| Step | Code |
|---|---|
| data | `cli data prepare|prepare-hf|shard|gsm8k|validate` → `data/preprocessing.py`, `data/streaming.py`, `data/records.py` |
| training / post-training | `ExperimentManifest.load` → `RunContext.create` → `training/factory.py::build_algorithm` → `Trainer.fit` (`training/common/trainer.py`) |
| rollout / reward | `rollouts/engine.py::RolloutEngine`, `rollouts/agent.py::AgentRolloutEngine`, `rollouts/rewards/__init__.py::build_reward_provider`, `rollouts/verifiers` |
| distributed execution | `distributed/strategies.py::build_strategy` + `cli/main.py::_apply_strategy`; Trainer broadcasts/all-reduces for DDP; FSDP wrap for pretrain/distill; optional Ray pools via `orchestration/ray_backend.py::attach_to_algorithm` |
| checkpoint | `Trainer.save_checkpoint/resume` → `checkpoints/manager.py::CheckpointManager` |
| evaluation | `cli evaluate` → `evaluation/suites`, `evaluation/quality.py`, `evaluation/reward.py`, `evaluation/regression.py` |
| quantization / export | `cli export` → `models/quantization/gguf.py::export_gguf`, `models/adapters/lora.py::merge_lora`; NF4 for QLoRA in `models/quantization/nf4.py` |
| inference | `inference/engine.py::InferenceEngine` (scheduler + paged allocator + KV caches) |
| serving | `serving/api.py::Router` + `serving/server.py::ServingServer`, `cli serve --routing` |
| lifecycle | `deployment/registry.py::CandidateRegistry`, `promotion.py::PromotionGate`, `rollout.py::RoutingPolicy`, `rollback.py`, `cli registry` |
| observability | `observability/metrics.py` sinks + `events.py`, `logging.py`, `tracing.py`; `cli dashboard` |
| sequential decisioning | `domains/allocation/*` via `allocation_ppo` in `training/factory.py` and `cli allocation {benchmark,ope,ab,shadow}` |

Contracts (`core/protocols.py`): `PolicyModel` (encode/decode/generate/logprobs/values/reference/trainable_parameters/state), `RewardProvider.score(Trajectory) → RewardResult(value, components, passed, info)`, `Verifier.verify(output, target) → RewardResult`, `Trajectory` (prompt_ids, response_ids, texts, task, segments, tool_results, truncated, old_logprobs, reward, meta), `DatasetProvider`, `MetricsSink`, `EvaluationResult(suite, metrics, n_samples, details, per_item)`, `PostTrainingAlgorithm` (`parameters`, `collect`, `loss`, `modules`, `evaluate`, `state`, `load_state`, `model_spec`, `tokenizer_meta`, `on_step_end`).

---

## 4. Data system

| Component | File | What it does |
|---|---|---|
| Tokenizers | `data/tokenizers.py::CharTokenizer` (dependency-free; `ascii()` factory), `TiktokenTokenizer` (`training` extra), `HFTokenizer` (`huggingface` extra); `resolve_tokenizer("auto|char|tiktoken[:enc]|hf:<name>")`, `save/load_tokenizer_metadata` (`meta.pkl`), `tokenizer_from_metadata` (checkpoints embed the metadata) | |
| Record schemas | `data/records.py::SFTRecord(prompt, response)`, `PreferenceRecord(prompt, chosen, rejected)`, `VerifiableTask(prompt, answer|test_code|format)`, `ProcessRecord`, `RewardRecord`; `read_jsonl(path, schema, on_error="raise|skip") → records, ReadReport`; malformed records raise `MalformedRecordError` with `file:line`; `deterministic_split(items, fraction, seed)` |
| Pretraining corpora | `data/pretraining.py::MemmapCorpus(data_dir, split, block_size)` — memory-mapped `train.bin`/`val.bin` with uint16/uint32 detection; `sample_batch` draws random windows with a torch `Generator` (per-rank seeds under DDP) |
| Streaming | `data/streaming.py::ShardedTokenDataset` (shards split by rank and DataLoader worker), `TextGlobDataset`, `HFStreamingDataset` (optional), `pack_documents`, `shard_token_file`, `write_shards_from_documents` (constant-memory writer, `flush` every N tokens), `merge_shards`, `build_streaming_loader` |
| Preprocessing | `data/preprocessing.py::prepare_text_corpus` (text → bins + meta), `prepare_hf_corpus` (HF dataset streaming → shards) |
| Task datasets | `data/supervised.py::SupervisedDataset` (prompt masked with `IGNORE_INDEX = -1`, matching the native model's `ignore_index`), `data/preference.py::PreferenceDataset`, `data/verifiable.py::VerifiableTaskDataset`, `load_prompts`, GSM8K → tool-task builder, `data/collators.py` |
| Sample data (shipped) | `data/samples/text/tiny_corpus.txt`; `supervised/arithmetic_sft.jsonl`; `preference/arithmetic_pairs.jsonl`; `verifiable/{arithmetic_tasks, tool_tasks, process_records}.jsonl`; `synthesis/trajectories_literature.jsonl` (500 nested records), `trajectories_improvable.jsonl` (500), `trajectories_yield_series.jsonl` (50), `trajectories_labeled_small.jsonl` (100 flat), `trajectories_labeled_expanded.jsonl` (500 flat) |
| Synthesis records | nested: `{molecule, cas_number, procedure_id, parameters{temperature_celsius, time_hours, catalyst_loading_M, solvent_ratio_ml_mmol, hazards}, outcomes{yield, selectivity, safety_risk, steps, molar_mass_product}}`; flat: `{molecule, yield, selectivity, safety_risk, steps}`; `domains/synthesis/reward.py::extract_outcomes` normalises both |
| Data facts | literature file: 5 molecules × 100 records in order Aspirin, Ibuprofen, Paracetamol, Naproxen, Ketoprofen; yields 0.671–1.037 (8 records exceed 1.0 — a data-quality defect, H82); improvable file: yields 0.25–0.949, mean 0.360 |

Tests: `tests/unit/test_data.py` (9), `tests/failure/test_failure_modes.py::test_dataset_failures`.

---

## 5. Model architectures

### 5.1 Native transformer (`models/transformer/model.py::TransformerLM`)

Decoder-only, pre-norm RMSNorm (`models/transformer/norm.py`), tied input/output embeddings (`wte.weight = lm_head.weight`), learned absolute positions when `use_rope: false`. `forward(idx, targets)` returns `(logits, loss)`; loss = token cross-entropy (`ignore_index=-1`) + Σ MoE auxiliary losses + `mtp_loss_weight × mean` of multi-token-prediction head losses. `forward_hidden` exposes final hidden states (used by value heads and reward models; `_last_hidden` detached). `prefill(idx) → (logits, cache)` and `step(next_ids, cache, position)` implement KV-cached decoding; `generate` samples with the shared sampling filters. `enable_gradient_checkpointing()` wraps blocks with `torch.utils.checkpoint`.

Blocks (`models/transformer/block.py`) select attention by spec: `CausalSelfAttention` (`models/attention/causal.py`: MHA/GQA with `n_rep` KV repetition, SDPA fast path when no cache/window/cap, sliding window with optional alternating layers, tanh logit soft-cap, `(k, v)` cache), `MultiHeadLatentAttention` (`models/attention/latent.py`: low-rank KV compression `kv_lora_rank`, decoupled RoPE dims `qk_rope_head_dim`, caches `(c_kv, k_rope)`, naive and absorbed paths), `NativeSparseAttention` (`models/attention/sparse.py`: compressed + top-k selected + sliding-window branches fused by learned gates; no cached decode path). FFN: SwiGLU or GELU (`feedforward.py`) or `MoELayer` (`models/moe/layer.py`) with `ExpertGate` (`models/moe/gate.py`: softmax/sigmoid scoring, `route_scale`, aux-loss-free bias updated at `bias_update_speed`, group-limited routing `n_expert_groups`/`n_limited_groups`, shared experts, `n_dense_layers` dense prefix). RoPE (`models/attention/rotary.py::RotaryEmbedding`) supports `linear` and `yarn` scaling.

Presets (`core/config.py::MODEL_PRESETS`, vocab supplied at build time):

| Preset | Layers | Heads (Q/KV) | d_model | Context | Notable |
|---|---|---|---|---|---|
| tiny | 2 | 2/2 | 64 | 64 | CPU test model (~0.1M params with a 100-token vocab) |
| small | 6 | 6/6 | 384 | 256 | 10.65M with vocab 65 (H01) |
| medium | 12 | 12/4 | 768 | 512 | GQA |
| large | 24 | 16/4 | 1024 | 2048 | 421.15M with vocab 151,643 (H05) |
| xl | 24 | 16/8 | 2048 | 4096 | |
| moe_32l | 32 | 32/8 | 4096 | 8192 | MoE 8 experts, 2 active |
| latent_moe_27l | 27 | 16/16 | 2048 | 4096 | MLA, MoE 64 experts/6 active + 2 shared, 3 MTP heads |
| sliding_window_32l | 32 | 32/8 | 4096 | 8192 | window 4096 |
| alternating_28l | 28 | 16/4 | 2304 | 8192 | alternating window 4096, logit cap 50 |

Generation (`models/generation/sampling.py`): `apply_sampling_filters(logits, temperature, top_k, top_p, min_p, repetition_penalty, past_tokens)`; `temperature == 0` ⇒ greedy. `models/generation/speculative.py::SpeculativeGenerator` (draft model proposes `k` tokens, target verifies in one pass, reports acceptance rate) and `MTPSpeculativeGenerator` (self-drafting from MTP heads).

### 5.2 Policies (`models/policy.py`)

`NativePolicy(model, tokenizer, adapter=AdapterConfig|None, reference_model=None, value_head=False)`: applies LoRA/QLoRA when `adapter.method` is set; `reference()` context = adapters disabled (`set_lora_enabled(False)`) when adapters exist, else a frozen deep copy created by `with_frozen_reference()`; `logprobs(prompt_ids, response_ids)` computes per-token log p(label) via `token_logprobs_from_logits` (cross-entropy trick, never materialises log-softmax); `logprobs_and_values` adds `ValueHead` (linear on final hidden) outputs; `state_for_checkpoint` returns model/adapter/value-head dicts.

`HFPolicy(model_name, lora_r=8, lora_alpha=16, lora_dropout=0.05, target_modules=[q,k,v,o]_proj, load_in_8bit=False, device_map="auto", torch_dtype=None, gradient_checkpointing=True, value_head=False, use_lora=True)`: `AutoModelForCausalLM` + `peft.get_peft_model(LoraConfig(...))`; `reference()` = `model.disable_adapter()`; left-padded tokenizer; `state_for_checkpoint` stores only `lora_` tensors when adapters exist. Same surface as `NativePolicy` so every algorithm is backend-agnostic. Raises `OptionalDependencyError` without transformers/peft (smoke test).

Tests: `tests/unit/test_models.py` (20 incl. every variant forward/backward, cached = uncached, MLA cached, sampling filters, speculative generators), `tests/integration/test_huggingface_backend.py` (7, tiny local Llama/BERT built in the test, see §21 note in the HF section below).

---

## 6. Pretraining / SFT / distillation

**Pretraining** (`training/pretrain.py::PretrainAlgorithm`): `collect` samples a `[B, T]` window batch from `MemmapCorpus` (or a streaming loader); `loss` = `model(x, y)[1]` (token CE + aux + MTP). Optimizer (`training/common/optim.py::build_optimizer`): AdamW with decay only on tensors with `dim ≥ 2` when `decay_only_matrices` (biases/norms undecayed), `fused` on CUDA. Schedules (`training/common/schedule.py::learning_rate_at`): cosine with linear warmup to `decay_steps`, `wsd` (warmup–stable–decay, `wsd_stable_fraction`), `constant`, `linear_decay`; the cosine formula is pinned by `tests/unit/test_benchmark_semantics.py::test_cosine_schedule_reference_formula`. Precision (`training/common/precision.py::Precision`): bf16/fp16 autocast, GradScaler for fp16, `unscale` before clipping. `torch.compile` via `trainer.compile_model` (in-place `nn.Module.compile`, `compile_backend`), restored in this audit.

Evidence: H01–H09 (CPU char model 1.479; A40 421M model 3.5834, 20.7k tok/s). Semantics preserved: same objective, optimizer grouping, schedule, autocast; `configs/training/pretrain_large_gpu.yaml` reproduces the A40 configuration (lr 1.5e-4, warmup 200, 30k steps, batch 8 × 2048, weight decay 0.1, bf16, compile).

**SFT** (`training/sft.py::SFTAlgorithm`, `SFTConfig(batch_size, max_length, mask_prompt=True)`): prompt tokens and padding set to `IGNORE_INDEX`; token-mean cross-entropy over response tokens via `policy.token_logits`; full fine-tuning or adapters decided by the policy. HF path used for the recorded Qwen SFT (H31–H33: 139 examples, loss 1.663 → 1.008).

**Distillation** (`training/distill.py::DistillationAlgorithm`): `L = α·T²·KL(softmax(t/T) ‖ softmax(s/T)) + (1−α)·CE(s, y)` (`distillation_loss`, `batchmean`), optional `FeatureDistiller` (linear projections of student hidden states to teacher dims, MSE, `feature_weight`). Teacher loads from a Forgeline checkpoint (`algorithm_params.teacher_checkpoint`); vocabularies must match. No historical measurement.

Tests: `tests/integration/test_training_lifecycle.py::test_pretrain_loss_decreases, test_sft[none|lora|qlora], test_distillation`; `test_checkpoint_resume.py::test_compile_model_matches_eager_and_checkpoints_load`.

---

## 7. PEFT / LoRA / QLoRA

| Aspect | Native (`models/adapters/lora.py`, `qlora.py`, `models/quantization/nf4.py`) | HuggingFace (`models/policy.py::HFPolicy`) |
|---|---|---|
| Injection | `apply_lora(model, rank, alpha, dropout, target_modules)` replaces matching `nn.Linear` children with `LoRALinear(original, A, B)`; default targets: attention/FFN projections named in `DEFAULT_TARGET_MODULES`; all other params frozen | `peft.get_peft_model` with `LoraConfig(r, lora_alpha, lora_dropout, target_modules=[q_proj,k_proj,v_proj,o_proj], bias="none", task_type CAUSAL_LM)` |
| Forward | `y = W x + (x A B)·α/r`, `A` Kaiming-initialised, `B` zero ⇒ identity at init (tested) | PEFT |
| Enable/disable | `set_lora_enabled(model, bool)` — disabled adapters = reference policy | `model.disable_adapter()` |
| Save/load/merge | `lora_state_dict`, `save_lora`, `load_lora_state_dict`, `merge_lora` (adds `(A@B)ᵀ·scaling` into the base weight); `trainer.adapter.merge_on_save` writes a `final_merged` checkpoint (restored in this audit) | adapter tensors (`lora_` keys) saved in Forgeline checkpoints; `scripts/export_adapter.py` exports/publishes |
| Trainable-parameter accounting | `trainable_parameter_report` | PEFT; verified 5,046,272 for Qwen2.5-7B r=8 (H30): 28 layers × 8 × [(3584+3584) + 2×(3584+512) + (3584+3584)] |
| QLoRA | `apply_qlora`: base weights → `NF4Linear` (packed NF4 + per-block absmax, dequantised on the fly) + LoRA; `quantize_nf4` block size 64, 16-level NormalFloat table (pinned by `test_nf4_table_and_block_size_unchanged`), `quantization_report` memory estimate | `load_in_8bit=True` passthrough (bitsandbytes, GPU only) — not exercised here |
| Tests | `tests/unit/test_adapters_quantization.py` (identity at init, merge, save/load, NF4 round trip, QLoRA gradients), `test_training_lifecycle.py::test_sft[lora|qlora]` (frozen base untouched), `test_checkpoint_resume.py::test_auto_resume_from_latest_and_adapter_state, test_merge_on_save_writes_plain_model_checkpoint` | `test_huggingface_backend.py` (LoRA on a tiny local Llama; base weights unchanged after training; adapter checkpoint round trip) |

Historical adapter artifacts (saved `adapter_model.safetensors` files, 20.2 MB each, fp32): SFT best, DPO best, PPO checkpoints at iterations 50/100/150/200 and best. Derived statistics (this audit): ‖ΔW‖_F = 1.727 (SFT), 1.765 (DPO), 0.118 → 0.228 across PPO iterations 50→200; DPO−SFT displacement 0.256; PPO adapters start from a fresh LoRA (A-matrix cosine to SFT ≈ 0).

---

## 8. Reward modeling

| Model | File | Objective | Use |
|---|---|---|---|
| `SequenceRewardModel` | `models/reward.py` | native transformer backbone + scalar head on the last position; trained by `training/reward_model.py::RewardModelAlgorithm` with `bradley_terry_loss = −log σ(r_chosen − r_rejected)`; `RewardModelConfig(batch_size, freeze_backbone, max_length)` | `rollouts/rewards/learned.py::RewardModelProvider` scores prompt+response tokens for RL |
| `EncoderRewardModel` | `models/reward.py` (huggingface extra) | Transformer encoder (e.g. BERT) + head; `fit_regression` (MSE on scalar targets) or `fit_preference` (Bradley-Terry) | `BlendedReward(neural, rule, blend)` = `blend·neural + (1−blend)·rule` clipped to [0, 1], rule-only fallback when untrained (the recorded blends were 0.6 and 0.7) |
| Rule rewards | `rollouts/rewards/rules.py::LengthReward, TextFormatReward, RepetitionReward, ConstantReward`; `domains/synthesis/reward.py::SynthesisRuleReward` | deterministic | baselines, tests, synthesis task |
| Composition | `rollouts/rewards/composite.py::CompositeReward([(provider, weight)], clip)`, `VerifierReward`, `TagFormatReward`, `CallableReward`; `score_trajectories` (uses `score_many` for batch-capable providers such as Ray pools) | non-finite values raise `InvalidRewardError` naming the provider | |

Synthesis rule reward (`rule_score`): `0.40·yield + 0.30·selectivity + 0.20·(1 − safety_risk) + 0.10/(1 + steps/10)`, clipped to [0, 1]. An earlier efficiency term `0.10·(1 − steps/10)` produced the first recorded runs; `tests/unit/test_benchmark_semantics.py::test_legacy_efficiency_term_reproduces_early_recorded_values` reproduces those numbers (H74, H75). Nested `outcomes` take precedence over decoded conditions, so for nested records the reward is policy-independent (`test_synthesis_rule_reward_ignores_generated_text_for_nested_records`) — see §32.

Tests: `test_training_lifecycle.py::test_reward_model_learns_ranking`, `test_huggingface_backend.py::test_encoder_reward_models`, `tests/unit/test_rollouts_rewards.py`.

---

## 9. DPO

File: `training/dpo.py::DPOAlgorithm`, `DPOConfig(beta=0.1, batch_size=4, logprob_reduction="sum"|"mean", reference_free=False, label_smoothing=0.0, seed)`.

Objective: with sequence scores `s(y|x) = Σ_t log π(y_t|x, y_<t)` (sum) or the token mean (mean, length-normalised; the recorded 7B configuration),

`L = −log σ( β[s_π(y_w) − s_ref(y_w)] − β[s_π(y_l) − s_ref(y_l)] )`, batch-averaged (`dpo_loss`), with label smoothing `−(1−ε) log σ(z) − ε log σ(−z)` when `ε > 0`; `reference_free=True` drops the reference terms (used by RLAIF self-training rounds). Metrics: `reward_margin`, `chosen_reward`, `rejected_reward`, `preference_accuracy`.

Data flow: `PreferenceDataset` tokenises `{prompt, chosen, rejected}` (`max_prompt`, `max_response`); `collect` draws a seeded batch; `loss` computes policy scores for chosen/rejected, reference scores inside `policy.reference()` (adapters disabled, or frozen copy) under `no_grad`. Reference model: adapters disabled or `with_frozen_reference()`. Checkpoints: `policy.state_for_checkpoint()`. Evaluation: loss/accuracy/margin on up to 32 examples.

Bug history: the original native DPO implementation called an undefined helper and could not execute (§35 B1). The HuggingFace DPO ran; its length-normalised variant is `logprob_reduction: mean`, pinned by `test_dpo_mean_reduction_reference`.

Historical runs: H34–H38 (80 pairs, β 0.1, 2 epochs; loss 0.6929 → 0.6919; best epoch margin 0.00034; adapter displacement 0.256 from SFT). Interpretation: the preference signal was weak (loss ≈ ln 2); the held-out 0.808 attached to this run is a record statistic (H39).

Tests: `test_training_lifecycle.py::test_dpo_increases_preference_margin[sum|mean]` (margin rises to 100% accuracy on a toy set), `tests/unit/test_training_utils.py::test_objectives`, `test_benchmark_semantics.py::test_dpo_mean_reduction_reference`, HF backend DPO on the tiny Llama.

---

## 10. PPO

File: `training/ppo.py::PPOAlgorithm`, `PPOConfig(prompts_per_step=4, samples_per_prompt=1, ppo_epochs=3, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01, kl_coef=0.1, ratio_level="sequence_mean"|"token", kl_mode="monitor"|"penalty"|"reward", entropy_mode="sampled_logprob"|"full", normalize_advantages=True)`. Requires `value_head=True`.

Collection (`collect`): sample `prompts_per_step` prompts → `RolloutEngine.rollout` (`samples_per_prompt` completions each) → `score_trajectories(reward)` → left-pad prompts / right-pad responses into `[N, P]`, `[N, R]`, mask → `old_lp, old_values = policy.logprobs_and_values` → `ref_lp` under `policy.reference()`; with `kl_mode="reward"` the per-sequence `mean(ref − old)` is subtracted from rewards.

Objective (`loss`, repeated `ppo_epochs` times on the same batch, averaged):
* advantages `A = r − V_old`, normalised across the batch (`normalize_advantages`);
* `sequence_mean`: `ρ = exp(mean_t(log π_new − log π_old))`, `L_pg = −mean_i min(ρ_i A_i, clip(ρ_i, 1−ε, 1+ε) A_i)` (`ppo_clipped_objective`); `token`: per-token ratios with token-level clipping, masked mean;
* `L_v = MSE(V(x, y), r)`; entropy `H = −mean(log π(y_t))` over sampled tokens (or exact full-distribution entropy);
* KL: `monitor` (recorded configuration) logs `mean(old_lp − ref_lp)` detached; `penalty` adds `kl_coef · mean(ref − new)` clamped at 0; total `L = L_pg + value_coef·L_v − entropy_coef·H (+ kl_coef·KL)`.
* GAE utility `training/common/advantages.py::compute_gae(rewards, values, γ, λ)` exists for per-token rewards (not used by the default sequence-level loss).

Reference: adapters disabled or frozen copy. Checkpoints: model/adapter + value head. Evaluation: none built in beyond the collected stats (`reward_mean`, `reward_std`, `response_len`, `policy_loss`, `value_loss`, `kl`, `entropy`, `advantage_mean`).

Ratio correctness: the original native PPO/GRPO steps computed `new_lp = log_softmax(logits).mean()` over *all vocabulary entries and positions* and took `ratio = exp(new_lp − old_lp.mean())`, i.e. a ratio of average log-probabilities of the whole distribution, not of the sampled tokens (§35 B2). Forgeline gathers the sampled tokens' log-probs (`token_logprobs_from_logits`) and forms the ratio per sequence (`test_sequence_mean_ppo_ratio_reference`).

Historical runs (7B value-head PPO, `configs/post_training/synthesis_ppo_qwen7b_lora.yaml`): H43–H52. What the run proves: 200 iterations of generation + value head + clipped update + adapter-disabled reference on one GPU, adapters saved every 50 iterations, adapter norm growing 0.118 → 0.228. What it does not prove: reward improvement — the reward is a record statistic (§32). Normalisation difference vs the recorded run: constant factor 0.75 (§34), classified as carry-forward with note, not revalidate.

Tests: `test_training_lifecycle.py::test_ppo` (3 configurations), `test_training_utils.py::test_gae_and_group_advantages, test_objectives`, `test_benchmark_semantics.py::test_sequence_mean_ppo_ratio_reference`, `test_huggingface_backend.py::test_synthesis_pipelines_on_hf_backend[ppo]`, `test_ddp_two_process.py` (GRPO under DDP shares the same trainer path).

---

## 11. GRPO

File: `training/grpo.py::GRPOAlgorithm`, `GRPOConfig(group_size=4, prompts_per_step=2, clip_ratio=0.2, kl_coef=0.04, objective="clipped"|"reinforce", ratio_level="sequence_mean"|"token", kl_estimator="logprob_diff"|"k3", advantage_eps=1e-8, skip_zero_variance_groups=False, agent=False)`.

Grouping and advantages: for each of `prompts_per_step` prompts, `G` completions; `A_g = (r_g − mean(r)) / (std(r) + ε)` (`group_relative_advantages`; `clamp_std=True` variant used by DAPO). No value model.

Objectives:
* `clipped`: `ρ = exp(mean_t(log π_new − log π_old))` per sequence (or per token), `L = mean_g[−min(ρA, clip(ρ)A)] + β·KL`, with `KL = clamp(mean_g mean_t(log π_ref − log π_new), 0)` for `logprob_diff` (clamp *after* averaging over the group — the recorded configuration) or the k3 estimator `exp(Δ) − Δ − 1`;
* `reinforce`: `L = Σ_g −A_g · Σ_t log π(y_t)` (no clipping, no KL; the recorded process-reward configuration).
Loss over groups is averaged (`total / n_used`); skipped zero-variance groups contribute nothing (a zero loss with a grad path is returned if all are skipped).

Agent mode (`agent=True`): rollouts through `AgentRolloutEngine`; trajectory log-prob = average over model segments of each segment's mean token log-prob conditioned on the full context including tool results (§13); ratio and KL are computed on that scalar per trajectory: `L = mean_g[−min(ρA, clip(ρ)A)] + β·mean_g clamp(ref − new, 0)`.

Evaluation: greedy rollout on up to 8 prompts → `reward`, `pass_rate`. Checkpoints via the policy. Historical runs: H53–H58 (agent GRPO on GSM8K, best batch reward 0.5575 at iteration 180, greedy accuracy 3/20 → 1/20, tool-use rate 0.5 → 0.9), H59–H62 (process-reward GRPO), H66 (synthesis GRPO 0.823 — no artifact). Normalisation difference vs recorded runs: 0.5 (agent, with 2 problems), B (process reward); see §34.

Tests: `test_training_lifecycle.py::test_grpo` (3 configurations incl. reinforce and k3), `test_agent_grpo_runs_tools`, `test_ddp_two_process.py::test_ddp_replicas_stay_identical` (GRPO under 2-process DDP), `test_ray_orchestration.py::test_grpo_trainer_with_ray_rollout_and_reward_workers`, HF backend GRPO and agent GRPO.

---

## 12. RLVR

RLVR in Forgeline is GRPO or DAPO driven by deterministic verifiers instead of learned rewards (`training/rlvr.py`). `RLVRConfig(task_type="math"|"code"|"exact", optimizer="dapo"|"grpo", require_format=False, format_spec="cot"|"steps"|"json", correctness_weight=0.7, format_weight=0.3, verifier_params, grpo=GRPOConfig, dapo=DAPOConfig)`. `build_verifiable_reward` composes `VerifierReward(task verifier)` and optionally `VerifierReward(FormatVerifier)` into a `CompositeReward`; `build_rlvr_algorithm` returns the chosen optimiser; `evaluate_pass_rate` greedy-decodes every task once and reports pass rate, mean reward and malformed-output rate.

Verifiers (`rollouts/verifiers/__init__.py`), each returning `RewardResult(value ∈ [−1, 1], passed, info)` and wrapped by `_safe` so exceptions become failed results (never abort a batch):
* `MathVerifier` — `extract_final_answer` (`\boxed{}`, `#### x`, "the answer is", last number) + `normalize_answer` (commas, trailing zeros); `ExactMatchVerifier`;
* `TaggedAnswerVerifier` — 1.0 for a correct `<final_answer>`, 0.1 partial credit when a `<tool_call>` and `<tool_result>` are present, else 0.0; `tag_format_reward` adds 0.02 per `<think>/<tool_call>/<final_answer>` tag (max 0.06) — the reward schedule of the recorded agent-GRPO run (`test_tagged_answer_reward_schedule`);
* `CodeExecutionVerifier` — extracts code, appends `test_code`, runs it through the sandboxed executor with a timeout;
* `FormatVerifier` — `cot` (think blocks), `steps` (Step 1/2…), `json` (parseable).

Flow: prompt → `RolloutEngine`/`AgentRolloutEngine` → completion → verifier(s) → `RewardResult` → `Trajectory.reward` → group advantages → policy update. Manifests: `configs/post_training/rlvr_math_tiny_cpu.yaml`, `grpo_tiny_cpu.yaml`, `grpo_agent_tools_tiny_cpu.yaml`, `grpo_process_reward_tiny_cpu.yaml`, `agent_grpo_gsm8k_qwen7b_lora.yaml`; `examples/02_rlvr_math_verifier.py`.

Tests: `tests/unit/test_rollouts_rewards.py` (math/exact/tagged/format verifiers, code verifier + executor, parser, composite/builder), `test_training_lifecycle.py::test_rlvr_with_format_verifier`, `tests/failure/test_failure_modes.py::test_invalid_reward_and_verifier_exception, test_tool_execution_failure_is_injected`.

---

## 13. Agent GRPO / tool use

Trajectory representation: `Trajectory(prompt_ids, response_ids = tokens of the concatenated model segments, prompt_text, response_text, task, segments=[model text per turn], tool_results=[injected <tool_result> blocks], meta{n_turns, n_tool_calls})`.

Episode (`rollouts/agent.py::AgentRolloutEngine.run_episode`): context = `build_agent_prompt(problem)` (system prompt describing `<think>`, `<tool_call>{"name","args"}</tool_call>`, `<final_answer>`); loop up to `max_turns + 1`: generate ≤ `max_new_tokens` from the last `max_context_tokens` of the context → `parse_tagged_output` (`rollouts/tools/parser.py`: regex-extracted tool call parsed as JSON `{name, args}`, final answer, thinking, tool results) → if a final answer: stop; if a known tool call: run the tool, wrap the output as `\n<tool_result>\n…\n</tool_result>\n`, append segment + result to the context, continue; otherwise stop. Tool errors become `ERROR: …` results (`ToolExecutionError` text), never exceptions.

Tool schema and executor: tools are `{name: callable(args: dict) → str}` (`AgentRolloutConfig.tools`, default `python_executor`). `rollouts/tools/python_executor.py::execute_python` runs `sys.executable -I -c <code>` in a fresh isolated-mode subprocess with `timeout` (5 s default) and returns `ExecutionResult(stdout, stderr, timed_out, error)`; `output` caps stdout at 500 chars and reports timeouts/errors as `ERROR:` strings. Not a security sandbox (process + timeout isolation only).

Log-probs: `_segment_logprob(context, segment)` = mean per-token log-prob of the segment given the context (truncated to the model's block size); `trajectory_logprob` = mean over segments, contexts including earlier tool results, so tool outputs are conditioned on but never differentiated through. GRPO agent mode uses this scalar for old/new/reference log-probs (§11).

Rewards: `{type: tagged, format_bonus: true}` → `TaggedAnswerVerifier` + `TagFormatReward`; `{type: critic}` (`rollouts/rewards/critic.py::CriticReward`, heuristic or LLM judge via `critic_fn`) composable with weights — `configs/post_training/grpo_agent_critic_tiny_cpu.yaml` reproduces the generator+critic design (`combined = verifiable + 0.3·critic`) whose original trainer crashed (§35 B5). Invalid responses: no answer → 0.0 (or 0.1 with tool syntax); unparsable tool JSON → treated as no tool call.

Metrics: `reward_mean`, `reward_std`, `group_std`, `groups_skipped`, `response_len`, `tool_calls`; evaluation `AgentAccuracySuite` (`evaluation/quality.py`) reports accuracy and tool-use rate on tagged-answer tasks.

Historical evidence (`agent_grpo_results.json`, 200 iterations, Qwen2.5-7B LoRA, GSM8K tool tasks): H53–H57; the run's own greedy evaluation went 0.15 → 0.05 accuracy and 0.5 → 0.9 tool use on 20 problems. Tests: `test_training_lifecycle.py::test_agent_rollout_executes_tool_calls, test_agent_grpo_runs_tools`; `examples/03_agent_tool_rollout.py`; `test_huggingface_backend.py::test_hf_agent_grpo_and_checkpoint_roundtrip`.

---

## 14. Process rewards (PRM)

File: `rollouts/rewards/process.py`. `parse_steps(response)` splits chain-of-thought into steps (numbered lines, sentences; short fragments merged); `score_step`: 0.15 for a code block that executes with output or an arithmetic assignment whose stated result matches `safe_eval_arithmetic` (AST-restricted, no builtins), 0.10 for parsable arithmetic without a checkable claim, 0.05 for code that failed, 0 otherwise (`score_step("x = 3 * 24 = 73") == 0.10` pinned). `compute_process_reward(response, ground_truth, γ=0.9, step_weight)`: `total = step_weight · Σ_t γ^(T−1−t) r_t + final`, where `final` = `TaggedAnswerVerifier` value; `ProcessReward` wraps it as a provider (`passed` = final ≥ 1). Rewards can exceed 1.0 by construction.

Used with GRPO `objective: reinforce` (`configs/post_training/grpo_process_reward_tiny_cpu.yaml`) to mirror the recorded run (`L = −A·Σ_t log π`). Historical: H59–H62 (best 1.0659 at iteration 1, final 0.7822; no improvement across 30 iterations; a historical change described as a step-reward-scaling fix altered only device placement and a dtype keyword). Tests: `test_rollouts_rewards.py::test_process_reward`, `test_benchmark_semantics.py::test_process_reward_reference`.

---

## 15. RLAIF / AI feedback

File: `training/rlaif.py`.

* **Self-judge rounds** — `RLAIFTrainer(policy, problems, RLAIFConfig(num_rounds=3, candidates_per_prompt=4, pair_gap=2.0, dpo_epochs=1, dpo_beta=0.1, learning_rate=5e-6, max_new_tokens, judge_max_tokens, temperature, num_problems))`: per round, sample K candidates per prompt, score each with the policy's own judgement (`parse_judge_score`: `Score: X/10` → X/10; `judge_fn` overridable), form (chosen, rejected) pairs whose score gap ≥ `pair_gap/10`, train reference-free DPO on the pairs; reports per-round pair counts and DPO loss (`best_loss`).
* **Pairwise judge** — `run_pairwise_rlaif(prompts, generator, judge, candidates=4)`: all C(4,2)=6 comparisons per prompt → DPO records; `rule_generator`/`rule_judge` provide a deterministic demo; `make_policy_generator`/`make_policy_pairwise_judge` wrap a checkpoint. CLI `forgeline rlaif pairwise`. Reproduces H64 exactly (10 prompts → 60 pairs, chosen 312 / rejected 194 chars).
* **Constitutional critique → revise** — `run_constitutional(prompts, model, ConstitutionalConfig(constitution=[5 principles], n_principles_per_example=2, seed=42))`: for each prompt, initial response → for each sampled principle: critique then revision → SFT record (final revision) + DPO record (revised vs initial); `score_critique_quality`. CLI `forgeline rlaif constitutional`. Reproduces H65 (10 SFT + 10 DPO, +53 chars).

Historical H67 (RLAIF 0.814) has no artifact (§33). Tests: `test_cli_pipeline.py::test_rlaif_and_synthesis_commands, test_star_and_rlaif_trainers_run`.

---

## 16. STaR

`training/star.py::STaRTrainer(policy, problems, STaRConfig(num_rounds=3, samples_per_problem=8, sft_epochs=1, learning_rate=5e-6, max_new_tokens=256, num_problems=20, temperature=0.9, eval_fraction=0.1))`: round 0 evaluates baseline accuracy on the held-out fraction; each round samples rationales (`star_prompt`), keeps those whose `extract_star_answer` matches (`answers_match`), runs SFT (`SFTAlgorithm`, token-mean CE) on the correct rationales, re-evaluates, keeps the best; history records accuracy, correct-rationale count, yield rate and SFT loss. Historical H68 (0.791) has no artifact; the trainer reports accuracy, not reward. Test: `test_cli_pipeline.py::test_star_and_rlaif_trainers_run`.

---

## 17. DAPO

`training/dapo.py::DAPOAlgorithm`, `DAPOConfig(group_size=4, prompts_per_step=2, clip_eps_low=0.2, clip_eps_high=0.28, kl_coef=0.04, entropy_coef=0.0, dynamic_sampling=True, overlong_penalty=−1.0, max_response_len=None)`.

Relative to GRPO: (1) clip-higher — token ratio clipped to `[1−ε_low, 1+ε_high]` (`dapo_token_objective`); (2) dynamic sampling — groups with zero reward variance are skipped (`groups_skipped`, `active_groups` metrics); (3) token-level normalisation — per-token surrogate terms summed over every response and divided by the total token count (long responses not under-weighted); (4) overlong shaping — responses hitting the budget get `overlong_penalty · max(0, len/max_len − 1)` and are marked truncated; (5) optional entropy bonus from full-distribution logits. KL term `Σ_t (log π_new − log π_ref)` on sampled tokens, weighted `kl_coef`, same token normalisation. Advantages use `clamp_std=True`.

No historical run (the earlier DAPO script was never executed). Tests: `test_training_lifecycle.py::test_dapo_dynamic_sampling_skips_constant_reward`, `test_training_utils.py::test_objectives`, HF backend DAPO, `examples/02_rlvr_math_verifier.py` (DAPO with math + format verifiers).

---

## 18. Hill climbing and other post-training

* **Hill climbing** — `training/star.py::HillClimber(policy, tasks, HillClimbConfig(num_rounds=5, rollouts_per_problem=4, reward_threshold=0.5, top_k_per_problem=2, grpo_steps_per_round=50, grpo_group_size=4, grpo_learning_rate=5e-6, grpo_clip=0.2, grpo_kl_coef=0.04, grpo_prompts_per_step=2, max_new_tokens=200, eval_problems=20, seed=42))`: each round rejection-samples tool-use trajectories above the threshold into the task set (top-k per problem), runs agent GRPO for `grpo_steps_per_round`, evaluates accuracy/tool use, checkpoints on improvement. Historical H63: the recorded run's threshold filtered almost everything and accuracy fell from 0.125 to 0 — a negative result retained as such.
* **Tabular actor-critic PPO** — `domains/synthesis/tabular_ppo.py::TabularPPOTrainer` on `encode_trajectory` 128-d states (8 normalised chemistry features + 100-bit Morgan fingerprint via RDKit when available, zero-padded): shared 128→256→256 MLP, 32-way softmax actor, critic; loss = clipped surrogate + 0.5·value MSE − 0.01·entropy with the ratio against detached log-probs of the same forward pass (ratio 1 on the first pass, as in the recorded baseline where the policy loss was ~1e-7). CLI `forgeline synthesis-ppo`. Historical H71–H81 are dataset statistics.
* **Synthesis data tooling** — `domains/synthesis/preferences.py` (`build_preference_pairs` per molecule, top-half vs bottom-half by yield, seeded; `build_sft_records` with a yield threshold; reproduces 80 pairs / 139 SFT examples), `prompts.py` (prompt template, JSON condition codec with clipping to `constraints.py` ranges and a validity flag), `generator.py` (simulated and yield-series records), `examples/05_synthesis_generalization.py` (leave-one-molecule-out).

---

## 19. Distributed systems

For each mechanism: implementation → current local validation → historical measured execution → limitations. "Historical" refers to the recorded runs; every retained run recorded `world_size = 1`, so no multi-GPU execution exists historically either.

| Mechanism | Implementation (files, classes) | Topology / backend / launch | State sync & communication | Checkpoint interaction | Failure handling | Current local validation | Historical measured execution | Limitations |
|---|---|---|---|---|---|---|---|---|
| **DDP** | `distributed/strategies.py::DDPStrategy`; `training/common/trainer.py::Trainer._broadcast_parameters`, `_all_reduce_gradients`; `cli/main.py::_apply_strategy` (per-rank seeds/RNG for data sampling and rollouts) | one process per rank via `torchrun`; NCCL on CUDA, gloo on CPU; `read_launcher_env` (RANK/WORLD_SIZE) | rank-0 parameter broadcast at start; `all_reduce(SUM)/world` on every trainable gradient after accumulation, before clipping — applied to every algorithm including RL loops (rollouts differ per rank, gradients averaged) | rank 0 writes checkpoints; all ranks hold identical parameters | misconfiguration raises `DistributedConfigError`; process-group init failures propagate | `tests/integration/test_ddp_two_process.py`: two real CPU processes (gloo) run pretraining and GRPO; replicas stay bit-identical after updates while ranks sample different batches; `test_distributed_gloo.py::test_ddp_strategy_single_process` | none (an earlier torchrun launch path existed; all retained runs world_size 1) | no NCCL execution here; gradient bucketing is per-tensor (no fusion) |
| **FSDP** | `FSDPStrategy.wrap_model` (`FullyShardedDataParallel`, size-based auto-wrap `fsdp_min_params_to_wrap`, `MixedPrecision`, `FULL_SHARD`), `Trainer._model_state` gathers `FULL_STATE_DICT` (`offload_to_cpu`, `rank0_only`), `strategy.clip_grad_norm` | CUDA only; torchrun | parameter/gradient/optimizer sharding by PyTorch FSDP | full state gathered collectively, written by rank 0; optimizer state not saved; `resume` refused (`checkpoint_path` initialisation instead) | `DistributedConfigError` without CUDA | configuration validation on CPU (`tests/unit/test_distributed.py`, failure tests); execution test `tests/hardware/test_hardware.py::test_fsdp_wrap_under_torchrun` **skipped (needs ≥2 GPUs)** | none (an earlier FSDP wrap path existed; never used in the recorded single-GPU runs) | step trainer supports pretrain/distill only; no sharded checkpoint files |
| **DeepSpeed** | `DeepSpeedStrategy` (`load_config`, `initialize` → engine), `configs/distributed/deepspeed_zero2.json` | `deepspeed` launcher or torchrun; CUDA | ZeRO-2 by DeepSpeed | DeepSpeed engine checkpoints (custom loops) | config missing/without `zero_optimization` → `DistributedConfigError`; extra missing → `OptionalDependencyError` | config validation on CPU; `test_hardware.py::test_deepspeed_initialize` **skipped (extra not installed)** | none (an earlier script parsed `--deepspeed` but never called `deepspeed.initialize`) | custom loops only |
| **Tensor parallelism** | `distributed/tensor_parallel.py::ColumnParallelLinear` (`Y_i = X A_iᵀ`, identity fwd / all-reduce bwd), `RowParallelLinear` (all-reduce fwd), `apply_tensor_parallel(block, mesh)` (q/k/v & FFN up column-parallel, output & FFN down row-parallel; MLA projections handled), `from_linear` slices trained weights; `TensorParallelStrategy` | `ParallelMesh(tp, pp)` groups; NCCL within a node | all-reduce of partial products per layer | replicated embeddings/head; sharded linears | `world_size % (tp·pp) != 0` → error | `test_distributed_gloo.py::test_tp_size1_equivalence_and_pipeline` (degree-1 surgery reproduces logits to 1e-5 in a real gloo group); `test_hardware.py::test_tensor_parallel_two_ranks` **skipped** | none | embeddings/head not sharded; intra-node assumption |
| **Pipeline parallelism** | `distributed/pipeline_parallel.py::PipelineStage` (contiguous block slice; first embeds, last has norm+head), `one_f_one_b_schedule` (warm-up/steady/cool-down), `PipelineScheduler.run` (P2P `send/recv` between neighbours, loss on last stage); `PipelineParallelStrategy.wrap_model`, `build_scheduler` (uses `pipeline_micro_batches`, restored in this audit) | mesh `prev/next_pipeline_rank`; NCCL/gloo | activations and gradients point-to-point | per-stage parameters | fixed activation shape assumed | `test_tp_size1_equivalence_and_pipeline` (single-stage 1F1B loss equals model loss, gradients produced), `test_pipeline_strategy_uses_configured_micro_batches`, `tests/unit/test_distributed.py::test_schedule_orders` | none | float32 activations; single-stage local validation only |
| **3-D mesh** | `distributed/topology.py::ParallelMesh` (`global_rank = dp·(tp·pp) + pp_rank·tp + tp_rank`), `distributed/runtime.py::validate_topology` | planning without process groups (`create_groups=False`) | — | — | non-tiling world sizes rejected | `tests/unit/test_distributed.py::test_mesh_layout, test_topology_validation`; `forgeline validate <manifest> --world-size N` | none | — |
| **torchrun / multi-process** | `distributed/runtime.py::read_launcher_env, init_process_group` (creates a 1-process group when unlaunched; sets MASTER_ADDR/PORT), `cli train` | `torchrun --nproc_per_node=N -m forgeline.cli.main train …` | — | — | — | `test_ddp_two_process.py` launches two subprocesses with launcher env vars; `test_build_strategy_and_env` | historical notes document `torchrun` and a 3×A30 command — no retained multi-process run | — |
| **Ray** | `orchestration/ray_backend.py` (§20) | actors on a local instance or cluster | object-store weight broadcast; no gradient traffic | learner checkpoints as usual | worker replacement + bounded retries | 11 CPU tests on a local 4-CPU Ray instance | none (only a cluster configuration file existed; its entry script did not) | single machine validated only |
| **Distributed checkpointing** | `Trainer.save_checkpoint` (main rank writes; FSDP full-state gather collective), `CheckpointManager` atomic writes | — | — | — | truncated/incomplete files rejected on load | `tests/unit/test_checkpoints.py`, `test_failure_modes.py` | A40 run: checkpoint corruption on a network filesystem and 5.1 GB rotation exhausting disk were diagnosed and fixed during that run — the design lessons behind atomic writes, rotation and CPU-mapped loading | no per-rank sharded files; FSDP optimizer state not saved |

Manifests: `configs/distributed/{single, ddp, fsdp, tensor_pipeline_3d, deepspeed}.yaml`, `deepspeed_zero2.json`.

---

## 20. Ray orchestration

Status: **restored as the optional `ray` extra** (`pip install -e ".[ray]"`). No earlier Ray code existed: only a cluster configuration file (single-node 4×A30 comments, `ray_actor_options: {num_gpus: 1, num_cpus: 8}`, runtime env) referencing an entry script that was never written. Forgeline's implementation is new, designed around the intended uses (rollout, reward/verifier, tool and evaluation workers; placement; recovery).

Architecture (`orchestration/ray_backend.py`, `orchestration/config.py::RayConfig`, `orchestration/__init__.py::validate_orchestration`):

| Pool | Actor | Runs in the worker | Driver-side API | GPU-eligible |
|---|---|---|---|---|
| `RayRolloutEngine` | `RolloutWorker(model, tokenizer, pad, RolloutConfig, seed)` — a `NativePolicy` copy + `RolloutEngine`; `load_weights(state, version)`; `rollout(prompt_ids, task, text, G)` | `rollout(...)` splits `G` across workers, `rollout_many` round-robins prompts; `sync_weights(force)`; `worker_versions()`; `weight_pushes`, `restarts` | yes |
| `RayRewardPool` | `RewardWorker(reward)` — builds the manifest reward via `build_reward_provider` (or unpickles a provider) | `RewardProvider` protocol + `score_many(trajectories)` (contiguous shards) — `score_trajectories` uses it automatically | no |
| `RayToolPool` | `ToolWorker` — `execute_python` per snippet | `execute_many(codes)` (ordered) | no |
| `RayEvaluator` | `EvaluationWorker(checkpoint, data_dir)` — loads a policy, runs `BenchmarkSuite.run(task_indices=shard)` | `run(benchmark, max_new_tokens, **kwargs)` merges per-item results exactly (`details.workers`, `shard_sizes`); CLI `forgeline evaluate --ray-workers N [--ray-address …]` | yes |

Shared base `_WorkerPool`: `ray.remote(actor_class).options(num_cpus, num_gpus (GPU pools only), memory, max_restarts, scheduling_strategy=PlacementGroupSchedulingStrategy(bundle i))`; placement group with one bundle per worker when `placement_strategy ∈ {PACK, SPREAD, STRICT_PACK, STRICT_SPREAD}` (removed and `OrchestrationError` raised if not ready within `placement_timeout_s`); `call([(worker, method, args)])` submits concurrently and waits with `ray.wait` (optional `task_timeout_s`).

On-policy weight synchronisation: `_weights_version(model)` = Σ tensor `_version` counters of the parameters (changes on any optimizer step or `load_state_dict`); `sync_weights` puts a CPU copy of the state dict in the object store once and broadcasts `load_weights` when the version changed; old/reference log-probs are recomputed by the learner from returned token ids (no numerics cross the worker boundary).

Recovery: failures of type `RayActorError`, `ActorUnavailableError`, `WorkerCrashedError`, `ObjectLostError` or a `GetTimeoutError` mark the call failed; `_replace(index, generation)` kills and respawns the actor once per generation, increments `restarts`, calls `_restore` (rollout workers reload the latest weights) and resubmits; after `max_call_retries` resubmissions an `OrchestrationError` is raised. Exceptions raised by user code propagate unchanged (no retry). Ray-level `max_restarts` also applies.

Manifest integration: `orchestration: {type: ray, rollout_workers, reward_workers, cpus_per_worker, gpus_per_worker, memory_per_worker_mb, placement_strategy, placement_timeout_s, max_restarts, max_call_retries, task_timeout_s, worker_torch_threads, address, num_cpus, num_gpus, namespace}` validated without importing Ray (`validate_orchestration`); `training/factory.py::_attach_orchestration` swaps the algorithm's `rollouts` and `reward` for pools (PPO/GRPO/DAPO single-turn; agent mode keeps rollouts in process); `cli train` shuts pools down in `finally`. Examples: `configs/post_training/grpo_ray_tiny_cpu.yaml`, `configs/orchestration/ray_cluster.yaml`.

Serving: Ray Serve is intentionally not used — the stdlib OpenAI-compatible server plus `RoutingPolicy` cover the serving path and nothing requires replica autoscaling.

Tests (`tests/integration/test_ray_orchestration.py`, local `ray.init(num_cpus=4)`, all CPU): reward pool equals in-process scoring on 2 worker processes; tool pool ordering; rollout workers equal learner rollouts before and after a parameter update (one push per change, none when unchanged, uniform versions); a killed rollout worker is replaced with current weights; an `os._exit` crash inside a reward worker is retried and succeeds; user errors propagate without retry; retry budget → `OrchestrationError`; placement group created / infeasible request rejected; sharded evaluation identical to serial; GRPO through `Trainer.fit` with 2 rollout + 1 reward workers (≥3 weight pushes); `forgeline train` with the Ray manifest. `tests/unit/test_orchestration_config.py` validates configs and proves the core never imports Ray (subprocess check). `tests/hardware/test_hardware.py::test_ray_rollout_worker_uses_assigned_gpu` (CUDA, skipped here).

Not done / not claimed: no multi-node cluster run, no GPU Ray run, no Ray throughput numbers, no asynchronous off-policy rollout buffers, no HuggingFace policies in rollout workers.

---

## 21. Checkpointing / recovery

`checkpoints/manager.py::CheckpointManager(root, keep_last=3, async_save=False)`; each checkpoint is a directory `<root>/<name>/`:

| File | Content |
|---|---|
| `manifest.json` | `format_version`, `created_at`, `step`, `algorithm`, `model_spec`, `tokenizer_meta`, `metadata`, `files{name: {size}}`, `complete: true` |
| `model.pt` | model state (`{"model": …, "adapter": …, "value_head": …}` for policies; adapter-only for HF LoRA) |
| `optimizer.pt`, `scheduler.pt`, `scaler.pt`, `adapter.pt`, `rng.pt` | optional: optimizer state, scheduler state, GradScaler state, adapter tensors, RNG state (`core/runtime.py::rng_state`: Python, NumPy, torch CPU/CUDA) |
| `trainer.pt` | `best_metric`, `history_len` (trainer step is in the manifest) |
| `experiment.json` | the full `ExperimentManifest` (config, data, distributed, algorithm params, runtime environment) |

Writes go to `<name>.tmp` then `os.replace` (atomic rename), `LATEST` pointer updated, `best` copied when `is_best`, rotation keeps `keep_last` step checkpoints; `async_save` uses a background thread with `wait()`. Not saved: epoch (steps are the unit), streaming-loader position (limitation), distributed rank metadata beyond the manifest's `distributed` config; FSDP runs omit optimizer state.

Validation (`CheckpointManager.validate` / `read_manifest`): missing directory → `CheckpointNotFoundError`; unparsable or key-missing manifest → `CheckpointCorruptError`; `complete` false, missing file or size mismatch (truncation) → `CheckpointCorruptError`; format version or model-spec/algorithm mismatch → `CheckpointMismatchError`; `.tmp` directories are ignored by discovery. `checkpoints/validation.py::compare_state_shapes, assert_compatible, finite_state`. `models/loading.py::load_model_state` strips `_orig_mod.` (torch.compile) prefixes.

Resume (`Trainer.resume(path=None, strict_spec=True)`): explicit path, `checkpoint.resume_from`, or `LATEST` auto-resume; loads model/adapter/value head, optimizer, scaler, RNG, step and best metric; checks spec and algorithm. Exact-continuation evidence: `tests/integration/test_checkpoint_resume.py::test_resume_matches_uninterrupted` trains N steps, saves at step 3, reloads into a fresh trainer and continues — per-step losses match the uninterrupted run within 1e-5; `test_auto_resume_from_latest_and_adapter_state` covers LoRA state and `LATEST`; `test_compile_model_matches_eager_and_checkpoints_load` reloads compiled-training checkpoints; `test_merge_on_save_writes_plain_model_checkpoint`.

Failure tests (`tests/failure/test_failure_modes.py`): `test_checkpoint_missing`, `test_checkpoint_bad_metadata`, `test_checkpoint_incomplete_and_truncated`, `test_checkpoint_incomplete_flag_and_tmp_dir_ignored`, `test_checkpoint_mismatched_config_and_state` (spec mismatch, algorithm mismatch, incompatible tensor shapes); `tests/unit/test_checkpoints.py` (save/load/latest/rotation, async save, compile prefixes, shape comparison). Dashboard surfaces validity per checkpoint.

---

## 22. Evaluation

| Evaluator | File | Metrics |
|---|---|---|
| `HeldOutLossSuite(corpus)` | `evaluation/quality.py` | held-out loss, perplexity over `eval_batches` |
| `VerifierPassRateSuite(tasks, reward, settings)` | `evaluation/quality.py` | greedy pass rate, mean reward, malformed-output rate |
| `AgentAccuracySuite` | `evaluation/quality.py` | accuracy and tool-use rate on tagged-answer tasks (multi-turn) |
| `RecordRewardSuite(records, scorer)` | `evaluation/reward.py` | `reward_statistics` (mean, std, min, max, fraction above thresholds) over dataset records; `details.policy_dependent = False` |
| `TrajectoryRewardSuite` | `evaluation/reward.py` | same statistics over policy trajectories (policy-dependent) |
| `PreferenceWinRateSuite` | `evaluation/reward.py` | fraction of pairs where the scorer ranks chosen above rejected |
| `BenchmarkSuite(build_benchmark(name))` | `evaluation/suites/__init__.py` | MMLU, HellaSwag, ARC, GSM8K, TruthfulQA, HumanEval: generative scoring, `n_shot`, `max_tasks`, `offline` sample fallbacks (a few items each, wiring only), remote loading with the `huggingface` extra; `task_indices` sharding (Ray) |
| `PerformanceSuite` / `measure_generation` | `evaluation/performance.py` | tokens/s, per-run latency, device memory on the current device |
| `evaluate_regression(candidate, baseline, rules)` | `evaluation/regression.py` | fail-closed rule checks (missing/NaN fail) — same semantics as promotion gates |
| Inspection | `evaluation/inspection.py` | `weight_statistics` (per-parameter mean/std/min/max/norm/histogram), `layer_summary(spec)`, `activation_flow` (residual-stream stats per block), `attention_patterns` (exact head-averaged/per-head weights per layer; tested to reconstruct layer outputs) |

Results IO: `evaluation/results.py::save_results/load_results` (JSON/JSONL), `merge_metrics` flattens to `suite/metric` for gates (`forgeline registry register --metrics-file`). CLI: `forgeline evaluate --suites … [--offline] [--heldout-data] [--output] [--ray-workers N]`.

Tests: `tests/unit/test_evaluation.py` (7), `test_dashboard.py::test_attention_patterns_reproduce_the_attention_output` (2 variants), `test_ray_orchestration.py::test_sharded_evaluation_equals_serial`. Limitations: multiple-choice benchmarks are scored generatively (no log-likelihood scoring); offline samples never produce reportable scores.

---

## 23. Quantization / export

| Capability | Implementation | Delegated to | Tests / evidence |
|---|---|---|---|
| NF4 (4-bit NormalFloat) | `models/quantization/nf4.py::quantize_nf4` (block 64, absmax scale, nearest of 16 NF4 levels, 2 indices/byte), `dequantize_nf4`, `NF4Linear`, `quantization_report` | — (pure PyTorch) | `test_nf4_roundtrip_and_qlora`, `test_nf4_table_and_block_size_unchanged` |
| QLoRA | `models/adapters/qlora.py::apply_qlora` (NF4 base + LoRA) | — | `test_sft[qlora]` |
| INT8 base loading | `HFPolicy(load_in_8bit=True)` | bitsandbytes via transformers (GPU) | not executed here |
| GGUF export | `models/quantization/gguf.py::export_gguf(state_dict, spec, path, quantize="none|q8_0|q4_0")`: GGUF v3, magic `GGUF`, metadata KVs (`general.architecture = "forgeline"`, name, file type, alignment 32, model dims), tensor infos with **offsets computed relative to the aligned data-section start and padded to 32 bytes** (asserted on write), MTP tensors skipped; `read_gguf_header`, `read_gguf_tensor`, `dequantize_q8_0/q4_0` | — | `test_gguf_export_roundtrip` (offsets strictly increasing and aligned; FP16/Q8_0/Q4_0 read back within tolerance), `test_block_quantizers_roundtrip`, `test_cli_pipeline.py::test_generate_evaluate_export_registry`, `test_failure_modes.py::test_quantization_and_export_errors` |
| Q8_0 / Q4_0 layouts | Q8_0: fp16 scale + 32 int8 (`amax/127`); Q4_0: fp16 scale + 16 bytes, low nibbles = elements 0–15, high nibbles = 16–31, stored as `q + 8` (ggml layout) | — | round-trip tests |
| Adapter merge before export | `forgeline export --merge-adapters` → `merge_lora`; `trainer.adapter.merge_on_save` | — | `test_lora_identity_at_init_and_merge`, `test_merge_on_save_writes_plain_model_checkpoint` |

Bug history (§35 B4): the original exporter wrote `0` for every tensor offset and packed Q4_0 pairs interleaved (`lo = q[i]`, `hi = q[i+1]`) without the +8 bias — files could not be read by GGUF consumers. No quantized-model accuracy or speed measurements exist (`benchmarks/quantization/summary.md`).

---

## 24. Inference engine

Terminology: Forgeline implements **Python-level paged KV-cache block accounting with continuous batching**, not PagedAttention kernels — attention runs over each sequence's contiguous per-layer cache tensors.

`inference/engine.py::InferenceEngine(model, max_batch=8, max_seq_len=block_size, block_size=16, max_blocks, eos_token_id)`:
* **Admission** — `submit(prompt_tokens, SamplingParams)` validates params and enqueues a `GenerationRequest` (`inference/requests.py`, statuses PENDING/RUNNING/FINISHED/CANCELLED). `Scheduler.admit` (`inference/scheduler.py`) moves FIFO pending requests into the running set while `len(running) < max_batch`, rejects `prompt + max_tokens > max_seq_len` with `finish("length", error)`, and stops admitting when `PagedBlockAllocator.can_allocate(prompt + 1)` fails (waits for blocks).
* **Paged memory** — `inference/paged_memory.py::PagedBlockAllocator(block_size, max_blocks)`: free-list of block ids, per-sequence block tables and lengths; `allocate` reserves `ceil(n/block_size)` blocks, `append_token` grows by one token, `free` recycles blocks on retirement; `OutOfBlocksError`; `stats` (used/free/utilization/sequences). Capacity defaults to `max_batch × (ceil(max_seq_len/block) + 1)` blocks.
* **Prefill** — `model.prefill(prompt)` produces logits and per-layer caches (`(k, v)` or `(c_kv, k_rope)`); first token sampled immediately (`_emit`).
* **Decode / batching** — each `step()` groups running sequences by cache length (`position`); groups are decoded in one batched `model.step` by concatenating caches along the batch dim (`_cat_cache`) and splitting the results (`_split_cache`); singletons decode alone. New requests join between steps (continuous batching); finished sequences release blocks immediately.
* **Sampling & stopping** — shared `apply_sampling_filters` per request (temperature, top-k, top-p, min-p, repetition penalty over the request's own tokens); stop on `stop_token_ids`/EOS ("stop"), `max_tokens` or `max_seq_len` ("length"), or a cancel flag ("cancelled"); `Scheduler.cancel` removes pending requests or flags running ones.
* **Failure isolation** — an exception in prefill/decode fails only the affected sequences (`_fail` → `retire(reason="error")`), others continue (`test_engine_isolates_failing_sequence`).
* **Concurrency** — `start()` runs the loop in a daemon thread polling the scheduler; `step` holds an internal lock; `generate` and `run_until_idle` are blocking helpers; `Tracer` spans `prefill`/`decode`; `stats()`.
* **Backends** — native `TransformerLM` only in the engine; `HFPolicy` generation uses `transformers.generate` (rollouts/evaluation), not this engine; quantized loading: NF4 base weights (native) or 8-bit (HF, GPU).

Evidence: `tests/unit/test_inference_serving.py::test_engine_continuous_batching_and_greedy_equivalence` (batched outputs equal single-sequence greedy decoding token for token), `test_engine_stop_token_and_cancel`, `test_paged_allocator`, `test_scheduler_admission_and_rejection`; `test_hardware.py::test_cuda_inference_engine` (skipped). No throughput/latency measurements of the engine exist (`benchmarks/inference/summary.md`); historical single-stream speeds (49 tok/s CPU, ~53 tok/s A40) belong to the original generation code (H04, H13).

---

## 25. Serving

`serving/api.py::Router(backends: {name: Backend}, default_model, route_fn=None, metrics=None)` — transport-agnostic: `handle(method, path, body) → (status, payload | SSE iterator)`. Routes: `GET /health`, `GET /v1/models`, `POST /v1/completions`, `POST /v1/chat/completions` (messages flattened to a prompt), `GET /metrics` (request/error counters, p50/p95 latency, completion tokens). Schemas (`serving/schemas.py`) validate `model`, `prompt|messages`, `max_tokens`, `temperature`, `top_p`, `stop`, `stream`; invalid → 400; unknown/disabled backend → 503 (`BackendUnavailableError`); backend exception → 500 (`ServingError`). `route_fn(request_id, requested_model)` lets `RoutingPolicy.route(...).primary` pick the backend (`forgeline serve --routing routing.json`).

Backends (`serving/backends.py`): `EngineBackend(name, InferenceEngine, tokenizer)` — encodes, submits to the running engine, `complete` blocks on the request, `stream` yields tokens as they are pushed; `disable/enable` for kill switches; `EchoBackend` for tests. `serving/server.py::ServingServer` — stdlib `ThreadingHTTPServer` handler with JSON responses and `text/event-stream` chunks (`data: {...}` … `data: [DONE]`); no FastAPI dependency (the `serving` extra is optional).

Tests: `test_inference_serving.py::test_schemas, test_router_with_echo_backend, test_engine_backend_end_to_end`; `test_cli_pipeline.py::test_http_server_roundtrip` (live server over a trained tiny checkpoint, streaming); `test_failure_modes.py::test_serving_backend_failure`. Limitations: no auth/TLS/rate limiting; shadow traffic is decided but not dispatched; no serving latency/throughput measurements exist.

---

## 26. Model lifecycle

State machine (`deployment/candidates.py::CandidateState`, `ALLOWED_TRANSITIONS`):

```
EXPERIMENTAL → {SHADOW, CHALLENGER, RETIRED}
SHADOW       → {CHALLENGER, EXPERIMENTAL, RETIRED}
CHALLENGER   → {CHAMPION, SHADOW, EXPERIMENTAL, RETIRED}
CHAMPION     → {RETIRED, CHALLENGER}
RETIRED      → {CHALLENGER, CHAMPION}      (only via explicit rollback / restore)
```
`ModelCandidate` fields: `name, version, algorithm, checkpoint, tokenizer, dataset, precision, distributed_strategy, model_spec, metrics, state, id (uuid12), created_at, updated_at, promotion_status (none|passed|rejected), promotion_report, rollback_target, tags, history[{from,to,reason,ts}]`; `transition` raises `RegistryError` on disallowed moves.

Registry (`deployment/registry.py::CandidateRegistry(path, metrics=None)`): JSON file `{"format_version": 1, "candidates": [...]}`, written atomically (`mkstemp` + `os.replace`), corrupt file → `RegistryError`; duplicate `name:version` rejected; `transition(id, CHAMPION)` retires the current champion (reason `replaced by <id>`) and records it as the new champion's `rollback_target`; at most one champion per name (`champion()` raises if more); `record_metrics`; events `candidate.registered|promoted|rejected|rolled_back` through the metrics sink.

Promotion gates (`deployment/promotion.py`): `GateRule(metric, min, max, min_delta, max_increase, relative)`; `PromotionGate(rules, require_baseline, when_no_baseline="fail"|"skip_relative")`; evaluation fails a rule when the candidate metric is missing or non-finite, when an absolute bound is violated, when a relative rule has no finite baseline (unless `skip_relative` and there is no incumbent at all), or when `candidate < baseline + min_delta·scale` / `> baseline + max_increase·scale` (`scale = |baseline|` if relative). `assert_promotable` raises `PromotionError` listing failing rules. CLI `forgeline registry promote --id … --to champion --gate configs/deployment/promotion_gate.yaml` exits 2 on rejection and records `promotion_status`/`report`; promotion from EXPERIMENTAL to CHAMPION auto-stages through CHALLENGER.

Routing (`deployment/rollout.py::RoutingPolicy(champion, challenger, challenger_percent, shadow, shadow_percent, salt, pinned, disabled)`): `bucket = sha256(f"{salt}:{request_id}")[:8] mod 10_000`; challenger when `bucket < challenger_percent × 100`; shadow decided by an independent hash (`salt:shadow`); explicit model requests and pinned ids bypass hashing; disabled backends never selected; `simulate_traffic` measures splits; `offline_replay(backends, prompts, scorer)` compares candidates on identical prompts. Feature flags (`deployment/feature_flags.py::FeatureFlags`, JSON-persisted): `enabled`, `percentage`, allow/deny lists, deterministic `stable_bucket`.

Rollback (`deployment/rollback.py::rollback(registry, name, reason, routing)`): retires the champion, promotes its `rollback_target` back to CHAMPION (allowed from RETIRED), clears the target's own rollback pointer, updates routing (champion key, clears a challenger that was involved, disables the retired backend), emits `candidate.rolled_back`; fails when there is no champion or no target so a model name never ends up unserved. `disable_candidate` = kill switch (routing only, tags `disabled`, history entry).

Tests (`tests/unit/test_deployment.py`, 7): registry lifecycle and invalid transitions; gate success/rejection; boundaries, missing metric, NaN, invalid config; deterministic routing and percentage accuracy (10%/20% splits over many ids); offline replay; feature flags; rollback and disable. `test_cli_pipeline.py::test_generate_evaluate_export_registry` (register → gated promotion rejection exit code 2 → promotion → rollback via CLI); `examples/01_end_to_end_lifecycle.py`.

---

## 27. Dashboard / observability

Observability (`observability/`): `logging.py::get_logger(name).info("event", key=value)` → `event key=value` text or JSON lines (`FORGELINE_LOG_FORMAT=json`, `FORGELINE_LOG_LEVEL`); `metrics.py::MetricsSink` (in-memory history, `latest`, `series`, counters, events) with `NoOpMetricsSink`, `LocalJsonlMetricsSink` (`metrics.jsonl` records `{step, ts, …}` and `events.jsonl` `{event, ts, step, …}`; non-finite → null), optional `WandbMetricsSink` / `TensorBoardMetricsSink`; `build_metrics_sink(backend)`; `events.py::Event` = `run.started|finished|failed`, `checkpoint.saved|loaded|failed`, `evaluation.started|finished`, `rollout.batch|failed`, `candidate.registered|promoted|rejected|rolled_back`, `serving.request|error`; `tracing.py::Tracer.span` (count/mean/max per span); `gradient_norms(model)`, `device_memory_metrics(device)`. Trainer logs `train/loss`, `train/lr`, `train/grad_norm` and algorithm metrics every `log_every`, `eval/*` every `eval_every`; Router records serving counters and latency. Everything except W&B/TensorBoard is stdlib/torch.

Dashboard (`dashboard/`, restored; optional to run, zero dependencies): `forgeline dashboard --runs runs [...] --registry registry/candidates.json --benchmarks benchmarks --port 8765` or `--export snapshot.json`.

| Layer | File | Detail |
|---|---|---|
| data | `dashboard/data.py` | `discover_runs(roots)` (directories with `metrics.jsonl`/`events.jsonl`; status from `run.*` events; `live` when updated < 120 s; manifest fields incl. strategy/orchestration), `run_detail` (series per metric downsampled to ≤ 1,500 points, last 500 events, `run.finished` summary, checkpoints, evaluation files), `list_checkpoints` (validity via `CheckpointManager.validate`, sizes, `LATEST`/`best`), `list_evaluations` (`save_results` files), `registry_view` (candidates grouped by name with history), `benchmark_records` (`results.json` + `summary.md` with evidence and ledger flags), `inspect_checkpoint` (spec, parameter stats, tokens, attention maps, activation flow), `DashboardSources.checkpoint_allowed` (only under configured run roots — `torch.load` runs pickled code), `snapshot` |
| server | `dashboard/server.py::DashboardApp.handle(path)` → `/`, `/health`, `/api/overview`, `/api/runs`, `/api/runs/<id>`, `/api/registry`, `/api/benchmarks`, `/api/inspect?checkpoint&text` (403 outside run roots; 4-entry cache); `DashboardServer` (stdlib `ThreadingHTTPServer`, `Cache-Control: no-store`) | read-only |
| UI | `dashboard/static/index.html` (inline CSS/JS, dark-mode aware) | tabs Runs (list, status badges, metric SVG charts, checkpoint table with Inspect buttons, evaluations, events; 5 s refresh while live), Registry (per-model tables with states, gate status, metrics, transitions), Benchmarks (evidence badges, rerun/not-published flags, summaries), Inspect (model facts, per-layer attention heatmaps with head entropies, residual-norm bars, weight table) |

The earlier visualisation tool (architecture graph, weight stats, attention hooks, activation flow for one checkpoint) is a subset of the Inspect tab; run/registry/benchmark views are new. Tests: `tests/integration/test_dashboard.py` (7): discovery/detail on a real CLI run, truncated checkpoint flagged, API routes, inspection path restriction, attention reconstruction (2 variants), live HTTP + CLI export.

---

## 28. Failure engineering

Errors (`core/errors.py`, all with a `hint`): `ConfigError`, `UnknownPresetError`, `DatasetError`, `MalformedRecordError`, `TokenizerError`, `ModelError`, `IncompatibleStateError`, `QuantizationError`, `ExportError`, `RewardError`, `InvalidRewardError`, `VerifierError`, `ToolExecutionError`, `RolloutError`, `CheckpointError`, `CheckpointNotFoundError`, `CheckpointCorruptError`, `CheckpointMismatchError`, `DistributedConfigError`, `EvaluationError`, `PromotionError`, `RegistryError`, `ServingError`, `BackendUnavailableError`, `OptionalDependencyError`, `OrchestrationError` (`test_all_errors_have_hints`).

Recovery behaviours: atomic checkpoints + validation + resume (§21); verifier exceptions → failed results; tool errors injected as `<tool_result>ERROR…` and training continues; non-finite loss aborts with `ConfigError` (`run.failed` event emitted); non-finite reward → `InvalidRewardError` naming the provider; inference sequence failures isolated; serving 400/503/500 mapping; promotion fails closed; rollback never leaves a name unserved; Ray worker replacement with bounded retries; gloo/NCCL misconfiguration rejected before training. Tests: `tests/failure/test_failure_modes.py` (17 tests): checkpoint missing/bad metadata/incomplete/truncated/tmp ignored/mismatched; dataset failures; distributed misconfiguration; engine isolation; invalid reward and verifier exceptions; non-finite loss; promotion failure; quantization/export errors; rollout failure; serving backend failure; unsupported strategy combinations; tool execution failure injection; error hints.

---

## 29. Complete test evidence

Full environment: **251 passed, 8 skipped** (259 collected). Clean `.[dev]` install: **234 passed, 9 skipped** (HuggingFace module 7 tests and Ray module 11 tests skip via `importorskip`; `test_ray_backend_reports_missing_extra` runs instead of skipping). Previous state before this audit: 192/6 and 185/7.

By directory (full environment): unit 133 · integration 98 · failure 17 · smoke 3 (+1 skipped) · hardware 0 (+7 skipped).

By subsystem (test functions; parametrised cases counted once):

| Subsystem | Tests | Files |
|---|---|---|
| config / manifest | 8 + 3 orchestration config | `test_config.py`, `test_orchestration_config.py` |
| data | 9 | `test_data.py` |
| models / generation | 7 (20 with parametrisation) | `test_models.py` |
| adapters / quantization / GGUF | 5 | `test_adapters_quantization.py` |
| training math (schedules, GAE, KL, objectives, optimizer groups) | 5 | `test_training_utils.py` |
| benchmark semantics (pretrain CE, cosine, rule reward, 0.808, DPO mean, PPO ratio, tagged reward, PRM, NF4, LOO, generator, policy-independence, labeled statistics, legacy term) | 14 | `test_benchmark_semantics.py` |
| SFT / pretrain / distill / reward model | 4 (6 with adapters) | `test_training_lifecycle.py` |
| DPO | 1 (2) + HF | `test_training_lifecycle.py`, `test_huggingface_backend.py` |
| PPO | 1 (3) + HF | same |
| GRPO / agent GRPO / DAPO / RLVR | 5 + HF + DDP + Ray | same, `test_ddp_two_process.py`, `test_ray_orchestration.py` |
| rollouts / verifiers / rewards / tools | 9 | `test_rollouts_rewards.py` |
| checkpoints / resume / compile / merge | 3 + 4 + 5 failure | `test_checkpoints.py`, `test_checkpoint_resume.py`, `test_failure_modes.py` |
| distributed | 4 + 3 gloo + 1 two-process + 3 hardware(skipped) | `test_distributed.py`, `test_distributed_gloo.py`, `test_ddp_two_process.py` |
| Ray | 11 + 3 config + 1 smoke + 1 hardware(skipped) | `test_ray_orchestration.py` |
| evaluation | 7 | `test_evaluation.py` |
| inference / serving | 7 + 2 CLI + 1 failure | `test_inference_serving.py`, `test_cli_pipeline.py` |
| lifecycle (registry, gates, routing, flags, rollback) | 7 + CLI | `test_deployment.py`, `test_cli_pipeline.py` |
| dashboard | 6 (7) | `test_dashboard.py` |
| synthesis domain | 6 | `test_synthesis_domain.py` |
| observability / CLI | 5 + 6 CLI pipeline | `test_observability_cli.py`, `test_cli_pipeline.py` |
| failure handling | 17 | `test_failure_modes.py` |
| smoke / packaging | 4 | `test_smoke.py` |
| sequential decisioning: environment, oracle, pacing metrics | 10 | `test_allocation_env.py` |
| sequential decisioning: OPE formulas, schema, support, bootstrap | 7 | `test_allocation_ope.py` |
| sequential decisioning: baselines, GRU history, PPO via trainer, OPE on logs, A/B, shadow, lifecycle, CLI | 13 | `test_allocation_pipeline.py` |

Especially valuable tests: `test_sequence_mean_ppo_ratio_reference` (ratio correctness against an independent formula); `test_dpo_mean_reduction_reference`; `test_resume_matches_uninterrupted` (exact continuation); `test_checkpoint_incomplete_and_truncated`; `test_routing_deterministic_and_percentages`; `test_gate_succeeds_and_rejects` / `test_promotion_failure`; `test_gguf_export_roundtrip` (offsets); `test_code_verifier_and_executor` / `test_agent_rollout_executes_tool_calls`; `test_ddp_replicas_stay_identical` (two real processes); clean-install run (204/9) proving optional dependencies; `test_core_import_does_not_import_ray`; `test_synthesis_pipelines_on_hf_backend` (HF backend genuinely executed on CPU with a tiny local Llama built in the fixture); `test_synthesis_rule_reward_ignores_generated_text_for_nested_records` (pins the 0.808/0.9007 interpretation); `test_attention_patterns_reproduce_the_attention_output`; `test_core_formulas_against_hand_values` (IPS/SNIPS/PDIS/DR/ESS/clipping against hand-computed numbers); `test_endogenous_feedback_changes_future_state_under_same_seed`; `test_ppo_improves_over_random_init`; `test_permutation_test_and_ab_decisions`; `test_lifecycle_with_ab_gate`.

Skips and why: `test_cuda_bf16_training_step`, `test_cuda_fp16_grad_scaler_enabled`, `test_cuda_inference_engine`, `test_ray_rollout_worker_uses_assigned_gpu` — no CUDA device; `test_fsdp_wrap_under_torchrun`, `test_tensor_parallel_two_ranks` — fewer than 2 CUDA devices; `test_deepspeed_initialize` — deepspeed extra not installed; `test_ray_backend_reports_missing_extra` — Ray is installed (runs in the clean install). Clean install additionally skips the HuggingFace and Ray integration modules.

---

## 30. Complete benchmark ledger

Every historical number retained with the project, one record each; machine-readable copy in `benchmarks/historical_ledger.json`. Totals: **82 records** — CARRY_FORWARD 19, CARRY_FORWARD_WITH_INTERPRETATION_NOTE 42, NEEDS_EVIDENCE_RECOVERY 5, REVALIDATE 0, DO_NOT_USE 16.

Status definitions: CARRY_FORWARD — artifact exists and current semantics are materially equivalent (or the number was recomputed exactly); cite as is. CARRY_FORWARD_WITH_INTERPRETATION_NOTE — real measurement whose meaning must be stated (record statistic, noisy batch mean, historical implementation timing, negative result). NEEDS_EVIDENCE_RECOVERY — documented but the producing artifact/log/config is missing; not cited until recovered or rerun. REVALIDATE — semantics changed enough that the old number no longer describes current code (none). DO_NOT_USE — contradicted, copied, estimated or measuring something other than its label.

### 30.1 Summary table

| ID | Metric | Value | Subsystem | Status |
|---|---|---|---|---|
| H01 | Parameters, `small` preset (char vocab) | 10.6M | model architecture | CARRY_FORWARD |
| H02 | Best validation loss, char-level `small` model | 1.479 (iteration 1,500 of 5,000) | pretraining | CARRY_FORWARD |
| H03 | Wall-clock, char-level training | ~45 min | pretraining | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H04 | Single-stream generation speed, `small`, KV cache, CPU | 49 tokens/s | inference | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H05 | Parameters, `large` preset (vocab 151,643, ctx 2048) | 421.1M | model architecture | CARRY_FORWARD |
| H06 | Best validation loss, `large` on FineWeb-Edu | 3.5834 (step 29,500) | pretraining | CARRY_FORWARD |
| H07 | Final training loss, `large` | 3.5478 (step 30,000) | pretraining | CARRY_FORWARD |
| H08 | Loss progression, `large` | 12.12 (0) → 5.44 (1k) → 4.65 (3k) → 4.30 (5k) → 4.00 (10k) → 3.70 (20k) → 3.58 (30k) | pretraining | CARRY_FORWARD |
| H09 | Tokens processed, `large` | ~491M (30,000 × 16,384) | pretraining | CARRY_FORWARD |
| H10 | Average training throughput, `large`, A40 | 20,700 tokens/s | pretraining systems | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H11 | Peak training throughput, `large`, A40 | 20,900 tokens/s | pretraining systems | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H12 | torch.compile throughput effect, A40 | ~20,800 vs ~15,000 tokens/s (~39%) | pretraining systems | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H13 | Generation speed, `large`, KV cache, A40 | ~53 tokens/s | inference | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H14 | Wall-clock, `large` run | ~14 hours | pretraining operations | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H15 | VRAM breakdown, `large` batch 8 | ~0.8 weights + 3.2 optimizer + 0.8 grads + 35 activations + 2 overhead ≈ 42 GB | memory | DO_NOT_USE |
| H16 | Memory feasibility, `large` | batch 8 × 2048 tokens trained on a 46 GB A40 | memory | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H17 | Prepared corpus size | 1B tokens FineWeb-Edu (≈7.5 GB token files) | data pipeline | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H18 | Checkpoint size, `large` | 5.1 GB per full checkpoint | checkpointing operations | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H19 | Expected losses and budgets in launch scripts | val loss ~2.5–3.0 / ~2.0–2.5; ~$3–4; ETA ~2–3 h | planning | DO_NOT_USE |
| H20 | Cloud tutorial throughput and example logs | 61,411 tok/s sample log; 1×A100 ~60K, 4×A100 ~80K, 8×A100 ~100K tok/s | planning | DO_NOT_USE |
| H21 | Feature speed/memory multipliers in feature table | MLA ~10× smaller cache; SDPA 2–4× faster; NSA 9× faster; LoRA 100× fewer params; speculative 2–3× faster; ~60% VRAM saving | marketing claims | DO_NOT_USE |
| H22 | Generic training-cost tables | 125M: $100–500 … 405B: $30M+ | planning | DO_NOT_USE |
| H23 | Inconsistent `large` size labels | ~333M (preset table), 350M (monitor banner/directory) | documentation | DO_NOT_USE |
| H30 | LoRA trainable parameters, Qwen2.5-7B | 5,046,272 (5.05M) of 7.62B (0.066%) | PEFT | CARRY_FORWARD |
| H31 | SFT dataset size | 139 (prompt, conditions) pairs with yield ≥ 0.88 from the first 400 records | SFT | CARRY_FORWARD |
| H32 | SFT loss per epoch | 1.6630 → 1.0433 → 1.0078 | SFT | CARRY_FORWARD |
| H33 | SFT adapter update norm | ‖ΔW‖_F = 1.727 over 112 LoRA modules (ΔW = (α/r)·B·A) | SFT | CARRY_FORWARD |
| H34 | DPO preference pairs | 80 (20 per molecule × 4 molecules in the train split) | DPO | CARRY_FORWARD |
| H35 | DPO loss (first → last step) | 0.69287 → 0.69189 over 80 steps | DPO | CARRY_FORWARD |
| H36 | DPO best epoch mean reward margin | 0.000343 | DPO | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H37 | DPO max per-step reward margin | 0.00815 (step index 62) | DPO | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H38 | DPO adapter displacement from SFT adapter | ‖ΔW_dpo − ΔW_sft‖_F = 0.256 (≈15% of ‖ΔW_sft‖); A matrices cosine 0.9999 | DPO | CARRY_FORWARD |
| H39 | Held-out record reward (DPO / PPO / rule baseline / MLP rows) | 0.80799 ± 0.01512 (min 0.7739, max 0.8567); 100/100 > 0.75; 100/100 > 0.6 | evaluation | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H40 | 'Improvement' percentages vs constant 0.5 | +61.6% (0.808), +16.0% (0.580), +62.4% (0.812), +63.6% (0.818), +62.8% (0.814) | evaluation | DO_NOT_USE |
| H41 | 'LLM-PPO matches DPO at 0.808' / charts showing identical bars | 0.808 for rule baseline, MLP-PPO, LLM-PPO, DPO | evaluation | DO_NOT_USE |
| H42 | Harness row 'LLM-PPO (Llama-2-7B + LoRA r=8)', train time 0.0 s | 0.808, 0.0 s | evaluation | DO_NOT_USE |
| H43 | PPO iterations completed on 7B LoRA policy | 200 iterations, world_size 1, adapters saved at 50/100/150/200 + best | PPO | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H44 | PPO best iteration reward | 0.9007 | PPO | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H45 | PPO logged iteration rewards | 0.8476 (it 10) … 0.7811 (it 190), 0.8621 (it 200); mean 0.850, range 0.7811–0.8793 | PPO | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H46 | PPO logged KL (detached old−reference estimator) | −0.0001 (it 10) → −0.0105 (it 200) | PPO | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H47 | PPO adapter update norms over training | ‖ΔW‖_F 0.118 (it 50) → 0.160 → 0.193 → 0.228 (it 200); best checkpoint 0.123; A matrices unrelated to SFT (fresh LoRA) | PPO | CARRY_FORWARD |
| H48 | PPO pilot (50 iterations, improvable records): best iteration reward | 0.7318 | PPO | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H49 | PPO pilot held-out record score | 0.5801 (26/100 > 0.6, 1/100 > 0.75) | evaluation | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H50 | PPO pilot losses | total 2.053 → 0.459; value 4.113 → 0.923; policy ~1e-5 | PPO | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H51 | Claim: 'PPO batch=2, 50 iterations shows ±0.13 variance' | ±0.13 | PPO | DO_NOT_USE |
| H52 | Claim: 'stable plateau 0.82–0.88 from iteration 10' | 0.82–0.88 | PPO | DO_NOT_USE |
| H53 | Agent GRPO best iteration reward | 0.5575 (iteration 180) | Agent GRPO / RLVR | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H54 | Agent GRPO reward trend | mean of first 20 iterations 0.109 → last 20 iterations 0.201; final iteration 0.2425 | Agent GRPO / RLVR | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H55 | Agent GRPO greedy accuracy before → after | 0.15 → 0.05 (3/20 → 1/20) | Agent GRPO / RLVR | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H56 | Agent GRPO tool-use rate before → after | 0.5 → 0.9 | Agent GRPO / RLVR | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H57 | Agent GRPO narrative points | iter 1: 0.29, iter 16: 0.41, iter 50: 0.28, iter 106: 0.4125 | Agent GRPO / RLVR | CARRY_FORWARD |
| H58 | Comparison-table Agent GRPO 'final 0.558 / peak 0.621 / convergence step 180' | 0.558 / 0.621 / 180 | Agent GRPO | DO_NOT_USE |
| H59 | Process-reward GRPO best reward | 1.0659 at iteration 1 (step-reward share 0.148) | PRM-GRPO | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H60 | Process-reward GRPO final iteration | 0.7822 (step-reward share 0.323) | PRM-GRPO | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H61 | Process-reward GRPO logged loss range | −11.21 … +7.64 | PRM-GRPO | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H62 | Comparison-table PRM-GRPO 'final 1.066 / peak 1.089 / convergence step 1' | 1.066 / 1.089 / 1 | PRM-GRPO | DO_NOT_USE |
| H63 | Hill climb rounds | round 0: accuracy 0.125 (4/32), avg reward 0.1375, dataset 32; rounds 1–3: accuracy 0.0, reward 0.0, dataset 44/32/32 | rejection sampling + agent GRPO | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H64 | Pairwise AI-feedback demo statistics | 10 prompts, 40 candidates, 60 comparisons, 60 DPO pairs; chosen 312 vs rejected 194 chars (gap 118) | RLAIF data generation | CARRY_FORWARD |
| H65 | Constitutional critique→revise demo statistics | 10 examples → 10 SFT + 10 DPO records; +53.0 chars after revision | RLAIF data generation | CARRY_FORWARD |
| H66 | GRPO (G=4) on synthesis records, final reward | 0.823 (peak 0.878) | GRPO | NEEDS_EVIDENCE_RECOVERY |
| H67 | RLAIF self-judge → DPO, final reward | 0.814 (peak 0.867; '100% > 0.75'; '+62.8%') | RLAIF | NEEDS_EVIDENCE_RECOVERY |
| H68 | STaR (iteration 3), final reward | 0.791 (peak 0.843) | STaR | NEEDS_EVIDENCE_RECOVERY |
| H69 | SFT warm-up 'final reward' | 0.412 | SFT | NEEDS_EVIDENCE_RECOVERY |
| H70 | Comparison-table 'peak 0.901' for DPO and PPO; 'convergence step' 63 (DPO) and 10 (PPO) | 0.901; 63; 10 | comparison script | DO_NOT_USE |
| H71 | Labeled expanded records: training-record reward | 0.85767 (identical in all 5 epochs) | tabular actor-critic PPO | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H72 | Labeled expanded records: held-out record reward | 0.81208 ± 0.02190 (min 0.7591, max 0.8851); 100/100 > 0.75 | evaluation | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H73 | Tabular PPO losses (labeled expanded) | total 0.208 → −0.034; value 0.486 → 0.0015; policy ≈ 1e-7 | tabular actor-critic PPO | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H74 | Literature records, earlier efficiency term: train mean / held-out mean / above-baseline | 0.84005 / 0.80107 / 1 of 100; −4.64% | tabular actor-critic PPO | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H75 | Improvable records, earlier efficiency term | baseline 0.60335 / held-out 0.59347 / 30 of 100 above baseline; −1.64% (docs: 0.594, −1.6%) | tabular actor-critic PPO | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H76 | Improvable records 'best_training' reward | 0.60464 | tabular actor-critic PPO | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H77 | First run (100 labeled records): train mean / held-out mean | 0.85588 / 0.81811 (20 records); policy loss 1.1e-16; value loss 0.0038; 20/20 > 0.75 | tabular PPO | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H78 | 'Training +50.6%, test −2.4% (overfitting)' | +50.6% / −2.4% | tabular PPO | NEEDS_EVIDENCE_RECOVERY |
| H79-ASP | Leave-one-molecule-out held-out record reward: Aspirin | 0.89659 (train records 0.82482) | evaluation | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H79-IBU | Leave-one-molecule-out held-out record reward: Ibuprofen | 0.85176 (train records 0.83603) | evaluation | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H79-NAP | Leave-one-molecule-out held-out record reward: Naproxen | 0.76134 (train records 0.85864) | evaluation | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H79-PAR | Leave-one-molecule-out held-out record reward: Paracetamol | 0.87821 (train records 0.82942) | evaluation | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H79-KET | Leave-one-molecule-out held-out record reward: Ketoprofen | 0.80799 (train records 0.84697) | evaluation | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H80 | Leave-one-molecule-out 'improvement' values | −8.0, −1.8, +12.8, −5.6, +4.8 % | evaluation | DO_NOT_USE |
| H81 | Harness MLP-PPO train time | 2.4 s | tabular PPO | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H82 | Literature records yield range | documented 0.73–0.93; actual 0.671–1.037 (8 records > 1.0, Aspirin/Paracetamol) | data | CARRY_FORWARD_WITH_INTERPRETATION_NOTE |
| H83 | Improvable records yield range / mean | 0.25–0.949; mean 0.360 (docs: 25%–95%, baseline yield 36%) | data | CARRY_FORWARD |
| H84 | Multi-GPU / Ray hardware statements | '3× A30 single node' torchrun and DeepSpeed commands; Ray cluster '4× A30' | infrastructure | DO_NOT_USE |

### 30.2 Full records

**H01 — Parameters, `small` preset (char vocab)**  
Value: 10.6M · Subsystem: model architecture · **CARRY_FORWARD**  
Model: small preset: 6 layers, 6 heads, 384-d, char vocab 65 · Dataset: Tiny Shakespeare (~1.1 MB, ~300K chars; data/input.txt) · Split: train/val by prepare script · Hardware: Apple M-series laptop CPU · Precision: fp32 · Batch: 32 · Seed: not recorded · Steps: 5,000 iterations · Hyper-parameters: small preset defaults (lr 3e-4 cosine as in train.py)  
Source: historical run report (character-level training results)  
Code path: core/config.py::MODEL_PRESETS['small'], models/transformer/model.py::TransformerLM; benchmarks/training/char_small_cpu  
Computation materially equivalent: YES — recomputed from the preset with vocab 65: 10.65M · Carries forward: YES · Interpretation changed: no · Rerun required: no  
Resume-safe wording: Trained a 10.6M-parameter character-level transformer from scratch.  
Limitations: Parameter count only.

**H02 — Best validation loss, char-level `small` model**  
Value: 1.479 (iteration 1,500 of 5,000) · Subsystem: pretraining · **CARRY_FORWARD**  
Model: small preset: 6 layers, 6 heads, 384-d, char vocab 65 · Dataset: Tiny Shakespeare (~1.1 MB, ~300K chars; data/input.txt) · Split: train/val by prepare script · Hardware: Apple M-series laptop CPU · Precision: fp32 · Batch: 32 · Seed: not recorded · Steps: 5,000 iterations · Hyper-parameters: small preset defaults (lr 3e-4 cosine as in train.py)  
Source: historical run report (character-level training results)  
Code path: training/pretrain.py::PretrainAlgorithm; benchmarks/training/char_small_cpu  
Computation materially equivalent: YES — token cross-entropy, AdamW grouping, cosine schedule, memmap random windows unchanged · Carries forward: YES · Interpretation changed: no · Rerun required: no  
Resume-safe wording: Reached 1.479 validation loss (character-level, 10.6M params, CPU, best at step 1,500 of 5,000).  
Limitations: Run report only; no per-step log or checkpoint retained. Best at step 1,500 means later steps overfit the ~1 MB corpus.

**H03 — Wall-clock, char-level training**  
Value: ~45 min · Subsystem: pretraining · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: small preset: 6 layers, 6 heads, 384-d, char vocab 65 · Dataset: Tiny Shakespeare (~1.1 MB, ~300K chars; data/input.txt) · Split: train/val by prepare script · Hardware: Apple M-series laptop CPU · Precision: fp32 · Batch: 32 · Seed: not recorded · Steps: 5,000 iterations · Hyper-parameters: small preset defaults (lr 3e-4 cosine as in train.py)  
Source: historical run report  
Code path: benchmarks/training/char_small_cpu  
Computation materially equivalent: n/a (timing of original code) · Carries forward: YES (approximate) · Interpretation changed: yes — approximate, hardware model unspecified · Rerun required: no  
Resume-safe wording: Trained in about 45 minutes on a laptop CPU.  
Limitations: Approximate; exact CPU model not recorded.

**H04 — Single-stream generation speed, `small`, KV cache, CPU**  
Value: 49 tokens/s · Subsystem: inference · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: small preset: 6 layers, 6 heads, 384-d, char vocab 65 · Dataset: Tiny Shakespeare (~1.1 MB, ~300K chars; data/input.txt) · Split: train/val by prepare script · Hardware: Apple M-series laptop CPU · Precision: fp32 · Batch: 32 · Seed: not recorded · Steps: 5,000 iterations · Hyper-parameters: small preset defaults (lr 3e-4 cosine as in train.py)  
Source: historical run report  
Code path: models/transformer/model.py::TransformerLM.prefill/step; evaluation/performance.py::measure_generation  
Computation materially equivalent: PARTIAL — same KV-cache algorithm, different implementation; not re-timed · Carries forward: YES (as historical measurement) · Interpretation changed: yes — speed belongs to the original implementation · Rerun required: no  
Resume-safe wording: Measured ~49 tokens/s single-stream generation with KV caching on CPU (historical run).  
Limitations: Single prompt, batch 1, unspecified length; not a throughput benchmark.

**H05 — Parameters, `large` preset (vocab 151,643, ctx 2048)**  
Value: 421.1M · Subsystem: model architecture · **CARRY_FORWARD**  
Model: large preset: 24 layers, 16 query / 4 KV heads, 1024-d, context 2048, Qwen2.5-0.5B tokenizer (vocab 151,643) · Dataset: FineWeb-Edu, ~1B tokens prepared, ~491M consumed · Split: train/val shards from data pipeline · Hardware: 1× NVIDIA A40 46 GB (rented cloud pod) · Precision: bf16 autocast, torch.compile · Batch: 8 × 2048 tokens (16,384 tokens/step) · Seed: not recorded · Steps: 30,000 · Hyper-parameters: AdamW lr 1.5e-4 cosine, warmup 200, weight decay 0.1, grad clip 1.0  
Source: historical model card and training walkthrough  
Code path: core/config.py::MODEL_PRESETS['large']; benchmarks/training/fineweb_edu_large_a40  
Computation materially equivalent: YES — recomputed: 421.15M · Carries forward: YES · Interpretation changed: no · Rerun required: no  
Resume-safe wording: Pretrained a 421M-parameter GQA transformer.  
Limitations: Directory name '350m' and a '350M' monitor banner are wrong; 421M is correct.

**H06 — Best validation loss, `large` on FineWeb-Edu**  
Value: 3.5834 (step 29,500) · Subsystem: pretraining · **CARRY_FORWARD**  
Model: large preset: 24 layers, 16 query / 4 KV heads, 1024-d, context 2048, Qwen2.5-0.5B tokenizer (vocab 151,643) · Dataset: FineWeb-Edu, ~1B tokens prepared, ~491M consumed · Split: train/val shards from data pipeline · Hardware: 1× NVIDIA A40 46 GB (rented cloud pod) · Precision: bf16 autocast, torch.compile · Batch: 8 × 2048 tokens (16,384 tokens/step) · Seed: not recorded · Steps: 30,000 · Hyper-parameters: AdamW lr 1.5e-4 cosine, warmup 200, weight decay 0.1, grad clip 1.0  
Source: historical training walkthrough (training metrics) and model card  
Code path: training/pretrain.py::PretrainAlgorithm; configs/training/pretrain_large_gpu.yaml; benchmarks/training/fineweb_edu_large_a40  
Computation materially equivalent: YES — objective, optimizer grouping, schedule and bf16 autocast unchanged; torch.compile now wired through trainer.compile_model · Carries forward: YES · Interpretation changed: no · Rerun required: no  
Resume-safe wording: Pretrained a 421M-parameter transformer on ~491M FineWeb-Edu tokens on one A40 to 3.58 validation loss.  
Limitations: Run report + model card; no step log or checkpoint retained. Undertrained relative to Chinchilla (~8.4B tokens).

**H07 — Final training loss, `large`**  
Value: 3.5478 (step 30,000) · Subsystem: pretraining · **CARRY_FORWARD**  
Model: large preset: 24 layers, 16 query / 4 KV heads, 1024-d, context 2048, Qwen2.5-0.5B tokenizer (vocab 151,643) · Dataset: FineWeb-Edu, ~1B tokens prepared, ~491M consumed · Split: train/val shards from data pipeline · Hardware: 1× NVIDIA A40 46 GB (rented cloud pod) · Precision: bf16 autocast, torch.compile · Batch: 8 × 2048 tokens (16,384 tokens/step) · Seed: not recorded · Steps: 30,000 · Hyper-parameters: AdamW lr 1.5e-4 cosine, warmup 200, weight decay 0.1, grad clip 1.0  
Source: historical training walkthrough  
Code path: training/pretrain.py  
Computation materially equivalent: YES · Carries forward: YES · Interpretation changed: no · Rerun required: no  
Resume-safe wording: Final train loss 3.55 after 30k steps.  
Limitations: As H06.

**H08 — Loss progression, `large`**  
Value: 12.12 (0) → 5.44 (1k) → 4.65 (3k) → 4.30 (5k) → 4.00 (10k) → 3.70 (20k) → 3.58 (30k) · Subsystem: pretraining · **CARRY_FORWARD**  
Model: large preset: 24 layers, 16 query / 4 KV heads, 1024-d, context 2048, Qwen2.5-0.5B tokenizer (vocab 151,643) · Dataset: FineWeb-Edu, ~1B tokens prepared, ~491M consumed · Split: train/val shards from data pipeline · Hardware: 1× NVIDIA A40 46 GB (rented cloud pod) · Precision: bf16 autocast, torch.compile · Batch: 8 × 2048 tokens (16,384 tokens/step) · Seed: not recorded · Steps: 30,000 · Hyper-parameters: AdamW lr 1.5e-4 cosine, warmup 200, weight decay 0.1, grad clip 1.0  
Source: historical training walkthrough (loss progression)  
Code path: benchmarks/training/fineweb_edu_large_a40/results.json::loss_progression  
Computation materially equivalent: YES · Carries forward: YES · Interpretation changed: no · Rerun required: no  
Resume-safe wording: Loss fell from 12.1 at initialisation to 3.58 over 30k steps.  
Limitations: Rounded values transcribed from the monitor.

**H09 — Tokens processed, `large`**  
Value: ~491M (30,000 × 16,384) · Subsystem: pretraining · **CARRY_FORWARD**  
Model: large preset: 24 layers, 16 query / 4 KV heads, 1024-d, context 2048, Qwen2.5-0.5B tokenizer (vocab 151,643) · Dataset: FineWeb-Edu, ~1B tokens prepared, ~491M consumed · Split: train/val shards from data pipeline · Hardware: 1× NVIDIA A40 46 GB (rented cloud pod) · Precision: bf16 autocast, torch.compile · Batch: 8 × 2048 tokens (16,384 tokens/step) · Seed: not recorded · Steps: 30,000 · Hyper-parameters: AdamW lr 1.5e-4 cosine, warmup 200, weight decay 0.1, grad clip 1.0  
Source: historical training walkthrough (configuration)  
Code path: configs/training/pretrain_large_gpu.yaml  
Computation materially equivalent: YES (arithmetic from configuration) · Carries forward: YES · Interpretation changed: no · Rerun required: no  
Resume-safe wording: Trained on ~491M tokens.  
Limitations: Derived from configuration.

**H10 — Average training throughput, `large`, A40**  
Value: 20,700 tokens/s · Subsystem: pretraining systems · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: large preset: 24 layers, 16 query / 4 KV heads, 1024-d, context 2048, Qwen2.5-0.5B tokenizer (vocab 151,643) · Dataset: FineWeb-Edu, ~1B tokens prepared, ~491M consumed · Split: train/val shards from data pipeline · Hardware: 1× NVIDIA A40 46 GB (rented cloud pod) · Precision: bf16 autocast, torch.compile · Batch: 8 × 2048 tokens (16,384 tokens/step) · Seed: not recorded · Steps: 30,000 · Hyper-parameters: AdamW lr 1.5e-4 cosine, warmup 200, weight decay 0.1, grad clip 1.0  
Source: historical training walkthrough (training metrics)  
Code path: Trainer + TransformerLM (not re-timed)  
Computation materially equivalent: PARTIAL — same architecture and precision; Forgeline's code not timed on an A40 · Carries forward: YES (historical) · Interpretation changed: yes — historical implementation · Rerun required: no  
Resume-safe wording: Sustained ~20.7k tokens/s training a 421M model on a single A40 (bf16, torch.compile).  
Limitations: Monitor reading, not a profiler measurement; single GPU only.

**H11 — Peak training throughput, `large`, A40**  
Value: 20,900 tokens/s · Subsystem: pretraining systems · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: large preset: 24 layers, 16 query / 4 KV heads, 1024-d, context 2048, Qwen2.5-0.5B tokenizer (vocab 151,643) · Dataset: FineWeb-Edu, ~1B tokens prepared, ~491M consumed · Split: train/val shards from data pipeline · Hardware: 1× NVIDIA A40 46 GB (rented cloud pod) · Precision: bf16 autocast, torch.compile · Batch: 8 × 2048 tokens (16,384 tokens/step) · Seed: not recorded · Steps: 30,000 · Hyper-parameters: AdamW lr 1.5e-4 cosine, warmup 200, weight decay 0.1, grad clip 1.0  
Source: historical training walkthrough  
Code path: as H10  
Computation materially equivalent: PARTIAL · Carries forward: YES (historical) · Interpretation changed: yes · Rerun required: no  
Resume-safe wording: Peak ~20.9k tokens/s.  
Limitations: As H10.

**H12 — torch.compile throughput effect, A40**  
Value: ~20,800 vs ~15,000 tokens/s (~39%) · Subsystem: pretraining systems · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: large preset: 24 layers, 16 query / 4 KV heads, 1024-d, context 2048, Qwen2.5-0.5B tokenizer (vocab 151,643) · Dataset: FineWeb-Edu, ~1B tokens prepared, ~491M consumed · Split: train/val shards from data pipeline · Hardware: 1× NVIDIA A40 46 GB (rented cloud pod) · Precision: bf16 autocast, torch.compile · Batch: 8 × 2048 tokens (16,384 tokens/step) · Seed: not recorded · Steps: 30,000 · Hyper-parameters: AdamW lr 1.5e-4 cosine, warmup 200, weight decay 0.1, grad clip 1.0  
Source: historical training walkthrough (torch.compile note)  
Code path: training/common/trainer.py::Trainer._compile_modules (restored in this audit)  
Computation materially equivalent: PARTIAL — compilation path restored; not re-measured · Carries forward: YES (historical observation) · Interpretation changed: yes — observation, not a controlled A/B · Rerun required: no  
Resume-safe wording: Observed ~39% higher throughput with torch.compile on the A40 run.  
Limitations: No controlled A/B; approximate numbers.

**H13 — Generation speed, `large`, KV cache, A40**  
Value: ~53 tokens/s · Subsystem: inference · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: large preset: 24 layers, 16 query / 4 KV heads, 1024-d, context 2048, Qwen2.5-0.5B tokenizer (vocab 151,643) · Dataset: FineWeb-Edu, ~1B tokens prepared, ~491M consumed · Split: train/val shards from data pipeline · Hardware: 1× NVIDIA A40 46 GB (rented cloud pod) · Precision: bf16 autocast, torch.compile · Batch: 8 × 2048 tokens (16,384 tokens/step) · Seed: not recorded · Steps: 30,000 · Hyper-parameters: AdamW lr 1.5e-4 cosine, warmup 200, weight decay 0.1, grad clip 1.0  
Source: historical training walkthrough (generation tests)  
Code path: TransformerLM.prefill/step  
Computation materially equivalent: PARTIAL · Carries forward: YES (historical) · Interpretation changed: yes · Rerun required: no  
Resume-safe wording: ~53 tokens/s single-stream generation on A40.  
Limitations: Single stream, temperature 0.7, top-k 50; not a serving benchmark.

**H14 — Wall-clock, `large` run**  
Value: ~14 hours · Subsystem: pretraining operations · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: large preset: 24 layers, 16 query / 4 KV heads, 1024-d, context 2048, Qwen2.5-0.5B tokenizer (vocab 151,643) · Dataset: FineWeb-Edu, ~1B tokens prepared, ~491M consumed · Split: train/val shards from data pipeline · Hardware: 1× NVIDIA A40 46 GB (rented cloud pod) · Precision: bf16 autocast, torch.compile · Batch: 8 × 2048 tokens (16,384 tokens/step) · Seed: not recorded · Steps: 30,000 · Hyper-parameters: AdamW lr 1.5e-4 cosine, warmup 200, weight decay 0.1, grad clip 1.0  
Source: historical model card  
Code path: benchmarks/training/fineweb_edu_large_a40  
Computation materially equivalent: n/a · Carries forward: YES (as elapsed time) · Interpretation changed: yes — includes evaluation, checkpointing and three incident recoveries; 491M tokens at 20.7k tok/s is ≈6.6 h of compute · Rerun required: no  
Resume-safe wording: The run took about 14 hours end to end including incident recovery.  
Limitations: Do not derive throughput from it.

**H15 — VRAM breakdown, `large` batch 8**  
Value: ~0.8 weights + 3.2 optimizer + 0.8 grads + 35 activations + 2 overhead ≈ 42 GB · Subsystem: memory · **DO_NOT_USE**  
Model: large preset: 24 layers, 16 query / 4 KV heads, 1024-d, context 2048, Qwen2.5-0.5B tokenizer (vocab 151,643) · Dataset: FineWeb-Edu, ~1B tokens prepared, ~491M consumed · Split: train/val shards from data pipeline · Hardware: 1× NVIDIA A40 46 GB (rented cloud pod) · Precision: bf16 autocast, torch.compile · Batch: 8 × 2048 tokens (16,384 tokens/step) · Seed: not recorded · Steps: 30,000 · Hyper-parameters: AdamW lr 1.5e-4 cosine, warmup 200, weight decay 0.1, grad clip 1.0  
Source: historical training walkthrough (VRAM estimate)  
Code path: —  
Computation materially equivalent: n/a · Carries forward: NO · Interpretation changed: — · Rerun required: no  
Resume-safe wording: Do not cite the breakdown; cite only that batch 8 × 2048 fit on a 46 GB GPU (H16).  
Limitations: Written estimate, internally inconsistent with bf16 autocast (parameters and gradients stay fp32 ≈ 1.7 GB each; AdamW moments ≈ 3.4 GB); activations figure is a guess, not profiler output.

**H16 — Memory feasibility, `large`**  
Value: batch 8 × 2048 tokens trained on a 46 GB A40 · Subsystem: memory · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: large preset: 24 layers, 16 query / 4 KV heads, 1024-d, context 2048, Qwen2.5-0.5B tokenizer (vocab 151,643) · Dataset: FineWeb-Edu, ~1B tokens prepared, ~491M consumed · Split: train/val shards from data pipeline · Hardware: 1× NVIDIA A40 46 GB (rented cloud pod) · Precision: bf16 autocast, torch.compile · Batch: 8 × 2048 tokens (16,384 tokens/step) · Seed: not recorded · Steps: 30,000 · Hyper-parameters: AdamW lr 1.5e-4 cosine, warmup 200, weight decay 0.1, grad clip 1.0  
Source: historical training walkthrough  
Code path: configs/training/pretrain_large_gpu.yaml  
Computation materially equivalent: YES (configuration) · Carries forward: YES · Interpretation changed: yes — feasibility, not a measured peak · Rerun required: no  
Resume-safe wording: Fit batch 8 × 2048-token sequences for a 421M model on one 46 GB GPU in bf16.  
Limitations: Peak memory not recorded.

**H17 — Prepared corpus size**  
Value: 1B tokens FineWeb-Edu (≈7.5 GB token files) · Subsystem: data pipeline · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: large preset: 24 layers, 16 query / 4 KV heads, 1024-d, context 2048, Qwen2.5-0.5B tokenizer (vocab 151,643) · Dataset: FineWeb-Edu, ~1B tokens prepared, ~491M consumed · Split: train/val shards from data pipeline · Hardware: 1× NVIDIA A40 46 GB (rented cloud pod) · Precision: bf16 autocast, torch.compile · Batch: 8 × 2048 tokens (16,384 tokens/step) · Seed: not recorded · Steps: 30,000 · Hyper-parameters: AdamW lr 1.5e-4 cosine, warmup 200, weight decay 0.1, grad clip 1.0  
Source: historical model card and incident notes  
Code path: data/streaming.py::write_shards_from_documents; cli `data prepare-hf`  
Computation materially equivalent: PARTIAL — constant-memory shard writer preserved · Carries forward: YES · Interpretation changed: yes — prepared, not consumed (491M used) · Rerun required: no  
Resume-safe wording: Built a constant-memory streaming pipeline that prepared a 1B-token FineWeb-Edu corpus.  
Limitations: Pipeline reliability incidents (OOM) documented, no timing.

**H18 — Checkpoint size, `large`**  
Value: 5.1 GB per full checkpoint · Subsystem: checkpointing operations · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: large preset: 24 layers, 16 query / 4 KV heads, 1024-d, context 2048, Qwen2.5-0.5B tokenizer (vocab 151,643) · Dataset: FineWeb-Edu, ~1B tokens prepared, ~491M consumed · Split: train/val shards from data pipeline · Hardware: 1× NVIDIA A40 46 GB (rented cloud pod) · Precision: bf16 autocast, torch.compile · Batch: 8 × 2048 tokens (16,384 tokens/step) · Seed: not recorded · Steps: 30,000 · Hyper-parameters: AdamW lr 1.5e-4 cosine, warmup 200, weight decay 0.1, grad clip 1.0  
Source: historical incident notes  
Code path: checkpoints/manager.py (atomic tmp+rename, rotation keep_last)  
Computation materially equivalent: n/a · Carries forward: YES · Interpretation changed: yes — operational fact behind the checkpoint design · Rerun required: no  
Resume-safe wording: Diagnosed checkpoint corruption on a network filesystem and disk exhaustion from 5.1 GB rotating checkpoints.  
Limitations: Operational observation.

**H19 — Expected losses and budgets in launch scripts**  
Value: val loss ~2.5–3.0 / ~2.0–2.5; ~$3–4; ETA ~2–3 h · Subsystem: planning · **DO_NOT_USE**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: historical launch scripts  
Code path: —  
Computation materially equivalent: n/a · Carries forward: NO · Interpretation changed: — · Rerun required: no  
Resume-safe wording: Never cite.  
Limitations: Expectations written before running; contradicted by the measured 3.58.

**H20 — Cloud tutorial throughput and example logs**  
Value: 61,411 tok/s sample log; 1×A100 ~60K, 4×A100 ~80K, 8×A100 ~100K tok/s · Subsystem: planning · **DO_NOT_USE**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: historical cloud tutorial  
Code path: —  
Computation materially equivalent: n/a · Carries forward: NO · Interpretation changed: — · Rerun required: no  
Resume-safe wording: Never cite.  
Limitations: Illustrative tables; no such runs exist.

**H21 — Feature speed/memory multipliers in feature table**  
Value: MLA ~10× smaller cache; SDPA 2–4× faster; NSA 9× faster; LoRA 100× fewer params; speculative 2–3× faster; ~60% VRAM saving · Subsystem: marketing claims · **DO_NOT_USE**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: historical feature table  
Code path: models/attention/*, models/generation/speculative.py  
Computation materially equivalent: n/a · Carries forward: NO · Interpretation changed: — · Rerun required: no  
Resume-safe wording: Never cite as measured; describe mechanisms instead.  
Limitations: Literature figures, never measured in either project.

**H22 — Generic training-cost tables**  
Value: 125M: $100–500 … 405B: $30M+ · Subsystem: planning · **DO_NOT_USE**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: historical data guide  
Code path: —  
Computation materially equivalent: n/a · Carries forward: NO · Interpretation changed: — · Rerun required: no  
Resume-safe wording: Never cite.  
Limitations: Generic external estimates.

**H23 — Inconsistent `large` size labels**  
Value: ~333M (preset table), 350M (monitor banner/directory) · Subsystem: documentation · **DO_NOT_USE**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: historical preset table and monitor banner  
Code path: —  
Computation materially equivalent: n/a · Carries forward: NO · Interpretation changed: — · Rerun required: no  
Resume-safe wording: Use 421M (H05).  
Limitations: Contradicted by exact count.

**H30 — LoRA trainable parameters, Qwen2.5-7B**  
Value: 5,046,272 (5.05M) of 7.62B (0.066%) · Subsystem: PEFT · **CARRY_FORWARD**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: — · Split: — · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: — · Seed: not recorded · Steps: — · Hyper-parameters: r=8, α=16, dropout 0.05, q/k/v/o_proj, PEFT 0.19.1  
Source: historical training-details table; saved adapter tensors  
Code path: models/policy.py::HFPolicy (PEFT LoraConfig r/α/targets); benchmarks/peft/lora_qwen2_5_7b  
Computation materially equivalent: YES — verified from the saved adapter (112 modules, 224 tensors) and arithmetic 28 × 8 × [(3584+3584)+(3584+512)×2+(3584+3584)] · Carries forward: YES · Interpretation changed: no · Rerun required: no  
Resume-safe wording: Post-trained Qwen2.5-7B-Instruct with LoRA adapters (5.05M trainable parameters, 0.066% of 7.62B).  
Limitations: Parameter accounting only.

**H31 — SFT dataset size**  
Value: 139 (prompt, conditions) pairs with yield ≥ 0.88 from the first 400 records · Subsystem: SFT · **CARRY_FORWARD**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: literature synthesis records (500 nested records, 5 molecules × 100, file order Aspirin, Ibuprofen, Paracetamol, Naproxen, Ketoprofen) · Split: records 0–399 (file order) · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: — · Seed: not recorded · Steps: — · Hyper-parameters: —  
Source: retained results file sft_results.json  
Code path: domains/synthesis/preferences.py::build_sft_records; training/factory.py (data.kind trajectories); tests/integration/test_huggingface_backend.py asserts 139  
Computation materially equivalent: YES — reproduced exactly · Carries forward: YES · Interpretation changed: no · Rerun required: no  
Resume-safe wording: Built a 139-example SFT set by yield filtering.  
Limitations: Small dataset.

**H32 — SFT loss per epoch**  
Value: 1.6630 → 1.0433 → 1.0078 · Subsystem: SFT · **CARRY_FORWARD**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: literature synthesis records (500 nested records, 5 molecules × 100, file order Aspirin, Ibuprofen, Paracetamol, Naproxen, Ketoprofen) · Split: 139 filtered train examples · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: 4 (script default; README table says 2) · Seed: not recorded · Steps: 3 epochs · Hyper-parameters: lr 2e-5  
Source: retained results file sft_results.json  
Code path: training/sft.py::SFTAlgorithm with HFPolicy; configs/post_training/synthesis_sft_qwen7b_lora.yaml  
Computation materially equivalent: YES — token-mean cross-entropy with prompt masking · Carries forward: YES · Interpretation changed: no · Rerun required: no  
Resume-safe wording: SFT reduced loss from 1.66 to 1.01 in 3 epochs on 139 examples.  
Limitations: Training loss on 139 examples; no held-out loss.

**H33 — SFT adapter update norm**  
Value: ‖ΔW‖_F = 1.727 over 112 LoRA modules (ΔW = (α/r)·B·A) · Subsystem: SFT · **CARRY_FORWARD**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: — · Split: — · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: — · Seed: not recorded · Steps: — · Hyper-parameters: —  
Source: saved SFT adapter tensors  
Code path: derived in this audit from the artifact  
Computation materially equivalent: n/a (artifact) · Carries forward: YES · Interpretation changed: no · Rerun required: no  
Resume-safe wording: The saved SFT adapter carries a non-trivial learned update (B initialised to zero).  
Limitations: Weight-space statistic; says nothing about quality.

**H34 — DPO preference pairs**  
Value: 80 (20 per molecule × 4 molecules in the train split) · Subsystem: DPO · **CARRY_FORWARD**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: literature synthesis records (500 nested records, 5 molecules × 100, file order Aspirin, Ibuprofen, Paracetamol, Naproxen, Ketoprofen) · Split: records 0–399 · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: — · Seed: not recorded · Steps: — · Hyper-parameters: —  
Source: retained results file dpo_results.json  
Code path: domains/synthesis/preferences.py::build_preference_pairs (seeded); tests assert pair counts  
Computation materially equivalent: YES — count reproduced (seeded sampling differs from the original global RNG) · Carries forward: YES · Interpretation changed: no · Rerun required: no  
Resume-safe wording: Built 80 yield-ranked preference pairs.  
Limitations: Pair membership differs from the original run because sampling is now seeded.

**H35 — DPO loss (first → last step)**  
Value: 0.69287 → 0.69189 over 80 steps · Subsystem: DPO · **CARRY_FORWARD**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: 80 pairs · Split: — · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: 4 (script default) · Seed: not recorded · Steps: 2 epochs (80 steps) · Hyper-parameters: β=0.1, lr 1e-5, mean log-prob reduction, reference = disable_adapter  
Source: retained results file dpo_results.json (step history)  
Code path: training/dpo.py::dpo_loss with logprob_reduction=mean; configs/post_training/synthesis_dpo_qwen7b_lora.yaml  
Computation materially equivalent: YES — length-normalised DPO with adapter-disabled reference · Carries forward: YES · Interpretation changed: no · Rerun required: no  
Resume-safe wording: Ran length-normalised DPO (β=0.1) from the SFT adapter on a 7B model.  
Limitations: Loss stayed ≈ ln 2: weak preference signal.

**H36 — DPO best epoch mean reward margin**  
Value: 0.000343 · Subsystem: DPO · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: — · Split: — · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: — · Seed: not recorded · Steps: — · Hyper-parameters: —  
Source: retained results file dpo_results.json  
Code path: training/dpo.py (metric reward_margin)  
Computation materially equivalent: YES · Carries forward: YES · Interpretation changed: yes — tiny margin: the preference signal barely separated chosen/rejected · Rerun required: no  
Resume-safe wording: Implicit reward margins remained near zero.  
Limitations: Do not present as improvement.

**H37 — DPO max per-step reward margin**  
Value: 0.00815 (step index 62) · Subsystem: DPO · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: — · Split: — · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: — · Seed: not recorded · Steps: — · Hyper-parameters: —  
Source: retained results file dpo_results.json (step history)  
Code path: training/dpo.py  
Computation materially equivalent: YES · Carries forward: YES · Interpretation changed: yes — single-batch maximum · Rerun required: no  
Resume-safe wording: —  
Limitations: Noise-level; used (mislabelled) as 'convergence step 63' in a comparison table.

**H38 — DPO adapter displacement from SFT adapter**  
Value: ‖ΔW_dpo − ΔW_sft‖_F = 0.256 (≈15% of ‖ΔW_sft‖); A matrices cosine 0.9999 · Subsystem: DPO · **CARRY_FORWARD**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: — · Split: — · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: — · Seed: not recorded · Steps: — · Hyper-parameters: —  
Source: saved SFT and DPO adapter tensors  
Code path: derived in this audit  
Computation materially equivalent: n/a (artifact) · Carries forward: YES · Interpretation changed: no · Rerun required: no  
Resume-safe wording: DPO measurably updated the SFT adapter even though logged margins stayed near zero.  
Limitations: Weight-space only.

**H39 — Held-out record reward (DPO / PPO / rule baseline / MLP rows)**  
Value: 0.80799 ± 0.01512 (min 0.7739, max 0.8567); 100/100 > 0.75; 100/100 > 0.6 · Subsystem: evaluation · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: none (records scored directly) · Dataset: literature synthesis records (500 nested records, 5 molecules × 100, file order Aspirin, Ibuprofen, Paracetamol, Naproxen, Ketoprofen) · Split: records 400–499 = all Ketoprofen · Hardware: any · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: retained results files dpo_results.json, rlhf_llm_200iter.json, benchmark_results.json; comparison chart; record-scoring evaluator  
Code path: evaluation/reward.py::RecordRewardSuite (policy_dependent=False); domains/synthesis/reward.py::rule_score; tests/unit/test_benchmark_semantics.py::test_heldout_record_reward_reproduces_recorded_statistics  
Computation materially equivalent: YES — reproduced exactly · Carries forward: YES · Interpretation changed: YES — it is the mean rule score of 100 dataset records (all Ketoprofen), independent of any policy · Rerun required: no  
Resume-safe wording: The held-out synthesis records average 0.808 under the rule reward (dataset statistic).  
Limitations: Cannot be attributed to DPO, PPO or any method.

**H40 — 'Improvement' percentages vs constant 0.5**  
Value: +61.6% (0.808), +16.0% (0.580), +62.4% (0.812), +63.6% (0.818), +62.8% (0.814) · Subsystem: evaluation · **DO_NOT_USE**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: record-scoring evaluator (constant baseline 0.5); retained results files and historical README  
Code path: —  
Computation materially equivalent: n/a · Carries forward: NO · Interpretation changed: — · Rerun required: no  
Resume-safe wording: Never cite.  
Limitations: Improvement over a hard-coded 0.5, not over a baseline run.

**H41 — 'LLM-PPO matches DPO at 0.808' / charts showing identical bars**  
Value: 0.808 for rule baseline, MLP-PPO, LLM-PPO, DPO · Subsystem: evaluation · **DO_NOT_USE**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: historical README, comparison chart, benchmark_results.json  
Code path: —  
Computation materially equivalent: n/a · Carries forward: NO · Interpretation changed: — · Rerun required: no  
Resume-safe wording: Never cite as a method comparison.  
Limitations: Identical because the evaluator scores the same dataset records.

**H42 — Harness row 'LLM-PPO (Llama-2-7B + LoRA r=8)', train time 0.0 s**  
Value: 0.808, 0.0 s · Subsystem: evaluation · **DO_NOT_USE**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: retained results file benchmark_results.json  
Code path: —  
Computation materially equivalent: n/a · Carries forward: NO · Interpretation changed: — · Rerun required: no  
Resume-safe wording: Never cite.  
Limitations: No Llama-2 run exists; time 0.0 shows nothing was trained.

**H43 — PPO iterations completed on 7B LoRA policy**  
Value: 200 iterations, world_size 1, adapters saved at 50/100/150/200 + best · Subsystem: PPO · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: literature synthesis records (500 nested records, 5 molecules × 100, file order Aspirin, Ibuprofen, Paracetamol, Naproxen, Ketoprofen) · Split: train records 0–399, held-out 400–499 · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: 4 trajectories/iteration · Seed: not recorded · Steps: 200 iterations · Hyper-parameters: ppo_epochs 3, grad-accum 4 (one optimizer step/iteration), lr 1e-5 Adam, clip 0.2, value 0.5, entropy 0.01, KL weight 0.1 (detached), max_new_tokens 128, temperature 0.7  
Source: retained results file rlhf_llm_200iter.json; saved adapter checkpoints  
Code path: training/ppo.py::PPOAlgorithm (sequence_mean ratio, kl_mode monitor, sampled entropy); configs/post_training/synthesis_ppo_qwen7b_lora.yaml  
Computation materially equivalent: YES up to a constant loss factor 0.75 (see normalisation analysis) · Carries forward: YES · Interpretation changed: yes — pipeline execution evidence, not quality evidence · Rerun required: no  
Resume-safe wording: Ran 200 iterations of value-head PPO on a Qwen2.5-7B LoRA policy on one GPU (generation, value head, clipped update, adapter-disabled reference).  
Limitations: Reward did not depend on the policy (H44).

**H44 — PPO best iteration reward**  
Value: 0.9007 · Subsystem: PPO · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: literature synthesis records (500 nested records, 5 molecules × 100, file order Aspirin, Ibuprofen, Paracetamol, Naproxen, Ketoprofen) · Split: train records 0–399, held-out 400–499 · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: 4 trajectories/iteration · Seed: not recorded · Steps: 200 iterations · Hyper-parameters: ppo_epochs 3, grad-accum 4 (one optimizer step/iteration), lr 1e-5 Adam, clip 0.2, value 0.5, entropy 0.01, KL weight 0.1 (detached), max_new_tokens 128, temperature 0.7  
Source: retained results file rlhf_llm_200iter.json  
Code path: domains/synthesis/reward.py::SynthesisRuleReward; tests/unit/test_benchmark_semantics.py::test_synthesis_rule_reward_ignores_generated_text_for_nested_records  
Computation materially equivalent: YES · Carries forward: YES · Interpretation changed: YES — best batch mean of rule scores of the 4 sampled records; reproduced statistically by sampling records (best-of-200 median 0.9014) · Rerun required: no  
Resume-safe wording: Do not cite as reward improvement.  
Limitations: Record statistic; the generated conditions never entered the reward.

**H45 — PPO logged iteration rewards**  
Value: 0.8476 (it 10) … 0.7811 (it 190), 0.8621 (it 200); mean 0.850, range 0.7811–0.8793 · Subsystem: PPO · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: literature synthesis records (500 nested records, 5 molecules × 100, file order Aspirin, Ibuprofen, Paracetamol, Naproxen, Ketoprofen) · Split: train records 0–399, held-out 400–499 · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: 4 trajectories/iteration · Seed: not recorded · Steps: 200 iterations · Hyper-parameters: ppo_epochs 3, grad-accum 4 (one optimizer step/iteration), lr 1e-5 Adam, clip 0.2, value 0.5, entropy 0.01, KL weight 0.1 (detached), max_new_tokens 128, temperature 0.7  
Source: retained results file rlhf_llm_200iter.json (log)  
Code path: benchmarks/post_training/ppo_synthesis  
Computation materially equivalent: YES · Carries forward: YES · Interpretation changed: YES — record sampling noise (simulated mean 0.848, spread matches) · Rerun required: no  
Resume-safe wording: —  
Limitations: Logged every 10 iterations, rounded.

**H46 — PPO logged KL (detached old−reference estimator)**  
Value: −0.0001 (it 10) → −0.0105 (it 200) · Subsystem: PPO · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: literature synthesis records (500 nested records, 5 molecules × 100, file order Aspirin, Ibuprofen, Paracetamol, Naproxen, Ketoprofen) · Split: train records 0–399, held-out 400–499 · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: 4 trajectories/iteration · Seed: not recorded · Steps: 200 iterations · Hyper-parameters: ppo_epochs 3, grad-accum 4 (one optimizer step/iteration), lr 1e-5 Adam, clip 0.2, value 0.5, entropy 0.01, KL weight 0.1 (detached), max_new_tokens 128, temperature 0.7  
Source: retained results file rlhf_llm_200iter.json (log)  
Code path: training/ppo.py kl_mode=monitor (masked_mean(old_lp − ref_lp))  
Computation materially equivalent: YES · Carries forward: YES · Interpretation changed: yes — sampled-token estimator can be negative; growing magnitude shows the adapter moved away from the base model · Rerun required: no  
Resume-safe wording: Monitored policy/reference divergence during PPO.  
Limitations: Not a true KL; sign not meaningful.

**H47 — PPO adapter update norms over training**  
Value: ‖ΔW‖_F 0.118 (it 50) → 0.160 → 0.193 → 0.228 (it 200); best checkpoint 0.123; A matrices unrelated to SFT (fresh LoRA) · Subsystem: PPO · **CARRY_FORWARD**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: literature synthesis records (500 nested records, 5 molecules × 100, file order Aspirin, Ibuprofen, Paracetamol, Naproxen, Ketoprofen) · Split: train records 0–399, held-out 400–499 · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: 4 trajectories/iteration · Seed: not recorded · Steps: 200 iterations · Hyper-parameters: ppo_epochs 3, grad-accum 4 (one optimizer step/iteration), lr 1e-5 Adam, clip 0.2, value 0.5, entropy 0.01, KL weight 0.1 (detached), max_new_tokens 128, temperature 0.7  
Source: saved PPO adapter checkpoints (iterations 50/100/150/200, best)  
Code path: derived in this audit  
Computation materially equivalent: n/a (artifact) · Carries forward: YES · Interpretation changed: no · Rerun required: no  
Resume-safe wording: PPO updates accumulated steadily in the LoRA weights across 200 iterations (~0.09 Frobenius per 50 iterations).  
Limitations: Weight-space only; the reward could not observe these changes.

**H48 — PPO pilot (50 iterations, improvable records): best iteration reward**  
Value: 0.7318 · Subsystem: PPO · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: improvable synthesis records (500 nested) · Split: train records 0–399, held-out 400–499 · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: 4 (per record file) · Seed: not recorded · Steps: 50 iterations · Hyper-parameters: ppo_epochs 3, grad-accum 4 (one optimizer step/iteration), lr 1e-5 Adam, clip 0.2, value 0.5, entropy 0.01, KL weight 0.1 (detached), max_new_tokens 128, temperature 0.7  
Source: retained results file rlhf_llm_distributed.json  
Code path: training/ppo.py  
Computation materially equivalent: YES up to constant factor · Carries forward: YES · Interpretation changed: yes — reward computed by a trained learned/preference blend on record text; generated conditions did not reach it · Rerun required: no  
Resume-safe wording: Ran a 50-iteration PPO pilot.  
Limitations: Record statistic under an unretained reward model.

**H49 — PPO pilot held-out record score**  
Value: 0.5801 (26/100 > 0.6, 1/100 > 0.75) · Subsystem: evaluation · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: improvable records 400–499 (all Ketoprofen) · Split: train records 0–399, held-out 400–499 · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: 4 trajectories/iteration · Seed: not recorded · Steps: 200 iterations · Hyper-parameters: ppo_epochs 3, grad-accum 4 (one optimizer step/iteration), lr 1e-5 Adam, clip 0.2, value 0.5, entropy 0.01, KL weight 0.1 (detached), max_new_tokens 128, temperature 0.7  
Source: retained results file rlhf_llm_distributed.json (evaluation)  
Code path: evaluation/reward.py::RecordRewardSuite  
Computation materially equivalent: NO for exact value (reward model weights not retained; rule score would be 0.6011) · Carries forward: YES (historical) · Interpretation changed: yes — policy-independent score from an unretained learned reward · Rerun required: no  
Resume-safe wording: —  
Limitations: Not reproducible; not a policy metric.

**H50 — PPO pilot losses**  
Value: total 2.053 → 0.459; value 4.113 → 0.923; policy ~1e-5 · Subsystem: PPO · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: literature synthesis records (500 nested records, 5 molecules × 100, file order Aspirin, Ibuprofen, Paracetamol, Naproxen, Ketoprofen) · Split: train records 0–399, held-out 400–499 · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: 4 trajectories/iteration · Seed: not recorded · Steps: 200 iterations · Hyper-parameters: ppo_epochs 3, grad-accum 4 (one optimizer step/iteration), lr 1e-5 Adam, clip 0.2, value 0.5, entropy 0.01, KL weight 0.1 (detached), max_new_tokens 128, temperature 0.7  
Source: retained results file rlhf_llm_distributed.json (history)  
Code path: training/ppo.py  
Computation materially equivalent: YES up to constant factor · Carries forward: YES · Interpretation changed: yes — decrease is the value head fitting rewards; the policy term is ~0 · Rerun required: no  
Resume-safe wording: —  
Limitations: Loss scale depends on normalisation.

**H51 — Claim: 'PPO batch=2, 50 iterations shows ±0.13 variance'**  
Value: ±0.13 · Subsystem: PPO · **DO_NOT_USE**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: historical README  
Code path: —  
Computation materially equivalent: n/a · Carries forward: NO · Interpretation changed: — · Rerun required: no  
Resume-safe wording: Never cite.  
Limitations: Pilot log shows iteration-mean range 0.525–0.732 (±0.10) and median within-batch std 0.065; batch size not recorded as 2.

**H52 — Claim: 'stable plateau 0.82–0.88 from iteration 10'**  
Value: 0.82–0.88 · Subsystem: PPO · **DO_NOT_USE**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: historical README and run notes  
Code path: —  
Computation materially equivalent: n/a · Carries forward: NO · Interpretation changed: — · Rerun required: no  
Resume-safe wording: Never cite.  
Limitations: Contradicted by the log (0.7811 at iteration 190) and the reward is a record statistic.

**H53 — Agent GRPO best iteration reward**  
Value: 0.5575 (iteration 180) · Subsystem: Agent GRPO / RLVR · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: GSM8K train problems converted to tool tasks (gsm8k_train_tool_dataset.jsonl) · Split: random problems per iteration; 20 held-out problems for greedy eval · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: 2 problems × G=4 rollouts · Seed: not recorded · Steps: 200 iterations · Hyper-parameters: lr 5e-6 AdamW, β (KL) 0.04, clip 0.2, grad-accum 4, max_new_tokens 256, temperature 0.9, ≤ tool turns per trainer constant  
Source: retained results file agent_grpo_results.json  
Code path: training/grpo.py::GRPOAlgorithm(agent=True); rollouts/agent.py::AgentRolloutEngine; configs/post_training/agent_grpo_gsm8k_qwen7b_lora.yaml  
Computation materially equivalent: YES up to constant factor 0.5 (see normalisation analysis) · Carries forward: YES · Interpretation changed: yes — mean over 8 rollouts; single-iteration maxima are noisy · Rerun required: no  
Resume-safe wording: Trained a tool-using 7B agent with GRPO and verifiable rewards on GSM8K (best batch reward 0.56).  
Limitations: Reward includes 0.1 partial credit for tool-call syntax and 0.02 per tag.

**H54 — Agent GRPO reward trend**  
Value: mean of first 20 iterations 0.109 → last 20 iterations 0.201; final iteration 0.2425 · Subsystem: Agent GRPO / RLVR · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: GSM8K train problems converted to tool tasks (gsm8k_train_tool_dataset.jsonl) · Split: random problems per iteration; 20 held-out problems for greedy eval · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: 2 problems × G=4 rollouts · Seed: not recorded · Steps: 200 iterations · Hyper-parameters: lr 5e-6 AdamW, β (KL) 0.04, clip 0.2, grad-accum 4, max_new_tokens 256, temperature 0.9, ≤ tool turns per trainer constant  
Source: retained results file agent_grpo_results.json (history)  
Code path: benchmarks/rlvr/agent_grpo_gsm8k_tools  
Computation materially equivalent: YES · Carries forward: YES · Interpretation changed: yes — rise is consistent with learning tool-call format (partial credit), not accuracy (H55) · Rerun required: no  
Resume-safe wording: Batch reward roughly doubled over 200 iterations, driven by tool-use format rather than correctness.  
Limitations: Noisy; small batches.

**H55 — Agent GRPO greedy accuracy before → after**  
Value: 0.15 → 0.05 (3/20 → 1/20) · Subsystem: Agent GRPO / RLVR · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: GSM8K train problems converted to tool tasks (gsm8k_train_tool_dataset.jsonl) · Split: random problems per iteration; 20 held-out problems for greedy eval · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: 2 problems × G=4 rollouts · Seed: not recorded · Steps: 200 iterations · Hyper-parameters: lr 5e-6 AdamW, β (KL) 0.04, clip 0.2, grad-accum 4, max_new_tokens 256, temperature 0.9, ≤ tool turns per trainer constant  
Source: retained results file agent_grpo_results.json (baseline, final_eval)  
Code path: evaluation/quality.py::AgentAccuracySuite  
Computation materially equivalent: YES · Carries forward: YES · Interpretation changed: yes — n=20; difference not statistically meaningful; no accuracy gain · Rerun required: no  
Resume-safe wording: Evaluated tool-use accuracy on held-out problems (no accuracy gain observed).  
Limitations: n=20 greedy.

**H56 — Agent GRPO tool-use rate before → after**  
Value: 0.5 → 0.9 · Subsystem: Agent GRPO / RLVR · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: GSM8K train problems converted to tool tasks (gsm8k_train_tool_dataset.jsonl) · Split: random problems per iteration; 20 held-out problems for greedy eval · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: 2 problems × G=4 rollouts · Seed: not recorded · Steps: 200 iterations · Hyper-parameters: lr 5e-6 AdamW, β (KL) 0.04, clip 0.2, grad-accum 4, max_new_tokens 256, temperature 0.9, ≤ tool turns per trainer constant  
Source: retained results file agent_grpo_results.json  
Code path: evaluation/quality.py::AgentAccuracySuite (tool_use_rate)  
Computation materially equivalent: YES · Carries forward: YES · Interpretation changed: yes — the policy learned to call the tool (reward shaping effect) · Rerun required: no  
Resume-safe wording: GRPO raised the agent's tool-call rate from 50% to 90% on held-out problems.  
Limitations: n=20.

**H57 — Agent GRPO narrative points**  
Value: iter 1: 0.29, iter 16: 0.41, iter 50: 0.28, iter 106: 0.4125 · Subsystem: Agent GRPO / RLVR · **CARRY_FORWARD**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: GSM8K train problems converted to tool tasks (gsm8k_train_tool_dataset.jsonl) · Split: random problems per iteration; 20 held-out problems for greedy eval · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: 2 problems × G=4 rollouts · Seed: not recorded · Steps: 200 iterations · Hyper-parameters: lr 5e-6 AdamW, β (KL) 0.04, clip 0.2, grad-accum 4, max_new_tokens 256, temperature 0.9, ≤ tool turns per trainer constant  
Source: historical README table (matches results file)  
Code path: benchmarks/rlvr/agent_grpo_gsm8k_tools/results.json::step_history  
Computation materially equivalent: YES · Carries forward: YES · Interpretation changed: no · Rerun required: no  
Resume-safe wording: —  
Limitations: Individual noisy iterations.

**H58 — Comparison-table Agent GRPO 'final 0.558 / peak 0.621 / convergence step 180'**  
Value: 0.558 / 0.621 / 180 · Subsystem: Agent GRPO · **DO_NOT_USE**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: historical README comparison table and comparison-script fallback values  
Code path: —  
Computation materially equivalent: n/a · Carries forward: NO · Interpretation changed: — · Rerun required: no  
Resume-safe wording: Never cite.  
Limitations: 0.558 is the best (not final) iteration; 0.621 does not occur in the log.

**H59 — Process-reward GRPO best reward**  
Value: 1.0659 at iteration 1 (step-reward share 0.148) · Subsystem: PRM-GRPO · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: GSM8K tool tasks · Split: random problems per iteration · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: batch_size problems × G=4 (batch size not recorded; script default 2) · Seed: not recorded · Steps: 30 iterations (script default 60) · Hyper-parameters: lr 5e-6 AdamW, γ 0.9, step weight 0.5, grad clip 1.0, loss −A·Σ_t log π (no clip, no KL; clip_ratio and kl_coeff arguments unused), max_new_tokens 256  
Source: retained results file prm_grpo_results.json  
Code path: rollouts/rewards/process.py::ProcessReward; training/grpo.py objective='reinforce'; configs/post_training/grpo_process_reward_tiny_cpu.yaml  
Computation materially equivalent: YES up to constant factor B and re-tokenisation (see analysis) · Carries forward: YES · Interpretation changed: yes — first iteration is sampled before any update; later iterations never exceeded it · Rerun required: no  
Resume-safe wording: Implemented step-level process rewards (verified arithmetic/code steps, γ-discounted) combined with final-answer rewards for GRPO.  
Limitations: No improvement over 30 iterations.

**H60 — Process-reward GRPO final iteration**  
Value: 0.7822 (step-reward share 0.323) · Subsystem: PRM-GRPO · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: GSM8K tool tasks · Split: random problems per iteration · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: batch_size problems × G=4 (batch size not recorded; script default 2) · Seed: not recorded · Steps: 30 iterations (script default 60) · Hyper-parameters: lr 5e-6 AdamW, γ 0.9, step weight 0.5, grad clip 1.0, loss −A·Σ_t log π (no clip, no KL; clip_ratio and kl_coeff arguments unused), max_new_tokens 256  
Source: retained results file prm_grpo_results.json (history)  
Code path: as H59  
Computation materially equivalent: YES · Carries forward: YES · Interpretation changed: yes — step-reward share rose while total reward fell · Rerun required: no  
Resume-safe wording: —  
Limitations: Noisy 30-iteration run.

**H61 — Process-reward GRPO logged loss range**  
Value: −11.21 … +7.64 · Subsystem: PRM-GRPO · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: GSM8K tool tasks · Split: random problems per iteration · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: batch_size problems × G=4 (batch size not recorded; script default 2) · Seed: not recorded · Steps: 30 iterations (script default 60) · Hyper-parameters: lr 5e-6 AdamW, γ 0.9, step weight 0.5, grad clip 1.0, loss −A·Σ_t log π (no clip, no KL; clip_ratio and kl_coeff arguments unused), max_new_tokens 256  
Source: retained results file prm_grpo_results.json (history)  
Code path: training/grpo.py::_reinforce_loss  
Computation materially equivalent: YES (scale differs by B) · Carries forward: YES · Interpretation changed: yes — unnormalised token-sum REINFORCE; magnitude tracks response lengths · Rerun required: no  
Resume-safe wording: —  
Limitations: Not comparable to normalised losses.

**H62 — Comparison-table PRM-GRPO 'final 1.066 / peak 1.089 / convergence step 1'**  
Value: 1.066 / 1.089 / 1 · Subsystem: PRM-GRPO · **DO_NOT_USE**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: historical README comparison table and comparison-script fallback values  
Code path: —  
Computation materially equivalent: n/a · Carries forward: NO · Interpretation changed: — · Rerun required: no  
Resume-safe wording: Never cite.  
Limitations: 1.066 is the first (best) iteration, not final; 1.089 does not occur in the log.

**H63 — Hill climb rounds**  
Value: round 0: accuracy 0.125 (4/32), avg reward 0.1375, dataset 32; rounds 1–3: accuracy 0.0, reward 0.0, dataset 44/32/32 · Subsystem: rejection sampling + agent GRPO · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: Qwen/Qwen2.5-7B-Instruct + PEFT LoRA r=8, α=16, dropout 0.05, q/k/v/o_proj · Dataset: GSM8K tool tasks (32 problems) · Split: same problems per round · Hardware: single CUDA GPU (world_size 1; GPU model not recorded — repository docs reference A30 nodes; data paths point to a cluster /storage volume) · Precision: fp16 autocast + GradScaler (trainer code) · Batch: rollouts 4/problem; GRPO batch 2 × G=4 · Seed: not recorded · Steps: rounds 0–3 recorded (script default 5), 50 GRPO iterations/round · Hyper-parameters: reward threshold 0.5, top-k 2/problem, max_new_tokens 200  
Source: retained results file hill_climb_results.json  
Code path: training/star.py::HillClimber; benchmarks/rlvr/hill_climb_gsm8k  
Computation materially equivalent: YES · Carries forward: YES · Interpretation changed: yes — negative result: threshold 0.5 retained almost nothing and accuracy collapsed · Rerun required: no  
Resume-safe wording: Implemented a rejection-sampling hill-climbing loop (negative result documented).  
Limitations: Tool-use 0.0 in every round.

**H64 — Pairwise AI-feedback demo statistics**  
Value: 10 prompts, 40 candidates, 60 comparisons, 60 DPO pairs; chosen 312 vs rejected 194 chars (gap 118) · Subsystem: RLAIF data generation · **CARRY_FORWARD**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: retained statistics file rlaif_stats.json  
Code path: training/rlaif.py::run_pairwise_rlaif; cli `rlaif pairwise`; tests/integration/test_cli_pipeline.py  
Computation materially equivalent: YES — reproduced exactly · Carries forward: YES · Interpretation changed: no (deterministic rule-based generator/judge) · Rerun required: no  
Resume-safe wording: Built pairwise AI-feedback preference generation (4 candidates → 6 comparisons per prompt).  
Limitations: Rule-based demo judge, no LLM.

**H65 — Constitutional critique→revise demo statistics**  
Value: 10 examples → 10 SFT + 10 DPO records; +53.0 chars after revision · Subsystem: RLAIF data generation · **CARRY_FORWARD**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: retained statistics file cai_stats.json  
Code path: training/rlaif.py::run_constitutional; cli `rlaif constitutional`  
Computation materially equivalent: YES — reproduced exactly · Carries forward: YES · Interpretation changed: no · Rerun required: no  
Resume-safe wording: Implemented a constitutional critique-and-revise data pipeline.  
Limitations: Rule-based demo model.

**H66 — GRPO (G=4) on synthesis records, final reward**  
Value: 0.823 (peak 0.878) · Subsystem: GRPO · **NEEDS_EVIDENCE_RECOVERY**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: historical README comparison table and comparison-script fallback values only  
Code path: training/grpo.py; configs/post_training/synthesis_grpo_qwen7b_lora.yaml  
Computation materially equivalent: unknown · Carries forward: NO (not until recovered) · Interpretation changed: if recovered: cannot be the committed train_grpo.py test output (that evaluation returns 0.8080 for any policy on this split) · Rerun required: rerun required to obtain evidence  
Resume-safe wording: Do not cite.  
Limitations: grpo_results.json never committed; no log, chart, checkpoint or W&B export.

**H67 — RLAIF self-judge → DPO, final reward**  
Value: 0.814 (peak 0.867; '100% > 0.75'; '+62.8%') · Subsystem: RLAIF · **NEEDS_EVIDENCE_RECOVERY**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: historical README comparison table and comparison-script fallback values only  
Code path: training/rlaif.py::RLAIFTrainer  
Computation materially equivalent: unknown · Carries forward: NO · Interpretation changed: if recovered: the RLAIF trainer outputs DPO loss per round, trains on GSM8K by default, and +62.8% is the record-evaluator formula · Rerun required: rerun required  
Resume-safe wording: Do not cite.  
Limitations: No results file; labels inconsistent with the trainer's outputs.

**H68 — STaR (iteration 3), final reward**  
Value: 0.791 (peak 0.843) · Subsystem: STaR · **NEEDS_EVIDENCE_RECOVERY**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: historical README comparison table and comparison-script fallback values only  
Code path: training/star.py::STaRTrainer  
Computation materially equivalent: unknown · Carries forward: NO · Interpretation changed: if recovered: STaRTrainer reports accuracy (baseline_acc, best_acc), not reward · Rerun required: rerun required  
Resume-safe wording: Do not cite.  
Limitations: No results file.

**H69 — SFT warm-up 'final reward'**  
Value: 0.412 · Subsystem: SFT · **NEEDS_EVIDENCE_RECOVERY**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: historical README comparison table and comparison-script fallback values only  
Code path: training/sft.py  
Computation materially equivalent: unknown · Carries forward: NO · Interpretation changed: sft_results.json has losses only; the value equals an agent-GRPO iteration reward (0.4125) to three decimals, possibly a transcription error · Rerun required: rerun/eval required  
Resume-safe wording: Do not cite.  
Limitations: No artifact.

**H70 — Comparison-table 'peak 0.901' for DPO and PPO; 'convergence step' 63 (DPO) and 10 (PPO)**  
Value: 0.901; 63; 10 · Subsystem: comparison script · **DO_NOT_USE**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: historical README comparison table and comparison-script fallback values  
Code path: —  
Computation materially equivalent: n/a · Carries forward: NO · Interpretation changed: — · Rerun required: no  
Resume-safe wording: Never cite.  
Limitations: 0.901 is PPO's best training reward copied to DPO (whose log has no reward); 63 is the index+1 of the max DPO margin; 10 is PPO's first logged iteration.

**H71 — Labeled expanded records: training-record reward**  
Value: 0.85767 (identical in all 5 epochs) · Subsystem: tabular actor-critic PPO · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: 128→256→256 shared MLP, 32-way softmax actor, value critic · Dataset: labeled expanded records (500 flat) · Split: 0–399 · Hardware: CPU · Precision: fp32 · Batch: 8 · Seed: not recorded · Steps: 5 epochs · Hyper-parameters: Adam lr 1e-4, clip 0.2, value 0.5, entropy 0.01  
Source: retained results file rlhf_results.json (training metrics)  
Code path: domains/synthesis/tabular_ppo.py::TabularPPOTrainer; data/samples/synthesis/trajectories_labeled_expanded.jsonl; tests/unit/test_benchmark_semantics.py::test_labeled_record_statistics_reproduce_recorded_values  
Computation materially equivalent: YES — reproduced exactly · Carries forward: YES · Interpretation changed: YES — mean rule score of training records; policy-independent · Rerun required: no  
Resume-safe wording: —  
Limitations: Not learning evidence.

**H72 — Labeled expanded records: held-out record reward**  
Value: 0.81208 ± 0.02190 (min 0.7591, max 0.8851); 100/100 > 0.75 · Subsystem: evaluation · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: 128→256→256 shared MLP, 32-way softmax actor, value critic · Dataset: labeled expanded records · Split: 400–499 · Hardware: CPU · Precision: fp32 · Batch: — · Seed: not recorded · Steps: — · Hyper-parameters: Adam lr 1e-4, clip 0.2, value 0.5, entropy 0.01  
Source: retained results file rlhf_results.json (evaluation)  
Code path: evaluation/reward.py::RecordRewardSuite; test as H71  
Computation materially equivalent: YES — reproduced exactly · Carries forward: YES · Interpretation changed: YES — dataset statistic · Rerun required: no  
Resume-safe wording: —  
Limitations: Policy-independent.

**H73 — Tabular PPO losses (labeled expanded)**  
Value: total 0.208 → −0.034; value 0.486 → 0.0015; policy ≈ 1e-7 · Subsystem: tabular actor-critic PPO · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: 128→256→256 shared MLP, 32-way softmax actor, value critic · Dataset: labeled expanded records · Split: — · Hardware: CPU · Precision: fp32 · Batch: 8 · Seed: not recorded · Steps: 5 epochs · Hyper-parameters: Adam lr 1e-4, clip 0.2, value 0.5, entropy 0.01  
Source: retained results file rlhf_results.json; mlp_curves chart  
Code path: domains/synthesis/tabular_ppo.py  
Computation materially equivalent: YES — ratio is 1 on the first pass by construction (as recorded) · Carries forward: YES · Interpretation changed: yes — critic fits rewards; actor gradient ~0 · Rerun required: no  
Resume-safe wording: —  
Limitations: No policy learning.

**H74 — Literature records, earlier efficiency term: train mean / held-out mean / above-baseline**  
Value: 0.84005 / 0.80107 / 1 of 100; −4.64% · Subsystem: tabular actor-critic PPO · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: 128→256→256 shared MLP, 32-way softmax actor, value critic · Dataset: literature synthesis records (500 nested records, 5 molecules × 100, file order Aspirin, Ibuprofen, Paracetamol, Naproxen, Ketoprofen) · Split: 0–399 / 400–499 · Hardware: CPU · Precision: fp32 · Batch: 8 · Seed: not recorded · Steps: 5 epochs · Hyper-parameters: Adam lr 1e-4, clip 0.2, value 0.5, entropy 0.01  
Source: retained results file rlhf_results_real.json  
Code path: tests/unit/test_benchmark_semantics.py::test_legacy_efficiency_term_reproduces_early_recorded_values  
Computation materially equivalent: YES — reproduced exactly with 0.1·(1 − steps/10) · Carries forward: YES · Interpretation changed: YES — record statistics under an earlier reward formula · Rerun required: no  
Resume-safe wording: —  
Limitations: Formula differs from the current rule reward.

**H75 — Improvable records, earlier efficiency term**  
Value: baseline 0.60335 / held-out 0.59347 / 30 of 100 above baseline; −1.64% (docs: 0.594, −1.6%) · Subsystem: tabular actor-critic PPO · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: 128→256→256 shared MLP, 32-way softmax actor, value critic · Dataset: improvable records (500 nested) · Split: 0–399 / 400–499 · Hardware: CPU · Precision: fp32 · Batch: 16 · Seed: not recorded · Steps: 10 epochs · Hyper-parameters: Adam lr 1e-4, clip 0.2, value 0.5, entropy 0.01  
Source: retained results files rlhf_results_production.json and ppo_fixed.json; historical notes  
Code path: test as H74  
Computation materially equivalent: YES — reproduced exactly · Carries forward: YES · Interpretation changed: YES — record statistics · Rerun required: no  
Resume-safe wording: —  
Limitations: Policy-independent.

**H76 — Improvable records 'best_training' reward**  
Value: 0.60464 · Subsystem: tabular actor-critic PPO · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: 128→256→256 shared MLP, 32-way softmax actor, value critic · Dataset: improvable records · Split: — · Hardware: CPU · Precision: fp32 · Batch: — · Seed: not recorded · Steps: — · Hyper-parameters: Adam lr 1e-4, clip 0.2, value 0.5, entropy 0.01  
Source: retained results file ppo_fixed.json  
Code path: domains/synthesis/tabular_ppo.py  
Computation materially equivalent: not re-derived · Carries forward: YES (historical) · Interpretation changed: yes — 0.2% above the train-record mean; not an improvement · Rerun required: no  
Resume-safe wording: —  
Limitations: Computation not re-derived.

**H77 — First run (100 labeled records): train mean / held-out mean**  
Value: 0.85588 / 0.81811 (20 records); policy loss 1.1e-16; value loss 0.0038; 20/20 > 0.75 · Subsystem: tabular PPO · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: 128→256→256 shared MLP, 32-way softmax actor, value critic · Dataset: labeled records (100 flat) · Split: 0–79 / 80–99 · Hardware: CPU · Precision: fp32 · Batch: — · Seed: not recorded · Steps: 3 epochs · Hyper-parameters: Adam lr 1e-4, clip 0.2, value 0.5, entropy 0.01  
Source: earliest retained version of rlhf_results.json  
Code path: data/samples/synthesis/trajectories_labeled_small.jsonl; test as H71  
Computation materially equivalent: YES — reproduced exactly · Carries forward: YES · Interpretation changed: YES — record statistics; advantages identically zero · Rerun required: no  
Resume-safe wording: —  
Limitations: Historical first version.

**H78 — 'Training +50.6%, test −2.4% (overfitting)'**  
Value: +50.6% / −2.4% · Subsystem: tabular PPO · **NEEDS_EVIDENCE_RECOVERY**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: early historical README revisions  
Code path: —  
Computation materially equivalent: unknown · Carries forward: NO · Interpretation changed: no results file contains these values (closest retained: −4.64% in H74) · Rerun required: no (superseded)  
Resume-safe wording: Do not cite.  
Limitations: README-only.

**H79-ASP — Leave-one-molecule-out held-out record reward: Aspirin**  
Value: 0.89659 (train records 0.82482) · Subsystem: evaluation · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: 128→256→256 shared MLP, 32-way softmax actor, value critic · Dataset: literature synthesis records (500 nested records, 5 molecules × 100, file order Aspirin, Ibuprofen, Paracetamol, Naproxen, Ketoprofen) · Split: train: other 4 molecules; test: Aspirin · Hardware: CPU · Precision: fp32 · Batch: — · Seed: not recorded · Steps: — · Hyper-parameters: Adam lr 1e-4, clip 0.2, value 0.5, entropy 0.01  
Source: retained results file generalization_results.json  
Code path: examples/05_synthesis_generalization.py; tests/unit/test_benchmark_semantics.py::test_leave_one_molecule_out_reproduces_recorded_values  
Computation materially equivalent: YES — reproduced exactly · Carries forward: YES · Interpretation changed: YES — mean rule score of that molecule's 100 records; not policy generalisation · Rerun required: no  
Resume-safe wording: —  
Limitations: Dataset statistic.

**H79-IBU — Leave-one-molecule-out held-out record reward: Ibuprofen**  
Value: 0.85176 (train records 0.83603) · Subsystem: evaluation · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: 128→256→256 shared MLP, 32-way softmax actor, value critic · Dataset: literature synthesis records (500 nested records, 5 molecules × 100, file order Aspirin, Ibuprofen, Paracetamol, Naproxen, Ketoprofen) · Split: train: other 4 molecules; test: Ibuprofen · Hardware: CPU · Precision: fp32 · Batch: — · Seed: not recorded · Steps: — · Hyper-parameters: Adam lr 1e-4, clip 0.2, value 0.5, entropy 0.01  
Source: retained results file generalization_results.json  
Code path: examples/05_synthesis_generalization.py; tests/unit/test_benchmark_semantics.py::test_leave_one_molecule_out_reproduces_recorded_values  
Computation materially equivalent: YES — reproduced exactly · Carries forward: YES · Interpretation changed: YES — mean rule score of that molecule's 100 records; not policy generalisation · Rerun required: no  
Resume-safe wording: —  
Limitations: Dataset statistic.

**H79-NAP — Leave-one-molecule-out held-out record reward: Naproxen**  
Value: 0.76134 (train records 0.85864) · Subsystem: evaluation · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: 128→256→256 shared MLP, 32-way softmax actor, value critic · Dataset: literature synthesis records (500 nested records, 5 molecules × 100, file order Aspirin, Ibuprofen, Paracetamol, Naproxen, Ketoprofen) · Split: train: other 4 molecules; test: Naproxen · Hardware: CPU · Precision: fp32 · Batch: — · Seed: not recorded · Steps: — · Hyper-parameters: Adam lr 1e-4, clip 0.2, value 0.5, entropy 0.01  
Source: retained results file generalization_results.json  
Code path: examples/05_synthesis_generalization.py; tests/unit/test_benchmark_semantics.py::test_leave_one_molecule_out_reproduces_recorded_values  
Computation materially equivalent: YES — reproduced exactly · Carries forward: YES · Interpretation changed: YES — mean rule score of that molecule's 100 records; not policy generalisation · Rerun required: no  
Resume-safe wording: —  
Limitations: Dataset statistic.

**H79-PAR — Leave-one-molecule-out held-out record reward: Paracetamol**  
Value: 0.87821 (train records 0.82942) · Subsystem: evaluation · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: 128→256→256 shared MLP, 32-way softmax actor, value critic · Dataset: literature synthesis records (500 nested records, 5 molecules × 100, file order Aspirin, Ibuprofen, Paracetamol, Naproxen, Ketoprofen) · Split: train: other 4 molecules; test: Paracetamol · Hardware: CPU · Precision: fp32 · Batch: — · Seed: not recorded · Steps: — · Hyper-parameters: Adam lr 1e-4, clip 0.2, value 0.5, entropy 0.01  
Source: retained results file generalization_results.json  
Code path: examples/05_synthesis_generalization.py; tests/unit/test_benchmark_semantics.py::test_leave_one_molecule_out_reproduces_recorded_values  
Computation materially equivalent: YES — reproduced exactly · Carries forward: YES · Interpretation changed: YES — mean rule score of that molecule's 100 records; not policy generalisation · Rerun required: no  
Resume-safe wording: —  
Limitations: Dataset statistic.

**H79-KET — Leave-one-molecule-out held-out record reward: Ketoprofen**  
Value: 0.80799 (train records 0.84697) · Subsystem: evaluation · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: 128→256→256 shared MLP, 32-way softmax actor, value critic · Dataset: literature synthesis records (500 nested records, 5 molecules × 100, file order Aspirin, Ibuprofen, Paracetamol, Naproxen, Ketoprofen) · Split: train: other 4 molecules; test: Ketoprofen · Hardware: CPU · Precision: fp32 · Batch: — · Seed: not recorded · Steps: — · Hyper-parameters: Adam lr 1e-4, clip 0.2, value 0.5, entropy 0.01  
Source: retained results file generalization_results.json  
Code path: examples/05_synthesis_generalization.py; tests/unit/test_benchmark_semantics.py::test_leave_one_molecule_out_reproduces_recorded_values  
Computation materially equivalent: YES — reproduced exactly · Carries forward: YES · Interpretation changed: YES — mean rule score of that molecule's 100 records; not policy generalisation · Rerun required: no  
Resume-safe wording: —  
Limitations: Dataset statistic.

**H80 — Leave-one-molecule-out 'improvement' values**  
Value: −8.0, −1.8, +12.8, −5.6, +4.8 % · Subsystem: evaluation · **DO_NOT_USE**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: retained results file generalization_results.json (improvement fields)  
Code path: —  
Computation materially equivalent: n/a · Carries forward: NO · Interpretation changed: — · Rerun required: no  
Resume-safe wording: Never cite as improvement.  
Limitations: It is the relative gap between two record means.

**H81 — Harness MLP-PPO train time**  
Value: 2.4 s · Subsystem: tabular PPO · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: 128→256→256 shared MLP, 32-way softmax actor, value critic · Dataset: — · Split: — · Hardware: CPU · Precision: fp32 · Batch: — · Seed: not recorded · Steps: — · Hyper-parameters: Adam lr 1e-4, clip 0.2, value 0.5, entropy 0.01  
Source: retained results file benchmark_results.json  
Code path: domains/synthesis/tabular_ppo.py  
Computation materially equivalent: n/a · Carries forward: YES (historical) · Interpretation changed: yes — CPU of unknown type, 400 records × few epochs · Rerun required: no  
Resume-safe wording: —  
Limitations: Trivial timing.

**H82 — Literature records yield range**  
Value: documented 0.73–0.93; actual 0.671–1.037 (8 records > 1.0, Aspirin/Paracetamol) · Subsystem: data · **CARRY_FORWARD_WITH_INTERPRETATION_NOTE**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: historical README notes; the literature records file  
Code path: data/samples/synthesis/trajectories_literature.jsonl  
Computation materially equivalent: YES (same file) · Carries forward: YES (actual range) · Interpretation changed: yes — documented range is wrong; some yields exceed 1.0 (data-quality issue) · Rerun required: no  
Resume-safe wording: —  
Limitations: Yields > 1 are physically invalid.

**H83 — Improvable records yield range / mean**  
Value: 0.25–0.949; mean 0.360 (docs: 25%–95%, baseline yield 36%) · Subsystem: data · **CARRY_FORWARD**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: retained results file rlhf_results_production.json (data section)  
Code path: data/samples/synthesis/trajectories_improvable.jsonl  
Computation materially equivalent: YES · Carries forward: YES · Interpretation changed: no · Rerun required: no  
Resume-safe wording: —  
Limitations: Simulated records.

**H84 — Multi-GPU / Ray hardware statements**  
Value: '3× A30 single node' torchrun and DeepSpeed commands; Ray cluster '4× A30' · Subsystem: infrastructure · **DO_NOT_USE**  
Model: — · Dataset: — · Split: — · Hardware: — · Precision: — · Batch: — · Seed: — · Steps: — · Hyper-parameters: —  
Source: historical README multi-GPU section, cluster configuration file and distributed launch script  
Code path: distributed/strategies.py; orchestration/ray_backend.py  
Computation materially equivalent: n/a · Carries forward: NO · Interpretation changed: — · Rerun required: no  
Resume-safe wording: Never cite as executed.  
Limitations: Every retained run records world_size 1; the Ray entry script (train_ppo_ray.py) never existed; the --deepspeed flag was parsed but deepspeed was never initialised.

---

## 31. Historical benchmark carry-forward analysis

Mapping of every recorded run to its current reproduction path and the reason its numbers carry.

| Original run (artifact) | Original implementation | Forgeline implementation | Manifest | Semantic equivalence | Ledger ids |
|---|---|---|---|---|---|
| Char-level `small` pretraining (run report) | original training script | `training/pretrain.py` + `Trainer` | `configs/training/pretrain_small_char.yaml` | same CE objective, AdamW grouping, cosine LR, random memmap windows | H01–H04 |
| FineWeb-Edu `large` on A40 (walkthrough + model card) | original training script with `--compile`, bf16, checkpoint manager | `training/pretrain.py`, `Trainer._compile_modules`, `Precision`, `CheckpointManager` | `configs/training/pretrain_large_gpu.yaml` | same objective/optimizer/schedule/precision; compile restored; throughput not re-timed | H05–H18 |
| Qwen SFT (`sft_results.json`) | original SFT warm-up script | `training/sft.py` + `HFPolicy` + `build_sft_records` | `configs/post_training/synthesis_sft_qwen7b_lora.yaml` | token-mean CE, prompt masked, same 139 examples | H31–H33 |
| Qwen DPO (`dpo_results.json`) | original DPO trainer | `training/dpo.py` (`mean`) | `configs/post_training/synthesis_dpo_qwen7b_lora.yaml` | same loss, β, reference; pairs seeded | H34–H39 |
| Qwen value-head PPO 200 it (`rlhf_llm_200iter.json`) | original LLM PPO trainer | `training/ppo.py` (`sequence_mean`, `monitor`, `sampled_logprob`) | `configs/post_training/synthesis_ppo_qwen7b_lora.yaml` | same objective; loss factor 0.75 (§34); reward policy-independent (§32) | H43–H47 |
| PPO pilot 50 it (`rlhf_llm_distributed.json`) | same + learned/preference reward | same + `BlendedReward` | same manifest with improvable data | same; reward weights not retained | H48–H51 |
| Agent GRPO GSM8K (`agent_grpo_results.json`) | original agent GRPO trainer | `training/grpo.py` agent mode + `rollouts/agent.py` | `configs/post_training/agent_grpo_gsm8k_qwen7b_lora.yaml` | same segment-mean log-probs, clipped ratio, clamped KL, reward schedule; loss factor 0.5 | H53–H57 |
| Process-reward GRPO (`prm_grpo_results.json`) | original PRM-GRPO script | `ProcessReward` + GRPO `reinforce` | `configs/post_training/grpo_process_reward_tiny_cpu.yaml` (HF variant via `backend`) | same reward and REINFORCE loss; factor B and generated-ids vs re-tokenised text | H59–H61 |
| Hill climb (`hill_climb_results.json`) | original hill-climb loop | `training/star.py::HillClimber` | — | same loop | H63 |
| Pairwise RLAIF / CAI (`rlaif_stats.json`, `cai_stats.json`) | original RLAIF and CAI scripts | `training/rlaif.py` | CLI | reproduced exactly | H64, H65 |
| Tabular PPO and record statistics (`rlhf_results*.json`, `ppo_fixed.json`, `benchmark_results.json`, `generalization_results.json`) | original tabular PPO, generalisation and benchmark scripts | `domains/synthesis/tabular_ppo.py`, `evaluation/reward.py`, `examples/05_*` | CLI | all statistics reproduced exactly (current and legacy reward terms) | H71–H81 |

No number was discarded because of renaming, module moves, class renames, or CPU-only final validation. Numbers were downgraded only for content reasons (§32–§34).

---

## 32. Metrics requiring interpretation

### 32.1 The 0.808 "held-out reward" (H39)

* **Evaluator:** the original record-scoring evaluator `evaluate_rlhf_policy(test_trajectories, reward_model)` → `reward_model.score_batch(test_trajectories)`; it never calls the policy. Current equivalent: `evaluation/reward.py::RecordRewardSuite` with `details.policy_dependent = False`.
* **Dataset/split:** `data/samples/synthesis/trajectories_literature.jsonl`, `split = int(500 × 0.8)` → records 400–499, which are the 100 Ketoprofen records because the file is ordered by molecule.
* **Formula:** `0.40·yield + 0.30·selectivity + 0.20·(1 − safety_risk) + 0.10/(1 + steps/10)` clipped to [0, 1], applied to each record's stored `outcomes`. Mean 0.8079880769, std 0.0151181505, min 0.7739, max 0.8567, 100/100 > 0.75 — reproduced exactly by `tests/unit/test_benchmark_semantics.py::test_heldout_record_reward_reproduces_recorded_statistics`.
* **Why identical across methods:** every original training and benchmark script evaluates the same 100 records with the rule reward (or a reward model that falls back to the rule when untrained). The "improvement" is `(0.808 − 0.5)/0.5` against a constant.
* **Policy independence:** confirmed from code and reproduced without any model; additionally the rule reward ignores generated conditions for nested records (`test_synthesis_rule_reward_ignores_generated_text_for_nested_records`).
* **Safe interpretation:** "the held-out synthesis records average 0.808 under the rule reward" — a dataset statistic that describes the evaluation set, useful as the ceiling/floor context for the task, never as a method result.

### 32.2 PPO training rewards (H44–H45, H48)

The 7B policy generates JSON *conditions* (temperature, time, catalyst, solvent); `_compute_rewards` merges them into the record (`{**traj, **cond}`) but `_rule_score` reads `outcomes.yield/selectivity/safety_risk/steps`, which the merge cannot change (nested `outcomes` win). Therefore each iteration's reward is the mean rule score of the 4 sampled training records. Simulation (this audit): sampling 4 random records from 0–399 for 200 iterations reproduces the logged statistics — mean of iteration rewards 0.848 vs 0.850 recorded, best-of-200 median 0.9014 with P(≥ 0.9007) = 0.56 vs the recorded 0.9007. Interpretation: the run is evidence that the 7B LoRA PPO pipeline executed (and the adapters show the optimizer moved them, H47), not that the policy improved. The pilot's held-out 0.5801 came from a trained learned reward whose weights were not retained (rule score would be 0.6011).

### 32.3 Other interpretation notes

* **DPO** (H35–H37): loss ≈ ln 2 and margins ≈ 3×10⁻⁴ mean the preference signal was weak; the adapter still moved (H38).
* **Agent GRPO** (H53–H56): 8 samples per iteration → noisy maxima; accuracy 3/20 → 1/20 is not a meaningful change; tool-use 0.5 → 0.9 is the clearest learned behaviour, driven by the 0.1 partial credit for tool syntax.
* **Process-reward GRPO** (H59–H61): best value is iteration 1 (pre-update); rewards > 1 by construction; loss magnitude is length-dependent.
* **Hill climb** (H63): negative result (threshold too high).
* **A40 throughput/wall clock** (H10–H14): monitor readings of the original implementation; 14 h includes incidents; no controlled compile A/B.
* **Tabular PPO / LOO** (H71–H81): all dataset statistics; the identical epoch rewards and ~0 policy loss show no learning signal reached the actor.
* **Parameter labels** (H05/H23): the `large` model is 421M, not 350M/333M.

---

## 33. Metrics requiring evidence recovery

After searching every retained artifact and historical revision (results files, README revisions, comparison-script fallbacks, chart images, deleted files, model artifacts), the following remain documented-only:

| ID | Value | What was searched | What exists instead | Recovery path |
|---|---|---|---|---|
| H66 GRPO synthesis | 0.823 (peak 0.878) | `0.823`/`823` across every revision; `grpo_results.json` (never saved); charts; change notes | historical README tables and comparison-script fallback values only; data files coincidentally contain 0.823 as selectivity/yield values | run `configs/post_training/synthesis_grpo_qwen7b_lora.yaml` with a condition-aware reward (§42) |
| H67 RLAIF | 0.814 (peak 0.867) | same | same tables; trainer emits DPO loss, not reward | run `RLAIFTrainer` on GPU and report loss/pair statistics honestly |
| H68 STaR | 0.791 (peak 0.843) | same | same tables; trainer emits accuracy | run `STaRTrainer` on GSM8K and report accuracy |
| H69 SFT "reward" | 0.412 | same | same tables; equals an agent-GRPO iteration reward 0.4125 | evaluate the saved SFT adapter with a policy-dependent reward |
| H78 early "+50.6% / −2.4%" | — | early README revisions | superseded by retained −4.64%/−1.64% results | none needed (superseded) |

These are not called fake: the artifacts that would support them were not saved. Until recovered they are excluded from every summary and the dashboard shows them as *not published*.

---

## 34. Metrics requiring revalidation

None. The only computational differences between the recorded runs and Forgeline are constant loss-normalisation factors and one tokenisation path; analysed below, they do not change what the runs measured.

**Where the factor occurs.** Forgeline's `Trainer` averages `loss/n_micro` over accumulation micro-steps and the algorithms average over PPO epochs / groups. The recorded scripts summed: PPO summed three epoch losses each divided by `gradient_accumulation_steps = 4` before a single `optimizer.step()` → gradient `0.75·∇L̄` vs Forgeline's `∇L̄`; agent GRPO divided each trajectory loss by `G·4` and summed over `2` problems → `0.5·∇L̄`; process-reward GRPO summed `−A·Σ_t log π` over all `B·G` rollouts without normalisation → `B·∇L̄` relative to Forgeline's per-group mean.

**Scalar vs gradients.** A constant `c > 0` scales both the reported loss and every gradient by `c`; the direction is unchanged.

**Optimizer.** All three runs used Adam/AdamW. With `g_t = c·ĝ_t`: `m_t = c·m̂_t`, `v_t = c²·v̂_t`, so the update `lr·m̂_t/(√v̂_t + ε)` is unchanged except through `ε` (1e-8, negligible against gradient magnitudes of order 1e-3–1e-1 seen in the logs). Decoupled weight decay does not see the gradient. Learning-rate schedules do not depend on the loss scale.

**Clipping.** `clip_grad_norm_(max_norm = 1.0)` rescales when `‖g‖ > 1`; with `‖g‖ = c‖ĝ‖` the clip activates at different steps when `‖ĝ‖` straddles `1/c`. When clipping is active in both, the clipped gradient `g/‖g‖` is identical; when inactive in both, Adam cancels `c`. Only steps where exactly one of the two clips differ, and then by a factor between `c` and 1 in Adam's pre-normalised statistics — a second-order effect on the trajectory.

**PPO epochs.** In the recorded PPO the parameters did not change between the three inner epochs (one optimizer step per iteration), so the ratio `exp(mean(log π_new − log π_old))` was exactly 1 and the clipped term inactive; Forgeline's `loss` also evaluates the epochs before one step. Identical structure.

**Tokenisation path (process-reward run).** The original recomputed log-probs by re-tokenising decoded text; Forgeline uses generated token ids. For tokenisers where decode∘encode is not the identity the log-prob targets differ slightly. This changes the exact gradient but not the objective.

**Evaluation attribution.** Evaluation metrics in these runs are either record statistics (PPO) or greedy accuracy/tool-use from the run's own script (agent GRPO), none of which depend on the loss scale.

**Conclusion.** Reported loss magnitudes differ by known constants; learned parameters and results are expected to match up to `ε` and clip-timing effects. Classified CARRY_FORWARD_WITH_INTERPRETATION_NOTE with `revalidation_recommended: true` in `benchmarks/` (a rerun is useful to obtain a Forgeline-produced log, not required to keep the numbers).

---

## 35. Bug-fix history

| # | Bug | Original location / behaviour | Root cause | Forgeline fix | Test | Impact / metric dependency | Story value |
|---|---|---|---|---|---|---|---|
| B1 | Native DPO could not execute | the original native DPO helper `_sequence_log_prob` calls `_forward_blocks(model, input_ids)`, defined nowhere → `NameError` on first use | helper never written; script untested | `training/dpo.py::DPOAlgorithm._sequence_logprobs` uses `policy.logprobs` over response tokens with `reduce_logprobs(sum|mean)` | `test_dpo_increases_preference_margin`, `test_dpo_mean_reduction_reference` | P1 DPO never produced numbers → no metric depended on it | good: "the alignment path had never run" |
| B2 | PPO/GRPO probability ratio over the whole vocabulary | original native `ppo_step` and `grpo_step`: `log_probs_new = log_softmax(logits[...])`, `new_lp = log_probs_new.mean()` — mean over *all V logits and positions*, then `ratio = exp(new_lp − old_lp.mean())` | conflated the log-probability of the sampled tokens with the mean of the full log-distribution; the ratio was dominated by non-sampled tokens and clipping was meaningless | `models/policy.py::token_logprobs_from_logits` gathers sampled-token log-probs; `training/ppo.py` sequence-mean or per-token ratios; `training/grpo.py` same | `test_sequence_mean_ppo_ratio_reference` (independent formula), `test_ppo`, `test_grpo` | no native-RL numbers exist; the HuggingFace trainers used correct gathered log-probs, so recorded results are unaffected | strong: PPO ratio correctness |
| B3 | KL in the original DAPO step | the original `dapo_step` used token ratios correctly but its KL used full log-softmax means | as B2 | `training/dapo.py` token-level KL on sampled tokens | `test_dapo_dynamic_sampling_skips_constant_reward` | none | minor |
| B4 | GGUF exporter wrote invalid files | original exporter: every tensor info offset written as `struct.pack('<Q', 0)` ("will be relative"); Q4_0 packed `(q[i+1] << 4) | q[i]` interleaved without the `+8` bias | offsets never computed; ggml Q4_0 layout misunderstood (low nibbles = first 16 elements, values biased by 8) | `models/quantization/gguf.py` computes aligned offsets before writing (asserted at write time), correct Q4_0/Q8_0 layouts, header/tensor readers | `test_gguf_export_roundtrip`, `test_block_quantizers_roundtrip`, CLI export test | no exported-model metric existed | good: binary-format correctness |
| B5 | Multi-agent GRPO trainer crashed | original multi-agent trainer `train_step` does `lp, ref_lp = trajectory_logprobs(...)` but the imported `trajectory_logprobs` returns a single scalar tensor → unpack error on the first update; it also computed `ratio = exp(lp − lp.detach()) ≡ 1` | wrong return contract; ratio against itself | `rollouts/rewards/critic.py::CriticReward` composed with `CompositeReward` (`configs/post_training/grpo_agent_critic_tiny_cpu.yaml`), GRPO agent mode computes old/ref/new log-probs properly | `test_critic`, `test_agent_grpo_runs_tools`, manifest validated by `test_all_manifests_validate` | the run never completed; no metric | good |
| B6 | "Test evaluation" scored dataset records | original record-scoring evaluator (see §32.1) and a second evaluator that returned a fabricated KL | evaluator ignored the policy | `RecordRewardSuite` labelled policy-independent; `TrajectoryRewardSuite` for policy outputs; fabricated-KL evaluator removed | `test_record_and_preference_suites`, `test_heldout_record_reward_reproduces_recorded_statistics` | every recorded "held-out reward" (0.808, 0.812, 0.818, 0.580, LOO values) depended on it → reinterpreted, not deleted | strong: evaluation integrity |
| B7 | Synthesis reward ignores generated conditions | original `_rule_score` reads nested `outcomes`; generated JSON cannot affect it | reward/policy interface mismatch | documented; pinned by test; condition-aware reward listed as future work | `test_synthesis_rule_reward_ignores_generated_text_for_nested_records` | PPO training rewards (0.9007) reinterpreted | strong when told honestly |
| B8 | Preference pairs from the global RNG | original `build_preference_pairs` used `random.sample` unseeded | non-reproducible pairs | `domains/synthesis/preferences.py` seeded | `test_preferences_and_sft` | pair membership differs from the recorded run (count identical) | minor |
| B9 | `--deepspeed` accepted but never initialised | original distributed PPO script parsed the flag; no `deepspeed.initialize` | dead option | `DeepSpeedStrategy.initialize` | config tests; hardware test skipped | none | minor |
| B10 | Ray config without implementation | a Ray cluster config referencing a non-existent entry script | never built | `orchestration/ray_backend.py` | 11 Ray tests | none | good |
| B11 | Continuous batcher without cache reuse | original `ContinuousBatcher` allocated `PagedKVCache` blocks but ran `self.model(input_ids)` over the whole left-padded context every step (quadratic, no attention mask for padding) | cache tables never fed to the model | `InferenceEngine` uses `prefill`/`step` caches, batched same-length decode; block accounting gates admission | `test_engine_continuous_batching_and_greedy_equivalence` | no serving numbers existed | good: what "paged" really meant |
| B12 | `torch.compile`, `merge_on_save`, `pipeline_micro_batches` unwired in 0.1.0 | initial release | config fields without consumers | wired in this audit | `test_compile_model_matches_eager_and_checkpoints_load`, `test_merge_on_save_writes_plain_model_checkpoint`, `test_pipeline_strategy_uses_configured_micro_batches` | A40 throughput claim relied on compile | honest process story |
| B13 | Checkpoint corruption on network FS / disk exhaustion / OOM on resume (historical) | `torch.save` ZIP writes to a network mount truncated; 5.1 GB step checkpoints filled the container disk; `torch.load(map_location=device)` doubled GPU memory | environment | atomic tmp+rename, size validation, `keep_last` rotation, `map_location="cpu"` on load | checkpoint failure tests | the A40 run completed because of these fixes | strong operational story |
| B14 | Nested trajectory formats broke reward models | flat vs nested records | schema drift | `extract_outcomes` normalises both; flat labeled files shipped | `test_rule_score_flat_and_nested` | reproductions depend on it | minor |
| B15 | Initial-release fixes | lifecycle rollback from RETIRED, cancelled-status overwrite, composite double-weighting, YAML `TorchVersion`, block-size overflow in RLAIF DPO, FSDP guards, KL clamp placement, PPO monitor-KL sign, weight-decay defaults | — | fixed in 0.1.0 | covered by unit/integration tests | — | — |

---

## 36. Implementation ownership matrix

| Capability | Ownership | What Forgeline implements itself | What is delegated |
|---|---|---|---|
| Native transformer (RMSNorm, GQA, MLA, NSA, MoE, MTP, RoPE/YaRN, sliding window, soft-cap) | FORGELINE_IMPLEMENTATION_USING_PYTORCH_PRIMITIVES | every layer, cache, routing and loss in plain `torch.nn` | `F.scaled_dot_product_attention` for the dense fast path |
| Sampling, KV-cached generation, speculative/MTP decoding | FORGELINE_IMPLEMENTATION_USING_PYTORCH_PRIMITIVES | filters, caches, verification loop | — |
| LoRA / QLoRA / NF4 (native) | FORGELINE_IMPLEMENTATION_USING_PYTORCH_PRIMITIVES | injection, merge, save/load, NF4 table/pack/unpack, dequant-on-forward | — |
| PEFT LoRA on HF models | FORGELINE_INTEGRATION_OF_EXTERNAL_LIBRARY | policy surface, reference via `disable_adapter`, checkpoint of adapter tensors | transformers, peft, bitsandbytes (8-bit) |
| DPO, PPO, GRPO, DAPO, RLVR, process/critic rewards, RLAIF, STaR, hill climbing | NATIVE_FORGELINE | objectives, advantages, KL estimators, rollouts, verifiers, loops | torch autograd/optimizers |
| Trainer lifecycle (accumulation, clipping, precision, schedules, checkpoints, events, DDP sync) | NATIVE_FORGELINE | all | `torch.cuda.amp`, `torch.distributed` collectives |
| torch.compile | THIN_WRAPPER | in-place module compilation, prefix-stripping on load | torch.compile/inductor |
| DDP | FORGELINE_IMPLEMENTATION_USING_PYTORCH_PRIMITIVES | broadcast + all-reduce gradient averaging inside the trainer (not `DistributedDataParallel` wrapping in the step path), per-rank seeding | process groups |
| FSDP | FORGELINE_INTEGRATION_OF_EXTERNAL_LIBRARY | wrap policy, mixed-precision config, full-state gathering | `torch.distributed.fsdp` |
| DeepSpeed | THIN_WRAPPER / CONFIGURATION_ONLY | config validation, `initialize` call | deepspeed |
| Tensor parallel layers, 3-D mesh, pipeline stages + 1F1B | FORGELINE_IMPLEMENTATION_USING_PYTORCH_PRIMITIVES | column/row layers with custom autograd all-reduce, stage surgery, schedule, P2P | `dist.all_reduce/send/recv` |
| Ray orchestration | FORGELINE_INTEGRATION_OF_EXTERNAL_LIBRARY | actor classes, pools, weight-version sync, sharding, placement groups, replacement/retry logic, manifest integration | ray core (actors, object store, placement groups) |
| Checkpoint manager, validation, exact resume | NATIVE_FORGELINE | directory format, manifest, atomicity, rotation, validation | `torch.save/load` |
| Data pipeline (memmap, shards, streaming, schemas) | NATIVE_FORGELINE | all | numpy memmap; `datasets` for HF streaming (optional) |
| Continuous batching, paged block accounting, scheduler | NATIVE_FORGELINE | scheduler, allocator, engine loop, batching by cache length | — (no attention kernels) |
| GGUF export / quantizers | NATIVE_FORGELINE | file format writer/reader, Q8_0/Q4_0 packing | — |
| OpenAI-compatible serving | NATIVE_FORGELINE | router, schemas, SSE, stdlib server | — |
| Registry, gates, routing, flags, rollback | NATIVE_FORGELINE | all | — |
| Benchmarks (MMLU etc.) | NATIVE_FORGELINE (harness) | prompt formatting, scoring, sharding | `datasets` for downloads |
| Observability sinks | NATIVE_FORGELINE (JSONL/logging/tracing); THIN_WRAPPER (W&B, TensorBoard) | | wandb, tensorboard |
| Dashboard | NATIVE_FORGELINE | data layer, routes, page, attention recomputation | — |
| Encoder reward models | FORGELINE_INTEGRATION_OF_EXTERNAL_LIBRARY | heads, losses, blending | transformers encoders |
| Budgeted allocation environment, baselines, GRU/MLP policies, OPE, oracle, A/B, shadow | NATIVE_FORGELINE | simulator, primal-dual pacer, actor-critics, estimators, bootstrap/permutation tests, assignment | numpy, `torch.nn.GRU` |

Wording guide: "implemented/built" for NATIVE_FORGELINE and FORGELINE_IMPLEMENTATION_USING_PYTORCH_PRIMITIVES; "integrated" for FORGELINE_INTEGRATION_OF_EXTERNAL_LIBRARY; "configured/wired" for THIN_WRAPPER and CONFIGURATION_ONLY.

---

## 37. Technology evidence matrix

| Technology | Forgeline evidence | Exact file(s) | Implementation type | Test | Benchmark | Safe wording | Limitation |
|---|---|---|---|---|---|---|---|
| Python | 134 modules, typed contracts, packaging with extras | `pyproject.toml`, `core/protocols.py` | native | 229 tests | — | "Python ML systems framework with optional-extra dependency design" | — |
| PyTorch | every model/training/distributed component | `models/`, `training/`, `distributed/` | native on primitives | unit + integration | H02, H06 | "PyTorch from-scratch transformer training and RL post-training" | — |
| HF Transformers | `HFPolicy`, encoder reward models, HF streaming/tokenizers | `models/policy.py`, `models/reward.py`, `data/tokenizers.py` | integration | `test_huggingface_backend.py` (7, CPU tiny models) | H30–H47 (7B runs used HF+PEFT in the original code with equivalent semantics) | "integrated HuggingFace causal LMs behind a backend-agnostic policy interface" | 8-bit path untested here |
| PEFT / LoRA | HF LoRA policy; native LoRA | `models/policy.py`, `models/adapters/lora.py` | integration + native | adapters tests, HF tests | H30 (5.05M params) | "LoRA fine-tuning of a 7B model (0.066% trainable) and a native LoRA implementation" | — |
| QLoRA / NF4 | native NF4 base + LoRA | `models/quantization/nf4.py`, `models/adapters/qlora.py` | native | `test_nf4_roundtrip_and_qlora`, `test_sft[qlora]` | none measured | "implemented NF4 block quantization and QLoRA training path" | no memory/accuracy numbers on GPU |
| DPO | `training/dpo.py` | | native | 2 + HF | H35 | "implemented DPO (standard, length-normalised, reference-free); ran DPO on Qwen2.5-7B LoRA" | weak signal in the recorded run |
| PPO | `training/ppo.py` | | native | 3 + HF + reference test | H43–H47 | "implemented value-head PPO with sequence/token ratios and KL modes; executed 200 iterations on a 7B LoRA policy" | reward was policy-independent |
| GRPO | `training/grpo.py` | | native | 3 + HF + DDP + Ray | H53–H61 | "implemented GRPO (clipped and REINFORCE) incl. multi-turn agent mode; trained a tool-using 7B agent on GSM8K" | small batches, no accuracy gain |
| RLVR | `training/rlvr.py`, `rollouts/verifiers/` | | native | verifier tests, `test_rlvr_with_format_verifier` | H53–H57 | "RL with verifiable rewards: math/code/format verifiers, tagged-answer rewards" | — |
| RLAIF | `training/rlaif.py` | | native | CLI tests | H64–H65 | "built self-judge, pairwise-judge and constitutional AI-feedback pipelines" | recorded stats are rule-based demos |
| STaR | `training/star.py` | | native | CLI test | none (H68 unrecovered) | "implemented STaR rejection-sampling self-training" | no measured run |
| DAPO | `training/dapo.py` | | native | 1 + HF | none | "implemented DAPO (clip-higher, dynamic sampling, token-level loss, overlong shaping)" | no measured run |
| DDP | trainer sync | `training/common/trainer.py`, `distributed/strategies.py` | primitives | `test_ddp_two_process.py` | none | "data-parallel training with gradient averaging for every algorithm, validated with two processes" | CPU/gloo only |
| FSDP | strategy + full-state checkpoint gather | `distributed/strategies.py`, `trainer.py` | integration | config tests; GPU test skipped | none | "integrated FSDP sharding with mixed precision and full-state checkpointing" | not executed on GPU |
| DeepSpeed | strategy | `distributed/strategies.py`, `configs/distributed/deepspeed_zero2.json` | thin wrapper | config test | none | "DeepSpeed ZeRO-2 strategy and configuration" | not executed |
| Ray | worker pools | `orchestration/ray_backend.py` | integration | 11 CPU tests | none | "built Ray actor pools for rollouts, rewards, tools and sharded evaluation with weight sync, placement groups and fault recovery" | single machine only |
| Tensor parallelism | column/row layers, surgery | `distributed/tensor_parallel.py` | primitives | gloo degree-1 test; GPU test skipped | none | "implemented Megatron-style column/row-parallel layers" | multi-rank not executed |
| Pipeline parallelism | stages + 1F1B | `distributed/pipeline_parallel.py` | primitives | gloo tests, schedule unit tests | none | "implemented pipeline stages with a 1F1B schedule" | single stage executed |
| Distributed checkpointing | rank-0 writes, FSDP gather | `trainer.py`, `checkpoints/manager.py` | native | checkpoint tests | H18 (operational) | "atomic checkpoints with validation and exact resume; FSDP full-state gathering" | no sharded files |
| GGUF | writer/reader | `models/quantization/gguf.py` | native | round-trip test | none | "implemented a spec-compliant GGUF exporter (FP16/Q8_0/Q4_0) with aligned offsets" | own architecture tag |
| Quantization (Q8_0/Q4_0) | block quantizers | same | native | round-trip tests | none | "block-wise 8-/4-bit quantizers" | no accuracy numbers |
| Continuous batching | engine | `inference/engine.py`, `scheduler.py`, `paged_memory.py` | native | equivalence tests | none | "continuous-batching KV-cached inference engine with paged block accounting" | Python-level; no kernels; no throughput numbers |
| OpenAI-compatible serving | router/server | `serving/` | native | live HTTP test | none | "OpenAI-compatible HTTP serving with SSE streaming, routing and metrics" | no auth; no latency numbers |
| Model lifecycle | registry/gates/routing/rollback | `deployment/` | native | 7 + CLI | — | "champion/challenger lifecycle with fail-closed gates, deterministic routing and rollback" | single-writer JSON |
| torch.compile | trainer | `trainer.py` | thin wrapper | eager-backend equivalence test | H12 (historical ~39%) | "torch.compile integration (historical ~39% throughput gain on A40)" | not re-measured |
| Sequential decision-making under uncertainty | budgeted allocation POMDP | `domains/allocation/env.py` | native | 10 env tests | decisioning/budgeted_allocation | "designed a finite-horizon budget-constrained decision environment with stochastic, non-stationary opportunities and hidden policy-dependent dynamics" | simulator |
| Sequence modeling (GRU policy) | history-window GRU actor-critic | `domains/allocation/policies.py::GRUActorCritic` | native on `nn.GRU` | `test_gru_policy_consumes_history_window` | +0.19 over stateless PPO across 5 seeds | "sequence-conditioned RL policy that infers latent state from history" | did not beat the online pacer |
| Online optimization / constrained optimization | primal-dual pacing controller | `policies.py::DualPacingPolicy` | native | pacer tests | best policy (7.71) | "Lagrangian dual-descent budget pacing" | fixed step size |
| Offline policy evaluation / counterfactual evaluation | IPS, SNIPS, PDIS, DR, diagnostics, bootstrap CIs | `domains/allocation/ope.py` | native | 7 hand-value tests + log test | DR within 0.1–0.9 of truth | "implemented and verified IPS/SNIPS/DR estimators with support diagnostics" | trajectory IS degenerates at long horizons |
| A/B testing / experimentation | deterministic assignment, bootstrap CI, permutation test, guardrails → gates | `domains/allocation/experiment.py` | native | A/B + lifecycle tests | 400-episode simulated experiment | "simulated A/B harness with statistical tests wired into promotion gates" | simulated only |

---

## 38. Claim bank

Raw, evidence-backed candidate claims (not final bullets). Confidence reflects evidence strength; Resume-safe YES = citable as written, QUALIFIED = citable with the stated wording/limitation, NO = do not use.

| ID | Capability | What was built / how it works | Scale / workload | Measured result | Evidence files | Tests | Hardware | Personally implemented | Limitation | Conf. | Resume-safe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C01 | From-scratch LLM pretraining | GQA transformer, AdamW grouping, cosine LR, bf16, compile, atomic checkpoints | 421M params, ~491M FineWeb-Edu tokens, 30k steps | val loss 3.5834; 20.7k tok/s avg on 1×A40 | `benchmarks/training/fineweb_edu_large_a40`, H05–H12 | `test_pretrain_loss_decreases`, `test_pretraining_loss_is_token_cross_entropy` | A40 (historical) | model, trainer, data pipeline | throughput is of the original implementation; single GPU | HIGH | YES |
| C02 | CPU-scale pretraining | char-level `small` model | 10.6M params, 5k steps | val loss 1.479 | `benchmarks/training/char_small_cpu`, H01–H04 | same | laptop CPU | all | run report only | HIGH | YES |
| C03 | Constant-memory data pipeline | HF streaming → 10M-token shards → merge | 1B tokens prepared | pipeline survived OOM incidents (documented fixes) | H17; `data/streaming.py` | `test_streaming_pack_and_shards` | cloud pod | all | no timing | MEDIUM | QUALIFIED ("prepared a 1B-token corpus with a constant-memory sharded pipeline") |
| C04 | Fault-tolerant checkpointing | atomic tmp+rename, manifest sizes, rotation, CPU-mapped load, exact resume | 5.1 GB checkpoints in the A40 run | resume reproduces losses within 1e-5 | `checkpoints/manager.py`, H18, B13 | resume + failure tests | CPU (current), A40 (historical incidents) | all | FSDP optimizer state not saved | HIGH | YES |
| C05 | Unified post-training framework | one `Trainer` + `PostTrainingAlgorithm` contract over 15 algorithms; native + HF backends | tiny models on CPU; 7B LoRA historically | all algorithms run through one lifecycle | `training/`, §3 | `test_training_lifecycle.py` (18) | CPU | all | — | HIGH | YES |
| C06 | LoRA post-training of a 7B model | PEFT LoRA r=8 on q/k/v/o, adapter-disabled reference | Qwen2.5-7B, 5.05M trainable (0.066%) | SFT loss 1.663 → 1.008 (139 ex.); DPO and 200-it PPO executed | H30–H47; saved adapters | HF backend tests (CPU) | single GPU (historical) | policy surface, algorithms; PEFT integrated | reward for PPO was policy-independent | HIGH | QUALIFIED (describe execution, not improvement) |
| C07 | PPO with correct ratios | sampled-token log-probs, sequence/token ratios, KL modes, value head, entropy | tiny CPU + 7B historical | ratio test vs independent formula | `training/ppo.py`, B2 | `test_sequence_mean_ppo_ratio_reference`, `test_ppo` | CPU | all | — | HIGH | YES |
| C08 | GRPO family | clipped/REINFORCE, k3 KL, dynamic sampling, agent mode | — | — | `training/grpo.py` | `test_grpo`, DDP, Ray | CPU | all | — | HIGH | YES |
| C09 | Tool-using agent RL | multi-turn `<tool_call>` episodes, sandboxed Python, segment-conditioned log-probs, tagged verifiable rewards | Qwen2.5-7B on GSM8K, 200 it × 8 rollouts | best batch reward 0.5575; tool-use rate 0.5→0.9 on 20 held-out problems | H53–H57, `benchmarks/rlvr/agent_grpo_gsm8k_tools` | agent tests | single GPU (historical) | all | accuracy did not improve (3/20→1/20) | HIGH | QUALIFIED |
| C10 | Process reward model | step parsing, arithmetic/code verification, γ-discounted step credit | GSM8K, 30 it | best 1.0659 (iteration 1) | H59–H61 | `test_process_reward_reference` | single GPU (historical) | all | no improvement over run | MEDIUM | QUALIFIED ("implemented"; do not cite as gain) |
| C11 | DAPO | clip-higher, dynamic sampling, token-level normalisation, overlong shaping | tiny CPU | runs; zero-variance groups skipped | `training/dapo.py` | `test_dapo_dynamic_sampling_skips_constant_reward` | CPU | all | no measured run | HIGH | YES (implemented) |
| C12 | RLVR verifiers | math/exact/tagged/code/format, exception-safe | — | — | `rollouts/verifiers` | 9 rollout tests | CPU | all | — | HIGH | YES |
| C13 | AI-feedback data generation | pairwise judge, constitutional critique→revise, self-judge DPO rounds | 10 prompts demo | 60 pairs; 10 SFT + 10 DPO (exact) | H64–H65 | CLI tests | CPU | all | rule-based judges in the demo | HIGH | QUALIFIED |
| C14 | DDP for RL loops | broadcast + all-reduce inside the trainer, per-rank rollout seeds | 2 processes | replicas bit-identical after GRPO/pretrain updates | `test_ddp_two_process.py` | same | CPU gloo | all | no NCCL run | HIGH | QUALIFIED ("validated on two CPU processes") |
| C15 | FSDP / DeepSpeed strategies | wrap, mixed precision, full-state checkpoint; ZeRO-2 config | — | config-validated | `distributed/strategies.py` | config tests; GPU tests skipped | none | integration | not executed on GPU | MEDIUM | QUALIFIED ("integrated", not "ran") |
| C16 | Tensor & pipeline parallel components | column/row layers, 3-D mesh, 1F1B | degree 1 / single stage | logits equal to 1e-5; pipeline loss equals model loss | `distributed/` | gloo tests | CPU | all | multi-rank untested | MEDIUM | QUALIFIED ("implemented … validated at degree 1") |
| C17 | Ray orchestration | rollout/reward/tool/eval actor pools, version-based weight sync, placement groups, replacement + retries | local 4-CPU instance | worker rollouts equal learner rollouts; crashes recovered; sharded eval exact | `orchestration/ray_backend.py` | 11 tests | CPU | all (on Ray core) | no cluster/GPU run | HIGH | QUALIFIED ("validated locally") |
| C18 | Sharded distributed evaluation | benchmark tasks sharded across Ray workers, exact merge | — | identical to serial | same | `test_sharded_evaluation_equals_serial` | CPU | all | — | HIGH | QUALIFIED |
| C19 | Continuous-batching inference engine | FIFO admission bounded by paged blocks, per-sequence KV caches, batched same-length decode, cancellation, failure isolation | tiny models | batched = greedy token-for-token | `inference/` | 7 tests | CPU | all | Python-level; no throughput numbers | HIGH | YES ("built"; avoid "PagedAttention") |
| C20 | Paged KV block accounting | free-list allocator, block tables, admission gating | — | — | `inference/paged_memory.py` | `test_paged_allocator` | CPU | all | accounting only | HIGH | QUALIFIED (say "block accounting") |
| C21 | OpenAI-compatible serving | router, schemas, SSE, metrics, routing hook | — | live round trip | `serving/` | HTTP test | CPU | all | no auth, no latency numbers | HIGH | YES |
| C22 | GGUF export | spec-compliant writer/reader, FP16/Q8_0/Q4_0 | tiny models | read-back within tolerance; offsets aligned | `models/quantization/gguf.py`, B4 | round-trip test | CPU | all | own architecture tag | HIGH | YES |
| C23 | NF4 / QLoRA | block-64 NF4, dequant on forward, LoRA on top | tiny | round trip; frozen base | `nf4.py`, `qlora.py` | tests | CPU | all | no GPU memory numbers | HIGH | YES (implemented) |
| C24 | Model lifecycle | registry state machine, fail-closed gates, sha256 routing, shadow decisions, offline replay, flags, rollback, kill switch | — | routing splits within tolerance; rejections exit 2 | `deployment/` | 7 + CLI | CPU | all | single-writer JSON | HIGH | YES |
| C25 | Regression gates | absolute/relative rules, missing/NaN fail closed | — | — | `promotion.py`, `regression.py` | tests | CPU | all | — | HIGH | YES |
| C26 | Observability | structured logs, JSONL metrics/events, spans, sinks | — | — | `observability/` | 5 tests | CPU | all | W&B/TB thin | HIGH | YES |
| C27 | Dashboard | read-only UI over runs/checkpoints/evals/registry/benchmarks + exact attention maps | — | attention reconstruction to 1e-5 | `dashboard/` | 7 tests | CPU | all | no auth | HIGH | YES |
| C28 | Failure engineering | 27 typed errors with hints; 17 failure-mode tests | — | — | `core/errors.py`, `tests/failure` | 17 | CPU | all | — | HIGH | YES |
| C29 | Test suite | 229 tests, clean-install 204/9, hardware skips with reasons | — | 221/8 | `tests/` | — | CPU | all | GPU tests skipped | HIGH | YES |
| C30 | Optional-dependency design | extras; core never imports Ray/HF; actionable `OptionalDependencyError` | — | subprocess-verified | `pyproject.toml`, smoke tests | 4 | CPU | all | — | HIGH | YES |
| C31 | Evaluation integrity | identified that every synthesis "held-out reward" scored dataset records; relabelled and pinned by tests | — | 0.808 reproduced exactly as a dataset statistic | §32 | 14 semantics tests | CPU | all | — | HIGH | YES (as a story) |
| C32 | Bug fixes in earlier code | DPO could not run; PPO/GRPO ratio over vocabulary; GGUF offsets; multi-agent crash; batcher without cache reuse | — | — | §35 | listed | CPU | all | — | HIGH | YES |
| C33 | Speculative decoding | draft-model and MTP self-drafting with acceptance-rate reporting | tiny | — | `speculative.py` | `test_speculative_and_mtp_generators` | CPU | all | no speed numbers | MEDIUM | QUALIFIED ("implemented") |
| C34 | MoE / MLA / NSA architectures | aux-loss-free routing, shared experts, latent KV cache, sparse branches | tiny configs | forward/backward/generation tests; MLA cached = uncached | `models/` | `test_variants_forward_backward`, `test_mla_cached_matches_uncached` | CPU | all | no trained runs | MEDIUM | QUALIFIED ("implemented and unit-tested") |
| C35 | torch.compile | in-place compile with checkpoint compatibility | — | historical ~39% throughput on A40 | H12 | eager-equivalence test | A40 (historical) | wiring | not re-measured | MEDIUM | QUALIFIED |
| C36 | Knowledge distillation | logit KL + CE + feature matching | tiny | runs | `distill.py` | `test_distillation` | CPU | all | no measured run | HIGH | YES (implemented) |
| C37 | Synthesis-domain RL task family | rule reward, constraints, prompt codec, preference/SFT builders, tabular PPO, LOO evaluation | 500-record datasets | exact reproduction of all recorded statistics | `domains/synthesis`, H71–H81 | 6 + 14 | CPU | all | reward is policy-independent for nested records | HIGH | QUALIFIED |
| C38 | Multi-GPU scaling numbers | — | — | none exist | — | — | — | — | — | — | NO |
| C39 | Reward improvement from RL on the synthesis task | — | — | none (record statistics) | — | — | — | — | — | — | NO |
| C40 | GRPO 0.823 / RLAIF 0.814 / STaR 0.791 | — | — | unrecovered | §33 | — | — | — | — | — | NO |
| C41 | Sequential decisioning environment | finite-horizon budgeted allocation with hidden pressure that raises future costs and lowers responses; seeded, vectorised | 48 steps, budget 24; ~50k env steps/s | — | `domains/allocation/env.py`, §45 | 10 env tests | CPU | all | synthetic dynamics | HIGH | YES ("simulated") |
| C42 | Online primal-dual pacing | λ-priced allocation with subgradient budget-rate control | 300 held-out episodes | 7.71 value, 0.92 utilisation, pacing error 0.08 (best policy) | §45 | pacer tests | CPU | all | fixed α | HIGH | YES |
| C43 | Stateless vs sequence-conditioned PPO | MLP vs GRU history policy through the shared trainer | 5 seeds × 9,600 training episodes each | 7.40 ± 0.10 vs 7.60 ± 0.12 | §45 | PPO tests | CPU (8 s / 26 s per seed) | all | neither beat the pacer | HIGH | QUALIFIED ("history improved PPO; online optimizer still best") |
| C44 | OPE estimators verified and applied | IPS/SNIPS/PDIS/DR + ESS/support/overlap + bootstrap CIs | 300 logged episodes × 5 targets | DR abs. error 0.1–0.9; IPS collapses (ESS ≈ 0) | §45 | 7 hand-value tests + log test | CPU | all | long-horizon IS variance | HIGH | YES |
| C45 | Simulated A/B with guardrails feeding promotion gates | hash assignment, bootstrap CI, permutation test, guardrails, gate metrics | 400 episodes | Δ +0.57, CI [−0.04, +1.16], p 0.084 → rejected | §45 | A/B + lifecycle tests | CPU | all | simulated | HIGH | YES ("simulated A/B") |
| C46 | Shadow evaluation | candidate proposes in the incumbent's episodes; divergence and replay value | 100 episodes | divergence 0.40 | §45 | shadow test | CPU | all | — | HIGH | YES |
| C47 | Hindsight oracle / regret | fractional-knapsack upper bound ignoring pressure | 300 episodes | oracle 13.22; regrets 5.5–5.9 | §45 | oracle test | CPU | all | loose bound | HIGH | QUALIFIED (label as upper bound) |

---

## 39. Interview story bank

**S1 — Diagnosing the PPO/GRPO probability ratio.** *Situation:* the earlier native RL code reported ratios and clipped losses that never triggered clipping. *Why it mattered:* PPO's stability guarantee depends on the ratio of sampled-token probabilities. *Diagnosis:* reading `ppo_step` showed `log_softmax(logits).mean()` — a mean over the whole vocabulary and all positions — used as "log-prob"; the ratio compared two distribution averages, not the sampled tokens. *Alternatives:* per-token ratios (standard) vs sequence-mean ratios (the recorded 7B runs). *Decision:* implement both behind `ratio_level`, gather sampled-token log-probs via a cross-entropy trick that never materialises the full log-softmax. *Validation:* `test_sequence_mean_ppo_ratio_reference` recomputes the reference formula independently. *Result:* correct ratios; recorded HF runs (which gathered correctly) remain comparable. *Trade-offs:* sequence-mean ratios lose per-token credit assignment. *Could fail:* long responses under token-level clipping have many inactive tokens. *Follow-ups:* why clip at all; k3 vs log-diff KL; why detach the KL in monitor mode. *Files:* `training/ppo.py`, `models/policy.py::token_logprobs_from_logits`, `tests/unit/test_benchmark_semantics.py`.

**S2 — Repairing DPO that had never run.** The native DPO called an undefined helper; rewrote sequence scoring over response tokens with sum/mean reductions and adapter-disabled references; verified margins rise to 100% pairwise accuracy on a toy set and that mean-reduction matches an independent formula. Follow-ups: length bias of sum vs mean; reference-free DPO in RLAIF rounds; label smoothing. Files: `training/dpo.py`, `test_dpo_increases_preference_margin`.

**S3 — Fixing GGUF offsets and Q4_0 packing.** The exporter wrote offset 0 for every tensor and interleaved nibbles; wrote a reader to prove files were unreadable, then computed aligned offsets before the header is written (asserting `f.tell()` matches) and adopted ggml's low/high-nibble layout with the +8 bias. Follow-ups: alignment padding, why the architecture tag matters for llama.cpp, dequantisation error bounds. Files: `models/quantization/gguf.py`, `test_gguf_export_roundtrip`.

**S4 — Exact checkpoint resume.** Designed directory checkpoints with a manifest of file sizes, atomic rename, RNG/optimizer/scaler/adapter state; the test trains, saves at step 3, reloads into a new trainer and matches every subsequent loss within 1e-5. Historical context: the A40 run hit network-FS corruption, disk exhaustion and resume OOM which motivated size validation, rotation and CPU-mapped loading. Follow-ups: what breaks determinism (CUDA kernels, data-loader position under streaming), FSDP resume gaps. Files: `checkpoints/manager.py`, `test_checkpoint_resume.py`, `test_failure_modes.py`.

**S5 — Unifying 15 post-training algorithms.** One `PostTrainingAlgorithm` contract (`collect`/`loss`/`state`) and one `Trainer` owning accumulation, clipping, precision, schedules, checkpoints, events and DDP sync; algorithms became 150–250-line files. Trade-off: algorithms that need per-token rewards (GAE) fit less naturally. Follow-ups: how agent GRPO differs from single-turn; how the HF backend shares the loop. Files: `training/common/trainer.py`, `training/factory.py`.

**S6 — Ray rollout orchestration without touching gradients.** Decided Ray should own work around the learner (rollouts, rewards, tools, evaluation) and never gradient sync; workers hold policy copies, the learner pushes weights only when tensor version counters change (cheap change detection), old/reference log-probs are recomputed locally to avoid mixing numerics. Recovery: per-call worker generations, replacement once per generation, state restore, bounded retries, user errors not retried. Validation on a 4-CPU local instance incl. `os._exit` crashes and killed actors. Follow-ups: off-policy staleness if pushes were skipped; placement groups vs autoscaling; why not Ray Train. Files: `orchestration/ray_backend.py`, `test_ray_orchestration.py`.

**S7 — Champion/challenger lifecycle with fail-closed gates.** State machine with a single champion per name, gates that fail on missing/NaN metrics, first-promotion handling (`skip_relative`), rollback that restores the recorded predecessor and updates routing so no name is left unserved. Follow-ups: multi-writer registries, shadow dispatch, gate rule design for latency vs quality. Files: `deployment/`, `test_deployment.py`.

**S8 — Deterministic rollout routing.** `sha256(salt:request_id)` buckets mod 10,000 give sticky, reproducible splits; independent salt for shadow; pinned ids and explicit model requests bypass; disabled backends excluded; `simulate_traffic` checks split accuracy. Follow-ups: rebalancing when the challenger changes, salt rotation. Files: `deployment/rollout.py`, `feature_flags.py::stable_bucket`.

**S9 — Clean optional dependencies.** Core depends on torch/numpy/pyyaml only; HF, Ray, DeepSpeed, W&B are extras; modules that wrap extras raise `OptionalDependencyError` naming the extra; a subprocess test proves the core never imports Ray; the clean install passes 204 tests. Follow-ups: lazy imports vs import-time errors. Files: `pyproject.toml`, `tests/smoke/test_smoke.py`, `test_orchestration_config.py`.

**S10 — Multi-process DDP for RL.** Instead of wrapping models in `DistributedDataParallel` (awkward for rollouts that call `generate`), the trainer broadcasts parameters once and all-reduces gradients after accumulation; per-rank seeds make ranks sample different prompts/rollouts. Two real gloo processes prove replicas stay bit-identical after GRPO updates. Follow-ups: bucketed all-reduce, overlap with backward, NCCL specifics. Files: `training/common/trainer.py`, `test_ddp_two_process.py`.

**S11 — Inference scheduling.** FIFO admission bounded by both `max_batch` and free KV blocks; sequences grouped by cache length so a batched decode concatenates caches; finished sequences free blocks immediately (continuous batching); cancellation and failure isolation; batched output equals greedy single-sequence decoding token-for-token. Honest framing: Python-level block accounting, not PagedAttention kernels; the earlier batcher recomputed the whole padded context each step. Follow-ups: cross-length batching, prefix caching, what real paged attention needs. Files: `inference/`.

**S12 — Evaluation integrity: the 0.808 that every method "achieved".** Traced the evaluator to record scoring, reproduced the number exactly without a model, showed the rule reward could not see generated conditions, simulated the PPO training-reward statistics from record sampling, and relabelled every dependent number instead of deleting it. Follow-ups: what a policy-dependent evaluation needs (outcome model), why negative results were kept. Files: `evaluation/reward.py`, `test_benchmark_semantics.py`, §32.

**S14 — Making the environment react to the policy.** *Problem:* a benchmark where future opportunities are exogenous lets a myopic policy look sequential without being so. *Decision:* a hidden pressure state `p_{t+1} = δp_t + ηu_t` that multiplies future costs by `(1+κp)` and response rates by `(1−φp)`; pressure is not observed, so the problem is a POMDP and history carries information. *Validation:* same seed, different action prefixes → different effective costs/rates (`test_endogenous_feedback_changes_future_state_under_same_seed`); `pressure_gain: 0` restores exogeneity. *Result:* the GRU policy, which can infer pressure from realised costs, beat the stateless policy on every seed. *Follow-ups:* identifiability of pressure, why the oracle ignores it (upper bound), what a model-based baseline would need. Files: `domains/allocation/env.py`, `oracle.py`.

**S15 — Why IPS failed and DR worked.** With 48-step trajectories the product of per-step ratios has ESS ≈ 0 for any target far from the ε-mixed behaviour policy; SNIPS becomes undefined when all weights vanish. Implemented per-decision IS and doubly-robust estimation with a ridge Q̂ and reported support/ESS diagnostics instead of a number; DR landed within 0.1–0.9 of simulator truth for every target, IPS did not. Follow-ups: bias of clipped weights, choice of Q̂, when to trust DR CIs. Files: `domains/allocation/ope.py`, `tests/unit/test_allocation_ope.py`.

**S16 — Letting the A/B harness say no.** The GRU challenger was +0.57 on average but its bootstrap CI crossed zero (p = 0.084) and it clipped the budget more often; the guardrails rejected it and the same numbers, exported as `ab/*` gate metrics, fail the `PromotionGate`. Kept as a negative result rather than tuning until it passed. Follow-ups: sequential testing / early stopping, power analysis, guardrail selection. Files: `domains/allocation/experiment.py`, `configs/allocation/promotion_gate.yaml`.

**S13 — Failure recovery in tool-using agents.** Tool exceptions and timeouts are injected as `<tool_result>ERROR…` observations so episodes continue and the policy can learn from failures; verifier exceptions become zero rewards; non-finite rewards abort with the provider's name. Files: `rollouts/agent.py`, `rollouts/tools/python_executor.py`, `test_tool_execution_failure_is_injected`.

---

## 40. Truth / boundary ledger

| Claim | Implemented | Executed | Where | CPU/GPU | Historical or current | Measured | Artifact | Semantics preserved | Resume status |
|---|---|---|---|---|---|---|---|---|---|
| 421M pretraining to 3.58 val loss | yes | yes | cloud A40 pod | GPU | historical | yes | walkthrough + model card | yes | YES |
| 20.7k tok/s on A40 | n/a (original code) | yes | same | GPU | historical | yes (monitor) | same | architecture yes; code not re-timed | QUALIFIED |
| 7B LoRA SFT/DPO/PPO executed | yes | yes | single GPU | GPU | historical | yes (logs) | results JSON + saved adapters | yes (constant loss factor) | YES for execution; NO for "improvement" |
| Tool-agent GRPO on GSM8K | yes | yes | single GPU | GPU | historical | yes | results JSON | yes | QUALIFIED |
| PRM-GRPO | yes | yes | single GPU | GPU | historical | yes | results JSON | yes (factor B, tokenisation) | QUALIFIED |
| Synthesis GRPO 0.823, RLAIF 0.814, STaR 0.791 | yes (code) | unknown | — | — | — | unrecovered | none | — | NO |
| DDP | yes | yes | 2 CPU processes (tests) | CPU | current | correctness | test | — | QUALIFIED |
| FSDP / DeepSpeed | yes (integration) | no | — | — | — | no | — | — | "integrated" only |
| TP / PP | yes | degree-1 / single stage | CPU gloo | CPU | current | correctness | test | — | QUALIFIED |
| Ray pools | yes | yes | local Ray, 4 CPUs | CPU | current | correctness | tests | — | QUALIFIED |
| Exact resume | yes | yes | CPU | CPU | current | yes (1e-5) | test | — | YES |
| Continuous batching engine | yes | yes | CPU | CPU | current | equivalence only | test | — | YES (no throughput) |
| GGUF export | yes | yes | CPU | CPU | current | round trip | test | — | YES |
| NF4/QLoRA | yes | yes | CPU | CPU | current | round trip | test | — | YES (no GPU memory numbers) |
| Serving | yes | yes | CPU live HTTP | CPU | current | round trip | test | — | YES (no latency) |
| Lifecycle | yes | yes | CPU | CPU | current | split accuracy | tests | — | YES |
| Dashboard | yes | yes | CPU live HTTP | CPU | current | — | tests | — | YES |
| HF backend | yes | yes | CPU, tiny local Llama/BERT | CPU | current | — | tests | — | YES |
| 0.808 held-out | yes (evaluator) | yes | — | any | historical + reproduced | yes | results JSON, test | yes | dataset statistic only |
| Multi-GPU scaling | — | no | — | — | — | no | — | — | NO |
| Budgeted-allocation benchmark (pacer, PPO, GRU PPO) | yes | yes | laptop CPU | CPU | current | yes (300 episodes × 5 seeds) | `benchmarks/decisioning/budgeted_allocation` | — | YES (simulated) |
| OPE accuracy | yes | yes | CPU | CPU | current | yes | same | — | YES |
| Simulated A/B, shadow | yes | yes | CPU | CPU | current | yes | same | — | YES ("simulated") |
| Real workload / traffic / budgets | — | no | — | — | — | no | — | — | NO |

---

## 41. Hardware validation matrix

| Feature | CPU (current) | Single GPU | Multi-GPU | Historical validation | Current validation | Recommendation |
|---|---|---|---|---|---|---|
| Native model variants, generation | ✓ tests | — | — | `small`/`large` runs | unit tests | — |
| Pretraining | ✓ (tiny/small) | ✓ historical (A40) | — | H02, H06 | `test_pretrain_loss_decreases` | rerun `pretrain_large_gpu.yaml` briefly to time Forgeline's code |
| SFT / DPO / PPO / GRPO (native) | ✓ | — | — | — | integration tests | — |
| SFT / DPO / PPO / GRPO / DAPO / agent (HF) | ✓ tiny local model | ✓ historical (7B) | — | H31–H61 | `test_huggingface_backend.py` | rerun the `synthesis_*_qwen7b_lora` and `agent_grpo_gsm8k_qwen7b_lora` manifests to obtain Forgeline-produced logs |
| fp16 GradScaler / bf16 autocast | bf16 CPU only | needs CUDA | — | A40 bf16 (H06) | `test_cuda_*` skipped | `pytest tests/hardware` on one GPU |
| FSDP | config only | — | needs ≥2 | none | skipped | `torchrun --nproc_per_node=2 -m pytest tests/hardware -m multi_gpu` |
| DeepSpeed | config only | needs CUDA + extra | ✓ | none | skipped | install extra, run `test_deepspeed_initialize` |
| Tensor parallel | degree 1 | — | needs ≥2 | none | gloo test | multi-GPU hardware test |
| Pipeline parallel | 1 stage | — | needs ≥2 | none | gloo test | multi-stage run |
| DDP | ✓ 2 processes | — | NCCL untested | none | `test_ddp_two_process.py` | 2–8 GPU run of `ddp.yaml` |
| Ray | ✓ local | GPU worker test skipped | cluster untested | none | 11 tests | GPU worker test, then a small cluster run |
| Inference engine / serving | ✓ | `test_cuda_inference_engine` skipped | — | single-stream speeds (H04, H13) | tests | measure throughput/latency on GPU |
| NF4 / QLoRA / GGUF | ✓ | — | — | none | tests | GPU memory and accuracy measurements |
| 8-bit HF loading | — | needs CUDA + bitsandbytes | — | none | none | one load on GPU |
| Lifecycle / dashboard / observability | ✓ | — | — | — | tests | — |
| Sequential decisioning (env, PPO, OPE, A/B) | ✓ full benchmark | not needed | not needed | — | 30 tests + benchmark | none; CPU is the target |

---

## 42. Future optional experiments

Only experiments that would add evidence (not reruns for repository reasons):

1. **Policy-dependent synthesis reward** — implement a condition→outcome model (or use the yield-series simulator) so PPO/GRPO/DPO on the synthesis task can be evaluated on generated conditions; rerun `synthesis_ppo_qwen7b_lora.yaml` and `synthesis_grpo_qwen7b_lora.yaml`. This is the only way to obtain a real reward-improvement number and to replace H66.
2. **Forgeline-produced logs for the recorded 7B runs** — rerun SFT/DPO/PPO/agent GRPO/PRM-GRPO manifests on one GPU to retire the `revalidation_recommended` flags and give the throughput of Forgeline's own code.
3. **RLAIF and STaR on GSM8K with retained results** — replace H67/H68 with measured loss/accuracy.
4. **Multi-GPU** — DDP scaling 1/2/4/8 on `large` pretraining; FSDP and TP hardware tests; record under `benchmarks/distributed`.
5. **Ray** — GPU rollout workers and a 2-node cluster; measure rollout throughput with 1/2/4 workers vs in-process.
6. **Inference** — engine throughput/latency vs concurrency on GPU; Q8_0/Q4_0 accuracy vs FP16; NF4 memory.
7. **Full benchmark suites** (non-offline) on a trained checkpoint with promotion gated on them.
8. **Agent accuracy** — agent GRPO with a curriculum and lower hill-climb threshold to test whether tool use converts to accuracy (the recorded runs show it did not).

---

## 43. GenAI infrastructure concept mapping

| Concept | Forgeline implementation | Evidence | Metric / test | Safe wording | Gap |
|---|---|---|---|---|---|
| Distributed training | strategies single/ddp/fsdp/deepspeed/tp/pp; trainer sync | `distributed/`, `trainer.py` | `test_ddp_two_process.py`, gloo tests | "designed and validated data-parallel training for supervised and RL stages; integrated FSDP/DeepSpeed; implemented TP/PP components" | no multi-GPU execution |
| Data / model parallelism | DDP gradient averaging; column/row-parallel layers; pipeline stages; 3-D mesh | same | degree-1 equivalence | as above | — |
| Checkpoint management | atomic manifest checkpoints, validation, rotation, exact resume, FSDP gather | `checkpoints/` | 1e-5 resume | "fault-tolerant checkpointing with exact resume" | sharded files |
| Gradient communication | all-reduce averaging, custom autograd all-reduce in TP layers | `trainer.py`, `tensor_parallel.py` | tests | "implemented gradient all-reduce and TP collectives" | bucketing |
| Training orchestration | manifests, CLI, Ray worker pools, torchrun | `cli/`, `orchestration/` | Ray tests | "Ray-based rollout/reward orchestration around a torch.distributed learner" | cluster run |
| Post-training / RLHF | SFT, RM, DPO, PPO, GRPO, DAPO, RLAIF | `training/` | H30–H47 | "built and executed RLHF pipelines on a 7B model" | improvement evidence |
| RLVR | verifiers, tool agents, process rewards | `rollouts/`, `training/rlvr.py` | H53–H61 | "RL with verifiable rewards and tool use" | accuracy gain |
| Profiling / performance | tok/s, latency, memory metrics; spans | `evaluation/performance.py`, `observability/tracing.py` | H10 (historical) | "throughput/latency instrumentation" | GPU measurements |
| Mixed precision | bf16/fp16 autocast + scaler | `precision.py` | H06 (bf16 run) | "bf16 training on A40" | — |
| Quantization | NF4, Q8_0/Q4_0, GGUF, 8-bit loading | `models/quantization`, `adapters/qlora.py` | tests | "implemented NF4/QLoRA and GGUF quantized export" | accuracy numbers |
| Inference | continuous batching, KV caches, block accounting, speculative decoding | `inference/`, `speculative.py` | equivalence tests | "built a continuous-batching inference engine" | kernels, throughput |
| Model serving | OpenAI-compatible HTTP + SSE, routing, metrics | `serving/` | live test | "OpenAI-compatible serving with champion/challenger routing" | auth, latency numbers |
| Latency / throughput | `/metrics` p50/p95; `measure_generation` | same | — | instrumentation only | measurements |
| Failure handling | typed errors, isolation, retries, fail-closed gates | §28 | 17 tests | "failure-mode test suite and recovery paths" | — |
| Monitoring | JSONL metrics/events, dashboard, W&B/TB | `observability/`, `dashboard/` | tests | "structured observability and a run/registry dashboard" | — |
| Staged rollouts / A-B / champion-challenger | registry, gates, deterministic routing, shadow, replay, rollback | `deployment/` | split tests | "champion/challenger lifecycle with deterministic traffic splits and one-command rollback" | shadow dispatch |
| Regression detection | gate rules, `evaluate_regression` | `promotion.py`, `regression.py` | tests | "fail-closed regression gates" | — |
| AI-assisted workflow | RLAIF pairwise/constitutional data generation, self-judge rounds | `rlaif.py` | H64–H65 | "AI-feedback data pipelines" | LLM judges not measured |
| Agents / tool execution | multi-turn tool episodes, sandboxed executor, tagged rewards | `rollouts/agent.py`, `tools/` | H53–H57 | "tool-using agent RL with sandboxed Python execution" | security sandboxing |
| Sequential decision-making under uncertainty | budgeted allocation POMDP: act before future opportunities are revealed | `domains/allocation/env.py` | env tests, benchmark | "finite-horizon decisions under uncertainty with a hard budget" | simulator |
| Policy-dependent environments | hidden pressure driven by past intensities | `env.py` | endogeneity test | "policy-dependent (endogenous) dynamics" | stylised |
| Reinforcement learning (control) | PPO on the environment through the shared trainer | `domains/allocation/ppo.py` | PPO tests, 5-seed benchmark | "trained RL allocation policies with PPO" | did not beat the pacer |
| Sequence modeling | GRU over history windows | `policies.py::GRUActorCritic` | history test, +0.19 | "sequence-conditioned policy" | — |
| Online optimization / constrained optimization | dual-price pacing with subgradient updates | `policies.py::DualPacingPolicy` | best policy | "Lagrangian budget pacing" | — |
| Offline / counterfactual policy evaluation | IPS, SNIPS, PDIS, DR, diagnostics, bootstrap | `ope.py` | hand-value tests, accuracy vs truth | "verified OPE estimators with support diagnostics" | long-horizon variance |
| A/B testing / experimentation | deterministic assignment, bootstrap CI, permutation test, guardrails | `experiment.py` | A/B tests, benchmark | "simulated A/B experiments with statistical guardrails" | simulated |
| Champion/challenger deployment | A/B gate metrics → `PromotionGate` → registry → rollback | `experiment.py`, `deployment/` | lifecycle test | "experiment-gated promotion with rollback" | — |

---

## 44. Resume-safe summary

Facts that can be used as written (see §38 for qualifications):

* Forgeline: a PyTorch framework (≈14.4k lines, 229 tests, 221 pass on CPU; 204 on a minimal install) covering pretraining, SFT, distillation, reward modeling, DPO, PPO, GRPO, agent GRPO with tools, process-reward GRPO, DAPO, RLVR, RLAIF (self-judge, pairwise, constitutional), STaR and hill climbing behind one trainer lifecycle and one manifest, with native and HuggingFace+PEFT policy backends.
* Pretrained a 421M-parameter GQA transformer on ~491M FineWeb-Edu tokens on a single A40 (bf16, torch.compile) to 3.58 validation loss at ~20.7k tokens/s; a 10.6M char-level model on CPU to 1.479.
* Post-trained Qwen2.5-7B-Instruct with LoRA (5.05M trainable parameters, 0.066%): SFT loss 1.66 → 1.01 on 139 examples; DPO on 80 preference pairs; 200 iterations of value-head PPO — with the caveat that the synthesis reward was policy-independent, so these runs demonstrate pipeline execution, not reward improvement.
* Trained a tool-using 7B agent with GRPO and verifiable rewards on GSM8K (best batch reward 0.56; held-out tool-use rate 0.5 → 0.9; accuracy unchanged); implemented step-level process rewards.
* Data-parallel training for every algorithm (validated with two processes), FSDP/DeepSpeed strategies, Megatron-style tensor-parallel layers and a 1F1B pipeline scheduler, and optional Ray actor pools for rollouts, rewards, tools and sharded evaluation with on-policy weight sync, placement groups and worker recovery (validated on a local Ray instance).
* Fault-tolerant directory checkpoints with size validation, rotation and exact resume (losses match within 1e-5); NF4/QLoRA; spec-compliant GGUF export (FP16/Q8_0/Q4_0); continuous-batching KV-cached inference engine with paged block accounting (batched output identical to greedy); OpenAI-compatible SSE serving; candidate registry with fail-closed promotion gates, sha256 champion/challenger routing, shadow decisions, feature flags and rollback; structured metrics/events and a read-only dashboard with exact attention maps.
* Sequential decisioning: built a finite-horizon budgeted allocation environment with hidden policy-dependent dynamics, a primal-dual pacing controller, stateless and GRU sequence-conditioned PPO policies through the shared trainer, verified IPS/SNIPS/PDIS/DR offline evaluation with support diagnostics and bootstrap CIs, a hindsight oracle, simulated A/B experiments with guardrails feeding the promotion gates, and shadow evaluation; measured on CPU over 5 seeds × 300 episodes (pacer 7.71 > GRU PPO 7.60 ± 0.12 > stateless PPO 7.40 ± 0.10 > heuristic 7.29; oracle 13.22; A/B rejected the challenger).
* Fixed defects in earlier versions: DPO that could not execute, PPO/GRPO ratios computed over the whole vocabulary, GGUF offsets/Q4_0 packing, a multi-agent trainer that crashed, a batcher that never reused KV caches; identified that all "held-out reward" numbers were dataset statistics and relabelled them.

Not to be claimed: real workloads, advertiser/traffic/budget data, live delivery or production A/B tests, SOTA bidding, multi-GPU scaling numbers, reward improvement on the synthesis task, GRPO 0.823 / RLAIF 0.814 / STaR 0.791, "PagedAttention", GPU throughput of Forgeline's own code, Ray cluster execution.

Evidence locations: `benchmarks/**/results.json` (+ `ledger_status`), `benchmarks/historical_ledger.json`, `tests/unit/test_benchmark_semantics.py`, and this dossier.

---

## 45. Sequential decisioning under uncertainty: budgeted allocation

**Status:** implemented, tested and measured on CPU in this extension (package `forgeline.domains.allocation`, 8 modules; manifest algorithm `allocation_ppo`; CLI group `forgeline allocation`). **Framing that is true:** a simulated, general finite-horizon resource-allocation benchmark with policy-dependent dynamics, real RL / sequence-model / online-optimisation / offline-evaluation / experimentation mechanisms. **Framing that is false:** any real workload, advertiser or traffic data, live delivery, production experiments, or state-of-the-art bidding claims.

### 45.1 Architecture after the extension

```
domains/allocation/
  env.py         AllocationEnvConfig, BudgetedAllocationEnv (reset/transition/step/observation), VectorEnv, episode_seed
  policies.py    Policy interface; ThresholdPacingPolicy; DualPacingPolicy; MLPActorCritic; GRUActorCritic; NeuralPolicy; HistoryBuffer; EpsilonMixPolicy
  ppo.py         AllocationPPOAlgorithm (PostTrainingAlgorithm) — collect/loss on top of the shared Trainer; build_allocation_ppo; load_allocation_policy
  rollout.py     LoggedStep / Episode schema v1, write/read_episodes (validated), run_episodes (lockstep batches), episode_metrics, aggregate_metrics
  ope.py         cumulative_weights, ips/snips/pdis/dr estimators, effective_sample_size, bootstrap_ci, fit_q_ridge, evaluate_policy, evaluate_target_policy
  oracle.py      hindsight_upper_bound (fractional knapsack, KKT + bisection), oracle_for_seed, oracle_values
  experiment.py  ABConfig, assign_arms (stable_bucket), run_ab_experiment (bootstrap CI, permutation test, guardrails, gate_metrics), shadow_evaluate
  benchmark.py   BenchmarkConfig, run_benchmark (train → evaluate → OPE → A/B → shadow → results.json + summary.md)
```
Integration points (no duplicates created): `training/factory.py::build_algorithm` dispatches `allocation_ppo` → `build_allocation_ppo`; `core/config.py::KNOWN_ALGORITHMS` lists it; the shared `Trainer`, `CheckpointManager`, metrics sinks and DDP path are reused unchanged; `deployment/feature_flags.py::stable_bucket` provides A/B assignment; `PromotionGate`, `CandidateRegistry`, `rollback` consume `ABResult.gate_metrics()`; new events `ope.evaluated`, `experiment.ab`, `experiment.shadow` in `observability/events.py`; the dashboard shows the `benchmarks/decisioning` record and any `runs/` produced by training.

### 45.2 Exact MDP

| Element | Definition |
|---|---|
| Horizon `T` | 48 (benchmark); fixed length; no early termination |
| Budget `B` | 24.0; hard constraint `x_t = min(m_t ĉ_t, B_t)`; a clipped request is a recorded *violation* |
| Exogenous opportunity `o_t = (v_t, c_t, q_t, s_t)` | `s_t = sin(2π(t+phase)/T)`, `phase ~ U[0,T)`; `v_t = v̄ e^{σ_v ε − σ_v²/2} (1 + A_v s_t)`; `c_t = c̄ e^{σ_c ε' − σ_c²/2} (1 + A_c s_t)`; `q_t ~ Beta(α,β)`; `v̄ = c̄ = 1`, `σ_v = 0.5`, `σ_c = 0.3`, `A_v = 0.4`, `A_c = −0.2`, `α = β = 2` |
| Action `a_t` | `{abstain, low, medium, high}` → multiplier `m_t ∈ {0, 0.5, 1.0, 1.5}`; chosen before `o_{t+1}` is drawn |
| Latent `p_t` | pressure, `p_0 = 0`, **not observed** |
| Effective cost | `ĉ_t = c_t (1 + κ p_t)`, `κ = 0.6` |
| Intensity | `u_t = x_t / ĉ_t` |
| Response | `ρ_t = q_t (1 − e^{−u_t}) max(0, 1 − φ p_t)`, `φ = 0.35`; `y_t ~ Bernoulli(ρ_t)` |
| Reward / cost | `r_t = y_t v_t`; cost `x_t` |
| Transitions | `B_{t+1} = B_t − x_t`; `p_{t+1} = δ p_t + η u_t`, `δ = 0.85`, `η = 0.25` |
| Observation (14-d) | remaining/B, t/T, (T−t)/T, v_t/v̄−1, c_t/c̄−1, q_t, s_t, cum_spend/B − t/T, cum_value/(T v̄), last multiplier/1.5, last reward/v̄, last spend/c̄, mean relative value and cost of the last 4 opportunities |
| Objective | maximise `Σ_t r_t` s.t. `Σ_t x_t ≤ B`; undiscounted (γ = 1 in PPO) |
| Stochasticity | opportunity draws and Bernoulli outcomes, both fixed by the episode seed (`episode_seed(base, i)`), so every policy faces identical exogenous streams per seed |
| Partial observability | pressure is hidden; it must be inferred from realised effective costs/outcomes → POMDP |

### 45.3 Policy-dependent dynamics (endogenous feedback)

`p_{t+1} = δ p_t + η u_t` accumulates the intensities the policy actually affords; `p_t` multiplies later effective costs by `(1 + κ p_t)` and later response rates by `(1 − φ p_t)`. Aggressive allocation therefore both consumes budget and degrades later opportunities. Proof of endogeneity in tests: with identical seed and identical actions from step 6, an aggressive prefix (5 × high) yields strictly higher effective cost and lower response rate than a cautious prefix (5 × abstain) on the identical opportunity, across 30 seeds; `pressure_gain: 0` makes the two prefixes indistinguishable (`tests/unit/test_allocation_env.py`). Realised effect in the benchmark: final pressure 0.25 (heuristic) vs 0.68 (pacer) vs ~0.5 (PPO) shows the policies induce different latent-state distributions.

### 45.4 Uncertainty model

Value, cost and quality are drawn per step from the log-normal / Beta processes above with a seasonal modulation whose phase is random per episode (non-stationarity the policy cannot know in advance); outcomes are Bernoulli. The policy sees only the current opportunity and observable history.

### 45.5 Baselines

* **Static heuristic** — `ThresholdPacingPolicy(threshold 0.6, level medium, slack 0.05)`: allocate when `v_t q_t / c_t ≥ 0.6` and cumulative spend ≤ `B (t/T + 0.05)`.
* **Online optimizer** — `DualPacingPolicy(α = 0.2, λ₀ = 0.8)`: objective `max Σ_t E[r_t]` s.t. `Σ x_t ≤ B`; per-step Lagrangian `L(m) = v_t q_t (1 − e^{−m}) − λ m c_t`, action `argmax_m L`; dual update `λ ← max(0, λ + α (x_t − B/T))`. Budget pressure enters through λ; adaptation is to observed spend (not to the latent pressure).

### 45.6 RL policies and sequence model

* **Stateless PPO** — `MLPActorCritic`: `14 → 64 → tanh → 64 → tanh → {4 logits, value}`.
* **Sequence-conditioned PPO** — `GRUActorCritic`: history features `h_j = [obs_j (14), onehot(a_j) (4), r_j/v̄, x_j/c̄, B_j/B]` (21-d) over a window `W = 8`; `HistoryBuffer` keeps a left-padded rolling window with lengths; the forward pass right-aligns and packs valid steps, runs a 1-layer GRU (hidden 64), masks empty histories, concatenates the final GRU state with `tanh(Linear(obs))`, then `Linear(128 → 64) → tanh → {logits, value}`. Test: identical current observation with different histories yields different action distributions; the window truncates to the most recent 8 steps.
* **PPO integration** — `AllocationPPOAlgorithm.collect` runs `episodes_per_rollout = 32` lockstep episodes (`VectorEnv`), storing obs, history tensors, actions, old log-probs, rewards and values; GAE (`compute_gae`, γ = 1, λ = 0.95) per episode; advantages normalised; the same rollout is returned for `ppo_epochs = 4` consecutive trainer steps with a fresh minibatch permutation (`gradient_accumulation_steps = 4` minibatches); `loss` = `ppo_clipped_objective` (ε 0.2) + 0.5 value MSE − 0.01 entropy; metrics `reward_mean/std`, `clipped_per_episode`, `utilization`, `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_fraction`. Optimizer AdamW lr 3e-3 → 3e-4 linear decay, grad clip 0.5; 1,200 trainer steps = 300 rollouts = 9,600 episodes per seed. Checkpoints via `Trainer` (`model_spec.family = allocation_policy`); `load_allocation_policy` rebuilds the policy from a checkpoint.

### 45.7 Logged trajectories

`LoggedStep` fields: `episode_id, seed, t, obs[14], history_summary{cum_spend_frac, elapsed_frac}, action, propensity, action_probs[4], reward, cost, next_obs[14], cum_spend, cum_reward, remaining_budget, terminal, clipped, pressure`. Validation on read: `0 < propensity ≤ 1`, `propensity == action_probs[action]`, `action_probs` a finite distribution, finite reward/non-negative cost, contiguous timesteps, terminated episodes, schema version. Behaviour policy for the benchmark log: `EpsilonMixPolicy(best stateless PPO, ε = 0.2)` → every action has propensity ≥ 0.05.

### 45.8 Offline policy evaluation

Estimators as in `ope.py` docstring (IPS, SNIPS, PDIS, per-decision DR with ridge Q̂; clipped variants at `max_weight = 20`); diagnostics ESS, ESS fraction, max/mean weight, unsupported fraction (target mass on behaviour-impossible actions → `OPEError`), zero-target fraction, clipped fraction, overlap mean/min; warnings for ESS < 10, ESS/n < 0.05, max weight > 100, zero-target > 50 %; refusals for zero/invalid propensities, invalid target distributions, non-finite weights. Bootstrap: 500 percentile resamples over episodes (SNIPS bootstraps the ratio). Hand-computed verification: two 2-step episodes with uniform behaviour and a 0.8/0.2 target give IPS 6.08, SNIPS 3.8, PDIS 5.96, DR (Q̂ ≡ 1) 5.36, ESS 1.47 (`test_core_formulas_against_hand_values`), and `evaluate_policy` reproduces them end to end.

**Measured OPE accuracy** (300 logged episodes, behaviour `ppo_mlp_eps0.2`, truth = simulator on the same seeds):

| Target | Truth | IPS | SNIPS | PDIS | DR | DR clipped | ESS | warnings |
|---|---|---|---|---|---|---|---|---|
| ppo_mlp_best | 7.566 | 5.235 (err 2.33) | 8.536 (err 0.97) | 5.958 (err 1.61) | 7.671 (err 0.11) | 7.474 (err 0.09) | 4.4 | 2 |
| ppo_gru_best | 7.742 | 0.257 (err 7.48) | 5.535 (err 2.21) | 2.374 (err 5.37) | 6.817 (err 0.92) | 6.928 (err 0.81) | 5.2 | 2 |
| dual_pacing | 7.697 | 0.000 (err 7.70) | — | 1.784 (err 5.91) | 6.917 (err 0.78) | 6.742 (err 0.96) | 0.0 | 4 |
| threshold_pacing | 7.138 | 0.000 (err 7.14) | — | 0.603 (err 6.53) | 6.677 (err 0.46) | 6.747 (err 0.39) | 0.0 | 3 |
| behaviour_itself | 6.927 | 6.802 (err 0.12) | 6.802 (err 0.12) | 6.802 (err 0.12) | 6.828 (err 0.10) | 6.828 (err 0.10) | 300.0 | 0 |

DR is the only estimator that stays close to truth for every target; trajectory IPS collapses (ESS ≈ 0–5 of 300) because 48-step ratio products vanish or explode; PDIS is biased low for targets that abstain where the behaviour spends. The self-evaluation row (`behaviour_itself`) recovers the logged value with ESS = 300.

### 45.9 Oracle and regret

`hindsight_upper_bound` solves `max Σ v_t q_t (1 − e^{−u_t})` s.t. `Σ u_t c_t ≤ B`, `0 ≤ u_t ≤ 1.5` with the KKT solution `u_t = clip(ln(v_t q_t/(λ c_t)), 0, 1.5)` and λ by bisection; it knows the future, ignores pressure (which only hurts) and allocates fractionally, so it upper-bounds every realisable policy's expected return (tested: the expected value of every baseline plan ≤ oracle). Benchmark oracle mean 13.219; regrets in the table below. It is labelled HINDSIGHT ORACLE / UPPER BOUND and is not a deployable policy.

### 45.10 Measured benchmark (`benchmarks/decisioning/budgeted_allocation`, `forgeline allocation benchmark --config configs/allocation/benchmark.yaml`)

Environment: horizon 48, budget 24; 300 held-out episodes (seed family 20000) shared by every policy; PPO trained on 5 seeds (0–4), evaluated with its stochastic action distribution.

| Policy | Value (mean ± seed std) | Utilisation | Value/budget | Pacing error | Early exhaustion | Unused budget | Violations | Regret vs oracle | Δ vs dual pacer (dual − policy) |
|---|---|---|---|---|---|---|---|---|---|
| threshold_pacing | 7.292 ± 0.000 | 0.700 | 0.304 | 0.178 | 0.000 | 0.300 | 0.02 | 5.927 | +0.415 |
| dual_pacing | 7.707 ± 0.000 | 0.920 | 0.321 | 0.082 | 0.000 | 0.080 | 0.03 | 5.512 | +0.000 |
| ppo_mlp | 7.402 ± 0.098 | 0.873 | 0.308 | 0.153 | 0.045 | 0.127 | 0.60 | 5.818 | +0.306 |
| ppo_gru | 7.595 ± 0.115 | 0.948 | 0.316 | 0.091 | 0.198 | 0.052 | 1.82 | 5.624 | +0.112 |

Per-seed PPO values (same held-out episodes):

| Seed | stateless PPO | GRU PPO | GRU − stateless |
|---|---|---|---|
| 0 | 7.441 | 7.662 | +0.221 |
| 1 | 7.542 | 7.718 | +0.177 |
| 2 | 7.409 | 7.564 | +0.155 |
| 3 | 7.309 | 7.616 | +0.307 |
| 4 | 7.308 | 7.416 | +0.108 |

Interpretation: history helps — the GRU policy beats the stateless policy on every seed (+0.12 to +0.26, mean +0.19, ≈ 1.7 seed-std) and paces better (pacing error 0.09 vs 0.15), but it exhausts the budget early more often (0.20 vs 0.05) and clips more (1.8 vs 0.6 attempts/episode). The online dual pacer remains the best policy at this training budget (7.71), the static heuristic the worst (7.29, 70 % utilisation). All four sit far below the hindsight bound (13.2), which is loose because it also ignores pressure and discreteness. **Negative result preserved:** RL did not beat the online optimizer; no tuning was performed to change that.

**Simulated A/B** (`dual_pacing` incumbent vs best GRU PPO challenger, 400 episodes hashed into arms 197/203): Δvalue +0.567, 95 % bootstrap CI [-0.041, +1.157], permutation p = 0.084; guardrail failures: ['primary metric CI lower bound -0.0405 < -0.0', 'violations increased by 1.029 > 0.5']; decision **reject**. The same metrics fail `configs/allocation/promotion_gate.yaml` (`ab/value_delta_ci_low ≥ 0`, `ab/violations_delta ≤ 0.5`).

**Shadow evaluation** (100 episodes): divergence rate 0.398 (by horizon third [0.381, 0.378, 0.435]); incumbent value 7.656, candidate replay value 8.066.

**Throughput (laptop CPU):** 50886 env steps/s in evaluation; 297 OPE trajectory-evaluations/s; 518 A/B episodes/s; PPO training 8 s (MLP) / 26 s (GRU) per seed for 9,600 episodes; whole benchmark 179 s; 96,000 training episodes simulated in total.

### 45.11 Pacing / budget metrics

Per episode (`episode_metrics`): value, spend, utilisation, value per budget, pacing error `mean_t |cum_spend_t/B − (t+1)/T|`, early exhaustion (budget gone before 90 % of the horizon), unused-budget fraction, violations (clipped requests), reward variance, value and spend by horizon third, action distribution, final pressure; `aggregate_metrics` adds means, standard deviations and a 95 % half-width for value.

### 45.12 Lifecycle integration

`train candidate → held-out simulator evaluation (Trainer.evaluate on a fixed seed family) → OPE from a logged behaviour set → simulated A/B → ABResult.gate_metrics() → CandidateRegistry.register(metrics) → PromotionGate (configs/allocation/promotion_gate.yaml) → promote or reject → rollback`. Demonstrated by `examples/06_allocation_lifecycle.py` and `tests/integration/test_allocation_pipeline.py::test_lifecycle_with_ab_gate` (a regressing challenger is rejected, a better one is promoted and rolled back). Shadow decisions reuse the same policy interface without any second lifecycle.

### 45.13 Tests (30 new; all CPU)

`tests/unit/test_allocation_env.py` (10): transition equations, budget conservation/clipping, termination, invalid action/config, seeded reproducibility, endogenous feedback (and its absence with `pressure_gain: 0`), vector env equivalence, oracle upper bound/budget, pacing metrics. `tests/unit/test_allocation_ope.py` (7): hand-computed IPS/SNIPS/PDIS/DR/ESS/clipping/discount, end-to-end match, bootstrap behaviour, support and validity errors, weak-support warnings, weight explosion refusal, logged schema round trip and validation. `tests/integration/test_allocation_pipeline.py` (13): heuristic pacing and dual-price adaptation, pacer budget adaptation, GRU history consumption and window truncation, ε-mix support, PPO training + checkpoint reload for MLP and GRU, PPO improvement over initialisation, OPE recovering simulator value for a supported target and GRU replay, deterministic/balanced assignment, permutation test and A/B decisions (rejection, promotion, small-sample warning, invalid guardrail), shadow divergence, lifecycle with gate and rollback, CLI end to end (benchmark, ope, ab with gate metrics and events, shadow, train).

### 45.14 Bugs found while building

* GRU history windows are stored left-padded (most recent last); `pack_padded_sequence` expects valid steps first — fixed by gathering a per-row shift before packing (a wrong fix would have fed padding into the recurrent state).
* OPE "support" was first defined as the target giving zero probability to a logged action; that only zeroes weights (an ESS problem, now `zero_target_fraction`) — the invalid case is target mass on behaviour-impossible actions, which is what `unsupported_fraction` now measures.
* SNIPS is undefined when every trajectory weight is zero (deterministic targets at horizon 48); it is now reported as unavailable with a warning instead of aborting the report.
* The first heuristic threshold (1.0) used only 28 % of the budget; the benchmark uses 0.6 (documented; not tuned against PPO).
* A stray wall-clock measurement suggested training took minutes; profiling showed 20 trainer steps in 0.36 s — the pipe, not the code.

### 45.15 Commands

```
forgeline allocation benchmark --config configs/allocation/benchmark.yaml
forgeline allocation benchmark --config configs/allocation/benchmark_tiny_cpu.yaml
forgeline train configs/allocation/ppo_mlp.yaml | ppo_gru.yaml | ppo_tiny_cpu.yaml
forgeline allocation ope --log runs/allocation-benchmark/logged_trajectories.jsonl --target <checkpoint>|dual|threshold [--max-weight 20]
forgeline allocation ab --incumbent dual --challenger <checkpoint> --episodes 400 --gate-metrics gate.json [--metrics local]
forgeline allocation shadow --incumbent dual --candidate <checkpoint> --episodes 100
forgeline registry register … --metrics "$(cat gate.json)"; forgeline registry promote --id <id> --to champion --gate configs/allocation/promotion_gate.yaml
python examples/06_allocation_lifecycle.py
```

### 45.16 Coexistence with the generative stack

The decisioning subsystem shares the manifest/`RunContext`/`Trainer`/checkpoint/metrics/registry/gate machinery with the language-model post-training stack; `AllocationPPOAlgorithm` is a sibling of `PPOAlgorithm` (LM) under the same `PostTrainingAlgorithm` contract, and the A/B harness consumes the same `stable_bucket` routing hash as serving. No new generative-model experiment was added for this extension.

### 45.17 Truth boundaries (this subsystem)

Safe: "simulated constrained sequential-decision environment", "general resource-allocation benchmark", "controlled offline A/B experiment", "policy-dependent simulator", "implemented and verified RL / sequence / online-optimisation / OPE / experimentation mechanisms", "RL policies did not beat the online pacer at this training budget". Unsafe: advertiser workloads, real bidding traffic, real campaign budgets, live delivery, production A/B tests, SOTA bidding, production impact, any claim that the sequence model beats the online optimizer.
