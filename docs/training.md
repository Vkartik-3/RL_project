# Training

## What

The shared training lifecycle (`Trainer`) and the supervised stages built on it: language-model pretraining, supervised fine-tuning (full, LoRA, QLoRA) and knowledge distillation.

## Why

Optimizer grouping, schedules, precision, accumulation, clipping, checkpointing and metrics are identical for every stage; algorithms supply only data collection and the loss.

## How

### Trainer

`Trainer(ctx, algorithm, config=None, checkpoint_manager=None, strategy=None)`

* **Optimizer** — AdamW (or Adam) with weight decay applied only to parameters with `dim >= 2` (`decay_only_matrices`); fused kernels on CUDA.
* **Schedules** (`training/common/schedule.py`):
  * `cosine` — linear warmup to peak, cosine decay to `min_lr` at `decay_steps`, then flat;
  * `wsd` — warmup, stable at peak until `decay_steps × wsd_stable_fraction`, cosine decay to `min_lr`;
  * `constant`, `linear_decay`.
* **Precision** — autocast bf16/fp16 on CUDA (bf16 also on CPU when requested); `GradScaler` enabled automatically for fp16.
* **Accumulation** — `gradient_accumulation_steps` micro-batches per optimizer step; each micro loss divided by the count.
* **Clipping** — global-norm clipping at `grad_clip` (0 disables) after unscaling.
* **Safety** — non-finite losses abort the run with `ConfigError` and a `run.failed` event.
* **Evaluation** — every `eval_every` steps and at the end; when a `loss` (or `primary`) metric improves, a `best_candidate` checkpoint is saved.
* **Checkpoints** — every `save_every` steps and `final` at the end; RNG, optimizer, scaler and trainer state are included. `resume(path=None)` loads an explicit path or the latest checkpoint and validates spec and algorithm.

### Pretraining (`training/pretrain.py`)

`PretrainAlgorithm(model, train_corpus, PretrainConfig(batch_size, eval_batches, seed), val_corpus)`: random windows from `MemmapCorpus` (or batches from a streaming loader); loss = model cross-entropy; logs EMA loss and tokens seen; evaluation reports `train_loss`, `val_loss`.

### Supervised fine-tuning (`training/sft.py`)

`SFTAlgorithm(policy, SupervisedDataset, SFTConfig(batch_size, max_length, mask_prompt, seed))`: epoch-shuffled batches, token cross-entropy on response tokens only (prompt and padding masked). With `trainer.adapter.method: lora|qlora` only adapter parameters train; base weights stay frozen (tested).

### Distillation (`training/distill.py`)

`DistillationAlgorithm(student, teacher, corpus, DistillConfig(temperature, alpha, batch_size, feature_weight))`:

`L = α · T² · KL(softmax(teacher/T) ‖ softmax(student/T)) + (1 − α) · CE(student, labels)` plus optional `feature_weight · MSE(proj(student_hidden_l), teacher_hidden_l)` across layers. Student and teacher must share a vocabulary.

### Mixed precision and memory

| Mechanism | How to enable |
|---|---|
| bf16 / fp16 autocast | `trainer.precision.dtype` |
| gradient scaler | automatic for fp16 (`grad_scaler: on|off|auto`) |
| gradient checkpointing | `trainer.gradient_checkpointing: true` |
| gradient accumulation | `trainer.gradient_accumulation_steps` |
| LoRA / QLoRA (NF4 base) | `trainer.adapter` |
| sharding | `distributed.strategy: fsdp|deepspeed` |
| torch.compile | `trainer.compile_model: true`, `trainer.compile_backend: inductor|eager|aot_eager` — each trained module is compiled in place (`nn.Module.compile`), so state-dict keys and checkpoints are unchanged |
| merged adapter checkpoint | `trainer.adapter.merge_on_save: true` writes `final_merged` (LoRA folded into base weights) next to `final` |

## Configuration

`configs/training/*.yaml`. Command: `forgeline train <manifest> [--max-steps N] [--resume PATH] [--metrics local|none|wandb|tensorboard]`.

## Failure modes

Non-finite loss, empty datasets, missing corpora, spec/algorithm mismatch on resume, and missing checkpoints raise typed errors (see [checkpointing](checkpointing.md)).

## Local validation

`tests/integration/test_training_lifecycle.py` (pretraining loss decreases; SFT full/LoRA/QLoRA with frozen base; distillation) and `tests/integration/test_checkpoint_resume.py` (save at step 3, reload, continue — losses match uninterrupted training to 1e-5; compiled training with the eager backend produces losses identical to uncompiled training and reloadable checkpoints; `merge_on_save` produces a plain checkpoint whose logits equal the adapter model's).

## Hardware requirements

Tiny presets: CPU. `large` pretraining with batch 8 × 2048 tokens in bf16 used about 42 GB on a 46 GB GPU (see `benchmarks/training/fineweb_edu_large_a40`).

## Limitations

* `torch.compile` is not combined with FSDP wrapping.
* Streaming loaders are consumed sequentially; resuming does not fast-forward a streaming iterator.
