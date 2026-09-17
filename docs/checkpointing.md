# Checkpointing and recovery

## What

Directory checkpoints with a manifest, atomic writes, rotation, best-checkpoint tracking, latest discovery, validation (completeness, sizes, spec, algorithm) and resume.

## Why

Long runs fail. A checkpoint must either be complete and verifiably compatible, or be rejected with a clear reason.

## How

```
<output_dir>/checkpoints/
  LATEST                    → name of the newest checkpoint
  step_00001000/
    manifest.json           format_version, step, algorithm, model_spec, tokenizer_meta, metadata, files {name: {size}}, complete
    model.pt                model state (plus adapter and value-head state for policies)
    optimizer.pt  scaler.pt  rng.pt  trainer.pt
    experiment.json         full experiment manifest
  best/                     copy of the best checkpoint (when requested)
  final/                    written at the end of fit()
```

* **Atomic save** — written to `<name>.tmp`, then `os.replace`; a crash never leaves a partial checkpoint under the final name.
* **Async save** — optional background thread; `wait()` surfaces errors.
* **Rotation** — keeps the newest `keep_last` `step_*` checkpoints.
* **Discovery** — `find_latest()` follows `LATEST`, falling back to the highest `step_*` with a manifest (`.tmp` directories are ignored).
* **Validation** — `CheckpointManager.validate(path, expected_spec, expected_algorithm)` checks manifest parse, `complete` flag, presence and exact byte size of each file, spec field equality and algorithm name.
* **State captured** — model (and adapter/value head), optimizer, grad scaler, Python/NumPy/Torch/CUDA RNG, step, best metric, tokenizer metadata, experiment manifest.
* **Resume** — `Trainer.resume(path=None)` restores all of the above; with no path, `auto_resume` loads the latest.
* **Compatibility helpers** — `compare_state_shapes`, `assert_compatible`, `finite_state`, `load_model_state` (strips `_orig_mod.` prefixes; raises `IncompatibleStateError`).

## Configuration

```yaml
trainer:
  checkpoint: {directory: checkpoints, save_every: 1000, keep_last: 3, async_save: false, resume_from: "", auto_resume: true}
```

CLI: `forgeline train m.yaml --resume runs/x/checkpoints/step_00001000`.

## Failure modes

| Condition | Error |
|---|---|
| path does not exist | `CheckpointNotFoundError` |
| no manifest / unreadable JSON / missing keys | `CheckpointCorruptError` |
| `complete: false`, missing file, size mismatch (truncated) | `CheckpointCorruptError` |
| unreadable tensor file | `CheckpointCorruptError` |
| format version, spec or algorithm mismatch | `CheckpointMismatchError` |
| state dict shapes/keys differ | `IncompatibleStateError` |

## Local validation

* `tests/unit/test_checkpoints.py` — save/load/rotation/latest/best, async save, compile prefixes, shape comparison.
* `tests/integration/test_checkpoint_resume.py` — train 3 steps, save, reload into fresh objects, continue 3 steps: losses equal uninterrupted training within 1e-5; LoRA adapter state restored on auto-resume.
* `tests/failure/test_failure_modes.py` — missing, bad metadata, truncated, missing model file, missing manifest, incomplete flag, crashed `.tmp` directory, mismatched spec, mismatched algorithm, incompatible state.

## Hardware requirements

Disk. Checkpoints are single-process `torch.save` files.

## Limitations

* FSDP checkpoints are saved from the wrapped model's state dict on the main process; sharded (per-rank) checkpoint files are not produced.
* Streaming-dataset position is not checkpointed.
