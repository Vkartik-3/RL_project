# Quantization and export

## What

NF4 4-bit weight storage (for QLoRA), GGUF export in FP16, Q8_0 and Q4_0, adapter merging, and optional 8-bit loading for HuggingFace policies.

## Why

Memory-efficient fine-tuning and portable inference files without GPU-specific number formats.

## How

### NF4 (`models/quantization/nf4.py`)

* Weights flattened into blocks of 64; each block scaled by its abs-max into [−1, 1].
* Each value mapped to the nearest of 16 normal-float levels; two 4-bit indices packed per byte.
* `NF4Linear` stores packed indices + per-block scales as buffers and dequantizes in `forward`; bias stays full precision and frozen.
* `QLoRALinear` = `NF4Linear` base + trainable LoRA; `quantization_report` reports layers, bytes and compression vs fp16.

### GGUF (`models/quantization/gguf.py`)

| Format | Matrices | 1-D tensors |
|---|---|---|
| `none` (FP16) | float16 | float32 |
| `q8_0` | blocks of 32: fp16 scale (abs-max/127) + 32 int8 | float32 |
| `q4_0` | blocks of 32: fp16 scale (abs-max/7) + 16 bytes, values stored as q+8, low nibbles hold elements 0–15, high nibbles 16–31 | float32 |

* Header: magic, version 3, tensor count, metadata (`general.architecture = forgeline`, context length, embedding size, block count, head counts, vocabulary, feature flags, alignment 32).
* Tensor infos with dimensions and correct aligned offsets relative to the data section.
* Multi-token-prediction heads are excluded (training-only).
* `read_gguf_header` and `read_gguf_tensor` (with Q8_0/Q4_0 dequantization) read files back.

The file uses the `forgeline` architecture tag and native tensor names; running it in third-party GGUF runtimes requires a matching architecture definition there.

### Adapter merging

`forgeline export --merge-adapters` or `merge_lora(model)` folds `A·B·α/r` into base weights before export.

### INT8 loading

`HFPolicy(..., load_in_8bit=True)` passes 8-bit loading through to `transformers` (requires `bitsandbytes` and CUDA).

## Configuration

`forgeline export --checkpoint DIR --output model.gguf --quantize none|q8_0|q4_0 [--merge-adapters]`; QLoRA training via `trainer.adapter.method: qlora`.

## Failure modes

`QuantizationError` (empty tensors), `ExportError` (unknown format, unreadable file, unsupported tensor type, tensor not found).

## Local validation

`tests/unit/test_adapters_quantization.py`: NF4 round-trip error, QLoRA gradients and compression, GGUF export and read-back for all formats with reconstruction tolerances, aligned increasing offsets, Q8_0/Q4_0 block round trips.

## Hardware requirements

CPU for all quantization and export. 8-bit HuggingFace loading needs CUDA.

## Limitations

* NF4 dequantization happens in Python/PyTorch each forward (no fused kernels), trading speed for memory.
* No accuracy or speed measurements of quantized models are recorded.
* FP8 formats are not provided.
