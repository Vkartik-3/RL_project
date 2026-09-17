# Inference

## What

A KV-cached inference engine with continuous batching, paged block accounting and a request lifecycle, plus single-prompt text generation and speculative decoding.

## Why

Serving many concurrent requests needs admission control, bounded memory and batched decoding; the same engine is the backend for the HTTP server.

## How

```
submit(prompt_tokens, SamplingParams) ──► Scheduler.pending
step():
  admit()   FIFO while running < max_batch and the block allocator can hold prompt + 1 token
            (requests whose prompt + max_tokens exceed max_seq_len fail immediately)
  prefill   model.prefill(prompt) → first-token logits + per-layer cache
  decode    group running sequences by cache length → one batched model.step per group
            (singletons decode alone); sample; append; allocate one more token of blocks
  retire    stop token / max_tokens / max_seq_len / cancel → free blocks immediately
```

| Component | Module | Responsibility |
|---|---|---|
| `GenerationRequest`, `SamplingParams`, `RequestStatus` | `inference/requests.py` | per-request state, token queue for streaming, timing |
| `PagedBlockAllocator` | `inference/paged_memory.py` | fixed-size blocks, per-sequence block tables, capacity checks, utilisation |
| `Scheduler` | `inference/scheduler.py` | pending/running sets, admission, retirement, cancellation, stats |
| `InferenceEngine` | `inference/engine.py` | prefill, grouped batched decode, sampling, failure isolation, background loop, spans |
| `generate_text` | `inference/generation.py` | single prompt with timing; optional speculative generator |

Sampling uses the same filter implementation as training rollouts (temperature, top-k, top-p, min-p, repetition penalty).

Decoding correctness: batched engine outputs equal single-sequence greedy `generate` outputs token-for-token (tested for GQA and latent attention).

## Configuration

`InferenceEngine(model, max_batch=8, max_seq_len=None, block_size=16, max_blocks=None, eos_token_id=None)`.

CLI: `forgeline generate --checkpoint DIR --prompt "…" [--temperature 0] [--top-p 0.9] [--min-p 0.05] [--repetition-penalty 1.2] [--draft-checkpoint DIR --spec-k 5] [--interactive] [--no-cache]`.

## Failure modes

| Condition | Behaviour |
|---|---|
| invalid sampling parameters, empty prompt | `ServingError` at submit |
| prompt + max_tokens > max_seq_len | request fails with reason `length` and an error message |
| blocks exhausted | admission waits; mid-decode exhaustion fails that request (`OutOfBlocksError`) |
| model error for one sequence | that request fails; others continue |
| engine not idle after `max_steps` | `ServingError` |

## Local validation

`tests/unit/test_inference_serving.py` (allocator capacity, scheduler admission/rejection, `max_batch` honoured, batched = greedy equivalence, stop tokens, cancellation, blocks freed) and `tests/failure/test_failure_modes.py::test_engine_isolates_failing_sequence`.

## Hardware requirements

CPU or a single GPU.

## Limitations

* Caches are per-sequence tensors; the paged allocator bounds admission but decode does not use block-indexed attention kernels.
* Sequences are batched only with others at the same cache length.
* Native sparse attention models are not supported by the cached engine.
