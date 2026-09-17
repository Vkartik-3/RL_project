# Models

## What

A native decoder-only transformer (`TransformerLM`) with configurable attention, feed-forward, mixture-of-experts and multi-token prediction; sampling and speculative decoding; LoRA and QLoRA adapters; and two policy backends that expose the same interface to training code.

## Why

A readable, dependency-free model makes every algorithm testable on a laptop CPU, while the HuggingFace backend runs the same algorithms on pretrained checkpoints.

## How

### Architecture options (`ModelSpec`)

| Feature | Fields | Implementation |
|---|---|---|
| Pre-norm RMSNorm blocks, tied embeddings | — | `transformer/block.py`, `transformer/norm.py` |
| Multi-head / grouped-query attention | `n_head`, `n_kv_head` | `attention/causal.py` (SDPA fast path when no cache/window/cap) |
| Rotary embeddings + linear / YaRN scaling | `use_rope`, `rope_scaling_type`, `rope_scaling_factor` | `attention/rotary.py` |
| Learned absolute positions | `use_rope: false` | `TransformerLM.wpe` |
| Sliding-window attention, alternating global/local layers | `sliding_window`, `alternating_layers` | `attention/causal.py` |
| Attention logit soft-capping | `attn_logit_cap` | all attention modules |
| Multi-head latent attention (compressed KV cache, decoupled RoPE, absorbed decode) | `use_mla`, `kv_lora_rank`, `q_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim` | `attention/latent.py` |
| Native sparse attention (compressed + top-k + window branches with learned gates) | `use_nsa`, `nsa_block_size`, `nsa_top_k`, `nsa_window_size` | `attention/sparse.py` |
| SwiGLU or GELU FFN | `use_swiglu` | `transformer/feedforward.py` |
| Mixture of experts: softmax/sigmoid gates, top-k, shared experts, auxiliary-loss or bias-based (aux-loss-free) balancing, group-limited routing, first N dense layers | `use_moe`, `n_experts`, `n_experts_active`, `n_shared_experts`, `score_func`, `aux_loss_free`, `bias_update_speed`, `n_expert_groups`, `n_limited_groups`, `n_dense_layers` | `moe/gate.py`, `moe/layer.py` |
| Multi-token prediction heads | `n_predict_tokens`, `mtp_loss_weight` | `transformer/block.py::MTPModule` |
| Gradient checkpointing | `trainer.gradient_checkpointing` | `TransformerLM.enable_gradient_checkpointing` |

Presets (`forgeline presets --count-params`): `tiny`, `small`, `medium`, `large`, `xl`, `moe_32l`, `latent_moe_27l`, `sliding_window_32l`, `alternating_28l`.

### Forward APIs

* `forward(idx, targets)` → `(logits, loss)`; loss = token cross-entropy (`ignore_index=-1`) + MoE auxiliary losses + weighted MTP losses.
* `forward_hidden(idx)` → final hidden states.
* `full_logits(idx)` → logits at every position.
* `prefill(idx)` / `step(next_ids, cache, position)` → incremental decoding with per-layer caches (`(k, v)` for GQA, `(latent, k_rope)` for MLA).

### Generation

`models/generation/sampling.py`: temperature (0 = greedy), top-k, top-p, min-p, repetition penalty, stop tokens, optional sampled-token log-probs. The cached and uncached paths produce identical greedy outputs (tested).

`models/generation/speculative.py`: `SpeculativeGenerator` (draft model proposes `k` tokens, target verifies in one pass with rejection sampling; exposes acceptance rate) and `MTPSpeculativeGenerator` (the model's own MTP heads draft).

### Adapters and quantization

* `apply_lora(model, rank, alpha, dropout, target_modules)` wraps matching `nn.Linear` layers: `y = Wx + (x·A·B)·α/r`, `A` Kaiming-initialised, `B` zero (identity at init). `merge_lora`, `save_lora`/`load_lora`, `set_lora_enabled`, `trainable_parameter_report`.
* `apply_qlora` stores base weights as NF4 (`NF4Linear`: 16-level normal-float table, block size 64, two indices per byte, per-block abs-max) and trains full-precision LoRA on top. `quantization_report` gives the compression ratio.
* Details: [quantization](quantization.md).

### Policies (`models/policy.py`)

| | `NativePolicy` | `HFPolicy` |
|---|---|---|
| Model | `TransformerLM` | `AutoModelForCausalLM` (+ PEFT LoRA) |
| Reference policy | adapters disabled, or frozen deep copy (`with_frozen_reference`) | `disable_adapter()` |
| Log-probs | `cross_entropy` over response positions | same |
| Value head | optional `ValueHead` on last hidden state | same |
| Extra | — | `huggingface` extra; 8-bit loading; gradient checkpointing |
| Manifest | default | `backend: {type: huggingface, model_name: …}` |

### Reward model

`SequenceRewardModel(spec, pooling="last"|"mean")` — transformer backbone + scalar head; `bradley_terry_loss(chosen, rejected) = −log σ(r_c − r_r)`.

`EncoderRewardModel(backbone, objective="regression"|"preference")` (`huggingface` extra) scores text with a Transformer encoder: regression fits a sigmoid head with MSE to scalar targets (for example rule-reward pseudo-labels, `fit_regression`); preference fits a linear head with the Bradley-Terry loss on (chosen, rejected) texts (`fit_preference`). Combine with rules through `BlendedReward`.

### Loading

`load_model_from_checkpoint`, `load_policy_from_checkpoint`, `load_reward_model_from_checkpoint` rebuild models from checkpoint manifests (the spec and tokenizer metadata are stored inside the checkpoint). `torch.compile` prefixes are stripped.

## Configuration

```yaml
model_preset: tiny
model: {block_size: 64, dropout: 0.0, n_kv_head: 1}
trainer:
  adapter: {method: lora, rank: 8, alpha: 16.0, dropout: 0.05, target_modules: [q_proj, v_proj]}
  gradient_checkpointing: true
```

## Failure modes

| Condition | Error |
|---|---|
| inconsistent spec (e.g. `n_embd % n_head`, MLA+NSA) | `ConfigError` at construction |
| sequence longer than `block_size` | `ValueError` / `ModelError` |
| no layers match adapter targets | `ModelError` |
| state dict does not fit the model | `IncompatibleStateError` |
| HuggingFace backend without extra | `OptionalDependencyError` |

## Local validation

`tests/unit/test_models.py` runs forward, backward and generation for every attention/FFN/MoE/MTP/RoPE variant, checks cached = uncached decoding, sampling filters and both speculative generators. `tests/unit/test_adapters_quantization.py` checks LoRA identity-at-init, enable/disable, save/load, merge, NF4 round trips and QLoRA gradients.

## Hardware requirements

All variants run on CPU at tiny sizes. Large presets require GPUs; `moe_32l`, `latent_moe_27l`, `sliding_window_32l` and `alternating_28l` are multi-GPU scale.

## Limitations

* Native sparse attention has no incremental-decode cache path; it runs full-sequence forwards.
* The MoE layer dispatches experts with a Python loop (clear, not throughput-optimised).
* Sliding-window cached decoding masks by position but keeps the full cache in memory.
