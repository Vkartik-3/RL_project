# Benchmarks

Structured records of measurements. Each directory holds `results.json` (numbers + evidence level), `config.yaml` (the configuration that produced them), `environment.txt` (what is known about the hardware) and `summary.md`.

Evidence levels:

* `raw_log` — numbers copied from a retained machine-written results file.
* `documented_measurement` — numbers from a written run report without a retained machine log.
* `documented_without_raw_log` — not published as supported results.

`revalidation_required` is true when the computation behind a number has materially changed; no record currently requires it. `revalidation_recommended` marks records whose objective is preserved but whose loss normalisation constant or tokenisation path differs from the recorded run (see "Normalisation differences" below).

Every record also carries `ledger_status`, one status per reported number:

| Status | Meaning |
|---|---|
| `CARRY_FORWARD` | verified or recomputed exactly; cite as is |
| `CARRY_FORWARD_WITH_INTERPRETATION_NOTE` | real measurement; cite only with the stated meaning (e.g. record statistic, noisy batch mean, documented not logged) |
| `NEEDS_EVIDENCE_RECOVERY` | documented but the producing file/log is missing; not cited until recovered or rerun |
| `REVALIDATE` | computation changed enough that the number no longer describes the current code |
| `DO_NOT_USE` | the number does not measure what its label says (policy-independent, copied between rows, unit-mismatched, estimate) |

| Area | Record | Headline | Evidence | Headline status |
|---|---|---|---|---|
| Pretraining | training/char_small_cpu | val loss 1.479, 10.65M params, CPU | documented_measurement | carry forward with note |
| Pretraining | training/fineweb_edu_large_a40 | val loss 3.5834, 421M params, 20.7k tok/s (A40) | documented_measurement | carry forward with note |
| PEFT | peft/lora_qwen2_5_7b | 5.05M / 7.62B trainable (0.066%) | documented_measurement | carry forward (verified arithmetically) |
| SFT | post_training/sft_synthesis | loss 1.663 → 1.008 | raw_log | carry forward with note |
| DPO | post_training/dpo_synthesis | best epoch margin 0.00034 (loss ≈ ln 2) | raw_log | carry forward with note |
| PPO | post_training/ppo_synthesis | 200 iterations; rewards are record statistics | raw_log | pipeline evidence only |
| PPO pilot | post_training/ppo_synthesis_pilot | 50 iterations | raw_log | pipeline evidence only |
| Agent GRPO | rlvr/agent_grpo_gsm8k_tools | best batch reward 0.5575; accuracy 3/20 → 1/20; tool use 0.5 → 0.9 | raw_log | carry forward with note |
| Process-reward GRPO | rlvr/process_reward_grpo_gsm8k | best reward 1.0659 at iteration 1 | raw_log | carry forward with note |
| Hill climb | rlvr/hill_climb_gsm8k | accuracy 0.125 → 0.0 | raw_log | carry forward with note (negative result) |
| Tabular PPO | post_training/tabular_ppo_synthesis | record statistics; leave-one-molecule-out record means | raw_log | carry forward with note |
| AI feedback data | post_training/ai_feedback_data | 60 pairwise DPO pairs; 10 constitutional revisions | raw_log | carry forward (reproduced) |
| — | post_training/runs_without_retained_logs | GRPO 0.823, RLAIF 0.814, STaR 0.791, SFT 0.412 | documented_without_raw_log | needs evidence recovery |

## Normalisation differences

Forgeline keeps each recorded objective but normalises losses as means over groups/epochs. A positive constant factor `c` on the loss scales every gradient by `c`. Adam(W) divides the first moment by the square root of the second, so a constant `c` cancels except through `eps` and gradient-norm clipping (clipping to norm 1 happens before Adam and activates at different times when gradient norms are scaled). Decoupled weight decay is unaffected.

| Record | Recorded per-step gradient | Forgeline per-step gradient | Ratio recorded / Forgeline |
|---|---|---|---|
| PPO (3 PPO epochs, accumulation 4, one optimizer step) | Σ over 3 epochs of ∇L/4 = 0.75·∇L (parameters do not change between the epochs, so the ratio is exactly 1 and clipping is inactive) | mean over epochs = ∇L | 0.75 |
| Agent GRPO (2 problems × G=4, accumulation 4) | Σ_problems Σ_g ∇ℓ/(G·4) = 0.5·mean | mean over groups of mean over G | 0.5 (with 2 problems) |
| Process-reward GRPO (B problems × G=4) | Σ_problems Σ_g −A·∇Σ_t log π (no normalisation) | mean over groups of Σ_g | B |

With a constant ratio and gradient norms mostly above the clip threshold, the updates are identical up to `eps`; when norms straddle the threshold the two runs clip on different steps. The process-reward run also recomputed log-probs by re-tokenising decoded text, while Forgeline uses the generated token ids. These records are therefore marked `revalidation_recommended`, not `revalidation_required`.
