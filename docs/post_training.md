# Post-training

## What

Preference optimisation, reward modelling, reinforcement learning and self-improvement methods, all implemented as `PostTrainingAlgorithm`s (or loops built from them):

| Method | Module | Summary |
|---|---|---|
| Reward model | `training/reward_model.py` | Bradley-Terry on preference pairs |
| DPO | `training/dpo.py` | direct preference optimisation (sum or length-normalised; optional reference-free, label smoothing) |
| PPO | `training/ppo.py` | clipped policy gradient with value head, KL and entropy options |
| GRPO | `training/grpo.py` | group-relative advantages, no value model; single-turn, tool-agent and REINFORCE modes |
| DAPO | `training/dapo.py` | clip-higher, dynamic sampling, token-level loss, overlong shaping, entropy bonus |
| RLVR | `training/rlvr.py` | GRPO/DAPO driven by verifiers |
| Process-reward GRPO | `grpo` + `ProcessReward` | dense step-level rewards |
| Critic-augmented agent GRPO | `grpo` + `CompositeReward(verifier, CriticReward)` | verifiable + heuristic/LLM critic reward |
| AI feedback | `training/rlaif.py` | self-judge rounds → reference-free DPO; pairwise judge → preference data; constitutional critique→revise |
| STaR | `training/star.py` | rejection-sampled rationales → SFT, repeated |
| Hill climbing | `training/star.py` | rejection-sample high-reward tool trajectories into the dataset, then agent GRPO rounds |

Details: [RLHF](rlhf.md) (reward models, DPO, PPO, GRPO, DAPO, AI feedback) and [RLVR](rlvr.md) (verifiers, tools, process rewards, STaR, hill climbing).

## Why

These methods share rollouts, rewards, reference policies and optimisation machinery; implementing them against one policy interface and one trainer keeps their mathematics isolated and comparable.

## How

Every step-based method follows: `collect` (sample prompts/pairs, run rollouts, score, compute advantages, record old/reference log-probs) → `loss` (objective) → shared `Trainer` step. Reference policies are LoRA-disabled views when adapters are used, otherwise frozen copies.

## Configuration

`configs/post_training/*.yaml`; `forgeline train <manifest>`. AI-feedback data generation: `forgeline rlaif pairwise|constitutional`.

## Failure modes

Invalid rewards (NaN/inf) raise `InvalidRewardError`; verifier exceptions become failed results; generation failures raise `RolloutError`; zero-variance groups are skipped (DAPO, or GRPO with `skip_zero_variance_groups`) and yield a zero loss that keeps autograd consistent.

## Local validation

`tests/integration/test_training_lifecycle.py` runs every method on the tiny model: DPO raises the preference margin and reaches accuracy 1.0 on a toy set; the reward model ranks chosen above rejected; PPO in three configurations; GRPO in three configurations; agent GRPO with the tool loop; DAPO dynamic sampling; RLVR with format verification. `tests/unit/test_benchmark_semantics.py` pins the objectives behind recorded measurements.

## Hardware requirements

CPU at tiny scale. 7B-class policies via `HFPolicy` require a GPU with enough memory for the base model plus LoRA states (fp16) or 8-bit loading.

## Limitations

See [rlhf.md](rlhf.md#limitations) and [rlvr.md](rlvr.md#limitations).
