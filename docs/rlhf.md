# RLHF: reward models, DPO, PPO, GRPO, DAPO and AI feedback

## What

Learning from preferences and scalar rewards.

## Why

Different alignment signals (human or AI preferences, learned rewards, rule rewards) need different estimators; a single rollout/reward interface lets them be swapped by configuration.

## How

### Rollout → reward → objective

```
prompt ──► RolloutEngine (G samples per prompt, old log-probs recorded)
       ──► RewardProvider.score(trajectory) → RewardResult(value, passed, components)
       ──► advantages (value baseline | group-relative | none)
       ──► objective ──► Trainer step
```

`Trajectory` holds prompt and response token ids, decoded text, the task record, agent segments/tool results, truncation flag and sampled-time log-probs.

### Reward models

`RewardModelAlgorithm` trains `SequenceRewardModel` with `−log σ(r_chosen − r_rejected)` and reports pairwise accuracy. `RewardModelProvider` turns a trained model into a reward for RL. `BlendedReward` mixes a rule score and a neural score (`(1 − w)·rule + w·neural`, clipped to [0, 1]).

### DPO

```
L = −log σ( β·(log π(y_w|x) − log π_ref(y_w|x)) − β·(log π(y_l|x) − log π_ref(y_l|x)) )
```

| Option | Values |
|---|---|
| `beta` | > 0 |
| `logprob_reduction` | `sum` (standard) · `mean` (length-normalised) |
| `reference_free` | drop reference terms |
| `label_smoothing` | conservative DPO in [0, 0.5) |

Metrics: implicit chosen/rejected rewards, margin, preference accuracy.

### PPO

Value-head actor-critic. Advantages `A = normalise(r − V_old)`; loss per PPO epoch:

```
policy  = mean( −min(ρ·A, clip(ρ, 1−ε, 1+ε)·A) )
value   = MSE(V, r)
total   = policy + c_v·value − c_e·entropy (+ β·KL when kl_mode = penalty)
```

| Option | Values | Meaning |
|---|---|---|
| `ratio_level` | `sequence_mean` | `ρ = exp(mean_t(log π − log π_old))` per sequence |
| | `token` | `ρ_t` per token, loss averaged over response tokens |
| `kl_mode` | `monitor` | KL = mean(log π_old − log π_ref), logged only |
| | `penalty` | `β·clamp(mean(log π_ref − log π), 0)` added to the loss |
| | `reward` | `r ← r − β·mean(log π_old − log π_ref)` before advantages |
| `entropy_mode` | `sampled_logprob` | `−mean(log π(y_t))` |
| | `full` | exact entropy of the full distribution |

`compute_gae(rewards, values, γ, λ)` is available for per-token rewards.

### GRPO

For each prompt, `G` completions; `A_i = (r_i − mean(r)) / (std(r) + ε)`.

| `objective` | Loss |
|---|---|
| `clipped` | `mean_i(−min(ρ_i·A_i, clip(ρ_i)·A_i)) + β·clamp(mean_i mean_t(log π_ref − log π), 0)` (`ratio_level` `sequence_mean` or `token`; `kl_estimator` `logprob_diff` or `k3`) |
| `reinforce` | `Σ_i −A_i · Σ_t log π(y_{i,t})` (no clipping, no KL) |

`agent: true` runs multi-turn tool episodes (see [RLVR](rlvr.md)).

### DAPO

```
per-token surrogate  = −min(ρ_t·A, clip(ρ_t, 1−ε_low, 1+ε_high)·A)
loss = ( Σ_groups Σ_tokens surrogate + β·Σ(log π − log π_ref) − c_e·Σ entropy ) / total_response_tokens
```

* clip-higher (`clip_eps_high ≥ clip_eps_low`);
* dynamic sampling: groups with zero reward variance are skipped;
* overlong shaping: `r += overlong_penalty · max(0, len/max_len − 1)` for truncated responses;
* optional entropy bonus.

### AI feedback

* **Self-judge rounds** (`RLAIFTrainer`): sample `K` candidates per task, score each with a 0–10 judge prompt (`parse_judge_score`), form pairs with score gap ≥ `pair_gap`, run reference-free DPO epochs; pairs written per round.
* **Pairwise judge** (`run_pairwise_rlaif`): all `K·(K−1)/2` comparisons, winner becomes `chosen`; policy-backed generator/judge (`make_policy_generator`, `make_policy_pairwise_judge`) or deterministic rule-based demo components.
* **Constitutional** (`run_constitutional`): initial response → per sampled principle critique → revision; outputs SFT records (final revision) and DPO records (revision ≻ initial). `score_critique_quality` checks critique substance.

## Configuration

```yaml
algorithm: ppo
reward: {type: rule, name: length, target_length: 12}
algorithm_params:
  prompts_per_step: 4
  samples_per_prompt: 1
  ppo_epochs: 3
  clip_ratio: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  kl_coef: 0.1
  ratio_level: sequence_mean
  kl_mode: monitor
  entropy_mode: sampled_logprob
  rollout: {max_new_tokens: 16, temperature: 0.7}
```

Reward specifications (`rollouts/rewards/__init__.py::build_reward_provider`):

| `type` | Parameters |
|---|---|
| `rule` | `name: length|text_format|repetition|constant` + its parameters |
| `verifier` | `verifier: math|exact_match|tagged_answer|code|format`, `target_key`, `weight` |
| `composite` | `providers: [ {…, weight} ]`, optional `clip: [lo, hi]` |
| `tagged` | `format_bonus: true|false` |
| `process` | `gamma`, `step_weight`, `execute_code` |
| `critic` | `mode: heuristic` |

## Failure modes

| Condition | Behaviour |
|---|---|
| reward NaN/inf | `InvalidRewardError` naming the provider |
| PPO policy without value head | `ConfigError` |
| GRPO/DAPO `group_size < 2` | `ConfigError` |
| DAPO `clip_eps_high < clip_eps_low` | `ConfigError` |
| all groups skipped | zero loss, `groups_skipped` metric |
| no reference policy for KL | `ConfigError` (policies create one automatically when needed) |

## Local validation

See `tests/integration/test_training_lifecycle.py`, `tests/unit/test_training_utils.py` (closed-form checks of PPO clipping, DAPO clip-higher, DPO loss at 0 and large margins, Bradley-Terry, GAE, group advantages) and `tests/unit/test_benchmark_semantics.py`.

## Hardware requirements

CPU for tiny policies. PPO holds policy, value head and reference log-probs; with LoRA the reference costs no extra parameters.

## Limitations

* Rollouts are generated per prompt group rather than fully batched across prompts.
* PPO value estimates are per sequence (last hidden state); per-token GAE is provided as a utility, not wired into the default PPO loss.
* LLM-based judges and critics are only as reliable as the judging model.
