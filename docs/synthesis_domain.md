# Synthesis domain

## What

A task family for optimising pharmaceutical synthesis conditions (temperature, time, catalyst loading, solvent ratio) against yield, selectivity, safety and step count.

## Why

It is a compact, fully verifiable environment for exercising SFT, preference optimisation and RL with structured (JSON) outputs, and a reference for adding new domains.

## How

| Component | Module | Behaviour |
|---|---|---|
| Record format | — | flat or nested (`parameters`, `outcomes`) trajectories |
| Rule reward | `reward.py` | `0.40·yield + 0.30·selectivity + 0.20·(1 − safety_risk) + 0.10/(1 + steps/10)`, clipped to [0, 1]; `SynthesisRuleReward` decodes generated JSON conditions and merges them into the record |
| Constraints | `constraints.py` | temperature 0–300 °C, time 0.25–48 h, catalyst 0.001–1 M, solvent 0.5–20 mL/mmol; missing/non-numeric → defaults; `validity_penalty` |
| Prompt / codec | `prompts.py` | prompt template; `conditions_to_json`; `decode_conditions` (first JSON object, fallback to defaults, clipped) |
| Features | `features.py` | 128-d state: 8 normalised features + 100-bit Morgan fingerprint (RDKit when installed, zeros otherwise) |
| Preferences / SFT data | `preferences.py` | per molecule: sort by yield, sample (top half, bottom half) pairs with a seeded RNG; `pairs_to_records`; `filter_high_reward`; `build_sft_records` |
| Simulated data | `generator.py` | bell-shaped temperature, S-shaped time, banded catalyst/solvent effects with noise; yield-series sweeps around per-molecule optima |
| Tabular actor-critic PPO | `tabular_ppo.py` | shared 128→256→256 MLP, softmax actor (32), critic; clipped surrogate + 0.5·value MSE − 0.01·entropy |

Datasets: `data/samples/synthesis/trajectories_literature.jsonl` (500 records, 5 molecules), `trajectories_improvable.jsonl` (500 simulated records with sub-optimal conditions), `trajectories_yield_series.jsonl` (50), `trajectories_labeled_small.jsonl` (100 flat records) and `trajectories_labeled_expanded.jsonl` (500 flat records). Nested records keep conditions under `parameters` and results under `outcomes`; flat records hold `yield`, `selectivity`, `safety_risk`, `steps` at the top level.

## Configuration

`forgeline synthesis-ppo --data data/samples/synthesis/trajectories_literature.jsonl --epochs 5 --output results.json`; `scripts/generate_synthesis_data.py --kind simulated --output data.jsonl`. Language-model policies use `SynthesisRuleReward` as the reward provider (`reward: {type: synthesis_rule}`); manifests with `data.kind: trajectories` build SFT records, preference pairs and prompts directly from trajectory files (`configs/post_training/synthesis_*_qwen7b_lora.yaml`).

## Failure modes

`DatasetError` for missing/empty trajectory files; `ConfigError` for invalid network or optimizer settings. Out-of-range generated conditions are clipped and flagged invalid.

## Local validation

`tests/unit/test_synthesis_domain.py`; `tests/unit/test_benchmark_semantics.py` reproduces every recorded record statistic exactly (held-out and training means on the literature, labeled and improvable files, including the earlier `0.1·(1 − steps/10)` efficiency term) and the leave-one-molecule-out values, and pins that generated text cannot change the rule reward of nested records; `examples/05_synthesis_generalization.py`.

## Hardware requirements

CPU.

## Limitations

* Evaluating a policy by scoring dataset records measures the data, not the policy (`RecordRewardSuite` marks such results `policy_dependent: false`); use `SynthesisRuleReward` on generated conditions to evaluate a policy.
* Outcomes of generated conditions are not simulated. The rule reward reads yield, selectivity, safety and steps from the record, and nested `outcomes` take precedence over any keys decoded from generated text, so for nested records the reward of a generated response equals the reward of the record it was prompted from. RL on this reward optimises nothing about the policy; a condition-aware outcome model is required before synthesis RL results can describe policy quality.
