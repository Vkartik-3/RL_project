"""Pharmaceutical-synthesis domain: trajectory schema, rule reward, constraints,
prompt/condition codec, state features, preference-pair construction and a
tabular actor-critic PPO baseline."""

from forgeline.domains.synthesis.constraints import CONDITION_BOUNDS, CONDITION_DEFAULTS, batch_validate, validate_conditions, validity_penalty
from forgeline.domains.synthesis.features import encode_trajectory, molecule_fingerprint
from forgeline.domains.synthesis.preferences import build_preference_pairs, build_sft_records, filter_high_reward
from forgeline.domains.synthesis.prompts import CONDITION_KEYS, build_synthesis_prompt, conditions_to_json, decode_conditions
from forgeline.domains.synthesis.reward import RULE_WEIGHTS, SynthesisRuleReward, extract_outcomes, rule_score, trajectory_to_text
from forgeline.domains.synthesis.tabular_ppo import TabularActorCritic, TabularPPOConfig, TabularPPOTrainer, load_trajectories

__all__ = [
    "CONDITION_BOUNDS", "CONDITION_DEFAULTS", "batch_validate", "validate_conditions", "validity_penalty",
    "encode_trajectory", "molecule_fingerprint", "build_preference_pairs", "build_sft_records", "filter_high_reward",
    "CONDITION_KEYS", "build_synthesis_prompt", "conditions_to_json", "decode_conditions", "RULE_WEIGHTS",
    "SynthesisRuleReward", "extract_outcomes", "rule_score", "trajectory_to_text", "TabularActorCritic",
    "TabularPPOConfig", "TabularPPOTrainer", "load_trajectories",
]
