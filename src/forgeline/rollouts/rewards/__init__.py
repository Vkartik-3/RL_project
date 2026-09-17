"""Reward providers: rules, verifier adapters, composites, process rewards, critics, learned models."""

from typing import Any, Dict

from forgeline.core.errors import ConfigError
from forgeline.rollouts.rewards.composite import CallableReward, CompositeReward, TagFormatReward, VerifierReward, score_trajectories
from forgeline.rollouts.rewards.critic import CriticReward, heuristic_critic_score
from forgeline.rollouts.rewards.learned import BlendedReward, RewardModelProvider
from forgeline.rollouts.rewards.process import ProcessReward, compute_process_reward, parse_steps, score_step
from forgeline.rollouts.rewards.rules import ConstantReward, LengthReward, RepetitionReward, TextFormatReward
from forgeline.rollouts.verifiers import build_verifier

RULE_REWARDS = {"length": LengthReward, "text_format": TextFormatReward, "repetition": RepetitionReward, "constant": ConstantReward}


def build_reward_provider(config: Dict[str, Any]):
    """Build a provider from a manifest ``reward`` mapping.

    Examples::

        {"type": "verifier", "verifier": "math", "target_key": "answer"}
        {"type": "rule", "name": "length", "target_length": 64}
        {"type": "composite", "providers": [{...,"weight":0.7}, {...,"weight":0.3}]}
        {"type": "process", "gamma": 0.9, "step_weight": 0.5}
        {"type": "tagged", "format_bonus": true}
        {"type": "critic", "mode": "heuristic"}
    """
    kind = config.get("type")
    if kind == "verifier":
        params = {k: v for k, v in config.items() if k not in ("type", "verifier", "target_key", "weight")}
        return VerifierReward(build_verifier(config["verifier"], **params), target_key=config.get("target_key", "answer"),
                              weight=float(config.get("weight", 1.0)))
    if kind == "rule":
        params = {k: v for k, v in config.items() if k not in ("type", "name", "weight")}
        return RULE_REWARDS[config["name"]](**params)
    if kind == "composite":
        # weights are applied once, by the composite
        providers = [(build_reward_provider({k: v for k, v in p.items() if k != "weight"}), float(p.get("weight", 1.0)))
                     for p in config["providers"]]
        clip = tuple(config["clip"]) if config.get("clip") else None
        return CompositeReward(providers, clip=clip)
    if kind == "process":
        return ProcessReward(gamma=float(config.get("gamma", 0.9)), step_weight=float(config.get("step_weight", 1.0)),
                             target_key=config.get("target_key", "answer"), execute_code=bool(config.get("execute_code", True)))
    if kind == "tagged":
        base = VerifierReward(build_verifier("tagged_answer"), target_key=config.get("target_key", "answer"))
        if config.get("format_bonus", True):
            return CompositeReward([(base, 1.0), (TagFormatReward(), 1.0)], name="tagged+format")
        return base
    if kind == "critic":
        return CriticReward(mode=config.get("mode", "heuristic"))
    if kind == "synthesis_rule":
        from forgeline.domains.synthesis.reward import SynthesisRuleReward

        return SynthesisRuleReward(decode_from_text=bool(config.get("decode_from_text", True)))
    raise ConfigError(f"unknown reward type {kind!r}")


__all__ = [
    "CallableReward", "CompositeReward", "TagFormatReward", "VerifierReward", "score_trajectories", "CriticReward",
    "heuristic_critic_score", "BlendedReward", "RewardModelProvider", "ProcessReward", "compute_process_reward",
    "parse_steps", "score_step", "ConstantReward", "LengthReward", "RepetitionReward", "TextFormatReward",
    "RULE_REWARDS", "build_reward_provider",
]
