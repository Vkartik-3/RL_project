"""Reinforcement learning with verifiable rewards.

RLVR is not a new objective: it is GRPO or DAPO driven by deterministic
verifiers instead of a learned reward model. This module builds the reward
(task verifier + optional format verifier, weighted) and the chosen optimiser
from a compact config, and exposes a quick pass-rate evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from forgeline.core.errors import ConfigError
from forgeline.data.records import VerifiableTask
from forgeline.models.policy import NativePolicy
from forgeline.rollouts.engine import RolloutConfig
from forgeline.rollouts.rewards.composite import CompositeReward, VerifierReward
from forgeline.rollouts.verifiers import FormatVerifier, build_verifier
from forgeline.training.dapo import DAPOAlgorithm, DAPOConfig
from forgeline.training.grpo import GRPOAlgorithm, GRPOConfig


@dataclass
class RLVRConfig:
    task_type: str = "math"  # math | code | tagged_answer | exact_match
    optimizer: str = "dapo"  # dapo | grpo
    require_format: bool = False
    format_spec: str = "cot"
    correctness_weight: float = 0.7
    format_weight: float = 0.3
    verifier_params: Dict[str, Any] = field(default_factory=dict)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    dapo: DAPOConfig = field(default_factory=DAPOConfig)

    def validate(self) -> None:
        if self.optimizer not in ("dapo", "grpo"):
            raise ConfigError("RLVR optimizer must be dapo|grpo")
        if self.task_type not in ("math", "code", "tagged_answer", "exact_match"):
            raise ConfigError("task_type must be math|code|tagged_answer|exact_match")


def build_verifiable_reward(cfg: RLVRConfig):
    target_key = "test_code" if cfg.task_type == "code" else "answer"
    correctness = VerifierReward(build_verifier(cfg.task_type, **cfg.verifier_params), target_key=target_key)
    if not cfg.require_format:
        return correctness
    fmt = VerifierReward(FormatVerifier(cfg.format_spec), target_key="format_spec")
    return CompositeReward([(correctness, cfg.correctness_weight), (fmt, cfg.format_weight)], name="rlvr")


def build_rlvr_algorithm(policy: NativePolicy, tasks: Sequence[VerifiableTask], cfg: RLVRConfig,
                         rollout: Optional[RolloutConfig] = None):
    cfg.validate()
    reward = build_verifiable_reward(cfg)
    task_dicts = [t.to_dict() for t in tasks]
    for d in task_dicts:
        d.setdefault("format_spec", cfg.format_spec)
    prompts = [torch.tensor(policy.encode(t.prompt), dtype=torch.long) for t in tasks]
    if cfg.optimizer == "dapo":
        alg = DAPOAlgorithm(policy, prompts, reward, cfg.dapo, rollout, tasks=task_dicts)
    else:
        alg = GRPOAlgorithm(policy, prompts, reward, cfg.grpo, rollout, tasks=task_dicts)
    alg.name = "rlvr"
    return alg


@torch.no_grad()
def evaluate_pass_rate(policy: NativePolicy, tasks: Sequence[VerifiableTask], reward, max_new_tokens: int = 64,
                       temperature: float = 0.0, limit: Optional[int] = None) -> Dict[str, float]:
    """Greedy-decode each task once; report pass rate, mean reward and malformed-output rate."""
    from forgeline.rollouts.engine import RolloutEngine

    engine = RolloutEngine(policy, RolloutConfig(group_size=1, max_new_tokens=max_new_tokens, temperature=temperature,
                                                 record_old_logprobs=False))
    n = 0
    passed = 0
    malformed = 0
    total = 0.0
    for task in list(tasks)[: limit or None]:
        t = engine.rollout(torch.tensor(policy.encode(task.prompt)), task.to_dict())[0]
        r = reward.score(t)
        n += 1
        total += r.value
        passed += int(bool(r.passed))
        if r.info.get("predicted", "x") is None or "error" in r.info:
            malformed += 1
    return {"pass_rate": passed / max(n, 1), "reward": total / max(n, 1), "malformed_rate": malformed / max(n, 1), "n": n}
