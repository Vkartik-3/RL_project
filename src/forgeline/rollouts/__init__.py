"""Rollouts: single-turn sampling, multi-turn tool agents, verifiers and rewards."""

from forgeline.core.protocols import RewardResult, Trajectory
from forgeline.rollouts.agent import AGENT_SYSTEM_PROMPT, AgentRolloutConfig, AgentRolloutEngine, build_agent_prompt
from forgeline.rollouts.engine import RolloutConfig, RolloutEngine, stack_group, stack_old_logprobs

__all__ = [
    "RewardResult", "Trajectory", "AGENT_SYSTEM_PROMPT", "AgentRolloutConfig", "AgentRolloutEngine",
    "build_agent_prompt", "RolloutConfig", "RolloutEngine", "stack_group", "stack_old_logprobs",
]
