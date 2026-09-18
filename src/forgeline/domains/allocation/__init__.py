"""Budgeted sequential allocation: a finite-horizon, budget-constrained decision environment with policy-dependent
dynamics, heuristic / online-optimisation / PPO (stateless and sequence-conditioned) policies, logged trajectories,
offline policy evaluation, a hindsight oracle, simulated A/B experiments and shadow evaluation."""

from forgeline.domains.allocation.env import ACTION_NAMES, OBS_DIM, OBS_NAMES, AllocationEnvConfig, BudgetedAllocationEnv, VectorEnv, episode_seed
from forgeline.domains.allocation.experiment import ABConfig, ABResult, ShadowResult, assign_arms, permutation_test, run_ab_experiment, shadow_evaluate
from forgeline.domains.allocation.ope import OPEConfig, OPEError, OPEReport, evaluate_policy, evaluate_target_policy, target_probabilities
from forgeline.domains.allocation.oracle import hindsight_upper_bound, oracle_for_seed, oracle_values
from forgeline.domains.allocation.policies import (
    DualPacingPolicy, EpsilonMixPolicy, GRUActorCritic, MLPActorCritic, NeuralPolicy, Policy, PolicyNetConfig, ThresholdPacingPolicy,
    build_actor_critic,
)
from forgeline.domains.allocation.ppo import AllocationPPOAlgorithm, AllocationPPOConfig, build_allocation_ppo, load_allocation_policy
from forgeline.domains.allocation.rollout import Episode, LoggedStep, aggregate_metrics, episode_metrics, read_episodes, run_episodes, seeds_for, write_episodes

__all__ = [n for n in dir() if not n.startswith("_")]
