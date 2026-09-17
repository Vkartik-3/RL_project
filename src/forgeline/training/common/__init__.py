"""Shared training lifecycle and numerical utilities."""

from forgeline.training.common.advantages import compute_gae, group_relative_advantages, normalize_advantages
from forgeline.training.common.optim import build_optimizer
from forgeline.training.common.precision import Precision
from forgeline.training.common.schedule import learning_rate_at
from forgeline.training.common.trainer import PostTrainingAlgorithm, StepResult, Trainer

__all__ = ["compute_gae", "group_relative_advantages", "normalize_advantages", "build_optimizer", "Precision",
           "learning_rate_at", "PostTrainingAlgorithm", "StepResult", "Trainer"]
