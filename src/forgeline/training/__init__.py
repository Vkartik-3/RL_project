"""Training stages: pretraining, SFT, distillation, reward modelling and the post-training stack."""

from forgeline.training.common.trainer import PostTrainingAlgorithm, StepResult, Trainer
from forgeline.training.dapo import DAPOAlgorithm, DAPOConfig
from forgeline.training.distill import DistillationAlgorithm, DistillConfig, distillation_loss
from forgeline.training.dpo import DPOAlgorithm, DPOConfig, dpo_loss
from forgeline.training.grpo import GRPOAlgorithm, GRPOConfig
from forgeline.training.ppo import PPOAlgorithm, PPOConfig, ppo_clipped_objective
from forgeline.training.pretrain import PretrainAlgorithm, PretrainConfig
from forgeline.training.reward_model import RewardModelAlgorithm, RewardModelConfig
from forgeline.training.rlaif import RLAIFConfig, RLAIFTrainer, run_constitutional, run_pairwise_rlaif
from forgeline.training.rlvr import RLVRConfig, build_rlvr_algorithm, evaluate_pass_rate
from forgeline.training.sft import SFTAlgorithm, SFTConfig
from forgeline.training.star import HillClimbConfig, HillClimber, STaRConfig, STaRTrainer

__all__ = [
    "PostTrainingAlgorithm", "StepResult", "Trainer", "DAPOAlgorithm", "DAPOConfig", "DistillationAlgorithm",
    "DistillConfig", "distillation_loss", "DPOAlgorithm", "DPOConfig", "dpo_loss", "GRPOAlgorithm", "GRPOConfig",
    "PPOAlgorithm", "PPOConfig", "ppo_clipped_objective", "PretrainAlgorithm", "PretrainConfig", "RewardModelAlgorithm",
    "RewardModelConfig", "RLAIFConfig", "RLAIFTrainer", "run_constitutional", "run_pairwise_rlaif", "RLVRConfig",
    "build_rlvr_algorithm", "evaluate_pass_rate", "SFTAlgorithm", "SFTConfig", "HillClimbConfig", "HillClimber",
    "STaRConfig", "STaRTrainer",
]
