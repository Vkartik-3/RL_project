"""Core: configuration, protocols, registry, runtime, lifecycle and errors."""

from forgeline.core.config import (
    AdapterConfig,
    CheckpointConfig,
    DataConfig,
    DistributedConfig,
    ExperimentManifest,
    ModelSpec,
    MODEL_PRESETS,
    OptimizerConfig,
    PrecisionConfig,
    ScheduleConfig,
    TrainerConfig,
    load_config,
    model_spec_from_preset,
)
from forgeline.core.errors import ForgelineError
from forgeline.core.lifecycle import RunContext, Stage
from forgeline.core.protocols import (
    EvaluationResult,
    GenerationSettings,
    PolicyModel,
    RewardProvider,
    RewardResult,
    Trajectory,
    Verifier,
)
from forgeline.core.registry import Registry
from forgeline.core.runtime import resolve_device, resolve_dtype, seed_everything

__all__ = [
    "AdapterConfig", "CheckpointConfig", "DataConfig", "DistributedConfig", "ExperimentManifest",
    "ModelSpec", "MODEL_PRESETS", "OptimizerConfig", "PrecisionConfig", "ScheduleConfig",
    "TrainerConfig", "load_config", "model_spec_from_preset", "ForgelineError", "RunContext",
    "Stage", "EvaluationResult", "GenerationSettings", "PolicyModel", "RewardProvider",
    "RewardResult", "Trajectory", "Verifier", "Registry", "resolve_device", "resolve_dtype",
    "seed_everything",
]
