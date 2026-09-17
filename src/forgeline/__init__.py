"""Forgeline — post-training, distributed training, evaluation, inference and
model-lifecycle framework for language models."""

__version__ = "0.1.0"

from forgeline.core.config import ExperimentManifest, ModelSpec, model_spec_from_preset

__all__ = ["__version__", "ExperimentManifest", "ModelSpec", "model_spec_from_preset"]
