"""Structured, actionable error types for every Forgeline subsystem.

Every error carries a short ``hint`` describing what the caller can do about
it. Subsystems raise the most specific subclass they can so that callers (and
tests) can distinguish, for example, a missing checkpoint from a corrupt one.
"""

from __future__ import annotations


class ForgelineError(Exception):
    """Base class for all framework errors."""

    hint: str = ""

    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.message = message
        if hint is not None:
            self.hint = hint

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.message} ({self.hint})" if self.hint else self.message


# ── configuration ───────────────────────────────────────────────────────────

class ConfigError(ForgelineError):
    """Invalid or inconsistent configuration."""

    hint = "Check the manifest / config values named in the message."


class UnknownPresetError(ConfigError):
    hint = "Use `forgeline presets` to list valid preset names."


# ── data ────────────────────────────────────────────────────────────────────

class DatasetError(ForgelineError):
    hint = "Verify the dataset path, format and schema."


class MalformedRecordError(DatasetError):
    """A single record failed schema validation."""

    hint = "Fix or drop the offending record; see `line`/`field` in the message."


class TokenizerError(ForgelineError):
    hint = "Ensure tokenizer metadata (meta.pkl) or the tokenizer name is available."


# ── models ──────────────────────────────────────────────────────────────────

class ModelError(ForgelineError):
    hint = "Check the model spec and checkpoint compatibility."


class IncompatibleStateError(ModelError):
    """State dict does not match the model architecture."""

    hint = "The checkpoint was produced by a different architecture; load with a matching spec."


class QuantizationError(ForgelineError):
    hint = "Check tensor shapes and the requested quantization format."


class ExportError(ForgelineError):
    hint = "Check the export format and destination path."


# ── rewards / rollouts ──────────────────────────────────────────────────────

class RewardError(ForgelineError):
    hint = "The reward provider returned a non-finite or malformed value."


class InvalidRewardError(RewardError):
    pass


class VerifierError(ForgelineError):
    hint = "A verifier raised while scoring; the completion is treated as failed."


class ToolExecutionError(ForgelineError):
    hint = "The tool call failed or timed out; the error text is injected as the tool result."


class RolloutError(ForgelineError):
    hint = "Generation failed for one or more prompts; check prompt lengths and the model."


# ── checkpoints ─────────────────────────────────────────────────────────────

class CheckpointError(ForgelineError):
    hint = "Inspect the checkpoint directory and its manifest.json."


class CheckpointNotFoundError(CheckpointError):
    hint = "No checkpoint exists at that path; train first or pass --resume with a valid path."


class CheckpointCorruptError(CheckpointError):
    hint = "The checkpoint is incomplete or unreadable; use an earlier one or delete it."


class CheckpointMismatchError(CheckpointError):
    hint = "The checkpoint was written for a different model spec / algorithm."


# ── distributed ─────────────────────────────────────────────────────────────

class DistributedConfigError(ForgelineError):
    hint = "Check world size, parallel degrees and the launcher (torchrun) environment."


# ── evaluation / deployment ─────────────────────────────────────────────────

class EvaluationError(ForgelineError):
    hint = "An evaluator failed; see the message for the suite name."


class PromotionError(ForgelineError):
    hint = "Promotion failed closed; see the gate report for the failing dimension."


class RegistryError(ForgelineError):
    hint = "The candidate registry is inconsistent or the candidate does not exist."


class ServingError(ForgelineError):
    hint = "The serving backend failed; check the model and request schema."


class BackendUnavailableError(ServingError):
    hint = "The requested backend is not loaded or is disabled by a feature flag."


class OptionalDependencyError(ForgelineError):
    hint = "Install the optional extra named in the message (e.g. pip install 'forgeline[huggingface]')."


# ── orchestration ───────────────────────────────────────────────────────────

class OrchestrationError(ForgelineError):
    hint = "A worker pool could not complete the request; check worker resources, retries and the Ray logs."
