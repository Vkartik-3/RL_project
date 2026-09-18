"""Lifecycle event names emitted through the metrics sink and logger."""

from __future__ import annotations

from enum import Enum


class Event(str, Enum):
    RUN_STARTED = "run.started"
    RUN_FINISHED = "run.finished"
    RUN_FAILED = "run.failed"
    CHECKPOINT_SAVED = "checkpoint.saved"
    CHECKPOINT_LOADED = "checkpoint.loaded"
    CHECKPOINT_FAILED = "checkpoint.failed"
    EVAL_STARTED = "evaluation.started"
    EVAL_FINISHED = "evaluation.finished"
    ROLLOUT_BATCH = "rollout.batch"
    ROLLOUT_FAILED = "rollout.failed"
    CANDIDATE_REGISTERED = "candidate.registered"
    CANDIDATE_PROMOTED = "candidate.promoted"
    CANDIDATE_REJECTED = "candidate.rejected"
    CANDIDATE_ROLLED_BACK = "candidate.rolled_back"
    SERVING_REQUEST = "serving.request"
    SERVING_ERROR = "serving.error"
    OPE_EVALUATED = "ope.evaluated"
    EXPERIMENT_AB = "experiment.ab"
    EXPERIMENT_SHADOW = "experiment.shadow"
