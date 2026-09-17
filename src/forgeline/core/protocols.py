"""Structural protocols that decouple subsystems.

Concrete implementations live in their own packages; these protocols are what
the training, rollout, evaluation and serving layers program against.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, runtime_checkable

import torch


# ── generation / policies ───────────────────────────────────────────────────

@dataclass
class GenerationSettings:
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    use_cache: bool = True
    stop_token_ids: Sequence[int] = ()

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


@runtime_checkable
class PolicyModel(Protocol):
    """A language model usable as an RL policy.

    ``logprobs`` must return the per-token log-probability of ``response_ids``
    conditioned on ``prompt_ids`` (shape ``[B, R]``), with gradients attached
    when the underlying parameters require them.
    """

    @property
    def device(self) -> torch.device: ...

    @property
    def vocab_size(self) -> int: ...

    def generate(self, prompt_ids: torch.Tensor, settings: GenerationSettings) -> torch.Tensor:
        """Return response token ids of shape ``[B, R]`` (prompt excluded)."""
        ...

    def logprobs(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor: ...

    def reference(self) -> AbstractContextManager[Any]:
        """Context in which the policy behaves as its frozen reference."""
        ...

    def trainable_parameters(self) -> List[torch.nn.Parameter]: ...

    def train(self, mode: bool = True) -> Any: ...

    def eval(self) -> Any: ...

    def state_for_checkpoint(self) -> Dict[str, Any]: ...

    def load_checkpoint_state(self, state: Mapping[str, Any]) -> None: ...


# ── rewards / verifiers ─────────────────────────────────────────────────────

@dataclass
class RewardResult:
    """Scalar reward for one trajectory plus diagnostics."""

    value: float
    components: Dict[str, float] = field(default_factory=dict)
    passed: Optional[bool] = None
    info: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RewardProvider(Protocol):
    name: str

    def score(self, trajectory: "Trajectory") -> RewardResult: ...


@runtime_checkable
class Verifier(Protocol):
    """Deterministic check of a completion against a target."""

    name: str

    def verify(self, output: str, target: Any = None) -> RewardResult: ...


# ── trajectories ────────────────────────────────────────────────────────────

@dataclass
class Trajectory:
    """One sampled completion (single-turn or multi-turn agent episode)."""

    prompt_ids: torch.Tensor  # [P]
    response_ids: torch.Tensor  # [R] (all model-generated tokens, tool results excluded)
    prompt_text: str = ""
    response_text: str = ""
    task: Dict[str, Any] = field(default_factory=dict)
    # multi-turn agent data (optional)
    segments: List[str] = field(default_factory=list)
    tool_results: List[str] = field(default_factory=list)
    truncated: bool = False
    old_logprobs: Optional[torch.Tensor] = None  # [R] sampled-time token log-probs
    reward: Optional[RewardResult] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def response_length(self) -> int:
        return int(self.response_ids.numel())


# ── data ────────────────────────────────────────────────────────────────────

@runtime_checkable
class DatasetProvider(Protocol):
    kind: str

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterable[Any]: ...


# ── metrics / events ────────────────────────────────────────────────────────

@runtime_checkable
class MetricsSink(Protocol):
    def log(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None: ...

    def event(self, name: str, payload: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...


# ── distributed ─────────────────────────────────────────────────────────────

@runtime_checkable
class DistributedStrategy(Protocol):
    name: str

    def setup(self) -> None: ...

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module: ...

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module: ...

    def is_main_process(self) -> bool: ...

    def world_size(self) -> int: ...

    def rank(self) -> int: ...

    def barrier(self) -> None: ...

    def teardown(self) -> None: ...


# ── evaluation ──────────────────────────────────────────────────────────────

@runtime_checkable
class EvaluationSuite(Protocol):
    name: str

    def run(self, target: Any, **kwargs: Any) -> "EvaluationResult": ...


@dataclass
class EvaluationResult:
    suite: str
    metrics: Dict[str, float]
    n_samples: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
    per_item: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite": self.suite,
            "metrics": dict(self.metrics),
            "n_samples": self.n_samples,
            "details": dict(self.details),
            "per_item": list(self.per_item),
        }
