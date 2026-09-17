"""Request objects and per-request state for the inference engine."""

from __future__ import annotations

import queue
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from forgeline.core.errors import ServingError


class RequestStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    stop_token_ids: List[int] = field(default_factory=list)

    def validate(self) -> None:
        if self.max_tokens < 1:
            raise ServingError("max_tokens must be >= 1")
        if self.temperature < 0:
            raise ServingError("temperature must be >= 0")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ServingError("top_p must be in (0, 1]")
        if self.top_k is not None and self.top_k < 0:
            raise ServingError("top_k must be >= 0")


@dataclass
class GenerationRequest:
    prompt_tokens: List[int]
    params: SamplingParams = field(default_factory=SamplingParams)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    prompt: str = ""
    stream: bool = False
    created_at: float = field(default_factory=time.time)
    # state
    status: RequestStatus = RequestStatus.PENDING
    generated_tokens: List[int] = field(default_factory=list)
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    token_queue: "queue.Queue[Optional[int]]" = field(default_factory=queue.Queue)
    meta: dict = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return self.status in (RequestStatus.FINISHED, RequestStatus.FAILED, RequestStatus.CANCELLED)

    @property
    def latency_s(self) -> Optional[float]:
        if self.finished_at is None:
            return None
        return self.finished_at - self.created_at

    def push_token(self, token: int) -> None:
        self.generated_tokens.append(token)
        self.token_queue.put(token)

    def finish(self, reason: str, error: Optional[str] = None) -> None:
        if error:
            self.status = RequestStatus.FAILED
        elif reason == "cancelled":
            self.status = RequestStatus.CANCELLED
        else:
            self.status = RequestStatus.FINISHED
        self.finish_reason = reason
        self.error = error
        self.finished_at = time.time()
        self.token_queue.put(None)
