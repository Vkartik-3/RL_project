"""Serving backends: wrap an inference engine + tokenizer behind ``complete``/``stream``."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Protocol

from forgeline.core.errors import BackendUnavailableError, ServingError
from forgeline.data.tokenizers import Tokenizer
from forgeline.inference.engine import InferenceEngine
from forgeline.inference.requests import GenerationRequest, SamplingParams


@dataclass
class CompletionResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    backend: str
    request_id: str


class Backend(Protocol):
    name: str

    def complete(self, prompt: str, params: SamplingParams, request_id: Optional[str] = None) -> CompletionResult: ...

    def stream(self, prompt: str, params: SamplingParams, request_id: Optional[str] = None) -> Iterator[str]: ...

    def healthy(self) -> bool: ...


class EngineBackend:
    """A model served by :class:`InferenceEngine` with a tokenizer for text IO."""

    def __init__(self, name: str, engine: InferenceEngine, tokenizer: Tokenizer, background: bool = True):
        self.name = name
        self.engine = engine
        self.tokenizer = tokenizer
        self._background = background
        self._enabled = True
        if background:
            engine.start()

    def healthy(self) -> bool:
        return self._enabled

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def _submit(self, prompt: str, params: SamplingParams, request_id: Optional[str]) -> GenerationRequest:
        if not self._enabled:
            raise BackendUnavailableError(f"backend {self.name!r} is disabled")
        ids = self.tokenizer.encode(prompt)
        if not ids:
            raise ServingError("prompt produced no tokens under this tokenizer")
        kwargs: Dict[str, Any] = {"prompt": prompt}
        if request_id:
            kwargs["id"] = request_id
        return self.engine.submit(ids, params, **kwargs)

    def _drive(self, req: GenerationRequest) -> None:
        if not self._background:
            while not req.finished:
                self.engine.step()

    def complete(self, prompt: str, params: SamplingParams, request_id: Optional[str] = None) -> CompletionResult:
        req = self._submit(prompt, params, request_id)
        self._drive(req)
        while not req.finished:
            threading.Event().wait(0.005)
        if req.error:
            raise ServingError(f"generation failed: {req.error}")
        return CompletionResult(text=self.tokenizer.decode(req.generated_tokens), prompt_tokens=len(req.prompt_tokens),
                                completion_tokens=len(req.generated_tokens), finish_reason=req.finish_reason or "length",
                                backend=self.name, request_id=req.id)

    def stream(self, prompt: str, params: SamplingParams, request_id: Optional[str] = None) -> Iterator[str]:
        req = self._submit(prompt, params, request_id)
        pending: List[int] = []
        while True:
            if not self._background and not req.finished:
                self.engine.step()
            token = req.token_queue.get()
            if token is None:
                break
            pending.append(token)
            text = self.tokenizer.decode(pending)
            if text and "�" not in text:  # emit only once the byte sequence decodes cleanly
                yield text
                pending = []
        if req.error:
            raise ServingError(f"generation failed: {req.error}")

    def close(self) -> None:
        self.engine.stop()


class EchoBackend:
    """Deterministic backend for tests: echoes the prompt reversed, truncated to ``max_tokens`` chars."""

    name = "echo"

    def __init__(self, name: str = "echo"):
        self.name = name
        self._enabled = True

    def healthy(self) -> bool:
        return self._enabled

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def complete(self, prompt: str, params: SamplingParams, request_id: Optional[str] = None) -> CompletionResult:
        if not self._enabled:
            raise BackendUnavailableError(f"backend {self.name!r} is disabled")
        text = prompt[::-1][: params.max_tokens]
        return CompletionResult(text=text, prompt_tokens=len(prompt), completion_tokens=len(text), finish_reason="length",
                                backend=self.name, request_id=request_id or "echo")

    def stream(self, prompt: str, params: SamplingParams, request_id: Optional[str] = None) -> Iterator[str]:
        for ch in self.complete(prompt, params, request_id).text:
            yield ch
