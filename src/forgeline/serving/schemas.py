"""Request/response schemas for the OpenAI-compatible API (validated dataclasses)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from forgeline.core.errors import ServingError
from forgeline.inference.requests import SamplingParams


@dataclass
class CompletionRequest:
    prompt: str
    model: Optional[str] = None
    max_tokens: int = 64
    temperature: float = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = None
    stream: bool = False
    stop: List[str] = field(default_factory=list)
    user: Optional[str] = None
    request_id: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CompletionRequest":
        if not isinstance(d, dict):
            raise ServingError("request body must be a JSON object")
        prompt = d.get("prompt")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        if not isinstance(prompt, str) or not prompt:
            raise ServingError("'prompt' must be a non-empty string")
        req = cls(prompt=prompt, model=d.get("model"), max_tokens=int(d.get("max_tokens", 64)),
                  temperature=float(d.get("temperature", 0.8)), top_k=d.get("top_k", 50), top_p=d.get("top_p"),
                  stream=bool(d.get("stream", False)), stop=_stop_list(d.get("stop")), user=d.get("user"),
                  request_id=d.get("request_id"))
        req.sampling().validate()
        return req

    def sampling(self) -> SamplingParams:
        return SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p)


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatCompletionRequest:
    messages: List[ChatMessage]
    model: Optional[str] = None
    max_tokens: int = 64
    temperature: float = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = None
    stream: bool = False
    user: Optional[str] = None
    request_id: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChatCompletionRequest":
        if not isinstance(d, dict):
            raise ServingError("request body must be a JSON object")
        raw = d.get("messages")
        if not isinstance(raw, list) or not raw:
            raise ServingError("'messages' must be a non-empty list")
        msgs = []
        for m in raw:
            if not isinstance(m, dict) or not isinstance(m.get("content"), str):
                raise ServingError("each message needs a string 'content'")
            role = m.get("role", "user")
            if role not in ("system", "user", "assistant", "tool"):
                raise ServingError(f"invalid role {role!r}")
            msgs.append(ChatMessage(role=role, content=m["content"]))
        req = cls(messages=msgs, model=d.get("model"), max_tokens=int(d.get("max_tokens", 64)),
                  temperature=float(d.get("temperature", 0.8)), top_k=d.get("top_k", 50), top_p=d.get("top_p"),
                  stream=bool(d.get("stream", False)), user=d.get("user"), request_id=d.get("request_id"))
        req.sampling().validate()
        return req

    def to_prompt(self) -> str:
        return "".join(f"<|{m.role}|>\n{m.content}\n" for m in self.messages) + "<|assistant|>\n"

    def sampling(self) -> SamplingParams:
        return SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p)


def _stop_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list) and all(isinstance(s, str) for s in v):
        return v
    raise ServingError("'stop' must be a string or list of strings")


def completion_response(request_id: str, model: str, text: str, prompt_tokens: int, completion_tokens: int,
                        finish_reason: str, backend: Optional[str] = None) -> Dict[str, Any]:
    out = {
        "id": f"cmpl-{request_id}", "object": "text_completion", "created": int(time.time()), "model": model,
        "choices": [{"text": text, "index": 0, "finish_reason": finish_reason, "logprobs": None}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
    }
    if backend:
        out["forgeline"] = {"backend": backend}
    return out


def chat_response(request_id: str, model: str, text: str, prompt_tokens: int, completion_tokens: int, finish_reason: str,
                  backend: Optional[str] = None) -> Dict[str, Any]:
    out = {
        "id": f"chatcmpl-{request_id}", "object": "chat.completion", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
    }
    if backend:
        out["forgeline"] = {"backend": backend}
    return out


def completion_chunk(request_id: str, model: str, text: str, finish_reason: Optional[str] = None) -> Dict[str, Any]:
    return {"id": f"cmpl-{request_id}", "object": "text_completion", "created": int(time.time()), "model": model,
            "choices": [{"text": text, "index": 0, "finish_reason": finish_reason}]}


def chat_chunk(request_id: str, model: str, text: str, finish_reason: Optional[str] = None) -> Dict[str, Any]:
    return {"id": f"chatcmpl-{request_id}", "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish_reason}]}


def error_response(message: str, kind: str = "invalid_request_error", code: int = 400) -> Dict[str, Any]:
    return {"error": {"message": message, "type": kind, "code": code}}


def new_request_id() -> str:
    return uuid.uuid4().hex[:24]
