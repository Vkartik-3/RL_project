"""Record schemas and JSONL IO with validation and malformed-record handling.

Every dataset category has a small dataclass schema. ``read_jsonl`` validates
records against a schema and either raises :class:`MalformedRecordError` or
(with ``on_error="skip"``) drops the record and reports it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Type, TypeVar

from forgeline.core.errors import DatasetError, MalformedRecordError

T = TypeVar("T")


@dataclass
class SFTRecord:
    prompt: str
    response: str
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SFTRecord":
        prompt = d.get("prompt")
        response = d.get("response", d.get("completion", d.get("target")))
        if not isinstance(prompt, str) or not prompt:
            raise MalformedRecordError("SFT record needs a non-empty string 'prompt'")
        if not isinstance(response, str) or not response:
            raise MalformedRecordError("SFT record needs a non-empty string 'response'")
        meta = {k: v for k, v in d.items() if k not in ("prompt", "response", "completion", "target")}
        return cls(prompt=prompt, response=response, meta=meta)

    def to_dict(self) -> Dict[str, Any]:
        return {"prompt": self.prompt, "response": self.response, **self.meta}


@dataclass
class PreferenceRecord:
    prompt: str
    chosen: str
    rejected: str
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PreferenceRecord":
        for key in ("prompt", "chosen", "rejected"):
            if not isinstance(d.get(key), str):
                raise MalformedRecordError(f"preference record needs string field {key!r}")
        if d["chosen"] == d["rejected"]:
            raise MalformedRecordError("preference record has identical chosen and rejected responses")
        meta = {k: v for k, v in d.items() if k not in ("prompt", "chosen", "rejected")}
        return cls(prompt=d["prompt"], chosen=d["chosen"], rejected=d["rejected"], meta=meta)

    def to_dict(self) -> Dict[str, Any]:
        return {"prompt": self.prompt, "chosen": self.chosen, "rejected": self.rejected, **self.meta}


@dataclass
class VerifiableTask:
    """A prompt with a verifiable target (answer string, unit tests, or format spec)."""

    prompt: str
    answer: Optional[str] = None
    test_code: Optional[str] = None
    task_type: str = "math"  # math | code | format | tagged
    tools: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VerifiableTask":
        prompt = d.get("prompt", d.get("problem", d.get("question")))
        if not isinstance(prompt, str) or not prompt:
            raise MalformedRecordError("verifiable task needs a non-empty 'prompt' (or 'problem')")
        answer = d.get("answer")
        test_code = d.get("test_code", d.get("test"))
        if answer is None and test_code is None:
            raise MalformedRecordError("verifiable task needs an 'answer' or 'test_code'")
        if answer is not None and not isinstance(answer, str):
            answer = str(answer)
        tools = d.get("available_tools", d.get("tools", [])) or []
        if not isinstance(tools, list):
            raise MalformedRecordError("'tools' must be a list")
        task_type = d.get("task_type", "code" if test_code and answer is None else "math")
        meta = {k: v for k, v in d.items()
                if k not in ("prompt", "problem", "question", "answer", "test_code", "test", "available_tools", "tools", "task_type")}
        return cls(prompt=prompt, answer=answer, test_code=test_code, task_type=task_type, tools=list(tools), meta=meta)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"prompt": self.prompt, "task_type": self.task_type, "tools": list(self.tools)}
        if self.answer is not None:
            out["answer"] = self.answer
        if self.test_code is not None:
            out["test_code"] = self.test_code
        out.update(self.meta)
        return out


@dataclass
class ProcessRecord:
    """A response with per-step labels for process-reward supervision."""

    prompt: str
    steps: List[str]
    step_labels: List[float]
    answer: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProcessRecord":
        steps = d.get("steps")
        labels = d.get("step_labels")
        if not isinstance(d.get("prompt"), str) or not isinstance(steps, list) or not isinstance(labels, list):
            raise MalformedRecordError("process record needs 'prompt', list 'steps' and list 'step_labels'")
        if len(steps) != len(labels):
            raise MalformedRecordError("'steps' and 'step_labels' must have the same length")
        for lab in labels:
            if not isinstance(lab, (int, float)) or not math.isfinite(float(lab)):
                raise MalformedRecordError("step labels must be finite numbers")
        meta = {k: v for k, v in d.items() if k not in ("prompt", "steps", "step_labels", "answer")}
        return cls(prompt=d["prompt"], steps=[str(s) for s in steps], step_labels=[float(x) for x in labels],
                   answer=d.get("answer"), meta=meta)


@dataclass
class RewardRecord:
    """A scored text, used to train scalar reward models on pseudo-labels."""

    text: str
    score: float
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RewardRecord":
        if not isinstance(d.get("text"), str) or not isinstance(d.get("score"), (int, float)):
            raise MalformedRecordError("reward record needs string 'text' and numeric 'score'")
        if not math.isfinite(float(d["score"])):
            raise MalformedRecordError("reward record score must be finite")
        meta = {k: v for k, v in d.items() if k not in ("text", "score")}
        return cls(text=d["text"], score=float(d["score"]), meta=meta)


SCHEMAS: Dict[str, Type[Any]] = {
    "supervised": SFTRecord,
    "preference": PreferenceRecord,
    "verifiable": VerifiableTask,
    "process": ProcessRecord,
    "reward": RewardRecord,
}


@dataclass
class ReadReport:
    read: int = 0
    skipped: int = 0
    errors: List[Tuple[int, str]] = field(default_factory=list)


def iter_jsonl(path: str | Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MalformedRecordError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from None
            if not isinstance(obj, dict):
                raise MalformedRecordError(f"{path}:{lineno}: record must be a JSON object")
            yield lineno, obj


def read_jsonl(path: str | Path, schema: Type[T] | Callable[[Dict[str, Any]], T], *,
               on_error: str = "raise", report: Optional[ReadReport] = None) -> List[T]:
    """Read and validate JSONL records. ``on_error`` is ``raise`` or ``skip``."""
    if on_error not in ("raise", "skip"):
        raise DatasetError("on_error must be 'raise' or 'skip'")
    parse = schema.from_dict if hasattr(schema, "from_dict") else schema  # type: ignore[union-attr]
    out: List[T] = []
    rep = report if report is not None else ReadReport()
    for lineno, obj in iter_jsonl(path):
        try:
            out.append(parse(obj))
            rep.read += 1
        except MalformedRecordError as exc:
            if on_error == "raise":
                raise MalformedRecordError(f"{path}:{lineno}: {exc.message}") from None
            rep.skipped += 1
            rep.errors.append((lineno, exc.message))
    return out


def write_jsonl(path: str | Path, records: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            d = r.to_dict() if hasattr(r, "to_dict") else r
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def deterministic_split(items: List[T], val_fraction: float, seed: int) -> Tuple[List[T], List[T]]:
    """Seeded shuffle then split; identical output for identical inputs and seed."""
    import random

    if not 0.0 <= val_fraction < 1.0:
        raise DatasetError("val_fraction must be in [0, 1)")
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    n_val = int(round(len(items) * val_fraction))
    val_idx = set(idx[:n_val])
    train = [items[i] for i in range(len(items)) if i not in val_idx]
    val = [items[i] for i in idx[:n_val]]
    return train, val
