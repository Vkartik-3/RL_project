"""Verifiable tasks (math answers, unit tests, tagged answers) and prompt sets for RL."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

from forgeline.core.errors import DatasetError, OptionalDependencyError
from forgeline.data.records import ReadReport, VerifiableTask, read_jsonl, write_jsonl
from forgeline.data.tokenizers import Tokenizer

DEFAULT_ARITHMETIC_TASKS: List[VerifiableTask] = [
    VerifiableTask(prompt="What is 15 + 27?", answer="42"),
    VerifiableTask(prompt="What is 8 * 7?", answer="56"),
    VerifiableTask(prompt="What is 100 / 4?", answer="25"),
    VerifiableTask(prompt="What is 2^10?", answer="1024"),
    VerifiableTask(prompt="If x + 3 = 7, what is x?", answer="4"),
    VerifiableTask(prompt="What is the square root of 144?", answer="12"),
    VerifiableTask(prompt="What is 3! (3 factorial)?", answer="6"),
    VerifiableTask(prompt="What is 17 - 9?", answer="8"),
]

DEFAULT_PROMPTS: List[str] = [
    "Once upon a time",
    "The meaning of life is",
    "In a world where",
    "The key to success is",
]


class VerifiableTaskDataset:
    kind = "verifiable"

    def __init__(self, tasks: Sequence[VerifiableTask]):
        self.tasks = list(tasks)
        if not self.tasks:
            raise DatasetError("verifiable task dataset is empty")

    @classmethod
    def from_jsonl(cls, path: str | Path, on_error: str = "raise", report: Optional[ReadReport] = None,
                   limit: Optional[int] = None) -> "VerifiableTaskDataset":
        tasks = read_jsonl(path, VerifiableTask, on_error=on_error, report=report)
        return cls(tasks[:limit] if limit else tasks)

    @classmethod
    def builtin_arithmetic(cls) -> "VerifiableTaskDataset":
        return cls(DEFAULT_ARITHMETIC_TASKS)

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, i: int) -> VerifiableTask:
        return self.tasks[i]

    def __iter__(self) -> Iterator[VerifiableTask]:
        return iter(self.tasks)


def load_prompts(path: Optional[str | Path], tokenizer: Tokenizer, max_length: int = 128) -> List[List[int]]:
    """Prompts from JSONL (``prompt``/``text`` field) or plain text, tokenized and truncated."""
    texts: List[str] = []
    if path and Path(path).exists():
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    texts.append(obj.get("prompt", obj.get("text", line)) if isinstance(obj, dict) else line)
                except json.JSONDecodeError:
                    texts.append(line)
    else:
        texts = list(DEFAULT_PROMPTS)
    out = []
    for t in texts:
        ids = tokenizer.encode(t)[:max_length]
        if ids:
            out.append(ids)
    if not out:
        raise DatasetError("no usable prompts found")
    return out


_GSM8K_ANSWER = re.compile(r"####\s*(.+)")


def extract_gsm8k_answer(solution: str) -> str:
    """GSM8K stores the final answer after ``####``; commas/whitespace are stripped."""
    for line in reversed(solution.strip().split("\n")):
        m = _GSM8K_ANSWER.search(line)
        if m:
            return re.sub(r"[,\s]", "", m.group(1))
    return ""


def build_gsm8k_tool_dataset(output_path: str | Path, split: str = "train", limit: Optional[int] = None) -> int:
    """Download GSM8K (optional ``datasets`` extra) and write tool-use tasks as JSONL."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise OptionalDependencyError("datasets is not installed", hint="pip install 'forgeline[huggingface]'") from exc
    ds = load_dataset("openai/gsm8k", "main", split=split)
    tasks = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        ans = extract_gsm8k_answer(row["answer"])
        if not ans:
            continue
        tasks.append(VerifiableTask(prompt=row["question"], answer=ans, task_type="tagged", tools=["python_executor"]))
    return write_jsonl(output_path, tasks)
