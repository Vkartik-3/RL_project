"""Instruction / SFT datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

from forgeline.data.records import ReadReport, SFTRecord, read_jsonl
from forgeline.data.tokenizers import Tokenizer


class SupervisedDataset:
    """Tokenized ``(prompt_ids, response_ids)`` pairs for SFT."""

    kind = "supervised"

    def __init__(self, records: Sequence[SFTRecord], tokenizer: Tokenizer, max_length: int = 512,
                 response_suffix: str = ""):
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.response_suffix = response_suffix
        self._examples: List[Tuple[List[int], List[int]]] = []
        for r in self.records:
            p = tokenizer.encode(r.prompt)
            t = tokenizer.encode(r.response + response_suffix)
            budget = max_length - len(p)
            if budget < 1:
                p, t = p[: max_length - 1], t[:1]
            else:
                t = t[:budget]
            if len(p) + len(t) >= 2:
                self._examples.append((p, t))

    @classmethod
    def from_jsonl(cls, path: str | Path, tokenizer: Tokenizer, max_length: int = 512,
                   on_error: str = "raise", report: Optional[ReadReport] = None) -> "SupervisedDataset":
        return cls(read_jsonl(path, SFTRecord, on_error=on_error, report=report), tokenizer, max_length)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, i: int) -> Tuple[List[int], List[int]]:
        return self._examples[i]

    def __iter__(self) -> Iterator[Tuple[List[int], List[int]]]:
        return iter(self._examples)
