"""Preference-pair datasets for DPO and reward-model training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

from forgeline.data.records import PreferenceRecord, ReadReport, read_jsonl
from forgeline.data.tokenizers import Tokenizer


class PreferenceDataset:
    """Tokenized ``{prompt, chosen, rejected}`` examples."""

    kind = "preference"

    def __init__(self, records: Sequence[PreferenceRecord], tokenizer: Tokenizer, max_prompt_length: int = 384,
                 max_response_length: int = 128):
        self.records = list(records)
        self.tokenizer = tokenizer
        self._examples: List[Dict[str, List[int]]] = []
        for r in self.records:
            p = tokenizer.encode(r.prompt)[-max_prompt_length:]
            c = tokenizer.encode(r.chosen)[:max_response_length]
            j = tokenizer.encode(r.rejected)[:max_response_length]
            if not c or not j or not p:
                continue
            self._examples.append({"prompt": p, "chosen": c, "rejected": j})

    @classmethod
    def from_jsonl(cls, path: str | Path, tokenizer: Tokenizer, max_prompt_length: int = 384,
                   max_response_length: int = 128, on_error: str = "raise",
                   report: Optional[ReadReport] = None) -> "PreferenceDataset":
        return cls(read_jsonl(path, PreferenceRecord, on_error=on_error, report=report), tokenizer,
                   max_prompt_length, max_response_length)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, i: int) -> Dict[str, List[int]]:
        return self._examples[i]

    def __iter__(self) -> Iterator[Dict[str, List[int]]]:
        return iter(self._examples)
