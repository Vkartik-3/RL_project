"""Pre-tokenized memory-mapped corpora for language-model pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import torch

from forgeline.core.errors import DatasetError
from forgeline.data.tokenizers import load_tokenizer_metadata


def detect_token_dtype(data_dir: str | Path) -> np.dtype:
    """uint32 for large vocabularies / tiktoken, uint16 otherwise (from ``meta.pkl``)."""
    meta_path = Path(data_dir) / "meta.pkl"
    if not meta_path.exists():
        return np.dtype(np.uint16)
    meta = load_tokenizer_metadata(data_dir)
    if int(meta.get("vocab_size", 0)) > 65535 or meta.get("tokenizer_type") == "tiktoken":
        return np.dtype(np.uint32)
    return np.dtype(np.uint16)


class MemmapCorpus:
    """Random ``(x, y)`` windows from ``<split>.bin`` (next-token targets)."""

    kind = "pretraining"

    def __init__(self, data_dir: str | Path, split: str, block_size: int, dtype: Optional[np.dtype] = None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.block_size = block_size
        path = self.data_dir / f"{split}.bin"
        if not path.exists():
            raise DatasetError(f"Token file not found: {path}", hint="Run `forgeline data prepare` to create train.bin / val.bin.")
        self.dtype = dtype or detect_token_dtype(self.data_dir)
        self.data = np.memmap(path, dtype=self.dtype, mode="r")
        if len(self.data) <= block_size + 1:
            raise DatasetError(f"{path} has {len(self.data)} tokens; need more than block_size+1={block_size + 1}")

    def __len__(self) -> int:
        return len(self.data) - self.block_size - 1

    @property
    def n_tokens(self) -> int:
        return int(len(self.data))

    def sample_batch(self, batch_size: int, device: torch.device | str = "cpu",
                     generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + self.block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for i in range(0, len(self.data) - self.block_size - 1, self.block_size):
            x = torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64))
            y = torch.from_numpy(self.data[i + 1 : i + 1 + self.block_size].astype(np.int64))
            yield x, y


def write_token_file(tokens: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(path)
    return path
