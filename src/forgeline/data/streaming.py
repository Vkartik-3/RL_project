"""Streaming corpora: sharded token files, raw text globs, HuggingFace datasets,
plus a constant-memory shard writer.
"""

from __future__ import annotations

import glob
import math
import os
import random
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from forgeline.core.errors import DatasetError, OptionalDependencyError
from forgeline.data.tokenizers import Tokenizer
from forgeline.observability.logging import get_logger

log = get_logger("forgeline.data.streaming")


class ShardedTokenDataset(IterableDataset):
    """Streams ``(x, y)`` windows from ``shard_*.bin`` files, sharded by rank and worker."""

    kind = "pretraining"

    def __init__(self, shard_dir: str | Path, block_size: int, shuffle: bool = True,
                 world_size: int = 1, rank: int = 0, dtype=np.uint16, seed: int = 0,
                 pattern: str = "shard_*.bin"):
        super().__init__()
        self.block_size = block_size
        self.shuffle = shuffle
        self.dtype = dtype
        self.seed = seed
        paths = sorted(glob.glob(os.path.join(str(shard_dir), pattern)))
        if not paths:
            raise DatasetError(f"No shard files matching {pattern!r} in {shard_dir}")
        self.shard_paths = paths[rank::world_size]
        self.total_tokens = sum(os.path.getsize(p) // np.dtype(dtype).itemsize for p in self.shard_paths)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker = torch.utils.data.get_worker_info()
        shards = list(self.shard_paths)
        if worker is not None:
            shards = shards[worker.id :: worker.num_workers]
        rng = random.Random(self.seed + (worker.id if worker else 0))
        if self.shuffle:
            rng.shuffle(shards)
        for path in shards:
            data = np.memmap(path, dtype=self.dtype, mode="r")
            n = len(data)
            if n <= self.block_size:
                continue
            n_samples = n // self.block_size
            starts = list(range(0, n - self.block_size, self.block_size)) if not self.shuffle else \
                [rng.randrange(0, n - self.block_size) for _ in range(n_samples)]
            for s in starts[:n_samples]:
                x = torch.from_numpy(data[s : s + self.block_size].astype(np.int64))
                y = torch.from_numpy(data[s + 1 : s + 1 + self.block_size].astype(np.int64))
                yield x, y


class TextGlobDataset(IterableDataset):
    """On-the-fly tokenization of raw text files matching a glob."""

    kind = "pretraining"

    def __init__(self, pattern: str, tokenizer: Tokenizer, block_size: int, shuffle: bool = True, seed: int = 0):
        super().__init__()
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise DatasetError(f"No files matching {pattern!r}")
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        files = list(self.files)
        if self.shuffle:
            random.Random(self.seed).shuffle(files)
        buffer: List[int] = []
        for fpath in files:
            with open(fpath, "r", errors="ignore") as f:
                for line in f:
                    buffer.extend(self.tokenizer.encode(line))
                    while len(buffer) >= self.block_size + 1:
                        chunk = buffer[: self.block_size + 1]
                        buffer = buffer[self.block_size :]
                        yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


class HFStreamingDataset(IterableDataset):
    """Stream a HuggingFace dataset and pack tokens into fixed windows (optional extra)."""

    kind = "pretraining"

    def __init__(self, dataset_name: str, tokenizer: Tokenizer, block_size: int, subset: Optional[str] = None,
                 split: str = "train", text_field: str = "text", max_tokens: Optional[int] = None,
                 rank: int = 0, world_size: int = 1, seed: int = 42, min_chars: int = 50):
        super().__init__()
        self.dataset_name, self.subset, self.split = dataset_name, subset, split
        self.tokenizer, self.block_size, self.text_field = tokenizer, block_size, text_field
        self.max_tokens, self.rank, self.world_size, self.seed, self.min_chars = max_tokens, rank, world_size, seed, min_chars

    def _dataset(self):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise OptionalDependencyError("datasets is not installed", hint="pip install 'forgeline[huggingface]'") from exc
        kwargs = {"streaming": True, "split": self.split}
        ds = load_dataset(self.dataset_name, self.subset, **kwargs) if self.subset else load_dataset(self.dataset_name, **kwargs)
        if self.world_size > 1:
            ds = ds.shard(num_shards=self.world_size, index=self.rank)
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            ds = ds.shard(num_shards=worker.num_workers, index=worker.id)
        return ds.shuffle(seed=self.seed, buffer_size=10_000)

    def __iter__(self):
        yield from pack_documents(
            (ex.get(self.text_field, "") for ex in self._dataset()), self.tokenizer, self.block_size,
            max_tokens=self.max_tokens, min_chars=self.min_chars,
        )


def pack_documents(texts: Iterable[str], tokenizer: Tokenizer, block_size: int, *,
                   max_tokens: Optional[int] = None, min_chars: int = 0, drop_token_zero: bool = False):
    """Concatenate tokenized documents and yield ``(x, y)`` windows with constant memory."""
    buffer: List[int] = []
    total = 0
    for text in texts:
        if not text or len(text.strip()) < min_chars:
            continue
        ids = tokenizer.encode(text)
        if drop_token_zero:
            ids = [t for t in ids if t != 0]
        if not ids:
            continue
        buffer.extend(ids)
        total += len(ids)
        while len(buffer) >= block_size + 1:
            chunk = buffer[: block_size + 1]
            buffer = buffer[block_size:]
            yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)
        if max_tokens is not None and total >= max_tokens:
            break


def shard_token_file(input_path: str | Path, output_dir: str | Path, n_shards: int, dtype=np.uint16) -> List[Path]:
    """Split one ``.bin`` file into ``n_shards`` files ``shard_XXXX.bin``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.memmap(str(input_path), dtype=dtype, mode="r")
    per_shard = math.ceil(len(data) / n_shards)
    out: List[Path] = []
    for i in range(n_shards):
        start, end = i * per_shard, min((i + 1) * per_shard, len(data))
        if start >= end:
            break
        p = output_dir / f"shard_{i:04d}.bin"
        np.array(data[start:end], dtype=dtype).tofile(p)
        out.append(p)
    return out


def write_shards_from_documents(
    texts: Iterable[str], tokenizer: Tokenizer, output_dir: str | Path, *, max_tokens: int,
    shard_size: int = 10_000_000, dtype=np.uint32, min_chars: int = 50, drop_token_zero: bool = True,
    first_shard_is_val: bool = True, log_every: int = 1,
) -> dict:
    """Tokenize a document stream into fixed-size shard files using a pre-allocated buffer.

    Peak memory is ``shard_size * itemsize`` regardless of corpus size. The
    first shard becomes ``val_000000.bin``; the rest ``train_XXXXXX.bin``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    buf = np.empty((shard_size,), dtype=dtype)
    count, shard_index, total, n_docs = 0, 0, 0, 0
    t0 = time.time()
    written: List[Path] = []

    def flush(n: int) -> None:
        nonlocal shard_index, total
        split = "val" if (first_shard_is_val and shard_index == 0) else "train"
        p = output_dir / f"{split}_{shard_index:06d}.bin"
        buf[:n].tofile(p)
        written.append(p)
        total += n
        if log_every and shard_index % log_every == 0:
            el = time.time() - t0
            log.info("shard_written", index=shard_index, split=split, tokens=total, docs=n_docs,
                     tok_per_s=round(total / el) if el > 0 else 0)
        shard_index += 1

    for text in texts:
        if not text or len(text.strip()) < min_chars:
            continue
        ids = tokenizer.encode(text)
        if drop_token_zero:
            ids = [t for t in ids if t != 0]
        if not ids:
            continue
        tokens = np.asarray(ids, dtype=dtype)
        n_docs += 1
        if count + len(tokens) < shard_size:
            buf[count : count + len(tokens)] = tokens
            count += len(tokens)
        else:
            remainder = shard_size - count
            buf[count:shard_size] = tokens[:remainder]
            flush(shard_size)
            leftover = len(tokens) - remainder
            buf[:leftover] = tokens[remainder:]
            count = leftover
        if total + count >= max_tokens:
            break
    if count > 0:
        flush(count)
    merge_shards(output_dir, dtype=dtype)
    return {"total_tokens": total, "num_shards": shard_index, "docs": n_docs, "files": [str(p) for p in written]}


def merge_shards(output_dir: str | Path, dtype=np.uint32, chunk: int = 5_000_000) -> None:
    """Concatenate ``train_*.bin``/``val_*.bin`` into ``train.bin``/``val.bin`` with chunked IO."""
    output_dir = Path(output_dir)
    for split in ("train", "val"):
        files = sorted(output_dir.glob(f"{split}_*.bin"))
        if not files:
            continue
        with (output_dir / f"{split}.bin").open("wb") as out:
            for fp in files:
                data = np.fromfile(fp, dtype=dtype)
                for i in range(0, len(data), chunk):
                    data[i : i + chunk].tofile(out)


def build_streaming_loader(dataset: IterableDataset, batch_size: int, num_workers: int = 0,
                           pin_memory: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      pin_memory=pin_memory and torch.cuda.is_available(), drop_last=True)
