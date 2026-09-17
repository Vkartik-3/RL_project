"""Corpus preparation: text → tokens → ``train.bin`` / ``val.bin`` + ``meta.pkl``."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from forgeline.core.errors import DatasetError
from forgeline.data.tokenizers import (
    CharTokenizer,
    HFTokenizer,
    TiktokenTokenizer,
    Tokenizer,
    save_tokenizer_metadata,
    token_dtype_for,
)


def prepare_text_corpus(input_file: str | Path, output_dir: str | Path, tokenizer: str = "char",
                        val_fraction: float = 0.1) -> Dict[str, object]:
    """Tokenize a single text file into ``train.bin``/``val.bin`` with a tokenizer fitted or loaded."""
    input_file = Path(input_file)
    if not input_file.exists():
        raise DatasetError(f"Input text file not found: {input_file}")
    text = input_file.read_text(encoding="utf-8")
    if tokenizer == "char":
        tok: Tokenizer = CharTokenizer().fit(text)
    elif tokenizer.startswith("tiktoken"):
        _, _, enc = tokenizer.partition(":")
        tok = TiktokenTokenizer(enc or "cl100k_base")
    elif tokenizer.startswith("hf:"):
        tok = HFTokenizer(tokenizer[3:])
    else:
        raise DatasetError(f"Unknown tokenizer {tokenizer!r}; use char|tiktoken[:enc]|hf:<name>")
    ids = tok.encode(text)
    if len(ids) < 10:
        raise DatasetError("corpus is too small after tokenization")
    dtype = token_dtype_for(tok.vocab_size)
    split_idx = int(len(ids) * (1 - val_fraction))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.array(ids[:split_idx], dtype=dtype).tofile(output_dir / "train.bin")
    np.array(ids[split_idx:], dtype=dtype).tofile(output_dir / "val.bin")
    save_tokenizer_metadata(tok, output_dir)
    return {"tokens": len(ids), "train_tokens": split_idx, "val_tokens": len(ids) - split_idx,
            "vocab_size": tok.vocab_size, "tokenizer": tok.kind, "output_dir": str(output_dir)}


def prepare_hf_corpus(dataset_name: str, output_dir: str | Path, tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
                      subset: Optional[str] = None, max_tokens: int = 100_000_000, shard_size: int = 10_000_000,
                      text_field: str = "text", split: str = "train") -> Dict[str, object]:
    """Stream a HuggingFace dataset into shard files (optional ``huggingface`` extra)."""
    from forgeline.core.errors import OptionalDependencyError
    from forgeline.data.streaming import write_shards_from_documents

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise OptionalDependencyError("datasets is not installed", hint="pip install 'forgeline[huggingface]'") from exc
    tok = HFTokenizer(tokenizer_name)
    kwargs = {"split": split, "streaming": True}
    ds = load_dataset(dataset_name, subset, **kwargs) if subset else load_dataset(dataset_name, **kwargs)
    result = write_shards_from_documents((row.get(text_field, "") for row in ds), tok, output_dir,
                                         max_tokens=max_tokens, shard_size=shard_size, dtype=np.uint32)
    save_tokenizer_metadata(tok, output_dir)
    result.update({"dataset": dataset_name, "subset": subset, "tokenizer": tokenizer_name, "vocab_size": tok.vocab_size})
    return result
