"""Tokenizers: character-level (default, dependency-free), tiktoken BPE, HuggingFace.

Tokenizer metadata is persisted as ``meta.pkl`` next to the token files so
that training, evaluation and serving always resolve the same vocabulary.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from forgeline.core.errors import OptionalDependencyError, TokenizerError


@runtime_checkable
class Tokenizer(Protocol):
    kind: str
    vocab_size: int

    def encode(self, text: str) -> List[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...

    def metadata(self) -> Dict[str, Any]: ...


class CharTokenizer:
    """Maps each unique character to an id. ``unk`` handling: unknown chars are skipped on encode."""

    kind = "char"

    def __init__(self, chars: Optional[Sequence[str]] = None):
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.vocab_size = 0
        if chars is not None:
            self._set_vocab(list(chars))

    def _set_vocab(self, chars: List[str]) -> None:
        chars = sorted(set(chars))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def fit(self, text: str) -> "CharTokenizer":
        self._set_vocab(list(set(text)))
        return self

    @classmethod
    def ascii(cls) -> "CharTokenizer":
        """Printable ASCII vocabulary (95 symbols + newline + tab) for tiny local runs."""
        chars = [chr(i) for i in range(32, 127)] + ["\n", "\t"]
        return cls(chars)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.idx_to_char.get(int(i), "?") for i in ids)

    def metadata(self) -> Dict[str, Any]:
        return {
            "tokenizer_type": "char",
            "vocab_size": self.vocab_size,
            "char_to_idx": dict(self.char_to_idx),
            "idx_to_char": dict(self.idx_to_char),
        }

    @classmethod
    def from_metadata(cls, meta: Dict[str, Any]) -> "CharTokenizer":
        tok = cls()
        tok.vocab_size = int(meta["vocab_size"])
        tok.char_to_idx = {str(k): int(v) for k, v in meta["char_to_idx"].items()}
        tok.idx_to_char = {int(k): str(v) for k, v in meta["idx_to_char"].items()}
        return tok


class TiktokenTokenizer:
    kind = "tiktoken"

    def __init__(self, encoding_name: str = "cl100k_base"):
        try:
            import tiktoken
        except ImportError as exc:
            raise OptionalDependencyError("tiktoken is not installed", hint="pip install 'forgeline[training]'") from exc
        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, allowed_special=set())

    def decode(self, ids: Sequence[int]) -> str:
        return self.enc.decode(list(ids))

    def metadata(self) -> Dict[str, Any]:
        return {"tokenizer_type": "tiktoken", "encoding_name": self.encoding_name, "vocab_size": self.vocab_size}


class HFTokenizer:
    kind = "huggingface"

    def __init__(self, name: str, padding_side: str = "left"):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise OptionalDependencyError("transformers is not installed", hint="pip install 'forgeline[huggingface]'") from exc
        self.name = name
        self.hf = AutoTokenizer.from_pretrained(name, trust_remote_code=True, padding_side=padding_side)
        if self.hf.pad_token is None:
            self.hf.pad_token = self.hf.eos_token
        self.vocab_size = int(self.hf.vocab_size)

    def encode(self, text: str) -> List[int]:
        return self.hf.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self.hf.decode(list(ids), skip_special_tokens=True)

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.hf.eos_token_id

    def metadata(self) -> Dict[str, Any]:
        return {"tokenizer_type": "huggingface", "tokenizer_name": self.name, "vocab_size": self.vocab_size}


def save_tokenizer_metadata(tokenizer: Tokenizer, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "meta.pkl"
    with path.open("wb") as f:
        pickle.dump(tokenizer.metadata(), f)
    return path


def load_tokenizer_metadata(directory: str | Path) -> Dict[str, Any]:
    path = Path(directory) / "meta.pkl"
    if not path.exists():
        raise TokenizerError(f"Tokenizer metadata not found at {path}", hint="Run `forgeline data prepare` first.")
    with path.open("rb") as f:
        meta = pickle.load(f)
    if not isinstance(meta, dict) or "tokenizer_type" not in meta:
        raise TokenizerError(f"Malformed tokenizer metadata at {path}")
    return meta


def tokenizer_from_metadata(meta: Dict[str, Any]) -> Tokenizer:
    kind = meta.get("tokenizer_type", "char")
    if kind == "char":
        return CharTokenizer.from_metadata(meta)
    if kind == "tiktoken":
        return TiktokenTokenizer(meta.get("encoding_name", "cl100k_base"))
    if kind == "huggingface":
        return HFTokenizer(meta.get("tokenizer_name", ""))
    raise TokenizerError(f"Unknown tokenizer type {kind!r}")


def load_tokenizer(directory: str | Path) -> Tokenizer:
    return tokenizer_from_metadata(load_tokenizer_metadata(directory))


def resolve_tokenizer(spec: str, data_dir: Optional[str | Path] = None) -> Tokenizer:
    """Resolve a tokenizer spec: ``auto`` (from meta.pkl), ``char``, ``tiktoken[:enc]``, ``hf:<name>``."""
    if spec == "auto":
        if data_dir is None:
            raise TokenizerError("tokenizer=auto requires a data directory containing meta.pkl")
        return load_tokenizer(data_dir)
    if spec == "char":
        if data_dir is not None and (Path(data_dir) / "meta.pkl").exists():
            return load_tokenizer(data_dir)
        return CharTokenizer.ascii()
    if spec.startswith("tiktoken"):
        _, _, enc = spec.partition(":")
        return TiktokenTokenizer(enc or "cl100k_base")
    if spec.startswith("hf:"):
        return HFTokenizer(spec[3:])
    raise TokenizerError(f"Unknown tokenizer spec {spec!r}; use auto|char|tiktoken[:enc]|hf:<name>")


def token_dtype_for(vocab_size: int):
    """uint16 fits vocabularies up to 65535; larger vocabularies use uint32."""
    import numpy as np

    return np.uint32 if vocab_size > 65535 else np.uint16
