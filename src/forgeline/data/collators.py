"""Collators: turn tokenized records into padded tensors with masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from forgeline.core.errors import DatasetError

IGNORE_INDEX = -1  # matches the native model's cross-entropy ``ignore_index``


def pad_sequences(seqs: Sequence[Sequence[int]], pad_value: int, side: str = "right") -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad to the longest sequence. Returns ``(ids, attention_mask)``."""
    if side not in ("left", "right"):
        raise DatasetError("padding side must be 'left' or 'right'")
    max_len = max((len(s) for s in seqs), default=0)
    ids = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        n = len(s)
        if n == 0:
            continue
        if side == "right":
            ids[i, :n] = torch.tensor(s, dtype=torch.long)
            mask[i, :n] = 1
        else:
            ids[i, max_len - n :] = torch.tensor(s, dtype=torch.long)
            mask[i, max_len - n :] = 1
    return ids, mask


@dataclass
class SupervisedBatch:
    input_ids: torch.Tensor  # [B, T]
    targets: torch.Tensor  # [B, T] with IGNORE_INDEX on prompt / padding
    attention_mask: torch.Tensor
    prompt_lengths: List[int]


def collate_supervised(examples: Sequence[Tuple[List[int], List[int]]], pad_value: int = 0,
                       max_length: int | None = None, mask_prompt: bool = True) -> SupervisedBatch:
    """``examples`` are ``(prompt_ids, response_ids)``; produces next-token targets."""
    seqs, prompt_lens = [], []
    for prompt_ids, response_ids in examples:
        full = list(prompt_ids) + list(response_ids)
        if max_length is not None:
            full = full[:max_length]
        if len(full) < 2:
            raise DatasetError("supervised example must contain at least two tokens")
        seqs.append(full)
        prompt_lens.append(min(len(prompt_ids), len(full)))
    ids, mask = pad_sequences(seqs, pad_value, side="right")
    input_ids = ids[:, :-1]
    targets = ids[:, 1:].clone()
    tmask = mask[:, 1:]
    targets[tmask == 0] = IGNORE_INDEX
    if mask_prompt:
        for i, p in enumerate(prompt_lens):
            # targets index j corresponds to predicting token j+1; prompt tokens 1..p-1 are masked
            targets[i, : max(p - 1, 0)] = IGNORE_INDEX
    return SupervisedBatch(input_ids=input_ids, targets=targets, attention_mask=mask[:, :-1], prompt_lengths=prompt_lens)


def collate_lm(examples: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = torch.stack([x for x, _ in examples])
    ys = torch.stack([y for _, y in examples])
    return xs, ys


def collate_preference(examples: Sequence[Dict[str, List[int]]], pad_value: int = 0) -> Dict[str, torch.Tensor]:
    """Keys: ``prompt``, ``chosen``, ``rejected`` (token id lists)."""
    prompts, _ = pad_sequences([e["prompt"] for e in examples], pad_value, side="left")
    chosen, chosen_mask = pad_sequences([e["chosen"] for e in examples], pad_value, side="right")
    rejected, rejected_mask = pad_sequences([e["rejected"] for e in examples], pad_value, side="right")
    return {"prompt": prompts, "chosen": chosen, "chosen_mask": chosen_mask,
            "rejected": rejected, "rejected_mask": rejected_mask}
