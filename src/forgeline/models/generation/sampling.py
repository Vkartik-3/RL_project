"""Token sampling: temperature, top-k, top-p, min-p, repetition penalty.

``apply_sampling_filters`` operates on ``[B, V]`` logits and is shared by the
model's ``generate``, the rollout engine and the inference engine, so all
sampling paths have identical semantics.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def apply_sampling_filters(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    past_tokens: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Return filtered logits ``[B, V]``. ``temperature == 0`` leaves logits unscaled (greedy)."""
    logits = logits.clone()
    if repetition_penalty != 1.0 and past_tokens:
        past_ids = torch.tensor(list(past_tokens), device=logits.device).unique()
        penalised = logits[:, past_ids]
        penalised = torch.where(penalised > 0, penalised / repetition_penalty, penalised * repetition_penalty)
        logits[:, past_ids] = penalised

    if temperature > 0:
        logits = logits / temperature

    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")

    if min_p is not None and min_p > 0:
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        logits[probs < min_p * max_prob] = float("-inf")

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative - sorted_probs >= top_p
        sorted_logits[sorted_mask] = float("-inf")
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    return logits


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Greedy when ``temperature == 0`` else multinomial. Returns ``[B, 1]``."""
    probs = F.softmax(logits, dim=-1)
    if temperature <= 0:
        return torch.argmax(probs, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_tokens(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    use_cache: bool = True,
    stop_token_ids: Iterable[int] = (),
    return_logprobs: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Autoregressive sampling for a batch of prompts ``[B, T]``.

    With ``use_cache`` the model's ``prefill``/``step`` cache path is used
    (O(1) per token); otherwise the full context is re-run each step.
    Generation stops early for the whole batch when every sequence has
    produced a stop token. ``return_logprobs`` additionally returns the
    sampled-token log-probabilities ``[B, R]`` (post-filter distribution).
    """
    stop = set(int(t) for t in stop_token_ids)
    B = idx.size(0)
    past_tokens: Optional[List[int]] = idx[0].tolist() if (repetition_penalty != 1.0 and B == 1) else None
    generated: List[torch.Tensor] = []
    logprobs: List[torch.Tensor] = []
    finished = torch.zeros(B, dtype=torch.bool, device=idx.device)
    block_size = model.spec.block_size

    def _sample(logits: torch.Tensor) -> torch.Tensor:
        filtered = apply_sampling_filters(logits, temperature, top_k, top_p, min_p, repetition_penalty, past_tokens)
        nxt = sample_from_logits(filtered, temperature)
        if return_logprobs:
            logprobs.append(F.log_softmax(filtered, dim=-1).gather(1, nxt).squeeze(1))
        return nxt

    if use_cache and idx.size(1) < block_size:
        logits, cache = model.prefill(idx)
        position = idx.size(1)
        for _ in range(max_new_tokens):
            nxt = _sample(logits)
            generated.append(nxt)
            if past_tokens is not None:
                past_tokens.append(int(nxt.item()))
            finished |= torch.tensor([int(t) in stop for t in nxt.squeeze(1).tolist()], device=idx.device)
            if bool(finished.all()) or position >= block_size:
                break
            logits, cache = model.step(nxt, cache, position)
            position += 1
    else:
        seq = idx
        for _ in range(max_new_tokens):
            ctx = seq if seq.size(1) <= block_size else seq[:, -block_size:]
            logits, _ = model(ctx)
            nxt = _sample(logits[:, -1, :])
            generated.append(nxt)
            seq = torch.cat([seq, nxt], dim=1)
            if past_tokens is not None:
                past_tokens.append(int(nxt.item()))
            finished |= torch.tensor([int(t) in stop for t in nxt.squeeze(1).tolist()], device=idx.device)
            if bool(finished.all()):
                break

    out = torch.cat([idx] + generated, dim=1) if generated else idx
    if return_logprobs:
        lp = torch.stack(logprobs, dim=1) if logprobs else torch.zeros(B, 0, device=idx.device)
        return out, lp
    return out
