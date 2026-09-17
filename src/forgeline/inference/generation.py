"""Simple text generation with timing statistics (uses the model's cached decode path)."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import torch

from forgeline.core.protocols import GenerationSettings
from forgeline.data.tokenizers import Tokenizer
from forgeline.models.transformer.model import TransformerLM


@torch.no_grad()
def generate_text(model: TransformerLM, tokenizer: Tokenizer, prompt: str, settings: Optional[GenerationSettings] = None,
                  device: Optional[torch.device] = None, spec_generator: Any = None) -> Tuple[str, Dict[str, float]]:
    settings = settings or GenerationSettings(max_new_tokens=200, temperature=0.8, top_k=50)
    device = device or next(model.parameters()).device
    ids = tokenizer.encode(prompt) or [0]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    t0 = time.perf_counter()
    if spec_generator is not None:
        y = spec_generator.generate(x, max_new_tokens=settings.max_new_tokens, temperature=settings.temperature, top_k=settings.top_k)
    else:
        y = model.generate(x, settings.max_new_tokens, temperature=settings.temperature, top_k=settings.top_k, top_p=settings.top_p,
                           min_p=settings.min_p, repetition_penalty=settings.repetition_penalty, use_cache=settings.use_cache,
                           stop_token_ids=tuple(settings.stop_token_ids))
    elapsed = time.perf_counter() - t0
    out_ids = y[0].tolist()
    n_new = len(out_ids) - len(ids)
    text = tokenizer.decode(out_ids)
    return text, {"n_generated": n_new, "elapsed_s": elapsed, "tokens_per_sec": n_new / elapsed if elapsed > 0 else 0.0}
