"""Latency / throughput / memory measurement for generation."""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List, Optional

import torch

from forgeline.core.protocols import EvaluationResult, GenerationSettings


def measure_generation(policy, prompt_ids: torch.Tensor, max_new_tokens: int = 32, runs: int = 3, warmup: int = 1,
                       temperature: float = 0.0) -> Dict[str, float]:
    """Tokens/sec and per-run latency for one prompt on the current device (no numbers are assumed)."""
    settings = GenerationSettings(max_new_tokens=max_new_tokens, temperature=temperature)
    device = policy.device
    ids = prompt_ids.unsqueeze(0).to(device) if prompt_ids.dim() == 1 else prompt_ids.to(device)
    for _ in range(warmup):
        policy.generate(ids, settings)
    latencies: List[float] = []
    tokens: List[int] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = policy.generate(ids, settings)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - t0)
        tokens.append(int(out.shape[1]))
    total_tokens = sum(tokens)
    total_time = sum(latencies)
    metrics = {
        "tokens_per_second": total_tokens / total_time if total_time > 0 else 0.0,
        "latency_mean_s": statistics.mean(latencies),
        "latency_p50_s": statistics.median(latencies),
        "latency_max_s": max(latencies),
        "tokens_generated": float(total_tokens),
        "runs": float(runs),
    }
    if device.type == "cuda":
        metrics["peak_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    return metrics


class PerformanceSuite:
    name = "performance"

    def __init__(self, prompt_ids: torch.Tensor, max_new_tokens: int = 32, runs: int = 3):
        self.prompt_ids, self.max_new_tokens, self.runs = prompt_ids, max_new_tokens, runs

    def run(self, policy, **kwargs: Any) -> EvaluationResult:
        m = measure_generation(policy, self.prompt_ids, self.max_new_tokens, self.runs)
        return EvaluationResult(self.name, m, n_samples=self.runs, details={"device": str(policy.device)})
