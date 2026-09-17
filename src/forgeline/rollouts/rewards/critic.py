"""Heuristic critic: dense trajectory-level feedback to supplement sparse verifiable rewards."""

from __future__ import annotations

import re
from typing import Callable, Optional

from forgeline.core.protocols import RewardResult, Trajectory

_FINAL_ANS_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_RE = re.compile(r"<tool_call>", re.IGNORECASE)
_REASONING_KWS = {"because", "therefore", "first", "next", "then", "since", "step", "compute", "calculate", "find", "let", "we"}


def heuristic_critic_score(problem: str, output: str) -> float:
    """Four 0–0.25 dimensions: answer present, reasoning depth, tool use, coherence."""
    score = 0.0
    words = output.lower().split()
    if _FINAL_ANS_RE.search(output):
        score += 0.25
    kw = sum(1 for w in words if w in _REASONING_KWS)
    score += 0.25 if kw >= 3 else (0.10 if kw >= 1 else 0.0)
    if _TOOL_CALL_RE.search(output):
        score += 0.25
    if len(words) >= 20:
        ngrams = [" ".join(words[i : i + 4]) for i in range(len(words) - 3)]
        ratio = len(set(ngrams)) / max(len(ngrams), 1)
        score += 0.25 if ratio > 0.8 else (0.10 if ratio > 0.5 else 0.0)
    return round(min(score, 1.0), 4)


class CriticReward:
    """Heuristic or LLM-backed critic returning a score in [0, 1]."""

    name = "critic"

    def __init__(self, mode: str = "heuristic", critic_fn: Optional[Callable[[str], str]] = None):
        if mode not in ("heuristic", "llm"):
            raise ValueError("mode must be heuristic|llm")
        if mode == "llm" and critic_fn is None:
            raise ValueError("mode='llm' requires critic_fn(prompt) -> str")
        self.mode, self.critic_fn = mode, critic_fn

    def score(self, trajectory: Trajectory) -> RewardResult:
        problem = trajectory.task.get("prompt", trajectory.prompt_text)
        if self.mode == "heuristic":
            return RewardResult(value=heuristic_critic_score(problem, trajectory.response_text))
        prompt = (
            "You are an expert evaluator. Score this agent trajectory from 0.0 to 1.0.\n"
            "Criteria: answer present, reasoning depth, appropriate tool use, coherence.\n"
            f"Problem: {problem}\nAgent output:\n{trajectory.response_text}\n\n"
            "Respond with ONLY a decimal number between 0.0 and 1.0."
        )
        raw = (self.critic_fn(prompt) or "").strip()  # type: ignore[misc]
        m = re.search(r"\d+\.?\d*", raw)
        value = min(max(float(m.group()), 0.0), 1.0) if m else 0.0
        return RewardResult(value=value)
