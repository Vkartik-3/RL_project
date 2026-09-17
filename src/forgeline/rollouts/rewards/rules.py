"""Rule-based rewards that need no model or target (length, format, repetition)."""

from __future__ import annotations

from typing import Optional

from forgeline.core.protocols import RewardResult, Trajectory


class LengthReward:
    """Closer to ``target_length`` tokens → higher reward in [0, 1]."""

    name = "length"

    def __init__(self, target_length: int = 100):
        self.target_length = target_length

    def score(self, trajectory: Trajectory) -> RewardResult:
        diff = abs(trajectory.response_length - self.target_length)
        return RewardResult(value=max(0.0, 1.0 - diff / self.target_length))


class TextFormatReward:
    """Structure heuristics on decoded text: punctuation, newlines, non-trivial length."""

    name = "text_format"

    def score(self, trajectory: Trajectory) -> RewardResult:
        text = trajectory.response_text
        score = 0.0
        if "." in text:
            score += 0.3
        if "\n" in text:
            score += 0.3
        if len(text) > 20:
            score += 0.2
        if text.strip():
            score += 0.2
        return RewardResult(value=score)


class RepetitionReward:
    """Unique-bigram ratio of the response tokens (1.0 = no repeated bigrams)."""

    name = "repetition"

    def score(self, trajectory: Trajectory) -> RewardResult:
        tokens = trajectory.response_ids.tolist()
        if len(tokens) < 4:
            return RewardResult(value=0.5)
        bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        return RewardResult(value=len(set(bigrams)) / max(len(bigrams), 1))


class ConstantReward:
    """Deterministic constant (useful for tests and dynamic-sampling edge cases)."""

    name = "constant"

    def __init__(self, value: float = 0.0):
        self.value = value

    def score(self, trajectory: Trajectory) -> RewardResult:
        return RewardResult(value=self.value)
