"""Deterministic verifiers: they judge a completion, never generate one.

Every verifier returns a :class:`RewardResult` with ``passed`` and a scalar
``value`` in ``[-1, 1]``. Exceptions inside a verifier are converted into a
failed result with the error recorded, so a single bad completion can never
abort a rollout batch.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Tuple

from forgeline.core.errors import VerifierError
from forgeline.core.protocols import RewardResult
from forgeline.core.registry import Registry
from forgeline.rollouts.tools.python_executor import execute_python

VERIFIERS: Registry = Registry("verifier")


def _safe(fn):
    """Decorator: turn verifier exceptions into a failed RewardResult."""

    def wrapper(self, output: str, target: Any = None) -> RewardResult:
        try:
            return fn(self, output, target)
        except Exception as exc:  # noqa: BLE001 - verifier robustness is the point
            return RewardResult(value=0.0, passed=False, info={"error": f"{type(exc).__name__}: {exc}"})

    return wrapper


# ── math ────────────────────────────────────────────────────────────────────

@VERIFIERS.register("math")
class MathVerifier:
    """Extract a final answer (``\\boxed{}``, ``#### x``, "the answer is", last number) and compare."""

    name = "math"

    def __init__(self, tolerance: float = 1e-6, partial_credit: bool = True):
        self.tolerance = tolerance
        self.partial_credit = partial_credit

    @staticmethod
    def extract_answer(text: str) -> Optional[str]:
        boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
        if boxed:
            return boxed[-1].strip()
        hashes = re.findall(r"####\s*(.+?)(?:\n|$)", text)
        if hashes:
            return hashes[-1].strip()
        answer_is = re.findall(r"(?:the\s+)?answer\s+is[:\s]*([^\n.]+)", text, re.IGNORECASE)
        if answer_is:
            return answer_is[-1].strip()
        numbers = re.findall(r"-?[\d,]+\.?\d*", text)
        if numbers:
            return numbers[-1].replace(",", "")
        return None

    @_safe
    def verify(self, output: str, target: Any = None) -> RewardResult:
        if target is None:
            raise VerifierError("math verifier needs a target answer")
        predicted = self.extract_answer(output)
        if predicted is None:
            return RewardResult(value=0.0, passed=False, info={"predicted": None})
        predicted = predicted.replace(",", "").replace("$", "").strip()
        gold = str(target).replace(",", "").replace("$", "").strip()
        if predicted.lower() == gold.lower():
            return RewardResult(value=1.0, passed=True, info={"predicted": predicted})
        try:
            p_val, g_val = float(predicted), float(gold)
        except ValueError:
            return RewardResult(value=0.0, passed=False, info={"predicted": predicted})
        if abs(p_val - g_val) < self.tolerance:
            return RewardResult(value=1.0, passed=True, info={"predicted": predicted})
        rel = abs(p_val - g_val) / max(abs(g_val), 1e-8)
        if self.partial_credit and rel < 0.01:
            return RewardResult(value=0.5, passed=False, info={"predicted": predicted, "relative_error": rel})
        return RewardResult(value=0.0, passed=False, info={"predicted": predicted, "relative_error": rel})


# ── exact match ─────────────────────────────────────────────────────────────

@VERIFIERS.register("exact_match")
class ExactMatchVerifier:
    name = "exact_match"

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    @_safe
    def verify(self, output: str, target: Any = None) -> RewardResult:
        if target is None:
            raise VerifierError("exact_match verifier needs a target")
        a, b = (self._norm(output), self._norm(str(target))) if self.normalize else (output, str(target))
        ok = a == b
        return RewardResult(value=1.0 if ok else 0.0, passed=ok)


# ── tagged answer (agent format) ────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[$,%]", "", s)
    s = re.sub(r"\s+", " ", s)
    try:
        return str(float(s.replace(",", "")))
    except ValueError:
        return s


_FINAL_ANS_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>")
_TOOL_RES_RE = re.compile(r"<tool_result>")


def extract_final_answer(text: str) -> Optional[str]:
    m = _FINAL_ANS_RE.search(text)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"(?<!\w)-?\d+(?:,\d{3})*(?:\.\d+)?(?!\w)", text)
    return nums[-1] if nums else None


@VERIFIERS.register("tagged_answer")
class TaggedAnswerVerifier:
    """Reward schedule: 1.0 correct ``<final_answer>``; 0.1 correct tool-use syntax; else 0.0."""

    name = "tagged_answer"

    def __init__(self, tool_use_partial: float = 0.1):
        self.tool_use_partial = tool_use_partial

    @_safe
    def verify(self, output: str, target: Any = None) -> RewardResult:
        if target is None:
            raise VerifierError("tagged_answer verifier needs a target")
        answer = extract_final_answer(output)
        if answer is not None and normalize_answer(answer) == normalize_answer(str(target)):
            return RewardResult(value=1.0, passed=True, info={"predicted": answer})
        if _TOOL_CALL_RE.search(output) and _TOOL_RES_RE.search(output):
            return RewardResult(value=self.tool_use_partial, passed=False, info={"predicted": answer, "tool_use": True})
        return RewardResult(value=0.0, passed=False, info={"predicted": answer})


def tag_format_reward(response: str) -> float:
    """Auxiliary 0.02 per ``<think>`` / ``<tool_call>`` / ``<final_answer>`` tag (max 0.06)."""
    score = 0.0
    if "<think>" in response:
        score += 0.02
    if "<tool_call>" in response:
        score += 0.02
    if "<final_answer>" in response:
        score += 0.02
    return score


# ── code execution ──────────────────────────────────────────────────────────

@VERIFIERS.register("code")
class CodeExecutionVerifier:
    """Extract code, append test code, execute in a subprocess with a timeout."""

    name = "code"

    def __init__(self, timeout: float = 5.0, failure_penalty: float = -0.1):
        self.timeout = timeout
        self.failure_penalty = failure_penalty

    @staticmethod
    def extract_code(text: str) -> str:
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if blocks:
            return blocks[-1].strip()
        lines = text.split("\n")
        code_lines = [l for l in lines if l.startswith("    ") or l.startswith("\t") or l.startswith("def ") or l.startswith("class ")]
        return "\n".join(code_lines) if code_lines else text

    @_safe
    def verify(self, output: str, target: Any = None) -> RewardResult:
        if not target:
            raise VerifierError("code verifier needs test code as the target")
        code = self.extract_code(output)
        result = execute_python(code + "\n" + str(target), timeout=self.timeout)
        if not result.error:
            return RewardResult(value=1.0, passed=True)
        if "AssertionError" in result.stderr:
            return RewardResult(value=0.0, passed=False, info={"stderr": result.stderr[:200]})
        return RewardResult(value=self.failure_penalty, passed=False,
                            info={"stderr": result.stderr[:200], "timed_out": result.timed_out})


# ── format ──────────────────────────────────────────────────────────────────

@VERIFIERS.register("format")
class FormatVerifier:
    """``cot`` (<think> blocks), ``steps`` (Step 1/2...), ``json`` (parseable JSON)."""

    name = "format"

    def __init__(self, format_spec: str = "cot"):
        if format_spec not in ("cot", "steps", "json"):
            raise VerifierError(f"unknown format_spec {format_spec!r}")
        self.format_spec = format_spec

    @_safe
    def verify(self, output: str, target: Any = None) -> RewardResult:
        spec = target or self.format_spec
        if spec == "cot":
            ok, score = self._check_cot(output)
        elif spec == "steps":
            ok, score = self._check_steps(output)
        else:
            ok, score = self._check_json(output)
        return RewardResult(value=score, passed=ok)

    @staticmethod
    def _check_cot(text: str) -> Tuple[bool, float]:
        score = 0.0
        if "<think>" in text and "</think>" in text:
            score += 0.5
            m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            if m and len(m.group(1).strip()) > 20:
                score += 0.3
        elif "think" in text.lower() or "let me" in text.lower():
            score += 0.2
        if "</think>" in text and len(text.split("</think>")[-1].strip()) > 5:
            score += 0.2
        return score >= 0.7, score

    @staticmethod
    def _check_steps(text: str) -> Tuple[bool, float]:
        steps = re.findall(r"(?:step\s+\d|^\d+[.):])", text, re.IGNORECASE | re.MULTILINE)
        return len(steps) >= 2, min(len(steps) * 0.25, 1.0)

    @staticmethod
    def _check_json(text: str) -> Tuple[bool, float]:
        try:
            json.loads(text.strip())
            return True, 1.0
        except (json.JSONDecodeError, ValueError):
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    json.loads(m.group())
                    return True, 0.8
                except (json.JSONDecodeError, ValueError):
                    pass
            return False, 0.0


def build_verifier(name: str, **kwargs: Any):
    return VERIFIERS.get(name)(**kwargs)


__all__ = [
    "VERIFIERS", "MathVerifier", "ExactMatchVerifier", "TaggedAnswerVerifier", "CodeExecutionVerifier",
    "FormatVerifier", "build_verifier", "extract_final_answer", "normalize_answer", "tag_format_reward",
]
