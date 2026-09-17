"""Process reward: dense step-level credit for chain-of-thought reasoning.

``total = step_weight · Σ_t γ^(T−1−t) · r_t + final_reward`` where each step
scores 0.15 (verified numeric computation / valid code), 0.10 (parsable
arithmetic) or 0.05 (code that failed), and ``final_reward`` comes from the
tagged-answer verifier.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from forgeline.core.protocols import RewardResult, Trajectory
from forgeline.rollouts.tools.python_executor import execute_python
from forgeline.rollouts.verifiers import TaggedAnswerVerifier

_STEP_SEP = re.compile(
    r"\n\s*\n"
    r"|(?:^|\n)(?:Step\s+\d+[:\.]|#+\s)"
    r"|(?:^|\n)(?:First|Second|Third|Fourth|Next|Then|Finally)[,:\s]",
    re.MULTILINE | re.IGNORECASE,
)
_CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
_ASSIGN_STMT = re.compile(r"([\w_]+)\s*=\s*(.+)")
_MATH_OPS = re.compile(r"[-\d\s()+\-*/%^.]+")
_NUMERIC = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

STEP_REWARD_VERIFIED = 0.15
STEP_REWARD_PARSED = 0.10
STEP_REWARD_CODE_FAILED = 0.05


def safe_eval_arithmetic(expr: str) -> Optional[float]:
    """Evaluate a pure arithmetic expression without builtins; ``None`` on any error."""
    expr = expr.replace(",", "").replace("^", "**")
    if not re.fullmatch(r"[-\d\s()+*/%.]+", expr):
        return None
    try:
        result = eval(expr, {"__builtins__": {}})  # noqa: S307 - input restricted to arithmetic chars
        if isinstance(result, (int, float)) and result == result and abs(result) != float("inf"):
            return float(result)
    except Exception:
        pass
    return None


def parse_steps(response: str) -> List[str]:
    raw = _STEP_SEP.split(response)
    steps: List[str] = []
    for s in raw:
        s = (s or "").strip()
        if not s:
            continue
        if steps and len(s) < 25:
            steps[-1] += " " + s
        else:
            steps.append(s)
    return steps if steps else [response]


def score_step(step: str, execute_code: bool = True) -> float:
    for code in _CODE_BLOCK.findall(step):
        code = code.strip()
        if not code:
            continue
        if not execute_code:
            return STEP_REWARD_CODE_FAILED
        out = execute_python(code, timeout=3.0)
        return STEP_REWARD_VERIFIED if (not out.error and out.stdout) else STEP_REWARD_CODE_FAILED
    for m in _ASSIGN_STMT.finditer(step):
        rhs = m.group(2).strip()
        parts = re.split(r"\s*=\s*(?=\d)", rhs, maxsplit=1)
        result = safe_eval_arithmetic(parts[0].strip())
        if result is None:
            continue
        if len(parts) > 1:
            stated_nums = _NUMERIC.findall(parts[1])
            if stated_nums:
                try:
                    stated = float(stated_nums[0].replace(",", ""))
                    if abs(result - stated) < max(1e-4, abs(stated) * 1e-4):
                        return STEP_REWARD_VERIFIED
                    continue
                except ValueError:
                    pass
        return STEP_REWARD_PARSED
    for m in _MATH_OPS.finditer(step):
        expr = m.group(0).strip()
        if len(expr) < 4 or not any(op in expr for op in "+-*/"):
            continue
        if safe_eval_arithmetic(expr) is not None:
            return STEP_REWARD_PARSED
    return 0.0


def compute_process_reward(response: str, ground_truth: str, gamma: float = 0.9, step_weight: float = 1.0,
                           execute_code: bool = True) -> Tuple[float, List[float]]:
    steps = parse_steps(response)
    step_scores = [score_step(s, execute_code=execute_code) for s in steps]
    T = len(step_scores)
    discounted = step_weight * sum((gamma ** (T - 1 - t)) * r for t, r in enumerate(step_scores))
    final = TaggedAnswerVerifier().verify(response, ground_truth).value
    return discounted + final, step_scores + [final]


class ProcessReward:
    """:class:`RewardProvider` producing the discounted step reward + final-answer reward."""

    name = "process"

    def __init__(self, gamma: float = 0.9, step_weight: float = 1.0, target_key: str = "answer", execute_code: bool = True):
        self.gamma, self.step_weight, self.target_key, self.execute_code = gamma, step_weight, target_key, execute_code

    def score(self, trajectory: Trajectory) -> RewardResult:
        total, scores = compute_process_reward(trajectory.response_text, str(trajectory.task.get(self.target_key, "")),
                                               gamma=self.gamma, step_weight=self.step_weight, execute_code=self.execute_code)
        final = scores[-1]
        return RewardResult(value=total, passed=final >= 1.0,
                            components={"step": total - final, "final": final},
                            info={"step_scores": scores[:-1], "step_fraction": (total - final) / (total + 1e-8)})
