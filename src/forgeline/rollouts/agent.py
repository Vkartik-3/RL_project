"""Multi-turn agent rollouts with tool execution.

The policy generates until it emits a ``<tool_call>``; the tool result is
injected as an observation (never part of the policy gradient) and generation
continues, until a ``<final_answer>`` or the turn budget is reached.
Per-segment log-probs are conditioned on the full context including prior
tool results, i.e. ``p(turn_k | prompt, turn_1, result_1, ..., turn_{k-1}, result_{k-1})``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from forgeline.core.errors import ToolExecutionError
from forgeline.core.protocols import GenerationSettings, Trajectory
from forgeline.rollouts.tools.parser import format_tool_result, parse_tagged_output
from forgeline.rollouts.tools.python_executor import PythonExecutorTool

AGENT_SYSTEM_PROMPT = (
    "You are a reasoning agent with access to a Python executor.\n"
    "Workflow:\n"
    "  1. Think inside <think>...</think>\n"
    "  2. Call tools: <tool_call>{\"name\": \"python_executor\", \"args\": {\"code\": \"...\"}}</tool_call>\n"
    "  3. After <tool_result>, continue reasoning or give your answer\n"
    "  4. Final answer: <final_answer>...</final_answer>\n\n"
    "python_executor runs arbitrary Python and returns stdout."
)


def build_agent_prompt(problem: str, system_prompt: str = AGENT_SYSTEM_PROMPT) -> str:
    return f"{system_prompt}\n\nProblem: {problem}\n\nSolve step by step:"


@dataclass
class AgentRolloutConfig:
    max_turns: int = 3
    max_new_tokens: int = 128
    temperature: float = 0.9
    max_context_tokens: int = 256
    tools: Dict[str, Callable[[dict], str]] = field(default_factory=lambda: {"python_executor": PythonExecutorTool()})


class AgentRolloutEngine:
    """Runs tool-augmented episodes and computes per-segment log-probs."""

    def __init__(self, policy, config: AgentRolloutConfig):
        self.policy = policy
        self.cfg = config

    def _generate_text(self, context: str, temperature: float) -> str:
        ids = torch.tensor(self.policy.encode(context)[-self.cfg.max_context_tokens:], dtype=torch.long).unsqueeze(0)
        settings = GenerationSettings(max_new_tokens=self.cfg.max_new_tokens, temperature=temperature)
        with torch.no_grad():
            resp = self.policy.generate(ids.to(self.policy.device), settings)
        return self.policy.decode(resp[0])

    def run_episode(self, prompt: str, temperature: Optional[float] = None) -> Tuple[List[str], List[str], str]:
        """Returns ``(model_segments, tool_result_blocks, final_output)``."""
        temperature = self.cfg.temperature if temperature is None else temperature
        context = prompt
        segments: List[str] = []
        tool_blocks: List[str] = []
        for _ in range(self.cfg.max_turns + 1):
            segment = self._generate_text(context, temperature)
            segments.append(segment)
            parsed = parse_tagged_output(segment)
            if parsed.final_answer is not None:
                context += segment
                break
            if parsed.tool_call is not None and parsed.tool_call.name in self.cfg.tools:
                try:
                    result = self.cfg.tools[parsed.tool_call.name](parsed.tool_call.args)
                except Exception as exc:  # noqa: BLE001
                    result = f"ERROR: {ToolExecutionError(str(exc)).message}"
                block = format_tool_result(result)
                tool_blocks.append(block)
                context += segment + block
            else:
                context += segment
                break
        return segments, tool_blocks, "".join(segments)

    def rollout(self, task: Dict[str, Any], group_size: int, temperature: Optional[float] = None) -> List[Trajectory]:
        prompt = build_agent_prompt(task["prompt"])
        prompt_ids = torch.tensor(self.policy.encode(prompt), dtype=torch.long)
        out: List[Trajectory] = []
        for _ in range(group_size):
            segs, blocks, final = self.run_episode(prompt, temperature)
            resp_ids = torch.tensor(self.policy.encode(final) or [self.policy.pad_token_id], dtype=torch.long)
            out.append(Trajectory(prompt_ids=prompt_ids, response_ids=resp_ids, prompt_text=prompt, response_text=final,
                                  task=dict(task), segments=segs, tool_results=blocks,
                                  meta={"n_turns": len(segs), "n_tool_calls": len(blocks)}))
        return out

    # ── log-probs over multi-turn trajectories ────────────────────────────
    def _segment_logprob(self, context: str, segment: str) -> torch.Tensor:
        """Mean per-token log-prob of ``segment`` given ``context`` (scalar)."""
        q = self.policy.encode(context)[-self.cfg.max_context_tokens:]
        r = self.policy.encode(segment)[: self.cfg.max_context_tokens]
        if not q or not r:
            return torch.zeros((), device=self.policy.device)
        block = getattr(getattr(self.policy, "spec", None), "block_size", None)
        if block is not None and len(q) + len(r) > block:
            q = q[-(block - len(r)):] if block > len(r) else q[-1:]
            r = r[: block - len(q)]
        q_ids = torch.tensor(q, dtype=torch.long).unsqueeze(0)
        r_ids = torch.tensor(r, dtype=torch.long).unsqueeze(0)
        return self.policy.logprobs(q_ids, r_ids).mean()

    def trajectory_logprob(self, trajectory: Trajectory) -> torch.Tensor:
        """Average over segments of the mean per-token segment log-prob (differentiable)."""
        total = torch.zeros((), device=self.policy.device)
        context = trajectory.prompt_text
        for i, segment in enumerate(trajectory.segments):
            total = total + self._segment_logprob(context, segment)
            context += segment
            if i < len(trajectory.tool_results):
                context += trajectory.tool_results[i]
        return total / max(len(trajectory.segments), 1)
