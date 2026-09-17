"""Rollout engine: samples completions from a policy and returns ``Trajectory`` objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from forgeline.core.errors import RolloutError
from forgeline.core.protocols import GenerationSettings, PolicyModel, Trajectory
from forgeline.observability.logging import get_logger

log = get_logger("forgeline.rollouts")


@dataclass
class RolloutConfig:
    group_size: int = 4
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_prompt_length: int = 128
    record_old_logprobs: bool = True
    stop_token_ids: Sequence[int] = ()


class RolloutEngine:
    """Single-turn rollouts. ``group_size`` completions are sampled per prompt in one batch."""

    def __init__(self, policy: PolicyModel, config: RolloutConfig):
        self.policy = policy
        self.cfg = config

    def _settings(self) -> GenerationSettings:
        return GenerationSettings(max_new_tokens=self.cfg.max_new_tokens, temperature=self.cfg.temperature,
                                  top_k=self.cfg.top_k, top_p=self.cfg.top_p, stop_token_ids=tuple(self.cfg.stop_token_ids))

    @torch.no_grad()
    def rollout(self, prompt_ids: torch.Tensor, task: Optional[Dict[str, Any]] = None, prompt_text: str = "",
                group_size: Optional[int] = None) -> List[Trajectory]:
        """Sample ``group_size`` completions for one prompt ``[P]``."""
        G = group_size or self.cfg.group_size
        prompt_ids = prompt_ids[-self.cfg.max_prompt_length:]
        if prompt_ids.numel() == 0:
            raise RolloutError("empty prompt")
        batch = prompt_ids.unsqueeze(0).repeat(G, 1).to(self.policy.device)
        try:
            responses = self.policy.generate(batch, self._settings())
        except Exception as exc:  # noqa: BLE001
            raise RolloutError(f"generation failed: {type(exc).__name__}: {exc}") from exc
        if responses.shape[1] == 0:
            raise RolloutError("policy produced an empty response", hint="Increase max_new_tokens or check the prompt length vs block_size.")
        old_lp = self.policy.logprobs(batch, responses).detach() if self.cfg.record_old_logprobs else None
        stop = set(int(t) for t in self.cfg.stop_token_ids)
        out: List[Trajectory] = []
        for g in range(G):
            resp = responses[g]
            if stop:
                keep = [i for i, t in enumerate(resp.tolist()) if int(t) in stop]
                if keep:
                    resp = resp[: keep[0] + 1]
            traj = Trajectory(
                prompt_ids=prompt_ids.detach().cpu(), response_ids=resp.detach().cpu(), prompt_text=prompt_text,
                response_text=self._decode(resp), task=dict(task or {}),
                truncated=bool(resp.numel() >= self.cfg.max_new_tokens),
                old_logprobs=old_lp[g, : resp.numel()].cpu() if old_lp is not None else None,
            )
            out.append(traj)
        return out

    def rollout_many(self, prompts: Sequence[torch.Tensor], tasks: Optional[Sequence[Dict[str, Any]]] = None,
                     prompt_texts: Optional[Sequence[str]] = None) -> List[List[Trajectory]]:
        groups = []
        for i, p in enumerate(prompts):
            groups.append(self.rollout(p, tasks[i] if tasks else None, prompt_texts[i] if prompt_texts else ""))
        return groups

    def _decode(self, ids: torch.Tensor) -> str:
        decode = getattr(self.policy, "decode", None)
        return decode(ids) if decode is not None else ""


def stack_group(trajectories: Sequence[Trajectory], pad_value: int = 0):
    """Pad a group of trajectories sharing a prompt into ``(prompt [1,P], responses [G,R], mask [G,R])``."""
    prompt = trajectories[0].prompt_ids.unsqueeze(0)
    R = max(t.response_length for t in trajectories)
    G = len(trajectories)
    resp = torch.full((G, R), pad_value, dtype=torch.long)
    mask = torch.zeros((G, R), dtype=torch.float32)
    for i, t in enumerate(trajectories):
        n = t.response_length
        resp[i, :n] = t.response_ids
        mask[i, :n] = 1.0
    return prompt.repeat(G, 1), resp, mask


def stack_old_logprobs(trajectories: Sequence[Trajectory]) -> torch.Tensor:
    R = max(t.response_length for t in trajectories)
    out = torch.zeros((len(trajectories), R), dtype=torch.float32)
    for i, t in enumerate(trajectories):
        if t.old_logprobs is not None:
            out[i, : t.old_logprobs.numel()] = t.old_logprobs.float()
    return out
