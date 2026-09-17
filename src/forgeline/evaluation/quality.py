"""Quality evaluators: held-out loss / perplexity, verifier pass rate, agent accuracy, malformed rate."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import torch

from forgeline.core.protocols import EvaluationResult, GenerationSettings, RewardProvider
from forgeline.data.pretraining import MemmapCorpus
from forgeline.data.records import VerifiableTask
from forgeline.rollouts.engine import RolloutConfig, RolloutEngine


class HeldOutLossSuite:
    name = "heldout_loss"

    def __init__(self, corpus: MemmapCorpus, batches: int = 20, batch_size: int = 8, seed: int = 0):
        self.corpus, self.batches, self.batch_size = corpus, batches, batch_size
        self.gen = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def run(self, model, **kwargs: Any) -> EvaluationResult:
        model.eval()
        device = next(model.parameters()).device
        losses = []
        for _ in range(self.batches):
            x, y = self.corpus.sample_batch(self.batch_size, device, self.gen)
            _, loss = model(x, y)
            losses.append(float(loss))
        mean = sum(losses) / len(losses)
        return EvaluationResult(self.name, {"loss": mean, "perplexity": math.exp(min(mean, 50.0))},
                                n_samples=self.batches * self.batch_size)


class VerifierPassRateSuite:
    """Greedy-decode every task once, score with a reward provider (pass rate, reward, malformed rate)."""

    name = "verifier_pass_rate"

    def __init__(self, tasks: Sequence[VerifiableTask], reward: RewardProvider, max_new_tokens: int = 64,
                 temperature: float = 0.0, limit: Optional[int] = None):
        self.tasks = list(tasks)[: limit or None]
        self.reward = reward
        self.settings = RolloutConfig(group_size=1, max_new_tokens=max_new_tokens, temperature=temperature, record_old_logprobs=False)

    @torch.no_grad()
    def run(self, policy, **kwargs: Any) -> EvaluationResult:
        engine = RolloutEngine(policy, self.settings)
        per_item: List[Dict[str, Any]] = []
        passed = malformed = 0
        total = 0.0
        for task in self.tasks:
            t = engine.rollout(torch.tensor(policy.encode(task.prompt), dtype=torch.long), task.to_dict())[0]
            r = self.reward.score(t)
            ok = bool(r.passed)
            bad = ("error" in r.info) or (r.info.get("predicted", "x") is None)
            passed += int(ok)
            malformed += int(bad)
            total += r.value
            per_item.append({"prompt": task.prompt[:80], "output": t.response_text[:200], "reward": r.value, "passed": ok})
        n = max(len(self.tasks), 1)
        return EvaluationResult(self.name, {"pass_rate": passed / n, "reward": total / n, "malformed_rate": malformed / n},
                                n_samples=len(self.tasks), per_item=per_item)


class AgentAccuracySuite:
    """Multi-turn tool-use accuracy and tool-use rate on tagged-answer tasks."""

    name = "agent_accuracy"

    def __init__(self, tasks: Sequence[VerifiableTask], agent_engine, limit: Optional[int] = None):
        self.tasks = list(tasks)[: limit or None]
        self.engine = agent_engine

    @torch.no_grad()
    def run(self, policy=None, **kwargs: Any) -> EvaluationResult:
        from forgeline.rollouts.agent import build_agent_prompt
        from forgeline.rollouts.verifiers import TaggedAnswerVerifier

        verifier = TaggedAnswerVerifier()
        correct = used_tool = 0
        for task in self.tasks:
            _, blocks, output = self.engine.run_episode(build_agent_prompt(task.prompt), temperature=0.0)
            correct += int(verifier.verify(output, task.answer).value >= 1.0)
            used_tool += int("<tool_call>" in output)
        n = max(len(self.tasks), 1)
        return EvaluationResult(self.name, {"accuracy": correct / n, "tool_use_rate": used_tool / n}, n_samples=len(self.tasks))
