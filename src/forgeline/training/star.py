"""Self-taught reasoning loops.

* :class:`STaRTrainer` — sample rationales, keep the ones whose answer is
  correct (rejection sampling), SFT on them, repeat.
* :class:`HillClimber` — rejection-sample high-reward tool-use trajectories
  into the training set, then run agent GRPO rounds; checkpoint on improvement.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from forgeline.core.errors import ConfigError
from forgeline.core.protocols import GenerationSettings
from forgeline.data.records import VerifiableTask, write_jsonl
from forgeline.observability.logging import get_logger
from forgeline.rollouts.agent import AgentRolloutConfig, AgentRolloutEngine, build_agent_prompt
from forgeline.rollouts.tools.parser import parse_tagged_output
from forgeline.rollouts.verifiers import TaggedAnswerVerifier, extract_final_answer

log = get_logger("forgeline.star")

_ANS_RE = re.compile(r"<answer>\s*([\d,.\-]+)\s*</answer>", re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def star_prompt(problem: str) -> str:
    return ("Solve the math problem step by step. Show every calculation.\n"
            f"End with: <answer>NUMBER</answer>\nProblem: {problem}\nSolution:")


def extract_star_answer(text: str) -> Optional[str]:
    m = _ANS_RE.search(text)
    if m:
        return m.group(1).strip().replace(",", "")
    nums = _NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def answers_match(pred: Optional[str], gt: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gt.replace(",", ""))) < 1e-4
    except ValueError:
        return pred.strip().lower() == gt.strip().lower()


@dataclass
class STaRConfig:
    num_rounds: int = 3
    samples_per_problem: int = 8
    sft_epochs: int = 1
    learning_rate: float = 5e-6
    max_new_tokens: int = 256
    num_problems: int = 20
    temperature: float = 0.9
    eval_fraction: float = 0.1
    seed: int = 0


class STaRTrainer:
    def __init__(self, policy, tasks: Sequence[VerifiableTask], cfg: STaRConfig, output_dir: str | Path):
        if len(tasks) < 2:
            raise ConfigError("STaR needs at least two tasks (train + eval)")
        self.policy, self.cfg = policy, cfg
        rows = list(tasks)
        n_eval = max(1, int(len(rows) * cfg.eval_fraction))
        self.eval_set = rows[-n_eval:]
        self.train_set = rows[: cfg.num_problems]
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=cfg.learning_rate)
        self.history: List[Dict[str, Any]] = []

    def _generate(self, prompt: str, temperature: float) -> str:
        ids = torch.tensor(self.policy.encode(prompt), dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            out = self.policy.generate(ids.to(self.policy.device),
                                       GenerationSettings(max_new_tokens=self.cfg.max_new_tokens, temperature=temperature, top_p=0.95 if temperature > 0 else None))
        return self.policy.decode(out[0])

    def generate_rationales(self) -> List[Dict[str, str]]:
        correct = []
        for task in self.train_set:
            prompt = star_prompt(task.prompt)
            for _ in range(self.cfg.samples_per_problem):
                rationale = self._generate(prompt, self.cfg.temperature)
                if answers_match(extract_star_answer(rationale), task.answer or ""):
                    correct.append({"prompt": prompt, "rationale": rationale, "answer": task.answer or ""})
        return correct

    def sft_on_rationales(self, examples: List[Dict[str, str]]) -> float:
        total, steps = 0.0, 0
        self.policy.train()
        for _ in range(self.cfg.sft_epochs):
            for ex in examples:
                r = self.policy.encode(ex["rationale"])
                if not r:
                    continue
                q = torch.tensor(self.policy.encode(ex["prompt"]), dtype=torch.long).unsqueeze(0)
                r_ids = torch.tensor(r, dtype=torch.long).unsqueeze(0)
                block = getattr(self.policy.spec, "block_size", None)
                if block is not None and q.shape[1] + r_ids.shape[1] > block:
                    q = q[:, -(block - r_ids.shape[1]):] if block > r_ids.shape[1] else q[:, -1:]
                    r_ids = r_ids[:, : block - q.shape[1]]
                loss = -self.policy.logprobs(q, r_ids).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.trainable_parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                total += float(loss)
                steps += 1
        self.policy.eval()
        return total / max(steps, 1)

    def evaluate(self) -> float:
        correct = 0
        for task in self.eval_set:
            pred = extract_star_answer(self._generate(star_prompt(task.prompt), 0.0))
            correct += int(answers_match(pred, task.answer or ""))
        return correct / max(len(self.eval_set), 1)

    def run(self) -> Dict[str, Any]:
        baseline = self.evaluate()
        best = baseline
        self.history.append({"round": 0, "accuracy": round(baseline, 4), "correct_rationales": 0, "sft_loss": None})
        for rnd in range(1, self.cfg.num_rounds + 1):
            t0 = time.time()
            correct = self.generate_rationales()
            yield_rate = len(correct) / max(len(self.train_set) * self.cfg.samples_per_problem, 1)
            sft_loss = self.sft_on_rationales(correct) if correct else None
            acc = self.evaluate()
            if acc > best:
                best = acc
                torch.save(self.policy.state_for_checkpoint(), self.out / "best_policy.pt")
            self.history.append({"round": rnd, "accuracy": round(acc, 4), "correct_rationales": len(correct),
                                 "yield_rate": round(yield_rate, 4), "sft_loss": round(sft_loss, 6) if sft_loss is not None else None,
                                 "elapsed_s": round(time.time() - t0, 1)})
            log.info("star_round", round=rnd, accuracy=acc, correct=len(correct))
        results = {"method": "star", "num_rounds": self.cfg.num_rounds, "baseline_acc": round(baseline, 4),
                   "best_acc": round(best, 4), "improvement": round(best - baseline, 4), "history": self.history}
        (self.out / "star_results.json").write_text(json.dumps(results, indent=2))
        return results


# ═══════════════════════════════════════════════════════════════════════════
#  Hill climbing: rejection sampling + agent GRPO rounds
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HillClimbConfig:
    num_rounds: int = 5
    rollouts_per_problem: int = 4
    reward_threshold: float = 0.5
    top_k_per_problem: int = 2
    grpo_steps_per_round: int = 50
    grpo_group_size: int = 4
    grpo_learning_rate: float = 5e-6
    grpo_clip: float = 0.2
    grpo_kl_coef: float = 0.04
    grpo_prompts_per_step: int = 2
    max_new_tokens: int = 200
    eval_problems: int = 20
    seed: int = 42


class HillClimber:
    def __init__(self, policy, tasks: Sequence[VerifiableTask], cfg: HillClimbConfig, output_dir: str | Path,
                 agent_config: Optional[AgentRolloutConfig] = None):
        self.policy, self.cfg = policy, cfg
        rows = list(tasks)
        n_eval = min(cfg.eval_problems, max(1, len(rows) // 5))
        self.eval_set, self.train_set = rows[-n_eval:], rows[:-n_eval] or rows
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.agent_cfg = agent_config or AgentRolloutConfig(max_new_tokens=cfg.max_new_tokens)
        self.engine = AgentRolloutEngine(policy, self.agent_cfg)
        self.verifier = TaggedAnswerVerifier()
        self.results: List[Dict[str, Any]] = []
        self.best_accuracy = 0.0

    def collect_rollouts(self) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for task in self.train_set:
            prompt = build_agent_prompt(task.prompt)
            cands = []
            for _ in range(self.cfg.rollouts_per_problem):
                _, _, output = self.engine.run_episode(prompt, temperature=0.9)
                reward = self.verifier.verify(output, task.answer).value
                if reward >= self.cfg.reward_threshold:
                    cands.append({"trajectory": output, "reward": reward})
            cands.sort(key=lambda c: c["reward"], reverse=True)
            for c in cands[: self.cfg.top_k_per_problem]:
                kept.append({"prompt": task.prompt, "answer": task.answer, "trajectory": c["trajectory"], "reward": c["reward"],
                             "tools": list(task.tools) or ["python_executor"]})
        return kept

    def evaluate(self) -> Dict[str, float]:
        correct = tool_uses = 0
        rewards = []
        for task in self.eval_set:
            _, _, output = self.engine.run_episode(build_agent_prompt(task.prompt), temperature=0.0)
            rewards.append(self.verifier.verify(output, task.answer).value)
            pred = extract_final_answer(output)
            correct += int(pred is not None and pred.strip() == (task.answer or "").strip())
            tool_uses += int(parse_tagged_output(output).tool_call is not None)
        n = max(len(self.eval_set), 1)
        return {"accuracy": correct / n, "tool_use": tool_uses / n, "avg_reward": sum(rewards) / n}

    def _grpo_round(self, tasks: Sequence[VerifiableTask], rnd: int) -> None:
        from forgeline.core.config import ExperimentManifest, TrainerConfig, OptimizerConfig, ScheduleConfig
        from forgeline.core.lifecycle import RunContext
        from forgeline.rollouts.rewards import build_reward_provider
        from forgeline.training.common.trainer import Trainer
        from forgeline.training.grpo import GRPOAlgorithm, GRPOConfig

        manifest = ExperimentManifest(
            name=f"hill_climb_round{rnd}", algorithm="grpo",
            model=self.policy.spec if isinstance(self.policy.spec, __import__("forgeline").core.config.ModelSpec) else __import__("forgeline").core.config.ModelSpec(),
            output_dir=str(self.out / f"round_{rnd}"),
            trainer=TrainerConfig(max_steps=self.cfg.grpo_steps_per_round, eval_every=0, log_every=10,
                                  optimizer=OptimizerConfig(learning_rate=self.cfg.grpo_learning_rate, weight_decay=0.0),
                                  schedule=ScheduleConfig(name="constant", warmup_steps=0)),
        )
        ctx = RunContext.create(manifest, metrics_backend="local")
        alg = GRPOAlgorithm(self.policy, [t.to_dict() for t in tasks], build_reward_provider({"type": "tagged"}),
                            GRPOConfig(group_size=self.cfg.grpo_group_size, prompts_per_step=self.cfg.grpo_prompts_per_step,
                                       clip_ratio=self.cfg.grpo_clip, kl_coef=self.cfg.grpo_kl_coef, agent=True, seed=self.cfg.seed + rnd),
                            agent_rollout=self.agent_cfg)
        Trainer(ctx, alg).fit()
        ctx.close()

    def run(self) -> List[Dict[str, Any]]:
        baseline = self.evaluate()
        self.best_accuracy = baseline["accuracy"]
        self.results.append({"round": 0, **baseline, "dataset_size": len(self.train_set)})
        for rnd in range(1, self.cfg.num_rounds + 1):
            t0 = time.time()
            extra = self.collect_rollouts()
            augmented = list(self.train_set) + [VerifiableTask(prompt=r["prompt"], answer=r["answer"], task_type="tagged",
                                                                tools=r["tools"], meta={"trajectory": r["trajectory"]}) for r in extra]
            write_jsonl(self.out / f"augmented_round{rnd}.jsonl", augmented)
            self._grpo_round(augmented, rnd)
            metrics = self.evaluate()
            if metrics["accuracy"] > self.best_accuracy:
                self.best_accuracy = metrics["accuracy"]
                torch.save(self.policy.state_for_checkpoint(), self.out / f"best_round_{rnd}.pt")
            self.results.append({"round": rnd, "dataset_size": len(augmented), "added": len(extra),
                                 "elapsed_s": round(time.time() - t0, 1), **metrics})
            log.info("hill_climb_round", round=rnd, **metrics)
        (self.out / "hill_climb_results.json").write_text(json.dumps(self.results, indent=2))
        return self.results
