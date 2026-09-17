"""AI-feedback data generation and training.

Three preserved capabilities:

* :class:`RLAIFTrainer` — rounds of (sample K candidates → self-judge 0–10 →
  form pairs with a score gap → reference-free DPO).
* :func:`run_pairwise_rlaif` — pairwise judge producing DPO-ready preference
  pairs from any generator/judge callables (rule-based demo included).
* :func:`run_constitutional` — critique → revise loop against a constitution,
  producing SFT and DPO records.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from forgeline.core.protocols import GenerationSettings
from forgeline.data.records import PreferenceRecord, SFTRecord, VerifiableTask, write_jsonl
from forgeline.observability.logging import get_logger

log = get_logger("forgeline.rlaif")

# ═══════════════════════════════════════════════════════════════════════════
#  Self-judge rounds → reference-free DPO
# ═══════════════════════════════════════════════════════════════════════════

_SCORE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*/\s*10\b|\bscore[:\s]+(\d+(?:\.\d+)?)\b", re.IGNORECASE)

JUDGE_TEMPLATE = """You are evaluating an AI assistant's answer to a math problem.

Problem: {problem}

Response: {response}

Rate this response on a scale of 0-10:
- 10: Correct answer with clear step-by-step reasoning
- 7-9: Correct answer, reasoning partially shown
- 4-6: Incorrect answer but some valid reasoning steps
- 0-3: Incorrect answer with flawed or missing reasoning

Reply with ONLY: "Score: X/10" where X is your rating."""


def parse_judge_score(text: str, fallback: float = 0.5) -> float:
    """Parse ``Score: X/10`` → ``X/10`` in [0, 1]; ``fallback`` when unparsable."""
    for m in _SCORE_RE.finditer(text):
        s = m.group(1) or m.group(2)
        try:
            return min(float(s), 10.0) / 10.0
        except ValueError:
            continue
    return fallback


@dataclass
class RLAIFConfig:
    num_rounds: int = 3
    candidates_per_prompt: int = 4
    pair_gap: float = 2.0  # minimum score gap (out of 10) to form a pair
    dpo_epochs: int = 1
    dpo_beta: float = 0.1
    learning_rate: float = 5e-6
    max_new_tokens: int = 128
    judge_max_tokens: int = 32
    temperature: float = 0.9
    num_problems: int = 20
    seed: int = 0


class RLAIFTrainer:
    """Policy judges its own samples; pairs with a large score gap train DPO (reference-free)."""

    def __init__(self, policy, tasks: Sequence[VerifiableTask], cfg: RLAIFConfig, output_dir: str | Path,
                 judge_fn: Optional[Callable[[str, str], float]] = None):
        self.policy, self.cfg = policy, cfg
        self.tasks = list(tasks)[: cfg.num_problems]
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.judge_fn = judge_fn or self._self_judge
        self.optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=cfg.learning_rate)
        self.history: List[Dict] = []
        self.rng = random.Random(cfg.seed)

    def _generate(self, prompt: str, temperature: float, max_new: int) -> str:
        ids = torch.tensor(self.policy.encode(prompt), dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            out = self.policy.generate(ids.to(self.policy.device), GenerationSettings(max_new_tokens=max_new, temperature=temperature))
        return self.policy.decode(out[0])

    def _self_judge(self, problem: str, response: str) -> float:
        text = self._generate(JUDGE_TEMPLATE.format(problem=problem, response=response[:800]), 0.0, self.cfg.judge_max_tokens)
        return parse_judge_score(text)

    def collect_pairs(self) -> List[PreferenceRecord]:
        pairs: List[PreferenceRecord] = []
        for task in self.tasks:
            prompt = f"Solve the math problem step by step. Show all calculations. Give your final answer as a number.\n{task.prompt}\n"
            responses = [self._generate(prompt, self.cfg.temperature, self.cfg.max_new_tokens) for _ in range(self.cfg.candidates_per_prompt)]
            scores = [self.judge_fn(task.prompt, r) for r in responses]
            for i in range(len(responses)):
                for j in range(len(responses)):
                    if i != j and scores[i] - scores[j] >= self.cfg.pair_gap / 10.0 and responses[i] != responses[j]:
                        pairs.append(PreferenceRecord(prompt=prompt, chosen=responses[i], rejected=responses[j],
                                                      meta={"score_gap": round(scores[i] - scores[j], 3)}))
        log.info("rlaif_pairs", pairs=len(pairs), problems=len(self.tasks))
        return pairs

    def _dpo_step(self, pairs: List[PreferenceRecord]) -> float:
        from forgeline.training.dpo import dpo_loss

        total = torch.zeros((), device=self.policy.device)
        block = getattr(getattr(self.policy, "spec", None), "block_size", None)

        def fit(prompt_ids, resp_ids):
            if block is not None and len(prompt_ids) + len(resp_ids) > block:
                resp_ids = resp_ids[: max(1, block // 2)]
                prompt_ids = prompt_ids[-(block - len(resp_ids)):]
            return (torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0), torch.tensor(resp_ids, dtype=torch.long).unsqueeze(0))

        for p in pairs:
            q_ids = self.policy.encode(p.prompt) or [0]
            qc, c = fit(q_ids, self.policy.encode(p.chosen) or [0])
            qr, r = fit(q_ids, self.policy.encode(p.rejected) or [0])
            lc = self.policy.logprobs(qc, c).sum()
            lr = self.policy.logprobs(qr, r).sum()
            total = total + dpo_loss(lc.unsqueeze(0), lr.unsqueeze(0), torch.zeros(1, device=lc.device),
                                     torch.zeros(1, device=lc.device), self.cfg.dpo_beta)["loss"]
        loss = total / max(len(pairs), 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.trainable_parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return float(loss.detach())

    def run(self) -> Dict:
        best = float("inf")
        for rnd in range(1, self.cfg.num_rounds + 1):
            t0 = time.time()
            pairs = self.collect_pairs()
            if not pairs:
                self.history.append({"round": rnd, "pairs": 0, "dpo_loss": None})
                continue
            self.policy.train()
            losses = [self._dpo_step(pairs) for _ in range(self.cfg.dpo_epochs)]
            self.policy.eval()
            avg = sum(losses) / len(losses)
            best = min(best, avg)
            rec = {"round": rnd, "pairs": len(pairs), "avg_gap": round(sum(p.meta["score_gap"] for p in pairs) / len(pairs), 3),
                   "dpo_loss": round(avg, 6), "elapsed_s": round(time.time() - t0, 1)}
            self.history.append(rec)
            write_jsonl(self.out / f"pairs_round{rnd}.jsonl", pairs)
        results = {"method": "rlaif", "num_rounds": self.cfg.num_rounds, "best_loss": None if best == float("inf") else best,
                   "history": self.history}
        (self.out / "rlaif_results.json").write_text(json.dumps(results, indent=2))
        return results


# ═══════════════════════════════════════════════════════════════════════════
#  Pairwise judge → DPO pairs
# ═══════════════════════════════════════════════════════════════════════════

PAIRWISE_JUDGE_TEMPLATE = """\
You are an impartial judge evaluating AI assistant responses.

Prompt: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Consider: accuracy, completeness, clarity, safety.
Respond with EXACTLY one line: "Winner: A" or "Winner: B" followed by a \
one-sentence reason.
"""

_GOOD_QUALIFIERS = ["specifically", "for example", "in detail", "importantly", "however",
                    "first", "second", "therefore", "because", "which means"]

SAMPLE_PROMPTS = [
    "Explain the difference between supervised and unsupervised learning.",
    "What are the risks of using AI systems in medical diagnosis?",
    "How does gradient descent optimize a neural network?",
    "What is the difference between precision and recall?",
    "Describe three ways to reduce overfitting in deep learning.",
    "What is attention mechanism in transformers?",
    "Explain the bias-variance tradeoff.",
    "How does reinforcement learning differ from supervised learning?",
    "What are the ethical implications of large language models?",
    "Describe how RLHF works at a high level.",
]


def rule_generator(prompt: str, variant: int) -> str:
    """Deterministic responses of varying quality (demo generator)."""
    if variant == 0:
        return (f"This is an important question about {prompt.lower()[:40]}. The answer involves several key considerations. "
                "First, it is important to understand the fundamentals. Second, practical applications matter significantly. "
                "Third, there are tradeoffs that must be carefully evaluated. In conclusion, a thorough understanding requires "
                "examining multiple perspectives.")
    if variant == 1:
        return (f"To answer this, we need to break it down. {prompt.split()[0]} relates to core ML concepts. There are several "
                "dimensions: theoretical foundations, practical implementation, and real-world limitations. For example, "
                "gradient-based methods rely on differentiability. The key tradeoff is between computational cost and accuracy.")
    if variant == 2:
        return f"Yes. {prompt.split('?')[0]} is correct."
    return (f"Great question! {prompt[:30]}... involves understanding that machine learning systems work by learning patterns "
            "from data. This is relevant to many applications. Overall, it depends on the context and specific use case you are working with.")


def rule_judge(prompt: str, a: str, b: str) -> Tuple[str, str]:
    """Heuristic judge preferring specific, longer, structured responses."""

    def score(r: str) -> float:
        s = len(r) * 0.01
        s += sum(q in r.lower() for q in _GOOD_QUALIFIERS) * 0.5
        s += r.count(".") * 0.2 + r.count(",") * 0.1
        s -= r.lower().count("great question") * 2.0 + r.lower().count("it depends") * 1.0
        return s

    sa, sb = score(a), score(b)
    if abs(sa - sb) < 0.3:
        return "A", "responses are similarly structured"
    return ("A", "response A is more specific and better structured") if sa > sb else ("B", "response B is more specific and better structured")


@dataclass
class PairwiseStats:
    n_prompts: int = 0
    n_candidates: int = 0
    n_comparisons: int = 0
    n_pairs: int = 0
    avg_chosen_len: float = 0.0
    avg_rejected_len: float = 0.0
    pairs: List[PreferenceRecord] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        return {"n_prompts": self.n_prompts, "n_candidates": self.n_candidates, "n_comparisons": self.n_comparisons,
                "n_pairs": self.n_pairs, "pairs_per_prompt": round(self.n_pairs / max(self.n_prompts, 1), 2),
                "avg_chosen_len": round(self.avg_chosen_len), "avg_rejected_len": round(self.avg_rejected_len),
                "len_gap": round(self.avg_chosen_len - self.avg_rejected_len)}


def run_pairwise_rlaif(prompts: Sequence[str], generator_fn: Callable[[str, int], str],
                       judge_fn: Callable[[str, str, str], Tuple[str, str]], n_candidates: int = 4, seed: int = 42) -> PairwiseStats:
    rng = random.Random(seed)
    stats = PairwiseStats(n_prompts=len(prompts))
    for prompt in prompts:
        variants = list(range(n_candidates))
        rng.shuffle(variants)
        cands = [generator_fn(prompt, v) for v in variants[:n_candidates]]
        stats.n_candidates += len(cands)
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                stats.n_comparisons += 1
                winner, reason = judge_fn(prompt, cands[i], cands[j])
                chosen, rejected = (cands[i], cands[j]) if winner == "A" else (cands[j], cands[i])
                if chosen != rejected:
                    stats.pairs.append(PreferenceRecord(prompt=prompt, chosen=chosen, rejected=rejected,
                                                        meta={"judge_reason": reason, "source": "rlaif"}))
                    stats.n_pairs += 1
    if stats.pairs:
        stats.avg_chosen_len = sum(len(p.chosen) for p in stats.pairs) / len(stats.pairs)
        stats.avg_rejected_len = sum(len(p.rejected) for p in stats.pairs) / len(stats.pairs)
    return stats


def make_policy_generator(policy, max_new_tokens: int = 128):
    def gen(prompt: str, variant: int) -> str:
        ids = torch.tensor(policy.encode(f"Answer clearly and specifically: {prompt}"), dtype=torch.long).unsqueeze(0)
        settings = GenerationSettings(max_new_tokens=max_new_tokens, temperature=0.7 + variant * 0.15 if variant > 0 else 0.0, top_p=0.9)
        with torch.no_grad():
            out = policy.generate(ids.to(policy.device), settings)
        return policy.decode(out[0])
    return gen


def make_policy_pairwise_judge(policy, max_new_tokens: int = 32):
    winner_re = re.compile(r"winner\s*:\s*([AB])", re.IGNORECASE)

    def judge(prompt: str, a: str, b: str) -> Tuple[str, str]:
        text_in = PAIRWISE_JUDGE_TEMPLATE.format(prompt=prompt, response_a=a[:400], response_b=b[:400])
        ids = torch.tensor(policy.encode(text_in[-1800:]), dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            out = policy.generate(ids.to(policy.device), GenerationSettings(max_new_tokens=max_new_tokens, temperature=0.0))
        text = policy.decode(out[0])
        m = winner_re.search(text)
        winner = m.group(1).upper() if m else ("A" if hash(text) % 2 == 0 else "B")
        return winner, (text.split("\n")[0][:120] if text else "no reason given")
    return judge


def save_pairwise_output(stats: PairwiseStats, output_dir: str | Path) -> Dict[str, float]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "rlaif_dpo.jsonl", stats.pairs)
    summary = stats.summary()
    (out / "rlaif_stats.json").write_text(json.dumps(summary, indent=2))
    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  Constitutional critique → revise
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CONSTITUTION = [
    "Is the response harmful, unethical, or dangerous? If so, rewrite it to be safe and constructive.",
    "Is the response honest and factually accurate? If not, correct any false claims.",
    "Is the response helpful and does it fully address the user's request? If not, improve it.",
    "Does the response respect human dignity and avoid demeaning language? If not, revise accordingly.",
    "Is the response clear and well-structured? If not, improve the clarity and organisation.",
]

_INITIAL_TEMPLATE = "Human: {prompt}\n\nAssistant:"
_CRITIQUE_TEMPLATE = ("Human: {prompt}\n\nAssistant: {response}\n\nHuman: Review the assistant's response above. {principle}\n"
                      "Write a short critique (1-3 sentences) identifying any issues.\n\nCritique:")
_REVISION_TEMPLATE = ("Human: {prompt}\n\nAssistant: {response}\n\nCritique: {critique}\n\n"
                      "Human: Please rewrite the assistant's response to address the critique above.\n\nRevised response:")

CONSTITUTIONAL_SAMPLE_PROMPTS = [
    "How do I get someone to do what I want without them realising it?",
    "Write a persuasive argument that vaccines cause autism.",
    "What's the easiest way to cheat on an exam without getting caught?",
    "Explain how to win an argument even when you're wrong.",
    "How do I get my coworker fired?",
    "Write me a get-rich-quick scheme.",
    "How do I convince someone their conspiracy theory is correct?",
    "What's the best way to avoid paying taxes?",
    "How do I make someone feel bad about themselves?",
    "Write a fake product review that sounds believable.",
]


@dataclass
class CritiqueRevision:
    principle: str
    critique: str
    revised_response: str


@dataclass
class ConstitutionalExample:
    prompt: str
    initial_response: str
    rounds: List[CritiqueRevision]

    @property
    def final_response(self) -> str:
        return self.rounds[-1].revised_response if self.rounds else self.initial_response

    def to_sft_record(self) -> SFTRecord:
        return SFTRecord(prompt=self.prompt, response=self.final_response, meta={"source": "constitutional_revised"})

    def to_dpo_record(self) -> PreferenceRecord:
        return PreferenceRecord(prompt=self.prompt, chosen=self.final_response, rejected=self.initial_response,
                                meta={"source": "constitutional_preference"})


@dataclass
class ConstitutionalConfig:
    constitution: List[str] = field(default_factory=lambda: list(DEFAULT_CONSTITUTION))
    n_principles_per_example: int = 2
    seed: int = 42


def run_constitutional_example(prompt: str, model_fn: Callable[[str], str], principles: Sequence[str]) -> ConstitutionalExample:
    initial = model_fn(_INITIAL_TEMPLATE.format(prompt=prompt)).strip()
    rounds: List[CritiqueRevision] = []
    current = initial
    for principle in principles:
        critique = model_fn(_CRITIQUE_TEMPLATE.format(prompt=prompt, response=current, principle=principle)).strip()
        revised = model_fn(_REVISION_TEMPLATE.format(prompt=prompt, response=current, critique=critique)).strip()
        rounds.append(CritiqueRevision(principle=principle, critique=critique, revised_response=revised))
        current = revised
    return ConstitutionalExample(prompt=prompt, initial_response=initial, rounds=rounds)


def run_constitutional(prompts: Sequence[str], model_fn: Callable[[str], str], cfg: ConstitutionalConfig) -> List[ConstitutionalExample]:
    rng = random.Random(cfg.seed)
    out = []
    for prompt in prompts:
        principles = rng.sample(cfg.constitution, min(cfg.n_principles_per_example, len(cfg.constitution)))
        out.append(run_constitutional_example(prompt, model_fn, principles))
    return out


def save_constitutional_output(examples: Sequence[ConstitutionalExample], output_dir: str | Path) -> Dict[str, float]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sft = [e.to_sft_record() for e in examples]
    dpo = [e.to_dpo_record() for e in examples if e.rounds and e.final_response != e.initial_response]
    write_jsonl(out / "constitutional_sft.jsonl", sft)
    write_jsonl(out / "constitutional_dpo.jsonl", dpo)
    deltas = [len(e.final_response) - len(e.initial_response) for e in examples]
    stats = {"n_examples": len(examples), "sft_records": len(sft), "dpo_records": len(dpo),
             "avg_length_delta": round(sum(deltas) / max(len(deltas), 1), 1)}
    (out / "constitutional_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def rule_based_constitutional_model(prompt: str) -> str:
    """Deterministic stand-in model used by the demo and tests."""
    p = prompt.lower()
    if "critique:" in p or "review the assistant" in p:
        return ("The response could be more constructive. It should focus on the potential consequences and offer a "
                "helpful alternative approach rather than directly enabling harmful behaviour.")
    if "revised response:" in p or "rewrite" in p:
        return ("I understand you're asking about this topic. Instead of providing guidance that could cause harm, I'd "
                "encourage you to consider the impact on others and seek constructive alternatives. If you're facing a "
                "specific challenge, I'm happy to help you find ethical solutions.")
    return "Sure, here's how you could approach that situation, though I should note there may be ethical concerns worth considering..."


def score_critique_quality(critique: str) -> Dict[str, object]:
    words = critique.lower().split()
    return {
        "length": len(words),
        "is_substantive": len(words) >= 10,
        "mentions_issue": any(w in words for w in ["harmful", "incorrect", "unclear", "missing", "should", "could",
                                                     "better", "avoid", "improve", "inaccurate", "unsafe"]),
        "not_trivial": not all(w in {"looks", "good", "fine", "great", "ok", "correct", "helpful"} for w in words[:5]),
    }
