"""Episode runner, logged-trajectory schema and pacing/budget metrics for the allocation environment."""

from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from forgeline.core.errors import ConfigError, DatasetError
from forgeline.domains.allocation.env import OBS_NAMES, AllocationEnvConfig, VectorEnv, episode_seed
from forgeline.domains.allocation.policies import NeuralPolicy, Policy, StepContext

SCHEMA_VERSION = 1


@dataclass
class LoggedStep:
    """One logged decision. ``propensity`` is the behaviour policy's probability of the taken action; ``action_probs``
    the full behaviour distribution (needed to evaluate target policies that put mass elsewhere)."""

    episode_id: str
    seed: int
    t: int
    obs: List[float]
    history_summary: Dict[str, float]
    action: int
    propensity: float
    action_probs: List[float]
    reward: float
    cost: float
    next_obs: List[float]
    cum_spend: float
    cum_reward: float
    remaining_budget: float
    terminal: bool
    clipped: bool
    pressure: float  # latent state, logged for analysis only (never given to policies)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Episode:
    episode_id: str
    seed: int
    steps: List[LoggedStep]

    @property
    def total_reward(self) -> float:
        return float(sum(s.reward for s in self.steps))

    @property
    def total_spend(self) -> float:
        return float(sum(s.cost for s in self.steps))


def write_episodes(episodes: Iterable[Episode], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ep in episodes:
            for s in ep.steps:
                rec = s.to_dict()
                rec["schema_version"] = SCHEMA_VERSION
                f.write(json.dumps(rec) + "\n")
    return path


def read_episodes(path: str | Path) -> List[Episode]:
    """Read a JSONL log written by :func:`write_episodes`; validates the schema and propensities."""
    by_ep: Dict[str, Episode] = {}
    order: List[str] = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{n}: invalid JSON ({exc})") from None
        if d.get("schema_version") != SCHEMA_VERSION:
            raise DatasetError(f"{path}:{n}: unsupported schema_version {d.get('schema_version')!r}")
        d.pop("schema_version")
        try:
            step = LoggedStep(**d)
        except TypeError as exc:
            raise DatasetError(f"{path}:{n}: malformed logged step ({exc})") from None
        validate_step(step, f"{path}:{n}")
        if step.episode_id not in by_ep:
            by_ep[step.episode_id] = Episode(step.episode_id, step.seed, [])
            order.append(step.episode_id)
        by_ep[step.episode_id].steps.append(step)
    eps = [by_ep[k] for k in order]
    for ep in eps:
        ep.steps.sort(key=lambda s: s.t)
        if [s.t for s in ep.steps] != list(range(len(ep.steps))):
            raise DatasetError(f"episode {ep.episode_id} has missing or duplicate timesteps")
        if not ep.steps[-1].terminal:
            raise DatasetError(f"episode {ep.episode_id} is not terminated")
    return eps


def validate_step(step: LoggedStep, where: str = "") -> None:
    p = step.propensity
    if not (isinstance(p, (int, float)) and math.isfinite(p)) or p <= 0.0 or p > 1.0 + 1e-9:
        raise DatasetError(f"{where}: propensity {p!r} is not in (0, 1]")
    probs = np.asarray(step.action_probs, dtype=float)
    if probs.ndim != 1 or not np.all(np.isfinite(probs)) or abs(probs.sum() - 1.0) > 1e-6 or (probs < 0).any():
        raise DatasetError(f"{where}: action_probs must be a finite probability vector")
    if not 0 <= step.action < len(probs) or abs(probs[step.action] - p) > 1e-6:
        raise DatasetError(f"{where}: propensity {p} does not match action_probs[{step.action}]={probs[step.action] if 0 <= step.action < len(probs) else 'n/a'}")
    if not math.isfinite(step.reward) or not math.isfinite(step.cost) or step.cost < -1e-12:
        raise DatasetError(f"{where}: non-finite reward/cost or negative cost")


# ─────────────────────────────────────────────────────────────────────────────
#  Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def _contexts(venv: VectorEnv) -> List[StepContext]:
    out = []
    for e in venv.envs:
        o = e.opportunities[e.t] if not e.done else e.opportunities[-1]
        out.append(StepContext(e.t, e.cfg.horizon, e.remaining, e.cfg.budget, e.cum_spend, o.value, o.cost, o.quality))
    return out


def run_episodes(policy: Policy, env_cfg: AllocationEnvConfig, seeds: Sequence[int], *, greedy: bool = False,
                 rng_seed: int = 0, episode_prefix: str = "", batch: int = 64) -> List[Episode]:
    """Run ``policy`` on one episode per seed (in lockstep batches) and return fully logged episodes."""
    rng = np.random.default_rng(rng_seed)
    episodes: List[Episode] = []
    seeds = [int(s) for s in seeds]
    for start in range(0, len(seeds), batch):
        chunk = seeds[start:start + batch]
        venv = VectorEnv(env_cfg, len(chunk))
        obs = venv.reset(chunk)
        policy.start(len(chunk))
        logs: List[List[LoggedStep]] = [[] for _ in chunk]
        while not venv.done:
            ctx = _contexts(venv)
            actions, probs = policy.act(obs, ctx, rng, greedy=greedy)
            next_obs, rewards, dones, infos = venv.step(actions)
            spends = np.array([i.spend for i in infos])
            remaining = np.array([i.remaining_after for i in infos])
            for j, info in enumerate(infos):
                e = venv.envs[j]
                logs[j].append(LoggedStep(
                    episode_id=f"{episode_prefix}{chunk[j]}", seed=chunk[j], t=info.t, obs=[float(x) for x in obs[j]],
                    history_summary={"cum_spend_frac": float(info.remaining_before and (env_cfg.budget - info.remaining_before) / env_cfg.budget),
                                     "elapsed_frac": info.t / env_cfg.horizon},
                    action=int(actions[j]), propensity=float(probs[j, actions[j]]), action_probs=[float(x) for x in probs[j]],
                    reward=float(info.reward), cost=float(info.spend), next_obs=[float(x) for x in next_obs[j]],
                    cum_spend=float(e.cum_spend), cum_reward=float(e.cum_value), remaining_budget=float(info.remaining_after),
                    terminal=bool(dones[j]), clipped=bool(info.clipped), pressure=float(info.pressure_after)))
            policy.observe(actions, rewards, spends, remaining)
            obs = next_obs
        episodes.extend(Episode(f"{episode_prefix}{s}", s, logs[j]) for j, s in enumerate(chunk))
    return episodes


def seeds_for(base_seed: int, n_episodes: int, offset: int = 0) -> List[int]:
    return [episode_seed(base_seed, offset + i) for i in range(n_episodes)]


# ─────────────────────────────────────────────────────────────────────────────
#  Pacing / budget metrics
# ─────────────────────────────────────────────────────────────────────────────

def episode_metrics(ep: Episode, env_cfg: AllocationEnvConfig, exhaustion_frac: float = 0.9) -> Dict[str, float]:
    T, B = env_cfg.horizon, env_cfg.budget
    spend = np.array([s.cost for s in ep.steps]); reward = np.array([s.reward for s in ep.steps])
    cum = np.cumsum(spend) / B
    ideal = (np.arange(1, T + 1)) / T
    exhausted_at = next((i for i, c in enumerate(cum) if c >= 1.0 - 1e-9), None)
    thirds = np.array_split(np.arange(T), 3)
    actions = np.array([s.action for s in ep.steps])
    dist = np.bincount(actions, minlength=env_cfg.n_actions) / T
    return {
        "value": float(reward.sum()), "spend": float(spend.sum()), "utilization": float(spend.sum() / B),
        "value_per_budget": float(reward.sum() / B), "pacing_error": float(np.mean(np.abs(cum - ideal))),
        "early_exhaustion": float(exhausted_at is not None and exhausted_at < exhaustion_frac * T - 1),
        "unused_budget_frac": float(max(0.0, 1.0 - spend.sum() / B)),
        "violations": float(sum(s.clipped for s in ep.steps)),
        "reward_variance": float(reward.var()),
        "value_early": float(reward[thirds[0]].sum()), "value_mid": float(reward[thirds[1]].sum()), "value_late": float(reward[thirds[2]].sum()),
        "spend_early": float(spend[thirds[0]].sum()), "spend_mid": float(spend[thirds[1]].sum()), "spend_late": float(spend[thirds[2]].sum()),
        **{f"action_frac_{i}": float(dist[i]) for i in range(env_cfg.n_actions)},
        "final_pressure": float(ep.steps[-1].pressure),
    }


def aggregate_metrics(episodes: Sequence[Episode], env_cfg: AllocationEnvConfig) -> Dict[str, float]:
    rows = [episode_metrics(e, env_cfg) for e in episodes]
    keys = rows[0].keys()
    out: Dict[str, float] = {"n_episodes": float(len(rows))}
    for k in keys:
        v = np.array([r[k] for r in rows])
        out[k] = float(v.mean())
        out[f"{k}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    out["value_ci95_halfwidth"] = float(1.96 * out["value_std"] / math.sqrt(len(rows))) if len(rows) > 1 else 0.0
    return out
