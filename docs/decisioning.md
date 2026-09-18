# Sequential decisioning: budgeted allocation

## What

A general finite-horizon, budget-constrained sequential allocation environment with policy-dependent dynamics, four
policy families (static heuristic, online primal-dual pacer, stateless PPO, sequence-conditioned PPO), logged
trajectories with valid propensities, offline policy evaluation (IPS, SNIPS, PDIS, DR), a hindsight oracle, pacing
metrics, simulated A/B experiments with guardrails, shadow evaluation, and integration with the candidate registry and
promotion gates. Package: `forgeline.domains.allocation`.

## Why

Language-model post-training is one kind of sequential decision problem. This subsystem gives the same trainer,
checkpoints, metrics and lifecycle a second, fully observable-by-construction task family where decisions are made
under uncertainty about future opportunities, a hard resource constraint binds, and past actions change the future —
the setting of resource allocation, bidding, scheduling, exposure control or compute pacing. It is a **simulator**;
no real workload, traffic or budget data is involved.

## MDP

| Element | Definition |
|---|---|
| Horizon | `T` decisions (default 48); fixed length, no early termination (a depleted budget forces zero spend) |
| Budget | `B` (default 24 = half of `T·c̄`); hard: `x_t = min(m_t·ĉ_t, B_t)`, a clipped request is counted as a *violation* |
| Opportunity `o_t = (v_t, c_t, q_t, s_t)` | `s_t = sin(2π(t+phase)/T)`; `v_t = v̄·exp(σ_v ε − σ_v²/2)(1 + A_v s_t)`; `c_t = c̄·exp(σ_c ε' − σ_c²/2)(1 + A_c s_t)`; `q_t ~ Beta(α, β)` — stochastic, seasonally non-stationary, exogenous |
| Action | `a_t ∈ {abstain, low, medium, high}` → multiplier `m_t ∈ {0, 0.5, 1, 1.5}`, chosen before `o_{t+1}` is revealed |
| Latent state | pressure `p_t ≥ 0`, `p_0 = 0` |
| Effective cost | `ĉ_t = c_t (1 + κ p_t)` |
| Intensity | `u_t = x_t / ĉ_t` (what was actually afforded) |
| Response | `ρ_t = q_t · (1 − e^{−u_t}) · max(0, 1 − φ p_t)`; outcome `y_t ~ Bernoulli(ρ_t)` |
| Reward | `r_t = y_t · v_t` (value realised); cost `x_t` |
| Transitions | `B_{t+1} = B_t − x_t`; `p_{t+1} = δ p_t + η u_t` |
| Observation (14-d) | remaining budget fraction, elapsed and remaining horizon fractions, `v_t/v̄−1`, `c_t/c̄−1`, `q_t`, `s_t`, pacing error `cum_spend/B − t/T`, cumulative value, last multiplier / reward / spend, mean relative value and cost of the last `k` opportunities. **Pressure is not observed** → POMDP |
| Return | undiscounted sum of rewards (γ = 1 in PPO by default) |
| Defaults | `δ = 0.85, η = 0.25, κ = 0.6, φ = 0.35, σ_v = 0.5, σ_c = 0.3, A_v = 0.4, A_c = −0.2, α = β = 2` (`configs/allocation/env_default.yaml`) |

Policy-dependent dynamics: allocating intensity `u_t` today raises pressure, which multiplies every later effective
cost by `(1 + κ p)` and every later response rate by `(1 − φ p)`. Under the same seed, an aggressive prefix and a
cautious prefix lead to different costs and success probabilities for the identical later opportunities
(`tests/unit/test_allocation_env.py::test_endogenous_feedback_changes_future_state_under_same_seed`); setting
`pressure_gain: 0` makes the dynamics exogenous (tested).

## Baselines

* **Static heuristic** `ThresholdPacingPolicy(threshold, level)` — allocate `level` when `v_t q_t / c_t ≥ threshold` and
  cumulative spend ≤ pro-rata budget (+5% slack); otherwise abstain.
* **Online primal-dual pacer** `DualPacingPolicy(learning_rate α, initial_price λ₀)` — per step chooses
  `argmax_m [v_t q_t (1 − e^{−m}) − λ m c_t]`, then `λ ← max(0, λ + α (x_t − B/T))`: dual subgradient descent on the
  budget-rate constraint. Budget pressure enters through λ: overspending raises the price and suppresses allocation.
* **Stateless PPO** — `MLPActorCritic` (14 → 64 → 64 → {4 logits, value}) on the current observation.
* **Sequence-conditioned PPO** — `GRUActorCritic`: a GRU (hidden 64) over the last `W = 8` steps of
  `[obs_j, onehot(a_j), r_j/v̄, x_j/c̄, B_j/B]`, left-padded and masked, whose final state is concatenated with a
  projection of the current observation before the actor/critic heads. It genuinely consumes the history window
  (`test_gru_policy_consumes_history_window`: identical current observation, different histories → different action
  distributions).

PPO for both runs through the shared `Trainer` via `AllocationPPOAlgorithm` (`domains/allocation/ppo.py`): GAE
(`compute_gae`), normalised advantages, `ppo_clipped_objective`, value MSE, entropy bonus; PPO epochs are trainer steps
that reuse the same rollout with frozen old log-probs; `gradient_accumulation_steps` = minibatches. Manifest algorithm
`allocation_ppo` (`configs/allocation/ppo_mlp.yaml`, `ppo_gru.yaml`, `ppo_tiny_cpu.yaml`).

## Logged trajectories

`rollout.py::LoggedStep` (schema version 1): `episode_id, seed, t, obs, history_summary, action, propensity,
action_probs, reward, cost, next_obs, cum_spend, cum_reward, remaining_budget, terminal, clipped, pressure` (pressure is
logged for analysis only). `write_episodes`/`read_episodes` write JSONL and validate propensities (`0 < p ≤ 1`,
`p == action_probs[action]`, proper distributions, terminated episodes). Behaviour data for OPE is produced by an
`EpsilonMixPolicy` (`(1−ε)·base + ε·uniform`) so every action has propensity ≥ ε/4.

## Offline policy evaluation (`ope.py`)

Trajectory IS (IPS), self-normalised IS (SNIPS), per-decision IS (PDIS) and per-decision doubly robust (DR, Q̂ from
ridge regression of reward-to-go on `[obs, onehot(a)]`), each with percentile bootstrap CIs over episodes, plus clipped
variants (`max_weight`). Diagnostics: ESS, max/mean weight, unsupported fraction (target mass where the behaviour policy
had none → `OPEError`), zero-target fraction, clipped fraction, overlap. Non-finite weights raise. Formulas are checked
against hand-computed values in `tests/unit/test_allocation_ope.py`.

## Oracle and regret

`oracle.py::hindsight_upper_bound` solves the expected-value fractional knapsack over the whole revealed opportunity
stream ignoring pressure (KKT: `u_t = clip(ln(v_t q_t/(λ c_t)), 0, u_max)`, λ by bisection). It is an **upper bound,
not a policy**; regret is reported against it and against each baseline.

## Experiments (`experiment.py`)

`run_ab_experiment`: episodes assigned to arms by `stable_bucket(salt:episode_seed)` (the routing hash), per-arm
metrics, bootstrap CI of every metric difference, permutation test on value, minimum-sample warning, guardrails
(primary CI, utilisation bounds, violations, early exhaustion, custom). `ABResult.gate_metrics()` feeds
`PromotionGate` (`configs/allocation/promotion_gate.yaml`). `shadow_evaluate`: the candidate proposes at every step of
the incumbent's episodes without acting; reports divergence rate (overall and by horizon third), an action confusion
matrix, the candidate's replay value on the same seeds and its OPE estimate from the incumbent's log.

## Results (`benchmarks/decisioning/budgeted_allocation`)

300 held-out episodes, 5 training seeds, laptop CPU, 3 minutes total:

| Policy | Value | Utilisation | Pacing error | Early exhaustion | Regret vs oracle (13.22) |
|---|---|---|---|---|---|
| threshold heuristic | 7.29 | 0.70 | 0.178 | 0.00 | 5.93 |
| dual pacer | **7.71** | 0.92 | 0.082 | 0.00 | 5.51 |
| stateless PPO | 7.40 ± 0.10 | 0.87 | 0.153 | 0.05 | 5.82 |
| sequence PPO (GRU) | 7.60 ± 0.12 | 0.95 | 0.091 | 0.20 | 5.62 |

History helps (+0.19 over stateless PPO, consistent across seeds), but the online dual pacer remains the best policy
at this training budget; the simulated A/B (dual vs best GRU, 400 episodes) measured Δ = +0.57 with CI [−0.04, +1.16],
p = 0.084, and more clipped-budget attempts, so the gate rejected the challenger. OPE: DR estimates lie within
0.1–0.9 of simulator truth for all targets; IPS collapses for targets far from the behaviour policy (ESS ≈ 0).
Throughput: ~50k env steps/s, ~300 OPE trajectory-evaluations/s, ~500 A/B episodes/s.

## Commands

```bash
forgeline allocation benchmark --config configs/allocation/benchmark.yaml        # full benchmark (~3 min)
forgeline allocation benchmark --config configs/allocation/benchmark_tiny_cpu.yaml
forgeline train configs/allocation/ppo_gru.yaml                                   # one policy through the trainer
forgeline allocation ope --log runs/allocation-benchmark/logged_trajectories.jsonl --target <checkpoint>|dual|threshold
forgeline allocation ab --incumbent dual --challenger <checkpoint> --episodes 400 --gate-metrics gate.json
forgeline registry register --name allocator --version v2 --algorithm allocation_ppo --checkpoint <checkpoint> --metrics "$(cat gate.json)"
forgeline registry promote --id <id> --to champion --gate configs/allocation/promotion_gate.yaml
forgeline allocation shadow --incumbent dual --candidate <checkpoint>
```

## Failure modes

`ConfigError` for invalid env/policy/PPO/A/B configuration and invalid actions; `DatasetError` for malformed logs or
propensities; `OPEError` for unsupported targets, zero propensities, non-finite weights; deterministic incumbents make
OPE impossible in shadow mode and the report says so.

## Local validation

`tests/unit/test_allocation_env.py` (10), `tests/unit/test_allocation_ope.py` (7), `tests/integration/test_allocation_pipeline.py` (13).

## Limitations

* Simulator only; opportunity and response models are synthetic and the pressure feedback is a stylised monotone effect.
* PPO was trained for a fixed small budget (9,600 episodes per policy per seed); the ordering against the dual pacer may change with more training or tuning, which was deliberately not done.
* Trajectory-level importance sampling is unusable at horizon 48 for dissimilar policies; rely on DR/PDIS and the diagnostics.
* Multi-seed evaluation uses common seeds across policies; results are means over 300 episodes, not a statistical proof of superiority.
