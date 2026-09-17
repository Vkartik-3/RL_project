# RLVR: verifiable rewards, tools, process rewards and self-improvement

## What

Reinforcement learning where rewards come from deterministic checks — answer extraction, exact match, unit-test execution, output format, tool-use outcomes and step-level verification — instead of a learned model.

## Why

Verifiable rewards cannot be reward-hacked the way learned rewards can, and they need no labelled preferences. Separating verification from generation lets any verifier drive any RL method.

## How

### Verifiers (`rollouts/verifiers`)

| Verifier | Target | Scoring |
|---|---|---|
| `math` | answer string | extracts `\boxed{}`, `#### x`, "the answer is", or last number; exact or numeric match (tolerance) → 1.0; within 1% relative error → 0.5 (partial credit); else 0.0 |
| `exact_match` | string | whitespace/case-normalised equality |
| `tagged_answer` | answer | `<final_answer>` numeric-normalised match → 1.0; tool call and tool result present → 0.1; else 0.0 |
| `code` | unit-test code | extracts fenced/indented code, appends tests, runs in a subprocess with timeout: pass 1.0, assertion failure 0.0, crash/timeout −0.1 |
| `format` | `cot` / `steps` / `json` | `<think>` structure, step count, JSON validity (graded) |

Verifier exceptions never propagate: they return `passed=False` with `info.error`.

`tag_format_reward` adds 0.02 per `<think>`, `<tool_call>`, `<final_answer>` tag present.

### Tools and agent rollouts

* `execute_python(code, timeout)` runs code in a fresh interpreter process (`-I` isolated mode), capturing stdout/stderr, with a hard timeout and 500-character output cap.
* `parse_tagged_output` extracts thinking, the JSON tool call, tool results and the final answer.
* `AgentRolloutEngine.run_episode(prompt)`: generate → if `<final_answer>` stop; if `<tool_call>` for a registered tool, execute it and inject `<tool_result>…</tool_result>` into the context → continue, up to `max_turns`. Tool errors are injected as `ERROR: …` observations.
* Trajectory log-prob for GRPO: each model segment's mean token log-prob is computed conditioned on the full context so far (including earlier tool results), then averaged across segments. Tool-result tokens are observations and receive no gradient.

### Process rewards (`rollouts/rewards/process.py`)

1. Split the response into steps (blank lines, `Step N:`, headings, transition words; short fragments merge).
2. Score each step: executable fenced code with output 0.15 (0.05 if it fails); an assignment whose arithmetic matches its stated result 0.15; any evaluable arithmetic 0.10; otherwise 0.
3. `total = step_weight · Σ_t γ^(T−1−t)·s_t + tagged_answer_reward`.

Arithmetic evaluation only accepts digits, operators and parentheses, with builtins disabled.

### Critic reward

`heuristic_critic_score` gives up to 0.25 each for a final answer, reasoning keywords, a tool call and non-repetitive text; `CriticReward(mode="llm", critic_fn=…)` parses a 0–1 score from any model. Combine with verifiable rewards: `CompositeReward([(verifier, 1.0), (CriticReward(), α)])`.

### RLVR builder

```python
build_rlvr_algorithm(policy, tasks, RLVRConfig(task_type="math", optimizer="dapo", require_format=True,
                                                format_spec="cot", correctness_weight=0.7, format_weight=0.3))
```

`evaluate_pass_rate(policy, tasks, reward)` decodes greedily and reports pass rate, mean reward and malformed-output rate.

### STaR

For each round: sample `samples_per_problem` rationales ending in `<answer>N</answer>`; keep correct ones; SFT on them (mean token negative log-likelihood); evaluate greedy accuracy; save the best policy; results in `star_results.json`.

### Hill climbing

Round 0 evaluates the policy. Each round: run tool-agent episodes per training task, keep up to `top_k_per_problem` trajectories with reward ≥ `reward_threshold`, append them to the dataset, run agent-GRPO steps, evaluate accuracy / tool-use rate / reward and save improvements.

## Configuration

```yaml
algorithm: rlvr
data: {kind: verifiable, path: data/samples/verifiable/arithmetic_tasks.jsonl}
algorithm_params:
  task_type: math            # math | code | tagged_answer | exact_match
  optimizer: dapo            # dapo | grpo
  require_format: true
  format_spec: cot
  dapo: {group_size: 4, prompts_per_step: 2}
```

Agent GRPO: `configs/post_training/grpo_agent_tools_tiny_cpu.yaml` (`reward: {type: tagged}`, `algorithm_params.agent: true`). Process-reward GRPO: `configs/post_training/grpo_process_reward_tiny_cpu.yaml`.

## Failure modes

| Condition | Behaviour |
|---|---|
| verifier raises | failed result with error info |
| tool raises / times out / not registered | error text injected as observation / episode ends |
| malformed tool-call JSON | treated as no tool call |
| empty generation | `RolloutError` |
| task without answer or tests | `MalformedRecordError` at load |

## Local validation

`tests/unit/test_rollouts_rewards.py` (every verifier, executor timeout, parser, composite weighting, process-reward scores), `tests/integration/test_training_lifecycle.py::test_agent_rollout_executes_tool_calls` (tool result is injected and conditions the next turn), `tests/failure/test_failure_modes.py::test_tool_execution_failure_is_injected`, `examples/03_agent_tool_rollout.py`.

## Hardware requirements

CPU. Code execution uses one subprocess per call.

## Limitations

* The Python executor isolates by process and timeout only; it is not a security sandbox. Run untrusted-model code inside a container or VM.
* Step splitting is heuristic; unusual formatting reduces process-reward coverage.
