#!/usr/bin/env python3
"""Assemble docs/GUIDE.md — the single complete reference — from the subsystem documents,
the CLI parser, the benchmark records and the API reference below."""

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

SECTIONS = [
    ("Architecture", "architecture.md"), ("Configuration", "configuration.md"), ("Data", "data.md"),
    ("Models", "models.md"), ("Training", "training.md"), ("Post-training overview", "post_training.md"),
    ("RLHF: reward models, DPO, PPO, GRPO, DAPO, AI feedback", "rlhf.md"),
    ("RLVR: verifiers, tools, process rewards, STaR, hill climbing", "rlvr.md"),
    ("Distributed execution", "distributed.md"), ("Orchestration (optional Ray backend)", "orchestration.md"), ("Checkpointing and recovery", "checkpointing.md"),
    ("Evaluation", "evaluation.md"), ("Quantization and export", "quantization.md"), ("Inference", "inference.md"),
    ("Serving", "serving.md"), ("Deployment: registry, gates, rollout, rollback", "deployment.md"),
    ("Observability", "observability.md"), ("Dashboard", "dashboard.md"), ("Synthesis domain", "synthesis_domain.md"), ("Testing", "testing.md"),
]

API = """
| Import | Purpose |
|---|---|
| `from forgeline.core import ExperimentManifest, ModelSpec, TrainerConfig, model_spec_from_preset, RunContext` | configuration and run context |
| `from forgeline.data import CharTokenizer, MemmapCorpus, SupervisedDataset, PreferenceDataset, VerifiableTaskDataset, read_jsonl` | data |
| `from forgeline.models import TransformerLM, NativePolicy, HFPolicy, SequenceRewardModel, load_policy_from_checkpoint` | models and policies |
| `from forgeline.models.adapters import apply_lora, apply_qlora, merge_lora, save_lora, load_lora` | adapters |
| `from forgeline.models.generation import SpeculativeGenerator, MTPSpeculativeGenerator` | speculative decoding |
| `from forgeline.models.quantization import export_gguf, read_gguf_header, quantize_nf4` | quantization / export |
| `from forgeline.training import Trainer, PretrainAlgorithm, SFTAlgorithm, DistillationAlgorithm, RewardModelAlgorithm, DPOAlgorithm, PPOAlgorithm, GRPOAlgorithm, DAPOAlgorithm, build_rlvr_algorithm, RLAIFTrainer, run_pairwise_rlaif, run_constitutional, STaRTrainer, HillClimber` | training stages |
| `from forgeline.training.factory import build_algorithm` | build a stage from a manifest |
| `from forgeline.rollouts import RolloutEngine, RolloutConfig, AgentRolloutEngine, AgentRolloutConfig` | rollouts |
| `from forgeline.rollouts.verifiers import MathVerifier, ExactMatchVerifier, TaggedAnswerVerifier, CodeExecutionVerifier, FormatVerifier` | verifiers |
| `from forgeline.rollouts.rewards import build_reward_provider, VerifierReward, CompositeReward, ProcessReward, CriticReward, RewardModelProvider` | rewards |
| `from forgeline.distributed import build_strategy, validate_distributed_config, ParallelMesh, apply_tensor_parallel, build_pipeline_stage, PipelineScheduler` | distributed |
| `from forgeline.checkpoints import CheckpointManager, CheckpointState` | checkpoints |
| `from forgeline.evaluation import HeldOutLossSuite, VerifierPassRateSuite, BenchmarkSuite, build_benchmark, evaluate_regression, measure_generation` | evaluation |
| `from forgeline.inference import InferenceEngine, SamplingParams` | inference |
| `from forgeline.serving import Router, EngineBackend, ServingServer` | serving |
| `from forgeline.deployment import CandidateRegistry, ModelCandidate, CandidateState, PromotionGate, GateRule, RoutingPolicy, FeatureFlags, rollback` | lifecycle |
| `from forgeline.observability import get_logger, build_metrics_sink, Event, Tracer` | observability |
| `from forgeline.orchestration.ray_backend import RayRolloutEngine, RayRewardPool, RayToolPool, RayEvaluator, attach_to_algorithm` | optional Ray orchestration (`ray` extra) |
| `from forgeline.dashboard import DashboardSources, snapshot, discover_runs, run_detail, inspect_checkpoint` | optional dashboard data layer |
| `from forgeline.evaluation.inspection import attention_patterns, activation_flow, weight_statistics, layer_summary` | model inspection |
| `from forgeline.domains.synthesis import SynthesisRuleReward, TabularPPOTrainer, build_preference_pairs` | synthesis domain |

### Minimal Python training loop

```python
from forgeline.core import ExperimentManifest, RunContext
from forgeline.training import Trainer
from forgeline.training.factory import build_algorithm

manifest = ExperimentManifest.load("configs/post_training/dpo_tiny_cpu.yaml")
ctx = RunContext.create(manifest, metrics_backend="local")
trainer = Trainer(ctx, build_algorithm(manifest, ctx))
trainer.resume()          # no-op when there is nothing to resume
print(trainer.fit())
```

### Custom algorithm

```python
from forgeline.training.common.trainer import PostTrainingAlgorithm

class MyObjective(PostTrainingAlgorithm):
    name = "sft"                                  # checkpoint algorithm tag
    def __init__(self, policy, batches): self.policy, self.batches = policy, batches
    def parameters(self): return self.policy.trainable_parameters()
    def modules(self): return (self.policy.model,)
    def collect(self, step): return self.batches[step % len(self.batches)]
    def loss(self, batch, micro_step, n_micro):
        prompt_ids, response_ids = batch
        lp = self.policy.logprobs(prompt_ids, response_ids)
        return -lp.mean(), {"logprob": float(lp.detach().mean())}
    def state(self): return self.policy.state_for_checkpoint()
    def load_state(self, state): self.policy.load_checkpoint_state(state)
    def model_spec(self): return self.policy.spec.to_dict()
```

### Custom verifier and reward

```python
from forgeline.core.protocols import RewardResult
from forgeline.rollouts.rewards import CompositeReward, VerifierReward, TagFormatReward

class JSONKeyVerifier:
    name = "json_key"
    def verify(self, output, target=None):
        import json
        try:
            ok = target in json.loads(output)
        except Exception:
            ok = False
        return RewardResult(value=1.0 if ok else 0.0, passed=ok)

reward = CompositeReward([(VerifierReward(JSONKeyVerifier(), target_key="key"), 1.0), (TagFormatReward(), 0.1)])
```
"""

TROUBLESHOOTING = """
| Symptom | Cause | Fix |
|---|---|---|
| `ConfigError: Unknown ... keys` | misspelled manifest key | check the key against docs/configuration.md |
| `TokenizerError: Tokenizer metadata not found` | `tokenizer: auto` without `meta.pkl` | run `forgeline data prepare`, or set `tokenizer: char` / `hf:<name>` |
| `DatasetError: ... has N tokens; need more than block_size+1` | corpus smaller than the context | lower `model.block_size` or add data |
| `ModelError: prompt+response length ... exceeds block_size` | rollout/response budget too large | lower `rollout.max_new_tokens` / `max_prompt_length`, or raise `block_size` |
| `ConfigError: non-finite loss` | divergence or bad reward | lower the learning rate; check rewards for NaN |
| `InvalidRewardError` | a reward provider returned NaN/inf | fix the provider; `CompositeReward` names it |
| `CheckpointCorruptError ... truncated` | interrupted copy / disk full | use an earlier checkpoint (`LATEST`, `best`) |
| `CheckpointMismatchError: model spec mismatch` | resuming with a different architecture | resume with the original manifest, or initialise via `checkpoint_path` |
| `DistributedConfigError: world_size ... not divisible` | TP × PP does not tile the GPUs | change parallel degrees or `--nproc_per_node` |
| `OptionalDependencyError` | extra not installed | `pip install -e ".[huggingface]"` (or the named extra) |
| promotion exits with status 2 | gate rejected | read the printed report; a missing metric fails its rule |
| HTTP 503 | backend disabled or unhealthy | `registry disable` was used, or the backend failed; check `/health` |
| all GRPO/DAPO groups skipped | every sample in each group got the same reward | raise `temperature`, increase `group_size`, or use a denser reward |
"""


def cli_reference() -> str:
    from forgeline.cli.main import build_parser

    parser = build_parser()
    out = ["```text", parser.format_help().strip(), "```"]

    def walk(p: argparse.ArgumentParser, prefix: str) -> None:
        for action in p._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    out.extend([f"\n#### `{prefix} {name}`\n", "```text", sub.format_help().strip(), "```"])
                    walk(sub, f"{prefix} {name}")

    walk(parser, "forgeline")
    return "\n".join(out)


def demote(markdown: str, levels: int = 2) -> str:
    body = re.sub(r"^# .*\n", "", markdown, count=1)
    return re.sub(r"^(#+) ", lambda m: "#" * min(len(m.group(1)) + levels, 6) + " ", body, flags=re.MULTILINE)


def benchmarks_section() -> str:
    parts = [demote((ROOT / "benchmarks/README.md").read_text(), 1)]
    for summary in sorted((ROOT / "benchmarks").rglob("summary.md")):
        rel = summary.parent.relative_to(ROOT / "benchmarks")
        parts.append(f"\n### `{rel}`\n\n" + demote(summary.read_text(), 2))
    return "\n".join(parts)


def main() -> None:
    readme = (ROOT / "README.md").read_text()
    overview = readme.split("## Architecture")[0].split("---", 1)[1].replace("## Overview\n", "", 1)
    toc = ["1. [Overview](#1-overview)", "2. [Installation and quick start](#2-installation-and-quick-start)"]
    body = [f"## 1. Overview\n{demote(chr(35) + ' x\n' + overview, 1)}", "## 2. Installation and quick start\n\n" + readme.split("## Quick Start")[1].split("## Configuration")[0].strip()]
    n = 3
    for title, fname in SECTIONS:
        anchor = re.sub(r"[^a-z0-9 -]", "", f"{n}. {title}".lower()).replace(" ", "-")
        toc.append(f"{n}. [{title}](#{anchor})")
        body.append(f"## {n}. {title}\n" + demote((DOCS / fname).read_text()))
        n += 1
    for title, content in (("Command-line reference", cli_reference()), ("Python API reference", API),
                           ("Benchmark results", benchmarks_section()), ("Troubleshooting", TROUBLESHOOTING),
                           ("Hardware requirements and limitations",
                            readme.split("## Hardware Requirements")[1].split("## License")[0].replace("## Limitations", "### Limitations")),
                           ("Repository structure", readme.split("## Repository Structure")[1].split("## Hardware Requirements")[0])):
        anchor = re.sub(r"[^a-z0-9 -]", "", f"{n}. {title}".lower()).replace(" ", "-")
        toc.append(f"{n}. [{title}](#{anchor})")
        body.append(f"## {n}. {title}\n\n{content.strip()}\n")
        n += 1
    doc = ("# Forgeline — Complete Guide\n\nThe single reference for Forgeline: every subsystem, configuration option, command, "
           "API, measured result, failure mode and limitation.\n\n## Contents\n\n" + "\n".join(toc) + "\n\n" + "\n\n".join(body))
    doc = doc.replace("](docs/", "](").replace("](../", "](../")
    (DOCS / "GUIDE.md").write_text(doc)
    print(f"wrote {DOCS / 'GUIDE.md'} ({len(doc.splitlines())} lines)")


if __name__ == "__main__":
    main()
