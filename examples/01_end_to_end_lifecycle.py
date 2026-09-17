"""End-to-end lifecycle on CPU in one script:

corpus → pretrain → SFT → DPO → evaluation → candidate registry → promotion gate
→ champion/challenger routing → serving (in-process) → rollback.

Run:  python examples/01_end_to_end_lifecycle.py
"""

import json
import tempfile
from pathlib import Path

import torch

from forgeline.core.config import CheckpointConfig, ExperimentManifest, OptimizerConfig, TrainerConfig, model_spec_from_preset
from forgeline.core.lifecycle import RunContext
from forgeline.data import CharTokenizer, MemmapCorpus, PreferenceDataset, PreferenceRecord, SFTRecord, SupervisedDataset
from forgeline.data.preprocessing import prepare_text_corpus
from forgeline.data.records import VerifiableTask
from forgeline.deployment import CandidateRegistry, CandidateState, GateRule, ModelCandidate, PromotionGate, RoutingPolicy, rollback
from forgeline.evaluation import VerifierPassRateSuite
from forgeline.inference import InferenceEngine, SamplingParams
from forgeline.models import NativePolicy, TransformerLM
from forgeline.rollouts.rewards import VerifierReward
from forgeline.rollouts.verifiers import MathVerifier
from forgeline.serving import EngineBackend, Router
from forgeline.training import DPOAlgorithm, DPOConfig, PretrainAlgorithm, PretrainConfig, SFTAlgorithm, SFTConfig, Trainer

ROOT = Path(__file__).resolve().parents[1]
work = Path(tempfile.mkdtemp(prefix="forgeline-"))


def context(name, algorithm, spec, steps):
    m = ExperimentManifest(name=name, algorithm=algorithm, model=spec, output_dir=str(work / name),
                           trainer=TrainerConfig(max_steps=steps, batch_size=8, eval_every=0, log_every=25, device="cpu",
                                                 optimizer=OptimizerConfig(learning_rate=2e-3), checkpoint=CheckpointConfig(save_every=0)))
    return RunContext.create(m, metrics_backend="local")


# 1. data + pretraining
info = prepare_text_corpus(ROOT / "data/samples/text/tiny_corpus.txt", work / "data")
tok = CharTokenizer.from_metadata(__import__("pickle").load(open(work / "data/meta.pkl", "rb")))
spec = model_spec_from_preset("tiny", vocab_size=tok.vocab_size, block_size=64)
model = TransformerLM(spec)
pre = PretrainAlgorithm(model, MemmapCorpus(work / "data", "train", 64), PretrainConfig(batch_size=8), MemmapCorpus(work / "data", "val", 64))
Trainer(context("pretrain", "pretrain", spec, 150), pre).fit()
print("pretrain val loss", round(pre.evaluate(0)["val_loss"], 3))

# 2. SFT on arithmetic
policy = NativePolicy(model, tok)
sft_records = [SFTRecord(r["prompt"], r["response"]) for r in map(json.loads, open(ROOT / "data/samples/supervised/arithmetic_sft.jsonl"))]
Trainer(context("sft", "sft", spec, 100), SFTAlgorithm(policy, SupervisedDataset(sft_records, tok, 48), SFTConfig(batch_size=8))).fit()

# 3. DPO
prefs = [PreferenceRecord(r["prompt"], r["chosen"], r["rejected"]) for r in map(json.loads, open(ROOT / "data/samples/preference/arithmetic_pairs.jsonl"))]
dpo = DPOAlgorithm(policy, PreferenceDataset(prefs, tok, 32, 8), DPOConfig(batch_size=4, beta=0.1))
Trainer(context("dpo", "dpo", spec, 40), dpo).fit()
print("dpo preference accuracy", dpo.evaluate(0)["preference_accuracy"])

# 4. evaluation
tasks = [VerifiableTask(prompt=f"Q: What is {a} + {b}?\nA:", answer=str(a + b)) for a, b in [(1, 2), (3, 4), (5, 5), (7, 1)]]
result = VerifierPassRateSuite(tasks, VerifierReward(MathVerifier()), max_new_tokens=4).run(policy)
print("pass rate", result.metrics)

# 5. registry + promotion gate + routing
registry = CandidateRegistry(work / "registry.json")
v1 = registry.register(ModelCandidate("assistant", "v1", "sft", str(work / "sft"), metrics={"pass_rate": 0.0, "latency_ms": 10.0}))
registry.transition(v1.id, CandidateState.CHALLENGER); registry.transition(v1.id, CandidateState.CHAMPION)
v2 = registry.register(ModelCandidate("assistant", "v2", "dpo", str(work / "dpo"), metrics={**result.metrics, "latency_ms": 11.0}))
gate = PromotionGate([GateRule("pass_rate", min_delta=0.0), GateRule("latency_ms", max_increase=0.5, relative=True)])
report = gate.evaluate(registry.get(v2.id).metrics, registry.get(v1.id).metrics)
print("gate passed:", report.passed)
if report.passed:
    registry.transition(v2.id, CandidateState.CHALLENGER); registry.transition(v2.id, CandidateState.CHAMPION)

routing = RoutingPolicy(champion=registry.champion("assistant").key, challenger="assistant:v1", challenger_percent=10.0)
engine = InferenceEngine(policy.model, max_batch=4)
backends = {"assistant:v1": EngineBackend("assistant:v1", engine, tok, background=False),
            "assistant:v2": EngineBackend("assistant:v2", engine, tok, background=False)}
router = Router(backends, routing.champion, route_fn=lambda rid, req: routing.route(rid, req).primary)
status, body = router.handle("POST", "/v1/completions", {"prompt": "Q: What is 2 + 2?\nA:", "max_tokens": 4, "temperature": 0})
print("served by", body["model"], "→", repr(body["choices"][0]["text"]))

# 6. rollback
if registry.champion("assistant").version == "v2":
    restored = rollback(registry, "assistant", "demo rollback", routing=routing)
    print("rolled back to", restored.key)
print("artifacts in", work)
