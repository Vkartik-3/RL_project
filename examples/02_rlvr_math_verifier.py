"""RL with verifiable rewards: DAPO driven by a math verifier + chain-of-thought format verifier (CPU)."""

import tempfile

from forgeline.core.config import CheckpointConfig, ExperimentManifest, OptimizerConfig, TrainerConfig, model_spec_from_preset
from forgeline.core.lifecycle import RunContext
from forgeline.data import CharTokenizer, VerifiableTaskDataset
from forgeline.models import NativePolicy, TransformerLM
from forgeline.rollouts.engine import RolloutConfig
from forgeline.training import DAPOConfig, RLVRConfig, Trainer, build_rlvr_algorithm, evaluate_pass_rate
from forgeline.training.rlvr import build_verifiable_reward

tok = CharTokenizer.ascii()
spec = model_spec_from_preset("tiny", vocab_size=tok.vocab_size, block_size=96)
policy = NativePolicy(TransformerLM(spec), tok)
tasks = VerifiableTaskDataset.from_jsonl("data/samples/verifiable/arithmetic_tasks.jsonl", limit=16).tasks
cfg = RLVRConfig(task_type="math", optimizer="dapo", require_format=True, dapo=DAPOConfig(group_size=4, prompts_per_step=2))
alg = build_rlvr_algorithm(policy, tasks, cfg, RolloutConfig(max_new_tokens=24, temperature=0.9))
m = ExperimentManifest(name="rlvr", algorithm="rlvr", model=spec, output_dir=tempfile.mkdtemp(),
                       trainer=TrainerConfig(max_steps=10, eval_every=0, log_every=2, device="cpu",
                                             optimizer=OptimizerConfig(learning_rate=5e-5, weight_decay=0.0), checkpoint=CheckpointConfig(save_every=0)))
ctx = RunContext.create(m)
print(Trainer(ctx, alg).fit())
print(evaluate_pass_rate(policy, tasks[:8], build_verifiable_reward(cfg), max_new_tokens=24))
