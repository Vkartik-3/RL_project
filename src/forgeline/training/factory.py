"""Build models, datasets and algorithms from an :class:`ExperimentManifest`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from forgeline.core.config import ExperimentManifest
from forgeline.core.errors import ConfigError
from forgeline.core.lifecycle import RunContext
from forgeline.data.preference import PreferenceDataset
from forgeline.data.pretraining import MemmapCorpus
from forgeline.data.records import PreferenceRecord, SFTRecord, VerifiableTask, deterministic_split, read_jsonl
from forgeline.data.supervised import SupervisedDataset
from forgeline.data.tokenizers import CharTokenizer, Tokenizer, load_tokenizer_metadata, resolve_tokenizer, tokenizer_from_metadata
from forgeline.data.verifiable import VerifiableTaskDataset, load_prompts
from forgeline.models.loading import load_model_from_checkpoint
from forgeline.models.policy import NativePolicy
from forgeline.models.reward import SequenceRewardModel
from forgeline.models.transformer.model import TransformerLM
from forgeline.rollouts.engine import RolloutConfig
from forgeline.rollouts.rewards import build_reward_provider
from forgeline.training.common.trainer import PostTrainingAlgorithm


def _kw(params: Dict[str, Any], cls):
    """Filter ``params`` to the fields of dataclass ``cls`` (unknown keys are errors)."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(params) - names
    if unknown:
        raise ConfigError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**params)


def _attach_orchestration(algorithm: PostTrainingAlgorithm, manifest: ExperimentManifest,
                          reward_cfg: Dict[str, Any]) -> PostTrainingAlgorithm:
    """Move rollouts / reward scoring into Ray worker pools when ``orchestration.type`` is ``ray``."""
    orch = manifest.orchestration or {}
    if orch.get("type", "local") != "ray":
        return algorithm
    from forgeline.orchestration.ray_backend import attach_to_algorithm

    attach_to_algorithm(algorithm, orch, reward_cfg, seed=manifest.trainer.seed)
    return algorithm


def build_tokenizer(manifest: ExperimentManifest) -> Tokenizer:
    data_dir = Path(manifest.data.path)
    if manifest.checkpoint_path:
        from forgeline.checkpoints.manager import CheckpointManager

        meta = CheckpointManager.read_manifest(manifest.checkpoint_path).get("tokenizer_meta")
        if meta:
            return tokenizer_from_metadata(meta)
    spec = manifest.data.tokenizer
    try:
        return resolve_tokenizer(spec, data_dir if data_dir.exists() else None)
    except Exception:
        if spec in ("auto", "char"):
            return CharTokenizer.ascii()
        raise


def build_model(manifest: ExperimentManifest, ctx: RunContext, tokenizer: Optional[Tokenizer] = None) -> TransformerLM:
    if manifest.checkpoint_path:
        model, spec, _ = load_model_from_checkpoint(manifest.checkpoint_path, ctx.device)
        manifest.model = spec
        return model
    if tokenizer is not None and manifest.model.vocab_size != tokenizer.vocab_size:
        manifest.model.vocab_size = tokenizer.vocab_size
    model = TransformerLM(manifest.model).to(ctx.device)
    if manifest.trainer.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    return model


def build_policy(manifest: ExperimentManifest, ctx: RunContext, value_head: bool = False):
    backend = manifest.backend or {}
    if backend.get("type") == "huggingface":
        from forgeline.models.policy import HFPolicy

        opts = {k: v for k, v in backend.items() if k not in ("type", "model_name", "adapter_path")}
        dtype = opts.pop("torch_dtype", None)
        if isinstance(dtype, str):
            opts["torch_dtype"] = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        opts.setdefault("device_map", None if ctx.device.type == "cpu" else {"": ctx.device.index or 0})
        policy = HFPolicy(backend["model_name"], value_head=value_head, **opts)
        if backend.get("adapter_path"):
            policy.model.load_adapter(backend["adapter_path"], adapter_name="default", is_trainable=True)
        if manifest.checkpoint_path:
            from forgeline.checkpoints.manager import CheckpointManager

            policy.load_checkpoint_state(CheckpointManager.load(manifest.checkpoint_path).model_state)
        return policy.to(ctx.device) if opts.get("device_map") is None else policy
    tok = build_tokenizer(manifest)
    model = build_model(manifest, ctx, tok)
    adapter = manifest.trainer.adapter if manifest.trainer.adapter.method != "none" else None
    return NativePolicy(model, tok, adapter=adapter, value_head=value_head).to(ctx.device)


def _trajectory_split(manifest: ExperimentManifest):
    from forgeline.domains.synthesis.tabular_ppo import load_trajectories

    rows = load_trajectories(manifest.data.path)
    split = int(len(rows) * float(manifest.data.extra.get("train_fraction", 0.8)))  # ordered split
    return rows[:split], rows[split:]


def _tasks(manifest: ExperimentManifest) -> List[VerifiableTask]:
    path = Path(manifest.data.path)
    if path.is_file():
        return VerifiableTaskDataset.from_jsonl(path, on_error=manifest.data.extra.get("on_error", "raise"),
                                               limit=manifest.data.extra.get("limit")).tasks
    return VerifiableTaskDataset.builtin_arithmetic().tasks


def build_algorithm(manifest: ExperimentManifest, ctx: RunContext) -> PostTrainingAlgorithm:
    alg = manifest.algorithm
    p = dict(manifest.algorithm_params)
    rollout_params = p.pop("rollout", {}) or {}
    rollout = _kw(rollout_params, RolloutConfig) if rollout_params else None

    if alg == "pretrain":
        from forgeline.training.pretrain import PretrainAlgorithm, PretrainConfig

        tok_meta = load_tokenizer_metadata(manifest.data.path) if (Path(manifest.data.path) / "meta.pkl").exists() else None
        if tok_meta:
            manifest.model.vocab_size = int(tok_meta["vocab_size"])
        model = build_model(manifest, ctx)
        train = MemmapCorpus(manifest.data.path, "train", manifest.model.block_size)
        val = MemmapCorpus(manifest.data.path, "val", manifest.model.block_size) if (Path(manifest.data.path) / "val.bin").exists() else None
        cfg = _kw({"batch_size": manifest.trainer.batch_size, "eval_batches": manifest.trainer.eval_batches, **p}, PretrainConfig)
        return PretrainAlgorithm(model, train, cfg, val, device=ctx.device, tokenizer_meta=tok_meta)

    if alg == "distill":
        from forgeline.training.distill import DistillationAlgorithm, DistillConfig

        teacher_path = p.pop("teacher_checkpoint", None)
        if not teacher_path:
            raise ConfigError("distill needs algorithm_params.teacher_checkpoint")
        teacher, _, _ = load_model_from_checkpoint(teacher_path, ctx.device)
        manifest.model.vocab_size = teacher.spec.vocab_size
        student = build_model(manifest, ctx)
        corpus = MemmapCorpus(manifest.data.path, "train", manifest.model.block_size)
        cfg = _kw({"batch_size": manifest.trainer.batch_size, **p}, DistillConfig)
        return DistillationAlgorithm(student, teacher, corpus, cfg, device=ctx.device)

    if alg == "sft":
        from forgeline.training.sft import SFTAlgorithm, SFTConfig

        policy = build_policy(manifest, ctx)
        if manifest.data.kind == "trajectories":
            from forgeline.domains.synthesis import build_sft_records, filter_high_reward

            from forgeline.domains.synthesis.tabular_ppo import load_trajectories

            rows = load_trajectories(manifest.data.path)  # high-reward filtering over all records
            train = build_sft_records(filter_high_reward(rows, float(manifest.data.extra.get("reward_threshold", 0.8))))
            val = []
        else:
            records = read_jsonl(manifest.data.path, SFTRecord, on_error=manifest.data.extra.get("on_error", "raise"))
            train, val = deterministic_split(records, manifest.data.val_fraction, manifest.data.seed)
        cfg = _kw({"batch_size": manifest.trainer.batch_size, "max_length": manifest.data.max_length, **p}, SFTConfig)
        return SFTAlgorithm(policy, SupervisedDataset(train, policy.tokenizer, cfg.max_length), cfg,
                            SupervisedDataset(val, policy.tokenizer, cfg.max_length) if val else None)

    if alg in ("dpo", "reward_model"):
        if manifest.data.kind == "trajectories":
            from forgeline.domains.synthesis import build_preference_pairs
            from forgeline.domains.synthesis.preferences import pairs_to_records

            rows, _ = _trajectory_split(manifest)
            train = pairs_to_records(build_preference_pairs(rows, int(manifest.data.extra.get("pairs_per_molecule", 20)), seed=manifest.data.seed))
            val = []
        else:
            records = read_jsonl(manifest.data.path, PreferenceRecord, on_error=manifest.data.extra.get("on_error", "raise"))
            train, val = deterministic_split(records, manifest.data.val_fraction, manifest.data.seed)
        max_p = manifest.data.extra.get("max_prompt_length", 384)
        max_r = manifest.data.extra.get("max_response_length", manifest.data.max_length)
        if alg == "dpo":
            from forgeline.training.dpo import DPOAlgorithm, DPOConfig

            policy = build_policy(manifest, ctx)
            cfg = _kw({"batch_size": manifest.trainer.batch_size, **p}, DPOConfig)
            return DPOAlgorithm(policy, PreferenceDataset(train, policy.tokenizer, max_p, max_r), cfg,
                                PreferenceDataset(val, policy.tokenizer, max_p, max_r) if val else None)
        from forgeline.training.reward_model import RewardModelAlgorithm, RewardModelConfig

        tok = build_tokenizer(manifest)
        manifest.model.vocab_size = tok.vocab_size
        rm = SequenceRewardModel(manifest.model, pooling=p.pop("pooling", "last")).to(ctx.device)
        if manifest.checkpoint_path:
            backbone, _, _ = load_model_from_checkpoint(manifest.checkpoint_path, ctx.device)
            rm.backbone = backbone
        cfg = _kw({"batch_size": manifest.trainer.batch_size, **p}, RewardModelConfig)
        return RewardModelAlgorithm(rm, PreferenceDataset(train, tok, max_p, max_r), cfg,
                                    PreferenceDataset(val, tok, max_p, max_r) if val else None, tokenizer_meta=tok.metadata())

    if alg in ("ppo", "grpo", "dapo"):
        reward_cfg = manifest.reward or {"type": "rule", "name": "length", "target_length": 32}
        reward = build_reward_provider(reward_cfg)
        policy = build_policy(manifest, ctx, value_head=(alg == "ppo"))
        if manifest.data.kind == "trajectories":
            from forgeline.domains.synthesis import build_synthesis_prompt

            tasks, _ = _trajectory_split(manifest)
            tasks = [dict(t, prompt=build_synthesis_prompt(t)) for t in tasks]
        else:
            tasks = [t.to_dict() for t in _tasks(manifest)] if manifest.data.kind == "verifiable" else None
        if tasks:
            prompts = [torch.tensor(policy.encode(t["prompt"]), dtype=torch.long) for t in tasks]
        else:
            prompts = [torch.tensor(ids, dtype=torch.long) for ids in load_prompts(manifest.data.path if Path(manifest.data.path).is_file() else None,
                                                                              policy.tokenizer, manifest.data.max_length)]
        if alg == "ppo":
            from forgeline.training.ppo import PPOAlgorithm, PPOConfig

            built = PPOAlgorithm(policy, prompts, reward, _kw(p, PPOConfig), rollout, tasks=tasks)
        elif alg == "grpo":
            from forgeline.training.grpo import GRPOAlgorithm, GRPOConfig

            cfg = _kw(p, GRPOConfig)
            if cfg.agent:
                built = GRPOAlgorithm(policy, tasks or [], reward, cfg)
            else:
                built = GRPOAlgorithm(policy, prompts, reward, cfg, rollout, tasks=tasks)
        else:
            from forgeline.training.dapo import DAPOAlgorithm, DAPOConfig

            built = DAPOAlgorithm(policy, prompts, reward, _kw(p, DAPOConfig), rollout, tasks=tasks)
        return _attach_orchestration(built, manifest, reward_cfg)

    if alg == "rlvr":
        from forgeline.training.dapo import DAPOConfig
        from forgeline.training.grpo import GRPOConfig
        from forgeline.training.rlvr import RLVRConfig, build_rlvr_algorithm

        policy = build_policy(manifest, ctx)
        grpo = _kw(p.pop("grpo", {}) or {}, GRPOConfig)
        dapo = _kw(p.pop("dapo", {}) or {}, DAPOConfig)
        cfg = _kw({**p, "grpo": grpo, "dapo": dapo}, RLVRConfig)
        return build_rlvr_algorithm(policy, _tasks(manifest), cfg, rollout)

    raise ConfigError(f"algorithm {alg!r} is not run through the step trainer; use the dedicated CLI command",
                      hint="rlaif/constitutional/star/hill_climb/tabular_ppo have their own commands.")
