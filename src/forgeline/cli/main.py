"""``forgeline`` command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from forgeline import __version__
from forgeline.core.errors import ForgelineError


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str) if not isinstance(obj, str) else obj)


# ── commands ────────────────────────────────────────────────────────────────

def cmd_presets(args: argparse.Namespace) -> int:
    from forgeline.core.config import MODEL_PRESETS, model_spec_from_preset
    from forgeline.models.transformer.model import TransformerLM

    rows = []
    for name in MODEL_PRESETS:
        spec = model_spec_from_preset(name)
        rows.append({"preset": name, "layers": spec.n_layer, "heads": spec.n_head, "kv_heads": spec.n_kv_head, "dim": spec.n_embd,
                     "context": spec.block_size, "moe": spec.use_moe, "mla": spec.use_mla, "sliding_window": spec.sliding_window})
    if args.count_params:
        for row in rows:
            if row["dim"] <= 1024:
                spec = model_spec_from_preset(row["preset"], vocab_size=args.vocab_size)
                row["params_M"] = round(TransformerLM(spec).num_parameters() / 1e6, 1)
    _print(rows)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from forgeline.core.config import ExperimentManifest
    from forgeline.distributed.strategies import validate_distributed_config

    m = ExperimentManifest.load(args.manifest)
    topo = validate_distributed_config(m.distributed, world_size=args.world_size)
    _print({"ok": True, "name": m.name, "algorithm": m.algorithm, "model": m.model.to_dict(), "topology": topo})
    return 0


def cmd_data(args: argparse.Namespace) -> int:
    if args.data_cmd == "prepare":
        from forgeline.data.preprocessing import prepare_text_corpus

        _print(prepare_text_corpus(args.input, args.output, tokenizer=args.tokenizer, val_fraction=args.val_fraction))
    elif args.data_cmd == "prepare-hf":
        from forgeline.data.preprocessing import prepare_hf_corpus

        _print(prepare_hf_corpus(args.dataset, args.output, tokenizer_name=args.tokenizer, subset=args.subset,
                                 max_tokens=args.max_tokens, shard_size=args.shard_size))
    elif args.data_cmd == "shard":
        import numpy as np

        from forgeline.data.pretraining import detect_token_dtype
        from forgeline.data.streaming import shard_token_file

        dtype = detect_token_dtype(Path(args.input).parent)
        _print([str(p) for p in shard_token_file(args.input, args.output, args.n_shards, dtype=dtype)])
    elif args.data_cmd == "gsm8k":
        from forgeline.data.verifiable import build_gsm8k_tool_dataset

        _print({"written": build_gsm8k_tool_dataset(args.output, split=args.split, limit=args.limit)})
    elif args.data_cmd == "validate":
        from forgeline.data.records import SCHEMAS, ReadReport, read_jsonl

        rep = ReadReport()
        rows = read_jsonl(args.input, SCHEMAS[args.kind], on_error="skip", report=rep)
        _print({"valid": len(rows), "skipped": rep.skipped, "errors": rep.errors[:20]})
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from forgeline.core.config import ExperimentManifest
    from forgeline.core.lifecycle import RunContext
    from forgeline.distributed.strategies import build_strategy
    from forgeline.training.common.trainer import Trainer
    from forgeline.training.factory import build_algorithm

    manifest = ExperimentManifest.load(args.manifest)
    for override in args.set or []:
        _apply_override(manifest, override)
    if args.max_steps is not None:
        manifest.trainer.max_steps = args.max_steps
    if args.output_dir:
        manifest.output_dir = args.output_dir
    manifest.with_runtime_environment()
    strategy = build_strategy(manifest.distributed)
    strategy.setup()
    ctx = RunContext.create(manifest, metrics_backend=args.metrics, distributed=manifest.distributed.strategy != "single")
    manifest.save(ctx.output_dir / "manifest.yaml")
    alg = build_algorithm(manifest, ctx)
    _apply_strategy(alg, strategy, manifest)
    trainer = Trainer(ctx, alg, strategy=strategy)
    if args.resume or manifest.trainer.checkpoint.resume_from:
        trainer.resume(args.resume or None)
    try:
        summary = trainer.fit()
    finally:
        if (manifest.orchestration or {}).get("type") == "ray":
            from forgeline.orchestration.ray_backend import shutdown_pools

            shutdown_pools()
    ctx.close()
    strategy.teardown()
    _print(summary)
    return 0


def _apply_strategy(alg, strategy, manifest) -> None:
    """Attach the distributed strategy to a built algorithm.

    * ``single`` — nothing to do.
    * ``ddp`` — the Trainer broadcasts parameters and averages gradients for every algorithm; data-sampling
      and rollout seeds are offset by rank so ranks see different batches.
    * ``fsdp`` — the model of model-forward stages (pretrain, distill) is wrapped; other stages must use ddp.
    * ``deepspeed`` / ``tensor_parallel`` / ``pipeline_parallel`` — provided as strategies for custom loops
      (see docs/distributed.md); the step trainer rejects them with a clear message.
    """
    from forgeline.core.errors import DistributedConfigError
    from forgeline.core.runtime import seed_everything

    name = strategy.name
    if name == "single":
        return
    rank = strategy.rank()
    seed_everything(manifest.trainer.seed + rank)  # model weights were built with the base seed; sampling differs per rank
    for attr in ("rng",):
        if hasattr(alg, attr):
            import random

            setattr(alg, attr, random.Random(manifest.trainer.seed + rank))
    if hasattr(alg, "gen"):
        import torch

        alg.gen = torch.Generator().manual_seed(manifest.trainer.seed + rank)
    if name == "ddp":
        return
    if name == "fsdp":
        target = "model" if manifest.algorithm == "pretrain" else "student" if manifest.algorithm == "distill" else None
        if target is None:
            raise DistributedConfigError(f"fsdp is supported for pretrain and distill in the step trainer; use ddp for {manifest.algorithm}")
        setattr(alg, target, strategy.wrap_model(getattr(alg, target)))
        return
    raise DistributedConfigError(f"strategy {name!r} is available for custom training loops but not for `forgeline train`",
                                 hint="Use single, ddp or fsdp with the step trainer.")


def _apply_override(manifest, override: str) -> None:
    """``a.b.c=value`` dotted overrides (YAML-parsed values)."""
    import yaml

    key, _, raw = override.partition("=")
    value = yaml.safe_load(raw)
    obj = manifest
    parts = key.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part) if hasattr(obj, part) else obj[part]
    if isinstance(obj, dict):
        obj[parts[-1]] = value
    else:
        if not hasattr(obj, parts[-1]):
            raise ForgelineError(f"unknown manifest field {key!r}")
        setattr(obj, parts[-1], value)
    manifest.validate()


def _load_policy(args: argparse.Namespace):
    import torch

    from forgeline.core.runtime import resolve_device
    from forgeline.models.loading import load_policy_from_checkpoint

    device = resolve_device(args.device)
    return load_policy_from_checkpoint(args.checkpoint, device, data_dir=args.data_dir)


def cmd_generate(args: argparse.Namespace) -> int:
    from forgeline.core.protocols import GenerationSettings
    from forgeline.inference.generation import generate_text

    policy = _load_policy(args)
    settings = GenerationSettings(max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k or None,
                                  top_p=args.top_p, min_p=args.min_p, repetition_penalty=args.repetition_penalty, use_cache=not args.no_cache)
    spec_gen = None
    if args.draft_checkpoint:
        from forgeline.models.generation.speculative import SpeculativeGenerator
        from forgeline.models.loading import load_model_from_checkpoint

        draft, _, _ = load_model_from_checkpoint(args.draft_checkpoint, policy.device)
        spec_gen = SpeculativeGenerator(policy.model, draft, k=args.spec_k)
    prompts = [args.prompt]
    if args.interactive:
        prompts = []
        print("Interactive mode — type a prompt (empty line or 'quit' to exit).")
        while True:
            try:
                line = input("> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not line or line.lower() in ("quit", "exit"):
                break
            text, stats = generate_text(policy.model, policy.tokenizer, line, settings, spec_generator=spec_gen)
            print(text)
            print(f"[{stats['n_generated']} tokens, {stats['tokens_per_sec']:.1f} tok/s]")
        return 0
    for p in prompts:
        text, stats = generate_text(policy.model, policy.tokenizer, p, settings, spec_generator=spec_gen)
        print(text)
        print(f"[{stats['n_generated']} tokens in {stats['elapsed_s']:.2f}s, {stats['tokens_per_sec']:.1f} tok/s]", file=sys.stderr)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from forgeline.evaluation.results import save_results
    from forgeline.evaluation.suites import BenchmarkSuite, build_benchmark

    results = []
    if args.ray_workers:
        from forgeline.orchestration.config import RayConfig
        from forgeline.orchestration.ray_backend import RayEvaluator

        ray_cfg = RayConfig(address=args.ray_address, cpus_per_worker=args.ray_cpus_per_worker)
        with RayEvaluator(args.checkpoint, args.ray_workers, ray_cfg, data_dir=args.data_dir) as evaluator:
            for name in args.suites:
                results.append(evaluator.run(name, max_new_tokens=args.max_tokens, n_shot=args.n_shot,
                                             max_tasks=args.max_tasks, offline=args.offline))
        policy = _load_policy(args) if args.heldout_data else None
    else:
        policy = _load_policy(args)
        for name in args.suites:
            bench = build_benchmark(name, n_shot=args.n_shot, max_tasks=args.max_tasks, offline=args.offline)
            results.append(BenchmarkSuite(bench, max_new_tokens=args.max_tokens).run(policy))
    if args.heldout_data:
        from forgeline.data.pretraining import MemmapCorpus
        from forgeline.evaluation.quality import HeldOutLossSuite

        results.append(HeldOutLossSuite(MemmapCorpus(args.heldout_data, "val", policy.spec.block_size)).run(policy.model))
    if args.output:
        save_results(results, args.output)
    _print([{"suite": r.suite, "metrics": r.metrics, "n": r.n_samples, "offline_samples": r.details.get("offline_samples")} for r in results])
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from forgeline.models.loading import load_model_from_checkpoint
    from forgeline.models.quantization.gguf import export_gguf

    model, spec, _ = load_model_from_checkpoint(args.checkpoint, "cpu")
    if args.merge_adapters:
        from forgeline.models.adapters.lora import merge_lora

        merge_lora(model)
    _print(export_gguf(model.state_dict(), spec, args.output, quantize=args.quantize))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from forgeline.inference.engine import InferenceEngine
    from forgeline.observability.metrics import build_metrics_sink
    from forgeline.serving.api import Router
    from forgeline.serving.backends import EngineBackend
    from forgeline.serving.server import ServingServer

    policy = _load_policy(args)
    engine = InferenceEngine(policy.model, max_batch=args.max_batch, max_seq_len=args.max_seq_len)
    backend = EngineBackend(args.model_name, engine, policy.tokenizer)
    route_fn = None
    if args.routing:
        from forgeline.deployment.rollout import RoutingPolicy

        policy_cfg = RoutingPolicy.load(args.routing)
        route_fn = lambda rid, requested: policy_cfg.route(rid, requested).primary  # noqa: E731
    router = Router({args.model_name: backend}, args.model_name, route_fn=route_fn,
                    metrics=build_metrics_sink(args.metrics, output_dir=args.output_dir, run_name="serve"))
    server = ServingServer(router, host=args.host, port=args.port)
    print(f"forgeline serving {args.model_name} on http://{args.host}:{server.port}  (endpoints: /health /v1/models /v1/completions /v1/chat/completions /metrics)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        backend.close()
        server.stop()
    return 0


def cmd_registry(args: argparse.Namespace) -> int:
    from forgeline.deployment.candidates import CandidateState, ModelCandidate
    from forgeline.deployment.promotion import PromotionGate
    from forgeline.deployment.registry import CandidateRegistry
    from forgeline.deployment.rollback import disable_candidate, rollback

    reg = CandidateRegistry(args.registry)
    if args.registry_cmd == "list":
        _print([{"id": c.id, "key": c.key, "state": c.state.value, "algorithm": c.algorithm, "metrics": c.metrics,
                 "promotion": c.promotion_status} for c in reg.list()])
    elif args.registry_cmd == "register":
        metrics = json.loads(args.metrics) if args.metrics else {}
        if args.metrics_file:
            from forgeline.evaluation.results import load_results, merge_metrics

            metrics.update(merge_metrics(load_results(args.metrics_file)))
        c = reg.register(ModelCandidate(name=args.name, version=args.version, algorithm=args.algorithm, checkpoint=args.checkpoint,
                                        metrics=metrics, tags=args.tags or []))
        _print({"registered": c.id, "key": c.key})
    elif args.registry_cmd == "promote":
        c = reg.get(args.id)
        target = CandidateState(args.to)
        if target == CandidateState.CHAMPION and args.gate:
            from forgeline.core.config import load_config

            gate = PromotionGate.from_config(load_config(args.gate))
            baseline = reg.champion(c.name)
            report = gate.evaluate(c.metrics, baseline.metrics if baseline else None)
            c.promotion_status = "passed" if report.passed else "rejected"
            c.promotion_report = report.to_dict()
            reg.update(c)
            if not report.passed:
                _print({"promoted": False, "report": report.to_dict()})
                return 2
        if target == CandidateState.CHAMPION and c.state not in (CandidateState.CHALLENGER, CandidateState.CHAMPION):
            reg.transition(c.id, CandidateState.CHALLENGER, reason="staged before champion promotion")
        reg.transition(c.id, target, reason=args.reason)
        _print({"promoted": True, "id": c.id, "state": target.value})
    elif args.registry_cmd == "rollback":
        restored = rollback(reg, args.name, reason=args.reason)
        _print({"champion": restored.key})
    elif args.registry_cmd == "disable":
        c = disable_candidate(reg, args.id, reason=args.reason)
        _print({"disabled": c.key})
    return 0


def cmd_rlaif(args: argparse.Namespace) -> int:
    from forgeline.training.rlaif import (
        CONSTITUTIONAL_SAMPLE_PROMPTS, SAMPLE_PROMPTS, ConstitutionalConfig, rule_based_constitutional_model, rule_generator,
        rule_judge, run_constitutional, run_pairwise_rlaif, save_constitutional_output, save_pairwise_output,
    )

    if args.rlaif_cmd == "pairwise":
        prompts = (SAMPLE_PROMPTS * 10)[: args.n_prompts]
        if args.checkpoint:
            from forgeline.training.rlaif import make_policy_generator, make_policy_pairwise_judge

            policy = _load_policy(args)
            gen, judge = make_policy_generator(policy), make_policy_pairwise_judge(policy)
        else:
            gen, judge = rule_generator, rule_judge
        stats = run_pairwise_rlaif(prompts, gen, judge, n_candidates=args.candidates)
        _print(save_pairwise_output(stats, args.output))
    elif args.rlaif_cmd == "constitutional":
        prompts = (CONSTITUTIONAL_SAMPLE_PROMPTS * 10)[: args.n_prompts]
        model_fn = rule_based_constitutional_model
        if args.checkpoint:
            from forgeline.core.protocols import GenerationSettings

            policy = _load_policy(args)

            def model_fn(prompt: str) -> str:  # type: ignore[misc]
                import torch

                ids = torch.tensor(policy.encode(prompt)[-policy.spec.block_size // 2:], dtype=torch.long).unsqueeze(0)
                return policy.decode(policy.generate(ids.to(policy.device), GenerationSettings(max_new_tokens=args.max_tokens, temperature=0.7))[0])
        examples = run_constitutional(prompts, model_fn, ConstitutionalConfig(n_principles_per_example=args.n_principles))
        _print(save_constitutional_output(examples, args.output))
    return 0


def cmd_synthesis(args: argparse.Namespace) -> int:
    from forgeline.domains.synthesis import TabularActorCritic, TabularPPOConfig, TabularPPOTrainer, load_trajectories, rule_score
    from forgeline.evaluation.reward import RecordRewardSuite

    rows = load_trajectories(args.data)
    split = int(len(rows) * 0.8)
    train, test = rows[:split], rows[split:]
    trainer = TabularPPOTrainer(TabularActorCritic(), TabularPPOConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr))
    metrics = trainer.train(train)
    record_eval = RecordRewardSuite(test, rule_score).run()
    out = {"training": {k: v for k, v in metrics.items()}, "heldout_record_reward": record_eval.to_dict()}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))
    _print({"final_train_reward": metrics["final_reward"], "heldout_record_reward_mean": record_eval.metrics["mean"],
            "note": "heldout_record_reward scores dataset records (policy-independent)"})
    return 0


# ── parser ──────────────────────────────────────────────────────────────────

def cmd_dashboard(args: argparse.Namespace) -> int:
    from forgeline.dashboard.data import DashboardSources, snapshot

    sources = DashboardSources(runs=list(args.runs), registry=args.registry, benchmarks=args.benchmarks)
    if args.export:
        path = Path(args.export)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot(sources), indent=1, default=str))
        print(f"wrote dashboard snapshot to {path}")
        return 0
    from forgeline.dashboard.server import DashboardApp, DashboardServer

    server = DashboardServer(DashboardApp(sources), host=args.host, port=args.port)
    print(f"forgeline dashboard on {server.url}  (runs: {', '.join(sources.runs)} · registry: {sources.registry} · benchmarks: {sources.benchmarks})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forgeline", description="Forgeline — post-training, distributed training, evaluation, inference and model lifecycle for language models.")
    p.add_argument("--version", action="version", version=f"forgeline {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("presets", help="list model presets"); s.add_argument("--count-params", action="store_true"); s.add_argument("--vocab-size", type=int, default=65536); s.set_defaults(func=cmd_presets)

    s = sub.add_parser("validate", help="validate an experiment manifest without running it")
    s.add_argument("manifest"); s.add_argument("--world-size", type=int, default=None); s.set_defaults(func=cmd_validate)

    d = sub.add_parser("data", help="dataset preparation and validation"); ds = d.add_subparsers(dest="data_cmd", required=True)
    x = ds.add_parser("prepare", help="tokenize a text file into train.bin/val.bin"); x.add_argument("--input", required=True); x.add_argument("--output", required=True)
    x.add_argument("--tokenizer", default="char", help="char | tiktoken[:enc] | hf:<name>"); x.add_argument("--val-fraction", type=float, default=0.1)
    x = ds.add_parser("prepare-hf", help="stream a HuggingFace dataset into shard files"); x.add_argument("--dataset", required=True); x.add_argument("--subset", default=None)
    x.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B"); x.add_argument("--output", required=True); x.add_argument("--max-tokens", type=int, default=100_000_000); x.add_argument("--shard-size", type=int, default=10_000_000)
    x = ds.add_parser("shard", help="split a .bin token file into shards"); x.add_argument("--input", required=True); x.add_argument("--output", required=True); x.add_argument("--n-shards", type=int, default=16)
    x = ds.add_parser("gsm8k", help="download GSM8K as verifiable tool-use tasks"); x.add_argument("--output", required=True); x.add_argument("--split", default="train"); x.add_argument("--limit", type=int, default=None)
    x = ds.add_parser("validate", help="validate a JSONL dataset against a schema"); x.add_argument("--input", required=True); x.add_argument("--kind", required=True, choices=["supervised", "preference", "verifiable", "process", "reward"])
    d.set_defaults(func=cmd_data)

    s = sub.add_parser("train", help="run a training stage from a manifest")
    s.add_argument("manifest"); s.add_argument("--set", action="append", help="dotted override, e.g. trainer.max_steps=10")
    s.add_argument("--max-steps", type=int, default=None); s.add_argument("--output-dir", default=None); s.add_argument("--resume", default=None)
    s.add_argument("--metrics", default="local", help="local | none | wandb | tensorboard"); s.set_defaults(func=cmd_train)

    for name, fn in (("generate", cmd_generate), ("evaluate", cmd_evaluate), ("serve", cmd_serve)):
        s = sub.add_parser(name)
        s.add_argument("--checkpoint", required=True); s.add_argument("--data-dir", default=None, help="directory with meta.pkl when the checkpoint lacks tokenizer metadata"); s.add_argument("--device", default="auto")
        if name == "generate":
            s.add_argument("--prompt", default="\n"); s.add_argument("--max-tokens", type=int, default=200); s.add_argument("--temperature", type=float, default=0.8)
            s.add_argument("--top-k", type=int, default=50); s.add_argument("--top-p", type=float, default=None); s.add_argument("--min-p", type=float, default=None)
            s.add_argument("--repetition-penalty", type=float, default=1.0); s.add_argument("--no-cache", action="store_true"); s.add_argument("--interactive", action="store_true")
            s.add_argument("--draft-checkpoint", default=None); s.add_argument("--spec-k", type=int, default=5)
        elif name == "evaluate":
            s.add_argument("--suites", nargs="+", default=["gsm8k"]); s.add_argument("--n-shot", type=int, default=0); s.add_argument("--max-tasks", type=int, default=0)
            s.add_argument("--max-tokens", type=int, default=64); s.add_argument("--offline", action="store_true", help="use built-in sample tasks (no download)")
            s.add_argument("--heldout-data", default=None); s.add_argument("--output", default=None)
            s.add_argument("--ray-workers", type=int, default=0, help="shard benchmark tasks across N Ray workers (needs the ray extra)")
            s.add_argument("--ray-address", default=None, help="join an existing Ray cluster (default: start a local instance)")
            s.add_argument("--ray-cpus-per-worker", type=float, default=1.0)
        else:
            s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000); s.add_argument("--model-name", default="forgeline-model")
            s.add_argument("--max-batch", type=int, default=8); s.add_argument("--max-seq-len", type=int, default=None); s.add_argument("--routing", default=None, help="routing policy JSON")
            s.add_argument("--metrics", default="none"); s.add_argument("--output-dir", default="runs/serve")
        s.set_defaults(func=fn)

    s = sub.add_parser("export", help="export a checkpoint to GGUF"); s.add_argument("--checkpoint", required=True); s.add_argument("--output", required=True)
    s.add_argument("--quantize", default="none", choices=["none", "q8_0", "q4_0"]); s.add_argument("--merge-adapters", action="store_true"); s.set_defaults(func=cmd_export)

    r = sub.add_parser("registry", help="model candidate registry"); r.add_argument("--registry", default="registry/candidates.json"); rs = r.add_subparsers(dest="registry_cmd", required=True)
    rs.add_parser("list")
    x = rs.add_parser("register"); x.add_argument("--name", required=True); x.add_argument("--version", required=True); x.add_argument("--algorithm", required=True); x.add_argument("--checkpoint", required=True)
    x.add_argument("--metrics", default=None, help="JSON object"); x.add_argument("--metrics-file", default=None, help="evaluation results JSON"); x.add_argument("--tags", nargs="*")
    x = rs.add_parser("promote"); x.add_argument("--id", required=True); x.add_argument("--to", required=True, choices=["shadow", "challenger", "champion", "retired"]); x.add_argument("--gate", default=None, help="promotion gate YAML"); x.add_argument("--reason", default="")
    x = rs.add_parser("rollback"); x.add_argument("--name", required=True); x.add_argument("--reason", default="manual rollback")
    x = rs.add_parser("disable"); x.add_argument("--id", required=True); x.add_argument("--reason", default="disabled")
    r.set_defaults(func=cmd_registry)

    a = sub.add_parser("rlaif", help="AI-feedback data generation"); asub = a.add_subparsers(dest="rlaif_cmd", required=True)
    for name in ("pairwise", "constitutional"):
        x = asub.add_parser(name); x.add_argument("--output", required=True); x.add_argument("--n-prompts", type=int, default=10)
        x.add_argument("--checkpoint", default=None); x.add_argument("--data-dir", default=None); x.add_argument("--device", default="auto"); x.add_argument("--max-tokens", type=int, default=128)
        if name == "pairwise":
            x.add_argument("--candidates", type=int, default=4)
        else:
            x.add_argument("--n-principles", type=int, default=2)
    a.set_defaults(func=cmd_rlaif)

    s = sub.add_parser("dashboard", help="local read-only dashboard: runs, checkpoints, evaluations, registry, benchmarks, model inspection")
    s.add_argument("--runs", nargs="+", default=["runs"], help="directories searched for run outputs"); s.add_argument("--registry", default="registry/candidates.json")
    s.add_argument("--benchmarks", default="benchmarks"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8765)
    s.add_argument("--export", default=None, help="write a JSON snapshot instead of serving"); s.set_defaults(func=cmd_dashboard)

    s = sub.add_parser("synthesis-ppo", help="tabular actor-critic PPO on synthesis trajectories")
    s.add_argument("--data", required=True); s.add_argument("--epochs", type=int, default=5); s.add_argument("--batch-size", type=int, default=8); s.add_argument("--lr", type=float, default=1e-4)
    s.add_argument("--output", default=None); s.set_defaults(func=cmd_synthesis)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except ForgelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
