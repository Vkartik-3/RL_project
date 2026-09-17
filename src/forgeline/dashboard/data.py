"""Read-only views over what Forgeline writes to disk.

Sources:

* run directories — ``metrics.jsonl`` / ``events.jsonl`` from :class:`LocalJsonlMetricsSink`, ``manifest.yaml`` saved by
  ``forgeline train``, checkpoint directories (``manifest.json`` + ``LATEST``), evaluation result files;
* the candidate registry JSON;
* ``benchmarks/**/results.json`` evidence records;
* checkpoints for model inspection (architecture, weight statistics, attention patterns, activation flow).

Every function is pure file IO so it can be tested and exported without the HTTP server. Checkpoints are only loaded
when they were discovered under a configured runs root (``torch.load`` executes pickled code, so arbitrary paths from a
request are never loaded).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

MAX_POINTS = 1500
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "site-packages"}


def _run_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]


def _read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a partially written trailing line while a run is live
    return rows[-limit:] if limit else rows


def _walk(root: Path, max_depth: int) -> Iterable[Path]:
    root = root.resolve()
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - base_depth
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.endswith(".tmp")]
        if depth >= max_depth:
            dirnames[:] = []
        yield Path(dirpath)


def _downsample(points: List[List[float]], limit: int = MAX_POINTS) -> List[List[float]]:
    if len(points) <= limit:
        return points
    stride = len(points) / limit
    out = [points[int(i * stride)] for i in range(limit)]
    out[-1] = points[-1]
    return out


# ── runs ─────────────────────────────────────────────────────────────────────

def _load_manifest(run_dir: Path) -> Dict[str, Any]:
    for name in ("manifest.yaml", "manifest.yml", "manifest.json"):
        p = run_dir / name
        if p.exists():
            try:
                from forgeline.core.config import load_config

                return dict(load_config(p))
            except Exception:  # noqa: BLE001 - show the run even if its manifest no longer parses
                return {}
    return {}


def _status(events: List[Dict[str, Any]]) -> str:
    names = [e.get("event") for e in events]
    if "run.failed" in names:
        return "failed"
    if "run.finished" in names:
        return "finished"
    if "run.started" in names:
        return "running"
    return "unknown"


def discover_runs(roots: Iterable[str | Path], max_depth: int = 4) -> List[Dict[str, Any]]:
    """Every directory holding ``metrics.jsonl`` or ``events.jsonl`` under the roots, newest first."""
    runs: Dict[str, Dict[str, Any]] = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for d in _walk(root, max_depth):
            metrics, events_path = d / "metrics.jsonl", d / "events.jsonl"
            if not (metrics.exists() or events_path.exists()):
                continue
            rid = _run_id(d)
            events = _read_jsonl(events_path)
            manifest = _load_manifest(d)
            last = _read_jsonl(metrics, limit=1)
            mtime = max((p.stat().st_mtime for p in (metrics, events_path) if p.exists()), default=0.0)
            runs[rid] = {
                "id": rid, "name": manifest.get("name") or d.name, "path": str(d),
                "algorithm": manifest.get("algorithm") or next((e.get("algorithm") for e in events if e.get("algorithm")), ""),
                "model_preset": manifest.get("model_preset", ""),
                "strategy": (manifest.get("distributed") or {}).get("strategy", ""),
                "orchestration": (manifest.get("orchestration") or {}).get("type", "local") if manifest else "",
                "status": _status(events), "last_step": last[0].get("step") if last else None,
                "n_events": len(events), "updated_at": mtime,
                "live": _status(events) == "running" and time.time() - mtime < 120,
            }
    return sorted(runs.values(), key=lambda r: r["updated_at"], reverse=True)


def run_detail(run_dir: str | Path) -> Dict[str, Any]:
    d = Path(run_dir)
    rows = _read_jsonl(d / "metrics.jsonl")
    series: Dict[str, List[List[float]]] = {}
    for r in rows:
        step = r.get("step")
        for k, v in r.items():
            if k in ("step", "ts") or not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            series.setdefault(k, []).append([step, v])
    events = _read_jsonl(d / "events.jsonl")
    summary = next((e for e in reversed(events) if e.get("event") == "run.finished"), None)
    return {
        "id": _run_id(d), "path": str(d), "manifest": _load_manifest(d), "status": _status(events),
        "series": {k: _downsample(v) for k, v in sorted(series.items())},
        "events": events[-500:], "summary": summary,
        "checkpoints": list_checkpoints(d), "evaluations": list_evaluations(d),
    }


# ── checkpoints ──────────────────────────────────────────────────────────────

def list_checkpoints(root: str | Path, max_depth: int = 3) -> List[Dict[str, Any]]:
    from forgeline.checkpoints.manager import CheckpointManager

    out: List[Dict[str, Any]] = []
    root = Path(root)
    if not root.exists():
        return out
    for d in _walk(root, max_depth):
        if not (d / "manifest.json").exists() or not (d / "model.pt").exists():
            continue
        try:
            manifest = CheckpointManager.validate(d)
            valid, error = True, ""
        except Exception as exc:  # noqa: BLE001
            manifest, valid, error = _safe_json(d / "manifest.json"), False, str(exc)
        latest_ptr = d.parent / "LATEST"
        spec = manifest.get("model_spec") or {}
        out.append({
            "path": str(d), "name": d.name, "step": manifest.get("step"), "algorithm": manifest.get("algorithm", ""),
            "created_at": manifest.get("created_at"), "valid": valid, "error": error,
            "size_bytes": sum(int(f.get("size", 0)) for f in (manifest.get("files") or {}).values()),
            "files": sorted((manifest.get("files") or {}).keys()),
            "is_latest": latest_ptr.exists() and latest_ptr.read_text().strip() == d.name,
            "is_best": d.name == "best", "model": {k: spec.get(k) for k in ("n_layer", "n_head", "n_kv_head", "n_embd", "vocab_size", "block_size") if k in spec},
        })
    return sorted(out, key=lambda c: (c["step"] is None, c["step"] or 0))


def _safe_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {}


# ── evaluation results ───────────────────────────────────────────────────────

def list_evaluations(root: str | Path, max_depth: int = 3) -> List[Dict[str, Any]]:
    """Files written by ``save_results`` (a list of ``{suite, metrics, n_samples, ...}``)."""
    out: List[Dict[str, Any]] = []
    root = Path(root)
    if not root.exists():
        return out
    for d in _walk(root, max_depth):
        for p in list(d.glob("*.json")) + list(d.glob("*.jsonl")):
            if p.name in ("manifest.json", "experiment.json", "metrics.jsonl", "events.jsonl") or p.stat().st_size > 20_000_000:
                continue
            try:
                from forgeline.evaluation.results import load_results

                results = load_results(p)
            except Exception:  # noqa: BLE001 - not an evaluation file
                continue
            if not results:
                continue
            out.append({"path": str(p), "results": [{"suite": r.suite, "metrics": r.metrics, "n_samples": r.n_samples,
                                                      "details": r.details} for r in results]})
    return out


# ── registry ─────────────────────────────────────────────────────────────────

def registry_view(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False, "models": {}}
    from forgeline.deployment.registry import CandidateRegistry

    reg = CandidateRegistry(p)
    models: Dict[str, List[Dict[str, Any]]] = {}
    for c in sorted(reg.list(), key=lambda c: c.created_at):
        models.setdefault(c.name, []).append(c.to_dict())
    return {"path": str(p), "exists": True, "models": models}


# ── benchmark evidence ───────────────────────────────────────────────────────

def benchmark_records(root: str | Path) -> List[Dict[str, Any]]:
    root = Path(root)
    out: List[Dict[str, Any]] = []
    if not root.exists():
        return out
    for results in sorted(root.rglob("results.json")):
        data = _safe_json(results)
        summary = results.parent / "summary.md"
        out.append({
            "record": str(results.parent.relative_to(root)), "evidence": data.get("evidence", ""),
            "revalidation_required": bool(data.get("revalidation_required", False)),
            "revalidation_recommended": bool(data.get("revalidation_recommended", False)),
            "published": data.get("published", True), "metrics": data.get("metrics", {}),
            "notes": data.get("notes", []), "summary": summary.read_text() if summary.exists() else "",
        })
    return out


# ── model inspection ─────────────────────────────────────────────────────────

def inspect_checkpoint(path: str | Path, text: str = "", max_tokens: int = 48, data_dir: Optional[str] = None) -> Dict[str, Any]:
    import torch

    from forgeline.evaluation.inspection import activation_flow, attention_patterns, layer_summary, weight_statistics
    from forgeline.models.loading import load_model_from_checkpoint, resolve_tokenizer_for_checkpoint

    model, spec, state = load_model_from_checkpoint(path, "cpu")
    stats = weight_statistics(model)
    n_params = sum(s["numel"] for s in stats)
    out: Dict[str, Any] = {
        "checkpoint": str(path), "step": state.step, "algorithm": state.algorithm, "spec": spec.to_dict(),
        "parameters": n_params, "layers": layer_summary(spec),
        "weights": [{k: v for k, v in s.items()} for s in stats],
    }
    if text:
        try:
            tok = resolve_tokenizer_for_checkpoint(state, data_dir)
            ids = torch.tensor([tok.encode(text)[-max_tokens:]], dtype=torch.long)
            out["tokens"] = [tok.decode([i]) for i in ids[0].tolist()]
            out["attention"] = attention_patterns(model, ids, max_tokens=max_tokens)
            out["activations"] = activation_flow(model, ids)
        except Exception as exc:  # noqa: BLE001
            out["inspection_error"] = f"{type(exc).__name__}: {exc}"
    return out


# ── aggregate ────────────────────────────────────────────────────────────────

@dataclass
class DashboardSources:
    runs: List[str] = field(default_factory=lambda: ["runs"])
    registry: str = "registry/candidates.json"
    benchmarks: str = "benchmarks"

    def checkpoint_allowed(self, path: str | Path) -> bool:
        target = Path(path).resolve()
        for root in self.runs:
            r = Path(root).resolve()
            if r == target or r in target.parents:
                return (target / "manifest.json").exists()
        return False


def snapshot(sources: DashboardSources, include_runs_detail: bool = True) -> Dict[str, Any]:
    runs = discover_runs(sources.runs)
    return {
        "generated_at": time.time(), "sources": {"runs": sources.runs, "registry": sources.registry, "benchmarks": sources.benchmarks},
        "runs": runs,
        "run_details": {r["id"]: run_detail(r["path"]) for r in runs} if include_runs_detail else {},
        "registry": registry_view(sources.registry),
        "benchmarks": benchmark_records(sources.benchmarks),
    }
