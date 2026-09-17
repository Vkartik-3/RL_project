"""Dashboard data layer, HTTP API and CLI against real run outputs (CPU)."""

import json
import math
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch

from forgeline.cli.main import main
from forgeline.dashboard.data import DashboardSources, discover_runs, list_checkpoints, run_detail, snapshot
from forgeline.dashboard.server import DashboardApp, DashboardServer
from forgeline.deployment.candidates import CandidateState, ModelCandidate
from forgeline.deployment.registry import CandidateRegistry
from forgeline.evaluation.inspection import attention_patterns

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dash")
    corpus = tmp / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog\n" * 200)
    assert main(["data", "prepare", "--input", str(corpus), "--output", str(tmp / "prepared"), "--tokenizer", "char"]) == 0
    assert main(["train", str(ROOT / "configs/training/pretrain_tiny_cpu.yaml"), "--max-steps", "4", "--output-dir", str(tmp / "runs/pre"),
                 "--set", f"data.path={tmp / 'prepared'}", "--set", "trainer.eval_every=2", "--set", "trainer.checkpoint.save_every=2"]) == 0
    reg = CandidateRegistry(tmp / "registry.json")
    c = reg.register(ModelCandidate("tiny", "v1", "pretrain", str(tmp / "runs/pre/checkpoints/final"), metrics={"val_loss": 3.2}))
    reg.transition(c.id, CandidateState.CHALLENGER, "offline eval")
    (tmp / "runs/pre/eval.json").write_text(json.dumps([{"suite": "benchmark/arc", "metrics": {"accuracy": 0.25}, "n_samples": 4,
                                                          "details": {"offline_samples": True}}]))
    return tmp


def sources(ws):
    return DashboardSources(runs=[str(ws / "runs")], registry=str(ws / "registry.json"), benchmarks=str(ROOT / "benchmarks"))


def test_run_discovery_and_detail(workspace):
    runs = discover_runs([workspace / "runs"])
    assert len(runs) == 1
    run = runs[0]
    assert run["algorithm"] == "pretrain" and run["status"] == "finished" and run["last_step"] >= 3
    detail = run_detail(run["path"])
    assert "eval/val_loss" in detail["series"] and all(math.isfinite(v) for _, v in detail["series"]["eval/val_loss"])
    assert any(e["event"] == "run.finished" for e in detail["events"])
    names = {c["name"] for c in detail["checkpoints"]}
    assert "final" in names and all(c["valid"] for c in detail["checkpoints"])
    assert detail["evaluations"][0]["results"][0]["metrics"] == {"accuracy": 0.25}


def test_truncated_checkpoint_is_reported_invalid(workspace, tmp_path):
    import shutil

    broken = tmp_path / "ckpts" / "final"
    shutil.copytree(workspace / "runs/pre/checkpoints/final", broken)
    with (broken / "model.pt").open("r+b") as f:
        f.truncate(10)
    [ckpt] = list_checkpoints(tmp_path / "ckpts")
    assert ckpt["valid"] is False and "truncated" in ckpt["error"]


def test_api_routes(workspace):
    app = DashboardApp(sources(workspace))
    status, ctype, html = app.handle("/")
    assert status == 200 and "text/html" in ctype and "<title>Forgeline Dashboard</title>" in html
    status, _, ov = app.handle("/api/overview")
    assert status == 200 and len(ov["runs"]) == 1 and ov["registry"]["models"]["tiny"][0]["state"] == "challenger"
    assert any(b["record"] == "training/fineweb_edu_large_a40" for b in ov["benchmarks"])
    status, _, detail = app.handle(f"/api/runs/{ov['runs'][0]['id']}")
    assert status == 200 and detail["series"]
    assert app.handle("/api/runs/unknown")[0] == 404
    status, _, benches = app.handle("/api/benchmarks")
    unpublished = [b for b in benches if b["published"] is False]
    assert unpublished and unpublished[0]["evidence"] == "documented_without_raw_log"


def test_inspection_is_restricted_to_run_roots(workspace, tmp_path):
    app = DashboardApp(sources(workspace))
    ckpt = workspace / "runs/pre/checkpoints/final"
    status, _, r = app.handle(f"/api/inspect?checkpoint={ckpt}&text=the%20quick")
    assert status == 200
    assert r["parameters"] == sum(w["numel"] for w in r["weights"])
    assert len(r["attention"]) == r["spec"]["n_layer"] and r["tokens"] == list("the quick")
    assert [a["name"] for a in r["activations"]][0] == "embedding"
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "manifest.json").write_text("{}")
    assert app.handle(f"/api/inspect?checkpoint={outside}")[0] == 403
    assert app.handle(f"/api/inspect?checkpoint={workspace / 'runs/../../'}")[0] == 403


@pytest.mark.parametrize("variant", [{}, {"sliding_window": 4, "alternating_layers": True, "attn_logit_cap": 20.0}])
def test_attention_patterns_reproduce_the_attention_output(tokenizer, variant):
    from forgeline.core.config import model_spec_from_preset
    from forgeline.models.transformer.model import TransformerLM

    torch.manual_seed(0)
    spec = model_spec_from_preset("tiny", vocab_size=tokenizer.vocab_size, block_size=64, **variant)
    model = TransformerLM(spec).eval()
    ids = torch.randint(0, spec.vocab_size, (1, 12))
    captured = {}
    hooks = []
    for i, block in enumerate(model.transformer.h):
        hooks.append(block.attn.register_forward_pre_hook(lambda m, a, k, i=i: captured.__setitem__(("x", i), a[0]), with_kwargs=True))
        hooks.append(block.attn.register_forward_hook(lambda m, inp, o, i=i: captured.__setitem__(("y", i), o[0])))
    model(ids)
    for h in hooks:
        h.remove()
    patterns = attention_patterns(model, ids, per_head=True)
    for i, block in enumerate(model.transformer.h):
        attn, x = block.attn, captured[("x", i)]
        w = torch.tensor(patterns[i]["head_weights"])  # [H, T, T]
        v = attn._repeat_kv(attn.v_proj(x).view(1, 12, attn.n_kv_head, attn.head_dim).transpose(1, 2))
        y = attn.c_proj((w.unsqueeze(0) @ v).transpose(1, 2).reshape(1, 12, spec.n_embd))
        assert torch.allclose(y, captured[("y", i)], atol=1e-5), f"layer {i}"
        assert torch.allclose(torch.tensor(patterns[i]["weights"]), w.mean(0), atol=1e-6)


def test_http_server_and_cli_export(workspace, tmp_path):
    server = DashboardServer(DashboardApp(sources(workspace)), port=0).start()
    try:
        with urllib.request.urlopen(f"{server.url}/api/runs") as r:
            assert json.loads(r.read())[0]["algorithm"] == "pretrain"
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"{server.url}/api/nope")
        assert err.value.code == 404
    finally:
        server.stop()
    out = tmp_path / "snap.json"
    assert main(["dashboard", "--runs", str(workspace / "runs"), "--registry", str(workspace / "registry.json"),
                 "--benchmarks", str(ROOT / "benchmarks"), "--export", str(out)]) == 0
    snap = json.loads(out.read_text())
    assert len(snap["runs"]) == 1 and snap["run_details"][snap["runs"][0]["id"]]["checkpoints"]
    assert snapshot(sources(workspace), include_runs_detail=False)["run_details"] == {}
