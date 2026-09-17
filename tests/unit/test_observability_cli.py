import json
import logging

from forgeline.cli.main import build_parser, main
from forgeline.observability.events import Event
from forgeline.observability.logging import get_logger
from forgeline.observability.metrics import LocalJsonlMetricsSink, NoOpMetricsSink, build_metrics_sink, gradient_norms
from forgeline.observability.tracing import Tracer


def test_local_metrics_sink(tmp_path):
    sink = LocalJsonlMetricsSink(tmp_path, "run")
    sink.log({"loss": 1.0, "nan": float("nan")}, step=1)
    sink.event(Event.RUN_STARTED.value, {"x": 1})
    sink.increment("errors")
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["loss"] == 1.0 and json.loads(lines[0])["nan"] is None
    assert json.loads((tmp_path / "events.jsonl").read_text())["event"] == "run.started"
    assert sink.latest("loss") == 1.0 and sink.series("loss") == [1.0] and sink.counters["errors"] == 1
    assert isinstance(build_metrics_sink("none"), NoOpMetricsSink)


def test_structured_logger(caplog):
    log = get_logger("forgeline.test")
    with caplog.at_level(logging.INFO, logger="forgeline"):
        log.info("hello", step=3, value=[1, 2])
    assert any("hello step=3" in r.getMessage() for r in caplog.records)


def test_tracer_and_grad_norms():
    import torch

    t = Tracer()
    with t.span("x"):
        pass
    assert t.summary()["x"]["count"] == 1
    lin = torch.nn.Linear(2, 2)
    lin(torch.ones(1, 2)).sum().backward()
    assert gradient_norms(lin)["grad_norm/total"] > 0


def test_cli_parser_and_presets(capsys):
    p = build_parser()
    args = p.parse_args(["train", "m.yaml", "--set", "trainer.max_steps=3", "--max-steps", "5"])
    assert args.set == ["trainer.max_steps=3"] and args.max_steps == 5
    assert main(["presets"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert any(r["preset"] == "tiny" for r in out)


def test_cli_validate_and_errors(tmp_path, capsys):
    from forgeline.core.config import ExperimentManifest, model_spec_from_preset

    ExperimentManifest(model=model_spec_from_preset("tiny")).save(tmp_path / "m.yaml")
    assert main(["validate", str(tmp_path / "m.yaml")]) == 0
    assert main(["export", "--checkpoint", str(tmp_path / "missing"), "--output", str(tmp_path / "x.gguf")]) == 1
    assert "error:" in capsys.readouterr().err
