import pytest

from forgeline.core.errors import ConfigError, PromotionError, RegistryError
from forgeline.deployment.candidates import CandidateState, ModelCandidate
from forgeline.deployment.feature_flags import FeatureFlag, FeatureFlags, stable_bucket
from forgeline.deployment.promotion import GateRule, PromotionGate
from forgeline.deployment.registry import CandidateRegistry
from forgeline.deployment.rollback import disable_candidate, rollback
from forgeline.deployment.rollout import RoutingPolicy, offline_replay, simulate_traffic
from forgeline.serving.backends import EchoBackend


def gate():
    return PromotionGate([GateRule("reward", min_delta=0.05), GateRule("accuracy", min=0.8), GateRule("safety_regression", max=0.01),
                          GateRule("latency_ms", max_increase=0.1, relative=True)])


def test_gate_succeeds_and_rejects():
    base = {"reward": 0.5, "accuracy": 0.85, "safety_regression": 0.0, "latency_ms": 100}
    good = {"reward": 0.56, "accuracy": 0.9, "safety_regression": 0.005, "latency_ms": 105}
    assert gate().evaluate(good, base).passed
    assert gate().assert_promotable(good, base).passed
    worse = dict(good, accuracy=0.7)
    rep = gate().evaluate(worse, base)
    assert not rep.passed and rep.failures()[0].metric == "accuracy"
    with pytest.raises(PromotionError):
        gate().assert_promotable(dict(good, latency_ms=130), base)


def test_gate_boundaries_missing_nan_and_config():
    base = {"reward": 0.5, "accuracy": 0.85, "safety_regression": 0.0, "latency_ms": 100}
    boundary = {"reward": 0.55, "accuracy": 0.8, "safety_regression": 0.01, "latency_ms": 110}
    assert gate().evaluate(boundary, base).passed  # inclusive bounds
    assert not gate().evaluate(dict(boundary, reward=0.549), base).passed
    assert not gate().evaluate({k: v for k, v in boundary.items() if k != "accuracy"}, base).passed  # missing
    assert not gate().evaluate(dict(boundary, reward=float("nan")), base).passed  # NaN
    assert not gate().evaluate(boundary, None).passed  # relative rule with no baseline fails closed
    with pytest.raises(ConfigError):
        GateRule("x").validate()
    with pytest.raises(ConfigError):
        PromotionGate([])
    with pytest.raises(ConfigError):
        PromotionGate.from_config({"rules": []})
    g = PromotionGate.from_config({"rules": [{"metric": "reward", "min_delta": 0.1}], "when_no_baseline": "skip_relative"})
    assert g.evaluate({"reward": 0.1}, None).passed and not g.evaluate({"reward": 0.1}, {"reward": 0.05}).passed


def test_registry_lifecycle(tmp_path):
    reg = CandidateRegistry(tmp_path / "reg.json")
    a = reg.register(ModelCandidate("m", "v1", "sft", "ck/a", metrics={"reward": 0.5}))
    with pytest.raises(RegistryError):
        reg.register(ModelCandidate("m", "v1", "sft", "ck/a"))
    with pytest.raises(RegistryError):
        reg.transition(a.id, CandidateState.CHAMPION)  # experimental → champion is not allowed
    reg.transition(a.id, CandidateState.CHALLENGER)
    reg.transition(a.id, CandidateState.CHAMPION)
    b = reg.register(ModelCandidate("m", "v2", "dpo", "ck/b"))
    reg.transition(b.id, CandidateState.SHADOW)
    reg.transition(b.id, CandidateState.CHALLENGER)
    reg.transition(b.id, CandidateState.CHAMPION)
    assert reg.champion("m").id == b.id and reg.get(a.id).state == CandidateState.RETIRED and reg.get(b.id).rollback_target == a.id
    reg2 = CandidateRegistry(tmp_path / "reg.json")  # persisted
    assert reg2.champion("m").id == b.id and len(reg2.list()) == 2
    reg2.record_metrics(b.id, {"accuracy": 0.9})
    assert reg2.get(b.id).metrics["accuracy"] == 0.9
    with pytest.raises(RegistryError):
        reg2.get("nope")


def test_rollback_and_disable(tmp_path):
    reg = CandidateRegistry(tmp_path / "reg.json")
    a = reg.register(ModelCandidate("m", "v1", "sft", "ck/a"))
    with pytest.raises(RegistryError):
        rollback(reg, "m")
    reg.transition(a.id, CandidateState.CHALLENGER); reg.transition(a.id, CandidateState.CHAMPION)
    with pytest.raises(RegistryError):
        rollback(reg, "m")  # no rollback target
    b = reg.register(ModelCandidate("m", "v2", "dpo", "ck/b"))
    reg.transition(b.id, CandidateState.CHALLENGER); reg.transition(b.id, CandidateState.CHAMPION)
    pol = RoutingPolicy(champion="m:v2", challenger="m:v1", challenger_percent=10)
    restored = rollback(reg, "m", "regression", routing=pol)
    assert restored.id == a.id and reg.champion("m").id == a.id and reg.get(b.id).state == CandidateState.RETIRED
    assert pol.champion == "m:v1" and "m:v2" in pol.disabled and pol.challenger is None
    c = disable_candidate(reg, a.id, routing=pol)
    assert "disabled" in c.tags and "m:v1" in pol.disabled


def test_routing_deterministic_and_percentages():
    pol = RoutingPolicy(champion="A", challenger="B", challenger_percent=10, shadow="C", shadow_percent=20)
    assert pol.route("req-1") == pol.route("req-1")
    frac = simulate_traffic(pol, [f"r{i}" for i in range(20000)])
    assert abs(frac["B"] - 0.10) < 0.01 and abs(frac["A"] - 0.90) < 0.01 and abs(frac["shadow_fraction"] - 0.20) < 0.015
    pol.disable_backend("B")
    assert simulate_traffic(pol, [f"r{i}" for i in range(500)]).get("B", 0) == 0
    pol.enable_backend("B")
    pol.pinned["req-x"] = "B"
    assert pol.route("req-x").primary == "B"
    assert pol.route("anything", requested="B").primary == "B"
    with pytest.raises(ConfigError):
        RoutingPolicy(champion="A", challenger_percent=10).validate()
    assert stable_bucket("k", "s") == stable_bucket("k", "s") and 0 <= stable_bucket("k") < 10_000
    assert RoutingPolicy.from_dict(pol.to_dict()).route("r5") == pol.route("r5")


def test_feature_flags(tmp_path):
    flags = FeatureFlags(tmp_path / "flags.json")
    flags.set(FeatureFlag("new_decoder", percentage=50, allow={"vip"}, deny={"banned"}))
    assert flags.is_on("new_decoder", "vip") and not flags.is_on("new_decoder", "banned")
    on = sum(flags.is_on("new_decoder", f"u{i}") for i in range(2000))
    assert 900 < on < 1100
    assert flags.is_on("new_decoder", "u1") == FeatureFlags(tmp_path / "flags.json").is_on("new_decoder", "u1")
    flags.disable("new_decoder")
    assert not flags.is_on("new_decoder", "vip") and flags.is_on("missing", default=True)
    with pytest.raises(ConfigError):
        FeatureFlag("x", percentage=120).validate()


def test_offline_replay():
    res = offline_replay({"a": EchoBackend("a"), "b": EchoBackend("b")}, ["hello", "world"], scorer=lambda p, out: float(len(out)))
    assert res.n == 2 and set(res.per_backend) == {"a", "b"}
