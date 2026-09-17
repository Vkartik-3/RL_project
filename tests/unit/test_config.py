import pytest

from forgeline.core.config import (
    DistributedConfig, ExperimentManifest, ModelSpec, MODEL_PRESETS, TrainerConfig, model_spec_from_preset,
)
from forgeline.core.errors import ConfigError, UnknownPresetError


def test_presets_build():
    for name in MODEL_PRESETS:
        spec = model_spec_from_preset(name)
        assert spec.n_embd % spec.n_head == 0


def test_unknown_preset():
    with pytest.raises(UnknownPresetError):
        model_spec_from_preset("nope")


@pytest.mark.parametrize("bad", [dict(n_embd=65, n_head=2), dict(n_head=4, n_kv_head=3), dict(rope_scaling_type="x"),
                                 dict(use_moe=True, n_experts=2, n_experts_active=3), dict(use_mla=True, use_nsa=True)])
def test_spec_validation(bad):
    with pytest.raises(ConfigError):
        model_spec_from_preset("tiny", **bad)


def test_unknown_spec_key():
    with pytest.raises(ConfigError):
        ModelSpec.from_dict({"n_layers": 2})


def test_manifest_roundtrip_yaml_toml_json(tmp_path):
    m = ExperimentManifest(name="x", algorithm="sft", model=model_spec_from_preset("tiny"), algorithm_params={"beta": 0.1})
    for ext in ("yaml", "toml", "json"):
        p = m.save(tmp_path / f"m.{ext}")
        back = ExperimentManifest.load(p)
        assert back.name == "x" and back.algorithm == "sft" and back.model.n_layer == 2 and back.algorithm_params["beta"] == 0.1


def test_manifest_rejects_unknown_algorithm_and_keys(tmp_path):
    with pytest.raises(ConfigError):
        ExperimentManifest.from_dict({"algorithm": "magic"})
    with pytest.raises(ConfigError):
        ExperimentManifest.from_dict({"trainer": {"max_stepz": 1}})


def test_trainer_and_distributed_validation():
    with pytest.raises(ConfigError):
        TrainerConfig(batch_size=0).validate()
    with pytest.raises(ConfigError):
        DistributedConfig(strategy="deepspeed").validate()
    with pytest.raises(ConfigError):
        DistributedConfig(strategy="ring").validate()


def test_runtime_environment_serialisable(tmp_path):
    m = ExperimentManifest(model=model_spec_from_preset("tiny")).with_runtime_environment()
    assert "torch" in m.runtime
    m.save(tmp_path / "m.yaml")
