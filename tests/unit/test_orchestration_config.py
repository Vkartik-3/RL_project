"""Orchestration settings validate without Ray installed; the core never imports Ray."""

import subprocess
import sys

import pytest

from forgeline.core.config import ExperimentManifest
from forgeline.core.errors import ConfigError
from forgeline.orchestration import validate_orchestration
from forgeline.orchestration.config import RayConfig


def test_ray_config_validation():
    RayConfig.from_dict({"rollout_workers": 2, "reward_workers": 2, "placement_strategy": "SPREAD"})
    with pytest.raises(ConfigError):
        RayConfig.from_dict({"address": "auto", "num_cpus": 4})
    with pytest.raises(ConfigError):
        RayConfig.from_dict({"placement_strategy": "ANYWHERE"})
    with pytest.raises(ConfigError):
        RayConfig.from_dict({"rollout_workers": -1})
    with pytest.raises(ConfigError):
        RayConfig.from_dict({"workers": 2})


def test_manifest_orchestration_field_roundtrip(tmp_path):
    m = ExperimentManifest.from_dict({"algorithm": "grpo", "orchestration": {"type": "ray", "rollout_workers": 2}})
    path = m.save(tmp_path / "m.yaml")
    assert ExperimentManifest.load(path).orchestration == {"type": "ray", "rollout_workers": 2}
    with pytest.raises(ConfigError):
        validate_orchestration({"type": "kubernetes"})
    with pytest.raises(ConfigError):
        ExperimentManifest.from_dict({"algorithm": "grpo", "orchestration": {"type": "ray", "bogus": 1}})


def test_core_import_does_not_import_ray():
    code = ("import sys, forgeline.training, forgeline.cli.main, forgeline.core.config, forgeline.orchestration; "
            "print('ray' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
