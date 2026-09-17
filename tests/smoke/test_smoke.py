"""Fast import + wiring checks for the whole package."""

import importlib
import pkgutil

import forgeline


def test_every_module_imports():
    """Every module imports; modules that wrap an optional extra fail only with the actionable dependency error."""
    from forgeline.core.errors import OptionalDependencyError

    failures = []
    for mod in pkgutil.walk_packages(forgeline.__path__, prefix="forgeline."):
        try:
            importlib.import_module(mod.name)
        except OptionalDependencyError as exc:
            if "pip install" not in str(exc):  # pragma: no cover
                failures.append((mod.name, repr(exc)))
        except Exception as exc:  # pragma: no cover
            failures.append((mod.name, repr(exc)))
    assert not failures, failures


def test_ray_backend_reports_missing_extra():
    import pytest

    from forgeline.orchestration import ray_available

    if ray_available():
        pytest.skip("ray is installed")
    from forgeline.core.errors import OptionalDependencyError

    with pytest.raises(OptionalDependencyError, match=r"\.\[ray\]"):
        importlib.import_module("forgeline.orchestration.ray_backend")


def test_version_and_public_api():
    assert forgeline.__version__
    from forgeline.core import ExperimentManifest, RunContext  # noqa: F401
    from forgeline.deployment import CandidateRegistry, PromotionGate, RoutingPolicy  # noqa: F401
    from forgeline.inference import InferenceEngine  # noqa: F401
    from forgeline.training import DAPOAlgorithm, DPOAlgorithm, GRPOAlgorithm, PPOAlgorithm, Trainer  # noqa: F401


def test_hf_policy_reports_missing_extra():
    import pytest

    try:
        import transformers  # noqa: F401
        import peft  # noqa: F401
    except ImportError:
        from forgeline.core.errors import OptionalDependencyError
        from forgeline.models.policy import HFPolicy

        with pytest.raises(OptionalDependencyError):
            HFPolicy("any-model")
