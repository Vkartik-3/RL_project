import pytest
import torch

from forgeline.checkpoints.manager import CheckpointManager, CheckpointState, load_model_state
from forgeline.checkpoints.validation import assert_compatible, compare_state_shapes, finite_state
from forgeline.core.config import model_spec_from_preset
from forgeline.core.errors import CheckpointCorruptError, CheckpointMismatchError, CheckpointNotFoundError, IncompatibleStateError
from forgeline.models.transformer.model import TransformerLM


def test_save_load_latest_rotation(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_last=2)
    m = TransformerLM(model_spec_from_preset("tiny", vocab_size=20))
    for step in (1, 2, 3):
        mgr.save(CheckpointState(model_state=m.state_dict(), step=step, algorithm="sft", model_spec=m.spec.to_dict(),
                                 optimizer_state={"x": step}, trainer_state={"best": 1.0}), is_best=(step == 2))
    names = [p.name for p in mgr.list()]
    assert "step_00000001" not in names and "step_00000003" in names and "best" in names
    assert mgr.find_latest().name == "step_00000003"
    state = CheckpointManager.load(mgr.find_latest(), expected_spec=m.spec.to_dict(), expected_algorithm="sft")
    assert state.step == 3 and state.optimizer_state == {"x": 3} and state.trainer_state["best"] == 1.0
    m2 = TransformerLM(m.spec)
    load_model_state(m2, state.model_state)
    assert torch.equal(m2.lm_head.weight, m.lm_head.weight)


def test_async_save(tmp_path):
    mgr = CheckpointManager(tmp_path, async_save=True)
    mgr.save(CheckpointState(model_state={"w": torch.zeros(2)}, step=1), name="a")
    mgr.wait()
    assert (tmp_path / "a" / "manifest.json").exists()


def test_compile_prefix_and_compat():
    m = TransformerLM(model_spec_from_preset("tiny", vocab_size=20))
    prefixed = {f"_orig_mod.{k}": v for k, v in m.state_dict().items()}
    load_model_state(TransformerLM(m.spec), prefixed)
    other = TransformerLM(model_spec_from_preset("tiny", vocab_size=30))
    rep = compare_state_shapes(other, m.state_dict())
    assert rep["mismatched"]
    with pytest.raises(IncompatibleStateError):
        assert_compatible(other, m.state_dict())
    with pytest.raises(IncompatibleStateError):
        load_model_state(other, m.state_dict())
    assert finite_state(m.state_dict()) and not finite_state({"w": torch.tensor([float("nan")])})
