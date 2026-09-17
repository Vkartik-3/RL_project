import math

import pytest
import torch

from forgeline.core.config import OptimizerConfig, ScheduleConfig
from forgeline.training.common.advantages import compute_gae, group_relative_advantages, normalize_advantages
from forgeline.training.common.logprobs import entropy_from_logits, kl_k3, kl_logprob_diff, masked_mean, reduce_logprobs
from forgeline.training.common.optim import build_optimizer
from forgeline.training.common.schedule import learning_rate_at
from forgeline.training.distill import distillation_loss
from forgeline.training.dpo import dpo_loss
from forgeline.training.ppo import ppo_clipped_objective
from forgeline.training.dapo import dapo_token_objective
from forgeline.models.reward import bradley_terry_loss


def test_cosine_and_wsd_schedules():
    cos = ScheduleConfig(name="cosine", warmup_steps=10, decay_steps=100, min_lr=0.1)
    assert learning_rate_at(0, 1.0, cos) == 0.0 and abs(learning_rate_at(10, 1.0, cos) - 1.0) < 1e-9
    assert abs(learning_rate_at(100, 1.0, cos) - 0.1) < 1e-9 and learning_rate_at(500, 1.0, cos) == 0.1
    wsd = ScheduleConfig(name="wsd", warmup_steps=10, decay_steps=100, min_lr=0.1, wsd_stable_fraction=0.8)
    assert learning_rate_at(50, 1.0, wsd) == 1.0 and abs(learning_rate_at(100, 1.0, wsd) - 0.1) < 1e-9
    assert learning_rate_at(5, 1.0, ScheduleConfig(name="constant", warmup_steps=0)) == 1.0
    lin = ScheduleConfig(name="linear_decay", warmup_steps=0, decay_steps=10, min_lr=0.0)
    assert abs(learning_rate_at(5, 1.0, lin) - 0.5) < 1e-9


def test_optimizer_groups():
    lin = torch.nn.Linear(4, 4)
    opt = build_optimizer(lin.parameters(), OptimizerConfig(weight_decay=0.1), torch.device("cpu"))
    assert len(opt.param_groups) == 2 and opt.param_groups[1]["weight_decay"] == 0.0


def test_gae_and_group_advantages():
    adv, ret = compute_gae(torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0, 0.0]), gamma=1.0, lam=1.0)
    assert adv.tolist() == [2.0, 1.0] and ret.tolist() == [2.0, 1.0]
    g = group_relative_advantages(torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert abs(float(g.mean())) < 1e-6 and abs(float(g.std()) - 1.0) < 1e-3
    assert group_relative_advantages(torch.tensor([1.0, 1.0])).abs().max() < 1e-3
    assert normalize_advantages(torch.tensor([3.0])).tolist() == [0.0]


def test_logprob_reductions_and_kl():
    lp = torch.tensor([[-1.0, -2.0, -3.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    assert reduce_logprobs(lp, mask, "sum").item() == -3.0 and reduce_logprobs(lp, mask, "mean").item() == -1.5
    with pytest.raises(ValueError):
        reduce_logprobs(lp, mask, "max")
    assert kl_logprob_diff(torch.zeros(1, 3), lp, mask).item() == 1.5
    assert kl_logprob_diff(lp, torch.zeros(1, 3), mask).item() == 0.0  # clamped
    assert kl_k3(lp, lp).item() == 0.0
    ent = entropy_from_logits(torch.zeros(1, 2, 4))
    assert torch.allclose(ent, torch.full((1, 2), math.log(4)))


def test_objectives():
    ratio = torch.tensor([0.5, 1.0, 2.0])
    adv = torch.tensor([1.0, 1.0, 1.0])
    loss = ppo_clipped_objective(ratio, adv, 0.2)
    assert loss.tolist() == pytest.approx([-0.5, -1.0, -1.2])
    tok = dapo_token_objective(torch.tensor([[2.0]]), torch.tensor([[1.0]]), 0.2, 0.28)
    assert tok.item() == pytest.approx(-1.28)
    out = dpo_loss(torch.tensor([0.0]), torch.tensor([-10.0]), torch.zeros(1), torch.zeros(1), beta=1.0)
    assert out["loss"].item() < 0.01 and out["accuracy"].item() == 1.0
    out = dpo_loss(torch.tensor([0.0]), torch.tensor([0.0]), torch.zeros(1), torch.zeros(1), beta=1.0)
    assert out["loss"].item() == pytest.approx(math.log(2))
    assert bradley_terry_loss(torch.tensor([5.0]), torch.tensor([-5.0])).item() < 0.01
    s = torch.randn(2, 3, 5)
    assert distillation_loss(s, s, torch.zeros(2, 3, dtype=torch.long), alpha=1.0).item() == pytest.approx(0.0, abs=1e-6)
