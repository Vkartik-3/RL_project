import pytest
import torch

from forgeline.core.config import model_spec_from_preset
from forgeline.models.generation.sampling import apply_sampling_filters, generate_tokens
from forgeline.models.generation.speculative import MTPSpeculativeGenerator, SpeculativeGenerator
from forgeline.models.transformer.model import TransformerLM

V = 50


@pytest.mark.parametrize("kw", [
    {}, dict(n_kv_head=1), dict(use_mla=True, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16),
    dict(use_mla=True, q_lora_rank=16, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16),
    dict(use_moe=True, n_experts=4, n_experts_active=2, n_shared_experts=1, aux_loss_free=True, score_func="sigmoid"),
    dict(use_moe=True, n_experts=4, n_experts_active=2, n_expert_groups=2, n_limited_groups=1),
    dict(use_nsa=True, nsa_block_size=4, nsa_top_k=4, nsa_window_size=8), dict(n_predict_tokens=3),
    dict(sliding_window=8, alternating_layers=True, attn_logit_cap=30.0), dict(sliding_window=8),
    dict(rope_scaling_type="yarn", rope_scaling_factor=2.0), dict(rope_scaling_type="linear", rope_scaling_factor=2.0),
    dict(use_rope=False), dict(use_swiglu=False),
])
def test_variants_forward_backward(kw):
    spec = model_spec_from_preset("tiny", vocab_size=V, block_size=32, **kw)
    m = TransformerLM(spec)
    x = torch.randint(0, V, (2, 16))
    logits, loss = m(x, x)
    assert logits.shape == (2, 16, V) and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    if not kw.get("use_nsa"):
        out = m.generate(x[:, :4], 5, temperature=0.0)
        assert out.shape == (2, 9)


def test_cached_matches_uncached_greedy():
    m = TransformerLM(model_spec_from_preset("tiny", vocab_size=V, block_size=32)).eval()
    x = torch.randint(0, V, (1, 5))
    a = m.generate(x, 8, temperature=0.0, use_cache=True)
    b = m.generate(x, 8, temperature=0.0, use_cache=False)
    assert torch.equal(a, b)


def test_mla_cached_matches_uncached():
    spec = model_spec_from_preset("tiny", vocab_size=V, block_size=32, use_mla=True, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16)
    m = TransformerLM(spec).eval()
    x = torch.randint(0, V, (1, 5))
    assert torch.equal(m.generate(x, 6, temperature=0.0, use_cache=True), m.generate(x, 6, temperature=0.0, use_cache=False))


def test_sampling_filters():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    top1 = apply_sampling_filters(logits, top_k=1)
    assert torch.isinf(top1[0, :3]).all() and top1[0, 3] == 4.0
    nucleus = apply_sampling_filters(logits, top_p=0.5)
    assert torch.isinf(nucleus[0, 0])
    minp = apply_sampling_filters(logits, min_p=0.5)
    assert torch.isinf(minp[0, 0])
    rep = apply_sampling_filters(logits, repetition_penalty=2.0, past_tokens=[3])
    assert rep[0, 3] == 2.0


def test_generate_returns_logprobs_and_stops():
    m = TransformerLM(model_spec_from_preset("tiny", vocab_size=V, block_size=32)).eval()
    x = torch.randint(0, V, (2, 3))
    out, lp = generate_tokens(m, x, 6, temperature=1.0, return_logprobs=True)
    assert lp.shape == (2, out.shape[1] - 3) and (lp <= 0).all()
    tok = int(m.generate(x[:1], 1, temperature=0.0)[0, -1])
    stopped = m.generate(x[:1], 10, temperature=0.0, stop_token_ids=(tok,))
    assert stopped.shape[1] == 4


def test_speculative_and_mtp_generators():
    target = TransformerLM(model_spec_from_preset("tiny", vocab_size=V, block_size=64)).eval()
    draft = TransformerLM(model_spec_from_preset("tiny", vocab_size=V, block_size=64, n_layer=1)).eval()
    x = torch.randint(0, V, (1, 4))
    out = SpeculativeGenerator(target, draft, k=3).generate(x, 10, temperature=1.0)
    assert out.shape[1] >= 4 + 10
    mtp = TransformerLM(model_spec_from_preset("tiny", vocab_size=V, block_size=64, n_predict_tokens=3)).eval()
    out = MTPSpeculativeGenerator(mtp).generate(x, 8)
    assert out.shape[1] >= 4 + 8


def test_describe_and_param_counts():
    spec = model_spec_from_preset("tiny", vocab_size=V, use_moe=True, n_experts=4, n_experts_active=1)
    m = TransformerLM(spec)
    assert m.num_active_parameters() < m.num_parameters()
    assert "MoE" in m.describe()
    assert TransformerLM(model_spec_from_preset("tiny", vocab_size=V, n_kv_head=1)).describe().startswith("0.")
