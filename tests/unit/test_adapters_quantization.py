import pytest
import torch

from forgeline.core.config import AdapterConfig, model_spec_from_preset
from forgeline.core.errors import ModelError
from forgeline.data.tokenizers import CharTokenizer
from forgeline.models.adapters.lora import apply_lora, load_lora, merge_lora, save_lora, set_lora_enabled, trainable_parameter_report
from forgeline.models.adapters.qlora import apply_qlora
from forgeline.models.policy import NativePolicy
from forgeline.models.quantization.gguf import export_gguf, quantize_q4_0, quantize_q8_0, read_gguf_header, read_gguf_tensor, dequantize_q4_0, dequantize_q8_0
from forgeline.models.quantization.nf4 import NF4Linear, dequantize_nf4, quantization_report, quantize_nf4
from forgeline.models.transformer.model import TransformerLM

V = 40


def test_lora_identity_at_init_and_merge(tmp_path):
    m = TransformerLM(model_spec_from_preset("tiny", vocab_size=V)).eval()
    x = torch.randint(0, V, (1, 8))
    ref = m.full_logits(x)
    import copy
    base_copy = copy.deepcopy(m)
    apply_lora(m, rank=4, alpha=8)
    rep = trainable_parameter_report(m)
    assert 0 < rep["trainable"] < rep["total"]
    assert torch.allclose(m.full_logits(x), ref, atol=1e-5)  # B = 0 → identity
    for mod in m.modules():
        if hasattr(mod, "lora_B"):
            mod.lora_B.data.normal_()
    changed = m.full_logits(x)
    assert not torch.allclose(changed, ref)
    set_lora_enabled(m, False)
    assert torch.allclose(m.full_logits(x), ref, atol=1e-5)
    set_lora_enabled(m, True)
    n = save_lora(m, tmp_path / "lora.pt")
    assert n > 0
    m2 = base_copy
    apply_lora(m2, rank=4, alpha=8)
    load_lora(m2, tmp_path / "lora.pt")
    assert torch.allclose(m2.full_logits(x), changed, atol=1e-5)
    merge_lora(m)
    assert torch.allclose(m.full_logits(x), changed, atol=1e-4)
    with pytest.raises(ModelError):
        apply_lora(TransformerLM(model_spec_from_preset("tiny", vocab_size=V)), target_modules=["does_not_exist"])


def test_nf4_roundtrip_and_qlora():
    w = torch.randn(64, 96)
    packed, absmax, shape = quantize_nf4(w)
    assert packed.dtype == torch.uint8 and packed.numel() == w.numel() // 2
    back = dequantize_nf4(packed, absmax, shape)
    assert back.shape == w.shape and (back - w).abs().mean() < 0.15
    lin = torch.nn.Linear(96, 64)
    q = NF4Linear(lin)
    x = torch.randn(2, 96)
    assert (q(x) - lin(x)).abs().mean() < 0.5
    m = TransformerLM(model_spec_from_preset("tiny", vocab_size=V))
    apply_qlora(m, rank=4)
    rep = quantization_report(m)
    assert rep["nf4_layers"] > 0 and rep["compression_ratio"] > 3
    _, loss = m(torch.randint(0, V, (2, 8)), torch.randint(0, V, (2, 8)))
    loss.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_policy_reference_with_adapter(tokenizer):
    pol = NativePolicy(TransformerLM(model_spec_from_preset("tiny", vocab_size=tokenizer.vocab_size)), tokenizer,
                       adapter=AdapterConfig(method="lora", rank=2))
    assert pol.has_adapter
    with pol.reference():
        pass
    assert len(pol.trainable_parameters()) > 0


def test_gguf_export_roundtrip(tmp_path):
    spec = model_spec_from_preset("tiny", vocab_size=V, block_size=16)
    m = TransformerLM(spec)
    for fmt in ("none", "q8_0", "q4_0"):
        info = export_gguf(m.state_dict(), spec, tmp_path / f"m-{fmt}.gguf", quantize=fmt)
        header = read_gguf_header(tmp_path / f"m-{fmt}.gguf")
        assert header["metadata"]["forgeline.block_count"] == spec.n_layer and len(header["tensors"]) == info["tensors"]
        w = m.state_dict()["transformer.h.0.attn.q_proj.weight"].numpy()
        back = read_gguf_tensor(tmp_path / f"m-{fmt}.gguf", "transformer.h.0.attn.q_proj.weight")
        assert back.shape == w.shape
        tol = {"none": 1e-2, "q8_0": 0.05, "q4_0": 0.2}[fmt]
        assert abs(back - w).mean() < tol
    # offsets are strictly increasing and aligned
    offs = [t["offset"] for t in header["tensors"]]
    assert offs == sorted(offs) and all(o % 32 == 0 for o in offs)


def test_block_quantizers_roundtrip():
    x = torch.randn(64)
    d8, _, n = quantize_q8_0(x)
    assert abs(dequantize_q8_0(d8, n) - x.numpy()).max() < 0.05
    d4, _, n = quantize_q4_0(x)
    assert abs(dequantize_q4_0(d4, n) - x.numpy()).mean() < 0.3
