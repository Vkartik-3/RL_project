"""train N steps → save → reload → continue produces the same trajectory as uninterrupted training."""

import torch

from forgeline.core.config import OptimizerConfig
from forgeline.data.pretraining import MemmapCorpus
from forgeline.data.records import SFTRecord
from forgeline.data.supervised import SupervisedDataset
from forgeline.models.policy import NativePolicy
from forgeline.models.transformer.model import TransformerLM
from forgeline.training import PretrainAlgorithm, PretrainConfig, SFTAlgorithm, SFTConfig, Trainer
from forgeline.core.config import AdapterConfig
from forgeline.core.runtime import seed_everything


def _pretrain(ctx, spec, corpus_dir):
    return PretrainAlgorithm(TransformerLM(spec), MemmapCorpus(corpus_dir, "train", 64), PretrainConfig(batch_size=4, seed=7))


def test_resume_matches_uninterrupted(ctx_factory, tiny_spec, corpus_dir, tmp_path):
    seed_everything(0)
    full_ctx = ctx_factory("pretrain", max_steps=6, eval_every=0, save_every=0)
    ref_alg = _pretrain(full_ctx, tiny_spec, corpus_dir)
    init_state = {k: v.clone() for k, v in ref_alg.model.state_dict().items()}
    ref = Trainer(full_ctx, ref_alg)
    for _ in range(3):
        ref.train_step()
    ckpt = ref.save_checkpoint(name="mid")
    ref_losses = [ref.train_step().loss for _ in range(3)]

    ctx2 = ctx_factory("pretrain", run_name="pretrain_resume", max_steps=6, eval_every=0, save_every=0)
    alg2 = _pretrain(ctx2, tiny_spec, corpus_dir)
    trainer2 = Trainer(ctx2, alg2)
    assert trainer2.resume(str(ckpt)) and trainer2.step == 3
    # data sampling uses the algorithm's own generator; restore it to the same point
    alg2.gen = torch.Generator().manual_seed(7)
    probe = _pretrain(ctx2, tiny_spec, corpus_dir)
    for _ in range(3):
        probe.collect(0)
    alg2.gen.set_state(probe.gen.get_state())
    resumed = [trainer2.train_step().loss for _ in range(3)]
    assert all(abs(a - b) < 1e-5 for a, b in zip(ref_losses, resumed)), (ref_losses, resumed)


def test_auto_resume_from_latest_and_adapter_state(ctx_factory, tiny_spec, tokenizer):
    recs = [SFTRecord(f"Q{i}", f" {i}") for i in range(8)]
    ctx = ctx_factory("sft", max_steps=4, eval_every=0, save_every=2)
    pol = NativePolicy(TransformerLM(tiny_spec), tokenizer, adapter=AdapterConfig(method="lora", rank=2))
    t = Trainer(ctx, SFTAlgorithm(pol, SupervisedDataset(recs, tokenizer, 16), SFTConfig(batch_size=2)))
    t.fit()
    lora_before = {n: p.detach().clone() for n, p in pol.model.named_parameters() if "lora_" in n}
    pol2 = NativePolicy(TransformerLM(tiny_spec), tokenizer, adapter=AdapterConfig(method="lora", rank=2))
    t2 = Trainer(ctx, SFTAlgorithm(pol2, SupervisedDataset(recs, tokenizer, 16), SFTConfig(batch_size=2)))
    assert t2.resume() and t2.step == 4
    for n, p in pol2.model.named_parameters():
        if "lora_" in n:
            assert torch.allclose(p, lora_before[n])


def test_compile_model_matches_eager_and_checkpoints_load(ctx_factory, tiny_spec, corpus_dir):
    from forgeline.core.config import OptimizerConfig
    from forgeline.models.loading import load_model_from_checkpoint

    losses = {}
    for compiled in (False, True):
        torch.manual_seed(0)
        ctx = ctx_factory("pretrain", run_name=f"compile-{compiled}", max_steps=3, batch_size=4, eval_every=0, save_every=0,
                          compile_model=compiled, compile_backend="eager", optimizer=OptimizerConfig(learning_rate=1e-3))
        model = TransformerLM(tiny_spec)
        alg = PretrainAlgorithm(model, MemmapCorpus(corpus_dir, "train", 64), PretrainConfig(batch_size=4, seed=0))
        trainer = Trainer(ctx, alg)
        trainer.fit()
        losses[compiled] = [h.loss for h in trainer.history]
        if compiled:
            assert model._compiled_call_impl is not None
            reloaded, _, _ = load_model_from_checkpoint(ctx.output_dir / "ckpt" / "final")
            x = torch.randint(0, tiny_spec.vocab_size, (1, 8))
            assert torch.allclose(reloaded(x)[0], model(x)[0], atol=1e-5)
    assert losses[True] == losses[False]


def test_merge_on_save_writes_plain_model_checkpoint(ctx_factory, tiny_spec, tokenizer):
    from forgeline.core.config import AdapterConfig, OptimizerConfig
    from forgeline.data.records import SFTRecord
    from forgeline.data.supervised import SupervisedDataset
    from forgeline.models.loading import load_model_from_checkpoint
    from forgeline.models.policy import NativePolicy
    from forgeline.training import SFTAlgorithm, SFTConfig

    adapter = AdapterConfig(method="lora", rank=4, merge_on_save=True)
    ctx = ctx_factory("sft", run_name="merge", max_steps=3, eval_every=0, save_every=0, adapter=adapter,
                      optimizer=OptimizerConfig(learning_rate=1e-2))
    policy = NativePolicy(TransformerLM(tiny_spec), tokenizer, adapter=adapter)
    recs = [SFTRecord(f"Q: {i}+1?", f" {i+1}") for i in range(8)]
    Trainer(ctx, SFTAlgorithm(policy, SupervisedDataset(recs, tokenizer, 32), SFTConfig(batch_size=4))).fit()
    merged_dir = ctx.output_dir / "ckpt" / "final_merged"
    assert (merged_dir / "manifest.json").exists()
    merged, _, state = load_model_from_checkpoint(merged_dir)
    assert state.metadata == {"merged_adapter": True}
    assert not any("lora" in k for k in merged.state_dict())
    x = torch.randint(0, tiny_spec.vocab_size, (2, 10))
    policy.model.eval()
    assert torch.allclose(merged(x)[0], policy.model(x)[0], atol=1e-5)
