"""Knowledge distillation: ``α·T²·KL(teacher‖student) + (1−α)·CE`` plus optional feature matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from forgeline.data.pretraining import MemmapCorpus
from forgeline.models.transformer.model import TransformerLM
from forgeline.training.common.trainer import PostTrainingAlgorithm


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor,
                      temperature: float = 4.0, alpha: float = 0.7) -> torch.Tensor:
    V = student_logits.shape[-1]
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1).detach()
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_soft.view(-1, V), teacher_soft.view(-1, V), reduction="batchmean") * (temperature ** 2)
    ce = F.cross_entropy(student_logits.view(-1, V), labels.view(-1), ignore_index=-1)
    return alpha * kl + (1 - alpha) * ce


class FeatureDistiller(nn.Module):
    """Project student hidden states onto teacher hidden states and match with MSE."""

    def __init__(self, student_dim: int, teacher_dim: int, n_layers: int = 4):
        super().__init__()
        self.projectors = nn.ModuleList([nn.Linear(student_dim, teacher_dim, bias=False) for _ in range(n_layers)])

    def forward(self, student_hiddens: List[torch.Tensor], teacher_hiddens: List[torch.Tensor]) -> torch.Tensor:
        loss = torch.zeros(())
        n = min(len(self.projectors), len(student_hiddens), len(teacher_hiddens))
        for i in range(n):
            loss = loss + F.mse_loss(self.projectors[i](student_hiddens[i]), teacher_hiddens[i].detach())
        return loss / max(n, 1)


@dataclass
class DistillConfig:
    temperature: float = 4.0
    alpha: float = 0.7
    batch_size: int = 16
    feature_weight: float = 0.0
    eval_batches: int = 10
    seed: int = 0


class DistillationAlgorithm(PostTrainingAlgorithm):
    name = "distill"

    def __init__(self, student: TransformerLM, teacher: TransformerLM, corpus: MemmapCorpus, config: DistillConfig,
                 val_corpus: Optional[MemmapCorpus] = None, device: torch.device | str = "cpu"):
        if student.spec.vocab_size != teacher.spec.vocab_size:
            raise ValueError("student and teacher must share a vocabulary")
        self.student, self.teacher = student, teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.corpus, self.val_corpus, self.cfg = corpus, val_corpus, config
        self.device = torch.device(device)
        self.gen = torch.Generator().manual_seed(config.seed)
        self.feature = FeatureDistiller(student.spec.n_embd, teacher.spec.n_embd, min(student.spec.n_layer, teacher.spec.n_layer)) \
            if config.feature_weight > 0 else None

    def parameters(self) -> List[nn.Parameter]:
        params = list(self.student.parameters())
        if self.feature is not None:
            params += list(self.feature.parameters())
        return params

    def modules(self):
        return (self.student,)

    def collect(self, step: int):
        return self.corpus.sample_batch(self.cfg.batch_size, self.device, self.gen)

    def _hiddens(self, model: TransformerLM, x: torch.Tensor) -> List[torch.Tensor]:
        hs: List[torch.Tensor] = []
        handles = [b.register_forward_hook(lambda m, i, o: hs.append(o[0])) for b in model.transformer.h]
        try:
            model.full_logits(x)
        finally:
            for h in handles:
                h.remove()
        return hs

    def loss(self, batch, micro_step: int, n_micro: int):
        x, y = batch
        with torch.no_grad():
            t_logits = self.teacher.full_logits(x)
        s_logits = self.student.full_logits(x)
        loss = distillation_loss(s_logits, t_logits, y, self.cfg.temperature, self.cfg.alpha)
        metrics = {}
        if self.feature is not None:
            with torch.no_grad():
                th = self._hiddens(self.teacher, x)
            fl = self.feature(self._hiddens(self.student, x), th)
            loss = loss + self.cfg.feature_weight * fl
            metrics["feature_loss"] = float(fl.detach())
        return loss, metrics

    @torch.no_grad()
    def evaluate(self, step: int) -> Dict[str, float]:
        corpus = self.val_corpus or self.corpus
        losses = []
        for _ in range(self.cfg.eval_batches):
            x, y = corpus.sample_batch(self.cfg.batch_size, self.device, self.gen)
            losses.append(float(distillation_loss(self.student.full_logits(x), self.teacher.full_logits(x), y,
                                                  self.cfg.temperature, self.cfg.alpha)))
        return {"loss": sum(losses) / len(losses)}

    def state(self) -> Dict[str, Any]:
        return {"model": self.student.state_dict(), "teacher_spec": self.teacher.spec.to_dict()}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.student.load_state_dict(state["model"])

    def model_spec(self) -> Dict[str, Any]:
        return self.student.spec.to_dict()
