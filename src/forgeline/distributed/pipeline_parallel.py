"""Pipeline parallelism: stage extraction and a 1F1B micro-batch schedule."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

from forgeline.core.errors import DistributedConfigError
from forgeline.distributed.tensor_parallel import apply_tensor_parallel
from forgeline.distributed.topology import ParallelMesh


class PipelineStage(nn.Module):
    """A contiguous slice of transformer blocks; first stage embeds, last stage has the head."""

    def __init__(self, layers: nn.ModuleList, pp_rank: int, pp_size: int, embed: Optional[nn.Module] = None,
                 ln_f: Optional[nn.Module] = None, lm_head: Optional[nn.Module] = None):
        super().__init__()
        self.layers, self.pp_rank, self.pp_size = layers, pp_rank, pp_size
        self.is_first, self.is_last = pp_rank == 0, pp_rank == pp_size - 1
        self.embed = embed if self.is_first else None
        self.ln_f = ln_f if self.is_last else None
        self.lm_head = lm_head if self.is_last else None

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        if self.is_first and self.embed is not None:
            x = self.embed(x)
        for layer in self.layers:
            x, _ = layer(x)
        if self.is_last:
            if self.ln_f is not None:
                x = self.ln_f(x)
            if self.lm_head is not None:
                logits = self.lm_head(x)
                loss = None
                if targets is not None:
                    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
                return logits, loss
        return x, None


def build_pipeline_stage(model, mesh: ParallelMesh) -> PipelineStage:
    n_layers = len(model.transformer.h)
    if n_layers % mesh.pp_size != 0:
        raise DistributedConfigError(f"n_layer {n_layers} must be divisible by pipeline_parallel {mesh.pp_size}")
    per = n_layers // mesh.pp_size
    start = mesh.pp_rank * per
    layers = nn.ModuleList(list(model.transformer.h)[start : start + per])
    if mesh.tp_size > 1:
        for layer in layers:
            apply_tensor_parallel(layer, mesh)
    return PipelineStage(layers, mesh.pp_rank, mesh.pp_size, embed=model.transformer.wte,
                         ln_f=model.transformer.ln_f, lm_head=model.lm_head)


def one_f_one_b_schedule(n_micro: int, pp_rank: int, pp_size: int) -> List[Tuple[str, int]]:
    """The 1F1B action list for one rank: ``("F", i)`` / ``("B", i)`` over micro-batch indices."""
    n_warmup = min(pp_size - pp_rank - 1, n_micro)
    actions: List[Tuple[str, int]] = [("F", i) for i in range(n_warmup)]
    fwd = n_warmup
    bwd = 0
    for _ in range(n_micro - n_warmup):
        actions.append(("F", fwd)); fwd += 1
        actions.append(("B", bwd)); bwd += 1
    while bwd < n_micro:
        actions.append(("B", bwd)); bwd += 1
    return actions


class PipelineScheduler:
    """Executes the 1F1B schedule; point-to-point sends/receives between pipeline neighbours."""

    def __init__(self, stage: PipelineStage, mesh: ParallelMesh, n_micro: int = 4):
        self.stage, self.mesh, self.n_micro = stage, mesh, n_micro

    def _recv(self, like: torch.Tensor, device) -> torch.Tensor:
        shape = (like.shape[0], like.shape[1], self.stage.layers[0].ln_1.weight.numel())
        t = torch.empty(shape, dtype=torch.float32, device=device)
        dist.recv(t, src=self.mesh.prev_pipeline_rank, group=self.mesh.pp_group)
        return t.requires_grad_(True)

    def _send(self, t: torch.Tensor) -> None:
        dist.send(t.detach().contiguous(), dst=self.mesh.next_pipeline_rank, group=self.mesh.pp_group)

    def _forward(self, x, y, device):
        inp = x if self.mesh.pp_rank == 0 else self._recv(x, device)
        out = self.stage(inp, y)
        if self.mesh.pp_rank < self.mesh.pp_size - 1:
            self._send(out[0])
        return out

    def run(self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.stage.parameters()).device
        xs = inputs.chunk(self.n_micro)
        ys = targets.chunk(self.n_micro) if targets is not None else [None] * self.n_micro
        outputs: dict[int, Any] = {}
        total = torch.zeros((), device=device)
        for action, i in one_f_one_b_schedule(self.n_micro, self.mesh.pp_rank, self.mesh.pp_size):
            if action == "F":
                outputs[i] = self._forward(xs[i], ys[i], device)
            else:
                out = outputs.pop(i)
                if isinstance(out, tuple) and out[1] is not None:
                    loss = out[1] / self.n_micro
                    loss.backward()
                    total += loss.detach()
        return total
