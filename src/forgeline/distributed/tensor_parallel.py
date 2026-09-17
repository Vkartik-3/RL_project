"""Megatron-style tensor parallelism: column/row parallel linear layers + model surgery."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from forgeline.distributed.topology import ParallelMesh


class _AllReduceFwdIdentityBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        if group is not None and dist.is_initialized() and dist.get_world_size(group) > 1:
            dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _IdentityFwdAllReduceBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        if ctx.group is not None and dist.is_initialized() and dist.get_world_size(ctx.group) > 1:
            dist.all_reduce(grad, group=ctx.group)
        return grad, None


def all_reduce_forward(x, group):
    return _AllReduceFwdIdentityBwd.apply(x, group)


def all_reduce_backward(x, group):
    return _IdentityFwdAllReduceBwd.apply(x, group)


class ColumnParallelLinear(nn.Module):
    """Output dimension split across TP ranks: ``Y_i = X A_i^T``."""

    def __init__(self, in_features: int, out_features: int, tp_group, tp_size: int, bias: bool = False, gather_output: bool = False):
        super().__init__()
        if out_features % tp_size != 0:
            raise ValueError(f"out_features {out_features} not divisible by tp_size {tp_size}")
        self.tp_group, self.tp_size, self.gather_output = tp_group, tp_size, gather_output
        self.out_per_rank = out_features // tp_size
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_per_rank)) if bias else None
        nn.init.kaiming_uniform_(self.weight)

    @classmethod
    def from_linear(cls, linear: nn.Linear, tp_group, tp_size: int, tp_rank: int) -> "ColumnParallelLinear":
        layer = cls(linear.in_features, linear.out_features, tp_group, tp_size, bias=linear.bias is not None)
        shard = linear.weight.data.chunk(tp_size, dim=0)[tp_rank]
        layer.weight.data.copy_(shard)
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data.chunk(tp_size, dim=0)[tp_rank])
        return layer

    def forward(self, x):
        x = all_reduce_backward(x, self.tp_group)
        y = nn.functional.linear(x, self.weight, self.bias)
        if self.gather_output and self.tp_size > 1:
            outs = [torch.empty_like(y) for _ in range(self.tp_size)]
            dist.all_gather(outs, y, group=self.tp_group)
            y = torch.cat(outs, dim=-1)
        return y


class RowParallelLinear(nn.Module):
    """Input dimension split across TP ranks: partial products are all-reduced."""

    def __init__(self, in_features: int, out_features: int, tp_group, tp_size: int, bias: bool = False):
        super().__init__()
        if in_features % tp_size != 0:
            raise ValueError(f"in_features {in_features} not divisible by tp_size {tp_size}")
        self.tp_group, self.tp_size = tp_group, tp_size
        self.in_per_rank = in_features // tp_size
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_rank))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight)

    @classmethod
    def from_linear(cls, linear: nn.Linear, tp_group, tp_size: int, tp_rank: int) -> "RowParallelLinear":
        layer = cls(linear.in_features, linear.out_features, tp_group, tp_size, bias=linear.bias is not None)
        layer.weight.data.copy_(linear.weight.data.chunk(tp_size, dim=1)[tp_rank])
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        return layer

    def forward(self, x):
        y = nn.functional.linear(x, self.weight)
        y = all_reduce_forward(y, self.tp_group)
        if self.bias is not None:
            y = y + self.bias
        return y


def apply_tensor_parallel(layer: nn.Module, mesh: ParallelMesh) -> nn.Module:
    """Replace attention/FFN projections of one transformer block with parallel variants."""
    attn, ffn = layer.attn, layer.ffn
    g, n, r = mesh.tp_group, mesh.tp_size, mesh.tp_rank
    for name in ("q_proj", "k_proj", "v_proj", "wq", "wq_a", "wq_b", "wkv_a", "wkv_b"):
        proj = getattr(attn, name, None)
        if isinstance(proj, nn.Linear) and proj.out_features % n == 0:
            setattr(attn, name, ColumnParallelLinear.from_linear(proj, g, n, r))
    for name in ("c_proj", "wo"):
        proj = getattr(attn, name, None)
        if isinstance(proj, nn.Linear) and proj.in_features % n == 0:
            setattr(attn, name, RowParallelLinear.from_linear(proj, g, n, r))
    if hasattr(ffn, "w1") and isinstance(ffn.w1, nn.Linear) and ffn.w1.out_features % n == 0:
        ffn.w1 = ColumnParallelLinear.from_linear(ffn.w1, g, n, r)
        ffn.w3 = ColumnParallelLinear.from_linear(ffn.w3, g, n, r)
        ffn.w2 = RowParallelLinear.from_linear(ffn.w2, g, n, r)
    elif hasattr(ffn, "c_fc") and isinstance(ffn.c_fc, nn.Linear) and ffn.c_fc.out_features % n == 0:
        ffn.c_fc = ColumnParallelLinear.from_linear(ffn.c_fc, g, n, r)
        ffn.c_proj = RowParallelLinear.from_linear(ffn.c_proj, g, n, r)
    return layer
