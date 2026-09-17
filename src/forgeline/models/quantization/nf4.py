"""4-bit NormalFloat (NF4) block quantization (block size 64, 2 values per byte)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from forgeline.core.errors import QuantizationError

NF4_TABLE = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=torch.float32)

NF4_BLOCK_SIZE = 64


def quantize_nf4(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
    """Quantize to NF4. Returns ``(packed uint8, per-block absmax, original shape)``."""
    if tensor.numel() == 0:
        raise QuantizationError("cannot quantize an empty tensor")
    flat = tensor.detach().flatten().float().cpu()
    n = flat.numel()
    pad = (NF4_BLOCK_SIZE - n % NF4_BLOCK_SIZE) % NF4_BLOCK_SIZE
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad)])
    blocks = flat.reshape(-1, NF4_BLOCK_SIZE)
    absmax = blocks.abs().max(dim=1).values.clamp(min=1e-8)
    normalized = blocks / absmax.unsqueeze(1)
    diffs = (normalized.unsqueeze(-1) - NF4_TABLE.view(1, 1, -1)).abs()
    indices = diffs.argmin(dim=-1).reshape(-1)
    packed = (indices[::2] << 4 | indices[1::2]).to(torch.uint8)
    return packed, absmax, tensor.shape


def dequantize_nf4(packed: torch.Tensor, absmax: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    high = (packed >> 4).to(torch.long)
    low = (packed & 0x0F).to(torch.long)
    indices = torch.stack([high, low], dim=-1).flatten()
    values = NF4_TABLE.to(packed.device)[indices]
    total = absmax.numel() * NF4_BLOCK_SIZE
    values = values[:total].reshape(-1, NF4_BLOCK_SIZE) * absmax.to(values.device).unsqueeze(1)
    n_elements = 1
    for s in shape:
        n_elements *= int(s)
    return values.flatten()[:n_elements].reshape(shape)


class NF4Linear(nn.Module):
    """Linear layer whose frozen weight is stored as packed NF4 and dequantized on the fly."""

    def __init__(self, original: nn.Linear):
        super().__init__()
        packed, absmax, shape = quantize_nf4(original.weight.data)
        self.register_buffer("weight_packed", packed)
        self.register_buffer("weight_absmax", absmax)
        self.weight_shape = tuple(shape)
        self.in_features = original.in_features
        self.out_features = original.out_features
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

    def dequantized_weight(self) -> torch.Tensor:
        return dequantize_nf4(self.weight_packed, self.weight_absmax, torch.Size(self.weight_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantized_weight().to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, self.bias)

    def memory_bytes(self) -> int:
        return self.weight_packed.numel() + self.weight_absmax.numel() * 4


def quantization_report(model: nn.Module) -> dict:
    """Count NF4 layers and estimate memory versus fp16 storage."""
    n_layers, packed_bytes, fp16_bytes = 0, 0, 0
    for m in model.modules():
        if isinstance(m, NF4Linear):
            n_layers += 1
            packed_bytes += m.memory_bytes()
            fp16_bytes += m.in_features * m.out_features * 2
    return {
        "nf4_layers": n_layers,
        "nf4_bytes": packed_bytes,
        "fp16_equivalent_bytes": fp16_bytes,
        "compression_ratio": (fp16_bytes / packed_bytes) if packed_bytes else 0.0,
    }
