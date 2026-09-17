"""Low-rank adapters: ``y = W x + (x A B) * alpha / rank``."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from forgeline.core.errors import ModelError

DEFAULT_TARGET_MODULES: List[str] = [
    "q_proj", "k_proj", "v_proj", "c_proj",  # grouped-query attention
    "wq", "wq_a", "wq_b", "wkv_a", "wkv_b", "wo",  # latent attention
    "w1", "w2", "w3",  # SwiGLU
    "c_fc",  # GELU FFN
]


class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank: int, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if rank < 1:
            raise ModelError("LoRA rank must be >= 1")
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(original.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True

    @property
    def in_features(self) -> int:
        return self.original.in_features

    @property
    def out_features(self) -> int:
        return self.original.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.original(x)
        if not self.enabled:
            return result
        lora_out = self.dropout(x) @ self.lora_A.to(x.dtype) @ self.lora_B.to(x.dtype)
        return result + lora_out * self.scaling

    def merge(self) -> nn.Linear:
        with torch.no_grad():
            self.original.weight.add_((self.lora_A @ self.lora_B).t().to(self.original.weight.dtype) * self.scaling)
        return self.original


def _iter_linear_attrs(model: nn.Module):
    for name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            yield name, module, attr_name, child


def apply_lora(model: nn.Module, rank: int = 16, alpha: float = 32.0, dropout: float = 0.0,
               target_modules: Optional[Sequence[str]] = None) -> nn.Module:
    """Wrap matching ``nn.Linear`` layers with LoRA and freeze everything else."""
    targets = list(target_modules) if target_modules else DEFAULT_TARGET_MODULES
    for p in model.parameters():
        p.requires_grad = False
    count = 0
    for _, parent, attr_name, child in _iter_linear_attrs(model):
        if isinstance(child, nn.Linear) and any(t in attr_name for t in targets):
            setattr(parent, attr_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            count += 1
    if count == 0:
        raise ModelError(f"No linear layers matched target_modules={targets}")
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    for _, parent, attr_name, child in _iter_linear_attrs(model):
        if isinstance(child, LoRALinear):
            merged = child.merge()
            merged.weight.requires_grad = True
            if merged.bias is not None:
                merged.bias.requires_grad = True
            setattr(parent, attr_name, merged)
    return model


def _is_adapter(m: nn.Module) -> bool:
    return hasattr(m, "lora_A") and hasattr(m, "lora_B") and hasattr(m, "enabled")


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    """Enable/disable every LoRA/QLoRA adapter (disabled == frozen base = reference policy)."""
    for m in model.modules():
        if _is_adapter(m):
            m.enabled = enabled


def lora_state_dict(model: nn.Module) -> Dict[str, Dict[str, object]]:
    state: Dict[str, Dict[str, object]] = {}
    for name, m in model.named_modules():
        if _is_adapter(m):
            state[name] = {"lora_A": m.lora_A.data.clone(), "lora_B": m.lora_B.data.clone(),
                           "rank": m.rank, "alpha": m.alpha}
    return state


def load_lora_state_dict(model: nn.Module, state: Dict[str, Dict[str, object]], strict: bool = True) -> int:
    loaded = 0
    modules = {n: m for n, m in model.named_modules() if _is_adapter(m)}
    for name, entry in state.items():
        if name not in modules:
            if strict:
                raise ModelError(f"LoRA state contains layer {name!r} that the model does not have")
            continue
        m = modules[name]
        if m.rank != entry["rank"]:
            raise ModelError(f"LoRA rank mismatch for {name}: model {m.rank} vs state {entry['rank']}")
        m.lora_A.data.copy_(entry["lora_A"])
        m.lora_B.data.copy_(entry["lora_B"])
        loaded += 1
    return loaded


def save_lora(model: nn.Module, path: str) -> int:
    state = lora_state_dict(model)
    torch.save(state, path)
    return sum(int(s["lora_A"].numel() + s["lora_B"].numel()) for s in state.values())


def load_lora(model: nn.Module, path: str, device: Optional[str] = None) -> int:
    state = torch.load(path, map_location=device, weights_only=True)
    return load_lora_state_dict(model, state)


def trainable_parameter_report(model: nn.Module) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "fraction": trainable / max(total, 1)}
