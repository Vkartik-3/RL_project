"""Parameter-efficient adapters (LoRA / QLoRA) for the native model."""

from forgeline.models.adapters.lora import (
    DEFAULT_TARGET_MODULES,
    LoRALinear,
    apply_lora,
    load_lora,
    load_lora_state_dict,
    lora_state_dict,
    merge_lora,
    save_lora,
    set_lora_enabled,
    trainable_parameter_report,
)
from forgeline.models.adapters.qlora import QLoRALinear, apply_qlora

__all__ = [
    "DEFAULT_TARGET_MODULES", "LoRALinear", "apply_lora", "load_lora", "load_lora_state_dict",
    "lora_state_dict", "merge_lora", "save_lora", "set_lora_enabled", "trainable_parameter_report",
    "QLoRALinear", "apply_qlora",
]
