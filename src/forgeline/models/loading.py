"""Build models / policies from specs and checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from forgeline.checkpoints.manager import CheckpointManager, CheckpointState, load_model_state
from forgeline.core.config import AdapterConfig, ModelSpec
from forgeline.core.errors import CheckpointError, ModelError
from forgeline.data.tokenizers import Tokenizer, load_tokenizer, tokenizer_from_metadata
from forgeline.models.policy import NativePolicy
from forgeline.models.reward import SequenceRewardModel
from forgeline.models.transformer.model import TransformerLM


def build_model(spec: ModelSpec, device: torch.device | str = "cpu") -> TransformerLM:
    model = TransformerLM(spec)
    return model.to(device)


def load_model_from_checkpoint(path: str | Path, device: torch.device | str = "cpu",
                               spec_override: Optional[ModelSpec] = None) -> Tuple[TransformerLM, ModelSpec, CheckpointState]:
    """Rebuild a :class:`TransformerLM` from a checkpoint directory."""
    state = CheckpointManager.load(path, map_location="cpu")
    if not state.model_spec and spec_override is None:
        raise CheckpointError(f"checkpoint {path} has no model spec; pass spec_override")
    spec = spec_override or ModelSpec.from_dict(state.model_spec)
    model = TransformerLM(spec)
    model_state = state.model_state["model"] if "model" in state.model_state else state.model_state
    load_model_state(model, model_state, strict=False)
    if "adapter" in state.model_state and state.model_state["adapter"]:
        # adapter tensors are re-attached by NativePolicy when an adapter config is provided
        pass
    model.to(device).eval()
    return model, spec, state


def resolve_tokenizer_for_checkpoint(state: CheckpointState, data_dir: Optional[str | Path] = None) -> Tokenizer:
    if state.tokenizer_meta:
        return tokenizer_from_metadata(state.tokenizer_meta)
    if data_dir is not None:
        return load_tokenizer(data_dir)
    raise ModelError("checkpoint has no tokenizer metadata and no data_dir was given",
                     hint="Pass --data-dir pointing at the meta.pkl used for training.")


def load_policy_from_checkpoint(path: str | Path, device: torch.device | str = "cpu",
                                data_dir: Optional[str | Path] = None, adapter: Optional[AdapterConfig] = None,
                                value_head: bool = False) -> NativePolicy:
    model, spec, state = load_model_from_checkpoint(path, device)
    tokenizer = resolve_tokenizer_for_checkpoint(state, data_dir)
    policy = NativePolicy(model, tokenizer, adapter=adapter, value_head=value_head)
    if "adapter" in state.model_state and policy.has_adapter:
        from forgeline.models.adapters.lora import load_lora_state_dict

        load_lora_state_dict(policy.model, state.model_state["adapter"], strict=False)
    if "value_head" in state.model_state and policy.value_head is not None:
        policy.value_head.load_state_dict(state.model_state["value_head"])
    return policy.to(device)


def load_reward_model_from_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> SequenceRewardModel:
    state = CheckpointManager.load(path, map_location="cpu")
    spec = ModelSpec.from_dict(state.model_spec)
    rm = SequenceRewardModel(spec, pooling=state.model_state.get("pooling", "last"))
    rm.load_state_dict(state.model_state["model"])
    return rm.to(device).eval()
