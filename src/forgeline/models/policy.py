"""Policy backends implementing :class:`forgeline.core.protocols.PolicyModel`.

* :class:`NativePolicy` — the built-in :class:`TransformerLM` (CPU-friendly).
* :class:`HFPolicy` — a HuggingFace causal LM with optional PEFT LoRA
  (requires the ``huggingface`` extra).

Both expose the same surface so every post-training algorithm is backend
agnostic: ``generate``, ``logprobs``, ``values``, ``reference()``.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from forgeline.core.config import AdapterConfig, ModelSpec
from forgeline.core.errors import ConfigError, ModelError, OptionalDependencyError
from forgeline.core.protocols import GenerationSettings
from forgeline.data.tokenizers import Tokenizer
from forgeline.models.adapters.lora import LoRALinear, apply_lora, lora_state_dict, load_lora_state_dict, set_lora_enabled
from forgeline.models.adapters.qlora import QLoRALinear, apply_qlora
from forgeline.models.generation.sampling import generate_tokens
from forgeline.models.transformer.model import TransformerLM


class ValueHead(nn.Module):
    """Scalar critic head on the final hidden state."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).squeeze(-1)


def token_logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-token log p(label) computed with ``cross_entropy`` (never materialises log-softmax)."""
    B, T, V = logits.shape
    return -F.cross_entropy(logits.reshape(B * T, V).float(), labels.reshape(B * T), reduction="none").reshape(B, T)


class NativePolicy(nn.Module):
    """Native transformer as an RL policy."""

    backend = "native"

    def __init__(self, model: TransformerLM, tokenizer: Tokenizer, *, pad_token_id: int = 0,
                 adapter: Optional[AdapterConfig] = None, reference_model: Optional[TransformerLM] = None,
                 value_head: bool = False):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.spec: ModelSpec = model.spec
        self.adapter = adapter or AdapterConfig()
        self._has_adapter = False
        if self.adapter.method == "lora":
            apply_lora(self.model, self.adapter.rank, self.adapter.alpha, self.adapter.dropout,
                       self.adapter.target_modules or None)
            self._has_adapter = True
        elif self.adapter.method == "qlora":
            apply_qlora(self.model, self.adapter.rank, self.adapter.alpha, self.adapter.dropout,
                        self.adapter.target_modules or None)
            self._has_adapter = True
        self.reference_model = reference_model
        self.value_head: Optional[ValueHead] = ValueHead(self.spec.n_embd) if value_head else None

    # ── construction helpers ─────────────────────────────────────────────
    @classmethod
    def from_spec(cls, spec: ModelSpec, tokenizer: Tokenizer, **kwargs: Any) -> "NativePolicy":
        return cls(TransformerLM(spec), tokenizer, **kwargs)

    def with_frozen_reference(self) -> "NativePolicy":
        """Create a frozen deep copy of the current model as the KL reference."""
        if self._has_adapter:
            return self  # reference = adapters disabled, no copy needed
        ref = copy.deepcopy(self.model).eval()
        for p in ref.parameters():
            p.requires_grad = False
        self.reference_model = ref
        return self

    # ── PolicyModel protocol ─────────────────────────────────────────────
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def vocab_size(self) -> int:
        return self.spec.vocab_size

    @property
    def has_adapter(self) -> bool:
        return self._has_adapter

    def trainable_parameters(self) -> List[nn.Parameter]:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.value_head is not None:
            params += list(self.value_head.parameters())
        return params

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, settings: GenerationSettings) -> torch.Tensor:
        self.model.eval()
        prompt_ids = prompt_ids.to(self.device)
        out = generate_tokens(
            self.model, prompt_ids, settings.max_new_tokens, temperature=settings.temperature,
            top_k=settings.top_k, top_p=settings.top_p, min_p=settings.min_p,
            repetition_penalty=settings.repetition_penalty, use_cache=settings.use_cache,
            stop_token_ids=settings.stop_token_ids,
        )
        return out[:, prompt_ids.shape[1]:]

    def token_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Logits at every position ``[B, T, V]``."""
        return self.model.full_logits(input_ids.to(self.device))

    @property
    def max_length(self) -> int:
        return self.spec.block_size

    def full_sequence_logits(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
        full = torch.cat([prompt_ids.to(self.device), response_ids.to(self.device)], dim=1)
        if full.shape[1] > self.spec.block_size:
            raise ModelError(f"prompt+response length {full.shape[1]} exceeds block_size {self.spec.block_size}")
        return self.model.full_logits(full)

    def logprobs(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor,
                 response_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-token log-probs of response tokens ``[B, R]`` (masked positions → 0)."""
        full = torch.cat([prompt_ids.to(self.device), response_ids.to(self.device)], dim=1)
        if full.shape[1] > self.spec.block_size:
            raise ModelError(f"prompt+response length {full.shape[1]} exceeds block_size {self.spec.block_size}")
        logits = self.model.full_logits(full)[:, :-1, :]
        labels = full[:, 1:]
        token_lp = token_logprobs_from_logits(logits, labels)
        q_len = prompt_ids.shape[1]
        lp = token_lp[:, q_len - 1:]
        if response_mask is not None:
            lp = lp * response_mask.to(lp.dtype).to(lp.device)
        return lp

    def values(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
        """Scalar value per sequence from the last hidden state (requires a value head)."""
        if self.value_head is None:
            raise ConfigError("values() requires value_head=True on the policy")
        full = torch.cat([prompt_ids.to(self.device), response_ids.to(self.device)], dim=1)
        h = self.model.forward_hidden(full)
        return self.value_head(h[:, -1, :].float())

    def logprobs_and_values(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor):
        full = torch.cat([prompt_ids.to(self.device), response_ids.to(self.device)], dim=1)
        h = self.model.forward_hidden(full)
        logits = self.model.lm_head(h)[:, :-1, :]
        lp = token_logprobs_from_logits(logits, full[:, 1:])[:, prompt_ids.shape[1] - 1:]
        values = self.value_head(h[:, -1, :].float()) if self.value_head is not None else None
        return lp, values

    @contextmanager
    def reference(self) -> Iterator["NativePolicy"]:
        """Behave as the frozen reference policy inside the context."""
        if self._has_adapter:
            set_lora_enabled(self.model, False)
            try:
                yield self
            finally:
                set_lora_enabled(self.model, True)
        elif self.reference_model is not None:
            live = self.model
            self.model = self.reference_model
            try:
                yield self
            finally:
                self.model = live
        else:
            raise ConfigError("no reference policy: call with_frozen_reference() or use an adapter",
                              hint="KL-regularised algorithms need a frozen reference.")

    # ── checkpoint plumbing ──────────────────────────────────────────────
    def state_for_checkpoint(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"model": self.model.state_dict()}
        if self._has_adapter:
            state["adapter"] = lora_state_dict(self.model)
        if self.value_head is not None:
            state["value_head"] = self.value_head.state_dict()
        return state

    def load_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        if "model" in state:
            from forgeline.checkpoints.manager import load_model_state

            load_model_state(self.model, state["model"], strict=False)
        if "adapter" in state and self._has_adapter:
            load_lora_state_dict(self.model, state["adapter"], strict=False)
        if "value_head" in state and self.value_head is not None:
            self.value_head.load_state_dict(state["value_head"])

    # ── text helpers ─────────────────────────────────────────────────────
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: torch.Tensor | List[int]) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode([int(t) for t in ids])

    def to(self, *args: Any, **kwargs: Any) -> "NativePolicy":  # type: ignore[override]
        super().to(*args, **kwargs)
        if self.reference_model is not None:
            self.reference_model.to(*args, **kwargs)
        return self


class HFPolicy(nn.Module):
    """HuggingFace causal LM (+ optional PEFT LoRA) behind the same policy surface.

    Requires ``pip install 'forgeline[huggingface]'``. Reference log-probs use
    ``disable_adapter()`` when LoRA is attached (no second copy of the model).
    """

    backend = "huggingface"

    def __init__(self, model_name: str, *, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
                 target_modules: Optional[List[str]] = None, load_in_8bit: bool = False, device_map: Any = "auto",
                 torch_dtype: Any = None, gradient_checkpointing: bool = True, value_head: bool = False,
                 use_lora: bool = True):
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise OptionalDependencyError("transformers is not installed", hint="pip install 'forgeline[huggingface]'") from exc
        self.model_name = model_name
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.hf_tokenizer.pad_token is None:
            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
        kwargs: Dict[str, Any] = {"device_map": device_map}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if load_in_8bit:
            kwargs["load_in_8bit"] = True
        base = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self._has_adapter = False
        if use_lora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as exc:
                raise OptionalDependencyError("peft is not installed", hint="pip install 'forgeline[huggingface]'") from exc
            cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                             target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"], bias="none")
            base = get_peft_model(base, cfg)
            self._has_adapter = True
        if gradient_checkpointing:
            if hasattr(base, "enable_input_require_grads"):
                base.enable_input_require_grads()
            base.gradient_checkpointing_enable()
        self.model = base
        hidden = self.model.config.hidden_size
        self.value_head: Optional[ValueHead] = ValueHead(hidden).to(self.device) if value_head else None
        self.pad_token_id = int(self.hf_tokenizer.pad_token_id)
        self.tokenizer = _HFTokenizerAdapter(self.hf_tokenizer)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    @property
    def has_adapter(self) -> bool:
        return self._has_adapter

    def trainable_parameters(self) -> List[nn.Parameter]:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.value_head is not None:
            params += list(self.value_head.parameters())
        return params

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, settings: GenerationSettings) -> torch.Tensor:
        self.model.eval()
        prompt_ids = prompt_ids.to(self.device)
        attn = (prompt_ids != self.pad_token_id).long()
        kwargs: Dict[str, Any] = dict(max_new_tokens=settings.max_new_tokens, pad_token_id=self.pad_token_id,
                                      eos_token_id=self.hf_tokenizer.eos_token_id)
        if settings.temperature > 0:
            kwargs.update(do_sample=True, temperature=settings.temperature)
            if settings.top_p:
                kwargs["top_p"] = settings.top_p
            if settings.top_k:
                kwargs["top_k"] = settings.top_k
        else:
            kwargs["do_sample"] = False
        out = self.model.generate(input_ids=prompt_ids, attention_mask=attn, **kwargs)
        return out[:, prompt_ids.shape[1]:]

    @property
    def max_length(self) -> int:
        return int(getattr(self.model.config, "max_position_embeddings", 4096))

    @property
    def spec(self):
        """Architecture summary for checkpoints (HuggingFace config)."""
        return _HFSpec(self.model_name, self.model.config)

    def token_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        return self.model(input_ids=input_ids, attention_mask=(input_ids != self.pad_token_id).long()).logits

    def full_sequence_logits(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
        return self.token_logits(torch.cat([prompt_ids.to(self.device), response_ids.to(self.device)], dim=1))

    def _forward(self, prompt_ids, response_ids, hidden: bool = False):
        full = torch.cat([prompt_ids.to(self.device), response_ids.to(self.device)], dim=1)
        attn = (full != self.pad_token_id).long()
        return full, self.model(input_ids=full, attention_mask=attn, output_hidden_states=hidden)

    def logprobs(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor,
                 response_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        full, out = self._forward(prompt_ids, response_ids)
        lp = token_logprobs_from_logits(out.logits[:, :-1, :], full[:, 1:])[:, prompt_ids.shape[1] - 1:]
        if response_mask is not None:
            lp = lp * response_mask.to(lp.dtype).to(lp.device)
        return lp

    def values(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
        if self.value_head is None:
            raise ConfigError("values() requires value_head=True on the policy")
        _, out = self._forward(prompt_ids, response_ids, hidden=True)
        return self.value_head(out.hidden_states[-1][:, -1, :].float())

    def logprobs_and_values(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor):
        full, out = self._forward(prompt_ids, response_ids, hidden=self.value_head is not None)
        lp = token_logprobs_from_logits(out.logits[:, :-1, :], full[:, 1:])[:, prompt_ids.shape[1] - 1:]
        values = self.value_head(out.hidden_states[-1][:, -1, :].float()) if self.value_head is not None else None
        return lp, values

    @contextmanager
    def reference(self) -> Iterator["HFPolicy"]:
        if not self._has_adapter:
            raise ConfigError("HFPolicy reference requires LoRA adapters (use_lora=True)")
        with self.model.disable_adapter():
            yield self

    def state_for_checkpoint(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        if self._has_adapter:
            state["adapter"] = {k: v.detach().cpu() for k, v in self.model.state_dict().items() if "lora_" in k}
        else:
            state["model"] = self.model.state_dict()
        if self.value_head is not None:
            state["value_head"] = self.value_head.state_dict()
        return state

    def load_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        if "adapter" in state:
            self.model.load_state_dict(dict(state["adapter"]), strict=False)
        if "model" in state:
            self.model.load_state_dict(dict(state["model"]), strict=False)
        if "value_head" in state and self.value_head is not None:
            self.value_head.load_state_dict(state["value_head"])

    def save_adapter(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.hf_tokenizer.save_pretrained(path)
        if self.value_head is not None:
            torch.save(self.value_head.state_dict(), f"{path}/value_head.pt")

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: torch.Tensor | List[int]) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids)


class _HFSpec:
    """Minimal spec object so checkpoints of HuggingFace policies record their architecture."""

    def __init__(self, name: str, config: Any):
        self.name, self.config = name, config
        self.block_size = int(getattr(config, "max_position_embeddings", 4096))
        self.vocab_size = int(getattr(config, "vocab_size", 0))

    def to_dict(self) -> Dict[str, Any]:
        return {"hf_model": self.name, "model_type": getattr(self.config, "model_type", ""), "vocab_size": self.vocab_size,
                "hidden_size": getattr(self.config, "hidden_size", None), "num_hidden_layers": getattr(self.config, "num_hidden_layers", None)}


class _HFTokenizerAdapter:
    kind = "huggingface"

    def __init__(self, hf):
        self.hf = hf
        self.vocab_size = int(hf.vocab_size)

    def encode(self, text: str) -> List[int]:
        return self.hf.encode(text, add_special_tokens=False)

    def decode(self, ids) -> str:
        return self.hf.decode(list(ids), skip_special_tokens=True)

    def metadata(self) -> Dict[str, Any]:
        return {"tokenizer_type": "huggingface", "tokenizer_name": getattr(self.hf, "name_or_path", ""),
                "vocab_size": self.vocab_size}


def has_lora_modules(model: nn.Module) -> bool:
    return any(isinstance(m, (LoRALinear, QLoRALinear)) for m in model.modules())
