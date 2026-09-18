"""Configuration objects and the experiment manifest.

Three layers:

* :class:`ModelSpec` — architecture hyper-parameters (+ named presets).
* :class:`TrainerConfig` — optimizer / schedule / precision / checkpoint knobs
  shared by every training stage.
* :class:`ExperimentManifest` — the reproducible, human-readable record of a
  run (YAML or TOML), composed of the specs above plus data, algorithm,
  distributed, evaluation and reward settings.

All objects are plain dataclasses with explicit validation so that they stay
serialisable and dependency-free.
"""

from __future__ import annotations

import dataclasses
import os
import platform
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping

from forgeline.core.errors import ConfigError, UnknownPresetError

# ═════════════════════════════════════════════════════════════════════════════
#  Model specification
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class ModelSpec:
    """Architecture specification for the native decoder-only transformer."""

    # core
    block_size: int = 256
    vocab_size: int = 65536
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False
    use_rope: bool = True
    use_swiglu: bool = True

    # sliding window / logit cap
    sliding_window: int = 0
    alternating_layers: bool = False
    attn_logit_cap: float = 0.0

    # context extension
    rope_scaling_type: str = "none"  # none | linear | yarn
    rope_scaling_factor: float = 1.0

    # multi-head latent attention
    use_mla: bool = False
    kv_lora_rank: int = 512
    q_lora_rank: int = 0
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # mixture of experts
    use_moe: bool = False
    n_experts: int = 8
    n_experts_active: int = 2
    n_shared_experts: int = 0
    n_dense_layers: int = 0
    score_func: str = "softmax"  # softmax | sigmoid
    route_scale: float = 1.0
    aux_loss_free: bool = False
    bias_update_speed: float = 0.001
    moe_aux_loss_weight: float = 0.01
    n_expert_groups: int = 1
    n_limited_groups: int = 1

    # multi-token prediction
    n_predict_tokens: int = 1
    mtp_loss_weight: float = 1.0

    # native sparse attention
    use_nsa: bool = False
    nsa_block_size: int = 32
    nsa_top_k: int = 16
    nsa_window_size: int = 256

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ConfigError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if not self.use_mla and self.n_head % self.n_kv_head != 0:
            raise ConfigError(
                f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
            )
        if self.rope_scaling_type not in ("none", "linear", "yarn"):
            raise ConfigError(f"rope_scaling_type must be none|linear|yarn, got {self.rope_scaling_type!r}")
        if self.score_func not in ("softmax", "sigmoid"):
            raise ConfigError(f"score_func must be softmax|sigmoid, got {self.score_func!r}")
        if self.use_moe and self.n_experts_active > self.n_experts:
            raise ConfigError("n_experts_active cannot exceed n_experts")
        if self.use_moe and self.n_experts % self.n_expert_groups != 0:
            raise ConfigError("n_experts must be divisible by n_expert_groups")
        if self.use_mla and self.use_nsa:
            raise ConfigError("use_mla and use_nsa are mutually exclusive")
        if self.n_predict_tokens < 1:
            raise ConfigError("n_predict_tokens must be >= 1")
        if self.block_size < 1 or self.vocab_size < 1:
            raise ConfigError("block_size and vocab_size must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelSpec":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"Unknown ModelSpec keys: {sorted(unknown)}")
        return cls(**dict(data))

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


# Named presets. Values (other than names) match the measured configurations
# recorded under benchmarks/.
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny": dict(n_layer=2, n_head=2, n_kv_head=2, n_embd=64, block_size=64, dropout=0.0),
    "small": dict(n_layer=6, n_head=6, n_kv_head=6, n_embd=384, block_size=256, dropout=0.1),
    "medium": dict(n_layer=12, n_head=12, n_kv_head=4, n_embd=768, block_size=512, dropout=0.1),
    "large": dict(n_layer=24, n_head=16, n_kv_head=4, n_embd=1024, block_size=2048, dropout=0.1),
    "xl": dict(n_layer=24, n_head=16, n_kv_head=8, n_embd=2048, block_size=4096, dropout=0.1),
    "moe_32l": dict(
        n_layer=32, n_head=32, n_kv_head=8, n_embd=4096, block_size=8192, dropout=0.1,
        use_moe=True, n_experts=8, n_experts_active=2,
    ),
    "latent_moe_27l": dict(
        n_layer=27, n_head=16, n_kv_head=16, n_embd=2048, block_size=4096, dropout=0.0, bias=False,
        use_mla=True, kv_lora_rank=512, q_lora_rank=1536,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        use_moe=True, n_experts=64, n_experts_active=6, n_shared_experts=2, n_dense_layers=1,
        score_func="sigmoid", route_scale=1.0, aux_loss_free=True, bias_update_speed=0.001,
        n_predict_tokens=3,
    ),
    "sliding_window_32l": dict(
        n_layer=32, n_head=32, n_kv_head=8, n_embd=4096, block_size=8192, dropout=0.0, bias=False,
        sliding_window=4096,
    ),
    "alternating_28l": dict(
        n_layer=28, n_head=16, n_kv_head=4, n_embd=2304, block_size=8192, dropout=0.0, bias=False,
        sliding_window=4096, alternating_layers=True, attn_logit_cap=50.0,
    ),
}


def model_spec_from_preset(preset: str, **overrides: Any) -> ModelSpec:
    """Build a :class:`ModelSpec` from a preset name plus overrides."""
    if preset not in MODEL_PRESETS:
        raise UnknownPresetError(
            f"Unknown model preset {preset!r}. Available: {sorted(MODEL_PRESETS)}"
        )
    values = dict(MODEL_PRESETS[preset])
    values.update(overrides)
    return ModelSpec.from_dict(values)


# ═════════════════════════════════════════════════════════════════════════════
#  Trainer configuration
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class OptimizerConfig:
    name: str = "adamw"  # adamw | adam
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    decay_only_matrices: bool = True  # weight decay on dim>=2 params only
    fused: str = "auto"  # auto | always | never

    def validate(self) -> None:
        if self.name not in ("adamw", "adam"):
            raise ConfigError(f"optimizer name must be adamw|adam, got {self.name!r}")
        if self.learning_rate <= 0:
            raise ConfigError("learning_rate must be positive")
        if self.grad_clip < 0:
            raise ConfigError("grad_clip must be >= 0 (0 disables clipping)")


@dataclass
class ScheduleConfig:
    name: str = "cosine"  # cosine | wsd | constant | linear_decay
    warmup_steps: int = 100
    decay_steps: int = 5000
    min_lr: float = 3e-5
    wsd_stable_fraction: float = 0.8

    def validate(self) -> None:
        if self.name not in ("cosine", "wsd", "constant", "linear_decay"):
            raise ConfigError(f"schedule name must be cosine|wsd|constant|linear_decay, got {self.name!r}")
        if self.warmup_steps < 0 or self.decay_steps < 0:
            raise ConfigError("warmup_steps and decay_steps must be >= 0")
        if not 0.0 <= self.wsd_stable_fraction <= 1.0:
            raise ConfigError("wsd_stable_fraction must be in [0, 1]")


@dataclass
class PrecisionConfig:
    dtype: str = "auto"  # auto | float32 | bfloat16 | float16
    grad_scaler: str = "auto"  # auto | on | off

    def validate(self) -> None:
        if self.dtype not in ("auto", "float32", "bfloat16", "float16"):
            raise ConfigError(f"precision dtype must be auto|float32|bfloat16|float16, got {self.dtype!r}")
        if self.grad_scaler not in ("auto", "on", "off"):
            raise ConfigError("grad_scaler must be auto|on|off")


@dataclass
class CheckpointConfig:
    directory: str = "checkpoints"
    save_every: int = 1000
    keep_last: int = 3
    async_save: bool = False
    resume_from: str = ""  # path, or "" for auto-resume from latest in `directory`
    auto_resume: bool = True

    def validate(self) -> None:
        if self.save_every < 0:
            raise ConfigError("save_every must be >= 0")
        if self.keep_last < 1:
            raise ConfigError("keep_last must be >= 1")


@dataclass
class AdapterConfig:
    method: str = "none"  # none | lora | qlora
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=list)
    merge_on_save: bool = False

    def validate(self) -> None:
        if self.method not in ("none", "lora", "qlora"):
            raise ConfigError(f"adapter method must be none|lora|qlora, got {self.method!r}")
        if self.rank < 1:
            raise ConfigError("adapter rank must be >= 1")


@dataclass
class TrainerConfig:
    max_steps: int = 1000
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    eval_every: int = 200
    eval_batches: int = 20
    log_every: int = 10
    seed: int = 1337
    device: str = "auto"
    compile_model: bool = False  # torch.compile the modules the algorithm trains (in place; state_dict keys unchanged)
    compile_backend: str = "inductor"  # inductor (default) | eager | aot_eager | any registered dynamo backend
    gradient_checkpointing: bool = False
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)

    def validate(self) -> None:
        if self.max_steps < 0:
            raise ConfigError("max_steps must be >= 0")
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ConfigError("batch_size and gradient_accumulation_steps must be >= 1")
        for sub in (self.optimizer, self.schedule, self.precision, self.checkpoint, self.adapter):
            sub.validate()


# ═════════════════════════════════════════════════════════════════════════════
#  Experiment manifest
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class DistributedConfig:
    strategy: str = "single"  # single | ddp | fsdp | deepspeed | tensor_parallel | pipeline_parallel
    backend: str = "auto"  # auto | nccl | gloo
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    pipeline_micro_batches: int = 4
    fsdp_min_params_to_wrap: int = 100_000
    deepspeed_config: str = ""

    def validate(self) -> None:
        valid = ("single", "ddp", "fsdp", "deepspeed", "tensor_parallel", "pipeline_parallel")
        if self.strategy not in valid:
            raise ConfigError(f"distributed strategy must be one of {valid}, got {self.strategy!r}")
        if self.tensor_parallel_size < 1 or self.pipeline_parallel_size < 1:
            raise ConfigError("parallel sizes must be >= 1")
        if self.backend not in ("auto", "nccl", "gloo"):
            raise ConfigError("backend must be auto|nccl|gloo")
        if self.strategy == "deepspeed" and not self.deepspeed_config:
            raise ConfigError("deepspeed strategy requires deepspeed_config path")


@dataclass
class DataConfig:
    kind: str = "pretraining"  # pretraining | supervised | preference | reward | verifiable | process
    path: str = "data"
    tokenizer: str = "auto"  # auto | char | tiktoken | hf:<name>
    val_fraction: float = 0.1
    max_length: int = 256
    seed: int = 1337
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        valid = ("pretraining", "supervised", "preference", "reward", "verifiable", "process", "trajectories")  # trajectories = synthesis records
        if self.kind not in valid:
            raise ConfigError(f"data kind must be one of {valid}, got {self.kind!r}")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ConfigError("val_fraction must be in [0, 1)")


@dataclass
class ExperimentManifest:
    """Everything needed to reproduce a run."""

    name: str = "experiment"
    algorithm: str = "pretrain"
    model: ModelSpec = field(default_factory=ModelSpec)
    model_preset: str = ""
    checkpoint_path: str = ""
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    backend: Dict[str, Any] = field(default_factory=dict)  # {} / {type: native} or {type: huggingface, model_name: ..., ...}
    reward: Dict[str, Any] = field(default_factory=dict)
    orchestration: Dict[str, Any] = field(default_factory=dict)  # {} / {type: local} or {type: ray, rollout_workers: .., reward_workers: ..}
    evaluation: list[str] = field(default_factory=list)
    output_dir: str = "runs/experiment"
    tags: list[str] = field(default_factory=list)
    runtime: Dict[str, Any] = field(default_factory=dict)

    KNOWN_ALGORITHMS = (
        "pretrain", "sft", "distill", "reward_model", "dpo", "ppo", "grpo", "dapo",
        "rlvr", "rlaif", "constitutional", "star", "hill_climb", "tabular_ppo", "allocation_ppo",
    )

    def validate(self) -> None:
        if self.algorithm not in self.KNOWN_ALGORITHMS:
            raise ConfigError(
                f"Unknown algorithm {self.algorithm!r}; expected one of {self.KNOWN_ALGORITHMS}"
            )
        if self.backend and self.backend.get("type", "native") not in ("native", "huggingface"):
            raise ConfigError("backend.type must be native|huggingface")
        if self.backend.get("type") == "huggingface" and not self.backend.get("model_name"):
            raise ConfigError("backend.type=huggingface requires backend.model_name")
        from forgeline.orchestration import validate_orchestration

        validate_orchestration(self.orchestration)
        self.model.validate()
        self.trainer.validate()
        self.data.validate()
        self.distributed.validate()

    # ── (de)serialisation ────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentManifest":
        data = dict(data)
        preset = data.get("model_preset", "") or ""
        model_data = data.get("model", {}) or {}
        if preset:
            model = model_spec_from_preset(preset, **model_data)
        else:
            model = ModelSpec.from_dict(model_data)
        manifest = cls(
            name=data.get("name", "experiment"),
            algorithm=data.get("algorithm", "pretrain"),
            model=model,
            model_preset=preset,
            checkpoint_path=data.get("checkpoint_path", ""),
            trainer=_nested(TrainerConfig, data.get("trainer", {})),
            data=_nested(DataConfig, data.get("data", {})),
            distributed=_nested(DistributedConfig, data.get("distributed", {})),
            algorithm_params=dict(data.get("algorithm_params", {}) or {}),
            backend=dict(data.get("backend", {}) or {}),
            reward=dict(data.get("reward", {}) or {}),
            orchestration=dict(data.get("orchestration", {}) or {}),
            evaluation=list(data.get("evaluation", []) or []),
            output_dir=data.get("output_dir", "runs/experiment"),
            tags=list(data.get("tags", []) or []),
            runtime=dict(data.get("runtime", {}) or {}),
        )
        manifest.validate()
        return manifest

    def with_runtime_environment(self) -> "ExperimentManifest":
        """Attach the current runtime environment (torch, python, platform)."""
        self.runtime = capture_runtime_environment()
        return self

    def save(self, path: str | os.PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = dump_config(self.to_dict(), fmt=_format_for(path))
        path.write_text(text)
        return path

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ExperimentManifest":
        return cls.from_dict(load_config(path))


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _nested(cls, data: Mapping[str, Any]):
    """Build a (possibly nested) dataclass from a mapping with strict keys."""
    data = dict(data or {})
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        ftype = f.type if not isinstance(f.type, str) else None
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else None  # type: ignore[misc]
        if dataclasses.is_dataclass(default) and isinstance(value, Mapping):
            kwargs[name] = _nested(type(default), value)
        else:
            kwargs[name] = value
    obj = cls(**kwargs)
    if hasattr(obj, "validate"):
        obj.validate()
    return obj


def _format_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return "yaml"
    if suffix == ".toml":
        return "toml"
    if suffix == ".json":
        return "json"
    raise ConfigError(f"Unsupported config extension {suffix!r}; use .yaml, .toml or .json")


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML / TOML / JSON mapping from disk."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    fmt = _format_for(path)
    text = path.read_text()
    if fmt == "yaml":
        import yaml

        data = yaml.safe_load(text) or {}
    elif fmt == "toml":
        if sys.version_info >= (3, 11):
            import tomllib

            data = tomllib.loads(text)
        else:  # pragma: no cover
            import tomli

            data = tomli.loads(text)
    else:
        import json

        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ConfigError(f"Config root must be a mapping: {path}")
    return dict(data)


def dump_config(data: Mapping[str, Any], fmt: str = "yaml") -> str:
    if fmt == "yaml":
        import yaml

        return yaml.safe_dump(dict(data), sort_keys=False)
    if fmt == "json":
        import json

        return json.dumps(dict(data), indent=2)
    if fmt == "toml":
        return _dump_toml(dict(data))
    raise ConfigError(f"Unsupported dump format {fmt!r}")


def _dump_toml(data: Mapping[str, Any], prefix: str = "") -> str:
    """Minimal TOML writer sufficient for manifests (no external dependency)."""
    lines: list[str] = []
    tables: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in data.items():
        if isinstance(value, Mapping):
            tables.append((f"{prefix}{key}" if not prefix else f"{prefix}.{key}", value))
        else:
            lines.append(f"{key} = {_toml_value(value)}")
    out = "\n".join(lines)
    for name, table in tables:
        out += f"\n\n[{name}]\n" + _dump_toml(table, prefix=name)
    return out.strip() + "\n"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if value is None:
        return '""'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def capture_runtime_environment() -> Dict[str, Any]:
    import torch

    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }
    if env["cuda_available"]:
        env["cuda_device_name"] = torch.cuda.get_device_name(0)
    return env
