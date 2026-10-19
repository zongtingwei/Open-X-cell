from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real


DEFAULT_PRIOR_DIMS: dict[str, int] = {
    "esm": 5120,
    "string": 512,
    "genept": 3072,
    "depmap": 1150,
    "cell_painting": 259,
}


@dataclass(frozen=True)
class LossConfig:
    mmd: float = 1.0
    fold_change_ccc: float = 1.0
    per_gene_ccc: float = 1.0
    delta_norm: float = 1.0
    delta_mse: float = 1.0
    contrastive: float = 1.0
    prior_reconstruction: float = 1.0
    l1_output: float = 0.0  # PaLM-style L1 output regularization (paper: Mini=1.0, Ultra=0)

    def __post_init__(self) -> None:
        invalid = {
            name: value
            for name, value in self.__dict__.items()
            if isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value < 0
        }
        if invalid:
            raise ValueError(f"loss weights must be finite and non-negative: {invalid}")


@dataclass(frozen=True)
class ModelConfig:
    num_genes: int = 19_408
    context_length: int = 4_000
    num_layers: int = 12
    hidden_dim: int = 512
    num_heads: int = 8
    ffn_dim: int | None = None
    dropout: float = 0.0
    cross_attention_layers: tuple[int, ...] = (2, 5, 8, 11)
    # Cross-attention has its own residual/FFN specification in the paper.  The
    # optional values preserve backwards compatibility with early clean-room
    # checkpoints, where the cross block inherited the backbone settings.
    cross_attention_ffn_dim: int | None = None
    cross_attention_activation: str | None = None
    cross_attention_norm: str | None = None
    cross_attention_pre_norm: bool | None = None
    prior_dims: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_PRIOR_DIMS))
    activation: str = "relu"
    norm: str = "layernorm"
    pre_norm: bool = False
    use_qk_norm: bool = False
    attention_bias: bool = True
    tied_output: bool = False
    value_clip: float = 512.0
    diffusion_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)

    def __post_init__(self) -> None:
        integer_fields = {
            "num_genes": self.num_genes,
            "context_length": self.context_length,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
        }
        invalid_integer_fields = [
            name
            for name, value in integer_fields.items()
            if isinstance(value, bool) or not isinstance(value, Integral)
        ]
        if invalid_integer_fields:
            raise ValueError(f"model integer fields have invalid types: {invalid_integer_fields}")
        if self.num_genes <= 0 or self.num_layers <= 0 or self.hidden_dim <= 0:
            raise ValueError("num_genes, num_layers, and hidden_dim must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.context_length > self.num_genes:
            raise ValueError("context_length cannot exceed num_genes")
        if len(set(self.cross_attention_layers)) != len(self.cross_attention_layers):
            raise ValueError("cross_attention_layers must not contain duplicates")
        if any(
            isinstance(layer, bool)
            or not isinstance(layer, Integral)
            or layer < 0
            or layer >= self.num_layers
            for layer in self.cross_attention_layers
        ):
            raise ValueError("cross_attention_layers must be valid layer indices")
        if self.ffn_dim is not None and (
            isinstance(self.ffn_dim, bool)
            or not isinstance(self.ffn_dim, Integral)
            or self.ffn_dim <= 0
        ):
            raise ValueError("ffn_dim must be positive when provided")
        if self.cross_attention_ffn_dim is not None and (
            isinstance(self.cross_attention_ffn_dim, bool)
            or not isinstance(self.cross_attention_ffn_dim, Integral)
            or self.cross_attention_ffn_dim <= 0
        ):
            raise ValueError("cross_attention_ffn_dim must be positive when provided")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, Real)
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if self.activation not in {"relu", "swiglu"}:
            raise ValueError("activation must be relu or swiglu")
        if self.cross_attention_activation not in {None, "relu", "swiglu"}:
            raise ValueError("cross_attention_activation must be relu, swiglu, or None")
        if self.norm not in {"layernorm", "rmsnorm"}:
            raise ValueError("norm must be layernorm or rmsnorm")
        if self.cross_attention_norm not in {None, "layernorm", "rmsnorm"}:
            raise ValueError("cross_attention_norm must be layernorm, rmsnorm, or None")
        boolean_fields = {
            "pre_norm": self.pre_norm,
            "use_qk_norm": self.use_qk_norm,
            "attention_bias": self.attention_bias,
            "tied_output": self.tied_output,
        }
        if any(not isinstance(value, bool) for value in boolean_fields.values()):
            raise ValueError("model boolean fields must contain bool values")
        if self.cross_attention_pre_norm is not None and not isinstance(
            self.cross_attention_pre_norm, bool
        ):
            raise ValueError("cross_attention_pre_norm must be bool or None")
        if (
            isinstance(self.value_clip, bool)
            or not isinstance(self.value_clip, Real)
            or not math.isfinite(self.value_clip)
            or self.value_clip <= 0
        ):
            raise ValueError("value_clip must be positive")
        if (
            not self.diffusion_fractions
            or tuple(sorted(self.diffusion_fractions)) != self.diffusion_fractions
            or len(set(self.diffusion_fractions)) != len(self.diffusion_fractions)
            or self.diffusion_fractions[-1] != 1.0
            or any(
                isinstance(fraction, bool)
                or not isinstance(fraction, Real)
                or not math.isfinite(fraction)
                or not 0 < fraction <= 1
                for fraction in self.diffusion_fractions
            )
        ):
            raise ValueError(
                "diffusion_fractions must be strictly increasing positive values ending at 1.0"
            )
        invalid_priors = {
            name
            for name in DEFAULT_PRIOR_DIMS
            if name not in self.prior_dims
            or isinstance(self.prior_dims[name], bool)
            or not isinstance(self.prior_dims[name], Integral)
            or self.prior_dims[name] <= 0
        }
        if invalid_priors:
            raise ValueError(f"prior_dims are missing or invalid for: {sorted(invalid_priors)}")

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def effective_ffn_dim(self) -> int:
        return self.ffn_dim if self.ffn_dim is not None else self.hidden_dim * 4

    @property
    def effective_cross_attention_ffn_dim(self) -> int:
        return (
            self.cross_attention_ffn_dim
            if self.cross_attention_ffn_dim is not None
            else self.effective_ffn_dim
        )

    @property
    def effective_cross_attention_activation(self) -> str:
        return self.cross_attention_activation or self.activation

    @property
    def effective_cross_attention_norm(self) -> str:
        return self.cross_attention_norm or self.norm

    @property
    def effective_cross_attention_pre_norm(self) -> bool:
        return (
            self.cross_attention_pre_norm
            if self.cross_attention_pre_norm is not None
            else self.pre_norm
        )


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 2
    set_size: int = 8
    max_steps: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_fraction: float = 0.1
    grad_clip_norm: float = 1.0
    seed: int = 0
    device: str = "cpu"
    precision: str = "float32"
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    num_workers: int = 0
    checkpoint_every: int = 0
    log_every: int = 1

    def __post_init__(self) -> None:
        integer_fields = {
            "batch_size": self.batch_size,
            "set_size": self.set_size,
            "max_steps": self.max_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_workers": self.num_workers,
            "checkpoint_every": self.checkpoint_every,
            "log_every": self.log_every,
            "seed": self.seed,
        }
        invalid_integer_fields = [
            name
            for name, value in integer_fields.items()
            if isinstance(value, bool) or not isinstance(value, Integral)
        ]
        if invalid_integer_fields:
            raise ValueError(f"training integer fields have invalid types: {invalid_integer_fields}")
        if self.batch_size <= 0 or self.set_size <= 0 or self.max_steps <= 0:
            raise ValueError("batch_size, set_size, and max_steps must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.precision not in {"float32", "bfloat16", "float16"}:
            raise ValueError("precision must be float32, bfloat16, or float16")
        if not isinstance(self.gradient_checkpointing, bool):
            raise ValueError("gradient_checkpointing must be a bool")
        if self.num_workers < 0 or self.checkpoint_every < 0 or self.log_every <= 0:
            raise ValueError("num_workers/checkpoint_every must be non-negative and log_every positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if (
            isinstance(self.learning_rate, bool)
            or isinstance(self.weight_decay, bool)
            or not isinstance(self.learning_rate, Real)
            or not isinstance(self.weight_decay, Real)
            or not math.isfinite(self.learning_rate)
            or not math.isfinite(self.weight_decay)
            or self.learning_rate <= 0
            or self.weight_decay < 0
        ):
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if (
            isinstance(self.beta1, bool)
            or isinstance(self.beta2, bool)
            or not isinstance(self.beta1, Real)
            or not isinstance(self.beta2, Real)
            or not math.isfinite(self.beta1)
            or not math.isfinite(self.beta2)
            or not 0 <= self.beta1 < 1
            or not 0 <= self.beta2 < 1
        ):
            raise ValueError("Adam beta values must satisfy 0 <= beta < 1")
        if (
            isinstance(self.warmup_fraction, bool)
            or not isinstance(self.warmup_fraction, Real)
            or not math.isfinite(self.warmup_fraction)
            or not 0 <= self.warmup_fraction <= 1
        ):
            raise ValueError("warmup_fraction must satisfy 0 <= warmup_fraction <= 1")
        if (
            isinstance(self.grad_clip_norm, bool)
            or not isinstance(self.grad_clip_norm, Real)
            or not math.isfinite(self.grad_clip_norm)
            or self.grad_clip_norm <= 0
        ):
            raise ValueError("grad_clip_norm must be positive")


def tiny_config() -> ModelConfig:
    return ModelConfig(
        num_genes=32,
        context_length=16,
        num_layers=2,
        hidden_dim=32,
        num_heads=4,
        ffn_dim=64,
        cross_attention_layers=(1,),
        activation="swiglu",
        norm="rmsnorm",
        pre_norm=True,
        attention_bias=False,
        tied_output=True,
    )


def mini_config() -> ModelConfig:
    return ModelConfig(
        num_genes=19_408,
        # Mini uses the complete available feature panel (about 19k genes for
        # X-Atlas; 2k HVGs for the published Replogle benchmark).
        context_length=19_408,
        num_layers=12,
        hidden_dim=512,
        num_heads=8,
        ffn_dim=512,
        cross_attention_layers=(2, 5, 8, 11),
        # Section 4.1.3 specifies a separate Pre-LN SwiGLU sublayer for each
        # cross-attention block.  Together with the Table A2 backbone, this
        # resolves to the 60M Mini parameterization reported in Table A3.
        cross_attention_ffn_dim=2_048,
        cross_attention_activation="swiglu",
        cross_attention_norm="rmsnorm",
        cross_attention_pre_norm=True,
        activation="relu",
        norm="layernorm",
        pre_norm=False,
        use_qk_norm=False,
        attention_bias=True,
        tied_output=False,
    )


def ultra_config() -> ModelConfig:
    return ModelConfig(
        num_genes=19_408,
        context_length=4_000,
        num_layers=34,
        hidden_dim=2560,
        num_heads=40,
        ffn_dim=10_240,
        cross_attention_layers=(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 33),
        activation="swiglu",
        norm="rmsnorm",
        pre_norm=True,
        use_qk_norm=True,
        attention_bias=False,
        tied_output=True,
    )


def model_config(name: str) -> ModelConfig:
    if name == "tiny":
        return tiny_config()
    if name == "mini":
        return mini_config()
    if name == "ultra":
        return ultra_config()
    raise ValueError(f"unknown model config {name!r}; choose from tiny, mini, ultra")
