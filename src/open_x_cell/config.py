from __future__ import annotations

from dataclasses import dataclass, field


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
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.context_length > self.num_genes:
            raise ValueError("context_length cannot exceed num_genes")
        if any(layer < 0 or layer >= self.num_layers for layer in self.cross_attention_layers):
            raise ValueError("cross_attention_layers must be valid layer indices")

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def effective_ffn_dim(self) -> int:
        return self.ffn_dim if self.ffn_dim is not None else self.hidden_dim * 4


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 2
    set_size: int = 8
    max_steps: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_fraction: float = 0.1
    grad_clip_norm: float = 1.0
    seed: int = 0
    device: str = "cpu"


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
        context_length=4_000,
        num_layers=12,
        hidden_dim=512,
        num_heads=8,
        ffn_dim=512,
        cross_attention_layers=(2, 5, 8, 11),
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
