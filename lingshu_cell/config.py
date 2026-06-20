from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class ModelConfig:
    gene_length: int = 18_080
    expression_vocab_size: int = 281
    condition_vocab_size: int = 512
    embedding_dim: int = 640
    num_layers: int = 13
    num_heads: int = 10
    ffn_dim: int = 2_560
    compression_group_size: int = 8
    dropout: float = 0.0
    permutation_seed: int = 0

    def __post_init__(self) -> None:
        if self.gene_length <= 0:
            raise ValueError("gene_length must be positive")
        if self.expression_vocab_size <= 0:
            raise ValueError("expression_vocab_size must be positive")
        if self.condition_vocab_size < 0:
            raise ValueError("condition_vocab_size cannot be negative")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.compression_group_size <= 0:
            raise ValueError("compression_group_size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def mask_token_id(self) -> int:
        return self.expression_vocab_size

    @property
    def condition_token_start(self) -> int:
        return self.mask_token_id + 1

    @property
    def total_vocab_size(self) -> int:
        return self.condition_token_start + self.condition_vocab_size

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.num_heads

    @property
    def compressed_gene_length(self) -> int:
        return ceil(self.gene_length / self.compression_group_size)


@dataclass(frozen=True)
class TrainingConfig:
    peak_lr: float = 2e-4
    min_lr: float = 2e-5
    warmup_steps: int = 1_000
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    batch_size: int = 256
    ema_rel_sigma: float = 0.05

    def __post_init__(self) -> None:
        if self.peak_lr <= 0:
            raise ValueError("peak_lr must be positive")
        if self.min_lr < 0:
            raise ValueError("min_lr cannot be negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


def paper_model_config(task: str = "unconditional") -> ModelConfig:
    group_size = 32 if task == "genetic" else 8
    if task not in {"unconditional", "genetic", "cytokine"}:
        raise ValueError("task must be one of: unconditional, genetic, cytokine")
    return ModelConfig(
        gene_length=18_080,
        expression_vocab_size=281,
        condition_vocab_size=512,
        embedding_dim=640,
        num_layers=13,
        num_heads=10,
        ffn_dim=2_560,
        compression_group_size=group_size,
    )


def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        gene_length=16,
        expression_vocab_size=281,
        condition_vocab_size=32,
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        compression_group_size=4,
    )

