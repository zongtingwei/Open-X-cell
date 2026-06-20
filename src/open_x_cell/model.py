from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from open_x_cell.config import ModelConfig
from open_x_cell.priors import PRIOR_SOURCES, PriorBatch


@dataclass(frozen=True)
class ModelOutput:
    expression: torch.Tensor
    cell_embedding: torch.Tensor
    prior_reconstructions: dict[str, torch.Tensor]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


def _norm(kind: str, dim: int) -> nn.Module:
    if kind == "rmsnorm":
        return RMSNorm(dim)
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"unknown norm {kind!r}")


class QKNormMultiheadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, bias: bool, use_qk_norm: bool) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_qk_norm = use_qk_norm
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        if use_qk_norm:
            self.q_scale = nn.Parameter(torch.ones(num_heads, self.head_dim))
            self.k_scale = nn.Parameter(torch.ones(num_heads, self.head_dim))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, length, self.hidden_dim)

    def _qk_norm(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        denom = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return x * denom * scale[None, :, None, :]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))
        if self.use_qk_norm:
            q = self._qk_norm(q, self.q_scale)
            k = self._qk_norm(k, self.k_scale)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v)
        return self.out_proj(self._merge_heads(attended))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.activation = config.activation
        ffn_dim = config.effective_ffn_dim
        if self.activation == "swiglu":
            self.gate = nn.Linear(config.hidden_dim, ffn_dim, bias=False)
            self.up = nn.Linear(config.hidden_dim, ffn_dim, bias=False)
            self.down = nn.Linear(ffn_dim, config.hidden_dim, bias=False)
        elif self.activation == "relu":
            self.net = nn.Sequential(
                nn.Linear(config.hidden_dim, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, config.hidden_dim),
            )
        else:
            raise ValueError(f"unknown activation {self.activation!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            return self.down(F.silu(self.gate(x)) * self.up(x))
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.pre_norm = config.pre_norm
        self.attn_norm = _norm(config.norm, config.hidden_dim)
        self.ffn_norm = _norm(config.norm, config.hidden_dim)
        self.attn = QKNormMultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            bias=config.attention_bias,
            use_qk_norm=config.use_qk_norm,
        )
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.attn(self.attn_norm(x), self.attn_norm(x), self.attn_norm(x))
            x = x + self.ffn(self.ffn_norm(x))
            return x
        x = self.attn_norm(x + self.attn(x, x, x))
        return self.ffn_norm(x + self.ffn(x))


class CrossAttentionBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.pre_norm = config.pre_norm
        self.query_norm = _norm(config.norm, config.hidden_dim)
        self.ffn_norm = _norm(config.norm, config.hidden_dim)
        self.attn = QKNormMultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            bias=config.attention_bias,
            use_qk_norm=config.use_qk_norm,
        )
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, context: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        query = self.query_norm(x) if self.pre_norm else x
        x = x + self.attn(query, context, context, key_padding_mask=key_padding_mask)
        if self.pre_norm:
            return x + self.ffn(self.ffn_norm(x))
        return self.ffn_norm(x + self.ffn(x))


class PriorProjector(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict(
            {
                source: nn.Sequential(
                    nn.Linear(config.prior_dims[source], config.hidden_dim),
                    nn.LeakyReLU(0.01),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    _norm("layernorm", config.hidden_dim),
                )
                for source in PRIOR_SOURCES
            }
        )
        self.decoders = nn.ModuleDict(
            {
                source: nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.LeakyReLU(0.01),
                    nn.Linear(config.hidden_dim, config.prior_dims[source]),
                )
                for source in PRIOR_SOURCES
            }
        )

    def forward(self, priors: PriorBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        projected = []
        reconstructions = {}
        for source in PRIOR_SOURCES:
            token = self.projectors[source](priors.values[source])
            projected.append(token)
            reconstructions[source] = self.decoders[source](token)
        return torch.stack(projected, dim=1), reconstructions


class OpenXCellModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.gene_embedding = nn.Embedding(config.num_genes, d)
        self.gene_embedding_norm = _norm("layernorm", d)
        self.value_encoder = nn.Sequential(
            nn.Linear(1, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.Dropout(config.dropout),
            nn.LayerNorm(d),
        )
        self.mask_embedding = nn.Embedding(2, d)
        self.mask_embedding_norm = nn.LayerNorm(d)
        self.cls = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)
        self.prior_projector = PriorProjector(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.cross_blocks = nn.ModuleDict(
            {str(layer): CrossAttentionBlock(config) for layer in config.cross_attention_layers}
        )
        self.final_norm = _norm(config.norm, d)
        self.decoder = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.LeakyReLU(0.01),
            nn.Linear(d, d),
            nn.LeakyReLU(0.01),
            nn.Linear(d, d),
        )
        self.scalar_head = nn.Linear(d, 1)
        self.gene_bias = nn.Embedding(config.num_genes, 1)
        nn.init.zeros_(self.gene_bias.weight)

    def _expand_gene_ids(self, gene_ids: torch.Tensor, batch: int, set_size: int) -> torch.Tensor:
        if gene_ids.dim() == 2:
            gene_ids = gene_ids[:, None, :].expand(batch, set_size, gene_ids.shape[-1])
        if gene_ids.shape != (batch, set_size, self.config.context_length):
            raise ValueError(
                "gene_ids must have shape [batch, context_length] or [batch, set_size, context_length]"
            )
        return gene_ids

    def _encode_tokens(self, x: torch.Tensor, mask: torch.Tensor, gene_ids: torch.Tensor) -> torch.Tensor:
        batch, set_size, genes = x.shape
        flat_gene_ids = gene_ids.reshape(batch * set_size, genes)
        gene = self.gene_embedding_norm(self.gene_embedding(flat_gene_ids))
        value = self.value_encoder(torch.clamp(x.reshape(batch * set_size, genes, 1), max=self.config.value_clip))
        mask_token = self.mask_embedding_norm(self.mask_embedding(mask.long().reshape(batch * set_size, genes)))
        tokens = gene + value + mask_token
        cls = self.cls[None, None, :].expand(batch * set_size, 1, self.config.hidden_dim)
        return torch.cat([cls, tokens], dim=1)

    def _prior_context(
        self,
        priors: PriorBatch,
        perturbation_ids: torch.Tensor,
        batch: int,
        set_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        external, reconstructions = self.prior_projector(priors)
        gene_context = self.gene_embedding(perturbation_ids).detach()[:, None, :]
        context = torch.cat([external, gene_context], dim=1)
        gene_missing = torch.zeros(batch, 1, dtype=torch.bool, device=priors.missing.device)
        missing = torch.cat([priors.missing, gene_missing], dim=1)
        context = context[:, None, :, :].expand(batch, set_size, context.shape[1], context.shape[2])
        missing = missing[:, None, :].expand(batch, set_size, missing.shape[1])
        return context.reshape(batch * set_size, context.shape[2], context.shape[3]), missing.reshape(
            batch * set_size, missing.shape[2]
        ), reconstructions

    def _decode(self, h: torch.Tensor, gene_ids: torch.Tensor, batch: int, set_size: int) -> torch.Tensor:
        gene_h = h[:, 1:, :]
        cell_h = h[:, :1, :].expand(-1, gene_h.shape[1], -1)
        decoded = self.decoder(torch.cat([gene_h, cell_h], dim=-1))
        flat_gene_ids = gene_ids.reshape(batch * set_size, self.config.context_length)
        if self.config.tied_output:
            raw_gene = self.gene_embedding(flat_gene_ids)
            bias = self.gene_bias(flat_gene_ids).squeeze(-1)
            expression = (decoded * raw_gene).sum(dim=-1) / math.sqrt(self.config.hidden_dim) + bias
        else:
            expression = self.scalar_head(decoded).squeeze(-1)
        return expression.reshape(batch, set_size, self.config.context_length)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        gene_ids: torch.Tensor,
        perturbation_ids: torch.Tensor,
        priors: PriorBatch,
    ) -> ModelOutput:
        if x.shape != mask.shape:
            raise ValueError("x and mask must have the same shape")
        batch, set_size, genes = x.shape
        if genes != self.config.context_length:
            raise ValueError("input gene dimension must equal config.context_length")
        gene_ids = self._expand_gene_ids(gene_ids.to(x.device), batch, set_size)
        perturbation_ids = perturbation_ids.to(x.device)
        priors = priors.to(x.device)
        h = self._encode_tokens(x, mask.to(x.device), gene_ids)
        context, missing, reconstructions = self._prior_context(priors, perturbation_ids, batch, set_size)

        for layer_idx, block in enumerate(self.blocks):
            h = block(h)
            layer_key = str(layer_idx)
            if layer_key in self.cross_blocks:
                h = self.cross_blocks[layer_key](h, context, missing)
        h = self.final_norm(h)
        expression = self._decode(h, gene_ids, batch, set_size)
        cell_embedding = h[:, 0, :].reshape(batch, set_size, self.config.hidden_dim)
        return ModelOutput(
            expression=expression,
            cell_embedding=cell_embedding,
            prior_reconstructions=reconstructions,
        )
