"""scGPT warm-start: initialize an Open-X-Cell model from a pretrained scGPT checkpoint.

Implements the paper's scGPT warm-start beyond a bare gene-embedding copy:

  * multi-level gene-id resolution (symbol direct, or Ensembl -> symbol via a map),
  * an overlap check (warn / raise below a threshold),
  * a fallback embedding (the ``<pad>`` token) for unmapped genes, and
  * strict=False transfer of the transformer stack (gene embedding + its norm,
    the value encoder, and every encoder block with the fused ``Wqkv`` split into
    separate q/k/v projections) when the architecture is scGPT-compatible
    (standard MHA / ReLU FFN / LayerNorm / post-LN / ffn_dim == hidden_dim).

Reads ``vocab.json`` (plain JSON) and ``best_model.pt`` (a torch state dict) directly,
so the ``scgpt`` package is NOT required.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class WarmStartStats:
    n_genes: int
    n_mapped: int
    n_via_symbol: int
    n_via_ensembl: int
    n_fallback: int
    transformer_loaded: bool
    loaded_tensors: int
    skipped_tensors: int
    notes: list[str] = field(default_factory=list)

    @property
    def overlap(self) -> float:
        return self.n_mapped / self.n_genes if self.n_genes else 0.0


def _is_ensembl(gene: str) -> bool:
    return gene.startswith("ENSG") or gene.startswith("ENSMUSG")


def load_scgpt(model_dir: str | Path) -> tuple[dict[str, int], dict, int]:
    """Return (vocab symbol->id, state_dict, embedding_dim) from a scGPT model dir."""
    model_dir = Path(model_dir)
    with open(model_dir / "vocab.json") as fh:
        vocab = {str(k): int(v) for k, v in json.load(fh).items()}
    state = torch.load(model_dir / "best_model.pt", map_location="cpu")
    emb = state["encoder.embedding.weight"]
    return vocab, state, int(emb.shape[1])


def _first_symbol(value) -> str | None:
    """ens2sym values may be a str, or a list/tuple/ndarray of candidate symbols."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        for item in value:
            if item is not None and str(item):
                return str(item)
        return None
    text = str(value)
    return text or None


def _normalize_ens2sym(mapping) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in dict(mapping).items():
        symbol = _first_symbol(v)
        if symbol is None:
            continue
        out[str(k)] = symbol
        out[str(k).split(".")[0]] = symbol  # tolerate version suffixes (ENSG...\.3)
    return out


def resolve_gene_matrix(
    var_names: list[str],
    vocab: dict[str, int],
    embedding: torch.Tensor,
    ens2sym: dict[str, str] | None,
    fallback_token: str,
) -> tuple[torch.Tensor, WarmStartStats]:
    """Build an (n_genes, dim) matrix of scGPT embeddings aligned to ``var_names``."""
    dim = embedding.shape[1]
    fallback_id = vocab.get(fallback_token)
    if fallback_id is None:
        raise ValueError(f"fallback token {fallback_token!r} not in scGPT vocab")
    fallback_vec = embedding[fallback_id]
    ens2sym = ens2sym or {}

    matrix = torch.empty(len(var_names), dim, dtype=torch.float32)
    n_sym = n_ens = n_fb = 0
    for i, gene in enumerate(var_names):
        gene = str(gene)
        token_id = vocab.get(gene)
        if token_id is not None:
            matrix[i] = embedding[token_id]
            n_sym += 1
            continue
        if _is_ensembl(gene):
            symbol = ens2sym.get(gene) or ens2sym.get(gene.split(".")[0])
            if symbol is not None and symbol in vocab:
                matrix[i] = embedding[vocab[symbol]]
                n_ens += 1
                continue
        matrix[i] = fallback_vec  # unmapped -> <pad> token embedding
        n_fb += 1

    stats = WarmStartStats(
        n_genes=len(var_names), n_mapped=n_sym + n_ens,
        n_via_symbol=n_sym, n_via_ensembl=n_ens, n_fallback=n_fb,
        transformer_loaded=False, loaded_tensors=0, skipped_tensors=0,
    )
    return matrix, stats


def _scgpt_compatible(config) -> bool:
    return (
        config.activation == "relu"
        and config.norm == "layernorm"
        and not config.pre_norm
        and not config.use_qk_norm
        and config.attention_bias
        and config.effective_ffn_dim == config.hidden_dim
    )


def _transformer_remap(state: dict, model, config) -> dict[str, torch.Tensor]:
    """Map scGPT params -> Open-X-Cell param names (value encoder + encoder blocks)."""
    d = config.hidden_dim
    own = dict(model.named_parameters())
    out: dict[str, torch.Tensor] = {}

    # gene-embedding LayerNorm (scGPT enc_norm) and value encoder (identical structure)
    pairs = {
        "gene_embedding_norm.weight": "encoder.enc_norm.weight",
        "gene_embedding_norm.bias": "encoder.enc_norm.bias",
        "value_encoder.0.weight": "value_encoder.linear1.weight",
        "value_encoder.0.bias": "value_encoder.linear1.bias",
        "value_encoder.2.weight": "value_encoder.linear2.weight",
        "value_encoder.2.bias": "value_encoder.linear2.bias",
        "value_encoder.4.weight": "value_encoder.norm.weight",
        "value_encoder.4.bias": "value_encoder.norm.bias",
    }
    for dst, src in pairs.items():
        if src in state and dst in own and state[src].shape == own[dst].shape:
            out[dst] = state[src].float()

    # encoder blocks: split fused Wqkv -> q/k/v, FFN, norms
    n_src = sum(1 for k in state if k.startswith("transformer_encoder.layers."))
    n_src_layers = len({k.split(".")[2] for k in state if k.startswith("transformer_encoder.layers.")})
    for layer in range(min(config.num_layers, n_src_layers)):
        p = f"transformer_encoder.layers.{layer}."
        if p + "self_attn.Wqkv.weight" not in state:
            continue
        wqkv = state[p + "self_attn.Wqkv.weight"]  # (3d, d)
        bqkv = state[p + "self_attn.Wqkv.bias"]     # (3d,)
        q_w, k_w, v_w = wqkv[:d], wqkv[d:2 * d], wqkv[2 * d:]
        q_b, k_b, v_b = bqkv[:d], bqkv[d:2 * d], bqkv[2 * d:]
        block = {
            f"blocks.{layer}.attn.q_proj.weight": q_w, f"blocks.{layer}.attn.q_proj.bias": q_b,
            f"blocks.{layer}.attn.k_proj.weight": k_w, f"blocks.{layer}.attn.k_proj.bias": k_b,
            f"blocks.{layer}.attn.v_proj.weight": v_w, f"blocks.{layer}.attn.v_proj.bias": v_b,
            f"blocks.{layer}.attn.out_proj.weight": state[p + "self_attn.out_proj.weight"],
            f"blocks.{layer}.attn.out_proj.bias": state[p + "self_attn.out_proj.bias"],
            f"blocks.{layer}.ffn.net.0.weight": state[p + "linear1.weight"],
            f"blocks.{layer}.ffn.net.0.bias": state[p + "linear1.bias"],
            f"blocks.{layer}.ffn.net.2.weight": state[p + "linear2.weight"],
            f"blocks.{layer}.ffn.net.2.bias": state[p + "linear2.bias"],
            f"blocks.{layer}.attn_norm.weight": state[p + "norm1.weight"],
            f"blocks.{layer}.attn_norm.bias": state[p + "norm1.bias"],
            f"blocks.{layer}.ffn_norm.weight": state[p + "norm2.weight"],
            f"blocks.{layer}.ffn_norm.bias": state[p + "norm2.bias"],
        }
        for dst, tensor in block.items():
            if dst in own and tensor.shape == own[dst].shape:
                out[dst] = tensor.float()
    return out


def warm_start_from_scgpt(
    model,
    var_names: list[str],
    model_dir: str | Path,
    ens2sym: dict[str, str] | None = None,
    min_overlap: float = 0.95,
    fallback_token: str = "<pad>",
    load_transformer: bool = True,
    strict_overlap: bool = False,
) -> WarmStartStats:
    """Warm-start ``model`` (in place) from a pretrained scGPT checkpoint.

    ``var_names`` are the model's gene order (row i of the gene embedding). ``ens2sym``
    maps Ensembl ids to symbols (needed when var_names are Ensembl). Genes that cannot
    be mapped fall back to the ``<pad>`` token embedding. Raises (or warns) when the
    mapped fraction is below ``min_overlap``.
    """
    vocab, state, dim = load_scgpt(model_dir)
    own_hidden = model.gene_embedding.weight.shape[1]
    if dim != own_hidden:
        raise ValueError(
            f"scGPT embedding dim {dim} != model hidden_dim {own_hidden}; "
            "use a matching (512-d) architecture."
        )
    if ens2sym is not None and not isinstance(ens2sym, dict):
        ens2sym = _normalize_ens2sym(ens2sym)
    elif isinstance(ens2sym, dict):
        ens2sym = _normalize_ens2sym(ens2sym)

    embedding = state["encoder.embedding.weight"].float()
    matrix, stats = resolve_gene_matrix(var_names, vocab, embedding, ens2sym, fallback_token)

    if stats.overlap < min_overlap:
        msg = (f"scGPT gene overlap {stats.overlap:.1%} < required {min_overlap:.0%} "
               f"({stats.n_mapped}/{stats.n_genes} mapped)")
        if strict_overlap:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
        stats.notes.append(msg)

    with torch.no_grad():
        rows = min(model.gene_embedding.weight.shape[0], len(var_names))
        model.gene_embedding.weight[:rows].copy_(matrix[:rows].to(model.gene_embedding.weight.device))

    if load_transformer:
        if _scgpt_compatible(model.config):
            remap = _transformer_remap(state, model, model.config)
            own = dict(model.named_parameters())
            with torch.no_grad():
                for name, tensor in remap.items():
                    own[name].copy_(tensor.to(own[name].device))
            stats.transformer_loaded = bool(remap)
            stats.loaded_tensors = len(remap)
            stats.skipped_tensors = len(own) - len(remap)
        else:
            stats.notes.append(
                "architecture not scGPT-compatible (needs relu/layernorm/post-LN/no-qk-norm/"
                "ffn==hidden); only gene embeddings transferred."
            )
    return stats
