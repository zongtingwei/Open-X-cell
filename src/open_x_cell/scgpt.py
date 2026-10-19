"""scGPT warm-start: initialize an Open-X-Cell model from a pretrained scGPT checkpoint.

Implements the paper's scGPT warm-start beyond a bare gene-embedding copy:

  * multi-level gene-id resolution (symbol direct, or Ensembl -> symbol via a map),
  * an overlap check (warn / raise below a threshold),
  * preservation of random initialization for unmapped genes, and
  * strict=False transfer of the transformer stack (gene embedding + its norm
    and every encoder block with the fused ``Wqkv`` split into separate q/k/v
    projections) when the architecture is scGPT-compatible
    (standard MHA / ReLU FFN / LayerNorm / post-LN / ffn_dim == hidden_dim).

The scGPT value encoder is deliberately excluded by default: Appendix A.7 names
gene embeddings and transformer weights, but does not name value-encoder weights.
It remains an explicit opt-in for sensitivity/legacy reproduction.

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
    value_encoder_loaded: bool
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
    if not vocab:
        raise ValueError(f"scGPT vocabulary is empty: {model_dir / 'vocab.json'}")
    if len(set(vocab.values())) != len(vocab):
        raise ValueError("scGPT vocabulary contains duplicate token indices")
    state = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError("scGPT checkpoint must contain a state-dict mapping")
    if "encoder.embedding.weight" not in state:
        raise ValueError("scGPT checkpoint lacks encoder.embedding.weight")
    emb = state["encoder.embedding.weight"]
    if not torch.is_tensor(emb) or emb.dim() != 2:
        raise ValueError("scGPT encoder.embedding.weight must be a matrix")
    if not torch.is_floating_point(emb) or not torch.isfinite(emb).all():
        raise ValueError("scGPT encoder.embedding.weight must contain finite floating-point values")
    if any(index < 0 or index >= emb.shape[0] for index in vocab.values()):
        raise ValueError("scGPT vocabulary contains an out-of-range token index")
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
    fallback_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, WarmStartStats]:
    """Build an (n_genes, dim) matrix of scGPT embeddings aligned to ``var_names``."""
    dim = embedding.shape[1]
    fallback_id = vocab.get(fallback_token)
    if fallback_id is None:
        raise ValueError(f"fallback token {fallback_token!r} not in scGPT vocab")
    fallback_vec = embedding[fallback_id]
    ens2sym = ens2sym or {}
    if fallback_matrix is not None and fallback_matrix.shape != (len(var_names), dim):
        raise ValueError(
            f"fallback_matrix has shape {tuple(fallback_matrix.shape)}, "
            f"expected {(len(var_names), dim)}"
        )

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
        # Appendix A.7 specifies that genes unmapped during initial scGPT
        # warm-start retain their random initialization.  The <pad> fallback is
        # kept as the standalone helper's backwards-compatible default.
        matrix[i] = fallback_matrix[i] if fallback_matrix is not None else fallback_vec
        n_fb += 1

    stats = WarmStartStats(
        n_genes=len(var_names), n_mapped=n_sym + n_ens,
        n_via_symbol=n_sym, n_via_ensembl=n_ens, n_fallback=n_fb,
        transformer_loaded=False, loaded_tensors=0, skipped_tensors=0,
        value_encoder_loaded=False,
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


def _transformer_remap(
    state: dict,
    model,
    config,
    *,
    load_value_encoder: bool = False,
) -> dict[str, torch.Tensor]:
    """Map scGPT parameters to the embedding norm and encoder blocks."""
    d = config.hidden_dim
    own = dict(model.named_parameters())
    out: dict[str, torch.Tensor] = {}

    # The embedding LayerNorm is part of the gene-embedding path. The value
    # encoder is structurally compatible but is not among the weights named in
    # Appendix A.7, so it is available only as an explicit legacy sensitivity.
    pairs = {
        "gene_embedding_norm.weight": "encoder.enc_norm.weight",
        "gene_embedding_norm.bias": "encoder.enc_norm.bias",
    }
    if load_value_encoder:
        pairs.update(
            {
                "value_encoder.0.weight": "value_encoder.linear1.weight",
                "value_encoder.0.bias": "value_encoder.linear1.bias",
                "value_encoder.2.weight": "value_encoder.linear2.weight",
                "value_encoder.2.bias": "value_encoder.linear2.bias",
                "value_encoder.4.weight": "value_encoder.norm.weight",
                "value_encoder.4.bias": "value_encoder.norm.bias",
            }
        )
    for dst, src in pairs.items():
        if src not in state:
            continue
        tensor = state[src]
        if not torch.is_tensor(tensor) or dst not in own or tensor.shape != own[dst].shape:
            raise ValueError(f"incompatible scGPT transformer tensor: {src}")
        out[dst] = tensor.float()

    # encoder blocks: split fused Wqkv -> q/k/v, FFN, norms
    required_suffixes = (
        "self_attn.Wqkv.weight",
        "self_attn.Wqkv.bias",
        "self_attn.out_proj.weight",
        "self_attn.out_proj.bias",
        "linear1.weight",
        "linear1.bias",
        "linear2.weight",
        "linear2.bias",
        "norm1.weight",
        "norm1.bias",
        "norm2.weight",
        "norm2.bias",
    )
    for layer in range(config.num_layers):
        p = f"transformer_encoder.layers.{layer}."
        present = [key for key in state if key.startswith(p)]
        if not present:
            continue
        missing = [suffix for suffix in required_suffixes if p + suffix not in state]
        if missing:
            raise ValueError(
                f"incomplete scGPT transformer layer {layer}; missing tensors: {missing}"
            )
        wqkv = state[p + "self_attn.Wqkv.weight"]  # (3d, d)
        bqkv = state[p + "self_attn.Wqkv.bias"]     # (3d,)
        if (
            not torch.is_tensor(wqkv)
            or not torch.is_tensor(bqkv)
            or wqkv.shape != (3 * d, d)
            or bqkv.shape != (3 * d,)
        ):
            raise ValueError(f"scGPT layer {layer} has incompatible fused QKV tensors")
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
            if not torch.is_tensor(tensor) or dst not in own or tensor.shape != own[dst].shape:
                raise ValueError(f"incompatible scGPT transformer tensor mapped to {dst}")
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
    load_value_encoder: bool = False,
    strict_overlap: bool = False,
) -> WarmStartStats:
    """Warm-start ``model`` (in place) from a pretrained scGPT checkpoint.

    ``var_names`` are the model's gene order (row i of the gene embedding). ``ens2sym``
    maps Ensembl ids to symbols (needed when var_names are Ensembl). Genes that cannot
    be mapped retain the model's random initialization (Appendix A.7). Raises
    (or warns) when the mapped fraction is below ``min_overlap``.
    """
    if not 0 <= min_overlap <= 1:
        raise ValueError("min_overlap must satisfy 0 <= value <= 1")
    if not isinstance(load_transformer, bool) or not isinstance(load_value_encoder, bool):
        raise ValueError("scGPT transfer switches must be bool values")
    if load_value_encoder and not load_transformer:
        raise ValueError("scGPT value-encoder transfer requires transformer transfer")
    if len(var_names) != len(set(map(str, var_names))):
        raise ValueError("warm-start gene vocabulary contains duplicates")
    if len(var_names) > model.gene_embedding.num_embeddings:
        raise ValueError("warm-start gene vocabulary exceeds the model embedding table")
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
    random_initialization = model.gene_embedding.weight[: len(var_names)].detach().cpu().float()
    matrix, stats = resolve_gene_matrix(
        var_names,
        vocab,
        embedding,
        ens2sym,
        fallback_token,
        fallback_matrix=random_initialization,
    )
    if not torch.isfinite(matrix).all():
        raise ValueError("resolved scGPT gene embeddings contain non-finite values")

    if stats.overlap < min_overlap:
        msg = (f"scGPT gene overlap {stats.overlap:.1%} < required {min_overlap:.0%} "
               f"({stats.n_mapped}/{stats.n_genes} mapped)")
        if strict_overlap:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
        stats.notes.append(msg)

    remap: dict[str, torch.Tensor] = {}
    own = dict(model.named_parameters())
    if load_transformer:
        if _scgpt_compatible(model.config):
            remap = _transformer_remap(
                state,
                model,
                model.config,
                load_value_encoder=load_value_encoder,
            )
            non_finite = [
                name
                for name, tensor in remap.items()
                if not torch.is_floating_point(tensor) or not torch.isfinite(tensor).all()
            ]
            if non_finite:
                raise ValueError(
                    f"scGPT transformer mapping contains invalid tensors: {non_finite[:5]}"
                )
            stats.transformer_loaded = any(name.startswith("blocks.") for name in remap)
            stats.value_encoder_loaded = any(
                name.startswith("value_encoder.") for name in remap
            )
            stats.loaded_tensors = len(remap)
            stats.skipped_tensors = len(own) - len(remap)
        else:
            stats.notes.append(
                "architecture not scGPT-compatible (needs relu/layernorm/post-LN/no-qk-norm/"
                "ffn==hidden); only gene embeddings transferred."
            )
    rows = min(model.gene_embedding.weight.shape[0], len(var_names))
    prepared_matrix = matrix[:rows].to(
        device=model.gene_embedding.weight.device,
        dtype=model.gene_embedding.weight.dtype,
    )
    prepared_remap = {
        name: tensor.to(device=own[name].device, dtype=own[name].dtype)
        for name, tensor in remap.items()
    }
    # All checkpoint structure, shapes, values, and device conversions are
    # validated before the first in-place copy. A malformed transformer cannot
    # therefore leave only the gene embedding partially initialized.
    with torch.no_grad():
        model.gene_embedding.weight[:rows].copy_(prepared_matrix)
        for name, tensor in prepared_remap.items():
            own[name].copy_(tensor)
    return stats
