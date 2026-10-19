import json

import numpy as np
import pytest
import torch

from open_x_cell.config import ModelConfig
from open_x_cell.model import OpenXCellModel
from open_x_cell.scgpt import (
    _first_symbol,
    _normalize_ens2sym,
    resolve_gene_matrix,
    warm_start_from_scgpt,
)


def _mini_compatible_config(num_genes=12):
    return ModelConfig(
        num_genes=num_genes, context_length=num_genes,
        num_layers=2, hidden_dim=16, num_heads=2, ffn_dim=16,
        cross_attention_layers=(1,),
        activation="relu", norm="layernorm", pre_norm=False,
        use_qk_norm=False, attention_bias=True, tied_output=False,
    )


def _fake_scgpt_dir(tmp_path, dim=16, n_layers=2):
    # vocab: symbols A..E plus specials
    symbols = ["GENEA", "GENEB", "GENEC", "GENED", "GENEE"]
    vocab = {s: i for i, s in enumerate(symbols)}
    vocab["<pad>"] = len(vocab)
    vocab["<cls>"] = len(vocab)
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    n_vocab = len(vocab)
    state = {
        "encoder.embedding.weight": torch.randn(n_vocab, dim),
        "encoder.enc_norm.weight": torch.randn(dim),
        "encoder.enc_norm.bias": torch.randn(dim),
        "value_encoder.linear1.weight": torch.randn(dim, 1),
        "value_encoder.linear1.bias": torch.randn(dim),
        "value_encoder.linear2.weight": torch.randn(dim, dim),
        "value_encoder.linear2.bias": torch.randn(dim),
        "value_encoder.norm.weight": torch.randn(dim),
        "value_encoder.norm.bias": torch.randn(dim),
    }
    for layer in range(n_layers):
        p = f"transformer_encoder.layers.{layer}."
        state[p + "self_attn.Wqkv.weight"] = torch.randn(3 * dim, dim)
        state[p + "self_attn.Wqkv.bias"] = torch.randn(3 * dim)
        state[p + "self_attn.out_proj.weight"] = torch.randn(dim, dim)
        state[p + "self_attn.out_proj.bias"] = torch.randn(dim)
        state[p + "linear1.weight"] = torch.randn(dim, dim)
        state[p + "linear1.bias"] = torch.randn(dim)
        state[p + "linear2.weight"] = torch.randn(dim, dim)
        state[p + "linear2.bias"] = torch.randn(dim)
        state[p + "norm1.weight"] = torch.randn(dim)
        state[p + "norm1.bias"] = torch.randn(dim)
        state[p + "norm2.weight"] = torch.randn(dim)
        state[p + "norm2.bias"] = torch.randn(dim)
    torch.save(state, tmp_path / "best_model.pt")
    return tmp_path, vocab, state, symbols


def test_first_symbol_handles_list_and_scalar():
    assert _first_symbol(["DPM1"]) == "DPM1"
    assert _first_symbol(np.array(["X", "Y"])) == "X"
    assert _first_symbol("SCYL3") == "SCYL3"
    assert _first_symbol(None) is None
    assert _first_symbol([]) is None


def test_normalize_ens2sym_strips_version_and_unwraps_list():
    m = _normalize_ens2sym({"ENSG00000000419.3": ["DPM1"]})
    assert m["ENSG00000000419.3"] == "DPM1"
    assert m["ENSG00000000419"] == "DPM1"


def test_resolve_gene_matrix_multilevel_and_fallback(tmp_path):
    _, vocab, state, symbols = _fake_scgpt_dir(tmp_path)
    emb = state["encoder.embedding.weight"]
    ens2sym = {"ENSG0001": "GENEB"}
    var_names = ["GENEA", "ENSG0001", "MISSING"]  # direct symbol, ensembl->symbol, unmapped
    matrix, stats = resolve_gene_matrix(var_names, vocab, emb, ens2sym, "<pad>")
    assert stats.n_via_symbol == 1 and stats.n_via_ensembl == 1 and stats.n_fallback == 1
    assert torch.allclose(matrix[0], emb[vocab["GENEA"]])
    assert torch.allclose(matrix[1], emb[vocab["GENEB"]])
    assert torch.allclose(matrix[2], emb[vocab["<pad>"]])  # fallback


def test_warm_start_transfers_embeddings_and_transformer(tmp_path):
    model_dir, vocab, state, symbols = _fake_scgpt_dir(tmp_path)
    cfg = _mini_compatible_config(num_genes=5)
    model = OpenXCellModel(cfg)
    var_names = symbols  # all map directly by symbol
    q_before = model.blocks[0].attn.q_proj.weight.clone()
    value_before = model.value_encoder[0].weight.detach().clone()
    stats = warm_start_from_scgpt(model, var_names, model_dir, ens2sym={}, min_overlap=0.9)
    assert stats.n_mapped == 5 and stats.n_fallback == 0
    assert stats.transformer_loaded and stats.loaded_tensors > 0
    assert not stats.value_encoder_loaded
    # gene embedding row 0 == scGPT GENEA embedding
    assert torch.allclose(model.gene_embedding.weight[0], state["encoder.embedding.weight"][vocab["GENEA"]])
    # fused Wqkv was split correctly into q_proj
    assert torch.allclose(model.blocks[0].attn.q_proj.weight,
                          state["transformer_encoder.layers.0.self_attn.Wqkv.weight"][:cfg.hidden_dim])
    assert not torch.allclose(q_before, model.blocks[0].attn.q_proj.weight)
    assert torch.equal(model.value_encoder[0].weight, value_before)


def test_warm_start_value_encoder_is_explicit_legacy_opt_in(tmp_path):
    model_dir, _, state, symbols = _fake_scgpt_dir(tmp_path)
    model = OpenXCellModel(_mini_compatible_config(num_genes=5))

    stats = warm_start_from_scgpt(
        model,
        symbols,
        model_dir,
        ens2sym={},
        min_overlap=0.9,
        load_value_encoder=True,
    )

    assert stats.value_encoder_loaded
    assert torch.allclose(
        model.value_encoder[0].weight,
        state["value_encoder.linear1.weight"],
    )


def test_warm_start_preserves_random_embedding_for_unmapped_gene(tmp_path):
    model_dir, vocab, state, _ = _fake_scgpt_dir(tmp_path)
    model = OpenXCellModel(_mini_compatible_config(num_genes=2))
    missing_before = model.gene_embedding.weight[1].detach().clone()

    stats = warm_start_from_scgpt(
        model,
        ["GENEA", "MISSING"],
        model_dir,
        ens2sym={},
        min_overlap=0.4,
    )

    assert stats.n_mapped == 1 and stats.n_fallback == 1
    assert torch.allclose(
        model.gene_embedding.weight[0],
        state["encoder.embedding.weight"][vocab["GENEA"]],
    )
    assert torch.allclose(model.gene_embedding.weight[1], missing_before)


def test_warm_start_rejects_incomplete_transformer_without_mutating_model(tmp_path):
    model_dir, _, state, symbols = _fake_scgpt_dir(tmp_path)
    del state["transformer_encoder.layers.0.linear1.weight"]
    torch.save(state, model_dir / "best_model.pt")
    model = OpenXCellModel(_mini_compatible_config(num_genes=5))
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    with pytest.raises(ValueError, match="incomplete scGPT transformer layer 0"):
        warm_start_from_scgpt(model, symbols, model_dir, ens2sym={}, min_overlap=0.9)

    assert all(torch.equal(before[name], tensor) for name, tensor in model.state_dict().items())


def test_warm_start_skips_transformer_for_incompatible_arch(tmp_path):
    model_dir, *_ = _fake_scgpt_dir(tmp_path)
    cfg = ModelConfig(
        num_genes=5, context_length=5, num_layers=2, hidden_dim=16, num_heads=2,
        ffn_dim=64, cross_attention_layers=(1,), activation="swiglu", norm="rmsnorm",
        pre_norm=True, use_qk_norm=True, attention_bias=False, tied_output=True,
    )
    model = OpenXCellModel(cfg)
    stats = warm_start_from_scgpt(model, ["GENEA", "GENEB", "GENEC", "GENED", "GENEE"],
                                  model_dir, ens2sym={}, min_overlap=0.9)
    assert not stats.transformer_loaded
    assert any("not scGPT-compatible" in n for n in stats.notes)
