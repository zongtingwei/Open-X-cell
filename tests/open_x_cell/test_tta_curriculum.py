import numpy as np
import pytest
import torch

from open_x_cell.config import TrainingConfig, tiny_config
from open_x_cell.model import OpenXCellModel
from open_x_cell.synthetic import make_synthetic_adata
from open_x_cell.training import (
    high_effect_perturbations,
    load_checkpoint,
    train_model_curriculum,
    train_tiny,
)
from open_x_cell.tta import adapt_model_tta


def test_high_effect_perturbations_orders_by_effect():
    adata = make_synthetic_adata(n_cells=200, n_genes=32, n_perturbations=8, seed=1)
    top = high_effect_perturbations(adata, top_frac=0.25)
    assert 1 <= len(top) <= 2  # 25% of 8 perturbations (rounded), at least one
    assert all(p != "control" for p in top)
    # selecting a larger fraction is a superset of the smaller one's strongest hits
    more = high_effect_perturbations(adata, top_frac=0.5)
    assert set(top).issubset(set(more))


def test_curriculum_runs_two_phases(tmp_path):
    adata = make_synthetic_adata(n_cells=200, n_genes=32, n_perturbations=8, seed=2)
    path = tmp_path / "data.h5ad"
    adata.write_h5ad(path)
    tcfg = TrainingConfig(batch_size=2, set_size=8, learning_rate=1e-3, seed=0, device="cpu")
    res = train_model_curriculum(
        path, tmp_path / "out", model_size="tiny",
        phase1_steps=5, phase2_steps=3, top_frac=0.25,
        phase2_learning_rate=5e-4, training_config=tcfg, normalize="none",
    )
    assert len(res.history) == 8  # 5 + 3
    model, _, var_names = load_checkpoint(res.checkpoint)
    assert len(var_names) == 32


def test_tta_adapts_and_freezes_cross_attention():
    cfg = tiny_config()
    model = OpenXCellModel(cfg)
    control = np.random.RandomState(0).randn(40, cfg.context_length).astype(np.float32)

    cross_before = [p.detach().clone() for p in model.cross_blocks.parameters()]
    gene_emb_before = model.gene_embedding.weight.detach().clone()

    history = adapt_model_tta(model, cfg, control, steps=8, learning_rate=1e-3,
                              set_size=8, seed=0, device="cpu")
    assert len(history) == 8 and all(np.isfinite(history))

    # cross-attention blocks frozen -> unchanged; and requires_grad restored afterwards
    for before, p in zip(cross_before, model.cross_blocks.parameters()):
        assert torch.allclose(before, p)
        assert p.requires_grad
    assert model.training
    # a non-frozen parameter did update
    assert not torch.allclose(gene_emb_before, model.gene_embedding.weight)


def test_tta_rejects_negative_seed():
    cfg = tiny_config()
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        adapt_model_tta(
            OpenXCellModel(cfg),
            cfg,
            np.zeros((8, cfg.context_length), dtype=np.float32),
            steps=1,
            set_size=4,
            seed=-1,
        )


def test_tta_validation_failure_preserves_model_state_flags_and_mode():
    cfg = tiny_config()
    model = OpenXCellModel(cfg)
    model.eval()
    flags_before = [parameter.requires_grad for parameter in model.parameters()]

    with pytest.raises(ValueError, match="one entry per control matrix column"):
        adapt_model_tta(
            model,
            cfg,
            np.zeros((8, cfg.context_length), dtype=np.float32),
            steps=1,
            set_size=4,
            gene_ids=np.arange(cfg.context_length - 1),
        )

    assert not model.training
    assert [parameter.requires_grad for parameter in model.parameters()] == flags_before


def test_tta_rejects_boolean_learning_rate():
    cfg = tiny_config()
    with pytest.raises(ValueError, match="learning_rate"):
        adapt_model_tta(
            OpenXCellModel(cfg),
            cfg,
            np.zeros((8, cfg.context_length), dtype=np.float32),
            steps=1,
            learning_rate=True,
            set_size=4,
        )


def test_predict_with_tta_runs(tmp_path):
    from open_x_cell.inference import predict_adata

    adata = make_synthetic_adata(n_cells=120, n_genes=tiny_config().num_genes,
                                 n_perturbations=4, seed=3)
    path = tmp_path / "data.h5ad"
    adata.write_h5ad(path)
    tcfg = TrainingConfig(batch_size=2, set_size=8, max_steps=4, learning_rate=1e-3, seed=0, device="cpu")
    ckpt = train_tiny(path, tmp_path / "out", training_config=tcfg)
    pert = next(p for p in adata.obs["perturbation"].unique() if p != "control")
    pred = predict_adata(ckpt, path, str(pert), tmp_path / "pred.h5ad",
                         n_cells=8, tta=True, tta_steps=4, tta_lr=1e-3)
    assert pred.n_obs == 8 and pred.obs["perturbation"].iloc[0] == str(pert)
