import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from open_x_cell.data import OpenXCellDataset, normalize_cp10k_log1p
from open_x_cell.synthetic import make_synthetic_adata


def test_make_synthetic_adata_has_required_obs_and_var():
    adata = make_synthetic_adata(n_cells=64, n_genes=24, n_perturbations=3, seed=1)

    assert adata.shape == (64, 24)
    assert {"perturbation", "context", "batch", "is_control"}.issubset(adata.obs.columns)
    assert adata.var_names[0] == "GENE_0"


def test_dataset_returns_matched_control_target_set():
    adata = make_synthetic_adata(n_cells=96, n_genes=24, n_perturbations=4, seed=2)

    ds = OpenXCellDataset(adata, set_size=4, context_length=12, seed=3)
    sample = ds[0]

    assert sample.x_control.shape == (4, 12)
    assert sample.x_target.shape == (4, 12)
    assert sample.gene_ids.shape == (12,)
    assert sample.perturbation_id >= 0


def test_cp10k_log1p_normalization_matches_paper_preprocessing():
    adata = AnnData(X=np.array([[1.0, 3.0], [2.0, 2.0]], dtype=np.float32))

    normalized = normalize_cp10k_log1p(adata)

    expected = np.log1p(np.array([[2500.0, 7500.0], [5000.0, 5000.0]], dtype=np.float32))
    np.testing.assert_allclose(normalized.X, expected, rtol=1e-6)
    assert normalized.uns["open_x_cell"]["normalization"] == "cp10k_log1p"


def test_dataset_samples_are_stratified_by_context_perturbation_and_batch():
    obs = pd.DataFrame(
        {
            "perturbation": ["control", "control", "control", "GENE_1", "GENE_1", "GENE_2"],
            "context": ["ctxA", "ctxA", "ctxB", "ctxA", "ctxA", "ctxB"],
            "batch": ["batch0", "batch1", "batch0", "batch1", "batch1", "batch0"],
            "is_control": [True, True, True, False, False, False],
        },
        index=[f"cell_{idx}" for idx in range(6)],
    )
    var = pd.DataFrame(index=["GENE_0", "GENE_1", "GENE_2", "GENE_3"])
    adata = AnnData(X=np.arange(24, dtype=np.float32).reshape(6, 4), obs=obs, var=var)

    ds = OpenXCellDataset(adata, set_size=2, context_length=4, seed=0)
    sample = next(s for s in (ds[idx] for idx in range(len(ds))) if s.perturbation == "GENE_1")

    assert sample.context == "ctxA"
    assert sample.batch == "batch1"
    assert set(adata.obs.iloc[sample.target_indices]["batch"]) == {"batch1"}
    assert set(adata.obs.iloc[sample.control_indices]["batch"]) == {"batch1"}


def test_dataset_falls_back_to_context_controls_when_batch_controls_are_missing():
    obs = pd.DataFrame(
        {
            "perturbation": ["control", "control", "GENE_1", "GENE_1"],
            "context": ["ctxA", "ctxB", "ctxA", "ctxA"],
            "batch": ["batch0", "batch0", "batch1", "batch1"],
            "is_control": [True, True, False, False],
        },
        index=[f"cell_{idx}" for idx in range(4)],
    )
    var = pd.DataFrame(index=["GENE_0", "GENE_1", "GENE_2"])
    adata = AnnData(X=np.arange(12, dtype=np.float32).reshape(4, 3), obs=obs, var=var)

    ds = OpenXCellDataset(adata, set_size=2, context_length=3, seed=0)
    sample = ds[0]

    assert sample.context == "ctxA"
    assert sample.batch == "batch1"
    assert set(adata.obs.iloc[sample.control_indices]["context"]) == {"ctxA"}
    assert set(adata.obs.iloc[sample.control_indices]["batch"]) == {"batch0"}


def test_deterministic_set_construction_chunks_and_discards_incomplete():
    # GENE_1 has 5 target cells, set_size=2 -> 2 complete sets (1 discarded)
    obs = pd.DataFrame(
        {
            "perturbation": ["control", "control", "control"] + ["GENE_1"] * 5,
            "context": ["c"] * 8,
            "batch": ["b"] * 8,
            "is_control": [True, True, True, False, False, False, False, False],
        },
        index=[f"cell_{i}" for i in range(8)],
    )
    var = pd.DataFrame(index=["GENE_0", "GENE_1", "GENE_2"])
    adata = AnnData(X=np.arange(24, dtype=np.float32).reshape(8, 3), obs=obs, var=var)
    ds = OpenXCellDataset(adata, set_size=2, context_length=3, seed=0)
    assert len(ds) == 2  # 5 // 2 = 2 complete sets, remainder discarded
    for i in range(len(ds)):
        s = ds[i]
        assert s.x_target.shape == (2, 3) and s.x_control.shape == (2, 3)
    # target cells across the two sets are disjoint (chunked, not resampled)
    used = set(ds[0].target_indices.tolist()) | set(ds[1].target_indices.tolist())
    assert len(used) == 4  # 2 sets * 2 cells, all distinct


def test_set_construction_is_deterministic_across_instances():
    adata = make_synthetic_adata(n_cells=96, n_genes=24, n_perturbations=4, seed=2)
    a = OpenXCellDataset(adata, set_size=4, context_length=12, seed=7)
    b = OpenXCellDataset(adata, set_size=4, context_length=12, seed=7)
    assert len(a) == len(b)
    s0, s1 = a[0], b[0]
    np.testing.assert_array_equal(s0.target_indices, s1.target_indices)
    np.testing.assert_array_equal(s0.control_indices, s1.control_indices)
    assert torch.equal(s0.x_target, s1.x_target)


def test_control_set_replicates_when_pool_smaller_than_set_size():
    # one control cell, set_size 4 -> replicated to fill
    obs = pd.DataFrame(
        {
            "perturbation": ["control"] + ["GENE_1"] * 4,
            "context": ["c"] * 5,
            "batch": ["b"] * 5,
            "is_control": [True, False, False, False, False],
        },
        index=[f"cell_{i}" for i in range(5)],
    )
    var = pd.DataFrame(index=["GENE_0", "GENE_1"])
    adata = AnnData(X=np.arange(10, dtype=np.float32).reshape(5, 2), obs=obs, var=var)
    ds = OpenXCellDataset(adata, set_size=4, context_length=2, seed=0)
    s = ds[0]
    assert s.x_control.shape == (4, 2)
    assert set(s.control_indices.tolist()) == {0}  # all replicated from the single control
