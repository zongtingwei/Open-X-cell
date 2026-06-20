from open_x_cell.data import OpenXCellDataset
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
