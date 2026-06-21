import numpy as np

from open_x_cell.config import TrainingConfig, tiny_config
from open_x_cell.evaluation import evaluate_batch
from open_x_cell.inference import _select_control_pool, predict_adata
from open_x_cell.synthetic import make_synthetic_adata
from open_x_cell.training import train_tiny


class _FakeObs:
    def __init__(self, d):
        self._d = d

    def __contains__(self, k):
        return k in self._d

    def __getitem__(self, k):
        import pandas as pd
        return pd.Series(self._d[k])


class _FakeAdata:
    def __init__(self, d):
        self.obs = _FakeObs(d)


def test_select_control_pool_filters_by_context_and_batch():
    rows = np.array([0, 1, 2, 3])
    adata = _FakeAdata({
        "context": ["a", "a", "b", "b"],
        "batch": ["x", "y", "x", "y"],
    })
    # context only
    np.testing.assert_array_equal(_select_control_pool(adata, rows, "a", None), [0, 1])
    # context + batch
    np.testing.assert_array_equal(_select_control_pool(adata, rows, "a", "y"), [1])
    # empty (context,batch) -> falls back to context pool
    np.testing.assert_array_equal(_select_control_pool(adata, rows, "b", "zzz"), [2, 3])
    # unknown context -> all rows
    np.testing.assert_array_equal(_select_control_pool(adata, rows, "nope", None), rows)


def test_predict_ensemble_and_batch_eval(tmp_path):
    adata = make_synthetic_adata(n_cells=160, n_genes=tiny_config().num_genes,
                                 n_perturbations=4, seed=5)
    path = tmp_path / "data.h5ad"
    adata.write_h5ad(path)
    tcfg = TrainingConfig(batch_size=2, set_size=8, max_steps=4, learning_rate=1e-3, seed=0, device="cpu")
    ckpt = train_tiny(path, tmp_path / "out", training_config=tcfg)

    perts = [p for p in adata.obs["perturbation"].unique() if p != "control"][:2]
    pred_paths = []
    for p in perts:
        out = tmp_path / f"pred_{p}.h5ad"
        # ensemble of 3 control sets -> 3 * n_cells predicted rows
        pred = predict_adata(ckpt, path, str(p), out, n_cells=8, n_sets=3)
        assert pred.n_obs == 24
        pred_paths.append(out)

    result = evaluate_batch(path, pred_paths, tmp_path / "summary.json")
    assert "centroid_accuracy" in result["summary"]
    assert result["summary"]["n_perturbations"] == 2
    assert len(result["per_perturbation"]) == 2
    assert "de_spearman_lfc" in next(iter(result["per_perturbation"].values()))
