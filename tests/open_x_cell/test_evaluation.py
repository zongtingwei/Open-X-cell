import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from open_x_cell.evaluation import evaluate_batch, perturbation_metrics


def test_perturbation_metrics_returns_finite_values():
    control = torch.zeros(4, 8)
    truth = torch.ones(4, 8)
    pred = torch.ones(4, 8) * 0.9

    metrics = perturbation_metrics(prediction=pred, truth=truth, control=control, top_n=3)

    assert metrics["mae"] >= 0
    assert -1 <= metrics["pearson_delta"] <= 1
    assert 0 <= metrics["overlap_at_n"] <= 1


def test_perturbation_metrics_rejects_boolean_top_n():
    values = torch.ones(2, 3)

    with pytest.raises(ValueError, match="positive integer"):
        perturbation_metrics(values, values, values, top_n=True)


def test_mae_compares_unpaired_pseudobulk_centroids():
    truth = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    prediction = truth.flip(0)
    metrics = perturbation_metrics(prediction, truth, torch.zeros_like(truth), top_n=1)

    assert metrics["mae"] == 0.0


def test_metrics_use_all_cells_when_set_sizes_differ():
    prediction = torch.tensor([[0.0], [2.0]])
    truth = torch.tensor([[1.0]])
    control = torch.tensor([[0.0], [0.0], [100.0]])

    metrics = perturbation_metrics(prediction, truth, control, top_n=1)

    # Prediction and truth centroids match. Truncating all matrices to the
    # shortest row count would incorrectly compare 0 against 1 instead.
    assert metrics["mae"] == 0.0


def test_de_spearman_lfc_and_centroid_accuracy():
    import torch
    from open_x_cell.evaluation import perturbation_metrics, centroid_accuracy, _spearman

    # spearman is rank-based: monotone-but-nonlinear relation -> ~1.0
    x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    y = x ** 3
    assert _spearman(x, y).item() > 0.999

    ctrl = torch.zeros(8, 6)
    true = torch.tensor([[2.0, -1, 0, 3, -2, 1]]).repeat(8, 1)
    pred = true.clone()
    m = perturbation_metrics(pred, true, ctrl)
    assert "de_spearman_lfc" in m and m["de_spearman_lfc"] > 0.99

    # centroid accuracy: identical pred/true centroids -> perfect; well-separated
    preds = {"A": torch.tensor([1.0, 0.0]), "B": torch.tensor([0.0, 1.0])}
    truths = {"A": torch.tensor([1.0, 0.05]), "B": torch.tensor([0.05, 1.0])}
    acc = centroid_accuracy(preds, truths)
    assert acc["centroid_accuracy"] == 1.0 and acc["n_perturbations"] == 2

    # swapped predictions -> 0 accuracy
    swapped = {"A": torch.tensor([0.0, 1.0]), "B": torch.tensor([1.0, 0.0])}
    assert centroid_accuracy(swapped, truths)["centroid_accuracy"] == 0.0
    assert centroid_accuracy({"A": preds["A"]}, truths) == {
        "centroid_accuracy": None,
        "n_perturbations": 1,
    }


def test_batch_centroids_use_one_gene_order_across_prediction_files(tmp_path):
    truth = ad.AnnData(
        X=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        obs=pd.DataFrame(
            {
                "perturbation": ["control", "A", "B"],
                "is_control": [True, False, False],
            },
            index=["control", "a", "b"],
        ),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    pred_a = ad.AnnData(
        X=np.asarray([[1.0, 0.0]], dtype=np.float32),
        obs=pd.DataFrame({"perturbation": ["A"]}, index=["pred-a"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    # Same biological vector as truth B, but in the per-file order produced
    # when inference moves a different perturbation gene to the first position.
    pred_b = ad.AnnData(
        X=np.asarray([[1.0, 0.0]], dtype=np.float32),
        obs=pd.DataFrame({"perturbation": ["B"]}, index=["pred-b"]),
        var=pd.DataFrame(index=["g2", "g1"]),
    )
    truth_path = tmp_path / "truth.h5ad"
    pred_a_path = tmp_path / "a.h5ad"
    pred_b_path = tmp_path / "b.h5ad"
    truth.write_h5ad(truth_path)
    pred_a.write_h5ad(pred_a_path)
    pred_b.write_h5ad(pred_b_path)

    truth_bytes = truth_path.read_bytes()
    with pytest.raises(ValueError, match="must not overwrite an input"):
        evaluate_batch(truth_path, [pred_a_path, pred_b_path], truth_path)
    assert truth_path.read_bytes() == truth_bytes

    result = evaluate_batch(truth_path, [pred_a_path, pred_b_path])

    assert result["summary"]["centroid_accuracy"] == 1.0
