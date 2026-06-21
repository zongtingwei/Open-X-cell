import torch

from open_x_cell.evaluation import perturbation_metrics


def test_perturbation_metrics_returns_finite_values():
    control = torch.zeros(4, 8)
    truth = torch.ones(4, 8)
    pred = torch.ones(4, 8) * 0.9

    metrics = perturbation_metrics(prediction=pred, truth=truth, control=control, top_n=3)

    assert metrics["mae"] >= 0
    assert -1 <= metrics["pearson_delta"] <= 1
    assert 0 <= metrics["overlap_at_n"] <= 1


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
