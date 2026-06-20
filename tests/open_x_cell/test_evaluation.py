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
