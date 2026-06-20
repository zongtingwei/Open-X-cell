import torch

from open_x_cell.losses import OpenXCellLoss, concordance_ccc, energy_distance


def test_energy_distance_is_zero_for_identical_sets():
    x = torch.randn(2, 4, 8)

    assert energy_distance(x, x).abs().item() < 1e-6


def test_ccc_is_one_for_identical_vectors():
    x = torch.randn(3, 10)

    value = concordance_ccc(x, x, dim=1)

    assert torch.allclose(value, torch.ones_like(value), atol=1e-5)


def test_open_x_cell_loss_returns_finite_weighted_terms():
    pred = torch.rand(2, 4, 8, requires_grad=True)
    target = torch.rand(2, 4, 8)
    ctrl = torch.rand(2, 4, 8)
    cell = torch.rand(2, 4, 16)
    loss_fn = OpenXCellLoss()

    result = loss_fn(prediction=pred, target=target, control=ctrl, cell_embedding=cell)

    assert torch.isfinite(result.total)
    assert {"mmd", "fold_change_ccc", "delta_mse"}.issubset(result.terms)
    result.total.backward()
    assert pred.grad is not None
