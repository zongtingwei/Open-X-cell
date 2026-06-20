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


def test_open_x_cell_loss_can_include_all_seven_paper_terms():
    pred = torch.rand(2, 4, 8, requires_grad=True)
    target = torch.rand(2, 4, 8)
    ctrl = torch.rand(2, 4, 8)
    cell = torch.rand(2, 4, 16)
    target_cell = torch.rand(2, 4, 16)
    prior_reconstructions = {"esm": torch.rand(2, 3, requires_grad=True)}
    prior_values = {"esm": torch.rand(2, 3)}
    prior_missing = torch.zeros(2, 1, dtype=torch.bool)
    loss_fn = OpenXCellLoss()

    result = loss_fn(
        prediction=pred,
        target=target,
        control=ctrl,
        cell_embedding=cell,
        target_cell_embedding=target_cell,
        prior_reconstructions=prior_reconstructions,
        prior_values=prior_values,
        prior_missing=prior_missing,
        prior_source_names=("esm",),
    )

    assert set(result.terms) == {
        "mmd",
        "fold_change_ccc",
        "per_gene_ccc",
        "delta_norm",
        "delta_mse",
        "contrastive",
        "prior_reconstruction",
    }
    result.total.backward()
    assert pred.grad is not None
