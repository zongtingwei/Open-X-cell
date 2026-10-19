import torch
import torch.distributed as dist
from torch.nn import functional as F

from open_x_cell.losses import OpenXCellLoss, concordance_ccc, energy_distance, info_nce


def test_energy_distance_is_zero_for_identical_sets():
    x = torch.randn(2, 4, 8)

    assert energy_distance(x, x).abs().item() < 1e-6


def test_energy_distance_matches_geomloss_half_statistic():
    x = torch.tensor([[[0.0]]])
    y = torch.tensor([[[2.0]]])

    # GeomLoss SamplesLoss("energy") = E|X-Y| - .5E|X-X'| - .5E|Y-Y'|.
    assert torch.allclose(energy_distance(x, y), torch.tensor(1.9999), atol=1e-7)


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


def test_prior_reconstruction_sums_source_mse_terms():
    prediction = torch.zeros(1, 2, 3)
    target = torch.zeros_like(prediction)
    control = torch.zeros_like(prediction)
    cell = torch.zeros(1, 2, 4)
    reconstructions = {
        "esm": torch.ones(1, 2),
        "string": torch.full((1, 2), 2.0),
    }
    priors = {name: torch.zeros_like(value) for name, value in reconstructions.items()}

    result = OpenXCellLoss()(
        prediction=prediction,
        target=target,
        control=control,
        cell_embedding=cell,
        prior_reconstructions=reconstructions,
        prior_values=priors,
    )

    assert torch.allclose(result.terms["prior_reconstruction"], torch.tensor(5.0))


def test_prior_reconstruction_uses_global_available_count_under_ddp(monkeypatch):
    """A source present on one of two ranks must not be halved by DDP averaging."""
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    # This logical rank has the sole available sample, so the all-reduced count
    # remains one. The local loss must be scaled by two before DDP averages it
    # with the other rank's zero contribution.
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op=None: None)

    reconstruction = torch.ones(1, 2, requires_grad=True)
    result = OpenXCellLoss()(
        prediction=torch.zeros(1, 2, 3),
        target=torch.zeros(1, 2, 3),
        control=torch.zeros(1, 2, 3),
        cell_embedding=torch.zeros(1, 2, 4),
        prior_reconstructions={"esm": reconstruction},
        prior_values={"esm": torch.zeros_like(reconstruction)},
        prior_missing=torch.tensor([[False]]),
        prior_source_names=("esm",),
    )

    assert torch.allclose(result.terms["prior_reconstruction"], torch.tensor(2.0))
    result.total.backward()
    assert torch.allclose(reconstruction.grad, torch.full_like(reconstruction, 2.0))


def test_globally_unavailable_prior_retains_a_zero_gradient_graph_edge():
    reconstruction = torch.ones(2, 3, requires_grad=True)

    result = OpenXCellLoss()(
        prediction=torch.zeros(2, 2, 3),
        target=torch.zeros(2, 2, 3),
        control=torch.zeros(2, 2, 3),
        cell_embedding=torch.zeros(2, 2, 4),
        prior_reconstructions={"esm": reconstruction},
        prior_values={"esm": torch.zeros_like(reconstruction)},
        prior_missing=torch.ones(2, 1, dtype=torch.bool),
        prior_source_names=("esm",),
    )

    assert result.terms["prior_reconstruction"].item() == 0.0
    result.total.backward()
    assert torch.equal(reconstruction.grad, torch.zeros_like(reconstruction))


def test_info_nce_rejects_boolean_temperature():
    embedding = torch.ones(2, 1, 3)

    try:
        info_nce(embedding, embedding, temperature=True)
    except ValueError as exc:
        assert "temperature" in str(exc)
    else:
        raise AssertionError("boolean temperature must be rejected")


def test_info_nce_treats_duplicate_perturbation_labels_as_positives():
    embedding = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]]])
    labels = torch.tensor([1, 1, 2])

    loss = info_nce(embedding, embedding, labels=labels)

    # Both label-1 target centers are positives, so duplicating the same
    # perturbation does not create a false negative.
    assert loss < 1e-3


def test_info_nce_optimizer_step_bank_matches_manual_global_objective():
    anchors = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    anchor_labels = torch.tensor([11, 22])
    target_bank = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    target_labels = torch.tensor([11, 22, 33])

    actual = info_nce(
        anchors,
        None,
        labels=anchor_labels,
        target_center_bank=target_bank,
        target_bank_labels=target_labels,
    )
    logits = F.normalize(anchors.mean(dim=1), dim=-1) @ F.normalize(
        target_bank, dim=-1
    ).T / 0.1
    expected = -(
        logits[torch.arange(2), torch.tensor([0, 1])]
        - torch.logsumexp(logits, dim=1)
    ).mean()

    assert torch.allclose(actual, expected)
    # The third perturbation is a real optimizer-batch negative and must affect
    # the objective compared with a microbatch-local two-center target set.
    local = info_nce(anchors, anchors, labels=anchor_labels)
    assert not torch.allclose(actual, local)


def test_info_nce_keeps_each_predicted_cell_as_an_anchor():
    anchors = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    target_bank = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target_labels = torch.tensor([1, 2])

    actual = info_nce(
        anchors,
        None,
        labels=torch.tensor([1]),
        target_center_bank=target_bank,
        target_bank_labels=target_labels,
    )
    cell_logits = F.normalize(anchors.reshape(2, 2), dim=-1) @ target_bank.T / 0.1
    expected = -(
        cell_logits[:, 0] - torch.logsumexp(cell_logits, dim=1)
    ).mean()

    assert torch.allclose(actual, expected)
    # Averaging the two opposing cell embeddings first would collapse the
    # anchor to zero and incorrectly yield log(2).
    assert not torch.allclose(actual, torch.log(torch.tensor(2.0)))


def test_info_nce_unlabeled_ddp_uses_rank_local_diagonal_offset(monkeypatch):
    """Retain the legacy unlabeled helper for distributed callers."""
    rank0_targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    rank1_targets = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)

    def fake_all_gather(outputs, local):
        assert torch.equal(local, rank1_targets)
        outputs[0].copy_(rank0_targets)
        outputs[1].copy_(rank1_targets)

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)
    anchors = rank1_targets[:, None, :]
    targets = rank1_targets[:, None, :]

    actual = info_nce(anchors, targets)
    logits = F.normalize(rank1_targets, dim=-1) @ F.normalize(
        torch.cat([rank0_targets, rank1_targets]), dim=-1
    ).T / 0.1
    expected = F.cross_entropy(logits, torch.tensor([2, 3]))

    assert torch.allclose(actual, expected)
