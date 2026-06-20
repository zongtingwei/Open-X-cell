import torch

from open_x_cell.config import tiny_config
from open_x_cell.priors import PriorBatch, make_zero_priors


def test_make_zero_priors_has_paper_sources_and_missing_mask():
    cfg = tiny_config()

    priors = make_zero_priors(batch_size=2, config=cfg)

    assert priors.values["esm"].shape == (2, 5120)
    assert priors.values["cell_painting"].shape == (2, 259)
    assert priors.missing.shape == (2, 5)
    assert torch.all(priors.missing)


def test_prior_batch_to_moves_all_tensors():
    priors = PriorBatch(values={"esm": torch.zeros(1, 5120)}, missing=torch.ones(1, 1, dtype=torch.bool))

    moved = priors.to(torch.device("cpu"))

    assert moved.values["esm"].device.type == "cpu"
    assert moved.missing.device.type == "cpu"
