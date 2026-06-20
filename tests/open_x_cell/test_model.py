import torch

from open_x_cell.config import tiny_config
from open_x_cell.model import OpenXCellModel
from open_x_cell.priors import make_zero_priors


def test_tiny_model_forward_returns_expression_and_embeddings():
    cfg = tiny_config()
    model = OpenXCellModel(cfg)
    x = torch.rand(2, 4, cfg.context_length)
    mask = torch.zeros(2, 4, cfg.context_length, dtype=torch.bool)
    gene_ids = torch.arange(cfg.context_length).repeat(2, 1)
    perturbation_ids = torch.tensor([1, 2])
    priors = make_zero_priors(batch_size=2, config=cfg)

    out = model(x, mask, gene_ids, perturbation_ids, priors)

    assert out.expression.shape == (2, 4, cfg.context_length)
    assert out.cell_embedding.shape[:2] == (2, 4)
    assert out.prior_reconstructions["esm"].shape == (2, 5120)
    assert torch.isfinite(out.expression).all()


def test_tiny_model_supports_backward_pass():
    cfg = tiny_config()
    model = OpenXCellModel(cfg)
    x = torch.rand(1, 2, cfg.context_length)
    mask = torch.zeros(1, 2, cfg.context_length, dtype=torch.bool)
    gene_ids = torch.arange(cfg.context_length).repeat(1, 1)
    perturbation_ids = torch.tensor([1])

    out = model(x, mask, gene_ids, perturbation_ids, make_zero_priors(1, cfg))
    loss = out.expression.mean()
    loss.backward()

    assert any(param.grad is not None for param in model.parameters())
