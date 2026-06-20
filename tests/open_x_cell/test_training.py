import pandas as pd
import torch
from torch.utils.data import DataLoader

from open_x_cell.config import tiny_config
from open_x_cell.data import OpenXCellDataset, collate_samples
from open_x_cell.priors import PriorStore
from open_x_cell.synthetic import make_synthetic_adata
from open_x_cell.training import compute_batch_loss, make_warmup_cosine_scheduler
from open_x_cell.model import OpenXCellModel
from open_x_cell.losses import OpenXCellLoss


def test_compute_batch_loss_uses_contrastive_and_prior_reconstruction(tmp_path):
    cfg = tiny_config()
    adata = make_synthetic_adata(n_cells=64, n_genes=32, n_perturbations=2, seed=5)
    ds = OpenXCellDataset(adata, set_size=4, context_length=cfg.context_length, seed=0)
    batch = next(iter(DataLoader(ds, batch_size=2, collate_fn=collate_samples)))
    esm_path = tmp_path / "esm.csv"
    pd.DataFrame(
        {
            "gene": ["GENE_1", "GENE_2"],
            **{f"f{idx}": [0.1, 0.2] for idx in range(cfg.prior_dims["esm"])},
        }
    ).to_csv(esm_path, index=False)
    priors = PriorStore.from_files({"esm": esm_path}, cfg)
    model = OpenXCellModel(cfg)

    output = compute_batch_loss(
        model=model,
        batch=batch,
        model_config=cfg,
        loss_fn=OpenXCellLoss(),
        prior_store=priors,
        step=0,
        seed=0,
        device=torch.device("cpu"),
    )

    assert "contrastive" in output.terms
    assert "prior_reconstruction" in output.terms
    assert torch.isfinite(output.total)


def test_warmup_cosine_scheduler_reaches_zero_after_training_horizon():
    param = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([param], lr=1.0)
    scheduler = make_warmup_cosine_scheduler(optimizer, warmup_steps=2, total_steps=6)

    lrs = []
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] > 0
    assert lrs[1] == 1.0
    assert lrs[-1] == 0.0
