from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from anndata import AnnData, read_h5ad

from open_x_cell.diffusion import cumulative_reveal_steps
from open_x_cell.priors import PriorStore, make_zero_priors
from open_x_cell.training import load_checkpoint


def predict_adata(
    checkpoint_path: str | Path,
    adata_path: str | Path,
    perturbation: str,
    output_path: str | Path,
    seed: int = 0,
    prior_files: dict[str, str | Path | None] | None = None,
    n_cells: int = 64,
) -> AnnData:
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    model, config, _ = load_checkpoint(checkpoint_path)
    adata = read_h5ad(adata_path)
    matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    control_mask = adata.obs["is_control"].to_numpy(dtype=bool)
    control_rows = np.flatnonzero(control_mask)
    if len(control_rows) == 0:
        raise ValueError("prediction requires control cells")
    # Sample n_cells control cells per prediction set (paper default 64); wrap-pad
    # when fewer control cells are available so the set is always full.
    used_rows = control_rows[:n_cells]
    if len(used_rows) < n_cells:
        used_rows = used_rows[np.resize(np.arange(len(used_rows)), n_cells)]
    control = np.asarray(matrix[used_rows][:, : config.context_length], dtype=np.float32)
    x_current = torch.as_tensor(control[None, :, :], dtype=torch.float32)
    mask = torch.zeros_like(x_current, dtype=torch.bool)
    gene_ids = torch.arange(config.context_length).repeat(1, 1)
    gene_to_id = {name: idx for idx, name in enumerate(adata.var_names[: config.num_genes])}
    perturbation_id = torch.tensor([gene_to_id.get(perturbation, 0)], dtype=torch.long)
    if prior_files:
        priors = PriorStore.from_files(prior_files, config).batch_for_perturbations((perturbation,), config)
    else:
        priors = make_zero_priors(1, config)
    steps = cumulative_reveal_steps(config.context_length, config.diffusion_fractions, seed=seed)
    with torch.no_grad():
        final = None
        for step_mask in steps:
            out = model(x_current, mask, gene_ids, perturbation_id, priors)
            final = out.expression.clamp_min(0.0)
            step = step_mask[None, None, :].expand_as(mask)
            x_current = torch.where(step, final, x_current)
            mask = mask | step
    assert final is not None
    var = adata.var.iloc[: config.context_length].copy()
    obs = adata.obs.iloc[used_rows].copy()
    obs["perturbation"] = perturbation
    pred = AnnData(X=final.squeeze(0).cpu().numpy(), obs=obs, var=var)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pred.write_h5ad(target)
    return pred
