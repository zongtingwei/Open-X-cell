from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from anndata import AnnData, read_h5ad

from open_x_cell.diffusion import cumulative_reveal_steps
from open_x_cell.priors import PriorStore, make_zero_priors
from open_x_cell.training import load_checkpoint


def _select_control_pool(
    adata: AnnData, control_rows: np.ndarray, context: str | None, batch: str | None
) -> np.ndarray:
    """Context-aware NTC pool: (context, batch) -> context -> all, falling back when empty."""
    pool = control_rows
    if context is not None and "context" in adata.obs:
        ctx = adata.obs["context"].astype(str).to_numpy()
        sel = pool[ctx[pool] == str(context)]
        if batch is not None and "batch" in adata.obs:
            bat = adata.obs["batch"].astype(str).to_numpy()
            sel_cb = sel[bat[sel] == str(batch)]
            if len(sel_cb):
                sel = sel_cb
        if len(sel):
            pool = sel
    return pool


def _diffuse(model, config, control: np.ndarray, perturbation_id, priors, seed: int) -> np.ndarray:
    x_current = torch.as_tensor(control[None, :, :], dtype=torch.float32)
    mask = torch.zeros_like(x_current, dtype=torch.bool)
    gene_ids = torch.arange(config.context_length).repeat(1, 1)
    steps = cumulative_reveal_steps(config.context_length, config.diffusion_fractions, seed=seed)
    with torch.no_grad():
        final = None
        for step_mask in steps:
            out = model(x_current, mask, gene_ids, perturbation_id, priors)
            final = out.expression.clamp_min(0.0)
            step = step_mask[None, None, :].expand_as(mask)
            x_current = torch.where(step, final, x_current)
            mask = mask | step
    return final.squeeze(0).cpu().numpy()


def predict_adata(
    checkpoint_path: str | Path,
    adata_path: str | Path,
    perturbation: str,
    output_path: str | Path,
    seed: int = 0,
    prior_files: dict[str, str | Path | None] | None = None,
    n_cells: int = 64,
    context: str | None = None,
    batch: str | None = None,
    n_sets: int = 1,
    tta: bool = False,
    tta_steps: int = 50,
    tta_lr: float = 1e-4,
) -> AnnData:
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    if n_sets <= 0:
        raise ValueError("n_sets must be positive")
    model, config, _ = load_checkpoint(checkpoint_path)
    adata = read_h5ad(adata_path)
    matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    control_rows = np.flatnonzero(adata.obs["is_control"].to_numpy(dtype=bool))
    if len(control_rows) == 0:
        raise ValueError("prediction requires control cells")
    pool = _select_control_pool(adata, control_rows, context, batch)

    if tta:
        from open_x_cell.tta import adapt_model_tta
        adapt_model_tta(model, config, np.asarray(matrix[pool], dtype=np.float32),
                        steps=tta_steps, learning_rate=tta_lr, set_size=n_cells, seed=seed)

    gene_to_id = {name: idx for idx, name in enumerate(adata.var_names[: config.num_genes])}
    perturbation_id = torch.tensor([gene_to_id.get(perturbation, 0)], dtype=torch.long)
    if prior_files:
        priors = PriorStore.from_files(prior_files, config).batch_for_perturbations((perturbation,), config)
    else:
        priors = make_zero_priors(1, config)

    rng = np.random.default_rng(seed)
    finals, all_rows = [], []
    for s in range(n_sets):
        if n_sets == 1:
            # deterministic first-n_cells set (wrap-pad) keeps single-set runs reproducible
            used = pool[:n_cells]
            if len(used) < n_cells:
                used = used[np.resize(np.arange(len(used)), n_cells)]
        else:
            used = rng.choice(pool, size=n_cells, replace=len(pool) < n_cells)
        control = np.asarray(matrix[used][:, : config.context_length], dtype=np.float32)
        finals.append(_diffuse(model, config, control, perturbation_id, priors, seed=seed + s))
        all_rows.append(used)

    final_x = np.concatenate(finals, axis=0)
    used_rows = np.concatenate(all_rows, axis=0)
    var = adata.var.iloc[: config.context_length].copy()
    obs = adata.obs.iloc[used_rows].copy()
    obs["perturbation"] = perturbation
    pred = AnnData(X=final_x, obs=obs, var=var)
    pred.obs_names_make_unique()  # ensemble may reuse control cells across sets
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pred.write_h5ad(target)
    return pred
