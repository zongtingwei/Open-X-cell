"""Test-time adaptation (TTA): self-supervised MMD-only adaptation on the target
context's non-targeting controls before prediction.

The paper adapts a trained model to a new context using only its NTC (control) cells:
a diffusion forward pass over control sets is optimized with the MMD (energy distance)
term alone, with the cross-attention (prior) blocks frozen. This nudges the encoder /
transformer toward the target NTC distribution without any labels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from open_x_cell.config import LossConfig
from open_x_cell.diffusion import make_reveal_mask, mix_control_target
from open_x_cell.losses import OpenXCellLoss
from open_x_cell.priors import make_zero_priors

_MMD_ONLY = LossConfig(mmd=1.0, fold_change_ccc=0.0, per_gene_ccc=0.0,
                       delta_norm=0.0, delta_mse=0.0, contrastive=0.0,
                       prior_reconstruction=0.0, l1_output=0.0)


def adapt_model_tta(
    model,
    config,
    control: np.ndarray,
    *,
    steps: int = 50,
    learning_rate: float = 1e-4,
    set_size: int = 64,
    seed: int = 0,
    device: str = "cpu",
    freeze_cross_attn: bool = True,
) -> list[float]:
    """Adapt ``model`` in place on ``control`` (NTC) cells; returns per-step MMD loss.

    Two disjoint control subsets are drawn each step (one revealed as the diffusion
    input, one as the MMD target). Only the MMD term is optimized; the cross-attention
    blocks are frozen by default.
    """
    control = np.asarray(control, dtype=np.float32)[:, : config.context_length]
    n = len(control)
    if n == 0:
        raise ValueError("TTA requires control cells")
    dev = torch.device(device)

    frozen = []
    if freeze_cross_attn:
        for p in model.cross_blocks.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                frozen.append(p)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate)
    loss_fn = OpenXCellLoss(_MMD_ONLY)
    gene_ids = torch.arange(config.context_length, device=dev).repeat(1, 1)
    pid = torch.zeros(1, dtype=torch.long, device=dev)
    fracs = config.diffusion_fractions
    rng = np.random.default_rng(seed)

    model.train()
    history: list[float] = []
    for step in range(steps):
        a = rng.choice(n, size=set_size, replace=n < set_size)
        b = rng.choice(n, size=set_size, replace=n < set_size)
        xc = torch.as_tensor(control[a][None, :, :], device=dev)
        xt = torch.as_tensor(control[b][None, :, :], device=dev)
        reveal = make_reveal_mask(1, set_size, config.context_length,
                                  fracs[step % len(fracs)], seed=seed + step, device=dev)
        xin = mix_control_target(xc, xt, reveal)
        priors = make_zero_priors(1, config, device=dev)
        out = model(xin, reveal, gene_ids, pid, priors)
        lo = loss_fn(out.expression, xt, xc, out.cell_embedding)
        if not torch.isfinite(lo.total):
            raise RuntimeError("non-finite loss during TTA")
        optimizer.zero_grad(set_to_none=True)
        lo.total.backward()
        optimizer.step()
        history.append(float(lo.total.detach().cpu()))

    for p in frozen:
        p.requires_grad_(True)
    model.eval()
    return history


def adapt_checkpoint_tta(
    checkpoint_path: str | Path,
    adata_path: str | Path,
    output_path: str | Path,
    context: str | None = None,
    batch: str | None = None,
    steps: int = 50,
    learning_rate: float = 1e-4,
    set_size: int = 64,
    seed: int = 0,
    device: str = "cpu",
    freeze_cross_attn: bool = True,
) -> tuple[Path, list[float]]:
    """Load a checkpoint, TTA-adapt on the target context's NTC, save an adapted checkpoint."""
    from anndata import read_h5ad

    from open_x_cell.data import dense_matrix
    from open_x_cell.inference import _select_control_pool
    from open_x_cell.training import load_checkpoint, save_checkpoint

    model, config, var_names = load_checkpoint(checkpoint_path, map_location=device)
    adata = read_h5ad(adata_path)
    matrix = dense_matrix(adata)
    control_rows = np.flatnonzero(adata.obs["is_control"].to_numpy(dtype=bool))
    pool = _select_control_pool(adata, control_rows, context, batch)
    history = adapt_model_tta(model, config, matrix[pool], steps=steps,
                              learning_rate=learning_rate, set_size=set_size,
                              seed=seed, device=device, freeze_cross_attn=freeze_cross_attn)
    out = Path(output_path)
    save_checkpoint(out, model, config, var_names)
    return out, history
