"""Test-time adaptation (TTA): self-supervised MMD-only adaptation on the target
context's non-targeting controls before prediction.

The paper adapts a trained model to a new context using only its NTC (control) cells:
a diffusion forward pass over control sets is optimized with the MMD (energy distance)
term alone, with the cross-attention (prior) blocks frozen. This nudges the encoder /
transformer toward the target NTC distribution without any labels.
"""
from __future__ import annotations

import hashlib
import math
from contextlib import nullcontext
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch

from open_x_cell.losses import energy_distance
from open_x_cell.priors import make_zero_priors


def adapt_model_tta(
    model,
    config,
    control: np.ndarray,
    *,
    steps: int = 200,
    learning_rate: float = 1e-5,
    set_size: int = 64,
    seed: int = 0,
    device: str = "cpu",
    freeze_cross_attn: bool = True,
    gene_ids: np.ndarray | None = None,
    precision: str = "float32",
) -> list[float]:
    """Adapt ``model`` in place on ``control`` (NTC) cells; returns per-step MMD loss.

    Two control subsets are drawn each step: one is the unmasked model input and the
    other is the distributional MMD target. Only the MMD term is optimized. Following
    the paper, prior cross-attention is both frozen and skipped during adaptation.
    """
    control = np.asarray(control, dtype=np.float32)
    if control.ndim != 2:
        raise ValueError("control must have shape [cells, genes]")
    if not np.isfinite(control).all():
        raise ValueError("TTA controls must contain finite expression values")
    if control.shape[1] <= 0:
        raise ValueError("TTA controls must contain at least one gene")
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, Real)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive")
    if isinstance(set_size, bool) or not isinstance(set_size, Integral) or set_size <= 0:
        raise ValueError("set_size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(freeze_cross_attn, bool):
        raise ValueError("freeze_cross_attn must be a bool")
    control = control[:, : min(control.shape[1], config.context_length)]
    n = len(control)
    if n == 0:
        raise ValueError("TTA requires control cells")
    dev = torch.device(device)
    if precision not in {"float32", "bfloat16", "float16"}:
        raise ValueError("precision must be float32, bfloat16, or float16")
    autocast_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    if dev.type == "cpu" and precision == "float16":
        raise ValueError("float16 autocast is not supported on CPU; use bfloat16 or float32")
    if gene_ids is None:
        gene_ids_array = np.arange(control.shape[1], dtype=np.int64)
    else:
        gene_ids_array = np.asarray(gene_ids)
        if gene_ids_array.shape != (control.shape[1],):
            raise ValueError("gene_ids must have one entry per control matrix column")
        if np.issubdtype(gene_ids_array.dtype, np.bool_) or not np.issubdtype(
            gene_ids_array.dtype, np.integer
        ):
            raise ValueError("gene_ids must contain integers")
        gene_ids_array = gene_ids_array.astype(np.int64, copy=False)
    if (
        np.any(gene_ids_array < 0)
        or np.any(gene_ids_array >= config.num_genes)
        or len(np.unique(gene_ids_array)) != len(gene_ids_array)
    ):
        raise ValueError("gene_ids must be unique indices in the model gene vocabulary")

    history: list[float] = []
    was_training = model.training
    original_requires_grad = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    try:
        model.to(dev)
        if freeze_cross_attn:
            for parameter in model.cross_blocks.parameters():
                parameter.requires_grad_(False)
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not params:
            raise ValueError("TTA has no trainable parameters after applying the freeze policy")
        # Appendix A.5 specifies zero weight decay and a 10% warmup followed by
        # cosine decay for test-time adaptation.
        optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0)
        from open_x_cell.training import make_warmup_cosine_scheduler

        scheduler = make_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=int(0.1 * steps),
            total_steps=steps,
        )
        gene_ids_tensor = torch.as_tensor(gene_ids_array[None, :], device=dev)
        pid = torch.zeros(1, dtype=torch.long, device=dev)
        rng = np.random.default_rng(seed)
        priors = make_zero_priors(1, config, device=dev)
        model.train()
        for step in range(steps):
            a = rng.choice(n, size=set_size, replace=n < set_size)
            b = rng.choice(n, size=set_size, replace=n < set_size)
            xc = torch.as_tensor(control[a][None, :, :], device=dev)
            xt = torch.as_tensor(control[b][None, :, :], device=dev)
            reveal = torch.zeros_like(xc, dtype=torch.bool)
            amp_context = (
                nullcontext()
                if precision == "float32"
                else torch.autocast(device_type=dev.type, dtype=autocast_dtype)
            )
            with amp_context:
                out = model(
                    xc,
                    reveal,
                    gene_ids_tensor,
                    pid,
                    priors,
                    skip_cross_attention=True,
                    reconstruct_priors=False,
                )
            # Distributional losses are accumulated in fp32 even for mixed-precision
            # model execution; pairwise distances are especially sensitive in fp16.
            loss = energy_distance(out.expression.float(), xt.float())
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss during TTA")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            history.append(float(loss.detach().cpu()))
    finally:
        for parameter, requires_grad in original_requires_grad:
            parameter.requires_grad_(requires_grad)
        model.train(was_training)
    return history


def adapt_checkpoint_tta(
    checkpoint_path: str | Path,
    adata_path: str | Path,
    output_path: str | Path,
    context: str | None = None,
    batch: str | None = None,
    steps: int = 200,
    learning_rate: float = 1e-5,
    set_size: int = 64,
    seed: int = 0,
    device: str = "cpu",
    freeze_cross_attn: bool = True,
    precision: str = "float32",
) -> tuple[Path, list[float]]:
    """Load a checkpoint, TTA-adapt on the target context's NTC, save an adapted checkpoint."""
    from anndata import read_h5ad

    from open_x_cell.data import boolean_obs_mask, dense_matrix
    from open_x_cell.inference import _aligned_gene_indices, _select_control_pool
    from open_x_cell.training import load_checkpoint, save_checkpoint

    checkpoint_path = Path(checkpoint_path).resolve()
    adata_path = Path(adata_path).resolve()
    out = Path(output_path).resolve()
    temporary = out.with_suffix(out.suffix + ".tmp")
    if {out, temporary} & {checkpoint_path, adata_path}:
        raise ValueError("TTA output and temporary paths must not overwrite an input")
    for label, value in (("context", context), ("batch", batch)):
        if value is not None and (
            not isinstance(value, str) or not value.strip() or value != value.strip()
        ):
            raise ValueError(f"{label} must be a non-empty canonical string when provided")

    model, config, var_names = load_checkpoint(checkpoint_path, map_location=device)
    adata = read_h5ad(adata_path)
    matrix = dense_matrix(adata)
    control_rows = np.flatnonzero(boolean_obs_mask(adata, "is_control"))
    pool = _select_control_pool(adata, control_rows, context, batch, min_size=set_size)
    input_ids, checkpoint_ids, _, _ = _aligned_gene_indices(
        adata, var_names, config, perturbation=None, max_genes=config.context_length
    )
    history = adapt_model_tta(model, config, matrix[pool][:, input_ids], steps=steps,
                              learning_rate=learning_rate, set_size=set_size,
                              seed=seed, device=device, freeze_cross_attn=freeze_cross_attn,
                              gene_ids=checkpoint_ids, precision=precision)
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    save_checkpoint(
        out,
        model,
        config,
        var_names,
        training_state={
            "tta": {
                "source_checkpoint": str(checkpoint_path),
                "source_checkpoint_sha256": digest.hexdigest(),
                "adata": str(adata_path),
                "context": context,
                "batch": batch,
                "steps": int(steps),
                "learning_rate": float(learning_rate),
                "set_size": int(set_size),
                "seed": int(seed),
                "freeze_cross_attn": bool(freeze_cross_attn),
                "precision": precision,
                "loss_history": history,
            }
        },
    )
    return out, history
