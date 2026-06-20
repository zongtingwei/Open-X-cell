from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def make_synthetic_adata(
    n_cells: int = 512,
    n_genes: int = 128,
    n_perturbations: int = 8,
    seed: int = 0,
) -> AnnData:
    if n_cells <= 0 or n_genes <= 1:
        raise ValueError("n_cells must be positive and n_genes must be greater than 1")
    if n_perturbations <= 0 or n_perturbations >= n_genes:
        raise ValueError("n_perturbations must be positive and smaller than n_genes")

    rng = np.random.default_rng(seed)
    gene_names = np.array([f"GENE_{idx}" for idx in range(n_genes)])
    labels = ["control"] + [f"GENE_{idx + 1}" for idx in range(n_perturbations)]
    per_label = n_cells // len(labels)
    remainder = n_cells % len(labels)
    perturbations: list[str] = []
    for idx, label in enumerate(labels):
        perturbations.extend([label] * (per_label + (1 if idx < remainder else 0)))
    perturbations = perturbations[:n_cells]

    base = rng.lognormal(mean=1.0, sigma=0.35, size=(n_cells, n_genes)).astype(np.float32)
    effects = rng.normal(loc=0.0, scale=0.05, size=(n_perturbations, n_genes)).astype(np.float32)
    for pert_idx in range(n_perturbations):
        effects[pert_idx, pert_idx + 1] -= 0.8
        effects[pert_idx, (pert_idx + 2) % n_genes] += 0.5

    x = np.log1p(base)
    for row_idx, label in enumerate(perturbations):
        if label != "control":
            pert_idx = int(label.split("_")[1]) - 1
            x[row_idx] = np.clip(x[row_idx] + effects[pert_idx], a_min=0.0, a_max=None)

    obs = pd.DataFrame(
        {
            "perturbation": perturbations,
            "context": ["synthetic"] * n_cells,
            "batch": ["batch0"] * n_cells,
            "is_control": [label == "control" for label in perturbations],
        }
    )
    var = pd.DataFrame(index=gene_names)
    return AnnData(X=x, obs=obs, var=var)
