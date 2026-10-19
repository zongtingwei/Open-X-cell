"""Build a compact HepG2 Nadig subset using the Open-X-Cell observation schema."""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def parse_args(
    argv: list[str] | None = None,
    *,
    default_output: str = "data/real/hepg2nadig_subset.h5ad",
    default_panel: int = 400,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Source HepG2 Nadig AnnData file.")
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--n-perturbations", type=int, default=12)
    parser.add_argument("--n-controls", type=int, default=1_500)
    parser.add_argument("--panel-size", type=int, default=default_panel)
    parser.add_argument("--cells-per-perturbation", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    default_output: str = "data/real/hepg2nadig_subset.h5ad",
    default_panel: int = 400,
) -> int:
    args = parse_args(
        argv,
        default_output=default_output,
        default_panel=default_panel,
    )
    positive = {
        "n_perturbations": args.n_perturbations,
        "n_controls": args.n_controls,
        "panel_size": args.panel_size,
        "cells_per_perturbation": args.cells_per_perturbation,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"subset sizes must be positive: {invalid}")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")

    source = Path(args.source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output = Path(args.output).resolve()
    temporary = output.with_suffix(".tmp.h5ad")
    if source in {output, temporary}:
        raise ValueError("output and temporary paths must not overwrite the source AnnData")

    rng = np.random.default_rng(args.seed)
    adata = ad.read_h5ad(source, backed="r")
    try:
        required_obs = {"transcript", "gene_id"}
        missing_obs = sorted(required_obs - set(adata.obs))
        if missing_obs:
            raise ValueError(f"source AnnData lacks obs columns: {missing_obs}")
        if "mean-1" not in adata.var:
            raise ValueError("source AnnData lacks var['mean-1']")
        if not adata.var_names.is_unique:
            raise ValueError("source AnnData has duplicate gene IDs")

        obs = adata.obs
        is_control = obs["transcript"].astype(str).eq("non-targeting").to_numpy()
        gene_id = obs["gene_id"].astype(str).to_numpy()
        var_ensembl = adata.var_names.astype(str).to_numpy()
        ensembl_to_column = {gene: index for index, gene in enumerate(var_ensembl)}

        perturbation_counts = pd.Series(gene_id[~is_control]).value_counts()
        candidates = [
            gene
            for gene in perturbation_counts.index.astype(str)
            if gene in ensembl_to_column
            and perturbation_counts[gene] >= args.cells_per_perturbation
        ][: args.n_perturbations]
        if len(candidates) != args.n_perturbations:
            raise ValueError(
                f"requested {args.n_perturbations} perturbations but only "
                f"{len(candidates)} satisfy the cell-count and gene-panel requirements"
            )

        mean_expression = pd.to_numeric(adata.var["mean-1"], errors="coerce").to_numpy()
        order = np.argsort(np.nan_to_num(mean_expression))[::-1]
        panel_columns = list(
            dict.fromkeys(
                [ensembl_to_column[gene] for gene in candidates]
                + [int(index) for index in order]
            )
        )[: args.panel_size]
        panel_columns.sort()
        if len(panel_columns) != args.panel_size:
            raise ValueError(
                f"requested a {args.panel_size}-gene panel but source has only "
                f"{len(panel_columns)} eligible genes"
            )
        panel_ensembl = var_ensembl[panel_columns]

        control_pool = np.flatnonzero(is_control)
        if len(control_pool) < args.n_controls:
            raise ValueError(
                f"requested {args.n_controls} controls but source contains {len(control_pool)}"
            )
        row_groups = [
            rng.choice(control_pool, size=args.n_controls, replace=False)
        ]
        labels = ["control"] * args.n_controls
        for gene in candidates:
            pool = np.flatnonzero((gene_id == gene) & ~is_control)
            row_groups.append(
                rng.choice(pool, size=args.cells_per_perturbation, replace=False)
            )
            labels.extend([gene] * args.cells_per_perturbation)
        rows = np.concatenate(row_groups)

        matrix = adata.X[rows][:, panel_columns]
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        matrix = np.asarray(matrix, dtype=np.float32)
    finally:
        adata.file.close()

    label_array = np.asarray(labels)
    subset = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {
                "perturbation": label_array,
                "context": "hepg2",
                "batch": "batch0",
                "is_control": label_array == "control",
            }
        ),
        var=pd.DataFrame(index=panel_ensembl),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    subset.write_h5ad(temporary)
    temporary.replace(output)
    print(
        f"wrote {output} {subset.shape} | controls {args.n_controls} | "
        f"perturbations {len(candidates)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
