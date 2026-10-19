"""Sample a small control/STAT1 example from the shared QC-complete Orion release."""
import argparse
import json
from pathlib import Path

import anndata as ad
from anndata.io import read_elem, sparse_dataset
import h5py
import numpy as np

from open_x_cell.bundle import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="QC-complete, log1p(CP10K) HEK293T H5AD.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="STAT1")
    parser.add_argument("--cells", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.cells <= 0 or args.seed < 0:
        raise ValueError("cells must be positive and seed nonnegative")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    # Read only the required H5AD elements: a newer writer's unrelated .uns
    # encodings need not be supported to preserve the expression and annotations.
    with h5py.File(args.source, "r") as source:
        normalization = source["uns/xatlas_processing/normalization"].asstr()[()]
        if normalization != "log1p(CP10K)":
            raise ValueError("requires the QC-complete source with recorded log1p(CP10K) normalization")
        obs = read_elem(source["obs"])
        var = read_elem(source["var"])
        matrix = sparse_dataset(source["X"]) if isinstance(source["X"], h5py.Group) else source["X"]
        labels = obs["gene_target"].astype(str).to_numpy()
        rng = np.random.default_rng(args.seed)
        selected = {}
        for label in ("Non-Targeting", args.target):
            candidates = np.flatnonzero(labels == label)
            if len(candidates) < args.cells:
                raise ValueError(f"not enough cells for {label}")
            selected[label] = np.sort(rng.choice(candidates, args.cells, replace=False))
        output.mkdir(parents=True)
        for label, filename in (("Non-Targeting", "controls.h5ad"), (args.target, "observed.h5ad")):
            subset = ad.AnnData(X=matrix[selected[label], :],
                                obs=obs.iloc[selected[label]][["gene_target", "sample"]].copy(),
                                var=var.copy())
            subset.obs["perturbation"] = "control" if label == "Non-Targeting" else label
            subset.obs["is_control"] = label == "Non-Targeting"
            subset.obs["context"] = "HEK293T"
            subset.obs["batch"] = subset.obs["sample"].astype(str)
            subset.uns = {"example": {"input_space": "log1p-cp10k", "seed": args.seed}}
            subset.write_h5ad(output / filename, compression="gzip")
        provenance = {
            "dataset": "X-Atlas-Orion HEK293T, shared QC-complete release",
            "source_url": "https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion",
            "source_release_created_at": source["uns/xatlas_processing/created_at_utc"].asstr()[()],
            "source_shape": list(matrix.shape),
            "input_space": "log1p-cp10k",
            "normalization": "Preserved source values; normalized over the shared release gene axis before sampling.",
            "gene_identifiers": "gene symbols in var_names; all source columns retained",
            "gene_count": len(var),
            "seed": args.seed,
            "cells_per_group": args.cells,
            "target": args.target,
            "selected_source_rows": {k: v.tolist() for k, v in selected.items()},
            "file_sha256": {n: sha256(output / n) for n in ("controls.h5ad", "observed.h5ad")},
            "usage": "Workflow example only. The model trained on Orion HEK293T; this is not an independent test set.",
        }
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(json.dumps({"shape_per_group": [args.cells, len(var)], "output": str(output)}))


if __name__ == "__main__":
    main()
