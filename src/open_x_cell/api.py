"""Control-only AnnData prediction with a verified pretrained bundle."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import anndata as ad
import numpy as np

from open_x_cell.bundle import verify_bundle, sha256
from open_x_cell.data import boolean_obs_mask, normalize_cp10k_log1p
from open_x_cell.inference import predict_adata
from open_x_cell.priors import PRIOR_SOURCES


class OpenXCell:
    def __init__(self, directory: str | Path, *, device: str = "cpu"):
        self.directory = Path(directory).resolve()
        self.manifest = verify_bundle(self.directory)
        self.device = device
        self.genes = json.loads((self.directory / "genes.json").read_text())

    @classmethod
    def from_pretrained(cls, directory: str | Path, *, device: str = "cpu") -> OpenXCell:
        """Validate a local downloaded bundle; weights load when predict is called."""
        return cls(directory, device=device)

    def predict(self, control: str | Path, perturbation: str, output: str | Path, *,
                input_space: str, layer: str | None = None, n_cells: int = 32,
                n_sets: int = 1, seed: int = 0, precision: str = "float32",
                max_genes: int | None = None) -> ad.AnnData:
        source = Path(control).resolve()
        target = Path(output).resolve()
        if target.exists():
            raise FileExistsError(target)
        if target.is_relative_to(self.directory):
            raise ValueError("output must be outside the model bundle")
        if input_space not in {"counts", "log1p-cp10k"}:
            raise ValueError("input_space must be counts or log1p-cp10k")
        controls = ad.read_h5ad(source)
        if not controls.n_obs or not controls.n_vars:
            raise ValueError("control input must have nonempty cells and genes")
        if "is_control" in controls.obs and not boolean_obs_mask(controls, "is_control").all():
            raise ValueError("the released inference API requires a control-only AnnData")
        controls.obs["is_control"] = True
        if layer is not None:
            if layer not in controls.layers:
                raise ValueError(f"missing expression layer: {layer}")
            controls.X = controls.layers[layer].copy()
        if input_space == "counts":
            controls = normalize_cp10k_log1p(controls)
        matrix = controls.X.toarray() if hasattr(controls.X, "toarray") else np.asarray(controls.X)
        if not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("control expression must be finite and nonnegative")
        controls.X = matrix.astype(np.float32)
        # Normalize on every measured column before the existing alignment/subsampling path.
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".xcell-infer-", dir=target.parent) as tmp:
            prepared = Path(tmp) / "controls.h5ad"
            temporary_output = Path(tmp) / "predictions.h5ad"
            controls.write_h5ad(prepared)
            result = predict_adata(
                self.directory / "model.safetensors", prepared, perturbation, temporary_output,
                prior_files={key: self.directory / "priors" / f"{key}.npz" for key in PRIOR_SOURCES},
                n_cells=n_cells, n_sets=n_sets, seed=seed, device=self.device,
                precision=precision, max_genes=max_genes,
            )
            provenance = result.uns["open_x_cell_prediction"]
            provenance["source_adata"] = str(source)
            provenance["input_space"] = input_space
            provenance["input_layer"] = layer or "X"
            provenance["source_adata_sha256"] = sha256(source)
            provenance["bundle_manifest_sha256"] = sha256(self.directory / "manifest.json")
            inputs = json.loads(provenance["inference_inputs_json"])
            inputs["source_adata"]["path"] = "prepared-controls.h5ad"
            provenance["inference_inputs_json"] = json.dumps(inputs, sort_keys=True)
            result.write_h5ad(temporary_output)
            # A hard link publishes a complete file and refuses an output created concurrently.
            os.link(temporary_output, target)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict a genetic perturbation from control-only cells.")
    parser.add_argument("--model", required=True, help="Directory produced by open-x-cell-download.")
    parser.add_argument("--control", required=True)
    parser.add_argument("--perturbation", required=True, help="One gene symbol in the checkpoint vocabulary.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-space", choices=("counts", "log1p-cp10k"), required=True)
    parser.add_argument("--layer")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--n-cells", type=int, default=32)
    parser.add_argument("--n-sets", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-genes", type=int, help="Optional reduced context for a smoke test; changes predictions.")
    args = parser.parse_args(argv)
    model = OpenXCell.from_pretrained(args.model, device=args.device)
    result = model.predict(args.control, args.perturbation, args.output, input_space=args.input_space,
                           layer=args.layer, n_cells=args.n_cells, n_sets=args.n_sets, seed=args.seed,
                           precision=args.precision, max_genes=args.max_genes)
    print(f"Wrote {result.n_obs} cells x {result.n_vars} genes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
