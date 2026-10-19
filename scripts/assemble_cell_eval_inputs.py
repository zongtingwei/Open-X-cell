"""Assemble sharded X-Cell predictions into exact cell-eval input AnnData files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np

from open_x_cell.data import boolean_obs_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--perturbations", required=True)
    parser.add_argument("--pred-output", required=True)
    parser.add_argument("--real-output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--n-controls", type=int, default=32)
    parser.add_argument(
        "--evaluation-gene-reference",
        default=None,
        help=(
            "Optional AnnData whose ordered genes define the cell-eval feature subset; "
            "prediction provenance and truth labels still come from --truth."
        ),
    )
    return parser.parse_args()


def _atomic_write(adata: ad.AnnData, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.h5ad")
    adata.write_h5ad(temporary, compression="lzf")
    temporary.replace(target)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main() -> int:
    args = parse_args()
    if args.n_controls <= 0:
        raise ValueError("n_controls must be positive")
    pred_output = Path(args.pred_output).resolve()
    real_output = Path(args.real_output).resolve()
    metadata_output = Path(args.metadata_output).resolve()
    if len({pred_output, real_output, metadata_output}) != 3:
        raise ValueError("pred, real, and metadata outputs must be distinct paths")
    truth_path = Path(args.truth).resolve()
    perturbations_path = Path(args.perturbations).resolve()
    evaluation_reference_path = (
        Path(args.evaluation_gene_reference).resolve()
        if args.evaluation_gene_reference is not None
        else None
    )
    paths = sorted(Path(args.predictions).glob("[0-9][0-9][0-9][0-9]_*.h5ad"))
    protected_inputs = {truth_path, perturbations_path, *(path.resolve() for path in paths)}
    if evaluation_reference_path is not None:
        protected_inputs.add(evaluation_reference_path)
    output_pairs = (
        (pred_output, pred_output.with_suffix(".tmp.h5ad")),
        (real_output, real_output.with_suffix(".tmp.h5ad")),
        (metadata_output, metadata_output.with_suffix(metadata_output.suffix + ".tmp")),
    )
    all_output_paths = [path for pair in output_pairs for path in pair]
    if len(all_output_paths) != len(set(all_output_paths)):
        raise ValueError("cell-eval output and temporary paths must all be distinct")
    for target, temporary in output_pairs:
        if {target, temporary} & protected_inputs:
            raise ValueError("cell-eval outputs and temporary paths must not overwrite inputs")

    perturbation_payload = json.loads(perturbations_path.read_text(encoding="utf-8"))
    if not isinstance(perturbation_payload, list) or not perturbation_payload:
        raise ValueError("perturbations must be a non-empty JSON list")
    if any(
        not isinstance(value, str) or not value.strip() or value != value.strip()
        for value in perturbation_payload
    ):
        raise ValueError("perturbations must be non-empty canonical strings")
    expected = list(perturbation_payload)
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        raise ValueError("perturbation list contains duplicates")
    predictions: dict[str, ad.AnnData] = {}
    prediction_rows: dict[str, int] = {}
    prediction_hashes: dict[str, str] = {}
    perturbation_gene_token_presence: dict[str, bool] = {}
    diffusion_return_modes: set[str] = set()
    checkpoint_paths: set[str] = set()
    checkpoint_hashes: set[str] = set()
    inference_input_signatures: set[str] = set()
    for path in paths:
        item = ad.read_h5ad(path)
        if "perturbation" not in item.obs or item.obs["perturbation"].isna().any():
            raise ValueError(f"{path} must contain finite perturbation labels")
        labels = item.obs["perturbation"].astype(str).unique().tolist()
        if len(labels) != 1:
            raise ValueError(f"{path} must contain exactly one perturbation, found {labels}")
        perturbation = labels[0]
        file_index = int(path.name.split("_", maxsplit=1)[0])
        if file_index >= len(expected) or expected[file_index] != perturbation:
            raise ValueError(f"prediction filename/index does not match its label: {path.name}")
        if perturbation in predictions:
            raise ValueError(f"duplicate prediction for {perturbation}")
        provenance = dict(item.uns.get("open_x_cell_prediction", {}))
        required_provenance = {
            "source_adata": str(truth_path),
            "perturbations_file": str(perturbations_path),
            "perturbation": perturbation,
            "perturbation_index": file_index,
            "n_cells": args.n_controls,
        }
        mismatches = {
            key: {"expected": value, "actual": provenance.get(key)}
            for key, value in required_provenance.items()
            if provenance.get(key) != value
        }
        if mismatches:
            raise ValueError(f"prediction {path.name} provenance mismatch: {mismatches}")
        token_present = provenance.get("perturbation_gene_token_present")
        if not isinstance(token_present, (bool, np.bool_)):
            raise ValueError(
                f"prediction {path.name} lacks boolean target-token provenance"
            )
        if bool(token_present) != (perturbation in set(map(str, item.var_names))):
            raise ValueError(
                f"prediction {path.name} target-token provenance differs from its gene panel"
            )
        return_mode = provenance.get("diffusion_return_mode")
        if return_mode not in {"cumulative_scattered", "final_full_proposal"}:
            raise ValueError(f"prediction {path.name} lacks a valid diffusion return mode")
        diffusion_return_modes.add(str(return_mode))
        checkpoint = str(provenance.get("checkpoint", ""))
        checkpoint_hash = str(provenance.get("checkpoint_sha256", ""))
        if not checkpoint or not checkpoint_hash:
            raise ValueError(f"prediction {path.name} lacks checkpoint provenance")
        checkpoint_paths.add(checkpoint)
        checkpoint_hashes.add(checkpoint_hash)
        inference_inputs_json = provenance.get("inference_inputs_json")
        if not isinstance(inference_inputs_json, str):
            raise ValueError(f"prediction {path.name} lacks canonical inference input provenance")
        try:
            inference_inputs = json.loads(inference_inputs_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"prediction {path.name} contains invalid inference input provenance"
            ) from exc
        if inference_inputs_json != _canonical_json(inference_inputs):
            raise ValueError(f"prediction {path.name} inference input provenance is not canonical")
        inference_input_signatures.add(inference_inputs_json)
        predictions[perturbation] = item
        prediction_rows[perturbation] = int(item.n_obs)
        prediction_hashes[perturbation] = _sha256(path)
        perturbation_gene_token_presence[perturbation] = bool(token_present)
    missing = sorted(expected_set - set(predictions))
    extra = sorted(set(predictions) - expected_set)
    if missing or extra:
        raise ValueError(f"prediction coverage mismatch: missing={missing[:10]}, extra={extra[:10]}")
    if len(checkpoint_paths) != 1 or len(checkpoint_hashes) != 1:
        raise ValueError("predictions were not generated from one identical checkpoint")
    if len(diffusion_return_modes) != 1:
        raise ValueError("predictions mix multiple diffusion return interpretations")
    if len(inference_input_signatures) != 1:
        raise ValueError("predictions were not generated from one identical inference input set")
    recorded_checkpoint = Path(next(iter(checkpoint_paths))).resolve()
    recorded_checkpoint_hash = next(iter(checkpoint_hashes))
    if not recorded_checkpoint.is_file() or _sha256(recorded_checkpoint) != recorded_checkpoint_hash:
        raise ValueError("recorded prediction checkpoint is missing or its SHA-256 has changed")
    inference_inputs = json.loads(next(iter(inference_input_signatures)))
    expected_inference_records = {
        "checkpoint": (recorded_checkpoint, recorded_checkpoint_hash),
        "source_adata": (truth_path, _sha256(truth_path)),
        "perturbations_file": (perturbations_path, _sha256(perturbations_path)),
    }
    for label, (expected_path, expected_hash) in expected_inference_records.items():
        record = inference_inputs.get(label)
        if not isinstance(record, dict):
            raise ValueError(f"inference input provenance lacks {label}")
        actual_path = Path(str(record.get("path", ""))).resolve()
        if actual_path != expected_path or record.get("sha256") != expected_hash:
            raise ValueError(f"inference input provenance differs for {label}")
        if int(record.get("bytes", -1)) != expected_path.stat().st_size:
            raise ValueError(f"inference input byte size differs for {label}")
    prior_artifacts = inference_inputs.get("prior_artifacts")
    if not isinstance(prior_artifacts, dict) or set(prior_artifacts) != {
        "esm",
        "string",
        "genept",
        "depmap",
        "cell_painting",
    }:
        raise ValueError("inference input provenance has an invalid prior inventory")
    for source, record in prior_artifacts.items():
        if record is None:
            continue
        if not isinstance(record, dict):
            raise ValueError(f"inference {source} prior provenance is malformed")
        prior_path = Path(str(record.get("path", ""))).resolve()
        if (
            not prior_path.is_file()
            or int(record.get("bytes", -1)) != prior_path.stat().st_size
            or record.get("sha256") != _sha256(prior_path)
        ):
            raise ValueError(f"inference {source} prior provenance no longer matches its file")

    truth = ad.read_h5ad(truth_path)
    if truth.n_obs <= 0 or truth.n_vars <= 0 or not truth.var_names.is_unique:
        raise ValueError("truth AnnData must be non-empty and have unique genes")
    if "perturbation" not in truth.obs or truth.obs["perturbation"].isna().any():
        raise ValueError("truth AnnData must contain finite perturbation labels")
    if "benchmark_control" not in truth.obs:
        raise ValueError("truth AnnData must mark obs['benchmark_control']")
    if evaluation_reference_path is None:
        evaluation_genes = list(map(str, truth.var_names))
        evaluation_reference_record = None
    else:
        reference = ad.read_h5ad(evaluation_reference_path, backed="r")
        try:
            if reference.n_obs != truth.n_obs or not reference.obs_names.equals(truth.obs_names):
                raise ValueError("evaluation gene reference cells differ from --truth")
            if not reference.var_names.is_unique or reference.n_vars <= 0:
                raise ValueError("evaluation gene reference must have unique non-empty genes")
            evaluation_genes = list(map(str, reference.var_names))
        finally:
            reference.file.close()
        if not set(evaluation_genes).issubset(set(map(str, truth.var_names))):
            raise ValueError("evaluation genes are not a subset of --truth")
        evaluation_reference_record = {
            "path": str(evaluation_reference_path),
            "bytes": int(evaluation_reference_path.stat().st_size),
            "sha256": _sha256(evaluation_reference_path),
            "genes": len(evaluation_genes),
        }
    control_rows = np.flatnonzero(boolean_obs_mask(truth, "benchmark_control"))
    if len(control_rows) != args.n_controls:
        raise ValueError(
            f"expected exactly {args.n_controls} marked benchmark controls, found {len(control_rows)}"
        )
    is_control = boolean_obs_mask(truth, "is_control")
    if not is_control[control_rows].all():
        raise ValueError("obs['benchmark_control'] marks one or more non-control cells")
    control = truth[control_rows, evaluation_genes].copy()
    control.obs["perturbation"] = "control"
    control_names = truth.obs_names[control_rows].astype(str).tolist()
    truth_genes = set(map(str, truth.var_names))
    evaluation_gene_set = set(evaluation_genes)
    for perturbation in expected:
        item = predictions[perturbation]
        if item.n_obs != args.n_controls:
            raise ValueError(
                f"prediction {perturbation} has {item.n_obs} rows; expected {args.n_controls}"
            )
        if len(item.var_names) != len(set(map(str, item.var_names))):
            raise ValueError(f"prediction {perturbation} contains duplicate genes")
        prediction_genes = set(map(str, item.var_names))
        if not evaluation_gene_set.issubset(prediction_genes):
            raise ValueError(f"prediction {perturbation} omits evaluation genes")
        if evaluation_reference_path is None and prediction_genes != truth_genes:
            raise ValueError(f"prediction {perturbation} gene panel differs from truth")
        if "source_control_obs_name" not in item.obs:
            raise ValueError(f"prediction {perturbation} lacks control provenance")
        if item.obs["source_control_obs_name"].astype(str).tolist() != control_names:
            raise ValueError(f"prediction {perturbation} used a different fixed control set")
        provenance = dict(item.uns["open_x_cell_prediction"])
        if list(map(str, provenance.get("fixed_control_obs_names", []))) != control_names:
            raise ValueError(f"prediction {perturbation} metadata records a different control set")
        values = item.X.toarray() if hasattr(item.X, "toarray") else np.asarray(item.X)
        if not np.isfinite(values).all() or np.min(values) < 0:
            raise ValueError(f"prediction {perturbation} contains invalid expression values")
    ordered_predictions = [predictions[name][:, evaluation_genes].copy() for name in expected]
    combined_prediction = ad.concat(
        [control, *ordered_predictions],
        axis=0,
        join="inner",
        merge="same",
        index_unique="-cell_eval",
    )
    combined_prediction.obs["perturbation"] = combined_prediction.obs["perturbation"].astype(str)

    labels = truth.obs["perturbation"].astype(str).to_numpy()
    missing_truth = sorted(expected_set - set(labels))
    if missing_truth:
        raise ValueError(f"truth has no cells for perturbations: {missing_truth[:10]}")
    for perturbation in expected:
        if not np.any((labels == perturbation) & ~is_control):
            raise ValueError(f"truth has no non-control cells for perturbation {perturbation!r}")
    real_keep = boolean_obs_mask(truth, "benchmark_control") | np.isin(labels, expected)
    combined_real = truth[real_keep, evaluation_genes].copy()
    combined_real.obs["perturbation"] = combined_real.obs["perturbation"].astype(str)
    if set(combined_prediction.var_names) != set(combined_real.var_names):
        raise ValueError("prediction and real gene panels differ")
    combined_real = combined_real[:, combined_prediction.var_names].copy()
    real_values = (
        combined_real.X.toarray()
        if hasattr(combined_real.X, "toarray")
        else np.asarray(combined_real.X)
    )
    if not np.isfinite(real_values).all() or np.min(real_values) < 0:
        raise ValueError("selected real expression contains invalid values")

    _atomic_write(combined_prediction, pred_output)
    _atomic_write(combined_real, real_output)
    checkpoint_path = str(recorded_checkpoint)
    checkpoint_sha256 = recorded_checkpoint_hash
    metadata = {
        "truth": str(truth_path),
        "truth_sha256": _sha256(truth_path),
        "source_truth_shape": list(map(int, truth.shape)),
        "evaluation_gene_reference": evaluation_reference_record,
        "evaluation_genes": len(evaluation_genes),
        "prediction_dir": str(Path(args.predictions).resolve()),
        "perturbations_file": str(perturbations_path),
        "perturbations_sha256": _sha256(perturbations_path),
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "perturbations": len(expected),
        "controls": len(control_rows),
        "prediction_rows_per_perturbation": prediction_rows,
        "prediction_sha256": prediction_hashes,
        "perturbation_gene_token_presence": perturbation_gene_token_presence,
        "perturbations_with_gene_token": int(
            sum(perturbation_gene_token_presence.values())
        ),
        "perturbations_without_gene_token": int(
            len(expected) - sum(perturbation_gene_token_presence.values())
        ),
        "diffusion_return_mode": next(iter(diffusion_return_modes)),
        "pred_shape": list(map(int, combined_prediction.shape)),
        "real_shape": list(map(int, combined_real.shape)),
        "pred_sha256": _sha256(pred_output),
        "real_sha256": _sha256(real_output),
        "assembly_script_sha256": _sha256(Path(__file__).resolve()),
        "control_obs_names": control_names,
        "prediction_control_provenance_verified": True,
        "inference_inputs": inference_inputs,
        "cell_eval_control_label": "control",
        "cell_eval_perturbation_column": "perturbation",
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_temporary = metadata_output.with_suffix(metadata_output.suffix + ".tmp")
    metadata_temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    metadata_temporary.replace(metadata_output)
    print(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
