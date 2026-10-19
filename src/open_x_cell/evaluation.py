from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path

import torch
from anndata import AnnData, read_h5ad

from open_x_cell.data import boolean_obs_mask


def _write_json(path: str | Path, payload) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(target)


def _as_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.float() if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)


def _pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if denom.item() == 0:
        return x.new_zeros(())
    return (x @ y) / denom


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    """Average ranks (ties shared), like scipy.stats.rankdata, for Spearman."""
    order = torch.argsort(x)
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(1, x.numel() + 1, dtype=x.dtype, device=x.device)
    # average tied ranks so constant inputs don't bias the correlation
    uniq, inv, counts = torch.unique(x, return_inverse=True, return_counts=True)
    sums = torch.zeros(uniq.numel(), dtype=x.dtype, device=x.device).scatter_add_(0, inv, ranks)
    return (sums / counts)[inv]


def _spearman(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _pearson(_rankdata(x), _rankdata(y))


def perturbation_metrics(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    control: torch.Tensor,
    top_n: int = 20,
) -> dict[str, float]:
    pred = _as_tensor(prediction)
    true = _as_tensor(truth)
    ctrl = _as_tensor(control)
    if pred.dim() != 2 or true.dim() != 2 or ctrl.dim() != 2:
        raise ValueError("prediction, truth, and control must be [cells, genes] matrices")
    if not pred.shape[0] or not true.shape[0] or not ctrl.shape[0]:
        raise ValueError("prediction, truth, and control must each contain at least one cell")
    if pred.shape[1] != true.shape[1] or pred.shape[1] != ctrl.shape[1]:
        raise ValueError("prediction, truth, and control must have the same gene dimension")
    if not torch.isfinite(pred).all() or not torch.isfinite(true).all() or not torch.isfinite(ctrl).all():
        raise ValueError("prediction, truth, and control must contain finite values")
    if isinstance(top_n, bool) or not isinstance(top_n, Integral) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    pred_delta = pred.mean(dim=0) - ctrl.mean(dim=0)
    true_delta = true.mean(dim=0) - ctrl.mean(dim=0)
    top_n = min(top_n, pred_delta.numel())
    pred_top = set(torch.topk(pred_delta.abs(), k=top_n).indices.tolist())
    true_top = set(torch.topk(true_delta.abs(), k=top_n).indices.tolist())
    same_direction = (torch.sign(pred_delta) == torch.sign(true_delta)).float().mean()
    return {
        # cell-eval compares perturbation pseudobulk centroids; cell rows are
        # draws from distributions and have no meaningful one-to-one pairing.
        "mae": float((pred.mean(dim=0) - true.mean(dim=0)).abs().mean().item()),
        "pearson_delta": float(_pearson(pred_delta, true_delta).clamp(-1, 1).item()),
        "de_spearman_lfc": float(_spearman(pred_delta, true_delta).clamp(-1, 1).item()),
        "overlap_at_n": float(len(pred_top & true_top) / top_n),
        "direction_match": float(same_direction.item()),
        "delta_norm_ratio": float((pred_delta.norm() / (true_delta.norm() + 1e-8)).item()),
    }


def centroid_accuracy(
    pred_centroids: dict[str, torch.Tensor],
    true_centroids: dict[str, torch.Tensor],
) -> dict[str, float | int | None]:
    """Nearest-centroid classification accuracy across perturbations (cell-eval).

    Each predicted perturbation centroid is assigned to the nearest *true* centroid
    (Euclidean) among all perturbations; accuracy is the fraction assigned to their
    own perturbation. Requires >= 2 perturbations to be meaningful.
    """
    perts = [p for p in pred_centroids if p in true_centroids]
    if len(perts) < 2:
        # ``None`` serializes as standards-compliant JSON ``null``; emitting the
        # non-standard token NaN breaks strict parsers for otherwise valid
        # one-perturbation reports.
        return {"centroid_accuracy": None, "n_perturbations": len(perts)}
    true_mat = torch.stack([_as_tensor(true_centroids[p]) for p in perts])
    correct = 0
    for i, p in enumerate(perts):
        pred = _as_tensor(pred_centroids[p])
        dists = (true_mat - pred[None, :]).pow(2).sum(dim=1)
        if int(dists.argmin().item()) == i:
            correct += 1
    return {"centroid_accuracy": correct / len(perts), "n_perturbations": len(perts)}


def evaluate_prediction_files(truth_path: str | Path, pred_path: str | Path, output_path: str | Path) -> dict[str, float]:
    truth_path = Path(truth_path).resolve()
    pred_path = Path(pred_path).resolve()
    output_path = Path(output_path).resolve()
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if {output_path, temporary} & {truth_path, pred_path}:
        raise ValueError("evaluation output and temporary paths must not overwrite an input")
    truth = read_h5ad(truth_path)
    pred = read_h5ad(pred_path)
    metrics = evaluate_prediction_adata(truth, pred)
    _write_json(output_path, metrics)
    return metrics


def _aligned_delta(
    truth: AnnData,
    pred: AnnData,
    gene_order: list[str] | None = None,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    """Return (perturbation, pred_delta, true_delta) over genes shared by both."""
    perturbation = _prediction_perturbation(pred)
    _validate_evaluation_schema(truth, pred)
    truth_var = {str(name): idx for idx, name in enumerate(truth.var_names)}
    pred_var = {str(name): idx for idx, name in enumerate(pred.var_names)}
    if gene_order is None:
        gene_order = sorted(set(truth_var) & set(pred_var))
    if not gene_order or any(name not in truth_var or name not in pred_var for name in gene_order):
        raise ValueError("prediction and truth do not contain the requested shared gene panel")
    pred_idx = [pred_var[name] for name in gene_order]
    truth_idx = [truth_var[name] for name in gene_order]
    truth_x = truth.X.toarray() if hasattr(truth.X, "toarray") else truth.X
    pred_x = pred.X.toarray() if hasattr(pred.X, "toarray") else pred.X
    target_mask = truth.obs["perturbation"].astype(str).to_numpy() == perturbation
    control_mask = boolean_obs_mask(truth, "is_control")
    ctrl_mean = _as_tensor(truth_x[control_mask][:, truth_idx]).mean(0)
    pred_delta = _as_tensor(pred_x[:, pred_idx]).mean(0) - ctrl_mean
    true_delta = _as_tensor(truth_x[target_mask][:, truth_idx]).mean(0) - ctrl_mean
    return perturbation, pred_delta, true_delta


def evaluate_batch(truth_path: str | Path, pred_paths: list[str | Path],
                   output_path: str | Path | None = None) -> dict:
    """Evaluate many single-perturbation predictions against one truth AnnData.

    Returns per-perturbation metrics, their means, and cross-perturbation
    Centroid Accuracy (nearest true centroid by predicted delta).
    """
    if not pred_paths:
        raise ValueError("evaluate_batch requires at least one prediction file")
    truth_path = Path(truth_path).resolve()
    pred_paths = [Path(path).resolve() for path in pred_paths]
    if output_path is not None:
        output_path = Path(output_path).resolve()
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        if {output_path, temporary} & {truth_path, *pred_paths}:
            raise ValueError("evaluation output and temporary paths must not overwrite an input")
    truth = read_h5ad(truth_path)
    loaded_predictions: list[tuple[str, AnnData]] = []
    seen_perturbations: set[str] = set()
    common_genes = set(map(str, truth.var_names))
    for path in pred_paths:
        pred = read_h5ad(path)
        perturbation = _prediction_perturbation(pred)
        if perturbation in seen_perturbations:
            raise ValueError(f"duplicate prediction for perturbation {perturbation!r}")
        seen_perturbations.add(perturbation)
        common_genes &= set(map(str, pred.var_names))
        loaded_predictions.append((perturbation, pred))
    if not common_genes:
        raise ValueError("batch predictions and truth have no gene shared by every file")
    # Inference deliberately moves a measured perturbation gene to position 1
    # after CLS. Use one canonical intersection for every centroid so distances
    # never compare coordinates from different per-file gene orderings.
    common_gene_order = sorted(common_genes)

    per_pert: dict[str, dict[str, float]] = {}
    pred_centroids: dict[str, torch.Tensor] = {}
    true_centroids: dict[str, torch.Tensor] = {}
    for perturbation, pred in loaded_predictions:
        per_pert[perturbation] = evaluate_prediction_adata(truth, pred)
        name, pd_delta, td_delta = _aligned_delta(
            truth, pred, gene_order=common_gene_order
        )
        pred_centroids[name], true_centroids[name] = pd_delta, td_delta
    keys = next(iter(per_pert.values())).keys() if per_pert else []
    summary = {k: float(sum(m[k] for m in per_pert.values()) / len(per_pert)) for k in keys}
    summary.update(centroid_accuracy(pred_centroids, true_centroids))
    result = {"summary": summary, "per_perturbation": per_pert}
    if output_path is not None:
        _write_json(output_path, result)
    return result


def evaluate_prediction_adata(truth: AnnData, pred: AnnData) -> dict[str, float]:
    perturbation = _prediction_perturbation(pred)
    _validate_evaluation_schema(truth, pred)
    truth_var = {name: idx for idx, name in enumerate(truth.var_names)}
    pred_indices = []
    truth_indices = []
    for pred_idx, name in enumerate(pred.var_names):
        if name in truth_var:
            pred_indices.append(pred_idx)
            truth_indices.append(truth_var[name])
    if not pred_indices:
        raise ValueError("prediction and truth AnnData objects have no overlapping genes")

    truth_x = truth.X.toarray() if hasattr(truth.X, "toarray") else truth.X
    pred_x = pred.X.toarray() if hasattr(pred.X, "toarray") else pred.X
    target_mask = truth.obs["perturbation"].astype(str).to_numpy() == perturbation
    control_mask = boolean_obs_mask(truth, "is_control")
    if not target_mask.any():
        raise ValueError(f"truth AnnData has no cells for perturbation {perturbation!r}")
    if not control_mask.any():
        raise ValueError("truth AnnData has no control cells")
    return perturbation_metrics(
        prediction=torch.as_tensor(pred_x[:, pred_indices], dtype=torch.float32),
        truth=torch.as_tensor(truth_x[target_mask][:, truth_indices], dtype=torch.float32),
        control=torch.as_tensor(truth_x[control_mask][:, truth_indices], dtype=torch.float32),
    )


def _prediction_perturbation(pred: AnnData) -> str:
    if pred.n_obs == 0:
        raise ValueError("prediction AnnData contains no cells")
    if "perturbation" not in pred.obs:
        raise ValueError("prediction AnnData must contain obs['perturbation']")
    if pred.obs["perturbation"].isna().any():
        raise ValueError("prediction perturbation labels contain missing values")
    labels = pred.obs["perturbation"].astype(str).unique().tolist()
    if len(labels) != 1:
        raise ValueError(f"prediction AnnData must contain one perturbation, found {labels}")
    return labels[0]


def _validate_evaluation_schema(truth: AnnData, pred: AnnData) -> None:
    if truth.n_obs == 0 or truth.n_vars == 0 or pred.n_vars == 0:
        raise ValueError("truth and prediction AnnData objects must be non-empty")
    if "perturbation" not in truth.obs:
        raise ValueError("truth AnnData must contain obs['perturbation']")
    if truth.obs["perturbation"].isna().any():
        raise ValueError("truth perturbation labels contain missing values")
    if not truth.var_names.is_unique or not pred.var_names.is_unique:
        raise ValueError("truth and prediction gene names must be unique")
    boolean_obs_mask(truth, "is_control")
