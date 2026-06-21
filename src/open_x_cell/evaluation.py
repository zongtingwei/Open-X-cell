from __future__ import annotations

import json
from pathlib import Path

import torch
from anndata import AnnData, read_h5ad


def _as_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.float() if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)


def _pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if denom.item() == 0:
        return torch.tensor(0.0)
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
    n = min(pred.shape[0], true.shape[0], ctrl.shape[0])
    pred = pred[:n]
    true = true[:n]
    ctrl = ctrl[:n]
    pred_delta = pred.mean(dim=0) - ctrl.mean(dim=0)
    true_delta = true.mean(dim=0) - ctrl.mean(dim=0)
    top_n = min(top_n, pred_delta.numel())
    pred_top = set(torch.topk(pred_delta.abs(), k=top_n).indices.tolist())
    true_top = set(torch.topk(true_delta.abs(), k=top_n).indices.tolist())
    same_direction = (torch.sign(pred_delta) == torch.sign(true_delta)).float().mean()
    return {
        "mae": float((pred - true).abs().mean().item()),
        "pearson_delta": float(_pearson(pred_delta, true_delta).clamp(-1, 1).item()),
        "de_spearman_lfc": float(_spearman(pred_delta, true_delta).clamp(-1, 1).item()),
        "overlap_at_n": float(len(pred_top & true_top) / top_n),
        "direction_match": float(same_direction.item()),
        "delta_norm_ratio": float((pred_delta.norm() / (true_delta.norm() + 1e-8)).item()),
    }


def centroid_accuracy(
    pred_centroids: dict[str, torch.Tensor],
    true_centroids: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Nearest-centroid classification accuracy across perturbations (cell-eval).

    Each predicted perturbation centroid is assigned to the nearest *true* centroid
    (Euclidean) among all perturbations; accuracy is the fraction assigned to their
    own perturbation. Requires >= 2 perturbations to be meaningful.
    """
    perts = [p for p in pred_centroids if p in true_centroids]
    if len(perts) < 2:
        return {"centroid_accuracy": float("nan"), "n_perturbations": len(perts)}
    true_mat = torch.stack([_as_tensor(true_centroids[p]) for p in perts])
    correct = 0
    for i, p in enumerate(perts):
        pred = _as_tensor(pred_centroids[p])
        dists = (true_mat - pred[None, :]).pow(2).sum(dim=1)
        if int(dists.argmin().item()) == i:
            correct += 1
    return {"centroid_accuracy": correct / len(perts), "n_perturbations": len(perts)}


def evaluate_prediction_files(truth_path: str | Path, pred_path: str | Path, output_path: str | Path) -> dict[str, float]:
    truth = read_h5ad(truth_path)
    pred = read_h5ad(pred_path)
    metrics = evaluate_prediction_adata(truth, pred)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def _aligned_delta(truth: AnnData, pred: AnnData) -> tuple[str, torch.Tensor, torch.Tensor]:
    """Return (perturbation, pred_delta, true_delta) over genes shared by both."""
    perturbation = str(pred.obs["perturbation"].iloc[0]) if "perturbation" in pred.obs else "unknown"
    truth_var = {name: idx for idx, name in enumerate(truth.var_names)}
    pred_idx, truth_idx = [], []
    for pi, name in enumerate(pred.var_names):
        if name in truth_var:
            pred_idx.append(pi)
            truth_idx.append(truth_var[name])
    truth_x = truth.X.toarray() if hasattr(truth.X, "toarray") else truth.X
    pred_x = pred.X.toarray() if hasattr(pred.X, "toarray") else pred.X
    target_mask = truth.obs["perturbation"].astype(str).to_numpy() == perturbation
    control_mask = truth.obs["is_control"].to_numpy(dtype=bool)
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
    truth = read_h5ad(truth_path)
    per_pert: dict[str, dict[str, float]] = {}
    pred_centroids: dict[str, torch.Tensor] = {}
    true_centroids: dict[str, torch.Tensor] = {}
    for path in pred_paths:
        pred = read_h5ad(path)
        per_pert[str(pred.obs["perturbation"].iloc[0])] = evaluate_prediction_adata(truth, pred)
        name, pd_delta, td_delta = _aligned_delta(truth, pred)
        pred_centroids[name], true_centroids[name] = pd_delta, td_delta
    keys = next(iter(per_pert.values())).keys() if per_pert else []
    summary = {k: float(sum(m[k] for m in per_pert.values()) / len(per_pert)) for k in keys}
    summary.update(centroid_accuracy(pred_centroids, true_centroids))
    result = {"summary": summary, "per_perturbation": per_pert}
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def evaluate_prediction_adata(truth: AnnData, pred: AnnData) -> dict[str, float]:
    perturbation = str(pred.obs["perturbation"].iloc[0]) if "perturbation" in pred.obs else "unknown"
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
    control_mask = truth.obs["is_control"].to_numpy(dtype=bool)
    if not target_mask.any():
        raise ValueError(f"truth AnnData has no cells for perturbation {perturbation!r}")
    return perturbation_metrics(
        prediction=torch.as_tensor(pred_x[:, pred_indices], dtype=torch.float32),
        truth=torch.as_tensor(truth_x[target_mask][:, truth_indices], dtype=torch.float32),
        control=torch.as_tensor(truth_x[control_mask][:, truth_indices], dtype=torch.float32),
    )
