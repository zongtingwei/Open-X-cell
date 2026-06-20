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
        "overlap_at_n": float(len(pred_top & true_top) / top_n),
        "direction_match": float(same_direction.item()),
        "delta_norm_ratio": float((pred_delta.norm() / (true_delta.norm() + 1e-8)).item()),
    }


def evaluate_prediction_files(truth_path: str | Path, pred_path: str | Path, output_path: str | Path) -> dict[str, float]:
    truth = read_h5ad(truth_path)
    pred = read_h5ad(pred_path)
    metrics = evaluate_prediction_adata(truth, pred)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


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
