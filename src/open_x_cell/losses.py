from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from open_x_cell.config import LossConfig


@dataclass(frozen=True)
class LossOutput:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError("energy_distance expects [batch, set, features] tensors")
    if x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
        raise ValueError("x and y must share batch and feature dimensions")
    xy = torch.cdist(x, y, p=2).mean(dim=(1, 2))
    xx = torch.cdist(x, x, p=2).mean(dim=(1, 2))
    yy = torch.cdist(y, y, p=2).mean(dim=(1, 2))
    return (2.0 * xy - xx - yy).mean()


def concordance_ccc(x: torch.Tensor, y: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError("x and y must have matching shapes")
    mean_x = x.mean(dim=dim, keepdim=True)
    mean_y = y.mean(dim=dim, keepdim=True)
    centered_x = x - mean_x
    centered_y = y - mean_y
    var_x = centered_x.pow(2).mean(dim=dim)
    var_y = centered_y.pow(2).mean(dim=dim)
    cov = (centered_x * centered_y).mean(dim=dim)
    mean_delta = (mean_x.squeeze(dim) - mean_y.squeeze(dim)).pow(2)
    return (2.0 * cov) / (var_x + var_y + mean_delta + eps)


def fold_change(prediction: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    return prediction.mean(dim=1) - control.mean(dim=1)


def info_nce(cell_embedding: torch.Tensor, target_embedding: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    pred = F.normalize(cell_embedding.mean(dim=1), dim=-1)
    target = F.normalize(target_embedding.mean(dim=1), dim=-1)
    logits = pred @ target.T / temperature
    labels = torch.arange(pred.shape[0], device=pred.device)
    return F.cross_entropy(logits, labels)


class OpenXCellLoss(nn.Module):
    def __init__(self, weights: LossConfig | None = None) -> None:
        super().__init__()
        self.weights = weights or LossConfig()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        control: torch.Tensor,
        cell_embedding: torch.Tensor,
        target_cell_embedding: torch.Tensor | None = None,
        prior_reconstructions: dict[str, torch.Tensor] | None = None,
        prior_values: dict[str, torch.Tensor] | None = None,
        prior_missing: torch.Tensor | None = None,
        prior_source_names: tuple[str, ...] | None = None,
    ) -> LossOutput:
        terms: dict[str, torch.Tensor] = {}
        pred_delta = fold_change(prediction, control)
        true_delta = fold_change(target, control)

        terms["mmd"] = energy_distance(prediction, target)
        terms["fold_change_ccc"] = (1.0 - concordance_ccc(pred_delta, true_delta, dim=1)).mean()
        terms["per_gene_ccc"] = (1.0 - concordance_ccc(prediction, target, dim=1)).mean()
        ratio = pred_delta.norm(dim=1) / (true_delta.norm(dim=1) + 1e-8)
        terms["delta_norm"] = (1.0 - ratio).pow(2).mean()
        terms["delta_mse"] = F.mse_loss(pred_delta, true_delta)

        if target_cell_embedding is not None and self.weights.contrastive:
            terms["contrastive"] = info_nce(cell_embedding, target_cell_embedding)

        if prior_reconstructions is not None and prior_values is not None and self.weights.prior_reconstruction:
            prior_losses = []
            for source, reconstruction in prior_reconstructions.items():
                if source in prior_values:
                    if prior_missing is not None:
                        source_names = prior_source_names or tuple(prior_reconstructions)
                        if source not in source_names:
                            continue
                        keep = ~prior_missing[:, source_names.index(source)]
                        if not torch.any(keep):
                            continue
                        prior_losses.append(F.mse_loss(reconstruction[keep], prior_values[source][keep]))
                    else:
                        prior_losses.append(F.mse_loss(reconstruction, prior_values[source]))
            if prior_losses:
                terms["prior_reconstruction"] = torch.stack(prior_losses).mean()

        weighted_terms = {
            "mmd": self.weights.mmd * terms["mmd"],
            "fold_change_ccc": self.weights.fold_change_ccc * terms["fold_change_ccc"],
            "per_gene_ccc": self.weights.per_gene_ccc * terms["per_gene_ccc"],
            "delta_norm": self.weights.delta_norm * terms["delta_norm"],
            "delta_mse": self.weights.delta_mse * terms["delta_mse"],
        }
        if "contrastive" in terms:
            weighted_terms["contrastive"] = self.weights.contrastive * terms["contrastive"]
        if "prior_reconstruction" in terms:
            weighted_terms["prior_reconstruction"] = self.weights.prior_reconstruction * terms["prior_reconstruction"]

        total = torch.stack(list(weighted_terms.values())).sum()
        return LossOutput(total=total, terms=terms)
