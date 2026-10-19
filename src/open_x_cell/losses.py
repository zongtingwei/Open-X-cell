from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from open_x_cell.config import LossConfig


@dataclass(frozen=True)
class LossOutput:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=x.device.type, enabled=False):
        return _energy_distance_float(x.float(), y.float())


def _energy_distance_float(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # GeomLoss' SamplesLoss("energy") returns one half of the conventional
    # energy-distance statistic:
    #   E||X-Y|| - 1/2 E||X-X'|| - 1/2 E||Y-Y'||.
    # X-Cell specifies that exact loss with unit weight, so retaining the 1/2
    # factors matters relative to the other six equally weighted objectives.
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError("energy_distance expects [batch, set, features] tensors")
    if x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
        raise ValueError("x and y must share batch and feature dimensions")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")
    if x.shape[0] == 0 or x.shape[2] == 0 or x.shape[1] == 0 or y.shape[1] == 0:
        raise ValueError("energy_distance requires non-empty batches, sets, and features")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("energy_distance inputs must contain finite values")

    def geomloss_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # This is the tensorized distance routine used by GeomLoss 0.2.6.  Its
        # 1e-8 clamp (rather than torch.cdist's exact diagonal zero) contributes
        # a tiny constant and is retained so the clean-room loss matches
        # SamplesLoss(loss="energy", blur=.05, scaling=.5) numerically as well
        # as algebraically.
        a2 = a.square().sum(dim=-1, keepdim=True)
        b2 = b.square().sum(dim=-1).unsqueeze(1)
        squared = a2 - 2.0 * (a @ b.transpose(1, 2)) + b2
        return squared.clamp_min(1e-8).sqrt()

    xy = geomloss_distances(x, y).mean(dim=(1, 2))
    xx = geomloss_distances(x, x).mean(dim=(1, 2))
    yy = geomloss_distances(y, y).mean(dim=(1, 2))
    return (xy - 0.5 * xx - 0.5 * yy).mean()


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


def info_nce(
    cell_embedding: torch.Tensor,
    target_embedding: torch.Tensor | None,
    temperature: float = 0.1,
    labels: torch.Tensor | None = None,
    target_center_bank: torch.Tensor | None = None,
    target_bank_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    if cell_embedding.dim() != 3:
        raise ValueError("predicted InfoNCE embeddings must have shape [batch, set, features]")
    if cell_embedding.shape[0] == 0 or cell_embedding.shape[1] == 0:
        raise ValueError("InfoNCE embeddings must be non-empty")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, Real)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("InfoNCE temperature must be finite and positive")
    batch_size, set_size, embedding_dim = cell_embedding.shape
    using_bank = target_center_bank is not None
    if using_bank:
        if (
            target_center_bank.dim() != 2
            or target_center_bank.shape[0] == 0
            or target_center_bank.shape[1] != embedding_dim
            or target_center_bank.device != cell_embedding.device
            or not torch.isfinite(target_center_bank).all()
        ):
            raise ValueError("InfoNCE target bank must be finite [targets, features] on-anchor-device")
        if labels is None or target_bank_labels is None:
            raise ValueError("optimizer-step InfoNCE banks require anchor and target labels")
        if (
            target_bank_labels.shape != (target_center_bank.shape[0],)
            or target_bank_labels.dtype not in {torch.int32, torch.int64}
        ):
            raise ValueError("InfoNCE target-bank labels must be an integer vector")
        target = target_center_bank
    else:
        if target_embedding is None or target_embedding.dim() != 3:
            raise ValueError("InfoNCE target embeddings must have shape [batch, set, features]")
        if cell_embedding.shape != target_embedding.shape:
            raise ValueError("predicted and target InfoNCE embeddings must have matching shapes")
        target = target_embedding.mean(dim=1)
    label_offset = 0
    target_labels = None
    if labels is not None:
        if labels.shape != (batch_size,) or labels.dtype not in {torch.int32, torch.int64}:
            raise ValueError("InfoNCE labels must be an integer tensor with shape [batch]")
        labels = labels.to(device=cell_embedding.device)
        target_labels = labels
    if not using_bank and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        # Each rank keeps gradients for its local predictions and uses the target
        # embeddings from every rank as negatives. The target branch is stop-grad
        # by design, so the regular collective is sufficient and avoids gathering
        # the much larger token activations.
        gathered = [torch.empty_like(target) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, target.detach().contiguous())
        target = torch.cat(gathered, dim=0)
        # ``pred`` is constructed only after optional target gathering.  The
        # legacy unlabeled path has one anchor centroid per local batch item,
        # so its diagonal offset is determined directly by ``batch_size``.
        label_offset = dist.get_rank() * batch_size
        if labels is not None:
            gathered_labels = [torch.empty_like(labels) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, labels.contiguous())
            target_labels = torch.cat(gathered_labels, dim=0)
    elif using_bank:
        target_labels = target_bank_labels.to(device=cell_embedding.device)
    if labels is not None:
        assert target_labels is not None
        # A.2.6 applies InfoNCE to per-cell CLS-derived embeddings. Ground-truth
        # branches supply one center per perturbation; predicted cells remain
        # individual anchors rather than being averaged into a set centroid.
        if not using_bank:
            unique_labels, inverse = torch.unique(
                target_labels,
                sorted=True,
                return_inverse=True,
            )
            center_sums = torch.zeros(
                (unique_labels.shape[0], embedding_dim),
                dtype=target.dtype,
                device=target.device,
            )
            center_sums.index_add_(0, inverse, target)
            counts = torch.zeros(
                unique_labels.shape[0],
                dtype=target.dtype,
                device=target.device,
            )
            counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=target.dtype))
            target = center_sums / counts[:, None]
            target_labels = unique_labels
        pred = F.normalize(cell_embedding.reshape(batch_size * set_size, embedding_dim), dim=-1)
        target = F.normalize(target, dim=-1)
        anchor_labels = labels.repeat_interleave(set_size)
        logits = pred @ target.T / temperature
        positives = anchor_labels[:, None] == target_labels[None, :]
        if not positives.any(dim=1).all():
            raise RuntimeError("InfoNCE batch contains an anchor without a positive target")
        positive_logits = logits.masked_fill(~positives, -torch.inf)
        return -(torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)).mean()
    # Backwards-compatible unlabeled helper behavior: one set centroid and one
    # matching target centroid per batch item. Formal training always supplies
    # perturbation labels and therefore follows the per-cell branch above.
    pred = F.normalize(cell_embedding.mean(dim=1), dim=-1)
    target = F.normalize(target, dim=-1)
    logits = pred @ target.T / temperature
    diagonal = torch.arange(pred.shape[0], device=pred.device) + label_offset
    return F.cross_entropy(logits, diagonal)


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
        contrastive_labels: torch.Tensor | None = None,
        contrastive_target_bank: torch.Tensor | None = None,
        contrastive_target_bank_labels: torch.Tensor | None = None,
    ) -> LossOutput:
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            return self._forward_float(
                prediction,
                target,
                control,
                cell_embedding,
                target_cell_embedding,
                prior_reconstructions,
                prior_values,
                prior_missing,
                prior_source_names,
                contrastive_labels,
                contrastive_target_bank,
                contrastive_target_bank_labels,
            )

    def _forward_float(
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
        contrastive_labels: torch.Tensor | None = None,
        contrastive_target_bank: torch.Tensor | None = None,
        contrastive_target_bank_labels: torch.Tensor | None = None,
    ) -> LossOutput:
        # Model matmuls may run in bf16/fp16, but the paper's distributional and
        # correlation objectives are substantially more stable when accumulated
        # in fp32.
        prediction = prediction.float()
        target = target.float()
        control = control.float()
        cell_embedding = cell_embedding.float()
        if target_cell_embedding is not None:
            target_cell_embedding = target_cell_embedding.float()
        if contrastive_target_bank is not None:
            contrastive_target_bank = contrastive_target_bank.float()
        if prediction.dim() != 3:
            raise ValueError("prediction, target, and control must have shape [batch, set, genes]")
        if prediction.shape != target.shape or prediction.shape != control.shape:
            raise ValueError("prediction, target, and control must have matching shapes")
        if not prediction.shape[0] or not prediction.shape[1] or not prediction.shape[2]:
            raise ValueError("prediction, target, and control must be non-empty")
        if cell_embedding.shape[:2] != prediction.shape[:2]:
            raise ValueError("cell_embedding batch/set dimensions must match prediction")
        terms: dict[str, torch.Tensor] = {}
        pred_delta = fold_change(prediction, control)
        true_delta = fold_change(target, control)

        terms["mmd"] = energy_distance(prediction, target)
        terms["fold_change_ccc"] = (1.0 - concordance_ccc(pred_delta, true_delta, dim=1)).mean()
        terms["per_gene_ccc"] = (1.0 - concordance_ccc(prediction, target, dim=1)).mean()
        ratio = pred_delta.norm(dim=1) / (true_delta.norm(dim=1) + 1e-8)
        terms["delta_norm"] = (1.0 - ratio).pow(2).mean()
        terms["delta_mse"] = F.mse_loss(pred_delta, true_delta)

        if self.weights.l1_output:
            terms["l1_output"] = prediction.abs().mean()

        if (
            target_cell_embedding is not None or contrastive_target_bank is not None
        ) and self.weights.contrastive:
            terms["contrastive"] = info_nce(
                cell_embedding,
                target_cell_embedding,
                labels=contrastive_labels,
                target_center_bank=contrastive_target_bank,
                target_bank_labels=contrastive_target_bank_labels,
            )

        if prior_reconstructions is not None and prior_values is not None and self.weights.prior_reconstruction:
            prior_losses = []
            source_names = prior_source_names or tuple(prior_reconstructions)
            if len(source_names) != len(set(source_names)):
                raise ValueError("prior_source_names must contain unique source names")
            available_by_source: dict[str, torch.Tensor] = {}
            prior_world_size = 1
            if prior_missing is not None:
                if prior_missing.shape != (prediction.shape[0], len(source_names)):
                    raise ValueError(
                        "prior_missing must have shape [batch, len(prior_source_names)]"
                    )
                if prior_missing.dtype != torch.bool:
                    raise ValueError("prior_missing must have boolean dtype")
                available_counts = (~prior_missing).sum(dim=0).to(
                    device=prediction.device, dtype=prediction.dtype
                )
                if dist.is_available() and dist.is_initialized():
                    prior_world_size = dist.get_world_size()
                    # One vector collective handles every source, avoiding five
                    # latency-bound collectives per training microbatch.
                    dist.all_reduce(available_counts, op=dist.ReduceOp.SUM)
                available_by_source = {
                    source: available_counts[index]
                    for index, source in enumerate(source_names)
                }
            for source, reconstruction in prior_reconstructions.items():
                if source in prior_values:
                    reconstruction = reconstruction.float()
                    prior_value = prior_values[source].float()
                    if (
                        reconstruction.dim() != 2
                        or reconstruction.shape != prior_value.shape
                        or reconstruction.shape[0] != prediction.shape[0]
                    ):
                        raise ValueError(
                            f"prior reconstruction/value tensors for {source!r} must have "
                            "matching shape [batch, features]"
                        )
                    if prior_missing is not None:
                        if source not in source_names:
                            raise ValueError(
                                f"prior reconstruction source {source!r} is absent from "
                                "prior_source_names"
                            )
                        keep = ~prior_missing[:, source_names.index(source)]
                        squared = (reconstruction - prior_value).pow(2)
                        weight = keep[:, None].to(squared.dtype)
                        available = available_by_source[source]
                        # DDP averages gradients across ranks. Normalize by the
                        # global number of available samples and multiply each
                        # local contribution by world_size so the post-DDP
                        # gradient equals a true global masked mean.
                        denom = (available * squared.shape[1]).clamp_min(1.0)
                        source_loss = (squared * weight).sum() * prior_world_size / denom
                        # Always retain the graph edge. If a source is absent on
                        # every rank, the masked numerator is exactly zero and
                        # clamp_min makes the loss zero without a GPU->CPU item()
                        # synchronization in every source/microbatch.
                        prior_losses.append(source_loss)
                    else:
                        prior_losses.append(F.mse_loss(reconstruction, prior_value))
            if prior_losses:
                # Equation A.27 defines the auxiliary objective as a sum over
                # available prior sources, with one mean-squared error per
                # source.  Keeping this as a sum also preserves equal weight for
                # every source despite their different raw dimensions.
                terms["prior_reconstruction"] = torch.stack(prior_losses).sum()

        weighted_terms = {
            "mmd": self.weights.mmd * terms["mmd"],
            "fold_change_ccc": self.weights.fold_change_ccc * terms["fold_change_ccc"],
            "per_gene_ccc": self.weights.per_gene_ccc * terms["per_gene_ccc"],
            "delta_norm": self.weights.delta_norm * terms["delta_norm"],
            "delta_mse": self.weights.delta_mse * terms["delta_mse"],
        }
        if "l1_output" in terms:
            weighted_terms["l1_output"] = self.weights.l1_output * terms["l1_output"]
        if "contrastive" in terms:
            weighted_terms["contrastive"] = self.weights.contrastive * terms["contrastive"]
        if "prior_reconstruction" in terms:
            weighted_terms["prior_reconstruction"] = self.weights.prior_reconstruction * terms["prior_reconstruction"]

        total = torch.stack(list(weighted_terms.values())).sum()
        return LossOutput(total=total, terms=terms)
