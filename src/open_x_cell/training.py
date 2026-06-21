from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from open_x_cell.config import LossConfig, ModelConfig, TrainingConfig, model_config
from open_x_cell.data import OpenXCellDataset, collate_samples, load_adata
from open_x_cell.diffusion import make_reveal_mask, mix_control_target
from open_x_cell.losses import LossOutput, OpenXCellLoss
from open_x_cell.model import OpenXCellModel
from open_x_cell.priors import (
    PriorStore,
    init_gene_embedding_from_table,
    load_gene_embedding_table,
    make_zero_priors,
)


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: Path
    history: list[dict[str, float]]


def _config_from_dict(payload: dict) -> ModelConfig:
    payload = dict(payload)
    if "cross_attention_layers" in payload:
        payload["cross_attention_layers"] = tuple(payload["cross_attention_layers"])
    if "diffusion_fractions" in payload:
        payload["diffusion_fractions"] = tuple(payload["diffusion_fractions"])
    return ModelConfig(**payload)


def save_checkpoint(path: str | Path, model: OpenXCellModel, config: ModelConfig, var_names: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(config),
            "var_names": var_names,
        },
        target,
    )


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> tuple[OpenXCellModel, ModelConfig, list[str]]:
    payload = torch.load(path, map_location=map_location)
    config = _config_from_dict(payload["model_config"])
    model = OpenXCellModel(config)
    model.load_state_dict(payload["model_state"])
    model.to(map_location)
    model.eval()
    return model, config, list(payload.get("var_names", []))


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = max(0, min(warmup_steps, total_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps and step <= warmup_steps:
            return step / warmup_steps
        if step >= total_steps:
            return 0.0
        if total_steps == warmup_steps:
            return 0.0
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_batch_loss(
    model: OpenXCellModel,
    batch,
    model_config: ModelConfig,
    loss_fn: OpenXCellLoss,
    prior_store: PriorStore | None,
    step: int,
    seed: int,
    device: torch.device | str,
) -> LossOutput:
    target_device = torch.device(device)
    x_control = batch.x_control.to(target_device)
    x_target = batch.x_target.to(target_device)
    gene_ids = batch.gene_ids.to(target_device)
    perturbation_ids = batch.perturbation_ids.to(target_device)
    # Paper samples the reveal fraction alpha ~ Uniform{0.25,0.5,0.75,1.0} each step
    # (not a fixed step%len cycle). Seed the draw by step for reproducibility.
    fractions = model_config.diffusion_fractions
    draw = torch.randint(len(fractions), (1,), generator=torch.Generator().manual_seed(seed + step))
    fraction = fractions[int(draw.item())]
    reveal = make_reveal_mask(
        batch_size=x_control.shape[0],
        set_size=x_control.shape[1],
        num_genes=x_control.shape[2],
        fraction=fraction,
        seed=seed + step,
        device=target_device,
    )
    x_input = mix_control_target(x_control, x_target, reveal)
    priors = (
        prior_store.batch_for_perturbations(batch.perturbations, model_config, device=target_device)
        if prior_store is not None
        else make_zero_priors(x_control.shape[0], model_config, device=target_device)
    )
    out = model(x_input, reveal, gene_ids, perturbation_ids, priors)
    with torch.no_grad():
        full_reveal = torch.ones_like(reveal, dtype=torch.bool)
        target_out = model(x_target, full_reveal, gene_ids, perturbation_ids, priors)
    return loss_fn(
        out.expression,
        x_target,
        x_control,
        out.cell_embedding,
        target_cell_embedding=target_out.cell_embedding.detach(),
        prior_reconstructions=out.prior_reconstructions,
        prior_values=priors.values,
        prior_missing=priors.missing,
        prior_source_names=priors.source_names,
    )


def _apply_scgpt(model, var_names, scgpt_init, scgpt_warm_start, gene_id_map,
                 min_overlap, strict_overlap) -> None:
    if scgpt_warm_start is not None:
        from open_x_cell.scgpt import warm_start_from_scgpt
        ens2sym = torch.load(gene_id_map, map_location="cpu") if gene_id_map is not None else None
        stats = warm_start_from_scgpt(
            model, var_names, scgpt_warm_start, ens2sym=ens2sym,
            min_overlap=min_overlap, strict_overlap=strict_overlap,
        )
        print(f"scGPT warm-start: mapped {stats.n_mapped}/{stats.n_genes} "
              f"({stats.overlap:.1%}; symbol={stats.n_via_symbol}, ensembl={stats.n_via_ensembl}, "
              f"fallback={stats.n_fallback}); transformer tensors loaded={stats.loaded_tensors}")
        for note in stats.notes:
            print(f"  note: {note}")
    elif scgpt_init is not None:
        table = load_gene_embedding_table(scgpt_init)
        n_init, n_total = init_gene_embedding_from_table(model, var_names, table)
        print(f"scGPT init: {n_init}/{n_total} gene embeddings initialized from {scgpt_init}")


def _run_loop(
    model, loader, model_cfg, prior_store, loss_fn, train_cfg,
    *, max_steps: int, learning_rate: float, warmup_fraction: float,
) -> list[dict[str, float]]:
    """A single AdamW + warmup-cosine training loop; returns per-step loss history."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2), weight_decay=train_cfg.weight_decay,
    )
    warmup_steps = int(max_steps * warmup_fraction)
    scheduler = make_warmup_cosine_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=max_steps)
    iterator = iter(loader)
    history: list[dict[str, float]] = []
    for step in range(max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        loss_output = compute_batch_loss(
            model=model, batch=batch, model_config=model_cfg, loss_fn=loss_fn,
            prior_store=prior_store, step=step, seed=train_cfg.seed, device=train_cfg.device,
        )
        loss = loss_output.total
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss during training")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        history.append({name: float(value.detach().cpu()) for name, value in loss_output.terms.items()})
        history[-1]["total"] = float(loss.detach().cpu())
    return history


def _training_config_with_overrides(
    training_config: TrainingConfig | None,
    max_steps: int | None,
    batch_size: int | None = None,
    set_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    device: str | None = None,
) -> TrainingConfig:
    cfg = training_config or TrainingConfig()
    updates = {}
    if max_steps is not None:
        updates["max_steps"] = max_steps
    if batch_size is not None:
        updates["batch_size"] = batch_size
    if set_size is not None:
        updates["set_size"] = set_size
    if learning_rate is not None:
        updates["learning_rate"] = learning_rate
    if weight_decay is not None:
        updates["weight_decay"] = weight_decay
    if device is not None:
        updates["device"] = device
    return replace(cfg, **updates)


def train_model(
    adata_path: str | Path,
    output_dir: str | Path,
    model_size: str = "tiny",
    max_steps: int | None = None,
    training_config: TrainingConfig | None = None,
    prior_files: dict[str, str | Path | None] | None = None,
    normalize: str = "cp10k-log1p",
    batch_size: int | None = None,
    set_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    device: str | None = None,
    scgpt_init: str | Path | None = None,
    scgpt_warm_start: str | Path | None = None,
    gene_id_map: str | Path | None = None,
    scgpt_min_overlap: float = 0.95,
    scgpt_strict_overlap: bool = False,
    loss_weights: "LossConfig | None" = None,
) -> TrainingResult:
    train_cfg = _training_config_with_overrides(
        training_config,
        max_steps=max_steps,
        batch_size=batch_size,
        set_size=set_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
    )
    torch.manual_seed(train_cfg.seed)
    adata = load_adata(str(adata_path), normalize=normalize)
    model_cfg = model_config(model_size)
    if adata.n_vars < model_cfg.context_length:
        raise ValueError(
            f"{model_size} requires at least {model_cfg.context_length} genes; got {adata.n_vars}. "
            "Use --model-size tiny for smoke tests or provide full-gene AnnData."
        )
    if adata.n_vars > model_cfg.num_genes:
        adata = adata[:, : model_cfg.num_genes].copy()
    dataset = OpenXCellDataset(
        adata,
        set_size=train_cfg.set_size,
        context_length=model_cfg.context_length,
        seed=train_cfg.seed,
    )
    loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate_samples)
    model = OpenXCellModel(model_cfg).to(train_cfg.device)
    _apply_scgpt(model, list(adata.var_names), scgpt_init, scgpt_warm_start,
                 gene_id_map, scgpt_min_overlap, scgpt_strict_overlap)
    loss_fn = OpenXCellLoss(loss_weights)
    prior_store = PriorStore.from_files(prior_files or {}, model_cfg)
    history = _run_loop(
        model, loader, model_cfg, prior_store, loss_fn, train_cfg,
        max_steps=train_cfg.max_steps, learning_rate=train_cfg.learning_rate,
        warmup_fraction=train_cfg.warmup_fraction,
    )
    checkpoint = Path(output_dir) / "checkpoint.pt"
    save_checkpoint(checkpoint, model, model_cfg, list(adata.var_names))
    return TrainingResult(checkpoint=checkpoint, history=history)


def high_effect_perturbations(adata, top_frac: float = 0.05) -> list[str]:
    """Perturbations with the largest pseudobulk effect ||mean_pert - mean_control||.

    Returns the top ``top_frac`` fraction (at least one) by descending effect size.

    Note: this is an approximation of the paper's high-effect ranking, which first
    keeps perturbations with detectable effects (binary classifier, FDR < 0.05) and
    then ranks by the number of significant LASSO-regression features. We use the
    pseudobulk L2 delta norm as a dependency-free proxy for effect size.
    """
    from open_x_cell.data import dense_matrix

    matrix = dense_matrix(adata)
    labels = adata.obs["perturbation"].astype(str).to_numpy()
    control_key = "is_control"
    control_mean = matrix[adata.obs[control_key].to_numpy(dtype=bool)].mean(axis=0)
    perts = [p for p in sorted(set(labels)) if p != "control"]
    effects = {p: float(np.linalg.norm(matrix[labels == p].mean(axis=0) - control_mean)) for p in perts}
    ordered = sorted(perts, key=lambda p: effects[p], reverse=True)
    k = max(1, int(round(top_frac * len(ordered))))
    return ordered[:k]


def train_model_curriculum(
    adata_path: str | Path,
    output_dir: str | Path,
    model_size: str = "tiny",
    phase1_steps: int = 100,
    phase2_steps: int = 100,
    top_frac: float = 0.05,
    phase2_learning_rate: float | None = None,
    phase2_warmup_fraction: float | None = None,
    training_config: TrainingConfig | None = None,
    normalize: str = "cp10k-log1p",
    prior_files: dict[str, str | Path | None] | None = None,
    device: str | None = None,
    scgpt_init: str | Path | None = None,
    scgpt_warm_start: str | Path | None = None,
    gene_id_map: str | Path | None = None,
    scgpt_min_overlap: float = 0.95,
    scgpt_strict_overlap: bool = False,
    loss_weights: "LossConfig | None" = None,
) -> TrainingResult:
    """Two-phase curriculum (paper Ultra, §4.x):

    * Phase 1 — high-effect initialization: train on the top ``top_frac`` highest-effect
      perturbations only, keeping all loss terms.
    * Phase 2 — full corpus: continue on every perturbation with a FRESH optimizer (its
      own LR / warmup), with the per-gene CCC loss term dropped (paper: full-dataset
      training removes per-gene CCC).
    """
    train_cfg = _training_config_with_overrides(training_config, max_steps=None, device=device)
    torch.manual_seed(train_cfg.seed)
    adata = load_adata(str(adata_path), normalize=normalize)
    model_cfg = model_config(model_size)
    if adata.n_vars < model_cfg.context_length:
        raise ValueError(f"{model_size} requires at least {model_cfg.context_length} genes; got {adata.n_vars}.")
    if adata.n_vars > model_cfg.num_genes:
        adata = adata[:, : model_cfg.num_genes].copy()

    model = OpenXCellModel(model_cfg).to(train_cfg.device)
    _apply_scgpt(model, list(adata.var_names), scgpt_init, scgpt_warm_start,
                 gene_id_map, scgpt_min_overlap, scgpt_strict_overlap)
    loss_fn = OpenXCellLoss(loss_weights)
    prior_store = PriorStore.from_files(prior_files or {}, model_cfg)

    def _loader(ad):
        ds = OpenXCellDataset(ad, set_size=train_cfg.set_size,
                              context_length=model_cfg.context_length, seed=train_cfg.seed)
        return DataLoader(ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate_samples)

    # Phase 1 (high-effect initialization): top-`top_frac` high-effect perturbations,
    # ALL losses (paper: high-effect curriculum keeps every loss term).
    selected = high_effect_perturbations(adata, top_frac=top_frac)
    labels = adata.obs["perturbation"].astype(str).to_numpy()
    keep = adata.obs["is_control"].to_numpy(dtype=bool) | np.isin(labels, selected)
    adata1 = adata[keep].copy()
    print(f"curriculum phase 1 (high-effect init): {len(selected)} perturbations "
          f"(top {top_frac:.0%}): {selected}")
    history = _run_loop(model, _loader(adata1), model_cfg, prior_store, loss_fn, train_cfg,
                        max_steps=phase1_steps, learning_rate=train_cfg.learning_rate,
                        warmup_fraction=train_cfg.warmup_fraction)

    # Phase 2 (full corpus): fresh optimizer (own LR / warmup), per-gene CCC removed
    # (paper: full-dataset training drops the per-gene CCC term).
    phase2_weights = replace(loss_weights or LossConfig(), per_gene_ccc=0.0)
    loss_fn2 = OpenXCellLoss(phase2_weights)
    print("curriculum phase 2: full corpus (per-gene CCC dropped)")
    history += _run_loop(
        model, _loader(adata), model_cfg, prior_store, loss_fn2, train_cfg,
        max_steps=phase2_steps,
        learning_rate=phase2_learning_rate if phase2_learning_rate is not None else train_cfg.learning_rate,
        warmup_fraction=phase2_warmup_fraction if phase2_warmup_fraction is not None else train_cfg.warmup_fraction,
    )

    checkpoint = Path(output_dir) / "checkpoint.pt"
    save_checkpoint(checkpoint, model, model_cfg, list(adata.var_names))
    return TrainingResult(checkpoint=checkpoint, history=history)


def train_tiny(
    adata_path: str | Path,
    output_dir: str | Path,
    max_steps: int | None = None,
    training_config: TrainingConfig | None = None,
) -> Path:
    return train_model(
        adata_path=adata_path,
        output_dir=output_dir,
        model_size="tiny",
        max_steps=max_steps,
        training_config=training_config,
        normalize="none",
    ).checkpoint
