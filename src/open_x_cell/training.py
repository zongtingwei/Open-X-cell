from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from open_x_cell.config import ModelConfig, TrainingConfig, model_config
from open_x_cell.data import OpenXCellDataset, collate_samples, load_adata
from open_x_cell.diffusion import make_reveal_mask, mix_control_target
from open_x_cell.losses import LossOutput, OpenXCellLoss
from open_x_cell.model import OpenXCellModel
from open_x_cell.priors import PriorStore, make_zero_priors


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
    fraction = model_config.diffusion_fractions[step % len(model_config.diffusion_fractions)]
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        weight_decay=train_cfg.weight_decay,
    )
    warmup_steps = int(train_cfg.max_steps * train_cfg.warmup_fraction)
    scheduler = make_warmup_cosine_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=train_cfg.max_steps)
    loss_fn = OpenXCellLoss()
    prior_store = PriorStore.from_files(prior_files or {}, model_cfg)
    iterator = iter(loader)
    history: list[dict[str, float]] = []
    for step in range(train_cfg.max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        loss_output = compute_batch_loss(
            model=model,
            batch=batch,
            model_config=model_cfg,
            loss_fn=loss_fn,
            prior_store=prior_store,
            step=step,
            seed=train_cfg.seed,
            device=train_cfg.device,
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
