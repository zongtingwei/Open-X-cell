from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from open_x_cell.config import ModelConfig, TrainingConfig, tiny_config
from open_x_cell.data import OpenXCellDataset, collate_samples, load_adata
from open_x_cell.diffusion import make_reveal_mask, mix_control_target
from open_x_cell.losses import OpenXCellLoss
from open_x_cell.model import OpenXCellModel
from open_x_cell.priors import make_zero_priors


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


def train_tiny(
    adata_path: str | Path,
    output_dir: str | Path,
    max_steps: int | None = None,
    training_config: TrainingConfig | None = None,
) -> Path:
    train_cfg = training_config or TrainingConfig()
    if max_steps is not None:
        train_cfg = TrainingConfig(
            batch_size=train_cfg.batch_size,
            set_size=train_cfg.set_size,
            max_steps=max_steps,
            learning_rate=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            beta1=train_cfg.beta1,
            beta2=train_cfg.beta2,
            warmup_fraction=train_cfg.warmup_fraction,
            grad_clip_norm=train_cfg.grad_clip_norm,
            seed=train_cfg.seed,
            device=train_cfg.device,
        )
    torch.manual_seed(train_cfg.seed)
    adata = load_adata(str(adata_path))
    model_cfg = tiny_config()
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
    loss_fn = OpenXCellLoss()
    fractions = model_cfg.diffusion_fractions
    iterator = iter(loader)
    for step in range(train_cfg.max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        x_control = batch.x_control.to(train_cfg.device)
        x_target = batch.x_target.to(train_cfg.device)
        fraction = fractions[step % len(fractions)]
        reveal = make_reveal_mask(
            batch_size=x_control.shape[0],
            set_size=x_control.shape[1],
            num_genes=x_control.shape[2],
            fraction=fraction,
            seed=train_cfg.seed + step,
            device=torch.device(train_cfg.device),
        )
        x_input = mix_control_target(x_control, x_target, reveal)
        priors = make_zero_priors(x_control.shape[0], model_cfg, device=train_cfg.device)
        out = model(x_input, reveal, batch.gene_ids.to(train_cfg.device), batch.perturbation_ids, priors)
        loss = loss_fn(out.expression, x_target, x_control, out.cell_embedding).total
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss during training")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        optimizer.step()
    checkpoint = Path(output_dir) / "checkpoint.pt"
    save_checkpoint(checkpoint, model, model_cfg, list(adata.var_names))
    return checkpoint
