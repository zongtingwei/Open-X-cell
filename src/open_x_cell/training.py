from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from open_x_cell.config import LossConfig, ModelConfig, TrainingConfig, model_config
from open_x_cell.data import OpenXCellDataset, boolean_obs_mask, collate_samples, load_adata
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


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_record(path: str | Path) -> dict[str, str | int]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": str(source),
        "bytes": int(source.stat().st_size),
        "sha256": _sha256_file(source),
    }


def _verify_artifact_record(record: dict, label: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"{label} artifact record must be a mapping")
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} artifact is missing: {path}")
    actual = _artifact_record(path)
    expected = {
        "path": str(path),
        "bytes": record.get("bytes"),
        "sha256": record.get("sha256"),
    }
    if actual != expected:
        raise ValueError(f"{label} artifact changed since the original training run")
    return path


def _retain_resume_checkpoint(
    resume_from: str | Path,
    output_dir: str | Path,
    completed_steps: int,
) -> dict:
    """Hard-link the exact resume input so later ``latest.pt`` writes cannot replace it."""
    source = Path(resume_from).resolve()
    source_record = _artifact_record(source)
    retained_dir = Path(output_dir).resolve() / "resume_inputs"
    retained_dir.mkdir(parents=True, exist_ok=True)
    retained = retained_dir / (
        f"step_{completed_steps:07d}_{source_record['sha256'][:16]}.pt"
    )
    if not retained.exists():
        try:
            os.link(source, retained)
        except OSError:
            temporary = retained.with_suffix(retained.suffix + ".tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(retained)
    retained_record = _artifact_record(retained)
    if (
        retained_record["bytes"] != source_record["bytes"]
        or retained_record["sha256"] != source_record["sha256"]
    ):
        raise ValueError("retained resume checkpoint differs from the requested input")
    return {
        "source_path": str(source),
        "checkpoint": retained_record,
        "completed_steps": int(completed_steps),
    }


def _atomic_json(path: str | Path, payload) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _source_fingerprints() -> dict[str, str]:
    """Hash the exact Open-X-Cell package sources used for a training run."""
    package_dir = Path(__file__).resolve().parent
    fingerprints: dict[str, str] = {}
    for path in sorted(package_dir.glob("*.py")):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        fingerprints[path.name] = digest.hexdigest()
    return fingerprints


def _json_normalized(value):
    """Normalize dataclass tuples to their JSON list representation."""
    return json.loads(json.dumps(value))


def _config_from_dict(payload: dict) -> ModelConfig:
    payload = dict(payload)
    if "cross_attention_layers" in payload:
        payload["cross_attention_layers"] = tuple(payload["cross_attention_layers"])
    if "diffusion_fractions" in payload:
        payload["diffusion_fractions"] = tuple(payload["diffusion_fractions"])
    return ModelConfig(**payload)


def _load_vocabulary(path: str | Path | None, measured_genes) -> list[str]:
    if path is None:
        vocabulary = list(map(str, measured_genes))
    else:
        source = Path(path)
        if source.suffix.lower() == ".json":
            with open(source, encoding="utf-8") as handle:
                payload = json.load(handle)
        elif source.suffix.lower() in {".pt", ".pth"}:
            payload = torch.load(source, map_location="cpu", weights_only=False)
        else:
            with open(source, encoding="utf-8") as handle:
                payload = [line.strip() for line in handle if line.strip()]
        if isinstance(payload, dict):
            # Accept either gene->index or index->gene mappings.
            if payload and all(isinstance(value, int) for value in payload.values()):
                vocabulary = [gene for gene, _ in sorted(payload.items(), key=lambda item: item[1])]
            else:
                vocabulary = [payload[key] for key in sorted(payload, key=lambda value: int(value))]
        else:
            vocabulary = list(payload)
        vocabulary = list(map(str, vocabulary))
    if len(vocabulary) != len(set(vocabulary)):
        raise ValueError("model vocabulary contains duplicate genes")
    if not vocabulary or any(not gene.strip() or gene != gene.strip() for gene in vocabulary):
        raise ValueError("model vocabulary must contain non-empty canonical gene names")
    return vocabulary


def _unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def save_checkpoint(
    path: str | Path,
    model: OpenXCellModel,
    config: ModelConfig,
    var_names: list[str],
    training_state: dict | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": _unwrap_model(model).state_dict(),
        "model_config": asdict(config),
        "var_names": var_names,
    }
    if training_state is not None:
        payload["training_state"] = training_state
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def _read_checkpoint_payload(path: str | Path, map_location: str = "cpu") -> dict:
    if Path(path).suffix == ".safetensors":
        from safetensors.torch import load_file
        from open_x_cell.bundle import verify_bundle
        directory = Path(path).resolve().parent
        verify_bundle(directory)
        if Path(path).name != "model.safetensors":
            raise ValueError("portable model filename must be model.safetensors")
        payload = {
            "model_state": load_file(str(path), device=str(map_location)),
            "model_config": json.loads((directory / "config.json").read_text()),
            "var_names": json.loads((directory / "genes.json").read_text()),
        }
    else:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    return payload


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> tuple[OpenXCellModel, ModelConfig, list[str]]:
    payload = _read_checkpoint_payload(path, map_location)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    if not isinstance(payload.get("model_config"), dict):
        raise ValueError("checkpoint lacks a valid model configuration")
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError("checkpoint model state must be a non-empty mapping")
    for name, tensor in model_state.items():
        if not torch.is_tensor(tensor):
            raise ValueError(f"checkpoint model entry is not a tensor: {name}")
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            raise ValueError(f"checkpoint model tensor is non-finite: {name}")
    config = _config_from_dict(payload["model_config"])
    var_names = payload.get("var_names", [])
    if not isinstance(var_names, (list, tuple)):
        raise ValueError("checkpoint gene vocabulary must be a list")
    var_names = list(map(str, var_names))
    if (
        not var_names
        or len(var_names) > config.num_genes
        or len(var_names) != len(set(var_names))
        or any(not gene.strip() or gene != gene.strip() for gene in var_names)
    ):
        raise ValueError(
            "checkpoint gene vocabulary must be non-empty, canonical, unique, and within model size"
        )
    model = OpenXCellModel(config)
    model.load_state_dict(model_state)
    model.to(map_location)
    model.eval()
    return model, config, var_names


def _initialize_from_xcell_checkpoint(
    model: OpenXCellModel,
    checkpoint_path: str | Path,
    destination_vocabulary: list[str],
    expected_config: ModelConfig,
    min_overlap: float = 0.95,
    unmapped_token: str = "<mask>",
) -> dict[str, int | float | str | bool | None]:
    """Warm-start model weights while deliberately discarding training state.

    Continual pretraining and downstream fine-tuning are different optimization
    runs in the paper.  This loader therefore transfers only model parameters;
    the caller creates a fresh AdamW optimizer, scheduler, and history.  Gene-row
    parameters are remapped by canonical gene symbol so a compatible checkpoint
    remains safe to use when the downstream vocabulary order differs. Appendix
    A.9's collator remap is materialized into destination-indexed parameter rows:
    shared genes copy their source row and unmapped genes copy the source
    ``<mask>`` row. This is numerically equivalent while keeping the resulting
    checkpoint self-contained in its destination vocabulary order.
    """
    if (
        isinstance(min_overlap, bool)
        or not isinstance(min_overlap, Real)
        or not math.isfinite(min_overlap)
        or not 0 <= min_overlap <= 1
    ):
        raise ValueError("pretrained vocabulary overlap threshold must be within [0, 1]")
    if not isinstance(unmapped_token, str) or not unmapped_token:
        raise ValueError("pretrained unmapped token must be a non-empty string")
    source = Path(checkpoint_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = _read_checkpoint_payload(source)
    if not isinstance(payload, dict):
        raise ValueError("pretrained checkpoint payload must be a mapping")
    if not isinstance(payload.get("model_config"), dict):
        raise ValueError("pretrained checkpoint lacks a valid model configuration")
    checkpoint_config = _config_from_dict(payload["model_config"])
    if checkpoint_config != expected_config:
        raise ValueError("pretrained checkpoint architecture does not match requested model size")
    source_vocabulary_raw = payload.get("var_names")
    if not isinstance(source_vocabulary_raw, (list, tuple)):
        raise ValueError("pretrained checkpoint gene vocabulary must be a list")
    source_vocabulary = list(map(str, source_vocabulary_raw))
    if (
        not source_vocabulary
        or len(source_vocabulary) > expected_config.num_genes
        or len(source_vocabulary) != len(set(source_vocabulary))
        or any(not gene.strip() or gene != gene.strip() for gene in source_vocabulary)
    ):
        raise ValueError(
            "pretrained checkpoint gene vocabulary must be non-empty, canonical, unique, "
            "and within model size"
        )
    if (
        not destination_vocabulary
        or len(destination_vocabulary) > expected_config.num_genes
        or len(destination_vocabulary) != len(set(destination_vocabulary))
    ):
        raise ValueError("destination gene vocabulary is invalid")

    source_state = payload.get("model_state")
    destination_state = model.state_dict()
    if not isinstance(source_state, dict) or set(source_state) != set(destination_state):
        raise ValueError("pretrained checkpoint model parameter inventory differs from the model")
    gene_row_parameters = {"gene_embedding.weight", "gene_bias.weight"}
    for name, destination_tensor in destination_state.items():
        source_tensor = source_state[name]
        if not torch.is_tensor(source_tensor) or source_tensor.shape != destination_tensor.shape:
            raise ValueError(f"pretrained checkpoint tensor has the wrong shape or type: {name}")
        if torch.is_floating_point(source_tensor) and not torch.isfinite(source_tensor).all():
            raise ValueError(f"pretrained checkpoint tensor is non-finite: {name}")

    source_index = {gene: index for index, gene in enumerate(source_vocabulary)}
    destination_index = {gene: index for index, gene in enumerate(destination_vocabulary)}
    shared_genes = sorted(set(source_index).intersection(destination_index))
    if not shared_genes:
        raise ValueError("pretrained and destination gene vocabularies do not overlap")
    overlap_fraction = len(shared_genes) / len(destination_vocabulary)
    if overlap_fraction < min_overlap:
        raise ValueError(
            "pretrained/destination gene overlap is below the Appendix A.9 threshold: "
            f"{len(shared_genes)}/{len(destination_vocabulary)} ({overlap_fraction:.2%}) "
            f"< {min_overlap:.2%}"
        )
    unmapped_genes = sorted(set(destination_vocabulary) - set(source_index))
    mask_index = source_index.get(unmapped_token)
    if unmapped_genes and mask_index is None:
        raise ValueError(
            f"pretrained vocabulary lacks {unmapped_token!r}, required to remap "
            f"{len(unmapped_genes)} downstream-only genes"
        )

    remapped_state = {name: tensor.clone() for name, tensor in source_state.items()}
    if source_vocabulary != destination_vocabulary:
        # Materialize Appendix A.9's source-vocabulary lookup into destination
        # rows. Unused table rows retain deterministic destination initialization.
        for name in gene_row_parameters:
            remapped = destination_state[name].clone()
            for gene in destination_vocabulary:
                source_row = source_index.get(gene, mask_index)
                assert source_row is not None
                remapped[destination_index[gene]].copy_(
                    source_state[name][source_row]
                )
            remapped_state[name] = remapped
    model.load_state_dict(remapped_state, strict=True)
    return {
        "path": str(source),
        "source_vocabulary_size": len(source_vocabulary),
        "destination_vocabulary_size": len(destination_vocabulary),
        "overlap_genes": len(shared_genes),
        "overlap_fraction_of_destination": overlap_fraction,
        "minimum_overlap_required": float(min_overlap),
        "unmapped_genes": len(unmapped_genes),
        "unmapped_token": unmapped_token if unmapped_genes else None,
        "unmapped_token_source_index": mask_index if unmapped_genes else None,
        "vocabulary_remap_semantics": (
            "appendix_a9_source_lookup_materialized_into_destination_parameter_rows"
        ),
        "vocabulary_identical": source_vocabulary == destination_vocabulary,
        "optimizer_state_loaded": False,
        "scheduler_state_loaded": False,
    }


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = max(0, min(warmup_steps, total_steps))

    def lr_lambda(step: int) -> float:
        # LambdaLR evaluates epoch 0 during construction, before the first
        # optimizer update. Map that state to one-indexed update 1 so the first
        # update uses the first warmup LR instead of a silent zero-LR no-op.
        return _warmup_cosine_multiplier(step + 1, warmup_steps, total_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _warmup_cosine_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    if step <= 0:
        return 0.0
    if warmup_steps and step <= warmup_steps:
        return step / warmup_steps
    if step > total_steps:
        return 0.0
    if total_steps == warmup_steps:
        return 1.0
    # The cosine portion contains ``total_steps - warmup_steps`` actual
    # optimizer updates. Sample it at zero-based update indices so every
    # scheduled update has a non-zero LR; the scheduler state prepared only
    # after the final update is exactly zero.
    progress = (step - warmup_steps - 1) / (total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _validate_resume_training_state(
    resume_payload: dict,
    train_cfg: TrainingConfig,
    *,
    max_steps: int,
    source_sha256: dict[str, str],
    expected_optimizer_parameters: int | None = None,
    expected_optimizer_active_ids: set[int] | None = None,
    expected_optimizer_parameter_shapes: list[tuple[int, ...]] | None = None,
) -> tuple[dict, int, list[dict[str, float]]]:
    """Validate recovery state before it is allowed to mutate a run directory."""
    if not isinstance(resume_payload, dict):
        raise ValueError("resume checkpoint payload must be a mapping")
    state = resume_payload.get("training_state")
    if not isinstance(state, dict):
        raise ValueError("resume checkpoint training_state must be a mapping")

    saved_training_config = state.get("training_config")
    if not isinstance(saved_training_config, dict):
        raise ValueError("resume checkpoint lacks a valid training configuration")
    saved_training_config = dict(saved_training_config)
    current_training_config = asdict(train_cfg)
    saved_training_config.pop("device", None)
    current_training_config.pop("device", None)
    if saved_training_config != current_training_config:
        raise ValueError("resume checkpoint training configuration differs from this run")
    if state.get("source_sha256") != source_sha256:
        raise ValueError("resume checkpoint was created by different package sources")

    raw_start_step = state.get("completed_steps")
    if (
        isinstance(raw_start_step, bool)
        or not isinstance(raw_start_step, (int, np.integer))
        or raw_start_step < 0
    ):
        raise ValueError("resume checkpoint completed_steps must be a non-negative integer")
    start_step = int(raw_start_step)
    if start_step >= max_steps:
        raise ValueError(
            f"resume checkpoint has {start_step} completed steps; requested training ends at "
            f"{max_steps} and therefore has no remaining step"
        )

    raw_history = state.get("history")
    if not isinstance(raw_history, list) or len(raw_history) != start_step:
        raise ValueError("resume checkpoint history length differs from completed_steps")
    history = list(raw_history)
    previous_elapsed = -math.inf
    warmup_steps = int(max_steps * train_cfg.warmup_fraction)
    for expected_step, row in enumerate(history, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"resume checkpoint history is malformed at step {expected_step}")
        step_value = row.get("step")
        if (
            isinstance(step_value, bool)
            or not isinstance(step_value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(step_value))
            or float(step_value) != expected_step
        ):
            raise ValueError(f"resume checkpoint history is malformed at step {expected_step}")
        for key, value in row.items():
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise ValueError(
                    f"resume checkpoint history has a non-numeric {key!r} at step {expected_step}"
                )
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"resume checkpoint history has a non-finite {key!r} at step {expected_step}"
                )
        required = {"step", "total", "grad_norm", "lr", "elapsed_seconds"}
        if not required.issubset(row):
            raise ValueError(
                f"resume checkpoint history lacks required fields at step {expected_step}"
            )
        if float(row["grad_norm"]) < 0:
            raise ValueError(f"resume checkpoint history has a negative gradient norm at step {expected_step}")
        elapsed = float(row["elapsed_seconds"])
        if elapsed < previous_elapsed:
            raise ValueError("resume checkpoint elapsed history is not monotonic")
        previous_elapsed = elapsed
        expected_lr = train_cfg.learning_rate * _warmup_cosine_multiplier(
            expected_step, warmup_steps, max_steps
        )
        if not math.isclose(float(row["lr"]), expected_lr, rel_tol=0, abs_tol=1e-12):
            raise ValueError(
                f"resume checkpoint history has the wrong learning rate at step {expected_step}"
            )

    scheduler_state = state.get("scheduler_state")
    if not isinstance(scheduler_state, dict) or scheduler_state.get("last_epoch") != start_step:
        raise ValueError("resume checkpoint scheduler state differs from completed_steps")
    optimizer_state = state.get("optimizer_state")
    if not isinstance(optimizer_state, dict):
        raise ValueError("resume checkpoint lacks optimizer state")
    optimizer_entries = optimizer_state.get("state")
    optimizer_groups = optimizer_state.get("param_groups")
    if not isinstance(optimizer_entries, dict) or not isinstance(optimizer_groups, list):
        raise ValueError("resume checkpoint optimizer state is malformed")
    if len(optimizer_groups) != 1 or not isinstance(optimizer_groups[0], dict):
        raise ValueError("resume checkpoint must contain one optimizer parameter group")
    saved_group = optimizer_groups[0]
    saved_parameters = saved_group.get("params")
    if (
        not isinstance(saved_parameters, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in saved_parameters)
        or len(saved_parameters) != len(set(saved_parameters))
    ):
        raise ValueError("resume checkpoint optimizer parameter inventory is malformed")
    if (
        expected_optimizer_parameters is not None
        and len(saved_parameters) != expected_optimizer_parameters
    ):
        raise ValueError("resume checkpoint optimizer parameter inventory differs from the model")
    if expected_optimizer_parameters is not None and saved_parameters != list(
        range(expected_optimizer_parameters)
    ):
        raise ValueError("resume checkpoint optimizer parameter ordering differs from the model")
    if not set(optimizer_entries).issubset(saved_parameters):
        raise ValueError("resume checkpoint optimizer state refers to unknown parameters")
    if start_step > 0 and expected_optimizer_active_ids is not None and set(
        optimizer_entries
    ) != expected_optimizer_active_ids:
        raise ValueError("resume checkpoint optimizer state inventory differs from trainable parameters")
    if (
        tuple(saved_group.get("betas", ())) != (train_cfg.beta1, train_cfg.beta2)
        or float(saved_group.get("weight_decay", math.nan)) != train_cfg.weight_decay
        or float(saved_group.get("initial_lr", math.nan)) != train_cfg.learning_rate
    ):
        raise ValueError("resume checkpoint optimizer hyperparameters differ from this run")
    expected_lr = train_cfg.learning_rate * _warmup_cosine_multiplier(
        start_step + 1, warmup_steps, max_steps
    )
    if not math.isclose(
        float(saved_group.get("lr", math.nan)), expected_lr, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("resume checkpoint optimizer learning rate differs from the saved step")
    expected_group_flags = {
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
    }
    if any(saved_group.get(key) != value for key, value in expected_group_flags.items()):
        raise ValueError("resume checkpoint optimizer algorithm settings differ from AdamW defaults")

    expected_state_fields = {"step", "exp_avg", "exp_avg_sq"}
    for parameter_id, parameter_state in optimizer_entries.items():
        if not isinstance(parameter_state, dict) or set(parameter_state) != expected_state_fields:
            raise ValueError(f"resume checkpoint optimizer entry {parameter_id} is malformed")
        step_tensor = parameter_state["step"]
        if not torch.is_tensor(step_tensor) or step_tensor.numel() != 1 or float(step_tensor) != start_step:
            raise ValueError(f"resume checkpoint optimizer entry {parameter_id} has the wrong step")
        if expected_optimizer_parameter_shapes is not None:
            if not 0 <= parameter_id < len(expected_optimizer_parameter_shapes):
                raise ValueError("resume checkpoint optimizer parameter ID is out of range")
            expected_shape = expected_optimizer_parameter_shapes[parameter_id]
            if tuple(parameter_state["exp_avg"].shape) != expected_shape or tuple(
                parameter_state["exp_avg_sq"].shape
            ) != expected_shape:
                raise ValueError(
                    f"resume checkpoint optimizer entry {parameter_id} has the wrong tensor shape"
                )

    expected_scheduler = {
        "last_epoch": start_step,
        "_step_count": start_step + 1,
        "base_lrs": [train_cfg.learning_rate],
        "_last_lr": [expected_lr],
    }
    for key, expected_value in expected_scheduler.items():
        actual_value = scheduler_state.get(key)
        if key == "_last_lr" and isinstance(actual_value, list) and len(actual_value) == 1:
            if math.isclose(float(actual_value[0]), expected_lr, rel_tol=0, abs_tol=1e-12):
                continue
        if actual_value != expected_value:
            raise ValueError(f"resume checkpoint scheduler field {key!r} differs from the saved step")

    def ensure_finite_tensors(value, label: str) -> None:
        if torch.is_tensor(value):
            if torch.is_floating_point(value) and not torch.isfinite(value).all():
                raise ValueError(f"resume checkpoint contains a non-finite tensor at {label}")
        elif isinstance(value, dict):
            for key, item in value.items():
                ensure_finite_tensors(item, f"{label}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                ensure_finite_tensors(item, f"{label}[{index}]")

    ensure_finite_tensors(optimizer_state, "optimizer_state")
    ensure_finite_tensors(scheduler_state, "scheduler_state")
    ensure_finite_tensors(state.get("scaler_state", {}), "scaler_state")
    return state, start_step, history


def compute_batch_loss(
    model: OpenXCellModel,
    batch,
    model_config: ModelConfig,
    loss_fn: OpenXCellLoss,
    prior_store: PriorStore | None,
    step: int,
    seed: int,
    device: torch.device | str,
    contrastive_target_bank: torch.Tensor | None = None,
    contrastive_target_bank_labels: torch.Tensor | None = None,
) -> LossOutput:
    target_device = torch.device(device)
    x_control = batch.x_control.to(target_device)
    x_target = batch.x_target.to(target_device)
    gene_ids = batch.gene_ids.to(target_device)
    perturbation_ids = batch.perturbation_ids.to(target_device)
    # Paper samples alpha ~ Uniform{0.25,0.5,0.75,1.0} independently for each
    # batch sample. Give every DDP rank a disjoint deterministic RNG stream while
    # retaining one shared gene mask across the 64 cells within each set.
    fractions = model_config.diffusion_fractions
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    # The reveal fraction and the gene subset are independent random choices in
    # the paper. Give them adjacent, non-overlapping deterministic streams
    # instead of resetting two generators to the same seed (which couples the
    # fraction draw to the first values used by ``randperm``).
    stream_index = step * world_size + rank
    fraction_seed = seed + 2 * stream_index
    mask_seed = fraction_seed + 1
    draws = torch.randint(
        len(fractions),
        (x_control.shape[0],),
        generator=torch.Generator().manual_seed(fraction_seed),
    )
    sample_fractions = tuple(fractions[int(index)] for index in draws.tolist())
    reveal = make_reveal_mask(
        batch_size=x_control.shape[0],
        set_size=x_control.shape[1],
        num_genes=x_control.shape[2],
        fraction=sample_fractions,
        seed=mask_seed,
        device=target_device,
    )
    x_input = mix_control_target(x_control, x_target, reveal)
    priors = (
        prior_store.batch_for_perturbations(batch.perturbations, model_config, device=target_device)
        if prior_store is not None
        else make_zero_priors(x_control.shape[0], model_config, device=target_device)
    )
    out = model(
        x_input,
        reveal,
        gene_ids,
        perturbation_ids,
        priors,
        reconstruct_priors=bool(loss_fn.weights.prior_reconstruction),
    )
    target_cell_embedding = None
    if loss_fn.weights.contrastive and contrastive_target_bank is None:
        with torch.no_grad():
            full_reveal = torch.ones_like(reveal, dtype=torch.bool)
            # The target branch is stop-gradient. Calling the underlying module
            # avoids starting a second DDP reducer pass before the primary
            # forward is backed up.
            target_model = _unwrap_model(model)
            target_out = target_model(
                x_target,
                full_reveal,
                gene_ids,
                perturbation_ids,
                priors,
                reconstruct_priors=False,
            )
            target_cell_embedding = target_out.cell_embedding.detach().float()
    # Explicitly leave the surrounding model autocast region.  In particular,
    # the pairwise matrix products in the GeomLoss-compatible energy statistic
    # must remain fp32 rather than being silently autocast back to bf16/fp16.
    with torch.autocast(device_type=target_device.type, enabled=False):
        return loss_fn(
            out.expression.float(),
            x_target.float(),
            x_control.float(),
            out.cell_embedding.float(),
            target_cell_embedding=target_cell_embedding,
            contrastive_target_bank=contrastive_target_bank,
            contrastive_target_bank_labels=contrastive_target_bank_labels,
            prior_reconstructions=out.prior_reconstructions,
            prior_values=priors.values,
            prior_missing=priors.missing,
            prior_source_names=priors.source_names,
            contrastive_labels=perturbation_ids,
        )


@torch.no_grad()
def _target_centers_for_batch(
    model: OpenXCellModel,
    batch,
    model_config: ModelConfig,
    prior_store: PriorStore | None,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one stop-gradient ground-truth CLS center per sampled cell set."""
    target_device = torch.device(device)
    x_target = batch.x_target.to(target_device)
    gene_ids = batch.gene_ids.to(target_device)
    perturbation_ids = batch.perturbation_ids.to(target_device)
    full_reveal = torch.ones_like(x_target, dtype=torch.bool)
    priors = (
        prior_store.batch_for_perturbations(
            batch.perturbations,
            model_config,
            device=target_device,
        )
        if prior_store is not None
        else make_zero_priors(x_target.shape[0], model_config, device=target_device)
    )
    # The contrastive target branch is explicitly stop-gradient. Bypass the DDP
    # wrapper because this auxiliary forward has no reducer-backed backward pass.
    target_out = _unwrap_model(model)(
        x_target,
        full_reveal,
        gene_ids,
        perturbation_ids,
        priors,
        reconstruct_priors=False,
    )
    centers = target_out.cell_embedding.mean(dim=1).detach().float()
    if centers.dim() != 2 or centers.shape[0] != perturbation_ids.shape[0]:
        raise RuntimeError("target branch produced malformed contrastive centers")
    if not torch.isfinite(centers).all():
        raise RuntimeError("target branch produced non-finite contrastive centers")
    return centers, perturbation_ids.detach()


def _gather_variable_first_dimension(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather a tensor whose leading dimension may differ across ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    world_size = dist.get_world_size()
    local_size = torch.tensor([tensor.shape[0]], dtype=torch.long, device=tensor.device)
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(gathered_sizes, local_size)
    sizes = [int(size.item()) for size in gathered_sizes]
    maximum = max(sizes)
    if maximum <= 0:
        raise RuntimeError("cannot gather an empty contrastive target bank")
    if tensor.shape[0] < maximum:
        padding = torch.zeros(
            (maximum - tensor.shape[0], *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.cat([tensor, padding], dim=0)
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(
        [rank_tensor[:rank_size] for rank_tensor, rank_size in zip(gathered, sizes)],
        dim=0,
    )


def _aggregate_target_centers_by_label(
    centers: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse repeated sets into one ground-truth center per perturbation."""
    if centers.dim() != 2 or not centers.shape[0] or not centers.shape[1]:
        raise ValueError("contrastive centers must be a non-empty matrix")
    if labels.shape != (centers.shape[0],) or labels.dtype not in {torch.int32, torch.int64}:
        raise ValueError("contrastive center labels must be an integer vector")
    if labels.device != centers.device:
        raise ValueError("contrastive centers and labels must be on the same device")
    if not torch.is_floating_point(centers) or not torch.isfinite(centers).all():
        raise ValueError("contrastive centers must contain finite floating-point values")
    unique_labels, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    center_sums = torch.zeros(
        (unique_labels.shape[0], centers.shape[1]),
        dtype=centers.dtype,
        device=centers.device,
    )
    center_sums.index_add_(0, inverse, centers)
    counts = torch.zeros(
        unique_labels.shape[0],
        dtype=centers.dtype,
        device=centers.device,
    )
    counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=centers.dtype))
    return center_sums / counts[:, None], unique_labels


def _optimizer_step_contrastive_target_bank(
    model: OpenXCellModel,
    batches: list,
    model_config: ModelConfig,
    prior_store: PriorStore | None,
    device: torch.device | str,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one global target-center bank for every set in an optimizer step."""
    if not batches:
        raise ValueError("contrastive target-bank construction requires at least one batch")
    target_device = torch.device(device)
    local_centers: list[torch.Tensor] = []
    local_labels: list[torch.Tensor] = []
    with _autocast_context(target_device, precision):
        for batch in batches:
            centers, labels = _target_centers_for_batch(
                model,
                batch,
                model_config,
                prior_store,
                target_device,
            )
            local_centers.append(centers)
            local_labels.append(labels)
    centers = _gather_variable_first_dimension(torch.cat(local_centers, dim=0))
    labels = _gather_variable_first_dimension(torch.cat(local_labels, dim=0))
    return _aggregate_target_centers_by_label(centers, labels)


def _apply_scgpt(model, var_names, scgpt_init, scgpt_warm_start, gene_id_map,
                 min_overlap, strict_overlap, embedding_only: bool = False) -> dict | None:
    if scgpt_warm_start is not None:
        from open_x_cell.scgpt import warm_start_from_scgpt
        ens2sym = (
            torch.load(gene_id_map, map_location="cpu", weights_only=False)
            if gene_id_map is not None
            else None
        )
        stats = warm_start_from_scgpt(
            model, var_names, scgpt_warm_start, ens2sym=ens2sym,
            min_overlap=min_overlap, strict_overlap=strict_overlap,
            load_transformer=not embedding_only,
            load_value_encoder=False,
        )
        print(f"scGPT warm-start: mapped {stats.n_mapped}/{stats.n_genes} "
              f"({stats.overlap:.1%}; symbol={stats.n_via_symbol}, ensembl={stats.n_via_ensembl}, "
              f"unmapped_random={stats.n_fallback}); transformer tensors loaded={stats.loaded_tensors}")
        for note in stats.notes:
            print(f"  note: {note}")
        return {
            "mode": "warm_start",
            **asdict(stats),
            "overlap": stats.overlap,
            "load_transformer": not embedding_only,
            "load_value_encoder": False,
            "paper_transfer_scope": (
                "gene_embedding_only_table_a3_wording"
                if embedding_only
                else "gene_embedding_norm_and_transformer_appendix_a7"
            ),
        }
    elif scgpt_init is not None:
        table = load_gene_embedding_table(scgpt_init)
        n_init, n_total = init_gene_embedding_from_table(model, var_names, table)
        print(f"scGPT init: {n_init}/{n_total} gene embeddings initialized from {scgpt_init}")
        return {
            "mode": "embedding_table",
            "n_initialized": n_init,
            "n_genes": n_total,
            "overlap": n_init / n_total if n_total else 0.0,
        }
    return None


def _autocast_context(device: torch.device, precision: str):
    if precision == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 autocast is not supported on CPU; use bfloat16 or float32")
    return torch.autocast(device_type=device.type, dtype=dtype)


def _distributed_mean(record: dict[str, float], device: torch.device) -> dict[str, float]:
    if not (dist.is_available() and dist.is_initialized()):
        return record
    keys = sorted(record)
    values = torch.tensor([record[key] for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return {key: float(value) for key, value in zip(keys, values.cpu().tolist())}


def _run_loop(
    model, loader, model_cfg, prior_store, loss_fn, train_cfg,
    *, max_steps: int, learning_rate: float, warmup_fraction: float,
    sampler: DistributedSampler | None = None,
    output_dir: str | Path | None = None,
    var_names: list[str] | None = None,
    resume_payload: dict | None = None,
) -> list[dict[str, float]]:
    """AdamW training with AMP, accumulation, DDP synchronization, and recovery checkpoints."""
    device = torch.device(train_cfg.device)
    accumulation = train_cfg.gradient_accumulation_steps
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2), weight_decay=train_cfg.weight_decay,
    )
    warmup_steps = int(max_steps * warmup_fraction)
    scheduler = make_warmup_cosine_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=max_steps)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=train_cfg.precision == "float16" and device.type == "cuda"
    )
    source_sha256 = _source_fingerprints()

    def training_state(completed_steps: int) -> dict:
        return {
            "completed_steps": completed_steps,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": history,
            "training_config": asdict(train_cfg),
            "source_sha256": source_sha256,
        }

    start_step = 0
    history: list[dict[str, float]] = []
    if resume_payload is not None:
        state, start_step, history = _validate_resume_training_state(
            resume_payload,
            train_cfg,
            max_steps=max_steps,
            source_sha256=source_sha256,
            expected_optimizer_parameters=sum(1 for _ in model.parameters()),
            expected_optimizer_active_ids={
                index for index, parameter in enumerate(model.parameters()) if parameter.requires_grad
            },
            expected_optimizer_parameter_shapes=[
                tuple(parameter.shape) for parameter in model.parameters()
            ],
        )
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        if state.get("scaler_state"):
            scaler.load_state_dict(state["scaler_state"])

    completed_micro_steps = start_step * accumulation
    epoch = completed_micro_steps // max(1, len(loader))
    micro_offset = completed_micro_steps % max(1, len(loader))
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)
    if sampler is not None:
        sampler.set_epoch(epoch)
    iterator = iter(loader)
    # A periodic checkpoint can land mid-epoch. Restore the dataloader cursor so
    # resuming does not silently repeat the beginning of that epoch.
    for _ in range(micro_offset):
        try:
            next(iterator)
        except StopIteration:  # defensive for unusual iterable loaders
            epoch += 1
            if hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(epoch)
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(loader)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    elapsed_offset = float(history[-1].get("elapsed_seconds", 0.0)) if history else 0.0
    started = time.monotonic()
    for step in range(start_step, max_steps):
        update_learning_rate = float(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        step_batches = []
        for micro_step in range(accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                if hasattr(loader.dataset, "set_epoch"):
                    loader.dataset.set_epoch(epoch)
                if sampler is not None:
                    sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)

            step_batches.append(batch)

        contrastive_target_bank = None
        contrastive_target_bank_labels = None
        if loss_fn.weights.contrastive:
            (
                contrastive_target_bank,
                contrastive_target_bank_labels,
            ) = _optimizer_step_contrastive_target_bank(
                model,
                step_batches,
                model_cfg,
                prior_store,
                device,
                train_cfg.precision,
            )

        for micro_step, batch in enumerate(step_batches):
            global_micro_step = step * accumulation + micro_step
            synchronize = micro_step == accumulation - 1
            sync_context = (
                nullcontext()
                if synchronize or not isinstance(model, DistributedDataParallel)
                else model.no_sync()
            )
            with sync_context:
                with _autocast_context(device, train_cfg.precision):
                    loss_output = compute_batch_loss(
                        model=model,
                        batch=batch,
                        model_config=model_cfg,
                        loss_fn=loss_fn,
                        prior_store=prior_store,
                        step=global_micro_step,
                        seed=train_cfg.seed,
                        device=device,
                        contrastive_target_bank=contrastive_target_bank,
                        contrastive_target_bank_labels=contrastive_target_bank_labels,
                    )
                    loss = loss_output.total / accumulation
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss during training at optimizer step {step}")
                scaler.scale(loss).backward()

            for name, value in loss_output.terms.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach().cpu()) / accumulation
            sums["total"] = sums.get("total", 0.0) + float(loss_output.total.detach().cpu()) / accumulation

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            train_cfg.grad_clip_norm,
            error_if_nonfinite=True,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        sums["grad_norm"] = float(grad_norm.detach().cpu())
        sums = _distributed_mean(sums, device)
        sums["step"] = float(step + 1)
        # Record the LR that actually produced this update, not the scheduler's
        # post-step value prepared for the next update.
        sums["lr"] = update_learning_rate
        sums["elapsed_seconds"] = elapsed_offset + float(time.monotonic() - started)
        history.append(sums)

        should_log = (step + 1) % train_cfg.log_every == 0 or step == start_step
        if rank == 0 and should_log:
            memory = (
                torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
            )
            print(
                f"step={step + 1}/{max_steps} total={sums['total']:.5f} "
                f"lr={sums['lr']:.3e} grad={sums['grad_norm']:.3f} "
                f"epoch~={epoch} max_mem_gib={memory:.2f}",
                flush=True,
            )

        should_checkpoint = (
            output_dir is not None
            and train_cfg.checkpoint_every > 0
            and (step + 1) % train_cfg.checkpoint_every == 0
        )
        if should_checkpoint and rank == 0:
            save_checkpoint(
                Path(output_dir) / "latest.pt",
                model,
                model_cfg,
                var_names or [],
                training_state=training_state(step + 1),
            )
        if should_checkpoint and dist.is_available() and dist.is_initialized():
            dist.barrier()

    if output_dir is not None and rank == 0:
        save_checkpoint(
            Path(output_dir) / "latest.pt",
            model,
            model_cfg,
            var_names or [],
            training_state=training_state(max_steps),
        )
    if output_dir is not None and dist.is_available() and dist.is_initialized():
        dist.barrier()
    return history


def _training_config_with_overrides(
    training_config: TrainingConfig | None,
    max_steps: int | None,
    batch_size: int | None = None,
    set_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    device: str | None = None,
    precision: str | None = None,
    gradient_accumulation_steps: int | None = None,
    gradient_checkpointing: bool | None = None,
    num_workers: int | None = None,
    checkpoint_every: int | None = None,
    log_every: int | None = None,
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
    if precision is not None:
        updates["precision"] = precision
    if gradient_accumulation_steps is not None:
        updates["gradient_accumulation_steps"] = gradient_accumulation_steps
    if gradient_checkpointing is not None:
        updates["gradient_checkpointing"] = gradient_checkpointing
    if num_workers is not None:
        updates["num_workers"] = num_workers
    if checkpoint_every is not None:
        updates["checkpoint_every"] = checkpoint_every
    if log_every is not None:
        updates["log_every"] = log_every
    return replace(cfg, **updates)


def _initialize_distributed(device_name: str) -> tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    initialized_here = False
    requested = torch.device(device_name)
    if world_size > 1 and requested.type == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = requested
        if device.type == "cuda":
            if device.index is None:
                device = torch.device("cuda", 0)
            torch.cuda.set_device(device)
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if requested.type == "cuda" else "gloo"
        if device.type == "cuda":
            dist.init_process_group(backend=backend, device_id=device)
        else:
            dist.init_process_group(backend=backend)
        initialized_here = True
        # Training validation can fail after process-group initialization (for
        # example while loading a malformed prior).  Ensure every rank tears
        # down NCCL/Gloo even when an exception bypasses the normal return path.
        atexit.register(_destroy_distributed_if_initialized)
    return device, rank, world_size, initialized_here


def _destroy_distributed_if_initialized() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


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
    precision: str | None = None,
    gradient_accumulation_steps: int | None = None,
    gradient_checkpointing: bool | None = None,
    num_workers: int | None = None,
    checkpoint_every: int | None = None,
    log_every: int | None = None,
    epochs: int | None = None,
    resume_from: str | Path | None = None,
    pretrained_checkpoint: str | Path | None = None,
    vocabulary_file: str | Path | None = None,
    scgpt_embedding_only: bool = False,
    backed: bool = False,
    training_context_length: int | None = None,
) -> TrainingResult:
    if max_steps is not None and epochs is not None:
        raise ValueError("max_steps and epochs are mutually exclusive")
    if not 0 <= scgpt_min_overlap <= 1:
        raise ValueError("scgpt_min_overlap must satisfy 0 <= value <= 1")
    if scgpt_init is not None and scgpt_warm_start is not None:
        raise ValueError("scgpt_init and scgpt_warm_start are mutually exclusive")
    if scgpt_embedding_only and scgpt_warm_start is None:
        raise ValueError("scgpt_embedding_only requires scgpt_warm_start")
    if not isinstance(backed, bool):
        raise ValueError("backed must be a bool")
    if training_context_length is not None and (
        isinstance(training_context_length, bool)
        or not isinstance(training_context_length, Integral)
        or training_context_length <= 0
    ):
        raise ValueError("training_context_length must be a positive integer when provided")
    if backed and normalize != "none":
        raise ValueError("backed training requires normalize='none'")
    initialization_inputs = [
        resume_from is not None,
        pretrained_checkpoint is not None,
        scgpt_init is not None,
        scgpt_warm_start is not None,
    ]
    if sum(initialization_inputs) > 1:
        raise ValueError(
            "resume, pretrained_checkpoint, scgpt_init, and scgpt_warm_start are "
            "mutually exclusive initialization modes"
        )
    train_cfg = _training_config_with_overrides(
        training_config,
        max_steps=max_steps,
        batch_size=batch_size,
        set_size=set_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        precision=precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        num_workers=num_workers,
        checkpoint_every=checkpoint_every,
        log_every=log_every,
    )
    runtime_device, rank, world_size, initialized_here = _initialize_distributed(train_cfg.device)
    train_cfg = replace(train_cfg, device=str(runtime_device))
    output = Path(output_dir)
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"training output path exists and is not a directory: {output}")
    if resume_from is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"training output directory is not empty: {output}; choose a fresh directory or --resume"
        )
    existing_manifest: dict | None = None
    if resume_from is not None:
        resume_path = Path(resume_from).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        if output.resolve() != resume_path.parent:
            raise ValueError("resume output directory must be the resume checkpoint's parent directory")
        manifest_path = output / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"exact resume requires the original run manifest: {manifest_path}"
            )
        with manifest_path.open(encoding="utf-8") as handle:
            loaded_manifest = json.load(handle)
        if not isinstance(loaded_manifest, dict):
            raise ValueError("original run manifest must be a mapping")
        existing_manifest = loaded_manifest
        if existing_manifest.get("source_sha256") != _source_fingerprints():
            raise ValueError("original run manifest was created by different package sources")
        if existing_manifest.get("model_size") != model_size:
            raise ValueError("resume model size differs from the original run manifest")
        if existing_manifest.get("normalization") != normalize:
            raise ValueError("resume normalization differs from the original run manifest")
        if bool(existing_manifest.get("backed", False)) != backed:
            raise ValueError("resume backed mode differs from the original run manifest")
        if existing_manifest.get("requested_training_context_length") != training_context_length:
            raise ValueError("resume training context request differs from the original run manifest")
        if existing_manifest.get("epochs") != epochs:
            raise ValueError("resume epoch configuration differs from the original run manifest")
        if Path(str(existing_manifest.get("adata", ""))).resolve() != Path(adata_path).resolve():
            raise ValueError("resume AnnData path differs from the original run manifest")
        original_artifacts = existing_manifest.get("input_artifacts")
        if not isinstance(original_artifacts, dict) or "training_adata" not in original_artifacts:
            raise ValueError("original run manifest lacks hashed training inputs")
        for artifact_name, artifact_record in original_artifacts.items():
            _verify_artifact_record(artifact_record, artifact_name)
    torch.manual_seed(train_cfg.seed)
    if runtime_device.type == "cuda":
        torch.cuda.manual_seed_all(train_cfg.seed)
        if train_cfg.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected CUDA device does not support bfloat16")
    adata = load_adata(str(adata_path), normalize=normalize, backed=backed)
    model_cfg = model_config(model_size)
    if adata.n_vars > model_cfg.num_genes:
        raise ValueError(
            f"training AnnData has {adata.n_vars} genes but {model_size} supports at most "
            f"{model_cfg.num_genes}; prepare an explicit protein-coding panel instead of truncating"
        )
    vocabulary = _load_vocabulary(vocabulary_file, adata.var_names)
    if len(vocabulary) > model_cfg.num_genes:
        raise ValueError(
            f"vocabulary has {len(vocabulary)} genes but {model_size} supports {model_cfg.num_genes}"
        )
    missing_measured = sorted(set(map(str, adata.var_names)) - set(vocabulary))
    if missing_measured:
        raise ValueError(f"{len(missing_measured)} measured genes are absent from the model vocabulary")
    if "perturbation" not in adata.obs:
        raise ValueError("training AnnData must contain obs['perturbation']")
    control_mask = boolean_obs_mask(adata, "is_control")
    perturbations = set(adata.obs.loc[~control_mask, "perturbation"].astype(str))
    missing_perturbations = sorted(perturbations - set(vocabulary))
    if missing_perturbations:
        raise ValueError(
            f"{len(missing_perturbations)} perturbations are absent from the model vocabulary; "
            f"first missing: {missing_perturbations[:5]}"
        )
    if training_context_length is None:
        resolved_training_context_length = min(model_cfg.context_length, adata.n_vars)
    else:
        resolved_training_context_length = int(training_context_length)
        if resolved_training_context_length > adata.n_vars:
            raise ValueError("training_context_length cannot exceed measured AnnData genes")
        if resolved_training_context_length > model_cfg.context_length:
            raise ValueError("training_context_length cannot exceed the model context length")
    if backed and train_cfg.num_workers != 0:
        raise ValueError("backed training currently requires num_workers=0")
    dataset = OpenXCellDataset(
        adata,
        set_size=train_cfg.set_size,
        context_length=resolved_training_context_length,
        seed=train_cfg.seed,
        vocabulary=vocabulary,
    )
    # Use the epoch-addressable sampler even for one process.  Besides matching
    # DDP behavior, this makes the dataloader cursor exactly reconstructible
    # from completed_steps when a checkpoint is resumed mid-epoch.
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=train_cfg.seed,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=collate_samples,
        num_workers=train_cfg.num_workers,
        pin_memory=runtime_device.type == "cuda",
        # Workers are recreated at epoch boundaries so the epoch-resampled
        # dataset state is propagated deterministically to each worker.
        persistent_workers=False,
    )

    epoch_schedule = None
    if epochs is not None:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        epoch_micro_steps = len(loader)
        derived_steps = math.ceil(epochs * epoch_micro_steps / train_cfg.gradient_accumulation_steps)
        train_cfg = replace(train_cfg, max_steps=derived_steps)
        consumed_micro_steps = derived_steps * train_cfg.gradient_accumulation_steps
        epoch_schedule = {
            "microbatches_per_rank_per_epoch": epoch_micro_steps,
            "requested_epochs": epochs,
            "scheduled_microbatches_per_rank": epochs * epoch_micro_steps,
            "consumed_microbatches_per_rank": consumed_micro_steps,
            "optimizer_rounding_extra_microbatches_per_rank": (
                consumed_micro_steps - epochs * epoch_micro_steps
            ),
            "effective_epochs_after_full_accumulation_rounding": (
                consumed_micro_steps / epoch_micro_steps
            ),
        }

    resume_payload = (
        torch.load(resume_from, map_location="cpu", weights_only=False)
        if resume_from is not None
        else None
    )
    if resume_payload is not None:
        if not isinstance(resume_payload, dict):
            raise ValueError("resume checkpoint payload must be a mapping")
        model_state = resume_payload.get("model_state")
        if not isinstance(model_state, dict) or not model_state:
            raise ValueError("resume checkpoint model state is empty")
        for name, tensor in model_state.items():
            if not torch.is_tensor(tensor):
                raise ValueError(f"resume checkpoint model entry is not a tensor: {name}")
            if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
                raise ValueError(f"resume checkpoint model tensor is non-finite: {name}")
    model = OpenXCellModel(model_cfg)
    resolved_weights = loss_weights or LossConfig(
        l1_output=1.0 if model_size == "mini" else 0.0
    )
    if not resolved_weights.prior_reconstruction:
        model.prior_projector.decoders.requires_grad_(False)
    scgpt_initialization_stats = None
    pretrained_initialization_stats = None
    if resume_payload is not None:
        checkpoint_cfg = _config_from_dict(resume_payload["model_config"])
        if checkpoint_cfg != model_cfg:
            raise ValueError("resume checkpoint architecture does not match requested model size")
        if list(resume_payload.get("var_names", [])) != vocabulary:
            raise ValueError("resume checkpoint gene vocabulary does not match the training AnnData")
        model.load_state_dict(resume_payload["model_state"])
        # Validate optimizer/scheduler/history state before recording a resume
        # event or otherwise mutating the original run directory.
        _validate_resume_training_state(
            resume_payload,
            train_cfg,
            max_steps=train_cfg.max_steps,
            source_sha256=_source_fingerprints(),
            expected_optimizer_parameters=sum(1 for _ in model.parameters()),
            expected_optimizer_active_ids={
                index for index, parameter in enumerate(model.parameters()) if parameter.requires_grad
            },
            expected_optimizer_parameter_shapes=[
                tuple(parameter.shape) for parameter in model.parameters()
            ],
        )
    elif pretrained_checkpoint is not None and rank == 0:
        pretrained_initialization_stats = _initialize_from_xcell_checkpoint(
            model,
            pretrained_checkpoint,
            vocabulary,
            model_cfg,
        )
    elif rank == 0:
        scgpt_initialization_stats = _apply_scgpt(
            model,
            vocabulary,
            scgpt_init,
            scgpt_warm_start,
            gene_id_map,
            scgpt_min_overlap,
            scgpt_strict_overlap,
            embedding_only=scgpt_embedding_only,
        )
    model.set_gradient_checkpointing(train_cfg.gradient_checkpointing)
    model.to(runtime_device)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[runtime_device.index] if runtime_device.type == "cuda" else None,
            output_device=runtime_device.index if runtime_device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    loss_fn = OpenXCellLoss(resolved_weights)
    resolved_prior_files = {
        key: str(Path(value).resolve()) if value is not None else None
        for key, value in (prior_files or {}).items()
    }
    if existing_manifest is not None:
        expected_vocabulary = existing_manifest.get("vocabulary")
        current_vocabulary = (
            str(Path(vocabulary_file).resolve()) if vocabulary_file is not None else None
        )
        if current_vocabulary != expected_vocabulary:
            raise ValueError("resume vocabulary path differs from the original run manifest")
        if resolved_prior_files != existing_manifest.get("prior_files"):
            raise ValueError("resume prior files differ from the original run manifest")
        runtime_training_config = asdict(train_cfg)
        original_training_config = dict(existing_manifest.get("training_config", {}))
        runtime_training_config.pop("device", None)
        original_training_config.pop("device", None)
        expected_resume_fields = {
            "shape": [int(adata.n_obs), int(adata.n_vars)],
            "training_context_length": int(dataset.context_length),
            "requested_training_context_length": training_context_length,
            "backed": backed,
            "vocabulary_size": len(vocabulary),
            "model_config": _json_normalized(asdict(model_cfg)),
            "training_config": runtime_training_config,
            "epoch_schedule": epoch_schedule,
            "loss_config": asdict(resolved_weights),
            "world_size": world_size,
            "effective_sets_per_optimizer_step": (
                world_size * train_cfg.batch_size * train_cfg.gradient_accumulation_steps
            ),
            "effective_cells_per_optimizer_step": (
                world_size
                * train_cfg.batch_size
                * train_cfg.gradient_accumulation_steps
                * train_cfg.set_size
            ),
            "parameters": sum(p.numel() for p in _unwrap_model(model).parameters()),
            "trainable_parameters": sum(
                p.numel() for p in _unwrap_model(model).parameters() if p.requires_grad
            ),
        }
        resume_mismatches = {}
        for key, expected_value in expected_resume_fields.items():
            actual_value = existing_manifest.get(key)
            if key == "training_config":
                actual_value = original_training_config
            if actual_value != expected_value:
                resume_mismatches[key] = {
                    "expected_from_runtime": expected_value,
                    "original_manifest": actual_value,
                }
        if resume_mismatches:
            raise ValueError(f"resume runtime differs from original run: {resume_mismatches}")
        original_artifacts = existing_manifest["input_artifacts"]
        if Path(original_artifacts["training_adata"]["path"]).resolve() != Path(
            adata_path
        ).resolve():
            raise ValueError("resume training artifact differs from the requested AnnData")
        if vocabulary_file is None:
            if "vocabulary" in original_artifacts:
                raise ValueError("original run recorded a vocabulary file but resume omitted it")
        elif Path(original_artifacts.get("vocabulary", {}).get("path", "")).resolve() != Path(
            vocabulary_file
        ).resolve():
            raise ValueError("resume vocabulary artifact differs from the requested file")
        for source, prior_path in resolved_prior_files.items():
            artifact_name = f"prior_{source}"
            if prior_path is None:
                if artifact_name in original_artifacts:
                    raise ValueError(f"resume omitted the original {source} prior")
            elif Path(original_artifacts.get(artifact_name, {}).get("path", "")).resolve() != Path(
                prior_path
            ).resolve():
                raise ValueError(f"resume {source} prior differs from the original input")
    prior_store = PriorStore.from_files(prior_files or {}, model_cfg)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if existing_manifest is not None:
            manifest = dict(existing_manifest)
            resume_events = list(manifest.get("resume_events", []))
            resume_events.append(
                _retain_resume_checkpoint(
                    resume_from,
                    output,
                    int(resume_payload["training_state"]["completed_steps"]),
                )
            )
            manifest["resume_events"] = resume_events
        else:
            input_artifacts: dict[str, dict[str, str | int]] = {
                "training_adata": _artifact_record(adata_path),
            }
            if vocabulary_file is not None:
                input_artifacts["vocabulary"] = _artifact_record(vocabulary_file)
            if gene_id_map is not None:
                input_artifacts["gene_id_map"] = _artifact_record(gene_id_map)
            if scgpt_init is not None:
                input_artifacts["scgpt_init"] = _artifact_record(scgpt_init)
            if scgpt_warm_start is not None:
                scgpt_dir = Path(scgpt_warm_start)
                input_artifacts["scgpt_vocab"] = _artifact_record(scgpt_dir / "vocab.json")
                input_artifacts["scgpt_checkpoint"] = _artifact_record(
                    scgpt_dir / "best_model.pt"
                )
            if pretrained_checkpoint is not None:
                input_artifacts["pretrained_checkpoint"] = _artifact_record(
                    pretrained_checkpoint
                )
            for source, prior_path in resolved_prior_files.items():
                if prior_path is not None:
                    input_artifacts[f"prior_{source}"] = _artifact_record(prior_path)
            manifest = {
                "adata": str(Path(adata_path).resolve()),
                "shape": [int(adata.n_obs), int(adata.n_vars)],
                "training_context_length": int(dataset.context_length),
                "requested_training_context_length": training_context_length,
                "backed": backed,
                "vocabulary": str(Path(vocabulary_file).resolve()) if vocabulary_file is not None else None,
                "vocabulary_size": len(vocabulary),
                "model_size": model_size,
                "epochs": epochs,
                "normalization": normalize,
                "model_config": asdict(model_cfg),
                "training_config": asdict(train_cfg),
                "epoch_schedule": epoch_schedule,
                "loss_config": asdict(resolved_weights),
                "world_size": world_size,
                "effective_sets_per_optimizer_step": (
                    world_size * train_cfg.batch_size * train_cfg.gradient_accumulation_steps
                ),
                "effective_cells_per_optimizer_step": (
                    world_size
                    * train_cfg.batch_size
                    * train_cfg.gradient_accumulation_steps
                    * train_cfg.set_size
                ),
                "parameters": sum(p.numel() for p in _unwrap_model(model).parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in _unwrap_model(model).parameters() if p.requires_grad
                ),
                "prior_files": resolved_prior_files,
                "scgpt_init": str(Path(scgpt_init).resolve()) if scgpt_init is not None else None,
                "scgpt_warm_start": str(Path(scgpt_warm_start).resolve()) if scgpt_warm_start is not None else None,
                "scgpt_embedding_only": scgpt_embedding_only,
                "scgpt_min_overlap": scgpt_min_overlap,
                "scgpt_strict_overlap": scgpt_strict_overlap,
                "scgpt_initialization_stats": scgpt_initialization_stats,
                "pretrained_checkpoint": (
                    str(Path(pretrained_checkpoint).resolve())
                    if pretrained_checkpoint is not None
                    else None
                ),
                "pretrained_initialization_stats": pretrained_initialization_stats,
                "resume_from": None,
                "resume_events": [],
                "loss_semantics": {
                    "mmd": (
                        "GeomLoss 0.2.6 tensorized energy: E_xy - 0.5*E_xx - "
                        "0.5*E_yy with squared-distance clamp_min(1e-8)"
                    ),
                    "prior_reconstruction": (
                        "sum of per-source mean-squared errors, globally normalized over "
                        "available samples across DDP ranks"
                    ),
                    "contrastive": (
                        "temperature-0.1 InfoNCE over per-cell predicted CLS embeddings; one "
                        "ground-truth per-cell CLS center is averaged per perturbation from every "
                        "gradient-accumulation microbatch and DDP rank in the optimizer step; "
                        "all other perturbation centers are negatives"
                    ),
                    "learning_rate_schedule": (
                        "one-indexed 10%-warmup then zero-based cosine update samples; every "
                        "scheduled optimizer update uses a non-zero LR and the post-horizon "
                        "scheduler state is zero"
                    ),
                },
                "source_sha256": _source_fingerprints(),
                "input_artifacts": input_artifacts,
            }
        _atomic_json(output / "run_manifest.json", manifest)
        print(
            f"training {model_size} on {world_size} rank(s): {len(dataset)} sets, "
            f"{train_cfg.max_steps} optimizer steps, effective "
            f"{manifest['effective_sets_per_optimizer_step']} sets / "
            f"{manifest['effective_cells_per_optimizer_step']} cells per step",
            flush=True,
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    history = _run_loop(
        model, loader, model_cfg, prior_store, loss_fn, train_cfg,
        max_steps=train_cfg.max_steps, learning_rate=train_cfg.learning_rate,
        warmup_fraction=train_cfg.warmup_fraction,
        sampler=sampler,
        output_dir=output,
        var_names=vocabulary,
        resume_payload=resume_payload,
    )
    checkpoint = output / "checkpoint.pt"
    if rank == 0:
        save_checkpoint(checkpoint, model, model_cfg, vocabulary)
        _atomic_json(output / "history.json", history)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if initialized_here:
        _destroy_distributed_if_initialized()
    if adata.isbacked:
        adata.file.close()
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

    if not 0 < top_frac <= 1:
        raise ValueError("top_frac must satisfy 0 < top_frac <= 1")
    matrix = dense_matrix(adata)
    labels = adata.obs["perturbation"].astype(str).to_numpy()
    control_key = "is_control"
    control_mask = boolean_obs_mask(adata, control_key)
    if not control_mask.any():
        raise ValueError("high-effect ranking requires at least one control cell")
    control_mean = matrix[control_mask].mean(axis=0)
    perts = sorted(set(labels[~control_mask]))
    if not perts:
        raise ValueError("high-effect ranking requires at least one perturbation")
    effects = {
        p: float(np.linalg.norm(matrix[(labels == p) & ~control_mask].mean(axis=0) - control_mean))
        for p in perts
    }
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
    batch_size: int | None = None,
    set_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    precision: str | None = None,
    gradient_accumulation_steps: int | None = None,
    gradient_checkpointing: bool | None = None,
    num_workers: int | None = None,
    log_every: int | None = None,
    vocabulary_file: str | Path | None = None,
    scgpt_embedding_only: bool = False,
) -> TrainingResult:
    """Two-phase curriculum (paper Ultra, §4.x):

    * Phase 1 — high-effect initialization: train on the top ``top_frac`` highest-effect
      perturbations only, keeping all loss terms.
    * Phase 2 — full corpus: continue on every perturbation with a FRESH optimizer (its
      own LR / warmup), with the per-gene CCC loss term dropped (paper: full-dataset
      training removes per-gene CCC).
    """
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("the clean-room curriculum trainer is single-process only")
    if phase1_steps <= 0 or phase2_steps <= 0:
        raise ValueError("phase1_steps and phase2_steps must be positive")
    if not 0 < top_frac <= 1:
        raise ValueError("top_frac must satisfy 0 < top_frac <= 1")
    if phase2_learning_rate is not None and phase2_learning_rate <= 0:
        raise ValueError("phase2_learning_rate must be positive")
    if phase2_warmup_fraction is not None and not 0 <= phase2_warmup_fraction <= 1:
        raise ValueError("phase2_warmup_fraction must satisfy 0 <= value <= 1")
    if scgpt_init is not None and scgpt_warm_start is not None:
        raise ValueError("scgpt_init and scgpt_warm_start are mutually exclusive")
    if scgpt_embedding_only and scgpt_warm_start is None:
        raise ValueError("scgpt_embedding_only requires scgpt_warm_start")
    output = Path(output_dir)
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"curriculum output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"curriculum output directory is not empty: {output}")
    train_cfg = _training_config_with_overrides(
        training_config,
        max_steps=None,
        batch_size=batch_size,
        set_size=set_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        precision=precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        num_workers=num_workers,
        log_every=log_every,
    )
    torch.manual_seed(train_cfg.seed)
    adata = load_adata(str(adata_path), normalize=normalize)
    model_cfg = model_config(model_size)
    if adata.n_vars > model_cfg.num_genes:
        raise ValueError(
            f"training AnnData has {adata.n_vars} genes but {model_size} supports "
            f"{model_cfg.num_genes}; prepare an explicit panel instead of truncating"
        )

    vocabulary = _load_vocabulary(vocabulary_file, adata.var_names)
    if len(vocabulary) > model_cfg.num_genes:
        raise ValueError("curriculum vocabulary exceeds the model embedding table")
    if set(map(str, adata.var_names)) - set(vocabulary):
        raise ValueError("curriculum vocabulary omits one or more measured genes")
    controls = boolean_obs_mask(adata, "is_control")
    perturbations = set(adata.obs.loc[~controls, "perturbation"].astype(str))
    missing_perturbations = perturbations - set(vocabulary)
    if missing_perturbations:
        raise ValueError(
            f"curriculum vocabulary omits perturbations: {sorted(missing_perturbations)[:5]}"
        )

    model = OpenXCellModel(model_cfg).to(train_cfg.device)
    model.set_gradient_checkpointing(train_cfg.gradient_checkpointing)
    scgpt_initialization_stats = _apply_scgpt(
        model,
        vocabulary,
        scgpt_init,
        scgpt_warm_start,
        gene_id_map,
        scgpt_min_overlap,
        scgpt_strict_overlap,
        embedding_only=scgpt_embedding_only,
    )
    resolved_phase1_weights = loss_weights or LossConfig(
        l1_output=1.0 if model_size == "mini" else 0.0
    )
    if not resolved_phase1_weights.prior_reconstruction:
        model.prior_projector.decoders.requires_grad_(False)
    loss_fn = OpenXCellLoss(resolved_phase1_weights)
    prior_store = PriorStore.from_files(prior_files or {}, model_cfg)

    def _loader(ad):
        ds = OpenXCellDataset(ad, set_size=train_cfg.set_size,
                              context_length=model_cfg.context_length, seed=train_cfg.seed,
                              vocabulary=vocabulary)
        sampler = DistributedSampler(
            ds, num_replicas=1, rank=0, shuffle=True, seed=train_cfg.seed, drop_last=False
        )
        loader = DataLoader(
            ds,
            batch_size=train_cfg.batch_size,
            sampler=sampler,
            collate_fn=collate_samples,
            num_workers=train_cfg.num_workers,
            pin_memory=torch.device(train_cfg.device).type == "cuda",
            persistent_workers=False,
        )
        return loader, sampler

    # Phase 1 (high-effect initialization): top-`top_frac` high-effect perturbations,
    # ALL losses (paper: high-effect curriculum keeps every loss term).
    selected = high_effect_perturbations(adata, top_frac=top_frac)
    labels = adata.obs["perturbation"].astype(str).to_numpy()
    keep = boolean_obs_mask(adata, "is_control") | np.isin(labels, selected)
    adata1 = adata[keep].copy()
    print(f"curriculum phase 1 (high-effect init): {len(selected)} perturbations "
          f"(top {top_frac:.0%}): {selected}")
    phase1_loader, phase1_sampler = _loader(adata1)
    phase1_history = _run_loop(
        model, phase1_loader, model_cfg, prior_store, loss_fn, train_cfg,
        max_steps=phase1_steps, learning_rate=train_cfg.learning_rate,
        warmup_fraction=train_cfg.warmup_fraction,
        sampler=phase1_sampler,
    )

    # Phase 2 (full corpus): fresh optimizer (own LR / warmup), per-gene CCC removed
    # (paper: full-dataset training drops the per-gene CCC term).
    phase2_weights = replace(resolved_phase1_weights, per_gene_ccc=0.0)
    loss_fn2 = OpenXCellLoss(phase2_weights)
    print("curriculum phase 2: full corpus (per-gene CCC dropped)")
    phase2_loader, phase2_sampler = _loader(adata)
    phase2_history = _run_loop(
        model, phase2_loader, model_cfg, prior_store, loss_fn2, train_cfg,
        max_steps=phase2_steps,
        learning_rate=phase2_learning_rate if phase2_learning_rate is not None else train_cfg.learning_rate,
        warmup_fraction=phase2_warmup_fraction if phase2_warmup_fraction is not None else train_cfg.warmup_fraction,
        sampler=phase2_sampler,
    )

    history: list[dict[str, float]] = []
    for phase, rows in ((1, phase1_history), (2, phase2_history)):
        for row in rows:
            enriched = dict(row)
            enriched["phase"] = float(phase)
            enriched["phase_step"] = float(row["step"])
            enriched["step"] = float(len(history) + 1)
            history.append(enriched)

    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    save_checkpoint(checkpoint, model, model_cfg, vocabulary)
    _atomic_json(output / "history.json", history)
    _atomic_json(
        output / "curriculum_manifest.json",
        {
            "adata": str(Path(adata_path).resolve()),
            "model_size": model_size,
            "model_config": asdict(model_cfg),
            "training_config": asdict(train_cfg),
            "phase1_steps": phase1_steps,
            "phase2_steps": phase2_steps,
            "top_frac": top_frac,
            "phase1_loss_config": asdict(resolved_phase1_weights),
            "phase2_loss_config": asdict(phase2_weights),
            "vocabulary_size": len(vocabulary),
            "scgpt_initialization_stats": scgpt_initialization_stats,
            "source_sha256": _source_fingerprints(),
        },
    )
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
