from __future__ import annotations

import hashlib
import json
import re
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch
from anndata import AnnData, read_h5ad

from open_x_cell.data import boolean_obs_mask
from open_x_cell.diffusion import cumulative_reveal_steps
from open_x_cell.priors import PRIOR_SOURCES, PriorStore, make_zero_priors
from open_x_cell.training import load_checkpoint


def _sha256(path: str | Path) -> str:
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
        "sha256": _sha256(source),
    }


def _source_fingerprints() -> dict[str, str]:
    package_dir = Path(__file__).resolve().parent
    return {path.name: _sha256(path) for path in sorted(package_dir.glob("*.py"))}


def _inference_inputs(
    checkpoint_path: str | Path,
    adata_path: str | Path,
    prior_files: dict[str, str | Path | None] | None,
    perturbations_path: str | Path | None = None,
) -> dict:
    files = prior_files or {}
    unknown = set(files) - set(PRIOR_SOURCES)
    if unknown:
        raise ValueError(f"unknown prior sources: {sorted(unknown)}")
    return {
        "checkpoint": _artifact_record(checkpoint_path),
        "source_adata": _artifact_record(adata_path),
        "perturbations_file": (
            _artifact_record(perturbations_path) if perturbations_path is not None else None
        ),
        "prior_artifacts": {
            source: _artifact_record(files[source])
            if source in files and files[source] is not None
            else None
            for source in PRIOR_SOURCES
        },
        "source_sha256": _source_fingerprints(),
    }


def _canonical_json(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_output_target(
    target: Path,
    protected_inputs: set[Path],
    temporary: Path,
) -> None:
    collisions = {target, temporary} & protected_inputs
    if collisions:
        paths = ", ".join(map(str, sorted(collisions)))
        raise ValueError(f"output and temporary paths must not overwrite an input: {paths}")


def _select_control_pool(
    adata: AnnData,
    control_rows: np.ndarray,
    context: str | None,
    batch: str | None,
    min_size: int = 1,
) -> np.ndarray:
    """Context-aware NTC pool: (context, batch) -> context -> all.

    A narrower pool is used only when it can fill a complete set.  This mirrors
    Appendix A.10.1 and avoids silently replicating a tiny batch pool when a
    sufficient context-level pool exists.
    """
    if isinstance(min_size, bool) or not isinstance(min_size, Integral) or min_size <= 0:
        raise ValueError("min_size must be a positive integer")
    pool = control_rows
    if context is not None:
        if "context" not in adata.obs:
            raise ValueError("a context was requested but AnnData.obs has no 'context' column")
        ctx = adata.obs["context"].astype(str).to_numpy()
        pool = pool[ctx[pool] == str(context)]
        if not len(pool):
            raise ValueError(f"no control cells are available for context {context!r}")
    if batch is not None and "batch" not in adata.obs:
        raise ValueError("a batch was requested but AnnData.obs has no 'batch' column")
    if batch is not None:
        bat = adata.obs["batch"].astype(str).to_numpy()
        batch_pool = pool[bat[pool] == str(batch)]
        if len(batch_pool) >= min_size:
            return batch_pool
    return pool


def _autocast_context(device: torch.device, precision: str):
    if precision == "float32":
        return torch.autocast(device_type=device.type, enabled=False)
    if precision not in {"bfloat16", "float16"}:
        raise ValueError("precision must be float32, bfloat16, or float16")
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 autocast is not supported on CPU; use bfloat16 or float32")
    return torch.autocast(device_type=device.type, dtype=dtype)


def _aligned_gene_indices(
    adata: AnnData,
    checkpoint_var_names: list[str],
    config,
    perturbation: str | None,
    max_genes: int | None,
) -> tuple[np.ndarray, np.ndarray, list[str], int | None]:
    """Align an inference AnnData to checkpoint vocabulary rows.

    The paper trains on random G' panels and performs full-context inference.  We
    therefore use every checkpoint gene found in the input (optionally capped),
    while keeping global checkpoint gene IDs for the embedding lookup.
    """
    input_names = list(map(str, adata.var_names))
    if len(input_names) != len(set(input_names)):
        raise ValueError("inference AnnData gene names must be unique")
    names = list(map(str, checkpoint_var_names)) or input_names[: config.num_genes]
    if len(names) != len(set(names)):
        raise ValueError("checkpoint gene vocabulary must be unique")
    alias_columns = ("gene_name", "symbol", "gene_symbol", "gene_id", "gene_ids", "ensembl_id")
    input_lookup = {name: idx for idx, name in enumerate(input_names)}
    for column in alias_columns:
        if column in adata.var:
            for row, alias in enumerate(adata.var[column].astype(str)):
                input_lookup.setdefault(alias, row)

    pairs = [
        (gene_id, input_lookup[name], name)
        for gene_id, name in enumerate(names)
        if name in input_lookup
    ]
    if not pairs:
        raise ValueError("no genes overlap between checkpoint vocabulary and inference AnnData")

    # Perturbation genes can belong to the global model vocabulary without being
    # part of a benchmark's measured HVG panel.
    alias_to_checkpoint: dict[str, int] = {name: gene_id for gene_id, name in enumerate(names)}
    for gene_id, row, checkpoint_name in pairs:
        alias_to_checkpoint.setdefault(checkpoint_name, gene_id)
        alias_to_checkpoint.setdefault(str(adata.var_names[row]), gene_id)
        for column in alias_columns:
            if column in adata.var:
                alias_to_checkpoint.setdefault(str(adata.var[column].iloc[row]), gene_id)
    perturbation_id = None if perturbation is None else alias_to_checkpoint.get(str(perturbation))
    if perturbation is not None and perturbation_id is None:
        raise ValueError(
            f"perturbation {perturbation!r} is absent from the checkpoint vocabulary; "
            "use its Ensembl ID or a symbol present in adata.var"
        )

    # Force the perturbation gene to the first position after CLS when the
    # released expression panel actually contains it, as in A.10.2.  Some
    # public benchmark HVG panels omit the perturbed gene; in that case the
    # perturbation embedding still conditions the model, but callers must
    # record that strict target-token forcing was impossible.
    pairs.sort(key=lambda item: (perturbation_id is None or item[0] != perturbation_id, item[0]))
    if max_genes is not None:
        if (
            isinstance(max_genes, bool)
            or not isinstance(max_genes, Integral)
            or max_genes <= 0
        ):
            raise ValueError("max_genes must be a positive integer")
        pairs = pairs[:max_genes]
    if len(pairs) > config.num_genes:
        pairs = pairs[: config.num_genes]
    checkpoint_ids = np.asarray([p[0] for p in pairs], dtype=np.int64)
    input_ids = np.asarray([p[1] for p in pairs], dtype=np.int64)
    aligned_names = [p[2] for p in pairs]
    return input_ids, checkpoint_ids, aligned_names, perturbation_id


def _diffuse(
    model,
    config,
    control: np.ndarray,
    gene_ids: torch.Tensor,
    perturbation_id: torch.Tensor,
    priors,
    seed: int,
    device: torch.device,
    precision: str,
    return_mode: str = "cumulative_scattered",
) -> np.ndarray:
    if return_mode not in {"cumulative_scattered", "final_full_proposal"}:
        raise ValueError(
            "diffusion return mode must be 'cumulative_scattered' or "
            "'final_full_proposal'"
        )
    x_current = torch.as_tensor(control[None, :, :], dtype=torch.float32, device=device)
    mask = torch.zeros_like(x_current, dtype=torch.bool)
    gene_ids = gene_ids.to(device=device)
    perturbation_id = perturbation_id.to(device=device)
    priors = priors.to(device)
    steps = cumulative_reveal_steps(control.shape[1], config.diffusion_fractions, seed=seed, device=device)
    with torch.no_grad():
        final_proposal = None
        for step_mask in steps:
            with _autocast_context(device, precision):
                out = model(
                    x_current,
                    mask,
                    gene_ids,
                    perturbation_id,
                    priors,
                    reconstruct_priors=False,
                )
            prediction = out.expression.clamp_min(0.0)
            final_proposal = prediction
            step = step_mask[None, None, :].expand_as(mask)
            x_current = torch.where(step, prediction, x_current)
            mask = mask | step
    if return_mode == "final_full_proposal":
        if final_proposal is None:
            raise AssertionError("diffusion schedule unexpectedly produced no proposal")
        # Appendix A.4's final sentence says to return X-hat(T), which can be
        # read as the last full-profile proposal even though the preceding
        # cumulative-update algorithm freezes earlier quarters. Keep that
        # paper-internal alternative explicit rather than silently conflating it
        # with the primary cumulative-scattered interpretation.
        return final_proposal.squeeze(0).float().cpu().numpy()
    # Every position has been scattered exactly once after the final step. This
    # follows Methods 4.1.3's explicit fixed/cumulative update semantics.
    return x_current.squeeze(0).cpu().numpy()


def predict_adata(
    checkpoint_path: str | Path,
    adata_path: str | Path,
    perturbation: str,
    output_path: str | Path,
    seed: int = 0,
    prior_files: dict[str, str | Path | None] | None = None,
    n_cells: int = 64,
    context: str | None = None,
    batch: str | None = None,
    n_sets: int = 1,
    tta: bool = False,
    tta_steps: int = 200,
    tta_lr: float = 1e-5,
    device: str = "cpu",
    precision: str = "float32",
    max_genes: int | None = None,
    diffusion_return_mode: str = "cumulative_scattered",
) -> AnnData:
    if (
        not isinstance(perturbation, str)
        or not perturbation.strip()
        or perturbation != perturbation.strip()
    ):
        raise ValueError("perturbation must be a non-empty canonical string")
    for label, value in (("context", context), ("batch", batch)):
        if value is not None and (
            not isinstance(value, str) or not value.strip() or value != value.strip()
        ):
            raise ValueError(f"{label} must be a non-empty canonical string when provided")
    if isinstance(n_cells, bool) or not isinstance(n_cells, Integral) or n_cells <= 0:
        raise ValueError("n_cells must be a positive integer")
    if isinstance(n_sets, bool) or not isinstance(n_sets, Integral) or n_sets <= 0:
        raise ValueError("n_sets must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(tta, bool):
        raise ValueError("tta must be a bool")
    if isinstance(tta_steps, bool) or not isinstance(tta_steps, Integral) or tta_steps <= 0:
        raise ValueError("tta_steps must be a positive integer")
    if (
        isinstance(tta_lr, bool)
        or not isinstance(tta_lr, Real)
        or not np.isfinite(tta_lr)
        or tta_lr <= 0
    ):
        raise ValueError("tta_lr must be finite and positive")
    if max_genes is not None and (
        isinstance(max_genes, bool) or not isinstance(max_genes, Integral) or max_genes <= 0
    ):
        raise ValueError("max_genes must be a positive integer when provided")
    if diffusion_return_mode not in {"cumulative_scattered", "final_full_proposal"}:
        raise ValueError("unsupported diffusion return mode")
    target_device = torch.device(device)
    # Validate the requested numerical mode before loading a potentially large
    # checkpoint or AnnData file.
    _autocast_context(target_device, precision)
    checkpoint_path = Path(checkpoint_path).resolve()
    adata_path = Path(adata_path).resolve()
    target = Path(output_path).resolve()
    temporary = target.with_suffix(".tmp.h5ad")
    protected_inputs = {checkpoint_path, adata_path}
    protected_inputs.update(
        Path(path).resolve()
        for path in (prior_files or {}).values()
        if path is not None
    )
    _validate_output_target(target, protected_inputs, temporary)
    inference_inputs = _inference_inputs(checkpoint_path, adata_path, prior_files)
    model, config, checkpoint_var_names = load_checkpoint(checkpoint_path, map_location=device)
    adata = read_h5ad(adata_path)
    matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("inference AnnData must contain a finite 2D expression matrix")
    if "is_control" not in adata.obs:
        raise ValueError("prediction AnnData must contain obs['is_control']")
    control_rows = np.flatnonzero(boolean_obs_mask(adata, "is_control"))
    if len(control_rows) == 0:
        raise ValueError("prediction requires control cells")
    pool = _select_control_pool(adata, control_rows, context, batch, min_size=n_cells)

    input_gene_ids, checkpoint_gene_ids, aligned_names, perturbation_id_value = _aligned_gene_indices(
        adata, checkpoint_var_names, config, perturbation, max_genes
    )

    if tta:
        from open_x_cell.tta import adapt_model_tta
        tta_length = min(config.context_length, len(input_gene_ids))
        adapt_model_tta(
            model,
            config,
            np.asarray(matrix[pool][:, input_gene_ids[:tta_length]], dtype=np.float32),
            steps=tta_steps,
            learning_rate=tta_lr,
            set_size=n_cells,
            seed=seed,
            device=device,
            gene_ids=checkpoint_gene_ids[:tta_length],
            precision=precision,
        )

    if perturbation_id_value is None:  # guarded by the non-optional perturbation argument
        raise AssertionError("perturbation alignment unexpectedly returned no gene ID")
    perturbation_gene_token_present = bool(
        np.any(checkpoint_gene_ids == perturbation_id_value)
    )
    perturbation_id = torch.tensor([perturbation_id_value], dtype=torch.long)
    model_gene_ids = torch.as_tensor(checkpoint_gene_ids[None, :], dtype=torch.long)
    if prior_files:
        canonical_target = checkpoint_var_names[perturbation_id_value]
        priors = PriorStore.from_files(prior_files, config).batch_for_perturbations((canonical_target,), config)
    else:
        priors = make_zero_priors(1, config)

    rng = np.random.default_rng(seed)
    finals, all_rows = [], []
    for s in range(n_sets):
        if n_sets == 1:
            # deterministic first-n_cells set (wrap-pad) keeps single-set runs reproducible
            used = pool[:n_cells]
            if len(used) < n_cells:
                used = used[np.resize(np.arange(len(used)), n_cells)]
        else:
            used = rng.choice(pool, size=n_cells, replace=len(pool) < n_cells)
        control = np.asarray(matrix[used][:, input_gene_ids], dtype=np.float32)
        finals.append(
            _diffuse(
                model,
                config,
                control,
                model_gene_ids,
                perturbation_id,
                priors,
                seed=seed + s,
                device=target_device,
                precision=precision,
                return_mode=diffusion_return_mode,
            )
        )
        all_rows.append(used)

    final_x = np.concatenate(finals, axis=0)
    used_rows = np.concatenate(all_rows, axis=0)
    var = adata.var.iloc[input_gene_ids].copy()
    var.index = aligned_names
    obs = adata.obs.iloc[used_rows].copy()
    obs["source_control_obs_name"] = adata.obs_names[used_rows].astype(str)
    obs["perturbation"] = perturbation
    obs["is_control"] = False
    # Ensembles may reuse a control cell. Assign unique prediction row IDs before
    # constructing AnnData so callers do not receive duplicate-index warnings,
    # while retaining exact provenance in ``source_control_obs_name``.
    obs.index = [f"{name}-xcell-{idx}" for idx, name in enumerate(obs.index.astype(str))]
    pred = AnnData(X=final_x, obs=obs, var=var)
    pred.uns["open_x_cell_prediction"] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": inference_inputs["checkpoint"]["sha256"],
        "source_adata": str(adata_path),
        "perturbation": str(perturbation),
        "perturbation_gene_token_present": perturbation_gene_token_present,
        "seed": int(seed),
        "n_cells_per_set": int(n_cells),
        "n_sets": int(n_sets),
        "context": context,
        "batch": batch,
        "tta": bool(tta),
        "tta_steps": int(tta_steps),
        "tta_learning_rate": float(tta_lr),
        "precision": precision,
        "max_genes": int(max_genes) if max_genes is not None else -1,
        "diffusion_return_mode": diffusion_return_mode,
        "source_control_obs_names": adata.obs_names[used_rows].astype(str).tolist(),
        "inference_inputs_json": _canonical_json(inference_inputs),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    pred.write_h5ad(temporary)
    temporary.replace(target)
    return pred


def predict_benchmark(
    checkpoint_path: str | Path,
    adata_path: str | Path,
    perturbations_path: str | Path,
    output_dir: str | Path,
    seed: int = 0,
    prior_files: dict[str, str | Path | None] | None = None,
    n_cells: int = 32,
    shard_index: int = 0,
    shard_count: int = 1,
    device: str = "cpu",
    precision: str = "float32",
    max_genes: int | None = None,
    overwrite: bool = False,
    diffusion_return_mode: str = "cumulative_scattered",
) -> dict:
    """Predict a complete held-out benchmark while loading model/data only once.

    The paper uses one fixed set of 32 NTCs for every test perturbation.  Prepared
    benchmark files can mark those rows with ``obs['benchmark_control']``; when
    absent, the first ``n_cells`` control rows are used deterministically.  Shards
    are independent and resumable, which allows one process per GPU.
    """
    if isinstance(n_cells, bool) or not isinstance(n_cells, Integral) or n_cells <= 0:
        raise ValueError("n_cells must be a positive integer")
    if (
        isinstance(shard_count, bool)
        or not isinstance(shard_count, Integral)
        or isinstance(shard_index, bool)
        or not isinstance(shard_index, Integral)
        or shard_count <= 0
        or not 0 <= shard_index < shard_count
    ):
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if max_genes is not None and (
        isinstance(max_genes, bool) or not isinstance(max_genes, Integral) or max_genes <= 0
    ):
        raise ValueError("max_genes must be a positive integer when provided")
    if diffusion_return_mode not in {"cumulative_scattered", "final_full_proposal"}:
        raise ValueError("unsupported diffusion return mode")

    target_device = torch.device(device)
    _autocast_context(target_device, precision)
    checkpoint_path = Path(checkpoint_path).resolve()
    adata_path = Path(adata_path).resolve()
    perturbations_path = Path(perturbations_path).resolve()
    protected_inputs = {checkpoint_path, adata_path, perturbations_path}
    protected_inputs.update(
        Path(path).resolve()
        for path in (prior_files or {}).values()
        if path is not None
    )
    inference_inputs = _inference_inputs(
        checkpoint_path,
        adata_path,
        prior_files,
        perturbations_path=perturbations_path,
    )
    inference_inputs_json = _canonical_json(inference_inputs)
    checkpoint_sha256 = str(inference_inputs["checkpoint"]["sha256"])
    model, config, checkpoint_var_names = load_checkpoint(checkpoint_path, map_location=device)
    adata = read_h5ad(adata_path)
    matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("benchmark AnnData must contain a finite 2D expression matrix")
    perturbation_payload = json.loads(perturbations_path.read_text(encoding="utf-8"))
    if not isinstance(perturbation_payload, list) or not perturbation_payload:
        raise ValueError("benchmark perturbations must be a non-empty JSON list")
    if any(
        not isinstance(value, str) or not value.strip() or value != value.strip()
        for value in perturbation_payload
    ):
        raise ValueError("benchmark perturbations must be non-empty canonical strings")
    perturbations = list(perturbation_payload)
    if len(perturbations) != len(set(perturbations)):
        raise ValueError("benchmark perturbation list contains duplicates")

    if "is_control" not in adata.obs:
        raise ValueError("benchmark AnnData must contain obs['is_control']")
    is_control = boolean_obs_mask(adata, "is_control")
    control_rows = np.flatnonzero(is_control)
    if "benchmark_control" in adata.obs:
        marked = np.flatnonzero(boolean_obs_mask(adata, "benchmark_control"))
        if len(marked) != n_cells:
            raise ValueError(
                f"benchmark marks {len(marked)} fixed controls, expected exactly {n_cells}"
            )
        if not is_control[marked].all():
            raise ValueError("obs['benchmark_control'] marks one or more non-control cells")
        control_rows = marked
    if len(control_rows) < n_cells:
        raise ValueError(
            f"benchmark requires {n_cells} fixed controls but only {len(control_rows)} are available"
        )
    fixed_control_rows = control_rows[:n_cells]
    prior_store = PriorStore.from_files(prior_files or {}, config)
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / f"manifest_shard_{shard_index:03d}.json"
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    _validate_output_target(manifest_path, protected_inputs, temporary_manifest)

    outputs: list[str] = []
    skipped: list[str] = []
    assigned = [
        (index, perturbation)
        for index, perturbation in enumerate(perturbations)
        if index % shard_count == shard_index
    ]

    def validate_existing(path: Path, index: int, perturbation: str) -> None:
        existing = read_h5ad(path, backed="r")
        try:
            _, expected_checkpoint_ids, expected_gene_names, expected_perturbation_id = _aligned_gene_indices(
                adata,
                checkpoint_var_names,
                config,
                perturbation,
                max_genes,
            )
            labels = existing.obs["perturbation"].astype(str).unique().tolist()
            metadata = dict(existing.uns.get("open_x_cell_prediction", {}))
            expected_controls = adata.obs_names[fixed_control_rows].astype(str).tolist()
            expected_metadata = {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "source_adata": str(adata_path),
                "perturbations_file": str(perturbations_path),
                "perturbation": perturbation,
                "perturbation_index": index,
                "perturbation_gene_token_present": bool(
                    expected_perturbation_id is not None
                    and np.any(expected_checkpoint_ids == expected_perturbation_id)
                ),
                "seed": seed + index,
                "n_cells": n_cells,
                "fixed_control_obs_names": expected_controls,
                "precision": precision,
                "max_genes": int(max_genes) if max_genes is not None else -1,
                "diffusion_return_mode": diffusion_return_mode,
                "inference_inputs_json": inference_inputs_json,
            }
            actual_metadata = {key: metadata.get(key) for key in expected_metadata}
            actual_metadata["fixed_control_obs_names"] = list(
                map(str, metadata.get("fixed_control_obs_names", []))
            )
            if labels != [perturbation] or actual_metadata != expected_metadata:
                raise ValueError(
                    f"existing prediction {path} has stale or incomplete provenance; "
                    "use --overwrite only after confirming it may be replaced"
                )
            if existing.shape != (n_cells, len(expected_gene_names)):
                raise ValueError(f"existing prediction {path} has an unexpected shape")
            if list(map(str, existing.var_names)) != expected_gene_names:
                raise ValueError(f"existing prediction {path} has a different gene ordering")
            controls = existing.obs["source_control_obs_name"].astype(str).tolist()
            if controls != expected_controls:
                raise ValueError(f"existing prediction {path} used a different control set")
            values = np.asarray(existing.X[:], dtype=np.float32)
            if not np.isfinite(values).all() or values.min() < 0:
                raise ValueError(f"existing prediction {path} contains invalid expression values")
        finally:
            existing.file.close()

    for local_index, (index, perturbation) in enumerate(assigned, start=1):
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", perturbation).strip("_") or "perturbation"
        target = target_dir / f"{index:04d}_{safe_name}.h5ad"
        temporary = target.with_suffix(".tmp.h5ad")
        _validate_output_target(target, protected_inputs, temporary)
        if target.exists() and not overwrite:
            validate_existing(target, index, perturbation)
            skipped.append(str(target))
            continue

        input_ids, checkpoint_ids, aligned_names, perturbation_id_value = _aligned_gene_indices(
            adata,
            checkpoint_var_names,
            config,
            perturbation,
            max_genes,
        )
        if perturbation_id_value is None:
            raise AssertionError("benchmark perturbation unexpectedly has no checkpoint gene ID")
        perturbation_gene_token_present = bool(
            np.any(checkpoint_ids == perturbation_id_value)
        )
        control = np.asarray(matrix[fixed_control_rows][:, input_ids], dtype=np.float32)
        canonical_target = checkpoint_var_names[perturbation_id_value]
        priors = prior_store.batch_for_perturbations((canonical_target,), config)
        final = _diffuse(
            model,
            config,
            control,
            torch.as_tensor(checkpoint_ids[None, :], dtype=torch.long),
            torch.tensor([perturbation_id_value], dtype=torch.long),
            priors,
            seed=seed + index,
            device=target_device,
            precision=precision,
            return_mode=diffusion_return_mode,
        )
        obs = adata.obs.iloc[fixed_control_rows].copy()
        obs["source_control_obs_name"] = adata.obs_names[fixed_control_rows].astype(str)
        obs["perturbation"] = perturbation
        obs["is_control"] = False
        var = adata.var.iloc[input_ids].copy()
        var.index = aligned_names
        prediction = AnnData(X=final, obs=obs, var=var)
        prediction.obs_names_make_unique()
        prediction.uns["open_x_cell_prediction"] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "source_adata": str(adata_path),
            "perturbations_file": str(perturbations_path),
            "perturbation": perturbation,
            "perturbation_index": int(index),
            "perturbation_gene_token_present": perturbation_gene_token_present,
            "seed": int(seed + index),
            "n_cells": int(n_cells),
            "fixed_control_obs_names": adata.obs_names[fixed_control_rows].astype(str).tolist(),
            "precision": precision,
            "max_genes": int(max_genes) if max_genes is not None else -1,
            "diffusion_return_mode": diffusion_return_mode,
            "inference_inputs_json": inference_inputs_json,
        }
        prediction.write_h5ad(temporary)
        temporary.replace(target)
        outputs.append(str(target))
        if local_index == 1 or local_index % 10 == 0 or local_index == len(assigned):
            print(
                f"benchmark shard {shard_index}/{shard_count}: "
                f"{local_index}/{len(assigned)} ({perturbation})",
                flush=True,
            )

    manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "adata": str(adata_path),
        "perturbations": str(perturbations_path),
        "total_perturbations": len(perturbations),
        "assigned_perturbations": len(assigned),
        "n_cells": n_cells,
        "fixed_control_obs_names": adata.obs_names[fixed_control_rows].astype(str).tolist(),
        "seed": seed,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "device": str(target_device),
        "precision": precision,
        "max_genes": max_genes,
        "diffusion_return_mode": diffusion_return_mode,
        "inference_inputs": inference_inputs,
        "assigned": [
            {"index": int(index), "perturbation": perturbation}
            for index, perturbation in assigned
        ],
        "written": outputs,
        "skipped_existing": skipped,
    }
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary_manifest.replace(manifest_path)
    return manifest
