from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from open_x_cell.config import ModelConfig


PRIOR_SOURCES: tuple[str, ...] = ("esm", "string", "genept", "depmap", "cell_painting")
MISSING_GENE_KEYS = {"", "nan", "none", "null"}


@dataclass(frozen=True)
class PriorBatch:
    values: dict[str, torch.Tensor]
    missing: torch.Tensor
    source_names: tuple[str, ...] = PRIOR_SOURCES

    def to(self, device: torch.device | str) -> PriorBatch:
        target = torch.device(device)
        return PriorBatch(
            values={name: value.to(target) for name, value in self.values.items()},
            missing=self.missing.to(target),
            source_names=self.source_names,
        )


def make_zero_priors(batch_size: int, config: ModelConfig, device: torch.device | str | None = None) -> PriorBatch:
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    target = torch.device(device) if device is not None else None
    values = {
        source: torch.zeros(batch_size, config.prior_dims[source], dtype=torch.float32, device=target)
        for source in PRIOR_SOURCES
    }
    missing = torch.ones(batch_size, len(PRIOR_SOURCES), dtype=torch.bool, device=target)
    return PriorBatch(values=values, missing=missing, source_names=PRIOR_SOURCES)


def _clean_gene_keys(genes, source: Path) -> np.ndarray:
    cleaned = np.asarray([str(gene).strip() for gene in genes])
    if any(gene.lower() in MISSING_GENE_KEYS for gene in cleaned):
        raise ValueError(f"{source} contains an empty or missing gene key")
    if len(set(cleaned.tolist())) != len(cleaned):
        raise ValueError(f"{source} contains duplicate gene keys after whitespace normalization")
    return cleaned


def _load_prior_table(path: str | Path, expected_dim: int) -> dict[str, np.ndarray]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            if "genes" not in payload or "embeddings" not in payload:
                raise ValueError(f"{source} must contain 'genes' and 'embeddings' arrays")
            genes = payload["genes"].astype(str)
            values = np.asarray(payload["embeddings"], dtype=np.float32)
    else:
        if suffix == ".parquet":
            frame = pd.read_parquet(source)
        else:
            sep = "\t" if suffix in {".tsv", ".txt"} else ","
            frame = pd.read_csv(source, sep=sep)
        gene_column = next((name for name in ("gene", "gene_name", "gene_id", "symbol") if name in frame), None)
        if gene_column is None:
            gene_column = frame.columns[0]
        genes = frame[gene_column].astype(str).to_numpy()
        values = frame.drop(columns=[gene_column]).to_numpy(dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{source} must contain a 2D embedding matrix")
    if values.shape[0] != len(genes):
        raise ValueError(f"{source} has {len(genes)} gene keys but {values.shape[0]} embedding rows")
    if values.shape[1] != expected_dim:
        raise ValueError(f"{source} has dim {values.shape[1]}, expected {expected_dim}")
    if not len(genes):
        raise ValueError(f"{source} contains no embeddings")
    genes = _clean_gene_keys(genes, source)
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains non-finite embedding values")
    return {gene: values[idx].copy() for idx, gene in enumerate(genes)}


def load_gene_embedding_table(path: str | Path) -> dict[str, np.ndarray]:
    """Load a gene -> embedding table (e.g. scGPT gene embeddings) with any width.

    Same on-disk format as priors (`.npz` with 'genes'/'embeddings', or a table
    file with a gene column), but without enforcing a specific embedding dim.
    """
    source = Path(path)
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            if "genes" not in payload or "embeddings" not in payload:
                raise ValueError(f"{source} must contain 'genes' and 'embeddings' arrays")
            genes = payload["genes"].astype(str)
            values = np.asarray(payload["embeddings"], dtype=np.float32)
    else:
        if source.suffix.lower() == ".parquet":
            frame = pd.read_parquet(source)
        else:
            sep = "\t" if source.suffix.lower() in {".tsv", ".txt"} else ","
            frame = pd.read_csv(source, sep=sep)
        gene_column = next((n for n in ("gene", "gene_name", "gene_id", "symbol") if n in frame), frame.columns[0])
        genes = frame[gene_column].astype(str).to_numpy()
        values = frame.drop(columns=[gene_column]).to_numpy(dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{source} must contain a 2D embedding matrix")
    if values.shape[0] != len(genes):
        raise ValueError(f"{source} has {len(genes)} gene keys but {values.shape[0]} embedding rows")
    if not len(genes):
        raise ValueError(f"{source} contains no embeddings")
    genes = _clean_gene_keys(genes, source)
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains non-finite embedding values")
    return {gene: values[idx].copy() for idx, gene in enumerate(genes)}


def init_gene_embedding_from_table(
    model, var_names: list[str], table: dict[str, np.ndarray]
) -> tuple[int, int]:
    """Overwrite ``model.gene_embedding`` rows with embeddings from ``table``.

    Row ``i`` corresponds to ``var_names[i]`` (the model indexes the gene embedding
    by gene position). Genes absent from ``table`` keep their random init. Returns
    ``(n_initialized, n_total)``. Requires the table width to equal hidden_dim.
    """
    weight = model.gene_embedding.weight
    if not table:
        raise ValueError("gene embedding table is empty")
    n_rows, hidden = weight.shape
    sample = next(iter(table.values()))
    if sample.shape[0] != hidden:
        raise ValueError(
            f"gene embedding table dim {sample.shape[0]} != model hidden_dim {hidden}; "
            "use a matching architecture (e.g. mini, 512-d) or project the embeddings."
        )
    updated = weight.detach().clone()
    count = 0
    limit = min(n_rows, len(var_names))
    for i in range(limit):
        vector = table.get(str(var_names[i]))
        if vector is None:
            continue
        updated[i] = torch.from_numpy(np.asarray(vector, dtype=np.float32))
        count += 1
    with torch.no_grad():
        weight.copy_(updated.to(weight.device))
    return count, limit


@dataclass(frozen=True)
class PriorStore:
    embeddings: dict[str, dict[str, np.ndarray]]

    @classmethod
    def empty(cls) -> PriorStore:
        return cls(embeddings={source: {} for source in PRIOR_SOURCES})

    @classmethod
    def from_files(cls, files: dict[str, str | Path | None], config: ModelConfig) -> PriorStore:
        embeddings = {source: {} for source in PRIOR_SOURCES}
        for source, path in files.items():
            if source not in PRIOR_SOURCES:
                raise ValueError(f"unknown prior source {source!r}; choose from {', '.join(PRIOR_SOURCES)}")
            if path is None:
                continue
            embeddings[source] = _load_prior_table(path, config.prior_dims[source])
        return cls(embeddings=embeddings)

    @classmethod
    def from_mapping(cls, mapping: dict[str, dict[str, np.ndarray]]) -> PriorStore:
        embeddings = {source: {} for source in PRIOR_SOURCES}
        for source, values in mapping.items():
            if source not in PRIOR_SOURCES:
                raise ValueError(f"unknown prior source {source!r}; choose from {', '.join(PRIOR_SOURCES)}")
            normalized: dict[str, np.ndarray] = {}
            for gene, vector in values.items():
                key = str(gene).strip()
                if key.lower() in MISSING_GENE_KEYS:
                    raise ValueError(f"{source} prior contains an empty or missing gene key")
                if key in normalized:
                    raise ValueError(f"{source} prior contains duplicate gene key {key!r}")
                array = np.asarray(vector, dtype=np.float32)
                if array.ndim != 1 or not np.isfinite(array).all():
                    raise ValueError(f"{source} prior for {key!r} must be a finite vector")
                normalized[key] = array.copy()
            embeddings[source] = normalized
        return cls(embeddings=embeddings)

    def batch_for_perturbations(
        self,
        perturbations: tuple[str, ...] | list[str],
        config: ModelConfig,
        device: torch.device | str | None = None,
    ) -> PriorBatch:
        if not perturbations:
            raise ValueError("perturbations must be non-empty")
        target = torch.device(device) if device is not None else None
        values: dict[str, torch.Tensor] = {}
        missing = torch.ones(len(perturbations), len(PRIOR_SOURCES), dtype=torch.bool, device=target)
        for source_idx, source in enumerate(PRIOR_SOURCES):
            dim = config.prior_dims[source]
            matrix = np.zeros((len(perturbations), dim), dtype=np.float32)
            source_embeddings = self.embeddings.get(source, {})
            for row_idx, perturbation in enumerate(perturbations):
                vector = source_embeddings.get(str(perturbation))
                if vector is None:
                    continue
                if vector.shape != (dim,):
                    raise ValueError(f"{source} prior for {perturbation!r} has shape {vector.shape}, expected {(dim,)}")
                matrix[row_idx] = vector
                missing[row_idx, source_idx] = False
            values[source] = torch.as_tensor(matrix, dtype=torch.float32, device=target)
        return PriorBatch(values=values, missing=missing, source_names=PRIOR_SOURCES)
