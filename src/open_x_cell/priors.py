from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from open_x_cell.config import ModelConfig


PRIOR_SOURCES: tuple[str, ...] = ("esm", "string", "genept", "depmap", "cell_painting")


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
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    target = torch.device(device) if device is not None else None
    values = {
        source: torch.zeros(batch_size, config.prior_dims[source], dtype=torch.float32, device=target)
        for source in PRIOR_SOURCES
    }
    missing = torch.ones(batch_size, len(PRIOR_SOURCES), dtype=torch.bool, device=target)
    return PriorBatch(values=values, missing=missing, source_names=PRIOR_SOURCES)


def _load_prior_table(path: str | Path, expected_dim: int) -> dict[str, np.ndarray]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npz":
        payload = np.load(source, allow_pickle=False)
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
    if values.shape[1] != expected_dim:
        raise ValueError(f"{source} has dim {values.shape[1]}, expected {expected_dim}")
    return {gene: values[idx].copy() for idx, gene in enumerate(genes)}


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
            if path is None:
                continue
            if source not in PRIOR_SOURCES:
                raise ValueError(f"unknown prior source {source!r}; choose from {', '.join(PRIOR_SOURCES)}")
            embeddings[source] = _load_prior_table(path, config.prior_dims[source])
        return cls(embeddings=embeddings)

    @classmethod
    def from_mapping(cls, mapping: dict[str, dict[str, np.ndarray]]) -> PriorStore:
        embeddings = {source: {} for source in PRIOR_SOURCES}
        for source, values in mapping.items():
            if source not in PRIOR_SOURCES:
                raise ValueError(f"unknown prior source {source!r}; choose from {', '.join(PRIOR_SOURCES)}")
            embeddings[source] = {
                gene: np.asarray(vector, dtype=np.float32).copy() for gene, vector in values.items()
            }
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
