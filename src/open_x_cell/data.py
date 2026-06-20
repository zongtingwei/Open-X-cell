from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from anndata import AnnData, read_h5ad


@dataclass(frozen=True)
class OpenXCellSample:
    x_control: torch.Tensor
    x_target: torch.Tensor
    gene_ids: torch.Tensor
    perturbation_id: int
    perturbation: str


@dataclass(frozen=True)
class OpenXCellBatch:
    x_control: torch.Tensor
    x_target: torch.Tensor
    gene_ids: torch.Tensor
    perturbation_ids: torch.Tensor
    perturbations: tuple[str, ...]


def load_adata(data: AnnData | str) -> AnnData:
    if isinstance(data, AnnData):
        return data
    return read_h5ad(data)


def dense_matrix(adata: AnnData) -> np.ndarray:
    x = adata.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


class OpenXCellDataset(torch.utils.data.Dataset[OpenXCellSample]):
    def __init__(
        self,
        adata: AnnData,
        set_size: int = 64,
        context_length: int = 4_000,
        seed: int = 0,
        perturbation_key: str = "perturbation",
        control_key: str = "is_control",
    ) -> None:
        if perturbation_key not in adata.obs:
            raise ValueError(f"AnnData.obs must contain {perturbation_key!r}")
        if control_key not in adata.obs:
            raise ValueError(f"AnnData.obs must contain {control_key!r}")
        if set_size <= 0:
            raise ValueError("set_size must be positive")
        self.adata = adata
        self.matrix = dense_matrix(adata)
        self.set_size = set_size
        self.context_length = min(context_length, adata.n_vars)
        self.rng = np.random.default_rng(seed)
        self.gene_to_id = {name: idx for idx, name in enumerate(adata.var_names)}
        self.control_indices = np.flatnonzero(adata.obs[control_key].to_numpy(dtype=bool))
        if len(self.control_indices) == 0:
            raise ValueError("dataset requires at least one control cell")
        labels = adata.obs[perturbation_key].astype(str).to_numpy()
        self.samples: list[tuple[str, np.ndarray]] = []
        for label in sorted(set(labels)):
            if label == "control":
                continue
            target_indices = np.flatnonzero(labels == label)
            if len(target_indices):
                self.samples.append((label, target_indices))
        if not self.samples:
            raise ValueError("dataset requires at least one perturbation")

    def __len__(self) -> int:
        return len(self.samples)

    def _choose(self, indices: np.ndarray) -> np.ndarray:
        replace = len(indices) < self.set_size
        return self.rng.choice(indices, size=self.set_size, replace=replace)

    def _gene_ids_for(self, perturbation: str) -> np.ndarray:
        all_ids = np.arange(self.adata.n_vars)
        if self.context_length == self.adata.n_vars:
            selected = all_ids
        else:
            selected = self.rng.choice(all_ids, size=self.context_length, replace=False)
        if perturbation in self.gene_to_id:
            pert_id = self.gene_to_id[perturbation]
            selected = selected.copy()
            selected[0] = pert_id
        return selected.astype(np.int64)

    def __getitem__(self, index: int) -> OpenXCellSample:
        perturbation, target_pool = self.samples[index % len(self.samples)]
        gene_ids = self._gene_ids_for(perturbation)
        control_rows = self._choose(self.control_indices)
        target_rows = self._choose(target_pool)
        perturbation_id = self.gene_to_id.get(perturbation, 0)
        return OpenXCellSample(
            x_control=torch.from_numpy(self.matrix[control_rows][:, gene_ids]),
            x_target=torch.from_numpy(self.matrix[target_rows][:, gene_ids]),
            gene_ids=torch.from_numpy(gene_ids),
            perturbation_id=perturbation_id,
            perturbation=perturbation,
        )


def collate_samples(samples: list[OpenXCellSample]) -> OpenXCellBatch:
    return OpenXCellBatch(
        x_control=torch.stack([sample.x_control for sample in samples]),
        x_target=torch.stack([sample.x_target for sample in samples]),
        gene_ids=torch.stack([sample.gene_ids for sample in samples]),
        perturbation_ids=torch.tensor([sample.perturbation_id for sample in samples], dtype=torch.long),
        perturbations=tuple(sample.perturbation for sample in samples),
    )
