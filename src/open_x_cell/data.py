from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    context: str
    batch: str
    control_indices: np.ndarray
    target_indices: np.ndarray


@dataclass(frozen=True)
class OpenXCellBatch:
    x_control: torch.Tensor
    x_target: torch.Tensor
    gene_ids: torch.Tensor
    perturbation_ids: torch.Tensor
    perturbations: tuple[str, ...]
    contexts: tuple[str, ...]
    batches: tuple[str, ...]


@dataclass(frozen=True)
class _SetGroup:
    perturbation: str
    context: str
    batch: str
    target_indices: np.ndarray
    control_indices: np.ndarray


def load_adata(
    data: AnnData | str,
    normalize: Literal["none", "cp10k-log1p"] = "none",
    layer: str | None = None,
) -> AnnData:
    if isinstance(data, AnnData):
        adata = data
    else:
        adata = read_h5ad(data)
    if normalize == "none":
        return adata
    if normalize == "cp10k-log1p":
        return normalize_cp10k_log1p(adata, layer=layer)
    raise ValueError("normalize must be 'none' or 'cp10k-log1p'")


def dense_matrix(adata: AnnData) -> np.ndarray:
    x = adata.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def normalize_cp10k_log1p(adata: AnnData, layer: str | None = None, target_sum: float = 10_000.0) -> AnnData:
    if target_sum <= 0:
        raise ValueError("target_sum must be positive")
    normalized = adata.copy()
    x = normalized.layers[layer] if layer is not None else normalized.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    matrix = np.asarray(x, dtype=np.float32)
    totals = matrix.sum(axis=1, keepdims=True)
    scaled = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0) * target_sum
    normalized.X = np.log1p(scaled).astype(np.float32)
    meta = dict(normalized.uns.get("open_x_cell", {}))
    meta["normalization"] = "cp10k_log1p"
    meta["normalization_target_sum"] = float(target_sum)
    normalized.uns["open_x_cell"] = meta
    return normalized


class OpenXCellDataset(torch.utils.data.Dataset[OpenXCellSample]):
    def __init__(
        self,
        adata: AnnData,
        set_size: int = 64,
        context_length: int = 4_000,
        seed: int = 0,
        perturbation_key: str = "perturbation",
        control_key: str = "is_control",
        context_key: str = "context",
        batch_key: str = "batch",
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
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.gene_to_id = {name: idx for idx, name in enumerate(adata.var_names)}
        self.control_indices = np.flatnonzero(adata.obs[control_key].to_numpy(dtype=bool))
        if len(self.control_indices) == 0:
            raise ValueError("dataset requires at least one control cell")
        labels = adata.obs[perturbation_key].astype(str).to_numpy()
        contexts = (
            adata.obs[context_key].astype(str).to_numpy()
            if context_key in adata.obs
            else np.array(["default"] * adata.n_obs, dtype=object)
        )
        batches = (
            adata.obs[batch_key].astype(str).to_numpy()
            if batch_key in adata.obs
            else np.array(["default"] * adata.n_obs, dtype=object)
        )
        self.contexts = contexts
        self.batches = batches
        control_by_context_batch: dict[tuple[str, str], np.ndarray] = {}
        control_by_context: dict[str, np.ndarray] = {}
        for context in sorted(set(contexts[self.control_indices])):
            control_by_context[context] = self.control_indices[contexts[self.control_indices] == context]
        for context in sorted(set(contexts[self.control_indices])):
            for batch in sorted(set(batches[self.control_indices])):
                mask = (contexts[self.control_indices] == context) & (batches[self.control_indices] == batch)
                if np.any(mask):
                    control_by_context_batch[(context, batch)] = self.control_indices[mask]
        self.samples: list[_SetGroup] = []
        for context in sorted(set(contexts)):
            for label in sorted(set(labels)):
                if label == "control":
                    continue
                for batch in sorted(set(batches)):
                    target_indices = np.flatnonzero((labels == label) & (contexts == context) & (batches == batch))
                    if not len(target_indices):
                        continue
                    control_pool = control_by_context_batch.get((context, batch))
                    if control_pool is None or not len(control_pool):
                        control_pool = control_by_context.get(context)
                    if control_pool is None or not len(control_pool):
                        control_pool = self.control_indices
                    self.samples.append(
                        _SetGroup(
                            perturbation=label,
                            context=context,
                            batch=batch,
                            target_indices=target_indices,
                            control_indices=control_pool,
                        )
                    )
        if not self.samples:
            raise ValueError("dataset requires at least one perturbation")
        self._build_set_index()

    def _control_set(self, pool: np.ndarray) -> np.ndarray:
        """A control set of exactly set_size cells; whole-set replicate (wrap) when short."""
        if len(pool) >= self.set_size:
            return pool[self.rng.permutation(len(pool))[: self.set_size]]
        reps = int(np.ceil(self.set_size / len(pool)))
        return np.tile(pool, reps)[: self.set_size]

    def _build_set_index(self) -> None:
        """Deterministic set construction (paper A.10.1): permute target cells, chunk
        into complete sets of set_size, discard the incomplete remainder; pair each
        with a context/batch-matched control set (NTC whole-sets replicated if short).

        If no group yields a complete target set (tiny data), fall back to one
        replicated set per group so the dataset is never empty.
        """
        self.set_index: list[tuple[int, np.ndarray, np.ndarray]] = []
        for gi, group in enumerate(self.samples):
            tgt = group.target_indices[self.rng.permutation(len(group.target_indices))]
            n_sets = len(tgt) // self.set_size
            for s in range(n_sets):
                target_set = tgt[s * self.set_size : (s + 1) * self.set_size]
                self.set_index.append((gi, target_set, self._control_set(group.control_indices)))
        if not self.set_index:
            for gi, group in enumerate(self.samples):
                reps = int(np.ceil(self.set_size / len(group.target_indices)))
                target_set = np.tile(group.target_indices, reps)[: self.set_size]
                self.set_index.append((gi, target_set, self._control_set(group.control_indices)))

    def __len__(self) -> int:
        return len(self.set_index)

    def _gene_ids_for(self, perturbation: str, index: int) -> np.ndarray:
        if self.context_length == self.adata.n_vars:
            selected = np.arange(self.adata.n_vars)
        else:
            # deterministic per-set context subsample
            rng = np.random.default_rng(self.seed + index)
            selected = rng.choice(self.adata.n_vars, size=self.context_length, replace=False)
        if perturbation in self.gene_to_id:
            selected = selected.copy()
            selected[0] = self.gene_to_id[perturbation]
        return selected.astype(np.int64)

    def __getitem__(self, index: int) -> OpenXCellSample:
        group_idx, target_rows, control_rows = self.set_index[index % len(self.set_index)]
        group = self.samples[group_idx]
        perturbation = group.perturbation
        gene_ids = self._gene_ids_for(perturbation, index)
        return OpenXCellSample(
            x_control=torch.from_numpy(self.matrix[control_rows][:, gene_ids]),
            x_target=torch.from_numpy(self.matrix[target_rows][:, gene_ids]),
            gene_ids=torch.from_numpy(gene_ids),
            perturbation_id=self.gene_to_id.get(perturbation, 0),
            perturbation=perturbation,
            context=group.context,
            batch=group.batch,
            control_indices=control_rows,
            target_indices=target_rows,
        )


def collate_samples(samples: list[OpenXCellSample]) -> OpenXCellBatch:
    return OpenXCellBatch(
        x_control=torch.stack([sample.x_control for sample in samples]),
        x_target=torch.stack([sample.x_target for sample in samples]),
        gene_ids=torch.stack([sample.gene_ids for sample in samples]),
        perturbation_ids=torch.tensor([sample.perturbation_id for sample in samples], dtype=torch.long),
        perturbations=tuple(sample.perturbation for sample in samples),
        contexts=tuple(sample.context for sample in samples),
        batches=tuple(sample.batch for sample in samples),
    )
