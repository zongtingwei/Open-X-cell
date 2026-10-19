from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
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


def boolean_obs_mask(adata: AnnData, key: str) -> np.ndarray:
    """Read a boolean AnnData observation field without Python truthiness traps."""
    if key not in adata.obs:
        raise ValueError(f"AnnData.obs must contain {key!r}")
    series = adata.obs[key]
    if series.isna().any():
        raise ValueError(f"AnnData.obs[{key!r}] contains missing values")
    values = series.to_numpy()
    if values.dtype == np.bool_ or all(isinstance(value, (bool, np.bool_)) for value in values):
        return np.asarray(values, dtype=bool)
    if np.issubdtype(values.dtype, np.number):
        numeric = np.asarray(values)
        if np.isin(numeric, (0, 1)).all():
            return numeric.astype(bool)
    normalized = np.asarray([str(value).strip().lower() for value in values], dtype=object)
    if np.isin(normalized, ("true", "false", "1", "0")).all():
        return np.isin(normalized, ("true", "1"))
    raise ValueError(
        f"AnnData.obs[{key!r}] must contain booleans (or unambiguous 0/1 values)"
    )


def load_adata(
    data: AnnData | str,
    normalize: Literal["none", "cp10k-log1p"] = "none",
    layer: str | None = None,
    backed: bool = False,
) -> AnnData:
    if not isinstance(backed, bool):
        raise ValueError("backed must be a bool")
    if backed and normalize != "none":
        raise ValueError("backed AnnData training requires normalize='none'")
    if isinstance(data, AnnData):
        adata = data
        if backed and not adata.isbacked:
            raise ValueError("backed=True requires a file-backed AnnData input")
    else:
        adata = read_h5ad(data, backed="r" if backed else None)
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


def expression_rows_columns(
    matrix,
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    backed: bool,
) -> np.ndarray:
    """Read rows/columns safely from in-memory or backed expression storage.

    h5py requires fancy row indices to be sorted and unique, while Appendix
    A.10.1 set padding intentionally produces duplicate rows. Read each unique
    backed row once, restore the requested order/duplicates in memory, and only
    then apply the (usually random) dynamic gene-column order.
    """
    row_index = np.asarray(rows, dtype=np.int64)
    column_index = np.asarray(columns, dtype=np.int64)
    if row_index.ndim != 1 or column_index.ndim != 1:
        raise ValueError("expression row and column indices must be one-dimensional")
    if not len(row_index) or not len(column_index):
        raise ValueError("expression row and column indices must be non-empty")
    if backed:
        unique_rows, inverse = np.unique(row_index, return_inverse=True)
        values = matrix[unique_rows]
    else:
        inverse = None
        values = matrix[row_index]
    if hasattr(values, "toarray"):
        values = values.toarray()
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("expression row read must be two-dimensional")
    if inverse is not None:
        array = array[inverse]
    array = np.ascontiguousarray(array[:, column_index], dtype=np.float32)
    if array.shape != (len(row_index), len(column_index)) or not np.isfinite(array).all():
        raise ValueError("expression slice has the wrong shape or contains non-finite values")
    return array


def normalize_cp10k_log1p(adata: AnnData, layer: str | None = None, target_sum: float = 10_000.0) -> AnnData:
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("target_sum must be finite and positive")
    normalized = adata.copy()
    if layer is not None and layer not in normalized.layers:
        raise ValueError(f"AnnData.layers does not contain {layer!r}")
    x = normalized.layers[layer] if layer is not None else normalized.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    matrix = np.asarray(x, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape != normalized.shape:
        raise ValueError("expression matrix must be two-dimensional and match AnnData.shape")
    if not np.isfinite(matrix).all():
        raise ValueError("expression matrix contains non-finite values")
    if np.any(matrix < 0):
        raise ValueError("CP10k/log1p normalization requires non-negative raw counts")
    totals = matrix.sum(axis=1, keepdims=True, dtype=np.float64)
    scaled = np.divide(
        matrix,
        totals,
        out=np.zeros(matrix.shape, dtype=np.float64),
        where=totals > 0,
    ) * target_sum
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
        vocabulary: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if perturbation_key not in adata.obs:
            raise ValueError(f"AnnData.obs must contain {perturbation_key!r}")
        if control_key not in adata.obs:
            raise ValueError(f"AnnData.obs must contain {control_key!r}")
        if isinstance(set_size, bool) or not isinstance(set_size, Integral) or set_size <= 0:
            raise ValueError("set_size must be a positive integer")
        if (
            isinstance(context_length, bool)
            or not isinstance(context_length, Integral)
            or context_length <= 0
        ):
            raise ValueError("context_length must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if adata.n_obs <= 0 or adata.n_vars <= 0:
            raise ValueError("dataset AnnData must contain cells and genes")
        if not adata.var_names.is_unique:
            raise ValueError("AnnData.var_names must be unique")
        schema_keys = [perturbation_key]
        if context_key in adata.obs:
            schema_keys.append(context_key)
        if batch_key in adata.obs:
            schema_keys.append(batch_key)
        for key in schema_keys:
            if adata.obs[key].isna().any():
                raise ValueError(f"AnnData.obs[{key!r}] contains missing values")
        self.adata = adata
        self.expression_backed = bool(adata.isbacked)
        self.matrix = adata.X if self.expression_backed else dense_matrix(adata)
        if self.matrix.shape != adata.shape:
            raise ValueError("AnnData expression matrix must match its shape")
        if not self.expression_backed and not np.isfinite(self.matrix).all():
            raise ValueError("AnnData expression matrix must contain finite values")
        matrix_dtype = getattr(self.matrix, "dtype", None)
        if matrix_dtype is None or not np.issubdtype(np.dtype(matrix_dtype), np.number):
            raise ValueError("AnnData expression matrix must have a numeric dtype")
        self.set_size = set_size
        self.context_length = min(context_length, adata.n_vars)
        self.seed = seed
        self.epoch = 0
        self.vocabulary = list(map(str, vocabulary)) if vocabulary is not None else list(map(str, adata.var_names))
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("gene vocabulary must not contain duplicates")
        self.gene_to_id = {name: idx for idx, name in enumerate(self.vocabulary)}
        try:
            self.model_gene_ids = np.asarray(
                [self.gene_to_id[str(name)] for name in adata.var_names], dtype=np.int64
            )
        except KeyError as exc:
            raise ValueError(f"measured gene {exc.args[0]!r} is absent from the model vocabulary") from exc
        self.column_by_gene = {str(name): idx for idx, name in enumerate(adata.var_names)}
        control_mask = boolean_obs_mask(adata, control_key)
        self.control_indices = np.flatnonzero(control_mask)
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
        control_lists_by_context_batch: dict[tuple[str, str], list[int]] = {}
        control_lists_by_context: dict[str, list[int]] = {}
        for row in self.control_indices:
            context = contexts[row]
            batch = batches[row]
            control_lists_by_context.setdefault(context, []).append(int(row))
            control_lists_by_context_batch.setdefault((context, batch), []).append(int(row))
        control_by_context = {
            key: np.asarray(rows, dtype=np.int64) for key, rows in control_lists_by_context.items()
        }
        control_by_context_batch = {
            key: np.asarray(rows, dtype=np.int64)
            for key, rows in control_lists_by_context_batch.items()
        }

        # Build target strata in one linear pass. The former context × label ×
        # batch nested loop rescanned every cell over 100,000 times on real data.
        target_lists: dict[tuple[str, str, str], list[int]] = {}
        for row, label in enumerate(labels):
            # ``is_control`` is the authoritative schema field. Public datasets
            # use several NTC labels, so relying on the literal string
            # ``control`` can accidentally train on controls as perturbations.
            if control_mask[row]:
                continue
            key = (contexts[row], label, batches[row])
            target_lists.setdefault(key, []).append(row)
        self.samples: list[_SetGroup] = []
        for (context, label, batch), rows in sorted(target_lists.items()):
            target_indices = np.asarray(rows, dtype=np.int64)
            control_pool = control_by_context_batch.get((context, batch))
            context_pool = control_by_context.get(context)
            # Appendix A.10.1 requires a context-level fallback whenever the
            # batch-specific NTC pool cannot fill one set, not only when it is
            # completely empty. If the whole context is still short,
            # ``_control_set`` performs deterministic replication.
            if context_pool is None or not len(context_pool):
                raise ValueError(
                    f"context {context!r} has perturbation cells but no context-matched controls"
                )
            if control_pool is None or len(control_pool) < self.set_size:
                control_pool = context_pool
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
        self.set_epoch(0)

    def _control_set(self, pool: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample one control set, using replacement only when the pool is short."""
        if len(pool) >= self.set_size:
            return pool[rng.permutation(len(pool))[: self.set_size]]
        return rng.choice(pool, size=self.set_size, replace=True)

    def set_epoch(self, epoch: int) -> None:
        """Deterministically resample cell sets and gene panels for one epoch."""
        if isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = int(epoch)
        self._build_set_index()

    def _build_set_index(self) -> None:
        """Deterministic set construction following paper A.10.1.

        Target cells are permuted and chunked into sets of ``set_size``.  The final
        incomplete chunk is padded by sampling cells from the same stratum with
        replacement.  Each target set is paired with context/batch-matched NTCs,
        with the documented context-level fallback when a batch pool is too small.
        """
        self.set_index: list[tuple[int, np.ndarray, np.ndarray]] = []
        rng = np.random.default_rng(self.seed + self.epoch)
        for gi, group in enumerate(self.samples):
            tgt = group.target_indices[rng.permutation(len(group.target_indices))]
            n_sets = int(np.ceil(len(tgt) / self.set_size))
            for s in range(n_sets):
                target_set = tgt[s * self.set_size : (s + 1) * self.set_size]
                if len(target_set) < self.set_size:
                    padding = rng.choice(
                        group.target_indices,
                        size=self.set_size - len(target_set),
                        replace=True,
                    )
                    target_set = np.concatenate([target_set, padding])
                self.set_index.append((gi, target_set, self._control_set(group.control_indices, rng)))

    def __len__(self) -> int:
        return len(self.set_index)

    def _gene_columns_for(self, perturbation: str, index: int) -> np.ndarray:
        perturbation_column = self.column_by_gene.get(perturbation)
        if self.context_length == self.adata.n_vars:
            selected = np.arange(self.adata.n_vars, dtype=np.int64)
            if perturbation_column is not None and perturbation_column != 0:
                # Force the perturbed gene to position 1 after CLS without
                # duplicating it elsewhere in the context.
                selected = np.concatenate(
                    [np.array([perturbation_column]), selected[selected != perturbation_column]]
                )
        else:
            # deterministic per-set context subsample
            rng = np.random.default_rng(self.seed + self.epoch * len(self) + index)
            if perturbation_column is None:
                selected = rng.choice(self.adata.n_vars, size=self.context_length, replace=False)
            else:
                candidates = np.delete(np.arange(self.adata.n_vars), perturbation_column)
                remainder = rng.choice(candidates, size=self.context_length - 1, replace=False)
                selected = np.concatenate([np.array([perturbation_column]), remainder])
        return selected.astype(np.int64)

    def __getitem__(self, index: int) -> OpenXCellSample:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("dataset index must be an integer")
        if index < 0:
            index += len(self.set_index)
        if index < 0 or index >= len(self.set_index):
            raise IndexError(index)
        group_idx, target_rows, control_rows = self.set_index[index]
        group = self.samples[group_idx]
        perturbation = group.perturbation
        gene_columns = self._gene_columns_for(perturbation, index)
        if perturbation not in self.gene_to_id:
            raise ValueError(f"perturbation {perturbation!r} is absent from the model vocabulary")
        return OpenXCellSample(
            x_control=torch.from_numpy(
                expression_rows_columns(
                    self.matrix,
                    control_rows,
                    gene_columns,
                    backed=self.expression_backed,
                )
            ),
            x_target=torch.from_numpy(
                expression_rows_columns(
                    self.matrix,
                    target_rows,
                    gene_columns,
                    backed=self.expression_backed,
                )
            ),
            gene_ids=torch.from_numpy(self.model_gene_ids[gene_columns]),
            perturbation_id=self.gene_to_id[perturbation],
            perturbation=perturbation,
            context=group.context,
            batch=group.batch,
            control_indices=control_rows,
            target_indices=target_rows,
        )


def collate_samples(samples: list[OpenXCellSample]) -> OpenXCellBatch:
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    return OpenXCellBatch(
        x_control=torch.stack([sample.x_control for sample in samples]),
        x_target=torch.stack([sample.x_target for sample in samples]),
        gene_ids=torch.stack([sample.gene_ids for sample in samples]),
        perturbation_ids=torch.tensor([sample.perturbation_id for sample in samples], dtype=torch.long),
        perturbations=tuple(sample.perturbation for sample in samples),
        contexts=tuple(sample.context for sample in samples),
        batches=tuple(sample.batch for sample in samples),
    )
