from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd

from open_x_cell.data import OpenXCellDataset, expression_rows_columns, load_adata
from open_x_cell.synthetic import make_synthetic_adata
from open_x_cell.training import train_model


def _dense_source() -> ad.AnnData:
    values = np.arange(8 * 6, dtype=np.float32).reshape(8, 6)
    obs = pd.DataFrame(
        {
            "perturbation": ["control"] * 4 + ["A"] * 2 + ["B"] * 2,
            "is_control": [True] * 4 + [False] * 4,
            "context": ["ctx"] * 8,
            "batch": ["batch"] * 8,
        },
        index=[f"cell_{index}" for index in range(8)],
    )
    var = pd.DataFrame(index=pd.Index(["A", "B", "C", "D", "E", "F"], name="gene"))
    return ad.AnnData(X=values, obs=obs, var=var)


def test_backed_dense_rows_restore_unsorted_duplicates_and_dynamic_target_first(tmp_path):
    source = _dense_source()
    path = tmp_path / "dense.h5ad"
    source.write_h5ad(path)
    backed = load_adata(str(path), normalize="none", backed=True)
    try:
        rows = np.asarray([6, 1, 6, 0], dtype=np.int64)
        columns = np.asarray([5, 2, 0], dtype=np.int64)
        actual = expression_rows_columns(backed.X, rows, columns, backed=True)
        np.testing.assert_array_equal(actual, source.X[rows][:, columns])

        dataset = OpenXCellDataset(
            backed,
            set_size=4,
            context_length=3,
            seed=17,
            vocabulary=list(source.var_names),
        )
        assert dataset.expression_backed is True
        panels = []
        for epoch in range(4):
            dataset.set_epoch(epoch)
            sample = dataset[0]
            panels.append(tuple(sample.gene_ids.tolist()))
            assert sample.gene_ids[0].item() == 0  # target A forced after CLS
            assert len(set(sample.gene_ids.tolist())) == 3
            assert sample.x_control.shape == (4, 3)
            assert sample.x_target.shape == (4, 3)
        assert len(set(panels)) > 1
    finally:
        backed.file.close()


def test_backed_train_records_runtime_context_without_changing_architecture(tmp_path):
    source = make_synthetic_adata(n_cells=64, n_genes=32, n_perturbations=2, seed=9)
    path = tmp_path / "training.h5ad"
    source.write_h5ad(path)
    output = tmp_path / "run"
    result = train_model(
        path,
        output,
        model_size="tiny",
        max_steps=1,
        normalize="none",
        batch_size=1,
        set_size=4,
        num_workers=0,
        backed=True,
        training_context_length=8,
    )
    assert result.checkpoint.is_file()
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["backed"] is True
    assert manifest["requested_training_context_length"] == 8
    assert manifest["training_context_length"] == 8
    assert manifest["model_config"]["context_length"] == 16
