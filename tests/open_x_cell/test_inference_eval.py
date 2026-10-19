import numpy as np
import json
import os
import pytest
import subprocess
import sys
import torch
from pathlib import Path
from types import SimpleNamespace

from open_x_cell.config import TrainingConfig, tiny_config
from open_x_cell.evaluation import evaluate_batch
from open_x_cell.inference import (
    _aligned_gene_indices,
    _diffuse,
    _select_control_pool,
    predict_adata,
    predict_benchmark,
)
from open_x_cell.synthetic import make_synthetic_adata
from open_x_cell.training import train_tiny
from open_x_cell.priors import make_zero_priors


class _FakeObs:
    def __init__(self, d):
        self._d = d

    def __contains__(self, k):
        return k in self._d

    def __getitem__(self, k):
        import pandas as pd
        return pd.Series(self._d[k])


class _FakeAdata:
    def __init__(self, d):
        self.obs = _FakeObs(d)


def test_select_control_pool_filters_by_context_and_batch():
    rows = np.array([0, 1, 2, 3])
    adata = _FakeAdata({
        "context": ["a", "a", "b", "b"],
        "batch": ["x", "y", "x", "y"],
    })
    # context only
    np.testing.assert_array_equal(_select_control_pool(adata, rows, "a", None), [0, 1])
    # context + batch
    np.testing.assert_array_equal(_select_control_pool(adata, rows, "a", "y"), [1])
    # empty (context,batch) -> falls back to context pool
    np.testing.assert_array_equal(_select_control_pool(adata, rows, "b", "zzz"), [2, 3])
    # non-empty but undersized batch pool -> context pool
    np.testing.assert_array_equal(_select_control_pool(adata, rows, "a", "x", min_size=2), [0, 1])
    # an explicitly requested unknown context must not silently use another context
    with pytest.raises(ValueError, match="no control cells"):
        _select_control_pool(adata, rows, "nope", None)


def test_inference_resolves_unmeasured_perturbation_from_global_vocabulary():
    adata = make_synthetic_adata(n_cells=64, n_genes=32, n_perturbations=3, seed=4)
    vocabulary = ["UNMEASURED_TARGET"] + list(map(str, adata.var_names[:31]))

    _, checkpoint_ids, _, perturbation_id = _aligned_gene_indices(
        adata,
        vocabulary,
        tiny_config(),
        perturbation="UNMEASURED_TARGET",
        max_genes=None,
    )

    assert perturbation_id == 0
    assert 0 not in checkpoint_ids


def test_inference_rejects_duplicate_input_gene_names():
    adata = make_synthetic_adata(n_cells=32, n_genes=8, n_perturbations=2, seed=9)
    adata.var_names = ["duplicate", "duplicate", *map(str, adata.var_names[2:])]

    with pytest.raises(ValueError, match="gene names must be unique"):
        _aligned_gene_indices(
            adata,
            list(map(str, adata.var_names)),
            tiny_config(),
            perturbation=None,
            max_genes=None,
        )


def test_diffusion_returns_values_frozen_at_each_cumulative_step():
    class StepModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, x, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(expression=torch.full_like(x, float(self.calls)))

    config = tiny_config()
    model = StepModel()
    result = _diffuse(
        model,
        config,
        np.zeros((2, 8), dtype=np.float32),
        torch.arange(8)[None, :],
        torch.tensor([1]),
        make_zero_priors(1, config),
        seed=3,
        device=torch.device("cpu"),
        precision="float32",
    )

    values, counts = np.unique(result, return_counts=True)
    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(counts, [4, 4, 4, 4])


def test_diffusion_can_return_the_appendix_final_full_proposal_sensitivity():
    class StepModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, x, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(expression=torch.full_like(x, float(self.calls)))

    config = tiny_config()
    result = _diffuse(
        StepModel(),
        config,
        np.zeros((2, 8), dtype=np.float32),
        torch.arange(8)[None, :],
        torch.tensor([1]),
        make_zero_priors(1, config),
        seed=3,
        device=torch.device("cpu"),
        precision="float32",
        return_mode="final_full_proposal",
    )

    np.testing.assert_array_equal(result, np.full((2, 8), 4.0, dtype=np.float32))


def test_predict_ensemble_and_batch_eval(tmp_path):
    adata = make_synthetic_adata(n_cells=160, n_genes=tiny_config().num_genes,
                                 n_perturbations=4, seed=5)
    path = tmp_path / "data.h5ad"
    adata.write_h5ad(path)
    tcfg = TrainingConfig(batch_size=2, set_size=8, max_steps=4, learning_rate=1e-3, seed=0, device="cpu")
    ckpt = train_tiny(path, tmp_path / "out", training_config=tcfg)

    perts = [p for p in adata.obs["perturbation"].unique() if p != "control"][:2]
    source_bytes = path.read_bytes()
    with pytest.raises(ValueError, match="must not overwrite an input"):
        predict_adata(ckpt, path, str(perts[0]), path, n_cells=8)
    assert path.read_bytes() == source_bytes

    pred_paths = []
    for p in perts:
        out = tmp_path / f"pred_{p}.h5ad"
        # ensemble of 3 control sets -> 3 * n_cells predicted rows
        pred = predict_adata(ckpt, path, str(p), out, n_cells=8, n_sets=3)
        assert pred.n_obs == 24
        pred_paths.append(out)

    result = evaluate_batch(path, pred_paths, tmp_path / "summary.json")
    assert "centroid_accuracy" in result["summary"]
    assert result["summary"]["n_perturbations"] == 2
    assert len(result["per_perturbation"]) == 2
    assert "de_spearman_lfc" in next(iter(result["per_perturbation"].values()))


def test_predict_benchmark_uses_fixed_controls_and_resumable_shards(tmp_path):
    adata = make_synthetic_adata(
        n_cells=96,
        n_genes=tiny_config().num_genes,
        n_perturbations=3,
        seed=7,
    )
    controls = np.flatnonzero(adata.obs["is_control"].to_numpy(dtype=bool))
    benchmark_control = np.full(adata.n_obs, "false", dtype=object)
    benchmark_control[controls[:4]] = "true"
    adata.obs["benchmark_control"] = benchmark_control
    path = tmp_path / "data.h5ad"
    adata.write_h5ad(path)
    checkpoint = train_tiny(
        path,
        tmp_path / "run",
        training_config=TrainingConfig(
            batch_size=2,
            set_size=4,
            max_steps=1,
            learning_rate=1e-3,
            seed=0,
            device="cpu",
        ),
    )
    perturbations = [p for p in adata.obs["perturbation"].unique() if p != "control"]
    perturbation_path = tmp_path / "perturbations.json"
    perturbation_path.write_text(json.dumps(perturbations))
    output = tmp_path / "predictions"

    first = predict_benchmark(
        checkpoint,
        path,
        perturbation_path,
        output,
        n_cells=4,
        shard_index=0,
        shard_count=2,
    )
    second = predict_benchmark(
        checkpoint,
        path,
        perturbation_path,
        output,
        n_cells=4,
        shard_index=0,
        shard_count=2,
    )

    assert len(first["written"]) == 2
    assert len(second["skipped_existing"]) == 2
    assert len(list(output.glob("*.h5ad"))) == 2
    assert first["fixed_control_obs_names"] == adata.obs_names[controls[:4]].astype(str).tolist()
    assert first["inference_inputs"]["source_adata"]["path"] == str(path.resolve())
    assert len(first["inference_inputs"]["source_adata"]["sha256"]) == 64
    assert set(first["inference_inputs"]["prior_artifacts"]) == {
        "esm",
        "string",
        "genept",
        "depmap",
        "cell_painting",
    }
    first_prediction = next(output.glob("*.h5ad"))
    import anndata as ad

    prediction = ad.read_h5ad(first_prediction)
    assert prediction.uns["open_x_cell_prediction"]["checkpoint_sha256"] == first["checkpoint_sha256"]
    assert bool(
        prediction.uns["open_x_cell_prediction"]["perturbation_gene_token_present"]
    ) is True
    assert (
        prediction.uns["open_x_cell_prediction"]["diffusion_return_mode"]
        == "cumulative_scattered"
    )
    assert json.loads(
        prediction.uns["open_x_cell_prediction"]["inference_inputs_json"]
    ) == first["inference_inputs"]
    original_prediction = prediction.copy()
    prediction[:, ::-1].copy().write_h5ad(first_prediction)
    with pytest.raises(ValueError, match="different gene ordering"):
        predict_benchmark(
            checkpoint,
            path,
            perturbation_path,
            output,
            n_cells=4,
            shard_index=0,
            shard_count=2,
        )
    original_prediction.write_h5ad(first_prediction)
    prediction = original_prediction
    prediction.uns["open_x_cell_prediction"]["max_genes"] = 7
    prediction.write_h5ad(first_prediction)
    with pytest.raises(ValueError, match="stale or incomplete provenance"):
        predict_benchmark(
            checkpoint,
            path,
            perturbation_path,
            output,
            n_cells=4,
            shard_index=0,
            shard_count=2,
        )


def test_cell_eval_assembly_preserves_inference_input_provenance(tmp_path):
    adata = make_synthetic_adata(
        n_cells=80,
        n_genes=tiny_config().num_genes,
        n_perturbations=2,
        seed=11,
    )
    controls = np.flatnonzero(adata.obs["is_control"].to_numpy(dtype=bool))
    adata.obs["benchmark_control"] = np.isin(np.arange(adata.n_obs), controls[:4])
    truth_path = tmp_path / "truth.h5ad"
    adata.write_h5ad(truth_path)
    checkpoint = train_tiny(
        truth_path,
        tmp_path / "run",
        training_config=TrainingConfig(
            batch_size=1,
            set_size=4,
            max_steps=1,
            learning_rate=1e-3,
            device="cpu",
        ),
    )
    perturbations = [
        str(value) for value in adata.obs["perturbation"].unique() if value != "control"
    ]
    perturbation_path = tmp_path / "perturbations.json"
    perturbation_path.write_text(json.dumps(perturbations), encoding="utf-8")
    prediction_dir = tmp_path / "predictions"
    prediction_manifest = predict_benchmark(
        checkpoint,
        truth_path,
        perturbation_path,
        prediction_dir,
        n_cells=4,
    )
    pred_output = tmp_path / "cell_eval" / "pred.h5ad"
    real_output = tmp_path / "cell_eval" / "real.h5ad"
    metadata_output = tmp_path / "cell_eval" / "metadata.json"
    repository = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "assemble_cell_eval_inputs.py"),
            "--truth",
            str(truth_path),
            "--predictions",
            str(prediction_dir),
            "--perturbations",
            str(perturbation_path),
            "--pred-output",
            str(pred_output),
            "--real-output",
            str(real_output),
            "--metadata-output",
            str(metadata_output),
            "--n-controls",
            "4",
        ],
        cwd=repository,
        env={**os.environ, "PYTHONPATH": str(repository / "src")},
        check=True,
    )
    metadata = json.loads(metadata_output.read_text(encoding="utf-8"))
    assert metadata["inference_inputs"] == prediction_manifest["inference_inputs"]
    assert metadata["prediction_control_provenance_verified"] is True

    evaluation_reference = tmp_path / "evaluation_reference.h5ad"
    adata[:, :10].copy().write_h5ad(evaluation_reference)
    subset_pred = tmp_path / "cell_eval_subset" / "pred.h5ad"
    subset_real = tmp_path / "cell_eval_subset" / "real.h5ad"
    subset_metadata = tmp_path / "cell_eval_subset" / "metadata.json"
    subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "assemble_cell_eval_inputs.py"),
            "--truth",
            str(truth_path),
            "--predictions",
            str(prediction_dir),
            "--perturbations",
            str(perturbation_path),
            "--pred-output",
            str(subset_pred),
            "--real-output",
            str(subset_real),
            "--metadata-output",
            str(subset_metadata),
            "--n-controls",
            "4",
            "--evaluation-gene-reference",
            str(evaluation_reference),
        ],
        cwd=repository,
        env={**os.environ, "PYTHONPATH": str(repository / "src")},
        check=True,
    )
    subset_report = json.loads(subset_metadata.read_text(encoding="utf-8"))
    assert subset_report["evaluation_genes"] == 10
    assert subset_report["pred_shape"][1] == 10
    assert subset_report["real_shape"][1] == 10
    assert subset_report["evaluation_gene_reference"]["path"] == str(
        evaluation_reference.resolve()
    )


def test_cell_eval_runner_cannot_overwrite_an_input_with_a_reserved_output(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    outdir = tmp_path / "results"
    outdir.mkdir()
    pred = outdir / "results.csv"
    real = tmp_path / "real.h5ad"
    pred.write_bytes(b"prediction-input")
    real.write_bytes(b"real-input")

    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_cell_eval.py"),
            "--pred",
            str(pred),
            "--real",
            str(real),
            "--outdir",
            str(outdir),
            "--metadata-output",
            str(tmp_path / "evaluation.json"),
        ],
        cwd=repository,
        env={**os.environ, "PYTHONPATH": str(repository / "src")},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "must not overwrite pred or real inputs" in result.stderr
    assert pred.read_bytes() == b"prediction-input"
    assert real.read_bytes() == b"real-input"


def test_cell_eval_runner_cannot_use_an_input_as_metadata_temporary_path(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    metadata = tmp_path / "evaluation.json"
    pred = tmp_path / "evaluation.json.tmp"
    real = tmp_path / "real.h5ad"
    pred.write_bytes(b"prediction-input")
    real.write_bytes(b"real-input")

    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_cell_eval.py"),
            "--pred",
            str(pred),
            "--real",
            str(real),
            "--outdir",
            str(tmp_path / "results"),
            "--metadata-output",
            str(metadata),
        ],
        cwd=repository,
        env={**os.environ, "PYTHONPATH": str(repository / "src")},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "metadata temporary path must not collide" in result.stderr
    assert pred.read_bytes() == b"prediction-input"
    assert real.read_bytes() == b"real-input"


def test_real_subset_preparation_cannot_use_source_as_temporary_output(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    source = tmp_path / "subset.tmp.h5ad"
    output = tmp_path / "subset.h5ad"
    source.write_bytes(b"source-ann-data")

    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "prep_real_subset.py"),
            "--source",
            str(source),
            "--output",
            str(output),
        ],
        cwd=repository,
        env={**os.environ, "PYTHONPATH": str(repository / "src")},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "temporary paths must not overwrite" in result.stderr
    assert source.read_bytes() == b"source-ann-data"
    assert not output.exists()


def test_full_run_validator_cannot_overwrite_benchmark_input(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    truth = tmp_path / "truth.h5ad"
    perturbations = tmp_path / "perturbations.json"
    truth.write_bytes(b"truth-input")
    perturbations.write_text("[]", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "validate_replogle_full_run.py"),
            "--run-dir",
            str(tmp_path / "run"),
            "--truth",
            str(truth),
            "--perturbations",
            str(perturbations),
            "--output",
            str(truth),
        ],
        cwd=repository,
        env={**os.environ, "PYTHONPATH": str(repository / "src")},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "must not overwrite benchmark inputs" in result.stderr
    assert truth.read_bytes() == b"truth-input"
    assert perturbations.read_text(encoding="utf-8") == "[]"


def test_cell_eval_assembly_requires_distinct_temporary_and_final_outputs(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    truth = tmp_path / "truth.h5ad"
    perturbations = tmp_path / "perturbations.json"
    truth.write_bytes(b"truth-input")
    perturbations.write_text("[]", encoding="utf-8")
    pred_output = tmp_path / "assembled.h5ad"
    real_output = tmp_path / "assembled.tmp.h5ad"

    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "assemble_cell_eval_inputs.py"),
            "--truth",
            str(truth),
            "--predictions",
            str(tmp_path / "predictions"),
            "--perturbations",
            str(perturbations),
            "--pred-output",
            str(pred_output),
            "--real-output",
            str(real_output),
            "--metadata-output",
            str(tmp_path / "metadata.json"),
        ],
        cwd=repository,
        env={**os.environ, "PYTHONPATH": str(repository / "src")},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "temporary paths must all be distinct" in result.stderr
    assert truth.read_bytes() == b"truth-input"
    assert perturbations.read_text(encoding="utf-8") == "[]"
    assert not pred_output.exists()
    assert not real_output.exists()
