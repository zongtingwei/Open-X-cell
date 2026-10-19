import json
from dataclasses import replace
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import torch

from open_x_cell.config import tiny_config
from open_x_cell.model import OpenXCellModel
from open_x_cell.priors import PRIOR_SOURCES
from open_x_cell.training import load_checkpoint, save_checkpoint
from open_x_cell.synthetic import make_synthetic_adata
from open_x_cell.inference import predict_adata


@pytest.mark.parametrize("corrupt", [False, True])
def test_hf_download_verifies_before_install(source, tmp_path, monkeypatch, corrupt):
    import shutil
    import huggingface_hub
    from open_x_cell.bundle import BUNDLE_FILES, export_bundle, verify_bundle
    from open_x_cell.download import download_bundle
    checkpoint, priors = source
    remote = export_bundle(checkpoint, tmp_path / "remote", priors, trust_legacy_pickle=True)
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        shutil.copytree(remote, kwargs["local_dir"], dirs_exist_ok=True)
        if corrupt:
            (Path(kwargs["local_dir"]) / "genes.json").write_text("[]")
        return str(kwargs["local_dir"])

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    target = tmp_path / "installed"
    options = dict(repo="test/model", revision="fixed-commit", token="test-token")
    if corrupt:
        with pytest.raises(ValueError, match="size|checksum"):
            download_bundle(target, **options)
        assert not target.exists()
    else:
        assert download_bundle(target, **options) == target
        verify_bundle(target)
        with pytest.raises(FileExistsError):
            download_bundle(target, **options)
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "test/model"
    assert calls[0]["revision"] == "fixed-commit"
    assert calls[0]["token"] == "test-token"
    assert set(calls[0]["allow_patterns"]) == BUNDLE_FILES | {"manifest.json"}
    assert not list(tmp_path.glob(".xcell-download-*"))


@pytest.fixture
def source(tmp_path):
    config = replace(tiny_config(), num_genes=24)
    torch.manual_seed(7)
    model = OpenXCellModel(config)
    checkpoint = tmp_path / "checkpoint.pt"
    genes = [f"GENE_{i}" for i in range(24)]
    save_checkpoint(checkpoint, model, config, genes)
    priors = {}
    for key in PRIOR_SOURCES:
        path = tmp_path / f"{key}.npz"
        np.savez(path, genes=np.array(["GENE_1", "OUTSIDE"]),
                 embeddings=np.ones((2, config.prior_dims[key]), dtype=np.float32))
        priors[key] = path
    return checkpoint, priors


def test_export_requires_trust_and_all_priors(source, tmp_path):
    from open_x_cell.bundle import export_bundle
    checkpoint, priors = source
    with pytest.raises(ValueError, match="trust"):
        export_bundle(checkpoint, tmp_path / "untrusted", priors)
    assert not (tmp_path / "untrusted").exists()
    with pytest.raises(ValueError, match="prior"):
        export_bundle(checkpoint, tmp_path / "incomplete", {}, trust_legacy_pickle=True)


def test_bundle_roundtrip_and_tampering(source, tmp_path):
    from open_x_cell.bundle import export_bundle, verify_bundle
    checkpoint, priors = source
    bundle = export_bundle(checkpoint, tmp_path / "bundle", priors, trust_legacy_pickle=True)
    manifest = verify_bundle(bundle)
    assert len(manifest["files"]) == 8
    original, _, genes = load_checkpoint(checkpoint)
    restored, _, restored_genes = load_checkpoint(bundle / "model.safetensors")
    assert restored_genes == genes
    for key, tensor in original.state_dict().items():
        assert torch.equal(restored.state_dict()[key], tensor)
    with np.load(bundle / "priors/esm.npz", allow_pickle=False) as data:
        assert data["genes"].tolist() == ["GENE_1"]
    with pytest.raises(FileExistsError):
        export_bundle(checkpoint, bundle, priors, trust_legacy_pickle=True)
    (bundle / "genes.json").write_text('[]')
    with pytest.raises(ValueError, match="checksum|size"):
        verify_bundle(bundle)


def test_manifest_rejects_missing_prior_and_path_escape(source, tmp_path):
    from open_x_cell.bundle import export_bundle, verify_bundle
    checkpoint, priors = source
    bundle = export_bundle(checkpoint, tmp_path / "bundle", priors, trust_legacy_pickle=True)
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["files"]["priors/esm.npz"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="files|prior"):
        verify_bundle(bundle)
    manifest["files"]["../outside"] = {"sha256": "0" * 64, "bytes": 0}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify_bundle(bundle)


def test_portable_prediction_matches_legacy_and_protects_output(source, tmp_path):
    from open_x_cell.bundle import export_bundle
    from open_x_cell import OpenXCell
    checkpoint, priors = source
    bundle = export_bundle(checkpoint, tmp_path / "bundle", priors, trust_legacy_pickle=True)
    data = make_synthetic_adata(n_cells=32, n_genes=24, n_perturbations=2)
    controls = data[data.obs["is_control"].to_numpy()].copy()
    control_path = tmp_path / "controls.h5ad"
    controls.write_h5ad(control_path)
    legacy = predict_adata(checkpoint, control_path, "GENE_1", tmp_path / "legacy.h5ad",
                           prior_files=priors, n_cells=4, seed=11, max_genes=8)
    model = OpenXCell.from_pretrained(bundle)
    output = tmp_path / "portable.h5ad"
    result = model.predict(control_path, "GENE_1", output, input_space="log1p-cp10k",
                           n_cells=4, seed=11, max_genes=8)
    np.testing.assert_array_equal(result.X, legacy.X)
    assert result.var_names.tolist() == legacy.var_names.tolist()
    assert result.obs["source_control_obs_name"].tolist() == legacy.obs["source_control_obs_name"].tolist()
    with pytest.raises(FileExistsError):
        model.predict(control_path, "GENE_1", output, input_space="log1p-cp10k")
    assert ad.read_h5ad(output).shape == (4, 8)


def test_control_only_and_input_validation(source, tmp_path):
    from open_x_cell.bundle import export_bundle
    from open_x_cell import OpenXCell
    checkpoint, priors = source
    bundle = export_bundle(checkpoint, tmp_path / "bundle", priors, trust_legacy_pickle=True)
    model = OpenXCell.from_pretrained(bundle)
    data = make_synthetic_adata(n_cells=32, n_genes=24, n_perturbations=2)
    path = tmp_path / "mixed.h5ad"
    data.write_h5ad(path)
    with pytest.raises(ValueError, match="control-only"):
        model.predict(path, "GENE_1", tmp_path / "bad.h5ad", input_space="log1p-cp10k")
    controls = data[data.obs["is_control"].to_numpy()].copy()
    controls.X[0, 0] = np.nan
    controls.write_h5ad(path)
    with pytest.raises(ValueError, match="finite"):
        model.predict(path, "GENE_1", tmp_path / "bad.h5ad", input_space="counts")


def test_archive_rejects_traversal(tmp_path):
    import io
    import tarfile
    from open_x_cell.download import extract_bundle
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("../escaped")
        member.size = 1
        tar.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="archive"):
        extract_bundle(archive, tmp_path / "destination")
    assert not (tmp_path / "escaped").exists()


def test_portable_weights_initialize_training(source, tmp_path):
    from open_x_cell.bundle import export_bundle
    from open_x_cell.training import _initialize_from_xcell_checkpoint
    checkpoint, priors = source
    bundle = export_bundle(checkpoint, tmp_path / "bundle", priors, trust_legacy_pickle=True)
    original, config, genes = load_checkpoint(checkpoint)
    destination = OpenXCellModel(config)
    _initialize_from_xcell_checkpoint(destination, bundle / "model.safetensors", genes, config)
    for key, value in original.state_dict().items():
        assert torch.equal(destination.state_dict()[key], value)


def test_counts_layer_normalizes_all_columns_before_alignment(source, tmp_path):
    from scipy.sparse import csr_matrix
    import pandas as pd
    from open_x_cell.bundle import export_bundle
    from open_x_cell import OpenXCell
    checkpoint, priors = source
    bundle = export_bundle(checkpoint, tmp_path / "bundle", priors, trust_legacy_pickle=True)
    # 24 measured model genes each have one count; an unmatched gene has 100.
    # Dropping that unmatched column before normalization would use 24, not 124.
    counts = np.ones((4, 25), dtype=np.float32)
    counts[:, -1] = 100
    controls = ad.AnnData(np.zeros_like(counts),
                         var=pd.DataFrame(index=[f"GENE_{i}" for i in range(24)] + ["UNMATCHED"]))
    controls.layers["counts"] = csr_matrix(counts)
    path = tmp_path / "raw.h5ad"
    controls.write_h5ad(path)
    manual = controls.copy()
    manual.X = np.full((4, 25), np.log1p(10000 / 124), dtype=np.float32)
    manual.X[:, -1] = np.log1p(1000000 / 124)
    manual.obs["is_control"] = True
    manual_path = tmp_path / "manual.h5ad"
    manual.write_h5ad(manual_path)
    expected = predict_adata(checkpoint, manual_path, "GENE_1", tmp_path / "expected.h5ad",
                             prior_files=priors, n_cells=4, max_genes=8)
    actual = OpenXCell.from_pretrained(bundle).predict(
        path, "GENE_1", tmp_path / "actual.h5ad", input_space="counts", layer="counts",
        n_cells=4, max_genes=8)
    np.testing.assert_array_equal(actual.X, expected.X)


def test_archive_roundtrip_and_cli(source, tmp_path):
    import tarfile
    from open_x_cell.bundle import export_bundle, BUNDLE_FILES
    from open_x_cell.download import main as download_main
    from open_x_cell.api import main as infer_main
    checkpoint, priors = source
    bundle = export_bundle(checkpoint, tmp_path / "bundle", priors, trust_legacy_pickle=True)
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in BUNDLE_FILES | {"manifest.json"}:
            tar.add(bundle / name, arcname=name, recursive=False)
    restored = tmp_path / "restored"
    assert download_main(["--archive", str(archive), "--output", str(restored)]) == 0
    data = make_synthetic_adata(n_cells=32, n_genes=24, n_perturbations=2)
    data = data[data.obs["is_control"].to_numpy()].copy()
    path = tmp_path / "controls.h5ad"
    data.write_h5ad(path)
    output = tmp_path / "predictions.h5ad"
    assert infer_main(["--model", str(restored), "--control", str(path),
                       "--perturbation", "GENE_1", "--input-space", "log1p-cp10k",
                       "--n-cells", "4", "--max-genes", "8", "--output", str(output)]) == 0
    result = ad.read_h5ad(output)
    assert result.shape == (4, 8)
    assert np.isfinite(result.X).all()


def test_bfloat16_final_proposal_and_prediction_labels(source, tmp_path):
    checkpoint, priors = source
    data = make_synthetic_adata(n_cells=32, n_genes=24, n_perturbations=2)
    path = tmp_path / "data.h5ad"
    data.write_h5ad(path)
    result = predict_adata(checkpoint, path, "GENE_1", tmp_path / "bf16.h5ad",
                           prior_files=priors, n_cells=4, max_genes=8, precision="bfloat16",
                           diffusion_return_mode="final_full_proposal")
    assert result.X.dtype == np.float32
    assert np.isfinite(result.X).all()
    assert not result.obs["is_control"].any()


def test_target_alias_preserves_priors(source, tmp_path):
    from open_x_cell.inference import predict_benchmark
    checkpoint, priors = source
    data = make_synthetic_adata(n_cells=32, n_genes=24, n_perturbations=2)
    data.var["ensembl_id"] = [f"ENSG{i:06d}" for i in range(24)]
    control_rows = np.flatnonzero(data.obs["is_control"].to_numpy())[:4]
    data.obs["benchmark_control"] = np.isin(np.arange(data.n_obs), control_rows)
    path = tmp_path / "data.h5ad"
    data.write_h5ad(path)
    by_symbol = predict_adata(checkpoint, path, "GENE_1", tmp_path / "symbol.h5ad",
                              prior_files=priors, n_cells=4, max_genes=8)
    by_alias = predict_adata(checkpoint, path, "ENSG000001", tmp_path / "alias.h5ad",
                             prior_files=priors, n_cells=4, max_genes=8)
    np.testing.assert_array_equal(by_symbol.X, by_alias.X)
    predictions = []
    for target in ["GENE_1", "ENSG000001"]:
        targets = tmp_path / f"{target}.json"
        targets.write_text(json.dumps([target]))
        directory = tmp_path / f"benchmark-{target}"
        predict_benchmark(checkpoint, path, targets, directory,
                          prior_files=priors, n_cells=4, max_genes=8)
        predictions.extend(ad.read_h5ad(p) for p in directory.glob("*.h5ad"))
    assert len(predictions) == 2
    np.testing.assert_array_equal(predictions[0].X, predictions[1].X)
    assert all(not p.obs["is_control"].any() for p in predictions)


def test_curriculum_rejects_ignored_pretrained_checkpoint(tmp_path):
    from open_x_cell.cli import main
    with pytest.raises(ValueError, match="curriculum.*pretrained"):
        main(["train", "--adata", str(tmp_path / "unused.h5ad"), "--output", str(tmp_path / "run"),
              "--curriculum", "--pretrained-checkpoint", str(tmp_path / "checkpoint.pt")])


def test_example_generator_handles_newer_null_metadata(tmp_path):
    import h5py
    import pandas as pd
    import subprocess
    import sys
    data = ad.AnnData(np.ones((8, 3), dtype=np.float32),
                     obs=pd.DataFrame({"gene_target": ["Non-Targeting"] * 4 + ["STAT1"] * 4,
                                       "sample": ["batch1"] * 8}, index=[f"cell{i}" for i in range(8)]),
                     var=pd.DataFrame(index=["STAT1", "A", "B"]))
    data.uns["xatlas_processing"] = {"normalization": "log1p(CP10K)", "created_at_utc": "2026-08-30"}
    path = tmp_path / "source.h5ad"
    data.write_h5ad(path)
    with h5py.File(path, "a") as handle:
        group = handle["uns"].create_group("log1p")
        group.attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})
        value = group.create_dataset("base", data=h5py.Empty("f"))
        value.attrs.update({"encoding-type": "null", "encoding-version": "0.1.0"})
    output = tmp_path / "example"
    script = Path(__file__).resolve().parents[2] / "examples/prepare_hek293t.py"
    subprocess.run([sys.executable, str(script), "--source", str(path), "--output", str(output),
                    "--cells", "2"], check=True)
    controls = ad.read_h5ad(output / "controls.h5ad")
    observed = ad.read_h5ad(output / "observed.h5ad")
    assert controls.shape == observed.shape == (2, 3)
    assert not set(controls.obs_names) & set(observed.obs_names)
    np.testing.assert_array_equal(controls.X, np.ones((2, 3), dtype=np.float32))
