import json
import subprocess
import sys


def test_cli_make_synthetic_train_predict_evaluate_smoke(tmp_path):
    adata = tmp_path / "synthetic.h5ad"
    run_dir = tmp_path / "run"
    pred = tmp_path / "pred.h5ad"
    metrics = tmp_path / "metrics.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "open_x_cell.cli",
            "make-synthetic",
            "--output",
            str(adata),
            "--n-cells",
            "64",
            "--n-genes",
            "24",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "open_x_cell.cli",
            "train",
            "--adata",
            str(adata),
            "--output",
            str(run_dir),
            "--model-size",
            "tiny",
            "--normalize",
            "none",
            "--max-steps",
            "2",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "open_x_cell.cli",
            "predict",
            "--checkpoint",
            str(run_dir / "checkpoint.pt"),
            "--adata",
            str(adata),
            "--perturbation",
            "GENE_1",
            "--output",
            str(pred),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "open_x_cell.cli",
            "evaluate",
            "--truth",
            str(adata),
            "--pred",
            str(pred),
            "--output",
            str(metrics),
        ],
        check=True,
    )
    payload = json.loads(metrics.read_text())
    assert "mae" in payload
