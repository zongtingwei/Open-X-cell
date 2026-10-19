import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_make_synthetic_train_predict_evaluate_smoke(tmp_path):
    adata = tmp_path / "synthetic.h5ad"
    run_dir = tmp_path / "run"
    pred = tmp_path / "pred.h5ad"
    metrics = tmp_path / "metrics.json"
    env = dict(os.environ)
    source_dir = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        [source_dir, env["PYTHONPATH"]] if env.get("PYTHONPATH") else [source_dir]
    )

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
        env=env,
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
        env=env,
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
        env=env,
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
        env=env,
    )
    payload = json.loads(metrics.read_text())
    assert "mae" in payload
