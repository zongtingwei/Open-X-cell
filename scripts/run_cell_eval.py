"""Run the official cell-eval CLI and record exact input/output provenance."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_OUTPUTS = ("pred_de.csv", "real_de.csv", "results.csv", "agg_results.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--real", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--expected-version", default="0.6.8")
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.num_threads <= 0:
        raise ValueError("num_threads must be positive")
    pred = Path(args.pred).resolve()
    real = Path(args.real).resolve()
    metadata_target = Path(args.metadata_output).resolve()
    metadata_temporary = metadata_target.with_suffix(metadata_target.suffix + ".tmp")
    if pred == real or {metadata_target, metadata_temporary} & {pred, real}:
        raise ValueError(
            "pred, real, metadata, and the metadata temporary path must not collide"
        )
    for path in (pred, real):
        if not path.is_file():
            raise FileNotFoundError(path)
    outdir = Path(args.outdir).resolve()
    if outdir.exists() and not outdir.is_dir():
        raise FileExistsError(f"cell-eval output path exists and is not a directory: {outdir}")
    reserved_outputs = {outdir / name for name in EXPECTED_OUTPUTS}
    if reserved_outputs & {pred, real}:
        raise ValueError("cell-eval output files must not overwrite pred or real inputs")
    if (
        outdir in {metadata_target, metadata_temporary}
        or {metadata_target, metadata_temporary} & reserved_outputs
    ):
        raise ValueError(
            "metadata output and temporary path must not collide with the cell-eval "
            "output directory/files"
        )
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"cell-eval output directory is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    version = importlib.metadata.version("cell-eval")
    if version != args.expected_version:
        raise ValueError(
            f"installed cell-eval version is {version}, expected {args.expected_version}"
        )
    command = [
        "cell-eval",
        "run",
        "-ap",
        str(pred),
        "-ar",
        str(real),
        "--pert-col",
        "perturbation",
        "--control-pert",
        "control",
        "--profile",
        "full",
        "--num-threads",
        str(args.num_threads),
        "--outdir",
        str(outdir),
    ]
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    subprocess.run(command, check=True)
    elapsed = time.monotonic() - started

    outputs: dict[str, dict[str, str | int]] = {}
    for name in EXPECTED_OUTPUTS:
        path = outdir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"cell-eval did not produce a non-empty {path}")
        outputs[name] = {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }

    metadata = {
        "status": "complete",
        "cell_eval_version": version,
        "profile": "full",
        "perturbation_column": "perturbation",
        "control_perturbation": "control",
        "num_threads": int(args.num_threads),
        "command": command,
        "started_at_utc": started_at,
        "elapsed_seconds": elapsed,
        "pred": {"path": str(pred), "sha256": _sha256(pred)},
        "real": {"path": str(real), "sha256": _sha256(real)},
        "outputs": outputs,
        "runner_sha256": _sha256(Path(__file__).resolve()),
    }
    target = metadata_target
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata_temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    metadata_temporary.replace(target)
    print(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
