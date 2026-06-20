from __future__ import annotations

import argparse
from pathlib import Path

from open_x_cell.evaluation import evaluate_prediction_files
from open_x_cell.inference import predict_adata
from open_x_cell.synthetic import make_synthetic_adata
from open_x_cell.training import train_tiny


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="open-x-cell")
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("make-synthetic")
    make.add_argument("--output", required=True)
    make.add_argument("--n-cells", type=int, default=512)
    make.add_argument("--n-genes", type=int, default=128)
    make.add_argument("--n-perturbations", type=int, default=8)
    make.add_argument("--seed", type=int, default=0)

    train = sub.add_parser("train")
    train.add_argument("--adata", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--max-steps", type=int, default=None)

    predict = sub.add_parser("predict")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--adata", required=True)
    predict.add_argument("--perturbation", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--seed", type=int, default=0)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--truth", required=True)
    evaluate.add_argument("--pred", required=True)
    evaluate.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "make-synthetic":
        adata = make_synthetic_adata(
            n_cells=args.n_cells,
            n_genes=args.n_genes,
            n_perturbations=args.n_perturbations,
            seed=args.seed,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(output)
        return 0
    if args.command == "train":
        train_tiny(args.adata, args.output, max_steps=args.max_steps)
        return 0
    if args.command == "predict":
        predict_adata(args.checkpoint, args.adata, args.perturbation, args.output, seed=args.seed)
        return 0
    if args.command == "evaluate":
        evaluate_prediction_files(args.truth, args.pred, args.output)
        return 0
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
