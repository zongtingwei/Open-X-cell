from __future__ import annotations

import argparse
from pathlib import Path

from open_x_cell.config import LossConfig
from open_x_cell.evaluation import evaluate_prediction_files
from open_x_cell.inference import predict_adata
from open_x_cell.synthetic import make_synthetic_adata
from open_x_cell.training import train_model


def _add_prior_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--esm-priors", default=None)
    parser.add_argument("--string-priors", default=None)
    parser.add_argument("--genept-priors", default=None)
    parser.add_argument("--depmap-priors", default=None)
    parser.add_argument("--cell-painting-priors", default=None)


def _prior_files_from_args(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "esm": args.esm_priors,
        "string": args.string_priors,
        "genept": args.genept_priors,
        "depmap": args.depmap_priors,
        "cell_painting": args.cell_painting_priors,
    }


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
    train.add_argument("--model-size", choices=("tiny", "mini", "ultra"), default="tiny")
    train.add_argument("--normalize", choices=("none", "cp10k-log1p"), default="cp10k-log1p")
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--set-size", type=int, default=None)
    train.add_argument("--learning-rate", type=float, default=None)
    train.add_argument("--weight-decay", type=float, default=None)
    train.add_argument("--device", default=None)
    train.add_argument("--scgpt-init", default=None,
                       help="Path to a gene->embedding table (.npz with genes/embeddings) "
                            "to initialize gene embeddings from scGPT. Requires hidden_dim to match.")
    train.add_argument("--scgpt-warm-start", default=None,
                       help="Path to a pretrained scGPT model dir (vocab.json + best_model.pt) "
                            "for full warm-start: gene embedding + value encoder + transformer layers.")
    train.add_argument("--gene-id-map", default=None,
                       help="Path to an Ensembl->symbol mapping (.pt) used by scGPT warm-start.")
    train.add_argument("--scgpt-min-overlap", type=float, default=0.95,
                       help="Minimum gene-vocab overlap for scGPT warm-start (warns below).")
    train.add_argument("--l1-output", type=float, default=None,
                       help="Weight of the L1 output regularization term (paper: Mini=1.0).")
    _add_prior_args(train)

    predict = sub.add_parser("predict")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--adata", required=True)
    predict.add_argument("--perturbation", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--seed", type=int, default=0)
    predict.add_argument("--n-cells", type=int, default=64,
                         help="Control cells sampled per prediction set (paper default 64).")
    _add_prior_args(predict)

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
        loss_weights = LossConfig(l1_output=args.l1_output) if args.l1_output is not None else None
        train_model(
            args.adata,
            args.output,
            model_size=args.model_size,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            set_size=args.set_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            normalize=args.normalize,
            prior_files=_prior_files_from_args(args),
            scgpt_init=args.scgpt_init,
            scgpt_warm_start=args.scgpt_warm_start,
            gene_id_map=args.gene_id_map,
            scgpt_min_overlap=args.scgpt_min_overlap,
            loss_weights=loss_weights,
        )
        return 0
    if args.command == "predict":
        predict_adata(
            args.checkpoint,
            args.adata,
            args.perturbation,
            args.output,
            seed=args.seed,
            prior_files=_prior_files_from_args(args),
            n_cells=args.n_cells,
        )
        return 0
    if args.command == "evaluate":
        evaluate_prediction_files(args.truth, args.pred, args.output)
        return 0
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
