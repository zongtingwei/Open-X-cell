from __future__ import annotations

import argparse
from pathlib import Path

from open_x_cell.config import LossConfig
from open_x_cell.evaluation import evaluate_batch, evaluate_prediction_files
from open_x_cell.inference import predict_adata, predict_benchmark
from open_x_cell.synthetic import make_synthetic_adata
from open_x_cell.training import train_model, train_model_curriculum


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
    train.add_argument(
        "--backed",
        action="store_true",
        help="Read a dense H5AD lazily from disk (requires --normalize none and --num-workers 0).",
    )
    train.add_argument(
        "--training-context-length",
        type=int,
        default=None,
        help="Runtime gene subsample length, independent of checkpoint architecture context metadata.",
    )
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--set-size", type=int, default=None)
    train.add_argument("--learning-rate", type=float, default=None)
    train.add_argument("--weight-decay", type=float, default=None)
    train.add_argument("--device", default=None)
    train.add_argument("--precision", choices=("float32", "bfloat16", "float16"), default=None)
    train.add_argument("--gradient-accumulation-steps", type=int, default=None)
    train.add_argument("--gradient-checkpointing", action="store_true")
    train.add_argument("--num-workers", type=int, default=None)
    train.add_argument("--checkpoint-every", type=int, default=None,
                       help="Write a resumable latest.pt every N optimizer steps (0 disables periodic writes).")
    train.add_argument("--log-every", type=int, default=None)
    train.add_argument("--epochs", type=int, default=None,
                       help="Train for this many dataset epochs; derives optimizer steps after DDP sharding.")
    train.add_argument("--resume", default=None, help="Resume model/optimizer/scheduler from latest.pt.")
    train.add_argument(
        "--pretrained-checkpoint",
        default=None,
        help=(
            "Warm-start model weights from an Open-X-Cell checkpoint for a new "
            "fine-tuning run; optimizer, scheduler, and history are reset."
        ),
    )
    train.add_argument("--vocabulary", default=None,
                       help="Global model gene vocabulary (JSON/PT/text), separate from measured AnnData genes.")
    train.add_argument("--scgpt-init", default=None,
                       help="Path to a gene->embedding table (.npz with genes/embeddings) "
                            "to initialize gene embeddings from scGPT. Requires hidden_dim to match.")
    train.add_argument("--scgpt-warm-start", default=None,
                       help="Path to a pretrained scGPT model dir (vocab.json + best_model.pt) "
                            "for the Appendix A.7 warm-start: gene embedding/norm + transformer "
                            "layers (the unmentioned value encoder is not transferred).")
    train.add_argument("--scgpt-embedding-only", action="store_true",
                       help="With --scgpt-warm-start, transfer only gene embeddings (paper Mini setting).")
    train.add_argument("--gene-id-map", default=None,
                       help="Path to an Ensembl->symbol mapping (.pt) used by scGPT warm-start.")
    train.add_argument("--scgpt-min-overlap", type=float, default=0.95,
                       help="Minimum gene-vocab overlap for scGPT warm-start (warns below).")
    train.add_argument("--scgpt-strict-overlap", action="store_true",
                       help="Fail (instead of warn) when scGPT gene overlap is below --scgpt-min-overlap.")
    train.add_argument("--curriculum", action="store_true",
                       help="Two-phase Ultra curriculum: Phase 1 on top high-effect perturbations, "
                            "Phase 2 on the full corpus with a fresh optimizer.")
    train.add_argument("--phase1-steps", type=int, default=None)
    train.add_argument("--phase2-steps", type=int, default=None)
    train.add_argument("--top-frac", type=float, default=0.05)
    train.add_argument("--phase2-lr", type=float, default=None)
    train.add_argument("--phase2-warmup", type=float, default=None)
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
    predict.add_argument("--context", default=None,
                         help="Target context: sample NTC controls from this context only.")
    predict.add_argument("--batch", default=None,
                         help="Target batch: further restrict NTC controls to this batch.")
    predict.add_argument("--n-sets", type=int, default=1,
                         help="Number of control sets to ensemble (averages NTC sampling noise).")
    predict.add_argument("--tta", action="store_true",
                         help="Test-time adaptation: MMD-only self-supervised adaptation on the "
                              "target NTC (cross-attention frozen) before predicting.")
    predict.add_argument("--tta-steps", type=int, default=200)
    predict.add_argument("--tta-lr", type=float, default=1e-5)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--precision", choices=("float32", "bfloat16", "float16"), default="float32")
    predict.add_argument("--max-genes", type=int, default=None,
                         help="Optional inference cap; default uses every overlapping checkpoint gene.")
    predict.add_argument(
        "--diffusion-return-mode",
        choices=("cumulative_scattered", "final_full_proposal"),
        default="cumulative_scattered",
        help="Resolve the paper's cumulative-state versus final-proposal return ambiguity.",
    )
    _add_prior_args(predict)

    benchmark = sub.add_parser("predict-benchmark")
    benchmark.add_argument("--checkpoint", required=True)
    benchmark.add_argument("--adata", required=True)
    benchmark.add_argument("--perturbations", required=True,
                           help="JSON list of held-out perturbation symbols.")
    benchmark.add_argument("--output", required=True,
                           help="Directory for one resumable prediction h5ad per perturbation.")
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--n-cells", type=int, default=32,
                           help="Fixed NTC cells used for every perturbation (paper default: 32).")
    benchmark.add_argument("--shard-index", type=int, default=0)
    benchmark.add_argument("--shard-count", type=int, default=1)
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument("--precision", choices=("float32", "bfloat16", "float16"),
                           default="float32")
    benchmark.add_argument("--max-genes", type=int, default=None)
    benchmark.add_argument("--overwrite", action="store_true")
    benchmark.add_argument(
        "--diffusion-return-mode",
        choices=("cumulative_scattered", "final_full_proposal"),
        default="cumulative_scattered",
    )
    _add_prior_args(benchmark)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--truth", required=True)
    evaluate.add_argument("--pred", required=True)
    evaluate.add_argument("--output", required=True)

    evaluate_batch = sub.add_parser("evaluate-batch")
    evaluate_batch.add_argument("--truth", required=True)
    evaluate_batch.add_argument("--pred", required=True, nargs="+",
                                help="One or more per-perturbation prediction h5ad files.")
    evaluate_batch.add_argument("--output", required=True)
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
        temporary = output.with_suffix(".tmp.h5ad")
        adata.write_h5ad(temporary)
        temporary.replace(output)
        return 0
    if args.command == "train":
        loss_weights = LossConfig(l1_output=args.l1_output) if args.l1_output is not None else None
        if args.curriculum:
            if args.pretrained_checkpoint is not None:
                raise ValueError("--curriculum does not support --pretrained-checkpoint")
            if args.epochs is not None or args.resume is not None:
                raise ValueError("--curriculum does not support --epochs or --resume; use phase step counts")
            if args.checkpoint_every not in {None, 0}:
                raise ValueError("periodic recovery checkpoints are not implemented for curriculum phases")
            if args.backed or args.training_context_length is not None:
                raise ValueError("curriculum training does not yet support backed/context overrides")
            train_model_curriculum(
                args.adata, args.output, model_size=args.model_size,
                phase1_steps=args.phase1_steps if args.phase1_steps is not None else (args.max_steps or 100),
                phase2_steps=args.phase2_steps if args.phase2_steps is not None else (args.max_steps or 100),
                top_frac=args.top_frac, phase2_learning_rate=args.phase2_lr,
                phase2_warmup_fraction=args.phase2_warmup, normalize=args.normalize,
                prior_files=_prior_files_from_args(args), device=args.device,
                scgpt_init=args.scgpt_init, scgpt_warm_start=args.scgpt_warm_start,
                gene_id_map=args.gene_id_map, scgpt_min_overlap=args.scgpt_min_overlap,
                scgpt_strict_overlap=args.scgpt_strict_overlap, loss_weights=loss_weights,
                batch_size=args.batch_size, set_size=args.set_size,
                learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                precision=args.precision,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                gradient_checkpointing=args.gradient_checkpointing,
                num_workers=args.num_workers, log_every=args.log_every,
                vocabulary_file=args.vocabulary,
                scgpt_embedding_only=args.scgpt_embedding_only,
            )
            return 0
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
            scgpt_strict_overlap=args.scgpt_strict_overlap,
            loss_weights=loss_weights,
            precision=args.precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            num_workers=args.num_workers,
            checkpoint_every=args.checkpoint_every,
            log_every=args.log_every,
            epochs=args.epochs,
            resume_from=args.resume,
            pretrained_checkpoint=args.pretrained_checkpoint,
            vocabulary_file=args.vocabulary,
            scgpt_embedding_only=args.scgpt_embedding_only,
            backed=args.backed,
            training_context_length=args.training_context_length,
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
            context=args.context,
            batch=args.batch,
            n_sets=args.n_sets,
            tta=args.tta,
            tta_steps=args.tta_steps,
            tta_lr=args.tta_lr,
            device=args.device,
            precision=args.precision,
            max_genes=args.max_genes,
            diffusion_return_mode=args.diffusion_return_mode,
        )
        return 0
    if args.command == "predict-benchmark":
        predict_benchmark(
            args.checkpoint,
            args.adata,
            args.perturbations,
            args.output,
            seed=args.seed,
            prior_files=_prior_files_from_args(args),
            n_cells=args.n_cells,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            device=args.device,
            precision=args.precision,
            max_genes=args.max_genes,
            overwrite=args.overwrite,
            diffusion_return_mode=args.diffusion_return_mode,
        )
        return 0
    if args.command == "evaluate":
        evaluate_prediction_files(args.truth, args.pred, args.output)
        return 0
    if args.command == "evaluate-batch":
        evaluate_batch(args.truth, args.pred, args.output)
        return 0
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
