# Open-X-Cell

Open-X-Cell is a clean-room PyTorch reproduction of the X-Cell architecture from
`/Users/weizongting/Desktop/CIFM_benchmark/CIFM_benchmark_paper/X-Cell.pdf`.
It does not copy the official implementation and does not claim numerical
equivalence to unreleased Xaira weights.

## Open-X-Cell Clean-Room Reproduction

The implemented local workflow includes:

- `tiny`, `mini`, and `ultra` model configs.
- Diffusion reveal masks and four-step cumulative remasking.
- Cross-attention prior loading for ESM-2, STRING, GenePT, DepMap, and JUMP
  Cell Painting embeddings from CSV/TSV/Parquet/NPZ tables.
- A Transformer model with RMSNorm/QK-Norm/SwiGLU/tied-output support.
- X-Cell-style objective terms, including contrastive target-cell embedding and
  prior reconstruction terms when the required targets are available.
- scGPT warm-start (`--scgpt-warm-start`): initializes the gene embedding, value
  encoder, and transformer blocks from a pretrained scGPT checkpoint, with
  Ensembl→symbol mapping, an overlap check (`--scgpt-min-overlap`, optionally
  hard-failing via `--scgpt-strict-overlap`), and `<pad>` fallback.
- CP10k/log1p AnnData preprocessing and set construction stratified by
  `(context, perturbation, batch)` with context-level NTC fallback.
- Synthetic AnnData generation.
- Training, prediction, and evaluation CLI commands (including `evaluate-batch`
  for cross-perturbation Centroid Accuracy) with `tiny`/`mini`/`ultra` sizes.

Deliberate clean-room boundaries: this repository implements scGPT warm-start but
does **not** ship pretrained scGPT/Xaira weights or the original production
vocabulary remap tables (bring your own scGPT checkpoint dir), nor does it ship
FSDP2 orchestration or large-cluster activation checkpointing.

Run tests for the Open-X-Cell package:

```bash
PYTHONPATH=src pytest tests/open_x_cell -v
```

Run a local smoke workflow:

```bash
PYTHONPATH=src python -m open_x_cell.cli make-synthetic --output data/synthetic.h5ad --n-cells 128 --n-genes 32
PYTHONPATH=src python -m open_x_cell.cli train --adata data/synthetic.h5ad --output runs/open_x_cell_tiny --model-size tiny --normalize none --max-steps 4
PYTHONPATH=src python -m open_x_cell.cli predict --checkpoint runs/open_x_cell_tiny/checkpoint.pt --adata data/synthetic.h5ad --perturbation GENE_1 --output runs/open_x_cell_tiny/pred.h5ad
PYTHONPATH=src python -m open_x_cell.cli evaluate --truth data/synthetic.h5ad --pred runs/open_x_cell_tiny/pred.h5ad --output runs/open_x_cell_tiny/metrics.json
```

Training defaults to `--normalize cp10k-log1p` for raw-count AnnData; pass
`--normalize none` for already normalized inputs such as the synthetic smoke
data above. Prior tables can be provided with `--esm-priors`,
`--string-priors`, `--genept-priors`, `--depmap-priors`, and
`--cell-painting-priors`; each table must have a gene column plus the
paper-specified embedding width for that source.
