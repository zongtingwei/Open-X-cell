# Open-X-Cell

Open-X-Cell is a clean-room PyTorch reproduction of the X-Cell architecture from
`/Users/weizongting/Desktop/CIFM_benchmark/CIFM_benchmark_paper/X-Cell.pdf`.
It does not copy the official implementation and does not claim numerical
equivalence to unreleased Xaira weights.

## Open-X-Cell Clean-Room Reproduction

The implemented local workflow includes:

- `tiny`, `mini`, and `ultra` model configs.
- Diffusion reveal masks and four-step cumulative remasking.
- Cross-attention prior containers for ESM-2, STRING, GenePT, DepMap, and JUMP
  Cell Painting embeddings.
- A Transformer model with RMSNorm/QK-Norm/SwiGLU/tied-output support.
- X-Cell-style objective terms.
- Synthetic AnnData generation.
- Tiny CPU training, prediction, and evaluation CLI commands.

Run tests for the Open-X-Cell package:

```bash
PYTHONPATH=src pytest tests/open_x_cell -v
```

Run a local smoke workflow:

```bash
PYTHONPATH=src python -m open_x_cell.cli make-synthetic --output data/synthetic.h5ad --n-cells 128 --n-genes 32
PYTHONPATH=src python -m open_x_cell.cli train --adata data/synthetic.h5ad --output runs/open_x_cell_tiny --max-steps 4
PYTHONPATH=src python -m open_x_cell.cli predict --checkpoint runs/open_x_cell_tiny/checkpoint.pt --adata data/synthetic.h5ad --perturbation GENE_1 --output runs/open_x_cell_tiny/pred.h5ad
PYTHONPATH=src python -m open_x_cell.cli evaluate --truth data/synthetic.h5ad --pred runs/open_x_cell_tiny/pred.h5ad --output runs/open_x_cell_tiny/metrics.json
```

## Existing Lingshu-Cell Reproduction

This repository is a paper-level reproduction of **Lingshu-Cell: A generative cellular world model for transcriptome modeling toward virtual cells**.

The implementation follows the local paper at `/Users/weizongting/Desktop/CIFM_benchmark/CIFM_benchmark_paper/lingshu-cell.pdf`. The public `alibaba-damo-academy/Lingshu-Cell` repository was checked and is currently empty, so this project implements the method from the paper rather than copying unreleased author code.

## Scope

Implemented scope:

- UMI-count quantization into discrete expression tokens.
- Masked discrete diffusion training objective.
- Embedding-space sequence compression and decompression.
- LLaMA-style bidirectional Transformer mask predictor.
- Conditional prefix tokens, classifier-free guidance, and prior injection.
- Data adapters and evaluation utilities for small reproducible workflows.

Not claimed:

- Exact leaderboard reproduction without the original large-scale data, private checkpoints, and A800-scale training setup.

## Quickstart

Run tests:

```bash
python3 -m pytest -q
```

The `tiny` profile is intended for local smoke tests. The `paper` profile exposes the reported architecture but is not suitable for CPU-only smoke runs.
