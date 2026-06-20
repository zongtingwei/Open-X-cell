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
