# Lingshu-Cell Reproduction

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

