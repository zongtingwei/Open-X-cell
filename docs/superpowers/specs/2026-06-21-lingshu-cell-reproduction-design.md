# Lingshu-Cell Reproduction Design

**Goal:** Build a paper-level, runnable reproduction of Lingshu-Cell from `lingshu-cell.pdf`, including the core model, training/sampling logic, scRNA-seq data adapters, and evaluation utilities.

**Source Basis:** Local paper `/Users/weizongting/Desktop/CIFM_benchmark/CIFM_benchmark_paper/lingshu-cell.pdf` plus a web check showing `alibaba-damo-academy/Lingshu-Cell` exists but is currently empty. The implementation therefore follows the paper, not unreleased author code.

**Scope:** Implement options 1 and 2 selected by the user: a runnable PyTorch reproduction and data/metric scripts. It will not claim reproduction of the paper leaderboard scores without the original large-scale data, checkpoints, and A800-scale training.

---

## Architecture

The project is a Python package named `lingshu_cell`. The model follows the paper's masked discrete diffusion model (MDDM): UMI counts are quantized into expression tokens, a forward process masks expression tokens at a sampled continuous timestep, and a bidirectional Transformer predicts the original tokens at masked positions.

The Transformer uses the paper's LLaMA-style blocks: RMSNorm, bidirectional multi-head self-attention with RoPE, SwiGLU feed-forward layers, and bias-free linear projections. Gene token embeddings are compressed by a fixed random permutation, grouped into blocks, projected down to compressed tokens, processed by the Transformer, then decompressed to gene-level representations before the output head. Condition tokens bypass compression and are prepended to the compressed gene sequence.

Data code provides lightweight AnnData, NumPy, CSV, and sparse Matrix Market support, including gene-list alignment and token dataset creation. Evaluation code implements unconditional metrics and a practical subset of perturbation metrics, with differential-expression hooks kept explicit because exact Cell-Eval/pdex behavior is external to the paper.

## Components

- `lingshu_cell.quantization`: Implements equations 3-8 from the paper. Counts `0..99` map exactly, counts `100..9999` map into decade bins, and values above `9999` map to overflow token `280`.
- `lingshu_cell.diffusion`: Implements forward masking, masked-token cross entropy, cosine unmask schedule, and random position selection for sampling.
- `lingshu_cell.model`: Implements the compressed bidirectional Transformer mask predictor.
- `lingshu_cell.conditioning`: Encodes source and perturbation labels as appended vocabulary tokens and protects prefix tokens from masking.
- `lingshu_cell.sampling`: Implements unconditional sampling, conditional classifier-free guidance, and biological prior injection with fixed prior positions.
- `lingshu_cell.data`: Loads and aligns expression matrices, then converts count matrices to token datasets.
- `lingshu_cell.metrics`: Implements Pearson, Spearman, MMD, 1-WD, iLISI, MAE, Pearson-delta, and PDS.
- `scripts/train.py`: Minimal training CLI for unconditional and conditional data.
- `scripts/sample.py`: Sampling CLI for unconditional, conditional, CFG, and prior-injected generation.
- `scripts/evaluate.py`: Metric CLI for generated vs. real expression matrices.

## Data Flow

Input expression data enters as raw UMI counts. The data layer optionally aligns genes to a reference gene list, fills missing genes with zeros, and quantizes counts into integer expression tokens. For conditional training, two prefix tokens are added for source context and perturbation identity. The diffusion layer samples `t ~ U(0, 1)`, masks expression positions with probability `t`, and leaves condition prefix positions unmasked. The model predicts logits for expression positions, and training computes cross entropy only where masking occurred.

Sampling starts from a fully masked expression sequence. For each reverse step, the model predicts categorical logits at all masked positions, the sampler chooses a scheduled subset of still-masked positions, samples token values, and keeps the rest masked. Conditional sampling evaluates both target and control labels and combines logits using the paper's CFG equation. Prior injection initializes selected gene positions to low expression and keeps them fixed throughout sampling.

## Configuration

Two model profiles are provided:

- `tiny`: Small dimensions for tests and local smoke runs.
- `paper`: Paper hyperparameters: gene length `18080`, embedding dimension `640`, `13` Transformer blocks, `10` attention heads, head dimension `64`, FFN dimension `2560`, compression group size `8` by default and `32` for genetic perturbation prediction.

Training defaults follow the paper where practical: AdamW, `beta1=0.9`, `beta2=0.95`, weight decay `0.1`, gradient clipping at `1.0`, cosine learning-rate decay with linear warmup, and optional EMA. Local smoke tests use the `tiny` profile.

## Error Handling

The implementation validates shapes, token ranges, condition-prefix lengths, gene-list compatibility, and metric inputs. Optional dependencies such as AnnData are imported only in the code paths that need them, so core model tests can run without large biology packages.

## Testing

Tests are written before implementation for the core behavioral contracts:

- Quantization maps exact low counts, decade bins, overflow, and inverse bin representatives correctly.
- Diffusion masks only allowed expression positions and computes loss only on masked tokens.
- Sequence compression/decompression preserves gene length and keeps condition prefixes outside the compression path.
- CFG combines conditional and control logits with the paper's formula.
- Biological prior injection fixes selected gene positions during sampling.
- Data alignment preserves requested gene order and fills missing genes with zeros.
- Metrics return known values on small synthetic matrices.

## Known Limits

The paper does not provide author source code, checkpoints, exact random seeds, dataset manifests, or all external Cell-Eval/pdex implementation details. This repository reproduces the algorithmic structure and runnable workflow from the paper, while making missing engineering choices explicit and configurable.

