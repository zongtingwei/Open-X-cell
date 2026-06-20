# Lingshu-Cell Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, runnable Python reproduction of Lingshu-Cell's model, training, sampling, data adapters, and metrics.

**Architecture:** A focused Python package implements paper-defined components as independent modules. Tests drive the core contracts first, then scripts wire the modules into runnable workflows.

**Tech Stack:** Python 3.13, PyTorch, NumPy, SciPy/scikit-learn/AnnData when available, pytest.

---

## File Structure

- Create `pyproject.toml`: package metadata, dependencies, pytest config.
- Create `README.md`: reproduction scope, quickstart, and limitations.
- Create `lingshu_cell/__init__.py`: package exports.
- Create `lingshu_cell/config.py`: model/training dataclasses and paper/tiny profiles.
- Create `lingshu_cell/quantization.py`: UMI count tokenization and approximate inverse mapping.
- Create `lingshu_cell/diffusion.py`: masking, masked loss, and sampling schedules.
- Create `lingshu_cell/model.py`: compressed bidirectional Transformer mask predictor.
- Create `lingshu_cell/conditioning.py`: condition vocabulary and prefix helpers.
- Create `lingshu_cell/sampling.py`: unconditional, CFG, and prior-injected sampling.
- Create `lingshu_cell/data.py`: matrix loading, gene alignment, token dataset.
- Create `lingshu_cell/metrics.py`: unconditional and perturbation metrics.
- Create `scripts/train.py`: training CLI.
- Create `scripts/sample.py`: sampling CLI.
- Create `scripts/evaluate.py`: evaluation CLI.
- Create `tests/`: focused behavior tests.

## Tasks

### Task 1: Project Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `lingshu_cell/__init__.py`
- Create: `lingshu_cell/config.py`

- [ ] Write initial tests for config profiles and imports.
- [ ] Run `python3 -m pytest tests/test_config.py -q`; expected failure because package files do not exist yet.
- [ ] Add package skeleton and config dataclasses.
- [ ] Run `python3 -m pytest tests/test_config.py -q`; expected pass.

### Task 2: Quantization

**Files:**
- Create: `lingshu_cell/quantization.py`
- Create: `tests/test_quantization.py`

- [ ] Test count bins `0`, `1`, `99`, `100`, `109`, `190`, `999`, `1000`, `9999`, and overflow.
- [ ] Run `python3 -m pytest tests/test_quantization.py -q`; expected failure because functions are missing.
- [ ] Implement vectorized quantization and approximate inverse representatives.
- [ ] Run `python3 -m pytest tests/test_quantization.py -q`; expected pass.

### Task 3: Diffusion

**Files:**
- Create: `lingshu_cell/diffusion.py`
- Create: `tests/test_diffusion.py`

- [ ] Test masking probability, protected prefix positions, loss on masked tokens only, and cosine schedule counts.
- [ ] Run `python3 -m pytest tests/test_diffusion.py -q`; expected failure.
- [ ] Implement forward masking, masked cross entropy, and schedule utilities.
- [ ] Run `python3 -m pytest tests/test_diffusion.py -q`; expected pass.

### Task 4: Model

**Files:**
- Create: `lingshu_cell/model.py`
- Create: `tests/test_model.py`

- [ ] Test tiny model output shape, prefix bypass, compression padding, deterministic permutation buffer, and paper profile values.
- [ ] Run `python3 -m pytest tests/test_model.py -q`; expected failure.
- [ ] Implement RMSNorm, RoPE, SwiGLU, bidirectional attention, compression/decompression, and mask predictor.
- [ ] Run `python3 -m pytest tests/test_model.py -q`; expected pass.

### Task 5: Conditioning and Sampling

**Files:**
- Create: `lingshu_cell/conditioning.py`
- Create: `lingshu_cell/sampling.py`
- Create: `tests/test_conditioning_sampling.py`

- [ ] Test condition-token allocation, CFG formula, prior initialization, fixed prior positions, and a tiny-model smoke sample.
- [ ] Run `python3 -m pytest tests/test_conditioning_sampling.py -q`; expected failure.
- [ ] Implement condition vocabulary and samplers.
- [ ] Run `python3 -m pytest tests/test_conditioning_sampling.py -q`; expected pass.

### Task 6: Data and Metrics

**Files:**
- Create: `lingshu_cell/data.py`
- Create: `lingshu_cell/metrics.py`
- Create: `tests/test_data_metrics.py`

- [ ] Test gene alignment, missing-gene zero fill, token dataset output, Pearson/Spearman/MMD/1-WD/iLISI, MAE, Pearson-delta, and PDS on toy matrices.
- [ ] Run `python3 -m pytest tests/test_data_metrics.py -q`; expected failure.
- [ ] Implement loaders, alignment, dataset, and metrics.
- [ ] Run `python3 -m pytest tests/test_data_metrics.py -q`; expected pass.

### Task 7: CLI Scripts

**Files:**
- Create: `scripts/train.py`
- Create: `scripts/sample.py`
- Create: `scripts/evaluate.py`
- Create: `tests/test_cli.py`

- [ ] Test `--help` for all scripts and a tiny synthetic train/sample/evaluate smoke path.
- [ ] Run `python3 -m pytest tests/test_cli.py -q`; expected failure.
- [ ] Implement CLI scripts.
- [ ] Run `python3 -m pytest tests/test_cli.py -q`; expected pass.

### Task 8: Final Verification

**Files:**
- Modify: docs and README if command behavior differs from plan.

- [ ] Run `python3 -m pytest -q`.
- [ ] Run a tiny one-step training command.
- [ ] Run a tiny sampling command.
- [ ] Run evaluation on generated toy output.
- [ ] Report exact commands and outcomes.

