# Open-X-Cell Clean-Room Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean-room, runnable Open-X-Cell reproduction with PyTorch model code, diffusion masking, prior embeddings, AnnData data handling, training/prediction/evaluation CLIs, and synthetic tests.

**Architecture:** Implement a dedicated `open_x_cell` package for this repository. Start from small tested primitives, then compose them into model, losses, data loaders, training loop, inference loop, and CLI workflows. Use a `tiny` preset for local verification while exposing paper-aligned `mini` and `ultra` configs.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, SciPy optional, AnnData optional for data/CLI tests, pytest, PyYAML optional for config files.

---

## File Structure

- Create `src/open_x_cell/__init__.py`: public package exports.
- Create `src/open_x_cell/config.py`: dataclass configs and `tiny`, `mini`, `ultra` presets.
- Create `src/open_x_cell/diffusion.py`: reveal-mask creation, input mixing, and cumulative schedule helpers.
- Create `src/open_x_cell/priors.py`: prior source definitions and tensor containers.
- Create `src/open_x_cell/model.py`: PyTorch model, attention blocks, encoders, decoder.
- Create `src/open_x_cell/losses.py`: differentiable objective terms.
- Create `src/open_x_cell/synthetic.py`: synthetic perturbation AnnData generation and tensor batch generation.
- Create `src/open_x_cell/data.py`: AnnData loading, preprocessing, dataset/collator.
- Create `src/open_x_cell/training.py`: train loop, checkpoint save/load.
- Create `src/open_x_cell/inference.py`: four-step prediction wrapper.
- Create `src/open_x_cell/evaluation.py`: metrics for predicted vs observed perturbation effects.
- Create `src/open_x_cell/cli.py`: command line interface.
- Modify `pyproject.toml`: add package discovery under `src`, dependencies, and CLI entrypoint while preserving existing project behavior.
- Modify `.gitignore`: ignore generated runs, data, caches, and checkpoint artifacts.
- Create `configs/open_x_cell_tiny.yaml`: tiny smoke-test training config.
- Create tests under `tests/open_x_cell/`.
- Modify `README.md`: add Open-X-Cell clean-room section and commands.

## Task 1: Package Metadata And Configs

**Files:**
- Create: `src/open_x_cell/__init__.py`
- Create: `src/open_x_cell/config.py`
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `configs/open_x_cell_tiny.yaml`
- Test: `tests/open_x_cell/test_config.py`

- [ ] **Step 1: Write failing config tests**

```python
from open_x_cell.config import ModelConfig, TrainingConfig, mini_config, tiny_config, ultra_config


def test_tiny_config_is_small_and_internally_consistent():
    cfg = tiny_config()
    assert cfg.num_genes == 32
    assert cfg.context_length == 16
    assert cfg.hidden_dim == 32
    assert cfg.hidden_dim % cfg.num_heads == 0
    assert cfg.cross_attention_layers == (1,)
    assert cfg.prior_dims["esm"] == 5120


def test_mini_config_matches_paper_shape():
    cfg = mini_config()
    assert cfg.num_layers == 12
    assert cfg.hidden_dim == 512
    assert cfg.num_heads == 8
    assert cfg.cross_attention_layers == (2, 5, 8, 11)
    assert cfg.use_qk_norm is False


def test_ultra_config_matches_paper_shape():
    cfg = ultra_config()
    assert cfg.num_layers == 34
    assert cfg.hidden_dim == 2560
    assert cfg.num_heads == 40
    assert cfg.cross_attention_layers[-1] == 33
    assert cfg.use_qk_norm is True
    assert cfg.tied_output is True


def test_model_config_rejects_invalid_attention_width():
    try:
        ModelConfig(hidden_dim=30, num_heads=8)
    except ValueError as exc:
        assert "hidden_dim must be divisible by num_heads" in str(exc)
    else:
        raise AssertionError("invalid attention width was accepted")


def test_training_config_defaults_are_local_smoke_safe():
    cfg = TrainingConfig()
    assert cfg.batch_size == 2
    assert cfg.set_size == 8
    assert cfg.learning_rate == 1e-3
    assert cfg.max_steps == 10
```

- [ ] **Step 2: Run config tests to verify RED**

Run: `PYTHONPATH=src pytest tests/open_x_cell/test_config.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'open_x_cell'`.

- [ ] **Step 3: Implement package config**

Create `src/open_x_cell/config.py` with `ModelConfig`, `TrainingConfig`, `LossConfig`, and preset functions. Include validation for attention divisibility and context length.

Create `src/open_x_cell/__init__.py` exporting `ModelConfig`, `TrainingConfig`, `tiny_config`, `mini_config`, and `ultra_config`.

Update `pyproject.toml` to include `src` package discovery and add `open-x-cell = "open_x_cell.cli:main"` under `[project.scripts]`.

Create `configs/open_x_cell_tiny.yaml` with:

```yaml
model: tiny
batch_size: 2
set_size: 8
max_steps: 4
learning_rate: 0.001
context_length: 16
```

- [ ] **Step 4: Run config tests to verify GREEN**

Run: `PYTHONPATH=src pytest tests/open_x_cell/test_config.py -v`

Expected: PASS.

- [ ] **Step 5: Commit task**

```bash
git add pyproject.toml .gitignore configs/open_x_cell_tiny.yaml src/open_x_cell/__init__.py src/open_x_cell/config.py tests/open_x_cell/test_config.py
git commit -m "feat: add open x-cell reproduction config"
```

## Task 2: Diffusion Masks And Prior Containers

**Files:**
- Create: `src/open_x_cell/diffusion.py`
- Create: `src/open_x_cell/priors.py`
- Test: `tests/open_x_cell/test_diffusion.py`
- Test: `tests/open_x_cell/test_priors.py`

- [ ] **Step 1: Write failing diffusion tests**

```python
import torch

from open_x_cell.diffusion import cumulative_reveal_steps, make_reveal_mask, mix_control_target


def test_make_reveal_mask_reveals_exact_fraction_per_sample():
    mask = make_reveal_mask(batch_size=2, set_size=3, num_genes=8, fraction=0.25, seed=7)
    assert mask.shape == (2, 3, 8)
    assert mask.dtype == torch.bool
    assert mask[0, 0].sum().item() == 2
    assert torch.equal(mask[0, 0], mask[0, 1])


def test_mix_control_target_uses_target_where_mask_is_true():
    ctrl = torch.zeros(1, 2, 4)
    target = torch.ones(1, 2, 4)
    mask = torch.tensor([[[True, False, True, False], [False, True, False, True]]])
    mixed = mix_control_target(ctrl, target, mask)
    assert mixed.tolist() == [[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]]


def test_cumulative_reveal_steps_partition_genes():
    steps = cumulative_reveal_steps(num_genes=8, fractions=(0.25, 0.5, 0.75, 1.0), seed=3)
    assert len(steps) == 4
    assert [step.sum().item() for step in steps] == [2, 2, 2, 2]
    total = torch.stack(steps).sum(dim=0)
    assert torch.all(total == 1)
```

- [ ] **Step 2: Write failing prior tests**

```python
import torch

from open_x_cell.config import tiny_config
from open_x_cell.priors import PriorBatch, make_zero_priors


def test_make_zero_priors_has_paper_sources_and_missing_mask():
    cfg = tiny_config()
    priors = make_zero_priors(batch_size=2, config=cfg)
    assert priors.values["esm"].shape == (2, 5120)
    assert priors.values["cell_painting"].shape == (2, 259)
    assert priors.missing.shape == (2, 5)
    assert torch.all(priors.missing)


def test_prior_batch_to_moves_all_tensors():
    priors = PriorBatch(values={"esm": torch.zeros(1, 5120)}, missing=torch.ones(1, 1, dtype=torch.bool))
    moved = priors.to(torch.device("cpu"))
    assert moved.values["esm"].device.type == "cpu"
    assert moved.missing.device.type == "cpu"
```

- [ ] **Step 3: Run tests to verify RED**

Run: `PYTHONPATH=src pytest tests/open_x_cell/test_diffusion.py tests/open_x_cell/test_priors.py -v`

Expected: FAIL because `open_x_cell.diffusion` and `open_x_cell.priors` are missing.

- [ ] **Step 4: Implement diffusion and priors**

Implement deterministic mask generation, input mixing, cumulative reveal partitions, `PriorBatch`, `PRIOR_SOURCES`, and `make_zero_priors`.

- [ ] **Step 5: Run tests to verify GREEN**

Run: `PYTHONPATH=src pytest tests/open_x_cell/test_diffusion.py tests/open_x_cell/test_priors.py -v`

Expected: PASS.

- [ ] **Step 6: Commit task**

```bash
git add src/open_x_cell/diffusion.py src/open_x_cell/priors.py tests/open_x_cell/test_diffusion.py tests/open_x_cell/test_priors.py
git commit -m "feat: add diffusion masks and prior containers"
```

## Task 3: Tiny X-Cell Model Forward Pass

**Files:**
- Create: `src/open_x_cell/model.py`
- Test: `tests/open_x_cell/test_model.py`

- [ ] **Step 1: Write failing model tests**

```python
import torch

from open_x_cell.config import tiny_config
from open_x_cell.model import OpenXCellModel
from open_x_cell.priors import make_zero_priors


def test_tiny_model_forward_returns_expression_and_embeddings():
    cfg = tiny_config()
    model = OpenXCellModel(cfg)
    x = torch.rand(2, 4, cfg.context_length)
    mask = torch.zeros(2, 4, cfg.context_length, dtype=torch.bool)
    gene_ids = torch.arange(cfg.context_length).repeat(2, 1)
    perturbation_ids = torch.tensor([1, 2])
    priors = make_zero_priors(batch_size=2, config=cfg)

    out = model(x, mask, gene_ids, perturbation_ids, priors)

    assert out.expression.shape == (2, 4, cfg.context_length)
    assert out.cell_embedding.shape[:2] == (2, 4)
    assert out.prior_reconstructions["esm"].shape == (2, 5120)
    assert torch.isfinite(out.expression).all()


def test_tiny_model_supports_backward_pass():
    cfg = tiny_config()
    model = OpenXCellModel(cfg)
    x = torch.rand(1, 2, cfg.context_length)
    mask = torch.zeros(1, 2, cfg.context_length, dtype=torch.bool)
    gene_ids = torch.arange(cfg.context_length).repeat(1, 1)
    perturbation_ids = torch.tensor([1])
    out = model(x, mask, gene_ids, perturbation_ids, make_zero_priors(1, cfg))
    loss = out.expression.mean()
    loss.backward()
    assert any(param.grad is not None for param in model.parameters())
```

- [ ] **Step 2: Run model tests to verify RED**

Run: `PYTHONPATH=src pytest tests/open_x_cell/test_model.py -v`

Expected: FAIL because `open_x_cell.model` is missing.

- [ ] **Step 3: Implement model**

Implement:

- `ModelOutput`
- `RMSNorm`
- `FeedForward`
- `TransformerBlock`
- `CrossAttentionBlock`
- `PriorProjector`
- `OpenXCellModel`

The implementation must flatten `[B, S]`, prepend CLS, run transformer blocks, concatenate CLS to gene tokens, decode expression, and reshape to `[B, S, G]`.

- [ ] **Step 4: Run model tests to verify GREEN**

Run: `PYTHONPATH=src pytest tests/open_x_cell/test_model.py -v`

Expected: PASS.

- [ ] **Step 5: Commit task**

```bash
git add src/open_x_cell/model.py tests/open_x_cell/test_model.py
git commit -m "feat: add open x-cell transformer model"
```

## Task 4: X-Cell Losses

**Files:**
- Create: `src/open_x_cell/losses.py`
- Test: `tests/open_x_cell/test_losses.py`

- [ ] **Step 1: Write failing loss tests**

```python
import torch

from open_x_cell.losses import OpenXCellLoss, concordance_ccc, energy_distance


def test_energy_distance_is_zero_for_identical_sets():
    x = torch.randn(2, 4, 8)
    assert energy_distance(x, x).abs().item() < 1e-6


def test_ccc_is_one_for_identical_vectors():
    x = torch.randn(3, 10)
    value = concordance_ccc(x, x, dim=1)
    assert torch.allclose(value, torch.ones_like(value), atol=1e-5)


def test_xcell_loss_returns_finite_weighted_terms():
    pred = torch.rand(2, 4, 8, requires_grad=True)
    target = torch.rand(2, 4, 8)
    ctrl = torch.rand(2, 4, 8)
    cell = torch.rand(2, 4, 16)
    loss_fn = OpenXCellLoss()
    result = loss_fn(prediction=pred, target=target, control=ctrl, cell_embedding=cell)
    assert torch.isfinite(result.total)
    assert {"mmd", "fold_change_ccc", "delta_mse"}.issubset(result.terms)
    result.total.backward()
    assert pred.grad is not None
```

- [ ] **Step 2: Run loss tests to verify RED**

Run: `PYTHONPATH=src pytest tests/open_x_cell/test_losses.py -v`

Expected: FAIL because `open_x_cell.losses` is missing.

- [ ] **Step 3: Implement losses**

Implement torch-only energy distance fallback, CCC, delta norm, delta MSE, optional per-gene CCC, optional contrastive term, optional prior reconstruction term, `LossOutput`, and `OpenXCellLoss`.

- [ ] **Step 4: Run loss tests to verify GREEN**

Run: `PYTHONPATH=src pytest tests/open_x_cell/test_losses.py -v`

Expected: PASS.

- [ ] **Step 5: Commit task**

```bash
git add src/open_x_cell/losses.py tests/open_x_cell/test_losses.py
git commit -m "feat: add open x-cell objective losses"
```

## Task 5: Synthetic Data And AnnData Dataset

**Files:**
- Create: `src/open_x_cell/synthetic.py`
- Create: `src/open_x_cell/data.py`
- Test: `tests/open_x_cell/test_data.py`

- [ ] **Step 1: Write failing data tests**

```python
from open_x_cell.data import OpenXCellDataset
from open_x_cell.synthetic import make_synthetic_adata


def test_make_synthetic_adata_has_required_obs_and_var():
    adata = make_synthetic_adata(n_cells=64, n_genes=24, n_perturbations=3, seed=1)
    assert adata.shape == (64, 24)
    assert {"perturbation", "context", "batch", "is_control"}.issubset(adata.obs.columns)
    assert adata.var_names[0] == "GENE_0"


def test_dataset_returns_matched_control_target_set():
    adata = make_synthetic_adata(n_cells=96, n_genes=24, n_perturbations=4, seed=2)
    ds = OpenXCellDataset(adata, set_size=4, context_length=12, seed=3)
    sample = ds[0]
    assert sample.x_control.shape == (4, 12)
    assert sample.x_target.shape == (4, 12)
    assert sample.gene_ids.shape == (12,)
    assert sample.perturbation_id >= 0
```

- [ ] **Step 2: Run data tests to verify RED**

Run: `PYTHONPATH=src pytest tests/open_x_cell/test_data.py -v`

Expected: FAIL because synthetic/data modules are missing.

- [ ] **Step 3: Implement synthetic and dataset modules**

Implement synthetic AnnData generation with control cells and perturbation-specific additive effects. Implement `OpenXCellSample`, `OpenXCellDataset`, and `collate_samples`.

- [ ] **Step 4: Run data tests to verify GREEN**

Run: `PYTHONPATH=src pytest tests/open_x_cell/test_data.py -v`

Expected: PASS.

- [ ] **Step 5: Commit task**

```bash
git add src/open_x_cell/synthetic.py src/open_x_cell/data.py tests/open_x_cell/test_data.py
git commit -m "feat: add open x-cell synthetic data pipeline"
```

## Task 6: Training, Inference, Evaluation, And CLI

**Files:**
- Create: `src/open_x_cell/training.py`
- Create: `src/open_x_cell/inference.py`
- Create: `src/open_x_cell/evaluation.py`
- Create: `src/open_x_cell/cli.py`
- Test: `tests/open_x_cell/test_cli.py`
- Test: `tests/open_x_cell/test_evaluation.py`

- [ ] **Step 1: Write failing evaluation and CLI tests**

```python
import json
import subprocess
import sys

import torch

from open_x_cell.evaluation import perturbation_metrics


def test_perturbation_metrics_returns_finite_values():
    control = torch.zeros(4, 8)
    truth = torch.ones(4, 8)
    pred = torch.ones(4, 8) * 0.9
    metrics = perturbation_metrics(prediction=pred, truth=truth, control=control, top_n=3)
    assert metrics["mae"] >= 0
    assert -1 <= metrics["pearson_delta"] <= 1
    assert 0 <= metrics["overlap_at_n"] <= 1


def test_cli_make_synthetic_train_predict_evaluate_smoke(tmp_path):
    adata = tmp_path / "synthetic.h5ad"
    run_dir = tmp_path / "run"
    pred = tmp_path / "pred.h5ad"
    metrics = tmp_path / "metrics.json"

    subprocess.run(
        [sys.executable, "-m", "open_x_cell.cli", "make-synthetic", "--output", str(adata), "--n-cells", "64", "--n-genes", "24"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "open_x_cell.cli", "train", "--adata", str(adata), "--output", str(run_dir), "--max-steps", "2"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "open_x_cell.cli", "predict", "--checkpoint", str(run_dir / "checkpoint.pt"), "--adata", str(adata), "--perturbation", "GENE_1", "--output", str(pred)],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "open_x_cell.cli", "evaluate", "--truth", str(adata), "--pred", str(pred), "--output", str(metrics)],
        check=True,
    )
    payload = json.loads(metrics.read_text())
    assert "mae" in payload
```

- [ ] **Step 2: Run tests to verify RED**

Run: `PYTHONPATH=src pytest tests/open_x_cell/test_evaluation.py tests/open_x_cell/test_cli.py -v`

Expected: FAIL because modules are missing.

- [ ] **Step 3: Implement training, inference, evaluation, and CLI**

Implement:

- `train_tiny`
- `save_checkpoint`
- `load_checkpoint`
- `predict_adata`
- `perturbation_metrics`
- CLI subcommands `make-synthetic`, `train`, `predict`, and `evaluate`.

The CLI smoke path must train on CPU using the tiny config and finish quickly.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `PYTHONPATH=src pytest tests/open_x_cell/test_evaluation.py tests/open_x_cell/test_cli.py -v`

Expected: PASS.

- [ ] **Step 5: Commit task**

```bash
git add src/open_x_cell/training.py src/open_x_cell/inference.py src/open_x_cell/evaluation.py src/open_x_cell/cli.py tests/open_x_cell/test_evaluation.py tests/open_x_cell/test_cli.py
git commit -m "feat: add open x-cell training and cli workflow"
```

## Task 7: Documentation And Full Verification

**Files:**
- Modify: `README.md`
- Test: full test suite

- [ ] **Step 1: Update README**

Add a section named `Open-X-Cell Clean-Room Reproduction` with:

```markdown
## Open-X-Cell Clean-Room Reproduction

This repository includes a clean-room PyTorch reproduction of the X-Cell architecture from the paper PDF. It does not copy the official implementation or claim numerical equivalence to unreleased Xaira weights.

Run a local smoke workflow:

```bash
PYTHONPATH=src python -m open_x_cell.cli make-synthetic --output data/synthetic.h5ad --n-cells 128 --n-genes 32
PYTHONPATH=src python -m open_x_cell.cli train --adata data/synthetic.h5ad --output runs/open_x_cell_tiny --max-steps 4
PYTHONPATH=src python -m open_x_cell.cli predict --checkpoint runs/open_x_cell_tiny/checkpoint.pt --adata data/synthetic.h5ad --perturbation GENE_1 --output runs/open_x_cell_tiny/pred.h5ad
PYTHONPATH=src python -m open_x_cell.cli evaluate --truth data/synthetic.h5ad --pred runs/open_x_cell_tiny/pred.h5ad --output runs/open_x_cell_tiny/metrics.json
```
```

- [ ] **Step 2: Run full verification**

Run: `PYTHONPATH=src pytest tests/open_x_cell -v`

Expected: PASS.

- [ ] **Step 3: Run CLI smoke manually**

Run:

```bash
rm -rf runs/open_x_cell_tiny data/xcell_synthetic.h5ad
PYTHONPATH=src python -m open_x_cell.cli make-synthetic --output data/xcell_synthetic.h5ad --n-cells 128 --n-genes 32
PYTHONPATH=src python -m open_x_cell.cli train --adata data/xcell_synthetic.h5ad --output runs/open_x_cell_tiny --max-steps 4
PYTHONPATH=src python -m open_x_cell.cli predict --checkpoint runs/open_x_cell_tiny/checkpoint.pt --adata data/xcell_synthetic.h5ad --perturbation GENE_1 --output runs/open_x_cell_tiny/pred.h5ad
PYTHONPATH=src python -m open_x_cell.cli evaluate --truth data/xcell_synthetic.h5ad --pred runs/open_x_cell_tiny/pred.h5ad --output runs/open_x_cell_tiny/metrics.json
```

Expected: all commands exit 0 and `runs/open_x_cell_tiny/metrics.json` contains finite metrics.

- [ ] **Step 4: Commit task**

```bash
git add README.md docs/superpowers/plans/2026-06-21-open-x-cell-implementation.md
git commit -m "docs: document open x-cell reproduction workflow"
```

## Self-Review Checklist

- Spec coverage: package, model, diffusion, priors, losses, AnnData, CLI, tests, and docs are covered.
- Placeholder scan: no task relies on "TBD", "TODO", "similar to", or unspecified implementation.
- Type consistency: config names, CLI commands, and tensor names are consistent across tasks.
- Scope: this plan targets a runnable clean-room reproduction and avoids H200-scale training claims.
