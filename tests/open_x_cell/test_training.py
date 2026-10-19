import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from open_x_cell.config import LossConfig, TrainingConfig, tiny_config
from open_x_cell.data import OpenXCellDataset, collate_samples
from open_x_cell.priors import PriorStore
from open_x_cell.synthetic import make_synthetic_adata
import open_x_cell.training as training_module
from open_x_cell.training import (
    _aggregate_target_centers_by_label,
    _initialize_from_xcell_checkpoint,
    _run_loop,
    compute_batch_loss,
    make_warmup_cosine_scheduler,
    save_checkpoint,
    train_model,
)
from open_x_cell.model import OpenXCellModel
from open_x_cell.losses import LossOutput, OpenXCellLoss


def test_compute_batch_loss_uses_contrastive_and_prior_reconstruction(tmp_path):
    cfg = tiny_config()
    adata = make_synthetic_adata(n_cells=64, n_genes=32, n_perturbations=2, seed=5)
    ds = OpenXCellDataset(adata, set_size=4, context_length=cfg.context_length, seed=0)
    batch = next(iter(DataLoader(ds, batch_size=2, collate_fn=collate_samples)))
    esm_path = tmp_path / "esm.csv"
    pd.DataFrame(
        {
            "gene": ["GENE_1", "GENE_2"],
            **{f"f{idx}": [0.1, 0.2] for idx in range(cfg.prior_dims["esm"])},
        }
    ).to_csv(esm_path, index=False)
    priors = PriorStore.from_files({"esm": esm_path}, cfg)
    model = OpenXCellModel(cfg)

    output = compute_batch_loss(
        model=model,
        batch=batch,
        model_config=cfg,
        loss_fn=OpenXCellLoss(),
        prior_store=priors,
        step=0,
        seed=0,
        device=torch.device("cpu"),
    )

    assert "contrastive" in output.terms
    assert "prior_reconstruction" in output.terms
    assert torch.isfinite(output.total)


def test_warmup_cosine_scheduler_reaches_zero_after_training_horizon():
    param = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([param], lr=1.0)
    scheduler = make_warmup_cosine_scheduler(optimizer, warmup_steps=2, total_steps=6)

    used_lrs = []
    for _ in range(6):
        used_lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert used_lrs[0] == 0.5
    assert used_lrs[1] == 1.0
    assert used_lrs[-1] > 0.0
    assert optimizer.param_groups[0]["lr"] == 0.0


def test_target_centers_are_aggregated_once_per_perturbation():
    centers = torch.tensor([[1.0, 0.0], [3.0, 2.0], [0.0, 4.0]])
    labels = torch.tensor([5, 5, 2])

    aggregated, unique_labels = _aggregate_target_centers_by_label(centers, labels)

    assert torch.equal(unique_labels, torch.tensor([2, 5]))
    assert torch.equal(aggregated, torch.tensor([[0.0, 4.0], [2.0, 1.0]]))


def test_run_loop_builds_one_contrastive_bank_from_every_accumulation_batch(monkeypatch):
    class Loader:
        dataset = object()

        def __len__(self):
            return 3

        def __iter__(self):
            return iter(("batch-0", "batch-1", "batch-2"))

    model = torch.nn.Linear(1, 1, bias=False)
    bank = torch.tensor([[1.0]])
    bank_labels = torch.tensor([7])
    captured = {"bank_calls": [], "loss_batches": []}

    def build_bank(model, batches, model_config, prior_store, device, precision):
        captured["bank_calls"].append(tuple(batches))
        return bank, bank_labels

    def batch_loss(**kwargs):
        assert kwargs["contrastive_target_bank"] is bank
        assert kwargs["contrastive_target_bank_labels"] is bank_labels
        captured["loss_batches"].append(kwargs["batch"])
        total = kwargs["model"].weight.square().sum()
        return LossOutput(total=total, terms={"contrastive": total.detach()})

    monkeypatch.setattr(
        training_module,
        "_optimizer_step_contrastive_target_bank",
        build_bank,
    )
    monkeypatch.setattr(training_module, "compute_batch_loss", batch_loss)
    cfg = TrainingConfig(
        batch_size=1,
        set_size=1,
        max_steps=1,
        learning_rate=1e-3,
        warmup_fraction=0.0,
        gradient_accumulation_steps=3,
        device="cpu",
    )
    loss_fn = OpenXCellLoss(
        LossConfig(
            mmd=0.0,
            fold_change_ccc=0.0,
            per_gene_ccc=0.0,
            delta_norm=0.0,
            delta_mse=0.0,
            contrastive=1.0,
            prior_reconstruction=0.0,
        )
    )

    history = _run_loop(
        model,
        Loader(),
        tiny_config(),
        None,
        loss_fn,
        cfg,
        max_steps=1,
        learning_rate=cfg.learning_rate,
        warmup_fraction=cfg.warmup_fraction,
    )

    assert captured["bank_calls"] == [("batch-0", "batch-1", "batch-2")]
    assert captured["loss_batches"] == ["batch-0", "batch-1", "batch-2"]
    assert len(history) == 1


def test_pretrained_checkpoint_remaps_gene_rows_and_resets_training_state(tmp_path):
    cfg = tiny_config()
    source_model = OpenXCellModel(cfg)
    with torch.no_grad():
        source_model.gene_embedding.weight[0].fill_(11.0)
        source_model.gene_embedding.weight[1].fill_(22.0)
        source_model.gene_embedding.weight[2].fill_(33.0)
        source_model.gene_bias.weight[0].fill_(1.0)
        source_model.gene_bias.weight[1].fill_(2.0)
        source_model.gene_bias.weight[2].fill_(3.0)
        source_model.cls.fill_(3.0)
    checkpoint = tmp_path / "pretrained.pt"
    save_checkpoint(checkpoint, source_model, cfg, ["GENE_A", "GENE_B", "<mask>"])

    with pytest.raises(ValueError, match="Appendix A.9 threshold"):
        _initialize_from_xcell_checkpoint(
            OpenXCellModel(cfg),
            checkpoint,
            ["GENE_B", "GENE_A", "GENE_C"],
            cfg,
        )

    destination_model = OpenXCellModel(cfg)
    stats = _initialize_from_xcell_checkpoint(
        destination_model,
        checkpoint,
        ["GENE_B", "GENE_A", "GENE_C"],
        cfg,
        min_overlap=0.5,
    )

    assert torch.equal(
        destination_model.gene_embedding.weight[0], source_model.gene_embedding.weight[1]
    )
    assert torch.equal(
        destination_model.gene_embedding.weight[1], source_model.gene_embedding.weight[0]
    )
    assert torch.equal(
        destination_model.gene_embedding.weight[2], source_model.gene_embedding.weight[2]
    )
    assert torch.equal(destination_model.gene_bias.weight[0], source_model.gene_bias.weight[1])
    assert torch.equal(destination_model.cls, source_model.cls)
    assert stats["overlap_genes"] == 2
    assert stats["unmapped_genes"] == 1
    assert stats["unmapped_token"] == "<mask>"
    assert stats["vocabulary_identical"] is False
    assert stats["optimizer_state_loaded"] is False


def test_pretrained_checkpoint_is_mutually_exclusive_with_other_initialization(tmp_path):
    adata = make_synthetic_adata(n_cells=64, n_genes=24, n_perturbations=2, seed=1)
    adata_path = tmp_path / "data.h5ad"
    adata.write_h5ad(adata_path)
    with pytest.raises(ValueError, match="mutually exclusive initialization modes"):
        train_model(
            adata_path,
            tmp_path / "run",
            model_size="tiny",
            max_steps=1,
            pretrained_checkpoint=tmp_path / "pretrained.pt",
            scgpt_init=tmp_path / "scgpt.npz",
        )


def test_exact_resume_preserves_manifest_and_rejects_missing_prior(tmp_path, monkeypatch):
    adata = make_synthetic_adata(n_cells=96, n_genes=32, n_perturbations=4, seed=7)
    adata_path = tmp_path / "data.h5ad"
    adata.write_h5ad(adata_path)
    prior_path = tmp_path / "esm.npz"
    np.savez(
        prior_path,
        genes=np.asarray(["GENE_1"]),
        embeddings=np.ones((1, tiny_config().prior_dims["esm"]), dtype=np.float32),
    )
    output = tmp_path / "run"
    captured_resume = output / "resume_step_1.pt"
    original_save_checkpoint = training_module.save_checkpoint

    def save_and_capture(path, model, config, var_names, training_state=None):
        original_save_checkpoint(path, model, config, var_names, training_state=training_state)
        if (
            Path(path).name == "latest.pt"
            and training_state is not None
            and training_state.get("completed_steps") == 1
        ):
            shutil.copy2(path, captured_resume)

    monkeypatch.setattr(training_module, "save_checkpoint", save_and_capture)
    config = TrainingConfig(
        batch_size=2,
        set_size=8,
        max_steps=3,
        learning_rate=1e-3,
        seed=0,
        device="cpu",
        checkpoint_every=1,
    )
    uninterrupted = train_model(
        adata_path,
        output,
        model_size="tiny",
        training_config=config,
        normalize="none",
        prior_files={"esm": prior_path},
    )
    assert captured_resume.is_file()
    uninterrupted_payload = torch.load(
        uninterrupted.checkpoint, map_location="cpu", weights_only=False
    )
    manifest_before = json.loads((output / "run_manifest.json").read_text())

    monkeypatch.setattr(training_module, "save_checkpoint", original_save_checkpoint)
    with pytest.raises(ValueError, match="resume prior files differ"):
        train_model(
            adata_path,
            output,
            model_size="tiny",
            training_config=config,
            normalize="none",
            prior_files={},
            resume_from=captured_resume,
        )
    assert json.loads((output / "run_manifest.json").read_text()) == manifest_before

    malformed_resume = output / "malformed_resume.pt"
    malformed_payload = torch.load(captured_resume, map_location="cpu", weights_only=False)
    malformed_payload["training_state"]["history"][0]["step"] = 1.5
    torch.save(malformed_payload, malformed_resume)
    with pytest.raises(ValueError, match="history is malformed at step 1"):
        train_model(
            adata_path,
            output,
            model_size="tiny",
            training_config=config,
            normalize="none",
            prior_files={"esm": prior_path},
            resume_from=malformed_resume,
        )
    assert json.loads((output / "run_manifest.json").read_text()) == manifest_before

    malformed_lr = output / "malformed_lr.pt"
    malformed_payload = torch.load(captured_resume, map_location="cpu", weights_only=False)
    malformed_payload["training_state"]["optimizer_state"]["param_groups"][0]["lr"] = 1.0
    torch.save(malformed_payload, malformed_lr)
    with pytest.raises(ValueError, match="optimizer learning rate differs"):
        train_model(
            adata_path,
            output,
            model_size="tiny",
            training_config=config,
            normalize="none",
            prior_files={"esm": prior_path},
            resume_from=malformed_lr,
        )
    assert json.loads((output / "run_manifest.json").read_text()) == manifest_before

    resumed = train_model(
        adata_path,
        output,
        model_size="tiny",
        training_config=config,
        normalize="none",
        prior_files={"esm": prior_path},
        resume_from=captured_resume,
    )
    resumed_payload = torch.load(resumed.checkpoint, map_location="cpu", weights_only=False)
    assert uninterrupted_payload["model_state"].keys() == resumed_payload["model_state"].keys()
    for name, tensor in uninterrupted_payload["model_state"].items():
        assert torch.equal(tensor, resumed_payload["model_state"][name]), name

    manifest_after = json.loads((output / "run_manifest.json").read_text())
    for key, value in manifest_before.items():
        if key != "resume_events":
            assert manifest_after[key] == value
    assert manifest_after["resume_events"][:-1] == manifest_before["resume_events"]
    event = manifest_after["resume_events"][-1]
    assert event["completed_steps"] == 1
    assert event["source_path"] == str(captured_resume.resolve())
    retained = Path(event["checkpoint"]["path"])
    assert retained.parent == (output / "resume_inputs").resolve()
    assert retained.is_file()
    assert len(event["checkpoint"]["sha256"]) == 64
    assert len(resumed.history) == 3
