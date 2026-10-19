"""Validate the complete Replogle Mini run and emit one machine-readable audit.

This script is intended to run on a pure CPU Slurm partition after training,
benchmark inference, input assembly, and the official cell-eval command finish.
It deliberately checks the artifacts themselves instead of trusting job exit codes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import torch

from open_x_cell.data import boolean_obs_mask


PAPER_METRICS = (
    "pearson_delta",
    "overlap_at_N",
    "de_direction_match",
    "de_spearman_lfc_sig",
    "mae",
    "discrimination_score_l2",
)

METRIC_BOUNDS = {
    "pearson_delta": (-1.0, 1.0),
    "overlap_at_N": (0.0, 1.0),
    "de_direction_match": (0.0, 1.0),
    "de_spearman_lfc_sig": (-1.0, 1.0),
    "mae": (0.0, math.inf),
    "discrimination_score_l2": (0.0, 1.0),
}

EXPECTED_MODEL_CONFIG = {
    "num_genes": 19_408,
    "context_length": 19_408,
    "num_layers": 12,
    "hidden_dim": 512,
    "num_heads": 8,
    "ffn_dim": 512,
    "cross_attention_layers": [2, 5, 8, 11],
    "cross_attention_ffn_dim": 2_048,
    "cross_attention_activation": "swiglu",
    "cross_attention_norm": "rmsnorm",
    "cross_attention_pre_norm": True,
    "dropout": 0.0,
    "prior_dims": {
        "esm": 5_120,
        "string": 512,
        "genept": 3_072,
        "depmap": 1_150,
        "cell_painting": 259,
    },
    "activation": "relu",
    "norm": "layernorm",
    "pre_norm": False,
    "use_qk_norm": False,
    "attention_bias": True,
    "tied_output": False,
    "value_clip": 512.0,
    "diffusion_fractions": [0.25, 0.5, 0.75, 1.0],
}

EXPECTED_TRAINING_CONFIG = {
    "batch_size": 1,
    "set_size": 64,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "beta1": 0.9,
    "beta2": 0.999,
    "warmup_fraction": 0.1,
    "grad_clip_norm": 1.0,
    "seed": 0,
    "device": "cuda:0",
    "precision": "bfloat16",
    "gradient_accumulation_steps": 32,
    "num_workers": 0,
    "checkpoint_every": 50,
    "log_every": 10,
}

EXPECTED_RUNTIME_VERSIONS = {
    "python": "3.10.19",
    "torch": "2.7.1+cu128",
    "torch_cuda_build": "12.8",
    "anndata": "0.11.4",
    "numpy": "2.2.6",
    "pandas": "2.3.3",
    "scipy": "1.15.3",
    "geomloss": "0.2.6",
    "polars": "1.38.1",
    "cell-eval": "0.6.8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--perturbations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-steps", type=int, default=1081)
    parser.add_argument("--expected-epochs", type=int, default=6)
    parser.add_argument("--expected-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--dataset-audit",
        default=None,
        help="Training-dataset audit JSON; defaults to the original six-epoch audit.",
    )
    parser.add_argument(
        "--normalization-audit",
        default=None,
        help="Replogle normalization provenance audit; defaults beside the truth H5AD.",
    )
    parser.add_argument(
        "--feature-space-audit",
        default=None,
        help="Replogle HVG/protein-coding ambiguity audit; defaults beside the truth H5AD.",
    )
    parser.add_argument(
        "--training-validation-audit",
        default=None,
        help=(
            "Appendix A.6 validation/checkpoint-selection boundary audit; defaults "
            "beside the truth H5AD."
        ),
    )
    parser.add_argument(
        "--training-input-profile",
        choices=(
            "released_hvg2k",
            "public_target_complete6808",
            "public_raw_union11332_dynamic4000",
        ),
        default="released_hvg2k",
        help=(
            "Keep benchmark genes separate from the training context. The 6,808-gene "
            "profile is the maximum target-complete public proxy, not Xaira's unreleased "
            "approximately 19k Table A3 input."
        ),
    )
    parser.add_argument("--expected-genes", type=int, default=2000)
    parser.add_argument("--expected-controls", type=int, default=32)
    parser.add_argument("--expected-shards", type=int, default=2)
    parser.add_argument("--expected-world-size", type=int, default=2)
    parser.add_argument(
        "--expected-initialization",
        choices=("scgpt", "pretrained"),
        default="scgpt",
    )
    parser.add_argument(
        "--expected-scgpt-transfer-scope",
        choices=("appendix_a7", "table_a3_embeddings_only"),
        default="appendix_a7",
        help=(
            "Only applies to --expected-initialization=scgpt. Appendix A.7 loads gene "
            "embeddings/norm plus transformer weights; Table A3 literally says embeddings."
        ),
    )
    parser.add_argument(
        "--expected-pretrained-checkpoint",
        default=None,
        help="Required exact source checkpoint when --expected-initialization=pretrained.",
    )
    parser.add_argument(
        "--expected-loss-profile",
        choices=(
            "eq5_all",
            "appendix_a2_7_l1_zero",
            "paper_mini_explicit_sentence",
        ),
        default="eq5_all",
    )
    parser.add_argument("--expected-gradient-checkpointing", action="store_true")
    parser.add_argument("--expected-cell-eval-version", default="0.6.8")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _json_normalized(value: Any) -> Any:
    """Normalize tuple/list representation before comparing Torch and JSON payloads."""
    return json.loads(json.dumps(value))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _current_source_fingerprints() -> dict[str, str]:
    package_dir = Path(__file__).resolve().parents[1] / "src" / "open_x_cell"
    return {path.name: _sha256(path) for path in sorted(package_dir.glob("*.py"))}


def _check_mapping(actual: dict, expected: dict, label: str) -> None:
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label} mismatch: {mismatches}")


def _verify_artifact(record: dict, label: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"{label} artifact record must be a mapping")
    path = Path(record.get("path", "")).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if int(record.get("bytes", -1)) != path.stat().st_size:
        raise ValueError(f"{label} byte size changed after training")
    if record.get("sha256") != _sha256(path):
        raise ValueError(f"{label} SHA-256 changed after training")
    return path


def _required_training_input_artifacts(initialization: str) -> set[str]:
    required = {
        "training_adata",
        "vocabulary",
        "prior_esm",
        "prior_string",
        "prior_genept",
        "prior_depmap",
        "prior_cell_painting",
    }
    if initialization == "scgpt":
        required.update({"gene_id_map", "scgpt_vocab", "scgpt_checkpoint"})
    elif initialization == "pretrained":
        required.add("pretrained_checkpoint")
    else:
        raise ValueError(f"unexpected initialization mode: {initialization}")
    return required


def _assert_normalization_data_identity(
    normalization_dataset_audit: dict,
    requested_dataset_audit: dict,
) -> None:
    """Allow schedule/source-code audits to differ only when prepared data are identical."""
    data_identity_fields = (
        "shape",
        "vocabulary_size",
        "measured_gene_context",
        "strata",
        "perturbations",
        "contexts",
        "batches",
        "batch_collapsed",
        "target_cells",
        "sets_per_epoch",
        "set_size",
        "padded_target_cells_per_epoch",
        "padded_fraction",
        "adata",
        "adata_sha256",
        "vocabulary",
        "vocabulary_sha256",
    )
    data_mismatches = {
        field: {
            "normalization_linked": normalization_dataset_audit.get(field),
            "requested_run": requested_dataset_audit.get(field),
        }
        for field in data_identity_fields
        if normalization_dataset_audit.get(field) != requested_dataset_audit.get(field)
    }
    if data_mismatches:
        raise ValueError(
            "normalization audit references a non-equivalent prepared dataset: "
            f"{data_mismatches}"
        )
    normalization_prepared = dict(
        normalization_dataset_audit.get("prepared_source_audit", {})
    )
    requested_prepared = dict(requested_dataset_audit.get("prepared_source_audit", {}))
    # Candidate audits deliberately bind a frozen implementation with a new
    # source hash. That is schedule/code provenance, not a change in the exact
    # public source artifacts or reconstructed train/truth matrices.
    normalization_prepared.pop("audited_against_preparation_script_sha256", None)
    requested_prepared.pop("audited_against_preparation_script_sha256", None)
    if normalization_prepared != requested_prepared:
        raise ValueError(
            "normalization-linked and requested dataset audits have different "
            "public-source artifacts/facts"
        )


def _verify_protocol_audits(
    normalization_path: Path,
    feature_space_path: Path,
    training_validation_path: Path,
    *,
    dataset_audit_path: Path,
    prepared_source_audit: dict,
    training_adata: Path,
    truth: Path,
    vocabulary: Path,
) -> dict:
    current_dataset_audit = _load_json(dataset_audit_path)
    normalization = _load_json(normalization_path)
    if (
        normalization.get("status") != "complete"
        or normalization.get("runtime_normalization_action") != "skip_cp10k_and_log1p"
        or normalization.get("paper_protocol_followed_by_normalize_none") is not True
        or normalization.get("double_normalization_detected") is not False
        or normalization.get(
            "upstream_cp10k_denominator_reconstructable_from_processed_release"
        )
        is not False
    ):
        raise ValueError("Replogle normalization audit is incomplete or overclaims provenance")
    normalization_script = _verify_artifact(
        normalization.get("audit_script"), "normalization audit script"
    )
    if normalization_script != Path(__file__).resolve().parent / "audit_replogle_normalization.py":
        raise ValueError("normalization audit was produced by an unexpected script")
    matrix = normalization.get("matrix")
    if (
        not isinstance(matrix, dict)
        or matrix.get("shape") != [643_413, 6_642]
        or matrix.get("x_hvg_shape") != [643_413, 2_000]
        or matrix.get("full_matrix_rows_scanned") != 643_413
        or matrix.get("full_matrix_values_finite") is not True
        or matrix.get("full_matrix_values_nonnegative") is not True
        or matrix.get("full_x_hvg_exact_marked_x_columns") is not True
    ):
        raise ValueError("Replogle normalization full-matrix proof is incomplete")
    normalization_source = normalization.get("source")
    notebook = normalization.get("state_notebook", {}).get("artifact")
    if normalization_source != prepared_source_audit.get("source"):
        raise ValueError("normalization audit and training audit reference different STATE releases")
    if notebook != prepared_source_audit.get("split_notebook"):
        raise ValueError("normalization audit and training audit reference different STATE notebooks")
    prepared_training = normalization.get("prepared_training")
    if (
        not isinstance(prepared_training, dict)
        or prepared_training.get("prepared_expression_exact_source_x_hvg") is not True
        or prepared_training.get(
            "all_training_scripts_explicitly_skip_runtime_normalization"
        )
        is not True
    ):
        raise ValueError("normalization audit lacks prepared-data/runtime skip proof")
    normalization_dataset_audit_path = _verify_artifact(
        prepared_training.get("artifact"), "normalization-linked dataset audit"
    )
    if normalization_dataset_audit_path == dataset_audit_path:
        normalization_linkage = {
            "mode": "direct_dataset_audit_identity",
            "normalization_dataset_audit": {
                "path": str(normalization_dataset_audit_path),
                "sha256": _sha256(normalization_dataset_audit_path),
            },
        }
    else:
        normalization_dataset_audit = _load_json(normalization_dataset_audit_path)
        _assert_normalization_data_identity(
            normalization_dataset_audit, current_dataset_audit
        )
        normalization_linkage = {
            "mode": "exact_prepared_data_identity_across_schedule_audits",
            "normalization_dataset_audit": {
                "path": str(normalization_dataset_audit_path),
                "sha256": _sha256(normalization_dataset_audit_path),
            },
            "requested_dataset_audit": {
                "path": str(dataset_audit_path),
                "sha256": _sha256(dataset_audit_path),
            },
            "schedule_fields_intentionally_allowed_to_differ": [
                "epochs",
                "optimizer_steps",
                "scheduled_microbatches_per_rank",
                "consumed_microbatches_per_rank",
                "optimizer_rounding_extra_microbatches_per_rank",
                "effective_epochs_after_full_accumulation_rounding",
                "dataset_source_sha256",
                "audit_script_sha256",
            ],
        }

    feature_space = _load_json(feature_space_path)
    if (
        feature_space.get("status") != "complete_with_public_spec_ambiguity"
        or feature_space.get("direct_table_b7_protocol_identity") is not False
    ):
        raise ValueError("Replogle feature-space audit is incomplete or overclaims identity")
    feature_script = _verify_artifact(feature_space.get("audit_script"), "feature audit script")
    if feature_script != Path(__file__).resolve().parent / "audit_replogle_feature_space.py":
        raise ValueError("feature-space audit was produced by an unexpected script")
    if _verify_artifact(feature_space.get("training", {}).get("artifact"), "feature-audit train") != training_adata:
        raise ValueError("feature-space audit references a different training matrix")
    if _verify_artifact(feature_space.get("truth", {}).get("artifact"), "feature-audit truth") != truth:
        raise ValueError("feature-space audit references a different truth matrix")
    feature_vocabulary = {
        key: value
        for key, value in feature_space.get("vocabulary", {}).items()
        if key in {"path", "bytes", "sha256"}
    }
    if _verify_artifact(feature_vocabulary, "feature-audit vocabulary") != vocabulary:
        raise ValueError("feature-space audit references a different model vocabulary")
    expected_feature_counts = {
        "genes": 2_000,
        "protein_coding": 1_932,
        "non_protein_coding": 68,
    }
    _check_mapping(
        feature_space.get("released_hvg_feature_space", {}),
        expected_feature_counts,
        "released Replogle feature-space audit",
    )
    if feature_space.get("perturbation_feature_space", {}).get("non_protein_coding") != 0:
        raise ValueError("feature-space audit unexpectedly finds non-protein-coding targets")
    expected_target_forcing = {
        "training_perturbations": 2_015,
        "training_perturbations_measured": 404,
        "training_perturbations_unmeasured": 1_611,
        "truth_perturbations": 380,
        "truth_perturbations_measured": 71,
        "truth_perturbations_unmeasured": 309,
        "union_perturbations": 2_023,
        "union_perturbations_measured": 406,
        "union_perturbations_unmeasured": 1_617,
        "hvg_plus_all_perturbation_coordinates": 3_617,
        "all_perturbations_in_model_vocabulary": True,
        "strict_target_first_possible_for_every_perturbation_in_released_hvg_panel": False,
    }
    if feature_space.get("target_token_forcing") != expected_target_forcing:
        raise ValueError("feature-space audit has unexpected target-token coverage")
    interpretation = feature_space.get("current_clean_room_interpretation")
    if (
        not isinstance(interpretation, dict)
        or interpretation.get("retains_all_released_hvgs") is not True
        or interpretation.get("strictly_protein_coding_vocabulary") is not False
    ):
        raise ValueError("feature-space audit does not record the clean-room interpretation")

    training_validation = _load_json(training_validation_path)
    expected_training_validation = {
        "status": "complete_training_validation_protocol_boundary",
        "direct_private_training_validation_identity": False,
    }
    _check_mapping(
        training_validation,
        expected_training_validation,
        "training-validation protocol boundary",
    )
    claims = training_validation.get("paper", {}).get("claims", {})
    decision = training_validation.get("decision", {})
    implementation = training_validation.get("clean_room_implementation", {})
    released = training_validation.get("released_split", {})
    if (
        claims.get("held_out_validation_monitoring_stated") is not True
        or claims.get("identities_disclosed") is not False
        or claims.get("evaluation_cadence_disclosed") is not False
        or claims.get("checkpoint_selection_rule_disclosed") is not False
        or released.get("train_shape") != [616_535, 2_000]
        or released.get("present_validation_genes_included_in_training") != 60
        or released.get("present_validation_cells_included_in_training") != 4_825
        or implementation.get("checkpoint_policy")
        != "final_scheduled_optimizer_update"
        or implementation.get("test_set_used_for_checkpoint_selection") is not False
        or implementation.get("appendix_a6_monitoring_reproduced") is not False
        or decision.get("train_all_published_616535_cells") is not True
        or decision.get("reserve_published_380_gene_test_set") is not True
        or decision.get("use_test_metrics_for_checkpoint_selection") is not False
        or decision.get("report_final_scheduled_checkpoint") is not True
        or decision.get("claim_appendix_a6_identity") is not False
    ):
        raise ValueError("training-validation protocol boundary is incomplete")
    audit_dir = Path(__file__).resolve().parent
    source_dir = audit_dir.parent / "src" / "open_x_cell"
    expected_records = {
        "metadata": truth.parent / "metadata.json",
        "split_audit": truth.parent / "split_audit.json",
        "training_runner": audit_dir / "train_paper_mini_loss_profile.py",
        "cli_source": source_dir / "cli.py",
        "training_source": source_dir / "training.py",
        "auditor": audit_dir / "audit_training_validation_protocol.py",
    }
    observed_records = {
        "metadata": released.get("metadata"),
        "split_audit": released.get("split_audit"),
        "training_runner": implementation.get("training_runner"),
        "cli_source": implementation.get("cli_source"),
        "training_source": implementation.get("training_source"),
        "auditor": training_validation.get("auditor"),
    }
    for name, expected_path in expected_records.items():
        observed_path = _verify_artifact(
            observed_records[name], f"training-validation {name}"
        )
        if name in {"metadata", "split_audit"}:
            if observed_path != expected_path.resolve():
                raise ValueError(f"training-validation {name} points outside formal data")
        elif _sha256(observed_path) != _sha256(expected_path):
            raise ValueError(f"training-validation {name} differs from current source")
    paper_pdf = _verify_artifact(
        training_validation.get("paper", {}).get("pdf"),
        "training-validation paper PDF",
    )
    if _sha256(paper_pdf) != (
        "54bb91132d46b897a1173857fdc7257168500bc5909ff820c"
        "1b3b748999d7867"
    ):
        raise ValueError("training-validation boundary uses another paper snapshot")
    return {
        "normalization": {
            "path": str(normalization_path),
            "sha256": _sha256(normalization_path),
            "report": normalization,
            "dataset_audit_linkage": normalization_linkage,
        },
        "feature_space": {
            "path": str(feature_space_path),
            "sha256": _sha256(feature_space_path),
            "report": feature_space,
        },
        "training_validation_boundary": {
            "path": str(training_validation_path),
            "sha256": _sha256(training_validation_path),
            "report": training_validation,
        },
    }


def _verify_target_complete_profile(
    schedule_path: Path,
    normalization_path: Path,
    feature_space_path: Path,
    training_validation_path: Path,
    *,
    training_adata: Path,
    truth: Path,
    perturbations: Path,
    vocabulary: Path,
    input_artifacts: dict,
    manifest_source_sha256: dict[str, str],
    expected_epochs: int,
    expected_steps: int,
    expected_world_size: int,
    expected_gradient_checkpointing: bool,
) -> tuple[dict, dict, Path]:
    """Verify the 6,808-gene public proxy and its released-HVG evaluation base."""
    schedule = _load_json(schedule_path)
    expected_schedule = {
        "status": "complete_public_replogle_target_complete_training_schedule",
        "direct_table_a3_full_context_identity": False,
        "shape": [616_535, 6_808],
        "measured_genes": 6_642,
        "zero_appended_genes": 166,
        "training_targets_using_appended_zero_coordinate": 161,
        "heldout_only_targets_using_appended_zero_coordinate": 5,
        "contexts": ["hepg2", "jurkat", "k562", "rpe1"],
        "batches": 1,
        "strata": 5_379,
        "perturbations": 2_015,
        "control_cells": 39_165,
        "target_cells": 577_370,
        "sets_per_epoch": 11_524,
        "set_size": 64,
        "padded_target_cells_per_epoch": 160_166,
        "world_size": expected_world_size,
        "batch_size_per_rank": 1,
        "gradient_accumulation_steps": 32,
        "samples_per_rank_per_epoch": 5_762,
        "microbatches_per_rank_per_epoch": 5_762,
        "epochs": expected_epochs,
        "optimizer_steps": expected_steps,
        "scheduled_microbatches_per_rank": expected_epochs * 5_762,
        "consumed_microbatches_per_rank": expected_steps * 32,
        "optimizer_rounding_extra_microbatches_per_rank": (
            expected_steps * 32 - expected_epochs * 5_762
        ),
        "effective_sets_per_optimizer_step": 64,
        "effective_target_cells_per_optimizer_step": 4_096,
        "all_training_targets_forceable_first": True,
        "all_strata_target_first_verified": True,
        "backed_vds_missing_target_read_smoke_verified": True,
    }
    _check_mapping(schedule, expected_schedule, "target-complete training schedule")
    if not math.isclose(
        float(schedule.get("padded_fraction", -1)),
        160_166 / (11_524 * 64),
        rel_tol=0,
        abs_tol=1e-15,
    ) or not math.isclose(
        float(schedule.get("effective_epochs_after_full_accumulation_rounding", -1)),
        expected_steps * 32 / 5_762,
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise ValueError("target-complete padding or epoch rounding differs")
    if schedule.get("source_sha256") != manifest_source_sha256:
        raise ValueError("target-complete schedule used different package sources")
    schedule_driver = _verify_artifact(schedule.get("auditor"), "target schedule auditor")
    if schedule_driver != Path(__file__).resolve().parent / "audit_replogle_target_complete_schedule.py":
        raise ValueError("target-complete schedule was produced by an unexpected auditor")
    if _verify_artifact(schedule.get("adata"), "target schedule AnnData") != training_adata:
        raise ValueError("target-complete schedule points to a different training AnnData")
    if _verify_artifact(schedule.get("vocabulary"), "target schedule vocabulary") != vocabulary:
        raise ValueError("target-complete schedule points to a different vocabulary")

    preparation_path = _verify_artifact(
        schedule.get("preparation"), "target-complete preparation metadata"
    )
    preparation = _load_json(preparation_path)
    expected_preparation = {
        "status": "complete_public_replogle_full_measured_target_complete_input",
        "direct_table_a3_full_context_identity": False,
        "shape": [616_535, 6_808],
        "training_cells": 616_535,
        "public_measured_genes": 6_642,
        "zero_appended_missing_target_genes": 166,
        "zero_appended_missing_training_target_genes": 161,
        "zero_appended_heldout_only_target_genes": 5,
        "all_training_targets_forceable_first": True,
        "all_public_observed_perturbations_addressable": True,
    }
    _check_mapping(preparation, expected_preparation, "target-complete preparation")
    appended = preparation.get("appended_target_genes")
    if not isinstance(appended, list) or len(appended) != 166 or len(set(appended)) != 166:
        raise ValueError("target-complete appended-target inventory differs")
    if _verify_artifact(preparation.get("output_adata"), "target-complete output") != training_adata:
        raise ValueError("target-complete preparation output differs from the training input")
    if _verify_artifact(preparation.get("vocabulary"), "target-complete vocabulary") != vocabulary:
        raise ValueError("target-complete preparation vocabulary differs from training")
    preparation_driver = _verify_artifact(
        preparation.get("preparation_script"), "target-complete preparation script"
    )
    if preparation_driver != Path(__file__).resolve().parent / "prepare_replogle_full_measured_target_complete.py":
        raise ValueError("target-complete input was produced by an unexpected script")
    logical = preparation.get("logical_matrix", {})
    if (
        logical.get("values") != 616_535 * 6_808
        or logical.get("minimum") != 0.0
        or logical.get("measured_block_exact_external_source") is not True
        or logical.get("all_appended_values_exact_zero") is not True
        or logical.get("all_values_finite_nonnegative") is not True
        or logical.get("hdf5_virtual_dataset") is not True
        or not isinstance(logical.get("logical_matrix_sha256"), str)
        or len(logical["logical_matrix_sha256"]) != 64
    ):
        raise ValueError("target-complete logical-matrix proof is incomplete")
    with h5py.File(training_adata, "r") as handle:
        matrix = handle.get("X")
        if (
            not isinstance(matrix, h5py.Dataset)
            or matrix.shape != (616_535, 6_808)
            or not matrix.is_virtual
        ):
            raise ValueError("target-complete training X is not the audited HDF5 VDS")

    measured_preparation_path = _verify_artifact(
        preparation.get("measured_preparation"), "full-measured preparation metadata"
    )
    measured_preparation = _load_json(measured_preparation_path)
    expected_measured = {
        "status": "complete_public_replogle_full_measured_training_input",
        "direct_table_a3_full_context_identity": False,
        "shape": [616_535, 6_642],
        "training_cells": 616_535,
        "public_measured_genes": 6_642,
        "test_perturbations": 380,
        "held_out_test_cells": 26_878,
    }
    _check_mapping(measured_preparation, expected_measured, "full-measured preparation")
    measured_adata = _verify_artifact(
        measured_preparation.get("output_adata"), "full-measured training AnnData"
    )
    if _verify_artifact(preparation.get("measured_adata"), "target measured source") != measured_adata:
        raise ValueError("target-complete VDS points to a different measured source")
    binding = preparation.get("external_source_binding", {})
    if (
        Path(str(binding.get("path", ""))).resolve() != measured_adata
        or binding.get("sha256") != _sha256(measured_adata)
        or binding.get("hdf5_dataset") != "X"
        or binding.get("mapped_columns") != [0, 6_642]
        or binding.get("fill_value_for_appended_columns") != 0.0
    ):
        raise ValueError("target-complete external-source binding differs")
    if _verify_artifact(measured_preparation.get("output_vocabulary"), "full vocabulary") != vocabulary:
        raise ValueError("full-measured and target-complete vocabularies differ")
    if _verify_artifact(measured_preparation.get("test_perturbation_artifact"), "test perturbations") != perturbations:
        raise ValueError("full-measured preparation used a different test split")
    public_source = _verify_artifact(measured_preparation.get("source"), "public STATE matrix")
    if _verify_artifact(preparation.get("public_source"), "target public source") != public_source:
        raise ValueError("full-measured and target-complete public sources differ")
    measured_driver = _verify_artifact(
        measured_preparation.get("preparation_script"), "full-measured preparation script"
    )
    if measured_driver != Path(__file__).resolve().parent / "prepare_replogle_full_measured_training.py":
        raise ValueError("full-measured input was produced by an unexpected script")
    matrix_copy = measured_preparation.get("matrix_copy", {})
    if (
        matrix_copy.get("values") != 616_535 * 6_642
        or matrix_copy.get("source_selected_matrix_exact") is not True
        or matrix_copy.get("all_values_finite_nonnegative") is not True
        or matrix_copy.get("selected_source_matrix_sha256")
        != matrix_copy.get("output_matrix_sha256")
    ):
        raise ValueError("full-measured matrix-copy proof is incomplete")

    base_vocabulary = _verify_artifact(
        measured_preparation.get("base_vocabulary"), "released-HVG base vocabulary"
    )
    hvg_training = _verify_artifact(
        measured_preparation.get("hvg_training_reference"), "released-HVG training reference"
    )
    normalization = _load_json(normalization_path)
    base_dataset_audit_path = _verify_artifact(
        normalization.get("prepared_training", {}).get("artifact"),
        "released-HVG dataset audit",
    )
    base_dataset_audit = _load_json(base_dataset_audit_path)
    prepared_source_audit = base_dataset_audit.get("prepared_source_audit")
    if not isinstance(prepared_source_audit, dict):
        raise ValueError("released-HVG dataset audit lacks source reconstruction proof")
    protocol = _verify_protocol_audits(
        normalization_path,
        feature_space_path,
        training_validation_path,
        dataset_audit_path=base_dataset_audit_path,
        prepared_source_audit=prepared_source_audit,
        training_adata=hvg_training,
        truth=truth,
        vocabulary=base_vocabulary,
    )
    protocol["target_complete_training"] = {
        "schedule": {"path": str(schedule_path), "sha256": _sha256(schedule_path)},
        "preparation": {
            "path": str(preparation_path),
            "sha256": _sha256(preparation_path),
            "report": preparation,
        },
        "measured_preparation": {
            "path": str(measured_preparation_path),
            "sha256": _sha256(measured_preparation_path),
            "report": measured_preparation,
        },
        "training_artifact": input_artifacts["training_adata"],
        "vocabulary_artifact": input_artifacts["vocabulary"],
    }
    workspace = training_adata.parents[3]
    literal_gate_path = (
        workspace
        / "runs/hardware_feasibility/mini_replogle_target_complete_6808_table_a3_training_gate.json"
    )
    literal_gate = _load_json(literal_gate_path)
    literal_configuration = literal_gate.get("configuration", {})
    if (
        literal_gate.get("direct_table_a3_full_context_identity") is not False
        or literal_configuration.get("genes") != 6_808
        or literal_configuration.get("set_size") != 64
        or literal_configuration.get("gradient_checkpointing") is not False
        or literal_configuration.get("matches_table_a3_no_gradient_checkpointing") is not True
        or literal_gate.get("source_sha256") != manifest_source_sha256
    ):
        raise ValueError("target-complete literal Table A3 gate configuration differs")
    literal_validator = _verify_artifact(
        literal_gate.get("validator"), "literal target-complete gate validator"
    )
    if literal_validator != Path(__file__).resolve().parent / "validate_target_complete_hardware_gate.py":
        raise ValueError("literal target-complete gate used an unexpected validator")

    runtime_gates: dict[str, dict] = {}
    if expected_gradient_checkpointing:
        literal_outcome = literal_gate.get("outcome", {})
        if (
            literal_gate.get("status")
            != "blocked_public_target_complete6808_table_a3_hardware_gate"
            or literal_gate.get("classification_complete") is not True
            or literal_gate.get("training_authorized") is not False
            or literal_outcome.get("feasible") is not False
            or literal_outcome.get("failure") != "cuda_out_of_memory"
            or not str(literal_outcome.get("error", ""))
        ):
            raise ValueError("literal no-checkpointing path is not a captured fail-closed OOM")
        hardware_gate_path = (
            workspace
            / "runs/hardware_feasibility/mini_replogle_target_complete_6808_checkpointed_fallback_gate.json"
        )
        hardware_gate = _load_json(hardware_gate_path)
        configuration = hardware_gate.get("configuration", {})
        if (
            hardware_gate.get("status")
            != "passed_public_target_complete6808_checkpointed_fallback_hardware_gate"
            or hardware_gate.get("training_authorized") is not True
            or hardware_gate.get("direct_table_a3_full_context_identity") is not False
            or hardware_gate.get("matches_table_a3_no_gradient_checkpointing") is not False
            or configuration.get("genes") != 6_808
            or configuration.get("set_size") != 64
            or configuration.get("gradient_checkpointing") is not True
            or configuration.get("expected_parameters") != 60_001_618
            or hardware_gate.get("outcome", {}).get("feasible") is not True
            or hardware_gate.get("outcome", {}).get("parameters") != 60_001_618
            or hardware_gate.get("source_sha256") != manifest_source_sha256
            or _verify_artifact(
                hardware_gate.get("blocked_literal_table_a3_gate"),
                "checkpointed fallback blocked literal gate",
            )
            != literal_gate_path.resolve()
        ):
            raise ValueError("target-complete checkpointed fallback gate is not passed")
        expected_probe_path = (
            workspace
            / "runs/hardware_feasibility/mini_replogle_target_complete_6808_a6000.json"
        )
        if _verify_artifact(
            hardware_gate.get("checkpointed_probe"), "checkpointed fallback probe"
        ) != expected_probe_path.resolve():
            raise ValueError("checkpointed fallback gate points to a different probe")
        fallback_validator = _verify_artifact(
            hardware_gate.get("validator"), "checkpointed fallback gate validator"
        )
        if fallback_validator != (
            Path(__file__).resolve().parent
            / "validate_target_complete_checkpointed_fallback_gate.py"
        ):
            raise ValueError("checkpointed fallback gate used an unexpected validator")
        smoke_path = (
            workspace
            / "runs/hardware_feasibility/xcell_mini_replogle_target_complete6808_checkpointed_one_step_smoke/smoke_audit.json"
        )
        expected_smoke_status = (
            "complete_target_complete6808_checkpointed_one_optimizer_step_smoke"
        )
        runtime_gates["literal_table_a3"] = {
            "path": str(literal_gate_path),
            "sha256": _sha256(literal_gate_path),
            "report": literal_gate,
        }
        runtime_gates["checkpointed_fallback"] = {
            "path": str(hardware_gate_path),
            "sha256": _sha256(hardware_gate_path),
            "report": hardware_gate,
        }
    else:
        if (
            literal_gate.get("status")
            != "passed_public_target_complete6808_table_a3_hardware_gate"
            or literal_gate.get("training_authorized") is not True
            or literal_gate.get("outcome", {}).get("feasible") is not True
        ):
            raise ValueError("target-complete no-checkpointing hardware gate is not passed")
        hardware_gate_path = literal_gate_path
        hardware_gate = literal_gate
        smoke_path = (
            workspace
            / "runs/hardware_feasibility/xcell_mini_replogle_target_complete6808_table_a3_one_step_smoke/smoke_audit.json"
        )
        expected_smoke_status = (
            "complete_target_complete6808_table_a3_one_optimizer_step_smoke"
        )
        runtime_gates["hardware"] = {
            "path": str(hardware_gate_path),
            "sha256": _sha256(hardware_gate_path),
            "report": hardware_gate,
        }

    smoke = _load_json(smoke_path)
    if (
        smoke.get("status") != expected_smoke_status
        or smoke.get("training_authorized") is not True
        or smoke.get("exact_formal_training_path_exercised") is not True
        or smoke.get("optimizer_steps") != 1
        or smoke.get("gradient_accumulation_microbatches_per_rank") != 32
        or smoke.get("effective_sets") != 64
        or smoke.get("effective_cells") != 4_096
        or smoke.get("training_genes") != 6_808
        or smoke.get("gradient_checkpointing") is not expected_gradient_checkpointing
        or smoke.get("source_sha256") != manifest_source_sha256
        or _verify_artifact(smoke.get("hardware_gate"), "training-smoke gate")
        != hardware_gate_path.resolve()
        or _verify_artifact(smoke.get("training_schedule"), "training-smoke schedule")
        != schedule_path.resolve()
        or _verify_artifact(smoke.get("input_preparation"), "training-smoke preparation")
        != preparation_path.resolve()
    ):
        raise ValueError("target-complete exact one-step training smoke is incomplete")
    smoke_inputs = smoke.get("input_artifacts", {})
    if (
        not isinstance(smoke_inputs, dict)
        or _verify_artifact(smoke_inputs.get("training_adata"), "smoke training input")
        != training_adata
        or _verify_artifact(smoke_inputs.get("vocabulary"), "smoke vocabulary")
        != vocabulary
    ):
        raise ValueError("target-complete smoke and formal run use different primary inputs")
    smoke_validator = _verify_artifact(
        smoke.get("validator"), "target-complete training-smoke validator"
    )
    if smoke_validator != Path(__file__).resolve().parent / "validate_target_complete_training_smoke.py":
        raise ValueError("target-complete training smoke used an unexpected validator")
    runtime_gates["one_optimizer_step"] = {
        "path": str(smoke_path),
        "sha256": _sha256(smoke_path),
        "report": smoke,
    }
    protocol["target_complete_training"]["runtime_gates"] = runtime_gates
    return protocol, prepared_source_audit, base_vocabulary


def _verify_raw_union_dynamic4k_profile(
    schedule_path: Path,
    normalization_path: Path,
    feature_space_path: Path,
    training_validation_path: Path,
    *,
    training_adata: Path,
    truth: Path,
    perturbations: Path,
    vocabulary: Path,
    input_artifacts: dict,
    manifest_source_sha256: dict[str, str],
    expected_epochs: int,
    expected_steps: int,
    expected_world_size: int,
    expected_gradient_checkpointing: bool,
) -> tuple[dict, dict, Path]:
    """Verify the raw-release union with A.10.2 dynamic 4,000-gene contexts."""
    if not expected_gradient_checkpointing:
        raise ValueError("the measured A6000 dynamic-4k fallback requires checkpointing")
    schedule = _load_json(schedule_path)
    expected_schedule = {
        "status": "complete_public_replogle_raw_union_dynamic4k_training_schedule",
        "direct_table_a3_full_context_identity": False,
        "direct_table_b7_comparison": False,
        "shape": [616_535, 11_332],
        "stored_public_union_genes": 11_332,
        "dynamic_context_genes": 4_000,
        "context_selection": "fresh deterministic per-set target-first subsample",
        "context_specific_absent_coordinate_policy": "explicit_zero_proxy",
        "contexts": ["hepg2", "jurkat", "k562", "rpe1"],
        "batches": 1,
        "strata": 5_379,
        "perturbations": 2_015,
        "control_cells": 39_165,
        "target_cells": 577_370,
        "sets_per_epoch": 11_524,
        "set_size": 64,
        "padded_target_cells_per_epoch": 160_166,
        "world_size": expected_world_size,
        "batch_size_per_rank": 1,
        "gradient_accumulation_steps": 32,
        "samples_per_rank_per_epoch": 5_762,
        "microbatches_per_rank_per_epoch": 5_762,
        "epochs": expected_epochs,
        "optimizer_steps": expected_steps,
        "scheduled_microbatches_per_rank": expected_epochs * 5_762,
        "consumed_microbatches_per_rank": expected_steps * 32,
        "optimizer_rounding_extra_microbatches_per_rank": (
            expected_steps * 32 - expected_epochs * 5_762
        ),
        "effective_sets_per_optimizer_step": 64,
        "effective_target_cells_per_optimizer_step": 4_096,
        "all_training_targets_in_stored_union": True,
        "all_strata_target_first_verified": True,
        "dynamic_context_unique_gene_count_verified": True,
        "backed_dynamic_context_read_smoke_verified": True,
    }
    _check_mapping(schedule, expected_schedule, "raw-union dynamic-4k schedule")
    if (
        not math.isclose(
            float(schedule.get("padded_fraction", -1)),
            160_166 / (11_524 * 64),
            rel_tol=0,
            abs_tol=1e-15,
        )
        or not math.isclose(
            float(schedule.get("effective_epochs_after_full_accumulation_rounding", -1)),
            expected_steps * 32 / 5_762,
            rel_tol=0,
            abs_tol=1e-15,
        )
        or schedule.get("source_sha256") != manifest_source_sha256
    ):
        raise ValueError("raw-union dynamic-4k schedule arithmetic/source differs")
    schedule_driver = _verify_artifact(schedule.get("auditor"), "raw-union schedule auditor")
    if schedule_driver != (
        Path(__file__).resolve().parent / "audit_replogle_raw_union_dynamic4k_schedule.py"
    ):
        raise ValueError("raw-union schedule used an unexpected auditor")
    if _verify_artifact(schedule.get("adata"), "raw-union schedule AnnData") != training_adata:
        raise ValueError("raw-union schedule points to a different training input")
    if _verify_artifact(schedule.get("vocabulary"), "raw-union schedule vocabulary") != vocabulary:
        raise ValueError("raw-union schedule points to a different vocabulary")

    preparation_path = _verify_artifact(schedule.get("preparation"), "raw-union preparation")
    preparation = _load_json(preparation_path)
    if (
        preparation.get("status")
        != "complete_public_replogle_raw_source_union_training_input"
        or preparation.get("direct_table_a3_full_context_identity") is not False
        or preparation.get("direct_table_b7_comparison") is not False
        or preparation.get("shape") != [616_535, 11_332]
        or preparation.get("training_cells") != 616_535
        or preparation.get("vocabulary_aligned_raw_union_genes") != 11_332
        or preparation.get("context_specific_absent_coordinate_policy")
        != "explicit_zero_proxy"
        or preparation.get("matrix", {}).get("values") != 6_986_574_620
        or preparation.get("matrix", {}).get("all_values_finite_nonnegative") is not True
        or preparation.get("matrix", {}).get("terminal_rescan_exact") is not True
        or preparation.get("matrix", {}).get("all_shared_state_values_match_at_1e-6")
        is not True
        or _verify_artifact(preparation.get("output_adata"), "raw-union prepared AnnData")
        != training_adata
        or _verify_artifact(preparation.get("vocabulary"), "raw-union prepared vocabulary")
        != vocabulary
    ):
        raise ValueError("raw-union preparation is incomplete or points elsewhere")
    preparation_driver = _verify_artifact(
        preparation.get("preparation_script"), "raw-union preparation driver"
    )
    if preparation_driver != Path(__file__).resolve().parent / "prepare_replogle_raw_source_union.py":
        raise ValueError("raw-union input used an unexpected preparation driver")
    raw_boundary_path = _verify_artifact(
        preparation.get("raw_union_audit"), "raw-union reconstruction boundary"
    )
    if _verify_artifact(schedule.get("raw_union_boundary"), "schedule raw-union boundary") != raw_boundary_path:
        raise ValueError("raw-union schedule/preparation boundary linkage differs")
    raw_boundary = _load_json(raw_boundary_path)
    if (
        raw_boundary.get("status")
        != "complete_public_replogle_raw_source_union_boundary"
        or raw_boundary.get("decision", {}).get("public_raw_union_preparation_authorized")
        is not True
        or raw_boundary.get("decision", {}).get("claim_private_xaira_processed_matrix_identity")
        is not False
    ):
        raise ValueError("raw-union reconstruction boundary overclaims or is incomplete")

    normalization = _load_json(normalization_path)
    base_dataset_audit_path = _verify_artifact(
        normalization.get("prepared_training", {}).get("artifact"),
        "released-HVG dataset audit",
    )
    base_dataset_audit = _load_json(base_dataset_audit_path)
    prepared_source_audit = base_dataset_audit.get("prepared_source_audit")
    if not isinstance(prepared_source_audit, dict):
        raise ValueError("released-HVG dataset audit lacks source reconstruction proof")
    base_training = Path(str(base_dataset_audit.get("adata", ""))).resolve()
    base_vocabulary = Path(str(base_dataset_audit.get("vocabulary", ""))).resolve()
    if (
        not base_training.is_file()
        or not base_vocabulary.is_file()
        or _verify_artifact(preparation.get("training_reference"), "raw-union training reference")
        != base_training
        or _verify_artifact(preparation.get("test_perturbations"), "raw-union test split")
        != perturbations
    ):
        raise ValueError("raw-union branch is not linked to the released-HVG base split")
    protocol = _verify_protocol_audits(
        normalization_path,
        feature_space_path,
        training_validation_path,
        dataset_audit_path=base_dataset_audit_path,
        prepared_source_audit=prepared_source_audit,
        training_adata=base_training,
        truth=truth,
        vocabulary=base_vocabulary,
    )
    protocol["raw_union_dynamic4k_training"] = {
        "schedule": {
            "path": str(schedule_path),
            "sha256": _sha256(schedule_path),
            "report": schedule,
        },
        "preparation": {
            "path": str(preparation_path),
            "sha256": _sha256(preparation_path),
            "report": preparation,
        },
        "raw_union_boundary": {
            "path": str(raw_boundary_path),
            "sha256": _sha256(raw_boundary_path),
            "report": raw_boundary,
        },
        "training_artifact": input_artifacts["training_adata"],
        "vocabulary_artifact": input_artifacts["vocabulary"],
    }

    workspace = training_adata.parents[3]
    gate_path = (
        workspace
        / "runs/hardware_feasibility/mini_replogle_raw_union_dynamic4000_checkpointed_training_gate.json"
    )
    gate = _load_json(gate_path)
    if (
        gate.get("status")
        != "passed_public_raw_union11332_dynamic4000_checkpointed_hardware_gate"
        or gate.get("training_authorized") is not True
        or gate.get("direct_table_a3_full_context_identity") is not False
        or gate.get("stored_public_union_genes") != 11_332
        or gate.get("configuration", {}).get("dynamic_context_genes") != 4_000
        or gate.get("configuration", {}).get("set_size") != 64
        or gate.get("configuration", {}).get("gradient_checkpointing") is not True
        or gate.get("outcome", {}).get("feasible") is not True
        or gate.get("outcome", {}).get("parameters") != 60_001_618
        or gate.get("source_sha256") != manifest_source_sha256
    ):
        raise ValueError("raw-union dynamic-4k checkpointed gate is not passed")
    gate_validator = _verify_artifact(gate.get("validator"), "raw-union dynamic gate validator")
    if gate_validator != (
        Path(__file__).resolve().parent / "validate_raw_union_dynamic4k_checkpointed_gate.py"
    ):
        raise ValueError("raw-union dynamic gate used an unexpected validator")
    smoke_path = (
        workspace
        / "runs/hardware_feasibility/xcell_mini_replogle_raw_union11332_dynamic4000_checkpointed_one_step_smoke/smoke_audit.json"
    )
    smoke = _load_json(smoke_path)
    if (
        smoke.get("status")
        != "complete_raw_union11332_dynamic4000_checkpointed_one_optimizer_step_smoke"
        or smoke.get("training_authorized") is not True
        or smoke.get("exact_formal_training_path_exercised") is not True
        or smoke.get("optimizer_steps") != 1
        or smoke.get("gradient_accumulation_microbatches_per_rank") != 32
        or smoke.get("effective_sets") != 64
        or smoke.get("effective_cells") != 4_096
        or smoke.get("stored_training_genes") != 11_332
        or smoke.get("dynamic_context_genes") != 4_000
        or smoke.get("gradient_checkpointing") is not True
        or smoke.get("source_sha256") != manifest_source_sha256
        or _verify_artifact(smoke.get("hardware_gate"), "raw-union smoke gate")
        != gate_path.resolve()
        or _verify_artifact(smoke.get("training_schedule"), "raw-union smoke schedule")
        != schedule_path.resolve()
        or _verify_artifact(smoke.get("input_preparation"), "raw-union smoke preparation")
        != preparation_path.resolve()
    ):
        raise ValueError("raw-union dynamic-4k one-step smoke is incomplete")
    smoke_inputs = smoke.get("input_artifacts", {})
    if (
        not isinstance(smoke_inputs, dict)
        or _verify_artifact(smoke_inputs.get("training_adata"), "raw-union smoke input")
        != training_adata
        or _verify_artifact(smoke_inputs.get("vocabulary"), "raw-union smoke vocabulary")
        != vocabulary
    ):
        raise ValueError("raw-union smoke and formal run use different primary inputs")
    smoke_validator = _verify_artifact(smoke.get("validator"), "raw-union smoke validator")
    if smoke_validator != (
        Path(__file__).resolve().parent / "validate_raw_union_dynamic4k_training_smoke.py"
    ):
        raise ValueError("raw-union smoke used an unexpected validator")
    protocol["raw_union_dynamic4k_training"]["runtime_gates"] = {
        "checkpointed_hardware": {
            "path": str(gate_path),
            "sha256": _sha256(gate_path),
            "report": gate,
        },
        "one_optimizer_step": {
            "path": str(smoke_path),
            "sha256": _sha256(smoke_path),
            "report": smoke,
        },
    }
    return protocol, prepared_source_audit, base_vocabulary


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a JSON number, found {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite: {value!r}")
    return result


def _read_backed_rows_in_gene_order(
    adata: ad.AnnData,
    row_indexer: Any,
    gene_names: Any,
    *,
    label: str,
) -> np.ndarray:
    """Read backed rows first, then reorder columns in memory.

    h5py permits only one fancy index per dataset access. AnnData otherwise
    forwards simultaneous row and column arrays to h5py for backed dense
    matrices, so applying both selectors in one view fails before validation.
    """
    requested_genes = list(map(str, gene_names))
    if len(requested_genes) != len(set(requested_genes)):
        raise ValueError(f"{label} requested gene ordering contains duplicates")
    column_positions = adata.var_names.get_indexer(requested_genes)
    missing = [
        requested_genes[index]
        for index, position in enumerate(column_positions)
        if position < 0
    ]
    if missing:
        raise ValueError(f"{label} genes are absent from the source: {missing[:10]}")

    matrix = adata[row_indexer, :].X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{label} source expression is not two-dimensional")
    return values[:, column_positions]


def _read_metric_results(path: Path, expected_perturbations: set[str]) -> tuple[dict, list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"official metric output is empty: {path}")
    row_perts = [str(row["perturbation"]) for row in rows]
    if len(row_perts) != len(set(row_perts)):
        raise ValueError("cell-eval results contain duplicate perturbation rows")
    if set(row_perts) != expected_perturbations:
        missing = sorted(expected_perturbations - set(row_perts))
        extra = sorted(set(row_perts) - expected_perturbations)
        raise ValueError(f"cell-eval perturbation mismatch: missing={missing[:10]}, extra={extra[:10]}")

    columns = list(rows[0])
    absent = [metric for metric in PAPER_METRICS if metric not in columns]
    if absent:
        raise ValueError(f"cell-eval results are missing paper metrics: {absent}")

    summary: dict[str, dict[str, float | int]] = {}
    for metric in PAPER_METRICS:
        values: list[float] = []
        for row in rows:
            raw = row.get(metric, "")
            if raw is None or raw.strip().lower() in {"", "nan", "null", "none"}:
                continue
            value = float(raw)
            if math.isnan(value):
                continue
            if not math.isfinite(value):
                raise ValueError(f"cell-eval metric {metric} contains an infinite value")
            lower, upper = METRIC_BOUNDS[metric]
            if value < lower - 1e-8 or value > upper + 1e-8:
                raise ValueError(f"cell-eval metric {metric} is out of bounds: {value}")
            values.append(value)
        if not values:
            raise ValueError(f"cell-eval metric {metric} has no finite values")
        array = np.asarray(values, dtype=np.float64)
        summary[metric] = {
            "mean": float(array.mean()),
            # cell-eval 0.6.8 delegates the 50% aggregate to Polars, whose
            # quantile uses nearest interpolation. For an even number of rows
            # this selects the upper middle value instead of NumPy's average of
            # the two middle values.
            "median": float(np.quantile(array, 0.5, method="nearest")),
            "min": float(array.min()),
            "max": float(array.max()),
            "finite_count": int(array.size),
        }
    return summary, columns


def main() -> int:
    args = parse_args()
    if args.expected_epochs <= 0:
        raise ValueError("expected epochs must be positive")
    if not math.isfinite(args.expected_learning_rate) or args.expected_learning_rate <= 0:
        raise ValueError("expected learning rate must be finite and positive")
    if args.expected_initialization == "pretrained" and args.expected_pretrained_checkpoint is None:
        raise ValueError("pretrained initialization requires an expected source checkpoint")
    if (
        args.expected_initialization != "scgpt"
        and args.expected_scgpt_transfer_scope != "appendix_a7"
    ):
        raise ValueError("scGPT transfer scope applies only to scGPT initialization")
    if args.expected_initialization == "scgpt" and args.expected_pretrained_checkpoint is not None:
        raise ValueError("scGPT initialization must not declare a pretrained checkpoint")
    run_dir = Path(args.run_dir).resolve()
    truth_path = Path(args.truth).resolve()
    perturbations_path = Path(args.perturbations).resolve()
    target_complete_profile = args.training_input_profile == "public_target_complete6808"
    raw_union_profile = (
        args.training_input_profile == "public_raw_union11332_dynamic4000"
    )
    extended_public_profile = target_complete_profile or raw_union_profile
    training_genes = (
        6_808
        if target_complete_profile
        else 11_332
        if raw_union_profile
        else args.expected_genes
    )
    expected_training_context = 4_000 if raw_union_profile else training_genes
    expected_backed = extended_public_profile
    expected_requested_context = expected_training_context if extended_public_profile else None
    if extended_public_profile and args.dataset_audit is None:
        raise ValueError(
            f"{args.training_input_profile} requires --dataset-audit pointing to its schedule audit"
        )
    dataset_audit_path = (
        Path(args.dataset_audit).resolve()
        if args.dataset_audit is not None
        else truth_path.parent / "training_dataset_audit_corrected.json"
    )
    normalization_audit_path = (
        Path(args.normalization_audit).resolve()
        if args.normalization_audit is not None
        else truth_path.parent / "normalization_audit.json"
    )
    feature_space_audit_path = (
        Path(args.feature_space_audit).resolve()
        if args.feature_space_audit is not None
        else truth_path.parent / "feature_space_audit.json"
    )
    training_validation_audit_path = (
        Path(args.training_validation_audit).resolve()
        if args.training_validation_audit is not None
        else truth_path.parent / "training_validation_protocol_audit.json"
    )
    output = Path(args.output).resolve()
    temporary = output.with_suffix(output.suffix + ".tmp")
    protected_direct_inputs = {
        truth_path,
        perturbations_path,
        dataset_audit_path,
        normalization_audit_path,
        feature_space_audit_path,
        training_validation_audit_path,
    }
    if {output, temporary} & protected_direct_inputs:
        raise ValueError("audit output and temporary paths must not overwrite benchmark inputs")
    existing_run_artifacts = {
        path.resolve()
        for path in run_dir.rglob("*")
        if path.is_file() and path.resolve() not in {output, temporary}
    }
    if {output, temporary} & existing_run_artifacts:
        raise ValueError("audit output and temporary paths must not overwrite run artifacts")
    runtime_versions = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_cuda_build": str(torch.version.cuda),
        **{
            package: importlib.metadata.version(package)
            for package in (
                "anndata",
                "numpy",
                "pandas",
                "scipy",
                "geomloss",
                "polars",
                "cell-eval",
            )
        },
    }
    if runtime_versions != EXPECTED_RUNTIME_VERSIONS:
        raise ValueError(
            "formal runtime versions differ from the audited environment: "
            f"expected={EXPECTED_RUNTIME_VERSIONS}, actual={runtime_versions}"
        )
    if not truth_path.is_file():
        raise FileNotFoundError(truth_path)
    perturbation_payload = _load_json(perturbations_path)
    if not isinstance(perturbation_payload, list) or not perturbation_payload:
        raise ValueError("formal perturbations must be a non-empty JSON list")
    if any(
        not isinstance(value, str) or not value.strip() or value != value.strip()
        for value in perturbation_payload
    ):
        raise ValueError("formal perturbations must be non-empty canonical strings")
    expected = list(perturbation_payload)
    if len(expected) != len(set(expected)):
        raise ValueError("expected perturbation list contains duplicates")
    expected_set = set(expected)
    if len(expected) != 380:
        raise ValueError(f"formal benchmark requires 380 perturbations, found {len(expected)}")

    truth_schema = ad.read_h5ad(truth_path, backed="r")
    try:
        if truth_schema.shape != (31_854, args.expected_genes):
            raise ValueError(f"unexpected benchmark truth shape: {truth_schema.shape}")
        if not truth_schema.var_names.is_unique:
            raise ValueError("benchmark truth contains duplicate genes")
        truth_controls = boolean_obs_mask(truth_schema, "is_control")
        truth_marked = boolean_obs_mask(truth_schema, "benchmark_control")
        truth_labels_for_schema = truth_schema.obs["perturbation"].astype(str).to_numpy()
        if int(truth_controls.sum()) != 4_976 or int(truth_marked.sum()) != args.expected_controls:
            raise ValueError("benchmark truth has unexpected control counts")
        if not truth_controls[truth_marked].all():
            raise ValueError("benchmark truth marks non-control cells as fixed controls")
        target_mask_for_schema = np.isin(truth_labels_for_schema, expected) & ~truth_controls
        if int(target_mask_for_schema.sum()) != 26_878:
            raise ValueError("benchmark truth has an unexpected held-out target-cell count")
        if set(truth_labels_for_schema[target_mask_for_schema]) != expected_set:
            raise ValueError("benchmark truth perturbation coverage differs from the expected list")
    finally:
        truth_schema.file.close()

    checkpoint = run_dir / "checkpoint.pt"
    latest = run_dir / "latest.pt"
    history_path = run_dir / "history.json"
    manifest_path = run_dir / "run_manifest.json"
    for required in (checkpoint, latest, history_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    checkpoint_sha256 = _sha256(checkpoint)

    manifest = _load_json(manifest_path)
    if manifest.get("model_size") != "mini":
        raise ValueError("run manifest is not an X-Cell Mini run")
    if manifest.get("shape") != [616_535, training_genes]:
        raise ValueError(f"run manifest has an unexpected training shape: {manifest.get('shape')}")
    if manifest.get("epochs") != args.expected_epochs or manifest.get("normalization") != "none":
        raise ValueError(
            "run manifest must record the requested epochs on pre-normalized expression"
        )
    if int(manifest["training_config"]["max_steps"]) != args.expected_steps:
        raise ValueError("run manifest has an unexpected number of optimizer steps")
    if int(manifest["training_context_length"]) != expected_training_context:
        raise ValueError("run manifest has an unexpected training context length")
    if (
        manifest.get("backed") is not expected_backed
        or manifest.get("requested_training_context_length") != expected_requested_context
    ):
        raise ValueError("Replogle run has an unexpected backed/context request profile")
    if int(manifest["world_size"]) != args.expected_world_size:
        raise ValueError("run manifest has an unexpected DDP world size")
    _check_mapping(manifest["model_config"], EXPECTED_MODEL_CONFIG, "model config")
    expected_training_config = dict(EXPECTED_TRAINING_CONFIG)
    expected_training_config["learning_rate"] = args.expected_learning_rate
    _check_mapping(manifest["training_config"], expected_training_config, "training config")
    if bool(manifest["training_config"].get("gradient_checkpointing")) != bool(
        args.expected_gradient_checkpointing
    ):
        raise ValueError("run manifest has an unexpected gradient-checkpointing setting")
    if int(manifest.get("effective_sets_per_optimizer_step", -1)) != 64:
        raise ValueError("effective optimizer batch must contain 64 cell sets")
    if int(manifest.get("effective_cells_per_optimizer_step", -1)) != 4_096:
        raise ValueError("effective optimizer batch must contain 4,096 cells")
    scheduled_microbatches = args.expected_epochs * 5_762
    consumed_microbatches = args.expected_steps * 32
    expected_epoch_schedule = {
        "microbatches_per_rank_per_epoch": 5_762,
        "requested_epochs": args.expected_epochs,
        "scheduled_microbatches_per_rank": scheduled_microbatches,
        "consumed_microbatches_per_rank": consumed_microbatches,
        "optimizer_rounding_extra_microbatches_per_rank": (
            consumed_microbatches - scheduled_microbatches
        ),
        "effective_epochs_after_full_accumulation_rounding": (
            consumed_microbatches / 5_762
        ),
    }
    if manifest.get("epoch_schedule") != expected_epoch_schedule:
        raise ValueError("run manifest has an unexpected full-accumulation epoch schedule")
    explicit_mini = args.expected_loss_profile == "paper_mini_explicit_sentence"
    appendix_l1_zero = args.expected_loss_profile == "appendix_a2_7_l1_zero"
    expected_loss_weights = {
        "mmd": 1.0,
        "fold_change_ccc": 0.0 if explicit_mini else 1.0,
        "per_gene_ccc": 0.0 if explicit_mini else 1.0,
        "delta_norm": 0.0 if explicit_mini else 1.0,
        "delta_mse": 0.0 if explicit_mini else 1.0,
        "contrastive": 1.0,
        "prior_reconstruction": 1.0,
        "l1_output": 0.0 if appendix_l1_zero else 1.0,
    }
    _check_mapping(manifest["loss_config"], expected_loss_weights, "loss config")
    loss_profile_record = None
    if args.expected_loss_profile == "paper_mini_explicit_sentence":
        loss_profile_path = run_dir / "loss_profile_provenance.json"
        if not loss_profile_path.is_file():
            raise FileNotFoundError(loss_profile_path)
        loss_profile_record = _load_json(loss_profile_path)
        if (
            loss_profile_record.get("status") != "complete"
            or loss_profile_record.get("profile") != args.expected_loss_profile
            or loss_profile_record.get("loss_config") != expected_loss_weights
        ):
            raise ValueError("paper Mini loss-profile provenance is malformed")
        profile_manifest = loss_profile_record.get("manifest", {})
        if (
            Path(str(profile_manifest.get("path", ""))).resolve() != manifest_path
            or profile_manifest.get("sha256") != _sha256(manifest_path)
        ):
            raise ValueError("paper Mini loss-profile sidecar does not bind this manifest")
        profile_runner = loss_profile_record.get("runner", {})
        profile_runner_path = Path(str(profile_runner.get("path", ""))).resolve()
        if (
            not profile_runner_path.is_file()
            or profile_runner.get("sha256") != _sha256(profile_runner_path)
        ):
            raise ValueError("paper Mini loss-profile runner is missing or changed")
    expected_mmd = (
        "GeomLoss 0.2.6 tensorized energy: E_xy - 0.5*E_xx - 0.5*E_yy "
        "with squared-distance clamp_min(1e-8)"
    )
    if manifest.get("loss_semantics", {}).get("mmd") != expected_mmd:
        raise ValueError("run manifest does not prove the GeomLoss-compatible energy convention")
    expected_contrastive = (
        "temperature-0.1 InfoNCE over per-cell predicted CLS embeddings; one "
        "ground-truth per-cell CLS center is averaged per perturbation from every "
        "gradient-accumulation microbatch and DDP rank in the optimizer step; "
        "all other perturbation centers are negatives"
    )
    if manifest.get("loss_semantics", {}).get("contrastive") != expected_contrastive:
        raise ValueError("run manifest does not prove the label-aware DDP InfoNCE convention")
    expected_prior_reconstruction = (
        "sum of per-source mean-squared errors, globally normalized over available samples "
        "across DDP ranks"
    )
    if manifest.get("loss_semantics", {}).get("prior_reconstruction") != (
        expected_prior_reconstruction
    ):
        raise ValueError("run manifest does not prove globally normalized prior reconstruction")
    expected_lr_semantics = (
        "one-indexed 10%-warmup then zero-based cosine update samples; every "
        "scheduled optimizer update uses a non-zero LR and the post-horizon "
        "scheduler state is zero"
    )
    if manifest.get("loss_semantics", {}).get(
        "learning_rate_schedule"
    ) != expected_lr_semantics:
        raise ValueError("run manifest does not prove the non-zero-update LR convention")
    if manifest.get("source_sha256") != _current_source_fingerprints():
        raise ValueError("current package sources differ from the exact sources used for training")
    resume_events = manifest.get("resume_events")
    if not isinstance(resume_events, list):
        raise ValueError("run manifest must contain a resume_events list")
    verified_resume_events = []
    previous_resume_step = -1
    for event_index, event in enumerate(resume_events):
        if not isinstance(event, dict):
            raise ValueError(f"resume event {event_index} must be a mapping")
        completed_steps = event.get("completed_steps")
        if (
            isinstance(completed_steps, bool)
            or not isinstance(completed_steps, int)
            or not 0 <= completed_steps < args.expected_steps
        ):
            raise ValueError(f"resume event {event_index} has invalid completed_steps")
        if completed_steps < previous_resume_step:
            raise ValueError("resume events are not ordered by completed step")
        previous_resume_step = completed_steps
        source_path = Path(str(event.get("source_path", ""))).resolve()
        retained_path = _verify_artifact(
            event.get("checkpoint", {}), f"resume event {event_index} checkpoint"
        )
        if retained_path.parent != run_dir / "resume_inputs":
            raise ValueError("retained resume checkpoint is outside run_dir/resume_inputs")
        verified_resume_events.append(
            {
                "source_path": str(source_path),
                "completed_steps": completed_steps,
                "checkpoint": event["checkpoint"],
            }
        )
    input_artifacts = manifest.get("input_artifacts", {})
    required_input_artifacts = _required_training_input_artifacts(
        args.expected_initialization
    )
    if set(input_artifacts) != required_input_artifacts:
        raise ValueError(
            f"training input artifact inventory mismatch: {sorted(set(input_artifacts))}"
        )
    verified_inputs = {
        name: _verify_artifact(record, name) for name, record in input_artifacts.items()
    }
    if verified_inputs["training_adata"] != Path(manifest["adata"]).resolve():
        raise ValueError("training AnnData artifact does not match manifest['adata']")
    if verified_inputs["vocabulary"] != Path(manifest["vocabulary"]).resolve():
        raise ValueError("vocabulary artifact does not match manifest['vocabulary']")
    expected_inference_inputs = {
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": int(checkpoint.stat().st_size),
            "sha256": checkpoint_sha256,
        },
        "source_adata": {
            "path": str(truth_path),
            "bytes": int(truth_path.stat().st_size),
            "sha256": _sha256(truth_path),
        },
        "perturbations_file": {
            "path": str(perturbations_path),
            "bytes": int(perturbations_path.stat().st_size),
            "sha256": _sha256(perturbations_path),
        },
        "prior_artifacts": {
            source: input_artifacts[f"prior_{source}"]
            for source in ("esm", "string", "genept", "depmap", "cell_painting")
        },
        "source_sha256": manifest["source_sha256"],
    }
    expected_inference_inputs_json = _canonical_json(expected_inference_inputs)
    if not dataset_audit_path.is_file():
        raise FileNotFoundError(dataset_audit_path)
    dataset_audit = _load_json(dataset_audit_path)
    if target_complete_profile:
        (
            protocol_audits,
            prepared_source_audit,
            prior_reconstruction_vocabulary,
        ) = _verify_target_complete_profile(
            dataset_audit_path,
            normalization_audit_path,
            feature_space_audit_path,
            training_validation_audit_path,
            training_adata=verified_inputs["training_adata"],
            truth=truth_path,
            perturbations=perturbations_path,
            vocabulary=verified_inputs["vocabulary"],
            input_artifacts=input_artifacts,
            manifest_source_sha256=manifest["source_sha256"],
            expected_epochs=args.expected_epochs,
            expected_steps=args.expected_steps,
            expected_world_size=args.expected_world_size,
            expected_gradient_checkpointing=args.expected_gradient_checkpointing,
        )
    elif raw_union_profile:
        (
            protocol_audits,
            prepared_source_audit,
            prior_reconstruction_vocabulary,
        ) = _verify_raw_union_dynamic4k_profile(
            dataset_audit_path,
            normalization_audit_path,
            feature_space_audit_path,
            training_validation_audit_path,
            training_adata=verified_inputs["training_adata"],
            truth=truth_path,
            perturbations=perturbations_path,
            vocabulary=verified_inputs["vocabulary"],
            input_artifacts=input_artifacts,
            manifest_source_sha256=manifest["source_sha256"],
            expected_epochs=args.expected_epochs,
            expected_steps=args.expected_steps,
            expected_world_size=args.expected_world_size,
            expected_gradient_checkpointing=args.expected_gradient_checkpointing,
        )
    else:
        prior_reconstruction_vocabulary = verified_inputs["vocabulary"]
        expected_dataset_audit = {
            "shape": [616_535, args.expected_genes],
            "vocabulary_size": 19_408,
            "measured_gene_context": args.expected_genes,
            "strata": 5_379,
            "perturbations": 2_015,
            "contexts": ["hepg2", "jurkat", "k562", "rpe1"],
            "batches": 1,
            "batch_collapsed": False,
            "target_cells": 577_370,
            "sets_per_epoch": 11_524,
            "set_size": 64,
            "padded_target_cells_per_epoch": 160_166,
            "world_size": args.expected_world_size,
            "batch_size_per_rank": 1,
            "gradient_accumulation_steps": 32,
            "samples_per_rank_per_epoch": 5_762,
            "microbatches_per_rank_per_epoch": 5_762,
            "epochs": args.expected_epochs,
            "optimizer_steps": args.expected_steps,
            "scheduled_microbatches_per_rank": scheduled_microbatches,
            "consumed_microbatches_per_rank": consumed_microbatches,
            "optimizer_rounding_extra_microbatches_per_rank": (
                consumed_microbatches - scheduled_microbatches
            ),
            "effective_sets_per_optimizer_step": 64,
            "effective_target_cells_per_optimizer_step": 4_096,
        }
        _check_mapping(dataset_audit, expected_dataset_audit, "training dataset audit")
    if not math.isclose(
        float(dataset_audit.get("effective_epochs_after_full_accumulation_rounding", -1)),
        consumed_microbatches / 5_762,
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise ValueError("training dataset audit has an unexpected epoch-rounding value")
    if not extended_public_profile:
        if Path(dataset_audit.get("adata", "")).resolve() != verified_inputs["training_adata"]:
            raise ValueError("training dataset audit points to a different AnnData")
        if dataset_audit.get("adata_sha256") != input_artifacts["training_adata"]["sha256"]:
            raise ValueError("training dataset audit AnnData hash differs from the run manifest")
        if Path(dataset_audit.get("vocabulary", "")).resolve() != verified_inputs["vocabulary"]:
            raise ValueError("training dataset audit points to a different vocabulary")
        if dataset_audit.get("vocabulary_sha256") != input_artifacts["vocabulary"]["sha256"]:
            raise ValueError("training dataset audit vocabulary hash differs from the run manifest")
        dataset_audit_script = Path(__file__).resolve().parent / "audit_training_dataset.py"
        if dataset_audit.get("audit_script_sha256") != _sha256(dataset_audit_script):
            raise ValueError("training dataset audit script differs from the recorded version")
        if dataset_audit.get("dataset_source_sha256") != manifest["source_sha256"].get("data.py"):
            raise ValueError("training dataset audit used a different dataset implementation")
        prepared_source_audit = dataset_audit.get("prepared_source_audit")
    if not isinstance(prepared_source_audit, dict):
        raise ValueError("training dataset audit lacks public-source reconstruction proof")
    verified_prepared_sources = {
        name: _verify_artifact(prepared_source_audit.get(name), f"prepared source {name}")
        for name in (
            "source",
            "split_notebook",
            "preparation_metadata",
            "truth",
            "perturbations",
            "gene_id_map",
            "esm_prior",
            "gtf",
            "esm_embeddings",
            "esm_uniprots",
            "esm_gene_map",
        )
    }
    if verified_prepared_sources["truth"] != truth_path:
        raise ValueError("prepared-source audit points to a different benchmark truth file")
    if verified_prepared_sources["perturbations"] != perturbations_path:
        raise ValueError("prepared-source audit points to a different perturbation list")
    # The Ensembl map is an input only to direct scGPT warm-start. A downstream
    # run initialized from an X-Cell checkpoint already uses the canonical model
    # vocabulary and must not pretend that it consumed this map again.
    if (
        args.expected_initialization == "scgpt"
        and verified_prepared_sources["gene_id_map"] != verified_inputs["gene_id_map"]
    ):
        raise ValueError("prepared-source audit points to a different Ensembl-to-symbol map")
    if verified_prepared_sources["esm_prior"] != verified_inputs["prior_esm"]:
        raise ValueError("prepared-source audit points to a different ESM prior")
    preparation_script = Path(__file__).resolve().parent / "prepare_replogle_official.py"
    if prepared_source_audit.get("audited_against_preparation_script_sha256") != _sha256(
        preparation_script
    ):
        raise ValueError("prepared-source audit references a different preparation implementation")
    expected_prepared_source_facts = {
        "split_rule": (
            "STATE few-shot notebook test-list intersection with observed HepG2 targets"
        ),
        "test_perturbations": 380,
        "test_target_cells": 26_878,
        "fixed_controls": args.expected_controls,
        "training_obs_exact_source_split": True,
        "truth_obs_exact_source_split": True,
        "gene_order_exact_source_x_hvg": True,
        "expression_exact_source_x_hvg": True,
        "vocabulary_exact_gtf_esm_reconstruction": True,
        "gene_id_map_exact_reconstruction": True,
        "esm_prior_exact_raw_asset_reconstruction": True,
        "gtf_protein_coding_symbols": 19_933,
        "esm_raw_shape": [20_588, 5_120],
        "esm_unique_uniprot_ids": 19_018,
        "esm_duplicate_uniprot_ids": 1_461,
        "esm_duplicate_uniprot_rows": 1_570,
        "esm_duplicate_uniprot_embeddings_exact": True,
        "esm_prior_genes": 1_949,
    }
    _check_mapping(
        prepared_source_audit,
        expected_prepared_source_facts,
        "prepared public-source audit",
    )
    if not extended_public_profile:
        protocol_audits = _verify_protocol_audits(
            normalization_audit_path,
            feature_space_audit_path,
            training_validation_audit_path,
            dataset_audit_path=dataset_audit_path,
            prepared_source_audit=prepared_source_audit,
            training_adata=verified_inputs["training_adata"],
            truth=truth_path,
            vocabulary=verified_inputs["vocabulary"],
        )
    expected_padded_fraction = 160_166 / (11_524 * 64)
    if not math.isclose(
        float(dataset_audit.get("padded_fraction", -1.0)),
        expected_padded_fraction,
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise ValueError("training dataset audit has an unexpected padding fraction")
    if args.expected_initialization == "scgpt":
        if (
            not manifest.get("scgpt_strict_overlap")
            or float(manifest.get("scgpt_min_overlap", 0)) != 0.95
        ):
            raise ValueError("scGPT warm-start must enforce the paper's 95% overlap gate")
        if not manifest.get("scgpt_warm_start") or manifest.get("pretrained_checkpoint") is not None:
            raise ValueError("run manifest does not record an exclusive scGPT warm-start")
        scgpt_stats = manifest.get("scgpt_initialization_stats")
        if not isinstance(scgpt_stats, dict):
            raise ValueError("run manifest lacks measured scGPT initialization statistics")
        embeddings_only = args.expected_scgpt_transfer_scope == "table_a3_embeddings_only"
        if bool(manifest.get("scgpt_embedding_only")) != embeddings_only:
            raise ValueError("run manifest has the wrong scGPT transfer scope")
        expected_scgpt_mapped = 18_872 if extended_public_profile else 18_831
        expected_scgpt_fallback = 19_408 - expected_scgpt_mapped
        expected_scgpt_stats = {
            "mode": "warm_start",
            "n_genes": 19_408,
            "n_mapped": expected_scgpt_mapped,
            "n_via_symbol": expected_scgpt_mapped,
            "n_via_ensembl": 0,
            "n_fallback": expected_scgpt_fallback,
            "transformer_loaded": not embeddings_only,
            "value_encoder_loaded": False,
            "loaded_tensors": 0 if embeddings_only else 194,
            "skipped_tensors": 0 if embeddings_only else 124,
            "load_transformer": not embeddings_only,
            "load_value_encoder": False,
            "paper_transfer_scope": (
                "gene_embedding_only_table_a3_wording"
                if embeddings_only
                else "gene_embedding_norm_and_transformer_appendix_a7"
            ),
            "notes": [],
        }
        _check_mapping(scgpt_stats, expected_scgpt_stats, "scGPT initialization statistics")
        if not math.isclose(
            float(scgpt_stats.get("overlap", -1)),
            expected_scgpt_mapped / 19_408,
            rel_tol=0,
            abs_tol=1e-15,
        ):
            raise ValueError("scGPT initialization overlap differs from the audited vocabulary")
        scgpt_dir = Path(manifest["scgpt_warm_start"]).resolve()
        if verified_inputs["scgpt_vocab"] != scgpt_dir / "vocab.json" or verified_inputs[
            "scgpt_checkpoint"
        ] != scgpt_dir / "best_model.pt":
            raise ValueError("scGPT input artifacts differ from the recorded warm-start directory")
    else:
        expected_pretrained = Path(args.expected_pretrained_checkpoint).resolve()
        if verified_inputs["pretrained_checkpoint"] != expected_pretrained:
            raise ValueError("pretrained checkpoint differs from the requested public-data run")
        if Path(str(manifest.get("pretrained_checkpoint", ""))).resolve() != expected_pretrained:
            raise ValueError("manifest points to a different pretrained checkpoint")
        if manifest.get("scgpt_warm_start") is not None or manifest.get(
            "scgpt_initialization_stats"
        ) is not None:
            raise ValueError("pretrained fine-tuning must not also apply scGPT initialization")
        pretrained_stats = manifest.get("pretrained_initialization_stats")
        expected_pretrained_stats = {
            "path": str(expected_pretrained),
            "source_vocabulary_size": 19_408,
            "destination_vocabulary_size": 19_408,
            "overlap_genes": 19_408,
            "overlap_fraction_of_destination": 1.0,
            "minimum_overlap_required": 0.95,
            "unmapped_genes": 0,
            "unmapped_token": None,
            "unmapped_token_source_index": None,
            "vocabulary_remap_semantics": (
                "appendix_a9_source_lookup_materialized_into_destination_parameter_rows"
            ),
            "vocabulary_identical": True,
            "optimizer_state_loaded": False,
            "scheduler_state_loaded": False,
        }
        if pretrained_stats != expected_pretrained_stats:
            raise ValueError("pretrained initialization statistics differ from exact weight-only transfer")
    prior_files = manifest.get("prior_files", {})
    if set(prior_files) != {"esm", "string", "genept", "depmap", "cell_painting"}:
        raise ValueError("run manifest does not declare exactly the five external priors")
    for source, prior_path in prior_files.items():
        if not prior_path or not Path(prior_path).is_file():
            raise FileNotFoundError(f"missing {source} prior recorded by run manifest: {prior_path}")
        if Path(prior_path).resolve() != verified_inputs[f"prior_{source}"]:
            raise ValueError(f"{source} prior path differs from its hashed input artifact")
    with np.load(verified_inputs["prior_esm"], allow_pickle=False) as esm_payload:
        if set(esm_payload.files) != {"genes", "embeddings"}:
            raise ValueError("ESM prior must contain exactly genes and embeddings arrays")
        esm_genes = esm_payload["genes"].astype(str)
        esm_values = np.asarray(esm_payload["embeddings"], dtype=np.float32)
    if (
        esm_values.shape != (1_949, 5_120)
        or len(esm_genes) != len(set(esm_genes.tolist()))
        or any(not gene.strip() or gene.strip().lower() in {"nan", "none", "null"} for gene in esm_genes)
        or not np.isfinite(esm_values).all()
    ):
        raise ValueError("ESM prior shape, keys, or values differ from the audited formal artifact")

    processed_prior_sources = ("string", "genept", "depmap", "cell_painting")
    processed_prior_parents = {
        verified_inputs[f"prior_{source}"].parent for source in processed_prior_sources
    }
    if len(processed_prior_parents) != 1:
        raise ValueError("processed priors do not share one provenance directory")
    processed_prior_dir = next(iter(processed_prior_parents))
    processed_prior_metadata_path = processed_prior_dir / "metadata.json"
    if not processed_prior_metadata_path.is_file():
        raise FileNotFoundError(processed_prior_metadata_path)
    processed_prior_metadata = _load_json(processed_prior_metadata_path)
    raw_prior_records = processed_prior_metadata.get("sources")
    if not isinstance(raw_prior_records, dict) or not raw_prior_records:
        raise ValueError("processed prior metadata lacks raw source artifacts")
    verified_raw_prior_sources = {
        name: str(_verify_artifact(record, f"raw prior source {name}"))
        for name, record in raw_prior_records.items()
    }
    processed_prior_records = processed_prior_metadata.get("processed")
    if not isinstance(processed_prior_records, dict) or set(processed_prior_records) != set(
        processed_prior_sources
    ):
        raise ValueError("processed prior metadata has an unexpected output inventory")
    expected_prior_dims = {
        "string": 512,
        "genept": 3_072,
        "depmap": 1_150,
        "cell_painting": 259,
    }
    for source in processed_prior_sources:
        record = processed_prior_records[source]
        artifact = input_artifacts[f"prior_{source}"]
        if (
            Path(str(record.get("path", ""))).resolve() != verified_inputs[f"prior_{source}"]
            or int(record.get("bytes", -1)) != artifact["bytes"]
            or record.get("sha256") != artifact["sha256"]
            or record.get("finite") is not True
            or not isinstance(record.get("shape"), list)
            or len(record["shape"]) != 2
            or int(record["shape"][1]) != expected_prior_dims[source]
        ):
            raise ValueError(f"processed {source} prior metadata differs from the training input")
    depmap_ids_path = processed_prior_dir / "depmap_selected_model_ids.json"
    depmap_ids = list(map(str, _load_json(depmap_ids_path)))
    if len(depmap_ids) != 1_150 or len(depmap_ids) != len(set(depmap_ids)):
        raise ValueError("DepMap selected-model inventory must contain 1,150 unique IDs")
    prior_preparation_script = Path(__file__).resolve().parent / "prepare_xcell_priors.py"
    prior_reconstruction_script = Path(__file__).resolve().parent / "audit_xcell_priors.py"
    prior_reconstruction_path = processed_prior_dir / "reconstruction_audit.json"
    if not prior_reconstruction_path.is_file():
        raise FileNotFoundError(prior_reconstruction_path)
    prior_reconstruction = _load_json(prior_reconstruction_path)
    if (
        prior_reconstruction.get("schema_version") != 1
        or prior_reconstruction.get("all_array_payloads_exact") is not True
        or prior_reconstruction.get("depmap_model_selection_exact") is not True
    ):
        raise ValueError("processed-prior reconstruction audit is incomplete")
    if prior_reconstruction.get("audit_script_sha256") != _sha256(
        prior_reconstruction_script
    ):
        raise ValueError("processed-prior reconstruction audit used a different audit script")
    reconstructed_preparation_script = _verify_artifact(
        prior_reconstruction.get("preparation_script"),
        "prior reconstruction preparation script",
    )
    if reconstructed_preparation_script != prior_preparation_script.resolve():
        raise ValueError("processed-prior reconstruction used a different preparation script")
    reconstructed_raw_sources = prior_reconstruction.get("raw_sources")
    if not isinstance(reconstructed_raw_sources, dict) or set(
        reconstructed_raw_sources
    ) != set(verified_raw_prior_sources):
        raise ValueError("processed-prior reconstruction has an unexpected raw-source inventory")
    for name, record in reconstructed_raw_sources.items():
        if str(_verify_artifact(record, f"reconstructed raw prior source {name}")) != (
            verified_raw_prior_sources[name]
        ):
            raise ValueError(f"processed-prior reconstruction used a different {name} source")
    reconstructed_processed = prior_reconstruction.get("processed")
    if not isinstance(reconstructed_processed, dict) or set(
        reconstructed_processed
    ) != set(processed_prior_sources):
        raise ValueError("processed-prior reconstruction has an unexpected output inventory")
    for source, record in reconstructed_processed.items():
        if record.get("exact_reconstruction") is not True:
            raise ValueError(f"processed {source} prior lacks exact-reconstruction proof")
        if set(record.get("npz_member_payloads", {})) != {"embeddings.npy", "genes.npy"}:
            raise ValueError(f"processed {source} prior lacks exact NPZ member hashes")
        reconstructed_path = _verify_artifact(
            record.get("artifact"), f"reconstructed processed {source} prior"
        )
        if reconstructed_path != verified_inputs[f"prior_{source}"]:
            raise ValueError(f"processed-prior reconstruction points to a different {source} input")
    reconstructed_depmap_ids = _verify_artifact(
        prior_reconstruction.get("depmap_selected_model_ids"),
        "reconstructed DepMap model IDs",
    )
    if reconstructed_depmap_ids != depmap_ids_path.resolve():
        raise ValueError("processed-prior reconstruction points to different DepMap model IDs")
    reconstructed_metadata = _verify_artifact(
        prior_reconstruction.get("processed_metadata"),
        "reconstructed processed-prior metadata",
    )
    if reconstructed_metadata != processed_prior_metadata_path.resolve():
        raise ValueError("processed-prior reconstruction points to different metadata")
    reconstructed_benchmark_inputs = prior_reconstruction.get("benchmark_inputs")
    expected_reconstruction_benchmark_inputs = {
        # The four processed prior arrays are vocabulary-independent gene-keyed
        # tables. Their exact-reconstruction audit was generated against the
        # released-HVG base vocabulary; the target-complete profile separately
        # verifies its derived vocabulary and recomputes coverage below.
        "vocabulary": prior_reconstruction_vocabulary,
        "perturbations": Path(args.perturbations).resolve(),
    }
    if not isinstance(reconstructed_benchmark_inputs, dict) or set(
        reconstructed_benchmark_inputs
    ) != set(expected_reconstruction_benchmark_inputs):
        raise ValueError("processed-prior reconstruction has unexpected benchmark inputs")
    for name, expected_path in expected_reconstruction_benchmark_inputs.items():
        if _verify_artifact(
            reconstructed_benchmark_inputs[name], f"prior reconstruction benchmark {name}"
        ) != expected_path:
            raise ValueError(f"processed-prior reconstruction used a different {name}")
    prior_preparation_audit = {
        "metadata": {
            "path": str(processed_prior_metadata_path.resolve()),
            "bytes": int(processed_prior_metadata_path.stat().st_size),
            "sha256": _sha256(processed_prior_metadata_path),
        },
        "raw_sources": verified_raw_prior_sources,
        "depmap_selected_model_ids": {
            "path": str(depmap_ids_path.resolve()),
            "bytes": int(depmap_ids_path.stat().st_size),
            "sha256": _sha256(depmap_ids_path),
            "count": len(depmap_ids),
        },
        "audited_against_preparation_script_sha256": _sha256(prior_preparation_script),
        "exact_reconstruction": {
            "artifact": {
                "path": str(prior_reconstruction_path.resolve()),
                "bytes": int(prior_reconstruction_path.stat().st_size),
                "sha256": _sha256(prior_reconstruction_path),
            },
            "audit_script_sha256": _sha256(prior_reconstruction_script),
            "all_array_payloads_exact": True,
            "depmap_model_selection_exact": True,
        },
    }
    if extended_public_profile:
        target_vocabulary = set(map(str, _load_json(verified_inputs["vocabulary"])))
        if len(target_vocabulary) != 19_408:
            raise ValueError("target-complete vocabulary coverage input is malformed")
        test_perturbations = set(map(str, expected))
        target_coverage: dict[str, dict[str, int]] = {}
        for source in ("esm", *processed_prior_sources):
            with np.load(verified_inputs[f"prior_{source}"], allow_pickle=False) as payload:
                prior_genes = set(payload["genes"].astype(str).tolist())
            target_coverage[source] = {
                "prior_genes": len(prior_genes),
                "vocabulary_covered": len(target_vocabulary & prior_genes),
                "test_perturbations_covered": len(test_perturbations & prior_genes),
            }
        expected_target_coverage = {
            "esm": {
                "prior_genes": 1_949,
                "vocabulary_covered": 1_949,
                "test_perturbations_covered": 364,
            },
            "string": {
                "prior_genes": 19_699,
                "vocabulary_covered": 18_866,
                "test_perturbations_covered": 374,
            },
            "genept": {
                "prior_genes": 103_966,
                "vocabulary_covered": 18_611,
                "test_perturbations_covered": 373,
            },
            "depmap": {
                "prior_genes": 17_916,
                "vocabulary_covered": 17_343,
                "test_perturbations_covered": 364,
            },
            "cell_painting": {
                "prior_genes": 7_975,
                "vocabulary_covered": 7_853,
                "test_perturbations_covered": 156,
            },
        }
        if target_coverage != expected_target_coverage:
            raise ValueError("target-complete prior coverage differs from the audited tables")
        prior_preparation_audit["target_complete_vocabulary_coverage"] = target_coverage

    history = _load_json(history_path)
    if len(history) != args.expected_steps:
        raise ValueError(f"expected {args.expected_steps} history rows, found {len(history)}")
    previous_elapsed = -math.inf
    warmup_steps = int(args.expected_steps * 0.1)
    for index, row in enumerate(history, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"history row {index} must be a mapping")
        step_value = _finite_float(row["step"], f"history[{index}].step")
        if step_value != index:
            raise ValueError(f"history step sequence breaks at row {index}")
        for key, value in row.items():
            _finite_float(value, f"history[{index}].{key}")
        required_history_terms = {
            "mmd",
            "fold_change_ccc",
            "per_gene_ccc",
            "delta_norm",
            "delta_mse",
            "contrastive",
            "prior_reconstruction",
            "l1_output",
            "total",
            "grad_norm",
            "lr",
            "elapsed_seconds",
        }
        if appendix_l1_zero:
            # The trainer omits the zero-weight L1 term from history rows under
            # the Appendix A.2.7 L1=0 reading; the configured weight is pinned
            # to 0.0 above, so an absent key is equivalent to logging 0.0.
            required_history_terms = required_history_terms - {"l1_output"}
        if not required_history_terms.issubset(row):
            raise ValueError(
                f"history row {index} lacks terms: {sorted(required_history_terms - set(row))}"
            )
        term_sum = sum(
            (
                float(row.get(name, 0.0))
                if (appendix_l1_zero and name == "l1_output")
                else float(row[name])
            )
            * float(expected_loss_weights[name])
            for name in expected_loss_weights
        )
        if not math.isclose(term_sum, float(row["total"]), rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"history total does not equal the configured weighted losses at row {index}"
            )
        if float(row["grad_norm"]) < 0:
            raise ValueError(f"history gradient norm is negative at row {index}")
        elapsed = float(row["elapsed_seconds"])
        if elapsed < previous_elapsed:
            raise ValueError(f"history elapsed time decreases at row {index}")
        previous_elapsed = elapsed
        if warmup_steps and index <= warmup_steps:
            lr_multiplier = index / warmup_steps
        elif args.expected_steps == warmup_steps:
            lr_multiplier = 1.0
        else:
            progress = (index - warmup_steps - 1) / (
                args.expected_steps - warmup_steps
            )
            lr_multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
        expected_lr = args.expected_learning_rate * lr_multiplier
        if not math.isclose(float(row["lr"]), expected_lr, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"history learning rate differs from warmup-cosine at row {index}")

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_vocabulary = list(map(str, checkpoint_payload.get("var_names", [])))
    expected_checkpoint_vocabulary = list(map(str, _load_json(verified_inputs["vocabulary"])))
    if len(checkpoint_vocabulary) != int(manifest["vocabulary_size"]):
        raise ValueError("checkpoint vocabulary size differs from run manifest")
    if checkpoint_vocabulary != expected_checkpoint_vocabulary:
        raise ValueError("checkpoint vocabulary/order differs from the hashed vocabulary input")
    if len(checkpoint_vocabulary) != len(set(checkpoint_vocabulary)):
        raise ValueError("checkpoint vocabulary contains duplicates")
    if not checkpoint_payload.get("model_state"):
        raise ValueError("checkpoint model state is empty")
    if "training_state" in checkpoint_payload:
        raise ValueError("inference checkpoint unexpectedly contains optimizer training state")
    if _json_normalized(checkpoint_payload.get("model_config")) != manifest["model_config"]:
        raise ValueError("checkpoint model config differs from run manifest")
    model_state = checkpoint_payload["model_state"]
    if len(model_state) != 318:
        raise ValueError(f"Mini checkpoint has {len(model_state)} rather than 318 model tensors")
    parameter_count = sum(int(tensor.numel()) for tensor in model_state.values())
    if parameter_count != int(manifest["parameters"]):
        raise ValueError("checkpoint tensor count differs from the manifest parameter count")
    if parameter_count != 60_001_618:
        raise ValueError(f"Mini checkpoint has {parameter_count} rather than 60,001,618 parameters")
    if int(manifest.get("trainable_parameters", -1)) != 59_982_210:
        raise ValueError("Mini manifest has an unexpected trainable parameter count")
    for name, tensor in model_state.items():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            raise ValueError(f"checkpoint tensor is non-finite: {name}")

    latest_payload = torch.load(latest, map_location="cpu", weights_only=False)
    if _json_normalized(latest_payload.get("model_config")) != manifest["model_config"]:
        raise ValueError("latest checkpoint model config differs from the run manifest")
    if list(map(str, latest_payload.get("var_names", []))) != checkpoint_vocabulary:
        raise ValueError("latest and inference checkpoint vocabularies differ")
    latest_state = latest_payload.get("training_state", {})
    if int(latest_state.get("completed_steps", -1)) != args.expected_steps:
        raise ValueError("latest checkpoint does not record all optimizer steps")
    if len(latest_state.get("history", [])) != args.expected_steps:
        raise ValueError("latest checkpoint history is incomplete")
    if latest_state.get("history") != history:
        raise ValueError("latest checkpoint history differs from history.json")
    scheduler_state = latest_state.get("scheduler_state", {})
    if scheduler_state.get("last_epoch") != args.expected_steps:
        raise ValueError("latest checkpoint scheduler state is incomplete")
    if (
        scheduler_state.get("_step_count") != args.expected_steps + 1
        or scheduler_state.get("base_lrs") != [args.expected_learning_rate]
        or scheduler_state.get("_last_lr") != [0.0]
    ):
        raise ValueError("latest checkpoint scheduler state differs from the completed cosine schedule")
    if latest_state.get("training_config") != manifest["training_config"]:
        raise ValueError("latest checkpoint training config differs from the run manifest")
    if latest_state.get("source_sha256") != manifest["source_sha256"]:
        raise ValueError("latest checkpoint source hashes differ from the run manifest")
    latest_model_state = latest_payload.get("model_state", {})
    if latest_model_state.keys() != model_state.keys():
        raise ValueError("latest and inference checkpoints have different model tensors")
    for name, tensor in model_state.items():
        if not torch.equal(tensor, latest_model_state[name]):
            raise ValueError(f"latest and inference checkpoint differ at tensor {name}")
    optimizer_entries = len(latest_state.get("optimizer_state", {}).get("state", {}))
    if optimizer_entries != 317:
        raise ValueError(f"latest checkpoint has {optimizer_entries} rather than 317 optimizer entries")
    for parameter_id, state in latest_state.get("optimizer_state", {}).get("state", {}).items():
        for name, value in state.items():
            if torch.is_tensor(value) and torch.is_floating_point(value) and not torch.isfinite(value).all():
                raise ValueError(f"optimizer state is non-finite: parameter={parameter_id}, field={name}")
    optimizer_groups = latest_state.get("optimizer_state", {}).get("param_groups", [])
    if len(optimizer_groups) != 1:
        raise ValueError("latest checkpoint must contain exactly one optimizer parameter group")
    optimizer_group = optimizer_groups[0]
    optimizer_parameter_ids = optimizer_group.get("params")
    if (
        tuple(optimizer_group.get("betas", ())) != (0.9, 0.999)
        or float(optimizer_group.get("weight_decay", -1)) != 0.01
        or float(optimizer_group.get("lr", -1)) != 0.0
        or float(optimizer_group.get("initial_lr", -1)) != args.expected_learning_rate
        or float(optimizer_group.get("eps", -1)) != 1e-8
        or optimizer_group.get("amsgrad") is not False
        or optimizer_group.get("maximize") is not False
        or optimizer_group.get("capturable") is not False
        or optimizer_group.get("differentiable") is not False
    ):
        raise ValueError("latest checkpoint optimizer group differs from the completed schedule")
    if optimizer_parameter_ids != list(range(318)):
        raise ValueError("latest checkpoint optimizer parameter inventory/order is incomplete")
    if set(latest_state["optimizer_state"]["state"]) != set(range(317)):
        raise ValueError("latest checkpoint optimizer state does not cover every trainable tensor")
    del latest_payload, latest_model_state

    prediction_dir = run_dir / "benchmark_predictions"
    prediction_paths = sorted(prediction_dir.glob("[0-9][0-9][0-9][0-9]_*.h5ad"))
    if len(prediction_paths) != len(expected):
        raise ValueError(f"expected {len(expected)} prediction files, found {len(prediction_paths)}")

    reference_var_set: set[str] | None = None
    reference_controls: list[str] | None = None
    prediction_gene_orderings: set[tuple[str, ...]] = set()
    observed_predictions: set[str] = set()
    prediction_min = math.inf
    prediction_max = -math.inf
    prediction_hashes: dict[str, str] = {}
    perturbation_gene_token_presence: dict[str, bool] = {}
    for path in prediction_paths:
        item = ad.read_h5ad(path, backed="r")
        try:
            if item.shape != (args.expected_controls, args.expected_genes):
                raise ValueError(f"unexpected prediction shape for {path.name}: {item.shape}")
            labels = item.obs["perturbation"].astype(str).unique().tolist()
            if len(labels) != 1:
                raise ValueError(f"prediction {path.name} has labels {labels}")
            perturbation = labels[0]
            file_index = int(path.name.split("_", maxsplit=1)[0])
            if file_index >= len(expected) or expected[file_index] != perturbation:
                raise ValueError(f"prediction filename/index does not match its label: {path.name}")
            if perturbation in observed_predictions:
                raise ValueError(f"duplicate prediction perturbation: {perturbation}")
            observed_predictions.add(perturbation)
            controls = item.obs["source_control_obs_name"].astype(str).tolist()
            provenance = dict(item.uns.get("open_x_cell_prediction", {}))
            var_names = item.var_names.astype(str).tolist()
            expected_token_present = perturbation in set(var_names)
            expected_provenance = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "source_adata": str(truth_path),
                "perturbations_file": str(Path(args.perturbations).resolve()),
                "perturbation": perturbation,
                "perturbation_index": file_index,
                "perturbation_gene_token_present": expected_token_present,
                "seed": file_index,
                "n_cells": args.expected_controls,
                "precision": "bfloat16",
                "max_genes": -1,
                "diffusion_return_mode": "cumulative_scattered",
                "inference_inputs_json": expected_inference_inputs_json,
            }
            provenance_mismatches = {
                key: {"expected": value, "actual": provenance.get(key)}
                for key, value in expected_provenance.items()
                if provenance.get(key) != value
            }
            if provenance_mismatches:
                raise ValueError(
                    f"prediction provenance mismatch in {path.name}: {provenance_mismatches}"
                )
            if list(map(str, provenance.get("fixed_control_obs_names", []))) != controls:
                raise ValueError(f"prediction metadata control mismatch in {path.name}")
            if len(var_names) != len(set(var_names)):
                raise ValueError(f"prediction has duplicate genes: {path.name}")
            prediction_gene_orderings.add(tuple(var_names))
            if reference_controls is None:
                reference_controls = controls
                reference_var_set = set(var_names)
            elif controls != reference_controls:
                raise ValueError(f"fixed control set differs in {path.name}")
            elif set(var_names) != reference_var_set:
                # The perturbed gene is deliberately moved to position 1 after
                # CLS when it is measured, so per-file order may differ. AnnData
                # concatenation aligns by var name; the gene *set* must match.
                raise ValueError(f"gene panel differs in {path.name}")
            values = np.asarray(item.X[:], dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"prediction contains non-finite expression: {path.name}")
            prediction_min = min(prediction_min, float(values.min()))
            prediction_max = max(prediction_max, float(values.max()))
            prediction_hashes[perturbation] = _sha256(path)
            perturbation_gene_token_presence[perturbation] = expected_token_present
        finally:
            item.file.close()
    if observed_predictions != expected_set:
        raise ValueError("prediction files do not exactly cover the expected perturbations")
    if prediction_min < 0:
        raise ValueError(f"prediction expression contains a negative value: {prediction_min}")
    if sum(perturbation_gene_token_presence.values()) != 71:
        raise ValueError("benchmark target-token coverage differs from the audited 2,000-HVG panel")

    shard_manifests = []
    for shard_index in range(args.expected_shards):
        shard_path = prediction_dir / f"manifest_shard_{shard_index:03d}.json"
        shard = _load_json(shard_path)
        if int(shard["shard_index"]) != shard_index:
            raise ValueError(f"wrong shard index in {shard_path}")
        if int(shard["shard_count"]) != args.expected_shards:
            raise ValueError(f"wrong shard count in {shard_path}")
        if list(map(str, shard["fixed_control_obs_names"])) != reference_controls:
            raise ValueError(f"control provenance mismatch in {shard_path}")
        if Path(shard["checkpoint"]).resolve() != checkpoint:
            raise ValueError(f"checkpoint provenance mismatch in {shard_path}")
        if shard.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError(f"checkpoint hash mismatch in {shard_path}")
        if Path(shard["adata"]).resolve() != truth_path:
            raise ValueError(f"benchmark AnnData provenance mismatch in {shard_path}")
        if Path(shard["perturbations"]).resolve() != Path(args.perturbations).resolve():
            raise ValueError(f"perturbation-list provenance mismatch in {shard_path}")
        if int(shard["n_cells"]) != args.expected_controls:
            raise ValueError(f"wrong fixed-control count in {shard_path}")
        if int(shard.get("seed", -1)) != 0 or shard.get("device") != "cuda":
            raise ValueError(f"unexpected benchmark seed/device in {shard_path}")
        if shard["precision"] != "bfloat16" or shard["max_genes"] is not None:
            raise ValueError(f"unexpected benchmark numerical configuration in {shard_path}")
        if shard.get("diffusion_return_mode") != "cumulative_scattered":
            raise ValueError(f"unexpected diffusion return mode in {shard_path}")
        if shard.get("inference_inputs") != expected_inference_inputs:
            raise ValueError(f"inference input provenance mismatch in {shard_path}")
        expected_assigned = sum(
            index % args.expected_shards == shard_index for index in range(len(expected))
        )
        if int(shard["assigned_perturbations"]) != expected_assigned:
            raise ValueError(f"wrong perturbation assignment count in {shard_path}")
        expected_assignment = [
            {"index": index, "perturbation": expected[index]}
            for index in range(len(expected))
            if index % args.expected_shards == shard_index
        ]
        if shard.get("assigned") != expected_assignment:
            raise ValueError(f"wrong perturbation assignment list in {shard_path}")
        recorded_outputs = shard.get("written", []) + shard.get("skipped_existing", [])
        expected_output_paths = [
            str(prediction_paths[item["index"]]) for item in expected_assignment
        ]
        if sorted(map(str, recorded_outputs)) != sorted(expected_output_paths):
            raise ValueError(f"shard output inventory mismatch in {shard_path}")
        shard_manifests.append(shard)
    assigned_total = sum(int(item["assigned_perturbations"]) for item in shard_manifests)
    if assigned_total != len(expected):
        raise ValueError(f"shard manifests assign {assigned_total} rather than {len(expected)} perturbations")

    cell_eval_dir = run_dir / "cell_eval"
    cell_eval_metadata = _load_json(cell_eval_dir / "metadata.json")
    expected_pred_rows = args.expected_controls * (len(expected) + 1)
    expected_real_rows = 26878 + args.expected_controls
    if cell_eval_metadata["pred_shape"] != [expected_pred_rows, args.expected_genes]:
        raise ValueError(f"unexpected assembled prediction shape: {cell_eval_metadata['pred_shape']}")
    if cell_eval_metadata["real_shape"] != [expected_real_rows, args.expected_genes]:
        raise ValueError(f"unexpected assembled real shape: {cell_eval_metadata['real_shape']}")
    if Path(cell_eval_metadata["prediction_dir"]).resolve() != prediction_dir.resolve():
        raise ValueError("cell-eval metadata points to a different prediction directory")
    if Path(cell_eval_metadata.get("truth", "")).resolve() != truth_path:
        raise ValueError("cell-eval metadata points to a different truth AnnData")
    if cell_eval_metadata.get("truth_sha256") != _sha256(truth_path):
        raise ValueError("cell-eval metadata truth SHA-256 mismatch")
    if Path(cell_eval_metadata.get("perturbations_file", "")).resolve() != Path(
        args.perturbations
    ).resolve():
        raise ValueError("cell-eval metadata points to a different perturbation list")
    if int(cell_eval_metadata.get("perturbations", -1)) != len(expected):
        raise ValueError("cell-eval metadata has the wrong perturbation count")
    if cell_eval_metadata.get("perturbations_sha256") != _sha256(
        Path(args.perturbations).resolve()
    ):
        raise ValueError("cell-eval metadata perturbation-list SHA-256 mismatch")
    if Path(cell_eval_metadata.get("checkpoint", "")).resolve() != checkpoint:
        raise ValueError("cell-eval metadata points to a different checkpoint")
    if cell_eval_metadata.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("cell-eval metadata checkpoint SHA-256 mismatch")
    if cell_eval_metadata.get("prediction_sha256") != prediction_hashes:
        raise ValueError("cell-eval metadata prediction hashes differ from source files")
    if cell_eval_metadata.get(
        "perturbation_gene_token_presence"
    ) != perturbation_gene_token_presence:
        raise ValueError("cell-eval metadata target-token provenance differs from predictions")
    if (
        int(cell_eval_metadata.get("perturbations_with_gene_token", -1)) != 71
        or int(cell_eval_metadata.get("perturbations_without_gene_token", -1)) != 309
    ):
        raise ValueError("cell-eval metadata has unexpected target-token coverage")
    if cell_eval_metadata.get("diffusion_return_mode") != "cumulative_scattered":
        raise ValueError("cell-eval metadata has the wrong diffusion return interpretation")
    if cell_eval_metadata.get("inference_inputs") != expected_inference_inputs:
        raise ValueError("cell-eval metadata has different inference input provenance")
    if not cell_eval_metadata.get("prediction_control_provenance_verified"):
        raise ValueError("cell-eval assembly did not verify control provenance")
    assembly_script = Path(__file__).resolve().parent / "assemble_cell_eval_inputs.py"
    if cell_eval_metadata.get("assembly_script_sha256") != _sha256(assembly_script):
        raise ValueError("cell-eval assembly script differs from the recorded version")
    if set(cell_eval_metadata["prediction_rows_per_perturbation"]) != expected_set:
        raise ValueError("cell-eval metadata perturbation coverage is incomplete")
    if set(cell_eval_metadata["prediction_rows_per_perturbation"].values()) != {
        args.expected_controls
    }:
        raise ValueError("cell-eval metadata has an unexpected prediction row count")
    if list(map(str, cell_eval_metadata["control_obs_names"])) != reference_controls:
        raise ValueError("cell-eval metadata control provenance differs from benchmark inference")

    assembled_pred = ad.read_h5ad(cell_eval_dir / "pred.h5ad", backed="r")
    assembled_real = ad.read_h5ad(cell_eval_dir / "real.h5ad", backed="r")
    if cell_eval_metadata.get("pred_sha256") != _sha256(cell_eval_dir / "pred.h5ad"):
        raise ValueError("assembled pred.h5ad SHA-256 mismatch")
    if cell_eval_metadata.get("real_sha256") != _sha256(cell_eval_dir / "real.h5ad"):
        raise ValueError("assembled real.h5ad SHA-256 mismatch")
    try:
        if list(assembled_pred.shape) != cell_eval_metadata["pred_shape"]:
            raise ValueError("assembled pred.h5ad shape differs from metadata")
        if list(assembled_real.shape) != cell_eval_metadata["real_shape"]:
            raise ValueError("assembled real.h5ad shape differs from metadata")
        expected_labels = expected_set | {"control"}
        if set(assembled_pred.obs["perturbation"].astype(str)) != expected_labels:
            raise ValueError("assembled prediction labels are incomplete")
        if set(assembled_real.obs["perturbation"].astype(str)) != expected_labels:
            raise ValueError("assembled real labels are incomplete")
        if not assembled_pred.var_names.equals(assembled_real.var_names):
            raise ValueError("assembled prediction and real gene ordering differs")

        truth = ad.read_h5ad(truth_path, backed="r")
        try:
            marked_controls = np.flatnonzero(
                boolean_obs_mask(truth, "benchmark_control")
            )
            truth_control = _read_backed_rows_in_gene_order(
                truth,
                marked_controls,
                assembled_pred.var_names,
                label="fixed-control verification",
            )
            assembled_control = np.asarray(
                assembled_pred.X[: args.expected_controls], dtype=np.float32
            )
            if not np.array_equal(assembled_control, truth_control):
                raise ValueError("assembled prediction controls differ from source truth controls")

            row_start = args.expected_controls
            for perturbation, path in zip(expected, prediction_paths):
                item = ad.read_h5ad(path, backed="r")
                try:
                    source_values = np.asarray(
                        item[:, assembled_pred.var_names].X, dtype=np.float32
                    )
                finally:
                    item.file.close()
                row_end = row_start + args.expected_controls
                assembled_values = np.asarray(
                    assembled_pred.X[row_start:row_end], dtype=np.float32
                )
                if not np.array_equal(assembled_values, source_values):
                    raise ValueError(
                        f"assembled prediction block differs from source file: {perturbation}"
                    )
                row_start = row_end
            if row_start != assembled_pred.n_obs:
                raise ValueError("assembled prediction row accounting is incomplete")

            truth_labels = truth.obs["perturbation"].astype(str).to_numpy()
            truth_keep = boolean_obs_mask(truth, "benchmark_control") | np.isin(
                truth_labels, expected
            )
            expected_real = _read_backed_rows_in_gene_order(
                truth,
                truth_keep,
                assembled_real.var_names,
                label="assembled-real verification",
            )
            assembled_real_values = np.asarray(assembled_real.X[:], dtype=np.float32)
            if not np.array_equal(assembled_real_values, expected_real):
                raise ValueError("assembled real.h5ad differs from the selected source truth rows")
            expected_real_obs_names = truth.obs_names[truth_keep].astype(str).tolist()
            if assembled_real.obs_names.astype(str).tolist() != expected_real_obs_names:
                raise ValueError("assembled real.h5ad observation ordering differs from source truth")
        finally:
            truth.file.close()
    finally:
        assembled_pred.file.close()
        assembled_real.file.close()

    results_dir = cell_eval_dir / "results"
    for required in ("pred_de.csv", "real_de.csv", "results.csv", "agg_results.csv"):
        if not (results_dir / required).is_file():
            raise FileNotFoundError(results_dir / required)
    metric_summary, metric_columns = _read_metric_results(
        results_dir / "results.csv", expected_set
    )

    evaluation_manifest = _load_json(cell_eval_dir / "evaluation_manifest.json")
    if evaluation_manifest.get("status") != "complete":
        raise ValueError("official cell-eval evaluation manifest is not complete")
    if evaluation_manifest.get("cell_eval_version") != args.expected_cell_eval_version:
        raise ValueError("official cell-eval version differs from the required version")
    if evaluation_manifest.get("profile") != "full":
        raise ValueError("official cell-eval profile was not full")
    if (
        evaluation_manifest.get("perturbation_column") != "perturbation"
        or evaluation_manifest.get("control_perturbation") != "control"
        or int(evaluation_manifest.get("num_threads", -1)) != 32
    ):
        raise ValueError("official cell-eval schema/thread configuration differs from the formal run")
    expected_cell_eval_command = [
        "cell-eval",
        "run",
        "-ap",
        str((cell_eval_dir / "pred.h5ad").resolve()),
        "-ar",
        str((cell_eval_dir / "real.h5ad").resolve()),
        "--pert-col",
        "perturbation",
        "--control-pert",
        "control",
        "--profile",
        "full",
        "--num-threads",
        "32",
        "--outdir",
        str(results_dir.resolve()),
    ]
    if evaluation_manifest.get("command") != expected_cell_eval_command:
        raise ValueError("official cell-eval command differs from the formal full-profile command")
    if _finite_float(
        evaluation_manifest.get("elapsed_seconds"), "cell-eval elapsed_seconds"
    ) < 0:
        raise ValueError("cell-eval elapsed time is negative")
    if not isinstance(evaluation_manifest.get("started_at_utc"), str):
        raise ValueError("cell-eval manifest lacks a UTC start timestamp")
    cell_eval_runner = Path(__file__).resolve().parent / "run_cell_eval.py"
    if evaluation_manifest.get("runner_sha256") != _sha256(cell_eval_runner):
        raise ValueError("cell-eval runner differs from the recorded version")
    installed_cell_eval_version = importlib.metadata.version("cell-eval")
    if installed_cell_eval_version != args.expected_cell_eval_version:
        raise ValueError("installed cell-eval version differs from the required version")
    if Path(evaluation_manifest.get("pred", {}).get("path", "")).resolve() != (
        cell_eval_dir / "pred.h5ad"
    ).resolve():
        raise ValueError("cell-eval evaluation manifest points to a different prediction input")
    if Path(evaluation_manifest.get("real", {}).get("path", "")).resolve() != (
        cell_eval_dir / "real.h5ad"
    ).resolve():
        raise ValueError("cell-eval evaluation manifest points to a different truth input")
    if evaluation_manifest["pred"].get("sha256") != cell_eval_metadata["pred_sha256"]:
        raise ValueError("cell-eval prediction input hash differs from assembly metadata")
    if evaluation_manifest["real"].get("sha256") != cell_eval_metadata["real_sha256"]:
        raise ValueError("cell-eval truth input hash differs from assembly metadata")
    for name in ("pred_de.csv", "real_de.csv", "results.csv", "agg_results.csv"):
        record = evaluation_manifest.get("outputs", {}).get(name, {})
        path = results_dir / name
        if Path(record.get("path", "")).resolve() != path.resolve():
            raise ValueError(f"cell-eval manifest path mismatch for {name}")
        if int(record.get("bytes", -1)) != path.stat().st_size or record.get(
            "sha256"
        ) != _sha256(path):
            raise ValueError(f"cell-eval output provenance mismatch for {name}")

    with (results_dir / "agg_results.csv").open(newline="", encoding="utf-8") as handle:
        aggregate_rows = {row["statistic"]: row for row in csv.DictReader(handle)}
    if "mean" not in aggregate_rows or "50%" not in aggregate_rows:
        raise ValueError("cell-eval aggregate output lacks mean or median rows")
    for metric, summary in metric_summary.items():
        # cell-eval 0.6.8's aggregate DE rows use a different missing/NaN
        # reduction from results.csv. Cross-check exact aggregates only for
        # metrics that are finite for every perturbation; retain our explicit
        # finite-value summary for sparse DE metrics.
        if int(summary["finite_count"]) != len(expected):
            continue
        aggregate_mean = float(aggregate_rows["mean"][metric])
        aggregate_median = float(aggregate_rows["50%"][metric])
        if not math.isclose(aggregate_mean, float(summary["mean"]), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"aggregate mean mismatch for {metric}")
        if not math.isclose(
            aggregate_median, float(summary["median"]), rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(f"aggregate median mismatch for {metric}")

    first_window = np.asarray([float(row["total"]) for row in history[:100]], dtype=np.float64)
    last_window = np.asarray([float(row["total"]) for row in history[-100:]], dtype=np.float64)
    initialization_limitation = (
        "Xaira's Pisces pretrained checkpoint and exact production vocabulary/remap "
        "artifacts are not public; this run uses the documented scGPT warm-start and "
        "a deterministic GENCODE/ESM-backed 19,408-gene vocabulary."
        if args.expected_initialization == "scgpt"
        else
        "Xaira's Pisces pretrained checkpoint is not public; this run weight-initializes "
        "from the separately audited public HCT116+HEK293T Orion proxy checkpoint, with a "
        "fresh optimizer/scheduler, and must not be labelled a Pisces reproduction."
    )
    if args.expected_initialization == "scgpt":
        scgpt_scope_limitation = (
            "This branch follows Table A3's literal 'scGPT embeddings' wording and transfers "
            "only the mapped gene-embedding rows; the paper's Methods/Appendix A.7 instead "
            "states that transformer weights are loaded, so this is a sensitivity branch."
            if args.expected_scgpt_transfer_scope == "table_a3_embeddings_only"
            else
            "This branch follows Methods 4.1.3/Appendix A.7 by transferring mapped gene "
            "embeddings, embedding normalization, and compatible transformer weights while "
            "excluding the unmentioned value encoder; Table A3's shorter wording says only "
            "'scGPT embeddings'."
        )
    else:
        scgpt_scope_limitation = (
            "The public-Orion checkpoint was itself initialized under the separately audited "
            "Methods/Appendix A.7 scGPT transfer interpretation."
        )
    clean_room_limitations = [
        initialization_limitation,
        scgpt_scope_limitation,
        (
            "This branch follows Appendix A.2.7/A37 literally: all seven objectives "
            "have unit weight and L1 output regularization is zero. The main Methods "
            "text separately says X-Cell uses L1 weight 1.0, so this is a bounded "
            "paper-internal sensitivity rather than a uniquely identified protocol."
            if args.expected_loss_profile == "appendix_a2_7_l1_zero"
            else (
                "This branch follows Equation (5)'s seven unit-weight objectives together "
                "with the main Methods statement that X-Cell uses L1 weight 1.0. Appendix "
                "A.2.7 instead says L1 is zero, and the model-specific sentence lists only "
                "MMD, contrastive, and prior reconstruction."
                if args.expected_loss_profile == "eq5_all"
                else (
                    "This branch follows the model-specific Methods sentence: X-Cell uses "
                    "MMD, contrastive, and prior reconstruction, plus the separately stated "
                    "X-Cell L1 weight 1.0. Equation (5)/Appendix A.2.7 give conflicting "
                    "seven-term interpretations."
                )
            )
        ),
        "For branches with L1 weight 1, the clean-room implementation uses the "
        "elementwise mean absolute output. The paper writes lambda*||X-hat||_1 "
        "without stating sum-versus-mean reduction, so that numerical scale cannot "
        "be proven identical to the private implementation.",
        (
            "This branch follows Table A3's six epochs, 1e-4 peak learning rate, 0.01 "
            "weight decay, 10% warmup, bf16, and set size 64. Methods 4.1.4 separately "
            "reports 20 epochs and 3e-4 for Replogle fine-tuning, so the two paper "
            "descriptions are retained as distinct branches."
            if args.expected_epochs == 6 and args.expected_learning_rate == 1e-4
            else
            "Methods 4.1.4 directly specifies 20 fine-tuning epochs, peak LR 3e-4, "
            "and set size 64. It does not separately publish AdamW betas, weight decay, "
            "epsilon, warmup, clipping, mixed precision, or optimizer-step size for Table B7; "
            "this run declares its Table A3 Mini extrapolations rather than treating "
            "them as benchmark-unique official values."
        ),
        "Appendix A.4 does not state whether its 'fixed random permutation' is global, "
        "per epoch, per optimizer step, or per training sample. This clean-room run uses "
        "a fresh deterministic permutation per sample/microstep, shared across all 64 cells "
        "in the set, with disjoint fraction and mask RNG streams on each DDP rank.",
        "Appendix A.1.1 includes a Dropout operation in the continuous value encoder "
        "but gives no probability, and the paper does not publish all initialization/"
        "normalization epsilon details. This clean-room Mini declares dropout=0.0 and "
        "its recorded PyTorch initialization; no arbitrary dropout sweep is labelled "
        "an official protocol substitute.",
        "The main text and Table A2 label X-Cell as approximately 55M parameters, while "
        "Table A3 labels Mini as 60M. The detailed Methods/A.1/A.7 module specification "
        "instantiates to 60,001,618 parameters in the clean-room implementation; neither "
        "rounded label proves Xaira's private module inventory.",
        "Methods calls the auxiliary prior-decoder objective MSE, whereas Equation A27 "
        "writes a sum of squared L2 norms. This implementation records a sum of per-source "
        "mean-squared errors over available samples; Xaira's private reduction across source "
        "dimensions and samples is not stated.",
        "The paper names the cell-eval package but does not pin a package version or commit. "
        "Formal scoring uses and records cell-eval 0.6.8 with the full profile; exact metric "
        "identity to the authors' private environment cannot be established from the paper.",
        "Appendix A.2.6 identifies per-cell predicted CLS anchors and ground-truth "
        "perturbation centers but does not specify the target-embedding mask, stop-gradient "
        "choice, or negative-bank scope across accumulation and DDP. The formal run records "
        "a full-reveal stop-gradient target encoder and optimizer-step global center bank.",
        "The paper specifies GeomLoss energy with blur=0.05 and scaling=0.5 but does not "
        "pin a package version/backend. The clean-room implementation is numerically pinned "
        "to GeomLoss 0.2.6's tensorized energy and 1e-8 squared-distance clamp.",
        (
            "Table A3 reports a full approximately 19,000-gene context for X-Cell Mini. "
            "The strongest public proxy here trains on all 6,642 released measured genes "
            "plus 166 explicit zero coordinates needed to address every observed target. "
            "Those zeros restore target-token addressability, not the unreleased expression "
            "values/remap, so the 6,808-gene run is not a full-context reproduction."
            if target_complete_profile
            else (
                "Table A3 reports a full approximately 19,000-gene context for X-Cell Mini, "
                "whereas Appendix A.10.2 permits a fresh G-prime<=4,000 subsample. This run "
                "stores the exact 11,332-coordinate union of the four cited public raw "
                "releases but trains on target-first 4,000-gene contexts; coordinates absent "
                "from a cell line are explicit zeros. It is the bounded A.10.2 reading, not "
                "Table A3 full-context or private preprocessing identity."
                if raw_union_profile
                else
            "Table A3 reports a full approximately 19,000-gene context for X-Cell Mini. "
            "The released STATE Replogle matrix contains 6,642 measured genes, and this "
            "run uses the recorded deterministic 2,000-gene HVG panel while retaining the "
            "19,408-gene model vocabulary; it is therefore not a full-context reproduction."
            )
        ),
        "The five prior tables are exactly reconstructible under this repository's "
        "recorded preprocessing of the cited public releases. Xaira's production "
        "prior mappings and processed tables are not public, so matching the declared "
        "dimensions does not establish identical values or coverage.",
        "Appendix A.10.3 says input prior vectors may optionally be LayerNorm-normalized "
        "before collation, but does not publish whether that option was enabled or its "
        "epsilon/affine convention. The formal branch consumes raw public prior rows and "
        "retains the learned projector's terminal LayerNorm; a separate all-five-prior, "
        "inference-only epsilon=1e-5 sensitivity is reported and is not labelled "
        "normalized-prior fine-tuning.",
        "The public DepMap 24Q4 table has 1,178 models while the paper declares a "
        "1,150-dimensional prior; the recorded coverage-then-ModelID selection is used.",
        "Two-GPU gradient accumulation preserves the 64-set/4,096-cell optimizer batch. "
        "InfoNCE keeps all predicted cells as anchors and constructs one perturbation-center "
        "negative bank from every accumulation microbatch and both DDP ranks.",
        "Table A3 lists one node/eight GPUs for Mini, whereas these clean-room runs use two "
        "GPU ranks with gradient accumulation to preserve the stated 64-set optimizer batch. "
        "Methods 4.1.4 does not publish the Table B7 fine-tuning topology, and matching the "
        "effective batch does not prove identical collective or floating-point reduction order.",
        "Methods 4.1.3 describes random gene ranks and four rounds that reveal 25% of genes, "
        "but does not publish the private RNG, seed, or disambiguate exact permutation "
        "quartiles from independent uniform-threshold draws with variable round counts. "
        "This clean-room branch uses a deterministic CPU torch.randperm ordering split into "
        "four exact quarters; the separately reported five-seed analysis measures only that declared "
        "interpretation and is not an official inference-RNG reproduction.",
        f"Rounding {args.expected_epochs} epochs to {args.expected_steps:,} complete "
        f"32-microbatch optimizer steps consumes "
        f"{consumed_microbatches - scheduled_microbatches} additional microbatches per rank "
        f"({consumed_microbatches / 5_762:.8f} effective epochs) rather than emitting one "
        "undersized final update.",
        "Appendix A.11 reports 3,320 optimizer steps for a separate 20-epoch "
        "83M-3.1B scaling experiment. The exact released split reconstructed here "
        "contains 11,524 sets per epoch, which at 64 sets/update gives 3,602 complete "
        "updates after rounding. Table B7 does not publish its Mini step count, and "
        "Xaira's production batch-stratum mapping is unavailable, so 3,320 is retained "
        "as a distinct scaling-study sensitivity rather than silently substituted.",
        "The 380-gene HepG2 split is deterministically reconstructed from the exact "
        "STATE notebook and processed release cited by the paper and reproduces 380 "
        "genes, 26,878 test cells, and 616,535 test-excluded training cells; the "
        "paper's fixed 32 control identities remain unpublished.",
        "Appendix A.6 states that private training monitored a held-out validation split, "
        "but does not publish its identities, cadence, controls, or checkpoint-selection "
        "rule. The 60 STATE-notebook validation candidates present publicly (4,825 cells) "
        "are included in the published 616,535-cell training count. This run therefore "
        "uses every published training cell, never selects on the 380-gene test set, "
        "reports the final scheduled checkpoint, and does not claim Appendix A.6 identity.",
        "The public STATE table exposes a technical gem_group field, but Xaira's mapping "
        "from that field to the batch strata used for X-Cell set matching is not public. "
        "This run preserves source_gem_group for provenance and treats Replogle-Nadig as "
        "one dataset-level batch; using raw gem_group would create about 216,000 strata "
        "and 95.8% padded target cells.",
        "With the declared dataset-level batch, Appendix A.10.1 replacement padding repeats "
        "160,166 target cells per epoch (21.7164% of set slots). Padding is paper-specified, "
        "but its amount depends on Xaira's unpublished production batch strata and therefore "
        "can alter per-cell training weights.",
        (
            "Appendix A.10.2 target-first forcing is satisfied for every training stratum: "
            "6,642 coordinates are measured and 166 missing observed targets are appended "
            "as explicit zeros (161 training, five held-out-only). Public test scoring still "
            "uses the released 2,000-HVG truth, where 71/380 targets have measured tokens."
            if target_complete_profile
            else (
                "Appendix A.10.2 target-first forcing is verified for all 5,379 training "
                "strata under fresh deterministic 4,000-gene contexts sampled from the "
                "11,332-coordinate raw-release union. Every stratum's target coordinate is "
                "measured in its own cell-line release; other coordinates absent from that "
                "cell line remain explicit-zero public proxies rather than private values."
                if raw_union_profile
                else
            "Appendix A.10.2 requires the perturbed gene as the first gene token, but the "
            "released 2,000-HVG panel contains only 404/2,015 training perturbation genes and "
            "71/380 benchmark perturbation genes. The remaining perturbations condition the "
            "model through the separate perturbation embedding; no unpublished zero-fill or "
            "feature-restoration rule is assumed in this released-panel branch."
            )
        ),
        "The paper labels the small model as both 55M and Mini 60M. This run follows the "
        "Table A2 ReLU/Post-LN/512-FFN self-attention backbone together with the separately "
        "specified Pre-LN RMSNorm 4d-SwiGLU cross-attention sublayers, yielding 60,001,618 "
        "parameters and matching Table A3's 60M total.",
    ]
    if args.expected_gradient_checkpointing:
        clean_room_limitations.append(
            (
                "Activation checkpointing is enabled because even the 4,000-gene/64-cell "
                "dynamic microbatch produced a captured CUDA OOM on a 48 GiB-class RTX "
                "A6000 without it, while the paired checkpointed probe completed. This is "
                "an explicitly labelled A.10.2/local-hardware fallback."
                if raw_union_profile
                else
                "Activation checkpointing is enabled because the literal 6,808-gene Mini "
                "microbatch produced a captured CUDA OOM on a 48 GiB-class RTX A6000, while "
                "the checkpointed probe completed. It recomputes activations without changing "
                "the model or objective, but Table A3 reports checkpointing disabled for Mini; "
                "this run is therefore an explicitly labelled executable fallback."
            )
        )
    audit = {
        "status": "complete",
        "direct_table_b7_comparison": False,
        "direct_table_a3_full_context_identity": False,
        "training_input_profile": args.training_input_profile,
        "run_dir": str(run_dir),
        "runtime_versions": runtime_versions,
        "clean_room_limitations": clean_room_limitations,
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_sha256,
            "latest_path": str(latest),
            "latest_bytes": latest.stat().st_size,
            "latest_sha256": _sha256(latest),
            "vocabulary_size": len(checkpoint_vocabulary),
            "parameter_tensors": len(model_state),
            "parameters": parameter_count,
            "optimizer_entries": optimizer_entries,
        },
        "training": {
            "input_profile": args.training_input_profile,
            "training_genes": training_genes,
            "training_context_genes": expected_training_context,
            "evaluation_genes": args.expected_genes,
            "gradient_checkpointing": bool(args.expected_gradient_checkpointing),
            "scgpt_transfer_scope": (
                args.expected_scgpt_transfer_scope
                if args.expected_initialization == "scgpt"
                else "inherited_from_public_orion_pretrained_checkpoint"
            ),
            "loss_profile": args.expected_loss_profile,
            "loss_config": expected_loss_weights,
            "loss_profile_provenance": loss_profile_record,
            "optimizer_steps": len(history),
            "first_total": float(history[0]["total"]),
            "last_total": float(history[-1]["total"]),
            "first_100_mean_total": float(first_window.mean()),
            "last_100_mean_total": float(last_window.mean()),
            "elapsed_seconds": float(history[-1]["elapsed_seconds"]),
            "all_recorded_values_finite": True,
            "source_sha256": manifest["source_sha256"],
            "input_artifacts": manifest["input_artifacts"],
            "dataset_audit": {
                "path": str(dataset_audit_path),
                "sha256": _sha256(dataset_audit_path),
                "report": dataset_audit,
            },
            "protocol_audits": protocol_audits,
            "prior_preparation_audit": prior_preparation_audit,
            "resume_events": verified_resume_events,
        },
        "benchmark": {
            "perturbations": len(observed_predictions),
            "prediction_files": len(prediction_paths),
            "cells_per_prediction": args.expected_controls,
            "genes": args.expected_genes,
            "unique_gene_orderings": len(prediction_gene_orderings),
            "fixed_control_obs_names": reference_controls,
            "perturbation_gene_token_presence": perturbation_gene_token_presence,
            "perturbations_with_gene_token": 71,
            "perturbations_without_gene_token": 309,
            "prediction_min": prediction_min,
            "prediction_max": prediction_max,
            "shards": args.expected_shards,
            "inference_inputs": expected_inference_inputs,
        },
        "cell_eval": {
            "version": installed_cell_eval_version,
            "profile": "full",
            "pred_shape": cell_eval_metadata["pred_shape"],
            "real_shape": cell_eval_metadata["real_shape"],
            "result_columns": metric_columns,
            "paper_metric_summary": metric_summary,
            "evaluation_manifest_sha256": _sha256(
                cell_eval_dir / "evaluation_manifest.json"
            ),
        },
        "audit_provenance": {
            "validator": str(Path(__file__).resolve()),
            "validator_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
