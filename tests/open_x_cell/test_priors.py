import torch
import pandas as pd

from open_x_cell.config import tiny_config
from open_x_cell.priors import PRIOR_SOURCES, PriorBatch, PriorStore, make_zero_priors


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


def test_prior_store_loads_csv_embeddings_and_marks_missing_perturbations(tmp_path):
    cfg = tiny_config()
    path = tmp_path / "esm.csv"
    pd.DataFrame(
        {
            "gene": ["GENE_1"],
            **{f"f{idx}": [float(idx)] for idx in range(cfg.prior_dims["esm"])},
        }
    ).to_csv(path, index=False)

    store = PriorStore.from_files({"esm": path}, cfg)
    batch = store.batch_for_perturbations(("GENE_1", "GENE_2"), cfg)

    assert batch.values["esm"].shape == (2, cfg.prior_dims["esm"])
    assert torch.allclose(batch.values["esm"][0, :4], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.all(batch.values["esm"][1] == 0)
    assert batch.missing.shape == (2, len(PRIOR_SOURCES))
    assert batch.missing[0, PRIOR_SOURCES.index("esm")].item() is False
    assert batch.missing[1, PRIOR_SOURCES.index("esm")].item() is True
