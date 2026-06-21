"""Real-data validation through the LIBRARY training entrypoint.

Same data + same 400-gene X-Cell architecture + same hyperparameters as
scripts/validate_real.py, but training is driven by the project's own
training.train_model (compute_batch_loss + warmup-cosine scheduler +
PriorStore) instead of an inline loop. This isolates the effect of the
"align training pipeline with x-cell paper" commit.

Differences vs the inline loop (validate_real.py):
  - target_cell_embedding comes from a SECOND full-reveal forward on the
    target (a real contrastive target), not out.cell_embedding.detach().
  - loss receives prior_missing / prior_source_names.
  - warmup-cosine LR schedule instead of constant LR.

Normalize is kept 'none' so the train space matches predict_adata (which
reads raw X), exactly as validate_real does.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
import open_x_cell.training as T
from open_x_cell.config import ModelConfig, TrainingConfig
from open_x_cell.data import load_adata
from open_x_cell.inference import predict_adata
from open_x_cell.evaluation import evaluate_prediction_files

ADATA = "data/real/hepg2nadig_subset.h5ad"
OUT = Path("runs/real_hepg2_libpipeline"); OUT.mkdir(parents=True, exist_ok=True)

# 400-gene X-Cell arch identical to validate_real.py, injected into train_model
def _custom_model_config(_name: str) -> ModelConfig:
    return ModelConfig(
        num_genes=400, context_length=400,
        num_layers=4, hidden_dim=128, num_heads=4, ffn_dim=256,
        cross_attention_layers=(1, 3),
        activation="swiglu", norm="rmsnorm", pre_norm=True,
        use_qk_norm=True, attention_bias=False, tied_output=True,
    )
T.model_config = _custom_model_config  # train_model resolves arch via this

tcfg = TrainingConfig(batch_size=4, set_size=16, max_steps=300,
                      learning_rate=3e-4, grad_clip_norm=1.0, seed=0, device="cpu")

result = T.train_model(
    ADATA, OUT,
    model_size="custom-400",          # ignored by the patched resolver
    training_config=tcfg,
    normalize="none",                 # match predict_adata (raw X)
    prior_files=None,
)
ckpt = result.checkpoint
hist = result.history
first10 = float(np.mean([h["total"] for h in hist[:10]]))
last10 = float(np.mean([h["total"] for h in hist[-10:]]))
print("library train_model done | steps", len(hist),
      "| loss first10 %.4f last10 %.4f" % (first10, last10))

# --- project predict + evaluate, per perturbation (same as validate_real) ---
adata = load_adata(ADATA)
perts = [p for p in sorted(adata.obs["perturbation"].astype(str).unique()) if p != "control"]
rows = []
for p in perts:
    pred_path = OUT / f"pred_{p}.h5ad"; met_path = OUT / f"metrics_{p}.json"
    predict_adata(ckpt, ADATA, p, pred_path, seed=0)
    m = evaluate_prediction_files(ADATA, pred_path, met_path)
    rows.append((p, m))

def avg(k): return float(np.mean([m[k] for _, m in rows]))
summary = {k: round(avg(k), 4) for k in rows[0][1]}
print("\n=== per-perturbation eval (library pipeline) ===")
for p, m in rows:
    print(f"{p}: pearson_delta={m['pearson_delta']:.3f} dir={m['direction_match']:.2f} "
          f"overlap@20={m['overlap_at_n']:.2f} dnorm_ratio={m['delta_norm_ratio']:.2f} mae={m['mae']:.3f}")
print("\nMEAN over %d perturbations:" % len(rows), summary)

n_params = sum(p.numel() for p in T.load_checkpoint(ckpt)[0].parameters())
json.dump({"summary": summary, "loss_first10": first10, "loss_last10": last10,
           "n_params_M": n_params / 1e6, "per_pert": {p: m for p, m in rows}},
          open(OUT / "summary.json", "w"), indent=2)
print("saved", OUT / "summary.json")
