"""Test scGPT warm-start on a paper-flavored 512-d architecture, 3-arm ladder:

  base       — random init (reference)
  embed      — scGPT gene-embedding init only (npz table)
  warmstart  — full scGPT warm-start: gene embedding + value encoder + transformer
               layers (Ensembl->symbol mapping, 95% overlap check, <pad> fallback)

Everything else identical (data / arch / hyperparams / seed). Reports the cell-eval
metrics including DE Spearman LFC (per-pert) and Centroid Accuracy (across perts).

Full paper-mini (4000 genes) OOMs a single 48G GPU; we use a compact 512-d / 2000-gene
config where scGPT 512-d transfer applies directly.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import anndata as ad

sys.path.insert(0, "src")
import open_x_cell.training as T
from open_x_cell.config import ModelConfig, TrainingConfig
from open_x_cell.inference import predict_adata
from open_x_cell.evaluation import evaluate_prediction_files, centroid_accuracy

ADATA = "data/real/hepg2nadig_mini2k.h5ad"
SCGPT_NPZ = "data/priors/scgpt_gene_embedding.npz"
SCGPT_DIR = "/data5/zongtingwei/vcc/scgpt_models/scGPT_human"
ENS2SYM = "/data5/zongtingwei/vcc/baseline_splits/ens2sym.pt"
G = 2000
STEPS = int(os.environ.get("STEPS", "500"))
DEVICE = os.environ.get("DEVICE", "cuda")

def _cfg(_name):
    return ModelConfig(
        num_genes=G, context_length=G,
        num_layers=6, hidden_dim=512, num_heads=8, ffn_dim=512,
        cross_attention_layers=(2, 5),
        activation="relu", norm="layernorm", pre_norm=False,
        use_qk_norm=False, attention_bias=True, tied_output=False,
    )
T.model_config = _cfg

_adata = ad.read_h5ad(ADATA)
_X = _adata.X.toarray() if hasattr(_adata.X, "toarray") else np.asarray(_adata.X)
_lab = _adata.obs["perturbation"].astype(str).to_numpy()
_ctrl_mean = _X[_adata.obs["is_control"].to_numpy(dtype=bool)].mean(0)
PERTS = [p for p in sorted(set(_lab)) if p != "control"]
TRUE_DELTA = {p: torch.tensor(_X[_lab == p].mean(0) - _ctrl_mean) for p in PERTS}

def run_arm(name, **kw):
    out = Path(f"runs/mini_scgpt/{name}"); out.mkdir(parents=True, exist_ok=True)
    tcfg = TrainingConfig(batch_size=1, set_size=16, max_steps=STEPS,
                          learning_rate=3e-4, grad_clip_norm=1.0, seed=0, device=DEVICE)
    t0 = time.time()
    res = T.train_model(ADATA, out, model_size="x", training_config=tcfg,
                        normalize="none", **kw)
    h = res.history
    first10, last10 = float(np.mean([x["total"] for x in h[:10]])), float(np.mean([x["total"] for x in h[-10:]]))
    print(f"[{name}] {len(h)} steps in {time.time()-t0:.0f}s | loss {first10:.2f}->{last10:.2f}")
    rows, pred_delta = [], {}
    for p in PERTS:
        pp = out / f"pred_{p}.h5ad"
        predict_adata(res.checkpoint, ADATA, p, pp, seed=0, n_cells=64)
        rows.append((p, evaluate_prediction_files(ADATA, pp, out / f"metrics_{p}.json")))
        px = ad.read_h5ad(pp); pxX = px.X.toarray() if hasattr(px.X, "toarray") else np.asarray(px.X)
        pred_delta[p] = torch.tensor(pxX.mean(0) - _ctrl_mean)
    summary = {k: round(float(np.mean([m[k] for _, m in rows])), 4) for k in rows[0][1]}
    summary.update(centroid_accuracy(pred_delta, TRUE_DELTA))
    json.dump({"summary": summary, "loss_first10": first10, "loss_last10": last10,
               "per_pert": {p: m for p, m in rows}}, open(out / "summary.json", "w"), indent=2)
    print(f"[{name}] {summary}")
    return summary

print(f"config: 512-d / {G} genes / 6 layers | steps={STEPS} | device={DEVICE}")
res = {}
res["base"] = run_arm("base")
res["embed"] = run_arm("embed", scgpt_init=SCGPT_NPZ)
res["warmstart"] = run_arm("warmstart", scgpt_warm_start=SCGPT_DIR, gene_id_map=ENS2SYM)

print("\n=== ladder: base -> embed-init -> full warm-start (512-d real HepG2) ===")
keys = list(res["base"].keys())
hdr = f"{'metric':18s} {'base':>9s} {'embed':>9s} {'warmstart':>9s}"
print(hdr)
for k in keys:
    print(f"{k:18s} {res['base'][k]:>9.4f} {res['embed'][k]:>9.4f} {res['warmstart'][k]:>9.4f}")
json.dump(res, open("runs/mini_scgpt/comparison3.json", "w"), indent=2)
print("saved runs/mini_scgpt/comparison3.json")
