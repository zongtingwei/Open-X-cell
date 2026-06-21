"""Test scGPT gene-embedding initialization on a paper-flavored 512-d architecture.

Trains TWO arms that differ ONLY in whether gene embeddings are initialized from
scGPT (everything else identical: data, arch, hyperparams, seed), then runs the
project's predict (n_cells=64) + evaluate on the 12 held perturbations and compares.

Full paper-mini (4000 genes) OOMs a single 48G GPU (O(genes^2) attention), so we use
a compact 512-d / 2000-gene config where scGPT 512-d init still applies directly.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
import open_x_cell.training as T
from open_x_cell.config import ModelConfig, TrainingConfig
from open_x_cell.data import load_adata
from open_x_cell.inference import predict_adata
from open_x_cell.evaluation import evaluate_prediction_files

ADATA = "data/real/hepg2nadig_mini2k.h5ad"
SCGPT = "data/priors/scgpt_gene_embedding.npz"
G = 2000
STEPS = int(os.environ.get("STEPS", "600"))
DEVICE = os.environ.get("DEVICE", "cuda")

def _cfg(_name):
    # mini-flavored: ReLU / LayerNorm / Post-LN, hidden 512 (== scGPT dim)
    return ModelConfig(
        num_genes=G, context_length=G,
        num_layers=6, hidden_dim=512, num_heads=8, ffn_dim=512,
        cross_attention_layers=(2, 5),
        activation="relu", norm="layernorm", pre_norm=False,
        use_qk_norm=False, attention_bias=True, tied_output=False,
    )
T.model_config = _cfg

def run_arm(name, scgpt):
    out = Path(f"runs/mini_scgpt/{name}"); out.mkdir(parents=True, exist_ok=True)
    tcfg = TrainingConfig(batch_size=1, set_size=16, max_steps=STEPS,
                          learning_rate=3e-4, grad_clip_norm=1.0, seed=0, device=DEVICE)
    t0 = time.time()
    res = T.train_model(ADATA, out, model_size="x", training_config=tcfg,
                        normalize="none", scgpt_init=(SCGPT if scgpt else None))
    h = res.history
    first10 = float(np.mean([x["total"] for x in h[:10]]))
    last10 = float(np.mean([x["total"] for x in h[-10:]]))
    print(f"[{name}] trained {len(h)} steps in {time.time()-t0:.0f}s | "
          f"loss {first10:.3f}->{last10:.3f}")
    adata = load_adata(ADATA)
    perts = [p for p in sorted(adata.obs["perturbation"].astype(str).unique()) if p != "control"]
    rows = []
    for p in perts:
        pp = out / f"pred_{p}.h5ad"; mp = out / f"metrics_{p}.json"
        predict_adata(res.checkpoint, ADATA, p, pp, seed=0, n_cells=64)
        rows.append((p, evaluate_prediction_files(ADATA, pp, mp)))
    summary = {k: round(float(np.mean([m[k] for _, m in rows])), 4) for k in rows[0][1]}
    json.dump({"summary": summary, "loss_first10": first10, "loss_last10": last10,
               "per_pert": {p: m for p, m in rows}}, open(out / "summary.json", "w"), indent=2)
    print(f"[{name}] MEAN:", summary)
    return summary

print(f"config: 512-d / {G} genes / 6 layers | steps={STEPS} | device={DEVICE}")
base = run_arm("base", scgpt=False)
scg = run_arm("scgpt", scgpt=True)
print("\n=== scGPT init vs base (512-d, real HepG2) ===")
for k in base:
    d = scg[k] - base[k]
    print(f"{k:18s} base={base[k]:.4f}  scgpt={scg[k]:.4f}  delta={d:+.4f}")
json.dump({"base": base, "scgpt": scg}, open("runs/mini_scgpt/comparison.json", "w"), indent=2)
print("saved runs/mini_scgpt/comparison.json")
