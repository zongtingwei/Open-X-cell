"""End-to-end real-data validation of Open-X-Cell on HepG2 Nadig CRISPRi.
Trains a CPU-sized, ultra-flavored X-Cell config (SwiGLU/RMSNorm/Pre-LN/
QK-Norm/tied-output -- the paper's production architecture), then runs the
project's own predict + evaluate code paths on held perturbations."""
import json, sys, numpy as np, torch
from pathlib import Path
from dataclasses import asdict
from torch.utils.data import DataLoader
import anndata as ad

sys.path.insert(0, "src")
from open_x_cell.config import ModelConfig, TrainingConfig
from open_x_cell.data import OpenXCellDataset, collate_samples, load_adata
from open_x_cell.diffusion import make_reveal_mask, mix_control_target
from open_x_cell.losses import OpenXCellLoss
from open_x_cell.model import OpenXCellModel
from open_x_cell.priors import make_zero_priors
from open_x_cell.training import save_checkpoint
from open_x_cell.inference import predict_adata
from open_x_cell.evaluation import evaluate_prediction_files

ADATA = "data/real/hepg2nadig_subset.h5ad"
OUT = Path("runs/real_hepg2"); OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cpu"
adata = load_adata(ADATA)
G = adata.n_vars

cfg = ModelConfig(
    num_genes=G, context_length=G,
    num_layers=4, hidden_dim=128, num_heads=4, ffn_dim=256,
    cross_attention_layers=(1, 3),
    activation="swiglu", norm="rmsnorm", pre_norm=True,
    use_qk_norm=True, attention_bias=False, tied_output=True,
)
tcfg = TrainingConfig(batch_size=4, set_size=16, max_steps=300,
                      learning_rate=3e-4, grad_clip_norm=1.0, seed=0, device=DEVICE)
torch.manual_seed(tcfg.seed)

ds = OpenXCellDataset(adata, set_size=tcfg.set_size,
                      context_length=cfg.context_length, seed=tcfg.seed)
loader = DataLoader(ds, batch_size=tcfg.batch_size, shuffle=True,
                    collate_fn=collate_samples)
model = OpenXCellModel(cfg).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"model params: {n_params/1e6:.2f}M | genes={G} | perts={len(ds)}")
opt = torch.optim.AdamW(model.parameters(), lr=tcfg.learning_rate)
loss_fn = OpenXCellLoss()
fracs = cfg.diffusion_fractions

losses, it = [], iter(loader)
for step in range(tcfg.max_steps):
    try: batch = next(it)
    except StopIteration: it = iter(loader); batch = next(it)
    xc = batch.x_control.to(DEVICE); xt = batch.x_target.to(DEVICE)
    frac = fracs[step % len(fracs)]
    reveal = make_reveal_mask(xc.shape[0], xc.shape[1], xc.shape[2], frac,
                              seed=tcfg.seed+step, device=torch.device(DEVICE))
    xin = mix_control_target(xc, xt, reveal)
    priors = make_zero_priors(xc.shape[0], cfg, device=DEVICE)
    out = model(xin, reveal, batch.gene_ids.to(DEVICE), batch.perturbation_ids, priors)
    lo = loss_fn(out.expression, xt, xc, out.cell_embedding,
                 target_cell_embedding=out.cell_embedding.detach(),
                 prior_reconstructions=out.prior_reconstructions,
                 prior_values=priors.values)
    if not torch.isfinite(lo.total): raise RuntimeError("non-finite loss")
    opt.zero_grad(set_to_none=True); lo.total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip_norm)
    opt.step(); losses.append(float(lo.total))
    if step % 50 == 0 or step == tcfg.max_steps-1:
        t = {k: round(float(v),4) for k,v in lo.terms.items()}
        print(f"step {step:3d} loss {float(lo.total):.4f} {t}")

ckpt = OUT/"checkpoint.pt"
save_checkpoint(ckpt, model, cfg, list(adata.var_names))
print("\nloss: first10 mean %.4f  last10 mean %.4f" %
      (np.mean(losses[:10]), np.mean(losses[-10:])))

# --- use the project's own predict + evaluate paths, per perturbation ---
perts = [p for p in sorted(adata.obs['perturbation'].astype(str).unique()) if p!='control']
rows = []
for p in perts:
    pred_path = OUT/f"pred_{p}.h5ad"; met_path = OUT/f"metrics_{p}.json"
    predict_adata(ckpt, ADATA, p, pred_path, seed=0)
    m = evaluate_prediction_files(ADATA, pred_path, met_path)
    rows.append((p, m))
import numpy as np
def avg(k): return float(np.mean([m[k] for _,m in rows]))
summary = {k: round(avg(k),4) for k in rows[0][1]}
print("\n=== per-perturbation eval (project predict+evaluate on real data) ===")
for p,m in rows:
    print(f"{p}: pearson_delta={m['pearson_delta']:.3f} dir={m['direction_match']:.2f} "
          f"overlap@20={m['overlap_at_n']:.2f} dnorm_ratio={m['delta_norm_ratio']:.2f} mae={m['mae']:.3f}")
print("\nMEAN over %d perturbations:" % len(rows), summary)
json.dump({"summary": summary, "loss_first10": float(np.mean(losses[:10])),
           "loss_last10": float(np.mean(losses[-10:])), "n_params_M": n_params/1e6,
           "per_pert": {p:m for p,m in rows}}, open(OUT/"summary.json","w"), indent=2)
print("saved", OUT/"summary.json")
