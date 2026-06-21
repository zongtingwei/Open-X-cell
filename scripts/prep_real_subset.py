"""Build a compact, CPU-sized real perturb-seq subset (HepG2 Nadig CRISPRi)
mapped to the Open-X-Cell obs schema. Real X-Atlas/Pisces-style data."""
import anndata as ad, numpy as np, pandas as pd

SRC = "/public2/YouYuning/zongtingwei/vcc_benchmarks/datasets/hepg2nadig.h5ad"
OUT = "data/real/hepg2nadig_subset.h5ad"
N_PERT = 12          # number of real perturbations to keep
N_CTRL = 1500        # control (non-targeting) cells
PANEL = 400          # gene panel size (num_genes/context_length)
CELLS_PER_PERT = 200
rng = np.random.default_rng(0)

a = ad.read_h5ad(SRC, backed='r')
obs = a.obs
is_ctrl = (obs['transcript'] == 'non-targeting').to_numpy()
gene_id = obs['gene_id'].astype(str).to_numpy()           # ENSG, matches var_names
var_ensg = np.asarray(a.var_names)
ensg_to_col = {g: i for i, g in enumerate(var_ensg)}

# candidate perturbations: targeted, ENSG present in var, enough cells
pert_counts = pd.Series(gene_id[~is_ctrl]).value_counts()
cand = [g for g in pert_counts.index
        if g in ensg_to_col and pert_counts[g] >= CELLS_PER_PERT][:N_PERT]
print("chosen perturbations (ENSG):", cand)

# gene panel: the perturbed genes + top-expressed genes by var mean
mean_expr = pd.to_numeric(a.var['mean-1'], errors='coerce').to_numpy()
order = np.argsort(np.nan_to_num(mean_expr))[::-1]
panel_cols = list(dict.fromkeys([ensg_to_col[g] for g in cand] +
                                [int(i) for i in order]))[:PANEL]
panel_cols = sorted(panel_cols)
panel_ensg = var_ensg[panel_cols]
print("panel size:", len(panel_cols))

# pick rows: controls + cells for each perturbation
ctrl_rows = np.flatnonzero(is_ctrl)
ctrl_rows = rng.choice(ctrl_rows, size=min(N_CTRL, len(ctrl_rows)), replace=False)
rows = [ctrl_rows]
labels = ['control'] * len(ctrl_rows)
for g in cand:
    r = np.flatnonzero((gene_id == g) & ~is_ctrl)
    r = rng.choice(r, size=min(CELLS_PER_PERT, len(r)), replace=False)
    rows.append(r); labels += [g] * len(r)
rows = np.concatenate(rows)

X = a.X[rows][:, panel_cols]
X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X, dtype=np.float32)
labels = np.array(labels)
new_obs = pd.DataFrame({
    'perturbation': labels,
    'context': 'hepg2',
    'batch': 'batch0',
    'is_control': labels == 'control',
})
sub = ad.AnnData(X=X.astype(np.float32), obs=new_obs,
                 var=pd.DataFrame(index=panel_ensg))
sub.write_h5ad(OUT)
print("wrote", OUT, sub.shape,
      "| controls", int((labels=='control').sum()),
      "| perts", len(cand))
print("X log1p? max=%.2f mean=%.3f" % (X.max(), X.mean()))
