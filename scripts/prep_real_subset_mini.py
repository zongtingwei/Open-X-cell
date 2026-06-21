"""Build a 4000-gene HepG2 Nadig subset so the *mini* (512-d) architecture can run
(mini needs >= context_length = 4000 genes). Same 12 perturbations and control/cell
sampling as prep_real_subset.py, only the gene panel grows 400 -> 4000. Used to test
scGPT gene-embedding initialization on a paper-faithful architecture."""
import anndata as ad, numpy as np, pandas as pd

SRC = "/public2/YouYuning/zongtingwei/vcc_benchmarks/datasets/hepg2nadig.h5ad"
OUT = "data/real/hepg2nadig_mini4k.h5ad"
N_PERT = 12
N_CTRL = 1500
PANEL = 4000          # = mini context_length / num_genes lower bound
CELLS_PER_PERT = 200
rng = np.random.default_rng(0)

a = ad.read_h5ad(SRC, backed='r')
obs = a.obs
is_ctrl = (obs['transcript'] == 'non-targeting').to_numpy()
gene_id = obs['gene_id'].astype(str).to_numpy()
var_ensg = np.asarray(a.var_names)
ensg_to_col = {g: i for i, g in enumerate(var_ensg)}

pert_counts = pd.Series(gene_id[~is_ctrl]).value_counts()
cand = [g for g in pert_counts.index
        if g in ensg_to_col and pert_counts[g] >= CELLS_PER_PERT][:N_PERT]
print("chosen perturbations (ENSG):", cand)

mean_expr = pd.to_numeric(a.var['mean-1'], errors='coerce').to_numpy()
order = np.argsort(np.nan_to_num(mean_expr))[::-1]
panel_cols = list(dict.fromkeys([ensg_to_col[g] for g in cand] +
                                [int(i) for i in order]))[:PANEL]
panel_cols = sorted(panel_cols)
panel_ensg = var_ensg[panel_cols]
print("panel size:", len(panel_cols))

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
      "| controls", int((labels == 'control').sum()), "| perts", len(cand))
# how many panel genes have scGPT embeddings (for the init coverage estimate)
import os
npz = "data/priors/scgpt_gene_embedding.npz"
if os.path.exists(npz):
    have = set(np.load(npz, allow_pickle=False)["genes"].astype(str))
    cov = sum(g in have for g in panel_ensg)
    print(f"scGPT coverage of panel: {cov}/{len(panel_ensg)} ({100*cov/len(panel_ensg):.1f}%)")
