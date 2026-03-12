# Panel E — GP 38 vs Flashier F58 Pair Comparison

2x2 standalone figure comparing RQVI GP 38 to Flashier F58 (best-match r=0.573, above the 0.5 threshold).

## Layout

- **Top row:** UMAP cell loading plots (RQVI GP 38 left, Flashier F58 right), colored by loading magnitude
- **Bottom row:** MD (mean-expression vs. gene-effect) scatter plots for the same pair, showing gene-level effect sizes against mean log expression

## Inputs

| Path | What |
|------|------|
| `/data/tianzew/immgenT/david_final_10k_genes.h5ad` | Main dataset (obs metadata + expression for MD plots) via `utils.py` |
| `/data/tianzew/immgenT/RQVI_multiseeds/results/cmtloss08_64by4GPs_seed0.h5ad` | RQVI cell loadings (seed 0) via `utils.py` |
| `/data/tianzew/immgenT/RQVI/cmtloss08_64by4GPs_mde_totalVI.h5ad` | UMAP coordinates for cell-level plots |
| `.../Evaluation/Subcluster/cell_factor_matrix.txt` | Flashier cell loadings |
| `.../Evaluation/Subcluster/gene_factor_matrix.txt` | Flashier gene effects |

RQVI gene effects are loaded via `utils.load_gene_effects()`.

## Script

`scripts/fig_pair_GP38_F58.py` -> `figures/pair_GP38_F58.pdf`

## How to reproduce

```bash
python scripts/fig_pair_GP38_F58.py
```

No upstream dependencies — all data is loaded directly from external files.
