# GP 45 vs Flashier F35 Pair Comparison

> **Main figure panels c, d, e, f.** This is the rhetorical centerpiece of the main figure: the *worst-matched* best pair in the histogram (cluster-level r ≈ 0.436, below the 0.5 threshold), shown deliberately to make a stronger point than the histogram alone could. See the panel-level storyline in [`figures_summary.md`](figures_summary.md).

## Why this pair is in the main figure

GP45 / F35 was selected as the **lowest-r best-match pair** in the cross-method best-match histogram (`hist_standalone.pdf`). The rhetorical claim is:

> Even when the cross-method correlation is at its lowest, the underlying biology converges. Both GP45 and F35 are TCR-activation programs; the lead genes labelled in the MD scatters overlap, and the UMAP loadings highlight the same activated-T-cell populations.

The point is *not* that RQVI and Flashier agree on this pair by Pearson r — they explicitly do not. The point is that the *biology* still converges. This is a stronger version of the convergent-discovery claim than the histogram on its own can support: if even the worst pair recovers the same gene programme, the method-level convergence claim holds throughout the distribution.

For the companion good-match pair, see [`pair_GP38_F58.md`](pair_GP38_F58.md) (r ≈ 0.573, above threshold).

---

## Methodology

2x2 standalone figure comparing RQVI GP 45 to Flashier F35 (best-match r=0.436, below the 0.5 threshold).

## Layout

- **Top row:** UMAP cell loading plots (RQVI GP 45 left, Flashier F35 right), colored by loading magnitude
- **Bottom row:** MD (mean-expression vs. gene-effect) scatter plots for the same pair, showing gene-level effect sizes against mean log expression
- The UMAP cell loadings are display-scaled to 0-1 using the positive 99.5th percentile as the saturation point, so the colorbars remain 0-1 while avoiding compression by extreme outliers.

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

`scripts/fig_pair_GP45_F35.py` -> `figures/main_figures/pair_GP45_F35.pdf`

The script also writes editable standalone panel PDFs to:

- `figures/main_figures/pair_GP45_F35_panels/pair_GP45_F35_panel_C_rqvi_GP45_umap.pdf`
- `figures/main_figures/pair_GP45_F35_panels/pair_GP45_F35_panel_D_flashier_F35_umap.pdf`
- `figures/main_figures/pair_GP45_F35_panels/pair_GP45_F35_panel_E_rqvi_GP45_md_scatter.pdf`
- `figures/main_figures/pair_GP45_F35_panels/pair_GP45_F35_panel_F_flashier_F35_md_scatter.pdf`

## How to reproduce

```bash
python scripts/fig_pair_GP45_F35.py
```

No upstream dependencies — all data is loaded directly from external files.
