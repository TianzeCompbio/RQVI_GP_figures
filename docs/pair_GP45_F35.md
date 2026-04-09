# GP 45 vs Flashier F35 Pair Comparison

> **Main figure panels c, d, e, f.** This is the rhetorical centerpiece of the main figure: the *worst-matched* best pair in the histogram (cluster-level r ≈ 0.436, below the 0.5 threshold), shown deliberately to make a stronger point than the histogram alone could. See the panel-level storyline in [`internal_report.md`](internal_report.md).

## Why this pair is in the main figure

GP45 / F35 was selected as the **lowest-r best-match pair** in the cross-method best-match histogram (`hist_standalone.pdf`). The rhetorical claim is:

> Even when the cross-method correlation is at its lowest, the underlying biology converges. Both GP45 and F35 are TCR-activation programs; the lead genes labelled in the MD scatters overlap, and the UMAP loadings highlight the same activated-T-cell populations.

The point is *not* that RQVI and Flashier agree on this pair by Pearson r — they explicitly do not. The point is that the *biology* still converges. This is a stronger version of the convergent-discovery claim than the histogram on its own can support: if even the worst pair recovers the same gene programme, the method-level convergence claim holds throughout the distribution.

For the companion good-match pair, see [`pair_GP38_F58.md`](pair_GP38_F58.md) (r ≈ 0.573, above threshold).

> **Action items:**
> - The r = 0.436 value is currently **not annotated on the PDF** — it lives only in the filename and this doc. Add it as a subtitle in `scripts/fig_pair_GP45_F35.py` so a reader of the figure sees the number.
> - The shared TCR-activation lead genes are visible in the MD scatters (top 50 are labelled by `md_scatter` in `utils.py`); naming a few in the figure caption / legend would make the rhetorical claim land harder.

---

## Methodology

2x2 standalone figure comparing RQVI GP 45 to Flashier F35 (best-match r=0.436, below the 0.5 threshold).

## Layout

- **Top row:** UMAP cell loading plots (RQVI GP 45 left, Flashier F35 right), colored by loading magnitude
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

`scripts/fig_pair_GP45_F35.py` -> `figures/pair_GP45_F35.pdf`

## How to reproduce

```bash
python scripts/fig_pair_GP45_F35.py
```

No upstream dependencies — all data is loaded directly from external files.
