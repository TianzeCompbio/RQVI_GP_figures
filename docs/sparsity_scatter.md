# GP Sparsity Scatter Plot

> **Sup figure panel b.** Empirical evidence that most RQVI GPs are sparse in *both* dimensions (cells and genes) and that PVE spans several orders of magnitude. Supports the sparsity language in §Figure 1 of the paper draft. See the panel-level storyline in [`internal_report.md`](internal_report.md).

Scatter plot of 256 GPs showing cell sparsity vs gene sparsity, colored by variance explained.

## Axes & color encoding

### X-axis — Proportion of activated cells

Read from `data/gp_summary_stats_seed0.csv`, column `frac_active` (cells with loading > 0.01). Displayed on a **log scale**; GP 207 (value = 0) is clipped to half the minimum positive value.

### Y-axis — Number of active genes

The raw gene-effect matrix `W` (19,805 genes × 256 GPs) is min-max normalized to
[-1, 1] using the global minimum and maximum across the entire matrix:

```
W_scaled = 2 * (W - W.min()) / (W.max() - W.min()) - 1
```

A gene is counted as "active" for a GP if `|W_scaled| > 0.45`. The threshold 0.45
was selected by generating a 2×3 grid of scatter plots at thresholds 0.40–0.50
(step 0.02) and choosing the value that best separated sparse vs. dense GPs.

> **Caveat:** 0.45 is a **heuristic** chosen visually on the same data, not a statistical cutoff. If reviewers ask, the simplest sensitivity analysis is to regenerate the same scatter at a few neighbouring thresholds (0.40, 0.45, 0.50) and confirm the qualitative shape is unchanged.

### Color — % Variance Explained (PVE)

Computed as:

```
var_j = sum(X[:,j]^2) * sum(W[:,j]^2)
PVE_j = var_j / sum(var_j)
```

- Cell loadings `X` come from the h5ad file
- Gene effects `W` come from the CSV
- Colorbar uses log scale (`LogNorm`)

> **Important clarification:** PVE here is the **fraction of total reconstructed variance attributable to each GP** — the 256 PVE values sum to 1. It is *not* the fraction of the dataset's total variance. A reader who sees "low PVE" should not interpret it as "this GP explains very little of the data"; it means "this GP contributes a small fraction of the total RQVI reconstruction", which is consistent with rare-cell programs that are still biologically interpretable (e.g. GP1 baseline housekeeping).

## Input data

| File | Description |
|------|-------------|
| `data/gp_summary_stats_seed0.csv` | 256 rows; provides `frac_active` |
| `data/gp_effects_matrix_seed0.csv` | 19,805 genes x 256 GPs; gene effect weights |
| `/data/tianzew/immgenT/RQVI_multiseeds/results/cmtloss08_64by4GPs_seed0.h5ad` | Cell loadings (external) |

## Script & output

- **Script:** `scripts/fig_gp_sparsity_scatter.py`
- **Intermediate CSV:** `data/gp_sparsity_scatter_data.csv` (256 rows: `gp_idx`, `frac_active_cells`, `n_active_genes`, `pve`)
- **Output figure:** `figures/gp_sparsity_scatter.pdf`

## How to reproduce

```bash
python figures_version_v2/scripts/fig_gp_sparsity_scatter.py
```
