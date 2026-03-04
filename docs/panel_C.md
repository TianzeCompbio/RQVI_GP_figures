# Panel C — GP Sparsity Scatter Plot

Scatter plot of 256 GPs showing cell sparsity vs gene sparsity, colored by variance explained.

## Axes & color encoding

### X-axis — Proportion of activated cells

Read from `data/gp_summary_stats_seed0.csv`, column `frac_active` (cells with loading > 0.01). Displayed on a **log scale**; GP 207 (value = 0) is clipped to half the minimum positive value.

### Y-axis — Proportion of activated genes

Computed via MAD-based outlier detection on each GP's gene effect column from `data/gp_effects_matrix_seed0.csv`. A gene is "activated" if its modified Z-score exceeds 3.5 in absolute value:

```
z = (x - median) / (MAD * 1.4826)
activated if |z| > 3.5
```

### Color — % Variance Explained (PVE)

Computed as:

```
var_j = sum(X[:,j]^2) * sum(W[:,j]^2)
PVE_j = var_j / sum(var_j)
```

- Cell loadings `X` come from the h5ad file
- Gene effects `W` come from the CSV
- Colorbar uses log scale (`LogNorm`)

## Input data

| File | Description |
|------|-------------|
| `data/gp_summary_stats_seed0.csv` | 256 rows; provides `frac_active` |
| `data/gp_effects_matrix_seed0.csv` | 19,805 genes x 256 GPs; gene effect weights |
| `/data/tianzew/immgenT/RQVI_multiseeds/results/cmtloss08_64by4GPs_seed0.h5ad` | Cell loadings (external) |

## Script & output

- **Script:** `scripts/fig_gp_sparsity_scatter.py`
- **Intermediate CSV:** `data/gp_sparsity_scatter_data.csv` (256 rows: `gp_idx`, `frac_active_cells`, `frac_active_genes`, `pve`)
- **Output figure:** `figures/gp_sparsity_scatter.pdf`

## How to reproduce

```bash
python figures_version_v2/scripts/fig_gp_sparsity_scatter.py
```
