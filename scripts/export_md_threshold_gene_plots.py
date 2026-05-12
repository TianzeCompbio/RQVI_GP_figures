"""
Export MD scatter plots and gene-effect tables for the >0.1 request.

This script makes exploratory versions of the Figure 7 / Figure S7 MD scatter
panels. Every gene with positive gene effect > THRESHOLD is highlighted, while
only a compact set of top genes is labelled so the panels remain readable.
It also writes the exact values used for plotting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from scipy import sparse
from utils import (
    EXAMPLE_BG_POINT,
    EXAMPLE_BLUE,
    EXAMPLE_TEXT,
    _nearest_edge_anchor,
    _vertical_dodge,
    style_example_axes,
)


V2_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = V2_DIR.parent
OUT_DIR = V2_DIR / "md_scatter_gene_effect_gt0p1"
PLOT_DIR = OUT_DIR / "plots"
TABLE_DIR = OUT_DIR / "tables"

PATH_MAIN_H5AD = Path("/data/tianzew/immgenT/david_final_10k_genes.h5ad")
PATH_FLASHIER_GENE = Path(
    "/homes/gws/tianzew/projects/gene_program_model/Evaluation/Subcluster/gene_factor_matrix.txt"
)
PATH_RQVI_GENE_TEMPLATE = WORKSPACE_DIR / "data" / "RQVI_gene_factors" / "gp_effects_matrix_seed{seed}.csv"

THRESHOLD = 0.1
DEFAULT_LABEL_TOP_N_MAIN = 12
DEFAULT_LABEL_TOP_N_SUPP = 10

# Manual highlight hook:
# Add genes here when you want them explicitly labelled and circled in every
# panel, even if they do not pass THRESHOLD. This is useful for checking where
# canonical markers sit in the MD scatter before editing the final figure.
MANUAL_HIGHLIGHT_GENES = ["Cd8a", "Cd8b1", "Cd8b"]


@dataclass(frozen=True)
class PanelSpec:
    panel: str
    figure_label: str
    method: str
    display_name: str
    seed: int | None
    gp: int | None
    flashier_factor: int | None
    source_path: Path
    source_column: str
    label_top_n: int


PANELS = [
    PanelSpec(
        panel="Fig7f",
        figure_label="Figure 7f",
        method="RQVI",
        display_name="RQVI GP45 (seed 0)",
        seed=0,
        gp=45,
        flashier_factor=None,
        source_path=Path(str(PATH_RQVI_GENE_TEMPLATE).format(seed=0)),
        source_column="45",
        label_top_n=DEFAULT_LABEL_TOP_N_MAIN,
    ),
    PanelSpec(
        panel="Fig7g",
        figure_label="Figure 7g",
        method="Flashier",
        display_name="Flashier F35",
        seed=None,
        gp=None,
        flashier_factor=35,
        source_path=PATH_FLASHIER_GENE,
        source_column="V35",
        label_top_n=DEFAULT_LABEL_TOP_N_MAIN,
    ),
    PanelSpec(
        panel="FigS7g",
        figure_label="Figure S7g",
        method="RQVI",
        display_name="RQVI GP96 (seed 6), best match for Flashier F22",
        seed=6,
        gp=96,
        flashier_factor=22,
        source_path=Path(str(PATH_RQVI_GENE_TEMPLATE).format(seed=6)),
        source_column="96",
        label_top_n=DEFAULT_LABEL_TOP_N_SUPP,
    ),
    PanelSpec(
        panel="FigS7h",
        figure_label="Figure S7h",
        method="RQVI",
        display_name="RQVI GP112 (seed 5), best match for Flashier F30",
        seed=5,
        gp=112,
        flashier_factor=30,
        source_path=Path(str(PATH_RQVI_GENE_TEMPLATE).format(seed=5)),
        source_column="112",
        label_top_n=DEFAULT_LABEL_TOP_N_SUPP,
    ),
    PanelSpec(
        panel="FigS7i",
        figure_label="Figure S7i",
        method="RQVI",
        display_name="RQVI GP110 (seed 1), best match for Flashier F58",
        seed=1,
        gp=110,
        flashier_factor=58,
        source_path=Path(str(PATH_RQVI_GENE_TEMPLATE).format(seed=1)),
        source_column="110",
        label_top_n=DEFAULT_LABEL_TOP_N_SUPP,
    ),
    PanelSpec(
        panel="FigS7j",
        figure_label="Figure S7j",
        method="RQVI",
        display_name="RQVI GP76 (seed 3), best match for Flashier F68",
        seed=3,
        gp=76,
        flashier_factor=68,
        source_path=Path(str(PATH_RQVI_GENE_TEMPLATE).format(seed=3)),
        source_column="76",
        label_top_n=DEFAULT_LABEL_TOP_N_SUPP,
    ),
]


def ensure_dirs() -> None:
    for path in (PLOT_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def compute_or_load_mean_expression() -> pd.Series:
    cache_path = TABLE_DIR / "mean_log_expression_used.csv"
    if cache_path.exists():
        print(f"Loading cached mean expression: {cache_path}")
        return pd.read_csv(cache_path, index_col=0)["mean_log_expr"]

    print(f"Computing mean expression from {PATH_MAIN_H5AD}")
    adata = ad.read_h5ad(PATH_MAIN_H5AD, backed="r")
    n_cells, n_genes = adata.shape
    gene_names = adata.var_names.to_numpy()
    gene_sums = np.zeros(n_genes, dtype=np.float64)
    chunk_size = 50_000

    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        chunk = adata.X[start:end]
        if sparse.issparse(chunk):
            chunk = chunk.toarray()
        gene_sums += np.asarray(chunk).sum(axis=0).ravel()
        if start % 200_000 == 0:
            print(f"  processed {start:,}/{n_cells:,} cells")

    adata.file.close()
    mean_expr = pd.Series(gene_sums / n_cells, index=gene_names, name="mean_log_expr")
    mean_expr.to_csv(cache_path)
    print(f"Saved mean expression cache: {cache_path}")
    return mean_expr


def load_effect_series(spec: PanelSpec) -> pd.Series:
    print(f"Loading {spec.display_name}: {spec.source_path} [{spec.source_column}]")
    df = pd.read_csv(spec.source_path, sep="\t" if spec.method == "Flashier" else ",", index_col=0)
    if spec.source_column not in df.columns:
        raise KeyError(f"{spec.source_column} not found in {spec.source_path}")
    return df[spec.source_column].rename("gene_effect")


def panel_table(spec: PanelSpec, gene_effect: pd.Series, mean_expr: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"gene_effect": gene_effect}).join(mean_expr, how="inner").dropna()
    df.index.name = "gene"
    df = df.reset_index()
    df.insert(0, "panel", spec.panel)
    df.insert(1, "figure_label", spec.figure_label)
    df.insert(2, "method", spec.method)
    df.insert(3, "program", spec.display_name)
    df.insert(4, "seed", spec.seed)
    df.insert(5, "rqvi_gp", spec.gp)
    df.insert(6, "flashier_factor", spec.flashier_factor)
    df["source_column"] = spec.source_column
    df["above_0p1"] = df["gene_effect"] > THRESHOLD
    df["abs_above_0p1"] = df["gene_effect"].abs() > THRESHOLD
    df["rank_positive_effect"] = (
        df["gene_effect"].rank(method="first", ascending=False).astype(int)
    )
    df["rank_abs_effect"] = (
        df["gene_effect"].abs().rank(method="first", ascending=False).astype(int)
    )
    return df.sort_values("gene_effect", ascending=False)


def plot_threshold_panel(spec: PanelSpec, values: pd.DataFrame) -> None:
    all_genes = values.set_index("gene")
    highlights = all_genes[all_genes["gene_effect"] > THRESHOLD].copy()

    label_genes = list(
        highlights.sort_values("gene_effect", ascending=False)
        .head(spec.label_top_n)
        .index
    )
    for marker in MANUAL_HIGHLIGHT_GENES:
        if marker in all_genes.index and marker not in label_genes:
            label_genes.append(marker)

    fig, ax = plt.subplots(figsize=(4.4, 3.6), dpi=180)

    ax.scatter(
        all_genes["gene_effect"],
        all_genes["mean_log_expr"],
        s=7,
        c=EXAMPLE_BG_POINT,
        alpha=0.18,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    ax.scatter(
        highlights["gene_effect"],
        highlights["mean_log_expr"],
        s=18,
        c=EXAMPLE_BLUE,
        alpha=0.70,
        edgecolors="white",
        linewidths=0.25,
        zorder=3,
    )

    marker_genes = [
        gene for gene in MANUAL_HIGHLIGHT_GENES
        if gene in all_genes.index and gene not in highlights.index
    ]
    if marker_genes:
        marker_rows = all_genes.loc[marker_genes]
        ax.scatter(
            marker_rows["gene_effect"],
            marker_rows["mean_log_expr"],
            s=26,
            facecolors="white",
            edgecolors="0.35",
            linewidths=0.7,
            zorder=3.2,
        )

    ax.margins(x=0.08, y=0.08)
    style_example_axes(ax, grid=True)
    ax.set_title(
        spec.display_name,
        fontsize=10,
        fontweight="normal",
        loc="left",
        color=EXAMPLE_TEXT,
        pad=6,
    )
    ax.set_xlabel("Gene effect", fontsize=10)
    ax.set_ylabel("Mean log expr", fontsize=10)

    label_rows = all_genes.loc[label_genes].copy()
    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    x_offset = 0.018 * x_span
    label_rows["side_right"] = label_rows["gene_effect"] >= 0
    label_rows["x_lab"] = label_rows["gene_effect"] + np.where(
        label_rows["side_right"], x_offset, -x_offset
    )
    label_rows["y_lab"] = label_rows["mean_log_expr"]

    ax.figure.canvas.draw()
    for side_right, mask in (
        (True, label_rows["side_right"].values),
        (False, ~label_rows["side_right"].values),
    ):
        if not mask.any():
            continue
        side_idx = label_rows.index[mask]
        label_rows.loc[side_idx, "y_lab"] = _vertical_dodge(
            ax,
            label_rows.loc[side_idx, "x_lab"].values,
            label_rows.loc[side_idx, "y_lab"].values,
            min_sep_px=12,
            max_shift_px=70,
        )

    texts = {}
    for gene, row in label_rows.iterrows():
        ha = "left" if row["side_right"] else "right"
        texts[gene] = ax.text(
            row["x_lab"],
            row["y_lab"],
            gene,
            ha=ha,
            va="center",
            fontsize=7.0,
            color=EXAMPLE_TEXT,
            path_effects=[pe.withStroke(linewidth=1.4, foreground="white")],
            clip_on=False,
            zorder=4,
        )

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    for gene, row in label_rows.iterrows():
        sx, sy = _nearest_edge_anchor(
            ax,
            texts[gene],
            row["gene_effect"],
            row["mean_log_expr"],
            pad_px=3,
            renderer=renderer,
        )
        ax.annotate(
            "",
            xy=(row["gene_effect"], row["mean_log_expr"]),
            xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-",
                lw=0.55,
                color="0.45",
                alpha=0.85,
                shrinkA=0,
                shrinkB=2,
            ),
            clip_on=False,
            zorder=3.5,
        )

    fig.tight_layout()
    stem = f"{spec.panel}_{spec.method}_{spec.source_column}_gene_effect_gt0p1"
    pdf_path = PLOT_DIR / f"{stem}.pdf"
    png_path = PLOT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"Saved {pdf_path}")


def write_readme(panel_summaries: pd.DataFrame) -> None:
    readme = OUT_DIR / "README.md"
    display_cols = [
        "panel",
        "program",
        "source_column",
        "genes_gt_0.1",
        "max_gene_effect",
        "source_path",
    ]
    summary_df = panel_summaries[display_cols].copy()
    summary_df["max_gene_effect"] = summary_df["max_gene_effect"].map(lambda x: f"{x:.3f}")
    header = "| " + " | ".join(display_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(display_cols)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in display_cols) + " |"
        for _, row in summary_df.iterrows()
    ]
    summary_md = "\n".join([header, sep, *rows])
    readme.write_text(
        f"""# MD Scatter Gene-Effect Export (>0.1)

This folder contains exploratory versions of the Figure 7 / Figure S7 MD scatter panels with all genes above the requested effect threshold highlighted.

Threshold definition: genes are highlighted in blue when their positive gene effect/loading is greater than `{THRESHOLD}`. The labels are kept compact, following the style of the original MD scatter panels: only the top positive-effect genes plus manually highlighted genes are labelled. Grey points are all genes used in the MD scatter. The y-axis is mean log expression computed from the main immgenT h5ad file.

## Panel Summary

{summary_md}

## Files

- `plots/*.pdf` and `plots/*.png`: one original-style MD scatter per panel, with all >`{THRESHOLD}` genes highlighted.
- `tables/all_gene_values_for_md_scatter.csv`: long-format table with every gene used in every MD scatter.
- `tables/genes_effect_gt0p1.csv`: subset of genes with positive gene effect/loading > `{THRESHOLD}`.
- `tables/<panel>_*_gt0p1.csv`: per-panel threshold tables.
- `tables/mean_log_expression_used.csv`: cached mean log expression used on the y-axis.
- `tables/manual_highlight_gene_positions.csv`: lookup table for manually highlighted genes (`{", ".join(MANUAL_HIGHLIGHT_GENES)}`) even when they do not pass the >`{THRESHOLD}` threshold.

## Manual Highlight Genes

To force specific genes to appear on the plots, edit the `MANUAL_HIGHLIGHT_GENES` list near the top of `scripts/export_md_threshold_gene_plots.py`.

For example:

```python
MANUAL_HIGHLIGHT_GENES = ["Cd8a", "Cd8b1", "Cd4", "Nkg7"]
```

Then rerun:

```bash
MPLCONFIGDIR=/tmp/mplconfig python scripts/export_md_threshold_gene_plots.py
```

Manual-highlight genes are labelled and circled in every panel where they are present. Genes with `gene_effect > {THRESHOLD}` are still highlighted in blue; manual-highlight genes below the threshold are shown as white circles with dark outlines. Their exact coordinates are written to `tables/manual_highlight_gene_positions.csv`.

## Source Data

- Main expression h5ad for mean log expression: `{PATH_MAIN_H5AD}`
- Flashier gene effects: `{PATH_FLASHIER_GENE}`
- RQVI gene effects: `{PATH_RQVI_GENE_TEMPLATE}`

## Reproduction

Run from the `figures_version_v2` repository root:

```bash
MPLCONFIGDIR=/tmp/mplconfig python scripts/export_md_threshold_gene_plots.py
```
""",
        encoding="utf-8",
    )
    print(f"Saved {readme}")


def main() -> None:
    ensure_dirs()
    mean_expr = compute_or_load_mean_expression()

    all_tables = []
    summary_rows = []
    for spec in PANELS:
        effects = load_effect_series(spec)
        values = panel_table(spec, effects, mean_expr)
        all_tables.append(values)

        threshold_values = values[values["above_0p1"]].copy()
        per_panel_path = TABLE_DIR / f"{spec.panel}_{spec.method}_{spec.source_column}_gt0p1.csv"
        threshold_values.to_csv(per_panel_path, index=False)

        plot_threshold_panel(spec, values)

        summary_rows.append(
            {
                "panel": spec.panel,
                "program": spec.display_name,
                "source_column": spec.source_column,
                "genes_gt_0.1": len(threshold_values),
                "max_gene_effect": values["gene_effect"].max(),
                "source_path": str(spec.source_path),
            }
        )

    all_values = pd.concat(all_tables, ignore_index=True)
    all_values_path = TABLE_DIR / "all_gene_values_for_md_scatter.csv"
    threshold_path = TABLE_DIR / "genes_effect_gt0p1.csv"
    all_values.to_csv(all_values_path, index=False)
    all_values[all_values["above_0p1"]].to_csv(threshold_path, index=False)
    marker_positions = all_values[all_values["gene"].isin(MANUAL_HIGHLIGHT_GENES)].copy()
    marker_positions_path = TABLE_DIR / "manual_highlight_gene_positions.csv"
    marker_positions.to_csv(marker_positions_path, index=False)
    print(f"Saved {all_values_path}")
    print(f"Saved {threshold_path}")
    print(f"Saved {marker_positions_path}")

    panel_summaries = pd.DataFrame(summary_rows)
    panel_summaries.to_csv(TABLE_DIR / "panel_summary.csv", index=False)
    write_readme(panel_summaries)


if __name__ == "__main__":
    main()
