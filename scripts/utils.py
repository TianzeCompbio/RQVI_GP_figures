"""
Shared utilities for GP overview figures (v2).
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.patheffects as pe
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

PATH_MAIN_H5AD = "/data/tianzew/immgenT/david_final_10k_genes.h5ad"
PATH_RQVI_H5AD = "/data/tianzew/immgenT/RQVI_multiseeds/results/cmtloss08_64by4GPs_seed0.h5ad"
PATH_UMAP_CSV = "/data/tianzew/immgenT/totalvi_20241006_mde.csv"
PATH_CLUSTER_MEANS = "/homes/gws/tianzew/projects/gene_program_model/Evaluation/function_analysis/corr_rst/rqvi_seed0_gp_cell_level.csv"
PATH_GENE_EFFECTS = str(PROJECT_DIR / "data/gp_effects_matrix_seed0.csv")

# ─── Constants ───────────────────────────────────────────────────────────────
LEVEL1_ORDER = ["CD4", "CD8", "Treg", "gdT", "CD8aa", "DN", "nonconv", "DP", "thymocyte"]
CLUSTER_COL = "Cluster_totalvi20240525rmigtsample_Res0.5"


# ─── Data loading ────────────────────────────────────────────────────────────

def load_main_obs() -> pd.DataFrame:
    """Load obs metadata from main h5ad (backed mode to save memory)."""
    adata = sc.read_h5ad(PATH_MAIN_H5AD, backed="r")
    obs = adata.obs.copy()
    adata.file.close()
    return obs


def load_cluster_means() -> pd.DataFrame:
    """Load pre-computed cluster-level mean GP loadings (114 clusters x 256 GPs)."""
    df = pd.read_csv(PATH_CLUSTER_MEANS, index_col="group")
    df.columns = df.columns.astype(str)
    return df


def load_gene_effects() -> pd.DataFrame:
    """Load gene effects matrix (19805 genes x 256 GPs)."""
    df = pd.read_csv(PATH_GENE_EFFECTS, index_col=0)
    df.columns = df.columns.astype(str)
    return df


def load_umap_coords() -> pd.DataFrame:
    """Load UMAP/MDE coords (633684 x 2)."""
    return pd.read_csv(PATH_UMAP_CSV, index_col=0)


# ─── Level1 aggregation ─────────────────────────────────────────────────────

def extract_level1(cluster_name: str) -> str:
    """Extract level1 prefix from cluster name, e.g. 'CD4_cl12' -> 'CD4'."""
    parts = cluster_name.rsplit("_cl", 1)
    return parts[0] if len(parts) == 2 else cluster_name


def compute_level1_means(
    cluster_means: pd.DataFrame,
    obs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute weighted mean GP loading per level1 cell type.
    Weights = number of cells in each cluster.
    Returns DataFrame: 9 level1 types x 256 GPs.
    """
    cluster_counts = obs[CLUSTER_COL].value_counts()
    cluster_to_level1 = {c: extract_level1(c) for c in cluster_means.index}

    rows = []
    for level1 in LEVEL1_ORDER:
        member_clusters = [c for c, l in cluster_to_level1.items() if l == level1]
        member_clusters = [c for c in member_clusters if c in cluster_counts.index]
        if not member_clusters:
            continue
        weights = cluster_counts[member_clusters].values.astype(float)
        weights /= weights.sum()
        vals = cluster_means.loc[member_clusters].values
        weighted_mean = (vals * weights[:, None]).sum(axis=0)
        rows.append(pd.Series(weighted_mean, index=cluster_means.columns, name=level1))

    return pd.DataFrame(rows)


# ─── MD scatter with de-crowded labels ──────────────────────────────────────

def _vertical_dodge(ax, xs, ys, min_sep_px=12):
    """Stack labels vertically so adjacent y-positions are >= min_sep_px apart."""
    trans = ax.transData.transform
    inv = ax.transData.inverted()
    xy_px = trans(np.c_[xs, ys])
    order = np.argsort(xy_px[:, 1])  # bottom -> top

    y0_px = trans((0, ax.get_ylim()[0]))[1]
    y1_px = trans((0, ax.get_ylim()[1]))[1]

    for i, idx in enumerate(order):
        if i == 0:
            xy_px[idx, 1] = max(xy_px[idx, 1], y0_px + 2)
        else:
            prev = xy_px[order[i - 1], 1]
            xy_px[idx, 1] = max(xy_px[idx, 1], prev + min_sep_px)
        xy_px[idx, 1] = min(xy_px[idx, 1], y1_px - 2)

    return inv.transform(xy_px)[:, 1]


def _nearest_edge_anchor(ax, text_artist, x_point, y_point, pad_px=3):
    """Return (x, y) in data coords on nearest edge of text bbox to point."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = text_artist.get_window_extent(renderer=renderer)

    candidates_px = [
        (bb.x0 - pad_px, 0.5 * (bb.y0 + bb.y1)),  # left
        (bb.x1 + pad_px, 0.5 * (bb.y0 + bb.y1)),  # right
        (0.5 * (bb.x0 + bb.x1), bb.y0 - pad_px),  # bottom
        (0.5 * (bb.x0 + bb.x1), bb.y1 + pad_px),  # top
    ]

    px, py = ax.transData.transform((x_point, y_point))
    d2 = [(px - cx) ** 2 + (py - cy) ** 2 for (cx, cy) in candidates_px]
    cx, cy = candidates_px[int(np.argmin(d2))]

    return ax.transData.inverted().transform((cx, cy))


def md_scatter(ax, gene_weights, mean_expr, title, top_k=30,
               point_size_bg=8, point_size_hl=20, min_label_sep_px=12,
               color_hl="#1f77b4", fontsize=5.5, x_offset_frac=0.02,
               leader_gap_px=3):
    """MD scatter with de-crowded gene labels on a given ax."""
    df = pd.DataFrame({"x": gene_weights, "y": mean_expr}).dropna()

    # Grey cloud
    ax.scatter(df["x"], df["y"], s=point_size_bg, c="0.85", alpha=0.2,
               linewidths=0, zorder=1)

    # Top genes by absolute weight
    top_genes = df["x"].abs().nlargest(top_k).index
    hl = df.loc[top_genes].copy()
    ax.scatter(hl["x"], hl["y"], s=point_size_hl, facecolors="none",
               edgecolors=color_hl, linewidths=0.9, zorder=3)

    # Build label positions with side-aware x-offset
    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    x_off = x_offset_frac * x_span
    hl["side_right"] = hl["x"] >= 0
    hl["x_lab"] = np.where(hl["side_right"], hl["x"] + x_off, hl["x"] - x_off)
    hl["y_lab"] = hl["y"].values.copy()

    # Vertical dodge per side
    right_mask = hl["side_right"].values
    if right_mask.any():
        hl.loc[right_mask, "y_lab"] = _vertical_dodge(
            ax, hl.loc[right_mask, "x_lab"].values,
            hl.loc[right_mask, "y_lab"].values,
            min_sep_px=min_label_sep_px,
        )
    if (~right_mask).any():
        hl.loc[~right_mask, "y_lab"] = _vertical_dodge(
            ax, hl.loc[~right_mask, "x_lab"].values,
            hl.loc[~right_mask, "y_lab"].values,
            min_sep_px=min_label_sep_px,
        )

    # Draw text labels with white stroke
    texts = {}
    for g, r in hl.iterrows():
        ha = "left" if r["side_right"] else "right"
        t = ax.text(
            r["x_lab"], r["y_lab"], g,
            ha=ha, va="center", fontsize=fontsize, color=color_hl,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
            zorder=5, clip_on=False,
        )
        texts[g] = t

    # Leader lines from text edge to data point
    for g, r in hl.iterrows():
        t = texts[g]
        sx, sy = _nearest_edge_anchor(ax, t, r["x"], r["y"], pad_px=leader_gap_px)
        ax.annotate(
            "", xy=(r["x"], r["y"]), xytext=(sx, sy),
            arrowprops=dict(arrowstyle="-", lw=0.6, color=color_hl, alpha=0.7),
            zorder=4,
        )

    ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
    ax.set_xlabel("Gene effect", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
