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

def _vertical_dodge(ax, xs, ys, min_sep_px=14, max_shift_px=100):
    """
    Spread labels vertically so adjacent y-positions are at least
    `min_sep_px` pixels apart using a two-pass greedy sweep.
    Isolated labels stay near their data points; only crowded labels
    get pushed apart.  Capped at `max_shift_px` from original.
    Returns new y (data coords).
    """
    n = len(ys)
    if n == 0:
        return ys
    trans = ax.transData.transform
    inv = ax.transData.inverted()
    xy_px = trans(np.c_[xs, ys])
    orig_y_px = xy_px[:, 1].copy()
    order = np.argsort(orig_y_px)  # bottom -> top

    y0_px = trans((0, ax.get_ylim()[0]))[1]
    y1_px = trans((0, ax.get_ylim()[1]))[1]

    # Work on a sorted copy of y positions
    y_px = orig_y_px.copy()

    # Forward pass (bottom → top): push labels up only when too close
    for i in range(1, n):
        cur, prev = order[i], order[i - 1]
        if y_px[cur] - y_px[prev] < min_sep_px:
            y_px[cur] = y_px[prev] + min_sep_px

    # Backward pass (top → bottom): pull each label back toward its
    # original position while respecting min_sep with the label above
    for i in range(n - 2, -1, -1):
        cur = order[i]
        nxt = order[i + 1]
        ceiling = y_px[nxt] - min_sep_px
        floor = y_px[order[i - 1]] + min_sep_px if i > 0 else -np.inf
        y_px[cur] = np.clip(orig_y_px[cur], floor, ceiling)

    # Clamp to max_shift from original and plot bounds
    y_px = np.clip(y_px, orig_y_px - max_shift_px, orig_y_px + max_shift_px)
    y_px = np.clip(y_px, y0_px + 2, y1_px - 2)

    xy_px[:, 1] = y_px
    return inv.transform(xy_px)[:, 1]


def _multicolumn_dodge(ax, data_xs, data_ys, x_off_near, x_off_far,
                       min_sep_px=12, max_shift_px=180):
    """
    Split labels into near/far columns by alternating sorted-y,
    dodge each column independently.
    Returns: is_near (bool[]), x_lab[], y_lab[]
    """
    n = len(data_xs)
    if n == 0:
        return np.array([], dtype=bool), np.array([]), np.array([])

    order = np.argsort(data_ys)
    is_near = np.zeros(n, dtype=bool)
    is_near[order[0::2]] = True  # even-indexed (by sorted y) → near

    x_lab = np.where(is_near, data_xs + x_off_near, data_xs + x_off_far)
    y_lab = data_ys.copy()

    near_mask = is_near
    far_mask = ~is_near

    if near_mask.any():
        y_lab[near_mask] = _vertical_dodge(
            ax, x_lab[near_mask], y_lab[near_mask],
            min_sep_px=min_sep_px, max_shift_px=max_shift_px,
        )
    if far_mask.any():
        y_lab[far_mask] = _vertical_dodge(
            ax, x_lab[far_mask], y_lab[far_mask],
            min_sep_px=min_sep_px, max_shift_px=max_shift_px,
        )

    return is_near, x_lab, y_lab


def _nearest_edge_anchor(ax, text_artist, x_point, y_point, pad_px=3,
                         renderer=None):
    """Return (x, y) in data coords on nearest edge of text bbox to point."""
    if renderer is None:
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


def md_scatter(ax, gene_weights, mean_expr, title, top_k=50,
               point_size_bg=8, point_size_hl=25, min_label_sep_px=12,
               color_hl="#1f77b4", fontsize=7.0, x_offset_near_frac=0.04,
               x_offset_far_frac=0.10, max_shift_px=180,
               leader_gap_px=3):
    """MD scatter with de-crowded gene labels on a given ax.

    Uses multi-column label placement when a side has >15 labels.
    """
    df = pd.DataFrame({"x": gene_weights, "y": mean_expr}).dropna()

    # Grey cloud
    ax.scatter(df["x"], df["y"], s=point_size_bg, c="0.85", alpha=0.2,
               linewidths=0, zorder=1)

    # Top genes by absolute weight
    top_genes = df["x"].abs().nlargest(top_k).index
    hl = df.loc[top_genes].copy()
    ax.scatter(hl["x"], hl["y"], s=point_size_hl, facecolors="none",
               edgecolors=color_hl, linewidths=1.2, zorder=3)

    # Compute x-offsets in data coords
    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    x_off_near = x_offset_near_frac * x_span
    x_off_far = x_offset_far_frac * x_span

    hl["side_right"] = hl["x"] >= 0
    hl["x_lab"] = hl["x"].values.copy()
    hl["y_lab"] = hl["y"].values.copy()
    hl["is_near"] = True  # default: near column

    MULTICOLUMN_THRESHOLD = 15

    # Dodge per side
    right_mask = hl["side_right"].values
    for side_right, mask in [(True, right_mask), (False, ~right_mask)]:
        if not mask.any():
            continue
        side_idx = hl.index[mask]
        n_side = mask.sum()
        sign = 1.0 if side_right else -1.0

        if n_side > MULTICOLUMN_THRESHOLD:
            is_near, x_lab, y_lab = _multicolumn_dodge(
                ax,
                hl.loc[side_idx, "x"].values,
                hl.loc[side_idx, "y"].values,
                sign * x_off_near,
                sign * x_off_far,
                min_sep_px=min_label_sep_px,
                max_shift_px=max_shift_px,
            )
            hl.loc[side_idx, "x_lab"] = x_lab
            hl.loc[side_idx, "y_lab"] = y_lab
            hl.loc[side_idx, "is_near"] = is_near
        else:
            hl.loc[side_idx, "x_lab"] = hl.loc[side_idx, "x"] + sign * x_off_near
            hl.loc[side_idx, "y_lab"] = _vertical_dodge(
                ax,
                hl.loc[side_idx, "x_lab"].values,
                hl.loc[side_idx, "y_lab"].values,
                min_sep_px=min_label_sep_px,
                max_shift_px=max_shift_px,
            )

    # Draw text labels with white stroke — differentiated near/far styling
    texts = {}
    for g, r in hl.iterrows():
        ha = "left" if r["side_right"] else "right"
        near = r["is_near"]
        fs = fontsize if near else fontsize * 0.94  # 5.0 vs ~4.7
        zord = 5 if near else 4.5
        t = ax.text(
            r["x_lab"], r["y_lab"], g,
            ha=ha, va="center", fontsize=fs, color=color_hl,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            zorder=zord, clip_on=False,
        )
        texts[g] = t

    # Single draw for all text artists, then compute leader lines
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for g, r in hl.iterrows():
        t = texts[g]
        near = r["is_near"]
        lw = 0.5 if near else 0.4
        alpha = 0.6 if near else 0.45
        sx, sy = _nearest_edge_anchor(ax, t, r["x"], r["y"],
                                       pad_px=leader_gap_px, renderer=renderer)
        ax.annotate(
            "", xy=(r["x"], r["y"]), xytext=(sx, sy),
            arrowprops=dict(arrowstyle="-", lw=lw, color=color_hl, alpha=alpha),
            zorder=4,
        )

    ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
    ax.set_xlabel("Gene effect", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(x=0.05)
