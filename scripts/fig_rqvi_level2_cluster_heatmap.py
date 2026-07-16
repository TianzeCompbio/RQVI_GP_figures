"""Plot mean seed-0 RQVI loading for every GP and level2 cluster.

Rows are the 256 RQVI gene programs.  Columns are the 114 fine-grained
level2 clusters.  The displayed values are untransformed arithmetic means of
the cell-level loadings.  Clusters are grouped by level1 lineage; programs are
ordered by hierarchical clustering of their L2-normalized cluster profiles.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_DIR / "data" / "rqvi_cell_loadings_seed0.h5"
DEFAULT_MEANS = PROJECT_DIR / "data" / "rqvi_seed0_mean_loadings_by_level2_cluster.csv"
DEFAULT_GP_ORDER = PROJECT_DIR / "data" / "rqvi_seed0_level2_heatmap_gp_order.csv"
DEFAULT_CLUSTER_ORDER = (
    PROJECT_DIR / "data" / "rqvi_seed0_level2_heatmap_cluster_order.csv"
)
DEFAULT_PDF = PROJECT_DIR / "figures" / "main_figures" / "rqvi_level2_cluster_heatmap.pdf"
DEFAULT_PNG = PROJECT_DIR / "figures" / "main_figures" / "rqvi_level2_cluster_heatmap.png"

LEVEL1_ORDER = [
    "CD4",
    "CD8",
    "Treg",
    "gdT",
    "CD8aa",
    "DN",
    "nonconv",
    "DP",
    "thymocyte",
]

LEVEL1_COLORS = {
    "CD4": "#3B82C4",
    "CD8": "#F28E2B",
    "Treg": "#D84A9B",
    "gdT": "#59A14F",
    "CD8aa": "#B07AA1",
    "DN": "#35A7C9",
    "nonconv": "#9C755F",
    "DP": "#76B7B2",
    "thymocyte": "#BAB0AC",
}


def _decode(dataset: h5py.Dataset) -> np.ndarray:
    return dataset.asstr()[:]


def _cluster_natural_key(label: str) -> tuple[int, int, str]:
    """Sort numeric clusters before miniverse/proliferating meta-clusters."""
    match = re.search(r"(?:_cl|_)(\d+)$", label)
    if match:
        return (0, int(match.group(1)), label)
    if label.endswith("_miniverse"):
        return (1, 0, label)
    if label.endswith("_prolif"):
        return (2, 0, label)
    return (3, 0, label)


def _aggregate_cluster_means(
    archive: h5py.File, chunk_rows: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    loadings = archive["cell_loadings"]
    metadata = archive["metadata"]
    cluster_codes = metadata["level2_cluster_codes"][:].astype(np.int64)
    cluster_labels = _decode(metadata["level2_cluster_labels"])
    level1_labels = _decode(metadata["level1_labels"])
    cluster_level1_codes = metadata["level2_cluster_level1_codes"][:].astype(int)
    program_indices = metadata["program_indices"][:].astype(int)

    n_cells, n_programs = loadings.shape
    n_clusters = len(cluster_labels)
    if len(cluster_codes) != n_cells:
        raise ValueError("Number of cluster codes does not match loading-matrix rows")
    if len(program_indices) != n_programs:
        raise ValueError("Number of program indices does not match loading-matrix columns")
    if np.any(cluster_codes < 0) or np.any(cluster_codes >= n_clusters):
        raise ValueError("Invalid level2 cluster code in archive")

    sums = np.zeros((n_clusters, n_programs), dtype=np.float64)
    counts = np.bincount(cluster_codes, minlength=n_clusters).astype(np.int64)

    for start in range(0, n_cells, chunk_rows):
        stop = min(start + chunk_rows, n_cells)
        codes = cluster_codes[start:stop]
        values = np.asarray(loadings[start:stop], dtype=np.float64)

        # Grouped reduce is substantially faster than element-wise np.add.at.
        row_order = np.argsort(codes, kind="stable")
        sorted_codes = codes[row_order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
        unique_codes = sorted_codes[starts]
        sums[unique_codes] += np.add.reduceat(values[row_order], starts, axis=0)

        if start == 0 or stop == n_cells or stop % 100_000 < chunk_rows:
            print(f"  aggregated {stop:,}/{n_cells:,} cells", flush=True)

    if np.any(counts == 0):
        empty = cluster_labels[counts == 0].tolist()
        raise ValueError(f"Empty level2 clusters in archive: {empty}")

    means = sums / counts[:, None]
    program_labels = [f"GP{index}" for index in program_indices]
    mean_df = pd.DataFrame(means, index=cluster_labels, columns=program_labels)
    mean_df.index.name = "level2_cluster"

    cluster_info = pd.DataFrame(
        {
            "level2_cluster": cluster_labels,
            "level1": level1_labels[cluster_level1_codes],
            "n_cells": counts,
        }
    ).set_index("level2_cluster")
    return mean_df, cluster_info


def _order_clusters(cluster_info: pd.DataFrame) -> list[str]:
    level1_rank = {label: rank for rank, label in enumerate(LEVEL1_ORDER)}

    def key(cluster: str) -> tuple[int, tuple[int, int, str]]:
        lineage = cluster_info.loc[cluster, "level1"]
        return (level1_rank.get(lineage, len(LEVEL1_ORDER)), _cluster_natural_key(cluster))

    return sorted(cluster_info.index.tolist(), key=key)


def _order_programs(mean_df: pd.DataFrame) -> list[int]:
    profiles = mean_df.to_numpy().T
    norms = np.linalg.norm(profiles, axis=1)
    informative = np.flatnonzero(norms > np.finfo(float).eps)
    uninformative = np.flatnonzero(norms <= np.finfo(float).eps)

    if len(informative) >= 2:
        normalized = profiles[informative] / norms[informative, None]
        tree = linkage(
            normalized,
            method="average",
            metric="cosine",
            optimal_ordering=True,
        )
        informative = informative[leaves_list(tree)]

    return np.r_[informative, uninformative].astype(int).tolist()


def _group_spans(lineages: list[str]) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    start = 0
    for position in range(1, len(lineages) + 1):
        if position == len(lineages) or lineages[position] != lineages[start]:
            spans.append((lineages[start], start, position))
            start = position
    return spans


def _plot_heatmap(
    plot_values: np.ndarray,
    program_labels: list[str],
    cluster_labels: list[str],
    cluster_lineages: list[str],
    pdf_path: Path,
    png_path: Path,
    color_cap_quantile: float,
) -> float:
    positive_values = plot_values[plot_values > 0]
    if len(positive_values) == 0:
        raise ValueError("All cluster-level mean loadings are zero")
    vmax = float(np.quantile(positive_values, color_cap_quantile))
    vmax = max(vmax, np.finfo(float).eps)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(12.0, 9.2), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.24, 8.0],
        width_ratios=[36, 1.0],
        hspace=0.02,
        wspace=0.20,
    )
    ax_strip = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0])
    ax_colorbar = fig.add_subplot(grid[1, 1])

    lineage_categories = list(dict.fromkeys(cluster_lineages))
    lineage_to_code = {label: index for index, label in enumerate(lineage_categories)}
    strip_codes = np.asarray([[lineage_to_code[label] for label in cluster_lineages]])
    strip_cmap = mcolors.ListedColormap(
        [LEVEL1_COLORS.get(label, "#BDBDBD") for label in lineage_categories]
    )
    ax_strip.imshow(strip_codes, aspect="auto", cmap=strip_cmap, interpolation="none")
    ax_strip.set_xlim(-0.5, len(cluster_labels) - 0.5)
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    spans = _group_spans(cluster_lineages)
    for lineage, start, stop in spans:
        ax_strip.text(
            (start + stop - 1) / 2,
            -0.6,
            lineage,
            ha="center",
            va="bottom",
            fontsize=7,
            clip_on=False,
        )
        if start > 0:
            ax_strip.axvline(start - 0.5, color="white", linewidth=1.0)

    image = ax.imshow(
        plot_values,
        aspect="auto",
        interpolation="none",
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        rasterized=True,
    )
    for _, start, _ in spans[1:]:
        ax.axvline(start - 0.5, color="#4D4D4D", linewidth=0.45)

    ax.set_xticks(np.arange(len(cluster_labels)))
    ax.set_xticklabels(cluster_labels, rotation=90, ha="center", va="top", fontsize=4.2)
    y_tick_positions = np.arange(0, len(program_labels), 8)
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels([program_labels[i] for i in y_tick_positions], fontsize=5.2)
    ax.tick_params(axis="x", length=1.5, width=0.4, pad=1)
    ax.tick_params(axis="y", length=1.5, width=0.4, pad=1)
    ax.set_xlabel("Level2 cluster", fontsize=9, labelpad=7)
    ax.set_ylabel("RQVI gene programs (hierarchically ordered)", fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.5)

    colorbar = fig.colorbar(image, cax=ax_colorbar, extend="max")
    colorbar.set_label("Mean cell loading", fontsize=8)
    colorbar.ax.tick_params(labelsize=7, length=2)
    colorbar.outline.set_linewidth(0.5)
    colorbar.ax.text(
        0.5,
        1.045,
        f"cap={vmax:.3g}",
        transform=colorbar.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6,
    )

    fig.suptitle(
        "Mean RQVI GP loading by level2 cluster (seed 0)",
        fontsize=10,
        y=0.985,
    )
    fig.subplots_adjust(left=0.08, right=0.94, top=0.93, bottom=0.22)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return vmax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--means-output", type=Path, default=DEFAULT_MEANS)
    parser.add_argument("--gp-order-output", type=Path, default=DEFAULT_GP_ORDER)
    parser.add_argument("--cluster-order-output", type=Path, default=DEFAULT_CLUSTER_ORDER)
    parser.add_argument("--pdf-output", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png-output", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--chunk-rows", type=int, default=8192)
    parser.add_argument(
        "--color-cap-quantile",
        type=float,
        default=0.995,
        help="Upper quantile used only for the color scale (default: 0.995).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Input archive not found: {args.input}\n"
            "Generate it with scripts/export_rqvi_cell_loadings.py."
        )
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")
    if not 0 < args.color_cap_quantile <= 1:
        raise ValueError("--color-cap-quantile must be in (0, 1]")

    print(f"Aggregating exact cell loadings from {args.input}")
    with h5py.File(args.input, "r") as archive:
        mean_df, cluster_info = _aggregate_cluster_means(archive, args.chunk_rows)

    ordered_clusters = _order_clusters(cluster_info)
    mean_df = mean_df.loc[ordered_clusters]
    cluster_info = cluster_info.loc[ordered_clusters].copy()
    program_row_order = _order_programs(mean_df)

    program_labels = mean_df.columns.to_numpy()[program_row_order].tolist()
    plot_values = mean_df.to_numpy().T[program_row_order]
    cluster_lineages = cluster_info["level1"].tolist()

    args.means_output.parent.mkdir(parents=True, exist_ok=True)
    mean_df.to_csv(args.means_output, float_format="%.17g")

    gp_order = pd.DataFrame(
        {
            "display_row": np.arange(len(program_row_order)),
            "program_index": [int(label.removeprefix("GP")) for label in program_labels],
            "program_label": program_labels,
            "max_mean_loading": plot_values.max(axis=1),
            "dominant_level2_cluster": [
                ordered_clusters[index] for index in np.argmax(plot_values, axis=1)
            ],
        }
    )
    gp_order.to_csv(args.gp_order_output, index=False, float_format="%.17g")

    cluster_info.insert(0, "display_column", np.arange(len(cluster_info)))
    cluster_info.reset_index().to_csv(args.cluster_order_output, index=False)

    vmax = _plot_heatmap(
        plot_values=plot_values,
        program_labels=program_labels,
        cluster_labels=ordered_clusters,
        cluster_lineages=cluster_lineages,
        pdf_path=args.pdf_output,
        png_path=args.png_output,
        color_cap_quantile=args.color_cap_quantile,
    )

    print(f"Saved raw cluster means: {args.means_output}")
    print(f"Saved GP display order: {args.gp_order_output}")
    print(f"Saved cluster display order: {args.cluster_order_output}")
    print(f"Saved heatmap PDF: {args.pdf_output}")
    print(f"Saved heatmap PNG: {args.png_output}")
    print(
        f"Matrix: {plot_values.shape[0]} GPs x {plot_values.shape[1]} clusters; "
        f"color cap={vmax:.6g}; raw values retained in CSV"
    )


if __name__ == "__main__":
    main()
