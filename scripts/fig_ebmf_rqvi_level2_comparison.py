"""Plot corresponding EBMF and RQVI factor profiles across level2 clusters.

By default, the script reads the two saved 200 x 114 display matrices and only
renders the figure. Pass --recompute-matches to rebuild the one-to-one matching
and the display matrices from the repository's cluster-mean inputs before
plotting.

The 256 RQVI programs from each of 10 random seeds are treated as 2,560
seed-specific candidates.  A maximum-weight bipartite assignment then pairs
all 200 EBMF factors with 200 distinct RQVI candidates, using signed Pearson
correlation between their mean-loading profiles across the same 114 level2
clusters.  The displayed RQVI heatmap contains exactly those 200 matches.

Values are z-scored within each program across clusters for matching. For the
heatmaps, every program is scaled to the interval [0, 1] across clusters so the
display follows the loading-matrix style of Figure 1C. Raw arithmetic cluster
means are stored separately in the repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linear_sum_assignment


# All default inputs and outputs are anchored to the cloned repository root.
# No default path depends on a user name, home directory, or current directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EBMF_MEANS = REPO_ROOT / "data" / "ebmf_mean_loadings_by_level2_cluster.csv"
DEFAULT_CLUSTER_ORDER = REPO_ROOT / "data" / "rqvi_seed0_level2_heatmap_cluster_order.csv"
DEFAULT_MATCHES = (
    REPO_ROOT / "data" / "ebmf_rqvi_multiseed_level2_one_to_one_matches.csv"
)
DEFAULT_SEED_SUMMARY = (
    REPO_ROOT / "data" / "ebmf_rqvi_multiseed_one_to_one_seed_summary.csv"
)
DEFAULT_POOLED_RQVI_MEANS = (
    REPO_ROOT / "data" / "rqvi_multiseed_mean_loadings_by_level2_cluster.csv"
)
DEFAULT_RQVI_CANDIDATE_METADATA = (
    REPO_ROOT / "data" / "rqvi_multiseed_candidate_metadata.csv"
)
DEFAULT_EBMF_SCALED = (
    REPO_ROOT / "data" / "ebmf_level2_scaled_loadings_for_comparison.csv"
)
DEFAULT_MATCHED_RQVI_SCALED = (
    REPO_ROOT
    / "data"
    / "matched_rqvi_multiseed_level2_scaled_loadings_for_comparison.csv"
)
DEFAULT_PDF = (
    REPO_ROOT / "figures" / "sup_figures" / "ebmf_rqvi_level2_comparison.pdf"
)
DEFAULT_PNG = (
    REPO_ROOT / "figures" / "sup_figures" / "ebmf_rqvi_level2_comparison.png"
)
DEFAULT_SUBFIGURE_DIR = (
    REPO_ROOT
    / "figures"
    / "sup_figures"
    / "ebmf_rqvi_level2_comparison_subfigures"
)

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

def _zscore_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    values = df.to_numpy(dtype=np.float64)
    means = values.mean(axis=0, keepdims=True)
    stds = values.std(axis=0, ddof=0, keepdims=True)
    informative = stds.ravel() > np.finfo(float).eps
    z = np.zeros_like(values)
    z[:, informative] = (values[:, informative] - means[:, informative]) / stds[:, informative]
    return pd.DataFrame(z, index=df.index, columns=df.columns), informative


def _scale_columns_to_unit_interval(df: pd.DataFrame) -> pd.DataFrame:
    """Scale each program across clusters without changing Pearson r."""
    values = df.to_numpy(dtype=np.float64)
    minima = values.min(axis=0, keepdims=True)
    ranges = values.max(axis=0, keepdims=True) - minima
    informative = ranges.ravel() > np.finfo(float).eps
    scaled = np.zeros_like(values)
    scaled[:, informative] = (
        values[:, informative] - minima[:, informative]
    ) / ranges[:, informative]
    return pd.DataFrame(scaled, index=df.index, columns=df.columns)


def _group_spans(lineages: list[str]) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    start = 0
    for position in range(1, len(lineages) + 1):
        if position == len(lineages) or lineages[position] != lineages[start]:
            spans.append((lineages[start], start, position))
            start = position
    return spans


def _draw_lineage_strip(
    ax: plt.Axes,
    cluster_lineages: list[str],
    spans: list[tuple[str, int, int]],
) -> None:
    categories = list(dict.fromkeys(cluster_lineages))
    category_to_code = {label: index for index, label in enumerate(categories)}
    codes = np.asarray([[category_to_code[label] for label in cluster_lineages]])
    cmap = mcolors.ListedColormap(
        [LEVEL1_COLORS.get(label, "#BDBDBD") for label in categories]
    )
    ax.imshow(codes, aspect="auto", cmap=cmap, interpolation="none")
    ax.set_xlim(-0.5, len(cluster_lineages) - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for lineage, start, stop in spans:
        if stop - start >= 4:
            ax.text(
                (start + stop - 1) / 2,
                -0.6,
                lineage,
                ha="center",
                va="bottom",
                fontsize=7.5,
                clip_on=False,
            )
        if start > 0:
            ax.axvline(start - 0.5, color="white", linewidth=1.0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _style_heatmap(
    ax: plt.Axes,
    spans: list[tuple[str, int, int]],
) -> None:
    for _, start, _ in spans[1:]:
        ax.axvline(start - 0.5, color="#777777", linewidth=0.35)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.55)


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def _plot_heatmap_subfigure(
    matrix: np.ndarray,
    cluster_lineages: list[str],
    ylabel: str,
    ylabel_on_right: bool,
    output_pdf: Path,
) -> None:
    """Export one heatmap as a standalone manuscript subfigure."""
    _set_plot_style()
    cmap = plt.get_cmap("Blues")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    spans = _group_spans(cluster_lineages)

    fig = plt.figure(figsize=(5.1, 7.8), facecolor="white")
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.18, 7.6],
        hspace=0.02,
    )
    ax_strip = fig.add_subplot(grid[0, 0])
    ax_heatmap = fig.add_subplot(grid[1, 0])

    _draw_lineage_strip(ax_strip, cluster_lineages, spans)
    ax_heatmap.imshow(
        matrix,
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    _style_heatmap(ax_heatmap, spans)
    ax_heatmap.set_ylabel(ylabel, fontsize=10, labelpad=8)
    if ylabel_on_right:
        ax_heatmap.yaxis.set_label_position("right")

    if ylabel_on_right:
        fig.subplots_adjust(left=0.04, right=0.86, top=0.95, bottom=0.04)
    else:
        fig.subplots_adjust(left=0.14, right=0.96, top=0.95, bottom=0.04)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_shared_colorbar(output_pdf: Path) -> None:
    """Export the common loading scale for use below the two subfigures."""
    _set_plot_style()
    cmap = plt.get_cmap("Blues")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = plt.figure(figsize=(2.1, 0.48), facecolor="white")
    colorbar_axis = fig.add_axes([0.08, 0.54, 0.84, 0.27])
    colorbar = fig.colorbar(
        scalar_mappable,
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=[0.0, 0.5, 1.0],
    )
    colorbar.set_label("Relative loading", fontsize=8, labelpad=2)
    colorbar.ax.tick_params(labelsize=7, length=2, pad=1)
    colorbar.outline.set_linewidth(0.45)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _plot_subfigures(
    ebmf_plot: np.ndarray,
    rqvi_plot: np.ndarray,
    cluster_lineages: list[str],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    ebmf_pdf = output_dir / "panel_A_ebmf_factors.pdf"
    rqvi_pdf = output_dir / "panel_B_corresponding_rqvi_factors.pdf"
    colorbar_pdf = output_dir / "shared_relative_loading_colorbar.pdf"
    _plot_heatmap_subfigure(
        matrix=ebmf_plot,
        cluster_lineages=cluster_lineages,
        ylabel="EBMF factors",
        ylabel_on_right=False,
        output_pdf=ebmf_pdf,
    )
    _plot_heatmap_subfigure(
        matrix=rqvi_plot,
        cluster_lineages=cluster_lineages,
        ylabel="Corresponding RQVI factors",
        ylabel_on_right=True,
        output_pdf=rqvi_pdf,
    )
    _plot_shared_colorbar(colorbar_pdf)
    return ebmf_pdf, rqvi_pdf, colorbar_pdf


def _plot(
    ebmf_plot: np.ndarray,
    rqvi_plot: np.ndarray,
    cluster_lineages: list[str],
    output_pdf: Path,
    output_png: Path,
) -> None:
    _set_plot_style()
    cmap = plt.get_cmap("Blues")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(10.8, 7.8), facecolor="white")
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.18, 7.6],
        width_ratios=[1.0, 0.14, 1.0],
        hspace=0.02,
        wspace=0.0,
    )
    ax_strip_ebmf = fig.add_subplot(grid[0, 0])
    ax_ebmf = fig.add_subplot(grid[1, 0])
    ax_strip_rqvi = fig.add_subplot(grid[0, 2])
    ax_rqvi = fig.add_subplot(grid[1, 2])

    spans = _group_spans(cluster_lineages)
    _draw_lineage_strip(ax_strip_ebmf, cluster_lineages, spans)
    _draw_lineage_strip(ax_strip_rqvi, cluster_lineages, spans)

    ax_ebmf.imshow(
        ebmf_plot,
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    ax_rqvi.imshow(
        rqvi_plot,
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    _style_heatmap(ax_ebmf, spans)
    _style_heatmap(ax_rqvi, spans)
    ax_ebmf.set_ylabel("EBMF factors", fontsize=10, labelpad=8)
    ax_rqvi.set_ylabel("Corresponding RQVI factors", fontsize=10, labelpad=8)
    ax_rqvi.yaxis.set_label_position("right")

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar_axis = fig.add_axes([0.80, 0.055, 0.12, 0.016])
    colorbar = fig.colorbar(
        scalar_mappable,
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=[0.0, 0.5, 1.0],
    )
    colorbar.set_label("Relative loading", fontsize=8, labelpad=2)
    colorbar.ax.tick_params(labelsize=7, length=2, pad=1)
    colorbar.outline.set_linewidth(0.45)

    fig.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.09)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ebmf-means", type=Path, default=DEFAULT_EBMF_MEANS)
    parser.add_argument(
        "--rqvi-means",
        type=Path,
        default=DEFAULT_POOLED_RQVI_MEANS,
        help="Pooled RQVI cluster-mean CSV from all model runs.",
    )
    parser.add_argument("--cluster-order", type=Path, default=DEFAULT_CLUSTER_ORDER)
    parser.add_argument("--matches-output", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--seed-summary-output", type=Path, default=DEFAULT_SEED_SUMMARY)
    parser.add_argument(
        "--rqvi-candidate-metadata",
        type=Path,
        default=DEFAULT_RQVI_CANDIDATE_METADATA,
    )
    parser.add_argument(
        "--ebmf-scaled-output", type=Path, default=DEFAULT_EBMF_SCALED
    )
    parser.add_argument(
        "--matched-rqvi-scaled-output",
        type=Path,
        default=DEFAULT_MATCHED_RQVI_SCALED,
    )
    parser.add_argument("--pdf-output", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png-output", type=Path, default=DEFAULT_PNG)
    parser.add_argument(
        "--subfigure-dir",
        type=Path,
        default=DEFAULT_SUBFIGURE_DIR,
        help="Directory for standalone EBMF, RQVI, and shared-colorbar PDFs.",
    )
    parser.add_argument(
        "--recompute-matches",
        action="store_true",
        help="Rebuild matching and display matrices before plotting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.cluster_order.exists():
        raise FileNotFoundError(args.cluster_order)
    cluster_info = pd.read_csv(args.cluster_order)
    required_columns = {"level2_cluster", "level1", "display_column"}
    if not required_columns.issubset(cluster_info.columns):
        missing = required_columns - set(cluster_info.columns)
        raise ValueError(f"Cluster-order file is missing {missing}")
    cluster_info = cluster_info.sort_values("display_column")
    cluster_labels = cluster_info["level2_cluster"].astype(str).tolist()
    cluster_lineages = cluster_info["level1"].astype(str).tolist()

    if not args.recompute_matches:
        for display_path in (
            args.ebmf_scaled_output,
            args.matched_rqvi_scaled_output,
        ):
            if not display_path.exists():
                raise FileNotFoundError(
                    f"Saved display matrix not found: {display_path}. "
                    "Run with --recompute-matches to rebuild it."
                )

        ebmf_display = pd.read_csv(args.ebmf_scaled_output, index_col=0)
        rqvi_display = pd.read_csv(args.matched_rqvi_scaled_output, index_col=0)
        expected_shape = (200, len(cluster_labels))
        if ebmf_display.shape != expected_shape or rqvi_display.shape != expected_shape:
            raise ValueError(
                "Saved display matrices must both have shape "
                f"{expected_shape}; got {ebmf_display.shape} and {rqvi_display.shape}"
            )
        if (
            ebmf_display.columns.astype(str).tolist() != cluster_labels
            or rqvi_display.columns.astype(str).tolist() != cluster_labels
        ):
            raise ValueError(
                "Saved display matrices do not use the current cluster display order"
            )
        for name, display in (
            ("EBMF", ebmf_display),
            ("RQVI", rqvi_display),
        ):
            values = display.to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} display matrix contains non-finite values")
            if values.min() < -1e-12 or values.max() > 1 + 1e-12:
                raise ValueError(f"{name} display matrix is outside the [0, 1] range")

        _plot(
            ebmf_plot=ebmf_display.to_numpy(dtype=np.float64),
            rqvi_plot=rqvi_display.to_numpy(dtype=np.float64),
            cluster_lineages=cluster_lineages,
            output_pdf=args.pdf_output,
            output_png=args.png_output,
        )
        subfigure_paths = _plot_subfigures(
            ebmf_plot=ebmf_display.to_numpy(dtype=np.float64),
            rqvi_plot=rqvi_display.to_numpy(dtype=np.float64),
            cluster_lineages=cluster_lineages,
            output_dir=args.subfigure_dir,
        )
        print(f"Loaded saved EBMF display matrix: {args.ebmf_scaled_output}")
        print(f"Loaded saved RQVI display matrix: {args.matched_rqvi_scaled_output}")
        print(f"Saved comparison PDF: {args.pdf_output}")
        print(f"Saved comparison PNG: {args.png_output}")
        print(f"Saved subfigure PDFs: {', '.join(map(str, subfigure_paths))}")
        return

    for input_path in (
        args.ebmf_means,
        args.rqvi_means,
        args.rqvi_candidate_metadata,
    ):
        if not input_path.exists():
            raise FileNotFoundError(input_path)

    ebmf_raw = pd.read_csv(args.ebmf_means, index_col="level2_cluster")
    if set(cluster_labels) != set(ebmf_raw.index.astype(str)):
        raise ValueError("EBMF and display-order files do not contain the same clusters")
    ebmf_raw.index = ebmf_raw.index.astype(str)
    ebmf_raw = ebmf_raw.loc[cluster_labels]
    expected_ebmf = [f"F{index}" for index in range(1, 201)]
    if ebmf_raw.columns.astype(str).tolist() != expected_ebmf:
        raise ValueError("Expected EBMF columns F1 through F200")

    ebmf_z, ebmf_informative = _zscore_columns(ebmf_raw)
    if not np.all(ebmf_informative):
        bad = ebmf_raw.columns[~ebmf_informative].tolist()
        raise ValueError(f"EBMF factors with constant cluster profile: {bad}")

    rqvi_raw = pd.read_csv(args.rqvi_means, index_col="level2_cluster")
    rqvi_raw.index = rqvi_raw.index.astype(str)
    if set(cluster_labels) != set(rqvi_raw.index):
        raise ValueError("RQVI means and display-order files contain different clusters")
    rqvi_raw = rqvi_raw.loc[cluster_labels]
    rqvi_raw.columns = rqvi_raw.columns.astype(str)

    candidate_info = pd.read_csv(args.rqvi_candidate_metadata)
    required_metadata = {"rqvi_candidate", "rqvi_seed", "rqvi_gp", "informative"}
    if not required_metadata.issubset(candidate_info.columns):
        missing = required_metadata - set(candidate_info.columns)
        raise ValueError(f"RQVI candidate metadata is missing {missing}")
    candidate_info["rqvi_candidate"] = candidate_info["rqvi_candidate"].astype(str)
    if candidate_info["rqvi_candidate"].duplicated().any():
        raise ValueError("RQVI candidate metadata contains duplicate candidate labels")
    candidate_info = candidate_info.set_index("rqvi_candidate")
    if not rqvi_raw.columns.is_unique:
        raise ValueError("Pooled seed-specific RQVI candidate labels are not unique")
    if set(rqvi_raw.columns) != set(candidate_info.index):
        raise ValueError("RQVI means and candidate metadata contain different candidates")
    candidate_info = candidate_info.loc[rqvi_raw.columns]
    rqvi_z, informative = _zscore_columns(rqvi_raw)
    metadata_informative = (
        candidate_info["informative"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )
    if metadata_informative.isna().any():
        raise ValueError("RQVI candidate metadata has invalid informative values")
    if not np.array_equal(metadata_informative.to_numpy(bool), informative):
        raise ValueError("RQVI informative flags do not match the pooled mean matrix")
    if rqvi_raw.shape[1] < ebmf_raw.shape[1]:
        raise ValueError("Fewer RQVI candidates than EBMF factors")

    n_clusters = len(cluster_labels)
    correlation_matrix = ebmf_z.to_numpy().T @ rqvi_z.to_numpy() / n_clusters
    correlation_matrix[:, ~informative] = -np.inf

    finite_for_assignment = np.where(np.isfinite(correlation_matrix), correlation_matrix, -1e9)
    assigned_rows, assigned_columns = linear_sum_assignment(-finite_for_assignment)
    if len(assigned_rows) != ebmf_raw.shape[1]:
        raise RuntimeError("One-to-one assignment did not cover every EBMF factor")
    selected_positions = np.full(ebmf_raw.shape[1], -1, dtype=int)
    selected_positions[assigned_rows] = assigned_columns
    if np.any(selected_positions < 0) or len(np.unique(selected_positions)) != len(selected_positions):
        raise RuntimeError("Assignment is incomplete or reuses an RQVI candidate")

    factor_positions = np.arange(ebmf_raw.shape[1])
    assigned_correlations = correlation_matrix[factor_positions, selected_positions]
    unconstrained_positions = np.argmax(correlation_matrix, axis=1)
    unconstrained_correlations = correlation_matrix[factor_positions, unconstrained_positions]

    ebmf_profiles = ebmf_z.to_numpy().T
    tree = linkage(
        ebmf_profiles,
        method="average",
        metric="correlation",
        optimal_ordering=True,
    )
    display_order = leaves_list(tree)

    ebmf_factors = ebmf_raw.columns.to_numpy()
    rqvi_candidates = rqvi_raw.columns.to_numpy()
    selected_candidates = rqvi_candidates[selected_positions]
    ordered_ebmf = ebmf_factors[display_order].tolist()
    ordered_candidates = selected_candidates[display_order].tolist()
    ordered_correlations = assigned_correlations[display_order]
    ordered_selected_positions = selected_positions[display_order]

    ebmf_scaled = _scale_columns_to_unit_interval(ebmf_raw)
    rqvi_scaled = _scale_columns_to_unit_interval(rqvi_raw)
    ebmf_plot = ebmf_scaled.to_numpy().T[display_order]
    rqvi_profiles = rqvi_scaled.to_numpy().T
    rqvi_plot = rqvi_profiles[ordered_selected_positions]

    dominant_ebmf = ebmf_raw.idxmax(axis=0).to_numpy()
    dominant_rqvi = rqvi_raw.idxmax(axis=0).to_numpy()
    unconstrained_candidates = rqvi_candidates[unconstrained_positions]
    ordered_selected_info = candidate_info.loc[ordered_candidates]
    matches = pd.DataFrame(
        {
            "display_row": np.arange(len(display_order)),
            "ebmf_factor": ordered_ebmf,
            "rqvi_seed": ordered_selected_info["rqvi_seed"].to_numpy(dtype=int),
            "rqvi_gp": ordered_selected_info["rqvi_gp"].to_numpy(),
            "rqvi_candidate": ordered_candidates,
            "pearson_r_level2_one_to_one": ordered_correlations,
            "unconstrained_best_candidate": unconstrained_candidates[display_order],
            "unconstrained_best_r_level2": unconstrained_correlations[display_order],
            "one_to_one_r_penalty": (
                unconstrained_correlations[display_order] - ordered_correlations
            ),
            "ebmf_dominant_level2_cluster": dominant_ebmf[display_order],
            "rqvi_dominant_level2_cluster": dominant_rqvi[ordered_selected_positions],
        }
    )

    seed_values = sorted(candidate_info["rqvi_seed"].astype(int).unique())
    seed_summary = (
        matches.groupby("rqvi_seed", sort=True)["pearson_r_level2_one_to_one"]
        .agg(selected_matches="size", median_r="median", mean_r="mean", min_r="min", max_r="max")
        .reindex(seed_values, fill_value=0)
        .reset_index()
    )

    for output in (
        args.matches_output,
        args.seed_summary_output,
        args.ebmf_scaled_output,
        args.matched_rqvi_scaled_output,
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.matches_output, index=False, float_format="%.17g")
    seed_summary.to_csv(args.seed_summary_output, index=False, float_format="%.17g")

    ebmf_scaled_display = pd.DataFrame(
        ebmf_plot,
        index=ordered_ebmf,
        columns=cluster_labels,
    )
    ebmf_scaled_display.index.name = "ebmf_factor"
    ebmf_scaled_display.to_csv(args.ebmf_scaled_output, float_format="%.17g")

    matched_rqvi_scaled_display = pd.DataFrame(
        rqvi_plot,
        index=[
            f"{ebmf}->{candidate}"
            for ebmf, candidate in zip(ordered_ebmf, ordered_candidates)
        ],
        columns=cluster_labels,
    )
    matched_rqvi_scaled_display.index.name = "matched_pair"
    matched_rqvi_scaled_display.to_csv(
        args.matched_rqvi_scaled_output, float_format="%.17g"
    )

    _plot(
        ebmf_plot=ebmf_plot,
        rqvi_plot=rqvi_plot,
        cluster_lineages=cluster_lineages,
        output_pdf=args.pdf_output,
        output_png=args.png_output,
    )
    subfigure_paths = _plot_subfigures(
        ebmf_plot=ebmf_plot,
        rqvi_plot=rqvi_plot,
        cluster_lineages=cluster_lineages,
        output_dir=args.subfigure_dir,
    )

    print(f"Saved one-to-one match table: {args.matches_output}")
    print(f"Saved selected-seed summary: {args.seed_summary_output}")
    print(f"Loaded pooled raw RQVI cluster means: {args.rqvi_means}")
    print(f"Loaded RQVI candidate metadata: {args.rqvi_candidate_metadata}")
    print(f"Saved displayed EBMF scaled loadings: {args.ebmf_scaled_output}")
    print(
        "Saved displayed matched-RQVI scaled loadings: "
        f"{args.matched_rqvi_scaled_output}"
    )
    print(f"Saved comparison PDF: {args.pdf_output}")
    print(f"Saved comparison PNG: {args.png_output}")
    print(f"Saved subfigure PDFs: {', '.join(map(str, subfigure_paths))}")
    print(
        f"Pooled {len(seed_values)}-run one-to-one matches: "
        f"median r={np.median(assigned_correlations):.6f}; "
        f"mean r={np.mean(assigned_correlations):.6f}; "
        f"range={np.min(assigned_correlations):.6f}–{np.max(assigned_correlations):.6f}; "
        f"r>=0.5={np.mean(assigned_correlations >= 0.5) * 100:.1f}%; "
        f"distinct candidates={len(np.unique(selected_positions))}"
    )
    print(
        f"Eligible RQVI candidates={informative.sum()}/{len(informative)}; "
        f"excluded constant profiles={(~informative).sum()}"
    )
    print(
        "One-to-one cost relative to unconstrained pooled best matches: "
        f"mean delta r={np.mean(unconstrained_correlations - assigned_correlations):.6f}; "
        f"unchanged={np.mean(np.isclose(unconstrained_correlations, assigned_correlations)) * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
