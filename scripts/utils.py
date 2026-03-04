"""
Shared utilities for GP overview figures (v2).
"""

import numpy as np
import pandas as pd
import scanpy as sc
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
