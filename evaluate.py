"""
treeflow/evaluate.py

Evaluate generated tree point clouds against real trees using:
- Per-pair metrics: CD, 2D histogram JSD, crown profile MAE (p50/p75/p98)
- Baselines: intra-class (same species + scan type + height bin) and inter-class
- Stratified 3D voxel JSD: per species, scan type, height bin
- Breakdowns by species, height, scan type, and CFG scale
"""

import json
import numpy as np
import pandas as pd
import argparse
import zarr
from pathlib import Path
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import multiprocessing as mp
import time


# =============================================================================
# Constants
# =============================================================================

HEIGHT_BIN_EDGES = [0, 5, 10, 15, 20, 25, 30, 35, 40, float("inf")]
HEIGHT_BIN_LABELS = [
    "0-5",
    "5-10",
    "10-15",
    "15-20",
    "20-25",
    "25-30",
    "30-35",
    "35-40",
    "40+",
]

PAIR_METRICS = [
    "cd",
    "histogram_jsd",
    "crown_mae_p50",
    "crown_mae_p75",
    "crown_mae_p98",
]


def get_height_bin(h: float) -> str:
    """Assign a height value to its 5m bin label."""
    for i, (lo, hi) in enumerate(zip(HEIGHT_BIN_EDGES[:-1], HEIGHT_BIN_EDGES[1:])):
        if lo <= h < hi:
            return HEIGHT_BIN_LABELS[i]
    return HEIGHT_BIN_LABELS[-1]


# =============================================================================
# Distance Functions
# =============================================================================


def chamfer_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Compute Chamfer Distance between two point clouds.

    Uses squared L2 distance (standard formulation).
    Both clouds should be (N, 3).
    """
    dist_matrix = cdist(p1, p2, metric="sqeuclidean")
    min_p1_to_p2 = dist_matrix.min(axis=1).mean()
    min_p2_to_p1 = dist_matrix.min(axis=0).mean()
    return float((min_p1_to_p2 + min_p2_to_p1) / 2)


def earth_movers_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Compute Earth Mover's Distance (optimal bipartite matching) between two
    point clouds of equal size using L2 cost.

    Both clouds must be (N, 3) with the same N.
    """
    dist_matrix = cdist(p1, p2, metric="euclidean")
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    return float(dist_matrix[row_ind, col_ind].mean())


# =============================================================================
# Shape Metrics (2D Histogram JSD, Crown Profile)
# =============================================================================


def compute_rz(cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert (N,3) point cloud to (r, z) in trunk-aligned cylindrical coordinates.

    Uses SVD to find the principal axis (direction of maximum variance),
    which corresponds to the main growth direction. r is perpendicular
    distance from that axis, z is projection along it.
    """
    centroid = cloud.mean(axis=0)
    centered = cloud - centroid

    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    axis = Vt[0]

    # Ensure axis points "up" (positive z component)
    if axis[2] < 0:
        axis = -axis

    z = centered @ axis
    r = np.linalg.norm(centered - np.outer(z, axis), axis=1)
    return r, z


def compute_bin_edges(
    r_arrays: list[np.ndarray],
    z_arrays: list[np.ndarray],
    n_radial: int,
    n_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute shared radial and height bin edges from multiple r,z arrays."""
    all_r = np.concatenate(r_arrays)
    all_z = np.concatenate(z_arrays)
    eps = 1e-6
    radial_edges = np.linspace(0, all_r.max() + eps, n_radial + 1)
    height_edges = np.linspace(all_z.min() - eps, all_z.max() + eps, n_height + 1)
    return radial_edges, height_edges


def compute_2d_histogram_jsd(
    rz_a: tuple[np.ndarray, np.ndarray],
    rz_b: tuple[np.ndarray, np.ndarray],
    radial_edges: np.ndarray,
    height_edges: np.ndarray,
) -> float:
    """
    Compute JSD between 2D (r, z) histograms of two point clouds.

    Uses Laplace smoothing (add 1 to all bins) for numerical stability.
    Returns JSD in [0, ln(2)].
    """
    r_a, z_a = rz_a
    r_b, z_b = rz_b

    hist_a, _, _ = np.histogram2d(r_a, z_a, bins=[radial_edges, height_edges])
    hist_b, _, _ = np.histogram2d(r_b, z_b, bins=[radial_edges, height_edges])

    # Laplace smoothing
    hist_a = hist_a + 1.0
    hist_b = hist_b + 1.0

    p = (hist_a / hist_a.sum()).flatten()
    q = (hist_b / hist_b.sum()).flatten()

    m = 0.5 * (p + q)
    jsd = float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))
    return jsd


def compute_crown_profile(
    r: np.ndarray, z: np.ndarray, height_edges: np.ndarray, min_points: int = 5
) -> dict:
    """
    Compute crown radial profile: 50th/75th/98th percentile of r per height bin.

    Bins with < min_points are set to NaN.
    """
    n_bins = len(height_edges) - 1
    p50 = np.full(n_bins, np.nan)
    p75 = np.full(n_bins, np.nan)
    p98 = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (z >= height_edges[i]) & (z < height_edges[i + 1])
        counts[i] = mask.sum()
        if counts[i] >= min_points:
            r_bin = r[mask]
            p50[i] = np.percentile(r_bin, 50)
            p75[i] = np.percentile(r_bin, 75)
            p98[i] = np.percentile(r_bin, 98)

    return {"p50": p50, "p75": p75, "p98": p98, "counts": counts}


def crown_profile_mae(profile_a: dict, profile_b: dict) -> dict[str, float]:
    """
    MAE between two crown profiles, excluding bins where either has < min_points.
    """
    results = {}
    for key in ["p50", "p75", "p98"]:
        valid = ~np.isnan(profile_a[key]) & ~np.isnan(profile_b[key])
        if valid.sum() > 0:
            results[key] = float(
                np.mean(np.abs(profile_a[key][valid] - profile_b[key][valid]))
            )
        else:
            results[key] = float("nan")
    return results


def compute_shape_metrics(
    rz_a: tuple[np.ndarray, np.ndarray],
    rz_b: tuple[np.ndarray, np.ndarray],
    radial_edges: np.ndarray,
    height_edges: np.ndarray,
    profile_a: dict | None = None,
) -> tuple[float, float, float, float]:
    """
    Compute all shape metrics between two clouds given precomputed (r,z) and edges.

    Returns (histogram_jsd, crown_mae_p50, crown_mae_p75, crown_mae_p98).
    """
    hjsd = compute_2d_histogram_jsd(rz_a, rz_b, radial_edges, height_edges)

    if profile_a is None:
        profile_a = compute_crown_profile(rz_a[0], rz_a[1], height_edges)
    profile_b = compute_crown_profile(rz_b[0], rz_b[1], height_edges)
    mae = crown_profile_mae(profile_a, profile_b)

    return hjsd, mae["p50"], mae["p75"], mae["p98"]


# =============================================================================
# Data Loading
# =============================================================================


def normalize_tree_id(tree_id) -> str:
    """Normalize tree ID to 5-digit zero-padded string format."""
    return str(tree_id).zfill(5)


def load_generated_metadata(experiment_dir: Path) -> pd.DataFrame:
    """
    Load generated samples metadata CSV from experiment directory.

    Returns DataFrame with normalized source_tree_id.
    """
    samples_dir = experiment_dir / "samples"
    csv_path = samples_dir / "samples_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {csv_path}\n" "Run postprocess_samples.py first."
        )

    df = pd.read_csv(csv_path)
    df["source_tree_id"] = df["source_tree_id"].apply(normalize_tree_id)
    print(f"Loaded metadata for {len(df)} generated samples")
    print(f"  Unique source trees: {df['source_tree_id'].nunique()}")
    print(f"  Species: {df['species'].nunique()}")
    if "cfg_scale" in df.columns:
        print(
            f"  CFG scale range: {df['cfg_scale'].min():.2f} - {df['cfg_scale'].max():.2f}"
        )
    return df


def load_real_metadata(data_path: Path) -> pd.DataFrame:
    """
    Load real tree metadata, filter to test split, derive file_id and file_path.

    Returns DataFrame with columns: file_id, file_path, species, data_type, tree_H, etc.
    """
    data_path = Path(data_path)
    csv_path = data_path / "metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        raise ValueError("CSV missing 'split' column. Run preprocess_laz.py first.")

    # Derive file_id and file_path, filter for existence
    df["file_id"] = df["filename"].apply(lambda x: Path(x).stem)
    df["file_path"] = df["file_id"].apply(lambda x: str(data_path / f"{x}.zarr"))
    df = df[df["file_path"].apply(lambda x: Path(x).exists())]

    # Filter to test split
    test_df = df[df["split"] == "test"].copy()
    print(
        f"Loaded real metadata: {len(test_df)} test trees (of {len(df)} with zarr files)"
    )
    return test_df


def load_point_cloud(
    path: str, max_points: int | None = None, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Load a point cloud from a zarr file, optionally downsampling."""
    points = zarr.load(path).astype(np.float32)
    if max_points is not None and len(points) > max_points:
        if rng is None:
            rng = np.random.default_rng(42)
        indices = rng.choice(len(points), size=max_points, replace=False)
        points = points[indices]
    return points


# =============================================================================
# Per-Tree Evaluation (Parallelized)
# =============================================================================


def build_tree_tasks(
    gen_metadata: pd.DataFrame,
    real_metadata: pd.DataFrame,
    zarr_dir: Path,
    seed: int,
    max_points: int | None = None,
    histogram_radial_bins: int = 16,
    histogram_height_bins: int = 32,
) -> list[dict]:
    """
    Build task dicts for parallel per-tree evaluation.

    Groups generated samples by source_tree_id and matches to real metadata.
    """
    # Index real metadata by file_id
    real_lookup = {}
    for _, row in real_metadata.iterrows():
        real_lookup[row["file_id"]] = {
            "file_path": row["file_path"],
            "species": row["species"],
            "data_type": row["data_type"],
            "tree_H": row["tree_H"],
        }

    # Group generated samples by source_tree_id
    grouped = gen_metadata.groupby("source_tree_id")

    tasks = []
    skipped_trees = 0
    for tree_id, group in grouped:
        if tree_id not in real_lookup:
            skipped_trees += 1
            continue

        real_info = real_lookup[tree_id]

        gen_samples = []
        for _, row in group.iterrows():
            sample_info = {
                "sample_id": row["sample_id"],
                "zarr_path": str(zarr_dir / f"{row['sample_file']}"),
                "cfg_scale": float(row.get("cfg_scale", 0.0)),
                "species": row["species"],
                "height_m": float(row.get("height_m", real_info["tree_H"])),
                "scan_type": row.get("scan_type", real_info["data_type"]),
            }
            gen_samples.append(sample_info)

        tasks.append(
            {
                "source_tree_id": tree_id,
                "real_path": real_info["file_path"],
                "real_species": real_info["species"],
                "real_data_type": real_info["data_type"],
                "real_height": float(real_info["tree_H"]),
                "gen_samples": gen_samples,
                "seed": seed,
                "max_points": max_points,
                "histogram_radial_bins": histogram_radial_bins,
                "histogram_height_bins": histogram_height_bins,
            }
        )

    if skipped_trees > 0:
        print(f"  Skipped {skipped_trees} trees not found in real test set")
    print(f"  Built {len(tasks)} tree evaluation tasks")
    return tasks


def evaluate_tree_worker(task: dict) -> dict:
    """
    Worker: evaluate all generated samples for one real tree.

    Computes CD, 2D histogram JSD, and crown profile MAE for each
    generated sample against the real tree.
    """
    max_points = task.get("max_points")
    seed = task.get("seed", 42)
    n_radial = task.get("histogram_radial_bins", 16)
    n_height = task.get("histogram_height_bins", 32)
    rng = np.random.default_rng(seed)

    real_cloud = load_point_cloud(task["real_path"], max_points=max_points, rng=rng)

    # Load all generated clouds
    gen_clouds = []
    gen_infos = []
    for gen_info in task["gen_samples"]:
        try:
            gen_cloud = load_point_cloud(gen_info["zarr_path"])
            gen_clouds.append(gen_cloud)
            gen_infos.append(gen_info)
        except Exception:
            continue

    if not gen_clouds:
        return {
            "pairs": [],
            "real_cloud": real_cloud,
            "source_tree_id": task["source_tree_id"],
        }

    # Compute (r, z) for real and all gen
    r_real, z_real = compute_rz(real_cloud)
    gen_rz_list = [compute_rz(gc) for gc in gen_clouds]

    # Shared bin edges for this tree (from real + all gen)
    r_arrays = [r_real] + [rz[0] for rz in gen_rz_list]
    z_arrays = [z_real] + [rz[1] for rz in gen_rz_list]
    radial_edges, height_edges = compute_bin_edges(r_arrays, z_arrays, n_radial, n_height)

    # Precompute real crown profile
    real_rz = (r_real, z_real)
    real_profile = compute_crown_profile(r_real, z_real, height_edges)

    # Evaluate each generated sample
    pairs = []
    for gen_cloud, gen_rz, gen_info in zip(gen_clouds, gen_rz_list, gen_infos):
        cd = chamfer_distance(real_cloud, gen_cloud)
        hjsd, mae_p50, mae_p75, mae_p98 = compute_shape_metrics(
            real_rz, gen_rz, radial_edges, height_edges, profile_a=real_profile
        )

        pairs.append(
            {
                "source_tree_id": task["source_tree_id"],
                "sample_id": gen_info["sample_id"],
                "species": gen_info["species"],
                "height_m": gen_info["height_m"],
                "scan_type": gen_info["scan_type"],
                "cfg_scale": gen_info["cfg_scale"],
                "cd": cd,
                "histogram_jsd": hjsd,
                "crown_mae_p50": mae_p50,
                "crown_mae_p75": mae_p75,
                "crown_mae_p98": mae_p98,
            }
        )

    return {
        "pairs": pairs,
        "real_cloud": real_cloud,
        "gen_cloud": gen_clouds[0] if gen_clouds else real_cloud,
        "source_tree_id": task["source_tree_id"],
    }


def evaluate_all_trees(
    tasks: list[dict], num_workers: int
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Evaluate all trees in parallel.

    Returns:
        pair_df: DataFrame with one row per generated sample
        real_clouds: dict mapping source_tree_id -> np.ndarray
        gen_clouds: dict mapping source_tree_id -> np.ndarray (one representative)
    """
    all_pairs = []
    real_clouds = {}
    gen_clouds = {}

    print(f"Evaluating {len(tasks)} trees with {num_workers} workers...")
    start = time.time()

    with mp.Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(evaluate_tree_worker, tasks),
                total=len(tasks),
                desc="Per-tree metrics",
            )
        )

    for result in results:
        all_pairs.extend(result["pairs"])
        real_clouds[result["source_tree_id"]] = result["real_cloud"]
        gen_clouds[result["source_tree_id"]] = result["gen_cloud"]

    elapsed = time.time() - start
    print(f"  Completed {len(all_pairs)} pairs in {elapsed:.0f}s")

    pair_df = pd.DataFrame(all_pairs)
    return pair_df, real_clouds, gen_clouds


def aggregate_per_tree(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pair-level metrics to tree-level."""
    agg = (
        pair_df.groupby("source_tree_id")
        .agg(
            species=("species", "first"),
            height_m=("height_m", "first"),
            scan_type=("scan_type", "first"),
            n_samples=("cd", "count"),
            cd_mean=("cd", "mean"),
            cd_std=("cd", "std"),
            cd_median=("cd", "median"),
            cd_min=("cd", "min"),
            cd_max=("cd", "max"),
            hjsd_mean=("histogram_jsd", "mean"),
            hjsd_std=("histogram_jsd", "std"),
            hjsd_median=("histogram_jsd", "median"),
            hjsd_min=("histogram_jsd", "min"),
            hjsd_max=("histogram_jsd", "max"),
            crown_mae_p50_mean=("crown_mae_p50", "mean"),
            crown_mae_p75_mean=("crown_mae_p75", "mean"),
            crown_mae_p98_mean=("crown_mae_p98", "mean"),
        )
        .reset_index()
    )
    return agg


# =============================================================================
# Baselines (Parallelized)
# =============================================================================


def baseline_worker(task: dict) -> dict:
    """
    Worker: compute all per-pair metrics between two real point clouds.
    """
    max_points = task.get("max_points")
    n_radial = task.get("histogram_radial_bins", 16)
    n_height = task.get("histogram_height_bins", 32)
    rng = np.random.default_rng(task.get("seed", 42))

    cloud_a = load_point_cloud(task["path_a"], max_points=max_points, rng=rng)
    cloud_b = load_point_cloud(task["path_b"], max_points=max_points, rng=rng)

    cd = chamfer_distance(cloud_a, cloud_b)

    rz_a = compute_rz(cloud_a)
    rz_b = compute_rz(cloud_b)
    radial_edges, height_edges = compute_bin_edges(
        [rz_a[0], rz_b[0]], [rz_a[1], rz_b[1]], n_radial, n_height
    )
    hjsd, mae_p50, mae_p75, mae_p98 = compute_shape_metrics(
        rz_a, rz_b, radial_edges, height_edges
    )

    return {
        "species": task["species"],
        "scan_type": task["scan_type"],
        "height_bin": task["height_bin"],
        "label": task["label"],
        "cd": cd,
        "histogram_jsd": hjsd,
        "crown_mae_p50": mae_p50,
        "crown_mae_p75": mae_p75,
        "crown_mae_p98": mae_p98,
    }


def build_baseline_tasks(
    real_metadata: pd.DataFrame,
    interclass_pairs: int,
    seed: int,
    max_points: int | None = None,
    histogram_radial_bins: int = 16,
    histogram_height_bins: int = 32,
) -> list[dict]:
    """
    Build baseline task dicts.

    Intra-class: stratified by (species, scan_type, 5m height bin).
    All C(n,2) pairs for every group with >= 2 trees.
    Inter-class: random pairs from different species.
    """
    rng = np.random.default_rng(seed)

    # Assign height bins
    real_metadata = real_metadata.copy()
    real_metadata["height_bin"] = real_metadata["tree_H"].apply(get_height_bin)

    common_task_fields = {
        "max_points": max_points,
        "seed": seed,
        "histogram_radial_bins": histogram_radial_bins,
        "histogram_height_bins": histogram_height_bins,
    }

    tasks = []

    # Intra-class: stratified by (species, data_type, height_bin)
    groups = real_metadata.groupby(["species", "data_type", "height_bin"])
    n_singleton = 0
    for (species, scan_type, height_bin), group in groups:
        rows = list(group.itertuples(index=False))
        if len(rows) < 2:
            n_singleton += 1
            continue
        for a, b in combinations(range(len(rows)), 2):
            tasks.append(
                {
                    "path_a": rows[a].file_path,
                    "path_b": rows[b].file_path,
                    "species": species,
                    "scan_type": scan_type,
                    "height_bin": height_bin,
                    "label": "intra",
                    **common_task_fields,
                }
            )

    # Inter-class: random pairs from different species
    all_rows = list(real_metadata.itertuples(index=False))
    for _ in range(interclass_pairs):
        attempts = 0
        while attempts < 100:
            i, j = rng.choice(len(all_rows), size=2, replace=False)
            if all_rows[i].species != all_rows[j].species:
                break
            attempts += 1
        if attempts >= 100:
            continue
        tasks.append(
            {
                "path_a": all_rows[i].file_path,
                "path_b": all_rows[j].file_path,
                "species": "inter",
                "scan_type": "mixed",
                "height_bin": "mixed",
                "label": "inter",
                **common_task_fields,
            }
        )

    intra_count = sum(1 for t in tasks if t["label"] == "intra")
    inter_count = sum(1 for t in tasks if t["label"] == "inter")
    n_groups = sum(1 for _, g in groups if len(g) >= 2)
    print(
        f"  {n_groups} stratified groups ({n_singleton} singletons skipped)"
    )
    print(
        f"  Built {intra_count} intra-class + {inter_count} inter-class baseline tasks"
    )
    return tasks


def compute_baselines(tasks: list[dict], num_workers: int) -> pd.DataFrame:
    """Compute baseline metrics in parallel."""
    if not tasks:
        return pd.DataFrame(
            columns=["species", "scan_type", "height_bin", "label"] + PAIR_METRICS
        )

    print(f"Computing baselines with {num_workers} workers...")
    start = time.time()

    with mp.Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(baseline_worker, tasks),
                total=len(tasks),
                desc="Baselines",
            )
        )

    elapsed = time.time() - start
    print(f"  Completed {len(results)} baseline pairs in {elapsed:.0f}s")

    return pd.DataFrame(results)


# =============================================================================
# 3D Voxel JSD (Stratified)
# =============================================================================


def compute_jsd(
    real_clouds: list[np.ndarray],
    gen_clouds: list[np.ndarray],
    bins: int | None = None,
) -> float:
    """
    Compute Jensen-Shannon Divergence between real and generated point distributions.

    Voxelizes the combined point pool into bins^3 voxels, then computes JSD
    between the marginal occupancy distributions.

    If bins is None, uses adaptive formula: min(28, int((n_points/10)^(1/3))).
    """
    real_all = np.concatenate(real_clouds, axis=0)
    gen_all = np.concatenate(gen_clouds, axis=0)

    if bins is None:
        n_points = min(len(real_all), len(gen_all))
        bins = min(28, int((n_points / 10) ** (1 / 3)))
        bins = max(2, bins)

    all_points = np.concatenate([real_all, gen_all], axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    eps = 1e-6
    edges = [np.linspace(mins[d] - eps, maxs[d] + eps, bins + 1) for d in range(3)]

    hist_real, _ = np.histogramdd(real_all, bins=edges)
    hist_gen, _ = np.histogramdd(gen_all, bins=edges)

    hist_real = hist_real.flatten().astype(np.float64)
    hist_gen = hist_gen.flatten().astype(np.float64)

    p = hist_real / (hist_real.sum() + 1e-30)
    q = hist_gen / (hist_gen.sum() + 1e-30)

    m = 0.5 * (p + q)
    mask = m > 0
    kl_pm = np.sum(p[mask] * np.log(p[mask] / m[mask] + 1e-30))
    kl_qm = np.sum(q[mask] * np.log(q[mask] / m[mask] + 1e-30))

    return float(0.5 * kl_pm + 0.5 * kl_qm)


def compute_stratified_jsd(
    tree_df: pd.DataFrame,
    real_clouds: dict[str, np.ndarray],
    gen_clouds_map: dict[str, np.ndarray],
) -> dict:
    """
    Compute 3D voxel JSD stratified by species, scan type, and height bin.
    """
    results = {}

    def _jsd_for_tids(tids):
        real = [real_clouds[t] for t in tids if t in real_clouds]
        gen = [gen_clouds_map[t] for t in tids if t in gen_clouds_map]
        if len(real) >= 2 and len(gen) >= 2:
            return compute_jsd(real, gen)
        return None

    # Overall
    all_tids = tree_df["source_tree_id"].tolist()
    results["overall"] = _jsd_for_tids(all_tids)

    # By species
    results["by_species"] = {}
    for species, group in tree_df.groupby("species"):
        val = _jsd_for_tids(group["source_tree_id"].tolist())
        if val is not None:
            results["by_species"][species] = val

    # By scan type
    results["by_scan_type"] = {}
    for scan_type, group in tree_df.groupby("scan_type"):
        val = _jsd_for_tids(group["source_tree_id"].tolist())
        if val is not None:
            results["by_scan_type"][scan_type] = val

    # By height bin
    results["by_height_bin"] = {}
    tree_df_h = tree_df.copy()
    tree_df_h["height_bin"] = tree_df_h["height_m"].apply(get_height_bin)
    for hb, group in tree_df_h.groupby("height_bin"):
        val = _jsd_for_tids(group["source_tree_id"].tolist())
        if val is not None:
            results["by_height_bin"][hb] = val

    return results


# =============================================================================
# Breakdowns
# =============================================================================


def compute_breakdowns(
    pair_df: pd.DataFrame,
    tree_df: pd.DataFrame,
    stratified_jsd: dict,
) -> dict:
    """Compute metric breakdowns by species, height, scan type, and CFG scale.

    Merges per-pair metric aggregations with stratified 3D voxel JSD into
    unified tables per dimension.
    """
    breakdowns = {}

    tree_agg = dict(
        n_trees=("source_tree_id", "count"),
        cd_mean=("cd_mean", "mean"),
        cd_std=("cd_mean", "std"),
        hjsd_mean=("hjsd_mean", "mean"),
        hjsd_std=("hjsd_mean", "std"),
        crown_mae_p50_mean=("crown_mae_p50_mean", "mean"),
        crown_mae_p75_mean=("crown_mae_p75_mean", "mean"),
        crown_mae_p98_mean=("crown_mae_p98_mean", "mean"),
    )

    # --- By Species ---
    df = tree_df.groupby("species").agg(**tree_agg).reset_index()
    jsd_map = stratified_jsd.get("by_species", {})
    df["voxel_jsd"] = df["species"].map(jsd_map)
    breakdowns["by_species"] = df

    # --- By Height (5m bins) ---
    tree_df = tree_df.copy()
    tree_df["height_bin"] = tree_df["height_m"].apply(get_height_bin)
    df = tree_df.groupby("height_bin", observed=True).agg(**tree_agg).reset_index()
    jsd_map = stratified_jsd.get("by_height_bin", {})
    df["voxel_jsd"] = df["height_bin"].map(jsd_map)
    breakdowns["by_height"] = df

    # --- By Scan Type ---
    df = tree_df.groupby("scan_type").agg(**tree_agg).reset_index()
    jsd_map = stratified_jsd.get("by_scan_type", {})
    df["voxel_jsd"] = df["scan_type"].map(jsd_map)
    breakdowns["by_scan_type"] = df

    # --- By CFG Scale (pair-level, no voxel JSD) ---
    if "cfg_scale" in pair_df.columns:
        cfg_bins = np.arange(1.0, 5.0, 0.5)
        cfg_labels = [f"{b:.1f}-{b + 0.5:.1f}" for b in cfg_bins[:-1]]
        pair_df = pair_df.copy()
        pair_df["cfg_bin"] = pd.cut(
            pair_df["cfg_scale"], bins=cfg_bins, labels=cfg_labels, right=False
        )
        pair_agg = dict(
            n_pairs=("cd", "count"),
            cd_mean=("cd", "mean"),
            cd_std=("cd", "std"),
            hjsd_mean=("histogram_jsd", "mean"),
            hjsd_std=("histogram_jsd", "std"),
            crown_mae_p50_mean=("crown_mae_p50", "mean"),
            crown_mae_p75_mean=("crown_mae_p75", "mean"),
            crown_mae_p98_mean=("crown_mae_p98", "mean"),
        )
        breakdowns["by_cfg"] = (
            pair_df.groupby("cfg_bin", observed=True).agg(**pair_agg).reset_index()
        )

    return breakdowns


# =============================================================================
# Output / Reporting
# =============================================================================


def _summarize_metric(series: pd.Series) -> dict:
    """Compute summary stats for a single metric column."""
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "median": float(series.median()),
        "n": int(len(series)),
    }


def _summarize_baselines(baseline_df: pd.DataFrame) -> dict:
    """Build baselines summary dict from raw baseline DataFrame."""
    intra_df = baseline_df[baseline_df["label"] == "intra"]
    inter_df = baseline_df[baseline_df["label"] == "inter"]

    summary = {"intra": {}, "inter": {}}

    # Intra: overall + by stratification dimension
    if len(intra_df) > 0:
        summary["intra"]["overall"] = {
            m: _summarize_metric(intra_df[m]) for m in PAIR_METRICS
        }
        for dim in ["species", "scan_type", "height_bin"]:
            summary["intra"][f"by_{dim}"] = {}
            for val, grp in intra_df.groupby(dim):
                summary["intra"][f"by_{dim}"][str(val)] = {
                    m: _summarize_metric(grp[m]) for m in PAIR_METRICS
                }

    # Inter: overall only
    if len(inter_df) > 0:
        summary["inter"]["overall"] = {
            m: _summarize_metric(inter_df[m]) for m in PAIR_METRICS
        }

    return summary


def save_results(
    pair_df: pd.DataFrame,
    tree_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    stratified_jsd: dict,
    breakdowns: dict,
    output_dir: Path,
):
    """Save all evaluation results to the output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    breakdowns_dir = output_dir / "breakdowns"
    breakdowns_dir.mkdir(parents=True, exist_ok=True)

    # Per-pair CSV
    pair_csv = output_dir / "per_pair.csv"
    pair_df.to_csv(pair_csv, index=False)
    print(f"  Saved {len(pair_df)} pairs to {pair_csv}")

    # Per-tree CSV
    tree_csv = output_dir / "per_tree.csv"
    tree_df.to_csv(tree_csv, index=False)
    print(f"  Saved {len(tree_df)} trees to {tree_csv}")

    # Baselines CSV
    baselines_csv = output_dir / "baselines.csv"
    baseline_df.to_csv(baselines_csv, index=False)
    print(f"  Saved {len(baseline_df)} baseline pairs to {baselines_csv}")

    # Baselines summary JSON
    baselines_summary = _summarize_baselines(baseline_df)
    baselines_summary_path = output_dir / "baselines_summary.json"
    with open(baselines_summary_path, "w") as f:
        json.dump(baselines_summary, f, indent=2)
    print(f"  Saved baselines summary to {baselines_summary_path}")

    # Stratified JSD JSON
    jsd_path = output_dir / "stratified_jsd.json"
    with open(jsd_path, "w") as f:
        json.dump(stratified_jsd, f, indent=2)
    print(f"  Saved stratified JSD to {jsd_path}")

    # Breakdowns CSVs
    for name, df in breakdowns.items():
        csv_path = breakdowns_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved {name} breakdown to {csv_path}")

    # Summary JSON
    summary = {
        "per_pair": {
            "n_pairs": len(pair_df),
            **{m: _summarize_metric(pair_df[m]) for m in PAIR_METRICS},
        },
        "per_tree": {
            "n_trees": len(tree_df),
            "cd_mean_of_means": float(tree_df["cd_mean"].mean()),
            "cd_std_of_means": float(tree_df["cd_mean"].std()),
            "hjsd_mean_of_means": float(tree_df["hjsd_mean"].mean()),
            "hjsd_std_of_means": float(tree_df["hjsd_mean"].std()),
        },
        "baselines": baselines_summary,
        "stratified_jsd": stratified_jsd,
    }

    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to {summary_path}")


def _print_breakdown_table(df: pd.DataFrame, key_col: str, key_width: int = 25):
    """Print a unified breakdown table with all metrics."""
    cols = ["cd_mean", "hjsd_mean", "crown_mae_p50_mean", "crown_mae_p75_mean", "crown_mae_p98_mean"]
    has_voxel = "voxel_jsd" in df.columns

    header = f"  {'':>{key_width}s}  {'N':>5s}  {'CD':>8s}  {'HJSD':>8s}  {'CrP50':>8s}  {'CrP75':>8s}  {'CrP98':>8s}"
    if has_voxel:
        header += f"  {'VoxJSD':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for _, row in df.iterrows():
        n = int(row.get("n_trees", row.get("n_pairs", 0)))
        line = f"  {str(row[key_col]):>{key_width}s}  {n:>5d}"
        for c in cols:
            val = row.get(c, float("nan"))
            line += f"  {val:>8.4f}" if pd.notna(val) else f"  {'--':>8s}"
        if has_voxel:
            val = row.get("voxel_jsd", float("nan"))
            line += f"  {val:>8.4f}" if pd.notna(val) else f"  {'--':>8s}"
        print(line)


def print_results(
    pair_df: pd.DataFrame,
    tree_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    stratified_jsd: dict,
    breakdowns: dict,
):
    """Print formatted evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    # Per-pair
    print(f"\nPer-pair metrics (n={len(pair_df)} pairs):")
    for m in PAIR_METRICS:
        s = pair_df[m]
        print(f"  {m:20s}  mean={s.mean():.6f}  std={s.std():.6f}  median={s.median():.6f}")

    # Baselines
    intra_df = baseline_df[baseline_df["label"] == "intra"]
    inter_df = baseline_df[baseline_df["label"] == "inter"]

    if len(intra_df) > 0:
        print(f"\nIntra-class baselines (n={len(intra_df)}):")
        for m in PAIR_METRICS:
            s = intra_df[m]
            print(f"  {m:20s}  mean={s.mean():.6f}  std={s.std():.6f}")

    if len(inter_df) > 0:
        print(f"\nInter-class baselines (n={len(inter_df)}):")
        for m in PAIR_METRICS:
            s = inter_df[m]
            print(f"  {m:20s}  mean={s.mean():.6f}  std={s.std():.6f}")

    # Paper Table 1: Global Summary
    if len(intra_df) > 0:
        print(f"\n{'='*70}")
        print("TABLE 1: GLOBAL SUMMARY")
        print(f"{'='*70}")
        header = f"{'Metric':20s} | {'Gen vs Real':>22s} | {'Intra Baseline':>22s} | {'Ratio':>6s}"
        print(header)
        print("-" * len(header))
        for m in PAIR_METRICS:
            gen_val = pair_df[m].mean()
            gen_std = pair_df[m].std()
            base_val = intra_df[m].mean()
            base_std = intra_df[m].std()
            ratio = gen_val / base_val if base_val > 0 else float("nan")
            print(
                f"{m:20s} | {gen_val:9.6f} +/- {gen_std:8.6f} | "
                f"{base_val:9.6f} +/- {base_std:8.6f} | {ratio:6.2f}"
            )
        if stratified_jsd.get("overall") is not None:
            print(f"{'voxel_jsd':20s} | {stratified_jsd['overall']:9.6f}{'':>15s} |{'':>24s} |{'':>6s}")

    # Stratified breakdowns
    print(f"\n{'='*70}")
    print("TABLE 2: STRATIFIED BREAKDOWNS")
    print(f"{'='*70}")

    if "by_species" in breakdowns:
        print("\n  By Species:")
        _print_breakdown_table(breakdowns["by_species"], "species", 25)

    if "by_scan_type" in breakdowns:
        print("\n  By Scan Type:")
        _print_breakdown_table(breakdowns["by_scan_type"], "scan_type", 10)

    if "by_height" in breakdowns:
        print("\n  By Height:")
        _print_breakdown_table(breakdowns["by_height"], "height_bin", 10)

    if "by_cfg" in breakdowns:
        print("\n  By CFG Scale:")
        _print_breakdown_table(breakdowns["by_cfg"], "cfg_bin", 10)

    print("\n" + "=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated tree point clouds against real trees"
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the experiment (e.g., 'transformer-8-512-4096')",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/preprocessed-4096",
        help="Path to preprocessed dataset directory",
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="experiments",
        help="Base directory containing experiments",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=4096,
        help="Downsample real point clouds to this many points",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=40,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--interclass_pairs",
        type=int,
        default=200,
        help="Number of inter-class baseline pairs",
    )
    parser.add_argument(
        "--histogram_radial_bins",
        type=int,
        default=16,
        help="Number of radial bins for 2D histogram JSD",
    )
    parser.add_argument(
        "--histogram_height_bins",
        type=int,
        default=32,
        help="Number of height bins for 2D histogram JSD",
    )

    args = parser.parse_args()

    # Resolve paths
    experiments_dir = Path(args.experiments_dir)
    experiment_dir = experiments_dir / args.experiment_name
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment not found: {experiment_dir}")

    data_path = Path(args.data_path)
    zarr_dir = experiment_dir / "samples" / "zarr"
    if not zarr_dir.exists():
        raise FileNotFoundError(f"Generated samples not found: {zarr_dir}")

    output_dir = experiment_dir / "samples" / "evaluation"

    print(f"Experiment: {args.experiment_name}")
    print(f"Data path: {data_path}")
    print(f"Generated samples: {zarr_dir}")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.num_workers}")
    print(f"Histogram bins: {args.histogram_radial_bins} radial x {args.histogram_height_bins} height")
    print()

    # =========================================================================
    # 1. Load Metadata
    # =========================================================================

    print("=" * 60)
    print("LOADING METADATA")
    print("=" * 60)

    gen_metadata = load_generated_metadata(experiment_dir)
    real_metadata = load_real_metadata(data_path)

    # =========================================================================
    # 2. Per-Tree Evaluation
    # =========================================================================

    print("\n" + "=" * 60)
    print("PER-TREE EVALUATION (CD + Histogram JSD + Crown MAE)")
    print("=" * 60)

    tasks = build_tree_tasks(
        gen_metadata,
        real_metadata,
        zarr_dir,
        args.seed,
        max_points=args.max_points,
        histogram_radial_bins=args.histogram_radial_bins,
        histogram_height_bins=args.histogram_height_bins,
    )
    pair_df, real_clouds, gen_clouds_map = evaluate_all_trees(tasks, args.num_workers)

    if pair_df.empty:
        print("ERROR: No valid pairs found. Check data paths.")
        return

    tree_df = aggregate_per_tree(pair_df)
    print(f"  {len(pair_df)} pairs across {len(tree_df)} trees")

    # =========================================================================
    # 3. Stratified Baselines
    # =========================================================================

    print("\n" + "=" * 60)
    print("BASELINES (Stratified Intra-class + Inter-class)")
    print("=" * 60)

    baseline_tasks = build_baseline_tasks(
        real_metadata,
        interclass_pairs=args.interclass_pairs,
        seed=args.seed + 1000,
        max_points=args.max_points,
        histogram_radial_bins=args.histogram_radial_bins,
        histogram_height_bins=args.histogram_height_bins,
    )
    baseline_df = compute_baselines(baseline_tasks, args.num_workers)

    # =========================================================================
    # 4. Stratified 3D Voxel JSD
    # =========================================================================

    print("\n" + "=" * 60)
    print("STRATIFIED 3D VOXEL JSD")
    print("=" * 60)

    # gen_clouds_map already populated from per-tree evaluation (one representative per tree)
    stratified_jsd = compute_stratified_jsd(tree_df, real_clouds, gen_clouds_map)
    print(f"  Overall JSD: {stratified_jsd.get('overall', 'N/A')}")
    print(f"  Species groups: {len(stratified_jsd.get('by_species', {}))}")
    print(f"  Scan type groups: {len(stratified_jsd.get('by_scan_type', {}))}")
    print(f"  Height bin groups: {len(stratified_jsd.get('by_height_bin', {}))}")

    # =========================================================================
    # 5. Breakdowns
    # =========================================================================

    print("\n" + "=" * 60)
    print("BREAKDOWNS")
    print("=" * 60)

    breakdowns = compute_breakdowns(pair_df, tree_df, stratified_jsd)

    # =========================================================================
    # 6. Save and Report
    # =========================================================================

    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    save_results(pair_df, tree_df, baseline_df, stratified_jsd, breakdowns, output_dir)
    print_results(pair_df, tree_df, baseline_df, stratified_jsd, breakdowns)

    print("Evaluation complete!")


if __name__ == "__main__":
    main()
