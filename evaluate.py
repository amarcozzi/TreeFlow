"""
treeflow/evaluate.py

Evaluate generated tree point clouds against real trees using:
- Per-pair metrics: Chamfer Distance (CD) and Earth Mover's Distance (EMD)
- Baselines: intra-class (same species) and inter-class (different species)
- Global distributional metrics (PointFlow): JSD, MMD-CD, COV-CD, 1-NNA-CD
- Breakdowns by species, height, scan type, and CFG scale
"""

import json
import numpy as np
import pandas as pd
import argparse
import zarr
from pathlib import Path
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import multiprocessing as mp
import time


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


def compute_distances(
    real: np.ndarray, gen: np.ndarray, skip_emd: bool = False
) -> tuple[float, float]:
    """Compute CD and optionally EMD between real and generated point clouds."""
    cd = chamfer_distance(real, gen)
    emd = float("nan") if skip_emd else earth_movers_distance(real, gen)
    return cd, emd


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
    skip_emd: bool = False,
    max_points: int | None = None,
) -> list[dict]:
    """
    Build task dicts for parallel per-tree evaluation.

    Groups generated samples by source_tree_id and matches to real metadata.
    Returns list of serializable task dicts.
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

        # Build list of generated sample info
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
                "skip_emd": skip_emd,
                "max_points": max_points,
            }
        )

    if skipped_trees > 0:
        print(f"  Skipped {skipped_trees} trees not found in real test set")
    print(f"  Built {len(tasks)} tree evaluation tasks")
    return tasks


def evaluate_tree_worker(task: dict) -> dict:
    """
    Worker function: evaluate all generated samples for one real tree.

    Returns dict with:
        - 'pairs': list of flat metric dicts (one per generated sample)
        - 'real_cloud': the loaded real point cloud (for global metrics reuse)
        - 'source_tree_id': tree identifier
    """
    max_points = task.get("max_points")
    seed = task.get("seed", 42)
    rng = np.random.default_rng(seed)
    real_cloud = load_point_cloud(task["real_path"], max_points=max_points, rng=rng)
    skip_emd = task.get("skip_emd", False)
    pairs = []

    for gen_info in task["gen_samples"]:
        try:
            gen_cloud = load_point_cloud(gen_info["zarr_path"])
        except Exception:
            continue

        cd, emd = compute_distances(real_cloud, gen_cloud, skip_emd=skip_emd)

        pairs.append(
            {
                "source_tree_id": task["source_tree_id"],
                "sample_id": gen_info["sample_id"],
                "species": gen_info["species"],
                "height_m": gen_info["height_m"],
                "scan_type": gen_info["scan_type"],
                "cfg_scale": gen_info["cfg_scale"],
                "cd": cd,
                "emd": emd,
            }
        )

    return {
        "pairs": pairs,
        "real_cloud": real_cloud,
        "source_tree_id": task["source_tree_id"],
    }


def evaluate_all_trees(
    tasks: list[dict], num_workers: int
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate all trees in parallel using multiprocessing.

    Returns:
        pair_df: DataFrame with one row per generated sample
        real_clouds: dict mapping source_tree_id -> np.ndarray
    """
    all_pairs = []
    real_clouds = {}

    print(f"Evaluating {len(tasks)} trees with {num_workers} workers...")
    start = time.time()

    with mp.Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(evaluate_tree_worker, tasks),
                total=len(tasks),
                desc="Per-tree CD+EMD",
            )
        )

    for result in results:
        all_pairs.extend(result["pairs"])
        real_clouds[result["source_tree_id"]] = result["real_cloud"]

    elapsed = time.time() - start
    print(f"  Completed {len(all_pairs)} pairs in {elapsed:.0f}s")

    pair_df = pd.DataFrame(all_pairs)
    return pair_df, real_clouds


def aggregate_per_tree(pair_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pair-level metrics to tree-level.

    Returns DataFrame with one row per source tree.
    """
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
            emd_mean=("emd", "mean"),
            emd_std=("emd", "std"),
            emd_median=("emd", "median"),
            emd_min=("emd", "min"),
            emd_max=("emd", "max"),
        )
        .reset_index()
    )
    return agg


# =============================================================================
# Baselines (Parallelized)
# =============================================================================


def baseline_worker(task: dict) -> dict:
    """
    Worker function: compute CD + EMD between two real point clouds.

    Task dict must have 'path_a', 'path_b', 'label' (intra/inter), and 'species_a'/'species_b'.
    """
    max_points = task.get("max_points")
    rng = np.random.default_rng(task.get("seed", 42))
    cloud_a = load_point_cloud(task["path_a"], max_points=max_points, rng=rng)
    cloud_b = load_point_cloud(task["path_b"], max_points=max_points, rng=rng)
    cd, emd = compute_distances(cloud_a, cloud_b, skip_emd=task.get("skip_emd", False))
    return {
        "species_a": task["species_a"],
        "species_b": task["species_b"],
        "label": task["label"],
        "cd": cd,
        "emd": emd,
    }


def build_baseline_tasks(
    real_metadata: pd.DataFrame,
    pairs_per_species: int,
    interclass_pairs: int,
    seed: int,
    skip_emd: bool = False,
    max_points: int | None = None,
) -> list[dict]:
    """Build task dicts for intra-class and inter-class baselines."""
    rng = np.random.default_rng(seed)

    # Group trees by species
    species_to_rows = defaultdict(list)
    for _, row in real_metadata.iterrows():
        species_to_rows[row["species"]].append(row)

    tasks = []

    # Intra-class: pairs of different trees, same species
    for species, rows in species_to_rows.items():
        if len(rows) < 2:
            continue
        n_possible = len(rows) * (len(rows) - 1) // 2
        n_pairs = min(pairs_per_species, n_possible)
        for _ in range(n_pairs):
            i, j = rng.choice(len(rows), size=2, replace=False)
            tasks.append(
                {
                    "path_a": rows[i]["file_path"],
                    "path_b": rows[j]["file_path"],
                    "species_a": species,
                    "species_b": species,
                    "label": "intra",
                    "skip_emd": skip_emd,
                    "max_points": max_points,
                    "seed": seed,
                }
            )

    # Inter-class: pairs of trees from different species
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
                "species_a": all_rows[i].species,
                "species_b": all_rows[j].species,
                "label": "inter",
                "skip_emd": skip_emd,
                "max_points": max_points,
                "seed": seed,
            }
        )

    intra_count = sum(1 for t in tasks if t["label"] == "intra")
    inter_count = sum(1 for t in tasks if t["label"] == "inter")
    print(
        f"  Built {intra_count} intra-class + {inter_count} inter-class baseline tasks"
    )
    return tasks


def compute_baselines(tasks: list[dict], num_workers: int) -> pd.DataFrame:
    """
    Compute baseline distances in parallel.

    Returns DataFrame with columns: species_a, species_b, label, cd, emd.
    """
    if not tasks:
        return pd.DataFrame(columns=["species_a", "species_b", "label", "cd", "emd"])

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
# Global Distributional Metrics (CD-only)
# =============================================================================


_cd_matrix_clouds_b = None


def _cd_matrix_init(clouds_b: list[np.ndarray]):
    """Pool initializer: store clouds_b as a global to avoid per-task pickling."""
    global _cd_matrix_clouds_b
    _cd_matrix_clouds_b = clouds_b


def _cd_row_worker(args: tuple) -> tuple[int, np.ndarray]:
    """Compute one row of a CD cross-distance matrix."""
    row_idx, cloud_a = args
    row = np.array(
        [chamfer_distance(cloud_a, cb) for cb in _cd_matrix_clouds_b], dtype=np.float32
    )
    return row_idx, row


def compute_cd_matrix(
    clouds_a: list[np.ndarray],
    clouds_b: list[np.ndarray],
    num_workers: int,
    desc: str = "CD matrix",
) -> np.ndarray:
    """
    Compute pairwise CD matrix between two sets of point clouds.

    Returns (len(clouds_a), len(clouds_b)) matrix.
    Uses a pool initializer to share clouds_b across workers without per-task pickling.
    """
    n_a = len(clouds_a)
    n_b = len(clouds_b)

    # Each task is just (row_idx, cloud_a) — clouds_b is shared via initializer
    tasks = [(i, clouds_a[i]) for i in range(n_a)]

    matrix = np.zeros((n_a, n_b), dtype=np.float32)

    with mp.Pool(num_workers, initializer=_cd_matrix_init, initargs=(clouds_b,)) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_cd_row_worker, tasks),
                total=n_a,
                desc=desc,
            )
        )

    for row_idx, row_data in results:
        matrix[row_idx] = row_data

    return matrix


def compute_jsd(
    real_clouds: list[np.ndarray], gen_clouds: list[np.ndarray], bins: int = 28
) -> float:
    """
    Compute Jensen-Shannon Divergence between real and generated point distributions.

    Voxelizes the combined point pool into bins^3 voxels, then computes JSD
    between the marginal occupancy distributions.
    """
    # Pool all points
    real_all = np.concatenate(real_clouds, axis=0)
    gen_all = np.concatenate(gen_clouds, axis=0)
    all_points = np.concatenate([real_all, gen_all], axis=0)

    # Determine bin edges from combined data
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    # Add small epsilon to avoid edge issues
    eps = 1e-6
    edges = [np.linspace(mins[d] - eps, maxs[d] + eps, bins + 1) for d in range(3)]

    # Histogram for real and generated
    hist_real, _ = np.histogramdd(real_all, bins=edges)
    hist_gen, _ = np.histogramdd(gen_all, bins=edges)

    # Normalize to probability distributions
    hist_real = hist_real.flatten().astype(np.float64)
    hist_gen = hist_gen.flatten().astype(np.float64)

    p = hist_real / (hist_real.sum() + 1e-30)
    q = hist_gen / (hist_gen.sum() + 1e-30)

    # JSD = 0.5 * KL(p||m) + 0.5 * KL(q||m) where m = 0.5*(p+q)
    m = 0.5 * (p + q)

    # Avoid log(0) by only computing where m > 0
    mask = m > 0
    kl_pm = np.sum(p[mask] * np.log(p[mask] / m[mask] + 1e-30))
    kl_qm = np.sum(q[mask] * np.log(q[mask] / m[mask] + 1e-30))

    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    return float(jsd)


def compute_global_metrics_from_clouds(
    real_list: list[np.ndarray],
    gen_list: list[np.ndarray],
    num_workers: int,
) -> dict:
    """
    Compute global distributional metrics from pre-loaded cloud lists.

    Args:
        real_list: List of real point clouds (one per tree)
        gen_list: List of generated point clouds (one per tree)
        num_workers: Number of parallel workers
    """
    n = len(real_list)
    assert (
        len(gen_list) == n
    ), f"Mismatched lengths: {len(real_list)} real vs {len(gen_list)} gen"

    print(f"Computing global metrics for {n} trees...")

    # 1. JSD
    print("  Computing JSD...")
    jsd = compute_jsd(real_list, gen_list, bins=28)
    print(f"    JSD = {jsd:.6f}")

    # 2. Cross-distance matrix (real x gen)
    print("  Computing cross-distance matrix (real x gen)...")
    start = time.time()
    cross_matrix = compute_cd_matrix(
        real_list, gen_list, num_workers, desc="Cross CD (real x gen)"
    )
    print(f"    Cross matrix computed in {time.time() - start:.0f}s")

    # 3. MMD-CD: For each real, find min CD to any generated. Mean across all.
    min_real_to_gen = cross_matrix.min(
        axis=1
    )  # (n,) min CD from each real to closest gen
    mmd_cd = float(min_real_to_gen.mean())
    print(f"    MMD-CD = {mmd_cd:.6f}")

    # 4. COV-CD: For each gen, find its NN real. Count how many unique reals are matched.
    nn_real_for_gen = cross_matrix.argmin(
        axis=0
    )  # (n,) for each gen, which real is closest
    unique_matched = len(set(nn_real_for_gen.tolist()))
    cov_cd = float(unique_matched / n)
    print(f"    COV-CD = {cov_cd:.4f} ({unique_matched}/{n} reals covered)")

    # 5. 1-NNA-CD
    # Need within-set distance matrices
    print("  Computing within-real distance matrix...")
    start = time.time()
    within_real = compute_cd_matrix(
        real_list, real_list, num_workers, desc="Within-real CD"
    )
    print(f"    Within-real matrix computed in {time.time() - start:.0f}s")

    print("  Computing within-gen distance matrix...")
    start = time.time()
    within_gen = compute_cd_matrix(
        gen_list, gen_list, num_workers, desc="Within-gen CD"
    )
    print(f"    Within-gen matrix computed in {time.time() - start:.0f}s")

    # 1-NNA: For each point in the pool (real + gen), find 1-NN excluding self.
    # Classify as "same set" or "different set". If distributions match, accuracy ~ 50%.
    # For real[i]: NN in real set (excluding self) vs NN in gen set
    # For gen[i]: NN in gen set (excluding self) vs NN in real set

    correct = 0
    total = 2 * n

    for i in range(n):
        # Real[i]: compare nearest real (excl self) vs nearest gen
        within_dists = within_real[i].copy()
        within_dists[i] = np.inf  # exclude self
        nn_same = within_dists.min()
        nn_diff = cross_matrix[i].min()  # nearest gen
        if nn_same < nn_diff:
            correct += 1

    # cross_matrix is (real x gen), so cross_matrix.T is (gen x real)
    cross_T = cross_matrix.T

    for j in range(n):
        # Gen[j]: compare nearest gen (excl self) vs nearest real
        within_dists = within_gen[j].copy()
        within_dists[j] = np.inf  # exclude self
        nn_same = within_dists.min()
        nn_diff = cross_T[j].min()  # nearest real
        if nn_same < nn_diff:
            correct += 1

    nna_cd = float(correct / total * 100)  # percentage
    print(f"    1-NNA-CD = {nna_cd:.2f}% (50% = perfect)")

    return {
        "jsd": jsd,
        "mmd_cd": mmd_cd,
        "cov_cd": cov_cd,
        "cov_cd_count": unique_matched,
        "cov_cd_total": n,
        "one_nna_cd": nna_cd,
        "n_trees": n,
    }


# =============================================================================
# Breakdowns
# =============================================================================


def compute_breakdowns(pair_df: pd.DataFrame, tree_df: pd.DataFrame) -> dict:
    """Compute metric breakdowns by species, height, scan type, and CFG scale."""
    breakdowns = {}

    # --- By Species (tree-level) ---
    by_species = (
        tree_df.groupby("species")
        .agg(
            n_trees=("source_tree_id", "count"),
            cd_mean=("cd_mean", "mean"),
            cd_std=("cd_mean", "std"),
            cd_median=("cd_mean", "median"),
            emd_mean=("emd_mean", "mean"),
            emd_std=("emd_mean", "std"),
            emd_median=("emd_mean", "median"),
        )
        .reset_index()
    )
    breakdowns["by_species"] = by_species

    # --- By Height (tree-level) ---
    height_bins = [0, 5, 10, 20, float("inf")]
    height_labels = ["0-5m", "5-10m", "10-20m", "20m+"]
    tree_df = tree_df.copy()
    tree_df["height_bin"] = pd.cut(
        tree_df["height_m"], bins=height_bins, labels=height_labels, right=False
    )
    by_height = (
        tree_df.groupby("height_bin", observed=True)
        .agg(
            n_trees=("source_tree_id", "count"),
            cd_mean=("cd_mean", "mean"),
            cd_std=("cd_mean", "std"),
            cd_median=("cd_mean", "median"),
            emd_mean=("emd_mean", "mean"),
            emd_std=("emd_mean", "std"),
            emd_median=("emd_mean", "median"),
        )
        .reset_index()
    )
    breakdowns["by_height"] = by_height

    # --- By Scan Type (tree-level) ---
    by_scan = (
        tree_df.groupby("scan_type")
        .agg(
            n_trees=("source_tree_id", "count"),
            cd_mean=("cd_mean", "mean"),
            cd_std=("cd_mean", "std"),
            cd_median=("cd_mean", "median"),
            emd_mean=("emd_mean", "mean"),
            emd_std=("emd_mean", "std"),
            emd_median=("emd_mean", "median"),
        )
        .reset_index()
    )
    breakdowns["by_scan_type"] = by_scan

    # --- By CFG Scale (pair-level) ---
    if "cfg_scale" in pair_df.columns:
        cfg_bins = np.arange(1.0, 5.0, 0.5)
        cfg_labels = [f"{b:.1f}-{b+0.5:.1f}" for b in cfg_bins[:-1]]
        pair_df = pair_df.copy()
        pair_df["cfg_bin"] = pd.cut(
            pair_df["cfg_scale"], bins=cfg_bins, labels=cfg_labels, right=False
        )
        by_cfg = (
            pair_df.groupby("cfg_bin", observed=True)
            .agg(
                n_pairs=("cd", "count"),
                cd_mean=("cd", "mean"),
                cd_std=("cd", "std"),
                cd_median=("cd", "median"),
                emd_mean=("emd", "mean"),
                emd_std=("emd", "std"),
                emd_median=("emd", "median"),
            )
            .reset_index()
        )
        breakdowns["by_cfg"] = by_cfg

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
        "n": len(series),
    }


def save_results(
    pair_df: pd.DataFrame,
    tree_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    global_metrics: dict | None,
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

    # Baselines — derive summary from the raw DataFrame
    intra_df = baseline_df[baseline_df["label"] == "intra"]
    inter_df = baseline_df[baseline_df["label"] == "inter"]

    baselines_summary = {
        "intra": {
            "cd": _summarize_metric(intra_df["cd"]) if len(intra_df) else {},
            "emd": _summarize_metric(intra_df["emd"]) if len(intra_df) else {},
        },
        "inter": {
            "cd": _summarize_metric(inter_df["cd"]) if len(inter_df) else {},
            "emd": _summarize_metric(inter_df["emd"]) if len(inter_df) else {},
        },
    }

    baselines_path = output_dir / "baselines.json"
    with open(baselines_path, "w") as f:
        json.dump(baselines_summary, f, indent=2)
    print(f"  Saved baselines to {baselines_path}")

    # Global metrics JSON
    if global_metrics is not None:
        global_path = output_dir / "global_metrics.json"
        with open(global_path, "w") as f:
            json.dump(global_metrics, f, indent=2)
        print(f"  Saved global metrics to {global_path}")

    # Breakdowns CSVs
    for name, df in breakdowns.items():
        csv_path = breakdowns_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved {name} breakdown to {csv_path}")

    # Summary JSON
    summary = {
        "per_pair": {
            "n_pairs": len(pair_df),
            "cd": _summarize_metric(pair_df["cd"]),
            "emd": _summarize_metric(pair_df["emd"]),
        },
        "per_tree": {
            "n_trees": len(tree_df),
            "cd_mean_of_means": float(tree_df["cd_mean"].mean()),
            "cd_std_of_means": float(tree_df["cd_mean"].std()),
            "emd_mean_of_means": float(tree_df["emd_mean"].mean()),
            "emd_std_of_means": float(tree_df["emd_mean"].std()),
        },
        "baselines": baselines_summary,
    }
    if global_metrics is not None:
        summary["global_metrics"] = global_metrics

    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to {summary_path}")


def print_results(
    pair_df: pd.DataFrame,
    tree_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    global_metrics: dict | None,
):
    """Print formatted evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    # Per-pair
    print(f"\nPer-pair metrics (n={len(pair_df)} pairs):")
    print(
        f"  CD:  mean={pair_df['cd'].mean():.6f}  std={pair_df['cd'].std():.6f}  median={pair_df['cd'].median():.6f}"
    )
    print(
        f"  EMD: mean={pair_df['emd'].mean():.6f}  std={pair_df['emd'].std():.6f}  median={pair_df['emd'].median():.6f}"
    )

    # Per-tree
    print(f"\nPer-tree metrics (n={len(tree_df)} trees):")
    print(
        f"  CD:  mean={tree_df['cd_mean'].mean():.6f}  std={tree_df['cd_mean'].std():.6f}"
    )
    print(
        f"  EMD: mean={tree_df['emd_mean'].mean():.6f}  std={tree_df['emd_mean'].std():.6f}"
    )

    # Baselines
    intra_df = baseline_df[baseline_df["label"] == "intra"]
    inter_df = baseline_df[baseline_df["label"] == "inter"]

    if len(intra_df) > 0:
        print(f"\nBaselines:")
        print(
            f"  Intra-class CD:  mean={intra_df['cd'].mean():.6f}  std={intra_df['cd'].std():.6f}  (n={len(intra_df)})"
        )
        print(
            f"  Intra-class EMD: mean={intra_df['emd'].mean():.6f}  std={intra_df['emd'].std():.6f}"
        )
    if len(inter_df) > 0:
        print(
            f"  Inter-class CD:  mean={inter_df['cd'].mean():.6f}  std={inter_df['cd'].std():.6f}  (n={len(inter_df)})"
        )
        print(
            f"  Inter-class EMD: mean={inter_df['emd'].mean():.6f}  std={inter_df['emd'].std():.6f}"
        )

    # Interpretation
    gen_cd = pair_df["cd"].mean()
    if len(intra_df) > 0 and intra_df["cd"].mean() > 0:
        ratio = gen_cd / intra_df["cd"].mean()
        print(f"\n  Gen/Intra ratio: {ratio:.2f} (< 1.5 is good)")
    if len(inter_df) > 0 and inter_df["cd"].mean() > 0:
        print(
            f"  Gen < Inter: {gen_cd < inter_df['cd'].mean()} (conditioning works if True)"
        )

    # Global metrics
    if global_metrics is not None:
        print(f"\nGlobal distributional metrics (PointFlow):")
        print(f"  JSD:      {global_metrics['jsd']:.6f}")
        print(f"  MMD-CD:   {global_metrics['mmd_cd']:.6f}")
        print(
            f"  COV-CD:   {global_metrics['cov_cd']:.4f} ({global_metrics['cov_cd_count']}/{global_metrics['cov_cd_total']})"
        )
        print(f"  1-NNA-CD: {global_metrics['one_nna_cd']:.2f}% (50% = perfect)")

    print("=" * 70 + "\n")


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
        help="Downsample real point clouds to this many points (should match training)",
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
        "--skip_emd",
        action="store_true",
        help="Skip Earth Mover's Distance (EMD is O(n^3), very slow at 4096 points)",
    )
    parser.add_argument(
        "--skip_global",
        action="store_true",
        help="Skip global distributional metrics (saves ~1h)",
    )
    parser.add_argument(
        "--baseline_pairs_per_species",
        type=int,
        default=20,
        help="Number of intra-class baseline pairs per species",
    )
    parser.add_argument(
        "--interclass_pairs",
        type=int,
        default=200,
        help="Number of inter-class baseline pairs",
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

    metrics_label = "CD" if args.skip_emd else "CD + EMD"
    print("\n" + "=" * 60)
    print(f"PER-TREE EVALUATION ({metrics_label})")
    print("=" * 60)

    tasks = build_tree_tasks(
        gen_metadata,
        real_metadata,
        zarr_dir,
        args.seed,
        skip_emd=args.skip_emd,
        max_points=args.max_points,
    )
    pair_df, real_clouds = evaluate_all_trees(tasks, args.num_workers)

    if pair_df.empty:
        print("ERROR: No valid pairs found. Check data paths.")
        return

    tree_df = aggregate_per_tree(pair_df)
    print(f"  {len(pair_df)} pairs across {len(tree_df)} trees")

    # =========================================================================
    # 3. Baselines
    # =========================================================================

    print("\n" + "=" * 60)
    print("BASELINES (Intra-class + Inter-class)")
    print("=" * 60)

    baseline_tasks = build_baseline_tasks(
        real_metadata,
        pairs_per_species=args.baseline_pairs_per_species,
        interclass_pairs=args.interclass_pairs,
        seed=args.seed + 1000,
        skip_emd=args.skip_emd,
        max_points=args.max_points,
    )
    baseline_df = compute_baselines(baseline_tasks, args.num_workers)

    # =========================================================================
    # 4. Global Distributional Metrics
    # =========================================================================

    global_metrics = None
    if not args.skip_global:
        print("\n" + "=" * 60)
        print("GLOBAL DISTRIBUTIONAL METRICS")
        print("=" * 60)

        # Select one representative generated sample per tree
        rng = np.random.default_rng(args.seed + 2000)
        tree_ids = sorted(real_clouds.keys())
        real_list = [real_clouds[tid] for tid in tree_ids]
        gen_list = []

        for tid in tree_ids:
            tree_pairs = pair_df[pair_df["source_tree_id"] == tid]
            if len(tree_pairs) == 0:
                gen_list.append(real_clouds[tid])  # fallback
                continue

            # Pick random sample and reload its cloud
            chosen = tree_pairs.iloc[rng.integers(len(tree_pairs))]
            sample_file = f"{chosen['sample_id']}.zarr"
            gen_path = str(zarr_dir / sample_file)
            try:
                gen_cloud = load_point_cloud(gen_path)
                gen_list.append(gen_cloud)
            except Exception:
                gen_list.append(real_clouds[tid])  # fallback

        global_metrics = compute_global_metrics_from_clouds(
            real_list, gen_list, args.num_workers
        )

    # =========================================================================
    # 5. Breakdowns
    # =========================================================================

    print("\n" + "=" * 60)
    print("BREAKDOWNS")
    print("=" * 60)

    breakdowns = compute_breakdowns(pair_df, tree_df)

    # =========================================================================
    # 6. Save and Report
    # =========================================================================

    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    save_results(pair_df, tree_df, baseline_df, global_metrics, breakdowns, output_dir)
    print_results(pair_df, tree_df, baseline_df, global_metrics)

    print("Evaluation complete!")


if __name__ == "__main__":
    main()
