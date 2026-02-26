"""
TreeFlow evaluation v2: stem-tracker morphological metrics + paired-then-aggregate.

Replaces evaluate.py with:
  - Stem tracker as unified reference axis (degree-3 polynomial spine)
  - Dataframe-first architecture (df_real, df_gen, df_pairs, df_per_tree)
  - Per-tree anchored baselines (intra-class, inter-class)
  - Population metrics (Coverage, MMD, Voxel JSD)
  - Statistical tests (Wasserstein)
"""

import sys
import json
import time
import argparse

# Force line-buffered stdout so prints appear immediately under SLURM
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import zarr
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde, wasserstein_distance
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from stem_tracker import compute_rs_spine


# =============================================================================
# Constants
# =============================================================================

HEIGHT_BIN_EDGES = [0, 5, 10, 15, 20, 25, 30, 35, 40, float("inf")]
HEIGHT_BIN_LABELS = [
    "0-5", "5-10", "10-15", "15-20", "20-25",
    "25-30", "30-35", "35-40", "40+",
]

PAIR_METRICS = [
    "reconstruction_cd",
    "vert_kde_jsd",
    "hist_2d_jsd",
    "delta_crown_vol",
    "delta_max_crown_r",
    "delta_max_crown_r_rel_s",
    "delta_hcb",
]

PAIR_METRIC_UNITS = {
    "reconstruction_cd": "",
    "vert_kde_jsd": "",
    "hist_2d_jsd": "",
    "delta_crown_vol": "m\u00b3",
    "delta_max_crown_r": "m",
    "delta_max_crown_r_rel_s": "",
    "delta_hcb": "m",
}

METRIC_DISPLAY = {
    # Panel (a) — population: point-cloud space
    "coverage": "Coverage",
    "mmd": "MMD (CD)",
    "one_nna": "1-NNA",
    "diversity_ratio": "Diversity ratio",
    "voxel_jsd": "Voxel JSD",
    # Panel (a) — population: morphological marginals
    "w1_crown_volume": ("Crown volume", "m\u00b3"),
    "w1_max_crown_r": ("Max crown radius", "m"),
    "w1_hcb": ("Height to crown base", "m"),
    # Panel (b) — conditioning fidelity
    "reconstruction_cd": "Chamfer distance",
    "vert_kde_jsd": "Vertical KDE JSD",
    "hist_2d_jsd": "2D histogram JSD",
    "delta_crown_vol": ("\u0394 Crown volume", "m\u00b3"),
    "delta_max_crown_r": ("\u0394 Max crown radius", "m"),
    "delta_max_crown_r_rel_s": ("\u0394 Crown radius / stem rad.", ""),
    "delta_hcb": ("\u0394 Height to crown base", "m"),
}

EVAL_VERSION = 2


# =============================================================================
# Helpers (reused from evaluate.py)
# =============================================================================


def get_height_bin(h: float) -> str:
    """Assign a height value to its 5m bin label."""
    for i, (lo, hi) in enumerate(zip(HEIGHT_BIN_EDGES[:-1], HEIGHT_BIN_EDGES[1:])):
        if lo <= h < hi:
            return HEIGHT_BIN_LABELS[i]
    return HEIGHT_BIN_LABELS[-1]


def normalize_tree_id(tree_id) -> str:
    """Normalize tree ID to 5-digit zero-padded string format."""
    return str(tree_id).zfill(5)


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


def chamfer_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Chamfer Distance between two point clouds (squared L2, not L1-CD)."""
    dist_matrix = cdist(p1, p2, metric="sqeuclidean")
    min_p1_to_p2 = dist_matrix.min(axis=1).mean()
    min_p2_to_p1 = dist_matrix.min(axis=0).mean()
    return float((min_p1_to_p2 + min_p2_to_p1) / 2)


def compute_jsd_3d(
    real_clouds: list[np.ndarray],
    gen_clouds: list[np.ndarray],
    bins: int | None = None,
) -> float:
    """3D voxel JSD between real and generated point distributions."""
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


# =============================================================================
# Pose canonicalization (Layer 1)
# =============================================================================


def canonicalize(points: np.ndarray) -> np.ndarray:
    """Centroid-center, PCA-align XY, resolve sign ambiguity. O(N)."""
    points = points.copy()
    # 1. Re-center at centroid
    points -= points.mean(axis=0)
    # 2. PCA on XY projection, rotate PC1 → +X
    xy = points[:, :2]
    _, _, Vt = np.linalg.svd(xy - xy.mean(axis=0), full_matrices=False)
    R = np.eye(3)
    R[:2, :2] = Vt
    points = points @ R.T
    # 3. Sign ambiguity: heavy side → +X
    if np.sum(points[:, 0] ** 3) < 0:
        points[:, 0] *= -1
    return points


# =============================================================================
# JSD helper
# =============================================================================


def jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Jensen-Shannon divergence between two probability distributions."""
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * kl_pm + 0.5 * kl_qm)


# =============================================================================
# Stem-tracker metric extraction
# =============================================================================


def extract_stem_metrics(
    cloud: np.ndarray,
    height_m: float = 1.0,
    kde_bins: int = 64,
    hist_r_bins: int = 16,
    hist_s_bins: int = 32,
    volume_bins: int = 30,
    hcb_bins: int = 50,
    skip_hcb: bool = False,
    spine_num_bins: int = 20,
    spine_sigma_frac: float = 0.10,
    spine_degree: int = 3,
    spine_density_k: int = 16,
    spine_n_refine: int = 1,
    spine_max_step_frac: float = 0.05,
    spine_outlier_mad_k: float = 2.5,
) -> dict:
    """Extract all stem-tracker morphological metrics from a single point cloud.

    Returns dict with:
      vert_kde      : (kde_bins,) array — 1D KDE of z-axis projection
      hist_2d       : (hist_r_bins * hist_s_bins,) array — flattened 2D (r,s) histogram
      crown_volume  : float
      max_crown_r   : float
      max_crown_r_rel_s : float — normalized arc-length position of max crown r
      hcb           : float or NaN
    """
    z = cloud[:, 2]
    z_min, z_max = z.min(), z.max()

    # --- a. Run stem tracker ---
    r, s, spine_xyz, poly_x, poly_y = compute_rs_spine(
        cloud,
        num_bins=spine_num_bins,
        sigma_frac=spine_sigma_frac,
        degree=spine_degree,
        density_k=spine_density_k,
        n_refine=spine_n_refine,
        max_step_frac=spine_max_step_frac,
        outlier_mad_k=spine_outlier_mad_k,
    )

    # --- b. Vertical 1D KDE ---
    try:
        kde = gaussian_kde(z)
        z_eval = np.linspace(z_min, z_max, kde_bins)
        vert_kde = kde(z_eval)
        vert_kde = vert_kde / (vert_kde.sum() + 1e-30)
    except Exception:
        vert_kde = np.ones(kde_bins) / kde_bins

    # --- c. 2D (r, s) histogram ---
    # NOTE: Bin edges are tree-specific (based on each tree's r_max, s_max).
    # This means hist_2d_jsd compares normalized radial-arc-length profile shapes
    # rather than absolute spatial distributions. This is intentional: absolute
    # scale differences are captured by delta_crown_vol and delta_max_crown_r.
    eps = 1e-6
    r_max = r.max() + eps
    s_max = s.max() + eps if s.max() > 0 else eps
    r_edges = np.linspace(0, r_max, hist_r_bins + 1)
    s_edges = np.linspace(0, s_max, hist_s_bins + 1)
    hist_2d, _, _ = np.histogram2d(r, s, bins=[r_edges, s_edges])
    hist_2d_flat = hist_2d.flatten()
    hist_2d_sum = hist_2d_flat.sum()
    if hist_2d_sum > 0:
        hist_2d_flat = hist_2d_flat / hist_2d_sum

    # --- d. Crown volume + max crown radius ---
    s_bin_edges = np.linspace(0, s_max, volume_bins + 1)
    ds = s_max / volume_bins if volume_bins > 0 else 1.0
    crown_volume = 0.0
    max_crown_r = 0.0
    max_crown_r_bin = 0

    for i in range(volume_bins):
        mask = (s >= s_bin_edges[i]) & (s < s_bin_edges[i + 1])
        if mask.sum() > 0:
            r_mean = r[mask].mean()
        else:
            r_mean = 0.0
        crown_volume += ds * np.pi * r_mean ** 2
        if r_mean > max_crown_r:
            max_crown_r = r_mean
            max_crown_r_bin = i

    max_crown_r_rel_s = (max_crown_r_bin + 0.5) / volume_bins if volume_bins > 0 else 0.5

    # --- e. HCB (tentative) ---
    hcb_val = float("nan")
    if not skip_hcb:
        try:
            hcb_s_edges = np.linspace(0, s_max, hcb_bins + 1)
            counts = np.zeros(hcb_bins)
            for i in range(hcb_bins):
                counts[i] = ((s >= hcb_s_edges[i]) & (s < hcb_s_edges[i + 1])).sum()

            # Smooth with Gaussian kernel
            smoothed = gaussian_filter1d(counts, sigma=2.5)

            # Find local min before first major max
            mean_freq = smoothed.mean()
            # Find first peak above mean
            first_peak = None
            for i in range(1, len(smoothed) - 1):
                if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
                    if smoothed[i] > mean_freq:
                        first_peak = i
                        break

            if first_peak is not None and first_peak > 1:
                # Find minimum before this peak
                min_idx = np.argmin(smoothed[:first_peak])
                hcb_val = float((min_idx + 0.5) / hcb_bins)  # normalized [0, 1]
            else:
                # Fallback: CDF at 5%
                cdf = np.cumsum(counts)
                cdf = cdf / (cdf[-1] + 1e-30)
                idx_5 = np.searchsorted(cdf, 0.05)
                hcb_val = float((idx_5 + 0.5) / hcb_bins)
        except Exception:
            hcb_val = float("nan")

    # Convert from normalized space to metric units
    # Normalization: normalized = (raw - center) / height_m * 2.0
    # So: metric_distance = normalized_distance * (height_m / 2.0)
    scale = height_m / 2.0
    return {
        "vert_kde": vert_kde,                                    # probability — no conversion
        "hist_2d": hist_2d_flat,                                 # probability — no conversion
        "crown_volume": float(crown_volume * scale**3),          # normalized³ → m³
        "max_crown_r": float(max_crown_r * scale),               # normalized → m
        "max_crown_r_rel_s": float(max_crown_r_rel_s),           # unitless [0,1]
        "hcb": float(hcb_val * height_m),                        # fraction × height → m
    }


# =============================================================================
# Data loading
# =============================================================================


def load_real_metadata(data_path: Path) -> pd.DataFrame:
    """Load real tree metadata, filter to test split."""
    data_path = Path(data_path)
    csv_path = data_path / "metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        raise ValueError("CSV missing 'split' column.")

    df["file_id"] = df["filename"].apply(lambda x: Path(x).stem)
    df["file_path"] = df["file_id"].apply(lambda x: str(data_path / f"{x}.zarr"))
    df = df[df["file_path"].apply(lambda x: Path(x).exists())]
    test_df = df[df["split"] == "test"].copy()

    # Derive height_bin
    test_df["height_bin"] = test_df["tree_H"].apply(get_height_bin)

    # Ensure genus column exists
    if "genus" not in test_df.columns:
        test_df["genus"] = "unknown"

    print(f"Loaded real metadata: {len(test_df)} test trees (of {len(df)} with zarr files)")
    return test_df


def load_generated_metadata(experiment_dir: Path, real_metadata: pd.DataFrame) -> pd.DataFrame:
    """Load generated samples metadata, join genus from real metadata."""
    samples_dir = Path(experiment_dir) / "samples"
    csv_path = samples_dir / "samples_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata not found: {csv_path}\nRun postprocess_samples.py first.")

    df = pd.read_csv(csv_path)
    df["source_tree_id"] = df["source_tree_id"].apply(normalize_tree_id)

    # Join genus from real metadata
    real_genus = real_metadata.set_index("file_id")[["genus"]].rename(
        columns={"genus": "genus"}
    )
    df = df.merge(
        real_genus, left_on="source_tree_id", right_index=True, how="left"
    )
    if "genus" not in df.columns or df["genus"].isna().all():
        df["genus"] = "unknown"

    print(f"Loaded metadata for {len(df)} generated samples")
    print(f"  Unique source trees: {df['source_tree_id'].nunique()}")
    print(f"  Species: {df['species'].nunique()}")
    if "cfg_scale" in df.columns:
        print(f"  CFG scale range: {df['cfg_scale'].min():.2f} - {df['cfg_scale'].max():.2f}")
    return df


# =============================================================================
# Build df_real
# =============================================================================

# Module-level config for workers (set before pool creation)
_worker_config = {}


def _real_tree_worker(task: dict) -> dict | None:
    """Process one real tree: load, canonicalize, extract metrics."""
    try:
        rng = np.random.default_rng(task["seed"])
        cloud = load_point_cloud(task["file_path"], max_points=task["max_points"], rng=rng)

        canon_cloud = canonicalize(cloud)
        metrics = extract_stem_metrics(
            cloud,
            height_m=task["height_m"],
            skip_hcb=task.get("skip_hcb", False),
            spine_num_bins=task.get("spine_bins", 20),
        )

        result = {
            "tree_id": task["tree_id"],
            "species": task["species"],
            "genus": task.get("genus", "unknown"),
            "scan_type": task["scan_type"],
            "height_m": task["height_m"],
            "height_bin": task["height_bin"],
            "num_points": len(cloud),
            "crown_volume": metrics["crown_volume"],
            "max_crown_r": metrics["max_crown_r"],
            "max_crown_r_rel_s": metrics["max_crown_r_rel_s"],
            "hcb": metrics["hcb"],
            # Array metrics stored separately — will be in parquet
            "vert_kde": metrics["vert_kde"].tolist(),
            "hist_2d": metrics["hist_2d"].tolist(),
            # Store clouds for later CD computation
            "_canon_cloud": canon_cloud,
        }
        return result
    except Exception as e:
        return {"tree_id": task["tree_id"], "_error": str(e)}


def build_df_real(
    real_metadata: pd.DataFrame,
    data_path: Path,
    max_points: int = 4096,
    num_workers: int = 40,
    seed: int = 42,
    skip_hcb: bool = False,
    spine_bins: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """Build metrics DataFrame for all real test trees.

    Returns (df_real, real_clouds) where real_clouds maps tree_id → canonicalized array.
    """
    tasks = []
    for _, row in real_metadata.iterrows():
        tasks.append({
            "tree_id": row["file_id"],
            "file_path": row["file_path"],
            "species": row["species"],
            "genus": row.get("genus", "unknown"),
            "scan_type": row.get("data_type", "unknown"),
            "height_m": float(row["tree_H"]),
            "height_bin": row["height_bin"],
            "max_points": max_points,
            "seed": seed,
            "skip_hcb": skip_hcb,
            "spine_bins": spine_bins,
        })

    print(f"Processing {len(tasks)} real trees with {num_workers} workers...")
    t0 = time.time()

    if num_workers <= 1:
        results = [_real_tree_worker(t) for t in tqdm(tasks, desc="Real trees")]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(_real_tree_worker, tasks, chunksize=4),
                total=len(tasks), desc="Real trees",
            ))

    elapsed = time.time() - t0
    print(f"  Real tree processing: {elapsed:.1f}s")

    # Separate clouds and build DataFrame
    real_clouds = {}
    rows = []
    errors = 0
    for r in results:
        if r is None or "_error" in r:
            errors += 1
            continue
        real_clouds[r["tree_id"]] = r.pop("_canon_cloud")
        rows.append(r)

    if errors > 0:
        print(f"  {errors} trees failed processing")

    df_real = pd.DataFrame(rows)
    df_real = df_real.set_index("tree_id")
    print(f"  Built df_real: {len(df_real)} trees")
    return df_real, real_clouds


# =============================================================================
# Build df_gen
# =============================================================================


def _gen_tree_worker(task: dict) -> dict | None:
    """Process one generated tree: load, canonicalize, extract metrics."""
    try:
        rng = np.random.default_rng(task.get("seed", 42))
        cloud = load_point_cloud(task["zarr_path"], max_points=task.get("max_points"), rng=rng)
        canon_cloud = canonicalize(cloud)
        height_m = task.get("height_m")
        if height_m is None or height_m <= 0:
            return {"gen_id": task.get("gen_id", "?"), "_error": f"invalid height_m={height_m}"}

        metrics = extract_stem_metrics(
            cloud,
            height_m=height_m,
            skip_hcb=task.get("skip_hcb", False),
            spine_num_bins=task.get("spine_bins", 20),
        )

        result = {
            "gen_id": task["gen_id"],
            "real_id": task["real_id"],
            "gen_idx": task.get("gen_idx", 0),
            "cfg_scale": task.get("cfg_scale", 0.0),
            "species": task.get("species", "unknown"),
            "genus": task.get("genus", "unknown"),
            "scan_type": task.get("scan_type", "unknown"),
            "height_m": task.get("height_m", 0.0),
            "num_points": len(cloud),
            "crown_volume": metrics["crown_volume"],
            "max_crown_r": metrics["max_crown_r"],
            "max_crown_r_rel_s": metrics["max_crown_r_rel_s"],
            "hcb": metrics["hcb"],
            "vert_kde": metrics["vert_kde"].tolist(),
            "hist_2d": metrics["hist_2d"].tolist(),
            "_canon_cloud": canon_cloud,
        }
        return result
    except Exception as e:
        return {"gen_id": task.get("gen_id", "?"), "_error": str(e)}


def build_df_gen(
    gen_metadata: pd.DataFrame,
    zarr_dir: Path,
    max_points: int = 4096,
    num_workers: int = 40,
    seed: int = 42,
    skip_hcb: bool = False,
    spine_bins: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """Build metrics DataFrame for all generated trees.

    Returns (df_gen, gen_clouds) where gen_clouds maps gen_id → canonicalized array.
    """
    tasks = []
    for idx, row in gen_metadata.iterrows():
        sample_file = row["sample_file"]
        gen_id = Path(sample_file).stem
        tasks.append({
            "gen_id": gen_id,
            "real_id": row["source_tree_id"],
            "gen_idx": int(row.get("sample_idx", idx)),
            "cfg_scale": float(row.get("cfg_scale", 0.0)),
            "species": row.get("species", "unknown"),
            "genus": row.get("genus", "unknown"),
            "scan_type": row.get("scan_type", "unknown"),
            "height_m": float(row["height_m"]) if pd.notna(row.get("height_m")) and float(row.get("height_m", 0)) > 0 else None,
            "zarr_path": str(zarr_dir / sample_file),
            "max_points": max_points,
            "seed": seed,
            "skip_hcb": skip_hcb,
            "spine_bins": spine_bins,
        })

    print(f"Processing {len(tasks)} generated trees with {num_workers} workers...")
    t0 = time.time()

    if num_workers <= 1:
        results = [_gen_tree_worker(t) for t in tqdm(tasks, desc="Gen trees")]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(_gen_tree_worker, tasks, chunksize=16),
                total=len(tasks), desc="Gen trees",
            ))

    elapsed = time.time() - t0
    print(f"  Generated tree processing: {elapsed:.1f}s")

    gen_clouds = {}
    rows = []
    errors = 0
    for r in results:
        if r is None or "_error" in r:
            errors += 1
            continue
        gen_clouds[r["gen_id"]] = r.pop("_canon_cloud")
        rows.append(r)

    if errors > 0:
        print(f"  {errors} generated trees failed processing")

    df_gen = pd.DataFrame(rows)
    if not df_gen.empty:
        df_gen = df_gen.set_index("gen_id")
    print(f"  Built df_gen: {len(df_gen)} trees")
    return df_gen, gen_clouds


# =============================================================================
# Build df_pairs
# =============================================================================


def _cd_worker(task: tuple) -> tuple[int, float]:
    """Compute CD for a single (real, gen) pair. Returns (index, cd)."""
    idx, real_cloud, gen_cloud = task
    cd = chamfer_distance(real_cloud, gen_cloud)
    return idx, cd


def build_df_pairs(
    df_real: pd.DataFrame,
    df_gen: pd.DataFrame,
    real_clouds: dict,
    gen_clouds: dict,
    num_workers: int = 40,
) -> pd.DataFrame:
    """Compute pairwise scores for each (real, gen) pair.

    Scalar deltas are vectorized via merge. JSD requires per-row computation
    on array columns. CD is parallelized across workers.
    """
    # Merge gen with real on real_id to align rows
    scalar_cols = ["crown_volume", "max_crown_r", "max_crown_r_rel_s", "hcb"]
    real_scalar = df_real[scalar_cols + ["vert_kde", "hist_2d"]].rename(
        columns={c: f"real_{c}" for c in scalar_cols + ["vert_kde", "hist_2d"]}
    )
    merged = df_gen.join(real_scalar, on="real_id", how="inner")

    if merged.empty:
        print("  Built df_pairs: 0 pairs")
        return pd.DataFrame()

    # Vectorized scalar deltas
    for c in scalar_cols:
        merged[f"delta_{c}"] = (merged[c] - merged[f"real_{c}"]).abs()

    # JSD requires per-row computation (array columns)
    print(f"  Computing JSD for {len(merged)} pairs...")
    vert_kde_jsds = np.empty(len(merged))
    hist_2d_jsds = np.empty(len(merged))
    for i, (_, row) in enumerate(merged.iterrows()):
        vert_kde_jsds[i] = jsd(np.array(row["real_vert_kde"]), np.array(row["vert_kde"]))
        hist_2d_jsds[i] = jsd(np.array(row["real_hist_2d"]), np.array(row["hist_2d"]))

    # Build pairs DataFrame
    gen_ids = merged.index.tolist()
    df_pairs = pd.DataFrame({
        "gen_id": gen_ids,
        "real_id": merged["real_id"].values,
        "cfg_scale": merged["cfg_scale"].values if "cfg_scale" in merged.columns else 0.0,
        "species": merged["species"].values if "species" in merged.columns else "unknown",
        "genus": merged["genus"].values if "genus" in merged.columns else "unknown",
        "scan_type": merged["scan_type"].values if "scan_type" in merged.columns else "unknown",
        "height_m": merged["height_m"].values if "height_m" in merged.columns else 0.0,
        "reconstruction_cd": float("nan"),
        "vert_kde_jsd": vert_kde_jsds,
        "hist_2d_jsd": hist_2d_jsds,
        "delta_crown_vol": merged["delta_crown_volume"].values,
        "delta_max_crown_r": merged["delta_max_crown_r"].values,
        "delta_max_crown_r_rel_s": merged["delta_max_crown_r_rel_s"].values,
        "delta_hcb": merged["delta_hcb"].values,
    })

    # Parallel CD computation
    cd_tasks = []
    for i, (gen_id, real_id) in enumerate(zip(df_pairs["gen_id"], df_pairs["real_id"])):
        if gen_id in gen_clouds and real_id in real_clouds:
            cd_tasks.append((i, real_clouds[real_id], gen_clouds[gen_id]))

    if cd_tasks:
        print(f"  Computing {len(cd_tasks)} Chamfer distances with {num_workers} workers...")
        t0 = time.time()
        if num_workers <= 1:
            cd_results = [_cd_worker(t) for t in tqdm(cd_tasks, desc="CD")]
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                cd_results = list(tqdm(
                    executor.map(_cd_worker, cd_tasks, chunksize=64),
                    total=len(cd_tasks), desc="CD",
                ))
        cd_vals = df_pairs["reconstruction_cd"].values.copy()
        for idx, cd in cd_results:
            cd_vals[idx] = cd
        df_pairs["reconstruction_cd"] = cd_vals
        print(f"  CD computation: {time.time() - t0:.1f}s")

    print(f"  Built df_pairs: {len(df_pairs)} pairs")
    return df_pairs


# =============================================================================
# Build df_per_tree
# =============================================================================


def build_df_per_tree(df_pairs: pd.DataFrame, df_real: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 32 generations per conditioning tree → mean/std."""
    if df_pairs.empty:
        return pd.DataFrame()

    grouped = df_pairs.groupby("real_id")[PAIR_METRICS]
    agg = grouped.agg(["mean", "std"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index().set_index("real_id")

    # Merge stratification from df_real
    strat_cols = ["species", "genus", "scan_type", "height_m", "height_bin"]
    available = [c for c in strat_cols if c in df_real.columns]
    agg = agg.join(df_real[available])

    # Add count
    agg["n_gen"] = grouped.size().values

    print(f"  Built df_per_tree: {len(agg)} trees")
    return agg


# =============================================================================
# Baselines
# =============================================================================


def _build_baseline(
    df_real: pd.DataFrame,
    real_clouds: dict,
    neighbor_map: dict[str, list[str]],
    num_workers: int = 40,
    label: str = "Baseline",
) -> pd.DataFrame:
    """Shared baseline builder: compute JSD/deltas inline, parallelize CD.

    neighbor_map: {tree_id: [neighbor_id, ...]} — pre-computed neighbor assignments.
    """
    real_index = set(df_real.index)
    cloud_keys = set(real_clouds.keys())

    # Collect per-pair JSD/deltas (cheap) and CD tasks (expensive)
    # Structure: pair_data[tree_id] = list of per-neighbor dicts (without CD)
    pair_data: dict[str, list[dict]] = {}
    cd_tasks = []  # (tree_id, neighbor_idx, real_cloud, neighbor_cloud)

    for tid, neighbors in neighbor_map.items():
        if tid not in real_index:
            continue
        anchor = df_real.loc[tid]
        anchor_kde = np.array(anchor["vert_kde"])
        anchor_hist = np.array(anchor["hist_2d"])

        pairs = []
        for nid in neighbors:
            if nid not in real_index:
                continue

            neighbor = df_real.loc[nid]
            pair = {
                "vert_kde_jsd": jsd(anchor_kde, np.array(neighbor["vert_kde"])),
                "hist_2d_jsd": jsd(anchor_hist, np.array(neighbor["hist_2d"])),
                "delta_crown_vol": abs(anchor["crown_volume"] - neighbor["crown_volume"]),
                "delta_max_crown_r": abs(anchor["max_crown_r"] - neighbor["max_crown_r"]),
                "delta_max_crown_r_rel_s": abs(anchor["max_crown_r_rel_s"] - neighbor["max_crown_r_rel_s"]),
                "delta_hcb": abs(anchor["hcb"] - neighbor["hcb"]),
                "reconstruction_cd": float("nan"),
            }
            pairs.append(pair)

            if tid in cloud_keys and nid in cloud_keys:
                cd_tasks.append((len(cd_tasks), tid, len(pairs) - 1,
                                 real_clouds[tid], real_clouds[nid]))

        if pairs:
            pair_data[tid] = pairs

    # Parallel CD computation
    if cd_tasks:
        print(f"  Computing {len(cd_tasks)} baseline CDs with {num_workers} workers...")
        t0 = time.time()

        # Repack for _cd_worker: (index, cloud_a, cloud_b)
        cd_worker_tasks = [(t[0], t[3], t[4]) for t in cd_tasks]

        if num_workers <= 1:
            cd_results = [_cd_worker(t) for t in tqdm(cd_worker_tasks, desc=f"{label} CD")]
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                cd_results = list(tqdm(
                    executor.map(_cd_worker, cd_worker_tasks, chunksize=64),
                    total=len(cd_worker_tasks), desc=f"{label} CD",
                ))

        # Map results back: cd_tasks[task_idx] has (_, tree_id, pair_idx, ...)
        cd_lookup = {task_idx: cd for task_idx, cd in cd_results}
        for task_idx, (_, tid, pair_idx, _, _) in enumerate(cd_tasks):
            if task_idx in cd_lookup:
                pair_data[tid][pair_idx]["reconstruction_cd"] = cd_lookup[task_idx]

        print(f"  {label} CD computation: {time.time() - t0:.1f}s")

    # Aggregate per-tree: mean/std across neighbors
    rows = []
    for tid, pairs in pair_data.items():
        result = {"tree_id": tid}
        for m in PAIR_METRICS:
            vals = [p[m] for p in pairs]
            result[f"{m}_mean"] = float(np.nanmean(vals))
            result[f"{m}_std"] = float(np.nanstd(vals))
        rows.append(result)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("tree_id")
    return df


def build_intra_baseline(
    df_real: pd.DataFrame,
    real_clouds: dict,
    n_neighbors: int = 32,
    num_workers: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """Intra-class baseline: each real tree vs N trees from same (species, height_bin)."""
    rng = np.random.default_rng(seed)

    strat_cols = ["species", "genus", "height_bin"]
    available = [c for c in strat_cols if c in df_real.columns]
    if not available:
        print("  Warning: no stratification columns available for intra baseline")
        return pd.DataFrame()

    # Build neighbor map
    neighbor_map = {}
    grouped = df_real.groupby(available)
    for _, group in grouped:
        tree_ids = list(group.index)
        if len(tree_ids) < 2:
            continue
        for tid in tree_ids:
            others = [t for t in tree_ids if t != tid]
            if not others:
                continue
            if len(others) >= n_neighbors:
                neighbors = rng.choice(others, size=n_neighbors, replace=False).tolist()
            else:
                neighbors = rng.choice(others, size=n_neighbors, replace=True).tolist()
            neighbor_map[tid] = neighbors

    print(f"  Intra baseline: {len(neighbor_map)} trees, {sum(len(v) for v in neighbor_map.values())} pairs")
    df_intra = _build_baseline(df_real, real_clouds, neighbor_map,
                               num_workers=num_workers, label="Intra")

    # Add stratification columns
    if not df_intra.empty:
        for c in available:
            df_intra[c] = df_real.loc[df_intra.index, c]

    print(f"  Built intra baseline: {len(df_intra)} trees")
    return df_intra


def build_inter_baseline(
    df_real: pd.DataFrame,
    real_clouds: dict,
    n_neighbors: int = 32,
    num_workers: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """Inter-class baseline: each real tree vs N trees from different species."""
    rng = np.random.default_rng(seed)

    species_map = df_real["species"].to_dict() if "species" in df_real.columns else {}
    all_ids = list(df_real.index)

    neighbor_map = {}
    for tid in all_ids:
        tree_species = species_map.get(tid, "unknown")
        others = [t for t in all_ids if t != tid and species_map.get(t, "unknown") != tree_species]
        if not others:
            continue
        if len(others) >= n_neighbors:
            neighbors = rng.choice(others, size=n_neighbors, replace=False).tolist()
        else:
            neighbors = rng.choice(others, size=n_neighbors, replace=True).tolist()
        neighbor_map[tid] = neighbors

    print(f"  Inter baseline: {len(neighbor_map)} trees, {sum(len(v) for v in neighbor_map.values())} pairs")
    df_inter = _build_baseline(df_real, real_clouds, neighbor_map,
                               num_workers=num_workers, label="Inter")
    print(f"  Built inter baseline: {len(df_inter)} trees")
    return df_inter


# =============================================================================
# Population metrics (COV / MMD / Voxel JSD)
# =============================================================================


def compute_cd_matrix(
    clouds_a: list[np.ndarray],
    clouds_b: list[np.ndarray],
    num_workers: int = 1,
) -> np.ndarray:
    """Pairwise CD matrix (len_a × len_b)."""
    n_a, n_b = len(clouds_a), len(clouds_b)
    tasks = []
    for i in range(n_a):
        for j in range(n_b):
            tasks.append((i * n_b + j, clouds_a[i], clouds_b[j]))

    if num_workers <= 1 or len(tasks) < 100:
        cd_results = [_cd_worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            cd_results = list(executor.map(_cd_worker, tasks, chunksize=64))

    mat = np.zeros((n_a, n_b))
    for idx, cd in cd_results:
        mat[idx // n_b, idx % n_b] = cd
    return mat


def coverage(dist_matrix: np.ndarray) -> float:
    """Fraction of real trees that are NN of at least one gen tree."""
    nn_of_gen = dist_matrix.argmin(axis=0)  # for each gen, nearest real
    return len(set(nn_of_gen.tolist())) / dist_matrix.shape[0]


def mmd(dist_matrix: np.ndarray) -> float:
    """Mean min-CD from each real tree to nearest gen."""
    return float(dist_matrix.min(axis=1).mean())


def one_nn_accuracy(real_real: np.ndarray, gen_gen: np.ndarray, real_gen: np.ndarray) -> float:
    """1-Nearest Neighbor classification accuracy from three distance sub-matrices.

    Pool real and gen points. For each point, find its NN among all OTHER points.
    If the NN has the same label (real/gen), it's a correct classification.
    50% = distributions are indistinguishable. 100% = completely separable.
    """
    n_r, n_g = real_gen.shape
    correct = 0
    total = n_r + n_g

    for i in range(n_r):
        rr = real_real[i].copy()
        rr[i] = np.inf  # exclude self
        nn_real_dist = rr.min()
        nn_gen_dist = real_gen[i].min()
        if nn_real_dist <= nn_gen_dist:
            correct += 1

    for j in range(n_g):
        gg = gen_gen[j].copy()
        gg[j] = np.inf
        nn_gen_dist = gg.min()
        nn_real_dist = real_gen[:, j].min()
        if nn_gen_dist <= nn_real_dist:
            correct += 1

    return correct / total


def compute_population_metrics(
    df_real: pd.DataFrame,
    df_gen: pd.DataFrame,
    real_clouds: dict,
    gen_clouds: dict,
    max_per_stratum: int = 200,
    max_per_nna: int = 100,
    min_per_stratum: int = 10,
    num_workers: int = 40,
    seed: int = 42,
) -> dict:
    """COV, MMD, 1-NNA, Voxel JSD — computed at atomic (genus, height_bin) strata, then aggregated.

    max_per_stratum: max trees per side for COV/MMD (cross matrix only).
    max_per_nna: max trees per side for 1-NNA (requires rr + gg + cross matrices, ~4× cost).
    """
    rng = np.random.default_rng(seed)
    results = {"by_genus_height": {}, "by_genus": {}, "by_height_bin": {}, "global": None}

    def _subsample_ids(ids, max_n):
        if len(ids) <= max_n:
            return ids
        return rng.choice(ids, size=max_n, replace=False).tolist()

    # For gen, pick one per real tree to avoid duplicates dominating
    gen_by_real = df_gen.groupby("real_id").nth(0)
    gen_real_id_map = gen_by_real["real_id"].to_dict() if "real_id" in gen_by_real.columns else {}
    if not gen_real_id_map:
        raise ValueError("gen_by_real has no 'real_id' column — cannot map gen trees to real trees")
    gen_by_real_ids = set(gen_by_real.index)

    # Build lookup: real_id → gen_id (first gen per real tree)
    real_to_gen = {}
    for gid in gen_by_real_ids:
        rid = gen_real_id_map.get(gid)
        if rid is not None:
            real_to_gen[rid] = gid

    # Step 1: Compute atomic (genus, height_bin) strata
    if "genus" not in df_real.columns or "height_bin" not in df_real.columns:
        print("  Warning: missing genus or height_bin columns for population metrics")
        return results

    atomic_results = {}
    groups = list(df_real.groupby(["genus", "height_bin"]))
    print(f"  Computing population metrics for {len(groups)} (genus, height_bin) strata...")

    for (genus, hb), grp in tqdm(groups, desc="Pop strata"):
        real_ids = [i for i in grp.index if i in real_clouds]
        gen_ids = [real_to_gen[i] for i in real_ids if i in real_to_gen and real_to_gen[i] in gen_clouds]
        real_ids = [i for i in real_ids if i in real_to_gen and real_to_gen[i] in gen_clouds]

        if len(real_ids) < min_per_stratum or len(gen_ids) < min_per_stratum:
            print(f"    ({genus}, {hb}): skipped (n_real={len(real_ids)}, n_gen={len(gen_ids)})")
            continue

        real_ids = _subsample_ids(real_ids, max_per_stratum)
        gen_ids = [real_to_gen[i] for i in real_ids if real_to_gen[i] in gen_clouds]
        real_ids = [i for i in real_ids if real_to_gen.get(i) in gen_clouds]

        if len(real_ids) < min_per_stratum or len(gen_ids) < min_per_stratum:
            continue

        label = f"({genus}, {hb})"
        n_r, n_g = len(real_ids), len(gen_ids)
        print(f"    {label}: {n_r} real × {n_g} gen")

        rc = [real_clouds[i] for i in real_ids]
        gc = [gen_clouds[i] for i in gen_ids]

        # Cross matrix (real × gen) — used for COV, MMD
        cross_mat = compute_cd_matrix(rc, gc, num_workers=num_workers)
        cov = coverage(cross_mat)
        mmd_val = mmd(cross_mat)

        # 1-NNA on subsampled sets (rr + gg + cross matrices are expensive)
        nna_n_r = min(n_r, max_per_nna)
        nna_n_g = min(n_g, max_per_nna)
        if nna_n_r < n_r or nna_n_g < n_g:
            nna_r_idx = sorted(rng.choice(n_r, size=nna_n_r, replace=False).tolist())
            nna_g_idx = sorted(rng.choice(n_g, size=nna_n_g, replace=False).tolist())
            nna_rc = [rc[i] for i in nna_r_idx]
            nna_gc = [gc[i] for i in nna_g_idx]
            nna_cross = compute_cd_matrix(nna_rc, nna_gc, num_workers=num_workers)
        else:
            nna_rc, nna_gc, nna_cross = rc, gc, cross_mat
        rr_mat = compute_cd_matrix(nna_rc, nna_rc, num_workers=num_workers)
        gg_mat = compute_cd_matrix(nna_gc, nna_gc, num_workers=num_workers)
        nna = one_nn_accuracy(rr_mat, gg_mat, nna_cross)

        # Diversity diagnostic: gen-gen vs real-real pairwise CD
        rr_upper = rr_mat[np.triu_indices_from(rr_mat, k=1)]
        gg_upper = gg_mat[np.triu_indices_from(gg_mat, k=1)]
        real_real_cd = float(rr_upper.mean()) if len(rr_upper) > 0 else float("nan")
        gen_gen_cd = float(gg_upper.mean()) if len(gg_upper) > 0 else float("nan")
        diversity_ratio = gen_gen_cd / real_real_cd if real_real_cd > 0 else float("nan")

        voxel_jsd = compute_jsd_3d(rc, gc)

        print(f"    {label}: COV={cov:.4f}, MMD={mmd_val:.6f}, 1-NNA={nna:.4f}, "
              f"Div={diversity_ratio:.4f}, JSD={voxel_jsd:.6f}")

        key = f"{genus}|{hb}"
        atomic_results[key] = {
            "genus": genus, "height_bin": hb,
            "coverage": cov, "mmd": mmd_val, "one_nna": nna,
            "diversity_ratio": diversity_ratio,
            "real_real_cd": real_real_cd, "gen_gen_cd": gen_gen_cd,
            "voxel_jsd": voxel_jsd,
            "n_real": n_r, "n_gen": n_g,
        }

    results["by_genus_height"] = atomic_results

    if not atomic_results:
        print("  Warning: no valid (genus, height_bin) strata for population metrics")
        return results

    # Step 2: Aggregate by genus (weighted mean by n_real)
    genus_groups: dict[str, list] = {}
    for v in atomic_results.values():
        genus_groups.setdefault(v["genus"], []).append(v)

    for g, items in genus_groups.items():
        weights = np.array([it["n_real"] for it in items], dtype=np.float64)
        w_sum = weights.sum()
        results["by_genus"][g] = {
            "coverage": float(np.average([it["coverage"] for it in items], weights=weights)),
            "mmd": float(np.average([it["mmd"] for it in items], weights=weights)),
            "one_nna": float(np.average([it["one_nna"] for it in items], weights=weights)),
            "diversity_ratio": float(np.average([it["diversity_ratio"] for it in items], weights=weights)),
            "voxel_jsd": float(np.average([it["voxel_jsd"] for it in items], weights=weights)),
            "n_real": int(w_sum),
            "n_strata": len(items),
        }

    # Step 3: Aggregate by height_bin
    hb_groups: dict[str, list] = {}
    for v in atomic_results.values():
        hb_groups.setdefault(v["height_bin"], []).append(v)

    for hb, items in hb_groups.items():
        weights = np.array([it["n_real"] for it in items], dtype=np.float64)
        w_sum = weights.sum()
        results["by_height_bin"][str(hb)] = {
            "coverage": float(np.average([it["coverage"] for it in items], weights=weights)),
            "mmd": float(np.average([it["mmd"] for it in items], weights=weights)),
            "one_nna": float(np.average([it["one_nna"] for it in items], weights=weights)),
            "diversity_ratio": float(np.average([it["diversity_ratio"] for it in items], weights=weights)),
            "voxel_jsd": float(np.average([it["voxel_jsd"] for it in items], weights=weights)),
            "n_real": int(w_sum),
            "n_strata": len(items),
        }

    # Step 4: Global aggregate
    all_items = list(atomic_results.values())
    weights = np.array([it["n_real"] for it in all_items], dtype=np.float64)
    w_sum = weights.sum()
    results["global"] = {
        "coverage": float(np.average([it["coverage"] for it in all_items], weights=weights)),
        "mmd": float(np.average([it["mmd"] for it in all_items], weights=weights)),
        "one_nna": float(np.average([it["one_nna"] for it in all_items], weights=weights)),
        "diversity_ratio": float(np.average([it["diversity_ratio"] for it in all_items], weights=weights)),
        "voxel_jsd": float(np.average([it["voxel_jsd"] for it in all_items], weights=weights)),
        "n_real": int(w_sum),
        "n_strata": len(all_items),
    }

    glob = results["global"]
    print(f"  Global — COV: {glob['coverage']:.4f}, MMD: {glob['mmd']:.6f}, "
          f"1-NNA: {glob['one_nna']:.4f}, Div: {glob['diversity_ratio']:.4f}, "
          f"Voxel JSD: {glob['voxel_jsd']:.6f} "
          f"({glob['n_strata']} strata, {glob['n_real']} trees)")

    return results


# =============================================================================
# Statistical tests
# =============================================================================


def compute_wasserstein_clipped(gen_scores, intra_scores, clip_percentile=99):
    """Wasserstein distance with percentile clipping for paired-delta robustness.

    Primary outlier removal uses hard physical caps (50,000 m³ / 25 m) to catch
    stem tracker failures. This secondary P99 clip on the pooled gen+intra
    paired-delta distribution handles residual heavy tails in the delta scores.
    """
    combined = np.concatenate([gen_scores, intra_scores])
    cap = np.percentile(combined, clip_percentile)
    gen_clipped = np.clip(gen_scores, 0, cap)
    intra_clipped = np.clip(intra_scores, 0, cap)
    return float(wasserstein_distance(gen_clipped, intra_clipped))


def compute_statistical_tests(
    df_per_tree: pd.DataFrame,
    df_intra: pd.DataFrame,
) -> dict:
    """Wasserstein distance per metric (gen vs intra), with P99 clipping."""
    results = {}

    for m in PAIR_METRICS:
        mean_col = f"{m}_mean"
        if mean_col not in df_per_tree.columns or mean_col not in df_intra.columns:
            continue

        gen_scores = df_per_tree[mean_col].dropna().values
        intra_scores = df_intra[mean_col].dropna().values

        if len(gen_scores) < 2 or len(intra_scores) < 2:
            continue

        results[m] = {
            "wasserstein": compute_wasserstein_clipped(gen_scores, intra_scores),
        }

    return results


# =============================================================================
# Ratios
# =============================================================================


def compute_ratios(
    df_per_tree: pd.DataFrame,
    df_intra: pd.DataFrame,
    df_inter: pd.DataFrame,
    group_by: str | None = None,
) -> pd.DataFrame:
    """ratio = median(gen_vs_real) / median(intra_class) per metric, optionally by stratum."""
    rows = []

    def _ratio_row(gen_df, intra_df, inter_df, label="global"):
        row = {"stratum": label}
        for m in PAIR_METRICS:
            mean_col = f"{m}_mean"
            gen_med = gen_df[mean_col].median() if mean_col in gen_df.columns else float("nan")
            intra_med = intra_df[mean_col].median() if mean_col in intra_df.columns else float("nan")
            inter_med = inter_df[mean_col].median() if mean_col in inter_df.columns else float("nan")
            row[f"{m}_gen"] = float(gen_med)
            row[f"{m}_intra"] = float(intra_med)
            row[f"{m}_inter"] = float(inter_med)
            row[f"{m}_ratio"] = float(gen_med / intra_med) if intra_med > 0 else float("nan")
        return row

    if group_by is None or group_by not in df_per_tree.columns:
        rows.append(_ratio_row(df_per_tree, df_intra, df_inter, "global"))
    else:
        for val, gen_grp in df_per_tree.groupby(group_by):
            intra_grp = df_intra[df_intra[group_by] == val] if group_by in df_intra.columns else df_intra.iloc[:0]
            inter_grp = df_inter  # inter has no stratum match
            rows.append(_ratio_row(gen_grp, intra_grp, inter_grp, str(val)))

    return pd.DataFrame(rows)


# =============================================================================
# Scan-type confound diagnostic
# =============================================================================


def compute_scan_type_diagnostic(
    df_real: pd.DataFrame,
    real_clouds: dict,
    num_workers: int = 40,
    seed: int = 42,
    max_pairs_per_stratum: int = 200,
) -> dict:
    """One-off diagnostic: does mixing scan types inflate intra distances?

    For strata with both TLS and ULS trees, compares within-scan-type
    CD to cross-scan-type CD. If cross/within ratio is < 1.1, scan type
    mixing has negligible effect on baselines.
    """
    rng = np.random.default_rng(seed)
    results = {"strata": [], "summary": {}}

    if "scan_type" not in df_real.columns:
        print("  Scan diagnostic: no scan_type column, skipping")
        return results

    groups = df_real.groupby(["species", "height_bin"])
    qualifying_strata = 0
    all_ratios = []

    for (species, hb), grp in groups:
        scan_types = grp["scan_type"].value_counts()
        # Need both TLS and ULS with >= 5 trees each
        if len(scan_types) < 2:
            continue
        eligible_types = [st for st, n in scan_types.items() if n >= 5]
        if len(eligible_types) < 2:
            continue

        qualifying_strata += 1
        type_ids = {}
        for st in eligible_types:
            ids = [i for i in grp[grp["scan_type"] == st].index if i in real_clouds]
            type_ids[st] = ids

        # Within-type pairs
        within_cd_vals = []
        for st, ids in type_ids.items():
            if len(ids) < 2:
                continue
            n_pairs = min(max_pairs_per_stratum // len(eligible_types), len(ids) * (len(ids) - 1) // 2)
            pairs_sampled = 0
            cd_tasks = []
            attempts = 0
            while pairs_sampled < n_pairs and attempts < n_pairs * 3:
                i, j = rng.choice(len(ids), size=2, replace=False)
                cd_tasks.append((pairs_sampled, real_clouds[ids[i]], real_clouds[ids[j]]))
                pairs_sampled += 1
                attempts += 1
            if cd_tasks:
                if num_workers <= 1 or len(cd_tasks) < 50:
                    cd_results = [_cd_worker(t) for t in cd_tasks]
                else:
                    with ProcessPoolExecutor(max_workers=num_workers) as executor:
                        cd_results = list(executor.map(_cd_worker, cd_tasks, chunksize=16))
                within_cd_vals.extend([cd for _, cd in cd_results])

        # Cross-type pairs
        cross_cd_vals = []
        types_list = list(type_ids.keys())
        if len(types_list) >= 2:
            ids_a = type_ids[types_list[0]]
            ids_b = type_ids[types_list[1]]
            n_pairs = min(max_pairs_per_stratum, len(ids_a) * len(ids_b))
            cd_tasks = []
            for p in range(n_pairs):
                i = rng.integers(len(ids_a))
                j = rng.integers(len(ids_b))
                cd_tasks.append((p, real_clouds[ids_a[i]], real_clouds[ids_b[j]]))
            if cd_tasks:
                if num_workers <= 1 or len(cd_tasks) < 50:
                    cd_results = [_cd_worker(t) for t in cd_tasks]
                else:
                    with ProcessPoolExecutor(max_workers=num_workers) as executor:
                        cd_results = list(executor.map(_cd_worker, cd_tasks, chunksize=16))
                cross_cd_vals.extend([cd for _, cd in cd_results])

        if within_cd_vals and cross_cd_vals:
            within_mean = float(np.mean(within_cd_vals))
            cross_mean = float(np.mean(cross_cd_vals))
            ratio = cross_mean / within_mean if within_mean > 0 else float("nan")
            all_ratios.append(ratio)
            results["strata"].append({
                "species": species, "height_bin": hb,
                "scan_types": eligible_types,
                "within_cd": within_mean,
                "cross_cd": cross_mean,
                "ratio": ratio,
                "n_within_pairs": len(within_cd_vals),
                "n_cross_pairs": len(cross_cd_vals),
            })

    if all_ratios:
        median_ratio = float(np.median(all_ratios))
        results["summary"] = {
            "n_qualifying_strata": qualifying_strata,
            "n_tested_strata": len(all_ratios),
            "median_cross_within_ratio": median_ratio,
            "mean_cross_within_ratio": float(np.mean(all_ratios)),
        }
        print(f"  Scan-type diagnostic: {len(all_ratios)} strata tested, "
              f"median cross/within ratio = {median_ratio:.4f}")
    else:
        print("  Scan-type diagnostic: no qualifying strata found")

    return results


# =============================================================================
# Morphological Wasserstein (population-level marginal comparison)
# =============================================================================


def compute_morphological_wasserstein(
    df_real: pd.DataFrame,
    df_gen: pd.DataFrame,
    stratum_cols: list[str] | None = None,
    properties: list[str] = ("crown_volume", "max_crown_r", "hcb"),
) -> pd.DataFrame:
    """W1 between gen and real marginal distributions per stratum.

    For each stratum (e.g., species x height_bin) and each morphological
    property, computes the 1-Wasserstein distance between the generated
    and real marginal distributions. This answers: "does the generated
    population in this stratum have the right distribution of crown
    volumes?" — a direct population-level test.

    If stratum_cols is None, computes a single global row.
    Returns a DataFrame with columns: [*stratum_cols, property, wasserstein].
    """
    rows = []

    if stratum_cols is None:
        # Global: single stratum
        for prop in properties:
            vals_real = df_real[prop].dropna().values
            vals_gen = df_gen[prop].dropna().values
            if len(vals_real) >= 5 and len(vals_gen) >= 5:
                w1 = wasserstein_distance(vals_real, vals_gen)
                rows.append({"property": prop, "wasserstein": float(w1)})
        return pd.DataFrame(rows)

    for keys, grp_real in df_real.groupby(stratum_cols):
        try:
            grp_gen = df_gen.groupby(stratum_cols).get_group(keys)
        except KeyError:
            continue
        for prop in properties:
            vals_real = grp_real[prop].dropna().values
            vals_gen = grp_gen[prop].dropna().values
            if len(vals_real) >= 5 and len(vals_gen) >= 5:
                w1 = wasserstein_distance(vals_real, vals_gen)
                row = dict(zip(stratum_cols, keys if isinstance(keys, tuple) else (keys,)))
                row["property"] = prop
                row["wasserstein"] = float(w1)
                rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Output tables
# =============================================================================


def build_table_1a(
    population_metrics: dict,
    morph_w1_global: pd.DataFrame,
) -> pd.DataFrame:
    """Table 1(a): Population metrics — point-cloud distribution + morphological marginals."""
    rows = []
    glob = population_metrics.get("global") or {}

    # Point-cloud distribution metrics
    pop_metrics = [
        ("coverage", "Coverage", 0.63),
        ("mmd", "MMD (CD)", 0),
        ("one_nna", "1-NNA", 0.50),
        ("diversity_ratio", "Diversity ratio", 1.00),
        ("voxel_jsd", "Voxel JSD", 0),
    ]
    for key, name, ideal in pop_metrics:
        rows.append({
            "section": "Point-cloud distribution",
            "metric": key,
            "display_name": name,
            "value": glob.get(key, float("nan")),
            "ideal": ideal,
        })

    # Morphological marginals (W1)
    morph_props = [
        ("crown_volume", "Crown volume (m\u00b3)"),
        ("max_crown_r", "Max crown radius (m)"),
        ("hcb", "Height to crown base (m)"),
    ]
    for prop, name in morph_props:
        w1_val = float("nan")
        if not morph_w1_global.empty:
            match = morph_w1_global[morph_w1_global["property"] == prop]
            if not match.empty:
                w1_val = float(match.iloc[0]["wasserstein"])
        rows.append({
            "section": "Morphological marginals (W\u2081)",
            "metric": f"w1_{prop}",
            "display_name": name,
            "value": w1_val,
            "ideal": 0,
        })

    return pd.DataFrame(rows)


def build_table_1b(
    df_per_tree: pd.DataFrame,
    df_intra: pd.DataFrame,
    df_inter: pd.DataFrame,
) -> pd.DataFrame:
    """Table 1(b): Conditioning fidelity — stratum medians with Gen/Intra/Inter/Ratio."""
    rows = []
    for m in PAIR_METRICS:
        mean_col = f"{m}_mean"
        gen_med = df_per_tree[mean_col].median() if mean_col in df_per_tree.columns else float("nan")
        intra_med = df_intra[mean_col].median() if mean_col in df_intra.columns else float("nan")
        inter_med = df_inter[mean_col].median() if mean_col in df_inter.columns else float("nan")
        ratio = gen_med / intra_med if pd.notna(intra_med) and intra_med > 0 else float("nan")

        display = METRIC_DISPLAY.get(m, m)
        if isinstance(display, tuple):
            display_name = f"{display[0]} ({display[1]})" if display[1] else display[0]
        else:
            display_name = display

        rows.append({
            "metric": m,
            "display_name": display_name,
            "gen": gen_med,
            "intra": intra_med,
            "inter": inter_med,
            "ratio": ratio,
        })
    return pd.DataFrame(rows)


def _build_stratified_table(
    df_per_tree: pd.DataFrame,
    df_intra: pd.DataFrame,
    df_inter: pd.DataFrame,
    group_col: str,
    population_metrics: dict | None = None,
    morph_w1: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build stratified table with population + conditioning fidelity columns.

    Population side: COV, 1-NNA, Diversity ratio, W1 crown_volume, W1 hcb.
    Conditioning side: CD ratio, Vert KDE ratio, Hist 2D ratio.
    """
    rows = []
    groups = sorted(df_per_tree[group_col].dropna().unique())

    # Map group_col to population_metrics key
    pop_key_map = {"genus": "by_genus", "height_bin": "by_height_bin"}
    pop_dict = {}
    if population_metrics and pop_key_map.get(group_col) in population_metrics:
        pop_dict = population_metrics[pop_key_map[group_col]]

    # Map morph W1 by group
    morph_w1_map = {}
    if morph_w1 is not None and not morph_w1.empty and group_col in morph_w1.columns:
        for g_val, grp in morph_w1.groupby(group_col):
            morph_w1_map[g_val] = {r["property"]: r["wasserstein"] for _, r in grp.iterrows()}

    for g in groups:
        gen_grp = df_per_tree[df_per_tree[group_col] == g]
        intra_grp = df_intra[df_intra[group_col] == g] if group_col in df_intra.columns else df_intra.iloc[:0]
        n_trees = len(gen_grp)

        row = {group_col: g, "n_trees": n_trees}

        # Population metrics for this stratum
        pop = pop_dict.get(str(g), {})
        row["coverage"] = pop.get("coverage", float("nan"))
        row["one_nna"] = pop.get("one_nna", float("nan"))
        row["diversity_ratio"] = pop.get("diversity_ratio", float("nan"))

        # Morphological W1
        mw1 = morph_w1_map.get(g, {})
        row["w1_crown_volume"] = mw1.get("crown_volume", float("nan"))
        row["w1_hcb"] = mw1.get("hcb", float("nan"))

        # Conditioning fidelity ratios (CD, Vert KDE, Hist 2D only)
        cond_metrics = ["reconstruction_cd", "vert_kde_jsd", "hist_2d_jsd"]
        for m in cond_metrics:
            mean_col = f"{m}_mean"
            gen_med = gen_grp[mean_col].median() if mean_col in gen_grp.columns else float("nan")
            intra_med = intra_grp[mean_col].median() if mean_col in intra_grp.columns else float("nan")
            ratio = gen_med / intra_med if pd.notna(intra_med) and intra_med > 0 else float("nan")
            row[f"{m}_ratio"] = ratio

        rows.append(row)

    return pd.DataFrame(rows)


def build_table_b(df_per_tree, df_intra, df_inter, population_metrics=None, morph_w1=None):
    """Table 2: By genus."""
    return _build_stratified_table(df_per_tree, df_intra, df_inter, "genus",
                                   population_metrics=population_metrics, morph_w1=morph_w1)


def build_table_c(df_per_tree, df_intra, df_inter, population_metrics=None, morph_w1=None):
    """Table 3: By Height Bin."""
    return _build_stratified_table(df_per_tree, df_intra, df_inter, "height_bin",
                                   population_metrics=population_metrics, morph_w1=morph_w1)


def build_table_e(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """Table E: By CFG Scale (pair-level)."""
    if "cfg_scale" not in df_pairs.columns:
        return pd.DataFrame()

    cfg_bins = np.arange(1.0, 5.5, 0.5)
    cfg_labels = [f"{b:.1f}-{b + 0.5:.1f}" for b in cfg_bins[:-1]]
    df = df_pairs.copy()
    df["cfg_bin"] = pd.cut(df["cfg_scale"], bins=cfg_bins, labels=cfg_labels, right=False)

    rows = []
    for label, grp in df.groupby("cfg_bin", observed=True):
        row = {"cfg_bin": label, "n_pairs": len(grp)}
        for m in PAIR_METRICS:
            if m in grp.columns:
                row[f"{m}_mean"] = grp[m].mean()
                row[f"{m}_median"] = grp[m].median()
                row[f"{m}_std"] = grp[m].std()
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Save results
# =============================================================================


def save_results(
    output_dir: Path,
    df_real: pd.DataFrame,
    df_gen: pd.DataFrame,
    df_pairs: pd.DataFrame,
    df_per_tree: pd.DataFrame,
    df_intra: pd.DataFrame,
    df_inter: pd.DataFrame,
    population_metrics: dict,
    stat_tests: dict,
    tables: dict,
    outlier_clipping_info: dict | None = None,
):
    """Save all evaluation results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # DataFrames with array columns → parquet
    df_real.to_parquet(output_dir / "df_real.parquet")
    print(f"  Saved df_real.parquet ({len(df_real)} rows)")

    df_gen.to_parquet(output_dir / "df_gen.parquet")
    print(f"  Saved df_gen.parquet ({len(df_gen)} rows)")

    # Pairs and aggregates → CSV
    df_pairs.to_csv(output_dir / "df_pairs.csv", index=False)
    print(f"  Saved df_pairs.csv ({len(df_pairs)} rows)")

    df_per_tree.to_csv(output_dir / "df_per_tree.csv")
    print(f"  Saved df_per_tree.csv ({len(df_per_tree)} rows)")

    df_intra.to_csv(output_dir / "baselines_intra.csv")
    print(f"  Saved baselines_intra.csv ({len(df_intra)} rows)")

    df_inter.to_csv(output_dir / "baselines_inter.csv")
    print(f"  Saved baselines_inter.csv ({len(df_inter)} rows)")

    # JSON outputs
    with open(output_dir / "population_metrics.json", "w") as f:
        json.dump(population_metrics, f, indent=2, default=str)
    print("  Saved population_metrics.json")

    with open(output_dir / "statistical_tests.json", "w") as f:
        json.dump(stat_tests, f, indent=2)
    print("  Saved statistical_tests.json")

    # Tables
    for name, df in tables.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(tables_dir / f"{name}.csv", index=False)
            print(f"  Saved tables/{name}.csv")

    # Summary JSON
    summary = {
        "eval_version": EVAL_VERSION,
        "n_real": len(df_real),
        "n_gen": len(df_gen),
        "n_pairs": len(df_pairs),
        "n_trees": len(df_per_tree),
        "population_metrics": population_metrics,
        "statistical_tests": stat_tests,
    }
    if outlier_clipping_info:
        summary["outlier_clipping"] = outlier_clipping_info
    with open(output_dir / "evaluation_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("  Saved evaluation_v2_summary.json")


# =============================================================================
# Print results
# =============================================================================


def print_results(
    df_per_tree: pd.DataFrame,
    df_intra: pd.DataFrame,
    df_inter: pd.DataFrame,
    population_metrics: dict,
    stat_tests: dict,
    tables: dict,
):
    """Print formatted evaluation results."""
    print("\n" + "=" * 80)
    print("EVALUATION v2 RESULTS")
    print("=" * 80)

    # ── Table 1(a): Population metrics ──
    table_1a = tables.get("table_1a")
    if table_1a is not None and not table_1a.empty:
        print("\nTABLE 1: GLOBAL EVALUATION SUMMARY")
        print("\n(a) Population metrics")
        print("\u2500" * 60)
        header = f"  {'':>34s} {'Value':>10s}  {'Ideal':>8s}"
        print(header)
        print("\u2500" * 60)
        prev_section = None
        for _, row in table_1a.iterrows():
            if row["section"] != prev_section:
                print(f"  {row['section']}")
                prev_section = row["section"]
            val = f"{row['value']:.4f}" if pd.notna(row["value"]) else "--"
            ideal = row["ideal"]
            if ideal == 0.63:
                ideal_s = "0.63\u2020"
            elif ideal == 0:
                ideal_s = "0"
            else:
                ideal_s = f"{ideal:.2f}"
            print(f"    {row['display_name']:<30s} {val:>10s}  {ideal_s:>8s}")
        print("\u2500" * 60)

    # ── Table 1(b): Conditioning fidelity ──
    table_1b = tables.get("table_1b")
    if table_1b is not None and not table_1b.empty:
        print("\n(b) Conditioning fidelity \u2014 stratum medians")
        print("\u2500" * 72)
        header = f"  {'Metric':<32s} {'Gen':>8s} {'Intra':>8s} {'Inter':>8s} {'G/I':>6s}"
        print(header)
        print("\u2500" * 72)
        for _, row in table_1b.iterrows():
            gen_v = f"{row['gen']:.4f}" if pd.notna(row['gen']) else "--"
            intra_v = f"{row['intra']:.4f}" if pd.notna(row['intra']) else "--"
            inter_v = f"{row['inter']:.4f}" if pd.notna(row['inter']) else "--"
            ratio_v = f"{row['ratio']:.2f}" if pd.notna(row['ratio']) else "--"
            print(f"  {row['display_name']:<32s} {gen_v:>8s} {intra_v:>8s} {inter_v:>8s} {ratio_v:>6s}")
        print("\u2500" * 72)

    # ── Table 2 (by genus) ──
    table_b = tables.get("table_b_genus")
    if table_b is not None and not table_b.empty:
        _print_stratified_table(table_b, "TABLE 2: BY GENUS", "genus")

    # ── Table 3 (by height bin) ──
    table_c = tables.get("table_c_height")
    if table_c is not None and not table_c.empty:
        _print_stratified_table(table_c, "TABLE 3: BY HEIGHT BIN", "height_bin")

    # ── Interpretive flags (Change 5) ──
    if table_1b is not None and not table_1b.empty:
        flags = []
        for _, row in table_1b.iterrows():
            if pd.isna(row.get("ratio")):
                continue
            r = row["ratio"]
            m = row["display_name"]
            if r < 0.75:
                flags.append(f"  * {m}: ratio={r:.2f} — gen substantially below intra. "
                             f"May indicate reduced diversity in this metric.")
            elif r > 1.20:
                flags.append(f"  * {m}: ratio={r:.2f} — gen exceeds intra. "
                             f"Model struggles with this metric.")
        if flags:
            print("\nInterpretation notes:")
            for f in flags:
                print(f)

    # ── Synthesis (Change 6) ──
    glob = population_metrics.get("global", {})
    cov = glob.get("coverage", float("nan"))
    nna = glob.get("one_nna", float("nan"))
    div_r = glob.get("diversity_ratio", float("nan"))

    ratios = []
    if table_1b is not None and not table_1b.empty:
        ratios = [row["ratio"] for _, row in table_1b.iterrows()
                  if pd.notna(row.get("ratio"))]
    median_ratio = float(np.median(ratios)) if ratios else float("nan")

    print("\n" + "=" * 80)
    print("SYNTHESIS")
    print("=" * 80)
    print(f"  Median conditioning fidelity ratio: {median_ratio:.2f}")
    print(f"  Population coverage: {cov:.2f} (ideal ~ 0.63)")
    print(f"  1-NNA accuracy: {nna:.2f} (ideal = 0.50)")
    print(f"  Diversity ratio: {div_r:.2f} (ideal = 1.00)")
    for prop in ["crown_volume", "max_crown_r", "hcb"]:
        w1_key = f"w1_{prop}"
        w1_val = glob.get(w1_key, float("nan"))
        if pd.notna(w1_val):
            print(f"  W\u2081 {prop}: {w1_val:.2f}")

    if pd.notna(median_ratio) and pd.notna(cov):
        if median_ratio < 1.0 and cov < 0.55:
            print(f"\n  Assessment: Model captures class-conditional structure "
                  f"(median ratio {median_ratio:.2f}) but underrepresents")
            print(f"  morphological diversity (COV {cov:.2f}, diversity ratio {div_r:.2f}).")
            print(f"  Generated trees are plausible but less variable than real populations.")
        elif median_ratio < 1.0 and cov >= 0.55:
            print(f"\n  Assessment: Model faithfully captures both structure and diversity.")
        else:
            print(f"\n  Assessment: Model shows conditioning-dependent quality variation.")
            print(f"  See per-stratum tables for detailed breakdown.")

    print("\n" + "=" * 80)


def _print_stratified_table(df: pd.DataFrame, title: str, key_col: str):
    """Print a stratified table with population + conditioning columns."""
    w1 = "W\u2081"
    sep = "\u2500" * 110
    print(f"\n{title}")
    print(sep)
    # Header
    w1_crv = f"{w1} CrV"
    w1_hcb_h = f"{w1} HCB"
    w1_section = f"--- {w1} ---"
    header = (f"  {key_col:<15s} {'n':>4s} "
              f"{'COV':>6s} {'1-NNA':>6s} {'Div.':>6s} "
              f"{w1_crv:>8s} {w1_hcb_h:>8s} "
              f"{'CD':>6s} {'Vert.':>6s} {'Hist.':>6s}")
    print(header)
    print(f"  {'':>15s} {'':>4s} "
          f"{'--- Population ---':^20s} "
          f"{w1_section:^18s} "
          f"{'-- G/I ratio --':^20s}")
    print(sep)

    for _, row in df.iterrows():
        n = int(row.get("n_trees", 0))
        cov = f"{row['coverage']:.2f}" if pd.notna(row.get('coverage')) else "--"
        nna = f"{row['one_nna']:.2f}" if pd.notna(row.get('one_nna')) else "--"
        div_r = f"{row['diversity_ratio']:.2f}" if pd.notna(row.get('diversity_ratio')) else "--"
        w1_cv = f"{row['w1_crown_volume']:.1f}" if pd.notna(row.get('w1_crown_volume')) else "--"
        w1_hcb = f"{row['w1_hcb']:.2f}" if pd.notna(row.get('w1_hcb')) else "--"
        cd_r = f"{row['reconstruction_cd_ratio']:.2f}" if pd.notna(row.get('reconstruction_cd_ratio')) else "--"
        vert_r = f"{row['vert_kde_jsd_ratio']:.2f}" if pd.notna(row.get('vert_kde_jsd_ratio')) else "--"
        hist_r = f"{row['hist_2d_jsd_ratio']:.2f}" if pd.notna(row.get('hist_2d_jsd_ratio')) else "--"

        label = str(row[key_col])[:15]
        print(f"  {label:<15s} {n:>4d} "
              f"{cov:>6s} {nna:>6s} {div_r:>6s} "
              f"{w1_cv:>8s} {w1_hcb:>8s} "
              f"{cd_r:>6s} {vert_r:>6s} {hist_r:>6s}")

    print("\u2500" * 110)


# =============================================================================
# CLI and main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="TreeFlow evaluation v2: stem-tracker morphological metrics"
    )
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/preprocessed-4096")
    parser.add_argument("--experiments_dir", type=str, default="experiments")
    parser.add_argument("--max_points", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_layer1", action="store_true",
                        help="Skip population metrics (COV/MMD)")
    parser.add_argument("--skip_hcb", action="store_true",
                        help="Skip HCB computation")
    parser.add_argument("--spine_bins", type=int, default=20,
                        help="Stem tracker height bins")
    parser.add_argument("--resume", action="store_true",
                        help="Skip metric extraction if parquet files exist")
    parser.add_argument("--scan_diagnostic", action="store_true",
                        help="Run scan-type confound diagnostic (one-off)")

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

    output_dir = experiment_dir / "samples" / "evaluation_v2"

    print(f"Experiment: {args.experiment_name}")
    print(f"Data path: {data_path}")
    print(f"Generated samples: {zarr_dir}")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.num_workers}")
    print(f"Skip Layer1: {args.skip_layer1}, Skip HCB: {args.skip_hcb}")
    print(f"Resume: {args.resume}")
    print()

    t_start = time.time()

    # =========================================================================
    # 1. Load metadata
    # =========================================================================
    print("=" * 60)
    print("LOADING METADATA")
    print("=" * 60)

    real_metadata = load_real_metadata(data_path)
    gen_metadata = load_generated_metadata(experiment_dir, real_metadata)

    # =========================================================================
    # 2-3. Build df_real and df_gen (or resume)
    # =========================================================================
    real_parquet = output_dir / "df_real.parquet"
    gen_parquet = output_dir / "df_gen.parquet"

    # Check resume compatibility via eval version in summary JSON
    summary_json = output_dir / "evaluation_v2_summary.json"
    can_resume = False
    if args.resume and real_parquet.exists() and gen_parquet.exists():
        if summary_json.exists():
            try:
                with open(summary_json) as f:
                    prev = json.load(f)
                if prev.get("eval_version") == EVAL_VERSION:
                    can_resume = True
                else:
                    print(f"  Resume skipped: version mismatch (saved={prev.get('eval_version')}, current={EVAL_VERSION})")
            except Exception:
                print("  Resume skipped: could not read summary JSON")
        else:
            print("  Resume skipped: no summary JSON found (fresh run required for metric-space values)")

    if can_resume:
        print("\n" + "=" * 60)
        print("RESUMING: Loading pre-computed DataFrames")
        print("=" * 60)

        df_real = pd.read_parquet(real_parquet)
        df_gen = pd.read_parquet(gen_parquet)
        print(f"  Loaded df_real: {len(df_real)}, df_gen: {len(df_gen)}")

        # Rebuild clouds from zarr for CD computation
        print("  Rebuilding canonicalized clouds...")
        real_clouds = {}
        for tid in tqdm(df_real.index, desc="Real clouds"):
            row = real_metadata[real_metadata["file_id"] == tid]
            if row.empty:
                continue
            try:
                rng = np.random.default_rng(args.seed)
                cloud = load_point_cloud(row.iloc[0]["file_path"], max_points=args.max_points, rng=rng)
                real_clouds[tid] = canonicalize(cloud)
            except Exception:
                pass

        gen_clouds = {}
        stem_to_file = {Path(f).stem: f for f in gen_metadata["sample_file"]}
        for gid in tqdm(df_gen.index, desc="Gen clouds"):
            sample_file = stem_to_file.get(gid)
            if sample_file is None:
                continue
            try:
                path = str(zarr_dir / sample_file)
                rng = np.random.default_rng(args.seed)
                cloud = load_point_cloud(path, max_points=args.max_points, rng=rng)
                gen_clouds[gid] = canonicalize(cloud)
            except Exception:
                pass

        print(f"  Rebuilt {len(real_clouds)} real + {len(gen_clouds)} gen clouds")
    else:
        print("\n" + "=" * 60)
        print("BUILDING df_real")
        print("=" * 60)

        df_real, real_clouds = build_df_real(
            real_metadata, data_path,
            max_points=args.max_points,
            num_workers=args.num_workers,
            seed=args.seed,
            skip_hcb=args.skip_hcb,
            spine_bins=args.spine_bins,
        )

        print("\n" + "=" * 60)
        print("BUILDING df_gen")
        print("=" * 60)

        df_gen, gen_clouds = build_df_gen(
            gen_metadata, zarr_dir,
            max_points=args.max_points,
            num_workers=args.num_workers,
            seed=args.seed,
            skip_hcb=args.skip_hcb,
            spine_bins=args.spine_bins,
        )

    # Ensure df_gen has height_bin for stratified morphological W1
    if "height_bin" not in df_gen.columns and "height_m" in df_gen.columns:
        df_gen["height_bin"] = df_gen["height_m"].apply(get_height_bin)

    # =========================================================================
    # 3.5. Outlier clipping (must precede all downstream computation)
    # =========================================================================
    # Hard physical caps for stem tracker failures.
    # The polynomial spine can diverge on trees with complex branching,
    # producing crown volumes orders of magnitude beyond physical plausibility
    # (e.g., 500,000 m³). These caps catch genuine failures only — normal large
    # trees (crown_volume ~700 m³, crown_r ~6 m) are left untouched.
    # The full gen distribution (including its upper tail) must be preserved
    # for honest morphological W₁ and population metrics.
    HARD_CAP_CROWN_VOL = 50_000  # m³ — no real tree exceeds this
    HARD_CAP_CROWN_R = 25.0       # m — no real crown radius exceeds this

    print("\n" + "=" * 60)
    print("OUTLIER CLIPPING (hard physical caps)")
    print("=" * 60)

    n_clipped_real_vol = int((df_real["crown_volume"] > HARD_CAP_CROWN_VOL).sum())
    n_clipped_gen_vol = int((df_gen["crown_volume"] > HARD_CAP_CROWN_VOL).sum())
    df_real["crown_volume"] = df_real["crown_volume"].clip(upper=HARD_CAP_CROWN_VOL)
    df_gen["crown_volume"] = df_gen["crown_volume"].clip(upper=HARD_CAP_CROWN_VOL)
    print(f"  Crown volume cap: {HARD_CAP_CROWN_VOL:,} m³")
    print(f"  Clipped {n_clipped_real_vol} real + {n_clipped_gen_vol} gen trees")

    n_clipped_real_r = int((df_real["max_crown_r"] > HARD_CAP_CROWN_R).sum())
    n_clipped_gen_r = int((df_gen["max_crown_r"] > HARD_CAP_CROWN_R).sum())
    df_real["max_crown_r"] = df_real["max_crown_r"].clip(upper=HARD_CAP_CROWN_R)
    df_gen["max_crown_r"] = df_gen["max_crown_r"].clip(upper=HARD_CAP_CROWN_R)
    print(f"  Crown radius cap: {HARD_CAP_CROWN_R:.1f} m")
    print(f"  Clipped {n_clipped_real_r} real + {n_clipped_gen_r} gen trees")

    outlier_clipping_info = {
        "crown_vol_cap_m3": HARD_CAP_CROWN_VOL,
        "crown_r_cap_m": HARD_CAP_CROWN_R,
        "n_real_clipped_vol": n_clipped_real_vol,
        "n_gen_clipped_vol": n_clipped_gen_vol,
        "n_real_clipped_r": n_clipped_real_r,
        "n_gen_clipped_r": n_clipped_gen_r,
        "method": "Hard physical caps, applied identically to real and gen",
        "rationale": "Catches stem tracker polynomial spine divergence only. "
                     "Normal large trees are preserved for honest distributional metrics.",
    }

    # =========================================================================
    # 4. Build df_pairs
    # =========================================================================
    print("\n" + "=" * 60)
    print("BUILDING df_pairs")
    print("=" * 60)

    df_pairs = build_df_pairs(df_real, df_gen, real_clouds, gen_clouds,
                              num_workers=args.num_workers)

    if df_pairs.empty:
        print("ERROR: No valid pairs found. Check data paths.")
        return

    # =========================================================================
    # 5. Build df_per_tree
    # =========================================================================
    print("\n" + "=" * 60)
    print("BUILDING df_per_tree")
    print("=" * 60)

    df_per_tree = build_df_per_tree(df_pairs, df_real)

    # =========================================================================
    # 6. Baselines
    # =========================================================================
    print("\n" + "=" * 60)
    print("BASELINES")
    print("=" * 60)

    df_intra = build_intra_baseline(df_real, real_clouds,
                                    num_workers=args.num_workers, seed=args.seed + 1000)
    df_inter = build_inter_baseline(df_real, real_clouds,
                                    num_workers=args.num_workers, seed=args.seed + 2000)

    # =========================================================================
    # 6.5. Scan-type confound diagnostic (optional, one-off)
    # =========================================================================
    if args.scan_diagnostic:
        print("\n" + "=" * 60)
        print("SCAN-TYPE CONFOUND DIAGNOSTIC")
        print("=" * 60)

        scan_diag = compute_scan_type_diagnostic(
            df_real, real_clouds,
            num_workers=args.num_workers, seed=args.seed,
        )
        scan_diag_path = output_dir / "scan_type_diagnostic.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(scan_diag_path, "w") as f:
            json.dump(scan_diag, f, indent=2, default=str)
        print(f"  Saved {scan_diag_path}")

    # =========================================================================
    # 7. Population metrics
    # =========================================================================
    if args.skip_layer1:
        print("\nSkipping population metrics (--skip_layer1)")
        population_metrics = {}
    else:
        print("\n" + "=" * 60)
        print("POPULATION METRICS (COV / MMD / 1-NNA / Voxel JSD)")
        print("=" * 60)

        population_metrics = compute_population_metrics(
            df_real, df_gen, real_clouds, gen_clouds,
            num_workers=args.num_workers, seed=args.seed,
        )

    # =========================================================================
    # 8. Statistical tests
    # =========================================================================
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)

    stat_tests = compute_statistical_tests(df_per_tree, df_intra)

    # =========================================================================
    # 9. Morphological Wasserstein + Build tables
    # =========================================================================
    print("\n" + "=" * 60)
    print("MORPHOLOGICAL WASSERSTEIN + BUILDING TABLES")
    print("=" * 60)

    # Compute morphological W1 at global and per-stratum levels
    morph_w1_global = compute_morphological_wasserstein(df_real, df_gen, stratum_cols=None)
    morph_w1_genus = compute_morphological_wasserstein(df_real, df_gen, stratum_cols=["genus"])
    morph_w1_height = compute_morphological_wasserstein(df_real, df_gen, stratum_cols=["height_bin"])

    if not morph_w1_global.empty:
        print("  Morphological W\u2081 (global):")
        for _, row in morph_w1_global.iterrows():
            print(f"    {row['property']}: {row['wasserstein']:.4f}")

    # Store global W1 values in population_metrics for downstream access
    if population_metrics.get("global") and not morph_w1_global.empty:
        for _, row in morph_w1_global.iterrows():
            population_metrics["global"][f"w1_{row['property']}"] = row["wasserstein"]

    tables = {
        "table_1a": build_table_1a(population_metrics, morph_w1_global),
        "table_1b": build_table_1b(df_per_tree, df_intra, df_inter),
        "table_b_genus": build_table_b(df_per_tree, df_intra, df_inter,
                                       population_metrics=population_metrics,
                                       morph_w1=morph_w1_genus),
        "table_c_height": build_table_c(df_per_tree, df_intra, df_inter,
                                         population_metrics=population_metrics,
                                         morph_w1=morph_w1_height),
        "table_e_cfg": build_table_e(df_pairs),
    }

    # =========================================================================
    # 10. Save + print
    # =========================================================================
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    save_results(
        output_dir, df_real, df_gen, df_pairs, df_per_tree,
        df_intra, df_inter, population_metrics, stat_tests, tables,
        outlier_clipping_info=outlier_clipping_info,
    )

    print_results(df_per_tree, df_intra, df_inter, population_metrics, stat_tests, tables)

    elapsed = time.time() - t_start
    print(f"\nTotal evaluation time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("Evaluation v2 complete!")


if __name__ == "__main__":
    main()
