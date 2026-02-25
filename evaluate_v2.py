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

    strat_cols = ["species", "height_bin"]
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
    max_per_nna: int = 50,
    min_per_stratum: int = 10,
    num_workers: int = 40,
    seed: int = 42,
) -> dict:
    """COV, MMD, 1-NNA, Voxel JSD — computed at atomic (species, height_bin) strata, then aggregated.

    max_per_stratum: max trees per side for COV/MMD (cross matrix only).
    max_per_nna: max trees per side for 1-NNA (requires rr + gg + cross matrices, ~4× cost).
    """
    rng = np.random.default_rng(seed)
    results = {"by_species_height": {}, "by_species": {}, "by_height_bin": {}, "global": None}

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

    # Step 1: Compute atomic (species, height_bin) strata
    if "species" not in df_real.columns or "height_bin" not in df_real.columns:
        print("  Warning: missing species or height_bin columns for population metrics")
        return results

    atomic_results = {}
    groups = list(df_real.groupby(["species", "height_bin"]))
    print(f"  Computing population metrics for {len(groups)} (species, height_bin) strata...")

    for (species, hb), grp in tqdm(groups, desc="Pop strata"):
        real_ids = [i for i in grp.index if i in real_clouds]
        gen_ids = [real_to_gen[i] for i in real_ids if i in real_to_gen and real_to_gen[i] in gen_clouds]
        real_ids = [i for i in real_ids if i in real_to_gen and real_to_gen[i] in gen_clouds]

        if len(real_ids) < min_per_stratum or len(gen_ids) < min_per_stratum:
            print(f"    ({species}, {hb}): skipped (n_real={len(real_ids)}, n_gen={len(gen_ids)})")
            continue

        real_ids = _subsample_ids(real_ids, max_per_stratum)
        gen_ids = [real_to_gen[i] for i in real_ids if real_to_gen[i] in gen_clouds]
        real_ids = [i for i in real_ids if real_to_gen.get(i) in gen_clouds]

        if len(real_ids) < min_per_stratum or len(gen_ids) < min_per_stratum:
            continue

        label = f"({species}, {hb})"
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

        voxel_jsd = compute_jsd_3d(rc, gc)

        print(f"    {label}: COV={cov:.4f}, MMD={mmd_val:.6f}, 1-NNA={nna:.4f}, JSD={voxel_jsd:.6f}")

        key = f"{species}|{hb}"
        atomic_results[key] = {
            "species": species, "height_bin": hb,
            "coverage": cov, "mmd": mmd_val, "one_nna": nna, "voxel_jsd": voxel_jsd,
            "n_real": n_r, "n_gen": n_g,
        }

    results["by_species_height"] = atomic_results

    if not atomic_results:
        print("  Warning: no valid (species, height_bin) strata for population metrics")
        return results

    # Step 2: Aggregate by species (weighted mean by n_real)
    species_groups: dict[str, list] = {}
    for v in atomic_results.values():
        species_groups.setdefault(v["species"], []).append(v)

    for sp, items in species_groups.items():
        weights = np.array([it["n_real"] for it in items], dtype=np.float64)
        w_sum = weights.sum()
        results["by_species"][sp] = {
            "coverage": float(np.average([it["coverage"] for it in items], weights=weights)),
            "mmd": float(np.average([it["mmd"] for it in items], weights=weights)),
            "one_nna": float(np.average([it["one_nna"] for it in items], weights=weights)),
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
        "voxel_jsd": float(np.average([it["voxel_jsd"] for it in all_items], weights=weights)),
        "n_real": int(w_sum),
        "n_strata": len(all_items),
    }

    glob = results["global"]
    print(f"  Global — COV: {glob['coverage']:.4f}, MMD: {glob['mmd']:.6f}, "
          f"1-NNA: {glob['one_nna']:.4f}, Voxel JSD: {glob['voxel_jsd']:.6f} "
          f"({glob['n_strata']} strata, {glob['n_real']} trees)")

    return results


# =============================================================================
# Statistical tests
# =============================================================================


def compute_statistical_tests(
    df_per_tree: pd.DataFrame,
    df_intra: pd.DataFrame,
) -> dict:
    """Wasserstein distance per metric (gen vs intra)."""
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
            "wasserstein": float(wasserstein_distance(gen_scores, intra_scores)),
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
# Output tables
# =============================================================================


def build_table_a(
    df_per_tree: pd.DataFrame,
    df_intra: pd.DataFrame,
    df_inter: pd.DataFrame,
    population_metrics: dict,
) -> pd.DataFrame:
    """Table A: Global summary — all metrics with Gen/Intra/Inter/Ratio columns."""
    rows = []

    # Population metrics
    glob = population_metrics.get("global")
    if glob:
        for pop_m in ["voxel_jsd", "coverage", "mmd", "one_nna"]:
            rows.append({
                "layer": "1 (population)",
                "metric": pop_m,
                "gen_vs_real": glob.get(pop_m, float("nan")),
                "intra_class": float("nan"),
                "inter_class": float("nan"),
                "ratio": float("nan"),
            })

    # Paired metrics
    for m in PAIR_METRICS:
        mean_col = f"{m}_mean"
        gen_med = df_per_tree[mean_col].median() if mean_col in df_per_tree.columns else float("nan")
        intra_med = df_intra[mean_col].median() if mean_col in df_intra.columns else float("nan")
        inter_med = df_inter[mean_col].median() if mean_col in df_inter.columns else float("nan")
        ratio = gen_med / intra_med if pd.notna(intra_med) and intra_med > 0 else float("nan")

        layer = "1 (paired)" if m == "reconstruction_cd" else "2 (paired)"
        rows.append({
            "layer": layer,
            "metric": m,
            "gen_vs_real": gen_med,
            "intra_class": intra_med,
            "inter_class": inter_med,
            "ratio": ratio,
        })

    return pd.DataFrame(rows)


def _build_stratified_table(
    df_per_tree: pd.DataFrame,
    df_intra: pd.DataFrame,
    df_inter: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """Build stratified table for a given grouping column."""
    rows = []
    groups = sorted(df_per_tree[group_col].dropna().unique())

    for g in groups:
        gen_grp = df_per_tree[df_per_tree[group_col] == g]
        intra_grp = df_intra[df_intra[group_col] == g] if group_col in df_intra.columns else df_intra.iloc[:0]
        n_trees = len(gen_grp)

        row = {group_col: g, "n_trees": n_trees}
        for m in PAIR_METRICS:
            mean_col = f"{m}_mean"
            gen_med = gen_grp[mean_col].median() if mean_col in gen_grp.columns else float("nan")
            intra_med = intra_grp[mean_col].median() if mean_col in intra_grp.columns else float("nan")
            ratio = gen_med / intra_med if pd.notna(intra_med) and intra_med > 0 else float("nan")
            row[f"{m}_gen"] = gen_med
            row[f"{m}_intra"] = intra_med
            row[f"{m}_ratio"] = ratio
        rows.append(row)

    return pd.DataFrame(rows)


def build_table_b(df_per_tree, df_intra, df_inter):
    """Table B: By Species."""
    return _build_stratified_table(df_per_tree, df_intra, df_inter, "species")


def build_table_c(df_per_tree, df_intra, df_inter):
    """Table C: By Height Bin."""
    return _build_stratified_table(df_per_tree, df_intra, df_inter, "height_bin")


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

    # Table A
    table_a = tables.get("table_a_global")
    if table_a is not None and not table_a.empty:
        print("\nTABLE A: GLOBAL SUMMARY")
        print("-" * 80)
        header = f"{'Layer':<16s} {'Metric':<28s} {'Gen vs Real':>12s} {'Intra':>12s} {'Inter':>12s} {'Ratio':>8s}"
        print(header)
        print("-" * 80)
        for _, row in table_a.iterrows():
            metric_name = row['metric']
            unit = PAIR_METRIC_UNITS.get(metric_name, "")
            display_name = f"{metric_name} ({unit})" if unit else metric_name
            gen_v = f"{row['gen_vs_real']:.6f}" if pd.notna(row['gen_vs_real']) else "--"
            intra_v = f"{row['intra_class']:.6f}" if pd.notna(row['intra_class']) else "--"
            inter_v = f"{row['inter_class']:.6f}" if pd.notna(row['inter_class']) else "--"
            ratio_v = f"{row['ratio']:.2f}" if pd.notna(row['ratio']) else "--"
            print(f"{row['layer']:<16s} {display_name:<28s} {gen_v:>12s} {intra_v:>12s} {inter_v:>12s} {ratio_v:>8s}")

    # Statistical tests
    if stat_tests:
        print(f"\nSTATISTICAL TESTS (gen vs intra)")
        print("-" * 80)
        header = f"{'Metric':<28s} {'Wasserstein':>12s}"
        print(header)
        for m, vals in stat_tests.items():
            print(f"{m:<28s} {vals['wasserstein']:>12.6f}")

    # Stratified tables summary
    for name in ["table_b_species", "table_c_height"]:
        df = tables.get(name)
        if df is not None and not df.empty:
            key_col = df.columns[0]
            print(f"\n{name.upper().replace('_', ' ')}:")
            print(f"  {len(df)} strata, key column: {key_col}")
            # Print first metric ratio
            ratio_col = f"{PAIR_METRICS[0]}_ratio"
            if ratio_col in df.columns:
                for _, row in df.iterrows():
                    n = row.get("n_trees", 0)
                    ratio = row.get(ratio_col, float("nan"))
                    ratio_str = f"{ratio:.2f}" if pd.notna(ratio) else "--"
                    print(f"    {str(row[key_col]):<25s}  n={int(n):>5d}  CD ratio={ratio_str}")

    print("\n" + "=" * 80)


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
    # 9. Build tables
    # =========================================================================
    print("\n" + "=" * 60)
    print("BUILDING TABLES")
    print("=" * 60)

    tables = {
        "table_a_global": build_table_a(df_per_tree, df_intra, df_inter, population_metrics),
        "table_b_species": build_table_b(df_per_tree, df_intra, df_inter),
        "table_c_height": build_table_c(df_per_tree, df_intra, df_inter),
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
    )

    print_results(df_per_tree, df_intra, df_inter, population_metrics, stat_tests, tables)

    elapsed = time.time() - t_start
    print(f"\nTotal evaluation time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("Evaluation v2 complete!")


if __name__ == "__main__":
    main()
