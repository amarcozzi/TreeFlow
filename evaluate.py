"""
TreeFlow evaluation: morphological evaluation of generated tree point clouds.

Per-pair metrics:
  - Chamfer distance (m) — symmetric, PCA-canonicalized clouds in metric coords
  - Height at max crown radius (m) — from stem tracker
  - Max crown radius (m) — from stem tracker
  - Height to crown base (m) — from stem tracker
  - Vertical KDE JSD — 1D density along z-axis
  - 2D histogram JSD — radial × arc-length profile from stem tracker

Tables:
  1. Global summary with gen / intra-class / inter-class baselines + W₁
  2. By genus (gen mean-of-medians + W₁)
  3. By height bin (gen mean-of-medians + W₁)
"""

import sys
import json
import time
import argparse

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import zarr
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde, wasserstein_distance
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from stem_tracker import compute_rs_spine


# ── Constants ────────────────────────────────────────────────────────────────

HEIGHT_BIN_EDGES = [0, 5, 10, 15, 20, 25, 30, 35, 40, float("inf")]
HEIGHT_BIN_LABELS = [
    "0-5", "5-10", "10-15", "15-20", "20-25",
    "25-30", "30-35", "35-40", "40+",
]

METRICS = ["chamfer_dist", "delta_h_max_cr", "delta_max_crown_r", "delta_hcb",
           "vert_kde_jsd", "hist_2d_jsd"]

METRIC_DISPLAY = {
    "chamfer_dist":      ("Chamfer distance",   "m"),
    "delta_h_max_cr":    ("Δ Height at max crown R", "m"),
    "delta_max_crown_r": ("Δ Max crown radius", "m"),
    "delta_hcb":         ("Δ Height to crown base", "m"),
    "vert_kde_jsd":      ("Vertical KDE JSD",  ""),
    "hist_2d_jsd":       ("2D histogram JSD",  ""),
}

MORPH_PROPERTIES = ["h_max_cr", "max_crown_r", "hcb"]

MORPH_DISPLAY = {
    "h_max_cr":     ("Height at max crown R", "m"),
    "max_crown_r":  ("Max crown radius", "m"),
    "hcb":          ("Height to crown base", "m"),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_height_bin(h: float) -> str:
    for i, (lo, hi) in enumerate(zip(HEIGHT_BIN_EDGES[:-1], HEIGHT_BIN_EDGES[1:])):
        if lo <= h < hi:
            return HEIGHT_BIN_LABELS[i]
    return HEIGHT_BIN_LABELS[-1]


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two probability vectors."""
    eps = 1e-10
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def canonicalize(points: np.ndarray) -> np.ndarray:
    """PCA-align XY axes, resolve sign ambiguity via cubic moment."""
    pts = points.copy()
    pts -= pts.mean(axis=0)
    xy = pts[:, :2]
    _, _, Vt = np.linalg.svd(xy - xy.mean(axis=0), full_matrices=False)
    R = np.eye(3)
    R[:2, :2] = Vt
    pts = pts @ R.T
    if np.sum(pts[:, 0] ** 3) < 0:
        pts[:, 0] *= -1
    return pts


def chamfer_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Symmetric Chamfer distance (euclidean, in meters)."""
    D = cdist(p1, p2, metric="euclidean")
    return float((D.min(axis=1).mean() + D.min(axis=0).mean()) / 2)


# ── Chamfer distance parallel helpers ────────────────────────────────────────

_CD_CLOUDS_A: dict[str, np.ndarray] = {}
_CD_CLOUDS_B: dict[str, np.ndarray] = {}


def _cd_worker(task: tuple[str, str]) -> float:
    """Compute CD for one pair using module-level cloud dicts (fork-shared)."""
    id_a, id_b = task
    return chamfer_distance(_CD_CLOUDS_A[id_a], _CD_CLOUDS_B[id_b])


def _compute_cd_parallel(
    clouds_a: dict[str, np.ndarray],
    clouds_b: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
    num_workers: int,
    desc: str,
) -> list[float]:
    """Compute Chamfer distances for a list of (id_a, id_b) pairs in parallel."""
    global _CD_CLOUDS_A, _CD_CLOUDS_B
    _CD_CLOUDS_A = clouds_a
    _CD_CLOUDS_B = clouds_b
    if num_workers <= 1:
        results = [_cd_worker(p) for p in tqdm(pairs, desc=desc)]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            results = list(tqdm(
                pool.map(_cd_worker, pairs, chunksize=64),
                total=len(pairs), desc=desc,
            ))
    _CD_CLOUDS_A = {}
    _CD_CLOUDS_B = {}
    return results


# ── Shared crown-metric helpers ──────────────────────────────────────────────

def compute_mean_r_per_slice(s, r, s_max, n_slices=30):
    """Mean radial distance in each arc-length slice.

    Returns (slice_centers, mean_r_per_slice) — both numpy arrays of length n_slices.
    """
    slice_edges = np.linspace(0, s_max, n_slices + 1)
    slice_centers = 0.5 * (slice_edges[:-1] + slice_edges[1:])
    mean_r_per_slice = np.zeros(n_slices)
    for i in range(n_slices):
        mask = (s >= slice_edges[i]) & (s < slice_edges[i + 1])
        if mask.sum() > 0:
            mean_r_per_slice[i] = r[mask].mean()
    return slice_centers, mean_r_per_slice


def compute_hcb(slice_centers, mean_r_per_slice, s_max):
    """Detect height-to-crown-base via Kneedle on cumulative mean-r.

    Returns (hcb_val, kneedle_data) where:
        hcb_val     — float in [0, 1] normalised arc-length (nan if degenerate)
        kneedle_data — dict with x_norm, y_norm, d, knee_idx (empty if degenerate)
    """
    if mean_r_per_slice.max() <= 0 or np.allclose(mean_r_per_slice, mean_r_per_slice[0]):
        return float("nan"), {}
    cumr = np.cumsum(mean_r_per_slice)
    x_norm = (slice_centers - slice_centers[0]) / (slice_centers[-1] - slice_centers[0])
    y_norm = (cumr - cumr[0]) / (cumr[-1] - cumr[0])
    d = (x_norm - y_norm) * (1 - x_norm) ** 0.5
    knee_idx = int(np.argmax(d))
    hcb_val = slice_centers[knee_idx] / s_max
    kneedle_data = {"x_norm": x_norm, "y_norm": y_norm, "d": d, "knee_idx": knee_idx}
    return hcb_val, kneedle_data


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_features(cloud: np.ndarray, height_m: float) -> dict:
    """Extract morphological features from one point cloud.

    Returns dict with:
        h_max_cr      (m)  — height at max crown radius (arc-length position)
        max_crown_r   (m)  — from stem tracker (max mean-r per arc-length slice)
        hcb           (m)  — from stem tracker (arc-length density analysis)
        vert_kde      (64,) — 1D KDE of z-axis (no stem tracker)
        hist_2d       (512,) — 2D (r, s) histogram (stem tracker)
    """
    z = cloud[:, 2]
    z_min, z_max = z.min(), z.max()
    scale = height_m / 2.0  # normalized → metric

    # Stem tracker → cylindrical coordinates (r, s)
    r, s, _, _, _ = compute_rs_spine(cloud)

    # 2. Vertical 1D KDE (uses raw z, not stem tracker)
    kde_bins = 64
    try:
        kde = gaussian_kde(z)
        z_eval = np.linspace(z_min, z_max, kde_bins)
        vert_kde = kde(z_eval)
        vert_kde = vert_kde / (vert_kde.sum() + 1e-30)
    except Exception:
        vert_kde = np.ones(kde_bins) / kde_bins

    # 3. 2D (r, s) histogram — bin edges relative to each tree's own range
    hist_r_bins, hist_s_bins = 16, 32
    eps = 1e-6
    r_max = r.max() + eps
    s_max = s.max() + eps if s.max() > 0 else eps
    hist_2d, _, _ = np.histogram2d(
        r, s,
        bins=[np.linspace(0, r_max, hist_r_bins + 1),
              np.linspace(0, s_max, hist_s_bins + 1)],
    )
    hist_2d = hist_2d.flatten()
    total = hist_2d.sum()
    if total > 0:
        hist_2d = hist_2d / total

    # 4. Mean-r per arc-length slice (shared by max crown radius + HCB)
    slice_centers, mean_r_per_slice = compute_mean_r_per_slice(s, r, s_max)
    max_crown_r = float(mean_r_per_slice.max())

    # 5. Height at max crown radius (arc-length of widest slice, in meters)
    h_max_cr = float(slice_centers[np.argmax(mean_r_per_slice)] / s_max * height_m)

    # 6. Height to crown base (Kneedle on cumulative mean-r)
    hcb_val, _ = compute_hcb(slice_centers, mean_r_per_slice, s_max)

    return {
        "h_max_cr":     h_max_cr,
        "max_crown_r":  float(max_crown_r * scale),
        "hcb":          float(hcb_val * height_m),
        "vert_kde":     vert_kde,
        "hist_2d":      hist_2d,
    }


# ── Worker functions ─────────────────────────────────────────────────────────

def _extract_worker(task: dict) -> dict | None:
    """Process one tree (real or generated): load zarr, extract features."""
    try:
        points = zarr.load(task["zarr_path"]).astype(np.float32)
        if task.get("max_points") and len(points) > task["max_points"]:
            rng = np.random.default_rng(task.get("seed", 42))
            idx = rng.choice(len(points), size=task["max_points"], replace=False)
            points = points[idx]
        feats = extract_features(points, task["height_m"])
        feats["tree_id"] = task["tree_id"]
        # Canonicalized cloud in metric coordinates for Chamfer distance
        feats["_canon_cloud"] = canonicalize(points) * (task["height_m"] / 2.0)
        return feats
    except Exception as e:
        return {"tree_id": task["tree_id"], "_error": str(e)}


def _run_extraction(tasks: list[dict], num_workers: int, desc: str) -> list[dict]:
    """Run feature extraction on a list of tasks, with progress bar."""
    if num_workers <= 1:
        results = [_extract_worker(t) for t in tqdm(tasks, desc=desc)]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            results = list(tqdm(
                pool.map(_extract_worker, tasks, chunksize=8),
                total=len(tasks), desc=desc,
            ))
    good = [r for r in results if r is not None and "_error" not in r]
    bad  = [r for r in results if r is not None and "_error" in r]
    if bad:
        print(f"  {len(bad)} failures")
        for b in bad[:5]:
            print(f"    {b['tree_id']}: {b['_error']}")
    return good


# ── Data loading ─────────────────────────────────────────────────────────────

def load_metadata(data_path: Path, experiment_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load real (test split) and generated metadata.

    Returns (real_meta, gen_meta) DataFrames.
    """
    # Real
    csv = data_path / "metadata.csv"
    real = pd.read_csv(csv)
    real["file_id"] = real["filename"].apply(lambda x: Path(x).stem)
    real["file_path"] = real["file_id"].apply(lambda x: str(data_path / f"{x}.zarr"))
    real = real[real["file_path"].apply(lambda x: Path(x).exists())]
    real = real[real["split"] == "test"].copy()
    real["height_bin"] = real["tree_H"].apply(get_height_bin)
    if "genus" not in real.columns:
        real["genus"] = "unknown"
    print(f"Real test trees: {len(real)}")

    # Generated
    gen_csv = experiment_dir / "samples" / "samples_metadata.csv"
    gen = pd.read_csv(gen_csv)
    gen["source_tree_id"] = gen["source_tree_id"].apply(lambda x: str(x).zfill(5))
    # Join genus from real metadata
    id_to_genus = real.set_index("file_id")["genus"].to_dict()
    gen["genus"] = gen["source_tree_id"].map(id_to_genus).fillna("unknown")
    gen["height_bin"] = gen["height_m"].apply(get_height_bin)
    print(f"Generated samples: {len(gen)}")

    return real, gen


# ── Build the core dataframe ─────────────────────────────────────────────────

def build_pair_dataframe(
    real_meta: pd.DataFrame,
    gen_meta: pd.DataFrame,
    zarr_dir: Path,
    max_points: int,
    num_workers: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extract features for all trees and build the pair dataframe.

    Returns (df_pairs, df_real_feats, df_gen_feats, real_clouds, gen_clouds).
    real_clouds/gen_clouds map tree_id → canonicalized metric-scale point cloud.
    """
    # Extract real features
    real_tasks = [
        {"tree_id": row["file_id"], "zarr_path": row["file_path"],
         "height_m": float(row["tree_H"]), "max_points": max_points, "seed": seed}
        for _, row in real_meta.iterrows()
    ]
    print(f"\nExtracting features for {len(real_tasks)} real trees...")
    real_results = _run_extraction(real_tasks, num_workers, "Real trees")

    real_feats = {r["tree_id"]: r for r in real_results}
    real_clouds = {r["tree_id"]: r["_canon_cloud"] for r in real_results}

    # Extract generated features
    gen_tasks = [
        {"tree_id": Path(row["sample_file"]).stem,
         "zarr_path": str(zarr_dir / row["sample_file"]),
         "height_m": float(row["height_m"]), "max_points": max_points, "seed": seed}
        for _, row in gen_meta.iterrows()
        if pd.notna(row.get("height_m")) and float(row["height_m"]) > 0
    ]
    print(f"\nExtracting features for {len(gen_tasks)} generated trees...")
    gen_results = _run_extraction(gen_tasks, num_workers, "Gen trees")

    gen_feats = {r["tree_id"]: r for r in gen_results}
    gen_clouds = {r["tree_id"]: r["_canon_cloud"] for r in gen_results}

    # Build gen_id → metadata lookup
    gen_id_to_meta = {}
    for _, row in gen_meta.iterrows():
        gid = Path(row["sample_file"]).stem
        gen_id_to_meta[gid] = row

    # Build pair rows (without CD — added after parallel CD computation)
    pair_keys = []  # (rid, gid) for CD
    rows = []
    for gid, gf in gen_feats.items():
        meta = gen_id_to_meta.get(gid)
        if meta is None:
            continue
        rid = meta["source_tree_id"]
        rf = real_feats.get(rid)
        if rf is None:
            continue

        pair_keys.append((rid, gid))
        rows.append({
            "real_id": rid,
            "gen_id": gid,
            "genus": meta.get("genus", "unknown"),
            "species": meta.get("species", "unknown"),
            "height_bin": meta.get("height_bin", "unknown"),
            "scan_type": meta.get("scan_type", "unknown"),
            "height_m": float(meta.get("height_m", 0)),
            "cfg_scale": float(meta.get("cfg_scale", 0)),
            "delta_h_max_cr":    abs(gf["h_max_cr"]     - rf["h_max_cr"]),
            "delta_max_crown_r": abs(gf["max_crown_r"]  - rf["max_crown_r"]),
            "delta_hcb":         abs(gf["hcb"]          - rf["hcb"]),
            "vert_kde_jsd":      jsd(rf["vert_kde"], gf["vert_kde"]),
            "hist_2d_jsd":       jsd(rf["hist_2d"],  gf["hist_2d"]),
        })

    # Compute Chamfer distances in parallel
    print(f"\nComputing Chamfer distances for {len(pair_keys)} gen pairs...")
    cd_values = _compute_cd_parallel(
        real_clouds, gen_clouds, pair_keys, num_workers, "CD (gen)",
    )
    for row, cd in zip(rows, cd_values):
        row["chamfer_dist"] = cd

    df_pairs = pd.DataFrame(rows)
    print(f"\nPair dataframe: {len(df_pairs)} rows, "
          f"{df_pairs['real_id'].nunique()} unique real trees")

    # Build feature dataframes for downstream (W₁, baselines)
    df_real_feats = pd.DataFrame([
        {"tree_id": tid, "h_max_cr": r["h_max_cr"],
         "max_crown_r": r["max_crown_r"], "hcb": r["hcb"],
         "vert_kde": r["vert_kde"], "hist_2d": r["hist_2d"]}
        for tid, r in real_feats.items()
    ]).set_index("tree_id")

    df_gen_feats = pd.DataFrame([
        {"gen_id": gid, "real_id": gen_id_to_meta[gid]["source_tree_id"],
         "h_max_cr": r["h_max_cr"],
         "max_crown_r": r["max_crown_r"], "hcb": r["hcb"]}
        for gid, r in gen_feats.items()
        if gid in gen_id_to_meta
    ]).set_index("gen_id")

    # Add stratification columns to feature dataframes
    id_to_row = real_meta.set_index("file_id")
    for col in ["genus", "height_bin"]:
        df_real_feats[col] = df_real_feats.index.map(
            lambda x, c=col: id_to_row.loc[x, c] if x in id_to_row.index else "unknown"
        )
    for col in ["genus", "height_bin"]:
        df_gen_feats[col] = df_gen_feats["real_id"].map(
            lambda x, c=col: id_to_row.loc[x, c] if x in id_to_row.index else "unknown"
        )

    return df_pairs, df_real_feats, df_gen_feats, real_clouds, gen_clouds


# ── Baselines ────────────────────────────────────────────────────────────────

def build_baselines(
    df_real: pd.DataFrame,
    real_clouds: dict[str, np.ndarray],
    num_workers: int = 1,
    n_neighbors: int = 32,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build intra-class and inter-class baseline pair dataframes.

    Intra-class: same (genus, height_bin). Inter-class: different genus.
    Each row has the same columns as df_pairs (minus gen_id).
    Returns (df_intra, df_inter).
    """
    rng = np.random.default_rng(seed)
    all_ids = list(df_real.index)

    def _build(neighbor_map: dict[str, list[str]], label: str) -> pd.DataFrame:
        rows = []
        pair_keys = []
        for tid, neighbors in neighbor_map.items():
            anchor = df_real.loc[tid]
            anchor_kde = np.array(anchor["vert_kde"])
            anchor_hist = np.array(anchor["hist_2d"])
            for nid in neighbors:
                if nid not in df_real.index:
                    continue
                nb = df_real.loc[nid]
                pair_keys.append((tid, nid))
                rows.append({
                    "anchor_id": tid,
                    "neighbor_id": nid,
                    "delta_h_max_cr":    abs(anchor["h_max_cr"]     - nb["h_max_cr"]),
                    "delta_max_crown_r": abs(anchor["max_crown_r"]  - nb["max_crown_r"]),
                    "delta_hcb":         abs(anchor["hcb"]          - nb["hcb"]),
                    "vert_kde_jsd":      jsd(anchor_kde, np.array(nb["vert_kde"])),
                    "hist_2d_jsd":       jsd(anchor_hist, np.array(nb["hist_2d"])),
                })
        # Compute CD in parallel
        cd_values = _compute_cd_parallel(
            real_clouds, real_clouds, pair_keys, num_workers,
            f"CD ({label})",
        )
        for row, cd in zip(rows, cd_values):
            row["chamfer_dist"] = cd
        df = pd.DataFrame(rows)
        print(f"  {label}: {len(df)} pairs from {len(neighbor_map)} anchor trees")
        return df

    # Intra-class neighbors: same (genus, height_bin)
    intra_map = {}
    for _, grp in df_real.groupby(["genus", "height_bin"]):
        ids = list(grp.index)
        if len(ids) < 2:
            continue
        for tid in ids:
            others = [x for x in ids if x != tid]
            if len(others) >= n_neighbors:
                chosen = rng.choice(others, size=n_neighbors, replace=False).tolist()
            else:
                chosen = rng.choice(others, size=n_neighbors, replace=True).tolist()
            intra_map[tid] = chosen

    # Inter-class neighbors: different genus
    genus_map = df_real["genus"].to_dict()
    inter_map = {}
    for tid in all_ids:
        others = [x for x in all_ids if x != tid and genus_map.get(x) != genus_map.get(tid)]
        if not others:
            continue
        if len(others) >= n_neighbors:
            chosen = rng.choice(others, size=n_neighbors, replace=False).tolist()
        else:
            chosen = rng.choice(others, size=n_neighbors, replace=True).tolist()
        inter_map[tid] = chosen

    print("Building baselines...")
    df_intra = _build(intra_map, "Intra-class")
    df_inter = _build(inter_map, "Inter-class")
    return df_intra, df_inter


# ── Population W₁ ───────────────────────────────────────────────────────────

def compute_population_w1(
    df_real: pd.DataFrame,
    df_gen: pd.DataFrame,
    group_col: str | None = None,
) -> pd.DataFrame:
    """W₁ between real and generated marginal distributions per property.

    If group_col is given, computes per stratum. Otherwise global.
    Returns DataFrame with columns: [group_col], property, display, unit, w1, n_real, n_gen.
    """
    rows = []

    def _w1_for_group(r_df, g_df, group_label=None):
        for prop in MORPH_PROPERTIES:
            vals_r = r_df[prop].dropna().values
            vals_g = g_df[prop].dropna().values
            if len(vals_r) < 5 or len(vals_g) < 5:
                continue
            w1 = float(wasserstein_distance(vals_r, vals_g))
            display, unit = MORPH_DISPLAY[prop]
            row = {"property": prop, "display": display, "unit": unit,
                   "w1": w1, "n_real": len(vals_r), "n_gen": len(vals_g)}
            if group_col and group_label is not None:
                row[group_col] = group_label
            rows.append(row)

    if group_col is None:
        _w1_for_group(df_real, df_gen)
    else:
        for key, grp_r in df_real.groupby(group_col):
            try:
                grp_g = df_gen.groupby(group_col).get_group(key)
            except KeyError:
                continue
            _w1_for_group(grp_r, grp_g, group_label=key)

    return pd.DataFrame(rows)


# ── Population COV / MMD / 1-NNA (supplementary; cf. Diff-Tree) ───────────────
#
# Reported only in the supplementary material, for comparison with prior
# point-cloud generation work (e.g. Diff-Tree reports COV and MMD). Computed on
# canonicalized, metric-scale clouds subsampled to a fixed point count (default
# 2048, the PointFlow/Diff-Tree convention) and stratified by (genus, height
# bin) so distances compare trees of similar type and size.

def coverage(dist_matrix: np.ndarray) -> float:
    """COV (↑): fraction of real reference trees that are the nearest neighbor
    of at least one generated tree. dist_matrix is (n_real, n_gen)."""
    if dist_matrix.size == 0:
        return float("nan")
    nn_real_for_gen = dist_matrix.argmin(axis=0)
    return len(set(nn_real_for_gen.tolist())) / dist_matrix.shape[0]


def mmd(dist_matrix: np.ndarray) -> float:
    """MMD (↓): mean over real reference trees of the Chamfer distance to the
    nearest generated tree (meters). dist_matrix is (n_real, n_gen)."""
    if dist_matrix.size == 0:
        return float("nan")
    return float(dist_matrix.min(axis=1).mean())


def one_nn_accuracy(rr: np.ndarray, gg: np.ndarray, rg: np.ndarray) -> float:
    """1-NNA (→0.5): leave-one-out 1-NN classification accuracy over the pooled
    real+generated set. 0.5 ⇒ indistinguishable; far from 0.5 ⇒ separable.

    rr: (n_r, n_r) real-real, gg: (n_g, n_g) gen-gen, rg: (n_r, n_g) real-gen.
    """
    n_r, n_g = rg.shape
    if n_r == 0 or n_g == 0:
        return float("nan")
    correct = 0
    for i in range(n_r):
        row = rr[i].copy()
        row[i] = np.inf  # exclude self
        if row.min() <= rg[i].min():
            correct += 1
    for j in range(n_g):
        col = gg[j].copy()
        col[j] = np.inf
        if col.min() <= rg[:, j].min():
            correct += 1
    return correct / (n_r + n_g)


def _cd_matrix(clouds_a, ids_a, clouds_b, ids_b, num_workers, desc):
    """Dense pairwise Chamfer matrix (len(ids_a) × len(ids_b))."""
    pairs = [(a, b) for a in ids_a for b in ids_b]
    vals = _compute_cd_parallel(clouds_a, clouds_b, pairs, num_workers, desc)
    return np.asarray(vals, dtype=np.float64).reshape(len(ids_a), len(ids_b))


def _subsample_clouds(clouds: dict, n_points: int, seed: int) -> dict:
    """Copy of `clouds` with each cloud randomly subsampled to n_points."""
    rng = np.random.default_rng(seed)
    out = {}
    for tid in sorted(clouds.keys()):
        c = clouds[tid]
        if len(c) > n_points:
            idx = rng.choice(len(c), size=n_points, replace=False)
            out[tid] = c[idx]
        else:
            out[tid] = c
    return out


def compute_population_cov_mmd(
    df_real_feats: pd.DataFrame,
    df_gen_feats: pd.DataFrame,
    real_clouds: dict[str, np.ndarray],
    gen_clouds: dict[str, np.ndarray],
    num_workers: int = 1,
    n_points: int = 2048,
    max_per_stratum: int = 150,
    max_per_nna: int = 75,
    min_per_stratum: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """COV / MMD / 1-NNA per (genus, height_bin) stratum.

    Returns a per-stratum DataFrame; the caller aggregates to global/by-bin.
    """
    print(f"  Subsampling clouds to {n_points} points for COV/MMD/1-NNA...")
    real_sub = _subsample_clouds(real_clouds, n_points, seed)
    gen_sub = _subsample_clouds(gen_clouds, n_points, seed + 1)

    rng = np.random.default_rng(seed)
    real_ok = [t for t in df_real_feats.index if t in real_sub]
    gen_ok = [g for g in df_gen_feats.index if g in gen_sub]
    real_groups = df_real_feats.loc[real_ok].groupby(["genus", "height_bin"])
    gen_grouped = df_gen_feats.loc[gen_ok].groupby(["genus", "height_bin"])

    rows = []
    for key, grp_r in real_groups:
        try:
            grp_g = gen_grouped.get_group(key)
        except KeyError:
            continue
        genus, hb = key
        real_ids = list(grp_r.index)
        gen_ids = list(grp_g.index)
        if len(real_ids) < min_per_stratum or len(gen_ids) < min_per_stratum:
            continue
        if len(real_ids) > max_per_stratum:
            real_ids = sorted(rng.choice(real_ids, max_per_stratum, replace=False).tolist())
        if len(gen_ids) > max_per_stratum:
            gen_ids = sorted(rng.choice(gen_ids, max_per_stratum, replace=False).tolist())

        cross = _cd_matrix(real_sub, real_ids, gen_sub, gen_ids,
                           num_workers, f"COV/MMD {genus}/{hb}")
        cov = coverage(cross)
        mmd_v = mmd(cross)

        # 1-NNA on a (possibly) smaller balanced subset — needs rr + gg + rg.
        nr = min(len(real_ids), max_per_nna)
        ng = min(len(gen_ids), max_per_nna)
        nna_r = real_ids if nr == len(real_ids) else sorted(rng.choice(real_ids, nr, replace=False).tolist())
        nna_g = gen_ids if ng == len(gen_ids) else sorted(rng.choice(gen_ids, ng, replace=False).tolist())
        rr = _cd_matrix(real_sub, nna_r, real_sub, nna_r, num_workers, f"1-NNA rr {genus}/{hb}")
        gg = _cd_matrix(gen_sub, nna_g, gen_sub, nna_g, num_workers, f"1-NNA gg {genus}/{hb}")
        rg = _cd_matrix(real_sub, nna_r, gen_sub, nna_g, num_workers, f"1-NNA rg {genus}/{hb}")
        nna = one_nn_accuracy(rr, gg, rg)

        print(f"    [{genus}, {hb}] COV={cov:.3f} MMD={mmd_v:.4f} 1-NNA={nna:.3f} "
              f"(n_real={len(real_ids)}, n_gen={len(gen_ids)})")
        rows.append({
            "genus": genus, "height_bin": hb,
            "coverage": cov, "mmd": mmd_v, "one_nna": nna,
            "n_real": len(real_ids), "n_gen": len(gen_ids),
        })

    return pd.DataFrame(rows)


def aggregate_population_metrics(strata: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Aggregate per-stratum COV/MMD/1-NNA into an n_real-weighted global summary
    and a by-height-bin summary."""
    if strata.empty:
        return {}, pd.DataFrame()

    def _wavg(df):
        w = df["n_real"].to_numpy(dtype=float)
        return {
            "coverage": float(np.average(df["coverage"], weights=w)),
            "mmd": float(np.average(df["mmd"], weights=w)),
            "one_nna": float(np.average(df["one_nna"], weights=w)),
            "n_real": int(w.sum()),
            "n_strata": int(len(df)),
        }

    global_agg = _wavg(strata)
    by_hb_rows = []
    for hb, grp in strata.groupby("height_bin"):
        r = _wavg(grp)
        r["height_bin"] = hb
        by_hb_rows.append(r)
    by_hb = pd.DataFrame(by_hb_rows)
    return global_agg, by_hb


def build_population_table(strata: pd.DataFrame, global_agg: dict,
                           by_hb: pd.DataFrame) -> str:
    lines = []
    lines.append("SUPPLEMENTARY: POPULATION METRICS (COV / MMD / 1-NNA)")
    lines.append("=" * 72)
    lines.append("COV ↑ coverage of real set | MMD ↓ (m) fidelity | 1-NNA → 0.50 indistinguishable")
    lines.append("Chamfer on canonicalized metric clouds, stratified by (genus, height bin).")
    lines.append("")
    if not global_agg:
        lines.append("  (no strata met the minimum-count threshold)")
        return "\n".join(lines)
    lines.append(f"  {'':<14s} {'COV↑':>8s} {'MMD↓':>10s} {'1-NNA':>8s} "
                 f"{'n_real':>8s} {'n_strata':>9s}")
    lines.append("─" * 72)
    lines.append(f"  {'Global':<14s} {global_agg['coverage']:>8.3f} {global_agg['mmd']:>10.4f} "
                 f"{global_agg['one_nna']:>8.3f} {global_agg['n_real']:>8d} {global_agg['n_strata']:>9d}")
    if not by_hb.empty:
        lines.append("")
        lines.append("  By height bin:")
        order = {lab: i for i, lab in enumerate(HEIGHT_BIN_LABELS)}
        by_hb_sorted = by_hb.sort_values(
            "height_bin", key=lambda s: s.map(lambda x: order.get(x, 999))
        )
        for _, r in by_hb_sorted.iterrows():
            lines.append(f"  {str(r['height_bin']):<14s} {r['coverage']:>8.3f} {r['mmd']:>10.4f} "
                         f"{r['one_nna']:>8.3f} {int(r['n_real']):>8d} {int(r['n_strata']):>9d}")
    lines.append("─" * 72)
    return "\n".join(lines)


# ── Table building ───────────────────────────────────────────────────────────

def _median_per_tree(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """Median of each metric across the K generations per conditioning tree.

    Returns a DataFrame indexed by real_id with one row per conditioning tree.
    Downstream tables then take the median of these per-tree medians, so each
    conditioning tree contributes equally regardless of how many generations
    survived filtering.
    """
    return df_pairs.groupby("real_id")[METRICS].median()



def _format_val(v, fmt=".4f"):
    return f"{v:{fmt}}" if pd.notna(v) else "—"


def build_table_1(
    df_pairs: pd.DataFrame,
    df_intra: pd.DataFrame,
    df_inter: pd.DataFrame,
    w1_global: pd.DataFrame,
) -> str:
    """Table 1: Global summary.

    For each metric: mean of per-tree medians for gen / intra / inter.
    Plus population W₁.
    """
    # Per-tree medians, then mean across trees
    gen_vals = _median_per_tree(df_pairs).mean()
    intra_vals = df_intra.groupby("anchor_id")[METRICS].median().mean()
    inter_vals = df_inter.groupby("anchor_id")[METRICS].median().mean()

    lines = []
    lines.append("TABLE 1: GLOBAL SUMMARY")
    lines.append("=" * 100)

    # Part A: Conditioning fidelity — mean of medians
    lines.append("")
    lines.append("(a) Conditioning fidelity — mean of per-tree medians")
    lines.append("─" * 100)
    header = f"  {'Metric':<35s} {'Gen':>8s} {'Intra':>8s} {'Inter':>8s}"
    lines.append(header)
    lines.append("─" * 100)

    for m in METRICS:
        display, unit = METRIC_DISPLAY[m]
        label = f"{display} ({unit})" if unit else display
        g = gen_vals[m]
        i = intra_vals[m]
        x = inter_vals[m]
        lines.append(
            f"  {label:<35s} {_format_val(g):>8s} {_format_val(i):>8s} "
            f"{_format_val(x):>8s}"
        )
    lines.append("─" * 100)

    # Part B: Population distributions
    lines.append("")
    lines.append("(b) Population distributions — W₁ distance (real vs generated)")
    lines.append("─" * 72)
    header = f"  {'Property':<35s} {'W₁':>10s} {'Unit':>6s} {'n_real':>7s} {'n_gen':>7s}"
    lines.append(header)
    lines.append("─" * 72)
    for _, row in w1_global.iterrows():
        lines.append(
            f"  {row['display']:<35s} {row['w1']:>10.4f} {row['unit']:>6s} "
            f"{int(row['n_real']):>7d} {int(row['n_gen']):>7d}"
        )
    lines.append("─" * 72)

    return "\n".join(lines)


def build_stratified_table(
    df_pairs: pd.DataFrame,
    df_real_feats: pd.DataFrame,
    w1_df: pd.DataFrame,
    group_col: str,
    title: str,
) -> str:
    """Tables 2/3: By genus or height bin.

    Two sub-tables per stratum:
      (a) Conditioning fidelity — gen mean-of-medians for all 6 metrics.
      (b) Population W₁ per morphological property.
    """
    groups = sorted(df_pairs[group_col].dropna().unique(),
                    key=lambda x: (HEIGHT_BIN_LABELS.index(x)
                                   if x in HEIGHT_BIN_LABELS else 999, x))

    # W₁ lookup
    w1_lookup = {}
    if not w1_df.empty and group_col in w1_df.columns:
        for _, row in w1_df.iterrows():
            w1_lookup[(row[group_col], row["property"])] = row["w1"]

    # Short names for metrics
    metric_short = {
        "chamfer_dist": "CD",
        "delta_h_max_cr": "Δ HmCR",
        "delta_max_crown_r": "Δ CrR",
        "delta_hcb": "Δ HCB",
        "vert_kde_jsd": "V-KDE",
        "hist_2d_jsd": "H2D",
    }

    lines = []
    lines.append(title)
    lines.append("=" * 100)

    # (a) Conditioning fidelity — mean of medians
    lines.append("")
    lines.append("(a) Conditioning fidelity — mean of per-tree medians")
    lines.append("─" * 100)

    h_parts = [f"  {group_col:<15s} {'n':>5s}"]
    for m in METRICS:
        h_parts.append(f"{metric_short[m]:>10s}")
    lines.append("".join(h_parts))
    lines.append("─" * 100)

    for g in groups:
        g_pairs = df_pairs[df_pairs[group_col] == g]
        n = g_pairs["real_id"].nunique()
        gen_vals = _median_per_tree(g_pairs).mean()

        parts = [f"  {str(g)[:15]:<15s} {n:>5d}"]
        for m in METRICS:
            parts.append(f"{_format_val(gen_vals.get(m, float('nan'))):>10s}")
        lines.append("".join(parts))
    lines.append("─" * 100)

    # (b) Population W₁
    lines.append("")
    lines.append("(b) Population W₁ (real vs generated)")
    lines.append("─" * 100)

    w1_h_parts = [f"  {group_col:<15s} {'n':>5s}"]
    for prop in MORPH_PROPERTIES:
        _, unit = MORPH_DISPLAY[prop]
        short = {"h_max_cr": "W₁ HmCR", "max_crown_r": "W₁ CrR", "hcb": "W₁ HCB"}[prop]
        w1_h_parts.append(f"{short + (' (' + unit + ')' if unit else ''):>16s}")
    lines.append("".join(w1_h_parts))
    lines.append("─" * 100)

    for g in groups:
        n = df_pairs[df_pairs[group_col] == g]["real_id"].nunique()
        parts = [f"  {str(g)[:15]:<15s} {n:>5d}"]
        for prop in MORPH_PROPERTIES:
            w1 = w1_lookup.get((g, prop), float("nan"))
            parts.append(f"{_format_val(w1, '.2f'):>16s}")
        lines.append("".join(parts))
    lines.append("─" * 100)

    return "\n".join(lines)


# ── Save and print ───────────────────────────────────────────────────────────

def save_results(
    output_dir: Path,
    df_pairs: pd.DataFrame,
    df_real_feats: pd.DataFrame,
    df_gen_feats: pd.DataFrame,
    tables: dict[str, str],
    w1_global: pd.DataFrame,
    w1_genus: pd.DataFrame,
    w1_height: pd.DataFrame,
    run_args: dict | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    df_pairs.to_csv(output_dir / "df_pairs.csv", index=False)
    print(f"  Saved df_pairs.csv ({len(df_pairs)} rows)")

    # Save feature dataframes (drop array columns for CSV)
    scalar_cols = [c for c in df_real_feats.columns
                   if c not in ("vert_kde", "hist_2d")]
    df_real_feats[scalar_cols].to_csv(output_dir / "df_real_features.csv")
    df_gen_feats.to_csv(output_dir / "df_gen_features.csv")
    print(f"  Saved feature CSVs (real: {len(df_real_feats)}, gen: {len(df_gen_feats)})")

    for name, w1_df in [("w1_global", w1_global), ("w1_genus", w1_genus),
                        ("w1_height", w1_height)]:
        if not w1_df.empty:
            w1_df.to_csv(output_dir / f"{name}.csv", index=False)

    # Save formatted tables as text
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    for name, text in tables.items():
        (tables_dir / f"{name}.txt").write_text(text)

    # Summary JSON
    summary = {
        "eval_version": 3,
        "n_pairs": len(df_pairs),
        "n_real_trees": df_pairs["real_id"].nunique(),
        "n_gen_trees": len(df_pairs),
    }
    if run_args:
        summary["args"] = run_args
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary.json")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TreeFlow evaluation v3: stem-tracker morphological metrics"
    )
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/preprocessed-4096")
    parser.add_argument("--experiments_dir", type=str, default="experiments")
    parser.add_argument("--max_points", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cov_mmd", action="store_true",
                        help="Also compute supplementary COV/MMD/1-NNA population metrics.")
    parser.add_argument("--cov_mmd_points", type=int, default=2048,
                        help="Points per cloud for COV/MMD/1-NNA (PointFlow/Diff-Tree use 2048).")
    parser.add_argument("--max_per_stratum", type=int, default=150,
                        help="Max trees per side per stratum for the COV/MMD cross matrix.")
    parser.add_argument("--max_per_nna", type=int, default=75,
                        help="Max trees per side per stratum for 1-NNA (rr+gg+rg matrices).")
    parser.add_argument("--min_per_stratum", type=int, default=10,
                        help="Min trees per side to include a (genus, height bin) stratum.")
    args = parser.parse_args()

    experiment_dir = Path(args.experiments_dir) / args.experiment_name
    data_path = Path(args.data_path)
    zarr_dir = experiment_dir / "samples" / "zarr"
    output_dir = experiment_dir / "samples" / "evaluation"

    for p, name in [(experiment_dir, "Experiment"), (zarr_dir, "Samples")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    print(f"Experiment: {args.experiment_name}")
    print(f"Data: {data_path}")
    print(f"Samples: {zarr_dir}")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.num_workers}")
    print()

    t_start = time.time()

    # 1. Load metadata
    real_meta, gen_meta = load_metadata(data_path, experiment_dir)

    # 2. Extract features and build pair dataframe
    df_pairs, df_real_feats, df_gen_feats, real_clouds, gen_clouds = build_pair_dataframe(
        real_meta, gen_meta, zarr_dir,
        max_points=args.max_points, num_workers=args.num_workers, seed=args.seed,
    )
    if df_pairs.empty:
        print("ERROR: No valid pairs. Check data paths.")
        return

    # 3. Baselines
    print()
    df_intra, df_inter = build_baselines(
        df_real_feats, real_clouds,
        num_workers=args.num_workers, seed=args.seed + 1000,
    )

    # 4. Population W₁
    print("\nComputing population W₁...")
    w1_global = compute_population_w1(df_real_feats, df_gen_feats)
    w1_genus  = compute_population_w1(df_real_feats, df_gen_feats, group_col="genus")
    w1_height = compute_population_w1(df_real_feats, df_gen_feats, group_col="height_bin")

    # 4b. Supplementary population metrics (COV / MMD / 1-NNA), opt-in.
    pop_strata = pd.DataFrame()
    pop_global, pop_by_hb = {}, pd.DataFrame()
    if args.cov_mmd:
        print("\nComputing supplementary population metrics (COV / MMD / 1-NNA)...")
        pop_strata = compute_population_cov_mmd(
            df_real_feats, df_gen_feats, real_clouds, gen_clouds,
            num_workers=args.num_workers, n_points=args.cov_mmd_points,
            max_per_stratum=args.max_per_stratum, max_per_nna=args.max_per_nna,
            min_per_stratum=args.min_per_stratum, seed=args.seed + 2000,
        )
        pop_global, pop_by_hb = aggregate_population_metrics(pop_strata)

    # 5. Build tables
    print("\nBuilding tables...")
    # Add group columns to df_pairs for stratified tables
    id_to_genus = df_real_feats["genus"].to_dict()
    id_to_hbin  = df_real_feats["height_bin"].to_dict()
    if "genus" not in df_pairs.columns:
        df_pairs["genus"] = df_pairs["real_id"].map(id_to_genus)
    if "height_bin" not in df_pairs.columns:
        df_pairs["height_bin"] = df_pairs["real_id"].map(id_to_hbin)

    table_1 = build_table_1(df_pairs, df_intra, df_inter, w1_global)
    table_2 = build_stratified_table(
        df_pairs, df_real_feats, w1_genus,
        "genus", "TABLE 2: BY GENUS",
    )
    table_3 = build_stratified_table(
        df_pairs, df_real_feats, w1_height,
        "height_bin", "TABLE 3: BY HEIGHT BIN",
    )

    tables = {"table_1_global": table_1, "table_2_genus": table_2,
              "table_3_height": table_3}
    if args.cov_mmd:
        tables["table_4_population"] = build_population_table(
            pop_strata, pop_global, pop_by_hb)

    # 7. Print
    for t in tables.values():
        print()
        print(t)

    # 8. Save
    print(f"\nSaving results to {output_dir}...")
    save_results(output_dir, df_pairs, df_real_feats, df_gen_feats,
                 tables, w1_global, w1_genus, w1_height,
                 run_args=vars(args))

    # Save supplementary population metrics (CSV + JSON) alongside the tables.
    if args.cov_mmd and not pop_strata.empty:
        pop_strata.to_csv(output_dir / "pop_metrics_strata.csv", index=False)
        if not pop_by_hb.empty:
            pop_by_hb.to_csv(output_dir / "pop_metrics_by_height.csv", index=False)
        with open(output_dir / "pop_metrics_global.json", "w") as f:
            json.dump(pop_global, f, indent=2)
        print(f"  Saved population metrics (COV/MMD/1-NNA) CSV + JSON")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
