"""
TreeFlow evaluation v3: morphological evaluation of generated tree point clouds.

Per-pair metrics:
  - Chamfer distance (m) — symmetric, PCA-canonicalized clouds in metric coords
  - Convex hull volume (m³) — scipy ConvexHull, stem-tracker independent
  - Max crown radius (m) — from stem tracker
  - Height to crown base (m) — from stem tracker
  - Vertical KDE JSD — 1D density along z-axis
  - 2D histogram JSD — radial × arc-length profile from stem tracker

Tables:
  1. Global summary with gen / intra-class / inter-class baselines + W₁
  2. By genus (gen / intra / inter medians + W₁)
  3. By height bin (gen / intra / inter medians + W₁)
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
from scipy.spatial import ConvexHull
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

METRICS = ["chamfer_dist", "delta_hull_vol", "delta_max_crown_r", "delta_hcb",
           "vert_kde_jsd", "hist_2d_jsd"]

METRIC_DISPLAY = {
    "chamfer_dist":      ("Chamfer distance",   "m"),
    "delta_hull_vol":    ("Δ Hull volume",      "m³"),
    "delta_max_crown_r": ("Δ Max crown radius", "m"),
    "delta_hcb":         ("Δ Height to crown base", "m"),
    "vert_kde_jsd":      ("Vertical KDE JSD",  ""),
    "hist_2d_jsd":       ("2D histogram JSD",  ""),
}

MORPH_PROPERTIES = ["hull_volume", "max_crown_r", "hcb"]

MORPH_DISPLAY = {
    "hull_volume":  ("Convex hull volume", "m³"),
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
        hull_volume   (m³) — convex hull of the full point cloud
        max_crown_r   (m)  — from stem tracker (max mean-r per arc-length slice)
        hcb           (m)  — from stem tracker (arc-length density analysis)
        vert_kde      (64,) — 1D KDE of z-axis (no stem tracker)
        hist_2d       (512,) — 2D (r, s) histogram (stem tracker)
    """
    z = cloud[:, 2]
    z_min, z_max = z.min(), z.max()
    scale = height_m / 2.0  # normalized → metric

    # 1. Convex hull volume (stem-tracker independent)
    try:
        hull_volume = float(ConvexHull(cloud).volume * scale ** 3)
    except Exception:
        hull_volume = float("nan")

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

    # 5. Height to crown base (Kneedle on cumulative mean-r)
    hcb_val, _ = compute_hcb(slice_centers, mean_r_per_slice, s_max)

    return {
        "hull_volume":  hull_volume,
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Extract features for all trees and build the pair dataframe.

    Returns (df_pairs, df_real_feats, df_gen_feats, real_clouds).
    real_clouds maps tree_id → canonicalized metric-scale point cloud.
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
            "delta_hull_vol":    abs(gf["hull_volume"]  - rf["hull_volume"]),
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
        {"tree_id": tid, "hull_volume": r["hull_volume"],
         "max_crown_r": r["max_crown_r"], "hcb": r["hcb"],
         "vert_kde": r["vert_kde"], "hist_2d": r["hist_2d"]}
        for tid, r in real_feats.items()
    ]).set_index("tree_id")

    df_gen_feats = pd.DataFrame([
        {"gen_id": gid, "real_id": gen_id_to_meta[gid]["source_tree_id"],
         "hull_volume": r["hull_volume"],
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

    return df_pairs, df_real_feats, df_gen_feats, real_clouds


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
                    "delta_hull_vol":    abs(anchor["hull_volume"]  - nb["hull_volume"]),
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

    For each metric: median across generated trees, intra-class median,
    inter-class median. Plus population W₁.
    """
    # Per-tree medians, then global median
    gen_medians = _median_per_tree(df_pairs).median()
    # Baselines: aggregate per anchor, then median
    intra_medians = df_intra.groupby("anchor_id")[METRICS].median().median()
    inter_medians = df_inter.groupby("anchor_id")[METRICS].median().median()

    lines = []
    lines.append("TABLE 1: GLOBAL SUMMARY")
    lines.append("=" * 72)

    # Part A: Conditioning fidelity
    lines.append("")
    lines.append("(a) Conditioning fidelity — median across test trees")
    lines.append("─" * 72)
    header = f"  {'Metric':<35s} {'Gen':>8s} {'Intra':>8s} {'Inter':>8s}"
    lines.append(header)
    lines.append("─" * 72)

    for m in METRICS:
        display, unit = METRIC_DISPLAY[m]
        label = f"{display} ({unit})" if unit else display
        g = gen_medians[m]
        i = intra_medians[m]
        x = inter_medians[m]
        lines.append(
            f"  {label:<35s} {_format_val(g):>8s} {_format_val(i):>8s} "
            f"{_format_val(x):>8s}"
        )
    lines.append("─" * 72)

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
      (a) Conditioning fidelity — gen median for all 6 metrics.
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
        "delta_hull_vol": "Δ HuV",
        "delta_max_crown_r": "Δ CrR",
        "delta_hcb": "Δ HCB",
        "vert_kde_jsd": "V-KDE",
        "hist_2d_jsd": "H2D",
    }

    lines = []
    lines.append(title)
    lines.append("=" * 100)

    # (a) Conditioning fidelity — gen median only
    lines.append("")
    lines.append("(a) Conditioning fidelity — gen median across test trees")
    lines.append("─" * 100)

    h_parts = [f"  {group_col:<15s} {'n':>5s}"]
    for m in METRICS:
        h_parts.append(f"{metric_short[m]:>10s}")
    lines.append("".join(h_parts))
    lines.append("─" * 100)

    for g in groups:
        g_pairs = df_pairs[df_pairs[group_col] == g]
        n = g_pairs["real_id"].nunique()
        gen_med = _median_per_tree(g_pairs).median()

        parts = [f"  {str(g)[:15]:<15s} {n:>5d}"]
        for m in METRICS:
            parts.append(f"{_format_val(gen_med.get(m, float('nan'))):>10s}")
        lines.append("".join(parts))
    lines.append("─" * 100)

    # (b) Population W₁
    lines.append("")
    lines.append("(b) Population W₁ (real vs generated)")
    lines.append("─" * 100)

    w1_h_parts = [f"  {group_col:<15s} {'n':>5s}"]
    for prop in MORPH_PROPERTIES:
        _, unit = MORPH_DISPLAY[prop]
        short = {"hull_volume": "W₁ HuV", "max_crown_r": "W₁ CrR", "hcb": "W₁ HCB"}[prop]
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
    args = parser.parse_args()

    experiment_dir = Path(args.experiments_dir) / args.experiment_name
    data_path = Path(args.data_path)
    zarr_dir = experiment_dir / "samples" / "zarr"
    output_dir = experiment_dir / "samples" / "evaluation_v3"

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
    df_pairs, df_real_feats, df_gen_feats, real_clouds = build_pair_dataframe(
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

    # 7. Print
    for t in tables.values():
        print()
        print(t)

    # 8. Save
    print(f"\nSaving results to {output_dir}...")
    save_results(output_dir, df_pairs, df_real_feats, df_gen_feats,
                 tables, w1_global, w1_genus, w1_height,
                 run_args=vars(args))

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
