"""
novelty.py — memorization / novelty analysis (reviewer #3 backbone).

Tests whether TreeFlow generates *novel* trees or merely reproduces training
examples. For each query cloud we find its nearest neighbour in the training set
(same species, Chamfer on shape-normalized clouds) and report that distance.

Three query sets place the result on a calibrated number line:
  - gen   : TreeFlow's valid (non-degenerate) generations            → the question
  - test  : held-out real test trees                                  → genuine-novelty reference
  - train : training trees vs. their nearest *other* training tree    → memorization floor
            (this is where the retrieval baseline sits, since it emits training trees)

If gen ≈ test, the model's samples are as far from training as real held-out
trees are (i.e. novel), while COV/MMD already show they are realistic. If gen ≈
train-floor (≈0), the model is memorizing. Run on the retrieval experiment as a
sanity check: its gen→train should be ≈ the floor (≈0).

Nearest-neighbour pool is same-SPECIES (any height); results are reported overall
and stratified by height bin. Degenerate generations (failed stem fits, from the
experiment's evaluation CSVs) are excluded, consistent with the main analysis.

Parallelism: queries are distributed across workers; each worker caches the
KD-trees of the training pool for the species it is currently processing (tasks
are species-sorted), so the trees are built once and reused across that species'
queries. Chamfer via cKDTree is exact and identical to evaluate.py's cdist form.
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
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import canonicalize, get_height_bin, HEIGHT_BIN_LABELS, chamfer_distance


# ── Cloud loading ─────────────────────────────────────────────────────────────

def _load_worker(task: tuple) -> tuple | None:
    """Load one cloud, subsample to n_points, shape-canonicalize (no metric scale)."""
    cid, path, n_points, seed = task
    try:
        pts = zarr.load(path).astype(np.float64)
        if n_points and len(pts) > n_points:
            rng = np.random.default_rng(seed)
            pts = pts[rng.choice(len(pts), n_points, replace=False)]
        return cid, canonicalize(pts)
    except Exception as e:  # noqa: BLE001
        return cid, ("_err", str(e))


def load_clouds(tasks: list, num_workers: int, desc: str) -> dict:
    out = {}
    if num_workers <= 1:
        it = (_load_worker(t) for t in tasks)
        for cid, res in tqdm(it, total=len(tasks), desc=desc):
            if isinstance(res, np.ndarray):
                out[cid] = res
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for cid, res in tqdm(pool.map(_load_worker, tasks, chunksize=16),
                                 total=len(tasks), desc=desc):
                if isinstance(res, np.ndarray):
                    out[cid] = res
    return out


# ── Nearest-neighbour search (fork-shared clouds, per-worker train-tree cache) ─

_CLOUDS: dict = {}            # id -> (n_points, 3) shape-canonical points
_SPECIES_TRAIN: dict = {}     # species -> [train ids]
_TREE_CACHE: dict = {}        # worker-local: train id -> cKDTree
_CACHE_SP = [None]            # worker-local: species the cache is built for


def _train_tree(tid):
    t = _TREE_CACHE.get(tid)
    if t is None:
        t = cKDTree(_CLOUDS[tid])
        _TREE_CACHE[tid] = t
    return t


def _nn_worker(task: tuple) -> tuple:
    """Min Chamfer from query qid to any same-species training tree."""
    qid, species, exclude_id = task
    if _CACHE_SP[0] != species:        # bound memory: keep one species' trees
        _TREE_CACHE.clear()
        _CACHE_SP[0] = species
    pa = _CLOUDS[qid]
    qt = cKDTree(pa)                   # query tree, built once, reused over pool
    best, best_id = np.inf, None
    for tid in _SPECIES_TRAIN.get(species, ()):
        if tid == exclude_id:
            continue
        pb = _CLOUDS[tid]
        d_ab, _ = _train_tree(tid).query(pa)   # each query pt -> nearest train pt
        d_ba, _ = qt.query(pb)                 # each train pt -> nearest query pt
        c = 0.5 * (d_ab.mean() + d_ba.mean())
        if c < best:
            best, best_id = c, tid
    return qid, float(best), best_id


def run_nn(tasks: list, num_workers: int, desc: str) -> dict:
    """tasks already species-sorted for cache locality. Returns qid -> (dist, nn_id)."""
    res = {}
    if num_workers <= 1:
        for t in tqdm(tasks, desc=desc):
            qid, d, nid = _nn_worker(t)
            res[qid] = (d, nid)
    else:
        chunk = max(1, len(tasks) // (num_workers * 8))
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for qid, d, nid in tqdm(pool.map(_nn_worker, tasks, chunksize=chunk),
                                    total=len(tasks), desc=desc):
                res[qid] = (d, nid)
    return res


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Novelty / memorization analysis")
    ap.add_argument("--experiment_name", required=True)
    ap.add_argument("--data_path", default="data/preprocessed-16384")
    ap.add_argument("--experiments_dir", default="experiments")
    ap.add_argument("--n_points", type=int, default=2048,
                    help="Points per cloud for the NN Chamfer (PointFlow/COV-MMD convention).")
    ap.add_argument("--num_workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_floor_sample", type=int, default=200,
                    help="Train trees per species sampled as queries for the memorization floor.")
    args = ap.parse_args()

    exp_dir = Path(args.experiments_dir) / args.experiment_name
    data_path = Path(args.data_path)
    eval_dir = exp_dir / "samples" / "evaluation"
    out_dir = exp_dir / "samples" / "novelty"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # 1. Metadata: train / test pools (existing zarr only), keyed by species.
    meta = pd.read_csv(data_path / "metadata.csv")
    meta["file_id"] = meta["filename"].apply(lambda x: Path(x).stem)
    meta["path"] = meta["file_id"].apply(lambda x: str(data_path / f"{x}.zarr"))
    meta = meta[meta["path"].apply(lambda p: Path(p).exists())].copy()
    train = meta[meta["split"] == "train"]
    test = meta[meta["split"] == "test"]
    print(f"Train pool: {len(train)}  Test: {len(test)}  Species: {train['species'].nunique()}")

    # 2. Generated samples; exclude degenerate (stem-fit failures) using eval CSVs.
    sm = pd.read_csv(exp_dir / "samples" / "samples_metadata.csv")
    sm["gen_id"] = sm["sample_file"].apply(lambda x: Path(x).stem)
    sm["path"] = sm["sample_file"].apply(lambda x: str(exp_dir / "samples" / "zarr" / x))
    gfp, rfp = eval_dir / "df_gen_features.csv", eval_dir / "df_real_features.csv"
    if gfp.exists() and rfp.exists():
        gf = pd.read_csv(gfp, dtype={"gen_id": str})
        T_real = float(pd.read_csv(rfp)["max_crown_r"].max())
        failed = set(gf.loc[gf["max_crown_r"] > T_real, "gen_id"].astype(str))
        sm = sm[~sm["gen_id"].isin(failed)].copy()
        print(f"Excluded {len(failed)} degenerate generations (crown radius > T_real={T_real:.2f} m)")
    else:
        print(f"WARNING: {gfp} / {rfp} not found — keeping all generations (run evaluate.py first).")
    sm = sm[sm["path"].apply(lambda p: Path(p).exists())]
    print(f"Generated (valid): {len(sm)}")

    # 3. Load + shape-canonicalize all needed clouds.
    load_tasks = []
    for _, r in train.iterrows():
        load_tasks.append((r["file_id"], r["path"], args.n_points, args.seed))
    for _, r in test.iterrows():
        load_tasks.append((r["file_id"], r["path"], args.n_points, args.seed))
    for _, r in sm.iterrows():
        load_tasks.append((r["gen_id"], r["path"], args.n_points, args.seed))
    print(f"\nLoading {len(load_tasks)} clouds...")
    clouds = load_clouds(load_tasks, args.num_workers, "Load clouds")

    # Species lookups.
    train_sp = train.set_index("file_id")["species"].to_dict()
    species_train: dict = {}
    for fid, sp in train_sp.items():
        if fid in clouds:
            species_train.setdefault(sp, []).append(fid)
    sp_test = test.set_index("file_id")["species"].to_dict()
    h_test = test.set_index("file_id")["tree_H"].astype(float).to_dict()
    h_train = train.set_index("file_id")["tree_H"].astype(float).to_dict()

    # 4. Build query tasks (species-sorted for cache locality) + metadata.
    tasks, qmeta = [], {}
    for _, r in sm.iterrows():
        gid, sp = r["gen_id"], r["species"]
        if gid in clouds and sp in species_train:
            tasks.append((gid, sp, None))
            qmeta[gid] = ("gen", sp, get_height_bin(float(r["height_m"])))
    for fid, sp in sp_test.items():
        if fid in clouds and sp in species_train:
            tasks.append((fid, sp, None))
            qmeta[fid] = ("test", sp, get_height_bin(h_test[fid]))
    for sp, ids in species_train.items():
        pool_ids = ids if len(ids) <= args.train_floor_sample else \
            sorted(rng.choice(ids, args.train_floor_sample, replace=False).tolist())
        for fid in pool_ids:
            tasks.append((fid, sp, fid))   # exclude self
            qmeta[fid] = ("train", sp, get_height_bin(h_train[fid]))
    tasks.sort(key=lambda t: t[1])          # sort by species
    n_by_set = pd.Series([qmeta[t[0]][0] for t in tasks]).value_counts().to_dict()
    print(f"\nNN queries: {len(tasks)}  ({n_by_set})")

    # 5. Run NN search (fork-shares clouds + species index).
    global _CLOUDS, _SPECIES_TRAIN
    _CLOUDS = clouds
    _SPECIES_TRAIN = species_train
    print(f"\nNearest-neighbour Chamfer search ({args.n_points} pts, {args.num_workers} workers)...")
    nn = run_nn(tasks, args.num_workers, "NN search")

    # 6. Assemble + summarize.
    rows = [{"query_id": q, "set": qmeta[q][0], "species": qmeta[q][1],
             "height_bin": qmeta[q][2], "nn_dist": d, "nn_train_id": nid}
            for q, (d, nid) in nn.items()]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "novelty_distances.csv", index=False)

    def stats(s):
        s = s.dropna()
        return dict(n=len(s), median=float(s.median()),
                    q25=float(s.quantile(.25)), q75=float(s.quantile(.75)),
                    mean=float(s.mean()))
    overall = {st: stats(df[df["set"] == st]["nn_dist"]) for st in ["gen", "test", "train"]}

    lines = ["NOVELTY / MEMORIZATION (nearest same-species training tree, Chamfer on shape)",
             "=" * 78,
             "gen vs train-floor (≈ retrieval) vs test (genuine-novelty reference).",
             "gen ≈ test ⇒ novel like held-out real trees; gen ≈ floor ⇒ memorizing.", ""]
    lines.append(f"  {'set':<8s}{'n':>7s}{'median':>9s}{'IQR':>18s}{'mean':>9s}")
    for st in ["train", "gen", "test"]:
        s = overall[st]
        iqr = f"[{s['q25']:.3f}, {s['q75']:.3f}]"
        lines.append(f"  {st:<8s}{s['n']:>7d}{s['median']:>9.4f}{iqr:>18s}{s['mean']:>9.4f}")
    lines.append("")
    lines.append("  By height bin (median nearest-train Chamfer):")
    lines.append(f"    {'bin':<10s}{'train':>9s}{'gen':>9s}{'test':>9s}")

    def _fmt(v):
        return f"{v:>9.4f}" if pd.notna(v) else f"{'—':>9s}"

    by_h_rows = []
    for hb in HEIGHT_BIN_LABELS:
        sub = df[df["height_bin"] == hb]
        if sub.empty:
            continue
        med = {st: sub.loc[sub["set"] == st, "nn_dist"].median() for st in ["train", "gen", "test"]}
        by_h_rows.append({"height_bin": hb, **{f"{k}_median": med[k] for k in med}})
        lines.append(f"    {hb:<10s}{_fmt(med['train'])}{_fmt(med['gen'])}{_fmt(med['test'])}")
    lines.append("=" * 78)
    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "novelty_summary.txt").write_text(report)
    pd.DataFrame(by_h_rows).to_csv(out_dir / "novelty_by_height.csv", index=False)
    with open(out_dir / "novelty_overall.json", "w") as f:
        json.dump(overall, f, indent=2)

    # 7. Figure: distributions overall.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    hi = np.nanpercentile(df["nn_dist"], 99)
    bins = np.linspace(0, hi, 60)
    for st, c in [("train", "0.6"), ("gen", "#1f77b4"), ("test", "#2ca02c")]:
        v = df[df["set"] == st]["nn_dist"].dropna().values
        ax.hist(v, bins=bins, density=True, histtype="step", lw=2, color=c,
                label=f"{st} (med {np.median(v):.3f})")
    ax.set_xlabel("Chamfer distance to nearest same-species training tree (shape-normalized)")
    ax.set_ylabel("density")
    ax.legend(title="query set")
    fig.tight_layout()
    fig.savefig(out_dir / "novelty_distributions.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "novelty_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved novelty results to {out_dir}/  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
