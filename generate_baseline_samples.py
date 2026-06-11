"""
generate_baseline_samples.py

Nearest-neighbor retrieval-and-scaling baseline for TreeFlow.

For every tree in the test split, this script retrieves the K most similar
trees from the training split (same species, optionally same acquisition
platform, nearest in height) and writes them out as "samples" in exactly the
same on-disk format produced by generate_samples.py. Because preprocessing
normalizes every cloud to a vertical extent of [-1, 1] (height -> 2 units) and
evaluate.py rescales each sample by (target height / 2) at metric-conversion
time, assigning the *target* tree's height to a retrieved cloud isotropically
scales that retrieved tree to the target height. Retrieval therefore reduces to
"copy the neighbor cloud, label it with the target's conditioning attributes."

This is the classic retrieve-and-scale reference (cf. Schaefer et al. and
Li et al.): instead of generating a novel tree, return the closest real scan
from an existing collection, rescaled to the requested size. Retrieval draws
only from the training split, so there is no test-set leakage.

The output directory (experiments/<experiment_name>/samples/) can be scored
directly with evaluate.py, identically to a trained model's samples.

Usage:
    python generate_baseline_samples.py \
        --data_path data/preprocessed-16384 \
        --experiment_name retrieval-baseline-16384 \
        --max_points 16384 \
        --num_samples_per_tree 16 \
        --seed 42
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def load_metadata(data_path: Path) -> pd.DataFrame:
    """Load preprocessed metadata.csv and attach file_id / file_path columns.

    Keeps only rows whose zarr cloud exists on disk (matching dataset.py).
    """
    csv = data_path / "metadata.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv}")
    df = pd.read_csv(csv)
    if "split" not in df.columns:
        raise ValueError("metadata.csv missing 'split' column")
    df["file_id"] = df["filename"].apply(lambda x: Path(x).stem)
    df["file_path"] = df["file_id"].apply(lambda x: str(data_path / f"{x}.zarr"))
    initial = len(df)
    df = df[df["file_path"].apply(lambda x: Path(x).exists())].copy()
    print(f"Found {len(df)}/{initial} trees with matching zarr files")
    if "genus" not in df.columns:
        df["genus"] = "unknown"
    return df


def build_retrieval_plan(
    test_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    k: int,
    match_platform: bool,
) -> tuple[list[dict], dict]:
    """For each test tree, choose its K nearest-in-height retrieval neighbors.

    Candidate preference order:
      1. same species AND same acquisition platform (if >= k available)
      2. same species (any platform)
      3. same genus (any species)      [fallback, should be rare]
      4. entire pool                    [last resort]
    Within the chosen candidate set, neighbors are ranked by |height difference|
    with ties broken by file_id for determinism. The returned plan contains one
    task per test tree with the resolved neighbor list (metadata only -- no
    point clouds are loaded here).

    Returns (tasks, stats).
    """
    # Group pool by species and (species, platform) for fast lookup.
    pool_by_species: dict[str, pd.DataFrame] = {
        s: g for s, g in pool_df.groupby("species")
    }
    pool_by_species_platform: dict[tuple, pd.DataFrame] = {
        key: g for key, g in pool_df.groupby(["species", "data_type"])
    }
    pool_by_genus: dict[str, pd.DataFrame] = {
        g_: grp for g_, grp in pool_df.groupby("genus")
    }

    tasks = []
    stats = {
        "species_platform": 0, "species": 0, "genus": 0, "global": 0,
        "fewer_than_k": 0,
    }

    for _, row in test_df.iterrows():
        species = row["species"]
        platform = row["data_type"]
        target_h = float(row["tree_H"])

        # Resolve candidate set with graceful fallback.
        cand = pool_by_species_platform.get((species, platform))
        if match_platform and cand is not None and len(cand) >= k:
            tier = "species_platform"
        else:
            cand = pool_by_species.get(species)
            if cand is not None and len(cand) > 0:
                tier = "species"
            else:
                cand = pool_by_genus.get(row["genus"])
                if cand is not None and len(cand) > 0:
                    tier = "genus"
                else:
                    cand = pool_df
                    tier = "global"
        stats[tier] += 1

        # Rank by |height difference|, deterministic tie-break on file_id.
        order = cand.assign(_dh=(cand["tree_H"].astype(float) - target_h).abs())
        order = order.sort_values(["_dh", "file_id"], kind="stable")
        chosen = order.head(k)
        if len(chosen) < k:
            stats["fewer_than_k"] += 1

        neighbors = [
            {
                "neighbor_id": c["file_id"],
                "neighbor_path": c["file_path"],
                "neighbor_height": float(c["tree_H"]),
                "neighbor_data_type": c["data_type"],
            }
            for _, c in chosen.iterrows()
        ]

        tasks.append({
            "source_tree_id": row["file_id"],
            "target_height": target_h,
            "species": species,
            "scan_type": platform,
            "genus": row["genus"],
            "match_tier": tier,
            "neighbors": neighbors,
        })

    return tasks, stats


# ── Worker: materialize one test tree's retrieved samples ─────────────────────

_WORKER_CFG: dict = {}


def _init_worker(zarr_dir: str, max_points: int, seed: int):
    _WORKER_CFG["zarr_dir"] = Path(zarr_dir)
    _WORKER_CFG["max_points"] = max_points
    _WORKER_CFG["seed"] = seed


def _write_sample(task: dict) -> list[dict]:
    """Load each neighbor cloud and write it out under the target's conditioning."""
    zarr_dir = _WORKER_CFG["zarr_dir"]
    max_points = _WORKER_CFG["max_points"]
    seed = _WORKER_CFG["seed"]

    rows = []
    for sample_idx, nb in enumerate(task["neighbors"], start=1):
        try:
            points = zarr.load(nb["neighbor_path"]).astype(np.float32)
        except Exception as e:  # noqa: BLE001 - record and skip a bad neighbor
            rows.append({"_error": f"{nb['neighbor_id']}: {e}",
                         "source_tree_id": task["source_tree_id"]})
            continue

        # Cap point count (evaluate.py subsamples to max_points anyway).
        if max_points and len(points) > max_points:
            rng = np.random.default_rng(seed + sample_idx)
            idx = rng.choice(len(points), size=max_points, replace=False)
            points = points[idx]

        sample_id = f"{task['source_tree_id']}_{sample_idx}"
        metadata = {
            "sample_id": sample_id,
            "sample_file": f"{sample_id}.zarr",
            "source_tree_id": task["source_tree_id"],
            "source_split": "test",
            "species": task["species"],
            "scan_type": task["scan_type"],
            "genus": task["genus"],
            # Target conditioning height -> drives the metric rescaling in eval.
            "height_m": float(task["target_height"]),
            "num_points": int(len(points)),
            "cfg_scale": 0.0,
            "method": "retrieval",
            # Provenance of the retrieved exemplar.
            "retrieved_tree_id": nb["neighbor_id"],
            "retrieved_height_m": float(nb["neighbor_height"]),
            "retrieved_data_type": nb["neighbor_data_type"],
            "delta_height_m": float(abs(nb["neighbor_height"] - task["target_height"])),
            "match_tier": task["match_tier"],
            "seed": seed,
            "generation_timestamp": datetime.now().isoformat(),
        }

        out = zarr_dir / f"{sample_id}.zarr"
        z = zarr.open(str(out), mode="w", shape=points.shape, dtype=np.float32)
        z[:] = points
        for kk, vv in metadata.items():
            z.attrs[kk] = vv
        rows.append(metadata)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Nearest-neighbor retrieval-and-scaling baseline for TreeFlow"
    )
    parser.add_argument("--data_path", type=str, default="data/preprocessed-16384")
    parser.add_argument("--experiments_dir", type=str, default="experiments")
    parser.add_argument("--experiment_name", type=str, default="retrieval-baseline")
    parser.add_argument("--max_points", type=int, default=16384,
                        help="Cap on points per saved sample (eval subsamples to this).")
    parser.add_argument("--num_samples_per_tree", type=int, default=16,
                        help="K retrieved neighbors per test tree (match the model's K).")
    parser.add_argument("--retrieval_pool", type=str, default="train",
                        choices=["train", "train+val"],
                        help="Which split(s) form the retrieval database.")
    parser.add_argument("--no_match_platform", action="store_true",
                        help="Do not prefer same acquisition platform when retrieving.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    experiment_dir = Path(args.experiments_dir) / args.experiment_name
    samples_dir = experiment_dir / "samples"
    zarr_dir = samples_dir / "zarr"
    zarr_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data: {data_path}")
    print(f"Output experiment: {experiment_dir}")
    print(f"K (samples per tree): {args.num_samples_per_tree}")
    print(f"Retrieval pool: {args.retrieval_pool}, match_platform: {not args.no_match_platform}")
    print()

    # 1. Load metadata and split into test (targets) / pool (retrieval database).
    df = load_metadata(data_path)
    test_df = df[df["split"] == "test"].copy()
    if args.retrieval_pool == "train+val":
        pool_df = df[df["split"].isin(["train", "val"])].copy()
    else:
        pool_df = df[df["split"] == "train"].copy()
    print(f"Test trees (targets): {len(test_df)}")
    print(f"Retrieval pool trees: {len(pool_df)}")
    if test_df.empty or pool_df.empty:
        print("ERROR: empty test or pool set.")
        return 1

    # 2. Build the retrieval plan (metadata only).
    print("\nResolving nearest-neighbor retrievals...")
    tasks, stats = build_retrieval_plan(
        test_df, pool_df,
        k=args.num_samples_per_tree,
        match_platform=not args.no_match_platform,
    )
    print("  Match tiers: "
          f"species+platform={stats['species_platform']}, "
          f"species={stats['species']}, genus={stats['genus']}, global={stats['global']}")
    if stats["fewer_than_k"]:
        print(f"  {stats['fewer_than_k']} test trees had fewer than K candidates "
              f"(used all available, no replacement).")

    # 3. Materialize samples in parallel.
    print(f"\nWriting samples for {len(tasks)} test trees...")
    all_rows = []
    if args.num_workers <= 1:
        _init_worker(str(zarr_dir), args.max_points, args.seed)
        for t in tqdm(tasks, desc="Retrieve"):
            all_rows.extend(_write_sample(t))
    else:
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_init_worker,
            initargs=(str(zarr_dir), args.max_points, args.seed),
        ) as pool:
            for rows in tqdm(pool.map(_write_sample, tasks, chunksize=4),
                             total=len(tasks), desc="Retrieve"):
                all_rows.extend(rows)

    errors = [r for r in all_rows if "_error" in r]
    good = [r for r in all_rows if "_error" not in r]
    if errors:
        print(f"  {len(errors)} neighbor failures (showing up to 5):")
        for e in errors[:5]:
            print(f"    {e['_error']}")

    # 4. Write samples_metadata.csv (so postprocess_samples.py is not needed).
    meta_df = pd.DataFrame(good)
    meta_csv = samples_dir / "samples_metadata.csv"
    meta_df.to_csv(meta_csv, index=False)
    print(f"\nWrote {len(meta_df)} samples to {zarr_dir}")
    print(f"Wrote metadata to {meta_csv}")

    # 5. Provenance.
    gen_config = {
        "method": "nearest_neighbor_retrieval_and_scaling",
        "experiment_name": args.experiment_name,
        "data_path": str(data_path),
        "max_points": args.max_points,
        "num_samples_per_tree": args.num_samples_per_tree,
        "retrieval_pool": args.retrieval_pool,
        "match_platform": not args.no_match_platform,
        "seed": args.seed,
        "n_test_trees": int(len(test_df)),
        "n_pool_trees": int(len(pool_df)),
        "n_samples_written": int(len(meta_df)),
        "match_tier_counts": stats,
        "generation_timestamp": datetime.now().isoformat(),
    }
    with open(samples_dir / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)

    print("\nDone. Evaluate with:")
    print(f"  python evaluate.py --experiment_name {args.experiment_name} "
          f"--data_path {data_path} --max_points {args.max_points} --cov_mmd")
    return 0


if __name__ == "__main__":
    sys.exit(main())
