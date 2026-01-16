"""
preprocess_laz_to_npy.py - Convert LAZ files to normalized NPY format

Normalization is the ONLY place point clouds are transformed:
    points_norm = (points - centroid) / z_extent * 2.0

This produces points in approximately [-1, 1] range, matching Gaussian noise variance.
All downstream code (dataset, training, sampling, evaluation) assumes normalized inputs.
"""

import laspy
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial
import json


def normalize_point_cloud(points):
    """
    Normalize point cloud to ~[-1, 1] range.

    Steps:
        1. Center at mean (Center of Mass) - optimal for flow matching transport
        2. Scale by z-extent (tree height) - preserves aspect ratio
        3. Scale by 2.0 - matches variance of Gaussian noise input

    Args:
        points: (N, 3) point cloud in metric coordinates

    Returns:
        normalized_points: (N, 3) normalized point cloud
        centroid: (3,) original centroid
        z_extent: float, original height (z_max - z_min)
    """
    centroid = points.mean(axis=0)
    points_centered = points - centroid

    z_extent = points[:, 2].max() - points[:, 2].min()
    points_normalized = points_centered / (z_extent + 1e-6)
    points_final = points_normalized * 2.0

    return points_final.astype(np.float32), centroid, float(z_extent)


def process_single_file(laz_path, output_dir, min_points=100):
    """
    Process a single LAZ file: load, normalize, save as NPY.

    Args:
        laz_path: Path to input LAZ file
        output_dir: Directory to save NPY file
        min_points: Minimum points required (skip files with fewer)

    Returns:
        tuple: (file_id, success, stats_or_error)
    """
    try:
        file_id = laz_path.stem
        output_path = output_dir / f"{file_id}.npy"

        # Skip if already processed
        if output_path.exists():
            points = np.load(output_path)
            return file_id, True, {"num_points": len(points), "skipped": True}

        # Read LAZ file
        with laspy.open(laz_path) as laz_file:
            las_data = laz_file.read()
            points = np.array(las_data.xyz, dtype=np.float32)

        num_points_original = len(points)

        if num_points_original < min_points:
            return file_id, False, f"Too few points: {num_points_original}"

        # Normalize
        points_norm, centroid, z_extent = normalize_point_cloud(points)

        # Save
        np.save(output_path, points_norm)

        stats = {
            "num_points": num_points_original,
            "z_extent": z_extent,
            "centroid": centroid.tolist(),
            "bounds_normalized": {
                "x": (float(points_norm[:, 0].min()), float(points_norm[:, 0].max())),
                "y": (float(points_norm[:, 1].min()), float(points_norm[:, 1].max())),
                "z": (float(points_norm[:, 2].min()), float(points_norm[:, 2].max())),
            },
        }
        return file_id, True, stats

    except Exception as e:
        return laz_path.stem, False, str(e)


def preprocess_dataset(
    data_path,
    output_path,
    splits=("train", "test"),
    num_workers=None,
    min_points=100,
):
    """
    Preprocess all LAZ files to normalized NPY format.

    Args:
        data_path: Path to FOR-species20K directory
        output_path: Output directory for NPY files
        splits: List of splits to process ('train', 'test')
        num_workers: Number of parallel workers (default: CPU count)
        min_points: Minimum points required per file
    """
    data_path = Path(data_path)
    output_path = Path(output_path)

    if num_workers is None:
        num_workers = mp.cpu_count()

    print(f"Preprocessing LAZ → NPY (normalized)")
    print(f"Using {num_workers} workers")
    print(f"Minimum points: {min_points}")

    all_stats = {}

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing: {split}")
        print("=" * 60)

        # Setup paths
        if split == "train":
            laz_dir = data_path / "laz" / "dev"
            out_dir = output_path / "dev"
        elif split == "test":
            laz_dir = data_path / "laz" / "test"
            out_dir = output_path / "test"
        else:
            raise ValueError(f"Unknown split: {split}")

        out_dir.mkdir(parents=True, exist_ok=True)
        laz_files = sorted(laz_dir.glob("*.laz"))

        if len(laz_files) == 0:
            print(f"Warning: No LAZ files found in {laz_dir}")
            continue

        print(f"Found {len(laz_files)} LAZ files")
        print(f"Output directory: {out_dir}")

        # Process in parallel
        process_fn = partial(
            process_single_file, output_dir=out_dir, min_points=min_points
        )

        with mp.Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(process_fn, laz_files),
                    total=len(laz_files),
                    desc=f"Converting {split}",
                )
            )

        # Collect statistics
        successful = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]

        print(f"\nSummary:")
        print(f"  Successful: {len(successful)}/{len(laz_files)}")
        print(f"  Failed: {len(failed)}")

        if successful:
            # Get stats from non-skipped files
            new_stats = [r[2] for r in successful if isinstance(r[2], dict) and not r[2].get("skipped")]
            if new_stats:
                point_counts = [s["num_points"] for s in new_stats]
                z_extents = [s["z_extent"] for s in new_stats]
                print(f"\n  Point counts: min={min(point_counts)}, max={max(point_counts)}, mean={np.mean(point_counts):.0f}")
                print(f"  Z-extents (heights): min={min(z_extents):.2f}m, max={max(z_extents):.2f}m, mean={np.mean(z_extents):.2f}m")

        if failed:
            print(f"\n  Failed files:")
            for file_id, _, error in failed[:5]:
                print(f"    {file_id}: {error}")
            if len(failed) > 5:
                print(f"    ... and {len(failed) - 5} more")

        all_stats[split] = {
            "total": len(laz_files),
            "successful": len(successful),
            "failed": len(failed),
        }

    # Save summary
    summary_path = output_path / "preprocessing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n{'='*60}")
    print("Preprocessing complete!")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")

    return all_stats


def verify_preprocessing(output_path, num_samples=5):
    """Verify preprocessing by checking a few samples."""
    output_path = Path(output_path)

    print(f"\nVerifying preprocessing...")

    for split_dir in ["dev", "test"]:
        npy_dir = output_path / split_dir
        if not npy_dir.exists():
            continue

        npy_files = list(npy_dir.glob("*.npy"))[:num_samples]

        print(f"\n{split_dir}/ ({len(list(npy_dir.glob('*.npy')))} files):")

        for npy_path in npy_files:
            points = np.load(npy_path)
            centroid = points.mean(axis=0)
            bounds = [(points[:, i].min(), points[:, i].max()) for i in range(3)]

            # Check if normalized (should be roughly [-1, 1])
            max_abs = np.abs(points).max()
            is_normalized = max_abs <= 1.5

            status = "OK" if is_normalized else "WARN"
            print(
                f"  [{status}] {npy_path.stem}: {len(points)} pts, "
                f"centroid=[{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}], "
                f"z=[{bounds[2][0]:.2f}, {bounds[2][1]:.2f}]"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess LAZ files to normalized NPY format"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./FOR-species20K",
        help="Path to FOR-species20K directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./FOR-species20K/preprocessed",
        help="Output directory for preprocessed NPY files",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Splits to process (train and/or test)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=100,
        help="Minimum points required per file",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify preprocessing after completion",
    )

    args = parser.parse_args()

    preprocess_dataset(
        data_path=args.data_path,
        output_path=args.output_path,
        splits=args.splits,
        num_workers=args.num_workers,
        min_points=args.min_points,
    )

    if args.verify:
        verify_preprocessing(args.output_path)


if __name__ == "__main__":
    main()
