"""
preprocess_laz.py
"""

import laspy
import numpy as np
import pandas as pd
import zarr
from pathlib import Path
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial
import json
from sklearn.model_selection import train_test_split


def assign_splits(
    csv_path: Path,
    output_csv_path: Path = None,
    split_ratios: tuple = (0.8, 0.1, 0.1),
    seed: int = 42,
):
    """
    Assign train/val/test splits to metadata and save as new CSV.

    This ensures consistent splits across all dataset loads, rather than
    computing splits at runtime which can vary if the dataset changes.

    Args:
        csv_path: Path to input metadata CSV
        output_csv_path: Path to save CSV with splits (defaults to overwriting input)
        split_ratios: Tuple of (train, val, test) ratios
        seed: Random seed for reproducibility

    Returns:
        DataFrame with 'split' column added
    """
    csv_path = Path(csv_path)
    if output_csv_path is None:
        output_csv_path = csv_path
    else:
        output_csv_path = Path(output_csv_path)

    print(f"\nAssigning train/val/test splits to {csv_path}...")

    df = pd.read_csv(csv_path)

    # Normalize ratios
    train_ratio, val_ratio, test_ratio = split_ratios
    total = sum(split_ratios)
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    # Split train vs (val+test)
    train_idx, temp_idx = train_test_split(
        df.index,
        test_size=(1 - train_ratio),
        random_state=seed,
        stratify=df["species"],
    )

    # Split val vs test
    temp_df = df.loc[temp_idx]
    relative_test_ratio = test_ratio / (test_ratio + val_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_df["species"],
    )

    # Assign splits
    df["split"] = None
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    # Save
    df.to_csv(output_csv_path, index=False)

    print(
        f"  Split assignments: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )
    print(f"  Saved to: {output_csv_path}")

    return df


def normalize_point_cloud(points, height):
    """
    Normalize point cloud by height while preserving aspect ratio.

    1. Geometric Center -> 0,0,0 (Symmetric bounds for Flow Matching)
    2. Scale by Height (Preserves aspect ratio)
    3. Scale by 2.0 (Z spans [-1, 1], matching Gaussian noise variance)
    """
    # 1. Center at geometric center (midpoint of bounding box)
    # This ensures symmetric bounds, unlike center of mass which is biased
    # toward regions with more points (e.g., lower canopy in trees)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2.0
    points_centered = points - bbox_center

    # 2. Normalize by height
    # Z-axis now spans [-0.5, 0.5]
    points_normalized = points_centered / height

    # 3. Apply Variance Scaling
    # Z-axis now spans [-1.0, 1.0] to match Gaussian noise input
    points_final = points_normalized * 2.0

    return points_final.astype(np.float32), bbox_center


def process_single_file(laz_path, output_dir, min_points=0):
    """
    Process a single LAZ file and save as Zarr.

    Args:
        laz_path: Path to input LAZ file
        output_dir: Directory to save Zarr file
        min_points: Minimum number of points required (files with fewer are skipped)

    Returns:
        tuple: (file_id, num_points_original, num_points_final, success, stats_dict)
    """
    try:
        file_id = laz_path.stem
        output_path = output_dir / f"{file_id}.zarr"

        # Skip if already processed
        if output_path.exists():
            points = zarr.load(output_path)
            return file_id, None, len(points), True, None

        # Read LAZ file
        with laspy.open(laz_path) as laz_file:
            las_data = laz_file.read()
            points = np.array(las_data.xyz, dtype=np.float32)

        num_points_original = len(points)

        stats = {
            "original_bounds": {
                "x": (points[:, 0].min(), points[:, 0].max()),
                "y": (points[:, 1].min(), points[:, 1].max()),
                "z": (points[:, 2].min(), points[:, 2].max()),
            }
        }

        # Normalize by height
        z_extent = points[:, 2].max() - points[:, 2].min()
        points, centroid = normalize_point_cloud(points, float(z_extent))
        stats["normalization"] = {
            "scale_factor": float(z_extent),
            "centroid": centroid.tolist(),
            "normalized_bounds": {
                "x": (points[:, 0].min(), points[:, 0].max()),
                "y": (points[:, 1].min(), points[:, 1].max()),
                "z": (points[:, 2].min(), points[:, 2].max()),
            },
        }

        # Ensure minimum points
        if len(points) < min_points:
            return file_id, num_points_original, len(points), False, stats

        # Save as Zarr
        z = zarr.open(str(output_path), mode="w", shape=points.shape, dtype=np.float32)
        z[:] = points

        return file_id, num_points_original, len(points), True, stats

    except Exception as e:
        print(f"Error processing {laz_path.name}: {e}")
        return (laz_path.stem, 0, 0, False, None)


def preprocess_dataset(
    data_path,
    output_path,
    num_workers=None,
    min_points=0,
):
    """
    Preprocess LAZ files from FOR-species20K to normalized Zarr format.

    Args:
        data_path: Path to FOR-species20K directory (contains laz/dev/)
        output_path: Output directory for dataset (e.g., data/full or data/4096)
        num_workers: Number of parallel workers (None = CPU count)
        min_points: Minimum number of points required (files with fewer are skipped).
    """
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = mp.cpu_count()

    print(f"Using {num_workers} workers for preprocessing")

    # Get LAZ files from dev folder (the only one with metadata)
    laz_dir = data_path / "laz" / "dev"
    laz_files = sorted(laz_dir.glob("*.laz"))

    if len(laz_files) == 0:
        print(f"Warning: No LAZ files found in {laz_dir}")
        return {}

    print(f"\n{'='*80}")
    print(f"Processing {len(laz_files)} LAZ files")
    print(f"Output: {output_path}")
    if min_points > 0:
        print(f"Min points filter: {min_points}")
    print(f"{'='*80}")

    # Process files in parallel
    process_fn = partial(
        process_single_file,
        output_dir=output_path,
        min_points=min_points,
    )

    with mp.Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_fn, laz_files),
                total=len(laz_files),
                desc="Converting",
            )
        )

    # Collect statistics
    successful_results = [r for r in results if r[3]]
    successful = len(successful_results)
    total_points_original = sum(
        n for _, n, _, _, _ in successful_results if n is not None
    )
    final_points_counts = [n for _, _, n, _, _ in successful_results]
    total_points_final = sum(final_points_counts)

    # Calculate point distribution stats
    point_dist_stats = {}
    if successful > 0:
        points_arr = np.array(final_points_counts)
        p25, p75 = np.percentile(points_arr, [25, 75])
        point_dist_stats = {
            "mean": int(points_arr.mean()),
            "median": int(np.median(points_arr)),
            "std_dev": int(points_arr.std()),
            "min": int(points_arr.min()),
            "max": int(points_arr.max()),
            "25th_percentile": int(p25),
            "75th_percentile": int(p75),
        }

    stats = {
        "total_files": len(laz_files),
        "successful": successful,
        "failed": len(laz_files) - successful,
        "min_points_filter": min_points,
        "total_points_final": total_points_final,
        "point_count_distribution": point_dist_stats,
    }

    print(f"\nSummary:")
    print(f"  Successfully processed: {successful}/{len(laz_files)}")
    print(f"  Total points: {total_points_final:,}")
    if successful > 0:
        dist = stats["point_count_distribution"]
        print(
            f"  Points per file: min={dist['min']:,}, max={dist['max']:,}, mean={dist['mean']:,}"
        )

    # Save summary
    summary_path = output_path / "preprocessing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    return stats


def verify_preprocessing(output_path, num_samples=5):
    """
    Verify that preprocessing worked correctly.

    Args:
        output_path: Path to Zarr files
        num_samples: Number of samples to verify
    """
    print("\n" + "=" * 60)
    print("Verifying preprocessing...")
    print("=" * 60)

    output_path = Path(output_path)

    if not output_path.exists():
        print(f"✗ Directory not found: {output_path}")
        return False

    zarr_files = list(output_path.glob("*.zarr"))[:num_samples]

    if not zarr_files:
        print(f"✗ No zarr files found in {output_path}")
        return False

    all_valid = True
    for zarr_path in zarr_files:
        zarr_points = zarr.load(zarr_path)

        if len(zarr_points) == 0 or zarr_points.shape[1] != 3:
            print(f"✗ {zarr_path.stem}: Invalid shape {zarr_points.shape}")
            all_valid = False
            continue

        # Check if normalized correctly (Z should span ~[-1, 1])
        z_min, z_max = zarr_points[:, 2].min(), zarr_points[:, 2].max()
        is_normalized = abs(z_min + 1.0) < 0.1 and abs(z_max - 1.0) < 0.1

        status = "✓" if is_normalized else "⚠"
        print(
            f"{status} {zarr_path.stem}: {len(zarr_points)} points, z=[{z_min:.2f}, {z_max:.2f}]"
        )

        if not is_normalized:
            all_valid = False

    if all_valid:
        print("\n✓ All verified samples passed!")
    else:
        print("\n⚠ Some samples may not be normalized correctly")

    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess LAZ files to Zarr format with normalization"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./FOR-species20K",
        help="Path to FOR-species20K directory (raw data from Zenodo)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/preprocessed-full",
        help="Output directory for dataset (e.g., data/preprocessed-full or data/preprocessed-4096)",
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=0,
        help="Minimum number of points required (files with fewer are skipped)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--split_ratios",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        help="Train/val/test split ratios (default: 0.8 0.1 0.1)",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Random seed for split assignment (default: 42)",
    )

    args = parser.parse_args()

    # Run preprocessing
    preprocess_dataset(
        data_path=args.data_path,
        output_path=args.output_path,
        num_workers=args.num_workers,
        min_points=args.min_points,
    )

    # Verify
    verify_preprocessing(output_path=args.output_path, num_samples=5)

    # Copy metadata CSV to output directory with splits assigned
    csv_path = Path(args.data_path) / "tree_metadata_dev.csv"
    output_csv_path = Path(args.output_path) / "metadata.csv"
    assign_splits(
        csv_path=csv_path,
        output_csv_path=output_csv_path,
        split_ratios=tuple(args.split_ratios),
        seed=args.split_seed,
    )


if __name__ == "__main__":
    main()
