"""
preprocess_laz.py
"""

import laspy
import numpy as np
import zarr
from pathlib import Path
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial
import json


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
        zarr.save_array(output_path, points)

        return file_id, num_points_original, len(points), True, stats

    except Exception as e:
        print(f"Error processing {laz_path.name}: {e}")
        return (laz_path.stem, 0, 0, False, None)


def preprocess_dataset(
    data_path,
    output_base_path,
    splits=("train", "test"),
    num_workers=None,
    min_points=0,
):
    """
    Preprocess all LAZ files in the dataset.

    Args:
        data_path: Path to FOR-species20K directory
        output_base_path: Base path for Zarr output
        splits: List of splits to process ('train', 'test')
        num_workers: Number of parallel workers (None = CPU count)
        min_points: Minimum number of points required (files with fewer are skipped)
    """
    data_path = Path(data_path)
    output_base_path = Path(output_base_path)

    if num_workers is None:
        num_workers = mp.cpu_count()

    print(f"Using {num_workers} workers for preprocessing")

    all_stats = {}

    print(f"\n{'='*80}")
    print(f"Processing LAZ files")
    print(f"{'='*80}")

    stats = {}

    for split in splits:
        print(f"\n{'-'*60}")
        print(f"Split: {split}")
        print("-" * 60)

        # Setup paths
        if split == "train":
            laz_dir = data_path / "laz" / "dev"
            split_name = "dev"  # Keep original naming
        elif split == "test":
            laz_dir = data_path / "laz" / "test"
            split_name = "test"
        else:
            raise ValueError(f"Unknown split: {split}")

        output_dir = output_base_path / split_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all LAZ files
        laz_files = sorted(laz_dir.glob("*.laz"))

        if len(laz_files) == 0:
            print(f"Warning: No LAZ files found in {laz_dir}")
            continue

        print(f"Found {len(laz_files)} LAZ files to convert to Zarr")
        print(f"Output directory: {output_dir}")

        # Process files in parallel
        process_fn = partial(
            process_single_file,
            output_dir=output_dir,
            min_points=min_points,
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
        successful_results = [r for r in results if r[3]]
        successful = len(successful_results)
        total_points_original = sum(
            n for _, n, _, _, _ in successful_results if n is not None
        )
        final_points_counts = [n for _, _, n, _, _ in successful_results]
        total_points_final = sum(final_points_counts)

        # Collect normalization stats from a few samples
        sample_stats = [s for _, _, _, _, s in successful_results[:10] if s is not None]

        # Calculate point distribution stats
        point_dist_stats = {}
        if successful > 0:
            points_arr = np.array(final_points_counts)
            p25, p75 = np.percentile(points_arr, [25, 75])
            point_dist_stats = {
                "mean": points_arr.mean(),
                "median": np.median(points_arr),
                "std_dev": points_arr.std(),
                "min": points_arr.min(),
                "max": points_arr.max(),
                "25th_percentile": p25,
                "75th_percentile": p75,
            }

        stats[split] = {
            "total_files": len(laz_files),
            "successful": successful,
            "failed": len(laz_files) - successful,
            "total_points_original": (
                total_points_original
                if total_points_original > 0
                else "N/A (already processed)"
            ),
            "total_points_final": total_points_final,
            "point_count_distribution": {
                k: int(v) for k, v in point_dist_stats.items()
            },
            "sample_stats": sample_stats,
        }

        print(f"\n{split.upper()} Split Summary:")
        print(f"  Successfully processed: {successful}/{len(laz_files)}")
        if total_points_original > 0:
            print(f"  Total points (original): {total_points_original:,}")
        print(f"  Total points (final): {total_points_final:,}")
        if successful > 0:
            dist = stats[split]["point_count_distribution"]
            print(f"  Point Count Distribution (final):")
            print(f"    Mean: {dist['mean']:,}")
            print(f"    Median: {dist['median']:,}")
            print(f"    Std Dev: {dist['std_dev']:,}")
            print(f"    Min: {dist['min']:,} | Max: {dist['max']:,}")
            print(
                f"    25%-75%: [{dist['25th_percentile']:,} - {dist['75th_percentile']:,}]"
            )

        # Print sample normalization info
        if sample_stats:
            print(f"\n  Sample normalization stats (first 10 files):")
            for i, s in enumerate(sample_stats):
                if "normalization" in s:
                    norm = s["normalization"]
                    bounds = norm["normalized_bounds"]
                    print(
                        f"    File {i+1}: scale={norm['scale_factor']:.2f}, "
                        f"bounds X:[{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}], "
                        f"Y:[{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}], "
                        f"Z:[{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}]"
                    )

    all_stats["raw"] = stats

    # Save metadata
    metadata_path = output_base_path / "preprocessing_info.txt"
    with open(metadata_path, "w") as f:
        f.write("LAZ to Zarr Preprocessing Summary\n")
        f.write("Normalized by height: Yes\n")
        f.write("=" * 60 + "\n\n")
        for split, split_stats in stats.items():
            f.write(f"{split.upper()} Split:\n")
            for key, value in split_stats.items():
                if key == "point_count_distribution":
                    f.write(f"  {key}:\n")
                    for stat_name, stat_val in value.items():
                        f.write(f"    {stat_name}: {stat_val:,}\n")
                elif key != "sample_stats":  # Skip detailed stats in text file
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

    print(f"\nMetadata saved to {metadata_path}")

    # Save global summary
    summary_path = output_base_path / "preprocessing_summary.json"
    with open(summary_path, "w") as f:
        # A custom JSON encoder could handle numpy types, but this is simpler
        def default_serializer(o):
            if isinstance(o, (np.int64, np.int32)):
                return int(o)
            if isinstance(o, (np.float64, np.float32)):
                return float(o)
            raise TypeError(
                f"Object of type {o.__class__.__name__} is not JSON serializable"
            )

        json.dump(all_stats, f, indent=2, default=default_serializer)

    print(f"\n{'='*80}")
    print("Preprocessing complete!")
    print(f"{'='*80}")
    print(f"Global summary saved to: {summary_path}")
    print(f"Zarr files saved to: {output_base_path}")
    print(f"\nDirectory structure:")
    print(f"  {output_base_path}/")
    for split in splits:
        split_name = "dev" if split == "train" else "test"
        print(f"    └── {split_name}/")

    return all_stats


def verify_preprocessing(data_path, output_path, num_samples=5):
    """
    Verify that preprocessing worked correctly by comparing a few samples.

    Args:
        data_path: Path to original LAZ files
        output_path: Path to Zarr files
        num_samples: Number of samples to verify
    """
    print("\n" + "=" * 60)
    print(f"Verifying preprocessing...")
    print("=" * 60)

    data_path = Path(data_path)
    output_path = Path(output_path)

    # Check dev split
    zarr_dir = output_path / "dev"

    if not zarr_dir.exists():
        print(f"✗ Directory not found: {zarr_dir}")
        return False

    zarr_files = list(zarr_dir.glob("*.zarr"))[:num_samples]

    all_valid = True
    for zarr_path in zarr_files:
        # Load Zarr
        zarr_points = zarr.load(zarr_path)

        # Check basic validity
        if len(zarr_points) == 0 or zarr_points.shape[1] != 3:
            print(f"✗ {zarr_path.stem}: Invalid shape {zarr_points.shape}")
            all_valid = False
            continue

        # Compute statistics
        bbox_center = (zarr_points.min(axis=0) + zarr_points.max(axis=0)) / 2.0
        bounds = [(zarr_points[:, i].min(), zarr_points[:, i].max()) for i in range(3)]

        # Check if normalized correctly:
        # 1. Z should span approximately [-1, 1] (allow small numerical error)
        # 2. Geometric center should be at origin
        z_min, z_max = bounds[2]
        z_range_valid = abs(z_min + 1.0) < 0.1 and abs(z_max - 1.0) < 0.1
        center_valid = np.abs(bbox_center).max() < 0.1
        is_normalized = z_range_valid and center_valid

        status = "✓" if is_normalized else "⚠"
        norm_str = "normalized" if is_normalized else "NOT normalized"

        print(f"{status} {zarr_path.stem}: {len(zarr_points)} points, {norm_str}")
        print(
            f"   Bbox Center: [{bbox_center[0]:.4f}, {bbox_center[1]:.4f}, {bbox_center[2]:.4f}]"
        )
        print(
            f"   Bounds: X:[{bounds[0][0]:.3f}, {bounds[0][1]:.3f}], "
            f"Y:[{bounds[1][0]:.3f}, {bounds[1][1]:.3f}], "
            f"Z:[{bounds[2][0]:.3f}, {bounds[2][1]:.3f}]"
        )

        if not is_normalized:
            all_valid = False

    if all_valid:
        print("\n✓ All verified samples passed!")
    else:
        print("\n✗ Some samples failed verification")

    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess LAZ files to Zarr format with normalization"
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
        default="./FOR-species20K/zarr",
        help="Base path for Zarr files",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Splits to process (train and/or test)",
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
        "--verify",
        action="store_true",
        default=True,
        help="Verify preprocessing by comparing samples",
    )

    args = parser.parse_args()

    # Run preprocessing
    stats = preprocess_dataset(
        data_path=args.data_path,
        output_base_path=args.output_path,
        splits=args.splits,
        num_workers=args.num_workers,
        min_points=args.min_points,
    )

    # Optional verification
    if args.verify:
        verify_preprocessing(
            data_path=args.data_path,
            output_path=args.output_path,
            num_samples=5,
        )


if __name__ == "__main__":
    main()
