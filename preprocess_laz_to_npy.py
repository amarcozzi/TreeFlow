"""
preprocess_laz_to_npy.py - Convert LAZ files to NPY format with optional voxelization and normalization
"""
import laspy
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial
import json

MIN_POINTS = {
    None: 10000,
    0.05: 5000,
    0.1: 2500,
    0.2: 1000,
}

def normalize_point_cloud(points, height):
    """
    Normalize point cloud by height while preserving aspect ratio.

    1. Center of Mass -> 0,0,0 (Optimal for Flow Matching transport)
    2. Scale by Height (Preserves aspect ratio)
    3. Scale by 2.0 (Matches Variance of Gaussian Noise)
    """
    # 1. Center at mean (Center of Mass)
    centroid = points.mean(axis=0)
    points_centered = points - centroid

    # 2. Normalize by height
    # Z-axis now spans approx [-0.5, 0.5] * aspect_ratio logic
    points_normalized = points_centered / height

    # 3. Apply Variance Scaling
    # Expands Z-axis to approx [-1.0, 1.0] to match Gaussian noise input
    points_final = points_normalized * 2.0

    return points_final.astype(np.float32), centroid


def voxelize_points(points, voxel_size):
    """
    Voxelize point cloud using numpy for efficiency.

    Args:
        points: Input points, shape (N, 3)
        voxel_size: Voxel size in meters

    Returns:
        voxelized_points: Points at voxel centers, shape (M, 3) where M <= N
    """
    if voxel_size is None or voxel_size <= 0:
        return points

    # Compute voxel indices for each point
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    # Find unique voxels (occupied voxels)
    unique_voxels = np.unique(voxel_indices, axis=0)

    # Compute voxel centers: center = (index + 0.5) * voxel_size
    voxel_centers = (unique_voxels + 0.5) * voxel_size

    return voxel_centers.astype(np.float32)


def process_single_file(laz_path, output_dir, voxel_size=None, normalize=True):
    """
    Process a single LAZ file and save as NPY.

    Args:
        laz_path: Path to input LAZ file
        output_dir: Directory to save NPY file
        voxel_size: Optional voxel size for downsampling (None = no voxelization)
        normalize: Whether to normalize to unit cube

    Returns:
        tuple: (file_id, num_points_original, num_points_final, success, stats_dict)
    """
    try:
        file_id = laz_path.stem
        output_path = output_dir / f"{file_id}.npy"

        # Skip if already processed
        if output_path.exists():
            points = np.load(output_path)
            return file_id, None, len(points), True, None

        # Read LAZ file
        with laspy.open(laz_path) as laz_file:
            las_data = laz_file.read()
            points = np.array(las_data.xyz, dtype=np.float32)

        num_points_original = len(points)

        stats = {
            'original_bounds': {
                'x': (points[:, 0].min(), points[:, 0].max()),
                'y': (points[:, 1].min(), points[:, 1].max()),
                'z': (points[:, 2].min(), points[:, 2].max())
            }
        }

        # Voxelize if requested (before normalization for consistent voxel sizes)
        if voxel_size is not None:
            points_before_voxel = len(points)
            points = voxelize_points(points, voxel_size)
            stats['voxelization'] = {
                'voxel_size': voxel_size,
                'points_before': points_before_voxel,
                'points_after': len(points),
                'reduction_ratio': len(points) / points_before_voxel
            }

        # Normalize to unit cube
        if normalize:
            x_extent = points[:, 0].max() - points[:, 0].min()
            y_extent = points[:, 1].max() - points[:, 1].min()
            z_extent = points[:, 2].max() - points[:, 2].min()
            # points_test, scale_factor_test, final_centroid_test = normalize_to_unit_cube(points)
            points, centroid = normalize_point_cloud(points, float(z_extent))
            stats['normalization'] = {
                'scale_factor': float(z_extent),
                'centroid': centroid.tolist(),
                'normalized_bounds': {
                    'x': (points[:, 0].min(), points[:, 0].max()),
                    'y': (points[:, 1].min(), points[:, 1].max()),
                    'z': (points[:, 2].min(), points[:, 2].max())
                }
            }

        # Ensure minimum points
        min_points = MIN_POINTS.get(voxel_size, 1000)
        if len(points) < min_points:
            return file_id, num_points_original, len(points), False, stats

        # Save as NPY (uncompressed for speed)
        np.save(output_path, points)

        return file_id, num_points_original, len(points), True, stats

    except Exception as e:
        print(f"Error processing {laz_path.name}: {e}")
        return (laz_path.stem, 0, 0, False, None)


def preprocess_dataset(
    data_path,
    output_base_path,
    voxel_sizes=None,
    splits=('train', 'test'),
    num_workers=None,
    normalize=True
):
    """
    Preprocess all LAZ files in the dataset with multiple voxel resolutions.

    Args:
        data_path: Path to FOR-species20K directory
        output_base_path: Base path for output (will create subdirs for each voxel size)
        voxel_sizes: List of voxel sizes to process (None, 0.05, 0.1, 0.2, etc.)
                    None = raw (no voxelization)
        splits: List of splits to process ('train', 'test')
        num_workers: Number of parallel workers (None = CPU count)
        normalize: Whether to normalize to unit cube
    """
    data_path = Path(data_path)
    output_base_path = Path(output_base_path)

    if num_workers is None:
        num_workers = mp.cpu_count()

    if voxel_sizes is None:
        voxel_sizes = [None]  # Just process raw

    print(f"Using {num_workers} workers for preprocessing")
    print(f"Normalization: {'ENABLED' if normalize else 'DISABLED'}")
    print(f"Processing voxel sizes: {voxel_sizes}")

    all_stats = {}

    for voxel_size in voxel_sizes:
        # Determine output directory name
        if voxel_size is None:
            voxel_dir_name = "raw"
        else:
            voxel_dir_name = f"voxel_{voxel_size}m"

        print(f"\n{'='*80}")
        print(f"Processing: {voxel_dir_name}")
        print(f"{'='*80}")

        stats = {}

        for split in splits:
            print(f"\n{'-'*60}")
            print(f"Split: {split}")
            print('-'*60)

            # Setup paths
            if split == 'train':
                laz_dir = data_path / "laz" / "dev"
                split_name = "dev"  # Keep original naming
            elif split == 'test':
                laz_dir = data_path / "laz" / "test"
                split_name = "test"
            else:
                raise ValueError(f"Unknown split: {split}")

            output_dir = output_base_path / voxel_dir_name / split_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Get all LAZ files
            laz_files = sorted(laz_dir.glob("*.laz"))

            if len(laz_files) == 0:
                print(f"Warning: No LAZ files found in {laz_dir}")
                continue

            print(f"Found {len(laz_files)} LAZ files")
            print(f"Output directory: {output_dir}")

            # Process files in parallel
            process_fn = partial(
                process_single_file,
                output_dir=output_dir,
                voxel_size=voxel_size,
                normalize=normalize
            )

            with mp.Pool(num_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_fn, laz_files),
                    total=len(laz_files),
                    desc=f"Converting {split}"
                ))

            # Collect statistics
            successful_results = [r for r in results if r[3]]
            successful = len(successful_results)
            total_points_original = sum(n for _, n, _, _, _ in successful_results if n is not None)
            final_points_counts = [n for _, _, n, _, _ in successful_results]
            total_points_final = sum(final_points_counts)

            # Collect normalization stats from a few samples
            sample_stats = [s for _, _, _, _, s in successful_results[:5] if s is not None]

            # Handle case where we skipped already-processed files
            if total_points_original == 0 and successful > 0:
                reduction_pct = None
            else:
                reduction_pct = (total_points_final / total_points_original * 100) if total_points_original > 0 else 0

            # Calculate point distribution stats
            point_dist_stats = {}
            if successful > 0:
                points_arr = np.array(final_points_counts)
                p25, p75 = np.percentile(points_arr, [25, 75])
                point_dist_stats = {
                    'mean': points_arr.mean(),
                    'median': np.median(points_arr),
                    'std_dev': points_arr.std(),
                    'min': points_arr.min(),
                    'max': points_arr.max(),
                    '25th_percentile': p25,
                    '75th_percentile': p75,
                }

            stats[split] = {
                'total_files': len(laz_files),
                'successful': successful,
                'failed': len(laz_files) - successful,
                'total_points_original': total_points_original if total_points_original > 0 else "N/A (already processed)",
                'total_points_final': total_points_final,
                'reduction_pct': f"{reduction_pct:.1f}%" if reduction_pct is not None else "N/A",
                'point_count_distribution': {k: int(v) for k, v in point_dist_stats.items()},
                'sample_stats': sample_stats
            }

            print(f"\n{split.upper()} Split Summary:")
            print(f"  Successfully processed: {successful}/{len(laz_files)}")
            if total_points_original > 0:
                print(f"  Total points (original): {total_points_original:,}")
            print(f"  Total points (final): {total_points_final:,}")
            if successful > 0:
                dist = stats[split]['point_count_distribution']
                print(f"  Point Count Distribution (final):")
                print(f"    Mean: {dist['mean']:,}")
                print(f"    Median: {dist['median']:,}")
                print(f"    Std Dev: {dist['std_dev']:,}")
                print(f"    Min: {dist['min']:,} | Max: {dist['max']:,}")
                print(f"    25%-75%: [{dist['25th_percentile']:,} - {dist['75th_percentile']:,}]")

            if reduction_pct is not None:
                print(f"  Point reduction: {reduction_pct:.1f}% retained")

            # Print sample normalization info
            if normalize and sample_stats:
                print(f"\n  Sample normalization stats (first 5 files):")
                for i, s in enumerate(sample_stats):
                    if 'normalization' in s:
                        norm = s['normalization']
                        bounds = norm['normalized_bounds']
                        print(f"    File {i+1}: scale={norm['scale_factor']:.2f}, "
                              f"bounds X:[{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}], "
                              f"Y:[{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}], "
                              f"Z:[{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}]")

        all_stats[voxel_dir_name] = stats

        # Save metadata for this voxel size
        metadata_path = output_base_path / voxel_dir_name / "preprocessing_info.txt"
        with open(metadata_path, 'w') as f:
            f.write(f"LAZ to NPY Preprocessing Summary\n")
            f.write(f"Voxel size: {voxel_size if voxel_size is not None else 'None (raw)'}\n")
            f.write(f"Normalized to unit cube: {normalize}\n")
            f.write("="*60 + "\n\n")
            for split, split_stats in stats.items():
                f.write(f"{split.upper()} Split:\n")
                for key, value in split_stats.items():
                    if key == 'point_count_distribution':
                        f.write(f"  {key}:\n")
                        for stat_name, stat_val in value.items():
                            f.write(f"    {stat_name}: {stat_val:,}\n")
                    elif key != 'sample_stats':  # Skip detailed stats in text file
                        f.write(f"  {key}: {value}\n")
                f.write("\n")

        print(f"\nMetadata saved to {metadata_path}")

    # Save global summary
    summary_path = output_base_path / "preprocessing_summary.json"
    with open(summary_path, 'w') as f:
        # A custom JSON encoder could handle numpy types, but this is simpler
        def default_serializer(o):
            if isinstance(o, (np.int64, np.int32)):
                return int(o)
            if isinstance(o, (np.float64, np.float32)):
                return float(o)
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
        json.dump(all_stats, f, indent=2, default=default_serializer)


    print(f"\n{'='*80}")
    print("Preprocessing complete!")
    print(f"{'='*80}")
    print(f"Global summary saved to: {summary_path}")
    print(f"NPY files saved to: {output_base_path}")
    print(f"\nDirectory structure:")
    for voxel_size in voxel_sizes:
        if voxel_size is None:
            voxel_dir_name = "raw"
        else:
            voxel_dir_name = f"voxel_{voxel_size}m"
        print(f"  {output_base_path / voxel_dir_name}/")
        for split in splits:
            split_name = "dev" if split == "train" else "test"
            print(f"    └── {split_name}/")

    return all_stats


def verify_preprocessing(data_path, output_path, voxel_dir="raw", num_samples=5):
    """
    Verify that preprocessing worked correctly by comparing a few samples.

    Args:
        data_path: Path to original LAZ files
        output_path: Path to NPY files
        voxel_dir: Which voxel directory to verify
        num_samples: Number of samples to verify
    """
    print("\n" + "="*60)
    print(f"Verifying preprocessing ({voxel_dir})...")
    print("="*60)

    data_path = Path(data_path)
    output_path = Path(output_path)

    # Check dev split
    npy_dir = output_path / voxel_dir / "dev"

    if not npy_dir.exists():
        print(f"✗ Directory not found: {npy_dir}")
        return False

    npy_files = list(npy_dir.glob("*.npy"))[:num_samples]

    all_valid = True
    for npy_path in npy_files:
        # Load NPY
        npy_points = np.load(npy_path)

        # Check basic validity
        if len(npy_points) == 0 or npy_points.shape[1] != 3:
            print(f"✗ {npy_path.stem}: Invalid shape {npy_points.shape}")
            all_valid = False
            continue

        # Check if normalized
        point_range = np.abs(npy_points).max()
        is_normalized = point_range <= 1.1  # Allow small numerical error

        # Compute statistics
        centroid = npy_points.mean(axis=0)
        bounds = [
            (npy_points[:, i].min(), npy_points[:, i].max())
            for i in range(3)
        ]

        status = "✓" if is_normalized else "⚠"
        norm_str = "normalized" if is_normalized else "NOT normalized"

        print(f"{status} {npy_path.stem}: {len(npy_points)} points, {norm_str}")
        print(f"   Centroid: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]")
        print(f"   Bounds: X:[{bounds[0][0]:.3f}, {bounds[0][1]:.3f}], "
              f"Y:[{bounds[1][0]:.3f}, {bounds[1][1]:.3f}], "
              f"Z:[{bounds[2][0]:.3f}, {bounds[2][1]:.3f}]")

        if not is_normalized:
            all_valid = False

    if all_valid:
        print("\n✓ All verified samples passed!")
    else:
        print("\n✗ Some samples failed verification")

    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess LAZ files to NPY format with optional voxelization and normalization"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./FOR-species20K",
        help="Path to FOR-species20K directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./FOR-species20K/npy",
        help="Base path for NPY files (subdirs will be created for each voxel size)"
    )
    parser.add_argument(
        "--voxel_sizes",
        nargs='+',
        type=float,
        default=None,
        help="Voxel sizes to process in meters (e.g., 0.05 0.1 0.2). Omit for raw only."
    )
    parser.add_argument(
        "--include_raw",
        action='store_true',
        help="Also process raw (non-voxelized) version"
    )
    parser.add_argument(
        "--normalize",
        action='store_true',
        default=True,
        help="Normalize to unit cube [-1, 1]³ (default: True)"
    )
    parser.add_argument(
        "--no_normalize",
        dest='normalize',
        action='store_false',
        help="Disable normalization"
    )
    parser.add_argument(
        "--splits",
        nargs='+',
        default=['train', 'test'],
        help="Splits to process (train and/or test)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--verify",
        action='store_true',
        help="Verify preprocessing by comparing samples"
    )
    parser.add_argument(
        "--verify_dir",
        type=str,
        default="raw",
        help="Which directory to verify (e.g., 'raw', 'voxel_0.1m')"
    )

    args = parser.parse_args()

    # Build list of voxel sizes to process
    voxel_sizes_to_process = []
    if args.include_raw:
        voxel_sizes_to_process.append(None)
    if args.voxel_sizes is not None:
        voxel_sizes_to_process.extend(args.voxel_sizes)

    if not voxel_sizes_to_process:
        # Default to raw if nothing is specified
        print("No voxel sizes specified. Defaulting to --include_raw.")
        voxel_sizes_to_process.append(None)


    # Run preprocessing
    stats = preprocess_dataset(
        data_path=args.data_path,
        output_base_path=args.output_path,
        voxel_sizes=voxel_sizes_to_process,
        splits=args.splits,
        num_workers=args.num_workers,
        normalize=args.normalize
    )

    # Optional verification
    if args.verify:
        verify_preprocessing(
            data_path=args.data_path,
            output_path=args.output_path,
            voxel_dir=args.verify_dir,
            num_samples=5
        )


if __name__ == "__main__":
    main()
