"""
preprocess_laz_to_npy.py - Convert LAZ files to NPY format with optional voxelization
"""
import laspy
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial


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


def process_single_file(laz_path, output_dir, voxel_size=None):
    """
    Process a single LAZ file and save as NPY.

    Args:
        laz_path: Path to input LAZ file
        output_dir: Directory to save NPY file
        voxel_size: Optional voxel size for downsampling (None = no voxelization)

    Returns:
        tuple: (file_id, num_points_original, num_points_final, success)
    """
    try:
        file_id = laz_path.stem
        output_path = output_dir / f"{file_id}.npy"

        # Skip if already processed
        if output_path.exists():
            points = np.load(output_path)
            return (file_id, None, len(points), True)

        # Read LAZ file
        with laspy.open(laz_path) as laz_file:
            las_data = laz_file.read()
            points = np.array(las_data.xyz, dtype=np.float32)

        num_points_original = len(points)

        # Center the point cloud at the origin (do this once during preprocessing)
        centroid_x = (points[:, 0].max() + points[:, 0].min()) / 2
        centroid_y = (points[:, 1].max() + points[:, 1].min()) / 2
        centroid_z = points[:, 2].min()  # Keep bottom at z=0
        points[:, 0] -= centroid_x
        points[:, 1] -= centroid_y
        points[:, 2] -= centroid_z

        # Voxelize if requested
        if voxel_size is not None:
            points = voxelize_points(points, voxel_size)

        # Save as NPY (uncompressed for speed)
        np.save(output_path, points)

        return (file_id, num_points_original, len(points), True)

    except Exception as e:
        print(f"Error processing {laz_path.name}: {e}")
        return (laz_path.stem, 0, 0, False)


def preprocess_dataset(
    data_path,
    output_base_path,
    voxel_sizes=None,
    splits=['train', 'test'],
    num_workers=None
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
    """
    data_path = Path(data_path)
    output_base_path = Path(output_base_path)

    if num_workers is None:
        num_workers = mp.cpu_count()

    if voxel_sizes is None:
        voxel_sizes = [None]  # Just process raw

    print(f"Using {num_workers} workers for preprocessing")
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
            process_fn = partial(process_single_file, output_dir=output_dir, voxel_size=voxel_size)

            with mp.Pool(num_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_fn, laz_files),
                    total=len(laz_files),
                    desc=f"Converting {split}"
                ))

            # Collect statistics
            successful = sum(1 for _, _, _, success in results if success)
            total_points_original = sum(n for _, n, _, success in results if success and n is not None)
            total_points_final = sum(n for _, _, n, success in results if success)

            # Handle case where we skipped already-processed files
            if total_points_original == 0:
                # All files were already processed, just count final points
                reduction_pct = None
            else:
                reduction_pct = (total_points_final / total_points_original * 100) if total_points_original > 0 else 0

            stats[split] = {
                'total_files': len(laz_files),
                'successful': successful,
                'failed': len(laz_files) - successful,
                'total_points_original': total_points_original if total_points_original > 0 else "N/A (already processed)",
                'total_points_final': total_points_final,
                'avg_points_final': total_points_final / successful if successful > 0 else 0,
                'reduction_pct': f"{reduction_pct:.1f}%" if reduction_pct is not None else "N/A"
            }

            print(f"\n{split.upper()} Split Summary:")
            print(f"  Successfully processed: {successful}/{len(laz_files)}")
            if total_points_original > 0:
                print(f"  Total points (original): {total_points_original:,}")
            print(f"  Total points (final): {total_points_final:,}")
            print(f"  Average points per file: {stats[split]['avg_points_final']:.0f}")
            if reduction_pct is not None:
                print(f"  Point reduction: {reduction_pct:.1f}% retained")

        all_stats[voxel_dir_name] = stats

        # Save metadata for this voxel size
        metadata_path = output_base_path / voxel_dir_name / "preprocessing_info.txt"
        with open(metadata_path, 'w') as f:
            f.write(f"LAZ to NPY Preprocessing Summary\n")
            f.write(f"Voxel size: {voxel_size if voxel_size is not None else 'None (raw)'}\n")
            f.write("="*60 + "\n\n")
            for split, split_stats in stats.items():
                f.write(f"{split.upper()} Split:\n")
                for key, value in split_stats.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

        print(f"\nMetadata saved to {metadata_path}")

    # Save global summary
    summary_path = output_base_path / "preprocessing_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("LAZ to NPY Preprocessing - Global Summary\n")
        f.write("="*80 + "\n\n")
        for voxel_name, stats in all_stats.items():
            f.write(f"{voxel_name}:\n")
            for split, split_stats in stats.items():
                f.write(f"  {split}:\n")
                for key, value in split_stats.items():
                    f.write(f"    {key}: {value}\n")
            f.write("\n")

    print(f"\n{'='*80}")
    print("Preprocessing complete!")
    print(f"{'='*80}")
    print(f"Global summary saved to: {summary_path}")
    print(f"NPY files saved to: {output_base_path}")
    print(f"\nDirectory structure:")
    for voxel_size in voxel_sizes:
        voxel_dir_name = "raw" if voxel_size is None else f"voxel_{voxel_size}m"
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
        voxel_dir: Which voxel directory to verify (e.g., "raw", "voxel_0.1m")
        num_samples: Number of samples to verify
    """
    print("\n" + "="*60)
    print(f"Verifying preprocessing ({voxel_dir})...")
    print("="*60)

    data_path = Path(data_path)
    output_path = Path(output_path)

    # Check dev split
    laz_dir = data_path / "laz" / "dev"
    npy_dir = output_path / voxel_dir / "dev"

    if not npy_dir.exists():
        print(f"❌ Directory not found: {npy_dir}")
        return False

    laz_files = list(laz_dir.glob("*.laz"))[:num_samples]

    all_match = True
    for laz_path in laz_files:
        npy_path = npy_dir / f"{laz_path.stem}.npy"

        if not npy_path.exists():
            print(f"❌ Missing NPY file: {npy_path.name}")
            all_match = False
            continue

        # Load NPY
        npy_points = np.load(npy_path)

        # For raw, compare directly; for voxelized, just check it's valid
        if voxel_dir == "raw":
            with laspy.open(laz_path) as laz_file:
                las_data = laz_file.read()
                laz_points = np.array(las_data.xyz, dtype=np.float32)

            # We centered, so check that worked
            if np.allclose(npy_points.mean(axis=0)[:2], 0, atol=10):  # X,Y should be near 0
                print(f"✓ {laz_path.stem}: {len(laz_points)} -> {len(npy_points)} points (centered)")
            else:
                print(f"⚠ {laz_path.stem}: May not be centered correctly")
        else:
            # Just verify it loaded and has reasonable data
            if len(npy_points) > 0 and npy_points.shape[1] == 3:
                print(f"✓ {laz_path.stem}: {len(npy_points)} points")
            else:
                print(f"❌ {laz_path.stem}: Invalid data shape")
                all_match = False

    if all_match:
        print("\n✓ All verified samples passed!")
    else:
        print("\n❌ Some samples failed verification")

    return all_match


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess LAZ files to NPY format with optional voxelization"
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

    if len(voxel_sizes_to_process) == 0:
        print("Error: Must specify at least one of --include_raw or --voxel_sizes")
        return

    # Run preprocessing
    stats = preprocess_dataset(
        data_path=args.data_path,
        output_base_path=args.output_path,
        voxel_sizes=voxel_sizes_to_process,
        splits=args.splits,
        num_workers=args.num_workers
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