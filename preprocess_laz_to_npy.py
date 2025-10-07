"""
preprocess_laz_to_npy.py - Convert LAZ files to NPY format for fast loading
"""
import laspy
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial


def process_single_file(laz_path, output_dir):
    """
    Process a single LAZ file and save as NPY.

    Args:
        laz_path: Path to input LAZ file
        output_dir: Directory to save NPY file

    Returns:
        tuple: (file_id, num_points, success)
    """
    try:
        file_id = laz_path.stem
        output_path = output_dir / f"{file_id}.npy"

        # Skip if already processed
        if output_path.exists():
            points = np.load(output_path)
            return (file_id, len(points), True)

        # Read LAZ file
        with laspy.open(laz_path) as laz_file:
            las_data = laz_file.read()
            points = np.array(las_data.xyz, dtype=np.float32)

        # Center the point cloud at the origin (do this once during preprocessing)
        centroid_x = (points[:, 0].max() + points[:, 0].min()) / 2
        centroid_y = (points[:, 1].max() + points[:, 1].min()) / 2
        centroid_z = (points[:, 2].max() + points[:, 2].min()) / 2
        points[:, 0] -= centroid_x
        points[:, 1] -= centroid_y
        points[:, 2] -= centroid_z

        # Save as NPY (uncompressed for speed)
        np.save(output_path, points)

        return (file_id, len(points), True)

    except Exception as e:
        print(f"Error processing {laz_path.name}: {e}")
        return (laz_path.stem, 0, False)


def preprocess_dataset(
    data_path,
    output_path,
    splits=['train', 'test'],
    num_workers=None
):
    """
    Preprocess all LAZ files in the dataset.

    Args:
        data_path: Path to FOR-species20K directory
        output_path: Path to output directory for NPY files
        splits: List of splits to process ('train', 'test')
        num_workers: Number of parallel workers (None = CPU count)
    """
    data_path = Path(data_path)
    output_path = Path(output_path)

    if num_workers is None:
        num_workers = mp.cpu_count()

    print(f"Using {num_workers} workers for preprocessing")

    stats = {}

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print('='*60)

        # Setup paths
        if split == 'train':
            laz_dir = data_path / "laz" / "dev"
            split_name = "dev"  # Keep original naming
        elif split == 'test':
            laz_dir = data_path / "laz" / "test"
            split_name = "test"
        else:
            raise ValueError(f"Unknown split: {split}")

        output_dir = output_path / split_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all LAZ files
        laz_files = sorted(laz_dir.glob("*.laz"))

        if len(laz_files) == 0:
            print(f"Warning: No LAZ files found in {laz_dir}")
            continue

        print(f"Found {len(laz_files)} LAZ files")
        print(f"Output directory: {output_dir}")

        # Process files in parallel
        process_fn = partial(process_single_file, output_dir=output_dir)

        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_fn, laz_files),
                total=len(laz_files),
                desc=f"Converting {split}"
            ))

        # Collect statistics
        successful = sum(1 for _, _, success in results if success)
        total_points = sum(n for _, n, success in results if success)

        stats[split] = {
            'total_files': len(laz_files),
            'successful': successful,
            'failed': len(laz_files) - successful,
            'total_points': total_points,
            'avg_points': total_points / successful if successful > 0 else 0
        }

        print(f"\n{split.upper()} Split Summary:")
        print(f"  Successfully processed: {successful}/{len(laz_files)}")
        print(f"  Total points: {total_points:,}")
        print(f"  Average points per file: {stats[split]['avg_points']:.0f}")

    # Save metadata
    metadata_path = output_path / "preprocessing_info.txt"
    with open(metadata_path, 'w') as f:
        f.write("LAZ to NPY Preprocessing Summary\n")
        f.write("="*60 + "\n\n")
        for split, split_stats in stats.items():
            f.write(f"{split.upper()} Split:\n")
            for key, value in split_stats.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    print(f"\nPreprocessing complete! Metadata saved to {metadata_path}")
    print(f"NPY files saved to {output_path}")

    return stats


def verify_preprocessing(data_path, output_path, num_samples=5):
    """
    Verify that preprocessing worked correctly by comparing a few samples.

    Args:
        data_path: Path to original LAZ files
        output_path: Path to NPY files
        num_samples: Number of samples to verify
    """
    print("\n" + "="*60)
    print("Verifying preprocessing...")
    print("="*60)

    data_path = Path(data_path)
    output_path = Path(output_path)

    # Check dev split
    laz_dir = data_path / "laz" / "dev"
    npy_dir = output_path / "dev"

    laz_files = list(laz_dir.glob("*.laz"))[:num_samples]

    all_match = True
    for laz_path in laz_files:
        npy_path = npy_dir / f"{laz_path.stem}.npy"

        if not npy_path.exists():
            print(f"❌ Missing NPY file: {npy_path.name}")
            all_match = False
            continue

        # Load both
        with laspy.open(laz_path) as laz_file:
            las_data = laz_file.read()
            laz_points = np.array(las_data.xyz, dtype=np.float32)

        npy_points = np.load(npy_path)

        # Compare
        if np.allclose(laz_points, npy_points):
            print(f"✓ {laz_path.stem}: {len(laz_points)} points match")
        else:
            print(f"❌ {laz_path.stem}: Data mismatch!")
            all_match = False

    if all_match:
        print("\n✓ All verified samples match!")
    else:
        print("\n❌ Some samples failed verification")

    return all_match


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess LAZ files to NPY format for faster loading"
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
        help="Path to output directory for NPY files"
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

    args = parser.parse_args()

    # Run preprocessing
    stats = preprocess_dataset(
        data_path=args.data_path,
        output_path=args.output_path,
        splits=args.splits,
        num_workers=args.num_workers
    )

    # Optional verification
    if args.verify:
        verify_preprocessing(
            data_path=args.data_path,
            output_path=args.output_path,
            num_samples=5
        )


if __name__ == "__main__":
    main()