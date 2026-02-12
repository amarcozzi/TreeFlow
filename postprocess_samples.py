"""
postprocess_samples.py

Collect metadata from zarr sample files into a single CSV file.
Run this after generation completes (or is interrupted) to create
the samples_metadata.csv file inside the experiment's samples/ directory.

Usage:
    python postprocess_samples.py --experiment_name transformer-8-256-4096
    python postprocess_samples.py --experiment_name transformer-8-512-16384 --experiments_dir experiments
"""

import argparse
from pathlib import Path

import pandas as pd
import zarr
from tqdm import tqdm


def collect_metadata(samples_dir: Path, show_progress: bool = True) -> pd.DataFrame:
    """
    Read all zarr files in the samples directory and collect attributes into DataFrame.

    Args:
        samples_dir: Path to experiment's samples/ directory containing zarr/ subdirectory
        show_progress: Whether to show progress bar

    Returns:
        DataFrame with one row per sample containing all metadata
    """
    zarr_dir = samples_dir / "zarr"
    if not zarr_dir.exists():
        raise FileNotFoundError(f"zarr directory not found: {zarr_dir}")

    zarr_files = sorted(zarr_dir.glob("*.zarr"))
    if not zarr_files:
        print(f"Warning: No zarr files found in {zarr_dir}")
        return pd.DataFrame()

    rows = []
    iterator = tqdm(zarr_files, desc="Reading zarr files") if show_progress else zarr_files

    for zarr_path in iterator:
        try:
            z = zarr.open(str(zarr_path), mode="r")
            metadata = dict(z.attrs)
            rows.append(metadata)
        except Exception as e:
            print(f"Warning: Failed to read {zarr_path}: {e}")
            continue

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Collect metadata from zarr samples into CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single experiment
  python postprocess_samples.py --experiment_name transformer-8-256-4096

  # Custom experiments directory
  python postprocess_samples.py --experiment_name transformer-8-512-16384 --experiments_dir experiments

  # Specify custom output CSV path
  python postprocess_samples.py --experiment_name transformer-8-256-4096 --output results/metadata.csv
        """,
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the experiment (e.g., 'transformer-8-256-4096')",
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="experiments",
        help="Base directory containing experiments (default: experiments)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Default: {experiments_dir}/{experiment_name}/samples/samples_metadata.csv",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bar",
    )

    args = parser.parse_args()

    # Resolve experiment samples directory
    experiment_dir = Path(args.experiments_dir) / args.experiment_name
    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        return 1

    samples_dir = experiment_dir / "samples"
    if not samples_dir.exists():
        print(f"Error: Samples directory not found: {samples_dir}")
        print("Have you run generate_samples.py for this experiment?")
        return 1

    print(f"Processing experiment: {args.experiment_name}")
    print(f"  Samples directory: {samples_dir}")

    try:
        df = collect_metadata(samples_dir, show_progress=not args.quiet)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    if df.empty:
        print("No samples found")
        return 1

    print(f"  Found {len(df)} samples")

    # Determine output path
    if args.output:
        csv_path = Path(args.output)
    else:
        csv_path = samples_dir / "samples_metadata.csv"

    # Ensure parent directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} sample records to {csv_path}")

    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Total samples: {len(df)}")
    if "species" in df.columns:
        print(f"  Unique species: {df['species'].nunique()}")
    if "source_tree_id" in df.columns:
        print(f"  Unique source trees: {df['source_tree_id'].nunique()}")
    if "cfg_scale" in df.columns:
        print(
            f"  CFG scale range: {df['cfg_scale'].min():.2f} - "
            f"{df['cfg_scale'].max():.2f}"
        )

    return 0


if __name__ == "__main__":
    exit(main())
