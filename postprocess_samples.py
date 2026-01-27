"""
postprocess_samples.py

Collect metadata from zarr sample files into a single CSV file.
Run this after generation completes (or is interrupted) to create
the samples_metadata.csv file.

Usage:
    python postprocess_samples.py generated_samples/experiment_20240126_143000
    python postprocess_samples.py generated_samples/experiment_* --output combined.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import zarr
from tqdm import tqdm


def collect_metadata(output_dir: Path, show_progress: bool = True) -> pd.DataFrame:
    """
    Read all zarr files in the output directory and collect attributes into DataFrame.

    Args:
        output_dir: Path to generation output directory containing zarr/ subdirectory
        show_progress: Whether to show progress bar

    Returns:
        DataFrame with one row per sample containing all metadata
    """
    zarr_dir = output_dir / "zarr"
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
  # Process single output directory
  python postprocess_samples.py generated_samples/experiment_20240126_143000

  # Process multiple directories (glob pattern)
  python postprocess_samples.py generated_samples/experiment_*

  # Specify custom output path
  python postprocess_samples.py generated_samples/exp1 --output results/metadata.csv
        """,
    )
    parser.add_argument(
        "output_dirs",
        type=str,
        nargs="+",
        help="Path(s) to generation output directory/directories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Default: {output_dir}/samples_metadata.csv for single dir, "
        "or 'combined_metadata.csv' for multiple dirs",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bar",
    )

    args = parser.parse_args()

    # Expand glob patterns and collect all directories
    all_dirs = []
    for pattern in args.output_dirs:
        path = Path(pattern)
        if path.exists():
            all_dirs.append(path)
        else:
            # Try as glob pattern
            matches = list(Path(".").glob(pattern))
            if matches:
                all_dirs.extend(matches)
            else:
                print(f"Warning: No matches found for '{pattern}'")

    if not all_dirs:
        print("Error: No valid directories found")
        return 1

    # Process directories
    all_dfs = []
    for output_dir in all_dirs:
        if not output_dir.is_dir():
            print(f"Skipping non-directory: {output_dir}")
            continue

        print(f"Processing: {output_dir}")
        try:
            df = collect_metadata(output_dir, show_progress=not args.quiet)
            if not df.empty:
                df["output_dir"] = str(output_dir)
                all_dfs.append(df)
                print(f"  Found {len(df)} samples")
        except FileNotFoundError as e:
            print(f"  {e}")
            continue

    if not all_dfs:
        print("No samples found in any directory")
        return 1

    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Determine output path
    if args.output:
        csv_path = Path(args.output)
    elif len(all_dirs) == 1:
        csv_path = all_dirs[0] / "samples_metadata.csv"
    else:
        csv_path = Path("combined_metadata.csv")

    # Ensure parent directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV
    combined_df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(combined_df)} sample records to {csv_path}")

    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Total samples: {len(combined_df)}")
    if "species" in combined_df.columns:
        print(f"  Unique species: {combined_df['species'].nunique()}")
    if "source_tree_id" in combined_df.columns:
        print(f"  Unique source trees: {combined_df['source_tree_id'].nunique()}")
    if "cfg_scale" in combined_df.columns:
        print(
            f"  CFG scale range: {combined_df['cfg_scale'].min():.2f} - "
            f"{combined_df['cfg_scale'].max():.2f}"
        )

    return 0


if __name__ == "__main__":
    exit(main())
