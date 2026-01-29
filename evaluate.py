"""
treeflow/evaluate.py
"""

import json
import laspy
import numpy as np
import pandas as pd
import argparse
import zarr
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
from scipy.spatial.distance import cdist
from tqdm import tqdm


# =============================================================================
# Data Classes for Results
# =============================================================================


@dataclass
class CDResult:
    """Chamfer Distance result with uncertainty from downsampling."""

    mean: float
    std: float
    min: float
    max: float
    n_downsamples: int
    was_downsampled: bool
    original_real_count: int
    comparison_count: int
    all_values: list[float]  # All CD values from each downsample

    def __repr__(self):
        if self.std > 0:
            return f"CD={self.mean:.6f} ± {self.std:.6f} (n={self.n_downsamples})"
        return f"CD={self.mean:.6f}"


@dataclass
class PairResult:
    """Evaluation result for a single real-generated pair."""

    source_tree_id: str
    sample_id: str
    species: str
    height_m: float
    scan_type: str
    cd: CDResult


@dataclass
class TreeResult:
    """Aggregated evaluation result for all samples of a single real tree."""

    source_tree_id: str
    species: str
    height_m: float
    scan_type: str
    n_samples: int
    cd_mean: float  # Mean CD across all generated samples
    cd_std: float  # Std of CD across generated samples
    cd_min: float  # Best (lowest) CD among samples
    cd_max: float  # Worst (highest) CD among samples
    mean_downsample_std: float  # Average downsampling uncertainty


# =============================================================================
# Chamfer Distance Implementation
# =============================================================================


def chamfer_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Compute Chamfer Distance between two point clouds.

    Uses squared L2 distance (standard formulation).

    Args:
        p1: (N, 3) point cloud
        p2: (M, 3) point cloud

    Returns:
        Chamfer distance (mean of both directions)
    """
    # Pairwise squared distances
    dist_matrix = cdist(p1, p2, metric="sqeuclidean")

    # Mean min distance from p1 to p2
    min_p1_to_p2 = dist_matrix.min(axis=1).mean()

    # Mean min distance from p2 to p1
    min_p2_to_p1 = dist_matrix.min(axis=0).mean()

    return (min_p1_to_p2 + min_p2_to_p1) / 2


def chamfer_distance_with_downsampling(
    real_points: np.ndarray,
    gen_points: np.ndarray,
    num_downsamples: int = 10,
    seed: int = 42,
) -> CDResult:
    """
    Compute Chamfer Distance with multiple random downsamples of the real cloud.

    Handles the point count mismatch between real clouds (up to millions of points)
    and generated clouds (max 4096 points) by downsampling the real cloud to match.

    Args:
        real_points: (N, 3) real point cloud (potentially millions of points)
        gen_points: (M, 3) generated point cloud (up to 4096 points)
        num_downsamples: Number of random downsamples to average over
        seed: Random seed for reproducibility

    Returns:
        CDResult with mean, std, min, max across downsamples
    """
    rng = np.random.default_rng(seed)
    target_count = len(gen_points)
    original_count = len(real_points)

    # If real cloud is already smaller or equal, no downsampling needed
    if original_count <= target_count:
        cd = chamfer_distance(real_points, gen_points)
        return CDResult(
            mean=cd,
            std=0.0,
            min=cd,
            max=cd,
            n_downsamples=1,
            was_downsampled=False,
            original_real_count=original_count,
            comparison_count=original_count,
            all_values=[cd],
        )

    # Multiple random downsamples
    cd_values = []
    for _ in range(num_downsamples):
        # Random subsample of real cloud
        indices = rng.choice(original_count, size=target_count, replace=False)
        real_downsampled = real_points[indices]

        cd = chamfer_distance(real_downsampled, gen_points)
        cd_values.append(cd)

    return CDResult(
        mean=float(np.mean(cd_values)),
        std=float(np.std(cd_values)),
        min=float(np.min(cd_values)),
        max=float(np.max(cd_values)),
        n_downsamples=num_downsamples,
        was_downsampled=True,
        original_real_count=original_count,
        comparison_count=target_count,
        all_values=[float(v) for v in cd_values],
    )


def chamfer_distance_real_to_real(
    real_a: np.ndarray,
    real_b: np.ndarray,
    target_count: int = 4096,
    num_downsamples: int = 10,
    seed: int = 42,
) -> CDResult:
    """
    Compute CD between two real clouds, downsampling both to target_count.

    This gives us the "natural variation" baseline at the same resolution
    as our generated samples.

    Args:
        real_a: (N, 3) first real point cloud
        real_b: (M, 3) second real point cloud
        target_count: Target point count for both clouds
        num_downsamples: Number of random downsamples to average over
        seed: Random seed for reproducibility

    Returns:
        CDResult with mean, std, min, max across downsamples
    """
    rng = np.random.default_rng(seed)

    # Determine actual comparison count (minimum of target and both cloud sizes)
    actual_count = min(target_count, len(real_a), len(real_b))

    cd_values = []
    for i in range(num_downsamples):
        # Downsample cloud A
        if len(real_a) > actual_count:
            idx_a = rng.choice(len(real_a), size=actual_count, replace=False)
            a_down = real_a[idx_a]
        else:
            a_down = real_a

        # Downsample cloud B
        if len(real_b) > actual_count:
            idx_b = rng.choice(len(real_b), size=actual_count, replace=False)
            b_down = real_b[idx_b]
        else:
            b_down = real_b

        cd_values.append(chamfer_distance(a_down, b_down))

    was_downsampled = len(real_a) > actual_count or len(real_b) > actual_count

    return CDResult(
        mean=float(np.mean(cd_values)),
        std=float(np.std(cd_values)),
        min=float(np.min(cd_values)),
        max=float(np.max(cd_values)),
        n_downsamples=num_downsamples,
        was_downsampled=was_downsampled,
        original_real_count=max(len(real_a), len(real_b)),
        comparison_count=actual_count,
        all_values=[float(v) for v in cd_values],
    )


# =============================================================================
# Evaluation Functions
# =============================================================================


def compute_pairwise_cd(
    pairs: list[dict],
    num_downsamples: int = 10,
    seed: int = 42,
) -> list[PairResult]:
    """
    Compute Chamfer Distance for all real-generated pairs.

    Args:
        pairs: List of evaluation pairs from build_evaluation_pairs()
        num_downsamples: Number of downsamples for uncertainty estimation
        seed: Random seed for reproducibility

    Returns:
        List of PairResult objects
    """
    results = []

    for pair in tqdm(pairs, desc="Computing CD"):
        cd_result = chamfer_distance_with_downsampling(
            real_points=pair["real_points"],
            gen_points=pair["gen_points"],
            num_downsamples=num_downsamples,
            seed=seed,
        )

        results.append(
            PairResult(
                source_tree_id=pair["source_tree_id"],
                sample_id=pair["sample_id"],
                species=pair["species"],
                height_m=pair["height_m"],
                scan_type=pair["scan_type"],
                cd=cd_result,
            )
        )

    return results


def aggregate_by_tree(pair_results: list[PairResult]) -> list[TreeResult]:
    """
    Aggregate pair results by source tree.

    For each tree, computes statistics across all generated samples.

    Args:
        pair_results: List of PairResult objects

    Returns:
        List of TreeResult objects
    """
    # Group by source tree
    grouped = defaultdict(list)
    for pr in pair_results:
        grouped[pr.source_tree_id].append(pr)

    tree_results = []
    for tree_id, pairs in grouped.items():
        # Extract CD means for this tree's samples
        cd_means = [p.cd.mean for p in pairs]
        downsample_stds = [p.cd.std for p in pairs]

        # Use first pair for metadata (all should be same)
        first = pairs[0]

        tree_results.append(
            TreeResult(
                source_tree_id=tree_id,
                species=first.species,
                height_m=first.height_m,
                scan_type=first.scan_type,
                n_samples=len(pairs),
                cd_mean=float(np.mean(cd_means)),
                cd_std=float(np.std(cd_means)),
                cd_min=float(np.min(cd_means)),
                cd_max=float(np.max(cd_means)),
                mean_downsample_std=float(np.mean(downsample_stds)),
            )
        )

    return tree_results


def compute_intraclass_baseline(
    real_point_clouds: dict[str, np.ndarray],
    real_metadata: pd.DataFrame,
    num_pairs_per_species: int = 20,
    target_count: int = 4096,
    num_downsamples: int = 10,
    seed: int = 42,
) -> dict:
    """
    Compute intra-class (same species) real-to-real CD baseline.

    For each species, randomly samples pairs of trees and computes CD.

    Args:
        real_point_clouds: Dict mapping file_id -> point cloud
        real_metadata: DataFrame with species info
        num_pairs_per_species: Number of pairs to sample per species
        target_count: Target point count for downsampling
        num_downsamples: Number of downsamples per pair
        seed: Random seed

    Returns:
        Dict with per-species and global statistics
    """
    rng = np.random.default_rng(seed)

    # Group trees by species
    species_to_trees = defaultdict(list)
    for _, row in real_metadata.iterrows():
        file_id = row["file_id"]
        if file_id in real_point_clouds:
            species_to_trees[row["species"]].append(file_id)

    all_cd_values = []
    per_species_results = {}

    for species, tree_ids in tqdm(
        species_to_trees.items(), desc="Intra-class baseline"
    ):
        if len(tree_ids) < 2:
            continue

        species_cd_values = []

        # Sample pairs
        n_pairs = min(num_pairs_per_species, len(tree_ids) * (len(tree_ids) - 1) // 2)
        for _ in range(n_pairs):
            # Random pair of different trees
            idx_a, idx_b = rng.choice(len(tree_ids), size=2, replace=False)
            tree_a, tree_b = tree_ids[idx_a], tree_ids[idx_b]

            cd_result = chamfer_distance_real_to_real(
                real_point_clouds[tree_a],
                real_point_clouds[tree_b],
                target_count=target_count,
                num_downsamples=num_downsamples,
                seed=seed + hash((tree_a, tree_b)) % 10000,
            )

            species_cd_values.append(cd_result.mean)
            all_cd_values.append(cd_result.mean)

        if species_cd_values:
            per_species_results[species] = {
                "mean": float(np.mean(species_cd_values)),
                "std": float(np.std(species_cd_values)),
                "n_pairs": len(species_cd_values),
            }

    return {
        "global": {
            "mean": float(np.mean(all_cd_values)) if all_cd_values else 0.0,
            "std": float(np.std(all_cd_values)) if all_cd_values else 0.0,
            "median": float(np.median(all_cd_values)) if all_cd_values else 0.0,
            "n_pairs": len(all_cd_values),
        },
        "per_species": per_species_results,
    }


def compute_interclass_baseline(
    real_point_clouds: dict[str, np.ndarray],
    real_metadata: pd.DataFrame,
    num_pairs: int = 200,
    target_count: int = 4096,
    num_downsamples: int = 10,
    seed: int = 42,
) -> dict:
    """
    Compute inter-class (different species) real-to-real CD baseline.

    Randomly samples pairs of trees from different species.

    Args:
        real_point_clouds: Dict mapping file_id -> point cloud
        real_metadata: DataFrame with species info
        num_pairs: Total number of pairs to sample
        target_count: Target point count for downsampling
        num_downsamples: Number of downsamples per pair
        seed: Random seed

    Returns:
        Dict with global statistics
    """
    rng = np.random.default_rng(seed)

    # Build list of (file_id, species) tuples
    trees_with_species = [
        (row["file_id"], row["species"])
        for _, row in real_metadata.iterrows()
        if row["file_id"] in real_point_clouds
    ]

    if len(trees_with_species) < 2:
        return {"mean": 0.0, "std": 0.0, "n_pairs": 0}

    cd_values = []

    for _ in tqdm(range(num_pairs), desc="Inter-class baseline"):
        # Sample two different trees
        attempts = 0
        while attempts < 100:
            idx_a, idx_b = rng.choice(len(trees_with_species), size=2, replace=False)
            tree_a, species_a = trees_with_species[idx_a]
            tree_b, species_b = trees_with_species[idx_b]

            # Require different species
            if species_a != species_b:
                break
            attempts += 1

        if attempts >= 100:
            continue  # Couldn't find different species pair

        cd_result = chamfer_distance_real_to_real(
            real_point_clouds[tree_a],
            real_point_clouds[tree_b],
            target_count=target_count,
            num_downsamples=num_downsamples,
            seed=seed + hash((tree_a, tree_b)) % 10000,
        )

        cd_values.append(cd_result.mean)

    return {
        "mean": float(np.mean(cd_values)) if cd_values else 0.0,
        "std": float(np.std(cd_values)) if cd_values else 0.0,
        "median": float(np.median(cd_values)) if cd_values else 0.0,
        "n_pairs": len(cd_values),
    }


# =============================================================================
# Reporting Functions
# =============================================================================


def print_cd_results(
    pair_results: list[PairResult],
    tree_results: list[TreeResult],
    intraclass_baseline: dict,
    interclass_baseline: dict,
):
    """Print formatted Chamfer Distance results."""

    print("\n" + "=" * 70)
    print("CHAMFER DISTANCE RESULTS")
    print("=" * 70)

    # Global statistics across all pairs
    all_cd_means = [p.cd.mean for p in pair_results]
    all_downsample_stds = [p.cd.std for p in pair_results]

    print(f"\nPer-pair CD (n={len(pair_results)} pairs):")
    print(f"  Mean:   {np.mean(all_cd_means):.6f}")
    print(f"  Std:    {np.std(all_cd_means):.6f}")
    print(f"  Median: {np.median(all_cd_means):.6f}")
    print(f"  Min:    {np.min(all_cd_means):.6f}")
    print(f"  Max:    {np.max(all_cd_means):.6f}")

    # Downsampling variance
    n_downsampled = sum(1 for p in pair_results if p.cd.was_downsampled)
    print(f"\nDownsampling variance:")
    print(f"  Pairs requiring downsampling: {n_downsampled}/{len(pair_results)}")
    print(f"  Mean std within pairs: {np.mean(all_downsample_stds):.6f}")
    print(f"  Max std within pairs:  {np.max(all_downsample_stds):.6f}")

    # Per-tree statistics
    tree_cd_means = [t.cd_mean for t in tree_results]
    tree_cd_stds = [t.cd_std for t in tree_results]

    print(f"\nPer-tree CD (n={len(tree_results)} trees):")
    print(f"  Mean of tree means: {np.mean(tree_cd_means):.6f}")
    print(f"  Std of tree means:  {np.std(tree_cd_means):.6f}")
    print(f"  Mean within-tree std: {np.mean(tree_cd_stds):.6f}")
    print(f"    (variation across generated samples for same tree)")

    # MMD (Minimum Matching Distance)
    tree_cd_mins = [t.cd_min for t in tree_results]
    print(f"\nMinimum Matching Distance (best sample per tree):")
    print(f"  Mean MMD: {np.mean(tree_cd_mins):.6f}")
    print(f"  Std MMD:  {np.std(tree_cd_mins):.6f}")

    # Baselines
    print(f"\nBaselines (real-to-real CD at same resolution):")
    print(
        f"  Intra-class (same species): {intraclass_baseline['global']['mean']:.6f} "
        f"± {intraclass_baseline['global']['std']:.6f} "
        f"(n={intraclass_baseline['global']['n_pairs']} pairs)"
    )
    print(
        f"  Inter-class (diff species): {interclass_baseline['mean']:.6f} "
        f"± {interclass_baseline['std']:.6f} "
        f"(n={interclass_baseline['n_pairs']} pairs)"
    )

    # Interpretation
    gen_to_real_mean = np.mean(all_cd_means)
    intra_mean = intraclass_baseline["global"]["mean"]
    inter_mean = interclass_baseline["mean"]

    print(f"\nInterpretation:")
    if intra_mean > 0:
        ratio = gen_to_real_mean / intra_mean
        print(f"  Gen-to-Real / Intra-class ratio: {ratio:.2f}")
        if ratio < 0.5:
            print(
                "  ⚠️  Generated samples are suspiciously close to source (possible memorization)"
            )
        elif ratio < 1.5:
            print(
                "  ✓  Generated samples show similar variation to real intra-class pairs"
            )
        else:
            print("  ⚠️  Generated samples differ more than expected from source trees")

    if inter_mean > 0 and intra_mean > 0:
        if gen_to_real_mean < inter_mean:
            print("  ✓  Gen-to-Real CD is lower than inter-class (conditioning works)")
        else:
            print(
                "  ⚠️  Gen-to-Real CD is higher than inter-class (conditioning may be weak)"
            )

    print("=" * 70 + "\n")


def save_results(
    pair_results: list[PairResult],
    tree_results: list[TreeResult],
    intraclass_baseline: dict,
    interclass_baseline: dict,
    output_dir: Path,
):
    """Save evaluation results to JSON and CSV files."""

    output_dir = Path(output_dir)

    # Compute summary statistics
    all_cd_means = [p.cd.mean for p in pair_results]
    all_downsample_stds = [p.cd.std for p in pair_results]
    tree_cd_means = [t.cd_mean for t in tree_results]
    tree_cd_mins = [t.cd_min for t in tree_results]

    summary = {
        "pair_statistics": {
            "n_pairs": len(pair_results),
            "cd_mean": float(np.mean(all_cd_means)),
            "cd_std": float(np.std(all_cd_means)),
            "cd_median": float(np.median(all_cd_means)),
            "cd_min": float(np.min(all_cd_means)),
            "cd_max": float(np.max(all_cd_means)),
        },
        "downsampling_variance": {
            "mean_std_within_pairs": float(np.mean(all_downsample_stds)),
            "max_std_within_pairs": float(np.max(all_downsample_stds)),
            "n_pairs_downsampled": sum(1 for p in pair_results if p.cd.was_downsampled),
        },
        "tree_statistics": {
            "n_trees": len(tree_results),
            "mean_of_tree_means": float(np.mean(tree_cd_means)),
            "std_of_tree_means": float(np.std(tree_cd_means)),
            "mmd_mean": float(np.mean(tree_cd_mins)),
            "mmd_std": float(np.std(tree_cd_mins)),
        },
        "baselines": {
            "intraclass": intraclass_baseline["global"],
            "interclass": interclass_baseline,
        },
    }

    # Save summary JSON
    summary_path = output_dir / "cd_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    # Save per-pair results as CSV
    pair_rows = []
    for p in pair_results:
        pair_rows.append(
            {
                "source_tree_id": p.source_tree_id,
                "sample_id": p.sample_id,
                "species": p.species,
                "height_m": p.height_m,
                "scan_type": p.scan_type,
                "cd_mean": p.cd.mean,
                "cd_std": p.cd.std,
                "cd_min": p.cd.min,
                "cd_max": p.cd.max,
                "was_downsampled": p.cd.was_downsampled,
                "original_real_count": p.cd.original_real_count,
                "comparison_count": p.cd.comparison_count,
                "cd_all_values": p.cd.all_values,  # List of all CD values from downsamples
            }
        )
    pair_df = pd.DataFrame(pair_rows)
    pair_csv_path = output_dir / "cd_per_pair.csv"
    pair_df.to_csv(pair_csv_path, index=False)
    print(f"Saved per-pair results to {pair_csv_path}")

    # Save per-tree results as CSV
    tree_rows = [asdict(t) for t in tree_results]
    tree_df = pd.DataFrame(tree_rows)
    tree_csv_path = output_dir / "cd_per_tree.csv"
    tree_df.to_csv(tree_csv_path, index=False)
    print(f"Saved per-tree results to {tree_csv_path}")

    # Save intra-class per-species results
    if intraclass_baseline.get("per_species"):
        species_rows = [
            {"species": sp, **stats}
            for sp, stats in intraclass_baseline["per_species"].items()
        ]
        species_df = pd.DataFrame(species_rows)
        species_csv_path = output_dir / "cd_intraclass_per_species.csv"
        species_df.to_csv(species_csv_path, index=False)
        print(f"Saved per-species baseline to {species_csv_path}")


# =============================================================================
# Data Loading Functions
# =============================================================================


def normalize_tree_id(tree_id) -> str:
    """
    Normalize tree ID to 5-digit zero-padded string format.

    Generated samples store source_tree_id as int (e.g., 3607)
    Real tree metadata stores file_id as string (e.g., "03607")
    """
    return str(tree_id).zfill(5)


def load_generated_samples(samples_dir: Path) -> tuple[pd.DataFrame, dict]:
    """
    Load generated samples metadata and point clouds.

    Args:
        samples_dir: Path to generated samples directory (e.g., generated_samples/experiment_timestamp/)

    Returns:
        metadata_df: DataFrame with sample metadata
        point_clouds: Dict mapping sample_id -> np.ndarray of shape (N, 3) in meters
    """
    samples_dir = Path(samples_dir)

    # Load metadata CSV
    metadata_path = samples_dir / "samples_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    metadata_df = pd.read_csv(metadata_path)

    # Normalize source_tree_id to zero-padded string format
    metadata_df["source_tree_id"] = metadata_df["source_tree_id"].apply(
        normalize_tree_id
    )

    print(f"Loaded metadata for {len(metadata_df)} generated samples")

    # Determine point cloud format and directory
    laz_dir = samples_dir / "laz"
    zarr_dir = samples_dir / "zarr"

    if laz_dir.exists():
        pc_dir = laz_dir
        pc_format = "laz"
    elif zarr_dir.exists():
        pc_dir = zarr_dir
        pc_format = "zarr"
    else:
        raise FileNotFoundError(f"No point cloud directory found in {samples_dir}")

    print(f"Loading point clouds from {pc_dir} (format: {pc_format})")

    # Load point clouds
    point_clouds = {}
    missing_count = 0

    for _, row in tqdm(
        metadata_df.iterrows(), total=len(metadata_df), desc="Loading generated"
    ):
        sample_id = row["sample_id"]

        if pc_format == "laz":
            pc_path = pc_dir / f"{sample_id}.laz"
            if pc_path.exists():
                las = laspy.read(str(pc_path))
                points = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float32)
                point_clouds[sample_id] = points
            else:
                missing_count += 1
        else:  # zarr
            pc_path = pc_dir / f"{sample_id}.zarr"
            if pc_path.exists():
                point_clouds[sample_id] = zarr.load(pc_path).astype(np.float32)
            else:
                missing_count += 1

    if missing_count > 0:
        print(f"  Warning: {missing_count} point cloud files not found")
    return metadata_df, point_clouds


def load_real_samples(
    data_path: Path,
    csv_path: Path,
    source_tree_ids: list[str],
    preprocessed_version: str = "raw",
) -> tuple[pd.DataFrame, dict]:
    """
    Load real tree point clouds for the specified source tree IDs.

    Point clouds are kept in normalized form (as stored in .zarr files).

    Normalization: points_norm = (points_centered / height) * 2.0

    Args:
        data_path: Path to FOR-species20K directory
        csv_path: Path to tree_metadata_dev.csv
        source_tree_ids: List of tree IDs to load (from generated samples metadata)
        preprocessed_version: Preprocessing version subdirectory (default: use root zarr dir)

    Returns:
        metadata_df: DataFrame with real tree metadata (filtered to requested IDs)
        point_clouds: Dict mapping file_id -> np.ndarray of shape (N, 3) normalized
    """
    data_path = Path(data_path)
    csv_path = Path(csv_path)

    # Load full metadata
    full_df = pd.read_csv(csv_path)

    # Extract file_id from filename (e.g., "/train/00070.las" -> "00070")
    full_df["file_id"] = full_df["filename"].apply(lambda x: Path(x).stem)

    # Filter to requested IDs
    source_tree_ids_set = set(source_tree_ids)
    metadata_df = full_df[full_df["file_id"].isin(source_tree_ids_set)].copy()

    print(
        f"Found {len(metadata_df)}/{len(source_tree_ids_set)} requested real trees in metadata"
    )

    # Build file paths
    if preprocessed_version:
        zarr_dir = data_path / "zarr" / preprocessed_version / "dev"
    else:
        zarr_dir = data_path / "zarr" / "dev"
    metadata_df["file_path"] = metadata_df["file_id"].apply(
        lambda x: zarr_dir / f"{x}.zarr"
    )

    # Load point clouds (keep normalized)
    point_clouds = {}
    missing_count = 0

    for _, row in tqdm(
        metadata_df.iterrows(), total=len(metadata_df), desc="Loading real"
    ):
        file_id = row["file_id"]
        file_path = row["file_path"]

        if file_path.exists():
            # Load normalized points directly from .zarr
            # These are already normalized: points_norm = (points_centered / height) * 2.0
            points_norm = zarr.load(file_path).astype(np.float32)
            point_clouds[file_id] = points_norm
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"  Warning: {missing_count} real point cloud files not found")

    print(f"  Loaded {len(point_clouds)} real point clouds (normalized)")
    return metadata_df, point_clouds


def build_evaluation_pairs(
    gen_metadata: pd.DataFrame,
    gen_point_clouds: dict,
    real_metadata: pd.DataFrame,
    real_point_clouds: dict,
) -> list[dict]:
    """
    Build list of (real, generated) pairs for evaluation.

    Each generated sample is paired with its source real tree.
    Both real and generated point clouds are already in normalized coordinates
    (from preprocessing and model output respectively).

    Normalization scheme: points_norm = (points_centered / height) * 2.0

    Returns:
        List of dicts with keys:
            - source_tree_id: ID of the real tree
            - sample_id: ID of the generated sample
            - real_points: np.ndarray (N, 3) normalized
            - gen_points: np.ndarray (M, 3) normalized
            - species: species name
            - height_m: tree height in meters
            - scan_type: scanner type (TLS, MLS, ULS)
    """
    pairs = []
    skipped = 0

    for _, row in gen_metadata.iterrows():
        source_tree_id = row["source_tree_id"]
        sample_id = row["sample_id"]
        height_m = row["height_m"]

        # Check if we have both point clouds
        if sample_id not in gen_point_clouds:
            skipped += 1
            continue
        if source_tree_id not in real_point_clouds:
            skipped += 1
            continue

        # Get generated points (already in normalized coordinates from model output)
        gen_points_norm = gen_point_clouds[sample_id]

        pairs.append(
            {
                "source_tree_id": source_tree_id,
                "sample_id": sample_id,
                "real_points": real_point_clouds[source_tree_id],  # Already normalized
                "gen_points": gen_points_norm,  # Already normalized from model
                "species": row["species"],
                "height_m": height_m,
                "scan_type": row["scan_type"],
            }
        )

    if skipped > 0:
        print(f"  Skipped {skipped} pairs due to missing point clouds")

    print(f"Built {len(pairs)} evaluation pairs (all point clouds normalized)")
    return pairs


def group_pairs_by_tree(pairs: list[dict]) -> dict[str, list[dict]]:
    """
    Group evaluation pairs by source tree ID.

    Returns:
        Dict mapping source_tree_id -> list of pairs for that tree
    """
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair["source_tree_id"]].append(pair)
    return dict(grouped)


def print_data_summary(pairs: list[dict], grouped: dict):
    """Print summary statistics about the loaded data."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    print(f"Total evaluation pairs: {len(pairs)}")
    print(f"Unique source trees: {len(grouped)}")

    # Samples per tree
    samples_per_tree = [len(v) for v in grouped.values()]
    print(
        f"Samples per tree: min={min(samples_per_tree)}, max={max(samples_per_tree)}, "
        f"mean={np.mean(samples_per_tree):.1f}"
    )

    # Species distribution
    species_counts = defaultdict(int)
    for pair in pairs:
        species_counts[pair["species"]] += 1

    print(f"\nSpecies distribution ({len(species_counts)} species):")
    for species, count in sorted(species_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {species}: {count} samples")
    if len(species_counts) > 10:
        print(f"  ... and {len(species_counts) - 10} more species")

    # Height distribution
    heights = [pair["height_m"] for pair in pairs]
    print(f"\nHeight distribution:")
    print(
        f"  min={min(heights):.1f}m, max={max(heights):.1f}m, "
        f"mean={np.mean(heights):.1f}m, std={np.std(heights):.1f}m"
    )

    # Point count distribution
    real_point_counts = [len(pair["real_points"]) for pair in pairs]
    gen_point_counts = [len(pair["gen_points"]) for pair in pairs]
    print(
        f"\nPoint counts (real): min={min(real_point_counts)}, max={max(real_point_counts)}, "
        f"mean={np.mean(real_point_counts):.0f}"
    )
    print(
        f"Point counts (gen):  min={min(gen_point_counts)}, max={max(gen_point_counts)}, "
        f"mean={np.mean(gen_point_counts):.0f}"
    )

    print("=" * 60 + "\n")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated tree point clouds against real trees"
    )

    # Input paths
    parser.add_argument(
        "--samples_dir",
        type=str,
        default="generated_samples/transformer-8-256-4096_20260105_172326",
        help="Path to generated samples directory",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="FOR-species20K",
        help="Path to FOR-species20K directory",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="FOR-species20K/tree_metadata_dev.csv",
        help="Path to real tree metadata CSV",
    )
    parser.add_argument(
        "--preprocessed_version",
        type=str,
        default=None,
        help="Preprocessing version subdirectory (default: use root zarr dir)",
    )

    # Evaluation parameters
    parser.add_argument(
        "--num_downsamples",
        type=int,
        default=20,
        help="Number of random downsamples for uncertainty estimation",
    )
    parser.add_argument(
        "--baseline_pairs_per_species",
        type=int,
        default=100,
        help="Number of pairs to sample per species for intra-class baseline",
    )
    parser.add_argument(
        "--interclass_pairs",
        type=int,
        default=100,
        help="Number of pairs to sample for inter-class baseline",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: samples_dir/evaluation/)",
    )

    args = parser.parse_args()

    # Resolve paths
    samples_dir = Path(args.samples_dir)
    data_path = Path(args.data_path)
    csv_path = Path(args.csv_path)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = samples_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Samples directory: {samples_dir}")
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Downsamples per pair: {args.num_downsamples}")
    print()

    # ==========================================================================
    # 1. Load Data
    # ==========================================================================

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    # Load generated samples
    print("\nLoading generated samples...")
    gen_metadata, gen_point_clouds = load_generated_samples(samples_dir)

    # Get unique source tree IDs
    source_tree_ids = gen_metadata["source_tree_id"].unique().tolist()
    print(f"\nFound {len(source_tree_ids)} unique source trees")

    # Load corresponding real samples
    print("\nLoading real samples...")
    real_metadata, real_point_clouds = load_real_samples(
        data_path=data_path,
        csv_path=csv_path,
        source_tree_ids=source_tree_ids,
        preprocessed_version=args.preprocessed_version,
    )

    # Build evaluation pairs
    print("\nBuilding evaluation pairs...")
    pairs = build_evaluation_pairs(
        gen_metadata,
        gen_point_clouds,
        real_metadata,
        real_point_clouds,
    )

    # Group by tree and print summary
    grouped = group_pairs_by_tree(pairs)
    print_data_summary(pairs, grouped)

    # ==========================================================================
    # 2. Compute Baselines
    # ==========================================================================

    print("=" * 60)
    print("COMPUTING BASELINES")
    print("=" * 60)

    print("\nComputing intra-class (same species) baseline...")
    intraclass_baseline = compute_intraclass_baseline(
        real_point_clouds=real_point_clouds,
        real_metadata=real_metadata,
        num_pairs_per_species=args.baseline_pairs_per_species,
        target_count=4096,
        num_downsamples=args.num_downsamples,
        seed=args.seed,
    )

    print("\nComputing inter-class (different species) baseline...")
    interclass_baseline = compute_interclass_baseline(
        real_point_clouds=real_point_clouds,
        real_metadata=real_metadata,
        num_pairs=args.interclass_pairs,
        target_count=4096,
        num_downsamples=args.num_downsamples,
        seed=args.seed + 1000,
    )

    # ==========================================================================
    # 3. Compute Pairwise Chamfer Distance
    # ==========================================================================

    print("\n" + "=" * 60)
    print("COMPUTING CHAMFER DISTANCE")
    print("=" * 60)

    print(f"\nComputing CD for {len(pairs)} real-generated pairs...")
    pair_results = compute_pairwise_cd(
        pairs=pairs,
        num_downsamples=args.num_downsamples,
        seed=args.seed,
    )

    # Aggregate by tree
    print("\nAggregating results by tree...")
    tree_results = aggregate_by_tree(pair_results)

    # ==========================================================================
    # 4. Report and Save Results
    # ==========================================================================

    print_cd_results(
        pair_results=pair_results,
        tree_results=tree_results,
        intraclass_baseline=intraclass_baseline,
        interclass_baseline=interclass_baseline,
    )

    save_results(
        pair_results=pair_results,
        tree_results=tree_results,
        intraclass_baseline=intraclass_baseline,
        interclass_baseline=interclass_baseline,
        output_dir=output_dir,
    )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
