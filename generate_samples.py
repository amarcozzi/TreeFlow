"""
treeflow/generate_samples.py

Generate synthetic tree point clouds from a trained Flow Matching model.
Iterates through the test (and optionally validation) set and generates
N samples per source tree with configurable CFG values.

Supports resumable generation - samples are saved as zarr arrays with
metadata in attributes. If interrupted, can resume from where it left off.
"""

import matplotlib

matplotlib.use("Agg")

import torch
import numpy as np
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from types import SimpleNamespace
import matplotlib.pyplot as plt
import zarr

from models import get_model
from dataset import create_datasets
from sample import sample_conditional


def load_experiment_config(experiment_dir: Path) -> dict:
    """Load the config.json from a trained experiment."""
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        return json.load(f)


def load_checkpoint(checkpoint_path: Path, model, device):
    """Load model weights from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        epoch = checkpoint.get("epoch", "unknown")
    else:
        state_dict = checkpoint
        epoch = "unknown"

    # Handle torch.compile() prefix: compiled models save keys with "_orig_mod." prefix
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        print("  Detected compiled model checkpoint, stripping '_orig_mod.' prefix...")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    print(f"  Loaded model from epoch {epoch}")

    return model


def parse_cfg_scale(
    cfg_scale_str: str, num_samples: int, rng: np.random.Generator
) -> list:
    """
    Parse CFG scale argument and generate CFG values for sampling.

    Args:
        cfg_scale_str: Either a single value ("3.0") or a range ("2.0 5.0")
        num_samples: Number of CFG values to generate
        rng: NumPy random generator for reproducibility

    Returns:
        List of CFG values, one per sample
    """
    parts = cfg_scale_str.strip().split()

    if len(parts) == 1:
        # Single value - use for all samples
        cfg_value = float(parts[0])
        return [cfg_value] * num_samples
    elif len(parts) == 2:
        # Range - sample uniformly
        cfg_low, cfg_high = float(parts[0]), float(parts[1])
        return list(rng.uniform(cfg_low, cfg_high, num_samples))
    else:
        raise ValueError(
            f"Invalid cfg_scale format: '{cfg_scale_str}'. "
            "Expected single value (e.g., '3.0') or range (e.g., '2.0 5.0')"
        )


def save_sample_zarr(points: np.ndarray, filepath: Path, metadata: dict):
    """Save point cloud as zarr array with metadata in attributes."""
    z = zarr.open(
        str(filepath.with_suffix(".zarr")),
        mode="w",
        shape=points.shape,
        dtype=np.float32,
    )
    z[:] = points.astype(np.float32)
    for key, value in metadata.items():
        z.attrs[key] = value


def scan_existing_samples(zarr_dir: Path) -> dict:
    """
    Scan zarr directory and return mapping of tree_id to completed sample indices.

    Returns:
        dict: {tree_id: [sample_indices]} where sample_indices are 1-indexed
    """
    existing = {}
    if not zarr_dir.exists():
        return existing

    for zarr_path in zarr_dir.glob("*.zarr"):
        # Parse filename: {tree_id}_{sample_idx}.zarr
        name = zarr_path.stem
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            tree_id, idx_str = parts[0], parts[1]
            try:
                idx = int(idx_str)
                existing.setdefault(tree_id, []).append(idx)
            except ValueError:
                continue  # Skip malformed filenames
    return existing


def save_comparison_image(
    real_points: np.ndarray,
    gen_points: np.ndarray,
    filepath: Path,
    species_name: str,
    type_name: str,
    height_m: float,
    cfg_scale: float,
):
    """
    Save a side-by-side comparison image of real vs generated point clouds.
    Both point clouds should be in normalized coordinates.
    """
    fig = plt.figure(figsize=(12, 6))

    # Normalized coordinates are roughly in [-1, 1] range
    limit = 1.2

    # --- Plot Real ---
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(
        real_points[:, 0],
        real_points[:, 1],
        real_points[:, 2],
        s=1,
        c=real_points[:, 2],
        cmap="viridis",
    )
    ax1.set_title(
        f"REAL | {species_name}\nH={height_m:.2f}m | N={len(real_points)} | {type_name}"
    )
    ax1.set_xlim(-limit, limit)
    ax1.set_ylim(-limit, limit)
    ax1.set_zlim(-limit, limit)

    # --- Plot Generated ---
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(
        gen_points[:, 0],
        gen_points[:, 1],
        gen_points[:, 2],
        s=1,
        c=gen_points[:, 2],
        cmap="viridis",
    )
    ax2.set_title(f"GENERATED | CFG={cfg_scale:.2f}")
    ax2.set_xlim(-limit, limit)
    ax2.set_ylim(-limit, limit)
    ax2.set_zlim(-limit, limit)

    plt.savefig(filepath, dpi=150)
    plt.close()


def generate_samples(args):
    """Main generation function."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load experiment config
    experiments_dir = Path(args.experiments_dir)
    experiment_dir = experiments_dir / args.experiment_name
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment not found: {experiment_dir}")

    config = load_experiment_config(experiment_dir)
    species_list = config["species_list"]
    type_list = config["type_list"]

    print(f"Loaded config from {args.experiment_name}")
    print(f"  Species: {len(species_list)}, Types: {len(type_list)}")

    # Output always under experiment directory
    output_dir = experiment_dir / "samples"

    # Handle resume mode
    if args.resume:
        gen_config_path = output_dir / "generation_config.json"
        if gen_config_path.exists():
            with open(gen_config_path) as f:
                gen_config = json.load(f)

            # Use settings from previous run
            args.checkpoint = gen_config["checkpoint"]
            args.data_path = gen_config["data_path"]
            args.max_points = gen_config["max_points"]
            args.num_samples_per_tree = gen_config["num_samples_per_tree"]
            args.cfg_scale = gen_config["cfg_scale"]
            args.num_ode_steps = gen_config["num_ode_steps"]
            args.solver_method = gen_config["solver_method"]
            args.use_validation = gen_config["use_validation"]
            # Use seed from previous run for consistent CFG sampling
            args.seed = gen_config["seed"]
            # Restore index range if not overridden on command line
            if args.start_idx is None:
                args.start_idx = gen_config.get("start_idx")
            if args.end_idx is None:
                args.end_idx = gen_config.get("end_idx")

            print(f"Resuming generation from {output_dir}")
        else:
            print(f"Resume: no generation_config.json found, using command-line args")

        # Scan for existing samples
        zarr_dir = output_dir / "zarr"
        existing_samples = scan_existing_samples(zarr_dir)
        total_existing = sum(len(v) for v in existing_samples.values())
        print(
            f"  Found {total_existing} existing samples from {len(existing_samples)} trees"
        )
    else:
        existing_samples = {}

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Create datasets with same settings as training
    print(f"\nPreparing datasets from {args.data_path}...")
    _, _, test_ds, ds_species_list, ds_type_list = create_datasets(
        data_path=args.data_path,
        sample_exponent=None,  # No augmentation for generation
        rotation_augment=False,
        shuffle_augment=False,
        cache_train=False,
        cache_val=False,
        cache_test=True,
        max_points=args.max_points,
    )

    # Verify mappings match
    if ds_species_list != species_list:
        raise ValueError(
            f"Species list mismatch between config and dataset!\n"
            f"Config: {species_list[:5]}...\n"
            f"Dataset: {ds_species_list[:5]}..."
        )
    if ds_type_list != type_list:
        raise ValueError(
            f"Type list mismatch between config and dataset!\n"
            f"Config: {type_list}\n"
            f"Dataset: {ds_type_list}"
        )

    # Select which datasets to use
    datasets_to_process = [("test", test_ds)]

    total_trees = sum(len(ds) for _, ds in datasets_to_process)
    print(f"Will process {total_trees} trees from {len(datasets_to_process)} split(s)")

    # Load model
    model_args = SimpleNamespace(
        model_type=config["model_type"],
        model_dim=config["model_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config.get("dropout", 0.1),
        num_freq_bands=config.get("num_freq_bands", 12),
        species_list=species_list,
        type_list=type_list,
    )
    model = get_model(model_args, device)

    # Load checkpoint
    checkpoint_dir = experiment_dir / "checkpoints"
    if args.checkpoint == "best":
        checkpoint_path = checkpoint_dir / "best_model.pt"
    else:
        checkpoint_path = checkpoint_dir / args.checkpoint
    model = load_checkpoint(checkpoint_path, model, device)
    model.eval()
    model = model.to(dtype=torch.bfloat16)

    # Create output directories
    zarr_dir = output_dir / "zarr"
    zarr_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    sample_counter = 0
    skipped_counter = 0

    # Save generation config (do this early so resume can work if interrupted)
    if not args.resume:
        gen_config = {
            "experiment_name": args.experiment_name,
            "checkpoint": args.checkpoint,
            "data_path": args.data_path,
            "max_points": args.max_points,
            "num_samples_per_tree": args.num_samples_per_tree,
            "cfg_scale": args.cfg_scale,
            "num_ode_steps": args.num_ode_steps,
            "solver_method": args.solver_method,
            "use_validation": args.use_validation,
            "seed": args.seed,
            "start_idx": args.start_idx,
            "end_idx": args.end_idx,
            "source_config": config,
        }
        config_path = output_dir / "generation_config.json"
        with open(config_path, "w") as f:
            json.dump(gen_config, f, indent=2)
        print(f"Saved generation config to {config_path}")

    # Compute total trees and index range
    total_trees = sum(len(ds) for _, ds in datasets_to_process)
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else total_trees

    # Generation loop
    print(f"\nGenerating {args.num_samples_per_tree} samples per tree...")
    print(f"CFG scale: {args.cfg_scale}")
    print(f"Solver: {args.solver_method}, Steps: {args.num_ode_steps}")
    print(f"Processing trees [{start_idx}, {end_idx}) of {total_trees} total")

    global_idx = 0  # Track position across all splits

    for split_name, dataset in datasets_to_process:
        split_size = len(dataset)

        # Skip entire split if outside range
        if global_idx + split_size <= start_idx:
            global_idx += split_size
            continue
        if global_idx >= end_idx:
            break

        # Calculate which indices in this split to process
        split_start = max(0, start_idx - global_idx)
        split_end = min(split_size, end_idx - global_idx)

        print(
            f"\nProcessing {split_name} split (indices {split_start}-{split_end} of {split_size})..."
        )

        pbar = tqdm(range(split_start, split_end), desc=split_name)
        for idx in pbar:
            sample = dataset[idx]

            # Extract tree metadata
            source_tree_id = sample["file_id"]
            species_idx = sample["species_idx"].item()
            type_idx = sample["type_idx"].item()
            height_raw = sample["height_raw"].item()
            num_points = sample["num_points"]
            real_points = sample["points"].numpy()  # Already in normalized coordinates

            species_name = species_list[species_idx]
            type_name = type_list[type_idx]

            # Check for existing samples (resume mode)
            completed_indices = set(existing_samples.get(source_tree_id, []))
            all_indices = set(range(1, args.num_samples_per_tree + 1))
            needed_indices = sorted(all_indices - completed_indices)

            if not needed_indices:
                # All samples for this tree already exist
                skipped_counter += args.num_samples_per_tree
                pbar.set_postfix(
                    {"generated": sample_counter, "skipped": skipped_counter}
                )
                continue

            # Generate CFG values for this tree's samples (all of them for consistency)
            # We generate all CFG values to maintain reproducibility with seed
            all_cfg_values = parse_cfg_scale(
                args.cfg_scale, args.num_samples_per_tree, rng
            )

            # Only generate samples for needed indices
            cfg_values_needed = [all_cfg_values[i - 1] for i in needed_indices]

            # Generate samples (batched for this tree)
            start_time = time.time()
            generated_samples = sample_conditional(
                model=model,
                num_points=num_points,
                device=device,
                target_height=height_raw,
                species_idx=species_idx,
                type_idx=type_idx,
                cfg_values=cfg_values_needed,
                num_steps=args.num_ode_steps,
                solver_method=args.solver_method,
                batch_size=args.batch_size,
            )
            generation_time = time.time() - start_time
            time_per_sample = generation_time / len(needed_indices)

            # Save each sample with metadata in zarr attributes
            for sample_points, sample_idx, cfg_val in zip(
                generated_samples, needed_indices, cfg_values_needed
            ):
                sample_counter += 1
                # Name samples based on source tree: {tree_id}_{sample_index}
                sample_id = f"{source_tree_id}_{sample_idx}"

                # Prepare metadata for zarr attributes
                metadata = {
                    "sample_id": sample_id,
                    "sample_file": f"{sample_id}.zarr",
                    "source_tree_id": source_tree_id,
                    "source_split": split_name,
                    "species": species_name,
                    "species_idx": int(species_idx),
                    "scan_type": type_name,
                    "type_idx": int(type_idx),
                    "height_m": float(height_raw),
                    "num_points": int(num_points),
                    "cfg_scale": float(cfg_val),
                    "num_ode_steps": (
                        args.num_ode_steps if args.solver_method != "dopri5" else -1
                    ),
                    "solver_method": args.solver_method,
                    "checkpoint": args.checkpoint,
                    "seed": args.seed,
                    "generation_time_s": float(time_per_sample),
                    "generation_timestamp": datetime.now().isoformat(),
                }

                # Save point cloud as zarr with metadata
                save_sample_zarr(sample_points, zarr_dir / sample_id, metadata)

                # Save comparison image (real vs generated in normalized coordinates)
                save_comparison_image(
                    real_points=real_points,
                    gen_points=sample_points,
                    filepath=images_dir / f"{sample_id}.png",
                    species_name=species_name,
                    type_name=type_name,
                    height_m=height_raw,
                    cfg_scale=cfg_val,
                )

            # Update skipped counter for any pre-existing samples from this tree
            skipped_counter += len(completed_indices)
            pbar.set_postfix({"generated": sample_counter, "skipped": skipped_counter})

        global_idx += split_size

    # Summary
    print(f"\n{'='*50}")
    print("Generation complete!")
    print(f"  New samples generated: {sample_counter}")
    print(f"  Samples skipped (already existed): {skipped_counter}")
    print(f"  Output directory: {output_dir}")
    print(f"  Point clouds (zarr): {zarr_dir}")
    print(f"  Comparison images: {images_dir}")
    print(
        f"\nRun postprocess_samples.py to collect metadata into CSV:"
        f"\n  python postprocess_samples.py {output_dir}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic tree point clouds from a trained model"
    )

    # Experiment/Model Arguments
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of trained experiment (e.g., 'transformer-8-256-4096')",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="Checkpoint to use: 'best' or filename (e.g., 'epoch_100.pt')",
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="experiments",
        help="Base directory containing experiments",
    )

    # Data Arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/preprocessed-full",
        help="Path to preprocessed dataset directory (e.g., data/preprocessed-full or data/preprocessed-4096)",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=None,
        help="Maximum points per sample (should match training)",
    )

    # Generation Arguments
    parser.add_argument(
        "--num_samples_per_tree",
        type=int,
        default=1,
        help="Number of samples to generate per source tree",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=None,
        help="Start index for trees to process (inclusive). For multi-GPU parallelism.",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index for trees to process (exclusive). For multi-GPU parallelism.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=str,
        default="3.0",
        help="CFG scale: single value (e.g., '3.0') or range (e.g., '2.0 5.0')",
    )
    parser.add_argument(
        "--num_ode_steps",
        type=int,
        default=None,
        help="Number of ODE solver steps (for euler/midpoint)",
    )
    parser.add_argument(
        "--solver_method",
        type=str,
        default="dopri5",
        choices=["euler", "midpoint", "dopri5"],
        help="ODE solver method",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Max samples per ODE solve to limit GPU memory usage (0 = all at once)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Data Selection
    parser.add_argument(
        "--use_validation",
        action="store_true",
        help="Include validation set in addition to test set",
    )

    # Resume
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume generation from existing samples in the experiment directory",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.experiment_name:
        parser.error("--experiment_name is required")

    generate_samples(args)


if __name__ == "__main__":
    main()
