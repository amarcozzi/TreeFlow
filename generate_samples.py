"""
treeflow/generate_samples.py

Generate synthetic tree point clouds from a trained Flow Matching model.
Iterates through the test (and optionally validation) set and generates
N samples per source tree with configurable CFG values.
"""

import torch
import numpy as np
import pandas as pd
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from types import SimpleNamespace

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


def save_point_cloud(points: np.ndarray, filepath: Path, fmt: str):
    """Save point cloud in specified format."""
    if fmt == "npy":
        np.save(filepath.with_suffix(".npy"), points.astype(np.float32))
    elif fmt == "laz":
        import laspy

        las = laspy.create(point_format=0)
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]
        las.write(str(filepath.with_suffix(".laz")))


def generate_samples(args):
    """Main generation function."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

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

    # Create datasets with same settings as training
    # We need to use the same split seed (42) to ensure identical train/val/test splits
    print(f"\nPreparing datasets from {args.csv_path}...")
    _, val_ds, test_ds, ds_species_list, ds_type_list = create_datasets(
        data_path=args.data_path,
        csv_path=args.csv_path,
        preprocessed_version=args.preprocessed_version,
        sample_exponent=None,  # No augmentation for generation
        rotation_augment=False,
        shuffle_augment=False,
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
    if args.use_validation:
        datasets_to_process.append(("validation", val_ds))

    total_trees = sum(len(ds) for _, ds in datasets_to_process)
    print(f"Will process {total_trees} trees from {len(datasets_to_process)} split(s)")

    # Load model
    model_args = SimpleNamespace(
        model_type=config["model_type"],
        model_dim=config["model_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config.get("dropout", 0.1),
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

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.experiment_name}_{timestamp}"
    points_dir = output_dir / args.output_format
    points_dir.mkdir(parents=True, exist_ok=True)

    # Prepare metadata collection
    metadata_rows = []
    sample_counter = 0

    # Generation loop
    print(f"\nGenerating {args.num_samples_per_tree} samples per tree...")
    print(f"CFG scale: {args.cfg_scale}")
    print(f"Solver: {args.solver_method}, Steps: {args.num_ode_steps}")

    for split_name, dataset in datasets_to_process:
        print(f"\nProcessing {split_name} split ({len(dataset)} trees)...")

        pbar = tqdm(range(len(dataset)), desc=split_name)
        for idx in pbar:
            sample = dataset[idx]

            # Extract tree metadata
            source_tree_id = sample["file_id"]
            species_idx = sample["species_idx"].item()
            type_idx = sample["type_idx"].item()
            height_raw = sample["height_raw"].item()
            num_points = sample["num_points"]

            species_name = species_list[species_idx]
            type_name = type_list[type_idx]

            # Generate CFG values for this tree's samples
            cfg_values = parse_cfg_scale(args.cfg_scale, args.num_samples_per_tree, rng)

            # Generate samples (batched for this tree)
            start_time = time.time()
            generated_samples = sample_conditional(
                model=model,
                num_points=num_points,
                device=device,
                target_height=height_raw,
                species_idx=species_idx,
                type_idx=type_idx,
                cfg_values=cfg_values,
                num_steps=args.num_ode_steps,
                solver_method=args.solver_method,
            )
            generation_time = time.time() - start_time
            time_per_sample = generation_time / args.num_samples_per_tree

            # Save each sample and record metadata
            for i, (sample_points, cfg_val) in enumerate(
                zip(generated_samples, cfg_values)
            ):
                sample_counter += 1
                # Name samples based on source tree: {tree_id}_{sample_index}
                sample_id = f"{source_tree_id}_{i + 1}"
                sample_filename = f"{sample_id}.{args.output_format}"

                # Save point cloud
                save_point_cloud(
                    sample_points, points_dir / sample_id, args.output_format
                )

                # Record metadata
                metadata_rows.append(
                    {
                        "sample_id": sample_id,
                        "sample_file": sample_filename,
                        "source_tree_id": source_tree_id,
                        "source_split": split_name,
                        "species": species_name,
                        "species_idx": species_idx,
                        "scan_type": type_name,
                        "type_idx": type_idx,
                        "height_m": height_raw,
                        "num_points": num_points,
                        "cfg_scale": cfg_val,
                        "num_ode_steps": (
                            args.num_ode_steps if args.solver_method != "dopri5" else -1
                        ),
                        "solver_method": args.solver_method,
                        "checkpoint": args.checkpoint,
                        "seed": args.seed,
                        "generation_time_s": time_per_sample,
                        "generation_timestamp": datetime.now().isoformat(),
                    }
                )

            pbar.set_postfix(
                {"samples": sample_counter, "time": f"{generation_time:.2f}s"}
            )

    # Save metadata CSV
    metadata_df = pd.DataFrame(metadata_rows)
    csv_path = output_dir / "samples_metadata.csv"
    metadata_df.to_csv(csv_path, index=False)
    print(f"\nSaved metadata to {csv_path}")

    # Save generation config
    gen_config = {
        "experiment_name": args.experiment_name,
        "checkpoint": args.checkpoint,
        "data_path": args.data_path,
        "csv_path": args.csv_path,
        "preprocessed_version": args.preprocessed_version,
        "max_points": args.max_points,
        "num_samples_per_tree": args.num_samples_per_tree,
        "cfg_scale": args.cfg_scale,
        "num_ode_steps": args.num_ode_steps,
        "solver_method": args.solver_method,
        "output_format": args.output_format,
        "use_validation": args.use_validation,
        "seed": args.seed,
        "total_samples_generated": sample_counter,
        "generation_timestamp": timestamp,
        "source_config": config,
    }
    config_path = output_dir / "generation_config.json"
    with open(config_path, "w") as f:
        json.dump(gen_config, f, indent=2)
    print(f"Saved generation config to {config_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Generation complete!")
    print(f"  Total samples: {sample_counter}")
    print(f"  Output directory: {output_dir}")
    print(f"  Point clouds: {points_dir}")
    print(f"  Metadata CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic tree point clouds from a trained model"
    )

    # Experiment/Model Arguments
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
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
        default="FOR-species20K",
        help="Path to FOR-species20K directory",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="FOR-species20K/tree_metadata_dev.csv",
        help="Path to metadata CSV",
    )
    parser.add_argument(
        "--preprocessed_version",
        type=str,
        default="voxel_0.2m",
        help="Preprocessing version to use (raw, voxel_0.1m, voxel_0.2m)",
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
        "--output_dir",
        type=str,
        default="generated_samples",
        help="Output directory for generated samples",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="npy",
        choices=["npy", "laz"],
        help="Output format for point clouds",
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

    args = parser.parse_args()
    generate_samples(args)


if __name__ == "__main__":
    main()
