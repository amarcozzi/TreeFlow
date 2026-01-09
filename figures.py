"""
Figure 1: Example input data to the model
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import create_datasets


def create_figure_1(
    data_path: str = "./FOR-species20K",
    csv_path: str = "./FOR-species20K/tree_metadata_dev.csv",
    output_dir: str = "figures",
    seed: int = 42,
):
    """
    Create 4 PDF subfigures showing example tree point clouds as input to the model.

    Args:
        data_path: Path to FOR-species20K directory
        csv_path: Path to tree metadata CSV
        output_dir: Directory to save PDF figures
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load dataset with same settings as training
    # (from submit_train_transformer_8_256.sh)
    print("Loading dataset...")
    train_ds, val_ds, test_ds, species_list, type_list = create_datasets(
        data_path=data_path,
        csv_path=csv_path,
        preprocessed_version="raw",
        rotation_augment=True,
        shuffle_augment=True,
        max_points=4096,
    )

    # Select 4 diverse samples from training set
    # Pick samples with different species for variety
    n_samples = 4
    indices = np.random.choice(
        len(train_ds), size=min(100, len(train_ds)), replace=False
    )

    # Try to get diverse species
    selected_indices = []
    seen_species = set()

    for idx in indices:
        sample = train_ds[idx]
        species_idx = sample["species_idx"].item()
        if species_idx not in seen_species or len(selected_indices) < n_samples:
            selected_indices.append(idx)
            seen_species.add(species_idx)
        if len(selected_indices) >= n_samples:
            break

    # If we don't have enough, just take random ones
    while len(selected_indices) < n_samples:
        idx = np.random.randint(len(train_ds))
        if idx not in selected_indices:
            selected_indices.append(idx)

    print(f"Selected {len(selected_indices)} samples for visualization")

    # Collect metadata for JSON output
    metadata = {}

    # Create individual PDF figures
    for i, idx in enumerate(selected_indices):
        sample = train_ds[idx]
        points = sample["points"].numpy()
        height_raw = sample["height_raw"].item()
        species_idx = sample["species_idx"].item()
        type_idx = sample["type_idx"].item()
        file_id = sample["file_id"]

        # Convert normalized points back to meters
        points_meters = (points / 2.0) * height_raw
        # Shift Z so ground is at 0
        points_meters[:, 2] -= points_meters[:, 2].min()

        species_name = species_list[species_idx]
        type_name = type_list[type_idx]

        print(
            f"  Sample {i+1}: {file_id} - {species_name} ({type_name}), "
            f"H={height_raw:.1f}m, {len(points)} points"
        )

        # Store metadata
        subfig_key = chr(ord("a") + i)
        metadata[subfig_key] = {
            "file_id": file_id,
            "species": species_name,
            "data_type": type_name,
            "height_m": round(height_raw, 2),
            "num_points": len(points),
        }

        # Create figure
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")

        # Plot point cloud colored by height
        scatter = ax.scatter(
            points_meters[:, 0],
            points_meters[:, 1],
            points_meters[:, 2],
            c=points_meters[:, 2],
            cmap="viridis",
            s=2,
            alpha=0.7,
        )

        # Set equal aspect ratio
        max_range = (
            np.array(
                [
                    points_meters[:, 0].max() - points_meters[:, 0].min(),
                    points_meters[:, 1].max() - points_meters[:, 1].min(),
                    points_meters[:, 2].max() - points_meters[:, 2].min(),
                ]
            ).max()
            / 2.0
        )

        mid_x = (points_meters[:, 0].max() + points_meters[:, 0].min()) / 2
        mid_y = (points_meters[:, 1].max() + points_meters[:, 1].min()) / 2
        mid_z = (points_meters[:, 2].max() + points_meters[:, 2].min()) / 2

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Labels
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        # Clean up the view
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()

        # Save as PDF
        output_path = output_dir / f"figure_1_{chr(ord('a') + i)}.pdf"
        fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)

        print(f"    Saved: {output_path}")

    # Save metadata to JSON
    metadata_path = output_dir / "figure_1.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")

    print(f"\nFigure 1 subfigures saved to {output_dir}/")


if __name__ == "__main__":
    create_figure_1()
