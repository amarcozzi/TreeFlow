"""
dataset.py - Point cloud dataset loader with optional augmentation
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class PointCloudDataset(Dataset):
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        data_path: Path,
        preprocessed_version: str = "raw",
        species_map: dict = None,
        type_map: dict = None,
        sample_exponent: float = None,
        rotation_augment: bool = False,
        shuffle_augment: bool = False,
        max_points: int = None,
    ):
        """
        Dataset for loading preprocessed point clouds (NPY format) with conditioning info.

        Args:
            metadata_df: Pandas DataFrame containing ['filename', 'species', 'data_type', 'tree_H', 'file_path']
            data_path: Base path to FOR-species20K directory
            preprocessed_version: "voxel_0.1m", etc.
            species_map: Dictionary mapping species string to integer index
            type_map: Dictionary mapping data_type string to integer index
            sample_exponent, rotation_augment, etc.: Augmentation parameters
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.data_path = Path(data_path)
        self.sample_exponent = sample_exponent
        self.rotation_augment = rotation_augment
        self.shuffle_augment = shuffle_augment
        self.max_points = max_points

        self.species_map = species_map
        self.type_map = type_map

        print(f"  Initialized dataset with {len(self.metadata)} samples.")

    def __len__(self):
        return len(self.metadata)

    def _sample_points(self, points, min_points=8):
        """Sample a subset of points using a power law distribution."""
        n = len(points)
        if n <= min_points:
            return points

        u = np.random.uniform(0, 1)
        sample_ratio = u**self.sample_exponent
        num_to_sample = max(min_points, int(sample_ratio * n))
        indices = np.random.choice(n, num_to_sample, replace=False)

        return points[indices]

    def _rotate_z(self, points):
        """Apply random rotation around Z-axis."""
        theta = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rotation_matrix = np.array(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]],
            dtype=np.float32,
        )

        return points @ rotation_matrix.T

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Load NPY file
        points = np.load(row["file_path"])

        # 1. Augmentation (Sampling/Rotation)
        if self.sample_exponent is not None:
            points = self._sample_points(points)

        if self.rotation_augment:
            points = self._rotate_z(points)

        if self.shuffle_augment:
            np.random.shuffle(points)

        if self.max_points is not None and len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]

        # 2. Normalization & Conditioning
        # Strategy: Center at Mass -> Scale by Height -> Scale by 2.0 (for variance matching)
        height = float(row["tree_H"])

        # Center of Mass
        centroid = points.mean(axis=0)
        points_centered = points - centroid

        # Normalize by Height (Aspect Ratio Preserved)
        # Add epsilon to prevent div by zero
        points_normalized = points_centered / (height + 1e-6)

        # Scale to match Neural Net variance (approx [-1, 1] range)
        points_final = points_normalized * 2.0

        # Log-Height for conditioning
        height_norm = np.log(height)

        return {
            "points": torch.from_numpy(points_final).float(),
            "file_id": row["file_id"],
            "num_points": len(points_final),
            "species_idx": torch.tensor(
                self.species_map[row["species"]], dtype=torch.long
            ),
            "type_idx": torch.tensor(self.type_map[row["data_type"]], dtype=torch.long),
            "height_norm": torch.tensor(height_norm, dtype=torch.float32),
            "height_raw": torch.tensor(height, dtype=torch.float32),
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length point clouds."""
    return batch


def collate_fn_batched(batch):
    """
    Custom collate function that samples all point clouds to the minimum size in the batch.
    Includes conditioning information.
    """
    min_points = min(sample["num_points"] for sample in batch)

    sampled_points = []
    file_ids = []
    original_num_points = []

    # Conditioning lists
    species_idxs = []
    type_idxs = []
    height_norms = []
    height_raws = []

    for sample in batch:
        points = sample["points"]
        num_points = sample["num_points"]

        if num_points > min_points:
            indices = torch.randperm(num_points)[:min_points]
            points = points[indices]

        sampled_points.append(points)
        file_ids.append(sample["file_id"])
        original_num_points.append(num_points)

        # Collect conditioning
        species_idxs.append(sample["species_idx"])
        type_idxs.append(sample["type_idx"])
        height_norms.append(sample["height_norm"])
        height_raws.append(sample["height_raw"])

    return {
        "points": torch.stack(sampled_points, dim=0),
        "file_ids": file_ids,
        "original_num_points": original_num_points,
        "sampled_num_points": min_points,
        "species_idx": torch.stack(species_idxs),
        "type_idx": torch.stack(type_idxs),
        "height_norm": torch.stack(height_norms),
        "height_raw": torch.stack(height_raws),
    }


def create_datasets(
    data_path: str,
    csv_path: str,
    preprocessed_version: str = "voxel_0.1m",
    split_ratios: tuple = (0.8, 0.1, 0.1),  # Train, Val, Test
    seed: int = 42,
    **dataset_kwargs,
):
    """
    Factory function to create Train, Val, and Test datasets from the dev metadata CSV.
    We do not use the 'test' folder as it lacks metadata.
    """
    data_path = Path(data_path)
    csv_path = Path(csv_path)

    # 1. Load Metadata
    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 2. Filter for existence
    # The CSV has filenames like "/train/00070.las". We need "00070.npy" in the dev folder.
    npy_base_dir = data_path / "npy" / preprocessed_version / "dev"

    if not npy_base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {npy_base_dir}")

    # Extract ID from filename and check existence
    # Assuming standard format /train/XXXXX.las
    df["file_id"] = df["filename"].apply(lambda x: Path(x).stem)
    df["file_path"] = df["file_id"].apply(lambda x: npy_base_dir / f"{x}.npy")

    # Filter rows where file exists
    initial_len = len(df)
    df = df[df["file_path"].apply(lambda x: x.exists())]
    print(f"Found {len(df)}/{initial_len} matching NPY files in {npy_base_dir}")

    # 3. Create Global Mappings
    species_list = sorted(df["species"].unique())
    type_list = sorted(df["data_type"].unique())

    species_map = {s: i for i, s in enumerate(species_list)}
    type_map = {t: i for i, t in enumerate(type_list)}

    print(f"Species found: {len(species_list)}")
    print(f"Data types found: {len(type_list)}")

    # 4. Split Data
    train_ratio, val_ratio, test_ratio = split_ratios
    # Normalize ratios just in case
    total = sum(split_ratios)
    train_ratio /= total
    val_ratio /= total

    # Split Train vs (Val+Test)
    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio), random_state=seed, stratify=df["species"]
    )

    # Split Val vs Test
    relative_test_ratio = test_ratio / (test_ratio + val_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_df["species"],
    )

    print(f"Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 5. Create Datasets
    # Pass mappings to all datasets to ensure consistency
    common_args = {
        "data_path": data_path,
        "preprocessed_version": preprocessed_version,
        "species_map": species_map,
        "type_map": type_map,
        **dataset_kwargs,
    }

    train_ds = PointCloudDataset(train_df, **common_args)
    val_ds = PointCloudDataset(val_df, **common_args)
    test_ds = PointCloudDataset(test_df, **common_args)

    return train_ds, val_ds, test_ds, species_list, type_list


def visualize_augmentation(dataset, idx, num_samples=6):
    """
    Visualize the effect of augmentation by showing multiple samples of the same point cloud.
    """
    sample = dataset[idx]
    file_id = sample["file_id"]

    s_name = species_list[sample["species_idx"]]
    t_name = type_list[sample["type_idx"]]

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"Augmentation - File: {file_id}\nSpecies ID: {s_name} | Type: {t_name} | Height: {sample['height_raw']:.2f}m",
        fontsize=16,
    )

    for i in range(num_samples):
        # Re-fetch to trigger random augmentation
        s = dataset[idx]
        points = s["points"].numpy()

        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=points[:, 2],
            cmap="viridis",
            s=1,
            alpha=0.6,
        )
        ax.set_title(f"Sample {i + 1} ({len(points)} pts)")

        # Set limits based on normalization (-1 to 1 approx)
        # ax.set_xlim(-1.5, 1.5)
        # ax.set_ylim(-1.5, 1.5)
        # ax.set_zlim(-1.5, 1.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage for debugging
    data_path = "./FOR-species20K"
    csv_path = "./FOR-species20K/tree_metadata_dev.csv"

    try:
        train_ds, val_ds, test_ds, species_list, type_list = create_datasets(
            data_path=data_path,
            csv_path=csv_path,
            preprocessed_version="voxel_0.1m",
            sample_exponent=0.3,
            rotation_augment=True,
        )

        print("\nVisualizing random training sample...")
        idx = np.random.randint(len(train_ds))
        visualize_augmentation(train_ds, idx)

    except Exception as e:
        print(f"Skipping visualization (setup required): {e}")
