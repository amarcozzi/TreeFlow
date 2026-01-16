"""
dataset.py - Load pre-normalized point clouds with augmentation

Run preprocess_laz_to_npy.py first to create normalized NPY files.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class PointCloudDataset(Dataset):
    """
    Dataset for pre-normalized point clouds.

    Expects NPY files already normalized via preprocess_laz_to_npy.py:
        points_norm = (points - centroid) / z_extent * 2.0

    This produces points in approximately [-1, 1] range.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        data_path: Path,
        species_map: dict = None,
        type_map: dict = None,
        max_points: int = None,
        rotation_augment: bool = False,
        shuffle_augment: bool = False,
        split_name: str = None,
    ):
        """
        Args:
            metadata_df: DataFrame with columns ['filename', 'species', 'data_type', 'tree_H', 'file_path']
            data_path: Base path to data directory
            species_map: Dict mapping species name to index
            type_map: Dict mapping data_type to index
            max_points: Maximum points to sample (None = use all)
            rotation_augment: Apply random Z-axis rotation
            shuffle_augment: Shuffle point order
            split_name: Name for logging (train/val/test)
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.data_path = Path(data_path)
        self.species_map = species_map
        self.type_map = type_map
        self.max_points = max_points
        self.rotation_augment = rotation_augment
        self.shuffle_augment = shuffle_augment

        print(
            f"  Initialized {split_name + ' ' if split_name else ''}dataset "
            f"with {len(self.metadata)} samples."
        )
        print(
            f"    max_points={max_points}, rotation={rotation_augment}, shuffle={shuffle_augment}"
        )

    def __len__(self):
        return len(self.metadata)

    def _rotate_z(self, points):
        """Apply random rotation around Z-axis."""
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        return (points @ R.T).astype(np.float32)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Load pre-normalized points
        points = np.load(row["file_path"]).astype(np.float32)

        # Subsample if needed
        if self.max_points and len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]

        # Augmentation
        if self.rotation_augment:
            points = self._rotate_z(points)
        if self.shuffle_augment:
            np.random.shuffle(points)

        # Height conditioning (log of raw height from metadata)
        height = float(row["tree_H"])

        return {
            "points": torch.from_numpy(points),
            "file_id": row["file_id"],
            "num_points": len(points),
            "species_idx": torch.tensor(
                self.species_map[row["species"]], dtype=torch.long
            ),
            "type_idx": torch.tensor(self.type_map[row["data_type"]], dtype=torch.long),
            "height_norm": torch.tensor(np.log(height), dtype=torch.float32),
            "height_raw": torch.tensor(height, dtype=torch.float32),
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length point clouds."""
    return batch


def collate_fn_batched(batch):
    """
    Collate function that samples all point clouds to the minimum size in the batch.
    """
    min_points = min(sample["num_points"] for sample in batch)

    sampled_points = []
    file_ids = []
    original_num_points = []
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
    npy_subdir: str = "preprocessed",
    split_ratios: tuple = (0.8, 0.1, 0.1),
    seed: int = 42,
    **dataset_kwargs,
):
    """
    Create Train, Val, and Test datasets from metadata CSV.

    Args:
        data_path: Path to FOR-species20K directory
        csv_path: Path to tree_metadata_dev.csv
        npy_subdir: Subdirectory containing NPY files (default: "preprocessed")
        split_ratios: (train, val, test) ratios
        seed: Random seed for reproducible splits
        **dataset_kwargs: Passed to PointCloudDataset (max_points, rotation_augment, etc.)

    Returns:
        train_ds, val_ds, test_ds, species_list, type_list
    """
    data_path = Path(data_path)
    csv_path = Path(csv_path)

    # Load metadata
    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Build file paths
    npy_dir = data_path / npy_subdir / "dev"
    if not npy_dir.exists():
        raise FileNotFoundError(f"NPY directory not found: {npy_dir}")

    df["file_id"] = df["filename"].apply(lambda x: Path(x).stem)
    df["file_path"] = df["file_id"].apply(lambda x: npy_dir / f"{x}.npy")

    # Filter to existing files
    initial_len = len(df)
    df = df[df["file_path"].apply(lambda x: x.exists())]
    print(f"Found {len(df)}/{initial_len} matching NPY files in {npy_dir}")

    # Create mappings
    species_list = sorted(df["species"].unique())
    type_list = sorted(df["data_type"].unique())
    species_map = {s: i for i, s in enumerate(species_list)}
    type_map = {t: i for i, t in enumerate(type_list)}

    print(f"Species: {len(species_list)}, Types: {len(type_list)}")

    # Split data
    train_ratio, val_ratio, test_ratio = split_ratios
    total = sum(split_ratios)
    train_ratio /= total
    val_ratio /= total

    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio), random_state=seed, stratify=df["species"]
    )
    relative_test_ratio = test_ratio / (test_ratio + val_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_df["species"],
    )

    print(f"Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Create datasets
    common_args = {
        "data_path": data_path,
        "species_map": species_map,
        "type_map": type_map,
        **dataset_kwargs,
    }

    train_ds = PointCloudDataset(train_df, split_name="train", **common_args)
    val_ds = PointCloudDataset(val_df, split_name="val", **common_args)
    test_ds = PointCloudDataset(test_df, split_name="test", **common_args)

    return train_ds, val_ds, test_ds, species_list, type_list


def visualize_samples(dataset, idx, num_samples=6):
    """Visualize multiple augmented versions of the same sample."""
    sample = dataset[idx]
    file_id = sample["file_id"]
    height = sample["height_raw"].item()

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f"File: {file_id} | Height: {height:.2f}m", fontsize=16)

    for i in range(num_samples):
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
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    data_path = "./FOR-species20K"
    csv_path = "./FOR-species20K/tree_metadata_dev.csv"

    try:
        train_ds, val_ds, test_ds, species_list, type_list = create_datasets(
            data_path=data_path,
            csv_path=csv_path,
            max_points=16834,
            rotation_augment=True,
        )

        print("\nVisualizing random training sample...")
        idx = np.random.randint(len(train_ds))
        visualize_samples(train_ds, idx)

    except Exception as e:
        print(f"Skipping visualization: {e}")
