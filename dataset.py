"""
dataset.py - Point cloud dataset loader with optional augmentation
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zarr
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class PointCloudDataset(Dataset):
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        data_path: Path,
        species_map: dict = None,
        type_map: dict = None,
        sample_exponent: float = None,
        rotation_augment: bool = False,
        shuffle_augment: bool = False,
        max_points: int = None,
        split_name: str = None,
        cache_in_memory: bool = True,
    ):
        """
        Dataset for loading preprocessed point clouds (Zarr format) with conditioning info.

        Args:
            metadata_df: Pandas DataFrame containing ['filename', 'species', 'data_type', 'tree_H', 'file_path']
            data_path: Base path to FOR-species20K directory
            species_map: Dictionary mapping species string to integer index
            type_map: Dictionary mapping data_type string to integer index
            sample_exponent, rotation_augment, etc.: Augmentation parameters
            cache_in_memory: If True, load all point clouds into RAM at initialization
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.data_path = Path(data_path)
        self.sample_exponent = sample_exponent
        self.rotation_augment = rotation_augment
        self.shuffle_augment = shuffle_augment
        self.max_points = max_points
        self.cache_in_memory = cache_in_memory

        self.species_map = species_map
        self.type_map = type_map

        # Pre-load all data into memory to avoid I/O bottleneck during training
        self.point_cache = None
        if cache_in_memory:
            self._load_cache(split_name)

        print(
            f"  Initialized {split_name + ' ' if split_name is not None else ''}dataset with {len(self.metadata)} samples."
        )
        print(
            f"    Augmentation: max_points={max_points}, sample_exponent={sample_exponent}, rotation={rotation_augment}, shuffle={shuffle_augment}"
        )
        if cache_in_memory:
            print(f"    Data cached in memory.")

    def _load_cache(self, split_name: str = None, num_threads: int = 32):
        """Load all point clouds into memory in parallel."""
        n_samples = len(self.metadata)
        self.point_cache = [None] * n_samples
        desc = f"Caching {split_name}" if split_name else "Caching data"

        def load_one(idx):
            row = self.metadata.iloc[idx]
            return idx, zarr.load(row["file_path"])

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(load_one, idx) for idx in range(n_samples)]
            for future in tqdm(
                as_completed(futures), total=n_samples, desc=desc, leave=False
            ):
                idx, points = future.result()
                self.point_cache[idx] = points

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

        # Load from cache or disk
        if self.point_cache is not None:
            points = self.point_cache[idx].copy()  # Copy to avoid modifying cache
        else:
            points = zarr.load(row["file_path"])

        # Augmentation (Sampling/Rotation)
        if self.max_points is not None and len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]

        if self.sample_exponent is not None:
            points = self._sample_points(points)

        if self.rotation_augment:
            points = self._rotate_z(points)

        if self.shuffle_augment:
            np.random.shuffle(points)

        # Conditioning info
        height = float(row["tree_H"])
        height_norm = np.log(height)

        return {
            "points": torch.from_numpy(points).float(),
            "file_id": row["file_id"],
            "num_points": len(points),
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


def collate_fn_batched(
    batch, sample_exponent: float = None, min_points_floor: int = 256
):
    """
    Collate function that samples all point clouds to a target size.

    If sample_exponent is provided, applies batch-level exponential sampling:
    - Draws u ~ Uniform(0, 1)
    - Computes target = max(min_points_floor, int(u^sample_exponent * batch_max_points))
    - All samples are downsampled to target

    This provides regularization by varying sequence length across batches.

    Args:
        batch: List of samples from the dataset.
        sample_exponent: Exponent for power-law sampling. Higher = more samples near max.
                         None disables (uses min points in batch).
        min_points_floor: Minimum number of points (default 256).
    """
    max_points = batch[0]["num_points"]

    if sample_exponent is not None:
        u = torch.rand(1).item()
        target_points = max(min_points_floor, int((u**sample_exponent) * max_points))
    else:
        target_points = max_points

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

        # Points are already shuffled in __getitem__, just slice
        sampled_points.append(points[:target_points])
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
        "sampled_num_points": target_points,
        "species_idx": torch.stack(species_idxs),
        "type_idx": torch.stack(type_idxs),
        "height_norm": torch.stack(height_norms),
        "height_raw": torch.stack(height_raws),
    }


def make_collate_fn(sample_exponent: float = None, min_points_floor: int = 256):
    """
    Factory to create a collate function with batch-level point sampling.

    Args:
        sample_exponent: Exponent for power-law sampling (e.g., 0.3).
        min_points_floor: Minimum points per batch (default 256).
    """
    from functools import partial

    return partial(
        collate_fn_batched,
        sample_exponent=sample_exponent,
        min_points_floor=min_points_floor,
    )


def create_datasets(
    data_path: str,
    sample_exponent: float = None,
    sample_exp_train: bool = False,
    sample_exp_val: bool = True,
    sample_exp_test: bool = True,
    cache_train: bool = True,
    cache_val: bool = True,
    cache_test: bool = True,
    **dataset_kwargs,
):
    """
    Factory function to create Train, Val, and Test datasets.

    Args:
        data_path: Path to preprocessed dataset directory (e.g., "data/full" or "data/4096").
                   Must contain metadata.csv and *.zarr files.
        sample_exponent: Point sampling exponent value (e.g., 0.3).
        sample_exp_train: Apply sample_exponent to train? Default False (handled by collate_fn).
        sample_exp_val: Apply sample_exponent to val? Default True (for visualization).
        sample_exp_test: Apply sample_exponent to test? Default True (for evaluation).
        cache_train: Cache train data in memory? Default True.
        cache_val: Cache val data in memory? Default True.
        cache_test: Cache test data in memory? Default True.
        **dataset_kwargs: Passed to PointCloudDataset (max_points, rotation_augment, etc.)

    Returns:
        train_ds, val_ds, test_ds, species_list, type_list
    """
    data_path = Path(data_path)
    csv_path = data_path / "metadata.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    # 1. Load Metadata
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(csv_path)

    # Verify split column exists
    if "split" not in df.columns:
        raise ValueError(
            f"CSV missing 'split' column. Run preprocess_laz.py to create the dataset."
        )

    # 2. Build file paths and filter for existence
    df["file_id"] = df["filename"].apply(lambda x: Path(x).stem)
    df["file_path"] = df["file_id"].apply(lambda x: data_path / f"{x}.zarr")

    initial_len = len(df)
    df = df[df["file_path"].apply(lambda x: x.exists())]
    print(f"Found {len(df)}/{initial_len} matching Zarr files")

    # 3. Create Global Mappings
    species_list = sorted(df["species"].unique())
    type_list = sorted(df["data_type"].unique())

    species_map = {s: i for i, s in enumerate(species_list)}
    type_map = {t: i for i, t in enumerate(type_list)}

    print(f"Species found: {len(species_list)}")
    print(f"Data types found: {len(type_list)}")

    # 4. Split Data using pre-assigned split column
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    print(f"Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 5. Create Datasets
    common_args = {
        "data_path": data_path,
        "species_map": species_map,
        "type_map": type_map,
        **dataset_kwargs,
    }

    train_ds = PointCloudDataset(
        train_df,
        split_name="train",
        sample_exponent=sample_exponent if sample_exp_train else None,
        cache_in_memory=cache_train,
        **common_args,
    )
    val_ds = PointCloudDataset(
        val_df,
        split_name="val",
        sample_exponent=sample_exponent if sample_exp_val else None,
        cache_in_memory=cache_val,
        **common_args,
    )
    test_ds = PointCloudDataset(
        test_df,
        split_name="test",
        sample_exponent=sample_exponent if sample_exp_test else None,
        cache_in_memory=cache_test,
        **common_args,
    )

    return train_ds, val_ds, test_ds, species_list, type_list


def visualize_augmentation(dataset, idx, num_samples=6, denormalize=True):
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

        # Scale points to meters
        # points *= 2.0
        if denormalize:
            points = (points / 2.0) * s["height_raw"].numpy()
            points[:, 2] -= points[:, 2].min()

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
    data_path = "./data/preprocessed-full"

    try:
        sample_exp = 0.5
        train_ds, val_ds, test_ds, species_list, type_list = create_datasets(
            data_path=data_path,
            sample_exponent=sample_exp,
            cache_train=False,
            cache_val=False,
            cache_test=False,
            rotation_augment=True,
        )

        # Visualize from val_ds to see effect of sample_exponent
        # (train_ds doesn't use sample_exponent - it's handled by collate_fn)
        print(f"\nVisualizing random val sample with sample_exponent={sample_exp}...")
        idx = np.random.randint(len(val_ds))
        visualize_augmentation(val_ds, idx, denormalize=False)

        # Histogram of point counts to show impact of sample_exponent
        print(f"\nSampling same tree 100 times to show point count distribution...")
        n_iterations = 100
        point_counts = []
        for _ in range(n_iterations):
            sample = val_ds[idx]
            point_counts.append(sample["num_points"])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(point_counts, bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(
            np.mean(point_counts),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(point_counts):.0f}",
        )
        ax.axvline(
            np.median(point_counts),
            color="orange",
            linestyle="--",
            label=f"Median: {np.median(point_counts):.0f}",
        )
        ax.set_xlabel("Number of Points")
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"Point Count Distribution (sample_exponent={sample_exp}, n={n_iterations})"
        )
        ax.legend()
        plt.tight_layout()
        plt.show()

        print(f"  Min: {min(point_counts)}, Max: {max(point_counts)}")
        print(f"  Mean: {np.mean(point_counts):.1f}, Std: {np.std(point_counts):.1f}")

    except Exception as e:
        print(f"Skipping visualization (setup required): {e}")
