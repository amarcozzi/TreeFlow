"""
dataset.py - Point cloud dataset loader with optional augmentation
(Voxelization is now done during preprocessing)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    def __init__(
            self,
            data_path: Path,
            split: str = "train",
            preprocessed_version: str = "raw",
            num_samples: int = None,
            sample_exponent: float = None,
            rotation_augment: bool = False
    ):
        """
        Dataset for loading preprocessed point clouds (NPY format) with optional augmentation.

        Args:
            data_path: Path to FOR-species20K directory
            split: 'train' for dev set, 'test' for test set
            preprocessed_version: Which preprocessed version to use:
                                 - "raw": Original points (no voxelization)
                                 - "voxel_0.05m", "voxel_0.1m", etc.: Voxelized versions
            num_samples: Limit number of files to load (for debugging)
            sample_exponent: Exponent for power law sampling (lower = more aggressive skew toward full count)
                           0.5 = moderate skew, 0.3 = aggressive skew, 1.0 = uniform
                           None = no sampling
            rotation_augment: Whether to apply random rotation around Z-axis
        """
        self.data_path = Path(data_path)
        self.sample_exponent = sample_exponent
        self.rotation_augment = rotation_augment

        if split == "train":
            split_name = "dev"
        elif split == "test":
            split_name = "test"
        else:
            raise ValueError("Split must be 'train' or 'test'")

        # Build path to preprocessed data
        self.npy_directory = self.data_path / "npy" / preprocessed_version / split_name

        if not self.npy_directory.exists():
            raise FileNotFoundError(
                f"Preprocessed data not found at {self.npy_directory}\n"
                f"Please run preprocessing first with:\n"
                f"  python preprocess_laz_to_npy.py --voxel_sizes <size> (or --include_raw for raw)"
            )

        self.npy_files = sorted(self.npy_directory.glob("*.npy"))

        if num_samples is not None:
            # Randomly select a subset of files
            self.npy_files = list(np.random.choice(self.npy_files, num_samples, replace=False))

        self.file_id_to_idx = {f.stem: i for i, f in enumerate(self.npy_files)}

        if len(self.npy_files) == 0:
            raise FileNotFoundError(f"No .npy files found in {self.npy_directory}")

        print(f"Loaded dataset from: {self.npy_directory}")
        print(f"  Number of files: {len(self.npy_files)}")

    def __len__(self):
        return len(self.npy_files)

    def _sample_points(self, points, min_points=8):
        """
        Sample a subset of points using a power law distribution.
        This creates a right-skewed distribution that favors sampling near the full point count.

        The distribution is: ratio = uniform(0,1)^exponent
        - exponent < 1: Right-skewed (concentrates near 1)
        - exponent = 1: Uniform distribution
        - exponent > 1: Left-skewed (concentrates near 0)

        With exponent=0.5 (sqrt):
          - 50% of samples will have >75% of points
          - 75% of samples will have >56% of points
          - 90% of samples will have >32% of points

        Args:
            points: Input points, shape (N, 3)

        Returns:
            sampled_points: Subset of points, shape (M, 3) where M <= N
        """
        n = len(points)
        if n <= min_points:  # Don't sample if we have very few points
            return points

        # Sample from power law distribution
        u = np.random.uniform(0, 1)
        sample_ratio = u ** self.sample_exponent

        # Ensure at least 8 points are sampled
        num_to_sample = max(min_points, int(sample_ratio * n))

        # Randomly select points
        indices = np.random.choice(n, num_to_sample, replace=False)

        return points[indices]

    def _rotate_z(self, points):
        """
        Apply random rotation around Z-axis (vertical axis for trees).

        Args:
            points: Input points, shape (N, 3)

        Returns:
            rotated_points: Rotated points, shape (N, 3)
        """
        # Random angle in [0, 2π)
        theta = np.random.uniform(0, 2 * np.pi)

        # Rotation matrix around Z-axis
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # Apply rotation
        return points @ rotation_matrix.T

    def __getitem__(self, idx):
        filepath = self.npy_files[idx]
        file_id = filepath.stem

        # Load from NPY (already preprocessed/voxelized if requested)
        points = np.load(filepath)

        # Random point sampling (fast operation, keep it online)
        if self.sample_exponent is not None:
            points = self._sample_points(points)

        # Random rotation around Z-axis (fast operation, keep it online)
        if self.rotation_augment:
            points = self._rotate_z(points)

        return {
            'points': torch.from_numpy(points).float(),
            'file_id': file_id,
            'num_points': len(points)
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length point clouds."""
    return batch  # Return list of dicts, each with variable-length tensors


def visualize_augmentation(dataset, idx, num_samples=6):
    """
    Visualize the effect of augmentation by showing multiple samples of the same point cloud.

    Args:
        dataset: PointCloudDataset with augmentation enabled
        idx: Index of point cloud to visualize
        num_samples: Number of augmented versions to show
    """
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f"Augmentation Examples - File: {dataset.npy_files[idx].stem}", fontsize=16)

    for i in range(num_samples):
        sample = dataset[idx]
        points = sample['points']
        num_pts = sample['num_points']

        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=points[:, 2], cmap='viridis', s=1, alpha=0.6)
        ax.set_title(f"Sample {i + 1} ({num_pts} points)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Set equal aspect ratio
        max_range = np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]).max() / 2.0

        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()


def main():
    # Example usage with preprocessed NPY files
    dataset = PointCloudDataset(
        data_path=Path("./FOR-species20K"),
        split="train",
        preprocessed_version="voxel_0.1m",  # Use pre-voxelized data
        sample_exponent=0.3,
        rotation_augment=True
    )
    print(f"Dataset size: {len(dataset)}")

    # Visualize augmentation on a sample
    random_idx = np.random.randint(len(dataset))
    visualize_augmentation(dataset, random_idx, num_samples=6)

    # Show distribution of sampled point counts
    num_points_list = []
    for _ in range(1000):
        sample = dataset[random_idx]
        num_points_list.append(sample['num_points'])

    plt.figure(figsize=(10, 6))
    plt.hist(num_points_list, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Points Sampled')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Sampled Points (Power Law, exponent={dataset.sample_exponent})')
    plt.axvline(np.mean(num_points_list), color='r', linestyle='--', label=f'Mean: {np.mean(num_points_list):.0f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()