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
            rotation_augment: bool = False,
            max_points: int = None,
    ):
        """
        Dataset for loading preprocessed point clouds (NPY format) with optional augmentation.

        Args:
            data_path: Path to FOR-species20K directory
            split: 'train' for dev set, 'test' for test set, or 'mixed' for both combined.
            preprocessed_version: Which preprocessed version to use:
                                 - "raw": Original points (no voxelization)
                                 - "voxel_0.05m", "voxel_0.1m", etc.: Voxelized versions
            num_samples: Limit number of files to load (for debugging)
            sample_exponent: Exponent for power law sampling (lower = more aggressive skew toward full count)
                           0.5 = moderate skew, 0.3 = aggressive skew, 1.0 = uniform
                           None = no sampling
            rotation_augment: Whether to apply random rotation around Z-axis
            max_points: Maximum number of points per sample (uniformly samples if exceeded)
        """
        self.data_path = Path(data_path)
        self.sample_exponent = sample_exponent
        self.rotation_augment = rotation_augment
        self.max_points = max_points

        # Determine which directories to load from based on the split
        if split == "train":
            split_dirs = [self.data_path / "npy" / preprocessed_version / "dev"]
        elif split == "test":
            split_dirs = [self.data_path / "npy" / preprocessed_version / "test"]
        elif split == "mixed":
            dev_dir = self.data_path / "npy" / preprocessed_version / "dev"
            test_dir = self.data_path / "npy" / preprocessed_version / "test"
            split_dirs = [dev_dir, test_dir]
        else:
            raise ValueError("Split must be 'train', 'test', or 'mixed'")

        # Load file paths from all specified directories
        self.npy_files = []
        print(f"Loading dataset for split: '{split}'")
        for directory in split_dirs:
            if not directory.exists():
                raise FileNotFoundError(
                    f"Preprocessed data not found at {directory}\n"
                    f"Please run preprocessing first with:\n"
                    f"  python preprocess_laz_to_npy.py --normalize --voxel_sizes <size> (or --include_raw)"
                )
            print(f"  - Scanning directory: {directory}")
            self.npy_files.extend(directory.glob("*.npy"))

        # Sort the combined list for deterministic behavior
        self.npy_files = sorted(self.npy_files)

        if num_samples is not None:
            # Randomly select a subset of files
            self.npy_files = list(np.random.choice(self.npy_files, num_samples, replace=False))

        self.file_id_to_idx = {f.stem: i for i, f in enumerate(self.npy_files)}

        if len(self.npy_files) == 0:
            raise FileNotFoundError(f"No .npy files found in the specified directories: {split_dirs}")

        print(f"  Total number of files found: {len(self.npy_files)}")

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

        # Ensure at least min_points are sampled
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

        # Load from NPY (already preprocessed/voxelized/normalized)
        points = np.load(filepath)

        # Random point sampling
        if self.sample_exponent is not None:
            points = self._sample_points(points)

        # Random rotation around Z-axis
        if self.rotation_augment:
            points = self._rotate_z(points)

        # Sample to max_points if specified
        if self.max_points is not None and len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]

        return {
            'points': torch.from_numpy(points).float(),
            'file_id': file_id,
            'num_points': len(points)
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length point clouds."""
    return batch  # Return list of dicts, each with variable-length tensors


def collate_fn_batched(batch):
    """
    Custom collate function that samples all point clouds to the minimum size in the batch.
    This allows proper batching for faster training, at the cost of variable resolution.

    Args:
        batch: List of dicts, each containing 'points' (N_i, 3), 'file_id', 'num_points'

    Returns:
        dict with:
            'points': (B, N_min, 3) tensor - all point clouds sampled to minimum size
            'file_ids': list of file IDs
            'original_num_points': list of original point counts
            'sampled_num_points': minimum number of points (same for all)
    """
    # Find minimum number of points in this batch
    min_points = min(sample['num_points'] for sample in batch)

    # Sample each point cloud to min_points
    sampled_points = []
    file_ids = []
    original_num_points = []

    for sample in batch:
        points = sample['points']  # (N, 3)
        num_points = sample['num_points']

        if num_points > min_points:
            # Randomly sample min_points from this point cloud
            indices = torch.randperm(num_points)[:min_points]
            points = points[indices]

        sampled_points.append(points)
        file_ids.append(sample['file_id'])
        original_num_points.append(num_points)

    # Stack into a batch tensor
    batched_points = torch.stack(sampled_points, dim=0)  # (B, N_min, 3)

    return {
        'points': batched_points,
        'file_ids': file_ids,
        'original_num_points': original_num_points,
        'sampled_num_points': min_points
    }


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
        split="mixed",
        preprocessed_version="voxel_0.1m",
        sample_exponent=0.3,
        rotation_augment=True,
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