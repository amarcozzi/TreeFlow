"""
dataset.py - Point cloud dataset loader with optional augmentation
"""
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend for multiprocessing safety

import torch
import laspy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    """
    Dataset for loading LiDAR point clouds with optional voxelization and augmentation.

    Features:
    - Optional voxel-based downsampling
    - Point sampling with power law distribution (training only)
    - Random Z-axis rotation (training only)
    - Automatic centering at origin
    """

    def __init__(
            self,
            data_path: Path,
            split: str = "train",
            voxel_size: float = None,
            num_samples: int = None,
            augment: bool = True,
            sample_exponent: float = 0.5,
            rotation_augment: bool = True
    ):
        """
        Args:
            data_path: Path to FOR-species20K directory
            split: 'train' for dev set, 'test' for test set
            voxel_size: Voxel size in meters for downsampling. None = no voxelization
            num_samples: Limit number of files to load (for debugging)
            augment: Whether to apply data augmentation
            sample_exponent: Exponent for power law sampling
                           Lower values (0.3-0.5) = more aggressive toward full count
                           1.0 = uniform distribution
            rotation_augment: Whether to apply random Z-axis rotation
        """
        self.data_path = Path(data_path)
        self.voxel_size = voxel_size
        self.augment = augment and (split == "train")
        self.sample_exponent = sample_exponent
        self.rotation_augment = rotation_augment

        # Set directory based on split
        if split == "train":
            self.laz_directory = self.data_path / "dev"
        elif split == "test":
            self.laz_directory = self.data_path / "test"
        else:
            raise ValueError(f"Split must be 'train' or 'test', got '{split}'")

        # Find all LAZ files
        self.laz_files = sorted(self.laz_directory.glob("*.laz"))

        if num_samples is not None:
            self.laz_files = self.laz_files[:num_samples]

        if len(self.laz_files) == 0:
            raise FileNotFoundError(
                f"No .laz files found in {self.laz_directory}. "
                f"Please check your data path."
            )

        self.file_id_to_idx = {f.stem: i for i, f in enumerate(self.laz_files)}

        print(f"Found {len(self.laz_files)} files for '{split}' split")
        if self.augment:
            print(f"Augmentation: point_sampling (exp={sample_exponent}), "
                  f"rotation={rotation_augment}")

    def __len__(self):
        return len(self.laz_files)

    def _voxelize(self, points: np.ndarray) -> np.ndarray:
        """
        Downsample point cloud using voxel grid.

        Args:
            points: Input points (N, 3)

        Returns:
            Voxel centers (M, 3) where M <= N
        """
        if self.voxel_size is None:
            return points

        # Compute voxel indices
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)

        # Get unique voxels and their centers
        unique_voxels = np.unique(voxel_indices, axis=0)
        voxel_centers = (unique_voxels + 0.5) * self.voxel_size

        return voxel_centers.astype(np.float32)

    def _sample_points(self, points: np.ndarray) -> np.ndarray:
        """
        Sample points using power law distribution (right-skewed).

        With exponent=0.5:
        - 50% of samples have >75% of points
        - 75% of samples have >56% of points
        - 90% of samples have >32% of points

        Args:
            points: Input points (N, 3)

        Returns:
            Sampled points (M, 3) where M <= N
        """
        n = len(points)
        if n <= 1:
            return points

        # Power law sampling
        u = np.random.uniform(0, 1)
        sample_ratio = u ** self.sample_exponent

        # Ensure minimum of 8 points
        num_to_sample = max(8, int(sample_ratio * n))
        num_to_sample = min(num_to_sample, n)  # Don't exceed available points

        # Random selection without replacement
        indices = np.random.choice(n, num_to_sample, replace=False)

        return points[indices]

    def _rotate_z(self, points: np.ndarray) -> np.ndarray:
        """
        Apply random rotation around Z-axis.

        Args:
            points: Input points (N, 3)

        Returns:
            Rotated points (N, 3)
        """
        theta = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        return points @ rotation_matrix.T

    def __getitem__(self, idx: int):
        """
        Load and process a single point cloud.

        Returns:
            dict with keys:
                - points: numpy array (N, 3)
                - file_id: string identifier
                - num_points: int
        """
        filepath = self.laz_files[idx]
        file_id = filepath.stem

        try:
            # Read LAZ file
            with laspy.open(str(filepath)) as laz_file:
                las_data = laz_file.read()
                points = np.array(las_data.xyz, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load {filepath}: {e}")

        # Center at origin
        centroid = points.mean(axis=0)
        points -= centroid

        # Apply voxelization
        if self.voxel_size is not None:
            points = self._voxelize(points)

        # Apply augmentation (training only)
        if self.augment:
            points = self._sample_points(points)
            if self.rotation_augment:
                points = self._rotate_z(points)

        return {
            'points': points,
            'file_id': file_id,
            'num_points': len(points)
        }


def collate_fn(batch):
    """
    Custom collate function for variable-length point clouds.
    Returns list of dicts instead of batching.
    """
    return batch


def visualize_sample(dataset, idx, save_path=None):
    """
    Visualize a single point cloud sample.

    Args:
        dataset: PointCloudDataset instance
        idx: Sample index
        save_path: Optional path to save figure
    """
    sample = dataset[idx]
    points = sample['points']

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Color by height
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=points[:, 2], cmap='viridis', s=1, alpha=0.6)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f"File: {sample['file_id']}\nPoints: {sample['num_points']}")

    # Equal aspect ratio
    max_range = np.ptp(points, axis=0).max() / 2.0
    mid = points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_augmentation(dataset, idx, num_samples=6, save_path=None):
    """
    Visualize multiple augmented versions of the same point cloud.

    Args:
        dataset: PointCloudDataset with augmentation enabled
        idx: Index of point cloud to visualize
        num_samples: Number of augmented versions to show
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f"Augmentation - {dataset.laz_files[idx].stem}", fontsize=16)

    for i in range(num_samples):
        sample = dataset[idx]
        points = sample['points']

        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=points[:, 2], cmap='viridis', s=1, alpha=0.6)
        ax.set_title(f"Sample {i + 1} ({sample['num_points']} pts)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Equal aspect
        max_range = np.ptp(points, axis=0).max() / 2.0
        mid = points.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    """Example usage and testing."""
    data_path = Path("./FOR-species20K")

    # Create dataset with augmentation
    dataset = PointCloudDataset(
        data_path=data_path,
        split="train",
        voxel_size=0.1,
        augment=True,
        sample_exponent=0.5,
        rotation_augment=True
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test loading
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Points shape: {sample['points'].shape}")
    print(f"File ID: {sample['file_id']}")

    # Visualize
    visualize_sample(dataset, 0, save_path="sample.png")
    visualize_augmentation(dataset, 0, num_samples=6, save_path="augmentation.png")
    print("\nVisualizations saved!")


if __name__ == "__main__":
    main()