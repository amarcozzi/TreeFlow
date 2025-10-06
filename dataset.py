# dataset.py
import torch
import laspy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
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
        Dataset for loading point clouds with optional augmentation.

        Args:
            data_path: Path to FOR-species20K directory
            split: 'train' for dev set, 'test' for test set
            voxel_size: Optional voxel size in meters for downsampling (e.g., 0.1).
                       If None, no voxelization is performed.
            num_samples: Limit number of files to load (for debugging)
            augment: Whether to apply data augmentation (point sampling and rotation)
            sample_exponent: Exponent for power law sampling (lower = more aggressive skew toward full count)
                           0.5 = moderate skew, 0.3 = aggressive skew, 1.0 = uniform
            rotation_augment: Whether to apply random rotation around Z-axis
        """
        self.data_path = data_path
        self.voxel_size = voxel_size
        self.augment = augment and (split == "train")  # Only augment training data
        self.sample_exponent = sample_exponent
        self.rotation_augment = rotation_augment

        if split == "train":
            self.laz_directory = data_path / "dev"
        elif split == "test":
            self.laz_directory = data_path / "test"
        else:
            raise ValueError("Split must be 'train' or 'test'")

        self.laz_files = list(self.laz_directory.glob("*.laz"))

        if num_samples is not None:
            self.laz_files = self.laz_files[:num_samples]

        self.file_id_to_idx = {f.stem: i for i, f in enumerate(self.laz_files)}

        print(f"Found {len(self.laz_files)} files for '{split}' split in {self.laz_directory}")
        if self.augment:
            print(f"Augmentation enabled: point_sampling=True (power_law, exponent={sample_exponent}), "
                  f"rotation={rotation_augment}")

        if len(self.laz_files) == 0:
            raise FileNotFoundError(f"No .laz files found in {self.laz_directory}")

    def __len__(self):
        return len(self.laz_files)

    def _voxelize(self, points):
        """
        Voxelize point cloud using numpy for efficiency.
        Returns points at the centers of occupied voxels.

        Args:
            points: Input points, shape (N, 3)

        Returns:
            voxelized_points: Points at voxel centers, shape (M, 3) where M <= N
        """
        if self.voxel_size is None:
            return points

        # Compute voxel indices for each point
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)

        # Find unique voxels (occupied voxels)
        unique_voxels = np.unique(voxel_indices, axis=0)

        # Compute voxel centers: center = (index + 0.5) * voxel_size
        voxel_centers = (unique_voxels + 0.5) * self.voxel_size

        return voxel_centers.astype(np.float32)

    def _sample_points(self, points):
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
        if n <= 1:
            return points

        # Sample from power law distribution
        u = np.random.uniform(0, 1)
        sample_ratio = u ** self.sample_exponent

        # Ensure at least 1 point is sampled
        num_to_sample = max(1, int(sample_ratio * n))

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
        filepath = self.laz_files[idx]
        file_id = filepath.stem

        with laspy.open(filepath) as laz_file:
            las_data = laz_file.read()
            # Load XYZ coordinates - shape (N, 3)
            points = np.array(las_data.xyz, dtype=np.float32)

        # Center the point cloud at the origin
        centroid_x = (points[:, 0].max() + points[:, 0].min()) / 2
        centroid_y = (points[:, 1].max() + points[:, 1].min()) / 2
        centroid_z = (points[:, 2].max() + points[:, 2].min()) / 2
        points[:, 0] -= centroid_x
        points[:, 1] -= centroid_y
        points[:, 2] -= centroid_z

        # Voxelize if requested
        if self.voxel_size is not None:
            points = self._voxelize(points)

        # Apply augmentation if enabled
        if self.augment:
            # Random point sampling with right-skewed distribution
            points = self._sample_points(points)

            # Random rotation around Z-axis
            if self.rotation_augment:
                points = self._rotate_z(points)

        return {
            'points': torch.from_numpy(points),
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
    fig.suptitle(f"Augmentation Examples - File: {dataset.laz_files[idx].stem}", fontsize=16)

    for i in range(num_samples):
        sample = dataset[idx]
        points = sample['points'].numpy()
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
    # Example usage with augmentation
    dataset = PointCloudDataset(
        data_path=Path("./FOR-species20K"),
        split="train",
        voxel_size=0.1,
        augment=True,
        sample_exponent=0.5,
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