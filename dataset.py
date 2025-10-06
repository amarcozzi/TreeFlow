# dataset.py
import torch
import laspy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    def __init__(self, data_path: Path, split: str = "train", voxel_size: float = None, num_samples: int = None):
        """
        Dataset for loading point clouds without resampling or padding.

        Args:
            data_path: Path to FOR-species20K directory
            split: 'train' for dev set, 'test' for test set
            voxel_size: Optional voxel size in meters for downsampling (e.g., 0.1).
                       If None, no voxelization is performed.
        """
        self.data_path = data_path
        self.voxel_size = voxel_size

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
            raw_points = points.copy()
            points = self._voxelize(points)

        return {
            'points': torch.from_numpy(points),
            'raw_points': None if self.voxel_size is None else torch.from_numpy(raw_points),
            'file_id': file_id,
            'num_points': len(points)
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length point clouds."""
    return batch  # Return list of dicts, each with variable-length tensors


def visualize_voxelization(raw_points, voxelized_points, voxel_size, file_id):
    """
    Generates a 3-panel plot to compare a raw point cloud with its voxelized version.
    """
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle(f"Voxelization Validation | File: {file_id} | Voxel Size: {voxel_size}m", fontsize=16)

    # Use oblique view to show 3D structure
    view_elev = 25  # Look down at 25 degrees
    view_azim = 45  # Diagonal view

    # --- Plot 1: Original Point Cloud ---
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(raw_points[:, 0], raw_points[:, 1], raw_points[:, 2],
                s=0.5, alpha=0.2, c=raw_points[:, 2], cmap='viridis')  # Smaller points, less alpha
    ax1.set_title(f"Original ({len(raw_points)} points)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.view_init(elev=view_elev, azim=view_azim)  # Side view to see thinness

    # --- Plot 2: Voxelized Point Cloud ---
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(voxelized_points[:, 0], voxelized_points[:, 1], voxelized_points[:, 2],
                s=5, c='r', alpha=0.6)  # Reduced from s=15
    ax2.set_title(f"Voxelized ({len(voxelized_points)} points)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.view_init(elev=view_elev, azim=view_azim)  # Same view

    # --- Plot 3: Overlay ---
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(raw_points[:, 0], raw_points[:, 1], raw_points[:, 2],
                s=0.5, alpha=0.1, label='Original', c='blue')
    ax3.scatter(voxelized_points[:, 0], voxelized_points[:, 1], voxelized_points[:, 2],
                s=5, c='r', alpha=0.6, label='Voxelized')
    ax3.set_title("Overlay")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.legend()
    ax3.view_init(elev=view_elev, azim=view_azim)  # Same view

    # Use same axis limits for all plots
    x_lim = (raw_points[:, 0].min(), raw_points[:, 0].max())
    y_lim = (raw_points[:, 1].min(), raw_points[:, 1].max())
    z_lim = (raw_points[:, 2].min(), raw_points[:, 2].max())

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        # Force equal aspect ratio to see true proportions
        ax.set_box_aspect([x_lim[1]-x_lim[0], y_lim[1]-y_lim[0], z_lim[1]-z_lim[0]])

    plt.tight_layout()
    plt.show()

def main():
    # Example usage
    dataset = PointCloudDataset(data_path=Path("./FOR-species20K"), split="test", voxel_size=0.5)
    print(f"Dataset size: {len(dataset)}")

    # Sample a random point cloud
    file_id = "00085"
    file_id = None
    if file_id:
        idx = dataset.file_id_to_idx.get(file_id)
        sample = dataset[idx]
    else:
        random_idx = np.random.randint(len(dataset))
        sample = dataset[random_idx]
    print(f"Sampled file ID: {sample['file_id']}, Number of points: {sample['num_points']}")
    visualize_voxelization(sample['raw_points'].numpy(), sample['points'].numpy(), dataset.voxel_size, sample['file_id'])

    # # Find the largest point cloud
    # max_pts_idx = max(range(len(dataset)), key=lambda i: dataset[i]['num_points'])
    # max_pts = dataset[max_pts_idx]['num_points']
    # file_id = dataset[max_pts_idx]['file_id']
    # print(f"Largest point cloud: {file_id} with {max_pts} points")

if __name__ == "__main__":
    main()