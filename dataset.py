# dataset.py
import torch
import laspy
import numpy as np
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

        # Voxelize if requested
        if self.voxel_size is not None:
            raw_points = points.copy()
            points = self._voxelize(points)

        # Normalize to zero mean and unit variance
        points_mean = points.mean(axis=0, keepdims=True)
        points_std = points.std(axis=0, keepdims=True) + 1e-8
        points_normalized = (points - points_mean) / points_std

        return {
            'points': torch.from_numpy(points_normalized),
            'raw_points': None if self.voxel_size is None else torch.from_numpy(raw_points),
            'mean': torch.from_numpy(points_mean).squeeze(0),
            'std': torch.from_numpy(points_std).squeeze(0),
            'file_id': file_id,
            'num_points': len(points)
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length point clouds."""
    return batch  # Return list of dicts, each with variable-length tensors

def main():
    # Example usage
    dataset = PointCloudDataset(data_path=Path("./FOR-species20K"), split="train", voxel_size=0.5)
    print(f"Dataset size: {len(dataset)}")

    # Find the largest point cloud
    max_pts_idx = max(range(len(dataset)), key=lambda i: dataset[i]['num_points'])
    max_pts = dataset[max_pts_idx]['num_points']
    file_id = dataset[max_pts_idx]['file_id']
    print(f"Largest point cloud: {file_id} with {max_pts} points")

if __name__ == "__main__":
    main()