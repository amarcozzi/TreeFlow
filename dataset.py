# dataset.py
import torch
import laspy
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    def __init__(self, data_path: Path, split: str = "train"):
        """
        Dataset for loading point clouds without resampling or padding.

        Args:
            data_path: Path to FOR-species20K directory
            split: 'train' for dev set, 'test' for test set
        """
        self.data_path = data_path

        if split == "train":
            self.laz_directory = data_path / "dev"
        elif split == "test":
            self.laz_directory = data_path / "test"
        else:
            raise ValueError("Split must be 'train' or 'test'")

        self.laz_files = list(self.laz_directory.glob("*.laz"))
        print(f"Found {len(self.laz_files)} files for '{split}' split in {self.laz_directory}")

        if len(self.laz_files) == 0:
            raise FileNotFoundError(f"No .laz files found in {self.laz_directory}")

    def __len__(self):
        return len(self.laz_files)

    def __getitem__(self, idx):
        filepath = self.laz_files[idx]
        file_id = filepath.stem

        with laspy.open(filepath) as laz_file:
            las_data = laz_file.read()
            # Load XYZ coordinates - shape (N, 3)
            points = np.array(las_data.xyz, dtype=np.float32)

        # Normalize to zero mean and unit variance
        points_mean = points.mean(axis=0, keepdims=True)
        points_std = points.std(axis=0, keepdims=True) + 1e-8
        points_normalized = (points - points_mean) / points_std

        return {
            'points': torch.from_numpy(points_normalized),
            'mean': torch.from_numpy(points_mean).squeeze(0),
            'std': torch.from_numpy(points_std).squeeze(0),
            'file_id': file_id
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length point clouds."""
    return batch  # Return list of dicts, each with variable-length tensors