# dataset.py
import torch
import laspy
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset



class PointCloudDataset(Dataset):
    def __init__(self, data_path: Path, split: str = "train", num_points: int = 2048):
        self.data_path = data_path
        self.num_points = num_points

        if split == "train":
            self.laz_directory = data_path / "dev"
        elif split == "test":
            self.laz_directory = data_path / "test"
        else:
            raise ValueError("Split must be 'train' or 'test'")

        self.laz_files = list(self.laz_directory.glob("*.laz"))
        print(f"Found {len(self.laz_files)} files for the '{split}' split in {self.laz_directory}")
        if len(self.laz_files) == 0:
            raise FileNotFoundError(f"No .laz files found in {self.laz_directory}. Please check the data_path.")

    def __len__(self):
        return len(self.laz_files)

    def __getitem__(self, idx):
        filepath = self.laz_files[idx]
        file_id = filepath.stem

        with laspy.open(filepath) as laz_file:
            las_data = laz_file.read()
            points = np.array(las_data.xyz, dtype=np.float32)

        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
        elif len(points) < self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=True)
            points = points[indices]

        return torch.from_numpy(points), file_id