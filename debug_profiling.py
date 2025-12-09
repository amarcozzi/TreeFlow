"""
debug_profiling.py - Enhanced profiling to identify DataLoader bottlenecks
"""

import torch
import time
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.transformer import PointNet2UnetForFlowMatching
from dataset import PointCloudDataset, collate_fn
from flow_matching.path import CondOTProbPath


def profile_dataset_components(data_path, num_samples=100):
    """Profile individual components of the dataset __getitem__ method."""
    print("\n" + "=" * 80)
    print("PROFILING DATASET COMPONENTS (Detailed Breakdown)")
    print("=" * 80)

    dataset = PointCloudDataset(
        Path(data_path),
        split="train",
        voxel_size=0.1,
        sample_exponent=0.5,
        rotation_augment=True,
    )

    times = {
        "file_io": [],
        "voxelization": [],
        "sampling": [],
        "rotation": [],
        "tensor_conversion": [],
        "total": [],
    }

    point_counts = {"raw": [], "after_voxel": [], "after_sample": []}

    print(f"\nProcessing {num_samples} samples...")

    for i in tqdm(range(num_samples)):
        idx = np.random.randint(len(dataset))
        filepath = dataset.npy_files[idx]

        total_start = time.time()

        # 1. FILE I/O
        io_start = time.time()
        points = np.load(filepath)
        times["file_io"].append(time.time() - io_start)
        point_counts["raw"].append(len(points))

        # 2. VOXELIZATION
        voxel_start = time.time()
        if dataset.voxel_size is not None:
            points = dataset._voxelize(points)
        times["voxelization"].append(time.time() - voxel_start)
        point_counts["after_voxel"].append(len(points))

        # 3. SAMPLING
        sample_start = time.time()
        if dataset.sample_exponent is not None:
            points = dataset._sample_points(points)
        times["sampling"].append(time.time() - sample_start)
        point_counts["after_sample"].append(len(points))

        # 4. ROTATION
        rotation_start = time.time()
        if dataset.rotation_augment:
            points = dataset._rotate_z(points)
        times["rotation"].append(time.time() - rotation_start)

        # 5. TENSOR CONVERSION
        tensor_start = time.time()
        _ = torch.from_numpy(points).float()
        times["tensor_conversion"].append(time.time() - tensor_start)

        times["total"].append(time.time() - total_start)

    # Print results
    print("\n" + "-" * 80)
    print("TIME BREAKDOWN (per sample)")
    print("-" * 80)

    total_avg = np.mean(times["total"]) * 1000

    for key in ["file_io", "voxelization", "sampling", "rotation", "tensor_conversion"]:
        avg = np.mean(times[key]) * 1000
        std = np.std(times[key]) * 1000
        pct = (np.mean(times[key]) / np.mean(times["total"])) * 100
        print(f"  {key:20s}: {avg:7.2f}ms ± {std:6.2f}ms  ({pct:5.1f}%)")

    print(f"  {'TOTAL':20s}: {total_avg:7.2f}ms")

    # Print point count statistics
    print("\n" + "-" * 80)
    print("POINT COUNT STATISTICS")
    print("-" * 80)
    print(
        f"  Raw points:          {np.mean(point_counts['raw']):8.0f} ± {np.std(point_counts['raw']):7.0f} (min={min(point_counts['raw'])}, max={max(point_counts['raw'])})"
    )
    print(
        f"  After voxelization:  {np.mean(point_counts['after_voxel']):8.0f} ± {np.std(point_counts['after_voxel']):7.0f} (min={min(point_counts['after_voxel'])}, max={max(point_counts['after_voxel'])})"
    )
    print(
        f"  After sampling:      {np.mean(point_counts['after_sample']):8.0f} ± {np.std(point_counts['after_sample']):7.0f} (min={min(point_counts['after_sample'])}, max={max(point_counts['after_sample'])})"
    )

    # Calculate reduction ratios
    voxel_reduction = (
        np.mean(point_counts["after_voxel"]) / np.mean(point_counts["raw"]) * 100
    )
    sample_reduction = (
        np.mean(point_counts["after_sample"])
        / np.mean(point_counts["after_voxel"])
        * 100
    )
    total_reduction = (
        np.mean(point_counts["after_sample"]) / np.mean(point_counts["raw"]) * 100
    )

    print("\n" + "-" * 80)
    print("REDUCTION RATIOS")
    print("-" * 80)
    print(f"  Voxelization keeps:  {voxel_reduction:5.1f}% of points")
    print(f"  Sampling keeps:      {sample_reduction:5.1f}% of voxelized points")
    print(f"  Total reduction:     {total_reduction:5.1f}% of original points")


def profile_voxelization_algorithms(data_path, num_samples=50):
    """Compare different voxelization approaches."""
    print("\n" + "=" * 80)
    print("TESTING VOXELIZATION ALGORITHMS")
    print("=" * 80)

    dataset = PointCloudDataset(
        Path(data_path),
        split="train",
        voxel_size=None,  # We'll do voxelization manually
    )

    # Load some sample point clouds
    samples = []
    for i in range(num_samples):
        idx = np.random.randint(len(dataset))
        filepath = dataset.npy_files[idx]
        points = np.load(filepath)
        samples.append(points)

    voxel_size = 0.1

    print(f"\nComparing voxelization methods on {num_samples} samples...")
    print(f"Voxel size: {voxel_size}m")

    # Method 1: Current numpy-based approach
    print("\n1. Current numpy method (unique voxels):")
    times_np = []
    for points in tqdm(samples, desc="Numpy method"):
        start = time.time()
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        unique_voxels = np.unique(voxel_indices, axis=0)
        voxel_centers = (unique_voxels + 0.5) * voxel_size
        times_np.append(time.time() - start)

    print(f"   Time: {np.mean(times_np)*1000:.2f}ms ± {np.std(times_np)*1000:.2f}ms")

    # Method 2: Dictionary-based (might be slower but let's check)
    print("\n2. Dictionary-based method:")
    times_dict = []
    for points in tqdm(samples, desc="Dict method"):
        start = time.time()
        voxel_dict = {}
        for point in points:
            voxel_idx = tuple(np.floor(point / voxel_size).astype(np.int32))
            if voxel_idx not in voxel_dict:
                voxel_dict[voxel_idx] = []
            voxel_dict[voxel_idx].append(point)

        # Use voxel centers
        voxel_centers = np.array(
            [(np.array(k) + 0.5) * voxel_size for k in voxel_dict.keys()]
        )
        times_dict.append(time.time() - start)

    print(
        f"   Time: {np.mean(times_dict)*1000:.2f}ms ± {np.std(times_dict)*1000:.2f}ms"
    )
    print(f"   Speedup vs current: {np.mean(times_dict)/np.mean(times_np):.2f}x")


def profile_file_io_formats(data_path, num_samples=50):
    """Test if NPY loading is the bottleneck."""
    print("\n" + "=" * 80)
    print("PROFILING FILE I/O")
    print("=" * 80)

    dataset = PointCloudDataset(Path(data_path), split="train", voxel_size=None)

    times = []
    file_sizes = []

    print(f"\nLoading {num_samples} NPY files...")

    for i in tqdm(range(num_samples)):
        idx = np.random.randint(len(dataset))
        filepath = dataset.npy_files[idx]

        # Get file size
        file_sizes.append(filepath.stat().st_size / 1024)  # KB

        # Time loading
        start = time.time()
        points = np.load(filepath)
        times.append(time.time() - start)

    print("\n" + "-" * 80)
    print("FILE I/O STATISTICS")
    print("-" * 80)
    print(f"  Avg load time:  {np.mean(times)*1000:.2f}ms ± {np.std(times)*1000:.2f}ms")
    print(f"  Avg file size:  {np.mean(file_sizes):.1f}KB ± {np.std(file_sizes):.1f}KB")
    print(f"  Min load time:  {min(times)*1000:.2f}ms")
    print(f"  Max load time:  {max(times)*1000:.2f}ms")
    print(f"  Throughput:     {np.mean(file_sizes)/np.mean(times)/1024:.1f} MB/s")


def profile_with_different_configs(data_path, batch_size=64):
    """Test different dataset configurations."""
    print("\n" + "=" * 80)
    print("COMPARING DATASET CONFIGURATIONS")
    print("=" * 80)

    configs = [
        {
            "name": "Baseline (no processing)",
            "voxel_size": None,
            "sample_exponent": None,
            "rotation_augment": False,
        },
        {
            "name": "Only voxelization",
            "voxel_size": 0.1,
            "sample_exponent": None,
            "rotation_augment": False,
        },
        {
            "name": "Only sampling",
            "voxel_size": None,
            "sample_exponent": 0.5,
            "rotation_augment": False,
        },
        {
            "name": "Only rotation",
            "voxel_size": None,
            "sample_exponent": None,
            "rotation_augment": True,
        },
        {
            "name": "Voxel + sampling",
            "voxel_size": 0.1,
            "sample_exponent": 0.5,
            "rotation_augment": False,
        },
        {
            "name": "All augmentations",
            "voxel_size": 0.1,
            "sample_exponent": 0.5,
            "rotation_augment": True,
        },
    ]

    for config in configs:
        print(f"\n{config['name']}:")
        print(
            f"  voxel_size={config['voxel_size']}, sample_exponent={config['sample_exponent']}, rotation={config['rotation_augment']}"
        )

        dataset = PointCloudDataset(
            Path(data_path),
            split="train",
            **{k: v for k, v in config.items() if k != "name"},
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        times = []
        start = time.time()

        for i, batch in enumerate(loader):
            if i >= 5:
                break
            if i > 0:  # Skip first batch (warmup)
                times.append(time.time() - start)
            start = time.time()

        print(f"  Avg batch time: {np.mean(times):.3f}s")


def profile_data_loading(data_path, batch_size=64, num_batches=10):
    """Profile data loading speed."""
    print("\n" + "=" * 80)
    print("PROFILING DATA LOADING")
    print("=" * 80)

    dataset = PointCloudDataset(
        Path(data_path),
        split="train",
        voxel_size=0.1,
        sample_exponent=0.5,
        rotation_augment=True,
    )

    # Test different worker configurations
    for num_workers in [0, 2, 4, 8, 16, 32]:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        times = []
        point_counts = []

        print(f"\nTesting with num_workers={num_workers}")
        start = time.time()

        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            batch_time = time.time()
            times.append(batch_time - start)
            point_counts.append(batch[0]["num_points"] if len(batch) > 0 else 0)
            start = batch_time

        print(f"  Avg time per batch: {np.mean(times):.3f}s")
        print(f"  First batch time: {times[0]:.3f}s")
        print(f"  Subsequent batches: {np.mean(times[1:]):.3f}s")
        print(
            f"  Point counts: min={min(point_counts)}, max={max(point_counts)}, avg={np.mean(point_counts):.0f}"
        )


def profile_model_forward(device="cuda", batch_size=64):
    """Profile model forward pass with different point counts."""
    print("\n" + "=" * 80)
    print("PROFILING MODEL FORWARD PASS")
    print("=" * 80)

    model = PointNet2UnetForFlowMatching(time_embed_dim=256).to(device)
    model.eval()

    # Test different point cloud sizes
    point_counts = [256, 512, 1024, 2048, 4096]

    for num_points in point_counts:
        x = torch.randn(batch_size, 3, num_points, device=device)
        t = torch.rand(batch_size, device=device)

        # Warmup
        with torch.no_grad():
            _ = model(x, t)

        torch.cuda.synchronize()

        # Time multiple iterations
        times = []
        for _ in range(10):
            start = time.time()
            with torch.no_grad():
                _ = model(x, t)
            torch.cuda.synchronize()
            times.append(time.time() - start)

        print(
            f"  {num_points} points: {np.mean(times)*1000:.2f}ms ± {np.std(times)*1000:.2f}ms"
        )
        print(f"    Throughput: {batch_size / np.mean(times):.1f} samples/sec")


def profile_full_training_step(data_path, device="cuda", batch_size=64):
    """Profile a complete training iteration."""
    print("\n" + "=" * 80)
    print("PROFILING FULL TRAINING STEP")
    print("=" * 80)

    # Setup
    dataset = PointCloudDataset(
        Path(data_path),
        split="train",
        voxel_size=0.1,
        sample_exponent=0.5,
        rotation_augment=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = PointNet2UnetForFlowMatching(time_embed_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    flow_path = CondOTProbPath()

    model.train()

    # Profile 10 iterations
    times = {
        "data_loading": [],
        "to_device": [],
        "forward": [],
        "backward": [],
        "optimizer": [],
        "total": [],
    }

    data_iter = iter(loader)

    for i in range(10):
        iter_start = time.time()

        # Data loading
        load_start = time.time()
        batch = next(data_iter)
        times["data_loading"].append(time.time() - load_start)

        # Transfer to device
        device_start = time.time()
        sample = batch[0]
        points = sample["points"].to(device)
        points = points.unsqueeze(0).transpose(1, 2)
        torch.cuda.synchronize()
        times["to_device"].append(time.time() - device_start)

        # Forward pass
        optimizer.zero_grad()
        forward_start = time.time()

        batch_size_actual = points.shape[0]
        t = torch.rand(batch_size_actual, device=device)
        x_0 = torch.randn_like(points)
        path_sample = flow_path.sample(t=t, x_0=x_0, x_1=points)
        pred = model(path_sample.x_t, t)
        loss = torch.nn.functional.mse_loss(pred, path_sample.dx_t)

        torch.cuda.synchronize()
        times["forward"].append(time.time() - forward_start)

        # Backward pass
        backward_start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        times["backward"].append(time.time() - backward_start)

        # Optimizer step
        opt_start = time.time()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        torch.cuda.synchronize()
        times["optimizer"].append(time.time() - opt_start)

        times["total"].append(time.time() - iter_start)

        print(
            f"  Iteration {i+1}: {times['total'][-1]:.3f}s (pts={sample['num_points']})"
        )

    print("\n  Average times:")
    for key, values in times.items():
        print(
            f"    {key:15s}: {np.mean(values)*1000:.2f}ms ± {np.std(values)*1000:.2f}ms"
        )

    # Calculate percentages
    total_avg = np.mean(times["total"])
    print("\n  Time breakdown:")
    for key in ["data_loading", "to_device", "forward", "backward", "optimizer"]:
        pct = (np.mean(times[key]) / total_avg) * 100
        print(f"    {key:15s}: {pct:.1f}%")


def check_memory_usage(device="cuda", batch_size=64):
    """Check memory usage with different configurations."""
    print("\n" + "=" * 80)
    print("CHECKING MEMORY USAGE")
    print("=" * 80)

    model = PointNet2UnetForFlowMatching(time_embed_dim=256).to(device)

    point_counts = [256, 512, 1024, 2048, 4096]

    for num_points in point_counts:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        x = torch.randn(batch_size, 3, num_points, device=device)
        t = torch.rand(batch_size, device=device)

        # Forward pass
        with torch.no_grad():
            _ = model(x, t)

        allocated = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3

        print(f"  {num_points} points: allocated={allocated:.2f}GB, peak={peak:.2f}GB")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="FOR-species20K")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Run enhanced profiling - focus on data loading bottleneck
    profile_dataset_components(args.data_path, num_samples=100)
    profile_file_io_formats(args.data_path, num_samples=50)
    profile_voxelization_algorithms(args.data_path, num_samples=50)
    profile_with_different_configs(args.data_path, batch_size=args.batch_size)

    # Original profiling
    profile_data_loading(args.data_path, batch_size=args.batch_size, num_batches=10)
    profile_model_forward(device=device, batch_size=args.batch_size)
    check_memory_usage(device=device, batch_size=args.batch_size)
    profile_full_training_step(
        args.data_path, device=device, batch_size=args.batch_size
    )

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
    print("\nKEY INSIGHTS TO LOOK FOR:")
    print("  1. Which operation takes the most time in dataset loading?")
    print("  2. Is file I/O fast enough or is disk a bottleneck?")
    print("  3. How much does voxelization cost vs sampling vs rotation?")
    print("  4. What's the speedup potential from optimizing each component?")


if __name__ == "__main__":
    main()
