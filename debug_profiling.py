"""
debug_profiling.py - Profile and diagnose training bottlenecks
"""
import torch
import time
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import PointNet2UnetForFlowMatching
from dataset import PointCloudDataset, collate_to_batch_min
from flow_matching.path import CondOTProbPath


def profile_data_loading(data_path, batch_size=64, num_batches=10):
    """Profile data loading speed."""
    print("\n" + "="*80)
    print("PROFILING DATA LOADING")
    print("="*80)
    
    dataset = PointCloudDataset(
        Path(data_path),
        split='train',
        voxel_size=0.1,
        augment=True,
        sample_exponent=0.5,
        rotation_augment=True
    )
    
    # Test different worker configurations
    for num_workers in [0, 2, 4, 8]:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
            pin_memory=True,
            collate_fn=collate_to_batch_min,
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
            point_counts.append(batch['num_points'])
            start = batch_time
        
        print(f"  Avg time per batch: {np.mean(times):.3f}s")
        print(f"  First batch time: {times[0]:.3f}s")
        print(f"  Subsequent batches: {np.mean(times[1:]):.3f}s")
        print(f"  Point counts: min={min(point_counts)}, max={max(point_counts)}, avg={np.mean(point_counts):.0f}")


def profile_model_forward(device='cuda', batch_size=64):
    """Profile model forward pass with different point counts."""
    print("\n" + "="*80)
    print("PROFILING MODEL FORWARD PASS")
    print("="*80)
    
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
        
        print(f"  {num_points} points: {np.mean(times)*1000:.2f}ms ± {np.std(times)*1000:.2f}ms")
        print(f"    Throughput: {batch_size / np.mean(times):.1f} samples/sec")


def profile_full_training_step(data_path, device='cuda', batch_size=64):
    """Profile a complete training iteration."""
    print("\n" + "="*80)
    print("PROFILING FULL TRAINING STEP")
    print("="*80)
    
    # Setup
    dataset = PointCloudDataset(
        Path(data_path),
        split='train',
        voxel_size=0.1,
        augment=True,
        sample_exponent=0.5,
        rotation_augment=True
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=True,
        collate_fn=collate_to_batch_min,
    )
    
    model = PointNet2UnetForFlowMatching(time_embed_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    flow_path = CondOTProbPath()
    
    model.train()
    
    # Profile 10 iterations
    times = {
        'data_loading': [],
        'to_device': [],
        'forward': [],
        'backward': [],
        'optimizer': [],
        'total': []
    }
    
    data_iter = iter(loader)
    
    for i in range(10):
        iter_start = time.time()
        
        # Data loading
        load_start = time.time()
        batch = next(data_iter)
        times['data_loading'].append(time.time() - load_start)
        
        # Transfer to device
        device_start = time.time()
        points = batch['points'].to(device)
        torch.cuda.synchronize()
        times['to_device'].append(time.time() - device_start)
        
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
        times['forward'].append(time.time() - forward_start)
        
        # Backward pass
        backward_start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        times['backward'].append(time.time() - backward_start)
        
        # Optimizer step
        opt_start = time.time()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        torch.cuda.synchronize()
        times['optimizer'].append(time.time() - opt_start)
        
        times['total'].append(time.time() - iter_start)
        
        print(f"  Iteration {i+1}: {times['total'][-1]:.3f}s (pts={batch['num_points']})")
    
    print("\n  Average times:")
    for key, values in times.items():
        print(f"    {key:15s}: {np.mean(values)*1000:.2f}ms ± {np.std(values)*1000:.2f}ms")
    
    # Calculate percentages
    total_avg = np.mean(times['total'])
    print("\n  Time breakdown:")
    for key in ['data_loading', 'to_device', 'forward', 'backward', 'optimizer']:
        pct = (np.mean(times[key]) / total_avg) * 100
        print(f"    {key:15s}: {pct:.1f}%")


def check_memory_usage(device='cuda', batch_size=64):
    """Check memory usage with different configurations."""
    print("\n" + "="*80)
    print("CHECKING MEMORY USAGE")
    print("="*80)
    
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
    parser.add_argument('--data_path', type=str, default='FOR-species20K')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Run profiling
    profile_data_loading(args.data_path, batch_size=args.batch_size, num_batches=10)
    profile_model_forward(device=device, batch_size=args.batch_size)
    check_memory_usage(device=device, batch_size=args.batch_size)
    profile_full_training_step(args.data_path, device=device, batch_size=args.batch_size)
    
    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
