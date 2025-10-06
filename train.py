"""
train.py - Train a Flow Matching model on tree point clouds
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for multiprocessing safety

import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import argparse
from datetime import datetime

from model import PointNet2UnetForFlowMatching
from dataset import PointCloudDataset, collate_to_batch_min
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver


def compute_loss(model, x_1, flow_path, device):
    """
    Compute flow matching loss for a BATCH of point clouds.

    Args:
        model: Velocity prediction model
        x_1: Target point clouds (B, 3, N)
        flow_path: Flow matching path object
        device: torch device

    Returns:
        Scalar loss tensor
    """
    batch_size = x_1.shape[0]
    t = torch.rand(batch_size, device=device)

    # Sample source from standard normal
    x_0 = torch.randn_like(x_1)

    # Sample points along the flow path
    path_sample = flow_path.sample(t=t, x_0=x_0, x_1=x_1)
    x_t = path_sample.x_t
    u_t = path_sample.dx_t

    # Predict velocity
    pred_u_t = model(x_t, t)

    # MSE loss
    loss = nn.functional.mse_loss(pred_u_t, u_t)

    return loss


def train_epoch(model, train_loader, optimizer, flow_path, device, epoch):
    """Train for one epoch with batched processing."""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d} [Train]")

    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        # Get batched points - already sampled to same size
        points = batch['points'].to(device)  # (B, 3, min_points)

        # Single forward pass for entire batch
        loss = compute_loss(model, points, flow_path, device)
        loss.backward()

        # Gradient clipping and optimization step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'avg': f'{total_loss / (batch_idx + 1):.6f}',
            'pts': batch['num_points']
        })

        # Log first batch info
        if batch_idx == 0:
            print(f"\n  First batch: {batch['num_points']} points per sample")

    return total_loss / num_batches


@torch.no_grad()
def validate(model, val_loader, flow_path, device, epoch):
    """Validate the model with batched processing."""
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)

    pbar = tqdm(val_loader, desc=f"Epoch {epoch:3d} [Val]")

    for batch_idx, batch in enumerate(pbar):
        points = batch['points'].to(device)

        loss = compute_loss(model, points, flow_path, device)
        total_loss += loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'avg': f'{total_loss / (batch_idx + 1):.6f}',
            'pts': batch['num_points']
        })

    return total_loss / num_batches


@torch.no_grad()
def sample(model, num_points, device, method='dopri5'):
    """
    Generate a point cloud by solving the ODE from noise.

    Args:
        model: Velocity prediction model
        num_points: Number of points to generate
        device: torch device
        method: ODE solver method

    Returns:
        Generated points (N, 3) as numpy array
    """
    model.eval()

    # Start from noise
    x_init = torch.randn(1, 3, num_points, device=device)

    # Define ODE function
    def ode_fn(t, x):
        t_batch = torch.full((1,), t, device=device, dtype=x.dtype)
        return model(x, t_batch)

    # Solve ODE
    solver = ODESolver(velocity_model=ode_fn)
    x_final = solver.sample(x_init, method=method, step_size=None)

    return x_final[0].T.cpu().numpy()  # (3, N) -> (N, 3)


def visualize_point_cloud(points, title="Point Cloud", save_path=None):
    """
    Visualize a 3D point cloud.

    Args:
        points: Point cloud (N, 3) numpy array
        title: Plot title
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=points[:, 2], cmap='viridis', s=1, alpha=0.6)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)

    # Equal aspect ratio
    max_range = np.ptp(points, axis=0).max() / 2.0
    mid = points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_losses(train_losses, val_losses, save_path):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_output_dirs(base_dir='output_flow_matching'):
    """Create output directory structure."""
    output_dir = Path(base_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    vis_dir = output_dir / "visualizations"

    for d in [output_dir, checkpoint_dir, log_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return output_dir, checkpoint_dir, log_dir, vis_dir


def save_checkpoint(checkpoint_path, epoch, model, optimizer, scheduler,
                   train_loss, val_loss, args):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'args': vars(args),
    }
    torch.save(checkpoint, checkpoint_path)


def train(args):
    """Main training function."""
    # Setup directories
    output_dir, checkpoint_dir, log_dir, vis_dir = create_output_dirs()

    # Print configuration
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Data path:        {args.data_path}")
    print(f"Voxel size:       {args.voxel_size}")
    print(f"Augmentation:     {args.augment}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Epochs:           {args.num_epochs}")
    print(f"Learning rate:    {args.lr}")
    print(f"Device:           {'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'}")
    print("=" * 80 + "\n")

    # Load datasets
    print("Loading datasets...")
    train_dataset = PointCloudDataset(
        Path(args.data_path),
        split='train',
        voxel_size=args.voxel_size,
        augment=args.augment,
        sample_exponent=args.sample_exponent,
        rotation_augment=args.rotation_augment
    )
    test_dataset = PointCloudDataset(
        Path(args.data_path),
        split='test',
        voxel_size=args.voxel_size,
        augment=False
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset:  {len(test_dataset)} samples")

    # Sample statistics
    sample_sizes = [train_dataset[i]['num_points']
                   for i in range(min(100, len(train_dataset)))]
    print(f"Point cloud sizes (n={len(sample_sizes)}): "
          f"min={min(sample_sizes)}, max={max(sample_sizes)}, "
          f"mean={np.mean(sample_sizes):.0f}\n")

    # Create dataloaders with new collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=False,
        prefetch_factor=4 if args.num_workers > 0 else None,
        pin_memory=True,
        collate_fn=collate_to_batch_min,
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=False,
        prefetch_factor=4 if args.num_workers > 0 else None,
        pin_memory=True,
        collate_fn=collate_to_batch_min,
    )

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    # Initialize model
    print("Initializing model...")
    model = PointNet2UnetForFlowMatching(time_embed_dim=args.time_embed_dim).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M\n")

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.min_lr
    )

    # Setup flow matching
    flow_path = CondOTProbPath()

    # Training loop
    print("Starting training...")
    print("=" * 80 + "\n")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, flow_path, device, epoch)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, flow_path, device, epoch)
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print summary
        print(f"\nEpoch {epoch:3d}/{args.num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  LR:         {current_lr:.2e}")

        # Save best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            save_checkpoint(
                checkpoint_dir / 'best_model.pt',
                epoch, model, optimizer, scheduler,
                train_loss, val_loss, args
            )
            print(f"  *** New best model saved! ***")

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt',
                epoch, model, optimizer, scheduler,
                train_loss, val_loss, args
            )

        # Plot losses
        plot_losses(train_losses, val_losses, log_dir / 'losses.png')

        # Generate samples
        if epoch % args.visualize_every == 0:
            print("  Generating samples...")
            for num_pts in args.sample_sizes:
                try:
                    generated = sample(model, num_pts, device, method=args.ode_method)
                    visualize_point_cloud(
                        generated,
                        title=f"Generated Tree (Epoch {epoch}, {num_pts} points)",
                        save_path=vis_dir / f'generated_e{epoch:03d}_n{num_pts}.png'
                    )
                except Exception as e:
                    print(f"  Warning: Failed to generate {num_pts} points: {e}")

        print("=" * 80 + "\n")

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints:          {checkpoint_dir}")
    print(f"Visualizations:       {vis_dir}")
    print("=" * 80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Flow Matching model on tree point clouds',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument('--data_path', type=str, default='FOR-species20K',
                        help='Path to FOR-species20K dataset')
    parser.add_argument('--voxel_size', type=float, default=0.05,
                        help='Voxel size for downsampling (meters)')

    # Augmentation
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Enable data augmentation')
    parser.add_argument('--no_augment', action='store_false', dest='augment',
                        help='Disable data augmentation')
    parser.add_argument('--sample_exponent', type=float, default=0.5,
                        help='Power law exponent for point sampling')
    parser.add_argument('--rotation_augment', action='store_true', default=True,
                        help='Enable rotation augmentation')
    parser.add_argument('--no_rotation', action='store_false', dest='rotation_augment',
                        help='Disable rotation augmentation')

    # Model
    parser.add_argument('--time_embed_dim', type=int, default=256,
                        help='Dimension of time embedding')

    # Training
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')

    # Sampling
    parser.add_argument('--ode_method', type=str, default='dopri5',
                        choices=['dopri5', 'euler', 'midpoint', 'rk4'],
                        help='ODE solver method')
    parser.add_argument('--sample_sizes', type=int, nargs='+',
                        default=[100, 500, 2000, 8000],
                        help='Point cloud sizes to generate')

    # Logging
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--visualize_every', type=int, default=10,
                        help='Generate samples every N epochs')

    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()