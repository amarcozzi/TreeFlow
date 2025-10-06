"""
train.py

Train a Flow Matching model on tree point clouds from the FOR-species20K dataset.
"""

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

from model import PointNet2UnetForFlowMatching
from dataset import PointCloudDataset, collate_fn
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver


def compute_loss(model, x_1, flow_path, device):
    """
    Compute flow matching loss for a single point cloud.

    Args:
        model: The velocity prediction model
        x_1: (1, 3, N) - target point cloud
        flow_path: Flow matching path object
        device: torch device

    Returns:
        loss: scalar tensor
    """

    # Sample time uniformly from [0, 1]
    batch_size = x_1.shape[0]
    t = torch.rand(batch_size, device=device)

    # Sample source (noise) from standard normal
    x_0 = torch.randn_like(x_1)

    # Sample points along the flow path
    path_sample = flow_path.sample(t=t, x_0=x_0, x_1=x_1)
    x_t = path_sample.x_t
    u_t = path_sample.dx_t

    # Predict velocity with model
    pred_u_t = model(x_t, t)

    # MSE loss between predicted and target velocity
    loss = nn.functional.mse_loss(pred_u_t, u_t)

    return loss


def train_epoch(model, train_loader, optimizer, flow_path, device, accumulation_steps):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="Training", dynamic_ncols=True, disable=False)
    for i, batch in enumerate(pbar):
        # batch is a list of dicts (length 1 since batch_size=1)
        sample = batch[0]
        points = sample['points']  # (N, 3)
        num_points = sample['num_points']

        # Add batch dimension and transpose: (N, 3) -> (1, N, 3) -> (1, 3, N)
        points = points.unsqueeze(0).transpose(1, 2).to(device)

        # Compute loss
        loss = compute_loss(model, points, flow_path, device)

        # Normalize loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Accumulate gradients
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}', 'pts': num_points})

    # Final step if there are remaining gradients
    if num_batches % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / num_batches


@torch.no_grad()
def validate(model, val_loader, flow_path, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(val_loader, desc="Validation", dynamic_ncols=True, disable=False):
        sample = batch[0]
        points = sample['points']

        # Add batch dimension and transpose
        points = points.unsqueeze(0).transpose(1, 2).to(device)

        loss = compute_loss(model, points, flow_path, device)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def sample(model, num_points, device, method='dopri5'):
    """
    Generate a point cloud by solving the ODE from noise using a sophisticated solver.

    Args:
        model: The velocity prediction model
        num_points: Number of points to generate
        device: torch device
        method: ODE solver method ('dopri5', 'euler', 'midpoint', etc.)

    Returns:
        points: (N, 3) numpy array
    """
    model.eval()

    # Start from noise
    x_init = torch.randn(1, 3, num_points, device=device)

    # Define the ODE function: dx/dt = v(x, t)
    def ode_fn(t, x):
        # t is a scalar, x is (1, 3, N)
        t_batch = torch.full((1,), t, device=device, dtype=x.dtype)
        return model(x, t_batch)

    # 1. Initialize the solver with the velocity function (the model)
    solver = ODESolver(velocity_model=ode_fn)

    # 2. Call .sample() with the starting tensor and the method name.
    # The solver from your other project returns the final state directly, not the full trajectory.
    x_final = solver.sample(x_init, method=method, step_size=None)

    return x_final[0].T.cpu().numpy()  # (3, N) -> (N, 3)


def visualize_point_cloud(points, title="Point Cloud", save_path=None):
    """
    Visualize a 3D point cloud.

    Args:
        points: (N, 3) numpy array
        title: Plot title
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=points[:, 2], cmap='viridis', s=1, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Equal aspect ratio
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

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_losses(train_losses, val_losses, save_path):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def train(args):
    """Main training function."""

    # Setup directories
    output_dir = Path('output_flow_matching')
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    print(f"Using device: {device}")
    print(f"Voxel size: {args.voxel_size}")
    print(f"Augmentation: {args.augment}")

    # Load datasets
    print("\nLoading datasets...")
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
        augment=False  # No augmentation for test set
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")

    # Check point cloud size distribution
    train_sizes = [train_dataset[i]['num_points'] for i in range(min(100, len(train_dataset)))]
    print(f"Point cloud sizes (sample of {len(train_sizes)}): "
          f"min={min(train_sizes)}, max={max(train_sizes)}, mean={np.mean(train_sizes):.0f}")

    # Only pin memory when using CUDA
    pin_memory = device.type == 'cuda'

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        prefetch_factor=4,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        prefetch_factor=4,
        pin_memory=pin_memory
    )

    # Initialize model
    print("\nInitializing model...")
    model = PointNet2UnetForFlowMatching(time_embed_dim=args.time_embed_dim).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.min_lr)

    # Setup flow matching
    flow_path = CondOTProbPath()

    # Training loop
    print("\nStarting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'=' * 60}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, flow_path, device, args.accumulation_steps)
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.6f}")

        # Validate
        # val_loss = validate(model, val_loader, flow_path, device)
        # val_losses.append(val_loss)
        # print(f"Val Loss: {val_loss:.6f}")

        # Update learning rate
        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6e}")

        # # Save checkpoint
        # is_best = val_loss < best_val_loss
        # if is_best:
        #     best_val_loss = val_loss
        #     print(f"New best model! Val Loss: {best_val_loss:.6f}")

        # if epoch % args.save_every == 0 or is_best:
        #     checkpoint = {
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'train_losses': train_losses,
        #         'val_losses': val_losses,
        #         'args': vars(args),
        #     }
        #
        #     torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
        #
        #     if is_best:
        #         torch.save(checkpoint, checkpoint_dir / 'best_model.pt')

        # Plot losses
        plot_losses(train_losses, val_losses, log_dir / 'losses.png')

        # Generate and visualize samples
        if epoch % args.visualize_every == 0:
            print("Generating samples...")
            # Sample different sizes to see how model handles it
            for i, num_pts in enumerate(args.sample_sizes):
                generated = sample(model, num_pts, device, method=args.ode_method)
                visualize_point_cloud(
                    generated,
                    title=f"Generated Tree (Epoch {epoch}, {num_pts} points)",
                    save_path=vis_dir / f'generated_epoch_{epoch}_size_{num_pts}.png'
                )

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print(f"Visualizations saved in: {vis_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Flow Matching model on tree point clouds')

    # Data arguments
    parser.add_argument('--data_path', type=str, default='FOR-species20K',
                        help='Path to FOR-species20K dataset')
    parser.add_argument('--voxel_size', type=float, default=0.25,
                        help='Voxel size for downsampling (None for no downsampling)')

    # Augmentation arguments
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Enable data augmentation')
    parser.add_argument('--no_augment', action='store_false', dest='augment',
                        help='Disable data augmentation')
    parser.add_argument('--sample_exponent', type=float, default=0.5,
                        help='Exponent for power law point sampling (lower=more skew toward full count, 0.5=moderate, 0.3=aggressive)')
    parser.add_argument('--rotation_augment', action='store_true', default=True,
                        help='Enable rotation augmentation')
    parser.add_argument('--no_rotation', action='store_false', dest='rotation_augment',
                        help='Disable rotation augmentation')

    # Model arguments
    parser.add_argument('--time_embed_dim', type=int, default=256,
                        help='Dimension of time embedding')

    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--accumulation_steps', type=int, default=32,
                        help='Gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')

    # Sampling arguments
    parser.add_argument('--ode_method', type=str, default='dopri5',
                        choices=['dopri5', 'euler', 'midpoint', 'rk4'],
                        help='ODE solver method')
    parser.add_argument('--sample_sizes', type=int, nargs='+', default=[100, 250, 500, 1000, 2000, 4000],
                        help='Point cloud sizes to generate during visualization')

    # Logging arguments
    parser.add_argument('--save_every', type=int, default=20,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--visualize_every', type=int, default=1,
                        help='Generate and visualize samples every N epochs')

    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()