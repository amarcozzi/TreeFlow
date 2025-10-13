"""
train.py

Train a Flow Matching model on tree point clouds from the FOR-species20K dataset.
"""

import matplotlib
matplotlib.use('Agg')

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
from dataset import PointCloudDataset, collate_fn, collate_fn_batched
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver


def compute_loss(model, x_1, flow_path, device):
    """Compute flow matching loss for a single point cloud."""
    batch_size = x_1.shape[0]
    t = torch.rand(batch_size, device=device)
    x_0 = torch.randn_like(x_1)

    path_sample = flow_path.sample(t=t, x_0=x_0, x_1=x_1)
    x_t = path_sample.x_t
    u_t = path_sample.dx_t

    pred_u_t = model(x_t, t)
    loss = nn.functional.mse_loss(pred_u_t, u_t)

    return loss


def train_epoch(model, train_loader, optimizer, flow_path, device, batch_mode):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_samples = 0

    pbar = tqdm(total=len(train_loader), desc="Training", dynamic_ncols=True)

    for batch in train_loader:
        optimizer.zero_grad()

        if batch_mode == 'accumulate':
            # Original mode: accumulate gradients for each sample
            for sample in batch:
                points = sample['points']
                points = points.unsqueeze(0).transpose(1, 2).to(device)

                loss = compute_loss(model, points, flow_path, device)

                # Check for NaN loss during training
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nWarning: NaN/Inf loss detected, skipping batch")
                    continue

                normalized_loss = loss / len(batch)
                normalized_loss.backward()

                total_loss += loss.item()
                num_samples += 1

        elif batch_mode == 'sample_to_min':
            # Process entire batch at once
            points = batch['points'].to(device)  # (B, N_min, 3)
            points = points.transpose(1, 2)  # (B, 3, N_min)

            loss = compute_loss(model, points, flow_path, device)

            # Check for NaN loss during training
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWarning: NaN/Inf loss detected, skipping batch")
                pbar.update(1)
                continue

            loss.backward()

            total_loss += loss.item() * points.shape[0]  # Scale by batch size
            num_samples += points.shape[0]

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pbar.update(1)
        if num_samples > 0:
            avg_loss = total_loss / num_samples
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

    pbar.close()
    return total_loss / num_samples if num_samples > 0 else 0.0


@torch.no_grad()
def sample(model, num_points, device, method='euler', num_steps=100):
    """
    Generate a point cloud using ODE integration.

    Args:
        model: The velocity prediction model
        num_points: Number of points to generate
        device: torch device
        method: ODE solver method
        num_steps: Number of integration steps (determines step_size = 1.0/num_steps)

    Returns:
        points: (N, 3) numpy array (may contain NaN/Inf which will be filtered during visualization)
    """
    model.eval()

    # Start from noise - scale to reasonable range for normalized data
    # Using std ~ 0.5 keeps most points within [-1.5, 1.5] initially
    x_init = torch.randn(1, 3, num_points, device=device) * 0.5

    # Define the ODE function
    def ode_fn(t, x):
        t_batch = torch.full((1,), t, device=device, dtype=x.dtype)
        return model(x, t_batch)

    try:
        # Initialize solver
        solver = ODESolver(velocity_model=ode_fn)

        if num_steps:
            step_size = 1.0 / num_steps
        else:
            step_size = None  # Let solver choose adaptive step size

        # Sample using the specified method
        x_final = solver.sample(x_init, method=method, step_size=step_size)

        # Convert to numpy
        points = x_final[0].T.cpu().numpy()  # (3, N) -> (N, 3)

        return points

    except Exception as e:
        print(f"\nError during sampling: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_point_cloud(points, title="Point Cloud", save_path=None):
    """
    Visualize a 3D point cloud with NaN/Inf filtering.

    Args:
        points: (N, 3) numpy array of points
        title: Plot title
        save_path: Path to save figure
    """
    if points is None:
        print(f"Skipping visualization: {title} - No valid points")
        return

    original_count = len(points)

    # Filter out NaN/Inf points
    valid_mask = ~(np.isnan(points).any(axis=1) | np.isinf(points).any(axis=1))
    points = points[valid_mask]

    filtered_count = original_count - len(points)
    if filtered_count > 0:
        print(f"{title}: Filtered out {filtered_count}/{original_count} points with NaN/Inf values")

    # Check if we have enough valid points left
    if len(points) < 10:
        print(f"Skipping visualization: {title} - Too few valid points ({len(points)}/10 minimum)")
        return

    # Check if all points are identical (degenerate case)
    if len(points) > 1:
        point_range = points.max(axis=0) - points.min(axis=0)
        if np.allclose(point_range, 0, atol=1e-6):
            print(f"Skipping visualization: {title} - Degenerate point cloud (all points identical)")
            return

    try:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=points[:, 2], cmap='viridis', s=1, alpha=0.6)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Add filtered count to title if any were removed
        if filtered_count > 0:
            title_with_info = f"{title}\n({len(points)}/{original_count} valid points)"
        else:
            title_with_info = title
        ax.set_title(title_with_info)

        # Equal aspect ratio with safety checks
        ranges = np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ])

        max_range = ranges.max() / 2.0

        # Avoid zero range
        if max_range < 1e-6:
            max_range = 1.0

        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if filtered_count == 0:
                print(f"✓ Saved visualization: {save_path.name}")
        plt.close()

    except Exception as e:
        print(f"Error during visualization of {title}: {e}")
        plt.close('all')  # Clean up any partial plots


def plot_losses(train_losses, val_losses, save_path):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
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

    print(f"Using preprocessed version: {args.preprocessed_version}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = PointCloudDataset(
        Path(args.data_path),
        split='mixed',
        preprocessed_version=args.preprocessed_version,
        sample_exponent=args.sample_exponent,
        rotation_augment=args.rotation_augment,
        max_points=args.max_points
    )
    print(f"Loaded training dataset\n"
          f" - Number of samples: {len(train_dataset)}\n"
          f" - Preprocessed version: {args.preprocessed_version}\n"
          f" - Sample exponent: {args.sample_exponent}\n"
          f" - Rotation augment: {args.rotation_augment}"
          f" - Max points: {args.max_points}"
          )

    print(f"Train dataset: {len(train_dataset)} samples")

    # Check point cloud size distribution
    train_sizes = [train_dataset[i]['num_points'] for i in range(min(100, len(train_dataset)))]
    print(f"Point cloud sizes (sample of {len(train_sizes)}): "
          f"min={min(train_sizes)}, max={max(train_sizes)}, mean={np.mean(train_sizes):.0f}")

    # Select a collate function based on batch mode
    if args.batch_mode == 'accumulate':
        collate_function = collate_fn
        print(f"Batch mode: accumulate (process each sample individually, full resolution)")
    elif args.batch_mode == 'sample_to_min':
        collate_function = collate_fn_batched
        print(f"Batch mode: sample_to_min (batch processing, samples to minimum size)")
    else:
        raise ValueError(f"Invalid batch mode: {args.batch_mode}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.num_workers - 1 if args.num_workers > 0 else None,
        pin_memory=True,
        collate_fn=collate_function,
    )

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

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
    best_train_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'=' * 60}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, flow_path, device, args.batch_mode)
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.6f}")

        # Update learning rate
        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6e}")

        # Save checkpoint
        is_best = train_loss < best_train_loss
        if is_best:
            best_train_loss = train_loss
            print(f"New best model! Train Loss: {best_train_loss:.6f}")

        if epoch % args.save_every == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'args': vars(args),
            }

            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')

            if is_best:
                torch.save(checkpoint, checkpoint_dir / 'best_model.pt')

        # Plot losses
        plot_losses(train_losses, val_losses, log_dir / 'losses.png')

        # Generate and visualize samples
        if epoch % args.visualize_every == 0:
            print("Generating samples...")
            # Sample different sizes to see how the model handles it
            # sample_size = np.random.randint(1000, 5000)
            pbar = tqdm(total=args.num_visualizations, desc="Sampling", dynamic_ncols=True)
            for _ in range(args.num_visualizations):
                try:
                    num_pts = np.random.randint(1000, 45000)
                    pbar.set_description(f"Sampling {num_pts} points")
                    generated = sample(
                        model,
                        num_pts,
                        device,
                        method=args.ode_method,
                        num_steps=args.ode_steps
                    )
                    visualize_point_cloud(
                        generated,
                        title=f"Generated Tree (Epoch {epoch}, {num_pts} points)",
                        save_path=vis_dir / f'generated_epoch_{epoch}_size_{num_pts}.png'
                    )
                except Exception as e:
                    print(f"Error during sampling/visualization: {e}")
                pbar.update(1)

            pbar.close()

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best training loss: {best_train_loss:.6f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print(f"Visualizations saved in: {vis_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Flow Matching model on tree point clouds')

    # Data arguments
    parser.add_argument('--data_path', type=str, default='FOR-species20K')
    parser.add_argument('--preprocessed_version', type=str, default='voxel_0.1m')

    # Augmentation arguments
    parser.add_argument('--sample_exponent', type=float, default=None)
    parser.add_argument('--rotation_augment', action='store_true', default=False)

    # Model arguments
    parser.add_argument('--time_embed_dim', type=int, default=256)

    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_mode', type=str, default='accumulate',
                        choices=['accumulate', 'sample_to_min'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_cuda', action='store_true')

    # Sampling arguments
    parser.add_argument('--ode_method', type=str, default='dopri5',
                        choices=['euler', 'midpoint', 'rk4', 'dopri5'])
    parser.add_argument('--ode_steps', type=int, default=None,
                        help='Number of ODE integration steps (step_size = 1.0/ode_steps)')
    parser.add_argument('--max_points', type=int, default=None)

    # Logging arguments
    parser.add_argument('--save_every', type=int, default=20)
    parser.add_argument('--visualize_every', type=int, default=10)
    parser.add_argument('--num_visualizations', type=int, default=4,)

    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()