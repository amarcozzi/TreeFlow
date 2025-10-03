# train_autoencoder.py
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for thread-safe plotting

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

from dataset import PointCloudDataset, collate_fn
from model_autoencoder import PerceiverIOAutoencoder


def set_seed(seed):
    """
    Set random seed for reproducibility across random, numpy, and pytorch.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def chamfer_distance(pred, target):
    """
    Compute Chamfer Distance between two point clouds.

    Args:
        pred: Predicted points, shape (N, 3)
        target: Target points, shape (M, 3)

    Returns:
        loss: Chamfer distance
    """
    # Compute pairwise distances
    dist_matrix = torch.cdist(pred.unsqueeze(0), target.unsqueeze(0)).squeeze(0)  # (N, M)

    # Forward direction: for each predicted point, find nearest target
    min_dist_pred_to_target = dist_matrix.min(dim=1)[0]  # (N,)

    # Backward direction: for each target point, find nearest prediction
    min_dist_target_to_pred = dist_matrix.min(dim=0)[0]  # (M,)

    # Chamfer distance is sum of both directions
    chamfer_loss = min_dist_pred_to_target.mean() + min_dist_target_to_pred.mean()

    return chamfer_loss


def sample_points(points, num_samples):
    """
    Randomly sample a fixed number of points from a point cloud.

    Args:
        points: Input points, shape (N, 3)
        num_samples: Number of points to sample

    Returns:
        sampled_points: shape (num_samples, 3)
    """
    num_points = points.size(0)

    if num_points >= num_samples:
        # Randomly sample without replacement
        indices = torch.randperm(num_points)[:num_samples]
        return points[indices]
    else:
        # If not enough points, sample with replacement
        indices = torch.randint(0, num_points, (num_samples,))
        return points[indices]


def save_visualization(epoch, output_dir, ground_truth, reconstruction, file_id, prefix='reconstruction'):
    """
    Save side-by-side visualization of ground truth and reconstruction.

    Args:
        epoch: Current epoch number
        output_dir: Directory to save visualizations
        ground_truth: Ground truth point cloud, shape (N, 3)
        reconstruction: Reconstructed point cloud, shape (M, 3)
        file_id: File identifier
        prefix: Prefix for saved filename
    """
    fig = plt.figure(figsize=(18, 6))

    # Convert to numpy if needed
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.cpu().numpy()
    if torch.is_tensor(reconstruction):
        reconstruction = reconstruction.cpu().numpy()

    # Compute elevation (Z) for coloring
    gt_colors = ground_truth[:, 2]
    rec_colors = reconstruction[:, 2]

    # Use oblique view to show 3D structure
    view_elev = 25  # Look down at 25 degrees
    view_azim = 45  # Diagonal view

    # Ground Truth
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
                           c=gt_colors, cmap='viridis', s=2, alpha=0.3)
    ax1.set_title(f'Ground Truth\n{file_id}\nPoints: {len(ground_truth)}', fontsize=10)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=view_elev, azim=view_azim)

    # Reconstruction
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(reconstruction[:, 0], reconstruction[:, 1], reconstruction[:, 2],
                           c=rec_colors, cmap='viridis', s=2, alpha=0.3)
    ax2.set_title(f'Reconstruction\nPoints: {len(reconstruction)}', fontsize=10)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=view_elev, azim=view_azim)

    # Overlay comparison
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
                c='blue', s=2, alpha=0.2, label='Ground Truth')
    ax3.scatter(reconstruction[:, 0], reconstruction[:, 1], reconstruction[:, 2],
                c='red', s=2, alpha=0.3, label='Reconstruction')
    ax3.set_title(f'Overlay Comparison', fontsize=10)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend(markerscale=5)
    ax3.view_init(elev=view_elev, azim=view_azim)

    # Set consistent axis limits and aspect ratios
    x_lim = (min(ground_truth[:, 0].min(), reconstruction[:, 0].min()),
             max(ground_truth[:, 0].max(), reconstruction[:, 0].max()))
    y_lim = (min(ground_truth[:, 1].min(), reconstruction[:, 1].min()),
             max(ground_truth[:, 1].max(), reconstruction[:, 1].max()))
    z_lim = (min(ground_truth[:, 2].min(), reconstruction[:, 2].min()),
             max(ground_truth[:, 2].max(), reconstruction[:, 2].max()))

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        # Force equal aspect ratio to see true proportions
        ax.set_box_aspect([x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]])

    # Add colorbars
    fig.colorbar(scatter1, ax=ax1, label='Z elevation', shrink=0.5, pad=0.1)
    fig.colorbar(scatter2, ax=ax2, label='Z elevation', shrink=0.5, pad=0.1)

    plt.suptitle(f'Epoch {epoch}', fontsize=12, y=0.98)
    plt.tight_layout()

    filepath = output_dir / f'{prefix}_epoch_{epoch:03d}_{file_id}.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def train_epoch(model, dataloader, optimizer, device, epoch, min_output_points, max_output_points):
    """
    Train for one epoch with variable-length generation.

    Args:
        min_output_points: Minimum number of points to generate during training
        max_output_points: Maximum number of points to generate during training
    """
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        batch_loss = 0
        for sample in batch:
            points = sample['points'].to(device)

            # Sample variable number of target points for teacher forcing
            # This teaches the model to generate different resolutions
            # num_target = random.randint(min_output_points,
            #                             min(max_output_points, len(points)))
            actual_min = min(min_output_points, len(points))
            actual_max = min(max_output_points, len(points))
            num_target = random.randint(actual_min, actual_max)
            target_points = sample_points(points, num_target)

            # Forward pass with teacher forcing (model.train() mode)
            # When target_points is provided in training mode, the model uses teacher forcing
            # and generates exactly len(target_points) points regardless of num_points arg
            reconstructed, latent = model(points, target_points=target_points)

            # Compute loss against sampled target
            loss = chamfer_distance(reconstructed, target_points)

            (loss / len(batch)).backward()  # Accumulate gradients
            batch_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        batch_loss /= len(batch)
        total_loss += batch_loss

        pbar.set_postfix({
            'loss': f'{batch_loss:.6f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.6f}'
        })

    return total_loss / num_batches


def validate(model, dataloader, device, num_output_points, epoch=None, vis_dir=None,
             num_visualizations=5):
    """
    Validate the model using true autoregressive generation.

    Note: model.eval() forces autoregressive generation (no teacher forcing),
    so we always test true generation quality during validation.

    Args:
        num_output_points: Number of points to generate during validation
    """
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)

    visualized_count = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]" if epoch else "Validating")

        for batch_idx, batch in enumerate(pbar):
            batch_loss = 0

            for sample in batch:
                points = sample['points'].to(device)

                # Determine how many points to generate (limit to available points)
                actual_num_points = min(num_output_points, len(points))

                # Always use autoregressive generation (model.eval() enforces this)
                # Explicitly pass num_points, no target_points needed
                reconstructed, latent = model(points, num_points=actual_num_points, target_points=None)

                # Compute loss against full point cloud for consistent evaluation
                loss = chamfer_distance(reconstructed, points)
                batch_loss += loss

            batch_loss = batch_loss / len(batch)
            total_loss += batch_loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            pbar.set_postfix({
                'loss': f'{batch_loss.item():.6f}',
                'avg_loss': f'{avg_loss:.6f}'
            })

            # Save visualizations for first few batches
            if vis_dir is not None and visualized_count < num_visualizations:
                for sample in batch:
                    if visualized_count >= num_visualizations:
                        break

                    points = sample['points'].to(device)
                    actual_num_points = min(num_output_points, len(points))

                    # Generate using autoregressive (model.eval() enforces this)
                    reconstructed, _ = model(points, num_points=actual_num_points, target_points=None)

                    save_visualization(
                        epoch=epoch if epoch is not None else 0,
                        output_dir=vis_dir,
                        ground_truth=points.cpu(),
                        reconstruction=reconstructed.cpu(),
                        file_id=sample['file_id'],
                        prefix='val'
                    )
                    visualized_count += 1

    return total_loss / num_batches


def main(args):
    # Set random seed for reproducibility
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        set_seed(args.seed)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = checkpoint_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)

    # Create datasets
    data_path = Path(args.data_path)
    train_dataset = PointCloudDataset(data_path, split="train", voxel_size=args.voxel_size)
    test_dataset = PointCloudDataset(data_path, split="test", voxel_size=args.voxel_size)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Check if resuming from checkpoint
    resume_checkpoint = None
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\nLoading checkpoint from: {resume_path}")
            resume_checkpoint = torch.load(resume_path, map_location=device)
            print(f"Resuming from epoch {resume_checkpoint['epoch']}")
            print(f"Previous train loss: {resume_checkpoint['train_loss']:.6f}")
            print(f"Previous val loss: {resume_checkpoint['val_loss']:.6f}")
        else:
            print(f"\nWarning: Checkpoint not found at {resume_path}")
            print("Starting training from scratch...")
            resume_checkpoint = None

    # Create dataloaders
    print("Create train loader with shuffle=True")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True if args.num_workers > 0 else False
    )
    print(f"Create test loader with shuffle={args.shuffle_test}")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle_test,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=collate_fn
    )

    # Create model
    model = PerceiverIOAutoencoder(
        input_dim=3,
        latent_dim=args.latent_dim,
        num_latents=args.num_latents,
        num_encoder_layers=args.num_encoder_layers,
        num_processor_layers=args.num_processor_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_output_points=args.max_output_points
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    print(f"Decoder type: autoregressive")
    print(f"Max output points: {args.max_output_points}")
    print(f"Training with variable output: {args.min_output_points}-{args.max_output_points} points")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # Load checkpoint if resuming
    start_epoch = 1
    best_val_loss = float('inf')

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint['model_state_dict'])
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        start_epoch = resume_checkpoint['epoch'] + 1
        best_val_loss = resume_checkpoint['val_loss']
        print(f"Resumed successfully! Starting from epoch {start_epoch}")
        print(f"Best validation loss so far: {best_val_loss:.6f}\n")

    # Training loop
    print("\nStarting training...")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs + 1):
        # Train with variable-length generation
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch,
            args.min_output_points, args.max_output_points
        )

        # Validate using true autoregressive generation
        val_loss = validate(
            model, test_loader, device, args.val_output_points, epoch,
            vis_dir=vis_dir if epoch % args.vis_freq == 0 else None,
            num_visualizations=args.num_vis
        )

        # Step scheduler
        scheduler.step()

        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
            print(f"  *** New best model saved! ***")

        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth')

        print("=" * 80)

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PerceiverIO Autoencoder for Point Clouds')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=968324,
                        help='Random seed for reproducibility (default: None, no seed set)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., ./checkpoints/best_model.pth)')

    # Data
    parser.add_argument('--data_path', type=str, default='./FOR-species20K',
                        help='Path to FOR-species20K dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (number of point clouds per batch)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--voxel_size', type=float, default=0.5,
                        help='Voxel size in meters for downsampling (default: 0.5). Set to 0 or None for no voxelization')
    parser.add_argument('--shuffle_test', action='store_true',
                        help='Shuffle test dataloader (default: False)')

    # Model
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Dimension of latent space')
    parser.add_argument('--num_latents', type=int, default=64,
                        help='Number of latent vectors')
    parser.add_argument('--num_encoder_layers', type=int, default=2,
                        help='Number of encoder layers')
    parser.add_argument('--num_processor_layers', type=int, default=2,
                        help='Number of processor layers')
    parser.add_argument('--num_decoder_layers', type=int, default=2,
                        help='Number of decoder layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Decoder-specific (VARIABLE LENGTH GENERATION)
    parser.add_argument('--max_output_points', type=int, default=4196,
                        help='Maximum number of output points decoder can generate')
    parser.add_argument('--min_output_points', type=int, default=256,
                        help='Minimum number of output points during training (for variable-length training)')
    parser.add_argument('--val_output_points', type=int, default=2048,
                        help='Number of points to generate during validation')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')

    # Logging and visualization
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--vis_freq', type=int, default=1,
                        help='Visualize results every N epochs')
    parser.add_argument('--num_vis', type=int, default=10,
                        help='Number of visualizations to save per validation')

    args = parser.parse_args()

    main(args)