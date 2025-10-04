# train_generative_perceiver.py
import matplotlib

matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

from dataset import PointCloudDataset, collate_fn
from model_autoencoder import GenerativePerceiver


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def chamfer_distance(pred, target):
    """
    Chamfer distance - permutation invariant spatial similarity.
    This is better than MSE for point clouds.
    """
    # Pairwise distances
    dist_matrix = torch.cdist(pred.unsqueeze(0), target.unsqueeze(0)).squeeze(0)

    # Forward: pred -> target
    min_dist_pred_to_target = dist_matrix.min(dim=1)[0]
    # Backward: target -> pred
    min_dist_target_to_pred = dist_matrix.min(dim=0)[0]

    chamfer = min_dist_pred_to_target.mean() + min_dist_target_to_pred.mean()
    return chamfer


def compute_losses(pred_coords, stop_logits, target_points, stop_loss_weight=1.0):
    """
    Compute combined loss with Chamfer distance and stop prediction.

    Args:
        pred_coords: (N, 3) predicted coordinates
        stop_logits: (N,) stop prediction logits
        target_points: (N, 3) ground truth sorted points
        stop_loss_weight: Weight for stop loss

    Returns:
        total_loss, coord_loss, stop_loss
    """
    # Coordinate loss: Chamfer distance (spatially aware, permutation invariant)
    coord_loss = chamfer_distance(pred_coords, target_points)

    # Stop loss: predict 0 for all positions except last = 1
    stop_targets = torch.zeros_like(stop_logits)
    stop_targets[-1] = 1.0
    stop_loss = F.binary_cross_entropy_with_logits(stop_logits, stop_targets)

    total_loss = coord_loss + stop_loss_weight * stop_loss

    return total_loss, coord_loss, stop_loss


def get_curriculum_max_length(epoch, max_size):
    """
    Curriculum learning: start with shorter sequences, gradually increase.
    This helps stabilize training.
    """
    if epoch <= 5:
        return max_size * 0.1  # 10% for first 5 epochs
    elif epoch <= 10:
        return max_size * 0.25  # 25% for next 10 epochs
    elif epoch <= 15:
        return max_size * 0.5  # 50% for next 15 epochs
    elif epoch <= 20:
        return max_size * 0.75  # 75% for next 20 epochs
    else:
        return max_size  # Full length afterwards


def save_visualization(epoch, output_dir, ground_truth, reconstruction, file_id, num_generated, prefix='val'):
    """Save visualization comparing ground truth and generated point clouds."""
    fig = plt.figure(figsize=(18, 6))

    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.cpu().numpy()
    if torch.is_tensor(reconstruction):
        reconstruction = reconstruction.cpu().numpy()

    view_elev, view_azim = 25, 45

    # Ground Truth
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
                c=ground_truth[:, 2], cmap='viridis', s=2, alpha=0.3)
    ax1.set_title(f'Ground Truth\n{file_id}\nPoints: {len(ground_truth)}', fontsize=10)
    ax1.view_init(elev=view_elev, azim=view_azim)

    # Generated
    ax2 = fig.add_subplot(132, projection='3d')
    if len(reconstruction) > 0:
        ax2.scatter(reconstruction[:, 0], reconstruction[:, 1], reconstruction[:, 2],
                    c=reconstruction[:, 2], cmap='viridis', s=2, alpha=0.3)
    ax2.set_title(f'Generated (Z-sorted)\nPoints: {num_generated}', fontsize=10)
    ax2.view_init(elev=view_elev, azim=view_azim)

    # Overlay
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
                c='blue', s=2, alpha=0.2, label='Ground Truth')
    if len(reconstruction) > 0:
        ax3.scatter(reconstruction[:, 0], reconstruction[:, 1], reconstruction[:, 2],
                    c='red', s=2, alpha=0.3, label='Generated')
    ax3.set_title('Overlay', fontsize=10)
    ax3.legend(markerscale=5)
    ax3.view_init(elev=view_elev, azim=view_azim)

    # Set consistent limits
    if len(reconstruction) > 0:
        all_points = np.vstack([ground_truth, reconstruction])
    else:
        all_points = ground_truth

    x_lim = (all_points[:, 0].min(), all_points[:, 0].max())
    y_lim = (all_points[:, 1].min(), all_points[:, 1].max())
    z_lim = (all_points[:, 2].min(), all_points[:, 2].max())

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        ax.set_box_aspect([x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]])

    plt.suptitle(f'Epoch {epoch} - Autoregressive Generation', fontsize=12, y=0.98)
    plt.tight_layout()

    filepath = output_dir / f'{prefix}_epoch_{epoch:03d}_{file_id}.png'
    plt.savefig(filepath, dpi=150)
    plt.close(fig)


def train_epoch(model, dataloader, optimizer, device, epoch, stop_loss_weight, max_length):
    """Training loop with improved loss computation."""
    model.train()

    total_loss = 0
    total_coord_loss = 0
    total_stop_loss = 0
    num_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        batch_loss = 0
        batch_coord_loss = 0
        batch_stop_loss = 0

        for sample in batch:
            points = sample['points'].to(device)

            # Skip very small point clouds
            if len(points) < 2:
                continue

            # Limit sequence length for curriculum learning
            if len(points) > max_length:
                # Randomly sample subset
                indices = torch.randperm(len(points))[:max_length]
                points = points[indices]

            # Forward (points already sorted by dataset)
            pred_coords, stop_logits = model(points)

            # Compute losses (points already sorted by dataset)
            loss, coord_loss, stop_loss = compute_losses(
                pred_coords, stop_logits, points, stop_loss_weight
            )

            # Accumulate gradients
            (loss / len(batch)).backward()

            batch_loss += loss.item()
            batch_coord_loss += coord_loss.item()
            batch_stop_loss += stop_loss.item()
            num_samples += 1

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        # Update totals
        total_loss += batch_loss / len(batch)
        total_coord_loss += batch_coord_loss / len(batch)
        total_stop_loss += batch_stop_loss / len(batch)

        # Progress bar
        pbar.set_postfix({
            'loss': f'{batch_loss / len(batch):.4f}',
            'coord': f'{batch_coord_loss / len(batch):.4f}',
            'stop': f'{batch_stop_loss / len(batch):.4f}',
            'max_len': max_length
        })

    avg_loss = total_loss / len(dataloader)
    avg_coord = total_coord_loss / len(dataloader)
    avg_stop = total_stop_loss / len(dataloader)

    return avg_loss, avg_coord, avg_stop


def validate(model, dataloader, device, epoch, vis_dir, num_visualizations, stop_threshold):
    """Validation with generation and visualization."""
    model.eval()

    visualized_count = 0
    generated_lengths = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

    with torch.no_grad():
        for batch in pbar:
            if visualized_count >= num_visualizations:
                break

            for sample in batch:
                if visualized_count >= num_visualizations:
                    break

                points = sample['points'].to(device)

                # Generate
                generated = model.generate(points, max_len=8192, stop_threshold=stop_threshold)
                num_generated = len(generated)
                generated_lengths.append(num_generated)

                # Visualize
                if vis_dir is not None:
                    save_visualization(
                        epoch, vis_dir, points.cpu(), generated.cpu(),
                        sample['file_id'], num_generated, prefix='val'
                    )

                visualized_count += 1

    # Statistics
    if generated_lengths:
        avg_length = np.mean(generated_lengths)
        min_length = np.min(generated_lengths)
        max_length = np.max(generated_lengths)
    else:
        avg_length = min_length = max_length = 0

    return avg_length, min_length, max_length


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = checkpoint_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)

    # Load data with Z-sorting and augmentation
    data_path = Path(args.data_path)
    train_dataset = PointCloudDataset(
        data_path, split="train", voxel_size=args.voxel_size,
        z_axis_sort=True,  # Always sort for autoregressive
        rotation_augment=args.use_rotation_augment  # Augment during training
    )
    test_dataset = PointCloudDataset(
        data_path, split="test", voxel_size=args.voxel_size,
        z_axis_sort=True,  # Always sort
        rotation_augment=False  # No augmentation for test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=True if args.num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=True if args.num_workers > 0 else False
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")

    # Create model (no longer needs rotation_augment flag)
    model = GenerativePerceiver(
        input_dim=3,
        latent_dim=args.latent_dim,
        num_latents=args.num_latents,
        num_encoder_layers=args.num_encoder_layers,
        num_processor_layers=args.num_processor_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        min_points=args.min_points
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel has {num_params:,} parameters")
    print(f"Min points: {args.min_points}")
    print(f"Stop loss weight: {args.stop_loss_weight}")
    print(f"Stop threshold: {args.stop_threshold}")
    print(f"Rotation augmentation: {args.use_rotation_augment}")
    print(f"Curriculum learning: {args.use_curriculum}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Training loop
    print("\nStarting training...")
    print("=" * 80)

    best_avg_length = 0  # Track best average generated length

    for epoch in range(1, args.epochs + 1):
        # Curriculum learning: gradually increase max sequence length
        if args.use_curriculum:
            max_length = get_curriculum_max_length(epoch, 8192)
        else:
            max_length = 8192

        # Train
        train_loss, train_coord, train_stop = train_epoch(
            model, train_loader, optimizer, device, epoch,
            args.stop_loss_weight, max_length
        )

        # Validate
        if epoch % args.vis_freq == 0:
            avg_length, min_length, max_length_gen = validate(
                model, test_loader, device, epoch, vis_dir,
                args.num_vis, args.stop_threshold
            )
        else:
            avg_length = min_length = max_length_gen = 0

        scheduler.step()

        # Print summary
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.6f} (coord: {train_coord:.6f}, stop: {train_stop:.6f})")
        if epoch % args.vis_freq == 0:
            print(f"  Generated Lengths: avg={avg_length:.0f}, range=[{min_length}-{max_length_gen}]")

            # Check for degenerate solutions
            if avg_length < 50:
                print(f"  WARNING: Model generating very few points! Consider:")
                print(f"    - Decreasing stop_loss_weight (currently {args.stop_loss_weight})")
                print(f"    - Increasing min_points (currently {args.min_points})")

        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save checkpoint
        if epoch % args.save_freq == 0 or (epoch % args.vis_freq == 0 and avg_length > best_avg_length):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'avg_generated_length': avg_length,
                'args': vars(args)
            }

            if avg_length > best_avg_length and avg_length > 0:
                torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
                print(f"  *** Best model saved (avg_length: {avg_length:.0f}) ***")
                best_avg_length = avg_length

            if epoch % args.save_freq == 0:
                torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth')

        print("=" * 80)

    print("\nTraining complete!")
    print(f"Best average generated length: {best_avg_length:.0f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Improved Autoregressive Point Cloud Generator')

    # Data
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, default='./FOR-species20K')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--voxel_size', type=float, default=0.5)

    # Model
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--num_latents', type=int, default=256)
    parser.add_argument('--num_encoder_layers', type=int, default=4)
    parser.add_argument('--num_processor_layers', type=int, default=2)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Autoregressive-specific
    parser.add_argument('--min_points', type=int, default=4,
                        help='Minimum points before stopping allowed')
    parser.add_argument('--stop_loss_weight', type=float, default=1,
                        help='Weight for stop loss (start low!)')
    parser.add_argument('--stop_threshold', type=float, default=0.5,
                        help='Stop probability threshold for inference')
    parser.add_argument('--use_rotation_augment', action='store_true', default=True,
                        help='Use rotation augmentation around Z-axis')
    parser.add_argument('--use_curriculum', action='store_true', default=True,
                        help='Use curriculum learning for sequence length')

    # Training
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    # Logging
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_improved')
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--vis_freq', type=int, default=1)
    parser.add_argument('--num_vis', type=int, default=10)

    args = parser.parse_args()
    main(args)