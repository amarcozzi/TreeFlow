# train_generative_perceiver.py
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
from model_autoencoder import GenerativePerceiver


def set_seed(seed):
    """ Set random seed for reproducibility across random, numpy, and pytorch. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_visualization(epoch, output_dir, ground_truth, reconstruction, file_id, prefix='reconstruction'):
    """ Save side-by-side visualization of ground truth and generative reconstruction. """
    fig = plt.figure(figsize=(18, 6))

    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.cpu().numpy()
    if torch.is_tensor(reconstruction):
        reconstruction = reconstruction.cpu().numpy()

    view_elev, view_azim = 25, 45

    # Ground Truth
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], c='blue', s=2, alpha=0.3)
    ax1.set_title(f'Ground Truth\n{file_id}\nPoints: {len(ground_truth)}', fontsize=10)
    ax1.view_init(elev=view_elev, azim=view_azim)

    # Generative Reconstruction
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(reconstruction[:, 0], reconstruction[:, 1], reconstruction[:, 2], c='red', s=2, alpha=0.3)
    ax2.set_title(f'Generative Reconstruction\nPoints: {len(reconstruction)}', fontsize=10)
    ax2.view_init(elev=view_elev, azim=view_azim)

    # Overlay
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], c='blue', s=2, alpha=0.2,
                label='Ground Truth')
    ax3.scatter(reconstruction[:, 0], reconstruction[:, 1], reconstruction[:, 2], c='red', s=2, alpha=0.3,
                label='Reconstruction')
    ax3.set_title('Overlay Comparison', fontsize=10)
    ax3.legend(markerscale=5)
    ax3.view_init(elev=view_elev, azim=view_azim)

    # Set consistent axis limits
    x_lim = (ground_truth[:, 0].min(), ground_truth[:, 0].max())
    y_lim = (ground_truth[:, 1].min(), ground_truth[:, 1].max())
    z_lim = (ground_truth[:, 2].min(), ground_truth[:, 2].max())
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(x_lim);
        ax.set_ylim(y_lim);
        ax.set_zlim(z_lim)
        ax.set_box_aspect([x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]])

    plt.suptitle(f'Epoch {epoch}', fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = output_dir / f'{prefix}_epoch_{epoch:03d}_{file_id}.png'
    plt.savefig(filepath, dpi=150)
    plt.close(fig)


def train_epoch(model, dataloader, optimizer, device, epoch, loss_weights):
    """ New training loop for the autoregressive model. """
    model.train()
    total_loss, total_coord_loss, total_stop_loss = 0, 0, 0

    coord_loss_fn = nn.MSELoss()
    stop_loss_fn = nn.BCEWithLogitsLoss()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        # This loop handles cases where batch items have different numbers of points
        # and accumulates gradients before the optimizer step.
        for sample in batch:
            points = sample['points'].to(device)
            if len(points) < 2: continue  # Need at least two points to form a sequence

            # Forward pass using teacher forcing
            pred_coords, stop_logits = model(points)

            # --- Prepare Targets ---
            # 1. Target coordinates are the original sorted sequence
            target_coords = points[torch.argsort(points[:, 0])]

            # 2. Target for stop logits: 0 for all points, 1 for the very last one
            stop_targets = torch.zeros_like(stop_logits)
            stop_targets[-1] = 1.0

            # --- Calculate Losses ---
            coord_loss = coord_loss_fn(pred_coords, target_coords)
            stop_loss = stop_loss_fn(stop_logits, stop_targets)

            # Combine with weights
            loss = (loss_weights['coord'] * coord_loss +
                    loss_weights['stop'] * stop_loss)

            # Scale loss by number of samples in batch to average gradients
            (loss / len(batch)).backward()

            # Log individual loss components
            total_coord_loss += coord_loss.item()
            total_stop_loss += stop_loss.item()
            total_loss += loss.item()

        # Clip gradients and step optimizer
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pbar.set_postfix({
            'avg_loss': f'{total_loss / (batch_idx + 1) / len(batch):.4f}',
            'coord_l': f'{total_coord_loss / (batch_idx + 1) / len(batch):.4f}',
            'stop_l': f'{total_stop_loss / (batch_idx + 1) / len(batch):.4f}'
        })

    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, device, epoch, vis_dir, num_visualizations):
    """ New validation loop for generative visualization. """
    model.eval()
    visualized_count = 0
    print(f"\nRunning generative validation for epoch {epoch}...")
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

    with torch.no_grad():
        for batch in pbar:
            if visualized_count >= num_visualizations:
                break

            for sample in batch:
                if visualized_count >= num_visualizations:
                    break

                points = sample['points'].to(device)
                if len(points) < 2: continue

                # Generate a new point cloud based on the latent representation of the input
                reconstruction = model.generate(points, max_len=len(points) + 500)

                save_visualization(
                    epoch=epoch, output_dir=vis_dir,
                    ground_truth=points.cpu(),
                    reconstruction=reconstruction.cpu(),
                    file_id=sample['file_id'], prefix='val_generative'
                )
                visualized_count += 1
    print(f"Saved {visualized_count} visualizations to {vis_dir}")


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = checkpoint_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)

    data_path = Path(args.data_path)
    train_dataset = PointCloudDataset(data_path, split="train", voxel_size=args.voxel_size)
    test_dataset = PointCloudDataset(data_path, split="test", voxel_size=args.voxel_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             collate_fn=collate_fn)

    model = GenerativePerceiver(
        input_dim=3,
        latent_dim=args.latent_dim,
        num_latents=args.num_latents,
        num_encoder_layers=args.num_encoder_layers,
        num_processor_layers=args.num_processor_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Loss weights are now a critical hyperparameter
    loss_weights = {'coord': args.coord_loss_weight, 'stop': args.stop_loss_weight}

    print("\nStarting training for Generative Perceiver...")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, loss_weights)

        if epoch % args.vis_freq == 0:
            validate(model, test_loader, device, epoch, vis_dir, args.num_vis)

        scheduler.step()

        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.2e}")

        if epoch % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth')
            print(f"  Checkpoint saved to {checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth'}")

        print("=" * 80)

    print("\nTraining complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Generative Perceiver for Point Clouds')

    # Data and Reproducibility
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, default='./FOR-species20K', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--voxel_size', type=float, default=0.5, help='Voxel size for downsampling')

    # Model Hyperparameters
    parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of latent space')
    parser.add_argument('--num_latents', type=int, default=128, help='Number of latent vectors')
    parser.add_argument('--num_encoder_layers', type=int, default=4, help='Number of encoder layers')
    parser.add_argument('--num_processor_layers', type=int, default=2, help='Number of processor layers')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Number of decoder layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Training and Loss Hyperparameters
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--coord_loss_weight', type=float, default=1.0, help='Weight for coordinate prediction loss')
    parser.add_argument('--stop_loss_weight', type=float, default=1.0, help='Weight for stop token prediction loss')

    # Logging and Saving
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_generative',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=25, help='Save checkpoint every N epochs')
    parser.add_argument('--vis_freq', type=int, default=5, help='Visualize results every N epochs')
    parser.add_argument('--num_vis', type=int, default=5, help='Number of visualizations to save per validation run')

    args = parser.parse_args()
    main(args)