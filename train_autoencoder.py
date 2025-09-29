# train_autoencoder.py
import os
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import random
import matplotlib.pyplot as plt

# Import model and dataset classes from their respective files
from model_autoencoder import PointAutoencoder
from dataset import PointCloudDataset


class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, p1, p2):
        p1_sq = torch.sum(p1 ** 2, dim=2, keepdim=True)
        p2_sq = torch.sum(p2 ** 2, dim=2, keepdim=True)
        dot_product = -2 * torch.bmm(p1, p2.transpose(1, 2))
        dist_matrix_sq = p1_sq.expand(-1, -1, p2.shape[1]) + \
                         dot_product + \
                         p2_sq.transpose(1, 2).expand(-1, p1.shape[1], -1)
        dist_matrix_sq = torch.clamp(dist_matrix_sq, min=0.0)
        dist_p1_to_p2, _ = torch.min(dist_matrix_sq, dim=2)
        dist_p2_to_p1, _ = torch.min(dist_matrix_sq, dim=1)
        loss = torch.mean(dist_p1_to_p2) + torch.mean(dist_p2_to_p1)
        return loss


def save_visualization(epoch, output_dir, ground_truth, reconstruction, file_id, prefix='reconstruction'):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], s=1, c='blue')
    ax1.set_title(f'Ground Truth: {file_id}\n(Epoch {epoch + 1})')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(reconstruction[:, 0], reconstruction[:, 1], reconstruction[:, 2], s=1, c='red')
    ax2.set_title(f'Reconstruction (Epoch {epoch + 1})')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    plt.tight_layout()
    filepath = os.path.join(output_dir, f'{prefix}_epoch_{epoch + 1:03d}.png')
    plt.savefig(filepath)
    plt.close(fig)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = PointCloudDataset(Path(args.data_path), split="train", num_points=args.num_points)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, prefetch_factor=2, persistent_workers=True, pin_memory=True,
    )
    test_dataset = PointCloudDataset(Path(args.data_path), split="test", num_points=args.num_points)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

    fixed_vis_points, fixed_vis_ids = next(iter(test_loader))
    fixed_vis_points = fixed_vis_points.to(device)
    # We'll use the ID of the first item in the fixed batch for its visualization
    fixed_vis_id = fixed_vis_ids[0]

    model = PointAutoencoder(latent_dim=args.latent_dim, num_points=args.num_points).to(device)
    chamfer_loss = ChamferDistance().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for points, _ in train_pbar:
            points = points.to(device).float()
            points_transposed = points.transpose(1, 2)
            optimizer.zero_grad()
            reconstructed_points_transposed = model(points_transposed)
            reconstructed_points = reconstructed_points_transposed.transpose(1, 2)
            loss = chamfer_loss(points, reconstructed_points)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]")
            for points, _ in val_pbar:
                points = points.to(device).float()
                points_transposed = points.transpose(1, 2)
                reconstructed_points_transposed = model(points_transposed)
                reconstructed_points = reconstructed_points_transposed.transpose(1, 2)
                loss = chamfer_loss(points, reconstructed_points)
                val_loss += loss.item()
                val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

            avg_val_loss = val_loss / len(test_loader)
            print(
                f"Epoch {epoch + 1}/{args.epochs} -> Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

            # --- Fixed Visualization ---
            reconstructed_fixed = model(fixed_vis_points.transpose(1, 2)).transpose(1, 2)
            gt_points_fixed = fixed_vis_points[0].cpu().numpy()
            recon_points_fixed = reconstructed_fixed[0].cpu().numpy()
            save_visualization(epoch, args.output_dir, gt_points_fixed, recon_points_fixed, fixed_vis_id, prefix='fixed')

            # --- Random Visualization ---
            random_idx = random.randint(0, len(test_dataset) - 1)
            random_points, random_id = test_dataset[random_idx]
            random_sample = random_points.unsqueeze(0).to(device).float()
            reconstructed_random = model(random_sample.transpose(1, 2)).transpose(1, 2)
            gt_points_random = random_sample[0].cpu().numpy()
            recon_points_random = reconstructed_random[0].cpu().numpy()
            save_visualization(epoch, args.output_dir, gt_points_random, recon_points_random, random_id, prefix='random')

        scheduler.step()
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"autoencoder_epoch_{epoch + 1}.pth"))

    print(f"Training finished. Models and visualizations saved in '{args.output_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Point Cloud Autoencoder.")
    parser.add_argument('--data_path', type=str, default='FOR-species20K', help='Path to the dataset directory.')
    parser.add_argument('--output_dir', type=str, default='output_autoencoder', help='Directory to save models and visualizations.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--num_points', type=int, default=16000, help='Number of points to sample from each cloud.')
    parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of the latent space.')
    args = parser.parse_args()
    train(args)