"""
treeflow/train.py

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
import json
from datetime import datetime

from model import TransformerVelocityField
from dataset import PointCloudDataset, collate_fn, collate_fn_batched
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver


def save_config(args, output_dir):
    """
    Save training configuration to JSON for reproducibility.

    Args:
        args: Parsed command-line arguments
        output_dir: Directory to save config
    """
    config = vars(args).copy()

    # Add metadata
    config['timestamp'] = datetime.now().isoformat()
    config['pytorch_version'] = torch.__version__
    config['cuda_available'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        config['cuda_version'] = torch.version.cuda
        config['gpu_name'] = torch.cuda.get_device_name(0)

    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Configuration saved to: {config_path}")
    return config_path


def save_losses_json(train_losses, val_losses, output_dir, epoch=None):
    """
    Save training losses to JSON file.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        output_dir: Directory to save losses
        epoch: Current epoch (optional, for metadata)
    """
    losses_data = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'num_epochs': len(train_losses),
        'best_train_loss': float(min(train_losses)) if train_losses else None,
        'best_val_loss': float(min(val_losses)) if val_losses else None,
        'final_train_loss': float(train_losses[-1]) if train_losses else None,
        'final_val_loss': float(val_losses[-1]) if val_losses else None,
        'timestamp': datetime.now().isoformat(),
    }

    if epoch is not None:
        losses_data['current_epoch'] = epoch

    losses_path = output_dir / 'losses.json'
    with open(losses_path, 'w') as f:
        json.dump(losses_data, f, indent=2)

    return losses_path


def enable_flash_attention():
    """Enable Flash Attention if available."""
    try:
        import torch.nn.functional as F
        if hasattr(F, 'scaled_dot_product_attention'):
            # Flash Attention is available in PyTorch 2.0+
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            print("✓ Flash Attention enabled")
            return True
        else:
            print("⚠  Flash Attention not available (requires PyTorch 2.0+)")
            return False
    except Exception as e:
        print(f"⚠  Could not enable Flash Attention: {e}")
        return False


def compute_loss(model, x_1, flow_path, device, use_amp=False):
    """Compute flow matching loss for a single point cloud."""
    batch_size = x_1.shape[0]
    t = torch.rand(batch_size, device=device)
    x_0 = torch.randn_like(x_1)

    # Use mixed precision if enabled
    with torch.amp.autocast('cuda', enabled=use_amp):
        path_sample = flow_path.sample(t=t, x_0=x_0, x_1=x_1)
        x_t = path_sample.x_t
        u_t = path_sample.dx_t

        pred_u_t = model(x_t, t)
        loss = nn.functional.mse_loss(pred_u_t, u_t)

    return loss


def train_epoch(model, train_loader, optimizer, flow_path, device, batch_mode, scaler=None, grad_clip_norm=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_samples = 0
    use_amp = scaler is not None

    pbar = tqdm(total=len(train_loader), desc="Training", dynamic_ncols=True)

    for batch in train_loader:
        optimizer.zero_grad()

        if batch_mode == 'accumulate':
            # Original mode: accumulate gradients for each sample
            for sample in batch:
                points = sample['points']
                points = points.unsqueeze(0).to(device)  # (1, N, 3)

                loss = compute_loss(model, points, flow_path, device, use_amp=use_amp)

                # Check for NaN loss during training
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nWarning: NaN/Inf loss detected, skipping batch")
                    continue

                normalized_loss = loss / len(batch)

                # Backward pass with AMP support
                if scaler is not None:
                    scaler.scale(normalized_loss).backward()
                else:
                    normalized_loss.backward()

                total_loss += loss.item()
                num_samples += 1

        elif batch_mode == 'sample_to_min':
            # Process entire batch at once
            points = batch['points'].to(device)  # (B, N_min, 3)

            loss = compute_loss(model, points, flow_path, device, use_amp=use_amp)

            # Check for NaN loss during training
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWarning: NaN/Inf loss detected, skipping batch")
                pbar.update(1)
                continue

            # Backward pass with AMP support
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * points.shape[0]  # Scale by batch size
            num_samples += points.shape[0]

        # Gradient clipping and optimizer step with AMP support
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
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

    # Start from noise
    x_init = torch.randn(1, num_points, 3, device=device)  # (1, N, 3)

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
        points = x_final[0].cpu().numpy()  # (N, 3)

        return points

    except Exception as e:
        print(f"\nError during sampling: {e}")
        import traceback
        traceback.print_exc()
        return None


def rotate_point_cloud_z(points, angle_degrees):
    """
    Rotate point cloud around Z-axis by a specified angle.

    Args:
        points: (N, 3) numpy array of points
        angle_degrees: Rotation angle in degrees

    Returns:
        rotated_points: (N, 3) numpy array of rotated points
    """
    theta = np.radians(angle_degrees)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    return points @ rotation_matrix.T


def visualize_point_cloud(points, title="Point Cloud", save_path=None):
    """
    Visualize a 3D point cloud with NaN/Inf filtering, showing 4 rotations in a 2x2 grid.

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
        # Create 2x2 grid of subplots
        fig = plt.figure(figsize=(16, 16))

        # Add main title
        if filtered_count > 0:
            title_with_info = f"{title}\n({len(points)}/{original_count} valid points)"
        else:
            title_with_info = title
        fig.suptitle(title_with_info, fontsize=16)

        # Four rotation angles: 0°, 90°, 180°, 270°
        rotation_angles = [0, 90, 180, 270]

        # Compute global bounds for consistent axis limits across all views
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

        # Create 4 subplots with different rotations
        for i, angle in enumerate(rotation_angles):
            ax = fig.add_subplot(2, 2, i + 1, projection='3d')

            # Rotate points around Z-axis
            rotated_points = rotate_point_cloud_z(points, angle)

            # Plot with color based on height (z-coordinate)
            ax.scatter(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2],
                      c=rotated_points[:, 2], cmap='viridis', s=1, alpha=0.6)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Rotation: {angle}°')

            # Set equal aspect ratio with same limits for all views
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()

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


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler=None, device='cuda', load_weights_only=False):
    """
    Load checkpoint and optionally restore training state.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer (state loaded only if load_weights_only=False)
        scheduler: Scheduler (state loaded only if load_weights_only=False)
        scaler: AMP scaler (state loaded only if load_weights_only=False)
        device: Device to load checkpoint to
        load_weights_only: If True, only load model weights (reset optimizer/scheduler)

    Returns:
        start_epoch: Epoch to resume from
        train_losses: List of training losses (empty if load_weights_only=True)
        best_train_loss: Best training loss so far (inf if load_weights_only=True)
    """
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Always load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Loaded model state")

    # Get epoch from checkpoint
    start_epoch = checkpoint['epoch'] + 1

    if load_weights_only:
        # Skip loading optimizer, scheduler, and scaler state
        print("⚠  load_weights_only=True: Skipping optimizer, scheduler, and scaler state")
        print(f"⚠  Continuing from epoch {start_epoch} with reset optimizer/scheduler")
        return start_epoch, [], float('inf')
    else:
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Loaded optimizer state")

        # Load scheduler state
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("✓ Loaded scheduler state")

        # Load scaler state if using AMP
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("✓ Loaded AMP scaler state")

        # Get training history
        train_losses = checkpoint.get('train_losses', [])
        best_train_loss = min(train_losses) if train_losses else float('inf')

        print(f"✓ Resuming from epoch {start_epoch}")
        print(f"✓ Training history: {len(train_losses)} epochs")
        print(f"✓ Best train loss so far: {best_train_loss:.6f}")

        return start_epoch, train_losses, best_train_loss


def setup_directories(output_dir, experiment_name):
    """
    Setup output directories for training.

    Args:
        output_dir: Base output directory
        experiment_name: Name of the experiment

    Returns:
        Dictionary containing all output paths
    """
    # Create base output directory
    base_dir = Path(output_dir)
    base_dir.mkdir(exist_ok=True, parents=True)

    # Create experiment directory
    if experiment_name:
        exp_dir = base_dir / experiment_name
    else:
        # Use timestamp if no experiment name provided
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = base_dir / f"exp_{timestamp}"

    exp_dir.mkdir(exist_ok=True, parents=True)

    # Create subdirectories
    dirs = {
        'experiment': exp_dir,
        'checkpoints': exp_dir / "checkpoints",
        'logs': exp_dir / "logs",
        'visualizations': exp_dir / "visualizations",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*60}")
    print(f"Output Directory Structure:")
    print(f"{'='*60}")
    print(f"  Experiment root:  {dirs['experiment']}")
    print(f"  Checkpoints:      {dirs['checkpoints']}")
    print(f"  Logs:             {dirs['logs']}")
    print(f"  Visualizations:   {dirs['visualizations']}")
    print(f"{'='*60}\n")

    return dirs


def train(args):
    """Main training function."""

    # Setup directories
    dirs = setup_directories(args.output_dir, args.experiment_name)

    # Save configuration first
    config_path = save_config(args, dirs['experiment'])

    print(f"Using preprocessed version: {args.preprocessed_version}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = PointCloudDataset(
        Path(args.data_path),
        split='mixed',
        preprocessed_version=args.preprocessed_version,
        sample_exponent=args.sample_exponent,
        rotation_augment=args.rotation_augment,
        shuffle_augment=args.shuffle_augment,
        max_points=args.max_points
    )
    print(f"Loaded training dataset\n"
          f" - Number of samples: {len(train_dataset)}\n"
          f" - Preprocessed version: {args.preprocessed_version}\n"
          f" - Sample exponent: {args.sample_exponent}\n"
          f" - Rotation augment: {args.rotation_augment}"
          f" - Shuffle augment: {args.shuffle_augment}\n"
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

    # Enable Flash Attention if requested and available
    if args.use_flash_attention and device.type == 'cuda':
        enable_flash_attention()
    elif args.use_flash_attention:
        print("⚠  Flash Attention requires CUDA, skipping")

    # Initialize model
    print("\nInitializing model...")
    model = TransformerVelocityField(
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.min_lr)

    # Setup AMP scaler if enabled
    scaler = None
    if args.use_amp and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        print("✓ Automatic Mixed Precision (AMP) enabled")
    elif args.use_amp:
        print("⚠  AMP requires CUDA, training without mixed precision")

    # Setup flow matching
    flow_path = CondOTProbPath()

    # Load checkpoint if resuming
    start_epoch = 1
    train_losses = []
    val_losses = []
    best_train_loss = float('inf')

    if args.resume_from:
        checkpoint_path = Path(args.resume_from)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        start_epoch, train_losses, best_train_loss = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, scaler, device,
            load_weights_only=args.load_weights_only
        )

    # Print training configuration
    print("\n" + "=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    print(f"  Device:              {device}")
    print(f"  Model parameters:    {num_params / 1e6:.2f}M")
    print(f"  Model dimensions:    {args.model_dim}")
    print(f"  Number of layers:    {args.num_layers}")
    print(f"  Number of heads:     {args.num_heads}")
    print(f"  Batch size:          {args.batch_size}")
    print(f"  Batch mode:          {args.batch_mode}")
    print(f"  Epochs:              {args.num_epochs}")
    print(f"  Starting epoch:      {start_epoch}")
    print(f"  Learning rate:       {args.lr}")
    print(f"  Min learning rate:   {args.min_lr}")
    print(f"  Weight decay:        {args.weight_decay}")
    print(f"  Gradient clip norm:  {args.grad_clip_norm}")
    print(f"  Use AMP:             {scaler is not None}")
    print(f"  Flash Attention:     {args.use_flash_attention}")
    if args.resume_from:
        print(f"  Resuming from:       {args.resume_from}")
        print(f"  Load weights only:   {args.load_weights_only}")
    print("=" * 60 + "\n")

    # Training loop
    print("Starting training...")

    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'=' * 60}")

        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            flow_path,
            device,
            args.batch_mode,
            scaler=scaler,
            grad_clip_norm=args.grad_clip_norm
        )
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.6f}")

        # Update learning rate
        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6e}")

        # Save checkpoint
        is_best = train_loss < best_train_loss
        if is_best:
            best_train_loss = train_loss
            print(f"✓ New best model! Train Loss: {best_train_loss:.6f}")

        if epoch % args.save_every == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'args': vars(args),
            }

            # Save scaler state if using AMP
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()

            torch.save(checkpoint, dirs['checkpoints'] / f'checkpoint_epoch_{epoch}.pt')

            if is_best:
                torch.save(checkpoint, dirs['checkpoints'] / 'best_model.pt')
                print(f"✓ Saved best model checkpoint")

        # Save losses (both plot and JSON)
        plot_losses(train_losses, val_losses, dirs['logs'] / 'losses.png')
        losses_json_path = save_losses_json(train_losses, val_losses, dirs['logs'], epoch=epoch)

        # Generate and visualize samples
        if epoch % args.visualize_every == 0:
            print("Generating samples...")
            pbar = tqdm(total=args.num_visualizations, desc="Sampling", dynamic_ncols=True)
            for _ in range(args.num_visualizations):
                try:
                    num_pts = np.random.randint(args.min_visualization_points, args.max_visualization_points)
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
                        save_path=dirs['visualizations'] / f'generated_epoch_{epoch}_size_{num_pts}.png'
                    )
                except Exception as e:
                    print(f"Error during sampling/visualization: {e}")
                pbar.update(1)

            pbar.close()

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best training loss: {best_train_loss:.6f}")
    print(f"Experiment directory: {dirs['experiment']}")
    print(f"Checkpoints saved in: {dirs['checkpoints']}")
    print(f"Visualizations saved in: {dirs['visualizations']}")
    print(f"Configuration saved in: {config_path}")
    print(f"Losses saved in: {losses_json_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Flow Matching model on tree point clouds')

    # Output directory arguments
    parser.add_argument('--output_dir', type=str, default='experiments',
                        help='Base output directory for all experiments')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment (creates subdirectory under output_dir). '
                             'If not provided, uses timestamp.')

    # Data arguments
    parser.add_argument('--data_path', type=str, default='FOR-species20K')
    parser.add_argument('--preprocessed_version', type=str, default='voxel_0.1m')

    # Augmentation arguments
    parser.add_argument('--sample_exponent', type=float, default=None)
    parser.add_argument('--rotation_augment', action='store_true', default=False)
    parser.add_argument('--shuffle_augment', action='store_true', default=False)

    # Model arguments
    parser.add_argument('--time_embed_dim', type=int, default=256)
    parser.add_argument('--model_dim', type=int, default=256,
                        help='Transformer model dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=8,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

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

    # Training optimization
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Enable Automatic Mixed Precision (AMP) training')
    parser.add_argument('--use_flash_attention', action='store_true', default=False,
                        help='Enable Flash Attention (requires PyTorch 2.0+)')
    parser.add_argument('--grad_clip_norm', type=float, default=2.0,
                        help='Gradient clipping norm')

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
    parser.add_argument('--min_visualization_points', type=int, default=1050)
    parser.add_argument('--max_visualization_points', type=int, default=20000)

    # Resume training
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume training from (e.g., experiments/baseline/checkpoints/best_model.pt)')
    parser.add_argument('--load_weights_only', action='store_true', default=False,
                        help='Load only model weights from checkpoint, reset optimizer and scheduler (for transfer learning or new training tasks)')

    return parser.parse_args()


def main():
    args = parse_args()
    if args.ode_method in ["dopri5"] and args.ode_steps is not None:
        print(f"Warning: ode_steps is ignored when using adaptive solver '{args.ode_method}'")
        args.ode_steps = None
    train(args)


if __name__ == '__main__':
    main()