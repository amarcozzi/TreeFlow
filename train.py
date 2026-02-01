"""
treeflow/train.py
"""

import matplotlib

matplotlib.use("Agg")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import random
import logging
from logging import NullHandler

from accelerate import Accelerator

from models import get_model
from dataset import create_datasets, collate_fn_batched
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_config(args, output_dir, species_list, type_list):
    """Save training config and mappings."""
    config = vars(args).copy()
    config["timestamp"] = datetime.now().isoformat()
    config["species_list"] = species_list
    config["type_list"] = type_list

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    return output_dir / "config.json"


def compute_loss(
    model,
    x_1,
    t,
    species,
    dtype,
    h_norm,
    flow_path,
    p_uncond=0.1,
):
    """
    Compute loss with Classifier-Free Guidance (CFG) dropout.
    """
    batch_size = x_1.shape[0]
    device = x_1.device

    # 1. Generate Mask for CFG (% chance to drop conditions)
    if p_uncond > 0:
        drop_mask = torch.rand(batch_size, device=device) < p_uncond
    else:
        drop_mask = None

    # 2. Flow Matching Setup
    x_0 = torch.randn_like(x_1)
    path_sample = flow_path.sample(t=t, x_0=x_0, x_1=x_1)
    x_t = path_sample.x_t
    u_t = path_sample.dx_t

    # 3. Forward Pass with Mask
    pred_u_t = model(
        x=x_t,
        t=t,
        species_idx=species,
        type_idx=dtype,
        height_norm=h_norm,
        drop_mask=drop_mask,
    )

    loss = nn.functional.mse_loss(pred_u_t, u_t)

    return loss


def train_epoch(
    model,
    train_loader,
    optimizer,
    accelerator,
    flow_path,
    cfg_dropout_prob=0.1,
    grad_clip_norm=1.0,
):
    model.train()
    total_loss = 0.0
    num_samples = 0

    pbar = tqdm(
        total=len(train_loader),
        desc="Training",
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )

    for batch in train_loader:
        optimizer.zero_grad(set_to_none=True)

        # Data is already on the correct device via accelerator.prepare()
        points = batch["points"]  # (B, N, 3)
        species = batch["species_idx"]  # (B,)
        dtype = batch["type_idx"]  # (B,)
        h_norm = batch["height_norm"]  # (B,)

        # Sample random timesteps
        t = torch.rand(points.shape[0], device=accelerator.device)

        loss = compute_loss(
            model,
            points,
            t,
            species,
            dtype,
            h_norm,
            flow_path,
            p_uncond=cfg_dropout_prob,
        )

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN/Inf loss detected, skipping batch")
            continue

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item() * points.shape[0]
        num_samples += points.shape[0]

        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    pbar.close()
    return total_loss / num_samples if num_samples > 0 else 0.0


@torch.no_grad()
def sample_conditional(
    model, num_points, accelerator, target_height, species_idx, type_idx, cfg_scale=1.0
):
    """
    Generate a tree with specific conditions using CFG.
    Reconstructs from Unit Cube -> Meters.
    """
    model.eval()
    device = accelerator.device

    # Prepare single-item batch
    x_init = torch.randn(1, num_points, 3, device=device)
    s_tensor = torch.tensor([species_idx], device=device, dtype=torch.long)
    t_tensor = torch.tensor([type_idx], device=device, dtype=torch.long)
    h_val_log = np.log(target_height + 1e-6)
    h_tensor = torch.tensor([h_val_log], device=device, dtype=torch.float32)

    # ODE Function with CFG
    def ode_fn(t, x):
        t_batch = torch.full((x.shape[0],), t, device=device, dtype=x.dtype)

        # 1. Unconditional Pass (Mask = True)
        drop_mask_uncond = torch.ones(x.shape[0], dtype=torch.bool, device=device)
        v_uncond = model(
            x, t_batch, s_tensor, t_tensor, h_tensor, drop_mask=drop_mask_uncond
        )

        # 2. Conditional Pass (Mask = False)
        if cfg_scale != 0:
            drop_mask_cond = torch.zeros(x.shape[0], dtype=torch.bool, device=device)
            v_cond = model(
                x, t_batch, s_tensor, t_tensor, h_tensor, drop_mask=drop_mask_cond
            )
            return v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            return v_uncond

    # Solve
    solver = ODESolver(velocity_model=ode_fn)
    x_final = solver.sample(x_init, method="dopri5", step_size=None)[0].cpu().numpy()

    # --- RECONSTRUCTION ---
    # x_norm = (x_centered / height) * 2.0
    x_meters = (x_final / 2.0) * target_height

    return x_meters


def visualize_validation_comparisons(
    model, val_ds, species_list, type_list, epoch, save_dir, accelerator, num_samples=3
):
    """
    Samples N trees from the validation set.
    Generates a synthetic counterpart for each.
    Plots Real (Left) vs Generated (Right) side-by-side.
    Only runs on main process.
    """
    logger.info(f"Generating {num_samples} validation comparisons...")

    # 1. Pick Random Indices
    indices = random.sample(range(len(val_ds)), num_samples)

    for i, idx in enumerate(indices):
        sample = val_ds[idx]

        # Extract Metadata
        # Note: These tensors are on CPU
        s_idx = sample["species_idx"].item()
        t_idx = sample["type_idx"].item()
        h_raw = sample["height_raw"].item()
        num_points = sample["num_points"]
        real_points_norm = sample["points"].numpy()

        s_name = species_list[s_idx]
        t_name = type_list[t_idx]

        # 2. Reconstruct Real Points to Meters for visualization
        real_points_meters = (real_points_norm / 2.0) * h_raw

        # 3. Generate Synthetic Counterpart
        # Using the exact same point count and conditions
        cfg_scale = np.random.uniform(2.0, 5.0)
        gen_points_meters = sample_conditional(
            model,
            num_points=num_points,
            accelerator=accelerator,
            target_height=h_raw,
            species_idx=s_idx,
            type_idx=t_idx,
            cfg_scale=cfg_scale,
        )

        # 4. Shift to Ground (Visualization Only)
        real_points_meters[:, 2] -= real_points_meters[:, 2].min()
        gen_points_meters[:, 2] -= gen_points_meters[:, 2].min()

        # 5. Plot Side-by-Side
        fig = plt.figure(figsize=(12, 6))

        # Determine axis limits (shared for both plots for fair comparison)
        # Trees are now grounded at Z=0. Height is h_raw.
        limit = h_raw / 2.0 * 1.2
        z_min, z_max = -h_raw * 0.05, h_raw * 1.05

        # --- Plot Real ---
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.scatter(
            real_points_meters[:, 0],
            real_points_meters[:, 1],
            real_points_meters[:, 2],
            s=1,
            c=real_points_meters[:, 2],
            cmap="viridis",
        )
        ax1.set_title(
            f"REAL | S={s_name}\nH={h_raw:.2f}m | N={num_points} | T={t_name}"
        )
        ax1.set_xlim(-limit, limit)
        ax1.set_ylim(-limit, limit)
        ax1.set_zlim(z_min, z_max)

        # --- Plot Generated ---
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.scatter(
            gen_points_meters[:, 0],
            gen_points_meters[:, 1],
            gen_points_meters[:, 2],
            s=1,
            c=gen_points_meters[:, 2],
            cmap="viridis",
        )
        ax2.set_title(f"GENERATED | CFG={cfg_scale:.2f}")
        ax2.set_xlim(-limit, limit)
        ax2.set_ylim(-limit, limit)
        ax2.set_zlim(z_min, z_max)

        plt.savefig(save_dir / f"ep{epoch}_val_{idx}_{s_name}.png", dpi=300)
        plt.close()


def train(args):
    # Initialize Accelerator
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    # Silence logging on non-main processes
    if not accelerator.is_main_process:
        logger.handlers.clear()
        logger.addHandler(NullHandler())
        logger.propagate = False

    # Setup Directories (only on main process)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.experiment_name if args.experiment_name else f"dit_{timestamp}"
    output_dir = Path(args.output_dir) / exp_name

    dirs = {
        "ckpt": output_dir / "checkpoints",
        "viz": output_dir / "visualizations",
        "logs": output_dir / "logs",
    }

    if accelerator.is_main_process:
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Accelerate found {accelerator.num_processes} GPUs to use.")
        logger.info(f"Output directory: {output_dir.resolve()}")

    accelerator.wait_for_everyone()

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 1. Create Datasets
    logger.info(f"Preparing datasets from {args.data_path}...")
    train_ds, val_ds, test_ds, species_list, type_list = create_datasets(
        data_path=args.data_path,
        sample_exponent=args.sample_exponent,
        rotation_augment=args.rotation_augment,
        shuffle_augment=args.shuffle_augment,
        max_points=args.max_points,
        cache_in_memory=not args.no_cache,
    )

    if accelerator.is_main_process:
        save_config(args, output_dir, species_list, type_list)

    # 2. Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_batched,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=True,
    )

    # 3. Model
    logger.info(
        f"Using device: {accelerator.device}. Process index: {accelerator.process_index}"
    )

    args.species_list = species_list
    args.type_list = type_list
    # Don't pass device - model will be moved by accelerator.prepare()
    model = get_model(args, device=None)

    logger.info(f"Model Parameters: {model.count_parameters()/1e6:.2f}M")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    if args.lr_scheduler == "constant":
        scheduler = None
    elif args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=args.min_lr
        )
    elif args.lr_scheduler == "warmup_constant":

        def warmup_lambda(epoch):
            if epoch < args.warmup_epochs:
                return epoch / args.warmup_epochs
            return 1.0

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    elif args.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=500, min_lr=args.min_lr
        )
    else:
        scheduler = None

    logger.info(
        f"LR Scheduler: {args.lr_scheduler}"
        + (
            f" (warmup: {args.warmup_epochs} epochs)"
            if args.lr_scheduler == "warmup_constant"
            else ""
        )
    )

    flow_path = CondOTProbPath()

    # Prepare model, optimizer, and dataloader with accelerator
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # Compile after prepare() for better compatibility
    if args.compile:
        try:
            logger.info("Compiling model with torch.compile...")
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"Compilation failed (safe to ignore): {e}")

    start_epoch = 1
    if args.resume_from:
        checkpoint_path = output_dir / "checkpoints" / args.resume_from
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            # Handle compiled model state dict
            state_dict = checkpoint["model"]
            new_state_dict = {
                key.replace("_orig_mod.", ""): value
                for key, value in state_dict.items()
            }
            accelerator.unwrap_model(model).load_state_dict(new_state_dict)
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
        else:
            accelerator.unwrap_model(model).load_state_dict(checkpoint)

        logger.info(f"Resuming training from epoch {start_epoch}")

    # 4. Training Loop
    best_loss = float("inf")

    for epoch in range(start_epoch, args.num_epochs + 1):
        logger.info(f"Epoch {epoch}/{args.num_epochs}")

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            accelerator,
            flow_path,
            cfg_dropout_prob=args.cfg_dropout_prob,
            grad_clip_norm=args.grad_clip_norm,
        )
        logger.info(f"Train Loss: {train_loss:.6f}")

        # Step scheduler
        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(train_loss)
            else:
                scheduler.step()

        # Save Checkpoint (only on main process)
        if accelerator.is_main_process:
            checkpoint = {
                "model": accelerator.unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "seed": args.seed,
                "epoch": epoch,
                "args": vars(args),
            }
            if scheduler is not None:
                checkpoint["scheduler"] = scheduler.state_dict()

            if train_loss < best_loss:
                best_loss = train_loss
                accelerator.save(checkpoint, dirs["ckpt"] / "best_model.pt")

            if epoch % args.save_every == 0:
                accelerator.save(checkpoint, dirs["ckpt"] / f"epoch_{epoch}.pt")

        # Visualization (only on main process)
        if epoch % args.visualize_every == 0 and accelerator.is_main_process:
            try:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.eval()
                visualize_validation_comparisons(
                    model=unwrapped_model,
                    val_ds=val_ds,
                    species_list=species_list,
                    type_list=type_list,
                    epoch=epoch,
                    save_dir=dirs["viz"],
                    accelerator=accelerator,
                    num_samples=args.num_viz_samples,
                )
            except Exception as e:
                logger.error(f"Visualization error: {e}")
                import traceback

                traceback.print_exc()

        accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="experiments")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/preprocessed-full",
        help="Path to preprocessed dataset directory (e.g., data/preprocessed-full or data/preprocessed-4096)",
    )

    # Model
    parser.add_argument(
        "--model_type",
        type=str,
        default="dit",
        choices=["dit", "pointnext"],
        help="Architecture to use: 'dit' (Transformer) or 'pointnext' (U-Net)",
    )
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["constant", "cosine", "warmup_constant", "plateau"],
        help="Learning rate scheduler: constant, cosine, warmup_constant, or plateau",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=100,
        help="Number of warmup epochs (for warmup_constant scheduler)",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=2.0)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training mode (no, fp16, or bf16)",
    )
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--cfg_dropout_prob", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)

    # Augmentation
    parser.add_argument("--sample_exponent", type=float, default=None)
    parser.add_argument("--rotation_augment", action="store_true", default=True)
    parser.add_argument("--shuffle_augment", action="store_true", default=True)
    parser.add_argument("--max_points", type=int, default=None)
    parser.add_argument(
        "--no_cache",
        action="store_true",
        default=False,
        help="Disable in-memory caching of point clouds (use if RAM is limited)",
    )

    # Misc
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--visualize_every", type=int, default=10)
    parser.add_argument("--num_viz_samples", type=int, default=4)

    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(0, 100000)

    print(f"Random Seed: {args.seed}")

    train(args)


if __name__ == "__main__":
    main()
