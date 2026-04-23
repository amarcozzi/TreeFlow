"""
animate_time_evolution.py

Create an animation analog to figures.py::create_figure_time_evolution.

Shows a 3D point cloud evolving from noise (t=0) to a generated tree (t=1),
rendered as a GIF and (optionally) an MP4 suitable for embedding in the
repo README.

All model, solver, seed, tree-id, and CFG settings mirror the static figure
exactly, so the final frame matches the last panel of
figures/figure_time_evolution.pdf.
"""

import io
import json
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch

from dataset import create_datasets
from models import get_model
from generate_samples import load_experiment_config, load_checkpoint
from sample import sample_conditional


def build_time_schedule(fps: int, duration_s: float, hold_frames: int, ease_power: float):
    """Build a monotonic array of model times t ∈ [0, 1] for each motion frame.

    Ease-out so more frames cluster near t=1, where the tree structure emerges.
    """
    total_frames = int(round(fps * duration_s))
    n_motion = max(2, total_frames - hold_frames)
    u = np.linspace(0.0, 1.0, n_motion)
    t = 1.0 - (1.0 - u) ** ease_power
    t[0] = 0.0
    t[-1] = 1.0
    # torchdiffeq requires strictly increasing time grid; de-duplicate just in case.
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1e-6
    t[-1] = 1.0
    return t, hold_frames


def main():
    parser = argparse.ArgumentParser(
        description="Animate the time evolution of a generated tree point cloud."
    )
    parser.add_argument("--experiment_dir", default="experiments/finetune-8-512-16384")
    parser.add_argument("--data_path", default="./data/preprocessed-16384")
    parser.add_argument("--tree_id", default="6069",
                        help="Tree file id (will be zero-padded to 5 digits).")
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--solver_method", default="dopri5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="figures")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--duration_s", type=float, default=8.0)
    parser.add_argument("--hold_frames", type=int, default=40,
                        help="Frames held on the final tree at the end of the animation.")
    parser.add_argument("--ease_power", type=float, default=3.0,
                        help="k in t = 1 - (1-u)^k. k=1 linear; k>1 lingers near t=1.")
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--fig_inches", type=float, default=4.0)
    parser.add_argument("--point_size", type=float, default=0.4)
    parser.add_argument("--no_mp4", action="store_true", help="Skip MP4 output.")
    parser.add_argument("--no_gif", action="store_true", help="Skip GIF output.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    experiment_dir = Path(args.experiment_dir)
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────
    config = load_experiment_config(experiment_dir)
    _, _, _, species_list, type_list = create_datasets(
        data_path=str(data_path),
        rotation_augment=False,
        shuffle_augment=False,
        max_points=config.get("max_points", 16384),
        cache_train=False,
        cache_val=False,
        cache_test=False,
    )
    config["species_list"] = species_list
    config["type_list"] = type_list
    model_args = SimpleNamespace(**config)
    model = get_model(model_args, device=device)

    ckpt_dir = experiment_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    model = load_checkpoint(ckpts[-1], model, device)

    # ── Look up tree metadata ──────────────────────────────────────────
    tree_id_str = str(args.tree_id).zfill(5)
    meta_csv = pd.read_csv(data_path / "metadata.csv")
    meta_csv["file_id"] = meta_csv["filename"].apply(lambda x: Path(x).stem)
    tree_row = meta_csv[meta_csv["file_id"] == tree_id_str]
    if len(tree_row) == 0:
        raise ValueError(f"Tree {tree_id_str} not found in metadata.csv")
    tree_row = tree_row.iloc[0]

    height_m = float(tree_row["tree_H"])
    species_raw = tree_row["species"]
    data_type_raw = tree_row["data_type"]
    species_idx = species_list.index(species_raw)
    type_idx = type_list.index(data_type_raw)

    print(f"Tree {tree_id_str} | {species_raw} | H={height_m:.1f}m | "
          f"{data_type_raw.upper()} | cfg={args.cfg_scale}")

    # ── Build animation time schedule ──────────────────────────────────
    model_times, hold_frames = build_time_schedule(
        fps=args.fps,
        duration_s=args.duration_s,
        hold_frames=args.hold_frames,
        ease_power=args.ease_power,
    )
    total_frames = len(model_times) + hold_frames
    print(f"Schedule: {len(model_times)} motion + {hold_frames} hold "
          f"= {total_frames} frames @ {args.fps} fps "
          f"(~{total_frames / args.fps:.2f}s)")

    # ── Generate trajectory (single ODE solve) ─────────────────────────
    num_points = config.get("max_points", 16384)
    trajectory = sample_conditional(
        model=model,
        num_points=num_points,
        device=device,
        target_height=height_m,
        species_idx=species_idx,
        type_idx=type_idx,
        cfg_values=args.cfg_scale,
        solver_method=args.solver_method,
        return_intermediates=True,
        intermediate_times=model_times.tolist(),
    )  # (T, num_points, 3) in normalized coords

    # Denormalize to meters and ground each frame (matches the static figure).
    trajectory_m = (trajectory / 2.0) * height_m
    for i in range(len(trajectory_m)):
        trajectory_m[i, :, 2] -= trajectory_m[i, :, 2].min()
    print(f"Trajectory shape: {trajectory_m.shape}")

    # ── Rendering (matches create_figure_time_evolution style) ─────────
    elev, azim = 20, 45
    margin = 0.05

    def _panel_limits(pts):
        mids, ranges = [], []
        for dim in range(3):
            lo, hi = pts[:, dim].min(), pts[:, dim].max()
            span = hi - lo
            pad = span * margin
            ranges.append(max(span + 2 * pad, 0.1))
            mids.append((lo + hi) / 2)
        return mids, ranges

    def render_frame(pts, t_value):
        mids, ranges = _panel_limits(pts)
        fig = plt.figure(figsize=(args.fig_inches, args.fig_inches * 1.25))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=pts[:, 2], cmap="viridis", s=args.point_size, alpha=0.8,
            rasterized=True,
        )
        ax.set_xlim(mids[0] - ranges[0] / 2, mids[0] + ranges[0] / 2)
        ax.set_ylim(mids[1] - ranges[1] / 2, mids[1] + ranges[1] / 2)
        ax.set_zlim(mids[2] - ranges[2] / 2, mids[2] + ranges[2] / 2)
        ax.set_box_aspect(ranges)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        t_str = f"{t_value:.3f}" if t_value not in (0.0, 1.0) else f"{t_value:.1f}"
        ax.set_title(f"$t = {t_str}$", fontsize=10, y=-0.03)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05,
                    dpi=args.dpi, facecolor="white")
        plt.close(fig)
        buf.seek(0)
        img = np.array(Image.open(buf).convert("RGB"))
        return img

    print("Rendering frames...")
    frames = []
    for i, t in enumerate(model_times):
        frames.append(render_frame(trajectory_m[i], float(t)))
        if (i + 1) % 25 == 0 or i == 0 or i == len(model_times) - 1:
            print(f"  frame {i + 1}/{len(model_times)} (t={float(t):.3f})")

    if hold_frames > 0:
        frames.extend([frames[-1]] * hold_frames)

    # Pad all frames to a consistent canvas size (frames can vary by 1-2 px).
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)

    def _pad_white(img, H, W):
        h, w = img.shape[:2]
        padded = np.full((H, W, 3), 255, dtype=np.uint8)
        y0 = (H - h) // 2
        x0 = (W - w) // 2
        padded[y0:y0 + h, x0:x0 + w] = img
        return padded

    frames = [_pad_white(f, max_h, max_w) for f in frames]
    print(f"Canvas: {max_w}x{max_h} px, {len(frames)} frames")

    frame_duration_ms = int(round(1000.0 / args.fps))

    gif_path = output_dir / "animation_time_evolution.gif"
    mp4_path = output_dir / "animation_time_evolution.mp4"

    wrote_gif = False
    if not args.no_gif:
        print(f"Writing GIF: {gif_path}")
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=frame_duration_ms,
            loop=0,
            optimize=True,
            disposal=2,
        )
        wrote_gif = True

    wrote_mp4 = False
    if not args.no_mp4:
        try:
            from matplotlib.animation import FuncAnimation, FFMpegWriter
            fig = plt.figure(figsize=(max_w / args.dpi, max_h / args.dpi), dpi=args.dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            im = ax.imshow(frames[0])

            def update(idx):
                im.set_array(frames[idx])
                return (im,)

            anim = FuncAnimation(
                fig, update, frames=len(frames),
                interval=frame_duration_ms, blit=True,
            )
            writer = FFMpegWriter(fps=args.fps, bitrate=6000, codec="libx264")
            anim.save(str(mp4_path), writer=writer, dpi=args.dpi)
            plt.close(fig)
            wrote_mp4 = True
            print(f"Wrote MP4: {mp4_path}")
        except Exception as e:
            print(f"MP4 output skipped: {e}")

    meta = {
        "tree_id": tree_id_str,
        "species": species_raw,
        "height_m": round(height_m, 2),
        "data_type": data_type_raw.upper(),
        "cfg_scale": args.cfg_scale,
        "solver_method": args.solver_method,
        "num_points": num_points,
        "seed": args.seed,
        "experiment_dir": str(experiment_dir),
        "fps": args.fps,
        "n_frames": len(frames),
        "n_motion_frames": int(len(model_times)),
        "hold_frames": int(hold_frames),
        "duration_s": len(frames) / args.fps,
        "ease_power": args.ease_power,
        "gif_path": str(gif_path) if wrote_gif else None,
        "mp4_path": str(mp4_path) if wrote_mp4 else None,
    }
    meta_path = output_dir / "animation_time_evolution.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
