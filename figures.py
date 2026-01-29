"""
figures.py - Create figures for paper
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import MaxNLocator


from dataset import create_datasets

DOWNSAMPLE_POINTS = 4096


def create_figure_1(
    data_path: str = "./FOR-species20K",
    csv_path: str = "./FOR-species20K/tree_metadata_dev.csv",
    output_dir: str = "figures",
    seed: int = 42,
):
    """
    Create 4 PDF subfigures showing example tree point clouds as input to the model.

    Args:
        data_path: Path to FOR-species20K directory
        csv_path: Path to tree metadata CSV
        output_dir: Directory to save PDF figures
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load dataset with same settings as training
    # (from submit_train_transformer_8_256.sh)
    print("Loading dataset...")
    train_ds, val_ds, test_ds, species_list, type_list = create_datasets(
        data_path=data_path,
        csv_path=csv_path,
        rotation_augment=True,
        shuffle_augment=True,
        max_points=DOWNSAMPLE_POINTS,
    )

    # Select 4 diverse samples from training set
    # Pick samples with different species for variety
    n_samples = 4
    indices = np.random.choice(
        len(train_ds), size=min(100, len(train_ds)), replace=False
    )

    # Try to get diverse species
    selected_indices = []
    seen_species = set()

    for idx in indices:
        sample = train_ds[idx]
        species_idx = sample["species_idx"].item()
        if species_idx not in seen_species or len(selected_indices) < n_samples:
            selected_indices.append(idx)
            seen_species.add(species_idx)
        if len(selected_indices) >= n_samples:
            break

    # If we don't have enough, just take random ones
    while len(selected_indices) < n_samples:
        idx = np.random.randint(len(train_ds))
        if idx not in selected_indices:
            selected_indices.append(idx)

    print(f"Selected {len(selected_indices)} samples for visualization")

    # Collect metadata for JSON output
    metadata = {}

    # Create individual PDF figures
    for i, idx in enumerate(selected_indices):
        sample = train_ds[idx]
        points = sample["points"].numpy()
        height_raw = sample["height_raw"].item()
        species_idx = sample["species_idx"].item()
        type_idx = sample["type_idx"].item()
        file_id = sample["file_id"]

        # Convert normalized points back to meters
        points_meters = (points / 2.0) * height_raw
        # Shift Z so ground is at 0
        points_meters[:, 2] -= points_meters[:, 2].min()

        species_name = species_list[species_idx]
        type_name = type_list[type_idx]

        print(
            f"  Sample {i+1}: {file_id} - {species_name} ({type_name}), "
            f"H={height_raw:.1f}m, {len(points)} points"
        )

        # Store metadata
        subfig_key = chr(ord("a") + i)
        metadata[subfig_key] = {
            "file_id": file_id,
            "species": species_name,
            "data_type": type_name,
            "height_m": round(height_raw, 2),
            "num_points": len(points),
        }

        # Create figure
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")

        # Plot point cloud colored by height
        scatter = ax.scatter(
            points_meters[:, 0],
            points_meters[:, 1],
            points_meters[:, 2],
            c=points_meters[:, 2],
            cmap="viridis",
            s=2,
            alpha=0.7,
        )

        # Set equal aspect ratio
        max_range = (
            np.array(
                [
                    points_meters[:, 0].max() - points_meters[:, 0].min(),
                    points_meters[:, 1].max() - points_meters[:, 1].min(),
                    points_meters[:, 2].max() - points_meters[:, 2].min(),
                ]
            ).max()
            / 2.0
        )

        mid_x = (points_meters[:, 0].max() + points_meters[:, 0].min()) / 2
        mid_y = (points_meters[:, 1].max() + points_meters[:, 1].min()) / 2
        mid_z = (points_meters[:, 2].max() + points_meters[:, 2].min()) / 2

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Labels
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        # Clean up the view
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()

        # Save as PDF
        output_path = output_dir / f"figure_1_{chr(ord('a') + i)}.pdf"
        fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)

        print(f"    Saved: {output_path}")

    # Save metadata to JSON
    metadata_path = output_dir / "figure_1.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")

    print(f"\nFigure 1 subfigures saved to {output_dir}/")


def create_figure_2(
    sample_idx=123,
    data_path: str = "./FOR-species20K",
    csv_path: str = "./FOR-species20K/tree_metadata_dev.csv",
    output_dir: str = "figures",
    seed: int = 42,
):
    """
    Create subfigures for the TreeFlow flow matching framework illustration.

    Generates:
    - figure_2_a.pdf: Source Gaussian distribution
    - figure_2_b.pdf: Target tree point cloud
    - figure_2_c.pdf: Probability density evolution
    """
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    train_ds, val_ds, test_ds, species_list, type_list = create_datasets(
        data_path=data_path,
        csv_path=csv_path,
        rotation_augment=False,
        shuffle_augment=False,
        max_points=DOWNSAMPLE_POINTS,
    )

    # Select a tree sample
    sample_idx = np.random.randint(len(val_ds)) if sample_idx is None else sample_idx
    sample = val_ds[sample_idx]
    tree_points = sample["points"].numpy()
    height_raw = sample["height_raw"].item()
    n_points = len(tree_points)

    # # Convert to meters for visualization
    # tree_points = (tree_points_norm / 2.0) * height_raw
    # tree_points[:, 2] -= tree_points[:, 2].min()

    print(
        f"Selected tree: {sample['file_id']}, {n_points} points, height={height_raw:.1f}m"
    )

    # Generate source Gaussian noise (same number of points)
    noise_points = np.random.randn(n_points, 3)

    # ==========================================
    # Figure 2a: Source Gaussian Distribution
    # ==========================================
    print("Creating Figure 2a: Source Gaussian distribution...")
    fig_a = plt.figure(figsize=(5, 5))
    ax_a = fig_a.add_subplot(111, projection="3d")

    ax_a.scatter(
        noise_points[:, 0],
        noise_points[:, 1],
        noise_points[:, 2],
        c=noise_points[:, 2],
        cmap="viridis",
        s=2,
        alpha=0.6,
    )

    # Set equal limits
    lim = 3.5
    ax_a.set_xlim(-lim, lim)
    ax_a.set_ylim(-lim, lim)
    ax_a.set_zlim(-lim, lim)
    ax_a.set_title(r"$p_0 \sim \mathcal{N}(0, I)$", fontsize=12, pad=10)
    ax_a.view_init(elev=20, azim=45)
    ax_a.set_box_aspect([1, 1, 1])

    # # Remove axis labels and tick labels
    # ax_a.set_xticklabels([])
    # ax_a.set_yticklabels([])
    # ax_a.set_zticklabels([])

    # White background
    ax_a.xaxis.pane.fill = False
    ax_a.yaxis.pane.fill = False
    ax_a.zaxis.pane.fill = False
    ax_a.xaxis.pane.set_edgecolor("lightgray")
    ax_a.yaxis.pane.set_edgecolor("lightgray")
    ax_a.zaxis.pane.set_edgecolor("lightgray")
    ax_a.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_a.savefig(
        output_dir / "figure_2_a.png", format="png", bbox_inches="tight", dpi=800
    )
    plt.close(fig_a)
    print(f"  Saved: {output_dir}/figure_2_a.png")

    # ==========================================
    # Figure 2b: Target Tree Distribution
    # ==========================================
    print("Creating Figure 2b: Target tree distribution...")
    fig_b = plt.figure(figsize=(5, 5))
    ax_b = fig_b.add_subplot(111, projection="3d")

    ax_b.scatter(
        tree_points[:, 0],
        tree_points[:, 1],
        tree_points[:, 2],
        c=tree_points[:, 2],
        cmap="viridis",
        s=2,
        alpha=0.7,
    )

    # Set equal aspect ratio
    max_range = (
        np.array(
            [
                tree_points[:, 0].max() - tree_points[:, 0].min(),
                tree_points[:, 1].max() - tree_points[:, 1].min(),
                tree_points[:, 2].max() - tree_points[:, 2].min(),
            ]
        ).max()
        / 2.0
    )

    mid_x = (tree_points[:, 0].max() + tree_points[:, 0].min()) / 2
    mid_y = (tree_points[:, 1].max() + tree_points[:, 1].min()) / 2
    mid_z = (tree_points[:, 2].max() + tree_points[:, 2].min()) / 2

    ax_b.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_b.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_b.set_zlim(mid_z - max_range, mid_z + max_range)

    ax_b.set_title(r"$p_1 \sim p_{\mathrm{data}}$", fontsize=12, pad=10)
    ax_b.view_init(elev=20, azim=45)

    # # Remove axis labels and tick labels
    # ax_b.set_xticklabels([])
    # ax_b.set_yticklabels([])
    # ax_b.set_zticklabels([])

    ax_b.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_b.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_b.zaxis.set_major_locator(MaxNLocator(nbins=4))

    # White background
    ax_b.xaxis.pane.fill = False
    ax_b.yaxis.pane.fill = False
    ax_b.zaxis.pane.fill = False
    ax_b.xaxis.pane.set_edgecolor("lightgray")
    ax_b.yaxis.pane.set_edgecolor("lightgray")
    ax_b.zaxis.pane.set_edgecolor("lightgray")
    ax_b.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_b.savefig(output_dir / "figure_2_b.png", bbox_inches="tight", dpi=800)
    plt.close(fig_b)
    print(f"  Saved: {output_dir}/figure_2_b.png")

    # ==========================================
    # Figure 2c: Probability Density Evolution
    # ==========================================
    print("Creating Figure 2c: Probability density evolution...")

    fig_c, ax_c = plt.subplots(figsize=(8, 4))

    # Create density evolution heatmap
    n_time = 200
    n_space = 300
    t_vals = np.linspace(0, 1, n_time)
    z_vals = np.linspace(-4, 4, n_space)

    density = np.zeros((n_space, n_time))

    # Source: Standard Gaussian p_0 ~ N(0, I)
    source = np.exp(-0.5 * z_vals**2) / np.sqrt(2 * np.pi)

    # Target: Bimodal distribution representing vertical tree structure
    # Lower peak (trunk region), upper peak (canopy region)
    target = (
        0.2 * np.exp(-0.5 * ((z_vals + 1.0) / 0.4) ** 2) / 0.4
        + 0.8 * np.exp(-0.5 * ((z_vals - 1.5) / 0.6) ** 2) / 0.6
    )
    target = target / target.sum() * source.sum()

    # Create smooth interpolation
    for i, t in enumerate(t_vals):
        alpha = t**0.8
        blended = (1 - alpha) * source + alpha * target
        if 0.3 < t < 0.7:
            blended = gaussian_filter(blended, sigma=1.5)
        density[:, i] = blended

    density = gaussian_filter(density, sigma=[3, 2])

    # Normalize columns
    for i in range(n_time):
        if density[:, i].max() > 0:
            density[:, i] = density[:, i] / density[:, i].max()

    # Plot heatmap
    im = ax_c.imshow(
        density,
        aspect="auto",
        origin="lower",
        extent=[0, 1, z_vals.min(), z_vals.max()],
        cmap="jet",
        interpolation="bilinear",
    )

    # Source distribution curve (left side)
    source_scaled = source / source.max() * 0.08
    ax_c.fill_betweenx(
        z_vals,
        -source_scaled,
        0,
        alpha=0.8,
        color="#00cfff",
        edgecolor="white",
        linewidth=1.5,
    )
    ax_c.plot(-source_scaled, z_vals, color="white", linewidth=1.5)

    # Target distribution curve (right side)
    target_scaled = target / target.max() * 0.08
    ax_c.fill_betweenx(
        z_vals,
        1,
        1 + target_scaled,
        alpha=0.8,
        color="#e8e855",
        edgecolor="white",
        linewidth=1.5,
    )
    ax_c.plot(1 + target_scaled, z_vals, color="white", linewidth=1.5)

    # Axis limits and labels
    ax_c.set_xlim(-0.12, 1.12)
    ax_c.set_ylim(z_vals.min(), z_vals.max())

    # Labels consistent with flow matching notation
    ax_c.set_xlabel("Time (t)", fontsize=11)
    ax_c.text(-0.06, z_vals.min() - 0.8, r"$x_0$", fontsize=12, ha="center", va="top")
    ax_c.text(1.06, z_vals.min() - 0.8, r"$x_1$", fontsize=12, ha="center", va="top")

    # Add velocity field notation
    ax_c.text(
        0.5,
        z_vals.max() + 0.3,
        r"$v_\theta(x_t, t, c)$",
        fontsize=11,
        ha="center",
        va="bottom",
        style="italic",
    )

    # Remove y-axis, clean styling
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.spines["left"].set_visible(False)
    ax_c.tick_params(left=False, labelleft=False)

    # White background
    fig_c.patch.set_facecolor("white")
    ax_c.set_facecolor("white")

    plt.tight_layout()
    fig_c.savefig(
        output_dir / "figure_2_c.png", format="png", bbox_inches="tight", dpi=800
    )
    plt.close(fig_c)
    print(f"  Saved: {output_dir}/figure_2_c.png")

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    create_figure_1()
    create_figure_2()
