"""
figures.py - Create figures for paper
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Tight layout not applied")
from pathlib import Path
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import MaxNLocator

from numpy.polynomial import Polynomial

from dataset import create_datasets
from stem_tracker import (
    find_trunk_base,
    track_spine_pass,
    compute_rz_spine,
    compute_rs_spine,
)

DOWNSAMPLE_POINTS = 16384


def create_figure_1(
    data_path: str = "./data/preprocessed-16384",
    output_dir: str = "figures",
    seed: int = 42,
):
    """
    Create 4 PDF subfigures showing example tree point clouds as input to the model.

    Args:
        data_path: Path to preprocessed dataset directory
        output_dir: Directory to save PDF figures
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load dataset with same settings as training
    print("Loading dataset...")
    train_ds, val_ds, test_ds, species_list, type_list = create_datasets(
        data_path=data_path,
        rotation_augment=True,
        shuffle_augment=True,
        max_points=DOWNSAMPLE_POINTS,
        cache_train=False,
        cache_val=False,
        cache_test=False,
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
    data_path: str = "./data/preprocessed-16384",
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
        rotation_augment=False,
        shuffle_augment=False,
        max_points=DOWNSAMPLE_POINTS,
        cache_train=False,
        cache_val=False,
        cache_test=False,
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
    # Figure 2b: Source Gaussian Distribution (3D point cloud)
    # ==========================================
    print("Creating Figure 2b: Source Gaussian distribution (3D)...")
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
    ax_a.set_title(r"$p_0 \sim \mathcal{N}(0, I)$", fontsize=14, pad=10)
    ax_a.view_init(elev=20, azim=45)
    ax_a.set_box_aspect([1, 1, 1])

    # Clean axis styling
    ax_a.xaxis.pane.fill = False
    ax_a.yaxis.pane.fill = False
    ax_a.zaxis.pane.fill = False
    ax_a.xaxis.pane.set_edgecolor("lightgray")
    ax_a.yaxis.pane.set_edgecolor("lightgray")
    ax_a.zaxis.pane.set_edgecolor("lightgray")
    ax_a.grid(True, alpha=0.3)

    # White background
    fig_a.patch.set_facecolor("white")

    plt.tight_layout()
    fig_a.savefig(
        output_dir / "figure_2_b.png", format="png", bbox_inches="tight", dpi=300
    )
    plt.close(fig_a)
    print(f"  Saved: {output_dir}/figure_2_b.png")

    # ==========================================
    # Figure 2c: Target Tree Distribution (3D point cloud)
    # ==========================================
    print("Creating Figure 2c: Target tree distribution (3D)...")
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

    ax_b.set_title(r"$p_1 \sim p_{\mathrm{data}}$", fontsize=14, pad=10)
    ax_b.view_init(elev=20, azim=45)

    # Clean axis styling
    ax_b.xaxis.pane.fill = False
    ax_b.yaxis.pane.fill = False
    ax_b.zaxis.pane.fill = False
    ax_b.xaxis.pane.set_edgecolor("lightgray")
    ax_b.yaxis.pane.set_edgecolor("lightgray")
    ax_b.zaxis.pane.set_edgecolor("lightgray")
    ax_b.grid(True, alpha=0.3)

    ax_b.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_b.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_b.zaxis.set_major_locator(MaxNLocator(nbins=4))

    # White background
    fig_b.patch.set_facecolor("white")

    plt.tight_layout()
    fig_b.savefig(output_dir / "figure_2_c.png", bbox_inches="tight", dpi=300)
    plt.close(fig_b)
    print(f"  Saved: {output_dir}/figure_2_c.png")

    # ==========================================
    # Figure 2a (was 2c): 2D Probability Space with Flow Matching Paths
    # Horizontal layout to span full text width
    # ==========================================
    print("Creating Figure 2a: 2D probability space (horizontal)...")

    # Set seed for reproducible stochastic path
    np.random.seed(42)

    # Horizontal aspect ratio for full-width subfigure
    # With xlim 10 units and ylim 4.5 units, figsize should match this ratio
    fig_c, ax_c = plt.subplots(figsize=(14, 6.5))

    # Define grid for probability densities - horizontal layout
    grid_size = 500
    x_range = np.linspace(-0.5, 9.5, grid_size)
    y_range = np.linspace(-0.5, 4.0, grid_size)
    X, Y = np.meshgrid(x_range, y_range)

    # Define source and target points for paths (used throughout)
    x0 = np.array([1.5, 1.8])  # Source point (noise sample)
    x1 = np.array([7.5, 2.0])  # Target point (tree sample)

    # Source distribution: Simple Gaussian on the left side
    source_center = np.array([1.5, 1.8])
    source_sigma = 0.55
    source_density = np.exp(
        -((X - source_center[0]) ** 2 + (Y - source_center[1]) ** 2)
        / (2 * source_sigma**2)
    )

    # Target distribution: Complex multi-modal distribution on the right side
    # Many modes with varying shapes to show complexity of learned distribution
    target_centers = [
        (7.5, 2.0),  # Main mode
        (8.2, 2.6),  # Secondary mode
        (6.9, 2.7),  # Third mode
        (8.1, 1.4),  # Fourth mode
        (7.0, 1.3),  # Fifth mode (smaller)
        (8.5, 2.0),  # Sixth mode (smaller)
    ]
    target_sigmas = [
        (0.40, 0.44),  # Slightly elliptical
        (0.32, 0.36),
        (0.34, 0.30),
        (0.26, 0.30),
        (0.22, 0.24),
        (0.24, 0.20),
    ]
    target_weights = [0.30, 0.22, 0.18, 0.15, 0.08, 0.07]

    target_density = np.zeros_like(X)
    for center, sigmas, weight in zip(target_centers, target_sigmas, target_weights):
        # Elliptical Gaussians for more complex shapes
        target_density += weight * np.exp(
            -(
                (X - center[0]) ** 2 / (2 * sigmas[0] ** 2)
                + (Y - center[1]) ** 2 / (2 * sigmas[1] ** 2)
            )
        )

    # Create subtle background gradient
    combined_density = 0.25 * source_density + 0.75 * target_density
    combined_density = gaussian_filter(combined_density, sigma=4)

    # Plot filled contours for subtle background
    ax_c.contourf(
        X,
        Y,
        combined_density,
        levels=20,
        cmap="Reds",
        alpha=0.12,
    )

    # ==========================================
    # Velocity field: Shows complex learned transport
    # Dense field with smooth structured variation
    # ==========================================
    # Denser grid for more complexity
    n_arrows_x = 30
    n_arrows_y = 15
    arrow_x = np.linspace(0.2, 8.8, n_arrows_x)
    arrow_y = np.linspace(0.15, 3.35, n_arrows_y)
    Arrow_X, Arrow_Y = np.meshgrid(arrow_x, arrow_y)

    # Create velocity components with smooth, structured flow
    U = np.zeros_like(Arrow_X)
    V = np.zeros_like(Arrow_Y)

    for i in range(Arrow_X.shape[0]):
        for j in range(Arrow_X.shape[1]):
            px, py = Arrow_X[i, j], Arrow_Y[i, j]

            # Distance from source and target centers
            dist_to_source = np.sqrt((px - x0[0]) ** 2 + (py - x0[1]) ** 2)
            dist_to_target = np.sqrt((px - x1[0]) ** 2 + (py - x1[1]) ** 2)

            # Skip arrows inside distribution cores
            if dist_to_source < 0.85 or dist_to_target < 0.95:
                U[i, j] = 0
                V[i, j] = 0
                continue

            # Smooth blending based on position along transport
            t = (px - x0[0]) / (x1[0] - x0[0])
            t = np.clip(t, 0, 1)

            # Source influence: radial outward flow
            radial_from_source = np.array([px - x0[0], py - x0[1]])
            radial_from_source = radial_from_source / (
                np.linalg.norm(radial_from_source) + 1e-6
            )

            # Target influence: weighted attraction to all modes
            target_pull = np.array([0.0, 0.0])
            total_weight = 0
            for k, (tc, tw) in enumerate(zip(target_centers, target_weights)):
                dist_to_mode = np.sqrt((px - tc[0]) ** 2 + (py - tc[1]) ** 2)
                weight = tw / (dist_to_mode + 0.5)
                toward_mode = np.array([tc[0] - px, tc[1] - py])
                toward_mode = toward_mode / (np.linalg.norm(toward_mode) + 1e-6)
                target_pull += weight * toward_mode
                total_weight += weight
            target_pull = target_pull / (total_weight + 1e-6)

            # Smooth interpolation with smoothstep
            blend = 3 * t**2 - 2 * t**3
            local_dir = (1 - blend) * radial_from_source + blend * target_pull
            local_dir = local_dir / (np.linalg.norm(local_dir) + 1e-6)

            # Add smooth structured variation (gentle waves)
            # This adds complexity without chaos
            wave_angle = 0.25 * np.sin(0.8 * px + 0.3 * py) * np.cos(0.5 * py)
            cos_w, sin_w = np.cos(wave_angle), np.sin(wave_angle)
            local_dir = np.array(
                [
                    local_dir[0] * cos_w - local_dir[1] * sin_w,
                    local_dir[0] * sin_w + local_dir[1] * cos_w,
                ]
            )

            # Magnitude: fade at edges and near distribution centers
            edge_fade_x = min(px / 0.8, (9.0 - px) / 0.8, 1.0)
            edge_fade_y = min(py / 0.5, (3.5 - py) / 0.5, 1.0)
            source_fade = min(1.0, (dist_to_source - 0.85) / 0.7)
            target_fade = min(1.0, (dist_to_target - 0.95) / 0.8)
            magnitude = 0.15 * edge_fade_x * edge_fade_y * source_fade * target_fade

            U[i, j] = local_dir[0] * magnitude
            V[i, j] = local_dir[1] * magnitude

    # Plot velocity field
    ax_c.quiver(
        Arrow_X,
        Arrow_Y,
        U,
        V,
        color="#78909C",  # Blue-gray
        alpha=0.5,
        scale=4,
        width=0.0025,
        headwidth=3.5,
        headlength=4,
        headaxislength=3.5,
        zorder=1,
    )

    # Plot source distribution contours (blue)
    source_levels = np.linspace(0.12, 0.92, 7) * source_density.max()
    cs_source = ax_c.contour(
        X,
        Y,
        source_density,
        levels=source_levels,
        colors="#1565C0",
        linewidths=1.6,
        alpha=0.9,
    )

    # Plot target distribution contours (red/dark red)
    target_levels = np.linspace(0.08, 0.92, 8) * target_density.max()
    cs_target = ax_c.contour(
        X,
        Y,
        target_density,
        levels=target_levels,
        colors="#C62828",
        linewidths=1.6,
        alpha=0.9,
    )

    # ==========================================
    # Training path: Linear interpolation x_t = (1-t)*x_0 + t*x_1
    # ==========================================
    t_train = np.linspace(0, 1, 100)
    train_path = np.array([(1 - t) * x0 + t * x1 for t in t_train])

    ax_c.plot(
        train_path[:, 0],
        train_path[:, 1],
        color="#00ACC1",
        linewidth=3.5,
        linestyle="-",
        label=r"Training: $x_t = (1-t)x_0 + tx_1$",
        zorder=5,
    )

    # Add arrow showing direction on training path
    arrow_idx = int(len(train_path) * 0.55)
    ax_c.annotate(
        "",
        xy=(train_path[arrow_idx + 3, 0], train_path[arrow_idx + 3, 1]),
        xytext=(train_path[arrow_idx, 0], train_path[arrow_idx, 1]),
        arrowprops=dict(arrowstyle="-|>", color="#00ACC1", lw=3, mutation_scale=20),
        zorder=6,
    )

    # ==========================================
    # Inference path: ODE integration following the velocity field
    # Shows how inference actually follows the learned flow
    # ==========================================

    def compute_velocity(px, py):
        """Compute velocity at a point, following the flow field logic."""
        # Smooth blending based on position along transport
        t = (px - x0[0]) / (x1[0] - x0[0])
        t = np.clip(t, 0, 1)

        # Source influence: radial outward flow (or toward target if at source)
        radial_from_source = np.array([px - x0[0], py - x0[1]])
        norm_source = np.linalg.norm(radial_from_source)
        if norm_source > 0.1:
            radial_from_source = radial_from_source / norm_source
        else:
            # At or very near source: use direction toward target
            toward_target = np.array([x1[0] - px, x1[1] - py])
            radial_from_source = toward_target / np.linalg.norm(toward_target)

        # Target influence: weighted attraction to all modes
        target_pull = np.array([0.0, 0.0])
        total_weight = 0
        for k, (tc, tw) in enumerate(zip(target_centers, target_weights)):
            dist_to_mode = np.sqrt((px - tc[0]) ** 2 + (py - tc[1]) ** 2)
            weight = tw / (dist_to_mode + 0.5)
            toward_mode = np.array([tc[0] - px, tc[1] - py])
            norm_mode = np.linalg.norm(toward_mode)
            if norm_mode > 1e-6:
                toward_mode = toward_mode / norm_mode
            target_pull += weight * toward_mode
            total_weight += weight
        if total_weight > 1e-6:
            target_pull = target_pull / total_weight

        # Smooth interpolation with smoothstep
        blend = 3 * t**2 - 2 * t**3
        local_dir = (1 - blend) * radial_from_source + blend * target_pull
        norm_dir = np.linalg.norm(local_dir)
        if norm_dir > 1e-6:
            local_dir = local_dir / norm_dir

        # Add smooth structured variation (gentle waves) - same as quiver field
        wave_angle = 0.25 * np.sin(0.8 * px + 0.3 * py) * np.cos(0.5 * py)
        cos_w, sin_w = np.cos(wave_angle), np.sin(wave_angle)
        local_dir = np.array(
            [
                local_dir[0] * cos_w - local_dir[1] * sin_w,
                local_dir[0] * sin_w + local_dir[1] * cos_w,
            ]
        )

        return local_dir

    # Integrate ODE using Euler method with the velocity field
    n_steps = 7  # One fewer step, we'll add the final point manually
    infer_path = [x0.copy()]
    current_pos = x0.copy()

    # Step size to follow the flow
    total_distance = np.linalg.norm(x1 - x0)
    dt = total_distance / 8 * 1.0

    for i in range(n_steps):
        # Get velocity at current position
        vel = compute_velocity(current_pos[0], current_pos[1])

        # Euler step
        current_pos = current_pos + vel * dt
        infer_path.append(current_pos.copy())

    # Add final point close to target (but not exactly on it)
    final_point = x1 + np.array([-0.12, 0.08])  # Slightly offset from target
    infer_path.append(final_point)

    infer_path = np.array(infer_path)
    # Ensure start point is exact
    infer_path[0] = x0

    # Plot path segments connecting the steps
    ax_c.plot(
        infer_path[:, 0],
        infer_path[:, 1],
        color="#FF8F00",
        linewidth=2.5,
        linestyle="--",
        label="Inference: ODE integration",
        zorder=5,
    )

    # Plot intermediate step markers (exclude start and end)
    ax_c.scatter(
        infer_path[1:-1, 0],
        infer_path[1:-1, 1],
        color="#FF8F00",
        s=60,
        zorder=7,
        edgecolors="white",
        linewidths=1.5,
        marker="o",
    )

    # Plot endpoint marker (larger, to show final ODE result)
    ax_c.scatter(
        [infer_path[-1, 0]],
        [infer_path[-1, 1]],
        color="#FF8F00",
        s=180,
        zorder=9,
        edgecolors="white",
        linewidths=2.5,
        marker="o",
    )

    # ==========================================
    # Mark source and target points
    # ==========================================
    # Source point (x_0)
    ax_c.scatter(
        [x0[0]],
        [x0[1]],
        color="#1565C0",
        s=220,
        marker="o",
        zorder=10,
        edgecolors="white",
        linewidths=2.5,
    )

    # Target point (x_1)
    ax_c.scatter(
        [x1[0]],
        [x1[1]],
        color="#C62828",
        s=220,
        marker="o",
        zorder=10,
        edgecolors="white",
        linewidths=2.5,
    )

    # ==========================================
    # Labels and annotations
    # ==========================================
    # Label for source distribution
    ax_c.text(
        1.5,
        3.2,
        r"$X_0$ (Source)" + "\n" + r"$\mathcal{N}(0, I)$",
        fontsize=12,
        color="#1565C0",
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    # Label for target distribution
    ax_c.text(
        7.5,
        3.2,
        r"$X_1$ (Target)" + "\n" + r"$p_{\mathrm{data}}$",
        fontsize=12,
        color="#C62828",
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    # Axis labels
    ax_c.set_xlabel("X-axis", fontsize=12)
    ax_c.set_ylabel("Y-axis", fontsize=12)

    # Set axis limits for horizontal layout
    ax_c.set_xlim(-0.5, 9.5)
    ax_c.set_ylim(-0.5, 4.0)

    # Equal aspect ratio so circles look like circles
    ax_c.set_aspect("equal")

    # Add legend with proxy artist for velocity field
    from matplotlib.lines import Line2D

    velocity_proxy = Line2D(
        [0],
        [0],
        color="#78909C",
        marker=r"$\rightarrow$",
        markersize=15,
        linestyle="None",
        alpha=0.7,
        label=r"Learned velocity field $v_\theta$",
    )
    handles, labels = ax_c.get_legend_handles_labels()
    handles.append(velocity_proxy)
    labels.append(velocity_proxy.get_label())

    legend = ax_c.legend(
        handles,
        labels,
        loc="lower center",
        fontsize=11,
        framealpha=0.95,
        edgecolor="gray",
        ncol=3,
    )

    # White background
    fig_c.patch.set_facecolor("white")
    ax_c.set_facecolor("white")

    plt.tight_layout()
    fig_c.savefig(
        output_dir / "figure_2_a.png", format="png", bbox_inches="tight", dpi=300
    )
    plt.close(fig_c)
    print(f"  Saved: {output_dir}/figure_2_a.png")

    print(f"\nAll figures saved to {output_dir}/")


def _select_median_tree(
    per_pair_csv: str, metric: str, min_height: float = 0.0
) -> tuple[str, float]:
    """Select the tree whose median metric value is closest to the global median.

    Args:
        min_height: Minimum tree height in meters to consider.

    Returns (source_tree_id, median_value).
    """
    pair_df = pd.read_csv(per_pair_csv)
    if min_height > 0:
        pair_df = pair_df[pair_df["height_m"] >= min_height]
    global_median = pair_df[metric].median()
    tree_medians = pair_df.groupby("source_tree_id")[metric].median()
    best_tree = (tree_medians - global_median).abs().idxmin()
    return best_tree, float(tree_medians[best_tree])


def _load_pair_clouds(
    source_tree_id: str,
    per_pair_csv: str,
    data_path: str,
    experiment_dir: str,
    metric: str,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load real and generated point clouds for a specific tree.

    Picks the generated sample whose metric value is closest to the tree's
    median for that metric (most representative single pair).

    Returns (real_cloud, gen_cloud, pair_info).
    """
    import zarr

    pair_df = pd.read_csv(per_pair_csv)
    tree_pairs = pair_df[pair_df["source_tree_id"] == source_tree_id]

    # Pick the sample closest to this tree's median
    tree_median = tree_pairs[metric].median()
    best_idx = (tree_pairs[metric] - tree_median).abs().idxmin()
    pair_info = tree_pairs.loc[best_idx].to_dict()

    # Load real cloud
    data_path = Path(data_path)
    tree_id_str = f"{int(source_tree_id):05d}"
    real_path = data_path / f"{tree_id_str}.zarr"
    real_cloud = zarr.load(str(real_path)).astype(np.float32)

    # Load generated cloud
    experiment_dir = Path(experiment_dir)
    gen_meta = pd.read_csv(experiment_dir / "samples" / "samples_metadata.csv")
    sample_row = gen_meta[gen_meta["sample_id"] == pair_info["sample_id"]].iloc[0]
    gen_path = experiment_dir / "samples" / "zarr" / sample_row["sample_file"]
    gen_cloud = zarr.load(str(gen_path)).astype(np.float32)

    return real_cloud, gen_cloud, pair_info


def _compute_rz(cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SVD-based trunk-aligned cylindrical coordinates (mirrors evaluate.py)."""
    centroid = cloud.mean(axis=0)
    centered = cloud - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    axis = Vt[0]
    if axis[2] < 0:
        axis = -axis
    z = centered @ axis
    r = np.linalg.norm(centered - np.outer(z, axis), axis=1)
    return r, z


def create_figure_svd_axes(
    experiment_dir: str = "experiments/transformer-8-512-4096",
    data_path: str = "./data/preprocessed-4096",
    output_dir: str = "figures",
    seed: int = 42,
):
    """
    Create a figure showing the SVD-based coordinate system on a real tree.

    Shows a 3D point cloud with the three SVD axes drawn from the centroid,
    plus a translucent disc in the plane of axes 2-3 to illustrate the
    radial (r) vs height (z) decomposition used by HJSD and CrMAE.
    Uses the same tree selected for the HJSD figure for visual continuity.
    """
    import zarr

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    eval_dir = Path(experiment_dir) / "samples" / "evaluation"
    per_pair_csv = str(eval_dir / "per_pair.csv")

    # Use same tree as HJSD figure
    tree_id, _ = _select_median_tree(per_pair_csv, "histogram_jsd", min_height=10.0)

    # Load real cloud
    data_path = Path(data_path)
    tree_id_str = f"{int(tree_id):05d}"
    real_path = data_path / f"{tree_id_str}.zarr"
    cloud = zarr.load(str(real_path)).astype(np.float32)

    # Get species from per_pair
    pair_df = pd.read_csv(per_pair_csv)
    tree_rows = pair_df[pair_df["source_tree_id"] == tree_id]
    species = tree_rows.iloc[0]["species"]
    height_m = float(tree_rows.iloc[0]["height_m"])

    print(
        f"SVD axes figure: tree {tree_id}, {species}, H={height_m:.1f}m, "
        f"{len(cloud)} points"
    )

    # SVD decomposition
    centroid = cloud.mean(axis=0)
    centered = cloud - centroid
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    axes_svd = Vt[:3].copy()
    # Ensure axis 1 (trunk) points upward
    if axes_svd[0, 2] < 0:
        axes_svd[0] = -axes_svd[0]

    # Project for coloring by height along trunk
    z_proj = centered @ axes_svd[0]

    # Subsample for plotting
    rng = np.random.default_rng(seed)
    n_show = min(4000, len(cloud))
    idx = rng.choice(len(cloud), n_show, replace=False)

    # --- Plot ---
    fig = plt.figure(figsize=(7, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Point cloud colored by height along trunk axis
    pts = cloud[idx]
    colors_z = z_proj[idx]
    sc = ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c=colors_z,
        cmap="viridis",
        s=1.5,
        alpha=0.4,
        rasterized=True,
    )

    # Draw SVD axes as arrows from centroid
    axis_colors = ["#d62728", "#2ca02c", "#1f77b4"]  # red, green, blue
    axis_labels = [
        "Axis 1 (trunk / $z$)",
        "Axis 2",
        "Axis 3",
    ]
    # Scale arrows to ~40% of cloud extent for visibility
    cloud_extent = np.linalg.norm(centered, axis=1).max()
    arrow_len = cloud_extent * 0.45

    for i in range(3):
        direction = axes_svd[i] * arrow_len
        ax.quiver(
            centroid[0],
            centroid[1],
            centroid[2],
            direction[0],
            direction[1],
            direction[2],
            color=axis_colors[i],
            linewidth=3,
            arrow_length_ratio=0.08,
            zorder=10,
        )
        # Label at arrow tip
        tip = centroid + direction * 1.12
        ax.text(
            tip[0],
            tip[1],
            tip[2],
            axis_labels[i],
            color=axis_colors[i],
            fontsize=10,
            fontweight="bold",
            zorder=11,
        )

    # Draw translucent disc in the plane of axes 2-3 at the centroid
    disc_r = arrow_len * 0.55
    theta = np.linspace(0, 2 * np.pi, 60)
    disc_pts = (
        centroid[None, :]
        + np.outer(np.cos(theta), axes_svd[1] * disc_r)
        + np.outer(np.sin(theta), axes_svd[2] * disc_r)
    )
    # Fill as polygon
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts = [list(zip(disc_pts[:, 0], disc_pts[:, 1], disc_pts[:, 2]))]
    disc_poly = Poly3DCollection(
        verts, alpha=0.15, facecolor="#888888", edgecolor="#666666", linewidth=1
    )
    ax.add_collection3d(disc_poly)

    # Label the disc
    disc_label_pos = centroid + axes_svd[1] * disc_r * 0.7 + axes_svd[2] * disc_r * 0.7
    ax.text(
        disc_label_pos[0],
        disc_label_pos[1],
        disc_label_pos[2],
        "radial plane ($r$)",
        color="#666666",
        fontsize=9,
        fontstyle="italic",
        zorder=11,
    )

    # Centroid marker
    ax.scatter(
        [centroid[0]],
        [centroid[1]],
        [centroid[2]],
        color="black",
        s=60,
        marker="o",
        zorder=12,
        edgecolors="white",
        linewidths=1.5,
    )

    # Set equal aspect ratio
    max_range = (
        np.array(
            [
                cloud[:, 0].max() - cloud[:, 0].min(),
                cloud[:, 1].max() - cloud[:, 1].min(),
                cloud[:, 2].max() - cloud[:, 2].min(),
            ]
        ).max()
        / 2.0
    )

    mid = centroid
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.view_init(elev=20, azim=45)

    # Clean styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("lightgray")
    ax.yaxis.pane.set_edgecolor("lightgray")
    ax.zaxis.pane.set_edgecolor("lightgray")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_zlabel("Z", fontsize=10)

    species_display = species.replace("_", " ")
    ax.set_title(
        f"SVD-aligned coordinate system\n{species_display}, H = {height_m:.1f} m",
        fontsize=12,
        pad=15,
    )

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    out_path = output_dir / "figure_svd_axes.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Save metadata
    meta = {
        "source_tree_id": int(tree_id),
        "species": species,
        "height_m": height_m,
        "num_points": len(cloud),
        "singular_values": [float(s) for s in S],
    }
    meta_path = output_dir / "figure_svd_axes.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata: {meta_path}")


def create_figure_hjsd(
    experiment_dir: str = "experiments/transformer-8-512-4096",
    data_path: str = "./data/preprocessed-4096",
    output_dir: str = "figures",
    seed: int = 42,
    n_radial: int = 16,
    n_height: int = 32,
):
    """
    Create a figure explaining the Histogram JSD metric.

    Shows side-by-side 2D (r, z) density heatmaps for a real tree and its
    generated counterpart, with shared bin edges and the JSD value annotated.
    Selects the tree whose median HJSD is closest to the global median.
    """
    from scipy.spatial.distance import jensenshannon

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    eval_dir = Path(experiment_dir) / "samples" / "evaluation"
    per_pair_csv = str(eval_dir / "per_pair.csv")

    # Select representative tree (skip small trees < 10m)
    tree_id, median_hjsd = _select_median_tree(
        per_pair_csv, "histogram_jsd", min_height=10.0
    )
    print(f"HJSD figure: selected tree {tree_id} (median HJSD={median_hjsd:.4f})")

    # Load clouds
    real_cloud, gen_cloud, pair_info = _load_pair_clouds(
        tree_id, per_pair_csv, data_path, experiment_dir, "histogram_jsd", seed
    )
    print(
        f"  Species: {pair_info['species']}, Height: {pair_info['height_m']:.1f}m, "
        f"HJSD: {pair_info['histogram_jsd']:.4f}"
    )

    # Compute (r, z) for both
    r_real, z_real = _compute_rz(real_cloud)
    r_gen, z_gen = _compute_rz(gen_cloud)

    # Shared bin edges
    eps = 1e-6
    all_r = np.concatenate([r_real, r_gen])
    all_z = np.concatenate([z_real, z_gen])
    radial_edges = np.linspace(0, all_r.max() + eps, n_radial + 1)
    height_edges = np.linspace(all_z.min() - eps, all_z.max() + eps, n_height + 1)

    # Build histograms
    hist_real, _, _ = np.histogram2d(r_real, z_real, bins=[radial_edges, height_edges])
    hist_gen, _, _ = np.histogram2d(r_gen, z_gen, bins=[radial_edges, height_edges])

    # Normalize to densities (with Laplace smoothing for JSD)
    hist_real_s = hist_real + 1.0
    hist_gen_s = hist_gen + 1.0
    p = (hist_real_s / hist_real_s.sum()).flatten()
    q = (hist_gen_s / hist_gen_s.sum()).flatten()
    jsd_value = float(
        0.5 * np.sum(p * np.log(p / (0.5 * (p + q))))
        + 0.5 * np.sum(q * np.log(q / (0.5 * (p + q))))
    )

    # Normalize for display (no smoothing), mask empty bins
    density_real = np.ma.masked_equal(hist_real / hist_real.sum(), 0)
    density_gen = np.ma.masked_equal(hist_gen / hist_gen.sum(), 0)
    vmax = max(density_real.max(), density_gen.max())

    # Bin centers for axis labels
    r_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    z_centers = 0.5 * (height_edges[:-1] + height_edges[1:])

    # Colormap: white background for empty bins, sequential blue for density
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="white")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), width_ratios=[1, 1, 0.05])

    for ax, density, title in zip(
        axes[:2],
        [density_real, density_gen],
        ["Real Tree", "Generated Tree"],
    ):
        im = ax.imshow(
            density.T,
            origin="lower",
            aspect="auto",
            extent=[
                radial_edges[0],
                radial_edges[-1],
                height_edges[0],
                height_edges[-1],
            ],
            cmap=cmap,
            vmin=0,
            vmax=vmax,
        )
        ax.set_xlabel("Radial distance $r$ from trunk axis", fontsize=11)
        ax.set_ylabel("Height $z$ along trunk axis", fontsize=11)
        ax.set_title(title, fontsize=13)

    # Colorbar
    cb = fig.colorbar(im, cax=axes[2])
    cb.set_label("Point density", fontsize=11)

    # Annotation
    species_display = pair_info["species"].replace("_", " ")
    fig.suptitle(
        (
            f"Histogram JSD = {jsd_value:.4f}  "
            f"({n_radial}×{n_height} bins, SVD-aligned)\n"
            f"\\textit{{{species_display}}}"
            if plt.rcParams["text.usetex"]
            else f"Histogram JSD = {jsd_value:.4f}  "
            f"({n_radial}×{n_height} bins, SVD-aligned)\n"
            f"{species_display}"
        ),
        fontsize=13,
        y=1.02,
    )

    plt.tight_layout()
    out_path = output_dir / "figure_hjsd.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Save metadata (cast numpy types for JSON)
    meta = {
        "source_tree_id": int(tree_id),
        "sample_id": str(pair_info["sample_id"]),
        "species": pair_info["species"],
        "height_m": float(pair_info["height_m"]),
        "histogram_jsd": float(jsd_value),
        "global_median_hjsd": float(median_hjsd),
    }
    meta_path = output_dir / "figure_hjsd.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata: {meta_path}")


def create_figure_crown_mae(
    experiment_dir: str = "experiments/transformer-8-512-4096",
    data_path: str = "./data/preprocessed-4096",
    output_dir: str = "figures",
    seed: int = 42,
):
    """
    Create a figure explaining the Crown Profile MAE metric.

    Shows three top-down (bird's-eye) subfigures — one each for p50, p75, p98.
    Each subplot shows the real and generated point clouds projected onto the
    plane perpendicular to the trunk axis, with circles indicating the
    respective radial percentile for real (solid) and generated (dashed).
    Selects the tree whose median CrMAE_p75 is closest to the global median.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    eval_dir = Path(experiment_dir) / "samples" / "evaluation"
    per_pair_csv = str(eval_dir / "per_pair.csv")

    # Select representative tree (skip small trees < 10m)
    tree_id, median_mae = _select_median_tree(
        per_pair_csv, "crown_mae_p75", min_height=10.0
    )
    print(f"CrMAE figure: selected tree {tree_id} (median CrMAE_p75={median_mae:.4f})")

    # Load clouds
    real_cloud, gen_cloud, pair_info = _load_pair_clouds(
        tree_id, per_pair_csv, data_path, experiment_dir, "crown_mae_p75", seed
    )
    print(
        f"  Species: {pair_info['species']}, Height: {pair_info['height_m']:.1f}m, "
        f"CrMAE p50={pair_info['crown_mae_p50']:.4f}, "
        f"p75={pair_info['crown_mae_p75']:.4f}, "
        f"p98={pair_info['crown_mae_p98']:.4f}"
    )

    # SVD-aligned: project onto plane perpendicular to trunk axis
    def _project_topdown(cloud):
        centroid = cloud.mean(axis=0)
        centered = cloud - centroid
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        axis = Vt[0]
        if axis[2] < 0:
            axis = -axis
        e1 = Vt[1]
        e2 = Vt[2]
        x = centered @ e1
        y = centered @ e2
        z = centered @ axis
        r = np.sqrt(x**2 + y**2)
        return x, y, z, r

    x_real, y_real, z_real, r_real = _project_topdown(real_cloud)
    x_gen, y_gen, z_gen, r_gen = _project_topdown(gen_cloud)

    # Compute per-height-bin percentile radii (matching the actual metric)
    n_height = 32
    min_pts = 5
    eps = 1e-6
    all_z = np.concatenate([z_real, z_gen])
    height_edges = np.linspace(all_z.min() - eps, all_z.max() + eps, n_height + 1)

    def _mean_percentile_radius(r, z, pct):
        vals = []
        for i in range(n_height):
            mask = (z >= height_edges[i]) & (z < height_edges[i + 1])
            if mask.sum() >= min_pts:
                vals.append(np.percentile(r[mask], pct))
        return np.mean(vals) if vals else 0.0

    percentiles = [50, 75, 98]
    real_pcts = {p: _mean_percentile_radius(r_real, z_real, p) for p in percentiles}
    gen_pcts = {p: _mean_percentile_radius(r_gen, z_gen, p) for p in percentiles}

    # --- Plot: 3 subfigures ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = {50: "p50 (median)", 75: "p75", 98: "p98"}
    mae_keys = {50: "crown_mae_p50", 75: "crown_mae_p75", 98: "crown_mae_p98"}

    # Shared axis limits
    max_r = max(r_real.max(), r_gen.max()) * 1.15
    lim = (-max_r, max_r)

    for ax, pct in zip(axes, percentiles):
        # Scatter point clouds (subsample for clarity)
        rng_gen = np.random.default_rng(seed)
        n_show = min(2000, len(x_real), len(x_gen))
        idx_r = rng_gen.choice(len(x_real), n_show, replace=False)
        idx_g = rng_gen.choice(len(x_gen), n_show, replace=False)

        ax.scatter(
            x_real[idx_r],
            y_real[idx_r],
            s=1,
            alpha=0.25,
            color="#1f77b4",
            label="Real",
            rasterized=True,
        )
        ax.scatter(
            x_gen[idx_g],
            y_gen[idx_g],
            s=1,
            alpha=0.25,
            color="#ff7f0e",
            label="Generated",
            rasterized=True,
        )

        # Circles for mean per-height-bin percentile radii
        theta = np.linspace(0, 2 * np.pi, 200)
        r_r = real_pcts[pct]
        r_g = gen_pcts[pct]
        ax.plot(
            r_r * np.cos(theta),
            r_r * np.sin(theta),
            color="#1f77b4",
            linewidth=2,
            linestyle="-",
            label=f"Real {labels[pct]} = {r_r:.3f}",
        )
        ax.plot(
            r_g * np.cos(theta),
            r_g * np.sin(theta),
            color="#ff7f0e",
            linewidth=2,
            linestyle="--",
            label=f"Gen {labels[pct]} = {r_g:.3f}",
        )

        # Stem marker
        ax.plot(0, 0, "k+", markersize=10, markeredgewidth=2)

        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        mae_val = float(pair_info[mae_keys[pct]])
        ax.set_title(f"{labels[pct]}\nMAE = {mae_val:.4f}", fontsize=12)
        ax.legend(fontsize=8, loc="upper right", markerscale=4)

    axes[0].set_ylabel("SVD axis 3", fontsize=11)
    for ax in axes:
        ax.set_xlabel("SVD axis 2", fontsize=11)

    species_display = pair_info["species"].replace("_", " ")
    fig.suptitle(
        f"Crown Profile MAE  —  {species_display}, "
        f"H = {pair_info['height_m']:.1f} m",
        fontsize=13,
        y=1.02,
    )

    plt.tight_layout()
    out_path = output_dir / "figure_crown_mae.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Save metadata (cast numpy types for JSON)
    meta = {
        "source_tree_id": int(tree_id),
        "sample_id": str(pair_info["sample_id"]),
        "species": pair_info["species"],
        "height_m": float(pair_info["height_m"]),
        "crown_mae_p50": float(pair_info["crown_mae_p50"]),
        "crown_mae_p75": float(pair_info["crown_mae_p75"]),
        "crown_mae_p98": float(pair_info["crown_mae_p98"]),
        "global_median_crown_mae_p75": float(median_mae),
        "real_mean_p50": float(real_pcts[50]),
        "real_mean_p75": float(real_pcts[75]),
        "real_mean_p98": float(real_pcts[98]),
        "gen_mean_p50": float(gen_pcts[50]),
        "gen_mean_p75": float(gen_pcts[75]),
        "gen_mean_p98": float(gen_pcts[98]),
    }
    meta_path = output_dir / "figure_crown_mae.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata: {meta_path}")


def create_figure_spine_comparison(
    experiment_dir: str = "experiments/transformer-8-512-4096",
    data_path: str = "./data/preprocessed-4096",
    output_dir: str = "figures",
    seed: int = 42,
    n_radial: int = 32,
    n_height: int = 64,
    num_spine_bins: int = 20,
):
    """Compare SVD-axis vs stem-tracker cylindrical coordinate systems.

    Produces a 2x2 figure:
      Top row:   3D point clouds with SVD axis (left) and stem tracker (right)
      Bottom row: (r, z) density heatmaps, each with independent bin edges
                  and sqrt-normalized color to reveal both trunk and canopy
    """
    import zarr
    from matplotlib.colors import PowerNorm

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    eval_dir = Path(experiment_dir) / "samples" / "evaluation"
    per_pair_csv = str(eval_dir / "per_pair.csv")

    # Use same tree as HJSD figure for visual continuity
    tree_id, _ = _select_median_tree(per_pair_csv, "histogram_jsd", min_height=10.0)

    # Load real cloud
    data_path = Path(data_path)
    tree_id_str = f"{int(tree_id):05d}"
    real_path = data_path / f"{tree_id_str}.zarr"
    cloud = zarr.load(str(real_path)).astype(np.float32)

    pair_df = pd.read_csv(per_pair_csv)
    tree_rows = pair_df[pair_df["source_tree_id"] == tree_id]
    species = tree_rows.iloc[0]["species"]
    height_m = float(tree_rows.iloc[0]["height_m"])

    print(
        f"Spine comparison: tree {tree_id}, {species}, H={height_m:.1f}m, "
        f"{len(cloud)} points"
    )

    # ---- Compute both coordinate systems ----
    r_svd, z_svd = _compute_rz(cloud)
    r_spine, z_spine, spine_raw = compute_rz_spine(cloud, num_bins=num_spine_bins)

    # ---- SVD decomposition for 3D overlay ----
    centroid = cloud.mean(axis=0)
    centered = cloud - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    svd_axis = Vt[0]
    if svd_axis[2] < 0:
        svd_axis = -svd_axis

    # ---- Smooth polynomial spine curve for 3D overlay ----
    x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    z_min, z_max = z.min(), z.max()
    degree = 3
    actual_deg = min(degree, len(spine_raw) - 1) if len(spine_raw) >= 2 else 1
    if len(spine_raw) >= 2:
        p_x = Polynomial.fit(spine_raw[:, 2], spine_raw[:, 0], actual_deg)
        p_y = Polynomial.fit(spine_raw[:, 2], spine_raw[:, 1], actual_deg)
        spine_z_dense = np.linspace(z_min, z_max, 200)
        spine_curve = np.column_stack(
            [p_x(spine_z_dense), p_y(spine_z_dense), spine_z_dense]
        )
    else:
        spine_curve = spine_raw

    # ---- Subsample for 3D scatter ----
    rng = np.random.default_rng(seed)
    n_show = min(8000, len(cloud))
    idx = rng.choice(len(cloud), n_show, replace=False)

    # ---- Figure ----
    fig = plt.figure(figsize=(13, 12))

    # Common 3D helper
    def _setup_3d(ax, title, label):
        cloud_range = (
            np.array(
                [
                    cloud[:, 0].max() - cloud[:, 0].min(),
                    cloud[:, 1].max() - cloud[:, 1].min(),
                    cloud[:, 2].max() - cloud[:, 2].min(),
                ]
            ).max()
            / 2.0
        )
        mid = centroid
        ax.set_xlim(mid[0] - cloud_range, mid[0] + cloud_range)
        ax.set_ylim(mid[1] - cloud_range, mid[1] + cloud_range)
        ax.set_zlim(mid[2] - cloud_range, mid[2] + cloud_range)
        ax.view_init(elev=15, azim=135)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("lightgray")
        ax.yaxis.pane.set_edgecolor("lightgray")
        ax.zaxis.pane.set_edgecolor("lightgray")
        ax.grid(True, alpha=0.2)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        # Panel label
        ax.text2D(
            0.02,
            0.95,
            label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )

    # -- (a) 3D with SVD axis --
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.scatter(
        cloud[idx, 0],
        cloud[idx, 1],
        cloud[idx, 2],
        c=cloud[idx, 2],
        cmap="viridis",
        s=0.8,
        alpha=0.4,
        rasterized=True,
    )
    extent = np.linalg.norm(centered, axis=1).max() * 0.55
    for sign in [-1, 1]:
        tip = centroid + sign * svd_axis * extent
        ax1.plot(
            [centroid[0], tip[0]],
            [centroid[1], tip[1]],
            [centroid[2], tip[2]],
            color="#d62728",
            linewidth=3.5,
            zorder=10,
        )
    ax1.scatter(
        [centroid[0]],
        [centroid[1]],
        [centroid[2]],
        color="black",
        s=80,
        marker="o",
        zorder=12,
        edgecolors="white",
        linewidths=1.5,
    )
    _setup_3d(ax1, "SVD: single global axis", "(a)")

    # -- (b) 3D with stem tracker --
    ax2 = fig.add_subplot(222, projection="3d")
    ax2.scatter(
        cloud[idx, 0],
        cloud[idx, 1],
        cloud[idx, 2],
        c=cloud[idx, 2],
        cmap="viridis",
        s=0.8,
        alpha=0.4,
        rasterized=True,
    )
    ax2.plot(
        spine_curve[:, 0],
        spine_curve[:, 1],
        spine_curve[:, 2],
        color="#d62728",
        linewidth=3.5,
        zorder=10,
    )
    if len(spine_raw) > 0:
        ax2.scatter(
            spine_raw[:, 0],
            spine_raw[:, 1],
            spine_raw[:, 2],
            color="#d62728",
            s=30,
            marker="o",
            zorder=12,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.5,
        )
    _setup_3d(
        ax2,
        f"Stem tracker: {num_spine_bins} bins, degree-{actual_deg} polynomial",
        "(b)",
    )

    # ---- Heatmaps with INDEPENDENT bin edges per method ----
    cmap_heat = plt.cm.YlGnBu.copy()
    cmap_heat.set_bad(color="white")

    def _make_heatmap(ax, r, z_vals, title, label):
        eps = 1e-6
        r_edges = np.linspace(0, np.percentile(r, 99.5) + eps, n_radial + 1)
        z_edges = np.linspace(z_vals.min() - eps, z_vals.max() + eps, n_height + 1)
        hist, _, _ = np.histogram2d(r, z_vals, bins=[r_edges, z_edges])
        density = hist / hist.sum()
        density_ma = np.ma.masked_equal(density, 0)
        vmax = np.percentile(density[density > 0], 99) if (density > 0).any() else 1

        im = ax.pcolormesh(
            r_edges,
            z_edges,
            density_ma.T,
            cmap=cmap_heat,
            norm=PowerNorm(gamma=0.4, vmin=0, vmax=vmax),
            rasterized=True,
        )
        ax.set_xlabel("Radial distance $r$", fontsize=11)
        ax.set_ylabel("Height $z$", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.text(
            0.02,
            0.97,
            label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )
        return im

    # -- (c) SVD heatmap --
    ax3 = fig.add_subplot(223)
    im_svd = _make_heatmap(ax3, r_svd, z_svd, "SVD: $(r, z)$ density", "(c)")

    # -- (d) Stem tracker heatmap --
    ax4 = fig.add_subplot(224)
    im_spine = _make_heatmap(
        ax4, r_spine, z_spine, "Stem tracker: $(r, z)$ density", "(d)"
    )

    # Individual colorbars tight to each heatmap
    for ax_h, im_h in [(ax3, im_svd), (ax4, im_spine)]:
        cb = fig.colorbar(im_h, ax=ax_h, pad=0.02, fraction=0.046)
        cb.set_label("Point density", fontsize=9)
        cb.ax.tick_params(labelsize=8)

    species_display = species.replace("_", " ")
    fig.suptitle(
        f"SVD axis vs Stem Tracker  \u2014  {species_display}, "
        f"H = {height_m:.1f} m",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    fig.patch.set_facecolor("white")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_dir / "figure_spine_comparison.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Save metadata
    meta = {
        "source_tree_id": int(tree_id),
        "species": species,
        "height_m": height_m,
        "num_points": len(cloud),
        "num_spine_bins": num_spine_bins,
        "r_svd_mean": float(r_svd.mean()),
        "r_spine_mean": float(r_spine.mean()),
        "r_svd_std": float(r_svd.std()),
        "r_spine_std": float(r_spine.std()),
    }
    meta_path = output_dir / "figure_spine_comparison.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata: {meta_path}")


def create_figure_spine_audit(
    experiment_dir: str = "experiments/transformer-8-512-4096",
    data_path: str = "./data/preprocessed-4096",
    output_dir: str = "figures",
    seed: int = 42,
    n_radial: int = 32,
    n_height: int = 64,
    num_spine_bins: int = 20,
    trees_per_group: int = 2,
):
    """Generate stem-tracker comparison figures for every (species, height_bin).

    For each group, picks up to `trees_per_group` representative trees
    (closest to the group's median HJSD) and produces a 1x2 figure:
      Left:  3D point cloud with SVD axis and stem-tracker curve overlaid
      Right: side-by-side (r, z) heatmaps from both methods

    Saves into output_dir/spine_audit/<species>/<height_bin>/.
    Uses the same height bins as evaluate.py (5 m bins via get_height_bin).
    """
    import zarr
    from matplotlib.colors import PowerNorm
    from evaluate_v3 import get_height_bin

    output_dir = Path(output_dir)
    eval_dir = Path(experiment_dir) / "samples" / "evaluation_v3"
    pair_csv = eval_dir / "df_pairs.csv"
    data_path = Path(data_path)

    pair_df = pd.read_csv(pair_csv)
    if "height_bin" not in pair_df.columns:
        pair_df["height_bin"] = pair_df["height_m"].apply(get_height_bin)

    from tqdm import tqdm

    # Group by (species, height_bin), pick representative trees
    groups = pair_df.groupby(["species", "height_bin"], observed=True)
    print(f"Found {len(groups)} (species, height_bin) groups")

    # Build flat task list so we can wrap a single tqdm around all trees
    tasks = []
    for (species, height_bin), grp in groups:
        selected_trees = (
            grp["real_id"]
            .drop_duplicates()
            .sample(
                n=min(trees_per_group, grp["real_id"].nunique()),
                random_state=seed,
            )
            .tolist()
        )
        for tree_id in selected_trees:
            tasks.append(
                {
                    "species": species,
                    "height_bin": height_bin,
                    "tree_id": tree_id,
                    "grp": grp,
                }
            )

    audit_dir = output_dir / "spine_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(tasks)} figures...")

    total_figs = 0
    for task_info in tqdm(tasks, desc="Spine audit"):
        species = task_info["species"]
        height_bin = task_info["height_bin"]
        tree_id = task_info["tree_id"]
        grp = task_info["grp"]

        tree_id_str = f"{int(tree_id):05d}"
        zarr_path = data_path / f"{tree_id_str}.zarr"
        if not zarr_path.exists():
            continue

        cloud = zarr.load(str(zarr_path)).astype(np.float32)
        if len(cloud) > DOWNSAMPLE_POINTS:
            ds_rng = np.random.default_rng(seed)
            ds_idx = ds_rng.choice(len(cloud), size=DOWNSAMPLE_POINTS, replace=False)
            cloud = cloud[ds_idx]
        tree_row = grp[grp["real_id"] == tree_id].iloc[0]
        height_m = float(tree_row["height_m"])

        # Compute both coordinate systems
        r_svd, z_svd = _compute_rz(cloud)
        r_spine, z_spine, spine_raw = compute_rz_spine(cloud, num_bins=num_spine_bins)

        # SVD axis
        centroid = cloud.mean(axis=0)
        centered = cloud - centroid
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        svd_axis = Vt[0]
        if svd_axis[2] < 0:
            svd_axis = -svd_axis

        # Polynomial spine curve
        x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
        z_min, z_max = z.min(), z.max()
        degree = 3
        actual_deg = min(degree, len(spine_raw) - 1) if len(spine_raw) >= 2 else 1
        if len(spine_raw) >= 2:
            p_x = Polynomial.fit(spine_raw[:, 2], spine_raw[:, 0], actual_deg)
            p_y = Polynomial.fit(spine_raw[:, 2], spine_raw[:, 1], actual_deg)
            spine_z_dense = np.linspace(z_min, z_max, 200)
            spine_curve = np.column_stack(
                [p_x(spine_z_dense), p_y(spine_z_dense), spine_z_dense]
            )
        else:
            spine_curve = spine_raw

        # Subsample for 3D
        rng = np.random.default_rng(seed)
        n_show = min(8000, len(cloud))
        idx = rng.choice(len(cloud), n_show, replace=False)

        # ---- Figure: 2x2 matching spine_comparison layout ----
        fig = plt.figure(figsize=(13, 12))

        # Common 3D helper
        def _setup_3d(ax, title, label):
            cloud_range = (
                np.array(
                    [
                        cloud[:, 0].max() - cloud[:, 0].min(),
                        cloud[:, 1].max() - cloud[:, 1].min(),
                        cloud[:, 2].max() - cloud[:, 2].min(),
                    ]
                ).max()
                / 2.0
            )
            mid = centroid
            ax.set_xlim(mid[0] - cloud_range, mid[0] + cloud_range)
            ax.set_ylim(mid[1] - cloud_range, mid[1] + cloud_range)
            ax.set_zlim(mid[2] - cloud_range, mid[2] + cloud_range)
            ax.view_init(elev=15, azim=135)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("lightgray")
            ax.yaxis.pane.set_edgecolor("lightgray")
            ax.zaxis.pane.set_edgecolor("lightgray")
            ax.grid(True, alpha=0.2)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")
            ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
            ax.text2D(
                0.02,
                0.95,
                label,
                transform=ax.transAxes,
                fontsize=16,
                fontweight="bold",
                va="top",
            )

        # -- (a) 3D with SVD axis --
        ax1 = fig.add_subplot(221, projection="3d")
        ax1.scatter(
            cloud[idx, 0],
            cloud[idx, 1],
            cloud[idx, 2],
            c=cloud[idx, 2],
            cmap="viridis",
            s=0.8,
            alpha=0.4,
            rasterized=True,
        )
        extent = np.linalg.norm(centered, axis=1).max() * 0.55
        for sign in [-1, 1]:
            tip = centroid + sign * svd_axis * extent
            ax1.plot(
                [centroid[0], tip[0]],
                [centroid[1], tip[1]],
                [centroid[2], tip[2]],
                color="#d62728",
                linewidth=3.5,
                zorder=10,
            )
        ax1.scatter(
            [centroid[0]],
            [centroid[1]],
            [centroid[2]],
            color="black",
            s=80,
            marker="o",
            zorder=12,
            edgecolors="white",
            linewidths=1.5,
        )
        _setup_3d(ax1, "SVD: single global axis", "(a)")

        # -- (b) 3D with stem tracker --
        ax2 = fig.add_subplot(222, projection="3d")
        ax2.scatter(
            cloud[idx, 0],
            cloud[idx, 1],
            cloud[idx, 2],
            c=cloud[idx, 2],
            cmap="viridis",
            s=0.8,
            alpha=0.4,
            rasterized=True,
        )
        if len(spine_curve) > 0:
            ax2.plot(
                spine_curve[:, 0],
                spine_curve[:, 1],
                spine_curve[:, 2],
                color="#d62728",
                linewidth=3.5,
                zorder=10,
            )
        if len(spine_raw) > 0:
            ax2.scatter(
                spine_raw[:, 0],
                spine_raw[:, 1],
                spine_raw[:, 2],
                color="#d62728",
                s=30,
                marker="o",
                zorder=12,
                edgecolors="white",
                linewidths=0.8,
                alpha=0.5,
            )
        _setup_3d(
            ax2,
            f"Stem tracker: {num_spine_bins} bins, degree-{actual_deg} poly",
            "(b)",
        )

        # ---- Heatmaps with independent bin edges ----
        cmap_heat = plt.cm.YlGnBu.copy()
        cmap_heat.set_bad(color="white")

        def _make_heatmap(ax, r, z_vals, title, label):
            eps = 1e-6
            r_edges = np.linspace(0, np.percentile(r, 99.5) + eps, n_radial + 1)
            z_edges = np.linspace(z_vals.min() - eps, z_vals.max() + eps, n_height + 1)
            hist, _, _ = np.histogram2d(r, z_vals, bins=[r_edges, z_edges])
            density = hist / hist.sum()
            density_ma = np.ma.masked_equal(density, 0)
            vmax = np.percentile(density[density > 0], 99) if (density > 0).any() else 1
            im = ax.pcolormesh(
                r_edges,
                z_edges,
                density_ma.T,
                cmap=cmap_heat,
                norm=PowerNorm(gamma=0.4, vmin=0, vmax=vmax),
                rasterized=True,
            )
            ax.set_xlabel("Radial distance $r$", fontsize=11)
            ax.set_ylabel("Height $z$", fontsize=11)
            ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
            ax.text(
                0.02,
                0.97,
                label,
                transform=ax.transAxes,
                fontsize=16,
                fontweight="bold",
                va="top",
            )
            return im

        # -- (c) SVD heatmap --
        ax3 = fig.add_subplot(223)
        im_svd = _make_heatmap(ax3, r_svd, z_svd, "SVD: $(r, z)$ density", "(c)")

        # -- (d) Stem tracker heatmap --
        ax4 = fig.add_subplot(224)
        im_spine = _make_heatmap(
            ax4, r_spine, z_spine, "Stem tracker: $(r, z)$ density", "(d)"
        )

        # Individual colorbars
        for ax_h, im_h in [(ax3, im_svd), (ax4, im_spine)]:
            cb = fig.colorbar(im_h, ax=ax_h, pad=0.02, fraction=0.046)
            cb.set_label("Point density", fontsize=9)
            cb.ax.tick_params(labelsize=8)

        species_display = species.replace("_", " ")
        fig.suptitle(
            f"{species_display}  |  {height_bin} m  |  "
            f"H = {height_m:.1f} m  |  tree {tree_id}  |  "
            f"{len(cloud):,} pts",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        fig.patch.set_facecolor("white")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        species_safe = species.replace(" ", "_")
        fname = f"{species_safe}_{height_bin}m_tree_{tree_id_str}.pdf"
        fig.savefig(
            audit_dir / fname,
            format="pdf",
            bbox_inches="tight",
            dpi=200,
        )
        plt.close(fig)
        total_figs += 1

    print(f"\nSpine audit: {total_figs} figures saved to {audit_dir}/")


def create_figure_crown_audit(
    experiment_dir: str = "experiments/transformer-8-512-4096",
    data_path: str = "./data/preprocessed-4096",
    output_dir: str = "figures",
    seed: int = 42,
    num_spine_bins: int = 20,
    trees_per_group: int = 2,
):
    """Generate crown-metric verification figures for every (species, height_bin).

    For each group, picks up to `trees_per_group` representative trees and
    produces a 2x2 figure:
      (a) 3D point cloud with convex hull wireframe
      (b) 3D point cloud with HCB plane + max crown radius ring + spine
      (c) Arc-length density plot showing HCB detection algorithm
      (d) Mean radial distance vs arc-length showing max crown radius

    Saves into output_dir/crown_audit/<species>/<height_bin>/.
    Mirrors the feature extraction logic in evaluate_v3.py.
    """
    import zarr
    from scipy.spatial import ConvexHull
    from evaluate_v3 import get_height_bin

    output_dir = Path(output_dir)
    eval_dir = Path(experiment_dir) / "samples" / "evaluation_v3"
    pair_csv = eval_dir / "df_pairs.csv"
    data_path = Path(data_path)

    pair_df = pd.read_csv(pair_csv)
    if "height_bin" not in pair_df.columns:
        pair_df["height_bin"] = pair_df["height_m"].apply(get_height_bin)

    from tqdm import tqdm

    groups = pair_df.groupby(["species", "height_bin"], observed=True)
    print(f"Found {len(groups)} (species, height_bin) groups")

    tasks = []
    for (species, height_bin), grp in groups:
        selected_trees = (
            grp["real_id"]
            .drop_duplicates()
            .sample(
                n=min(trees_per_group, grp["real_id"].nunique()),
                random_state=seed,
            )
            .tolist()
        )
        for tree_id in selected_trees:
            tasks.append(
                {
                    "species": species,
                    "height_bin": height_bin,
                    "tree_id": tree_id,
                    "grp": grp,
                }
            )

    audit_dir = output_dir / "crown_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(tasks)} crown audit figures...")

    total_figs = 0
    for task_info in tqdm(tasks, desc="Crown audit"):
        species = task_info["species"]
        height_bin = task_info["height_bin"]
        tree_id = task_info["tree_id"]
        grp = task_info["grp"]

        tree_id_str = f"{int(tree_id):05d}"
        zarr_path = data_path / f"{tree_id_str}.zarr"
        if not zarr_path.exists():
            continue

        cloud = zarr.load(str(zarr_path)).astype(np.float32)
        if len(cloud) > DOWNSAMPLE_POINTS:
            ds_rng = np.random.default_rng(seed)
            ds_idx = ds_rng.choice(len(cloud), size=DOWNSAMPLE_POINTS, replace=False)
            cloud = cloud[ds_idx]
        tree_row = grp[grp["real_id"] == tree_id].iloc[0]
        height_m = float(tree_row["height_m"])
        scale = height_m / 2.0

        # ── Stem tracker → cylindrical (r, s) ──
        r, s, spine_raw, poly_x, poly_y = compute_rs_spine(
            cloud, num_bins=num_spine_bins
        )

        z = cloud[:, 2]
        z_min, z_max = z.min(), z.max()
        eps = 1e-6
        s_max = s.max() + eps if s.max() > 0 else eps

        # ── Convex hull ──
        try:
            hull = ConvexHull(cloud)
            hull_volume = float(hull.volume * scale**3)
        except Exception:
            hull = None
            hull_volume = float("nan")

        # ── Max crown radius (mirrors evaluate_v3 lines 194-201) ──
        n_slices = 30
        slice_edges = np.linspace(0, s_max, n_slices + 1)
        slice_centers = 0.5 * (slice_edges[:-1] + slice_edges[1:])
        mean_r_per_slice = np.zeros(n_slices)
        for i in range(n_slices):
            mask = (s >= slice_edges[i]) & (s < slice_edges[i + 1])
            if mask.sum() > 0:
                mean_r_per_slice[i] = r[mask].mean()
        max_crown_r_norm = mean_r_per_slice.max()
        max_crown_r_m = float(max_crown_r_norm * scale)
        max_r_slice_idx = int(np.argmax(mean_r_per_slice))

        # ── Height to crown base (Kneedle on cumulative mean-r) ──
        hcb_val = float("nan")
        kneedle_data = {}
        if mean_r_per_slice.max() > 0 and not np.allclose(
            mean_r_per_slice, mean_r_per_slice[0]
        ):
            cumr = np.cumsum(mean_r_per_slice)
            x_norm = (slice_centers - slice_centers[0]) / (
                slice_centers[-1] - slice_centers[0]
            )
            y_norm = (cumr - cumr[0]) / (cumr[-1] - cumr[0])
            d = (x_norm - y_norm) * (1 - x_norm)**0.5
            knee_idx = int(np.argmax(d))
            hcb_val = slice_centers[knee_idx] / s_max
            kneedle_data = {
                "x_norm": x_norm,
                "y_norm": y_norm,
                "d": d,
                "knee_idx": knee_idx,
            }

        hcb_m = float(hcb_val * height_m)

        # ── Spine polynomial for 3D overlay ──
        degree = 3
        actual_deg = min(degree, len(spine_raw) - 1) if len(spine_raw) >= 2 else 1
        if len(spine_raw) >= 2:
            spine_z_dense = np.linspace(z_min, z_max, 200)
            spine_curve = np.column_stack(
                [poly_x(spine_z_dense), poly_y(spine_z_dense), spine_z_dense]
            )
        else:
            spine_curve = spine_raw

        # Subsample for 3D scatter
        rng = np.random.default_rng(seed)
        n_show = min(8000, len(cloud))
        idx = rng.choice(len(cloud), n_show, replace=False)

        centroid = cloud.mean(axis=0)

        # ── Figure: 2×2 ──
        fig = plt.figure(figsize=(13, 12))

        def _setup_3d(ax, title, label):
            cloud_range = (
                np.array(
                    [
                        cloud[:, 0].max() - cloud[:, 0].min(),
                        cloud[:, 1].max() - cloud[:, 1].min(),
                        cloud[:, 2].max() - cloud[:, 2].min(),
                    ]
                ).max()
                / 2.0
            )
            mid = centroid
            ax.set_xlim(mid[0] - cloud_range, mid[0] + cloud_range)
            ax.set_ylim(mid[1] - cloud_range, mid[1] + cloud_range)
            ax.set_zlim(mid[2] - cloud_range, mid[2] + cloud_range)
            ax.view_init(elev=15, azim=135)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("lightgray")
            ax.yaxis.pane.set_edgecolor("lightgray")
            ax.zaxis.pane.set_edgecolor("lightgray")
            ax.grid(True, alpha=0.2)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")
            ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
            ax.text2D(
                0.02,
                0.95,
                label,
                transform=ax.transAxes,
                fontsize=16,
                fontweight="bold",
                va="top",
            )

        # -- (a) 3D with convex hull --
        ax1 = fig.add_subplot(221, projection="3d")
        ax1.scatter(
            cloud[idx, 0],
            cloud[idx, 1],
            cloud[idx, 2],
            c=cloud[idx, 2],
            cmap="viridis",
            s=0.8,
            alpha=0.4,
            rasterized=True,
        )
        if hull is not None:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            hull_faces = [cloud[s] for s in hull.simplices]
            hull_col = Poly3DCollection(
                hull_faces,
                alpha=0.04,
                facecolor="red",
                edgecolor="red",
                linewidth=0.3,
            )
            ax1.add_collection3d(hull_col)
        _setup_3d(ax1, f"Convex hull  (V = {hull_volume:.2f} m³)", "(a)")

        # -- (b) 3D with HCB plane + max crown radius ring --
        ax2 = fig.add_subplot(222, projection="3d")
        ax2.scatter(
            cloud[idx, 0],
            cloud[idx, 1],
            cloud[idx, 2],
            c=cloud[idx, 2],
            cmap="viridis",
            s=0.8,
            alpha=0.4,
            rasterized=True,
        )
        # Spine curve
        if len(spine_curve) > 0:
            ax2.plot(
                spine_curve[:, 0],
                spine_curve[:, 1],
                spine_curve[:, 2],
                color="#d62728",
                linewidth=3.0,
                zorder=10,
            )

        # HCB horizontal plane — find spine center at HCB z-height
        if not np.isnan(hcb_val):
            hcb_z = z_min + hcb_val * (z_max - z_min)
            hcb_cx = float(poly_x(hcb_z))
            hcb_cy = float(poly_y(hcb_z))
            theta = np.linspace(0, 2 * np.pi, 60)
            plane_r = 0.3 * (z_max - z_min)
            plane_x = hcb_cx + plane_r * np.cos(theta)
            plane_y = hcb_cy + plane_r * np.sin(theta)
            plane_z = np.full_like(theta, hcb_z)
            ax2.plot(
                plane_x, plane_y, plane_z, color="#2ca02c", linewidth=2.5, zorder=8
            )
            ax2.plot_trisurf(
                plane_x,
                plane_y,
                plane_z,
                color="#2ca02c",
                alpha=0.15,
                zorder=7,
            )

        # Max crown radius ring — at the arc-length slice with max mean-r
        if max_crown_r_norm > 0:
            # Find the z-height corresponding to the max-r slice center arc-length
            s_center = slice_centers[max_r_slice_idx]
            # Map arc-length back to z via interpolation on spine
            spine_z_fine = np.linspace(z_min, z_max, 500)
            from numpy.polynomial import Polynomial as Poly

            dpx = poly_x.deriv()
            dpy = poly_y.deriv()
            ds_dz = np.sqrt(dpx(spine_z_fine) ** 2 + dpy(spine_z_fine) ** 2 + 1.0)
            dz_vals = np.diff(spine_z_fine)
            ds_vals = 0.5 * (ds_dz[:-1] + ds_dz[1:]) * dz_vals
            s_fine = np.zeros(len(spine_z_fine))
            s_fine[1:] = np.cumsum(ds_vals)
            ring_z = float(np.interp(s_center, s_fine, spine_z_fine))
            ring_cx = float(poly_x(ring_z))
            ring_cy = float(poly_y(ring_z))
            theta = np.linspace(0, 2 * np.pi, 60)
            ring_x = ring_cx + max_crown_r_norm * np.cos(theta)
            ring_y = ring_cy + max_crown_r_norm * np.sin(theta)
            ring_zz = np.full_like(theta, ring_z)
            ax2.plot(
                ring_x,
                ring_y,
                ring_zz,
                color="#ff7f0e",
                linewidth=2.5,
                zorder=9,
            )

        _setup_3d(
            ax2,
            f"HCB = {hcb_m:.1f} m  |  Max CrR = {max_crown_r_m:.2f} m",
            "(b)",
        )

        # -- (c) HCB detection (Kneedle on cumulative r) --
        ax3 = fig.add_subplot(223)
        if kneedle_data:
            x_n = kneedle_data["x_norm"]
            y_n = kneedle_data["y_norm"]
            d_curve = kneedle_data["d"]
            ki = kneedle_data["knee_idx"]

            ax3.plot(
                x_n, y_n, color="steelblue", linewidth=2, label="Cumulative mean-$r$"
            )
            ax3.plot(
                [0, 1],
                [0, 1],
                color="gray",
                linestyle="--",
                linewidth=1,
                label="Diagonal",
            )
            ax3.fill_between(x_n, x_n, y_n, alpha=0.10, color="steelblue")
            ax3.axvline(
                x_n[ki],
                color="#2ca02c",
                linewidth=2,
                label=f"Knee → HCB = {hcb_m:.1f} m",
            )
            ax3.plot(x_n[ki], y_n[ki], "o", color="#2ca02c", markersize=8, zorder=10)

            # Inset: distance curve
            ax3_inset = ax3.twinx()
            ax3_inset.plot(
                x_n,
                d_curve,
                color="#d62728",
                linewidth=1.2,
                alpha=0.6,
                label="Distance $d$",
            )
            ax3_inset.set_ylabel("Distance $d$", fontsize=9, color="#d62728")
            ax3_inset.tick_params(axis="y", labelcolor="#d62728", labelsize=8)

            ax3.set_xlabel("Normalized arc-length", fontsize=11)
            ax3.set_ylabel("Normalized cumulative $r$", fontsize=11)
            ax3.set_title(
                "HCB detection (Kneedle on cumulative $r$)",
                fontsize=13,
                fontweight="bold",
                pad=8,
            )
            ax3.legend(fontsize=8, loc="upper left")
        ax3.text(
            0.02,
            0.97,
            "(c)",
            transform=ax3.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )

        # -- (d) Mean-r vs arc-length → max crown radius --
        ax4 = fig.add_subplot(224)
        slice_centers_m = slice_centers * scale
        mean_r_m = mean_r_per_slice * scale
        ax4.plot(
            slice_centers_m,
            mean_r_m,
            color="steelblue",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Mean $r$ per slice",
        )
        ax4.axhline(
            max_crown_r_m,
            color="#ff7f0e",
            linestyle="--",
            linewidth=1.5,
            label=f"Max = {max_crown_r_m:.2f} m",
        )
        ax4.plot(
            slice_centers_m[max_r_slice_idx],
            mean_r_m[max_r_slice_idx],
            "o",
            color="#ff7f0e",
            markersize=10,
            zorder=10,
        )
        if not np.isnan(hcb_val):
            hcb_s_m = hcb_val * (s_max * scale)
            ax4.axvline(
                hcb_s_m,
                color="#2ca02c",
                linestyle=":",
                linewidth=1.5,
                alpha=0.7,
                label=f"HCB = {hcb_m:.1f} m",
            )
        ax4.set_xlabel("Arc-length $s$ (m)", fontsize=11)
        ax4.set_ylabel("Mean radial distance $r$ (m)", fontsize=11)
        ax4.set_title(
            "Max crown radius detection", fontsize=13, fontweight="bold", pad=8
        )
        ax4.legend(fontsize=8, loc="upper right")
        ax4.text(
            0.02,
            0.97,
            "(d)",
            transform=ax4.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )

        species_display = species.replace("_", " ")
        fig.suptitle(
            f"{species_display}  |  {height_bin} m  |  "
            f"H = {height_m:.1f} m  |  tree {tree_id}  |  "
            f"{len(cloud):,} pts\n"
            f"HuV = {hull_volume:.2f} m³  |  CrR = {max_crown_r_m:.2f} m  |  HCB = {hcb_m:.1f} m",
            fontsize=14,
            fontweight="bold",
            y=0.99,
        )

        fig.patch.set_facecolor("white")
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        species_safe = species.replace(" ", "_")
        fname = f"{species_safe}_{height_bin}m_tree_{tree_id_str}.pdf"
        fig.savefig(
            audit_dir / fname,
            format="pdf",
            bbox_inches="tight",
            dpi=200,
        )
        plt.close(fig)
        total_figs += 1

    print(f"\nCrown audit: {total_figs} figures saved to {audit_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["spine_audit"],
        help="Which figures to generate (1, 2, svd_axes, hjsd, crown_mae, spine_comparison, spine_audit, crown_audit)",
    )
    parser.add_argument(
        "--experiment_dir", default="experiments/transformer-8-512-4096"
    )
    parser.add_argument("--data_path", default="./data/preprocessed-4096")
    parser.add_argument("--output_dir", default="figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for fig_name in args.figures:
        if fig_name == "1":
            create_figure_1(output_dir=args.output_dir, seed=args.seed)
        elif fig_name == "2":
            create_figure_2(output_dir=args.output_dir, seed=args.seed)
        elif fig_name == "svd_axes":
            create_figure_svd_axes(
                experiment_dir=args.experiment_dir,
                data_path=args.data_path,
                output_dir=args.output_dir,
                seed=args.seed,
            )
        elif fig_name == "hjsd":
            create_figure_hjsd(
                experiment_dir=args.experiment_dir,
                data_path=args.data_path,
                output_dir=args.output_dir,
                seed=args.seed,
            )
        elif fig_name == "crown_mae":
            create_figure_crown_mae(
                experiment_dir=args.experiment_dir,
                data_path=args.data_path,
                output_dir=args.output_dir,
                seed=args.seed,
            )
        elif fig_name == "spine_comparison":
            create_figure_spine_comparison(
                experiment_dir=args.experiment_dir,
                data_path=args.data_path,
                output_dir=args.output_dir,
                seed=args.seed,
            )
        elif fig_name == "spine_audit":
            create_figure_spine_audit(
                experiment_dir=args.experiment_dir,
                data_path=args.data_path,
                output_dir=args.output_dir,
                seed=args.seed,
            )
        elif fig_name == "crown_audit":
            create_figure_crown_audit(
                experiment_dir=args.experiment_dir,
                data_path=args.data_path,
                output_dir=args.output_dir,
                seed=args.seed,
            )
        else:
            print(f"Unknown figure: {fig_name}")
