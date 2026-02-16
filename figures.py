"""
figures.py - Create figures for paper
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import MaxNLocator


from dataset import create_datasets

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

    # Normalize for display (no smoothing)
    density_real = hist_real / hist_real.sum()
    density_gen = hist_gen / hist_gen.sum()
    vmax = max(density_real.max(), density_gen.max())

    # Bin centers for axis labels
    r_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    z_centers = 0.5 * (height_edges[:-1] + height_edges[1:])

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
            cmap="YlGnBu",
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
        # Two perpendicular axes in the plane
        e1 = Vt[1]
        e2 = Vt[2]
        x = centered @ e1
        y = centered @ e2
        r = np.sqrt(x**2 + y**2)
        return x, y, r

    x_real, y_real, r_real = _project_topdown(real_cloud)
    x_gen, y_gen, r_gen = _project_topdown(gen_cloud)

    # Compute overall radial percentiles
    percentiles = [50, 75, 98]
    real_pcts = {p: np.percentile(r_real, p) for p in percentiles}
    gen_pcts = {p: np.percentile(r_gen, p) for p in percentiles}

    # --- Plot: 3 subfigures ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = {50: "p50 (median)", 75: "p75", 98: "p98"}
    mae_keys = {50: "crown_mae_p50", 75: "crown_mae_p75", 98: "crown_mae_p98"}

    # Shared axis limits
    max_r = max(r_real.max(), r_gen.max()) * 1.15
    rng = (-max_r, max_r)

    for ax, pct in zip(axes, percentiles):
        # Scatter point clouds (subsample for clarity)
        rng_gen = np.random.default_rng(seed)
        n_show = min(2000, len(x_real), len(x_gen))
        idx_r = rng_gen.choice(len(x_real), n_show, replace=False)
        idx_g = rng_gen.choice(len(x_gen), n_show, replace=False)

        ax.scatter(
            x_real[idx_r], y_real[idx_r],
            s=1, alpha=0.25, color="#1f77b4", label="Real", rasterized=True,
        )
        ax.scatter(
            x_gen[idx_g], y_gen[idx_g],
            s=1, alpha=0.25, color="#ff7f0e", label="Generated", rasterized=True,
        )

        # Circles for percentile radii
        theta = np.linspace(0, 2 * np.pi, 200)
        r_r = real_pcts[pct]
        r_g = gen_pcts[pct]
        ax.plot(
            r_r * np.cos(theta), r_r * np.sin(theta),
            color="#1f77b4", linewidth=2, linestyle="-",
            label=f"Real {labels[pct]} = {r_r:.3f}",
        )
        ax.plot(
            r_g * np.cos(theta), r_g * np.sin(theta),
            color="#ff7f0e", linewidth=2, linestyle="--",
            label=f"Gen {labels[pct]} = {r_g:.3f}",
        )

        # Stem marker
        ax.plot(0, 0, "k+", markersize=10, markeredgewidth=2)

        ax.set_xlim(rng)
        ax.set_ylim(rng)
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
        "real_p50": float(real_pcts[50]),
        "real_p75": float(real_pcts[75]),
        "real_p98": float(real_pcts[98]),
        "gen_p50": float(gen_pcts[50]),
        "gen_p75": float(gen_pcts[75]),
        "gen_p98": float(gen_pcts[98]),
    }
    meta_path = output_dir / "figure_crown_mae.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata: {meta_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["1", "2", "hjsd", "crown_mae"],
        help="Which figures to generate (1, 2, hjsd, crown_mae)",
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
        else:
            print(f"Unknown figure: {fig_name}")
