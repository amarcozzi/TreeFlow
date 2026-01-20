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
        preprocessed_version="raw",
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
        preprocessed_version="raw",
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
    # No title - will be labeled in figure 2c
    ax_a.view_init(elev=20, azim=45)
    ax_a.set_box_aspect([1, 1, 1])

    # Keep minimal axis elements - light box frame
    ax_a.set_xticklabels([])
    ax_a.set_yticklabels([])
    ax_a.set_zticklabels([])
    ax_a.tick_params(axis='both', which='both', length=0)

    # Light gray panes and edges for subtle 3D box
    ax_a.xaxis.pane.fill = False
    ax_a.yaxis.pane.fill = False
    ax_a.zaxis.pane.fill = False
    ax_a.xaxis.pane.set_edgecolor('#CCCCCC')
    ax_a.yaxis.pane.set_edgecolor('#CCCCCC')
    ax_a.zaxis.pane.set_edgecolor('#CCCCCC')
    ax_a.xaxis.line.set_color('#AAAAAA')
    ax_a.yaxis.line.set_color('#AAAAAA')
    ax_a.zaxis.line.set_color('#AAAAAA')
    ax_a.grid(False)

    # White background
    fig_a.patch.set_facecolor('white')

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

    # No title - will be labeled in figure 2c
    ax_b.view_init(elev=20, azim=45)

    # Keep minimal axis elements - light box frame
    ax_b.set_xticklabels([])
    ax_b.set_yticklabels([])
    ax_b.set_zticklabels([])
    ax_b.tick_params(axis='both', which='both', length=0)

    # Light gray panes and edges for subtle 3D box
    ax_b.xaxis.pane.fill = False
    ax_b.yaxis.pane.fill = False
    ax_b.zaxis.pane.fill = False
    ax_b.xaxis.pane.set_edgecolor('#CCCCCC')
    ax_b.yaxis.pane.set_edgecolor('#CCCCCC')
    ax_b.zaxis.pane.set_edgecolor('#CCCCCC')
    ax_b.xaxis.line.set_color('#AAAAAA')
    ax_b.yaxis.line.set_color('#AAAAAA')
    ax_b.zaxis.line.set_color('#AAAAAA')
    ax_b.grid(False)

    # White background
    fig_b.patch.set_facecolor('white')

    fig_b.savefig(
        output_dir / "figure_2_b.png", bbox_inches="tight", dpi=800
    )
    plt.close(fig_b)
    print(f"  Saved: {output_dir}/figure_2_b.png")

    # ==========================================
    # Figure 2c: 2D Probability Space with Flow Matching Paths
    # (Similar to Yazdani et al. Figure 1a)
    # ==========================================
    print("Creating Figure 2c: 2D probability space with flow matching paths...")

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import matplotlib.image as mpimg

    # Set seed for reproducible stochastic path
    np.random.seed(42)

    fig_c, ax_c = plt.subplots(figsize=(9, 9))

    # Define grid for probability densities
    grid_size = 500
    x_range = np.linspace(-0.5, 6.0, grid_size)
    y_range = np.linspace(-0.5, 6.0, grid_size)
    X, Y = np.meshgrid(x_range, y_range)

    # Source distribution: Simple Gaussian - positioned to leave room for inset
    source_center = np.array([1.8, 1.8])
    source_sigma = 0.55
    source_density = np.exp(
        -((X - source_center[0]) ** 2 + (Y - source_center[1]) ** 2)
        / (2 * source_sigma**2)
    )

    # Target distribution: Complex multi-modal distribution
    # Many modes with varying shapes to show complexity of learned distribution
    target_centers = [
        (4.0, 4.2),   # Main mode
        (4.7, 4.8),   # Secondary mode
        (3.4, 4.8),   # Third mode
        (4.6, 3.6),   # Fourth mode
        (3.6, 3.6),   # Fifth mode (smaller)
        (5.0, 4.2),   # Sixth mode (smaller)
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
            -((X - center[0]) ** 2 / (2 * sigmas[0]**2) +
              (Y - center[1]) ** 2 / (2 * sigmas[1]**2))
        )

    # Create subtle background gradient
    combined_density = 0.25 * source_density + 0.75 * target_density
    combined_density = gaussian_filter(combined_density, sigma=4)

    # Plot filled contours for subtle background
    ax_c.contourf(
        X, Y, combined_density,
        levels=20,
        cmap="Reds",
        alpha=0.12,
    )

    # Plot source distribution contours (blue)
    source_levels = np.linspace(0.12, 0.92, 7) * source_density.max()
    cs_source = ax_c.contour(
        X, Y, source_density,
        levels=source_levels,
        colors="#1565C0",
        linewidths=1.6,
        alpha=0.9,
    )

    # Plot target distribution contours (red/dark red)
    target_levels = np.linspace(0.08, 0.92, 8) * target_density.max()
    cs_target = ax_c.contour(
        X, Y, target_density,
        levels=target_levels,
        colors="#C62828",
        linewidths=1.6,
        alpha=0.9,
    )

    # Define source and target points for paths
    x0 = np.array([1.5, 1.5])    # Source point (noise sample)
    x1 = np.array([3.8, 4.0])    # Target point (tree sample)

    # ==========================================
    # Add inset images - positioned in corners, behind the contours
    # ==========================================
    # Inset for source (noise) - in lower-left corner
    try:
        img_noise = mpimg.imread(output_dir / "figure_2_a.png")
        imagebox_noise = OffsetImage(img_noise, zoom=0.09)
        ab_noise = AnnotationBbox(
            imagebox_noise,
            (-0.2, -0.2),  # Lower-left corner
            frameon=False,
            zorder=0,  # Behind contours
        )
        ax_c.add_artist(ab_noise)
    except Exception as e:
        print(f"  Warning: Could not load figure_2_a.png for inset: {e}")

    # Inset for target (tree) - in upper-right corner
    try:
        img_tree = mpimg.imread(output_dir / "figure_2_b.png")
        imagebox_tree = OffsetImage(img_tree, zoom=0.09)
        ab_tree = AnnotationBbox(
            imagebox_tree,
            (5.5, 5.5),  # Upper-right corner
            frameon=False,
            zorder=0,  # Behind contours
        )
        ax_c.add_artist(ab_tree)
    except Exception as e:
        print(f"  Warning: Could not load figure_2_b.png for inset: {e}")

    # ==========================================
    # Training path: Linear interpolation x_t = (1-t)*x_0 + t*x_1
    # ==========================================
    t_train = np.linspace(0, 1, 100)
    train_path = np.array([(1 - t) * x0 + t * x1 for t in t_train])

    ax_c.plot(
        train_path[:, 0], train_path[:, 1],
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
    # Inference path: Moderately inefficient ODE stepping
    # Shows that inference deviates somewhat from the optimal straight line
    # ==========================================
    n_steps = 8
    t_infer = np.linspace(0, 1, n_steps + 1)

    # Create a path with moderate deviation from straight line
    infer_path = [x0.copy()]
    current_pos = x0.copy()

    np.random.seed(42)
    for i in range(1, len(t_infer)):
        t_curr = t_infer[i]
        dt = t_infer[i] - t_infer[i-1]

        # Base velocity direction
        base_velocity = x1 - x0

        # Perpendicular direction for deviation
        perp = np.array([-base_velocity[1], base_velocity[0]])
        perp = perp / np.linalg.norm(perp)

        # Moderate structured deviation - curved path
        # Deviation peaks in the middle of the path
        deviation_magnitude = 0.2 * np.sin(np.pi * t_curr)

        # Small random component
        random_component = 0.05 * np.random.randn(2)

        # Update position
        current_pos = current_pos + base_velocity * dt + perp * deviation_magnitude * dt * 5 + random_component
        infer_path.append(current_pos.copy())

    infer_path = np.array(infer_path)
    # Ensure endpoints are correct
    infer_path[0] = x0
    infer_path[-1] = x1

    # Plot path segments connecting the steps
    ax_c.plot(
        infer_path[:, 0], infer_path[:, 1],
        color="#FF8F00",
        linewidth=2.5,
        linestyle="--",
        label="Inference: ODE integration",
        zorder=5,
    )

    # Plot step markers
    ax_c.scatter(
        infer_path[1:-1, 0], infer_path[1:-1, 1],
        color="#FF8F00",
        s=60,
        zorder=7,
        edgecolors="white",
        linewidths=1.5,
        marker="o",
    )

    # ==========================================
    # Mark source and target points
    # ==========================================
    # Source point (x_0)
    ax_c.scatter(
        [x0[0]], [x0[1]],
        color="#1565C0",
        s=220,
        marker="o",
        zorder=10,
        edgecolors="white",
        linewidths=2.5,
    )

    # Target point (x_1)
    ax_c.scatter(
        [x1[0]], [x1[1]],
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
        1.5, 2.4,
        r"$X_0$ (Source)" + "\n" + r"$\mathcal{N}(0, I)$",
        fontsize=11,
        color="#1565C0",
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    # Label for target distribution
    ax_c.text(
        4.0, 2.8,
        r"$X_1$ (Target)" + "\n" + r"$p_{\mathrm{data}}$",
        fontsize=11,
        color="#C62828",
        fontweight="bold",
        ha="center",
        va="top",
    )

    # Axis labels
    ax_c.set_xlabel("X-axis", fontsize=12)
    ax_c.set_ylabel("Y-axis", fontsize=12)

    # Set axis limits with padding for corner insets
    ax_c.set_xlim(-1.5, 6.5)
    ax_c.set_ylim(-1.5, 6.5)

    # Add legend in bottom right
    legend = ax_c.legend(
        loc="lower right",
        fontsize=10,
        framealpha=0.95,
        edgecolor="gray",
    )

    # Clean styling
    ax_c.set_aspect("equal")

    # White background
    fig_c.patch.set_facecolor("white")
    ax_c.set_facecolor("white")

    fig_c.savefig(
        output_dir / "figure_2_c.png", format="png", bbox_inches="tight", dpi=800
    )
    plt.close(fig_c)
    print(f"  Saved: {output_dir}/figure_2_c.png")

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    create_figure_1()
    create_figure_2()
