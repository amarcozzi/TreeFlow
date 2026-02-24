"""
stem_tracker.py — Stem tracking module for tree point clouds.

Extracts the polynomial spine (stem axis) from a 3D point cloud using
bottom-up density-weighted tracking, then computes cylindrical coordinates
(r, z) or (r, s) relative to the fitted spine.

Public API:
  find_trunk_base()   — vertical persistence voting for trunk base XY
  track_spine_pass()  — single bottom-up tracking pass
  compute_rz_spine()  — full pipeline → (r, z, spine_xyz)
  compute_rs_spine()  — full pipeline → (r, s, spine_xyz, poly_x, poly_y)
"""

import numpy as np
from numpy.polynomial import Polynomial
from scipy.spatial import cKDTree


# =============================================================================
# Trunk base detection
# =============================================================================


def find_trunk_base(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bandwidth: float,
    n_sub: int = 6,
) -> tuple[float, float]:
    """Find the trunk base using vertical persistence in the lower portion.

    Instead of finding the densest cluster in a thin base slab (which can
    be hijacked by a low branch), this uses the bottom ~30% of the tree
    split into n_sub sub-slices. For each candidate XY position, it counts
    how many sub-slices have nearby points. The trunk wins because it
    persists across all sub-slices; a branch only appears in one or two.

    Algorithm:
    1. Split the lower 30% of the tree into n_sub horizontal sub-slices.
    2. In each sub-slice, run mean-shift to find the densest cluster center.
    3. Build candidate set from all sub-slice centers.
    4. Score each candidate by how many sub-slices have points within
       `bandwidth` of it — this is the vertical persistence score.
    5. Among the highest-scoring candidates, pick the one closest to the
       overall median (tiebreaker favoring the geometric center).
    """
    z_min, z_max = z.min(), z.max()
    base_top = z_min + (z_max - z_min) * 0.3
    base_mask = z <= base_top
    if base_mask.sum() < 5:
        return np.median(x), np.median(y)

    xb, yb, zb = x[base_mask], y[base_mask], z[base_mask]
    sub_edges = np.linspace(zb.min(), zb.max(), n_sub + 1)

    # Collect mean-shift centers from each sub-slice
    candidates = []
    for s in range(n_sub):
        smask = (zb >= sub_edges[s]) & (zb <= sub_edges[s + 1])
        if smask.sum() < 3:
            continue
        xs, ys = xb[smask], yb[smask]
        # Mean-shift convergence within sub-slice
        cx, cy = np.median(xs), np.median(ys)
        for _ in range(20):
            dist_sq = (xs - cx) ** 2 + (ys - cy) ** 2
            w = np.exp(-dist_sq / (2 * bandwidth**2))
            w_sum = w.sum()
            if w_sum < 1e-8:
                break
            nx = np.sum(xs * w) / w_sum
            ny = np.sum(ys * w) / w_sum
            if (nx - cx) ** 2 + (ny - cy) ** 2 < 1e-10:
                break
            cx, cy = nx, ny
        candidates.append((cx, cy))

    if not candidates:
        return np.median(x), np.median(y)

    # Score each candidate by vertical persistence:
    # how many sub-slices have points within `bandwidth` of this (x,y)?
    best_score = -1
    med_x, med_y = np.median(xb), np.median(yb)
    best_cx, best_cy = candidates[0]

    for cx, cy in candidates:
        score = 0
        for s in range(n_sub):
            smask = (zb >= sub_edges[s]) & (zb <= sub_edges[s + 1])
            if smask.sum() < 1:
                continue
            dists = np.sqrt((xb[smask] - cx) ** 2 + (yb[smask] - cy) ** 2)
            if np.any(dists <= bandwidth):
                score += 1

        # Tiebreak: prefer candidate closest to median (geometric center)
        tie = -np.sqrt((cx - med_x) ** 2 + (cy - med_y) ** 2)

        if score > best_score or (
            score == best_score
            and tie > -np.sqrt((best_cx - med_x) ** 2 + (best_cy - med_y) ** 2)
        ):
            best_score = score
            best_cx, best_cy = cx, cy

    return best_cx, best_cy


# =============================================================================
# Single tracking pass
# =============================================================================


def track_spine_pass(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bin_edges: np.ndarray,
    init_x: float,
    init_y: float,
    sigma_frac: float,
    density_k: int,
    sigma_cap: float | None = None,
    max_step: float | None = None,
) -> tuple[list[float], list[float], list[float]]:
    """Single bottom-up tracking pass with adaptive sigma and density weighting.

    Parameters
    ----------
    sigma_frac : float
        Gaussian sigma as a fraction of each slice's horizontal extent.
    density_k : int
        Number of nearest neighbors for local density estimation.
    sigma_cap : float or None
        Maximum allowed sigma (absolute). Prevents sigma from growing
        in wide canopy bins. Typically set from the base slice extent.
    max_step : float or None
        Maximum allowed horizontal movement between consecutive bins.
        Prevents jumps to canopy branches.
    """
    num_bins = len(bin_edges) - 1
    current_x, current_y = init_x, init_y
    spine_x, spine_y, spine_z = [], [], []

    for i in range(num_bins):
        mask = (z >= bin_edges[i]) & (z <= bin_edges[i + 1])
        if mask.sum() < 5:
            continue

        x_bin, y_bin, z_bin = x[mask], y[mask], z[mask]

        # Adaptive sigma: fraction of this slice's horizontal extent
        extent = max(np.ptp(x_bin), np.ptp(y_bin), 1e-6)
        sigma = sigma_frac * extent
        if sigma_cap is not None:
            sigma = min(sigma, sigma_cap)

        # Proximity weights (Gaussian centered on previous slice's center)
        dist_sq = (x_bin - current_x) ** 2 + (y_bin - current_y) ** 2
        prox_w = np.exp(-dist_sq / (2 * sigma**2))

        # Local density weights: points in tight clusters (trunk) get upweighted
        n_pts = len(x_bin)
        if n_pts > density_k:
            xy = np.column_stack([x_bin, y_bin])
            tree = cKDTree(xy)
            dists, _ = tree.query(xy, k=density_k + 1)  # +1 because self is included
            mean_knn_dist = dists[:, 1:].mean(axis=1)  # exclude self (dist=0)
            density_w = 1.0 / (mean_knn_dist + 1e-8)
            density_w /= density_w.max()
        else:
            density_w = np.ones(n_pts)

        weights = prox_w * density_w
        w_sum = weights.sum()

        if w_sum > 1e-6:
            next_x = np.sum(x_bin * weights) / w_sum
            next_y = np.sum(y_bin * weights) / w_sum
        else:
            next_x = np.median(x_bin)
            next_y = np.median(y_bin)

        # Enforce max step constraint
        if max_step is not None:
            dx = next_x - current_x
            dy = next_y - current_y
            step = np.sqrt(dx**2 + dy**2)
            if step > max_step:
                scale = max_step / step
                next_x = current_x + dx * scale
                next_y = current_y + dy * scale

        spine_x.append(next_x)
        spine_y.append(next_y)
        spine_z.append(np.median(z_bin))

        current_x, current_y = next_x, next_y

    return spine_x, spine_y, spine_z


# =============================================================================
# Shared spine fitting pipeline
# =============================================================================


def _fit_spine_polynomials(
    cloud: np.ndarray,
    num_bins: int = 20,
    sigma_frac: float = 0.10,
    degree: int = 3,
    density_k: int = 16,
    n_refine: int = 1,
    max_step_frac: float = 0.05,
    outlier_mad_k: float = 2.5,
) -> tuple:
    """Shared pipeline: base detection → tracking → refinement → outlier
    rejection → polynomial fitting.

    Returns (poly_x, poly_y, z_min, z_max, init_x, init_y, spine_xyz)
    where poly_x, poly_y are numpy Polynomial objects parametrized by z,
    and spine_xyz is (M, 3) raw tracked centers (for visualization).
    """
    x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    z_min, z_max = z.min(), z.max()
    tree_height = z_max - z_min
    bin_edges = np.linspace(z_min, z_max, num_bins + 1)

    # --- 1. Density-based initialization via vertical persistence ---
    base_top = z_min + tree_height * 0.3
    base_mask = z <= base_top
    if base_mask.sum() < 5:
        base_mask = (z >= bin_edges[0]) & (z <= bin_edges[1])

    base_extent = max(np.ptp(x[base_mask]), np.ptp(y[base_mask]), 1e-6)
    init_x, init_y = find_trunk_base(
        x[base_mask],
        y[base_mask],
        z[base_mask],
        bandwidth=0.15 * base_extent,
    )

    # Sigma cap: prevent sigma from exceeding the base extent scale
    sigma_cap = sigma_frac * base_extent * 2.0

    # Max step: fraction of tree height per bin
    max_step = max_step_frac * tree_height

    # --- 2 & 3. First tracking pass ---
    spine_x, spine_y, spine_z = track_spine_pass(
        x,
        y,
        z,
        bin_edges,
        init_x,
        init_y,
        sigma_frac,
        density_k,
        sigma_cap=sigma_cap,
        max_step=max_step,
    )

    # --- 4. Iterative refinement ---
    for _ in range(n_refine):
        if len(spine_z) < 2:
            break

        actual_deg = min(degree, len(spine_z) - 1)
        p_x = Polynomial.fit(spine_z, spine_x, actual_deg)
        p_y = Polynomial.fit(spine_z, spine_y, actual_deg)

        bin_mids = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(num_bins)]
        poly_init_x = float(p_x(bin_mids[0]))
        poly_init_y = float(p_y(bin_mids[0]))

        spine_x, spine_y, spine_z = track_spine_pass(
            x,
            y,
            z,
            bin_edges,
            poly_init_x,
            poly_init_y,
            sigma_frac,
            density_k,
            sigma_cap=sigma_cap,
            max_step=max_step,
        )

    # Raw tracked centers (for visualization)
    spine_xyz = (
        np.column_stack([spine_x, spine_y, spine_z]) if spine_z else np.empty((0, 3))
    )

    # --- 5. Outlier rejection before final polynomial fit ---
    if len(spine_z) >= 4:
        sz = np.array(spine_z)
        sx = np.array(spine_x)
        sy = np.array(spine_y)

        # Fit a line (degree 1) and compute residuals
        line_x = Polynomial.fit(sz, sx, 1)
        line_y = Polynomial.fit(sz, sy, 1)
        resid = np.sqrt((sx - line_x(sz)) ** 2 + (sy - line_y(sz)) ** 2)

        # MAD-based outlier detection
        med_resid = np.median(resid)
        mad = np.median(np.abs(resid - med_resid))
        if mad > 1e-10:
            keep = resid <= med_resid + outlier_mad_k * mad
        else:
            keep = np.ones(len(sz), dtype=bool)

        spine_x = sx[keep].tolist()
        spine_y = sy[keep].tolist()
        spine_z = sz[keep].tolist()

        spine_xyz = (
            np.column_stack([spine_x, spine_y, spine_z])
            if spine_z
            else np.empty((0, 3))
        )

    # Final polynomial smoothing
    if len(spine_z) >= 2:
        actual_deg = min(degree, len(spine_z) - 1)
        poly_x = Polynomial.fit(spine_z, spine_x, actual_deg)
        poly_y = Polynomial.fit(spine_z, spine_y, actual_deg)
    else:
        # Degenerate: constant polynomials at init position
        poly_x = Polynomial([init_x])
        poly_y = Polynomial([init_y])

    return poly_x, poly_y, z_min, z_max, init_x, init_y, spine_xyz


# =============================================================================
# Public: compute_rz_spine
# =============================================================================


def compute_rz_spine(
    cloud: np.ndarray,
    num_bins: int = 20,
    sigma_frac: float = 0.10,
    degree: int = 3,
    density_k: int = 16,
    n_refine: int = 1,
    max_step_frac: float = 0.05,
    outlier_mad_k: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust stem-tracking cylindrical coordinates.

    Returns (r, z, spine_xyz) where:
      r = perpendicular distance from polynomial spine
      z = height from base (z - z_min)
      spine_xyz = (M, 3) tracked centers before final polynomial smoothing
    """
    poly_x, poly_y, z_min, z_max, init_x, init_y, spine_xyz = _fit_spine_polynomials(
        cloud,
        num_bins,
        sigma_frac,
        degree,
        density_k,
        n_refine,
        max_step_frac,
        outlier_mad_k,
    )

    x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]

    center_x = poly_x(z)
    center_y = poly_y(z)

    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    z_out = z - z_min
    return r, z_out, spine_xyz


# =============================================================================
# Public: compute_rs_spine (arc-length version)
# =============================================================================


def compute_rs_spine(
    cloud: np.ndarray,
    num_bins: int = 20,
    sigma_frac: float = 0.10,
    degree: int = 3,
    density_k: int = 16,
    n_refine: int = 1,
    max_step_frac: float = 0.05,
    outlier_mad_k: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Polynomial, Polynomial]:
    """stem-tracking cylindrical coordinates with arc-length.

    Like compute_rz_spine but returns arc-length s instead of raw z.
    The stem is parametric (poly_x(t), poly_y(t), t) where t = z.
    Arc length s is measured from z_min along the curve.

    Returns (r, s, spine_xyz, poly_x, poly_y) where:
      r = perpendicular distance from polynomial spine
      s = arc length from base along the spine curve
      spine_xyz = (M, 3) tracked centers
      poly_x, poly_y = fitted Polynomial objects (parametrized by z)
    """
    poly_x, poly_y, z_min, z_max, init_x, init_y, spine_xyz = _fit_spine_polynomials(
        cloud,
        num_bins,
        sigma_frac,
        degree,
        density_k,
        n_refine,
        max_step_frac,
        outlier_mad_k,
    )

    x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]

    # --- Perpendicular distance from polynomial curve ---
    center_x = poly_x(z)
    center_y = poly_y(z)
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # --- Arc length computation ---
    # ds/dt = sqrt(poly_x'(t)^2 + poly_y'(t)^2 + 1) where t = z
    dpx = poly_x.deriv()
    dpy = poly_y.deriv()

    # Precompute s(z) on a fine grid via cumulative trapezoidal integration
    n_grid = 500
    z_grid = np.linspace(z_min, z_max, n_grid)
    ds_dz = np.sqrt(dpx(z_grid) ** 2 + dpy(z_grid) ** 2 + 1.0)
    # Cumulative arc length at each grid point
    dz = np.diff(z_grid)
    ds_vals = 0.5 * (ds_dz[:-1] + ds_dz[1:]) * dz  # trapezoidal rule
    s_grid = np.zeros(n_grid)
    s_grid[1:] = np.cumsum(ds_vals)

    # Interpolate to get s for each point's z coordinate
    s = np.interp(z, z_grid, s_grid)

    return r, s, spine_xyz, poly_x, poly_y
