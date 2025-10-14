"""
TreeFlow PointNet++ U-Net for Flow Matching on 3D Tree Scans from FOR20K Dataset
"""

import torch
from torch import nn
import torch.nn.functional as F
import math


# --- UTILITY FUNCTIONS ---

def farthest_point_sample(xyz, npoint):
    """
    Iterative farthest point sampling algorithm.

    Args:
        xyz (torch.Tensor): Point cloud data, shape (B, N, 3)
        npoint (int): Number of points to sample

    Returns:
        torch.Tensor: Indices of sampled points, shape (B, npoint)
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Finds all points within a given radius for each query point.

    Args:
        radius (float): Radius of the ball.
        nsample (int): Maximum number of points to sample in each local region.
        xyz (torch.Tensor): All points, shape (B, N, 3).
        new_xyz (torch.Tensor): Query points (centroids), shape (B, S, 3).

    Returns:
        torch.Tensor: Grouped indices, shape (B, S, nsample).
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = torch.cdist(new_xyz, xyz) ** 2
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def index_points(points, idx):
    """
    Index into a points tensor using indices.

    Args:
        points (torch.Tensor): Input features, shape (B, N, C).
        idx (torch.Tensor): Index tensor, shape (B, S) or (B, S, K).

    Returns:
        torch.Tensor: Indexed features, shape (B, S, C) or (B, S, K, C).
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


# --- CORE MODULES ---

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding module."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings


class ConditionalConv1d(nn.Module):
    """A Conv1d layer conditioned on a time embedding."""

    def __init__(self, in_channels, out_channels, kernel_size, time_embed_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False)
        self.time_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_embed_dim, out_channels)
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x, t_embed):
        # x: (B, C_in, N)
        # t_embed: (B, D_t)
        x = self.conv(x)

        # Add time conditioning BEFORE normalization
        t_embed = self.time_mlp(t_embed).unsqueeze(-1)  # (B, C_out, 1)
        x = x + t_embed

        x = self.norm(x)
        x = self.activation(x)
        return x


class PointNetSetAbstraction(nn.Module):
    """PointNet Set Abstraction (SA) module."""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, time_embed_dim):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        last_channel = in_channel + 3  # +3 for relative coordinates

        for out_channel in mlp:
            self.mlp_convs.append(ConditionalConv1d(last_channel, out_channel, 1, time_embed_dim))
            last_channel = out_channel

    def forward(self, xyz, points, t_embed):
        # xyz: (B, N, 3) - coordinates
        # points: (B, C, N) or None - features
        # t_embed: (B, D_t) - time embedding

        B, N, C_xyz = xyz.shape
        if self.npoint is not None:
            if self.npoint < N:
                new_xyz_idx = farthest_point_sample(xyz, self.npoint)
                new_xyz = index_points(xyz, new_xyz_idx)
            else:
                new_xyz = xyz  # Use all points if npoint is not smaller
        else:  # Global abstraction
            new_xyz = torch.zeros(B, 1, C_xyz, device=xyz.device)

        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

        if points is not None:
            grouped_points = index_points(points.transpose(1, 2), idx)
            # [B, C_in + 3, npoint, nsample]
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm

        new_points = new_points.permute(0, 3, 1, 2)  # (B, C_in+3, npoint, nsample)

        for i, conv in enumerate(self.mlp_convs):
            B, C, S, K = new_points.shape
            new_points_flat = new_points.reshape(B, C, S * K)
            new_points_conv = conv(new_points_flat, t_embed)
            new_points = new_points_conv.reshape(B, -1, S, K)

        new_points = torch.max(new_points, 3)[0]  # [B, C_out, npoint]

        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """PointNet Feature Propagation (FP) module."""

    def __init__(self, in_channel, mlp, time_embed_dim):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(ConditionalConv1d(last_channel, out_channel, 1, time_embed_dim))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2, t_embed):
        # xyz1: (B, N, 3) - points to interpolate to
        # xyz2: (B, S, 3) - points to interpolate from
        # points1: (B, C1, N) - skip-link features from xyz1
        # points2: (B, C2, S) - features from xyz2
        # t_embed: (B, D_t) - time embedding

        B, N, C = xyz1.shape
        S = xyz2.shape[1]

        if S == 1:
            interpolated_points = points2.repeat(1, 1, N)
        else:
            dists = torch.cdist(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # k=3 NN

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2.transpose(1, 2), idx) * weight.unsqueeze(-1), dim=2)
            interpolated_points = interpolated_points.transpose(1, 2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points

        for conv in self.mlp_convs:
            new_points = conv(new_points, t_embed)

        return new_points


# --- FINAL MODEL ---

class PointNet2UnetForFlowMatching(nn.Module):
    """
    A U-Net architecture based on PointNet++ for predicting a velocity field.
    This model is designed to handle the variable-sized point clouds of tree scans
    by processing inputs with a hierarchical feature learning approach.
    """

    def __init__(self, time_embed_dim=256):
        super().__init__()
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.GELU()
        )

        # --- ENCODER ---
        # Based on a mean of ~28k points per tree, these downsampling stages are appropriate.
        # SA1: Learns initial features from local neighborhoods. Input is raw xyz.
        self.sa1 = PointNetSetAbstraction(npoint=4096, radius=0.1, nsample=32,
                                          in_channel=0, mlp=[32, 32, 64],
                                          time_embed_dim=time_embed_dim)

        # SA2: Downsamples further and increases feature dimension.
        self.sa2 = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=32,
                                          in_channel=64, mlp=[64, 64, 128],
                                          time_embed_dim=time_embed_dim)

        # SA3: Captures more global context.
        self.sa3 = PointNetSetAbstraction(npoint=256, radius=0.4, nsample=32,
                                          in_channel=128, mlp=[128, 128, 256],
                                          time_embed_dim=time_embed_dim)

        # SA4 (Bottleneck): Global abstraction to a single feature vector.
        self.sa4 = PointNetSetAbstraction(npoint=None, radius=0.8, nsample=256,
                                          in_channel=256, mlp=[256, 512, 1024],
                                          time_embed_dim=time_embed_dim)

        # --- DECODER ---
        # FP layers upsample features, using skip connections from the encoder.
        self.fp4 = PointNetFeaturePropagation(in_channel=1024 + 256, mlp=[512, 256], time_embed_dim=time_embed_dim)
        self.fp3 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128], time_embed_dim=time_embed_dim)
        self.fp2 = PointNetFeaturePropagation(in_channel=128 + 64, mlp=[128, 128], time_embed_dim=time_embed_dim)

        # FP1: Final propagation to original point cloud resolution.
        # The skip connection uses the original coordinates (3 channels).
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 3, mlp=[128, 64, 64], time_embed_dim=time_embed_dim)

        # --- Final Head ---
        # Predicts the 3D velocity vector for each point.
        self.head = nn.Sequential(
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, x_t, t):
        # x_t: (B, 3, N) - Noisy point cloud at time t
        # t: (B,) - Time step

        # Convert to (B, N, 3) for coordinate-based operations
        xyz = x_t.transpose(1, 2)

        # Process time embedding
        t_embed = self.time_mlp(t)

        # --- Encoder Path ---
        # l0_points is the raw input coordinates, used for the final skip connection
        l0_points = x_t

        # Layer 1: Pass xyz with no features (points=None) as per in_channel=0
        l1_xyz, l1_features = self.sa1(xyz, None, t_embed)

        # Layer 2
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features, t_embed)

        # Layer 3
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features, t_embed)

        # Bottleneck Layer 4
        l4_xyz, l4_features = self.sa4(l3_xyz, l3_features, t_embed)

        # --- Decoder Path ---
        # Layer 3: Propagate from bottleneck to l3 resolution, skip-connect with l3_features
        dec_l3_features = self.fp4(l3_xyz, l4_xyz, l3_features, l4_features, t_embed)

        # Layer 2: Propagate from l3 to l2, skip-connect with l2_features
        dec_l2_features = self.fp3(l2_xyz, l3_xyz, l2_features, dec_l3_features, t_embed)

        # Layer 1: Propagate from l2 to l1, skip-connect with l1_features
        dec_l1_features = self.fp2(l1_xyz, l2_xyz, l1_features, dec_l2_features, t_embed)

        # Layer 0: Propagate to original resolution, skip-connect with raw coordinates
        dec_l0_features = self.fp1(xyz, l1_xyz, l0_points, dec_l1_features, t_embed)

        # --- Prediction Head ---
        # Generate the final velocity field
        pred_velocity = self.head(dec_l0_features)

        return pred_velocity


if __name__ == '__main__':
    # --- Verification ---
    # This block allows you to run the file directly to test the model
    BATCH_SIZE = 4
    NUM_POINTS = 28000  # A realistic number based on your dataset's mean
    TIME_EMBED_DIM = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the model and move it to the device
    model = PointNet2UnetForFlowMatching(time_embed_dim=TIME_EMBED_DIM).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created successfully. Parameter count: {num_params / 1e6:.2f}M")

    # Create dummy input tensors
    x_t_input = torch.randn(BATCH_SIZE, 3, NUM_POINTS, device=device)
    t_input = torch.rand(BATCH_SIZE, device=device)

    try:
        # Perform a forward pass
        predicted_velocity = model(x_t_input, t_input)

        print("\n--- Forward Pass Verification ---")
        print(f"Input point cloud shape: {x_t_input.shape}")
        print(f"Input time shape:          {t_input.shape}")
        print(f"Predicted velocity shape:  {predicted_velocity.shape}")

        # Check if the output shape is as expected
        expected_shape = (BATCH_SIZE, 3, NUM_POINTS)
        assert predicted_velocity.shape == expected_shape, \
            f"Shape mismatch! Expected {expected_shape}, but got {predicted_velocity.shape}"

        print("\n✓ Forward pass successful! Output shape is correct.")

    except Exception as e:
        print(f"\nError during forward pass: {e}")
        import traceback

        traceback.print_exc()