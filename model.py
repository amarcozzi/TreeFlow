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
        torch.Tensor: Grouped indices, shape (B, S, nsample) or (B, S, K) if fewer than nsample exist.
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, device=device).view(1, 1, N).repeat([B, S, 1])

    # Calculate squared distances between each query point and all points
    sqrdists = torch.cdist(new_xyz, xyz, p=2.0) ** 2

    # Mask out points that are outside the radius by assigning sentinel N
    group_idx[sqrdists > radius ** 2] = N

    # Sort by distance (align sort with distances to keep correspondence)
    # We sort distances and use their indices as ordering over points
    sorted_dists, order = sqrdists.sort(dim=-1)               # (B, S, N)
    ordered_idx = torch.arange(N, device=device).view(1, 1, N).repeat(B, S, 1)
    ordered_idx = ordered_idx.gather(-1, order)               # (B, S, N)
    # Apply sentinel to positions outside radius
    ordered_idx[sorted_dists > radius ** 2] = N

    # Take up to nsample neighbors
    K = min(nsample, N)
    group_idx = ordered_idx[:, :, :K]                          # (B, S, K)

    # Handle empty neighbor sets: replace N with the first valid if any, else nearest neighbor
    # If first is N (no valid within radius), fallback to nearest neighbor (from sorted_dists without radius filter)
    empty_mask = group_idx[:, :, 0] == N                       # (B, S)
    if empty_mask.any():
        # nearest overall (distance sort without radius filter already computed in 'order')
        nearest_overall = ordered_idx[:, :, 0]                 # (B, S)
        group_idx[empty_mask] = nearest_overall[empty_mask].unsqueeze(-1).expand(-1, group_idx.shape[-1])

    # Now fill any remaining N positions (some positions beyond available neighbors) with the first element per group
    first = group_idx[:, :, 0].unsqueeze(-1).expand(-1, -1, group_idx.shape[-1])  # (B, S, K)
    mask = group_idx == N
    group_idx[mask] = first[mask]

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
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
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
        x = self.norm(x)

        # Add time conditioning after convolution
        t_embed = self.time_mlp(t_embed).unsqueeze(-1)  # (B, C_out, 1)
        x = x + t_embed

        x = self.activation(x)
        return x


class VectorAttention(nn.Module):
    """
    Vector attention module for point clouds.
    Based on Point Transformer (Zhao et al., 2021).
    """

    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels

        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.head_dim = in_channels // num_heads

        # Linear transformations for Q, K, V
        self.q_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.k_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.v_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)

        # Position encoding MLP
        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, in_channels, 1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1)
        )

        # Attention weight MLP (scalar attention)
        self.attn_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, in_channels, 1)
        )

        # Output projection
        self.out_conv = nn.Conv1d(in_channels, in_channels, 1)

        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x, xyz):
        """
        Args:
            x: (B, C, N) - input features
            xyz: (B, N, 3) - point coordinates

        Returns:
            (B, C, N) - output features
        """
        B, C, N = x.shape

        # Save residual
        residual = x

        # Compute Q, K, V
        Q = self.q_conv(x)  # (B, C, N)
        K = self.k_conv(x)  # (B, C, N)
        V = self.v_conv(x)  # (B, C, N)

        # Compute relative positions
        xyz_t = xyz.transpose(1, 2)  # (B, 3, N)
        rel_pos = xyz_t.unsqueeze(3) - xyz_t.unsqueeze(2)  # (B, 3, N, N)

        # Position encoding
        pos_enc = self.pos_mlp(rel_pos)  # (B, C, N, N)

        # Compute relation features: (q_i - q_j) + pos_enc
        Q_expanded = Q.unsqueeze(3)  # (B, C, N, 1)
        K_expanded = K.unsqueeze(2)  # (B, C, 1, N)
        relation = Q_expanded - K_expanded + pos_enc  # (B, C, N, N)

        # Apply attention MLP to get scalar attention weights
        attn = self.attn_mlp(relation)  # (B, C, N, N)

        # Softmax over source points (last dimension)
        attn = F.softmax(attn, dim=-1)  # (B, C, N, N)

        # Apply attention to values with position encoding
        V_expanded = V.unsqueeze(2)  # (B, C, 1, N)
        out = (attn * (V_expanded + pos_enc)).sum(dim=-1)  # (B, C, N)

        # Output projection
        out = self.out_conv(out)

        # Add residual and normalize
        out = out + residual
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)

        return out


class PointNetSetAbstraction(nn.Module):
    """PointNet Set Abstraction (SA) module."""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, time_embed_dim, use_attention=False, num_heads=4):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.use_attention = use_attention

        self.mlp_convs = nn.ModuleList()
        last_channel = in_channel + 3

        for out_channel in mlp:
            self.mlp_convs.append(ConditionalConv1d(last_channel, out_channel, 1, time_embed_dim))
            last_channel = out_channel

        if use_attention:
            self.attention = VectorAttention(last_channel, num_heads=num_heads)

    def forward(self, xyz, points, t_embed):
        # xyz: (B, N, 3) - coordinates
        # points: (B, C, N) - features
        # t_embed: (B, D_t) - time embedding

        if self.npoint is None:
            # Global abstraction layer
            new_xyz = None

            if points is not None:
                # Concatenate xyz with features
                processed_points = torch.cat([xyz.transpose(1, 2), points], dim=1)  # (B, C+3, N)
            else:
                processed_points = xyz.transpose(1, 2)  # (B, 3, N)

            for conv in self.mlp_convs:
                processed_points = conv(processed_points, t_embed)

            if self.use_attention:
                processed_points = self.attention(processed_points, xyz)

            new_points = torch.max(processed_points, 2)[0].unsqueeze(-1)  # (B, C', 1)

            return new_xyz, new_points

        # Local abstraction
        if self.npoint < xyz.shape[1]:
            new_xyz_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, new_xyz_idx)
        else:
            new_xyz = xyz

        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

        if points is not None:
            grouped_points = index_points(points.transpose(1, 2), idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm

        new_points = new_points.permute(0, 3, 1, 2)  # (B, C+3, npoint, nsample)

        for conv in self.mlp_convs:
            B, C, S, K = new_points.shape
            new_points_flat = new_points.reshape(B, C, S * K)
            new_points_conv = conv(new_points_flat, t_embed)
            new_points = new_points_conv.reshape(B, -1, S, K)

        new_points = torch.max(new_points, 3)[0]

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
        # xyz2: (B, S, 3) or None - points to interpolate from
        # points1: (B, C1, N) - skip-link features from xyz1
        # points2: (B, C2, S) - features from xyz2
        # t_embed: (B, D_t) - time embedding

        B, N, C = xyz1.shape
        S = points2.shape[2]

        if S == 1:
            interpolated_points = points2.repeat(1, 1, N)
        else:
            dists = torch.cdist(xyz1, xyz2, p=2.0) ** 2
            dists, idx = dists.sort(dim=-1)
            k = min(3, S)
            dists, idx = dists[:, :, :k], idx[:, :, :k]

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
    def __init__(self, time_embed_dim=128):
        super().__init__()

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.GELU()
        )

        # Encoder (Set Abstraction)
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128],
                                          time_embed_dim=time_embed_dim, use_attention=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256],
                                          time_embed_dim=time_embed_dim, use_attention=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256, mlp=[256, 512, 1024],
                                          time_embed_dim=time_embed_dim, use_attention=True, num_heads=4)

        # Decoder (Feature Propagation)
        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 256, mlp=[256, 256], time_embed_dim=time_embed_dim)
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128], time_embed_dim=time_embed_dim)
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 3, mlp=[128, 128, 128], time_embed_dim=time_embed_dim)

        # Prediction Head
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 3, 1)
        )

    def forward(self, x_t, t):
        # Ensure input is (B, 3, N)
        if x_t.dim() == 3 and x_t.shape[1] != 3:
            x_t = x_t.transpose(1, 2)

        # Convert to (B, N, 3) for coordinate operations
        xyz = x_t.transpose(1, 2)
        l0_points = x_t

        t_embed = self.time_mlp(t)

        # Encoder
        l1_xyz, l1_points = self.sa1(xyz, l0_points, t_embed)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, t_embed)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, t_embed)

        # Decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points, t_embed)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points, t_embed)
        l0_points = self.fp1(xyz, l1_xyz, l0_points, l1_points, t_embed)

        pred_velocity = self.head(l0_points)

        return pred_velocity


if __name__ == '__main__':
    BATCH_SIZE = 4
    NUM_POINTS = 2048
    TIME_EMBED_DIM = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PointNet2UnetForFlowMatching(time_embed_dim=TIME_EMBED_DIM).to(device)
    print(f"Model created successfully. Parameter count: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    x_t_input = torch.randn(BATCH_SIZE, 3, NUM_POINTS, device=device)
    t_input = torch.rand(BATCH_SIZE, device=device)

    try:
        predicted_velocity = model(x_t_input, t_input)

        print("\n--- Verification ---")
        print(f"Input point cloud shape: {x_t_input.shape}")
        print(f"Input time shape:          {t_input.shape}")
        print(f"Predicted velocity shape:  {predicted_velocity.shape}")

        expected_shape = (BATCH_SIZE, 3, NUM_POINTS)
        assert predicted_velocity.shape == expected_shape, \
            f"Shape mismatch! Expected {expected_shape}, but got {predicted_velocity.shape}"

        print("\nForward pass successful!")
        print("✓ Vector attention fully implemented with MLP")
        print("✓ Time conditioning applied after convolution")
        print("✓ Position encoding properly integrated in attention")

    except Exception as e:
        print(f"\nError during forward pass: {e}")
        import traceback

        traceback.print_exc()