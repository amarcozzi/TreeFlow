# model.py
import torch
from torch import nn
import torch.nn.functional as F
import math


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
                      -1 is used for padding if fewer than nsample points are found.
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, device=device).view(1, 1, N).repeat([B, S, 1])

    # Calculate squared distances between each query point and all points
    sqrdists = torch.cdist(new_xyz, xyz, p=2.0) ** 2

    # Mask out points that are outside the radius
    group_idx[sqrdists > radius ** 2] = N

    # Sort by distance and take the first nsample points
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # In case a centroid has no neighbors, its first neighbor is itself
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

    if idx.dim() == 2:  # (B, S)
        new_points = points[batch_indices, idx, :]
    elif idx.dim() == 3:  # (B, S, K)
        new_points = points[batch_indices, idx, :]
    else:
        raise ValueError("idx must be 2D or 3D")

    return new_points


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
            nn.Linear(time_embed_dim, in_channels)
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x, t_embed):
        # x: (B, C_in, N)
        # t_embed: (B, D_t)

        # Project time embedding and reshape for broadcasting
        t_embed = self.time_mlp(t_embed).unsqueeze(-1)  # (B, C_in, 1)

        # Condition the input features
        x = x + t_embed

        # Apply convolution
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class PointNetSetAbstraction(nn.Module):
    """PointNet Set Abstraction (SA) module (Single-Scale Grouping)."""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, time_embed_dim):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        self.mlp_convs = nn.ModuleList()

        # The input to the first MLP is the channel dimension of the `points` tensor
        # PLUS the 3 dimensions of the normalized local coordinates (grouped_xyz_norm).
        last_channel = in_channel + 3

        for out_channel in mlp:
            self.mlp_convs.append(ConditionalConv1d(last_channel, out_channel, 1, time_embed_dim))
            last_channel = out_channel

    def forward(self, xyz, points, t_embed):
        # xyz: (B, N, 3) - coordinates
        # points: (B, C, N) - features
        # t_embed: (B, D_t) - time embedding

        # 1. Sampling
        if self.npoint is not None and self.npoint < xyz.shape[1]:
            new_xyz_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, new_xyz_idx)
        else:
            new_xyz = xyz  # Use all points as centroids if npoint is None or not smaller

        # 2. Grouping
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, 3)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

        if points is not None:
            grouped_points = index_points(points.transpose(1, 2), idx)  # (B, npoint, nsample, C)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # (B, npoint, nsample, C+3)
        else:
            new_points = grouped_xyz_norm

        # 3. Mini-PointNet (feature extraction)
        new_points = new_points.permute(0, 3, 1, 2)  # (B, C+3, npoint, nsample)
        for conv in self.mlp_convs:
            # Reshape for Conv1d: (B, C, npoint * nsample)
            B, C, S, K = new_points.shape
            new_points = conv(new_points.reshape(B, C, S * K), t_embed)
            new_points = new_points.reshape(B, -1, S, K)

        # Max pooling over the neighborhood points
        new_points = torch.max(new_points, 3)[0]  # (B, C', npoint)

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
        # xyz1: (B, N, 3) - points to interpolate to (e.g., original points)
        # xyz2: (B, S, 3) - points to interpolate from (e.g., subsampled points)
        # points1: (B, C1, N) - skip-link features from xyz1
        # points2: (B, C2, S) - features from xyz2
        # t_embed: (B, D_t) - time embedding

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:  # Global feature
            interpolated_points = points2.repeat(1, 1, N)
        else:
            # Inverse distance weighted interpolation
            dists = torch.cdist(xyz1, xyz2, p=2.0) ** 2
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # k=3 nearest neighbors

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_points = torch.sum(index_points(points2.transpose(1, 2), idx) * weight.unsqueeze(-1), dim=2)
            interpolated_points = interpolated_points.transpose(1, 2)  # (B, C2, N)

        # Concatenate skip-link features
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points

        # Apply MLPs
        for conv in self.mlp_convs:
            new_points = conv(new_points, t_embed)

        return new_points


class PointNet2UnetForFlowMatching(nn.Module):
    def __init__(self, time_embed_dim=128):
        super().__init__()

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.GELU()
        )

        # --- Encoder (Set Abstraction) ---
        # input_channels = 3 for XYZ
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128],
                                          time_embed_dim=time_embed_dim)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                          time_embed_dim=time_embed_dim)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512, 1024], time_embed_dim=time_embed_dim)

        # --- Decoder (Feature Propagation) ---
        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 256, mlp=[256, 256], time_embed_dim=time_embed_dim)
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128], time_embed_dim=time_embed_dim)
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 3, mlp=[128, 128, 128],
                                              time_embed_dim=time_embed_dim)  # Added initial features

        # --- Prediction Head ---
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 3, 1)  # Output 3 channels for XYZ velocity
        )

    def forward(self, x_t, t):
        """
        Forward pass for the PointNet++ U-Net.

        Args:
            x_t (torch.Tensor): Input point cloud at time t, shape (B, N, 3).
            t (torch.Tensor): Timestep tensor, shape (B,).

        Returns:
            torch.Tensor: Predicted velocity field, shape (B, 3, N).
        """
        # Ensure input is (B, N, C)
        if x_t.dim() == 3 and x_t.shape[2] != 3:
            x_t = x_t.transpose(1, 2)

        xyz = x_t
        l0_points = xyz.transpose(1, 2)  # (B, 3, N) - save for skip connection

        # Generate time embedding
        t_embed = self.time_mlp(t)

        # --- Encoder Path ---
        l1_xyz, l1_points = self.sa1(xyz, l0_points, t_embed)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, t_embed)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, t_embed)  # l3_xyz is None

        # --- Decoder Path ---
        # Note: The 'points1' are the skip connections from the corresponding encoder level
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points, t_embed)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points, t_embed)
        l0_points = self.fp1(xyz, l1_xyz, l0_points, l1_points, t_embed)

        # --- Final Prediction ---
        pred_velocity = self.head(l0_points)

        return pred_velocity  # (B, 3, N)


if __name__ == '__main__':
    # --- Example Usage ---

    # Model Hyperparameters
    BATCH_SIZE = 4
    NUM_POINTS = 2048
    TIME_EMBED_DIM = 256

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the model and move it to the device
    model = PointNet2UnetForFlowMatching(time_embed_dim=TIME_EMBED_DIM).to(device)

    print(f"Model created successfully. Parameter count: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Create dummy input data
    # Noisy point cloud at time t
    x_t_input = torch.randn(BATCH_SIZE, NUM_POINTS, 3, device=device)
    # Timesteps (random values between 0 and 1)
    t_input = torch.rand(BATCH_SIZE, device=device)

    # Perform a forward pass
    # try:
    predicted_velocity = model(x_t_input, t_input)

    # Print shapes to verify
    print("\n--- Verification ---")
    print(f"Input point cloud shape: {x_t_input.shape}")
    print(f"Input time shape:          {t_input.shape}")
    print(f"Predicted velocity shape:  {predicted_velocity.shape}")

    # Check output shape
    expected_shape = (BATCH_SIZE, 3, NUM_POINTS)
    assert predicted_velocity.shape == expected_shape, \
        f"Shape mismatch! Expected {expected_shape}, but got {predicted_velocity.shape}"

    print("\nForward pass successful and output shape is correct!")

    # except Exception as e:
    #     print(f"\nAn error occurred during the forward pass: {e}")