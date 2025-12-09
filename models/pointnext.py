"""
treeflow/models/pointnext.py

Conditional PointNeXt U-Net for Flow Matching.
Uses Dual (Factorized) FiLM to process Time and Context independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 1. Geometric Utilities
# ==============================================================================


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, C, N]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, C, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long, device=device)
        .view(view_shape)
        .repeat(repeat_shape)
    )

    # Advanced indexing
    new_points = points[batch_indices, :, idx]

    # Transpose and ensure contiguous memory for downstream .view() calls
    return new_points.transpose(1, 2).contiguous()


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src: [B, C, N]
    dst: [B, C, M]
    """
    B, C, N = src.shape
    _, _, M = dst.shape

    # Matrix multiplication: [B, N, C] x [B, C, M] -> [B, N, M]
    dist = -2 * torch.matmul(src.transpose(1, 2), dst)

    # Add squared norms
    # src is [B, C, N], so we sum over dim 1 (Channels) to get norm per point
    dist += torch.sum(src**2, dim=1).view(B, N, 1)
    dist += torch.sum(dst**2, dim=1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, 3, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, C, N = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, :, farthest].view(B, 3, 1)
        dist = torch.sum((xyz - centroid) ** 2, dim=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, 3, N]
        new_xyz: query points, [B, 3, S]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, C, N = xyz.shape
    _, _, S = new_xyz.shape

    sqrdists = square_distance(new_xyz, xyz)

    group_idx = (
        torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, S, 1)
    )
    group_idx[sqrdists > radius**2] = N  # Sentinel

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)

    # Safety Check: If a point has NO neighbors (group_first is still N),
    # replace it with 0 to prevent CUDA crash.
    mask_empty = group_first == N
    group_first[mask_empty] = 0

    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


# ==============================================================================
# 2. Neural Building Blocks (PointNeXt + Dual FiLM)
# ==============================================================================


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class InvResMLP(nn.Module):
    """
    Inverted Residual MLP Block with Dual FiLM.
    Processes Time and Context separately and sums their modulation parameters.
    """

    def __init__(self, in_dim, expansion=4):
        super().__init__()
        self.in_dim = in_dim
        expanded_dim = in_dim * expansion

        # 1. Depthwise / Expansion
        self.conv1 = nn.Conv1d(in_dim, expanded_dim, 1)
        self.norm1 = nn.GroupNorm(8, expanded_dim)
        self.act = nn.SiLU()

        # 2. Projection / Contraction
        self.conv2 = nn.Conv1d(expanded_dim, in_dim, 1)
        self.norm2 = nn.GroupNorm(8, in_dim)

        # --- Dual FiLM Heads ---
        # Time Modulation: Projects time_emb -> (scale, shift)
        self.time_proj = nn.Linear(in_dim, in_dim * 2)

        # Context Modulation: Projects context_emb -> (scale, shift)
        self.cond_proj = nn.Linear(in_dim, in_dim * 2)

        # Initialize to Identity
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x, t_emb, c_emb):
        # x: [B, C, N]
        # t_emb: [B, C]
        # c_emb: [B, C]

        identity = x

        # Expansion
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        # Contraction
        x = self.conv2(x)
        x = self.norm2(x)

        # --- Apply Dual FiLM ---
        # Predict parameters
        scale_t, shift_t = self.time_proj(t_emb).chunk(2, dim=1)
        scale_c, shift_c = self.cond_proj(c_emb).chunk(2, dim=1)

        # Combine (Additive modulation)
        total_scale = scale_t + scale_c
        total_shift = shift_t + shift_c

        # Modulate: [B, C, 1]
        scale = total_scale.unsqueeze(-1)
        shift = total_shift.unsqueeze(-1)

        x = x * (1 + scale) + shift

        return x + identity


class SetAbstraction(nn.Module):
    """Downsampling + Feature Extraction"""

    def __init__(self, in_dim, out_dim, npoint, radius, nsample, expansion=4):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        # Feature reduction
        self.mlp_convs = nn.Sequential(
            nn.Conv2d(in_dim + 3, out_dim, 1), nn.GroupNorm(8, out_dim), nn.SiLU()
        )

        # PointNeXt Block
        self.block = InvResMLP(out_dim, expansion=expansion)

    def forward(self, xyz, points, t_emb, c_emb):
        B, C, N = xyz.shape

        # 1. FPS (Downsampling)
        if N > self.npoint:
            idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, idx)
        else:
            new_xyz = xyz

        # 2. Grouping
        # new_xyz: [B, 3, S]
        # S can be self.npoint OR N (if N < npoint)
        S = new_xyz.shape[2]

        group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)

        # Use dynamic shape S instead of self.npoint
        grouped_xyz = index_points(xyz, group_idx.reshape(B, -1)).reshape(
            B, 3, S, self.nsample
        )
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, 3, S, 1)

        if points is not None:
            grouped_points = index_points(points, group_idx.reshape(B, -1)).reshape(
                B, points.shape[1], S, self.nsample
            )
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=1)
        else:
            new_points = grouped_xyz_norm

        # 3. Pooling
        new_points = self.mlp_convs(new_points)
        new_points = torch.max(new_points, 3)[0]  # [B, Out, S]

        # 4. Refinement (Dual FiLM)
        new_points = self.block(new_points, t_emb, c_emb)

        return new_xyz, new_points


class FeaturePropagation(nn.Module):
    """Upsampling + Feature Fusion"""

    def __init__(self, in_dim, skip_dim, out_dim, expansion=4):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim + skip_dim, out_dim, 1),
            nn.GroupNorm(8, out_dim),
            nn.SiLU(),
        )

        self.block = InvResMLP(out_dim, expansion=expansion)

    def forward(self, xyz1, points1, xyz2, points2, t_emb, c_emb):
        # xyz1: Dense coords [B, 3, N]
        # points1: Dense Skip features [B, C1, N]
        # xyz2: Sparse coords [B, 3, S]
        # points2: Sparse features [B, C2, S]

        B, C, N = xyz1.shape
        _, _, S = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, 1, N)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_points = torch.sum(
                index_points(points2, idx.reshape(B, -1)).reshape(B, -1, N, 3)
                * weight.view(B, 1, N, 3),
                dim=3,
            )

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points

        new_points = self.mlp(new_points)
        # Dual FiLM
        new_points = self.block(new_points, t_emb, c_emb)

        return new_points


# ==============================================================================
# 3. Main Model
# ==============================================================================


class FlowMatchingPointNeXt(nn.Module):
    def __init__(
        self,
        model_dim=64,
        num_species=10,
        num_types=3,
        dropout=0.1,
        # Legacy args ignored
        num_layers=None,
        num_heads=None,
        max_freq=None,
        learnable_pos_encoding=None,
    ):
        super().__init__()
        self.model_dim = model_dim

        # --- Conditioning ---
        self.t_embedder = TimestepEmbedder(model_dim)
        self.species_embedder = nn.Embedding(num_species + 1, model_dim)
        self.type_embedder = nn.Embedding(num_types + 1, model_dim)
        self.height_mlp = nn.Sequential(
            nn.Linear(1, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )
        self.null_height_embed = nn.Parameter(torch.randn(1, model_dim))

        # --- PointNeXt U-Net Configuration ---
        # Encoder (Down)
        # 3 -> 64
        self.sa1 = SetAbstraction(
            in_dim=0, out_dim=model_dim, npoint=2048, radius=0.1, nsample=32
        )
        # 64 -> 128
        self.sa2 = SetAbstraction(
            in_dim=model_dim, out_dim=model_dim * 2, npoint=512, radius=0.2, nsample=32
        )
        # 128 -> 256
        self.sa3 = SetAbstraction(
            in_dim=model_dim * 2,
            out_dim=model_dim * 4,
            npoint=128,
            radius=0.4,
            nsample=32,
        )
        # 256 -> 512
        self.sa4 = SetAbstraction(
            in_dim=model_dim * 4,
            out_dim=model_dim * 8,
            npoint=32,
            radius=0.8,
            nsample=32,
        )

        # Decoder (Up)
        # 512 + 256 -> 256
        self.fp4 = FeaturePropagation(
            in_dim=model_dim * 8, skip_dim=model_dim * 4, out_dim=model_dim * 4
        )
        # 256 + 128 -> 128
        self.fp3 = FeaturePropagation(
            in_dim=model_dim * 4, skip_dim=model_dim * 2, out_dim=model_dim * 2
        )
        # 128 + 64 -> 64
        self.fp2 = FeaturePropagation(
            in_dim=model_dim * 2, skip_dim=model_dim, out_dim=model_dim
        )
        # 64 + 0 -> 64
        self.fp1 = FeaturePropagation(in_dim=model_dim, skip_dim=0, out_dim=model_dim)

        # --- Separate Projections for Dual FiLM ---
        dims = [model_dim, model_dim * 2, model_dim * 4, model_dim * 8]
        self.time_projs = nn.ModuleList([nn.Linear(model_dim, d) for d in dims])
        self.cond_projs = nn.ModuleList([nn.Linear(model_dim, d) for d in dims])

        self.final = nn.Conv1d(model_dim, 3, 1)

    def forward(self, x, t, species_idx, type_idx, height_norm, drop_mask=None):
        # x: [B, N, 3] -> Transpose to [B, 3, N]
        x = x.transpose(1, 2).contiguous()

        # 1. Conditioning
        t_global = self.t_embedder(t)

        if drop_mask is not None:
            mask = drop_mask.unsqueeze(1).float()

            s_real = self.species_embedder(species_idx)
            type_real = self.type_embedder(type_idx)
            h_real = self.height_mlp(height_norm.unsqueeze(1))

            # Nulls
            B = x.shape[0]
            s_null_idx = self.species_embedder.num_embeddings - 1
            t_null_idx = self.type_embedder.num_embeddings - 1

            s_null = self.species_embedder(
                torch.full((B,), s_null_idx, device=x.device)
            )
            type_null = self.type_embedder(
                torch.full((B,), t_null_idx, device=x.device)
            )
            h_null = self.null_height_embed.expand(B, -1)

            s_emb = s_real * (1 - mask) + s_null * mask
            type_emb = type_real * (1 - mask) + type_null * mask
            h_emb = h_real * (1 - mask) + h_null * mask
        else:
            s_emb = self.species_embedder(species_idx)
            type_emb = self.type_embedder(type_idx)
            h_emb = self.height_mlp(height_norm.unsqueeze(1))

        c_global = s_emb + type_emb + h_emb

        # Project separate conditions for each scale
        t_feats = [proj(t_global) for proj in self.time_projs]
        c_feats = [proj(c_global) for proj in self.cond_projs]

        # 2. Encoder
        l0_xyz = x
        l0_points = None

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points, t_feats[0], c_feats[0])
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, t_feats[1], c_feats[1])
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, t_feats[2], c_feats[2])
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points, t_feats[3], c_feats[3])

        # 3. Decoder
        l3_points = self.fp4(
            l3_xyz, l3_points, l4_xyz, l4_points, t_feats[2], c_feats[2]
        )
        l2_points = self.fp3(
            l2_xyz, l2_points, l3_xyz, l3_points, t_feats[1], c_feats[1]
        )
        l1_points = self.fp2(
            l1_xyz, l1_points, l2_xyz, l2_points, t_feats[0], c_feats[0]
        )
        l0_points = self.fp1(l0_xyz, None, l1_xyz, l1_points, t_feats[0], c_feats[0])

        out = self.final(l0_points)

        # Return [B, N, 3]
        return out.transpose(1, 2)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing Dual-FiLM PointNeXt...")
    model = FlowMatchingPointNeXt(model_dim=64, num_species=33, num_types=3)
    print(f"Params: {model.count_parameters():,}")

    x = torch.randn(2, 4096, 3)
    t = torch.rand(2)
    s = torch.randint(0, 5, (2,))
    tp = torch.randint(0, 2, (2,))
    h = torch.randn(2)

    out = model(x, t, s, tp, h)
    print(f"Output: {out.shape}")
    assert out.shape == x.shape
    print("✓ Passed")
