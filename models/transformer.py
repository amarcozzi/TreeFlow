"""
treeflow/models/transformer.py - Plain Transformer with token-prepend conditioning
and U-ViT skip connections for flow matching on 3D point clouds.

Architecture:
  - NeRF-style sinusoidal positional encoding (fixed frequencies) + learnable projection
  - 4 prepended conditioning tokens (time, species, type, height)
  - Standard pre-norm transformer blocks with QK-norm
  - U-ViT long skip connections (block i -> block L-1-i)
  - Output head applied only to point tokens

References:
  - U-ViT: All are Worth Words (Bao et al., 2023)
  - NeRF: Neural Radiance Fields (Mildenhall et al., 2020)
  - DiT: Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

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
        """Standard sinusoidal embedding for scalar timesteps."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
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


class SinusoidalPointEncoding(nn.Module):
    """
    NeRF-style sinusoidal encoding for 3D point coordinates,
    followed by a learnable MLP projection.

    Frequencies: 2^0, 2^1, ..., 2^(L-1) per axis.
    For L=10: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    Output: raw_coords (3) + sin/cos features (L*2*3) -> MLP -> model_dim
    """

    def __init__(self, model_dim: int, num_freq_bands: int = 10):
        super().__init__()
        self.num_freq_bands = num_freq_bands

        # Fixed NeRF-style frequencies: 2^0, 2^1, ..., 2^(L-1)
        freqs = 2.0 ** torch.arange(num_freq_bands).float()
        self.register_buffer("freqs", freqs)

        # Raw features: 3 (raw coords) + L * 2 * 3 (sin/cos per axis)
        raw_dim = 3 + num_freq_bands * 2 * 3

        # Learnable projection
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 3) - 3D point coordinates in ~[-1, 1]
        Returns:
            (B, N, model_dim)
        """
        # coords: (B, N, 3), freqs: (L,)
        # Expand: (B, N, 3, 1) * (1, 1, 1, L) -> (B, N, 3, L)
        scaled = coords.unsqueeze(-1) * self.freqs[None, None, None, :]

        # sin/cos: each (B, N, 3, L) -> reshape to (B, N, 3*L)
        sin_features = torch.sin(scaled).reshape(*coords.shape[:2], -1)
        cos_features = torch.cos(scaled).reshape(*coords.shape[:2], -1)

        # Concatenate: raw coords + sin + cos
        features = torch.cat([coords, sin_features, cos_features], dim=-1)

        return self.proj(features)


class TransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block with QK-norm for stability.
    No conditioning modulation - purely vanilla.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)

        # QK-norm layers
        self.q_norm = nn.LayerNorm(hidden_size // num_heads, eps=1e-6)
        self.k_norm = nn.LayerNorm(hidden_size // num_heads, eps=1e-6)

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Attention projections
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)

        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, N, D = x.shape

        # --- Attention with QK-norm ---
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, N, num_heads, head_dim)

        # Apply QK-norm per head
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose for attention: (B, num_heads, N, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout.p if self.training else 0.0
        )

        # Reshape back: (B, N, D)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        attn_out = self.out_proj(attn_out)

        x = x + attn_out

        # --- MLP ---
        x = x + self.mlp(self.norm2(x))

        return x


class FlowMatchingTransformer(nn.Module):
    """
    Plain Transformer for Flow Matching with token-prepend conditioning
    and U-ViT skip connections.

    Input: Noisy Points (x_t) + Time (t) + Conditions (Species, Type, Height)
    Output: Velocity field (v)
    """

    NUM_COND_TOKENS = 4  # time, species, type, height

    def __init__(
        self,
        model_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        num_species: int = 10,
        num_types: int = 3,
        dropout: float = 0.1,
        num_freq_bands: int = 12,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers

        # 1. Point Embedding (NeRF sinusoidal + learned projection)
        self.point_embed = SinusoidalPointEncoding(model_dim, num_freq_bands)

        # 2. Conditioning Embedders
        self.t_embedder = TimestepEmbedder(model_dim)

        # +1 for null/unconditional token for CFG
        self.species_embedder = nn.Embedding(num_species + 1, model_dim)
        self.type_embedder = nn.Embedding(num_types + 1, model_dim)

        # Height: continuous + null handling
        self.height_mlp = nn.Sequential(
            nn.Linear(1, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )
        self.null_height_embed = nn.Parameter(torch.zeros(1, model_dim))

        # Token type embeddings to distinguish the 4 conditioning tokens
        self.token_type_embeds = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, model_dim))
                for _ in range(self.NUM_COND_TOKENS)
            ]
        )

        # Null token indices
        self.null_species_idx = num_species
        self.null_type_idx = num_types

        # 3. Transformer Blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(model_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        # 4. U-ViT Skip Connection Projections
        # block i connects to block (num_layers - 1 - i) for i < num_layers // 2
        self.num_skips = num_layers // 2
        self.skip_projections = nn.ModuleList(
            [nn.Linear(2 * model_dim, model_dim) for _ in range(self.num_skips)]
        )

        # 5. Output Head
        self.final_norm = nn.LayerNorm(model_dim, eps=1e-6)
        self.head = nn.Linear(model_dim, 3)

        self.initialize_weights()

    def initialize_weights(self):
        # Xavier initialization for all linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-init the output head (stable init for flow matching)
        nn.init.constant_(self.head.weight, 0)
        nn.init.constant_(self.head.bias, 0)

        # Normal init for embeddings
        nn.init.normal_(self.species_embedder.weight, std=0.02)
        nn.init.normal_(self.type_embedder.weight, std=0.02)

    def forward(self, x, t, species_idx, type_idx, height_norm, drop_mask=None):
        """
        Args:
            x: (B, N, 3) - Noisy points
            t: (B,) - Timesteps
            species_idx: (B,) - Species indices
            type_idx: (B,) - Data type indices
            height_norm: (B,) - Normalized height (log scale)
            drop_mask: (B,) - Boolean tensor. True = drop condition (CFG).
        """
        B = x.shape[0]

        # 1. Embed point coordinates
        point_tokens = self.point_embed(x)  # (B, N, D)

        # 2. Build conditioning tokens
        t_emb = self.t_embedder(t * 1000.0)  # (B, D) — scale t∈[0,1] for sinusoidal resolution

        if drop_mask is not None:
            mask = drop_mask.unsqueeze(1).float()

            # Real embeddings
            s_real = self.species_embedder(species_idx)
            type_real = self.type_embedder(type_idx)
            h_real = self.height_mlp(height_norm.unsqueeze(1))

            # Null embeddings
            s_null = self.species_embedder(
                torch.full((B,), self.null_species_idx, device=x.device)
            )
            type_null = self.type_embedder(
                torch.full((B,), self.null_type_idx, device=x.device)
            )
            h_null = self.null_height_embed.expand(B, -1)

            # Interpolate
            s_emb = s_real * (1 - mask) + s_null * mask
            type_emb = type_real * (1 - mask) + type_null * mask
            h_emb = h_real * (1 - mask) + h_null * mask
        else:
            s_emb = self.species_embedder(species_idx)
            type_emb = self.type_embedder(type_idx)
            h_emb = self.height_mlp(height_norm.unsqueeze(1))

        # Add token type embeddings and stack as (B, 4, D)
        cond_tokens = torch.stack(
            [
                t_emb + self.token_type_embeds[0],
                s_emb + self.token_type_embeds[1],
                type_emb + self.token_type_embeds[2],
                h_emb + self.token_type_embeds[3],
            ],
            dim=1,
        )

        # 3. Concatenate: [cond_tokens, point_tokens] -> (B, 4+N, D)
        tokens = torch.cat([cond_tokens, point_tokens], dim=1)

        # 4. Transformer with U-ViT skip connections
        # First half: save activations for skip connections
        skip_cache = []
        for i in range(self.num_skips):
            tokens = self.blocks[i](tokens)
            skip_cache.append(tokens)

        # Second half: apply skip connections
        for i in range(self.num_skips, self.num_layers):
            tokens = self.blocks[i](tokens)
            skip_idx = self.num_layers - 1 - i
            if skip_idx < self.num_skips:
                # Concatenate with skip and project back
                tokens = self.skip_projections[skip_idx](
                    torch.cat([tokens, skip_cache[skip_idx]], dim=-1)
                )

        # 5. Output: only point tokens (discard first 4 conditioning tokens)
        point_out = tokens[:, self.NUM_COND_TOKENS :]
        point_out = self.final_norm(point_out)

        return self.head(point_out)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing FlowMatchingTransformer...")

    model = FlowMatchingTransformer(
        model_dim=512, num_layers=8, num_heads=8, num_species=10, num_types=3
    )

    print(f"Model parameters: {model.count_parameters():,}")

    B, N = 4, 1000
    x_0 = torch.randn(B, N, 3)
    t_0 = torch.rand(B)
    s_0 = torch.randint(0, 10, (B,))
    dt_0 = torch.randint(0, 3, (B,))
    h_0 = torch.randn(B)

    # Test standard forward
    velocity = model(x_0, t_0, s_0, dt_0, h_0)
    assert velocity.shape == x_0.shape, f"Expected {x_0.shape}, got {velocity.shape}"

    # Test with masking
    mask = torch.tensor([True, False, True, False])
    velocity_masked = model(x_0, t_0, s_0, dt_0, h_0, drop_mask=mask)
    assert velocity_masked.shape == x_0.shape

    print(f"Output shape: {velocity.shape}")
    print("\nDone.")
