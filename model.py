"""
treeflow/model.py
"""

import torch
import torch.nn as nn
import math

def modulate(x, shift, scale):
    """
    FiLM / AdaLN modulation function.
    x: (B, N, D)
    shift, scale: (B, D)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
        """Standard sinusoidal embedding."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class PositionalEncoding3D(nn.Module):
    """3D positional encoding for point clouds (fixed sinusoidal or learnable MLP)."""

    def __init__(self, dim: int, max_freq: float = 10000.0, learnable: bool = False):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        self.learnable = learnable

        if learnable:
            # Learnable MLP-based positional encoding
            self.pos_mlp = nn.Sequential(
                nn.Linear(3, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            )
        else:
            # Fixed sinusoidal positional encoding
            dim_per_axis = dim // 3
            freq_bands = dim_per_axis // 2

            if freq_bands == 0:
                raise ValueError(f"Dimension {dim} is too small for 3D encoding (need at least 6)")

            inv_freq = 1.0 / (max_freq ** (torch.arange(0, freq_bands).float() / max(freq_bands - 1, 1)))
            self.register_buffer('inv_freq', inv_freq)
            self.freq_bands = freq_bands
            self.dim_per_axis = dim_per_axis

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if self.learnable:
            # Use learnable MLP
            return self.pos_mlp(coords)  # (B, N, dim)
        else:
            # Use fixed sinusoidal encoding
            B, N, _ = coords.shape

            x = coords[:, :, 0:1]
            y = coords[:, :, 1:2]
            z = coords[:, :, 2:3]

            x_freqs = x * self.inv_freq[None, None, :]
            y_freqs = y * self.inv_freq[None, None, :]
            z_freqs = z * self.inv_freq[None, None, :]

            x_enc = torch.cat([torch.sin(x_freqs), torch.cos(x_freqs)], dim=-1)
            y_enc = torch.cat([torch.sin(y_freqs), torch.cos(y_freqs)], dim=-1)
            z_enc = torch.cat([torch.sin(z_freqs), torch.cos(z_freqs)], dim=-1)

            encoding = torch.cat([x_enc, y_enc, z_enc], dim=-1)

            current_dim = encoding.shape[-1]
            if current_dim < self.dim:
                padding = torch.zeros(B, N, self.dim - current_dim, device=coords.device)
                encoding = torch.cat([encoding, padding], dim=-1)
            elif current_dim > self.dim:
                encoding = encoding[:, :, :self.dim]

            return encoding


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block with Adaptive Layer Norm (AdaLN).
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        # 1. Norms (elementwise_affine=False because AdaLN handles the affine part)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # 2. Attention
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)

        # 3. MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )

        # 4. AdaLN Modulation Regressor
        # Predicts 6 vectors: shift/scale/gate for Attn, shift/scale/gate for MLP
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # Predict modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # --- Block Part 1: Attention ---
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # --- Block Part 2: MLP ---
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class ConditionalFlowMatching(nn.Module):
    """
    Conditional DiT for Flow Matching.
    Input: Noisy Points (x_t) + Time (t) + Conditions (Species, Type, Height)
    Output: Velocity field (v)
    """
    def __init__(
        self,
        model_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        num_species: int = 10,
        num_types: int = 3,
        dropout: float = 0.1,
        max_freq: float = 10000.0,
        learnable_pos_encoding: bool = False
    ):
        super().__init__()
        self.model_dim = model_dim

        # 1. Spatial Embeddings
        self.pos_encoding = PositionalEncoding3D(model_dim, max_freq, learnable=learnable_pos_encoding)

        # 2. Conditioning Embedders
        self.t_embedder = TimestepEmbedder(model_dim)

        # +1 for the "Null/Unconditional" token for CFG
        self.species_embedder = nn.Embedding(num_species + 1, model_dim)
        self.type_embedder = nn.Embedding(num_types + 1, model_dim)

        # Height: Continuous + Null handling
        self.height_mlp = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim)
        )
        self.null_height_embed = nn.Parameter(torch.randn(1, model_dim))

        # 3. Transformer Backbone
        self.blocks = nn.ModuleList([
            DiTBlock(model_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])

        # 4. Final Output Head
        self.final_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, 2 * model_dim, bias=True)
        )
        self.head = nn.Linear(model_dim, 3)

        self.initialize_weights()

        # Store the indices for null tokens
        self.null_species_idx = num_species
        self.null_type_idx = num_types

    def initialize_weights(self):
        # Xavier initialization for standard layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-Init for AdaLN modulators
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-Init for final block
        nn.init.constant_(self.adaLN_final[-1].weight, 0)
        nn.init.constant_(self.adaLN_final[-1].bias, 0)

    def forward(self, x, t, species_idx, type_idx, height_norm, drop_mask=None):
        """
        Args:
            x: (B, N, 3) - Points
            t: (B,) - Timesteps
            species_idx: (B,) - Species Integers
            type_idx: (B,) - Type Integers
            height_norm: (B,) - Normalized Height (Log scale)
            drop_mask: (B,) - Boolean tensor. True = Drop condition (CFG).
        """
        # 1. Embed Spatial Inputs
        x = self.pos_encoding(x)

        # 2. Embed Time
        t_emb = self.t_embedder(t)

        # 3. Handle CFG / Dropping
        if drop_mask is not None:
            mask = drop_mask.unsqueeze(1).float()

            # Embed real
            s_real = self.species_embedder(species_idx)
            type_real = self.type_embedder(type_idx)

            # Embed null
            B = x.shape[0]
            s_null = self.species_embedder(torch.full((B,), self.null_species_idx, device=x.device))
            type_null = self.type_embedder(torch.full((B,), self.null_type_idx, device=x.device))

            # Interpolate
            s_emb = s_real * (1 - mask) + s_null * mask
            type_emb = type_real * (1 - mask) + type_null * mask

            # Height
            h_real = self.height_mlp(height_norm.unsqueeze(1))
            h_null = self.null_height_embed.expand(B, -1)
            h_emb = h_real * (1 - mask) + h_null * mask

        else:
            s_emb = self.species_embedder(species_idx)
            type_emb = self.type_embedder(type_idx)
            h_emb = self.height_mlp(height_norm.unsqueeze(1))

        # 4. Sum Conditioning Context
        cond = t_emb + s_emb + type_emb + h_emb

        # 5. Transformer
        for block in self.blocks:
            x = block(x, cond)

        # 6. Output
        shift, scale = self.adaLN_final(cond).chunk(2, dim=1)
        x = modulate(self.final_norm(x), shift, scale)

        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing ConditionalFlowMatching (DiT)...")

    model = ConditionalFlowMatching(
        model_dim=256,
        num_layers=8,
        num_heads=8,
        num_species=10,
        num_types=3
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
    assert velocity.shape == x_0.shape

    # Test with masking
    mask = torch.tensor([True, False, True, False])
    velocity_masked = model(x_0, t_0, s_0, dt_0, h_0, drop_mask=mask)
    assert velocity_masked.shape == x_0.shape

    print("\n✓ Model works correctly!")