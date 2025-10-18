"""
treeflow/model.py

Transformer for Flow Matching on 3D Tree Point Clouds
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Literal


class SinusoidalTimeMLP(nn.Module):
    """Sinusoidal time embedding with MLP projection."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        if dim < 2:
            raise ValueError(f"Dimension must be at least 2, got {dim}")
        self.half_dim = dim // 2

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.dim() == 0:
            time = time.unsqueeze(0)

        device = time.device
        freqs = math.log(10000) / max(self.half_dim - 1, 1)
        freqs = torch.exp(torch.arange(self.half_dim, device=device) * -freqs)

        embeddings = time[:, None] * freqs[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)

        current_dim = embeddings.shape[-1]
        if current_dim < self.dim:
            padding = torch.zeros(embeddings.shape[0], self.dim - current_dim, device=device)
            embeddings = torch.cat([embeddings, padding], dim=-1)
        elif current_dim > self.dim:
            embeddings = embeddings[:, :self.dim]

        return self.mlp(embeddings)


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


class TransformerVelocityField(nn.Module):
    """
    Unconditional Transformer-based velocity field model for flow matching.

    Uses self-attention with simple time conditioning via addition + learned projection.
    Suitable for unconditional generation like TreeFlow.
    """

    def __init__(
            self,
            model_dim: int = 256,
            num_heads: int = 8,
            num_layers: int = 8,
            dim_feedforward: Optional[int] = None,
            dropout: float = 0.1,
            max_freq: float = 10000.0,
            learnable_pos_encoding: bool = False
    ):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        if dim_feedforward is None:
            dim_feedforward = model_dim * 4

        # Positional encoding for 3D coordinates
        self.pos_encoding = PositionalEncoding3D(model_dim, max_freq, learnable=learnable_pos_encoding)

        # Time embedding
        self.time_mlp = SinusoidalTimeMLP(dim=model_dim)

        # Learned projection after adding time to spatial features
        self.fusion_proj = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU()
        )

        # Transformer encoder (self-attention only)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 3)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict velocity field.

        Args:
            x_t: Noisy points, shape (B, N, 3)
            t: Time steps, shape (B,) or scalar

        Returns:
            Predicted velocity, shape (B, N, 3)
        """
        B, N, _ = x_t.shape

        # Encode positions
        pos_features = self.pos_encoding(x_t)  # (B, N, dim)

        # Get time embedding
        time_emb = self.time_mlp(t)  # (B, dim)

        # Broadcast time embedding across all points and concatenate with spatial features
        time_emb_expanded = time_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, dim)
        features = torch.cat([pos_features, time_emb_expanded], dim=-1)  # (B, N, 2*dim)

        # Project back to model_dim
        features = self.fusion_proj(features)  # (B, N, dim)

        # Apply transformer (self-attention)
        features = self.transformer(features)  # (B, N, dim)

        # Predict velocity
        velocity = self.output_proj(features)  # (B, N, 3)

        return velocity

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing TransformerVelocityField...")

    model = TransformerVelocityField(
        model_dim=64,
        num_heads=8,
        num_layers=2,
        dropout=0.1
    )

    print(f"Model parameters: {model.count_parameters():,}")

    # Test with (B, N, 3) format
    batch_size = 4
    num_points = 1000

    x_t = torch.randn(batch_size, num_points, 3)
    t = torch.rand(batch_size)

    velocity = model(x_t, t)
    print(f"\nInput shape:  {x_t.shape}")
    print(f"Output shape: {velocity.shape}")
    assert velocity.shape == x_t.shape

    print("\n✓ Model works correctly!")