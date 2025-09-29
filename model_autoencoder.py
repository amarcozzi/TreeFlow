# model_autoencoder.py
import torch
import torch.nn as nn
import math


class PerceiverIOAutoencoder(nn.Module):
    def __init__(
            self,
            input_dim=3,
            latent_dim=512,
            num_latents=256,
            num_encoder_layers=6,
            num_processor_layers=8,
            num_decoder_layers=6,
            num_heads=8,
            dropout=0.0
    ):
        """
        PerceiverIO-based autoencoder for variable-size point clouds.

        Args:
            input_dim: Dimension of input points (3 for XYZ)
            latent_dim: Dimension of latent space
            num_latents: Number of latent vectors (fixed size)
            num_encoder_layers: Number of cross-attention encoder layers
            num_processor_layers: Number of latent self-attention layers
            num_decoder_layers: Number of cross-attention decoder layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents

        # Learnable latent array
        self.latent_array = nn.Parameter(torch.randn(num_latents, latent_dim))

        # Input projection
        self.input_proj = nn.Linear(input_dim, latent_dim)

        # Positional encoding for queries
        self.query_proj = nn.Linear(input_dim, latent_dim)

        # Encoder: cross-attention from input to latent
        self.encoder = nn.ModuleList([
            CrossAttentionBlock(latent_dim, num_heads, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Processor: self-attention on latent
        self.processor = nn.ModuleList([
            SelfAttentionBlock(latent_dim, num_heads, dropout)
            for _ in range(num_processor_layers)
        ])

        # Decoder: cross-attention from latent to output
        self.decoder = nn.ModuleList([
            CrossAttentionBlock(latent_dim, num_heads, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, input_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, x):
        """
        Encode variable-size point cloud to fixed-size latent.

        Args:
            x: Input points, shape (N, 3)

        Returns:
            latent: Encoded latent, shape (num_latents, latent_dim)
        """
        # Project input
        x_proj = self.input_proj(x)  # (N, latent_dim)

        # Initialize latent
        batch_size = 1  # Single point cloud
        latent = self.latent_array.unsqueeze(0)  # (1, num_latents, latent_dim)

        # Encoder: cross-attend from latent (query) to input (key/value)
        for encoder_layer in self.encoder:
            latent = encoder_layer(latent, x_proj.unsqueeze(0))

        return latent.squeeze(0)  # (num_latents, latent_dim)

    def process(self, latent):
        """
        Process latent with self-attention.

        Args:
            latent: Latent vectors, shape (num_latents, latent_dim)

        Returns:
            latent: Processed latent, shape (num_latents, latent_dim)
        """
        latent = latent.unsqueeze(0)  # (1, num_latents, latent_dim)

        for processor_layer in self.processor:
            latent = processor_layer(latent)

        return latent.squeeze(0)  # (num_latents, latent_dim)

    def decode(self, latent, query_positions):
        """
        Decode fixed-size latent to variable-size point cloud.

        Args:
            latent: Latent vectors, shape (num_latents, latent_dim)
            query_positions: Query positions for output, shape (M, 3)

        Returns:
            output: Reconstructed points, shape (M, 3)
        """
        # Create queries from positions
        queries = self.query_proj(query_positions)  # (M, latent_dim)

        latent = latent.unsqueeze(0)  # (1, num_latents, latent_dim)
        queries = queries.unsqueeze(0)  # (1, M, latent_dim)

        # Decoder: cross-attend from queries to latent
        for decoder_layer in self.decoder:
            queries = decoder_layer(queries, latent)

        # Project to output space
        output = self.output_proj(queries.squeeze(0))  # (M, 3)

        return output

    def forward(self, x):
        """
        Forward pass through autoencoder.

        Args:
            x: Input points, shape (N, 3)

        Returns:
            output: Reconstructed points, shape (N, 3)
            latent: Encoded latent representation
        """
        # Encode
        latent = self.encode(x)

        # Process
        latent = self.process(latent)

        # Decode using input positions as queries
        output = self.decode(latent, x)

        return output, latent


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # MLP with residual
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, key_value):
        """
        Cross-attention from query to key_value.

        Args:
            query: Query tensor, shape (B, N_q, dim)
            key_value: Key and value tensor, shape (B, N_kv, dim)
        """
        # Cross-attention with residual
        attn_out, _ = self.attention(query, key_value, key_value)
        query = self.norm1(query + attn_out)

        # MLP with residual
        mlp_out = self.mlp(query)
        query = self.norm2(query + mlp_out)

        return query