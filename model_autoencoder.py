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
            dropout=0.0,
            max_output_points=2048
    ):
        """
        PerceiverIO-based autoencoder for variable-size point clouds.

        Args:
            input_dim: Dimension of input points (3 for XYZ)
            latent_dim: Dimension of latent space
            num_latents: Number of latent vectors (fixed size)
            num_encoder_layers: Number of cross-attention encoder layers
            num_processor_layers: Number of latent self-attention layers
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_output_points: Maximum number of points decoder can generate
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.max_output_points = max_output_points

        # Learnable latent array
        self.latent_array = nn.Parameter(torch.randn(num_latents, latent_dim))

        # Input projection
        self.input_proj = nn.Linear(input_dim, latent_dim)

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

        # Count prediction head
        self.count_head = nn.Sequential(
            nn.Linear(num_latents * latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, 1),
            nn.ReLU()  # Ensure non-negative count prediction
        )

        # Decoder is exclusively the AutoregressiveDecoder
        self.decoder = AutoregressiveDecoder(
            latent_dim=latent_dim,
            max_points=max_output_points,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout=dropout
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

    def decode(self, latent, num_points=None, target_points=None):
        """
        Decode fixed-size latent to point cloud using the autoregressive decoder.

        Args:
            latent: Latent vectors, shape (num_latents, latent_dim)
            num_points: Number of points to generate
            target_points: Target points for teacher forcing (training only), shape (M, 3)

        Returns:
            output: Generated/reconstructed points
        """
        return self.decoder(latent, num_points=num_points, target_points=target_points)

    def forward(self, x, target_points=None):
        """
        Forward pass through autoencoder. Predicts point count and reconstructs.

        Args:
            x: Input points, shape (N, 3)
            target_points: Target points for teacher forcing (training only).
                           The length of this tensor is used as the target for the count head.

        Returns:
            reconstructed_points: The generated point cloud.
            predicted_count: The predicted number of points (scalar tensor).
            latent: Encoded latent representation.
        """
        # Encode
        latent = self.encode(x)

        # Process
        latent = self.process(latent)

        # Predict count from the latent space
        latent_flat = latent.flatten()
        predicted_count = self.count_head(latent_flat).squeeze(-1)

        # Decode
        if self.training and target_points is not None:
            # Training: Use teacher forcing for the decoder with the provided target points
            reconstructed_points = self.decode(latent, target_points=target_points)
        else:
            # Inference: Use the predicted count to generate points
            num_to_generate = torch.clamp(predicted_count.round(), 2, self.max_output_points).int().item()
            reconstructed_points = self.decode(latent, num_points=num_to_generate)

        return reconstructed_points, predicted_count, latent


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding that can handle any sequence length.
    Similar to the original Transformer paper.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        """
        Args:
            seq_len: Length of sequence

        Returns:
            Positional encodings, shape (seq_len, d_model)
        """
        return self.pe[:seq_len]


class AutoregressiveDecoder(nn.Module):
    def __init__(self, latent_dim, max_points=2048, num_layers=6, num_heads=8, dropout=0.0):
        """
        Autoregressive decoder that generates variable number of points.
        Uses sinusoidal positional encoding for flexibility.

        Args:
            latent_dim: Dimension of latent space
            max_points: Maximum number of points that can be generated
            num_layers: Number of transformer decoder layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.max_points = max_points

        # Start token - learnable initial point
        self.start_token = nn.Parameter(torch.zeros(1, 3))

        # Point embedding: embed 3D coordinates to latent_dim
        self.point_embed = nn.Sequential(
            nn.Linear(3, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # Sinusoidal positional encoding (can handle any length up to max_points)
        self.pos_encoding = SinusoidalPositionalEncoding(latent_dim, max_len=max_points)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output head: predict next point coordinates
        self.output_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 3)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.output_head[0].weight)
        nn.init.xavier_uniform_(self.output_head[3].weight)
        nn.init.zeros_(self.output_head[0].bias)
        nn.init.zeros_(self.output_head[3].bias)

    def forward(self, latent, num_points=None, target_points=None):
        """
        Generate points autoregressively with variable output length.

        Args:
            latent: Latent representation, shape (num_latents, latent_dim)
            num_points: Number of points to generate (for inference/eval mode)
            target_points: Ground truth points for teacher forcing (for training mode), shape (M, 3)

        Returns:
            points: Generated points, shape (num_points or M, 3)

        Notes:
            - In training mode with target_points: Uses teacher forcing, ignores num_points
            - In eval mode: Uses autoregressive generation, requires num_points (or infers from target_points)
        """
        latent = latent.unsqueeze(0)  # (1, num_latents, latent_dim)

        if self.training and target_points is not None:
            # Training mode with teacher forcing - num_points is ignored
            if num_points is not None and num_points != len(target_points):
                import warnings
                warnings.warn(
                    f"In training mode with target_points, num_points={num_points} is ignored. "
                    f"Using len(target_points)={len(target_points)} instead."
                )
            return self._forward_teacher_forcing(latent, target_points)

        # Inference/eval mode - determine num_points
        if num_points is None:
            if target_points is not None:
                num_points = len(target_points)
            else:
                num_points = self.max_points

        if num_points > self.max_points:
            raise ValueError(f"num_points ({num_points}) exceeds max_points ({self.max_points})")

        return self._forward_autoregressive(latent, num_points)

    def _forward_teacher_forcing(self, latent, target_points):
        """
        Training with teacher forcing - parallel generation with causal masking.

        Args:
            latent: (1, num_latents, latent_dim)
            target_points: (M, 3) ground truth points

        Returns:
            predicted_points: (M, 3)
        """
        num_target_points = target_points.size(0)

        # Prepend start token to target points
        start = self.start_token.expand(1, -1)  # (1, 3)
        points_with_start = torch.cat([start, target_points[:-1]], dim=0)  # (M, 3)

        # Embed points
        point_embeds = self.point_embed(points_with_start)  # (M, latent_dim)

        # Add positional encodings (sinusoidal - works for any length)
        pos_enc = self.pos_encoding(num_target_points).to(point_embeds.device)  # (M, latent_dim)
        point_embeds = point_embeds + pos_enc
        point_embeds = point_embeds.unsqueeze(0)  # (1, M, latent_dim)

        # Create causal mask (each position can only attend to previous positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            num_target_points,
            device=latent.device
        )  # (M, M)

        # Apply transformer decoder with causal masking
        output = self.transformer(
            point_embeds,
            latent,
            tgt_mask=causal_mask
        )  # (1, M, latent_dim)

        # Predict coordinates
        predicted_points = self.output_head(output.squeeze(0))  # (M, 3)

        return predicted_points

    def _forward_autoregressive(self, latent, num_points):
        """
        Inference with autoregressive generation - variable length output.

        Args:
            latent: (1, num_latents, latent_dim)
            num_points: Number of points to generate

        Returns:
            points: (num_points, 3)
        """
        device = latent.device

        # Start sequence with start token
        sequence = [self.start_token]  # List of (1, 3) tensors

        # Generate num_points autoregressively
        for i in range(num_points):
            # Concatenate all points in sequence so far (including start token)
            points_so_far = torch.cat(sequence, dim=0)  # (len(sequence), 3)

            # Embed points
            point_embeds = self.point_embed(points_so_far)  # (len(sequence), latent_dim)

            # Add positional encoding (for current sequence length)
            seq_len = len(sequence)
            pos_enc = self.pos_encoding(seq_len).to(device)  # (seq_len, latent_dim)
            point_embeds = point_embeds + pos_enc
            point_embeds = point_embeds.unsqueeze(0)  # (1, seq_len, latent_dim)

            # Decode (no mask needed since we're only looking at past)
            output = self.transformer(point_embeds, latent)  # (1, seq_len, latent_dim)

            # Predict next point (take last position)
            next_point = self.output_head(output[:, -1:, :])  # (1, 1, 3)
            next_point = next_point.squeeze(0)  # (1, 3)

            # Add to sequence
            sequence.append(next_point)

        # Return all generated points (excluding start token at index 0)
        generated_points = torch.cat(sequence[1:], dim=0)  # (num_points, 3)

        return generated_points


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