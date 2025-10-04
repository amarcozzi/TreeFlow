# model_autoencoder.py
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, dropout=0.1, max_len=25000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PerceiverIOEncoder(nn.Module):
    """Encoder to compress point cloud to fixed latent representation."""

    def __init__(self, input_dim=3, latent_dim=512, num_latents=256,
                 num_encoder_layers=6, num_processor_layers=8, num_heads=8, dropout=0.1):
        super().__init__()

        self.latent_array = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.input_proj = nn.Linear(input_dim, latent_dim)

        self.encoder = nn.ModuleList([
            CrossAttentionBlock(latent_dim, num_heads, dropout)
            for _ in range(num_encoder_layers)
        ])

        self.processor = nn.ModuleList([
            SelfAttentionBlock(latent_dim, num_heads, dropout)
            for _ in range(num_processor_layers)
        ])

    def forward(self, x):
        """
        Args:
            x: (N, 3) point cloud

        Returns:
            latent: (num_latents, latent_dim)
        """
        x_proj = self.input_proj(x)  # (N, latent_dim)
        latent = self.latent_array.unsqueeze(0)  # (1, num_latents, latent_dim)

        # Cross-attention: latents attend to input points
        for layer in self.encoder:
            latent = layer(latent, x_proj.unsqueeze(0))

        # Self-attention on latents
        for layer in self.processor:
            latent = layer(latent)

        return latent.squeeze(0)  # (num_latents, latent_dim)


class AutoregressiveDecoder(nn.Module):
    """
    Autoregressive decoder with:
    - Proper handling of variable-length sequences
    - Dual-head for coordinates and stop prediction
    - Support for teacher forcing and autoregressive generation
    """

    def __init__(self, latent_dim=512, output_dim=3, num_heads=8,
                 num_decoder_layers=6, dropout=0.1, min_points=16):
        super().__init__()

        self.latent_dim = latent_dim
        self.min_points = min_points

        # Learnable start token
        self.start_token = nn.Parameter(torch.randn(1, 1, latent_dim))

        # Positional encoding
        self.pos_encoder = PositionalEncoding(latent_dim, dropout)

        # Project point coordinates to latent dimension
        self.input_proj = nn.Linear(output_dim, latent_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Coordinate prediction head
        self.coord_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, output_dim)
        )

        # Stop prediction head
        self.stop_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights carefully."""
        nn.init.xavier_uniform_(self.coord_head[0].weight)
        nn.init.xavier_uniform_(self.coord_head[3].weight)
        nn.init.zeros_(self.coord_head[0].bias)
        nn.init.zeros_(self.coord_head[3].bias)

        nn.init.xavier_uniform_(self.stop_head[0].weight)
        nn.init.xavier_uniform_(self.stop_head[2].weight)
        nn.init.zeros_(self.stop_head[0].bias)
        # Initialize stop head bias to make early stopping less likely
        nn.init.constant_(self.stop_head[2].bias, -2.0)

    def generate_causal_mask(self, sz, device):
        """Generate causal mask for autoregressive attention."""
        return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)

    def forward(self, memory, tgt_points):
        """
        Training with teacher forcing.

        Args:
            memory: (num_latents, latent_dim) - encoded latent from input
            tgt_points: (N, 3) - target points (already sorted)

        Returns:
            pred_coords: (N, 3) - predicted coordinates
            stop_logits: (N,) - stop prediction logits
        """
        memory = memory.unsqueeze(0)  # (1, num_latents, latent_dim)
        tgt_points = tgt_points.unsqueeze(0)  # (1, N, 3)

        # Prepare decoder input: [start_token, tgt[0], tgt[1], ..., tgt[N-2]]
        tgt_proj = self.input_proj(tgt_points)  # (1, N, latent_dim)
        decoder_input = torch.cat([self.start_token, tgt_proj[:, :-1, :]], dim=1)

        # Add positional encoding
        decoder_input = self.pos_encoder(decoder_input.transpose(0, 1)).transpose(0, 1)

        # Create causal mask
        tgt_mask = self.generate_causal_mask(decoder_input.size(1), memory.device)

        # Decode
        output = self.transformer_decoder(decoder_input, memory, tgt_mask=tgt_mask)

        # Predict coordinates and stop
        pred_coords = self.coord_head(output)  # (1, N, 3)
        stop_logits = self.stop_head(output).squeeze(-1)  # (1, N)

        return pred_coords.squeeze(0), stop_logits.squeeze(0)

    @torch.no_grad()
    def generate(self, memory, max_len=2500, stop_threshold=0.5):
        """
        Autoregressive generation for inference.

        Args:
            memory: (num_latents, latent_dim)
            max_len: Maximum points to generate
            stop_threshold: Sigmoid probability threshold to stop

        Returns:
            generated_points: (M, 3) where M is learned
        """
        memory = memory.unsqueeze(0)  # (1, num_latents, latent_dim)

        # Start with start token
        generated_seq = self.start_token
        generated_points = []

        for i in range(max_len):
            # Add positional encoding
            pos_encoded = self.pos_encoder(generated_seq.transpose(0, 1)).transpose(0, 1)

            # Decode (no causal mask needed - we only have past)
            output = self.transformer_decoder(pos_encoded, memory)

            # Get last position prediction
            last_output = output[:, -1:, :]

            # Predict coordinate and stop
            next_coord = self.coord_head(last_output)  # (1, 1, 3)
            stop_logit = self.stop_head(last_output)  # (1, 1, 1)

            generated_points.append(next_coord.squeeze(0))

            # Check stop condition (after minimum points)
            if i >= self.min_points:
                stop_prob = torch.sigmoid(stop_logit).item()
                if stop_prob > stop_threshold:
                    break

            # Prepare next input
            next_proj = self.input_proj(next_coord)
            generated_seq = torch.cat([generated_seq, next_proj], dim=1)

        if len(generated_points) == 0:
            return torch.empty(0, 3, device=memory.device)

        return torch.cat(generated_points, dim=0)


class GenerativePerceiver(nn.Module):
    """
    Complete model: Encoder + Autoregressive Decoder.
    Assumes input is already Z-sorted by the dataset.
    """

    def __init__(self, input_dim=3, latent_dim=512, num_latents=256,
                 num_encoder_layers=6, num_processor_layers=8, num_decoder_layers=6,
                 num_heads=8, dropout=0.1, min_points=16):
        super().__init__()

        self.encoder = PerceiverIOEncoder(
            input_dim, latent_dim, num_latents,
            num_encoder_layers, num_processor_layers,
            num_heads, dropout
        )

        self.decoder = AutoregressiveDecoder(
            latent_dim, input_dim, num_heads,
            num_decoder_layers, dropout, min_points
        )

    def forward(self, points):
        """
        Training forward pass.

        Args:
            points: (N, 3) point cloud (already Z-sorted by dataset)

        Returns:
            pred_coords: (N, 3) predicted coordinates
            stop_logits: (N,) stop prediction logits
        """
        # Encode (points already sorted and augmented by dataset)
        memory = self.encoder(points)

        # Decode with teacher forcing
        pred_coords, stop_logits = self.decoder(memory, points)

        return pred_coords, stop_logits

    @torch.no_grad()
    def generate(self, points, max_len=2500, stop_threshold=0.5):
        """
        Generation for inference.

        Args:
            points: (N, 3) input point cloud (for encoding only, already Z-sorted)
            max_len: Maximum points to generate
            stop_threshold: Stop probability threshold

        Returns:
            generated_points: (M, 3) where M is learned
        """
        memory = self.encoder(points)
        generated = self.decoder.generate(memory, max_len, stop_threshold)
        return generated


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
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
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
        attn_out, _ = self.attention(query, key_value, key_value)
        query = self.norm1(query + attn_out)
        mlp_out = self.mlp(query)
        query = self.norm2(query + mlp_out)
        return query