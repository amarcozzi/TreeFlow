# model_autoregressive.py
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """ Standard sinusoidal positional encoding """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 25000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PerceiverIOEncoder(nn.Module):
    """ The Encoder part of the original model, isolated for modularity. """

    def __init__(self, input_dim=3, latent_dim=512, num_latents=256, num_encoder_layers=6,
                 num_processor_layers=8, num_heads=8, dropout=0.1):
        super().__init__()
        self.latent_array = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.input_proj = nn.Linear(input_dim, latent_dim)

        self.encoder = nn.ModuleList([
            CrossAttentionBlock(latent_dim, num_heads, dropout) for _ in range(num_encoder_layers)
        ])
        self.processor = nn.ModuleList([
            SelfAttentionBlock(latent_dim, num_heads, dropout) for _ in range(num_processor_layers)
        ])

    def forward(self, x):
        # x is a single point cloud (N, 3)
        # Project input to latent dimension
        x_proj = self.input_proj(x)  # (N, latent_dim)

        # Initialize latent array for the batch
        latent = self.latent_array.unsqueeze(0)  # (1, num_latents, latent_dim)

        # Cross-attend from latents (query) to input (key/value)
        for layer in self.encoder:
            latent = layer(latent, x_proj.unsqueeze(0))

        # Self-attend on latents
        for layer in self.processor:
            latent = layer(latent)

        return latent.squeeze(0)  # (num_latents, latent_dim)


class AutoregressiveTransformerDecoder(nn.Module):
    def __init__(self, latent_dim=512, output_dim=3, num_heads=8, num_decoder_layers=6, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim

        # A learnable token to start the generation sequence
        self.start_token = nn.Parameter(torch.randn(1, 1, latent_dim))

        self.pos_encoder = PositionalEncoding(latent_dim, dropout)
        self.input_proj = nn.Linear(output_dim, latent_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, nhead=num_heads, dim_feedforward=latent_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Heads to predict the next point coordinates and the stop token
        self.coord_head = nn.Linear(latent_dim, output_dim)
        self.stop_head = nn.Linear(latent_dim, 1)  # Logit for stopping

    def generate_square_subsequent_mask(self, sz: int, device: torch.device):
        """Generates a square mask for the sequence. True values are positions that are NOT allowed to attend."""
        return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)

    def forward(self, memory, tgt_points):
        """
        Forward pass for training using Teacher Forcing.

        Args:
            memory (torch.Tensor): The latent memory from the encoder. Shape: (num_latents, latent_dim).
            tgt_points (torch.Tensor): The ground truth target points. Shape: (N, 3).
        """
        # Add a batch dimension
        memory = memory.unsqueeze(0)  # (1, num_latents, latent_dim)
        tgt_points = tgt_points.unsqueeze(0)  # (1, N, 3)

        # Prepare target sequence for decoder: prepend start token
        # This creates the input sequence for the decoder
        tgt_proj = self.input_proj(tgt_points)  # (1, N, latent_dim)
        decoder_input = torch.cat([self.start_token, tgt_proj[:, :-1, :]], dim=1)
        decoder_input = self.pos_encoder(decoder_input.transpose(0, 1)).transpose(0, 1)  # Add PE

        # Create causal mask to prevent decoder from seeing future points
        tgt_mask = self.generate_square_subsequent_mask(decoder_input.size(1), memory.device)

        # Get decoder output
        output = self.transformer_decoder(decoder_input, memory, tgt_mask=tgt_mask)

        # Predict coordinates and stop logits
        pred_coords = self.coord_head(output)  # (1, N, 3)
        stop_logits = self.stop_head(output).squeeze(-1)  # (1, N)

        return pred_coords.squeeze(0), stop_logits.squeeze(0)

    @torch.no_grad()
    def generate(self, memory, max_len=2500, stop_threshold=0.5):
        """
        Autoregressive generation for inference.

        Args:
            memory (torch.Tensor): The latent memory from the encoder. Shape: (num_latents, latent_dim).
            max_len (int): Maximum number of points to generate.
            stop_threshold (float): Sigmoid probability threshold to stop generation.
        """
        memory = memory.unsqueeze(0)  # (1, num_latents, latent_dim)
        generated_seq = self.start_token

        generated_points = []

        for i in range(max_len):
            pos_encoded_seq = self.pos_encoder(generated_seq.transpose(0, 1)).transpose(0, 1)
            output = self.transformer_decoder(pos_encoded_seq, memory)  # No mask needed for generation

            # Get the prediction from the last token in the sequence
            last_output = output[:, -1, :]

            # Predict next coordinate and stop token
            next_coord = self.coord_head(last_output)  # (1, 3)
            stop_logit = self.stop_head(last_output)  # (1, 1)

            generated_points.append(next_coord)

            # Check stop condition
            if torch.sigmoid(stop_logit).item() > stop_threshold:
                break

            # Prepare next input
            next_input_proj = self.input_proj(next_coord.unsqueeze(1))
            generated_seq = torch.cat([generated_seq, next_input_proj], dim=1)

        return torch.cat(generated_points, dim=0)


class GenerativePerceiver(nn.Module):
    """ The final model combining the encoder and the new autoregressive decoder. """

    def __init__(self, input_dim=3, latent_dim=512, num_latents=256,
                 num_encoder_layers=6, num_processor_layers=8, num_decoder_layers=6,
                 num_heads=8, dropout=0.1):
        super().__init__()
        self.encoder = PerceiverIOEncoder(
            input_dim, latent_dim, num_latents, num_encoder_layers,
            num_processor_layers, num_heads, dropout
        )
        self.decoder = AutoregressiveTransformerDecoder(
            latent_dim, input_dim, num_heads, num_decoder_layers, dropout
        )

    def forward(self, points):
        """ For training with teacher forcing """
        # Sort points for a consistent sequence order during training
        sorted_points = points[torch.argsort(points[:, 0])]

        memory = self.encoder(sorted_points)
        pred_coords, stop_logits = self.decoder(memory, sorted_points)
        return pred_coords, stop_logits

    @torch.no_grad()
    def generate(self, points, max_len=2500, stop_threshold=0.5):
        """ For inference/generation """
        # We still need to encode an input point cloud to get a latent memory
        sorted_points = points[torch.argsort(points[:, 0])]
        memory = self.encoder(sorted_points)
        generated_points = self.decoder.generate(memory, max_len, stop_threshold)
        return generated_points


# --- Attention Blocks (copied from your original code for completeness) ---

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * 4, dim), nn.Dropout(dropout))

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
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * 4, dim), nn.Dropout(dropout))

    def forward(self, query, key_value):
        attn_out, _ = self.attention(query, key_value, key_value)
        query = self.norm1(query + attn_out)
        mlp_out = self.mlp(query)
        query = self.norm2(query + mlp_out)
        return query