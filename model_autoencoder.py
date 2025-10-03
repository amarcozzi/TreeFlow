# model_autoencoder.py
import torch
import torch.nn as nn


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
            num_slots=4196
    ):
        """
        PerceiverIO-based autoencoder with parallel slots decoder.
        Uses PyTorch's built-in transformer layers throughout.

        Args:
            input_dim: Dimension of input points (3 for XYZ)
            latent_dim: Dimension of latent space
            num_latents: Number of latent vectors (fixed size)
            num_encoder_layers: Number of cross-attention encoder layers
            num_processor_layers: Number of latent self-attention layers
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            num_slots: Number of parallel query slots for generation
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.num_slots = num_slots

        # Learnable latent array
        self.latent_array = nn.Parameter(torch.randn(num_latents, latent_dim))

        # Input projection
        self.input_proj = nn.Linear(input_dim, latent_dim)

        # Encoder: cross-attention from latent (tgt) to input (memory)
        # Using TransformerDecoder because we need cross-attention
        encoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerDecoder(encoder_layer, num_encoder_layers)

        # Processor: self-attention on latent
        processor_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.processor = nn.TransformerEncoder(processor_layer, num_processor_layers)

        # Decoder: parallel slots
        self.decoder = ParallelSlotsDecoder(
            latent_dim=latent_dim,
            num_slots=num_slots,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            output_dim=input_dim
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Initialize latent array
        nn.init.normal_(self.latent_array, std=0.02)

        # Initialize input projection
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

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
        x_proj = x_proj.unsqueeze(0)  # (1, N, latent_dim)

        # Initialize latent
        latent = self.latent_array.unsqueeze(0)  # (1, num_latents, latent_dim)

        # Encoder: cross-attend from latent (tgt) to input (memory)
        # TransformerDecoder does: tgt attends to memory (cross-attention)
        latent = self.encoder(tgt=latent, memory=x_proj)  # (1, num_latents, latent_dim)

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

        # Self-attention on latents
        latent = self.processor(latent)  # (1, num_latents, latent_dim)

        return latent.squeeze(0)  # (num_latents, latent_dim)

    def decode(self, latent, return_validity=False):
        """
        Decode latent to variable-size point cloud using parallel slots.

        Args:
            latent: Latent vectors, shape (num_latents, latent_dim)
            return_validity: If True, return validity scores for all slots

        Returns:
            If return_validity=False:
                points: Variable-size point cloud, shape (M, 3) where M <= num_slots
            If return_validity=True:
                all_coords: All slot coordinates, shape (num_slots, 3)
                validity: Validity scores for all slots, shape (num_slots,)
        """
        return self.decoder(latent, return_validity=return_validity)

    def forward(self, x, return_validity=False):
        """
        Forward pass through autoencoder.

        Args:
            x: Input points, shape (N, 3)
            return_validity: If True, return all slots and validity (for training)

        Returns:
            If return_validity=False:
                output: Reconstructed points, shape (M, 3)
                latent: Encoded latent representation
            If return_validity=True:
                all_coords: All slot coordinates, shape (num_slots, 3)
                validity: Validity scores, shape (num_slots,)
                latent: Encoded latent representation
        """
        # Encode
        latent = self.encode(x)

        # Process
        latent = self.process(latent)

        # Decode
        if return_validity:
            all_coords, validity = self.decode(latent, return_validity=True)
            return all_coords, validity, latent
        else:
            output = self.decode(latent, return_validity=False)
            return output, latent


class ParallelSlotsDecoder(nn.Module):
    """
    Parallel slots decoder inspired by DETR.
    Uses PyTorch's built-in TransformerDecoder.

    All slots decode in parallel (one forward pass), each predicting:
    1. A 3D coordinate
    2. A validity score (is this slot active?)

    Variable length is achieved by having different numbers of active slots.
    """

    def __init__(self, latent_dim, num_slots=2048, num_layers=6, num_heads=8, dropout=0.0, output_dim=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_slots = num_slots
        self.output_dim = output_dim

        # Learnable query slots (like object queries in DETR)
        self.query_slots = nn.Parameter(torch.randn(num_slots, latent_dim))

        # Transformer decoder: cross-attend from slots (tgt) to latent (memory)
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

        # Output heads
        # Head 1: Predict 3D coordinates for each slot
        self.coord_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, output_dim)
        )

        # Head 2: Predict if slot is valid (active)
        self.validity_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Initialize query slots with small values
        nn.init.normal_(self.query_slots, std=0.02)

        # Initialize output heads
        for module in [self.coord_head, self.validity_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, latent, return_validity=False):
        """
        Decode latent to points using parallel slots.

        Args:
            latent: Latent representation, shape (num_latents, latent_dim)
            return_validity: If True, return all coords and validity scores

        Returns:
            If return_validity=False:
                points: Active points only, shape (M, 3) where M = number of valid slots
            If return_validity=True:
                all_coords: All slot coordinates, shape (num_slots, 3)
                validity_scores: Validity scores, shape (num_slots,)
        """
        # Add batch dimension
        latent = latent.unsqueeze(0)  # (1, num_latents, latent_dim)
        slots = self.query_slots.unsqueeze(0)  # (1, num_slots, latent_dim)

        # Cross-attend from slots (tgt) to latent (memory)
        decoded_slots = self.transformer(tgt=slots, memory=latent)  # (1, num_slots, latent_dim)
        decoded_slots = decoded_slots.squeeze(0)  # (num_slots, latent_dim)

        # Predict coordinates and validity for each slot
        all_coords = self.coord_head(decoded_slots)  # (num_slots, 3)
        validity_logits = self.validity_head(decoded_slots).squeeze(-1)  # (num_slots,)
        validity_scores = torch.sigmoid(validity_logits)  # (num_slots,) in [0, 1]

        if return_validity:
            # Return everything (for training with validity loss)
            return all_coords, validity_scores
        else:
            # Filter to valid slots only (for inference)
            # Use threshold of 0.5
            valid_mask = validity_scores > 0.5
            valid_points = all_coords[valid_mask]
            return valid_points
