"""
models/__init__.py
"""

from .transformer import FlowMatchingTransformer


def get_model(args, device=None):
    """
    Factory function to initialize models based on arguments.

    Args:
        args: Arguments containing model configuration
        device: Device to move model to. If None, model stays on CPU
                (useful when using accelerator.prepare() to handle device placement)
    """
    model_type = args.model_type.lower()

    # Determine dimensions for embedding layers
    # (These come from dataset setup in train.py)
    num_species = len(args.species_list)
    num_types = len(args.type_list)

    if model_type == "transformer":
        print(f"Initializing Transformer Model (token-prepend + U-ViT skips)...")
        model = FlowMatchingTransformer(
            model_dim=args.model_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_species=num_species,
            num_types=num_types,
            dropout=args.dropout,
            num_freq_bands=getattr(args, "num_freq_bands", 12),
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}. Choices: ['transformer']")

    if device is not None:
        model = model.to(device)

    return model
