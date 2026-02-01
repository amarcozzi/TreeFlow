"""
models/__init__.py
"""

from .transformer import FlowMatchingDiT
from .pointnext import FlowMatchingPointNeXt


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

    if model_type == "dit":
        print(f"Initializing DiT (Transformer) Model...")
        model = FlowMatchingDiT(
            model_dim=args.model_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_species=num_species,
            num_types=num_types,
            dropout=args.dropout,
        )

    elif model_type == "pointnext":
        print(f"Initializing PointNeXt U-Net Model...")
        model = FlowMatchingPointNeXt(
            model_dim=args.model_dim,
            num_species=num_species,
            num_types=num_types,
            dropout=args.dropout,
        )

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Choices: ['dit', 'pointnext']"
        )

    if device is not None:
        model = model.to(device)

    return model
