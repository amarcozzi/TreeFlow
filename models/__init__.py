"""
models/__init__.py
"""

from .transformer import FlowMatchingDiT
from .pointnext import FlowMatchingPointNeXt


def get_model(args, device):
    """
    Factory function to initialize models based on arguments.
    """
    model_type = args.model_type.lower()

    # Determine dimensions for embedding layers
    # (These come from dataset setup in train.py)
    num_species = len(args.species_list)
    num_types = len(args.type_list)

    if model_type == "dit":
        print(f"Initializing DiT (Transformer) Model...")
        return FlowMatchingDiT(
            model_dim=args.model_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_species=num_species,
            num_types=num_types,
            dropout=args.dropout,
        ).to(device)

    elif model_type == "pointnext":
        print(f"Initializing PointNeXt U-Net Model...")
        # Note: PointNeXt uses 'model_dim' as base width (e.g. 32 or 64)
        return FlowMatchingPointNeXt(
            model_dim=args.model_dim,  # e.g., 64
            num_species=num_species,
            num_types=num_types,
            dropout=args.dropout,
        ).to(device)

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Choices: ['dit', 'pointnext']"
        )
