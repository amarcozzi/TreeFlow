"""
treeflow/sample.py

Core sampling functions for generating tree point clouds using trained Flow Matching models.
Used by both train.py (for visualization) and generate_samples.py (for batch generation).
"""

import torch
import numpy as np
from typing import List, Union
from flow_matching.solver import ODESolver


@torch.no_grad()
def sample_conditional(
    model,
    num_points: int,
    device: torch.device,
    target_height: float,
    species_idx: int,
    type_idx: int,
    cfg_values: Union[float, List[float]],
    num_steps: int = 50,
    solver_method: str = "dopri5",
    batch_size: int = 0,
) -> np.ndarray:
    """
    Generate samples for a single tree with specified CFG value(s).
    If multiple CFG values are provided, generates one sample per value in a batch.

    Args:
        model: Trained flow matching model
        num_points: Number of points per sample
        device: Torch device (cuda/cpu)
        target_height: Target tree height in meters
        species_idx: Species index for conditioning
        type_idx: Scan type index for conditioning
        cfg_values: Single CFG value or list of CFG values (one per sample)
        num_steps: Number of ODE solver steps (for non-adaptive solvers)
        solver_method: ODE solver method ('euler', 'midpoint', 'dopri5')
        batch_size: Max samples per ODE solve. 0 = all at once.

    Returns:
        np.ndarray: Point clouds of shape (num_samples, num_points, 3) in normalized coordinates
                    If single cfg_value was provided, returns (num_points, 3)
    """
    model.eval()

    # Handle single CFG value
    single_sample = not isinstance(cfg_values, list)
    if single_sample:
        cfg_values = [cfg_values]

    num_samples = len(cfg_values)

    # If batch_size is set and smaller than num_samples, chunk the generation
    if batch_size > 0 and batch_size < num_samples:
        all_results = []
        for i in range(0, num_samples, batch_size):
            chunk_cfg = cfg_values[i : i + batch_size]
            chunk_result = sample_conditional(
                model=model,
                num_points=num_points,
                device=device,
                target_height=target_height,
                species_idx=species_idx,
                type_idx=type_idx,
                cfg_values=chunk_cfg,
                num_steps=num_steps,
                solver_method=solver_method,
                batch_size=0,  # no further chunking
            )
            all_results.append(chunk_result)
        x_final = np.concatenate(all_results, axis=0)
        if single_sample:
            return x_final[0]
        return x_final

    # Prepare batch (all samples share same conditioning except CFG)
    x_init = torch.randn(num_samples, num_points, 3, device=device)
    s_tensor = torch.full((num_samples,), species_idx, device=device, dtype=torch.long)
    t_tensor = torch.full((num_samples,), type_idx, device=device, dtype=torch.long)
    h_val_log = np.log(target_height + 1e-6)
    h_tensor = torch.full((num_samples,), h_val_log, device=device, dtype=torch.float32)

    # Convert cfg_values to tensor for vectorized operations
    cfg_tensor = torch.tensor(cfg_values, device=device, dtype=torch.float32)

    # Check if we can skip conditional pass (all cfg values are 0)
    all_cfg_zero = (cfg_tensor == 0).all().item()

    # ODE Function with per-sample CFG
    def ode_fn(t, x):
        bs = x.shape[0]
        t_batch = torch.full((bs,), t, device=device, dtype=x.dtype)

        # 1. Unconditional Pass
        drop_mask_uncond = torch.ones(bs, dtype=torch.bool, device=device)
        v_uncond = model(
            x, t_batch, s_tensor, t_tensor, h_tensor, drop_mask=drop_mask_uncond
        )

        # 2. Conditional Pass (skip if all CFG values are 0)
        if all_cfg_zero:
            return v_uncond

        drop_mask_cond = torch.zeros(bs, dtype=torch.bool, device=device)
        v_cond = model(
            x, t_batch, s_tensor, t_tensor, h_tensor, drop_mask=drop_mask_cond
        )

        # Apply per-sample CFG scaling: v = v_uncond + cfg * (v_cond - v_uncond)
        # cfg_tensor shape: (num_samples,) -> (num_samples, 1, 1)
        cfg_expanded = cfg_tensor[:, None, None]
        return v_uncond + cfg_expanded * (v_cond - v_uncond)

    # Solve ODE
    solver = ODESolver(velocity_model=ode_fn)

    if solver_method == "dopri5":
        x_final = solver.sample(x_init, method="dopri5", step_size=None)
    else:
        step_size = 1.0 / num_steps
        x_final = solver.sample(x_init, method=solver_method, step_size=step_size)

    x_final = x_final.cpu().numpy()

    # Return single sample without batch dimension if single cfg was provided
    if single_sample:
        return x_final[0]

    return x_final
