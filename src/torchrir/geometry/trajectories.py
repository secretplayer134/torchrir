from __future__ import annotations

"""Trajectory helpers for dynamic scenes."""

import torch
from torch import Tensor


def linear_trajectory(start: Tensor, end: Tensor, steps: int) -> Tensor:
    """Create a linear trajectory between start and end."""
    return torch.stack(
        [start + (end - start) * t / (steps - 1) for t in range(steps)],
        dim=0,
    )
