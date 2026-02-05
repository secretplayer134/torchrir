from __future__ import annotations

"""Sampling helpers for scene geometry."""

import random
from typing import List

import torch
from torch import Tensor


def sample_positions(
    *,
    num: int,
    room_size: Tensor,
    rng: random.Random,
    margin: float = 0.5,
) -> Tensor:
    """Sample random positions within a room with a safety margin."""
    dim = room_size.numel()
    low = [margin] * dim
    high = [float(room_size[i].item()) - margin for i in range(dim)]
    coords: List[List[float]] = []
    for _ in range(num):
        point = [rng.uniform(low[i], high[i]) for i in range(dim)]
        coords.append(point)
    return torch.tensor(coords, dtype=torch.float32)


def sample_positions_with_z_range(
    *,
    num: int,
    room_size: Tensor,
    rng: random.Random,
    z_range: tuple[float, float] = (1.5, 1.8),
    margin: float = 0.5,
) -> Tensor:
    """Sample random positions with an explicit z-range constraint."""
    positions = sample_positions(num=num, room_size=room_size, rng=rng, margin=margin)
    if room_size.numel() < 3:
        return positions
    z_min, z_max = z_range
    z_low = max(margin, float(z_min))
    z_high = min(float(room_size[2].item()) - margin, float(z_max))
    if z_high <= z_low:
        return positions
    z_vals = [rng.uniform(z_low, z_high) for _ in range(num)]
    positions[:, 2] = torch.tensor(z_vals, dtype=positions.dtype)
    return positions


def clamp_positions(
    positions: Tensor, room_size: Tensor, margin: float = 0.1
) -> Tensor:
    """Clamp positions to remain inside the room with a margin."""
    min_v = torch.full_like(room_size, margin)
    max_v = room_size - margin
    return torch.max(torch.min(positions, max_v), min_v)
