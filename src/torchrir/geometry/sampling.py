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


def clamp_positions(
    positions: Tensor, room_size: Tensor, margin: float = 0.1
) -> Tensor:
    """Clamp positions to remain inside the room with a margin."""
    min_v = torch.full_like(room_size, margin)
    max_v = room_size - margin
    return torch.max(torch.min(positions, max_v), min_v)
