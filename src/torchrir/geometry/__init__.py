"""Geometry helpers for arrays, trajectories, and sampling.

Includes standard array layouts (linear, circular, polyhedron, binaural,
Eigenmike) plus position sampling utilities.
"""

from .arrays import (
    binaural_array,
    circular_array,
    eigenmike_em32,
    eigenmike_em64,
    linear_array,
    polyhedron_array,
)
from .sampling import clamp_positions, sample_positions
from .trajectories import linear_trajectory

__all__ = [
    "binaural_array",
    "circular_array",
    "clamp_positions",
    "eigenmike_em32",
    "eigenmike_em64",
    "linear_array",
    "linear_trajectory",
    "polyhedron_array",
    "sample_positions",
]
