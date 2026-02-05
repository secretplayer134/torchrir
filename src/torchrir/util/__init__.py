"""General-purpose math, device, and tensor utilities for torchrir."""

from .acoustics import (
    att2t_SabineEstimation,
    att2t_sabine_estimation,
    beta_SabineEstimation,
    estimate_beta_from_t60,
    estimate_t60_from_beta,
    t2n,
)
from .device import DeviceSpec, infer_device_dtype, resolve_device
from .orientation import normalize_orientation, orientation_to_unit
from .tensor import as_tensor, ensure_dim, extend_size

__all__ = [
    "DeviceSpec",
    "as_tensor",
    "att2t_SabineEstimation",
    "att2t_sabine_estimation",
    "beta_SabineEstimation",
    "ensure_dim",
    "estimate_beta_from_t60",
    "estimate_t60_from_beta",
    "extend_size",
    "infer_device_dtype",
    "normalize_orientation",
    "orientation_to_unit",
    "resolve_device",
    "t2n",
]
