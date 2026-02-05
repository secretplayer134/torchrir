from __future__ import annotations

"""ISM API for static and dynamic RIR simulation."""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from ..config import SimulationConfig, default_config
from ..directivity import split_directivity
from ...models import MicrophoneArray, Room, Source
from ...util.device import infer_device_dtype, resolve_device
from ...util.orientation import orientation_to_unit
from ...util.tensor import as_tensor, ensure_dim
from .accumulate import _accumulate_rir_batch
from .contributions import (
    _compute_image_contributions_batch,
    _compute_image_contributions_time_batch,
)
from .diffuse import _apply_diffuse_tail
from .helpers import _prepare_entities, _resolve_beta, _validate_beta
from .images import _image_source_indices, _reflection_coefficients


def simulate_rir(
    *,
    room: Room,
    sources: Source | Tensor,
    mics: MicrophoneArray | Tensor,
    max_order: int | None,
    nb_img: Optional[Tensor | Tuple[int, ...]] = None,
    nsample: Optional[int] = None,
    tmax: Optional[float] = None,
    tdiff: Optional[float] = None,
    directivity: str | tuple[str, str] | None = "omni",
    orientation: Optional[Tensor | tuple[Tensor, Tensor]] = None,
    config: Optional[SimulationConfig] = None,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Simulate a static RIR using the image source method."""
    cfg = config or default_config()
    cfg.validate()

    if device is None and cfg.device is not None:
        device = cfg.device

    if max_order is None:
        if cfg.max_order is None:
            raise ValueError("max_order must be provided if not set in config")
        max_order = cfg.max_order

    if tmax is None and nsample is None and cfg.tmax is not None:
        tmax = cfg.tmax

    if directivity is None:
        directivity = cfg.directivity or "omni"

    if not isinstance(room, Room):
        raise TypeError("room must be a Room instance")
    if nsample is None:
        if tmax is None:
            raise ValueError("nsample or tmax must be provided")
        nsample = int(math.ceil(tmax * room.fs))
    if nsample <= 0:
        raise ValueError("nsample must be positive")
    if max_order < 0:
        raise ValueError("max_order must be non-negative")

    if isinstance(device, str):
        device = resolve_device(device)

    src_pos, src_ori = _prepare_entities(
        sources, orientation, which="source", device=device, dtype=dtype
    )
    mic_pos, mic_ori = _prepare_entities(
        mics, orientation, which="mic", device=device, dtype=dtype
    )

    device, dtype = infer_device_dtype(
        src_pos, mic_pos, room.size, device=device, dtype=dtype
    )
    src_pos = as_tensor(src_pos, device=device, dtype=dtype)
    mic_pos = as_tensor(mic_pos, device=device, dtype=dtype)

    if src_ori is not None:
        src_ori = as_tensor(src_ori, device=device, dtype=dtype)
    if mic_ori is not None:
        mic_ori = as_tensor(mic_ori, device=device, dtype=dtype)

    room_size = as_tensor(room.size, device=device, dtype=dtype)
    room_size = ensure_dim(room_size)
    dim = room_size.numel()

    if src_pos.ndim == 1:
        src_pos = src_pos.unsqueeze(0)
    if mic_pos.ndim == 1:
        mic_pos = mic_pos.unsqueeze(0)
    if src_pos.ndim != 2 or src_pos.shape[1] != dim:
        raise ValueError("sources must be of shape (n_src, dim)")
    if mic_pos.ndim != 2 or mic_pos.shape[1] != dim:
        raise ValueError("mics must be of shape (n_mic, dim)")

    beta = _resolve_beta(room, room_size, device=device, dtype=dtype)
    beta = _validate_beta(beta, dim)

    n_vec = _image_source_indices(max_order, dim, device=device, nb_img=nb_img)
    refl = _reflection_coefficients(n_vec, beta)

    src_pattern, mic_pattern = split_directivity(directivity)
    mic_dir = None
    if mic_pattern != "omni":
        if mic_ori is None:
            raise ValueError("mic orientation required for non-omni directivity")
        mic_dir = orientation_to_unit(mic_ori, dim)

    n_src = src_pos.shape[0]
    n_mic = mic_pos.shape[0]
    rir = torch.zeros((n_src, n_mic, nsample), device=device, dtype=dtype)
    fdl = cfg.frac_delay_length
    fdl2 = (fdl - 1) // 2
    img_chunk = cfg.image_chunk_size
    if img_chunk <= 0:
        img_chunk = n_vec.shape[0]

    src_dirs = None
    if src_pattern != "omni":
        if src_ori is None:
            raise ValueError("source orientation required for non-omni directivity")
        src_dirs = orientation_to_unit(src_ori, dim)
        if src_dirs.ndim == 1:
            src_dirs = src_dirs.unsqueeze(0).repeat(n_src, 1)
        if src_dirs.ndim != 2 or src_dirs.shape[0] != n_src:
            raise ValueError("source orientation must match number of sources")

    for start in range(0, n_vec.shape[0], img_chunk):
        end = min(start + img_chunk, n_vec.shape[0])
        n_vec_chunk = n_vec[start:end]
        refl_chunk = refl[start:end]
        sample_chunk, attenuation_chunk = _compute_image_contributions_batch(
            src_pos,
            mic_pos,
            room_size,
            n_vec_chunk,
            refl_chunk,
            room,
            fdl2,
            src_pattern=src_pattern,
            mic_pattern=mic_pattern,
            src_dirs=src_dirs,
            mic_dir=mic_dir,
        )
        _accumulate_rir_batch(rir, sample_chunk, attenuation_chunk, cfg)

    if tdiff is not None and tmax is not None and tdiff < tmax:
        rir = _apply_diffuse_tail(rir, room, beta, tdiff, tmax, seed=cfg.seed)
    return rir


def simulate_dynamic_rir(
    *,
    room: Room,
    src_traj: Tensor,
    mic_traj: Tensor,
    max_order: int | None,
    nsample: Optional[int] = None,
    tmax: Optional[float] = None,
    directivity: str | tuple[str, str] | None = "omni",
    orientation: Optional[Tensor | tuple[Tensor, Tensor]] = None,
    config: Optional[SimulationConfig] = None,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Simulate time-varying RIRs for source/mic trajectories."""
    cfg = config or default_config()
    cfg.validate()

    if device is None and cfg.device is not None:
        device = cfg.device

    if max_order is None:
        if cfg.max_order is None:
            raise ValueError("max_order must be provided if not set in config")
        max_order = cfg.max_order

    if tmax is None and nsample is None and cfg.tmax is not None:
        tmax = cfg.tmax

    if directivity is None:
        directivity = cfg.directivity or "omni"

    if isinstance(device, str):
        device = resolve_device(device)

    src_traj = as_tensor(src_traj, device=device, dtype=dtype)
    mic_traj = as_tensor(mic_traj, device=device, dtype=dtype)
    device, dtype = infer_device_dtype(
        src_traj, mic_traj, room.size, device=device, dtype=dtype
    )
    src_traj = as_tensor(src_traj, device=device, dtype=dtype)
    mic_traj = as_tensor(mic_traj, device=device, dtype=dtype)

    if src_traj.ndim == 2:
        src_traj = src_traj.unsqueeze(1)
    if mic_traj.ndim == 2:
        mic_traj = mic_traj.unsqueeze(1)
    if src_traj.ndim != 3:
        raise ValueError("src_traj must be of shape (T, n_src, dim)")
    if mic_traj.ndim != 3:
        raise ValueError("mic_traj must be of shape (T, n_mic, dim)")
    if src_traj.shape[0] != mic_traj.shape[0]:
        raise ValueError("src_traj and mic_traj must have the same time length")

    if not isinstance(room, Room):
        raise TypeError("room must be a Room instance")
    if nsample is None:
        if tmax is None:
            raise ValueError("nsample or tmax must be provided")
        nsample = int(math.ceil(tmax * room.fs))
    if nsample <= 0:
        raise ValueError("nsample must be positive")
    if max_order < 0:
        raise ValueError("max_order must be non-negative")

    room_size = as_tensor(room.size, device=device, dtype=dtype)
    room_size = ensure_dim(room_size)
    dim = room_size.numel()
    if src_traj.shape[2] != dim:
        raise ValueError("src_traj must match room dimension")
    if mic_traj.shape[2] != dim:
        raise ValueError("mic_traj must match room dimension")

    src_ori = None
    mic_ori = None
    if orientation is not None:
        if isinstance(orientation, (list, tuple)):
            if len(orientation) != 2:
                raise ValueError("orientation tuple must have length 2")
            src_ori, mic_ori = orientation
        else:
            src_ori = orientation
            mic_ori = orientation
    if src_ori is not None:
        src_ori = as_tensor(src_ori, device=device, dtype=dtype)
    if mic_ori is not None:
        mic_ori = as_tensor(mic_ori, device=device, dtype=dtype)

    beta = _resolve_beta(room, room_size, device=device, dtype=dtype)
    beta = _validate_beta(beta, dim)
    n_vec = _image_source_indices(max_order, dim, device=device, nb_img=None)
    refl = _reflection_coefficients(n_vec, beta)

    src_pattern, mic_pattern = split_directivity(directivity)
    mic_dir = None
    if mic_pattern != "omni":
        if mic_ori is None:
            raise ValueError("mic orientation required for non-omni directivity")
        mic_dir = orientation_to_unit(mic_ori, dim)

    n_src = src_traj.shape[1]
    n_mic = mic_traj.shape[1]
    rirs = torch.zeros(
        (src_traj.shape[0], n_src, n_mic, nsample), device=device, dtype=dtype
    )
    fdl = cfg.frac_delay_length
    fdl2 = (fdl - 1) // 2
    img_chunk = cfg.image_chunk_size
    if img_chunk <= 0:
        img_chunk = n_vec.shape[0]

    src_dirs = None
    if src_pattern != "omni":
        if src_ori is None:
            raise ValueError("source orientation required for non-omni directivity")
        src_dirs = orientation_to_unit(src_ori, dim)
        if src_dirs.ndim == 1:
            src_dirs = src_dirs.unsqueeze(0).repeat(n_src, 1)
        if src_dirs.ndim != 2 or src_dirs.shape[0] != n_src:
            raise ValueError("source orientation must match number of sources")

    for start in range(0, n_vec.shape[0], img_chunk):
        end = min(start + img_chunk, n_vec.shape[0])
        n_vec_chunk = n_vec[start:end]
        refl_chunk = refl[start:end]
        sample_chunk, attenuation_chunk = _compute_image_contributions_time_batch(
            src_traj,
            mic_traj,
            room_size,
            n_vec_chunk,
            refl_chunk,
            room,
            fdl2,
            src_pattern=src_pattern,
            mic_pattern=mic_pattern,
            src_dirs=src_dirs,
            mic_dir=mic_dir,
        )
        t_steps = src_traj.shape[0]
        sample_flat = sample_chunk.reshape(t_steps * n_src, n_mic, -1)
        attenuation_flat = attenuation_chunk.reshape(t_steps * n_src, n_mic, -1)
        rir_flat = rirs.view(t_steps * n_src, n_mic, nsample)
        _accumulate_rir_batch(rir_flat, sample_flat, attenuation_flat, cfg)

    return rirs
