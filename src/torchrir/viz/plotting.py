from __future__ import annotations

"""Matplotlib-based plotting helpers for room scenes."""

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..models import MicrophoneArray, Room, Source
from ..util.tensor import as_tensor, ensure_dim


def plot_scene_static(
    *,
    room: Room | Sequence[float] | Tensor,
    sources: Source | Tensor | Sequence,
    mics: MicrophoneArray | Tensor | Sequence,
    ax: Any | None = None,
    title: Optional[str] = None,
    show: bool = False,
):
    """Plot a static room with source and mic positions.

    Example:
        >>> ax = plot_scene_static(
        ...     room=[6.0, 4.0, 3.0],
        ...     sources=[[1.0, 2.0, 1.5]],
        ...     mics=[[2.0, 2.0, 1.5]],
        ... )
    """
    plt, ax = _setup_axes(ax, room)

    size = _room_size(room, ax)
    _draw_room(ax, size)

    src = _extract_positions(sources, ax)
    mic = _extract_positions(mics, ax)

    _scatter_positions(ax, src, label="sources", marker="^")
    _scatter_positions(ax, mic, label="mics", marker="o")

    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    if show:
        plt.show()
    return ax


def plot_scene_dynamic(
    *,
    room: Room | Sequence[float] | Tensor,
    src_traj: Tensor | Sequence,
    mic_traj: Tensor | Sequence,
    step: int = 1,
    src_pos: Optional[Tensor | Sequence] = None,
    mic_pos: Optional[Tensor | Sequence] = None,
    ax: Any | None = None,
    title: Optional[str] = None,
    show: bool = False,
):
    """Plot source and mic trajectories within a room.

    If trajectories are static, only positions are plotted.

    Example:
        >>> ax = plot_scene_dynamic(
        ...     room=[6.0, 4.0, 3.0],
        ...     src_traj=src_traj,
        ...     mic_traj=mic_traj,
        ... )
    """
    plt, ax = _setup_axes(ax, room)

    size = _room_size(room, ax)
    _draw_room(ax, size)

    src_traj = _as_trajectory(src_traj)
    mic_traj = _as_trajectory(mic_traj)
    src_pos_t = _extract_positions(src_pos, ax) if src_pos is not None else src_traj[0]
    mic_pos_t = _extract_positions(mic_pos, ax) if mic_pos is not None else mic_traj[0]

    _plot_entity(ax, src_traj, src_pos_t, step=step, label="sources", marker="^")
    _plot_entity(ax, mic_traj, mic_pos_t, step=step, label="mics", marker="o")

    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    if show:
        plt.show()
    return ax


def _setup_axes(
    ax: Any | None, room: Room | Sequence[float] | Tensor
) -> tuple[Any, Any]:
    """Create 2D/3D axes based on room dimension."""
    import matplotlib.pyplot as plt

    size = _room_size(room, ax)
    dim = size.numel()
    if ax is None:
        if dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            _, ax = plt.subplots()
    return plt, ax


def _room_size(room: Room | Sequence[float] | Tensor, ax: Any | None) -> Tensor:
    """Normalize room size input to a 1D tensor."""
    if isinstance(room, Room):
        size = room.size
    else:
        size = room
    size = as_tensor(size)
    size = ensure_dim(size)
    return size


def _draw_room(ax: Any, size: Tensor) -> None:
    """Draw a 2D or 3D room outline."""
    dim = size.numel()
    if dim == 2:
        _draw_room_2d(ax, size)
    else:
        _draw_room_3d(ax, size)


def _draw_room_2d(ax: Any, size: Tensor) -> None:
    """Draw a 2D rectangular room."""
    import matplotlib.patches as patches

    rect = patches.Rectangle(
        (0.0, 0.0), size[0].item(), size[1].item(), fill=False, edgecolor="black"
    )
    ax.add_patch(rect)
    ax.set_xlim(0, size[0].item())
    ax.set_ylim(0, size[1].item())
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _draw_room_3d(ax: Any, size: Tensor) -> None:
    """Draw a 3D box representing the room."""
    x, y, z = size.tolist()
    corners = torch.tensor(
        [
            [0, 0, 0],
            [x, 0, 0],
            [x, y, 0],
            [0, y, 0],
            [0, 0, z],
            [x, 0, z],
            [x, y, z],
            [0, y, z],
        ],
        dtype=torch.float32,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for a, b in edges:
        ax.plot(
            [corners[a, 0], corners[b, 0]],
            [corners[a, 1], corners[b, 1]],
            [corners[a, 2], corners[b, 2]],
            color="black",
        )
    ax.set_xlim(0, x)
    ax.set_ylim(0, y)
    ax.set_zlim(0, z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def _extract_positions(
    entity: Source | MicrophoneArray | Tensor | Sequence, ax: Any | None
) -> Tensor:
    """Extract positions from Source/MicrophoneArray or raw tensor."""
    if isinstance(entity, (Source, MicrophoneArray)):
        pos = entity.positions
    else:
        pos = entity
    pos = as_tensor(pos)
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    return pos


def _scatter_positions(
    ax: Any,
    positions: Tensor,
    *,
    label: str,
    marker: str,
    color: Optional[str] = None,
) -> None:
    """Scatter-plot positions in 2D or 3D."""
    if positions.numel() == 0:
        return
    dim = positions.shape[1]
    if dim == 2:
        ax.scatter(
            positions[:, 0], positions[:, 1], label=label, marker=marker, color=color
        )
    else:
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            label=label,
            marker=marker,
            color=color,
        )


def _as_trajectory(traj: Tensor | Sequence) -> Tensor:
    """Validate and normalize a trajectory tensor."""
    traj = as_tensor(traj)
    if traj.ndim != 3:
        raise ValueError("trajectory must be of shape (T, N, dim)")
    return traj


def _plot_entity(
    ax: Any,
    traj: Tensor,
    positions: Tensor,
    *,
    step: int,
    label: str,
    marker: str,
) -> None:
    """Plot trajectories and/or static positions with a unified legend entry."""
    if traj.numel() == 0:
        return
    import matplotlib.pyplot as plt

    if positions.shape != traj.shape[1:]:
        positions = traj[0]
    moving = _is_moving(traj, positions)
    colors = plt.rcParams.get("axes.prop_cycle", None)
    if colors is not None:
        palette = colors.by_key().get("color", [])
    else:
        palette = []
    if not palette:
        palette = ["C0", "C1", "C2", "C3", "C4", "C5"]

    dim = traj.shape[2]
    for idx in range(traj.shape[1]):
        color = palette[idx % len(palette)]
        lbl = label if idx == 0 else "_nolegend_"
        if moving:
            if dim == 2:
                xy = traj[::step, idx]
                ax.plot(
                    xy[:, 0],
                    xy[:, 1],
                    label=lbl,
                    color=color,
                    marker=marker,
                    markevery=[0],
                )
            else:
                xyz = traj[::step, idx]
                ax.plot(
                    xyz[:, 0],
                    xyz[:, 1],
                    xyz[:, 2],
                    label=lbl,
                    color=color,
                    marker=marker,
                    markevery=[0],
                )
        pos = positions[idx : idx + 1]
        _scatter_positions(ax, pos, label="_nolegend_", marker=marker, color=color)
    if not moving:
        # ensure legend marker uses the group label
        _scatter_positions(
            ax,
            positions[:1],
            label=label,
            marker=marker,
            color=palette[0],
        )


def _is_moving(traj: Tensor, positions: Tensor, *, tol: float = 1e-6) -> bool:
    """Return True if any trajectory deviates from the provided positions."""
    if traj.numel() == 0:
        return False
    pos0 = positions.unsqueeze(0).expand_as(traj)
    return bool(torch.any(torch.linalg.norm(traj - pos0, dim=-1) > tol).item())


def plot_scene_and_save(
    *,
    out_dir: Path,
    room: Sequence[float] | torch.Tensor,
    sources: object | torch.Tensor | Sequence,
    mics: object | torch.Tensor | Sequence,
    src_traj: Optional[torch.Tensor | Sequence] = None,
    mic_traj: Optional[torch.Tensor | Sequence] = None,
    prefix: str = "scene",
    step: int = 1,
    show: bool = False,
    plot_2d: bool = True,
    plot_3d: bool = True,
) -> tuple[list[Path], list[Path]]:
    """Plot static and dynamic scenes and save images to disk.

    Dynamic plots show trajectories for moving entities and points for fixed ones.

    Args:
        out_dir: Output directory for PNGs.
        room: Room size tensor or sequence.
        sources: Source positions or Source-like object.
        mics: Microphone positions or MicrophoneArray-like object.
        src_traj: Optional source trajectory (T, n_src, dim).
        mic_traj: Optional mic trajectory (T, n_mic, dim).
        prefix: Filename prefix for saved images.
        step: Subsampling step for trajectories.
        show: Whether to show figures interactively.
        plot_2d: Save 2D projections.
        plot_3d: Save 3D projections (only if dim == 3).

    Returns:
        Tuple of (static_paths, dynamic_paths).

    Example:
        >>> plot_scene_and_save(
        ...     out_dir=Path("outputs"),
        ...     room=[6.0, 4.0, 3.0],
        ...     sources=[[1.0, 2.0, 1.5]],
        ...     mics=[[2.0, 2.0, 1.5]],
        ...     src_traj=src_traj,
        ...     mic_traj=mic_traj,
        ...     prefix="scene",
        ... )
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    room_size = _to_cpu(room)
    src_pos = _positions_to_cpu(sources)
    mic_pos = _positions_to_cpu(mics)
    dim = int(room_size.numel())

    static_paths: list[Path] = []
    dynamic_paths: list[Path] = []

    for view_dim, enabled in ((2, plot_2d), (3, plot_3d)):
        if not enabled:
            continue
        if view_dim == 2 and dim < 2:
            continue
        if view_dim == 3 and dim < 3:
            continue
        view_room = room_size[:view_dim]
        view_src = src_pos[:, :view_dim]
        view_mic = mic_pos[:, :view_dim]

        ax = plot_scene_static(
            room=view_room,
            sources=view_src,
            mics=view_mic,
            title=f"Room scene ({view_dim}D static)",
            show=False,
        )
        static_path = out_dir / f"{prefix}_static_{view_dim}d.png"
        _save_axes(ax, static_path, show=show)
        static_paths.append(static_path)

        if src_traj is not None or mic_traj is not None:
            steps = _traj_steps(src_traj, mic_traj)
            src_traj = _trajectory_to_cpu(src_traj, src_pos, steps)
            mic_traj = _trajectory_to_cpu(mic_traj, mic_pos, steps)
            view_src_traj = src_traj[:, :, :view_dim]
            view_mic_traj = mic_traj[:, :, :view_dim]
            ax = plot_scene_dynamic(
                room=view_room,
                src_traj=view_src_traj,
                mic_traj=view_mic_traj,
                src_pos=view_src,
                mic_pos=view_mic,
                step=step,
                title=f"Room scene ({view_dim}D trajectories)",
                show=False,
            )
            dynamic_path = out_dir / f"{prefix}_dynamic_{view_dim}d.png"
            _save_axes(ax, dynamic_path, show=show)
            dynamic_paths.append(dynamic_path)

    return static_paths, dynamic_paths


def _to_cpu(value: Any) -> torch.Tensor:
    """Move a value to CPU as a tensor."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.as_tensor(value).detach().cpu()


def _positions_to_cpu(entity: torch.Tensor | object) -> torch.Tensor:
    """Extract positions from an entity and move to CPU."""
    pos = getattr(entity, "positions", entity)
    pos = _to_cpu(pos)
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    return pos


def _traj_steps(
    src_traj: Optional[torch.Tensor | Sequence],
    mic_traj: Optional[torch.Tensor | Sequence],
) -> int:
    """Infer the number of trajectory steps."""
    if src_traj is not None:
        return int(_to_cpu(src_traj).shape[0])
    return int(_to_cpu(mic_traj).shape[0])


def _trajectory_to_cpu(
    traj: Optional[torch.Tensor | Sequence], fallback_pos: torch.Tensor, steps: int
) -> torch.Tensor:
    """Normalize trajectory to CPU tensor with shape (T, N, dim)."""
    if traj is None:
        return fallback_pos.unsqueeze(0).repeat(steps, 1, 1)
    traj = _to_cpu(traj)
    if traj.ndim != 3:
        raise ValueError("trajectory must be of shape (T, N, dim)")
    return traj


def _save_axes(ax: Any, path: Path, *, show: bool) -> None:
    """Save a matplotlib axis to disk."""
    import matplotlib.pyplot as plt

    fig = ax.figure
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
