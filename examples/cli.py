from __future__ import annotations

"""Unified CLI for static/dynamic RIR examples."""

import argparse
import random
import sys
from pathlib import Path

import torch

try:
    from torchrir import (
        CmuArcticDataset,
        DynamicConvolver,
        LoggingConfig,
        MicrophoneArray,
        Room,
        Source,
        get_logger,
        plot_scene_and_save,
        resolve_device,
        save_wav,
        setup_logging,
        simulate_dynamic_rir,
        simulate_rir,
    )
except ModuleNotFoundError:  # allow running without installation
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    from torchrir import (
        CmuArcticDataset,
        DynamicConvolver,
        LoggingConfig,
        MicrophoneArray,
        Room,
        Source,
        get_logger,
        plot_scene_and_save,
        resolve_device,
        save_wav,
        setup_logging,
        simulate_dynamic_rir,
        simulate_rir,
    )

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from torchrir import (
    binaural_mic_positions,
    clamp_positions,
    linear_trajectory,
    load_dataset_sources,
    sample_positions,
)


def _dataset_factory(root: Path, download: bool, speaker: str | None):
    spk = speaker or "bdl"
    return CmuArcticDataset(root, speaker=spk, download=download)


def _load_sources(args, rng: random.Random, device: torch.device):
    signals, fs, info = load_dataset_sources(
        dataset_factory=lambda speaker: _dataset_factory(args.dataset_dir, args.download, speaker),
        num_sources=args.num_sources,
        duration_s=args.duration,
        rng=rng,
    )
    return signals.to(device), fs, info


def _plot_scene(args, room, sources, mics, src_traj=None, mic_traj=None, prefix="scene"):
    if not args.plot:
        return
    try:
        plot_scene_and_save(
            out_dir=args.out_dir,
            room=room.size,
            sources=sources,
            mics=mics,
            src_traj=src_traj,
            mic_traj=mic_traj,
            prefix=prefix,
            show=args.show,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        logger = get_logger("examples.cli")
        logger.warning("Plot skipped: %s", exc)


def _run_static(args, rng: random.Random, logger):
    device = resolve_device(args.device)
    signals, fs, info = _load_sources(args, rng, device)
    room = Room.shoebox(size=args.room, fs=fs, beta=[0.9] * (6 if len(args.room) == 3 else 4))
    room_size = torch.tensor(args.room, dtype=torch.float32)

    sources_pos = sample_positions(num=args.num_sources, room_size=room_size, rng=rng)
    mic_center = sample_positions(num=1, room_size=room_size, rng=rng).squeeze(0)
    mic_pos = clamp_positions(binaural_mic_positions(mic_center), room_size)

    sources = Source.positions(sources_pos.tolist())
    mics = MicrophoneArray.positions(mic_pos.tolist())

    _plot_scene(args, room, sources, mics, prefix="static")

    rirs = simulate_rir(
        room=room,
        sources=sources,
        mics=mics,
        max_order=args.order,
        tmax=args.tmax,
        device=device,
    )
    from torchrir import convolve_rir

    y = convolve_rir(signals, rirs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "static_binaural.wav"
    save_wav(out_path, y, fs)
    logger.info("sources: %s", info)
    logger.info("RIR shape: %s", tuple(rirs.shape))
    logger.info("output shape: %s", tuple(y.shape))
    logger.info("saved: %s", out_path)


def _run_dynamic_src(args, rng: random.Random, logger):
    device = resolve_device(args.device)
    signals, fs, info = _load_sources(args, rng, device)
    room = Room.shoebox(size=args.room, fs=fs, beta=[0.9] * (6 if len(args.room) == 3 else 4))
    room_size = torch.tensor(args.room, dtype=torch.float32)

    steps = max(2, args.steps)
    src_start = sample_positions(num=args.num_sources, room_size=room_size, rng=rng)
    src_end = sample_positions(num=args.num_sources, room_size=room_size, rng=rng)
    src_traj = torch.stack(
        [linear_trajectory(src_start[i], src_end[i], steps) for i in range(args.num_sources)],
        dim=1,
    )
    src_traj = clamp_positions(src_traj, room_size).to(device)

    mic_center = sample_positions(num=1, room_size=room_size, rng=rng).squeeze(0)
    mic_pos = clamp_positions(binaural_mic_positions(mic_center), room_size)
    mic_traj = mic_pos.unsqueeze(0).repeat(steps, 1, 1).to(device)

    sources = Source.positions(src_start.tolist())
    mics = MicrophoneArray.positions(mic_pos.tolist())

    _plot_scene(args, room, sources, mics, src_traj=src_traj, mic_traj=mic_traj, prefix="dynamic_src")

    rirs = simulate_dynamic_rir(
        room=room,
        src_traj=src_traj,
        mic_traj=mic_traj,
        max_order=args.order,
        tmax=args.tmax,
        device=device,
    )
    y = DynamicConvolver(mode="trajectory").convolve(signals, rirs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "dynamic_src_binaural.wav"
    save_wav(out_path, y, fs)
    logger.info("sources: %s", info)
    logger.info("dynamic RIR shape: %s", tuple(rirs.shape))
    logger.info("output shape: %s", tuple(y.shape))
    logger.info("saved: %s", out_path)


def _run_dynamic_mic(args, rng: random.Random, logger):
    device = resolve_device(args.device)
    signals, fs, info = _load_sources(args, rng, device)
    room = Room.shoebox(size=args.room, fs=fs, beta=[0.9] * (6 if len(args.room) == 3 else 4))
    room_size = torch.tensor(args.room, dtype=torch.float32)

    sources_pos = sample_positions(num=args.num_sources, room_size=room_size, rng=rng)
    steps = max(2, args.steps)
    mic_center_start = sample_positions(num=1, room_size=room_size, rng=rng).squeeze(0)
    mic_center_end = sample_positions(num=1, room_size=room_size, rng=rng).squeeze(0)
    mic_center_traj = linear_trajectory(mic_center_start, mic_center_end, steps)
    mic_traj = torch.stack([binaural_mic_positions(center) for center in mic_center_traj], dim=0)
    mic_traj = clamp_positions(mic_traj, room_size).to(device)

    src_traj = sources_pos.unsqueeze(0).repeat(steps, 1, 1).to(device)

    sources = Source.positions(sources_pos.tolist())
    mics = MicrophoneArray.positions(mic_traj[0].tolist())

    _plot_scene(args, room, sources, mics, src_traj=src_traj, mic_traj=mic_traj, prefix="dynamic_mic")

    rirs = simulate_dynamic_rir(
        room=room,
        src_traj=src_traj,
        mic_traj=mic_traj,
        max_order=args.order,
        tmax=args.tmax,
        device=device,
    )
    y = DynamicConvolver(mode="trajectory").convolve(signals, rirs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "dynamic_mic_binaural.wav"
    save_wav(out_path, y, fs)
    logger.info("sources: %s", info)
    logger.info("dynamic RIR shape: %s", tuple(rirs.shape))
    logger.info("output shape: %s", tuple(y.shape))
    logger.info("saved: %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified CMU ARCTIC RIR examples")
    parser.add_argument("--mode", choices=("static", "dynamic_src", "dynamic_mic"), default="static")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/cmu_arctic"))
    parser.add_argument("--download", action="store_true", default=True)
    parser.add_argument("--no-download", action="store_false", dest="download")
    parser.add_argument("--num-sources", type=int, default=2)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--room", type=float, nargs="+", default=[6.0, 4.0, 3.0])
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--tmax", type=float, default=0.4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--plot", action="store_true", help="plot room and trajectories")
    parser.add_argument("--show", action="store_true", help="show plots interactively")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(LoggingConfig(level=args.log_level))
    logger = get_logger("examples.cli")
    rng = random.Random(args.seed)

    if args.mode == "static":
        _run_static(args, rng, logger)
    elif args.mode == "dynamic_src":
        _run_dynamic_src(args, rng, logger)
    elif args.mode == "dynamic_mic":
        _run_dynamic_mic(args, rng, logger)
    else:
        raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
