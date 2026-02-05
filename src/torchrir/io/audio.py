from __future__ import annotations

"""Audio file utilities (dataset-agnostic)."""

from pathlib import Path
from typing import Tuple
import warnings

import torch


def load_wav_mono(path: Path) -> Tuple[torch.Tensor, int]:
    """Load a wav/flac file and return mono audio and sample rate.

    Notes:
        - Multichannel input uses channel 0 only (warns).
        - The original file subtype/format are stored on the returned tensor
          as `_torchrir_subtype` and `_torchrir_format` for reuse by save_wav.

    Example:
        >>> audio, fs = load_wav_mono(Path("datasets/cmu_arctic/.../arctic_a0001.wav"))
    """
    import soundfile as sf

    info = sf.info(str(path))
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio_t = torch.from_numpy(audio)
    if audio_t.shape[1] > 1:
        warnings.warn(
            f"load_wav_mono received {audio_t.shape[1]} channels; using channel 0 only.",
            RuntimeWarning,
        )
        audio_t = audio_t[:, 0]
    else:
        audio_t = audio_t.squeeze(1)
    setattr(audio_t, "_torchrir_subtype", info.subtype)
    setattr(audio_t, "_torchrir_format", info.format)
    return audio_t, sample_rate


def save_wav(
    path: Path,
    audio: torch.Tensor,
    sample_rate: int,
    *,
    normalize: bool = True,
    peak: float = 1.0,
    subtype: str | None = None,
) -> None:
    """Save a mono or multi-channel wav to disk.

    By default this normalizes to the specified peak and preserves the input
    file subtype when `subtype=None` and the tensor came from `load_wav_mono`.
    Values outside [-1, 1] are preserved when normalization is disabled.

    Example:
        >>> save_wav(Path("outputs/example.wav"), audio, sample_rate)
    """
    import soundfile as sf

    audio = audio.detach().cpu().to(torch.float32)
    if normalize:
        if peak <= 0:
            raise ValueError("peak must be positive when normalize=True")
        max_val = float(audio.abs().max().item()) if audio.numel() else 0.0
        if max_val > 0:
            audio = audio / max_val * peak
    if audio.ndim == 2 and audio.shape[0] <= 8:
        audio = audio.transpose(0, 1)
    if subtype is None:
        subtype = getattr(audio, "_torchrir_subtype", None)
    sf.write(str(path), audio.numpy(), sample_rate, subtype=subtype)
