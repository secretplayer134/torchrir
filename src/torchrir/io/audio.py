from __future__ import annotations

"""Audio file utilities (dataset-agnostic)."""

from pathlib import Path
from typing import Tuple

import torch


def load_wav_mono(path: Path) -> Tuple[torch.Tensor, int]:
    """Load a wav/flac file and return mono audio and sample rate.

    Example:
        >>> audio, fs = load_wav_mono(Path("datasets/cmu_arctic/.../arctic_a0001.wav"))
    """
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio_t = torch.from_numpy(audio)
    if audio_t.shape[1] > 1:
        audio_t = audio_t.mean(dim=1)
    else:
        audio_t = audio_t.squeeze(1)
    return audio_t, sample_rate


def save_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    """Save a mono or multi-channel wav to disk.

    Example:
        >>> save_wav(Path("outputs/example.wav"), audio, sample_rate)
    """
    import soundfile as sf

    audio = audio.detach().cpu().clamp(-1.0, 1.0).to(torch.float32)
    if audio.ndim == 2 and audio.shape[0] <= 8:
        audio = audio.transpose(0, 1)
    sf.write(str(path), audio.numpy(), sample_rate)
