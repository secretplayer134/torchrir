# Getting Started

This page shows the minimum workflow for the core TorchRIR APIs:

1. Define room / source / microphone geometry.
2. Simulate static or dynamic RIRs.
3. Convolve dry signals with the generated RIRs.

## Install

```bash
pip install torchrir
```

## 1) Static RIR + Convolution

```python
import torch

from torchrir import MicrophoneArray, Room, Source
from torchrir.signal import convolve_rir
from torchrir.sim import simulate_rir

fs = 16000

room = Room.shoebox(size=[6.0, 4.0, 3.0], fs=fs, beta=[0.9] * 6)
sources = Source.from_positions([[1.0, 1.5, 1.2]])
mics = MicrophoneArray.from_positions([[2.5, 2.0, 1.2], [2.7, 2.0, 1.2]])

rirs = simulate_rir(
    room=room,
    sources=sources,
    mics=mics,
    max_order=6,
    tmax=0.3,
    directivity="omni",
    device="auto",
)
print("static RIR shape:", tuple(rirs.shape))  # (n_src, n_mic, rir_len)

dry = torch.randn(1, fs * 2)  # (n_src, n_samples)
wet = convolve_rir(dry, rirs)
print("convolved shape:", tuple(wet.shape))  # (n_mic, n_samples + rir_len - 1)
```

## 2) Dynamic RIR + Trajectory Convolution

```python
import torch

from torchrir import Room
from torchrir.signal import DynamicConvolver
from torchrir.sim import simulate_dynamic_rir

fs = 16000
room = Room.shoebox(size=[6.0, 4.0, 3.0], fs=fs, beta=[0.9] * 6)

steps = 16
src_start = torch.tensor([1.0, 1.2, 1.0], dtype=torch.float32)
src_end = torch.tensor([3.0, 2.4, 1.2], dtype=torch.float32)
alpha = torch.linspace(0.0, 1.0, steps, dtype=torch.float32).view(steps, 1, 1)
src_traj = src_start.view(1, 1, 3) + alpha * (src_end - src_start).view(1, 1, 3)

# Fixed microphone in a dynamic test: repeat the same position over time.
mic_pos = torch.tensor([[2.5, 2.0, 1.2]], dtype=torch.float32)
mic_traj = mic_pos.unsqueeze(0).repeat(steps, 1, 1)

dynamic_rirs = simulate_dynamic_rir(
    room=room,
    src_traj=src_traj,
    mic_traj=mic_traj,
    max_order=6,
    tmax=0.3,
    directivity="omni",
    device="auto",
)
print("dynamic RIR shape:", tuple(dynamic_rirs.shape))  # (T, n_src, n_mic, rir_len)

dry = torch.randn(1, fs * 2)  # (n_src, n_samples)
wet = DynamicConvolver(mode="trajectory").convolve(dry, dynamic_rirs)
print("dynamic convolved shape:", tuple(wet.shape))
```

## Next Steps

- See [Examples](examples.md) for CLI workflows and dataset generation scripts.
- See [API documentation](api.md) for all options and full signatures.
