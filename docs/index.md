# TorchRIR

## Summary
TorchRIR is a PyTorch-based toolkit for room impulse response (RIR) simulation
with CPU/CUDA/MPS support, static and dynamic scenes, and dataset utilities.
TorchRIR is under active development and may contain bugs or breaking changes.
Please validate results for your use case.
If you find bugs or have feature requests, please open an issue.
Contributions are welcome.

## Installation
```bash
pip install torchrir
```

## Overview
### Capabilities
- ISM-based static and dynamic RIR simulation for 2D/3D shoebox rooms.
- Dynamic convolution via trajectory or hop-based modes.
- Scene visualization (plots, GIFs) and metadata export (JSON).
- Dataset utilities for building small mixtures from speech corpora.

### Limitations
- Ray tracing and FDTD simulators are placeholders.
- Deterministic mode is best-effort and backend-dependent.
- MPS disables the LUT path for fractional delay (slower, minor numerical diffs).
- Experimental status: APIs and outputs may change as the library matures.

### Supported datasets
- CMU ARCTIC
- LibriSpeech
- TemplateDataset (stub for future integrations)

### License
TorchRIR is released under the Apache 2.0 license. See `LICENSE`.

## Main features
### Static room acoustic simulation
- Compute static RIRs with `simulate_rir`.
- Convolve dry signals with `convolve_rir`.

### Dynamic room acoustic simulation
- Compute time-varying RIRs with `simulate_dynamic_rir`.
- Convolve with `DynamicConvolver(mode="trajectory")`.

### Building dataset
- Use `load_dataset_sources` to build fixed-length sources.
- Use dataset examples to generate per-scene WAV + metadata.

## API documentation
See the API reference: `api`.

## Index
- {ref}`genindex`

```{toctree}
:maxdepth: 2
:caption: Contents

api
overview
examples
changelog
```
