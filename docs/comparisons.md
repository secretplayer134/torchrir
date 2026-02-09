# Library Comparisons

This page summarizes implementation-level differences between TorchRIR and related libraries in this scope:
- `torchrir`
- `gpuRIR`
- `rir-generator`
- `pyroomacoustics`

## Dynamic Simulation Feature Comparison

| Feature | `torchrir` | `gpuRIR` | `pyroomacoustics` | `rir-generator` |
|---|---|---|---|---|
| 🎯 Dynamic Sources | ✅ | 🟡 Single-source workflow | 🟡 Manual loop | ❌ |
| 🎤 Dynamic Microphones | ✅ | ❌ | 🟡 Manual loop | ❌ |
| 🖥️ CPU | ✅ | ❌ | ✅ | ✅ |
| 🧮 CUDA | 🚧 Coming soon | ✅ | ❌ | ❌ |
| 🍎 MPS | ✅ | ❌ | ❌ | ❌ |
| 📊 Visualization | ✅ | ❌ | ✅ | ❌ |
| 🗂️ Dataset Build | ✅ | ❌ | 🟡 Custom scripts | ❌ |

Legend:
- `✅` native support
- `🟡` manual setup
- `🚧` coming soon
- `❌` unavailable

## ISM High-Pass Filter (HPF) Implementations

This section focuses on libraries (in this comparison scope) that implement a built-in HPF for ISM-generated RIRs: `torchrir`, `rir-generator`, and `pyroomacoustics`.

### `torchrir`: parameterization and equations

Defaults:
- `rir_hpf_enable=True`
- `rir_hpf_fc=10.0` (Hz)
- `rir_hpf_kwargs={"n": 2, "rp": 5.0, "rs": 60.0, "type": "butter"}`

For sampling frequency `f_s` and cutoff `f_c`, the normalized digital cutoff is:

```{math}
w_c = \frac{2f_c}{f_s}
```

Second-order sections are designed as:

```{math}
\mathrm{SOS} = \mathrm{iirfilter}\left(
n,\; W_n=w_c,\; rp,\; rs,\;
\text{btype}=\text{"highpass"},\;
\text{ftype}=\mathrm{type},\;
\text{output}=\text{"sos"}
\right)
```

For each generated RIR tensor `x`, TorchRIR applies:

```{math}
y = \mathrm{sosfiltfilt}(\mathrm{SOS}, x)
```

along the time axis (last dimension), i.e. static `(n_src, n_mic, nsample)` and dynamic `(T, n_src, n_mic, nsample)` outputs are filtered in-place along `nsample`.

### `rir-generator`: parameterization and equations

Default: `hp_filter=True`.

The filter coefficients are derived from sampling frequency `f_s`:

```{math}
W = \frac{2\pi f_c}{f_s} = \frac{2\pi \cdot 100}{f_s}
```

```{math}
R_1 = e^{-W}, \quad
B_1 = 2R_1\cos(W), \quad
B_2 = -R_1^2, \quad
A_1 = -(1+R_1)
```

With input sample `x[n]`, internal state `v[n]`, and output `y[n]`:

```{math}
v[n] = x[n] + B_1 v[n-1] + B_2 v[n-2]
```

```{math}
y[n] = v[n] + A_1 v[n-1] + R_1 v[n-2]
```

State is initialized to zero (`v[-1] = v[-2] = 0`).

### `pyroomacoustics`: parameterization and equations

Defaults:
- `rir_hpf_enable=True`
- `rir_hpf_fc=10.0` (Hz)
- `rir_hpf_kwargs={"n": 2, "rp": 5.0, "rs": 60.0, "type": "butter"}`

For sampling frequency `f_s` and cutoff `f_c`, the normalized digital cutoff is:

```{math}
w_c = \frac{2f_c}{f_s}
```

Second-order sections are designed as:

```{math}
\mathrm{SOS} = \mathrm{iirfilter}\left(
n,\; W_n=w_c,\; rp,\; rs,\;
\text{btype}=\text{"highpass"},\;
\text{ftype}=\mathrm{type},\;
\text{output}=\text{"sos"}
\right)
```

For each generated RIR `x`, the library applies:

```{math}
y = \mathrm{sosfiltfilt}(\mathrm{SOS}, x)
```

This is forward-backward filtering (zero-phase response).
