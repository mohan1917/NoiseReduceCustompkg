# noisereducecustom

A fully self-contained, from-scratch spectral-gating noise reduction
library for audio, built only on `numpy` and `scipy` (`matplotlib` is
optional, for plotting). It does not import or wrap the third-party
`noisereduce` package in any way — so it keeps working even if that
project is ever pulled from PyPI, paywalled, or relicensed.

## Folder layout
```
noisereducecustom_pkg/          <- unzip and cd into THIS folder
├── noisereducecustom/          <- the actual importable package
│   ├── __init__.py
│   ├── utils.py
│   ├── stft_utils.py
│   ├── base.py
│   ├── stationary.py
│   ├── nonstationary.py
│   ├── core.py
│   ├── generate_noise.py
│   └── plotting.py
├── tests/test_core.py
├── examples/demo.py
├── pyproject.toml
├── setup.py
├── README.md
└── LICENSE
```

## Install (local, editable)
```bash
cd noisereducecustom_pkg      # the folder containing pyproject.toml
pip install -e .
# or, with plotting support:
pip install -e ".[plotting]"
```

Verify it installed correctly:
```bash
python -c "import noisereducecustom as nrc; print(nrc.__version__)"
```
This should print `0.2.0` with no errors, from **any** directory (not
just inside the package folder).

## Quick usage
```python
from scipy.io import wavfile
import numpy as np
import noisereducecustom as nrc

sr, audio = wavfile.read("bird_audio.wav")
if audio.ndim > 1:
    audio = audio.mean(axis=1)
audio = audio.astype(np.float32)

reduced = nrc.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.95)

reduced = reduced / (np.max(np.abs(reduced)) + 1e-9)
wavfile.write("bird_audio_noise_reduced.wav", sr, (reduced * 32767).astype(np.int16))
```

## Run the demo
No .wav file needed — it synthesizes a fake bird chirp + background noise:
```bash
pip install -e ".[demo]"
python examples/demo.py
```
This writes `demo_noisy.wav`, `demo_denoised_stationary.wav`, and
`demo_denoised_nonstationary.wav` to your working directory, and (if
matplotlib is installed) shows a before/after spectrogram plot.

## Live/streaming usage (for continuous mic input)

`reduce_noise()` needs the whole signal up front — it's for complete
files. For a continuous mic feed (e.g. a forest deployment), use the
streaming classes instead. They process audio block by block with
zero added latency, carrying real audio context between calls so
there's no click/pop at block boundaries.

```python
import noisereducecustom as nrc

# Capture a few seconds of quiet ambient sound first (no calls in it)
noise_profile = capture_from_mic(seconds=3)   # your mic code

reducer = nrc.StreamingNoiseReducerStationary(
    sr=16000, y_noise=noise_profile, prop_decrease=0.9
)

while True:
    block = mic.read(block_samples)            # your mic code, e.g. pyaudio
    clean_block = reducer.process_block(block)
    # feed clean_block into your classifier (e.g. YAMNet) here
```

If your background noise changes over time (wind gusts, changing
insect activity) instead of being roughly constant, use
`StreamingNoiseReducerNonStationary` instead — no `y_noise` needed,
it adapts its noise-floor estimate continuously and carries that
state between blocks automatically.

Try it hands-on with no hardware required:
```bash
python examples/live_mic_demo.py
```
This simulates a live mic feed using synthetic audio and streams it
through block by block. Pass `--mic` (and `pip install pyaudio`) to
use a real microphone instead.

## Parallel processing (for long batch/offline files)

For processing long archived recordings, set `chunk_size` and
`n_jobs > 1` to spread work across multiple CPU cores using Python's
built-in `concurrent.futures` (no extra dependency):

```python
reduced = nrc.reduce_noise(
    y=long_audio, sr=sr, stationary=True,
    chunk_size=sr * 30,   # 30-second chunks
    n_jobs=4,             # use 4 CPU processes
)
```

Note: `n_jobs` only matters when `chunk_size` is set on files longer
than that chunk size, and only helps for long files — on short audio,
process-pool startup overhead outweighs the benefit, so `n_jobs=1`
(the default) is faster. This has no relevance to streaming/live mic
input, since chunks there necessarily arrive one at a time regardless.

## Run the tests
```bash
pip install -e .
python -m pytest tests/ -v
# or, without pytest:
python tests/test_core.py
```

---

## What each file does

### `noisereducecustom/__init__.py`
The package's front door. Imports and re-exports the public functions
(`reduce_noise`, `band_limited_noise`, `plotting`) so users can just
`import noisereducecustom as nrc` and call `nrc.reduce_noise(...)`
without knowing the internal file layout.

### `noisereducecustom/utils.py`
Small shared helper functions used everywhere else:
- `ensure_channel_first` — normalizes mono *or* stereo/multi-channel
  input into one consistent internal shape, and remembers the original
  layout so the output can be reshaped back to match.
- `amp_to_db` / `db_to_amp` — convert between linear amplitude and
  decibels (noise gating thresholds are easier to reason about in dB).
- `sigmoid` — a smooth 0→1 step function, used by the non-stationary
  gate so the mask doesn't snap on/off abruptly.
- `triangular_smoothing_kernel` — builds a small pyramid-shaped 2D
  filter used to blur the gating mask across time and frequency, which
  prevents "musical noise" (robotic/watery) artifacts.
- `hz_to_bins` / `ms_to_frames` — convert human-friendly units (Hz,
  milliseconds) into FFT bin counts / frame counts.

### `noisereducecustom/stft_utils.py`
Thin wrappers around `scipy.signal.stft`/`istft` so the rest of the
codebase doesn't repeat window/hop-length bookkeeping. This is where
the actual time-domain ↔ time-frequency-domain conversion happens.

### `noisereducecustom/base.py`
Shared plumbing used by both gating strategies:
- Reshapes input to `(n_channels, n_frames)` and loops the algorithm
  over each channel independently (so stereo files "just work"), then
  restores the original layout on output.
- Optional **chunked processing**: for very long recordings, splits
  the signal into chunks (with a little overlap/padding for context)
  so memory stays bounded and the noise statistics stay locally
  relevant rather than averaged over an entire multi-hour file.
- Optional **parallel chunk processing** (`n_jobs > 1`): farms chunks
  out across multiple CPU processes using Python's built-in
  `concurrent.futures.ProcessPoolExecutor` — no third-party dependency
  like `joblib` needed. Only relevant for long batch files.
- Subclasses only need to implement `_process_channel()` — the actual
  per-channel denoising math.

### `noisereducecustom/stationary.py`
**Stationary mode** (`stationary=True`): assumes the background noise
profile is constant for the whole clip (steady hiss, fan hum, room
tone). Computes one noise threshold per frequency bin (mean + N
standard deviations, in dB) from either the whole signal or a
dedicated noise-only clip (`y_noise`), then gates anything below that
threshold.

### `noisereducecustom/nonstationary.py`
**Non-stationary mode** (`stationary=False`, the default): the noise
floor is estimated as a *rolling* average over time (via an
exponentially-weighted moving average), so the gate adapts as
background noise changes throughout the recording — useful for
outdoor recordings where wind, insects, or traffic come and go.

### `noisereducecustom/core.py`
The public `reduce_noise()` function. Reads all the parameters,
decides whether to build a `SpectralGateStationary` or
`SpectralGateNonStationary` instance, and returns its output. This is
the only function most users will ever call directly.

### `noisereducecustom/streaming.py`
Block-based streaming versions of both gating modes
(`StreamingNoiseReducerStationary`, `StreamingNoiseReducerNonStationary`),
for continuous mic input rather than complete files. Each call to
`process_block()` prepends real audio carried over from the previous
block (not silence) for STFT context, so there's no click at block
boundaries and no added latency waiting for "future" audio. The
non-stationary version also carries its adaptive noise-floor estimate
between blocks, so it has genuine memory across the whole stream.

### `noisereducecustom/generate_noise.py`
`band_limited_noise(min_freq, max_freq, samples, samplerate)` —
synthesizes noise whose energy is confined to a chosen frequency band,
using random-phase FFT synthesis. Useful for building demos/tests
without needing a real noisy recording (see `examples/demo.py`), or
for generating a synthetic `y_noise` profile clip.

### `noisereducecustom/plotting.py`
Optional (matplotlib-only) visualization helpers: `plot_spectrogram`,
`plot_before_after`, `plot_waveform_comparison`. Purely for demos and
sanity-checking — not used internally by the noise reduction itself.

### `tests/test_core.py`
Automated checks: stationary/non-stationary modes, mono, stereo,
chunked processing, using a separate noise clip, and a sanity check
that `prop_decrease=0` leaves the signal essentially untouched.

### `examples/demo.py`
End-to-end runnable demo — synthesizes a fake bird chirp + band-limited
noise, runs both gating modes, saves 3 `.wav` files you can listen to
and compare, and shows a before/after spectrogram if matplotlib is
installed. Good for a live demo since it needs no external audio file.

### `examples/live_mic_demo.py`
Demonstrates the streaming API. Runs in a hardware-free simulation
mode by default (synthesizes a fake continuous forest recording and
streams it through block by block); pass `--mic` with `pyaudio`
installed to use a real microphone.

### `pyproject.toml` / `setup.py`
Packaging metadata (PyPI-ready) and a small compatibility shim so
`pip install -e .` works even on older `setuptools` versions.

---

## Publishing to PyPI (optional)
```bash
pip install build twine
python -m build
twine upload dist/*
```

## Installing straight from your GitHub repo (no PyPI needed)
```bash
pip install git+https://github.com/mohan1917/noisereducecustom.git
```
