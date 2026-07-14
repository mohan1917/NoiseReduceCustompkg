"""
utils.py
--------
Small, reusable building blocks shared by both the stationary and
non-stationary spectral gating implementations:

- shaping raw audio into a consistent (n_channels, n_frames) layout
  so mono AND stereo/multi-channel input both work the same way
- linear-amplitude <-> decibel conversion helpers
- a 2D triangular smoothing kernel, used to blur the noise mask across
  time and frequency so gating doesn't produce abrupt "clicky" artifacts
  (this is sometimes called "musical noise" in the audio DSP literature)
- a numerically-stable sigmoid, used by the non-stationary gate to turn
  a continuous "how many multiples above the noise floor" score into a
  smooth 0..1 mask instead of a hard on/off cut
"""

import numpy as np


def ensure_channel_first(y):
    """
    Normalize an input waveform into shape (n_channels, n_frames).

    Accepts:
        - 1D mono array of shape (n_frames,)
        - 2D array of shape (n_frames, n_channels) OR (n_channels, n_frames)

    Returns
    -------
    y2d : ndarray, shape (n_channels, n_frames)
    layout : str
        One of "flat" (input was 1D mono), "channels_first" (input was
        already (n_channels, n_frames)), or "channels_last" (input was
        (n_frames, n_channels)) -- so callers can restore the original
        layout on output.
    """
    y = np.asarray(y)

    if y.ndim == 1:
        return y[np.newaxis, :], "flat"

    if y.ndim == 2:
        # Heuristic: assume the longer axis is time (n_frames), the
        # shorter axis is channels. Real audio always has far more
        # samples than channels.
        if y.shape[0] < y.shape[1]:
            # already (n_channels, n_frames)
            return y, "channels_first"
        else:
            # (n_frames, n_channels) -> (n_channels, n_frames)
            return y.T, "channels_last"

    raise ValueError(
        f"Expected a 1D (mono) or 2D (multi-channel) array, got shape {y.shape}"
    )


def amp_to_db(amplitude, eps=1e-10):
    """Convert linear magnitude to decibels."""
    return 20.0 * np.log10(amplitude + eps)


def db_to_amp(db):
    """Convert decibels back to linear magnitude."""
    return 10.0 ** (db / 20.0)


def sigmoid(x, shift=0.0, slope=1.0):
    """
    Numerically stable sigmoid, shifted and scaled.

    mask = sigmoid(x, shift, slope) smoothly transitions from 0 to 1 as
    x crosses `-shift`; `slope` controls how sharp that transition is.
    """
    z = slope * (x + shift)
    # clip to avoid overflow in exp() for very large |z|
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def triangular_smoothing_kernel(n_freq_bins, n_time_bins):
    """
    Build a 2D smoothing kernel shaped like a pyramid: it peaks at the
    center and linearly tapers to zero at the edges, along both the
    frequency and time axes independently, then the two 1D triangles
    are combined with an outer product into one 2D kernel.

    This is used to blur the (frequency, time) noise mask so that
    the gate turns on/off gradually rather than abruptly -- abrupt
    gating is what causes the "underwater / robotic" artifacts often
    called musical noise.

    Parameters
    ----------
    n_freq_bins : int
        Half-width of the kernel along the frequency axis (in bins).
    n_time_bins : int
        Half-width of the kernel along the time axis (in frames).

    Returns
    -------
    kernel : ndarray, shape (2*n_freq_bins + 1, 2*n_time_bins + 1)
        Normalized so it sums to 1 (i.e. it's an averaging filter).
    """
    def _triangle(half_width):
        if half_width <= 0:
            return np.array([1.0])
        ramp_up = np.linspace(0, 1, half_width + 1, endpoint=False)
        ramp_down = np.linspace(1, 0, half_width + 1, endpoint=False)
        return np.concatenate([ramp_up, ramp_down])[1:]

    freq_triangle = _triangle(n_freq_bins)
    time_triangle = _triangle(n_time_bins)
    kernel = np.outer(freq_triangle, time_triangle)
    kernel = kernel / np.sum(kernel)
    return kernel


def hz_to_bins(freq_hz, sr, n_fft):
    """Convert a frequency span in Hz to a number of FFT bins."""
    bin_width_hz = sr / n_fft
    return max(1, int(round(freq_hz / bin_width_hz)))


def ms_to_frames(time_ms, sr, hop_length):
    """Convert a time span in milliseconds to a number of STFT frames."""
    frame_duration_ms = (hop_length / sr) * 1000.0
    return max(1, int(round(time_ms / frame_duration_ms)))
