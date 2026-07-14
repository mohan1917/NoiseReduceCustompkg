"""
plotting.py
-----------
Optional visualization helpers, useful for building a demo/notebook
that shows what noise reduction is actually doing under the hood.
Requires matplotlib (not a hard dependency of the core package --
only imported when these functions are called).
"""

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting functions. "
            "Install it with: pip install matplotlib"
        ) from e
    return plt


def plot_spectrogram(y, sr, n_fft=1024, hop_length=None, title="Spectrogram"):
    """Plot the dB-scaled spectrogram of a 1D audio signal."""
    plt = _require_matplotlib()
    from .stft_utils import compute_stft
    from .utils import amp_to_db

    Zxx = compute_stft(y, sr, n_fft=n_fft, hop_length=hop_length)
    mag_db = amp_to_db(np.abs(Zxx))

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(mag_db, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frequency bin")
    fig.colorbar(im, ax=ax, label="dB")
    plt.tight_layout()
    plt.show()


def plot_before_after(y_original, y_denoised, sr, n_fft=1024, hop_length=None):
    """Side-by-side spectrograms of the signal before and after denoising."""
    plt = _require_matplotlib()
    from .stft_utils import compute_stft
    from .utils import amp_to_db

    Zxx_before = compute_stft(y_original, sr, n_fft=n_fft, hop_length=hop_length)
    Zxx_after = compute_stft(y_denoised, sr, n_fft=n_fft, hop_length=hop_length)
    db_before = amp_to_db(np.abs(Zxx_before))
    db_after = amp_to_db(np.abs(Zxx_after))

    fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharey=True)
    for ax, data, title in zip(axes, [db_before, db_after], ["Before", "After"]):
        im = ax.imshow(data, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("Frame")
    axes[0].set_ylabel("Frequency bin")
    fig.colorbar(im, ax=axes, label="dB")
    plt.show()


def plot_waveform_comparison(y_original, y_denoised, sr):
    """Overlay the original and denoised waveforms in the time domain."""
    plt = _require_matplotlib()

    t = np.arange(len(y_original)) / sr
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, y_original, alpha=0.5, label="Original")
    ax.plot(t[: len(y_denoised)], y_denoised, alpha=0.8, label="Denoised")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.set_title("Waveform: original vs. denoised")
    plt.tight_layout()
    plt.show()
