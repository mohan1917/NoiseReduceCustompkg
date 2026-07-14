"""
stft_utils.py
-------------
Thin wrappers around scipy.signal.stft / istft so the rest of the
package doesn't need to worry about window/overlap bookkeeping.
"""

import numpy as np
from scipy.signal import stft, istft


def compute_stft(y, sr, n_fft=1024, hop_length=None, win_length=None):
    """
    Compute the Short-Time Fourier Transform of a 1D signal.

    Returns
    -------
    Zxx : complex ndarray, shape (n_freq_bins, n_frames)
    """
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 4

    noverlap = win_length - hop_length
    _, _, Zxx = stft(
        y,
        fs=sr,
        nperseg=win_length,
        noverlap=noverlap,
        nfft=n_fft,
        padded=True,
        boundary="zeros",
    )
    return Zxx


def compute_istft(Zxx, sr, n_fft=1024, hop_length=None, win_length=None, length=None):
    """
    Reconstruct a 1D signal from a complex STFT matrix, trimmed/padded
    to `length` samples if given.
    """
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 4

    noverlap = win_length - hop_length
    _, y = istft(Zxx, fs=sr, nperseg=win_length, noverlap=noverlap, nfft=n_fft)

    if length is not None:
        if len(y) < length:
            y = np.pad(y, (0, length - len(y)))
        else:
            y = y[:length]
    return y
