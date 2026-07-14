"""
core.py
-------
Public entry point. `reduce_noise()` picks the stationary or
non-stationary strategy and hands off to it. Parameter names intentionally
mirror the conventions used by other spectral-gating tools so switching
your import line is close to a drop-in swap.
"""

from .stationary import SpectralGateStationary
from .nonstationary import SpectralGateNonStationary


def reduce_noise(
    y,
    sr,
    stationary=False,
    y_noise=None,
    prop_decrease=1.0,
    n_fft=1024,
    win_length=None,
    hop_length=None,
    freq_mask_smooth_hz=500,
    time_mask_smooth_ms=50,
    n_std_thresh_stationary=1.5,
    time_constant_s=2.0,
    thresh_n_mult_nonstationary=2.0,
    sigmoid_slope_nonstationary=10.0,
    chunk_size=None,
    padding=30000,
    n_jobs=1,
):
    """
    Reduce noise in an audio signal via spectral gating.

    Parameters
    ----------
    y : np.ndarray
        Audio signal. 1D (mono) of shape (n_frames,), or 2D
        multi-channel of shape (n_frames, n_channels) / (n_channels, n_frames).
    sr : int
        Sample rate of `y`.
    stationary : bool, default False
        If True, use a single noise profile for the whole clip
        (good for constant background hum/hiss). If False, adapt the
        noise floor over time (good for changing background noise).
    y_noise : np.ndarray, optional
        A separate noise-only clip to estimate the noise profile from.
        Only used when stationary=True. If omitted, the profile is
        estimated from `y` itself.
    prop_decrease : float, default 1.0
        Proportion of detected noise to remove, 0.0 (none) to 1.0 (full).
    n_fft, win_length, hop_length : int
        STFT parameters. win_length defaults to n_fft; hop_length
        defaults to win_length // 4.
    freq_mask_smooth_hz : float
        Frequency range (Hz) to smooth the mask over, to avoid
        artifacts from an overly sharp mask.
    time_mask_smooth_ms : float
        Time range (ms) to smooth the mask over, same purpose.
    n_std_thresh_stationary : float
        (stationary mode only) number of standard deviations above the
        mean noise level to set as the gating threshold.
    time_constant_s : float
        (non-stationary mode only) how many seconds of history the
        rolling noise-floor estimate is averaged over.
    thresh_n_mult_nonstationary : float
        (non-stationary mode only) how many multiples above the local
        noise floor a bin needs to be to count as signal.
    sigmoid_slope_nonstationary : float
        (non-stationary mode only) steepness of the soft sigmoid gate.
    chunk_size : int, optional
        If set, process the signal in chunks of this many samples
        (useful to bound memory use on very long recordings).
    padding : int
        Extra samples of context added around each chunk edge before
        filtering, then trimmed off afterward.
    n_jobs : int, default 1
        Number of worker processes to use when chunk_size is set.
        1 = sequential (default, no extra overhead). >1 = parallel,
        using Python's built-in concurrent.futures (no third-party
        dependency). Only relevant for batch processing of long files;
        has no effect on files smaller than chunk_size.

    Returns
    -------
    np.ndarray
        Denoised audio, same shape as the input `y` (float32).
    """
    common_kwargs = dict(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        chunk_size=chunk_size,
        padding=padding,
        n_jobs=n_jobs,
    )

    if stationary:
        gate = SpectralGateStationary(
            y=y,
            sr=sr,
            y_noise=y_noise,
            prop_decrease=prop_decrease,
            n_std_thresh_stationary=n_std_thresh_stationary,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
            time_mask_smooth_ms=time_mask_smooth_ms,
            **common_kwargs,
        )
    else:
        gate = SpectralGateNonStationary(
            y=y,
            sr=sr,
            prop_decrease=prop_decrease,
            time_constant_s=time_constant_s,
            thresh_n_mult_nonstationary=thresh_n_mult_nonstationary,
            sigmoid_slope_nonstationary=sigmoid_slope_nonstationary,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
            time_mask_smooth_ms=time_mask_smooth_ms,
            **common_kwargs,
        )

    return gate.get_traces()
