"""
nonstationary.py
-----------------
Non-stationary spectral gating: instead of one fixed noise profile for
the whole clip, this continuously estimates a *local* noise floor per
frequency bin using an exponentially-weighted moving average (EWMA)
over time. This lets the gate adapt as background noise changes
throughout the recording -- e.g. wind gusts picking up, insects
starting/stopping, traffic passing by -- which a single global
threshold can't track well.

Algorithm
---------
1. STFT the signal, take magnitude (linear, not dB this time).
2. For each frequency bin, compute a smoothed "running average"
   magnitude over time using an EWMA whose time constant is set by
   `time_constant_s` (roughly: how many seconds of history influence
   the current noise-floor estimate).
3. For every (frequency, time) point, compute how many multiples above
   that local running average the actual magnitude is:
       ratio = (magnitude - running_average) / running_average
4. Turn that ratio into a smooth 0..1 mask with a sigmoid, centered at
   `thresh_n_mult_nonstationary` multiples above the floor, with
   steepness controlled by `sigmoid_slope_nonstationary`. Using a
   sigmoid instead of a hard cutoff avoids abrupt on/off gating.
5. Smooth the mask across time/frequency with a small triangular
   kernel (same idea as the stationary version).
6. Blend with prop_decrease, apply to magnitude, keep original phase,
   inverse-STFT back to a waveform.
"""

import numpy as np
from scipy.signal import fftconvolve

from .base import SpectralGateBase
from .stft_utils import compute_stft, compute_istft
from .utils import triangular_smoothing_kernel, hz_to_bins, ms_to_frames, sigmoid


def _ewma_time_smooth(magnitude, sr, hop_length, time_constant_s):
    """
    Exponentially-weighted moving average along the time axis of a
    (n_freq_bins, n_frames) magnitude array, applied forward then
    backward (so the estimate isn't lagged/shifted in time).
    """
    frames_per_constant = max(1.0, (time_constant_s * sr) / hop_length)
    alpha = 1.0 - np.exp(-1.0 / frames_per_constant)

    def _forward_pass(x):
        out = np.empty_like(x)
        out[:, 0] = x[:, 0]
        for i in range(1, x.shape[1]):
            out[:, i] = alpha * x[:, i] + (1 - alpha) * out[:, i - 1]
        return out

    smoothed_fwd = _forward_pass(magnitude)
    smoothed_bwd = _forward_pass(smoothed_fwd[:, ::-1])[:, ::-1]
    return smoothed_bwd


class SpectralGateNonStationary(SpectralGateBase):
    def __init__(self, y, sr, prop_decrease=1.0, time_constant_s=2.0,
                 thresh_n_mult_nonstationary=2.0, sigmoid_slope_nonstationary=10.0,
                 n_fft=1024, win_length=None, hop_length=None,
                 freq_mask_smooth_hz=500, time_mask_smooth_ms=50,
                 chunk_size=None, padding=30000, n_jobs=1):
        super().__init__(y, sr, n_fft=n_fft, win_length=win_length,
                          hop_length=hop_length, chunk_size=chunk_size,
                          padding=padding, n_jobs=n_jobs)

        self.prop_decrease = prop_decrease
        self.time_constant_s = time_constant_s
        self.thresh_n_mult_nonstationary = thresh_n_mult_nonstationary
        self.sigmoid_slope_nonstationary = sigmoid_slope_nonstationary

        n_grad_freq = hz_to_bins(freq_mask_smooth_hz, sr, self.n_fft) if freq_mask_smooth_hz else 1
        n_grad_time = ms_to_frames(time_mask_smooth_ms, sr, self.hop_length) if time_mask_smooth_ms else 1
        self.smoothing_kernel = triangular_smoothing_kernel(n_grad_freq, n_grad_time)

    def _process_channel(self, channel):
        Zxx = compute_stft(channel, self.sr, n_fft=self.n_fft,
                            hop_length=self.hop_length, win_length=self.win_length)
        mag, phase = np.abs(Zxx), np.angle(Zxx)

        running_floor = _ewma_time_smooth(mag, self.sr, self.hop_length, self.time_constant_s)
        running_floor = np.maximum(running_floor, 1e-10)  # avoid div-by-zero

        ratio_above_floor = (mag - running_floor) / running_floor
        mask = sigmoid(ratio_above_floor, shift=-self.thresh_n_mult_nonstationary,
                        slope=self.sigmoid_slope_nonstationary)

        mask = fftconvolve(mask, self.smoothing_kernel, mode="same")
        mask = np.clip(mask, 0.0, 1.0)
        mask = mask * self.prop_decrease + (1.0 - self.prop_decrease)

        mag_denoised = mag * mask
        Zxx_denoised = mag_denoised * np.exp(1j * phase)

        return compute_istft(Zxx_denoised, self.sr, n_fft=self.n_fft,
                              hop_length=self.hop_length, win_length=self.win_length,
                              length=len(channel))
