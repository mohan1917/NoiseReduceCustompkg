"""
stationary.py
-------------
Stationary spectral gating: assumes the noise's frequency profile is
constant for the whole clip (or is supplied separately via `y_noise`).
Best for steady background hums/hiss (fans, mic self-noise, room tone)
where the noise character doesn't change much over time.

Algorithm
---------
1. STFT the (noise) signal, convert magnitude to dB.
2. For each frequency bin, compute mean + std across time.
3. threshold[bin] = mean[bin] + n_std_thresh_stationary * std[bin]
4. Any (frequency, time) point in the *signal's* spectrogram whose dB
   is below that bin's threshold is treated as noise -> mask = 0,
   otherwise mask = 1.
5. Smooth the binary mask with a small 2D triangular filter (see
   utils.triangular_smoothing_kernel) to avoid abrupt gating clicks.
6. Blend the smoothed mask with prop_decrease: at prop_decrease=1.0 the
   mask is applied fully; at 0.0 nothing changes.
7. Apply mask to the magnitude spectrogram (phase is preserved
   unchanged) and inverse-STFT back to a waveform.
"""

import numpy as np
from scipy.signal import fftconvolve

from .base import SpectralGateBase
from .stft_utils import compute_stft, compute_istft
from .utils import amp_to_db, triangular_smoothing_kernel, hz_to_bins, ms_to_frames


class SpectralGateStationary(SpectralGateBase):
    def __init__(self, y, sr, y_noise=None, prop_decrease=1.0,
                 n_std_thresh_stationary=1.5, n_fft=1024, win_length=None,
                 hop_length=None, freq_mask_smooth_hz=500, time_mask_smooth_ms=50,
                 chunk_size=None, padding=30000, n_jobs=1):
        super().__init__(y, sr, n_fft=n_fft, win_length=win_length,
                          hop_length=hop_length, chunk_size=chunk_size,
                          padding=padding, n_jobs=n_jobs)

        self.prop_decrease = prop_decrease
        self.n_std_thresh_stationary = n_std_thresh_stationary

        n_grad_freq = hz_to_bins(freq_mask_smooth_hz, sr, self.n_fft) if freq_mask_smooth_hz else 1
        n_grad_time = ms_to_frames(time_mask_smooth_ms, sr, self.hop_length) if time_mask_smooth_ms else 1
        self.smoothing_kernel = triangular_smoothing_kernel(n_grad_freq, n_grad_time)

        # Precompute the noise threshold once, up front, from either a
        # dedicated noise clip or the signal itself.
        noise_source = y_noise if y_noise is not None else y
        noise2d, _ = self._reshape(noise_source)
        # Average the per-channel thresholds if multi-channel noise is given
        thresholds = []
        for ch in range(noise2d.shape[0]):
            Zxx = compute_stft(noise2d[ch], sr, n_fft=self.n_fft,
                                hop_length=self.hop_length, win_length=self.win_length)
            mag_db = amp_to_db(np.abs(Zxx))
            mean = np.mean(mag_db, axis=1, keepdims=True)
            std = np.std(mag_db, axis=1, keepdims=True)
            thresholds.append(mean + n_std_thresh_stationary * std)
        self.noise_threshold = np.mean(thresholds, axis=0)  # shape (n_freq_bins, 1)

    def _process_channel(self, channel):
        Zxx = compute_stft(channel, self.sr, n_fft=self.n_fft,
                            hop_length=self.hop_length, win_length=self.win_length)
        mag, phase = np.abs(Zxx), np.angle(Zxx)
        mag_db = amp_to_db(mag)

        mask = (mag_db > self.noise_threshold).astype(np.float32)
        mask = fftconvolve(mask, self.smoothing_kernel, mode="same")
        mask = np.clip(mask, 0.0, 1.0)
        mask = mask * self.prop_decrease + (1.0 - self.prop_decrease)

        mag_denoised = mag * mask
        Zxx_denoised = mag_denoised * np.exp(1j * phase)

        return compute_istft(Zxx_denoised, self.sr, n_fft=self.n_fft,
                              hop_length=self.hop_length, win_length=self.win_length,
                              length=len(channel))
