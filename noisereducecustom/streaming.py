"""
streaming.py
------------
Block-based streaming noise reduction, for continuous mic input (e.g.
audio captured live via pyaudio/sounddevice in a forest deployment).

WHY THIS IS A SEPARATE MODULE FROM core.reduce_noise()
-------------------------------------------------------
`reduce_noise()` expects the ENTIRE signal up front: it computes one
STFT over the whole array, processes it, and does one inverse STFT.
That's fine for a complete .wav file, but it can't be called block by
block on live audio without two problems:

  1. Boundary artifacts -- each call would window/FFT its block in
     total isolation, so consecutive blocks can click/pop at the seam.
  2. No live audio has "future" samples -- reduce_noise's chunk padding
     pads on BOTH sides, including the future, which live audio doesn't
     have yet.

HOW STREAMING SOLVES IT
------------------------
Each call to `process_block()` prepends a short buffer of REAL audio
carried over from the end of the previous block (not silence, not
guessed data) before running STFT -> mask -> ISTFT. This gives the
STFT proper context on the left edge, without needing to wait for any
future audio -- so there's no added latency. Only the newly-arrived
portion of the output is returned; the rest is discarded (it was only
there for context) and the tail of the raw block is kept as history
for the next call.

This is a CAUSAL, low-latency design: output for a block is available
as soon as that block itself has been captured, with latency equal to
roughly one STFT window, not "wait for more audio to arrive."

Two classes are provided, matching the two gating strategies:
    StreamingNoiseReducerStationary     -- fixed noise profile
    StreamingNoiseReducerNonStationary  -- adaptive noise floor, with
                                            its state carried between
                                            blocks so the "memory" of
                                            the noise floor doesn't
                                            reset every block
"""

import numpy as np
from scipy.signal import fftconvolve

from .stft_utils import compute_stft, compute_istft
from .utils import amp_to_db, triangular_smoothing_kernel, hz_to_bins, ms_to_frames, sigmoid


class StreamingNoiseReducerStationary:
    """
    Streaming version of stationary spectral gating. The noise profile
    is computed ONCE (from a noise-only sample you provide at startup,
    e.g. a few seconds of ambient forest sound before any animal calls)
    and reused for every block -- ideal for a roughly constant
    background noise (wind, insects, distant traffic hum).

    Usage
    -----
        noise_sample = capture_ambient_noise(seconds=3)  # your mic code

        reducer = StreamingNoiseReducerStationary(
            sr=16000, y_noise=noise_sample, prop_decrease=0.9
        )

        while True:
            block = mic.read(block_samples)          # your mic code
            clean_block = reducer.process_block(block)
            classify_with_yamnet(clean_block)          # your model code
    """

    def __init__(self, sr, y_noise, prop_decrease=1.0, n_std_thresh_stationary=1.5,
                 n_fft=1024, win_length=None, hop_length=None,
                 freq_mask_smooth_hz=500, time_mask_smooth_ms=50,
                 history_seconds=None):
        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4
        self.prop_decrease = prop_decrease

        n_grad_freq = hz_to_bins(freq_mask_smooth_hz, sr, self.n_fft) if freq_mask_smooth_hz else 1
        n_grad_time = ms_to_frames(time_mask_smooth_ms, sr, self.hop_length) if time_mask_smooth_ms else 1
        self.smoothing_kernel = triangular_smoothing_kernel(n_grad_freq, n_grad_time)

        # Noise profile computed once, up front, from the supplied
        # noise-only sample.
        y_noise = np.asarray(y_noise, dtype=np.float32)
        Zxx = compute_stft(y_noise, sr, n_fft=self.n_fft, hop_length=self.hop_length,
                            win_length=self.win_length)
        mag_db = amp_to_db(np.abs(Zxx))
        mean = np.mean(mag_db, axis=1, keepdims=True)
        std = np.std(mag_db, axis=1, keepdims=True)
        self.noise_threshold = mean + n_std_thresh_stationary * std

        # How much REAL past audio to carry over as left-context for
        # each new block. One window's worth is enough for the STFT to
        # have proper context at the start of the new block.
        self.history_samples = (
            int(history_seconds * sr) if history_seconds else self.win_length
        )
        self._history = np.zeros(self.history_samples, dtype=np.float32)

    def _gate(self, segment):
        Zxx = compute_stft(segment, self.sr, n_fft=self.n_fft,
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
                              length=len(segment))

    def process_block(self, block):
        """
        Denoise one newly-arrived block of live audio.

        Parameters
        ----------
        block : np.ndarray, 1D float32
            The newest chunk of mic audio (mono).

        Returns
        -------
        np.ndarray, 1D float32, same length as `block`
            The denoised version of just this block. Zero added
            latency beyond the block itself having been captured.
        """
        block = np.asarray(block, dtype=np.float32)
        extended = np.concatenate([self._history, block])

        denoised_extended = self._gate(extended)

        # Update history for next call BEFORE returning, using the
        # RAW (not denoised) tail -- so context always reflects real
        # audio, not our own previous output.
        if len(block) >= self.history_samples:
            self._history = block[-self.history_samples:].copy()
        else:
            self._history = np.concatenate(
                [self._history[len(block):], block]
            )

        return denoised_extended[self.history_samples: self.history_samples + len(block)]

    def reset(self):
        """Clear the history buffer (e.g. after a long silence/gap)."""
        self._history = np.zeros(self.history_samples, dtype=np.float32)


class StreamingNoiseReducerNonStationary:
    """
    Streaming version of non-stationary (adaptive) spectral gating.
    The rolling noise-floor estimate's state is carried between blocks
    (instead of resetting every call), so the adaptive noise floor
    keeps a genuine memory of recent history across block boundaries.

    Usage
    -----
        reducer = StreamingNoiseReducerNonStationary(sr=16000, prop_decrease=0.9)

        while True:
            block = mic.read(block_samples)
            clean_block = reducer.process_block(block)
            classify_with_yamnet(clean_block)
    """

    def __init__(self, sr, prop_decrease=1.0, time_constant_s=2.0,
                 thresh_n_mult_nonstationary=2.0, sigmoid_slope_nonstationary=10.0,
                 n_fft=1024, win_length=None, hop_length=None,
                 freq_mask_smooth_hz=500, time_mask_smooth_ms=50,
                 history_seconds=None):
        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4
        self.prop_decrease = prop_decrease
        self.time_constant_s = time_constant_s
        self.thresh_n_mult_nonstationary = thresh_n_mult_nonstationary
        self.sigmoid_slope_nonstationary = sigmoid_slope_nonstationary

        n_grad_freq = hz_to_bins(freq_mask_smooth_hz, sr, self.n_fft) if freq_mask_smooth_hz else 1
        n_grad_time = ms_to_frames(time_mask_smooth_ms, sr, self.hop_length) if time_mask_smooth_ms else 1
        self.smoothing_kernel = triangular_smoothing_kernel(n_grad_freq, n_grad_time)

        self.history_samples = (
            int(history_seconds * sr) if history_seconds else self.win_length
        )
        self._history = np.zeros(self.history_samples, dtype=np.float32)

        # Carried-over EWMA noise-floor state (per frequency bin). None
        # until the first block establishes it.
        self._running_floor_state = None

    def _ewma_with_carryover(self, magnitude):
        frames_per_constant = max(1.0, (self.time_constant_s * self.sr) / self.hop_length)
        alpha = 1.0 - np.exp(-1.0 / frames_per_constant)

        n_bins, n_frames = magnitude.shape
        out = np.empty_like(magnitude)

        prev = (
            self._running_floor_state
            if self._running_floor_state is not None
            else magnitude[:, 0]
        )
        for i in range(n_frames):
            prev = alpha * magnitude[:, i] + (1 - alpha) * prev
            out[:, i] = prev

        self._running_floor_state = prev  # carry forward to next block
        return out

    def _gate(self, segment):
        Zxx = compute_stft(segment, self.sr, n_fft=self.n_fft,
                            hop_length=self.hop_length, win_length=self.win_length)
        mag, phase = np.abs(Zxx), np.angle(Zxx)

        running_floor = self._ewma_with_carryover(mag)
        running_floor = np.maximum(running_floor, 1e-10)

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
                              length=len(segment))

    def process_block(self, block):
        """Denoise one newly-arrived block. See StreamingNoiseReducerStationary
        for the same causal, zero-added-latency approach."""
        block = np.asarray(block, dtype=np.float32)
        extended = np.concatenate([self._history, block])

        denoised_extended = self._gate(extended)

        if len(block) >= self.history_samples:
            self._history = block[-self.history_samples:].copy()
        else:
            self._history = np.concatenate([self._history[len(block):], block])

        return denoised_extended[self.history_samples: self.history_samples + len(block)]

    def reset(self):
        """Clear history AND the adaptive noise-floor memory (e.g. after
        a long silence, or when moving the mic to a new location)."""
        self._history = np.zeros(self.history_samples, dtype=np.float32)
        self._running_floor_state = None
