"""
base.py
-------
Shared plumbing for both spectral-gating strategies (stationary and
non-stationary). Subclasses only need to implement `_process_channel`,
which takes one channel of raw float audio and returns the denoised
version of that same channel.

Two things are handled here so subclasses don't have to:

1. Multi-channel audio: audio is normalized to (n_channels, n_frames)
   and each channel is denoised independently, then re-stacked, then
   restored to whatever layout the input originally had.

2. Chunked processing: very long recordings are processed in
   overlapping chunks (with `padding` extra samples on each side, which
   get trimmed off after filtering) so memory usage stays bounded and
   the noise/mask statistics stay locally relevant instead of being
   averaged over an entire multi-hour file.

3. Optional parallel chunk processing: when both `chunk_size` and
   `n_jobs != 1` are set, chunks are farmed out across multiple CPU
   processes using Python's built-in `concurrent.futures` module --
   no third-party dependency (like `joblib`) required. This only
   matters for batch/offline processing of long files; it's irrelevant
   for live streaming, where chunks necessarily arrive one at a time.
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor


def _run_filter_span(args):
    """
    Module-level helper (required so it can be pickled and sent to a
    worker process). Unpacks (gate, channel, start, end) and calls the
    gate's own _filter_span on it.
    """
    gate, channel, start, end = args
    return start, end, gate._filter_span(channel, start, end)


class SpectralGateBase:
    def __init__(self, y, sr, n_fft=1024, win_length=None, hop_length=None,
                 chunk_size=None, padding=30000, n_jobs=1):
        y2d, layout = self._reshape(y)
        self.y = y2d
        self.layout = layout
        self.dtype = np.asarray(y).dtype
        self.n_channels, self.n_frames = self.y.shape

        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4
        self.chunk_size = chunk_size
        self.padding = padding
        self.n_jobs = n_jobs

    @staticmethod
    def _reshape(y):
        from .utils import ensure_channel_first
        return ensure_channel_first(y)

    def _process_channel(self, channel):
        """Subclasses implement the actual gating algorithm here."""
        raise NotImplementedError

    def _filter_span(self, channel, start, end):
        """
        Filter one span [start, end) of a channel, padding on both
        sides by `self.padding` samples for context, then trimming the
        padding back off after filtering.
        """
        i1 = start - self.padding
        i2 = end + self.padding

        pad_left = max(0, -i1)
        pad_right = max(0, i2 - len(channel))
        i1c = max(0, i1)
        i2c = min(len(channel), i2)

        segment = channel[i1c:i2c]
        if pad_left or pad_right:
            segment = np.pad(segment, (pad_left, pad_right))

        filtered = self._process_channel(segment)

        trim_start = self.padding
        trim_end = trim_start + (end - start)
        return filtered[trim_start:trim_end]

    def _filter_channel(self, channel):
        """Filter a full channel, chunked if chunk_size is set, and
        optionally parallelized across processes if n_jobs != 1."""
        n = len(channel)
        if not self.chunk_size or n <= self.chunk_size:
            return self._filter_span(channel, 0, n)

        spans = [
            (start, min(start + self.chunk_size, n))
            for start in range(0, n, self.chunk_size)
        ]

        out = np.zeros(n, dtype=np.float32)

        if self.n_jobs == 1 or len(spans) == 1:
            # Sequential path (default): simplest, zero extra dependency,
            # no process-pool startup overhead -- best for short/medium
            # files or single-core environments.
            for start, end in spans:
                out[start:end] = self._filter_span(channel, start, end)
        else:
            # Parallel path: farm chunks out across multiple CPU
            # processes using only the Python standard library.
            # Worthwhile for long batch/offline files on multi-core
            # hardware (e.g. the ASRock AI-350's 8-core/16-thread CPU).
            work_items = [(self, channel, start, end) for start, end in spans]
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                for start, end, result in executor.map(_run_filter_span, work_items):
                    out[start:end] = result

        return out

    def get_traces(self):
        """Run the gate over every channel and return the denoised audio,
        re-shaped back to whatever layout the input originally had."""
        out = np.zeros_like(self.y, dtype=np.float32)
        for ch in range(self.n_channels):
            out[ch] = self._filter_channel(self.y[ch])

        out = out.astype(np.float32)
        if self.layout == "flat":
            return out[0]
        elif self.layout == "channels_last":
            return out.T  # back to (n_frames, n_channels)
        return out  # "channels_first" stays (n_channels, n_frames)
