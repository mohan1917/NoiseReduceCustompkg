"""
generate_noise.py
------------------
Utility for generating synthetic band-limited noise. Handy for demos
and tests where you want a reproducible "noise-only" clip to feed as
`y_noise`, or want to add controllable synthetic noise on top of a
clean signal to test how well `reduce_noise` recovers it.

Technique: build a frequency-domain mask that's 1 inside
[min_freq, max_freq] and 0 elsewhere, assign each active bin a random
phase, then inverse-FFT to get a real-valued time-domain noise signal
band-limited to that range. This is a standard, well-known way to
synthesize colored/band-limited noise.
"""

import numpy as np


def _random_phase_spectrum(freq_mask):
    """Given a real-valued 0/1 frequency mask, build a full complex
    spectrum with random phase at every active bin, respecting
    conjugate symmetry so the inverse FFT is real-valued."""
    spectrum = np.array(freq_mask, dtype="complex")
    n_active = (len(spectrum) - 1) // 2

    random_phases = np.random.rand(n_active) * 2 * np.pi
    phase_factors = np.cos(random_phases) + 1j * np.sin(random_phases)

    spectrum[1: n_active + 1] *= phase_factors
    spectrum[-1: -1 - n_active: -1] = np.conj(spectrum[1: n_active + 1])
    return spectrum


def band_limited_noise(min_freq, max_freq, samples=44100, samplerate=44100):
    """
    Generate `samples` samples of noise whose energy is limited to the
    [min_freq, max_freq] Hz band.

    Parameters
    ----------
    min_freq, max_freq : float
        Frequency band (Hz) the noise energy should be limited to.
    samples : int
        Number of time-domain samples to generate.
    samplerate : int
        Sample rate (Hz) used to map FFT bins to frequencies.

    Returns
    -------
    np.ndarray, shape (samples,), float64
        Real-valued noise signal, roughly in [-1, 1].
    """
    freqs = np.abs(np.fft.fftfreq(samples, 1.0 / samplerate))
    band_mask = np.zeros(samples)
    band_mask[(freqs >= min_freq) & (freqs <= max_freq)] = 1.0

    spectrum = _random_phase_spectrum(band_mask)
    noise = np.fft.ifft(spectrum).real

    # normalize to roughly [-1, 1]
    peak = np.max(np.abs(noise)) + 1e-12
    return noise / peak
