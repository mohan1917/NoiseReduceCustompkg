"""
demo.py
-------
A self-contained demonstration of noisereducecustom that doesn't
require any input .wav file -- it synthesizes a "bird call" (a chirp)
plus band-limited background noise, denoises it, and saves both the
noisy and denoised versions to disk so you can compare them by ear.

Run with:
    python examples/demo.py
"""

import numpy as np
from scipy.io import wavfile

import noisereducecustom as nrc


def make_fake_bird_call(sr, duration=2.0):
    """A rising-then-falling chirp, loosely mimicking a bird call."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq_sweep = 2000 + 1500 * np.sin(2 * np.pi * 0.5 * t)  # sweeps 500-3500 Hz
    phase = 2 * np.pi * np.cumsum(freq_sweep) / sr
    chirp = 0.6 * np.sin(phase)
    # amplitude envelope so it sounds like discrete calls, not a constant tone
    envelope = (np.sin(2 * np.pi * 1.5 * t) > 0).astype(np.float32)
    return (chirp * envelope).astype(np.float32)


def main():
    sr = 22050
    duration = 3.0

    bird = make_fake_bird_call(sr, duration)
    background_noise = nrc.band_limited_noise(
        min_freq=50, max_freq=6000, samples=len(bird), samplerate=sr
    ) * 0.15

    noisy = (bird + background_noise).astype(np.float32)

    print("Running stationary noise reduction...")
    denoised_stationary = nrc.reduce_noise(
        y=noisy, sr=sr, stationary=True, prop_decrease=0.9
    )

    print("Running non-stationary noise reduction...")
    denoised_nonstationary = nrc.reduce_noise(
        y=noisy, sr=sr, stationary=False, prop_decrease=0.9
    )

    def save(name, data):
        data = data / (np.max(np.abs(data)) + 1e-9)
        wavfile.write(name, sr, (data * 32767).astype(np.int16))
        print(f"  saved {name}")

    save("demo_noisy.wav", noisy)
    save("demo_denoised_stationary.wav", denoised_stationary)
    save("demo_denoised_nonstationary.wav", denoised_nonstationary)

    # Optional: visualize (requires matplotlib)
    try:
        nrc.plotting.plot_before_after(noisy, denoised_stationary, sr)
    except ImportError:
        print("(install matplotlib to see spectrogram plots: pip install matplotlib)")


if __name__ == "__main__":
    main()
