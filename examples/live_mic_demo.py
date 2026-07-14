"""
live_mic_demo.py
-----------------
Demonstrates the streaming API in two modes:

  1. REAL MIC MODE (requires `pyaudio`: pip install pyaudio)
     Captures live audio from your microphone, denoises it block by
     block in real time, and prints a running status.

  2. SIMULATION MODE (no hardware/pyaudio needed)
     Feeds a synthetic "forest recording" through the same streaming
     pipeline as if it were arriving live, block by block, and saves
     the result to a .wav file. Useful for testing/demoing this on a
     laptop with no mic, or in a CI/sandbox environment.

Run with:
    python examples/live_mic_demo.py            # auto-picks simulation
                                                   # mode if pyaudio isn't
                                                   # installed
    python examples/live_mic_demo.py --mic       # force real mic mode
"""

import sys
import time
import numpy as np

import noisereducecustom as nrc


SR = 16000
BLOCK_SECONDS = 0.5
BLOCK_SAMPLES = int(SR * BLOCK_SECONDS)


def run_simulation():
    """No hardware required: simulates a continuous mic feed using a
    synthetic noisy 'forest' recording, processed block by block."""
    from scipy.io import wavfile

    print("No microphone / pyaudio detected -- running in SIMULATION mode.\n")

    duration = 6.0
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)

    # Fake bird chirps + constant background hiss (simulating steady
    # forest ambient noise -- wind through leaves, insects, etc.)
    chirp = 0.2 * np.sin(2 * np.pi * (1800 + 800 * np.sin(2 * np.pi * 0.6 * t)) * t)
    envelope = (np.sin(2 * np.pi * 1.2 * t) > 0.3).astype(np.float32)
    background_noise = nrc.band_limited_noise(50, 6000, samples=len(t), samplerate=SR) * 0.08

    fake_stream = (chirp * envelope + background_noise).astype(np.float32)

    # A separate short noise-only sample to prime the stationary
    # reducer -- in a real deployment, capture a few seconds of quiet
    # ambient sound before any calls happen.
    noise_profile = nrc.band_limited_noise(50, 6000, samples=SR * 2, samplerate=SR) * 0.08

    reducer = nrc.StreamingNoiseReducerStationary(
        sr=SR, y_noise=noise_profile, prop_decrease=0.9
    )

    denoised_blocks = []
    print("Processing simulated live stream in blocks...")
    for i, start in enumerate(range(0, len(fake_stream), BLOCK_SAMPLES)):
        block = fake_stream[start:start + BLOCK_SAMPLES]
        if len(block) == 0:
            break
        clean_block = reducer.process_block(block)
        denoised_blocks.append(clean_block)

        # --- This is where you'd hand clean_block off to YAMNet ---
        # scores = yamnet_interpreter.run(clean_block)
        # if scores[BIRD_CLASS_INDEX] > THRESHOLD:
        #     send_notification("Bird detected")
        print(f"  block {i}: processed {len(block)} samples "
              f"(peak level {np.max(np.abs(clean_block)):.3f})")

    result = np.concatenate(denoised_blocks)
    out = result / (np.max(np.abs(result)) + 1e-9)
    wavfile.write("live_sim_denoised.wav", SR, (out * 32767).astype(np.int16))

    raw_out = fake_stream / (np.max(np.abs(fake_stream)) + 1e-9)
    wavfile.write("live_sim_noisy.wav", SR, (raw_out * 32767).astype(np.int16))

    print("\nSaved live_sim_noisy.wav and live_sim_denoised.wav for comparison.")


def run_real_mic():
    """Requires `pyaudio`. Captures real microphone input continuously."""
    import pyaudio

    print("Starting REAL microphone streaming mode. Press Ctrl+C to stop.\n")
    print("Capturing 2 seconds of ambient noise to build the noise profile...")

    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=SR,
                      input=True, frames_per_buffer=BLOCK_SAMPLES)

    # Capture a short ambient noise sample first (assumes it's quiet
    # for these first couple seconds -- adjust to your deployment).
    noise_chunks = [
        np.frombuffer(stream.read(BLOCK_SAMPLES, exception_on_overflow=False), dtype=np.float32)
        for _ in range(int(2.0 / BLOCK_SECONDS))
    ]
    noise_profile = np.concatenate(noise_chunks)

    reducer = nrc.StreamingNoiseReducerStationary(
        sr=SR, y_noise=noise_profile, prop_decrease=0.9
    )

    print("Noise profile captured. Listening...\n")
    try:
        while True:
            raw = stream.read(BLOCK_SAMPLES, exception_on_overflow=False)
            block = np.frombuffer(raw, dtype=np.float32)
            clean_block = reducer.process_block(block)

            # --- This is where you'd hand clean_block off to YAMNet ---
            # scores = yamnet_interpreter.run(clean_block)
            # if scores[BIRD_CLASS_INDEX] > THRESHOLD:
            #     send_notification("Bird detected")

            level = np.max(np.abs(clean_block))
            print(f"\rlevel: {level:.3f}", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    force_mic = "--mic" in sys.argv
    if force_mic:
        run_real_mic()
    else:
        try:
            import pyaudio  # noqa: F401
            print("pyaudio detected -- pass --mic to use your real microphone.")
            print("Running simulation mode by default.\n")
        except ImportError:
            pass
        run_simulation()
