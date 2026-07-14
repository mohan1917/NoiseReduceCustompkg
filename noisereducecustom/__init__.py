"""
noisereducecustom
==================

A fully self-contained, from-scratch spectral-gating audio noise reduction
library. Built only on numpy/scipy (matplotlib optional, for plotting).

It does not import or depend on the third-party `noisereduce` package in
any way, so it keeps working even if that package is ever removed from
PyPI, paywalled, or changes its license.

Quick start
-----------
    import noisereducecustom as nrc

    reduced = nrc.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.95)
"""

from .core import reduce_noise
from .generate_noise import band_limited_noise
from .streaming import StreamingNoiseReducerStationary, StreamingNoiseReducerNonStationary
from . import plotting

__all__ = [
    "reduce_noise",
    "band_limited_noise",
    "plotting",
    "StreamingNoiseReducerStationary",
    "StreamingNoiseReducerNonStationary",
]
__version__ = "0.3.0"
