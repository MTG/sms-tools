"""
Sinusoidal plus Stochastic Model (S+S) analysis and synthesis functions.
Implements analysis, synthesis, and full model for S+S.
"""

import math
from typing import Tuple

import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import resample
from scipy.signal.windows import blackmanharris, hann, triang

from smstools.models import dftModel as DFT
from smstools.models import sineModel as SM
from smstools.models import stochasticModel as STM
from smstools.models import utilFunctions as UF


def spsModelAnal(
    x: np.ndarray,
    fs: float,
    w: np.ndarray,
    N: int,
    H: int,
    t: float,
    minSineDur: float,
    maxnSines: int,
    freqDevOffset: float,
    freqDevSlope: float,
    stocf: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze a sound using the sinusoidal plus stochastic model.

    Args:
        x: Input sound array.
        fs: Sampling rate.
        w: Analysis window array.
        N: FFT size.
        H: Hop size.
        t: Peak threshold in negative dB.
        minSineDur: Minimum sinusoidal track duration.
        maxnSines: Maximum number of parallel sinusoids.
        freqDevOffset: Allowed frame-to-frame frequency deviation at 0 Hz.
        freqDevSlope: Frequency-deviation slope for higher frequencies.
        stocf: Decimation factor for stochastic envelope.

    Returns:
        tfreq: Sinusoidal track frequencies.
        tmag: Sinusoidal track magnitudes.
        tphase: Sinusoidal track phases.
        stocEnv: Stochastic envelope.
    """
    tfreq, tmag, tphase = SM.sineModelAnal(
        x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope
    )
    Ns = 512
    xr = UF.sineSubtraction(x, Ns, H, tfreq, tmag, tphase, fs)
    stocEnv = STM.stochasticModelAnal(xr, H, H * 2, stocf)
    return tfreq, tmag, tphase, stocEnv


def spsModelSynth(
    tfreq: np.ndarray,
    tmag: np.ndarray,
    tphase: np.ndarray,
    stocEnv: np.ndarray,
    N: int,
    H: int,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesize a sound using the sinusoidal plus stochastic model.

    Args:
        tfreq: Sinusoidal track frequencies.
        tmag: Sinusoidal track magnitudes.
        tphase: Sinusoidal track phases.
        stocEnv: Stochastic envelope.
        N: Synthesis FFT size.
        H: Hop size.
        fs: Sampling rate.

    Returns:
        y: Output sound.
        ys: Sinusoidal component.
        yst: Stochastic component.
    """
    ys = SM.sineModelSynth(tfreq, tmag, tphase, N, H, fs)
    yst = STM.stochasticModelSynth(stocEnv, H, H * 2)
    y = ys[: min(ys.size, yst.size)] + yst[: min(ys.size, yst.size)]
    return y, ys, yst


def spsModel(
    x: np.ndarray,
    fs: float,
    w: np.ndarray,
    N: int,
    H: int = 128,
    t: float = -80,
    minSineDur: float = 0.02,
    maxnSines: int = 100,
    freqDevOffset: float = 20.0,
    freqDevSlope: float = 0.01,
    Ns: int = 512,
    stocf: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full analysis/synthesis of a sound using the sinusoidal plus stochastic model.

    Args:
        x: Input sound array.
        fs: Sampling rate.
        w: Analysis window array.
        N: FFT size.
        t: Peak threshold in negative dB.
        stocf: Decimation factor for stochastic envelope.

    Returns:
        y: Output sound.
        ys: Sinusoidal component.
        yst: Stochastic component.
    """
    # Use analysis then synthesis, ensure output lengths match input
    tfreq, tmag, tphase, stocEnv = spsModelAnal(
        x, fs, w, N, H, t, minSineDur, maxnSines, freqDevOffset, freqDevSlope, stocf
    )
    y, ys, yst = spsModelSynth(tfreq, tmag, tphase, stocEnv, Ns, H, fs)
    # Ensure output lengths match input
    def match_length(arr, target_len):
        if len(arr) > target_len:
            return arr[:target_len]
        elif len(arr) < target_len:
            return np.pad(arr, (0, target_len - len(arr)))
        else:
            return arr
    y = match_length(y, len(x))
    ys = match_length(ys, len(x))
    yst = match_length(yst, len(x))
    return y, ys, yst
