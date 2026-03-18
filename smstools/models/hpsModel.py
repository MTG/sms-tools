"""
Harmonic plus Stochastic (HPS) Model analysis and synthesis functions.
Implements analysis, synthesis, and full model for HPS.
"""

import numpy as np
from typing import Tuple
from smstools.models import harmonicModel as HM
from smstools.models import sineModel as SM
from smstools.models import stochasticModel as STM
from smstools.models import utilFunctions as UF


def hpsModelAnal(
    x: np.ndarray,
    fs: float,
    w: np.ndarray,
    N: int,
    H: int,
    t: float,
    nH: int,
    minf0: float,
    maxf0: float,
    f0et: float,
    harmDevSlope: float,
    minSineDur: float,
    Ns: int,
    stocf: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze a sound using the harmonic plus stochastic model.

    Args:
        x: Input sound array.
        fs: Sampling rate.
        w: Analysis window array.
        N: FFT size.
        H: Hop size.
        t: Peak threshold in negative dB.
        nH: Maximum number of harmonics.
        minf0: Minimum f0 frequency in Hz.
        maxf0: Maximum f0 frequency in Hz.
        f0et: Error threshold in f0 detection.
        harmDevSlope: Slope of harmonic deviation.
        minSineDur: Minimum harmonic track duration.
        Ns: Synthesis FFT size.
        stocf: Decimation factor for stochastic envelope.

    Returns:
        hfreq: Harmonic track frequencies.
        hmag: Harmonic track magnitudes.
        hphase: Harmonic track phases.
        stocEnv: Stochastic envelope.
    """
    hfreq, hmag, hphase = HM.harmonicModelAnal(
        x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur
    )
    xr = UF.sineSubtraction(x, Ns, H, hfreq, hmag, hphase, fs)
    stocEnv = STM.stochasticModelAnal(xr, H, H * 2, stocf)
    return hfreq, hmag, hphase, stocEnv


def hpsModelSynth(
    hfreq: np.ndarray,
    hmag: np.ndarray,
    hphase: np.ndarray,
    stocEnv: np.ndarray,
    N: int,
    H: int,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesize a sound using the harmonic plus stochastic model.

    Args:
        hfreq: Harmonic track frequencies.
        hmag: Harmonic track magnitudes.
        hphase: Harmonic track phases.
        stocEnv: Stochastic envelope.
        N: Synthesis FFT size.
        H: Hop size.
        fs: Sampling rate.

    Returns:
        y: Output sound.
        yh: Harmonic component.
        yst: Stochastic component.
    """
    yh = SM.sineModelSynth(hfreq, hmag, hphase, N, H, fs)
    yst = STM.stochasticModelSynth(stocEnv, H, H * 2)
    n = min(yh.size, yst.size)
    y = yh[:n] + yst[:n]
    return y, yh, yst


def hpsModel(
    x: np.ndarray,
    fs: float,
    w: np.ndarray,
    N: int,
    H: int = 128,
    t: float = -80,
    nH: int = 100,
    minf0: float = 50.0,
    maxf0: float = 5000.0,
    f0et: float = 5.0,
    harmDevSlope: float = 0.01,
    minSineDur: float = 0.02,
    Ns: int = 512,
    stocf: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full analysis/synthesis of a sound using the harmonic plus stochastic model.

    Args:
        x: Input sound array.
        fs: Sampling rate.
        w: Analysis window array.
        N: FFT size.
        t: Peak threshold in negative dB.
        nH: Maximum number of harmonics.
        minf0: Minimum f0 frequency in Hz.
        maxf0: Maximum f0 frequency in Hz.
        f0et: Error threshold in f0 detection.
        stocf: Decimation factor for stochastic envelope.

    Returns:
        y: Output sound.
        yh: Harmonic component.
        yst: Stochastic component.
    """
    # Use analysis then synthesis, ensure output lengths match input
    hfreq, hmag, hphase, stocEnv = hpsModelAnal(
        x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf
    )
    y, yh, yst = hpsModelSynth(hfreq, hmag, hphase, stocEnv, Ns, H, fs)
    # Ensure output lengths match input
    def match_length(arr, target_len):
        if len(arr) > target_len:
            return arr[:target_len]
        elif len(arr) < target_len:
            return np.pad(arr, (0, target_len - len(arr)))
        else:
            return arr
    y = match_length(y, len(x))
    yh = match_length(yh, len(x))
    yst = match_length(yst, len(x))
    return y, yh, yst
