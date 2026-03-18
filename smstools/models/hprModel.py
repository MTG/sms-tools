"""
Harmonic plus Residual (HPR) Model analysis and synthesis functions.
Implements analysis, synthesis, and full model for HPR.
"""

from typing import Tuple

import numpy as np

from smstools.models import harmonicModel as HM
from smstools.models import sineModel as SM
from smstools.models import utilFunctions as UF


def hprModelAnal(
    x: np.ndarray,
    fs: float,
    w: np.ndarray,
    N: int,
    H: int,
    t: float,
    minSineDur: float,
    nH: int,
    minf0: float,
    maxf0: float,
    f0et: float,
    harmDevSlope: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze a sound using the harmonic plus residual model.

    Args:
        x: Input sound array.
        fs: Sampling rate.
        w: Analysis window array.
        N: FFT size.
        H: Hop size.
        t: Peak threshold in negative dB.
        minSineDur: Minimum harmonic track duration.
        nH: Maximum number of harmonics.
        minf0: Minimum f0 frequency in Hz.
        maxf0: Maximum f0 frequency in Hz.
        f0et: Error threshold in f0 detection.
        harmDevSlope: Slope of harmonic deviation.

    Returns:
        hfreq: Harmonic track frequencies.
        hmag: Harmonic track magnitudes.
        hphase: Harmonic track phases.
        xr: Residual signal.
    """
    hfreq, hmag, hphase = HM.harmonicModelAnal(
        x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur
    )
    Ns = 512
    xr = UF.sineSubtraction(x, Ns, H, hfreq, hmag, hphase, fs)
    return hfreq, hmag, hphase, xr


def hprModelSynth(
    hfreq: np.ndarray,
    hmag: np.ndarray,
    hphase: np.ndarray,
    xr: np.ndarray,
    N: int,
    H: int,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesize a sound using the harmonic plus residual model.

    Args:
        hfreq: Harmonic track frequencies.
        hmag: Harmonic track magnitudes.
        hphase: Harmonic track phases.
        xr: Residual signal.
        N: Synthesis FFT size.
        H: Hop size.
        fs: Sampling rate.

    Returns:
        y: Output sound.
        yh: Harmonic component.
    """
    yh = SM.sineModelSynth(hfreq, hmag, hphase, N, H, fs)
    n = min(yh.size, xr.size)
    y = yh[:n] + xr[:n]
    return y, yh


def hprModel(
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full analysis/synthesis of a sound using the harmonic plus residual model.

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

    Returns:
        y: Output sound.
        yh: Harmonic component.
        xr: Residual component.
    """
    # Use analysis then synthesis, ensure output lengths match input
    hfreq, hmag, hphase, xr = hprModelAnal(
        x, fs, w, N, H, t, minSineDur, nH, minf0, maxf0, f0et, harmDevSlope
    )
    y, yh = hprModelSynth(hfreq, hmag, hphase, xr, Ns, H, fs)
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
    xr = match_length(xr, len(x))
    return y, yh, xr
