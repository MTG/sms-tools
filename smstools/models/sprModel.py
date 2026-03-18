"""Sinusoidal plus residual (SPR) model analysis and synthesis.

Provides frame-based SPR decomposition where sinusoidal content is modeled via
peak interpolation and additive synthesis, and residual is obtained by spectral
subtraction.
"""

import numpy as np
from scipy.fft import fft, ifft
from scipy.signal.windows import blackmanharris, triang

from smstools.models import dftModel as DFT
from smstools.models import sineModel as SM
from smstools.models import utilFunctions as UF


def _build_overlap_add_window(Ns: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Build normalized overlap-add synthesis window used by SPR.

    Args:
        Ns: Synthesis FFT size.
        H: Hop size.
    Returns:
        sw: Overlap-add window.
        bh: Blackman-Harris window.
    """
    hNs = Ns // 2
    sw = np.zeros(Ns)
    ow = triang(2 * H)
    sw[hNs - H : hNs + H] = ow
    bh = blackmanharris(Ns)
    bh = bh / np.sum(bh)
    sw[hNs - H : hNs + H] = sw[hNs - H : hNs + H] / bh[hNs - H : hNs + H]
    return sw, bh


def _undo_zero_phase(fftbuffer: np.ndarray, hNs: int) -> np.ndarray:
    """Undo zero-phase FFT arrangement back to time-domain frame order.

    Args:
        fftbuffer: FFT buffer.
        hNs: Half synthesis FFT size.
    Returns:
        yw: Time-domain frame.
    """
    yw = np.zeros(fftbuffer.size)
    yw[: hNs - 1] = fftbuffer[hNs + 1 :]
    yw[hNs - 1 :] = fftbuffer[: hNs + 1]
    return yw


def sprModelAnal(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze sound using sinusoidal-plus-residual decomposition.

    Args:
        x: Input sound.
        fs: Sampling rate.
        w: Analysis window.
        N: FFT size.
        H: Hop size.
        t: Peak threshold in negative dB.
        minSineDur: Minimum sinusoidal track duration.
        maxnSines: Maximum number of parallel sinusoids.
        freqDevOffset: Allowed frame-to-frame frequency deviation at 0 Hz.
        freqDevSlope: Frequency-deviation slope for higher frequencies.

    Returns:
        tfreq: Sinusoidal track frequencies.
        tmag: Sinusoidal track magnitudes.
        tphase: Sinusoidal track phases.
        xr: Residual signal.
    """

    tfreq, tmag, tphase = SM.sineModelAnal(
        x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope
    )
    Ns = 512
    xr = UF.sineSubtraction(x, Ns, H, tfreq, tmag, tphase, fs)
    return tfreq, tmag, tphase, xr


def sprModelSynth(
    tfreq: np.ndarray,
    tmag: np.ndarray,
    tphase: np.ndarray,
    xr: np.ndarray,
    N: int,
    H: int,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesize sound from sinusoidal tracks and residual signal.

    Args:
        tfreq: Sinusoidal track frequencies.
        tmag: Sinusoidal track magnitudes.
        tphase: Sinusoidal track phases.
        xr: Residual signal.
        N: Synthesis FFT size.
        H: Hop size.
        fs: Sampling rate.

    Returns:
        y: Combined output sound.
        ys: Sinusoidal component.
    """

    ys = SM.sineModelSynth(tfreq, tmag, tphase, N, H, fs)
    n = min(ys.size, xr.size)
    y = ys[:n] + xr[:n]
    return y, ys


def sprModel(
    x: np.ndarray,
    fs: float,
    w: np.ndarray,
    N: int,
    H: int,
    t: float,
    minSineDur: float = 0.02,
    maxnSines: int = 100,
    freqDevOffset: float = 20,
    freqDevSlope: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One-pass analysis/synthesis using sinusoidal-plus-residual model.

    Args:
        x: Input sound.
        fs: Sampling rate.
        w: Analysis window.
        N: FFT size (minimum 512).
        H: Hop size.
        t: Peak threshold in negative dB.
        minSineDur: Minimum sinusoidal track duration (in seconds, default 0.02).
        maxnSines: Maximum number of parallel sinusoids (default 100).
        freqDevOffset: Allowed frame-to-frame frequency deviation at 0 Hz (default 20).
        freqDevSlope: Frequency-deviation slope for higher frequencies (default 0.01).

    Returns:
        y: Reconstructed signal.
        ys: Sinusoidal component.
        xr: Residual component.
    """

    # Use analysis then synthesis, ensure output lengths match input
    tfreq, tmag, tphase, xr = sprModelAnal(
        x, fs, w, N, H, t, minSineDur, maxnSines, freqDevOffset, freqDevSlope
    )
    Ns = 512
    y, ys = sprModelSynth(tfreq, tmag, tphase, xr, Ns, H, fs)
    # Ensure output lengths match input
    if len(y) > len(x):
        y = y[:len(x)]
        ys = ys[:len(x)]
        xr = xr[:len(x)]
    elif len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
        ys = np.pad(ys, (0, len(x) - len(ys)))
        xr = np.pad(xr, (0, len(x) - len(xr)))
    return y, ys, xr
