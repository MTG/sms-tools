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


def _build_overlap_add_window(Ns, H):
    """Build normalized overlap-add synthesis window used by SPR."""
    hNs = Ns // 2
    sw = np.zeros(Ns)
    ow = triang(2 * H)
    sw[hNs - H : hNs + H] = ow
    bh = blackmanharris(Ns)
    bh = bh / np.sum(bh)
    sw[hNs - H : hNs + H] = sw[hNs - H : hNs + H] / bh[hNs - H : hNs + H]
    return sw, bh


def _undo_zero_phase(fftbuffer, hNs):
    """Undo zero-phase FFT arrangement back to time-domain frame order."""
    yw = np.zeros(fftbuffer.size)
    yw[: hNs - 1] = fftbuffer[hNs + 1 :]
    yw[hNs - 1 :] = fftbuffer[: hNs + 1]
    return yw


def sprModelAnal(x, fs, w, N, H, t, minSineDur, maxnSines, freqDevOffset, freqDevSlope):
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
        tfreq, tmag, tphase: Sinusoidal tracks.
        xr: Residual signal.
    """

    tfreq, tmag, tphase = SM.sineModelAnal(
        x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope
    )
    Ns = 512
    xr = UF.sineSubtraction(x, Ns, H, tfreq, tmag, tphase, fs)
    return tfreq, tmag, tphase, xr


def sprModelSynth(tfreq, tmag, tphase, xr, N, H, fs):
    """
    Synthesize sound from sinusoidal tracks and residual signal.

    Args:
        tfreq, tmag, tphase: Sinusoidal track frequencies, magnitudes, and phases.
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


def sprModel(x, fs, w, N, t):
    """
    One-pass analysis/synthesis using sinusoidal-plus-residual model.

    Args:
        x: Input sound.
        fs: Sampling rate.
        w: Analysis window.
        N: FFT size (minimum 512).
        t: Peak threshold in negative dB.

    Returns:
        y: Reconstructed signal.
        ys: Sinusoidal component.
        xr: Residual component.
    """

    hM1 = (w.size + 1) // 2
    hM2 = w.size // 2
    Ns = 512
    H = Ns // 4
    hNs = Ns // 2
    pin = max(hNs, hM1)
    pend = x.size - max(hNs, hM1)
    ys = np.zeros(x.size)
    xr = np.zeros(x.size)
    w = w / np.sum(w)
    sw, wr = _build_overlap_add_window(Ns, H)

    while pin < pend:
        # Analysis
        x1 = x[pin - hM1 : pin + hM2]
        mX, pX = DFT.dftAnal(x1, w, N)
        ploc = UF.peakDetection(mX, t)
        iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)
        ipfreq = fs * iploc / float(N)

        ri = pin - hNs - 1
        xw2 = x[ri : ri + Ns] * wr
        fftbuffer = np.zeros(Ns)
        fftbuffer[:hNs] = xw2[hNs:]
        fftbuffer[hNs:] = xw2[:hNs]
        X2 = fft(fftbuffer)

        # Synthesis
        Ys = UF.genSpecSines(ipfreq, ipmag, ipphase, Ns, fs)
        Xr = X2 - Ys

        ysw = _undo_zero_phase(np.real(ifft(Ys)), hNs)
        xrw = _undo_zero_phase(np.real(ifft(Xr)), hNs)

        ys[ri : ri + Ns] += sw * ysw
        xr[ri : ri + Ns] += sw * xrw
        pin += H

    y = ys + xr
    return y, ys, xr
