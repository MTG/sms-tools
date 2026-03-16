
"""
Sinusoidal plus Stochastic Model (S+S) analysis and synthesis functions.
Implements analysis, synthesis, and full model for S+S.
"""

import math
import numpy as np
from typing import Tuple
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
    Returns: tfreq, tmag, tphase, stocEnv
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
    Returns: y (output), ys (sinusoidal), yst (stochastic)
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
    t: float,
    stocf: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full analysis/synthesis of a sound using the sinusoidal plus stochastic model.
    Returns: y (output), ys (sinusoidal), yst (stochastic)
    """
    hM1 = int(math.floor((w.size + 1) / 2))
    hM2 = int(math.floor(w.size / 2))
    Ns = 512
    H = Ns // 4
    hNs = Ns // 2
    pin = max(hNs, hM1)
    pend = x.size - max(hNs, hM1)
    ysw = np.zeros(Ns)
    ystw = np.zeros(Ns)
    ys = np.zeros(x.size)
    yst = np.zeros(x.size)
    w = w / sum(w)
    sw = np.zeros(Ns)
    ow = triang(2 * H)
    sw[hNs - H : hNs + H] = ow
    bh = blackmanharris(Ns)
    bh = bh / sum(bh)
    wr = bh
    sw[hNs - H : hNs + H] = sw[hNs - H : hNs + H] / bh[hNs - H : hNs + H]
    sws = H * hann(Ns) / 2

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
        mXr = 20 * np.log10(abs(Xr[:hNs]))
        mXrenv = resample(np.maximum(-200, mXr), int(mXr.size * stocf))
        stocEnv = resample(mXrenv, hNs)
        pYst = 2 * np.pi * np.random.rand(hNs)
        Yst = np.zeros(Ns, dtype=complex)
        Yst[:hNs] = 10 ** (stocEnv / 20) * np.exp(1j * pYst)
        Yst[hNs + 1 :] = 10 ** (stocEnv[:0:-1] / 20) * np.exp(-1j * pYst[:0:-1])

        fftbuffer = np.real(ifft(Ys))
        ysw[: hNs - 1] = fftbuffer[hNs + 1 :]
        ysw[hNs - 1 :] = fftbuffer[: hNs + 1]

        fftbuffer = np.real(ifft(Yst))
        ystw[: hNs - 1] = fftbuffer[hNs + 1 :]
        ystw[hNs - 1 :] = fftbuffer[: hNs + 1]

        ys[ri : ri + Ns] += sw * ysw
        yst[ri : ri + Ns] += sws * ystw
        pin += H

    y = ys + yst
    return y, ys, yst
