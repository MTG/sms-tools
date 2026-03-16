
"""
Harmonic plus Stochastic (HPS) Model analysis and synthesis functions.
Implements analysis, synthesis, and full model for HPS.
"""

import math
import numpy as np
from typing import Tuple
from scipy.fft import fft, ifft
from scipy.signal import resample
from scipy.signal.windows import blackmanharris, hann, triang

from smstools.models import dftModel as DFT
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
    Returns: hfreq, hmag, hphase (harmonic tracks), stocEnv (stochastic residual)
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
    Returns: y (output), yh (harmonic), yst (stochastic)
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
    t: float,
    nH: int,
    minf0: float,
    maxf0: float,
    f0et: float,
    stocf: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full analysis/synthesis of a sound using the harmonic plus stochastic model.
    Returns: y (output), yh (harmonic), yst (stochastic)
    """
    hM1 = int(math.floor((w.size + 1) / 2))
    hM2 = int(math.floor(w.size / 2))
    Ns = 512
    H = Ns // 4
    hNs = Ns // 2
    pin = max(hNs, hM1)
    pend = x.size - max(hNs, hM1)
    yhw = np.zeros(Ns)
    ystw = np.zeros(Ns)
    yh = np.zeros(x.size)
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
    hfreqp = []
    f0t = 0
    f0stable = 0
    while pin < pend:
        # Analysis
        x1 = x[pin - hM1 : pin + hM2]
        mX, pX = DFT.dftAnal(x1, w, N)
        ploc = UF.peakDetection(mX, t)
        iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)
        ipfreq = fs * iploc / N
        f0t = UF.f0Twm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable, fs=fs)
        if ((f0stable == 0) and (f0t > 0)) or (
            (f0stable > 0) and (np.abs(f0stable - f0t) < f0stable / 5.0)
        ):
            f0stable = f0t
        else:
            f0stable = 0
        hfreq, hmag, hphase = HM.harmonicDetection(
            ipfreq, ipmag, ipphase, f0t, nH, hfreqp, fs
        )
        hfreqp = hfreq
        ri = pin - hNs - 1
        xw2 = x[ri : ri + Ns] * wr
        fftbuffer = np.zeros(Ns)
        fftbuffer[:hNs] = xw2[hNs:]
        fftbuffer[hNs:] = xw2[:hNs]
        X2 = fft(fftbuffer)
        # Synthesis
        Yh = UF.genSpecSines(hfreq, hmag, hphase, Ns, fs)
        Xr = X2 - Yh
        mXr = 20 * np.log10(abs(Xr[:hNs]))
        mXrenv = resample(np.maximum(-200, mXr), int(mXr.size * stocf))
        stocEnv = resample(mXrenv, hNs)
        pYst = 2 * np.pi * np.random.rand(hNs)
        Yst = np.zeros(Ns, dtype=complex)
        Yst[:hNs] = 10 ** (stocEnv / 20) * np.exp(1j * pYst)
        Yst[hNs + 1 :] = 10 ** (stocEnv[:0:-1] / 20) * np.exp(-1j * pYst[:0:-1])

        fftbuffer = np.real(ifft(Yh))
        yhw[: hNs - 1] = fftbuffer[hNs + 1 :]
        yhw[hNs - 1 :] = fftbuffer[: hNs + 1]

        fftbuffer = np.real(ifft(Yst))
        ystw[: hNs - 1] = fftbuffer[hNs + 1 :]
        ystw[hNs - 1 :] = fftbuffer[: hNs + 1]

        yh[ri : ri + Ns] += sw * yhw
        yst[ri : ri + Ns] += sws * ystw
        pin += H

    y = yh + yst
    return y, yh, yst
