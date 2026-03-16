
"""
Harmonic plus Residual (HPR) Model analysis and synthesis functions.
Implements analysis, synthesis, and full model for HPR.
"""

import math
import numpy as np
from typing import Tuple
from scipy.fft import fft, ifft
from scipy.signal.windows import blackmanharris, triang

from smstools.models import dftModel as DFT
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
    Returns: hfreq, hmag, hphase (harmonic tracks), xr (residual signal)
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
    Returns: y (output), yh (harmonic component)
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
    t: float,
    nH: int,
    minf0: float,
    maxf0: float,
    f0et: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full analysis/synthesis of a sound using the harmonic plus residual model.
    Returns: y (output), yh (harmonic), xr (residual)
    """
    hN = N // 2
    hM1 = int(math.floor((w.size + 1) / 2))
    hM2 = int(math.floor(w.size / 2))
    Ns = 512
    H = Ns // 4
    hNs = Ns // 2
    pin = max(hNs, hM1)
    pend = x.size - max(hNs, hM1)
    yhw = np.zeros(Ns)
    xrw = np.zeros(Ns)
    yh = np.zeros(x.size)
    xr = np.zeros(x.size)
    w = w / sum(w)
    sw = np.zeros(Ns)
    ow = triang(2 * H)
    sw[hNs - H : hNs + H] = ow
    bh = blackmanharris(Ns)
    bh = bh / sum(bh)
    wr = bh
    sw[hNs - H : hNs + H] = sw[hNs - H : hNs + H] / bh[hNs - H : hNs + H]
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
        fftbuffer = np.real(ifft(Yh))
        yhw[: hNs - 1] = fftbuffer[hNs + 1 :]
        yhw[hNs - 1 :] = fftbuffer[: hNs + 1]
        fftbuffer = np.real(ifft(Xr))
        xrw[: hNs - 1] = fftbuffer[hNs + 1 :]
        xrw[hNs - 1 :] = fftbuffer[: hNs + 1]
        yh[ri : ri + Ns] += sw * yhw
        xr[ri : ri + Ns] += sw * xrw
        pin += H
    y = yh + xr
    return y, yh, xr
