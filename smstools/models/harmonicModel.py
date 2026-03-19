# Harmonic Model analysis/synthesis utilities used by sms-tools.
#
# This module includes:
# - `f0Detection`: frame-wise fundamental frequency estimation
# - `harmonicDetection`: harmonic peak selection from spectral peaks
# - `harmonicModel`: analysis+synthesis round-trip
# - `harmonicModelAnal` / `harmonicModelSynth`: separated analysis/synthesis APIs


import math

import numpy as np

from smstools.models import dftModel as DFT
from smstools.models import sineModel as SM
from smstools.models import utilFunctions as UF


def f0Detection(
    x: np.ndarray,
    fs: float,
    w: np.ndarray,
    N: int,
    H: int,
    t: float,
    minf0: float,
    maxf0: float,
    f0et: float,
) -> np.ndarray:
    """
    Fundamental frequency detection of a sound using twm algorithm.

    Args:
        x: Input sound array.
        fs: Sampling rate.
        w: Analysis window array.
        N: FFT size.
        H: Hop size.
        t: Threshold in negative dB.
        minf0: Minimum f0 frequency in Hz.
        maxf0: Maximum f0 frequency in Hz.
        f0et: Error threshold in the f0 detection.

    Returns:
        f0: Fundamental frequency array.
    """
    if minf0 < 0:  # raise exception if minf0 is smaller than 0
        raise ValueError(
            "Minumum fundamental frequency (minf0) smaller than 0"
        )

    if maxf0 >= fs / 2.0:  # raise exception if maxf0 is bigger than Nyquist
        raise ValueError(
            "Maximum fundamental frequency (maxf0) bigger than Nyquist frequency"
        )

    if H <= 0:  # raise error if hop size 0 or negative
        raise ValueError("Hop size (H) smaller or equal to 0")

    hM1 = int(
        math.floor((w.size + 1) / 2)
    )  # half analysis window size by rounding
    hM2 = int(math.floor(w.size / 2))  # half analysis window size by floor
    x = np.append(
        np.zeros(hM2), x
    )  # add zeros at beginning to center first window at sample 0
    x = np.append(
        x, np.zeros(hM1)
    )  # add zeros at the end to analyze last sample
    pin = hM1  # init sound pointer in middle of anal window
    pend = x.size - hM1  # last sample to start a frame
    w = w / sum(w)  # normalize analysis window
    f0 = []  # initialize f0 output
    f0t = 0  # initialize f0 track
    f0stable = 0  # initialize f0 stable
    # f0candidate = 0  # unused variable
    while pin < pend:
        x1 = x[pin - hM1 : pin + hM2]  # select frame
        mX, pX = DFT.dftAnal(x1, w, N)  # compute dft
        ploc = UF.peakDetection(mX, t)  # detect peak locations
        iploc, ipmag, ipphase = UF.peakInterp(
            mX, pX, ploc
        )  # refine peak values
        ipfreq = fs * iploc / N  # convert locations to Hz
        f0t = UF.f0Twm(
            ipfreq, ipmag, f0et, minf0, maxf0, f0stable, fs=fs
        )  # find f0
        f0stable = f0t if f0t > 0 and abs(f0t - f0stable) < f0et else 0
        # f0candidate = f0t  # unused variable
        f0 = np.append(f0, f0t)  # add f0 to output array
        pin += H  # advance sound pointer
    return f0


def harmonicDetection(
    pfreq: np.ndarray,
    pmag: np.ndarray,
    pphase: np.ndarray,
    f0: float,
    nH: int,
    hfreqp: np.ndarray,
    fs: float,
    harmDevSlope: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detection of the harmonics of a frame from a set of spectral peaks using f0.

    Args:
        pfreq: Peak frequencies.
        pmag: Peak magnitudes.
        pphase: Peak phases.
        f0: Fundamental frequency.
        nH: Number of harmonics.
        hfreqp: Harmonic frequencies of previous frame.
        fs: Sampling rate.
        harmDevSlope: Slope of change of the deviation allowed to perfect harmonic.

    Returns:
        hfreq: Harmonic frequencies.
        hmag: Harmonic magnitudes.
        hphase: Harmonic phases.
    """

    if f0 <= 0:  # if no f0 return no harmonics
        return np.zeros(nH), np.zeros(nH), np.zeros(nH)
    hfreq = np.zeros(nH)  # initialize harmonic frequencies
    hmag = np.zeros(nH) - 100  # initialize harmonic magnitudes
    hphase = np.zeros(nH)  # initialize harmonic phases
    hf = f0 * np.arange(1, nH + 1)  # initialize harmonic frequencies
    hi = 0  # initialize harmonic index
    if len(hfreqp) == 0:
        hfreqp = hf
    while (f0 > 0) and (hi < nH) and (hf[hi] < fs / 2):  # find harmonic peaks
        pei = np.argmin(abs(pfreq - hf[hi]))  # closest peak
        dev1 = abs(pfreq[pei] - hf[hi])  # deviation from perfect harmonic
        dev2 = (
            abs(pfreq[pei] - hfreqp[hi]) if hfreqp[hi] > 0 else fs
        )  # deviation from previous frame
        threshold = f0 / 3 + harmDevSlope * pfreq[pei]
        if (dev1 < threshold) or (dev2 < threshold):
            hfreq[hi] = pfreq[pei]
            hmag[hi] = pmag[pei]
            hphase[hi] = pphase[pei]
        hi += 1  # increase harmonic index
    return hfreq, hmag, hphase


def harmonicModelAnal(
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
    harmDevSlope: float = 0.01,
    minSineDur: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analysis of a sound using the sinusoidal harmonic model.

    Args:
        x: Input sound array.
        fs: Sampling rate.
        w: Analysis window array.
        N: FFT size (minimum 512).
        H: Hop size.
        t: Threshold in negative dB.
        nH: Maximum number of harmonics.
        minf0: Minimum f0 frequency in Hz.
        maxf0: Maximum f0 frequency in Hz.
        f0et: Error threshold in the f0 detection.
        harmDevSlope: Slope of harmonic deviation.
        minSineDur: Minimum length of harmonics.

    Returns:
        xhfreq: Harmonic frequencies.
        xhmag: Harmonic magnitudes.
        xhphase: Harmonic phases.
    """

    if minSineDur < 0:  # raise exception if minSineDur is smaller than 0
        raise ValueError("Minimum duration of sine tracks smaller than 0")

    hM1 = int(
        math.floor((w.size + 1) / 2)
    )  # half analysis window size by rounding
    hM2 = int(math.floor(w.size / 2))  # half analysis window size by floor
    x = np.append(
        np.zeros(hM2), x
    )  # add zeros at beginning to center first window at sample 0
    x = np.append(
        x, np.zeros(hM2)
    )  # add zeros at the end to analyze last sample
    pin = hM1  # init sound pointer in middle of anal window
    pend = x.size - hM1  # last sample to start a frame
    w = w / sum(w)  # normalize analysis window
    hfreqp = []  # initialize harmonic frequencies of previous frame
    f0t = 0  # initialize f0 track
    f0stable = 0  # initialize f0 stable
    xhfreq = []  # initialize output list for harmonic frequencies
    xhmag = []
    xhphase = []
    f0 = []
    while pin <= pend:
        x1 = x[pin - hM1 : pin + hM2]  # select frame
        mX, pX = DFT.dftAnal(x1, w, N)  # compute dft
        ploc = UF.peakDetection(mX, t)  # detect peak locations
        iploc, ipmag, ipphase = UF.peakInterp(
            mX, pX, ploc
        )  # refine peak values
        ipfreq = fs * iploc / N  # convert locations to Hz
        f0t = UF.f0Twm(
            ipfreq, ipmag, f0et, minf0, maxf0, f0stable, fs=fs
        )  # find f0
        f0stable = f0t if f0t > 0 and abs(f0t - f0stable) < f0et else 0
        # f0candidate = f0t  # unused variable
        f0.append(f0t)  # add f0 to output list
        # Harmonic detection
        hfreq, hmag, hphase = harmonicDetection(
            ipfreq, ipmag, ipphase, f0t, nH, hfreqp, fs, harmDevSlope
        )
        xhfreq.append(hfreq)
        xhmag.append(hmag)
        xhphase.append(hphase)
        hfreqp = hfreq
        pin += H  # advance sound pointer
    xhfreq = SM.cleaningSineTracks(
        np.array(xhfreq), round(fs * minSineDur / H)
    )  # delete tracks shorter than minSineDur
    xhmag = np.array(xhmag)
    xhphase = np.array(xhphase)
    # Zero out corresponding locations in xhmag and xhphase
    mask = xhfreq == 0
    xhmag[mask] = 0
    xhphase[mask] = 0
    return xhfreq, xhmag, xhphase


def harmonicModel(
    x: np.ndarray,
    fs: float,
    w: np.ndarray,
    N: int,
    t: float,
    nH: int,
    minf0: float,
    maxf0: float,
    f0et: float,
) -> np.ndarray:
    """
    Analysis/synthesis of a sound using the sinusoidal harmonic model.

    Args:
        x: Input sound array.
        fs: Sampling rate.
        w: Analysis window array.
        N: FFT size (minimum 512).
        t: Threshold in negative dB.
        nH: Maximum number of harmonics.
        minf0: Minimum f0 frequency in Hz.
        maxf0: Maximum f0 frequency in Hz.
        f0et: Error threshold in the f0 detection.

    Returns:
        y: Output sound array.
    """
    # Enforce fixed synthesis parameters
    # User can specify N for analysis
    H = 128
    Ns = 512
    hfreq, hmag, hphase = harmonicModelAnal(
        x, fs, w, N, H, t, nH, minf0, maxf0, f0et
    )
    # Synthesize using fixed synthesis parameters
    y = SM.sineModelSynth(hfreq, hmag, hphase, Ns, H, fs)
    # Ensure output length matches input
    if len(y) > len(x):
        y = y[: len(x)]
    elif len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y
