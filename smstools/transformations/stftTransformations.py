# functions that implement transformations using the stft

import math

import numpy as np
from scipy.signal import resample

from smstools.models import dftModel as DFT


def stftFiltering(
    x: np.ndarray, fs: float, w: np.ndarray, N: int, H: int, filter: np.ndarray
) -> np.ndarray:
    """
    Apply a filter to a sound by using the STFT.

    Args:
        x: Input sound array.
        fs: Sampling rate.
        w: Analysis window array.
        N: FFT size.
        H: Hop size.
        filter: Magnitude response of filter (in dB).
    Returns:
        y: Output sound array.
    """

    M = w.size  # size of analysis window
    hM1 = int(math.floor((M + 1) / 2))  # half analysis window size by rounding
    hM2 = int(math.floor(M / 2))  # half analysis window size by floor
    x = np.append(
        np.zeros(hM2), x
    )  # add zeros at beginning to center first window at sample 0
    x = np.append(
        x, np.zeros(hM1)
    )  # add zeros at the end to analyze last sample
    pin = hM1  # initialize sound pointer in middle of analysis window
    pend = x.size - hM1  # last sample to start a frame
    w = w / sum(w)  # normalize analysis window
    y = np.zeros(x.size)  # initialize output array
    while pin <= pend:  # while sound pointer is smaller than last sample
        # -----analysis-----
        x1 = x[pin - hM1 : pin + hM2]  # select one frame of input sound
        mX, pX = DFT.dftAnal(x1, w, N)  # compute dft
        # ------transformation-----
        mY = mX + filter  # filter input magnitude spectrum
        # -----synthesis-----
        y1 = DFT.dftSynth(mY, pX, M)  # compute idft
        y[pin - hM1 : pin + hM2] += (
            H * y1
        )  # overlap-add to generate output sound
        pin += H  # advance sound pointer
    y = np.delete(
        y, range(hM2)
    )  # delete half of first window which was added in stftAnal
    y = np.delete(
        y, range(y.size - hM1, y.size)
    )  # add zeros at the end to analyze last sample
    return y


def stftMorph(
    x1: np.ndarray,
    x2: np.ndarray,
    fs: float,
    w1: np.ndarray,
    N1: int,
    w2: np.ndarray,
    N2: int,
    H1: int,
    smoothf: float,
    balancef: float,
) -> np.ndarray:
    """
    Morph two sounds using the STFT.

    Args:
        x1: First input sound array.
        x2: Second input sound array.
        fs: Sampling rate.
        w1: Analysis window for x1.
        N1: FFT size for x1.
        w2: Analysis window for x2.
        N2: FFT size for x2.
        H1: Hop size for x1.
        smoothf: Smooth factor for sound 2 (0 < smoothf <= 1).
        balancef: Balance between the two sounds (0 = x1, 1 = x2).
    Returns:
        y: Output sound array.
    """

    if N2 / 2 * smoothf < 3:  # raise exception if decimation factor too small
        raise ValueError("Smooth factor too small")

    if smoothf > 1:  # raise exception if decimation factor too big
        raise ValueError("Smooth factor above 1")

    if balancef > 1 or balancef < 0:  # raise exception if balancef outside 0-1
        raise ValueError("Balance factor outside range")

    if H1 <= 0:  # raise error if hop size 0 or negative
        raise ValueError("Hop size (H1) smaller or equal to 0")

    M1 = w1.size  # size of analysis window
    hM1_1 = int(
        math.floor((M1 + 1) / 2)
    )  # half analysis window size by rounding
    hM1_2 = int(math.floor(M1 / 2))  # half analysis window size by floor
    L = int(x1.size / H1)  # number of frames for x1
    x1 = np.append(
        np.zeros(hM1_2), x1
    )  # add zeros at beginning to center first window at sample 0
    x1 = np.append(
        x1, np.zeros(hM1_1)
    )  # add zeros at the end to analyze last sample
    pin1 = hM1_1  # initialize sound pointer in middle of analysis window
    w1 = w1 / sum(w1)  # normalize analysis window
    M2 = w2.size  # size of analysis window
    hM2_1 = int(
        math.floor((M2 + 1) / 2)
    )  # half analysis window size by rounding
    hM2_2 = int(math.floor(M2 / 2))  # half analysis window size by floor2
    H2 = int(x2.size / L)  # hop size for second sound
    x2 = np.append(
        np.zeros(hM2_2), x2
    )  # add zeros at beginning to center first window at sample 0
    x2 = np.append(
        x2, np.zeros(hM2_1)
    )  # add zeros at the end to analyze last sample
    pin2 = hM2_1  # initialize sound pointer in middle of analysis window
    y = np.zeros(x1.size)  # initialize output array
    for _ in range(L):
        # -----analysis-----
        mX1, pX1 = DFT.dftAnal(
            x1[pin1 - hM1_1 : pin1 + hM1_2], w1, N1
        )  # compute dft
        mX2, pX2 = DFT.dftAnal(
            x2[pin2 - hM2_1 : pin2 + hM2_2], w2, N2
        )  # compute dft
        # -----transformation-----
        mX2smooth = resample(
            np.maximum(-200, mX2), int(mX2.size * smoothf)
        )  # smooth spectrum of second sound
        mX2 = resample(
            mX2smooth, mX1.size
        )  # generate back the same size spectrum
        mY = balancef * mX2 + (1 - balancef) * mX1  # generate output spectrum
        # -----synthesis-----
        y[pin1 - hM1_1 : pin1 + hM1_2] += H1 * DFT.dftSynth(
            mY, pX1, M1
        )  # overlap-add to generate output sound
        pin1 += H1  # advance sound pointer
        pin2 += H2  # advance sound pointer
    y = np.delete(
        y, range(hM1_2)
    )  # delete half of first window which was added in stftAnal
    y = np.delete(
        y, range(y.size - hM1_1, y.size)
    )  # add zeros at the end to analyze last sample
    return y
