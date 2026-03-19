# Short-Time Fourier Transform (STFT) utilities used by sms-tools.
#
# This module provides:
# - `stftAnal`: frame-wise STFT analysis (magnitude/phase)
# - `stftSynth`: overlap-add synthesis from STFT spectra
# - `stft`: convenience analysis+synthesis round-trip


import numpy as np

from smstools.models import dftModel as DFT


def stftAnal(
    x: np.ndarray, w: np.ndarray, N: int, H: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Analysis of a sound using the short-time Fourier transform.

    Args:
        x: Input sound array.
        w: Analysis window array.
        N: FFT size.
        H: Hop size.

    Returns:
        xmX: Magnitude spectra (frames x bins).
        xpX: Phase spectra (frames x bins).

    Raises:
        ValueError: If hop size H is zero or negative.
    """
    if H <= 0:
        raise ValueError(
            f"Hop size (H={H}) smaller or equal to 0. "
            f"Provided window size: {w.size}."
        )
    M = w.size
    hM1 = (M + 1) // 2
    hM2 = M // 2
    x = np.concatenate((np.zeros(hM2), x, np.zeros(hM2)))
    pin = hM1
    pend = x.size - hM1
    w = w / np.sum(w)
    if pin > pend:
        hN = (N // 2) + 1
        return np.empty((0, hN)), np.empty((0, hN))
    nFrames = 1 + (pend - pin) // H
    hN = (N // 2) + 1
    xmX = np.empty((nFrames, hN))
    xpX = np.empty((nFrames, hN))
    frame = 0
    while pin <= pend:
        x1 = x[pin - hM1 : pin + hM2]
        mX, pX = DFT.dftAnal(x1, w, N)
        xmX[frame, :] = mX
        xpX[frame, :] = pX
        frame += 1
        pin += H
    return xmX, xpX


def stftSynth(mY: np.ndarray, pY: np.ndarray, M: int, H: int) -> np.ndarray:
    """
    Synthesis of a sound using the short-time Fourier transform.

    Args:
        mY: Magnitude spectra (frames x bins).
        pY: Phase spectra (frames x bins).
        M: Window size.
        H: Hop size.

    Returns:
        y: Output sound array (same shape as input, minus padding).
    """
    hM1 = (M + 1) // 2
    hM2 = M // 2
    nFrames = mY.shape[0]
    y = np.zeros(nFrames * H + hM1 + hM2)
    pin = hM1
    for i in range(nFrames):
        y1 = DFT.dftSynth(mY[i, :], pY[i, :], M)
        y[pin - hM1 : pin + hM2] += H * y1
        pin += H
    return y[hM2 : y.size - hM1]


def stft(x: np.ndarray, w: np.ndarray, N: int, H: int) -> np.ndarray:
    """
    Analysis/synthesis of a sound using the short-time Fourier transform.

    Args:
        x: Input sound array.
        w: Analysis window array.
        N: FFT size.
        H: Hop size.

    Returns:
        y: Output sound array (same shape as input, minus padding).

    Raises:
        ValueError: If hop size H is zero or negative.
    """
    if H <= 0:
        raise ValueError(
            f"Hop size (H={H}) smaller or equal to 0. "
            f"Provided window size: {w.size}."
        )
    M = w.size
    mX, pX = stftAnal(x, w, N, H)
    y = stftSynth(mX, pX, M, H)
    # Ensure output length matches input
    if len(y) > len(x):
        y = y[: len(x)]
    elif len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y
