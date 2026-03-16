# Short-Time Fourier Transform (STFT) utilities used by sms-tools.
#
# This module provides:
# - `stftAnal`: frame-wise STFT analysis (magnitude/phase)
# - `stftSynth`: overlap-add synthesis from STFT spectra
# - `stft`: convenience analysis+synthesis round-trip

import numpy as np
from smstools.models import dftModel as DFT


def stft(x, w, N, H):
    """
    Analysis/synthesis of a sound using the short-time Fourier transform
    x: input sound, w: analysis window, N: FFT size, H: hop size
    returns y: output sound
    """

    if H <= 0:  # raise error if hop size 0 or negative
        raise ValueError("Hop size (H) smaller or equal to 0")

    M = w.size  # size of analysis window
    hM1 = (M + 1) // 2  # half analysis window size by rounding
    hM2 = M // 2  # half analysis window size by floor
    x = np.concatenate(
        (np.zeros(hM2), x, np.zeros(hM1))
    )  # add zeros at beginning/end for centered analysis
    pin = hM1  # initialize sound pointer in middle of analysis window
    pend = x.size - hM1  # last sample to start a frame
    w = w / np.sum(w)  # normalize analysis window
    y = np.zeros(x.size)  # initialize output array
    while pin <= pend:  # while sound pointer is smaller than last sample
        # -----analysis-----
        x1 = x[pin - hM1 : pin + hM2]  # select one frame of input sound
        mX, pX = DFT.dftAnal(x1, w, N)  # compute dft
        # -----synthesis-----
        y1 = DFT.dftSynth(mX, pX, M)  # compute idft
        y[pin - hM1 : pin + hM2] += H * y1  # overlap-add to generate output sound
        pin += H  # advance sound pointer
    return y[hM2 : y.size - hM1]  # remove padding


def stftAnal(x, w, N, H):
    """
    Analysis of a sound using the short-time Fourier transform
    x: input array sound, w: analysis window, N: FFT size, H: hop size
    returns xmX, xpX: magnitude and phase spectra
    """
    if H <= 0:  # raise error if hop size 0 or negative
        raise ValueError("Hop size (H) smaller or equal to 0")

    M = w.size  # size of analysis window
    hM1 = (M + 1) // 2  # half analysis window size by rounding
    hM2 = M // 2  # half analysis window size by floor
    x = np.concatenate(
        (np.zeros(hM2), x, np.zeros(hM2))
    )  # add zeros at beginning/end for centered analysis
    pin = hM1  # initialize sound pointer in middle of analysis window
    pend = x.size - hM1  # last sample to start a frame
    w = w / np.sum(w)  # normalize analysis window

    if pin > pend:
        hN = (N // 2) + 1
        return np.empty((0, hN)), np.empty((0, hN))

    nFrames = 1 + (pend - pin) // H
    hN = (N // 2) + 1
    xmX = np.empty((nFrames, hN))
    xpX = np.empty((nFrames, hN))

    frame = 0
    while pin <= pend:  # while sound pointer is smaller than last sample
        x1 = x[pin - hM1 : pin + hM2]  # select one frame of input sound
        mX, pX = DFT.dftAnal(x1, w, N)  # compute dft
        xmX[frame, :] = mX
        xpX[frame, :] = pX
        frame += 1
        pin += H  # advance sound pointer
    return xmX, xpX


def stftSynth(mY, pY, M, H):
    """
    Synthesis of a sound using the short-time Fourier transform
    mY: magnitude spectra, pY: phase spectra, M: window size, H: hop-size
    returns y: output sound
    """
    hM1 = (M + 1) // 2  # half analysis window size by rounding
    hM2 = M // 2  # half analysis window size by floor
    nFrames = mY.shape[0]  # number of frames
    y = np.zeros(nFrames * H + hM1 + hM2)  # initialize output array
    pin = hM1
    for i in range(nFrames):  # iterate over all frames
        y1 = DFT.dftSynth(mY[i, :], pY[i, :], M)  # compute idft
        y[pin - hM1 : pin + hM2] += H * y1  # overlap-add to generate output sound
        pin += H  # advance sound pointer
    return y[hM2 : y.size - hM1]  # remove padding
