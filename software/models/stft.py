# functions that implement analysis and synthesis of sounds using the Short-Time Fourier Transform
# (for example usage check stft_function.py in the models_interface directory)

import numpy as np
import dftModel as DFT


def stft(x, w, N, H):
    """
	Analysis/synthesis of a sound using the short-time Fourier transform
	x: input sound, w: analysis window, N: FFT size, H: hop size
	returns y: output sound
	"""

    if (H <= 0):  # raise error if hop size 0 or negative
        raise ValueError("Hop size (H) smaller or equal to 0")

    M = w.size  # size of analysis window
    hM1 = (M + 1) // 2  # half analysis window size by rounding
    hM2 = M // 2  # half analysis window size by floor
    x = np.append(np.zeros(hM2), x)  # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(hM1))  # add zeros at the end to analyze last sample
    pin = hM1  # initialize sound pointer in middle of analysis window
    pend = x.size - hM1  # last sample to start a frame
    w = w / sum(w)  # normalize analysis window
    y = np.zeros(x.size)  # initialize output array
    while pin <= pend:  # while sound pointer is smaller than last sample
        # -----analysis-----
        x1 = x[pin - hM1:pin + hM2]  # select one frame of input sound
        mX, pX = DFT.dftAnal(x1, w, N)  # compute dft
        # -----synthesis-----
        y1 = DFT.dftSynth(mX, pX, M)  # compute idft
        y[pin - hM1:pin + hM2] += H * y1  # overlap-add to generate output sound
        pin += H  # advance sound pointer
    y = np.delete(y, range(hM2))  # delete half of first window which was added in stftAnal
    y = np.delete(y, range(y.size - hM1, y.size))  # delete half of the last window which as added in stftAnal
    return y


def stftAnal(x, w, N, H):
    """
	Analysis of a sound using the short-time Fourier transform
	x: input array sound, w: analysis window, N: FFT size, H: hop size
	returns xmX, xpX: magnitude and phase spectra
	"""
    if (H <= 0):  # raise error if hop size 0 or negative
        raise ValueError("Hop size (H) smaller or equal to 0")

    M = w.size  # size of analysis window
    hM1 = (M + 1) // 2  # half analysis window size by rounding
    hM2 = M // 2  # half analysis window size by floor
    x = np.append(np.zeros(hM2), x)  # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(hM2))  # add zeros at the end to analyze last sample
    pin = hM1  # initialize sound pointer in middle of analysis window
    pend = x.size - hM1  # last sample to start a frame
    w = w / sum(w)  # normalize analysis window
    xmX = []  # Initialise empty list for mX
    xpX = []  # Initialise empty list for pX
    while pin <= pend:  # while sound pointer is smaller than last sample
        x1 = x[pin - hM1:pin + hM2]  # select one frame of input sound
        mX, pX = DFT.dftAnal(x1, w, N)  # compute dft
        xmX.append(np.array(mX))  # Append output to list
        xpX.append(np.array(pX))
        pin += H  # advance sound pointer
    xmX = np.array(xmX)  # Convert to numpy array
    xpX = np.array(xpX)
    return xmX, xpX


def stftSynth(mY, pY, M, H):
    """
	Synthesis of a sound using the short-time Fourier transform
	mY: magnitude spectra, pY: phase spectra, M: window size, H: hop-size
	returns y: output sound
	"""
    hM1 = (M + 1) // 2  # half analysis window size by rounding
    hM2 = M // 2  # half analysis window size by floor
    nFrames = mY[:, 0].size  # number of frames
    y = np.zeros(nFrames * H + hM1 + hM2)  # initialize output array
    pin = hM1
    for i in range(nFrames):  # iterate over all frames
        y1 = DFT.dftSynth(mY[i, :], pY[i, :], M)  # compute idft
        y[pin - hM1:pin + hM2] += H * y1  # overlap-add to generate output sound
        pin += H  # advance sound pointer
    y = np.delete(y, range(hM2))  # delete half of first window which was added in stftAnal
    y = np.delete(y, range(y.size - hM1, y.size))  # delete the end of the sound that was added in stftAnal
    return y
