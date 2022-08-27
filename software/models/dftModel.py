# functions that implement analysis and synthesis of sounds using the Discrete Fourier Transform
# (for example usage check dftModel_function.py in the models_interface directory)

import numpy as np
import math
from scipy.fftpack import fft, ifft
import utilFunctions as UF

tol = 1e-14  # threshold used to compute phase


def dftModel(x, w, N):
    """
	Analysis/synthesis of a signal using the discrete Fourier transform
	x: input signal, w: analysis window, N: FFT size
	returns y: output signal
	"""

    if not (UF.isPower2(N)):  # raise error if N not a power of two
        raise ValueError("FFT size (N) is not a power of 2")

    if (w.size > N):  # raise error if window size bigger than fft size
        raise ValueError("Window size (M) is bigger than FFT size")

    if all(x == 0):  # if input array is zeros return empty output
        return np.zeros(x.size)
    hN = (N // 2) + 1  # size of positive spectrum, it includes sample 0
    hM1 = (w.size + 1) // 2  # half analysis window size by rounding
    hM2 = int(math.floor(w.size / 2))  # half analysis window size by floor
    fftbuffer = np.zeros(N)  # initialize buffer for FFT
    y = np.zeros(x.size)  # initialize output array
    # ----analysis--------
    xw = x * w  # window the input sound
    fftbuffer[:hM1] = xw[hM2:]  # zero-phase window in fftbuffer
    fftbuffer[-hM2:] = xw[:hM2]
    X = fft(fftbuffer)  # compute FFT
    absX = abs(X[:hN])  # compute ansolute value of positive side
    absX[absX < np.finfo(float).eps] = np.finfo(float).eps  # if zeros add epsilon to handle log
    mX = 20 * np.log10(absX)  # magnitude spectrum of positive frequencies in dB
    pX = np.unwrap(np.angle(X[:hN]))  # unwrapped phase spectrum of positive frequencies
    # -----synthesis-----
    Y = np.zeros(N, dtype=complex)  # clean output spectrum
    Y[:hN] = 10 ** (mX / 20) * np.exp(1j * pX)  # generate positive frequencies
    Y[hN:] = 10 ** (mX[-2:0:-1] / 20) * np.exp(-1j * pX[-2:0:-1])  # generate negative frequencies
    fftbuffer = np.real(ifft(Y))  # compute inverse FFT
    y[:hM2] = fftbuffer[-hM2:]  # undo zero-phase window
    y[hM2:] = fftbuffer[:hM1]
    return y


def dftAnal(x, w, N):
    """
	Analysis of a signal using the discrete Fourier transform
	x: input signal, w: analysis window, N: FFT size 
	returns mX, pX: magnitude and phase spectrum
	"""

    if not (UF.isPower2(N)):  # raise error if N not a power of two
        raise ValueError("FFT size (N) is not a power of 2")

    if w.size > N:  # raise error if window size bigger than fft size
        raise ValueError("Window size (M) is bigger than FFT size")

    hN = (N // 2) + 1  # size of positive spectrum, it includes sample 0
    hM1 = (w.size + 1) // 2  # half analysis window size by rounding
    hM2 = w.size // 2  # half analysis window size by floor
    fftbuffer = np.zeros(N)  # initialize buffer for FFT
    w = w / sum(w)  # normalize analysis window
    xw = x * w  # window the input sound
    fftbuffer[:hM1] = xw[hM2:]  # zero-phase window in fftbuffer
    fftbuffer[-hM2:] = xw[:hM2]
    X = fft(fftbuffer)  # compute FFT
    absX = abs(X[:hN])  # compute ansolute value of positive side
    absX[absX < np.finfo(float).eps] = np.finfo(float).eps  # if zeros add epsilon to handle log
    mX = 20 * np.log10(absX)  # magnitude spectrum of positive frequencies in dB
    X[:hN].real[np.abs(X[:hN].real) < tol] = 0.0  # for phase calculation set to 0 the small values
    X[:hN].imag[np.abs(X[:hN].imag) < tol] = 0.0  # for phase calculation set to 0 the small values
    pX = np.unwrap(np.angle(X[:hN]))  # unwrapped phase spectrum of positive frequencies
    return mX, pX


def dftSynth(mX, pX, M):
    """
	Synthesis of a signal using the discrete Fourier transform
	mX: magnitude spectrum, pX: phase spectrum, M: window size
	returns y: output signal
	"""

    hN = mX.size  # size of positive spectrum, it includes sample 0
    N = (hN - 1) * 2  # FFT size
    if not (UF.isPower2(N)):  # raise error if N not a power of two, thus mX is wrong
        raise ValueError("size of mX is not (N/2)+1")

    hM1 = int(math.floor((M + 1) / 2))  # half analysis window size by rounding
    hM2 = int(math.floor(M / 2))  # half analysis window size by floor
    y = np.zeros(M)  # initialize output array
    Y = np.zeros(N, dtype=complex)  # clean output spectrum
    Y[:hN] = 10 ** (mX / 20) * np.exp(1j * pX)  # generate positive frequencies
    Y[hN:] = 10 ** (mX[-2:0:-1] / 20) * np.exp(-1j * pX[-2:0:-1])  # generate negative frequencies
    fftbuffer = np.real(ifft(Y))  # compute inverse FFT
    y[:hM2] = fftbuffer[-hM2:]  # undo zero-phase window
    y[hM2:] = fftbuffer[:hM1]
    return y
