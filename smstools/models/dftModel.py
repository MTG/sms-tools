# Discrete Fourier Transform (DFT) utilities used by sms-tools.
#
# This module provides:
# - `dftAnal`: analysis of a single windowed frame (magnitude/phase)
# - `dftSynth`: synthesis of a time-domain frame from magnitude/phase
# - `dftModel`: convenience analysis+synthesis round-trip for one frame

import numpy as np
from scipy.fft import irfft, rfft
from smstools.models import utilFunctions as UF
tol = 1e-14  # threshold used to compute phase
_EPS = np.finfo(float).eps


def _validate_window_fft_size(w, N):
    if not UF.isPower2(N):
        raise ValueError("FFT size (N) is not a power of 2")
    if w.size > N:
        raise ValueError("Window size (M) is bigger than FFT size")


def _positive_spectrum_from_fft(X, hN):
    """
    The function `_positive_spectrum_from_fft` calculates the magnitude spectrum in decibels from the
    FFT output.
    """
    absX = np.abs(X[:hN])
    np.maximum(absX, _EPS, out=absX)  # avoid log of zero by replacing small values with _EPS
    return 20 * np.log10(absX)


def _build_positive_spectrum(mX, pX):
    pos_mag = 10 ** (mX / 20)
    return pos_mag * np.exp(1j * pX)


def dftModel(x, w, N):
    """
    Analysis/synthesis of a signal using the discrete Fourier transform
    x: input signal, w: analysis window, N: FFT size
    returns y: output signal
    """

    _validate_window_fft_size(w, N)

    if not np.any(x):  # if input array is zeros return empty output
        return np.zeros(x.size)

    hN = (N // 2) + 1  # size of positive spectrum, it includes sample 0
    hM1 = (w.size + 1) // 2  # half analysis window size by rounding
    hM2 = w.size // 2  # half analysis window size by floor
    fftbuffer = np.zeros(N)  # initialize buffer for FFT
    y = np.zeros(x.size)  # initialize output array

    # ----analysis--------
    xw = x * w  # window the input sound
    fftbuffer[:hM1] = xw[hM2:]  # zero-phase window in fftbuffer
    fftbuffer[-hM2:] = xw[:hM2]
    Xh = rfft(fftbuffer, n=N)  # compute positive spectrum (real FFT)
    mX = _positive_spectrum_from_fft(Xh, hN)  # magnitude spectrum in dB
    pX = np.unwrap(np.angle(Xh))  # unwrapped phase spectrum of positive frequencies

    # -----synthesis-----
    Yh = _build_positive_spectrum(mX, pX)
    fftbuffer = irfft(Yh, n=N)  # compute inverse real FFT
    y[:hM2] = fftbuffer[-hM2:]  # undo zero-phase window
    y[hM2:] = fftbuffer[:hM1]
    return y


def dftAnal(x, w, N):
    """
    Analysis of a signal using the discrete Fourier transform
    x: input signal, w: analysis window, N: FFT size
    returns mX, pX: magnitude and phase spectrum

    The analysis window is internally normalized by sum(w), so the resulting
    spectra correspond to x * (w / sum(w)).
    """

    _validate_window_fft_size(w, N)

    hN = (N // 2) + 1  # size of positive spectrum, it includes sample 0
    hM1 = (w.size + 1) // 2  # half analysis window size by rounding
    hM2 = w.size // 2  # half analysis window size by floor
    fftbuffer = np.zeros(N)  # initialize buffer for FFT
    w = w / np.sum(w)  # normalize analysis window
    xw = x * w  # window the input sound
    fftbuffer[:hM1] = xw[hM2:]  # zero-phase window in fftbuffer
    fftbuffer[-hM2:] = xw[:hM2]
    Xh = rfft(fftbuffer, n=N)  # compute positive spectrum (real FFT)
    mX = _positive_spectrum_from_fft(Xh, hN)  # magnitude spectrum in dB

    Xh = Xh.copy()
    Xh.real[
        np.abs(Xh.real) < tol
    ] = 0.0  # for phase calculation set to 0 the small values
    Xh.imag[
        np.abs(Xh.imag) < tol
    ] = 0.0  # for phase calculation set to 0 the small values
    pX = np.unwrap(np.angle(Xh))  # unwrapped phase spectrum of positive frequencies
    return mX, pX


def dftSynth(mX, pX, M):
    """
    Synthesis of a signal using the discrete Fourier transform
    mX: magnitude spectrum, pX: phase spectrum, M: window size
    returns y: output signal

    If mX/pX come from dftAnal(), the output corresponds to the normalized
    windowed signal used there (x * (w / sum(w))).
    """

    hN = mX.size  # size of positive spectrum, it includes sample 0
    N = (hN - 1) * 2  # FFT size
    if not UF.isPower2(N):  # raise error if N not a power of two, thus mX is wrong
        raise ValueError("size of mX is not (N/2)+1")

    hM1 = (M + 1) // 2  # half analysis window size by rounding
    hM2 = M // 2  # half analysis window size by floor
    y = np.zeros(M)  # initialize output array
    Yh = _build_positive_spectrum(mX, pX)
    fftbuffer = irfft(Yh, n=N)  # compute inverse real FFT
    y[:hM2] = fftbuffer[-hM2:]  # undo zero-phase window
    y[hM2:] = fftbuffer[:hM1]
    return y
