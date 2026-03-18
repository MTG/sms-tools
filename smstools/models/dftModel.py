
# Discrete Fourier Transform (DFT) utilities used by sms-tools.
#
# This module provides:
# - `dftAnal`: analysis of a single windowed frame (magnitude/phase)
# - `dftSynth`: synthesis of a time-domain frame from magnitude/phase
# - `dftModel`: convenience analysis+synthesis round-trip for one frame


__all__ = [
    "dftAnal",
    "dftSynth",
    "dftModel",
]


import numpy as np
from scipy.fft import irfft, rfft

from smstools.models import utilFunctions as UF

tol: float = 1e-14  # threshold used to compute phase
_EPS: float = np.finfo(float).eps

def _validate_window_fft_size(w: np.ndarray, N: int) -> None:
    """
    Validate window and FFT size for DFT operations.

    Args:
        w: Analysis window array.
        N: FFT size.

    Raises:
        ValueError: If N is not a power of 2 or w is larger than N.
    """
    if not UF.isPower2(N):
        raise ValueError(
            f"FFT size (N={N}) is not a power of 2. Provided window size: {w.size}."
        )
    if w.size > N:
        raise ValueError(
            f"Window size (M={w.size}) is bigger than FFT size (N={N})."
        )

def _positive_spectrum_from_fft(X: np.ndarray, hN: int) -> np.ndarray:
    """
    Calculate magnitude spectrum in decibels from FFT output.

    Args:
        X: FFT output array.
        hN: Size of positive spectrum.

    Returns:
        Magnitude spectrum in dB.
    """
    absX = np.abs(X[:hN])
    np.maximum(absX, _EPS, out=absX)
    return 20 * np.log10(absX)

def _build_positive_spectrum(mX: np.ndarray, pX: np.ndarray) -> np.ndarray:
    """
    Build positive spectrum from magnitude and phase.

    Args:
        mX: Magnitude spectrum (dB).
        pX: Phase spectrum (radians).

    Returns:
        Complex positive spectrum.
    """
    pos_mag = 10 ** (mX / 20)
    return pos_mag * np.exp(1j * pX)

def dftAnal(x: np.ndarray, w: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Analysis of a signal using the discrete Fourier transform.

    Args:
        x: Input signal.
        w: Analysis window.
        N: FFT size.

    Returns:
        mX: Magnitude spectrum (dB).
        pX: Phase spectrum (radians).
    """
    _validate_window_fft_size(w, N)
    hN = (N // 2) + 1
    hM1 = (w.size + 1) // 2
    hM2 = w.size // 2
    fftbuffer = np.zeros(N)
    w = w / np.sum(w)
    xw = x * w
    fftbuffer[:hM1] = xw[hM2:]
    fftbuffer[-hM2:] = xw[:hM2]
    Xh = rfft(fftbuffer, n=N)
    mX = _positive_spectrum_from_fft(Xh, hN)
    Xh = Xh.copy()
    Xh.real[np.abs(Xh.real) < tol] = 0.0
    Xh.imag[np.abs(Xh.imag) < tol] = 0.0
    pX = np.unwrap(np.angle(Xh))
    return mX, pX

def dftSynth(mX: np.ndarray, pX: np.ndarray, M: int) -> np.ndarray:
    """
    Synthesis of a signal using the discrete Fourier transform.

    Args:
        mX: Magnitude spectrum (dB).
        pX: Phase spectrum (radians).
        M: Window size.

    Returns:
        y: Output signal (length M).
    """
    hN = mX.size
    N = (hN - 1) * 2
    if not UF.isPower2(N):
        raise ValueError(
            f"size of mX ({mX.size}) is not (N/2)+1 for N={N}. Check input spectrum size."
        )
    hM1 = (M + 1) // 2
    hM2 = M // 2
    y = np.zeros(M)
    Yh = _build_positive_spectrum(mX, pX)
    fftbuffer = irfft(Yh, n=N)
    y[:hM2] = fftbuffer[-hM2:]
    y[hM2:] = fftbuffer[:hM1]
    return y

def dftModel(x: np.ndarray, w: np.ndarray, N: int) -> np.ndarray:
    """
    Analysis/synthesis of a signal using the discrete Fourier transform.

    Args:
        x: Input signal.
        w: Analysis window.
        N: FFT size.

    Returns:
        y: Output signal (same shape as x).
    """
    # Analysis
    mX, pX = dftAnal(x, w, N)
    # Synthesis
    y = dftSynth(mX, pX, w.size)
    # Ensure output length matches input
    if len(y) > len(x):
        y = y[:len(x)]
    elif len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y